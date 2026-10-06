// Build with the trainer's flags and its cuDNN object. Run explicitly with
// LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD=0. No dataset or checkpoint I/O.
#define TESTING
#include "train_gpt2.cu"
#include <vector>
#include <cmath>
#include <cstdio>

namespace {
int failures=0;
void check(bool ok,const char* label){if(!ok){fprintf(stderr,"FAIL %s\n",label);++failures;}}
template<class T> struct Buffer {
    static constexpr size_t guard=128;
    T* allocation=nullptr;T* p=nullptr;size_t n;
    explicit Buffer(size_t count):n(count){cudaCheck(cudaMalloc((void**)&allocation,(n+2*guard)*sizeof(T)));p=allocation+guard;cudaCheck(cudaMemset(allocation,0xa5,(n+2*guard)*sizeof(T)));}
    ~Buffer(){check_guards();cudaFree(allocation);}
    void check_guards(){cudaCheck(cudaStreamSynchronize(main_stream));std::vector<unsigned char> bytes(guard*sizeof(T));for(T* edge:{allocation,p+n}){cudaCheck(cudaMemcpy(bytes.data(),edge,bytes.size(),cudaMemcpyDeviceToHost));check(std::all_of(bytes.begin(),bytes.end(),[](unsigned char x){return x==0xa5;}),"allocation edge canaries");}}
    void put(const std::vector<T>& v){cudaCheck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){cudaCheck(cudaStreamSynchronize(main_stream));std::vector<T> v(n);cudaCheck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
void test_relayout() {
    constexpr int tokens=19,C=37;
    Buffer<LlmcEosBits> source((size_t)tokens*3*C),planar(source.n),roundtrip(source.n);
    std::vector<LlmcEosBits> bits(source.n),expected(source.n);
    for(size_t i=0;i<bits.size();++i)bits[i]=(LlmcEosBits)(i*7919u+32768u);
    for(size_t i=0;i<bits.size();++i){const size_t token=i/(3*C),kind=(i/C)%3,channel=i%C;expected[kind*tokens*C+token*C+channel]=bits[i];}
    source.put(bits);
    llmc_eos_relayout_qkv_cuda((const floatX*)source.p,(floatX*)planar.p,tokens,C,true,main_stream);
    check(planar.get()==expected,"bitwise planar permutation including nonfinite payloads");
    llmc_eos_relayout_qkv_cuda((const floatX*)planar.p,(floatX*)roundtrip.p,tokens,C,false,main_stream);
    check(roundtrip.get()==bits&&source.get()==bits,"bitwise roundtrip and immutable source");
}
struct Oracle {std::vector<double> out,grad;};
Oracle oracle(const std::vector<double>& qkv,const std::vector<double>& dout,
              const std::vector<int>& tokens,int B,int T,int C,int H,int eos) {
    const int D=C/H;const double scale=1/std::sqrt((double)D);
    Oracle r{std::vector<double>((size_t)B*T*C),std::vector<double>(qkv.size())};
    for(int b=0;b<B;++b)for(int h=0;h<H;++h) {
        int start=0;
        for(int t=0;t<T;++t) {
            if(tokens[b*T+t]==eos)start=t;
            std::vector<double> p(t-start+1),dp(p.size());double mx=-1e300,sum=0;
            for(int k=start;k<=t;++k){double s=0;for(int d=0;d<D;++d)s+=qkv[((size_t)b*T+t)*3*C+h*D+d]*qkv[((size_t)b*T+k)*3*C+C+h*D+d];p[k-start]=s*scale;mx=std::max(mx,p[k-start]);}
            for(double& x:p){x=std::exp(x-mx);sum+=x;}for(double& x:p)x/=sum;
            for(int k=start;k<=t;++k)for(int d=0;d<D;++d) {
                const size_t oi=((size_t)b*T+t)*C+h*D+d,vi=((size_t)b*T+k)*3*C+2*C+h*D+d;
                r.out[oi]+=p[k-start]*qkv[vi];dp[k-start]+=dout[oi]*qkv[vi];r.grad[vi]+=p[k-start]*dout[oi];
            }
            double mean=0;for(size_t j=0;j<p.size();++j)mean+=p[j]*dp[j];
            for(int k=start;k<=t;++k){double ds=p[k-start]*(dp[k-start]-mean)*scale;for(int d=0;d<D;++d){size_t qi=((size_t)b*T+t)*3*C+h*D+d,ki=((size_t)b*T+k)*3*C+C+h*D+d;r.grad[qi]+=ds*qkv[ki];r.grad[ki]+=ds*qkv[qi];}}
        }
    }
    return r;
}
double objective(const Oracle& r,const std::vector<double>& cotangent){double sum=0;for(size_t i=0;i<r.out.size();++i)sum+=r.out[i]*cotangent[i];return sum;}
void compare(const std::vector<floatX>& actual,const std::vector<double>& expected,double atol,const char* name) {
    size_t bad=0;double max_error=0;
    for(size_t i=0;i<actual.size();++i){double err=std::abs((double)(float)actual[i]-expected[i]);max_error=std::max(max_error,err);if(!std::isfinite((float)actual[i])||err>atol+.015*std::abs(expected[i]))++bad;}
    printf("eos_oracle name=%s bad=%zu maximum_absolute_error=%.9g\n",name,bad,max_error);check(bad==0,name);
}
void run() {
    constexpr int B=2,T=96,C=128,H=2,D=64,EOS=127;
    auto* ctx=llmc_eos_attention_create(B,T,H,C,main_stream,llmc_eos_copy_stats_cuda,llmc_eos_relayout_qkv_cuda);
    LlmcRopeCache rope;llmc_rope_cache_reset(&rope);
    check(llmc_rope_cache_allocate(&rope,T,D,10000,0,main_stream),"RoPE allocate");
    Buffer<floatX> qkv((size_t)B*T*3*C),out((size_t)B*T*C),dqkv(qkv.n),dout(out.n),legacy(out.n);
    Buffer<float> stats((size_t)B*H*T),legacy_stats(stats.n);
    std::vector<floatX> seed(qkv.n),cotangent(out.n);
    for(size_t i=0;i<seed.size();++i)seed[i]=(floatX)(.13f*sinf((float)i*.071f));
    for(size_t i=0;i<cotangent.size();++i)cotangent[i]=(floatX)(.19f*cosf((float)i*.037f));
    std::vector<double> dc(cotangent.size());for(size_t i=0;i<dc.size();++i)dc[i]=(float)cotangent[i];
    std::vector<std::vector<int>> cases(4,std::vector<int>(B*T,1));
    cases[1][0]=EOS;cases[1][3]=EOS;cases[1][4]=EOS;cases[1][39]=EOS;cases[1][95]=EOS;
    cases[1][T+17]=EOS;cases[1][T+53]=EOS;
    std::fill(cases[2].begin(),cases[2].end(),EOS);
    cases[3][T-1]=EOS;cases[3][T]=EOS;cases[3][2*T-1]=EOS;
    std::vector<floatX> first_partition_output;
    for(int pass:{0,1,2,3,1}) {
        const auto& tokens=cases[pass];qkv.put(seed);dout.put(cotangent);
        llmc_eos_attention_prepare(ctx,tokens.data(),B,T,EOS,main_stream);
        check(llmc_rope_apply_qk_impl<false>(qkv.p,&rope,B,T,C,H,main_stream,llmc_eos_attention_positions(ctx)),"segment-local RoPE forward");
        const auto rotated=qkv.get();std::vector<double> qr(rotated.size());for(size_t i=0;i<qr.size();++i)qr[i]=(float)rotated[i];
        const auto plan=llmc_make_eos_plan(tokens.data(),B,T,C,EOS);
        // Independent local-coordinate phase oracle, including exact identity at EOS.
        std::vector<double> rotation_expected(seed.size());for(size_t i=0;i<seed.size();++i)rotation_expected[i]=(float)seed[i];
        for(int token=0;token<B*T;++token)for(int h=0;h<H;++h)for(int pair=0;pair<D/2;++pair){double a=plan.positions[token]*std::pow(10000.,-2.*pair/D),co=std::cos(a),si=std::sin(a);for(int kind=0;kind<2;++kind){size_t i=(size_t)token*3*C+kind*C+h*D+2*pair;double x=(float)seed[i],y=(float)seed[i+1];rotation_expected[i]=x*co-y*si;rotation_expected[i+1]=x*si+y*co;}}
        compare(rotated,rotation_expected,.001,"segment-local RoPE primal");
        llmc_eos_attention_forward(ctx,out.p,stats.p,qkv.p,main_stream);
        const auto actual=out.get();const auto expected=oracle(qr,dc,tokens,B,T,C,H,EOS);
        compare(actual,expected.out,.002,"segmented attention forward");
        if(pass==0){attention_forward_cudnn(legacy.p,legacy_stats.p,qkv.p,B,T,H,C,main_stream);const auto old=legacy.get();std::vector<double> oldd(old.size());for(size_t i=0;i<old.size();++i)oldd[i]=(float)old[i];compare(actual,oldd,.002,"no-EOS ordinary causal parity");}
        if(pass==1){if(first_partition_output.empty())first_partition_output=actual;else check(!memcmp(first_partition_output.data(),actual.data(),actual.size()*sizeof(floatX)),"changed-occupancy replay returns same forward");}
        const auto saved_stats=stats.get();
        llmc_eos_attention_backward(ctx,dqkv.p,dout.p,qkv.p,out.p,stats.p,main_stream);
        compare(dqkv.get(),expected.grad,.002,"segmented attention VJP");
        const auto after_qkv=qkv.get(),after_out=out.get(),after_dout=dout.get();
        const auto after_stats=stats.get();
        check(!memcmp(after_qkv.data(),rotated.data(),qkv.n*sizeof(floatX))&&
              !memcmp(after_out.data(),actual.data(),out.n*sizeof(floatX))&&
              !memcmp(after_dout.data(),cotangent.data(),dout.n*sizeof(floatX))&&
              !memcmp(after_stats.data(),saved_stats.data(),stats.n*sizeof(float)),"backward preserves immutable QKV/O/dO/stats");
        if(pass==1) {
            // Finite differences independently verify the dense oracle's VJP.
            for(size_t index:{(size_t)1,(size_t)C+7,(size_t)2*C+9,(size_t)39*3*C+13,(size_t)(T+53)*3*C+C+3}) {
                auto plus=qr,minus=qr;plus[index]+=1e-4;minus[index]-=1e-4;
                const double numeric=(objective(oracle(plus,dc,tokens,B,T,C,H,EOS),dc)-objective(oracle(minus,dc,tokens,B,T,C,H,EOS),dc))/(2e-4);
                check(std::abs(numeric-expected.grad[index])<1e-7,"independent finite-difference VJP");
            }
            // Loss on row1's final document has no Q/K/V adjoint before EOS53,
            // including other lanes. This is a graph-cut test, not cancellation.
            auto isolated=cotangent;for(int token=0;token<B*T;++token)if(token<T+53)for(int c=0;c<C;++c)isolated[(size_t)token*C+c]=(floatX)0;
            dout.put(isolated);llmc_eos_attention_backward(ctx,dqkv.p,dout.p,qkv.p,out.p,stats.p,main_stream);const auto cut=dqkv.get();
            bool zero=true;for(size_t i=0;i<(size_t)(T+53)*3*C;++i)if((float)cut[i]!=0)zero=false;check(zero,"exact zero cross-EOS gradients");
            auto perturbed=rotated;for(size_t i=0;i<(size_t)(T+53)*3*C;++i)perturbed[i]=(floatX)((float)perturbed[i]+.3f);
            qkv.put(perturbed);llmc_eos_attention_forward(ctx,out.p,stats.p,qkv.p,main_stream);const auto changed=out.get();
            check(!memcmp(changed.data()+(size_t)(T+53)*C,actual.data()+(size_t)(T+53)*C,(T-53)*C*sizeof(floatX)),"post-EOS perturbation invariance");
        }
        // RoPE transpose must use the SAME local position buffer as forward.
        std::vector<floatX> grad_seed(qkv.n);for(size_t i=0;i<grad_seed.size();++i)grad_seed[i]=(floatX)(.07f*cosf((float)i*.029f));dqkv.put(grad_seed);
        std::vector<double> inv(grad_seed.size());for(size_t i=0;i<inv.size();++i)inv[i]=(float)grad_seed[i];
        for(int token=0;token<B*T;++token)for(int h=0;h<H;++h)for(int pair=0;pair<D/2;++pair){double a=plan.positions[token]*std::pow(10000.,-2.*pair/D),co=std::cos(a),si=std::sin(a);for(int kind=0;kind<2;++kind){size_t i=(size_t)token*3*C+kind*C+h*D+2*pair;double x=(float)grad_seed[i],y=(float)grad_seed[i+1];inv[i]=x*co+y*si;inv[i+1]=-x*si+y*co;}}
        check(llmc_rope_apply_qk_impl<true>(dqkv.p,&rope,B,T,C,H,main_stream,llmc_eos_attention_positions(ctx)),"local RoPE backward call");compare(dqkv.get(),inv,.001,"segment-local RoPE transpose");
    }
    // Reuse the admitted context with fewer active rows: plane strides follow
    // this request, not the capacity. Unrequested gradient tail stays untouched.
    qkv.put(seed);dout.put(cotangent);
    std::vector<floatX> sentinel(qkv.n,(floatX).375f);dqkv.put(sentinel);
    llmc_eos_attention_prepare(ctx,cases[1].data(),1,T,EOS,main_stream);
    llmc_eos_attention_forward(ctx,out.p,stats.p,qkv.p,main_stream);
    llmc_eos_attention_backward(ctx,dqkv.p,dout.p,qkv.p,out.p,stats.p,main_stream);
    const auto fewer=dqkv.get();
    std::vector<double> short_q((size_t)T*3*C),short_do((size_t)T*C);
    for(size_t i=0;i<short_q.size();++i)short_q[i]=(float)seed[i];
    for(size_t i=0;i<short_do.size();++i)short_do[i]=(float)cotangent[i];
    const auto short_oracle=oracle(short_q,short_do,cases[1],1,T,C,H,EOS);
    compare(std::vector<floatX>(fewer.begin(),fewer.begin()+short_q.size()),short_oracle.grad,.002,"smaller active batch planar strides");
    check(!memcmp(fewer.data()+short_q.size(),sentinel.data()+short_q.size(),short_q.size()*sizeof(floatX)),"inactive gradient tail untouched");
    llmc_eos_attention_destroy(ctx);llmc_rope_cache_free(&rope);
}
void test_multi_group_tail() {
    // More singleton documents than the execution cap forces a full group and
    // a rounded tail. Every Q/K/V gradient is checked, including the last lane.
    constexpr int B=173,T=96,C=64,H=1,EOS=127;
    const size_t tokens=(size_t)B*T;
    std::vector<int> inputs(tokens,EOS);
    const auto host=llmc_make_eos_plan(inputs.data(),B,T,C,EOS);
    check(host.groups.size()==2&&host.groups[0].count==LLMC_EOS_GROUP_LIMIT&&
          host.groups[1].count>tokens-LLMC_EOS_GROUP_LIMIT,"full group plus rounded tail fixture");
    auto* ctx=llmc_eos_attention_create(B,T,H,C,main_stream,llmc_eos_copy_stats_cuda,llmc_eos_relayout_qkv_cuda);
    Buffer<floatX> qkv(tokens*3*C),out(tokens*C),dqkv(qkv.n),dout(out.n);
    Buffer<float> stats(tokens*H);
    std::vector<floatX> values(qkv.n),cotangent(out.n);
    for(size_t i=0;i<values.size();++i)values[i]=(floatX)(.01f*(1+(i%17)));
    for(size_t i=0;i<cotangent.size();++i)cotangent[i]=(floatX)(.02f*(1+(i%11)));
    qkv.put(values);dout.put(cotangent);
    llmc_eos_attention_prepare(ctx,inputs.data(),B,T,EOS,main_stream);
    llmc_eos_attention_forward(ctx,out.p,stats.p,qkv.p,main_stream);
    llmc_eos_attention_backward(ctx,dqkv.p,dout.p,qkv.p,out.p,stats.p,main_stream);
    const auto actual=out.get(),gradient=dqkv.get();
    bool covered=true;
    for(size_t token=0;token<tokens;++token)for(int c=0;c<C;++c){
        const size_t packed=token*3*C+c,dense=token*C+c;
        covered=covered&&(float)actual[dense]==(float)values[packed+2*C]&&
            std::abs((float)gradient[packed])<1e-6f&&std::abs((float)gradient[packed+C])<1e-6f&&
            (float)gradient[packed+2*C]==(float)cotangent[dense];
    }
    check(covered,"all-group full/tail output and gradient coverage");
    llmc_eos_attention_destroy(ctx);
}
}
int main() {
    multi_gpu_config=multi_gpu_config_init(1,0,1,nullptr,nullptr,nullptr);common_start(false);
    test_relayout();run();test_multi_group_tail();GPT2 empty={};common_free(empty);multi_gpu_config_free(&multi_gpu_config);
    printf("llmc_eos_attention_cuda: result=%s\n",failures?"fail":"pass");return failures?1:0;
}
