// External qualification: resolve train_gpt2.cu through -I to the frozen source.
// Build with the exact BF16/cuDNN trainer flags and its existing cudnn_att.o.
// This tiny diagnostic uses production b32 and real checkpointed replay; it is
// not a throughput or full-shape training qualification.
#define TESTING
#include "train_gpt2.cu"
#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <vector>

static_assert(PRECISION_MODE == PRECISION_BF16, "FFN qualification requires the production BF16 build");
namespace {
int failures=0,assertions=0;
constexpr int B=8,T=64,C=192,L=2,V=128;
void check(bool ok,const char* label) {
    ++assertions;
    if(!ok) { if(failures<20) fprintf(stderr,"FAIL %s\n",label); ++failures; }
}
void near(double got,double expected,double absolute,double relative,const char* label) {
    check(std::isfinite(got)&&std::isfinite(expected)&&
        std::abs(got-expected)<=absolute+relative*std::abs(expected),label);
}
float packed(double value) { return float(floatX(float(value))); }
void near_pack(float got,double expected,const char* label) {
    // Allow one BF16 ULP for a FP32 transcendental/reduction rounding seam.
    // The oracle's inputs are the actual stored BF16 weights and FP32 values.
    near(got,packed(expected),2e-6,.008,label);
}
template<class Q> struct Buffer {
    Q* p=nullptr; size_t n;
    explicit Buffer(size_t count):n(count){cudaCheck(cudaMalloc(&p,n*sizeof(Q)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<Q>& v) {
        assert(v.size()==n); cudaCheck(cudaMemcpy(p,v.data(),n*sizeof(Q),cudaMemcpyHostToDevice));
    }
    std::vector<Q> get() {
        cudaCheck(cudaStreamSynchronize(main_stream)); std::vector<Q> v(n);
        cudaCheck(cudaMemcpy(v.data(),p,n*sizeof(Q),cudaMemcpyDeviceToHost)); return v;
    }
};
template<class Q> std::vector<Q> read_device(const Q* p,size_t n) {
    cudaCheck(cudaStreamSynchronize(main_stream));std::vector<Q> out(n);
    cudaCheck(cudaMemcpy(out.data(),p,n*sizeof(Q),cudaMemcpyDeviceToHost));return out;
}
template<class Q> std::vector<double> widen(const std::vector<Q>& v) {
    std::vector<double> out(v.size());
    for(size_t i=0;i<v.size();++i)out[i]=float(v[i]);return out;
}
template<class Q> bool equal_bytes(const std::vector<Q>& a,const std::vector<Q>& b) {
    return a.size()==b.size()&&!memcmp(a.data(),b.data(),a.size()*sizeof(Q));
}

// Independent FP64 equations, with no call to any production activation/VJP.
struct GumOracle {
    std::vector<double> z,dh,local,du,dg,dv;
};
GumOracle gum_oracle(const std::vector<double>& h,const std::vector<double>& u,
        const std::vector<double>& g,const std::vector<double>& v,
        const std::vector<double>& dz,double scale) {
    constexpr int b=32;const size_t n=h.size();
    GumOracle q={std::vector<double>(n),std::vector<double>(n),std::vector<double>(3*n),
        std::vector<double>(b*b),std::vector<double>(b*b),std::vector<double>(b*b)};
    for(size_t base=0;base<n;base+=b) {
        for(int o=0;o<b;++o) {
            double ug=0,gg=0,vg=0;
            for(int k=0;k<b;++k) {const int w=o*b+k;
                ug+=u[w]*h[base+k];gg+=g[w]*h[base+k];vg+=v[w]*h[base+k];}
            const double s=1/(1+std::exp(-gg)),a=ug+scale*s*vg;
            q.z[base+o]=a>0?a*a:0;
            const double da=a>0?2*a*dz[base+o]:0;
            q.local[base+o]=da;q.local[n+base+o]=da*scale*vg*s*(1-s);
            q.local[2*n+base+o]=da*scale*s;
            for(int k=0;k<b;++k) {const int w=o*b+k;
                q.du[w]+=da*h[base+k];q.dg[w]+=q.local[n+base+o]*h[base+k];
                q.dv[w]+=q.local[2*n+base+o]*h[base+k];}
        }
        for(int k=0;k<b;++k)for(int o=0;o<b;++o) {const int w=o*b+k;
            q.dh[base+k]+=u[w]*q.local[base+o]+g[w]*q.local[n+base+o]+v[w]*q.local[2*n+base+o];}
    }
    return q;
}
double gum_loss(const std::vector<double>& h,const std::vector<double>& u,
        const std::vector<double>& g,const std::vector<double>& v,
        const std::vector<double>& dz,double scale) {
    constexpr int b=32;double loss=0;
    for(size_t base=0;base<h.size();base+=b)for(int o=0;o<b;++o) {
        double ug=0,gg=0,vg=0;
        for(int k=0;k<b;++k) {const int w=o*b+k;
            ug+=u[w]*h[base+k];gg+=g[w]*h[base+k];vg+=v[w]*h[base+k];}
        const double a=ug+scale*vg/(1+std::exp(-gg));
        if(a>0)loss+=a*a*dz[base+o];
    }
    return loss;
}
void isolated_gum(float scale) {
    constexpr int rows=2,H=128,b=32,n=rows*H;
    std::vector<float> h(n),dz(n);
    std::vector<floatX> bias(H),u(b*b),g(b*b),v(b*b);
    for(int i=0;i<n;++i){h[i]=float(i%19-9)/16;dz[i]=float(i%13-6)/32;}
    for(int i=0;i<H;++i)bias[i]=floatX(float(i%7-3)/128);
    for(int o=0;o<b;++o)for(int k=0;k<b;++k) {const int w=o*b+k;
        u[w]=floatX(float(w%11-5)/128+(o==k?.5f:0));
        g[w]=floatX(float(w%17-8)/128);v[w]=floatX(float(w%13-6)/128+(o==k?.25f:0));}
    Buffer<float> bh(n),bdz(n),local(3*n),partials(llmc_tiled_gum_backward_bank_scratch_floats(rows,H,b));
    Buffer<floatX> bb(H),bu(b*b),bg(b*b),bv(b*b),out(n),dh(n),du(b*b),dg(b*b),dv(b*b);
    bh.put(h);bdz.put(dz);bb.put(bias);bu.put(u);bg.put(g);bv.put(v);
    const std::vector<floatX> zeros(b*b,floatX(0.f));du.put(zeros);dg.put(zeros);dv.put(zeros);
    llmc_tiled_gum_forward(out.p,bh.p,bb.p,bu.p,bg.p,bv.p,rows,H,b,main_stream,scale);
    const auto actual_h=bh.get();const auto actual_z=out.get();
    auto rh=widen(h),ru=widen(u),rg=widen(g),rv=widen(v),rdz=widen(dz);
    for(int i=0;i<n;++i){rh[i]+=float(bias[i%H]);check(actual_h[i]==rh[i],"GUM adds outer bias once in FP32");}
    const auto oracle=gum_oracle(rh,ru,rg,rv,rdz,scale);
    for(int i=0;i<n;++i)near_pack(float(actual_z[i]),oracle.z[i],"GUM independent FP64 primal");
    // Replay starts from a fresh expansion, never from the already biased h.
    bh.put(h);llmc_tiled_gum_forward(out.p,bh.p,bb.p,bu.p,bg.p,bv.p,rows,H,b,main_stream,scale);
    check(equal_bytes(actual_z,out.get())&&actual_h==bh.get(),"GUM fresh-preactivation replay exact");
    llmc_tiled_gum_backward(dh.p,du.p,dg.p,dv.p,local.p,bdz.p,bh.p,bu.p,bg.p,bv.p,rows,H,b,
        partials.p,partials.n,main_stream,scale);
    const auto ah=dh.get(),au=du.get(),ag=dg.get(),av=dv.get();const auto al=local.get();
    for(int i=0;i<n;++i)near_pack(float(ah[i]),oracle.dh[i],"GUM independent FP64 input VJP");
    for(int i=0;i<3*n;++i)near(al[i],oracle.local[i],2e-6,3e-6,"GUM FP32 local adjoints before bank reduction");
    for(int i=0;i<b*b;++i) {
        near_pack(float(au[i]),oracle.du[i],"GUM U sum over every token/block use");
        near_pack(float(ag[i]),oracle.dg[i],"GUM G sum over every token/block use");
        near_pack(float(av[i]),oracle.dv[i],"GUM V sum over every token/block use");
    }
    // Full FP64 finite differences independently check each input/bank entry.
    auto finite=[&](std::vector<double>& parameter,const std::vector<floatX>& derivative) {
        constexpr double eps=1e-5;
        for(size_t i=0;i<parameter.size();++i) {
            const double old=parameter[i];parameter[i]=old+eps;
            const double plus=gum_loss(rh,ru,rg,rv,rdz,scale);parameter[i]=old-eps;
            const double minus=gum_loss(rh,ru,rg,rv,rdz,scale);parameter[i]=old;
            near_pack(float(derivative[i]),(plus-minus)/(2*eps),"GUM GPU derivative versus independent finite difference");
        }
    };
    finite(rh,ah);finite(ru,au);finite(rg,ag);finite(rv,av);
    const std::vector<floatX> initial_u(b*b,floatX(.125f)),initial_g(b*b,floatX(-.25f)),initial_v(b*b,floatX(.5f));
    du.put(initial_u);dg.put(initial_g);dv.put(initial_v);
    std::vector<double> expected_u(b*b,.125),expected_g(b*b,-.25),expected_v(b*b,.5);
    for(int micro=0;micro<2;++micro) {
        llmc_tiled_gum_backward(dh.p,du.p,dg.p,dv.p,local.p,bdz.p,bh.p,bu.p,bg.p,bv.p,rows,H,b,
            partials.p,partials.n,main_stream,scale);
        const auto xu=du.get(),xg=dg.get(),xv=dv.get();
        for(int i=0;i<b*b;++i) {
            expected_u[i]=packed(expected_u[i]+oracle.du[i]);expected_g[i]=packed(expected_g[i]+oracle.dg[i]);
            expected_v[i]=packed(expected_v[i]+oracle.dv[i]);
            near_pack(float(xu[i]),expected_u[i],"GUM U microbatch sum with one pack per use batch");
            near_pack(float(xg[i]),expected_g[i],"GUM G microbatch sum with one pack per use batch");
            near_pack(float(xv[i]),expected_v[i],"GUM V microbatch sum with one pack per use batch");
        }
    }
    printf("llmc_ffn_gum_oracle: block=32 scale=%.0f uses=%d fp64_primal=1 fp64_finite_difference=1 summed_banks=1 microbatch_sum=1 result=%s\n",
        scale,n/b,failures?"fail":"pass");
}
void partitioned_gum(float scale,int rows) {
    // H128 makes uses=4*rows. Cases cover exact/unequal partitions and both
    // the 32-use staged reduction and 256-use local-CTA tails. Positive
    // adjoints keep missing/duplicated tail contributions observable without
    // magnifying near-zero FP32 cancellation into a misleading BF16 seam.
    constexpr int H=128,b=32,guard=32;
    const size_t n=size_t(rows)*H,uses=n/b;
    const size_t expected_partitions=std::min<size_t>(1024,(uses+1023)/1024);
    const size_t capacity=llmc_tiled_gum_backward_bank_scratch_floats(rows,H,b);
    check(capacity==expected_partitions*3*b*b,"GUM explicit FP32 partition scratch capacity");
    std::vector<float> h(n),dz(n);
    std::vector<floatX> u(b*b),g(b*b),v(b*b);
    for(size_t i=0;i<n;++i){h[i]=float(1+i%19)/128;dz[i]=float(1+i%13)/64;}
    for(int o=0;o<b;++o)for(int k=0;k<b;++k){const int w=o*b+k;
        u[w]=floatX(float(w%11+1)/512+(o==k?.5f:0));
        g[w]=floatX(float(w%17-8)/512);v[w]=floatX(float(w%13+1)/512+(o==k?.25f:0));}
    const auto oracle=gum_oracle(widen(h),widen(u),widen(g),widen(v),widen(dz),scale);
    Buffer<float> bh(n),bdz(n),local(3*n),partials(capacity+2*guard);
    Buffer<floatX> bu(b*b),bg(b*b),bv(b*b),dh(n),du(b*b),dg(b*b),dv(b*b);
    bh.put(h);bdz.put(dz);bu.put(u);bg.put(g);bv.put(v);
    std::vector<float> workspace(capacity+2*guard,std::numeric_limits<float>::quiet_NaN());
    std::fill(workspace.begin(),workspace.begin()+guard,12345.25f);
    std::fill(workspace.end()-guard,workspace.end(),-23456.5f);
    const std::vector<floatX> iu(b*b,floatX(.125f)),ig(b*b,floatX(.25f)),iv(b*b,floatX(.5f));
    std::array<std::vector<floatX>,4> reference;
    auto run=[&](){
        llmc_tiled_gum_backward(dh.p,du.p,dg.p,dv.p,local.p,bdz.p,bh.p,bu.p,bg.p,bv.p,rows,H,b,
            partials.p+guard,capacity,main_stream,scale);
        const auto scratch=partials.get();
        check(std::all_of(scratch.begin(),scratch.begin()+guard,[](float x){return x==12345.25f;})&&
            std::all_of(scratch.end()-guard,scratch.end(),[](float x){return x==-23456.5f;}),
            "GUM partition scratch stays within exact capacity");
        check(std::all_of(scratch.begin()+guard,scratch.end()-guard,[](float x){return std::isfinite(x);}),
            "GUM overwrites every partition coordinate before final reduction");
        return std::array<std::vector<floatX>,4>{dh.get(),du.get(),dg.get(),dv.get()};
    };
    for(int repeat=0;repeat<3;++repeat){
        du.put(iu);dg.put(ig);dv.put(iv);partials.put(workspace);
        const auto actual=run();
        if(repeat==0)reference=actual;
        else for(size_t family=0;family<actual.size();++family)
            check(equal_bytes(reference[family],actual[family]),"GUM partitioned backward bitwise repeat determinism");
        for(size_t i=0;i<n;++i)near_pack(float(actual[0][i]),oracle.dh[i],"GUM batched input-tail FP64 VJP");
        for(int i=0;i<b*b;++i){
            near_pack(float(actual[1][i]),.125+oracle.du[i],"GUM U partition/tail FP64 sum plus existing gradient");
            near_pack(float(actual[2][i]),.25+oracle.dg[i],"GUM G partition/tail FP64 sum plus existing gradient");
            near_pack(float(actual[3][i]),.5+oracle.dv[i],"GUM V partition/tail FP64 sum plus existing gradient");
        }
    }
    // Consecutive microbatches reuse the same workspace and add to the packed
    // previous gradient; the workspace is not an optimizer-state accumulator.
    const auto accumulated=run();
    for(int i=0;i<b*b;++i){
        near_pack(float(accumulated[1][i]),float(reference[1][i])+oracle.du[i],"GUM U partitioned microbatch accumulation");
        near_pack(float(accumulated[2][i]),float(reference[2][i])+oracle.dg[i],"GUM G partitioned microbatch accumulation");
        near_pack(float(accumulated[3][i]),float(reference[3][i])+oracle.dv[i],"GUM V partitioned microbatch accumulation");
    }
    printf("llmc_ffn_gum_partitions: scale=%.0f uses=%zu partitions=%zu guarded_scratch=1 fp64_sum=1 deterministic=1 accumulation=1 result=%s\n",
        scale,uses,expected_partitions,failures?"fail":"pass");
}
__global__ void separable_bank_inputs(float* h,float* local,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const size_t use=i/32;
    // Binary fractions and an eight-use period give an exact FP32/FP64 sum.
    // k/o are independent coordinates even though this generator fills the
    // two arrays at the same channel index.
    h[i]=float(i%32+1)/32;
    const float value=float(i%32%7+1)/8*(use%8<4?1.f:2.f);
    local[i]=value;local[count+i]=-2*value;local[2*count+i]=.5f*value;
}
void bank_partition_cap() {
    constexpr int rows=262144,H=128,b=32,guard=32;
    constexpr size_t n=size_t(rows)*H,uses=n/b;
    constexpr size_t capacity=1024U*3U*b*b;
    check(llmc_tiled_gum_backward_bank_scratch_floats(rows,H,b)==capacity,
        "GUM production partition cap is 1024 with explicit 12MiB scratch");
    check(llmc_tiled_gum_backward_bank_scratch_floats(256*1024,3072,b)==capacity,
        "GUM actual production B256T1024H3072 has same bounded scratch capacity");
    Buffer<float> h(n),local(3*n),partials(capacity+2*guard);
    Buffer<floatX> du(b*b),dg(b*b),dv(b*b);
    separable_bank_inputs<<<CEIL_DIV(n,256),256,0,main_stream>>>(h.p,local.p,n);
    cudaCheck(cudaGetLastError());
    const std::vector<floatX> iu(b*b,floatX(.125f)),ig(b*b,floatX(.25f)),iv(b*b,floatX(.5f));
    std::vector<float> workspace(capacity+2*guard,std::numeric_limits<float>::quiet_NaN());
    std::fill(workspace.begin(),workspace.begin()+guard,12345.25f);
    std::fill(workspace.end()-guard,workspace.end(),-23456.5f);
    std::array<std::vector<floatX>,3> reference;
    for(int repeat=0;repeat<3;++repeat){
        du.put(iu);dg.put(ig);dv.put(iv);partials.put(workspace);
        llmc_tiled_gum_backward_bank(du.p,dg.p,dv.p,local.p,h.p,rows,H,b,
            partials.p+guard,capacity,main_stream);
        const auto scratch=partials.get();
        check(std::all_of(scratch.begin(),scratch.begin()+guard,[](float x){return x==12345.25f;})&&
            std::all_of(scratch.end()-guard,scratch.end(),[](float x){return x==-23456.5f;}),
            "GUM 1024-partition scratch guards remain intact");
        check(std::all_of(scratch.begin()+guard,scratch.end()-guard,[](float x){return std::isfinite(x);}),
            "GUM 1024-partition scratch fully overwritten");
        const auto actual=std::array<std::vector<floatX>,3>{du.get(),dg.get(),dv.get()};
        for(int o=0;o<b;++o)for(int k=0;k<b;++k){const int w=o*b+k;
            const double sum=double(uses)*1.5*double(k+1)/32*double(o%7+1)/8;
            check(float(actual[0][w])==packed(.125+sum),"GUM capped U exact analytical sum");
            check(float(actual[1][w])==packed(.25-2*sum),"GUM capped G exact analytical sum");
            check(float(actual[2][w])==packed(.5+.5*sum),"GUM capped V exact analytical sum");
        }
        if(repeat==0)reference=actual;
        else for(size_t family=0;family<actual.size();++family)
            check(equal_bytes(reference[family],actual[family]),"GUM capped reduction bitwise deterministic");
    }
    printf("llmc_ffn_gum_partition_cap: uses=%zu partitions=1024 scratch_bytes=%zu analytic_sum=1 deterministic=1 guarded_scratch=1 result=%s\n",
        uses,capacity*sizeof(float),failures?"fail":"pass");
}
void isolated_swiglu() {
    constexpr int rows=3,H=128,n=rows*H;
    std::vector<float> pre(2*n),dz(n);std::vector<floatX> bias(2*H);
    for(int i=0;i<2*n;++i)pre[i]=float(i%41-20)/16;
    for(int i=0;i<n;++i)dz[i]=float(i%13-6)/32;
    for(int i=0;i<2*H;++i)bias[i]=floatX(float(i%7-3)/128);
    Buffer<float> bp(2*n),bdz(n);Buffer<floatX> bb(2*H),out(n),dpre(2*n);
    bp.put(pre);bdz.put(dz);bb.put(bias);
    llmc_swiglu_forward(out.p,bp.p,bb.p,rows,H,main_stream);
    const auto actual_pre=bp.get();const auto actual_z=out.get();
    llmc_swiglu_backward(dpre.p,bdz.p,bp.p,rows,H,main_stream);const auto derivative=dpre.get();
    auto primal=[](double gate,double value){return gate*value/(1+std::exp(-gate));};
    constexpr double eps=1e-5;
    for(int r=0;r<rows;++r)for(int j=0;j<H;++j) {
        const int p=r*2*H+j,i=r*H+j;
        const double gate=double(pre[p])+float(bias[j]),value=double(pre[p+H])+float(bias[j+H]);
        check(actual_pre[p]==gate&&actual_pre[p+H]==value,"SwiGLU packed gate/value bias once in FP32");
        near_pack(float(actual_z[i]),primal(gate,value),"SwiGLU independent FP64 primal");
        near_pack(float(derivative[p]),dz[i]*(primal(gate+eps,value)-primal(gate-eps,value))/(2*eps),"SwiGLU GPU gate finite difference");
        near_pack(float(derivative[p+H]),dz[i]*(primal(gate,value+eps)-primal(gate,value-eps))/(2*eps),"SwiGLU GPU value finite difference");
    }
    bp.put(pre);llmc_swiglu_forward(out.p,bp.p,bb.p,rows,H,main_stream);
    check(equal_bytes(actual_z,out.get())&&actual_pre==bp.get(),"SwiGLU fresh-preactivation replay exact");
    printf("llmc_ffn_swiglu_oracle: packed_gate_value=1 fp64_primal=1 finite_difference=1 replay=1 result=%s\n",failures?"fail":"pass");
}

void optimizer_config(GPT2* m) {
    llmc_normuon_config_defaults(&m->optimizer_config);
    auto& cfg=m->optimizer_config;
    cfg.optimizer_selection=LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    cfg.orthogonalization_mode=LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
    cfg.execution_mode=LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    cfg.retraction_mode=LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    cfg.learning_rate=5e-4f;cfg.weight_decay=.001f;cfg.wdown_learning_rate_multiplier=1;
    cfg.targeted_family_mask=llmc_normuon_required_target_mask();llmc_normuon_resolve_schedules(&cfg);
}
void production_initializer_identity() {
    // Init-only production Small shapes: one ~248MB device parameter payload
    // at a time, with no full-shape activations, optimizer state or data batch.
    std::array<std::vector<floatX>,16> reference;
    for(int mode:std::array<int,3>{LLMC_MLP_RELU_SQUARED,LLMC_MLP_SWIGLU,LLMC_MLP_TIED_GUM_RELU_SQUARED}) {
        GPT2 m;gpt2_init_common(&m);
        gpt_build_from_descriptor(&m,"gpt2:rope:d12:t1024",mode,
            mode==LLMC_MLP_SWIGLU?2048:3072,mode==LLMC_MLP_TIED_GUM_RELU_SQUARED?32:0,
            true,mode==LLMC_MLP_TIED_GUM_RELU_SQUARED?2.f:1.f);
        check(m.config.num_layers==12&&m.config.channels==768&&m.config.num_heads==12&&
            m.config.rope_rotary_dim==64&&m.config.rope_theta==10000,"production initializer Small/RoPE geometry");
        size_t offset=0;
        for(int id=0;id<16;++id) {
            const auto actual=read_device(static_cast<const floatX*>(m.params_memory)+offset,m.param_elements[id]);
            const bool common=id<=9||id>=13;
            if(mode==LLMC_MLP_RELU_SQUARED)reference[id]=actual;
            else if(common||mode==LLMC_MLP_TIED_GUM_RELU_SQUARED)
                check(equal_bytes(reference[id],actual),common?
                    "all three production arms match cold common embeddings/LN/attention":
                    "production ReLU/GUM outer Wup/Wdown and biases match exactly");
            if(id==6)check(std::all_of(actual.begin(),actual.end(),[](floatX x){return float(x)==0;}),
                "production initializer all twelve attention Wo exactly zero");
            offset+=m.param_elements[id];
        }
        if(mode==LLMC_MLP_TIED_GUM_RELU_SQUARED) {
            check(m.config.ffn_gate_scale==2.f&&m.param_elements[18]==12U*32U*32U,
                "production initializer scale2 with independent b32 bank owners");
            const auto u=read_device(m.params.gum_u,m.param_elements[18]),
                g=read_device(m.params.gum_g,m.param_elements[19]),v=read_device(m.params.gum_v,m.param_elements[20]);
            check(!equal_bytes(u,g)&&!equal_bytes(u,v)&&!equal_bytes(g,v),"actual NAVIA families have unequal nonzero values");
            for(const auto* bank:{&u,&g,&v})for(int layer=0;layer<12;++layer) {
                check(std::any_of(bank->begin()+layer*1024,bank->begin()+(layer+1)*1024,
                    [](floatX x){return float(x)!=0;}),"each production NAVIA bank nonzero");
                if(layer)check(memcmp(bank->data(),bank->data()+layer*1024,1024*sizeof(floatX))!=0,
                    "production NAVIA layers have independent banks");
            }
        }
        cudaFreeCheck(&m.params_memory);
    }
    printf("llmc_ffn_initializer: target_shape=1 seed=42 modes=3,5,6 common_tensors_match=1 relu_gum_outer_match=1 all_attention_wo_zero=1 navia_nonidentity=1 result=%s\n",failures?"fail":"pass");
}
void tiny_model(GPT2* m,int mode,float scale,int recompute) {
    gpt2_init_common(m);m->config={};
    auto& c=m->config;c.max_seq_len=T;c.vocab_size=V;c.padded_vocab_size=V;
    c.num_layers=L;c.num_heads=3;c.channels=C;c.lexical_channels=C;
    c.position_encoding=LLMC_POSITION_ENCODING_ROPE;c.rope_rotary_dim=64;c.rope_theta=10000;
    c.ffn_mode=mode;c.ffn_hidden=mode==LLMC_MLP_SWIGLU?512:768;
    c.ffn_block=mode==LLMC_MLP_SWIGLU?0:32;c.ffn_gate_scale=scale;
    gpt2_set_initializer_defaults(&c);m->mlp_activation=mode;m->gelu_fusion=0;m->recompute=recompute;
    optimizer_config(m);gpt2_allocate_weights(m);gpt2_initialize_weights(m,true);
    set_zero_configs(&multi_gpu_config,0,m->num_parameters);gpt2_allocate_state(m,B,T);
    check(m->optimizer_plan.normuon_parameter_type_count==2&&m->optimizer_plan.normuon_view_count==2*L,
        "only unique outer Wup/Wdown assigned NorMuon");
    for(int id=0;id<NUM_PARAMETER_TENSORS;++id) {
        const auto& type=m->optimizer_plan.parameter_types[id];
        check(type.tensor_elements==m->param_elements[id],"optimizer plan covers full parameter inventory");
        check(type.backend_kind==((id==10||id==12)?LLMC_OPTIMIZER_BACKEND_NORMUON:LLMC_OPTIMIZER_BACKEND_ADAMW),
            "all other parameters including independent U/G/V use AdamW");
    }
    const auto wo=read_device(m->params.attprojw,m->param_elements[6]);
    check(std::all_of(wo.begin(),wo.end(),[](floatX x){return float(x)==0;}),"actual cold initializer zeros every attention Wo");
}
struct Run {float loss,norm;std::vector<floatX> gradients;};
Run backward(GPT2* m,int* inputs,const int* targets) {
    gpt2_forward(m,inputs,B,T);gpt2_backward_and_reduce(m,inputs,targets,1,0,true);
    Run r={m->mean_loss,gpt2_calculate_grad_norm(m,&multi_gpu_config),
        read_device(static_cast<const floatX*>(m->grads_memory),m->num_parameters)};
    check(std::isfinite(r.loss)&&std::isfinite(r.norm)&&r.norm>0,"finite fullmodel checkpointed forward/backward");
    return r;
}
void update(GPT2* m,const Run& r,int step) {
    gpt2_update(m,5e-4f,.9f,.95f,1e-8f,.1f,r.norm>1?1/r.norm:1,step,&multi_gpu_config,5e-4f);
    const auto master=read_device(m->master_weights,m->num_parameters);
    check(std::all_of(master.begin(),master.end(),[](float x){return std::isfinite(x);}),"every updated master finite");
}
void write_tokens(const std::filesystem::path& path) {
    std::string h="{'descr': '<u4', 'fortran_order': False, 'shape': (32, 64), }";
    while((10+h.size()+1)%64)h+=' ';h+='\n';
    unsigned char prefix[]={0x93,'N','U','M','P','Y',1,0,
        static_cast<unsigned char>(h.size()&255),static_cast<unsigned char>(h.size()>>8)};
    FILE* f=fopenCheck(path.string().c_str(),"wb");fwriteCheck(prefix,1,10,f);fwriteCheck(h.data(),1,h.size(),f);
    for(uint32_t i=0;i<32*64;++i){uint32_t token=(i*7+3)%V;fwriteCheck(&token,sizeof(token),1,f);}fcloseCheck(f);
}
std::array<int,256> header(const std::filesystem::path& path) {
    std::array<int,256> h;FILE* f=fopenCheck(path.string().c_str(),"rb");
    freadCheck(h.data(),sizeof(int),h.size(),f);fcloseCheck(f);return h;
}
void updated_recompute_witness(const std::filesystem::path& model_path,
        const GPT2& owner,int* inputs,const int* targets) {
    // Real first-update weights have nonzero biases/Wo. All witnesses use the
    // same stored BF16 payload and RNG, with no additional optimizer updates.
    const auto wo=read_device(owner.params.attprojw,owner.param_elements[6]);
    const auto bias=read_device(owner.params.fcb,owner.param_elements[11]);
    check(std::any_of(wo.begin(),wo.end(),[](floatX x){return float(x)!=0;})&&
        std::any_of(bias.begin(),bias.end(),[](floatX x){return float(x)!=0;}),
        "post-update replay witness has nonzero Wo and expansion bias");
    Run reference={};
    for(int recompute:{0,1,2}) {
        GPT2 witness;gpt2_init_common(&witness);gpt2_build_from_checkpoint(&witness,model_path.string().c_str());
        witness.recompute=recompute;optimizer_config(&witness);
        witness.rng_state=owner.rng_state;witness.rng_state_last_update=owner.rng_state_last_update;
        set_zero_configs(&multi_gpu_config,0,witness.num_parameters);gpt2_allocate_state(&witness,B,T);
        const auto actual=backward(&witness,inputs,targets);
        if(recompute==0)reference=actual;
        else {
            near(actual.loss,reference.loss,1e-6,0,"post-update recompute loss parity");
            check(actual.gradients.size()==reference.gradients.size(),"post-update replay full gradient inventory");
            for(size_t i=0;i<actual.gradients.size();++i)
                near(float(actual.gradients[i]),float(reference.gradients[i]),1e-4,.03,
                    "post-update all-parameter recompute gradient parity");
        }
        gpt2_free(&witness);
    }
}
void fullmodel(int mode,float scale,const std::filesystem::path& output) {
    const std::string name=mode==LLMC_MLP_SWIGLU?"swiglu":"gum"+std::to_string(int(scale));
    const auto dir=output/name;std::filesystem::create_directory(dir);
    const auto tokens=dir/"tokens.npy";write_tokens(tokens);
    DataLoader loader={};dataloader_init_with_policy(&loader,tokens.string().c_str(),B,T,0,1,0,1,0);
    dataloader_next_batch(&loader);
    GPT2 baseline;tiny_model(&baseline,mode,scale,0);const auto reference=backward(&baseline,loader.inputs,loader.targets);
    const auto initial=read_device(static_cast<const floatX*>(baseline.params_memory),baseline.num_parameters);
    gpt2_free(&baseline);
    // Modes0/1 are bounded diagnostic witnesses; the update/save path uses2.
    for(int recompute:{1,2}) {
        GPT2 m;tiny_model(&m,mode,scale,recompute);
        check(equal_bytes(initial,read_device(static_cast<const floatX*>(m.params_memory),m.num_parameters)),"seed42 outer/NAVIA payload identical across replay modes");
        const auto actual=backward(&m,loader.inputs,loader.targets);
        near(actual.loss,reference.loss,1e-6,0,"fullmodel recompute loss parity");
        check(actual.gradients.size()==reference.gradients.size(),"recompute gradient inventory parity");
        for(size_t i=0;i<actual.gradients.size();++i)
            near(float(actual.gradients[i]),float(reference.gradients[i]),1e-4,.03,"fullmodel recompute all-parameter gradient parity");
        if(recompute==1){gpt2_free(&m);continue;}
        update(&m,actual,1);
        const auto master=read_device(m.master_weights,m.num_parameters);
        const auto first=read_device(m.m_memory,m.num_parameters),second=read_device(m.v_memory,m.num_parameters);
        for(int id:{10,12,18,19,20}) {
            if(!m.param_elements[id])continue;
            const auto tensor=gpt2_get_tensor_at_layer(&m,0,id);
            for(int layer=0;layer<L;++layer) {
                size_t changed=0,nonzero_gradient=0;
                for(size_t j=0;j<tensor.size;++j) {
                    const size_t i=tensor.offset+layer*tensor.size+j;
                    changed+=master[i]!=float(initial[i]);nonzero_gradient+=float(actual.gradients[i])!=0;
                    if(id>=18) {
                        const double grad=float(actual.gradients[i])*(actual.norm>1?1/actual.norm:1);
                        near(first[i],(1-double(.9f))*grad,2e-10,3e-6,"independent bank AdamW first moment");
                        near(second[i],(1-double(.95f))*grad*grad,2e-12,3e-6,"independent bank AdamW second moment");
                        const double expected=float(initial[i])-double(5e-4f)*(grad/(std::abs(grad)+1e-8)+double(.1f)*float(initial[i]));
                        near(master[i],expected,2e-7,2e-6,"each shared bank receives one AdamW update after summation");
                    }
                }
                check(changed>0&&nonzero_gradient>0,"each independent layer outer/bank owner updates");
            }
        }
        const auto model_path=dir/"model.bin",state_path=dir/"state.bin",companion_path=dir/"normuon.bin";
        gpt2_write_to_checkpoint(&m,model_path.string().c_str());
        save_state(state_path.string().c_str(),1,&m,&loader,LLMC_SEQUENCE_BOUNDARY_ROW_RESET);
        check(llmc_normuon_save_companion(companion_path.string().c_str(),1,1,0,
            &m.optimizer_plan,&m.optimizer_config,&m.normuon_runtime,main_stream),"real rectangular NorMuon companion saves");
        const auto mh=header(model_path),sh=header(state_path);
        check(mh[1]==LLMC_MODEL_VERSION_BF16_FFN&&sh[1]==LLMC_OPTIMIZER_STATE_VERSION_FFN,
            "actual model19/state7 wrappers");
        check(llmc_ffn_contract_matches(mh.data(),mode,m.config.ffn_hidden,m.config.ffn_block,scale)&&
            llmc_ffn_contract_matches(sh.data(),mode,m.config.ffn_hidden,m.config.ffn_block,scale),"actual model/state own identical FFN scale/geometry");
        check(mh[81]==(scale==2?2:1)&&sh[81]==mh[81]&&mh[89]==(scale==2?0x40000000:0)&&sh[89]==mh[89],
            "actual schema1/schema2 scale encoding");
        updated_recompute_witness(model_path,m,loader.inputs,loader.targets);
        GPT2 restored;gpt2_init_common(&restored);gpt2_build_from_checkpoint(&restored,model_path.string().c_str());
        check(restored.mlp_activation==mode&&restored.config.ffn_gate_scale==scale&&
            restored.config.ffn_hidden==m.config.ffn_hidden&&restored.config.ffn_block==m.config.ffn_block,
            "reader automatically selects checkpoint-owned FFN mode/scale");
        check(equal_bytes(read_device(static_cast<const floatX*>(m.params_memory),m.num_parameters),
            read_device(static_cast<const floatX*>(restored.params_memory),restored.num_parameters)),"model payload roundtrip byte identical");
        restored.recompute=2;optimizer_config(&restored);set_zero_configs(&multi_gpu_config,0,restored.num_parameters);
        gpt2_allocate_state(&restored,B,T);
        DataLoader resumed={};dataloader_init_with_policy(&resumed,tokens.string().c_str(),B,T,0,1,0,1,0);
        int restored_step=-1;load_state(&restored_step,&restored,&resumed,state_path.string().c_str(),LLMC_SEQUENCE_BOUNDARY_ROW_RESET);
        check(llmc_normuon_load_companion(companion_path.string().c_str(),1,1,0,
            &restored.optimizer_plan,&restored.optimizer_config,&restored.normuon_runtime,main_stream),"real rectangular NorMuon companion restores");
        check(restored_step==1&&resumed.current_sample_idx==loader.current_sample_idx&&
            restored.rng_state==m.rng_state&&restored.rng_state_last_update==m.rng_state_last_update,
            "state restores step/cursor and ordinary/local rounding RNG");
        check(equal_bytes(read_device(static_cast<const floatX*>(m.params_memory),m.num_parameters),
            read_device(static_cast<const floatX*>(restored.params_memory),m.num_parameters)),"master-to-BF16 stochastic restore exact");
        check(first==read_device(restored.m_memory,m.num_parameters)&&second==read_device(restored.v_memory,m.num_parameters)&&
            master==read_device(restored.master_weights,m.num_parameters),"all optimizer/master payloads restore exactly");
        dataloader_next_batch(&loader);dataloader_next_batch(&resumed);
        check(!memcmp(loader.inputs,resumed.inputs,B*T*sizeof(int))&&!memcmp(loader.targets,resumed.targets,B*T*sizeof(int)),"resume consumes identical next row-reset batch");
        const auto continuous=backward(&m,loader.inputs,loader.targets),continued=backward(&restored,resumed.inputs,resumed.targets);
        near(continued.loss,continuous.loss,1e-6,0,"restored next loss matches uninterrupted loss");
        update(&m,continuous,2);update(&restored,continued,2);
        const auto a=read_device(m.master_weights,m.num_parameters),b=read_device(restored.master_weights,m.num_parameters);
        const auto am=read_device(m.m_memory,m.num_parameters),bm=read_device(restored.m_memory,m.num_parameters);
        const auto av=read_device(m.v_memory,m.num_parameters),bv=read_device(restored.v_memory,m.num_parameters);
        for(size_t i=0;i<a.size();++i) {
            near(b[i],a[i],1e-6,1e-4,"restored second optimizer update master parity");
            near(bm[i],am[i],1e-7,1e-4,"restored second optimizer update momentum parity");
            near(bv[i],av[i],1e-8,1e-4,"restored second optimizer update second moment parity");
        }
        check(restored.rng_state==m.rng_state&&restored.rng_state_last_update==m.rng_state_last_update,"continued update RNG parity");
        dataloader_free(&resumed);gpt2_free(&restored);gpt2_free(&m);
    }
    dataloader_free(&loader);
    printf("llmc_ffn_model: mode=%d gate_scale=%.0f block=%d recompute_modes=0,1,2 outer_normuon=1 bank_adamw=1 model_state_roundtrip=1 resumed_update=1 result=%s\n",
        mode,scale,mode==LLMC_MLP_SWIGLU?0:32,failures?"fail":"pass");
}
}
int main(int argc,char** argv) {
    if(argc!=3||strcmp(argv[1],"--output-dir")) {
        fprintf(stderr,"usage: test_ffn_interventions --output-dir NEW_DIRECTORY\n");return 2;
    }
    const std::filesystem::path output=argv[2];
    if(std::filesystem::exists(output)||!std::filesystem::create_directories(output)) {
        fprintf(stderr,"Fresh output directory required\n");return 2;
    }
    multi_gpu_config=multi_gpu_config_init(1,0,1,nullptr,nullptr,nullptr);common_start(true);
    isolated_swiglu();isolated_gum(1.f);isolated_gum(2.f);
    for(float scale:std::array<float,2>{1.f,2.f})for(int rows:std::array<int,5>{1,256,257,258,1025})
        partitioned_gum(scale,rows);
    bank_partition_cap();production_initializer_identity();
    fullmodel(LLMC_MLP_SWIGLU,1.f,output);fullmodel(LLMC_MLP_TIED_GUM_RELU_SQUARED,1.f,output);
    fullmodel(LLMC_MLP_TIED_GUM_RELU_SQUARED,2.f,output);
    cudaCheck(cudaDeviceSynchronize());
    GPT2 empty={};common_free(empty);multi_gpu_config_free(&multi_gpu_config);
    const auto receipt=output/"ffn_cuda_receipt.json";FILE* f=fopenCheck(receipt.string().c_str(),"w");
    fprintf(f,"{\n  \"schema\":\"llmc.worldmodel.ffn_cuda_qualification.v1\",\n  \"status\":\"%s\",\n  \"failures\":%d,\n  \"assertions\":%d,\n  \"precision\":\"bf16_storage_fp32_math\",\n  \"block_width\":32,\n  \"modes\":[5,6],\n  \"gum_gate_scales\":[1,2],\n  \"tiny_model\":{\"batch\":8,\"sequence\":64,\"channels\":192,\"layers\":2},\n  \"checks\":{\"independent_fp64_oracle\":%s,\"finite_differences\":%s,\"summed_bank_gradients\":%s,\"microbatch_accumulation\":%s,\"recompute_parity\":%s,\"optimizer_routing_update\":%s,\"checkpoint_roundtrip\":%s,\"resumed_update\":%s,\"production_initializer_identity\":%s,\"partition_tails\":%s,\"partition_scratch_bounds\":%s,\"partition_cap_analytic\":%s,\"partition_determinism\":%s},\n  \"bank_partition_checks\":{\"full_backward_uses\":[4,1024,1028,1032,4100],\"gate_scales\":[1,2],\"cap_reduction_uses\":1048576,\"maximum_partitions\":1024,\"scratch_bytes\":12582912,\"capacity_guards\":true,\"scratch_prefilled_nan\":true,\"repeat_count\":3,\"timing_measurement\":false},\n  \"full_shape_memory_qualified\":false,\n  \"throughput_qualified\":false\n}\n",
        failures?"failed":"passed",failures,assertions,failures?"false":"true",failures?"false":"true",failures?"false":"true",
        failures?"false":"true",failures?"false":"true",failures?"false":"true",failures?"false":"true",failures?"false":"true",failures?"false":"true",
        failures?"false":"true",failures?"false":"true",failures?"false":"true",failures?"false":"true");
    fcloseCheck(f);
    printf("llmc_ffn_interventions_cuda: assertions=%d failures=%d result=%s\n",assertions,failures,failures?"fail":"pass");
    return failures?1:0;
}
