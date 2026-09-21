// Build with the trainer's exact flags, include root, and cuDNN link object.
// GPU work is explicit: pass one NEW --output-dir for disposable roundtrip data.
#define TESTING
#include "train_gpt2.cu"
#include <algorithm>
#include <filesystem>
#include <limits>

namespace {
int failures=0;
int test_policy=LLMC_MLP_SWISH_POWER125_K8;
void check(bool ok,const char* what){if(!ok){fprintf(stderr,"FAIL %s\n",what);++failures;}}
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(size_t count):n(count){cudaCheck(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){cudaCheck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){cudaCheck(cudaStreamSynchronize(main_stream));std::vector<T> v(n);cudaCheck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
template<class T> std::vector<T> read_device(const T* p,size_t n) {
    cudaCheck(cudaStreamSynchronize(main_stream));std::vector<T> result(n);
    cudaCheck(cudaMemcpy(result.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return result;
}
void write_tokens(const std::filesystem::path& path) {
    // A real four-batch fixed-row NumPy cache exercises the exact-resume cursor.
    std::string h="{'descr': '<u4', 'fortran_order': False, 'shape': (32, 64), }";
    while((10+h.size()+1)%64)h+=' ';h+='\n';
    unsigned char prefix[]={0x93,'N','U','M','P','Y',1,0,
        static_cast<unsigned char>(h.size()&255),static_cast<unsigned char>(h.size()>>8)};
    FILE* f=fopenCheck(path.string().c_str(),"wb");fwriteCheck(prefix,1,10,f);fwriteCheck(h.data(),1,h.size(),f);
    for(uint32_t i=0;i<32*64;++i){uint32_t token=(i*7+3)%128;fwriteCheck(&token,sizeof(token),1,f);}fcloseCheck(f);
}
void mutate_header(const std::filesystem::path& source,const std::filesystem::path& destination,int word) {
    std::filesystem::copy_file(source,destination);
    FILE* f=fopenCheck(destination.string().c_str(),"r+b");int h[256];freadCheck(h,sizeof(int),256,f);
    h[word]^=1;fseekCheck(f,0,SEEK_SET);fwriteCheck(h,sizeof(int),256,f);fcloseCheck(f);
}
void change_state_activation(const std::filesystem::path& source,const std::filesystem::path& destination,int policy) {
    std::filesystem::copy_file(source,destination);
    FILE* f=fopenCheck(destination.string().c_str(),"r+b");int h[256];freadCheck(h,sizeof(int),256,f);
    llmc_store_mlp_contract(h,44,h[44],policy);
    fseekCheck(f,0,SEEK_SET);fwriteCheck(h,sizeof(int),256,f);fcloseCheck(f);
}
double reference(double x) {
    if(test_policy==LLMC_MLP_RELU_SQUARED)return x>0?x*x:0;
    const double e=std::exp(-std::abs(x)),s=x>=0?1/(1+e):e/(1+e),z=8*(x-1);
    if(test_policy==LLMC_MLP_SWISH)return x*s;
    if(test_policy==LLMC_MLP_SWISH_POWER2_K8)return x*s*(1+(std::max(z,0.0)+std::log1p(std::exp(-std::abs(z))))/8);
    return x*s*std::pow(1+(std::max(z,0.0)+std::log1p(std::exp(-std::abs(z))))/8,.25);
}
double derivative_reference(double x) {
    if(test_policy==LLMC_MLP_RELU_SQUARED)return x>0?2*x:0;
    const double e=std::exp(-std::abs(x)),s=x>=0?1/(1+e):e/(1+e);
    const double ds=e/((1+e)*(1+e)),swish=x*s,dswish=s+x*ds;
    if(test_policy==LLMC_MLP_SWISH)return dswish;
    const double z=8*(x-1),ez=std::exp(-std::abs(z)),t=z>=0?1/(1+ez):ez/(1+ez);
    const double b=1+(std::max(z,0.0)+std::log1p(ez))/8;
    if(test_policy==LLMC_MLP_SWISH_POWER2_K8)return b*dswish+swish*t;
    return std::pow(b,.25)*(dswish+.25*(swish/b)*t);
}
__global__ void edges(const float* x,float* y,float* dy,int n,int policy) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){y[i]=llmc_custom_mlp_activation(x[i],policy);dy[i]=llmc_custom_mlp_activation_derivative(x[i],policy);}
}
void test_edges() {
    std::vector<float> x={-std::numeric_limits<float>::max(),-100,-32,
        nextafterf(-15,-INFINITY),-15,nextafterf(-15,INFINITY),-6,-2,-.5f,-0.f,0,.5f,.875f,
        nextafterf(1,0),1,nextafterf(1,INFINITY),1.125f,2,6,
        nextafterf(17,0),17,nextafterf(17,INFINITY),32,1e10f,1e20f,1e30f,1e31f,
        std::numeric_limits<float>::max(),NAN,INFINITY,-INFINITY};
    if(test_policy==LLMC_MLP_SWISH_POWER2_K8) {
        const float seam=std::ldexp(1.0f,126),half_max=std::numeric_limits<float>::max()*.5f;
        for(float value:{1e18f,1e19f,2e19f,nextafterf(seam,0.f),seam,nextafterf(seam,INFINITY),
                         1e38f,half_max,nextafterf(half_max,INFINITY)})x.push_back(value);
    }
    Buffer<float> bx(x.size()),by(x.size()),bd(x.size());bx.put(x);
    edges<<<1,64,0,main_stream>>>(bx.p,by.p,bd.p,int(x.size()),test_policy);
    cudaCheck(cudaGetLastError());cudaCheck(cudaStreamSynchronize(main_stream));
    auto y=by.get(),d=bd.get();
    for(size_t i=0;i<x.size();++i) {
        if(!std::isfinite(x[i])) {check(std::isnan(y[i])&&std::isnan(d[i]),"edge nonfinite propagation");continue;}
        double v=reference(x[i]);
        if(v>std::numeric_limits<float>::max()) check(std::isinf(y[i])&&y[i]>0,"uncapped overflow");
        else check(std::isfinite(y[i])&&std::abs(y[i]-v)<=3e-6*std::abs(v)+std::numeric_limits<float>::min(),"edge primal");
        double dv=derivative_reference(x[i]);
        if(dv>std::numeric_limits<float>::max())check(std::isinf(d[i])&&d[i]>0,"uncapped derivative overflow");
        else check(std::isfinite(d[i])&&std::abs(d[i]-dv)<2e-6*(1+std::abs(dv)),"analytic derivative oracle");
        if(test_policy==LLMC_MLP_RELU_SQUARED&&x[i]<=0)check(y[i]==0&&d[i]==0,"ReLU nonpositive and kink exactly zero");
        if(std::abs(x[i])<=32) {
            double h=1e-5,expected=(reference(double(x[i])+h)-reference(double(x[i])-h))/(2*h);
            check(std::abs(d[i]-expected)<2e-5*(1+std::abs(expected)),"edge finite difference");
        }
    }
    printf("llmc_mlp_edges: transition_and_tails=1 finite_difference=1 nonfinite_and_overflow=1 result=%s\n",failures?"fail":"pass");
    if(test_policy==LLMC_MLP_SWISH_POWER2_K8)
        printf("llmc_mlp_power2_edges: analytic_forward=1 analytic_derivative=1 finite_difference=1 extended_positive_tails=1 nonfinite_propagation=1 result=%s\n",failures?"fail":"pass");
}
void test_gemm_boundary() {
    constexpr int R=16,C=64,H=256;
    std::vector<floatX> x(R*C),w(H*C),b(H),upstream(R*C),down(C*H);
    for(size_t i=0;i<x.size();++i)x[i]=(floatX)(float(int(i%31)-15)/32);
    for(size_t i=0;i<w.size();++i)w[i]=(floatX)(float(int(i%37)-18)/64);
    for(int i=0;i<H;++i)b[i]=(floatX)(float(i%11-5)/16);
    for(size_t i=0;i<upstream.size();++i)upstream[i]=(floatX)(float(int(i%17)-8)/32);
    for(size_t i=0;i<down.size();++i)down[i]=(floatX)(float(int(i%23)-11)/64);
    Buffer<floatX> bx(x.size()),bw(w.size()),bb(b.size()),bo(R*H),bdx(R*H),bu(upstream.size()),bdown(down.size());
    Buffer<float> pre(R*H),dh(R*H);
    bx.put(x);bw.put(w);bb.put(b);bu.put(upstream);bdown.put(down);
    llmc_mlp_activation_forward(bo.p,pre.p,bx.p,bw.p,bb.p,R,C,H,test_policy,main_stream);
    auto y=bo.get();auto p=pre.get();
    size_t precision_witnesses=0;
    for(int r=0;r<R;++r)for(int h=0;h<H;++h) {
        double expected=float(b[h]);
        for(int c=0;c<C;++c)expected+=double(float(x[r*C+c]))*float(w[h*C+c]);
        check(std::abs(p[r*H+h]-expected)<1e-6,"GEMM+bias preactivation remains FP32");
        float expected_out=float((floatX)float(reference(expected)));
        float rounded_out=float((floatX)float(reference(float((floatX)expected))));
        check(std::abs(float(y[r*H+h])-expected_out)<=0.008f*std::abs(expected_out)+1e-6f,"activation output oracle");
        if(expected_out!=rounded_out && float(y[r*H+h])==expected_out)++precision_witnesses;
    }
    check(precision_witnesses>0,"detectable avoided BF16 preactivation rounding");
    llmc_mlp_activation_forward(bo.p,pre.p,bx.p,bw.p,bb.p,R,C,H,test_policy,main_stream);
    auto replay=bo.get();auto replay_pre=pre.get();
    check(!memcmp(y.data(),replay.data(),y.size()*sizeof(floatX))&&p==replay_pre,"full GEMM replay exact");
    llmc_mlp_gemm_fp32(dh.p,bdown.p,bu.p,H,R,C,false,main_stream);
    llmc_mlp_activation_backward(bdx.p,dh.p,pre.p,R*H,test_policy,main_stream);
    auto dg=dh.get();auto dx=bdx.get();
    for(int r=0;r<R;++r)for(int h=0;h<H;++h) {
        double expected=0;
        for(int c=0;c<C;++c)expected+=double(float(upstream[r*C+c]))*float(down[c*H+h]);
        check(std::abs(dg[r*H+h]-expected)<1e-6,"dH GEMM remains FP32");
        float value=float((floatX)float(expected*derivative_reference(p[r*H+h])));
        check(std::abs(float(dx[r*H+h])-value)<=.008f*std::abs(value)+1e-6f,"FP32 VJP single pack");
    }
    printf("llmc_mlp_fp32_boundary: forward=1 backward=1 full_gemm_replay=1 avoided_bf16_rounding_witnesses=%zu result=%s\n",precision_witnesses,failures?"fail":"pass");
}
void test_seeded_initialization() {
    std::vector<floatX> expected;
    const bool target_shape=test_policy==LLMC_MLP_SWISH_POWER2_K8;
    for(int policy:{LLMC_MLP_GELU,LLMC_MLP_SWISH_POWER125_K8,LLMC_MLP_SWISH,LLMC_MLP_RELU_SQUARED,LLMC_MLP_SWISH_POWER2_K8}) {
        GPT2 m;gpt2_init_common(&m);m.config={};
        m.mlp_activation=policy;
        if(target_shape) {
            // Exercise the same descriptor and seed42 initializer as the actual
            // cold GPT-2 Small run, without allocating activations or an optimizer.
            gpt_build_from_descriptor(&m,"gpt2:rope:d12:t1024");
            check(m.config.num_layers==12&&m.config.channels==768&&m.config.num_heads==12&&
                m.config.vocab_size==50257&&m.config.padded_vocab_size==50304&&
                m.config.max_seq_len==1024&&m.num_parameters==123689472,"target initializer geometry");
        } else {
        m.config.max_seq_len=64;m.config.vocab_size=128;m.config.padded_vocab_size=128;
        m.config.num_layers=2;m.config.num_heads=2;m.config.channels=128;m.config.lexical_channels=128;
        m.config.position_encoding=LLMC_POSITION_ENCODING_ROPE;m.config.rope_rotary_dim=64;m.config.rope_theta=10000;
        gpt2_set_initializer_defaults(&m.config);
        gpt2_allocate_weights(&m);gpt2_initialize_weights(&m);
        }
        auto actual=read_device(static_cast<const floatX*>(m.params_memory),m.num_parameters);
        if(expected.empty())expected=actual;
        else check(actual.size()==expected.size()&&!memcmp(actual.data(),expected.data(),m.num_parameters_bytes),
            "seed42 initializer payload byte-identical across GELU/p1.25/Swish/ReLU-squared/p2");
        cudaFreeCheck(&m.params_memory);
    }
    if(target_shape)
        printf("llmc_mlp_initializer_geometry: layers=12 channels=768 heads=12 vocab=50257 padded_vocab=50304 sequence=1024 parameters=123689472 target_shape=1\n");
    printf("llmc_mlp_initializer: seed=42 selectors=gelu,swish_power125_k8,swish,relu_squared,swish_power2_k8 tensor_bytes_identical=1 result=%s\n",failures?"fail":"pass");
}
struct Run {float loss; std::vector<floatX> gradients;};
Run tiny_run(int recompute,const std::filesystem::path& output) {
    constexpr int B=8,T=64,C=128,L=2,V=128;
    GPT2 m;gpt2_init_common(&m);
    m.config={};m.config.max_seq_len=T;m.config.vocab_size=V;m.config.padded_vocab_size=V;
    m.config.num_layers=L;m.config.num_heads=2;m.config.channels=C;m.config.lexical_channels=C;
    m.config.position_encoding=LLMC_POSITION_ENCODING_ROPE;m.config.rope_rotary_dim=64;m.config.rope_theta=10000;
    gpt2_set_initializer_defaults(&m.config);
    m.mlp_activation=test_policy;m.gelu_fusion=0;m.recompute=recompute;
    gpt2_allocate_weights(&m);
    std::vector<floatX> weights(m.num_parameters);
    size_t at=0;
    for(int tensor=0;tensor<NUM_PARAMETER_TENSORS;++tensor)for(size_t j=0;j<m.param_elements[tensor];++j,++at)
        weights[at]=(floatX)((tensor==2||tensor==8||tensor==14)?1.f:
            (tensor==0||tensor==4||tensor==6||tensor==10||tensor==12)?0.02f*sinf(float(at)*.071f):0.f);
    cudaCheck(cudaMemcpy(m.params_memory,weights.data(),m.num_parameters_bytes,cudaMemcpyHostToDevice));
    set_zero_configs(&multi_gpu_config,0,m.num_parameters);
    gpt2_allocate_state(&m,B,T);
    std::vector<int> inputs(B*T),targets(B*T);
    for(int i=0;i<B*T;++i){inputs[i]=(i*7+3)%V;targets[i]=(i*11+5)%V;}
    gpt2_forward(&m,inputs.data(),B,T);
    gpt2_backward_and_reduce(&m,inputs.data(),targets.data(),1,0,true);
    Run result;result.loss=m.mean_loss;result.gradients.resize(m.num_parameters);
    cudaCheck(cudaStreamSynchronize(main_stream));
    cudaCheck(cudaMemcpy(result.gradients.data(),m.grads_memory,m.num_parameters_bytes,cudaMemcpyDeviceToHost));
    check(std::isfinite(result.loss)&&std::isfinite(gpt2_calculate_grad_norm(&m,&multi_gpu_config)),"tiny finite forward/backward");
    if(recompute==1) {
        float norm=gpt2_calculate_grad_norm(&m,&multi_gpu_config);
        gpt2_update(&m,5e-4f,.9f,.95f,1e-8f,0.f,norm>1?1/norm:1,1,&multi_gpu_config);
        check(std::isfinite(m.mean_loss),"one diagnostic optimizer step");
        auto tokens=output/"tokens.npy";write_tokens(tokens);
        DataLoader loader={};dataloader_init_with_policy(&loader,tokens.string().c_str(),B,T,0,1,0,1,0);
        dataloader_next_batch(&loader);
        auto path=output/"custom_model.bin";
        gpt2_write_to_checkpoint(&m,path.string().c_str());
        auto state=output/"custom_state.bin";
        save_state(state.string().c_str(),1,&m,&loader,LLMC_SEQUENCE_BOUNDARY_ROW_RESET);
        int header[256];FILE* file=fopenCheck(state.string().c_str(),"rb");freadCheck(header,sizeof(int),256,file);fcloseCheck(file);
        check(header[1]==LLMC_OPTIMIZER_STATE_VERSION_MLP_ACTIVATION&&header[44]==2&&llmc_mlp_contract_matches(header,44,test_policy),"optimizer state custom contract");
        mutate_header(path,output/"invalid_model_formula.bin",21);
        mutate_header(state,output/"invalid_state_numerics.bin",51);
        change_state_activation(state,output/"invalid_state_activation.bin",
            test_policy==LLMC_MLP_SWISH?LLMC_MLP_RELU_SQUARED:LLMC_MLP_SWISH);
        GPT2 restored;gpt2_init_common(&restored);gpt2_build_from_checkpoint(&restored,path.string().c_str());
        check(restored.mlp_activation==test_policy,"checkpoint auto selects custom activation");
        auto checkpoint_weights=read_device(static_cast<const floatX*>(m.params_memory),m.num_parameters);
        auto restored_weights=read_device(static_cast<const floatX*>(restored.params_memory),m.num_parameters);
        check(!memcmp(checkpoint_weights.data(),restored_weights.data(),m.num_parameters_bytes),"checkpoint weights byte identical");
        gpt2_allocate_state(&restored,B,T);
        DataLoader resumed={};dataloader_init_with_policy(&resumed,tokens.string().c_str(),B,T,0,1,0,1,0);
        int restored_step=-1;load_state(&restored_step,&restored,&resumed,state.string().c_str(),LLMC_SEQUENCE_BOUNDARY_ROW_RESET);
        check(restored_step==1&&resumed.current_sample_idx==loader.current_sample_idx&&
            restored.rng_state==m.rng_state&&restored.rng_state_last_update==m.rng_state_last_update,"actual state load step/cursor/RNG");
        auto resumed_weights=read_device(static_cast<const floatX*>(restored.params_memory),m.num_parameters);
        check(!memcmp(checkpoint_weights.data(),resumed_weights.data(),m.num_parameters_bytes),"master-weight stochastic roundtrip exact");
        check(read_device(restored.m_memory,m.num_parameters)==read_device(m.m_memory,m.num_parameters)&&
            read_device(restored.v_memory,m.num_parameters)==read_device(m.v_memory,m.num_parameters)&&
            read_device(restored.master_weights,m.num_parameters)==read_device(m.master_weights,m.num_parameters),"actual state load optimizer payload exact");
        dataloader_next_batch(&loader);dataloader_next_batch(&resumed);
        check(!memcmp(loader.inputs,resumed.inputs,B*T*sizeof(int))&&!memcmp(loader.targets,resumed.targets,B*T*sizeof(int)),"resume consumes identical next batch");
        gpt2_forward(&m,loader.inputs,B,T);gpt2_backward_and_reduce(&m,loader.inputs,loader.targets,1,0,true);
        float next_loss=m.mean_loss;
        norm=gpt2_calculate_grad_norm(&m,&multi_gpu_config);gpt2_update(&m,5e-4f,.9f,.95f,1e-8f,0.f,norm>1?1/norm:1,2,&multi_gpu_config);
        gpt2_forward(&restored,resumed.inputs,B,T);gpt2_backward_and_reduce(&restored,resumed.inputs,resumed.targets,1,0,true);
        check(std::abs(next_loss-restored.mean_loss)<1e-6f,"continued resumed loss parity");
        norm=gpt2_calculate_grad_norm(&restored,&multi_gpu_config);gpt2_update(&restored,5e-4f,.9f,.95f,1e-8f,0.f,norm>1?1/norm:1,2,&multi_gpu_config);
        auto continuous_master=read_device(m.master_weights,m.num_parameters),resumed_master=read_device(restored.master_weights,m.num_parameters);
        size_t continuation_bad=0;
        for(size_t i=0;i<m.num_parameters;++i)if(!std::isfinite(resumed_master[i])||std::abs(continuous_master[i]-resumed_master[i])>1e-6f+1e-4f*std::abs(continuous_master[i]))++continuation_bad;
        check(continuation_bad==0,"continued resumed optimizer-update parity");
        dataloader_free(&resumed);dataloader_free(&loader);gpt2_free(&restored);
    }
    gpt2_free(&m);
    return result;
}
}
int main(int argc,char** argv) {
    // These modes invoke the actual fail-closed readers. The parent qualification
    // runner must require nonzero exit on the deliberately corrupted fixtures.
    if(argc>=3&&(!strcmp(argv[1],"--load-model")||!strcmp(argv[1],"--load-state"))) {
        bool state=!strcmp(argv[1],"--load-state");
        if(argc!=(state?5:3))return 2;
        multi_gpu_config=multi_gpu_config_init(1,0,1,nullptr,nullptr,nullptr);common_start(false);
        GPT2 model;gpt2_init_common(&model);gpt2_build_from_checkpoint(&model,argv[2]);
        if(state){set_zero_configs(&multi_gpu_config,0,model.num_parameters);gpt2_allocate_state(&model,8,64);
            DataLoader loader={};dataloader_init_with_policy(&loader,argv[4],8,64,0,1,0,1,0);
            int step=-1;load_state(&step,&model,&loader,argv[3],LLMC_SEQUENCE_BOUNDARY_ROW_RESET);
            dataloader_free(&loader);gpt2_free(&model);
        }else cudaFreeCheck(&model.params_memory);
        common_free(model);multi_gpu_config_free(&multi_gpu_config);return 0;
    }
    const char* output_text=nullptr;
    for(int i=1;i<argc;i+=2) {
        if(i+1>=argc)return 2;
        if(!strcmp(argv[i],"--output-dir")&&!output_text)output_text=argv[i+1];
        else if(!strcmp(argv[i],"--activation")) {
            if(!llmc_parse_mlp_activation(argv[i+1],&test_policy)||!llmc_mlp_activation_is_custom(test_policy))return 2;
        } else return 2;
    }
    if(!output_text){fprintf(stderr,"usage: test_mlp_activation [--activation SELECTOR] --output-dir NEW_DIRECTORY\n");return 2;}
    std::filesystem::path output=output_text;
    if(std::filesystem::exists(output)||!std::filesystem::create_directories(output)){fprintf(stderr,"Fresh test output required\n");return 2;}
    multi_gpu_config=multi_gpu_config_init(1,0,1,nullptr,nullptr,nullptr);
    common_start(false);
    printf("mlp_activation: %s\n",llmc_mlp_activation_name(test_policy));
    test_edges();test_gemm_boundary();test_seeded_initialization();
    auto baseline=tiny_run(0,output);
    for(int recompute:{1,2}) {
        auto other=tiny_run(recompute,output);
        check(std::abs(other.loss-baseline.loss)<1e-6,"tiny recompute loss parity");
        check(other.gradients.size()==baseline.gradients.size(),"gradient inventory parity");
        size_t bad=0;
        for(size_t i=0;i<other.gradients.size();++i) {
            float a=float(baseline.gradients[i]),b=float(other.gradients[i]);
            if(!std::isfinite(b)||std::abs(a-b)>1e-4f+.03f*std::abs(a))++bad;
        }
        check(bad==0,"tiny fullmodel recompute gradient parity");
    }
    printf("llmc_mlp_model: recompute_modes=0,1,2 finite_backward=1 checkpoint_roundtrip=1 optimizer_state_contract=1 actual_state_load=1 resumed_update=1 result=%s\n",failures?"fail":"pass");
    GPT2 empty={};common_free(empty);multi_gpu_config_free(&multi_gpu_config);
    printf("llmc_mlp_activation_cuda: result=%s\n",failures?"fail":"pass");
    return failures?1:0;
}
