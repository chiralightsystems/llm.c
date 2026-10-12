#ifndef LLMC_FFN_INTERVENTIONS_CUH
#define LLMC_FFN_INTERVENTIONS_CUH
#include "ffn_interventions.h"
#include <cassert>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// Execution-only revision: local operands and sums remain FP32, with the same
// BF16/FP32 parameter and gradient publication boundaries as the original path.
#define LLMC_TIED_GUM_OPTIMIZED_KERNELS 1
constexpr int LLMC_GUM_LOCAL_WARPS = 8;
constexpr int LLMC_GUM_LOCAL_USES_PER_WARP = 32;
constexpr int LLMC_GUM_LOCAL_USES_PER_CTA = LLMC_GUM_LOCAL_WARPS * LLMC_GUM_LOCAL_USES_PER_WARP;
constexpr int LLMC_GUM_BANK_MAX_PARTITIONS = 1024;
constexpr int LLMC_GUM_BANK_MIN_USES_PER_PARTITION = 1024;
constexpr int LLMC_GUM_BANK_STAGE_USES = 32;
constexpr int LLMC_GUM_BANK_THREADS = 512;

inline void llmc_require_ffn_geometry(int policy,int rows,int hidden,int block) {
    if (!llmc_ffn_geometry_valid(policy,rows,hidden,block)) {
        fprintf(stderr,"Invalid FFN intervention geometry: mode=%d rows=%d hidden=%d block=%d\n",
            policy,rows,hidden,block);
        exit(EXIT_FAILURE);
    }
}
__global__ void llmc_swiglu_forward_kernel(floatX* out,float* pre,const floatX* bias,
        std::size_t count,int hidden) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const int j=i%hidden;
    const std::size_t p=(i/hidden)*2*hidden+j;
    const float gate=pre[p]+(bias?float(bias[j]):0.0f);
    const float value=pre[p+hidden]+(bias?float(bias[j+hidden]):0.0f);
    pre[p]=gate; pre[p+hidden]=value;
    out[i]=(floatX)llmc_ffn_swiglu(gate,value);
}
__global__ void llmc_swiglu_backward_kernel(floatX* dpre,const float* dz,const float* pre,
        std::size_t count,int hidden) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const std::size_t p=(i/hidden)*2*hidden+i%hidden;
    float dg,dv;
    llmc_ffn_swiglu_vjp(dz[i],pre[p],pre[p+hidden],&dg,&dv);
    dpre[p]=(floatX)dg; dpre[p+hidden]=(floatX)dv;
}
inline void llmc_swiglu_forward(floatX* out,float* pre,const floatX* bias,int rows,
        int hidden,cudaStream_t stream) {
    llmc_require_ffn_geometry(LLMC_MLP_SWIGLU,rows,hidden,0);
    const std::size_t count=std::size_t(rows)*hidden;
    llmc_swiglu_forward_kernel<<<CEIL_DIV(count,256),256,0,stream>>>(out,pre,bias,count,hidden);
    cudaCheck(cudaGetLastError());
}
inline void llmc_swiglu_backward(floatX* dpre,const float* dz,const float* pre,int rows,
        int hidden,cudaStream_t stream) {
    llmc_require_ffn_geometry(LLMC_MLP_SWIGLU,rows,hidden,0);
    const std::size_t count=std::size_t(rows)*hidden;
    llmc_swiglu_backward_kernel<<<CEIL_DIV(count,256),256,0,stream>>>(dpre,dz,pre,count,hidden);
    cudaCheck(cudaGetLastError());
}

// The shared banks are transposed [input,output] and padded. Both the forward
// row reads and the input-adjoint column reads avoid shared-memory bank
// conflicts. Converting parameters once per CTA introduces no new rounding.
__device__ __forceinline__ void llmc_tiled_gum_load_banks(
        float banks[3][LLMC_TIED_GUM_BLOCK_WIDTH][LLMC_TIED_GUM_BLOCK_WIDTH+1],
        const floatX* u,const floatX* g,const floatX* v) {
    constexpr int b=LLMC_TIED_GUM_BLOCK_WIDTH;
    for(int i=threadIdx.x;i<3*b*b;i+=blockDim.x) {
        const int family=i/(b*b),w=i%(b*b),o=w/b,k=w%b;
        const floatX* source=family==0?u:(family==1?g:v);
        banks[family][k][o]=float(source[w]);
    }
    __syncthreads();
}
__device__ __forceinline__ void llmc_tiled_gum_warp_channel(float x,
        const float banks[3][LLMC_TIED_GUM_BLOCK_WIDTH][LLMC_TIED_GUM_BLOCK_WIDTH+1],
        int o,float* a,float* s,float* c,float* value,float gate_scale) {
    constexpr int b=LLMC_TIED_GUM_BLOCK_WIDTH;
    float uh=0.0f,gh=0.0f,vh=0.0f;
    #pragma unroll
    for(int k=0;k<b;++k) {
        const float h=__shfl_sync(0xffffffffu,x,k);
        uh+=banks[0][k][o]*h;
        gh+=banks[1][k][o]*h;
        vh+=banks[2][k][o]*h;
    }
    llmc_ffn_sigmoid(gh,s,c);
    *value=vh;
    *a=uh+(gate_scale==1.0f?*s:gate_scale*(*s))*vh;
}

// Eight warps each process 32 uses, sharing one bank load across 256 blocks.
// All lanes of a participating warp are active, including the last CTA. Bias
// is added once to a fresh expansion GEMM before EACH forward/replay.
__global__ void llmc_tiled_gum_forward_kernel(floatX* out,float* h,const floatX* bias,
        const floatX* u,const floatX* g,const floatX* v,std::size_t uses,
        int hidden,float gate_scale) {
    constexpr int b=LLMC_TIED_GUM_BLOCK_WIDTH;
    __shared__ float banks[3][b][b+1];
    llmc_tiled_gum_load_banks(banks,u,g,v);
    const int o=threadIdx.x%b,warp=threadIdx.x/b;
    const std::size_t first=std::size_t(blockIdx.x)*LLMC_GUM_LOCAL_USES_PER_CTA;
    for(std::size_t use=first+warp;use<uses && use<first+LLMC_GUM_LOCAL_USES_PER_CTA;
            use+=LLMC_GUM_LOCAL_WARPS) {
        const std::size_t base=use*b;
        const float x=h[base+o]+(bias?float(bias[(base+o)%hidden]):0.0f);
        h[base+o]=x;
        float a,s,c,value;
        llmc_tiled_gum_warp_channel(x,banks,o,&a,&s,&c,&value,gate_scale);
        out[base+o]=(floatX)llmc_relu_squared(a);
    }
}
__global__ void llmc_tiled_gum_backward_input_kernel(floatX* dh,float* dlocal,
        const float* dz,const float* h,const floatX* u,const floatX* g,const floatX* v,
        std::size_t count,float gate_scale) {
    constexpr int b=LLMC_TIED_GUM_BLOCK_WIDTH;
    __shared__ float banks[3][b][b+1];
    llmc_tiled_gum_load_banks(banks,u,g,v);
    const int k=threadIdx.x%b,warp=threadIdx.x/b;
    const std::size_t uses=count/b;
    const std::size_t first=std::size_t(blockIdx.x)*LLMC_GUM_LOCAL_USES_PER_CTA;
    for(std::size_t use=first+warp;use<uses && use<first+LLMC_GUM_LOCAL_USES_PER_CTA;
            use+=LLMC_GUM_LOCAL_WARPS) {
        const std::size_t base=use*b;
        float a,s,c,value,pu,pg,pv;
        llmc_tiled_gum_warp_channel(h[base+k],banks,k,&a,&s,&c,&value,gate_scale);
        llmc_ffn_gum_local_vjp(dz[base+k],a,s,c,value,&pu,&pg,&pv,gate_scale);
        dlocal[base+k]=pu; dlocal[count+base+k]=pg; dlocal[2*count+base+k]=pv;
        float sum=0.0f;
        #pragma unroll
        for(int o=0;o<b;++o) {
            const float du=__shfl_sync(0xffffffffu,pu,o);
            const float dg=__shfl_sync(0xffffffffu,pg,o);
            const float dv=__shfl_sync(0xffffffffu,pv,o);
            sum+=banks[0][k][o]*du+banks[1][k][o]*dg+banks[2][k][o]*dv;
        }
        dh[base+k]=(floatX)sum;
    }
}

// Every partition owns a disjoint interval of uses and the complete 3x32x32
// FP32 outer-product sum. Coalesced operand staging is shared across all 1024
// bank coordinates, rather than rereading h/dlocal once per coordinate.
__global__ void llmc_tiled_gum_backward_bank_partial_kernel(float* partials,
        const float* dlocal,const float* h,std::size_t count,int partitions) {
    constexpr int b=LLMC_TIED_GUM_BLOCK_WIDTH;
    constexpr int stage=LLMC_GUM_BANK_STAGE_USES;
    constexpr int warps=LLMC_GUM_BANK_THREADS/b;
    static_assert(warps==16 && b==32,"GUM bank tile requires sixteen full warps");
    __shared__ float hs[stage*b],ds[3][stage*b];
    const int k=threadIdx.x%b,o0=threadIdx.x/b,o1=o0+warps;
    const std::size_t uses=count/b,chunk=CEIL_DIV(uses,std::size_t(partitions));
    const std::size_t begin=std::size_t(blockIdx.x)*chunk;
    const std::size_t end=begin+chunk<uses?begin+chunk:uses;
    float su0=0.0f,sg0=0.0f,sv0=0.0f,su1=0.0f,sg1=0.0f,sv1=0.0f;
    for(std::size_t tile=begin;tile<end;tile+=stage) {
        for(int i=threadIdx.x;i<stage*b;i+=blockDim.x) {
            const std::size_t use=tile+i/b,index=use*b+i%b;
            hs[i]=use<end?h[index]:0.0f;
            ds[0][i]=use<end?dlocal[index]:0.0f;
            ds[1][i]=use<end?dlocal[count+index]:0.0f;
            ds[2][i]=use<end?dlocal[2*count+index]:0.0f;
        }
        __syncthreads();
        #pragma unroll
        for(int t=0;t<stage;++t) {
            const float x=hs[t*b+k];
            su0+=ds[0][t*b+o0]*x; sg0+=ds[1][t*b+o0]*x; sv0+=ds[2][t*b+o0]*x;
            su1+=ds[0][t*b+o1]*x; sg1+=ds[1][t*b+o1]*x; sv1+=ds[2][t*b+o1]*x;
        }
        __syncthreads();
    }
    const std::size_t base=std::size_t(blockIdx.x)*3*b*b;
    const int w0=o0*b+k,w1=o1*b+k;
    partials[base+w0]=su0; partials[base+b*b+w0]=sg0; partials[base+2*b*b+w0]=sv0;
    partials[base+w1]=su1; partials[base+b*b+w1]=sg1; partials[base+2*b*b+w1]=sv1;
}
// Ordered FP32 sum of partitions, then one add to the previous microbatch's
// published gradient and one cast. No atomics, averaging or BF16 partials.
__global__ void llmc_tiled_gum_backward_bank_finalize_kernel(floatX* du,floatX* dg,floatX* dv,
        const float* partials,int partitions) {
    constexpr int b=LLMC_TIED_GUM_BLOCK_WIDTH;
    const int w=blockIdx.x*blockDim.x+threadIdx.x;
    if(w>=b*b) return;
    float su=0.0f,sg=0.0f,sv=0.0f;
    for(int p=0;p<partitions;++p) {
        const std::size_t base=std::size_t(p)*3*b*b;
        su+=partials[base+w]; sg+=partials[base+b*b+w]; sv+=partials[base+2*b*b+w];
    }
    du[w]=(floatX)(float(du[w])+su);
    dg[w]=(floatX)(float(dg[w])+sg);
    dv[w]=(floatX)(float(dv[w])+sv);
}
inline int llmc_tiled_gum_backward_bank_partitions(int rows,int hidden,int block) {
    llmc_require_ffn_geometry(LLMC_MLP_TIED_GUM_RELU_SQUARED,rows,hidden,block);
    const std::size_t uses=std::size_t(rows)*hidden/block;
    const std::size_t p=CEIL_DIV(uses,std::size_t(LLMC_GUM_BANK_MIN_USES_PER_PARTITION));
    return int(p<LLMC_GUM_BANK_MAX_PARTITIONS?p:LLMC_GUM_BANK_MAX_PARTITIONS);
}
// Caller-owned scratch, in float elements: [partition, family(U/G/V), o, k].
// It must be disjoint from all operands/outputs and live through both launches.
inline std::size_t llmc_tiled_gum_backward_bank_scratch_floats(int rows,int hidden,int block) {
    return std::size_t(llmc_tiled_gum_backward_bank_partitions(rows,hidden,block))*3*block*block;
}
inline void llmc_tiled_gum_backward_bank(floatX* du,floatX* dg,floatX* dv,
        const float* dlocal,const float* h,int rows,int hidden,int block,
        float* bank_partials,std::size_t bank_partial_capacity,cudaStream_t stream) {
    const int partitions=llmc_tiled_gum_backward_bank_partitions(rows,hidden,block);
    const std::size_t required=std::size_t(partitions)*3*block*block;
    if(!bank_partials || bank_partial_capacity<required) {
        fprintf(stderr,"GUM bank reduction requires %zu FP32 scratch elements; supplied %zu\n",
                required,bank_partial_capacity);
        exit(EXIT_FAILURE);
    }
    const std::size_t count=std::size_t(rows)*hidden;
    llmc_tiled_gum_backward_bank_partial_kernel<<<partitions,LLMC_GUM_BANK_THREADS,0,stream>>>(
            bank_partials,dlocal,h,count,partitions);
    cudaCheck(cudaGetLastError());
    llmc_tiled_gum_backward_bank_finalize_kernel<<<CEIL_DIV(block*block,256),256,0,stream>>>(
            du,dg,dv,bank_partials,partitions);
    cudaCheck(cudaGetLastError());
}
inline void llmc_tiled_gum_forward(floatX* out,float* h,const floatX* bias,
        const floatX* u,const floatX* g,const floatX* v,int rows,int hidden,int block,
        cudaStream_t stream,float gate_scale = 1.0f) {
    llmc_require_ffn_geometry(LLMC_MLP_TIED_GUM_RELU_SQUARED,rows,hidden,block);
    if (!llmc_ffn_gate_scale_valid(gate_scale)) { fprintf(stderr,"Invalid GUM gate scale\n");exit(EXIT_FAILURE); }
    const std::size_t count=std::size_t(rows)*hidden;
    const std::size_t uses=count/block;
    llmc_tiled_gum_forward_kernel<<<CEIL_DIV(uses,LLMC_GUM_LOCAL_USES_PER_CTA),LLMC_GUM_LOCAL_WARPS*block,0,stream>>>(
            out,h,bias,u,g,v,uses,hidden,gate_scale);
    cudaCheck(cudaGetLastError());
}
inline void llmc_tiled_gum_backward(floatX* dh,floatX* du,floatX* dg,floatX* dv,
        float* dlocal,const float* dz,const float* h,const floatX* u,const floatX* g,
        const floatX* v,int rows,int hidden,int block,float* bank_partials,
        std::size_t bank_partial_capacity,cudaStream_t stream,float gate_scale = 1.0f) {
    llmc_require_ffn_geometry(LLMC_MLP_TIED_GUM_RELU_SQUARED,rows,hidden,block);
    if (!llmc_ffn_gate_scale_valid(gate_scale)) { fprintf(stderr,"Invalid GUM gate scale\n");exit(EXIT_FAILURE); }
    const std::size_t required=llmc_tiled_gum_backward_bank_scratch_floats(rows,hidden,block);
    if(!bank_partials || bank_partial_capacity<required) {
        fprintf(stderr,"GUM bank reduction requires %zu FP32 scratch elements; supplied %zu\n",
                required,bank_partial_capacity);
        exit(EXIT_FAILURE);
    }
    const std::size_t count=std::size_t(rows)*hidden;
    const std::size_t uses=count/block;
    llmc_tiled_gum_backward_input_kernel<<<CEIL_DIV(uses,LLMC_GUM_LOCAL_USES_PER_CTA),LLMC_GUM_LOCAL_WARPS*block,0,stream>>>(
            dh,dlocal,dz,h,u,g,v,count,gate_scale);
    cudaCheck(cudaGetLastError());
    llmc_tiled_gum_backward_bank(du,dg,dv,dlocal,h,rows,hidden,block,bank_partials,bank_partial_capacity,stream);
}
#endif
