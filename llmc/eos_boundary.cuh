#ifndef LLMC_EOS_BOUNDARY_CUH
#define LLMC_EOS_BOUNDARY_CUH
#include "cuda_common.h"
#include "eos_boundary.h"
#include <type_traits>

// cuDNN ragged SDPA statistics remain dense within an execution group. Store
// only real entries in the unchanged [B,H,T] per-layer tape and gather them
// back before backward. O and its adjoint keep their original packed layout.
__global__ void llmc_eos_stats_kernel(float* dense, float* grouped,
        const int32_t* lengths, const int64_t* output_offsets,
        int count, int S, int H, int T, int C, bool gather) {
    const size_t total = (size_t)count * H * S;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)blockDim.x * gridDim.x) {
        const int t = idx % S;
        const int h = (idx / S) % H;
        const int segment = idx / ((size_t)H * S);
        if (t >= lengths[segment]) { if (gather) grouped[idx] = 0.0f; continue; }
        const size_t token = (size_t)(output_offsets[segment] / C) + t;
        const size_t dense_idx = (token / T * H + h) * T + token % T;
        if (gather) grouped[idx] = dense[dense_idx]; else dense[dense_idx] = grouped[idx];
    }
}
void llmc_eos_copy_stats_cuda(float* dense, float* grouped,
        const int32_t* lengths, const int64_t* output_offsets,
        int count, int S, int H, int T, int C, bool gather, cudaStream_t stream) {
    const size_t total = (size_t)count * H * S;
    llmc_eos_stats_kernel<<<(unsigned int)std::min((size_t)65535, (total + 255) / 256), 256, 0, stream>>>(
        dense, grouped, lengths, output_offsets, count, S, H, T, C, gather);
    cudaCheck(cudaGetLastError());
}

// SM120 ragged backward requires separate token-contiguous Q/K/V planes.
// Copy representation bits, including signed zero and NaN payloads: there is
// no FP32 intermediate, rounding, attention arithmetic, or precision change.
using LlmcEosBits = typename std::conditional<sizeof(floatX)==2,uint16_t,uint32_t>::type;
static_assert(sizeof(LlmcEosBits)==sizeof(floatX), "unsupported EOS scalar width");
__global__ void llmc_eos_relayout_qkv_kernel(const LlmcEosBits* input,LlmcEosBits* output,
        size_t tokens,int C,bool to_planar) {
    const size_t plane=tokens*C,total=3*plane;
    for(size_t packed=blockIdx.x*blockDim.x+threadIdx.x;packed<total;
        packed+=(size_t)blockDim.x*gridDim.x) {
        const size_t token=packed/(3ull*C),kind=(packed/C)%3,channel=packed%C;
        const size_t planar=kind*plane+token*C+channel;
        if(to_planar) output[planar]=input[packed]; else output[packed]=input[planar];
    }
}
void llmc_eos_relayout_qkv_cuda(const floatX* input,floatX* output,size_t tokens,
        int C,bool to_planar,cudaStream_t stream) {
    if(!tokens||C<=0||tokens>SIZE_MAX/(3ull*C*sizeof(floatX))) {
        fprintf(stderr,"EOS planar relayout shape overflow\n");exit(EXIT_FAILURE);
    }
    const size_t elements=3*tokens*C,bytes=elements*sizeof(floatX);
    const uintptr_t in=(uintptr_t)input,out=(uintptr_t)output;
    if(!input||!output||!tokens||C<=0||in>UINTPTR_MAX-bytes||out>UINTPTR_MAX-bytes||
       !(in+bytes<=out||out+bytes<=in)) {
        fprintf(stderr,"EOS planar relayout needs disjoint input/output spans\n");exit(EXIT_FAILURE);
    }
    llmc_eos_relayout_qkv_kernel<<<(unsigned int)std::min((size_t)65535,(elements+255)/256),256,0,stream>>>(
        (const LlmcEosBits*)input,(LlmcEosBits*)output,tokens,C,to_planar);
    cudaCheck(cudaGetLastError());
}
#endif
