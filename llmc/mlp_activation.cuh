#ifndef LLMC_MLP_ACTIVATION_CUH
#define LLMC_MLP_ACTIVATION_CUH
#include "mlp_activation.h"
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"

// Low-precision GEMM operands, explicit FP32 accumulation AND destination.
// No BF16 auxiliary/preactivation roundtrip and no activation epilogue.
inline void llmc_mlp_gemm_fp32(float* out, const floatX* weight, const floatX* inp,
        int m, int n, int k, bool transpose_weight, cudaStream_t stream) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasSetStream(cublas_handle, stream));
    cublasCheck(cublasGemmEx(cublas_handle,
        transpose_weight ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha, weight, CUBLAS_LOWP, transpose_weight ? k : m,
        inp, CUBLAS_LOWP, k, &beta, out, CUDA_R_32F, m,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
__global__ void llmc_mlp_activation_forward_kernel(floatX* out, float* pre,
        const floatX* bias, size_t count, int width, int policy) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = pre[i] + (bias ? float(bias[i % width]) : 0.0f);
    pre[i] = x;
    out[i] = (floatX)llmc_custom_mlp_activation(x, policy);
}
__global__ void llmc_mlp_activation_backward_kernel(floatX* out, const float* dact,
        const float* pre, size_t count, int policy) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (floatX)(dact[i] * llmc_custom_mlp_activation_derivative(pre[i], policy));
}
inline void llmc_mlp_activation_forward(floatX* out, float* pre, const floatX* inp,
        const floatX* weight, const floatX* bias, int rows, int channels, int hidden,
        int policy, cudaStream_t stream) {
    llmc_mlp_gemm_fp32(pre, weight, inp, hidden, rows, channels, true, stream);
    const size_t count = size_t(rows) * hidden;
    llmc_mlp_activation_forward_kernel<<<CEIL_DIV(count, 256),256,0,stream>>>(out, pre, bias, count, hidden, policy);
    cudaCheck(cudaGetLastError());
}
inline void llmc_mlp_activation_backward(floatX* out, const float* dact, const float* pre,
        size_t count, int policy, cudaStream_t stream) {
    llmc_mlp_activation_backward_kernel<<<CEIL_DIV(count,256),256,0,stream>>>(out,dact,pre,count,policy);
    cudaCheck(cudaGetLastError());
}
#endif
