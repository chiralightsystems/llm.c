#ifndef LLMC_CACHED_MATMUL_H
#define LLMC_CACHED_MATMUL_H

// Include after llm.c's cuda_common.h / cublas_common.h. This freezes the
// existing forward-only matmul descriptors and first heuristic outside capture.
// CUDA 13.3 AlgoGetHeuristic invalidates capture even after an eager warmup.
#include <stdexcept>
#include <string>
#include "llmc/cublas_common.h"

class LlmcCachedMatmul {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr, d_layout = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic = {};
    floatX *output, *input, *weight;
    cudaStream_t owner;
public:
    LlmcCachedMatmul(floatX* out, floatX* inp, floatX* w, floatX* bias,
                    int batch, int channels, int outputs, cudaStream_t owner_stream,
                    floatX* pre_gelu = nullptr)
        : output(out), input(inp), weight(w), owner(owner_stream) {
        cudaStreamCaptureStatus capture;
        cudaCheck(cudaStreamIsCapturing(owner, &capture));
        if (capture != cudaStreamCaptureStatusNone)
            throw std::runtime_error("GEMM plans must be prepared outside graph capture");
        if (!out || !inp || !w || batch <= 0 || channels <= 0 || outputs <= 0 ||
            ((uintptr_t)out | (uintptr_t)inp | (uintptr_t)w | (uintptr_t)bias | (uintptr_t)pre_gelu) % 16)
            throw std::runtime_error("Invalid or unaligned cached GEMM operands");
        cublasCheck(cublasLtMatmulDescCreate(&operation, cublas_compute, CUDA_R_32F));
        cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
        cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &normal, sizeof(normal)));
        cublasCheck(cublasLtMatrixLayoutCreate(&a_layout, CUBLAS_LOWP, channels, outputs, channels));
        cublasCheck(cublasLtMatrixLayoutCreate(&b_layout, CUBLAS_LOWP, channels, batch, channels));
        cublasCheck(cublasLtMatrixLayoutCreate(&c_layout, CUBLAS_LOWP, outputs, batch, outputs));
        cublasCheck(cublasLtMatrixLayoutCreate(&d_layout, CUBLAS_LOWP, outputs, batch, outputs));
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        if (pre_gelu) {
            int64_t ld = outputs;
            cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld, sizeof(ld)));
            cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
            epilogue = bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        } else if (bias) epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        if (bias) {
            cublasDataType_t type = CUBLAS_LOWP;
            cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &type, sizeof(type)));
            cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        }
        cublasDataType_t scale_type = CUDA_R_32F;
        cublasCheck(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));
        cublasLtMatmulPreference_t preference;
        cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
        cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));
        int returned = 0;
        auto status = cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operation, a_layout, b_layout,
                                                    c_layout, d_layout, preference, 1, &heuristic, &returned);
        cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
        cublasCheck(status);
        if (returned != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("No cached GEMM algorithm for M=" + std::to_string(outputs) +
                                     " N=" + std::to_string(batch) + " K=" + std::to_string(channels));
    }
    LlmcCachedMatmul(const LlmcCachedMatmul&) = delete;
    LlmcCachedMatmul& operator=(const LlmcCachedMatmul&) = delete;
    ~LlmcCachedMatmul() {
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
        if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
        if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
        if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    }
    void execute(cudaStream_t stream) const {
        if (stream != owner) throw std::runtime_error("Cached GEMM stream ownership changed");
        const float alpha = 1.0f, beta = 0.0f;
        cublasCheck(cublasLtMatmul(cublaslt_handle, operation, &alpha, weight, a_layout, input, b_layout,
            &beta, output, c_layout, output, d_layout, &heuristic.algo,
            cublaslt_workspace, cublaslt_workspace_size, stream));
        cudaCheck(cudaGetLastError());
    }
};
#endif
