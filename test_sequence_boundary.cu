#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "llmc/cuda_common.h"
#include "llmc/fused_classifier.cuh"

cudaDeviceProp deviceProp;

static int failures = 0;

#define CHECK_BOUNDARY(condition, message)                                  \
    do {                                                                    \
        if (!(condition)) {                                                  \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static bool close_enough(float actual, float expected, float tolerance = 2.0e-3f) {
    return std::fabs(actual - expected) <= tolerance;
}

int main() {
    constexpr int B = 2;
    constexpr int T = 3;
    // The production classifier is specialized for a large, padded
    // vocabulary. Exercise both its packed and unaligned vocabulary paths.
    constexpr int V = 8191;
    constexpr int P = 8192;
    constexpr int N = B * T;
    constexpr int SUPERVISED = B * (T - 1);

    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

    std::vector<floatX> host_logits(static_cast<size_t>(N) * P, (floatX)0.0f);
    const std::vector<int> host_targets = {0, 1, 2, 3, 0, 1};
    std::vector<float> host_losses(N, 0.0f);

    floatX* logits = nullptr;
    int* targets = nullptr;
    float* losses = nullptr;
    cudaCheck(cudaMalloc((void**)&logits, host_logits.size() * sizeof(floatX)));
    cudaCheck(cudaMalloc((void**)&targets, host_targets.size() * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&losses, host_losses.size() * sizeof(float)));
    cudaCheck(cudaMemcpy(
        targets,
        host_targets.data(),
        host_targets.size() * sizeof(int),
        cudaMemcpyHostToDevice));

    // The standalone llm.c default remains the original flattened-stream loss.
    cudaCheck(cudaMemcpy(
        logits,
        host_logits.data(),
        host_logits.size() * sizeof(floatX),
        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(losses, 0, host_losses.size() * sizeof(float)));
    fused_classifier(
        logits,
        losses,
        1.0f / N,
        targets,
        B,
        T,
        V,
        P,
        std::bool_constant<false>{},
        stream);
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaMemcpy(
        host_losses.data(),
        losses,
        host_losses.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    for (int idx = 0; idx < N; ++idx) {
        CHECK_BOUNDARY(
            close_enough(host_losses[idx], std::log((float)V)),
            "flat_stream scores every position");
    }

    // row_reset masks each sequence-final loss and all corresponding dlogits.
    std::fill(host_logits.begin(), host_logits.end(), (floatX)1.0f);
    cudaCheck(cudaMemcpy(
        logits,
        host_logits.data(),
        host_logits.size() * sizeof(floatX),
        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(losses, 0, host_losses.size() * sizeof(float)));
    fused_classifier(
        logits,
        losses,
        1.0f / SUPERVISED,
        targets,
        B,
        T,
        V,
        P,
        std::bool_constant<true>{},
        stream,
        true);
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaMemcpy(
        host_losses.data(),
        losses,
        host_losses.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(
        host_logits.data(),
        logits,
        host_logits.size() * sizeof(floatX),
        cudaMemcpyDeviceToHost));

    for (int idx = 0; idx < N; ++idx) {
        const bool masked = idx % T == T - 1;
        const float expected_loss = masked ? 0.0f : std::log((float)V);
        CHECK_BOUNDARY(
            close_enough(host_losses[idx], expected_loss),
            "row_reset loss mask matches sequence boundaries");
        if (masked) {
            for (int token = 0; token < P; ++token) {
                CHECK_BOUNDARY(
                    static_cast<float>(host_logits[idx * P + token]) == 0.0f,
                    "masked sequence-final dlogits are exactly zero");
            }
        } else {
            for (int token = 0; token < V; ++token) {
                const float indicator = token == host_targets[idx] ? 1.0f : 0.0f;
                const float expected = (1.0f / V - indicator) / SUPERVISED;
                CHECK_BOUNDARY(
                    close_enough(
                        static_cast<float>(host_logits[idx * P + token]),
                        expected),
                    "unmasked dlogit uses B*(T-1) normalization");
            }
        }
    }

    cudaCheck(cudaFree(losses));
    cudaCheck(cudaFree(targets));
    cudaCheck(cudaFree(logits));
    cudaCheck(cudaStreamDestroy(stream));

    if (failures != 0) {
        fprintf(stderr, "sequence-boundary tests failed: %d\n", failures);
        return EXIT_FAILURE;
    }
    printf("sequence-boundary loss and gradient tests passed\n");
    return EXIT_SUCCESS;
}
