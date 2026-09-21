#ifndef LLMC_MLP_ACTIVATION_H
#define LLMC_MLP_ACTIVATION_H
#include <cmath>
#include <cstring>

enum LlmcMlpActivation {
    LLMC_MLP_GELU = 0,
    LLMC_MLP_SWISH_POWER125_K8 = 1,
    LLMC_MLP_SWISH = 2,
    LLMC_MLP_RELU_SQUARED = 3,
    LLMC_MLP_SWISH_POWER2_K8 = 4
};
inline bool llmc_mlp_activation_is_custom(int policy) {
    return policy == LLMC_MLP_SWISH_POWER125_K8 || policy == LLMC_MLP_SWISH ||
        policy == LLMC_MLP_RELU_SQUARED || policy == LLMC_MLP_SWISH_POWER2_K8;
}
inline const char* llmc_mlp_activation_name(int policy) {
    return policy == LLMC_MLP_GELU ? "gelu" :
        policy == LLMC_MLP_SWISH_POWER125_K8 ? "swish_power125_k8" :
        policy == LLMC_MLP_SWISH ? "swish" :
        policy == LLMC_MLP_RELU_SQUARED ? "relu_squared" :
        policy == LLMC_MLP_SWISH_POWER2_K8 ? "swish_power2_k8" : "invalid";
}
inline bool llmc_parse_mlp_activation(const char* text, int* policy) {
    if (!strcmp(text, "gelu")) { *policy = LLMC_MLP_GELU; return true; }
    if (!strcmp(text, "swish_power125_k8")) { *policy = LLMC_MLP_SWISH_POWER125_K8; return true; }
    if (!strcmp(text, "swish")) { *policy = LLMC_MLP_SWISH; return true; }
    if (!strcmp(text, "relu_squared")) { *policy = LLMC_MLP_RELU_SQUARED; return true; }
    if (!strcmp(text, "swish_power2_k8")) { *policy = LLMC_MLP_SWISH_POWER2_K8; return true; }
    return false;
}
#ifdef __CUDACC__
#define LLMC_MLP_INLINE __host__ __device__ __forceinline__
#else
#define LLMC_MLP_INLINE inline
#endif

LLMC_MLP_INLINE void llmc_swish_power_factors(float x, float* b, float* t, float* s) {
    const float delta = x - 1.0f, distance = fabsf(delta);
    // The omitted correction beyond this seam is below FP32 representability.
    // Do not form 8*x in an extreme finite tail. No output cap is applied.
    const float e = distance < 16.0f ? expf(-8.0f * distance) : 0.0f;
    *b = (x >= 1.0f ? x : 1.0f) + 0.125f * log1pf(e);
    *t = delta >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    const float es = expf(-fabsf(x));
    *s = x >= 0.0f ? 1.0f / (1.0f + es) : es / (1.0f + es);
}
LLMC_MLP_INLINE float llmc_swish_power125_k8(float x) {
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    float b, t, s;
    llmc_swish_power_factors(x, &b, &t, &s);
    return (x * s) * sqrtf(sqrtf(b));
}
LLMC_MLP_INLINE float llmc_swish_power125_k8_derivative(float x) {
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    float b, t, s;
    llmc_swish_power_factors(x, &b, &t, &s);
    const float swish = x * s;
    return sqrtf(sqrtf(b)) * (s + swish * (1.0f - s) + 0.25f * (swish / b) * t);
}
LLMC_MLP_INLINE float llmc_swish(float x) {
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    const float e = expf(-fabsf(x));
    const float s = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return x * s;
}
LLMC_MLP_INLINE float llmc_swish_derivative(float x) {
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    const float e = expf(-fabsf(x));
    const float s = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    const float complement = x >= 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
    return s + (x * s) * complement;
}
LLMC_MLP_INLINE float llmc_relu_squared(float x) {
    // In particular, do not let the comparison/fmax silently hide NaN or -Inf.
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    return x > 0.0f ? x * x : 0.0f;
}
LLMC_MLP_INLINE float llmc_relu_squared_derivative(float x) {
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    return x > 0.0f ? 2.0f * x : 0.0f; // zero at both signed-zero inputs
}
LLMC_MLP_INLINE float llmc_swish_power2_k8(float x) {
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    float b, t, s;
    llmc_swish_power_factors(x, &b, &t, &s);
    return (x * s) * b;
}
LLMC_MLP_INLINE float llmc_swish_power2_k8_derivative(float x) {
    if (!(x < INFINITY && x > -INFINITY)) return NAN;
    float b, t, s;
    llmc_swish_power_factors(x, &b, &t, &s);
    const float e = expf(-fabsf(x));
    const float complement = x >= 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
    const float swish = x * s;
    // p=2 removes the need for division by b; the tail derivative is 2*x.
    // Form the sigmoid complement directly so its small positive tail survives.
    return b * (s + swish * complement) + swish * t;
}
LLMC_MLP_INLINE float llmc_custom_mlp_activation(float x, int policy) {
    if (policy == LLMC_MLP_SWISH_POWER125_K8) return llmc_swish_power125_k8(x);
    if (policy == LLMC_MLP_SWISH) return llmc_swish(x);
    if (policy == LLMC_MLP_RELU_SQUARED) return llmc_relu_squared(x);
    if (policy == LLMC_MLP_SWISH_POWER2_K8) return llmc_swish_power2_k8(x);
    return NAN;
}
LLMC_MLP_INLINE float llmc_custom_mlp_activation_derivative(float x, int policy) {
    if (policy == LLMC_MLP_SWISH_POWER125_K8) return llmc_swish_power125_k8_derivative(x);
    if (policy == LLMC_MLP_SWISH) return llmc_swish_derivative(x);
    if (policy == LLMC_MLP_RELU_SQUARED) return llmc_relu_squared_derivative(x);
    if (policy == LLMC_MLP_SWISH_POWER2_K8) return llmc_swish_power2_k8_derivative(x);
    return NAN;
}
#undef LLMC_MLP_INLINE

// A custom header wraps one existing model/state format. Old binaries reject
// the outer version; new binaries validate every fixed formula/numerics word.
constexpr int LLMC_MODEL_VERSION_FP32_MLP_ACTIVATION = 14;
constexpr int LLMC_MODEL_VERSION_BF16_MLP_ACTIVATION = 15;
constexpr int LLMC_OPTIMIZER_STATE_VERSION_MLP_ACTIVATION = 5;
inline void llmc_store_mlp_contract(int* h, int offset, int base_version,
        int policy = LLMC_MLP_SWISH_POWER125_K8) {
    h[offset] = base_version;
    h[offset+1] = policy;
    h[offset+2] = 1; // fixed formula schema
    h[offset+3] = policy == LLMC_MLP_SWISH_POWER125_K8 ? 0x3fa00000 :
        (policy == LLMC_MLP_RELU_SQUARED || policy == LLMC_MLP_SWISH_POWER2_K8) ? 0x40000000 : 0; // power p or ReLU exponent
    h[offset+4] = (policy == LLMC_MLP_SWISH_POWER125_K8 || policy == LLMC_MLP_SWISH_POWER2_K8) ? 0x41000000 : 0; // k=8
    h[offset+5] = (policy == LLMC_MLP_SWISH_POWER125_K8 || policy == LLMC_MLP_SWISH_POWER2_K8) ? 0x3f800000 : 0; // transition=1
    h[offset+6] = policy == LLMC_MLP_RELU_SQUARED ? 0 : 0x3f800000; // Swish beta=1
    h[offset+7] = 1; // FP32 GEMM+bias+activation, one output rounding; FP32 replay/VJP
}
inline bool llmc_valid_mlp_contract(const int* h, int offset) {
    if (!llmc_mlp_activation_is_custom(h[offset+1])) return false;
    int expected[8] = {};
    llmc_store_mlp_contract(expected, 0, h[offset], h[offset+1]);
    return !memcmp(h+offset, expected, sizeof(expected));
}
inline bool llmc_mlp_contract_matches(const int* h, int offset, int policy) {
    return llmc_valid_mlp_contract(h, offset) && h[offset+1] == policy;
}
#endif
