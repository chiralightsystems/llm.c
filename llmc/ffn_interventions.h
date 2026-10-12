#ifndef LLMC_FFN_INTERVENTIONS_H
#define LLMC_FFN_INTERVENTIONS_H

#include "mlp_activation.h"
#include <cstddef>

// Opt-in model-owned selectors. Legacy activation values/header bytes stay fixed.
constexpr int LLMC_MLP_SWIGLU = 5;
constexpr int LLMC_MLP_TIED_GUM_RELU_SQUARED = 6;
constexpr int LLMC_TIED_GUM_BLOCK_WIDTH = 32;
inline bool llmc_ffn_gate_scale_valid(float gate_scale) {
    return gate_scale == 1.0f || gate_scale == 2.0f;
}

inline bool llmc_ffn_intervention(int policy) {
    return policy == LLMC_MLP_SWIGLU || policy == LLMC_MLP_TIED_GUM_RELU_SQUARED;
}
inline bool llmc_ffn_geometry_valid(int policy, int rows, int hidden, int block) {
    return rows > 0 && hidden > 0 &&
        (policy == LLMC_MLP_SWIGLU ? block == 0 :
         policy == LLMC_MLP_TIED_GUM_RELU_SQUARED &&
         block == LLMC_TIED_GUM_BLOCK_WIDTH && hidden % block == 0);
}

#ifdef __CUDACC__
#define LLMC_FFN_INLINE __host__ __device__ __forceinline__
#else
#define LLMC_FFN_INLINE inline
#endif

// Both sigmoid branches are formed directly: 1-s loses positive-tail credit.
LLMC_FFN_INLINE void llmc_ffn_sigmoid(float x, float* s, float* complement) {
    if (!(x < INFINITY && x > -INFINITY)) { *s = NAN; *complement = NAN; return; }
    const float e = expf(-fabsf(x)), denom = 1.0f + e;
    *s = x >= 0.0f ? 1.0f / denom : e / denom;
    *complement = x >= 0.0f ? e / denom : 1.0f / denom;
}
LLMC_FFN_INLINE float llmc_ffn_swiglu(float gate, float value) {
    return llmc_swish(gate) * value;
}
LLMC_FFN_INLINE void llmc_ffn_swiglu_vjp(float dz, float gate, float value,
        float* dgate, float* dvalue) {
    *dgate = dz * value * llmc_swish_derivative(gate);
    *dvalue = dz * llmc_swish(gate);
}

// U/G/V are row-major [output,input], one complete bank per layer. The bank
// has no local biases. Outer expansion/contraction retain llm.c bias policy.
// h contains the biased, FP32 expansion. All hidden blocks consume this bank.
template<class Weight>
LLMC_FFN_INLINE void llmc_ffn_gum_channel(const float* h, const Weight* u,
        const Weight* g, const Weight* v, int output, int block,
        float* a, float* sigmoid, float* complement, float* value,
        float gate_scale = 1.0f) {
    float uh = 0.0f, gh = 0.0f, vh = 0.0f;
    for (int k = 0; k < block; ++k) {
        const int i = output * block + k;
        uh += float(u[i]) * h[k];
        gh += float(g[i]) * h[k];
        vh += float(v[i]) * h[k];
    }
    llmc_ffn_sigmoid(gh, sigmoid, complement);
    *value = vh;
    // Keep the legacy multiplication order at scale1.
    *a = uh + (gate_scale == 1.0f ? *sigmoid : gate_scale * *sigmoid) * vh;
}
LLMC_FFN_INLINE void llmc_ffn_gum_local_vjp(float dz, float a,
        float sigmoid, float complement, float value,
        float* du, float* dg, float* dv, float gate_scale = 1.0f) {
    const float da = dz * llmc_relu_squared_derivative(a);
    *du = da;
    *dg = da * value * sigmoid * complement;
    *dv = da * sigmoid;
    if (gate_scale != 1.0f) { *dg *= gate_scale; *dv *= gate_scale; }
}
#undef LLMC_FFN_INLINE

// Host fixtures exercise the same pointwise/channel primitives used by CUDA.
// These do not replace the production trainer or make a GPU qualification claim.
inline void llmc_swiglu_forward_host(float* out, float* pre, const float* bias,
        int rows, int hidden) {
    for (int r = 0; r < rows; ++r) {
        float* row = pre + std::size_t(r) * 2 * hidden;
        for (int j = 0; j < 2 * hidden; ++j) if (bias) row[j] += bias[j];
        for (int j = 0; j < hidden; ++j)
            out[std::size_t(r) * hidden + j] = llmc_ffn_swiglu(row[j], row[hidden+j]);
    }
}
inline void llmc_swiglu_backward_host(float* dpre, const float* dz,
        const float* pre, int rows, int hidden) {
    for (int r = 0; r < rows; ++r) for (int j = 0; j < hidden; ++j) {
        const std::size_t i = std::size_t(r) * 2 * hidden + j;
        llmc_ffn_swiglu_vjp(dz[std::size_t(r)*hidden+j],pre[i],pre[i+hidden],
            dpre+i,dpre+i+hidden);
    }
}
inline void llmc_tiled_gum_forward_host(float* out, float* h, const float* bias,
        const float* u, const float* g, const float* v, int rows, int hidden, int block,
        float gate_scale = 1.0f) {
    const std::size_t count = std::size_t(rows) * hidden;
    for (std::size_t i = 0; i < count; ++i) if (bias) h[i] += bias[i % hidden];
    for (std::size_t base = 0; base < count; base += block)
        for (int o = 0; o < block; ++o) {
            float a,s,c,value;
            llmc_ffn_gum_channel(h+base,u,g,v,o,block,&a,&s,&c,&value,gate_scale);
            out[base+o] = llmc_relu_squared(a);
        }
}
// dlocal=[3,rows,hidden] FP32. Bank gradients ADD the sum over every token and
// hidden block to the existing gradient; no per-block normalization is applied.
inline void llmc_tiled_gum_backward_host(float* dh, float* du, float* dg, float* dv,
        float* dlocal, const float* dz, const float* h, const float* u,
        const float* g, const float* v, int rows, int hidden, int block,
        float gate_scale = 1.0f) {
    const std::size_t count = std::size_t(rows) * hidden;
    for (std::size_t base = 0; base < count; base += block) {
        for (int o = 0; o < block; ++o) {
            float a,s,c,value;
            llmc_ffn_gum_channel(h+base,u,g,v,o,block,&a,&s,&c,&value,gate_scale);
            llmc_ffn_gum_local_vjp(dz[base+o],a,s,c,value,
                dlocal+base+o,dlocal+count+base+o,dlocal+2*count+base+o,gate_scale);
        }
        for (int k = 0; k < block; ++k) {
            float sum = 0.0f;
            for (int o = 0; o < block; ++o) {
                const int w = o*block+k;
                sum += float(u[w])*dlocal[base+o] + float(g[w])*dlocal[count+base+o]
                     + float(v[w])*dlocal[2*count+base+o];
            }
            dh[base+k] = sum;
        }
    }
    for (int o = 0; o < block; ++o) for (int k = 0; k < block; ++k) {
        float su=0.0f,sg=0.0f,sv=0.0f;
        for (std::size_t base = 0; base < count; base += block) {
            const float x = h[base+k];
            su += dlocal[base+o]*x;
            sg += dlocal[count+base+o]*x;
            sv += dlocal[2*count+base+o]*x;
        }
        du[o*block+k] += su; dg[o*block+k] += sg; dv[o*block+k] += sv;
    }
}
inline void llmc_tiled_gum_identity_bank(float* u, float* g, float* v, int block) {
    for (int o=0;o<block;++o) for (int k=0;k<block;++k) {
        const int i=o*block+k; u[i]=o==k?1.0f:0.0f; g[i]=0.0f; v[i]=0.0f;
    }
}
#endif
