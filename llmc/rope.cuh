/*
Rotary position embeddings for GPT-style packed QKV tensors.

The active trainer stores Q, K, and V interleaved as [B, T, 3C], with each
projection laid out as [NH, head_dim]. RoPE is applied in-place to adjacent
channel pairs in Q and K only. V and any channels beyond rotary_dim are left
unchanged.

The FP32 cosine/sine table is shared by every layer. The backward operation is
the transpose (and, equivalently, inverse) of the forward rotation.
*/
#ifndef LLMC_ROPE_CUH
#define LLMC_ROPE_CUH

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "cuda_common.h"

struct LlmcRopeCache {
    float2* cos_sin;
    int max_seq_len;
    int rotary_dim;
    float theta;
    cudaStream_t owner_stream;
};

inline void llmc_rope_cache_reset(LlmcRopeCache* cache) {
    if (cache != nullptr) {
        memset(cache, 0, sizeof(*cache));
    }
}

inline bool llmc_rope_validate_config(
        int max_seq_len,
        int channels,
        int num_heads,
        int rotary_dim,
        float theta) {
    if (max_seq_len <= 0 || channels <= 0 || num_heads <= 0) {
        return false;
    }
    if (channels % num_heads != 0) {
        return false;
    }
    const int head_dim = channels / num_heads;
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim) {
        return false;
    }
    return isfinite(theta) && theta > 0.0f;
}

inline bool llmc_rope_checked_product(size_t lhs, size_t rhs, size_t* result) {
    if (result == nullptr || (rhs != 0U && lhs > SIZE_MAX / rhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

__global__ void llmc_rope_build_cache_kernel(
        float2* cos_sin,
        int rotary_dim,
        float log_theta,
        size_t table_elements) {
    const size_t rotary_pairs = (size_t)rotary_dim / 2U;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < table_elements;
         index += (size_t)blockDim.x * gridDim.x) {
        const int position = (int)(index / rotary_pairs);
        const int pair = (int)(index % rotary_pairs);
        const float inverse_frequency =
            expf(-log_theta * (2.0f * (float)pair / (float)rotary_dim));
        float sine;
        float cosine;
        sincosf((float)position * inverse_frequency, &sine, &cosine);
        cos_sin[index] = make_float2(cosine, sine);
    }
}

inline size_t llmc_rope_cache_bytes(const LlmcRopeCache* cache) {
    if (cache == nullptr || cache->max_seq_len <= 0 || cache->rotary_dim <= 0) {
        return 0U;
    }
    size_t elements = 0U;
    if (!llmc_rope_checked_product(
            (size_t)cache->max_seq_len,
            (size_t)cache->rotary_dim / 2U,
            &elements) ||
        elements > SIZE_MAX / sizeof(float2)) {
        return 0U;
    }
    return elements * sizeof(float2);
}

// Allocate and asynchronously populate one table. Repeating the call with an
// identical configuration on the same stream is a no-op; changing the stream
// or configuration requires an explicit free. This single-stream ownership
// makes construction/use ordering explicit without a per-cache CUDA event.
inline bool llmc_rope_cache_allocate(
        LlmcRopeCache* cache,
        int max_seq_len,
        int rotary_dim,
        float theta,
        cudaStream_t stream) {
    if (cache == nullptr || max_seq_len <= 0 || rotary_dim <= 0 ||
        (rotary_dim & 1) != 0 || !isfinite(theta) || theta <= 0.0f) {
        return false;
    }
    if (cache->cos_sin != nullptr) {
        return cache->max_seq_len == max_seq_len &&
               cache->rotary_dim == rotary_dim &&
               cache->theta == theta &&
               cache->owner_stream == stream;
    }

    size_t table_elements = 0U;
    if (!llmc_rope_checked_product(
            (size_t)max_seq_len,
            (size_t)rotary_dim / 2U,
            &table_elements) ||
        table_elements == 0U || table_elements > SIZE_MAX / sizeof(float2)) {
        return false;
    }
    const size_t blocks = (table_elements + 255U) / 256U;
    if (blocks == 0U || blocks > (size_t)INT_MAX) {
        return false;
    }

    float2* allocation = nullptr;
    cudaCheck(cudaMalloc(
        reinterpret_cast<void**>(&allocation), table_elements * sizeof(float2)));
    llmc_rope_build_cache_kernel<<<(unsigned int)blocks, 256, 0, stream>>>(
        allocation, rotary_dim, logf(theta), table_elements);
    cudaCheck(cudaGetLastError());

    cache->cos_sin = allocation;
    cache->max_seq_len = max_seq_len;
    cache->rotary_dim = rotary_dim;
    cache->theta = theta;
    cache->owner_stream = stream;
    return true;
}

inline void llmc_rope_cache_free(LlmcRopeCache* cache) {
    if (cache == nullptr) {
        return;
    }
    if (cache->cos_sin != nullptr) {
        cudaCheck(cudaFree(cache->cos_sin));
    }
    llmc_rope_cache_reset(cache);
}

template <bool TRANSPOSE>
__global__ void llmc_rope_apply_qk_kernel(
        floatX* qkv,
        const float2* cos_sin,
        int sequence_length,
        int channels,
        int num_heads,
        int head_dim,
        int rotary_dim,
        size_t work_items) {
    const size_t rotary_pairs = (size_t)rotary_dim / 2U;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < work_items;
         index += (size_t)blockDim.x * gridDim.x) {
        size_t quotient = index;
        const int pair = (int)(quotient % rotary_pairs);
        quotient /= rotary_pairs;
        const int head = (int)(quotient % (size_t)num_heads);
        quotient /= (size_t)num_heads;
        const int position = (int)(quotient % (size_t)sequence_length);
        const int batch = (int)(quotient / (size_t)sequence_length);

        const size_t token_offset =
            ((size_t)batch * (size_t)sequence_length + (size_t)position) *
            3U * (size_t)channels;
        const size_t head_offset = (size_t)head * head_dim;
        const size_t pair_offset = (size_t)(2 * pair);
        const size_t q_index = token_offset + head_offset + pair_offset;
        const size_t k_index = q_index + (size_t)channels;
        const float2 phase = cos_sin[(size_t)position * rotary_pairs + pair];

        const float q0 = (float)qkv[q_index];
        const float q1 = (float)qkv[q_index + 1U];
        const float k0 = (float)qkv[k_index];
        const float k1 = (float)qkv[k_index + 1U];
        if (TRANSPOSE) {
            qkv[q_index] = (floatX)(q0 * phase.x + q1 * phase.y);
            qkv[q_index + 1U] = (floatX)(-q0 * phase.y + q1 * phase.x);
            qkv[k_index] = (floatX)(k0 * phase.x + k1 * phase.y);
            qkv[k_index + 1U] = (floatX)(-k0 * phase.y + k1 * phase.x);
        } else {
            qkv[q_index] = (floatX)(q0 * phase.x - q1 * phase.y);
            qkv[q_index + 1U] = (floatX)(q0 * phase.y + q1 * phase.x);
            qkv[k_index] = (floatX)(k0 * phase.x - k1 * phase.y);
            qkv[k_index + 1U] = (floatX)(k0 * phase.y + k1 * phase.x);
        }
    }
}

template <bool TRANSPOSE>
inline bool llmc_rope_apply_qk_impl(
        floatX* qkv,
        const LlmcRopeCache* cache,
        int batch_size,
        int sequence_length,
        int channels,
        int num_heads,
        cudaStream_t stream) {
    if (qkv == nullptr || cache == nullptr || cache->cos_sin == nullptr ||
        cache->owner_stream != stream ||
        batch_size <= 0 || sequence_length <= 0 ||
        sequence_length > cache->max_seq_len ||
        !llmc_rope_validate_config(
            cache->max_seq_len,
            channels,
            num_heads,
            cache->rotary_dim,
            cache->theta)) {
        return false;
    }

    size_t work_items = 0U;
    if (!llmc_rope_checked_product(
            (size_t)batch_size, (size_t)sequence_length, &work_items) ||
        !llmc_rope_checked_product(
            work_items, (size_t)num_heads, &work_items) ||
        !llmc_rope_checked_product(
            work_items, (size_t)cache->rotary_dim / 2U, &work_items) ||
        work_items == 0U) {
        return false;
    }
    const size_t blocks = (work_items + 255U) / 256U;
    if (blocks == 0U || blocks > (size_t)INT_MAX) {
        return false;
    }

    const int head_dim = channels / num_heads;
    llmc_rope_apply_qk_kernel<TRANSPOSE><<<(unsigned int)blocks, 256, 0, stream>>>(
        qkv,
        cache->cos_sin,
        sequence_length,
        channels,
        num_heads,
        head_dim,
        cache->rotary_dim,
        work_items);
    cudaCheck(cudaGetLastError());
    return true;
}

inline bool llmc_rope_apply_qk(
        floatX* qkv,
        const LlmcRopeCache* cache,
        int batch_size,
        int sequence_length,
        int channels,
        int num_heads,
        cudaStream_t stream) {
    return llmc_rope_apply_qk_impl<false>(
        qkv, cache, batch_size, sequence_length, channels, num_heads, stream);
}

inline bool llmc_rope_apply_qk_backward(
        floatX* dqkv,
        const LlmcRopeCache* cache,
        int batch_size,
        int sequence_length,
        int channels,
        int num_heads,
        cudaStream_t stream) {
    return llmc_rope_apply_qk_impl<true>(
        dqkv, cache, batch_size, sequence_length, channels, num_heads, stream);
}

#endif // LLMC_ROPE_CUH
