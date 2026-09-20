#ifndef LLMC_CACHED_ATTENTION_H
#define LLMC_CACHED_ATTENTION_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// Frozen BF16 decode only. A plan may be shared by layers on its owner stream;
// each layer supplies independent K/V storage. No allocations occur at execute.
struct LlmcCachedAttentionPlan;
constexpr size_t LLMC_CACHED_ATTENTION_ERROR_BYTES = 512;
constexpr uint32_t LLMC_CACHED_ATTENTION_BAD_POSITION = 1u;

bool llmc_cached_attention_create(
    LlmcCachedAttentionPlan** result, int batch, int heads, int head_dim,
    int capacity, cudaStream_t stream, char* error);
void llmc_cached_attention_destroy(LlmcCachedAttentionPlan* plan);
size_t llmc_cached_attention_workspace_bytes(const LlmcCachedAttentionPlan* plan);

// packed_qkv is already rotated, [B,T,3,H,D]. Seed K/V [B,H,capacity,D].
// Prefix positions [0,T) are copied, and the remaining cache is untouched.
bool llmc_cached_attention_seed(
    const LlmcCachedAttentionPlan* plan, const __nv_bfloat16* packed_qkv,
    int sequence_length, __nv_bfloat16* key, __nv_bfloat16* value,
    cudaStream_t stream, char* error);

// qkv [B,3,H,D] is rotated in place using the stock FP32 phase table
// [phase_capacity, rotary_dim/2], then its K/V are appended at positions[B].
// This preserves the stock BF16 Q/K publication and leaves V unchanged.
// rotary_dim=0 selects no rotation. valid_lengths[B] becomes position+1.
// Device status is sticky and must start at zero. Invalid positions preserve
// the affected row's QKV and cache; valid_lengths is set to 1 for safe SDPA.
// Any nonzero status invalidates the job, including its attention output.
bool llmc_cached_attention_append(
    const LlmcCachedAttentionPlan* plan, __nv_bfloat16* qkv,
    __nv_bfloat16* key, __nv_bfloat16* value, const float2* phases,
    int rotary_dim, int phase_capacity, const int32_t* positions,
    int32_t* valid_lengths, uint32_t* device_status,
    cudaStream_t stream, char* error);

// Q is viewed directly from packed [B,3,H,D]. O is [B,H,D]. Each valid length
// must be in [1,capacity]; all valid keys are visible to this final query.
// The plan uses inference SDPA, FP32 compute and BF16 output, with device
// padding lengths and no top-left causal mask.
bool llmc_cached_attention_execute(
    LlmcCachedAttentionPlan* plan, const __nv_bfloat16* qkv,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, const int32_t* valid_lengths,
    cudaStream_t stream, char* error);

#endif
