#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "cached_attention.h"

#include <cudnn_frontend.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

struct LlmcCachedAttentionPlan {
    int batch = 0, heads = 0, head_dim = 0, capacity = 0;
    cudaStream_t stream = nullptr;
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<fe::graph::Graph> graph;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    int32_t* query_lengths = nullptr;
    float scale = 0.0f;

    ~LlmcCachedAttentionPlan() {
        if (workspace) cudaFree(workspace);
        if (query_lengths) cudaFree(query_lengths);
        if (handle) cudnnDestroy(handle);
    }
};

namespace {
enum : int64_t { Q_UID = 1, K_UID, V_UID, O_UID, SCALE_UID, QLEN_UID, KVLEN_UID };

bool failure(char* error, const char* message) {
    if (error) std::snprintf(error, LLMC_CACHED_ATTENTION_ERROR_BYTES, "%s", message);
    return false;
}
void clear_error(char* error) { if (error) error[0] = '\0'; }
void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void require_cudnn(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
}
void require_fe(const fe::error_object& status, const char* operation) {
    if (!status.is_good()) throw std::runtime_error(std::string(operation) + ": " + status.err_msg);
}
bool valid_plan(const LlmcCachedAttentionPlan* plan, cudaStream_t stream, char* error) {
    if (!plan || plan->stream != stream)
        return failure(error, "missing cached-attention plan or mismatched owner stream");
    return true;
}

__global__ void seed_kernel(
    const __nv_bfloat16* packed, __nv_bfloat16* key, __nv_bfloat16* value,
    int batch, int heads, int dim, int capacity, int sequence_length) {
    const size_t elements = (size_t)batch * heads * sequence_length * dim;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += (size_t)blockDim.x * gridDim.x) {
        size_t quotient = index;
        const int d = (int)(quotient % dim); quotient /= dim;
        const int t = (int)(quotient % sequence_length); quotient /= sequence_length;
        const int h = (int)(quotient % heads);
        const int b = (int)(quotient / heads);
        const size_t channels = (size_t)heads * dim;
        const size_t source = ((size_t)b * sequence_length + t) * 3 * channels + h * dim + d;
        const size_t destination = (((size_t)b * heads + h) * capacity + t) * dim + d;
        key[destination] = packed[source + channels];
        value[destination] = packed[source + 2 * channels];
    }
}

__global__ void append_kernel(
    __nv_bfloat16* qkv, __nv_bfloat16* key, __nv_bfloat16* value,
    const float2* phases, int rotary_dim, int phase_capacity,
    const int32_t* positions, int32_t* valid_lengths, uint32_t* status,
    int heads, int dim, int capacity) {
    const int b = (int)blockIdx.x, h = (int)blockIdx.y;
    const int position = positions[b];
    // All threads in this block must agree before reaching the barrier below.
    __shared__ int reject;
    if (threadIdx.x == 0) {
        reject = atomicAdd(status, 0u) != 0u;
        if (position < 0 || position >= capacity ||
            (rotary_dim > 0 && position >= phase_capacity)) {
            atomicOr(status, LLMC_CACHED_ATTENTION_BAD_POSITION);
            reject = 1;
        }
        if (h == 0) valid_lengths[b] = reject ? 1 : position + 1;
    }
    __syncthreads();
    if (reject) return;

    const size_t channels = (size_t)heads * dim;
    const size_t q_offset = (size_t)b * 3 * channels + h * dim;
    const size_t k_offset = q_offset + channels;
    const int pair = (int)threadIdx.x;
    if (pair < rotary_dim / 2) {
        const size_t offset = (size_t)pair * 2;
        const float2 phase = phases[(size_t)position * (rotary_dim / 2) + pair];
        const float q0 = (float)qkv[q_offset + offset];
        const float q1 = (float)qkv[q_offset + offset + 1];
        const float k0 = (float)qkv[k_offset + offset];
        const float k1 = (float)qkv[k_offset + offset + 1];
        // Keep the stock llmc/rope.cuh arithmetic and BF16 cast frontier.
        qkv[q_offset + offset] = (__nv_bfloat16)(q0 * phase.x - q1 * phase.y);
        qkv[q_offset + offset + 1] = (__nv_bfloat16)(q0 * phase.y + q1 * phase.x);
        qkv[k_offset + offset] = (__nv_bfloat16)(k0 * phase.x - k1 * phase.y);
        qkv[k_offset + offset + 1] = (__nv_bfloat16)(k0 * phase.y + k1 * phase.x);
    }
    __syncthreads();
    const int d = (int)threadIdx.x;
    if (d < dim) {
        const size_t destination = (((size_t)b * heads + h) * capacity + position) * dim + d;
        key[destination] = qkv[k_offset + d];
        value[destination] = qkv[k_offset + channels + d];
    }
}
} // namespace

bool llmc_cached_attention_create(
    LlmcCachedAttentionPlan** result, int batch, int heads, int head_dim,
    int capacity, cudaStream_t stream, char* error) {
    clear_error(error);
    if (!result) return failure(error, "missing cached-attention result pointer");
    *result = nullptr;
    if ((batch != 1 && batch != 8) || heads <= 0 || heads > 65535 ||
        head_dim != 64 || capacity <= 0 ||
        (size_t)capacity > SIZE_MAX / (size_t)batch / (size_t)heads / 64 / sizeof(__nv_bfloat16))
        return failure(error, "cached attention requires B1/B8, positive heads/capacity and head dimension 64");
    try {
        cudaStreamCaptureStatus capture;
        require_cuda(cudaStreamIsCapturing(stream, &capture), "capture status");
        if (capture != cudaStreamCaptureStatusNone)
            return failure(error, "cached-attention plans must be created before graph capture");
        auto plan = std::make_unique<LlmcCachedAttentionPlan>();
        plan->batch = batch; plan->heads = heads; plan->head_dim = head_dim;
        plan->capacity = capacity; plan->stream = stream;
        plan->scale = 1.0f / std::sqrt((float)head_dim);
        require_cudnn(cudnnCreate(&plan->handle), "create cuDNN handle");
        require_cudnn(cudnnSetStream(plan->handle, stream), "set cuDNN stream");
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        const int64_t B = batch, H = heads, D = head_dim, S = capacity, C = H * D;
        auto q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
            .set_uid(Q_UID).set_dim({B,H,1,D}).set_stride({3*C,D,C,1}));
        auto k = graph->tensor(fe::graph::Tensor_attributes().set_name("K_cache")
            .set_uid(K_UID).set_dim({B,H,S,D}).set_stride({H*S*D,S*D,D,1}));
        auto v = graph->tensor(fe::graph::Tensor_attributes().set_name("V_cache")
            .set_uid(V_UID).set_dim({B,H,S,D}).set_stride({H*S*D,S*D,D,1}));
        auto scale = graph->tensor(fe::graph::Tensor_attributes().set_name("scale")
            .set_uid(SCALE_UID).set_dim({1,1,1,1}).set_stride({1,1,1,1})
            .set_data_type(fe::DataType_t::FLOAT).set_is_pass_by_value(true));
        auto qlen = graph->tensor(fe::graph::Tensor_attributes().set_name("query_lengths")
            .set_uid(QLEN_UID).set_dim({B,1,1,1}).set_stride({1,1,1,1})
            .set_data_type(fe::DataType_t::INT32));
        auto kvlen = graph->tensor(fe::graph::Tensor_attributes().set_name("valid_lengths")
            .set_uid(KVLEN_UID).set_dim({B,1,1,1}).set_stride({1,1,1,1})
            .set_data_type(fe::DataType_t::INT32));
        auto attributes = fe::graph::SDPA_attributes().set_name("llmc_cached_decode")
            .set_is_inference(true).set_attn_scale(scale)
            .set_padding_mask(true).set_seq_len_q(qlen).set_seq_len_kv(kvlen);
        // The one query is the newest token: every non-padding key is causal.
        auto outputs = graph->sdpa(q, k, v, attributes);
        outputs[0]->set_output(true).set_uid(O_UID)
            .set_dim({B,H,1,D}).set_stride({C,D,C,1});
        require_fe(graph->validate(), "validate cached SDPA");
        require_fe(graph->build_operation_graph(plan->handle), "build cached SDPA graph");
        require_fe(graph->create_execution_plans({fe::HeurMode_t::A}), "cached SDPA heuristics");
        require_fe(graph->check_support(plan->handle), "cached SDPA support");
        require_fe(graph->build_plans(plan->handle), "build cached SDPA plan");
        const int64_t workspace_bytes = graph->get_workspace_size();
        if (workspace_bytes < 0) throw std::runtime_error("negative cuDNN workspace size");
        plan->workspace_bytes = (size_t)workspace_bytes;
        if (workspace_bytes > 0)
            require_cuda(cudaMalloc(&plan->workspace, (size_t)workspace_bytes), "allocate cuDNN workspace");
        require_cuda(cudaMalloc((void**)&plan->query_lengths, batch * sizeof(int32_t)), "allocate query lengths");
        const std::vector<int32_t> ones(batch, 1);
        require_cuda(cudaMemcpyAsync(plan->query_lengths, ones.data(), batch * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream), "initialize query lengths");
        require_cuda(cudaStreamSynchronize(stream), "finish cached-attention initialization");
        plan->graph = std::move(graph);
        *result = plan.release();
        return true;
    } catch (const std::exception& exception) { return failure(error, exception.what()); }
}

void llmc_cached_attention_destroy(LlmcCachedAttentionPlan* plan) {
    if (plan) cudaStreamSynchronize(plan->stream);
    delete plan;
}

size_t llmc_cached_attention_workspace_bytes(const LlmcCachedAttentionPlan* plan) {
    return plan ? plan->workspace_bytes : 0;
}

bool llmc_cached_attention_seed(
    const LlmcCachedAttentionPlan* plan, const __nv_bfloat16* packed_qkv,
    int sequence_length, __nv_bfloat16* key, __nv_bfloat16* value,
    cudaStream_t stream, char* error) {
    clear_error(error);
    if (!valid_plan(plan, stream, error)) return false;
    if (sequence_length < 0 || sequence_length > plan->capacity ||
        (sequence_length > 0 && (!packed_qkv || !key || !value)))
        return failure(error, "invalid cached-attention seed buffers or length");
    if (sequence_length == 0) return true;
    const size_t elements = (size_t)plan->batch * plan->heads * sequence_length * plan->head_dim;
    const unsigned int blocks = (unsigned int)std::min((elements + 255) / 256, (size_t)65535);
    seed_kernel<<<blocks,256,0,stream>>>(packed_qkv, key, value, plan->batch,
        plan->heads, plan->head_dim, plan->capacity, sequence_length);
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess || failure(error, cudaGetErrorString(status));
}

bool llmc_cached_attention_append(
    const LlmcCachedAttentionPlan* plan, __nv_bfloat16* qkv,
    __nv_bfloat16* key, __nv_bfloat16* value, const float2* phases,
    int rotary_dim, int phase_capacity, const int32_t* positions,
    int32_t* valid_lengths, uint32_t* device_status,
    cudaStream_t stream, char* error) {
    clear_error(error);
    if (!valid_plan(plan, stream, error)) return false;
    if (!qkv || !key || !value || !positions || !valid_lengths || !device_status ||
        rotary_dim < 0 || rotary_dim > plan->head_dim || rotary_dim % 2 != 0 ||
        (rotary_dim > 0 && (!phases || phase_capacity <= 0)))
        return failure(error, "invalid cached-attention append buffers or rotary geometry");
    append_kernel<<<dim3(plan->batch,plan->heads),64,0,stream>>>(qkv, key, value,
        phases, rotary_dim, phase_capacity, positions, valid_lengths, device_status,
        plan->heads, plan->head_dim, plan->capacity);
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess || failure(error, cudaGetErrorString(status));
}

bool llmc_cached_attention_execute(
    LlmcCachedAttentionPlan* plan, const __nv_bfloat16* qkv,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* output, const int32_t* valid_lengths,
    cudaStream_t stream, char* error) {
    clear_error(error);
    if (!valid_plan(plan, stream, error)) return false;
    if (!qkv || !key || !value || !output || !valid_lengths)
        return failure(error, "missing cached-attention execute buffer");
    try {
        std::unordered_map<int64_t,void*> pointers = {
            {Q_UID,const_cast<__nv_bfloat16*>(qkv)},
            {K_UID,const_cast<__nv_bfloat16*>(key)},
            {V_UID,const_cast<__nv_bfloat16*>(value)}, {O_UID,output},
            {SCALE_UID,&plan->scale}, {QLEN_UID,plan->query_lengths},
            {KVLEN_UID,const_cast<int32_t*>(valid_lengths)}};
        require_fe(plan->graph->execute(plan->handle, pointers, plan->workspace), "execute cached SDPA");
        require_cuda(cudaGetLastError(), "cached SDPA launch");
        return true;
    } catch (const std::exception& exception) { return failure(error, exception.what()); }
}
