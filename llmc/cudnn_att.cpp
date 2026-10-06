// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#define NOMINMAX
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include "cudnn_att.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
static_assert(false, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#else // Default to bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // provider-reported, shape/policy dependent
static void* cudnn_workspace = NULL;

static void cuDNNCheck(cudnnStatus_t error, const char *file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

static void checkCudnnFE(const fe::error_object& e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

// Keep historical determinism unless the run explicitly opts into cuDNN's
// recomputing parallel backward. Resolve once: graph policy must not change
// after a cached plan or its workspace has been selected.
static bool deterministic_attention_backward() {
    static const bool deterministic = []() {
        const char* value = std::getenv("LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "1") == 0) return true;
        if (std::strcmp(value, "0") == 0) return false;
        fprintf(stderr, "LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }();
    return deterministic;
}

static void reserve_attention_workspace(
        const std::shared_ptr<fe::graph::Graph>& graph,
        const char* direction, int B, int H, int T, int HS, int deterministic) {
    const size_t required = graph->get_workspace_size();
    printf("cudnn_attention_plan direction=%s B=%d H=%d T=%d HS=%d deterministic=%d workspace_bytes=%zu\n",
           direction, B, H, T, HS, deterministic, required);
    fflush(stdout);
    if (required <= cudnn_workspace_size) return;
    if (cudnn_workspace != nullptr) cudaCheck(cudaFree(cudnn_workspace));
    cudnn_workspace = nullptr;
    cudnn_workspace_size = 0;
    size_t free_bytes = 0, total_bytes = 0;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (required > free_bytes) {
        fprintf(stderr, "cuDNN %s workspace needs %zu bytes; only %zu device bytes free\n",
                direction, required, free_bytes);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaMalloc(&cudnn_workspace, required));
    cudnn_workspace_size = required;
}

enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID
};

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::map<std::tuple<int,int,int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<int,int,int,int,bool>, std::shared_ptr<fe::graph::Graph>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
auto lookup_cache_or_build_graph_fwd(
    int B,
    int H,
    int T,
    int HS,
    int is_inference_only,
    int physical_T = -1) {

    static cache_type_fwd user_maintained_cache_fwd;

    if (physical_T < 0) {
        physical_T = T;
    }
    assert(T > 0 && physical_T >= T);
    auto key = std::make_tuple(B, H, T, physical_T, HS, is_inference_only);

    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({3 * H * HS * physical_T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_uid(K_UID)
                               .set_stride({3 * H * HS * physical_T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_uid(V_UID)
                               .set_stride({3 * H * HS * physical_T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * physical_T, HS, H * HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * physical_T, physical_T, 1, 1})
                               .set_uid(Stats_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));
    reserve_attention_workspace(graph, "forward", B, H, T, HS, -1);

    user_maintained_cache_fwd.insert({key, graph});

    return graph;
}

auto lookup_cache_or_build_graph_bwd(int B, int NH, int T, int HS) {
    static cache_type_bwd user_maintained_cache_bwd;

    auto key = std::make_tuple(B, NH, T, HS, deterministic_attention_backward());

    auto it = user_maintained_cache_bwd.find(key);
    if (it != user_maintained_cache_bwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, NH, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, NH, T, HS})
                            .set_uid(K_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, NH, T, HS})
                            .set_uid(V_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, NH, T, HS})
                            .set_uid(O_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, NH, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));

    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, NH, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({NH * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(deterministic_attention_backward())
#endif
                            .set_causal_mask(true)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    reserve_attention_workspace(graph, "backward", B, NH, T, HS, deterministic_attention_backward() ? 1 : 0);

    user_maintained_cache_bwd.insert({key, graph});
    return graph;
}

void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void attention_forward_cudnn_recent_blackout(
    floatX* out,
    floatX* inp,
    int B,
    int T,
    int NH,
    int C,
    int blackout_width,
    cudaStream_t stream) {
    NVTX_RANGE_FN();
    assert(blackout_width > 0 && blackout_width < T);

    const int HS = C / NH;
    const int visible_T = T - blackout_width;
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Query t may see exactly keys k <= t-blackout_width. Relabeling query
    // rows blackout_width..T-1 and key/value rows 0..T-blackout_width-1 onto
    // 0..visible_T-1 turns this into ordinary top-left causal attention.
    // Physical strides retain T so each batch stays in its original buffer.
    cudaCheck(cudaMemsetAsync(out, 0, (size_t)B * T * C * sizeof(floatX), stream));
    auto graph = lookup_cache_or_build_graph_fwd(
        B, NH, visible_T, HS, true, T);

    void* devPtrQ = inp + (size_t)blackout_width * 3 * C;
    void* devPtrK = inp + C;
    void* devPtrV = inp + 2 * C;
    void* devPtrO = out + (size_t)blackout_width * C;
    float attn_scale_cpu = 1.0f / sqrtf(HS);
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ},
        {K_UID, devPtrK},
        {V_UID, devPtrV},
        {Attn_scale_UID, &attn_scale_cpu},
        {O_UID, devPtrO},
    };
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = qkvr;
    void* devPtrK = (qkvr + NH * HS);
    void* devPtrV = (qkvr + 2 * NH * HS);
    void* devPtrO = o;
    void* devPtrdO = dout;
    void* devPtrStats = stats;
    float attn_scale_cpu = 1.0 / sqrtf(HS);

    void* devPtrdQ = dqkvr;
    void* devPtrdK = (dqkvr + NH * HS);
    void* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

// EOS isolation uses the same cuDNN ragged/padding/causal construction as the
// native FRNA provider. Forward retains packed QKV. SM120 backward requires
// token-contiguous Q/K/V planes; bitwise relayout uses the dead final dqkv
// destination as input staging and one admitted scratch for planar adjoints.
// INT64 absolute offsets select disjoint document spans in both layouts.
struct LlmcEosAttention {
    int max_B, max_T, H, C;
    size_t max_tokens, metadata_capacity;
    LlmcEosHostPlan host;
    int32_t* positions = nullptr;
    int32_t* lengths = nullptr;
    int64_t* qkv_offsets = nullptr;
    int64_t* output_offsets = nullptr;
    float* stats = nullptr;
    floatX* planar_gradients = nullptr;
    bool prepared = false;
    LlmcEosStatsCopy copy_stats = nullptr;
    LlmcEosQkvRelayout relayout_qkv = nullptr;
};

enum { EOS_RQ = 101, EOS_RK, EOS_RV, EOS_RO, EOS_LQ, EOS_LK };

static auto eos_graph(int count, int H, int S, int C, int physical_tokens, bool backward) {
    using Key = std::tuple<int,int,int,int,int,bool>;
    static std::map<Key, std::shared_ptr<fe::graph::Graph>> cache;
    const Key key{count,H,S,C,physical_tokens,backward};
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    if (deterministic_attention_backward()) {
        fprintf(stderr, "EOS ragged SDPA requires explicit LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD=0; no automatic override\n");
        exit(EXIT_FAILURE);
    }
    const int HS = C / H;
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT).set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto offsets = [&](int uid) {
        return graph->tensor(fe::graph::Tensor_attributes().set_uid(uid)
            .set_dim({count+1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT64)
            .set_alignment(alignof(int64_t)));
    };
    auto lengths = [&](int uid) {
        return graph->tensor(fe::graph::Tensor_attributes().set_uid(uid)
            .set_dim({count,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32)
            .set_alignment(alignof(int32_t)));
    };
    auto rq = offsets(EOS_RQ), rk = offsets(EOS_RK), rv = offsets(EOS_RV), ro = offsets(EOS_RO);
    auto lq = lengths(EOS_LQ), lk = lengths(EOS_LK);
    auto data = [&](int uid, int multiplier, auto offset) {
        auto tensor = graph->tensor(fe::graph::Tensor_attributes().set_uid(uid)
            .set_dim({count,H,S,HS}).set_stride({(int64_t)multiplier*C*S,HS,multiplier*C,1}));
        tensor->set_ragged_offset(offset);
        return tensor;
    };
    const int qkv_multiplier=backward?1:3;
    auto q = data(Q_UID,qkv_multiplier,rq), k = data(K_UID,qkv_multiplier,rk), v = data(V_UID,qkv_multiplier,rv);
    auto set_output = [&](auto tensor, int uid, int multiplier, auto offset) {
        tensor->set_output(true).set_uid(uid).set_dim({count,H,S,HS})
            .set_stride({(int64_t)multiplier*C*S,HS,multiplier*C,1});
        tensor->set_ragged_offset(offset);
    };
    if (!backward) {
        auto attr = fe::graph::SDPA_attributes().set_name("eos_ragged_forward")
            .set_is_inference(false).set_attn_scale(1.0f / sqrtf((float)HS))
            .set_causal_mask(true).set_padding_mask(true).set_seq_len_q(lq).set_seq_len_kv(lk);
        auto [out, stats] = graph->sdpa(q,k,v,attr);
        set_output(out,O_UID,1,ro);
        stats->set_output(true).set_uid(Stats_UID).set_data_type(fe::DataType_t::FLOAT)
            .set_dim({count,H,S,1}).set_stride({(int64_t)H*S,S,1,1});
    } else {
        auto out = data(O_UID,1,ro), dout = data(dO_UID,1,ro);
        auto stats = graph->tensor(fe::graph::Tensor_attributes().set_uid(Stats_UID)
            .set_data_type(fe::DataType_t::FLOAT).set_dim({count,H,S,1}).set_stride({(int64_t)H*S,S,1,1}));
        auto attr = fe::graph::SDPA_backward_attributes().set_name("eos_ragged_backward")
            .set_attn_scale(1.0f / sqrtf((float)HS)).set_deterministic_algorithm(false)
            .set_causal_mask(true).set_padding_mask(true).set_seq_len_q(lq).set_seq_len_kv(lk)
            .set_max_total_seq_len_q(physical_tokens).set_max_total_seq_len_kv(physical_tokens);
        auto [dq,dk,dv] = graph->sdpa_backward(q,k,v,out,dout,stats,attr);
        set_output(dq,dQ_UID,1,rq); set_output(dk,dK_UID,1,rk); set_output(dv,dV_UID,1,rv);
    }
    checkCudnnFE(graph->validate());
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    checkCudnnFE(graph->create_execution_plans({fe::HeurMode_t::A}));
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));
    reserve_attention_workspace(graph, backward ? "eos_backward" : "eos_forward", count,H,S,HS,0);
    cache.emplace(key,graph);
    return graph;
}

LlmcEosAttention* llmc_eos_attention_create(int B, int T, int NH, int C, cudaStream_t stream,
        LlmcEosStatsCopy copy_stats,LlmcEosQkvRelayout relayout_qkv) {
    if (!copy_stats || !relayout_qkv || B <= 0 || T < 2 || NH <= 0 || C <= 0 || C % NH ||
        (uint64_t)B*T > INT32_MAX || (uint64_t)B*T > SIZE_MAX/(3ull*C*sizeof(floatX))) {
        fprintf(stderr, "Invalid EOS attention allocation shape\n"); exit(EXIT_FAILURE);
    }
    if (deterministic_attention_backward()) {
        fprintf(stderr, "EOS ragged SDPA requires explicit LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD=0\n");
        exit(EXIT_FAILURE);
    }
    auto* ctx = new LlmcEosAttention;
    ctx->max_B=B; ctx->max_T=T; ctx->H=NH; ctx->C=C;
    ctx->copy_stats=copy_stats;
    ctx->relayout_qkv=relayout_qkv;
    ctx->max_tokens=(size_t)B*T; ctx->metadata_capacity=4*ctx->max_tokens+64;
    const size_t stats_count=(size_t)std::max(T,LLMC_EOS_TOKEN_BUDGET)*NH;
    const size_t planar_bytes=3*ctx->max_tokens*C*sizeof(floatX);
    const size_t metadata_bytes=ctx->max_tokens*sizeof(int32_t)+ctx->metadata_capacity*(sizeof(int32_t)+2*sizeof(int64_t))+
        stats_count*sizeof(float);
    if(planar_bytes>SIZE_MAX-metadata_bytes) {fprintf(stderr,"EOS allocation size overflow\n");exit(EXIT_FAILURE);}
    const size_t bytes=metadata_bytes+planar_bytes;
    size_t free_bytes=0,total_bytes=0;
    cudaCheck(cudaMemGetInfo(&free_bytes,&total_bytes));
    if (bytes > free_bytes) { fprintf(stderr,"EOS attention metadata/stats/planar gradients need %zu bytes; only %zu free\n",bytes,free_bytes); exit(EXIT_FAILURE); }
    cudaCheck(cudaMalloc((void**)&ctx->positions,ctx->max_tokens*sizeof(int32_t)));
    cudaCheck(cudaMalloc((void**)&ctx->lengths,ctx->metadata_capacity*sizeof(int32_t)));
    cudaCheck(cudaMalloc((void**)&ctx->qkv_offsets,ctx->metadata_capacity*sizeof(int64_t)));
    cudaCheck(cudaMalloc((void**)&ctx->output_offsets,ctx->metadata_capacity*sizeof(int64_t)));
    cudaCheck(cudaMalloc((void**)&ctx->stats,stats_count*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&ctx->planar_gradients,planar_bytes));
    cuDNNCheck(cudnnSetStream(cudnn_handle,stream));
    // Pre-admit every selectable count class, including rounded tails. Future
    // EOS occupancy cannot request a plan/workspace outside this envelope.
    int lower=1;
    for (int S=std::min(2,T);; S=S>T/2?T:S*2) {
        const int cap=llmc_eos_group_capacity(S);
        const int max_actual=(int)std::min((size_t)cap,(size_t)B*(T/lower));
        for (int count=1;;count=count>cap/2?cap:count*2) {
            eos_graph(count,NH,S,C,(int)ctx->max_tokens,false);
            eos_graph(count,NH,S,C,(int)ctx->max_tokens,true);
            if(count>=max_actual) break;
        }
        if(S==T) break;
        lower=S+1;
    }
    printf("eos_attention_allocation metadata_stats_bytes=%zu planar_gradient_bytes=%zu total_owned_bytes=%zu group_token_budget=%d max_group_segments=%d workspace_bytes=%zu\n",
        metadata_bytes,planar_bytes,bytes,LLMC_EOS_TOKEN_BUDGET,LLMC_EOS_GROUP_LIMIT,cudnn_workspace_size);
    printf("eos_attention_backward_layout: bitwise_planar_qkv_v1\n");
    return ctx;
}

void llmc_eos_attention_prepare(LlmcEosAttention* ctx,const int* inputs,int B,int T,int eos,cudaStream_t stream) {
    if (!ctx || B<=0 || B>ctx->max_B || T!=ctx->max_T) {
        fprintf(stderr,"EOS attention batch exceeds admitted shape\n"); exit(EXIT_FAILURE);
    }
    // Own the host arrays until all queued copies and their consumers finish.
    cudaCheck(cudaStreamSynchronize(stream));
    ctx->host=llmc_make_eos_plan(inputs,B,T,ctx->C,eos);
    if(ctx->host.lengths.size()>ctx->metadata_capacity) { fprintf(stderr,"EOS metadata bound exceeded\n"); exit(EXIT_FAILURE); }
    cudaCheck(cudaMemcpyAsync(ctx->positions,ctx->host.positions.data(),ctx->host.positions.size()*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
    cudaCheck(cudaMemcpyAsync(ctx->lengths,ctx->host.lengths.data(),ctx->host.lengths.size()*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
    cudaCheck(cudaMemcpyAsync(ctx->qkv_offsets,ctx->host.qkv_offsets.data(),ctx->host.qkv_offsets.size()*sizeof(int64_t),cudaMemcpyHostToDevice,stream));
    cudaCheck(cudaMemcpyAsync(ctx->output_offsets,ctx->host.output_offsets.data(),ctx->host.output_offsets.size()*sizeof(int64_t),cudaMemcpyHostToDevice,stream));
    ctx->prepared=true;
}
const int32_t* llmc_eos_attention_positions(const LlmcEosAttention* ctx) { return ctx ? ctx->positions : nullptr; }

static void eos_require_disjoint(const void* a,size_t a_bytes,const void* b,size_t b_bytes) {
    const uintptr_t ab=(uintptr_t)a,bb=(uintptr_t)b;
    if(!a||!b||ab>UINTPTR_MAX-a_bytes||bb>UINTPTR_MAX-b_bytes||
        !(ab+a_bytes<=bb||bb+b_bytes<=ab)) {
        fprintf(stderr,"EOS backward input staging overlaps an immutable input\n");exit(EXIT_FAILURE);
    }
}

static void eos_execute(LlmcEosAttention* ctx,floatX* out,float* dense_stats,floatX* qkv,
                        floatX* dqkv,floatX* dout,cudaStream_t stream) {
    if(!ctx || !ctx->prepared || !dense_stats) { fprintf(stderr,"EOS attention needs prepared metadata and statistics\n"); exit(EXIT_FAILURE); }
    const bool backward=dqkv!=nullptr;
    const size_t tokens=(size_t)ctx->host.B*ctx->host.T,plane=tokens*ctx->C;
    if(backward) {
        const size_t qkv_bytes=3*plane*sizeof(floatX),output_bytes=plane*sizeof(floatX);
        eos_require_disjoint(dqkv,qkv_bytes,qkv,qkv_bytes);
        eos_require_disjoint(dqkv,qkv_bytes,out,output_bytes);
        eos_require_disjoint(dqkv,qkv_bytes,dout,output_bytes);
        eos_require_disjoint(dqkv,qkv_bytes,dense_stats,tokens*ctx->H*sizeof(float));
        // dqkv has no live gradient until this call returns. Preserve original
        // qkv/O/dO/tape; all groups see the same immutable planar input staging.
        ctx->relayout_qkv(qkv,dqkv,tokens,ctx->C,true,stream);
        cudaCheck(cudaMemsetAsync(ctx->planar_gradients,0,qkv_bytes,stream));
    }
    cuDNNCheck(cudnnSetStream(cudnn_handle,stream));
    for(const auto& group:ctx->host.groups) {
        const size_t begin=group.metadata_begin;
        auto graph=eos_graph(group.count,ctx->H,group.max_sequence,ctx->C,(int)ctx->max_tokens,backward);
        if(backward) ctx->copy_stats(dense_stats,ctx->stats,ctx->lengths+begin,ctx->output_offsets+begin,
            group.count,group.max_sequence,ctx->H,ctx->host.T,ctx->C,true,stream);
        floatX* input=backward?dqkv:qkv;
        const size_t qkv_step=backward?plane:(size_t)ctx->C;
        int64_t* offsets=(backward?ctx->output_offsets:ctx->qkv_offsets)+begin;
        std::unordered_map<int64_t,void*> pack={
            {Q_UID,input},{K_UID,input+qkv_step},{V_UID,input+2*qkv_step},{O_UID,out},{Stats_UID,ctx->stats},
            {EOS_RQ,offsets},{EOS_RK,offsets},{EOS_RV,offsets},
            {EOS_RO,ctx->output_offsets+begin},{EOS_LQ,ctx->lengths+begin},{EOS_LK,ctx->lengths+begin}};
        if(backward) {pack[dO_UID]=dout;pack[dQ_UID]=ctx->planar_gradients;
            pack[dK_UID]=ctx->planar_gradients+plane;pack[dV_UID]=ctx->planar_gradients+2*plane;}
        checkCudnnFE(graph->execute(cudnn_handle,pack,cudnn_workspace));
        cudaCheck(cudaGetLastError());
        if(!backward) ctx->copy_stats(dense_stats,ctx->stats,ctx->lengths+begin,ctx->output_offsets+begin,
            group.count,group.max_sequence,ctx->H,ctx->host.T,ctx->C,false,stream);
    }
    if(backward) ctx->relayout_qkv(ctx->planar_gradients,dqkv,tokens,ctx->C,false,stream);
}
void llmc_eos_attention_forward(LlmcEosAttention* ctx,floatX* out,float* stats,floatX* qkv,cudaStream_t stream) {
    eos_execute(ctx,out,stats,qkv,nullptr,nullptr,stream);
}
void llmc_eos_attention_backward(LlmcEosAttention* ctx,floatX* dqkv,floatX* dout,floatX* qkv,floatX* out,float* stats,cudaStream_t stream) {
    eos_execute(ctx,out,stats,qkv,dqkv,dout,stream);
}
void llmc_eos_attention_destroy(LlmcEosAttention* ctx) {
    if(!ctx) return;
    cudaCheck(cudaFree(ctx->positions));cudaCheck(cudaFree(ctx->lengths));
    cudaCheck(cudaFree(ctx->qkv_offsets));cudaCheck(cudaFree(ctx->output_offsets));cudaCheck(cudaFree(ctx->stats));
    cudaCheck(cudaFree(ctx->planar_gradients));
    delete ctx;
}

void destroy_cudnn() {
    if (cudnn_workspace != NULL) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}
