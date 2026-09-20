// Source-pinned llm.c inference benchmark. No optimizer or training allocation.
#define TESTING
#include "train_gpt2.cu"
#include "cached_attention.h"
#include "cached_matmul.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <time.h>

namespace {
// Physical elapsed time must not inherit Linux CLOCK_MONOTONIC frequency
// discipline. A nested CUDA-event audit exposed a measurable WSL clock slew.
struct Clock {
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<Clock, duration>;
    static time_point now() {
        timespec value{};
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0)
            throw std::runtime_error("Cannot read raw monotonic clock");
        return time_point(std::chrono::seconds(value.tv_sec) + duration(value.tv_nsec));
    }
};
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct ModelProfile {
    const char* name;
    const char* descriptor;
    int layers, channels, heads, lexical, reference_max;
    bool rope;
};
const ModelProfile& model_profile(const std::string& name) {
    static const ModelProfile medium{"medium-rope-e4096", "gpt2:rope:d24:t2048:e4096", 24,1024,16,4096,1152,true};
    static const ModelProfile xl{"xl-nope-e8192", "gpt2:nope:d48:t2048:e8192", 48,1600,25,8192,128,false};
    if (name==medium.name) return medium;
    if (name==xl.name) return xl;
    throw std::runtime_error("Unsupported model profile: "+name);
}
GPT2Config profile_config(const ModelProfile& profile) {
    GPT2Config config = {};
    require(gpt2_config_from_descriptor(&config,profile.descriptor),"Invalid model profile descriptor");
    config.vocab_size=50257; config.padded_vocab_size=50304;
    require(config.channels==profile.channels && config.num_layers==profile.layers &&
            config.num_heads==profile.heads && gpt2_lexical_channels(&config)==profile.lexical &&
            config.position_encoding==(profile.rope?LLMC_POSITION_ENCODING_ROPE:LLMC_POSITION_ENCODING_NONE),
            "Stock descriptor geometry differs from the explicit profile");
    return config;
}
void write_profile(std::ostream& out, const ModelProfile& profile) {
    out << "\"model_profile\":\"" << profile.name << "\",\"descriptor\":\"" << profile.descriptor
        << "\",\"layers\":" << profile.layers << ",\"body_width\":" << profile.channels
        << ",\"heads\":" << profile.heads << ",\"head_dim\":64,\"lexical_width\":" << profile.lexical
        << ",\"position_policy\":\"" << (profile.rope?"rope":"none")
        << "\",\"reference_max_prefix\":" << profile.reference_max;
}

struct MemoryAdmission {
    size_t free_bytes = 0, total_bytes = 0;
    uint64_t parameters = 0, kv = 0, scratch = 0, rope = 0, reference = 0;
    // Account for library plans, graph resources and allocation granularity.
    // Existing CUDA context/handles/workspace are already excluded by free_bytes.
    uint64_t reserve = 1ULL << 30;
    bool evaluated = false;
    uint64_t required() const { return parameters + kv + scratch + rope + reference + reserve; }
    bool admitted() const { return required() <= free_bytes; }
    void write(std::ostream& out) const {
        out << "\"memory_admission\":{\"admission_evaluated\":" << (evaluated ? "true" : "false")
            << ",\"admitted\":" << (admitted() ? "true" : "false")
            << ",\"free_bytes_before_weights\":" << free_bytes << ",\"total_bytes\":" << total_bytes
            << ",\"parameter_bytes\":" << parameters << ",\"kv_cache_bytes\":" << kv
            << ",\"decode_scratch_bytes\":" << scratch << ",\"rope_cache_bytes\":" << rope
            << ",\"qualification_reference_bytes\":" << reference << ",\"reserve_bytes\":" << reserve
            << ",\"required_free_bytes\":" << required()
            << ",\"policy\":\"device_allocations_only_no_offload_or_batch_reduction\"}";
    }
};

MemoryAdmission memory_admission(const ModelProfile& profile, int B, int capacity, int steps, bool qualify) {
    const auto config = profile_config(profile);
    size_t elements[NUM_PARAMETER_TENSORS], widths[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(elements, widths, config);
    MemoryAdmission result;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) result.parameters += elements[i] * widths[i];
    const uint64_t rows = B, C = config.channels, E = gpt2_lexical_channels(&config);
    result.kv = 2 * (uint64_t)config.num_layers * rows * capacity * C * sizeof(floatX);
    result.scratch = rows * (E + 16*C + config.padded_vocab_size) * sizeof(floatX)
        + 2*rows*sizeof(float) + rows*(1ULL + steps)*sizeof(int)
        + 3*rows*sizeof(int32_t) + sizeof(uint32_t); // positions, lengths, attention query lengths
    result.rope = (uint64_t)capacity * (config.rope_rotary_dim/2) * sizeof(float2);
    if (qualify) {
        ActivationTensors acts = {};
        TensorSpec specs[NUM_ACTIVATION_TENSORS];
        fill_in_activation_sizes(&acts, specs, B, profile.reference_max, config, 1);
        result.reference = activation_allocation_bytes(specs) + rows*profile.reference_max*sizeof(int);
    }
    return result;
}

template<class T> struct Buffer {
    T* data = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) { cudaCheck(cudaMalloc(&data, n * sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void clear() { cudaCheck(cudaMemsetAsync(data, 0, count * sizeof(T), main_stream)); }
    std::vector<T> read() const {
        std::vector<T> result(count);
        cudaCheck(cudaMemcpyAsync(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, main_stream));
        cudaCheck(cudaStreamSynchronize(main_stream));
        return result;
    }
};

// The benchmark uses the same four-token prefix in every request as FRNA.
// Qualification offsets each request's pattern to expose accidental row sharing.
__global__ void prefix_tokens(int* tokens, const int32_t* positions, int B, bool varied) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int pattern[4] = {464, 3797, 3332, 319};
    if (b < B) tokens[b] = pattern[(positions[b] + (varied ? b : 0)) % 4] + (varied ? b/4 : 0);
}
__global__ void advance_positions(int32_t* positions, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) ++positions[b];
}
__global__ void greedy_tokens(const floatX* logits, int* tokens, int* history,
                             const int32_t* positions, int V, int Vp,
                             int prefix, int steps, uint32_t* status) {
    const int b = blockIdx.x;
    float best = -INFINITY;
    int token = INT_MAX;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = (float)logits[(size_t)b * Vp + i];
        if (!isfinite(v)) atomicOr(status, 2u);
        if (v > best || (v == best && i < token)) { best = v; token = i; }
    }
    __shared__ float values[256];
    __shared__ int ids[256];
    values[threadIdx.x] = best;
    ids[threadIdx.x] = token;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            float other = values[threadIdx.x + stride];
            int id = ids[threadIdx.x + stride];
            if (other > values[threadIdx.x] ||
                (other == values[threadIdx.x] && id < ids[threadIdx.x])) {
                values[threadIdx.x] = other;
                ids[threadIdx.x] = id;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        tokens[b] = ids[0] == INT_MAX ? 0 : ids[0];
        const int step = positions[b] - prefix;
        if (step >= 0 && step < steps) history[(size_t)b * steps + step] = tokens[b];
    }
}

struct Decode {
    GPT2& model;
    const int B, C, E, H, L, V, Vp, capacity, prefix, steps;
    Buffer<floatX> lexical, residual, residual2, norm, qkv, atty, projection, fch, gelu, logits, key, value;
    Buffer<float> mean, rstd;
    Buffer<int> tokens, history;
    Buffer<int32_t> positions, lengths;
    Buffer<uint32_t> status;
    LlmcCachedAttentionPlan* attention = nullptr;
    cudaGraphExec_t prefill_graph = nullptr, decode_graph = nullptr;
    size_t prefill_nodes = 0, decode_nodes = 0;
    std::unique_ptr<LlmcCachedMatmul> down_bridge, up_bridge, readout;
    struct LayerMatmuls {
        std::unique_ptr<LlmcCachedMatmul> qkv, attention, up, down;
    };
    std::vector<LayerMatmuls> matmuls;
    char error[LLMC_CACHED_ATTENTION_ERROR_BYTES] = {};

    Decode(GPT2& m, int batch, int pre, int count, int cache_capacity)
      : model(m), B(batch), C(m.config.channels), E(gpt2_lexical_channels(&m.config)),
        H(m.config.num_heads), L(m.config.num_layers), V(m.config.vocab_size),
        Vp(m.config.padded_vocab_size), capacity(cache_capacity), prefix(pre), steps(count),
        lexical((size_t)B*E), residual((size_t)B*C), residual2((size_t)B*C), norm((size_t)B*C),
        qkv((size_t)B*3*C), atty((size_t)B*C), projection((size_t)B*C),
        fch((size_t)B*4*C), gelu((size_t)B*4*C), logits((size_t)B*Vp),
        key((size_t)L*B*capacity*C), value((size_t)L*B*capacity*C),
        mean(B), rstd(B), tokens(B), history((size_t)B*steps), positions(B), lengths(B), status(1) {
        require(C%8==0 && C/H==64 && E%8==0, "Unsupported decode vector/head geometry");
        require(llmc_cached_attention_create(&attention, B, H, C/H, capacity, main_stream, error), error);
        auto p = model.params;
        down_bridge = std::make_unique<LlmcCachedMatmul>(residual.data, lexical.data, p.lexical_downw, nullptr, B, E, C, main_stream);
        up_bridge = std::make_unique<LlmcCachedMatmul>(lexical.data, norm.data, p.lexical_upw, nullptr, B, C, E, main_stream);
        readout = std::make_unique<LlmcCachedMatmul>(logits.data, lexical.data, p.wte, nullptr, B, E, Vp, main_stream);
        for (size_t l=0; l<(size_t)L; ++l) {
            LayerMatmuls layer;
            layer.qkv = std::make_unique<LlmcCachedMatmul>(qkv.data, norm.data,
                p.qkvw+l*3*C*C, p.qkvb+l*3*C, B, C, 3*C, main_stream);
            layer.attention = std::make_unique<LlmcCachedMatmul>(projection.data, atty.data,
                p.attprojw+l*C*C, p.attprojb+l*C, B, C, C, main_stream);
            layer.up = std::make_unique<LlmcCachedMatmul>(gelu.data, norm.data,
                p.fcw+l*4*C*C, p.fcb+l*4*C, B, C, 4*C, main_stream, fch.data);
            layer.down = std::make_unique<LlmcCachedMatmul>(projection.data, gelu.data,
                p.fcprojw+l*C*4*C, p.fcprojb+l*C, B, 4*C, C, main_stream);
            matmuls.push_back(std::move(layer));
        }
        reset();
    }
    ~Decode() {
        if (prefill_graph) cudaGraphExecDestroy(prefill_graph);
        if (decode_graph) cudaGraphExecDestroy(decode_graph);
        llmc_cached_attention_destroy(attention);
    }
    void reset() {
        positions.clear(); lengths.clear(); status.clear(); history.clear();
        key.clear(); value.clear(); logits.clear();
        cudaCheck(cudaStreamSynchronize(main_stream));
    }
    void head() {
        up_bridge->execute(main_stream);
        readout->execute(main_stream);
    }
    void body(bool with_head) {
        ParameterTensors p = model.params;
        encoder_forward(lexical.data, tokens.data, p.wte, nullptr, B, 1, E, main_stream);
        down_bridge->execute(main_stream);
        layernorm_forward(norm.data, mean.data, rstd.data, residual.data, p.ln1w, p.ln1b, B, 1, C, main_stream);
        for (size_t l = 0; l < (size_t)L; ++l) {
            matmuls[l].qkv->execute(main_stream);
            floatX* k = key.data + l*B*capacity*C;
            floatX* v = value.data + l*B*capacity*C;
            require(llmc_cached_attention_append(attention, qkv.data, k, v,
                model.rope_cache.cos_sin, model.config.rope_rotary_dim, capacity,
                positions.data, lengths.data, status.data, main_stream, error), error);
            require(llmc_cached_attention_execute(attention, qkv.data, k, v, atty.data,
                lengths.data, main_stream, error), error);
            matmuls[l].attention->execute(main_stream);
            fused_residual_forward5(residual2.data, norm.data, mean.data, rstd.data,
                residual.data, projection.data, p.ln2w + l*C, p.ln2b + l*C, B, C, main_stream);
            // Preserve the prior timing control's cuBLASLt GELU_AUX_BIAS epilogue.
            matmuls[l].up->execute(main_stream);
            matmuls[l].down->execute(main_stream);
            floatX* w = l + 1 < (size_t)L ? p.ln1w + (l+1)*C : p.lnfw;
            floatX* b = l + 1 < (size_t)L ? p.ln1b + (l+1)*C : p.lnfb;
            fused_residual_forward5(residual.data, norm.data, mean.data, rstd.data,
                residual2.data, projection.data, w, b, B, C, main_stream);
        }
        if (with_head) head();
        advance_positions<<<1,32,0,main_stream>>>(positions.data, B);
        cudaCheck(cudaGetLastError());
    }
    void prefill_step(bool varied = false) {
        prefix_tokens<<<1,32,0,main_stream>>>(tokens.data, positions.data, B, varied);
        body(false);
    }
    void decode_step() {
        greedy_tokens<<<B,256,0,main_stream>>>(logits.data, tokens.data, history.data,
            positions.data, V, Vp, prefix, steps, status.data);
        body(true);
    }
    template<class F> void capture(cudaGraphExec_t& executable, size_t& nodes, F call) {
        cudaGraph_t graph = nullptr;
        cudaCheck(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeGlobal));
        call();
        cudaCheck(cudaStreamEndCapture(main_stream, &graph));
        cudaCheck(cudaGraphGetNodes(graph, nullptr, &nodes));
        cudaCheck(cudaGraphInstantiate(&executable, graph, 0));
        cudaCheck(cudaGraphDestroy(graph));
    }
    void prepare_graphs(bool varied = false) {
        // Resolve all library plans before stream capture. Warmups are untimed.
        prefill_step(varied); head(); decode_step();
        cudaCheck(cudaStreamSynchronize(main_stream));
        reset();
        capture(prefill_graph, prefill_nodes, [&] { prefill_step(varied); });
        capture(decode_graph, decode_nodes, [&] { decode_step(); });
        for (int i = 0; i < 8; ++i) cudaCheck(cudaGraphLaunch(prefill_graph, main_stream));
        head();
        for (int i = 0; i < 4; ++i) cudaCheck(cudaGraphLaunch(decode_graph, main_stream));
        cudaCheck(cudaStreamSynchronize(main_stream));
        reset();
    }
    void check_position(int expected) {
        require(status.read()[0] == 0, "Nonzero cached decode device status");
        for (int p : positions.read()) require(p == expected, "Incorrect final decode position");
        for (int n : lengths.read()) require(n == expected, "Incorrect final KV length");
    }
};

// Retain the pinned source's full-prefix implementation as an independent
// topology/cache oracle. Only its forward activation allocation is used.
struct FullPrefix {
    GPT2 ref = {};
    explicit FullPrefix(GPT2& model, int B, int T) {
        ref.config = model.config;
        ref.params = model.params;
        ref.params_memory = model.params_memory;
        ref.rope_cache = model.rope_cache;
        ref.batch_size = B; ref.seq_len = T; ref.recompute = 1;
        ref.gelu_fusion = model.gelu_fusion;
        fill_in_activation_sizes(&ref.acts, ref.acts_specs, B, T, ref.config, ref.recompute);
        ref.acts_memory = malloc_and_point_activations(ref.acts_specs);
        cudaCheck(cudaMalloc(&ref.inputs, (size_t)B*T*sizeof(int)));
    }
    ~FullPrefix() { cudaFree(ref.acts_memory); cudaFree(ref.inputs); }
    std::vector<floatX> run(int B, int T) {
        const int pattern[4] = {464,3797,3332,319};
        std::vector<int> input((size_t)B*T);
        for (int b=0; b<B; ++b) for (int t=0; t<T; ++t) input[b*T+t]=pattern[(b+t)%4]+b/4;
        gpt2_forward(&ref, input.data(), B, T);
        std::vector<floatX> output((size_t)B*ref.config.padded_vocab_size);
        for (int b=0; b<B; ++b) cudaCheck(cudaMemcpy(output.data()+(size_t)b*ref.config.padded_vocab_size,
            ref.acts.output+((size_t)b*T+T-1)*ref.config.padded_vocab_size,
            ref.config.padded_vocab_size*sizeof(floatX), cudaMemcpyDeviceToHost));
        return output;
    }
};

void qualify(GPT2& model, const ModelProfile& profile, int B, int capacity, std::ostream& out) {
    Decode d(model,B,16,4,capacity);
    FullPrefix reference(model,B,profile.reference_max);
    d.prepare_graphs(true);
    out << "\"checks\": [";
    bool first=true;
    for (int t=1; t<=profile.reference_max; ++t) {
        cudaCheck(cudaGraphLaunch(d.prefill_graph, main_stream));
        if (t!=8 && t!=32 && t!=128 && t!=1024 && t!=1152) continue;
        d.head();
        auto actual=d.logits.read();
        auto expected=reference.run(B,t);
        double square=0, reference_square=0, max_error=0, max_row_relative=0;
        int matches=0;
        for (int b=0; b<B; ++b) {
            int ai=0, ei=0;
            double row_square=0, row_reference_square=0;
            for (int v=0; v<d.V; ++v) {
                size_t i=(size_t)b*d.Vp+v;
                double a=(float)actual[i], e=(float)expected[i];
                require(std::isfinite(a) && std::isfinite(e), "Nonfinite qualification logit");
                double delta=a-e;
                square+=delta*delta; reference_square+=e*e;
                row_square+=delta*delta; row_reference_square+=e*e;
                max_error=std::max(max_error,std::abs(delta));
                if (a>(float)actual[(size_t)b*d.Vp+ai]) ai=v;
                if (e>(float)expected[(size_t)b*d.Vp+ei]) ei=v;
            }
            matches += ai==ei;
            max_row_relative=std::max(max_row_relative,std::sqrt(row_square/row_reference_square));
        }
        double relative=std::sqrt(square/reference_square);
        if (!first) out << ',';
        first=false;
        out << "{\"prefix\":" << t << ",\"relative_l2\":" << relative
            << ",\"max_row_relative_l2\":" << max_row_relative
            << ",\"max_abs_logit_error\":" << max_error << ",\"greedy_rows_equal\":" << matches << '}';
        out.flush();
        // Different GEMM geometries and inference/training SDPA need BF16
        // tolerance, not bit equality. Bounds are fixed before measurement.
        require(max_row_relative<=0.03 && max_error<=0.125, "Cached/full-prefix BF16 logit tolerance failed");
        d.check_position(t);
    }
    out << "],\"relative_l2_limit\":0.03,\"max_abs_logit_error_limit\":0.125";
    // Capture must preserve eager computation and reset must erase history.
    d.reset();
    for(int i=0;i<16;++i) d.prefill_step(true);
    d.head();
    for(int i=0;i<4;++i) d.decode_step();
    auto eager_logits=d.logits.read(); auto eager_history=d.history.read();
    d.check_position(20);
    for(int repeat=0;repeat<2;++repeat) {
        d.reset();
        for(int i=0;i<16;++i) cudaCheck(cudaGraphLaunch(d.prefill_graph,main_stream));
        d.head();
        for(int i=0;i<4;++i) cudaCheck(cudaGraphLaunch(d.decode_graph,main_stream));
        auto graph_logits=d.logits.read(); auto graph_history=d.history.read();
        d.check_position(20);
        require(std::memcmp(graph_logits.data(),eager_logits.data(),eager_logits.size()*sizeof(floatX))==0,
                "Captured/eager logits differ");
        require(graph_history==eager_history,"Captured/eager greedy history differs");
    }
    out << ",\"capture_eager_bitwise_equal\":true,\"reset_repeats\":2,\"passed\":true";
}

void benchmark(GPT2& model, int B, int prefix, int steps, int capacity, bool timing_audit, std::ostream& out) {
    const auto session_setup=Clock::now();
    Decode d(model,B,prefix,steps,capacity);
    const auto setup=Clock::now();
    d.prepare_graphs();
    const double graph_setup_ms=elapsed_ms(setup);
    const double session_setup_ms=elapsed_ms(session_setup);
    size_t free=0,total=0;
    cudaCheck(cudaMemGetInfo(&free,&total));
    auto begin=Clock::now();
    for(int i=0;i<prefix;++i) cudaCheck(cudaGraphLaunch(d.prefill_graph,main_stream));
    d.head();
    cudaCheck(cudaStreamSynchronize(main_stream));
    double prefill_ms=elapsed_ms(begin);
    d.check_position(prefix);
    begin=Clock::now();
    for(int i=0;i<steps;++i) cudaCheck(cudaGraphLaunch(d.decode_graph,main_stream));
    cudaCheck(cudaStreamSynchronize(main_stream));
    const double decode_ms=elapsed_ms(begin);
    d.check_position(prefix+steps);
    auto history=d.history.read();
    uint64_t checksum=1469598103934665603ULL;
    for(int token:history) { require(token>=0 && token<d.V,"Invalid generated token"); checksum=(checksum^token)*1099511628211ULL; }
    out << "\"prefix_tokens_per_request\":" << prefix << ",\"generated_tokens_per_request\":" << steps
        << ",\"total_generated_tokens\":" << (uint64_t)B*steps
        << ",\"decode_wall_ms\":" << decode_ms << ",\"timer\":\"clock_monotonic_raw_graph_submit_and_completion_no_events\""
        << ",\"aggregate_tokens_per_second\":" << B*(double)steps*1000.0/decode_ms
        << ",\"tokens_per_second_per_request\":" << steps*1000.0/decode_ms
        << ",\"ms_per_decode_step\":" << decode_ms/steps
        << ",\"prefill_wall_ms\":" << prefill_ms << ",\"graph_setup_wall_ms\":" << graph_setup_ms
        << ",\"decode_session_setup_wall_ms\":" << session_setup_ms
        << ",\"prefill_graph_nodes\":" << d.prefill_nodes << ",\"decode_graph_nodes\":" << d.decode_nodes
        << ",\"prepared_gemm_plans\":" << 4*d.L+3
        << ",\"kv_cache_bytes\":" << 2*d.key.count*sizeof(floatX)
        << ",\"attention_workspace_bytes\":" << llmc_cached_attention_workspace_bytes(d.attention)
        << ",\"device_used_bytes_after_setup\":" << total-free
        << ",\"device_total_bytes\":" << total
        << ",\"history_checksum_fnv1a_u32\":\"" << checksum << "\",\"device_status\":0,\"passed\":true";
    if (timing_audit) {
        // Events get a separate, reset run. Their host API cost must not enter
        // the primary timer, whose boundary matches the native FRNA driver.
        d.reset();
        for(int i=0;i<prefix;++i) cudaCheck(cudaGraphLaunch(d.prefill_graph,main_stream));
        d.head(); cudaCheck(cudaStreamSynchronize(main_stream));
        cudaEvent_t start,end;
        cudaCheck(cudaEventCreate(&start)); cudaCheck(cudaEventCreate(&end));
        timespec raw_start,raw_end;
        require(clock_gettime(CLOCK_MONOTONIC_RAW,&raw_start)==0,"Cannot read raw monotonic clock");
        const auto adjusted_start=std::chrono::steady_clock::now();
        auto outer=Clock::now();
        cudaCheck(cudaEventRecord(start,main_stream));
        double start_api_ms=elapsed_ms(outer);
        auto submit_start=Clock::now();
        for(int i=0;i<steps;++i) cudaCheck(cudaGraphLaunch(d.decode_graph,main_stream));
        double submit_ms=elapsed_ms(submit_start);
        auto end_api_start=Clock::now();
        cudaCheck(cudaEventRecord(end,main_stream));
        double end_api_ms=elapsed_ms(end_api_start);
        cudaCheck(cudaStreamSynchronize(main_stream));
        double outer_ms=elapsed_ms(outer);
        const double adjusted_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-adjusted_start).count();
        require(clock_gettime(CLOCK_MONOTONIC_RAW,&raw_end)==0,"Cannot read raw monotonic clock");
        double raw_ms=(raw_end.tv_sec-raw_start.tv_sec)*1000.0+(raw_end.tv_nsec-raw_start.tv_nsec)*1.0e-6;
        float event_ms=0; cudaCheck(cudaEventElapsedTime(&event_ms,start,end));
        d.check_position(prefix+steps);
        // Free-running cold-model choices are diagnostic, not a timing gate.
        // Every pass performs the same fixed number of complete decode steps.
        bool same_history=d.history.read()==history;
        out << ",\"separate_timing_audit\":{\"outer_host_wall_ms\":" << outer_ms
            << ",\"adjusted_steady_clock_wall_ms\":" << adjusted_ms
            << ",\"raw_monotonic_wall_ms\":" << raw_ms << ",\"cuda_event_ms\":" << event_ms
            << ",\"start_event_record_host_ms\":" << start_api_ms
            << ",\"graph_submit_host_ms\":" << submit_ms << ",\"end_event_record_host_ms\":" << end_api_ms
            << ",\"reset_replay_history_equal\":" << (same_history?"true":"false") << '}';
        cudaCheck(cudaEventDestroy(start)); cudaCheck(cudaEventDestroy(end));
    }
}
}

int main(int argc,char** argv) {
    try {
        int B=1,prefix=1024,steps=128,capacity=2048; bool check=false,timing_audit=false,describe=false;
        std::string profile_name="medium-rope-e4096";
        std::string output;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--qualify") { check=true; continue; }
            if(arg=="--timing-audit") { timing_audit=true; continue; }
            if(arg=="--describe-profile") { describe=true; continue; }
            require(i+1<argc,"Expected argument value");
            std::string value=argv[++i];
            if(arg=="--batch") B=std::stoi(value);
            else if(arg=="--prefix") prefix=std::stoi(value);
            else if(arg=="--steps") steps=std::stoi(value);
            else if(arg=="--capacity") capacity=std::stoi(value);
            else if(arg=="--model-profile") profile_name=value;
            else if(arg=="--output") output=value;
            else throw std::runtime_error("Unknown option: "+arg);
        }
        const auto& profile=model_profile(profile_name);
        require(B==1 || B==8,"Only true batches 1 and 8 are admitted");
        require(capacity>=20 && capacity<=INT_MAX-8, "Capacity must admit graph warmup and int32 positions");
        require(check ? capacity>=profile.reference_max : (prefix>=1 && prefix<=capacity && steps>=1 && steps<=capacity-prefix),
                "Invalid prefix/step capacity for model profile");
        if(describe) {
            // CPU-only inspection: no CUDA initialization or device query.
            const auto static_memory=memory_admission(profile,B,capacity,check?4:steps,check);
            std::cout << "{\"schema\":\"worldmodel.llmc_cached_decode_profile.v1\",\"gpu_executions\":0,";
            write_profile(std::cout,profile);
            std::cout << ",\"batch\":" << B << ",\"capacity\":" << capacity
                      << ",\"parameter_bytes\":" << static_memory.parameters << ',';
            static_memory.write(std::cout);
            std::cout << "}\n";
            return 0;
        }
        require(!output.empty() && !std::filesystem::exists(output),"Output must be a fresh JSON path");
        std::ofstream out(output);
        require(out.good(),"Cannot create output JSON");
        out << std::setprecision(12);
        multi_gpu_config=multi_gpu_config_init(1,0,1,nullptr,nullptr,nullptr);
        common_start(false,true);
        GPT2 model={}; gpt2_init_common(&model);
        model.use_master_weights=0; model.gelu_fusion=2;
        auto admission = memory_admission(profile,B,capacity,check?4:steps,check);
        cudaCheck(cudaMemGetInfo(&admission.free_bytes,&admission.total_bytes));
        admission.evaluated=true;
        out << "{\"schema\":\"worldmodel.llmc_cached_decode.v1\",\"mode\":\"" << (check?"qualify":"benchmark")
            << "\",";
        write_profile(out,profile);
        out << ",\"initializer\":\"stock_mt19937_seed42\",\"seed\":42"
            << ",\"weights\":\"cold\",\"precision\":\"BF16\",\"gelu_fusion\":2,\"batch\":" << B
            << ",\"capacity\":" << capacity
            << ",\"hardware\":\"" << deviceProp.name << "\",\"compute_major\":" << deviceProp.major
            << ",\"compute_minor\":" << deviceProp.minor << ',';
        admission.write(out);
        if (!admission.admitted()) {
            out << ",\"passed\":false,\"failure_reason\":\"insufficient_device_memory_before_weights\"}\n";
            out.close();
            fprintf(stderr,"Memory admission rejected: required %llu bytes including reserve, free %zu bytes\n",
                    (unsigned long long)admission.required(),admission.free_bytes);
            common_free(model); multi_gpu_config_free(&multi_gpu_config);
            return 2;
        }
        out << ',';
        out.flush();
        auto init=Clock::now();
        // Keep the original descriptor and RNG traversal independent of runtime
        // capacity. Only the unlearned RoPE table and resident KV cache expand.
        gpt_build_from_descriptor(&model,profile.descriptor);
        require(model.num_parameters_bytes==admission.parameters,"Parameter memory estimate differs from stock allocation");
        if(profile.rope) {
            require(llmc_rope_cache_allocate(&model.rope_cache,capacity,model.config.rope_rotary_dim,
                model.config.rope_theta,model.config.rope_lowest_frequency_plane_is_dc,main_stream),"RoPE allocation failed");
        } else {
            require(model.params.wpe==nullptr && model.config.rope_rotary_dim==0 && model.rope_cache.cos_sin==nullptr,
                    "NoPE must own no learned position weights or RoPE table");
        }
        cudaCheck(cudaStreamSynchronize(main_stream));
        out << "\"num_parameters_allocated\":" << model.num_parameters << ",\"parameter_bytes\":" << model.num_parameters_bytes
            << ",\"initialization_wall_ms\":" << elapsed_ms(init) << ',';
        out.flush();
        if(check) qualify(model,profile,B,capacity,out); else benchmark(model,B,prefix,steps,capacity,timing_audit,out);
        out << "}\n"; out.close();
        llmc_rope_cache_free(&model.rope_cache);
        cudaCheck(cudaFree(model.params_memory));
        common_free(model); multi_gpu_config_free(&multi_gpu_config);
        printf("PASS: %s\n",output.c_str());
        return 0;
    } catch(const std::exception& error) {
        fprintf(stderr,"Cached decode failed: %s\n",error.what());
        return 1;
    }
}
