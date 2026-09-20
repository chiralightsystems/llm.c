// Standalone GPU qualification. Root orchestration selects and serializes GPU.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "cached_attention.h"
#include "llmc/rope.cuh"
#include "llmc/cudnn_att.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

cudaDeviceProp deviceProp;

namespace {
using BF16 = __nv_bfloat16;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check_cuda(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }

template<class T> struct Device {
    T* pointer = nullptr;
    size_t count;
    explicit Device(size_t n) : count(n) { check_cuda(cudaMalloc((void**)&pointer, n * sizeof(T))); }
    ~Device() { if (pointer) cudaFree(pointer); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& data, cudaStream_t stream) {
        require(data.size() == count, "upload size mismatch");
        check_cuda(cudaMemcpyAsync(pointer, data.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
        // Test callers may pass temporary host vectors. Complete the transfer
        // before their lifetime ends; uploads are outside all measured graphs.
        check_cuda(cudaStreamSynchronize(stream));
    }
    std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> data(count);
        check_cuda(cudaMemcpyAsync(data.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaStreamSynchronize(stream));
        return data;
    }
};

bool same_bits(const std::vector<BF16>& a, const std::vector<BF16>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(BF16)) == 0;
}

// Independent double-precision scaled-dot-product/softmax/value reduction.
// Inputs come from the stock full-sequence RoPE operation, never from the new
// append or attention kernels. Only the output is compared in BF16 precision.
std::vector<double> attention_reference(
    const std::vector<BF16>& rotated, const std::vector<int32_t>& positions,
    int B, int H, int D, int T) {
    const int C = H * D;
    std::vector<double> result((size_t)B * C, 0.0);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        const int p = positions[b];
        const size_t qbase = ((size_t)b * T + p) * 3 * C + h * D;
        std::vector<double> scores(p + 1);
        double maximum = -INFINITY;
        for (int t = 0; t <= p; ++t) {
            const size_t kbase = ((size_t)b * T + t) * 3 * C + C + h * D;
            double dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += (double)(float)rotated[qbase + d] * (double)(float)rotated[kbase + d];
            scores[t] = dot / std::sqrt((double)D);
            maximum = std::max(maximum, scores[t]);
        }
        double denominator = 0.0;
        for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (int t = 0; t <= p; ++t) {
            const size_t vbase = ((size_t)b * T + t) * 3 * C + 2 * C + h * D;
            for (int d = 0; d < D; ++d)
                result[(size_t)b * C + h * D + d] +=
                    scores[t] / denominator * (double)(float)rotated[vbase + d];
        }
    }
    return result;
}

double compare_output(
    const std::vector<BF16>& actual, const std::vector<BF16>& full,
    const std::vector<BF16>& rotated, const std::vector<int32_t>& positions,
    int B, int H, int D, int T) {
    const int C = H * D;
    const auto reference = attention_reference(rotated, positions, B, H, D, T);
    double maximum = 0.0;
    for (int b = 0; b < B; ++b) for (int d = 0; d < C; ++d) {
        const size_t i = (size_t)b * C + d;
        const double value = (float)actual[i];
        const double expected = reference[i];
        const double stock = (float)full[((size_t)b * T + positions[b]) * C + d];
        maximum = std::max(maximum, std::fabs(value - expected));
        // One BF16 publication plus provider reduction-order differences.
        const double tolerance = 0.004 + 0.004 * std::fabs(expected);
        if (!std::isfinite(value) || std::fabs(value - expected) > tolerance ||
            std::fabs(value - stock) > tolerance) {
            std::fprintf(stderr, "attention mismatch b=%d d=%d actual=%.9g oracle=%.9g stock=%.9g\n",
                b, d, value, expected, stock);
            throw std::runtime_error("cached attention differs from independent or stock reference");
        }
    }
    return maximum;
}

std::vector<BF16> gather_rows(
    const std::vector<BF16>& source, const std::vector<int32_t>& positions, int T, int C) {
    std::vector<BF16> rows((size_t)positions.size() * 3 * C);
    for (size_t b = 0; b < positions.size(); ++b)
        std::copy_n(source.data() + (b * T + positions[b]) * 3 * C,
            3 * C, rows.data() + b * 3 * C);
    return rows;
}

void run_case(int B, int rotary_dim, int dc, int T, cudaStream_t stream) {
    const int H = 16, D = 64, C = H * D, capacity = 2048;
    char error[LLMC_CACHED_ATTENTION_ERROR_BYTES] = {};
    LlmcCachedAttentionPlan* raw = nullptr;
    require(llmc_cached_attention_create(&raw, B, H, D, capacity, stream, error), error);
    std::unique_ptr<LlmcCachedAttentionPlan,decltype(&llmc_cached_attention_destroy)>
        plan(raw, llmc_cached_attention_destroy);
    LlmcRopeCache phases = {};
    if (rotary_dim > 0)
        require(llmc_rope_cache_allocate(&phases, capacity, rotary_dim, 10000.0f, dc, stream),
            "stock RoPE cache creation failed");

    std::vector<BF16> original((size_t)B * T * 3 * C);
    for (int b = 0; b < B; ++b) for (int t = 0; t < T; ++t)
        for (int kind = 0; kind < 3; ++kind) for (int d = 0; d < C; ++d) {
            // Distinct rows, heads, positions, and Q/K/V. A shared-batch or
            // incorrect-position implementation cannot satisfy this oracle.
            uint32_t bits = (uint32_t)(b + 1) * 747796405u + (uint32_t)t * 2891336453u
                + (uint32_t)d * 277803737u + (uint32_t)kind * 1181783497u;
            bits ^= bits >> 16; bits *= 2246822519u; bits ^= bits >> 13;
            const float value = ((int)(bits % 1021u) - 510) / 512.0f;
            original[((size_t)b * T + t) * 3 * C + kind * C + d] = (BF16)value;
        }
    Device<BF16> packed(original.size()), full((size_t)B * T * C);
    packed.upload(original, stream);
    if (rotary_dim > 0)
        require(llmc_rope_apply_qk(packed.pointer, &phases, B, T, C, H, stream),
            "stock full-sequence RoPE failed");
    const auto rotated = packed.download(stream);
    attention_forward_cudnn(full.pointer, nullptr, packed.pointer, B, T, H, C, stream);
    const auto stock_output = full.download(stream);

    Device<BF16> keys((size_t)B * H * capacity * D), values(keys.count), qkv((size_t)B * 3 * C), output((size_t)B * C);
    std::vector<BF16> sentinel(keys.count, (BF16)9.0f);
    keys.upload(sentinel, stream); values.upload(sentinel, stream);
    require(llmc_cached_attention_seed(plan.get(), packed.pointer, T, keys.pointer, values.pointer, stream, error), error);
    auto seeded_keys = keys.download(stream), seeded_values = values.download(stream);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h)
        for (int t = 0; t < capacity; ++t) for (int d = 0; d < D; ++d) {
            const size_t i = (((size_t)b * H + h) * capacity + t) * D + d;
            const size_t source = ((size_t)b * T + t) * 3 * C + h * D + d;
            const BF16 k = t < T ? rotated[source + C] : (BF16)9.0f;
            const BF16 v = t < T ? rotated[source + 2 * C] : (BF16)9.0f;
            require(std::memcmp(&k, &seeded_keys[i], sizeof(BF16)) == 0 &&
                std::memcmp(&v, &seeded_values[i], sizeof(BF16)) == 0,
                "seed cache layout or suffix preservation failed");
        }

    std::vector<int32_t> positions(B, T - 2);
    if (B == 8) {
        const int candidates[8] = {0,1,2,15,255,1023,1024,T-2};
        for (int b = 0; b < B; ++b) positions[b] = std::min(candidates[b], T - 2);
    }
    Device<int32_t> device_positions(B), lengths(B);
    Device<uint32_t> status(1);
    status.upload(std::vector<uint32_t>{0}, stream);
    device_positions.upload(positions, stream);
    qkv.upload(gather_rows(original, positions, T, C), stream);
    auto append = [&]() {
        require(llmc_cached_attention_append(plan.get(), qkv.pointer, keys.pointer, values.pointer,
            phases.cos_sin, rotary_dim, capacity, device_positions.pointer, lengths.pointer,
            status.pointer, stream, error), error);
    };
    auto execute = [&]() {
        require(llmc_cached_attention_execute(plan.get(), qkv.pointer, keys.pointer, values.pointer,
            output.pointer, lengths.pointer, stream, error), error);
    };
    append(); execute();
    const auto first = output.download(stream);
    require(status.download(stream)[0] == 0, "valid append set device error status");
    require(same_bits(qkv.download(stream), gather_rows(rotated, positions, T, C)),
        "decode RoPE differs bitwise from stock RoPE or changed V");
    const auto actual_lengths = lengths.download(stream);
    for (int b = 0; b < B; ++b)
        require(actual_lengths[b] == positions[b] + 1, "valid length was not position+1");
    double max_error = compare_output(first, stock_output, rotated, positions, B, H, D, T);

    // Poison every key/value after each independent valid length. The cuDNN
    // padding mask must exclude these slots, even inside the seeded prefix.
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h)
        for (int t = positions[b] + 1; t < capacity; ++t) for (int d = 0; d < D; ++d) {
            const size_t i = (((size_t)b * H + h) * capacity + t) * D + d;
            seeded_keys[i] = (BF16)((d & 1) ? 32.0f : -32.0f);
            seeded_values[i] = (BF16)((d & 1) ? -17.0f : 19.0f);
        }
    keys.upload(seeded_keys, stream); values.upload(seeded_values, stream);
    execute();
    require(same_bits(first, output.download(stream)), "future cache slots affected attention output");

    // Capture once, then change device positions and QKV between replays.
    // Advancing by one replaces exactly the formerly poisoned next slot.
    check_cuda(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    append(); execute();
    check_cuda(cudaStreamEndCapture(stream, &graph));
    check_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        if (replay) for (int& position : positions) ++position;
        device_positions.upload(positions, stream);
        qkv.upload(gather_rows(original, positions, T, C), stream);
        check_cuda(cudaGraphLaunch(executable, stream));
        const auto actual = output.download(stream);
        max_error = std::max(max_error, compare_output(actual, stock_output, rotated, positions, B, H, D, T));
        require(same_bits(qkv.download(stream), gather_rows(rotated, positions, T, C)),
            "captured RoPE ignored updated absolute positions");
    }
    check_cuda(cudaGraphExecDestroy(executable));
    check_cuda(cudaGraphDestroy(graph));

    // Bad positions and a sticky error cannot publish an affected cache row.
    const auto before_keys = keys.download(stream), before_values = values.download(stream);
    const auto before_qkv = qkv.download(stream);
    for (int invalid_position : {-1, capacity}) {
        positions[0] = invalid_position;
        device_positions.upload(positions, stream);
        status.upload(std::vector<uint32_t>{0}, stream);
        append();
        require((status.download(stream)[0] & LLMC_CACHED_ATTENTION_BAD_POSITION) != 0,
            "invalid position was not reported");
        const auto after_keys = keys.download(stream), after_values = values.download(stream), after_qkv = qkv.download(stream);
        require(std::memcmp(before_keys.data(), after_keys.data(), (size_t)H * capacity * D * sizeof(BF16)) == 0 &&
            std::memcmp(before_values.data(), after_values.data(), (size_t)H * capacity * D * sizeof(BF16)) == 0 &&
            std::memcmp(before_qkv.data(), after_qkv.data(), 3 * C * sizeof(BF16)) == 0,
            "invalid position modified its QKV or cache row");
    }
    const auto sticky_keys = keys.download(stream), sticky_values = values.download(stream), sticky_qkv = qkv.download(stream);
    std::fill(positions.begin(), positions.end(), 0);
    device_positions.upload(positions, stream);
    status.upload(std::vector<uint32_t>{0x100u}, stream);
    append();
    require(status.download(stream)[0] == 0x100u && same_bits(sticky_keys, keys.download(stream)) &&
        same_bits(sticky_values, values.download(stream)) && same_bits(sticky_qkv, qkv.download(stream)),
        "sticky device error allowed publication");

    std::printf("cached_attention_case B=%d H=%d D=%d capacity=%d reference_T=%d rotary=%d dc=%d "
        "max_abs_oracle_error=%.9g workspace_bytes=%zu graph_replays=2 status=pass\n",
        B,H,D,capacity,T,rotary_dim,dc,max_error,llmc_cached_attention_workspace_bytes(plan.get()));
    if (rotary_dim > 0) llmc_rope_cache_free(&phases);
}
} // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        int device = 0;
        check_cuda(cudaGetDevice(&device));
        check_cuda(cudaGetDeviceProperties(&deviceProp, device));
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        create_cudnn();
        char error[LLMC_CACHED_ATTENTION_ERROR_BYTES] = {};
        LlmcCachedAttentionPlan* rejected = nullptr;
        require(!llmc_cached_attention_create(&rejected,2,16,64,2048,stream,error) && !rejected,
            "unsupported batch admitted");
        require(!llmc_cached_attention_create(&rejected,1,16,32,2048,stream,error) && !rejected,
            "unsupported head dimension admitted");
        require(!llmc_cached_attention_create(&rejected,1,16,64,0,stream,error) && !rejected,
            "zero capacity admitted");
        for (int batch : {1,8}) {
            run_case(batch,64,0,1153,stream);
            run_case(batch,64,1,1153,stream);
            run_case(batch,64,0,2048,stream);
            run_case(batch,32,0,17,stream);
            run_case(batch,0,0,17,stream);
        }
        destroy_cudnn();
        check_cuda(cudaStreamDestroy(stream));
        std::printf("cached_attention_selftest cases=10 status=pass\n");
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "cached_attention_selftest status=fail error=%s\n", exception.what());
        if (stream) cudaStreamDestroy(stream);
        return 1;
    }
}
