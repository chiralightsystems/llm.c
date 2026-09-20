// Build against the isolated, patched llm.c snapshot by adding its root to
// the include path and linking its cuDNN attention object. Run each explicit
// LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD=0/1 policy in a fresh process.
// --host-only checks allocation contracts without initializing CUDA.
#define TESTING
#ifdef LLMC_ATTENTION_DIAGNOSTIC_TRACE
// Opt-in diagnosis only: preserve snapshot sources while synchronizing every
// instrumented trainer function boundary to identify a stalled CUDA kernel.
#include <unistd.h>  // Preserve the trainer's Windows compatibility include order.
#include "llmc/cuda_common.h"
class AttentionDiagnosticRange {
public:
    explicit AttentionDiagnosticRange(const char* name) : name_(name) {
        fprintf(stderr, "CUDA stage begin %s\n", name_);
        fflush(stderr);
    }
    ~AttentionDiagnosticRange() {
        cudaCheck(cudaDeviceSynchronize());
        fprintf(stderr, "CUDA stage complete %s\n", name_);
        fflush(stderr);
    }
private:
    const char* name_;
};
#undef NVTX_RANGE_FN
#define NVTX_RANGE_FN() AttentionDiagnosticRange attention_diagnostic_range(__FUNCTION__)
#endif
#include "train_gpt2.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef ENABLE_CUDNN
#error This test qualifies the cuDNN attention memory path.
#endif

namespace {
int failures = 0;

void check(bool passed, const char* description) {
    if (!passed) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        ++failures;
    }
}

GPT2Config config_for(int C, int E, int L, int vocab, int T) {
    GPT2Config config = {};
    config.max_seq_len = T;
    config.vocab_size = vocab;
    config.padded_vocab_size = vocab;
    config.num_layers = L;
    config.num_heads = C / 64;
    config.channels = C;
    config.lexical_channels = E;
    config.position_encoding = LLMC_POSITION_ENCODING_ROPE;
    config.rope_rotary_dim = 64;
    config.rope_theta = LLMC_ROPE_THETA_DEFAULT;
    gpt2_set_initializer_defaults(&config);
    return config;
}

void check_allocation(int B, int T, const GPT2Config& config,
                      bool require_linear_logits_bound) {
    ActivationTensors activations = {};
    TensorSpec tensors[NUM_ACTIVATION_TENSORS];
    fill_in_activation_sizes(&activations, tensors, B, T, config, 1);
    const size_t bytes = tensors[ACTIVATION_TENSOR_OUTPUT].size *
        sizeof_dtype(tensors[ACTIVATION_TENSOR_OUTPUT].type);
    const size_t C = config.channels;
    const size_t E = gpt2_lexical_channels(&config);
    const size_t BT = static_cast<size_t>(B) * T;
    const size_t linear_bytes = BT * std::max(
        std::max(3 * C, 3 * E), static_cast<size_t>(config.padded_vocab_size)) *
        sizeof(floatX);
    check(bytes >= linear_bytes, "output holds logits and linear scratch");
    const size_t ln_bytes = (32 + 4 * C * deviceProp.multiProcessorCount) *
        sizeof(float);
    check(bytes >= ln_bytes, "output holds hardware-sized FP32 layernorm scratch");
    for (size_t OC : {C, 3 * C, 4 * C}) {
        const size_t block = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
        const size_t gx = (OC + 8 * x128::size - 1) / (8 * x128::size);
        const size_t gy = std::max<size_t>(1,
            static_cast<size_t>(deviceProp.maxThreadsPerMultiProcessor) *
            deviceProp.multiProcessorCount / (block * gx));
        if (gy > 1) {
            check(bytes >= OC * gy * sizeof(float),
                  "output holds FP32 matmul bias reductions");
        }
    }
    int slices[] = {1, config.num_layers};
    check(bytes >= static_cast<size_t>(get_max_num_block_sums(slices, 2)) * sizeof(float),
          "output holds gradient norm reductions");
    const size_t groups = (E + 32 * x128::size - 1) / (32 * x128::size);
    const size_t encoder_extent = BT * groups *
        (sizeof(int4) * sizeof(floatX) + sizeof(int));
    check(bytes >= encoder_extent, "output holds embedding bucket metadata and indices");
    if (require_linear_logits_bound) {
        check(bytes == linear_bytes, "cuDNN long rows do not reserve quadratic non-cuDNN scratch");
    }
    std::printf("allocation B=%d T=%d C=%d E=%zu output_bytes=%zu\n",
                B, T, config.channels, E, bytes);
}

void test_host_allocations() {
    const cudaDeviceProp saved = deviceProp;
    // Synthetic device properties exercise small rows as well as the actual
    // long-context allocation formula. These calls allocate no device memory.
    for (int sms : {1, 132, 188}) {
        deviceProp.multiProcessorCount = sms;
        deviceProp.maxThreadsPerMultiProcessor = 1536;
        GPT2Config medium = config_for(1024, 4096, 24, 50304, 16384);
        check_allocation(12, 8192, medium, true);
        check_allocation(8, 8192, medium, true);
        check_allocation(6, 16384, medium, true);
        check_allocation(4, 16384, medium, true);
        check_allocation(1, 1, medium, false);
        check_allocation(1, 32, config_for(384, 512, 2, 512, 32), false);
    }
    deviceProp = saved;
}

template<typename T> class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceArray() { cudaCheck(cudaFree(ptr_)); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* data() { return ptr_; }
    void upload(const std::vector<T>& values) {
        check(values.size() == count_, "upload extent matches allocation");
        cudaCheck(cudaMemcpy(ptr_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(count_);
        cudaCheck(cudaMemcpy(values.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
private:
    T* ptr_ = nullptr;
    size_t count_;
};

std::vector<double> as_double(const std::vector<floatX>& values) {
    std::vector<double> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = static_cast<float>(values[i]);
    return result;
}

struct Oracle { std::vector<double> output, gradient; };

Oracle causal_attention(const std::vector<double>& qkv,
                        const std::vector<double>& dout,
                        int B, int T, int NH, int HS) {
    const int C = NH * HS;
    Oracle result{std::vector<double>(static_cast<size_t>(B) * T * C, 0.0),
                  std::vector<double>(qkv.size(), 0.0)};
    const double scale = 1.0 / std::sqrt(static_cast<double>(HS));
    auto index = [T, C, HS](int b, int t, int qkv_part, int h, int d) {
        return (static_cast<size_t>(b) * T + t) * 3 * C + qkv_part * C + h * HS + d;
    };
    for (int b = 0; b < B; ++b) for (int h = 0; h < NH; ++h) {
        for (int t = 0; t < T; ++t) {
            std::vector<double> probability(t + 1), dp(t + 1);
            double maximum = -INFINITY;
            for (int s = 0; s <= t; ++s) {
                double score = 0.0;
                for (int d = 0; d < HS; ++d)
                    score += qkv[index(b,t,0,h,d)] * qkv[index(b,s,1,h,d)];
                probability[s] = score * scale;
                maximum = std::max(maximum, probability[s]);
            }
            double denominator = 0.0;
            for (double& value : probability) { value = std::exp(value - maximum); denominator += value; }
            for (double& value : probability) value /= denominator;
            double mean_dp = 0.0;
            for (int s = 0; s <= t; ++s) {
                for (int d = 0; d < HS; ++d) {
                    const size_t oi = (static_cast<size_t>(b) * T + t) * C + h * HS + d;
                    const double v = qkv[index(b,s,2,h,d)];
                    result.output[oi] += probability[s] * v;
                    dp[s] += dout[oi] * v;
                    result.gradient[index(b,s,2,h,d)] += probability[s] * dout[oi];
                }
                mean_dp += probability[s] * dp[s];
            }
            for (int s = 0; s <= t; ++s) {
                const double ds = probability[s] * (dp[s] - mean_dp) * scale;
                for (int d = 0; d < HS; ++d) {
                    result.gradient[index(b,t,0,h,d)] += ds * qkv[index(b,s,1,h,d)];
                    result.gradient[index(b,s,1,h,d)] += ds * qkv[index(b,t,0,h,d)];
                }
            }
        }
    }
    return result;
}

void compare(const std::vector<floatX>& actual, const std::vector<double>& expected,
             double absolute, double relative, const char* label) {
    check(actual.size() == expected.size(), "comparison extents match");
    double max_error = 0.0, error2 = 0.0, reference2 = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double value = static_cast<float>(actual[i]);
        const double error = std::abs(value - expected[i]);
        if (!std::isfinite(value) || error > absolute + relative * std::abs(expected[i])) ++bad;
        max_error = std::max(max_error, error);
        error2 += error * error;
        reference2 += expected[i] * expected[i];
    }
    const double relative_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
    std::printf("%s max_abs=%.9g relative_l2=%.9g bad=%zu\n", label, max_error, relative_l2, bad);
    check(bad == 0 && relative_l2 < 0.03, label);
}

void test_attention_oracle(int T, bool deterministic, int B = 2, int NH = 2) {
    const int HS = 64, C = NH * HS;
    std::printf("attention oracle begin B=%d T=%d NH=%d HS=%d\n", B, T, NH, HS);
    std::fflush(stdout);
    const size_t output_count = static_cast<size_t>(B) * T * C;
    std::vector<floatX> qkv(3 * output_count), dout(output_count);
    for (size_t i = 0; i < qkv.size(); ++i)
        qkv[i] = static_cast<floatX>(0.35f * std::sin(static_cast<float>(i) * 0.071f));
    for (size_t i = 0; i < dout.size(); ++i)
        dout[i] = static_cast<floatX>(0.2f * std::cos(static_cast<float>(i) * 0.113f));
    const auto qkv_reference = as_double(qkv);
    const auto dout_reference = as_double(dout);
    const Oracle reference = causal_attention(qkv_reference, dout_reference, B, T, NH, HS);
    // Check the analytic CPU adjoint independently with centered differences.
    for (size_t element : {size_t(9 * C + 15), size_t(C + 11), size_t(2 * C + 23), qkv.size() - 7}) {
        auto plus = qkv_reference, minus = qkv_reference;
        constexpr double delta = 1e-4;
        plus[element] += delta; minus[element] -= delta;
        const auto op = causal_attention(plus, dout_reference, B, T, NH, HS).output;
        const auto om = causal_attention(minus, dout_reference, B, T, NH, HS).output;
        double difference = 0.0;
        for (size_t i = 0; i < output_count; ++i) difference += (op[i] - om[i]) * dout_reference[i];
        const double finite_difference = difference / (2 * delta);
        check(std::abs(finite_difference - reference.gradient[element]) < 1e-7,
              "CPU analytic attention adjoint agrees with finite differences");
    }
    DeviceArray<floatX> dqkv(qkv.size()), input(qkv.size()), output(output_count), upstream(output_count);
    DeviceArray<float> stats(static_cast<size_t>(B) * NH * T);
    input.upload(qkv); upstream.upload(dout);
    attention_forward_cudnn(output.data(), stats.data(), input.data(), B, T, NH, C, main_stream);
    cudaCheck(cudaStreamSynchronize(main_stream));
    std::printf("attention oracle forward complete B=%d T=%d NH=%d\n", B, T, NH);
    std::fflush(stdout);
    attention_backward_cudnn(dqkv.data(), upstream.data(), input.data(), output.data(), stats.data(),
                            B, T, NH, C, main_stream);
    const auto first_gradient = dqkv.download();
    compare(output.download(), reference.output, 0.003, 0.012, "cuDNN causal forward vs CPU");
    compare(first_gradient, reference.gradient, 0.0005, 0.035, "cuDNN causal backward vs CPU");
    for (float value : stats.download()) check(std::isfinite(value), "cuDNN saved stats are finite");
    attention_backward_cudnn(dqkv.data(), upstream.data(), input.data(), output.data(), stats.data(),
                            B, T, NH, C, main_stream);
    const auto repeated_gradient = dqkv.download();
    compare(repeated_gradient, reference.gradient, 0.0005, 0.035, "repeated cuDNN backward vs CPU");
    if (deterministic)
        check(std::memcmp(first_gradient.data(), repeated_gradient.data(), qkv.size() * sizeof(floatX)) == 0,
              "explicit deterministic backward repeats exactly");

    // Future-only token changes must leave all earlier outputs unchanged.
    const auto original_output = output.download();
    for (int b = 0; b < B; ++b) for (int t = T / 2; t < T; ++t)
        for (int c = 0; c < 3 * C; ++c) {
            const size_t i = (static_cast<size_t>(b) * T + t) * 3 * C + c;
            qkv[i] = static_cast<floatX>(static_cast<float>(qkv[i]) + 0.5f);
        }
    input.upload(qkv);
    attention_forward_cudnn(output.data(), stats.data(), input.data(), B, T, NH, C, main_stream);
    const auto changed_output = output.download();
    for (int b = 0; b < B; ++b) {
        const size_t offset = static_cast<size_t>(b) * T * C;
        check(std::memcmp(original_output.data() + offset, changed_output.data() + offset,
                          static_cast<size_t>(T / 2) * C * sizeof(floatX)) == 0,
              "future-only perturbations cannot affect earlier attention outputs");
    }
    std::printf("attention oracle B=%d T=%d NH=%d HS=%d complete\n", B, T, NH, HS);
}

void test_model_gradients(bool bridge, bool production_vocabulary) {
    // Tiny diagnostic: r=1 retains the existing checkpointed GELU policy.
    // Two layers plus two microbatches exercise scratch reuse and accumulation.
    GPT2 model = {};
    gpt2_init_common(&model);
    // Preserve the small-vocabulary regression: its inactive classifier
    // threads must skip the packed tail without unsigned index promotion.
    // The production case also exercises the non-multiple-of-eight tail.
    model.config = config_for(384, bridge ? 512 : 384, 2,
                              production_vocabulary ? 50304 : 512, 32);
    model.config.vocab_size = production_vocabulary ? 50257 : 512;
    std::printf("tiny model begin bridge=%d V=%d Vp=%d\n", bridge ? 1 : 0,
                model.config.vocab_size, model.config.padded_vocab_size);
    std::fflush(stdout);
    check(gpt2_validate_position_config(&model.config), "tiny model config validates");
    gpt2_allocate_weights(&model);
    std::vector<floatX> parameters(model.num_parameters, static_cast<floatX>(0.0f));
    size_t offset = 0;
    for (int tensor = 0; tensor < NUM_PARAMETER_TENSORS; ++tensor) {
        const bool norm = tensor == 2 || tensor == 8 || tensor == 14;
        const bool weight = tensor == 0 || tensor == 4 || tensor == 6 || tensor == 10 ||
                            tensor == 12 || tensor == 16 || tensor == 17;
        for (size_t i = 0; i < model.param_elements[tensor]; ++i)
            if (norm) parameters[offset + i] = static_cast<floatX>(1.0f);
            else if (weight) parameters[offset + i] = static_cast<floatX>(
                0.015f * std::sin(static_cast<float>(offset + i) * 0.173f));
        offset += model.param_elements[tensor];
    }
    cudaCheck(cudaMemcpy(model.params_memory, parameters.data(), model.num_parameters_bytes, cudaMemcpyHostToDevice));
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);
    gpt2_allocate_state(&model, 1, 32);
    const LlmcSequenceBoundaryPolicy boundary_policy = LLMC_SEQUENCE_BOUNDARY_ROW_RESET;
    const bool mask_final_target = llmc_masks_sequence_final_target(boundary_policy);
    std::vector<int> tokens(32), targets(32);
    for (int micro = 0; micro < 2; ++micro) {
        for (int t = 0; t < 32; ++t) {
            tokens[t] = (t * 7 + micro * 3) % 512;
            targets[t] = (t * 7 + micro * 3 + 7) % 512;
        }
        std::printf("tiny model bridge=%d micro=%d forward begin\n", bridge ? 1 : 0, micro);
        std::fflush(stdout);
        gpt2_forward(&model, tokens.data(), 1, 32);
        std::printf("tiny model bridge=%d micro=%d forward complete; backward begin\n", bridge ? 1 : 0, micro);
        std::fflush(stdout);
        gpt2_backward_and_reduce(&model, tokens.data(), targets.data(), 2, micro, mask_final_target);
        cudaCheck(cudaStreamSynchronize(main_stream));
        std::printf("tiny model bridge=%d micro=%d backward complete\n", bridge ? 1 : 0, micro);
        std::fflush(stdout);
    }
    std::vector<floatX> gradient(model.num_parameters);
    cudaCheck(cudaMemcpy(gradient.data(), model.grads_memory, gradient.size() * sizeof(floatX), cudaMemcpyDeviceToHost));
    check(std::isfinite(model.mean_loss) && model.mean_loss > 0, "tiny model training loss is finite");
    float final_target_loss = -1.0f;
    cudaCheck(cudaMemcpy(&final_target_loss, model.acts.losses + 31, sizeof(float), cudaMemcpyDeviceToHost));
    check(final_target_loss == 0.0f, "row_reset excludes the final target in both microbatches");
    offset = 0;
    for (int tensor = 0; tensor < NUM_PARAMETER_TENSORS; ++tensor) {
        double norm2 = 0.0;
        for (size_t i = 0; i < model.param_elements[tensor]; ++i) {
            const float value = static_cast<float>(gradient[offset + i]);
            check(std::isfinite(value), "tiny model parameter gradient is finite");
            norm2 += static_cast<double>(value) * value;
        }
        if (tensor == 0 || tensor == 4 || tensor == 6 || tensor == 10 || tensor == 12 ||
            (bridge && (tensor == 16 || tensor == 17)))
            check(norm2 > 0, "tiny model required weight gradient is nonzero");
        offset += model.param_elements[tensor];
    }
    const float norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
    check(std::isfinite(norm) && norm > 0, "gradient norm scratch produces a finite positive result");
    std::printf("tiny model bridge=%d V=%d Vp=%d layers=2 microbatches=2 loss=%.8f grad_norm=%.8f\n",
                bridge ? 1 : 0, model.config.vocab_size, model.config.padded_vocab_size,
                model.mean_loss, norm);
    gpt2_free(&model);
    set_zero_configs(&multi_gpu_config, 0, 1);
}
}  // namespace

int main(int argc, char** argv) {
    const bool host_only = argc == 2 && std::strcmp(argv[1], "--host-only") == 0;
    if (argc != 1 && !host_only) {
        std::fprintf(stderr, "usage: test_attention_memory [--host-only]\n");
        return EXIT_FAILURE;
    }
    test_host_allocations();
    if (!host_only && failures == 0) {
        const char* policy = std::getenv("LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD");
        if (policy == nullptr || (std::strcmp(policy, "0") != 0 && std::strcmp(policy, "1") != 0)) {
            std::fprintf(stderr, "Set LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD explicitly to 0 or 1.\n");
            return EXIT_FAILURE;
        }
        std::printf("attention deterministic_backward=%s\n", policy);
        char server_ip[2] = "", filesystem_path[2] = "", init_method[4] = "mpi";
        multi_gpu_config = multi_gpu_config_init(1, 0, 1, server_ip, filesystem_path, init_method);
        set_zero_configs(&multi_gpu_config, 0, 1);
        common_start(false, false);
        test_attention_oracle(64, std::strcmp(policy, "1") == 0);
        test_attention_oracle(128, std::strcmp(policy, "1") == 0);
        test_attention_oracle(32, std::strcmp(policy, "1") == 0, 1, 6);
        test_model_gradients(false, false);
        test_model_gradients(true, false);
        test_model_gradients(false, true);
        test_model_gradients(true, true);
        GPT2 empty = {};
        common_free(empty);
        multi_gpu_config_free(&multi_gpu_config);
    }
    std::printf("attention memory qualification failures=%d host_only=%d\n", failures, host_only ? 1 : 0);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
