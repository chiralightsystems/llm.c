// Build against the reconstructed, patched llm.c source, not the preserved input.
// No training state, model checkpoint, or cuDNN plan is required by this test.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "llmc/layernorm.cuh"
#include "llmc/matmul.cuh"
#include "cached_matmul.h"

static_assert(PRECISION_MODE == PRECISION_BF16, "decode qualification requires BF16");
cudaDeviceProp deviceProp;

namespace {
constexpr int Width = 1024;
constexpr int StatGuard = 8;
constexpr uint16_t Bf16GuardBits = 0x4b23;
constexpr float FloatGuard = -123456.25f;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
uint16_t bits(floatX value) {
    uint16_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
floatX from_bits(uint16_t value) {
    floatX result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
float rounded(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }

template<class T> struct Buffer {
    T* data = nullptr;
    size_t count;
    explicit Buffer(size_t count) : count(count) {
        checked(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    Buffer(const Buffer&) = delete;
    ~Buffer() { if (data) cudaFree(data); }
    void put(const std::vector<T>& values) {
        require(values.size() == count, "incorrect upload extent");
        checked(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> values(count);
        checked(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

struct Results {
    int norm_cases = 0, matmul_cases = 0;
    int frozen_matmul_cases = 0, captured_matmul_replays = 0;
    float residual_error = 0, norm_error = 0, mean_error = 0, rstd_error = 0;
    float pre_gelu_error = 0, gelu_error = 0;
};

void check_bf16_guards(const std::vector<floatX>& values, size_t begin, size_t count, const char* label) {
    for (size_t index = 0; index < values.size(); ++index)
        if (index < begin || index >= begin + count)
            require(bits(values[index]) == Bf16GuardBits, std::string(label) + " guard overwritten at " + std::to_string(index));
}
void check_float_guards(const std::vector<float>& values, size_t begin, size_t count, const char* label) {
    for (size_t index = 0; index < values.size(); ++index)
        if (index < begin || index >= begin + count)
            require(values[index] == FloatGuard, std::string(label) + " guard overwritten at " + std::to_string(index));
}

void norm_case(int batch, int pattern, cudaStream_t stream, Results& results, int Width = 1024) {
    const size_t elements = static_cast<size_t>(batch) * Width;
    // A whole extra row catches the inherited idx==N bug deterministically.
    // Prefix and suffix stats guards catch the same bug's independent stores.
    const size_t guarded_elements = elements + 2 * Width;
    std::vector<floatX> a(guarded_elements, from_bits(Bf16GuardBits));
    std::vector<floatX> b = a, empty = a;
    std::vector<floatX> weight(Width), bias(Width), expected_residual(elements), expected_norm(elements);
    std::vector<float> expected_mean(batch), expected_rstd(batch);
    for (int col = 0; col < Width; ++col) {
        weight[col] = __float2bfloat16_rn(0.75f + static_cast<float>(col % 9) / 32.0f);
        bias[col] = __float2bfloat16_rn(static_cast<float>(col % 13 - 6) / 64.0f);
    }
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < Width; ++col) {
            const size_t at = static_cast<size_t>(row) * Width + col;
            const int code = (col * 37 + row * 19) % 127 - 63;
            float x, y;
            if (pattern == 0) {
                x = code / 16.0f; y = ((col * 11 + row * 7) % 31 - 15) / 64.0f;
            } else if (pattern == 1) {
                x = 32.0f + code / 8.0f; y = -x + ((col + row) % 9 - 4) / 8.0f;
            } else if (pattern == 2) {
                // Half-ULP tie rounds to 1 before normalization. Keeping the
                // residual in FP32 would incorrectly publish mean=1.00390625.
                x = 1.0f; y = 0.00390625f;
            } else {
                x = 1.0f + (col % 5) / 128.0f; y = ((col + row) % 3 - 1) / 256.0f;
            }
            a[Width + at] = __float2bfloat16_rn(x);
            b[Width + at] = __float2bfloat16_rn(y);
            expected_residual[at] = __float2bfloat16_rn(
                __bfloat162float(a[Width + at]) + __bfloat162float(b[Width + at]));
        }
        // Independent CPU reduction, with FP32 statistics and the required
        // BF16 residual boundary. No GPU warp/reduction implementation copied.
        float mean = 0;
        for (int col = 0; col < Width; ++col)
            mean += __bfloat162float(expected_residual[static_cast<size_t>(row) * Width + col]);
        mean /= Width;
        float variance = 0;
        for (int col = 0; col < Width; ++col) {
            const float centered = __bfloat162float(expected_residual[static_cast<size_t>(row) * Width + col]) - mean;
            variance += centered * centered;
        }
        const float inverse_std = 1.0f / std::sqrt(variance / Width + 1.0e-5f);
        expected_mean[row] = mean; expected_rstd[row] = inverse_std;
        for (int col = 0; col < Width; ++col) {
            const size_t at = static_cast<size_t>(row) * Width + col;
            const float normalized = inverse_std * (__bfloat162float(expected_residual[at]) - mean);
            expected_norm[at] = __float2bfloat16_rn(std::fma(normalized,
                __bfloat162float(weight[col]), __bfloat162float(bias[col])));
        }
    }
    Buffer<floatX> da(guarded_elements), db(guarded_elements), dw(Width), dbias(Width);
    Buffer<floatX> residual(guarded_elements), norm(guarded_elements);
    Buffer<float> mean(batch + 2 * StatGuard), rstd(batch + 2 * StatGuard);
    da.put(a); db.put(b); dw.put(weight); dbias.put(bias); residual.put(empty); norm.put(empty);
    mean.put(std::vector<float>(mean.count, FloatGuard)); rstd.put(std::vector<float>(rstd.count, FloatGuard));
    fused_residual_forward5(residual.data + Width, norm.data + Width,
        mean.data + StatGuard, rstd.data + StatGuard, da.data + Width, db.data + Width,
        dw.data, dbias.data, batch, Width, stream);
    checked(cudaStreamSynchronize(stream));
    const auto actual_residual = residual.get(), actual_norm = norm.get();
    const auto actual_mean = mean.get(), actual_rstd = rstd.get();
    check_bf16_guards(actual_residual, Width, elements, "residual");
    check_bf16_guards(actual_norm, Width, elements, "normalized output");
    check_float_guards(actual_mean, StatGuard, batch, "mean");
    check_float_guards(actual_rstd, StatGuard, batch, "rstd");
    for (size_t at = 0; at < elements; ++at) {
        const float residual_error = std::fabs(__bfloat162float(actual_residual[Width + at]) - __bfloat162float(expected_residual[at]));
        results.residual_error = std::max(results.residual_error, residual_error);
        require(bits(actual_residual[Width + at]) == bits(expected_residual[at]), "BF16 residual rounding mismatch");
        const float observed = __bfloat162float(actual_norm[Width + at]), expected = __bfloat162float(expected_norm[at]);
        const float error = std::fabs(observed - expected);
        results.norm_error = std::max(results.norm_error, error);
        require(std::isfinite(observed) && error <= 0.0005f + 0.008f * std::fabs(expected), "normalized output differs from CPU reference");
    }
    for (int row = 0; row < batch; ++row) {
        const float m = actual_mean[StatGuard + row], s = actual_rstd[StatGuard + row];
        const float me = std::fabs(m - expected_mean[row]), se = std::fabs(s - expected_rstd[row]);
        results.mean_error = std::max(results.mean_error, me);
        results.rstd_error = std::max(results.rstd_error, se);
        require(std::isfinite(m) && me <= 1.0e-6f + 2.0e-6f * std::fabs(expected_mean[row]), "mean differs from CPU reference");
        require(std::isfinite(s) && se <= 2.0e-5f + 2.0e-5f * std::fabs(expected_rstd[row]), "rstd differs from CPU reference");
        if (pattern == 2) require(m == 1.0f, "mean bypassed BF16 residual rounding");
    }
    ++results.norm_cases;
    std::fprintf(stderr, "norm_case B=%d C=%d pattern=%d status=pass completed_norm_cases=%d "
        "max_residual_error=%.9g max_normalized_error=%.9g max_mean_error=%.9g max_rstd_error=%.9g\n",
        batch, Width, pattern, results.norm_cases, results.residual_error, results.norm_error,
        results.mean_error, results.rstd_error);
}

float gelu(float x) {
    const float cube = 0.044715f * x * x * x;
    const float scale = std::sqrt(2.0f / 3.14159265358979323846f);
    return 0.5f * x * (1.0f + std::tanh(scale * (x + cube)));
}

void fused_matmul_case(int batch, cudaStream_t stream, Results& results) {
    constexpr int Outputs = 4 * Width;
    const size_t elements = static_cast<size_t>(batch) * Outputs;
    std::vector<floatX> x(static_cast<size_t>(batch) * Width), w(static_cast<size_t>(Outputs) * Width,
        __float2bfloat16_rn(0.0f)), bias(Outputs);
    for (int row = 0; row < batch; ++row)
        for (int col = 0; col < Width; ++col)
            x[static_cast<size_t>(row) * Width + col] = __float2bfloat16_rn(
                col == 0 ? 1.0078125f : col == 1 ? 1.0f : ((col * 7 + row * 11) % 61 - 30) / 16.0f);
    for (int output = 0; output < Outputs; ++output) {
        w[static_cast<size_t>(output) * Width + output % Width] = __float2bfloat16_rn(
            output % Width == 0 ? 1.0078125f : output % Width == 1 ? 1.0f : 0.75f + (output % 7) / 16.0f);
        bias[output] = __float2bfloat16_rn(output % Width == 0 ? -1.015625f :
            output % Width == 1 ? 0.00390625f : (output % 13 - 6) / 32.0f);
    }
    Buffer<floatX> dx(x.size()), dw(w.size()), dbias(bias.size());
    Buffer<floatX> out(elements + 2 * Outputs), aux(elements + 2 * Outputs);
    dx.put(x); dw.put(w); dbias.put(bias);
    out.put(std::vector<floatX>(out.count, from_bits(Bf16GuardBits)));
    aux.put(std::vector<floatX>(aux.count, from_bits(Bf16GuardBits)));
    // Qualify the observed stock CUDA 13.3 / SM120 algorithms at these shapes.
    // Their GEMM product is BF16-rounded before FP32 bias addition. GELU sees
    // that FP32 sum, while AUX separately publishes its BF16 rounding. This is
    // a pinned implementation boundary, not a universal cuBLASLt guarantee.
    matmul_forward_cublaslt(out.data + Outputs, dx.data, dw.data, dbias.data,
        batch, 1, Width, Outputs, stream, aux.data + Outputs, 2);
    checked(cudaStreamSynchronize(stream));
    const auto actual = out.get(), pre = aux.get();
    check_bf16_guards(actual, Outputs, elements, "GELU output");
    check_bf16_guards(pre, Outputs, elements, "pre-GELU output");
    size_t pre_failures = 0, gelu_failures = 0, witness_failures = 0, printed = 0;
    float case_pre_error = 0, case_gelu_error = 0;
    size_t candidate_pre_failures[4] = {}, candidate_gelu_failures[4] = {}, candidate_gelu_bit_differences[4] = {};
    float candidate_gelu_errors[4] = {};
    for (int row = 0; row < batch; ++row) for (int output = 0; output < Outputs; ++output) {
        const size_t at = static_cast<size_t>(row) * Outputs + output;
        const size_t weight_at = static_cast<size_t>(output) * Width + output % Width;
        const size_t input_at = static_cast<size_t>(row) * Width + output % Width;
        const float weight_value = __bfloat162float(w[weight_at]);
        const float input_value = __bfloat162float(x[input_at]);
        const float bias_value = __bfloat162float(bias[output]);
        const float product = weight_value * input_value;
        const float product_bias = std::fma(weight_value, input_value, bias_value);
        const float staged_bias = rounded(product) + bias_value;
        const float expected_pre = rounded(staged_bias), expected = rounded(gelu(staged_bias));
        const float observed_pre = __bfloat162float(pre[Outputs + at]), observed = __bfloat162float(actual[Outputs + at]);
        const float candidate_inputs[4] = {product_bias, rounded(product_bias), staged_bias, rounded(staged_bias)};
        for (int candidate = 0; candidate < 4; ++candidate) {
            const float candidate_pre = rounded(candidate_inputs[candidate]);
            const float candidate_gelu = rounded(gelu(candidate_inputs[candidate]));
            const float candidate_error = std::fabs(observed - candidate_gelu);
            candidate_pre_failures[candidate] += observed_pre != candidate_pre;
            candidate_gelu_bit_differences[candidate] += observed != candidate_gelu;
            candidate_gelu_failures[candidate] += !std::isfinite(observed) ||
                candidate_error > 0.00005f + 0.008f * std::fabs(candidate_gelu);
            candidate_gelu_errors[candidate] = std::max(candidate_gelu_errors[candidate], candidate_error);
        }
        const float pe = std::fabs(observed_pre - expected_pre), error = std::fabs(observed - expected);
        results.pre_gelu_error = std::max(results.pre_gelu_error, pe);
        results.gelu_error = std::max(results.gelu_error, error);
        case_pre_error = std::max(case_pre_error, pe);
        case_gelu_error = std::max(case_gelu_error, error);
        const bool bad_pre = observed_pre != expected_pre;
        const bool bad_gelu = !std::isfinite(observed) || error > 0.00005f + 0.008f * std::fabs(expected);
        // Two exact witnesses distinguish both publication boundaries:
        // (1) roundBF16(1.0078125^2)-1.015625 is exactly zero;
        // (2) 1+2^-8 publishes AUX=1 but GELU(1+2^-8)=BF16(0.84375),
        //     whereas consuming rounded AUX would yield BF16(0.83984375).
        const bool bad_witness =
            (output % Width == 0 && (observed_pre != 0.0f || observed != 0.0f)) ||
            (output % Width == 1 && (observed_pre != 1.0f || observed != 0.84375f));
        pre_failures += bad_pre; gelu_failures += bad_gelu; witness_failures += bad_witness;
        if ((bad_pre || bad_gelu || bad_witness) && printed++ < 8) {
            // Retain the original oracle and thresholds. These alternatives are
            // diagnostic values, never accepted fallback references.
            std::fprintf(stderr, "fused_gelu_mismatch B=%d row=%d output=%d input_col=%d "
                "weight=%.9g input=%.9g bias=%.9g weight_bits=0x%04x input_bits=0x%04x bias_bits=0x%04x "
                "product_fp32=%.9g product_bf16=%.9g product_plus_bias_fp32=%.9g "
                "observed_pre=%.9g expected_pre=%.9g observed_pre_bits=0x%04x expected_pre_bits=0x%04x "
                "rounded_product_then_bias=%.9g observed_gelu=%.9g expected_gelu=%.9g "
                "gelu_from_rounded_pre=%.9g pre_error=%.9g gelu_error=%.9g "
                "bad_pre=%d bad_gelu=%d bad_witness=%d\n",
                batch, row, output, output % Width, weight_value, input_value, bias_value,
                static_cast<unsigned>(bits(w[weight_at])), static_cast<unsigned>(bits(x[input_at])),
                static_cast<unsigned>(bits(bias[output])), product, rounded(product), product_bias,
                observed_pre, expected_pre, static_cast<unsigned>(bits(pre[Outputs + at])),
                static_cast<unsigned>(bits(__float2bfloat16_rn(expected_pre))),
                rounded(rounded(product) + bias_value), observed, expected, rounded(gelu(expected_pre)),
                pe, error, bad_pre, bad_gelu, bad_witness);
        }
    }
    std::fprintf(stderr, "fused_gelu_case B=%d elements=%zu pre_failures=%zu gelu_failures=%zu witness_failures=%zu "
        "max_pre_gelu_error=%.9g max_gelu_error=%.9g completed_norm_cases=%d "
        "max_residual_error=%.9g max_normalized_error=%.9g max_mean_error=%.9g max_rstd_error=%.9g\n",
        batch, elements, pre_failures, gelu_failures, witness_failures, case_pre_error, case_gelu_error,
        results.norm_cases, results.residual_error, results.norm_error, results.mean_error, results.rstd_error);
    const char* candidate_names[4] = {"fp32_product_bias", "fp32_product_bias_then_bf16_pre",
        "bf16_product_then_fp32_bias", "bf16_product_then_bf16_bias"};
    for (int candidate = 0; candidate < 4; ++candidate)
        std::fprintf(stderr, "fused_gelu_candidate B=%d name=%s pre_failures=%zu gelu_failures=%zu "
            "gelu_bit_differences=%zu max_gelu_error=%.9g diagnostic_only=1\n",
            batch, candidate_names[candidate], candidate_pre_failures[candidate],
            candidate_gelu_failures[candidate], candidate_gelu_bit_differences[candidate], candidate_gelu_errors[candidate]);
    require(pre_failures == 0, "GELU AUX differs from pinned stock BF16-product/FP32-bias oracle");
    require(gelu_failures == 0, "fused GELU differs from CPU oracle");
    require(witness_failures == 0, "fused GELU changed a pinned stock rounding witness");
    // The fixed plan must preserve stock descriptors/math and move heuristic
    // selection outside capture. Poison outputs before each execution so an
    // empty graph or missing publication cannot inherit a passing result.
    LlmcCachedMatmul frozen(out.data + Outputs, dx.data, dw.data, dbias.data,
        batch, Width, Outputs, stream, aux.data + Outputs);
    auto reset_outputs = [&]() {
        out.put(std::vector<floatX>(out.count, from_bits(Bf16GuardBits)));
        aux.put(std::vector<floatX>(aux.count, from_bits(Bf16GuardBits)));
    };
    auto compare_stock = [&]() {
        checked(cudaStreamSynchronize(stream));
        const auto frozen_out = out.get(), frozen_aux = aux.get();
        require(std::memcmp(frozen_out.data(), actual.data(), actual.size() * sizeof(floatX)) == 0,
            "frozen/captured matmul output differs bitwise from stock or changed a guard");
        require(std::memcmp(frozen_aux.data(), pre.data(), pre.size() * sizeof(floatX)) == 0,
            "frozen/captured matmul AUX differs bitwise from stock or changed a guard");
    };
    reset_outputs(); frozen.execute(stream); compare_stock();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    frozen.execute(stream);
    checked(cudaStreamEndCapture(stream, &graph));
    checked(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        reset_outputs();
        checked(cudaGraphLaunch(executable, stream));
        compare_stock();
        ++results.captured_matmul_replays;
    }
    checked(cudaGraphExecDestroy(executable));
    checked(cudaGraphDestroy(graph));
    ++results.frozen_matmul_cases;
    std::fprintf(stderr, "frozen_gelu_matmul B=%d eager_stock_bits=equal captured_stock_bits=equal replays=2 guards=pass\n", batch);
    ++results.matmul_cases;
}
} // namespace

int main(int argc, char** argv) {
    try {
        int selected_batch = 0, norm_width = 1024;
        bool norm_only = false;
        for (int i=1; i<argc; ++i) {
            const std::string option=argv[i];
            if(option=="--norm-only") { norm_only=true; continue; }
            require(i+1<argc,"Expected primitive-test option value");
            const std::string value=argv[++i];
            if(option=="--batch") {
                require(value=="1" || value=="8","Only batches 1 and 8 are supported");
                selected_batch=std::stoi(value);
            } else if(option=="--norm-width") {
                require(value=="1024" || value=="1600","Only norm widths 1024 and 1600 are supported");
                norm_width=std::stoi(value);
            } else throw std::runtime_error("Unknown primitive-test option: "+option);
        }
        require(norm_width==1024 || norm_only,"C1600 qualification is a norm-only test; GEMMs use the XL full-model oracle");
        checked(cudaSetDevice(0)); // CUDA_VISIBLE_DEVICES must isolate the authorized GPU.
        checked(cudaGetDeviceProperties(&deviceProp, 0));
        cudaStream_t stream;
        checked(cudaStreamCreate(&stream));
        cublasCheck(cublasLtCreate(&cublaslt_handle));
        cublas_compute = CUBLAS_COMPUTE_32F;
        checked(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
        Results results;
        for (int batch : {1, 8}) {
            if (selected_batch && selected_batch != batch) continue;
            for (int pattern = 0; pattern < 4; ++pattern) norm_case(batch, pattern, stream, results, norm_width);
            if(!norm_only) fused_matmul_case(batch, stream, results);
        }
        checked(cudaStreamSynchronize(stream));
        checked(cudaFree(cublaslt_workspace)); cublaslt_workspace = nullptr;
        cublasCheck(cublasLtDestroy(cublaslt_handle));
        checked(cudaStreamDestroy(stream));
        std::printf("{\"schema\":\"llmc.decode_primitives.v1\",\"status\":\"passed\","
            "\"batches\":%s,\"body_width\":%d,\"norm_only\":%s,\"norm_cases\":%d,\"fused_gelu_cases\":%d,"
            "\"frozen_matmul_cases\":%d,\"captured_matmul_replays\":%d,"
            "\"max_residual_error\":%.9g,\"max_normalized_error\":%.9g,\"max_mean_error\":%.9g,"
            "\"max_rstd_error\":%.9g,\"max_pre_gelu_error\":%.9g,\"max_gelu_error\":%.9g,"
            "\"gemm_gelu_oracle\":\"BF16 product plus FP32 bias; GELU before AUX BF16 publication\","
            "\"guard_sentinels\":\"passed\"}\n", selected_batch == 1 ? "[1]" : selected_batch == 8 ? "[8]" : "[1,8]",
            norm_width, norm_only?"true":"false", results.norm_cases, results.matmul_cases,
            results.frozen_matmul_cases, results.captured_matmul_replays,
            results.residual_error, results.norm_error, results.mean_error, results.rstd_error,
            results.pre_gelu_error, results.gelu_error);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "decode primitive qualification failed: %s\n", error.what());
        return 1;
    }
}
