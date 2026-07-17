#define TESTING
#include "train_gpt2.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int test_failures = 0;

#define TEST_CHECK(condition, message)                                           \
    do {                                                                         \
        if (!(condition)) {                                                       \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
            test_failures++;                                                      \
        }                                                                         \
    } while (0)

static bool all_finite(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static float max_abs_difference(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {
    TEST_CHECK(lhs.size() == rhs.size(), "vector sizes match");
    float maximum = 0.0f;
    const size_t count = std::min(lhs.size(), rhs.size());
    for (size_t index = 0; index < count; ++index) {
        maximum = std::max(maximum, std::fabs(lhs[index] - rhs[index]));
    }
    return maximum;
}

static std::vector<float> matrix_multiply(
    const std::vector<float>& lhs,
    bool transpose_lhs,
    const std::vector<float>& rhs,
    bool transpose_rhs,
    int width) {
    std::vector<float> output(static_cast<size_t>(width) * width, 0.0f);
    auto load = [width](
                    const std::vector<float>& matrix,
                    bool transpose,
                    int row,
                    int column) {
        return transpose
            ? matrix[static_cast<size_t>(column) * width + row]
            : matrix[static_cast<size_t>(row) * width + column];
    };
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            double sum = 0.0;
            for (int inner = 0; inner < width; ++inner) {
                sum += static_cast<double>(
                           load(lhs, transpose_lhs, row, inner)) *
                       static_cast<double>(
                           load(rhs, transpose_rhs, inner, column));
            }
            output[static_cast<size_t>(row) * width + column] =
                static_cast<float>(sum);
        }
    }
    return output;
}

static void apply_polynomial_reference(
    std::vector<float>* matrix,
    int width,
    uint32_t stage_count,
    const LlmcNormuonPolynomialStep* schedule) {
    for (uint32_t stage = 0; stage < stage_count; ++stage) {
        std::vector<float> gram =
            matrix_multiply(*matrix, true, *matrix, false, width);
        std::vector<float> gram_squared =
            matrix_multiply(gram, false, gram, false, width);
        std::vector<float> projected(
            static_cast<size_t>(width) * width, 0.0f);
        for (size_t index = 0; index < projected.size(); ++index) {
            projected[index] =
                schedule[stage].b * gram[index] +
                schedule[stage].c * gram_squared[index];
        }
        projected =
            matrix_multiply(*matrix, false, projected, false, width);
        for (size_t index = 0; index < matrix->size(); ++index) {
            (*matrix)[index] =
                schedule[stage].a * (*matrix)[index] + projected[index];
        }
    }
}

static std::vector<float> prepare_direction_reference(
    const std::vector<float>& gradient,
    std::vector<float>* momentum,
    int width,
    const LlmcNormuonConfig& config,
    float gradient_scale) {
    std::vector<float> direction(gradient.size(), 0.0f);
    double norm_squared = 0.0;
    for (size_t index = 0; index < gradient.size(); ++index) {
        const float scaled_gradient = gradient_scale * gradient[index];
        const float next_momentum =
            config.momentum * (*momentum)[index] +
            (1.0f - config.momentum) * scaled_gradient;
        const float nesterov =
            (1.0f - config.momentum) * scaled_gradient +
            config.momentum * next_momentum;
        (*momentum)[index] = next_momentum;
        direction[index] = nesterov;
        norm_squared += static_cast<double>(nesterov) * nesterov;
    }
    const float denominator =
        1.02f * std::sqrt(static_cast<float>(norm_squared)) +
        config.epsilon;
    for (float& value : direction) {
        value = denominator > 0.0f ? value / denominator : 0.0f;
    }
    return direction;
}

static void finalize_reference(
    const std::vector<float>& direction,
    std::vector<float>* second_moment,
    std::vector<float>* master,
    int width,
    const LlmcNormuonConfig& config,
    float learning_rate) {
    std::vector<float> row_contributions(width, 0.0f);
    for (int row = 0; row < width; ++row) {
        double row_sum = 0.0;
        for (int column = 0; column < width; ++column) {
            const float value =
                direction[static_cast<size_t>(row) * width + column];
            row_sum += static_cast<double>(value) * value;
        }
        const float mean =
            static_cast<float>(row_sum / static_cast<double>(width));
        const float next =
            config.beta2 * (*second_moment)[row] +
            (1.0f - config.beta2) * mean;
        (*second_moment)[row] = next;
        row_contributions[row] =
            static_cast<float>(row_sum) /
            std::max(next, config.epsilon);
    }
    double normalized_norm_squared = 0.0;
    for (float contribution : row_contributions) {
        normalized_norm_squared += contribution;
    }
    const float global_scale =
        std::sqrt(static_cast<float>(width * width)) /
        std::sqrt(std::max(
            static_cast<float>(normalized_norm_squared),
            config.epsilon));
    const float decay_scale = 1.0f - learning_rate * config.weight_decay;
    const float update_learning_rate =
        learning_rate * config.update_scale;
    for (int row = 0; row < width; ++row) {
        const float local_scale =
            1.0f /
            std::sqrt(std::max((*second_moment)[row], config.epsilon)) *
            global_scale;
        for (int column = 0; column < width; ++column) {
            const size_t index =
                static_cast<size_t>(row) * width + column;
            (*master)[index] =
                (*master)[index] * decay_scale -
                update_learning_rate * direction[index] * local_scale;
        }
    }
}

static std::vector<float> tracker_correction_reference(
    const std::vector<float>& tracked_q,
    const std::vector<float>& normalized_momentum,
    int width,
    const LlmcNormuonConfig& config) {
    std::vector<float> phase =
        matrix_multiply(
            tracked_q, true, normalized_momentum, false, width);
    std::vector<float> skew(phase.size(), 0.0f);
    double symmetric_norm_squared = 0.0;
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index =
                static_cast<size_t>(row) * width + column;
            const float transpose =
                phase[static_cast<size_t>(column) * width + row];
            const float symmetric = 0.5f * (phase[index] + transpose);
            skew[index] = 0.5f * (phase[index] - transpose);
            symmetric_norm_squared +=
                static_cast<double>(symmetric) * symmetric;
        }
    }
    const float denominator =
        std::sqrt(static_cast<float>(symmetric_norm_squared)) +
        config.epsilon;
    std::vector<float> correction(phase.size(), 0.0f);
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index =
                static_cast<size_t>(row) * width + column;
            correction[index] =
                (row == column ? 1.0f : 0.0f) +
                config.correction_gain * skew[index] / denominator;
        }
    }
    apply_polynomial_reference(
        &correction,
        width,
        config.correction_iterations,
        config.correction_schedule);
    std::vector<float> direction =
        matrix_multiply(tracked_q, false, correction, false, width);
    if (config.retraction != 0U) {
        std::vector<float> qq_transpose =
            matrix_multiply(direction, false, direction, true, width);
        for (int row = 0; row < width; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t index =
                    static_cast<size_t>(row) * width + column;
                qq_transpose[index] =
                    (row == column ? 3.0f : 0.0f) -
                    qq_transpose[index];
            }
        }
        direction =
            matrix_multiply(
                qq_transpose, false, direction, false, width);
        for (float& value : direction) {
            value *= 0.5f;
        }
    }
    return direction;
}

static float orthogonality_error(
    const std::vector<float>& matrix,
    int width) {
    std::vector<float> gram =
        matrix_multiply(matrix, true, matrix, false, width);
    double error_squared = 0.0;
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index =
                static_cast<size_t>(row) * width + column;
            const float residual =
                gram[index] - (row == column ? 1.0f : 0.0f);
            error_squared += static_cast<double>(residual) * residual;
        }
    }
    return std::sqrt(static_cast<float>(error_squared));
}

template <typename T>
static std::vector<T> copy_from_device(const T* device, size_t count) {
    std::vector<T> host(count);
    cudaCheck(cudaMemcpy(
        host.data(),
        device,
        count * sizeof(T),
        cudaMemcpyDeviceToHost));
    return host;
}

template <typename T>
static void copy_to_device(T* device, const std::vector<T>& host) {
    cudaCheck(cudaMemcpy(
        device,
        host.data(),
        host.size() * sizeof(T),
        cudaMemcpyHostToDevice));
}

struct ViewBuffers {
    explicit ViewBuffers(int matrix_width)
        : width(matrix_width),
          elements(static_cast<size_t>(matrix_width) * matrix_width) {
        cudaCheck(cudaMalloc(&parameter, elements * sizeof(floatX)));
        cudaCheck(cudaMalloc(&gradient, elements * sizeof(floatX)));
        cudaCheck(cudaMalloc(&momentum, elements * sizeof(float)));
        cudaCheck(cudaMalloc(&second_moment, elements * sizeof(float)));
        cudaCheck(cudaMalloc(&master, elements * sizeof(float)));
    }

    ~ViewBuffers() {
        cudaFree(parameter);
        cudaFree(gradient);
        cudaFree(momentum);
        cudaFree(second_moment);
        cudaFree(master);
    }

    void load(
        const std::vector<floatX>& parameter_host,
        const std::vector<floatX>& gradient_host,
        const std::vector<float>& momentum_host,
        const std::vector<float>& second_moment_host,
        const std::vector<float>& master_host) {
        copy_to_device(parameter, parameter_host);
        copy_to_device(gradient, gradient_host);
        copy_to_device(momentum, momentum_host);
        copy_to_device(second_moment, second_moment_host);
        copy_to_device(master, master_host);
    }

    int width;
    size_t elements;
    floatX* parameter = nullptr;
    floatX* gradient = nullptr;
    float* momentum = nullptr;
    float* second_moment = nullptr;
    float* master = nullptr;
};

static LlmcOptimizerPlan minimal_runtime_plan(int width) {
    LlmcOptimizerPlan plan;
    llmc_optimizer_plan_reset(&plan);
    plan.built = true;
    plan.num_layers = 1;
    plan.channels = width;
    plan.normuon_parameter_type_count = 1;
    plan.normuon_view_count = 1;
    return plan;
}

static LlmcOptimizerParameterType contiguous_parameter_type(int width) {
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id = 10;
    parameter_type.name = "fcw_test";
    parameter_type.family_id = LLMC_OPTIMIZER_FAMILY_MLP_WUP;
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group =
        LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = 1;
    parameter_type.tensor_elements =
        static_cast<size_t>(width) * width;
    parameter_type.layer_elements = parameter_type.tensor_elements;
    parameter_type.matrix_width = width;
    parameter_type.views_per_layer = 1;
    return parameter_type;
}

static LlmcOptimizerMatrixView contiguous_view(int width) {
    LlmcOptimizerMatrixView view = {};
    view.rows = width;
    view.columns = width;
    view.row_stride = width;
    view.column_stride = 1;
    return view;
}

static std::vector<floatX> quantize_to_floatx(
    const std::vector<float>& values) {
    std::vector<floatX> output(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        output[index] = static_cast<floatX>(values[index]);
    }
    return output;
}

static std::vector<float> dequantize_floatx(
    const std::vector<floatX>& values) {
    std::vector<float> output(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        output[index] = static_cast<float>(values[index]);
    }
    return output;
}

static void test_parameter_plan_and_views() {
    GPT2Config model_config = {};
    model_config.max_seq_len = 1024;
    model_config.vocab_size = 50257;
    model_config.padded_vocab_size = 50304;
    model_config.num_layers = 12;
    model_config.num_heads = 12;
    model_config.channels = 768;
    size_t parameter_elements[NUM_PARAMETER_TENSORS];
    size_t parameter_sizeof[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(
        parameter_elements, parameter_sizeof, model_config);

    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    LlmcOptimizerPlan plan;
    char error[256];
    TEST_CHECK(
        llmc_build_optimizer_plan(
            &plan,
            &config,
            model_config.num_layers,
            model_config.channels,
            parameter_elements,
            error,
            sizeof(error)),
        "default AdamW plan builds");
    for (int tensor_id = 0; tensor_id < NUM_PARAMETER_TENSORS; ++tensor_id) {
        TEST_CHECK(
            plan.parameter_types[tensor_id].backend_kind ==
                LLMC_OPTIMIZER_BACKEND_ADAMW,
            "all-AdamW default routes every parameter to AdamW");
    }

    config.optimizer_selection =
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    TEST_CHECK(
        llmc_build_optimizer_plan(
            &plan,
            &config,
            model_config.num_layers,
            model_config.channels,
            parameter_elements,
            error,
            sizeof(error)),
        "mixed optimizer plan builds");
    TEST_CHECK(
        plan.normuon_parameter_type_count == 2,
        "exactly two parameter types route to NorMuon");
    TEST_CHECK(
        plan.normuon_view_count == 96,
        "GPT-2 small has exactly 96 square views");
    for (int tensor_id = 0; tensor_id < NUM_PARAMETER_TENSORS; ++tensor_id) {
        const bool expected_normuon = tensor_id == 10 || tensor_id == 12;
        TEST_CHECK(
            (plan.parameter_types[tensor_id].backend_kind ==
             LLMC_OPTIMIZER_BACKEND_NORMUON) == expected_normuon,
            "only fcw/fcprojw route to NorMuon");
    }
    const size_t q_elements =
        static_cast<size_t>(plan.normuon_view_count) *
        model_config.channels *
        model_config.channels;
    TEST_CHECK(
        q_elements == 56623104ULL,
        "GPT-2 small tracker Q element count is exact");
    TEST_CHECK(
        q_elements * sizeof(float) == 216ULL * 1024ULL * 1024ULL,
        "GPT-2 small tracker Q is exactly 216 MiB");

    GPT2Config small_config = model_config;
    small_config.num_layers = 2;
    small_config.num_heads = 1;
    small_config.channels = 17;
    fill_in_parameter_sizes(
        parameter_elements, parameter_sizeof, small_config);
    TEST_CHECK(
        llmc_build_optimizer_plan(
            &plan,
            &config,
            small_config.num_layers,
            small_config.channels,
            parameter_elements,
            error,
            sizeof(error)),
        "small plan builds for view enumeration");
    for (int tensor_id : {10, 12}) {
        const LlmcOptimizerParameterType& parameter_type =
            plan.parameter_types[tensor_id];
        std::vector<int> coverage(parameter_type.layer_elements, 0);
        for (int view_index = 0;
             view_index < parameter_type.views_per_layer;
             ++view_index) {
            LlmcOptimizerMatrixView view;
            TEST_CHECK(
                parameter_type.enumerate_matrix_view(
                    &parameter_type, view_index, &view),
                "view enumerator succeeds");
            TEST_CHECK(
                llmc_optimizer_view_within_bounds(
                    &parameter_type, &view),
                "view stays in bounds");
            TEST_CHECK(
                view.column_stride == 1U,
                "view column stride is one");
            if (tensor_id == 10) {
                TEST_CHECK(
                    view.element_offset ==
                        static_cast<size_t>(view_index) * 17U * 17U,
                    "Wup row-block base offset is exact");
                TEST_CHECK(
                    view.row_stride == 17U,
                    "Wup row stride is C");
            } else {
                TEST_CHECK(
                    view.element_offset ==
                        static_cast<size_t>(view_index) * 17U,
                    "Wdown column-panel base offset is exact");
                TEST_CHECK(
                    view.row_stride == 68U,
                    "Wdown row stride is 4C");
            }
            for (size_t row = 0; row < view.rows; ++row) {
                for (size_t column = 0; column < view.columns; ++column) {
                    const size_t offset =
                        view.element_offset +
                        row * view.row_stride +
                        column * view.column_stride;
                    TEST_CHECK(
                        offset < coverage.size(),
                        "enumerated element remains in bounds");
                    coverage[offset]++;
                }
            }
        }
        for (int count : coverage) {
            TEST_CHECK(
                count == 1,
                "square views cover each matrix element exactly once");
        }
    }
}

static void test_scratch_against_reference() {
    constexpr int width = 5;
    constexpr float learning_rate = 0.0125f;
    constexpr float gradient_scale = 0.75f;
    const size_t elements = static_cast<size_t>(width) * width;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection =
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
    llmc_normuon_resolve_schedules(&config);

    std::vector<float> gradient_fp32(elements);
    std::vector<float> momentum_initial(elements);
    std::vector<float> second_initial(elements, 0.0f);
    std::vector<float> master_initial(elements);
    for (size_t index = 0; index < elements; ++index) {
        gradient_fp32[index] =
            0.03f * std::sin(static_cast<float>(index + 1)) +
            0.002f * static_cast<float>(index);
        momentum_initial[index] =
            0.01f * std::cos(static_cast<float>(index + 2));
        master_initial[index] =
            -0.2f + 0.015f * static_cast<float>(index);
    }
    for (int row = 0; row < width; ++row) {
        second_initial[row] = 0.02f + 0.001f * row;
    }
    const std::vector<floatX> gradient_bf16 =
        quantize_to_floatx(gradient_fp32);
    const std::vector<float> quantized_gradient =
        dequantize_floatx(gradient_bf16);
    const std::vector<floatX> parameter_initial =
        quantize_to_floatx(master_initial);

    std::vector<float> momentum_reference = momentum_initial;
    std::vector<float> second_reference = second_initial;
    std::vector<float> master_reference = master_initial;
    std::vector<float> direction_reference =
        prepare_direction_reference(
            quantized_gradient,
            &momentum_reference,
            width,
            config,
            gradient_scale);
    apply_polynomial_reference(
        &direction_reference,
        width,
        LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
        config.refresh_schedule);
    finalize_reference(
        direction_reference,
        &second_reference,
        &master_reference,
        width,
        config,
        learning_rate);

    LlmcOptimizerPlan plan = minimal_runtime_plan(width);
    LlmcOptimizerParameterType parameter_type =
        contiguous_parameter_type(width);
    LlmcOptimizerMatrixView view = contiguous_view(width);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    TEST_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "scratch runtime allocates");
    ViewBuffers buffers(width);
    buffers.load(
        parameter_initial,
        gradient_bf16,
        momentum_initial,
        second_initial,
        master_initial);
    TEST_CHECK(
        llmc_normuon_update_view(
            &runtime,
            cublas_handle,
            main_stream,
            buffers.parameter,
            buffers.gradient,
            buffers.momentum,
            buffers.second_moment,
            buffers.master,
            &parameter_type,
            &view,
            &config,
            learning_rate,
            gradient_scale,
            0U,
            0,
            0),
        "five-stage scratch update succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<float> momentum_gpu =
        copy_from_device(buffers.momentum, elements);
    const std::vector<float> second_gpu =
        copy_from_device(buffers.second_moment, elements);
    const std::vector<float> master_gpu =
        copy_from_device(buffers.master, elements);
    TEST_CHECK(
        max_abs_difference(momentum_gpu, momentum_reference) < 2.0e-6f,
        "scratch momentum matches direct FP32 reference");
    TEST_CHECK(
        max_abs_difference(second_gpu, second_reference) < 3.0e-5f,
        "scratch second moment matches direct FP32 reference");
    TEST_CHECK(
        max_abs_difference(master_gpu, master_reference) < 5.0e-5f,
        "scratch master update matches direct FP32 reference");
    TEST_CHECK(all_finite(master_gpu), "scratch master output is finite");

    LlmcNormuonRuntime duplicate_runtime;
    llmc_normuon_runtime_reset(&duplicate_runtime);
    TEST_CHECK(
        llmc_normuon_runtime_allocate(
            &duplicate_runtime, &plan, &config),
        "duplicate scratch runtime allocates");
    ViewBuffers duplicate(width);
    duplicate.load(
        parameter_initial,
        gradient_bf16,
        momentum_initial,
        second_initial,
        master_initial);
    TEST_CHECK(
        llmc_normuon_update_view(
            &duplicate_runtime,
            cublas_handle,
            main_stream,
            duplicate.parameter,
            duplicate.gradient,
            duplicate.momentum,
            duplicate.second_moment,
            duplicate.master,
            &parameter_type,
            &view,
            &config,
            learning_rate,
            gradient_scale,
            0U,
            0,
            0),
        "duplicate scratch update succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<floatX> parameter_first =
        copy_from_device(buffers.parameter, elements);
    const std::vector<floatX> parameter_second =
        copy_from_device(duplicate.parameter, elements);
    TEST_CHECK(
        memcmp(
            parameter_first.data(),
            parameter_second.data(),
            elements * sizeof(floatX)) == 0,
        "tuple-keyed stochastic rounding is deterministic");

    std::vector<float> nan_gradient_fp32 = quantized_gradient;
    nan_gradient_fp32[0] = NAN;
    const std::vector<floatX> nan_gradient =
        quantize_to_floatx(nan_gradient_fp32);
    ViewBuffers nonfinite(width);
    nonfinite.load(
        parameter_initial,
        nan_gradient,
        momentum_initial,
        second_initial,
        master_initial);
    LlmcNormuonRuntime nonfinite_runtime;
    llmc_normuon_runtime_reset(&nonfinite_runtime);
    TEST_CHECK(
        llmc_normuon_runtime_allocate(
            &nonfinite_runtime, &plan, &config),
        "nonfinite runtime allocates");
    TEST_CHECK(
        !llmc_normuon_update_view(
            &nonfinite_runtime,
            cublas_handle,
            main_stream,
            nonfinite.parameter,
            nonfinite.gradient,
            nonfinite.momentum,
            nonfinite.second_moment,
            nonfinite.master,
            &parameter_type,
            &view,
            &config,
            learning_rate,
            gradient_scale,
            0U,
            0,
            0),
        "nonfinite gradient is rejected");
    const std::vector<float> nonfinite_master =
        copy_from_device(nonfinite.master, elements);
    TEST_CHECK(
        memcmp(
            nonfinite_master.data(),
            master_initial.data(),
            elements * sizeof(float)) == 0,
        "nonfinite guard prevents master-weight mutation");

    llmc_normuon_runtime_free(&nonfinite_runtime);
    llmc_normuon_runtime_free(&duplicate_runtime);
    llmc_normuon_runtime_free(&runtime);
}

static std::vector<float> tracker_gradient(int width, int step) {
    std::vector<float> gradient(
        static_cast<size_t>(width) * width);
    for (size_t index = 0; index < gradient.size(); ++index) {
        gradient[index] =
            0.025f *
                std::sin(
                    0.4f * static_cast<float>(index + 1) +
                    0.7f * static_cast<float>(step)) +
            0.003f * static_cast<float>((index + step) % width);
    }
    return gradient;
}

static bool run_tracker_step(
    ViewBuffers* buffers,
    LlmcNormuonRuntime* runtime,
    const LlmcOptimizerParameterType& parameter_type,
    const LlmcOptimizerMatrixView& view,
    const LlmcNormuonConfig& config,
    int step,
    float learning_rate,
    const std::vector<float>& gradient_fp32) {
    copy_to_device(buffers->gradient, quantize_to_floatx(gradient_fp32));
    return llmc_normuon_update_view(
        runtime,
        cublas_handle,
        main_stream,
        buffers->parameter,
        buffers->gradient,
        buffers->momentum,
        buffers->second_moment,
        buffers->master,
        &parameter_type,
        &view,
        &config,
        learning_rate,
        1.0f,
        static_cast<uint64_t>(step),
        0,
        0);
}

static void test_tracker_reference_and_resume() {
    constexpr int width = 5;
    constexpr float learning_rate = 0.01f;
    const size_t elements = static_cast<size_t>(width) * width;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection =
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    config.refresh_interval = 3U;
    config.refresh_policy =
        LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config.correction_policy =
        LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config.correction_iterations = 2U;
    config.correction_gain = 1.0f;
    config.retraction = 1U;
    llmc_normuon_resolve_schedules(&config);

    TEST_CHECK(
        config.correction_schedule[0].a == 1.875f &&
        config.correction_schedule[1].a == 1.875f,
        "canonical correction schedule begins with ordered stages");
    TEST_CHECK(
        config.refresh_schedule[0].a == 3.4445f &&
        config.refresh_schedule[4].c == 2.0315f,
        "stock refresh schedule contains all five exact stages");

    LlmcOptimizerPlan plan = minimal_runtime_plan(width);
    LlmcOptimizerParameterType parameter_type =
        contiguous_parameter_type(width);
    LlmcOptimizerMatrixView view = contiguous_view(width);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    TEST_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "tracker runtime allocates");
    TEST_CHECK(
        runtime.tracked_q_view_count == 8U,
        "tracker runtime owns eight logical views per layer");

    std::vector<float> momentum_reference(elements, 0.0f);
    std::vector<float> second_reference(elements, 0.0f);
    std::vector<float> master_reference(elements);
    for (size_t index = 0; index < elements; ++index) {
        momentum_reference[index] =
            0.004f * std::cos(static_cast<float>(index + 1));
        master_reference[index] =
            0.1f - 0.007f * static_cast<float>(index);
    }
    for (int row = 0; row < width; ++row) {
        second_reference[row] = 0.015f + 0.001f * row;
    }
    const std::vector<float> momentum_initial = momentum_reference;
    const std::vector<float> second_initial = second_reference;
    const std::vector<float> master_initial = master_reference;
    ViewBuffers buffers(width);
    buffers.load(
        quantize_to_floatx(master_initial),
        quantize_to_floatx(tracker_gradient(width, 0)),
        momentum_initial,
        second_initial,
        master_initial);

    const std::vector<float> gradient0 =
        dequantize_floatx(
            quantize_to_floatx(tracker_gradient(width, 0)));
    std::vector<float> q_reference =
        prepare_direction_reference(
            gradient0,
            &momentum_reference,
            width,
            config,
            1.0f);
    apply_polynomial_reference(
        &q_reference,
        width,
        LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
        config.refresh_schedule);
    finalize_reference(
        q_reference,
        &second_reference,
        &master_reference,
        width,
        config,
        learning_rate);
    TEST_CHECK(
        run_tracker_step(
            &buffers,
            &runtime,
            parameter_type,
            view,
            config,
            0,
            learning_rate,
            tracker_gradient(width, 0)),
        "tracker refresh at step zero succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    TEST_CHECK(runtime.q_valid[0] == 1U, "Q is initialized at step zero");
    TEST_CHECK(
        runtime.refresh_count[0] == 1U &&
        runtime.last_refresh_step[0] == 0,
        "step-zero refresh phase is recorded");
    std::vector<float> q_gpu =
        copy_from_device(runtime.tracked_q, elements);
    TEST_CHECK(
        max_abs_difference(q_gpu, q_reference) < 7.0e-5f,
        "tracker refresh Q matches five-stage scratch reference");

    const std::vector<float> gradient1 =
        dequantize_floatx(
            quantize_to_floatx(tracker_gradient(width, 1)));
    std::vector<float> normalized1 =
        prepare_direction_reference(
            gradient1,
            &momentum_reference,
            width,
            config,
            1.0f);
    q_reference =
        tracker_correction_reference(
            q_reference, normalized1, width, config);
    finalize_reference(
        q_reference,
        &second_reference,
        &master_reference,
        width,
        config,
        learning_rate);
    TEST_CHECK(
        run_tracker_step(
            &buffers,
            &runtime,
            parameter_type,
            view,
            config,
            1,
            learning_rate,
            tracker_gradient(width, 1)),
        "two-stage tracked correction succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    TEST_CHECK(
        runtime.refresh_count[0] == 1U &&
        runtime.last_refresh_step[0] == 0,
        "intervening correction does not advance refresh phase");
    q_gpu = copy_from_device(runtime.tracked_q, elements);
    const std::vector<float> master_gpu =
        copy_from_device(buffers.master, elements);
    TEST_CHECK(
        max_abs_difference(q_gpu, q_reference) < 2.5e-4f,
        "tracked correction and retraction match direct FP32 reference");
    TEST_CHECK(
        max_abs_difference(master_gpu, master_reference) < 3.5e-4f,
        "tracker master update matches direct FP32 reference");
    TEST_CHECK(
        orthogonality_error(q_gpu, width) < 0.35f,
        "tracked Q remains approximately orthogonal");
    TEST_CHECK(all_finite(q_gpu), "tracked Q is finite");

    const char* companion_path =
        "build/test_normuon_companion.bin";
    TEST_CHECK(
        llmc_normuon_save_companion(
            companion_path,
            2,
            1,
            0,
            &plan,
            &config,
            &runtime,
            main_stream),
        "tracker companion saves");
    LlmcNormuonCompanionInfo companion_info;
    TEST_CHECK(
        llmc_normuon_read_companion_info(
            companion_path, &companion_info),
        "tracker companion header reads");
    TEST_CHECK(
        companion_info.step == 2 &&
        companion_info.q_view_count == 8U &&
        companion_info.q_element_count == 8U * elements,
        "companion records exact step and Q shape");
    TEST_CHECK(
        llmc_normuon_config_equal(
            &companion_info.config, &config),
        "companion records exact policies and coefficient schedules");

    const std::vector<floatX> parameter_checkpoint =
        copy_from_device(buffers.parameter, elements);
    const std::vector<float> momentum_checkpoint =
        copy_from_device(buffers.momentum, elements);
    const std::vector<float> second_checkpoint =
        copy_from_device(buffers.second_moment, elements);
    const std::vector<float> master_checkpoint =
        copy_from_device(buffers.master, elements);
    ViewBuffers resumed_buffers(width);
    resumed_buffers.load(
        parameter_checkpoint,
        quantize_to_floatx(tracker_gradient(width, 2)),
        momentum_checkpoint,
        second_checkpoint,
        master_checkpoint);
    LlmcNormuonRuntime resumed_runtime;
    llmc_normuon_runtime_reset(&resumed_runtime);
    TEST_CHECK(
        llmc_normuon_runtime_allocate(
            &resumed_runtime, &plan, &config),
        "resume tracker runtime allocates");
    TEST_CHECK(
        llmc_normuon_load_companion(
            companion_path,
            2,
            1,
            0,
            &plan,
            &config,
            &resumed_runtime,
            main_stream),
        "exact tracker companion loads");
    cudaCheck(cudaStreamSynchronize(main_stream));
    TEST_CHECK(
        memcmp(
            runtime.q_valid,
            resumed_runtime.q_valid,
            runtime.tracked_q_view_count * sizeof(uint8_t)) == 0,
        "Q-valid flags resume exactly");
    TEST_CHECK(
        memcmp(
            runtime.refresh_count,
            resumed_runtime.refresh_count,
            runtime.tracked_q_view_count * sizeof(uint64_t)) == 0,
        "refresh counters resume exactly");
    TEST_CHECK(
        memcmp(
            runtime.last_refresh_step,
            resumed_runtime.last_refresh_step,
            runtime.tracked_q_view_count * sizeof(int64_t)) == 0,
        "refresh phase resumes exactly");
    const std::vector<float> resumed_q =
        copy_from_device(
            resumed_runtime.tracked_q,
            resumed_runtime.tracked_q_elements);
    const std::vector<float> original_q =
        copy_from_device(runtime.tracked_q, runtime.tracked_q_elements);
    TEST_CHECK(
        memcmp(
            resumed_q.data(),
            original_q.data(),
            original_q.size() * sizeof(float)) == 0,
        "persistent Q resumes bit-exactly");

    TEST_CHECK(
        run_tracker_step(
            &buffers,
            &runtime,
            parameter_type,
            view,
            config,
            2,
            learning_rate,
            tracker_gradient(width, 2)),
        "original tracker continuation succeeds");
    TEST_CHECK(
        run_tracker_step(
            &resumed_buffers,
            &resumed_runtime,
            parameter_type,
            view,
            config,
            2,
            learning_rate,
            tracker_gradient(width, 2)),
        "resumed tracker continuation succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<float> original_master_after =
        copy_from_device(buffers.master, elements);
    const std::vector<float> resumed_master_after =
        copy_from_device(resumed_buffers.master, elements);
    TEST_CHECK(
        memcmp(
            original_master_after.data(),
            resumed_master_after.data(),
            elements * sizeof(float)) == 0,
        "checkpoint continuation master weights are bit-exact");

    TEST_CHECK(
        run_tracker_step(
            &buffers,
            &runtime,
            parameter_type,
            view,
            config,
            3,
            learning_rate,
            tracker_gradient(width, 3)),
        "tracker refresh at step three succeeds");
    TEST_CHECK(
        runtime.refresh_count[0] == 2U &&
        runtime.last_refresh_step[0] == 3,
        "refresh cadence is exactly steps 0,3,6,...");

    LlmcNormuonConfig mismatch = config;
    mismatch.correction_iterations = 3U;
    llmc_normuon_resolve_schedules(&mismatch);
    TEST_CHECK(
        !llmc_normuon_load_companion(
            companion_path,
            2,
            1,
            0,
            &plan,
            &mismatch,
            &resumed_runtime,
            main_stream),
        "resume rejects policy/cadence mismatch");
    remove(companion_path);
    llmc_normuon_runtime_free(&resumed_runtime);
    llmc_normuon_runtime_free(&runtime);
}

static void test_init_from_master_only_no_mutation() {
    GPT2 model;
    gpt2_init_common(&model);
    model.config.max_seq_len = 4;
    model.config.vocab_size = 8;
    model.config.padded_vocab_size = 8;
    model.config.num_layers = 1;
    model.config.num_heads = 1;
    model.config.channels = 4;
    gpt2_allocate_weights(&model);
    model.grads_memory =
        malloc_and_point_parameters(
            &model.grads,
            model.param_elements,
            model.param_sizeof);
    cudaCheck(cudaMalloc(
        &model.m_memory, model.num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(
        &model.v_memory, model.num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(
        &model.master_weights, model.num_parameters * sizeof(float)));
    model.init_state = false;
    model.use_master_weights = 1;
    model.rng_state = 123456789ULL;
    model.rng_state_last_update = 987654321ULL;
    llmc_normuon_config_defaults(&model.optimizer_config);
    model.optimizer_config.optimizer_selection =
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    model.optimizer_config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    llmc_normuon_resolve_schedules(&model.optimizer_config);
    char error[256];
    TEST_CHECK(
        llmc_build_optimizer_plan(
            &model.optimizer_plan,
            &model.optimizer_config,
            model.config.num_layers,
            model.config.channels,
            model.param_elements,
            error,
            sizeof(error)),
        "tiny GPT2 mixed optimizer plan builds");
    TEST_CHECK(
        llmc_normuon_runtime_allocate(
            &model.normuon_runtime,
            &model.optimizer_plan,
            &model.optimizer_config),
        "tiny GPT2 tracker runtime allocates");
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    std::vector<float> master(model.num_parameters);
    std::vector<float> momentum(model.num_parameters);
    std::vector<float> second_moment(model.num_parameters);
    for (size_t index = 0; index < model.num_parameters; ++index) {
        master[index] =
            -0.4f + 0.0005f * static_cast<float>(index);
        momentum[index] =
            0.01f * std::sin(static_cast<float>(index));
        second_moment[index] =
            0.02f + 0.00001f * static_cast<float>(index % 101);
    }
    copy_to_device(model.master_weights, master);
    copy_to_device(model.m_memory, momentum);
    copy_to_device(model.v_memory, second_moment);
    const std::vector<floatX> parameter =
        quantize_to_floatx(master);
    copy_to_device(
        static_cast<floatX*>(model.params_memory), parameter);
    cudaCheck(cudaMemset(
        model.grads_memory, 0, model.num_parameters_bytes));

    std::vector<float> tracked_q(
        model.normuon_runtime.tracked_q_elements, 0.0f);
    for (size_t view_index = 0;
         view_index < model.normuon_runtime.tracked_q_view_count;
         ++view_index) {
        for (int diagonal = 0;
             diagonal < model.config.channels;
             ++diagonal) {
            tracked_q[
                view_index * 16U +
                static_cast<size_t>(diagonal) * 4U +
                diagonal] = 1.0f;
        }
    }
    copy_to_device(model.normuon_runtime.tracked_q, tracked_q);
    model.normuon_runtime.q_valid[0] = 1U;
    model.normuon_runtime.refresh_count[0] = 7U;
    model.normuon_runtime.last_refresh_step[0] = 3;
    const std::vector<uint8_t> q_valid_before(
        model.normuon_runtime.q_valid,
        model.normuon_runtime.q_valid +
            model.normuon_runtime.tracked_q_view_count);
    const std::vector<uint64_t> refresh_before(
        model.normuon_runtime.refresh_count,
        model.normuon_runtime.refresh_count +
            model.normuon_runtime.tracked_q_view_count);
    const std::vector<int64_t> phase_before(
        model.normuon_runtime.last_refresh_step,
        model.normuon_runtime.last_refresh_step +
            model.normuon_runtime.tracked_q_view_count);
    const unsigned long long rng_before = model.rng_state;
    const unsigned long long rng_last_before =
        model.rng_state_last_update;

    gpt2_update(
        &model,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        5,
        &multi_gpu_config,
        -1.0f,
        true);
    const std::vector<float> master_after =
        copy_from_device(model.master_weights, model.num_parameters);
    const std::vector<float> momentum_after =
        copy_from_device(model.m_memory, model.num_parameters);
    const std::vector<float> second_after =
        copy_from_device(model.v_memory, model.num_parameters);
    const std::vector<float> q_after =
        copy_from_device(
            model.normuon_runtime.tracked_q,
            model.normuon_runtime.tracked_q_elements);
    TEST_CHECK(
        memcmp(
            master.data(),
            master_after.data(),
            master.size() * sizeof(float)) == 0,
        "init_from_master_only does not mutate FP32 master weights");
    TEST_CHECK(
        memcmp(
            momentum.data(),
            momentum_after.data(),
            momentum.size() * sizeof(float)) == 0,
        "init_from_master_only does not mutate momentum");
    TEST_CHECK(
        memcmp(
            second_moment.data(),
            second_after.data(),
            second_moment.size() * sizeof(float)) == 0,
        "init_from_master_only does not mutate second moment");
    TEST_CHECK(
        memcmp(
            tracked_q.data(),
            q_after.data(),
            tracked_q.size() * sizeof(float)) == 0,
        "init_from_master_only does not mutate Q");
    TEST_CHECK(
        memcmp(
            q_valid_before.data(),
            model.normuon_runtime.q_valid,
            q_valid_before.size() * sizeof(uint8_t)) == 0 &&
        memcmp(
            refresh_before.data(),
            model.normuon_runtime.refresh_count,
            refresh_before.size() * sizeof(uint64_t)) == 0 &&
        memcmp(
            phase_before.data(),
            model.normuon_runtime.last_refresh_step,
            phase_before.size() * sizeof(int64_t)) == 0,
        "init_from_master_only does not mutate tracker phase");
    TEST_CHECK(
        model.rng_state == rng_before &&
        model.rng_state_last_update == rng_last_before,
        "init_from_master_only does not consume optimizer RNG state");

    llmc_normuon_runtime_free(&model.normuon_runtime);
    cudaFree(model.params_memory);
    cudaFree(model.grads_memory);
    cudaFree(model.m_memory);
    cudaFree(model.v_memory);
    cudaFree(model.master_weights);
}

int main() {
    char server_ip[2] = "";
    char filesystem_path[2] = "";
    char init_method[4] = "mpi";
    multi_gpu_config =
        multi_gpu_config_init(
            1,
            0,
            1,
            server_ip,
            filesystem_path,
            init_method);
    set_zero_configs(&multi_gpu_config, 0, 1);
    common_start(false, false);

    test_parameter_plan_and_views();
    test_scratch_against_reference();
    test_tracker_reference_and_resume();
    test_init_from_master_only_no_mutation();

    GPT2 unused_model = {};
    common_free(unused_model);
    multi_gpu_config_free(&multi_gpu_config);
    if (test_failures == 0) {
        printf("All focused llm.c NorMuon tests passed.\n");
        return EXIT_SUCCESS;
    }
    fprintf(stderr, "%d focused llm.c NorMuon tests failed.\n", test_failures);
    return EXIT_FAILURE;
}
