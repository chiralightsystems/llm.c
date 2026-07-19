#define TESTING
#include "train_gpt2.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int batched_test_failures = 0;

#define BATCHED_CHECK(condition, message)                                      \
    do {                                                                       \
        if (!(condition)) {                                                     \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
            batched_test_failures++;                                            \
        }                                                                       \
    } while (0)

template <typename T>
static std::vector<T> batched_copy_from_device(const T* device, size_t count) {
    std::vector<T> host(count);
    cudaCheck(cudaMemcpy(
        host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

template <typename T>
static void batched_copy_to_device(T* device, const std::vector<T>& host) {
    cudaCheck(cudaMemcpy(
        device,
        host.data(),
        host.size() * sizeof(T),
        cudaMemcpyHostToDevice));
}

static float round_to_floatx(float value) {
    return static_cast<float>(static_cast<floatX>(value));
}

static std::vector<floatX> quantize_floatx(const std::vector<float>& values) {
    std::vector<floatX> output(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        output[index] = static_cast<floatX>(values[index]);
    }
    return output;
}

static float batched_max_abs_difference(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {
    BATCHED_CHECK(lhs.size() == rhs.size(), "comparison vector sizes match");
    float maximum = 0.0f;
    const size_t count = std::min(lhs.size(), rhs.size());
    for (size_t index = 0; index < count; ++index) {
        maximum = std::max(maximum, std::fabs(lhs[index] - rhs[index]));
    }
    return maximum;
}

static bool batched_all_finite(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static std::vector<float> bf16_operand_matmul(
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
        const size_t index = transpose
            ? static_cast<size_t>(column) * width + row
            : static_cast<size_t>(row) * width + column;
        return round_to_floatx(matrix[index]);
    };
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            float sum = 0.0f;
            for (int inner = 0; inner < width; ++inner) {
                sum += load(lhs, transpose_lhs, row, inner) *
                       load(rhs, transpose_rhs, inner, column);
            }
            output[static_cast<size_t>(row) * width + column] = sum;
        }
    }
    return output;
}

static void apply_bf16_polynomial_reference(
    std::vector<float>* matrix,
    int width,
    uint32_t stage_count,
    const LlmcNormuonPolynomialStep* schedule) {
    for (uint32_t stage = 0; stage < stage_count; ++stage) {
        std::vector<float> gram = bf16_operand_matmul(
            *matrix, true, *matrix, false, width);
        std::vector<float> polynomial(gram.size());
        if (schedule[stage].c != 0.0f) {
            std::vector<float> gram_squared = bf16_operand_matmul(
                gram, false, gram, false, width);
            for (size_t index = 0; index < gram.size(); ++index) {
                polynomial[index] = round_to_floatx(
                    schedule[stage].b * gram[index] +
                    schedule[stage].c * gram_squared[index]);
            }
        } else {
            for (size_t index = 0; index < gram.size(); ++index) {
                polynomial[index] =
                    round_to_floatx(schedule[stage].b * gram[index]);
            }
        }
        std::vector<float> projected = bf16_operand_matmul(
            *matrix, false, polynomial, false, width);
        for (size_t index = 0; index < matrix->size(); ++index) {
            (*matrix)[index] =
                schedule[stage].a * (*matrix)[index] + projected[index];
        }
    }
}

static std::vector<float> prepare_direction_reference_batched(
    const std::vector<float>& gradient,
    std::vector<float>* momentum,
    const LlmcNormuonConfig& config,
    float gradient_scale) {
    std::vector<float> direction(gradient.size());
    float norm_squared = 0.0f;
    for (size_t index = 0; index < gradient.size(); ++index) {
        const float scaled_gradient = gradient_scale * gradient[index];
        const float next =
            config.momentum * (*momentum)[index] +
            (1.0f - config.momentum) * scaled_gradient;
        const float nesterov =
            (1.0f - config.momentum) * scaled_gradient +
            config.momentum * next;
        (*momentum)[index] = next;
        direction[index] = nesterov;
        norm_squared += nesterov * nesterov;
    }
    const float denominator =
        1.02f * std::sqrt(std::max(norm_squared, 0.0f)) + config.epsilon;
    for (float& value : direction) {
        value = denominator > 0.0f ? value / denominator : 0.0f;
    }
    return direction;
}

static void finalize_reference_batched(
    const std::vector<float>& direction,
    std::vector<float>* second_moment,
    std::vector<float>* master,
    int width,
    const LlmcNormuonConfig& config,
    float learning_rate) {
    float normalized_norm_squared = 0.0f;
    for (int row = 0; row < width; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < width; ++column) {
            const float value =
                direction[static_cast<size_t>(row) * width + column];
            sum += value * value;
        }
        const float mean = sum / static_cast<float>(width);
        const float next =
            config.beta2 * (*second_moment)[row] +
            (1.0f - config.beta2) * mean;
        (*second_moment)[row] = next;
        normalized_norm_squared += sum / std::max(next, config.epsilon);
    }
    const float global_scale =
        static_cast<float>(width) /
        std::sqrt(std::max(normalized_norm_squared, config.epsilon));
    const float decay_scale = 1.0f - learning_rate * config.weight_decay;
    const float update_learning_rate = learning_rate * config.update_scale;
    for (int row = 0; row < width; ++row) {
        const float local_scale =
            global_scale /
            std::sqrt(std::max((*second_moment)[row], config.epsilon));
        for (int column = 0; column < width; ++column) {
            const size_t index = static_cast<size_t>(row) * width + column;
            (*master)[index] =
                (*master)[index] * decay_scale -
                update_learning_rate * direction[index] * local_scale;
        }
    }
}

static std::vector<float> tracker_correction_bf16_reference(
    const std::vector<float>& tracked_q,
    const std::vector<float>& normalized,
    int width,
    const LlmcNormuonConfig& config,
    bool native_right_retraction) {
    std::vector<float> phase = bf16_operand_matmul(
        tracked_q, true, normalized, false, width);
    float symmetric_norm_squared = 0.0f;
    std::vector<float> correction(phase.size(), 0.0f);
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index = static_cast<size_t>(row) * width + column;
            const float transpose =
                phase[static_cast<size_t>(column) * width + row];
            const float symmetric = 0.5f * (phase[index] + transpose);
            symmetric_norm_squared += symmetric * symmetric;
        }
    }
    const float denominator =
        std::sqrt(std::max(symmetric_norm_squared, 0.0f)) + config.epsilon;
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index = static_cast<size_t>(row) * width + column;
            const float transpose =
                phase[static_cast<size_t>(column) * width + row];
            const float skew = 0.5f * (phase[index] - transpose);
            correction[index] =
                (row == column ? 1.0f : 0.0f) +
                config.correction_gain * skew / denominator;
        }
    }
    const bool commute_canonical_stage2 =
        config.retraction_mode ==
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    apply_bf16_polynomial_reference(
        &correction,
        width,
        commute_canonical_stage2 ? 1U : config.correction_iterations,
        config.correction_schedule);
    std::vector<float> direction = bf16_operand_matmul(
        tracked_q, false, correction, false, width);
    if (commute_canonical_stage2) {
        apply_bf16_polynomial_reference(
            &direction, width, 1U, config.correction_schedule + 1U);
    } else if (config.retraction_mode ==
               LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ) {
        std::vector<float> retraction = native_right_retraction
            ? bf16_operand_matmul(
                  direction, true, direction, false, width)
            : bf16_operand_matmul(
                  direction, false, direction, true, width);
        for (int row = 0; row < width; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t index = static_cast<size_t>(row) * width + column;
                retraction[index] =
                    (row == column ? 3.0f : 0.0f) - retraction[index];
            }
        }
        direction = native_right_retraction
            ? bf16_operand_matmul(
                  direction, false, retraction, false, width)
            : bf16_operand_matmul(
                  retraction, false, direction, false, width);
        for (float& value : direction) {
            value *= 0.5f;
        }
    }
    return direction;
}

static float batched_orthogonality_error(
    const std::vector<float>& matrix,
    int width) {
    std::vector<float> gram = bf16_operand_matmul(
        matrix, true, matrix, false, width);
    float sum = 0.0f;
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index = static_cast<size_t>(row) * width + column;
            const float residual =
                gram[index] - (row == column ? 1.0f : 0.0f);
            sum += residual * residual;
        }
    }
    return std::sqrt(sum);
}

static size_t tensor_offset_for_logical(
    int family_id,
    size_t matrix_index,
    size_t element_index,
    int width) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t layer = matrix_index / LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t view = matrix_index % LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t row = element_index / width;
    const size_t column = element_index % width;
    const size_t layer_offset =
        layer * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX * matrix_elements;
    if (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP) {
        return layer_offset + view * matrix_elements + element_index;
    }
    return layer_offset + view * width +
           row * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX * width + column;
}

static size_t second_offset_for_logical(
    int family_id,
    size_t matrix_index,
    size_t row,
    int width) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t layer = matrix_index / LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t view = matrix_index % LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t layer_offset =
        layer * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX * matrix_elements;
    return layer_offset +
           (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
                ? view * matrix_elements
                : view * width) +
           row;
}

static std::vector<float> pack_logical_tensor(
    const std::vector<float>& logical,
    int family_id,
    int width) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    std::vector<float> physical(logical.size(), 0.0f);
    const size_t matrix_count = logical.size() / matrix_elements;
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (size_t element = 0; element < matrix_elements; ++element) {
            physical[tensor_offset_for_logical(
                family_id, matrix, element, width)] =
                logical[matrix * matrix_elements + element];
        }
    }
    return physical;
}

static std::vector<float> unpack_logical_tensor(
    const std::vector<float>& physical,
    int family_id,
    int width) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    std::vector<float> logical(physical.size(), 0.0f);
    const size_t matrix_count = physical.size() / matrix_elements;
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (size_t element = 0; element < matrix_elements; ++element) {
            logical[matrix * matrix_elements + element] =
                physical[tensor_offset_for_logical(
                    family_id, matrix, element, width)];
        }
    }
    return logical;
}

static LlmcOptimizerPlan batched_plan(int width, int layers) {
    LlmcOptimizerPlan plan;
    llmc_optimizer_plan_reset(&plan);
    plan.built = true;
    plan.num_layers = layers;
    plan.channels = width;
    plan.normuon_parameter_type_count = 1;
    plan.normuon_view_count =
        layers * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    return plan;
}

static LlmcOptimizerParameterType batched_parameter_type(
    int family_id,
    int width,
    int layers) {
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 10 : 12;
    parameter_type.name =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? "fcw_batched_test"
            : "fcprojw_batched_test";
    parameter_type.family_id =
        static_cast<LlmcOptimizerFamilyId>(family_id);
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group =
        LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = layers;
    parameter_type.layer_elements =
        LLMC_NORMUON_VIEWS_PER_MLP_MATRIX *
        static_cast<size_t>(width) * width;
    parameter_type.tensor_elements =
        static_cast<size_t>(layers) * parameter_type.layer_elements;
    parameter_type.matrix_width = width;
    parameter_type.views_per_layer = LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    parameter_type.enumerate_matrix_view =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? llmc_enumerate_mlp_wup_view
            : llmc_enumerate_mlp_wdown_view;
    return parameter_type;
}

struct BatchedBuffers {
    explicit BatchedBuffers(size_t count) : elements(count) {
        cudaCheck(cudaMalloc(&parameter, elements * sizeof(floatX)));
        cudaCheck(cudaMalloc(&gradient, elements * sizeof(floatX)));
        cudaCheck(cudaMalloc(&momentum, elements * sizeof(float)));
        cudaCheck(cudaMalloc(&second_moment, elements * sizeof(float)));
        cudaCheck(cudaMalloc(&master, elements * sizeof(float)));
    }

    ~BatchedBuffers() {
        cudaFree(parameter);
        cudaFree(gradient);
        cudaFree(momentum);
        cudaFree(second_moment);
        cudaFree(master);
    }

    void load(
        const std::vector<float>& master_host,
        const std::vector<float>& gradient_host,
        const std::vector<float>& momentum_host,
        const std::vector<float>& second_host) {
        batched_copy_to_device(parameter, quantize_floatx(master_host));
        batched_copy_to_device(gradient, quantize_floatx(gradient_host));
        batched_copy_to_device(momentum, momentum_host);
        batched_copy_to_device(second_moment, second_host);
        batched_copy_to_device(master, master_host);
    }

    size_t elements;
    floatX* parameter = nullptr;
    floatX* gradient = nullptr;
    float* momentum = nullptr;
    float* second_moment = nullptr;
    float* master = nullptr;
};

struct ScratchResult {
    bool succeeded = false;
    bool second_padding_unchanged = false;
    std::vector<float> direction;
    std::vector<float> momentum;
    std::vector<float> second;
    std::vector<float> master;
    std::vector<floatX> parameter;
};

static void make_scratch_inputs(
    int width,
    int layers,
    std::vector<float>* gradient,
    std::vector<float>* momentum,
    std::vector<float>* master,
    std::vector<float>* second_axes) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t matrix_count =
        static_cast<size_t>(layers) * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    gradient->resize(matrix_count * matrix_elements);
    momentum->resize(matrix_count * matrix_elements);
    master->resize(matrix_count * matrix_elements);
    second_axes->resize(matrix_count * width);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (size_t element = 0; element < matrix_elements; ++element) {
            const size_t index = matrix * matrix_elements + element;
            (*gradient)[index] =
                0.021f * std::sin(0.17f * static_cast<float>(index + 1)) +
                0.0007f * static_cast<float>((element + matrix) % 11U);
            (*momentum)[index] =
                0.006f * std::cos(0.11f * static_cast<float>(index + 3));
            (*master)[index] =
                -0.18f + 0.0009f * static_cast<float>(index % 257U);
        }
        for (int row = 0; row < width; ++row) {
            (*second_axes)[matrix * width + row] =
                0.014f + 0.0004f * static_cast<float>(row + matrix);
        }
    }
}

static ScratchResult run_batched_scratch(
    int family_id,
    bool inject_nonfinite) {
    constexpr int width = 8;
    constexpr int layers = 2;
    constexpr float learning_rate = 0.0125f;
    constexpr float gradient_scale = 0.75f;
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t matrix_count =
        static_cast<size_t>(layers) * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t tensor_elements = matrix_count * matrix_elements;

    std::vector<float> gradient_logical;
    std::vector<float> momentum_logical;
    std::vector<float> master_logical;
    std::vector<float> second_axes;
    make_scratch_inputs(
        width,
        layers,
        &gradient_logical,
        &momentum_logical,
        &master_logical,
        &second_axes);
    if (inject_nonfinite) {
        gradient_logical[13] = NAN;
    }
    const std::vector<float> gradient_physical =
        pack_logical_tensor(gradient_logical, family_id, width);
    const std::vector<float> momentum_physical =
        pack_logical_tensor(momentum_logical, family_id, width);
    const std::vector<float> master_physical =
        pack_logical_tensor(master_logical, family_id, width);
    constexpr float second_sentinel = -9.25f;
    std::vector<float> second_physical(tensor_elements, second_sentinel);
    std::vector<uint8_t> second_used(tensor_elements, 0U);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (int row = 0; row < width; ++row) {
            const size_t offset = second_offset_for_logical(
                family_id, matrix, static_cast<size_t>(row), width);
            second_physical[offset] = second_axes[matrix * width + row];
            second_used[offset] = 1U;
        }
    }

    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode = LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
    llmc_normuon_resolve_schedules(&config);
    LlmcOptimizerPlan plan = batched_plan(width, layers);
    LlmcOptimizerParameterType parameter_type =
        batched_parameter_type(family_id, width, layers);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "batched scratch runtime allocates");
    BatchedBuffers buffers(tensor_elements);
    buffers.load(
        master_physical,
        gradient_physical,
        momentum_physical,
        second_physical);

    ScratchResult result;
    result.succeeded = llmc_normuon_update_parameter_type_batched_bf16(
        &runtime,
        cublas_handle,
        main_stream,
        buffers.parameter,
        buffers.gradient,
        buffers.momentum,
        buffers.second_moment,
        buffers.master,
        &parameter_type,
        &config,
        learning_rate,
        gradient_scale,
        0U);
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<float> momentum_after =
        batched_copy_from_device(buffers.momentum, tensor_elements);
    const std::vector<float> second_after =
        batched_copy_from_device(buffers.second_moment, tensor_elements);
    const std::vector<float> master_after =
        batched_copy_from_device(buffers.master, tensor_elements);
    result.direction = batched_copy_from_device(
        runtime.matrix[0], tensor_elements);
    result.momentum = unpack_logical_tensor(
        momentum_after, family_id, width);
    result.master = unpack_logical_tensor(master_after, family_id, width);
    result.second.resize(matrix_count * width);
    result.second_padding_unchanged = true;
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (int row = 0; row < width; ++row) {
            result.second[matrix * width + row] = second_after[
                second_offset_for_logical(
                    family_id, matrix, static_cast<size_t>(row), width)];
        }
    }
    for (size_t index = 0; index < tensor_elements; ++index) {
        if (second_used[index] == 0U &&
            second_after[index] != second_sentinel) {
            result.second_padding_unchanged = false;
        }
    }
    result.parameter = batched_copy_from_device(
        buffers.parameter, tensor_elements);
    llmc_normuon_runtime_free(&runtime);
    return result;
}

static void test_execution_mode_and_workspace() {
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    BATCHED_CHECK(
        config.optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW,
        "all-AdamW remains the optimizer default");
    BATCHED_CHECK(
        config.execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED,
        "explicit NorMuon selection defaults to the batched execution path");
    LlmcNormuonExecutionMode parsed = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    BATCHED_CHECK(
        llmc_parse_normuon_execution_mode("fp32_reference", &parsed) &&
            parsed == LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
        "fp32_reference execution mode parses");
    BATCHED_CHECK(
        llmc_parse_normuon_execution_mode("bf16_batched", &parsed) &&
            parsed == LLMC_NORMUON_EXECUTION_BF16_BATCHED,
        "bf16_batched execution mode parses");
    BATCHED_CHECK(
        !llmc_parse_normuon_execution_mode("silent_fallback", &parsed),
        "unknown execution mode is rejected");

    LlmcNormuonTrackerRetractionMode retraction =
        LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    BATCHED_CHECK(
        config.retraction_mode ==
            LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        "Newton-Schulz remains the default tracker retraction");
    BATCHED_CHECK(
        llmc_parse_normuon_tracker_retraction_mode(
            "commuted_canonical_stage2", &retraction) &&
            retraction ==
                LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2,
        "commuted canonical-stage-2 mode parses explicitly");
    BATCHED_CHECK(
        llmc_parse_normuon_tracker_retraction_mode("1", &retraction) &&
            retraction == LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        "legacy retraction value 1 maps to Newton-Schulz");
    BATCHED_CHECK(
        !llmc_parse_normuon_tracker_retraction_mode(
            "silent_substitution", &retraction),
        "unknown tracker retraction mode is rejected");
    LlmcNormuonConfig commuted = config;
    commuted.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    commuted.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    commuted.correction_policy =
        LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    commuted.correction_iterations = 2U;
    commuted.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    char config_error[256];
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &commuted, config_error, sizeof(config_error)),
        "commuted mode accepts its exact canonical2 tracker contract");
    commuted.correction_iterations = 3U;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &commuted, config_error, sizeof(config_error)),
        "commuted mode rejects a substituted correction stage count");
    commuted.correction_iterations = 2U;
    commuted.correction_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &commuted, config_error, sizeof(config_error)),
        "commuted mode rejects a substituted correction policy");

    constexpr int width = 7;
    constexpr int layers = 3;
    LlmcOptimizerPlan plan = batched_plan(width, layers);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "BF16 batch workspace allocates");
    BATCHED_CHECK(
        runtime.batch_matrix_capacity == 12U &&
            runtime.batch_total_elements == 12U * 49U &&
            runtime.batch_float_matrix_count == 2U &&
            runtime.matrix[0] != nullptr &&
            runtime.matrix[1] != nullptr &&
            runtime.matrix[2] == nullptr &&
            runtime.batch_bf16[0] != nullptr &&
            runtime.batch_bf16[1] != nullptr,
        "BF16 scratch workspace uses two FP32 matrices plus packed BF16 storage");
    const size_t scratch_workspace_bytes = runtime.workspace_bytes;
    llmc_normuon_runtime_free(&runtime);

    void* borrowed_workspace = nullptr;
    cudaCheck(cudaMalloc(&borrowed_workspace, scratch_workspace_bytes));
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(
            &runtime,
            &plan,
            &config,
            borrowed_workspace,
            scratch_workspace_bytes) &&
            runtime.workspace_is_borrowed &&
            runtime.workspace == borrowed_workspace &&
            runtime.workspace_allocation == nullptr,
        "runtime can borrow an exactly-sized external workspace");
    llmc_normuon_runtime_free(&runtime);
    cudaCheck(cudaMemset(borrowed_workspace, 0, scratch_workspace_bytes));
    cudaCheck(cudaFree(borrowed_workspace));

    config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "FP32 reference workspace allocates");
    BATCHED_CHECK(
        runtime.batch_matrix_capacity == 1U &&
            runtime.batch_total_elements == 49U &&
            runtime.batch_float_matrix_count == 5U &&
            runtime.batch_bf16[0] == nullptr,
        "FP32 reference workspace remains single-view sized");
    llmc_normuon_runtime_free(&runtime);
}

static void test_batched_layout_scratch_and_guard() {
    constexpr int width = 8;
    constexpr int layers = 2;
    constexpr float learning_rate = 0.0125f;
    constexpr float gradient_scale = 0.75f;
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t matrix_count =
        static_cast<size_t>(layers) * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;

    ScratchResult wup = run_batched_scratch(
        LLMC_OPTIMIZER_FAMILY_MLP_WUP, false);
    ScratchResult wdown = run_batched_scratch(
        LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, false);
    BATCHED_CHECK(wup.succeeded, "batched Wup scratch update succeeds");
    BATCHED_CHECK(wdown.succeeded, "batched Wdown scratch update succeeds");
    BATCHED_CHECK(
        wup.second_padding_unchanged && wdown.second_padding_unchanged,
        "batched second-moment mappings do not touch padding storage");
    BATCHED_CHECK(
        batched_max_abs_difference(wup.direction, wdown.direction) < 1.0e-6f,
        "Wup row blocks and Wdown column panels produce identical logical directions");
    BATCHED_CHECK(
        batched_max_abs_difference(wup.momentum, wdown.momentum) < 1.0e-7f,
        "Wup and Wdown momentum mappings are logically identical");
    BATCHED_CHECK(
        batched_max_abs_difference(wup.second, wdown.second) < 1.0e-7f,
        "Wup and Wdown second-moment mappings are logically identical");
    BATCHED_CHECK(
        batched_max_abs_difference(wup.master, wdown.master) < 1.0e-6f,
        "Wup and Wdown master updates are logically identical");
    BATCHED_CHECK(
        batched_all_finite(wup.direction) && batched_all_finite(wup.master),
        "batched scratch direction and master weights are finite");

    std::vector<float> gradient;
    std::vector<float> momentum_initial;
    std::vector<float> master_initial;
    std::vector<float> second_initial;
    make_scratch_inputs(
        width,
        layers,
        &gradient,
        &momentum_initial,
        &master_initial,
        &second_initial);
    std::vector<float> first_gradient(
        gradient.begin(), gradient.begin() + matrix_elements);
    for (float& value : first_gradient) {
        value = round_to_floatx(value);
    }
    std::vector<float> first_momentum(
        momentum_initial.begin(), momentum_initial.begin() + matrix_elements);
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    std::vector<float> direction_reference =
        prepare_direction_reference_batched(
            first_gradient, &first_momentum, config, gradient_scale);
    apply_bf16_polynomial_reference(
        &direction_reference,
        width,
        LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
        config.refresh_schedule);
    std::vector<float> first_direction(
        wup.direction.begin(), wup.direction.begin() + matrix_elements);
    const float polynomial_error = batched_max_abs_difference(
        first_direction, direction_reference);
    printf("batched scratch BF16 reference max_abs_error: %.9g\n", polynomial_error);
    BATCHED_CHECK(
        polynomial_error < 1.5e-2f,
        "five-stage BF16 batch polynomial matches direct downcast-aware reference");

    std::vector<float> master_reference = master_initial;
    std::vector<float> second_reference = second_initial;
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        std::vector<float> local_direction(
            wup.direction.begin() + matrix * matrix_elements,
            wup.direction.begin() + (matrix + 1U) * matrix_elements);
        std::vector<float> local_second(
            second_reference.begin() + matrix * width,
            second_reference.begin() + (matrix + 1U) * width);
        std::vector<float> local_master(
            master_reference.begin() + matrix * matrix_elements,
            master_reference.begin() + (matrix + 1U) * matrix_elements);
        finalize_reference_batched(
            local_direction,
            &local_second,
            &local_master,
            width,
            config,
            learning_rate);
        std::copy(
            local_second.begin(),
            local_second.end(),
            second_reference.begin() + matrix * width);
        std::copy(
            local_master.begin(),
            local_master.end(),
            master_reference.begin() + matrix * matrix_elements);
    }
    BATCHED_CHECK(
        batched_max_abs_difference(wup.second, second_reference) < 2.0e-5f,
        "batched second moment matches direct FP32 finalization reference");
    BATCHED_CHECK(
        batched_max_abs_difference(wup.master, master_reference) < 3.0e-5f,
        "batched FP32 master update matches direct finalization reference");

    ScratchResult duplicate = run_batched_scratch(
        LLMC_OPTIMIZER_FAMILY_MLP_WUP, false);
    BATCHED_CHECK(
        duplicate.parameter.size() == wup.parameter.size() &&
            memcmp(
                duplicate.parameter.data(),
                wup.parameter.data(),
                wup.parameter.size() * sizeof(floatX)) == 0,
        "batched tuple-keyed stochastic rounding is deterministic");

    ScratchResult nonfinite = run_batched_scratch(
        LLMC_OPTIMIZER_FAMILY_MLP_WUP, true);
    BATCHED_CHECK(!nonfinite.succeeded, "batched nonfinite gradient is rejected");
    BATCHED_CHECK(
        memcmp(
            nonfinite.master.data(),
            master_initial.data(),
            master_initial.size() * sizeof(float)) == 0,
        "batched nonfinite guard prevents FP32 master mutation");
}

static std::vector<float> tracker_logical_gradient(
    int width,
    int layers,
    int step) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t matrix_count =
        static_cast<size_t>(layers) * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    std::vector<float> gradient(matrix_count * matrix_elements);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (size_t element = 0; element < matrix_elements; ++element) {
            const size_t index = matrix * matrix_elements + element;
            const size_t row = element / static_cast<size_t>(width);
            const size_t column = element % static_cast<size_t>(width);
            gradient[index] =
                0.018f * std::sin(
                    0.31f * static_cast<float>((row + 1U) * (column + 2U)) +
                    0.19f * static_cast<float>(matrix) +
                    0.61f * static_cast<float>(step)) +
                0.012f * std::cos(
                    0.23f * static_cast<float>((row + 3U) * (column + 1U)) +
                    0.07f * static_cast<float>(matrix + step)) +
                (row == column ? 0.025f : 0.0f);
        }
    }
    return gradient;
}

static std::vector<float> selected_q_view(
    const std::vector<float>& q,
    int family_id,
    int layer,
    int view,
    int width) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t family_offset =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 4U;
    const size_t q_index =
        static_cast<size_t>(layer) * LLMC_NORMUON_VIEWS_PER_LAYER +
        family_offset + static_cast<size_t>(view);
    return std::vector<float>(
        q.begin() + q_index * matrix_elements,
        q.begin() + (q_index + 1U) * matrix_elements);
}

static void test_batched_tracker_and_checkpoint() {
    constexpr int width = 8;
    constexpr int layers = 2;
    constexpr float learning_rate = 0.01f;
    constexpr int family_id = LLMC_OPTIMIZER_FAMILY_MLP_WUP;
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t matrix_count =
        static_cast<size_t>(layers) * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t tensor_elements = matrix_count * matrix_elements;

    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    config.refresh_interval = 3U;
    config.refresh_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config.correction_policy = LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config.correction_iterations = 2U;
    config.correction_gain = 1.0f;
    config.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ;
    llmc_normuon_resolve_schedules(&config);
    LlmcOptimizerPlan plan = batched_plan(width, layers);
    LlmcOptimizerParameterType parameter_type =
        batched_parameter_type(family_id, width, layers);

    std::vector<float> master_logical(tensor_elements);
    std::vector<float> momentum_logical(tensor_elements);
    std::vector<float> second_physical(tensor_elements, -4.0f);
    for (size_t index = 0; index < tensor_elements; ++index) {
        master_logical[index] =
            0.14f - 0.0008f * static_cast<float>(index % 193U);
        momentum_logical[index] =
            0.004f * std::cos(0.09f * static_cast<float>(index + 1U));
    }
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) {
        for (int row = 0; row < width; ++row) {
            second_physical[second_offset_for_logical(
                family_id, matrix, static_cast<size_t>(row), width)] =
                0.012f + 0.0003f * static_cast<float>(matrix + row);
        }
    }
    const std::vector<float> master_physical =
        pack_logical_tensor(master_logical, family_id, width);
    const std::vector<float> momentum_physical =
        pack_logical_tensor(momentum_logical, family_id, width);
    BatchedBuffers buffers(tensor_elements);
    std::vector<float> gradient0 =
        tracker_logical_gradient(width, layers, 0);
    buffers.load(
        master_physical,
        pack_logical_tensor(gradient0, family_id, width),
        momentum_physical,
        second_physical);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "batched tracker runtime allocates");

    std::vector<float> first_momentum(
        momentum_logical.begin(), momentum_logical.begin() + matrix_elements);
    std::vector<float> first_gradient0(
        gradient0.begin(), gradient0.begin() + matrix_elements);
    for (float& value : first_gradient0) {
        value = round_to_floatx(value);
    }
    std::vector<float> normalized0_reference = prepare_direction_reference_batched(
        first_gradient0, &first_momentum, config, 1.0f);
    const float pre_refresh_orthogonality =
        batched_orthogonality_error(normalized0_reference, width);
    std::vector<float> q0_reference = normalized0_reference;
    apply_bf16_polynomial_reference(
        &q0_reference,
        width,
        LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
        config.refresh_schedule);
    BATCHED_CHECK(
        llmc_normuon_update_parameter_type_batched_bf16(
            &runtime,
            cublas_handle,
            main_stream,
            buffers.parameter,
            buffers.gradient,
            buffers.momentum,
            buffers.second_moment,
            buffers.master,
            &parameter_type,
            &config,
            learning_rate,
            1.0f,
            0U),
        "batched tracker refresh at step zero succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    std::vector<float> q_host = batched_copy_from_device(
        runtime.tracked_q, runtime.tracked_q_elements);
    std::vector<float> q0 = selected_q_view(
        q_host, family_id, 0, 0, width);
    const float refresh_error = batched_max_abs_difference(q0, q0_reference);
    printf("batched tracker refresh BF16 reference max_abs_error: %.9g\n", refresh_error);
    BATCHED_CHECK(
        refresh_error < 1.5e-2f,
        "batched tracker refresh matches five-stage BF16 reference");
    float max_refresh_orthogonality = 0.0f;
    for (int layer = 0; layer < layers; ++layer) {
        for (int view = 0; view < LLMC_NORMUON_VIEWS_PER_MLP_MATRIX; ++view) {
            const size_t q_index =
                static_cast<size_t>(layer) * LLMC_NORMUON_VIEWS_PER_LAYER +
                view;
            BATCHED_CHECK(
                runtime.q_valid[q_index] == 1U &&
                    runtime.refresh_count[q_index] == 1U &&
                    runtime.last_refresh_step[q_index] == 0,
                "all selected tracker views record the step-zero refresh");
            const std::vector<float> matrix = selected_q_view(
                q_host, family_id, layer, view, width);
            const float orthogonality =
                batched_orthogonality_error(matrix, width);
            max_refresh_orthogonality =
                std::max(max_refresh_orthogonality, orthogonality);
            BATCHED_CHECK(
                batched_all_finite(matrix) &&
                    orthogonality < 1.25f,
                "refreshed batched Q is finite and approximately orthogonal");
        }
        for (int view = 4; view < LLMC_NORMUON_VIEWS_PER_LAYER; ++view) {
            const size_t q_index =
                static_cast<size_t>(layer) * LLMC_NORMUON_VIEWS_PER_LAYER +
                view;
            BATCHED_CHECK(
                runtime.q_valid[q_index] == 0U &&
                    runtime.refresh_count[q_index] == 0U,
                "unselected family tracker metadata remains untouched");
        }
    }
    printf(
        "batched tracker max refresh orthogonality residual: %.9g\n",
        max_refresh_orthogonality);
    BATCHED_CHECK(
        batched_orthogonality_error(q0, width) < pre_refresh_orthogonality,
        "five-stage stock refresh improves Q orthogonality");

    std::vector<float> gradient1 =
        tracker_logical_gradient(width, layers, 1);
    batched_copy_to_device(
        buffers.gradient,
        quantize_floatx(pack_logical_tensor(gradient1, family_id, width)));
    std::vector<float> first_gradient1(
        gradient1.begin(), gradient1.begin() + matrix_elements);
    for (float& value : first_gradient1) {
        value = round_to_floatx(value);
    }
    std::vector<float> normalized1 = prepare_direction_reference_batched(
        first_gradient1, &first_momentum, config, 1.0f);
    const std::vector<float> q1_reference =
        tracker_correction_bf16_reference(
            q0, normalized1, width, config, true);
    const std::vector<float> q1_wrong_order =
        tracker_correction_bf16_reference(
            q0, normalized1, width, config, false);
    BATCHED_CHECK(
        llmc_normuon_update_parameter_type_batched_bf16(
            &runtime,
            cublas_handle,
            main_stream,
            buffers.parameter,
            buffers.gradient,
            buffers.momentum,
            buffers.second_moment,
            buffers.master,
            &parameter_type,
            &config,
            learning_rate,
            1.0f,
            1U),
        "batched two-stage tracker correction succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    q_host = batched_copy_from_device(
        runtime.tracked_q, runtime.tracked_q_elements);
    std::vector<float> q1 = selected_q_view(
        q_host, family_id, 0, 0, width);
    const float correction_error =
        batched_max_abs_difference(q1, q1_reference);
    const float wrong_order_error =
        batched_max_abs_difference(q1, q1_wrong_order);
    printf(
        "batched tracker correction BF16 reference max_abs_error: %.9g "
        "(left-form %.9g)\n",
        correction_error,
        wrong_order_error);
    BATCHED_CHECK(
        correction_error < 2.5e-2f,
        "batched correction and native right retraction match BF16 reference");
    BATCHED_CHECK(
        correction_error < wrong_order_error,
        "batched retraction uses native D(3I-D^TD) operand order");
    BATCHED_CHECK(
        batched_all_finite(q1) &&
            batched_orthogonality_error(q1, width) < 0.65f,
        "corrected batched Q remains finite and approximately orthogonal");
    for (int layer = 0; layer < layers; ++layer) {
        for (int view = 0; view < LLMC_NORMUON_VIEWS_PER_MLP_MATRIX; ++view) {
            const size_t q_index =
                static_cast<size_t>(layer) * LLMC_NORMUON_VIEWS_PER_LAYER +
                view;
            BATCHED_CHECK(
                runtime.refresh_count[q_index] == 1U &&
                    runtime.last_refresh_step[q_index] == 0,
                "correction does not advance tracker refresh phase");
        }
    }

    const char* current_path = "build/test_normuon_batched_v3.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            current_path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "batched tracker v3 companion saves");
    LlmcNormuonCompanionInfo info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(current_path, &info) &&
            info.config.execution_mode ==
                LLMC_NORMUON_EXECUTION_BF16_BATCHED,
        "v3 companion records the exact BF16 execution mode");
    LlmcNormuonRuntime resumed;
    llmc_normuon_runtime_reset(&resumed);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&resumed, &plan, &config) &&
            llmc_normuon_load_companion(
                current_path, 2, 1, 0, &plan, &config, &resumed, main_stream),
        "v3 batched tracker state resumes exactly");
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<float> resumed_q = batched_copy_from_device(
        resumed.tracked_q, resumed.tracked_q_elements);
    BATCHED_CHECK(
        resumed_q.size() == q_host.size() &&
            memcmp(
                resumed_q.data(),
                q_host.data(),
                q_host.size() * sizeof(float)) == 0 &&
            memcmp(
                resumed.q_valid,
                runtime.q_valid,
                runtime.tracked_q_view_count * sizeof(uint8_t)) == 0 &&
            memcmp(
                resumed.refresh_count,
                runtime.refresh_count,
                runtime.tracked_q_view_count * sizeof(uint64_t)) == 0 &&
            memcmp(
                resumed.last_refresh_step,
                runtime.last_refresh_step,
                runtime.tracked_q_view_count * sizeof(int64_t)) == 0,
        "v3 resume preserves Q and tracker phase bit-exactly");
    LlmcNormuonConfig mode_mismatch = config;
    mode_mismatch.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            current_path,
            2,
            1,
            0,
            &plan,
            &mode_mismatch,
            &resumed,
            main_stream),
        "resume rejects an execution-mode mismatch");

    const char* v2_path = "build/test_normuon_batched_v2.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            v2_path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "v2 compatibility fixture companion saves");
    FILE* v2_file = fopen(v2_path, "r+b");
    BATCHED_CHECK(v2_file != nullptr, "v2 compatibility fixture opens");
    if (v2_file != nullptr) {
        const int v2 = static_cast<int>(
            LLMC_NORMUON_COMPANION_VERSION_EXECUTION_MODE);
        fseek(v2_file, sizeof(int), SEEK_SET);
        fwrite(&v2, sizeof(v2), 1U, v2_file);
        fclose(v2_file);
    }
    LlmcNormuonCompanionInfo v2_info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(v2_path, &v2_info) &&
            v2_info.config.execution_mode ==
                LLMC_NORMUON_EXECUTION_BF16_BATCHED &&
            v2_info.config.retraction_mode ==
                LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        "v2 companion decodes execution mode and legacy retraction exactly");
    BATCHED_CHECK(
        llmc_normuon_load_companion(
            v2_path,
            2,
            1,
            0,
            &plan,
            &config,
            &resumed,
            main_stream),
        "v2 batched tracker state remains loadable");

    const char* v1_path = "build/test_normuon_batched_v1.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            v1_path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "compatibility fixture companion saves");
    FILE* v1_file = fopen(v1_path, "r+b");
    BATCHED_CHECK(v1_file != nullptr, "compatibility fixture opens");
    if (v1_file != nullptr) {
        const int v1 =
            static_cast<int>(LLMC_NORMUON_COMPANION_VERSION_FP32_ONLY);
        const int ignored_mode = 77;
        fseek(v1_file, sizeof(int), SEEK_SET);
        fwrite(&v1, sizeof(v1), 1U, v1_file);
        fseek(v1_file, 31L * static_cast<long>(sizeof(int)), SEEK_SET);
        fwrite(&ignored_mode, sizeof(ignored_mode), 1U, v1_file);
        fclose(v1_file);
    }
    LlmcNormuonCompanionInfo v1_info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(v1_path, &v1_info) &&
            v1_info.config.execution_mode ==
                LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
        "v1 companion decodes as the historical FP32 execution mode");
    LlmcNormuonConfig v1_config = config;
    v1_config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
    LlmcNormuonRuntime v1_runtime;
    llmc_normuon_runtime_reset(&v1_runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&v1_runtime, &plan, &v1_config) &&
            llmc_normuon_load_companion(
                v1_path,
                2,
                1,
                0,
                &plan,
                &v1_config,
                &v1_runtime,
                main_stream),
        "v1 FP32 companion remains loadable");

    std::vector<float> gradient3 =
        tracker_logical_gradient(width, layers, 3);
    batched_copy_to_device(
        buffers.gradient,
        quantize_floatx(pack_logical_tensor(gradient3, family_id, width)));
    BATCHED_CHECK(
        llmc_normuon_update_parameter_type_batched_bf16(
            &runtime,
            cublas_handle,
            main_stream,
            buffers.parameter,
            buffers.gradient,
            buffers.momentum,
            buffers.second_moment,
            buffers.master,
            &parameter_type,
            &config,
            learning_rate,
            1.0f,
            3U),
        "batched tracker refresh at step three succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    for (int layer = 0; layer < layers; ++layer) {
        for (int view = 0; view < LLMC_NORMUON_VIEWS_PER_MLP_MATRIX; ++view) {
            const size_t q_index =
                static_cast<size_t>(layer) * LLMC_NORMUON_VIEWS_PER_LAYER +
                view;
            BATCHED_CHECK(
                runtime.refresh_count[q_index] == 2U &&
                    runtime.last_refresh_step[q_index] == 3,
                "batched refresh cadence is exactly steps 0,3,6,...");
        }
    }

    remove(current_path);
    remove(v2_path);
    remove(v1_path);
    llmc_normuon_runtime_free(&v1_runtime);
    llmc_normuon_runtime_free(&resumed);
    llmc_normuon_runtime_free(&runtime);
}

static void test_batched_commuted_tracker() {
    constexpr int width = 8;
    constexpr int layers = 1;
    constexpr float learning_rate = 0.01f;
    constexpr int family_id = LLMC_OPTIMIZER_FAMILY_MLP_WUP;
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t matrix_count =
        static_cast<size_t>(layers) * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t tensor_elements = matrix_count * matrix_elements;

    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    config.refresh_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config.correction_policy =
        LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config.refresh_interval = 3U;
    config.correction_iterations = 2U;
    config.correction_gain = 1.0f;
    config.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    char config_error[256];
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &config, config_error, sizeof(config_error)),
        "commuted tracker configuration validates");

    LlmcOptimizerPlan plan = batched_plan(width, layers);
    LlmcOptimizerParameterType parameter_type =
        batched_parameter_type(family_id, width, layers);
    std::vector<float> master_logical(tensor_elements);
    std::vector<float> momentum_logical(tensor_elements, 0.0f);
    std::vector<float> second_physical(tensor_elements, 0.01f);
    for (size_t index = 0; index < tensor_elements; ++index) {
        master_logical[index] =
            0.1f - 0.0007f * static_cast<float>(index % 127U);
    }
    std::vector<float> gradient0 =
        tracker_logical_gradient(width, layers, 0);
    BatchedBuffers buffers(tensor_elements);
    buffers.load(
        pack_logical_tensor(master_logical, family_id, width),
        pack_logical_tensor(gradient0, family_id, width),
        pack_logical_tensor(momentum_logical, family_id, width),
        second_physical);

    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "commuted tracker runtime allocates");
    BATCHED_CHECK(
        runtime.batch_float_matrix_count == 3U &&
            runtime.matrix[2] != nullptr &&
            runtime.matrix[3] == nullptr,
        "batched tracker workspace uses only three FP32 matrix panels");

    std::vector<float> first_momentum(matrix_elements, 0.0f);
    std::vector<float> first_gradient0(
        gradient0.begin(), gradient0.begin() + matrix_elements);
    for (float& value : first_gradient0) {
        value = round_to_floatx(value);
    }
    (void)prepare_direction_reference_batched(
        first_gradient0, &first_momentum, config, 1.0f);
    BATCHED_CHECK(
        llmc_normuon_update_parameter_type_batched_bf16(
            &runtime,
            cublas_handle,
            main_stream,
            buffers.parameter,
            buffers.gradient,
            buffers.momentum,
            buffers.second_moment,
            buffers.master,
            &parameter_type,
            &config,
            learning_rate,
            1.0f,
            0U),
        "commuted tracker refresh at step zero succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    std::vector<float> q_host = batched_copy_from_device(
        runtime.tracked_q, runtime.tracked_q_elements);
    const std::vector<float> q0 = selected_q_view(
        q_host, family_id, 0, 0, width);

    std::vector<float> gradient1 =
        tracker_logical_gradient(width, layers, 1);
    batched_copy_to_device(
        buffers.gradient,
        quantize_floatx(pack_logical_tensor(gradient1, family_id, width)));
    std::vector<float> first_gradient1(
        gradient1.begin(), gradient1.begin() + matrix_elements);
    for (float& value : first_gradient1) {
        value = round_to_floatx(value);
    }
    const std::vector<float> normalized1 =
        prepare_direction_reference_batched(
            first_gradient1, &first_momentum, config, 1.0f);
    const std::vector<float> q1_reference =
        tracker_correction_bf16_reference(
            q0, normalized1, width, config, true);
    BATCHED_CHECK(
        llmc_normuon_update_parameter_type_batched_bf16(
            &runtime,
            cublas_handle,
            main_stream,
            buffers.parameter,
            buffers.gradient,
            buffers.momentum,
            buffers.second_moment,
            buffers.master,
            &parameter_type,
            &config,
            learning_rate,
            1.0f,
            1U),
        "commuted canonical2 tracker correction succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    q_host = batched_copy_from_device(
        runtime.tracked_q, runtime.tracked_q_elements);
    const std::vector<float> q1 = selected_q_view(
        q_host, family_id, 0, 0, width);
    const float correction_error =
        batched_max_abs_difference(q1, q1_reference);
    printf(
        "batched commuted tracker BF16 reference max_abs_error: %.9g\n",
        correction_error);
    BATCHED_CHECK(
        correction_error < 2.5e-2f,
        "commuted stage1-product-stage2 trajectory matches BF16 reference");
    BATCHED_CHECK(
        batched_all_finite(q1) &&
            batched_orthogonality_error(q1, width) < 0.75f,
        "commuted corrected Q is finite and approximately orthogonal");

    const char* path = "build/test_normuon_batched_v3_commuted.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "v3 commuted tracker companion saves");
    LlmcNormuonCompanionInfo info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(path, &info) &&
            info.config.retraction_mode ==
                LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2,
        "v3 companion records the exact commuted retraction mode");
    LlmcNormuonConfig mode_mismatch = config;
    mode_mismatch.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            path, 2, 1, 0, &plan, &mode_mismatch, &runtime, main_stream),
        "resume rejects a tracker retraction-mode mismatch");
    remove(path);
    llmc_normuon_runtime_free(&runtime);
}

static void test_polynomial_factor_override_preserves_packed_q() {
    constexpr int width = 8;
    constexpr int layers = 1;
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t matrix_count =
        static_cast<size_t>(layers) * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t total_elements = matrix_count * matrix_elements;

    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    LlmcOptimizerPlan plan = batched_plan(width, layers);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    const bool allocated =
        llmc_normuon_runtime_allocate(&runtime, &plan, &config);
    BATCHED_CHECK(
        allocated,
        "packed-Q retention test runtime allocates");
    if (!allocated) {
        llmc_normuon_runtime_free(&runtime);
        return;
    }

    std::vector<float> matrix(total_elements);
    for (size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        for (int row = 0; row < width; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t index =
                    matrix_index * matrix_elements +
                    static_cast<size_t>(row) * width + column;
                matrix[index] =
                    0.001f * static_cast<float>(static_cast<int>(index % 11U) - 5) +
                    (row == column ? 0.25f : 0.0f);
            }
        }
    }
    std::vector<uint16_t> packed_q_sentinel(total_elements);
    for (size_t index = 0; index < total_elements; ++index) {
        packed_q_sentinel[index] =
            static_cast<uint16_t>((0x3c00U + 37U * index) & 0xffffU);
    }
    batched_copy_to_device(runtime.matrix[2], matrix);
    batched_copy_to_device(runtime.batch_bf16[1], packed_q_sentinel);
    const bool applied = llmc_normuon_apply_polynomial_batched_bf16(
        &runtime,
        cublas_handle,
        main_stream,
        runtime.matrix[2],
        runtime.matrix[1],
        width,
        static_cast<int>(matrix_count),
        1U,
        config.correction_schedule,
        reinterpret_cast<uint16_t*>(runtime.matrix[0]));
    BATCHED_CHECK(applied, "polynomial accepts a borrowed packed-factor panel");
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<uint16_t> packed_q_after =
        batched_copy_from_device(runtime.batch_bf16[1], total_elements);
    BATCHED_CHECK(
        packed_q_after == packed_q_sentinel,
        "borrowed polynomial factor leaves the first packed Q bitwise intact");
    BATCHED_CHECK(
        batched_all_finite(
            batched_copy_from_device(runtime.matrix[2], total_elements)),
        "borrowed polynomial-factor output remains finite");
    llmc_normuon_runtime_free(&runtime);
}

int main() {
    char server_ip[2] = "";
    char filesystem_path[2] = "";
    char init_method[4] = "mpi";
    multi_gpu_config = multi_gpu_config_init(
        1, 0, 1, server_ip, filesystem_path, init_method);
    set_zero_configs(&multi_gpu_config, 0, 1);
    common_start(false, false);

    test_execution_mode_and_workspace();
    test_batched_layout_scratch_and_guard();
    test_batched_tracker_and_checkpoint();
    test_polynomial_factor_override_preserves_packed_q();
    test_batched_commuted_tracker();

    GPT2 unused_model = {};
    common_free(unused_model);
    multi_gpu_config_free(&multi_gpu_config);
    if (batched_test_failures == 0) {
        printf("All focused llm.c batched NorMuon tests passed.\n");
        return EXIT_SUCCESS;
    }
    fprintf(
        stderr,
        "%d focused llm.c batched NorMuon tests failed.\n",
        batched_test_failures);
    return EXIT_FAILURE;
}
