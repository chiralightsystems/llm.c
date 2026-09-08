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
    double direction_norm_squared = 0.0;
    for (int row = 0; row < width; ++row) {
        double row_sum = 0.0;
        for (int column = 0; column < width; ++column) {
            const float value =
                direction[static_cast<size_t>(row) * width + column];
            row_sum += static_cast<double>(value) * value;
            direction_norm_squared += static_cast<double>(value) * value;
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
        std::sqrt(std::max(static_cast<float>(direction_norm_squared), config.epsilon)) /
        std::sqrt(std::max(static_cast<float>(normalized_norm_squared), config.epsilon));
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
    float diagonal_scale = 1.0f;
    if (config.correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
        const float symmetric_scale =
            std::sqrt(std::max(static_cast<float>(symmetric_norm_squared), 0.0f)) /
            std::sqrt(static_cast<float>(width));
        const float diagonal_floor = std::max(
            LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale,
            config.epsilon);
        double correction_norm_squared = 0.0;
        for (int row = 0; row < width; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t index =
                    static_cast<size_t>(row) * width + column;
                const float skew_value = skew[index];
                const float row_stiffness = std::max(
                    phase[static_cast<size_t>(row) * width + row],
                    diagonal_floor);
                const float column_stiffness = std::max(
                    phase[static_cast<size_t>(column) * width + column],
                    diagonal_floor);
                const float omega = config.correction_gain * 2.0f * skew_value /
                    (row_stiffness + column_stiffness);
                correction[index] = omega;
                correction_norm_squared +=
                    static_cast<double>(omega) * omega;
            }
        }
        const float raw_norm =
            std::sqrt(std::max(static_cast<float>(correction_norm_squared), 0.0f));
        const float target_norm =
            LLMC_NORMUON_TRACKER_CORRECTION_CAP *
            std::sqrt(static_cast<float>(width));
        diagonal_scale = std::min(
            1.0f, target_norm / (raw_norm + config.epsilon));
    }
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index =
                static_cast<size_t>(row) * width + column;
            float correction_denominator = denominator;
            float correction_numerator = skew[index];
            if (config.correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
                correction[index] =
                    (row == column ? 1.0f : 0.0f) +
                    diagonal_scale * correction[index];
                continue;
            }
            correction[index] =
                (row == column ? 1.0f : 0.0f) +
                config.correction_gain * correction_numerator /
                    correction_denominator;
        }
    }
    const bool commute_canonical_stage2 =
        config.retraction_mode ==
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    apply_polynomial_reference(
        &correction,
        width,
        commute_canonical_stage2 ? 1U : config.correction_iterations,
        config.correction_schedule);
    std::vector<float> direction =
        matrix_multiply(tracked_q, false, correction, false, width);
    if (commute_canonical_stage2) {
        apply_polynomial_reference(
            &direction, width, 1U, config.correction_schedule + 1U);
    } else if (config.retraction_mode ==
               LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ) {
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
    explicit ViewBuffers(int matrix_width, size_t element_count = 0U)
        : width(matrix_width),
          elements(element_count != 0U
                       ? element_count
                       : static_cast<size_t>(matrix_width) * matrix_width) {
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

static LlmcOptimizerParameterType rectangular_parameter_type(
    int width,
    int family_id) {
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 10 : 12;
    parameter_type.name =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? "fcw_rectangular_test"
            : "fcprojw_rectangular_test";
    parameter_type.family_id = static_cast<LlmcOptimizerFamilyId>(family_id);
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group = LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = 1;
    parameter_type.tensor_elements = static_cast<size_t>(4 * width * width);
    parameter_type.layer_elements = parameter_type.tensor_elements;
    parameter_type.matrix_width = static_cast<size_t>(width);
    parameter_type.views_per_layer = 1;
    return parameter_type;
}

static LlmcOptimizerMatrixView rectangular_view(int width, bool wup) {
    LlmcOptimizerMatrixView view = {};
    view.rows = static_cast<size_t>(wup ? 4 * width : width);
    view.columns = static_cast<size_t>(wup ? width : 4 * width);
    view.row_stride = static_cast<size_t>(wup ? width : 4 * width);
    view.column_stride = 1U;
    return view;
}

static void test_gpt2_context_descriptor() {
    GPT2Config legacy = {};
    GPT2Config explicit_legacy = {};
    GPT2Config extended = {};
    GPT2Config midpoint = {};
    GPT2Config rope_midpoint = {};
    GPT2Config rope_xl = {};
    GPT2Config bridged_medium = {};
    GPT2Config bridged_medium_reordered = {};
    GPT2Config dc_bridged_medium = {};
    GPT2Config dc_bridged_medium_reordered = {};
    TEST_CHECK(
        gpt2_config_from_descriptor(&legacy, "d48"),
        "legacy GPT-2 XL descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(&explicit_legacy, "gpt2:d48"),
        "explicit GPT-2 XL descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(&extended, "gpt2:d48:t2048"),
        "explicit GPT-2 XL 2048-context descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(&midpoint, "gpt2:d30:t2048"),
        "explicit GPT-2 midpoint 2048-context descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(
            &rope_midpoint, "gpt2:rope:d30:t2048"),
        "RoPE GPT-2 midpoint descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(&rope_xl, "gpt2:rope:d48:t2048"),
        "RoPE GPT-2 XL descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(
            &bridged_medium, "gpt2:rope:d24:t2048:e4096"),
        "RoPE GPT-2 Medium lexical-bridge descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(
            &bridged_medium_reordered, "gpt2:rope:d24:e4096:t2048"),
        "lexical-width and context suffixes may be reordered");
    TEST_CHECK(
        gpt2_config_from_descriptor(
            &dc_bridged_medium, "gpt2:rope-dc:d24:t2048:e4096"),
        "lowest-plane-DC RoPE lexical-bridge descriptor parses");
    TEST_CHECK(
        gpt2_config_from_descriptor(
            &dc_bridged_medium_reordered,
            "gpt2:rope-dc:d24:e4096:t2048"),
        "lowest-plane-DC descriptor suffixes may be reordered");
    TEST_CHECK(
        legacy.num_layers == 48 && legacy.channels == 1600 &&
            legacy.num_heads == 25 && legacy.max_seq_len == 1024,
        "legacy GPT-2 XL shape is unchanged");
    TEST_CHECK(
        explicit_legacy.num_layers == 48 && explicit_legacy.channels == 1600 &&
            explicit_legacy.num_heads == 25 && explicit_legacy.max_seq_len == 1024,
        "explicit GPT-2 XL default context is unchanged");
    TEST_CHECK(
        extended.num_layers == 48 && extended.channels == 1600 &&
            extended.num_heads == 25 && extended.max_seq_len == 2048,
        "GPT-2 XL context override changes maxT only");
    TEST_CHECK(
        midpoint.num_layers == 30 && midpoint.channels == 1152 &&
            midpoint.num_heads == 18 && midpoint.max_seq_len == 2048,
        "GPT-2 midpoint descriptor selects the requested shape");
    TEST_CHECK(
        rope_midpoint.num_layers == 30 && rope_midpoint.channels == 1152 &&
            rope_midpoint.num_heads == 18 &&
            rope_midpoint.max_seq_len == 2048 &&
            rope_midpoint.position_encoding == LLMC_POSITION_ENCODING_ROPE &&
            rope_midpoint.rope_rotary_dim == 64 &&
            rope_midpoint.rope_theta == 10000.0f &&
            rope_midpoint.rope_lowest_frequency_plane_is_dc == 0,
        "RoPE midpoint uses full-head canonical rotary parameters");
    TEST_CHECK(
        explicit_legacy.position_encoding ==
                LLMC_POSITION_ENCODING_LEARNED_ABSOLUTE &&
            explicit_legacy.rope_rotary_dim == 0 &&
            explicit_legacy.rope_theta == 0.0f &&
            explicit_legacy.rope_lowest_frequency_plane_is_dc == 0,
        "legacy descriptors retain learned absolute positions");
    TEST_CHECK(
        std::fabs(rope_midpoint.initializer_std - 0.02f) < 1.0e-8f &&
            std::fabs(
                rope_midpoint.residual_projection_std -
                0.02f / std::sqrt(60.0f)) < 1.0e-8f,
        "RoPE midpoint records GPT-2 residual-scaled initialization");
    TEST_CHECK(
        bridged_medium.num_layers == 24 &&
            bridged_medium.channels == 1024 &&
            bridged_medium.lexical_channels == 4096 &&
            bridged_medium.num_heads == 16 &&
            bridged_medium.max_seq_len == 2048 &&
            bridged_medium.rope_rotary_dim == 64 &&
            std::fabs(bridged_medium.bridge_projection_std - 0.015625f) <
                1.0e-8f,
        "bridged descriptor separates lexical and residual widths with variance-preserving init");
    TEST_CHECK(
        dc_bridged_medium.num_layers == bridged_medium.num_layers &&
            dc_bridged_medium.channels == bridged_medium.channels &&
            dc_bridged_medium.lexical_channels ==
                bridged_medium.lexical_channels &&
            dc_bridged_medium.num_heads == bridged_medium.num_heads &&
            dc_bridged_medium.max_seq_len == bridged_medium.max_seq_len &&
            dc_bridged_medium.rope_rotary_dim ==
                bridged_medium.rope_rotary_dim &&
            dc_bridged_medium.rope_theta == bridged_medium.rope_theta &&
            dc_bridged_medium.rope_lowest_frequency_plane_is_dc == 1,
        "lowest-plane-DC descriptor changes only the explicit frequency mode");
    TEST_CHECK(
        std::memcmp(
            &bridged_medium,
            &bridged_medium_reordered,
            sizeof(GPT2Config)) == 0,
        "descriptor suffix order does not alter the model schema");
    TEST_CHECK(
        std::memcmp(
            &dc_bridged_medium,
            &dc_bridged_medium_reordered,
            sizeof(GPT2Config)) == 0,
        "lowest-plane-DC descriptor suffix order does not alter the model schema");
    const float legacy_residual_scale = 1.0f / std::sqrt(60.0f);
    const float legacy_residual_std = 0.02f * legacy_residual_scale;
    TEST_CHECK(
        std::memcmp(
            &rope_midpoint.residual_projection_std,
            &legacy_residual_std,
            sizeof(float)) == 0,
        "initializer preserves the legacy residual-scale operation order");

    const char* malformed[] = {
        "gpt2:d48:t", "gpt2:d48:t0", "gpt2:d48:t2048junk",
        "gpt2:d48:x2048", "gpt2:d:t2048", "gpt2:d999:t2048",
        "gpt2:rope:d48:t", "gpt2:rope:d48:t0",
        "gpt2:rope:d48:t2048junk", "gpt2:rope:d999:t2048",
        "gpt2:rope:d24:e", "gpt2:rope:d24:e0",
        "gpt2:rope:d24:e4097", "gpt2:rope:d24:e4096:e2048",
        "gpt2:rope:d24:t2048:t1024", "gpt2:d24:e4096",
        "gpt2:rope-dc:d24:t", "gpt2:rope-dc:d24:t0",
        "gpt2:rope-dc:d24:e0", "gpt2:rope-dc:d999:t2048",
    };
    for (const char* descriptor : malformed) {
        GPT2Config rejected = {};
        TEST_CHECK(
            !gpt2_config_from_descriptor(&rejected, descriptor),
            "malformed GPT-2 context descriptor is rejected");
    }

    explicit_legacy.vocab_size = extended.vocab_size = 50257;
    explicit_legacy.padded_vocab_size = extended.padded_vocab_size = 50304;
    size_t legacy_elements[NUM_PARAMETER_TENSORS];
    size_t legacy_sizeof[NUM_PARAMETER_TENSORS];
    size_t extended_elements[NUM_PARAMETER_TENSORS];
    size_t extended_sizeof[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(legacy_elements, legacy_sizeof, explicit_legacy);
    fill_in_parameter_sizes(extended_elements, extended_sizeof, extended);
    size_t legacy_total = 0;
    size_t extended_total = 0;
    for (int tensor = 0; tensor < NUM_PARAMETER_TENSORS; ++tensor) {
        legacy_total += legacy_elements[tensor];
        extended_total += extended_elements[tensor];
    }
    TEST_CHECK(legacy_total == 1557686400ULL, "GPT-2 XL padded parameter count is stable");
    TEST_CHECK(extended_total == 1559324800ULL, "GPT-2 XL 2048-context parameter count is correct");
    TEST_CHECK(
        extended_total - legacy_total == 1024ULL * 1600ULL,
        "context extension only adds positional embeddings");

    midpoint.vocab_size = 50257;
    midpoint.padded_vocab_size = 50304;
    size_t midpoint_elements[NUM_PARAMETER_TENSORS];
    size_t midpoint_sizeof[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(midpoint_elements, midpoint_sizeof, midpoint);
    size_t midpoint_total = 0;
    for (int tensor = 0; tensor < NUM_PARAMETER_TENSORS; ++tensor) {
        midpoint_total += midpoint_elements[tensor];
    }
    TEST_CHECK(
        midpoint_total == 538518528ULL,
        "GPT-2 midpoint 2048-context padded parameter count is correct");

    rope_midpoint.vocab_size = 50257;
    rope_midpoint.padded_vocab_size = 50304;
    rope_xl.vocab_size = 50257;
    rope_xl.padded_vocab_size = 50304;
    size_t rope_midpoint_elements[NUM_PARAMETER_TENSORS];
    size_t rope_midpoint_sizeof[NUM_PARAMETER_TENSORS];
    size_t rope_xl_elements[NUM_PARAMETER_TENSORS];
    size_t rope_xl_sizeof[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(
        rope_midpoint_elements, rope_midpoint_sizeof, rope_midpoint);
    fill_in_parameter_sizes(rope_xl_elements, rope_xl_sizeof, rope_xl);
    size_t rope_midpoint_total = 0;
    size_t rope_xl_total = 0;
    for (int tensor = 0; tensor < NUM_PARAMETER_TENSORS; ++tensor) {
        rope_midpoint_total += rope_midpoint_elements[tensor];
        rope_xl_total += rope_xl_elements[tensor];
    }
    TEST_CHECK(
        rope_midpoint_elements[1] == 0 && rope_xl_elements[1] == 0,
        "RoPE reserves tensor id 1 but allocates no WPE parameters");
    TEST_CHECK(
        rope_midpoint_total == 536159232ULL,
        "RoPE midpoint parameter count excludes learned WPE");
    TEST_CHECK(
        rope_xl_total == 1556048000ULL,
        "RoPE GPT-2 XL parameter count excludes learned WPE");

    bridged_medium.vocab_size = 50257;
    bridged_medium.padded_vocab_size = 50304;
    size_t bridged_elements[NUM_PARAMETER_TENSORS];
    size_t bridged_sizeof[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(
        bridged_elements, bridged_sizeof, bridged_medium);
    size_t bridged_total = 0;
    for (int tensor = 0; tensor < NUM_PARAMETER_TENSORS; ++tensor) {
        bridged_total += bridged_elements[tensor];
    }
    TEST_CHECK(
        bridged_elements[0] == 206045184ULL &&
            bridged_elements[1] == 0ULL &&
            bridged_elements[16] == 4194304ULL &&
            bridged_elements[17] == 4194304ULL,
        "bridged tied embedding and projection tensor shapes are exact");
    TEST_CHECK(
        bridged_total == 516745216ULL &&
            bridged_total - bridged_elements[0] == 310700032ULL,
        "bridged GPT-2 Medium total and non-embedding counts are exact");

    GPT2 rope_model = {};
    rope_model.config = rope_midpoint;
    std::memcpy(
        rope_model.param_elements,
        rope_midpoint_elements,
        sizeof(rope_midpoint_elements));
    TEST_CHECK(
        gpt2_zero_stage_one_parameter_shapes_compatible(
            &rope_model, 2048, 8),
        "RoPE midpoint tensor shapes permit eight-way ZeRO-1 sharding");
    int incompatible_tensor_id = -1;
    TEST_CHECK(
        !gpt2_zero_stage_one_parameter_shapes_compatible(
            &rope_model, 2048, 7, &incompatible_tensor_id) &&
            incompatible_tensor_id >= 0,
        "ZeRO-1 preflight rejects rank counts that only divide the flat total");
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

static bool floatx_bits_equal(floatX lhs, floatX rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(floatX)) == 0;
}

static void test_bridge_inplace_matmul_backward() {
    constexpr int B = 1;
    constexpr int T = 64;
    constexpr int C = 64;
    constexpr int E = 128;
    constexpr int V = 256;
    const size_t btc = static_cast<size_t>(B) * T * C;
    const size_t bte = static_cast<size_t>(B) * T * E;
    const size_t btv = static_cast<size_t>(B) * T * V;
    const size_t up_elements = static_cast<size_t>(E) * C;
    const size_t wte_elements = static_cast<size_t>(V) * E;

    auto patterned = [](size_t count, float scale, int modulus) {
        std::vector<float> values(count);
        for (size_t index = 0; index < count; ++index) {
            values[index] = scale *
                static_cast<float>(static_cast<int>(index % modulus) -
                                   modulus / 2);
        }
        return quantize_to_floatx(values);
    };
    const std::vector<floatX> lnf_host = patterned(btc, 0.003f, 29);
    const std::vector<floatX> up_host = patterned(up_elements, 0.002f, 31);
    const std::vector<floatX> wte_host = patterned(wte_elements, 0.001f, 37);
    const std::vector<floatX> dlogits_host = patterned(btv, 0.0005f, 41);

    floatX *lnf_standard, *lnf_inplace, *up, *lm_standard, *lm_inplace;
    floatX *wte, *dlogits, *dlexical_standard;
    floatX *dwte_standard, *dwte_inplace, *dup_standard, *dup_inplace;
    cudaCheck(cudaMalloc(&lnf_standard, btc * sizeof(floatX)));
    cudaCheck(cudaMalloc(&lnf_inplace, btc * sizeof(floatX)));
    cudaCheck(cudaMalloc(&up, up_elements * sizeof(floatX)));
    cudaCheck(cudaMalloc(&lm_standard, bte * sizeof(floatX)));
    cudaCheck(cudaMalloc(&lm_inplace, bte * sizeof(floatX)));
    cudaCheck(cudaMalloc(&wte, wte_elements * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dlogits, btv * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dlexical_standard, bte * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dwte_standard, wte_elements * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dwte_inplace, wte_elements * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dup_standard, up_elements * sizeof(floatX)));
    cudaCheck(cudaMalloc(&dup_inplace, up_elements * sizeof(floatX)));
    copy_to_device(lnf_standard, lnf_host);
    copy_to_device(lnf_inplace, lnf_host);
    copy_to_device(up, up_host);
    copy_to_device(wte, wte_host);
    copy_to_device(dlogits, dlogits_host);
    cudaCheck(cudaMemset(dwte_standard, 0, wte_elements * sizeof(floatX)));
    cudaCheck(cudaMemset(dwte_inplace, 0, wte_elements * sizeof(floatX)));
    cudaCheck(cudaMemset(dup_standard, 0, up_elements * sizeof(floatX)));
    cudaCheck(cudaMemset(dup_inplace, 0, up_elements * sizeof(floatX)));

    matmul_forward_cublaslt(
        lm_standard, lnf_standard, up, NULL, B, T, C, E, main_stream);
    cudaCheck(cudaMemcpyAsync(
        lm_inplace,
        lm_standard,
        bte * sizeof(floatX),
        cudaMemcpyDeviceToDevice,
        main_stream));
    matmul_backward(
        dlexical_standard,
        dwte_standard,
        NULL,
        dlogits,
        lm_standard,
        wte,
        NULL,
        B,
        T,
        E,
        V,
        main_stream);
    matmul_backward(
        lnf_standard,
        dup_standard,
        NULL,
        dlexical_standard,
        lnf_inplace,
        up,
        NULL,
        B,
        T,
        C,
        E,
        main_stream);
    matmul_backward_inplace_input(
        lm_inplace,
        dwte_inplace,
        dlogits,
        wte,
        B,
        T,
        E,
        V,
        main_stream);
    matmul_backward_inplace_input(
        lnf_inplace,
        dup_inplace,
        lm_inplace,
        up,
        B,
        T,
        C,
        E,
        main_stream);
    cudaCheck(cudaDeviceSynchronize());

    const std::vector<floatX> dlnf_standard =
        copy_from_device(lnf_standard, btc);
    const std::vector<floatX> dlnf_inplace =
        copy_from_device(lnf_inplace, btc);
    const std::vector<floatX> dwte_a =
        copy_from_device(dwte_standard, wte_elements);
    const std::vector<floatX> dwte_b =
        copy_from_device(dwte_inplace, wte_elements);
    const std::vector<floatX> dup_a =
        copy_from_device(dup_standard, up_elements);
    const std::vector<floatX> dup_b =
        copy_from_device(dup_inplace, up_elements);
    TEST_CHECK(
        std::memcmp(
            dlnf_standard.data(),
            dlnf_inplace.data(),
            btc * sizeof(floatX)) == 0,
        "in-place bridge backward preserves the activation gradient bit-exactly");
    TEST_CHECK(
        std::memcmp(
            dwte_a.data(),
            dwte_b.data(),
            wte_elements * sizeof(floatX)) == 0,
        "in-place tied-head backward preserves the WTE gradient bit-exactly");
    TEST_CHECK(
        std::memcmp(
            dup_a.data(),
            dup_b.data(),
            up_elements * sizeof(floatX)) == 0,
        "in-place bridge backward preserves the projection gradient bit-exactly");

    cudaCheck(cudaFree(lnf_standard));
    cudaCheck(cudaFree(lnf_inplace));
    cudaCheck(cudaFree(up));
    cudaCheck(cudaFree(lm_standard));
    cudaCheck(cudaFree(lm_inplace));
    cudaCheck(cudaFree(wte));
    cudaCheck(cudaFree(dlogits));
    cudaCheck(cudaFree(dlexical_standard));
    cudaCheck(cudaFree(dwte_standard));
    cudaCheck(cudaFree(dwte_inplace));
    cudaCheck(cudaFree(dup_standard));
    cudaCheck(cudaFree(dup_inplace));
}

static void test_bridged_model_forward_backward_and_checkpoint() {
    GPT2 model = {};
    gpt2_init_common(&model);
    model.config.max_seq_len = 32;
    model.config.vocab_size = 50257;
    model.config.padded_vocab_size = 50304;
    model.config.num_layers = 1;
    model.config.num_heads = 6;
    model.config.channels = 384;
    model.config.lexical_channels = 512;
    model.config.position_encoding = LLMC_POSITION_ENCODING_ROPE;
    model.config.rope_rotary_dim = 64;
    model.config.rope_theta = LLMC_ROPE_THETA_DEFAULT;
    model.config.rope_lowest_frequency_plane_is_dc = 0;
    gpt2_set_initializer_defaults(&model.config);
    TEST_CHECK(
        gpt2_validate_position_config(&model.config),
        "tiny bridged RoPE config validates");
    gpt2_allocate_weights(&model);

    std::vector<floatX> parameters(model.num_parameters, (floatX)0.0f);
    size_t tensor_offset = 0;
    for (int tensor_id = 0;
         tensor_id < NUM_PARAMETER_TENSORS;
         ++tensor_id) {
        const size_t elements = model.param_elements[tensor_id];
        if (tensor_id == 2 || tensor_id == 8 || tensor_id == 14) {
            for (size_t index = 0; index < elements; ++index) {
                parameters[tensor_offset + index] = (floatX)1.0f;
            }
        } else if (
            tensor_id == 0 || tensor_id == 4 || tensor_id == 6 ||
            tensor_id == 10 || tensor_id == 12 || tensor_id == 16 ||
            tensor_id == 17) {
            for (size_t index = 0; index < elements; ++index) {
                const float value = 0.002f *
                    static_cast<float>(static_cast<int>(index % 23U) - 11);
                parameters[tensor_offset + index] = (floatX)value;
            }
        }
        tensor_offset += elements;
    }
    copy_to_device((floatX*)model.params_memory, parameters);

    const char* checkpoint_path =
        "build/lexical_bridge_tiny_checkpoint.bin";
    gpt2_write_to_checkpoint(&model, checkpoint_path);
    GPT2 loaded = {};
    gpt2_init_common(&loaded);
    gpt2_build_from_checkpoint(&loaded, checkpoint_path);
    const std::vector<floatX> loaded_parameters = copy_from_device(
        (floatX*)loaded.params_memory, loaded.num_parameters);
    TEST_CHECK(
        loaded.config.lexical_channels == 512 &&
            loaded.config.channels == 384 &&
            loaded.config.position_encoding == LLMC_POSITION_ENCODING_ROPE &&
            loaded.config.rope_lowest_frequency_plane_is_dc == 0 &&
            loaded.num_parameters == model.num_parameters &&
            std::memcmp(
                loaded_parameters.data(),
                parameters.data(),
                parameters.size() * sizeof(floatX)) == 0,
        "bridged model checkpoint round-trips metadata and parameters exactly");
    cudaFreeCheck(&loaded.params_memory);
    remove(checkpoint_path);

    model.config.rope_lowest_frequency_plane_is_dc = 1;
    const char* dc_checkpoint_path =
        "build/lexical_bridge_tiny_rope_dc_checkpoint.bin";
    gpt2_write_to_checkpoint(&model, dc_checkpoint_path);
    GPT2 loaded_dc = {};
    gpt2_init_common(&loaded_dc);
    gpt2_build_from_checkpoint(&loaded_dc, dc_checkpoint_path);
    const std::vector<floatX> loaded_dc_parameters = copy_from_device(
        (floatX*)loaded_dc.params_memory, loaded_dc.num_parameters);
    TEST_CHECK(
        loaded_dc.config.lexical_channels == 512 &&
            loaded_dc.config.channels == 384 &&
            loaded_dc.config.position_encoding == LLMC_POSITION_ENCODING_ROPE &&
            loaded_dc.config.rope_rotary_dim == 64 &&
            loaded_dc.config.rope_lowest_frequency_plane_is_dc == 1 &&
            loaded_dc.num_parameters == model.num_parameters &&
            std::memcmp(
                loaded_dc_parameters.data(),
                parameters.data(),
                parameters.size() * sizeof(floatX)) == 0,
        "lowest-plane-DC bridged checkpoint round-trips its versioned schema and parameters exactly");
    cudaFreeCheck(&loaded_dc.params_memory);
    remove(dc_checkpoint_path);

    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);
    gpt2_allocate_state(&model, 1, 32);
    std::vector<int> inputs(32);
    std::vector<int> targets(32);
    for (int index = 0; index < 32; ++index) {
        inputs[index] = (7 * index + 5) % model.config.vocab_size;
        targets[index] = (7 * (index + 1) + 5) % model.config.vocab_size;
    }
    gpt2_forward(&model, inputs.data(), 1, 32);
    gpt2_backward_and_reduce(
        &model, inputs.data(), targets.data(), 1, 0);
    const std::vector<float> down_gradient = dequantize_floatx(
        copy_from_device(
            model.grads.lexical_downw,
            model.param_elements[16]));
    const std::vector<float> up_gradient = dequantize_floatx(
        copy_from_device(
            model.grads.lexical_upw,
            model.param_elements[17]));
    const std::vector<float> tied_gradient = dequantize_floatx(
        copy_from_device(model.grads.wte, model.param_elements[0]));
    auto max_abs = [](const std::vector<float>& values) {
        float result = 0.0f;
        for (float value : values) {
            result = std::max(result, std::fabs(value));
        }
        return result;
    };
    TEST_CHECK(
        std::isfinite(model.mean_loss) && model.mean_loss > 0.0f,
        "tiny bridged model produces a finite positive training loss");
    TEST_CHECK(
        all_finite(down_gradient) && max_abs(down_gradient) > 0.0f,
        "input bridge receives a finite nonzero gradient");
    TEST_CHECK(
        all_finite(up_gradient) && max_abs(up_gradient) > 0.0f,
        "output bridge receives a finite nonzero gradient");
    TEST_CHECK(
        all_finite(tied_gradient) && max_abs(tied_gradient) > 0.0f,
        "tied lexical table receives finite nonzero lookup/head gradients");
    gpt2_free(&model);
    set_zero_configs(&multi_gpu_config, 0, 1);
}

static void test_encoder_without_wpe() {
    constexpr int batch_size = 1;
    constexpr int sequence_length = 2;
    constexpr int channels = 128;
    constexpr int vocab_size = 4;
    const size_t embedding_elements =
        static_cast<size_t>(vocab_size) * channels;
    const size_t output_elements =
        static_cast<size_t>(batch_size) * sequence_length * channels;
    std::vector<floatX> embeddings(embedding_elements);
    for (size_t index = 0; index < embeddings.size(); ++index) {
        embeddings[index] = static_cast<floatX>(
            0.001f * static_cast<float>(index) - 0.2f);
    }
    const int inputs[] = {1, 3};
    floatX* device_embeddings = nullptr;
    floatX* device_output = nullptr;
    int* device_inputs = nullptr;
    cudaCheck(cudaMalloc(
        reinterpret_cast<void**>(&device_embeddings),
        embedding_elements * sizeof(floatX)));
    cudaCheck(cudaMalloc(
        reinterpret_cast<void**>(&device_output),
        output_elements * sizeof(floatX)));
    cudaCheck(cudaMalloc(
        reinterpret_cast<void**>(&device_inputs), sizeof(inputs)));
    cudaCheck(cudaMemcpy(
        device_embeddings,
        embeddings.data(),
        embedding_elements * sizeof(floatX),
        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(
        device_inputs, inputs, sizeof(inputs), cudaMemcpyHostToDevice));
    encoder_forward(
        device_output,
        device_inputs,
        device_embeddings,
        nullptr,
        batch_size,
        sequence_length,
        channels,
        main_stream);
    cudaCheck(cudaDeviceSynchronize());
    std::vector<floatX> output(output_elements);
    cudaCheck(cudaMemcpy(
        output.data(),
        device_output,
        output_elements * sizeof(floatX),
        cudaMemcpyDeviceToHost));
    bool token_only = true;
    for (int token = 0; token < sequence_length; ++token) {
        for (int channel = 0; channel < channels; ++channel) {
            token_only = token_only && floatx_bits_equal(
                output[static_cast<size_t>(token) * channels + channel],
                embeddings[static_cast<size_t>(inputs[token]) * channels +
                           channel]);
        }
    }
    TEST_CHECK(
        token_only,
        "null WPE makes the encoder output the token embedding exactly");
    cudaCheck(cudaFree(device_embeddings));
    cudaCheck(cudaFree(device_output));
    cudaCheck(cudaFree(device_inputs));
}

static void test_rope_shape(int head_dim) {
    constexpr int batch_size = 1;
    constexpr int sequence_length = 2048;
    constexpr int num_heads = 2;
    constexpr float theta = 10000.0f;
    const int channels = num_heads * head_dim;
    const int rotary_dim = head_dim;
    const int rotary_pairs = rotary_dim / 2;
    const size_t elements =
        static_cast<size_t>(batch_size) * sequence_length * 3 * channels;

    TEST_CHECK(
        llmc_rope_validate_config(
            sequence_length, channels, num_heads, rotary_dim, theta, 0),
        "valid full-head RoPE configuration is accepted");
    TEST_CHECK(
        !llmc_rope_validate_config(
            sequence_length, channels, num_heads, rotary_dim - 1, theta, 0),
        "odd rotary dimension is rejected");
    TEST_CHECK(
        !llmc_rope_validate_config(
            sequence_length, channels, num_heads, rotary_dim + 2, theta, 0),
        "rotary dimension wider than a head is rejected");
    TEST_CHECK(
        !llmc_rope_validate_config(
            sequence_length, channels, num_heads, rotary_dim, theta, 2),
        "unknown RoPE frequency modes are rejected");

    std::vector<floatX> input(elements);
    for (size_t index = 0; index < elements; ++index) {
        const float value =
            0.45f * std::sin(0.013f * static_cast<float>(index % 997U)) +
            0.20f * std::cos(0.007f * static_cast<float>(index % 577U));
        input[index] = static_cast<floatX>(value);
    }

    floatX* device_qkv = nullptr;
    cudaCheck(cudaMalloc(
        reinterpret_cast<void**>(&device_qkv), elements * sizeof(floatX)));
    cudaCheck(cudaMemcpy(
        device_qkv,
        input.data(),
        elements * sizeof(floatX),
        cudaMemcpyHostToDevice));

    LlmcRopeCache cache;
    llmc_rope_cache_reset(&cache);
    TEST_CHECK(
        llmc_rope_cache_allocate(
            &cache, sequence_length, rotary_dim, theta, 0, main_stream),
        "RoPE phase cache allocation succeeds");
    cudaStream_t other_stream;
    cudaCheck(cudaStreamCreate(&other_stream));
    TEST_CHECK(
        !llmc_rope_cache_allocate(
            &cache, sequence_length, rotary_dim, theta, 0, other_stream),
        "RoPE cache rejects implicit cross-stream reuse");
    TEST_CHECK(
        !llmc_rope_cache_allocate(
            &cache, sequence_length, rotary_dim, theta, 1, main_stream),
        "RoPE cache identity includes the lowest-plane-DC mode");
    cudaCheck(cudaStreamDestroy(other_stream));
    TEST_CHECK(
        llmc_rope_cache_bytes(&cache) ==
            static_cast<size_t>(sequence_length) * rotary_pairs *
                sizeof(float2),
        "RoPE phase cache has the expected size");
    TEST_CHECK(
        llmc_rope_apply_qk(
            device_qkv,
            &cache,
            batch_size,
            sequence_length,
            channels,
            num_heads,
            main_stream),
        "RoPE Q/K forward launch succeeds");
    cudaCheck(cudaDeviceSynchronize());

    std::vector<floatX> rotated(elements);
    cudaCheck(cudaMemcpy(
        rotated.data(),
        device_qkv,
        elements * sizeof(floatX),
        cudaMemcpyDeviceToHost));
    std::vector<float2> phases(
        static_cast<size_t>(sequence_length) * rotary_pairs);
    cudaCheck(cudaMemcpy(
        phases.data(),
        cache.cos_sin,
        phases.size() * sizeof(float2),
        cudaMemcpyDeviceToHost));

    LlmcRopeCache dc_cache;
    llmc_rope_cache_reset(&dc_cache);
    TEST_CHECK(
        llmc_rope_cache_allocate(
            &dc_cache, sequence_length, rotary_dim, theta, 1, main_stream),
        "lowest-plane-DC RoPE phase cache allocation succeeds");
    std::vector<float2> dc_phases(phases.size());
    cudaCheck(cudaMemcpy(
        dc_phases.data(),
        dc_cache.cos_sin,
        dc_phases.size() * sizeof(float2),
        cudaMemcpyDeviceToHost));
    bool non_dc_phases_unchanged = true;
    bool lowest_plane_is_exact_dc = true;
    for (int position = 0; position < sequence_length; ++position) {
        for (int pair = 0; pair < rotary_pairs; ++pair) {
            const size_t index =
                static_cast<size_t>(position) * rotary_pairs + pair;
            if (pair == rotary_pairs - 1) {
                lowest_plane_is_exact_dc = lowest_plane_is_exact_dc &&
                    dc_phases[index].x == 1.0f &&
                    dc_phases[index].y == 0.0f;
            } else {
                non_dc_phases_unchanged = non_dc_phases_unchanged &&
                    std::memcmp(
                        &dc_phases[index],
                        &phases[index],
                        sizeof(float2)) == 0;
            }
        }
    }
    TEST_CHECK(
        lowest_plane_is_exact_dc,
        "lowest-plane-DC RoPE cache stores exact identity phases");
    TEST_CHECK(
        non_dc_phases_unchanged,
        "lowest-plane-DC RoPE leaves every other canonical phase bit-identical");

    floatX* device_dc_qkv = nullptr;
    cudaCheck(cudaMalloc(
        reinterpret_cast<void**>(&device_dc_qkv), elements * sizeof(floatX)));
    cudaCheck(cudaMemcpy(
        device_dc_qkv,
        input.data(),
        elements * sizeof(floatX),
        cudaMemcpyHostToDevice));
    TEST_CHECK(
        llmc_rope_apply_qk(
            device_dc_qkv,
            &dc_cache,
            batch_size,
            sequence_length,
            channels,
            num_heads,
            main_stream),
        "lowest-plane-DC RoPE Q/K forward launch succeeds");
    cudaCheck(cudaDeviceSynchronize());
    std::vector<floatX> dc_rotated(elements);
    cudaCheck(cudaMemcpy(
        dc_rotated.data(),
        device_dc_qkv,
        elements * sizeof(floatX),
        cudaMemcpyDeviceToHost));
    bool lowest_qk_plane_is_identity = true;
    for (int position = 0; position < sequence_length; ++position) {
        const size_t token_offset =
            static_cast<size_t>(position) * 3U * channels;
        for (int head = 0; head < num_heads; ++head) {
            const size_t q_index = token_offset +
                static_cast<size_t>(head) * head_dim + rotary_dim - 2U;
            const size_t k_index = q_index + channels;
            const size_t indices[] = {q_index, k_index};
            for (size_t base : indices) {
                lowest_qk_plane_is_identity = lowest_qk_plane_is_identity &&
                    floatx_bits_equal(dc_rotated[base], input[base]) &&
                    floatx_bits_equal(
                        dc_rotated[base + 1U], input[base + 1U]);
            }
        }
    }
    TEST_CHECK(
        lowest_qk_plane_is_identity,
        "lowest-plane-DC RoPE leaves its Q/K channel pair bit-identical");

    bool v_unchanged = true;
    bool position_zero_unchanged = true;
    for (int position = 0; position < sequence_length; ++position) {
        const size_t token_offset =
            static_cast<size_t>(position) * 3U * channels;
        for (int channel = 0; channel < channels; ++channel) {
            v_unchanged = v_unchanged && floatx_bits_equal(
                rotated[token_offset + 2U * channels + channel],
                input[token_offset + 2U * channels + channel]);
            if (position == 0) {
                position_zero_unchanged = position_zero_unchanged &&
                    floatx_bits_equal(
                        rotated[token_offset + channel],
                        input[token_offset + channel]) &&
                    floatx_bits_equal(
                        rotated[token_offset + channels + channel],
                        input[token_offset + channels + channel]);
            }
        }
    }
    TEST_CHECK(v_unchanged, "RoPE leaves V byte-for-byte unchanged");
    TEST_CHECK(
        position_zero_unchanged,
        "RoPE position zero is a bit identity for Q and K");

    const int checked_positions[] = {0, 1, 2, 1023, 2047};
    float maximum_reference_error = 0.0f;
    float maximum_phase_error = 0.0f;
    float maximum_pair_norm_error = 0.0f;
    for (int position : checked_positions) {
        for (int pair = 0; pair < rotary_pairs; ++pair) {
            const double inverse_frequency = std::pow(
                static_cast<double>(theta),
                -2.0 * static_cast<double>(pair) /
                    static_cast<double>(rotary_dim));
            const double angle = static_cast<double>(position) * inverse_frequency;
            const float2 phase =
                phases[static_cast<size_t>(position) * rotary_pairs + pair];
            maximum_phase_error = std::max(
                maximum_phase_error,
                std::max(
                    std::fabs(phase.x - static_cast<float>(std::cos(angle))),
                    std::fabs(phase.y - static_cast<float>(std::sin(angle)))));
            for (int head = 0; head < num_heads; ++head) {
                const size_t q_index =
                    static_cast<size_t>(position) * 3U * channels +
                    static_cast<size_t>(head) * head_dim + 2U * pair;
                const size_t k_index = q_index + channels;
                const size_t indices[] = {q_index, k_index};
                for (size_t base : indices) {
                    const float x0 = static_cast<float>(input[base]);
                    const float x1 = static_cast<float>(input[base + 1U]);
                    const float expected0 = x0 * phase.x - x1 * phase.y;
                    const float expected1 = x0 * phase.y + x1 * phase.x;
                    const float actual0 = static_cast<float>(rotated[base]);
                    const float actual1 = static_cast<float>(rotated[base + 1U]);
                    maximum_reference_error = std::max(
                        maximum_reference_error,
                        std::max(
                            std::fabs(actual0 - expected0),
                            std::fabs(actual1 - expected1)));
                    maximum_pair_norm_error = std::max(
                        maximum_pair_norm_error,
                        std::fabs(
                            (actual0 * actual0 + actual1 * actual1) -
                            (x0 * x0 + x1 * x1)));
                }
            }
        }
    }
    TEST_CHECK(
        maximum_phase_error < 5.0e-4f,
        "RoPE cache matches the canonical theta frequency schedule");
    TEST_CHECK(
        maximum_reference_error < 5.0e-3f,
        "RoPE forward matches an FP32 adjacent-pair reference");
    TEST_CHECK(
        maximum_pair_norm_error < 1.0e-2f,
        "RoPE approximately preserves pair norms after floatX rounding");

    TEST_CHECK(
        llmc_rope_apply_qk_backward(
            device_qkv,
            &cache,
            batch_size,
            sequence_length,
            channels,
            num_heads,
            main_stream),
        "RoPE Q/K transposed backward launch succeeds");
    cudaCheck(cudaDeviceSynchronize());
    std::vector<floatX> roundtrip(elements);
    cudaCheck(cudaMemcpy(
        roundtrip.data(),
        device_qkv,
        elements * sizeof(floatX),
        cudaMemcpyDeviceToHost));
    float maximum_roundtrip_error = 0.0f;
    bool roundtrip_v_unchanged = true;
    for (int position = 0; position < sequence_length; ++position) {
        const size_t token_offset =
            static_cast<size_t>(position) * 3U * channels;
        for (int channel = 0; channel < 2 * channels; ++channel) {
            maximum_roundtrip_error = std::max(
                maximum_roundtrip_error,
                std::fabs(
                    static_cast<float>(roundtrip[token_offset + channel]) -
                    static_cast<float>(input[token_offset + channel])));
        }
        for (int channel = 0; channel < channels; ++channel) {
            roundtrip_v_unchanged = roundtrip_v_unchanged &&
                floatx_bits_equal(
                    roundtrip[token_offset + 2U * channels + channel],
                    input[token_offset + 2U * channels + channel]);
        }
    }
    TEST_CHECK(
        maximum_roundtrip_error < 1.0e-2f,
        "RoPE backward applies the transposed rotation");
    TEST_CHECK(
        roundtrip_v_unchanged,
        "RoPE backward also leaves V byte-for-byte unchanged");

    llmc_rope_cache_free(&dc_cache);
    llmc_rope_cache_free(&cache);
    cudaCheck(cudaFree(device_dc_qkv));
    cudaCheck(cudaFree(device_qkv));
}

static void test_rope_forward_backward() {
    test_rope_shape(64);
    test_rope_shape(96);
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
    GPT2Config bridged_config = {};
    TEST_CHECK(
        gpt2_config_from_descriptor(
            &bridged_config, "gpt2:rope:d24:t2048:e4096"),
        "bridged optimizer-plan descriptor parses");
    bridged_config.vocab_size = 50257;
    bridged_config.padded_vocab_size = 50304;
    fill_in_parameter_sizes(
        parameter_elements, parameter_sizeof, bridged_config);
    TEST_CHECK(
        llmc_build_optimizer_plan(
            &plan,
            &config,
            bridged_config.num_layers,
            bridged_config.channels,
            parameter_elements,
            error,
            sizeof(error)),
        "mixed optimizer plan builds for bridged GPT-2 Medium");
    TEST_CHECK(
        plan.parameter_types[16].tensor_elements == 4194304ULL &&
            plan.parameter_types[17].tensor_elements == 4194304ULL &&
            plan.parameter_types[16].backend_kind ==
                LLMC_OPTIMIZER_BACKEND_ADAMW &&
            plan.parameter_types[17].backend_kind ==
                LLMC_OPTIMIZER_BACKEND_ADAMW &&
            plan.normuon_parameter_type_count == 2,
        "lexical bridges stay on AdamW while only core Wup/Wdown use NorMuon");
    fill_in_parameter_sizes(
        parameter_elements, parameter_sizeof, model_config);
    TEST_CHECK(
        llmc_build_optimizer_plan(
            &plan,
            &config,
            model_config.num_layers,
            model_config.channels,
            parameter_elements,
            error,
            sizeof(error)),
        "GPT-2 small mixed plan rebuilds after bridge routing check");
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

    config.orthogonalization_mode = LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
    config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    TEST_CHECK(
        llmc_build_optimizer_plan(
            &plan,
            &config,
            small_config.num_layers,
            small_config.channels,
            parameter_elements,
            error,
            sizeof(error)),
        "rectangular Muon scratch plan builds");
    TEST_CHECK(
        plan.normuon_view_count == small_config.num_layers * 2,
        "rectangular Muon uses one view per Wup/Wdown layer");
    for (int tensor_id : {10, 12}) {
        const LlmcOptimizerParameterType& parameter_type =
            plan.parameter_types[tensor_id];
        TEST_CHECK(
            parameter_type.views_per_layer == 1,
            "rectangular Muon has one contiguous view per layer");
        LlmcOptimizerMatrixView view;
        TEST_CHECK(
            parameter_type.enumerate_matrix_view(&parameter_type, 0, &view),
            "rectangular view enumerator succeeds");
        TEST_CHECK(
            view.rows * view.columns == 4U * 17U * 17U,
            "rectangular view preserves all 4C^2 parameters");
        TEST_CHECK(
            llmc_optimizer_view_within_bounds(&parameter_type, &view),
            "rectangular view stays in bounds");
        TEST_CHECK(
            (tensor_id == 10 && view.rows == 68U && view.columns == 17U) ||
                (tensor_id == 12 && view.rows == 17U && view.columns == 68U),
            "rectangular Wup/Wdown orientation is correct");
    }
}

static void test_rectangular_scratch_update() {
    constexpr int width = 3;
    constexpr size_t elements = 4U * width * width;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
    config.orthogonalization_mode = LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
    config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    llmc_normuon_resolve_schedules(&config);
    LlmcOptimizerPlan plan = minimal_runtime_plan(width);
    LlmcOptimizerParameterType parameter_type = rectangular_parameter_type(
        width, LLMC_OPTIMIZER_FAMILY_MLP_WUP);
    LlmcOptimizerMatrixView view = rectangular_view(width, true);
    std::vector<float> gradient_host(elements);
    std::vector<float> momentum_host(elements, 0.0f);
    std::vector<float> second_host(elements, 0.01f);
    std::vector<float> master_host(elements);
    for (size_t index = 0; index < elements; ++index) {
        gradient_host[index] = 0.01f * std::sin(static_cast<float>(index + 1));
        master_host[index] = 0.05f * std::cos(static_cast<float>(index + 2));
    }
    const std::vector<floatX> gradient = quantize_to_floatx(gradient_host);
    const std::vector<floatX> parameter = quantize_to_floatx(master_host);
    ViewBuffers buffers(width, elements);
    buffers.load(parameter, gradient, momentum_host, second_host, master_host);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    TEST_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "rectangular scratch runtime allocates");
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
            0.01f,
            1.0f,
            0U,
            0,
            0),
        "rectangular scratch update succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<float> master = copy_from_device(buffers.master, elements);
    TEST_CHECK(all_finite(master), "rectangular scratch output is finite");
    llmc_normuon_runtime_free(&runtime);
}

static void test_rectangular_tracker_smoke() {
    constexpr int width = 3;
    constexpr size_t elements = 4U * width * width;
    for (int family_id : {LLMC_OPTIMIZER_FAMILY_MLP_WUP,
                          LLMC_OPTIMIZER_FAMILY_MLP_WDOWN}) {
        LlmcNormuonConfig config;
        llmc_normuon_config_defaults(&config);
        config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
        config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
        config.orthogonalization_mode =
            LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q;
        config.refresh_interval = 3U;
        config.refresh_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
        config.correction_policy =
            LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
        config.correction_iterations = 2U;
        config.retraction_mode =
            LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
        config.correction_mode =
            LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER;
        llmc_normuon_resolve_schedules(&config);
        char config_error[256] = {};
        TEST_CHECK(
            llmc_normuon_validate_config(
                &config, config_error, sizeof(config_error)),
            "rectangular tracker configuration validates");
        LlmcOptimizerPlan plan = minimal_runtime_plan(width);
        LlmcOptimizerParameterType parameter_type = rectangular_parameter_type(
            width, family_id);
        LlmcOptimizerMatrixView view = rectangular_view(
            width, family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP);
        std::vector<float> gradient_host(elements);
        std::vector<float> momentum_host(elements, 0.0f);
        std::vector<float> second_host(elements, 0.01f);
        std::vector<float> master_host(elements);
        for (size_t index = 0; index < elements; ++index) {
            gradient_host[index] = 0.01f * std::sin(
                static_cast<float>(index + 2 * family_id));
            master_host[index] = 0.05f * std::cos(
                static_cast<float>(index + 3));
        }
        ViewBuffers buffers(width, elements);
        buffers.load(
            quantize_to_floatx(master_host),
            quantize_to_floatx(gradient_host),
            momentum_host,
            second_host,
            master_host);
        LlmcNormuonRuntime runtime;
        llmc_normuon_runtime_reset(&runtime);
        TEST_CHECK(
            llmc_normuon_runtime_allocate(&runtime, &plan, &config),
            "rectangular tracker runtime allocates");
        for (uint64_t step = 0U; step < 2U; ++step) {
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
                    0.01f,
                    1.0f,
                    step,
                    0,
                    0),
                "rectangular tracker update succeeds");
            cudaCheck(cudaStreamSynchronize(main_stream));
        }
        const std::vector<float> master = copy_from_device(
            buffers.master, elements);
        const std::vector<float> tracked_q = copy_from_device(
            runtime.tracked_q, runtime.tracked_q_elements);
        TEST_CHECK(
            all_finite(master) && all_finite(tracked_q),
            "rectangular tracker outputs remain finite");
        TEST_CHECK(
            runtime.q_valid[family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0 : 1] != 0U,
            "rectangular tracker records Q validity");
        llmc_normuon_runtime_free(&runtime);
    }
}

static void test_rectangular_cachemuon_smoke() {
    constexpr int width = 3;
    constexpr size_t elements = 4U * width * width;
    for (int family_id : {LLMC_OPTIMIZER_FAMILY_MLP_WUP,
                          LLMC_OPTIMIZER_FAMILY_MLP_WDOWN}) {
        LlmcNormuonConfig config;
        llmc_normuon_config_defaults(&config);
        config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
        config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
        config.orthogonalization_mode =
            LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON;
        config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
        config.cache_residual_threshold = 1.0e6f;
        llmc_normuon_resolve_schedules(&config);
        char config_error[256] = {};
        TEST_CHECK(
            llmc_normuon_validate_config(
                &config, config_error, sizeof(config_error)),
            "rectangular CacheMuon configuration validates");
        TEST_CHECK(
            config.refresh_policy == LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS,
            "rectangular CacheMuon pins the paper FreshGNS schedule");

        LlmcOptimizerPlan plan = minimal_runtime_plan(width);
        LlmcOptimizerParameterType parameter_type = rectangular_parameter_type(
            width, family_id);
        LlmcOptimizerMatrixView view = rectangular_view(
            width, family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP);
        std::vector<float> gradient_host(elements);
        std::vector<float> momentum_host(elements, 0.0f);
        std::vector<float> second_host(elements, 0.01f);
        std::vector<float> master_host(elements);
        for (size_t index = 0; index < elements; ++index) {
            gradient_host[index] =
                0.01f * std::sin(static_cast<float>(index + family_id + 1));
            master_host[index] =
                0.05f * std::cos(static_cast<float>(index + 4));
        }
        ViewBuffers buffers(width, elements);
        buffers.load(
            quantize_to_floatx(master_host),
            quantize_to_floatx(gradient_host),
            momentum_host,
            second_host,
            master_host);
        LlmcNormuonRuntime runtime;
        llmc_normuon_runtime_reset(&runtime);
        TEST_CHECK(
            llmc_normuon_runtime_allocate(&runtime, &plan, &config),
            "rectangular CacheMuon runtime allocates");
        const size_t q_index = family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 1U;
        for (uint64_t step = 0U; step < 2U; ++step) {
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
                    0.01f,
                    1.0f,
                    step,
                    0,
                    0),
                "rectangular CacheMuon update succeeds");
            cudaCheck(cudaStreamSynchronize(main_stream));
        }
        TEST_CHECK(
            runtime.q_valid[q_index] != 0U && runtime.refresh_count[q_index] == 1U,
            "rectangular CacheMuon refreshes once then accepts the cached transform");
        TEST_CHECK(
            all_finite(copy_from_device(buffers.master, elements)) &&
                all_finite(copy_from_device(
                    runtime.tracked_q, runtime.tracked_q_elements)),
            "rectangular CacheMuon output and cached transform remain finite");
        llmc_normuon_runtime_free(&runtime);
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
    config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
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

static void test_tracker_reference_and_resume(
    LlmcNormuonTrackerCorrectionMode correction_mode) {
    constexpr int width = 5;
    constexpr float learning_rate = 0.01f;
    const size_t elements = static_cast<size_t>(width) * width;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection =
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    config.refresh_interval = 3U;
    config.refresh_policy =
        LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config.correction_policy =
        LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config.correction_iterations = 2U;
    config.correction_gain = 1.0f;
    config.correction_mode = correction_mode;
    config.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ;
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
        correction_mode == LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER
            ? "build/test_normuon_companion_diagonal.bin"
            : "build/test_normuon_companion.bin";
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
    model.optimizer_config.execution_mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
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

    test_gpt2_context_descriptor();
    test_bridge_inplace_matmul_backward();
    test_bridged_model_forward_backward_and_checkpoint();
    test_encoder_without_wpe();
    test_rope_forward_backward();
    test_parameter_plan_and_views();
    test_rectangular_scratch_update();
    test_rectangular_tracker_smoke();
    test_rectangular_cachemuon_smoke();
    test_scratch_against_reference();
    test_tracker_reference_and_resume(
        LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS);
    test_tracker_reference_and_resume(
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER);
    test_init_from_master_only_no_mutation();

    GPT2 unused_model = {};
    common_free(unused_model);
    multi_gpu_config_free(&multi_gpu_config);
    if (test_failures == 0) {
        printf("All focused llm.c RoPE/bridge/NorMuon tests passed.\n");
        return EXIT_SUCCESS;
    }
    fprintf(stderr, "%d focused llm.c RoPE/bridge/NorMuon tests failed.\n", test_failures);
    return EXIT_FAILURE;
}
