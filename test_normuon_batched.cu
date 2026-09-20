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

static void test_lr_dither_schedule() {
    BATCHED_CHECK(
        llmc_gpt2_normuon_global_step(200005) == 200004U,
        "trainer display step and optimizer global step share one boundary conversion");
    LlmcNormuonRuntime runtime = {};
    runtime.lr_dither_enabled = true;
    runtime.lr_dither_amplitude = 0.05f;
    runtime.lr_dither_interval = 12U;
    runtime.lr_dither_wup_scale = 1.0f;
    runtime.lr_dither_wdown_scale = 1.0f;
    const int expected_wup[4] = {1, 1, -1, -1};
    const int expected_wdown[4] = {1, -1, 1, -1};
    for (uint64_t probe = 0U; probe < 4U; ++probe) {
        const uint64_t step = probe * runtime.lr_dither_interval;
        BATCHED_CHECK(
            llmc_normuon_lr_dither_is_pulse_step(&runtime, step),
            "dither pulse occurs on the requested interval");
        BATCHED_CHECK(
            llmc_normuon_lr_dither_is_response_step(&runtime, step + 1U),
            "dither response is sampled at one-step lag");
        BATCHED_CHECK(
            llmc_normuon_lr_dither_sign(
                LLMC_OPTIMIZER_FAMILY_MLP_WUP,
                step,
                runtime.lr_dither_interval) == expected_wup[probe],
            "Wup dither uses the balanced Walsh sign sequence");
        BATCHED_CHECK(
            llmc_normuon_lr_dither_sign(
                LLMC_OPTIMIZER_FAMILY_MLP_WDOWN,
                step,
                runtime.lr_dither_interval) == expected_wdown[probe],
            "Wdown dither uses an orthogonal balanced Walsh sign sequence");
    }
    BATCHED_CHECK(
        fabsf(llmc_normuon_lr_dither_multiplier(
                  &runtime, LLMC_OPTIMIZER_FAMILY_MLP_WUP, 0U) -
              1.05f) < 1.0e-6f,
        "positive dither multiplier is applied only to the update");
    BATCHED_CHECK(
        llmc_normuon_lr_dither_multiplier(
            &runtime, LLMC_OPTIMIZER_FAMILY_MLP_WUP, 1U) == 1.0f,
        "non-pulse steps preserve the base update multiplier");

    LlmcNormuonRuntime sinusoidal = {};
    sinusoidal.lr_dither_enabled = true;
    sinusoidal.lr_dither_mode = LLMC_NORMUON_LR_DITHER_SINUSOIDAL;
    sinusoidal.lr_dither_amplitude = 0.05f;
    sinusoidal.lr_dither_wup_period = 8U;
    sinusoidal.lr_dither_wdown_period = 12U;
    sinusoidal.lr_dither_phase_polarity = -1.0f;
    sinusoidal.lr_dither_wup_scale = 1.0f;
    sinusoidal.lr_dither_wdown_scale = 0.0f;
    BATCHED_CHECK(
        llmc_normuon_lr_dither_is_pulse_step(&sinusoidal, 3U),
        "sinusoidal dither records a source update every step");
    BATCHED_CHECK(
        llmc_normuon_lr_dither_is_response_step(&sinusoidal, 3U),
        "sinusoidal dither records the continuously driven response");
    BATCHED_CHECK(
        !llmc_normuon_lr_dither_is_response_step(&sinusoidal, 0U),
        "sinusoidal dither has no response before a source update exists");
    BATCHED_CHECK(
        fabsf(llmc_normuon_lr_dither_multiplier(
                  &sinusoidal, LLMC_OPTIMIZER_FAMILY_MLP_WUP, 2U) -
              0.95f) < 1.0e-6f,
        "negative-polarity Wup sine reaches its negative peak at one quarter period");
    BATCHED_CHECK(
        llmc_normuon_lr_dither_multiplier(
            &sinusoidal, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, 3U) == 1.0f,
        "a zero family scale leaves Wdown unexcited");
    BATCHED_CHECK(
        fabsf(llmc_normuon_lr_dither_signal(
                  &sinusoidal, LLMC_OPTIMIZER_FAMILY_MLP_WUP, 6U) -
              1.0f) < 1.0e-6f,
        "phase polarity reverses the negative half-cycle");
    sinusoidal.lr_dither_wdown_scale = 1.0f;
    BATCHED_CHECK(
        fabsf(llmc_normuon_lr_dither_multiplier(
                  &sinusoidal, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, 3U) -
              0.95f) < 1.0e-6f,
        "enabled Wdown uses its independent sinusoidal period");

    LlmcNormuonRuntime heterodyne = {};
    heterodyne.lr_dither_enabled = true;
    heterodyne.lr_dither_mode = LLMC_NORMUON_LR_DITHER_HETERODYNE_CHOPPER;
    heterodyne.lr_dither_amplitude = 0.05f;
    heterodyne.lr_dither_envelope_blocks = 8U;
    heterodyne.lr_dither_phase_polarity = 1.0f;
    heterodyne.lr_dither_wup_scale = 0.0f;
    heterodyne.lr_dither_wdown_scale = 1.0f;
    BATCHED_CHECK(
        llmc_normuon_lr_dither_is_pulse_step(&heterodyne, 5U) &&
        llmc_normuon_lr_dither_is_response_step(&heterodyne, 5U),
        "heterodyne chopper retains continuous one-step response telemetry");
    BATCHED_CHECK(
        llmc_normuon_lr_dither_signal(
            &heterodyne, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, 4U) > 0.70f &&
        llmc_normuon_lr_dither_signal(
            &heterodyne, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, 6U) < -0.70f,
        "heterodyne chopper uses the +1,0,-1,0 carrier under one slow envelope");
    BATCHED_CHECK(
        llmc_normuon_lr_dither_multiplier(
            &heterodyne, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, 5U) == 1.0f &&
        llmc_normuon_lr_dither_multiplier(
            &heterodyne, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, 7U) == 1.0f,
        "heterodyne chopper zero carrier slots preserve the base learning rate");
    heterodyne.lr_dither_phase_polarity = -1.0f;
    BATCHED_CHECK(
        llmc_normuon_lr_dither_signal(
            &heterodyne, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, 4U) < -0.70f,
        "heterodyne envelope polarity is exactly reversible");
    BATCHED_CHECK(
        llmc_normuon_lr_dither_multiplier(
            &heterodyne, LLMC_OPTIMIZER_FAMILY_MLP_WUP, 4U) == 1.0f,
        "heterodyne family isolation leaves Wup at its base learning rate");
    BATCHED_CHECK(
        llmc_parse_normuon_lr_dither_mode(
            "heterodyne_chopper", &heterodyne.lr_dither_mode) &&
        strcmp(llmc_normuon_lr_dither_mode_name(heterodyne.lr_dither_mode),
               "heterodyne_chopper") == 0,
        "heterodyne chopper round-trips through the native mode surface");
}

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

static std::vector<float> bf16_rectangular_matmul_reference(
    const std::vector<float>& lhs,
    int lhs_rows,
    int inner,
    const std::vector<float>& rhs,
    int rhs_columns) {
    BATCHED_CHECK(
        lhs.size() == static_cast<size_t>(lhs_rows) * inner &&
            rhs.size() == static_cast<size_t>(inner) * rhs_columns,
        "rectangular reference GEMM dimensions match");
    std::vector<float> output(
        static_cast<size_t>(lhs_rows) * rhs_columns, 0.0f);
    for (int row = 0; row < lhs_rows; ++row) {
        for (int column = 0; column < rhs_columns; ++column) {
            float sum = 0.0f;
            for (int reduction = 0; reduction < inner; ++reduction) {
                sum += round_to_floatx(
                           lhs[static_cast<size_t>(row) * inner + reduction]) *
                    round_to_floatx(
                        rhs[static_cast<size_t>(reduction) * rhs_columns +
                            column]);
            }
            output[static_cast<size_t>(row) * rhs_columns + column] = sum;
        }
    }
    return output;
}

static std::vector<float> transpose_reference(
    const std::vector<float>& matrix,
    int rows,
    int columns) {
    BATCHED_CHECK(
        matrix.size() == static_cast<size_t>(rows) * columns,
        "reference transpose dimensions match");
    std::vector<float> transpose(matrix.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            transpose[static_cast<size_t>(column) * rows + row] =
                matrix[static_cast<size_t>(row) * columns + column];
        }
    }
    return transpose;
}

static std::vector<float> symmetric_sum_reference(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    int width,
    float rhs_scale = 1.0f) {
    const size_t elements = static_cast<size_t>(width) * width;
    BATCHED_CHECK(
        lhs.size() == elements && rhs.size() == elements,
        "symmetric reference operands match");
    std::vector<float> output(elements, 0.0f);
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index = static_cast<size_t>(row) * width + column;
            const size_t transpose =
                static_cast<size_t>(column) * width + row;
            output[index] = 0.5f * (
                lhs[index] + rhs_scale * rhs[index] + lhs[transpose] +
                rhs_scale * rhs[transpose]);
        }
    }
    return output;
}

static float frobenius_norm_reference(const std::vector<float>& matrix) {
    double sum = 0.0;
    for (float value : matrix) {
        sum += static_cast<double>(value) * value;
    }
    return static_cast<float>(std::sqrt(sum));
}

struct CacheMuonInverseRootCpuReference {
    std::vector<float> lagged_direction;
    std::vector<float> k;
    std::vector<float> delta;
    std::vector<float> next_transform;
    std::vector<float> next_direction;
    std::vector<float> wrong_unsandwiched_delta;
    std::vector<float> wrong_unsandwiched_next;
};

static CacheMuonInverseRootCpuReference
cachemuon_inverse_root_tracker_cpu_reference(
    const std::vector<float>& normalized,
    int rows,
    int columns,
    const std::vector<float>& transform,
    uint32_t cg_iterations,
    float epsilon) {
    const int side = std::min(rows, columns);
    const size_t small_elements = static_cast<size_t>(side) * side;
    BATCHED_CHECK(
        normalized.size() == static_cast<size_t>(rows) * columns &&
            transform.size() == small_elements,
        "CacheMuon CPU reference dimensions match");
    CacheMuonInverseRootCpuReference reference;
    if (rows >= columns) {
        reference.lagged_direction = bf16_rectangular_matmul_reference(
            normalized, rows, side, transform, side);
    } else {
        reference.lagged_direction = bf16_rectangular_matmul_reference(
            transform, side, side, normalized, columns);
    }
    const std::vector<float> direction_transpose = transpose_reference(
        reference.lagged_direction, rows, columns);
    std::vector<float> gram = rows >= columns
        ? bf16_rectangular_matmul_reference(
              direction_transpose,
              side,
              rows,
              reference.lagged_direction,
              side)
        : bf16_rectangular_matmul_reference(
              reference.lagged_direction,
              side,
              columns,
              direction_transpose,
              side);

    std::vector<float> residual(small_elements, 0.0f);
    std::vector<float> search(small_elements, 0.0f);
    reference.k.assign(small_elements, 0.0f);
    float residual_norm_squared = 0.0f;
    for (int row = 0; row < side; ++row) {
        for (int column = 0; column < side; ++column) {
            const size_t index = static_cast<size_t>(row) * side + column;
            const float e = gram[index] - (row == column ? 1.0f : 0.0f);
            residual[index] = -e;
            search[index] = -e;
            residual_norm_squared += e * e;
        }
    }
    for (uint32_t iteration = 0U; iteration < cg_iterations; ++iteration) {
        const std::vector<float> search_times_p =
            bf16_rectangular_matmul_reference(
                search, side, side, transform, side);
        std::vector<float> applied(small_elements, 0.0f);
        float denominator = 0.0f;
        for (int row = 0; row < side; ++row) {
            for (int column = 0; column < side; ++column) {
                const size_t index =
                    static_cast<size_t>(row) * side + column;
                const size_t transpose =
                    static_cast<size_t>(column) * side + row;
                applied[index] =
                    search_times_p[index] + search_times_p[transpose];
                denominator += search[index] * applied[index];
            }
        }
        const bool converged =
            std::isfinite(residual_norm_squared) &&
            residual_norm_squared <= epsilon;
        const float alpha =
            !converged && std::isfinite(denominator) &&
                    denominator > epsilon
                ? residual_norm_squared / denominator
                : 0.0f;
        float next_residual_norm_squared = 0.0f;
        for (size_t index = 0; index < small_elements; ++index) {
            reference.k[index] += alpha * search[index];
            residual[index] -= alpha * applied[index];
            next_residual_norm_squared += residual[index] * residual[index];
        }
        const float beta =
            std::isfinite(residual_norm_squared) &&
                    residual_norm_squared > epsilon &&
                    std::isfinite(next_residual_norm_squared)
                ? next_residual_norm_squared / residual_norm_squared
                : 0.0f;
        for (size_t index = 0; index < small_elements; ++index) {
            search[index] = residual[index] + beta * search[index];
        }
        residual_norm_squared = next_residual_norm_squared;
    }

    const std::vector<float> p_times_k =
        bf16_rectangular_matmul_reference(
            transform, side, side, reference.k, side);
    reference.delta = bf16_rectangular_matmul_reference(
        p_times_k, side, side, transform, side);
    reference.next_transform = symmetric_sum_reference(
        transform, reference.delta, side);
    if (rows >= columns) {
        reference.next_direction = bf16_rectangular_matmul_reference(
            normalized, rows, side, reference.next_transform, side);
    } else {
        reference.next_direction = bf16_rectangular_matmul_reference(
            reference.next_transform, side, side, normalized, columns);
    }

    const std::vector<float> p_squared =
        bf16_rectangular_matmul_reference(
            transform, side, side, transform, side);
    const std::vector<float> p_squared_times_k =
        bf16_rectangular_matmul_reference(
            p_squared, side, side, reference.k, side);
    const std::vector<float> k_times_p_squared =
        bf16_rectangular_matmul_reference(
            reference.k, side, side, p_squared, side);
    reference.wrong_unsandwiched_delta.resize(small_elements);
    for (size_t index = 0; index < small_elements; ++index) {
        reference.wrong_unsandwiched_delta[index] =
            0.5f * (p_squared_times_k[index] + k_times_p_squared[index]);
    }
    reference.wrong_unsandwiched_next = symmetric_sum_reference(
        transform, reference.wrong_unsandwiched_delta, side);
    return reference;
}

static float batched_spectral_norm_power_reference(
    const std::vector<float>& matrix,
    int width,
    size_t matrix_index = 0U) {
    if (width <= 1) {
        return 0.0f;
    }
    std::vector<float> x(static_cast<size_t>(width), 0.0f);
    std::vector<float> y(static_cast<size_t>(width), 0.0f);
    const float initial_scale =
        1.0f / std::sqrt(static_cast<float>(width));
    for (int index = 0; index < width; ++index) {
        const uint32_t hash = static_cast<uint32_t>(
            matrix_index * 0x9e3779b9U + static_cast<size_t>(index)) +
            0x7f4a7c15U;
        x[index] = (hash & 1U) != 0U ? initial_scale : -initial_scale;
    }
    float estimate = 0.0f;
    for (uint32_t iteration = 0U;
         iteration < LLMC_NORMUON_TRACKER_SPECTRAL_POWER_ITERATIONS;
         ++iteration) {
        float y_norm_squared = 0.0f;
        for (int row = 0; row < width; ++row) {
            float value = 0.0f;
            for (int column = 0; column < width; ++column) {
                value += matrix[static_cast<size_t>(row) * width + column] *
                    x[column];
            }
            y[row] = value;
            y_norm_squared += value * value;
        }
        estimate = std::sqrt(std::max(y_norm_squared, 0.0f));

        float x_norm_squared = 0.0f;
        for (int column = 0; column < width; ++column) {
            float value = 0.0f;
            for (int row = 0; row < width; ++row) {
                value += matrix[static_cast<size_t>(row) * width + column] *
                    y[row];
            }
            x[column] = value;
            x_norm_squared += value * value;
        }
        const float denominator =
            std::sqrt(std::max(x_norm_squared, 1.0e-20f));
        for (float& value : x) {
            value /= denominator;
        }
    }
    return estimate;
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
    float direction_norm_squared = 0.0f;
    for (int row = 0; row < width; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < width; ++column) {
            const float value =
                direction[static_cast<size_t>(row) * width + column];
            sum += value * value;
            direction_norm_squared += value * value;
        }
        const float mean = sum / static_cast<float>(width);
        const float next =
            config.beta2 * (*second_moment)[row] +
            (1.0f - config.beta2) * mean;
        (*second_moment)[row] = next;
        normalized_norm_squared += sum / std::max(next, config.epsilon);
    }
    const float global_scale =
        std::sqrt(std::max(direction_norm_squared, config.epsilon)) /
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
    float phase_trace = 0.0f;
    std::vector<float> correction(phase.size(), 0.0f);
    for (int row = 0; row < width; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t index = static_cast<size_t>(row) * width + column;
            const float transpose =
                phase[static_cast<size_t>(column) * width + row];
            const float symmetric = 0.5f * (phase[index] + transpose);
            symmetric_norm_squared += symmetric * symmetric;
            if (row == column) {
                phase_trace += phase[index];
            }
        }
    }
    const float denominator =
        std::sqrt(std::max(symmetric_norm_squared, 0.0f)) + config.epsilon;
    float diagonal_scale = 1.0f;
    float basis_free_scale = 1.0f;
    if (config.correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER) {
        const std::vector<float> phase_squared =
            bf16_operand_matmul(phase, false, phase, false, width);
        const float symmetric_rms = std::sqrt(
            std::max(symmetric_norm_squared, 0.0f) /
            static_cast<float>(width));
        const float alpha = std::max(
            std::max(
                phase_trace / static_cast<float>(width),
                LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_rms),
            config.epsilon);
        const float inverse_alpha = 1.0f / alpha;
        const float inverse_two_alpha_squared =
            0.5f * inverse_alpha * inverse_alpha;
        for (int row = 0; row < width; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t index =
                    static_cast<size_t>(row) * width + column;
                const size_t transpose =
                    static_cast<size_t>(column) * width + row;
                const float skew =
                    0.5f * (phase[index] - phase[transpose]);
                const float skew_squared =
                    0.5f * (phase_squared[index] - phase_squared[transpose]);
                correction[index] = config.correction_gain * (
                    2.0f * inverse_alpha * skew -
                    inverse_two_alpha_squared * skew_squared);
            }
        }
        const float spectral_norm = batched_spectral_norm_power_reference(
            correction, width, 0U);
        const float spectral_limit = std::sqrt(std::max(
            config.tracker_spectral_pmax *
                    config.tracker_spectral_pmax -
                1.0f,
            0.0f));
        basis_free_scale = std::min(
            1.0f,
            spectral_limit /
                (std::max(spectral_norm, 0.0f) + config.epsilon));
    } else if (config.correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
        const float symmetric_scale =
            std::sqrt(std::max(symmetric_norm_squared, 0.0f)) /
            std::sqrt(static_cast<float>(width));
        const float diagonal_floor = std::max(
            LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale,
            config.epsilon);
        double correction_norm_squared = 0.0;
        for (int row = 0; row < width; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t index = static_cast<size_t>(row) * width + column;
                const float transpose =
                    phase[static_cast<size_t>(column) * width + row];
                const float skew = 0.5f * (phase[index] - transpose);
                const float row_stiffness = std::max(
                    phase[static_cast<size_t>(row) * width + row],
                    diagonal_floor);
                const float column_stiffness = std::max(
                    phase[static_cast<size_t>(column) * width + column],
                    diagonal_floor);
                const float omega = config.correction_gain * 2.0f * skew /
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
            const size_t index = static_cast<size_t>(row) * width + column;
            const float transpose =
                phase[static_cast<size_t>(column) * width + row];
            const float skew = 0.5f * (phase[index] - transpose);
            float correction_denominator = denominator;
            float correction_numerator = skew;
            if (config.correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER) {
                correction[index] =
                    (row == column ? 1.0f : 0.0f) +
                    basis_free_scale * correction[index];
                continue;
            }
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

static void test_tracker_h_stability_kernel() {
    constexpr size_t width = 2U;
    constexpr size_t matrix_elements = width * width;
    float* phase = nullptr;
    float* previous_h = nullptr;
    float* stats = nullptr;
    int* nonfinite = nullptr;
    cudaCheck(cudaMalloc(&phase, matrix_elements * sizeof(float)));
    cudaCheck(cudaMalloc(&previous_h, matrix_elements * sizeof(float)));
    cudaCheck(cudaMalloc(&stats, 4U * sizeof(float)));
    cudaCheck(cudaMalloc(&nonfinite, sizeof(int)));
    cudaCheck(cudaMemset(previous_h, 0, matrix_elements * sizeof(float)));
    cudaCheck(cudaMemset(nonfinite, 0, sizeof(int)));

    batched_copy_to_device(
        phase,
        std::vector<float>{2.0f, 1.0f, 3.0f, 4.0f});
    llmc_normuon_batch_h_stability_kernel<<<
        1U, LLMC_NORMUON_BLOCK_SIZE, 0, main_stream>>>(
        phase,
        previous_h,
        stats,
        nonfinite,
        1U,
        matrix_elements,
        matrix_elements,
        width,
        false);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<float> first_history =
        batched_copy_from_device(previous_h, matrix_elements);
    BATCHED_CHECK(
        batched_max_abs_difference(
            first_history,
            std::vector<float>{2.0f, 2.0f, 2.0f, 4.0f}) == 0.0f,
        "H-stability snapshot stores the symmetric normalized phase");

    batched_copy_to_device(
        phase,
        std::vector<float>{3.0f, 0.0f, 4.0f, 5.0f});
    llmc_normuon_batch_h_stability_kernel<<<
        1U, LLMC_NORMUON_BLOCK_SIZE, 0, main_stream>>>(
        phase,
        previous_h,
        stats,
        nonfinite,
        1U,
        matrix_elements,
        matrix_elements,
        width,
        true);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(main_stream));
    const std::vector<float> measured = batched_copy_from_device(stats, 4U);
    BATCHED_CHECK(
        fabsf(measured[0] - 42.0f) < 1.0e-6f &&
            fabsf(measured[1] - 28.0f) < 1.0e-6f &&
            fabsf(measured[2] - 2.0f) < 1.0e-6f &&
            fabsf(measured[3] - 34.0f) < 1.0e-6f,
        "H-stability kernel reports current, previous, delta, and dot energies");
    const std::vector<float> second_history =
        batched_copy_from_device(previous_h, matrix_elements);
    BATCHED_CHECK(
        batched_max_abs_difference(
            second_history,
            std::vector<float>{3.0f, 2.0f, 2.0f, 5.0f}) == 0.0f,
        "H-stability snapshot advances only after measuring the prior state");

    cudaCheck(cudaFree(phase));
    cudaCheck(cudaFree(previous_h));
    cudaCheck(cudaFree(stats));
    cudaCheck(cudaFree(nonfinite));
}

static void test_square_wdown_batch_replay_restore() {
    constexpr int width = 2;
    constexpr int layers = 1;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode = LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    config.update_scale = 1.0f;
    config.wdown_learning_rate_multiplier = 1.0f;

    LlmcOptimizerPlan plan = batched_plan(width, layers);
    LlmcOptimizerParameterType parameter_type = batched_parameter_type(
        LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, width, layers);
    const size_t elements = parameter_type.tensor_elements;
    const size_t matrix_count = static_cast<size_t>(
        parameter_type.layer_multiplicity * parameter_type.views_per_layer);

    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "square tracker runtime allocates for batch replay");
    BATCHED_CHECK(
        runtime.tracked_q != nullptr && runtime.axis_stats != nullptr,
        "square tracker replay owns Q and row-scale state");

    std::vector<float> master(elements);
    for (size_t index = 0; index < elements; ++index) {
        master[index] = 0.12345f + 0.001f * static_cast<float>(index);
    }
    std::vector<float> zeros(elements, 0.0f);
    BatchedBuffers buffers(elements);
    buffers.load(master, zeros, zeros, zeros);
    std::vector<float> tracked_q(runtime.tracked_q_bytes / sizeof(float), 1.0f);
    std::vector<float> row_scales(matrix_count * width, 2.0f);
    batched_copy_to_device(runtime.tracked_q, tracked_q);
    batched_copy_to_device(runtime.axis_stats, row_scales);

    float* snapshot = nullptr;
    cudaCheck(cudaMalloc(&snapshot, elements * sizeof(float)));
    cudaCheck(cudaMemcpy(
        snapshot,
        buffers.master,
        elements * sizeof(float),
        cudaMemcpyDeviceToDevice));
    BATCHED_CHECK(
        llmc_normuon_batch_replay_set_square_wdown(
            &runtime,
            main_stream,
            buffers.parameter,
            buffers.master,
            snapshot,
            &parameter_type,
            &config,
            0.1f,
            0.0f,
            17U),
        "baseline stochastic rounding for replay test succeeds");
    const std::vector<floatX> parameter_before =
        batched_copy_from_device(buffers.parameter, elements);
    const uint64_t replay_rounding_step = 17U ^ 0xd1b54a32d192ed03ULL;
    BATCHED_CHECK(
        llmc_normuon_batch_replay_set_square_wdown(
            &runtime,
            main_stream,
            buffers.parameter,
            buffers.master,
            snapshot,
            &parameter_type,
            &config,
            0.1f,
            0.0f,
            replay_rounding_step),
        "common-random replay center rounding succeeds");
    const std::vector<floatX> replay_center =
        batched_copy_from_device(buffers.parameter, elements);
    BATCHED_CHECK(
        llmc_normuon_batch_replay_set_square_wdown(
            &runtime,
            main_stream,
            buffers.parameter,
            buffers.master,
            snapshot,
            &parameter_type,
            &config,
            0.1f,
            0.5f,
            replay_rounding_step),
        "positive square Wdown replay perturbation succeeds");
    const std::vector<float> advanced =
        batched_copy_from_device(buffers.master, elements);
    std::vector<float> expected_advanced = master;
    for (float& value : expected_advanced) {
        value -= 0.1f;
    }
    BATCHED_CHECK(
        batched_max_abs_difference(
            advanced, expected_advanced) < 1.0e-6f,
        "replay applies the requested extra LR multiple to Q times row scale");
    const std::vector<floatX> replay_advanced =
        batched_copy_from_device(buffers.parameter, elements);
    BATCHED_CHECK(
        std::memcmp(
            replay_center.data(),
            replay_advanced.data(),
            elements * sizeof(floatX)) != 0,
        "common-random replay exposes a BF16-resolved perturbation");

    BATCHED_CHECK(
        llmc_normuon_batch_replay_set_square_wdown(
            &runtime,
            main_stream,
            buffers.parameter,
            buffers.master,
            snapshot,
            &parameter_type,
            &config,
            0.1f,
            0.0f,
            17U),
        "zero-offset replay restore succeeds");
    const std::vector<float> restored =
        batched_copy_from_device(buffers.master, elements);
    BATCHED_CHECK(
        batched_max_abs_difference(restored, master) == 0.0f,
        "batch replay restores committed FP32 Wdown masters exactly");
    const std::vector<floatX> parameter_restored =
        batched_copy_from_device(buffers.parameter, elements);
    BATCHED_CHECK(
        std::memcmp(
            parameter_before.data(),
            parameter_restored.data(),
            elements * sizeof(floatX)) == 0,
        "batch replay restores committed BF16 Wdown parameters bit-exactly");

    cudaCheck(cudaFree(snapshot));
    llmc_normuon_runtime_free(&runtime);
}

static void run_rectangular_batched_update(int family_id, bool fresh_gns) {
    constexpr int width = 3;
    constexpr int layers = 2;
    constexpr size_t matrix_elements = 4U * width * width;
    const size_t tensor_elements = static_cast<size_t>(layers) * matrix_elements;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode = LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
    config.refresh_policy = fresh_gns
        ? LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS
        : LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    llmc_normuon_resolve_schedules(&config);
    LlmcOptimizerPlan plan = batched_plan(width, layers);
    plan.normuon_view_count = layers * LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER;
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 10 : 12;
    parameter_type.name =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? "fcw_rectangular_batched_test"
            : "fcprojw_rectangular_batched_test";
    parameter_type.family_id = static_cast<LlmcOptimizerFamilyId>(family_id);
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group = LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = layers;
    parameter_type.layer_elements = matrix_elements;
    parameter_type.tensor_elements = tensor_elements;
    parameter_type.matrix_width = width;
    parameter_type.views_per_layer = 1;
    parameter_type.enumerate_matrix_view =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? llmc_enumerate_mlp_wup_rectangular_view
            : llmc_enumerate_mlp_wdown_rectangular_view;
    std::vector<float> master(tensor_elements);
    std::vector<float> gradient(tensor_elements);
    std::vector<float> momentum(tensor_elements, 0.0f);
    std::vector<float> second(tensor_elements, 0.01f);
    for (size_t index = 0; index < tensor_elements; ++index) {
        master[index] = 0.02f * std::cos(static_cast<float>(index + 1));
        gradient[index] = 0.01f * std::sin(static_cast<float>(index + 2));
    }
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "rectangular batched runtime allocates");
    BATCHED_CHECK(
        (fresh_gns && runtime.cache_small[0] != nullptr &&
         runtime.cache_small[LLMC_CACHEMUON_SMALL_PANEL_COUNT - 1U] != nullptr &&
         runtime.cache_residuals == nullptr && runtime.tracked_q == nullptr) ||
            (!fresh_gns && runtime.cache_small[0] == nullptr),
        "rectangular scratch FreshGNS owns compact solver panels but no cache state");
    BatchedBuffers buffers(tensor_elements);
    buffers.load(master, gradient, momentum, second);
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
            0.01f,
            1.0f,
            0U),
        "rectangular batched update succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    BATCHED_CHECK(
        batched_all_finite(batched_copy_from_device(buffers.master, tensor_elements)),
        "rectangular batched output is finite");
    llmc_normuon_runtime_free(&runtime);
}

static void test_rectangular_batched_update() {
    run_rectangular_batched_update(LLMC_OPTIMIZER_FAMILY_MLP_WUP, false);
    run_rectangular_batched_update(LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, false);
    run_rectangular_batched_update(LLMC_OPTIMIZER_FAMILY_MLP_WUP, true);
    run_rectangular_batched_update(LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, true);
}

static void run_rectangular_batched_tracker(int family_id) {
    constexpr int width = 3;
    constexpr int layers = 2;
    constexpr size_t matrix_elements = 4U * width * width;
    const size_t tensor_elements = static_cast<size_t>(layers) * matrix_elements;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
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
    LlmcOptimizerPlan plan = batched_plan(width, layers);
    plan.normuon_view_count = layers * LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER;
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id = family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 10 : 12;
    parameter_type.name = "rectangular_tracker_batched_test";
    parameter_type.family_id = static_cast<LlmcOptimizerFamilyId>(family_id);
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group = LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = layers;
    parameter_type.layer_elements = matrix_elements;
    parameter_type.tensor_elements = tensor_elements;
    parameter_type.matrix_width = width;
    parameter_type.views_per_layer = 1;
    parameter_type.enumerate_matrix_view =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? llmc_enumerate_mlp_wup_rectangular_view
            : llmc_enumerate_mlp_wdown_rectangular_view;
    std::vector<float> master(tensor_elements);
    std::vector<float> gradient(tensor_elements);
    std::vector<float> momentum(tensor_elements, 0.0f);
    std::vector<float> second(tensor_elements, 0.01f);
    for (size_t index = 0; index < tensor_elements; ++index) {
        master[index] = 0.02f * std::cos(static_cast<float>(index + 1));
        gradient[index] = 0.01f * std::sin(static_cast<float>(index + 2));
    }
    BatchedBuffers buffers(tensor_elements);
    buffers.load(master, gradient, momentum, second);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "rectangular tracker batched runtime allocates");
    for (uint64_t step = 0U; step < 2U; ++step) {
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
                0.01f,
                1.0f,
                step),
            "rectangular tracker batched update succeeds");
        cudaCheck(cudaStreamSynchronize(main_stream));
    }
    BATCHED_CHECK(
        batched_all_finite(batched_copy_from_device(buffers.master, tensor_elements)) &&
            batched_all_finite(batched_copy_from_device(
                runtime.tracked_q, runtime.tracked_q_elements)),
        "rectangular tracker batched outputs remain finite");
    BATCHED_CHECK(
        runtime.q_valid[family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0 : 1] != 0U,
        "rectangular tracker batched records Q validity");
    llmc_normuon_runtime_free(&runtime);
}

static void test_rectangular_batched_tracker() {
    run_rectangular_batched_tracker(LLMC_OPTIMIZER_FAMILY_MLP_WUP);
    run_rectangular_batched_tracker(LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
}

static void run_rectangular_batched_cachemuon(int family_id) {
    constexpr int width = 3;
    constexpr int layers = 2;
    constexpr size_t matrix_elements = 4U * width * width;
    const size_t tensor_elements = static_cast<size_t>(layers) * matrix_elements;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON;
    config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    config.cache_residual_threshold = 1.0e6f;
    llmc_normuon_resolve_schedules(&config);
    LlmcOptimizerPlan plan = batched_plan(width, layers);
    plan.normuon_view_count = layers * LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER;
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 10 : 12;
    parameter_type.name = "rectangular_cachemuon_batched_test";
    parameter_type.family_id = static_cast<LlmcOptimizerFamilyId>(family_id);
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group = LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = layers;
    parameter_type.layer_elements = matrix_elements;
    parameter_type.tensor_elements = tensor_elements;
    parameter_type.matrix_width = width;
    parameter_type.views_per_layer = 1;
    parameter_type.enumerate_matrix_view =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? llmc_enumerate_mlp_wup_rectangular_view
            : llmc_enumerate_mlp_wdown_rectangular_view;

    std::vector<float> master(tensor_elements);
    std::vector<float> gradient(tensor_elements);
    std::vector<float> momentum(tensor_elements, 0.0f);
    std::vector<float> second(tensor_elements, 0.01f);
    for (size_t index = 0; index < tensor_elements; ++index) {
        master[index] = 0.02f * std::cos(static_cast<float>(index + 1));
        gradient[index] = 0.01f * std::sin(static_cast<float>(index + 2));
    }
    BatchedBuffers buffers(tensor_elements);
    buffers.load(master, gradient, momentum, second);
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "rectangular CacheMuon batched runtime allocates");
    for (uint64_t step = 0U; step < 2U; ++step) {
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
                0.01f,
                1.0f,
                step),
            "rectangular CacheMuon batched update succeeds");
        cudaCheck(cudaStreamSynchronize(main_stream));
    }
    bool exactly_one_refresh = true;
    for (int layer = 0; layer < layers; ++layer) {
        const size_t q_index = static_cast<size_t>(layer) *
                                   LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER +
                               (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 1U);
        exactly_one_refresh = exactly_one_refresh && runtime.q_valid[q_index] != 0U &&
                              runtime.refresh_count[q_index] == 1U;
    }
    BATCHED_CHECK(
        exactly_one_refresh,
        "rectangular CacheMuon compacts initial misses then accepts cached transforms");
    BATCHED_CHECK(
        batched_all_finite(batched_copy_from_device(buffers.master, tensor_elements)) &&
            batched_all_finite(batched_copy_from_device(
                runtime.tracked_q, runtime.tracked_q_elements)),
        "rectangular CacheMuon batched outputs remain finite");
    llmc_normuon_runtime_free(&runtime);
}

static void test_rectangular_batched_cachemuon() {
    run_rectangular_batched_cachemuon(LLMC_OPTIMIZER_FAMILY_MLP_WUP);
    run_rectangular_batched_cachemuon(LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
}

static void run_rectangular_batched_cachemuon_inverse_root_tracker(
    int family_id,
    bool force_tracker_rejection) {
    constexpr int width = 3;
    constexpr int layers = 2;
    constexpr size_t matrix_elements = 4U * width * width;
    constexpr size_t small_elements = width * width;
    const size_t tensor_elements = static_cast<size_t>(layers) * matrix_elements;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER;
    config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    llmc_normuon_resolve_schedules(&config);

    LlmcOptimizerPlan plan = batched_plan(width, layers);
    plan.normuon_view_count =
        layers * LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER;
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 10 : 12;
    parameter_type.name = "rectangular_cachemuon_inverse_root_tracker_test";
    parameter_type.family_id = static_cast<LlmcOptimizerFamilyId>(family_id);
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group = LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = layers;
    parameter_type.layer_elements = matrix_elements;
    parameter_type.tensor_elements = tensor_elements;
    parameter_type.matrix_width = width;
    parameter_type.views_per_layer = 1;
    parameter_type.enumerate_matrix_view =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? llmc_enumerate_mlp_wup_rectangular_view
            : llmc_enumerate_mlp_wdown_rectangular_view;

    std::vector<float> master(tensor_elements);
    std::vector<float> gradient(tensor_elements, 0.0f);
    std::vector<float> momentum(tensor_elements, 0.0f);
    std::vector<float> second(tensor_elements, 0.01f);
    for (size_t index = 0; index < tensor_elements; ++index) {
        master[index] = 0.017f * std::cos(0.13f * static_cast<float>(index + 1));
    }
    // Start inside the well-conditioned inverse-root basin: after Frobenius
    // normalization the smaller-side Gram is exactly I/width.  The rejection
    // branch below then supplies a deliberately different dense gradient.
    for (int layer = 0; layer < layers; ++layer) {
        const size_t layer_base = static_cast<size_t>(layer) * matrix_elements;
        const float scale = 1.0f + 0.25f * static_cast<float>(layer);
        for (int diagonal = 0; diagonal < width; ++diagonal) {
            const size_t offset =
                family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
                    ? static_cast<size_t>(diagonal) * width + diagonal
                    : static_cast<size_t>(diagonal) * (4 * width) + diagonal;
            gradient[layer_base + offset] = scale;
        }
    }
    BatchedBuffers buffers(tensor_elements);
    buffers.load(master, gradient, momentum, second);

    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "CacheMuon inverse-root tracker BF16 runtime allocates");
    BATCHED_CHECK(
        runtime.batch_float_matrix_count == 3U &&
            runtime.tracked_q_elements ==
                static_cast<size_t>(layers) *
                    LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER *
                    small_elements &&
            runtime.cache_tracker_stats != nullptr &&
            runtime.cache_tracker_host_stats != nullptr &&
            runtime.cache_tracker_step_view_diagnostics != nullptr,
        "CacheMuon inverse-root tracker exposes finite state and telemetry storage");

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
            0.01f,
            1.0f,
            0U),
        "CacheMuon inverse-root tracker first BF16 update succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));
    const int family_slot =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0 : 1;
    const LlmcCacheMuonTrackerDiagnostics first_diagnostics =
        runtime.cache_tracker_step_diagnostics[family_slot];
    BATCHED_CHECK(
        first_diagnostics.probe_count == layers &&
            first_diagnostics.refresh_count == layers &&
            first_diagnostics.r_q_valid_count == 0U &&
            first_diagnostics.relative_motion_valid_count == 0U &&
            first_diagnostics.solve_relative_residual_valid_count == 0U &&
            first_diagnostics.spd_trust_valid_count == 0U &&
            first_diagnostics.antisymmetry_valid_count == 0U &&
            first_diagnostics.refresh_reason_invalid_count == layers,
        "first use refreshes invalid transforms without averaging placeholder metrics");
    bool first_use_refreshed = true;
    for (int layer = 0; layer < layers; ++layer) {
        const size_t q_index =
            static_cast<size_t>(layer) *
                LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER +
            (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 1U);
        first_use_refreshed = first_use_refreshed &&
            runtime.q_valid[q_index] != 0U &&
            runtime.refresh_count[q_index] == 1U &&
            runtime.last_refresh_step[q_index] == 0;
    }
    BATCHED_CHECK(
        first_use_refreshed,
        "first-use FreshGNS initializes each tracked inverse root exactly once");

    runtime.cache_step_probe_count = 0U;
    runtime.cache_step_miss_count = 0U;
    runtime.cache_step_residual_sum = 0.0;
    runtime.cache_step_residual_max = 0.0f;
    memset(
        runtime.cache_tracker_step_diagnostics,
        0,
        sizeof(runtime.cache_tracker_step_diagnostics));
    memset(
        runtime.cache_tracker_step_view_diagnostics,
        0,
        runtime.cache_tracker_step_view_diagnostic_capacity *
            sizeof(LlmcCacheMuonTrackerViewDiagnostics));

    if (force_tracker_rejection) {
        for (size_t index = 0; index < tensor_elements; ++index) {
            gradient[index] =
                0.031f * std::cos(0.29f * static_cast<float>(index + 3)) -
                0.013f * std::sin(0.17f * static_cast<float>(index + 7));
        }
        batched_copy_to_device(buffers.gradient, quantize_floatx(gradient));
        config.cache_tracker_motion_threshold = 1.0e-12f;
    }
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
            0.01f,
            1.0f,
            1U),
        "CacheMuon inverse-root tracker second BF16 update succeeds");
    cudaCheck(cudaStreamSynchronize(main_stream));

    const LlmcCacheMuonTrackerDiagnostics& second_diagnostics =
        runtime.cache_tracker_step_diagnostics[family_slot];
    const uint64_t expected_second_refreshes =
        force_tracker_rejection ? static_cast<uint64_t>(layers) : 0U;
    if (second_diagnostics.refresh_count != expected_second_refreshes) {
        fprintf(
            stderr,
            "cache-tracker debug family=%d forced=%d probes=%llu refresh=%llu "
            "reasons invalid=%llu nonfinite=%llu rq=%llu malformed=%llu "
            "solve=%llu motion=%llu spd=%llu maxima rq=%g solve=%g motion=%g "
            "spd=%g anti=%g\n",
            family_id,
            force_tracker_rejection ? 1 : 0,
            static_cast<unsigned long long>(second_diagnostics.probe_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_reason_invalid_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_reason_nonfinite_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_reason_r_q_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_reason_malformed_solve_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_reason_solve_relative_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_reason_relative_motion_count),
            static_cast<unsigned long long>(second_diagnostics.refresh_reason_spd_trust_count),
            second_diagnostics.r_q_max,
            second_diagnostics.solve_relative_residual_max,
            second_diagnostics.relative_motion_max,
            second_diagnostics.spd_trust_max,
            second_diagnostics.antisymmetry_max);
    }
    BATCHED_CHECK(
        second_diagnostics.probe_count == layers &&
            second_diagnostics.refresh_count == expected_second_refreshes &&
            runtime.cache_step_probe_count == layers &&
            runtime.cache_step_miss_count == expected_second_refreshes,
        force_tracker_rejection
            ? "a tight motion gate refreshes rejected inverse-root updates"
            : "default trust gates accept a well-conditioned lagged inverse-root update");
    if (force_tracker_rejection) {
        BATCHED_CHECK(
            second_diagnostics.refresh_reason_relative_motion_count == layers,
            "the additional motion refresh trigger is reported per rejected view");
    } else {
        BATCHED_CHECK(
            second_diagnostics.refresh_reason_invalid_count == 0U &&
                second_diagnostics.refresh_reason_nonfinite_count == 0U &&
                second_diagnostics.refresh_reason_r_q_count == 0U &&
                second_diagnostics.refresh_reason_malformed_solve_count == 0U &&
                second_diagnostics.refresh_reason_solve_relative_count == 0U &&
                second_diagnostics.refresh_reason_relative_motion_count == 0U &&
                second_diagnostics.refresh_reason_spd_trust_count == 0U,
            "accepted inverse-root updates clear every refresh OR gate");
    }

    const size_t view_base =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? 0U
            : runtime.batch_matrix_capacity;
    bool finite_view_diagnostics = true;
    const uint32_t all_metric_mask =
        LLMC_CACHEMUON_TRACKER_METRIC_R_Q |
        LLMC_CACHEMUON_TRACKER_METRIC_RELATIVE_MOTION |
        LLMC_CACHEMUON_TRACKER_METRIC_SOLVE_RELATIVE_RESIDUAL |
        LLMC_CACHEMUON_TRACKER_METRIC_SPD_TRUST |
        LLMC_CACHEMUON_TRACKER_METRIC_ANTISYMMETRY;
    for (int matrix_index = 0; matrix_index < layers; ++matrix_index) {
        const LlmcCacheMuonTrackerViewDiagnostics& view =
            runtime.cache_tracker_step_view_diagnostics[
                view_base + static_cast<size_t>(matrix_index)];
        finite_view_diagnostics = finite_view_diagnostics && view.valid &&
            view.family_id == family_id &&
            view.matrix_index == matrix_index &&
            view.refreshed == force_tracker_rejection &&
            view.metric_valid_mask == all_metric_mask &&
            std::isfinite(view.r_q) &&
            std::isfinite(view.relative_motion) &&
            std::isfinite(view.solve_relative_residual) &&
            std::isfinite(view.spd_trust) &&
            std::isfinite(view.antisymmetry);
    }
    BATCHED_CHECK(
        finite_view_diagnostics &&
            second_diagnostics.r_q_valid_count == layers &&
            second_diagnostics.relative_motion_valid_count == layers &&
            second_diagnostics.solve_relative_residual_valid_count == layers &&
            second_diagnostics.spd_trust_valid_count == layers &&
            second_diagnostics.antisymmetry_valid_count == layers &&
            std::isfinite(second_diagnostics.r_q_sum) &&
            std::isfinite(second_diagnostics.relative_motion_sum) &&
            std::isfinite(second_diagnostics.solve_relative_residual_sum) &&
            std::isfinite(second_diagnostics.spd_trust_sum) &&
            std::isfinite(second_diagnostics.antisymmetry_sum),
        "CacheMuon tracker emits finite family and per-view diagnostics");

    const std::vector<float> tracked = batched_copy_from_device(
        runtime.tracked_q, runtime.tracked_q_elements);
    float maximum_antisymmetry = 0.0f;
    bool refresh_counts_match = true;
    bool cumulative_view_counts_match = true;
    for (int layer = 0; layer < layers; ++layer) {
        const size_t q_index =
            static_cast<size_t>(layer) *
                LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER +
            (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 1U);
        const size_t base = q_index * small_elements;
        refresh_counts_match = refresh_counts_match &&
            runtime.refresh_count[q_index] ==
                (force_tracker_rejection ? 2U : 1U) &&
            runtime.last_refresh_step[q_index] ==
                (force_tracker_rejection ? 1 : 0);
        const LlmcCacheMuonTrackerDiagnostics& cumulative_view =
            runtime.cache_tracker_total_view_diagnostics[
                view_base + static_cast<size_t>(layer)];
        cumulative_view_counts_match = cumulative_view_counts_match &&
            cumulative_view.probe_count == 2U &&
            cumulative_view.refresh_count ==
                (force_tracker_rejection ? 2U : 1U) &&
            cumulative_view.r_q_valid_count == 1U &&
            cumulative_view.relative_motion_valid_count == 1U &&
            cumulative_view.solve_relative_residual_valid_count == 1U &&
            cumulative_view.spd_trust_valid_count == 1U &&
            cumulative_view.antisymmetry_valid_count == 1U;
        for (int row = 0; row < width; ++row) {
            for (int column = 0; column < width; ++column) {
                maximum_antisymmetry = std::max(
                    maximum_antisymmetry,
                    fabsf(
                        tracked[base + static_cast<size_t>(row) * width + column] -
                        tracked[base + static_cast<size_t>(column) * width + row]));
            }
        }
    }
    const LlmcCacheMuonTrackerDiagnostics& total_diagnostics =
        runtime.cache_tracker_total_diagnostics[family_slot];
    BATCHED_CHECK(
        refresh_counts_match && cumulative_view_counts_match &&
            total_diagnostics.probe_count == 2U * layers &&
            total_diagnostics.refresh_count ==
                static_cast<uint64_t>(layers) + expected_second_refreshes &&
            total_diagnostics.r_q_valid_count == layers &&
            total_diagnostics.relative_motion_valid_count == layers &&
            total_diagnostics.solve_relative_residual_valid_count == layers &&
            total_diagnostics.spd_trust_valid_count == layers &&
            total_diagnostics.antisymmetry_valid_count == layers,
        "family and per-view totals preserve refreshes and metric-valid denominators");
    BATCHED_CHECK(
        maximum_antisymmetry < 1.0e-6f && batched_all_finite(tracked) &&
            batched_all_finite(
                batched_copy_from_device(buffers.master, tensor_elements)),
        "accepted and refreshed inverse-root transforms remain symmetric and finite");
    llmc_normuon_runtime_free(&runtime);
}

static void run_cachemuon_inverse_root_tracker_cpu_reference(int family_id) {
    constexpr int side = 3;
    constexpr int tall_rows = 4 * side;
    constexpr size_t matrix_elements =
        static_cast<size_t>(tall_rows) * side;
    constexpr size_t small_elements = static_cast<size_t>(side) * side;
    const bool tall = family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP;
    const int rows = tall ? tall_rows : side;
    const int columns = tall ? side : tall_rows;

    std::vector<float> tall_normalized(matrix_elements, 0.0f);
    const float leading_rows[small_elements] = {
        1.02f, 0.04f, -0.02f,
        0.01f, 0.97f, 0.03f,
        -0.03f, 0.02f, 1.01f,
    };
    std::copy(
        leading_rows,
        leading_rows + small_elements,
        tall_normalized.begin());
    for (int row = side; row < tall_rows; ++row) {
        for (int column = 0; column < side; ++column) {
            tall_normalized[static_cast<size_t>(row) * side + column] =
                0.006f * std::sin(
                    0.37f * static_cast<float>((row + 1) * (column + 2)));
        }
    }
    const std::vector<float> normalized = tall
        ? tall_normalized
        : transpose_reference(tall_normalized, tall_rows, side);
    const std::vector<float> transform = {
        0.84f, 0.07f, -0.03f,
        0.07f, 1.16f, 0.05f,
        -0.03f, 0.05f, 0.96f,
    };

    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER;
    config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    config.cache_residual_threshold = 1.0e6f;
    config.cache_tracker_motion_threshold = 1.0e6f;
    config.cache_tracker_solve_relative_threshold = 1.0e6f;
    config.cache_tracker_spd_trust_threshold = 0.99f;
    config.cache_tracker_cg_iterations = 2U;
    llmc_normuon_resolve_schedules(&config);
    const CacheMuonInverseRootCpuReference reference =
        cachemuon_inverse_root_tracker_cpu_reference(
            normalized,
            rows,
            columns,
            transform,
            config.cache_tracker_cg_iterations,
            config.epsilon);

    LlmcOptimizerPlan plan = batched_plan(side, 1);
    plan.normuon_view_count = LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER;
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id = tall ? 10 : 12;
    parameter_type.name = "cachemuon_inverse_root_cpu_reference_test";
    parameter_type.family_id = static_cast<LlmcOptimizerFamilyId>(family_id);
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group = LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = 1;
    parameter_type.layer_elements = matrix_elements;
    parameter_type.tensor_elements = matrix_elements;
    parameter_type.matrix_width = side;
    parameter_type.views_per_layer = 1;
    parameter_type.enumerate_matrix_view = tall
        ? llmc_enumerate_mlp_wup_rectangular_view
        : llmc_enumerate_mlp_wdown_rectangular_view;

    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    const bool allocated =
        llmc_normuon_runtime_allocate(&runtime, &plan, &config);
    BATCHED_CHECK(
        allocated,
        "CPU-reference CacheMuon tracker runtime allocates");
    if (!allocated) return;
    const size_t q_index = tall ? 0U : 1U;
    std::vector<float> tracked(runtime.tracked_q_elements, 0.0f);
    std::copy(
        transform.begin(),
        transform.end(),
        tracked.begin() + q_index * small_elements);
    batched_copy_to_device(runtime.tracked_q, tracked);
    runtime.q_valid[q_index] = 1U;
    batched_copy_to_device(runtime.matrix[0], normalized);

    const float* direction_device = nullptr;
    float* refreshed_transform = nullptr;
    int miss_count = -1;
    const bool succeeded = llmc_cachemuon_direction_batched_bf16(
        &runtime,
        cublas_handle,
        main_stream,
        &config,
        &parameter_type,
        rows,
        columns,
        1,
        &direction_device,
        &refreshed_transform,
        &miss_count);
    cudaCheck(cudaStreamSynchronize(main_stream));
    BATCHED_CHECK(
        succeeded && miss_count == 0 && refreshed_transform == nullptr &&
            direction_device == runtime.matrix[1],
        "nonzero hybrid correction is accepted without a FreshGNS fallback");
    if (!succeeded || miss_count != 0 || direction_device == nullptr) {
        llmc_normuon_runtime_free(&runtime);
        return;
    }

    const std::vector<float> direction = batched_copy_from_device(
        direction_device, matrix_elements);
    const std::vector<float> k = batched_copy_from_device(
        runtime.cache_small[2], small_elements);
    const std::vector<float> delta = batched_copy_from_device(
        runtime.cache_small[4], small_elements);
    const std::vector<float> next_workspace = batched_copy_from_device(
        runtime.cache_small[3], small_elements);
    tracked = batched_copy_from_device(
        runtime.tracked_q, runtime.tracked_q_elements);
    const std::vector<float> next_tracked(
        tracked.begin() + q_index * small_elements,
        tracked.begin() + (q_index + 1U) * small_elements);

    const float direction_error = batched_max_abs_difference(
        direction, reference.lagged_direction);
    const float eager_direction_error = batched_max_abs_difference(
        direction, reference.next_direction);
    const float k_error = batched_max_abs_difference(k, reference.k);
    std::vector<float> negative_k = reference.k;
    for (float& value : negative_k) value = -value;
    const float negative_k_error = batched_max_abs_difference(k, negative_k);
    const float delta_error = batched_max_abs_difference(
        delta, reference.delta);
    const float unsandwiched_delta_error = batched_max_abs_difference(
        delta, reference.wrong_unsandwiched_delta);
    const float workspace_next_error = batched_max_abs_difference(
        next_workspace, reference.next_transform);
    const float tracked_next_error = batched_max_abs_difference(
        next_tracked, reference.next_transform);
    const float unsandwiched_next_error = batched_max_abs_difference(
        next_tracked, reference.wrong_unsandwiched_next);
    const std::vector<float> wrong_sign_next = symmetric_sum_reference(
        transform, reference.delta, side, -1.0f);
    const float wrong_sign_next_error = batched_max_abs_difference(
        next_tracked, wrong_sign_next);
    std::vector<float> direction_change(reference.next_direction.size(), 0.0f);
    for (size_t index = 0; index < direction_change.size(); ++index) {
        direction_change[index] =
            reference.next_direction[index] -
            reference.lagged_direction[index];
    }
    const bool delta_contract_ok =
        delta_error < 5.0e-3f &&
        delta_error + 1.0e-3f < unsandwiched_delta_error;
    const bool next_contract_ok =
        workspace_next_error < 5.0e-3f &&
        tracked_next_error < 5.0e-3f &&
        tracked_next_error + 1.0e-3f < unsandwiched_next_error &&
        tracked_next_error + 1.0e-2f < wrong_sign_next_error;
    if (!delta_contract_ok || !next_contract_ok) {
        fprintf(
            stderr,
            "cache-tracker CPU debug family=%d direction=%g eager=%g k=%g "
            "negative_k=%g delta=%g unsandwiched_delta=%g workspace_next=%g "
            "tracked_next=%g unsandwiched_next=%g wrong_sign_next=%g "
            "k_norm=%g delta_norm=%g direction_change_norm=%g\n",
            family_id,
            direction_error,
            eager_direction_error,
            k_error,
            negative_k_error,
            delta_error,
            unsandwiched_delta_error,
            workspace_next_error,
            tracked_next_error,
            unsandwiched_next_error,
            wrong_sign_next_error,
            frobenius_norm_reference(reference.k),
            frobenius_norm_reference(reference.delta),
            frobenius_norm_reference(direction_change));
    }
    BATCHED_CHECK(
        frobenius_norm_reference(reference.k) > 1.0e-2f &&
            frobenius_norm_reference(reference.delta) > 1.0e-2f &&
            frobenius_norm_reference(direction_change) > 1.0e-2f,
        "CPU fixture exercises a genuinely nonzero tracked correction");
    BATCHED_CHECK(
        direction_error < 3.0e-3f &&
            direction_error + 3.0e-3f < eager_direction_error,
        "hybrid emits the old-transform direction and commits Pnext only for the next step");
    BATCHED_CHECK(
        k_error < 4.0e-3f && k_error + 1.0e-2f < negative_k_error,
        "two-step CG solves P K + K P = -E with the correct sign");
    BATCHED_CHECK(
        delta_contract_ok,
        "hybrid forms the oriented sandwiched correction delta=P K P");
    BATCHED_CHECK(
        next_contract_ok,
        "accepted Pnext=P+delta is symmetric, correctly oriented, and committed");

    const size_t view_index = tall ? 0U : runtime.batch_matrix_capacity;
    const LlmcCacheMuonTrackerViewDiagnostics& view =
        runtime.cache_tracker_step_view_diagnostics[view_index];
    const uint32_t all_metric_mask =
        LLMC_CACHEMUON_TRACKER_METRIC_R_Q |
        LLMC_CACHEMUON_TRACKER_METRIC_RELATIVE_MOTION |
        LLMC_CACHEMUON_TRACKER_METRIC_SOLVE_RELATIVE_RESIDUAL |
        LLMC_CACHEMUON_TRACKER_METRIC_SPD_TRUST |
        LLMC_CACHEMUON_TRACKER_METRIC_ANTISYMMETRY;
    const LlmcCacheMuonTrackerDiagnostics& cumulative_view =
        runtime.cache_tracker_total_view_diagnostics[view_index];
    BATCHED_CHECK(
        view.valid && !view.refreshed &&
            view.refresh_reason_mask == LLMC_CACHEMUON_TRACKER_REFRESH_NONE &&
            view.metric_valid_mask == all_metric_mask &&
            view.relative_motion > 0.0f &&
            view.solve_relative_residual >= 0.0f &&
            view.spd_trust > 0.0f && view.spd_trust < 0.99f &&
            std::isfinite(view.antisymmetry) &&
            cumulative_view.probe_count == 1U &&
            cumulative_view.refresh_count == 0U &&
            cumulative_view.r_q_valid_count == 1U &&
            cumulative_view.relative_motion_valid_count == 1U &&
            cumulative_view.solve_relative_residual_valid_count == 1U &&
            cumulative_view.spd_trust_valid_count == 1U &&
            cumulative_view.antisymmetry_valid_count == 1U,
        "accepted correction exposes finite per-step and cumulative view telemetry");
    llmc_normuon_runtime_free(&runtime);
}

static void test_rectangular_batched_cachemuon_inverse_root_tracker() {
    run_rectangular_batched_cachemuon_inverse_root_tracker(
        LLMC_OPTIMIZER_FAMILY_MLP_WUP, false);
    run_rectangular_batched_cachemuon_inverse_root_tracker(
        LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, false);
    run_rectangular_batched_cachemuon_inverse_root_tracker(
        LLMC_OPTIMIZER_FAMILY_MLP_WUP, true);
    run_rectangular_batched_cachemuon_inverse_root_tracker(
        LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, true);
    run_cachemuon_inverse_root_tracker_cpu_reference(
        LLMC_OPTIMIZER_FAMILY_MLP_WUP);
    run_cachemuon_inverse_root_tracker_cpu_reference(
        LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
}

static void test_cachemuon_inverse_root_tracker_companion() {
    constexpr int width = 3;
    constexpr int layers = 1;
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER;
    config.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    config.cache_tracker_motion_threshold = 0.125f;
    config.cache_tracker_solve_relative_threshold = 0.375f;
    config.cache_tracker_spd_trust_threshold = 0.625f;
    config.cache_tracker_cg_iterations = 3U;
    llmc_normuon_resolve_schedules(&config);

    LlmcOptimizerPlan plan = batched_plan(width, layers);
    plan.normuon_view_count =
        layers * LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER;
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&runtime, &plan, &config),
        "v10 CacheMuon tracker companion source runtime allocates");
    BATCHED_CHECK(
        LLMC_NORMUON_COMPANION_VERSION ==
            LLMC_NORMUON_COMPANION_VERSION_CACHE_INVERSE_ROOT_TRACKER &&
            LLMC_NORMUON_COMPANION_VERSION == 10U,
        "CacheMuon inverse-root tracker owns companion schema v10");

    const char* v10_path =
        "build/test_normuon_batched_v10_cache_inverse_root_tracker.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            v10_path, 7, 1, 0, &plan, &config, &runtime, main_stream),
        "v10 CacheMuon inverse-root tracker companion saves");
    LlmcNormuonCompanionInfo v10_info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(v10_path, &v10_info) &&
            v10_info.config.orthogonalization_mode ==
                LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER &&
            v10_info.config.cache_tracker_motion_threshold ==
                config.cache_tracker_motion_threshold &&
            v10_info.config.cache_tracker_solve_relative_threshold ==
                config.cache_tracker_solve_relative_threshold &&
            v10_info.config.cache_tracker_spd_trust_threshold ==
                config.cache_tracker_spd_trust_threshold &&
            v10_info.config.cache_tracker_cg_iterations ==
                config.cache_tracker_cg_iterations,
        "v10 companion round-trips every inverse-root tracker control exactly");

    LlmcNormuonRuntime resumed;
    llmc_normuon_runtime_reset(&resumed);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&resumed, &plan, &config) &&
            llmc_normuon_load_companion(
                v10_path, 7, 1, 0, &plan, &config, &resumed, main_stream),
        "v10 companion resumes with the exact inverse-root tracker controls");
    LlmcNormuonConfig mismatch = config;
    mismatch.cache_tracker_motion_threshold = 0.126f;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            v10_path, 7, 1, 0, &plan, &mismatch, &resumed, main_stream),
        "v10 resume rejects a motion-threshold mismatch");
    mismatch = config;
    mismatch.cache_tracker_solve_relative_threshold = 0.376f;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            v10_path, 7, 1, 0, &plan, &mismatch, &resumed, main_stream),
        "v10 resume rejects a solve-threshold mismatch");
    mismatch = config;
    mismatch.cache_tracker_spd_trust_threshold = 0.626f;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            v10_path, 7, 1, 0, &plan, &mismatch, &resumed, main_stream),
        "v10 resume rejects an SPD-threshold mismatch");
    mismatch = config;
    mismatch.cache_tracker_cg_iterations = 4U;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            v10_path, 7, 1, 0, &plan, &mismatch, &resumed, main_stream),
        "v10 resume rejects a CG-iteration mismatch");

    const char* v9_path =
        "build/test_normuon_batched_v9_cache_inverse_root_tracker.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            v9_path, 7, 1, 0, &plan, &config, &runtime, main_stream),
        "v9 compatibility fixture companion saves from the v10 layout");
    FILE* v9_file = fopen(v9_path, "r+b");
    BATCHED_CHECK(v9_file != nullptr, "v9 compatibility fixture opens");
    if (v9_file != nullptr) {
        const int v9 = static_cast<int>(
            LLMC_NORMUON_COMPANION_VERSION_TRACKER_SPECTRAL_PMAX);
        const float ignored_motion = 0.875f;
        const float ignored_solve = 0.775f;
        const float ignored_spd = 0.675f;
        const int ignored_iterations = 11;
        fseek(v9_file, sizeof(int), SEEK_SET);
        fwrite(&v9, sizeof(v9), 1U, v9_file);
        fseek(v9_file, 40L * static_cast<long>(sizeof(int)), SEEK_SET);
        fwrite(&ignored_motion, sizeof(ignored_motion), 1U, v9_file);
        fwrite(&ignored_solve, sizeof(ignored_solve), 1U, v9_file);
        fwrite(&ignored_spd, sizeof(ignored_spd), 1U, v9_file);
        fwrite(&ignored_iterations, sizeof(ignored_iterations), 1U, v9_file);
        fclose(v9_file);
    }
    LlmcNormuonCompanionInfo v9_info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(v9_path, &v9_info) &&
            v9_info.config.orthogonalization_mode ==
                LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER &&
            v9_info.config.cache_tracker_motion_threshold ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_MOTION_THRESHOLD &&
            v9_info.config.cache_tracker_solve_relative_threshold ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_SOLVE_RELATIVE_THRESHOLD &&
            v9_info.config.cache_tracker_spd_trust_threshold ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_SPD_TRUST_THRESHOLD &&
            v9_info.config.cache_tracker_cg_iterations ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_CG_ITERATIONS,
        "v9 companions ignore v10 slots and receive the exact tracker defaults");
    LlmcNormuonConfig v9_config = config;
    v9_config.cache_tracker_motion_threshold =
        LLMC_CACHEMUON_TRACKER_DEFAULT_MOTION_THRESHOLD;
    v9_config.cache_tracker_solve_relative_threshold =
        LLMC_CACHEMUON_TRACKER_DEFAULT_SOLVE_RELATIVE_THRESHOLD;
    v9_config.cache_tracker_spd_trust_threshold =
        LLMC_CACHEMUON_TRACKER_DEFAULT_SPD_TRUST_THRESHOLD;
    v9_config.cache_tracker_cg_iterations =
        LLMC_CACHEMUON_TRACKER_DEFAULT_CG_ITERATIONS;
    LlmcNormuonRuntime v9_runtime;
    llmc_normuon_runtime_reset(&v9_runtime);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&v9_runtime, &plan, &v9_config) &&
            llmc_normuon_load_companion(
                v9_path,
                7,
                1,
                0,
                &plan,
                &v9_config,
                &v9_runtime,
                main_stream),
        "v9 companion resumes only with the v10 defaulted tracker controls");
    LlmcNormuonConfig v9_mismatch = v9_config;
    v9_mismatch.cache_tracker_motion_threshold = 0.11f;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            v9_path,
            7,
            1,
            0,
            &plan,
            &v9_mismatch,
            &v9_runtime,
            main_stream),
        "v9 compatibility resume rejects a nondefault v10 tracker control");

    remove(v10_path);
    remove(v9_path);
    llmc_normuon_runtime_free(&v9_runtime);
    llmc_normuon_runtime_free(&resumed);
    llmc_normuon_runtime_free(&runtime);
}

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
    BATCHED_CHECK(
        config.tracker_spectral_pmax ==
            LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX,
        "square tracker spectral pmax retains the historical 1.2 default");
    BATCHED_CHECK(
        config.cache_tracker_motion_threshold ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_MOTION_THRESHOLD &&
            config.cache_tracker_solve_relative_threshold ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_SOLVE_RELATIVE_THRESHOLD &&
            config.cache_tracker_spd_trust_threshold ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_SPD_TRUST_THRESHOLD &&
            config.cache_tracker_cg_iterations ==
                LLMC_CACHEMUON_TRACKER_DEFAULT_CG_ITERATIONS,
        "CacheMuon inverse-root tracker retains the v10 trust defaults");
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
    LlmcNormuonOrthogonalizationMode orthogonalization =
        LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
    BATCHED_CHECK(
        llmc_parse_normuon_orthogonalization_mode(
            "rectangular_cache_muon", &orthogonalization) &&
            orthogonalization == LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON,
        "rectangular CacheMuon mode parses explicitly");
    BATCHED_CHECK(
        llmc_parse_normuon_orthogonalization_mode(
            "rectangular_cache_muon_inverse_root_tracker",
            &orthogonalization) &&
            orthogonalization ==
                LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER &&
            strcmp(
                llmc_normuon_orthogonalization_mode_name(orthogonalization),
                "rectangular_cache_muon_inverse_root_tracker") == 0,
        "CacheMuon inverse-root tracker mode parses and names exactly");
    LlmcNormuonApproximationPolicy approximation =
        LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    BATCHED_CHECK(
        llmc_parse_normuon_approximation_policy(
            "cache_muon_gram_gns", &approximation) &&
            approximation == LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS,
        "FreshGNS policy parses explicitly");

    LlmcNormuonConfig fresh_scratch = config;
    fresh_scratch.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    fresh_scratch.orthogonalization_mode = LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
    fresh_scratch.refresh_policy = LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS;
    fresh_scratch.retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    char config_error[256];
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &fresh_scratch, config_error, sizeof(config_error)),
        "rectangular scratch accepts FreshGNS as an every-step solver");
    LlmcNormuonConfig invalid_fresh_scratch = fresh_scratch;
    invalid_fresh_scratch.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_fresh_scratch, config_error, sizeof(config_error)),
        "square scratch rejects the rectangular FreshGNS solver policy");

    LlmcNormuonConfig cache_tracker = config;
    cache_tracker.optimizer_selection =
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    cache_tracker.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER;
    cache_tracker.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker accepts its BF16 trust contract");
    LlmcNormuonConfig invalid_cache_tracker = cache_tracker;
    invalid_cache_tracker.execution_mode =
        LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker rejects the unsupported FP32 path");
    invalid_cache_tracker = cache_tracker;
    invalid_cache_tracker.cache_tracker_motion_threshold = 0.0f;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker rejects a zero motion threshold");
    invalid_cache_tracker = cache_tracker;
    invalid_cache_tracker.cache_tracker_solve_relative_threshold = NAN;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker rejects a nonfinite solve threshold");
    invalid_cache_tracker = cache_tracker;
    invalid_cache_tracker.cache_tracker_spd_trust_threshold = 1.0f;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker keeps the SPD trust threshold below one");
    invalid_cache_tracker = cache_tracker;
    invalid_cache_tracker.cache_tracker_cg_iterations = 0U;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker rejects zero CG iterations");
    invalid_cache_tracker.cache_tracker_cg_iterations =
        LLMC_CACHEMUON_TRACKER_MAX_CG_ITERATIONS + 1U;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker rejects an unbounded CG loop");
    invalid_cache_tracker = cache_tracker;
    invalid_cache_tracker.correction_gain = 0.5f;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_cache_tracker, config_error, sizeof(config_error)),
        "CacheMuon inverse-root tracker rejects a non-unit correction gain");

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
    LlmcNormuonTrackerCorrectionMode correction_mode =
        LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS;
    BATCHED_CHECK(
        llmc_parse_normuon_tracker_correction_mode(
            "diagonal_sylvester", &correction_mode) &&
            correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER,
        "diagonal Sylvester tracker correction mode parses");
    BATCHED_CHECK(
        llmc_parse_normuon_tracker_correction_mode("0", &correction_mode) &&
            correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS,
        "global Frobenius tracker correction mode parses");
    BATCHED_CHECK(
        llmc_parse_normuon_tracker_correction_mode(
            "basis_free_first_order", &correction_mode) &&
            correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER,
        "basis-free first-order tracker correction mode parses");
    BATCHED_CHECK(
        llmc_parse_normuon_tracker_correction_mode("2", &correction_mode) &&
            correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER,
        "basis-free first-order enum value 2 parses");
    BATCHED_CHECK(
        !llmc_parse_normuon_tracker_correction_mode(
            "silent_substitution", &correction_mode),
        "unknown tracker correction mode is rejected");
    LlmcNormuonConfig commuted = config;
    commuted.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    commuted.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    commuted.correction_policy =
        LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    commuted.correction_iterations = 2U;
    commuted.correction_mode =
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER;
    commuted.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &commuted, config_error, sizeof(config_error)),
        "commuted mode accepts its exact canonical2 tracker contract");
    LlmcNormuonConfig basis_free = commuted;
    basis_free.correction_mode =
        LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER;
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &basis_free, config_error, sizeof(config_error)),
        "basis-free first-order mode accepts square tracker views");
    basis_free.tracker_spectral_pmax = 1.40f;
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &basis_free, config_error, sizeof(config_error)),
        "basis-free first-order mode accepts an explicit pmax of 1.4");
    LlmcNormuonConfig invalid_pmax = basis_free;
    invalid_pmax.tracker_spectral_pmax = 1.0f;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_pmax, config_error, sizeof(config_error)),
        "square tracker rejects pmax at the singular-value floor");
    invalid_pmax.tracker_spectral_pmax =
        LLMC_NORMUON_TRACKER_SPECTRAL_PMAX_MAX + 0.01f;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_pmax, config_error, sizeof(config_error)),
        "square tracker rejects pmax above the tested trust basin");
    invalid_pmax.tracker_spectral_pmax = NAN;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &invalid_pmax, config_error, sizeof(config_error)),
        "square tracker rejects nonfinite pmax");
    basis_free.tracker_spectral_pmax =
        LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX;
    basis_free.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q;
    BATCHED_CHECK(
        !llmc_normuon_validate_config(
            &basis_free, config_error, sizeof(config_error)),
        "basis-free first-order mode rejects rectangular tracker views");
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

    LlmcNormuonConfig cache_workspace_config;
    llmc_normuon_config_defaults(&cache_workspace_config);
    cache_workspace_config.optimizer_selection =
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    cache_workspace_config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON;
    cache_workspace_config.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(
            &runtime, &plan, &cache_workspace_config),
        "baseline rectangular CacheMuon workspace allocates");
    const size_t baseline_cache_workspace_bytes = runtime.workspace_bytes;
    BATCHED_CHECK(
        runtime.batch_float_matrix_count == 3U &&
            runtime.cache_tracker_stats == nullptr,
        "baseline CacheMuon workspace remains unchanged by the hybrid mode");
    llmc_normuon_runtime_free(&runtime);

    cache_workspace_config.orthogonalization_mode =
        LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON_INVERSE_ROOT_TRACKER;
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(
            &runtime, &plan, &cache_workspace_config),
        "CacheMuon inverse-root tracker workspace allocates");
    bool has_all_cache_panels = true;
    for (float* panel : runtime.cache_small) {
        has_all_cache_panels = has_all_cache_panels && panel != nullptr;
    }
    BATCHED_CHECK(
        runtime.workspace_bytes == baseline_cache_workspace_bytes +
                runtime.batch_matrix_capacity *
                    LLMC_CACHEMUON_TRACKER_STATS_STRIDE * sizeof(float) &&
            runtime.batch_float_matrix_count == 3U &&
            runtime.matrix[0] != nullptr && runtime.matrix[1] != nullptr &&
            runtime.matrix[2] != nullptr && runtime.matrix[3] == nullptr &&
            has_all_cache_panels && runtime.cache_tracker_stats != nullptr &&
            runtime.cache_tracker_host_stats != nullptr &&
            runtime.cache_tracker_step_view_diagnostics != nullptr &&
            runtime.cache_tracker_total_view_diagnostics != nullptr &&
            runtime.cache_tracker_step_view_diagnostic_capacity ==
                2U * runtime.batch_matrix_capacity,
        "hybrid reuses CacheMuon's five square panels and adds only finite telemetry workspace");
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

    const char* current_path = "build/test_normuon_batched_current.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            current_path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "batched tracker current companion saves");
    LlmcNormuonCompanionInfo info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(current_path, &info) &&
            info.config.execution_mode ==
                LLMC_NORMUON_EXECUTION_BF16_BATCHED &&
            info.config.correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS &&
            info.config.tracker_spectral_pmax ==
                LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX,
        "current v10 companion records BF16 mode, correction mode, and default pmax");
    LlmcNormuonRuntime resumed;
    llmc_normuon_runtime_reset(&resumed);
    BATCHED_CHECK(
        llmc_normuon_runtime_allocate(&resumed, &plan, &config) &&
            llmc_normuon_load_companion(
                current_path, 2, 1, 0, &plan, &config, &resumed, main_stream),
        "current batched tracker state resumes exactly");
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
        "current resume preserves Q and tracker phase bit-exactly");
    const char* v8_path = "build/test_normuon_batched_v8.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            v8_path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "v8 compatibility fixture companion saves");
    FILE* v8_file = fopen(v8_path, "r+b");
    BATCHED_CHECK(v8_file != nullptr, "v8 compatibility fixture opens");
    if (v8_file != nullptr) {
        const int v8 = static_cast<int>(
            LLMC_NORMUON_COMPANION_VERSION_ADAPTIVE_PENDING);
        const float ignored_nondefault_pmax = 1.40f;
        fseek(v8_file, sizeof(int), SEEK_SET);
        fwrite(&v8, sizeof(v8), 1U, v8_file);
        fseek(v8_file, 39L * static_cast<long>(sizeof(int)), SEEK_SET);
        fwrite(
            &ignored_nondefault_pmax,
            sizeof(ignored_nondefault_pmax),
            1U,
            v8_file);
        fclose(v8_file);
    }
    LlmcNormuonCompanionInfo v8_info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(v8_path, &v8_info) &&
            v8_info.config.tracker_spectral_pmax ==
                LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX,
        "v8 companion defaults tracker spectral pmax to 1.2");
    BATCHED_CHECK(
        llmc_normuon_load_companion(
            v8_path,
            2,
            1,
            0,
            &plan,
            &config,
            &resumed,
            main_stream),
        "v8 companion remains loadable with the historical pmax default");
    const char* v3_path = "build/test_normuon_batched_v3.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            v3_path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "v3 compatibility fixture companion saves");
    FILE* v3_file = fopen(v3_path, "r+b");
    BATCHED_CHECK(v3_file != nullptr, "v3 compatibility fixture opens");
    if (v3_file != nullptr) {
        const int v3 = static_cast<int>(
            LLMC_NORMUON_COMPANION_VERSION_TRACKER_CORRECTION_MODE - 1U);
        const int ignored_correction_mode = static_cast<int>(
            LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER);
        fseek(v3_file, sizeof(int), SEEK_SET);
        fwrite(&v3, sizeof(v3), 1U, v3_file);
        fseek(v3_file, 32L * static_cast<long>(sizeof(int)), SEEK_SET);
        fwrite(&ignored_correction_mode, sizeof(ignored_correction_mode), 1U,
               v3_file);
        fclose(v3_file);
    }
    LlmcNormuonCompanionInfo v3_info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(v3_path, &v3_info) &&
            v3_info.config.correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS,
        "v3 companion defaults the new correction mode to global Frobenius");
    BATCHED_CHECK(
        llmc_normuon_load_companion(
            v3_path,
            2,
            1,
            0,
            &plan,
            &config,
            &resumed,
            main_stream),
        "v3 companion remains loadable after the correction-mode extension");
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
    remove(v8_path);
    remove(v3_path);
    remove(v2_path);
    remove(v1_path);
    llmc_normuon_runtime_free(&v1_runtime);
    llmc_normuon_runtime_free(&resumed);
    llmc_normuon_runtime_free(&runtime);
}

static void test_batched_commuted_tracker(
    LlmcNormuonTrackerCorrectionMode correction_mode,
    float tracker_spectral_pmax =
        LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX,
    float correction_gain = 1.0f) {
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
    config.correction_gain = correction_gain;
    config.tracker_spectral_pmax = tracker_spectral_pmax;
    config.correction_mode = correction_mode;
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
    if (correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER) {
        BATCHED_CHECK(
            llmc_normuon_enable_tracker_diagnostics(&runtime, 1U),
            "basis-free tracker enables exact spectral-guard telemetry");
    }
    llmc_normuon_tracker_diagnostics_begin_step(&runtime, 0U);
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
    llmc_normuon_tracker_diagnostics_begin_step(&runtime, 1U);
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

    if (correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER) {
        const LlmcNormuonTrackerFamilyDiagnostics& diagnostics =
            runtime.tracker_step_diagnostics[0];
        BATCHED_CHECK(
            diagnostics.valid &&
                diagnostics.correction_guard_probe_count == matrix_count &&
                std::isfinite(diagnostics.correction_raw_frobenius_mean) &&
                std::isfinite(
                    diagnostics.correction_spectral_norm_estimate_mean) &&
                std::isfinite(diagnostics.correction_guard_scale_mean),
            "basis-free diagnostics report exact finite guard telemetry");
        if (correction_gain < 1.0e-3f) {
            BATCHED_CHECK(
                diagnostics.correction_guard_clipped_count == 0U &&
                    diagnostics.correction_guard_clipped_fraction == 0.0f &&
                    diagnostics.correction_guard_scale_min == 1.0f,
                "tiny basis-free correction records an unclipped guard");
        }
        if (correction_gain > 10.0f) {
            BATCHED_CHECK(
                diagnostics.correction_guard_clipped_count == matrix_count &&
                    diagnostics.correction_guard_clipped_fraction == 1.0f &&
                    diagnostics.correction_guard_scale_min < 1.0f,
                "large basis-free correction records a clipped guard");
        }
    }

    const char* path = "build/test_normuon_batched_v10_commuted.bin";
    BATCHED_CHECK(
        llmc_normuon_save_companion(
            path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "v10 commuted tracker companion saves");
    LlmcNormuonCompanionInfo info;
    BATCHED_CHECK(
        llmc_normuon_read_companion_info(path, &info) &&
            info.config.retraction_mode ==
                LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2 &&
            info.config.tracker_spectral_pmax == tracker_spectral_pmax,
        "v10 companion records commuted retraction and exact spectral pmax");
    BATCHED_CHECK(
        llmc_normuon_load_companion(
            path, 2, 1, 0, &plan, &config, &runtime, main_stream),
        "v10 companion resumes with the exact spectral pmax");
    LlmcNormuonConfig mode_mismatch = config;
    mode_mismatch.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            path, 2, 1, 0, &plan, &mode_mismatch, &runtime, main_stream),
        "resume rejects a tracker retraction-mode mismatch");
    LlmcNormuonConfig pmax_mismatch = config;
    pmax_mismatch.tracker_spectral_pmax =
        tracker_spectral_pmax == 1.40f ? 1.20f : 1.40f;
    BATCHED_CHECK(
        !llmc_normuon_load_companion(
            path, 2, 1, 0, &plan, &pmax_mismatch, &runtime, main_stream),
        "v10 resume rejects a tracker spectral-pmax mismatch");
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

static void test_batched_adaptive_tracker_refresh() {
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
    config.orthogonalization_mode = LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    config.tracker_refresh_mode =
        LLMC_NORMUON_TRACKER_REFRESH_ADAPTIVE_MEAN_SKEW;
    config.refresh_interval = 3U;
    config.tracker_max_refresh_age = 9U;
    config.tracker_wup_skew_threshold = 1.0e6f;
    config.tracker_wdown_skew_threshold = 1.0e6f;
    config.refresh_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config.correction_policy = LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config.correction_iterations = 2U;
    config.retraction_mode =
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    char config_error[256];
    BATCHED_CHECK(
        llmc_normuon_validate_config(
            &config, config_error, sizeof(config_error)),
        "adaptive square-tracker configuration validates");

    LlmcOptimizerPlan plan = batched_plan(width, layers);
    LlmcOptimizerParameterType parameter_type =
        batched_parameter_type(family_id, width, layers);
    std::vector<float> master_logical(tensor_elements);
    std::vector<float> momentum_logical(tensor_elements, 0.0f);
    std::vector<float> second_physical(tensor_elements, 0.01f);
    for (size_t index = 0; index < tensor_elements; ++index) {
        master_logical[index] =
            0.12f - 0.0005f * static_cast<float>(index % 127U);
    }
    const std::vector<float> master_physical =
        pack_logical_tensor(master_logical, family_id, width);
    const std::vector<float> momentum_physical =
        pack_logical_tensor(momentum_logical, family_id, width);

    {
        BatchedBuffers buffers(tensor_elements);
        buffers.load(
            master_physical,
            pack_logical_tensor(
                tracker_logical_gradient(width, layers, 0), family_id, width),
            momentum_physical,
            second_physical);
        LlmcNormuonRuntime runtime;
        llmc_normuon_runtime_reset(&runtime);
        BATCHED_CHECK(
            llmc_normuon_runtime_allocate(&runtime, &plan, &config),
            "adaptive tracker runtime allocates its scalar decision staging");
        for (uint64_t step = 0U; step <= 9U; ++step) {
            const std::vector<float> gradient =
                tracker_logical_gradient(width, layers, static_cast<int>(step));
            batched_copy_to_device(
                buffers.gradient,
                quantize_floatx(
                    pack_logical_tensor(gradient, family_id, width)));
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
                    step),
                "adaptive tracker high-threshold step succeeds");
        }
        cudaCheck(cudaStreamSynchronize(main_stream));
        BATCHED_CHECK(
            runtime.tracker_adaptive_initial_refresh_count[0] == 1U &&
                runtime.tracker_adaptive_check_count[0] == 8U &&
                runtime.tracker_adaptive_threshold_refresh_count[0] == 0U &&
                runtime.tracker_adaptive_forced_refresh_count[0] == 1U,
            "adaptive tracker checks every stale step then forces age nine");
        for (int view = 0; view < LLMC_NORMUON_VIEWS_PER_MLP_MATRIX; ++view) {
            const size_t q_index = static_cast<size_t>(view);
            BATCHED_CHECK(
                runtime.q_valid[q_index] == 1U &&
                    runtime.refresh_count[q_index] == 2U &&
                    runtime.last_refresh_step[q_index] == 9,
                "adaptive hard ceiling refreshes every family view together");
        }
        const char* adaptive_path =
            "build/test_normuon_batched_adaptive_v8.bin";
        BATCHED_CHECK(
            llmc_normuon_save_companion(
                adaptive_path,
                10,
                1,
                0,
                &plan,
                &config,
                &runtime,
                main_stream),
            "adaptive tracker companion saves");
        LlmcNormuonCompanionInfo adaptive_info;
        BATCHED_CHECK(
            llmc_normuon_read_companion_info(
                adaptive_path, &adaptive_info) &&
                adaptive_info.config.tracker_refresh_mode ==
                    LLMC_NORMUON_TRACKER_REFRESH_ADAPTIVE_MEAN_SKEW &&
                adaptive_info.config.tracker_max_refresh_age == 9U &&
                adaptive_info.config.tracker_wup_skew_threshold == 1.0e6f &&
                !adaptive_info.tracker_adaptive_refresh_due[0],
            "v8 companion records adaptive state and contract exactly");
        remove(adaptive_path);
        llmc_normuon_runtime_free(&runtime);
    }

    {
        LlmcNormuonConfig threshold_config = config;
        threshold_config.tracker_wup_skew_threshold = 1.0e-8f;
        BatchedBuffers buffers(tensor_elements);
        buffers.load(
            master_physical,
            pack_logical_tensor(
                tracker_logical_gradient(width, layers, 0), family_id, width),
            momentum_physical,
            second_physical);
        LlmcNormuonRuntime runtime;
        llmc_normuon_runtime_reset(&runtime);
        BATCHED_CHECK(
            llmc_normuon_runtime_allocate(
                &runtime, &plan, &threshold_config),
            "adaptive threshold tracker runtime allocates");
        for (uint64_t step = 0U; step <= 1U; ++step) {
            const std::vector<float> gradient =
                tracker_logical_gradient(width, layers, static_cast<int>(step));
            batched_copy_to_device(
                buffers.gradient,
                quantize_floatx(
                    pack_logical_tensor(gradient, family_id, width)));
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
                    &threshold_config,
                    learning_rate,
                    1.0f,
                    step),
                "adaptive tracker threshold-crossing step succeeds");
        }
        cudaCheck(cudaStreamSynchronize(main_stream));
        BATCHED_CHECK(
            runtime.tracker_adaptive_check_count[0] == 1U &&
                runtime.tracker_adaptive_threshold_refresh_count[0] == 0U &&
                runtime.tracker_adaptive_refresh_due[0],
            "every-step skew crossing schedules the next-step refresh");

        const char* pending_path =
            "build/test_normuon_batched_adaptive_pending_v8.bin";
        BATCHED_CHECK(
            llmc_normuon_save_companion(
                pending_path,
                2,
                1,
                0,
                &plan,
                &threshold_config,
                &runtime,
                main_stream),
            "adaptive tracker pending decision saves");
        LlmcNormuonRuntime resumed_runtime;
        llmc_normuon_runtime_reset(&resumed_runtime);
        BATCHED_CHECK(
            llmc_normuon_runtime_allocate(
                &resumed_runtime, &plan, &threshold_config) &&
                llmc_normuon_load_companion(
                    pending_path,
                    2,
                    1,
                    0,
                    &plan,
                    &threshold_config,
                    &resumed_runtime,
                    main_stream) &&
                resumed_runtime.tracker_adaptive_refresh_due[0],
            "v8 resume preserves a pending adaptive refresh exactly");
        remove(pending_path);
        llmc_normuon_runtime_free(&resumed_runtime);

        const uint64_t step = 2U;
        const std::vector<float> gradient =
            tracker_logical_gradient(width, layers, static_cast<int>(step));
        batched_copy_to_device(
            buffers.gradient,
            quantize_floatx(
                pack_logical_tensor(gradient, family_id, width)));
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
                &threshold_config,
                learning_rate,
                1.0f,
                step),
            "pending adaptive refresh executes on the next step");
        cudaCheck(cudaStreamSynchronize(main_stream));
        BATCHED_CHECK(
            runtime.tracker_adaptive_check_count[0] == 1U &&
                runtime.tracker_adaptive_threshold_refresh_count[0] == 1U &&
                runtime.tracker_adaptive_forced_refresh_count[0] == 0U &&
                !runtime.tracker_adaptive_refresh_due[0],
            "family-mean skew crossing refreshes one step after measurement");
        for (int view = 0; view < LLMC_NORMUON_VIEWS_PER_MLP_MATRIX; ++view) {
            const size_t q_index = static_cast<size_t>(view);
            BATCHED_CHECK(
                runtime.refresh_count[q_index] == 2U &&
                    runtime.last_refresh_step[q_index] == 2,
                "threshold crossing refreshes all family views atomically");
        }
        llmc_normuon_runtime_free(&runtime);
    }
}

int main() {
    char server_ip[2] = "";
    char filesystem_path[2] = "";
    char init_method[4] = "mpi";
    multi_gpu_config = multi_gpu_config_init(
        1, 0, 1, server_ip, filesystem_path, init_method);
    set_zero_configs(&multi_gpu_config, 0, 1);
    common_start(false, false);

    test_lr_dither_schedule();
    test_tracker_h_stability_kernel();
    test_square_wdown_batch_replay_restore();
    test_rectangular_batched_update();
    test_rectangular_batched_tracker();
    test_rectangular_batched_cachemuon();
    test_rectangular_batched_cachemuon_inverse_root_tracker();
    test_cachemuon_inverse_root_tracker_companion();
    test_execution_mode_and_workspace();
    test_batched_layout_scratch_and_guard();
    test_batched_tracker_and_checkpoint();
    test_batched_adaptive_tracker_refresh();
    test_polynomial_factor_override_preserves_packed_q();
    test_batched_commuted_tracker(
        LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS);
    test_batched_commuted_tracker(
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER);
    test_batched_commuted_tracker(
        LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER);
    test_batched_commuted_tracker(
        LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER,
        1.40f,
        1.0e-5f);
    test_batched_commuted_tracker(
        LLMC_NORMUON_TRACKER_CORRECTION_BASIS_FREE_FIRST_ORDER,
        1.40f,
        100.0f);

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
