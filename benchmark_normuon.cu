#define TESTING
#include "train_gpt2.cu"

#include <cstdio>

constexpr int BENCHMARK_WIDTH = 768;
constexpr int BENCHMARK_LAYERS = 12;
constexpr int BENCHMARK_REPETITIONS = 3;

__global__ void benchmark_initialize_kernel(
    floatX* parameter,
    floatX* gradient,
    float* momentum,
    float* second_moment,
    float* master,
    size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        uint32_t bits = static_cast<uint32_t>(index) * 1664525U + 1013904223U;
        bits ^= bits >> 15U;
        const float unit = static_cast<float>(bits & 0xFFFFU) / 65535.0f;
        const float gradient_value = 0.02f * (unit - 0.5f);
        const float master_value = 0.002f * (0.5f - unit);
        gradient[index] = static_cast<floatX>(gradient_value);
        momentum[index] = 0.0f;
        second_moment[index] = 0.01f;
        master[index] = master_value;
        parameter[index] = static_cast<floatX>(master_value);
    }
}

struct BenchmarkBuffers {
    explicit BenchmarkBuffers(size_t element_count) : elements(element_count) {
        cudaCheck(cudaMalloc(&parameter, elements * sizeof(floatX)));
        cudaCheck(cudaMalloc(&gradient, elements * sizeof(floatX)));
        cudaCheck(cudaMalloc(&momentum, elements * sizeof(float)));
        cudaCheck(cudaMalloc(&second_moment, elements * sizeof(float)));
        cudaCheck(cudaMalloc(&master, elements * sizeof(float)));
    }

    ~BenchmarkBuffers() {
        cudaFree(parameter);
        cudaFree(gradient);
        cudaFree(momentum);
        cudaFree(second_moment);
        cudaFree(master);
    }

    void initialize(cudaStream_t stream) {
        benchmark_initialize_kernel<<<65535U, 256U, 0, stream>>>(
            parameter,
            gradient,
            momentum,
            second_moment,
            master,
            elements);
        cudaCheck(cudaGetLastError());
    }

    size_t elements;
    floatX* parameter = nullptr;
    floatX* gradient = nullptr;
    float* momentum = nullptr;
    float* second_moment = nullptr;
    float* master = nullptr;
};

static LlmcOptimizerPlan benchmark_plan() {
    LlmcOptimizerPlan plan;
    llmc_optimizer_plan_reset(&plan);
    plan.built = true;
    plan.num_layers = BENCHMARK_LAYERS;
    plan.channels = BENCHMARK_WIDTH;
    plan.normuon_parameter_type_count = 2;
    plan.normuon_view_count =
        BENCHMARK_LAYERS * LLMC_NORMUON_VIEWS_PER_LAYER;
    return plan;
}

static LlmcOptimizerParameterType benchmark_parameter_type(
    LlmcOptimizerFamilyId family_id) {
    LlmcOptimizerParameterType parameter_type = {};
    parameter_type.tensor_id =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 10 : 12;
    parameter_type.name =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? "fcw" : "fcprojw";
    parameter_type.family_id = family_id;
    parameter_type.backend_kind = LLMC_OPTIMIZER_BACKEND_NORMUON;
    parameter_type.hyperparameter_group =
        LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP;
    parameter_type.weight_decay_policy = LLMC_WEIGHT_DECAY_ENABLED;
    parameter_type.layer_multiplicity = BENCHMARK_LAYERS;
    parameter_type.layer_elements =
        LLMC_NORMUON_VIEWS_PER_MLP_MATRIX *
        static_cast<size_t>(BENCHMARK_WIDTH) * BENCHMARK_WIDTH;
    parameter_type.tensor_elements =
        BENCHMARK_LAYERS * parameter_type.layer_elements;
    parameter_type.matrix_width = BENCHMARK_WIDTH;
    parameter_type.views_per_layer = LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    parameter_type.enumerate_matrix_view =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
            ? llmc_enumerate_mlp_wup_view
            : llmc_enumerate_mlp_wdown_view;
    return parameter_type;
}

static bool benchmark_update_serial(
    LlmcNormuonRuntime* runtime,
    BenchmarkBuffers* buffers,
    const LlmcOptimizerParameterType* parameter_type,
    const LlmcNormuonConfig* config,
    uint64_t global_step) {
    for (int layer = 0; layer < parameter_type->layer_multiplicity; ++layer) {
        const size_t layer_offset =
            static_cast<size_t>(layer) * parameter_type->layer_elements;
        for (int view_index = 0;
             view_index < parameter_type->views_per_layer;
             ++view_index) {
            LlmcOptimizerMatrixView view;
            if (!parameter_type->enumerate_matrix_view(
                    parameter_type, view_index, &view)) {
                return false;
            }
            const size_t parameter_offset = layer_offset + view.element_offset;
            const size_t second_offset =
                layer_offset + view.second_moment_offset;
            if (!llmc_normuon_update_view(
                    runtime,
                    cublas_handle,
                    main_stream,
                    buffers->parameter + parameter_offset,
                    buffers->gradient + parameter_offset,
                    buffers->momentum + parameter_offset,
                    buffers->second_moment + second_offset,
                    buffers->master + parameter_offset,
                    parameter_type,
                    &view,
                    config,
                    config->learning_rate,
                    1.0f,
                    global_step,
                    layer,
                    view_index)) {
                return false;
            }
        }
    }
    return true;
}

static bool benchmark_update_batched(
    LlmcNormuonRuntime* runtime,
    BenchmarkBuffers* buffers,
    const LlmcOptimizerParameterType* parameter_type,
    const LlmcNormuonConfig* config,
    uint64_t global_step) {
    return llmc_normuon_update_parameter_type_batched_bf16(
        runtime,
        cublas_handle,
        main_stream,
        buffers->parameter,
        buffers->gradient,
        buffers->momentum,
        buffers->second_moment,
        buffers->master,
        parameter_type,
        config,
        config->learning_rate,
        1.0f,
        global_step);
}

struct BenchmarkResult {
    float mean_milliseconds = 0.0f;
    size_t allocation_bytes = 0U;
    size_t workspace_bytes = 0U;
    size_t q_bytes = 0U;
};

static BenchmarkResult benchmark_case(
    LlmcNormuonExecutionMode execution_mode,
    LlmcNormuonOrthogonalizationMode orthogonalization_mode,
    LlmcNormuonTrackerRetractionMode retraction_mode,
    uint64_t measured_step,
    bool initialize_tracker) {
    LlmcNormuonConfig config;
    llmc_normuon_config_defaults(&config);
    config.optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
    config.execution_mode = execution_mode;
    config.orthogonalization_mode = orthogonalization_mode;
    config.refresh_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config.correction_policy =
        LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config.refresh_interval = 3U;
    config.correction_iterations = 2U;
    config.retraction_mode = retraction_mode;
    llmc_normuon_resolve_schedules(&config);

    size_t free_before = 0U;
    size_t total_bytes = 0U;
    cudaCheck(cudaMemGetInfo(&free_before, &total_bytes));
    LlmcOptimizerPlan plan = benchmark_plan();
    LlmcNormuonRuntime runtime;
    llmc_normuon_runtime_reset(&runtime);
    if (!llmc_normuon_runtime_allocate(&runtime, &plan, &config)) {
        fprintf(stderr, "benchmark runtime allocation failed\n");
        exit(EXIT_FAILURE);
    }
    const LlmcOptimizerParameterType wup_type =
        benchmark_parameter_type(LLMC_OPTIMIZER_FAMILY_MLP_WUP);
    const LlmcOptimizerParameterType wdown_type =
        benchmark_parameter_type(LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
    BenchmarkBuffers wup(wup_type.tensor_elements);
    BenchmarkBuffers wdown(wdown_type.tensor_elements);
    wup.initialize(main_stream);
    wdown.initialize(main_stream);
    cudaCheck(cudaStreamSynchronize(main_stream));
    size_t free_after = 0U;
    cudaCheck(cudaMemGetInfo(&free_after, &total_bytes));

    auto update_all = [&](uint64_t step) {
        const bool batched =
            execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED;
        const bool wup_ok = batched
            ? benchmark_update_batched(
                  &runtime, &wup, &wup_type, &config, step)
            : benchmark_update_serial(
                  &runtime, &wup, &wup_type, &config, step);
        const bool wdown_ok = batched
            ? benchmark_update_batched(
                  &runtime, &wdown, &wdown_type, &config, step)
            : benchmark_update_serial(
                  &runtime, &wdown, &wdown_type, &config, step);
        if (!wup_ok || !wdown_ok) {
            fprintf(stderr, "benchmark update failed\n");
            exit(EXIT_FAILURE);
        }
    };

    if (initialize_tracker) {
        update_all(0U);
    }
    update_all(measured_step);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start, main_stream));
    for (int repetition = 0;
         repetition < BENCHMARK_REPETITIONS;
         ++repetition) {
        update_all(measured_step);
    }
    cudaCheck(cudaEventRecord(stop, main_stream));
    cudaCheck(cudaEventSynchronize(stop));
    float total_milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&total_milliseconds, start, stop));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    BenchmarkResult result;
    result.mean_milliseconds =
        total_milliseconds / static_cast<float>(BENCHMARK_REPETITIONS);
    result.allocation_bytes = free_before - free_after;
    result.workspace_bytes = runtime.workspace_bytes;
    result.q_bytes = runtime.tracked_q_bytes;
    llmc_normuon_runtime_free(&runtime);
    return result;
}

static void print_case(
    const char* operation,
    const BenchmarkResult& reference,
    const BenchmarkResult& batched) {
    printf(
        "normuon_benchmark operation=%s fp32_reference_ms=%.6f "
        "bf16_batched_ms=%.6f speedup=%.3fx "
        "fp32_allocation_mib=%.3f bf16_allocation_mib=%.3f "
        "fp32_workspace_mib=%.3f bf16_workspace_mib=%.3f q_mib=%.3f\n",
        operation,
        reference.mean_milliseconds,
        batched.mean_milliseconds,
        reference.mean_milliseconds / batched.mean_milliseconds,
        static_cast<double>(reference.allocation_bytes) / (1024.0 * 1024.0),
        static_cast<double>(batched.allocation_bytes) / (1024.0 * 1024.0),
        static_cast<double>(reference.workspace_bytes) / (1024.0 * 1024.0),
        static_cast<double>(batched.workspace_bytes) / (1024.0 * 1024.0),
        static_cast<double>(batched.q_bytes) / (1024.0 * 1024.0));
}

int main() {
    char server_ip[2] = "";
    char filesystem_path[2] = "";
    char init_method[4] = "mpi";
    multi_gpu_config = multi_gpu_config_init(
        1, 0, 1, server_ip, filesystem_path, init_method);
    set_zero_configs(&multi_gpu_config, 0, 1);
    common_start(false, false);

    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, 0));
    int driver_version = 0;
    int runtime_version = 0;
    cudaCheck(cudaDriverGetVersion(&driver_version));
    cudaCheck(cudaRuntimeGetVersion(&runtime_version));
    printf(
        "normuon_benchmark_device name=%s compute=%d.%d driver=%d runtime=%d "
        "width=%d layers=%d views=96 repetitions=%d\n",
        properties.name,
        properties.major,
        properties.minor,
        driver_version,
        runtime_version,
        BENCHMARK_WIDTH,
        BENCHMARK_LAYERS,
        BENCHMARK_REPETITIONS);
    LlmcNormuonConfig schedules;
    llmc_normuon_config_defaults(&schedules);
    printf("normuon_benchmark_refresh_schedule=");
    for (int stage = 0; stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT; ++stage) {
        const LlmcNormuonPolynomialStep coefficient =
            schedules.refresh_schedule[stage];
        printf(
            "%s%.9g,%.9g,%.9g",
            stage == 0 ? "" : ";",
            coefficient.a,
            coefficient.b,
            coefficient.c);
    }
    printf("\nnormuon_benchmark_correction_schedule=");
    for (uint32_t stage = 0; stage < schedules.correction_iterations; ++stage) {
        const LlmcNormuonPolynomialStep coefficient =
            schedules.correction_schedule[stage];
        printf(
            "%s%.9g,%.9g,%.9g",
            stage == 0U ? "" : ";",
            coefficient.a,
            coefficient.b,
            coefficient.c);
    }
    printf("\n");

    const BenchmarkResult scratch_reference = benchmark_case(
        LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
        LLMC_NORMUON_ORTHO_NEWTON_SCHULZ,
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        1U,
        false);
    const BenchmarkResult scratch_batched = benchmark_case(
        LLMC_NORMUON_EXECUTION_BF16_BATCHED,
        LLMC_NORMUON_ORTHO_NEWTON_SCHULZ,
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        1U,
        false);
    print_case("scratch_stock5", scratch_reference, scratch_batched);

    const BenchmarkResult refresh_reference = benchmark_case(
        LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q,
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        0U,
        false);
    const BenchmarkResult refresh_batched = benchmark_case(
        LLMC_NORMUON_EXECUTION_BF16_BATCHED,
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q,
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        0U,
        false);
    print_case("tracker_refresh_stock5", refresh_reference, refresh_batched);

    const BenchmarkResult correction_reference = benchmark_case(
        LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q,
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        1U,
        true);
    const BenchmarkResult correction_batched = benchmark_case(
        LLMC_NORMUON_EXECUTION_BF16_BATCHED,
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q,
        LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ,
        1U,
        true);
    print_case(
        "tracker_correction_canonical2_newton_schulz",
        correction_reference,
        correction_batched);

    const BenchmarkResult commuted_correction_reference = benchmark_case(
        LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q,
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2,
        1U,
        true);
    const BenchmarkResult commuted_correction_batched = benchmark_case(
        LLMC_NORMUON_EXECUTION_BF16_BATCHED,
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q,
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2,
        1U,
        true);
    print_case(
        "tracker_correction_canonical2_commuted_stage2",
        commuted_correction_reference,
        commuted_correction_batched);
    printf(
        "normuon_benchmark tracker_cadence_newton_schulz_fp32_ms=%.6f "
        "tracker_cadence_newton_schulz_bf16_ms=%.6f "
        "tracker_cadence_newton_schulz_speedup=%.3fx\n",
        (refresh_reference.mean_milliseconds +
         2.0f * correction_reference.mean_milliseconds) / 3.0f,
        (refresh_batched.mean_milliseconds +
         2.0f * correction_batched.mean_milliseconds) / 3.0f,
        (refresh_reference.mean_milliseconds +
         2.0f * correction_reference.mean_milliseconds) /
            (refresh_batched.mean_milliseconds +
             2.0f * correction_batched.mean_milliseconds));
    printf(
        "normuon_benchmark tracker_cadence_commuted_fp32_ms=%.6f "
        "tracker_cadence_commuted_bf16_ms=%.6f "
        "tracker_cadence_commuted_speedup=%.3fx "
        "commutation_vs_newton_schulz_fp32=%.3fx "
        "commutation_vs_newton_schulz_bf16=%.3fx\n",
        (refresh_reference.mean_milliseconds +
         2.0f * commuted_correction_reference.mean_milliseconds) / 3.0f,
        (refresh_batched.mean_milliseconds +
         2.0f * commuted_correction_batched.mean_milliseconds) / 3.0f,
        (refresh_reference.mean_milliseconds +
         2.0f * commuted_correction_reference.mean_milliseconds) /
            (refresh_batched.mean_milliseconds +
             2.0f * commuted_correction_batched.mean_milliseconds),
        correction_reference.mean_milliseconds /
            commuted_correction_reference.mean_milliseconds,
        correction_batched.mean_milliseconds /
            commuted_correction_batched.mean_milliseconds);

    GPT2 unused_model = {};
    common_free(unused_model);
    multi_gpu_config_free(&multi_gpu_config);
    return EXIT_SUCCESS;
}
