/*
Blockwise NorMuon for the llm.c GPT-2 trainer, including square-view tracker
updates and the optional proper rectangular scratch/tracker updates.

This header intentionally owns only the small llm.c-facing optimizer plan and
the single-GPU CUDA implementation. It does not depend on the FRNA host runtime.
*/
#ifndef LLMC_NORMUON_CUH
#define LLMC_NORMUON_CUH

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

constexpr int LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT = 16;
constexpr int LLMC_OPTIMIZER_FAMILY_COUNT = 6;
constexpr int LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT = 5;
constexpr int LLMC_NORMUON_VIEWS_PER_LAYER = 8;
constexpr int LLMC_NORMUON_VIEWS_PER_MLP_MATRIX = 4;
constexpr int LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER = 2;
constexpr int LLMC_NORMUON_RECTANGULAR_VIEWS_PER_MLP_MATRIX = 1;
constexpr int LLMC_CACHEMUON_SMALL_PANEL_COUNT = 5;
constexpr uint32_t LLMC_NORMUON_COMPANION_MAGIC = 20260716U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION_PREVIOUS = 5U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION_EXECUTION_MODE = 2U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION_FP32_ONLY = 1U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION_TRACKER_CORRECTION_MODE = 4U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION_WDOWN_LR_MULTIPLIER = 5U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION_CACHE_MUON = 6U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION =
    LLMC_NORMUON_COMPANION_VERSION_CACHE_MUON;
constexpr int LLMC_NORMUON_COMPANION_HEADER_INTS = 256;
constexpr int LLMC_NORMUON_BLOCK_SIZE = 256;
constexpr size_t LLMC_NORMUON_BATCH_STATS_STRIDE = 4U;
// Damped diagonal-Sylvester tracker correction defaults.  The damping floor
// is a fraction of the dimension-normalized symmetric phase scale, while the
// cap is a dimensionless trust coefficient: the raw correction Frobenius norm
// is bounded by kappa * sqrt(d) before the correction polynomial.  The
// spectral guard keeps the singular values of I + Omega inside the stable
// basin of the canonical Taylor retraction.
constexpr float LLMC_NORMUON_TRACKER_DAMPING_ETA = 0.05f;
constexpr float LLMC_NORMUON_TRACKER_CORRECTION_CAP = 0.25f;
constexpr float LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX = 1.20f;
constexpr uint32_t LLMC_NORMUON_TRACKER_SPECTRAL_POWER_ITERATIONS = 4U;
// Rectangular tracker retraction defaults.  The horizontal component is
// capped to the trust-region slack below the unit singular-value target;
// the post-product candidate is then spectrally guarded before the canonical
// Taylor retraction is applied.
constexpr float LLMC_NORMUON_RECTANGULAR_HORIZONTAL_CAP =
    LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX - 1.0f;
constexpr float LLMC_NORMUON_RECTANGULAR_RETRACTION_PMAX =
    LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX;
// CacheMuon v1 (arXiv:2606.16371) uses FreshGNS with restart set {2},
// epsilon 1e-7, and gamma=5 for the reported GPT-2-small run.  Only gamma is
// exposed as a quality/efficiency control; the fresh solver remains pinned to
// the paper's five-stage coefficients and restart schedule.
constexpr float LLMC_CACHEMUON_EPSILON = 1.0e-7f;
constexpr float LLMC_CACHEMUON_DEFAULT_RESIDUAL_THRESHOLD = 5.0f;
constexpr uint32_t LLMC_CACHEMUON_RESTART_STAGE = 2U;

enum LlmcOptimizerSelection {
    LLMC_OPTIMIZER_SELECTION_ADAMW = 0,
    LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON = 1,
};

enum LlmcOptimizerBackendKind {
    LLMC_OPTIMIZER_BACKEND_ADAMW = 0,
    LLMC_OPTIMIZER_BACKEND_NORMUON = 1,
};

enum LlmcOptimizerFamilyId {
    LLMC_OPTIMIZER_FAMILY_EMBEDDINGS = 0,
    LLMC_OPTIMIZER_FAMILY_NORMALIZATION = 1,
    LLMC_OPTIMIZER_FAMILY_ATTENTION = 2,
    LLMC_OPTIMIZER_FAMILY_MLP_WUP = 3,
    LLMC_OPTIMIZER_FAMILY_MLP_WDOWN = 4,
    LLMC_OPTIMIZER_FAMILY_MLP_BIASES = 5,
};

enum LlmcOptimizerHyperparameterGroup {
    LLMC_OPTIMIZER_HYPERPARAM_ADAMW_DEFAULT = 0,
    LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP = 1,
};

enum LlmcWeightDecayPolicy {
    LLMC_WEIGHT_DECAY_DISABLED = 0,
    LLMC_WEIGHT_DECAY_ENABLED = 1,
};

enum LlmcNormuonOrthogonalizationMode {
    LLMC_NORMUON_ORTHO_NEWTON_SCHULZ = 0,
    LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q = 1,
    // Proper rectangular Muon using a smaller-side Gram matrix.  This is a
    // direct scratch-mode update; tracker state and retractions do not apply.
    LLMC_NORMUON_ORTHO_RECTANGULAR_MUON = 2,
    // Proper rectangular polar-factor tracking.  Wup is treated as tall and
    // Wdown as wide; the tracked factor keeps the native matrix layout.
    LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q = 3,
    // Split-family throughput ablation: square polar-factor tracking on Wup
    // and proper rectangular scratch Muon on Wdown.  The runtime keeps the
    // rectangular workspace stride so both family shapes share one batch
    // allocator, while the planner preserves each family's native views.
    LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON = 4,
    // Paper-faithful rectangular CacheMuon.  Each whole Wup/Wdown matrix
    // caches the smaller-side FreshGNS transform and probes Q_cache X on every
    // later step.  Only residual-gate misses run a fresh solve.
    LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON = 5,
};

enum LlmcNormuonApproximationPolicy {
    LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC = 0,
    LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC = 1,
    LLMC_NORMUON_APPROX_POLAR_EXPRESS = 2,
    LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS = 3,
};

enum LlmcNormuonExecutionMode {
    LLMC_NORMUON_EXECUTION_FP32_REFERENCE = 0,
    LLMC_NORMUON_EXECUTION_BF16_BATCHED = 1,
};

enum LlmcNormuonTrackerCorrectionMode {
    LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS = 0,
    LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER = 1,
};

enum LlmcNormuonTrackerRetractionMode {
    LLMC_NORMUON_TRACKER_RETRACTION_DISABLED = 0,
    LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ = 1,
    LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2 = 2,
    // Retract the small tangent factor before the single Q*B / B*Q GEMM;
    // the horizontal residual remains live and is guarded afterward.
    LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2 = 3,
};

inline bool llmc_normuon_is_rectangular_mode(
    LlmcNormuonOrthogonalizationMode mode) {
    return mode == LLMC_NORMUON_ORTHO_RECTANGULAR_MUON ||
           mode == LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q ||
           mode == LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON ||
           mode ==
               LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON;
}

inline bool llmc_normuon_is_cache_mode(
    LlmcNormuonOrthogonalizationMode mode) {
    return mode == LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON;
}

inline bool llmc_normuon_is_tracker_mode(
    LlmcNormuonOrthogonalizationMode mode) {
    return mode == LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q ||
           mode == LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q ||
           mode ==
               LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON;
}

inline bool llmc_normuon_has_persistent_transform(
    LlmcNormuonOrthogonalizationMode mode) {
    return llmc_normuon_is_tracker_mode(mode) || llmc_normuon_is_cache_mode(mode);
}

inline bool llmc_normuon_is_split_family_mode(
    LlmcNormuonOrthogonalizationMode mode) {
    return mode ==
           LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON;
}

inline LlmcNormuonOrthogonalizationMode llmc_normuon_mode_for_family(
    LlmcNormuonOrthogonalizationMode mode,
    int family_id) {
    if (mode ==
        LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON) {
        return family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
                   ? LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q
                   : LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
    }
    return mode;
}

inline bool llmc_normuon_family_is_rectangular(
    LlmcNormuonOrthogonalizationMode mode,
    int family_id) {
    return llmc_normuon_is_rectangular_mode(
        llmc_normuon_mode_for_family(mode, family_id));
}

inline bool llmc_normuon_family_is_tracker(
    LlmcNormuonOrthogonalizationMode mode,
    int family_id) {
    return llmc_normuon_is_tracker_mode(
        llmc_normuon_mode_for_family(mode, family_id));
}

inline int llmc_normuon_family_views_per_matrix(
    LlmcNormuonOrthogonalizationMode mode,
    int family_id) {
    return llmc_normuon_family_is_rectangular(mode, family_id)
               ? LLMC_NORMUON_RECTANGULAR_VIEWS_PER_MLP_MATRIX
               : LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
}

struct LlmcNormuonPolynomialStep {
    float a;
    float b;
    float c;
};

static constexpr LlmcNormuonPolynomialStep kLlmcCanonicalTaylorQuintic[
    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT] = {
    {1.875f, -1.25f, 0.375f},
    {1.875f, -1.25f, 0.375f},
    {1.875f, -1.25f, 0.375f},
    {1.875f, -1.25f, 0.375f},
    {1.875f, -1.25f, 0.375f},
};

static constexpr LlmcNormuonPolynomialStep kLlmcStockNormuonQuintic[
    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT] = {
    {3.4445f, -4.7750f, 2.0315f},
    {3.4445f, -4.7750f, 2.0315f},
    {3.4445f, -4.7750f, 2.0315f},
    {3.4445f, -4.7750f, 2.0315f},
    {3.4445f, -4.7750f, 2.0315f},
};

static constexpr LlmcNormuonPolynomialStep kLlmcPolarExpress[
    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

static constexpr LlmcNormuonPolynomialStep kLlmcCacheMuonGramGns[
    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT] = {
    {7.892582874f, -20.383013946f, 13.555306149f},
    {3.911484868f, -2.546463593f, 0.426898832f},
    {3.760657956f, -2.512819018f, 0.432364735f},
    {3.160399674f, -2.149649519f, 0.399636691f},
    {2.191097162f, -1.441662010f, 0.328146488f},
};

static constexpr LlmcNormuonPolynomialStep kLlmcRectangularCubicRetraction[1] = {
    {1.5f, -0.5f, 0.0f},
};

struct LlmcNormuonConfig {
    LlmcOptimizerSelection optimizer_selection;
    uint32_t targeted_family_mask;
    float learning_rate;
    float weight_decay;
    float momentum;
    float beta2;
    float epsilon;
    float update_scale;
    // Optional speedrun-style per-family multiplier.  The default keeps the
    // historical matched recipe unchanged; setting this to 2 applies the
    // explicit 2x c_proj/Wdown LR boost without changing Wup.
    float wdown_learning_rate_multiplier;
    LlmcNormuonExecutionMode execution_mode;
    LlmcNormuonOrthogonalizationMode orthogonalization_mode;
    LlmcNormuonApproximationPolicy refresh_policy;
    LlmcNormuonApproximationPolicy correction_policy;
    uint32_t refresh_interval;
    uint32_t correction_iterations;
    float correction_gain;
    float cache_residual_threshold;
    LlmcNormuonTrackerCorrectionMode correction_mode;
    LlmcNormuonTrackerRetractionMode retraction_mode;
    LlmcNormuonPolynomialStep refresh_schedule[LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT];
    LlmcNormuonPolynomialStep correction_schedule[LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT];
};

struct LlmcOptimizerMatrixView {
    size_t element_offset;
    size_t second_moment_offset;
    size_t rows;
    size_t columns;
    size_t row_stride;
    size_t column_stride;
};

struct LlmcOptimizerParameterType;
typedef bool (*LlmcOptimizerMatrixViewEnumerator)(
    const LlmcOptimizerParameterType* parameter_type,
    int view_index,
    LlmcOptimizerMatrixView* view);

struct LlmcOptimizerParameterType {
    int tensor_id;
    const char* name;
    LlmcOptimizerFamilyId family_id;
    LlmcOptimizerBackendKind backend_kind;
    LlmcOptimizerHyperparameterGroup hyperparameter_group;
    LlmcWeightDecayPolicy weight_decay_policy;
    int layer_multiplicity;
    size_t tensor_elements;
    size_t layer_elements;
    size_t matrix_width;
    int views_per_layer;
    LlmcOptimizerMatrixViewEnumerator enumerate_matrix_view;
};

struct LlmcOptimizerFamily {
    LlmcOptimizerFamilyId family_id;
    const char* name;
    int parameter_type_count;
    int parameter_type_indices[LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT];
};

struct LlmcOptimizerBackend {
    LlmcOptimizerBackendKind kind;
    const char* name;
};

struct LlmcOptimizerPlan {
    bool built;
    int num_layers;
    int channels;
    int family_count;
    LlmcOptimizerFamily families[LLMC_OPTIMIZER_FAMILY_COUNT];
    LlmcOptimizerParameterType parameter_types[LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT];
    LlmcOptimizerBackend backends[2];
    int normuon_parameter_type_count;
    int normuon_view_count;
};

struct LlmcNormuonRuntime {
    void* workspace_allocation;
    void* workspace;
    size_t workspace_bytes;
    size_t workspace_capacity_bytes;
    bool workspace_is_borrowed;
    size_t matrix_elements;
    float* matrix[5];
    size_t batch_float_matrix_count;
    size_t batch_matrix_capacity;
    size_t batch_total_elements;
    uint16_t* batch_bf16[2];
    float* axis_stats;
    size_t axis_stats_elements;
    float* stats;
    int* nonfinite_flag;
    float* tracked_q;
    size_t tracked_q_elements;
    size_t tracked_q_bytes;
    size_t tracked_q_view_count;
    uint8_t* q_valid;
    uint64_t* refresh_count;
    int64_t* last_refresh_step;
    size_t cache_small_total_elements;
    float* cache_small[LLMC_CACHEMUON_SMALL_PANEL_COUNT];
    float* cache_residuals;
    int* cache_miss_indices;
    float* cache_host_residuals;
    int* cache_host_miss_indices;
    uint64_t cache_step_probe_count;
    uint64_t cache_step_miss_count;
    double cache_step_residual_sum;
    float cache_step_residual_max;
    uint64_t cache_total_probe_count;
    uint64_t cache_total_miss_count;
    double cache_total_residual_sum;
    float cache_total_residual_max;
};

struct LlmcNormuonCompanionInfo {
    int step;
    int num_processes;
    int process_rank;
    int num_layers;
    int channels;
    size_t q_view_count;
    size_t q_element_count;
    LlmcNormuonConfig config;
};

inline uint32_t llmc_optimizer_family_mask(LlmcOptimizerFamilyId family_id) {
    return 1U << static_cast<uint32_t>(family_id);
}

inline uint32_t llmc_normuon_required_target_mask() {
    return llmc_optimizer_family_mask(LLMC_OPTIMIZER_FAMILY_MLP_WUP) |
           llmc_optimizer_family_mask(LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
}

inline const char* llmc_optimizer_selection_name(LlmcOptimizerSelection selection) {
    switch (selection) {
        case LLMC_OPTIMIZER_SELECTION_ADAMW: return "adamw";
        case LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON: return "adamw_normuon";
        default: return "invalid";
    }
}

inline const char* llmc_normuon_orthogonalization_mode_name(
    LlmcNormuonOrthogonalizationMode mode) {
    switch (mode) {
        case LLMC_NORMUON_ORTHO_NEWTON_SCHULZ: return "newton_schulz";
        case LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q: return "skew_polar_track_q";
        case LLMC_NORMUON_ORTHO_RECTANGULAR_MUON: return "rectangular_muon";
        case LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q:
            return "rectangular_skew_polar_track_q";
        case LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON:
            return "split_wup_square_tracker_wdown_rectangular_muon";
        case LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON:
            return "rectangular_cache_muon";
        default: return "invalid";
    }
}

inline const char* llmc_normuon_approximation_policy_name(
    LlmcNormuonApproximationPolicy policy) {
    switch (policy) {
        case LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC:
            return "canonical_taylor_quintic";
        case LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC:
            return "stock_normuon_quintic";
        case LLMC_NORMUON_APPROX_POLAR_EXPRESS:
            return "polar_express";
        case LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS:
            return "cache_muon_gram_gns";
        default:
            return "invalid";
    }
}

inline const char* llmc_normuon_execution_mode_name(
    LlmcNormuonExecutionMode mode) {
    switch (mode) {
        case LLMC_NORMUON_EXECUTION_FP32_REFERENCE:
            return "fp32_reference";
        case LLMC_NORMUON_EXECUTION_BF16_BATCHED:
            return "bf16_batched";
        default:
            return "invalid";
    }
}

inline const char* llmc_normuon_tracker_retraction_mode_name(
    LlmcNormuonTrackerRetractionMode mode) {
    switch (mode) {
        case LLMC_NORMUON_TRACKER_RETRACTION_DISABLED:
            return "disabled";
        case LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ:
            return "newton_schulz";
        case LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2:
            return "commuted_canonical_stage2";
        case LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2:
            return "thin_canonical_stage2";
        default:
            return "invalid";
    }
}

inline const char* llmc_normuon_tracker_correction_mode_name(
    LlmcNormuonTrackerCorrectionMode mode) {
    switch (mode) {
        case LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS:
            return "global_frobenius";
        case LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER:
            return "diagonal_sylvester";
        default:
            return "invalid";
    }
}

inline bool llmc_parse_optimizer_selection(
    const char* value,
    LlmcOptimizerSelection* selection) {
    if (value == nullptr || selection == nullptr) {
        return false;
    }
    if (strcmp(value, "adamw") == 0) {
        *selection = LLMC_OPTIMIZER_SELECTION_ADAMW;
        return true;
    }
    if (strcmp(value, "adamw_normuon") == 0) {
        *selection = LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
        return true;
    }
    return false;
}

inline bool llmc_parse_normuon_orthogonalization_mode(
    const char* value,
    LlmcNormuonOrthogonalizationMode* mode) {
    if (value == nullptr || mode == nullptr) {
        return false;
    }
    if (strcmp(value, "newton_schulz") == 0) {
        *mode = LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
        return true;
    }
    if (strcmp(value, "skew_polar_track_q") == 0) {
        *mode = LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
        return true;
    }
    if (strcmp(value, "rectangular_muon") == 0) {
        *mode = LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
        return true;
    }
    if (strcmp(value, "rectangular_skew_polar_track_q") == 0) {
        *mode = LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q;
        return true;
    }
    if (strcmp(value, "split_wup_square_tracker_wdown_rectangular_muon") == 0) {
        *mode =
            LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON;
        return true;
    }
    if (strcmp(value, "rectangular_cache_muon") == 0) {
        *mode = LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON;
        return true;
    }
    return false;
}

inline bool llmc_parse_normuon_execution_mode(
    const char* value,
    LlmcNormuonExecutionMode* mode) {
    if (value == nullptr || mode == nullptr) {
        return false;
    }
    if (strcmp(value, "fp32_reference") == 0) {
        *mode = LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
        return true;
    }
    if (strcmp(value, "bf16_batched") == 0) {
        *mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
        return true;
    }
    return false;
}

inline bool llmc_parse_normuon_tracker_retraction_mode(
    const char* value,
    LlmcNormuonTrackerRetractionMode* mode) {
    if (value == nullptr || mode == nullptr) {
        return false;
    }
    if (strcmp(value, "0") == 0 || strcmp(value, "disabled") == 0) {
        *mode = LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
        return true;
    }
    if (strcmp(value, "1") == 0 || strcmp(value, "newton_schulz") == 0) {
        *mode = LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ;
        return true;
    }
    if (strcmp(value, "commuted_canonical_stage2") == 0) {
        *mode =
            LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
        return true;
    }
    if (strcmp(value, "thin_canonical_stage2") == 0 ||
        strcmp(value, "thin_commuted_canonical_stage2") == 0) {
        *mode = LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2;
        return true;
    }
    return false;
}

inline bool llmc_parse_normuon_tracker_correction_mode(
    const char* value,
    LlmcNormuonTrackerCorrectionMode* mode) {
    if (value == nullptr || mode == nullptr) {
        return false;
    }
    if (strcmp(value, "0") == 0 || strcmp(value, "global_frobenius") == 0) {
        *mode = LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS;
        return true;
    }
    if (strcmp(value, "1") == 0 || strcmp(value, "diagonal_sylvester") == 0) {
        *mode = LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER;
        return true;
    }
    return false;
}

inline bool llmc_parse_normuon_approximation_policy(
    const char* value,
    LlmcNormuonApproximationPolicy* policy) {
    if (value == nullptr || policy == nullptr) {
        return false;
    }
    if (strcmp(value, "canonical_taylor_quintic") == 0) {
        *policy = LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
        return true;
    }
    if (strcmp(value, "stock_normuon_quintic") == 0) {
        *policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
        return true;
    }
    if (strcmp(value, "polar_express") == 0) {
        *policy = LLMC_NORMUON_APPROX_POLAR_EXPRESS;
        return true;
    }
    if (strcmp(value, "cache_muon_gram_gns") == 0) {
        *policy = LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS;
        return true;
    }
    return false;
}

inline bool llmc_parse_normuon_targeted_families(const char* value, uint32_t* mask) {
    if (value == nullptr || mask == nullptr) {
        return false;
    }
    std::string text(value);
    uint32_t parsed = 0U;
    size_t start = 0U;
    while (start <= text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        std::string token = text.substr(start, end - start);
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
            token.erase(token.begin());
        }
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
            token.pop_back();
        }
        if (token == "mlp_wup") {
            parsed |= llmc_optimizer_family_mask(LLMC_OPTIMIZER_FAMILY_MLP_WUP);
        } else if (token == "mlp_wdown") {
            parsed |= llmc_optimizer_family_mask(LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
        } else {
            return false;
        }
        if (end == text.size()) {
            break;
        }
        start = end + 1U;
    }
    *mask = parsed;
    return true;
}

inline const LlmcNormuonPolynomialStep* llmc_normuon_policy_schedule(
    LlmcNormuonApproximationPolicy policy) {
    switch (policy) {
        case LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC:
            return kLlmcCanonicalTaylorQuintic;
        case LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC:
            return kLlmcStockNormuonQuintic;
        case LLMC_NORMUON_APPROX_POLAR_EXPRESS:
            return kLlmcPolarExpress;
        case LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS:
            return kLlmcCacheMuonGramGns;
        default:
            return nullptr;
    }
}

inline void llmc_normuon_resolve_schedules(LlmcNormuonConfig* config) {
    if (llmc_normuon_is_cache_mode(config->orthogonalization_mode)) {
        // CacheMuon is a self-contained mode.  Its refresh solver is not a
        // user-selectable Muon polynomial; pin and report the paper's
        // FreshGNS coefficients so direct CLI and Python launches agree.
        config->refresh_policy = LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS;
    }
    const LlmcNormuonPolynomialStep* refresh =
        llmc_normuon_policy_schedule(config->refresh_policy);
    const LlmcNormuonPolynomialStep* correction =
        llmc_normuon_policy_schedule(config->correction_policy);
    if (refresh == nullptr || correction == nullptr) {
        return;
    }
    memcpy(
        config->refresh_schedule,
        refresh,
        sizeof(config->refresh_schedule));
    memcpy(
        config->correction_schedule,
        correction,
        sizeof(config->correction_schedule));
}

inline void llmc_normuon_config_defaults(LlmcNormuonConfig* config) {
    memset(config, 0, sizeof(*config));
    config->optimizer_selection = LLMC_OPTIMIZER_SELECTION_ADAMW;
    config->targeted_family_mask = llmc_normuon_required_target_mask();
    config->learning_rate = 2.5e-3f;
    config->weight_decay = 1.0e-3f;
    config->momentum = 0.95f;
    config->beta2 = 0.95f;
    config->epsilon = 1.0e-8f;
    config->update_scale = 1.0f;
    config->wdown_learning_rate_multiplier = 1.0f;
    config->execution_mode = LLMC_NORMUON_EXECUTION_BF16_BATCHED;
    config->orthogonalization_mode = LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
    config->refresh_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config->correction_policy = LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config->refresh_interval = 3U;
    config->correction_iterations = 2U;
    config->correction_gain = 1.0f;
    config->cache_residual_threshold =
        LLMC_CACHEMUON_DEFAULT_RESIDUAL_THRESHOLD;
    config->correction_mode =
        LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS;
    config->retraction_mode = LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ;
    llmc_normuon_resolve_schedules(config);
}

inline bool llmc_normuon_validate_config(
    LlmcNormuonConfig* config,
    char* error,
    size_t error_capacity) {
    if (config == nullptr) {
        return false;
    }
    llmc_normuon_resolve_schedules(config);
    const bool finite_hyperparameters =
        isfinite(config->learning_rate) &&
        isfinite(config->weight_decay) &&
        isfinite(config->momentum) &&
        isfinite(config->beta2) &&
        isfinite(config->epsilon) &&
        isfinite(config->update_scale) &&
        isfinite(config->wdown_learning_rate_multiplier) &&
        isfinite(config->correction_gain) &&
        isfinite(config->cache_residual_threshold);
    bool valid =
        (config->optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW ||
         config->optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON) &&
        (config->execution_mode == LLMC_NORMUON_EXECUTION_FP32_REFERENCE ||
         config->execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED) &&
        (config->orthogonalization_mode == LLMC_NORMUON_ORTHO_NEWTON_SCHULZ ||
         config->orthogonalization_mode == LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q ||
         config->orthogonalization_mode == LLMC_NORMUON_ORTHO_RECTANGULAR_MUON ||
         config->orthogonalization_mode ==
             LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q ||
         config->orthogonalization_mode ==
             LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON ||
         config->orthogonalization_mode ==
             LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON) &&
        finite_hyperparameters &&
        config->learning_rate > 0.0f &&
        config->weight_decay >= 0.0f &&
        config->momentum >= 0.0f && config->momentum < 1.0f &&
        config->beta2 >= 0.0f && config->beta2 < 1.0f &&
        config->epsilon > 0.0f &&
        config->update_scale >= 0.0f &&
        config->wdown_learning_rate_multiplier > 0.0f &&
        config->correction_gain >= 0.0f &&
        config->cache_residual_threshold > 0.0f &&
        (config->correction_mode ==
             LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS ||
         config->correction_mode ==
             LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) &&
        config->refresh_interval > 0U &&
        config->correction_iterations > 0U &&
        config->correction_iterations <= LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT &&
        (config->retraction_mode ==
             LLMC_NORMUON_TRACKER_RETRACTION_DISABLED ||
         config->retraction_mode == LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ ||
         config->retraction_mode == LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2 ||
         config->retraction_mode == LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2) &&
        llmc_normuon_policy_schedule(config->refresh_policy) != nullptr &&
        llmc_normuon_policy_schedule(config->correction_policy) != nullptr &&
        (llmc_normuon_is_cache_mode(config->orthogonalization_mode) ||
         config->refresh_policy != LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS) &&
        config->correction_policy != LLMC_NORMUON_APPROX_CACHE_MUON_GRAM_GNS;
    if (config->optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON &&
        config->targeted_family_mask != llmc_normuon_required_target_mask()) {
        valid = false;
        if (error != nullptr && error_capacity != 0U) {
            snprintf(
                error,
                error_capacity,
                "NorMuon currently requires targeted families exactly "
                "mlp_wup,mlp_wdown");
        }
        return false;
    }
    if ((config->retraction_mode ==
             LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2 ||
         config->retraction_mode ==
             LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2) &&
        (!llmc_normuon_is_tracker_mode(config->orthogonalization_mode) ||
         config->correction_policy !=
             LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC ||
         config->correction_iterations != 2U)) {
        if (error != nullptr && error_capacity != 0U) {
            snprintf(
                error,
                error_capacity,
                "canonical stage2 retraction requires a tracker mode, "
                "canonical_taylor_quintic, and exactly two correction stages");
        }
        return false;
    }
    if (config->retraction_mode ==
            LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2 &&
        config->execution_mode != LLMC_NORMUON_EXECUTION_BF16_BATCHED) {
        if (error != nullptr && error_capacity != 0U) {
            snprintf(
                error,
                error_capacity,
                "thin_canonical_stage2 requires bf16_batched execution");
        }
        return false;
    }
    if (llmc_normuon_is_split_family_mode(config->orthogonalization_mode) &&
        config->execution_mode != LLMC_NORMUON_EXECUTION_BF16_BATCHED) {
        if (error != nullptr && error_capacity != 0U) {
            snprintf(error, error_capacity,
                     "split Wup/Wdown mode requires bf16_batched execution");
        }
        return false;
    }
    if (config->orthogonalization_mode ==
            LLMC_NORMUON_ORTHO_RECTANGULAR_MUON &&
        config->retraction_mode != LLMC_NORMUON_TRACKER_RETRACTION_DISABLED) {
        if (error != nullptr && error_capacity != 0U) {
            snprintf(
                error,
                error_capacity,
                "rectangular_muon is scratch-only and cannot enable tracker retraction");
        }
        return false;
    }
    if (llmc_normuon_is_cache_mode(config->orthogonalization_mode) &&
        config->retraction_mode != LLMC_NORMUON_TRACKER_RETRACTION_DISABLED) {
        if (error != nullptr && error_capacity != 0U) {
            snprintf(
                error,
                error_capacity,
                "rectangular_cache_muon owns its FreshGNS residual gate and "
                "cannot enable tracker correction/retraction");
        }
        return false;
    }
    if (!valid && error != nullptr && error_capacity != 0U) {
        snprintf(error, error_capacity, "invalid llm.c NorMuon configuration");
    }
    return valid;
}

inline bool llmc_normuon_config_equal(
    const LlmcNormuonConfig* lhs,
    const LlmcNormuonConfig* rhs) {
    return lhs != nullptr && rhs != nullptr &&
           memcmp(lhs, rhs, sizeof(*lhs)) == 0;
}

inline bool llmc_enumerate_mlp_wup_view(
    const LlmcOptimizerParameterType* parameter_type,
    int view_index,
    LlmcOptimizerMatrixView* view) {
    if (parameter_type == nullptr || view == nullptr ||
        view_index < 0 || view_index >= LLMC_NORMUON_VIEWS_PER_MLP_MATRIX ||
        parameter_type->matrix_width == 0U) {
        return false;
    }
    const size_t width = parameter_type->matrix_width;
    view->element_offset = static_cast<size_t>(view_index) * width * width;
    view->second_moment_offset = view->element_offset;
    view->rows = width;
    view->columns = width;
    view->row_stride = width;
    view->column_stride = 1U;
    return true;
}

inline bool llmc_enumerate_mlp_wdown_view(
    const LlmcOptimizerParameterType* parameter_type,
    int view_index,
    LlmcOptimizerMatrixView* view) {
    if (parameter_type == nullptr || view == nullptr ||
        view_index < 0 || view_index >= LLMC_NORMUON_VIEWS_PER_MLP_MATRIX ||
        parameter_type->matrix_width == 0U) {
        return false;
    }
    const size_t width = parameter_type->matrix_width;
    view->element_offset = static_cast<size_t>(view_index) * width;
    view->second_moment_offset = view->element_offset;
    view->rows = width;
    view->columns = width;
    view->row_stride = 4U * width;
    view->column_stride = 1U;
    return true;
}

inline bool llmc_enumerate_mlp_wup_rectangular_view(
    const LlmcOptimizerParameterType* parameter_type,
    int view_index,
    LlmcOptimizerMatrixView* view) {
    if (parameter_type == nullptr || view == nullptr || view_index != 0 ||
        parameter_type->matrix_width == 0U) {
        return false;
    }
    const size_t width = parameter_type->matrix_width;
    view->element_offset = 0U;
    view->second_moment_offset = 0U;
    view->rows = 4U * width;
    view->columns = width;
    view->row_stride = width;
    view->column_stride = 1U;
    return true;
}

inline bool llmc_enumerate_mlp_wdown_rectangular_view(
    const LlmcOptimizerParameterType* parameter_type,
    int view_index,
    LlmcOptimizerMatrixView* view) {
    if (parameter_type == nullptr || view == nullptr || view_index != 0 ||
        parameter_type->matrix_width == 0U) {
        return false;
    }
    const size_t width = parameter_type->matrix_width;
    view->element_offset = 0U;
    view->second_moment_offset = 0U;
    view->rows = width;
    view->columns = 4U * width;
    view->row_stride = 4U * width;
    view->column_stride = 1U;
    return true;
}

inline bool llmc_optimizer_view_within_bounds(
    const LlmcOptimizerParameterType* parameter_type,
    const LlmcOptimizerMatrixView* view) {
    if (parameter_type == nullptr || view == nullptr ||
        view->rows == 0U || view->columns == 0U) {
        return false;
    }
    const size_t last =
        view->element_offset +
        (view->rows - 1U) * view->row_stride +
        (view->columns - 1U) * view->column_stride;
    const size_t axis_length =
        view->rows > view->columns ? view->rows : view->columns;
    return last < parameter_type->layer_elements &&
           view->second_moment_offset + axis_length <=
               parameter_type->layer_elements;
}

struct LlmcOptimizerParameterTemplate {
    const char* name;
    LlmcOptimizerFamilyId family_id;
    LlmcWeightDecayPolicy decay_policy;
    int layer_multiplicity;
    int views_per_layer;
    LlmcOptimizerMatrixViewEnumerator enumerator;
};

static constexpr LlmcOptimizerParameterTemplate kLlmcParameterTemplates[
    LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT] = {
    {"wte", LLMC_OPTIMIZER_FAMILY_EMBEDDINGS, LLMC_WEIGHT_DECAY_ENABLED, 0, 0, nullptr},
    {"wpe", LLMC_OPTIMIZER_FAMILY_EMBEDDINGS, LLMC_WEIGHT_DECAY_ENABLED, 0, 0, nullptr},
    {"ln1w", LLMC_OPTIMIZER_FAMILY_NORMALIZATION, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"ln1b", LLMC_OPTIMIZER_FAMILY_NORMALIZATION, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"qkvw", LLMC_OPTIMIZER_FAMILY_ATTENTION, LLMC_WEIGHT_DECAY_ENABLED, 1, 0, nullptr},
    {"qkvb", LLMC_OPTIMIZER_FAMILY_ATTENTION, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"attprojw", LLMC_OPTIMIZER_FAMILY_ATTENTION, LLMC_WEIGHT_DECAY_ENABLED, 1, 0, nullptr},
    {"attprojb", LLMC_OPTIMIZER_FAMILY_ATTENTION, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"ln2w", LLMC_OPTIMIZER_FAMILY_NORMALIZATION, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"ln2b", LLMC_OPTIMIZER_FAMILY_NORMALIZATION, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"fcw", LLMC_OPTIMIZER_FAMILY_MLP_WUP, LLMC_WEIGHT_DECAY_ENABLED, 1, 4, llmc_enumerate_mlp_wup_view},
    {"fcb", LLMC_OPTIMIZER_FAMILY_MLP_BIASES, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"fcprojw", LLMC_OPTIMIZER_FAMILY_MLP_WDOWN, LLMC_WEIGHT_DECAY_ENABLED, 1, 4, llmc_enumerate_mlp_wdown_view},
    {"fcprojb", LLMC_OPTIMIZER_FAMILY_MLP_BIASES, LLMC_WEIGHT_DECAY_DISABLED, 1, 0, nullptr},
    {"lnfw", LLMC_OPTIMIZER_FAMILY_NORMALIZATION, LLMC_WEIGHT_DECAY_DISABLED, 0, 0, nullptr},
    {"lnfb", LLMC_OPTIMIZER_FAMILY_NORMALIZATION, LLMC_WEIGHT_DECAY_DISABLED, 0, 0, nullptr},
};

inline void llmc_optimizer_plan_reset(LlmcOptimizerPlan* plan) {
    memset(plan, 0, sizeof(*plan));
}

inline bool llmc_build_optimizer_plan(
    LlmcOptimizerPlan* plan,
    const LlmcNormuonConfig* config,
    int num_layers,
    int channels,
    const size_t parameter_elements[LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT],
    char* error,
    size_t error_capacity) {
    if (plan == nullptr || config == nullptr || parameter_elements == nullptr ||
        num_layers <= 0 || channels <= 0) {
        return false;
    }
    LlmcNormuonConfig resolved = *config;
    if (!llmc_normuon_validate_config(&resolved, error, error_capacity)) {
        return false;
    }
    llmc_optimizer_plan_reset(plan);
    plan->num_layers = num_layers;
    plan->channels = channels;
    plan->family_count = LLMC_OPTIMIZER_FAMILY_COUNT;
    plan->backends[LLMC_OPTIMIZER_BACKEND_ADAMW] = {
        LLMC_OPTIMIZER_BACKEND_ADAMW, "adamw"};
    plan->backends[LLMC_OPTIMIZER_BACKEND_NORMUON] = {
        LLMC_OPTIMIZER_BACKEND_NORMUON, "normuon"};

    static constexpr const char* family_names[LLMC_OPTIMIZER_FAMILY_COUNT] = {
        "embeddings",
        "normalization",
        "attention",
        "mlp_wup",
        "mlp_wdown",
        "mlp_biases",
    };
    for (int family_index = 0; family_index < LLMC_OPTIMIZER_FAMILY_COUNT; ++family_index) {
        plan->families[family_index].family_id =
            static_cast<LlmcOptimizerFamilyId>(family_index);
        plan->families[family_index].name = family_names[family_index];
    }

    for (int tensor_id = 0; tensor_id < LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT; ++tensor_id) {
        const LlmcOptimizerParameterTemplate& source = kLlmcParameterTemplates[tensor_id];
        LlmcOptimizerParameterType& target = plan->parameter_types[tensor_id];
        target.tensor_id = tensor_id;
        target.name = source.name;
        target.family_id = source.family_id;
        target.weight_decay_policy = source.decay_policy;
        target.layer_multiplicity = source.layer_multiplicity ? num_layers : 1;
        target.tensor_elements = parameter_elements[tensor_id];
        if (target.tensor_elements % static_cast<size_t>(target.layer_multiplicity) != 0U) {
            if (error != nullptr && error_capacity != 0U) {
                snprintf(error, error_capacity, "tensor %d has invalid layer multiplicity", tensor_id);
            }
            return false;
        }
        target.layer_elements =
            target.tensor_elements / static_cast<size_t>(target.layer_multiplicity);
        target.matrix_width = static_cast<size_t>(channels);
        target.views_per_layer = source.views_per_layer;
        target.enumerate_matrix_view = source.enumerator;
        const bool family_rectangular = llmc_normuon_family_is_rectangular(
            resolved.orthogonalization_mode, source.family_id);
        if (family_rectangular &&
            source.family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP) {
            target.views_per_layer = LLMC_NORMUON_RECTANGULAR_VIEWS_PER_MLP_MATRIX;
            target.enumerate_matrix_view = llmc_enumerate_mlp_wup_rectangular_view;
        } else if (family_rectangular &&
                   source.family_id == LLMC_OPTIMIZER_FAMILY_MLP_WDOWN) {
            target.views_per_layer = LLMC_NORMUON_RECTANGULAR_VIEWS_PER_MLP_MATRIX;
            target.enumerate_matrix_view = llmc_enumerate_mlp_wdown_rectangular_view;
        }
        const bool targeted =
            (resolved.targeted_family_mask &
             llmc_optimizer_family_mask(source.family_id)) != 0U;
        target.backend_kind =
            resolved.optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON &&
            targeted
                ? LLMC_OPTIMIZER_BACKEND_NORMUON
                : LLMC_OPTIMIZER_BACKEND_ADAMW;
        target.hyperparameter_group =
            target.backend_kind == LLMC_OPTIMIZER_BACKEND_NORMUON
                ? LLMC_OPTIMIZER_HYPERPARAM_NORMUON_MLP
                : LLMC_OPTIMIZER_HYPERPARAM_ADAMW_DEFAULT;
        if (target.backend_kind == LLMC_OPTIMIZER_BACKEND_NORMUON) {
            const int expected_views = llmc_normuon_family_views_per_matrix(
                resolved.orthogonalization_mode, source.family_id);
            if (target.enumerate_matrix_view == nullptr ||
                target.views_per_layer != expected_views) {
                if (error != nullptr && error_capacity != 0U) {
                    snprintf(error, error_capacity, "NorMuon tensor %d has no valid matrix-view enumerator", tensor_id);
                }
                return false;
            }
            plan->normuon_parameter_type_count++;
            plan->normuon_view_count +=
                target.layer_multiplicity * target.views_per_layer;
            for (int view_index = 0; view_index < target.views_per_layer; ++view_index) {
                LlmcOptimizerMatrixView view;
                if (!target.enumerate_matrix_view(&target, view_index, &view) ||
                    !llmc_optimizer_view_within_bounds(&target, &view)) {
                    if (error != nullptr && error_capacity != 0U) {
                        snprintf(error, error_capacity, "NorMuon tensor %d view %d is out of bounds", tensor_id, view_index);
                    }
                    return false;
                }
            }
        }
        LlmcOptimizerFamily& family = plan->families[source.family_id];
        family.parameter_type_indices[family.parameter_type_count++] = tensor_id;
    }

    if (resolved.optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON &&
        (plan->parameter_types[10].backend_kind != LLMC_OPTIMIZER_BACKEND_NORMUON ||
         plan->parameter_types[12].backend_kind != LLMC_OPTIMIZER_BACKEND_NORMUON ||
         plan->normuon_parameter_type_count != 2 ||
        plan->normuon_view_count !=
             num_layers *
            (llmc_normuon_is_split_family_mode(resolved.orthogonalization_mode)
                 ? 5
                 : (llmc_normuon_is_rectangular_mode(resolved.orthogonalization_mode)
                        ? LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER
                        : LLMC_NORMUON_VIEWS_PER_LAYER)))) {
        if (error != nullptr && error_capacity != 0U) {
            snprintf(error, error_capacity, "only fcw/fcprojw may route to NorMuon");
        }
        return false;
    }
    plan->built = true;
    return true;
}

inline size_t llmc_normuon_q_view_index(
    const LlmcOptimizerParameterType* parameter_type,
    int layer_index,
    int view_index,
    bool rectangular = false) {
    if (rectangular) {
        const size_t family_view_offset =
            parameter_type->family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 1U;
        return static_cast<size_t>(layer_index) *
                   LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER +
               family_view_offset + static_cast<size_t>(view_index);
    }
    const size_t family_view_offset =
        parameter_type->family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 4U;
    return static_cast<size_t>(layer_index) * LLMC_NORMUON_VIEWS_PER_LAYER +
           family_view_offset + static_cast<size_t>(view_index);
}

inline uint32_t llmc_normuon_grid_for_count(size_t count) {
    size_t blocks = (count + LLMC_NORMUON_BLOCK_SIZE - 1U) / LLMC_NORMUON_BLOCK_SIZE;
    if (blocks == 0U) {
        blocks = 1U;
    }
    if (blocks > 65535U) {
        blocks = 65535U;
    }
    return static_cast<uint32_t>(blocks);
}

__device__ __forceinline__ size_t llmc_normuon_strided_offset(
    size_t index,
    size_t columns,
    size_t row_stride,
    size_t column_stride) {
    const size_t row = index / columns;
    const size_t column = index - row * columns;
    return row * row_stride + column * column_stride;
}

__device__ __forceinline__ void llmc_normuon_mark_nonfinite(int* flag) {
    atomicExch(flag, 1);
}

__global__ void llmc_normuon_prepare_momentum_kernel(
    const floatX* gradient,
    float* momentum,
    float* staged,
    float* partials,
    int* nonfinite,
    size_t rows,
    size_t columns,
    size_t row_stride,
    size_t column_stride,
    float momentum_beta,
    float gradient_scale) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    const size_t element_count = rows * columns;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    float sum = 0.0f;
    for (; index < element_count; index += stride) {
        const size_t offset =
            llmc_normuon_strided_offset(index, columns, row_stride, column_stride);
        const float grad = gradient_scale * static_cast<float>(gradient[offset]);
        const float next_momentum =
            momentum_beta * momentum[offset] + (1.0f - momentum_beta) * grad;
        const float nesterov =
            (1.0f - momentum_beta) * grad + momentum_beta * next_momentum;
        if (!isfinite(grad) || !isfinite(next_momentum) || !isfinite(nesterov)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        momentum[offset] = next_momentum;
        staged[index] = nesterov;
        sum += nesterov * nesterov;
    }
    local[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        partials[blockIdx.x] = local[0];
    }
}

__global__ void llmc_normuon_reduce_kernel(
    const float* values,
    float* output,
    size_t count) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    float sum = 0.0f;
    for (size_t index = threadIdx.x; index < count; index += blockDim.x) {
        sum += values[index];
    }
    local[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        *output = local[0];
    }
}

__global__ void llmc_normuon_normalize_kernel(
    float* matrix,
    const float* norm_squared,
    int* nonfinite,
    size_t element_count,
    float norm_multiplier,
    float epsilon) {
    const float norm = sqrtf(fmaxf(*norm_squared, 0.0f));
    const float denominator = norm_multiplier * norm + epsilon;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const float value = denominator > 0.0f ? matrix[index] / denominator : 0.0f;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        matrix[index] = value;
    }
}

__global__ void llmc_normuon_scale_copy_kernel(
    const float* source,
    float* destination,
    size_t count,
    float scale) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        destination[index] = scale * source[index];
    }
}

__global__ void llmc_normuon_linear_combination_kernel(
    float* destination,
    const float* source,
    size_t count,
    float destination_scale,
    float source_scale) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        destination[index] =
            destination_scale * destination[index] + source_scale * source[index];
    }
}

__global__ void llmc_normuon_add_projected_kernel(
    float* matrix,
    const float* projected,
    int* nonfinite,
    size_t count,
    float coefficient_a) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        const float value = coefficient_a * matrix[index] + projected[index];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        matrix[index] = value;
    }
}

__global__ void llmc_normuon_sym_skew_kernel(
    const float* matrix,
    float* skew,
    float* partials,
    int* nonfinite,
    size_t width,
    bool compute_symmetric_stats) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    const size_t element_count = width * width;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    float sum = 0.0f;
    for (; index < element_count; index += stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float transpose = matrix[column * width + row];
        const float symmetric = 0.5f * (matrix[index] + transpose);
        const float skew_value = 0.5f * (matrix[index] - transpose);
        if (!isfinite(symmetric) || !isfinite(skew_value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        skew[index] = skew_value;
        if (compute_symmetric_stats) {
            sum += symmetric * symmetric;
        }
    }
    if (!compute_symmetric_stats) {
        return;
    }
    local[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        partials[blockIdx.x] = local[0];
    }
}

__global__ void llmc_normuon_build_correction_kernel(
    const float* phase,
    const float* skew,
    float* correction,
    const float* symmetric_norm_squared,
    int* nonfinite,
    size_t width,
    float gain,
    float epsilon,
    int correction_mode) {
    const float denominator = sqrtf(fmaxf(*symmetric_norm_squared, 0.0f)) + epsilon;
    const size_t element_count = width * width;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        float correction_denominator = denominator;
        float correction_numerator = skew[index];
        if (correction_mode ==
            LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
            const float pair_stiffness =
                phase[row * width + row] + phase[column * width + column];
            // H is expected to be positive semidefinite in the tracker basin.
            // Clamp nonpositive or near-singular pair sums to keep the
            // regularized Jacobi approximation finite when Q is stale.
            correction_denominator = fmaxf(pair_stiffness, epsilon);
            correction_numerator = 2.0f * skew[index];
        }
        const float value =
            (row == column ? 1.0f : 0.0f) +
            gain * correction_numerator / correction_denominator;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        correction[index] = value;
    }
}

__global__ void llmc_cachemuon_identity_kernel(
    float* matrices,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width) {
    const size_t total = matrix_count * matrix_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / width;
        const size_t column = index - row * width;
        matrices[matrix_index * panel_stride + index] =
            row == column ? 1.0f : 0.0f;
    }
}

__global__ void llmc_cachemuon_form_polynomial_kernel(
    const float* gram,
    const float* gram_squared,
    float* polynomial,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    LlmcNormuonPolynomialStep coefficient) {
    const size_t total = matrix_count * matrix_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / width;
        const size_t column = index - row * width;
        const size_t offset = matrix_index * panel_stride + index;
        const float value = coefficient.b * gram[offset] +
                            coefficient.c * gram_squared[offset] +
                            (row == column ? coefficient.a : 0.0f);
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        polynomial[offset] = value;
    }
}

__global__ void llmc_cachemuon_residual_kernel(
    const float* gram,
    float* residuals,
    size_t matrix_count,
    size_t panel_stride,
    size_t width) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t matrix_elements = width * width;
    const float* matrix = gram + matrix_index * panel_stride;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    float sum = 0.0f;
    bool finite = true;
    for (size_t index = threadIdx.x; index < matrix_elements;
         index += blockDim.x) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float value = matrix[index] - (row == column ? 1.0f : 0.0f);
        finite = finite && isfinite(value);
        sum += value * value;
    }
    local[threadIdx.x] = finite ? sum : INFINITY;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        residuals[matrix_index] =
            sqrtf(fmaxf(local[0], 0.0f) / static_cast<float>(width));
    }
}

__global__ void llmc_normuon_reduce_squares_kernel(
    const float* values,
    float* output,
    size_t count) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    float sum = 0.0f;
    for (size_t index = threadIdx.x; index < count; index += blockDim.x) {
        const float value = values[index];
        sum += value * value;
    }
    local[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        *output = local[0];
    }
}

__global__ void llmc_normuon_build_damped_diagonal_correction_kernel(
    const float* phase,
    float* correction,
    const float* symmetric_norm_squared,
    float* correction_norm_squared,
    int* nonfinite,
    size_t width,
    float gain,
    float epsilon) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    const size_t element_count = width * width;
    const float symmetric_scale =
        sqrtf(fmaxf(*symmetric_norm_squared, 0.0f)) /
        sqrtf(static_cast<float>(width));
    const float diagonal_floor =
        fmaxf(LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale, epsilon);
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    float sum = 0.0f;
    for (; index < element_count; index += stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float transpose = phase[column * width + row];
        const float skew = 0.5f * (phase[index] - transpose);
        const float row_stiffness =
            fmaxf(phase[row * width + row], diagonal_floor);
        const float column_stiffness =
            fmaxf(phase[column * width + column], diagonal_floor);
        const float denominator = row_stiffness + column_stiffness;
        const float omega = gain * 2.0f * skew / denominator;
        if (!isfinite(omega)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        correction[index] = omega;
        sum += omega * omega;
    }
    local[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        *correction_norm_squared = local[0];
        if (!isfinite(local[0])) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
}

// Estimate ||Omega||_2 with a short FP32 power iteration.  The correction panel
// is small relative to the GEMMs that surround it, so a matrix-vector pass is
// materially cheaper than adding another matrix-matrix product.  The scratch
// panel stores x and Omega*x in its first two rows-worth of elements.
__global__ void llmc_normuon_power_iteration_kernel(
    const float* matrix,
    float* scratch,
    float* spectral_norm,
    int* nonfinite,
    size_t width,
    uint32_t iterations) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    if (width == 0U) {
        if (threadIdx.x == 0U) {
            *spectral_norm = 0.0f;
        }
        return;
    }
    if (width == 1U) {
        if (threadIdx.x == 0U) {
            *spectral_norm = 0.0f;
        }
        return;
    }
    float* x = scratch;
    float* y = scratch + width;
    const float initial_scale = rsqrtf(static_cast<float>(width));
    for (size_t index = threadIdx.x; index < width; index += blockDim.x) {
        const uint32_t hash =
            static_cast<uint32_t>(index) * 0x9e3779b9U + 0x7f4a7c15U;
        x[index] = (hash & 1U) != 0U ? initial_scale : -initial_scale;
    }
    __syncthreads();

    float estimate = 0.0f;
    for (uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        float y_sum = 0.0f;
        for (size_t row = threadIdx.x; row < width; row += blockDim.x) {
            float value = 0.0f;
            for (size_t column = 0U; column < width; ++column) {
                value += matrix[row * width + column] * x[column];
            }
            y[row] = value;
            y_sum += value * value;
        }
        local[threadIdx.x] = y_sum;
        __syncthreads();
        for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
            if (threadIdx.x < offset) {
                local[threadIdx.x] += local[threadIdx.x + offset];
            }
            __syncthreads();
        }
        estimate = sqrtf(fmaxf(local[0], 0.0f));

        float z_sum = 0.0f;
        for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
            float value = 0.0f;
            for (size_t row = 0U; row < width; ++row) {
                value += matrix[row * width + column] * y[row];
            }
            x[column] = value;
            z_sum += value * value;
        }
        local[threadIdx.x] = z_sum;
        __syncthreads();
        for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
            if (threadIdx.x < offset) {
                local[threadIdx.x] += local[threadIdx.x + offset];
            }
            __syncthreads();
        }
        const float denominator = sqrtf(fmaxf(local[0], 1.0e-20f));
        for (size_t index = threadIdx.x; index < width; index += blockDim.x) {
            x[index] /= denominator;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        *spectral_norm = estimate;
        if (!isfinite(estimate)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
}

__global__ void llmc_normuon_finalize_damped_diagonal_correction_kernel(
    float* correction,
    const float* correction_norm_squared,
    const float* spectral_norm,
    int* nonfinite,
    size_t width,
    float epsilon) {
    const float raw_norm =
        sqrtf(fmaxf(*correction_norm_squared, 0.0f));
    const float target_norm =
        LLMC_NORMUON_TRACKER_CORRECTION_CAP * sqrtf(static_cast<float>(width));
    const float frobenius_scale =
        fminf(1.0f, target_norm / (raw_norm + epsilon));
    const float spectral_limit = sqrtf(fmaxf(
        LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX *
                LLMC_NORMUON_TRACKER_SPECTRAL_RHO_MAX -
            1.0f,
        0.0f));
    const float spectral_scale = fminf(
        1.0f, spectral_limit / (fmaxf(*spectral_norm, 0.0f) + epsilon));
    const float scale = fminf(frobenius_scale, spectral_scale);
    const size_t element_count = width * width;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float value =
            (row == column ? 1.0f : 0.0f) + scale * correction[index];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        correction[index] = value;
    }
}

// Convert the already-polished relative factor C = I + Omega into the
// one-sided tangent factor B used by rectangular tracking.  With H replaced
// by its damped diagonal, the candidate is
//   Y = D H^{-1} + Q B,  B = C - S H^{-1},  S = Q^T D (or D Q^T).
// Keeping this as one elementwise pass avoids materializing the horizontal
// residual D - Q S and saves a large GEMM plus a launch on every stale step.
__global__ void llmc_normuon_build_rectangular_tangent_factor_kernel(
    const float* phase,
    float* factor,
    const float* symmetric_norm_squared,
    int* nonfinite,
    size_t width,
    bool tall,
    float epsilon) {
    const float symmetric_scale =
        sqrtf(fmaxf(*symmetric_norm_squared, 0.0f)) /
        sqrtf(static_cast<float>(width));
    const float diagonal_floor =
        fmaxf(LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale, epsilon);
    const size_t element_count = width * width;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float row_stiffness =
            fmaxf(phase[row * width + row], diagonal_floor);
        const float column_stiffness =
            fmaxf(phase[column * width + column], diagonal_floor);
        const float denominator = tall ? column_stiffness : row_stiffness;
        const float value = factor[index] - phase[index] / denominator;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        factor[index] = value;
    }
}

// Add the horizontal component D H^{-1} to the large-side product Q B.  The
// axis choice follows the matrix orientation: columns for tall Wup and rows
// for wide Wdown.  This is the second half of the fused candidate formation.
__global__ void llmc_normuon_add_rectangular_horizontal_kernel(
    float* candidate,
    const float* normalized,
    const float* phase,
    const float* symmetric_norm_squared,
    int* nonfinite,
    size_t rows,
    size_t columns,
    bool tall,
    float epsilon) {
    const size_t side = rows < columns ? rows : columns;
    const float symmetric_scale =
        sqrtf(fmaxf(*symmetric_norm_squared, 0.0f)) /
        sqrtf(static_cast<float>(side));
    const float diagonal_floor =
        fmaxf(LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale, epsilon);
    const size_t element_count = rows * columns;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / columns;
        const size_t column = index - row * columns;
        const size_t axis = tall ? column : row;
        const float stiffness =
            fmaxf(phase[axis * side + axis], diagonal_floor);
        const float value = candidate[index] + normalized[index] / stiffness;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        candidate[index] = value;
    }
}

__global__ void llmc_normuon_three_minus_kernel(float* matrix, size_t width) {
    const size_t element_count = width * width;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        matrix[index] = (row == column ? 3.0f : 0.0f) - matrix[index];
    }
}

__global__ void llmc_normuon_second_moment_kernel(
    const float* direction,
    float* second_moment,
    float* axis_stats,
    int* nonfinite,
    size_t rows,
    size_t columns,
    float beta2,
    float epsilon) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    // Canonical NorMuon uses row-wise statistics for both tall and wide
    // matrices; wide Wdown must not silently become column-normalized.
    const size_t axis_length = rows;
    const size_t reduced_length = columns;
    const size_t axis = blockIdx.x;
    if (axis >= axis_length) {
        return;
    }
    float sum = 0.0f;
    for (size_t reduced = threadIdx.x;
         reduced < reduced_length;
         reduced += blockDim.x) {
        const size_t index = axis * columns + reduced;
        const float value = direction[index];
        sum += value * value;
    }
    local[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const float mean = local[0] / static_cast<float>(reduced_length);
        const float next =
            beta2 * second_moment[axis] + (1.0f - beta2) * mean;
        const float contribution = local[0] / fmaxf(next, epsilon);
        if (!isfinite(next) || !isfinite(contribution)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        second_moment[axis] = next;
        axis_stats[axis] = contribution;
    }
}

__device__ __forceinline__ uint32_t llmc_normuon_rounding_seed(
    uint64_t global_step,
    int tensor_id,
    int layer_index,
    int view_index) {
    uint64_t value = global_step + 0x9e3779b97f4a7c15ULL;
    value ^= static_cast<uint64_t>(tensor_id + 0x100) * 0xbf58476d1ce4e5b9ULL;
    value ^= static_cast<uint64_t>(layer_index + 0x200) * 0x94d049bb133111ebULL;
    value ^= static_cast<uint64_t>(view_index + 0x300) * 0xd6e8feb86659fd93ULL;
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return static_cast<uint32_t>(value ^ (value >> 32U));
}

__device__ __forceinline__ void llmc_normuon_stochastic_round(
    float value,
    floatX* output,
    uint32_t seed,
    size_t element_index) {
#if defined(ENABLE_BF16)
    const uint32_t random = SquirrelNoise5(static_cast<uint32_t>(element_index), seed);
    const uint32_t threshold = random & 0xFFFFU;
    uint32_t bits = __float_as_uint(value);
    const uint32_t discarded = bits & 0xFFFFU;
    bits = discarded > threshold ? (bits | 0xFFFFU) : (bits & ~0xFFFFU);
    *output = __float2bfloat16_rn(__uint_as_float(bits));
#elif defined(ENABLE_FP16)
    *output = __float2half_rn(value);
#else
    *output = value;
#endif
}

__global__ void llmc_normuon_validate_update_kernel(
    const float* master,
    const float* direction,
    const float* second_moment,
    const float* normalized_update_norm_squared,
    const float* direction_norm_squared,
    int* nonfinite,
    size_t rows,
    size_t columns,
    size_t row_stride,
    size_t column_stride,
    float learning_rate,
    float weight_decay,
    float epsilon,
    float update_scale,
    float learning_rate_multiplier) {
    const size_t element_count = rows * columns;
    const float normalized_norm =
        sqrtf(fmaxf(*normalized_update_norm_squared, epsilon));
    const float direction_norm =
        sqrtf(fmaxf(*direction_norm_squared, epsilon));
    const float global_scale = direction_norm / normalized_norm;
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float shape_scale = sqrtf(fmaxf(
        1.0f, static_cast<float>(rows) / static_cast<float>(columns)));
    const float update_lr = learning_rate * update_scale *
                            learning_rate_multiplier * shape_scale;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / columns;
        const size_t offset =
            llmc_normuon_strided_offset(index, columns, row_stride, column_stride);
        const size_t axis = row;
        const float local_scale =
            rsqrtf(fmaxf(second_moment[axis], epsilon)) * global_scale;
        const float updated =
            master[offset] * decay_scale -
            update_lr * direction[index] * local_scale;
        if (!isfinite(master[offset]) || !isfinite(local_scale) ||
            !isfinite(direction[index]) || !isfinite(updated)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
}

__global__ void llmc_normuon_apply_update_kernel(
    floatX* parameter,
    float* master,
    const float* direction,
    const float* second_moment,
    const float* normalized_update_norm_squared,
    const float* direction_norm_squared,
    size_t rows,
    size_t columns,
    size_t row_stride,
    size_t column_stride,
    float learning_rate,
    float weight_decay,
    float epsilon,
    float update_scale,
    float learning_rate_multiplier,
    uint64_t global_step,
    int tensor_id,
    int layer_index,
    int view_index) {
    const size_t element_count = rows * columns;
    const float normalized_norm =
        sqrtf(fmaxf(*normalized_update_norm_squared, epsilon));
    const float direction_norm =
        sqrtf(fmaxf(*direction_norm_squared, epsilon));
    const float global_scale = direction_norm / normalized_norm;
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float shape_scale = sqrtf(fmaxf(
        1.0f, static_cast<float>(rows) / static_cast<float>(columns)));
    const float update_lr = learning_rate * update_scale *
                            learning_rate_multiplier * shape_scale;
    const uint32_t seed =
        llmc_normuon_rounding_seed(global_step, tensor_id, layer_index, view_index);
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / columns;
        const size_t offset =
            llmc_normuon_strided_offset(index, columns, row_stride, column_stride);
        const size_t axis = row;
        const float local_scale =
            rsqrtf(fmaxf(second_moment[axis], epsilon)) * global_scale;
        const float updated =
            master[offset] * decay_scale -
            update_lr * direction[index] * local_scale;
        master[offset] = updated;
        llmc_normuon_stochastic_round(updated, &parameter[offset], seed, index);
    }
}

__global__ void llmc_normuon_round_master_kernel(
    floatX* parameter,
    const float* master,
    size_t rows,
    size_t columns,
    size_t row_stride,
    size_t column_stride,
    uint64_t global_step,
    int tensor_id,
    int layer_index,
    int view_index) {
    const size_t element_count = rows * columns;
    const uint32_t seed =
        llmc_normuon_rounding_seed(global_step, tensor_id, layer_index, view_index);
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t offset =
            llmc_normuon_strided_offset(index, columns, row_stride, column_stride);
        llmc_normuon_stochastic_round(master[offset], &parameter[offset], seed, index);
    }
}

inline void llmc_normuon_row_major_gemm_ex(
    cublasHandle_t handle,
    cudaStream_t stream,
    const float* lhs,
    int lhs_rows,
    int lhs_columns,
    bool transpose_lhs,
    const float* rhs,
    int rhs_rows,
    int rhs_columns,
    bool transpose_rhs,
    float* output,
    int output_rows,
    int output_columns,
    float alpha = 1.0f,
    float beta = 0.0f) {
    const int lhs_op_rows = transpose_lhs ? lhs_columns : lhs_rows;
    const int lhs_op_columns = transpose_lhs ? lhs_rows : lhs_columns;
    const int rhs_op_rows = transpose_rhs ? rhs_columns : rhs_rows;
    const int rhs_op_columns = transpose_rhs ? rhs_rows : rhs_columns;
    if (lhs_op_columns != rhs_op_rows || lhs_op_rows != output_rows ||
        rhs_op_columns != output_columns || lhs_rows <= 0 || lhs_columns <= 0 ||
        rhs_rows <= 0 || rhs_columns <= 0 || output_rows <= 0 ||
        output_columns <= 0) {
        return;
    }
    cublasCheck(cublasSetStream(handle, stream));
    const cublasOperation_t rhs_operation =
        transpose_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t lhs_operation =
        transpose_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasCheck(cublasGemmEx(
        handle,
        rhs_operation,
        lhs_operation,
        output_columns,
        output_rows,
        lhs_op_columns,
        &alpha,
        rhs,
        CUDA_R_32F,
        rhs_columns,
        lhs,
        CUDA_R_32F,
        lhs_columns,
        &beta,
        output,
        CUDA_R_32F,
        output_columns,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));
}

inline void llmc_normuon_row_major_gemm(
    cublasHandle_t handle,
    cudaStream_t stream,
    const float* lhs,
    bool transpose_lhs,
    const float* rhs,
    bool transpose_rhs,
    float* output,
    int width,
    float alpha = 1.0f,
    float beta = 0.0f) {
    llmc_normuon_row_major_gemm_ex(
        handle,
        stream,
        lhs,
        width,
        width,
        transpose_lhs,
        rhs,
        width,
        width,
        transpose_rhs,
        output,
        width,
        width,
        alpha,
        beta);
}

inline bool llmc_normuon_apply_polynomial(
    cublasHandle_t handle,
    cudaStream_t stream,
    float* matrix,
    float* scratch_a,
    float* scratch_b,
    int* nonfinite,
    int width,
    uint32_t stage_count,
    const LlmcNormuonPolynomialStep* schedule) {
    if (handle == nullptr || matrix == nullptr || scratch_a == nullptr ||
        scratch_b == nullptr || nonfinite == nullptr || width <= 0 ||
        stage_count == 0U ||
        stage_count > LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT ||
        schedule == nullptr) {
        return false;
    }
    const size_t elements = static_cast<size_t>(width) * width;
    const uint32_t grid = llmc_normuon_grid_for_count(elements);
    for (uint32_t stage = 0U; stage < stage_count; ++stage) {
        llmc_normuon_row_major_gemm(
            handle, stream, matrix, true, matrix, false, scratch_a, width);
        llmc_normuon_scale_copy_kernel<<<grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            scratch_a, scratch_b, elements, schedule[stage].b);
        cudaCheck(cudaGetLastError());
        if (schedule[stage].c != 0.0f) {
            llmc_normuon_row_major_gemm(
                handle,
                stream,
                scratch_a,
                false,
                scratch_a,
                false,
                scratch_b,
                width,
                schedule[stage].c,
                1.0f);
        }
        llmc_normuon_row_major_gemm(
            handle, stream, matrix, false, scratch_b, false, scratch_a, width);
        llmc_normuon_add_projected_kernel<<<grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            matrix,
            scratch_a,
            nonfinite,
            elements,
            schedule[stage].a);
        cudaCheck(cudaGetLastError());
    }
    return true;
}

// Rectangular Muon polynomial.  For a tall matrix use the right Gram matrix
// X^T X; for a wide matrix use the left Gram matrix X X^T.  This keeps every
// polynomial GEMM on the smaller side while preserving the rectangular
// polar-factor update X <- aX + X p(G) or X <- aX + p(G)X.
inline bool llmc_normuon_apply_rectangular_polynomial(
    cublasHandle_t handle,
    cudaStream_t stream,
    float* matrix,
    float* scratch_a,
    float* scratch_b,
    int* nonfinite,
    int rows,
    int columns,
    uint32_t stage_count,
    const LlmcNormuonPolynomialStep* schedule) {
    if (handle == nullptr || matrix == nullptr || scratch_a == nullptr ||
        scratch_b == nullptr || nonfinite == nullptr || rows <= 0 ||
        columns <= 0 || stage_count == 0U ||
        stage_count > LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT || schedule == nullptr) {
        return false;
    }
    const int side = rows < columns ? rows : columns;
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const size_t gram_elements = static_cast<size_t>(side) * side;
    const uint32_t matrix_grid = llmc_normuon_grid_for_count(matrix_elements);
    const uint32_t gram_grid = llmc_normuon_grid_for_count(gram_elements);
    const bool tall = rows >= columns;
    for (uint32_t stage = 0U; stage < stage_count; ++stage) {
        const LlmcNormuonPolynomialStep coefficient = schedule[stage];
        if (tall) {
            llmc_normuon_row_major_gemm_ex(
                handle, stream, matrix, rows, columns, true,
                matrix, rows, columns, false, scratch_a,
                columns, columns);
        } else {
            llmc_normuon_row_major_gemm_ex(
                handle, stream, matrix, rows, columns, false,
                matrix, rows, columns, true, scratch_a,
                rows, rows);
        }
        llmc_normuon_scale_copy_kernel<<<
            gram_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            scratch_a, scratch_b, gram_elements, coefficient.b);
        cudaCheck(cudaGetLastError());
        if (coefficient.c != 0.0f) {
            llmc_normuon_row_major_gemm_ex(
                handle, stream, scratch_a, side, side, false,
                scratch_a, side, side, false, scratch_b,
                side, side, coefficient.c, 1.0f);
        }
        if (tall) {
            llmc_normuon_row_major_gemm_ex(
                handle, stream, matrix, rows, columns, false,
                scratch_b, columns, columns, false, scratch_a,
                rows, columns);
        } else {
            llmc_normuon_row_major_gemm_ex(
                handle, stream, scratch_b, rows, rows, false,
                matrix, rows, columns, false, scratch_a,
                rows, columns);
        }
        llmc_normuon_add_projected_kernel<<<
            matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            matrix, scratch_a, nonfinite, matrix_elements, coefficient.a);
        cudaCheck(cudaGetLastError());
    }
    return true;
}

inline void llmc_cachemuon_rectangular_gram_fp32(
    cublasHandle_t handle,
    cudaStream_t stream,
    const float* matrix,
    float* gram,
    int rows,
    int columns) {
    if (rows >= columns) {
        llmc_normuon_row_major_gemm_ex(
            handle, stream, matrix, rows, columns, true,
            matrix, rows, columns, false, gram, columns, columns);
    } else {
        llmc_normuon_row_major_gemm_ex(
            handle, stream, matrix, rows, columns, false,
            matrix, rows, columns, true, gram, rows, rows);
    }
}

inline void llmc_cachemuon_apply_transform_fp32(
    cublasHandle_t handle,
    cudaStream_t stream,
    const float* matrix,
    const float* transform,
    float* output,
    int rows,
    int columns) {
    const int side = rows < columns ? rows : columns;
    if (rows >= columns) {
        // The paper transposes tall inputs before applying its left transform.
        // Transposing the result back gives X Q^T in native Wup layout.
        llmc_normuon_row_major_gemm_ex(
            handle, stream, matrix, rows, columns, false,
            transform, side, side, true, output, rows, columns);
    } else {
        llmc_normuon_row_major_gemm_ex(
            handle, stream, transform, side, side, false,
            matrix, rows, columns, false, output, rows, columns);
    }
}

// FreshGNS from CacheMuon Algorithm A.3, specialized to the paper's restart
// set S={2}.  The input is already normalized.  Tall matrices keep their
// native layout while the accumulated cache transform remains the dxd
// smaller-side left transform of the paper's transposed orientation.
inline bool llmc_cachemuon_fresh_gns_fp32(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    float* matrix,
    float* alternate,
    int rows,
    int columns,
    float** direction_out,
    float** transform_out) {
    if (runtime == nullptr || handle == nullptr || matrix == nullptr ||
        alternate == nullptr || direction_out == nullptr ||
        transform_out == nullptr || rows <= 0 || columns <= 0 ||
        runtime->cache_small_total_elements == 0U) {
        return false;
    }
    const int side = rows < columns ? rows : columns;
    const size_t small_elements = static_cast<size_t>(side) * side;
    const uint32_t small_grid = llmc_normuon_grid_for_count(small_elements);
    float* gram = runtime->cache_small[0];
    float* polynomial = runtime->cache_small[1];
    float* q_local = runtime->cache_small[2];
    float* q_total = runtime->cache_small[3];
    float* temporary = runtime->cache_small[4];
    if (gram == nullptr || polynomial == nullptr || q_local == nullptr ||
        q_total == nullptr || temporary == nullptr) {
        return false;
    }
    llmc_cachemuon_identity_kernel<<<
        small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        q_local, 1U, small_elements, small_elements, side);
    cudaCheck(cudaGetLastError());
    float* current = matrix;
    float* next = alternate;
    llmc_cachemuon_rectangular_gram_fp32(
        handle, stream, current, gram, rows, columns);
    for (uint32_t stage = 1U;
         stage <= LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT;
         ++stage) {
        if (stage == LLMC_CACHEMUON_RESTART_STAGE) {
            llmc_cachemuon_apply_transform_fp32(
                handle, stream, current, q_local, next, rows, columns);
            float* exchange = current;
            current = next;
            next = exchange;
            cudaCheck(cudaMemcpyAsync(
                q_total,
                q_local,
                small_elements * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
            llmc_cachemuon_rectangular_gram_fp32(
                handle, stream, current, gram, rows, columns);
            llmc_cachemuon_identity_kernel<<<
                small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                q_local, 1U, small_elements, small_elements, side);
            cudaCheck(cudaGetLastError());
        }

        llmc_normuon_row_major_gemm_ex(
            handle, stream, gram, side, side, false,
            gram, side, side, false, temporary, side, side);
        llmc_cachemuon_form_polynomial_kernel<<<
            small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            gram,
            temporary,
            polynomial,
            runtime->nonfinite_flag,
            1U,
            small_elements,
            small_elements,
            side,
            kLlmcCacheMuonGramGns[stage - 1U]);
        cudaCheck(cudaGetLastError());

        if (stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT &&
            stage + 1U != LLMC_CACHEMUON_RESTART_STAGE) {
            llmc_normuon_row_major_gemm_ex(
                handle, stream, gram, side, side, false,
                polynomial, side, side, false, temporary, side, side);
            llmc_normuon_row_major_gemm_ex(
                handle, stream, polynomial, side, side, false,
                temporary, side, side, false, gram, side, side);
        }

        llmc_normuon_row_major_gemm_ex(
            handle, stream, q_local, side, side, false,
            polynomial, side, side, false, temporary, side, side);
        float* exchange = q_local;
        q_local = temporary;
        temporary = exchange;
    }

    llmc_cachemuon_apply_transform_fp32(
        handle, stream, current, q_local, next, rows, columns);
    current = next;
    llmc_normuon_row_major_gemm_ex(
        handle, stream, q_local, side, side, false,
        q_total, side, side, false, polynomial, side, side);
    *direction_out = current;
    *transform_out = polynomial;
    return true;
}

inline bool llmc_normuon_rectangular_tracker_direction(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    float* normalized,
    float* tracked_q,
    int* nonfinite,
    int rows,
    int columns,
    const LlmcNormuonConfig* config,
    float** direction_out) {
    if (runtime == nullptr || handle == nullptr || normalized == nullptr ||
        tracked_q == nullptr || nonfinite == nullptr || config == nullptr ||
        direction_out == nullptr || rows <= 0 || columns <= 0) {
        return false;
    }
    const int side = rows < columns ? rows : columns;
    const bool tall = rows >= columns;
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const uint32_t matrix_grid = llmc_normuon_grid_for_count(matrix_elements);
    const uint32_t phase_grid =
        llmc_normuon_grid_for_count(static_cast<size_t>(side) * side);
    float* phase = runtime->matrix[1];
    float* correction = runtime->matrix[2];
    float* scratch = runtime->matrix[3];
    float* skew = runtime->matrix[4];

    // S = Q^T D for tall matrices and S = D Q^T for wide matrices.  Both
    // cases produce the same small-side phase matrix and one large GEMM.
    if (tall) {
        llmc_normuon_row_major_gemm_ex(
            handle,
            stream,
            tracked_q,
            rows,
            columns,
            true,
            normalized,
            rows,
            columns,
            false,
            phase,
            side,
            side);
    } else {
        llmc_normuon_row_major_gemm_ex(
            handle,
            stream,
            normalized,
            rows,
            columns,
            false,
            tracked_q,
            rows,
            columns,
            true,
            phase,
            side,
            side);
    }
    llmc_normuon_sym_skew_kernel<<<
        phase_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        phase,
        skew,
        scratch,
        nonfinite,
        side,
        true);
    cudaCheck(cudaGetLastError());
    llmc_normuon_reduce_kernel<<<1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        scratch, runtime->stats + 2, phase_grid);
    cudaCheck(cudaGetLastError());

    if (config->correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
        llmc_normuon_build_damped_diagonal_correction_kernel<<<
            1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            phase,
            correction,
            runtime->stats + 2,
            runtime->stats + 3,
            nonfinite,
            side,
            config->correction_gain,
            config->epsilon);
        cudaCheck(cudaGetLastError());
        llmc_normuon_power_iteration_kernel<<<
            1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            correction,
            scratch,
            runtime->stats + 4,
            nonfinite,
            side,
            LLMC_NORMUON_TRACKER_SPECTRAL_POWER_ITERATIONS);
        cudaCheck(cudaGetLastError());
        llmc_normuon_finalize_damped_diagonal_correction_kernel<<<
            phase_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            correction,
            runtime->stats + 3,
            runtime->stats + 4,
            nonfinite,
            side,
            config->epsilon);
        cudaCheck(cudaGetLastError());
    } else {
        llmc_normuon_build_correction_kernel<<<
            phase_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            phase,
            skew,
            correction,
            runtime->stats + 2,
            nonfinite,
            side,
            config->correction_gain,
            config->epsilon,
            static_cast<int>(config->correction_mode));
        cudaCheck(cudaGetLastError());
    }

    const bool commute_canonical_stage2 =
        config->retraction_mode ==
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    if (!llmc_normuon_apply_polynomial(
            handle,
            stream,
            correction,
            scratch,
            skew,
            nonfinite,
            side,
            commute_canonical_stage2 ? 1U : config->correction_iterations,
            config->correction_schedule)) {
        return false;
    }

    // B = C - S H^{-1}; this is a small-side operation and leaves the
    // horizontal contribution to the following fused candidate kernel.
    llmc_normuon_build_rectangular_tangent_factor_kernel<<<
        phase_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        phase,
        correction,
        runtime->stats + 2,
        nonfinite,
        side,
        tall,
        config->epsilon);
    cudaCheck(cudaGetLastError());

    float* candidate = runtime->matrix[3];
    if (tall) {
        llmc_normuon_row_major_gemm_ex(
            handle,
            stream,
            tracked_q,
            rows,
            columns,
            false,
            correction,
            side,
            side,
            false,
            candidate,
            rows,
            columns);
    } else {
        llmc_normuon_row_major_gemm_ex(
            handle,
            stream,
            correction,
            side,
            side,
            false,
            tracked_q,
            rows,
            columns,
            false,
            candidate,
            rows,
            columns);
    }
    llmc_normuon_add_rectangular_horizontal_kernel<<<
        matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        candidate,
        normalized,
        phase,
        runtime->stats + 2,
        nonfinite,
        rows,
        columns,
        tall,
        config->epsilon);
    cudaCheck(cudaGetLastError());

    if (commute_canonical_stage2) {
        if (!llmc_normuon_apply_rectangular_polynomial(
                handle,
                stream,
                candidate,
                runtime->matrix[1],
                correction,
                nonfinite,
                rows,
                columns,
                1U,
                config->correction_schedule + 1U)) {
            return false;
        }
    } else if (config->retraction_mode ==
               LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ) {
        if (!llmc_normuon_apply_rectangular_polynomial(
                handle,
                stream,
                candidate,
                runtime->matrix[1],
                correction,
                nonfinite,
                rows,
                columns,
                1U,
                kLlmcRectangularCubicRetraction)) {
            return false;
        }
    }
    cudaCheck(cudaMemcpyAsync(
        tracked_q,
        candidate,
        matrix_elements * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));
    *direction_out = candidate;
    return true;
}

inline void llmc_normuon_runtime_reset(LlmcNormuonRuntime* runtime) {
    memset(runtime, 0, sizeof(*runtime));
}

inline void llmc_normuon_runtime_free(LlmcNormuonRuntime* runtime) {
    if (runtime == nullptr) {
        return;
    }
    // Borrowed activation storage is owned and freed by GPT2, not this runtime.
    if (runtime->workspace_allocation != nullptr) {
        cudaCheck(cudaFree(runtime->workspace_allocation));
    }
    if (runtime->tracked_q != nullptr) {
        cudaCheck(cudaFree(runtime->tracked_q));
    }
    free(runtime->q_valid);
    free(runtime->refresh_count);
    free(runtime->last_refresh_step);
    free(runtime->cache_host_residuals);
    free(runtime->cache_host_miss_indices);
    llmc_normuon_runtime_reset(runtime);
}

inline bool llmc_normuon_runtime_layout_workspace(
    LlmcNormuonRuntime* runtime,
    const LlmcNormuonConfig* config) {
    if (runtime == nullptr || config == nullptr || runtime->workspace == nullptr ||
        runtime->workspace_capacity_bytes < runtime->workspace_bytes) {
        return false;
    }
    for (float*& matrix : runtime->matrix) {
        matrix = nullptr;
    }
    runtime->batch_bf16[0] = nullptr;
    runtime->batch_bf16[1] = nullptr;
    for (float*& matrix : runtime->cache_small) {
        matrix = nullptr;
    }
    runtime->cache_residuals = nullptr;
    runtime->cache_miss_indices = nullptr;
    float* cursor = static_cast<float*>(runtime->workspace);
    for (size_t index = 0; index < runtime->batch_float_matrix_count; ++index) {
        runtime->matrix[index] = cursor;
        cursor += runtime->batch_total_elements;
    }
    if (config->execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED) {
        runtime->batch_bf16[0] = reinterpret_cast<uint16_t*>(cursor);
        runtime->batch_bf16[1] =
            runtime->batch_bf16[0] + runtime->batch_total_elements;
        cursor += runtime->batch_total_elements;
    }
    if (llmc_normuon_is_cache_mode(config->orthogonalization_mode)) {
        for (size_t index = 0U; index < LLMC_CACHEMUON_SMALL_PANEL_COUNT;
             ++index) {
            runtime->cache_small[index] = cursor;
            cursor += runtime->cache_small_total_elements;
        }
        runtime->cache_residuals = cursor;
        cursor += runtime->batch_matrix_capacity;
        runtime->cache_miss_indices = reinterpret_cast<int*>(cursor);
        cursor += runtime->batch_matrix_capacity;
    }
    runtime->axis_stats = cursor;
    cursor += runtime->axis_stats_elements;
    runtime->stats = cursor;
    cursor += runtime->batch_matrix_capacity *
                  LLMC_NORMUON_BATCH_STATS_STRIDE +
              8U;
    runtime->nonfinite_flag = reinterpret_cast<int*>(cursor);
    const size_t laid_out_bytes =
        reinterpret_cast<char*>(cursor + 16U) -
        static_cast<char*>(runtime->workspace);
    return laid_out_bytes <= runtime->workspace_bytes;
}

inline bool llmc_normuon_runtime_allocate(
    LlmcNormuonRuntime* runtime,
    const LlmcOptimizerPlan* plan,
    const LlmcNormuonConfig* config,
    void* borrowed_workspace = nullptr,
    size_t borrowed_workspace_bytes = 0U) {
    if (runtime == nullptr || plan == nullptr || config == nullptr || !plan->built) {
        return false;
    }
    llmc_normuon_runtime_free(runtime);
    if (plan->normuon_parameter_type_count == 0) {
        return true;
    }
    const size_t width = static_cast<size_t>(plan->channels);
    if (width == 0U || width > SIZE_MAX / width) {
        return false;
    }
    const bool rectangular = llmc_normuon_is_rectangular_mode(
        config->orthogonalization_mode);
    const size_t square_elements = width * width;
    if (rectangular && square_elements > SIZE_MAX / 4U) {
        return false;
    }
    const size_t matrix_elements =
        rectangular ? 4U * square_elements : square_elements;
    const size_t batch_views_per_matrix =
        llmc_normuon_is_split_family_mode(config->orthogonalization_mode)
            ? LLMC_NORMUON_VIEWS_PER_MLP_MATRIX
            : (rectangular ? LLMC_NORMUON_RECTANGULAR_VIEWS_PER_MLP_MATRIX
                           : LLMC_NORMUON_VIEWS_PER_MLP_MATRIX);
    const size_t batch_matrix_capacity =
        config->execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED
            ? static_cast<size_t>(plan->num_layers) *
                  batch_views_per_matrix
            : 1U;
    if (batch_matrix_capacity == 0U ||
        batch_matrix_capacity > SIZE_MAX / matrix_elements) {
        return false;
    }
    const size_t batch_total_elements =
        batch_matrix_capacity * matrix_elements;
    const size_t batch_float_matrix_count =
        config->execution_mode == LLMC_NORMUON_EXECUTION_FP32_REFERENCE
            ? 5U
            : (llmc_normuon_has_persistent_transform(
                   config->orthogonalization_mode)
                   ? 3U
                   : 2U);
    const size_t packed_float_panels =
        config->execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED ? 1U : 0U;
    if (batch_matrix_capacity >
        (SIZE_MAX - 8U) / LLMC_NORMUON_BATCH_STATS_STRIDE) {
        return false;
    }
    const size_t batch_stats_elements =
        batch_matrix_capacity * LLMC_NORMUON_BATCH_STATS_STRIDE + 8U;
    const size_t max_rows = rectangular ? 4U * width : width;
    if (batch_matrix_capacity > SIZE_MAX / max_rows) {
        return false;
    }
    const size_t axis_stats_elements =
        config->execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED
            ? batch_matrix_capacity * max_rows
            : max_rows;
    if (batch_stats_elements > SIZE_MAX - axis_stats_elements - 16U) {
        return false;
    }
    if (llmc_normuon_is_cache_mode(config->orthogonalization_mode) &&
        batch_matrix_capacity > SIZE_MAX / square_elements) {
        return false;
    }
    const size_t cache_small_total_elements =
        llmc_normuon_is_cache_mode(config->orthogonalization_mode)
            ? batch_matrix_capacity * square_elements
            : 0U;
    if (llmc_normuon_is_cache_mode(config->orthogonalization_mode) &&
        batch_matrix_capacity > SIZE_MAX / 2U) {
        return false;
    }
    if (cache_small_total_elements >
        (SIZE_MAX - 2U * batch_matrix_capacity) /
            LLMC_CACHEMUON_SMALL_PANEL_COUNT) {
        return false;
    }
    const size_t cache_elements =
        llmc_normuon_is_cache_mode(config->orthogonalization_mode)
            ? LLMC_CACHEMUON_SMALL_PANEL_COUNT * cache_small_total_elements +
                  2U * batch_matrix_capacity
            : 0U;
    if (cache_elements >
        SIZE_MAX - axis_stats_elements - batch_stats_elements - 16U) {
        return false;
    }
    const size_t tail_elements = axis_stats_elements + batch_stats_elements +
                                 cache_elements + 16U;
    const size_t panel_count =
        batch_float_matrix_count + packed_float_panels;
    if (panel_count == 0U ||
        batch_total_elements > (SIZE_MAX - tail_elements) / panel_count) {
        return false;
    }
    const size_t float_elements =
        panel_count * batch_total_elements + tail_elements;
    if (float_elements > SIZE_MAX / sizeof(float)) {
        return false;
    }
    runtime->workspace_bytes = float_elements * sizeof(float);
    if (borrowed_workspace != nullptr &&
        borrowed_workspace_bytes >= runtime->workspace_bytes) {
        runtime->workspace = borrowed_workspace;
        runtime->workspace_capacity_bytes = borrowed_workspace_bytes;
        runtime->workspace_is_borrowed = true;
    } else {
        cudaCheck(cudaMalloc(
            &runtime->workspace_allocation, runtime->workspace_bytes));
        runtime->workspace = runtime->workspace_allocation;
        runtime->workspace_capacity_bytes = runtime->workspace_bytes;
        cudaCheck(cudaMemset(
            runtime->workspace_allocation, 0, runtime->workspace_bytes));
    }
    runtime->matrix_elements = matrix_elements;
    runtime->batch_float_matrix_count = batch_float_matrix_count;
    runtime->batch_matrix_capacity = batch_matrix_capacity;
    runtime->batch_total_elements = batch_total_elements;
    runtime->axis_stats_elements = axis_stats_elements;
    runtime->cache_small_total_elements = cache_small_total_elements;
    if (!llmc_normuon_runtime_layout_workspace(runtime, config)) {
        llmc_normuon_runtime_free(runtime);
        return false;
    }

    if (llmc_normuon_has_persistent_transform(
            config->orthogonalization_mode)) {
        runtime->tracked_q_view_count =
            static_cast<size_t>(plan->num_layers) *
            (llmc_normuon_is_split_family_mode(config->orthogonalization_mode)
                 ? LLMC_NORMUON_VIEWS_PER_LAYER
                 : (rectangular ? LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER
                                 : LLMC_NORMUON_VIEWS_PER_LAYER));
        const size_t q_matrix_elements =
            (llmc_normuon_is_split_family_mode(config->orthogonalization_mode) ||
             llmc_normuon_is_cache_mode(config->orthogonalization_mode))
                ? square_elements
                : matrix_elements;
        if (runtime->tracked_q_view_count > SIZE_MAX / q_matrix_elements) {
            llmc_normuon_runtime_free(runtime);
            return false;
        }
        runtime->tracked_q_elements =
            runtime->tracked_q_view_count * q_matrix_elements;
        if (runtime->tracked_q_elements > SIZE_MAX / sizeof(float)) {
            llmc_normuon_runtime_free(runtime);
            return false;
        }
        runtime->tracked_q_bytes = runtime->tracked_q_elements * sizeof(float);
        cudaCheck(cudaMalloc(
            reinterpret_cast<void**>(&runtime->tracked_q),
            runtime->tracked_q_bytes));
        cudaCheck(cudaMemset(runtime->tracked_q, 0, runtime->tracked_q_bytes));
        runtime->q_valid = static_cast<uint8_t*>(
            calloc(runtime->tracked_q_view_count, sizeof(uint8_t)));
        runtime->refresh_count = static_cast<uint64_t*>(
            calloc(runtime->tracked_q_view_count, sizeof(uint64_t)));
        runtime->last_refresh_step = static_cast<int64_t*>(
            malloc(runtime->tracked_q_view_count * sizeof(int64_t)));
        if (llmc_normuon_is_cache_mode(config->orthogonalization_mode)) {
            runtime->cache_host_residuals = static_cast<float*>(
                malloc(batch_matrix_capacity * sizeof(float)));
            runtime->cache_host_miss_indices = static_cast<int*>(
                malloc(batch_matrix_capacity * sizeof(int)));
        }
        if (runtime->q_valid == nullptr || runtime->refresh_count == nullptr ||
            runtime->last_refresh_step == nullptr ||
            (llmc_normuon_is_cache_mode(config->orthogonalization_mode) &&
             (runtime->cache_host_residuals == nullptr ||
              runtime->cache_host_miss_indices == nullptr))) {
            llmc_normuon_runtime_free(runtime);
            return false;
        }
        for (size_t index = 0; index < runtime->tracked_q_view_count; ++index) {
            runtime->last_refresh_step[index] = -1;
        }
    }
    return true;
}

inline bool llmc_normuon_guard_ok(
    LlmcNormuonRuntime* runtime,
    cudaStream_t stream) {
    int host_flag = 0;
    cudaCheck(cudaMemcpyAsync(
        &host_flag,
        runtime->nonfinite_flag,
        sizeof(host_flag),
        cudaMemcpyDeviceToHost,
        stream));
    cudaCheck(cudaStreamSynchronize(stream));
    return host_flag == 0;
}

inline void llmc_normuon_reset_guard(
    LlmcNormuonRuntime* runtime,
    cudaStream_t stream) {
    cudaCheck(cudaMemsetAsync(
        runtime->nonfinite_flag, 0, sizeof(int), stream));
}

inline bool llmc_normuon_prepare_direction(
    LlmcNormuonRuntime* runtime,
    cudaStream_t stream,
    const floatX* gradient,
    float* momentum,
    const LlmcOptimizerMatrixView* view,
    const LlmcNormuonConfig* config,
    float gradient_scale) {
    const size_t elements = view->rows * view->columns;
    const uint32_t grid = llmc_normuon_grid_for_count(elements);
    llmc_normuon_reset_guard(runtime, stream);
    llmc_normuon_prepare_momentum_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        gradient,
        momentum,
        runtime->matrix[0],
        runtime->matrix[1],
        runtime->nonfinite_flag,
        view->rows,
        view->columns,
        view->row_stride,
        view->column_stride,
        config->momentum,
        gradient_scale);
    cudaCheck(cudaGetLastError());
    llmc_normuon_reduce_kernel<<<1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->matrix[1], runtime->stats, grid);
    cudaCheck(cudaGetLastError());
    llmc_normuon_normalize_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->matrix[0],
        runtime->stats,
        runtime->nonfinite_flag,
        elements,
        llmc_normuon_is_cache_mode(config->orthogonalization_mode) ? 1.0f
                                                                  : 1.02f,
        llmc_normuon_is_cache_mode(config->orthogonalization_mode)
            ? LLMC_CACHEMUON_EPSILON
            : config->epsilon);
    cudaCheck(cudaGetLastError());
    return true;
}

inline bool llmc_normuon_finalize_update(
    LlmcNormuonRuntime* runtime,
    cudaStream_t stream,
    floatX* parameter,
    float* master,
    float* second_moment,
    const float* direction,
    const LlmcOptimizerMatrixView* view,
    const LlmcNormuonConfig* config,
    float learning_rate,
    float learning_rate_multiplier,
    uint64_t global_step,
    int tensor_id,
    int layer_index,
    int view_index) {
    const size_t rows = view->rows;
    const size_t columns = view->columns;
    const size_t elements = rows * columns;
    const size_t axis_count = rows >= columns ? rows : columns;
    llmc_normuon_second_moment_kernel<<<
        static_cast<uint32_t>(axis_count), LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        direction,
        second_moment,
        runtime->axis_stats,
        runtime->nonfinite_flag,
        rows,
        columns,
        config->beta2,
        config->epsilon);
    cudaCheck(cudaGetLastError());
    llmc_normuon_reduce_kernel<<<1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->axis_stats, runtime->stats + 1, axis_count);
    cudaCheck(cudaGetLastError());
    llmc_normuon_reduce_squares_kernel<<<
        1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        direction, runtime->stats + 2, elements);
    cudaCheck(cudaGetLastError());
    const uint32_t grid = llmc_normuon_grid_for_count(elements);
    llmc_normuon_validate_update_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        master,
        direction,
        second_moment,
        runtime->stats + 1,
        runtime->stats + 2,
        runtime->nonfinite_flag,
        view->rows,
        view->columns,
        view->row_stride,
        view->column_stride,
        learning_rate,
        config->weight_decay,
        config->epsilon,
        config->update_scale,
        learning_rate_multiplier);
    cudaCheck(cudaGetLastError());
    if (!llmc_normuon_guard_ok(runtime, stream)) {
        return false;
    }
    llmc_normuon_apply_update_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        parameter,
        master,
        direction,
        second_moment,
        runtime->stats + 1,
        runtime->stats + 2,
        view->rows,
        view->columns,
        view->row_stride,
        view->column_stride,
        learning_rate,
        config->weight_decay,
        config->epsilon,
        config->update_scale,
        learning_rate_multiplier,
        global_step,
        tensor_id,
        layer_index,
        view_index);
    cudaCheck(cudaGetLastError());
    return true;
}

inline bool llmc_normuon_update_view(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    floatX* parameter,
    const floatX* gradient,
    float* momentum,
    float* second_moment,
    float* master,
    const LlmcOptimizerParameterType* parameter_type,
    const LlmcOptimizerMatrixView* view,
    const LlmcNormuonConfig* config,
    float learning_rate,
    float gradient_scale,
    uint64_t global_step,
    int layer_index,
    int view_index) {
    if (runtime == nullptr || handle == nullptr || parameter == nullptr ||
        gradient == nullptr || momentum == nullptr || second_moment == nullptr ||
        master == nullptr || parameter_type == nullptr || view == nullptr ||
        config == nullptr || view->rows == 0U || view->columns == 0U ||
        (!llmc_normuon_is_rectangular_mode(config->orthogonalization_mode) &&
         (view->rows != view->columns ||
          view->rows != static_cast<size_t>(parameter_type->matrix_width))) ||
        (llmc_normuon_is_rectangular_mode(config->orthogonalization_mode) &&
         (view->rows * view->columns !=
              4U * parameter_type->matrix_width * parameter_type->matrix_width)) ||
        !(learning_rate > 0.0f)) {
        return false;
    }
    if (!llmc_normuon_prepare_direction(
            runtime,
            stream,
            gradient,
            momentum,
            view,
            config,
            gradient_scale)) {
        return false;
    }

    float* direction = runtime->matrix[0];
    const int rows = static_cast<int>(view->rows);
    const int columns = static_cast<int>(view->columns);
    const size_t elements = view->rows * view->columns;
    const uint32_t grid = llmc_normuon_grid_for_count(elements);
    bool refreshed = false;
    size_t cache_q_view_index = SIZE_MAX;
    float* cache_q = nullptr;
    float* cache_transform_to_commit = nullptr;

    if (config->orthogonalization_mode == LLMC_NORMUON_ORTHO_NEWTON_SCHULZ) {
        if (!llmc_normuon_apply_polynomial(
                handle,
                stream,
                direction,
                runtime->matrix[2],
                runtime->matrix[3],
                runtime->nonfinite_flag,
                rows,
                LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                config->refresh_schedule)) {
            return false;
        }
    } else if (config->orthogonalization_mode ==
               LLMC_NORMUON_ORTHO_RECTANGULAR_MUON) {
        if (!llmc_normuon_apply_rectangular_polynomial(
                handle,
                stream,
                direction,
                runtime->matrix[2],
                runtime->matrix[3],
                runtime->nonfinite_flag,
                rows,
                columns,
                LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                config->refresh_schedule)) {
            return false;
        }
    } else if (config->orthogonalization_mode ==
               LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON) {
        const int side = rows < columns ? rows : columns;
        const size_t small_elements = static_cast<size_t>(side) * side;
        cache_q_view_index = llmc_normuon_q_view_index(
            parameter_type, layer_index, view_index, true);
        if (cache_q_view_index >= runtime->tracked_q_view_count ||
            runtime->cache_residuals == nullptr ||
            runtime->cache_host_residuals == nullptr) {
            return false;
        }
        cache_q = runtime->tracked_q + cache_q_view_index * small_elements;
        float* candidate = runtime->matrix[3];
        llmc_cachemuon_apply_transform_fp32(
            handle, stream, direction, cache_q, candidate, rows, columns);
        llmc_cachemuon_rectangular_gram_fp32(
            handle, stream, candidate, runtime->cache_small[0], rows, columns);
        llmc_cachemuon_residual_kernel<<<
            1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            runtime->cache_small[0],
            runtime->cache_residuals,
            1U,
            small_elements,
            side);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpyAsync(
            runtime->cache_host_residuals,
            runtime->cache_residuals,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
        cudaCheck(cudaStreamSynchronize(stream));
        const float residual = runtime->cache_host_residuals[0];
        refreshed = runtime->q_valid[cache_q_view_index] == 0U ||
                    !isfinite(residual) ||
                    residual > config->cache_residual_threshold;
        if (refreshed) {
            if (!llmc_cachemuon_fresh_gns_fp32(
                    runtime,
                    handle,
                    stream,
                    direction,
                    runtime->matrix[4],
                    rows,
                    columns,
                    &direction,
                    &cache_transform_to_commit)) {
                return false;
            }
        } else {
            direction = candidate;
        }
    } else if (config->orthogonalization_mode ==
               LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q) {
        const size_t q_view_index = llmc_normuon_q_view_index(
            parameter_type, layer_index, view_index, true);
        if (q_view_index >= runtime->tracked_q_view_count) {
            return false;
        }
        float* tracked_q =
            runtime->tracked_q + q_view_index * runtime->matrix_elements;
        const bool needs_refresh =
            runtime->q_valid[q_view_index] == 0U ||
            (global_step % config->refresh_interval) == 0U;
        if (needs_refresh) {
            if (!llmc_normuon_apply_rectangular_polynomial(
                    handle,
                    stream,
                    direction,
                    runtime->matrix[2],
                    runtime->matrix[3],
                    runtime->nonfinite_flag,
                    rows,
                    columns,
                    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                    config->refresh_schedule)) {
                return false;
            }
            cudaCheck(cudaMemcpyAsync(
                tracked_q,
                direction,
                elements * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
            refreshed = true;
        } else if (!llmc_normuon_rectangular_tracker_direction(
                       runtime,
                       handle,
                       stream,
                       direction,
                       tracked_q,
                       runtime->nonfinite_flag,
                       rows,
                       columns,
                       config,
                       &direction)) {
            return false;
        }
        if (!llmc_normuon_guard_ok(runtime, stream)) {
            return false;
        }
        if (refreshed) {
            runtime->q_valid[q_view_index] = 1U;
            runtime->refresh_count[q_view_index]++;
            runtime->last_refresh_step[q_view_index] =
                static_cast<int64_t>(global_step);
        }
    } else {
        const size_t q_view_index = llmc_normuon_q_view_index(
            parameter_type, layer_index, view_index, false);
        if (q_view_index >= runtime->tracked_q_view_count) {
            return false;
        }
        float* tracked_q =
            runtime->tracked_q + q_view_index * runtime->matrix_elements;
        const bool needs_refresh =
            runtime->q_valid[q_view_index] == 0U ||
            (global_step % config->refresh_interval) == 0U;
        if (needs_refresh) {
            if (!llmc_normuon_apply_polynomial(
                    handle,
                    stream,
                    direction,
                    runtime->matrix[2],
                    runtime->matrix[3],
                    runtime->nonfinite_flag,
                    rows,
                    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                    config->refresh_schedule)) {
                return false;
            }
            cudaCheck(cudaMemcpyAsync(
                tracked_q,
                direction,
                elements * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
            refreshed = true;
        } else {
            llmc_normuon_row_major_gemm(
                handle,
                stream,
                tracked_q,
                true,
                direction,
                false,
                 runtime->matrix[1],
                 rows);
            llmc_normuon_sym_skew_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                runtime->matrix[1],
                runtime->matrix[3],
                runtime->matrix[4],
                runtime->nonfinite_flag,
                view->rows,
                true);
            cudaCheck(cudaGetLastError());
            llmc_normuon_reduce_kernel<<<
                1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                runtime->matrix[4], runtime->stats + 2, grid);
            cudaCheck(cudaGetLastError());
            if (config->correction_mode ==
                LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
                llmc_normuon_build_damped_diagonal_correction_kernel<<<
                    1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                    runtime->matrix[1],
                    runtime->matrix[2],
                    runtime->stats + 2,
                    runtime->stats + 3,
                    runtime->nonfinite_flag,
                    view->rows,
                    config->correction_gain,
                    config->epsilon);
                cudaCheck(cudaGetLastError());
                llmc_normuon_power_iteration_kernel<<<
                    1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                    runtime->matrix[2],
                    runtime->matrix[4],
                    runtime->stats + 4,
                    runtime->nonfinite_flag,
                    view->rows,
                    LLMC_NORMUON_TRACKER_SPECTRAL_POWER_ITERATIONS);
                cudaCheck(cudaGetLastError());
                llmc_normuon_finalize_damped_diagonal_correction_kernel<<<
                    grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                    runtime->matrix[2],
                    runtime->stats + 3,
                    runtime->stats + 4,
                    runtime->nonfinite_flag,
                    view->rows,
                    config->epsilon);
                cudaCheck(cudaGetLastError());
            } else {
                llmc_normuon_build_correction_kernel<<<
                    grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                    runtime->matrix[1],
                    runtime->matrix[3],
                    runtime->matrix[2],
                    runtime->stats + 2,
                    runtime->nonfinite_flag,
                    view->rows,
                    config->correction_gain,
                    config->epsilon,
                    static_cast<int>(config->correction_mode));
                cudaCheck(cudaGetLastError());
            }
            const bool commute_canonical_stage2 =
                config->retraction_mode ==
                LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
            if (!llmc_normuon_apply_polynomial(
                    handle,
                    stream,
                    runtime->matrix[2],
                    runtime->matrix[3],
                    runtime->matrix[4],
                    runtime->nonfinite_flag,
                    rows,
                    commute_canonical_stage2 ? 1U : config->correction_iterations,
                    config->correction_schedule)) {
                return false;
            }
            llmc_normuon_row_major_gemm(
                handle,
                stream,
                tracked_q,
                false,
                runtime->matrix[2],
                false,
                 runtime->matrix[0],
                 rows);
            direction = runtime->matrix[0];
            if (commute_canonical_stage2) {
                if (!llmc_normuon_apply_polynomial(
                        handle,
                        stream,
                        direction,
                        runtime->matrix[2],
                        runtime->matrix[3],
                        runtime->nonfinite_flag,
                        rows,
                        1U,
                        config->correction_schedule + 1U)) {
                    return false;
                }
            } else if (config->retraction_mode ==
                       LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ) {
                llmc_normuon_row_major_gemm(
                    handle,
                    stream,
                    direction,
                    false,
                    direction,
                    true,
                    runtime->matrix[1],
                    rows);
                llmc_normuon_three_minus_kernel<<<
                    grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                    runtime->matrix[1], view->rows);
                cudaCheck(cudaGetLastError());
                llmc_normuon_row_major_gemm(
                    handle,
                    stream,
                    runtime->matrix[1],
                    false,
                    direction,
                    false,
                    runtime->matrix[2],
                    rows,
                    0.5f,
                    0.0f);
                direction = runtime->matrix[2];
            }
            cudaCheck(cudaMemcpyAsync(
                tracked_q,
                direction,
                elements * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
        }
        if (!llmc_normuon_guard_ok(runtime, stream)) {
            return false;
        }
        if (refreshed) {
            runtime->q_valid[q_view_index] = 1U;
            runtime->refresh_count[q_view_index]++;
            runtime->last_refresh_step[q_view_index] =
                static_cast<int64_t>(global_step);
        }
    }

    if (!llmc_normuon_guard_ok(runtime, stream)) {
        return false;
    }
    if (!llmc_normuon_finalize_update(
        runtime,
        stream,
        parameter,
        master,
        second_moment,
        direction,
        view,
        config,
        learning_rate,
        parameter_type->family_id == LLMC_OPTIMIZER_FAMILY_MLP_WDOWN
            ? config->wdown_learning_rate_multiplier
            : 1.0f,
        global_step,
        parameter_type->tensor_id,
        layer_index,
        view_index)) {
        return false;
    }
    if (cache_transform_to_commit != nullptr) {
        const size_t small_elements =
            parameter_type->matrix_width * parameter_type->matrix_width;
        cudaCheck(cudaMemcpyAsync(
            cache_q,
            cache_transform_to_commit,
            small_elements * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));
        runtime->q_valid[cache_q_view_index] = 1U;
        runtime->refresh_count[cache_q_view_index]++;
        runtime->last_refresh_step[cache_q_view_index] =
            static_cast<int64_t>(global_step);
    }
    return true;
}

inline bool llmc_normuon_round_master_view(
    cudaStream_t stream,
    floatX* parameter,
    const float* master,
    const LlmcOptimizerParameterType* parameter_type,
    const LlmcOptimizerMatrixView* view,
    uint64_t global_step,
    int layer_index,
    int view_index) {
    if (parameter == nullptr || master == nullptr || parameter_type == nullptr ||
        view == nullptr) {
        return false;
    }
    const size_t elements = view->rows * view->columns;
    llmc_normuon_round_master_kernel<<<
        llmc_normuon_grid_for_count(elements),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        parameter,
        master,
        view->rows,
        view->columns,
        view->row_stride,
        view->column_stride,
        global_step,
        parameter_type->tensor_id,
        layer_index,
        view_index);
    cudaCheck(cudaGetLastError());
    return true;
}

inline void llmc_normuon_header_write_u64(int* header, int index, uint64_t value) {
    memcpy(&header[index], &value, sizeof(value));
}

inline uint64_t llmc_normuon_header_read_u64(const int* header, int index) {
    uint64_t value = 0U;
    memcpy(&value, &header[index], sizeof(value));
    return value;
}

inline void llmc_normuon_header_write_float(int* header, int index, float value) {
    memcpy(&header[index], &value, sizeof(value));
}

inline float llmc_normuon_header_read_float(const int* header, int index) {
    float value = 0.0f;
    memcpy(&value, &header[index], sizeof(value));
    return value;
}

inline void llmc_normuon_encode_config(
    int* header,
    const LlmcNormuonConfig* config) {
    header[16] = static_cast<int>(config->optimizer_selection);
    header[17] = static_cast<int>(config->targeted_family_mask);
    header[18] = static_cast<int>(config->orthogonalization_mode);
    header[19] = static_cast<int>(config->refresh_policy);
    header[20] = static_cast<int>(config->correction_policy);
    header[21] = static_cast<int>(config->refresh_interval);
    header[22] = static_cast<int>(config->correction_iterations);
    header[23] = static_cast<int>(config->retraction_mode);
    llmc_normuon_header_write_float(header, 24, config->learning_rate);
    llmc_normuon_header_write_float(header, 25, config->weight_decay);
    llmc_normuon_header_write_float(header, 26, config->momentum);
    llmc_normuon_header_write_float(header, 27, config->beta2);
    llmc_normuon_header_write_float(header, 28, config->epsilon);
    llmc_normuon_header_write_float(header, 29, config->update_scale);
    llmc_normuon_header_write_float(header, 30, config->correction_gain);
    llmc_normuon_header_write_float(
        header, 33, config->wdown_learning_rate_multiplier);
    llmc_normuon_header_write_float(
        header, 34, config->cache_residual_threshold);
    header[31] = static_cast<int>(config->execution_mode);
    header[32] = static_cast<int>(config->correction_mode);
    int cursor = 64;
    for (int stage = 0; stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT; ++stage) {
        llmc_normuon_header_write_float(header, cursor++, config->refresh_schedule[stage].a);
        llmc_normuon_header_write_float(header, cursor++, config->refresh_schedule[stage].b);
        llmc_normuon_header_write_float(header, cursor++, config->refresh_schedule[stage].c);
    }
    cursor = 96;
    for (int stage = 0; stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT; ++stage) {
        llmc_normuon_header_write_float(header, cursor++, config->correction_schedule[stage].a);
        llmc_normuon_header_write_float(header, cursor++, config->correction_schedule[stage].b);
        llmc_normuon_header_write_float(header, cursor++, config->correction_schedule[stage].c);
    }
}

inline bool llmc_normuon_decode_config(
    const int* header,
    uint32_t version,
    LlmcNormuonConfig* config) {
    llmc_normuon_config_defaults(config);
    config->optimizer_selection =
        static_cast<LlmcOptimizerSelection>(header[16]);
    config->targeted_family_mask = static_cast<uint32_t>(header[17]);
    config->orthogonalization_mode =
        static_cast<LlmcNormuonOrthogonalizationMode>(header[18]);
    config->refresh_policy =
        static_cast<LlmcNormuonApproximationPolicy>(header[19]);
    config->correction_policy =
        static_cast<LlmcNormuonApproximationPolicy>(header[20]);
    config->refresh_interval = static_cast<uint32_t>(header[21]);
    config->correction_iterations = static_cast<uint32_t>(header[22]);
    config->retraction_mode =
        version >= LLMC_NORMUON_COMPANION_VERSION_WDOWN_LR_MULTIPLIER
            ? static_cast<LlmcNormuonTrackerRetractionMode>(header[23])
            : (header[23] == 0
                   ? LLMC_NORMUON_TRACKER_RETRACTION_DISABLED
                   : LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ);
    config->learning_rate = llmc_normuon_header_read_float(header, 24);
    config->weight_decay = llmc_normuon_header_read_float(header, 25);
    config->momentum = llmc_normuon_header_read_float(header, 26);
    config->beta2 = llmc_normuon_header_read_float(header, 27);
    config->epsilon = llmc_normuon_header_read_float(header, 28);
    config->update_scale = llmc_normuon_header_read_float(header, 29);
    config->correction_gain = llmc_normuon_header_read_float(header, 30);
    if (version >= LLMC_NORMUON_COMPANION_VERSION_WDOWN_LR_MULTIPLIER) {
        config->wdown_learning_rate_multiplier =
            llmc_normuon_header_read_float(header, 33);
    }
    if (version >= LLMC_NORMUON_COMPANION_VERSION_CACHE_MUON) {
        config->cache_residual_threshold =
            llmc_normuon_header_read_float(header, 34);
    }
    config->execution_mode =
        version >= LLMC_NORMUON_COMPANION_VERSION_EXECUTION_MODE
            ? static_cast<LlmcNormuonExecutionMode>(header[31])
            : LLMC_NORMUON_EXECUTION_FP32_REFERENCE;
    config->correction_mode =
        version >= LLMC_NORMUON_COMPANION_VERSION_TRACKER_CORRECTION_MODE
            ? static_cast<LlmcNormuonTrackerCorrectionMode>(header[32])
            : LLMC_NORMUON_TRACKER_CORRECTION_GLOBAL_FROBENIUS;
    int cursor = 64;
    for (int stage = 0; stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT; ++stage) {
        config->refresh_schedule[stage].a =
            llmc_normuon_header_read_float(header, cursor++);
        config->refresh_schedule[stage].b =
            llmc_normuon_header_read_float(header, cursor++);
        config->refresh_schedule[stage].c =
            llmc_normuon_header_read_float(header, cursor++);
    }
    cursor = 96;
    for (int stage = 0; stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT; ++stage) {
        config->correction_schedule[stage].a =
            llmc_normuon_header_read_float(header, cursor++);
        config->correction_schedule[stage].b =
            llmc_normuon_header_read_float(header, cursor++);
        config->correction_schedule[stage].c =
            llmc_normuon_header_read_float(header, cursor++);
    }
    LlmcNormuonConfig resolved = *config;
    char error[160];
    if (!llmc_normuon_validate_config(&resolved, error, sizeof(error))) {
        return false;
    }
    return memcmp(
               config->refresh_schedule,
               resolved.refresh_schedule,
               sizeof(config->refresh_schedule)) == 0 &&
           memcmp(
               config->correction_schedule,
               resolved.correction_schedule,
               sizeof(config->correction_schedule)) == 0;
}

inline bool llmc_normuon_companion_exists(const char* path) {
    if (path == nullptr) {
        return false;
    }
    FILE* file = fopen(path, "rb");
    if (file == nullptr) {
        return false;
    }
    fclose(file);
    return true;
}

inline void llmc_normuon_companion_path(
    char* destination,
    size_t destination_capacity,
    const char* output_dir,
    int step,
    int process_rank) {
    snprintf(
        destination,
        destination_capacity,
        "%s/normuon_state_%08d_%05d.bin",
        output_dir,
        step,
        process_rank);
}

inline bool llmc_normuon_read_companion_info(
    const char* path,
    LlmcNormuonCompanionInfo* info) {
    if (path == nullptr || info == nullptr) {
        return false;
    }
    FILE* file = fopen(path, "rb");
    if (file == nullptr) {
        return false;
    }
    int header[LLMC_NORMUON_COMPANION_HEADER_INTS];
    memset(header, 0, sizeof(header));
    const size_t count = fread(
        header, sizeof(int), LLMC_NORMUON_COMPANION_HEADER_INTS, file);
    fclose(file);
    const uint32_t version = static_cast<uint32_t>(header[1]);
    if (count != LLMC_NORMUON_COMPANION_HEADER_INTS ||
        static_cast<uint32_t>(header[0]) != LLMC_NORMUON_COMPANION_MAGIC ||
        version < LLMC_NORMUON_COMPANION_VERSION_FP32_ONLY ||
        version > LLMC_NORMUON_COMPANION_VERSION) {
        return false;
    }
    memset(info, 0, sizeof(*info));
    info->num_processes = header[2];
    info->process_rank = header[3];
    info->step = header[4];
    info->num_layers = header[5];
    info->channels = header[6];
    info->q_view_count =
        static_cast<size_t>(llmc_normuon_header_read_u64(header, 8));
    info->q_element_count =
        static_cast<size_t>(llmc_normuon_header_read_u64(header, 10));
    return llmc_normuon_decode_config(header, version, &info->config);
}

inline bool llmc_normuon_save_companion(
    const char* path,
    int step,
    int num_processes,
    int process_rank,
    const LlmcOptimizerPlan* plan,
    const LlmcNormuonConfig* config,
    const LlmcNormuonRuntime* runtime,
    cudaStream_t stream) {
    if (path == nullptr || plan == nullptr || config == nullptr ||
        runtime == nullptr || !plan->built ||
        config->optimizer_selection != LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON) {
        return false;
    }
    FILE* file = fopen(path, "wb");
    if (file == nullptr) {
        return false;
    }
    int header[LLMC_NORMUON_COMPANION_HEADER_INTS];
    memset(header, 0, sizeof(header));
    header[0] = static_cast<int>(LLMC_NORMUON_COMPANION_MAGIC);
    header[1] = static_cast<int>(LLMC_NORMUON_COMPANION_VERSION);
    header[2] = num_processes;
    header[3] = process_rank;
    header[4] = step;
    header[5] = plan->num_layers;
    header[6] = plan->channels;
    llmc_normuon_header_write_u64(
        header, 8, static_cast<uint64_t>(runtime->tracked_q_view_count));
    llmc_normuon_header_write_u64(
        header, 10, static_cast<uint64_t>(runtime->tracked_q_elements));
    llmc_normuon_encode_config(header, config);
    bool ok =
        fwrite(
            header,
            sizeof(int),
            LLMC_NORMUON_COMPANION_HEADER_INTS,
            file) == LLMC_NORMUON_COMPANION_HEADER_INTS;
    if (ok && llmc_normuon_has_persistent_transform(
                  config->orthogonalization_mode)) {
        if (runtime->tracked_q == nullptr || runtime->q_valid == nullptr ||
            runtime->refresh_count == nullptr ||
            runtime->last_refresh_step == nullptr) {
            ok = false;
        } else {
            device_to_file(
                file,
                runtime->tracked_q,
                runtime->tracked_q_bytes,
                8U * 1024U * 1024U,
                stream);
            ok =
                fwrite(
                    runtime->q_valid,
                    sizeof(uint8_t),
                    runtime->tracked_q_view_count,
                    file) == runtime->tracked_q_view_count &&
                fwrite(
                    runtime->refresh_count,
                    sizeof(uint64_t),
                    runtime->tracked_q_view_count,
                    file) == runtime->tracked_q_view_count &&
                fwrite(
                    runtime->last_refresh_step,
                    sizeof(int64_t),
                    runtime->tracked_q_view_count,
                    file) == runtime->tracked_q_view_count;
        }
    }
    if (fclose(file) != 0) {
        ok = false;
    }
    return ok;
}

inline bool llmc_normuon_load_companion(
    const char* path,
    int expected_step,
    int expected_num_processes,
    int expected_process_rank,
    const LlmcOptimizerPlan* plan,
    const LlmcNormuonConfig* config,
    LlmcNormuonRuntime* runtime,
    cudaStream_t stream) {
    LlmcNormuonCompanionInfo info;
    if (!llmc_normuon_read_companion_info(path, &info) ||
        info.step != expected_step ||
        info.num_processes != expected_num_processes ||
        info.process_rank != expected_process_rank ||
        info.num_layers != plan->num_layers ||
        info.channels != plan->channels ||
        !llmc_normuon_config_equal(&info.config, config)) {
        return false;
    }
    FILE* file = fopen(path, "rb");
    if (file == nullptr) {
        return false;
    }
    if (fseek(
            file,
            LLMC_NORMUON_COMPANION_HEADER_INTS * static_cast<long>(sizeof(int)),
            SEEK_SET) != 0) {
        fclose(file);
        return false;
    }
    bool ok = true;
    if (llmc_normuon_has_persistent_transform(
            config->orthogonalization_mode)) {
        if (info.q_view_count != runtime->tracked_q_view_count ||
            info.q_element_count != runtime->tracked_q_elements ||
            runtime->tracked_q == nullptr) {
            fclose(file);
            return false;
        }
        file_to_device(
            runtime->tracked_q,
            file,
            runtime->tracked_q_bytes,
            8U * 1024U * 1024U,
            stream);
        ok =
            fread(
                runtime->q_valid,
                sizeof(uint8_t),
                runtime->tracked_q_view_count,
                file) == runtime->tracked_q_view_count &&
            fread(
                runtime->refresh_count,
                sizeof(uint64_t),
                runtime->tracked_q_view_count,
                file) == runtime->tracked_q_view_count &&
            fread(
                runtime->last_refresh_step,
                sizeof(int64_t),
                runtime->tracked_q_view_count,
                file) == runtime->tracked_q_view_count;
    } else if (info.q_view_count != 0U || info.q_element_count != 0U) {
        ok = false;
    }
    if (ok && fgetc(file) != EOF) {
        ok = false;
    }
    fclose(file);
    return ok;
}

#endif // LLMC_NORMUON_CUH
