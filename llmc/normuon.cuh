/*
Blockwise square-view NorMuon for the llm.c GPT-2 trainer.

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
constexpr uint32_t LLMC_NORMUON_COMPANION_MAGIC = 20260716U;
constexpr uint32_t LLMC_NORMUON_COMPANION_VERSION = 1U;
constexpr int LLMC_NORMUON_COMPANION_HEADER_INTS = 256;
constexpr int LLMC_NORMUON_BLOCK_SIZE = 256;

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
};

enum LlmcNormuonApproximationPolicy {
    LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC = 0,
    LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC = 1,
    LLMC_NORMUON_APPROX_POLAR_EXPRESS = 2,
};

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

struct LlmcNormuonConfig {
    LlmcOptimizerSelection optimizer_selection;
    uint32_t targeted_family_mask;
    float learning_rate;
    float weight_decay;
    float momentum;
    float beta2;
    float epsilon;
    float update_scale;
    LlmcNormuonOrthogonalizationMode orthogonalization_mode;
    LlmcNormuonApproximationPolicy refresh_policy;
    LlmcNormuonApproximationPolicy correction_policy;
    uint32_t refresh_interval;
    uint32_t correction_iterations;
    float correction_gain;
    uint32_t retraction;
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
    size_t workspace_bytes;
    size_t matrix_elements;
    float* matrix[5];
    float* axis_stats;
    float* stats;
    int* nonfinite_flag;
    float* tracked_q;
    size_t tracked_q_elements;
    size_t tracked_q_bytes;
    size_t tracked_q_view_count;
    uint8_t* q_valid;
    uint64_t* refresh_count;
    int64_t* last_refresh_step;
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
        default:
            return nullptr;
    }
}

inline void llmc_normuon_resolve_schedules(LlmcNormuonConfig* config) {
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
    config->orthogonalization_mode = LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
    config->refresh_policy = LLMC_NORMUON_APPROX_STOCK_NORMUON_QUINTIC;
    config->correction_policy = LLMC_NORMUON_APPROX_CANONICAL_TAYLOR_QUINTIC;
    config->refresh_interval = 3U;
    config->correction_iterations = 2U;
    config->correction_gain = 1.0f;
    config->retraction = 1U;
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
        isfinite(config->correction_gain);
    bool valid =
        (config->optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW ||
         config->optimizer_selection == LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON) &&
        finite_hyperparameters &&
        config->learning_rate > 0.0f &&
        config->weight_decay >= 0.0f &&
        config->momentum >= 0.0f && config->momentum < 1.0f &&
        config->beta2 >= 0.0f && config->beta2 < 1.0f &&
        config->epsilon > 0.0f &&
        config->update_scale >= 0.0f &&
        config->correction_gain >= 0.0f &&
        config->refresh_interval > 0U &&
        config->correction_iterations > 0U &&
        config->correction_iterations <= LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT &&
        config->retraction <= 1U &&
        llmc_normuon_policy_schedule(config->refresh_policy) != nullptr &&
        llmc_normuon_policy_schedule(config->correction_policy) != nullptr;
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
    return last < parameter_type->layer_elements &&
           view->second_moment_offset + view->rows <= parameter_type->layer_elements;
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
            if (target.enumerate_matrix_view == nullptr || target.views_per_layer != 4) {
                if (error != nullptr && error_capacity != 0U) {
                    snprintf(error, error_capacity, "NorMuon tensor %d has no square-view enumerator", tensor_id);
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
         plan->normuon_view_count != num_layers * LLMC_NORMUON_VIEWS_PER_LAYER)) {
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
    int view_index) {
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
    float epsilon) {
    const float norm = sqrtf(fmaxf(*norm_squared, 0.0f));
    const float denominator = 1.02f * norm + epsilon;
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
    size_t width) {
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
        sum += symmetric * symmetric;
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
    const float* skew,
    float* correction,
    const float* symmetric_norm_squared,
    int* nonfinite,
    size_t width,
    float gain,
    float epsilon) {
    const float denominator = sqrtf(fmaxf(*symmetric_norm_squared, 0.0f)) + epsilon;
    const size_t element_count = width * width;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float value =
            (row == column ? 1.0f : 0.0f) + gain * skew[index] / denominator;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        correction[index] = value;
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
    size_t width,
    float beta2,
    float epsilon) {
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    const size_t row = blockIdx.x;
    float sum = 0.0f;
    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = direction[row * width + column];
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
        const float mean = local[0] / static_cast<float>(width);
        const float next =
            beta2 * second_moment[row] + (1.0f - beta2) * mean;
        const float contribution = local[0] / fmaxf(next, epsilon);
        if (!isfinite(next) || !isfinite(contribution)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        second_moment[row] = next;
        axis_stats[row] = contribution;
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
    int* nonfinite,
    size_t rows,
    size_t columns,
    size_t row_stride,
    size_t column_stride,
    float learning_rate,
    float weight_decay,
    float epsilon,
    float update_scale) {
    const size_t element_count = rows * columns;
    const float new_norm =
        sqrtf(fmaxf(*normalized_update_norm_squared, epsilon));
    const float target_norm = sqrtf(static_cast<float>(element_count));
    const float global_scale = target_norm / new_norm;
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float update_lr = learning_rate * update_scale;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / columns;
        const size_t offset =
            llmc_normuon_strided_offset(index, columns, row_stride, column_stride);
        const float local_scale =
            rsqrtf(fmaxf(second_moment[row], epsilon)) * global_scale;
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
    size_t rows,
    size_t columns,
    size_t row_stride,
    size_t column_stride,
    float learning_rate,
    float weight_decay,
    float epsilon,
    float update_scale,
    uint64_t global_step,
    int tensor_id,
    int layer_index,
    int view_index) {
    const size_t element_count = rows * columns;
    const float new_norm =
        sqrtf(fmaxf(*normalized_update_norm_squared, epsilon));
    const float target_norm = sqrtf(static_cast<float>(element_count));
    const float global_scale = target_norm / new_norm;
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float update_lr = learning_rate * update_scale;
    const uint32_t seed =
        llmc_normuon_rounding_seed(global_step, tensor_id, layer_index, view_index);
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < element_count; index += stride) {
        const size_t row = index / columns;
        const size_t offset =
            llmc_normuon_strided_offset(index, columns, row_stride, column_stride);
        const float local_scale =
            rsqrtf(fmaxf(second_moment[row], epsilon)) * global_scale;
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
    cublasCheck(cublasSetStream(handle, stream));
    const cublasOperation_t rhs_operation =
        transpose_rhs ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t lhs_operation =
        transpose_lhs ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasCheck(cublasGemmEx(
        handle,
        rhs_operation,
        lhs_operation,
        width,
        width,
        width,
        &alpha,
        rhs,
        CUDA_R_32F,
        width,
        lhs,
        CUDA_R_32F,
        width,
        &beta,
        output,
        CUDA_R_32F,
        width,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));
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

inline void llmc_normuon_runtime_reset(LlmcNormuonRuntime* runtime) {
    memset(runtime, 0, sizeof(*runtime));
}

inline void llmc_normuon_runtime_free(LlmcNormuonRuntime* runtime) {
    if (runtime == nullptr) {
        return;
    }
    if (runtime->workspace_allocation != nullptr) {
        cudaCheck(cudaFree(runtime->workspace_allocation));
    }
    if (runtime->tracked_q != nullptr) {
        cudaCheck(cudaFree(runtime->tracked_q));
    }
    free(runtime->q_valid);
    free(runtime->refresh_count);
    free(runtime->last_refresh_step);
    llmc_normuon_runtime_reset(runtime);
}

inline bool llmc_normuon_runtime_allocate(
    LlmcNormuonRuntime* runtime,
    const LlmcOptimizerPlan* plan,
    const LlmcNormuonConfig* config) {
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
    const size_t matrix_elements = width * width;
    const size_t float_elements =
        5U * matrix_elements + width + 16U;
    if (float_elements > SIZE_MAX / sizeof(float)) {
        return false;
    }
    runtime->workspace_bytes = float_elements * sizeof(float);
    cudaCheck(cudaMalloc(&runtime->workspace_allocation, runtime->workspace_bytes));
    cudaCheck(cudaMemset(runtime->workspace_allocation, 0, runtime->workspace_bytes));
    runtime->matrix_elements = matrix_elements;
    float* cursor = static_cast<float*>(runtime->workspace_allocation);
    for (int index = 0; index < 5; ++index) {
        runtime->matrix[index] = cursor;
        cursor += matrix_elements;
    }
    runtime->axis_stats = cursor;
    cursor += width;
    runtime->stats = cursor;
    cursor += 8U;
    runtime->nonfinite_flag = reinterpret_cast<int*>(cursor);

    if (config->orthogonalization_mode == LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q) {
        runtime->tracked_q_view_count =
            static_cast<size_t>(plan->num_layers) * LLMC_NORMUON_VIEWS_PER_LAYER;
        if (runtime->tracked_q_view_count > SIZE_MAX / matrix_elements) {
            llmc_normuon_runtime_free(runtime);
            return false;
        }
        runtime->tracked_q_elements =
            runtime->tracked_q_view_count * matrix_elements;
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
        if (runtime->q_valid == nullptr || runtime->refresh_count == nullptr ||
            runtime->last_refresh_step == nullptr) {
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
        config->epsilon);
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
    uint64_t global_step,
    int tensor_id,
    int layer_index,
    int view_index) {
    const size_t width = view->rows;
    const size_t elements = width * width;
    llmc_normuon_second_moment_kernel<<<
        static_cast<uint32_t>(width), LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        direction,
        second_moment,
        runtime->axis_stats,
        runtime->nonfinite_flag,
        width,
        config->beta2,
        config->epsilon);
    cudaCheck(cudaGetLastError());
    llmc_normuon_reduce_kernel<<<1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->axis_stats, runtime->stats + 1, width);
    cudaCheck(cudaGetLastError());
    const uint32_t grid = llmc_normuon_grid_for_count(elements);
    llmc_normuon_validate_update_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        master,
        direction,
        second_moment,
        runtime->stats + 1,
        runtime->nonfinite_flag,
        view->rows,
        view->columns,
        view->row_stride,
        view->column_stride,
        learning_rate,
        config->weight_decay,
        config->epsilon,
        config->update_scale);
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
        view->rows,
        view->columns,
        view->row_stride,
        view->column_stride,
        learning_rate,
        config->weight_decay,
        config->epsilon,
        config->update_scale,
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
        config == nullptr || view->rows != view->columns ||
        view->rows != static_cast<size_t>(parameter_type->matrix_width) ||
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
    const int width = static_cast<int>(view->rows);
    const size_t elements = view->rows * view->columns;
    const uint32_t grid = llmc_normuon_grid_for_count(elements);
    bool refreshed = false;

    if (config->orthogonalization_mode == LLMC_NORMUON_ORTHO_NEWTON_SCHULZ) {
        if (!llmc_normuon_apply_polynomial(
                handle,
                stream,
                direction,
                runtime->matrix[2],
                runtime->matrix[3],
                runtime->nonfinite_flag,
                width,
                LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                config->refresh_schedule)) {
            return false;
        }
    } else {
        const size_t q_view_index =
            llmc_normuon_q_view_index(parameter_type, layer_index, view_index);
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
                    width,
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
                width);
            llmc_normuon_sym_skew_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                runtime->matrix[1],
                runtime->matrix[3],
                runtime->matrix[4],
                runtime->nonfinite_flag,
                view->rows);
            cudaCheck(cudaGetLastError());
            llmc_normuon_reduce_kernel<<<
                1U, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                runtime->matrix[4], runtime->stats + 2, grid);
            cudaCheck(cudaGetLastError());
            llmc_normuon_build_correction_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                runtime->matrix[3],
                runtime->matrix[2],
                runtime->stats + 2,
                runtime->nonfinite_flag,
                view->rows,
                config->correction_gain,
                config->epsilon);
            cudaCheck(cudaGetLastError());
            if (!llmc_normuon_apply_polynomial(
                    handle,
                    stream,
                    runtime->matrix[2],
                    runtime->matrix[3],
                    runtime->matrix[4],
                    runtime->nonfinite_flag,
                    width,
                    config->correction_iterations,
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
                width);
            direction = runtime->matrix[0];
            if (config->retraction != 0U) {
                llmc_normuon_row_major_gemm(
                    handle,
                    stream,
                    direction,
                    false,
                    direction,
                    true,
                    runtime->matrix[1],
                    width);
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
                    width,
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
    return llmc_normuon_finalize_update(
        runtime,
        stream,
        parameter,
        master,
        second_moment,
        direction,
        view,
        config,
        learning_rate,
        global_step,
        parameter_type->tensor_id,
        layer_index,
        view_index);
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
    header[23] = static_cast<int>(config->retraction);
    llmc_normuon_header_write_float(header, 24, config->learning_rate);
    llmc_normuon_header_write_float(header, 25, config->weight_decay);
    llmc_normuon_header_write_float(header, 26, config->momentum);
    llmc_normuon_header_write_float(header, 27, config->beta2);
    llmc_normuon_header_write_float(header, 28, config->epsilon);
    llmc_normuon_header_write_float(header, 29, config->update_scale);
    llmc_normuon_header_write_float(header, 30, config->correction_gain);
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
    config->retraction = static_cast<uint32_t>(header[23]);
    config->learning_rate = llmc_normuon_header_read_float(header, 24);
    config->weight_decay = llmc_normuon_header_read_float(header, 25);
    config->momentum = llmc_normuon_header_read_float(header, 26);
    config->beta2 = llmc_normuon_header_read_float(header, 27);
    config->epsilon = llmc_normuon_header_read_float(header, 28);
    config->update_scale = llmc_normuon_header_read_float(header, 29);
    config->correction_gain = llmc_normuon_header_read_float(header, 30);
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
    const size_t count = fread(
        header, sizeof(int), LLMC_NORMUON_COMPANION_HEADER_INTS, file);
    fclose(file);
    if (count != LLMC_NORMUON_COMPANION_HEADER_INTS ||
        static_cast<uint32_t>(header[0]) != LLMC_NORMUON_COMPANION_MAGIC ||
        static_cast<uint32_t>(header[1]) != LLMC_NORMUON_COMPANION_VERSION) {
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
    return llmc_normuon_decode_config(header, &info->config);
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
    if (ok && config->orthogonalization_mode ==
                  LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q) {
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
    if (config->orthogonalization_mode ==
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q) {
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
