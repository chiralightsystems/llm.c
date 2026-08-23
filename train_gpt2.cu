/*
GPT-2 Transformer Neural Net training loop. See README.md for usage.
*/
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
// ----------- CPU utilities -----------
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
// defines: create_dir_if_not_exists, find_max_step, ends_with_bin
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
// defines: evalloader_init, evalloader_reset, evalloader_next_batch, evalloader_free
#include "llmc/dataloader.h"
// defines: manual_seed, normal_ (same as torch.manual_seed and torch.normal)
#include "llmc/rand.h"
// defines: lr_scheduler_init, get_learning_rate
#include "llmc/schedulers.h"
// defines: sample_softmax, random_f32
#include "llmc/sampler.h"
// defines: logger_init, logger_log_eval, logger_log_val, logger_log_train
#include "llmc/logger.h"
// defines: get_flops_promised
#include "llmc/mfu.h"
// defines: OutlierDetector, init_detector, update_detector
#include "llmc/outlier_detector.h"
// ----------- GPU utilities -----------
// defines:
// WARP_SIZE, MAX_1024_THREADS_BLOCKS, CEIL_DIV, cudaCheck, PRECISION_MODE
// NVTX_RANGE_FN
#include "llmc/cuda_common.h"
// defines:
// Packed128, f128, x128
// warpReduceSum, warpReduceMax, blockReduce, copy_and_cast_kernel, cudaMallocConditionallyManaged
#include "llmc/cuda_utils.cuh"
// defines: CUBLAS_LOWP, cublasCheck, cublaslt_workspace_size, cublaslt_workspace
// defines: cublas_compute, cublaslt_handle, cublas_handle
#include "llmc/cublas_common.h"
// ----------- Layer implementations in CUDA -----------
// defines: encoder_forward, encoder_backward
#include "llmc/encoder.cuh"
// defines: layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward
#include "llmc/layernorm.cuh"
// defines: matmul_cublaslt, matmul_forward, matmul_backward, gelu_forward, gelu_backward_inplace
#include "llmc/matmul.cuh"
#ifdef ENABLE_CUDNN
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "llmc/cudnn_att.h"
#else
// defines: attention_forward, attention_backward
#include "llmc/attention.cuh"
#endif
// defines: fused_classifier
#include "llmc/fused_classifier.cuh"
// defines: adamw_kernel3
#include "llmc/adamw.cuh"
// defines: global_norm_squared
#include "llmc/global_norm.cuh"
// defines: declarative optimizer plan and blockwise square-view NorMuon
#include "llmc/normuon.cuh"
// defines: optimized same-shape BF16/FP32-accumulation NorMuon batches
#include "llmc/normuon_batched.cuh"
// ----------- Multi-GPU support -----------
// defines: ncclFloatX, ncclCheck, MultiGpuConfig, ShardInfo
// defines: printf0, multi_gpu_config
// defines: multi_gpu_config_init, multi_gpu_config_free
// defines: set_zero_configs, multi_gpu_cpu_float_sum, multi_gpu_barrier
// defines: multi_gpu_get_shard_offset, multi_gpu_async_reduce_gradient
#include "llmc/zero.cuh"

// ----------------------------------------------------------------------------
// global vars for I/O
char filename_buffer[512];

// ----------------------------------------------------------------------------
// global vars containing information about the GPU this process is running on
cudaDeviceProp deviceProp; // fills in common_start()
cudaStream_t main_stream;
// buffer size to use for device <-> disk io
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

enum LlmcSequenceBoundaryPolicy {
    LLMC_SEQUENCE_BOUNDARY_FLAT_STREAM = 0,
    LLMC_SEQUENCE_BOUNDARY_ROW_RESET = 1,
};

const char* llmc_sequence_boundary_policy_name(
    LlmcSequenceBoundaryPolicy policy) {
    switch (policy) {
        case LLMC_SEQUENCE_BOUNDARY_FLAT_STREAM: return "flat_stream";
        case LLMC_SEQUENCE_BOUNDARY_ROW_RESET: return "row_reset";
        default: return "unknown";
    }
}

bool llmc_parse_sequence_boundary_policy(
    const char* text,
    LlmcSequenceBoundaryPolicy* policy) {
    if (strcmp(text, "flat_stream") == 0) {
        *policy = LLMC_SEQUENCE_BOUNDARY_FLAT_STREAM;
        return true;
    }
    if (strcmp(text, "row_reset") == 0) {
        *policy = LLMC_SEQUENCE_BOUNDARY_ROW_RESET;
        return true;
    }
    return false;
}

bool llmc_masks_sequence_final_target(LlmcSequenceBoundaryPolicy policy) {
    return policy == LLMC_SEQUENCE_BOUNDARY_ROW_RESET;
}

size_t llmc_supervised_target_count(
    size_t B,
    size_t T,
    bool mask_sequence_final_target) {
    assert(T > (mask_sequence_final_target ? 1U : 0U));
    return B * (T - (mask_sequence_final_target ? 1U : 0U));
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
constexpr const int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte; // (V, C)
    floatX* wpe; // (maxT, C)
    floatX* ln1w; // (L, C)
    floatX* ln1b; // (L, C)
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w; // (L, C)
    floatX* ln2b; // (L, C)
    floatX* fcw; // (L, 4*C, C)
    floatX* fcb; // (L, 4*C)
    floatX* fcprojw; // (L, C, 4*C)
    floatX* fcprojb; // (L, C)
    floatX* lnfw; // (C)
    floatX* lnfb; // (C)
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb

    // populate the parameter sizes in bytes (all the same for now, keeping for future use)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

// allocate memory for the parameters and point the individual tensors to the right places
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    // calculate the total number of parameters and bytes across all tensors
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }
    // malloc all parameters all at once on the device
    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));
    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }
    return params_memory;
}

constexpr int NUM_ACTIVATION_TENSORS = 21;
constexpr int ACTIVATION_TENSOR_OUTPUT = 18;
typedef struct {
    floatX* encoded; // (B, T, C)
    floatX* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    floatX* atty; // (L, B, T, C)
    // cuDNN saves only some statistics information
#if ENABLE_CUDNN
    float* att;  // (L, B, NH, T)
#else
    floatX* att; // (L, B, NH, T, T)
#endif

    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T), will be accumulated in micro-steps
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    floatX* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    floatX* output;

    // some additional scratch buffers
    floatX* scratch_bt4c;   // (B, T, 4*C)
    floatX* scratch_btc;    // (B, T, C)
} ActivationTensors;


struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};


#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, GPT2Config config, int recompute) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * T);
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(data->atty, L * B * T * C);
    #ifdef ENABLE_CUDNN
    // FP32 stats tensor for cuDNN to be passed to backward pass
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T);
    #else
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T * T);
    #endif
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[7] = TENSOR_SPEC(data->ln2, (recompute < 2) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * 4*C);
    // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * T * 4*C : B * T * 4*C);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkvr, L * B * T * 3*C);
    tensors[ACTIVATION_TENSOR_OUTPUT] = TENSOR_SPEC(data->output, B * T * max(3*C, max(NH*T, Vp)));

    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);
}

size_t activation_allocation_bytes(
    const TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0U;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; ++i) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }
    return bytes;
}

void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }

    printf0("allocating %d MiB for activations\n", (int)round(bytes / (1024 * 1024)));

    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));

    // cudaMalloc does not guarantee initial memory values so we memset the allocation here
    // this matters because e.g. non-cuDNN attention assumes the attention buffer is zeroed
    // todo - up to ~100ms on slow GPUs, could theoretically be more selective, but this is safer
    cudaCheck(cudaMemset(acts_memory, 0, bytes));

    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        // extra protection so we don't accidentally use an empty buffer
        if(tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        }else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    // gradients of the weights
    ParameterTensors grads;
    void* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    float* master_weights;     // is NULL unless fp32 weights is enabled.
    LlmcNormuonConfig optimizer_config;
    LlmcOptimizerPlan optimizer_plan;
    LlmcNormuonRuntime normuon_runtime;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    size_t acts_memory_bytes;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps
    float* accumulated_mean_loss; // GPU buffer used to accumulate loss across micro-steps
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
    unsigned long long rng_state; // the RNG state for seeding stochastic rounding etc.
    unsigned long long rng_state_last_update; // RNG before last gpt2_update() to re-round identically from master weights
    int use_master_weights; // keep master weights copy in float for optim update? 0|1
    bool init_state;   // set to true if master weights need to be initialized
    int gelu_fusion; // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    int recompute; // recompute gelu | layernorm forward during model backward? 0|1|2
    // todo - if other functions need cpu scratch buffers in the future, reuse as generic scratch?
    int* workload_indices; // encoder_backward, B*T*num_c_groups (int)
    int4* bucket_info;     // encoder_backward, B*T*num_c_groups (int4) - size for worst case
} GPT2;
struct Gpt2NormuonWorkspaceCandidate {
    void* data;
    size_t bytes;
};

inline Gpt2NormuonWorkspaceCandidate gpt2_normuon_workspace_candidate(
    GPT2* model) {
    if (model == nullptr) {
        return {nullptr, 0U};
    }
    const TensorSpec& output_spec =
        model->acts_specs[ACTIVATION_TENSOR_OUTPUT];
    return {
        model->acts.output,
        output_spec.size * sizeof_dtype(output_spec.type),
    };
}

void gpt2_init_common(GPT2 *model) {
    // common inits outside of the model weights
    // memory lazily initialized in forward()
    model->acts_memory = NULL;
    model->acts_memory_bytes = 0U;
    model->inputs = NULL;
    model->targets = NULL;
    model->accumulated_mean_loss = NULL;
    model->cpu_losses = NULL;
    // the B,T params are determined and set, fixed on first batch in forward()
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f designates no loss, set at end of forward()
    model->params_memory = NULL;
    // memory lazily initialized in backward()
    model->grads_memory = NULL;
    model->workload_indices = NULL; // on cpu, for encoder_backward
    model->bucket_info = NULL; // on cpu, for encoder_backward
    // memory lazily initialized in update()
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->master_weights = NULL;
    llmc_normuon_config_defaults(&model->optimizer_config);
    llmc_optimizer_plan_reset(&model->optimizer_plan);
    llmc_normuon_runtime_reset(&model->normuon_runtime);
    // other default settings
    model->rng_state = 13371337 + multi_gpu_config.process_rank; // used in stochastic rounding
    model->use_master_weights = 1; // safe default: do keep master weights in fp32
    model->init_state = true;
    model->recompute = 1; // good default: recompute gelu but not layernorm
    model->gelu_fusion = 0; //deviceProp.major >= 9 ? 2 : 0; // default: off for now (default must match main())
}

void gpt2_allocate_weights(GPT2 *model) {
    // fill in all the parameter tensor dimensions and types
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    // create memory for model parameters on the device
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

void gpt2_allocate_state(GPT2 *model, int B, int T) {
    printf0("allocating %d MiB for parameter gradients\n", (int)round(model->num_parameters * sizeof(floatX) / (1024 * 1024)));
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);

    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;

    // allocate the space
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    model->acts_memory_bytes =
        activation_allocation_bytes(model->acts_specs);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    // also create memory for caching inputs and targets
    cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(((void**)&model->accumulated_mean_loss), sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));

    // initialise cpu scratch buffers for encoder backward
    size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
    assert((size_t)(model->batch_size * model->seq_len) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * model->seq_len * num_c_groups);
    model->bucket_info = (int4*)mallocCheck(sizeof(int4) * model->batch_size * model->seq_len * num_c_groups);

    // cudaMallocConditionallyManaged can fall back to cudaMallocManaged if not enough memory on device
    // and returns a status code of 1 if it had to fall back, in that case we want to print warning.
    int memory_status = 0;

    // we will now init the optimizer states and master weights
    // this is usually a substantial amount of memory allocation right here.
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters; // num parameters we are responsible for
    printf0("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(float)) >> 20);
    printf0("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(float)) >> 20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    memory_status |= cudaMallocConditionallyManaged((void**)&model->m_memory, shard_num_parameters * sizeof(float));
    memory_status |= cudaMallocConditionallyManaged((void**)&model->v_memory, shard_num_parameters * sizeof(float));

    if (model->use_master_weights == 1) {
        assert(model->master_weights == nullptr);
        printf0("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        memory_status |= cudaMallocConditionallyManaged((void**) &model->master_weights, shard_num_parameters * sizeof(float));
    }
    if (!model->optimizer_plan.built) {
        char optimizer_error[256];
        if (!llmc_build_optimizer_plan(
                &model->optimizer_plan,
                &model->optimizer_config,
                model->config.num_layers,
                model->config.channels,
                model->param_elements,
                optimizer_error,
                sizeof(optimizer_error))) {
            fprintf(stderr, "Failed to build optimizer plan: %s\n", optimizer_error);
            exit(EXIT_FAILURE);
        }
    }
    // Gradient-norm readback makes only acts.output explicitly dead here.
    const Gpt2NormuonWorkspaceCandidate workspace =
        gpt2_normuon_workspace_candidate(model);
    if (!llmc_normuon_runtime_allocate(
            &model->normuon_runtime,
            &model->optimizer_plan,
            &model->optimizer_config,
            workspace.data,
            workspace.bytes)) {
        fprintf(stderr, "Failed to allocate llm.c NorMuon runtime\n");
        exit(EXIT_FAILURE);
    }
    if (model->optimizer_plan.normuon_parameter_type_count != 0) {
        if (model->normuon_runtime.workspace_is_borrowed) {
            printf0(
                "borrowing %zu MiB of synchronized output storage for "
                "reusable NorMuon workspace\n",
                model->normuon_runtime.workspace_bytes >> 20);
        } else {
            printf0(
                "allocating %zu MiB for reusable NorMuon workspace\n",
                model->normuon_runtime.workspace_bytes >> 20);
        }
        if (model->normuon_runtime.tracked_q_bytes != 0U) {
            printf0(
                "allocating %zu MiB for persistent tracker NorMuon Q\n",
                model->normuon_runtime.tracked_q_bytes >> 20);
        }
    }

    // report on mixed memory allocation status (re-using our float reduce function, bit awk ok)
    int reduced_memory_status = (int) multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced_memory_status >= 1) {
        printf0("WARNING: Fell back to cudaMallocManaged when initializing m,v,master_weights on %d GPUs\n", reduced_memory_status);
        printf0("         Prevents an OOM, but code may run much slower due to device <-> host memory movement\n");
    }
    // report on device memory usage
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf0("device memory usage: %zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);
    // give an estimate of the maximum batch size
    size_t bytes_per_sequence = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes_per_sequence += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type) / B;
    }
    printf0("memory per sequence: %zu MiB\n", bytes_per_sequence / 1024 / 1024);
    printf0(" -> estimated maximum batch size: %zu\n", B + free / bytes_per_sequence);
}

void gpt2_write_to_checkpoint(GPT2 *model, const char* checkpoint_path) {
    // write the model to a checkpoint file
    printf0("Writing model to %s\n", checkpoint_path);
    FILE *model_file = fopenCheck(checkpoint_path, "wb");
    // write the header first
    int model_header[256];
    memset(model_header, 0, sizeof(model_header));
    model_header[0] = 20240326; // magic number
    assert(PRECISION_MODE == PRECISION_FP32 || PRECISION_MODE == PRECISION_BF16);
    model_header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5; // version
    model_header[2] = model->config.max_seq_len;
    model_header[3] = model->config.vocab_size;
    model_header[4] = model->config.num_layers;
    model_header[5] = model->config.num_heads;
    model_header[6] = model->config.channels;
    model_header[7] = model->config.padded_vocab_size;
    fwriteCheck(model_header, sizeof(int), 256, model_file);
    // write the parameters
    device_to_file(model_file, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, main_stream);
    // close file, we're done
    fcloseCheck(model_file);
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, bool weight_init=true) {
    // If weight_init is true, we will load the weights from this checkpoint .bin file
    // We sometimes want this to be false, if we are going to initialize these weights from
    // the master weights that are instead stored in the state .bin file.
    // In that case, this function mostly loads the model hyperparameters from the header.

    if (PRECISION_MODE == PRECISION_FP16) {
        // TODO for later perhaps, would require us dynamically converting the
        // model weights from fp32 to fp16 online, here in this function, or writing
        // the fp16 weights directly from Python, which we only do for fp32/bf16 atm.
        fprintf(stderr, "build_from_checkpoint() does not support fp16 right now.\n");
        exit(EXIT_FAILURE);
    }

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        // 3 = fp32, padded vocab
        // 5 = bf16, padded vocab, layernorms also in bf16
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // check if the precision mode of the checkpoing matches the model precision
    if (weight_init) {
        if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
            fprintf(stderr, "Precision is configured as BF16 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: are you sure you're loading a _bf16.bin file?\n");
            exit(EXIT_FAILURE);
        }
        if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
            fprintf(stderr, "Precision is configured as FP32 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: to turn on FP32 you have to compile like: `make train_gpt2cu PRECISION=FP32`\n");
            fprintf(stderr, "---> HINT: are you sure you're loading a .bin file without any _bf16 in the name?\n");
            exit(EXIT_FAILURE);
        }
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate memory for the model parameters
    gpt2_allocate_weights(model);

    // read in the parameters if weight_init is true
    if (weight_init) {
        assert(model->params_memory != NULL);
        file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, main_stream);
    }
    fcloseCheck(model_file);

    // only return from this function once we are certain the params are ready on the GPU
    cudaCheck(cudaDeviceSynchronize());
}

bool gpt2_set_hyperparameters(GPT2Config* config, int depth, int max_seq_len) {
    int channels, num_heads;
    if      (depth == 6)  { channels = 384; num_heads = 6; }   // (unofficial) gpt2-tiny (30M)
    else if (depth == 12) { channels = 768; num_heads = 12; }  // gpt2 (124M)
    else if (depth == 24) { channels = 1024; num_heads = 16; } // gpt2-medium (350M)
    else if (depth == 36) { channels = 1280; num_heads = 20; } // gpt2-large (774M)
    else if (depth == 48) { channels = 1600; num_heads = 25; } // gpt2-xl (1558M)
    else if (depth == 60) { channels = 1920; num_heads = 30; } // (unofficial) 2.7B
    else if (depth == 72) { channels = 2880; num_heads = 30; } // (unofficial) 7.3B
    else if (depth == 84) { channels = 3456; num_heads = 36; } // (unofficial) 12.2B
    else { return false; }
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = num_heads;
    config->max_seq_len = max_seq_len;
    return true;
}

static bool parse_positive_int_(const char* text, const char** end, int* value) {
    if (text == NULL || *text == '\0') {
        return false;
    }
    errno = 0;
    char* parsed_end = NULL;
    long parsed = strtol(text, &parsed_end, 10);
    if (parsed_end == text || errno == ERANGE || parsed <= 0 || parsed > INT_MAX) {
        return false;
    }
    *end = parsed_end;
    *value = (int)parsed;
    return true;
}

bool gpt2_config_from_descriptor(GPT2Config* config, const char* descriptor) {
    // Preserve the historical dX and gpt2:dX forms at maxT=1024, while
    // allowing an explicit context override as gpt2:dX:tY.
    if (config == NULL || descriptor == NULL) {
        return false;
    }
    const char* depth_text = NULL;
    bool explicit_gpt2 = false;
    if (descriptor[0] == 'd') {
        depth_text = descriptor + 1;
    } else if (strncmp(descriptor, "gpt2:d", 6) == 0) {
        depth_text = descriptor + 6;
        explicit_gpt2 = true;
    } else {
        return false;
    }

    const char* depth_end = NULL;
    int depth = 0;
    if (!parse_positive_int_(depth_text, &depth_end, &depth)) {
        return false;
    }

    int max_seq_len = 1024;
    if (*depth_end != '\0') {
        if (!explicit_gpt2 || strncmp(depth_end, ":t", 2) != 0) {
            return false;
        }
        const char* seq_end = NULL;
        if (!parse_positive_int_(depth_end + 2, &seq_end, &max_seq_len) || *seq_end != '\0') {
            return false;
        }
    }
    return gpt2_set_hyperparameters(config, depth, max_seq_len);
}

void gpt3_set_hyperparameters(GPT2Config* config, const char* channels_str) {
    // we use channels instead of depth for GPT-3 because GPT-3 model depths are not one-to-one
    // note that our models are not necessarily identical to GPT-3 because
    // we use dense attention, not the alternating dense/banded attention of GPT-3
    int channels = atoi(channels_str);
    assert(channels > 0); // atoi returns 0 if not a number
    int depth, head_size;
    if      (channels == 384)   { depth = 6;  head_size = 64; }  // (unofficial) gpt3-tiny (31M)
    else if (channels == 768)   { depth = 12; head_size = 64; }  // gpt3-small (125M)
    else if (channels == 1024)  { depth = 24; head_size = 64; }  // gpt3-medium (350M)
    else if (channels == 1536)  { depth = 24; head_size = 96; }  // gpt3-large (760M)
    else if (channels == 2048)  { depth = 24; head_size = 128; } // gpt3-xl (1.3B) [heads fixed]
    else if (channels == 2560)  { depth = 32; head_size = 80; }  // gpt3-2.7B
    else if (channels == 4096)  { depth = 32; head_size = 128; } // gpt3-6.7B
    else if (channels == 5140)  { depth = 40; head_size = 128; } // gpt3-13B
    else if (channels == 12288) { depth = 96; head_size = 128; } // gpt3 (175B)
    else { fprintf(stderr, "Unsupported GPT-3 channels: %d\n", channels); exit(EXIT_FAILURE); }
    assert(channels % head_size == 0);
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = channels / head_size;
    config->max_seq_len = 2048; // NOTE: GPT-3 uses context length of 2048 tokens, up from 1024 in GPT-2
}

void gpt_build_from_descriptor(GPT2 *model, const char* descriptor) {
    // The model descriptor can be:
    // - legacy format "dX", where X is number, e.g. "d12". This creates GPT-2 model with 12 layers.
    // - explicit "gpt2:dX", or "gpt2:dX:tY" to override max context length.
    // - "gpt3:cX", where X is now the channel count, e.g. "gpt3:c768" is the smallest GPT-3 model.

    // check the valid prexies and dispatch to the right setup function
    assert(descriptor != NULL);
    size_t len = strlen(descriptor);
    if (gpt2_config_from_descriptor(&model->config, descriptor)) {
        // configured above
    } else if (len > 6 && strncmp(descriptor, "gpt3:c", 6) == 0) {
        gpt3_set_hyperparameters(&model->config, descriptor + 6); // pass along the channels str without the 'gpt3:c'
    } else {
        fprintf(stderr, "Unsupported model descriptor: %s\n", descriptor); exit(EXIT_FAILURE);
    }

    // both GPT-2 and GPT-3 use the same tokenizer with 50257 tokens
    model->config.vocab_size = 50257;
    model->config.padded_vocab_size = 50304; // padded to 128 for CUDA kernel efficiency

    gpt2_allocate_weights(model);

    // allocate and random init the memory for all the parameters with GPT-2 schema
    // weights ~N(0, 0.02), biases 0, c_proj weights ~N(0, 0.02/(2*L)**0.5)
    // NOTE: assuming all parameters are of the type floatX, could be relaxed later
    mt19937_state init_rng;
    manual_seed(&init_rng, 42);
    floatX* params_memory_cpu = (floatX*)mallocCheck(model->num_parameters_bytes);
    memset(params_memory_cpu, 0, model->num_parameters_bytes);
    // fill in all the weights with random values
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.num_layers);
    // we have to init all these tensors exactly in the order that PyTorch initializes them
    // so that we can match them up and get correctness and exactly the same initial conditions
    size_t L = model->config.num_layers;
    size_t offset = 0;
    for (int l = 0; l < L; l++) {
        offset = 0;
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            // the layernorm parameters are all initialized to 1
            if (l == 0 && (i == 2 || i == 8 || i == 14)) { // only at l = 0 to init these just once
                for (size_t j = 0; j < model->param_elements[i]; j++) {
                    params_memory_cpu[offset + j] = 1.0f;
                }
            }
            // weights tensors are handled here
            if ((l == 0 && (i == 0 || i == 1)) // only at l = 0, init the wte and wpe tensors
              || i == 4 || i == 6 || i == 10 || i == 12) {
                size_t n = model->param_elements[i];
                size_t layer_offset = 0;
                if (i == 0) {
                    // for wte tensor (padded vocab) override to init V instead of Vp rows
                    n = model->config.vocab_size * model->config.channels;
                }
                if (i == 4 || i == 6 || i == 10 || i == 12) {
                    // weight tensors, we are only initializing layer l
                    assert(n % L == 0);
                    n = n / L;
                    layer_offset = l * n;
                }
                // in GPT-2, the projections back into the residual stream are additionally
                // scaled by 1/sqrt(2*L) for training stability
                float scale = (i == 6 || i == 12) ? 0.02f * residual_scale : 0.02f;
                // okay let's draw the random numbers and write them
                float *fp32_buffer = (float*)mallocCheck(n * sizeof(float));
                normal_(fp32_buffer, n, 0.0f, scale, &init_rng);
                for (size_t j = 0; j < n; j++) {
                    params_memory_cpu[offset + layer_offset + j] = (floatX)fp32_buffer[j];
                }
                free(fp32_buffer);
            }
            offset += model->param_elements[i];
        }
    }

    // copy them to GPU
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, model->num_parameters_bytes, cudaMemcpyHostToDevice));
    free(params_memory_cpu);
}

// propagate inputs through the network to produce logits.
// right now, this function is fully synchronous with the host
void gpt2_forward(
    GPT2 *model,
    const int* inputs,
    size_t B,
    size_t T,
    int validation_attention_blackout_width = 0,
    bool validation_attention_disabled = false) {
    NVTX_RANGE_FN();
    // we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * T * T overflows int at B 16.

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    // validate B,T are not larger than the values used at initialisation
    // (smaller B,T are okay for inference only)
    if (B > model->batch_size || T > model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }

    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    // validate inputs, all indices must be in the range [0, V)
    // we can do this while the copies are already underway
    tokenCheck(inputs, B*T, V);

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream); // encoding goes into residual[0]

    // first layernorm isn't fused
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, main_stream);

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
        // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch = (floatX*)acts.output; // used for non-cudnn attention, fcproj, attproj, etc.

        // now do the forward pass
        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        if (!validation_attention_disabled) {
            matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
            if (validation_attention_blackout_width > 0) {
                attention_forward_cudnn_recent_blackout(
                    l_atty,
                    l_qkvr,
                    B,
                    T,
                    NH,
                    C,
                    validation_attention_blackout_width,
                    main_stream);
            } else {
                attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
            }
            matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        } else {
            // Exact attention-off ablation: remove the full residual branch,
            // including the learned attention output-projection bias.
            cudaCheck(cudaMemsetAsync(
                scratch,
                0,
                B * T * C * sizeof(floatX),
                main_stream));
        }
        #else
        if (validation_attention_blackout_width != 0 || validation_attention_disabled) {
            fprintf(stderr, "validation attention ablations require the cuDNN backend\n");
            exit(EXIT_FAILURE);
        }
        floatX* l_att = acts.att + l * B * NH * T * T;
        if (T != model->seq_len) { // unused parts of attention buffer must be zeroed (T-dependent)
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        }
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        #endif
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, model->gelu_fusion);
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);
        // OK, fusion across blocks.
        if(l+1 != L) {
            floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * T, C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B * T, C, main_stream);
        }
    }

    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}


// Forwards both the model and the loss and is used for validation splits and evals.
// In particular it populates cpu_losses with loss at each token.
// Some of the evals (e.g. HellaSwag) require the per-token losses, which are produced here.
float gpt2_validate(
    GPT2 *model,
    const int* inputs,
    const int* targets,
    size_t B,
    size_t T,
    bool mask_sequence_final_target = false,
    int validation_attention_blackout_width = 0,
    bool validation_attention_disabled = false,
    int validation_loss_ignore_prefix = 0) {
    assert(targets != NULL);
    assert(validation_attention_blackout_width >= 0);
    assert(validation_attention_blackout_width < (int)T);
    assert(validation_loss_ignore_prefix >= 0);
    assert((size_t)validation_loss_ignore_prefix +
               (mask_sequence_final_target ? 1U : 0U) <
           T);
    // forward the model itself
    gpt2_forward(
        model,
        inputs,
        B,
        T,
        validation_attention_blackout_width,
        validation_attention_disabled);
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    ActivationTensors acts = model->acts;
    float mean_loss = 0.0f;
    const size_t supervised_targets_per_row =
        T - (mask_sequence_final_target ? 1U : 0U) -
        (size_t)validation_loss_ignore_prefix;
    const size_t supervised_targets = B * supervised_targets_per_row;
    // fused classifier: does the forward pass and first part of the backward pass
    const float dloss = 1.0f / supervised_targets;
    // note: we don't need to generate dlogits here
    cudaCheck(cudaMemset(acts.losses, 0, B*T*sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V); // while the memcpy is underway, validate the targets
    fused_classifier(
        acts.output,
        acts.losses,
        dloss,
        model->targets,
        B,
        T,
        V,
        Vp,
        False,
        main_stream,
        mask_sequence_final_target);
    cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    const size_t final_position_exclusive =
        T - (mask_sequence_final_target ? 1U : 0U);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = (size_t)validation_loss_ignore_prefix;
             t < final_position_exclusive;
             ++t) {
            mean_loss += model->cpu_losses[b * T + t];
        }
    }
    mean_loss /= supervised_targets;
    cudaCheck(cudaDeviceSynchronize());
    return mean_loss;
}

void gpt2_backward_and_reduce(
    GPT2 *model,
    int* inputs,
    const int* targets,
    int grad_accum_steps,
    int micro_step,
    bool mask_sequence_final_target = false) {
    if(model->grads_memory == nullptr) {
        fprintf(stderr, "Need to allocate gradients before backward");
        exit(EXIT_FAILURE);
    }
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    // on the first micro-step zero the gradients, as we're about to += accumulate into them
    if (micro_step == 0) {
        // there are currently two state vars during the gradient accumulation inner loop:
        // 1) the losses accumulate += into acts.losses, reset here
        // 2) the gradients accumulate += into grads_memory, reset here
        cudaCheck(cudaMemsetAsync(model->acts.losses, 0, model->batch_size * model->seq_len * sizeof(float), main_stream));
        cudaCheck(cudaMemsetAsync(model->grads_memory, 0, model->num_parameters * sizeof(floatX), main_stream));
    }

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t B = model->batch_size;
    const size_t T = model->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    const size_t supervised_targets = llmc_supervised_target_count(
        B, T, mask_sequence_final_target);

    // accumulate the losses inside acts.losses, and kick off the backward pass inside the fused classifier
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    const float dloss = 1.0f / (float)(supervised_targets * grad_accum_steps);
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V);
    fused_classifier(
        acts.output,
        acts.losses,
        dloss,
        model->targets,
        B,
        T,
        V,
        Vp,
        True,
        main_stream,
        mask_sequence_final_target);

    // backward pass: go in the reverse order of the forward pass, and call backward() functions

    // reset residual stream gradients (put here to work with gradient accumulation)
    floatX* dresidual = (floatX*)model->acts.scratch_btc; // the main buffer holding the gradient in the backward pass
    cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));

    // re-use the output buffer of the forward pass as a scratchpad during backward pass
    float*  scratchF = (float*)acts.output;
    floatX* scratchX = (floatX*)acts.output;

    // we kick off the chain rule by scaling each supervised target uniformly
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over the configured target set
    // next: backward the classifier matmul
    matmul_backward(model->acts.scratch_bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    // backward the final layernorm
    floatX* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, model->acts.scratch_bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, main_stream);

    // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
    // scratch for backward computations
    floatX* dl_btc = residual;

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_ln1b = params.ln1b + l * C;
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        floatX* dl_ln1w = grads.ln1w + l * C;
        floatX* dl_ln1b = grads.ln1b + l * C;
        floatX* dl_qkvw = grads.qkvw + l * 3*C * C;
        floatX* dl_qkvb = grads.qkvb + l * 3*C;
        floatX* dl_attprojw = grads.attprojw + l * C * C;
        floatX* dl_attprojb = grads.attprojb + l * C;
        floatX* dl_ln2w = grads.ln2w + l * C;
        floatX* dl_ln2b = grads.ln2b + l * C;
        floatX* dl_fcw = grads.fcw + l * 4*C * C;
        floatX* dl_fcb = grads.fcb + l * 4*C;
        floatX* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        floatX* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch_pre_gelu = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        floatX* dl_bt4c = (floatX*)model->acts.scratch_bt4c;

        // start the backward pass for this layer
        if(model->recompute >= 1) {
            // recompute >= 1 means we recompute gelu. in this case,
            // l_fch_gelu is just a buffer, so re-compute the gelu from l_fch here
            gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, main_stream);
        }
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, scratchF, B, T, 4*C, C, main_stream, l_fch_pre_gelu, model->gelu_fusion);
        if(model->recompute >= 2) {
            // same as gelu above, l_ln1 and l_ln2 are just buffers if recompute >= 2, recompute them here on demand
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, main_stream);
        }
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, scratchF, B, T, C, 4 * C, main_stream);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, main_stream);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, scratchF, B, T, C, C, main_stream);

        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, (float*)l_att, B, T, NH, C, main_stream);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        // we need B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        floatX* buffer_a = l_atty;
        floatX* buffer_b = l_fch_pre_gelu;        // this is B x T x 4C, so even larger than what we need
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);
        #endif
        if(model->recompute >= 2) {
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, main_stream);
        }
        // QKV parameter gradients
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, scratchF, B, T, C, 3 * C, main_stream);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, main_stream);

        // Accumulate gradients from this layer in a background stream.
        if(last_step) {
            floatX* const pointers[] = {
                dl_ln1w, dl_ln1b,
                dl_qkvw, dl_qkvb,
                dl_attprojw, dl_attprojb,
                dl_ln2w, dl_ln2b,
                dl_fcw, dl_fcb,
                dl_fcprojw, dl_fcprojb
            };
            const size_t nelem[] = {
                C, C,
                3 * C * C, 3 * C,
                C * C, C,
                C, C,
                4 * C * C, 4 * C,
                C * 4 * C, C
            };
            multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
        }
    }
    encoder_backward(grads.wte, grads.wpe, scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), main_stream);

    // Aggregate all gradients that are not part of the transformer blocks
    if(last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(model->accumulated_mean_loss, model->accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&model->mean_loss, model->accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        // reduce the gradients for non-transformer block parameters
        floatX* const pointers[] = {grads.wte, grads.wpe, grads.lnfw, grads.lnfb};
        const size_t nelem[] = {Vp * C, T * C, C, C};
        multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
    }

    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        model->mean_loss /= supervised_targets * grad_accum_steps;
    } else {
        model->mean_loss = -1.f; // no loss available yet
    }
}

// Gets the offset of a specific tensor for a specific layer in the GPT2 model
// layer_id is ignored for weights that are not part of a transformer block
ShardInfo gpt2_get_tensor_at_layer(const GPT2 *model, int layer_id, int param_tensor_id) {
    // first offset our way to the parameter tensor start
    ptrdiff_t offset = 0;
    for (int i = 0; i < param_tensor_id; i++) {
        offset += (ptrdiff_t)model->param_elements[i];
    }
    size_t size = model->param_elements[param_tensor_id] ;
    // if we are in the transformer block, we need to additionally offset by the layer id
    if(2 <= param_tensor_id && param_tensor_id <= 13) {
        size /= model->config.num_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

float gpt2_calculate_grad_norm(GPT2 *model, MultiGpuConfig* multi_gpu_config) {
    NVTX_RANGE_FN();
    floatX* grads_memory = (floatX*)model->grads_memory;

    // repurposing this buffer (which isn't needed now) to write grad norm into it
    float* grad_norm_squared = (float*)model->acts.output;
    float grad_norm_squared_cpu = 0.0f;

    int num_slices[2] = {1, model->config.num_layers};
    int max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    if (multi_gpu_config->zero_stage == 1) {
        // because of the ncclReduceScatter() in backward,
        // grads_memory only contains the averaged gradients at the local shards,
        // so we only calculate the grad norm at the grads_memory belonging to the local shards
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
            ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
            ptrdiff_t offset = tensor.offset + shard.offset;
            bool is_first_pass = (i == 0);
            if((i < 2 || i > 13)) {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, 0, 1,
                                    max_num_block_sums, is_first_pass, main_stream);
            } else {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, tensor.size, model->config.num_layers,
                                    max_num_block_sums, is_first_pass, main_stream);
            }
        }
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
#if MULTI_GPU
        // further sum the (partial) squared norm across all GPUs
        ncclCheck(ncclAllReduce(grad_norm_squared, grad_norm_squared, sizeof(float), ncclFloat, ncclSum, multi_gpu_config->nccl_comm, main_stream));
#endif
    } else {
        // in regular DDP, backward has averaged the gradients across all GPUs
        // so each GPU can compute the squared norm over the whole grad vector, with no added comms needed
        global_norm_squared(grad_norm_squared, grads_memory, model->num_parameters, 0, 1, max_num_block_sums, true, main_stream);
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    }
    cudaCheck(cudaMemcpy(&grad_norm_squared_cpu, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));
    float grad_norm_cpu = sqrtf(grad_norm_squared_cpu);
    return grad_norm_cpu;
}

#include "llmc/gpt2_optimizer.cuh"
float gpt2_estimate_mfu(GPT2 *model, int num_tokens, float dt) {
    /*
    Estimate model flops utilization (MFU)
    ref: Section 2.1 of https://arxiv.org/pdf/2001.08361
    Note: Ideally, the N here would be only the parameters that actually
    participate in matrix multiplications. In this N, we are over-estimating by
    including LayerNorm params, biases, and the position embedding weights,
    but these are very small terms. Also keep in mind that we would want to exclude
    the token embedding weights, but in GPT-2 these are weight shared, so they
    participate in the classifier matmul, so they are correct to be included in N.
    Note 2: The first term (6 * N) in flops_per_token is all weight matmuls, the
    second is the attention matmul, which is also usually a small contribution.
    */
    size_t N = model->num_parameters;
    int L = model->config.num_layers;
    int C = model->config.channels;
    int T = model->seq_len;
    size_t flops_per_token = 6 * N + (size_t)6 * L * C * T;
    size_t flops_per_step = flops_per_token * num_tokens;
    // express our flops throughput as ratio of A100 bfloat16 peak flops
    float flops_achieved = (float)flops_per_step * (1.0f / dt); // per second
    float flops_promised = get_flops_promised(deviceProp.name, PRECISION_MODE) * 1e12f;
    if(flops_promised < 0) {
        return -1.f;   // don't know
    }
    float mfu = flops_achieved / flops_promised;
    return mfu;
}

void gpt2_free(GPT2 *model) {
    llmc_normuon_runtime_free(&model->normuon_runtime);
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->grads_memory);
    cudaFreeCheck(&model->m_memory);
    cudaFreeCheck(&model->v_memory);
    cudaFreeCheck(&model->master_weights);
    cudaFreeCheck(&model->acts_memory);
    cudaFreeCheck(&model->inputs);
    cudaFreeCheck(&model->targets);
    cudaFreeCheck(&model->accumulated_mean_loss);
    cudaCheck(cudaFreeHost(model->cpu_losses));
    free(model->workload_indices);
    free(model->bucket_info);
}

// ----------------------------------------------------------------------------
// common init & free code for all of train/test/profile

void common_start(bool override_enable_tf32 = true, bool print_device_info = true) {

    // get CUDA device infos
    cudaCheck(cudaGetDeviceProperties(&deviceProp, multi_gpu_config.local_device_idx));
    if (print_device_info) {
        printf("[System]\n");
        printf("Device %d: %s\n", multi_gpu_config.local_device_idx, deviceProp.name);
    }

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    #ifdef ENABLE_CUDNN
    create_cudnn();
    #endif
}

void common_free(GPT2 &model) {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    cublasCheck(cublasDestroy(cublas_handle));
    #ifdef ENABLE_CUDNN
    destroy_cudnn();
    #endif
}


void save_state(const char* filename, int step, GPT2* model, DataLoader* loader) {
    printf("Writing state to %s\n", filename);
    FILE *state_file = fopenCheck(filename, "wb");
    int state_header[256];
    memset(state_header, 0, sizeof(state_header));
    // basic identifying information
    state_header[0] = 20240527; // magic number
    state_header[1] = 1; // version number
    state_header[2] = multi_gpu_config.num_processes; // number of processes
    state_header[3] = multi_gpu_config.process_rank; // rank of this process
    state_header[4] = model->use_master_weights;  // whether we're using fp32 master weights
    state_header[5] = loader->should_shuffle; // shuffle state of the dataloader
    // int main state, start at 10 to leave some padding
    state_header[10] = step; // step of the optimization
    // model rng state, start at 20 to leave some padding
    *((unsigned long long*)&state_header[20]) = model->rng_state; // random number generator state
    *((unsigned long long*)&state_header[22]) = model->rng_state_last_update; // last gpt2_update
    // dataloader state, start at 30 to leave some padding
    *((size_t*)&state_header[30]) = loader->current_shard_idx; // shard of the dataset
    *((size_t*)&state_header[32]) = loader->current_sample_idx; // position in shard
    fwriteCheck(state_header, sizeof(int), 256, state_file);

    // write AdamW m, v, and master_weights here (they are all float)
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    device_to_file(state_file, model->m_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    device_to_file(state_file, model->v_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    if(model->use_master_weights) {
        device_to_file(state_file, model->master_weights, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    }

    // write dataloader state if we are using the Permuted version of it
    if (loader->should_shuffle) {
        fwriteCheck(&loader->glob_result.gl_pathc, sizeof(size_t), 1, state_file);  // number of shards
        fwriteCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        fwriteCheck(&loader->shard_num_samples, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        fwriteCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    fcloseCheck(state_file);
}

void load_state(int* step, GPT2* model, DataLoader* loader, const char* filename, bool reset_dataloader = false) {
    FILE *state_file = fopenCheck(filename, "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    assert(state_header[0] == 20240527); // magic number
    assert(state_header[1] == 1); // version number
    assert(state_header[2] == multi_gpu_config.num_processes); // number of processes
    assert(state_header[3] == multi_gpu_config.process_rank); // rank of this process
    int use_master_weights = state_header[4];  // whether we're using fp32 master weights
    int should_shuffle = state_header[5]; // shuffle state of the dataloader
    *step = state_header[10]; // step of the optimization
    model->rng_state = *((unsigned long long*)&state_header[20]); // random number generator state
    model->rng_state_last_update = *((unsigned long long*)&state_header[22]); // last gpt2_update
    size_t current_shard_idx = *((size_t*)&state_header[30]); // shard index
    size_t current_sample_idx = *((size_t*)&state_header[32]); // position in shard

    // read AdamW m, v, master_weights (they are all float)
    // allocate all the needed memory as necessary
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    if(use_master_weights == 1 && !model->use_master_weights) {
        printf0("Warning: Master weights are present in state, but not enabled for current run.");
    } else if (use_master_weights == 0 && model->use_master_weights) {
        printf0("Error: Master weights requested, but not present in state file.");
        exit(EXIT_FAILURE);
    }

    model->init_state = false;      // we just got the state from file, no need to do first-touch init
    assert(model->m_memory != nullptr);
    assert(model->v_memory != nullptr);
    file_to_device(model->m_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    file_to_device(model->v_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    if(model->use_master_weights) {
        assert(model->master_weights != nullptr);
        file_to_device(model->master_weights, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        // restore weights from the master weights using the RNG state before last weight update
        model->rng_state = model->rng_state_last_update;
        gpt2_update(model, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, *step, &multi_gpu_config, -1.0f, /* init_from_master_only */ true);
        model->rng_state = *((unsigned long long*)&state_header[20]); // use final RNG state from checkpoint after this
    }

    // A cache-layout change invalidates the serialized shard and intra-shard
    // permutations, but does not invalidate model or optimizer state. Keep the
    // freshly initialized loader when an explicit data-only reset is requested.
    if (reset_dataloader) {
        printf0("Resetting dataloader state while preserving checkpointed model and optimizer state.\n");
        fcloseCheck(state_file);
        return;
    }

    // revive the DataLoader object and its state
    loader->should_shuffle = should_shuffle;
    if (should_shuffle == 1) {
        // ensure the number of shards matches
        size_t glob_result_gl_pathc;
        freadCheck(&glob_result_gl_pathc, sizeof(size_t), 1, state_file);
        assert(glob_result_gl_pathc == loader->glob_result.gl_pathc);
        // read the shard indices
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        freadCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        // ensure the number of samples matches
        size_t shard_num_samples;
        freadCheck(&shard_num_samples, sizeof(size_t), 1, state_file);
        assert(shard_num_samples == loader->shard_num_samples);
        // read the intra-shard indices
        loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
        freadCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        // read the shuffle rng state
        freadCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    dataloader_resume(loader, current_shard_idx, current_sample_idx);

    // all done, close state file
    fcloseCheck(state_file);
}

void write_checkpoint(const char* output_log_dir, int step, GPT2* model, DataLoader* train_loader, MultiGpuConfig* multi_gpu_config) {
    // a checkpoint contains: model weights, optimizer/dataloader state, and a DONE file
    printf0("Writing checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    // only rank 0 writes the model file because it is the same across all ranks
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        gpt2_write_to_checkpoint(model, filename_buffer);
    }
    // all ranks write their state file
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    save_state(filename_buffer, step, model, train_loader);
    if (model->optimizer_config.optimizer_selection ==
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON) {
        char normuon_path[512];
        llmc_normuon_companion_path(
            normuon_path, sizeof(normuon_path), output_log_dir, step, rank);
        if (!llmc_normuon_save_companion(
                normuon_path,
                step,
                multi_gpu_config->num_processes,
                rank,
                &model->optimizer_plan,
                &model->optimizer_config,
                &model->normuon_runtime,
                main_stream)) {
            fprintf(stderr, "Failed to save NorMuon companion state: %s\n", normuon_path);
            exit(EXIT_FAILURE);
        }
    }
    // DONE file is a signal that this checkpoint as a whole is complete
    multi_gpu_barrier(multi_gpu_config);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        FILE* done_file = fopenCheck(filename_buffer, "w");
        fcloseCheck(done_file);
    }
}

void delete_checkpoint(const char* output_log_dir, int step, MultiGpuConfig* multi_gpu_config) {
    // mirrors write_checkpoint function, cleans up checkpoint from disk
    printf0("Deleting checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        remove(filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    remove(filename_buffer);
    char normuon_path[512];
    llmc_normuon_companion_path(normuon_path, sizeof(normuon_path), output_log_dir, step, rank);
    remove(normuon_path);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        remove(filename_buffer);
    }
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip everything below this point

// ----------------------------------------------------------------------------
// training resumption logic, very useful when jobs crash once in a while
// the goal is that we can resume optimization from any checkpoint, bit-perfect
// note that "state" refers to things not already saved in the model checkpoint file

// ----------------------------------------------------------------------------
// CLI, poor man's argparse
// (all single letters have been claimed now)

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2cu [options]\n");
    fprintf(stderr, "Options:\n");
    // file system input / output
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -e <string> input .bin filename or descriptor, see code comments as docs. (default = gpt2_124M_bf16.bin)\n");
    fprintf(stderr, "  -o <string> output log dir (default = NULL, no logging)\n");
    fprintf(stderr, "  -lg <int>   log gpu info every x steps (default = -1; disabled)\n");
    fprintf(stderr, "  -n <int>    write optimization checkpoints every how many steps? (default 0, don't)\n");
    fprintf(stderr, "  -nk <int>   max number of checkpoints to keep in the directory, removing old ones (0 = disable, default)\n");
    fprintf(stderr, "  -nm <int>   every how many step checkpoints are considered major? major checkpoints never get deleted.\n");
    fprintf(stderr, "  -y <int>    resume optimization found inside output log dir? (0=restart/overwrite, 1=resume/append)\n");
    fprintf(stderr, "  -yd <int>   reset only dataloader state on resume? (0=exact loader resume, 1=fresh loader; default=0)\n");
    fprintf(stderr, "  -yf <int>   fork NorMuon on resume? Preserve main optimizer/dataloader state but reinitialize polar/cache state (0=exact, 1=fork; default=0)\n");
    // token layout for each step of the optimization
    fprintf(stderr, "  -b <int>    (per-GPU, micro) batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -d <int>    total desired batch size (default = B * T * num_processes, i.e. no grad accumulation\n");
    fprintf(stderr, "  -bp <string> sequence boundary: flat_stream|row_reset (default = flat_stream)\n");
    // workload (number of steps)
    fprintf(stderr, "  -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)\n");
    // optimization
    fprintf(stderr, "  -k <string> learning rate scheduler (default = cosine)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -u <int>    learning rate warmup iterations (default = 0, no warmup)\n");
    fprintf(stderr, "  -q <float>  learning rate decay: final fraction, at end of training (default = 1.0 (no decay))\n");
    fprintf(stderr, "  -c <float>  weight decay (default = 0.0f)\n");
    fprintf(stderr, "  -op <string> optimizer: adamw|adamw_normuon (default = adamw)\n");
    fprintf(stderr, "  -nf <string> NorMuon families (required = mlp_wup,mlp_wdown)\n");
    fprintf(stderr, "  -nl <float>  NorMuon learning rate (default = 2.5e-3)\n");
    fprintf(stderr, "  -nw <float>  NorMuon weight decay (default = 1e-3)\n");
    fprintf(stderr, "  -nb <float>  NorMuon momentum (default = 0.95)\n");
    fprintf(stderr, "  -n2 <float>  NorMuon beta2 (default = 0.95)\n");
    fprintf(stderr, "  -ne <float>  NorMuon epsilon (default = 1e-8)\n");
    fprintf(stderr, "  -ns <float>  NorMuon update scale (default = 1.0)\n");
    fprintf(stderr, "  -nq <float>  NorMuon Wdown/c_proj learning-rate multiplier (default = 1.0)\n");
    fprintf(stderr, "  -nx <string> NorMuon execution: fp32_reference|bf16_batched\n");
    fprintf(stderr, "  -no <string> orthogonalization: newton_schulz|skew_polar_track_q|rectangular_muon|rectangular_skew_polar_track_q|split_wup_square_tracker_wdown_rectangular_muon|rectangular_cache_muon\n");
    fprintf(stderr, "  -nr <string> refresh policy: canonical_taylor_quintic|stock_normuon_quintic|polar_express|cache_muon_gram_gns (FreshGNS every step for rectangular_muon; CacheMuon pins it automatically)\n");
    fprintf(stderr, "  -nc <string> correction policy: canonical_taylor_quintic|stock_normuon_quintic|polar_express\n");
    fprintf(stderr, "  -na <string> tracker refresh mode: fixed_cadence|adaptive_mean_skew (default = fixed_cadence)\n");
    fprintf(stderr, "  -ni <int>    tracker fixed cadence (adaptive mode checks every stale step; default = 3)\n");
    fprintf(stderr, "  -nj <int>    adaptive tracker maximum refresh age (default = 9)\n");
    fprintf(stderr, "  -nu <float>  adaptive tracker Wup family-mean skew threshold (default = 0.52)\n");
    fprintf(stderr, "  -nv <float>  adaptive tracker Wdown family-mean skew threshold (default = 0.57)\n");
    fprintf(stderr, "  -nn <int>    tracker correction iterations (default = 2)\n");
    fprintf(stderr, "  -ng <float>  tracker correction gain (default = 1.0)\n");
    fprintf(stderr, "  -nh <float>  CacheMuon normalized polar-residual threshold gamma (default = 5.0)\n");
    fprintf(stderr, "  -nd <string> tracker correction: global_frobenius|diagonal_sylvester (default = global_frobenius; damp eta=0.05 raw-Frobenius-cap=0.25)\n");
    fprintf(stderr, "  -nt <string> tracker retraction: disabled|newton_schulz|commuted_canonical_stage2 (default = newton_schulz; 0|1 accepted)\n");
    fprintf(stderr, "  -np <int>    square-tracker diagnostics every N optimizer steps (0=off; default=0)\n");
    fprintf(stderr, "  -da <float>  square-tracker LR dither amplitude in [0,1) (0=off; default=0)\n");
    fprintf(stderr, "  -di <int>    square-tracker LR dither pulse interval (default=12)\n");
    fprintf(stderr, "  -dm <string> LR dither mode: walsh_pulse|sinusoidal|heterodyne_chopper (default=walsh_pulse)\n");
    fprintf(stderr, "  -dh <int>    heterodyne chopper slow-envelope period in carrier blocks (default=127)\n");
    fprintf(stderr, "  -du <int>    sinusoidal Wup period in optimizer steps (default=384)\n");
    fprintf(stderr, "  -dv <int>    sinusoidal Wdown period in optimizer steps (default=512)\n");
    fprintf(stderr, "  -dp <float>  sinusoidal phase polarity, exactly +1 or -1 (default=+1)\n");
    fprintf(stderr, "  -dx <float>  Wup excitation scale in [0,1] (default=1)\n");
    fprintf(stderr, "  -dy <float>  Wdown excitation scale in [0,1] (default=1)\n");
    fprintf(stderr, "  -dr <int>    non-committing same-batch Wdown LR replay every N updates (0=off)\n");
    fprintf(stderr, "  -dz <float>  mirrored Wdown replay LR half-span in (0,64] (default=0.5)\n");
    fprintf(stderr, "  -sl <float> outlier stability: skip update if loss goes above this in zscore (0.0f=off)\n");
    fprintf(stderr, "  -sg <float> outlier stability: skip update if grad_norm goes above this in zscore (0.0f=off)\n");
    // evaluation
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -vb <int>   validation recent-key blackout width, including current key (default = 0)\n");
    fprintf(stderr, "  -vd <int>   validation attention disabled, full residual branch removed (0/1, default = 0)\n");
    fprintf(stderr, "  -vp <int>   validation leading query positions excluded from loss (default = 0)\n");
    fprintf(stderr, "  -vl <int>   print each validation batch loss and target count (0/1, default = 0)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -gs <uint64> sample RNG seed (default = 1337)\n");
    fprintf(stderr, "  -gt <float> sample temperature (default = 1.0)\n");
    fprintf(stderr, "  -gk <int>   sample top-k; 0 disables (default = 0)\n");
    fprintf(stderr, "  -gu <float> sample top-p in (0,1]; 1 disables (default = 1.0)\n");
    fprintf(stderr, "  -gp <csv>   optional comma-separated prompt token ids (default = GPT-2 EOS)\n");
    fprintf(stderr, "  -h <int>    hellaswag eval run? (default = 0)\n");
    // debugging
    fprintf(stderr, "  -a <int>    overfit a single batch? 0/1. useful for debugging\n");
    // numerics
    fprintf(stderr, "  -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)\n");
    fprintf(stderr, "  -w <int>    keep f32 copy of weights for the optimizer? (default: 1)\n");
    fprintf(stderr, "  -ge <int>   gelu fusion: 0=none, 1=forward, 2=forward+backward (default: 2 for >=SM90, 0 for older GPUs)\n");
    // memory management
    fprintf(stderr, "  -z <int>    zero_stage, Zero Optimization Stage, 0,1,2,3 (default = 0)\n");
    fprintf(stderr, "  -r <int>    recompute: less memory but less speed. (default = 1), 0|1|2 = none,gelu,gelu+ln\n");
    // multi-node settings
    fprintf(stderr, "  -pn <int>    num_processes (default = 1)\n");
    fprintf(stderr, "  -pr <int>    process_rank (default = 0)\n");
    fprintf(stderr, "  -pg <int>    gpus_per_node (default = 8)\n");
    fprintf(stderr, "  -pm <string> nccl_init_method: tcp,fs,mpi (default = mpi)\n");
    fprintf(stderr, "  -ps <string> server_ip - used only when nccl_init_method is tcp (default = -1)\n");
    fprintf(stderr, "  -pp <string> fs_path - used only when nccl_init_method is fs (default = /tmp)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {
    // read in the (optional) command line arguments
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* load_filename = "gpt2_124M_bf16.bin"; // bf16 weights of the model
    const char* lr_scheduler_type = "cosine";
    const char* output_log_dir = NULL;
    int checkpoint_every = 0; // write checkpoints every how many steps?
    int checkpoints_keep = 0; // how long checkpoint history do we keep? (in units of checkpoints)
    int major_checkpoint_every = 0; // major checkpoints never get deleted when maintaining history
    int resume = 0; // resume the optimization, if one is found inside output_log_dir?
    int resume_reset_dataloader = 0; // preserve optimizer/model state but start the configured dataloader fresh
    int resume_fork_normuon = 0; // preserve main state while explicitly changing NorMuon config and resetting polar/cache state
    int B = 4; // batch size
    int T = 1024; // sequence length max
    LlmcSequenceBoundaryPolicy sequence_boundary_policy =
        LLMC_SEQUENCE_BOUNDARY_FLAT_STREAM;
    int total_batch_size = -1; // will be calculated down below later, if not provided
    float learning_rate = 3e-4f;
    int log_gpu_every = -1;
    int warmup_iterations = 0;
    float final_learning_rate_frac = 1.0f; // final fraction of learning rate, at end of training
    float weight_decay = 0.0f;
    float skip_update_lossz = 0.0f; // skip update if loss goes above this in zscore
    float skip_update_gradz = 0.0f; // skip update if grad_norm goes above this in zscore
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int validation_attention_blackout_width = 0;
    int validation_attention_disabled = 0;
    int validation_loss_ignore_prefix = 0;
    int validation_print_batch_losses = 0;
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    unsigned long long sample_rng_seed = 1337ULL;
    float sample_temperature = 1.0f;
    int sample_top_k = 0;
    float sample_top_p = 1.0f;
    const char* sample_prompt_token_ids_csv = nullptr;
    int overfit_single_batch = 0; // useful for debugging, 1 = only load a single data batch once
    int max_steps = -1;
    int override_enable_tf32 = 1;
    int use_master_weights = 1;
    int gelu_fusion = -1; // 0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)
    int recompute = 1; // recompute during backward setting, 0 = none, 1 = recompute gelu
    int zero_stage = 0; // Zero Optimization Stage for Multi-GPU training
    int hellaswag_eval = 0;
    LlmcNormuonConfig optimizer_config;
    llmc_normuon_config_defaults(&optimizer_config);
    int optimizer_cli_explicit = 0;
    int normuon_tracker_diagnostics_every = 0;
    float normuon_lr_dither_amplitude = 0.0f;
    int normuon_lr_dither_interval = 12;
    LlmcNormuonLrDitherMode normuon_lr_dither_mode =
        LLMC_NORMUON_LR_DITHER_WALSH_PULSE;
    int normuon_lr_dither_wup_period = 384;
    int normuon_lr_dither_wdown_period = 512;
    int normuon_lr_dither_envelope_blocks = 127;
    float normuon_lr_dither_phase_polarity = 1.0f;
    float normuon_lr_dither_wup_scale = 1.0f;
    float normuon_lr_dither_wdown_scale = 1.0f;
    int normuon_batch_replay_every = 0;
    float normuon_batch_replay_amplitude = 0.5f;
    // multi-node settings
    int num_processes = 1;  // this should be set by the slurm environment
    int process_rank = 0;  // this should be set by the slurm environment
    int gpus_per_node = 8;  // this should be set by the slurm environment
    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    char server_ip[256] = "";  // used if init_method set to "tcp" -> set to your server ip address
    char fs_path[256] = "";  // used if init_method set to "fs" -> set to a shared filesystem path
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { error_usage(); } // must be -x[y] (one dash, one or two letters)
        // Keep evaluation-only controls outside the already very deep legacy
        // else-if parser so MSVC does not exceed its nested-block limit.
        if (strcmp(argv[i], "-vb") == 0) {
            validation_attention_blackout_width = atoi(argv[i+1]);
            continue;
        }
        if (strcmp(argv[i], "-vd") == 0) {
            validation_attention_disabled = atoi(argv[i+1]);
            continue;
        }
        if (strcmp(argv[i], "-vp") == 0) {
            validation_loss_ignore_prefix = atoi(argv[i+1]);
            continue;
        }
        if (strcmp(argv[i], "-vl") == 0) {
            validation_print_batch_losses = atoi(argv[i+1]);
            continue;
        }
        // read in the args
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'e') { load_filename = argv[i+1]; }
        else if (argv[i][1] == 'o' && argv[i][2] == 'p') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_optimizer_selection(argv[i+1], &optimizer_config.optimizer_selection)) { error_usage(); }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'f') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_targeted_families(argv[i+1], &optimizer_config.targeted_family_mask)) { error_usage(); }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'x') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_execution_mode(argv[i+1], &optimizer_config.execution_mode)) { error_usage(); }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'o') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_orthogonalization_mode(argv[i+1], &optimizer_config.orthogonalization_mode)) { error_usage(); }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'r') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_approximation_policy(argv[i+1], &optimizer_config.refresh_policy)) { error_usage(); }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'c') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_approximation_policy(argv[i+1], &optimizer_config.correction_policy)) { error_usage(); }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'l') { optimizer_cli_explicit = 1; optimizer_config.learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'w') { optimizer_cli_explicit = 1; optimizer_config.weight_decay = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'b') { optimizer_cli_explicit = 1; optimizer_config.momentum = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == '2') { optimizer_cli_explicit = 1; optimizer_config.beta2 = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'e') { optimizer_cli_explicit = 1; optimizer_config.epsilon = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 's') { optimizer_cli_explicit = 1; optimizer_config.update_scale = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'q') { optimizer_cli_explicit = 1; optimizer_config.wdown_learning_rate_multiplier = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'a') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_tracker_refresh_mode(
                    argv[i+1], &optimizer_config.tracker_refresh_mode)) {
                error_usage();
            }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'i') { optimizer_cli_explicit = 1; optimizer_config.refresh_interval = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'j') { optimizer_cli_explicit = 1; optimizer_config.tracker_max_refresh_age = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'u') { optimizer_cli_explicit = 1; optimizer_config.tracker_wup_skew_threshold = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'v') { optimizer_cli_explicit = 1; optimizer_config.tracker_wdown_skew_threshold = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'n') { optimizer_cli_explicit = 1; optimizer_config.correction_iterations = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'g') { optimizer_cli_explicit = 1; optimizer_config.correction_gain = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'h') { optimizer_cli_explicit = 1; optimizer_config.cache_residual_threshold = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'd') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_tracker_correction_mode(
                    argv[i+1], &optimizer_config.correction_mode)) {
                error_usage();
            }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 't') {
            optimizer_cli_explicit = 1;
            if (!llmc_parse_normuon_tracker_retraction_mode(argv[i+1], &optimizer_config.retraction_mode)) { error_usage(); }
        }
        else if (argv[i][1] == 'n' && argv[i][2] == 'p') {
            normuon_tracker_diagnostics_every = atoi(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'a') {
            normuon_lr_dither_amplitude = atof(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'i') {
            normuon_lr_dither_interval = atoi(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'm') {
            if (!llmc_parse_normuon_lr_dither_mode(
                    argv[i+1], &normuon_lr_dither_mode)) {
                error_usage();
            }
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'u') {
            normuon_lr_dither_wup_period = atoi(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'v') {
            normuon_lr_dither_wdown_period = atoi(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'h') {
            normuon_lr_dither_envelope_blocks = atoi(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'p') {
            normuon_lr_dither_phase_polarity = atof(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'x') {
            normuon_lr_dither_wup_scale = atof(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'y') {
            normuon_lr_dither_wdown_scale = atof(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'r') {
            normuon_batch_replay_every = atoi(argv[i+1]);
        }
        else if (argv[i][1] == 'd' && argv[i][2] == 'z') {
            normuon_batch_replay_amplitude = atof(argv[i+1]);
        }
        else if (argv[i][1] == 'o') { output_log_dir = argv[i+1]; }
        else if (argv[i][1] == 'n' && argv[i][2] == '\0') { checkpoint_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'y' && argv[i][2] == 'd') { resume_reset_dataloader = atoi(argv[i+1]); }
        else if (argv[i][1] == 'y' && argv[i][2] == 'f') { resume_fork_normuon = atoi(argv[i+1]); }
        else if (argv[i][1] == 'y' && argv[i][2] == '\0') { resume = atoi(argv[i+1]); }
        else if (argv[i][1] == 'b' && argv[i][2] == 'p') {
            if (!llmc_parse_sequence_boundary_policy(
                    argv[i+1], &sequence_boundary_policy)) {
                error_usage();
            }
        }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); } // Per-GPU (micro) batch size
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'd') { total_batch_size = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == '\0') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == 'g') { log_gpu_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'u') { warmup_iterations = atoi(argv[i+1]); }
        else if (argv[i][1] == 'q') { final_learning_rate_frac = atof(argv[i+1]); }
        else if (argv[i][1] == 'c') { weight_decay = atof(argv[i+1]); }
        else if (argv[i][1] == 'x') { max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == '\0') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { gelu_fusion = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 's') { sample_rng_seed = strtoull(argv[i+1], nullptr, 10); }
        else if (argv[i][1] == 'g' && argv[i][2] == 't') { sample_temperature = atof(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'k') { sample_top_k = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'u') { sample_top_p = atof(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'p') { sample_prompt_token_ids_csv = argv[i+1]; }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'a') { overfit_single_batch = atoi(argv[i+1]); }
        else if (argv[i][1] == 'f') { override_enable_tf32 = atoi(argv[i+1]); }
        else if (argv[i][1] == 'w') { use_master_weights = atoi(argv[i+1]); }
        else if (argv[i][1] == 'z') { zero_stage = atoi(argv[i+1]); }
        else if (argv[i][1] == 'r') { recompute = atoi(argv[i+1]); }
        else if (argv[i][1] == 'h') { hellaswag_eval = atoi(argv[i+1]); }
        else if (argv[i][1] == 'k') { lr_scheduler_type = argv[i+1]; }
        else if (argv[i][1] == 'p' && argv[i][2] == 'i') { strcpy(nccl_init_method, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'f') { strcpy(fs_path, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 's') { strcpy(server_ip, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'n') { num_processes = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'r') { process_rank = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'g') { gpus_per_node = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'l') { skip_update_lossz = atof(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'g') { skip_update_gradz = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'k') { checkpoints_keep = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'm') { major_checkpoint_every = atoi(argv[i+1]); }
        else { error_usage(); }
    }

    char optimizer_config_error[256];
    if (resume_reset_dataloader < 0 || resume_reset_dataloader > 1) {
        fprintf(stderr, "-yd must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }
    if (resume_reset_dataloader != 0 && resume != 1) {
        fprintf(stderr, "-yd 1 requires -y 1\n");
        exit(EXIT_FAILURE);
    }
    if (resume_fork_normuon < 0 || resume_fork_normuon > 1) {
        fprintf(stderr, "-yf must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }
    if (resume_fork_normuon != 0 && resume != 1) {
        fprintf(stderr, "-yf 1 requires -y 1\n");
        exit(EXIT_FAILURE);
    }
    if (resume_fork_normuon != 0 &&
        (!optimizer_cli_explicit ||
         optimizer_config.optimizer_selection != LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON)) {
        fprintf(stderr, "-yf 1 requires an explicit adamw_normuon target configuration\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_tracker_diagnostics_every < 0) {
        fprintf(stderr, "-np must be nonnegative\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_tracker_diagnostics_every > 0 &&
        (optimizer_config.optimizer_selection !=
             LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON ||
         optimizer_config.execution_mode != LLMC_NORMUON_EXECUTION_BF16_BATCHED ||
         optimizer_config.orthogonalization_mode !=
             LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q)) {
        fprintf(
            stderr,
            "-np currently requires adamw_normuon, bf16_batched, and "
            "skew_polar_track_q\n");
        exit(EXIT_FAILURE);
    }
    if (!isfinite(normuon_lr_dither_amplitude) ||
        normuon_lr_dither_amplitude < 0.0f ||
        normuon_lr_dither_amplitude >= 1.0f) {
        fprintf(stderr, "-da must be finite and in [0,1)\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_lr_dither_amplitude > 0.0f &&
        (optimizer_config.optimizer_selection !=
             LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON ||
         optimizer_config.execution_mode != LLMC_NORMUON_EXECUTION_BF16_BATCHED ||
         optimizer_config.orthogonalization_mode !=
             LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q ||
         optimizer_config.tracker_refresh_mode !=
             LLMC_NORMUON_TRACKER_REFRESH_FIXED_CADENCE)) {
        fprintf(
            stderr,
            "LR dither requires adamw_normuon, bf16_batched square tracker, "
            "and fixed cadence\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_lr_dither_amplitude > 0.0f &&
        normuon_lr_dither_mode == LLMC_NORMUON_LR_DITHER_WALSH_PULSE &&
        (normuon_lr_dither_interval <= 0 ||
         (static_cast<uint32_t>(normuon_lr_dither_interval) %
              optimizer_config.refresh_interval) != 0U)) {
        fprintf(
            stderr,
            "Walsh LR dither requires a positive pulse interval divisible "
            "by the tracker refresh interval\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_lr_dither_amplitude > 0.0f &&
        normuon_lr_dither_mode == LLMC_NORMUON_LR_DITHER_SINUSOIDAL &&
        (normuon_lr_dither_wup_period < 4 ||
         normuon_lr_dither_wdown_period < 4 ||
         !isfinite(normuon_lr_dither_phase_polarity) ||
         fabsf(fabsf(normuon_lr_dither_phase_polarity) - 1.0f) > 1.0e-6f)) {
        fprintf(
            stderr,
            "Sinusoidal LR dither requires both periods >= 4 and phase "
            "polarity exactly +1 or -1\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_lr_dither_amplitude > 0.0f &&
        normuon_lr_dither_mode == LLMC_NORMUON_LR_DITHER_HETERODYNE_CHOPPER &&
        (normuon_lr_dither_envelope_blocks < 4 ||
         !isfinite(normuon_lr_dither_phase_polarity) ||
         fabsf(fabsf(normuon_lr_dither_phase_polarity) - 1.0f) > 1.0e-6f)) {
        fprintf(
            stderr,
            "Heterodyne-chopper LR dither requires envelope blocks >= 4 and "
            "phase polarity exactly +1 or -1\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_lr_dither_amplitude > 0.0f &&
        (!isfinite(normuon_lr_dither_wup_scale) ||
         !isfinite(normuon_lr_dither_wdown_scale) ||
         normuon_lr_dither_wup_scale < 0.0f ||
         normuon_lr_dither_wup_scale > 1.0f ||
         normuon_lr_dither_wdown_scale < 0.0f ||
         normuon_lr_dither_wdown_scale > 1.0f ||
         (normuon_lr_dither_wup_scale == 0.0f &&
          normuon_lr_dither_wdown_scale == 0.0f))) {
        fprintf(
            stderr,
            "LR dither family scales must be finite in [0,1], with at least "
            "one family active\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_batch_replay_every < 0 ||
        !isfinite(normuon_batch_replay_amplitude) ||
        !(normuon_batch_replay_amplitude > 0.0f) ||
        normuon_batch_replay_amplitude > 64.0f) {
        fprintf(stderr, "-dr must be nonnegative and -dz must be finite in (0,64]\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_batch_replay_every > 0 &&
        (optimizer_config.optimizer_selection !=
             LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON ||
         optimizer_config.execution_mode != LLMC_NORMUON_EXECUTION_BF16_BATCHED ||
         optimizer_config.orthogonalization_mode !=
             LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q ||
         optimizer_config.tracker_refresh_mode !=
             LLMC_NORMUON_TRACKER_REFRESH_FIXED_CADENCE ||
         num_processes != 1 || zero_stage != 0 || !use_master_weights)) {
        fprintf(
            stderr,
            "same-batch Wdown LR replay requires single-GPU zero-stage-0 "
            "adamw_normuon with master weights, bf16_batched square tracker, "
            "and fixed cadence\n");
        exit(EXIT_FAILURE);
    }
    const bool mask_sequence_final_target =
        llmc_masks_sequence_final_target(sequence_boundary_policy);
    if (mask_sequence_final_target && T < 2) {
        fprintf(stderr, "-bp row_reset requires sequence length T >= 2\n");
        exit(EXIT_FAILURE);
    }
    if (validation_attention_blackout_width < 0 ||
        validation_attention_blackout_width >= T) {
        fprintf(stderr, "-vb must be in [0,T)\n");
        exit(EXIT_FAILURE);
    }
    if (validation_attention_disabled < 0 || validation_attention_disabled > 1) {
        fprintf(stderr, "-vd must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }
    if (validation_attention_disabled != 0 &&
        validation_attention_blackout_width != 0) {
        fprintf(stderr, "-vd 1 and positive -vb are separate ablations\n");
        exit(EXIT_FAILURE);
    }
    if (validation_loss_ignore_prefix < 0 ||
        validation_loss_ignore_prefix +
                (mask_sequence_final_target ? 1 : 0) >=
            T) {
        fprintf(stderr, "-vp must leave at least one supervised validation target per row\n");
        exit(EXIT_FAILURE);
    }
    if (validation_print_batch_losses < 0 || validation_print_batch_losses > 1) {
        fprintf(stderr, "-vl must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }
    #ifndef ENABLE_CUDNN
    if (validation_attention_blackout_width != 0 ||
        validation_attention_disabled != 0) {
        fprintf(stderr, "-vb and -vd require an ENABLE_CUDNN build\n");
        exit(EXIT_FAILURE);
    }
    #endif
    if (!llmc_normuon_validate_config(
            &optimizer_config,
            optimizer_config_error,
            sizeof(optimizer_config_error))) {
        fprintf(stderr, "Invalid optimizer configuration: %s\n", optimizer_config_error);
        exit(EXIT_FAILURE);
    }
    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    common_start(override_enable_tf32, false); // common init code for train/test/profile

    // should do a bit more error checking here
    assert(warmup_iterations >= 0);
    if (output_log_dir != NULL) {
        assert(strlen(output_log_dir) < 400); // careful bunch of hardcoded snprintf around this
    }
    int tokens_per_fwdbwd = B * T * multi_gpu_config.num_processes; // one micro-batch processes this many tokens
    int supervised_targets_per_fwdbwd =
        B * (T - (mask_sequence_final_target ? 1 : 0)) *
        multi_gpu_config.num_processes;
    int validation_supervised_targets_per_batch =
        B * (T - (mask_sequence_final_target ? 1 : 0) -
             validation_loss_ignore_prefix) *
        multi_gpu_config.num_processes;
    // calculate sensible default for total batch size as assuming no gradient accumulation
    if (total_batch_size == -1) { total_batch_size = tokens_per_fwdbwd; }
    // in the future, we might want to set gelu fusion to 2 for SM90+ and 0 for other GPUs
    if (gelu_fusion == -1) { gelu_fusion = 0; } // (deviceProp.major >= 9) ? 2 : 0; } // in gpt2_init_common for test_gpt2cu...
    // calculate the number of gradient accumulation steps from the desired total batch size
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    if (normuon_batch_replay_every > 0 && grad_accum_steps != 1) {
        fprintf(stderr, "same-batch Wdown LR replay currently requires grad_accum_steps=1\n");
        exit(EXIT_FAILURE);
    }
    // if we're only overfitting a single batch for debugging, let's overfit the first batch
    // from val instead of train split, because val is smaller and faster. (train_gpt2.py does the same)
    if (overfit_single_batch == 1) { train_data_pattern = val_data_pattern; }
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| Parameter             | Value                                              |\n");
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| train data pattern    | %-50s |\n", train_data_pattern);
    printf0("| val data pattern      | %-50s |\n", val_data_pattern);
    printf0("| output log dir        | %-50s |\n", output_log_dir == NULL ? "NULL" : output_log_dir);
    printf0("| checkpoint_every      | %-50d |\n", checkpoint_every);
    printf0("| resume                | %-50d |\n", resume);
    printf0("| reset resume loader   | %-50d |\n", resume_reset_dataloader);
    printf0("| fork NorMuon state    | %-50d |\n", resume_fork_normuon);
    printf0("| micro batch size B    | %-50d |\n", B);
    printf0("| sequence length T     | %-50d |\n", T);
    printf0("| sequence boundary     | %-50s |\n",
            llmc_sequence_boundary_policy_name(sequence_boundary_policy));
    printf0("| supervised targets    | %-50d |\n",
            supervised_targets_per_fwdbwd);
    printf0("| val attention blackout | %-49d |\n",
            validation_attention_blackout_width);
    printf0("| val attention disabled | %-49d |\n",
            validation_attention_disabled);
    printf0("| val loss ignore prefix | %-49d |\n",
            validation_loss_ignore_prefix);
    printf0("| val supervised targets | %-49d |\n",
            validation_supervised_targets_per_batch);
    printf0("| val batch loss logging | %-49d |\n",
            validation_print_batch_losses);
    printf0("| total batch size      | %-50d |\n", total_batch_size);
    printf0("| LR scheduler          | %-50s |\n", lr_scheduler_type);
    printf0("| learning rate (LR)    | %-50e |\n", learning_rate);
    printf0("| warmup iterations     | %-50d |\n", warmup_iterations);
    printf0("| final LR fraction     | %-50e |\n", final_learning_rate_frac);
    printf0("| weight decay          | %-50e |\n", weight_decay);
    printf0("| optimizer             | %-50s |\n", llmc_optimizer_selection_name(optimizer_config.optimizer_selection));
    printf0("| NorMuon families      | %-50s |\n", "mlp_wup,mlp_wdown");
    printf0("| NorMuon LR            | %-50e |\n", optimizer_config.learning_rate);
    printf0("| NorMuon weight decay  | %-50e |\n", optimizer_config.weight_decay);
    printf0("| NorMuon momentum      | %-50e |\n", optimizer_config.momentum);
    printf0("| NorMuon beta2         | %-50e |\n", optimizer_config.beta2);
    printf0("| NorMuon epsilon       | %-50e |\n", optimizer_config.epsilon);
    printf0("| NorMuon update scale  | %-50e |\n", optimizer_config.update_scale);
    printf0("| NorMuon Wdown LR mult. | %-49e |\n", optimizer_config.wdown_learning_rate_multiplier);
    printf0("| NorMuon execution     | %-50s |\n",
            llmc_normuon_execution_mode_name(optimizer_config.execution_mode));
    printf0("| NorMuon ortho mode    | %-50s |\n", llmc_normuon_orthogonalization_mode_name(optimizer_config.orthogonalization_mode));
    printf0("| NorMuon refresh       | %-50s |\n", llmc_normuon_approximation_policy_name(optimizer_config.refresh_policy));
    printf0("| NorMuon correction    | %-50s |\n", llmc_normuon_approximation_policy_name(optimizer_config.correction_policy));
    printf0("| Tracker refresh mode  | %-50s |\n",
            llmc_normuon_tracker_refresh_mode_name(
                optimizer_config.tracker_refresh_mode));
    printf0("| NorMuon refresh int.  | %-50u |\n", optimizer_config.refresh_interval);
    printf0("| Tracker max age       | %-50u |\n", optimizer_config.tracker_max_refresh_age);
    printf0("| Tracker Wup skew gate | %-50e |\n", optimizer_config.tracker_wup_skew_threshold);
    printf0("| Tracker Wdn skew gate | %-50e |\n", optimizer_config.tracker_wdown_skew_threshold);
    printf0("| NorMuon correction N  | %-50u |\n", optimizer_config.correction_iterations);
    printf0("| NorMuon corr. gain    | %-50e |\n", optimizer_config.correction_gain);
    printf0("| CacheMuon gamma       | %-50e |\n", optimizer_config.cache_residual_threshold);
    printf0("| NorMuon corr. mode    | %-50s |\n",
            llmc_normuon_tracker_correction_mode_name(
                optimizer_config.correction_mode));
    printf0("| NorMuon retraction    | %-50s |\n",
            llmc_normuon_tracker_retraction_mode_name(optimizer_config.retraction_mode));
    printf0("| Tracker diagnostics N | %-50d |\n",
            normuon_tracker_diagnostics_every);
    printf0("| Tracker LR dither amp | %-50e |\n",
            normuon_lr_dither_amplitude);
    printf0("| Tracker LR dither mode| %-50s |\n",
            llmc_normuon_lr_dither_mode_name(normuon_lr_dither_mode));
    printf0("| Tracker LR dither int | %-50d |\n",
            normuon_lr_dither_interval);
    printf0("| Tracker LR sine Wup T | %-50d |\n",
            normuon_lr_dither_wup_period);
    printf0("| Tracker LR sine Wdn T | %-50d |\n",
            normuon_lr_dither_wdown_period);
    printf0("| Tracker LR chop env T | %-50d |\n",
            normuon_lr_dither_envelope_blocks);
    printf0("| Tracker LR sine phase | %-50e |\n",
            normuon_lr_dither_phase_polarity);
    printf0("| Tracker LR Wup scale  | %-50e |\n",
            normuon_lr_dither_wup_scale);
    printf0("| Tracker LR Wdn scale  | %-50e |\n",
            normuon_lr_dither_wdown_scale);
    printf0("| Wdown batch replay N  | %-50d |\n",
            normuon_batch_replay_every);
    printf0("| Wdown replay halfspan | %-50e |\n",
            normuon_batch_replay_amplitude);
    printf0("| skip update lossz     | %-50f |\n", skip_update_lossz);
    printf0("| skip update gradz     | %-50f |\n", skip_update_gradz);
    printf0("| max_steps             | %-50d |\n", max_steps);
    printf0("| val_loss_every        | %-50d |\n", val_loss_every);
    printf0("| val_max_steps         | %-50d |\n", val_max_steps);
    printf0("| sample_every          | %-50d |\n", sample_every);
    printf0("| genT                  | %-50d |\n", genT);
    printf0("| sample RNG seed       | %-50llu |\n", sample_rng_seed);
    printf0("| sample temperature    | %-50f |\n", sample_temperature);
    printf0("| sample top-k          | %-50d |\n", sample_top_k);
    printf0("| sample top-p          | %-50f |\n", sample_top_p);
    printf0("| sample prompt ids     | %-50s |\n",
            sample_prompt_token_ids_csv == nullptr ? "GPT-2 EOS" : sample_prompt_token_ids_csv);
    printf0("| overfit_single_batch  | %-50d |\n", overfit_single_batch);
    printf0("| use_master_weights    | %-50s |\n", use_master_weights ? "enabled" : "disabled");
    printf0("| gelu_fusion           | %-50d |\n", gelu_fusion);
    printf0("| recompute             | %-50d |\n", recompute);
    printf0("+-----------------------+----------------------------------------------------+\n");
    const char* precision_str = (PRECISION_MODE == PRECISION_FP32)
                              ? (cublas_compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : "FP32")
                              : (PRECISION_MODE == PRECISION_FP16 ? "FP16" : "BF16");
    printf0("| device                | %-50s |\n", deviceProp.name);
    printf0("| peak TFlops           | %-50.1f |\n", get_flops_promised(deviceProp.name, PRECISION_MODE));
    printf0("| precision             | %-50s |\n", precision_str);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // figure out if we are going to be resuming the optimization
    int resuming = 0;
    // find the DONE file with the highest step count
    int resume_max_step = find_max_step(output_log_dir);
    if (resume == 1) { // is -y 1 resume flag set?
        assert(output_log_dir != NULL);
        if (resume_max_step != -1) {
            resuming = 1; // -y 1 is set, and we found a checkpoint we can resume from
            snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, resume_max_step);
        }
    }

    // build the GPT-2 model
    GPT2 model;
    gpt2_init_common(&model);
    if (resuming == 1) {
        // if `-y 1` was set, then we are resuming from the latest checkpoint
        // if we are using master weights, we'll init them later inside load_state()
        bool weight_init = !use_master_weights;
        gpt2_build_from_checkpoint(&model, filename_buffer, weight_init);
    } else if (ends_with_bin(load_filename)) {
        // otherwise, if this is a .bin file, we assume it's a model, let's init from it
        gpt2_build_from_checkpoint(&model, load_filename);
    } else {
        // if it's not .bin, it could be a "special descriptor". This descriptor is used to
        // construct GPT-2 / GPT-3 models in a convenient format. See the function for docs.
        gpt_build_from_descriptor(&model, load_filename);
    }

    model.optimizer_config = optimizer_config;
    bool resume_has_normuon_companion = false;
    char resume_normuon_path[512] = {0};
    if (resuming == 1) {
        LlmcNormuonCompanionInfo companion_info;
        llmc_normuon_companion_path(
            resume_normuon_path,
            sizeof(resume_normuon_path),
            output_log_dir,
            resume_max_step,
            multi_gpu_config.process_rank);
        resume_has_normuon_companion =
            llmc_normuon_companion_exists(resume_normuon_path);
        if (resume_has_normuon_companion) {
            if (!llmc_normuon_read_companion_info(
                    resume_normuon_path, &companion_info) ||
                companion_info.step != resume_max_step ||
                companion_info.num_processes != multi_gpu_config.num_processes ||
                companion_info.process_rank != multi_gpu_config.process_rank ||
                companion_info.num_layers != model.config.num_layers ||
                companion_info.channels != model.config.channels) {
                fprintf(stderr, "Invalid or incompatible NorMuon companion state: %s\n", resume_normuon_path);
                exit(EXIT_FAILURE);
            }
            if (resume_fork_normuon != 0) {
                // The ordinary state file owns FP32 master weights, momentum,
                // second moment, and the exact dataloader cursor.  A declared
                // fork keeps those shared states but intentionally starts the
                // target mode's Q/cache transform invalid, so its first update
                // performs a fresh polar solve under the new configuration.
                resume_has_normuon_companion = false;
                printf0("Forking NorMuon configuration at step %d; preserving main optimizer/dataloader state and reinitializing polar/cache state.\n",
                        resume_max_step);
            } else if (optimizer_cli_explicit &&
                       !llmc_normuon_config_equal(&optimizer_config, &companion_info.config)) {
                fprintf(stderr, "Explicit optimizer CLI configuration does not match the checkpoint companion state\n");
                exit(EXIT_FAILURE);
            } else {
                model.optimizer_config = companion_info.config;
                optimizer_config = companion_info.config;
            }
        } else if (resume_fork_normuon == 0 &&
                   optimizer_config.optimizer_selection ==
                   LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON) {
            fprintf(stderr, "Cannot exactly resume NorMuon without its companion optimizer state file\n");
            exit(EXIT_FAILURE);
        }
    }
    model.use_master_weights = use_master_weights;
    model.gelu_fusion = gelu_fusion;
    model.recompute = recompute;
    printf0("| weight init method    | %-50s |\n", resuming == 1 ? "intermediate checkpoint" : load_filename);
    printf0("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf0("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf0("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf0("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf0("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf0("| channels C            | %-50d |\n", model.config.channels);
    printf0("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // build DataLoaders for both train and val
    int permute_train_loader = (overfit_single_batch == 1) ? 0 : 1;
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, permute_train_loader);
    dataloader_init(&val_loader, val_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 0);
    printf0("| train_data_format     | %-50s |\n", dataloader_token_format_name(train_loader.token_format));
    printf0("| val_data_format       | %-50s |\n", dataloader_token_format_name(val_loader.token_format));
    // figure out the number of training steps we will run for
    int train_num_batches = max_steps; // passed in from command line
    if (train_num_batches == -1) {
        // sensible default is to train for exactly one epoch
        size_t ntok = train_loader.num_tokens;
        // the number of (outer loop) steps each process should take for us to reach one epoch
        train_num_batches = ntok / total_batch_size;
    }
    // figure out the number of validation steps to run for
    int val_num_batches = val_max_steps; // passed in from command line
    if (val_num_batches == -1) {
        // sensible default is to evaluate the full validation split
        size_t ntok = val_loader.num_tokens;
        // note that unlike the training loop, there is no gradient accumulation inner loop here
        val_num_batches = ntok / tokens_per_fwdbwd;
    }
    printf0("| train_num_batches     | %-50d |\n", train_num_batches);
    printf0("| val_num_batches       | %-50d |\n", val_num_batches);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // build an EvalLoader for HellaSwag
    EvalLoader eval_loader;
    const char* hellaswag_path = "dev/data/hellaswag/hellaswag_val.bin";
    const bool hellaswag_available = access(hellaswag_path, F_OK) == 0;
    const bool run_hellaswag = hellaswag_eval && hellaswag_available;
    if (run_hellaswag) {
        evalloader_init(&eval_loader, hellaswag_path, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes);
    }
    printf0("| run hellaswag         | %-50s |\n", run_hellaswag ? "yes" : "no");
    printf0("+-----------------------+----------------------------------------------------+\n");

    // pretty print in a table the multi-gpu configuration as well
    set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);
    if (model.optimizer_config.optimizer_selection ==
        LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON) {
        if (multi_gpu_config.num_processes != 1) {
            fprintf(stderr, "llm.c NorMuon currently supports single-GPU execution only\n");
            exit(EXIT_FAILURE);
        }
        if (multi_gpu_config.zero_stage != 0) {
            fprintf(stderr, "llm.c NorMuon rejects zero_stage != 0 because flat ZeRO shards can split a square polar view\n");
            exit(EXIT_FAILURE);
        }
        if (PRECISION_MODE != PRECISION_BF16) {
            fprintf(stderr, "llm.c NorMuon currently requires BF16 model parameters and gradients\n");
            exit(EXIT_FAILURE);
        }
        if (!model.use_master_weights) {
            fprintf(stderr, "llm.c NorMuon requires FP32 master weights (-w 1)\n");
            exit(EXIT_FAILURE);
        }
    }
    char optimizer_plan_error[256];
    if (!llmc_build_optimizer_plan(
            &model.optimizer_plan,
            &model.optimizer_config,
            model.config.num_layers,
            model.config.channels,
            model.param_elements,
            optimizer_plan_error,
            sizeof(optimizer_plan_error))) {
        fprintf(stderr, "Failed to build optimizer plan: %s\n", optimizer_plan_error);
        exit(EXIT_FAILURE);
    }
    printf0("effective_optimizer: %s\n",
            llmc_optimizer_selection_name(model.optimizer_config.optimizer_selection));
    for (int parameter_index = 0;
         parameter_index < LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT;
         ++parameter_index) {
        const LlmcOptimizerParameterType* parameter_type =
            &model.optimizer_plan.parameter_types[parameter_index];
        printf0("optimizer_plan tensor=%d type=%s family=%s backend=%s hyperparameter_group=%d weight_decay=%s layers=%d views_per_layer=%d\n",
                parameter_type->tensor_id,
                parameter_type->name,
                model.optimizer_plan.families[parameter_type->family_id].name,
                model.optimizer_plan.backends[parameter_type->backend_kind].name,
                parameter_type->hyperparameter_group,
                parameter_type->weight_decay_policy == LLMC_WEIGHT_DECAY_ENABLED ? "enabled" : "disabled",
                parameter_type->layer_multiplicity,
                parameter_type->views_per_layer);
    }
    if (model.optimizer_plan.normuon_parameter_type_count != 0) {
        const bool fresh_gns_scratch =
            llmc_normuon_uses_fresh_gns_scratch(&model.optimizer_config);
        const bool commuted_canonical_stage2 =
            model.optimizer_config.retraction_mode ==
            LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
        const bool thin_canonical_stage2 =
            model.optimizer_config.retraction_mode ==
            LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2;
        const bool retraction_enabled =
            model.optimizer_config.retraction_mode !=
            LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
        printf0(
            "normuon_variant: %s\n",
            model.optimizer_config.orthogonalization_mode ==
                    LLMC_NORMUON_ORTHO_SPLIT_WUP_SQUARE_TRACKER_WDOWN_RECTANGULAR_MUON
                ? "split Wup square tracker + Wdown rectangular scratch Muon"
                : model.optimizer_config.orthogonalization_mode ==
                    LLMC_NORMUON_ORTHO_RECTANGULAR_MUON
                ? (fresh_gns_scratch
                       ? "proper rectangular Muon (scratch FreshGNS every step)"
                       : "proper rectangular Muon (scratch-only)")
                : model.optimizer_config.orthogonalization_mode ==
                          LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON
                      ? "proper rectangular CacheMuon (residual-gated FreshGNS)"
                : model.optimizer_config.orthogonalization_mode ==
                          LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q
                      ? "proper rectangular polar-factor tracker"
                      : "blockwise square-view NorMuon");
        printf0("normuon_view_count: %d\n", model.optimizer_plan.normuon_view_count);
        printf0("normuon_execution_mode: %s\n",
                llmc_normuon_execution_mode_name(model.optimizer_config.execution_mode));
        printf0("normuon_dense_operand_dtype: %s\n",
                model.optimizer_config.execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED ? "bf16" : "fp32");
        printf0("normuon_dense_accumulation_dtype: fp32\n");
        printf0("normuon_persistent_state_dtype: fp32\n");
        printf0("normuon_view_dispatch: %s\n",
                model.optimizer_config.execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED
                    ? "batched_per_parameter_family"
                    : "serial_per_view");
        printf0("normuon_tracker_packed_q_policy: %s\n",
                model.optimizer_config.execution_mode == LLMC_NORMUON_EXECUTION_BF16_BATCHED &&
                        llmc_normuon_is_tracker_mode(
                            model.optimizer_config.orthogonalization_mode)
                    ? "single_pack_reused_through_prefix_polynomial"
                    : "not_applicable");
        printf0("normuon_tracker_retraction_form: %s\n",
                (model.optimizer_config.orthogonalization_mode ==
                         LLMC_NORMUON_ORTHO_RECTANGULAR_MUON ||
                 model.optimizer_config.orthogonalization_mode ==
                         LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON)
                    ? "not_applicable_non_tracker"
                    : (!retraction_enabled
                    ? "disabled"
                    : (commuted_canonical_stage2
                           ? "canonical_taylor_stage2_post_product"
                           : (thin_canonical_stage2
                                  ? "canonical_taylor_stage2_thin_factor"
                                  : (model.optimizer_config.execution_mode ==
                                      LLMC_NORMUON_EXECUTION_BF16_BATCHED
                                      ? "D(3I-D^T D)/2"
                                      : "(3I-DD^T)D/2")))));
        printf0("normuon_tracker_retraction_mode: %s\n",
                llmc_normuon_tracker_retraction_mode_name(
                    model.optimizer_config.retraction_mode));
        printf0("normuon_orthogonalization_mode: %s\n",
                llmc_normuon_orthogonalization_mode_name(model.optimizer_config.orthogonalization_mode));
        printf0("normuon_refresh_policy: %s\n",
                llmc_normuon_approximation_policy_name(model.optimizer_config.refresh_policy));
        printf0("normuon_fresh_gns_scratch_active: %u\n",
                fresh_gns_scratch ? 1U : 0U);
        printf0("normuon_refresh_solver: %s\n",
                fresh_gns_scratch
                    ? "fresh_gns_gram_restart2_every_step"
                    : (llmc_normuon_is_cache_mode(
                           model.optimizer_config.orthogonalization_mode)
                           ? "fresh_gns_gram_restart2_residual_gated"
                           : "direct_matrix_polynomial"));
        printf0("normuon_correction_policy: %s\n",
                llmc_normuon_approximation_policy_name(model.optimizer_config.correction_policy));
        printf0("normuon_tracker_refresh_mode: %s\n",
                llmc_normuon_tracker_refresh_mode_name(
                    model.optimizer_config.tracker_refresh_mode));
        printf0("normuon_refresh_interval: %u\n", model.optimizer_config.refresh_interval);
        printf0("normuon_tracker_refresh_interval_semantics: %s\n",
                model.optimizer_config.tracker_refresh_mode ==
                        LLMC_NORMUON_TRACKER_REFRESH_ADAPTIVE_MEAN_SKEW
                    ? "fixed_cadence_parameter_inert_adaptive_checks_every_stale_step"
                    : "fixed_cadence");
        printf0("normuon_tracker_adaptive_check_interval: %u\n",
                model.optimizer_config.tracker_refresh_mode ==
                        LLMC_NORMUON_TRACKER_REFRESH_ADAPTIVE_MEAN_SKEW
                    ? 1U
                    : 0U);
        printf0("normuon_tracker_max_refresh_age: %u\n",
                model.optimizer_config.tracker_max_refresh_age);
        printf0("normuon_tracker_wup_skew_threshold: %.9g\n",
                model.optimizer_config.tracker_wup_skew_threshold);
        printf0("normuon_tracker_wdown_skew_threshold: %.9g\n",
                model.optimizer_config.tracker_wdown_skew_threshold);
        printf0("normuon_tracker_refresh_gate_scope: %s\n",
                model.optimizer_config.tracker_refresh_mode ==
                        LLMC_NORMUON_TRACKER_REFRESH_ADAPTIVE_MEAN_SKEW
                    ? "family_mean_skew_ratio_all_views_refresh_together"
                    : "not_applicable");
        printf0("normuon_correction_iterations: %u\n", model.optimizer_config.correction_iterations);
        printf0("normuon_correction_gain: %.9g\n", model.optimizer_config.correction_gain);
        printf0("normuon_cache_residual_threshold: %.9g\n",
                model.optimizer_config.cache_residual_threshold);
        printf0("normuon_cache_normalization_epsilon: %.9g\n",
                LLMC_CACHEMUON_EPSILON);
        printf0("normuon_cache_restart_stage: %u\n",
                LLMC_CACHEMUON_RESTART_STAGE);
        printf0("normuon_cache_gate_scope: %s\n",
                llmc_normuon_is_cache_mode(
                    model.optimizer_config.orthogonalization_mode)
                    ? "per_matrix_normalized_polar_residual"
                    : "not_applicable");
        printf0("normuon_tracker_correction_mode: %s\n",
                llmc_normuon_tracker_correction_mode_name(
                    model.optimizer_config.correction_mode));
        if (model.optimizer_config.correction_mode ==
            LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
            printf0("normuon_tracker_damping_eta: %.9g\n",
                    LLMC_NORMUON_TRACKER_DAMPING_ETA);
            printf0("normuon_tracker_correction_cap: %.9g\n",
                    LLMC_NORMUON_TRACKER_CORRECTION_CAP);
        }
        printf0("normuon_retraction: %u\n", retraction_enabled ? 1U : 0U);
        printf0("normuon_tracker_pre_product_correction_stage_count: %u\n",
                (commuted_canonical_stage2 || thin_canonical_stage2)
                    ? 1U
                    : model.optimizer_config.correction_iterations);
        printf0("normuon_tracker_post_product_correction_stage_count: %u\n",
                (commuted_canonical_stage2 || thin_canonical_stage2) ? 1U : 0U);
        printf0("normuon_optimizer_companion_format_version: %u\n",
                LLMC_NORMUON_COMPANION_VERSION);
        for (int stage = 0; stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT; ++stage) {
            const LlmcNormuonPolynomialStep coefficient =
                model.optimizer_config.refresh_schedule[stage];
            printf0("normuon_refresh_coefficient_%d: %.9g,%.9g,%.9g\n",
                    stage, coefficient.a, coefficient.b, coefficient.c);
        }
        for (int stage = 0; stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT; ++stage) {
            const LlmcNormuonPolynomialStep coefficient =
                model.optimizer_config.correction_schedule[stage];
            const char* placement = "";
            if ((commuted_canonical_stage2 || thin_canonical_stage2) && stage == 0) {
                placement = " (effective_pre_product)";
            } else if (commuted_canonical_stage2 && stage == 1) {
                placement = " (effective_post_product_retraction)";
            } else if (thin_canonical_stage2 && stage == 1) {
                placement = " (effective_thin_factor_retraction)";
            } else if (!commuted_canonical_stage2 && !thin_canonical_stage2 &&
                       stage < static_cast<int>(
                                   model.optimizer_config.correction_iterations)) {
                placement = " (effective_pre_product)";
            }
            printf0("normuon_correction_coefficient_%d: %.9g,%.9g,%.9g%s\n",
                    stage,
                    coefficient.a,
                    coefficient.b,
                    coefficient.c,
                    placement);
        }
    }
    printf0("| num_processes         | %-50d |\n", multi_gpu_config.num_processes);
    printf0("| zero_stage            | %-50d |\n", multi_gpu_config.zero_stage);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // prints outside of pretty table to here and below
    if (!hellaswag_available) {
        printf0("HellaSwag eval not found at %s, skipping its evaluation\n", hellaswag_path);
        printf0("You can run `python dev/data/hellaswag.py` to export and use it with `-h 1`.\n");
    }
    // more prints related to allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf0("num_parameters: %zu => bytes: %zu\n", model.num_parameters, model.num_parameters_bytes);
    printf0("allocated %d MiB for model parameters\n", (int)round(model.num_parameters_bytes / (1024 * 1024)));
    // few more prints for gradient accumulation math up above
    printf0("batch_size B=%d * seq_len T=%d * num_processes=%d and total_batch_size=%d\n",
            B, T, multi_gpu_config.num_processes, total_batch_size);
    printf0("supervised targets per forward/backward: %d (%d per sequence row)\n",
            supervised_targets_per_fwdbwd,
            T - (mask_sequence_final_target ? 1 : 0));
    printf0("=> setting grad_accum_steps=%d\n", grad_accum_steps);

    // set up logging
    if (multi_gpu_config.process_rank == 0) { create_dir_if_not_exists(output_log_dir); }
    Logger logger;
    logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resume);

    // set up the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // set up learning rate scheduler
    LearningRateScheduler lr_scheduler;
    lr_scheduler_init(&lr_scheduler, lr_scheduler_type, learning_rate,
                      warmup_iterations, train_num_batches, final_learning_rate_frac);
    LearningRateScheduler normuon_lr_scheduler;
    lr_scheduler_init(&normuon_lr_scheduler, lr_scheduler_type, model.optimizer_config.learning_rate,
                      warmup_iterations, train_num_batches, final_learning_rate_frac);

    // some memory for generating samples from the model
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));
    LlmcSamplingCandidate* sample_candidates = (LlmcSamplingCandidate*)mallocCheck(
        model.config.vocab_size * sizeof(LlmcSamplingCandidate));

    // if we found a checkpoint to resume from, load the optimization state
    int step = 0;
    gpt2_allocate_state(&model, B, T);
    if ((normuon_tracker_diagnostics_every > 0 ||
         normuon_lr_dither_amplitude > 0.0f) &&
        !llmc_normuon_enable_tracker_diagnostics(
            &model.normuon_runtime,
            static_cast<uint32_t>(normuon_tracker_diagnostics_every))) {
        fprintf(stderr, "Failed to allocate square-tracker diagnostics\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_batch_replay_every > 0 &&
        normuon_tracker_diagnostics_every > 0 &&
        !llmc_normuon_enable_tracker_h_stability(
            &model.normuon_runtime,
            LLMC_OPTIMIZER_FAMILY_MLP_WDOWN)) {
        fprintf(stderr, "Failed to allocate Wdown H-stability telemetry\n");
        exit(EXIT_FAILURE);
    }
    if (normuon_lr_dither_amplitude > 0.0f &&
        !llmc_normuon_enable_lr_dither_probe(
            &model.normuon_runtime,
            normuon_lr_dither_amplitude,
            static_cast<uint32_t>(normuon_lr_dither_interval),
            normuon_lr_dither_mode,
            static_cast<uint32_t>(normuon_lr_dither_wup_period),
            static_cast<uint32_t>(normuon_lr_dither_wdown_period),
            normuon_lr_dither_phase_polarity,
            normuon_lr_dither_wup_scale,
            normuon_lr_dither_wdown_scale,
            static_cast<uint32_t>(normuon_lr_dither_envelope_blocks))) {
        fprintf(stderr, "Failed to allocate square-tracker LR dither probe\n");
        exit(EXIT_FAILURE);
    }
    if (model.optimizer_plan.normuon_parameter_type_count != 0) {
        printf0("normuon_workspace_bytes: %zu\n", model.normuon_runtime.workspace_bytes);
        printf0("normuon_workspace_source: %s\n",
                model.normuon_runtime.workspace_is_borrowed
                    ? "post_gradient_norm_output_arena"
                    : "dedicated_allocation");
        printf0("normuon_workspace_float_matrix_panels: %zu\n",
                model.normuon_runtime.batch_float_matrix_count);
        printf0("normuon_tracker_h_stability_active: %d\n",
                model.normuon_runtime.tracker_h_stability_normalized != nullptr &&
                    model.normuon_runtime.tracker_h_stability_previous != nullptr);
        printf0("normuon_tracker_h_stability_bytes: %zu\n",
                model.normuon_runtime.tracker_h_stability_elements *
                    sizeof(float) * 2U);
        printf0("normuon_batch_matrix_capacity: %zu\n", model.normuon_runtime.batch_matrix_capacity);
        printf0("normuon_tracker_q_bytes: %zu\n", model.normuon_runtime.tracked_q_bytes);
    }
    if (resuming == 1) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, resume_max_step, multi_gpu_config.process_rank);
        load_state(&step, &model, &train_loader, filename_buffer, resume_reset_dataloader != 0);
        if (resume_has_normuon_companion &&
            !llmc_normuon_load_companion(
                resume_normuon_path,
                step,
                multi_gpu_config.num_processes,
                multi_gpu_config.process_rank,
                &model.optimizer_plan,
                &model.optimizer_config,
                &model.normuon_runtime,
                main_stream)) {
            fprintf(stderr, "Failed to load exact NorMuon companion state: %s\n", resume_normuon_path);
            exit(EXIT_FAILURE);
        }
    }

    float* normuon_batch_replay_wdown_snapshot = nullptr;
    floatX* normuon_batch_replay_center_parameter = nullptr;
    float* normuon_batch_replay_row_scales = nullptr;
    unsigned long long* normuon_batch_replay_changed_count = nullptr;
    size_t normuon_batch_replay_wdown_elements = 0U;
    size_t normuon_batch_replay_row_scale_elements = 0U;
    if (normuon_batch_replay_every > 0) {
        const LlmcOptimizerParameterType* wdown_parameter_type =
            gpt2_normuon_parameter_type_for_family(
                &model, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
        if (wdown_parameter_type == nullptr) {
            fprintf(stderr, "same-batch replay could not locate the NorMuon Wdown family\n");
            exit(EXIT_FAILURE);
        }
        normuon_batch_replay_wdown_elements = wdown_parameter_type->tensor_elements;
        normuon_batch_replay_row_scale_elements = static_cast<size_t>(
            wdown_parameter_type->layer_multiplicity *
            wdown_parameter_type->views_per_layer) *
            wdown_parameter_type->matrix_width;
        cudaCheck(cudaMalloc(
            &normuon_batch_replay_wdown_snapshot,
            normuon_batch_replay_wdown_elements * sizeof(float)));
        cudaCheck(cudaMalloc(
            &normuon_batch_replay_center_parameter,
            normuon_batch_replay_wdown_elements * sizeof(floatX)));
        cudaCheck(cudaMalloc(
            &normuon_batch_replay_row_scales,
            normuon_batch_replay_row_scale_elements * sizeof(float)));
        cudaCheck(cudaMalloc(
            &normuon_batch_replay_changed_count,
            sizeof(unsigned long long)));
        printf0(
            "normuon_batch_replay_snapshot_bytes: %zu\n",
            normuon_batch_replay_wdown_elements *
                (sizeof(float) + sizeof(floatX)) +
                normuon_batch_replay_row_scale_elements * sizeof(float) +
                sizeof(unsigned long long));
    }

    // init an OutlierDetector the training loss
    OutlierDetector loss_outlier_detector, grad_norm_outlier_detector;
    init_detector(&loss_outlier_detector);
    init_detector(&grad_norm_outlier_detector);

    // do some checks here before we kick off training
    // cross-check the desired sequence length T with the model's max sequence length
    if (T < model.config.max_seq_len) {
        printf0("!!!!!!!!\n");
        printf0("WARNING:\n");
        printf0("- The training sequence length is: T=%d (set with -t)\n", T);
        printf0("- The model's max sequence length is: max_seq_len=%d\n", model.config.max_seq_len);
        printf0("You are attempting to train with a sequence length shorter than the model's max.\n");
        printf0("This will lead to unused parameters in the wpe position embedding weights.\n");
        printf0("If you know what you're doing you can ignore this warning.\n");
        printf0("If you're like ???, you are most likely misconfiguring your training run.\n");
        printf0("---> HINT: If you're training GPT-2 use -t 1024. If GPT-3, use -t 2048.\n");
        printf0("!!!!!!!!\n");
    }
    // in any case, this must be true or we'd index beyond the model's wpe (position embedding table)
    assert(T <= model.config.max_seq_len);

    // train
    cudaEvent_t start, end;
    cudaEvent_t optimizer_start, optimizer_end;
    cudaEvent_t batch_replay_start = nullptr, batch_replay_end = nullptr;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaEventCreate(&optimizer_start));
    cudaCheck(cudaEventCreate(&optimizer_end));
    if (normuon_batch_replay_every > 0) {
        cudaCheck(cudaEventCreate(&batch_replay_start));
        cudaCheck(cudaEventCreate(&batch_replay_end));
    }
    cudaCheck(cudaProfilerStart());
    double total_sum_iteration_time_s = 0.0;
    float ema_tokens_per_second = 0.0f;
    size_t peak_device_memory_used_bytes = 0U;
    double total_optimizer_time_ms = 0.0;
    int completed_optimizer_steps = 0;
    for (; step <= train_num_batches; step++) {
        NvtxRange step_range("Train step", step);

        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss (all processes collaborate)
        if (step % val_loss_every == 0 || last_step) {
            NvtxRange validation_range("validation");
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                float val_batch_loss = gpt2_validate(
                    &model,
                    val_loader.inputs,
                    val_loader.targets,
                    B,
                    T,
                    mask_sequence_final_target,
                    validation_attention_blackout_width,
                    validation_attention_disabled != 0,
                    validation_loss_ignore_prefix);
                val_loss += val_batch_loss;
                if (validation_print_batch_losses != 0) {
                    printf0(
                        "val batch %d/%d loss %.9f targets %d\n",
                        i + 1,
                        val_num_batches,
                        val_batch_loss,
                        validation_supervised_targets_per_batch);
                }
            }
            val_loss /= val_num_batches;
            val_loss = multi_gpu_cpu_float_sum(val_loss, &multi_gpu_config) / multi_gpu_config.num_processes;
            printf0("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while estimate HellaSwag accuracy (all processes collaborate)
        if (run_hellaswag &&
           ((step > 0 && step % val_loss_every == 0) || last_step)) {
            NvtxRange evaluation_range("evaluation");
            float eval_acc_norm = 0.0f;
            evalloader_reset(&eval_loader);
            for (int i = 0; i < eval_loader.num_batches; i++) {
                if (i % 10 == 0) { printf("evaluating HellaSwag: %d/%d\r", i, eval_loader.num_batches); }
                evalloader_next_batch(&eval_loader);
                gpt2_validate(&model, eval_loader.inputs, eval_loader.targets, B, T);
                int correct = evalloader_stat_losses(&eval_loader, model.cpu_losses);
                eval_acc_norm += (float)correct;
            }
            // careful because not all ranks may have the exact same allocation of number of examples
            eval_acc_norm = multi_gpu_cpu_float_sum(eval_acc_norm, &multi_gpu_config);
            printf0("HellaSwag: %d/%d = %f\n", (int)eval_acc_norm, eval_loader.num_examples, eval_acc_norm / eval_loader.num_examples);
            logger_log_eval(&logger, step, eval_acc_norm / eval_loader.num_examples);
        }

        // once in a while do model inference to print generated text (only rank 0)
        if (multi_gpu_config.process_rank == 0 && sample_every > 0 &&
           (step > 0 && (step % sample_every) == 0 || last_step)) {
            NvtxRange generation_range("generation");
            if (!(sample_temperature > 0.0f) || !isfinite(sample_temperature)) {
                fprintf(stderr, "sample temperature must be finite and greater than zero\n");
                exit(EXIT_FAILURE);
            }
            if (sample_top_k < 0) {
                fprintf(stderr, "sample top-k must be nonnegative\n");
                exit(EXIT_FAILURE);
            }
            if (!(sample_top_p > 0.0f) || sample_top_p > 1.0f || !isfinite(sample_top_p)) {
                fprintf(stderr, "sample top-p must be finite and in (0, 1]\n");
                exit(EXIT_FAILURE);
            }
            if (genT <= 1 || genT > T) {
                fprintf(stderr, "genT must be in [2, sequence_length]\n");
                exit(EXIT_FAILURE);
            }
            std::vector<int> sample_prompt_tokens;
            if (sample_prompt_token_ids_csv != nullptr) {
                const char* cursor = sample_prompt_token_ids_csv;
                while (*cursor != '\0') {
                    char* end = nullptr;
                    const long long token = strtoll(cursor, &end, 10);
                    if (end == cursor || token < 0 || token >= model.config.vocab_size) {
                        fprintf(stderr, "invalid sample prompt token id near: %s\n", cursor);
                        exit(EXIT_FAILURE);
                    }
                    sample_prompt_tokens.push_back((int)token);
                    cursor = end;
                    while (*cursor == ' ' || *cursor == '\t') {
                        ++cursor;
                    }
                    if (*cursor == '\0') {
                        break;
                    }
                    if (*cursor != ',') {
                        fprintf(stderr, "sample prompt token ids must be comma-separated\n");
                        exit(EXIT_FAILURE);
                    }
                    ++cursor;
                    while (*cursor == ' ' || *cursor == '\t') {
                        ++cursor;
                    }
                }
                if (sample_prompt_tokens.empty()) {
                    fprintf(stderr, "sample prompt token ids cannot be empty\n");
                    exit(EXIT_FAILURE);
                }
            }
            unsigned long long sample_rng_state = sample_rng_seed;
            // fill up gen_tokens with the <|endoftext|> token, which kicks off the generation
            int eot_token = tokenizer.eot_token;
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = eot_token;
            }
            const int prompt_length = sample_prompt_tokens.empty() ? 1 : (int)sample_prompt_tokens.size();
            if (prompt_length >= genT) {
                fprintf(stderr, "sample prompt must leave at least one generation position\n");
                exit(EXIT_FAILURE);
            }
            for (int i = 0; i < (int)sample_prompt_tokens.size(); ++i) {
                gen_tokens[i] = sample_prompt_tokens[i];
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            if (!sample_prompt_tokens.empty()) {
                for (int i = 0; i < prompt_length; ++i) {
                    if (tokenizer.init_ok) {
                        safe_printf(tokenizer_decode(&tokenizer, gen_tokens[i]));
                    } else {
                        printf("%d ", gen_tokens[i]);
                    }
                }
                fflush(stdout);
            }
            for (int t = prompt_length; t < genT; t++) {
                NvtxRange generation_range("Generation step", t);
                // we try not to be too wasteful for inference by not calculating all of B,T
                // Using a smaller B is always bit-for-bit identical, but T is more tricky
                // for non-CUDNN, we need to make sure the attention buffer is memset to 0
                // for cuDNN, it might suddenly decide to use a slightly different algorithm...
                // on cuDNN 9.2.1 with cuDNN FrontEnd 1.5.2, T >= 256 seems bit-for-bit identical
                // (but even if it wasn't fully identical that's probably not the end of the world)
                // note this is still somewhat wasteful because we don't have a KV cache!
                gpt2_forward(&model, gen_tokens, 1, CEIL_DIV(t, min(T,256)) * min(T,256));
                // get the V-dimensional vector probs[0, t-1, :]
                floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
                // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
                for (int i = 0; i < model.config.vocab_size; i++) {
                    cpu_logits[i] = (float)cpu_logits_raw[i] / sample_temperature;
                }
                // sample the next token
                float coin = random_f32(&sample_rng_state);
                int next_token = sample_softmax_top_k_top_p(
                    cpu_logits,
                    model.config.vocab_size,
                    sample_top_k,
                    sample_top_p,
                    coin,
                    sample_candidates);
                if (next_token < 0) {
                    fprintf(stderr, "sample filtering encountered invalid or nonfinite logits\n");
                    exit(EXIT_FAILURE);
                }
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // once in a while checkpoint the optimization state (all ranks)
        if ((checkpoint_every > 0 && output_log_dir != NULL && resuming == 0) &&
            ((step > 0 && step % checkpoint_every == 0) || last_step)) {
            // writes model .bin file, state .bin files, and DONE file for step
            write_checkpoint(output_log_dir, step, &model, &train_loader, &multi_gpu_config);
            // we only keep checkpoints_keep checkpoints on disk to save space
            // so now that we wrote a new checkpoint, delete one old one (unless it is a "major" checkpoint)
            // we only do this is checkpoint keeping is turned on (checkpoints_keep > 0)
            int step_delete = step - checkpoints_keep * checkpoint_every;
            if (checkpoints_keep > 0 && step_delete > 0 &&
               (major_checkpoint_every == 0 || step_delete % major_checkpoint_every != 0)
                ) {
                delete_checkpoint(output_log_dir, step_delete, &multi_gpu_config);
            }
        }
        resuming = 0;

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // --------------- TRAINING SECTION BEGIN -----------------
        if (overfit_single_batch == 1) {
            // if we are trying to overfit a single batch, we reset the loader here
            dataloader_reset(&train_loader);
        }
        // do one training step, doing forward/backward/update on total_batch_size tokens
        cudaCheck(cudaEventRecord(start));
        // gradient and loss accumulation loop over micro-batches
        for (int micro_step = 0; micro_step < grad_accum_steps; micro_step++) {
            // fetch the next data batch
            dataloader_next_batch(&train_loader);
            // forward pass. note that we pass in grad_accum_steps, which scales down the loss
            gpt2_forward(&model, train_loader.inputs, B, T);
            // backward pass. all model params accumulate gradients with += inside this inner loop
            gpt2_backward_and_reduce(
                &model,
                train_loader.inputs,
                train_loader.targets,
                grad_accum_steps,
                micro_step,
                mask_sequence_final_target);
        }
        float zloss = (float)(update_detector(&loss_outlier_detector, (double)model.mean_loss)); // loss z-score
        // fetch the next learning rate
        float step_learning_rate = get_learning_rate(&lr_scheduler, step);
        float step_normuon_learning_rate = get_learning_rate(&normuon_lr_scheduler, step);
        float optimizer_time_ms = 0.0f;
        bool batch_replay_probed = false;
        float batch_replay_base_loss = 0.0f;
        float batch_replay_plus_loss = 0.0f;
        float batch_replay_minus_loss = 0.0f;
        float batch_replay_time_ms = 0.0f;
        unsigned long long batch_replay_plus_changed = 0ULL;
        unsigned long long batch_replay_minus_changed = 0ULL;
        float batch_replay_q_sample_max = 0.0f;
        float batch_replay_row_scale_sample_max = 0.0f;
        float batch_replay_sample_update_abs = 0.0f;
        float batch_replay_sample_base_master = 0.0f;
        float batch_replay_sample_plus_master = 0.0f;
        float batch_replay_sample_center_parameter = 0.0f;
        float batch_replay_sample_plus_parameter = 0.0f;
        // calculate the gradient norm and how much we wish to scale the gradient
        float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
        float zgrad = (float)(update_detector(&grad_norm_outlier_detector, (double)grad_norm)); // grad z-score
        // update the model parameters
        if (isfinite(zloss) && skip_update_lossz != 0.0f && zloss > skip_update_lossz) {
            printf0("skipping update due to loss z-score of %f\n", zloss);
        } else if (isfinite(zgrad) && skip_update_gradz != 0.0f && zgrad > skip_update_gradz) {
            printf0("skipping update due to grad z-score of %f\n", zgrad);
        } else {
            // clip the gradient norm to a maximum value
            float grad_clip = 1.0f;
            float grad_scale = (grad_norm > grad_clip) ? grad_clip / grad_norm : 1.0f;
            cudaCheck(cudaEventRecord(optimizer_start));
            gpt2_update(
                &model, step_learning_rate, 0.9f, 0.95f, 1e-8f,
                weight_decay, grad_scale, step+1, &multi_gpu_config,
                step_normuon_learning_rate);
            cudaCheck(cudaEventRecord(optimizer_end));
            cudaCheck(cudaEventSynchronize(optimizer_end));
            cudaCheck(cudaEventElapsedTime(&optimizer_time_ms, optimizer_start, optimizer_end));
            total_optimizer_time_ms += optimizer_time_ms;
            completed_optimizer_steps++;
            if (normuon_batch_replay_every > 0 &&
                (step % normuon_batch_replay_every) == 0) {
                const uint64_t optimizer_global_step =
                    llmc_gpt2_normuon_global_step(step + 1);
                const uint64_t replay_rounding_step =
                    optimizer_global_step ^ 0xd1b54a32d192ed03ULL;
                const LlmcOptimizerParameterType* replay_parameter_type =
                    gpt2_normuon_parameter_type_for_family(
                        &model, LLMC_OPTIMIZER_FAMILY_MLP_WDOWN);
                if (replay_parameter_type == nullptr) {
                    fprintf(stderr, "failed to locate Wdown replay parameter metadata\n");
                    exit(EXIT_FAILURE);
                }
                const size_t replay_width = replay_parameter_type->matrix_width;
                const size_t replay_matrix_elements = replay_width * replay_width;
                const size_t replay_q_sample_count =
                    replay_matrix_elements < 1024U ? replay_matrix_elements : 1024U;
                const size_t replay_scale_sample_count =
                    replay_width < 1024U ? replay_width : 1024U;
                float replay_q_sample[1024];
                float replay_scale_sample[1024];
                const size_t replay_q_offset =
                    LLMC_NORMUON_VIEWS_PER_MLP_MATRIX * replay_matrix_elements;
                cudaCheck(cudaMemcpyAsync(
                    normuon_batch_replay_row_scales,
                    model.normuon_runtime.axis_stats,
                    normuon_batch_replay_row_scale_elements * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    main_stream));
                cudaCheck(cudaMemcpy(
                    replay_q_sample,
                    model.normuon_runtime.tracked_q + replay_q_offset,
                    replay_q_sample_count * sizeof(float),
                    cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(
                    replay_scale_sample,
                    normuon_batch_replay_row_scales,
                    replay_scale_sample_count * sizeof(float),
                    cudaMemcpyDeviceToHost));
                for (size_t sample_index = 0;
                     sample_index < replay_q_sample_count;
                     ++sample_index) {
                    batch_replay_q_sample_max = fmaxf(
                        batch_replay_q_sample_max,
                        fabsf(replay_q_sample[sample_index]));
                }
                for (size_t sample_index = 0;
                     sample_index < replay_scale_sample_count;
                     ++sample_index) {
                    batch_replay_row_scale_sample_max = fmaxf(
                        batch_replay_row_scale_sample_max,
                        fabsf(replay_scale_sample[sample_index]));
                }
                size_t replay_sample_index = 0U;
                for (size_t sample_index = 0;
                     sample_index < replay_q_sample_count;
                     ++sample_index) {
                    const size_t row = sample_index / replay_width;
                    const float update_abs = fabsf(
                        replay_q_sample[sample_index] *
                        replay_scale_sample[row]);
                    if (update_abs > batch_replay_sample_update_abs) {
                        batch_replay_sample_update_abs = update_abs;
                        replay_sample_index = sample_index;
                    }
                }
                const size_t replay_sample_row = replay_sample_index / replay_width;
                const size_t replay_sample_column =
                    replay_sample_index - replay_sample_row * replay_width;
                const size_t replay_sample_parameter_offset =
                    replay_sample_row * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX *
                        replay_width +
                    replay_sample_column;
                const ShardInfo replay_tensor = gpt2_get_tensor_at_layer(
                    &model, 0, replay_parameter_type->tensor_id);
                cudaCheck(cudaEventRecord(batch_replay_start));
                if (!gpt2_normuon_batch_replay_snapshot_wdown(
                        &model, normuon_batch_replay_wdown_snapshot)) {
                    fprintf(stderr, "failed to snapshot Wdown for same-batch replay\n");
                    exit(EXIT_FAILURE);
                }
                if (!gpt2_normuon_batch_replay_set_wdown(
                        &model,
                        normuon_batch_replay_wdown_snapshot,
                        step_normuon_learning_rate,
                        0.0f,
                        replay_rounding_step,
                        nullptr,
                        nullptr,
                        normuon_batch_replay_row_scales)) {
                    fprintf(stderr, "failed to apply common replay rounding at the center point\n");
                    exit(EXIT_FAILURE);
                }
                if (!gpt2_normuon_batch_replay_snapshot_wdown_parameter(
                        &model, normuon_batch_replay_center_parameter)) {
                    fprintf(stderr, "failed to snapshot the common-rounded replay center\n");
                    exit(EXIT_FAILURE);
                }
                floatX replay_center_parameter_value;
                cudaCheck(cudaMemcpy(
                    &batch_replay_sample_base_master,
                    normuon_batch_replay_wdown_snapshot +
                        replay_sample_parameter_offset,
                    sizeof(float),
                    cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(
                    &replay_center_parameter_value,
                    normuon_batch_replay_center_parameter +
                        replay_sample_parameter_offset,
                    sizeof(floatX),
                    cudaMemcpyDeviceToHost));
                batch_replay_sample_center_parameter =
                    static_cast<float>(replay_center_parameter_value);
                batch_replay_base_loss = gpt2_validate(
                    &model,
                    train_loader.inputs,
                    train_loader.targets,
                    B,
                    T,
                    mask_sequence_final_target);
                if (!gpt2_normuon_batch_replay_set_wdown(
                        &model,
                        normuon_batch_replay_wdown_snapshot,
                        step_normuon_learning_rate,
                        normuon_batch_replay_amplitude,
                        replay_rounding_step,
                        normuon_batch_replay_center_parameter,
                        normuon_batch_replay_changed_count,
                        normuon_batch_replay_row_scales)) {
                    fprintf(stderr, "failed to apply the positive Wdown replay perturbation\n");
                    exit(EXIT_FAILURE);
                }
                cudaCheck(cudaMemcpy(
                    &batch_replay_plus_changed,
                    normuon_batch_replay_changed_count,
                    sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost));
                floatX replay_plus_parameter_value;
                cudaCheck(cudaMemcpy(
                    &batch_replay_sample_plus_master,
                    model.master_weights + replay_tensor.offset +
                        replay_sample_parameter_offset,
                    sizeof(float),
                    cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(
                    &replay_plus_parameter_value,
                    static_cast<floatX*>(model.params_memory) +
                        replay_tensor.offset + replay_sample_parameter_offset,
                    sizeof(floatX),
                    cudaMemcpyDeviceToHost));
                batch_replay_sample_plus_parameter =
                    static_cast<float>(replay_plus_parameter_value);
                batch_replay_plus_loss = gpt2_validate(
                    &model,
                    train_loader.inputs,
                    train_loader.targets,
                    B,
                    T,
                    mask_sequence_final_target);
                if (!gpt2_normuon_batch_replay_set_wdown(
                        &model,
                        normuon_batch_replay_wdown_snapshot,
                        step_normuon_learning_rate,
                        -normuon_batch_replay_amplitude,
                        replay_rounding_step,
                        normuon_batch_replay_center_parameter,
                        normuon_batch_replay_changed_count,
                        normuon_batch_replay_row_scales)) {
                    fprintf(stderr, "failed to apply the negative Wdown replay perturbation\n");
                    exit(EXIT_FAILURE);
                }
                cudaCheck(cudaMemcpy(
                    &batch_replay_minus_changed,
                    normuon_batch_replay_changed_count,
                    sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost));
                batch_replay_minus_loss = gpt2_validate(
                    &model,
                    train_loader.inputs,
                    train_loader.targets,
                    B,
                    T,
                    mask_sequence_final_target);
                if (!gpt2_normuon_batch_replay_set_wdown(
                        &model,
                        normuon_batch_replay_wdown_snapshot,
                        step_normuon_learning_rate,
                        0.0f,
                        optimizer_global_step,
                        nullptr,
                        nullptr,
                        normuon_batch_replay_row_scales)) {
                    fprintf(stderr, "failed to restore Wdown after same-batch replay\n");
                    exit(EXIT_FAILURE);
                }
                cudaCheck(cudaEventRecord(batch_replay_end));
                cudaCheck(cudaEventSynchronize(batch_replay_end));
                cudaCheck(cudaEventElapsedTime(
                    &batch_replay_time_ms, batch_replay_start, batch_replay_end));
                batch_replay_probed = true;
            }
        }
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end)); // wait for the end event to finish to get correct timings
        // --------------- TRAINING SECTION END -------------------
        // everything that follows now is just diagnostics, prints, logging, etc.

        // todo - move or double-buffer all of this timing logic to avoid idling the GPU at this point!
        float time_elapsed_ms;
        cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, start, end));
        size_t free_device_memory_bytes = 0U;
        size_t total_device_memory_bytes = 0U;
        cudaCheck(cudaMemGetInfo(&free_device_memory_bytes, &total_device_memory_bytes));
        const size_t device_memory_used_bytes =
            total_device_memory_bytes - free_device_memory_bytes;
        if (device_memory_used_bytes > peak_device_memory_used_bytes) {
            peak_device_memory_used_bytes = device_memory_used_bytes;
        }
        const bool finite_step = isfinite(model.mean_loss) && isfinite(grad_norm);
        size_t tokens_processed = (size_t)multi_gpu_config.num_processes * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / time_elapsed_ms * 1000.0f;
        float bias_corrected_ema_tokens_per_second = tokens_per_second; // by default set to non-ema version
        if (step > 0) { // consider the first batch to be a warmup (e.g. cuBLAS/cuDNN initialisation)
            total_sum_iteration_time_s += time_elapsed_ms / 1000.0f;
            // smooth out the tok/s with an exponential moving average, and bias correct just like in AdamW
            ema_tokens_per_second = 0.95f * ema_tokens_per_second + 0.05f * tokens_per_second;
            bias_corrected_ema_tokens_per_second = ema_tokens_per_second / (1.0f - powf(0.95f, step));
        }
        float mfu = gpt2_estimate_mfu(&model, B * T * grad_accum_steps, time_elapsed_ms / 1000.0f);
        printf0("step %4d/%d | loss %7.6f (%+.2fz)| norm %6.4f (%+.2fz)| adamw_lr %.2e | normuon_lr %.2e | optimizer %.2f ms | total %.2f ms | memory_used %.1f MiB | finite %s | %.1f%% bf16 MFU | %.0f tok/s\n",
                step + 1, train_num_batches, model.mean_loss, zloss, grad_norm, zgrad,
                step_learning_rate, step_normuon_learning_rate, optimizer_time_ms,
                time_elapsed_ms, device_memory_used_bytes / (1024.0 * 1024.0),
                finite_step ? "yes" : "no", 100*mfu,
                bias_corrected_ema_tokens_per_second);
        if (batch_replay_probed) {
            const float replay_amplitude = normuon_batch_replay_amplitude;
            const float replay_slope =
                (batch_replay_plus_loss - batch_replay_minus_loss) /
                (2.0f * replay_amplitude);
            const float replay_curvature =
                (batch_replay_plus_loss - 2.0f * batch_replay_base_loss +
                 batch_replay_minus_loss) /
                (replay_amplitude * replay_amplitude);
            const bool replay_optimum_valid =
                isfinite(replay_slope) && isfinite(replay_curvature) &&
                replay_curvature > 0.0f;
            const float replay_optimum_multiplier = replay_optimum_valid
                ? 1.0f - replay_slope / replay_curvature
                : 0.0f;
            printf0(
                "batch_replay {\"step\":%d,\"family\":\"mlp_wdown\","
                "\"amplitude\":%.9g,\"minus_multiplier\":%.9g,"
                "\"base_multiplier\":1,\"plus_multiplier\":%.9g,"
                "\"minus_loss\":%.9g,\"base_loss\":%.9g,"
                "\"plus_loss\":%.9g,\"slope_per_multiplier\":%.9g,"
                "\"curvature_per_multiplier2\":%.9g,"
                "\"quadratic_optimum_valid\":%s,"
                "\"quadratic_optimum_multiplier\":%.9g,"
                "\"common_random_rounding\":true,"
                "\"minus_changed_bf16\":%llu,\"plus_changed_bf16\":%llu,"
                "\"q_sample_max\":%.9g,\"row_scale_sample_max\":%.9g,"
                "\"sample_update_abs\":%.9g,"
                "\"sample_base_master\":%.9g,\"sample_plus_master\":%.9g,"
                "\"sample_center_parameter\":%.9g,"
                "\"sample_plus_parameter\":%.9g,"
                "\"replay_time_ms\":%.9g}\n",
                step + 1,
                replay_amplitude,
                1.0f - replay_amplitude,
                1.0f + replay_amplitude,
                batch_replay_minus_loss,
                batch_replay_base_loss,
                batch_replay_plus_loss,
                replay_slope,
                replay_curvature,
                replay_optimum_valid ? "true" : "false",
                replay_optimum_multiplier,
                batch_replay_minus_changed,
                batch_replay_plus_changed,
                batch_replay_q_sample_max,
                batch_replay_row_scale_sample_max,
                batch_replay_sample_update_abs,
                batch_replay_sample_base_master,
                batch_replay_sample_plus_master,
                batch_replay_sample_center_parameter,
                batch_replay_sample_plus_parameter,
                batch_replay_time_ms);
        }
        if (llmc_normuon_is_cache_mode(
                model.optimizer_config.orthogonalization_mode)) {
            const uint64_t probes = model.normuon_runtime.cache_step_probe_count;
            const double mean_residual = probes > 0U
                ? model.normuon_runtime.cache_step_residual_sum /
                      static_cast<double>(probes)
                : 0.0;
            printf0(
                "cachemuon step %d | probes %llu | misses %llu | hits %llu | "
                "mean_residual %.6f | max_residual %.6f\n",
                step + 1,
                static_cast<unsigned long long>(probes),
                static_cast<unsigned long long>(
                    model.normuon_runtime.cache_step_miss_count),
                static_cast<unsigned long long>(
                    probes - model.normuon_runtime.cache_step_miss_count),
                mean_residual,
                model.normuon_runtime.cache_step_residual_max);
        }
        if (model.normuon_runtime.tracker_diagnostics_active_step) {
            for (int diagnostic_slot = 0;
                 diagnostic_slot < LLMC_NORMUON_TRACKER_DIAGNOSTIC_FAMILY_COUNT;
                 ++diagnostic_slot) {
                const LlmcNormuonTrackerFamilyDiagnostics& diagnostics =
                    model.normuon_runtime
                        .tracker_step_diagnostics[diagnostic_slot];
                if (!diagnostics.valid) {
                    continue;
                }
                const char* family_name =
                    diagnostics.family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
                        ? "mlp_wup"
                        : "mlp_wdown";
                printf0(
                    "tracker_diag {\"step\":%d,\"family\":\"%s\","
                    "\"refreshed\":%s,\"age_since_refresh\":%lld,"
                    "\"phase_probe_count\":%llu,"
                    "\"skew_ratio_mean\":%.9g,\"skew_ratio_p50\":%.9g,"
                    "\"skew_ratio_p95\":%.9g,\"skew_ratio_max\":%.9g,"
                    "\"dimension_normalized_skew_mean\":%.9g,"
                    "\"dimension_normalized_skew_max\":%.9g,"
                    "\"correction_frobenius_mean\":%.9g,"
                    "\"correction_frobenius_max\":%.9g,"
                    "\"refresh_comparison_count\":%llu,"
                     "\"refresh_cosine_mean\":%.9g,"
                     "\"refresh_cosine_min\":%.9g,"
                     "\"refresh_relative_error_mean\":%.9g,"
                     "\"refresh_relative_error_p95\":%.9g,"
                     "\"refresh_relative_error_max\":%.9g,"
                     "\"lr_dither_response_probed\":%s,"
                     "\"lr_dither_source_step\":%llu,"
                     "\"lr_dither_mode\":\"%s\","
                     "\"lr_dither_sign\":%d,"
                     "\"lr_dither_amplitude\":%.9g,"
                     "\"lr_dither_signal\":%.9g,"
                     "\"lr_dither_multiplier\":%.9g,"
                     "\"gradient_dot_previous_update_sum\":%.9g,"
                     "\"normalized_phase_trace_sum\":%.9g,"
                     "\"raw_phase_trace_sum\":%.9g,"
                     "\"h_stability_comparison_count\":%llu,"
                     "\"h_stability_reference_step\":%llu,"
                     "\"h_stability_interval_steps\":%llu,"
                     "\"h_relative_frobenius\":%.9g,"
                     "\"h_cosine\":%.9g,"
                     "\"h_norm_ratio\":%.9g,"
                     "\"h_view_relative_p50\":%.9g,"
                     "\"h_view_relative_p95\":%.9g,"
                     "\"h_view_relative_max\":%.9g}\n",
                    step + 1,
                    family_name,
                    diagnostics.refreshed ? "true" : "false",
                    static_cast<long long>(diagnostics.age_since_refresh),
                    static_cast<unsigned long long>(
                        diagnostics.phase_probe_count),
                    diagnostics.skew_ratio_mean,
                    diagnostics.skew_ratio_p50,
                    diagnostics.skew_ratio_p95,
                    diagnostics.skew_ratio_max,
                    diagnostics.dimension_normalized_skew_mean,
                    diagnostics.dimension_normalized_skew_max,
                    diagnostics.correction_frobenius_mean,
                    diagnostics.correction_frobenius_max,
                    static_cast<unsigned long long>(
                        diagnostics.refresh_comparison_count),
                    diagnostics.refresh_cosine_mean,
                    diagnostics.refresh_cosine_min,
                     diagnostics.refresh_relative_error_mean,
                     diagnostics.refresh_relative_error_p95,
                     diagnostics.refresh_relative_error_max,
                     diagnostics.lr_dither_response_probed ? "true" : "false",
                     static_cast<unsigned long long>(
                         diagnostics.lr_dither_source_step),
                     llmc_normuon_lr_dither_mode_name(
                         model.normuon_runtime.lr_dither_mode),
                     diagnostics.lr_dither_sign,
                     diagnostics.lr_dither_amplitude,
                     diagnostics.lr_dither_signal,
                     diagnostics.lr_dither_multiplier,
                     diagnostics.gradient_dot_previous_update_sum,
                     diagnostics.normalized_phase_trace_sum,
                     diagnostics.raw_phase_trace_sum,
                     static_cast<unsigned long long>(
                         diagnostics.h_stability_comparison_count),
                     static_cast<unsigned long long>(
                         diagnostics.h_stability_reference_step),
                     static_cast<unsigned long long>(
                         diagnostics.h_stability_interval_steps),
                     diagnostics.h_relative_frobenius,
                     diagnostics.h_cosine,
                     diagnostics.h_norm_ratio,
                     diagnostics.h_view_relative_p50,
                     diagnostics.h_view_relative_p95,
                     diagnostics.h_view_relative_max);
            }
            for (size_t diagnostic_index = 0;
                 diagnostic_index <
                     model.normuon_runtime
                         .tracker_step_view_diagnostic_capacity;
                 ++diagnostic_index) {
                if (model.normuon_runtime.lr_dither_enabled &&
                    (model.normuon_runtime.lr_dither_mode ==
                         LLMC_NORMUON_LR_DITHER_SINUSOIDAL ||
                     model.normuon_runtime.lr_dither_mode ==
                         LLMC_NORMUON_LR_DITHER_HETERODYNE_CHOPPER)) {
                    break;
                }
                const LlmcNormuonTrackerViewDiagnostics& diagnostics =
                    model.normuon_runtime
                        .tracker_step_view_diagnostics[diagnostic_index];
                if (!diagnostics.valid || !diagnostics.phase_probed) {
                    continue;
                }
                const char* family_name =
                    diagnostics.family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
                        ? "mlp_wup"
                        : "mlp_wdown";
                const int layer_index =
                    diagnostics.matrix_index /
                    LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
                const int view_index =
                    diagnostics.matrix_index -
                    layer_index * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
                printf0(
                    "tracker_diag_view {\"step\":%d,"
                    "\"family\":\"%s\",\"matrix_index\":%d,"
                    "\"layer\":%d,\"view\":%d,"
                    "\"refreshed\":%s,\"age_since_refresh\":%lld,"
                    "\"skew_ratio\":%.9g,"
                    "\"dimension_normalized_skew\":%.9g,"
                    "\"correction_frobenius\":%.9g,"
                     "\"refresh_compared\":%s,"
                     "\"refresh_cosine\":%.9g,"
                     "\"refresh_relative_error\":%.9g,"
                     "\"lr_dither_response_probed\":%s,"
                     "\"lr_dither_source_step\":%llu,"
                     "\"lr_dither_mode\":\"%s\","
                     "\"lr_dither_sign\":%d,"
                     "\"lr_dither_amplitude\":%.9g,"
                     "\"lr_dither_signal\":%.9g,"
                     "\"lr_dither_multiplier\":%.9g,"
                     "\"gradient_dot_previous_update\":%.9g,"
                     "\"raw_nesterov_norm\":%.9g,"
                     "\"normalized_phase_trace\":%.9g,"
                     "\"raw_phase_trace\":%.9g,"
                     "\"h_stability_compared\":%s,"
                     "\"h_stability_reference_step\":%llu,"
                     "\"h_stability_interval_steps\":%llu,"
                     "\"h_relative_frobenius\":%.9g,"
                     "\"h_cosine\":%.9g,"
                     "\"h_norm_ratio\":%.9g}\n",
                    step + 1,
                    family_name,
                    diagnostics.matrix_index,
                    layer_index,
                    view_index,
                    diagnostics.refreshed ? "true" : "false",
                    static_cast<long long>(diagnostics.age_since_refresh),
                    diagnostics.skew_ratio,
                    diagnostics.dimension_normalized_skew,
                    diagnostics.correction_frobenius,
                     diagnostics.refresh_compared ? "true" : "false",
                     diagnostics.refresh_cosine,
                     diagnostics.refresh_relative_error,
                     diagnostics.lr_dither_response_probed ? "true" : "false",
                     static_cast<unsigned long long>(
                         diagnostics.lr_dither_source_step),
                     llmc_normuon_lr_dither_mode_name(
                         model.normuon_runtime.lr_dither_mode),
                     diagnostics.lr_dither_sign,
                     diagnostics.lr_dither_amplitude,
                     diagnostics.lr_dither_signal,
                     diagnostics.lr_dither_multiplier,
                     diagnostics.gradient_dot_previous_update,
                     diagnostics.raw_nesterov_norm,
                     diagnostics.normalized_phase_trace,
                     diagnostics.raw_phase_trace,
                     diagnostics.h_stability_compared ? "true" : "false",
                     static_cast<unsigned long long>(
                         diagnostics.h_stability_reference_step),
                     static_cast<unsigned long long>(
                         diagnostics.h_stability_interval_steps),
                     diagnostics.h_relative_frobenius,
                     diagnostics.h_cosine,
                     diagnostics.h_norm_ratio);
            }
        }
        if(log_gpu_every > 0 && (step + 1) % log_gpu_every == 0) {
            GPUUtilInfo gpu_info = get_gpu_utilization_info();
            printf0("                  compute %2.1f%% | memory: %2.1f%% | fan: %2d%% | %4d MHz / %4d MHz | %3d W / %3d W | %d°C / %d°C | %s\n",
                    gpu_info.gpu_utilization, gpu_info.mem_utilization, gpu_info.fan, gpu_info.clock, gpu_info.max_clock, gpu_info.power / 1000, gpu_info.power_limit / 1000,
                    gpu_info.temperature, gpu_info.temp_slowdown, gpu_info.throttle_reason);
        }
        logger_log_train(&logger, step, model.mean_loss, step_learning_rate, grad_norm);

        // disable the profiler after 3 steps of optimization
        if (step == 3) { cudaProfilerStop(); }
    }
    // add a total average, for optimizations that are only mild improvements (excluding 1st batch as warmup)
    printf0("total average iteration time: %f ms\n", total_sum_iteration_time_s / (train_num_batches-1) * 1000);
    printf0("average optimizer time: %f ms\n",
            completed_optimizer_steps > 0 ? total_optimizer_time_ms / completed_optimizer_steps : 0.0);
    printf0("peak device memory used: %zu bytes\n", peak_device_memory_used_bytes);
    printf0("CUDA allocator reserved memory: not applicable (direct cudaMalloc; no caching reserve)\n");
    if (llmc_normuon_is_cache_mode(model.optimizer_config.orthogonalization_mode)) {
        const uint64_t probes = model.normuon_runtime.cache_total_probe_count;
        const double mean_residual = probes > 0U
            ? model.normuon_runtime.cache_total_residual_sum /
                  static_cast<double>(probes)
            : 0.0;
        printf0("cachemuon_total_probes: %llu\n",
                static_cast<unsigned long long>(probes));
        printf0("cachemuon_total_misses: %llu\n",
                static_cast<unsigned long long>(
                    model.normuon_runtime.cache_total_miss_count));
        printf0("cachemuon_total_hits: %llu\n",
                static_cast<unsigned long long>(
                    probes - model.normuon_runtime.cache_total_miss_count));
        printf0("cachemuon_mean_residual: %.9g\n", mean_residual);
        printf0("cachemuon_max_residual: %.9g\n",
                model.normuon_runtime.cache_total_residual_max);
    }
    if (model.optimizer_config.tracker_refresh_mode ==
        LLMC_NORMUON_TRACKER_REFRESH_ADAPTIVE_MEAN_SKEW) {
        static constexpr const char* adaptive_family_names[2] = {
            "mlp_wup", "mlp_wdown"};
        for (int family_slot = 0; family_slot < 2; ++family_slot) {
            printf0("tracker_adaptive_%s_checks: %llu\n",
                    adaptive_family_names[family_slot],
                    static_cast<unsigned long long>(
                        model.normuon_runtime
                            .tracker_adaptive_check_count[family_slot]));
            printf0("tracker_adaptive_%s_threshold_refreshes: %llu\n",
                    adaptive_family_names[family_slot],
                    static_cast<unsigned long long>(
                        model.normuon_runtime
                            .tracker_adaptive_threshold_refresh_count[
                                family_slot]));
            printf0("tracker_adaptive_%s_forced_refreshes: %llu\n",
                    adaptive_family_names[family_slot],
                    static_cast<unsigned long long>(
                        model.normuon_runtime
                            .tracker_adaptive_forced_refresh_count[
                                family_slot]));
            printf0("tracker_adaptive_%s_initial_refreshes: %llu\n",
                    adaptive_family_names[family_slot],
                    static_cast<unsigned long long>(
                        model.normuon_runtime
                            .tracker_adaptive_initial_refresh_count[
                                family_slot]));
        }
    }

    // free and destroy everything
    if (batch_replay_end != nullptr) { cudaCheck(cudaEventDestroy(batch_replay_end)); }
    if (batch_replay_start != nullptr) { cudaCheck(cudaEventDestroy(batch_replay_start)); }
    cudaCheck(cudaEventDestroy(optimizer_end));
    cudaCheck(cudaEventDestroy(optimizer_start));
    cudaCheck(cudaEventDestroy(end));
    cudaCheck(cudaEventDestroy(start));
    if (run_hellaswag) { evalloader_free(&eval_loader); }
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    free(cpu_logits_raw);
    free(cpu_logits);
    free(sample_candidates);
    free(gen_tokens);
    if (normuon_batch_replay_wdown_snapshot != nullptr) {
        cudaCheck(cudaFree(normuon_batch_replay_wdown_snapshot));
    }
    if (normuon_batch_replay_center_parameter != nullptr) {
        cudaCheck(cudaFree(normuon_batch_replay_center_parameter));
    }
    if (normuon_batch_replay_row_scales != nullptr) {
        cudaCheck(cudaFree(normuon_batch_replay_row_scales));
    }
    if (normuon_batch_replay_changed_count != nullptr) {
        cudaCheck(cudaFree(normuon_batch_replay_changed_count));
    }
    multi_gpu_config_free(&multi_gpu_config);
    gpt2_free(&model);
    common_free(model);
    return 0;
}
#endif
