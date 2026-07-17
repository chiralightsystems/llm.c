/*
Planned optimizer dispatch for the llm.c GPT-2 trainer.

This header is included from train_gpt2.cu after GPT2, ShardInfo, and the
parameter-offset helpers are defined.
*/
#ifndef LLMC_GPT2_OPTIMIZER_CUH
#define LLMC_GPT2_OPTIMIZER_CUH

inline void gpt2_initialize_master_parameter_type(
    GPT2* model,
    const LlmcOptimizerParameterType* parameter_type,
    MultiGpuConfig* multi_gpu_config) {
    if (model->master_weights == nullptr) {
        return;
    }
    ShardInfo tensor =
        gpt2_get_tensor_at_layer(model, 0, parameter_type->tensor_id);
    ShardInfo shard =
        multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
    const ptrdiff_t local_offset_full = tensor.offset + shard.offset;
    const ptrdiff_t local_offset_partial =
        tensor.offset / multi_gpu_config->num_processes;
    const ptrdiff_t optimizer_offset =
        multi_gpu_config->zero_stage < 1
            ? local_offset_full
            : local_offset_partial;
    floatX* parameter =
        static_cast<floatX*>(model->params_memory) + local_offset_full;
    float* master = model->master_weights + optimizer_offset;
    const size_t grid_size = CEIL_DIV(shard.size, 512);
    copy_and_cast_kernel<<<
        dim3(grid_size, parameter_type->layer_multiplicity),
        512,
        0,
        main_stream>>>(
        master,
        parameter,
        shard.size,
        shard.size,
        tensor.size);
    cudaCheck(cudaGetLastError());
}

inline void gpt2_adamw_update_parameter_type(
    GPT2* model,
    const LlmcOptimizerParameterType* parameter_type,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float gradient_scale,
    int t,
    unsigned int seed,
    MultiGpuConfig* multi_gpu_config,
    bool init_from_master_only) {
    ShardInfo tensor =
        gpt2_get_tensor_at_layer(model, 0, parameter_type->tensor_id);
    ShardInfo shard =
        multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
    const ptrdiff_t local_offset_full = tensor.offset + shard.offset;
    const ptrdiff_t local_offset_partial =
        tensor.offset / multi_gpu_config->num_processes;
    const ptrdiff_t optimizer_offset =
        multi_gpu_config->zero_stage < 1
            ? local_offset_full
            : local_offset_partial;
    floatX* parameter =
        static_cast<floatX*>(model->params_memory) + local_offset_full;
    floatX* gradient =
        static_cast<floatX*>(model->grads_memory) + local_offset_full;
    float* momentum = model->m_memory + optimizer_offset;
    float* second_moment = model->v_memory + optimizer_offset;
    float* master = model->master_weights == nullptr
        ? nullptr
        : model->master_weights + optimizer_offset;
    const float parameter_weight_decay =
        parameter_type->weight_decay_policy == LLMC_WEIGHT_DECAY_ENABLED
            ? weight_decay
            : 0.0f;

    if (init_from_master_only) {
        init_from_master(
            parameter,
            master,
            shard.size,
            tensor.size,
            shard.size,
            parameter_type->layer_multiplicity,
            seed,
            main_stream);
    } else {
        adamw_update(
            parameter,
            master,
            gradient,
            momentum,
            second_moment,
            shard.size,
            tensor.size,
            tensor.size,
            shard.size,
            parameter_type->layer_multiplicity,
            learning_rate,
            beta1,
            beta2,
            t,
            epsilon,
            parameter_weight_decay,
            gradient_scale,
            seed,
            main_stream);
    }

    if (multi_gpu_config->zero_stage == 1) {
#if MULTI_GPU
        ncclCheck(ncclGroupStart());
        for (int layer_index = 0;
             layer_index < parameter_type->layer_multiplicity;
             ++layer_index) {
            ncclCheck(ncclAllGather(
                parameter + layer_index * tensor.size,
                static_cast<floatX*>(model->params_memory) +
                    tensor.offset + layer_index * tensor.size,
                shard.size,
                ncclFloatX,
                multi_gpu_config->nccl_comm,
                multi_gpu_config->nccl_stream));
        }
        ncclCheck(ncclGroupEnd());
#endif
    }
}

inline bool gpt2_normuon_update_parameter_type(
    GPT2* model,
    const LlmcOptimizerParameterType* parameter_type,
    float learning_rate,
    float gradient_scale,
    int t,
    bool init_from_master_only) {
    ShardInfo tensor =
        gpt2_get_tensor_at_layer(model, 0, parameter_type->tensor_id);
    const uint64_t global_step =
        t > 0 ? static_cast<uint64_t>(t - 1) : 0U;
    for (int layer_index = 0;
         layer_index < parameter_type->layer_multiplicity;
         ++layer_index) {
        const size_t layer_offset =
            static_cast<size_t>(layer_index) * tensor.size;
        for (int view_index = 0;
             view_index < parameter_type->views_per_layer;
             ++view_index) {
            LlmcOptimizerMatrixView view;
            if (!parameter_type->enumerate_matrix_view(
                    parameter_type, view_index, &view) ||
                !llmc_optimizer_view_within_bounds(parameter_type, &view)) {
                return false;
            }
            const size_t parameter_offset =
                static_cast<size_t>(tensor.offset) +
                layer_offset +
                view.element_offset;
            const size_t second_moment_offset =
                static_cast<size_t>(tensor.offset) +
                layer_offset +
                view.second_moment_offset;
            floatX* parameter =
                static_cast<floatX*>(model->params_memory) + parameter_offset;
            const floatX* gradient =
                static_cast<const floatX*>(model->grads_memory) +
                parameter_offset;
            float* momentum = model->m_memory + parameter_offset;
            float* second_moment =
                model->v_memory + second_moment_offset;
            float* master = model->master_weights + parameter_offset;
            if (init_from_master_only) {
                if (!llmc_normuon_round_master_view(
                        main_stream,
                        parameter,
                        master,
                        parameter_type,
                        &view,
                        global_step,
                        layer_index,
                        view_index)) {
                    return false;
                }
            } else if (!llmc_normuon_update_view(
                           &model->normuon_runtime,
                           cublas_handle,
                           main_stream,
                           parameter,
                           gradient,
                           momentum,
                           second_moment,
                           master,
                           parameter_type,
                           &view,
                           &model->optimizer_config,
                           learning_rate,
                           gradient_scale,
                           global_step,
                           layer_index,
                           view_index)) {
                return false;
            }
        }
    }
    return true;
}

void gpt2_update(
    GPT2* model,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float gradient_scale,
    int t,
    MultiGpuConfig* multi_gpu_config,
    float normuon_learning_rate = -1.0f,
    bool init_from_master_only = false) {
    NVTX_RANGE_FN();
    if (model->grads_memory == nullptr ||
        model->m_memory == nullptr ||
        model->v_memory == nullptr ||
        !model->optimizer_plan.built) {
        fprintf(stderr, "Need allocated optimizer state and a parameter plan before update\n");
        exit(EXIT_FAILURE);
    }
    const bool uses_normuon =
        model->optimizer_plan.normuon_parameter_type_count != 0;
    if (uses_normuon &&
        (multi_gpu_config->num_processes != 1 ||
         multi_gpu_config->zero_stage != 0)) {
        fprintf(
            stderr,
            "llm.c NorMuon currently supports single-GPU zero_stage=0 only; "
            "flat ZeRO shards can split a square polar view\n");
        exit(EXIT_FAILURE);
    }
    if (uses_normuon && model->master_weights == nullptr) {
        fprintf(stderr, "llm.c NorMuon requires FP32 master weights\n");
        exit(EXIT_FAILURE);
    }
    if (!init_from_master_only && uses_normuon &&
        !(normuon_learning_rate > 0.0f)) {
        fprintf(stderr, "llm.c NorMuon requires a positive scheduled learning rate\n");
        exit(EXIT_FAILURE);
    }
    if (init_from_master_only && model->init_state) {
        fprintf(
            stderr,
            "init_from_master_only requires optimizer state loaded from a checkpoint\n");
        exit(EXIT_FAILURE);
    }

    const bool initialize_state =
        model->init_state && !init_from_master_only;
    if (initialize_state) {
        model->init_state = false;
        NvtxRange range("InitOpt");
        cudaCheck(cudaMemset(
            model->m_memory,
            0,
            multi_gpu_config->shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(
            model->v_memory,
            0,
            multi_gpu_config->shard_num_parameters * sizeof(float)));
    }

    unsigned long long update_rng_state = model->rng_state;
    if (!init_from_master_only) {
        model->rng_state_last_update = model->rng_state;
    }
    unsigned int tensor_seeds[NUM_PARAMETER_TENSORS];
    for (int tensor_id = 0;
         tensor_id < NUM_PARAMETER_TENSORS;
         ++tensor_id) {
        tensor_seeds[tensor_id] = random_u32(&update_rng_state);
    }
    if (!init_from_master_only) {
        model->rng_state = update_rng_state;
    }

    for (int parameter_index = 0;
         parameter_index < LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT;
         ++parameter_index) {
        const LlmcOptimizerParameterType* parameter_type =
            &model->optimizer_plan.parameter_types[parameter_index];
        if (initialize_state && model->master_weights != nullptr) {
            gpt2_initialize_master_parameter_type(
                model, parameter_type, multi_gpu_config);
        }
        switch (parameter_type->backend_kind) {
            case LLMC_OPTIMIZER_BACKEND_ADAMW:
                gpt2_adamw_update_parameter_type(
                    model,
                    parameter_type,
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                    gradient_scale,
                    t,
                    tensor_seeds[parameter_type->tensor_id],
                    multi_gpu_config,
                    init_from_master_only);
                break;
            case LLMC_OPTIMIZER_BACKEND_NORMUON:
                if (!gpt2_normuon_update_parameter_type(
                        model,
                        parameter_type,
                        normuon_learning_rate,
                        gradient_scale,
                        t,
                        init_from_master_only)) {
                    fprintf(
                        stderr,
                        "Nonfinite or invalid NorMuon update at step %d "
                        "for parameter type %s\n",
                        t > 0 ? t - 1 : 0,
                        parameter_type->name);
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(
                    stderr,
                    "Unknown optimizer backend for parameter type %s\n",
                    parameter_type->name);
                exit(EXIT_FAILURE);
        }
    }
    cudaCheck(cudaDeviceSynchronize());
}

#endif // LLMC_GPT2_OPTIMIZER_CUH
