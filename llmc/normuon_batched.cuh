/*
Optimized same-shape batch execution for llm.c blockwise square-view NorMuon.

This adapts the native FRNA square-tile batch implementation to llm.c's BF16
gradient / FP32 master-weight optimizer contract. Dense polynomial and tracker
products use BF16 operands with FP32 accumulation. Persistent momentum, second
moment, master weights, and tracked Q remain FP32.

Tracker retraction intentionally follows the native batched right form
D(3I-D^T D)/2. The FP32 reference uses the algebraically equivalent left form;
the explicit execution-mode boundary records their BF16-rounding difference.
*/
#ifndef LLMC_NORMUON_BATCHED_CUH
#define LLMC_NORMUON_BATCHED_CUH

__device__ __forceinline__ size_t llmc_normuon_batch_tensor_offset(
    int family_id,
    size_t matrix_index,
    size_t element_index,
    size_t width) {
    const size_t matrix_elements = width * width;
    const size_t layer_index = matrix_index / LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t view_index =
        matrix_index - layer_index * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t row = element_index / width;
    const size_t column = element_index - row * width;
    const size_t layer_offset =
        layer_index * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX * matrix_elements;
    if (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP) {
        return layer_offset + view_index * matrix_elements + element_index;
    }
    return layer_offset + view_index * width +
           row * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX * width + column;
}

__device__ __forceinline__ size_t llmc_normuon_batch_second_moment_offset(
    int family_id,
    size_t matrix_index,
    size_t row,
    size_t width) {
    const size_t matrix_elements = width * width;
    const size_t layer_index = matrix_index / LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t view_index =
        matrix_index - layer_index * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t layer_offset =
        layer_index * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX * matrix_elements;
    return layer_offset +
           (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
                ? view_index * matrix_elements
                : view_index * width) +
           row;
}

__device__ __forceinline__ size_t llmc_normuon_batch_q_offset(
    int family_id,
    size_t matrix_index,
    size_t element_index,
    size_t matrix_elements) {
    const size_t layer_index = matrix_index / LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t view_index =
        matrix_index - layer_index * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t family_offset =
        family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 4U;
    const size_t q_view_index =
        layer_index * LLMC_NORMUON_VIEWS_PER_LAYER + family_offset + view_index;
    return q_view_index * matrix_elements + element_index;
}

__global__ void llmc_normuon_batch_prepare_momentum_kernel(
    const floatX* gradient,
    float* momentum,
    float* direction,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t width,
    int family_id,
    float momentum_beta,
    float gradient_scale) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t matrix_elements = width * width;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    float sum = 0.0f;
    for (size_t index = threadIdx.x;
         index < matrix_elements;
         index += blockDim.x) {
        const size_t offset = llmc_normuon_batch_tensor_offset(
            family_id, matrix_index, index, width);
        const float grad = gradient_scale * static_cast<float>(gradient[offset]);
        const float next_momentum =
            momentum_beta * momentum[offset] + (1.0f - momentum_beta) * grad;
        const float nesterov =
            (1.0f - momentum_beta) * grad + momentum_beta * next_momentum;
        if (!isfinite(grad) || !isfinite(next_momentum) || !isfinite(nesterov)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        momentum[offset] = next_momentum;
        direction[matrix_index * matrix_elements + index] = nesterov;
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
        float* matrix_stats =
            stats + matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
        matrix_stats[0] = local[0];
        matrix_stats[1] = 0.0f;
        matrix_stats[2] = 0.0f;
        matrix_stats[3] = 0.0f;
    }
}

__global__ void llmc_normuon_batch_normalize_pack_bf16_kernel(
    float* direction,
    const float* stats,
    uint16_t* packed,
    int* nonfinite,
    size_t total_elements,
    size_t matrix_elements,
    float epsilon) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const float norm = sqrtf(fmaxf(
            stats[matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE], 0.0f));
        const float denominator = 1.02f * norm + epsilon;
        const float value =
            denominator > 0.0f ? direction[linear] / denominator : 0.0f;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        direction[linear] = value;
        packed[linear] =
            __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
}

__global__ void llmc_normuon_batch_pack_bf16_kernel(
    const float* source,
    uint16_t* packed,
    size_t count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        packed[index] =
            __bfloat16_as_ushort(__float2bfloat16_rn(source[index]));
    }
}

__global__ void llmc_normuon_batch_pack_q_bf16_kernel(
    const float* tracked_q,
    uint16_t* packed,
    size_t total_elements,
    size_t matrix_elements,
    int family_id) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t element_index = linear - matrix_index * matrix_elements;
        const size_t q_offset = llmc_normuon_batch_q_offset(
            family_id, matrix_index, element_index, matrix_elements);
        packed[linear] = __bfloat16_as_ushort(
            __float2bfloat16_rn(tracked_q[q_offset]));
    }
}

__global__ void llmc_normuon_batch_scale_to_bf16_kernel(
    const float* source,
    uint16_t* destination,
    size_t count,
    float scale) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        destination[index] = __bfloat16_as_ushort(
            __float2bfloat16_rn(scale * source[index]));
    }
}

__global__ void llmc_normuon_batch_scaled_add_to_bf16_kernel(
    const float* source,
    uint16_t* destination,
    size_t count,
    float scale) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        const float current =
            __bfloat162float(__ushort_as_bfloat16(destination[index]));
        destination[index] = __bfloat16_as_ushort(
            __float2bfloat16_rn(current + scale * source[index]));
    }
}

__global__ void llmc_normuon_batch_add_projected_kernel(
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

__global__ void llmc_normuon_batch_sym_skew_stats_kernel(
    const float* phase,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t width) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t matrix_elements = width * width;
    const float* matrix = phase + matrix_index * matrix_elements;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    float sum = 0.0f;
    for (size_t index = threadIdx.x;
         index < matrix_elements;
         index += blockDim.x) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float symmetric =
            0.5f * (matrix[index] + matrix[column * width + row]);
        if (!isfinite(symmetric)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
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
        stats[matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE] = local[0];
    }
}

__global__ void llmc_normuon_batch_build_correction_kernel(
    const float* phase,
    float* correction,
    const float* stats,
    int* nonfinite,
    size_t total_elements,
    size_t matrix_elements,
    size_t width,
    float gain,
    float epsilon) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / width;
        const size_t column = index - row * width;
        const size_t transpose =
            matrix_index * matrix_elements + column * width + row;
        const float skew = 0.5f * (phase[linear] - phase[transpose]);
        const float denominator = sqrtf(fmaxf(
            stats[matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE], 0.0f)) +
            epsilon;
        const float value = (row == column ? 1.0f : 0.0f) +
                            gain * skew / denominator;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        correction[linear] = value;
    }
}

__global__ void llmc_normuon_batch_three_minus_kernel(
    float* matrix,
    size_t total_elements,
    size_t matrix_elements,
    size_t width) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t index = linear % matrix_elements;
        const size_t row = index / width;
        const size_t column = index - row * width;
        matrix[linear] = (row == column ? 3.0f : 0.0f) - matrix[linear];
    }
}

__global__ void llmc_normuon_batch_second_moment_kernel(
    const float* direction,
    float* second_moment,
    float* row_scales,
    int* nonfinite,
    size_t matrix_count,
    size_t width,
    int family_id,
    float beta2,
    float epsilon) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t matrix_elements = width * width;
    const float* matrix = direction + matrix_index * matrix_elements;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    __shared__ float global_scale;
    float normalized_sum = 0.0f;
    for (size_t row = threadIdx.x; row < width; row += blockDim.x) {
        float sum = 0.0f;
        for (size_t column = 0U; column < width; ++column) {
            const float value = matrix[row * width + column];
            sum += value * value;
        }
        const size_t second_offset = llmc_normuon_batch_second_moment_offset(
            family_id, matrix_index, row, width);
        const float mean = sum / static_cast<float>(width);
        const float next = beta2 * second_moment[second_offset] +
                           (1.0f - beta2) * mean;
        const float contribution = sum / fmaxf(next, epsilon);
        if (!isfinite(next) || !isfinite(contribution)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        second_moment[second_offset] = next;
        normalized_sum += contribution;
    }
    local[threadIdx.x] = normalized_sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        global_scale =
            static_cast<float>(width) / sqrtf(fmaxf(local[0], epsilon));
        if (!isfinite(global_scale)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
    __syncthreads();
    for (size_t row = threadIdx.x; row < width; row += blockDim.x) {
        const size_t second_offset = llmc_normuon_batch_second_moment_offset(
            family_id, matrix_index, row, width);
        const float local_scale =
            rsqrtf(fmaxf(second_moment[second_offset], epsilon)) *
            global_scale;
        if (!isfinite(local_scale)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        row_scales[matrix_index * width + row] = local_scale;
    }
}

__global__ void llmc_normuon_batch_validate_update_kernel(
    const float* master,
    const float* direction,
    const float* row_scales,
    int* nonfinite,
    size_t total_elements,
    size_t matrix_elements,
    size_t width,
    int family_id,
    float learning_rate,
    float weight_decay,
    float update_scale) {
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float update_lr = learning_rate * update_scale;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / width;
        const size_t parameter_offset = llmc_normuon_batch_tensor_offset(
            family_id, matrix_index, index, width);
        const float local_scale = row_scales[matrix_index * width + row];
        const float updated = master[parameter_offset] * decay_scale -
                              update_lr * direction[linear] * local_scale;
        if (!isfinite(master[parameter_offset]) || !isfinite(direction[linear]) ||
            !isfinite(local_scale) || !isfinite(updated)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
}

__global__ void llmc_normuon_batch_apply_update_kernel(
    floatX* parameter,
    float* master,
    const float* direction,
    const float* row_scales,
    float* tracked_q,
    size_t total_elements,
    size_t matrix_elements,
    size_t width,
    int family_id,
    float learning_rate,
    float weight_decay,
    float update_scale,
    uint64_t global_step,
    int tensor_id) {
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float update_lr = learning_rate * update_scale;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t layer_index =
            matrix_index / LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
        const size_t view_index =
            matrix_index - layer_index * LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
        const size_t row = index / width;
        const size_t parameter_offset = llmc_normuon_batch_tensor_offset(
            family_id, matrix_index, index, width);
        const float local_scale = row_scales[matrix_index * width + row];
        const float updated = master[parameter_offset] * decay_scale -
                              update_lr * direction[linear] * local_scale;
        const uint32_t seed = llmc_normuon_rounding_seed(
            global_step,
            tensor_id,
            static_cast<int>(layer_index),
            static_cast<int>(view_index));
        master[parameter_offset] = updated;
        llmc_normuon_stochastic_round(
            updated, &parameter[parameter_offset], seed, index);
        if (tracked_q != nullptr) {
            const size_t q_offset = llmc_normuon_batch_q_offset(
                family_id, matrix_index, index, matrix_elements);
            tracked_q[q_offset] = direction[linear];
        }
    }
}

__global__ void llmc_normuon_batch_copy_direction_to_q_kernel(
    const float* direction,
    float* tracked_q,
    size_t total_elements,
    size_t matrix_elements,
    int family_id) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t element_index = linear - matrix_index * matrix_elements;
        const size_t q_offset = llmc_normuon_batch_q_offset(
            family_id, matrix_index, element_index, matrix_elements);
        tracked_q[q_offset] = direction[linear];
    }
}

inline void llmc_normuon_batched_gemm_bf16(
    cublasHandle_t handle,
    const uint16_t* a,
    cublasOperation_t operation_a,
    const uint16_t* b,
    cublasOperation_t operation_b,
    float* output,
    int width,
    int batch_count,
    float alpha = 1.0f,
    float beta = 0.0f) {
    const long long matrix_stride =
        static_cast<long long>(width) * static_cast<long long>(width);
    cublasCheck(cublasGemmStridedBatchedEx(
        handle,
        operation_a,
        operation_b,
        width,
        width,
        width,
        &alpha,
        a,
        CUDA_R_16BF,
        width,
        matrix_stride,
        b,
        CUDA_R_16BF,
        width,
        matrix_stride,
        &beta,
        output,
        CUDA_R_32F,
        width,
        matrix_stride,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline bool llmc_normuon_apply_polynomial_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    float* matrix,
    float* scratch,
    int width,
    int matrix_count,
    uint32_t stage_count,
    const LlmcNormuonPolynomialStep* schedule,
    uint16_t* packed_factor_override = nullptr) {
    if (runtime == nullptr || handle == nullptr || matrix == nullptr ||
        scratch == nullptr || width <= 0 || matrix_count <= 0 ||
        stage_count == 0U ||
        stage_count > LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT ||
        schedule == nullptr || runtime->batch_bf16[0] == nullptr ||
        runtime->batch_bf16[1] == nullptr) {
        return false;
    }
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t total_elements =
        static_cast<size_t>(matrix_count) * matrix_elements;
    const uint32_t grid = llmc_normuon_grid_for_count(total_elements);
    uint16_t* packed_a = runtime->batch_bf16[0];
    uint16_t* packed_b = packed_factor_override != nullptr
        ? packed_factor_override
        : runtime->batch_bf16[1];
    for (uint32_t stage = 0U; stage < stage_count; ++stage) {
        const LlmcNormuonPolynomialStep coefficient = schedule[stage];
        llmc_normuon_batch_pack_bf16_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            matrix, packed_a, total_elements);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batched_gemm_bf16(
            handle,
            packed_a,
            CUBLAS_OP_N,
            packed_a,
            CUBLAS_OP_T,
            scratch,
            width,
            matrix_count);
        if (coefficient.c != 0.0f) {
            llmc_normuon_batch_pack_bf16_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch, packed_b, total_elements);
            cudaCheck(cudaGetLastError());
            llmc_normuon_batched_gemm_bf16(
                handle,
                packed_b,
                CUBLAS_OP_N,
                packed_b,
                CUBLAS_OP_N,
                scratch,
                width,
                matrix_count,
                coefficient.c,
                coefficient.b);
            llmc_normuon_batch_pack_bf16_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch, packed_b, total_elements);
        } else {
            llmc_normuon_batch_scale_to_bf16_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch, packed_b, total_elements, coefficient.b);
        }
        cudaCheck(cudaGetLastError());
        llmc_normuon_batched_gemm_bf16(
            handle,
            packed_b,
            CUBLAS_OP_N,
            packed_a,
            CUBLAS_OP_N,
            matrix,
            width,
            matrix_count,
            1.0f,
            coefficient.a);
    }
    return true;
}

inline bool llmc_normuon_tracker_direction_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    const LlmcNormuonConfig* config,
    int family_id,
    int width,
    int matrix_count,
    const float** direction_out) {
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t total_elements =
        static_cast<size_t>(matrix_count) * matrix_elements;
    const uint32_t grid = llmc_normuon_grid_for_count(total_elements);
    float* normalized = runtime->matrix[0];
    float* phase = runtime->matrix[1];
    float* correction = runtime->matrix[2];
    uint16_t* packed_a = runtime->batch_bf16[0];
    uint16_t* packed_b = runtime->batch_bf16[1];

    llmc_normuon_batch_pack_bf16_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        normalized, packed_a, total_elements);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batch_pack_q_bf16_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->tracked_q,
        packed_b,
        total_elements,
        matrix_elements,
        family_id);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batched_gemm_bf16(
        handle,
        packed_a,
        CUBLAS_OP_N,
        packed_b,
        CUBLAS_OP_T,
        phase,
        width,
        matrix_count);
    llmc_normuon_batch_sym_skew_stats_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        phase,
        runtime->stats,
        runtime->nonfinite_flag,
        matrix_count,
        width);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batch_build_correction_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        phase,
        correction,
        runtime->stats,
        runtime->nonfinite_flag,
        total_elements,
        matrix_elements,
        width,
        config->correction_gain,
        config->epsilon);
    cudaCheck(cudaGetLastError());
    const bool commute_canonical_stage2 =
        config->retraction_mode ==
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    // The normalized-momentum panel is dead after correction construction and
    // has twice the byte capacity needed for one packed BF16 panel. Borrow its
    // leading half so the correction polynomial does not overwrite packed_b,
    // which still contains the Q packed above for the subsequent Q*C product.
    uint16_t* prefix_polynomial_factor_bf16 =
        reinterpret_cast<uint16_t*>(normalized);
    if (!llmc_normuon_apply_polynomial_batched_bf16(
            runtime,
            handle,
            stream,
            correction,
            phase,
            width,
            matrix_count,
            commute_canonical_stage2 ? 1U : config->correction_iterations,
            config->correction_schedule,
            prefix_polynomial_factor_bf16)) {
        return false;
    }

    llmc_normuon_batch_pack_bf16_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        correction, packed_a, total_elements);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batched_gemm_bf16(
        handle,
        packed_a,
        CUBLAS_OP_N,
        packed_b,
        CUBLAS_OP_N,
        normalized,
        width,
        matrix_count);
    const float* direction = normalized;
    if (commute_canonical_stage2) {
        if (!llmc_normuon_apply_polynomial_batched_bf16(
                runtime,
                handle,
                stream,
                normalized,
                phase,
                width,
                matrix_count,
                1U,
                config->correction_schedule + 1U)) {
            return false;
        }
    } else if (config->retraction_mode ==
               LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ) {
        llmc_normuon_batch_pack_bf16_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            normalized, packed_a, total_elements);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batched_gemm_bf16(
            handle,
            packed_a,
            CUBLAS_OP_N,
            packed_a,
            CUBLAS_OP_T,
            phase,
            width,
            matrix_count);
        llmc_normuon_batch_three_minus_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            phase, total_elements, matrix_elements, width);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batch_pack_bf16_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            phase, packed_b, total_elements);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batched_gemm_bf16(
            handle,
            packed_b,
            CUBLAS_OP_N,
            packed_a,
            CUBLAS_OP_N,
            correction,
            width,
            matrix_count,
            0.5f);
        direction = correction;
    }
    *direction_out = direction;
    return true;
}

inline bool llmc_normuon_batch_tracker_refresh_state(
    const LlmcNormuonRuntime* runtime,
    const LlmcOptimizerParameterType* parameter_type,
    uint64_t global_step,
    uint32_t refresh_interval,
    bool* needs_refresh) {
    if (runtime == nullptr || parameter_type == nullptr ||
        needs_refresh == nullptr || refresh_interval == 0U) {
        return false;
    }
    const bool cadence_refresh = (global_step % refresh_interval) == 0U;
    bool first = false;
    bool have_first = false;
    for (int layer_index = 0;
         layer_index < parameter_type->layer_multiplicity;
         ++layer_index) {
        for (int view_index = 0;
             view_index < parameter_type->views_per_layer;
             ++view_index) {
            const size_t q_index = llmc_normuon_q_view_index(
                parameter_type, layer_index, view_index);
            if (q_index >= runtime->tracked_q_view_count) {
                return false;
            }
            const bool current =
                cadence_refresh || runtime->q_valid[q_index] == 0U;
            if (!have_first) {
                first = current;
                have_first = true;
            } else if (current != first) {
                return false;
            }
        }
    }
    *needs_refresh = first;
    return have_first;
}

inline void llmc_normuon_batch_commit_tracker_metadata(
    LlmcNormuonRuntime* runtime,
    const LlmcOptimizerParameterType* parameter_type,
    uint64_t global_step,
    bool refreshed) {
    for (int layer_index = 0;
         layer_index < parameter_type->layer_multiplicity;
         ++layer_index) {
        for (int view_index = 0;
             view_index < parameter_type->views_per_layer;
             ++view_index) {
            const size_t q_index = llmc_normuon_q_view_index(
                parameter_type, layer_index, view_index);
            runtime->q_valid[q_index] = 1U;
            if (refreshed) {
                runtime->refresh_count[q_index]++;
                runtime->last_refresh_step[q_index] =
                    static_cast<int64_t>(global_step);
            }
        }
    }
}

inline bool llmc_normuon_update_parameter_type_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    floatX* parameter,
    const floatX* gradient,
    float* momentum,
    float* second_moment,
    float* master,
    const LlmcOptimizerParameterType* parameter_type,
    const LlmcNormuonConfig* config,
    float learning_rate,
    float gradient_scale,
    uint64_t global_step) {
    if (runtime == nullptr || handle == nullptr || parameter == nullptr ||
        gradient == nullptr || momentum == nullptr || second_moment == nullptr ||
        master == nullptr || parameter_type == nullptr || config == nullptr ||
        config->execution_mode != LLMC_NORMUON_EXECUTION_BF16_BATCHED ||
        parameter_type->views_per_layer != LLMC_NORMUON_VIEWS_PER_MLP_MATRIX ||
        (parameter_type->family_id != LLMC_OPTIMIZER_FAMILY_MLP_WUP &&
         parameter_type->family_id != LLMC_OPTIMIZER_FAMILY_MLP_WDOWN) ||
        !(learning_rate > 0.0f)) {
        return false;
    }
    const int width = static_cast<int>(parameter_type->matrix_width);
    const int matrix_count = parameter_type->layer_multiplicity *
                             parameter_type->views_per_layer;
    if (width <= 0 || matrix_count <= 0 ||
        static_cast<size_t>(matrix_count) > runtime->batch_matrix_capacity) {
        return false;
    }
    const size_t matrix_elements = static_cast<size_t>(width) * width;
    const size_t total_elements =
        static_cast<size_t>(matrix_count) * matrix_elements;
    if (total_elements > runtime->batch_total_elements ||
        runtime->batch_bf16[0] == nullptr || runtime->batch_bf16[1] == nullptr) {
        return false;
    }
    cublasCheck(cublasSetStream(handle, stream));
    llmc_normuon_reset_guard(runtime, stream);
    llmc_normuon_batch_prepare_momentum_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        gradient,
        momentum,
        runtime->matrix[0],
        runtime->stats,
        runtime->nonfinite_flag,
        matrix_count,
        width,
        parameter_type->family_id,
        config->momentum,
        gradient_scale);
    cudaCheck(cudaGetLastError());
    const uint32_t grid = llmc_normuon_grid_for_count(total_elements);
    llmc_normuon_batch_normalize_pack_bf16_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->matrix[0],
        runtime->stats,
        runtime->batch_bf16[0],
        runtime->nonfinite_flag,
        total_elements,
        matrix_elements,
        config->epsilon);
    cudaCheck(cudaGetLastError());

    const float* direction = runtime->matrix[0];
    bool refreshed = false;
    if (config->orthogonalization_mode == LLMC_NORMUON_ORTHO_NEWTON_SCHULZ) {
        if (!llmc_normuon_apply_polynomial_batched_bf16(
                runtime,
                handle,
                stream,
                runtime->matrix[0],
                runtime->matrix[1],
                width,
                matrix_count,
                LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                config->refresh_schedule)) {
            return false;
        }
    } else {
        if (!llmc_normuon_batch_tracker_refresh_state(
                runtime,
                parameter_type,
                global_step,
                config->refresh_interval,
                &refreshed)) {
            return false;
        }
        if (refreshed) {
            if (!llmc_normuon_apply_polynomial_batched_bf16(
                    runtime,
                    handle,
                    stream,
                    runtime->matrix[0],
                    runtime->matrix[1],
                    width,
                    matrix_count,
                    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                    config->refresh_schedule)) {
                return false;
            }
        } else if (!llmc_normuon_tracker_direction_batched_bf16(
                       runtime,
                       handle,
                       stream,
                       config,
                       parameter_type->family_id,
                       width,
                       matrix_count,
                       &direction)) {
            return false;
        }
    }

    llmc_normuon_batch_second_moment_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        direction,
        second_moment,
        runtime->axis_stats,
        runtime->nonfinite_flag,
        matrix_count,
        width,
        parameter_type->family_id,
        config->beta2,
        config->epsilon);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batch_validate_update_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        master,
        direction,
        runtime->axis_stats,
        runtime->nonfinite_flag,
        total_elements,
        matrix_elements,
        width,
        parameter_type->family_id,
        learning_rate,
        config->weight_decay,
        config->update_scale);
    cudaCheck(cudaGetLastError());
    if (!llmc_normuon_guard_ok(runtime, stream)) {
        return false;
    }
    const bool update_tracker =
        config->orthogonalization_mode ==
        LLMC_NORMUON_ORTHO_SKEW_POLAR_TRACK_Q;
    if (update_tracker) {
        llmc_normuon_batch_commit_tracker_metadata(
            runtime, parameter_type, global_step, refreshed);
    }
    llmc_normuon_batch_apply_update_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        parameter,
        master,
        direction,
        runtime->axis_stats,
        update_tracker ? runtime->tracked_q : nullptr,
        total_elements,
        matrix_elements,
        width,
        parameter_type->family_id,
        learning_rate,
        config->weight_decay,
        config->update_scale,
        global_step,
        parameter_type->tensor_id);
    cudaCheck(cudaGetLastError());
    return true;
}

#endif // LLMC_NORMUON_BATCHED_CUH
