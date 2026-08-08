/*
Optimized same-shape batch execution for llm.c blockwise NorMuon.

This adapts the native FRNA square-tile batch implementation to llm.c's BF16
gradient / FP32 master-weight optimizer contract. Dense polynomial and tracker
products use BF16 operands with FP32 accumulation. Persistent momentum, second
moment, master weights, and tracked Q remain FP32.

The rectangular_muon scratch mode is handled as one contiguous 4d^2 matrix per
layer and uses the smaller-side Gram matrix for its polynomial update.

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
    size_t matrix_elements,
    bool rectangular = false) {
    // The batched rectangular path has one view per MLP matrix (one Wup and
    // one Wdown per layer), while the square path has four views per matrix.
    // Keep the matrix-index decomposition aligned with the active layout;
    // using the square stride here aliases Q across rectangular layers.
    const size_t views_per_matrix = rectangular
        ? LLMC_NORMUON_RECTANGULAR_VIEWS_PER_MLP_MATRIX
        : LLMC_NORMUON_VIEWS_PER_MLP_MATRIX;
    const size_t layer_index = matrix_index / views_per_matrix;
    const size_t view_index =
        matrix_index - layer_index * views_per_matrix;
    const size_t family_offset = rectangular
        ? (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 1U)
        : (family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP ? 0U : 4U);
    const size_t q_view_index =
        layer_index * (rectangular ? LLMC_NORMUON_RECTANGULAR_VIEWS_PER_LAYER
                                   : LLMC_NORMUON_VIEWS_PER_LAYER) +
        family_offset + view_index;
    return q_view_index * matrix_elements + element_index;
}

__device__ __forceinline__ size_t llmc_normuon_batch_rectangular_offset(
    size_t matrix_index,
    size_t element_index,
    size_t width) {
    return matrix_index * (4U * width * width) + element_index;
}

__device__ __forceinline__ size_t llmc_normuon_batch_rectangular_second_offset(
    size_t matrix_index,
    size_t row,
    size_t width) {
    return matrix_index * (4U * width * width) + row;
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

__global__ void llmc_normuon_batch_prepare_rectangular_momentum_kernel(
    const floatX* gradient,
    float* momentum,
    float* direction,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t width,
    size_t rows,
    size_t columns,
    float momentum_beta,
    float gradient_scale) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t matrix_elements = rows * columns;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    float sum = 0.0f;
    for (size_t index = threadIdx.x; index < matrix_elements; index += blockDim.x) {
        const size_t offset = llmc_normuon_batch_rectangular_offset(
            matrix_index, index, width);
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
    float norm_multiplier,
    float epsilon) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const float norm = sqrtf(fmaxf(
            stats[matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE], 0.0f));
        const float denominator = norm_multiplier * norm + epsilon;
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
    int family_id,
    bool rectangular = false) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t element_index = linear - matrix_index * matrix_elements;
        const size_t q_offset = llmc_normuon_batch_q_offset(
            family_id, matrix_index, element_index, matrix_elements, rectangular);
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
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    float coefficient_a) {
    const size_t count = matrix_count * matrix_elements;
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        const size_t matrix_index = index / matrix_elements;
        const size_t element_index = index - matrix_index * matrix_elements;
        const size_t offset = matrix_index * panel_stride + element_index;
        const float value = coefficient_a * matrix[offset] + projected[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        matrix[offset] = value;
    }
}

__global__ void llmc_cachemuon_batch_gather_selected_kernel(
    const float* source,
    float* destination,
    const int* selected_indices,
    size_t selected_count,
    size_t matrix_elements,
    size_t panel_stride) {
    const size_t total = selected_count * matrix_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total; linear += stride) {
        const size_t compact_index = linear / matrix_elements;
        const size_t element_index = linear - compact_index * matrix_elements;
        const size_t source_index =
            static_cast<size_t>(selected_indices[compact_index]);
        destination[compact_index * panel_stride + element_index] =
            source[source_index * panel_stride + element_index];
    }
}

__global__ void llmc_cachemuon_batch_scatter_selected_kernel(
    const float* source,
    float* destination,
    const int* selected_indices,
    size_t selected_count,
    size_t matrix_elements,
    size_t panel_stride) {
    const size_t total = selected_count * matrix_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total; linear += stride) {
        const size_t compact_index = linear / matrix_elements;
        const size_t element_index = linear - compact_index * matrix_elements;
        const size_t destination_index =
            static_cast<size_t>(selected_indices[compact_index]);
        destination[destination_index * panel_stride + element_index] =
            source[compact_index * panel_stride + element_index];
    }
}

__global__ void llmc_cachemuon_batch_scatter_transform_kernel(
    const float* transforms,
    float* tracked_q,
    const int* selected_indices,
    size_t selected_count,
    size_t small_elements,
    int family_id) {
    const size_t total = selected_count * small_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total; linear += stride) {
        const size_t compact_index = linear / small_elements;
        const size_t element_index = linear - compact_index * small_elements;
        const size_t matrix_index =
            static_cast<size_t>(selected_indices[compact_index]);
        const size_t q_offset = llmc_normuon_batch_q_offset(
            family_id,
            matrix_index,
            element_index,
            small_elements,
            true);
        tracked_q[q_offset] =
            transforms[compact_index * small_elements + element_index];
    }
}

__global__ void llmc_normuon_batch_sym_skew_stats_kernel(
    const float* phase,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t panel_stride,
    size_t width) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t matrix_elements = width * width;
    const float* matrix = phase + matrix_index * panel_stride;
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
    size_t panel_stride,
    size_t width,
    float gain,
    float epsilon,
    int correction_mode) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / width;
        const size_t column = index - row * width;
        const size_t panel_offset = matrix_index * panel_stride;
        const size_t offset = panel_offset + index;
        const size_t transpose = panel_offset + column * width + row;
        const float skew = 0.5f * (phase[offset] - phase[transpose]);
        const float denominator = sqrtf(fmaxf(
            stats[matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE], 0.0f)) +
            epsilon;
        float correction_denominator = denominator;
        float correction_numerator = skew;
        if (correction_mode ==
            LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
            const float pair_stiffness =
                phase[panel_offset + row * width + row] +
                phase[panel_offset + column * width + column];
            // H is expected to be positive semidefinite in the tracker basin.
            // Clamp nonpositive or near-singular pair sums to keep the
            // regularized Jacobi approximation finite when Q is stale.
            correction_denominator = fmaxf(pair_stiffness, epsilon);
            correction_numerator = 2.0f * skew;
        }
        const float value =
            (row == column ? 1.0f : 0.0f) +
            gain * correction_numerator / correction_denominator;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        correction[offset] = value;
    }
}

__global__ void llmc_normuon_batch_pack_strided_bf16_kernel(
    const float* source,
    uint16_t* packed,
    size_t matrix_count,
    size_t count_per_matrix,
    size_t source_stride,
    size_t packed_stride) {
    const size_t total = matrix_count * count_per_matrix;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total; linear += stride) {
        const size_t matrix_index = linear / count_per_matrix;
        const size_t element_index = linear - matrix_index * count_per_matrix;
        packed[matrix_index * packed_stride + element_index] =
            __bfloat16_as_ushort(__float2bfloat16_rn(
                source[matrix_index * source_stride + element_index]));
    }
}

__global__ void llmc_normuon_batch_scale_strided_to_bf16_kernel(
    const float* source,
    uint16_t* destination,
    size_t matrix_count,
    size_t count_per_matrix,
    size_t source_stride,
    size_t destination_stride,
    float scale) {
    const size_t total = matrix_count * count_per_matrix;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total; linear += stride) {
        const size_t matrix_index = linear / count_per_matrix;
        const size_t element_index = linear - matrix_index * count_per_matrix;
        destination[matrix_index * destination_stride + element_index] =
            __bfloat16_as_ushort(__float2bfloat16_rn(
                scale * source[matrix_index * source_stride + element_index]));
    }
}

__global__ void llmc_normuon_batch_build_damped_diagonal_correction_kernel(
    const float* phase,
    float* correction,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    float gain,
    float epsilon) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t stats_offset =
        matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
    const float symmetric_scale =
        sqrtf(fmaxf(stats[stats_offset], 0.0f)) /
        sqrtf(static_cast<float>(width));
    const float diagonal_floor =
        fmaxf(LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale, epsilon);
    const float* matrix = phase + matrix_index * panel_stride;
    float* output = correction + matrix_index * panel_stride;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    float sum = 0.0f;
    for (size_t index = threadIdx.x;
         index < matrix_elements;
         index += blockDim.x) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float skew =
            0.5f * (matrix[index] - matrix[column * width + row]);
        const float row_stiffness =
            fmaxf(matrix[row * width + row], diagonal_floor);
        const float column_stiffness =
            fmaxf(matrix[column * width + column], diagonal_floor);
        const float denominator = row_stiffness + column_stiffness;
        const float omega = gain * 2.0f * skew / denominator;
        if (!isfinite(omega)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        output[index] = omega;
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
        stats[stats_offset + 1U] = local[0];
        if (!isfinite(local[0])) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
}

// Two FP32 power iterations estimate the operator norm of each raw Omega
// panel.  The normalized-momentum panel is dead at this point and supplies
// the first 2*width scratch elements for every matrix in the batch.
__global__ void llmc_normuon_batch_power_iteration_kernel(
    const float* matrix,
    float* scratch,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    uint32_t iterations) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    if (width <= 1U) {
        if (threadIdx.x == 0U) {
            const size_t stats_offset =
                matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
            stats[stats_offset + 2U] = 0.0f;
        }
        return;
    }
    const float* input = matrix + matrix_index * panel_stride;
    float* panel_scratch = scratch + matrix_index * panel_stride;
    float* x = panel_scratch;
    float* y = panel_scratch + width;
    const float initial_scale = rsqrtf(static_cast<float>(width));
    for (size_t index = threadIdx.x; index < width; index += blockDim.x) {
        const uint32_t hash =
            static_cast<uint32_t>(matrix_index * 0x9e3779b9U + index) +
            0x7f4a7c15U;
        x[index] = (hash & 1U) != 0U ? initial_scale : -initial_scale;
    }
    __syncthreads();

    float estimate = 0.0f;
    for (uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        float y_sum = 0.0f;
        for (size_t row = threadIdx.x; row < width; row += blockDim.x) {
            float value = 0.0f;
            for (size_t column = 0U; column < width; ++column) {
                value += input[row * width + column] * x[column];
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
                value += input[row * width + column] * y[row];
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
        const size_t stats_offset =
            matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
        stats[stats_offset + 2U] = estimate;
        if (!isfinite(estimate)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
}

__global__ void llmc_normuon_batch_finalize_damped_diagonal_correction_kernel(
    float* correction,
    float* stats,
    int* nonfinite,
    size_t total_elements,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    float epsilon) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / width;
        const size_t column = index - row * width;
        const size_t offset = matrix_index * panel_stride + index;
        const size_t stats_offset =
            matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
    const float raw_norm =
        sqrtf(fmaxf(stats[stats_offset + 1U], 0.0f));
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
        1.0f,
        spectral_limit /
            (fmaxf(stats[stats_offset + 2U], 0.0f) + epsilon));
        const float scale = fminf(frobenius_scale, spectral_scale);
        const float value =
            (row == column ? 1.0f : 0.0f) + scale * correction[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        correction[offset] = value;
        if (index == 0U) {
            // Retain the applied trust-region scale for future telemetry.
            // The per-matrix stats slot is otherwise unused after this point.
            stats[stats_offset + 3U] = scale;
        }
    }
}

// Rectangular tracker candidate fusion.  After the relative correction has
// been polished into C = I + Omega, form B = C - S H^{-1} on the small side.
// The large-side horizontal term D H^{-1} is added after the single Q*B GEMM.
__global__ void llmc_normuon_batch_build_rectangular_tangent_factor_kernel(
    const float* phase,
    float* factor,
    const float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    bool tall,
    float epsilon) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t stats_offset =
        matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
    const float symmetric_scale =
        sqrtf(fmaxf(stats[stats_offset], 0.0f)) /
        sqrtf(static_cast<float>(width));
    const float diagonal_floor =
        fmaxf(LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale, epsilon);
    const float* phase_matrix = phase + matrix_index * panel_stride;
    float* factor_matrix = factor + matrix_index * panel_stride;
    for (size_t index = threadIdx.x; index < matrix_elements;
         index += blockDim.x) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        const float row_stiffness =
            fmaxf(phase_matrix[row * width + row], diagonal_floor);
        const float column_stiffness =
            fmaxf(phase_matrix[column * width + column], diagonal_floor);
        const float denominator = tall ? column_stiffness : row_stiffness;
        const float value = factor_matrix[index] - phase_matrix[index] / denominator;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        factor_matrix[index] = value;
    }
}

__global__ void llmc_normuon_batch_add_rectangular_horizontal_kernel(
    float* candidate,
    const float* normalized,
    const float* phase,
    const float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    size_t rows,
    size_t columns,
    bool tall,
    float epsilon) {
    const size_t total_elements = matrix_count * matrix_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t side = rows < columns ? rows : columns;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / columns;
        const size_t column = index - row * columns;
        const size_t axis = tall ? column : row;
        const size_t panel_offset = matrix_index * panel_stride;
        const size_t stats_offset =
            matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
        const float symmetric_scale =
            sqrtf(fmaxf(stats[stats_offset], 0.0f)) /
            sqrtf(static_cast<float>(side));
        const float diagonal_floor =
            fmaxf(LLMC_NORMUON_TRACKER_DAMPING_ETA * symmetric_scale, epsilon);
        const float stiffness = fmaxf(
            phase[panel_offset + axis * side + axis], diagonal_floor);
        const size_t offset = panel_offset + index;
        const float value = candidate[offset] + normalized[offset] / stiffness;
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        candidate[offset] = value;
    }
}

// Project a rectangular tracker residual into the horizontal tangent space
// E = D - Q * (Q^T D) (or D - (D Q^T) * Q for a wide matrix), then cap its
// Frobenius norm.  This is a live correction on every step; it deliberately
// does not fall back to the cached Q / zero-order hold when the residual is
// large.  The cap is expressed in units of sqrt(small_side), matching the
// scale of a polar factor while preventing a stale-Q residual from dominating
// the candidate.
__global__ void llmc_normuon_batch_project_rectangular_residual_kernel(
    float* normalized,
    const float* projected,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t rows,
    size_t columns,
    float cap,
    float epsilon) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t panel_offset = matrix_index * panel_stride;
    const size_t side = rows < columns ? rows : columns;
    const float target = cap * sqrtf(static_cast<float>(side));
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    __shared__ float residual_scale;
    float sum = 0.0f;
    for (size_t index = threadIdx.x; index < matrix_elements;
         index += blockDim.x) {
        const size_t offset = panel_offset + index;
        const float value = normalized[offset] - projected[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        normalized[offset] = value;
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
        const float norm = sqrtf(fmaxf(local[0], 0.0f));
        if (!isfinite(norm) || !isfinite(target)) {
            llmc_normuon_mark_nonfinite(nonfinite);
            // Preserve the actual residual for diagnostics; this is not a
            // zero-order hold and the nonfinite flag will abort the update.
            residual_scale = 1.0f;
        } else {
            residual_scale = norm > target
                ? target / (norm + epsilon)
                : 1.0f;
        }
    }
    __syncthreads();
    for (size_t index = threadIdx.x; index < matrix_elements;
         index += blockDim.x) {
        const size_t offset = panel_offset + index;
        const float value = residual_scale * normalized[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        normalized[offset] = value;
    }
}

__global__ void llmc_normuon_batch_add_rectangular_residual_kernel(
    float* candidate,
    const float* residual,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride) {
    const size_t total_elements = matrix_count * matrix_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t offset = matrix_index * panel_stride + index;
        const float value = residual != nullptr
            ? candidate[offset] + residual[offset]
            : candidate[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        candidate[offset] = value;
    }
}

// Fuse residual addition with the BF16 pack needed by the Gram guard.  The
// candidate is written back in FP32 so the following retraction sees the same
// direction that was norm-checked.
__global__ void llmc_normuon_batch_add_rectangular_residual_pack_kernel(
    float* candidate,
    const float* residual,
    uint16_t* packed,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride) {
    const size_t total_elements = matrix_count * matrix_elements;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t offset = matrix_index * panel_stride + index;
        const float value = residual != nullptr
            ? candidate[offset] + residual[offset]
            : candidate[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        candidate[offset] = value;
        packed[matrix_index * panel_stride + index] =
            __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
}

// Conservative rectangular spectral guard.  After a single Tensor-Core Gram
// GEMM, the maximum absolute row sum of the symmetric small-side Gram matrix
// is an upper bound on lambda_max(G), so sqrt(row_sum) upper-bounds sigma_max.
// This avoids repeated uncoalesced matrix-vector products while retaining a
// no-fallback, trust-region guard.
__global__ void llmc_normuon_batch_guard_rectangular_gram_kernel(
    float* candidate,
    const float* gram,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t side,
    float pmax,
    float epsilon) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t panel_offset = matrix_index * panel_stride;
    const float* gram_matrix = gram + panel_offset;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    __shared__ float scale;
    float local_max = 0.0f;
    for (size_t row = threadIdx.x; row < side; row += blockDim.x) {
        float row_sum = 0.0f;
        for (size_t column = 0; column < side; ++column) {
            const float value = gram_matrix[row * side + column];
            if (!isfinite(value)) {
                llmc_normuon_mark_nonfinite(nonfinite);
            }
            row_sum += fabsf(value);
        }
        local_max = fmaxf(local_max, row_sum);
    }
    local[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] = fmaxf(local[threadIdx.x], local[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const float sigma_upper = sqrtf(fmaxf(local[0], 0.0f));
        const size_t stats_offset =
            matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
        stats[stats_offset + 2U] = sigma_upper;
        if (!isfinite(sigma_upper) || !isfinite(pmax)) {
            llmc_normuon_mark_nonfinite(nonfinite);
            scale = 1.0f;
        } else {
            scale = sigma_upper > pmax
                ? pmax / (sigma_upper + epsilon)
                : 1.0f;
        }
        if (!isfinite(scale)) {
            llmc_normuon_mark_nonfinite(nonfinite);
            scale = 1.0f;
        }
        stats[stats_offset + 3U] = scale;
    }
    __syncthreads();
    for (size_t index = threadIdx.x; index < matrix_elements;
         index += blockDim.x) {
        const size_t offset = panel_offset + index;
        const float value = scale * candidate[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        candidate[offset] = value;
    }
}

// Legacy diagnostic implementation of the rectangular spectral guard.  The
// production path below uses a Tensor-Core Gram/Gershgorin bound; this kernel
// remains available for controlled power-iteration comparisons.
__global__ void llmc_normuon_batch_guard_rectangular_retraction_kernel(
    float* candidate,
    const float* residual,
    float* scratch,
    float* stats,
    int* nonfinite,
    size_t matrix_count,
    size_t matrix_elements,
    size_t panel_stride,
    size_t rows,
    size_t columns,
    uint32_t iterations,
    float pmax,
    float epsilon) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const size_t side = rows < columns ? rows : columns;
    const size_t large = rows > columns ? rows : columns;
    const size_t panel_offset = matrix_index * panel_stride;
    float* vector = scratch + panel_offset;
    float* image = vector + side;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    __shared__ float scalar;

    // Fuse the horizontal residual add into the guard's first full-matrix
    // pass.  This removes a separate launch and avoids rereading the
    // candidate before the live spectral estimate.
    for (size_t index = threadIdx.x; index < matrix_elements;
         index += blockDim.x) {
        const size_t offset = panel_offset + index;
        const float value = residual != nullptr
            ? candidate[offset] + residual[offset]
            : candidate[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        candidate[offset] = value;
    }
    __syncthreads();

    const float initial = rsqrtf(static_cast<float>(side));
    for (size_t index = threadIdx.x; index < side; index += blockDim.x) {
        vector[index] = initial;
    }
    __syncthreads();
    float sigma = 0.0f;
    for (uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        for (size_t index = threadIdx.x; index < large; index += blockDim.x) {
            float value = 0.0f;
            if (rows >= columns) {
                const size_t row = index;
                for (size_t column = 0; column < columns; ++column) {
                    value += candidate[panel_offset + row * columns + column] *
                             vector[column];
                }
            } else {
                const size_t column = index;
                for (size_t row = 0; row < rows; ++row) {
                    value += candidate[panel_offset + row * columns + column] *
                             vector[row];
                }
            }
            if (!isfinite(value)) {
                llmc_normuon_mark_nonfinite(nonfinite);
            }
            image[index] = value;
        }
        __syncthreads();
        float image_sum = 0.0f;
        for (size_t index = threadIdx.x; index < large; index += blockDim.x) {
            const float value = image[index];
            image_sum += value * value;
        }
        local[threadIdx.x] = image_sum;
        __syncthreads();
        for (uint32_t offset = blockDim.x >> 1U; offset > 0U;
             offset >>= 1U) {
            if (threadIdx.x < offset) {
                local[threadIdx.x] += local[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0U) {
            sigma = sqrtf(fmaxf(local[0], 0.0f));
            scalar = sigma;
        }
        __syncthreads();
        const float image_norm = scalar;
        for (size_t index = threadIdx.x; index < side; index += blockDim.x) {
            float value = 0.0f;
            if (rows >= columns) {
                const size_t column = index;
                for (size_t row = 0; row < rows; ++row) {
                    value += candidate[panel_offset + row * columns + column] *
                             image[row];
                }
            } else {
                const size_t row = index;
                for (size_t column = 0; column < columns; ++column) {
                    value += candidate[panel_offset + row * columns + column] *
                             image[column];
                }
            }
            if (!isfinite(value)) {
                llmc_normuon_mark_nonfinite(nonfinite);
            }
            vector[index] = value;
        }
        __syncthreads();
        float vector_sum = 0.0f;
        for (size_t index = threadIdx.x; index < side; index += blockDim.x) {
            const float value = vector[index];
            vector_sum += value * value;
        }
        local[threadIdx.x] = vector_sum;
        __syncthreads();
        for (uint32_t offset = blockDim.x >> 1U; offset > 0U;
             offset >>= 1U) {
            if (threadIdx.x < offset) {
                local[threadIdx.x] += local[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0U) {
            const float vector_norm = sqrtf(fmaxf(local[0], 0.0f));
            scalar = vector_norm;
            if (!isfinite(vector_norm) || !isfinite(image_norm)) {
                llmc_normuon_mark_nonfinite(nonfinite);
            }
        }
        __syncthreads();
        const float vector_norm = scalar;
        for (size_t index = threadIdx.x; index < side; index += blockDim.x) {
            vector[index] /= fmaxf(vector_norm, epsilon);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const size_t stats_offset =
            matrix_index * LLMC_NORMUON_BATCH_STATS_STRIDE;
        stats[stats_offset + 2U] = sigma;
        const float scale = isfinite(sigma) && isfinite(pmax) && sigma > pmax
            ? pmax / (sigma + epsilon)
            : 1.0f;
        if (!isfinite(scale)) {
            llmc_normuon_mark_nonfinite(nonfinite);
            scalar = 1.0f;
        } else {
            scalar = scale;
        }
        stats[stats_offset + 3U] = scalar;
        if (!isfinite(sigma)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
    __syncthreads();
    const float scale = scalar;
    for (size_t index = threadIdx.x; index < matrix_elements;
         index += blockDim.x) {
        const size_t offset = panel_offset + index;
        const float value = scale * candidate[offset];
        if (!isfinite(value)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        candidate[offset] = value;
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
    __shared__ float norm_local[LLMC_NORMUON_BLOCK_SIZE];
    __shared__ float global_scale;
    float normalized_sum = 0.0f;
    float direction_norm_squared = 0.0f;
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
        direction_norm_squared += sum;
    }
    local[threadIdx.x] = normalized_sum;
    norm_local[threadIdx.x] = direction_norm_squared;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            norm_local[threadIdx.x] += norm_local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        global_scale = sqrtf(fmaxf(norm_local[0], epsilon)) /
                       sqrtf(fmaxf(local[0], epsilon));
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
    float update_scale,
    float learning_rate_multiplier) {
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float update_lr = learning_rate * update_scale *
                            learning_rate_multiplier;
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
    float learning_rate_multiplier,
    uint64_t global_step,
    int tensor_id) {
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float update_lr = learning_rate * update_scale *
                            learning_rate_multiplier;
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

__global__ void llmc_normuon_batch_second_moment_rectangular_kernel(
    const float* direction,
    float* second_moment,
    float* row_scales,
    int* nonfinite,
    size_t matrix_count,
    size_t panel_stride,
    size_t width,
    size_t rows,
    size_t columns,
    float beta2,
    float epsilon) {
    const size_t matrix_index = blockIdx.x;
    if (matrix_index >= matrix_count) {
        return;
    }
    const float* matrix = direction + matrix_index * panel_stride;
    // Canonical NorMuon keeps a row-wise second moment for every matrix
    // shape.  In particular, wide Wdown matrices still normalize their C
    // output-neuron rows rather than switching to per-column statistics.
    const size_t axis_count = rows;
    const size_t reduced_count = columns;
    const size_t axis_stride = axis_count;
    __shared__ float local[LLMC_NORMUON_BLOCK_SIZE];
    __shared__ float norm_local[LLMC_NORMUON_BLOCK_SIZE];
    __shared__ float global_scale;
    float normalized_sum = 0.0f;
    float direction_norm_squared = 0.0f;
    for (size_t axis = threadIdx.x; axis < axis_count; axis += blockDim.x) {
        float sum = 0.0f;
        for (size_t reduced = 0U; reduced < reduced_count; ++reduced) {
            const size_t index = axis * columns + reduced;
            const float value = matrix[index];
            sum += value * value;
        }
        const size_t second_offset =
            llmc_normuon_batch_rectangular_second_offset(matrix_index, axis, width);
        const float mean = sum / static_cast<float>(reduced_count);
        const float next = beta2 * second_moment[second_offset] +
                           (1.0f - beta2) * mean;
        const float contribution = sum / fmaxf(next, epsilon);
        if (!isfinite(next) || !isfinite(contribution)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        second_moment[second_offset] = next;
        normalized_sum += contribution;
        direction_norm_squared += sum;
    }
    local[threadIdx.x] = normalized_sum;
    norm_local[threadIdx.x] = direction_norm_squared;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            local[threadIdx.x] += local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    for (uint32_t offset = blockDim.x >> 1U; offset > 0U; offset >>= 1U) {
        if (threadIdx.x < offset) {
            norm_local[threadIdx.x] += norm_local[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        global_scale = sqrtf(fmaxf(norm_local[0], epsilon)) /
                       sqrtf(fmaxf(local[0], epsilon));
        if (!isfinite(global_scale)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
    __syncthreads();
    for (size_t axis = threadIdx.x; axis < axis_count; axis += blockDim.x) {
        const size_t second_offset =
            llmc_normuon_batch_rectangular_second_offset(matrix_index, axis, width);
        const float local_scale =
            rsqrtf(fmaxf(second_moment[second_offset], epsilon)) * global_scale;
        if (!isfinite(local_scale)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
        row_scales[matrix_index * axis_stride + axis] = local_scale;
    }
}

__global__ void llmc_normuon_batch_validate_rectangular_update_kernel(
    const float* master,
    const float* direction,
    const float* row_scales,
    int* nonfinite,
    size_t total_elements,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    size_t rows,
    size_t columns,
    float learning_rate,
    float weight_decay,
    float update_scale,
    float learning_rate_multiplier) {
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float shape_scale = sqrtf(fmaxf(
        1.0f, static_cast<float>(rows) / static_cast<float>(columns)));
    const float update_lr = learning_rate * update_scale *
                            learning_rate_multiplier * shape_scale;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / columns;
        const size_t axis = row;
        const size_t axis_stride = rows;
        const size_t parameter_offset =
            llmc_normuon_batch_rectangular_offset(matrix_index, index, width);
        const float local_scale = row_scales[matrix_index * axis_stride + axis];
        const size_t direction_offset = matrix_index * panel_stride + index;
        const float updated = master[parameter_offset] * decay_scale -
                              update_lr * direction[direction_offset] * local_scale;
        if (!isfinite(master[parameter_offset]) || !isfinite(direction[direction_offset]) ||
            !isfinite(local_scale) || !isfinite(updated)) {
            llmc_normuon_mark_nonfinite(nonfinite);
        }
    }
}

__global__ void llmc_normuon_batch_apply_rectangular_update_kernel(
    floatX* parameter,
    float* master,
    const float* direction,
    const float* row_scales,
    size_t total_elements,
    size_t matrix_elements,
    size_t panel_stride,
    size_t width,
    size_t rows,
    size_t columns,
    float learning_rate,
    float weight_decay,
    float update_scale,
    float learning_rate_multiplier,
    uint64_t global_step,
    int tensor_id,
    float* tracked_q = nullptr,
    int family_id = LLMC_OPTIMIZER_FAMILY_MLP_WUP,
    bool rectangular_q = false) {
    const float decay_scale = 1.0f - learning_rate * weight_decay;
    const float shape_scale = sqrtf(fmaxf(
        1.0f, static_cast<float>(rows) / static_cast<float>(columns)));
    const float update_lr = learning_rate * update_scale *
                            learning_rate_multiplier * shape_scale;
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t index = linear - matrix_index * matrix_elements;
        const size_t row = index / columns;
        const size_t axis = row;
        const size_t axis_stride = rows;
        const size_t parameter_offset =
            llmc_normuon_batch_rectangular_offset(matrix_index, index, width);
        const float local_scale = row_scales[matrix_index * axis_stride + axis];
        const size_t direction_offset = matrix_index * panel_stride + index;
        const float updated = master[parameter_offset] * decay_scale -
                              update_lr * direction[direction_offset] * local_scale;
        const uint32_t seed = llmc_normuon_rounding_seed(
            global_step, tensor_id, static_cast<int>(matrix_index), 0);
        master[parameter_offset] = updated;
        llmc_normuon_stochastic_round(
            updated, &parameter[parameter_offset], seed, index);
        if (tracked_q != nullptr) {
            const size_t q_offset = llmc_normuon_batch_q_offset(
                family_id,
                matrix_index,
                index,
                matrix_elements,
                rectangular_q);
            tracked_q[q_offset] = direction[direction_offset];
        }
    }
}

__global__ void llmc_normuon_batch_copy_direction_to_q_kernel(
    const float* direction,
    float* tracked_q,
    size_t total_elements,
    size_t matrix_elements,
    size_t panel_stride,
    int family_id,
    bool rectangular = false) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; linear < total_elements; linear += stride) {
        const size_t matrix_index = linear / matrix_elements;
        const size_t element_index = linear - matrix_index * matrix_elements;
        const size_t q_offset = llmc_normuon_batch_q_offset(
            family_id, matrix_index, element_index, matrix_elements, rectangular);
        tracked_q[q_offset] = direction[matrix_index * panel_stride + element_index];
    }
}

inline void llmc_normuon_batched_gemm_bf16_ex(
    cublasHandle_t handle,
    const uint16_t* a,
    cublasOperation_t operation_a,
    int lda,
    long long stride_a,
    const uint16_t* b,
    cublasOperation_t operation_b,
    int ldb,
    long long stride_b,
    float* output,
    int ldc,
    long long stride_c,
    int m,
    int n,
    int k,
    int batch_count,
    float alpha = 1.0f,
    float beta = 0.0f) {
    cublasCheck(cublasGemmStridedBatchedEx(
        handle,
        operation_a,
        operation_b,
        m,
        n,
        k,
        &alpha,
        a,
        CUDA_R_16BF,
        lda,
        stride_a,
        b,
        CUDA_R_16BF,
        ldb,
        stride_b,
        &beta,
        output,
        CUDA_R_32F,
        ldc,
        stride_c,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
    llmc_normuon_batched_gemm_bf16_ex(
        handle, a, operation_a, width, matrix_stride, b, operation_b, width,
        matrix_stride, output, width, matrix_stride, width, width, width,
        batch_count, alpha, beta);
}

inline void llmc_cachemuon_small_gemm_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    const float* lhs,
    const float* rhs,
    float* output,
    int side,
    int matrix_count) {
    const size_t small_elements = static_cast<size_t>(side) * side;
    const size_t total = static_cast<size_t>(matrix_count) * small_elements;
    const uint32_t grid = llmc_normuon_grid_for_count(total);
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        lhs,
        runtime->batch_bf16[0],
        matrix_count,
        small_elements,
        small_elements,
        small_elements);
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        rhs,
        runtime->batch_bf16[1],
        matrix_count,
        small_elements,
        small_elements,
        small_elements);
    cudaCheck(cudaGetLastError());
    // cuBLAS sees the row-major operands in reverse order.
    llmc_normuon_batched_gemm_bf16_ex(
        handle,
        runtime->batch_bf16[1],
        CUBLAS_OP_N,
        side,
        static_cast<long long>(small_elements),
        runtime->batch_bf16[0],
        CUBLAS_OP_N,
        side,
        static_cast<long long>(small_elements),
        output,
        side,
        static_cast<long long>(small_elements),
        side,
        side,
        side,
        matrix_count);
}

inline void llmc_cachemuon_rectangular_gram_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    const float* matrix,
    float* gram,
    int rows,
    int columns,
    int matrix_count,
    size_t panel_stride) {
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const int side = rows < columns ? rows : columns;
    const size_t small_elements = static_cast<size_t>(side) * side;
    const size_t total = static_cast<size_t>(matrix_count) * matrix_elements;
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        llmc_normuon_grid_for_count(total),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        matrix,
        runtime->batch_bf16[0],
        matrix_count,
        matrix_elements,
        panel_stride,
        panel_stride);
    cudaCheck(cudaGetLastError());
    if (rows >= columns) {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            runtime->batch_bf16[0],
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            runtime->batch_bf16[0],
            CUBLAS_OP_T,
            columns,
            static_cast<long long>(panel_stride),
            gram,
            side,
            static_cast<long long>(small_elements),
            columns,
            columns,
            rows,
            matrix_count);
    } else {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            runtime->batch_bf16[0],
            CUBLAS_OP_T,
            columns,
            static_cast<long long>(panel_stride),
            runtime->batch_bf16[0],
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            gram,
            side,
            static_cast<long long>(small_elements),
            rows,
            rows,
            columns,
            matrix_count);
    }
}

inline void llmc_cachemuon_apply_transform_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    const float* matrix,
    const float* transform,
    float* output,
    int rows,
    int columns,
    int matrix_count,
    size_t panel_stride) {
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const int side = rows < columns ? rows : columns;
    const size_t small_elements = static_cast<size_t>(side) * side;
    const uint32_t matrix_grid = llmc_normuon_grid_for_count(
        static_cast<size_t>(matrix_count) * matrix_elements);
    const uint32_t small_grid = llmc_normuon_grid_for_count(
        static_cast<size_t>(matrix_count) * small_elements);
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        matrix,
        runtime->batch_bf16[0],
        matrix_count,
        matrix_elements,
        panel_stride,
        panel_stride);
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        transform,
        runtime->batch_bf16[1],
        matrix_count,
        small_elements,
        small_elements,
        small_elements);
    cudaCheck(cudaGetLastError());
    if (rows >= columns) {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            runtime->batch_bf16[1],
            CUBLAS_OP_T,
            side,
            static_cast<long long>(small_elements),
            runtime->batch_bf16[0],
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            output,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    } else {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            runtime->batch_bf16[0],
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            runtime->batch_bf16[1],
            CUBLAS_OP_N,
            side,
            static_cast<long long>(small_elements),
            output,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    }
}

inline bool llmc_cachemuon_fresh_gns_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    float* matrix,
    float* alternate,
    int rows,
    int columns,
    int matrix_count,
    size_t panel_stride,
    float** direction_out,
    float** transform_out) {
    if (runtime == nullptr || handle == nullptr || matrix == nullptr ||
        alternate == nullptr || direction_out == nullptr ||
        transform_out == nullptr || rows <= 0 || columns <= 0 ||
        matrix_count <= 0 || runtime->cache_small_total_elements == 0U) {
        return false;
    }
    const int side = rows < columns ? rows : columns;
    const size_t small_elements = static_cast<size_t>(side) * side;
    const size_t total_small = static_cast<size_t>(matrix_count) * small_elements;
    const uint32_t small_grid = llmc_normuon_grid_for_count(total_small);
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
        q_local,
        matrix_count,
        small_elements,
        small_elements,
        side);
    cudaCheck(cudaGetLastError());
    float* current = matrix;
    float* next = alternate;
    llmc_cachemuon_rectangular_gram_batched_bf16(
        runtime,
        handle,
        stream,
        current,
        gram,
        rows,
        columns,
        matrix_count,
        panel_stride);
    for (uint32_t stage = 1U;
         stage <= LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT;
         ++stage) {
        if (stage == LLMC_CACHEMUON_RESTART_STAGE) {
            llmc_cachemuon_apply_transform_batched_bf16(
                runtime,
                handle,
                stream,
                current,
                q_local,
                next,
                rows,
                columns,
                matrix_count,
                panel_stride);
            float* exchange = current;
            current = next;
            next = exchange;
            cudaCheck(cudaMemcpyAsync(
                q_total,
                q_local,
                total_small * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
            llmc_cachemuon_rectangular_gram_batched_bf16(
                runtime,
                handle,
                stream,
                current,
                gram,
                rows,
                columns,
                matrix_count,
                panel_stride);
            llmc_cachemuon_identity_kernel<<<
                small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                q_local,
                matrix_count,
                small_elements,
                small_elements,
                side);
            cudaCheck(cudaGetLastError());
        }

        llmc_cachemuon_small_gemm_batched_bf16(
            runtime,
            handle,
            stream,
            gram,
            gram,
            temporary,
            side,
            matrix_count);
        llmc_cachemuon_form_polynomial_kernel<<<
            small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            gram,
            temporary,
            polynomial,
            runtime->nonfinite_flag,
            matrix_count,
            small_elements,
            small_elements,
            side,
            kLlmcCacheMuonGramGns[stage - 1U]);
        cudaCheck(cudaGetLastError());
        if (stage < LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT &&
            stage + 1U != LLMC_CACHEMUON_RESTART_STAGE) {
            llmc_cachemuon_small_gemm_batched_bf16(
                runtime,
                handle,
                stream,
                gram,
                polynomial,
                temporary,
                side,
                matrix_count);
            llmc_cachemuon_small_gemm_batched_bf16(
                runtime,
                handle,
                stream,
                polynomial,
                temporary,
                gram,
                side,
                matrix_count);
        }
        llmc_cachemuon_small_gemm_batched_bf16(
            runtime,
            handle,
            stream,
            q_local,
            polynomial,
            temporary,
            side,
            matrix_count);
        float* exchange = q_local;
        q_local = temporary;
        temporary = exchange;
    }

    llmc_cachemuon_apply_transform_batched_bf16(
        runtime,
        handle,
        stream,
        current,
        q_local,
        next,
        rows,
        columns,
        matrix_count,
        panel_stride);
    current = next;
    llmc_cachemuon_small_gemm_batched_bf16(
        runtime,
        handle,
        stream,
        q_local,
        q_total,
        polynomial,
        side,
        matrix_count);
    *direction_out = current;
    *transform_out = polynomial;
    return true;
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
    const long long panel_stride =
        static_cast<long long>(runtime->matrix_elements);
    const uint32_t grid = llmc_normuon_grid_for_count(total_elements);
    uint16_t* packed_a = runtime->batch_bf16[0];
    uint16_t* packed_b = packed_factor_override != nullptr
        ? packed_factor_override
        : runtime->batch_bf16[1];
    for (uint32_t stage = 0U; stage < stage_count; ++stage) {
        const LlmcNormuonPolynomialStep coefficient = schedule[stage];
        llmc_normuon_batch_pack_strided_bf16_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            matrix,
            packed_a,
            matrix_count,
            matrix_elements,
            panel_stride,
            panel_stride);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_a,
            CUBLAS_OP_N,
            width,
            panel_stride,
            packed_a,
            CUBLAS_OP_T,
            width,
            panel_stride,
            scratch,
            width,
            panel_stride,
            width,
            width,
            width,
            matrix_count);
        if (coefficient.c != 0.0f) {
            llmc_normuon_batch_pack_strided_bf16_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch,
                packed_b,
                matrix_count,
                matrix_elements,
                panel_stride,
                panel_stride);
            cudaCheck(cudaGetLastError());
            llmc_normuon_batched_gemm_bf16_ex(
                handle,
                packed_b,
                CUBLAS_OP_N,
                width,
                panel_stride,
                packed_b,
                CUBLAS_OP_N,
                width,
                panel_stride,
                scratch,
                width,
                panel_stride,
                width,
                width,
                width,
                matrix_count,
                coefficient.c,
                coefficient.b);
            llmc_normuon_batch_pack_strided_bf16_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch,
                packed_b,
                matrix_count,
                matrix_elements,
                panel_stride,
                panel_stride);
        } else {
            llmc_normuon_batch_scale_strided_to_bf16_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch,
                packed_b,
                matrix_count,
                matrix_elements,
                panel_stride,
                panel_stride,
                coefficient.b);
        }
        cudaCheck(cudaGetLastError());
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_b,
            CUBLAS_OP_N,
            width,
            panel_stride,
            packed_a,
            CUBLAS_OP_N,
            width,
            panel_stride,
            matrix,
            width,
            panel_stride,
            width,
            width,
            width,
            matrix_count,
            1.0f,
            coefficient.a);
    }
    return true;
}

inline bool llmc_normuon_apply_rectangular_polynomial_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    float* matrix,
    float* scratch,
    int rows,
    int columns,
    int matrix_count,
    uint32_t stage_count,
    const LlmcNormuonPolynomialStep* schedule) {
    if (runtime == nullptr || handle == nullptr || matrix == nullptr ||
        scratch == nullptr || rows <= 0 || columns <= 0 || matrix_count <= 0 ||
        stage_count == 0U ||
        stage_count > LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT || schedule == nullptr ||
        runtime->batch_bf16[0] == nullptr || runtime->batch_bf16[1] == nullptr) {
        return false;
    }
    const int side = rows < columns ? rows : columns;
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const size_t gram_elements = static_cast<size_t>(side) * side;
    const size_t panel_stride = runtime->matrix_elements;
    const size_t total_elements = static_cast<size_t>(matrix_count) * matrix_elements;
    const size_t total_gram_elements = static_cast<size_t>(matrix_count) * gram_elements;
    const uint32_t matrix_grid = llmc_normuon_grid_for_count(total_elements);
    const uint32_t gram_grid = llmc_normuon_grid_for_count(total_gram_elements);
    uint16_t* packed_a = runtime->batch_bf16[0];
    uint16_t* packed_b = runtime->batch_bf16[1];
    const bool tall = rows >= columns;
    for (uint32_t stage = 0U; stage < stage_count; ++stage) {
        const LlmcNormuonPolynomialStep coefficient = schedule[stage];
        llmc_normuon_batch_pack_strided_bf16_kernel<<<
            matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            matrix, packed_a, matrix_count, matrix_elements, panel_stride,
            panel_stride);
        cudaCheck(cudaGetLastError());
        if (tall) {
            // G = X^T X, with the row-major GEMM operands reversed for cuBLAS.
            llmc_normuon_batched_gemm_bf16_ex(
                handle, packed_a, CUBLAS_OP_N, columns, panel_stride,
                packed_a, CUBLAS_OP_T, columns, panel_stride, scratch,
                side, panel_stride, columns, columns, rows, matrix_count);
        } else {
            // G = X X^T.
            llmc_normuon_batched_gemm_bf16_ex(
                handle, packed_a, CUBLAS_OP_T, columns, panel_stride,
                packed_a, CUBLAS_OP_N, columns, panel_stride, scratch,
                side, panel_stride, rows, rows, columns, matrix_count);
        }
        if (coefficient.c != 0.0f) {
            llmc_normuon_batch_pack_strided_bf16_kernel<<<
                gram_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch, packed_b, matrix_count, gram_elements, panel_stride,
                panel_stride);
            cudaCheck(cudaGetLastError());
            llmc_normuon_batched_gemm_bf16_ex(
                handle, packed_b, CUBLAS_OP_N, side, panel_stride,
                packed_b, CUBLAS_OP_N, side, panel_stride, scratch,
                side, panel_stride, side, side, side, matrix_count,
                coefficient.c, coefficient.b);
            llmc_normuon_batch_pack_strided_bf16_kernel<<<
                gram_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch, packed_b, matrix_count, gram_elements, panel_stride,
                panel_stride);
        } else {
            llmc_normuon_batch_scale_strided_to_bf16_kernel<<<
                gram_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                scratch, packed_b, matrix_count, gram_elements, panel_stride,
                panel_stride, coefficient.b);
        }
        cudaCheck(cudaGetLastError());
        if (tall) {
            // Y = X P.
            llmc_normuon_batched_gemm_bf16_ex(
                handle, packed_b, CUBLAS_OP_N, side, panel_stride,
                packed_a, CUBLAS_OP_N, columns, panel_stride, scratch,
                columns, panel_stride, columns, rows, columns, matrix_count,
                1.0f, 0.0f);
        } else {
            // Y = P X.
            llmc_normuon_batched_gemm_bf16_ex(
                handle, packed_a, CUBLAS_OP_N, columns, panel_stride,
                packed_b, CUBLAS_OP_N, side, panel_stride, scratch,
                columns, panel_stride, columns, rows, rows, matrix_count,
                1.0f, 0.0f);
        }
        // cuBLAS wrote Y; combine with the old matrix in a separate pass.
        llmc_normuon_batch_add_projected_kernel<<<
            matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            matrix, scratch, runtime->nonfinite_flag, matrix_count,
            matrix_elements, runtime->matrix_elements, coefficient.a);
        cudaCheck(cudaGetLastError());
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
        matrix_elements,
        width);
    cudaCheck(cudaGetLastError());
    if (config->correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
        llmc_normuon_batch_build_damped_diagonal_correction_kernel<<<
            static_cast<uint32_t>(matrix_count),
            LLMC_NORMUON_BLOCK_SIZE,
            0,
            stream>>>(
            phase,
            correction,
            runtime->stats,
            runtime->nonfinite_flag,
            matrix_count,
            matrix_elements,
            matrix_elements,
            width,
            config->correction_gain,
            config->epsilon);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batch_power_iteration_kernel<<<
            static_cast<uint32_t>(matrix_count),
            LLMC_NORMUON_BLOCK_SIZE,
            0,
            stream>>>(
            correction,
            normalized,
            runtime->stats,
            runtime->nonfinite_flag,
            matrix_count,
            matrix_elements,
            matrix_elements,
            width,
            LLMC_NORMUON_TRACKER_SPECTRAL_POWER_ITERATIONS);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batch_finalize_damped_diagonal_correction_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            correction,
            runtime->stats,
            runtime->nonfinite_flag,
            total_elements,
            matrix_elements,
            matrix_elements,
            width,
            config->epsilon);
        cudaCheck(cudaGetLastError());
    } else {
        llmc_normuon_batch_build_correction_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            phase,
            correction,
            runtime->stats,
            runtime->nonfinite_flag,
            total_elements,
            matrix_elements,
            matrix_elements,
            width,
            config->correction_gain,
            config->epsilon,
            static_cast<int>(config->correction_mode));
        cudaCheck(cudaGetLastError());
    }
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

inline bool llmc_normuon_rectangular_tracker_direction_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    const LlmcNormuonConfig* config,
    int family_id,
    int width,
    int rows,
    int columns,
    int matrix_count,
    const float** direction_out) {
    if (runtime == nullptr || handle == nullptr || config == nullptr ||
        direction_out == nullptr || width <= 0 ||
        rows <= 0 || columns <= 0 || matrix_count <= 0) {
        return false;
    }
    const int side = rows < columns ? rows : columns;
    const bool tall = rows >= columns;
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const size_t small_elements = static_cast<size_t>(side) * side;
    const size_t total_elements =
        static_cast<size_t>(matrix_count) * matrix_elements;
    const size_t total_small_elements =
        static_cast<size_t>(matrix_count) * small_elements;
    const size_t panel_stride = runtime->matrix_elements;
    const uint32_t matrix_grid = llmc_normuon_grid_for_count(total_elements);
    const uint32_t small_grid =
        llmc_normuon_grid_for_count(total_small_elements);
    float* normalized = runtime->matrix[0];
    float* phase = runtime->matrix[1];
    float* correction = runtime->matrix[2];
    uint16_t* packed_a = runtime->batch_bf16[0];
    uint16_t* packed_b = runtime->batch_bf16[1];

    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        normalized,
        packed_a,
        matrix_count,
        matrix_elements,
        panel_stride,
        panel_stride);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batch_pack_q_bf16_kernel<<<
        matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->tracked_q,
        packed_b,
        total_elements,
        matrix_elements,
        family_id,
        true);
    cudaCheck(cudaGetLastError());

    // The row-major batched helper reverses operands internally, so these are
    // Q^T D for tall Wup and D Q^T for wide Wdown.
    if (tall) {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_a,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            packed_b,
            CUBLAS_OP_T,
            columns,
            static_cast<long long>(panel_stride),
            phase,
            side,
            static_cast<long long>(panel_stride),
            side,
            side,
            rows,
            matrix_count);
    } else {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_b,
            CUBLAS_OP_T,
            columns,
            static_cast<long long>(panel_stride),
            packed_a,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            phase,
            side,
            static_cast<long long>(panel_stride),
            side,
            side,
            columns,
            matrix_count);
    }
    llmc_normuon_batch_sym_skew_stats_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        phase,
        runtime->stats,
        runtime->nonfinite_flag,
        matrix_count,
        panel_stride,
        side);
    cudaCheck(cudaGetLastError());

    if (config->correction_mode ==
        LLMC_NORMUON_TRACKER_CORRECTION_DIAGONAL_SYLVESTER) {
        llmc_normuon_batch_build_damped_diagonal_correction_kernel<<<
            static_cast<uint32_t>(matrix_count),
            LLMC_NORMUON_BLOCK_SIZE,
            0,
            stream>>>(
            phase,
            correction,
            runtime->stats,
            runtime->nonfinite_flag,
            matrix_count,
            small_elements,
            panel_stride,
            side,
            config->correction_gain,
            config->epsilon);
        cudaCheck(cudaGetLastError());
        // The correction is small-side, so the existing per-matrix power
        // iteration remains cheap relative to the surrounding rectangular
        // GEMMs.
        llmc_normuon_batch_power_iteration_kernel<<<
            static_cast<uint32_t>(matrix_count),
            LLMC_NORMUON_BLOCK_SIZE,
            0,
            stream>>>(
            correction,
            normalized,
            runtime->stats,
            runtime->nonfinite_flag,
            matrix_count,
            small_elements,
            panel_stride,
            side,
            LLMC_NORMUON_TRACKER_SPECTRAL_POWER_ITERATIONS);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batch_finalize_damped_diagonal_correction_kernel<<<
            small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            correction,
            runtime->stats,
            runtime->nonfinite_flag,
            total_small_elements,
            small_elements,
            panel_stride,
            side,
            config->epsilon);
        cudaCheck(cudaGetLastError());
    } else {
        llmc_normuon_batch_build_correction_kernel<<<
            small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            phase,
            correction,
            runtime->stats,
            runtime->nonfinite_flag,
            total_small_elements,
            small_elements,
            panel_stride,
            side,
            config->correction_gain,
            config->epsilon,
            static_cast<int>(config->correction_mode));
        cudaCheck(cudaGetLastError());
    }

    const bool commute_canonical_stage2 =
        config->retraction_mode ==
        LLMC_NORMUON_TRACKER_RETRACTION_COMMUTED_CANONICAL_STAGE2;
    const bool thin_canonical_stage2 =
        config->retraction_mode ==
        LLMC_NORMUON_TRACKER_RETRACTION_THIN_CANONICAL_STAGE2;
    // The rectangular normalized panel must survive for the horizontal
    // tangent term, so the correction polynomial is allowed to overwrite the
    // packed-Q panel.  Q is repacked once below before the candidate GEMM.
    if (!llmc_normuon_apply_polynomial_batched_bf16(
            runtime,
            handle,
            stream,
            correction,
            phase,
            side,
            matrix_count,
            (commute_canonical_stage2 || thin_canonical_stage2)
                ? 1U
                : config->correction_iterations,
            config->correction_schedule,
            nullptr)) {
        return false;
    }
    llmc_normuon_batch_build_rectangular_tangent_factor_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        phase,
        correction,
        runtime->stats,
        runtime->nonfinite_flag,
        matrix_count,
        small_elements,
        panel_stride,
        side,
        tall,
        config->epsilon);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        correction,
        packed_a,
        matrix_count,
        small_elements,
        panel_stride,
        panel_stride);
    cudaCheck(cudaGetLastError());
    llmc_normuon_batch_pack_q_bf16_kernel<<<
        matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->tracked_q,
        packed_b,
        total_elements,
        matrix_elements,
        family_id,
        true);
    cudaCheck(cudaGetLastError());

    // Build the large-side projection Q*phase (or phase*Q) into the phase
    // panel, then turn normalized into a bounded horizontal residual.  The
    // phase scratch is intentionally overwritten only after the tangent
    // factor has been formed above.
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        phase,
        packed_a,
        matrix_count,
        small_elements,
        panel_stride,
        panel_stride);
    cudaCheck(cudaGetLastError());
    if (tall) {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_a,
            CUBLAS_OP_N,
            side,
            static_cast<long long>(panel_stride),
            packed_b,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            phase,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    } else {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_b,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            packed_a,
            CUBLAS_OP_N,
            side,
            static_cast<long long>(panel_stride),
            phase,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    }
    llmc_normuon_batch_project_rectangular_residual_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        normalized,
        phase,
        runtime->nonfinite_flag,
        matrix_count,
        matrix_elements,
        panel_stride,
        rows,
        columns,
        LLMC_NORMUON_RECTANGULAR_HORIZONTAL_CAP,
        config->epsilon);
    cudaCheck(cudaGetLastError());

    if (thin_canonical_stage2) {
        // Thin retraction: the second canonical Taylor stage acts on the
        // small tangent factor before the single large Q*B / B*Q product.
        // The horizontal residual is already live in normalized and remains
        // outside this factor retraction; the post-product spectral guard
        // below bounds the combined candidate.
        // Guard the factor first as well: unlike the full post-product path,
        // this small polynomial sees B directly and must be kept inside its
        // Taylor basin before it is multiplied by Q.
        llmc_normuon_batch_add_rectangular_residual_pack_kernel<<<
            small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            correction,
            nullptr,
            packed_a,
            runtime->nonfinite_flag,
            matrix_count,
            small_elements,
            panel_stride);
        cudaCheck(cudaGetLastError());
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_a,
            CUBLAS_OP_N,
            side,
            static_cast<long long>(panel_stride),
            packed_a,
            CUBLAS_OP_T,
            side,
            static_cast<long long>(panel_stride),
            phase,
            side,
            static_cast<long long>(panel_stride),
            side,
            side,
            side,
            matrix_count);
        llmc_normuon_batch_guard_rectangular_gram_kernel<<<
            static_cast<uint32_t>(matrix_count),
            LLMC_NORMUON_BLOCK_SIZE,
            0,
            stream>>>(
            correction,
            phase,
            runtime->stats,
            runtime->nonfinite_flag,
            matrix_count,
            small_elements,
            panel_stride,
            side,
            LLMC_NORMUON_RECTANGULAR_RETRACTION_PMAX,
            config->epsilon);
        cudaCheck(cudaGetLastError());
        if (!llmc_normuon_apply_polynomial_batched_bf16(
                runtime,
                handle,
                stream,
                correction,
                phase,
                side,
                matrix_count,
                1U,
                config->correction_schedule + 1U,
                nullptr)) {
            return false;
        }
    }

    // Repack the tangent factor after borrowing packed_a for the projection.
    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        correction,
        packed_a,
        matrix_count,
        small_elements,
        panel_stride,
        panel_stride);
    cudaCheck(cudaGetLastError());

    // Write the candidate into the correction panel after B has been packed;
    // this reuses the existing three-panel tracker workspace without a fourth
    // large FP32 allocation.
    if (tall) {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_a,
            CUBLAS_OP_N,
            side,
            static_cast<long long>(panel_stride),
            packed_b,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            correction,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    } else {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_b,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            packed_a,
            CUBLAS_OP_N,
            side,
            static_cast<long long>(panel_stride),
            correction,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    }
    // Keep the post-product polynomial in its stable basin.  The residual
    // add is fused with the BF16 pack, then a Tensor-Core Gram GEMM supplies a
    // conservative Gershgorin spectral upper bound without any zero-order
    // hold fallback.
    llmc_normuon_batch_add_rectangular_residual_pack_kernel<<<
        matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        correction,
        normalized,
        packed_a,
        runtime->nonfinite_flag,
        matrix_count,
        matrix_elements,
        panel_stride);
    cudaCheck(cudaGetLastError());
    if (tall) {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_a,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            packed_a,
            CUBLAS_OP_T,
            columns,
            static_cast<long long>(panel_stride),
            phase,
            side,
            static_cast<long long>(panel_stride),
            columns,
            columns,
            rows,
            matrix_count);
    } else {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_a,
            CUBLAS_OP_T,
            columns,
            static_cast<long long>(panel_stride),
            packed_a,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            phase,
            side,
            static_cast<long long>(panel_stride),
            rows,
            rows,
            columns,
            matrix_count);
    }
    llmc_normuon_batch_guard_rectangular_gram_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        correction,
        phase,
        runtime->stats,
        runtime->nonfinite_flag,
        matrix_count,
        matrix_elements,
        panel_stride,
        side,
        LLMC_NORMUON_RECTANGULAR_RETRACTION_PMAX,
        config->epsilon);
    cudaCheck(cudaGetLastError());

    if (commute_canonical_stage2) {
        if (!llmc_normuon_apply_rectangular_polynomial_batched_bf16(
                runtime,
                handle,
                stream,
                correction,
                phase,
                rows,
                columns,
                matrix_count,
                1U,
                config->correction_schedule + 1U)) {
            return false;
        }
    } else if (thin_canonical_stage2) {
        // The thin factor was retracted before the large product above.
    } else if (config->retraction_mode ==
               LLMC_NORMUON_TRACKER_RETRACTION_NEWTON_SCHULZ) {
        if (!llmc_normuon_apply_rectangular_polynomial_batched_bf16(
                runtime,
                handle,
                stream,
                correction,
                phase,
                rows,
                columns,
                matrix_count,
                1U,
                kLlmcRectangularCubicRetraction)) {
            return false;
        }
    }
    *direction_out = correction;
    return true;
}

inline bool llmc_cachemuon_direction_batched_bf16(
    LlmcNormuonRuntime* runtime,
    cublasHandle_t handle,
    cudaStream_t stream,
    const LlmcNormuonConfig* config,
    const LlmcOptimizerParameterType* parameter_type,
    int rows,
    int columns,
    int matrix_count,
    const float** direction_out,
    float** refreshed_transform_out,
    int* miss_count_out) {
    if (runtime == nullptr || handle == nullptr || config == nullptr ||
        parameter_type == nullptr || direction_out == nullptr ||
        refreshed_transform_out == nullptr || miss_count_out == nullptr ||
        rows <= 0 || columns <= 0 || matrix_count <= 0 ||
        runtime->tracked_q == nullptr || runtime->cache_residuals == nullptr ||
        runtime->cache_miss_indices == nullptr ||
        runtime->cache_host_residuals == nullptr ||
        runtime->cache_host_miss_indices == nullptr) {
        return false;
    }
    const int side = rows < columns ? rows : columns;
    const bool tall = rows >= columns;
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const size_t small_elements = static_cast<size_t>(side) * side;
    const size_t total_elements =
        static_cast<size_t>(matrix_count) * matrix_elements;
    const size_t total_small =
        static_cast<size_t>(matrix_count) * small_elements;
    const size_t panel_stride = runtime->matrix_elements;
    const uint32_t matrix_grid = llmc_normuon_grid_for_count(total_elements);
    const uint32_t small_grid = llmc_normuon_grid_for_count(total_small);
    float* normalized = runtime->matrix[0];
    float* candidate = runtime->matrix[1];
    float* miss_workspace = runtime->matrix[2];
    uint16_t* packed_matrix = runtime->batch_bf16[0];
    uint16_t* packed_transform = runtime->batch_bf16[1];

    llmc_normuon_batch_pack_strided_bf16_kernel<<<
        matrix_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        normalized,
        packed_matrix,
        matrix_count,
        matrix_elements,
        panel_stride,
        panel_stride);
    llmc_normuon_batch_pack_q_bf16_kernel<<<
        small_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
        runtime->tracked_q,
        packed_transform,
        total_small,
        small_elements,
        parameter_type->family_id,
        true);
    cudaCheck(cudaGetLastError());
    if (tall) {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_transform,
            CUBLAS_OP_T,
            side,
            static_cast<long long>(small_elements),
            packed_matrix,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            candidate,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    } else {
        llmc_normuon_batched_gemm_bf16_ex(
            handle,
            packed_matrix,
            CUBLAS_OP_N,
            columns,
            static_cast<long long>(panel_stride),
            packed_transform,
            CUBLAS_OP_N,
            side,
            static_cast<long long>(small_elements),
            candidate,
            columns,
            static_cast<long long>(panel_stride),
            columns,
            rows,
            side,
            matrix_count);
    }
    llmc_cachemuon_rectangular_gram_batched_bf16(
        runtime,
        handle,
        stream,
        candidate,
        runtime->cache_small[0],
        rows,
        columns,
        matrix_count,
        panel_stride);
    llmc_cachemuon_residual_kernel<<<
        static_cast<uint32_t>(matrix_count),
        LLMC_NORMUON_BLOCK_SIZE,
        0,
        stream>>>(
        runtime->cache_small[0],
        runtime->cache_residuals,
        matrix_count,
        small_elements,
        side);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpyAsync(
        runtime->cache_host_residuals,
        runtime->cache_residuals,
        static_cast<size_t>(matrix_count) * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream));
    // CacheGNS has a data-dependent branch.  This is the deliberate host/GPU
    // boundary needed to compact only per-matrix misses into the fresh batch;
    // accepted matrices never pay the FreshGNS products.
    cudaCheck(cudaStreamSynchronize(stream));

    int miss_count = 0;
    double residual_sum = 0.0;
    float residual_max = 0.0f;
    for (int matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        const int layer_index =
            matrix_index / parameter_type->views_per_layer;
        const int view_index =
            matrix_index - layer_index * parameter_type->views_per_layer;
        const size_t q_index = llmc_normuon_q_view_index(
            parameter_type, layer_index, view_index, true);
        if (q_index >= runtime->tracked_q_view_count) {
            return false;
        }
        const float residual = runtime->cache_host_residuals[matrix_index];
        if (isfinite(residual)) {
            residual_sum += static_cast<double>(residual);
            residual_max = fmaxf(residual_max, residual);
        }
        if (runtime->q_valid[q_index] == 0U || !isfinite(residual) ||
            residual > config->cache_residual_threshold) {
            runtime->cache_host_miss_indices[miss_count++] = matrix_index;
        }
    }
    runtime->cache_step_probe_count += static_cast<uint64_t>(matrix_count);
    runtime->cache_step_miss_count += static_cast<uint64_t>(miss_count);
    runtime->cache_step_residual_sum += residual_sum;
    runtime->cache_step_residual_max =
        fmaxf(runtime->cache_step_residual_max, residual_max);
    runtime->cache_total_probe_count += static_cast<uint64_t>(matrix_count);
    runtime->cache_total_miss_count += static_cast<uint64_t>(miss_count);
    runtime->cache_total_residual_sum += residual_sum;
    runtime->cache_total_residual_max =
        fmaxf(runtime->cache_total_residual_max, residual_max);

    float* refreshed_transform = nullptr;
    if (miss_count > 0) {
        cudaCheck(cudaMemcpyAsync(
            runtime->cache_miss_indices,
            runtime->cache_host_miss_indices,
            static_cast<size_t>(miss_count) * sizeof(int),
            cudaMemcpyHostToDevice,
            stream));
        const size_t miss_elements =
            static_cast<size_t>(miss_count) * matrix_elements;
        const uint32_t miss_grid = llmc_normuon_grid_for_count(miss_elements);
        llmc_cachemuon_batch_gather_selected_kernel<<<
            miss_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            normalized,
            miss_workspace,
            runtime->cache_miss_indices,
            miss_count,
            matrix_elements,
            panel_stride);
        cudaCheck(cudaGetLastError());
        float* fresh_direction = nullptr;
        if (!llmc_cachemuon_fresh_gns_batched_bf16(
                runtime,
                handle,
                stream,
                miss_workspace,
                normalized,
                rows,
                columns,
                miss_count,
                panel_stride,
                &fresh_direction,
                &refreshed_transform)) {
            return false;
        }
        llmc_cachemuon_batch_scatter_selected_kernel<<<
            miss_grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            fresh_direction,
            candidate,
            runtime->cache_miss_indices,
            miss_count,
            matrix_elements,
            panel_stride);
        cudaCheck(cudaGetLastError());
    }
    *direction_out = candidate;
    *refreshed_transform_out = refreshed_transform;
    *miss_count_out = miss_count;
    return true;
}

inline void llmc_cachemuon_commit_batched_refreshes(
    LlmcNormuonRuntime* runtime,
    const LlmcOptimizerParameterType* parameter_type,
    uint64_t global_step,
    int miss_count) {
    for (int compact_index = 0; compact_index < miss_count; ++compact_index) {
        const int matrix_index =
            runtime->cache_host_miss_indices[compact_index];
        const int layer_index =
            matrix_index / parameter_type->views_per_layer;
        const int view_index =
            matrix_index - layer_index * parameter_type->views_per_layer;
        const size_t q_index = llmc_normuon_q_view_index(
            parameter_type, layer_index, view_index, true);
        runtime->q_valid[q_index] = 1U;
        runtime->refresh_count[q_index]++;
        runtime->last_refresh_step[q_index] =
            static_cast<int64_t>(global_step);
    }
}

inline bool llmc_normuon_batch_tracker_refresh_state(
    const LlmcNormuonRuntime* runtime,
    const LlmcOptimizerParameterType* parameter_type,
    uint64_t global_step,
    uint32_t refresh_interval,
    bool* needs_refresh,
    bool rectangular = false) {
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
                parameter_type, layer_index, view_index, rectangular);
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
    bool refreshed,
    bool rectangular = false) {
    for (int layer_index = 0;
         layer_index < parameter_type->layer_multiplicity;
         ++layer_index) {
        for (int view_index = 0;
             view_index < parameter_type->views_per_layer;
             ++view_index) {
            const size_t q_index = llmc_normuon_q_view_index(
                parameter_type, layer_index, view_index, rectangular);
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
        parameter_type->views_per_layer !=
            llmc_normuon_family_views_per_matrix(
                config->orthogonalization_mode, parameter_type->family_id) ||
        (parameter_type->family_id != LLMC_OPTIMIZER_FAMILY_MLP_WUP &&
         parameter_type->family_id != LLMC_OPTIMIZER_FAMILY_MLP_WDOWN) ||
        !(learning_rate > 0.0f)) {
        return false;
    }
    const int width = static_cast<int>(parameter_type->matrix_width);
    const LlmcNormuonOrthogonalizationMode family_mode =
        llmc_normuon_mode_for_family(
            config->orthogonalization_mode, parameter_type->family_id);
    const bool rectangular = llmc_normuon_is_rectangular_mode(family_mode);
    const bool rectangular_tracker =
        family_mode == LLMC_NORMUON_ORTHO_RECTANGULAR_SKEW_POLAR_TRACK_Q;
    const bool rectangular_cache =
        family_mode == LLMC_NORMUON_ORTHO_RECTANGULAR_CACHE_MUON;
    // Split mode shares one over-sized workspace allocation, but the active
    // family still needs its native panel stride.  Switching this stride at
    // the family boundary keeps square Wup GEMMs contiguous while retaining
    // the four-d^2 rectangular stride for Wdown scratch updates.
    runtime->matrix_elements =
        rectangular ? 4U * static_cast<size_t>(width) * width
                    : static_cast<size_t>(width) * width;
    const int rows = rectangular
        ? (parameter_type->family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
               ? 4 * width
               : width)
        : width;
    const int columns = rectangular
        ? (parameter_type->family_id == LLMC_OPTIMIZER_FAMILY_MLP_WUP
               ? width
               : 4 * width)
        : width;
    const float learning_rate_multiplier =
        parameter_type->family_id == LLMC_OPTIMIZER_FAMILY_MLP_WDOWN
            ? config->wdown_learning_rate_multiplier
            : 1.0f;
    const int matrix_count = parameter_type->layer_multiplicity *
                             parameter_type->views_per_layer;
    if (width <= 0 || matrix_count <= 0 ||
        static_cast<size_t>(matrix_count) > runtime->batch_matrix_capacity) {
        return false;
    }
    const size_t matrix_elements = static_cast<size_t>(rows) * columns;
    const size_t total_elements =
        static_cast<size_t>(matrix_count) * matrix_elements;
    if (total_elements > runtime->batch_total_elements ||
        runtime->batch_bf16[0] == nullptr || runtime->batch_bf16[1] == nullptr) {
        return false;
    }
    cublasCheck(cublasSetStream(handle, stream));
    llmc_normuon_reset_guard(runtime, stream);
    if (rectangular) {
        llmc_normuon_batch_prepare_rectangular_momentum_kernel<<<
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
            rows,
            columns,
            config->momentum,
            gradient_scale);
    } else {
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
    }
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
        llmc_normuon_is_cache_mode(family_mode) ? 1.0f : 1.02f,
        llmc_normuon_is_cache_mode(family_mode) ? LLMC_CACHEMUON_EPSILON
                                                : config->epsilon);
    cudaCheck(cudaGetLastError());

    const float* direction = runtime->matrix[0];
    bool refreshed = false;
    float* cache_refreshed_transform = nullptr;
    int cache_miss_count = 0;
    if (rectangular && !rectangular_tracker && !rectangular_cache) {
        if (!llmc_normuon_apply_rectangular_polynomial_batched_bf16(
                runtime,
                handle,
                stream,
                runtime->matrix[0],
                runtime->matrix[1],
                rows,
                columns,
                matrix_count,
                LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                config->refresh_schedule)) {
            return false;
        }
    } else if (rectangular_cache) {
        if (!llmc_cachemuon_direction_batched_bf16(
                runtime,
                handle,
                stream,
                config,
                parameter_type,
                rows,
                columns,
                matrix_count,
                &direction,
                &cache_refreshed_transform,
                &cache_miss_count)) {
            return false;
        }
    } else if (rectangular_tracker) {
        if (!llmc_normuon_batch_tracker_refresh_state(
                runtime,
                parameter_type,
                global_step,
                config->refresh_interval,
                &refreshed,
                true)) {
            return false;
        }
        if (refreshed) {
            if (!llmc_normuon_apply_rectangular_polynomial_batched_bf16(
                    runtime,
                    handle,
                    stream,
                    runtime->matrix[0],
                    runtime->matrix[1],
                    rows,
                    columns,
                    matrix_count,
                    LLMC_NORMUON_POLYNOMIAL_STAGE_COUNT,
                    config->refresh_schedule)) {
                return false;
            }
            llmc_normuon_batch_copy_direction_to_q_kernel<<<
                grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
                runtime->matrix[0],
                runtime->tracked_q,
                total_elements,
                matrix_elements,
                runtime->matrix_elements,
                parameter_type->family_id,
                true);
            cudaCheck(cudaGetLastError());
        } else if (!llmc_normuon_rectangular_tracker_direction_batched_bf16(
                       runtime,
                       handle,
                       stream,
                       config,
                       parameter_type->family_id,
                       width,
                       rows,
                       columns,
                       matrix_count,
                       &direction)) {
            return false;
        }
    } else if (family_mode == LLMC_NORMUON_ORTHO_NEWTON_SCHULZ) {
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

    if (rectangular) {
        llmc_normuon_batch_second_moment_rectangular_kernel<<<
            static_cast<uint32_t>(matrix_count),
            LLMC_NORMUON_BLOCK_SIZE,
            0,
            stream>>>(
            direction,
            second_moment,
            runtime->axis_stats,
            runtime->nonfinite_flag,
            matrix_count,
            runtime->matrix_elements,
            width,
            rows,
            columns,
            config->beta2,
            config->epsilon);
    } else {
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
    }
    cudaCheck(cudaGetLastError());
    if (rectangular) {
        llmc_normuon_batch_validate_rectangular_update_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            master,
            direction,
            runtime->axis_stats,
            runtime->nonfinite_flag,
            total_elements,
            matrix_elements,
            runtime->matrix_elements,
            width,
            rows,
            columns,
            learning_rate,
            config->weight_decay,
            config->update_scale,
            learning_rate_multiplier);
    } else {
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
            config->update_scale,
            learning_rate_multiplier);
    }
    cudaCheck(cudaGetLastError());
    if (!llmc_normuon_guard_ok(runtime, stream)) {
        return false;
    }
    if (rectangular_cache && cache_miss_count > 0) {
        if (cache_refreshed_transform == nullptr) {
            return false;
        }
        const size_t small_elements =
            static_cast<size_t>(width) * width;
        const size_t total_refreshed =
            static_cast<size_t>(cache_miss_count) * small_elements;
        llmc_cachemuon_batch_scatter_transform_kernel<<<
            llmc_normuon_grid_for_count(total_refreshed),
            LLMC_NORMUON_BLOCK_SIZE,
            0,
            stream>>>(
            cache_refreshed_transform,
            runtime->tracked_q,
            runtime->cache_miss_indices,
            cache_miss_count,
            small_elements,
            parameter_type->family_id);
        cudaCheck(cudaGetLastError());
        llmc_cachemuon_commit_batched_refreshes(
            runtime, parameter_type, global_step, cache_miss_count);
    }
    const bool update_tracker =
        llmc_normuon_is_tracker_mode(family_mode);
    if (update_tracker) {
        llmc_normuon_batch_commit_tracker_metadata(
            runtime,
            parameter_type,
            global_step,
            refreshed,
            rectangular_tracker);
    }
    if (rectangular) {
        llmc_normuon_batch_apply_rectangular_update_kernel<<<
            grid, LLMC_NORMUON_BLOCK_SIZE, 0, stream>>>(
            parameter,
            master,
            direction,
            runtime->axis_stats,
            total_elements,
            matrix_elements,
            runtime->matrix_elements,
            width,
            rows,
            columns,
            learning_rate,
            config->weight_decay,
            config->update_scale,
            learning_rate_multiplier,
            global_step,
            parameter_type->tensor_id,
            rectangular_tracker ? runtime->tracked_q : nullptr,
            parameter_type->family_id,
            rectangular_tracker);
    } else {
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
            learning_rate_multiplier,
            global_step,
            parameter_type->tensor_id);
    }
    cudaCheck(cudaGetLastError());
    return true;
}

#endif // LLMC_NORMUON_BATCHED_CUH
