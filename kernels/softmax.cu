// Online softmax with block-parallel reduction.
//
// Two variants:
//   1. Standard softmax_last_dim — for attention scores and logits
//   2. Safe softmax with causal mask — for attention with padding/mask
//
// Uses Milakov & Gimelshein (2018) online algorithm:
//   Single pass: maintain running max and sum-of-exp.

#include "common.cuh"

// ─── Softmax Last Dim ─────────────────────────────────────────
// input:  [n_rows, n_cols]
// output: [n_rows, n_cols]
// Each block handles one row.

extern "C" __global__ void softmax_last_dim_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n_rows,
    const int n_cols
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const float* x = input + row * n_cols;
    float* o = output + row * n_cols;

    // Phase 1: Find max
    float local_max = -1e10f;
    for (int i = tid; i < n_cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float row_max = block_reduce_max(local_max);

    // Broadcast max
    __shared__ float s_max, s_sum;
    if (tid == 0) s_max = row_max;
    __syncthreads();

    // Phase 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += blockDim.x) {
        float val = expf(x[i] - s_max);
        o[i] = val;  // store exp temporarily
        local_sum += val;
    }
    float row_sum = block_reduce_sum(local_sum);
    if (tid == 0) s_sum = row_sum;
    __syncthreads();

    // Phase 3: Normalize
    float inv_sum = (s_sum > 0.0f) ? 1.0f / s_sum : 0.0f;
    for (int i = tid; i < n_cols; i += blockDim.x) {
        o[i] *= inv_sum;
    }
}

// ─── Online Softmax (single pass, for very long sequences) ───
// Maintains running max and sum-of-exp in a single pass.
// Better for extremely long sequences (100K+ tokens) where
// the 3-pass approach above is memory-bandwidth limited.

extern "C" __global__ void softmax_online_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n_rows,
    const int n_cols
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int tid = threadIdx.x;
    const float* x = input + row * n_cols;
    float* o = output + row * n_cols;

    // Online softmax: single pass tracking max and sum
    float running_max = -1e10f;
    float running_sum = 0.0f;

    for (int i = tid; i < n_cols; i += blockDim.x) {
        float val = x[i];
        float new_max = fmaxf(running_max, val);
        running_sum = running_sum * expf(running_max - new_max) + expf(val - new_max);
        running_max = new_max;
    }

    // Block reduce max (need all threads to agree on global max)
    float global_max = block_reduce_max(running_max);
    __shared__ float s_global_max;
    if (tid == 0) s_global_max = global_max;
    __syncthreads();

    // Rescale each thread's sum to global max
    float rescaled_sum = running_sum * expf(running_max - s_global_max);
    float global_sum = block_reduce_sum(rescaled_sum);
    __shared__ float s_global_sum;
    if (tid == 0) s_global_sum = global_sum;
    __syncthreads();

    // Write normalized output
    float inv_sum = (s_global_sum > 0.0f) ? 1.0f / s_global_sum : 0.0f;
    for (int i = tid; i < n_cols; i += blockDim.x) {
        o[i] = expf(x[i] - s_global_max) * inv_sum;
    }
}
