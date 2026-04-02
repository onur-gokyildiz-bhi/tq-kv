// GPU token sampling — sort-free top-k/top-p.
//
// Based on FlashInfer's rejection sampling approach:
//   Dual-pivot binary search, O(log(1/eps)) rounds.
//   50%+ faster than vLLM sampling, 10x faster than torch.topk.
//
// For single-request decode, CPU sampling is fine.
// This kernel matters for batched serving (continuous batching).

#include "common.cuh"

// ─── Argmax (greedy) ─────────────────────────────────────────
// Find the index of the maximum value. One block per row.

extern "C" __global__ void argmax_f32(
    const float* __restrict__ logits,    // [n_rows, vocab_size]
    int* __restrict__ indices,           // [n_rows] output
    const int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x = logits + row * vocab_size;

    __shared__ float s_max_val[32];
    __shared__ int s_max_idx[32];

    float local_max = -1e30f;
    int local_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (x[i] > local_max) {
            local_max = x[i];
            local_idx = i;
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        s_max_val[warp_id] = local_max;
        s_max_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        local_max = (lane < (blockDim.x >> 5)) ? s_max_val[lane] : -1e30f;
        local_idx = (lane < (blockDim.x >> 5)) ? s_max_idx[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane == 0) {
            indices[row] = local_idx;
        }
    }
}

// ─── Temperature + Softmax (for top-k/top-p) ─────────────────
// Apply temperature scaling and convert logits to probabilities.
// In-place operation: logits → probabilities.

extern "C" __global__ void apply_temperature_softmax_f32(
    float* __restrict__ logits,          // [n_rows, vocab_size] — in-place
    const int vocab_size,
    const float temperature
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float* x = logits + row * vocab_size;
    float inv_temp = 1.0f / fmaxf(temperature, 1e-7f);

    // Apply temperature
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        x[i] *= inv_temp;
    }
    __syncthreads();

    // Softmax
    float local_max = -1e10f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float global_max = block_reduce_max(local_max);
    __shared__ float s_max, s_sum;
    if (tid == 0) s_max = global_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = expf(x[i] - s_max);
        x[i] = val;
        local_sum += val;
    }
    float global_sum = block_reduce_sum(local_sum);
    if (tid == 0) s_sum = global_sum;
    __syncthreads();

    float inv_sum = 1.0f / fmaxf(s_sum, 1e-10f);
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        x[i] *= inv_sum;
    }
}

// ─── Sort-free Top-P (Nucleus) Sampling ───────────────────────
// Uses rejection sampling: pick random index, accept if within top-p.
// Binary search on cumulative probability threshold.
//
// Algorithm:
//   1. Compute softmax probabilities (above kernel)
//   2. Find threshold t such that sum of probs >= t equals p
//   3. Sample from distribution, rejecting below-threshold tokens
//
// For small vocab or single requests, CPU is fine.
// This kernel is for batched serving throughput.

extern "C" __global__ void top_p_threshold_f32(
    const float* __restrict__ probs,     // [vocab_size] softmax output
    float* __restrict__ threshold_out,    // [1] output threshold
    const int vocab_size,
    const float top_p
) {
    // Binary search for the probability threshold
    // This is a cooperative kernel: all threads work on one row

    const int tid = threadIdx.x;

    // Find max prob (the pivot)
    float max_prob = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        max_prob = fmaxf(max_prob, probs[i]);
    }
    max_prob = block_reduce_max(max_prob);

    __shared__ float s_threshold;
    if (tid == 0) s_threshold = 0.0f;  // start with no threshold
    __syncthreads();

    // Binary search: find min threshold where sum(probs >= threshold) <= top_p
    float lo = 0.0f, hi = max_prob;
    for (int iter = 0; iter < 32; ++iter) {  // 32 iterations = 2^-32 precision
        float mid = (lo + hi) * 0.5f;

        // Count probability mass above mid
        float mass_above = 0.0f;
        for (int i = tid; i < vocab_size; i += blockDim.x) {
            if (probs[i] >= mid) mass_above += probs[i];
        }
        mass_above = block_reduce_sum(mass_above);

        __shared__ float s_mass;
        if (tid == 0) s_mass = mass_above;
        __syncthreads();

        if (s_mass > top_p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    if (tid == 0) {
        threshold_out[0] = lo;
    }
}
