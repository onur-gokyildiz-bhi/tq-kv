// FlashDecoding: Split-KV decode for long-context inference.
//
// Splits KV cache across multiple thread blocks for parallelism.
// Each block computes partial attention over a KV chunk.
// A reduction kernel combines partial results with online softmax rescaling.
//
// Benefits: O(1) blocks per Q head regardless of seq_len → scales to 128K+ context.
// Use when seq_len > 256 (below that, single-block gqa_decode is faster).
//
// Reference: Dao et al. "Flash-Decoding for long-context inference" (2023)

#include "common.cuh"

#define HEAD_DIM_MAX 256
#define WARP_SIZE 32

// ─── Phase 1: Partial Attention per KV Chunk ─────────────────
// Grid: (n_splits, n_heads, batch), Block: 32 threads
// Each block processes split_size KV tokens with online softmax.
// Outputs: partial_O (unnormalized), partial_max, partial_sum.

extern "C" __global__ void flash_decode_partial(
    const float* __restrict__ Q,          // [B, H, 1, D]
    const float* __restrict__ K,          // [B, Hkv, max_seq, D]
    const float* __restrict__ V,          // [B, Hkv, max_seq, D]
    float* __restrict__ partial_O,        // [B, H, n_splits, D]
    float* __restrict__ partial_max,      // [B, H, n_splits]
    float* __restrict__ partial_sum,      // [B, H, n_splits]
    const int batch_size,
    const int n_heads,
    const int n_kv_heads,
    const int seq_kv,
    const int head_dim,
    const float scale,
    const int split_size,                 // KV tokens per split
    const int max_seq                     // stride for KV buffer (may be > seq_kv)
) {
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.x;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);

    const int kv_start = split_idx * split_size;
    const int kv_len   = min(split_size, seq_kv - kv_start);
    if (kv_len <= 0) return;

    const int tid = threadIdx.x;

    // Load query (single token)
    __shared__ float s_q[HEAD_DIM_MAX];
    const float* q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q_ptr[d];
    }
    __syncthreads();

    // Compute attention scores and accumulate O for this split
    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_o[HEAD_DIM_MAX / WARP_SIZE + 1];  // per-thread partial O
    for (int d = 0; d < (head_dim + blockDim.x - 1) / blockDim.x; ++d) {
        local_o[d] = 0.0f;
    }

    const float* k_base = K + ((batch_idx * n_kv_heads + kv_head) * max_seq + kv_start) * head_dim;
    const float* v_base = V + ((batch_idx * n_kv_heads + kv_head) * max_seq + kv_start) * head_dim;

    for (int ki = 0; ki < kv_len; ++ki) {
        // Dot product Q·K[ki]
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot += s_q[d] * k_base[ki * head_dim + d];
        }
        dot = warp_reduce_sum(dot);
        // Broadcast to all threads
        __shared__ float s_dot;
        if (tid == 0) s_dot = dot * scale;
        __syncthreads();
        dot = s_dot;

        // Online softmax update
        float old_max = local_max;
        float new_max = fmaxf(old_max, dot);
        float rescale = expf(old_max - new_max);
        float p = expf(dot - new_max);

        // Rescale old + add new
        for (int d_idx = 0; d_idx < (head_dim + blockDim.x - 1) / blockDim.x; ++d_idx) {
            int d = d_idx * blockDim.x + tid;
            if (d < head_dim) {
                local_o[d_idx] = local_o[d_idx] * rescale + p * v_base[ki * head_dim + d];
            }
        }
        local_sum = local_sum * rescale + p;
        local_max = new_max;
    }

    // Write partial results
    const int n_splits = (seq_kv + split_size - 1) / split_size;
    float* po = partial_O + ((batch_idx * n_heads + head_idx) * n_splits + split_idx) * head_dim;
    for (int d_idx = 0; d_idx < (head_dim + blockDim.x - 1) / blockDim.x; ++d_idx) {
        int d = d_idx * blockDim.x + tid;
        if (d < head_dim) {
            po[d] = local_o[d_idx];
        }
    }
    if (tid == 0) {
        partial_max[(batch_idx * n_heads + head_idx) * n_splits + split_idx] = local_max;
        partial_sum[(batch_idx * n_heads + head_idx) * n_splits + split_idx] = local_sum;
    }
}

// ─── Phase 2: Reduce Partial Results ─────────────────────────
// Grid: (1, n_heads, batch), Block: 32 threads
// Combines partial results from all splits with online softmax rescaling.

extern "C" __global__ void flash_decode_reduce(
    const float* __restrict__ partial_O,    // [B, H, n_splits, D]
    const float* __restrict__ partial_max,   // [B, H, n_splits]
    const float* __restrict__ partial_sum,   // [B, H, n_splits]
    float* __restrict__ O,                   // [B, H, 1, D]
    const int n_heads,
    const int n_splits,
    const int head_dim
) {
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int tid = threadIdx.x;

    const int base = (batch_idx * n_heads + head_idx) * n_splits;

    // Find global max across splits
    float global_max = -1e10f;
    for (int s = 0; s < n_splits; ++s) {
        global_max = fmaxf(global_max, partial_max[base + s]);
    }

    // Combine partial sums and outputs with rescaling
    float total_sum = 0.0f;
    float* o_ptr = O + (batch_idx * n_heads + head_idx) * head_dim;

    // Initialize output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        o_ptr[d] = 0.0f;
    }
    __syncthreads();

    for (int s = 0; s < n_splits; ++s) {
        float rescale = expf(partial_max[base + s] - global_max);
        float w = partial_sum[base + s] * rescale;
        total_sum += w;

        const float* po = partial_O + (base + s) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            o_ptr[d] += po[d] * rescale;
        }
    }
    __syncthreads();

    // Normalize
    for (int d = tid; d < head_dim; d += blockDim.x) {
        o_ptr[d] /= (total_sum > 0.0f ? total_sum : 1.0f);
    }
}
