// FlashDecoding v2: Split-KV decode for long-context inference.
//
// v2 improvements over v1:
//   - 128 threads/block (4 warps) for memory bandwidth utilization
//   - __ldg() read-only cache hints for all K/V/Q global reads
//   - Warp-reduce + broadcast pattern: 1 __syncthreads per KV token (was 2)
//   - Configurable split_size via kernel parameter
//
// Splits KV cache across multiple thread blocks for parallelism.
// Each block computes partial attention over a KV chunk.
// A reduction kernel combines partial results with online softmax rescaling.
//
// Reference: Dao et al. "Flash-Decoding for long-context inference" (2023)

#include "common.cuh"

#define HEAD_DIM_MAX 256
#define WARP_SIZE 32
#define BLOCK_SIZE 128   // 4 warps — balance between parallelism and occupancy

// ─── Phase 1: Partial Attention per KV Chunk ─────────────────
// Grid: (n_splits, n_heads, batch), Block: BLOCK_SIZE threads
// Each block processes split_size KV tokens with online softmax.

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
    const int split_size,
    const int max_seq,                    // stride for KV buffer
    const int window_size                 // sliding window: 0 = global, >0 = attend last N
) {
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.x;
    const int n_rep     = n_heads / n_kv_heads;
    const int kv_head   = head_idx / n_rep;

    // Apply sliding window: restrict to last window_size tokens
    const int global_start = (window_size > 0 && seq_kv > window_size) ? (seq_kv - window_size) : 0;
    const int kv_start = split_idx * split_size;
    const int effective_start = max(kv_start, global_start);
    const int effective_end = min(kv_start + split_size, seq_kv);
    const int kv_len = effective_end - effective_start;
    if (kv_len <= 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid >> 5;
    const int n_warps = BLOCK_SIZE >> 5;  // 4

    // Load query into shared memory via __ldg() (read-only texture cache)
    __shared__ float s_q[HEAD_DIM_MAX];
    const float* q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        s_q[d] = __ldg(&q_ptr[d]);
    }
    __syncthreads();

    // Per-thread accumulators
    float local_max = -1e10f;
    float local_sum = 0.0f;
    const int n_d_per_thread = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float local_o[HEAD_DIM_MAX / BLOCK_SIZE + 1];
    for (int i = 0; i < n_d_per_thread; ++i) local_o[i] = 0.0f;

    const float* k_base = K + ((batch_idx * n_kv_heads + kv_head) * max_seq + effective_start) * head_dim;
    const float* v_base = V + ((batch_idx * n_kv_heads + kv_head) * max_seq + effective_start) * head_dim;

    // Shared memory for warp-level dot product broadcast
    // Pattern: warp reduce → shared write → __syncthreads → all threads read
    // Saves 1 __syncthreads per KV token vs block_reduce_sum + separate broadcast
    __shared__ float s_warp_dot[8];  // max 8 warps

    for (int ki = 0; ki < kv_len; ++ki) {
        // ── Q·K[ki] dot product with __ldg() for K reads ──
        float dot = 0.0f;
        const float* k_ptr = k_base + ki * head_dim;
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            dot += s_q[d] * __ldg(&k_ptr[d]);
        }

        // Warp-level reduce + cross-warp broadcast (1 sync total)
        dot = warp_reduce_sum(dot);
        if (lane == 0) s_warp_dot[warp_id] = dot;
        __syncthreads();
        // All threads compute total score (n_warps additions, no extra sync needed)
        float score = 0.0f;
        for (int w = 0; w < n_warps; ++w) score += s_warp_dot[w];
        score *= scale;

        // ── Online softmax update ──
        float old_max = local_max;
        float new_max = fmaxf(old_max, score);
        float rescale = expf(old_max - new_max);
        float p = expf(score - new_max);

        // ── Accumulate V weighted by attention probability ──
        const float* v_ptr = v_base + ki * head_dim;
        for (int i = 0; i < n_d_per_thread; ++i) {
            int d = i * BLOCK_SIZE + tid;
            if (d < head_dim) {
                local_o[i] = local_o[i] * rescale + p * __ldg(&v_ptr[d]);
            }
        }
        local_sum = local_sum * rescale + p;
        local_max = new_max;
    }

    // Write partial results
    const int n_splits = (seq_kv + split_size - 1) / split_size;
    float* po = partial_O + ((batch_idx * n_heads + head_idx) * n_splits + split_idx) * head_dim;
    for (int i = 0; i < n_d_per_thread; ++i) {
        int d = i * BLOCK_SIZE + tid;
        if (d < head_dim) {
            po[d] = local_o[i];
        }
    }
    if (tid == 0) {
        partial_max[(batch_idx * n_heads + head_idx) * n_splits + split_idx] = local_max;
        partial_sum[(batch_idx * n_heads + head_idx) * n_splits + split_idx] = local_sum;
    }
}

// ─── Phase 2: Reduce Partial Results ─────────────────────────
// Grid: (1, n_heads, batch), Block: BLOCK_SIZE threads
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

    // Find global max across splits (via __ldg read-only cache)
    float global_max = -1e10f;
    for (int s = 0; s < n_splits; ++s) {
        global_max = fmaxf(global_max, __ldg(&partial_max[base + s]));
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
        float rescale = expf(__ldg(&partial_max[base + s]) - global_max);
        float w = __ldg(&partial_sum[base + s]) * rescale;
        total_sum += w;

        const float* po = partial_O + (base + s) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            o_ptr[d] += __ldg(&po[d]) * rescale;
        }
    }
    __syncthreads();

    // Normalize
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 1.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        o_ptr[d] *= inv_sum;
    }
}
