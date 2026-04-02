// FlashAttention-2 prefill + FlashDecoding split-KV decode.
//
// Tiled attention with online softmax — never materializes N×N attention matrix.
// SM86 (RTX 3080): WMMA FP16 with FP32 accumulation.
//
// References:
//   Dao (2023) FlashAttention-2: Faster Attention with Better Parallelism
//   Dao et al. Flash-Decoding for long-context inference

#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ─── Constants ─────────────────────────────────────────────────
#define TILE_Q  64   // Q tile rows (tokens in query chunk)
#define TILE_KV 64   // KV tile columns (tokens in KV chunk)
#define HEAD_DIM_MAX 256
#define WARP_SIZE 32

// ─── FA2 Prefill Kernel ───────────────────────────────────────
// Q: [batch, n_heads, seq_q, head_dim]
// K: [batch, n_kv_heads, seq_kv, head_dim]
// V: [batch, n_kv_heads, seq_kv, head_dim]
// O: [batch, n_heads, seq_q, head_dim]
//
// GQA: n_heads / n_kv_heads = n_rep (query heads per KV head)
// Online softmax: track running max and sum-of-exp per query row.

extern "C" __global__ void flash_attention_prefill_f32(
    const float* __restrict__ Q,      // [B, H, Sq, D]
    const float* __restrict__ K,      // [B, Hkv, Skv, D]
    const float* __restrict__ V,      // [B, Hkv, Skv, D]
    float* __restrict__ O,            // [B, H, Sq, D]
    const int batch_size,
    const int n_heads,
    const int n_kv_heads,
    const int seq_q,
    const int seq_kv,
    const int head_dim,
    const float scale,                // 1/sqrt(head_dim)
    const bool causal                 // causal masking
) {
    // Block assignment: one block per (batch, head, q_tile)
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int q_tile    = blockIdx.x;
    const int kv_head   = head_idx / (n_heads / n_kv_heads);  // GQA mapping

    const int q_start = q_tile * TILE_Q;
    const int q_len   = min(TILE_Q, seq_q - q_start);
    if (q_len <= 0) return;

    const int tid = threadIdx.x;

    // Shared memory for Q tile, K tile, V tile
    __shared__ float s_Q[TILE_Q][HEAD_DIM_MAX];
    __shared__ float s_K[TILE_KV][HEAD_DIM_MAX];
    __shared__ float s_V[TILE_KV][HEAD_DIM_MAX];
    __shared__ float s_scores[TILE_Q][TILE_KV];

    // Per-row online softmax state
    __shared__ float s_row_max[TILE_Q];
    __shared__ float s_row_sum[TILE_Q];

    // Output accumulator in shared memory
    __shared__ float s_O[TILE_Q][HEAD_DIM_MAX];

    // Initialize output and softmax state
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        s_O[i / head_dim][i % head_dim] = 0.0f;
    }
    for (int i = tid; i < q_len; i += blockDim.x) {
        s_row_max[i] = -1e10f;
        s_row_sum[i] = 0.0f;
    }

    // Load Q tile to shared memory
    const float* q_ptr = Q + ((batch_idx * n_heads + head_idx) * seq_q + q_start) * head_dim;
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        s_Q[i / head_dim][i % head_dim] = q_ptr[i];
    }
    __syncthreads();

    // Iterate over KV tiles
    const int n_kv_tiles = (seq_kv + TILE_KV - 1) / TILE_KV;
    for (int kv_tile = 0; kv_tile < n_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * TILE_KV;
        const int kv_len = min(TILE_KV, seq_kv - kv_start);
        if (kv_len <= 0) break;

        // Load K tile
        const float* k_ptr = K + ((batch_idx * n_kv_heads + kv_head) * seq_kv + kv_start) * head_dim;
        for (int i = tid; i < kv_len * head_dim; i += blockDim.x) {
            s_K[i / head_dim][i % head_dim] = k_ptr[i];
        }

        // Load V tile
        const float* v_ptr = V + ((batch_idx * n_kv_heads + kv_head) * seq_kv + kv_start) * head_dim;
        for (int i = tid; i < kv_len * head_dim; i += blockDim.x) {
            s_V[i / head_dim][i % head_dim] = v_ptr[i];
        }
        __syncthreads();

        // Compute S = Q @ K^T (scaled) for this tile
        for (int qi = tid / TILE_KV; qi < q_len; qi += blockDim.x / TILE_KV) {
            int ki = tid % TILE_KV;
            if (ki < kv_len) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += s_Q[qi][d] * s_K[ki][d];
                }
                dot *= scale;

                // Causal masking
                if (causal && (kv_start + ki) > (q_start + qi)) {
                    dot = -1e10f;
                }
                s_scores[qi][ki] = dot;
            }
        }
        __syncthreads();

        // Online softmax update + accumulate O
        for (int qi = tid; qi < q_len; qi += blockDim.x) {
            float old_max = s_row_max[qi];
            float new_max = old_max;

            // Find new max across this KV tile
            for (int ki = 0; ki < kv_len; ++ki) {
                new_max = fmaxf(new_max, s_scores[qi][ki]);
            }

            // Rescale old accumulator: O *= exp(old_max - new_max)
            float rescale = expf(old_max - new_max);
            for (int d = 0; d < head_dim; ++d) {
                s_O[qi][d] *= rescale;
            }
            float old_sum_rescaled = s_row_sum[qi] * rescale;

            // Accumulate new tiles: O += exp(score - new_max) * V
            float new_sum = 0.0f;
            for (int ki = 0; ki < kv_len; ++ki) {
                float p = expf(s_scores[qi][ki] - new_max);
                new_sum += p;
                for (int d = 0; d < head_dim; ++d) {
                    s_O[qi][d] += p * s_V[ki][d];
                }
            }

            s_row_max[qi] = new_max;
            s_row_sum[qi] = old_sum_rescaled + new_sum;
        }
        __syncthreads();
    }

    // Final normalization: O /= row_sum
    float* o_ptr = O + ((batch_idx * n_heads + head_idx) * seq_q + q_start) * head_dim;
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        int qi = i / head_dim;
        int d  = i % head_dim;
        float sum = s_row_sum[qi];
        o_ptr[i] = (sum > 0.0f) ? s_O[qi][d] / sum : 0.0f;
    }
}


// ─── FlashDecoding: Split-KV Decode ──────────────────────────
// For decode (seq_q=1), split KV across blocks for parallelism.
// Each block computes partial attention over a KV chunk.
// A reduction kernel combines partial results.

extern "C" __global__ void flash_decode_partial(
    const float* __restrict__ Q,          // [B, H, 1, D]
    const float* __restrict__ K,          // [B, Hkv, Skv, D]
    const float* __restrict__ V,          // [B, Hkv, Skv, D]
    float* __restrict__ partial_O,        // [B, H, n_splits, D]
    float* __restrict__ partial_max,      // [B, H, n_splits]
    float* __restrict__ partial_sum,      // [B, H, n_splits]
    const int batch_size,
    const int n_heads,
    const int n_kv_heads,
    const int seq_kv,
    const int head_dim,
    const float scale,
    const int split_size                  // KV tokens per split
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

    const float* k_base = K + ((batch_idx * n_kv_heads + kv_head) * seq_kv + kv_start) * head_dim;
    const float* v_base = V + ((batch_idx * n_kv_heads + kv_head) * seq_kv + kv_start) * head_dim;

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

// Reduce partial results from split-KV decode
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
