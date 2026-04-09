// Rotary Position Embedding (RoPE) CUDA kernels.
//
// Two variants:
//   1. Halved RoPE (Qwen2, GPT-NeoX style): split into halves
//   2. Interleaved RoPE (Llama, GPT-J style): alternating pairs
//
// Fused with QKV split when possible (saves one kernel launch).
// Supports position offset for continuous batching.

#include "common.cuh"
#include <math.h>

// ─── Halved RoPE ─────────────────────────────────────────────
// x_out[..., :d/2] = x[..., :d/2] * cos - x[..., d/2:] * sin
// x_out[..., d/2:] = x[..., :d/2] * sin + x[..., d/2:] * cos
//
// cos, sin: [max_seq_len, rope_dim/2] precomputed frequency table

extern "C" __global__ void rope_halved_f32(
    float* __restrict__ x,              // [n_tokens, n_heads, head_dim] — in-place
    const float* __restrict__ cos_table, // [max_seq_len, rope_dim/2]
    const float* __restrict__ sin_table, // [max_seq_len, rope_dim/2]
    const int* __restrict__ positions,   // [n_tokens] position IDs (for continuous batching)
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int rope_dim,                  // typically == head_dim
    const int pos_offset,                // global position offset (if positions is NULL)
    const int* __restrict__ pos_offset_gpu // GPU scalar: overrides pos_offset when non-NULL (graph-replay-safe)
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= n_tokens) return;

    const int half = rope_dim / 2;
    const int effective_offset = pos_offset_gpu ? *pos_offset_gpu : pos_offset;
    const int pos = positions ? positions[token_idx] : (token_idx + effective_offset);

    float* head_ptr = x + (token_idx * n_heads + head_idx) * head_dim;
    const float* cos_ptr = cos_table + pos * half;
    const float* sin_ptr = sin_table + pos * half;

    for (int i = tid; i < half; i += blockDim.x) {
        float x0 = head_ptr[i];
        float x1 = head_ptr[i + half];
        float c = cos_ptr[i];
        float s = sin_ptr[i];

        head_ptr[i]        = x0 * c - x1 * s;
        head_ptr[i + half] = x0 * s + x1 * c;
    }
}

// ─── Interleaved RoPE ────────────────────────────────────────
// For dimension pairs (2i, 2i+1):
// x_out[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
// x_out[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]

extern "C" __global__ void rope_interleaved_f32(
    float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int rope_dim,
    const int pos_offset,
    const int* __restrict__ pos_offset_gpu // GPU scalar: overrides pos_offset when non-NULL (graph-replay-safe)
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= n_tokens) return;

    const int n_pairs = rope_dim / 2;
    const int effective_offset = pos_offset_gpu ? *pos_offset_gpu : pos_offset;
    const int pos = positions ? positions[token_idx] : (token_idx + effective_offset);

    float* head_ptr = x + (token_idx * n_heads + head_idx) * head_dim;
    const float* cos_ptr = cos_table + pos * n_pairs;
    const float* sin_ptr = sin_table + pos * n_pairs;

    for (int i = tid; i < n_pairs; i += blockDim.x) {
        float x0 = head_ptr[2 * i];
        float x1 = head_ptr[2 * i + 1];
        float c = cos_ptr[i];
        float s = sin_ptr[i];

        head_ptr[2 * i]     = x0 * c - x1 * s;
        head_ptr[2 * i + 1] = x0 * s + x1 * c;
    }
}

// ─── Precompute RoPE Frequency Table ─────────────────────────
// cos_table[pos, i] = cos(pos * theta_i)
// sin_table[pos, i] = sin(pos * theta_i)
// where theta_i = 1 / (base^(2i/rope_dim))

extern "C" __global__ void rope_precompute_freqs(
    float* __restrict__ cos_table,    // [max_seq_len, rope_dim/2]
    float* __restrict__ sin_table,    // [max_seq_len, rope_dim/2]
    const int max_seq_len,
    const int rope_dim,
    const float rope_base              // typically 10000.0 or 1000000.0
) {
    const int pos = blockIdx.x;
    const int tid = threadIdx.x;
    const int half = rope_dim / 2;

    if (pos >= max_seq_len) return;

    for (int i = tid; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(rope_base, (2.0f * i) / rope_dim);
        float angle = pos * freq;
        cos_table[pos * half + i] = cosf(angle);
        sin_table[pos * half + i] = sinf(angle);
    }
}
