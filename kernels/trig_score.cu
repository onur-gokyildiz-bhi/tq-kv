// TriAttention: Trigonometric KV cache importance scoring on GPU.
//
// Based on arXiv:2604.04921 (Mao et al., 2026).
// Scores pre-RoPE keys using trigonometric series approximation
// of attention — no full Q@K computation needed.
//
// S(k, Δ) = S_trig(k, Δ) + S_norm(k)
// S_trig = Σ_f ||E[q_f]|| * ||k_f|| * cos(omega_f * Δ + phi_f)
// S_norm = Σ_f (1 - R) * E[||q||] * ||k_f|| / n_pairs
//
// Multi-offset averaging: S̃(k) = mean over D = {1,2,4,...,2^16}
//
// Grid: (n_kv_heads, 1, 1)
// Block: 256 threads
// Each thread scores ceil(n_keys / blockDim.x) keys.

#include "common_v2.cuh"

#define MAX_HEAD_DIM 256
#define MAX_OFFSETS 17  // {2^0, 2^1, ..., 2^16}

// ─── TriAttention scoring kernel ─────────────────────────────
//
// Scores all keys for one KV head. For GQA, caller handles
// multi-head aggregation on CPU (z-score normalize + max).
//
// q_center: pre-RoPE Q center for one head [head_dim]
// keys_pre_rope: pre-RoPE keys [n_keys, head_dim]
// rope_freqs: omega_f = 1/theta^(2f/d) [n_pairs]
// key_positions: position of each key [n_keys]
// scores_out: importance score per key [n_keys]
// mrl: Mean Resultant Length for this Q head (scalar)
// q_norm_mean: E[||q||] for this Q head (scalar)
// current_pos: current generation position
// n_keys: number of keys to score
// head_dim: key dimension
// n_offsets: number of multi-offset entries (17 for {1..2^16})
// offsets: geometric offset set [n_offsets]

extern "C" __global__ void trig_score_keys_f32(
    const float* __restrict__ q_center,       // [head_dim]
    const float* __restrict__ keys_pre_rope,  // [n_keys, head_dim]
    const float* __restrict__ rope_freqs,     // [n_pairs]
    const int* __restrict__ key_positions,    // [n_keys]
    float* __restrict__ scores_out,           // [n_keys]
    const float mrl,
    const float q_norm_mean,
    const int current_pos,
    const int n_keys,
    const int head_dim,
    const int n_offsets,
    const int* __restrict__ offsets           // [n_offsets]
) {
    const int tid = threadIdx.x;
    const int n_pairs = head_dim / 2;

    // Load Q center into shared memory
    __shared__ float s_qc[MAX_HEAD_DIM];
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_qc[i] = q_center[i];
    }

    // Load rope frequencies into shared memory
    __shared__ float s_freq[MAX_HEAD_DIM / 2];
    for (int i = tid; i < n_pairs; i += blockDim.x) {
        s_freq[i] = rope_freqs[i];
    }

    // Load offsets into shared memory
    __shared__ int s_offsets[MAX_OFFSETS];
    if (tid < n_offsets) {
        s_offsets[tid] = offsets[tid];
    }
    __syncthreads();

    // Pre-compute Q center norms and phases per frequency pair
    // qc_norm[f] = sqrt(qc[2f]^2 + qc[2f+1]^2)
    // qc_phase[f] = atan2(qc[2f+1], qc[2f])
    __shared__ float s_qc_norm[MAX_HEAD_DIM / 2];
    __shared__ float s_qc_phase[MAX_HEAD_DIM / 2];
    for (int f = tid; f < n_pairs; f += blockDim.x) {
        float re = s_qc[2 * f];
        float im = s_qc[2 * f + 1];
        s_qc_norm[f] = sqrtf(re * re + im * im);
        s_qc_phase[f] = atan2f(im, re);
    }
    __syncthreads();

    // Each thread handles multiple keys
    // (blockIdx.x = kv_head, used by batched variant; caller offsets keys)

    for (int ki = tid; ki < n_keys; ki += blockDim.x) {
        const float* key = keys_pre_rope + ki * head_dim;
        int base_distance = current_pos - key_positions[ki];

        float total_score = 0.0f;

        // Multi-offset averaging
        for (int oi = 0; oi < n_offsets; oi++) {
            float delta = (float)(base_distance + s_offsets[oi]);
            float s_trig = 0.0f;
            float s_norm = 0.0f;

            for (int f = 0; f < n_pairs; f++) {
                float k_re = key[2 * f];
                float k_im = key[2 * f + 1];
                float k_norm = sqrtf(k_re * k_re + k_im * k_im);

                if (s_qc_norm[f] < 1e-12f || k_norm < 1e-12f) continue;

                // Phase difference
                float k_phase = atan2f(k_im, k_re);
                float phi = s_qc_phase[f] - k_phase;

                // S_trig += ||E[q_f]|| * ||k_f|| * cos(omega_f * delta + phi_f)
                float angle = s_freq[f] * delta + phi;
                s_trig += s_qc_norm[f] * k_norm * cosf(angle);

                // S_norm contribution (accumulated across pairs)
                s_norm += k_norm;
            }

            // S_norm = (1 - R) * E[||q||] * mean(||k_f||)
            s_norm = (1.0f - mrl) * q_norm_mean * s_norm / (float)n_pairs;

            total_score += s_trig + s_norm;
        }

        scores_out[ki] = total_score / (float)n_offsets;
    }
}

// ─── Batched version: score all KV heads in one launch ───────
// Grid: (n_kv_heads, 1, 1)
// Block: 256 threads
// Same algorithm but reads per-head parameters from arrays.

extern "C" __global__ void trig_score_keys_batched_f32(
    const float* __restrict__ q_centers,       // [n_kv_heads, head_dim] (one Q center per KV head, pre-mapped)
    const float* __restrict__ keys_pre_rope,   // [n_kv_heads, n_keys, head_dim]
    const float* __restrict__ rope_freqs,      // [n_pairs]
    const int* __restrict__ key_positions,     // [n_keys] (shared across heads)
    float* __restrict__ scores_out,            // [n_kv_heads, n_keys]
    const float* __restrict__ mrl_per_head,    // [n_kv_heads]
    const float* __restrict__ q_norm_per_head, // [n_kv_heads]
    const int current_pos,
    const int n_keys,
    const int head_dim,
    const int n_offsets,
    const int* __restrict__ offsets            // [n_offsets]
) {
    const int kv_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_pairs = head_dim / 2;

    const float mrl = mrl_per_head[kv_head];
    const float q_norm_mean = q_norm_per_head[kv_head];

    // Load this head's Q center
    __shared__ float s_qc[MAX_HEAD_DIM];
    const float* qc = q_centers + kv_head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_qc[i] = qc[i];
    }

    __shared__ float s_freq[MAX_HEAD_DIM / 2];
    for (int i = tid; i < n_pairs; i += blockDim.x) {
        s_freq[i] = rope_freqs[i];
    }

    __shared__ int s_offsets[MAX_OFFSETS];
    if (tid < n_offsets) {
        s_offsets[tid] = offsets[tid];
    }
    __syncthreads();

    // Pre-compute Q center norms and phases
    __shared__ float s_qc_norm[MAX_HEAD_DIM / 2];
    __shared__ float s_qc_phase[MAX_HEAD_DIM / 2];
    for (int f = tid; f < n_pairs; f += blockDim.x) {
        float re = s_qc[2 * f];
        float im = s_qc[2 * f + 1];
        s_qc_norm[f] = sqrtf(re * re + im * im);
        s_qc_phase[f] = atan2f(im, re);
    }
    __syncthreads();

    // Score keys for this head
    const float* head_keys = keys_pre_rope + kv_head * n_keys * head_dim;
    float* head_scores = scores_out + kv_head * n_keys;

    for (int ki = tid; ki < n_keys; ki += blockDim.x) {
        const float* key = head_keys + ki * head_dim;
        int base_distance = current_pos - key_positions[ki];

        float total_score = 0.0f;

        for (int oi = 0; oi < n_offsets; oi++) {
            float delta = (float)(base_distance + s_offsets[oi]);
            float s_trig = 0.0f;
            float s_norm = 0.0f;

            for (int f = 0; f < n_pairs; f++) {
                float k_re = key[2 * f];
                float k_im = key[2 * f + 1];
                float k_norm = sqrtf(k_re * k_re + k_im * k_im);

                if (s_qc_norm[f] < 1e-12f || k_norm < 1e-12f) continue;

                float k_phase = atan2f(k_im, k_re);
                float phi = s_qc_phase[f] - k_phase;
                float angle = s_freq[f] * delta + phi;
                s_trig += s_qc_norm[f] * k_norm * cosf(angle);
                s_norm += k_norm;
            }

            s_norm = (1.0f - mrl) * q_norm_mean * s_norm / (float)n_pairs;
            total_score += s_trig + s_norm;
        }

        head_scores[ki] = total_score / (float)n_offsets;
    }
}
