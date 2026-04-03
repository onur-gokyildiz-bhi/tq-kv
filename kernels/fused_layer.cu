// Fused transformer layer CUDA kernels — minimize kernel launch overhead.
//
// Strategy: combine multiple ops that share the same input vector into single
// kernel launches. Each block independently computes RmsNorm in shared memory,
// then uses the normalized result for its Q4_K_M matvec row.
//
// Kernel 1: fused_norm_q4km_qkv_bias — RmsNorm + QKV projection + bias
// Kernel 3: fused_addnorm_q4km_gateup_silu — residual add + RmsNorm + gate/up + SiLU*mul
// Kernel 4: fused_q4km_down_residual — down projection + residual add

#include "common.cuh"
#include <cuda_fp16.h>

// Q4_K_M constants (from qmatmul.cu)
#define QK_K 256
#define Q4K_BLOCK_SIZE 144

__device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* scales, uint8_t* sc, uint8_t* m
) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >> 4)  | ((scales[j] >> 6) << 4);
    }
}

// ─── Shared RmsNorm + Q4_K_M dot product ─────────────────────
// Every block independently computes RmsNorm of input → shared memory,
// then computes one Q4_K_M matvec row from the normalized input.
// Redundant norm computation per block (~200 FLOPs) is negligible vs matvec.

__device__ float q4km_dot_from_shared(
    const uint8_t* __restrict__ w_row,
    const float* __restrict__ s_normed,  // shared memory: normalized input [hidden_dim]
    const int in_features,
    const int tid,
    const int block_dim
) {
    const int n_superblocks = in_features / QK_K;
    float partial_sum = 0.0f;

    for (int sb = tid; sb < n_superblocks; sb += block_dim) {
        const uint8_t* block = w_row + sb * Q4K_BLOCK_SIZE;
        const float* x_sb = s_normed + sb * QK_K;

        uint16_t d_bits  = block[0] | (block[1] << 8);
        uint16_t dm_bits = block[2] | (block[3] << 8);
        float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));

        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;

        for (int grp = 0; grp < 4; ++grp) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * grp,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * grp + 1, scales, &sc_hi, &m_hi);
            float d_lo  = d * (float)sc_lo;
            float d_hi  = d * (float)sc_hi;
            float dm_lo = dmin * (float)m_lo;
            float dm_hi = dmin * (float)m_hi;
            int q_off = grp * 32;
            int x_off = grp * 64;
            for (int l = 0; l < 32; ++l) {
                partial_sum += (d_lo * (float)(qs[q_off + l] & 0xF) - dm_lo) * x_sb[x_off + l];
            }
            for (int l = 0; l < 32; ++l) {
                partial_sum += (d_hi * (float)(qs[q_off + l] >> 4) - dm_hi) * x_sb[x_off + 32 + l];
            }
        }
    }

    partial_sum = block_reduce_sum(partial_sum);
    return partial_sum;
}

// ─── Kernel 1: Fused Norm + QKV Projection + Bias ────────────
// Grid: (q_out + k_out + v_out) blocks, 256 threads
// Each block: compute RmsNorm(input) → shared, then Q4_K_M dot for its output row

extern "C" __global__ void fused_norm_q4km_qkv_bias_f32(
    const float* __restrict__ input,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ W_q,
    const uint8_t* __restrict__ W_k,
    const uint8_t* __restrict__ W_v,
    const float* __restrict__ bias_q,    // NULL if no bias
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    const int hidden_dim,
    const int q_out,
    const int k_out,
    const int v_out,
    const float eps
) {
    extern __shared__ float s_normed[];  // [hidden_dim]
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    // Phase 1: RmsNorm(input) → shared memory (redundant per block, cheap)
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = input[i];
        sum_sq += val * val;
        s_normed[i] = val;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / (float)hidden_dim + eps);
    }
    __syncthreads();

    // Normalize in shared memory
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_normed[i] = s_normed[i] * s_rms_inv * norm_weight[i];
    }
    __syncthreads();

    // Phase 2: Q4_K_M matvec from shared memory
    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    if (row < q_out) {
        // Q projection
        const uint8_t* w_row = W_q + row * bytes_per_row;
        float result = q4km_dot_from_shared(w_row, s_normed, hidden_dim, tid, blockDim.x);
        if (tid == 0) {
            out_q[row] = result + (bias_q ? bias_q[row] : 0.0f);
        }
    } else if (row < q_out + k_out) {
        // K projection
        int k_row = row - q_out;
        const uint8_t* w_row = W_k + k_row * bytes_per_row;
        float result = q4km_dot_from_shared(w_row, s_normed, hidden_dim, tid, blockDim.x);
        if (tid == 0) {
            out_k[k_row] = result + (bias_k ? bias_k[k_row] : 0.0f);
        }
    } else {
        // V projection
        int v_row = row - q_out - k_out;
        const uint8_t* w_row = W_v + v_row * bytes_per_row;
        float result = q4km_dot_from_shared(w_row, s_normed, hidden_dim, tid, blockDim.x);
        if (tid == 0) {
            out_v[v_row] = result + (bias_v ? bias_v[v_row] : 0.0f);
        }
    }
}

// ─── Kernel 3: Fused Add+Norm + Gate/Up Projection + SiLU*Mul ─
// Grid: intermediate_dim blocks, 256 threads
// Each block: compute (attn_out + residual), norm, gate+up matvec, silu*mul

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_f32(
    const float* __restrict__ attn_out,
    float* __restrict__ residual,          // updated in-place (idempotent writes)
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ W_gate,
    const uint8_t* __restrict__ W_up,
    float* __restrict__ intermediate_out,
    const int hidden_dim,
    const int intermediate_dim,
    const float eps
) {
    extern __shared__ float s_normed[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= intermediate_dim) return;

    // Phase 1: residual += attn_out, then RmsNorm → shared memory
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = attn_out[i] + residual[i];
        residual[i] = val;  // idempotent: all blocks write same value
        sum_sq += val * val;
        s_normed[i] = val;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / (float)hidden_dim + eps);
    }
    __syncthreads();

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_normed[i] = s_normed[i] * s_rms_inv * norm_weight[i];
    }
    __syncthreads();

    // Phase 2: gate and up matvec from shared normalized input
    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const uint8_t* gate_row = W_gate + row * bytes_per_row;
    float gate_val = q4km_dot_from_shared(gate_row, s_normed, hidden_dim, tid, blockDim.x);

    // Need to sync before reusing shared memory for second reduction
    __syncthreads();

    const uint8_t* up_row = W_up + row * bytes_per_row;
    float up_val = q4km_dot_from_shared(up_row, s_normed, hidden_dim, tid, blockDim.x);

    // Phase 3: SiLU(gate) * up
    if (tid == 0) {
        float silu_gate = gate_val / (1.0f + expf(-gate_val));
        intermediate_out[row] = silu_gate * up_val;
    }
}

// ─── Kernel 4: Fused Down Projection + Residual Add ──────────
// Grid: hidden_dim blocks, 256 threads

extern "C" __global__ void fused_q4km_down_residual_f32(
    const uint8_t* __restrict__ W_down,
    const float* __restrict__ intermediate,
    float* __restrict__ residual,
    const int hidden_dim,
    const int intermediate_dim
) {
    const int row = blockIdx.x;
    if (row >= hidden_dim) return;
    const int tid = threadIdx.x;

    // Q4_K_M dot product: W_down[row] @ intermediate
    const int n_superblocks = intermediate_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const uint8_t* w_row = W_down + row * bytes_per_row;

    float partial_sum = 0.0f;
    for (int sb = tid; sb < n_superblocks; sb += blockDim.x) {
        const uint8_t* block = w_row + sb * Q4K_BLOCK_SIZE;
        const float* x_sb = intermediate + sb * QK_K;

        uint16_t d_bits  = block[0] | (block[1] << 8);
        uint16_t dm_bits = block[2] | (block[3] << 8);
        float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;

        for (int grp = 0; grp < 4; ++grp) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * grp, scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * grp + 1, scales, &sc_hi, &m_hi);
            float d_lo = d * (float)sc_lo, d_hi = d * (float)sc_hi;
            float dm_lo = dmin * (float)m_lo, dm_hi = dmin * (float)m_hi;
            int q_off = grp * 32, x_off = grp * 64;
            for (int l = 0; l < 32; ++l)
                partial_sum += (d_lo * (float)(qs[q_off + l] & 0xF) - dm_lo) * x_sb[x_off + l];
            for (int l = 0; l < 32; ++l)
                partial_sum += (d_hi * (float)(qs[q_off + l] >> 4) - dm_hi) * x_sb[x_off + 32 + l];
        }
    }

    partial_sum = block_reduce_sum(partial_sum);
    if (tid == 0) {
        residual[row] += partial_sum;
    }
}
