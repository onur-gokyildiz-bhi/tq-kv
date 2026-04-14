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

// Q4K_M dot product with FULL 256-thread utilization.
//
// Previous version: thread-per-superblock striding → only 14/256 threads active (5.5%).
// New version: 256 threads map 1:1 to superblock positions, iterate over superblocks.
// Same pattern as standalone q4km_matvec_f32 — all threads active every iteration.
//
// Thread mapping within 256-element superblock:
//   tid[0..63]:   group 0 low nibbles  (positions 0-63)
//   tid[64..127]: group 1 low nibbles  (positions 64-127)
//   tid[128..191]: group 2 low nibbles (positions 128-191)
//   tid[192..255]: group 3 low nibbles (positions 192-255)
//   Within each 64-thread chunk: first 32 = low nibble, last 32 = high nibble
__device__ float q4km_dot_from_shared(
    const uint8_t* __restrict__ w_row,
    const float* __restrict__ s_normed,  // shared memory: normalized input [hidden_dim]
    const int in_features,
    const int tid,
    const int block_dim
) {
    const int n_superblocks = in_features / QK_K;
    const int grp   = tid >> 6;       // 0-3: which 64-element group
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;       // 0 = low nibble, 1 = high nibble
    const int l     = pos & 31;       // byte index within group (0-31)
    const int x_idx = grp * 64 + pos; // position in 256-element superblock

    float partial_sum = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        const uint8_t* block = w_row + sb * Q4K_BLOCK_SIZE;
        float x_val = s_normed[sb * QK_K + x_idx];

        uint16_t d_bits  = block[0] | (block[1] << 8);
        uint16_t dm_bits = block[2] | (block[3] << 8);
        float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));

        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;

        uint8_t sc_val, m_val;
        get_scale_min_k4(2 * grp + is_hi, scales, &sc_val, &m_val);

        uint8_t q_byte = qs[grp * 32 + l];
        float nibble = is_hi ? (float)(q_byte >> 4) : (float)(q_byte & 0xF);
        partial_sum += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
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

// ─── Kernel 3: Fused Norm + Gate/Up Projection + SiLU*Mul ─────
// Grid: intermediate_dim blocks, 256 threads
// Each block: compute RmsNorm(input) → shared, then gate+up matvec, silu*mul
// NOTE: input should be pre-combined (residual + attn_out) by caller.
// Previous version had a race condition writing to residual across blocks.

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_f32(
    const float* __restrict__ input,       // pre-combined: residual + attn_out
    const float* __restrict__ _unused,     // kept for ABI compat, ignored
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

    // Phase 1: RmsNorm(input) → shared memory
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

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_normed[i] = s_normed[i] * s_rms_inv * norm_weight[i];
    }
    __syncthreads();

    // Phase 2: DUAL gate+up dot product in single loop (x_val shared, halves iterations)
    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    const uint8_t* gate_row = W_gate + row * bytes_per_row;
    const uint8_t* up_row   = W_up   + row * bytes_per_row;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        float x_val = s_normed[sb * QK_K + x_idx];

        // Gate weight dequant
        const uint8_t* gblk = gate_row + sb * Q4K_BLOCK_SIZE;
        uint16_t gd_bits  = gblk[0] | (gblk[1] << 8);
        uint16_t gdm_bits = gblk[2] | (gblk[3] << 8);
        float gd    = __half2float(*reinterpret_cast<const __half*>(&gd_bits));
        float gdmin = __half2float(*reinterpret_cast<const __half*>(&gdm_bits));
        const uint8_t* gsc = gblk + 4;
        uint8_t g_sc, g_m;
        if (j < 4) { g_sc = gsc[j] & 63; g_m = gsc[j+4] & 63; }
        else { g_sc = (gsc[j+4]&0xF)|((gsc[j-4]>>6)<<4); g_m = (gsc[j+4]>>4)|((gsc[j]>>6)<<4); }
        uint8_t g_byte = gblk[16 + grp * 32 + l];
        float g_nib = is_hi ? (float)(g_byte >> 4) : (float)(g_byte & 0xF);
        gate_sum += (gd * (float)g_sc * g_nib - gdmin * (float)g_m) * x_val;

        // Up weight dequant (same x_val — no re-read from shared mem)
        const uint8_t* ublk = up_row + sb * Q4K_BLOCK_SIZE;
        uint16_t ud_bits  = ublk[0] | (ublk[1] << 8);
        uint16_t udm_bits = ublk[2] | (ublk[3] << 8);
        float ud    = __half2float(*reinterpret_cast<const __half*>(&ud_bits));
        float udmin = __half2float(*reinterpret_cast<const __half*>(&udm_bits));
        const uint8_t* usc = ublk + 4;
        uint8_t u_sc, u_m;
        if (j < 4) { u_sc = usc[j] & 63; u_m = usc[j+4] & 63; }
        else { u_sc = (usc[j+4]&0xF)|((usc[j-4]>>6)<<4); u_m = (usc[j+4]>>4)|((usc[j]>>6)<<4); }
        uint8_t u_byte = ublk[16 + grp * 32 + l];
        float u_nib = is_hi ? (float)(u_byte >> 4) : (float)(u_byte & 0xF);
        up_sum += (ud * (float)u_sc * u_nib - udmin * (float)u_m) * x_val;
    }

    // Single reduction pass for both gate and up
    gate_sum = block_reduce_sum(gate_sum);
    __syncthreads();
    up_sum = block_reduce_sum(up_sum);

    // Phase 3: SiLU(gate) * up
    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        intermediate_out[row] = silu_gate * up_sum;
    }
}

// ─── Kernel 1-cpasync: QKV fused with cp.async weight prefetch ──
// Same math as fused_norm_q4km_qkv_bias_f32 but replaces the
// q4km_dot_from_shared helper with an inline loop that pre-loads each
// next Q/K/V super-block (144 B) via cp.async while computing the
// current super-block's dequant+FMA. X stays in s_normed from RmsNorm.
//
// Shared layout: s_normed[hidden_dim] + s_wbuf[2][36] = hidden_dim*4 + 288 B.

extern "C" __global__ void fused_norm_q4km_qkv_bias_cpasync_f32(
    const float* __restrict__ input,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ W_q,
    const uint8_t* __restrict__ W_k,
    const uint8_t* __restrict__ W_v,
    const float* __restrict__ bias_q,
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
    extern __shared__ float s_mem[];
    float* s_normed = s_mem;              // [hidden_dim]
    float* s_wbuf   = s_mem + hidden_dim; // [2][36] = 288 B

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    // Phase 1: RmsNorm(input) → shared memory (same as baseline)
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
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_normed[i] = s_normed[i] * s_rms_inv * norm_weight[i];
    }
    __syncthreads();

    // Pick weight matrix + output slot + bias based on row range.
    const uint8_t* W_row_base;
    float* out;
    const float* bias;
    int out_row;
    if (row < q_out) {
        W_row_base = W_q;
        out = out_q;
        bias = bias_q;
        out_row = row;
    } else if (row < q_out + k_out) {
        W_row_base = W_k;
        out = out_k;
        bias = bias_k;
        out_row = row - q_out;
    } else {
        W_row_base = W_v;
        out = out_v;
        bias = bias_v;
        out_row = row - q_out - k_out;
    }

    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const uint8_t* w_row = W_row_base + out_row * bytes_per_row;

    // Thread mapping (identical to q4km_dot_from_shared).
    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;

    auto prefetch_w = [&](int sb) {
        const float* src =
            reinterpret_cast<const float*>(w_row + sb * Q4K_BLOCK_SIZE);
        float* dst = s_wbuf + (sb & 1) * 36;
        if (tid < 36) {
            tq_cp_async_f32(&dst[tid], &src[tid]);
        }
        tq_cp_async_commit();
    };

    if (n_superblocks > 0) prefetch_w(0);

    float partial_sum = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_w(sb + 1);
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        const uint8_t* block = reinterpret_cast<const uint8_t*>(s_wbuf + (sb & 1) * 36);

        uint16_t d_bits  = block[0] | (block[1] << 8);
        uint16_t dm_bits = block[2] | (block[3] << 8);
        float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));

        const uint8_t* scales = block + 4;
        const uint8_t* qs     = block + 16;

        uint8_t sc_val, m_val;
        get_scale_min_k4(2 * grp + is_hi, scales, &sc_val, &m_val);

        uint8_t q_byte = qs[grp * 32 + l];
        float nibble = is_hi ? (float)(q_byte >> 4) : (float)(q_byte & 0xF);
        partial_sum += (d * (float)sc_val * nibble - dmin * (float)m_val) * s_normed[sb * QK_K + x_idx];

        __syncthreads();  // protect buffer before next prefetch
    }

    float result = block_reduce_sum(partial_sum);
    if (tid == 0) {
        out[out_row] = result + (bias ? bias[out_row] : 0.0f);
    }
}

// ─── Kernel 3-cpasync: cp.async double-buffered weight pipeline ──
// Same math as fused_addnorm_q4km_gateup_silu_f32, but pre-loads each
// next superblock's gate+up weight blocks into shared memory via
// cp.async while computing the current superblock. Targets the MLP
// gateup bottleneck (~52% of per-layer time on Qwen2 7B).
//
// Shared-memory layout (on top of s_normed[hidden_dim]):
//   s_wbuf[2][72]  double-buffer weights, 72 floats = 288 bytes per
//                  superblock = gate_block(144B) + up_block(144B)
// Extra shmem cost: 576 bytes per block.

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_cpasync_f32(
    const float* __restrict__ input,
    const float* __restrict__ _unused,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ W_gate,
    const uint8_t* __restrict__ W_up,
    float* __restrict__ intermediate_out,
    const int hidden_dim,
    const int intermediate_dim,
    const float eps
) {
    extern __shared__ float s_mem[];
    float* s_normed = s_mem;                // [hidden_dim]
    float* s_wbuf   = s_mem + hidden_dim;   // [2 * 72] = 576 bytes

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= intermediate_dim) return;

    // Phase 1: RmsNorm(input) → shared memory
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
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_normed[i] = s_normed[i] * s_rms_inv * norm_weight[i];
    }
    __syncthreads();

    // Phase 2: gate+up pipelined dot
    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const int FLOATS_PER_BLOCK = Q4K_BLOCK_SIZE / 4;   // 144/4 = 36 floats per gate/up block
    const int SB_FLOATS = FLOATS_PER_BLOCK * 2;        // 72 floats = gate + up

    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    const uint8_t* gate_row = W_gate + row * bytes_per_row;
    const uint8_t* up_row   = W_up   + row * bytes_per_row;

    // Issue prefetch of superblock sb into s_wbuf[(sb & 1) * SB_FLOATS ...].
    auto prefetch_sb = [&](int sb) {
        const float* gblk_f = reinterpret_cast<const float*>(gate_row + sb * Q4K_BLOCK_SIZE);
        const float* ublk_f = reinterpret_cast<const float*>(up_row   + sb * Q4K_BLOCK_SIZE);
        float* dst = s_wbuf + (sb & 1) * SB_FLOATS;
        for (int i = tid; i < FLOATS_PER_BLOCK; i += blockDim.x) {
            tq_cp_async_f32(&dst[i], &gblk_f[i]);
            tq_cp_async_f32(&dst[FLOATS_PER_BLOCK + i], &ublk_f[i]);
        }
        tq_cp_async_commit();
    };

    // Pre-load sb=0
    if (n_superblocks > 0) prefetch_sb(0);

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // Pre-fetch next superblock so its load overlaps with this iter's compute.
        if (sb + 1 < n_superblocks) {
            prefetch_sb(sb + 1);
            tq_cp_async_wait_one();      // keep next in flight, current done
        } else {
            tq_cp_async_wait_all();      // last iter: drain
        }
        __syncthreads();

        const uint8_t* gblk = reinterpret_cast<const uint8_t*>(s_wbuf + (sb & 1) * SB_FLOATS);
        const uint8_t* ublk = gblk + Q4K_BLOCK_SIZE;  // up block follows gate in the buffer

        float x_val = s_normed[sb * QK_K + x_idx];

        // Gate dequant
        uint16_t gd_bits  = gblk[0] | (gblk[1] << 8);
        uint16_t gdm_bits = gblk[2] | (gblk[3] << 8);
        float gd    = __half2float(*reinterpret_cast<const __half*>(&gd_bits));
        float gdmin = __half2float(*reinterpret_cast<const __half*>(&gdm_bits));
        const uint8_t* gsc = gblk + 4;
        uint8_t g_sc, g_m;
        get_scale_min_k4(j, gsc, &g_sc, &g_m);
        uint8_t g_byte = gblk[16 + grp * 32 + l];
        float g_nib = is_hi ? (float)(g_byte >> 4) : (float)(g_byte & 0xF);
        gate_sum += (gd * (float)g_sc * g_nib - gdmin * (float)g_m) * x_val;

        // Up dequant
        uint16_t ud_bits  = ublk[0] | (ublk[1] << 8);
        uint16_t udm_bits = ublk[2] | (ublk[3] << 8);
        float ud    = __half2float(*reinterpret_cast<const __half*>(&ud_bits));
        float udmin = __half2float(*reinterpret_cast<const __half*>(&udm_bits));
        const uint8_t* usc = ublk + 4;
        uint8_t u_sc, u_m;
        get_scale_min_k4(j, usc, &u_sc, &u_m);
        uint8_t u_byte = ublk[16 + grp * 32 + l];
        float u_nib = is_hi ? (float)(u_byte >> 4) : (float)(u_byte & 0xF);
        up_sum += (ud * (float)u_sc * u_nib - udmin * (float)u_m) * x_val;

        __syncthreads();   // protect buffer before next prefetch overwrites it
    }

    gate_sum = block_reduce_sum(gate_sum);
    __syncthreads();
    up_sum   = block_reduce_sum(up_sum);

    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        intermediate_out[row] = silu_gate * up_sum;
    }
}

// ─── Kernel 3b: Warp-shuffle LUT Fused Gateup ───────────────
// Uses warp shuffle instead of shared memory for dequant LUT.
// Each warp (32 threads) is in the same sub-block. Lanes 0-15 compute
// gate LUT entries, lanes 16-31 compute up LUT entries.
// __shfl_sync replaces shared memory read — zero extra syncs needed.
// Saves ~15 instructions per thread per superblock vs original.

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_lut_f32(
    const float* __restrict__ input,
    const float* __restrict__ _unused,
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

    // Phase 1: RmsNorm(input) → shared memory
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
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        s_normed[i] = s_normed[i] * s_rms_inv * norm_weight[i];
    }
    __syncthreads();

    // Thread mapping (same as original)
    const int grp   = tid >> 6;   // 0-3
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;   // 0 or 1
    const int l     = pos & 31;   // 0-31
    const int lane  = tid & 31;   // warp lane
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;  // sub-block index 0-7

    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const uint8_t* gate_row = W_gate + row * bytes_per_row;
    const uint8_t* up_row   = W_up   + row * bytes_per_row;

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        float x_val = s_normed[sb * QK_K + x_idx];

        // ── Warp-shuffle LUT (divergence-free) ──
        // All lanes compute BOTH gate and up LUT for nibble = lane & 15.
        // Lanes 0-15 hold the canonical values; 16-31 are redundant but
        // avoid branch divergence. Shuffles read from lanes 0-15 only.
        const int nib = lane & 15;

        const uint8_t* gblk = gate_row + sb * Q4K_BLOCK_SIZE;
        uint16_t gd_bits  = gblk[0] | (gblk[1] << 8);
        uint16_t gdm_bits = gblk[2] | (gblk[3] << 8);
        float gd    = __half2float(*reinterpret_cast<const __half*>(&gd_bits));
        float gdmin = __half2float(*reinterpret_cast<const __half*>(&gdm_bits));
        uint8_t g_sc, g_m;
        get_scale_min_k4(j, gblk + 4, &g_sc, &g_m);
        float gate_lut = gd * (float)g_sc * (float)nib - gdmin * (float)g_m;

        const uint8_t* ublk = up_row + sb * Q4K_BLOCK_SIZE;
        uint16_t ud_bits  = ublk[0] | (ublk[1] << 8);
        uint16_t udm_bits = ublk[2] | (ublk[3] << 8);
        float ud    = __half2float(*reinterpret_cast<const __half*>(&ud_bits));
        float udmin = __half2float(*reinterpret_cast<const __half*>(&udm_bits));
        uint8_t u_sc, u_m;
        get_scale_min_k4(j, ublk + 4, &u_sc, &u_m);
        float up_lut = ud * (float)u_sc * (float)nib - udmin * (float)u_m;

        // ── Lookup via warp shuffle ──
        uint8_t g_byte = gblk[16 + grp * 32 + l];
        int g_nib = is_hi ? (g_byte >> 4) : (g_byte & 0xF);
        gate_sum += __shfl_sync(0xFFFFFFFF, gate_lut, g_nib) * x_val;

        uint8_t u_byte = ublk[16 + grp * 32 + l];
        int u_nib = is_hi ? (u_byte >> 4) : (u_byte & 0xF);
        up_sum += __shfl_sync(0xFFFFFFFF, up_lut, u_nib) * x_val;
    }

    // Phase 3: reduce + SiLU(gate) * up
    gate_sum = block_reduce_sum(gate_sum);
    __syncthreads();
    up_sum = block_reduce_sum(up_sum);

    if (tid == 0) {
        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        intermediate_out[row] = silu_gate * up_sum;
    }
}

// ─── Kernel 4-cpasync: Down Projection + Residual, Cooperative ──
// Same math as fused_q4km_down_residual_f32, restructured so all 256
// threads cooperate on one super-block at a time (vs the baseline's
// thread-per-super-block pattern that leaves 182/256 threads idle on
// Qwen2 7B where intermediate_dim=18944 → 74 super-blocks).
//
// Also adds cp.async double-buffer for the per-super-block weight read
// (144 B) so the load overlaps with the previous super-block's dequant+FMA.
// Intermediate (x) stays in global — it's already laid out contiguously
// per-super-block and L2 caches the stripe across blocks.
//
// Shared memory: s_wbuf[2][36] floats = 288 bytes (just the W buffer).

extern "C" __global__ void fused_q4km_down_residual_cpasync_f32(
    const uint8_t* __restrict__ W_down,
    const float* __restrict__ intermediate,
    float* __restrict__ residual,
    const int hidden_dim,
    const int intermediate_dim
) {
    const int row = blockIdx.x;
    if (row >= hidden_dim) return;
    const int tid = threadIdx.x;

    const int n_superblocks = intermediate_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const uint8_t* w_row = W_down + row * bytes_per_row;

    // Thread mapping within a QK_K=256 super-block (matches gateup kernel).
    const int grp   = tid >> 6;   // 0-3
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;   // 0 or 1
    const int l     = pos & 31;   // 0-31
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    __shared__ float s_wbuf[2][36];  // 2 × 144 B

    auto prefetch_w = [&](int sb) {
        const float* src =
            reinterpret_cast<const float*>(w_row + sb * Q4K_BLOCK_SIZE);
        float* dst = s_wbuf[sb & 1];
        if (tid < 36) {
            tq_cp_async_f32(&dst[tid], &src[tid]);
        }
        tq_cp_async_commit();
    };

    if (n_superblocks > 0) prefetch_w(0);

    float partial_sum = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_w(sb + 1);
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        const uint8_t* block = reinterpret_cast<const uint8_t*>(s_wbuf[sb & 1]);
        const float*   x_sb  = intermediate + sb * QK_K;

        uint16_t d_bits  = block[0] | (block[1] << 8);
        uint16_t dm_bits = block[2] | (block[3] << 8);
        float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));

        const uint8_t* scales = block + 4;
        const uint8_t* qs     = block + 16;

        uint8_t sc_val, m_val;
        get_scale_min_k4(j, scales, &sc_val, &m_val);

        uint8_t q_byte = qs[grp * 32 + l];
        float nibble = is_hi ? (float)(q_byte >> 4) : (float)(q_byte & 0xF);
        partial_sum += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_sb[x_idx];

        __syncthreads();   // protect buffer before next prefetch overwrites it
    }

    partial_sum = block_reduce_sum(partial_sum);
    if (tid == 0) {
        residual[row] += partial_sum;
    }
}

// ─── Kernel 4: Fused Down Projection + Residual Add ──────────
// Grid: hidden_dim blocks, 256 threads
// Thread-per-superblock striding for down projection (intermediate_dim=18944 → 74 sb).

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
