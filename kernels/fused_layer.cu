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

// ─── Kernel 3c: Multi-row (2 rows/block) Fused Gateup ───────
// Each block produces TWO consecutive intermediate_dim outputs
// (row_a = 2*blockIdx.x, row_b = row_a + 1) sharing the same normalized
// x values and s_normed layout. Amortizes x_val reads and boosts ILP
// by interleaving 4 independent dequant chains (gate_a, up_a, gate_b,
// up_b) per superblock.
//
// Halves grid size (18944 → 9472 on Qwen2 7B) → less launch/block-scheduling
// overhead while keeping the same threads-per-block.
//
// Shared-memory layout:
//   s_normed[hidden_dim]                // same as cpasync variant
//   s_wbuf[2][144] floats               // 2 × 576 B = 1152 B total
//                                       // slot layout:
//                                       //   [ 0..36): gate_a block  (144 B)
//                                       //   [36..72): up_a   block
//                                       //   [72..108): gate_b block
//                                       //   [108..144): up_b  block

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_mrow2_f32(
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
    float* s_normed = s_mem;                   // [hidden_dim]
    float* s_wbuf   = s_mem + hidden_dim;      // [2 * 144] = 1152 B

    const int tid   = threadIdx.x;
    const int row_a = 2 * blockIdx.x;
    const int row_b = row_a + 1;
    if (row_a >= intermediate_dim) return;
    const bool have_b = (row_b < intermediate_dim);

    // Phase 1: RmsNorm(input) → shared memory (identical to cpasync variant)
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

    // Phase 2: 4-way pipelined dot (gate_a, up_a, gate_b, up_b)
    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const int FLOATS_PER_BLOCK = Q4K_BLOCK_SIZE / 4;  // 36
    const int SLOT_FLOATS      = FLOATS_PER_BLOCK * 4; // 144 (gate_a+up_a+gate_b+up_b)

    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    const uint8_t* gate_row_a = W_gate + row_a * bytes_per_row;
    const uint8_t* up_row_a   = W_up   + row_a * bytes_per_row;
    const uint8_t* gate_row_b = have_b ? (W_gate + row_b * bytes_per_row) : gate_row_a;
    const uint8_t* up_row_b   = have_b ? (W_up   + row_b * bytes_per_row) : up_row_a;

    // Prefetch all 4 blocks of superblock sb into s_wbuf[(sb&1) * SLOT_FLOATS ...]
    auto prefetch_sb = [&](int sb) {
        const float* ga_f = reinterpret_cast<const float*>(gate_row_a + sb * Q4K_BLOCK_SIZE);
        const float* ua_f = reinterpret_cast<const float*>(up_row_a   + sb * Q4K_BLOCK_SIZE);
        const float* gb_f = reinterpret_cast<const float*>(gate_row_b + sb * Q4K_BLOCK_SIZE);
        const float* ub_f = reinterpret_cast<const float*>(up_row_b   + sb * Q4K_BLOCK_SIZE);
        float* dst = s_wbuf + (sb & 1) * SLOT_FLOATS;
        for (int i = tid; i < FLOATS_PER_BLOCK; i += blockDim.x) {
            tq_cp_async_f32(&dst[i],                         &ga_f[i]);
            tq_cp_async_f32(&dst[FLOATS_PER_BLOCK + i],      &ua_f[i]);
            tq_cp_async_f32(&dst[2 * FLOATS_PER_BLOCK + i],  &gb_f[i]);
            tq_cp_async_f32(&dst[3 * FLOATS_PER_BLOCK + i],  &ub_f[i]);
        }
        tq_cp_async_commit();
    };

    if (n_superblocks > 0) prefetch_sb(0);

    float gate_a = 0.0f, up_a = 0.0f;
    float gate_b = 0.0f, up_b = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_sb(sb + 1);
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        const uint8_t* slot  = reinterpret_cast<const uint8_t*>(s_wbuf + (sb & 1) * SLOT_FLOATS);
        const uint8_t* gblka = slot;
        const uint8_t* ublka = slot + Q4K_BLOCK_SIZE;
        const uint8_t* gblkb = slot + 2 * Q4K_BLOCK_SIZE;
        const uint8_t* ublkb = slot + 3 * Q4K_BLOCK_SIZE;

        float x_val = s_normed[sb * QK_K + x_idx];

        // Gate_a dequant
        uint16_t ga_d_bits  = gblka[0] | (gblka[1] << 8);
        uint16_t ga_dm_bits = gblka[2] | (gblka[3] << 8);
        float ga_d    = __half2float(*reinterpret_cast<const __half*>(&ga_d_bits));
        float ga_dmin = __half2float(*reinterpret_cast<const __half*>(&ga_dm_bits));
        uint8_t ga_sc, ga_m; get_scale_min_k4(j, gblka + 4, &ga_sc, &ga_m);
        uint8_t ga_byte = gblka[16 + grp * 32 + l];
        float   ga_nib  = is_hi ? (float)(ga_byte >> 4) : (float)(ga_byte & 0xF);
        gate_a += (ga_d * (float)ga_sc * ga_nib - ga_dmin * (float)ga_m) * x_val;

        // Up_a dequant
        uint16_t ua_d_bits  = ublka[0] | (ublka[1] << 8);
        uint16_t ua_dm_bits = ublka[2] | (ublka[3] << 8);
        float ua_d    = __half2float(*reinterpret_cast<const __half*>(&ua_d_bits));
        float ua_dmin = __half2float(*reinterpret_cast<const __half*>(&ua_dm_bits));
        uint8_t ua_sc, ua_m; get_scale_min_k4(j, ublka + 4, &ua_sc, &ua_m);
        uint8_t ua_byte = ublka[16 + grp * 32 + l];
        float   ua_nib  = is_hi ? (float)(ua_byte >> 4) : (float)(ua_byte & 0xF);
        up_a += (ua_d * (float)ua_sc * ua_nib - ua_dmin * (float)ua_m) * x_val;

        // Gate_b dequant
        uint16_t gb_d_bits  = gblkb[0] | (gblkb[1] << 8);
        uint16_t gb_dm_bits = gblkb[2] | (gblkb[3] << 8);
        float gb_d    = __half2float(*reinterpret_cast<const __half*>(&gb_d_bits));
        float gb_dmin = __half2float(*reinterpret_cast<const __half*>(&gb_dm_bits));
        uint8_t gb_sc, gb_m; get_scale_min_k4(j, gblkb + 4, &gb_sc, &gb_m);
        uint8_t gb_byte = gblkb[16 + grp * 32 + l];
        float   gb_nib  = is_hi ? (float)(gb_byte >> 4) : (float)(gb_byte & 0xF);
        gate_b += (gb_d * (float)gb_sc * gb_nib - gb_dmin * (float)gb_m) * x_val;

        // Up_b dequant
        uint16_t ub_d_bits  = ublkb[0] | (ublkb[1] << 8);
        uint16_t ub_dm_bits = ublkb[2] | (ublkb[3] << 8);
        float ub_d    = __half2float(*reinterpret_cast<const __half*>(&ub_d_bits));
        float ub_dmin = __half2float(*reinterpret_cast<const __half*>(&ub_dm_bits));
        uint8_t ub_sc, ub_m; get_scale_min_k4(j, ublkb + 4, &ub_sc, &ub_m);
        uint8_t ub_byte = ublkb[16 + grp * 32 + l];
        float   ub_nib  = is_hi ? (float)(ub_byte >> 4) : (float)(ub_byte & 0xF);
        up_b += (ub_d * (float)ub_sc * ub_nib - ub_dmin * (float)ub_m) * x_val;

        __syncthreads();  // protect slot before next prefetch overwrites
    }

    // Reduce 4 accumulators, emit 2 outputs
    gate_a = block_reduce_sum(gate_a);
    __syncthreads();
    up_a   = block_reduce_sum(up_a);
    __syncthreads();
    gate_b = block_reduce_sum(gate_b);
    __syncthreads();
    up_b   = block_reduce_sum(up_b);

    if (tid == 0) {
        float silu_a = gate_a / (1.0f + expf(-gate_a));
        intermediate_out[row_a] = silu_a * up_a;
        if (have_b) {
            float silu_b = gate_b / (1.0f + expf(-gate_b));
            intermediate_out[row_b] = silu_b * up_b;
        }
    }
}

// ─── Kernel 3d: 4-row Fused Gateup ───────────────────────────
// Pushes the mrow2 pattern to 4 consecutive rows per block. Grid
// (18944 → 4736 on Qwen2 7B), 8 accumulators (gate_{a..d}, up_{a..d}),
// 8 dequant chains per superblock sharing one x_val read.
//
// Shared-memory layout:
//   s_normed[hidden_dim]                // 14336 B on Qwen2 7B
//   s_wbuf[2][288] floats               // 2 × 1152 B = 2304 B
//                                       //   slot: 8 × 36 floats
//                                       //   = [ga, ua, gb, ub, gc, uc, gd, ud]

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_mrow4_f32(
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
    float* s_normed = s_mem;
    float* s_wbuf   = s_mem + hidden_dim;   // [2 * 288] = 2304 B

    const int tid   = threadIdx.x;
    const int row_a = 4 * blockIdx.x;
    if (row_a >= intermediate_dim) return;
    const int row_b = row_a + 1;
    const int row_c = row_a + 2;
    const int row_d = row_a + 3;
    const bool have_b = (row_b < intermediate_dim);
    const bool have_c = (row_c < intermediate_dim);
    const bool have_d = (row_d < intermediate_dim);

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

    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const int FLOATS_PER_BLOCK = Q4K_BLOCK_SIZE / 4;   // 36
    const int SLOT_FLOATS      = FLOATS_PER_BLOCK * 8; // 288

    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    const uint8_t* ga_r = W_gate + row_a * bytes_per_row;
    const uint8_t* ua_r = W_up   + row_a * bytes_per_row;
    const uint8_t* gb_r = have_b ? (W_gate + row_b * bytes_per_row) : ga_r;
    const uint8_t* ub_r = have_b ? (W_up   + row_b * bytes_per_row) : ua_r;
    const uint8_t* gc_r = have_c ? (W_gate + row_c * bytes_per_row) : ga_r;
    const uint8_t* uc_r = have_c ? (W_up   + row_c * bytes_per_row) : ua_r;
    const uint8_t* gd_r = have_d ? (W_gate + row_d * bytes_per_row) : ga_r;
    const uint8_t* ud_r = have_d ? (W_up   + row_d * bytes_per_row) : ua_r;

    auto prefetch_sb = [&](int sb) {
        const float* sa = reinterpret_cast<const float*>(ga_r + sb * Q4K_BLOCK_SIZE);
        const float* sb_p = reinterpret_cast<const float*>(ua_r + sb * Q4K_BLOCK_SIZE);
        const float* sc = reinterpret_cast<const float*>(gb_r + sb * Q4K_BLOCK_SIZE);
        const float* sd = reinterpret_cast<const float*>(ub_r + sb * Q4K_BLOCK_SIZE);
        const float* se = reinterpret_cast<const float*>(gc_r + sb * Q4K_BLOCK_SIZE);
        const float* sf = reinterpret_cast<const float*>(uc_r + sb * Q4K_BLOCK_SIZE);
        const float* sg = reinterpret_cast<const float*>(gd_r + sb * Q4K_BLOCK_SIZE);
        const float* sh = reinterpret_cast<const float*>(ud_r + sb * Q4K_BLOCK_SIZE);
        float* dst = s_wbuf + (sb & 1) * SLOT_FLOATS;
        for (int i = tid; i < FLOATS_PER_BLOCK; i += blockDim.x) {
            tq_cp_async_f32(&dst[i],                        &sa[i]);
            tq_cp_async_f32(&dst[FLOATS_PER_BLOCK + i],     &sb_p[i]);
            tq_cp_async_f32(&dst[2 * FLOATS_PER_BLOCK + i], &sc[i]);
            tq_cp_async_f32(&dst[3 * FLOATS_PER_BLOCK + i], &sd[i]);
            tq_cp_async_f32(&dst[4 * FLOATS_PER_BLOCK + i], &se[i]);
            tq_cp_async_f32(&dst[5 * FLOATS_PER_BLOCK + i], &sf[i]);
            tq_cp_async_f32(&dst[6 * FLOATS_PER_BLOCK + i], &sg[i]);
            tq_cp_async_f32(&dst[7 * FLOATS_PER_BLOCK + i], &sh[i]);
        }
        tq_cp_async_commit();
    };

    if (n_superblocks > 0) prefetch_sb(0);

    float ga = 0.0f, ua = 0.0f, gb = 0.0f, ub = 0.0f;
    float gc = 0.0f, uc = 0.0f, gd = 0.0f, ud = 0.0f;

    // Helper macro: inline Q4K dequant+FMA for one (blk_ptr, accum)
    #define Q4K_FMA(blk, accum) do { \
        uint16_t _d_bits  = (blk)[0] | ((blk)[1] << 8); \
        uint16_t _dm_bits = (blk)[2] | ((blk)[3] << 8); \
        float _d    = __half2float(*reinterpret_cast<const __half*>(&_d_bits)); \
        float _dmin = __half2float(*reinterpret_cast<const __half*>(&_dm_bits)); \
        uint8_t _sc, _m; get_scale_min_k4(j, (blk) + 4, &_sc, &_m); \
        uint8_t _byte = (blk)[16 + grp * 32 + l]; \
        float   _nib  = is_hi ? (float)(_byte >> 4) : (float)(_byte & 0xF); \
        (accum) += (_d * (float)_sc * _nib - _dmin * (float)_m) * x_val; \
    } while (0)

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_sb(sb + 1);
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        const uint8_t* slot = reinterpret_cast<const uint8_t*>(s_wbuf + (sb & 1) * SLOT_FLOATS);
        float x_val = s_normed[sb * QK_K + x_idx];

        Q4K_FMA(slot,                          ga);
        Q4K_FMA(slot +     Q4K_BLOCK_SIZE,     ua);
        Q4K_FMA(slot + 2 * Q4K_BLOCK_SIZE,     gb);
        Q4K_FMA(slot + 3 * Q4K_BLOCK_SIZE,     ub);
        Q4K_FMA(slot + 4 * Q4K_BLOCK_SIZE,     gc);
        Q4K_FMA(slot + 5 * Q4K_BLOCK_SIZE,     uc);
        Q4K_FMA(slot + 6 * Q4K_BLOCK_SIZE,     gd);
        Q4K_FMA(slot + 7 * Q4K_BLOCK_SIZE,     ud);

        __syncthreads();
    }
    #undef Q4K_FMA

    // Reduce 8 accumulators, emit up to 4 outputs
    ga = block_reduce_sum(ga); __syncthreads();
    ua = block_reduce_sum(ua); __syncthreads();
    gb = block_reduce_sum(gb); __syncthreads();
    ub = block_reduce_sum(ub); __syncthreads();
    gc = block_reduce_sum(gc); __syncthreads();
    uc = block_reduce_sum(uc); __syncthreads();
    gd = block_reduce_sum(gd); __syncthreads();
    ud = block_reduce_sum(ud);

    if (tid == 0) {
        float sa = ga / (1.0f + expf(-ga)); intermediate_out[row_a] = sa * ua;
        if (have_b) { float s = gb / (1.0f + expf(-gb)); intermediate_out[row_b] = s * ub; }
        if (have_c) { float s = gc / (1.0f + expf(-gc)); intermediate_out[row_c] = s * uc; }
        if (have_d) { float s = gd / (1.0f + expf(-gd)); intermediate_out[row_d] = s * ud; }
    }
}

// ─── Kernel 3e: 8-row Fused Gateup ───────────────────────────
// Pushes mrow pattern to 8 consecutive rows. Grid 18944 → 2368 on Qwen2 7B,
// 16 accumulators, 16 dequant chains per superblock. Shmem 2 × 2304 B = 4608 B.
// Register pressure is the key risk — occupancy may drop below mrow4.

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_mrow8_f32(
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
    float* s_normed = s_mem;
    float* s_wbuf   = s_mem + hidden_dim;   // [2 * 576] = 4608 B

    const int tid   = threadIdx.x;
    const int row0 = 8 * blockIdx.x;
    if (row0 >= intermediate_dim) return;

    // Phase 1: RmsNorm
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

    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const int FLOATS_PER_BLOCK = Q4K_BLOCK_SIZE / 4;   // 36
    const int SLOT_FLOATS      = FLOATS_PER_BLOCK * 16; // 576

    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    // Row base pointers (clamped to row0 for out-of-range rows)
    const uint8_t* gr[8];
    const uint8_t* ur[8];
    bool have[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int rr = row0 + r;
        have[r] = (rr < intermediate_dim);
        int safe = have[r] ? rr : row0;
        gr[r] = W_gate + safe * bytes_per_row;
        ur[r] = W_up   + safe * bytes_per_row;
    }

    auto prefetch_sb = [&](int sb) {
        float* dst = s_wbuf + (sb & 1) * SLOT_FLOATS;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float* g_src = reinterpret_cast<const float*>(gr[r] + sb * Q4K_BLOCK_SIZE);
            const float* u_src = reinterpret_cast<const float*>(ur[r] + sb * Q4K_BLOCK_SIZE);
            for (int i = tid; i < FLOATS_PER_BLOCK; i += blockDim.x) {
                tq_cp_async_f32(&dst[(2 * r)     * FLOATS_PER_BLOCK + i], &g_src[i]);
                tq_cp_async_f32(&dst[(2 * r + 1) * FLOATS_PER_BLOCK + i], &u_src[i]);
            }
        }
        tq_cp_async_commit();
    };

    if (n_superblocks > 0) prefetch_sb(0);

    float gs[8] = {0,0,0,0,0,0,0,0};
    float us[8] = {0,0,0,0,0,0,0,0};

    #define Q4K_FMA(blk, accum) do { \
        uint16_t _d_bits  = (blk)[0] | ((blk)[1] << 8); \
        uint16_t _dm_bits = (blk)[2] | ((blk)[3] << 8); \
        float _d    = __half2float(*reinterpret_cast<const __half*>(&_d_bits)); \
        float _dmin = __half2float(*reinterpret_cast<const __half*>(&_dm_bits)); \
        uint8_t _sc, _m; get_scale_min_k4(j, (blk) + 4, &_sc, &_m); \
        uint8_t _byte = (blk)[16 + grp * 32 + l]; \
        float   _nib  = is_hi ? (float)(_byte >> 4) : (float)(_byte & 0xF); \
        (accum) += (_d * (float)_sc * _nib - _dmin * (float)_m) * x_val; \
    } while (0)

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_sb(sb + 1);
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        const uint8_t* slot = reinterpret_cast<const uint8_t*>(s_wbuf + (sb & 1) * SLOT_FLOATS);
        float x_val = s_normed[sb * QK_K + x_idx];

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            Q4K_FMA(slot + (2 * r)     * Q4K_BLOCK_SIZE, gs[r]);
            Q4K_FMA(slot + (2 * r + 1) * Q4K_BLOCK_SIZE, us[r]);
        }

        __syncthreads();
    }
    #undef Q4K_FMA

    // Reduce 16 accumulators sequentially
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        gs[r] = block_reduce_sum(gs[r]); __syncthreads();
        us[r] = block_reduce_sum(us[r]); __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (have[r]) {
                float s = gs[r] / (1.0f + expf(-gs[r]));
                intermediate_out[row0 + r] = s * us[r];
            }
        }
    }
}

// ─── Kernel 3f: 16-row Fused Gateup (register-pressure edge) ──
// Grid 18944 → 1184 on Qwen2 7B. 32 accumulators, 32 dequant chains
// per superblock. Shmem 2 × 4608 B = 9216 B extra; likely pushes
// register pressure and SM-level occupancy below mrow8.

extern "C" __global__ void fused_addnorm_q4km_gateup_silu_mrow16_f32(
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
    float* s_normed = s_mem;
    float* s_wbuf   = s_mem + hidden_dim;

    const int tid   = threadIdx.x;
    const int row0  = 16 * blockIdx.x;
    if (row0 >= intermediate_dim) return;

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

    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const int FLOATS_PER_BLOCK = Q4K_BLOCK_SIZE / 4;     // 36
    const int SLOT_FLOATS      = FLOATS_PER_BLOCK * 32;  // 1152

    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    const uint8_t* gr[16];
    const uint8_t* ur[16];
    bool have[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        int rr = row0 + r;
        have[r] = (rr < intermediate_dim);
        int safe = have[r] ? rr : row0;
        gr[r] = W_gate + safe * bytes_per_row;
        ur[r] = W_up   + safe * bytes_per_row;
    }

    auto prefetch_sb = [&](int sb) {
        float* dst = s_wbuf + (sb & 1) * SLOT_FLOATS;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float* g_src = reinterpret_cast<const float*>(gr[r] + sb * Q4K_BLOCK_SIZE);
            const float* u_src = reinterpret_cast<const float*>(ur[r] + sb * Q4K_BLOCK_SIZE);
            for (int i = tid; i < FLOATS_PER_BLOCK; i += blockDim.x) {
                tq_cp_async_f32(&dst[(2 * r)     * FLOATS_PER_BLOCK + i], &g_src[i]);
                tq_cp_async_f32(&dst[(2 * r + 1) * FLOATS_PER_BLOCK + i], &u_src[i]);
            }
        }
        tq_cp_async_commit();
    };

    if (n_superblocks > 0) prefetch_sb(0);

    float gs[16]; float us[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) { gs[r] = 0.0f; us[r] = 0.0f; }

    #define Q4K_FMA(blk, accum) do { \
        uint16_t _d_bits  = (blk)[0] | ((blk)[1] << 8); \
        uint16_t _dm_bits = (blk)[2] | ((blk)[3] << 8); \
        float _d    = __half2float(*reinterpret_cast<const __half*>(&_d_bits)); \
        float _dmin = __half2float(*reinterpret_cast<const __half*>(&_dm_bits)); \
        uint8_t _sc, _m; get_scale_min_k4(j, (blk) + 4, &_sc, &_m); \
        uint8_t _byte = (blk)[16 + grp * 32 + l]; \
        float   _nib  = is_hi ? (float)(_byte >> 4) : (float)(_byte & 0xF); \
        (accum) += (_d * (float)_sc * _nib - _dmin * (float)_m) * x_val; \
    } while (0)

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_sb(sb + 1);
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        const uint8_t* slot = reinterpret_cast<const uint8_t*>(s_wbuf + (sb & 1) * SLOT_FLOATS);
        float x_val = s_normed[sb * QK_K + x_idx];

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            Q4K_FMA(slot + (2 * r)     * Q4K_BLOCK_SIZE, gs[r]);
            Q4K_FMA(slot + (2 * r + 1) * Q4K_BLOCK_SIZE, us[r]);
        }

        __syncthreads();
    }
    #undef Q4K_FMA

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        gs[r] = block_reduce_sum(gs[r]); __syncthreads();
        us[r] = block_reduce_sum(us[r]); __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (have[r]) {
                float s = gs[r] / (1.0f + expf(-gs[r]));
                intermediate_out[row0 + r] = s * us[r];
            }
        }
    }
}

// ─── Kernel 3c: dp4a Fused Gateup (MMVQ Step 3) ─────────────
// Mirrors mrow8 structure (RMSNorm + 8 rows/block) but replaces the FP32
// FMA inner loop with INT8 __dp4a ops.
//
// Phase 1: RMSNorm (identical to mrow8) → s_normed[hidden_dim] in shmem.
// Phase 2: Inline quantize s_normed → q8_1 blocks in shmem (36B × n_blocks).
//          Mirrors quantize_f32_to_q8_1_f32 algorithm. One warp per block,
//          8 warps × 14 passes to cover hidden_dim=3584 (112 blocks).
// Phase 3: dp4a matvec — 8 warps, warp w → row (row0+w). Mirrors
//          q4km_matvec_dp4a_f32 inner loop, but reads q8_1 activations
//          from shmem instead of global. Computes gate and up in one pass,
//          then SiLU(gate) × up written to intermediate_out.
//
// Thread layout: 256 threads = 8 warps.
//   - Warp w (0..7) computes output row row0 + w.
//   - Lane layout within warp: sub = lane >> 2 (0..7), slot = lane & 3.
//
// Shmem:
//   s_normed   : hidden_dim × 4 B
//   s_x_q8_1   : (hidden_dim / 32) × 36 B       — inline-quantized activations
//   (no s_wbuf: dp4a reads weights direct from global — mirrors Step 2)

#define Q8_1_BLOCK_SIZE_LOCAL 36

__launch_bounds__(256, 4)
extern "C" __global__ void fused_addnorm_q4km_gateup_silu_dp4a_f32(
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
    float* s_normed = s_mem;                                   // hidden_dim floats
    uint8_t* s_x_q8_1 = reinterpret_cast<uint8_t*>(s_mem + hidden_dim);
    // s_x_q8_1 byte size = (hidden_dim / 32) * 36

    const int tid   = threadIdx.x;
    const int row0  = 8 * blockIdx.x;
    if (row0 >= intermediate_dim) return;

    // ── Phase 1: RmsNorm (identical to mrow8) ────────────────
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

    // ── Phase 2: Inline quantize s_normed → q8_1 blocks in shmem ────
    // Mirrors quantize_f32_to_q8_1_f32. Each warp (32 lanes) handles one
    // 32-element block per pass. 8 warps run in parallel → stride 8 blocks.
    const int n_q8_blocks = hidden_dim / 32;           // e.g. 3584/32 = 112
    const int warp_id = tid >> 5;                       // 0..7
    const int lane    = tid & 31;                       // 0..31

    for (int blk = warp_id; blk < n_q8_blocks; blk += 8) {
        const int idx = blk * 32 + lane;
        const float xi = (idx < hidden_dim) ? s_normed[idx] : 0.0f;

        // Full-warp amax (butterfly, broadcast to all lanes).
        float amax = warp_reduce_max(fabsf(xi));

        const float d     = amax / 127.0f;
        const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        int qi = __float2int_rn(xi * d_inv);
        qi = max(-128, min(127, qi));
        const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)qi;

        // Need sum over the 32 quantized ints for the s field.
        float sum_q = warp_reduce_sum((float)q);

        uint8_t* block_ptr = s_x_q8_1 + (size_t)blk * Q8_1_BLOCK_SIZE_LOCAL;
        int8_t*  qs_ptr    = reinterpret_cast<int8_t*>(block_ptr + 4);
        qs_ptr[lane] = q;

        if (lane == 0) {
            const float s = d * sum_q;
            half2 ds = __floats2half2_rn(d, s);
            *reinterpret_cast<half2*>(block_ptr) = ds;
        }
    }
    __syncthreads();

    // ── Phase 3: dp4a matvec for gate and up projections ───
    // Mirror q4km_matvec_dp4a_f32 but with row0 base and dual accumulators.
    const int sub    = lane >> 2;                       // 0..7
    const int slot   = lane & 3;                        // 0..3
    const int grp_w  = sub >> 1;                        // 0..3 → Q4K byte-group
    const int is_hi  = sub & 1;                         // 0 | 1

    const int row = row0 + warp_id;
    const bool row_active = (row < intermediate_dim);

    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    float partial_g = 0.0f;
    float partial_u = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // Weight super-block pointers (dummy read for inactive warps).
        const uint8_t* blk_g = row_active
            ? (W_gate + row * bytes_per_row + sb * Q4K_BLOCK_SIZE)
            : (W_gate);
        const uint8_t* blk_u = row_active
            ? (W_up   + row * bytes_per_row + sb * Q4K_BLOCK_SIZE)
            : (W_up);

        // Gate scales/mins — direct __half read (block is 4-byte aligned).
        float d_g  = __half2float(*reinterpret_cast<const __half*>(blk_g));
        float dm_g = __half2float(*reinterpret_cast<const __half*>(blk_g + 2));
        uint8_t sc_g_u8, m_g_u8;
        get_scale_min_k4(sub, blk_g + 4, &sc_g_u8, &m_g_u8);

        // Up scales/mins — direct __half read.
        float d_u  = __half2float(*reinterpret_cast<const __half*>(blk_u));
        float dm_u = __half2float(*reinterpret_cast<const __half*>(blk_u + 2));
        uint8_t sc_u_u8, m_u_u8;
        get_scale_min_k4(sub, blk_u + 4, &sc_u_u8, &m_u_u8);

        // Q4_K qs lookup (shared byte-group between lo/hi pair) — __ldg for read-only cache.
        const int* gs32 = reinterpret_cast<const int*>(blk_g + 16 + grp_w * 32);
        const int* us32 = reinterpret_cast<const int*>(blk_u + 16 + grp_w * 32);
        int qg_0 = __ldg(gs32 + slot);
        int qg_1 = __ldg(gs32 + slot + 4);
        int qu_0 = __ldg(us32 + slot);
        int qu_1 = __ldg(us32 + slot + 4);

        int vg_0 = is_hi ? ((qg_0 >> 4) & 0x0F0F0F0F) : (qg_0 & 0x0F0F0F0F);
        int vg_1 = is_hi ? ((qg_1 >> 4) & 0x0F0F0F0F) : (qg_1 & 0x0F0F0F0F);
        int vu_0 = is_hi ? ((qu_0 >> 4) & 0x0F0F0F0F) : (qu_0 & 0x0F0F0F0F);
        int vu_1 = is_hi ? ((qu_1 >> 4) & 0x0F0F0F0F) : (qu_1 & 0x0F0F0F0F);

        // q8_1 activation block for this sub-block (shared mem).
        const uint8_t* q8_1_block = s_x_q8_1 + (size_t)(sb * 8 + sub) * Q8_1_BLOCK_SIZE_LOCAL;
        float d8 = __half2float(*reinterpret_cast<const __half*>(q8_1_block));

        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = qs8[slot];
        int u1 = qs8[slot + 4];

        // dp4a accumulators.
        int sumi_g = __dp4a(vg_0, u0, 0);
        sumi_g     = __dp4a(vg_1, u1, sumi_g);
        int sumi_u = __dp4a(vu_0, u0, 0);
        sumi_u     = __dp4a(vu_1, u1, sumi_u);

        // Shared sum of u (same for gate and up — lane owns same 8 u-bytes).
        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        float lane_g_d = d_g  * (float)sc_g_u8 * (float)sumi_g * d8;
        float lane_g_m = dm_g * (float)m_g_u8  * (float)sum_u  * d8;
        float lane_u_d = d_u  * (float)sc_u_u8 * (float)sumi_u * d8;
        float lane_u_m = dm_u * (float)m_u_u8  * (float)sum_u  * d8;

        partial_g += (lane_g_d - lane_g_m);
        partial_u += (lane_u_d - lane_u_m);
    }

    // Warp-reduce across 32 lanes → lane 0 owns final row sum.
    float gate_row = warp_reduce_sum(partial_g);
    float up_row   = warp_reduce_sum(partial_u);

    if (row_active && lane == 0) {
        float silu = gate_row / (1.0f + expf(-gate_row));
        intermediate_out[row] = silu * up_row;
    }
}

#undef Q8_1_BLOCK_SIZE_LOCAL

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

// ─── Kernel 4b: Multi-row (2 rows/block) Down+Residual ──────
// Same pattern as gateup mrow2: each block produces TWO consecutive
// hidden_dim outputs sharing the same intermediate[] reads across both
// weight rows. Halves grid (3584 → 1792 on Qwen2 7B) while keeping the
// cooperative cp.async pipeline.
//
// Shared memory: s_wbuf[2][72] floats = 576 B (2 slots × [w_a(36) | w_b(36)])

extern "C" __global__ void fused_q4km_down_residual_mrow2_cpasync_f32(
    const uint8_t* __restrict__ W_down,
    const float* __restrict__ intermediate,
    float* __restrict__ residual,
    const int hidden_dim,
    const int intermediate_dim
) {
    const int row_a = 2 * blockIdx.x;
    const int row_b = row_a + 1;
    if (row_a >= hidden_dim) return;
    const bool have_b = (row_b < hidden_dim);
    const int tid = threadIdx.x;

    const int n_superblocks = intermediate_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const uint8_t* w_row_a = W_down + row_a * bytes_per_row;
    const uint8_t* w_row_b = have_b ? (W_down + row_b * bytes_per_row) : w_row_a;

    // Thread mapping within a QK_K=256 super-block.
    const int grp   = tid >> 6;
    const int pos   = tid & 63;
    const int is_hi = pos >> 5;
    const int l     = pos & 31;
    const int x_idx = grp * 64 + pos;
    const int j     = 2 * grp + is_hi;

    __shared__ float s_wbuf[2][72];  // 2 slots × (w_a[36] + w_b[36])

    auto prefetch_w = [&](int sb) {
        const float* src_a = reinterpret_cast<const float*>(w_row_a + sb * Q4K_BLOCK_SIZE);
        const float* src_b = reinterpret_cast<const float*>(w_row_b + sb * Q4K_BLOCK_SIZE);
        float* dst = s_wbuf[sb & 1];
        if (tid < 36) {
            tq_cp_async_f32(&dst[tid],      &src_a[tid]);
            tq_cp_async_f32(&dst[36 + tid], &src_b[tid]);
        }
        tq_cp_async_commit();
    };

    if (n_superblocks > 0) prefetch_w(0);

    float partial_a = 0.0f;
    float partial_b = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_w(sb + 1);
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        const uint8_t* slot = reinterpret_cast<const uint8_t*>(s_wbuf[sb & 1]);
        const uint8_t* blk_a = slot;
        const uint8_t* blk_b = slot + Q4K_BLOCK_SIZE;
        const float*   x_sb  = intermediate + sb * QK_K;
        float x_val = x_sb[x_idx];

        // Row A dequant
        uint16_t ad_bits  = blk_a[0] | (blk_a[1] << 8);
        uint16_t adm_bits = blk_a[2] | (blk_a[3] << 8);
        float ad    = __half2float(*reinterpret_cast<const __half*>(&ad_bits));
        float admin = __half2float(*reinterpret_cast<const __half*>(&adm_bits));
        uint8_t a_sc, a_m; get_scale_min_k4(j, blk_a + 4, &a_sc, &a_m);
        uint8_t a_byte = blk_a[16 + grp * 32 + l];
        float   a_nib  = is_hi ? (float)(a_byte >> 4) : (float)(a_byte & 0xF);
        partial_a += (ad * (float)a_sc * a_nib - admin * (float)a_m) * x_val;

        // Row B dequant
        uint16_t bd_bits  = blk_b[0] | (blk_b[1] << 8);
        uint16_t bdm_bits = blk_b[2] | (blk_b[3] << 8);
        float bd    = __half2float(*reinterpret_cast<const __half*>(&bd_bits));
        float bdmin = __half2float(*reinterpret_cast<const __half*>(&bdm_bits));
        uint8_t b_sc, b_m; get_scale_min_k4(j, blk_b + 4, &b_sc, &b_m);
        uint8_t b_byte = blk_b[16 + grp * 32 + l];
        float   b_nib  = is_hi ? (float)(b_byte >> 4) : (float)(b_byte & 0xF);
        partial_b += (bd * (float)b_sc * b_nib - bdmin * (float)b_m) * x_val;

        __syncthreads();  // protect slot before next prefetch
    }

    partial_a = block_reduce_sum(partial_a);
    __syncthreads();
    partial_b = block_reduce_sum(partial_b);

    if (tid == 0) {
        residual[row_a] += partial_a;
        if (have_b) residual[row_b] += partial_b;
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

// ─── Kernel 4-dp4a: Down Projection + Residual, INT8 dp4a ───
// MMVQ Step 4. Same role as fused_q4km_down_residual_cpasync_f32 but the
// matvec inner loop uses __dp4a(int8×4, int8×4 → int32) instead of FP32 FMA.
//
// Pipeline:
//   Phase 1 (NEW — mirror of gateup dp4a, but reads from GLOBAL not shmem):
//     Inline quantize intermediate[intermediate_dim] → q8_1 blocks in shmem.
//     One warp per 32-element block, 8 warps stride. warp_reduce_max for d,
//     warp_reduce_sum for s.
//   Phase 2: dp4a matvec. mrow8 pattern — 8 output rows per block, one
//     warp per row. For each of the 8 q8_1 sub-blocks aligned with a Q4K
//     super-block: unpack nibbles (2× int32), dp4a × 2 → sumi, separate
//     sum_u via dp4a(0x01010101, u, 0). Per-sub-block scale applied.
//   Phase 3: warp_reduce_sum(partial) → residual[row] += row_sum.
//
// Thread layout: 256 threads = 8 warps. Warp w (0..7) → row row0+w.
//   Lane split: sub = lane>>2 (0..7), slot = lane&3.
//
// Shmem: s_x_q8_1 = (intermediate_dim / 32) × 36 B.
//   Qwen2 7B: 18944/32 = 592 blocks × 36 = 21 312 B ≈ 20.8 KB. Under 48 KB.
//
// NOTE: unlike gateup, intermediate is NOT RMSNormed here — it comes from
// SiLU(gate)*up already in the proper scale. Phase 1 reads it straight from
// global (warmed by L2 across blocks on same SM).

__launch_bounds__(256, 4)
extern "C" __global__ void fused_q4km_down_residual_dp4a_f32(
    const uint8_t* __restrict__ W_down,
    const float* __restrict__ intermediate,
    float* __restrict__ residual,
    const int hidden_dim,
    const int intermediate_dim
) {
    extern __shared__ uint8_t s_mem_down[];
    uint8_t* s_x_q8_1 = s_mem_down;
    // s_x_q8_1 byte size = (intermediate_dim / 32) * 36

    const int tid  = threadIdx.x;
    const int row0 = 8 * blockIdx.x;
    if (row0 >= hidden_dim) return;

    const int warp_id = tid >> 5;     // 0..7
    const int lane    = tid & 31;     // 0..31

    // ── Phase 1: Inline quantize intermediate[] → q8_1 blocks in shmem ──
    // Reads from GLOBAL (intermediate), writes to SHARED (s_x_q8_1).
    // Mirrors gateup dp4a's Phase 2 but with a global source.
    const int n_q8_blocks = intermediate_dim / 32;  // e.g. 18944/32 = 592

    for (int blk = warp_id; blk < n_q8_blocks; blk += 8) {
        const int idx = blk * 32 + lane;
        const float xi = (idx < intermediate_dim) ? intermediate[idx] : 0.0f;

        // Full-warp amax (butterfly broadcast).
        float amax = warp_reduce_max(fabsf(xi));

        const float d     = amax / 127.0f;
        const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        int qi = __float2int_rn(xi * d_inv);
        qi = max(-128, min(127, qi));
        const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)qi;

        float sum_q = warp_reduce_sum((float)q);

        uint8_t* block_ptr = s_x_q8_1 + (size_t)blk * 36;
        int8_t*  qs_ptr    = reinterpret_cast<int8_t*>(block_ptr + 4);
        qs_ptr[lane] = q;

        if (lane == 0) {
            const float s = d * sum_q;
            half2 ds = __floats2half2_rn(d, s);
            *reinterpret_cast<half2*>(block_ptr) = ds;
        }
    }
    __syncthreads();

    // ── Phase 2: dp4a matvec for W_down ──
    // Mirror q4km_matvec_dp4a_f32 inner loop. Single weight matrix (not
    // gate+up pair). Activations come from s_x_q8_1 (shared).
    const int sub    = lane >> 2;     // 0..7 sub-block index
    const int slot   = lane & 3;      // 0..3 int32 position
    const int grp_w  = sub >> 1;      // 0..3 Q4K byte-group
    const int is_hi  = sub & 1;       // 0 | 1

    const int row = row0 + warp_id;
    const bool row_active = (row < hidden_dim);

    const int n_superblocks = intermediate_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    float partial = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // Weight super-block pointer (dummy read for inactive warps to keep
        // warp-uniform control flow).
        const uint8_t* blk = row_active
            ? (W_down + row * bytes_per_row + sb * Q4K_BLOCK_SIZE)
            : (W_down);

        // Direct __half read of fp16 header (block is 4-byte aligned).
        float d_w    = __half2float(*reinterpret_cast<const __half*>(blk));
        float dmin_w = __half2float(*reinterpret_cast<const __half*>(blk + 2));

        uint8_t sc_u8, m_u8;
        get_scale_min_k4(sub, blk + 4, &sc_u8, &m_u8);

        // Q4K qs lookup (shared byte-group between lo/hi pair) — __ldg for read-only cache.
        const int* qs32 = reinterpret_cast<const int*>(blk + 16 + grp_w * 32);
        int q4_0 = __ldg(qs32 + slot);
        int q4_1 = __ldg(qs32 + slot + 4);

        int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
        int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

        // q8_1 activation block (shared memory).
        const uint8_t* q8_1_block = s_x_q8_1 + (size_t)(sb * 8 + sub) * 36;
        float d8 = __half2float(*reinterpret_cast<const __half*>(q8_1_block));

        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = qs8[slot];
        int u1 = qs8[slot + 4];

        // Two dp4a ops per sub-block → sumi = Σ(nibble × q8).
        int sumi = __dp4a(v0, u0, 0);
        sumi     = __dp4a(v1, u1, sumi);

        // sum_u = Σ(q8) over this lane's 8 q8 bytes via dp4a(0x01010101).
        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        float lane_d = d_w    * (float)sc_u8 * (float)sumi  * d8;
        float lane_m = dmin_w * (float)m_u8  * (float)sum_u * d8;

        partial += (lane_d - lane_m);
    }

    // ── Phase 3: reduce + residual add ──
    float row_sum = warp_reduce_sum(partial);

    if (row_active && lane == 0) {
        residual[row] += row_sum;
    }
}

// ─── Kernel 4-dp4a-v2: down_proj + residual, 1-row × 4-warp DP4A ──
//
// Mirror of q4km_matvec_dp4a_v2_f32 (kernels/qmatmul.cu, a613e1d) applied to
// the fused down+residual kernel. Rationale:
//
// v1 (fused_q4km_down_residual_dp4a_f32) = 8 rows/block × 1 warp/row, 256
//   threads. For Qwen2 7B (hidden_dim=3584) grid = 448 blocks. With ~68 SMs
//   on RTX 3080 and ~4-6 blocks resident each, the DRAM pipeline is only
//   partially saturated (Scout analysis: ~14% of peak DRAM BW).
//
// v2 = 1 row/block × 4 warps/row, 128 threads. Grid = hidden_dim (3584)
//   → ~8× more CTAs. Each SM queues ~18 blocks → full DRAM pipeline fill.
//   Math BIT-IDENTICAL to v1 (same dp4a pattern, same per-sub-block scale,
//   same FP32 accumulator). Only the *iteration partitioning* changes:
//   warps cooperate on ONE output row via sb-stride-4 instead of each warp
//   taking a separate row.
//
// Phase 1 (inline quantize intermediate[] → q8_1 in shmem):
//   128 threads = 4 warps. Warp w (0..3) handles q8_1 blocks w, w+4, w+8, ...
//   For Qwen2 7B (intermediate_dim=18944, n_q8_blocks=592) each warp does
//   ~148 blocks. Same shmem layout as v1 (s_x_q8_1 = (interm/32) × 36B).
//
// Phase 2 (dp4a matvec W_down × q8_1):
//   All 4 warps cooperate on THE row = blockIdx.x. Warp w iterates
//   sb = w, w+4, w+8, ... (stride-4 superblocks). Within warp: same
//   (sub = lane>>2, slot = lane&3) decomposition as v1.
//
// Phase 3 (reduce + residual add):
//   Intra-warp warp_reduce_sum → lane 0 of each warp holds its partial.
//   Cross-warp reduce via __shared__ float tmp_shared[4] + __syncthreads.
//   Warp 0 lane 0 sums the 4 partials and writes residual[row] += total.
//
// Grid:  hidden_dim blocks (was hidden_dim/8)
// Block: 128 threads = 4 warps (was 256 = 8 warps)
// Shmem: s_x_q8_1 = (intermediate_dim / 32) × 36 B, same as v1.
//        (tmp_shared[4] is static 16B, does NOT add to dynamic request.)

extern "C" __global__ void fused_q4km_down_residual_dp4a_v2_f32(
    const uint8_t* __restrict__ W_down,
    const float* __restrict__ intermediate,
    float* __restrict__ residual,
    const int hidden_dim,
    const int intermediate_dim
) {
    extern __shared__ uint8_t s_mem_down_v2[];
    uint8_t* s_x_q8_1 = s_mem_down_v2;

    const int row = blockIdx.x;
    if (row >= hidden_dim) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;     // 0..3
    const int lane    = tid & 31;     // 0..31

    // ── Phase 1: Inline quantize intermediate[] → q8_1 blocks in shmem ──
    // 4 warps, stride-4. Each warp owns its q8_1 block end-to-end (one warp
    // per 32-element block so warp_reduce_* is well-defined).
    const int n_q8_blocks = intermediate_dim / 32;  // e.g. 18944/32 = 592

    for (int blk = warp_id; blk < n_q8_blocks; blk += 4) {
        const int idx = blk * 32 + lane;
        const float xi = (idx < intermediate_dim) ? intermediate[idx] : 0.0f;

        float amax = warp_reduce_max(fabsf(xi));

        const float d     = amax / 127.0f;
        const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        int qi = __float2int_rn(xi * d_inv);
        qi = max(-128, min(127, qi));
        const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)qi;

        float sum_q = warp_reduce_sum((float)q);

        uint8_t* block_ptr = s_x_q8_1 + (size_t)blk * 36;
        int8_t*  qs_ptr    = reinterpret_cast<int8_t*>(block_ptr + 4);
        qs_ptr[lane] = q;

        if (lane == 0) {
            const float s = d * sum_q;
            half2 ds = __floats2half2_rn(d, s);
            *reinterpret_cast<half2*>(block_ptr) = ds;
        }
    }
    __syncthreads();

    // ── Phase 2: dp4a matvec for W_down, 4-warp stride-4 on THE row ──
    const int sub    = lane >> 2;     // 0..7 sub-block
    const int slot   = lane & 3;      // 0..3 int32 slot
    const int grp_w  = sub >> 1;      // 0..3 Q4K byte-group
    const int is_hi  = sub & 1;       // 0 | 1

    const int n_superblocks = intermediate_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const uint8_t* row_base = W_down + (size_t)row * bytes_per_row;

    float partial = 0.0f;

    for (int sb = warp_id; sb < n_superblocks; sb += 4) {
        const uint8_t* blk = row_base + sb * Q4K_BLOCK_SIZE;

        uint16_t d_bits  = blk[0] | (blk[1] << 8);
        uint16_t dm_bits = blk[2] | (blk[3] << 8);
        float d_w    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin_w = __half2float(*reinterpret_cast<const __half*>(&dm_bits));

        uint8_t sc_u8, m_u8;
        get_scale_min_k4(sub, blk + 4, &sc_u8, &m_u8);

        const int* qs32 = reinterpret_cast<const int*>(blk + 16 + grp_w * 32);
        int q4_0 = qs32[slot];
        int q4_1 = qs32[slot + 4];

        int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
        int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

        const uint8_t* q8_1_block = s_x_q8_1 + (size_t)(sb * 8 + sub) * 36;
        uint16_t d8_bits = q8_1_block[0] | (q8_1_block[1] << 8);
        float d8 = __half2float(*reinterpret_cast<const __half*>(&d8_bits));

        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = qs8[slot];
        int u1 = qs8[slot + 4];

        int sumi = __dp4a(v0, u0, 0);
        sumi     = __dp4a(v1, u1, sumi);

        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        float lane_d = d_w    * (float)sc_u8 * (float)sumi  * d8;
        float lane_m = dmin_w * (float)m_u8  * (float)sum_u * d8;

        partial += (lane_d - lane_m);
    }

    // ── Phase 3: reduce (intra-warp → cross-warp) + residual add ──
    // Step 1: intra-warp. Lane 0 of each warp holds its partial over sb=w+4k.
    float warp_sum = warp_reduce_sum(partial);

    // Step 2: cross-warp via shmem. 4 partials, one per warp.
    __shared__ float tmp_shared[4];
    if (lane == 0) {
        tmp_shared[warp_id] = warp_sum;
    }
    __syncthreads();

    // Warp 0 lane 0 finalizes.
    if (warp_id == 0 && lane == 0) {
        float total = tmp_shared[0] + tmp_shared[1] + tmp_shared[2] + tmp_shared[3];
        residual[row] += total;
    }
}

// ─── Kernel 4-dp4a-v3: down_proj + residual, 1-row × 4-warp, NO inline quantize ──
//
// Profile finding (2026-04-17, RTX 3080 sm_86, Qwen2 7B):
//   down+res = 360 μs/layer with v1 (8 rows × 1 warp/row, grid=448). Each CTA
//   re-quantizes the full 18,944-element intermediate into its own shmem copy.
//   Shmem pressure (21 KB/block) caps occupancy at ~4 blocks/SM. SM-starved.
//
//   v2 (1 row × 4 warps, grid=3584) made the problem worse: 3584 CTAs × full
//   redundant Phase-1 quantize = 8× the quantize work, regressed -29% in
//   isolation and -33% on Std.
//
// v3 fix: lift the quantize OUT of the kernel. Launcher calls
//   quantize_f32_to_q8_1 once (single launch, grid = n_q8_blocks) to fill a
//   global q8_1 pool buffer. Kernel reads q8_1 from GLOBAL (L2-cached, hot
//   across all 3584 CTAs). No shmem allocation → occupancy jumps from
//   4 blocks/SM to register-limited (~12 blocks/SM on sm_86).
//
// Math: BIT-IDENTICAL to v2/v1 (same dp4a pattern, same per-sub-block scale,
//   same FP32 accumulator). Only partitioning changes — no numerical drift.
//
// Grid:  hidden_dim blocks (1 row each, like q4km_matvec_dp4a_v2_f32)
// Block: 128 threads = 4 warps (cross-warp reduce via 16B __shared__)
// Shmem: 0 B dynamic (only 16B static tmp_shared[4])
__launch_bounds__(128, 12)
extern "C" __global__ void fused_q4km_down_residual_dp4a_v3_f32(
    const uint8_t* __restrict__ W_down,
    const void*    __restrict__ X_q8_1,      // pre-quantized (intermediate_dim/32)*36 B
    float*         __restrict__ residual,
    const int     hidden_dim,
    const int     intermediate_dim
) {
    const int row = blockIdx.x;
    if (row >= hidden_dim) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;     // 0..3
    const int lane    = tid & 31;     // 0..31
    const int sub     = lane >> 2;    // 0..7 sub-block
    const int slot    = lane & 3;     // 0..3 int32 slot
    const int grp_w   = sub >> 1;     // 0..3 Q4K byte-group
    const int is_hi   = sub & 1;      // 0 | 1

    const int n_superblocks = intermediate_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const uint8_t* row_base  = W_down + (size_t)row * bytes_per_row;
    const uint8_t* q8_1_base = reinterpret_cast<const uint8_t*>(X_q8_1);

    float partial = 0.0f;

    // 4-warp stride-4 sb iteration (same as q4km_matvec_dp4a_v2_f32). Adjacent
    // warps stride through sequential superblocks → strong L2 reuse on X_q8_1
    // across concurrent CTAs on the same SM (all 3584 CTAs read the same 21 KB
    // activation).
    for (int sb = warp_id; sb < n_superblocks; sb += 4) {
        const uint8_t* blk = row_base + sb * Q4K_BLOCK_SIZE;

        float d_w    = __half2float(*reinterpret_cast<const __half*>(blk));
        float dmin_w = __half2float(*reinterpret_cast<const __half*>(blk + 2));

        uint8_t sc_u8, m_u8;
        get_scale_min_k4(sub, blk + 4, &sc_u8, &m_u8);

        const int* qs32 = reinterpret_cast<const int*>(blk + 16 + grp_w * 32);
        int q4_0 = __ldg(qs32 + slot);
        int q4_1 = __ldg(qs32 + slot + 4);
        int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
        int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

        const uint8_t* q8_1_block = q8_1_base + (size_t)(sb * 8 + sub) * 36;
        float d8 = __half2float(*reinterpret_cast<const __half*>(q8_1_block));

        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = __ldg(qs8 + slot);
        int u1 = __ldg(qs8 + slot + 4);

        int sumi = __dp4a(v0, u0, 0);
        sumi     = __dp4a(v1, u1, sumi);
        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        float lane_d = d_w    * (float)sc_u8 * (float)sumi  * d8;
        float lane_m = dmin_w * (float)m_u8  * (float)sum_u * d8;
        partial += (lane_d - lane_m);
    }

    // Intra-warp reduce → lane 0 of each warp holds its stride-4 partial sum.
    float warp_sum = warp_reduce_sum(partial);

    // Cross-warp reduce via 16B shmem (static, does not impact occupancy).
    __shared__ float tmp_shared[4];
    if (lane == 0) tmp_shared[warp_id] = warp_sum;
    __syncthreads();

    // Warp 0 lane 0 finalizes + fused residual add.
    if (warp_id == 0 && lane == 0) {
        float total = tmp_shared[0] + tmp_shared[1] + tmp_shared[2] + tmp_shared[3];
        residual[row] += total;
    }
}

// ─── Kernel 1-dp4a: RMSNorm + QKV + bias, INT8 dp4a matvec ──────
//
// Same grid/row-range dispatch as fused_norm_q4km_qkv_bias_f32, but the
// dot product is done in INT8 using dp4a (+ inline q8_1 quantize of the
// normalized activations). Mirrors fused_addnorm_q4km_gateup_silu_dp4a_f32
// for Phase 1+2 and fused_q4km_down_residual_dp4a_f32 for the single-weight
// dp4a inner loop — but here each warp independently picks W_q / W_k / W_v
// (plus the matching output buffer + bias) based on its absolute row index.
//
// Structure:
//   Phase 1: RmsNorm(input) → s_normed        (same as baseline QKV kernel)
//   Phase 2: inline quantize s_normed → s_x_q8_1  (mirrors gateup dp4a)
//   Phase 3: per-warp dp4a matvec W_{q,k,v} × s_x_q8_1
//   Phase 4: bias add (optional per projection) → out_{q,k,v}
//
// Thread layout: 256 threads = 8 warps. Warp w (0..7) → row row0+w.
//   Lane split: sub = lane>>2 (0..7) → Q4K sub-block / q8_1 block aligned;
//               slot = lane&3 → int32 slot inside a 32-byte byte-group.
//
// Shmem: s_normed (hidden_dim × 4 B) + s_x_q8_1 ((hidden_dim/32) × 36 B).
//   Qwen2 7B (hidden_dim=3584): 14 336 + 4 032 = 18 368 B ≈ 18 KB, under 48 KB.

__launch_bounds__(256, 4)
extern "C" __global__ void fused_norm_q4km_qkv_bias_dp4a_f32(
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
    extern __shared__ float s_mem_qkv[];
    float*   s_normed  = s_mem_qkv;                                 // hidden_dim floats
    uint8_t* s_x_q8_1  = reinterpret_cast<uint8_t*>(s_mem_qkv + hidden_dim);
    // s_x_q8_1 byte size = (hidden_dim / 32) * 36

    const int tid      = threadIdx.x;
    const int row0     = 8 * blockIdx.x;
    const int total_rows = q_out + k_out + v_out;
    if (row0 >= total_rows) return;

    const int warp_id  = tid >> 5;     // 0..7
    const int lane     = tid & 31;     // 0..31

    // ── Phase 1: RmsNorm(input) → s_normed ───────────────────────
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

    // ── Phase 2: Inline quantize s_normed → q8_1 blocks in shmem ──
    // One warp per 32-element block; 8 warps stride by 8.
    const int n_q8_blocks = hidden_dim / 32;          // e.g. 3584/32 = 112

    for (int blk = warp_id; blk < n_q8_blocks; blk += 8) {
        const int idx = blk * 32 + lane;
        const float xi = (idx < hidden_dim) ? s_normed[idx] : 0.0f;

        float amax = warp_reduce_max(fabsf(xi));

        const float d     = amax / 127.0f;
        const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

        int qi = __float2int_rn(xi * d_inv);
        qi = max(-128, min(127, qi));
        const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)qi;

        float sum_q = warp_reduce_sum((float)q);

        uint8_t* block_ptr = s_x_q8_1 + (size_t)blk * 36;
        int8_t*  qs_ptr    = reinterpret_cast<int8_t*>(block_ptr + 4);
        qs_ptr[lane] = q;

        if (lane == 0) {
            const float s = d * sum_q;
            half2 ds = __floats2half2_rn(d, s);
            *reinterpret_cast<half2*>(block_ptr) = ds;
        }
    }
    __syncthreads();

    // ── Phase 3: dp4a matvec — each warp picks its W / out / bias ──
    const int sub    = lane >> 2;      // 0..7 sub-block index
    const int slot   = lane & 3;       // 0..3 int32 position
    const int grp_w  = sub >> 1;       // 0..3 Q4K byte-group
    const int is_hi  = sub & 1;        // 0 | 1

    const int row         = row0 + warp_id;
    const bool row_active = (row < total_rows);

    // Row-range dispatch: which projection does this warp serve?
    //   0              .. q_out              → Q
    //   q_out          .. q_out+k_out        → K
    //   q_out+k_out    .. total_rows         → V
    const uint8_t* W_sel;
    const float*   bias_sel;
    float*         out_sel;
    int            local_row;
    if (row < q_out) {
        W_sel     = W_q;
        bias_sel  = bias_q;
        out_sel   = out_q;
        local_row = row;
    } else if (row < q_out + k_out) {
        W_sel     = W_k;
        bias_sel  = bias_k;
        out_sel   = out_k;
        local_row = row - q_out;
    } else {
        W_sel     = W_v;
        bias_sel  = bias_v;
        out_sel   = out_v;
        local_row = row - q_out - k_out;
    }

    const int n_superblocks = hidden_dim / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    float partial = 0.0f;

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // Weight super-block pointer (dummy read for inactive warps keeps
        // warp-uniform control flow — W_sel is always a valid base pointer
        // because every warp routes to one of Q/K/V).
        const uint8_t* blk = row_active
            ? (W_sel + local_row * bytes_per_row + sb * Q4K_BLOCK_SIZE)
            : (W_sel);

        // Direct __half read of fp16 header (block is 4-byte aligned).
        float d_w    = __half2float(*reinterpret_cast<const __half*>(blk));
        float dmin_w = __half2float(*reinterpret_cast<const __half*>(blk + 2));

        uint8_t sc_u8, m_u8;
        get_scale_min_k4(sub, blk + 4, &sc_u8, &m_u8);

        // Q4K qs lookup — __ldg for read-only cache.
        const int* qs32 = reinterpret_cast<const int*>(blk + 16 + grp_w * 32);
        int q4_0 = __ldg(qs32 + slot);
        int q4_1 = __ldg(qs32 + slot + 4);

        int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
        int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

        // q8_1 activation block (shared memory).
        const uint8_t* q8_1_block = s_x_q8_1 + (size_t)(sb * 8 + sub) * 36;
        float d8 = __half2float(*reinterpret_cast<const __half*>(q8_1_block));

        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = qs8[slot];
        int u1 = qs8[slot + 4];

        int sumi = __dp4a(v0, u0, 0);
        sumi     = __dp4a(v1, u1, sumi);

        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        float lane_d = d_w    * (float)sc_u8 * (float)sumi  * d8;
        float lane_m = dmin_w * (float)m_u8  * (float)sum_u * d8;

        partial += (lane_d - lane_m);
    }

    // ── Phase 4: warp-reduce + bias add ──
    float row_sum = warp_reduce_sum(partial);

    if (row_active && lane == 0) {
        float b = (bias_sel != nullptr) ? bias_sel[local_row] : 0.0f;
        out_sel[local_row] = row_sum + b;
    }
}
