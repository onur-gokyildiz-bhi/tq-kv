// Quantized matrix-vector multiply (GEMV) for single-token decode.
//
// Fused dequantization + dot product — avoids intermediate F32 buffer.
// Memory-bandwidth bound: reading ~4 GB Q4_K_M weights per forward pass (7B model).
//
// Two kernels:
//   1. q4km_matvec: Fused dequant+dot for Q4_K_M (the dominant GGUF format)
//   2. q8_0_matvec: Fused dequant+dot for Q8_0 (norms, some weights)
//
// For prefill (batch > 1), use q4km_dequant + cuBLAS SGEMM instead.

#include "common.cuh"

// ─── Q4_K_M Constants ─────────────────────────────────────────
#define QK_K 256              // super-block element count
#define Q4K_BLOCK_SIZE 144    // bytes per super-block

// Extract per-sub-block (scale, min) from 12-byte scales array
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

// ─── Q4_K_M Fused Matvec ─────────────────────────────────────
// output[row] = sum_k( dequant(W[row, k]) * x[k] )
//
// Each block handles one output row.
// Threads cooperatively dequantize super-blocks and accumulate.
//
// W layout: [out_features, in_features] packed as Q4_K_M blocks.
// Supports non-256-aligned in_features (e.g. 896 for Qwen2 0.5B).

// Multi-row Q4K_M matvec: 4 output rows per block, shared memory x cache.
// x loaded ONCE per superblock, reused for all rows → halves x bandwidth.
// 256 threads map 1:1 to the 256 values per superblock.
#define MATVEC_ROWS_PER_BLOCK 4
#define Q4K_BLOCK_FLOATS 36          // Q4K_BLOCK_SIZE (144) / 4 bytes = 36 floats
#define WX_WBUF_FLOATS (MATVEC_ROWS_PER_BLOCK * Q4K_BLOCK_FLOATS)  // 144 floats per buffer

extern "C" __global__ void q4km_matvec_f32(
    const uint8_t* __restrict__ W_packed,  // [out_features * bytes_per_row]
    const float* __restrict__ x,           // [in_features]
    float* __restrict__ output,            // [out_features]
    const int out_features,
    const int in_features
) {
    const int base_row = blockIdx.x * MATVEC_ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    const int n_superblocks = (in_features + QK_K - 1) / QK_K;  // ceiling division
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const int grp = tid >> 6;
    const int pos = tid & 63;
    const int is_hi = pos >> 5;
    const int l = pos & 31;
    const int x_idx = grp * 64 + pos;  // position in 256-element superblock

    float sums[MATVEC_ROWS_PER_BLOCK] = {0.0f};

#if TQ_HAS_CP_ASYNC
    // ── Double-buffered x with async pipeline (sm_80+ / Ampere+) ──
    // Prefetch next superblock's x while computing with current.
    // cp.async bypasses registers: global → shared directly.
    __shared__ float s_x[2][QK_K];
    int cur_buf = 0;

    // Pre-fetch first superblock
    if (n_superblocks > 0) {
        if (tid < in_features) {
            tq_cp_async_f32(&s_x[0][tid], &x[tid]);
        } else {
            s_x[0][tid] = 0.0f;
        }
        tq_cp_async_commit();
    }

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // Wait for current buffer load to complete
        tq_cp_async_wait_all();
        __syncthreads();

        // Start async prefetch of NEXT superblock into alternate buffer
        if (sb + 1 < n_superblocks) {
            const int next_pos = (sb + 1) * QK_K + tid;
            if (next_pos < in_features) {
                tq_cp_async_f32(&s_x[1 - cur_buf][tid], &x[next_pos]);
            } else {
                s_x[1 - cur_buf][tid] = 0.0f;
            }
            tq_cp_async_commit();
        }

        // Compute with current buffer (weight reads overlap with x prefetch)
        float x_val = s_x[cur_buf][x_idx];

        #pragma unroll
        for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE;

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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }

        cur_buf = 1 - cur_buf;
        __syncthreads();
    }
#else
    // ── Synchronous x load (sm_75 / Turing fallback) ──
    __shared__ float s_x[QK_K];

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // Cooperative x load: 256 threads load 256 floats (ONCE for all rows)
        const int x_pos = sb * QK_K + tid;
        s_x[tid] = (x_pos < in_features) ? __ldg(&x[x_pos]) : 0.0f;
        __syncthreads();

        float x_val = s_x[x_idx];

        #pragma unroll
        for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE;

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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }

        __syncthreads();
    }
#endif

    // Reduce and write each row
    #pragma unroll
    for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
        int row = base_row + r;
        if (row >= out_features) break;
        float result = block_reduce_sum(sums[r]);
        if (tid == 0) {
            output[row] = result;
        }
        // sync between reductions (block_reduce_sum uses shared memory)
        if (r < MATVEC_ROWS_PER_BLOCK - 1) __syncthreads();
    }
}

// ─── Q6_K Fused Matvec ──────────────────────────────────────
// Native Q6K dequant+dot: reads 210 bytes/256 values instead of 1024 bytes F32.
// 4.9x less bandwidth than f32_matvec with dequantized weights.
//
// Q6K block layout (210 bytes → 256 values):
//   ql[0..127]:    4-bit lower quantized values
//   qh[128..191]:  2-bit upper quantized values (combined: 6-bit per value)
//   scales[192..207]: 16 × int8 scales
//   d[208..209]:   float16 overall scale
//
// 256 values = 2 groups × 4 sub-blocks × 32 positions.
// dequant: d * scale * (q6 - 32) where q6 = ql_nibble | (qh_bits << 4).
#define Q6K_BLOCK_SIZE 210

extern "C" __global__ void q6k_matvec_f32(
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ x,
    float* __restrict__ output,
    const int out_features,
    const int in_features
) {
    const int base_row = blockIdx.x * MATVEC_ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    const int n_superblocks = (in_features + QK_K - 1) / QK_K;
    const int bytes_per_row = n_superblocks * Q6K_BLOCK_SIZE;

    // Thread mapping: 256 threads → 256 positions in superblock
    const int grp = tid >> 7;        // 0 or 1 (which group of 128)
    const int pos = tid & 127;
    const int sub = pos >> 5;        // 0-3 (sub-block within group)
    const int l   = pos & 31;        // position within sub-block
    const int is  = l >> 4;          // 0 or 1 (scale pair index)
    const int x_idx_q6 = grp * 128 + sub * 32 + l;

    float sums[MATVEC_ROWS_PER_BLOCK] = {0.0f};

#if TQ_HAS_CP_ASYNC
    // ── Double-buffered x with async pipeline (sm_80+) ──
    __shared__ float s_x[2][QK_K];
    int cur_buf = 0;

    if (n_superblocks > 0) {
        if (tid < in_features) {
            tq_cp_async_f32(&s_x[0][tid], &x[tid]);
        } else {
            s_x[0][tid] = 0.0f;
        }
        tq_cp_async_commit();
    }

    for (int sb = 0; sb < n_superblocks; ++sb) {
        tq_cp_async_wait_all();
        __syncthreads();

        if (sb + 1 < n_superblocks) {
            const int next_pos = (sb + 1) * QK_K + tid;
            if (next_pos < in_features) {
                tq_cp_async_f32(&s_x[1 - cur_buf][tid], &x[next_pos]);
            } else {
                s_x[1 - cur_buf][tid] = 0.0f;
            }
            tq_cp_async_commit();
        }

        float x_val = s_x[cur_buf][x_idx_q6];

        #pragma unroll
        for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q6K_BLOCK_SIZE;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = (const int8_t*)(block + 192);
            float d = __half2float(*reinterpret_cast<const __half*>(block + 208));

            int ql_off = grp * 64;
            int qh_off = grp * 32;
            int sc_off = grp * 8;

            uint8_t ql_byte = (sub & 1) ? ql[ql_off + l + 32] : ql[ql_off + l];
            int q_lo = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);

            uint8_t qh_byte = qh[qh_off + l];
            int qh_bits = (qh_byte >> (sub * 2)) & 3;

            int q = q_lo | (qh_bits << 4);
            float sc = (float)scales[sc_off + sub * 2 + is];

            sums[r] += d * sc * (float)(q - 32) * x_val;
        }

        cur_buf = 1 - cur_buf;
        __syncthreads();
    }
#else
    // ── Synchronous x load (sm_75 fallback) ──
    __shared__ float s_x[QK_K];

    for (int sb = 0; sb < n_superblocks; ++sb) {
        const int x_pos = sb * QK_K + tid;
        s_x[tid] = (x_pos < in_features) ? __ldg(&x[x_pos]) : 0.0f;
        __syncthreads();

        float x_val = s_x[x_idx_q6];

        #pragma unroll
        for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q6K_BLOCK_SIZE;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = (const int8_t*)(block + 192);
            float d = __half2float(*reinterpret_cast<const __half*>(block + 208));

            int ql_off = grp * 64;
            int qh_off = grp * 32;
            int sc_off = grp * 8;

            uint8_t ql_byte = (sub & 1) ? ql[ql_off + l + 32] : ql[ql_off + l];
            int q_lo = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);

            uint8_t qh_byte = qh[qh_off + l];
            int qh_bits = (qh_byte >> (sub * 2)) & 3;

            int q = q_lo | (qh_bits << 4);
            float sc = (float)scales[sc_off + sub * 2 + is];

            sums[r] += d * sc * (float)(q - 32) * x_val;
        }

        __syncthreads();
    }
#endif

    #pragma unroll
    for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
        int row = base_row + r;
        if (row >= out_features) break;
        float result = block_reduce_sum(sums[r]);
        if (tid == 0) {
            output[row] = result;
        }
        if (r < MATVEC_ROWS_PER_BLOCK - 1) __syncthreads();
    }
}

// ─── Q6_K Matvec — 8-row variant ─────────────────────────────
// Same math as q6k_matvec_f32; 8 output rows per block (vs 4). Grid
// halves, 8 sums accumulators per thread, shared x reads amortized
// across 8 dequant chains.

#define MROW8_Q6K 8

extern "C" __global__ void q6k_matvec_mrow8_f32(
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ x,
    float* __restrict__ output,
    const int out_features,
    const int in_features
) {
    const int base_row = blockIdx.x * MROW8_Q6K;
    const int tid = threadIdx.x;

    const int n_superblocks = (in_features + QK_K - 1) / QK_K;
    const int bytes_per_row = n_superblocks * Q6K_BLOCK_SIZE;

    const int grp = tid >> 7;
    const int pos = tid & 127;
    const int sub = pos >> 5;
    const int l   = pos & 31;
    const int is  = l >> 4;
    const int x_idx_q6 = grp * 128 + sub * 32 + l;

    float sums[MROW8_Q6K] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#if TQ_HAS_CP_ASYNC
    __shared__ float s_x[2][QK_K];
    int cur_buf = 0;

    if (n_superblocks > 0) {
        if (tid < in_features) {
            tq_cp_async_f32(&s_x[0][tid], &x[tid]);
        } else {
            s_x[0][tid] = 0.0f;
        }
        tq_cp_async_commit();
    }

    for (int sb = 0; sb < n_superblocks; ++sb) {
        tq_cp_async_wait_all();
        __syncthreads();

        if (sb + 1 < n_superblocks) {
            const int next_pos = (sb + 1) * QK_K + tid;
            if (next_pos < in_features) {
                tq_cp_async_f32(&s_x[1 - cur_buf][tid], &x[next_pos]);
            } else {
                s_x[1 - cur_buf][tid] = 0.0f;
            }
            tq_cp_async_commit();
        }

        float x_val = s_x[cur_buf][x_idx_q6];

        #pragma unroll
        for (int r = 0; r < MROW8_Q6K; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q6K_BLOCK_SIZE;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = (const int8_t*)(block + 192);
            float d = __half2float(*reinterpret_cast<const __half*>(block + 208));

            int ql_off = grp * 64;
            int qh_off = grp * 32;
            int sc_off = grp * 8;

            uint8_t ql_byte = (sub & 1) ? ql[ql_off + l + 32] : ql[ql_off + l];
            int q_lo = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);

            uint8_t qh_byte = qh[qh_off + l];
            int qh_bits = (qh_byte >> (sub * 2)) & 3;

            int q = q_lo | (qh_bits << 4);
            float sc = (float)scales[sc_off + sub * 2 + is];

            sums[r] += d * sc * (float)(q - 32) * x_val;
        }

        cur_buf = 1 - cur_buf;
        __syncthreads();
    }
#else
    __shared__ float s_x[QK_K];
    for (int sb = 0; sb < n_superblocks; ++sb) {
        const int x_pos = sb * QK_K + tid;
        s_x[tid] = (x_pos < in_features) ? __ldg(&x[x_pos]) : 0.0f;
        __syncthreads();
        float x_val = s_x[x_idx_q6];
        #pragma unroll
        for (int r = 0; r < MROW8_Q6K; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;
            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q6K_BLOCK_SIZE;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = (const int8_t*)(block + 192);
            float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
            int ql_off = grp * 64;
            int qh_off = grp * 32;
            int sc_off = grp * 8;
            uint8_t ql_byte = (sub & 1) ? ql[ql_off + l + 32] : ql[ql_off + l];
            int q_lo = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);
            uint8_t qh_byte = qh[qh_off + l];
            int qh_bits = (qh_byte >> (sub * 2)) & 3;
            int q = q_lo | (qh_bits << 4);
            float sc = (float)scales[sc_off + sub * 2 + is];
            sums[r] += d * sc * (float)(q - 32) * x_val;
        }
        __syncthreads();
    }
#endif

    #pragma unroll
    for (int r = 0; r < MROW8_Q6K; ++r) {
        int row = base_row + r;
        if (row >= out_features) break;
        float result = block_reduce_sum(sums[r]);
        if (tid == 0) {
            output[row] = result;
        }
        if (r < MROW8_Q6K - 1) __syncthreads();
    }
}
#undef MROW8_Q6K

// ─── Q6_K × q8_1 dp4a Matvec — MMVQ-style for Q6K ─────────────────
//
// Mirror of q4km_matvec_dp4a_v2_f32 adapted to Q6K's 6-bit weight layout.
// Motivation: Q6K is used for lm_head + token_embd (Qwen2 7B: 151936×3584
// matmul, 306 MB of Q6K weights, dominates per-token cost at ~2.6ms / 13% of
// decode budget). Existing q6k_matvec_f32 uses fp32 dequant-style math and
// tops out at ~15% of peak DRAM bandwidth. DP4A INT8 path has higher
// arithmetic throughput and smaller per-thread state → better occupancy.
//
// Q6_K block layout (210 bytes → 256 weights):
//   ql[0..127]:    4-bit LO nibbles (128 bytes)
//   qh[128..191]:  2-bit HI bits, 4 sub-blocks packed per byte (64 bytes)
//   scales[192..207]: 16 × int8 per-pair scales
//   d[208..209]:   fp16 overall scale
//
// Weight value at position (grp, sub, l): q6 = (ql_nibble) | (qh_bits << 4)
// where ql/qh selection follows the reference kernel. Dequantized:
//   w = d * scale * (q6 - 32)    — symmetric around 32, no dmin/m zero-point
//
// Lane mapping (mirror Q4K dp4a_v2):
//   sub  = lane >> 2        0..7 (maps to q6_grp×4 + q6_sub)
//   slot = lane & 3         0..3 (int32 slot within a sub-block half)
//   q6_grp  = sub >> 2      0..1 (which 128-byte ql group)
//   q6_sub  = sub & 3       0..3 (which sub-block within group)
//   ql_half = q6_sub & 1    0..1 (low/high 32-byte half of ql group)
//   is_hi   = q6_sub >> 1   0..1 (lo nibble or hi nibble of ql)
//
// Each lane processes 8 weight positions within one sub-block:
//   q6_0 = 4 weights at positions slot*4 + {0..3}
//   q6_1 = 4 weights at positions slot*4 + {16..19}
// Two __dp4a ops per sub-block (× 4-warp stride-4 iteration over superblocks).
//
// Per-byte subtract-32 uses __vsub4 (SIMD byte subtract, no borrow between
// bytes). q6 bytes ∈ [0, 63]; after vsub4 with 0x20202020 → [-32, 31] as
// signed int8 (two's complement per byte). dp4a treats inputs as signed int8.
//
// Q6K blocks are 210 bytes → NOT uniformly 4-byte aligned across the row.
// We load ql/qh bytes individually and pack into int32 in-register. Compiler
// emits LDG.U32 when aligned, LDG.U8 × 4 otherwise. Slight overhead on
// misaligned blocks, but required for correctness (reinterpret_cast<int*> on
// misaligned pointer = UB / crash).
//
// Math: bit-identical-equivalent to q6k_matvec_f32 modulo dp4a int8
// quantization of x (via q8_1). PPL impact: identical to q4km dp4a path
// (~+0.04% noise from q8_1 rounding).
//
// Grid:  out_features blocks (1 row each)
// Block: 128 threads = 4 warps
// Shmem: 16 B static only (tmp_shared[4] for cross-warp reduce)
__launch_bounds__(128, 12)
extern "C" __global__ void q6k_matvec_dp4a_v2_f32(
    const uint8_t * __restrict__ W_q6k,
    const void    * __restrict__ X_q8_1,    // pre-quantized (in_features/32)*36 B
    float         * __restrict__ output,
    const int      out_features,
    const int      in_features              // must be multiple of QK_K (256)
) {
    const int row = blockIdx.x;
    if (row >= out_features) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;            // 0..3
    const int lane    = tid & 31;            // 0..31
    const int sub     = lane >> 2;           // 0..7
    const int slot    = lane & 3;            // 0..3

    const int q6_grp  = sub >> 2;            // 0 or 1
    const int q6_sub  = sub & 3;             // 0..3
    const int ql_half = q6_sub & 1;          // which 32B half of ql[grp*64..]
    const int is_hi   = q6_sub >> 1;         // 0 = lo nibble, 1 = hi nibble
    const int qh_shift = q6_sub * 2;         // which 2 bits of each qh byte

    const int n_superblocks = in_features / QK_K;
    const int bytes_per_row = n_superblocks * Q6K_BLOCK_SIZE;
    const uint8_t* row_base  = W_q6k + (size_t)row * bytes_per_row;
    const uint8_t* q8_1_base = reinterpret_cast<const uint8_t*>(X_q8_1);

    float partial = 0.0f;

    // 4-warp stride-4 superblock iteration — adjacent warps walk sequential
    // superblocks for L2 reuse on q8_1 (shared across all CTAs on same SM).
    for (int sb = warp_id; sb < n_superblocks; sb += 4) {
        const uint8_t* blk = row_base + sb * Q6K_BLOCK_SIZE;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t*  scales = reinterpret_cast<const int8_t*>(blk + 192);

        // fp16 overall scale (2-byte load, potentially unaligned).
        uint16_t d_bits = (uint16_t)blk[208] | ((uint16_t)blk[209] << 8);
        float d_w = __half2float(*reinterpret_cast<const __half*>(&d_bits));

        const int ql_off = q6_grp * 64;
        const int qh_off = q6_grp * 32;
        const int sc_off = q6_grp * 8;

        // ── Load 4 consecutive ql bytes for q6_0 (positions slot*4..+3) ──
        // Not using reinterpret_cast<int*> because Q6K blocks are 210 B and
        // not 4-byte aligned on every super-block.
        const uint8_t* ql_p0 = ql + ql_off + ql_half * 32 + slot * 4;
        int ql_0 = (int)ql_p0[0]
                 | ((int)ql_p0[1] << 8)
                 | ((int)ql_p0[2] << 16)
                 | ((int)ql_p0[3] << 24);

        // Load 4 ql bytes for q6_1 (positions slot*4+16..+19 within sub-block).
        // These are 16 bytes further within the same half.
        const uint8_t* ql_p1 = ql_p0 + 16;
        int ql_1 = (int)ql_p1[0]
                 | ((int)ql_p1[1] << 8)
                 | ((int)ql_p1[2] << 16)
                 | ((int)ql_p1[3] << 24);

        // Extract target nibbles: lo for is_hi=0, hi for is_hi=1.
        int n0 = is_hi ? ((ql_0 >> 4) & 0x0F0F0F0F) : (ql_0 & 0x0F0F0F0F);
        int n1 = is_hi ? ((ql_1 >> 4) & 0x0F0F0F0F) : (ql_1 & 0x0F0F0F0F);

        // ── Load 4 qh bytes for q6_0 (positions slot*4..+3) ──
        // qh is 64 B total (grp×32 per group), so qh[qh_off + l] ∈ [0, 32).
        const uint8_t* qh_p0 = qh + qh_off + slot * 4;
        int qh_0 = (int)qh_p0[0]
                 | ((int)qh_p0[1] << 8)
                 | ((int)qh_p0[2] << 16)
                 | ((int)qh_p0[3] << 24);

        const uint8_t* qh_p1 = qh_p0 + 16;
        int qh_1 = (int)qh_p1[0]
                 | ((int)qh_p1[1] << 8)
                 | ((int)qh_p1[2] << 16)
                 | ((int)qh_p1[3] << 24);

        // Extract 2 high bits per byte for this sub-block.
        int h0 = (qh_0 >> qh_shift) & 0x03030303;
        int h1 = (qh_1 >> qh_shift) & 0x03030303;

        // Combine → q6 value per byte in [0, 63], then subtract 32 per byte
        // via SIMD __vsub4 (no borrow between bytes) → signed int8 in [-32, 31].
        int q_raw_0 = n0 | (h0 << 4);
        int q_raw_1 = n1 | (h1 << 4);
        int q_signed_0 = __vsub4(q_raw_0, 0x20202020);
        int q_signed_1 = __vsub4(q_raw_1, 0x20202020);

        // ── q8_1 activation block for this sub-block ──
        // sub-block index within super-block = sub (0..7), matches Q4K pattern.
        const uint8_t* q8_1_block = q8_1_base + (size_t)(sb * 8 + sub) * 36;
        float d8 = __half2float(*reinterpret_cast<const __half*>(q8_1_block));
        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = __ldg(qs8 + slot);
        int u1 = __ldg(qs8 + slot + 4);

        // ── Two dp4a dots: signed int8 × signed int8 → int32 accumulator ──
        int dot0 = __dp4a(q_signed_0, u0, 0);
        int dot1 = __dp4a(q_signed_1, u1, 0);

        // Per-pair int8 scales (signed).
        int sc0 = (int)scales[sc_off + q6_sub * 2 + 0];  // is=0 pair (l=0..15)
        int sc1 = (int)scales[sc_off + q6_sub * 2 + 1];  // is=1 pair (l=16..31)

        // Accumulate: d_w * d8 * (sc0 * dot0 + sc1 * dot1). Note Q6K has no
        // per-sub-block zero-point (symmetric around 32, already handled in
        // q_signed), so no dmin × Σq8 correction term (unlike Q4K).
        partial += d_w * d8 * ((float)sc0 * (float)dot0 + (float)sc1 * (float)dot1);
    }

    // Intra-warp reduce → lane 0 of each warp holds its stride-4 partial.
    float warp_sum = warp_reduce_sum(partial);

    // Cross-warp reduce via 16B shmem.
    __shared__ float tmp_shared[4];
    if (lane == 0) tmp_shared[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        float total = tmp_shared[0] + tmp_shared[1] + tmp_shared[2] + tmp_shared[3];
        output[row] = total;
    }
}

// ─── Q8_0 Fused Matvec ───────────────────────────────────────
// Simpler: each block is 34 bytes = [f16 d][i8 × 32]
#define Q8_0_BLOCK_SIZE 34
#define QK8_0 32

extern "C" __global__ void q8_0_matvec_f32(
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ x,
    float* __restrict__ output,
    const int out_features,
    const int in_features
) {
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;

    const int n_blocks = in_features / QK8_0;
    const int bytes_per_row = n_blocks * Q8_0_BLOCK_SIZE;
    const uint8_t* w_row = W_packed + row * bytes_per_row;

    float partial_sum = 0.0f;

    for (int b = tid; b < n_blocks; b += blockDim.x) {
        const uint8_t* block = w_row + b * Q8_0_BLOCK_SIZE;
        uint16_t d_bits = block[0] | (block[1] << 8);
        float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        const float* x_b = x + b * QK8_0;

        float local = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            local += (float)__ldg(&qs[j]) * __ldg(&x_b[j]);
        }
        partial_sum += d * local;
    }

    partial_sum = block_reduce_sum(partial_sum);
    if (tid == 0) {
        output[row] = partial_sum;
    }
}

// ─── Q4_K_M Batch Dequantize (for prefill + cuBLAS SGEMM) ───
// Dequantize entire weight matrix to F32 on GPU.
// Each block handles one super-block row.

// ─── Q4_K_M Matvec — X + Weight cp.async pipelined ───────────
// Same math as q4km_matvec_f32; adds a second cp.async pipeline for the
// weight super-blocks (4 rows × 144 B = 576 B / iter). Targets the
// ~16%-of-peak-bandwidth regime on large Q4K matvecs like LM head (vocab
// × hidden) and Wo (hidden × hidden).
//
// Shared memory:
//   s_x[2][QK_K]                2 × 1024 = 2 KB  (existing X pipeline)
//   s_w[2][4 * 36 floats]       2 × 576  = 1152 B (new W pipeline)
// Total extra vs baseline: 1152 bytes — fits alongside s_x.

extern "C" __global__ void q4km_matvec_wx_cpasync_f32(
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ x,
    float* __restrict__ output,
    const int out_features,
    const int in_features
) {
    const int base_row = blockIdx.x * MATVEC_ROWS_PER_BLOCK;
    const int tid = threadIdx.x;

    const int n_superblocks = (in_features + QK_K - 1) / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const int grp = tid >> 6;
    const int pos = tid & 63;
    const int is_hi = pos >> 5;
    const int l = pos & 31;
    const int x_idx = grp * 64 + pos;

    float sums[MATVEC_ROWS_PER_BLOCK] = {0.0f};

#if TQ_HAS_CP_ASYNC
    __shared__ float s_x[2][QK_K];
    __shared__ float s_w[2][WX_WBUF_FLOATS];   // [buf][row * 36 + off]
    int cur_buf = 0;

    // Pre-fetch sb=0's X and W
    auto prefetch_x = [&](int sb) {
        const int pos_g = sb * QK_K + tid;
        if (pos_g < in_features) {
            tq_cp_async_f32(&s_x[sb & 1][tid], &x[pos_g]);
        } else {
            s_x[sb & 1][tid] = 0.0f;
        }
    };
    auto prefetch_w = [&](int sb) {
        float* dst = s_w[sb & 1];
        #pragma unroll
        for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;
            const float* src =
                reinterpret_cast<const float*>(W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE);
            // 36 floats = 144 bytes per row; 256 threads cooperatively cover all 4 rows.
            int idx = tid - r * Q4K_BLOCK_FLOATS;
            if (idx >= 0 && idx < Q4K_BLOCK_FLOATS) {
                tq_cp_async_f32(&dst[r * Q4K_BLOCK_FLOATS + idx], &src[idx]);
            }
        }
    };

    if (n_superblocks > 0) {
        prefetch_x(0);
        prefetch_w(0);
        tq_cp_async_commit();
    }

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // Issue prefetch of next; keep previous in flight via wait_one.
        if (sb + 1 < n_superblocks) {
            prefetch_x(sb + 1);
            prefetch_w(sb + 1);
            tq_cp_async_commit();
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        float x_val = s_x[cur_buf][x_idx];

        #pragma unroll
        for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block =
                reinterpret_cast<const uint8_t*>(&s_w[cur_buf][r * Q4K_BLOCK_FLOATS]);

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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }

        cur_buf = 1 - cur_buf;
        __syncthreads();
    }
#else
    // Pre-Ampere fallback: identical to baseline kernel (sync X load, no W pipeline).
    __shared__ float s_x[QK_K];
    for (int sb = 0; sb < n_superblocks; ++sb) {
        const int x_pos = sb * QK_K + tid;
        s_x[tid] = (x_pos < in_features) ? __ldg(&x[x_pos]) : 0.0f;
        __syncthreads();
        float x_val = s_x[x_idx];
        #pragma unroll
        for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;
            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE;
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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }
        __syncthreads();
    }
#endif

    #pragma unroll
    for (int r = 0; r < MATVEC_ROWS_PER_BLOCK; ++r) {
        int row = base_row + r;
        if (row >= out_features) break;
        float result = block_reduce_sum(sums[r]);
        if (tid == 0) output[row] = result;
        if (r < MATVEC_ROWS_PER_BLOCK - 1) __syncthreads();
    }
}

// ─── Q4_K_M Matvec — 16-row variant (register spill edge) ────
// Push mrow ladder beyond mrow8. Grid halves (mrow8 → 9504, mrow16 → 4752
// on lm_head=152064). 16 sums accumulators + deeper s_w pipeline.
// Expected: register spill on SM86 based on gateup mrow16 precedent.

#define MROW16 16
#define WX_WBUF_FLOATS16 (MROW16 * Q4K_BLOCK_FLOATS)   // 576 floats

extern "C" __global__ void q4km_matvec_mrow16_f32(
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ x,
    float* __restrict__ output,
    const int out_features,
    const int in_features
) {
    const int base_row = blockIdx.x * MROW16;
    const int tid = threadIdx.x;

    const int n_superblocks = (in_features + QK_K - 1) / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const int grp = tid >> 6;
    const int pos = tid & 63;
    const int is_hi = pos >> 5;
    const int l = pos & 31;
    const int x_idx = grp * 64 + pos;

    float sums[MROW16];
    #pragma unroll
    for (int r = 0; r < MROW16; ++r) sums[r] = 0.0f;

#if TQ_HAS_CP_ASYNC
    __shared__ float s_x[2][QK_K];
    __shared__ float s_w[2][WX_WBUF_FLOATS16];
    int cur_buf = 0;

    auto prefetch_x = [&](int sb) {
        const int pos_g = sb * QK_K + tid;
        if (pos_g < in_features) {
            tq_cp_async_f32(&s_x[sb & 1][tid], &x[pos_g]);
        } else {
            s_x[sb & 1][tid] = 0.0f;
        }
    };
    auto prefetch_w = [&](int sb) {
        float* dst = s_w[sb & 1];
        #pragma unroll
        for (int r = 0; r < MROW16; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;
            const float* src =
                reinterpret_cast<const float*>(W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE);
            for (int i = tid; i < Q4K_BLOCK_FLOATS; i += blockDim.x) {
                tq_cp_async_f32(&dst[r * Q4K_BLOCK_FLOATS + i], &src[i]);
            }
        }
    };

    if (n_superblocks > 0) {
        prefetch_x(0);
        prefetch_w(0);
        tq_cp_async_commit();
    }

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_x(sb + 1);
            prefetch_w(sb + 1);
            tq_cp_async_commit();
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        float x_val = s_x[cur_buf][x_idx];

        #pragma unroll
        for (int r = 0; r < MROW16; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block =
                reinterpret_cast<const uint8_t*>(&s_w[cur_buf][r * Q4K_BLOCK_FLOATS]);

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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }

        cur_buf = 1 - cur_buf;
        __syncthreads();
    }
#else
    __shared__ float s_x[QK_K];
    for (int sb = 0; sb < n_superblocks; ++sb) {
        const int x_pos = sb * QK_K + tid;
        s_x[tid] = (x_pos < in_features) ? __ldg(&x[x_pos]) : 0.0f;
        __syncthreads();
        float x_val = s_x[x_idx];
        #pragma unroll
        for (int r = 0; r < MROW16; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;
            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE;
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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }
        __syncthreads();
    }
#endif

    #pragma unroll
    for (int r = 0; r < MROW16; ++r) {
        int row = base_row + r;
        if (row >= out_features) break;
        float result = block_reduce_sum(sums[r]);
        if (tid == 0) output[row] = result;
        if (r < MROW16 - 1) __syncthreads();
    }
}
#undef MROW16
#undef WX_WBUF_FLOATS16

// ─── Q4_K_M Matvec — 8-row variant, X + W cp.async ──────────
// Stacks on wx_cpasync: 8 rows per block (vs 4). Grid halves, 8 sums
// accumulators per thread, shmem +1152 B for wider W buffer.
// Benefits: wo_matvec (per-layer ~83 μs) and LM head (per-token ~2500 μs).

#define MROW8 8
#define WX_WBUF_FLOATS8 (MROW8 * Q4K_BLOCK_FLOATS)   // 288 floats

extern "C" __global__ void q4km_matvec_mrow8_f32(
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ x,
    float* __restrict__ output,
    const int out_features,
    const int in_features
) {
    const int base_row = blockIdx.x * MROW8;
    const int tid = threadIdx.x;

    const int n_superblocks = (in_features + QK_K - 1) / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    const int grp = tid >> 6;
    const int pos = tid & 63;
    const int is_hi = pos >> 5;
    const int l = pos & 31;
    const int x_idx = grp * 64 + pos;

    float sums[MROW8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#if TQ_HAS_CP_ASYNC
    __shared__ float s_x[2][QK_K];
    __shared__ float s_w[2][WX_WBUF_FLOATS8];   // [buf][row * 36 + off], 288 floats
    int cur_buf = 0;

    auto prefetch_x = [&](int sb) {
        const int pos_g = sb * QK_K + tid;
        if (pos_g < in_features) {
            tq_cp_async_f32(&s_x[sb & 1][tid], &x[pos_g]);
        } else {
            s_x[sb & 1][tid] = 0.0f;
        }
    };
    auto prefetch_w = [&](int sb) {
        float* dst = s_w[sb & 1];
        #pragma unroll
        for (int r = 0; r < MROW8; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;
            const float* src =
                reinterpret_cast<const float*>(W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE);
            for (int i = tid; i < Q4K_BLOCK_FLOATS; i += blockDim.x) {
                tq_cp_async_f32(&dst[r * Q4K_BLOCK_FLOATS + i], &src[i]);
            }
        }
    };

    if (n_superblocks > 0) {
        prefetch_x(0);
        prefetch_w(0);
        tq_cp_async_commit();
    }

    for (int sb = 0; sb < n_superblocks; ++sb) {
        if (sb + 1 < n_superblocks) {
            prefetch_x(sb + 1);
            prefetch_w(sb + 1);
            tq_cp_async_commit();
            tq_cp_async_wait_one();
        } else {
            tq_cp_async_wait_all();
        }
        __syncthreads();

        float x_val = s_x[cur_buf][x_idx];

        #pragma unroll
        for (int r = 0; r < MROW8; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;

            const uint8_t* block =
                reinterpret_cast<const uint8_t*>(&s_w[cur_buf][r * Q4K_BLOCK_FLOATS]);

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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }

        cur_buf = 1 - cur_buf;
        __syncthreads();
    }
#else
    // Pre-Ampere fallback: sync X load, no W pipeline.
    __shared__ float s_x[QK_K];
    for (int sb = 0; sb < n_superblocks; ++sb) {
        const int x_pos = sb * QK_K + tid;
        s_x[tid] = (x_pos < in_features) ? __ldg(&x[x_pos]) : 0.0f;
        __syncthreads();
        float x_val = s_x[x_idx];
        #pragma unroll
        for (int r = 0; r < MROW8; ++r) {
            int row = base_row + r;
            if (row >= out_features) break;
            const uint8_t* block = W_packed + row * bytes_per_row + sb * Q4K_BLOCK_SIZE;
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
            sums[r] += (d * (float)sc_val * nibble - dmin * (float)m_val) * x_val;
        }
        __syncthreads();
    }
#endif

    #pragma unroll
    for (int r = 0; r < MROW8; ++r) {
        int row = base_row + r;
        if (row >= out_features) break;
        float result = block_reduce_sum(sums[r]);
        if (tid == 0) output[row] = result;
        if (r < MROW8 - 1) __syncthreads();
    }
}
#undef MROW8
#undef WX_WBUF_FLOATS8

extern "C" __global__ void q4km_dequant_f32(
    const uint8_t* __restrict__ W_packed,  // [n_rows * bytes_per_row]
    float* __restrict__ W_f32,             // [n_rows, in_features]
    const int n_rows,
    const int in_features
) {
    const int row = blockIdx.y;
    const int sb  = blockIdx.x;  // super-block index within row
    const int tid = threadIdx.x;

    if (row >= n_rows) return;
    const int n_sb = (in_features + QK_K - 1) / QK_K;
    if (sb >= n_sb) return;

    const uint8_t* block = W_packed + (row * n_sb + sb) * Q4K_BLOCK_SIZE;
    float* out = W_f32 + row * in_features + sb * QK_K;

    uint16_t d_bits  = block[0] | (block[1] << 8);
    uint16_t dm_bits = block[2] | (block[3] << 8);
    float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
    float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    // Canonical GGML Q4_K layout: 4 groups of 32 qs bytes.
    // Each group covers 2 sub-blocks: lo nibbles → sub 2*grp, hi nibbles → sub 2*grp+1.
    // Output: [32 lo values, 32 hi values] per group = 64 per group.
    // Each thread handles one byte (produces 2 values at stride 32).
    for (int grp = tid / 32; grp < 4; grp += blockDim.x / 32) {
        int l = tid % 32;
        uint8_t sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4(2 * grp,     scales, &sc_lo, &m_lo);
        get_scale_min_k4(2 * grp + 1, scales, &sc_hi, &m_hi);
        float d_lo  = d * (float)sc_lo;
        float d_hi  = d * (float)sc_hi;
        float dm_lo = dmin * (float)m_lo;
        float dm_hi = dmin * (float)m_hi;

        uint8_t byte = qs[grp * 32 + l];
        int base = grp * 64;
        int abs_lo = sb * QK_K + base + l;
        int abs_hi = sb * QK_K + base + 32 + l;
        if (abs_lo < in_features)
            out[base + l]      = d_lo * (float)(byte & 0xF) - dm_lo;
        if (abs_hi < in_features)
            out[base + 32 + l] = d_hi * (float)(byte >> 4)  - dm_hi;
    }
}

// ─── Q6_K Batch Dequantize (for prefill + cuBLAS SGEMM) ───
// Dequantize entire Q6K weight matrix to F32 on GPU.
// Grid: (n_superblocks, n_rows), Block: 256 threads (= QK_K values per superblock).
// Each thread produces 1 output value.

extern "C" __global__ void q6k_dequant_f32(
    const uint8_t* __restrict__ W_packed,  // [n_rows * bytes_per_row]
    float* __restrict__ W_f32,             // [n_rows, in_features]
    const int n_rows,
    const int in_features
) {
    const int row = blockIdx.y;
    const int sb  = blockIdx.x;  // super-block index within row
    const int tid = threadIdx.x; // 0..255 = position within superblock

    if (row >= n_rows) return;
    const int n_sb = (in_features + QK_K - 1) / QK_K;
    if (sb >= n_sb) return;

    const uint8_t* block = W_packed + (row * n_sb + sb) * Q6K_BLOCK_SIZE;
    float* out = W_f32 + row * in_features + sb * QK_K;

    const uint8_t* ql = block;           // 128 bytes
    const uint8_t* qh = block + 128;     // 64 bytes
    const int8_t* scales = (const int8_t*)(block + 192); // 16 bytes
    float d = __half2float(*reinterpret_cast<const __half*>(block + 208));

    // Thread mapping: same as matvec kernel
    const int grp = tid >> 7;        // 0 or 1
    const int pos = tid & 127;
    const int sub = pos >> 5;        // 0-3 (sub-block)
    const int l   = pos & 31;        // position within sub-block
    const int is  = l >> 4;          // 0 or 1

    int ql_off = grp * 64;
    int qh_off = grp * 32;
    int sc_off = grp * 8;

    uint8_t ql_byte = (sub & 1) ? ql[ql_off + l + 32] : ql[ql_off + l];
    int q_lo = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);

    uint8_t qh_byte = qh[qh_off + l];
    int qh_bits = (qh_byte >> (sub * 2)) & 3;

    int q = q_lo | (qh_bits << 4);
    float sc = (float)scales[sc_off + sub * 2 + is];

    int abs_pos = sb * QK_K + grp * 128 + sub * 32 + l;
    if (abs_pos < in_features)
        out[grp * 128 + sub * 32 + l] = d * sc * (float)(q - 32);
}

// ─── Q4_K_M Batch Dequantize to FP16 (for HGEMM prefill) ───
// Same as q4km_dequant_f32 but outputs __half for cuBLAS HGEMM.
// Half the scratch buffer size (136MB vs 272MB for gate/up).

extern "C" __global__ void q4km_dequant_f16(
    const uint8_t* __restrict__ W_packed,
    __half* __restrict__ W_f16,
    const int n_rows,
    const int in_features
) {
    const int row = blockIdx.y;
    const int sb  = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= n_rows) return;
    const int n_sb = (in_features + QK_K - 1) / QK_K;
    if (sb >= n_sb) return;

    const uint8_t* block = W_packed + (row * n_sb + sb) * Q4K_BLOCK_SIZE;
    __half* out = W_f16 + row * in_features + sb * QK_K;

    uint16_t d_bits  = block[0] | (block[1] << 8);
    uint16_t dm_bits = block[2] | (block[3] << 8);
    float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
    float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    for (int grp = tid / 32; grp < 4; grp += blockDim.x / 32) {
        int l = tid % 32;
        uint8_t sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4(2 * grp,     scales, &sc_lo, &m_lo);
        get_scale_min_k4(2 * grp + 1, scales, &sc_hi, &m_hi);
        float d_lo  = d * (float)sc_lo;
        float d_hi  = d * (float)sc_hi;
        float dm_lo = dmin * (float)m_lo;
        float dm_hi = dmin * (float)m_hi;

        uint8_t byte = qs[grp * 32 + l];
        int base = grp * 64;
        int abs_lo = sb * QK_K + base + l;
        int abs_hi = sb * QK_K + base + 32 + l;
        if (abs_lo < in_features)
            out[base + l]      = __float2half(d_lo * (float)(byte & 0xF) - dm_lo);
        if (abs_hi < in_features)
            out[base + 32 + l] = __float2half(d_hi * (float)(byte >> 4)  - dm_hi);
    }
}

// ─── FP32 → q8_1 Activation Quantizer (MMVQ Step 1) ─────────────
// Produces GGUF-compatible block_q8_1 layout for the upcoming dp4a
// q4km×q8_1 matvec path. Block size = 32 elements; per-block storage:
//
//   struct block_q8_1 {
//       half2  ds;        // d (scale) + s (sum * d), 4 bytes
//       int8_t qs[32];    // quantized values, 32 bytes
//   };                    // total 36 bytes — must match ggml-common.h
//
// Algorithm per block (matches llama.cpp quantize_q8_1):
//   d = max(|x|) / 127
//   q[i] = round(x[i] / d), clamped to [-128, 127]
//   s = d * sum(q[i])
//
// Launch: 1 warp (32 threads) per block, n_blocks = ceil(n / 32).
// Each thread owns one element; warp reductions compute amax + sum.

#define QK8_1 32
#define Q8_1_BLOCK_SIZE 36   // sizeof(block_q8_1) — must equal sizeof in ggml-common.h

// ─── Q4_K_M × q8_1 dp4a Matvec — MMVQ Step 2 ────────────────
// INT8 tensor-pipe: activation pre-quantized to q8_1 (36B blocks), weights kept
// in native Q4_K super-block layout (144B). Uses __dp4a (sm_61+) to compute
// 4×int8 dot products in a single instruction.
//
// For each super-block (256 elements):
//   - 8 sub-blocks × 32 int8s each, aligned 1:1 with 8 consecutive q8_1 blocks
//   - Within a sub-block: 8 int32s (32 int8s) decoded from Q4_K nibbles
//
// Thread layout (per CUDA block, 256 threads = 8 warps):
//   - Warp `w` (0..7) computes output row `base_row + w` entirely.
//   - Within a warp: lane = (sub << 2) | slot, where sub ∈ 0..7 and slot ∈ 0..3.
//     Each lane owns 2 packed int32s (slot and slot+4) of one sub-block.
//   - Per lane: sumi1 = dp4a(v0,u0) + dp4a(v1,u1), sumi2 = sum-of-u helper.
//     Scale with sc[sub], m[sub], d8[sub]; warp-reduce across all 32 lanes.
//
// Mirrors ggml/cuda vec_dot_q4_K_q8_1_impl_vmmq but reimplemented from scratch
// with FP32 accumulator (GGML uses FP32 too; we match for stability).
//
// Grid: ceil(out_features / 8) blocks, 256 threads each, no shmem (pure regs).

extern "C" __global__ void q4km_matvec_dp4a_f32(
    const uint8_t * __restrict__ W_q4k,      // [out_features * bytes_per_row]
    const void    * __restrict__ X_q8_1,     // [n_superblocks * 8] q8_1 blocks (36B each)
    float         * __restrict__ output,     // [out_features]
    const int      out_features,
    const int      in_features               // must be multiple of 256 (QK_K)
) {
    const int base_row = blockIdx.x * 8;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;            // 0..7 → which output row (within block)
    const int lane     = tid & 31;            // 0..31 within warp
    const int sub      = lane >> 2;           // 0..7 → sub-block index
    const int slot     = lane & 3;            // 0..3 → int32 position within 16-byte half

    const int row = base_row + warp_id;
    const bool row_active = (row < out_features);

    const int n_superblocks = in_features / QK_K;  // requires QK_K alignment
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    // Derived from sub: which group & which nibble (lo vs hi).
    // Group layout in qs: group `g` = 32 bytes starting at qs[g*32].
    //   - lo nibbles (byte & 0x0F) → sub-block (2*g)
    //   - hi nibbles (byte >> 4)   → sub-block (2*g+1)
    const int grp   = sub >> 1;               // 0..3
    const int is_hi = sub & 1;                // 0 or 1

    float partial = 0.0f;  // only warp lane 0 will own the final sum after reduction

    // q8_1 blocks: 8 per super-block, indexed contiguously.
    // Layout of block_q8_1 (36B): half2 ds (4B) + int8 qs[32] (32B).
    const uint8_t* q8_1_base = reinterpret_cast<const uint8_t*>(X_q8_1);

    for (int sb = 0; sb < n_superblocks; ++sb) {
        // ── Q4_K super-block layout ──────────────────────────────
        // Only proceed if this warp's row is valid; inactive warps still run
        // the loop to keep warp-uniform control flow and avoid divergence,
        // but their partial stays 0.
        const uint8_t* block = row_active
            ? (W_q4k + row * bytes_per_row + sb * Q4K_BLOCK_SIZE)
            : (W_q4k);  // dummy read base; contributions masked at end

        // block[0..2]  : d     (fp16)
        // block[2..4]  : dmin  (fp16)
        // block[4..16] : 12B packed scales/mins
        // block[16..144]: 128B qs (4 groups × 32 bytes)
        uint16_t d_bits  = block[0] | (block[1] << 8);
        uint16_t dm_bits = block[2] | (block[3] << 8);
        float d_w    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin_w = __half2float(*reinterpret_cast<const __half*>(&dm_bits));
        const uint8_t* scales = block + 4;
        const uint8_t* qs     = block + 16;

        uint8_t sc_u8, m_u8;
        get_scale_min_k4(sub, scales, &sc_u8, &m_u8);
        const int sc = (int)sc_u8;
        const int m  = (int)m_u8;

        // Load 2 int32s from Q4K qs (within this sub-block's group).
        // Group `grp` has 32 bytes = 8 int32s at qs + grp*32.
        // We read slot (0..3) and slot+4.
        const int* qs32 = reinterpret_cast<const int*>(qs + grp * 32);
        int q4_0 = qs32[slot];
        int q4_1 = qs32[slot + 4];

        // Extract this sub-block's nibbles: lo mask for is_hi=0, hi shift for is_hi=1.
        // Result: each byte of v0/v1 is an int8 in range [0, 15].
        int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
        int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

        // Load matching 2 int32s from q8_1 block `sub` (32 int8s = 8 int32s).
        const uint8_t* q8_1_block = q8_1_base + (size_t)(sb * 8 + sub) * Q8_1_BLOCK_SIZE;
        uint16_t d8_bits = q8_1_block[0] | (q8_1_block[1] << 8);
        float d8 = __half2float(*reinterpret_cast<const __half*>(&d8_bits));
        // s8 (block[2..4]) encodes d8 * Σq8 aggregated over all 32 elements — not
        // usable per-lane since each lane owns only 8 of the 32 values. We
        // recompute the sum via dp4a(0x01010101, u, ...) below.

        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = qs8[slot];
        int u1 = qs8[slot + 4];

        // dp4a: 4×int8 dot, accumulate into int32.
        int sumi = __dp4a(v0, u0, 0);
        sumi     = __dp4a(v1, u1, sumi);

        // FP32 accumulator (mirror reference's sumf_d / sumf_m split):
        //   row_sum = d_w*sc*d8*Σ(nibble*q8) - dmin_w*m*d8*Σ(q8)    per sub-block
        // Per lane owns 8 elements of the sub-block; warp-reduce at the end sums
        // (a) lane partials within each sub-block (4 lanes) AND
        // (b) across all 8 sub-blocks, since warp_reduce_sum uses a 32-lane mask.
        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        // Per-lane partials (scaled by this lane's sub-block constants).
        float lane_sumf_d = d_w    * (float)sc * (float)sumi  * d8;
        float lane_sumf_m = dmin_w * (float)m  * (float)sum_u * d8;

        partial += (lane_sumf_d - lane_sumf_m);
    }

    // Warp-reduce across all 32 lanes → lane 0 holds the final row sum.
    float result = warp_reduce_sum(partial);

    if (row_active && lane == 0) {
        output[row] = result;
    }
}

// ─── Q4_K_M × q8_1 dp4a Matvec V2 — MMVQ Step 6 ────────────────
// 1-row-per-block × 4-warp cross-warp reduction.
//
// Rationale vs v1 (8-rows/block, 1 warp/row):
//   v1 grid = ceil(out/8), so small matmuls (e.g. down_proj out=3584) launch
//   only 448 blocks → SM starvation on RTX 30/40 (≥68 SMs, ≥2 blocks/SM ideal).
//   v2 grid = out_features, so down_proj launches 3584 blocks → ~8× more
//   parallelism, saturates DRAM instead of leaving SMs idle.
//
// Thread layout (128 threads = 4 warps per block, 1 output row per block):
//   - All 4 warps cooperate on the same output row.
//   - Warp `w` (0..3) iterates sb = w, w+4, w+8, ...  (stride-4 superblocks).
//   - Within a warp: same (sub, slot) decomposition as v1 — lane owns
//     2 packed int32s of sub-block `lane>>2`, slot `lane&3`.
//   - Each warp intra-reduces its partial → lane 0 of each warp holds
//     that warp's contribution.
//   - 4 partials stored to shmem (tmp_shared[4], 16B).
//   - Warp 0 reads the 4 partials, sums, writes output[row].
//
// Math is BIT-IDENTICAL to v1: same dp4a pattern, same per-sub-block scale
// application, same FP32 accumulator. Only the *iteration partitioning*
// changes (sb-stride-4 across warps vs all-in-one-warp).
//
// Grid:  out_features blocks
// Block: 128 threads (4 warps × 32 lanes)
// Shmem: 16 B static (4 floats — one partial per warp)

__launch_bounds__(128, 8)
extern "C" __global__ void q4km_matvec_dp4a_v2_f32(
    const uint8_t * __restrict__ W_q4k,      // [out_features * bytes_per_row]
    const void    * __restrict__ X_q8_1,     // [n_superblocks * 8] q8_1 blocks (36B each)
    float         * __restrict__ output,     // [out_features]
    const int      out_features,
    const int      in_features               // must be multiple of 256 (QK_K)
) {
    const int row     = blockIdx.x;
    if (row >= out_features) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;             // 0..3 → which warp within block
    const int lane    = tid & 31;             // 0..31
    const int sub     = lane >> 2;            // 0..7 → sub-block within super-block
    const int slot    = lane & 3;             // 0..3 → int32 position within half

    const int n_superblocks = in_features / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

    // Sub-block → group (0..3) and lo/hi nibble select. Matches v1.
    const int grp   = sub >> 1;
    const int is_hi = sub & 1;

    float partial = 0.0f;

    const uint8_t* q8_1_base = reinterpret_cast<const uint8_t*>(X_q8_1);
    const uint8_t* row_base  = W_q4k + (size_t)row * bytes_per_row;

    // 4-warp stride-4 iteration: warp w takes sb = w, w+4, w+8, ...
    // Adjacent warps stride through sequential superblocks, so per-CTA L2
    // reuse on x_q8_1 is preserved (neighbour blocks also touch sb = w±1,
    // which the SM's L2 slice caches) — no need for shmem staging of x.
    for (int sb = warp_id; sb < n_superblocks; sb += 4) {
        const uint8_t* block = row_base + sb * Q4K_BLOCK_SIZE;

        // Q4_K super-block header — direct __half read (block is 4-byte
        // aligned so the 2-byte fp16 scalar is naturally aligned).
        float d_w    = __half2float(*reinterpret_cast<const __half*>(block));
        float dmin_w = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const uint8_t* scales = block + 4;
        const uint8_t* qs     = block + 16;

        uint8_t sc_u8, m_u8;
        get_scale_min_k4(sub, scales, &sc_u8, &m_u8);
        const int sc = (int)sc_u8;
        const int m  = (int)m_u8;

        // 2 int32s of nibbles for this lane's sub-block (group-aligned).
        const int* qs32 = reinterpret_cast<const int*>(qs + grp * 32);
        int q4_0 = __ldg(qs32 + slot);
        int q4_1 = __ldg(qs32 + slot + 4);
        int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
        int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

        // Matching q8_1 block — direct __half read on 4-byte aligned header.
        const uint8_t* q8_1_block = q8_1_base + (size_t)(sb * 8 + sub) * Q8_1_BLOCK_SIZE;
        float d8 = __half2float(*reinterpret_cast<const __half*>(q8_1_block));

        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = __ldg(qs8 + slot);
        int u1 = __ldg(qs8 + slot + 4);

        // VDR=2 dp4a: 2 × __dp4a per sub-block.
        int sumi = __dp4a(v0, u0, 0);
        sumi     = __dp4a(v1, u1, sumi);

        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        float lane_sumf_d = d_w    * (float)sc * (float)sumi  * d8;
        float lane_sumf_m = dmin_w * (float)m  * (float)sum_u * d8;

        partial += (lane_sumf_d - lane_sumf_m);
    }

    // Step 1: intra-warp reduce. After this, lane 0 of each warp holds the
    // full sum over this warp's assigned superblocks (sb = warp_id, +4, ...).
    float warp_sum = warp_reduce_sum(partial);

    // Step 2: cross-warp reduce via shmem. Only one partial per warp (lane 0).
    __shared__ float tmp_shared[4];
    if (lane == 0) {
        tmp_shared[warp_id] = warp_sum;
    }
    __syncthreads();

    // Warp 0 finalizes: 4 floats → single output scalar.
    if (warp_id == 0 && lane == 0) {
        float total = tmp_shared[0] + tmp_shared[1] + tmp_shared[2] + tmp_shared[3];
        output[row] = total;
    }
}

// ─── DP4A v2 × MROW4: 4 rows per block × 4 warps ────────────────
// Combines the dp4a_v2 math (1-row × 4-warp stride-4) with mrow4 pattern
// (4 output rows per block) to quarter the grid size + amortize the
// q8_1 activation reads in L1 across 4 rows. 128 threads = 4 warps.
// Each warp computes partials for ALL 4 rows using stride-4 sb walk;
// lane 0 of each warp writes per-row partials to shared memory;
// warp 0 finalizes a 4-value cross-warp reduce per row → 4 outputs.
__launch_bounds__(128, 8)
extern "C" __global__ void q4km_matvec_dp4a_mrow4_f32(
    const uint8_t * __restrict__ W_q4k,
    const void    * __restrict__ X_q8_1,
    float         * __restrict__ output,
    const int      out_features,
    const int      in_features
) {
    const int base_row = blockIdx.x * 4;
    if (base_row >= out_features) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int sub     = lane >> 2;
    const int slot    = lane & 3;

    const int n_superblocks = in_features / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const int grp   = sub >> 1;
    const int is_hi = sub & 1;

    float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint8_t* q8_1_base = reinterpret_cast<const uint8_t*>(X_q8_1);

    for (int sb = warp_id; sb < n_superblocks; sb += 4) {
        // q8_1 activation block (same for all 4 rows) — read once per sb.
        const uint8_t* q8_1_block = q8_1_base + (size_t)(sb * 8 + sub) * Q8_1_BLOCK_SIZE;
        float d8 = __half2float(*reinterpret_cast<const __half*>(q8_1_block));
        const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
        int u0 = __ldg(qs8 + slot);
        int u1 = __ldg(qs8 + slot + 4);
        int sum_u = __dp4a(0x01010101, u0, 0);
        sum_u     = __dp4a(0x01010101, u1, sum_u);

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = base_row + r;
            if (row >= out_features) break;
            const uint8_t* block = W_q4k + (size_t)row * bytes_per_row + sb * Q4K_BLOCK_SIZE;

            float d_w    = __half2float(*reinterpret_cast<const __half*>(block));
            float dmin_w = __half2float(*reinterpret_cast<const __half*>(block + 2));
            const uint8_t* scales = block + 4;
            const uint8_t* qs     = block + 16;

            uint8_t sc_u8, m_u8;
            get_scale_min_k4(sub, scales, &sc_u8, &m_u8);

            const int* qs32 = reinterpret_cast<const int*>(qs + grp * 32);
            int q4_0 = __ldg(qs32 + slot);
            int q4_1 = __ldg(qs32 + slot + 4);
            int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
            int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

            int sumi = __dp4a(v0, u0, 0);
            sumi     = __dp4a(v1, u1, sumi);

            float lane_d = d_w    * (float)sc_u8 * (float)sumi  * d8;
            float lane_m = dmin_w * (float)m_u8  * (float)sum_u * d8;
            partial[r] += (lane_d - lane_m);
        }
    }

    // Per-row warp-reduce, then cross-warp reduce.
    __shared__ float tmp_shared[4][4];  // [row][warp_id]
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        float warp_sum = warp_reduce_sum(partial[r]);
        if (lane == 0) tmp_shared[r][warp_id] = warp_sum;
    }
    __syncthreads();

    // Warp 0 finalizes all 4 rows (uses 4 threads, rest idle — trivial).
    if (warp_id == 0 && lane < 4) {
        const int r = lane;
        const int row = base_row + r;
        if (row < out_features) {
            float total = tmp_shared[r][0] + tmp_shared[r][1]
                        + tmp_shared[r][2] + tmp_shared[r][3];
            output[row] = total;
        }
    }
}

extern "C" __global__ void quantize_f32_to_q8_1_f32(
    const float* __restrict__ x,
    uint8_t* __restrict__ y,        // points at array of block_q8_1 (36B each)
    const int n_elements,           // total f32 elements (must be multiple of QK8_1)
    const int n_blocks              // ceil(n_elements / QK8_1)
) {
    const int block_id = blockIdx.x;
    const int lane     = threadIdx.x;       // 0..31 — exactly one warp
    if (block_id >= n_blocks) return;

    const int idx = block_id * QK8_1 + lane;
    const float xi = (idx < n_elements) ? x[idx] : 0.0f;

    // Full-warp reductions (warp_reduce_* in common.cuh use mask 0xFFFFFFFF
    // and butterfly XOR — result broadcast to all 32 lanes).
    float amax = warp_reduce_max(fabsf(xi));

    const float d     = amax / 127.0f;
    const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    // Quantize: round to nearest, clamp to int8 range.
    int qi = __float2int_rn(xi * d_inv);
    qi = max(-128, min(127, qi));
    const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)qi;

    // s = d * sum(q[i]) — every lane must participate in the reduction.
    float sum_q = warp_reduce_sum((float)q);

    // Block layout: [half2 ds (4B)][int8 qs[32] (32B)] = 36B
    uint8_t* block_ptr = y + (size_t)block_id * Q8_1_BLOCK_SIZE;
    int8_t* qs_ptr = (int8_t*)(block_ptr + 4);
    qs_ptr[lane] = q;

    if (lane == 0) {
        const float s = d * sum_q;
        half2 ds = __floats2half2_rn(d, s);
        *reinterpret_cast<half2*>(block_ptr) = ds;
    }
}
#undef QK8_1
#undef Q8_1_BLOCK_SIZE
