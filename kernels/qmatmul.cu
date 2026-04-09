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

#include "common_v2.cuh"

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
