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
#include <cuda_fp16.h>

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
// in_features must be divisible by QK_K (256).

extern "C" __global__ void q4km_matvec_f32(
    const uint8_t* __restrict__ W_packed,  // [out_features * bytes_per_row]
    const float* __restrict__ x,           // [in_features]
    float* __restrict__ output,            // [out_features]
    const int out_features,
    const int in_features
) {
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;

    const int n_superblocks = in_features / QK_K;
    const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;
    const uint8_t* w_row = W_packed + row * bytes_per_row;

    float partial_sum = 0.0f;

    // Each thread handles a subset of super-blocks
    for (int sb = tid; sb < n_superblocks; sb += blockDim.x) {
        const uint8_t* block = w_row + sb * Q4K_BLOCK_SIZE;
        const float* x_sb = x + sb * QK_K;

        // Decode f16 d, dmin
        uint16_t d_bits  = block[0] | (block[1] << 8);
        uint16_t dm_bits = block[2] | (block[3] << 8);
        // f16 → f32 (inline)
        float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));

        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;

        // Process 8 sub-blocks of 32 elements each
        for (int sub = 0; sub < 8; ++sub) {
            uint8_t sc, m;
            get_scale_min_k4(sub, scales, &sc, &m);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)m;

            // 16 bytes → 32 values (4-bit pairs)
            for (int j = 0; j < 16; ++j) {
                uint8_t byte = qs[sub * 16 + j];
                int idx = sub * 32 + j * 2;
                float v0 = d1 * (float)(byte & 0xF) - m1;
                float v1 = d1 * (float)(byte >> 4)   - m1;
                partial_sum += v0 * x_sb[idx] + v1 * x_sb[idx + 1];
            }
        }
    }

    // Block-level reduction
    partial_sum = block_reduce_sum(partial_sum);
    if (tid == 0) {
        output[row] = partial_sum;
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
            local += (float)qs[j] * x_b[j];
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
    const int n_sb = in_features / QK_K;
    if (sb >= n_sb) return;

    const uint8_t* block = W_packed + (row * n_sb + sb) * Q4K_BLOCK_SIZE;
    float* out = W_f32 + row * in_features + sb * QK_K;

    uint16_t d_bits  = block[0] | (block[1] << 8);
    uint16_t dm_bits = block[2] | (block[3] << 8);
    float d    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
    float dmin = __half2float(*reinterpret_cast<const __half*>(&dm_bits));
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    // Each thread handles elements within the super-block
    for (int sub = tid / 16; sub < 8; sub += blockDim.x / 16) {
        int j = tid % 16;
        uint8_t sc, m;
        get_scale_min_k4(sub, scales, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;

        uint8_t byte = qs[sub * 16 + j];
        int idx = sub * 32 + j * 2;
        out[idx]     = d1 * (float)(byte & 0xF) - m1;
        out[idx + 1] = d1 * (float)(byte >> 4)   - m1;
    }
}
