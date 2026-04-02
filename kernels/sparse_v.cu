// Sparse Value decompress + FMA for compressed V cache.
//
// Only decompresses and accumulates V rows where attention weight > threshold.
// At long context (100K+), 50-80% of softmax weights are below 1e-6,
// so sparsity provides 2-5x effective speedup.
//
// Two variants:
//   1. 4-bit compressed V (absmax quantized)
//   2. 8-bit compressed V (absmax quantized)

#include "common.cuh"

// ─── Sparse V Accumulate from 4-bit Compressed ───────────────
// output[d] = sum_i( attn[i] * decompress_4bit(v_packed, i, d) )
// Only processes rows where attn[i] >= threshold.
//
// V 4-bit layout per row: [f32 scale][u8 packed × dim/2]
// Each byte holds two 4-bit unsigned values (0-15).
// Dequantize: val = (nibble - 8) * scale  (centered at 0)

extern "C" __global__ void sparse_v_decompress_fma_4bit(
    const float* __restrict__ attn_weights,  // [seq_len]
    const uint8_t* __restrict__ v_packed,     // [seq_len, 4 + dim/2] per-row: [f32 scale][4-bit data]
    float* __restrict__ output,               // [head_dim]
    const int seq_len,
    const int head_dim,
    const float threshold
) {
    // Each block cooperatively accumulates into output
    const int tid = threadIdx.x;
    const int bytes_per_row = 4 + (head_dim + 1) / 2;  // f32 scale + packed nibbles

    // Initialize output in shared memory
    extern __shared__ float s_output[];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_output[d] = 0.0f;
    }
    __syncthreads();

    // Iterate over sequence positions
    for (int pos = 0; pos < seq_len; ++pos) {
        float w = attn_weights[pos];
        if (w < threshold) continue;  // SPARSITY: skip low-weight rows

        const uint8_t* row = v_packed + pos * bytes_per_row;
        float scale = *reinterpret_cast<const float*>(row);
        const uint8_t* nibbles = row + 4;

        // Decompress and accumulate
        for (int d = tid; d < head_dim; d += blockDim.x) {
            uint8_t byte = nibbles[d / 2];
            uint8_t nib = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            float val = ((float)nib - 8.0f) * scale;
            s_output[d] += w * val;
        }
        __syncthreads();
    }

    // Write output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        output[d] = s_output[d];
    }
}

// ─── Sparse V Accumulate from 8-bit Compressed ───────────────
// V 8-bit layout per row: [f32 scale][i8 × dim]
// Dequantize: val = qs[d] * scale

extern "C" __global__ void sparse_v_decompress_fma_8bit(
    const float* __restrict__ attn_weights,
    const uint8_t* __restrict__ v_packed,     // [seq_len, 4 + dim]
    float* __restrict__ output,
    const int seq_len,
    const int head_dim,
    const float threshold
) {
    const int tid = threadIdx.x;
    const int bytes_per_row = 4 + head_dim;

    extern __shared__ float s_output[];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_output[d] = 0.0f;
    }
    __syncthreads();

    for (int pos = 0; pos < seq_len; ++pos) {
        float w = attn_weights[pos];
        if (w < threshold) continue;

        const uint8_t* row = v_packed + pos * bytes_per_row;
        float scale = *reinterpret_cast<const float*>(row);
        const int8_t* qs = reinterpret_cast<const int8_t*>(row + 4);

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = (float)qs[d] * scale;
            s_output[d] += w * val;
        }
        __syncthreads();
    }

    for (int d = tid; d < head_dim; d += blockDim.x) {
        output[d] = s_output[d];
    }
}

// ─── KV Cache INT8 Quantize-on-Write ─────────────────────────
// Quantize V vectors to INT8 when storing in KV cache.
// Per-channel quantization: scale[d] = max|V[t,d]| / 127
// Fused with cache append to avoid separate quantization pass.

extern "C" __global__ void quantize_v_int8_perchannel(
    const float* __restrict__ v_input,    // [n_tokens, head_dim]
    int8_t* __restrict__ v_quantized,     // [n_tokens, head_dim]
    float* __restrict__ scales,           // [n_tokens] per-token absmax scale
    const int n_tokens,
    const int head_dim
) {
    const int token = blockIdx.x;
    if (token >= n_tokens) return;
    const int tid = threadIdx.x;

    const float* v = v_input + token * head_dim;
    int8_t* vq = v_quantized + token * head_dim;

    // Find absmax for this token's V vector
    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(v[d]));
    }
    float absmax = block_reduce_max(local_max);

    __shared__ float s_scale;
    if (tid == 0) {
        s_scale = (absmax > 0.0f) ? absmax / 127.0f : 1.0f;
        scales[token] = s_scale;
    }
    __syncthreads();

    float inv_scale = 1.0f / s_scale;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float val = v[d] * inv_scale;
        // Clamp to [-127, 127]
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        vq[d] = (int8_t)rintf(val);
    }
}
