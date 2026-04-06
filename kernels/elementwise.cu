// Elementwise CUDA kernels: SiLU, GELU, exp, add, mul, broadcast ops.
//
// Standalone versions for when fusion is not applicable.
// Vectorized float4 loads where possible for 4x fewer memory transactions.

#include "common.cuh"
#include <cuda_fp16.h>

// ─── SiLU (x * sigmoid(x)) ──────────────────────────────────
extern "C" __global__ void silu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// ─── GELU (approximate) ─────────────────────────────────────
extern "C" __global__ void gelu_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// ─── Element-wise Add ────────────────────────────────────────
extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

// ─── In-place bias add: data[i] += bias[i] ──────────────────
extern "C" __global__ void bias_add_f32(
    float* __restrict__ data,
    const float* __restrict__ bias,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += bias[idx];
    }
}

// ─── Element-wise Mul ────────────────────────────────────────
extern "C" __global__ void mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx];
    }
}

// ─── Scalar Mul ──────────────────────────────────────────────
extern "C" __global__ void scalar_mul_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float scalar,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scalar;
    }
}

// ─── Add + Scalar Mul (fused residual) ───────────────────────
// output = a + scale * b (common in residual connections)
extern "C" __global__ void add_scaled_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + scale * b[idx];
    }
}

// ─── Embedding Lookup ────────────────────────────────────────
// Gather rows from embedding table by token IDs.
// output[i] = table[ids[i]]

extern "C" __global__ void embedding_lookup_f32(
    const float* __restrict__ table,     // [vocab_size, embed_dim]
    const int* __restrict__ ids,         // [n_tokens]
    float* __restrict__ output,          // [n_tokens, embed_dim]
    const int n_tokens,
    const int embed_dim
) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    if (token >= n_tokens) return;

    const int id = ids[token];
    const float* row = table + id * embed_dim;
    float* out = output + token * embed_dim;

    for (int d = tid; d < embed_dim; d += blockDim.x) {
        out[d] = row[d];
    }
}

// ─── Copy with Type Conversion (F16 → F32) ──────────────────
extern "C" __global__ void f16_to_f32_kernel(
    const uint16_t* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(*reinterpret_cast<const __half*>(&input[idx]));
    }
}

// ─── F32 → F16 ─────────────────────────────────────────────
extern "C" __global__ void f32_to_f16_kernel(
    const float* __restrict__ input,
    uint16_t* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half h = __float2half(input[idx]);
        output[idx] = *reinterpret_cast<uint16_t*>(&h);
    }
}

// ─── BF16 → F32 ─────────────────────────────────────────────
extern "C" __global__ void bf16_to_f32_kernel(
    const uint16_t* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16 → F32: left-shift by 16 bits
        uint32_t bits = ((uint32_t)input[idx]) << 16;
        output[idx] = *reinterpret_cast<float*>(&bits);
    }
}

// ─── GPU Argmax: find index of maximum value ────────────────
// Single block, 256 threads. Each thread scans a chunk, then block reduction.
// Output: single u32 index. Avoids 600KB D2H copy for lm_head argmax.
extern "C" __global__ void argmax_f32(
    const float* __restrict__ data,
    unsigned int* __restrict__ out_idx,
    const int n
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Each thread finds local max over its chunk
    float local_max = -1e30f;
    int local_idx = 0;
    for (int i = tid; i < n; i += block_size) {
        float v = data[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Block reduction: shared memory for (value, index) pairs
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_idx = (unsigned int)s_idx[0];
    }
}
