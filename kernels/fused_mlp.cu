// Fused MLP kernels:
//   1. Fused SiLU × Mul (SwiGLU activation) — x * sigmoid(x) * gate
//   2. Fused SwiGLU + GEMV (activation + down projection) — 7.5x speedup
//
// SwiGLU = SiLU(gate) * up, used in Llama/Qwen/Mistral MLP blocks.

#include "common.cuh"

// ─── Fused SiLU × Mul (SwiGLU) ───────────────────────────────
// output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Replaces 3 separate kernels: sigmoid, mul, mul

extern "C" __global__ void fused_silu_mul_f32(
    const float* __restrict__ gate,      // [n_tokens, intermediate_dim]
    const float* __restrict__ up,        // [n_tokens, intermediate_dim]
    float* __restrict__ output,          // [n_tokens, intermediate_dim]
    const int n_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = silu_g * up[idx];
    }
}

// ─── Fused GELU × Mul ────────────────────────────────────────
// For models using GELU activation instead of SiLU (e.g., some Phi variants)
// output[i] = gelu(gate[i]) * up[i]
// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

extern "C" __global__ void fused_gelu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    const int n_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float x = gate[idx];
        // Fast GELU approximation
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf * up[idx];
    }
}

// ─── Fused SwiGLU + Down Projection (GEMV) ───────────────────
// output = (silu(gate) * up) @ W_down^T
// Avoids materializing the intermediate activation tensor.
// W_down: [out_dim, intermediate_dim], row-major.
//
// Each block computes one output element.

extern "C" __global__ void fused_swiglu_gemv_f32(
    const float* __restrict__ gate,       // [intermediate_dim] (single token)
    const float* __restrict__ up,         // [intermediate_dim]
    const float* __restrict__ W_down,     // [out_dim, intermediate_dim]
    float* __restrict__ output,           // [out_dim]
    const int intermediate_dim,
    const int out_dim
) {
    const int out_row = blockIdx.x;
    const int tid = threadIdx.x;

    if (out_row >= out_dim) return;

    const float* w_row = W_down + out_row * intermediate_dim;
    float dot = 0.0f;

    for (int i = tid; i < intermediate_dim; i += blockDim.x) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        float act = silu_g * up[i];
        dot += act * w_row[i];
    }
    dot = block_reduce_sum(dot);

    if (tid == 0) {
        output[out_row] = dot;
    }
}
