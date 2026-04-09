// Batched Fast Walsh-Hadamard Transform (inverse) for key decompression.
//
// Parallel butterfly network: each block handles one vector.
// For head_dim=128: 7 stages, 64 butterfly operations each.
// Entirely in shared memory — no global memory access between stages.
//
// Used in the decompress path when fused attention is not applicable
// (Pre-RoPE mode, compaction active, GPU).

#include "common.cuh"

#define MAX_DIM 256

// ─── Inverse Randomized Hadamard Transform ───────────────────
// x = D^T @ H @ x_rotated (undo sign flips + Hadamard)
//
// Signs array encodes the random diagonal D: signs[i] ∈ {-1, +1}.
// H is the normalized Walsh-Hadamard matrix (self-inverse up to scaling).
//
// Since H and D are self-inverse (H^T = H, D^T = D), the inverse is:
//   original = signs * WHT(x_rotated) * sqrt(dim)
// Wait, actually for our encoding:
//   forward:  x_rot = (1/sqrt(d)) * H * (signs * x)
//   inverse:  x = signs * H * x_rot  (the 1/sqrt(d) is absorbed since H is involutory at scale sqrt(d))
//
// Actually: forward = (1/sqrt(d)) * H * D * x
//           inverse = D * H * x_rot * (1/sqrt(d))  but H*H = d*I
// So: D * (1/sqrt(d)) * H * x_rot → wait, let me be precise:
//   forward: y = (1/√d) * H * D * x
//   then: √d * y = H * D * x
//   inverse: D * H * (√d * y) = D * H * H * D * x = D * d * D * x = d * x
//   so: x = (1/d) * D * H * (√d * y) = (1/√d) * D * H * y
//
// Final: inverse(y) = (1/√d) * D * H * y  (same operation as forward!)
// WHT is its own inverse (involutory) when properly normalized.

extern "C" __global__ void hadamard_inverse_batch_f32(
    float* __restrict__ data,          // [n_vectors, dim] — in-place
    const float* __restrict__ signs,   // [dim] — random sign flips
    const int n_vectors,
    const int dim
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid = threadIdx.x;
    float* x = data + vec_idx * dim;

    // Load into shared memory
    __shared__ float s_data[MAX_DIM];
    for (int i = tid; i < dim; i += blockDim.x) {
        s_data[i] = x[i];
    }
    __syncthreads();

    // Walsh-Hadamard butterfly: log2(dim) stages
    // Stage h: pairs (i, i+h) for i in [0, dim) step 2h
    for (int h = 1; h < dim; h <<= 1) {
        for (int i = tid; i < dim / 2; i += blockDim.x) {
            // Map thread to butterfly pair
            int block_idx = i / h;
            int offset = i % h;
            int j = block_idx * (2 * h) + offset;

            float a = s_data[j];
            float b = s_data[j + h];
            s_data[j]     = a + b;
            s_data[j + h] = a - b;
        }
        __syncthreads();
    }

    // Apply normalization (1/sqrt(dim)) and sign flips
    float inv_sqrt_dim = rsqrtf((float)dim);
    for (int i = tid; i < dim; i += blockDim.x) {
        x[i] = s_data[i] * inv_sqrt_dim * signs[i];
    }
}

// ─── Forward Randomized Hadamard (for key compression on GPU) ─
// y = (1/√d) * H * D * x
// D = diag(signs), H = Walsh-Hadamard

extern "C" __global__ void hadamard_forward_batch_f32(
    float* __restrict__ data,
    const float* __restrict__ signs,
    const int n_vectors,
    const int dim
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid = threadIdx.x;
    float* x = data + vec_idx * dim;

    __shared__ float s_data[MAX_DIM];

    // Apply sign flips first: D * x
    for (int i = tid; i < dim; i += blockDim.x) {
        s_data[i] = x[i] * signs[i];
    }
    __syncthreads();

    // Walsh-Hadamard butterfly
    for (int h = 1; h < dim; h <<= 1) {
        for (int i = tid; i < dim / 2; i += blockDim.x) {
            int block_idx = i / h;
            int offset = i % h;
            int j = block_idx * (2 * h) + offset;

            float a = s_data[j];
            float b = s_data[j + h];
            s_data[j]     = a + b;
            s_data[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize
    float inv_sqrt_dim = rsqrtf((float)dim);
    for (int i = tid; i < dim; i += blockDim.x) {
        x[i] = s_data[i] * inv_sqrt_dim;
    }
}
