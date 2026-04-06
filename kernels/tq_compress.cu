// GPU TurboQuant key compression: Hadamard + codebook quantize + pack.
//
// Replaces CPU-side compress_single_key_with_signs for decode (seq_len=1).
// One block per KV head. Each block processes one key vector (dim values).
//
// Pipeline: key → sign flip → Hadamard → normalize → quantize → pack → store
//
// References:
//   TurboQuant (ICLR 2026): Lloyd-Max codebook on Hadamard-rotated keys
//   tq-kv 3-Fix: sink tokens + POQ + cache reset

#include "common.cuh"

#define MAX_DIM 256
#define MAX_CENTROIDS 16  // 4-bit

// ─── Fast Walsh-Hadamard Transform (in-place, shared memory) ───
// O(d log d) butterfly. Input pre-multiplied by random signs.
__device__ void shared_hadamard(float* s_data, int dim) {
    int tid = threadIdx.x;
    for (int step = 1; step < dim; step <<= 1) {
        __syncthreads();
        for (int i = tid; i < dim / 2; i += blockDim.x) {
            int j = i / step * (step * 2) + i % step;
            int k = j + step;
            float a = s_data[j];
            float b = s_data[k];
            s_data[j] = a + b;
            s_data[k] = a - b;
        }
    }
    __syncthreads();
    // Normalize
    float scale = rsqrtf((float)dim);
    for (int i = tid; i < dim; i += blockDim.x) {
        s_data[i] *= scale;
    }
    __syncthreads();
}

// ─── TQ Compress Kernel ──────────────────────────────────────
// Grid: (n_kv_heads, 1, 1), Block: 128 threads
// Input:  key_vectors [n_kv_heads, head_dim] (flat, post-RoPE or pre-RoPE)
// Output: packed_indices [n_kv_heads, bytes_per_key]
//         norms [n_kv_heads]
//
// signs: [head_dim] randomized Hadamard signs (+1/-1)
// boundaries: [n_centroids - 1] Lloyd-Max decision boundaries (normalized)
// centroids: [n_centroids] Lloyd-Max centroids (normalized)

extern "C" __global__ void tq_compress_key_f32(
    const float* __restrict__ key_vectors,    // [n_kv_heads, head_dim]
    const float* __restrict__ signs,          // [head_dim]
    const float* __restrict__ boundaries,     // [n_boundaries] = n_centroids - 1
    const float* __restrict__ centroids,      // [n_centroids]
    uint8_t* __restrict__ packed_out,         // [n_kv_heads, bytes_per_key]
    float* __restrict__ norms_out,            // [n_kv_heads]
    const int head_dim,
    const int n_centroids,                    // 2^bits (4, 8, or 16)
    const int bytes_per_key                   // ceil(head_dim * bits / 8)
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_boundaries = n_centroids - 1;

    extern __shared__ float s_mem[];
    float* s_data = s_mem;                    // [head_dim] for key + Hadamard
    // boundaries + centroids loaded into shared mem after s_data
    float* s_bounds = s_mem + head_dim;       // [n_boundaries]
    float* s_cents = s_bounds + n_boundaries; // [n_centroids]

    // Load boundaries + centroids into shared memory
    for (int i = tid; i < n_boundaries; i += blockDim.x) {
        s_bounds[i] = boundaries[i];
    }
    for (int i = tid; i < n_centroids; i += blockDim.x) {
        s_cents[i] = centroids[i];
    }

    // Step 1: Load key vector × signs into shared memory
    const float* key = key_vectors + head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_data[i] = key[i] * signs[i];
    }
    __syncthreads();

    // Step 2: In-place Hadamard transform
    shared_hadamard(s_data, head_dim);

    // Step 3: Compute norm
    float partial_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        partial_sq += s_data[i] * s_data[i];
    }
    partial_sq = block_reduce_sum(partial_sq);
    __shared__ float s_norm, s_sigma, s_inv_sigma;
    if (tid == 0) {
        s_norm = sqrtf(partial_sq);
        s_sigma = s_norm / sqrtf((float)head_dim);
        s_inv_sigma = (s_sigma > 1e-10f) ? (1.0f / s_sigma) : 0.0f;
    }
    __syncthreads();

    float inv_sigma = s_inv_sigma;

    // Step 4: Quantize + pack (4-bit: 2 values per byte)
    // Thread i handles values at positions tid, tid+blockDim.x, ...
    uint8_t* out = packed_out + head * bytes_per_key;

    if (n_centroids == 16) {
        // 4-bit path: pack 2 indices per byte
        for (int pair = tid; pair < head_dim / 2; pair += blockDim.x) {
            int i0 = pair * 2;
            int i1 = pair * 2 + 1;

            // Quantize i0
            float v0 = s_data[i0] * inv_sigma;
            uint8_t idx0 = 0;
            for (int b = 0; b < n_boundaries; ++b) {
                if (v0 > s_bounds[b]) idx0++;
                else break;
            }

            // Quantize i1
            float v1 = s_data[i1] * inv_sigma;
            uint8_t idx1 = 0;
            for (int b = 0; b < n_boundaries; ++b) {
                if (v1 > s_bounds[b]) idx1++;
                else break;
            }

            out[pair] = idx0 | (idx1 << 4);
        }
    } else if (n_centroids == 4) {
        // 2-bit path: pack 4 indices per byte
        for (int quad = tid; quad < head_dim / 4; quad += blockDim.x) {
            uint8_t packed = 0;
            for (int j = 0; j < 4; ++j) {
                float v = s_data[quad * 4 + j] * inv_sigma;
                uint8_t idx = 0;
                for (int b = 0; b < n_boundaries; ++b) {
                    if (v > s_bounds[b]) idx++;
                    else break;
                }
                packed |= (idx << (j * 2));
            }
            out[quad] = packed;
        }
    }
    __syncthreads();

    // Step 5: Norm correction — compute ||reconstructed|| for corrected norm
    // corrected_norm = norm² / ||reconstructed||
    float partial_recon_sq = 0.0f;
    if (n_centroids == 16) {
        for (int pair = tid; pair < head_dim / 2; pair += blockDim.x) {
            uint8_t byte = out[pair];
            float r0 = s_cents[byte & 0xF] * s_sigma;
            float r1 = s_cents[byte >> 4] * s_sigma;
            partial_recon_sq += r0 * r0 + r1 * r1;
        }
    } else if (n_centroids == 4) {
        for (int quad = tid; quad < head_dim / 4; quad += blockDim.x) {
            uint8_t byte = out[quad];
            for (int j = 0; j < 4; ++j) {
                float r = s_cents[(byte >> (j * 2)) & 0x3] * s_sigma;
                partial_recon_sq += r * r;
            }
        }
    }
    partial_recon_sq = block_reduce_sum(partial_recon_sq);

    if (tid == 0) {
        float recon_norm = sqrtf(partial_recon_sq);
        float norm = s_norm;
        norms_out[head] = (recon_norm > 1e-10f) ? (norm * norm / recon_norm) : norm;
    }
}
