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
    float* __restrict__ means_out,            // [n_kv_heads] per-token means (or NULL)
    const float* __restrict__ channel_sigma,  // [head_dim] per-channel sigma (or NULL = per-vector)
    const int head_dim,
    const int n_centroids,                    // 2^bits (4, 8, or 16)
    const int bytes_per_key,                  // ceil(head_dim * bits / 8)
    const int center_keys                     // 1 = subtract per-token mean before Hadamard
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_boundaries = n_centroids - 1;

    extern __shared__ float s_mem[];
    float* s_data = s_mem;                    // [head_dim] for key + Hadamard
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

    // Step 1.5: Per-token mean removal (softmax shift-invariance)
    // Subtract mean BEFORE Hadamard → zero-mean vector → codebook gets more precision
    if (center_keys) {
        float partial_sum = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            partial_sum += s_data[i];
        }
        float mean = block_reduce_sum(partial_sum) / (float)head_dim;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_data[i] -= mean;
        }
        if (tid == 0 && means_out != nullptr) {
            means_out[head] = mean;
        }
        __syncthreads();
    }

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

    // KIVI per-channel sigma: compute mean_sigma for ratio calculation
    // When channel_sigma != NULL, effective_inv_sigma[d] = 1 / (per_vector_sigma * ratio[d])
    // where ratio[d] = channel_sigma[d] / mean(channel_sigma)
    // This gives each dimension its own quantization scale while preserving token magnitude.
    float mean_channel_sigma = 0.0f;
    if (channel_sigma != nullptr) {
        float partial = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            partial += channel_sigma[i];
        }
        mean_channel_sigma = block_reduce_sum(partial) / (float)head_dim;
    }

    // Step 4: Quantize + pack (4-bit: 2 values per byte)
    uint8_t* out = packed_out + head * bytes_per_key;

    // Helper: compute per-dimension inverse sigma
    #define INV_SIGMA_D(d) \
        (channel_sigma != nullptr \
            ? (mean_channel_sigma > 1e-10f ? (1.0f / (s_sigma * channel_sigma[d] / mean_channel_sigma)) : inv_sigma) \
            : inv_sigma)

    if (n_centroids == 16) {
        // 4-bit path: pack 2 indices per byte
        for (int pair = tid; pair < head_dim / 2; pair += blockDim.x) {
            int i0 = pair * 2;
            int i1 = pair * 2 + 1;

            // Quantize i0
            float v0 = s_data[i0] * INV_SIGMA_D(i0);
            uint8_t idx0 = 0;
            for (int b = 0; b < n_boundaries; ++b) {
                if (v0 > s_bounds[b]) idx0++;
                else break;
            }

            // Quantize i1
            float v1 = s_data[i1] * INV_SIGMA_D(i1);
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
                int d = quad * 4 + j;
                float v = s_data[d] * INV_SIGMA_D(d);
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

// ─── Grouped TQ Compress Kernel ───────────────────────────────
// Per-group sigma variant: head_dim is split into n_groups blocks of
// group_size each, and each block carries its own sigma derived from its
// own L2 norm. Matches the CPU compress_single_key_grouped path so the GPU
// pipeline can finally use per-block scales (Sprint 1A of compression-fix).
//
// Constraints: head_dim must be divisible by group_size. group_size is
// expected to be 16, 32, or 64 in practice.
// Center-keys + per-channel sigma (KIVI) deliberately omitted for now —
// they layer on top of per-vector sigma and would need redesign here.
//
// Grid: (n_kv_heads, 1, 1), Block: 128 threads
// Output: gnorms_out [n_kv_heads, n_groups]
//         packed_out [n_kv_heads, bytes_per_key]
extern "C" __global__ void tq_compress_key_grouped_f32(
    const float* __restrict__ key_vectors,    // [n_kv_heads, head_dim]
    const float* __restrict__ signs,          // [head_dim]
    const float* __restrict__ boundaries,     // [n_centroids - 1]
    const float* __restrict__ centroids,      // [n_centroids]
    uint8_t* __restrict__ packed_out,         // [n_kv_heads, bytes_per_key]
    float* __restrict__ gnorms_out,           // [n_kv_heads, n_groups]
    const int head_dim,
    const int group_size,
    const int n_centroids,                    // 2^bits (4, 8, or 16)
    const int bytes_per_key                   // ceil(head_dim * bits / 8)
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_boundaries = n_centroids - 1;
    const int n_groups = head_dim / group_size;

    extern __shared__ float s_mem[];
    float* s_data = s_mem;                    // [head_dim]
    float* s_bounds = s_mem + head_dim;       // [n_boundaries]
    float* s_cents = s_bounds + n_boundaries; // [n_centroids]

    // Load boundaries + centroids into shared memory once.
    for (int i = tid; i < n_boundaries; i += blockDim.x) {
        s_bounds[i] = boundaries[i];
    }
    for (int i = tid; i < n_centroids; i += blockDim.x) {
        s_cents[i] = centroids[i];
    }

    // Step 1: load key × signs.
    const float* key = key_vectors + head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_data[i] = key[i] * signs[i];
    }
    __syncthreads();

    // Step 2: in-place Hadamard.
    shared_hadamard(s_data, head_dim);

    // Step 3: per-group norm + sigma. Each thread handles a slice of groups
    // sequentially; n_groups is small (2-8) so this is cheap and avoids any
    // cross-thread reduction. Per-group sigma is published via shared memory
    // so the quantize step (which is parallel over dims) can read it.
    // gsig and gnorm reuse the tail of the dynamic shared memory region
    // (already declared above as s_mem), placed after s_cents:
    //   [head_dim] s_data
    //   [n_boundaries] s_bounds
    //   [n_centroids] s_cents
    //   [n_groups] s_gsig         (per-group inverse sigma)
    //   [n_groups] s_gnorm        (per-group raw norm)
    float* s_gsig = s_cents + n_centroids;
    float* s_gnorm = s_gsig + n_groups;

    if (tid == 0) {
        for (int g = 0; g < n_groups; ++g) {
            int d_start = g * group_size;
            float sq = 0.0f;
            for (int d = 0; d < group_size; ++d) {
                float v = s_data[d_start + d];
                sq += v * v;
            }
            float gnorm = sqrtf(sq);
            float sigma = gnorm / sqrtf((float)group_size);
            s_gnorm[g] = gnorm;
            s_gsig[g] = (sigma > 1e-10f) ? (1.0f / sigma) : 0.0f;
        }
    }
    __syncthreads();

    // Step 4: quantize + pack using per-group inverse sigma.
    uint8_t* out = packed_out + head * bytes_per_key;

    if (n_centroids == 16) {
        // 4-bit: pack 2 indices per byte. Each thread handles a pair.
        for (int pair = tid; pair < head_dim / 2; pair += blockDim.x) {
            int i0 = pair * 2;
            int i1 = i0 + 1;
            int g0 = i0 / group_size;
            int g1 = i1 / group_size;

            float v0 = s_data[i0] * s_gsig[g0];
            uint8_t idx0 = 0;
            for (int b = 0; b < n_boundaries; ++b) {
                if (v0 > s_bounds[b]) idx0++; else break;
            }

            float v1 = s_data[i1] * s_gsig[g1];
            uint8_t idx1 = 0;
            for (int b = 0; b < n_boundaries; ++b) {
                if (v1 > s_bounds[b]) idx1++; else break;
            }

            out[pair] = idx0 | (idx1 << 4);
        }
    } else if (n_centroids == 4) {
        // 2-bit: pack 4 indices per byte.
        for (int quad = tid; quad < head_dim / 4; quad += blockDim.x) {
            uint8_t packed = 0;
            for (int j = 0; j < 4; ++j) {
                int d = quad * 4 + j;
                int g = d / group_size;
                float v = s_data[d] * s_gsig[g];
                uint8_t idx = 0;
                for (int b = 0; b < n_boundaries; ++b) {
                    if (v > s_bounds[b]) idx++; else break;
                }
                packed |= (idx << (j * 2));
            }
            out[quad] = packed;
        }
    }
    __syncthreads();

    // Step 5: norm correction per group. We rebuild ||reconstructed_g||
    // from the just-packed indices and store norm² / ||recon|| so the
    // decoder side can faithfully reproduce magnitudes.
    if (tid == 0) {
        for (int g = 0; g < n_groups; ++g) {
            int d_start = g * group_size;
            float recon_sq = 0.0f;

            if (n_centroids == 16) {
                for (int d = d_start; d < d_start + group_size; d += 2) {
                    uint8_t byte = out[d / 2];
                    float r0 = s_cents[byte & 0xF];
                    float r1 = s_cents[byte >> 4];
                    // recon value = centroid * sigma_g
                    float sigma_g = (s_gsig[g] > 1e-20f) ? (1.0f / s_gsig[g]) : 0.0f;
                    r0 *= sigma_g;
                    r1 *= sigma_g;
                    recon_sq += r0 * r0 + r1 * r1;
                }
            } else if (n_centroids == 4) {
                for (int d = d_start; d < d_start + group_size; d += 4) {
                    uint8_t byte = out[d / 4];
                    float sigma_g = (s_gsig[g] > 1e-20f) ? (1.0f / s_gsig[g]) : 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        float r = s_cents[(byte >> (j * 2)) & 0x3] * sigma_g;
                        recon_sq += r * r;
                    }
                }
            }

            float recon_norm = sqrtf(recon_sq);
            float gnorm = s_gnorm[g];
            float corrected = (recon_norm > 1e-10f) ? (gnorm * gnorm / recon_norm) : gnorm;
            gnorms_out[head * n_groups + g] = corrected;
        }
    }
}

// ─── D2D Scatter: append compressed token to GpuCompressedKv ──
// Copies packed indices, norms, and V data from compress output to
// pre-allocated cache at the correct position. Zero CPU round-trip.
//
// Grid: (n_kv_heads, 1, 1), Block: 128 threads
// Each block handles one KV head's scatter.

extern "C" __global__ void tq_cache_scatter(
    const uint8_t* __restrict__ packed_src,   // [n_kv_heads, bytes_per_key]
    const float* __restrict__ norms_src,      // [n_kv_heads]
    const float* __restrict__ v_src,          // [n_kv_heads, head_dim]
    uint8_t* __restrict__ packed_dst,         // [n_kv_heads, max_seq, bytes_per_key]
    float* __restrict__ norms_dst,            // [n_kv_heads, max_seq]
    float* __restrict__ v_dst,                // [n_kv_heads, max_seq, head_dim]
    const int pos,                            // write position in cache
    const int max_seq,
    const int head_dim,
    const int bytes_per_key
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;

    // Scatter packed indices
    const uint8_t* src_packed = packed_src + head * bytes_per_key;
    uint8_t* dst_packed = packed_dst + (head * max_seq + pos) * bytes_per_key;
    for (int i = tid; i < bytes_per_key; i += blockDim.x) {
        dst_packed[i] = src_packed[i];
    }

    // Scatter norm (single float)
    if (tid == 0) {
        norms_dst[head * max_seq + pos] = norms_src[head];
    }

    // Scatter V data
    const float* src_v = v_src + head * head_dim;
    float* dst_v = v_dst + (head * max_seq + pos) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        dst_v[i] = src_v[i];
    }
}
