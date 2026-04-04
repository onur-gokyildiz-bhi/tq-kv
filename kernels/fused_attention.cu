// TQ-SPECIFIC: Fused attention from compressed KV cache indices.
//
// NO ANALOG in candle, llama.cpp, vLLM, or any other framework.
// This is tq-kv's competitive moat.
//
// Computes attention scores directly from packed TurboQuant indices
// without ever decompressing keys to FP32. The pre-rotated query trick
// exploits Hadamard orthogonality: <q, R^-1 k> = <Rq, k_compressed>.
//
// Algorithm:
//   1. Pre-rotate query with Hadamard signs (once per head)
//   2. For each compressed key position:
//      a. Unpack 2/3/4-bit indices from packed bytes
//      b. Gather centroids from LUT in shared memory
//      c. Compute: score = sum(rotated_q[d] * centroid[idx[d]] * sigma)
//   3. Output raw attention scores (softmax applied separately)

#include "common.cuh"

#define MAX_CENTROIDS 16  // 4-bit = 16 centroids
#define MAX_HEAD_DIM 256

// ─── Fused TQ Attention Scores ───────────────────────────────
// One block per query head. Threads split key positions.
//
// rotated_query: pre-rotated with Hadamard signs [n_heads, head_dim]
// packed_indices: [n_kv_heads, n_keys, bytes_per_key]
// norms: [n_kv_heads, n_keys] — per-key L2 norms
// centroids: [n_centroids] — Lloyd-Max codebook
// scores_out: [n_heads, n_keys] — output attention scores (before softmax)

extern "C" __global__ void tq_fused_attention_f32(
    const float* __restrict__ rotated_query,   // [n_heads, head_dim]
    const uint8_t* __restrict__ packed_indices, // [n_kv_heads, n_keys * bytes_per_key]
    const float* __restrict__ norms,            // [n_kv_heads, n_keys]
    const float* __restrict__ centroids,        // [n_centroids]
    float* __restrict__ scores_out,             // [n_heads, n_keys]
    const int n_heads,
    const int n_kv_heads,
    const int n_keys,
    const int head_dim,
    const int bits,                             // 2, 3, or 4
    const float scale                           // 1/sqrt(head_dim)
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head = head_idx / (n_heads / n_kv_heads);  // GQA mapping

    // Load centroids to shared memory (small: 4/8/16 floats)
    __shared__ float s_centroids[MAX_CENTROIDS];
    const int n_centroids = 1 << bits;
    if (tid < n_centroids) {
        s_centroids[tid] = centroids[tid];
    }

    // Load pre-rotated query to shared memory
    __shared__ float s_q[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = rotated_query[head_idx * head_dim + d];
    }
    __syncthreads();

    // Compute bytes per key
    const int bytes_per_key = (head_dim * bits + 7) / 8;
    const uint8_t* kv_packed = packed_indices + kv_head * n_keys * bytes_per_key;
    const float* kv_norms = norms + kv_head * n_keys;

    // Each thread handles a subset of key positions
    for (int k = tid; k < n_keys; k += blockDim.x) {
        float norm = kv_norms[k];

        // Skip near-zero keys
        if (norm < 1e-10f) {
            scores_out[head_idx * n_keys + k] = 0.0f;
            continue;
        }

        float sigma = norm / sqrtf((float)head_dim);
        const uint8_t* key_packed = kv_packed + k * bytes_per_key;

        // Unpack indices and compute fused dot product
        float dot = 0.0f;

        if (bits == 4) {
            // 4-bit: 2 indices per byte
            for (int d = 0; d < head_dim; d += 2) {
                uint8_t byte = key_packed[d / 2];
                float c0 = s_centroids[byte & 0xF];
                float c1 = s_centroids[byte >> 4];
                dot += s_q[d] * c0 + s_q[d + 1] * c1;
            }
        } else if (bits == 2) {
            // 2-bit: 4 indices per byte
            for (int d = 0; d < head_dim; d += 4) {
                uint8_t byte = key_packed[d / 4];
                dot += s_q[d]     * s_centroids[byte & 3];
                dot += s_q[d + 1] * s_centroids[(byte >> 2) & 3];
                dot += s_q[d + 2] * s_centroids[(byte >> 4) & 3];
                dot += s_q[d + 3] * s_centroids[(byte >> 6) & 3];
            }
        } else if (bits == 3) {
            // 3-bit: irregular packing, handle carefully
            int bit_offset = 0;
            for (int d = 0; d < head_dim; ++d) {
                int byte_idx = bit_offset / 8;
                int bit_pos  = bit_offset % 8;
                uint8_t idx;
                if (bit_pos <= 5) {
                    idx = (key_packed[byte_idx] >> bit_pos) & 7;
                } else {
                    // Straddles byte boundary
                    idx = (key_packed[byte_idx] >> bit_pos) |
                          ((key_packed[byte_idx + 1] << (8 - bit_pos)) & 7);
                    idx &= 7;
                }
                dot += s_q[d] * s_centroids[idx];
                bit_offset += 3;
            }
        }

        scores_out[head_idx * n_keys + k] = dot * sigma * scale;
    }
}

// ─── Fused TQ Attention with Per-Group Norms ─────────────────
// When using grouped quantization (TQ_GROUP=32), each group of 32
// dimensions has its own norm/sigma. More accurate but slightly slower.

extern "C" __global__ void tq_fused_attention_grouped_f32(
    const float* __restrict__ rotated_query,
    const uint8_t* __restrict__ packed_indices,
    const float* __restrict__ group_norms,      // [n_kv_heads, n_keys, n_groups]
    const float* __restrict__ centroids,
    float* __restrict__ scores_out,
    const int n_heads,
    const int n_kv_heads,
    const int n_keys,
    const int head_dim,
    const int bits,
    const int group_size,                       // typically 32
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    __shared__ float s_centroids[MAX_CENTROIDS];
    __shared__ float s_q[MAX_HEAD_DIM];

    const int n_centroids = 1 << bits;
    if (tid < n_centroids) s_centroids[tid] = centroids[tid];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = rotated_query[head_idx * head_dim + d];
    }
    __syncthreads();

    const int bytes_per_key = (head_dim * bits + 7) / 8;
    const int n_groups = head_dim / group_size;
    const uint8_t* kv_packed = packed_indices + kv_head * n_keys * bytes_per_key;
    const float* kv_gnorms = group_norms + kv_head * n_keys * n_groups;

    for (int k = tid; k < n_keys; k += blockDim.x) {
        const uint8_t* key_packed = kv_packed + k * bytes_per_key;
        const float* gnorms = kv_gnorms + k * n_groups;

        float dot = 0.0f;

        for (int g = 0; g < n_groups; ++g) {
            float sigma = gnorms[g] / sqrtf((float)group_size);
            int d_start = g * group_size;

            if (bits == 4) {
                for (int d = d_start; d < d_start + group_size; d += 2) {
                    uint8_t byte = key_packed[d / 2];
                    dot += sigma * (s_q[d] * s_centroids[byte & 0xF] +
                                    s_q[d + 1] * s_centroids[byte >> 4]);
                }
            } else if (bits == 2) {
                for (int d = d_start; d < d_start + group_size; d += 4) {
                    uint8_t byte = key_packed[d / 4];
                    dot += sigma * (
                        s_q[d]     * s_centroids[byte & 3] +
                        s_q[d + 1] * s_centroids[(byte >> 2) & 3] +
                        s_q[d + 2] * s_centroids[(byte >> 4) & 3] +
                        s_q[d + 3] * s_centroids[(byte >> 6) & 3]
                    );
                }
            }
        }

        scores_out[head_idx * n_keys + k] = dot * scale;
    }
}

// ─── Full Fused TQ Decode Attention ─────────────────────────
// Single kernel: compressed score + online softmax + V accumulation.
// For decode (seq_len=1): one query token attending to all cached keys.
//
// Each thread block handles ONE query head completely:
//   1. Load query (pre-rotated) to shared memory
//   2. Stream over KV tokens in chunks:
//      a. Decompress key → compute score (same as tq_fused_attention_f32)
//      b. Online softmax: track running max + sum_exp
//      c. Load V row, accumulate: output += softmax_weight * v
//   3. Final rescale by 1/sum_exp, write output
//
// Parallelism: threads split the head_dim for V accumulation.
// Sequential over KV tokens (online softmax requires serial scan).

extern "C" __global__ void tq_fused_decode_attention_f32(
    const float* __restrict__ rotated_query,   // [n_heads, head_dim] (pre-rotated with Hadamard)
    const uint8_t* __restrict__ packed_indices, // [n_kv_heads, n_keys * bytes_per_key]
    const float* __restrict__ norms,            // [n_kv_heads, n_keys]
    const float* __restrict__ centroids,        // [n_centroids]
    const float* __restrict__ V,               // [n_kv_heads, n_keys, head_dim] (uncompressed FP32)
    float* __restrict__ output,                // [n_heads, head_dim]
    const int n_heads,
    const int n_kv_heads,
    const int n_keys,
    const int head_dim,
    const int bits,
    const float scale                          // 1/sqrt(head_dim)
) {
    const int head_idx = blockIdx.x;
    if (head_idx >= n_heads) return;
    const int tid = threadIdx.x;
    const int kv_head = head_idx / (n_heads / n_kv_heads);  // GQA mapping

    // --- Shared memory ---
    __shared__ float s_centroids[MAX_CENTROIDS];
    __shared__ float s_q[MAX_HEAD_DIM];
    __shared__ float s_max;     // running max score
    __shared__ float s_sum_exp; // running sum of exp(score - max)

    const int n_centroids = 1 << bits;
    if (tid < n_centroids) s_centroids[tid] = centroids[tid];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = rotated_query[head_idx * head_dim + d];
    }
    __syncthreads();

    // Per-thread V accumulator (each thread handles a subset of head_dim)
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // unrolled for dims tid, tid+blockDim, ...
    float running_max = -1e10f;
    float running_sum = 0.0f;

    const int bytes_per_key = (head_dim * bits + 7) / 8;
    const uint8_t* kv_packed = packed_indices + kv_head * n_keys * bytes_per_key;
    const float* kv_norms = norms + kv_head * n_keys;
    const float* kv_v = V + kv_head * n_keys * head_dim;

    // --- Stream over KV tokens ---
    for (int k = 0; k < n_keys; ++k) {
        float norm_k = kv_norms[k];

        // Compute score: thread 0 does full dot product (head_dim is small, ~128)
        // All threads compute the score cooperatively then broadcast.
        float partial_dot = 0.0f;
        const uint8_t* key_packed = kv_packed + k * bytes_per_key;

        if (bits == 4) {
            // Each thread handles a subset of dimensions
            for (int d = tid * 2; d < head_dim; d += blockDim.x * 2) {
                uint8_t byte = key_packed[d / 2];
                partial_dot += s_q[d] * s_centroids[byte & 0xF]
                             + s_q[d + 1] * s_centroids[byte >> 4];
            }
        } else if (bits == 2) {
            for (int d = tid * 4; d < head_dim; d += blockDim.x * 4) {
                uint8_t byte = key_packed[d / 4];
                partial_dot += s_q[d]     * s_centroids[byte & 3]
                             + s_q[d + 1] * s_centroids[(byte >> 2) & 3]
                             + s_q[d + 2] * s_centroids[(byte >> 4) & 3]
                             + s_q[d + 3] * s_centroids[(byte >> 6) & 3];
            }
        }

        float sigma = (norm_k > 1e-10f) ? norm_k / sqrtf((float)head_dim) : 0.0f;
        float score = block_reduce_sum(partial_dot) * sigma * scale;

        // --- Online softmax update (all threads see the same score via shared mem) ---
        if (tid == 0) {
            float new_max = fmaxf(running_max, score);
            float rescale = expf(running_max - new_max);
            running_sum = running_sum * rescale + expf(score - new_max);
            running_max = new_max;
            s_max = running_max;
            s_sum_exp = running_sum;
        }
        __syncthreads();

        float w = expf(score - s_max); // unnormalized softmax weight
        float rescale = expf(running_max - s_max); // handle max update for old accumulators

        // Rescale previous accumulator (max changed) + add new contribution
        const float* v_row = kv_v + k * head_dim;
        for (int i = 0; i < 4; ++i) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                // When max increases, previous acc was too large by exp(old_max - new_max)
                // We track this: acc already has the rescale from the last iteration
                acc[i] = acc[i] * rescale + w * v_row[d];
            }
        }

        // Update running max for rescale tracking
        running_max = s_max;
    }

    // --- Final output: acc / sum_exp ---
    float inv_sum = (s_sum_exp > 0.0f) ? 1.0f / s_sum_exp : 0.0f;
    for (int i = 0; i < 4; ++i) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            output[head_idx * head_dim + d] = acc[i] * inv_sum;
        }
    }
}
