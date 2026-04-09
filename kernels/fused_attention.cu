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

#include "common_v2.cuh"

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
    const uint8_t* __restrict__ packed_indices, // [n_kv_heads, max_seq * bytes_per_key]
    const float* __restrict__ norms,            // [n_kv_heads, max_seq]
    const float* __restrict__ centroids,        // [n_centroids]
    float* __restrict__ scores_out,             // [n_heads, n_keys]
    const int n_heads,
    const int n_kv_heads,
    const int n_keys,
    const int head_dim,
    const int bits,                             // 2, 3, or 4
    const float scale,                          // 1/sqrt(head_dim)
    const int max_seq                           // buffer stride (pre-allocated size per head)
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
    const uint8_t* kv_packed = packed_indices + kv_head * max_seq * bytes_per_key;
    const float* kv_norms = norms + kv_head * max_seq;

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

// ─── GPU Key Decompression: packed indices → full FP32 keys ──
// Decompresses TQ-compressed keys directly on GPU for standard Q@K matmul.
// Each block handles one key position for one KV head.
// Grid: (n_kv_heads, n_keys), Block: (head_dim)
//
// Output: keys_out[kv_head * n_keys * head_dim + key * head_dim + d]
// = sigma * inverse_hadamard(centroids[indices[d]]) for each dimension d
//
// Note: This outputs keys in the Hadamard-rotated domain (pre-RoPE).
// The caller must handle RoPE separately if needed.

// DEPRECATED: fused centroid+WHT version had subtle bug (max_diff=1.23 vs CPU).
// Use tq_centroid_lookup_f32 + hadamard_inverse_batch_f32 two-step instead.
extern "C" __global__ void tq_decompress_keys_f32(
    const uint8_t* __restrict__ packed_indices,  // [n_kv_heads, max_seq * bytes_per_key]
    const float* __restrict__ norms,             // [n_kv_heads, max_seq]
    const float* __restrict__ centroids,         // [n_centroids]
    const float* __restrict__ signs,             // [head_dim] — Hadamard sign flips
    float* __restrict__ keys_out,                // [n_kv_heads, n_keys, head_dim]
    const int n_kv_heads,
    const int n_keys,
    const int head_dim,
    const int bits,
    const int max_seq                            // buffer stride
) {
    const int kv_head = blockIdx.x;
    const int key_idx = blockIdx.y;
    if (kv_head >= n_kv_heads || key_idx >= n_keys) return;

    const int tid = threadIdx.x;
    const int bytes_per_key = (head_dim * bits + 7) / 8;

    // Load centroids to shared memory
    __shared__ float s_centroids[MAX_CENTROIDS];
    const int n_centroids = 1 << bits;
    if (tid < n_centroids) s_centroids[tid] = centroids[tid];
    __syncthreads();

    // Get norm/sigma for this key
    float norm = norms[kv_head * max_seq + key_idx];
    float sigma = (norm > 1e-10f) ? norm / sqrtf((float)head_dim) : 0.0f;

    // Decompress: lookup centroid for each dimension, scale by sigma
    const uint8_t* key_packed = packed_indices + kv_head * max_seq * bytes_per_key + key_idx * bytes_per_key;

    __shared__ float s_key[MAX_HEAD_DIM];

    if (bits == 4) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            uint8_t byte = key_packed[d / 2];
            float c = (d & 1) ? s_centroids[byte >> 4] : s_centroids[byte & 0xF];
            s_key[d] = c * sigma;
        }
    } else if (bits == 2) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            uint8_t byte = key_packed[d / 4];
            int shift = (d & 3) * 2;
            float c = s_centroids[(byte >> shift) & 3];
            s_key[d] = c * sigma;
        }
    }
    __syncthreads();

    // Inverse Hadamard: WHT then sign flip
    // WHT butterfly stages
    for (int h = 1; h < head_dim; h <<= 1) {
        for (int i = tid; i < head_dim / 2; i += blockDim.x) {
            int block_idx = i / h;
            int offset = i % h;
            int j = block_idx * (2 * h) + offset;
            float a = s_key[j];
            float b = s_key[j + h];
            s_key[j] = a + b;
            s_key[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply sign flips (inverse Hadamard = WHT * 1/sqrt(d) * D)
    float inv_sqrt_dim = rsqrtf((float)head_dim);
    float* out_ptr = keys_out + kv_head * n_keys * head_dim + key_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] = s_key[d] * inv_sqrt_dim * signs[d];
    }
}

// ─── Centroid Lookup Only (Step 1 of two-step decompress) ───
// Unpacks indices, looks up centroids, scales by sigma.
// Output is in the Hadamard-rotated domain (before inverse WHT).
// Caller must run hadamard_inverse_batch_f32 on the output.
//
// Grid: (n_kv_heads, n_keys), Block: (min(head_dim, 256))
//
// key_offset: start reading from this position in packed/norms (for incremental)
// out_stride: output per-head stride in vectors (max_seq for cache, n_keys for contiguous)
//
// Output: out[(kv_head * out_stride + key_idx) * head_dim]
// For contiguous output: set out_stride = n_keys
// For strided cache:     set out_stride = max_seq, write at out_offset positions

extern "C" __global__ void tq_centroid_lookup_f32(
    const uint8_t* __restrict__ packed_indices,  // [n_kv_heads, max_seq * bytes_per_key]
    const float* __restrict__ norms,             // [n_kv_heads, max_seq]
    const float* __restrict__ centroids,         // [n_centroids]
    float* __restrict__ out,                     // output buffer
    const int n_kv_heads,
    const int n_keys,
    const int head_dim,
    const int bits,
    const int max_seq,                           // buffer stride for packed/norms
    const int key_offset,                        // read keys starting at this position
    const int out_stride                         // per-head stride in output (vectors, not bytes)
) {
    const int kv_head = blockIdx.x;
    const int key_idx = blockIdx.y;
    if (kv_head >= n_kv_heads || key_idx >= n_keys) return;

    const int tid = threadIdx.x;
    const int bytes_per_key = (head_dim * bits + 7) / 8;

    // Load centroids to shared memory
    __shared__ float s_centroids[MAX_CENTROIDS];
    const int n_centroids = 1 << bits;
    if (tid < n_centroids) s_centroids[tid] = centroids[tid];
    __syncthreads();

    // Read from key_offset + key_idx in the packed buffer
    const int src_pos = key_offset + key_idx;
    float norm = norms[kv_head * max_seq + src_pos];
    float sigma = (norm > 1e-10f) ? norm / sqrtf((float)head_dim) : 0.0f;

    const uint8_t* key_packed = packed_indices + kv_head * max_seq * bytes_per_key + src_pos * bytes_per_key;
    // Output row with configurable stride
    float* out_row = out + (kv_head * out_stride + key_idx) * head_dim;

    if (bits == 4) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            uint8_t byte = key_packed[d / 2];
            float c = (d & 1) ? s_centroids[byte >> 4] : s_centroids[byte & 0xF];
            out_row[d] = c * sigma;
        }
    } else if (bits == 2) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            uint8_t byte = key_packed[d / 4];
            int shift = (d & 3) * 2;
            float c = s_centroids[(byte >> shift) & 3];
            out_row[d] = c * sigma;
        }
    } else if (bits == 3) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            int bit_offset = d * 3;
            int byte_idx = bit_offset / 8;
            int bit_pos  = bit_offset % 8;
            uint8_t idx;
            if (bit_pos <= 5) {
                idx = (key_packed[byte_idx] >> bit_pos) & 7;
            } else {
                idx = (key_packed[byte_idx] >> bit_pos) |
                      ((key_packed[byte_idx + 1] << (8 - bit_pos)) & 7);
                idx &= 7;
            }
            out_row[d] = s_centroids[idx] * sigma;
        }
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

// v2: rescale broadcast fix
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
    const float scale,                         // 1/sqrt(head_dim)
    const int max_seq,                         // buffer stride (packed/norms/V pre-allocated)
    // Sink token support: uncompressed FP32 keys processed before compressed keys.
    // Set n_sink=0 and sink_K/sink_V=NULL to disable.
    const float* __restrict__ sink_K,          // [n_kv_heads, n_sink, head_dim] or NULL
    const float* __restrict__ sink_V,          // [n_kv_heads, n_sink, head_dim] or NULL
    const float* __restrict__ raw_query,       // [n_heads, head_dim] non-rotated (for sink dot product)
    const int n_sink
) {
    const int head_idx = blockIdx.x;
    if (head_idx >= n_heads) return;
    const int tid = threadIdx.x;
    const int kv_head = head_idx / (n_heads / n_kv_heads);  // GQA mapping

    // --- Shared memory ---
    __shared__ float s_centroids[MAX_CENTROIDS];
    __shared__ float s_q[MAX_HEAD_DIM];     // rotated query (for compressed keys)
    __shared__ float s_q_raw[MAX_HEAD_DIM]; // raw query (for sink keys)

    const int n_centroids = 1 << bits;
    if (tid < n_centroids) s_centroids[tid] = centroids[tid];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = rotated_query[head_idx * head_dim + d];
        if (raw_query && n_sink > 0) {
            s_q_raw[d] = raw_query[head_idx * head_dim + d];
        }
    }
    __syncthreads();

    // Per-thread V accumulator (each thread handles a subset of head_dim)
    const int n_acc = (head_dim + blockDim.x - 1) / blockDim.x;
    float acc[9]; // MAX_HEAD_DIM(256)/32+1; unrolled for dims tid, tid+blockDim, ...
    for (int i = 0; i < n_acc; ++i) acc[i] = 0.0f;
    float running_max = -1e10f;
    float running_sum = 0.0f;

    // --- Phase 1: Sink tokens (uncompressed FP32 Q@K dot product) ---
    if (sink_K && sink_V && n_sink > 0) {
        const float* sink_k_head = sink_K + kv_head * n_sink * head_dim;
        const float* sink_v_head = sink_V + kv_head * n_sink * head_dim;

        for (int k = 0; k < n_sink; ++k) {
            float partial_dot = 0.0f;
            const float* k_row = sink_k_head + k * head_dim;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                partial_dot += s_q_raw[d] * ld_readonly(&k_row[d]);
            }
            // v2: block_reduce_sum broadcasts to ALL threads
            float score = block_reduce_sum(partial_dot) * scale;

            // Online softmax — all threads have score
            float old_max = running_max;
            float new_max = fmaxf(old_max, score);
            float r = expf(old_max - new_max);
            running_sum = running_sum * r + expf(score - new_max);
            running_max = new_max;

            float w = expf(score - new_max);
            const float* v_row = sink_v_head + k * head_dim;
            for (int i = 0; i < n_acc; ++i) {
                int d = tid + i * blockDim.x;
                if (d < head_dim) {
                    acc[i] = acc[i] * r + w * ld_readonly(&v_row[d]);
                }
            }
        }
    }

    // --- Phase 2: Compressed keys (codebook lookup) ---
    // Use max_seq as stride (buffer is pre-allocated with max_seq slots per head)
    const int bytes_per_key = (head_dim * bits + 7) / 8;
    const uint8_t* kv_packed = packed_indices + kv_head * max_seq * bytes_per_key;
    const float* kv_norms = norms + kv_head * max_seq;
    const float* kv_v = V + kv_head * max_seq * head_dim;

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
        // v2: block_reduce_sum broadcasts to ALL threads
        float score = block_reduce_sum(partial_dot) * sigma * scale;

        // Online softmax — all threads have score
        float old_max = running_max;
        float new_max = fmaxf(old_max, score);
        float r = expf(old_max - new_max);
        running_sum = running_sum * r + expf(score - new_max);
        running_max = new_max;

        float w = expf(score - new_max);

        const float* v_row = kv_v + k * head_dim;
        for (int i = 0; i < n_acc; ++i) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                acc[i] = acc[i] * r + w * ld_readonly(&v_row[d]);
            }
        }
    }

    // --- Final output: acc / sum_exp ---
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    for (int i = 0; i < n_acc; ++i) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            output[head_idx * head_dim + d] = acc[i] * inv_sum;
        }
    }
}

// ─── GQA Decode Attention (uncompressed FP32 KV cache) ──────
//
// Fused Q@K^T + scale + softmax + @V for standard decode (seq_len=1).
// Handles GQA head mapping internally — NO repeat_kv copy needed.
//
// K/V layout: [n_kv_heads, max_seq, head_dim] (pre-allocated padded buffer).
// Only reads positions [0, seq_len) per head (ignores padding).
//
// One block per Q head. Threads cooperate on dot product + V accumulation.
// Online softmax (numerically stable, single pass).
// Graph-safe variant: reads seq_len from GPU scalar pointer (not baked kernel arg).
// Use for CUDA Graph replay — update the scalar via memcpy_htod before launch.
extern "C" __global__ void gqa_decode_attention_graph_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const int* __restrict__ seq_len_ptr,  // GPU scalar: pre-append valid length
    const int n_heads,
    const int n_kv_heads,
    const int max_seq,
    const int head_dim,
    const float scale,
    const int extra,       // added after kv_append (typically 1 for decode)
    const int window_size  // sliding window: 0 = global (attend all), >0 = attend last N tokens
) {
    const int seq_len = *seq_len_ptr + extra;  // post-append length
    const int head_idx = blockIdx.x;
    if (head_idx >= n_heads) return;
    const int tid = threadIdx.x;
    const int kv_head = head_idx / (n_heads / n_kv_heads);

    __shared__ float s_q[MAX_HEAD_DIM];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = ld_readonly(&Q[head_idx * head_dim + d]);
    }
    __syncthreads();

    const int n_acc = (head_dim + blockDim.x - 1) / blockDim.x;
    float acc[9];  // MAX_HEAD_DIM(256) / min_blockDim(32) + 1
    for (int i = 0; i < n_acc; ++i) acc[i] = 0.0f;
    float running_max = -1e10f;
    float running_sum = 0.0f;

    const float* kv_k = K + kv_head * max_seq * head_dim;
    const float* kv_v = V + kv_head * max_seq * head_dim;

    // Sliding window: only attend to last window_size tokens (0 = global)
    const int k_start = (window_size > 0 && seq_len > window_size) ? (seq_len - window_size) : 0;

    for (int k = k_start; k < seq_len; ++k) {
        const float* k_row = kv_k + k * head_dim;
        float partial_dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            partial_dot += s_q[d] * ld_readonly(&k_row[d]);
        }
        // v2: block_reduce_sum broadcasts to ALL threads
        float score = block_reduce_sum(partial_dot) * scale;

        // Online softmax — all threads have score
        float old_max = running_max;
        float new_max = fmaxf(old_max, score);
        float rf = expf(old_max - new_max);
        running_sum = running_sum * rf + expf(score - new_max);
        running_max = new_max;

        float w = expf(score - new_max);
        const float* v_row = kv_v + k * head_dim;
        for (int i = 0; i < n_acc; ++i) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                acc[i] = acc[i] * rf + w * ld_readonly(&v_row[d]);
            }
        }
    }

    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    for (int i = 0; i < n_acc; ++i) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            output[head_idx * head_dim + d] = acc[i] * inv_sum;
        }
    }
}

// Standard variant: seq_len as kernel argument (non-graph path).
extern "C" __global__ void gqa_decode_attention_f32(
    const float* __restrict__ Q,     // [n_heads, 1, head_dim] (current query)
    const float* __restrict__ K,     // [n_kv_heads, max_seq, head_dim] (padded KV cache)
    const float* __restrict__ V,     // [n_kv_heads, max_seq, head_dim]
    float* __restrict__ output,      // [n_heads, head_dim]
    const int n_heads,
    const int n_kv_heads,
    const int seq_len,               // actual valid length (not max_seq)
    const int max_seq,               // padded buffer size
    const int head_dim,
    const float scale                // 1/sqrt(head_dim)
) {
    const int head_idx = blockIdx.x;
    if (head_idx >= n_heads) return;
    const int tid = threadIdx.x;
    const int kv_head = head_idx / (n_heads / n_kv_heads);  // GQA mapping

    // Load query into shared memory
    __shared__ float s_q[MAX_HEAD_DIM];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = ld_readonly(&Q[head_idx * head_dim + d]);
    }
    __syncthreads();

    // Per-thread V accumulator
    const int n_acc = (head_dim + blockDim.x - 1) / blockDim.x;
    float acc[9];  // MAX_HEAD_DIM(256) / min_blockDim(32) + 1
    for (int i = 0; i < n_acc; ++i) acc[i] = 0.0f;
    float running_max = -1e10f;
    float running_sum = 0.0f;

    const float* kv_k = K + kv_head * max_seq * head_dim;
    const float* kv_v = V + kv_head * max_seq * head_dim;

    // Stream over valid KV positions
    for (int k = 0; k < seq_len; ++k) {
        // Cooperative dot product: Q[head] · K[kv_head, k] with read-only cache
        const float* k_row = kv_k + k * head_dim;
        float partial_dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            partial_dot += s_q[d] * ld_readonly(&k_row[d]);
        }
        // v2: block_reduce_sum broadcasts to ALL threads — no hack needed
        float score = block_reduce_sum(partial_dot) * scale;

        // Online softmax — all threads have score, each tracks own state
        float old_max = running_max;
        float new_max = fmaxf(old_max, score);
        float rf = expf(old_max - new_max);
        running_sum = running_sum * rf + expf(score - new_max);
        running_max = new_max;

        float w = expf(score - new_max);
        const float* v_row = kv_v + k * head_dim;
        for (int i = 0; i < n_acc; ++i) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                acc[i] = acc[i] * rf + w * ld_readonly(&v_row[d]);
            }
        }
    }

    // Final output: acc / sum_exp
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    for (int i = 0; i < n_acc; ++i) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            output[head_idx * head_dim + d] = acc[i] * inv_sum;
        }
    }
}
