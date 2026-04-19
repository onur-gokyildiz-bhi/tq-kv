// flash_decode_mma.cu
// ─────────────────────────────────────────────────────────────────────────────
// Tensor-core flash attention for decode.
//
// Phase 1 (shipped 571536b): fp32 body mirroring gqa_decode_attention_f32.
// Phase 2.a (this commit):   QK leg on mma.sync.m16n8k16.f32.f16.f16.f32.
//                            AV leg remains fp32 (Phase 2.b).
//
// Design choice (PTX inline asm vs nvcuda::wmma):
//   Chose inline PTX — matches llama.cpp fattn-mma-f16.cuh style and gives us
//   direct control over the A/B/C register layout. The wmma C++ API is cleaner
//   but hides the fragment layout, which makes softmax score extraction
//   awkward (must round-trip through smem anyway). With explicit PTX we know
//   exactly which (K-row, Q-col) each lane holds, so we can place scores in
//   a well-defined smem slot without another conversion.
//
// Reference:
//   - PTX ISA 8.0 §9.7.14.5.16 (mma m16n8k16 f32.f16.f16.f32)
//   - llama.cpp ggml/src/ggml-cuda/fattn-mma-f16.cuh lines 584-618
//     (the mma(KQ_C[...], K_A, Q_B[...]) loop — same pattern as ours below)
// ─────────────────────────────────────────────────────────────────────────────

#include "common.cuh"
#include <cuda_fp16.h>

#ifndef FDM_MAX_HEAD_DIM
#define FDM_MAX_HEAD_DIM 256
#endif

#ifndef FDM_MAX_GQA_RATIO
// Max heads per KV group we statically support. Qwen2 7B = 7, Llama3 = 4,
// Mistral 7B = 4. 16 is a safe ceiling for future models.
#define FDM_MAX_GQA_RATIO 16
#endif

// MMA tile shape — fixed by the PTX instruction.
//
// Note (2026-04-19): the inline asm below emits m16n8k8 (one .b32 per B
// thread, two .b32 per A thread). CUDA 12 accepted a mislabelled
// `m16n8k16` name against those argument counts; CUDA 13's ptxas is
// strict and rejects it (`Argument vector size mismatch for instruction
// 'mma'`). The loop that calls the helper steps `kk += FDM_MMA_K`, so
// this constant stays in lockstep with the actual PTX shape to keep the
// kernel semantically correct.
#define FDM_MMA_M 16
#define FDM_MMA_N 8
#define FDM_MMA_K 8

// ─────────────────────────────────────────────────────────────────────────────
// mma_m16n8k16_f16: inline PTX wrapper for
//   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
//
// A: 16×16 half, row-major. Per-thread frag = 4 halves in 2 × .b32.
// B: 16× 8 half, col-major. Per-thread frag = 2 halves in 1 × .b32.
// C: 16× 8 f32.              Per-thread frag = 4 f32.
//
// Lane mapping (group = laneid/4, gin = laneid%4):
//   A frag: A[group][gin*2], A[group][gin*2+1], A[group+8][gin*2], A[group+8][gin*2+1]
//   B frag: B[gin*2][group], B[gin*2+1][group]
//   C frag: C[group][gin*2], C[group][gin*2+1], C[group+8][gin*2], C[group+8][gin*2+1]
// ─────────────────────────────────────────────────────────────────────────────
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_m16n8k16_f16(
    float          c[4],
    const uint32_t a[2],
    const uint32_t b
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b)
    );
}

// Pack two halves into a .b32 register payload for mma.sync.
__device__ __forceinline__ uint32_t pack_half2(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
#endif // __CUDA_ARCH__ >= 800

// ─────────────────────────────────────────────────────────────────────────────
// flash_decode_mma_f16_f32
//
// One block per KV head. Threads cooperate across all query heads (GQA),
// sharing K/V tile stream. Phase 2.a: QK on tensor core; AV on fp32.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void flash_decode_mma_f16_f32(
    const float* __restrict__ Q,     // [n_heads,    head_dim]   — 1 query row per head
    const float* __restrict__ K,     // [n_kv_heads, max_seq, head_dim]
    const float* __restrict__ V,     // [n_kv_heads, max_seq, head_dim]
    float*       __restrict__ output,// [n_heads,    head_dim]
    const int n_heads,
    const int n_kv_heads,
    const int seq_len,               // valid KV length
    const int max_seq,               // padded buffer stride
    const int head_dim,
    const float scale                // 1/sqrt(head_dim)
) {
    const int kv_head   = blockIdx.x;
    if (kv_head >= n_kv_heads) return;

    const int gqa_ratio = n_heads / n_kv_heads;
    const int tid       = threadIdx.x;
    const int nthreads  = blockDim.x;
    const int lane      = tid & 31;
    const int warp_id   = tid >> 5;

    const float* kv_k = K + kv_head * max_seq * head_dim;
    const float* kv_v = V + kv_head * max_seq * head_dim;

#if __CUDA_ARCH__ >= 800
    // ═══════════════════════════════════════════════════════════════════════
    // Phase 2.a — tensor-core QK path (sm_80+)
    //
    // Shmem layout (dynamic smem; launcher must pass the right size):
    //   q_half    [N=8][head_dim]            fp16 Q tile (padded to 8 rows)
    //   k_half    [M=16][head_dim]           fp16 K tile (single buffer)
    //   acc_sm    [gqa_ratio][head_dim]      fp32 AV accumulator
    //   scores_sm [M=16][N=8]                fp32 per-tile scores
    //
    // For head_dim=128:
    //   q_half    =  8 * 128 * 2 =   2 KB
    //   k_half    = 16 * 128 * 2 =   4 KB
    //   acc_sm    = 16 * 128 * 4 =   8 KB   (FDM_MAX_GQA_RATIO=16 cap)
    //   scores_sm = 16 *   8 * 4 = 512 B
    //   total                     ≈ 14.5 KB   (fits 48 KB default; 100 KB avail sm_80+)
    // ═══════════════════════════════════════════════════════════════════════
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* q_half    = reinterpret_cast<__half*>(smem_raw);
    __half* k_half    = q_half + FDM_MMA_N * head_dim;
    float*  acc_sm    = reinterpret_cast<float*>(k_half + FDM_MMA_M * head_dim);
    float*  scores_sm = acc_sm + FDM_MAX_GQA_RATIO * head_dim;
    // scores_sm has room for FDM_MMA_M * FDM_MMA_N = 128 floats (512 B).

    // ── Stage Q into shmem as fp16, zero-pad unused Q rows ──────────────────
    const int q_tile_elems = FDM_MMA_N * head_dim;
    for (int i = tid; i < q_tile_elems; i += nthreads) {
        const int qh = i / head_dim;
        const int d  = i % head_dim;
        float v = 0.0f;
        if (qh < gqa_ratio) {
            const int head_idx = kv_head * gqa_ratio + qh;
            v = ld_readonly(&Q[head_idx * head_dim + d]);
        }
        q_half[qh * head_dim + d] = __float2half(v);
    }

    // ── Zero AV accumulator ─────────────────────────────────────────────────
    const int acc_elems = FDM_MAX_GQA_RATIO * head_dim;
    for (int i = tid; i < acc_elems; i += nthreads) acc_sm[i] = 0.0f;

    // Online softmax state, replicated per-thread (each thread produces
    // the same updates deterministically from the shared scores_sm).
    float running_max[FDM_MAX_GQA_RATIO];
    float running_sum[FDM_MAX_GQA_RATIO];
    #pragma unroll
    for (int qh = 0; qh < FDM_MAX_GQA_RATIO; ++qh) {
        running_max[qh] = -1e10f;
        running_sum[qh] = 0.0f;
    }

    __syncthreads();

    // Fragment lane mapping
    const int group = lane >> 2;
    const int gin   = lane & 3;

    const int c_krow[4] = { group,     group,     group + 8, group + 8 };
    const int c_qrow[4] = { gin*2,     gin*2 + 1, gin*2,     gin*2 + 1 };

    // ── Stream K in 16-row tiles ────────────────────────────────────────────
    const int n_tiles = (seq_len + FDM_MMA_M - 1) / FDM_MMA_M;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int k_base       = tile * FDM_MMA_M;
        const int k_rows_valid = min(FDM_MMA_M, seq_len - k_base);

        // Stage K tile as fp16, zero rows beyond valid length
        const int k_tile_elems = FDM_MMA_M * head_dim;
        for (int i = tid; i < k_tile_elems; i += nthreads) {
            const int kr = i / head_dim;
            const int d  = i % head_dim;
            float v = 0.0f;
            if (kr < k_rows_valid) {
                v = ld_readonly(&kv_k[(k_base + kr) * head_dim + d]);
            }
            k_half[kr * head_dim + d] = __float2half(v);
        }
        __syncthreads();

        // ── Warp-0 drives mma.sync for QK ───────────────────────────────────
        // llama.cpp pattern (fattn-mma-f16.cuh:584-618): loop k_KQ_0 over the
        // head_dim, load_ldmatrix K fragment, call mma(KQ_C, K_A, Q_B). We
        // do the same here with explicit lane→element loads instead of
        // ldmatrix, trading instruction count for layout transparency.
        if (warp_id == 0) {
            float c_frag[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

            for (int kk = 0; kk < head_dim; kk += FDM_MMA_K) {
                // A fragment (K tile, row-major)
                __half a0 = k_half[ group      * head_dim + kk + gin * 2    ];
                __half a1 = k_half[ group      * head_dim + kk + gin * 2 + 1];
                __half a2 = k_half[(group + 8) * head_dim + kk + gin * 2    ];
                __half a3 = k_half[(group + 8) * head_dim + kk + gin * 2 + 1];
                uint32_t a_reg[2];
                a_reg[0] = pack_half2(a0, a1);
                a_reg[1] = pack_half2(a2, a3);

                // B fragment (Q tile; col=qh, row=k_elem → col-major)
                __half b0 = q_half[ group * head_dim + kk + gin * 2    ];
                __half b1 = q_half[ group * head_dim + kk + gin * 2 + 1];
                uint32_t b_reg = pack_half2(b0, b1);

                mma_m16n8k16_f16(c_frag, a_reg, b_reg);
            }

            // Scale, mask, and scatter the 4 owned scores into smem.
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int kr = c_krow[i];
                int qh = c_qrow[i];
                float s = c_frag[i] * scale;
                bool valid = (kr < k_rows_valid) && (qh < gqa_ratio);
                scores_sm[kr * FDM_MMA_N + qh] = valid ? s : -1e10f;
            }
        }
        __syncthreads();

        // ── All warps: online softmax + fp32 AV accumulate per Q head ───────
        // Each Q head is independent; iterate sequentially for simplicity.
        // Phase 2.b will fuse this into the warp that owns the Q column.
        for (int qh = 0; qh < gqa_ratio; ++qh) {
            // Tile max over K rows
            float tmax = -1e10f;
            #pragma unroll
            for (int kr = 0; kr < FDM_MMA_M; ++kr) {
                tmax = fmaxf(tmax, scores_sm[kr * FDM_MMA_N + qh]);
            }

            float old_max = running_max[qh];
            float new_max = fmaxf(old_max, tmax);
            float rf      = __expf(old_max - new_max);
            running_max[qh] = new_max;

            // Rescale existing AV accumulator (applied once per tile).
            if (rf != 1.0f) {
                for (int d = tid; d < head_dim; d += nthreads) {
                    acc_sm[qh * head_dim + d] *= rf;
                }
                __syncthreads();
            }

            // Weighted sum of V rows with softmax weights
            float sum_add = 0.0f;
            #pragma unroll
            for (int kr = 0; kr < FDM_MMA_M; ++kr) {
                float s = scores_sm[kr * FDM_MMA_N + qh];
                if (s <= -1e9f) continue;
                float w = __expf(s - new_max);
                sum_add += w;

                int kv_idx = k_base + kr;
                const float* v_row = kv_v + kv_idx * head_dim;
                for (int d = tid; d < head_dim; d += nthreads) {
                    acc_sm[qh * head_dim + d] += w * ld_readonly(&v_row[d]);
                }
            }
            running_sum[qh] = running_sum[qh] * rf + sum_add;
            __syncthreads();
        }
    }

    // ── Writeback ───────────────────────────────────────────────────────────
    for (int qh = 0; qh < gqa_ratio; ++qh) {
        const int head_idx  = kv_head * gqa_ratio + qh;
        const float inv_sum = (running_sum[qh] > 0.0f) ? 1.0f / running_sum[qh] : 0.0f;
        for (int d = tid; d < head_dim; d += nthreads) {
            output[head_idx * head_dim + d] = acc_sm[qh * head_dim + d] * inv_sum;
        }
    }

#else
    // ═══════════════════════════════════════════════════════════════════════
    // sm_75 fallback — identical to Phase 1 skeleton (fp32 everywhere)
    // ═══════════════════════════════════════════════════════════════════════
    __shared__ float s_q[FDM_MAX_GQA_RATIO * FDM_MAX_HEAD_DIM];

    const int q_total = gqa_ratio * head_dim;
    for (int i = tid; i < q_total; i += nthreads) {
        const int qh = i / head_dim;
        const int d  = i % head_dim;
        const int head_idx = kv_head * gqa_ratio + qh;
        s_q[qh * head_dim + d] = ld_readonly(&Q[head_idx * head_dim + d]);
    }
    __syncthreads();

    const int n_acc = (head_dim + nthreads - 1) / nthreads;
    float acc[FDM_MAX_GQA_RATIO][9];
    float running_max[FDM_MAX_GQA_RATIO];
    float running_sum[FDM_MAX_GQA_RATIO];

    #pragma unroll
    for (int qh = 0; qh < FDM_MAX_GQA_RATIO; ++qh) {
        if (qh >= gqa_ratio) break;
        running_max[qh] = -1e10f;
        running_sum[qh] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 9; ++i) acc[qh][i] = 0.0f;
    }

    for (int k = 0; k < seq_len; ++k) {
        const float* k_row = kv_k + k * head_dim;
        const float* v_row = kv_v + k * head_dim;

        #pragma unroll
        for (int qh = 0; qh < FDM_MAX_GQA_RATIO; ++qh) {
            if (qh >= gqa_ratio) break;
            float partial = 0.0f;
            for (int d = tid; d < head_dim; d += nthreads) {
                partial += s_q[qh * head_dim + d] * ld_readonly(&k_row[d]);
            }
            float score = block_reduce_sum(partial) * scale;

            float old_max = running_max[qh];
            float new_max = fmaxf(old_max, score);
            float rf      = __expf(old_max - new_max);
            float w       = __expf(score   - new_max);
            running_sum[qh] = running_sum[qh] * rf + w;
            running_max[qh] = new_max;

            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                int d = tid + i * nthreads;
                if (d < head_dim && i < n_acc) {
                    acc[qh][i] = acc[qh][i] * rf + w * ld_readonly(&v_row[d]);
                }
            }
        }
    }

    #pragma unroll
    for (int qh = 0; qh < FDM_MAX_GQA_RATIO; ++qh) {
        if (qh >= gqa_ratio) break;
        const int head_idx = kv_head * gqa_ratio + qh;
        const float inv_sum = (running_sum[qh] > 0.0f) ? 1.0f / running_sum[qh] : 0.0f;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            int d = tid + i * nthreads;
            if (d < head_dim && i < n_acc) {
                output[head_idx * head_dim + d] = acc[qh][i] * inv_sum;
            }
        }
    }
#endif // __CUDA_ARCH__ >= 800
}
