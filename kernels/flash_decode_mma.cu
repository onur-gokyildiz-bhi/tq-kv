// flash_decode_mma.cu
// ─────────────────────────────────────────────────────────────────────────────
// Tensor-core flash attention for decode — Phase 1 skeleton.
//
// Goal: port the structure of llama.cpp's fattn-mma-f16.cuh (grid = one block
// per KV head, shared Q tile for GQA group, streaming K/V chunks with online
// softmax) so Phase 2 can drop mma.sync.m16n8k16 PTX into place without
// rewriting control flow.
//
// Phase 1 (this file): fp32 compute body that matches
// gqa_decode_attention_f32's numerics. NO mma.sync yet. Ships only to prove:
//   (1) nvcc picks up the file via build.rs auto-discovery;
//   (2) the Rust launcher compiles under the new feature flag;
//   (3) grid = one-block-per-KV-group shape wires cleanly.
//
// Phase 2 (next sprint): replace the marked "PHASE-2" blocks with
//   ldmatrix + mma.sync + fp16 K/V tiles. Softmax state stays fp32.
//
// See docs/tensor-core-flash-attn-plan.md for the full scope.
// ─────────────────────────────────────────────────────────────────────────────

#include "common.cuh"

#ifndef FDM_MAX_HEAD_DIM
#define FDM_MAX_HEAD_DIM 256
#endif

#ifndef FDM_MAX_GQA_RATIO
// Max heads per KV group we statically support. Qwen2 7B = 7, Llama3 = 4,
// Mistral 7B = 4. 16 is a safe ceiling for future models.
#define FDM_MAX_GQA_RATIO 16
#endif

// ─────────────────────────────────────────────────────────────────────────────
// flash_decode_mma_f16_f32
//
// One block per KV head. Threads in the block cooperate across all query heads
// that map to this KV head (GQA), sharing the K/V tile stream.
//
// Arg shape matches gqa_decode_attention_f32 so the launcher can A/B test.
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
    const int kv_head   = blockIdx.x;               // one block per KV group
    if (kv_head >= n_kv_heads) return;

    const int gqa_ratio = n_heads / n_kv_heads;     // queries per KV head
    const int tid       = threadIdx.x;
    const int nthreads  = blockDim.x;

    // ── Shared memory layout ────────────────────────────────────────────────
    // PHASE-2: K_tile[nbatch_fa * head_dim] + V_tile[nbatch_fa * head_dim] in
    // fp16, plus 2-stage cp.async double buffer. For now we only stage Q.
    __shared__ float s_q[FDM_MAX_GQA_RATIO * FDM_MAX_HEAD_DIM];

    // Load Q for every query head in this GQA group (once, reused across KV).
    const int q_total = gqa_ratio * head_dim;
    for (int i = tid; i < q_total; i += nthreads) {
        const int qh = i / head_dim;                // 0 .. gqa_ratio-1
        const int d  = i % head_dim;
        const int head_idx = kv_head * gqa_ratio + qh;
        s_q[qh * head_dim + d] = ld_readonly(&Q[head_idx * head_dim + d]);
    }
    __syncthreads();

    // ── Per-(thread, query-head) state ──────────────────────────────────────
    // Each thread owns a slice of the head_dim output per query head.
    // n_acc = ceil(head_dim / nthreads). Bounded by FDM_MAX_HEAD_DIM/32 + 1.
    const int n_acc = (head_dim + nthreads - 1) / nthreads;

    // acc[qh][slot] accumulates V-weighted sum per query head.
    // Register blowup check: gqa_ratio * n_acc. For Qwen2 7B at block=32:
    //   7 * (128/32) = 28 floats/thread — fine.
    // PHASE-2: these become tile<16,8,float> VKQ_C[] fragments, fp32-accum.
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

    const float* kv_k = K + kv_head * max_seq * head_dim;
    const float* kv_v = V + kv_head * max_seq * head_dim;

    // ── Stream over KV rows ─────────────────────────────────────────────────
    // PHASE-2: chunk in nbatch_fa=64-row tiles, load via cp.async into fp16
    // shared-memory buffers, run QK mma for all gqa_ratio queries in parallel,
    // softmax across the tile, then AV mma. For now, per-row online softmax.
    for (int k = 0; k < seq_len; ++k) {
        const float* k_row = kv_k + k * head_dim;
        const float* v_row = kv_v + k * head_dim;

        // Per-query-head score: s_q[qh] · k_row, cooperative reduce.
        // PHASE-2: this entire inner loop becomes a single mma.sync
        //          m16n8k16 call reducing 16 K rows × 8 Q heads at once.
        #pragma unroll
        for (int qh = 0; qh < FDM_MAX_GQA_RATIO; ++qh) {
            if (qh >= gqa_ratio) break;

            float partial = 0.0f;
            for (int d = tid; d < head_dim; d += nthreads) {
                partial += s_q[qh * head_dim + d] * ld_readonly(&k_row[d]);
            }
            float score = block_reduce_sum(partial) * scale;

            // Online softmax — identical math to gqa_decode_attention_f32.
            // Kept in fp32 on purpose; fp16 would overflow for long ctx.
            float old_max = running_max[qh];
            float new_max = fmaxf(old_max, score);
            float rf      = __expf(old_max - new_max);
            float w       = __expf(score   - new_max);
            running_sum[qh] = running_sum[qh] * rf + w;
            running_max[qh] = new_max;

            // Rescale prior acc and add w * v_row contribution.
            // PHASE-2: VKQ_C[qh] tiles scale in-register (fp16 mul), then
            //          the P_fp16 × V_fp16 mma accumulates into fp32 tiles.
            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                int d = tid + i * nthreads;
                if (d < head_dim && i < n_acc) {
                    acc[qh][i] = acc[qh][i] * rf + w * ld_readonly(&v_row[d]);
                }
            }
        }
    }

    // ── Writeback ───────────────────────────────────────────────────────────
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
}
