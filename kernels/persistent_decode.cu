// kernels/persistent_decode.cu
//
// Persistent Kernel Lite — Phase 2 bodies (Plan #6).
// See docs/persistent-kernel-lite-design.md.
//
// One block per SM, stays alive across the entire decoder layer. Walks a
// 9-phase micro-ISA (RMSNORM_QKV, ROPE, KV_APPEND, ATTN, WO, RES_ADD,
// RMSNORM_GATEUP, DOWN, DONE) synchronized via an atomic counter +
// __threadfence + __syncthreads. Hazy Research Llama-1B megakernel pattern.
//
// PHASE 2 STATUS: phases 0 (RMSNORM_QKV) and 1 (ROPE) have real device-
// function bodies mirroring fused_norm_q4km_qkv_bias_dp4a_f32 and the
// halved/interleaved fused rope_qk kernels. Phases 2-7 remain empty stubs
// pending Phase 3 (shmem handoff) and Phase 4 (weight prefetch) work.
//
// Row distribution across 68 persistent blocks:
//   Phase 0 (QKV): total_rows = q_out + k_out + v_out; grouped in 8-row
//   super-warps. Block b handles row-groups { b, b+gridDim.x, ... }.
//   Phase 1 (RoPE): heads stride the grid (head = b, b+gridDim.x, ...).
//
// Shared memory layout (dynamic, declared once at kernel entry):
//   [0 .. hidden_dim)               — s_normed (float)
//   [hidden_dim*4 .. +n_q8*36)      — s_x_q8_1 (uint8_t)
// Phase 0 alone uses this region; phases 1-7 ignore it.

#include "common.cuh"

// ─── Q4_K_M helpers (self-contained; mirror fused_layer.cu) ────────────
#ifndef QK_K
#define QK_K 256
#endif
#ifndef Q4K_BLOCK_SIZE
#define Q4K_BLOCK_SIZE 144
#endif

static __device__ __forceinline__ void pd_get_scale_min_k4(
    int j, const uint8_t* scales, uint8_t* sc, uint8_t* m
) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >> 4)  | ((scales[j] >> 6) << 4);
    }
}

// ─── Phase IDs ─────────────────────────────────────────────────────────
#define TQ_PHASE_RMSNORM_QKV     0
#define TQ_PHASE_ROPE            1
#define TQ_PHASE_KV_APPEND       2
#define TQ_PHASE_ATTN            3
#define TQ_PHASE_WO              4
#define TQ_PHASE_RES_ADD         5
#define TQ_PHASE_RMSNORM_GATEUP  6
#define TQ_PHASE_DOWN            7
#define TQ_PHASE_DONE            8
#define TQ_N_PHASES              9

// ─── Counter-based phase sync ──────────────────────────────────────────
static __device__ __forceinline__ void phase_wait(
    int* g_counter,
    int target_value
) {
    while (atomicAdd(g_counter, 0) < target_value) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(32);
#endif
    }
}

static __device__ __forceinline__ void phase_enter_and_wait(
    int* g_counter,
    int phase_idx,
    int n_blocks
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(g_counter, 1);
        __threadfence();
        phase_wait(g_counter, (phase_idx + 1) * n_blocks);
    }
    __syncthreads();
}

// ─── Phase 0 body: RMSNorm + fused Q4K_M QKV projection (dp4a) ─────────
//
// Mirrors fused_norm_q4km_qkv_bias_dp4a_f32 in kernels/fused_layer.cu.
// The standalone kernel uses 1 block per 8 output rows; here one persistent
// block iterates its assigned row-groups via block_id stride.
//
// For each row-group the block:
//   1. RmsNorm(input) → s_normed
//   2. q8_1 quantize s_normed into s_x_q8_1 (per-warp)
//   3. 8 warps × 1 row each → dp4a matvec with bias, dispatched across Q/K/V.
//
// RmsNorm output is bit-stable across row-groups (same input every time),
// so recompute is wasted work but keeps the body block-local and avoids a
// cross-block reduction. Trade-off sits at ~2 % of the matvec cost.
static __device__ __forceinline__ void phase_rmsnorm_qkv_body(
    const float* __restrict__ input,
    const float* __restrict__ norm_weight,
    const uint8_t* __restrict__ W_q,
    const uint8_t* __restrict__ W_k,
    const uint8_t* __restrict__ W_v,
    const float* __restrict__ bias_q,    // NULL if no bias
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    float* __restrict__ s_normed,        // shmem region: hidden_dim floats
    uint8_t* __restrict__ s_x_q8_1,      // shmem region: (hidden_dim/32)*36 bytes
    const int hidden_dim,
    const int q_out,
    const int k_out,
    const int v_out,
    const float eps,
    const int block_id,
    const int n_blocks
) {
    const int tid        = threadIdx.x;
    const int warp_id    = tid >> 5;
    const int lane       = tid & 31;
    const int total_rows = q_out + k_out + v_out;
    const int n_row_grp  = (total_rows + 7) >> 3;   // 8 rows per group
    const int n_q8_blocks = hidden_dim / 32;

    for (int rg = block_id; rg < n_row_grp; rg += n_blocks) {
        const int row0 = rg * 8;

        // 1) RmsNorm(input) → s_normed
        float sum_sq = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float val = input[i];
            sum_sq += val * val;
            s_normed[i] = val;
        }
        sum_sq = block_reduce_sum(sum_sq);
        __shared__ float s_rms_inv;
        if (tid == 0) {
            s_rms_inv = rsqrtf(sum_sq / (float)hidden_dim + eps);
        }
        __syncthreads();
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            s_normed[i] = s_normed[i] * s_rms_inv * norm_weight[i];
        }
        __syncthreads();

        // 2) q8_1 quantize s_normed → s_x_q8_1 (per-warp 32-element blocks)
        for (int blk = warp_id; blk < n_q8_blocks; blk += 8) {
            const int idx = blk * 32 + lane;
            const float xi = (idx < hidden_dim) ? s_normed[idx] : 0.0f;

            float amax = warp_reduce_max(fabsf(xi));

            const float d     = amax / 127.0f;
            const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

            int qi = __float2int_rn(xi * d_inv);
            qi = max(-128, min(127, qi));
            const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)qi;

            float sum_q = warp_reduce_sum((float)q);

            uint8_t* block_ptr = s_x_q8_1 + (size_t)blk * 36;
            int8_t*  qs_ptr    = reinterpret_cast<int8_t*>(block_ptr + 4);
            qs_ptr[lane] = q;

            if (lane == 0) {
                const float s = d * sum_q;
                half2 ds = __floats2half2_rn(d, s);
                *reinterpret_cast<half2*>(block_ptr) = ds;
            }
        }
        __syncthreads();

        // 3) dp4a matvec — 8 warps, 1 row each, dispatched by row range
        const int sub    = lane >> 2;
        const int slot   = lane & 3;
        const int grp_w  = sub >> 1;
        const int is_hi  = sub & 1;

        const int row         = row0 + warp_id;
        const bool row_active = (row < total_rows);

        const uint8_t* W_sel;
        const float*   bias_sel;
        float*         out_sel;
        int            local_row;
        if (row < q_out) {
            W_sel     = W_q;
            bias_sel  = bias_q;
            out_sel   = out_q;
            local_row = row;
        } else if (row < q_out + k_out) {
            W_sel     = W_k;
            bias_sel  = bias_k;
            out_sel   = out_k;
            local_row = row - q_out;
        } else {
            W_sel     = W_v;
            bias_sel  = bias_v;
            out_sel   = out_v;
            local_row = row - q_out - k_out;
        }

        const int n_superblocks = hidden_dim / QK_K;
        const int bytes_per_row = n_superblocks * Q4K_BLOCK_SIZE;

        float partial = 0.0f;

        for (int sb = 0; sb < n_superblocks; ++sb) {
            const uint8_t* blk = row_active
                ? (W_sel + local_row * bytes_per_row + sb * Q4K_BLOCK_SIZE)
                : (W_sel);

            uint16_t d_bits  = blk[0] | (blk[1] << 8);
            uint16_t dm_bits = blk[2] | (blk[3] << 8);
            float d_w    = __half2float(*reinterpret_cast<const __half*>(&d_bits));
            float dmin_w = __half2float(*reinterpret_cast<const __half*>(&dm_bits));

            uint8_t sc_u8, m_u8;
            pd_get_scale_min_k4(sub, blk + 4, &sc_u8, &m_u8);

            const int* qs32 = reinterpret_cast<const int*>(blk + 16 + grp_w * 32);
            int q4_0 = qs32[slot];
            int q4_1 = qs32[slot + 4];

            int v0 = is_hi ? ((q4_0 >> 4) & 0x0F0F0F0F) : (q4_0 & 0x0F0F0F0F);
            int v1 = is_hi ? ((q4_1 >> 4) & 0x0F0F0F0F) : (q4_1 & 0x0F0F0F0F);

            const uint8_t* q8_1_block = s_x_q8_1 + (size_t)(sb * 8 + sub) * 36;
            uint16_t d8_bits = q8_1_block[0] | (q8_1_block[1] << 8);
            float d8 = __half2float(*reinterpret_cast<const __half*>(&d8_bits));

            const int* qs8 = reinterpret_cast<const int*>(q8_1_block + 4);
            int u0 = qs8[slot];
            int u1 = qs8[slot + 4];

            int sumi = __dp4a(v0, u0, 0);
            sumi     = __dp4a(v1, u1, sumi);

            int sum_u = __dp4a(0x01010101, u0, 0);
            sum_u     = __dp4a(0x01010101, u1, sum_u);

            float lane_d = d_w    * (float)sc_u8 * (float)sumi  * d8;
            float lane_m = dmin_w * (float)m_u8  * (float)sum_u * d8;

            partial += (lane_d - lane_m);
        }

        float row_sum = warp_reduce_sum(partial);

        if (row_active && lane == 0) {
            float b = (bias_sel != nullptr) ? bias_sel[local_row] : 0.0f;
            out_sel[local_row] = row_sum + b;
        }
        __syncthreads();
    }
}

// ─── Phase 1 body: in-place RoPE on q_buf + k_buf ──────────────────────
//
// Mirrors rope_halved_qk_f32 / rope_interleaved_qk_f32 from kernels/rope.cu
// but collapses the grid. The standalone kernels launch (1, n_q+n_kv, 1)
// blocks; here each persistent block walks a stride over the (n_q+n_kv)
// head list. cos/sin tables are indexed by `pos` (scalar — decode is
// single-token, so n_tokens == 1 and pos_offset/positions collapse to the
// kernel-arg `pos`).
//
// head_dim == rope_dim is the common case; we parametrize rope_dim so
// partial-rope models (Qwen2 5/7 use full rope, Llama3 uses full) stay
// representable even though nothing in-tree uses partial today.
static __device__ __forceinline__ void phase_rope_body(
    float* __restrict__ q,                // [n_heads * head_dim]
    float* __restrict__ k,                // [n_kv_heads * head_dim]
    const float* __restrict__ cos_table,  // [max_seq, rope_dim/2]
    const float* __restrict__ sin_table,  // [max_seq, rope_dim/2]
    const int n_heads,
    const int n_kv_heads,
    const int head_dim,
    const int rope_dim,
    const int pos,
    const int interleaved,                // 0 = halved, 1 = interleaved
    const int block_id,
    const int n_blocks
) {
    const int tid         = threadIdx.x;
    const int total_heads = n_heads + n_kv_heads;
    const int half        = rope_dim / 2;

    const float* cos_ptr = cos_table + pos * half;
    const float* sin_ptr = sin_table + pos * half;

    for (int h = block_id; h < total_heads; h += n_blocks) {
        float* head_ptr;
        if (h < n_heads) {
            head_ptr = q + h * head_dim;
        } else {
            head_ptr = k + (h - n_heads) * head_dim;
        }

        if (interleaved) {
            for (int i = tid; i < half; i += blockDim.x) {
                float x0 = head_ptr[2 * i];
                float x1 = head_ptr[2 * i + 1];
                float c  = cos_ptr[i];
                float s  = sin_ptr[i];
                head_ptr[2 * i]     = x0 * c - x1 * s;
                head_ptr[2 * i + 1] = x0 * s + x1 * c;
            }
        } else {
            for (int i = tid; i < half; i += blockDim.x) {
                float x0 = head_ptr[i];
                float x1 = head_ptr[i + half];
                float c  = cos_ptr[i];
                float s  = sin_ptr[i];
                head_ptr[i]        = x0 * c - x1 * s;
                head_ptr[i + half] = x0 * s + x1 * c;
            }
        }
        __syncthreads();
    }
}

// ─── Phase stubs (3-7 — Phase 3+ work) ─────────────────────────────────

// Phase 2: scatter new K/V vectors into their cache slot at `pos`.
// Layout: K_cache, V_cache are [n_kv_heads, max_seq, head_dim] row-major.
// k_new, v_new are [n_kv_heads, head_dim] contiguous.
// For Qwen2 7B Q4K: n_kv_heads=4, head_dim=128 → 1024 floats total per K/V.
// Grid-strided loop over n_kv_heads * head_dim elements (×2 for K+V).
static __device__ __forceinline__ void phase_kv_append_body(
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    float* __restrict__ K_cache,
    float* __restrict__ V_cache,
    const int n_kv_heads,
    const int max_seq,
    const int head_dim,
    const int pos,
    const int block_id,
    const int n_blocks
) {
    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x;
    const int total_threads = n_blocks * threads_per_block;
    const int global_tid = block_id * threads_per_block + tid;
    const int total_elements = n_kv_heads * head_dim;
    // K scatter
    for (int i = global_tid; i < total_elements; i += total_threads) {
        const int h = i / head_dim;
        const int d = i % head_dim;
        const int dst_idx = (h * max_seq + pos) * head_dim + d;
        K_cache[dst_idx] = k_new[i];
    }
    // V scatter (same layout)
    for (int i = global_tid; i < total_elements; i += total_threads) {
        const int h = i / head_dim;
        const int d = i % head_dim;
        const int dst_idx = (h * max_seq + pos) * head_dim + d;
        V_cache[dst_idx] = v_new[i];
    }
}

static __device__ __noinline__ void phase_attn_stub(
    const float* __restrict__ /*Q*/,
    const float* __restrict__ /*K*/,
    const float* __restrict__ /*V*/,
    float* __restrict__ /*attn_out*/,
    int /*n_heads*/,
    int /*n_kv_heads*/,
    int /*seq_len*/,
    int /*max_seq*/,
    int /*head_dim*/,
    float /*scale*/
) {}

static __device__ __noinline__ void phase_wo_stub(
    const uint8_t* __restrict__ /*W_o*/,
    const float* __restrict__ /*attn_out*/,
    float* __restrict__ /*wo_out*/,
    int /*hidden_dim*/
) {}

// Phase 5: residual += delta. Trivial vector add. Each block strides through
// hidden_dim with its own slice via `block_id * threads + tid` global index.
// Promoted from stub 2026-04-18 (Phase 3 Step 0 — plumbing verification).
static __device__ __forceinline__ void phase_res_add_body(
    const float* __restrict__ delta,
    float* __restrict__ residual,
    const int hidden_dim,
    const int block_id,
    const int n_blocks
) {
    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x;
    const int total_threads = n_blocks * threads_per_block;
    const int global_tid = block_id * threads_per_block + tid;
    #pragma unroll 2
    for (int i = global_tid; i < hidden_dim; i += total_threads) {
        residual[i] += delta[i];
    }
}

static __device__ __noinline__ void phase_rmsnorm_gateup_stub(
    const float* __restrict__ /*input*/,
    const float* __restrict__ /*norm_weight*/,
    const uint8_t* __restrict__ /*W_gate*/,
    const uint8_t* __restrict__ /*W_up*/,
    float* __restrict__ /*intermediate*/,
    int /*hidden_dim*/,
    int /*intermediate_dim*/,
    float /*eps*/
) {}

static __device__ __noinline__ void phase_down_stub(
    const uint8_t* __restrict__ /*W_down*/,
    const float* __restrict__ /*intermediate*/,
    float* __restrict__ /*residual*/,
    int /*hidden_dim*/,
    int /*intermediate_dim*/
) {}

// ─── Persistent kernel entry point ─────────────────────────────────────
//
// Phase 2 signature: adds bias_q/k/v, cos/sin tables, rope_dim, and an
// interleaved flag so the rope body has everything it needs without a
// second kernel call. Nullable bias pointers match the standalone
// fused_norm_q4km_qkv_bias_dp4a contract.

extern "C" __global__
__launch_bounds__(256, 1)
void persistent_decode_layer_f32(
    // Residual stream (in-out). Phase 0 reads; phases 5 / 7 write.
    float* __restrict__ residual,
    // Norm weights
    const float* __restrict__ norm_attn_weight,
    const float* __restrict__ norm_ffn_weight,
    // QKV projections (Q4_K_M) + optional biases
    const uint8_t* __restrict__ W_q,
    const uint8_t* __restrict__ W_k,
    const uint8_t* __restrict__ W_v,
    const float* __restrict__ bias_q,        // may be NULL
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    // Attention output projection
    const uint8_t* __restrict__ W_o,
    // MLP weights (Q4_K_M)
    const uint8_t* __restrict__ W_gate,
    const uint8_t* __restrict__ W_up,
    const uint8_t* __restrict__ W_down,
    // RoPE tables
    const float* __restrict__ cos_table,     // [max_seq, rope_dim/2]
    const float* __restrict__ sin_table,     // [max_seq, rope_dim/2]
    // Scratch buffers
    float* __restrict__ q_buf,
    float* __restrict__ k_buf,
    float* __restrict__ v_buf,
    float* __restrict__ attn_buf,
    float* __restrict__ wo_buf,
    float* __restrict__ intermediate,
    // KV cache (padded)
    float* __restrict__ K_cache,
    float* __restrict__ V_cache,
    // Dims
    int hidden_dim,
    int intermediate_dim,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int rope_dim,
    int max_seq,
    int seq_len,
    int pos,
    int rope_interleaved,                    // 0 = halved, 1 = interleaved
    float rms_eps,
    float attn_scale,
    // Sync counter (zeroed by launcher before entry)
    int* __restrict__ g_phase_counter,
    int n_blocks
) {
    // Dynamic shmem carved once. Phase 0 owns it; later phases unused for now.
    extern __shared__ float s_mem_pd[];
    float*   s_normed = s_mem_pd;
    uint8_t* s_x_q8_1 = reinterpret_cast<uint8_t*>(s_mem_pd + hidden_dim);

    const int block_id = blockIdx.x;

    // ── Phase 0: RMSNorm + QKV projection ──
    phase_rmsnorm_qkv_body(
        residual, norm_attn_weight,
        W_q, W_k, W_v,
        bias_q, bias_k, bias_v,
        q_buf, k_buf, v_buf,
        s_normed, s_x_q8_1,
        hidden_dim,
        n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim,
        rms_eps,
        block_id, n_blocks
    );
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_RMSNORM_QKV, n_blocks);

    // ── Phase 1: RoPE on Q, K ──
    phase_rope_body(
        q_buf, k_buf,
        cos_table, sin_table,
        n_heads, n_kv_heads,
        head_dim, rope_dim, pos,
        rope_interleaved,
        block_id, n_blocks
    );
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_ROPE, n_blocks);

    // ── Phase 2: KV cache append ──
    phase_kv_append_body(k_buf, v_buf, K_cache, V_cache,
        n_kv_heads, max_seq, head_dim, pos, block_id, n_blocks);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_KV_APPEND, n_blocks);

    // ── Phase 3: Attention (stub) ──
    phase_attn_stub(q_buf, K_cache, V_cache, attn_buf,
        n_heads, n_kv_heads, seq_len, max_seq, head_dim, attn_scale);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_ATTN, n_blocks);

    // ── Phase 4: W_o projection (stub) ──
    phase_wo_stub(W_o, attn_buf, wo_buf, hidden_dim);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_WO, n_blocks);

    // ── Phase 5: residual += wo_buf ──
    phase_res_add_body(wo_buf, residual, hidden_dim, block_id, n_blocks);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_RES_ADD, n_blocks);

    // ── Phase 6: RMSNorm + gate*up*silu (stub) ──
    phase_rmsnorm_gateup_stub(
        residual, norm_ffn_weight,
        W_gate, W_up, intermediate,
        hidden_dim, intermediate_dim, rms_eps
    );
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_RMSNORM_GATEUP, n_blocks);

    // ── Phase 7: down projection + final residual add (stub) ──
    phase_down_stub(W_down, intermediate, residual,
        hidden_dim, intermediate_dim);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_DOWN, n_blocks);

    // ── Phase 8: drain ──
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_DONE, n_blocks);
}
