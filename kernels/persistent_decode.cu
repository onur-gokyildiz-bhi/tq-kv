// kernels/persistent_decode.cu
//
// Persistent Kernel Lite — Phase 1 skeleton (Plan #6).
// See docs/persistent-kernel-lite-design.md.
//
// One block per SM, stays alive across the entire decoder layer. Walks a
// 9-phase micro-ISA (RMSNORM_QKV, ROPE, KV_APPEND, ATTN, WO, RES_ADD,
// RMSNORM_GATEUP, DOWN, DONE) synchronized via an atomic counter +
// __threadfence + __syncthreads. Hazy Research Llama-1B megakernel pattern.
//
// PHASE 1 STATUS: skeleton only. Each phase is a stub — it advances the
// counter, emits a threadfence, and syncs. No real work. Goal is to prove
// the persistent-kernel control flow compiles and that the counter-based
// sync is wired correctly.
//
// Phase 2 will fold the existing dp4a kernels (fused_norm_q4km_qkv_bias_dp4a,
// fused_addnorm_q4km_gateup_silu_dp4a, fused_q4km_down_residual_dp4a,
// gqa_decode_attention) into __device__ functions callable from here.

#include "common.cuh"

// ─── Phase IDs ─────────────────────────────────────────────────────────
// Must match the Rust launcher's expectation.
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
//
// Each block atomically increments the counter when it enters a phase.
// Once counter == n_blocks_expected * (phase_idx + 1), all blocks have
// checked in and the phase is complete. Remaining blocks spin-wait.
//
// __nanosleep is sm_70+; we guard it to keep sm_75 builds clean.
static __device__ __forceinline__ void phase_wait(
    int* g_counter,
    int target_value
) {
    // Simple spin-wait. Phase 4 will replace this with a warp-elected
    // atomicAdd + bulk sync.
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
    // One thread per block owns the atomicAdd to avoid 256-way contention.
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(g_counter, 1);
        __threadfence();
    }
    // All blocks spin until n_blocks checked in for this phase.
    if (threadIdx.x == 0) {
        phase_wait(g_counter, (phase_idx + 1) * n_blocks);
    }
    __syncthreads();
}

// ─── Phase stubs ───────────────────────────────────────────────────────
//
// Phase 1 skeleton: each stub is a no-op. Phase 2 will call the
// corresponding dp4a device function. Marked __noinline__ so ptxas emits
// them as separate basic blocks — easier to read in SASS dumps.

static __device__ __noinline__ void phase_rmsnorm_qkv_stub(
    const float* __restrict__ /*input*/,
    const float* __restrict__ /*norm_weight*/,
    const uint8_t* __restrict__ /*W_q*/,
    const uint8_t* __restrict__ /*W_k*/,
    const uint8_t* __restrict__ /*W_v*/,
    float* __restrict__ /*q_out*/,
    float* __restrict__ /*k_slot*/,
    float* __restrict__ /*v_slot*/,
    int /*hidden_dim*/,
    int /*q_out_dim*/,
    int /*kv_out_dim*/,
    float /*eps*/
) {
    // no-op
}

static __device__ __noinline__ void phase_rope_stub(
    float* __restrict__ /*q*/,
    float* __restrict__ /*k*/,
    int /*n_heads*/,
    int /*n_kv_heads*/,
    int /*head_dim*/,
    int /*pos*/
) {
    // no-op
}

static __device__ __noinline__ void phase_kv_append_stub(
    const float* __restrict__ /*k_new*/,
    const float* __restrict__ /*v_new*/,
    float* __restrict__ /*K_cache*/,
    float* __restrict__ /*V_cache*/,
    int /*n_kv_heads*/,
    int /*max_seq*/,
    int /*head_dim*/,
    int /*pos*/
) {
    // no-op
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
) {
    // no-op
}

static __device__ __noinline__ void phase_wo_stub(
    const uint8_t* __restrict__ /*W_o*/,
    const float* __restrict__ /*attn_out*/,
    float* __restrict__ /*wo_out*/,
    int /*hidden_dim*/
) {
    // no-op
}

static __device__ __noinline__ void phase_res_add_stub(
    const float* __restrict__ /*delta*/,
    float* __restrict__ /*residual*/,
    int /*hidden_dim*/
) {
    // no-op
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
) {
    // no-op
}

static __device__ __noinline__ void phase_down_stub(
    const uint8_t* __restrict__ /*W_down*/,
    const float* __restrict__ /*intermediate*/,
    float* __restrict__ /*residual*/,
    int /*hidden_dim*/,
    int /*intermediate_dim*/
) {
    // no-op
}

// ─── Persistent kernel entry point ─────────────────────────────────────
//
// Launched once per decoder layer per token. Every block runs the full
// phase walk. The atomic counter must be zeroed by the launcher before
// entry (one counter per concurrent layer).
//
// Arguments are a superset of the current per-phase launchers — the
// persistent kernel owns all of the layer's buffers at once. Phase 2
// will split this into read/write banks to enable double buffering.

extern "C" __global__
__launch_bounds__(256, 1)
void persistent_decode_layer_f32(
    // Residual stream (in-out)
    float* __restrict__ residual,        // [hidden_dim]
    // Norm weights
    const float* __restrict__ norm_attn_weight,   // [hidden_dim]
    const float* __restrict__ norm_ffn_weight,    // [hidden_dim]
    // QKV projections (Q4_K_M)
    const uint8_t* __restrict__ W_q,
    const uint8_t* __restrict__ W_k,
    const uint8_t* __restrict__ W_v,
    // Attention output projection
    const uint8_t* __restrict__ W_o,
    // MLP weights (Q4_K_M)
    const uint8_t* __restrict__ W_gate,
    const uint8_t* __restrict__ W_up,
    const uint8_t* __restrict__ W_down,
    // Scratch buffers
    float* __restrict__ q_buf,           // [n_heads * head_dim]
    float* __restrict__ k_buf,           // [n_kv_heads * head_dim]
    float* __restrict__ v_buf,           // [n_kv_heads * head_dim]
    float* __restrict__ attn_buf,        // [n_heads * head_dim]
    float* __restrict__ wo_buf,          // [hidden_dim]
    float* __restrict__ intermediate,    // [intermediate_dim]
    // KV cache (padded)
    float* __restrict__ K_cache,         // [n_kv_heads, max_seq, head_dim]
    float* __restrict__ V_cache,         // [n_kv_heads, max_seq, head_dim]
    // Dims
    int hidden_dim,
    int intermediate_dim,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    int seq_len,
    int pos,
    float rms_eps,
    float attn_scale,
    // Sync counter (zeroed by launcher before entry)
    int* __restrict__ g_phase_counter,
    int n_blocks
) {
    // ── Phase 0: RMSNorm + QKV projection ──
    phase_rmsnorm_qkv_stub(
        residual, norm_attn_weight,
        W_q, W_k, W_v,
        q_buf, k_buf, v_buf,
        hidden_dim, n_heads * head_dim, n_kv_heads * head_dim,
        rms_eps
    );
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_RMSNORM_QKV, n_blocks);

    // ── Phase 1: RoPE on Q, K ──
    phase_rope_stub(q_buf, k_buf, n_heads, n_kv_heads, head_dim, pos);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_ROPE, n_blocks);

    // ── Phase 2: KV cache append ──
    phase_kv_append_stub(k_buf, v_buf, K_cache, V_cache,
        n_kv_heads, max_seq, head_dim, pos);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_KV_APPEND, n_blocks);

    // ── Phase 3: Attention ──
    phase_attn_stub(q_buf, K_cache, V_cache, attn_buf,
        n_heads, n_kv_heads, seq_len, max_seq, head_dim, attn_scale);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_ATTN, n_blocks);

    // ── Phase 4: W_o projection ──
    phase_wo_stub(W_o, attn_buf, wo_buf, hidden_dim);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_WO, n_blocks);

    // ── Phase 5: residual += wo_buf ──
    phase_res_add_stub(wo_buf, residual, hidden_dim);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_RES_ADD, n_blocks);

    // ── Phase 6: RMSNorm + gate*up*silu (dp4a-style fused) ──
    phase_rmsnorm_gateup_stub(
        residual, norm_ffn_weight,
        W_gate, W_up, intermediate,
        hidden_dim, intermediate_dim, rms_eps
    );
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_RMSNORM_GATEUP, n_blocks);

    // ── Phase 7: down projection + final residual add ──
    phase_down_stub(W_down, intermediate, residual,
        hidden_dim, intermediate_dim);
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_DOWN, n_blocks);

    // ── Phase 8: drain ──
    phase_enter_and_wait(g_phase_counter, TQ_PHASE_DONE, n_blocks);
}
