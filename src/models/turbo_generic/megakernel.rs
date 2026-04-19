//! Megakernel Sprint-1 helper: assemble a `PersistentDecodeBuffers` from a
//! live `LayerWeights` + per-model `DecodeScratch` + per-layer `GpuKvCache`.
//!
//! This doesn't wire the persistent kernel into `forward()` — that's Sprint 2.
//! It just isolates the weight / bias / norm / rope / scratch extraction so
//! the Sprint 2 wire-up is a one-call site rather than inline chaos.
//!
//! **Gates the caller must check before building** (we don't re-check here):
//! * CUDA feature on, `persistent-kernel` feature on, `TQ_MEGAKERNEL=1` env.
//! * `seq_len == 1` (single-token decode).
//! * Layer has no `post_attention_norm` / `post_ffn_norm` (Gemma-style
//!   double-norms not modelled by the persistent kernel).
//! * QKV is `Separate` (not `Merged` — Phi-3.5).
//! * MLP is `Mlp` (gate+up+down) — not `UpDown`, not MoE.
//! * All QMatMul slots are `Quantized(QWeight)` with GPU cache already
//!   populated (i.e. `warmup_qweight` has run for this layer).
//! * The compute layer is NOT using TQ compression for this step (uncompressed
//!   attention path — megakernel's Phase 3 only models standard flash-decode).
//!
//! Violations return `Err` so the caller can fall back to the standalone path.

#![cfg(feature = "persistent-kernel")]
#![cfg(feature = "cuda")]

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use crate::cuda::kernels::PersistentDecodeBuffers;
use super::kv_cache::{DecodeScratch, GpuKvCache};
use super::layer::{LayerWeights, QkvWeights};
use super::mlp::{MlpOrMoe, QMatMul};

/// Thin error type — string payload keeps it cheap and the failure modes are
/// all "caller must fall back to standalone" anyway.
#[derive(Debug)]
pub(crate) struct AssembleError(pub String);

impl std::fmt::Display for AssembleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "megakernel assemble: {}", self.0)
    }
}

/// Return the GPU byte slice of a Q4_K_M-quantized weight, or an error if
/// the weight is dequantised (`QMatMul::Full`) or hasn't been uploaded yet.
/// Uses the existing `q4k_gpu_data()` helper on `mlp::QMatMul`, which also
/// runs the lazy upload via `qw.gpu_cache_or_upload` if the cache is empty.
fn qweight_gpu_slice<'a>(qm: &'a QMatMul, what: &str) -> Result<&'a CudaSlice<u8>, AssembleError> {
    qm.q4k_gpu_data()
        .map(|(slice, _out, _in)| slice)
        .ok_or_else(|| AssembleError(format!(
            "{} is not Q4_K_M quantized — megakernel requires Q4_K_M",
            what
        )))
}

/// Assemble a `PersistentDecodeBuffers` view referencing the layer's weights,
/// biases, norms, rope tables, scratch buffers, KV cache, and the given
/// residual + phase_counter.
///
/// Borrows:
/// * `&'a layer`           — immutable weight + norm + rope access.
/// * `&'a mut scratch`     — splits into disjoint `q_buf`/`k_buf`/`v_buf`/
///   `attn_out`/`wo_out`/`intermediate_buf` mutable refs via `Arc::make_mut`.
/// * `&'a mut kv`          — disjoint `k_buf`/`v_buf` mutable refs.
/// * `&'a mut residual`    — the layer input buffer (overwritten by Phase 5/7).
/// * `&'a mut phase_counter` — zeroed before the launch by the caller.
///
/// The returned view captures `'a` across all inputs; it's only safe to use
/// while those backing stores stay alive, which is how `forward()`'s layer
/// loop already structures them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_persistent_buffers<'a>(
    layer: &'a LayerWeights,
    scratch: &'a mut DecodeScratch,
    kv: &'a mut GpuKvCache,
    residual: &'a mut CudaSlice<f32>,
    phase_counter: &'a mut CudaSlice<i32>,
) -> Result<PersistentDecodeBuffers<'a>, AssembleError> {
    // ── Hard gates (caller should pre-check; these are belt-and-braces) ──
    if layer.post_attention_norm.is_some() {
        return Err(AssembleError("layer has post_attention_norm (Gemma 2+)".into()));
    }
    if layer.post_ffn_norm.is_some() {
        return Err(AssembleError("layer has post_ffn_norm (Gemma 2+)".into()));
    }

    // ── QKV (separate only; merged is Phi-3.5, unsupported here) ──
    let (w_q, w_k, w_v) = match &layer.qkv {
        QkvWeights::Separate { wq, wk, wv } => (
            qweight_gpu_slice(wq, "wq")?,
            qweight_gpu_slice(wk, "wk")?,
            qweight_gpu_slice(wv, "wv")?,
        ),
        QkvWeights::Merged { .. } => {
            return Err(AssembleError("QKV is Merged (Phi-3.5 layout)".into()));
        }
    };
    let w_o = qweight_gpu_slice(&layer.attention_wo, "wo")?;

    // ── MLP (gate+up+down only; not UpDown, not MoE) ──
    let mlp = match &layer.mlp_or_moe {
        MlpOrMoe::Mlp(m) => m,
        MlpOrMoe::UpDown(_) => {
            return Err(AssembleError("MLP is UpDown (Phi-style)".into()));
        }
        MlpOrMoe::MoE { .. } => {
            return Err(AssembleError("MLP is MoE".into()));
        }
    };
    // Naming legacy: w1 = gate, w3 = up, w2 = down (LLaMA convention).
    let w_gate = qweight_gpu_slice(&mlp.feed_forward_w1, "ffn.gate")?;
    let w_up   = qweight_gpu_slice(&mlp.feed_forward_w3, "ffn.up")?;
    let w_down = qweight_gpu_slice(&mlp.feed_forward_w2, "ffn.down")?;

    // ── Optional QKV biases (Qwen2 has them, Llama doesn't) ──
    let bias_q = layer.attention_bq.as_ref().map(|t| t.cuda_data());
    let bias_k = layer.attention_bk.as_ref().map(|t| t.cuda_data());
    let bias_v = layer.attention_bv.as_ref().map(|t| t.cuda_data());

    // ── Norm weights (always present). RmsNorm.weight is a Tensor. ──
    let norm_attn_weight = layer.attention_norm.weight.cuda_data();
    let norm_ffn_weight  = layer.ffn_norm.weight.cuda_data();

    // ── RoPE cos/sin — per-layer in tq, keyed by `rope_dim` / `rope_style`. ──
    let cos_table = layer.cos.cuda_data();
    let sin_table = layer.sin.cuda_data();

    // ── Scratch — Arc::make_mut on disjoint fields. Refcount should be 1
    //   inside `forward()`'s `decode_scratch.take()` window, so this is a
    //   fast no-copy path. If refcount > 1 we'd clone a GB of scratch; the
    //   caller's gate should have prevented that. ──
    let q_buf        = Arc::make_mut(&mut scratch.q_buf);
    let k_buf        = Arc::make_mut(&mut scratch.k_buf);
    let v_buf        = Arc::make_mut(&mut scratch.v_buf);
    let attn_buf     = Arc::make_mut(&mut scratch.attn_out);
    let wo_buf       = Arc::make_mut(&mut scratch.wo_out);
    let intermediate = Arc::make_mut(&mut scratch.intermediate_buf);

    // ── KV cache — same Arc::make_mut trick. ──
    let k_cache = Arc::make_mut(&mut kv.k_buf);
    let v_cache = Arc::make_mut(&mut kv.v_buf);

    Ok(PersistentDecodeBuffers {
        residual,
        norm_attn_weight,
        norm_ffn_weight,
        w_q, w_k, w_v,
        bias_q, bias_k, bias_v,
        w_o,
        w_gate, w_up, w_down,
        cos_table, sin_table,
        q_buf, k_buf, v_buf,
        attn_buf, wo_buf, intermediate,
        k_cache, v_cache,
        phase_counter,
    })
}

// ─── Sprint-2 call site ─────────────────────────────────────────────────

/// Cached env-var gate. `TQ_MEGAKERNEL=1` is read once per process and
/// sticks — matches every other `TQ_*` knob's behaviour and keeps the
/// hot-path check free.
fn megakernel_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("TQ_MEGAKERNEL").ok().as_deref() == Some("1"))
}

/// Try to run the whole layer through `persistent_decode_layer`. Returns
/// `Ok(true)` when the megakernel handled the step (caller should
/// `continue` the layer loop), `Ok(false)` when any gate failed and the
/// caller should fall back to the standard dispatch.
///
/// Runtime gates:
///   * `TQ_MEGAKERNEL=1` (process-wide cache).
///   * `seq_len == 1` (decode only — no prefill path in the kernel).
///   * `capturing == false` (CUDA-graph capture not supported yet).
///   * `!layer_uses_compression` (TQ path is a separate kernel set).
///   * `layer.gpu_kv_cache` already initialised (first decode step after
///     prefill seeds it via the standalone path — we don't replicate the
///     seeding logic here).
///   * `layer_in.is_cuda()` (residual must be on GPU).
///   * Layer shape gates inside `assemble_persistent_buffers` — if any
///     fails we treat the whole layer as "not eligible" and fall back.
///
/// Note: we increment `gpu_kv_cache.seq_len` by 1 AFTER the launch to
/// match what `GpuKvCache::append` would have done for the standalone
/// path. The megakernel itself writes the K/V slot at `pos` via
/// `phase_kv_append_body` — this function only updates the Rust-side
/// length tracker.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_megakernel_layer(
    layer: &mut LayerWeights,
    layer_in: &mut crate::cuda::TqTensor,
    decode_scratch: &mut Option<DecodeScratch>,
    seq_len: usize,
    capturing: bool,
    layer_uses_compression: bool,
    index_pos: usize,
) -> Result<bool, AssembleError> {
    // ── Fast-out gates (no work, no allocs) ─────────────────────────────
    if !megakernel_enabled() { return Ok(false); }
    if seq_len != 1 { return Ok(false); }
    if capturing { return Ok(false); }
    if layer_uses_compression { return Ok(false); }
    if !layer_in.is_cuda() { return Ok(false); }

    // ── KV cache must already exist (seeded by the standalone path on
    //   the first decode step after prefill). The actual mutable borrow
    //   is taken below after all immutable field borrows — we only need
    //   to check existence here. ──
    if layer.gpu_kv_cache.is_none() { return Ok(false); }

    // ── DecodeScratch must be out of self (caller already took it). ──
    let scratch: &mut DecodeScratch = match decode_scratch.as_mut() {
        Some(s) => s,
        None => return Ok(false),
    };

    // ── Layer-shape gates (separate QKV, gated-silu MLP, Q4_K_M weights
    //   already uploaded). Weights fetched here double as the assembly's
    //   inputs below — we can't borrow them again after taking the &mut
    //   reference to layer.gpu_kv_cache. ──
    let (wq_bytes, wk_bytes, wv_bytes) = match &layer.qkv {
        QkvWeights::Separate { wq, wk, wv } => (
            wq.q4k_gpu_data().map(|(s, _, _)| s)
                .ok_or_else(|| AssembleError("wq not Q4K or not uploaded".into()))?,
            wk.q4k_gpu_data().map(|(s, _, _)| s)
                .ok_or_else(|| AssembleError("wk not Q4K or not uploaded".into()))?,
            wv.q4k_gpu_data().map(|(s, _, _)| s)
                .ok_or_else(|| AssembleError("wv not Q4K or not uploaded".into()))?,
        ),
        QkvWeights::Merged { .. } => return Ok(false),
    };
    let wo_bytes = layer.attention_wo.q4k_gpu_data().map(|(s, _, _)| s)
        .ok_or_else(|| AssembleError("wo not Q4K or not uploaded".into()))?;
    let (gate_bytes, up_bytes, down_bytes) = match &layer.mlp_or_moe {
        MlpOrMoe::Mlp(m) => (
            m.feed_forward_w1.q4k_gpu_data().map(|(s, _, _)| s)
                .ok_or_else(|| AssembleError("gate not Q4K".into()))?,
            m.feed_forward_w3.q4k_gpu_data().map(|(s, _, _)| s)
                .ok_or_else(|| AssembleError("up not Q4K".into()))?,
            m.feed_forward_w2.q4k_gpu_data().map(|(s, _, _)| s)
                .ok_or_else(|| AssembleError("down not Q4K".into()))?,
        ),
        _ => return Ok(false),
    };
    let bias_q_ref = layer.attention_bq.as_ref().map(|t| t.cuda_data());
    let bias_k_ref = layer.attention_bk.as_ref().map(|t| t.cuda_data());
    let bias_v_ref = layer.attention_bv.as_ref().map(|t| t.cuda_data());
    let norm_attn_ref = layer.attention_norm.weight.cuda_data();
    let norm_ffn_ref  = layer.ffn_norm.weight.cuda_data();
    let cos_ref = layer.cos.cuda_data();
    let sin_ref = layer.sin.cuda_data();

    // ── Dims + scale values (mirror forward_attn's `compute_qkv_and_rope`
    //   constants to keep the Rust↔kernel contract in one place). ──
    let hidden_dim = scratch.hidden_dim;
    let intermediate_dim = scratch.intermediate_dim;
    let n_heads = layer.n_head;
    let n_kv_heads = layer.n_kv_head;
    let head_dim = layer.head_dim;
    let rope_dim = layer.rope_dim;
    let rms_eps = layer.ffn_norm.eps as f32;
    let attn_scale = 1.0f32 / (layer.attn_scale_denom as f32);
    let rope_interleaved = matches!(layer.rope_style, super::kv_cache::RopeStyle::Interleaved);
    let pos = index_pos;

    // ── Phase counter: 1 × i32 zeroed. 4-byte alloc per call is trivial
    //   next to the 50 kernel launches of a layer. ──
    let reg = crate::cuda::kernels::global_registry()
        .ok_or_else(|| AssembleError("no GPU kernel registry".into()))?;
    let mut phase_counter: cudarc::driver::CudaSlice<i32> = reg.stream
        .alloc_zeros(1)
        .map_err(|e| AssembleError(format!("phase_counter alloc: {}", e)))?;

    // ── NOW take disjoint mutable field borrows. The immutable refs
    //   above are still live, but they don't intersect gpu_kv_cache,
    //   so rustc's disjoint-field-borrow check lets this through. ──
    let kv_slot = layer.gpu_kv_cache.as_mut()
        .expect("gpu_kv_cache checked is_some above");
    let max_seq = kv_slot.max_seq;

    let residual: &mut cudarc::driver::CudaSlice<f32> = layer_in.cuda_data_mut();

    let q_buf        = Arc::make_mut(&mut scratch.q_buf);
    let k_buf        = Arc::make_mut(&mut scratch.k_buf);
    let v_buf        = Arc::make_mut(&mut scratch.v_buf);
    let attn_buf     = Arc::make_mut(&mut scratch.attn_out);
    let wo_buf       = Arc::make_mut(&mut scratch.wo_out);
    let intermediate = Arc::make_mut(&mut scratch.intermediate_buf);
    let k_cache      = Arc::make_mut(&mut kv_slot.k_buf);
    let v_cache      = Arc::make_mut(&mut kv_slot.v_buf);

    let mut bufs = PersistentDecodeBuffers {
        residual,
        norm_attn_weight: norm_attn_ref,
        norm_ffn_weight:  norm_ffn_ref,
        w_q: wq_bytes, w_k: wk_bytes, w_v: wv_bytes,
        bias_q: bias_q_ref, bias_k: bias_k_ref, bias_v: bias_v_ref,
        w_o: wo_bytes,
        w_gate: gate_bytes, w_up: up_bytes, w_down: down_bytes,
        cos_table: cos_ref, sin_table: sin_ref,
        q_buf, k_buf, v_buf,
        attn_buf, wo_buf, intermediate,
        k_cache, v_cache,
        phase_counter: &mut phase_counter,
    };

    crate::cuda::kernels::persistent_decode_layer(
        reg,
        &mut bufs,
        hidden_dim, intermediate_dim,
        n_heads, n_kv_heads, head_dim, rope_dim,
        max_seq, seq_len, pos,
        rope_interleaved,
        rms_eps, attn_scale,
    ).map_err(|e| AssembleError(format!("persistent_decode_layer launch: {:?}", e)))?;

    // Drop `bufs` (its &mut borrows) before touching kv_slot.seq_len.
    drop(bufs);

    kv_slot.seq_len = (kv_slot.seq_len + 1).min(max_seq);

    Ok(true)
}
