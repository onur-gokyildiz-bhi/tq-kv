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
