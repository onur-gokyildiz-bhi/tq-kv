//! GenericTurboModel: GGUF loading, forward pass, CUDA graph management.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::ComputeBackend;
use crate::cuda::{TqTensor as Tensor, TqDevice as Device, TqDType as DType, TqError};
use crate::cuda::Result;
use crate::gguf::{GgufContent, WeightSource, GgufSource};
use crate::qmatmul as qmm;
use tq_kv::TurboQuantConfig;

use super::primitives::{Embedding, RmsNorm, apply_softcap, Module, MAX_SEQ_LEN};
use super::mlp::*;
use super::kv_cache::*;
use super::layer::*;

/// Bail macro compatible with our error type.
macro_rules! bail {
    ($($arg:tt)*) => { return Err(TqError::Msg(format!($($arg)*))) };
}

// ============================================================
// Generic TurboQuant Model
// ============================================================

pub struct GenericTurboModel {
    pub(crate) tok_embeddings: Embedding,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) norm: RmsNorm,
    pub(crate) output: QMatMul,
    pub(crate) masks: HashMap<usize, Tensor>,  // legacy (unused after rect migration)
    pub(crate) masks_rect: HashMap<(usize, usize), Tensor>,
    /// Gemma: scale embeddings by sqrt(hidden_dim)
    pub(crate) embed_scale: Option<f32>,
    /// Gemma2 final logit soft-capping: cap * tanh(logits / cap)
    pub(crate) final_logit_softcap: Option<f32>,
    pub(crate) backend: Arc<dyn ComputeBackend>,
    pub(crate) span: tracing::Span,
    pub(crate) span_output: tracing::Span,
    /// CUDA Graph manager for decode acceleration.
    #[cfg(feature = "cuda")]
    pub(crate) graph_manager: crate::cuda::graph::CudaGraphManager,
    /// GPU buffer address of the input embedding (for updating before graph replay).
    #[cfg(feature = "cuda")]
    pub(crate) graph_input_buffer: Option<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// GPU scalar for RoPE position offset (updated before graph replay).
    #[cfg(feature = "cuda")]
    pub(crate) rope_pos_gpu: Option<cudarc::driver::CudaSlice<i32>>,
    /// Pre-allocated scratch buffers for zero-alloc decode (fused kernel path).
    #[cfg(feature = "cuda")]
    pub(crate) decode_scratch: Option<DecodeScratch>,
    /// Optional layer-swap manager for streaming weights under VRAM pressure.
    /// `None` = everything pinned at load time (default). Set by engine.rs when
    /// `TQ_LAYER_SWAP=1|force` is active.
    #[cfg(feature = "cuda")]
    pub(crate) layer_swap: Option<crate::layer_swap::LayerSwapManager>,
    /// Pre-computed QWeight pointers per layer — cached so the forward loop
    /// can reach layers[i+1]'s weights while holding `&mut layers[i]`.
    /// `QWeightPtr` is a Send+Sync wrapper so the `Engine` stays `Send`.
    /// Rebuilt when `layer_swap` is installed.
    #[cfg(feature = "cuda")]
    pub(crate) layer_qweight_ptrs: Vec<Vec<crate::layer_swap::QWeightPtr>>,
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let n_theta = theta.len();
    let theta = Tensor::from_slice(&theta, vec![n_theta], device)?;
    let idx_theta = Tensor::arange(0, context_length, device)?
        .to_dtype(DType::F32)?
        .reshape(vec![context_length, 1])?
        .matmul(&theta.reshape(vec![1, n_theta])?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

/// Detect RoPE style from architecture name.
/// Most modern architectures use halved RoPE. Llama uses interleaved.
fn detect_rope_style(arch: &str) -> RopeStyle {
    match arch {
        "llama" => RopeStyle::Interleaved,
        // qwen2, mistral, gemma, phi3, etc. all use halved
        _ => RopeStyle::Halved,
    }
}

/// Hybrid graph helper: end capture + immediately replay when the layer
/// loop hits the first TQ (compressed) layer, then flip `capturing` off
/// so subsequent TQ layers run eagerly on the same stream.
///
/// Free function, not a method, so the iter_mut() borrow of
/// `self.layers` can coexist with `&mut self.graph_manager` via
/// disjoint-field borrows at the call site.
///
/// Per-layer profiling accumulator (TQ_PROFILE=1), also at module
/// scope so `try_fused_decode_layer` can accept it by &mut ref.
struct ProfAccum { qkv_ns: u64, rope_ns: u64, attn_ns: u64, mlp_ns: u64, norm_ns: u64, other_ns: u64, n: u32 }

#[cfg(feature = "cuda")]
fn maybe_end_hybrid_capture(
    graph_manager: &mut crate::cuda::graph::CudaGraphManager,
    capturing: &mut bool,
    layer_idx: usize,
    layer_uses_compression: bool,
) {
    if !*capturing || !layer_uses_compression {
        return;
    }
    if !matches!(graph_manager.status, crate::cuda::graph::GraphStatus::Capturing) {
        return;
    }
    let Some(reg) = crate::cuda::kernels::global_registry() else { return; };

    match graph_manager.end_capture(&reg.stream, 1) {
        Ok(()) => {
            eprintln!("[cuda-graph] hybrid: captured {} non-TQ layers", layer_idx);
            // Capture only records — replay to actually execute.
            let _ = reg.stream.context().check_err();
            match graph_manager.replay(&reg.stream, 1) {
                Ok(()) => {
                    let _ = reg.stream.synchronize();
                    eprintln!("[cuda-graph] hybrid: initial replay OK");
                }
                Err(e) => {
                    eprintln!("[cuda-graph] hybrid: replay FAILED: {}", e);
                    graph_manager.reset();
                }
            }
        }
        Err(e) => {
            eprintln!("[cuda-graph] hybrid: end_capture failed: {}", e);
            graph_manager.reset();
        }
    }
    *capturing = false;
}


/// Fused decode path: 5 kernel launches replace ~13 per layer.
/// Gate: CUDA decode (seq_len=1), input on GPU, not TQ-compressed,
/// Q4K separate QKV, standard MLP, no post-norms. Returns true when
/// the path handled the layer (caller should `continue`); returns
/// false when the gate fails or the QKV weights aren't Q4K (caller
/// falls through to the separate-kernel fallback). Extracted from
/// forward() layer loop on 2026-04-13.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments, unused_variables)]
fn try_fused_decode_layer(
    layer: &mut LayerWeights,
    layer_idx: usize,
    layer_in: &mut Tensor,
    mask: Option<&Tensor>,
    index_pos: usize,
    backend: &dyn ComputeBackend,
    decode_scratch: &mut Option<DecodeScratch>,
    seq_len: usize,
    capturing: bool,
    graph_replayed: bool,
    layer_uses_compression: bool,
    gpu_debug: bool,
    profiling: bool,
    prof: &mut ProfAccum,
    prof_stream: Option<&std::sync::Arc<cudarc::driver::CudaStream>>,
) -> Result<bool> {
    // Cheap clone (Arc ref-count++ for GPU storage) so the inner body
    // can read `x` while we also hold `&mut layer_in` for writes.
    let x = layer_in.clone();
    let fused_disabled = {
        static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *C.get_or_init(|| std::env::var("TQ_NO_FUSED").map(|v| v == "1").unwrap_or(false))
    };

    // Gate: fall through to the original separate-kernel path when any
    // precondition isn't met.
    if !(seq_len == 1 && x.is_cuda() && !layer_uses_compression && !fused_disabled
        && layer.post_attention_norm.is_none()
        && layer.post_ffn_norm.is_none()
        && matches!(&layer.qkv, QkvWeights::Separate { .. })
        && matches!(&layer.mlp_or_moe, MlpOrMoe::Mlp(_)))
    {
        return Ok(false);
    }

                // Phase 1: Compute QKV into scratch buffers (scoped borrow of layer.qkv).
                // Q and K require Q4K. V can be Q4K or Q6K (fallback to f32_matvec).
                // When scratch is used: qkv_in_scratch=true, tensors=None.
                let fused_qkv: Option<(bool, Option<(Tensor, Tensor, Tensor)>, usize)> = {
                    // Extract Q+K as Q4K, V as Q4K or Q6K fallback
                    let qk_data = if let QkvWeights::Separate { wq, wk, wv: _ } = &layer.qkv {
                        match (wq.q4k_gpu_data(), wk.q4k_gpu_data()) {
                            (Some((wq_g, qo, hd)), Some((wk_g, ko, _))) =>
                                Some((wq_g, wk_g, qo, ko, hd)),
                            _ => None,
                        }
                    } else { None };

                    if let Some((wq_gpu, wk_gpu, q_out, k_out, hidden_dim)) = qk_data {
                        let reg = crate::cuda::kernels::global_registry().unwrap();
                        let input_gpu = x.cuda_data();
                        let norm_w = layer.attention_norm.weight.cuda_data();

                        if let Some(ref mut scratch) = decode_scratch {
                            let pdet_qkv = profiling && layer_idx == 0;
                            let _tqkv = if pdet_qkv {
                                crate::cuda::event_timer::EventTimer::new().ok()
                                    .filter(|t| t.start(&reg.stream).is_ok())
                            } else { None };

                            // Try fully-fused path: single kernel does RMSNorm + Q/K/V Q4K matvec + biases.
                            // Requires all three weights to be Q4K (fallback for Q6K V below).
                            let wv = if let QkvWeights::Separate { wv, .. } = &layer.qkv { wv } else { unreachable!() };
                            let v_q4k = wv.q4k_gpu_data();
                            let use_fused = v_q4k.is_some();

                            if let Some((wv_gpu, v_out, _)) = v_q4k {
                                // Plan #1: fuse rms_norm + 3× q4km_matvec + 3× bias_add → 1 kernel.
                                // Saves 6 launches per layer. PPL-safe: math is identical, only dispatch fuses.
                                let bq_ref = layer.attention_bq.as_ref().map(|b| b.cuda_data());
                                let bk_ref = layer.attention_bk.as_ref().map(|b| b.cuda_data());
                                let bv_ref = layer.attention_bv.as_ref().map(|b| b.cuda_data());
                                // Disjoint-field borrow: q_buf, k_buf, v_buf are independent Arc<CudaSlice>
                                // fields on scratch — borrowck tracks them per-field in Rust 2021.
                                let (oq, ok, ov) = (
                                    Arc::get_mut(&mut scratch.q_buf).expect("q aliased"),
                                    Arc::get_mut(&mut scratch.k_buf).expect("k aliased"),
                                    Arc::get_mut(&mut scratch.v_buf).expect("v aliased"),
                                );
                                crate::cuda::kernels::fused_norm_q4km_qkv_bias(
                                    reg, input_gpu, norm_w,
                                    wq_gpu, wk_gpu, wv_gpu,
                                    bq_ref, bk_ref, bv_ref,
                                    oq, ok, ov,
                                    hidden_dim, q_out, k_out, v_out,
                                    layer.attention_norm.eps as f32,
                                ).map_err(|e| TqError::Msg(format!("fused_norm_q4km_qkv_bias: {}", e)))?;
                            } else {
                                // V is not Q4K (e.g. Q6K) — keep the legacy 4-to-7 kernel path.
                                let ci = layer_idx & 1;
                                {
                                    let normed = Arc::get_mut(&mut scratch.combined_bufs[ci])
                                        .expect("scratch combined aliased in QKV");
                                    crate::cuda::kernels::rms_norm(
                                        reg, input_gpu, norm_w, normed,
                                        1, hidden_dim, layer.attention_norm.eps as f32,
                                    ).map_err(|e| TqError::Msg(format!("scratch rms_norm: {}", e)))?;
                                }
                                let normed_ptr: &cudarc::driver::CudaSlice<f32> = &*scratch.combined_bufs[ci];

                                {
                                    let oq = Arc::get_mut(&mut scratch.q_buf).expect("q aliased");
                                    crate::cuda::kernels::q4km_matvec(
                                        reg, wq_gpu, normed_ptr, oq, q_out, hidden_dim,
                                    ).map_err(|e| TqError::Msg(format!("scratch Q matvec: {}", e)))?;
                                }
                                {
                                    let ok = Arc::get_mut(&mut scratch.k_buf).expect("k aliased");
                                    crate::cuda::kernels::q4km_matvec(
                                        reg, wk_gpu, normed_ptr, ok, k_out, hidden_dim,
                                    ).map_err(|e| TqError::Msg(format!("scratch K matvec: {}", e)))?;
                                }
                                {
                                    let ov = Arc::get_mut(&mut scratch.v_buf).expect("v aliased");
                                    if let qmm::QMatMul::Quantized(qw) = &wv.inner {
                                        // Q6K: native fused dequant matvec (4.9x less bandwidth than F32)
                                        let w_raw = qw.gpu_cache_or_upload(&reg.stream);
                                        crate::cuda::kernels::q6k_matvec(
                                            reg, w_raw, normed_ptr, ov, qw.out_features(), qw.in_features(),
                                        ).map_err(|e| TqError::Msg(format!("scratch V Q6K: {}", e)))?;
                                    } else {
                                        return Err(TqError::Msg("V weight: unsupported format".into()));
                                    }
                                }
                                if let Some(ref bq) = layer.attention_bq {
                                    let oq = Arc::get_mut(&mut scratch.q_buf).expect("q bias");
                                    crate::cuda::kernels::bias_add_inplace(reg, oq, bq.cuda_data(), q_out)
                                        .map_err(|e| TqError::Msg(format!("Q bias: {}", e)))?;
                                }
                                if let Some(ref bk) = layer.attention_bk {
                                    let ok = Arc::get_mut(&mut scratch.k_buf).expect("k bias");
                                    crate::cuda::kernels::bias_add_inplace(reg, ok, bk.cuda_data(), k_out)
                                        .map_err(|e| TqError::Msg(format!("K bias: {}", e)))?;
                                }
                                if let Some(ref bv) = layer.attention_bv {
                                    let ov = Arc::get_mut(&mut scratch.v_buf).expect("v bias");
                                    crate::cuda::kernels::bias_add_inplace(reg, ov, bv.cuda_data(), scratch.v_out)
                                        .map_err(|e| TqError::Msg(format!("V bias: {}", e)))?;
                                }
                            }
                            if let Some(t) = _tqkv {
                                let _ = t.stop(&reg.stream);
                                let label = if use_fused { "fused-qkv" } else { "norm+qkv" };
                                eprintln!("[kernel] {:>12}: {:.1}μs", label, t.elapsed_us().unwrap_or(0.0));
                            }
                            Some((true, None, hidden_dim))
                        } else {
                            // No scratch: compute via forward, wrap as tensors
                            let normed_x = layer.attention_norm.forward(&x, backend)?;
                            let (wq_w, wk_w, wv_w) = if let QkvWeights::Separate { wq, wk, wv } = &layer.qkv {
                                (wq, wk, wv)
                            } else { unreachable!() };
                            let q_t = wq_w.forward(&normed_x, backend)?;
                            let k_t = wk_w.forward(&normed_x, backend)?;
                            let v_t = wv_w.forward(&normed_x, backend)?;
                            Some((false, Some((q_t, k_t, v_t)), hidden_dim))
                        }
                    } else { None }
                }; // QKV borrows released

                if let Some((qkv_in_scratch, qkv_tensors, hidden_dim)) = fused_qkv {
                    let attn = if qkv_in_scratch
                        && layer.gpu_kv_cache.is_some()
                        && layer.attention_wo.q4k_gpu_data().is_some()
                        // (capturing allowed — scratch pointers are stable for graph)
                    {
                        let scratch = decode_scratch.as_mut().unwrap();
                        let reg = crate::cuda::kernels::global_registry().unwrap();

                        // Per-kernel detail profiling (TQ_PROFILE=1, layer 0 only).
                        // Uses CUDA events so each section measures exactly the GPU time
                        // of the kernels launched between start/stop — no pipeline drain
                        // or host-side polling artifacts (see cuda::event_timer docs).
                        let pdet = profiling && layer_idx == 0;
                        macro_rules! ptime {
                            ($label:expr, $body:expr) => {{
                                let timer = if pdet {
                                    crate::cuda::event_timer::EventTimer::new().ok()
                                        .filter(|t| t.start(&reg.stream).is_ok())
                                } else { None };
                                let r = $body;
                                if let Some(t) = timer {
                                    let _ = t.stop(&reg.stream);
                                    let us = t.elapsed_us().unwrap_or(0.0);
                                    eprintln!("[kernel] {:>12}: {:.1}μs", $label, us);
                                }
                                r
                            }};
                        }

                        // 1. RoPE in-place (graph-safe: uses rope_pos_gpu when available)
                        let rope_gpu_pos = super::ROPE_POS_GPU_PTR.with(|p| p.get());
                        let rope_gpu_ref = if rope_gpu_pos != 0 {
                            Some(unsafe { &*(rope_gpu_pos as *const cudarc::driver::CudaSlice<i32>) })
                        } else { None };
                        // Plan #2: fused Q+K RoPE — one launch replaces two.
                        // Grid covers n_head + n_kv_head; kernel dispatches buffer by blockIdx.y.
                        ptime!("rope_qk", {
                            // Disjoint-field borrow on scratch: q_buf and k_buf are separate
                            // Arc<CudaSlice> fields — Rust 2021 accepts both &mut simultaneously.
                            let (q_mut, k_mut) = (
                                Arc::get_mut(&mut scratch.q_buf).expect("q aliased"),
                                Arc::get_mut(&mut scratch.k_buf).expect("k aliased"),
                            );
                            let rope_qk_fn = match layer.rope_style {
                                RopeStyle::Halved => crate::cuda::kernels::rope_halved_qk_with_gpu_pos,
                                RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved_qk_with_gpu_pos,
                            };
                            rope_qk_fn(reg, q_mut, k_mut, layer.cos.cuda_data(), layer.sin.cuda_data(),
                                1, scratch.n_head, scratch.n_kv_head, scratch.head_dim,
                                layer.rope_dim, index_pos, rope_gpu_ref,
                            ).map_err(|e| TqError::Msg(format!("scratch RoPE QK fused: {}", e)))
                        })?;

                        // 2. Append K,V to GpuKvCache
                        ptime!("kv_append", {
                            let k_view = Tensor::from_cuda_arc(Arc::clone(&scratch.k_buf),
                                vec![1, scratch.n_kv_head, 1, scratch.head_dim], scratch.stream.clone());
                            let v_view = Tensor::from_cuda_arc(Arc::clone(&scratch.v_buf),
                                vec![1, scratch.n_kv_head, 1, scratch.head_dim], scratch.stream.clone());
                            layer.gpu_kv_cache.as_mut().unwrap().append(&k_view, &v_view, 1)
                        })?;

                        // 3. Fused GQA decode attention (graph-safe: reads seq_len from GPU scalar)
                        // Flash decode for long context (seq_len > 256) in eager mode only.
                        // CUDA Graph mode always uses gqa_decode (fixed buffer addresses).
                        ptime!("gqa_attn", {
                            let gpu_kv = layer.gpu_kv_cache.as_ref().unwrap();
                            let attn_mut = Arc::get_mut(&mut scratch.attn_out).expect("attn_out aliased");
                            // Always-flash default: persistent scratch partial
                            // buffers (e185d65) eliminate per-step alloc overhead,
                            // so flash path is viable even at tiny seq_kv where
                            // it degenerates to n_splits=1 (~same as single-block).
                            // Env TQ_FLASH_THRESHOLD overrides; set large to restore
                            // single-block attention for legacy comparison.
                            let flash_threshold: usize = {
                                static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                                *C.get_or_init(|| std::env::var("TQ_FLASH_THRESHOLD")
                                    .ok().and_then(|v| v.parse().ok()).unwrap_or(0))
                            };
                            let use_flash = !capturing && !graph_replayed
                                && gpu_kv.seq_len > flash_threshold;

                            if use_flash {
                                // Flash decode: split KV across blocks for parallelism.
                                // Uses persistent scratch partial buffers — no per-step
                                // alloc_zeros. Sized for max n_splits at TQ_MAX_SEQ.
                                // split_size=64: at seq_kv=200, 4 splits × 28 heads = 112
                                // blocks (~80% sm_86 utilization). Env override available.
                                let actual_seq = gpu_kv.seq_len;
                                let split_size: usize = {
                                    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                                    *C.get_or_init(|| std::env::var("TQ_FLASH_SPLIT")
                                        .ok().and_then(|v| v.parse().ok()).unwrap_or(32))
                                };
                                let n_splits = (actual_seq + split_size - 1) / split_size;
                                debug_assert!(n_splits <= scratch.flash_max_splits,
                                    "flash n_splits {} exceeds scratch capacity {}",
                                    n_splits, scratch.flash_max_splits);
                                let scale = 1.0 / (scratch.head_dim as f32).sqrt();

                                let partial_o_mut = Arc::get_mut(&mut scratch.flash_partial_o)
                                    .expect("flash_partial_o aliased");
                                let partial_max_mut = Arc::get_mut(&mut scratch.flash_partial_max)
                                    .expect("flash_partial_max aliased");
                                let partial_sum_mut = Arc::get_mut(&mut scratch.flash_partial_sum)
                                    .expect("flash_partial_sum aliased");

                                crate::cuda::kernels::flash_decode_partial(
                                    reg, &*scratch.q_buf, &*gpu_kv.k_buf, &*gpu_kv.v_buf,
                                    partial_o_mut, partial_max_mut, partial_sum_mut,
                                    1, scratch.n_head, scratch.n_kv_head,
                                    actual_seq, scratch.head_dim, scale, split_size,
                                    gpu_kv.max_seq,
                                    0, // window_size: 0 = global (TODO: per-layer for Gemma 2)
                                ).map_err(|e| TqError::Msg(format!("flash_decode_partial: {}", e)))?;

                                crate::cuda::kernels::flash_decode_reduce(
                                    reg,
                                    &*scratch.flash_partial_o,
                                    &*scratch.flash_partial_max,
                                    &*scratch.flash_partial_sum,
                                    attn_mut,
                                    scratch.n_head, n_splits, scratch.head_dim, 1,
                                ).map_err(|e| TqError::Msg(format!("flash_decode_reduce: {}", e)))?;
                                Ok::<(), TqError>(())
                            } else {
                                // Graph-safe single-block attention
                                let attn_extra = if capturing { 1i32 } else { 0 };
                                let scale_f = 1.0 / (scratch.head_dim as f32).sqrt();
                                #[cfg(feature = "sparse-v")]
                                let sparse_v_threshold: Option<f32> = {
                                    static C: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
                                    *C.get_or_init(|| {
                                        std::env::var("TQ_SPARSE_V").ok()
                                            .and_then(|v| v.parse::<f32>().ok())
                                            .filter(|&t| t > 0.0)
                                    })
                                };
                                #[cfg(not(feature = "sparse-v"))]
                                let sparse_v_threshold: Option<f32> = None;

                                if let Some(thr) = sparse_v_threshold {
                                    #[cfg(feature = "sparse-v")]
                                    crate::cuda::kernels::gqa_decode_attention_graph_sparse_v(
                                        reg, &*scratch.q_buf, &*gpu_kv.k_buf, &*gpu_kv.v_buf,
                                        attn_mut, &gpu_kv.valid_len_gpu,
                                        scratch.n_head, scratch.n_kv_head,
                                        gpu_kv.max_seq, scratch.head_dim,
                                        scale_f, attn_extra,
                                        0, // window_size: 0 = global
                                        thr,
                                    ).map_err(|e| TqError::Msg(format!("gqa_decode_sparse_v: {}", e)))?;
                                    #[cfg(not(feature = "sparse-v"))]
                                    { let _ = thr; unreachable!(); }
                                } else {
                                    crate::cuda::kernels::gqa_decode_attention_graph(
                                        reg, &*scratch.q_buf, &*gpu_kv.k_buf, &*gpu_kv.v_buf,
                                        attn_mut, &gpu_kv.valid_len_gpu,
                                        scratch.n_head, scratch.n_kv_head,
                                        gpu_kv.max_seq, scratch.head_dim,
                                        scale_f, attn_extra,
                                        0, // window_size: 0 = global (TODO: per-layer sliding window for Gemma 2)
                                    ).map_err(|e| TqError::Msg(format!("gqa_decode_attn: {}", e)))?;
                                }
                                Ok(())
                            }
                        })?;

                        // 4. Wo projection
                        ptime!("wo_matvec", {
                            let (wo_gpu, wo_out_feat, wo_in_feat) = layer.attention_wo.q4k_gpu_data().unwrap();
                            let wo_mut = Arc::get_mut(&mut scratch.wo_out).expect("wo_out aliased");
                            crate::cuda::kernels::q4km_matvec(
                                reg, wo_gpu, &*scratch.attn_out, wo_mut,
                                wo_out_feat, wo_in_feat,
                            ).map_err(|e| TqError::Msg(format!("scratch Wo matvec: {}", e)))
                        })?;

                        Tensor::from_cuda_arc(Arc::clone(&scratch.wo_out),
                            vec![1, 1, hidden_dim], scratch.stream.clone())
                    } else {
                        // Fallback: create tensor views, use forward_attn
                        let (q_t, k_t, v_t) = if let Some(qkv) = qkv_tensors {
                            qkv
                        } else {
                            // QKV in scratch but can't use ultra-fused → create views now
                            let scratch = decode_scratch.as_ref().unwrap();
                            let q_out = scratch.q_out;
                            let k_out = scratch.k_out;
                            let v_out = scratch.v_out;
                            (
                                Tensor::from_cuda_arc(Arc::clone(&scratch.q_buf), vec![1, 1, q_out], scratch.stream.clone()),
                                Tensor::from_cuda_arc(Arc::clone(&scratch.k_buf), vec![1, 1, k_out], scratch.stream.clone()),
                                Tensor::from_cuda_arc(Arc::clone(&scratch.v_buf), vec![1, 1, v_out], scratch.stream.clone()),
                            )
                        };
                        layer.forward_attn(&x, Some((q_t, k_t, v_t)), mask, index_pos, backend)?
                    };

                    // Phase 3: extract MLP GPU data + launch kernels 2+3 (scoped borrow)
                    let fused_mlp_result: Result<Tensor> = (|| {
                        let _mlp_data = if let MlpOrMoe::Mlp(mlp) = &layer.mlp_or_moe {
                            match (
                                mlp.feed_forward_w1.q4k_gpu_data(),
                                mlp.feed_forward_w3.q4k_gpu_data(),
                                mlp.feed_forward_w2.q4k_gpu_data(),
                            ) {
                                (Some((g, idim, _)), Some((u, _, _)), Some((d, _, _))) =>
                                    Some((g, u, d, idim)),
                                _ => None,
                            }
                        } else { None };
                        // Gate+up must be Q4K for fused gateup kernel.
                        // Down can be Q6K — handle separately.
                        let gateup_data = if let MlpOrMoe::Mlp(mlp) = &layer.mlp_or_moe {
                            match (mlp.feed_forward_w1.q4k_gpu_data(), mlp.feed_forward_w3.q4k_gpu_data()) {
                                (Some((g, idim, _)), Some((u, _, _))) => Some((g, u, idim)),
                                _ => None,
                            }
                        } else { None };
                        let (wgate_gpu, wup_gpu, intermediate_dim) =
                            gateup_data.ok_or_else(|| TqError::Msg("fused gateup: not Q4K".into()))?;

                        let reg = crate::cuda::kernels::global_registry().unwrap();
                        let stream = &reg.stream;
                        let _enter = layer.span_mlp.enter();
                        let attn_f32 = attn.to_dtype(DType::F32)?;
                        let residual_f32 = x.to_dtype(DType::F32)?;

                        // Kernel 2: norm + gate/up + silu*mul
                        // Kernel 3: down projection + residual add
                        if let Some(ref mut scratch) = decode_scratch {
                            let ci = layer_idx & 1;
                            let pdet = profiling && layer_idx == 0;

                            // 1. combined = residual + attn
                            let timer = if pdet {
                                crate::cuda::event_timer::EventTimer::new().ok()
                                    .filter(|t| t.start(&reg.stream).is_ok())
                            } else { None };
                            {
                                let comb = Arc::get_mut(&mut scratch.combined_bufs[ci])
                                    .expect("scratch combined ping-pong aliased");
                                crate::cuda::kernels::add(
                                    reg, residual_f32.cuda_data(), attn_f32.cuda_data(),
                                    comb, hidden_dim,
                                ).map_err(|e| TqError::Msg(format!("scratch add: {}", e)))?;
                            }
                            if let Some(t) = timer {
                                let _ = t.stop(&reg.stream);
                                eprintln!("[kernel] {:>12}: {:.1}μs", "res+attn", t.elapsed_us().unwrap_or(0.0));
                            }

                            // 2. fused gateup+silu
                            let timer = if pdet {
                                crate::cuda::event_timer::EventTimer::new().ok()
                                    .filter(|t| t.start(&reg.stream).is_ok())
                            } else { None };
                            {
                                let inter = Arc::get_mut(&mut scratch.intermediate_buf)
                                    .expect("scratch intermediate_buf aliased");
                                crate::cuda::kernels::fused_addnorm_q4km_gateup_silu(
                                    reg, &*scratch.combined_bufs[ci], layer.ffn_norm.weight.cuda_data(),
                                    wgate_gpu, wup_gpu,
                                    inter,
                                    hidden_dim, intermediate_dim,
                                    layer.ffn_norm.eps as f32,
                                ).map_err(|e| TqError::Msg(format!("fused gateup: {}", e)))?;
                            }
                            if let Some(t) = timer {
                                let _ = t.stop(&reg.stream);
                                eprintln!("[kernel] {:>12}: {:.1}μs", "gateup", t.elapsed_us().unwrap_or(0.0));
                            }

                            // 3. down projection + residual add
                            let timer = if pdet {
                                crate::cuda::event_timer::EventTimer::new().ok()
                                    .filter(|t| t.start(&reg.stream).is_ok())
                            } else { None };
                            {
                                let wdown = if let MlpOrMoe::Mlp(mlp) = &layer.mlp_or_moe {
                                    &mlp.feed_forward_w2
                                } else { unreachable!() };
                                let comb = Arc::get_mut(&mut scratch.combined_bufs[ci])
                                    .expect("scratch combined ping-pong aliased");
                                if let Some((wd_gpu, _, _)) = wdown.q4k_gpu_data() {
                                    // Q4K: fused down + residual add
                                    crate::cuda::kernels::fused_q4km_down_residual(
                                        reg, wd_gpu, &*scratch.intermediate_buf, comb,
                                        hidden_dim, intermediate_dim,
                                    ).map_err(|e| TqError::Msg(format!("fused down+res: {}", e)))?;
                                } else if let qmm::QMatMul::Quantized(qw) = &wdown.inner {
                                    // Q6K: native fused dequant matvec (4.9x less bandwidth)
                                    let w_raw = qw.gpu_cache_or_upload(&reg.stream);
                                    let wo_tmp = Arc::get_mut(&mut scratch.attn_out)
                                        .expect("attn_out temp aliased");
                                    crate::cuda::kernels::q6k_matvec(
                                        reg, w_raw, &*scratch.intermediate_buf, wo_tmp,
                                        qw.out_features(), qw.in_features(),
                                    ).map_err(|e| TqError::Msg(format!("down f32: {}", e)))?;
                                    // residual += down_out
                                    crate::cuda::kernels::bias_add_inplace(
                                        reg, comb, &*scratch.attn_out, hidden_dim,
                                    ).map_err(|e| TqError::Msg(format!("down+res add: {}", e)))?;
                                } else {
                                    return Err(TqError::Msg("down weight: unsupported".into()));
                                }
                            }
                            if let Some(t) = timer {
                                let _ = t.stop(&reg.stream);
                                eprintln!("[kernel] {:>12}: {:.1}μs", "down+res", t.elapsed_us().unwrap_or(0.0));
                            }

                            return Ok(Tensor::from_cuda_arc(Arc::clone(&scratch.combined_bufs[ci]),
                                vec![1, 1, hidden_dim], scratch.stream.clone()));
                        }
                        // Fallback: allocate combined + intermediate (no scratch)
                        let mut combined = (residual_f32 + attn_f32)?;
                        let mut intermediate: cudarc::driver::CudaSlice<f32> =
                            stream.alloc_zeros(intermediate_dim)
                                .map_err(|e| TqError::Msg(format!("fused MLP alloc: {}", e)))?;
                        crate::cuda::kernels::fused_addnorm_q4km_gateup_silu(
                            reg, combined.cuda_data(), layer.ffn_norm.weight.cuda_data(),
                            wgate_gpu, wup_gpu,
                            &mut intermediate,
                            hidden_dim, intermediate_dim,
                            layer.ffn_norm.eps as f32,
                        ).map_err(|e| TqError::Msg(format!("fused gateup: {}", e)))?;
                        // Down projection: Q4K fused or Q6K fallback
                        let wdown = if let MlpOrMoe::Mlp(mlp) = &layer.mlp_or_moe {
                            &mlp.feed_forward_w2
                        } else { unreachable!() };
                        if let Some((wd_gpu, _, _)) = wdown.q4k_gpu_data() {
                            crate::cuda::kernels::fused_q4km_down_residual(
                                reg, wd_gpu, &intermediate, combined.cuda_data_mut(),
                                hidden_dim, intermediate_dim,
                            ).map_err(|e| TqError::Msg(format!("fused down+res: {}", e)))?;
                        } else {
                            // Q6K: separate matvec + add
                            let down_out = wdown.forward(
                                &Tensor::from_cuda(intermediate, vec![1, 1, intermediate_dim], stream.clone()),
                                backend,
                            )?;
                            combined = (combined + &down_out)?;
                        }
                        Ok(combined)
                    })(); // MLP borrows released

                    match fused_mlp_result {
                        Ok(result) => {
                            *layer_in = result;
                            // Debug checkpoint: fused path output
                            if gpu_debug {
                                if let Ok(data) = layer_in.to_vec1() {
                                    let n = data.len();
                                    let sum: f64 = data.iter().map(|&v| v as f64).sum();
                                    let mean = sum / n as f64;
                                    let l2: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                                    let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
                                        |(mn, mx), &v| (mn.min(v), mx.max(v)));
                                    eprintln!("[gpu-debug] L{} fused-out: mean={:.6} l2={:.4} min={:.6} max={:.6} first5={:?}",
                                        layer_idx, mean, l2, mn, mx, &data[..5.min(n)]);
                                }
                            }
                            return Ok(true); // fused path completed
                        }
                        Err(_) => {
                            // Fused MLP failed (e.g., Q6K weights).
                            // Attention already computed; compute residual + norm + MLP below.
                            let _enter = layer.span_mlp.enter();
                            let attn_f32 = attn.to_dtype(DType::F32)?;
                            let residual_f32 = x.to_dtype(DType::F32)?;
                            #[cfg(feature = "cuda")]
                            let (mlp_in, residual_owned) = if attn_f32.is_cuda() {
                                attn_f32.fused_add_rms_norm_gpu(
                                    &residual_f32, &layer.ffn_norm.weight, layer.ffn_norm.eps as f32,
                                )?
                            } else {
                                let shape = attn_f32.shape().to_vec();
                                let hidden = *shape.last().unwrap();
                                let n_tokens = attn_f32.elem_count() / hidden;
                                let (normed, new_res) = backend.fused_add_rms_norm(
                                    attn_f32.as_slice(), residual_f32.as_slice(),
                                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                                    n_tokens, hidden,
                                );
                                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
                            };
                            #[cfg(not(feature = "cuda"))]
                            let (mlp_in, residual_owned) = {
                                let shape = attn_f32.shape().to_vec();
                                let hidden = *shape.last().unwrap();
                                let n_tokens = attn_f32.elem_count() / hidden;
                                let (normed, new_res) = backend.fused_add_rms_norm(
                                    attn_f32.as_slice(), residual_f32.as_slice(),
                                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                                    n_tokens, hidden,
                                );
                                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
                            };
                            let mlp_out = layer.mlp_or_moe.forward(&mlp_in, backend)?;
                            *layer_in = (mlp_out + &residual_owned)?;
                            // Debug checkpoint: fused-QKV + fallback-MLP output
                            if gpu_debug {
                                if let Ok(data) = layer_in.to_vec1() {
                                    let n = data.len();
                                    let sum: f64 = data.iter().map(|&v| v as f64).sum();
                                    let mean = sum / n as f64;
                                    let l2: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                                    let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
                                        |(mn, mx), &v| (mn.min(v), mx.max(v)));
                                    eprintln!("[gpu-debug] L{} fused-qkv+fb-mlp: mean={:.6} l2={:.4} min={:.6} max={:.6} first5={:?}",
                                        layer_idx, mean, l2, mn, mx, &data[..5.min(n)]);
                                }
                            }
                            return Ok(true);
                        }
                    }
                }

    // fused_qkv = None means no Q4K weights — caller runs fallback.
    Ok(false)
}
impl GenericTurboModel {
    /// Load from a GGUF file (thin wrapper over `build`).
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: GgufContent,
        reader: &mut R,
        device: &Device,
        tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        let src = GgufSource::new(&ct, reader);
        Self::build(&src, device, tq_config)
    }

    /// Public entry point: construct from any `WeightSource`.
    /// Used by engine.rs to switch between mmap/non-mmap GGUF paths.
    pub fn build_from_source<WS: WeightSource>(
        src: &WS,
        device: &Device,
        tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        Self::build(src, device, tq_config)
    }

    /// Format-agnostic model builder. Consumes a `WeightSource`
    /// (GGUF or safetensors) and constructs a `GenericTurboModel`.
    pub(crate) fn build<WS: WeightSource>(
        src: &WS,
        device: &Device,
        tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        let md_get = |s: &str| match src.metadata(s) {
            None => bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Read architecture from GGUF metadata
        let arch = md_get("general.architecture")
            .and_then(|v| v.to_string_val().map_err(|e| TqError::Msg(format!("{e:?}"))))
            .map(|s| s.clone())
            .unwrap_or_else(|_| "llama".to_string());

        // Read model parameters using architecture prefix
        let head_count = md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let head_count_kv = md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let block_count = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let embedding_length = md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;

        // Apply model-specific kernel dispatch preset before any kernel OnceLock
        // initializes on first inference. Safe: weight loading & QWeight construction
        // don't trigger matvec dispatch — that happens on first forward pass.
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            let (sm_major, sm_minor) = crate::cuda::device::compute_capability(0);
            let _ = crate::autocalib::apply_preset(
                &arch,
                embedding_length,
                sm_major as u8,
                sm_minor as u8,
            );
        }
        let rms_norm_eps = md_get(&format!("{arch}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let rope_freq_base = md_get(&format!("{arch}.rope.freq_base"))
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let context_length = md_get(&format!("{arch}.context_length"))
            .and_then(|m| m.to_u32())
            .unwrap_or(MAX_SEQ_LEN as u32) as usize;

        // MoE params (optional, defaults to 0 = dense model)
        let n_expert = md_get(&format!("{arch}.expert_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get(&format!("{arch}.expert_used_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;

        // head_dim: prefer explicit GGUF metadata (Gemma2 has head_dim != embedding_length/head_count)
        let head_dim = md_get(&format!("{arch}.attention.key_length"))
            .and_then(|m| m.to_u32())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);

        // Hadamard transform requires power-of-2 dimensions.
        // Models with non-power-of-2 head_dim (e.g., Phi-3.5 head_dim=96) are NOT supported
        // for TQ compression because zero-padding degrades quality significantly:
        // padding 96→128 adds 33% zeros that dilute signal after Hadamard rotation.
        let padded_head_dim = head_dim;
        if !head_dim.is_power_of_two() {
            bail!(
                "TurboQuant requires power-of-2 head_dim, but this model has head_dim={}. \
                 Models with non-standard head dimensions (Phi-3.5, etc.) are not supported \
                 for KV compression. Run without --turbo-quant for these models.",
                head_dim,
            );
        }

        // Gemma2 logit soft-capping: cap * tanh(logits / cap)
        let attn_logit_softcap = md_get(&format!("{arch}.attn_logit_softcapping"))
            .and_then(|m| m.to_f32()).ok();
        let final_logit_softcap = md_get(&format!("{arch}.final_logit_softcapping"))
            .and_then(|m| m.to_f32()).ok();
        if attn_logit_softcap.is_some() || final_logit_softcap.is_some() {
            eprintln!("  Logit soft-capping: attn={:?} final={:?}", attn_logit_softcap, final_logit_softcap);
        }

        // RoPE dimension: some models (llama) specify it explicitly, others use head_dim
        let rope_dim = md_get(&format!("{arch}.rope.dimension_count"))
            .and_then(|m| m.to_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        // Gemma 2 attention scale: Gemma 2 uses `1/sqrt(query_pre_attn_scalar)`
        // instead of the default `1/sqrt(head_dim)`. For Gemma-2-9B this is
        // 224 (vs head_dim=256), for Gemma-2-27B it is 144.
        //
        // bartowski's Gemma 2 GGUFs don't export the metadata key, so we fall
        // back to the canonical Gemma 2 formula `hidden / n_heads` (gives
        // 3584/16=224 for 9B, 4608/32=144 for 27B). For non-Gemma archs the
        // fallback stays `head_dim` (unchanged).
        let attn_scale_scalar: usize = md_get(&format!("{arch}.attention.query_pre_attn_scalar"))
            .and_then(|m| m.to_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        // Note: HF Gemma 2 config specifies query_pre_attn_scalar = hidden/n_heads
        // (224 for 9B, 144 for 27B), but llama.cpp's GGUF conversion doesn't export
        // this key AND its runtime uses sqrt(head_dim) anyway. Empirical test on
        // Qwen2.5-7B (head_dim = hidden/n_heads so no difference) and
        // bartowski/gemma-2-9b-it-GGUF (PPL 124.3 with head_dim vs 127.2 with
        // hidden/n_heads fallback) confirms the GGUF path is calibrated for
        // head_dim. Keep head_dim fallback so we match the weights' implicit
        // scale. Future: if a GGUF ever exports the scalar, it'll be used.
        let attn_scale_denom: f64 = (attn_scale_scalar as f64).sqrt();
        if attn_scale_scalar != head_dim {
            eprintln!(
                "  Attention scale denominator: sqrt({}) = {:.4} (Gemma override, head_dim={})",
                attn_scale_scalar, attn_scale_denom, head_dim
            );
        }

        let rope_style = detect_rope_style(&arch);

        // Dump ALL gemma*-prefixed metadata keys when TQ_DUMP_META=1 is set,
        // so we can see which keys exist in bartowski's GGUF export (vs what
        // HF config.json has). This is needed for the gemma2 PPL 150k fix:
        // we need to know whether rope.freq_base for global vs sliding
        // attention is exported, whether query_pre_attn_scalar exists, and
        // the full surface of gemma2 attention metadata.
        if arch.contains("gemma") {
            let dump_all = std::env::var("TQ_DUMP_META").ok().as_deref() == Some("1");
            if dump_all {
                eprintln!("  [gguf] all {}* metadata keys:", arch);
                // Try a broad list of suspected keys to check presence.
                for key in [
                    "attention.value_length",
                    "attention.sliding_window",
                    "attention.query_pre_attn_scalar",
                    "attention.q_lora_rank",
                    "attention.kv_lora_rank",
                    "attention.attn_logit_softcapping",
                    "attention.layer_norm_rms_epsilon",
                    "rope.freq_base",
                    "rope.dimension_count",
                    "rope.scaling.type",
                    "rope.scaling.factor",
                    "attn_logit_softcapping",
                    "final_logit_softcapping",
                    "rope_freq_base_train",
                    "sliding_window",
                    "context_length",
                ] {
                    let full_key = format!("{arch}.{key}");
                    match src.metadata(&full_key) {
                        Some(val) => eprintln!("    {full_key} = {val:?}"),
                        None => eprintln!("    {full_key} = <MISSING>"),
                    }
                }
                // Also a few top-level keys that might be relevant.
                for key in ["general.name", "general.architecture",
                            "tokenizer.ggml.bos_token_id",
                            "tokenizer.ggml.eos_token_id"] {
                    match src.metadata(key) {
                        Some(val) => eprintln!("    {key} = {val:?}"),
                        None => eprintln!("    {key} = <MISSING>"),
                    }
                }
            } else {
                for key in ["attention.value_length", "attention.sliding_window",
                            "attention.query_pre_attn_scalar"] {
                    let full_key = format!("{arch}.{key}");
                    if let Some(val) = src.metadata(&full_key) {
                        eprintln!("  [gguf] {full_key} = {val:?}");
                    }
                }
            }
        }

        // Auto-detect features from GGUF tensors
        let has_bias = src.tensor("blk.0.attn_q.bias", device).is_ok();
        let has_merged_qkv = src.tensor("blk.0.attn_qkv.weight", device).is_ok();
        let has_ffn_gate = src.tensor("blk.0.ffn_gate.weight", device).is_ok();
        let has_post_attn_norm = src.tensor("blk.0.post_attention_norm.weight", device).is_ok();
        let has_post_ffn_norm = src.tensor("blk.0.post_ffw_norm.weight", device).is_ok();

        let qkv_style = if has_merged_qkv { "merged" } else { "separate" };
        let mlp_style = if n_expert > 1 {
            "moe"
        } else if has_ffn_gate {
            "gated-silu"
        } else {
            "silu-up-down"
        };

        eprintln!(
            "TurboQuant Generic [{}]: {} layers, {} heads (kv={}), head_dim={}{}, emb={}, \
             eps={:.2e}, rope_base={}, rope_dim={}, rope={:?}, bias={}, moe={}, {}-bit KV cache",
            arch, block_count, head_count, head_count_kv, head_dim,
            if padded_head_dim != head_dim { format!(" (padded={})", padded_head_dim) } else { String::new() },
            embedding_length,
            rms_norm_eps, rope_freq_base, rope_dim, rope_style, has_bias,
            if n_expert > 1 { format!("{}of{}", n_expert_used, n_expert) } else { "no".into() },
            tq_config.bits,
        );
        let protect_last = get_protect_last_layers(&tq_config);
        let skip_first = get_skip_layers(&tq_config);
        // MLP activation: Gemma/Gemma2 HF config specifies `gelu_pytorch_tanh`,
        // but empirical test (2026-04-13) shows SiLU gives better PPL on Q4K
        // quants. On gemma:9b /tmp/ppl_short.txt:
        //   SiLU → 124.3  GELU → 525.7
        // On /tmp/ppl_hello.txt:
        //   SiLU → 276    GELU → 73.3
        // Mixed signal — likely quantization-aware co-adaptation in llama.cpp's
        // Gemma 2 GGUF conversion. Stick with SiLU until we can test against
        // a BF16/F16 reference; revisit when safetensors path for gemma lands.
        let mlp_activation = GateActivation::SiLU;
        eprintln!(
            "  qkv={}, mlp={}, post_attn_norm={}, post_ffn_norm={}",
            qkv_style, mlp_style, has_post_attn_norm, has_post_ffn_norm,
        );
        if protect_last > 0 {
            eprintln!(
                "  boundary protection: first {} + last {} layers uncompressed ({} compressed)",
                skip_first, protect_last, block_count.saturating_sub(skip_first + protect_last),
            );
        }

        // Pre-compute shared state
        let signs = tq_kv::hadamard::generate_signs(padded_head_dim, tq_config.rotation_seed);
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Embeddings: lazy dequant on GPU (saves ~2 GB), full dequant on CPU
        let tok_embeddings_q = src.tensor("token_embd.weight", device)?;
        let emb_dtype = tok_embeddings_q.dtype;
        let emb_shape = tok_embeddings_q.shape;
        #[cfg(feature = "cuda")]
        let tok_embeddings_lazy = crate::cuda::kernels::global_registry().is_some();
        #[cfg(not(feature = "cuda"))]
        let tok_embeddings_lazy = false;
        // Clone raw_data since tok_embeddings_q may be reused for tie_word_embeddings
        let emb_raw_data = tok_embeddings_q.raw_data.clone();
        let (emb_raw, emb_full) = if tok_embeddings_lazy {
            (Some(emb_raw_data), None)
        } else {
            let dequant = crate::quant::dequantize(emb_raw_data.as_slice(), emb_dtype,
                emb_shape.0 * emb_shape.1);
            let tensor = Tensor::from_vec(dequant, vec![emb_shape.0, emb_shape.1], device)?;
            (None, Some(tensor))
        };
        let norm = {
            let n = RmsNorm::from_qtensor(
                src.tensor("output_norm.weight", device)?, rms_norm_eps, device,
            )?;
            if arch.contains("gemma") { n.with_add_unit() } else { n }
        };
        // Detect tie_word_embeddings: if output.weight is missing, reuse token embeddings
        let output = match src.tensor("output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => {
                eprintln!("  (tie_word_embeddings: reusing token_embd.weight for output)");
                eprintln!("  [debug] emb shape=({}, {}), dtype={:?}",
                    tok_embeddings_q.shape.0, tok_embeddings_q.shape.1, tok_embeddings_q.dtype);
                tok_embeddings_q
            }
        };

        // Load layers
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            // Attention weights: merged QKV (Phi-3.5) or separate (most models)
            let qkv = if has_merged_qkv {
                let wqkv = src.tensor(&format!("{prefix}.attn_qkv.weight"), device)?;
                QkvWeights::Merged { wqkv: QMatMul::from_qtensor(wqkv)? }
            } else {
                let wq = src.tensor(&format!("{prefix}.attn_q.weight"), device)?;
                let wk = src.tensor(&format!("{prefix}.attn_k.weight"), device)?;
                let wv = src.tensor(&format!("{prefix}.attn_v.weight"), device)?;
                if layer_idx == 0 {
                    eprintln!("  [debug] L0 wq=({},{}) wk=({},{}) wv=({},{})",
                        wq.shape.0, wq.shape.1, wk.shape.0, wk.shape.1, wv.shape.0, wv.shape.1);
                }
                QkvWeights::Separate {
                    wq: QMatMul::from_qtensor(wq)?,
                    wk: QMatMul::from_qtensor(wk)?,
                    wv: QMatMul::from_qtensor(wv)?,
                }
            };
            let attention_wo = src.tensor(&format!("{prefix}.attn_output.weight"), device)?;

            // Optional biases (Qwen2 has them, Llama/Phi/Gemma don't)
            let attention_bq = if has_bias {
                Some(src.tensor(&format!("{prefix}.attn_q.bias"), device)?.dequantize_to_device(device)?)
            } else {
                None
            };
            let attention_bk = if has_bias {
                Some(src.tensor(&format!("{prefix}.attn_k.bias"), device)?.dequantize_to_device(device)?)
            } else {
                None
            };
            let attention_bv = if has_bias {
                Some(src.tensor(&format!("{prefix}.attn_v.bias"), device)?.dequantize_to_device(device)?)
            } else {
                None
            };

            // MLP: 3-gate (most models), 2-gate up/down (Phi-3.5), or MoE
            let mlp_or_moe = if n_expert > 1 {
                let gate_inp = src.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let w1 = src.tensor(&format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let w2 = src.tensor(&format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let w3 = src.tensor(&format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(w3)?,
                        activation: mlp_activation,
                    });
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(gate_inp)?,
                    experts,
                }
            } else if has_ffn_gate {
                let w1 = src.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
                let w2 = src.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let w3 = src.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(w3)?,
                    activation: mlp_activation,
                })
            } else {
                // Phi-style: only ffn_up and ffn_down (no ffn_gate)
                let up = src.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                let down = src.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                MlpOrMoe::UpDown(MlpUpDown {
                    ffn_up: QMatMul::from_qtensor(up)?,
                    ffn_down: QMatMul::from_qtensor(down)?,
                })
            };

            let attention_norm = src.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = src.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;

            // Optional post-norms (Gemma2)
            let post_attention_norm = if has_post_attn_norm {
                let t = src.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?;
                Some(RmsNorm::from_qtensor(t, rms_norm_eps, device)?)
            } else {
                None
            };
            let post_ffn_norm = if has_post_ffn_norm {
                let t = src.tensor(&format!("{prefix}.post_ffw_norm.weight"), device)?;
                Some(RmsNorm::from_qtensor(t, rms_norm_eps, device)?)
            } else {
                None
            };

            layers.push(LayerWeights {
                qkv,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_bq,
                attention_bk,
                attention_bv,
                attention_norm: {
                    let n = RmsNorm::from_qtensor(attention_norm, rms_norm_eps, device)?;
                    if arch.contains("gemma") { n.with_add_unit() } else { n }
                },
                post_attention_norm: post_attention_norm.map(|n| {
                    if arch.contains("gemma") { n.with_add_unit() } else { n }
                }),
                mlp_or_moe,
                ffn_norm: {
                    let n = RmsNorm::from_qtensor(ffn_norm, rms_norm_eps, device)?;
                    if arch.contains("gemma") { n.with_add_unit() } else { n }
                },
                post_ffn_norm: post_ffn_norm.map(|n| {
                    if arch.contains("gemma") { n.with_add_unit() } else { n }
                }),
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                padded_head_dim,
                rope_dim,
                rope_style,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                attn_logit_softcap,
                attn_scale_denom,
                layer_idx,
                n_layers: block_count,
                kv_cache: None,
                #[cfg(feature = "cuda")]
                gpu_kv_cache: None,
                #[cfg(feature = "cuda")]
                gpu_tq_cache: None,
                kv_compressed: None,
                tq_config: tq_config.clone(),
                signs: signs.clone(),
                #[cfg(feature = "cuda")]
                signs_gpu: None,
                #[cfg(feature = "cuda")]
                boundaries_gpu: None,
                #[cfg(feature = "cuda")]
                centroids_gpu: None,
                #[cfg(feature = "cuda")]
                channel_sigma_gpu: None,
                #[cfg(feature = "cuda")]
                signs_expanded_gpu: None,
                #[cfg(feature = "cuda")]
                sink_k_gpu: None,
                #[cfg(feature = "cuda")]
                sink_v_gpu: None,
                offload_cache: None,
                #[cfg(feature = "cuda")]
                gpu_cold_cache: None::<Box<super::kv_cache::GpuColdKv>>,
                smooth_k_scales: compute_smooth_scales(&tq_config, head_dim, device, false),
                smooth_q_scales: compute_smooth_scales(&tq_config, head_dim, device, true),
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
            });
        }

        let backend = crate::backend::create_backend();

        let mut model = Self {
            tok_embeddings: if let Some(raw) = emb_raw {
                Embedding::new_lazy(raw, emb_dtype, embedding_length)
            } else {
                Embedding::new(emb_full.unwrap(), embedding_length)
            },
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            masks_rect: HashMap::new(),
            embed_scale: if arch.contains("gemma") {
                Some((embedding_length as f32).sqrt())
            } else { None },
            final_logit_softcap,
            backend,
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
            #[cfg(feature = "cuda")]
            // CUDA Graph is disabled automatically when:
            //   1. TQ_LAYER_SWAP is active — swap prefetch changes device
            //      addresses between iterations, invalidating replay.
            //   2. TurboQuant KV compression is active — the compressed-KV
            //      attention path contains CPU-side ops (codebook lookups,
            //      group reductions, TriAttention eviction bookkeeping) that
            //      are not capture-safe. Running capture across these layers
            //      produces either a silent hang or truncates the bench so
            //      only the Standard row completes. The hybrid boundary in
            //      `maybe_end_hybrid_capture` was meant to cover this but is
            //      brittle in practice; treat TQ + graph as mutually exclusive
            //      until the compressed-KV path is fully GPU-resident.
            //
            // Default: ON for Std path (safe-gate verified, +4-5% tok/s).
            // Auto-disabled when:
            //   - TQ KV compression active (capture-unsafe)
            //   - layer-swap active (prefetch incompatible)
            //   - dp4a kernel path active (TQ_GATEUP/Q4KM/DOWN/QKV=dp4a) — dp4a
            //     variants use a process-lifetime q8_1 scratch buffer whose
            //     address can differ between capture and replay, producing
            //     CUDA_ERROR_ILLEGAL_ADDRESS on the first graph replay.
            // Opt-out:    TQ_GRAPH=0 | off | false
            // Opt-in TQ:  TQ_GRAPH=force (debug; expected unstable with TQ)
            graph_manager: crate::cuda::graph::CudaGraphManager::new({
                let graph_var = std::env::var("TQ_GRAPH").ok();
                let graph_off_explicit = matches!(
                    graph_var.as_deref(),
                    Some("0") | Some("off") | Some("false")
                );
                let graph_explicit_on = matches!(graph_var.as_deref(), Some("1") | Some("force"));
                let graph_forced = matches!(graph_var.as_deref(), Some("force"));
                // Default ON unless user explicitly opts out with TQ_GRAPH=0.
                let graph_wanted = !graph_off_explicit;
                let swap_active = matches!(
                    std::env::var("TQ_LAYER_SWAP").ok().as_deref(),
                    Some("1") | Some("force")
                );
                let tq_active = tq_config.skip_layers != Some(999);
                // dp4a dispatch auto-disables graph as a PERFORMANCE heuristic,
                // not a safety gate. Isolation test 2026-04-17 on RTX 3080 sm_86:
                // dp4a + graph runs clean (no CUDA_ERROR_ILLEGAL_ADDRESS on Std
                // path) but regresses Std -7% (67.5 → 62.8 tok/s avg, 2 runs).
                // Why: dp4a kernels already have low per-launch overhead that
                // amortizes well in eager mode; graph instantiation + replay
                // overhead exceeds the launch savings on this path.
                // The prior "not graph-safe" comment was wrong — pool is
                // pre-allocated to Q8_1_POOL_CAPACITY (64 KB) at registry init
                // and never grows for Qwen2 7B / Llama3 8B (max 21 KB needed).
                let path_uses_dp4a = |key: &str| -> bool {
                    match std::env::var(key).ok().as_deref() {
                        Some("dp4a") | Some("dp4a_v2") | Some("dp4a_v3") => true,
                        Some(_) => false,
                        None    => true,  // current defaults are all dp4a-based
                    }
                };
                let dp4a_active =
                    path_uses_dp4a("TQ_GATEUP")
                 || path_uses_dp4a("TQ_Q4KM")
                 || path_uses_dp4a("TQ_DOWN")
                 || path_uses_dp4a("TQ_QKV")
                 || path_uses_dp4a("TQ_Q6K");
                if graph_wanted && swap_active {
                    if graph_explicit_on {
                        eprintln!("[cuda] TQ_GRAPH disabled (TQ_LAYER_SWAP active)");
                    }
                    false
                } else if graph_wanted && tq_active && !graph_forced {
                    if graph_explicit_on {
                        eprintln!("[cuda] TQ_GRAPH disabled (TurboQuant KV active — compressed path not capture-safe). Set TQ_GRAPH=force to override.");
                    }
                    false
                } else if graph_wanted && dp4a_active && !graph_forced {
                    if graph_explicit_on {
                        eprintln!("[cuda] TQ_GRAPH disabled (dp4a path faster eager than with graph; -7% Std measured). Set TQ_GRAPH=force to override.");
                    }
                    false
                } else {
                    graph_wanted
                }
            }),
            #[cfg(feature = "cuda")]
            graph_input_buffer: None,
            #[cfg(feature = "cuda")]
            rope_pos_gpu: None,
            #[cfg(feature = "cuda")]
            decode_scratch: None,
            #[cfg(feature = "cuda")]
            layer_swap: None,
            #[cfg(feature = "cuda")]
            layer_qweight_ptrs: Vec::new(),
        };

        // Allocate DecodeScratch if CUDA is available and model has separate Q4K QKV + standard MLP.
        #[cfg(feature = "cuda")]
        if crate::cuda::kernels::global_registry().is_some() && !model.layers.is_empty() {
            let layer0 = &model.layers[0];
            let q_out = layer0.n_head * layer0.head_dim;
            let kv_out = layer0.n_kv_head * layer0.head_dim;

            // Extract intermediate_dim from MLP gate weight
            let intermediate_dim = if let MlpOrMoe::Mlp(mlp) = &layer0.mlp_or_moe {
                match &mlp.feed_forward_w1.inner {
                    qmm::QMatMul::Quantized(qw) => Some(qw.out_features()),
                    _ => None,
                }
            } else {
                None
            };

            if let Some(idim) = intermediate_dim {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    match DecodeScratch::new(
                        reg.stream.clone(), q_out, kv_out, kv_out, idim, embedding_length,
                        head_count, head_count_kv, head_dim,
                    ) {
                        Ok(scratch) => model.decode_scratch = Some(scratch),
                        Err(e) => eprintln!("[DecodeScratch] alloc failed: {} — decode will use per-op allocation", e),
                    }
                }
            }
        }

        // Pre-warm weight caches: dequant on CPU + upload to GPU.
        // - TQ_NO_WARMUP=1: skip (low-memory systems).
        // - TQ_LAYER_SWAP=1|force: skip (LayerSwapManager manages QWeight GPU
        //   residency via pin/prefetch; warmup would double-upload and OOM).
        let do_warmup = std::env::var("TQ_NO_WARMUP").map(|v| v != "1").unwrap_or(true);
        let swap_active = matches!(
            std::env::var("TQ_LAYER_SWAP").ok().as_deref(),
            Some("1") | Some("force")
        );
        let warmup_qweights = do_warmup && !swap_active;
        let b = model.backend.as_ref();

        if warmup_qweights {
            eprintln!("  Pre-warming weight caches ({} layers, backend={})...", block_count, b.name());
            model.output.warmup(b);
            // Final norm weight
            b.warmup_f32(model.norm.weight.as_slice());
            for layer in &model.layers {
                layer.attention_wo.warmup(b);
                match &layer.qkv {
                    QkvWeights::Separate { wq, wk, wv } => { wq.warmup(b); wk.warmup(b); wv.warmup(b); }
                    QkvWeights::Merged { wqkv } => { wqkv.warmup(b); }
                }
                // Norm weights → GPU cache
                b.warmup_f32(layer.attention_norm.weight.as_slice());
                b.warmup_f32(layer.ffn_norm.weight.as_slice());
                if let Some(ref n) = layer.post_attention_norm {
                    b.warmup_f32(n.weight.as_slice());
                }
                if let Some(ref n) = layer.post_ffn_norm {
                    b.warmup_f32(n.weight.as_slice());
                }
                match &layer.mlp_or_moe {
                    MlpOrMoe::Mlp(mlp) => {
                        mlp.feed_forward_w1.warmup(b);
                        mlp.feed_forward_w2.warmup(b);
                        mlp.feed_forward_w3.warmup(b);
                    }
                    MlpOrMoe::UpDown(ud) => {
                        ud.ffn_up.warmup(b);
                        ud.ffn_down.warmup(b);
                    }
                    MlpOrMoe::MoE { experts, feed_forward_gate_inp, .. } => {
                        feed_forward_gate_inp.warmup(b);
                        for exp in experts { exp.feed_forward_w1.warmup(b); exp.feed_forward_w2.warmup(b); exp.feed_forward_w3.warmup(b); }
                    }
                }
            }

            // Auto-release CPU raw_data for Owned (non-mmap) QWeights now
            // that GPU caches are populated. Saves ~model-size of heap RAM
            // on non-swap runs. Opt-in via TQ_RELEASE_CPU=1.
            //
            // Why opt-in: A/B on qwen2:7b showed ~3% Standard tok/s penalty
            // (16.8 → 16.2) when release runs. Mutation invalidates some
            // pool/Arc assumptions elsewhere in the hot path that still
            // need investigation. Default off preserves tok/s; users on
            // RAM-constrained systems enable the flag to reclaim heap.
            // (Swap + mmap path already avoids the heap Vec entirely.)
            #[cfg(feature = "cuda")]
            if b.is_gpu() && std::env::var("TQ_RELEASE_CPU").ok().as_deref() == Some("1") {
                eprintln!("  Releasing CPU weight heap (TQ_RELEASE_CPU=1)...");
                for layer in &mut model.layers {
                    layer.release_qweight_cpu();
                }
                if let qmm::QMatMul::Quantized(qw) = &mut model.output.inner {
                    qw.release_cpu_after_gpu();
                }
            }
            eprintln!("  Weight caches warmed.");
        } else if swap_active {
            eprintln!("  QWeight warmup skipped (layer-swap active; manager handles GPU residency).");
        }

        // Persistent tensor upload runs ALWAYS (even with swap / no-warmup) —
        // these are small (cos/sin/norm weights/biases) and required by the
        // forward kernels regardless of QWeight residency strategy.
        #[cfg(feature = "cuda")]
        if b.is_gpu() {
            if let Ok(gpu) = model.norm.weight.to_device_auto() { model.norm.weight = gpu; }
            for layer in &mut model.layers {
                if let Ok(gpu) = layer.cos.to_device_auto() { layer.cos = gpu; }
                if let Ok(gpu) = layer.sin.to_device_auto() { layer.sin = gpu; }
                if let Ok(gpu) = layer.attention_norm.weight.to_device_auto() { layer.attention_norm.weight = gpu; }
                if let Ok(gpu) = layer.ffn_norm.weight.to_device_auto() { layer.ffn_norm.weight = gpu; }
                if let Some(ref b) = layer.attention_bq { if let Ok(gpu) = b.to_device_auto() { layer.attention_bq = Some(gpu); } }
                if let Some(ref b) = layer.attention_bk { if let Ok(gpu) = b.to_device_auto() { layer.attention_bk = Some(gpu); } }
                if let Some(ref b) = layer.attention_bv { if let Ok(gpu) = b.to_device_auto() { layer.attention_bv = Some(gpu); } }
                if let Ok(gpu) = layer.neg_inf.to_device_auto() { layer.neg_inf = gpu; }
            }
            eprintln!("  Persistent tensors uploaded to GPU.");

            // FP16 weight caches for HGEMM are initialized lazily on first large prefill.
            // Eager warmup would double VRAM usage (f32 + f16 caches).
        }

        // Note: A.2 (raw_data release after GPU upload) was attempted but caused
        // crashes in CPU fallback paths (e.g. Q6K prefill dequant). Deferred until
        // all forward paths are fully GPU-resident with no CPU fallback.

        Ok(model)
    }

    /// Causal attention mask.
    /// - Prefill (offset=0): square [seq_len, seq_len]
    /// - Continuation (offset>0): rectangular [seq_len, offset+seq_len]
    ///   where past positions (0..offset) are always valid.
    fn mask(&mut self, seq_len: usize, offset: usize, _device: &Device) -> Result<Tensor> {
        let total_len = offset + seq_len;
        let key = (seq_len, total_len);
        if let Some(mask) = self.masks_rect.get(&key) {
            return Ok(mask.clone());
        }
        // Row i (query at position offset+i) can attend to column j if j <= offset+i
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                let query_pos = offset + i;
                (0..total_len).map(move |j| if j <= query_pos { 0.0f32 } else { -1e10f32 })
            })
            .collect();
        let mut mask = Tensor::from_slice(&mask, vec![seq_len, total_len], _device)?;
        #[cfg(feature = "cuda")]
        if let Ok(gpu) = mask.to_device_auto() {
            mask = gpu;
        }
        self.masks_rect.insert(key, mask.clone());
        Ok(mask)
    }

    /// Truncate KV caches to `target_len` tokens. Used by speculative decode
    /// to roll back draft model's KV cache on rejection without full re-prefill.
    pub fn truncate_kv_cache(&mut self, target_len: usize) {
        for layer in &mut self.layers {
            if let Some((ref k, ref v)) = layer.kv_cache {
                let current_len = k.shape()[2];
                if target_len < current_len {
                    layer.kv_cache = Some((
                        k.narrow(2, 0, target_len).expect("kv truncate k"),
                        v.narrow(2, 0, target_len).expect("kv truncate v"),
                    ));
                }
            }
            #[cfg(feature = "cuda")]
            if let Some(ref mut gpu_kv) = layer.gpu_kv_cache {
                if target_len < gpu_kv.seq_len {
                    gpu_kv.seq_len = target_len;
                }
            }
        }
    }

    /// Clear all KV caches (CPU + GPU) for fresh generation.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv_cache = None;
            #[cfg(feature = "cuda")]
            { layer.gpu_kv_cache = None; }
        }
        #[cfg(feature = "cuda")]
        self.graph_manager.reset();
        self.masks.clear();
        self.masks_rect.clear();
    }

    /// Compute token embedding (with optional Gemma sqrt-hidden scale),
    /// upload to GPU for GPU-resident forward, and (when graph capture is
    /// in progress) snapshot the embedding into a dedicated long-lived
    /// buffer so the captured graph can replay against the same address.
    fn prepare_embedding(
        &mut self,
        x: &Tensor,
        backend: &dyn ComputeBackend,
        #[cfg(feature = "cuda")] capturing: bool,
        #[cfg(feature = "cuda")] recording: bool,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        // Gemma: scale embeddings by sqrt(hidden_dim).
        if let Some(scale) = self.embed_scale {
            let data = layer_in.as_slice();
            let scaled: Vec<f32> = data.iter().map(|&v| v * scale).collect();
            layer_in = Tensor::from_vec(scaled, layer_in.shape().to_vec(), layer_in.device())?;
        }

        // Phase 3: upload embedding to GPU for GPU-resident forward pass.
        // All subsequent ops auto-dispatch to GPU when tensor is CUDA.
        #[cfg(feature = "cuda")]
        if backend.is_gpu() {
            if let Ok(gpu_tensor) = layer_in.to_device_auto() {
                layer_in = gpu_tensor;
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = backend;

        // Save input buffer for graph replay. Must be a SEPARATE allocation
        // (not Arc::clone of layer_in) so graph_input_buffer always has
        // refcount=1 and Arc::get_mut succeeds on replay without cloning
        // to a new address.
        #[cfg(feature = "cuda")]
        if capturing || recording {
            if layer_in.is_cuda() {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    let src = layer_in.cuda_data();
                    let n = layer_in.elem_count();
                    if self.graph_input_buffer.is_none() {
                        if let Ok(buf) = crate::cuda::gpu_alloc_zeros_pub(&reg.stream, n) {
                            self.graph_input_buffer = Some(std::sync::Arc::new(buf));
                        }
                    }
                    if let Some(ref mut buf) = self.graph_input_buffer {
                        let dst = std::sync::Arc::get_mut(buf).expect("graph_input_buffer shared");
                        let _ = reg.stream.memcpy_dtod(src, dst);
                        layer_in = Tensor::from_cuda_arc(
                            buf.clone(), layer_in.shape().to_vec(), reg.stream.clone(),
                        );
                    }
                }
            }
        }
        Ok(layer_in)
    }

    /// Try to replay the captured layer-loop graph for single-token decode.
    /// Returns `true` if replay happened (caller skips the layer loop body
    /// for non-TQ layers but still runs norm + lm_head eagerly), `false`
    /// otherwise. Skipped above kv_len > 256 so flash-decode (split-KV)
    /// can be used on long contexts where the single-block gqa_decode in
    /// the captured graph would be too slow.
    #[cfg(feature = "cuda")]
    fn try_replay_layer_loop_graph(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        seq_len: usize,
    ) -> Result<bool> {
        let kv_len_for_graph = self.layers.first()
            .and_then(|l| l.gpu_kv_cache.as_ref())
            .map(|kv| kv.seq_len)
            .unwrap_or(0);
        if !(seq_len == 1 && self.graph_manager.is_ready(1) && kv_len_for_graph <= 256) {
            return Ok(false);
        }
        let reg = crate::cuda::kernels::global_registry().unwrap();

        // Update embedding into dedicated buffer (same GPU address as capture).
        if let Some(ref mut input_buf) = self.graph_input_buffer {
            let new_emb = self.tok_embeddings.forward(x)?;
            let emb_data = new_emb.as_slice();
            let _ = reg.stream.context().check_err();
            let buf_mut = std::sync::Arc::get_mut(input_buf)
                .expect("graph_input_buffer refcount > 1");
            let _ = reg.stream.memcpy_htod(emb_data, buf_mut);
        }

        // Update RoPE position GPU scalar.
        if let Some(ref mut rope_buf) = self.rope_pos_gpu {
            let _ = reg.stream.memcpy_htod(&[index_pos as i32], rope_buf);
        }

        // Update KV cache valid_len (pre-append position).
        for layer in &mut self.layers {
            if let Some(ref mut gpu_kv) = layer.gpu_kv_cache {
                let _ = reg.stream.memcpy_htod(&[gpu_kv.seq_len as i32], &mut gpu_kv.valid_len_gpu);
            }
        }

        // No sync needed: memcpy_htod + graph launch are stream-ordered.
        if self.graph_manager.replay(&reg.stream, 1).is_ok() {
            // No post-replay sync: TQ layers launch on same stream after graph.
            // Post-replay: increment seq_len.
            for layer in &mut self.layers {
                if let Some(ref mut gpu_kv) = layer.gpu_kv_cache {
                    gpu_kv.seq_len += 1;
                }
            }
            // DON'T return — caller falls through to norm + lm_head.
            return Ok(true);
        }
        Ok(false)
    }

    /// End the layer-loop CUDA-graph capture and immediately replay it,
    /// then update `layer_in` to point at the scratch buffer holding the
    /// final layer's output. Called from `forward` once per prefill when
    /// graph capture is in progress.
    #[cfg(feature = "cuda")]
    fn finalize_layer_loop_capture(&mut self, layer_in: &mut Tensor) {
        if !matches!(self.graph_manager.status, crate::cuda::graph::GraphStatus::Capturing) {
            return;
        }
        let Some(reg) = crate::cuda::kernels::global_registry() else { return; };

        match self.graph_manager.end_capture(&reg.stream, 1) {
            Ok(()) => {
                eprintln!("[cuda-graph] captured layer loop (scratch-based)");
                // Immediate replay — capture only RECORDS, doesn't execute.
                let _ = reg.stream.context().check_err();
                match self.graph_manager.replay(&reg.stream, 1) {
                    Ok(()) => {
                        let _ = reg.stream.synchronize();
                        eprintln!("[cuda-graph] initial replay OK");
                        // seq_len already incremented by append() in the layer loop.
                        // Do NOT increment again here. Update layer_in from
                        // scratch combined buffer.
                        if let Some(ref scratch) = self.decode_scratch {
                            let last_ci = (self.layers.len() - 1) & 1;
                            *layer_in = Tensor::from_cuda_arc(
                                Arc::clone(&scratch.combined_bufs[last_ci]),
                                vec![1, 1, scratch.hidden_dim], scratch.stream.clone(),
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("[cuda-graph] initial replay FAILED: {}", e);
                        self.graph_manager.reset();
                    }
                }
            }
            Err(e) => {
                eprintln!("[cuda-graph] end_capture failed: {}", e);
                self.graph_manager.reset();
            }
        }
    }

    /// Final norm + last-token slice + LM head + softcap. Extracted from
    /// the tail of `forward` (2026-04-13) to shrink the 1068-line monolith.
    /// Optional GPU-debug compares CPU vs GPU lm_head logits.
    fn apply_norm_and_lm_head(
        &self,
        layer_in: &Tensor,
        backend: &dyn ComputeBackend,
        seq_len: usize,
        index_pos: usize,
        gpu_debug: bool,
        profiling: bool,
        #[cfg(feature = "cuda")]
        prof_stream: Option<&std::sync::Arc<cudarc::driver::CudaStream>>,
    ) -> Result<Tensor> {
        let x = self.norm.forward(layer_in, backend)?;
        // Batched verify (seq_len > 1, index_pos > 0): keep ALL positions' logits.
        // Normal prefill / decode: only last position (next-token prediction).
        let x = if seq_len > 1 && index_pos > 0 {
            x.reshape(vec![seq_len, x.shape()[2]])?  // [seq_len, hidden]
        } else {
            x.narrow(1, seq_len - 1, 1)?.squeeze(1)?  // [1, hidden]
        };
        let _enter = self.span_output.enter();

        #[cfg(feature = "cuda")]
        let _lm_timer: Option<crate::cuda::event_timer::EventTimer> = if profiling {
            if let Some(s) = prof_stream {
                crate::cuda::event_timer::EventTimer::new().ok()
                    .filter(|t| t.start(s).is_ok())
            } else { None }
        } else { None };
        #[cfg(not(feature = "cuda"))]
        let _lm_timer: Option<()> = { let _ = profiling; None };

        // Debug: compare GPU vs CPU lm_head output.
        if gpu_debug {
            if let Ok(h) = x.to_vec1() {
                let n = h.len();
                let l2: f64 = h.iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>().sqrt();
                eprintln!("[gpu-debug] pre-lm_head: n={} l2={:.4}", n, l2);

                if let qmm::QMatMul::Quantized(ref qw) = self.output.inner {
                    let cpu_logits = backend.qmatmul(&h, qw, 1, qw.in_features(), qw.out_features());
                    let (cmn, cmx) = cpu_logits.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
                        |(mn,mx),&v| (mn.min(v), mx.max(v)));
                    let top_cpu: Vec<(usize, f32)> = {
                        let mut idx: Vec<(usize,f32)> = cpu_logits.iter().copied().enumerate().collect();
                        idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
                        idx.into_iter().take(5).collect()
                    };
                    eprintln!("[gpu-debug] CPU lm_head: n={} min={:.4} max={:.4} top5={:?}",
                        cpu_logits.len(), cmn, cmx, top_cpu);
                }
            }
        }

        let output = self.output.forward(&x, backend)?;
        let output = if let Some(cap) = self.final_logit_softcap {
            apply_softcap(&output, cap)?
        } else { output };

        #[cfg(feature = "cuda")]
        if let Some(t) = _lm_timer {
            if let Some(s) = prof_stream {
                let _ = t.stop(s);
                eprintln!("[kernel] {:>12}: {:.1}μs", "lm_head", t.elapsed_us().unwrap_or(0.0));
            }
        }

        Ok(output)
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;

        // Reset graph on new sequence (prefill).
        #[cfg(feature = "cuda")]
        if seq_len > 1 {
            self.graph_manager.reset();
        }

        // ── CUDA Graph replay (layer loop only — norm+lm_head run eagerly after) ──
        #[cfg(feature = "cuda")]
        let graph_replayed = self.try_replay_layer_loop_graph(x, index_pos, seq_len)?;

        // ── CUDA Graph capture (scratch-based, no pool recording) ──
        #[cfg(feature = "cuda")]
        let mut capturing = seq_len == 1 && self.graph_manager.should_capture(1);
        #[cfg(feature = "cuda")]
        let recording = false; // recording pass disabled for scratch-based graph
        #[cfg(feature = "cuda")]
        if capturing {
            if let Some(reg) = crate::cuda::kernels::global_registry() {
                let pos_val = index_pos as i32;
                let _ = reg.stream.context().check_err();
                if self.rope_pos_gpu.is_none() {
                    if let Ok(buf) = reg.stream.clone_htod(&[pos_val]) {
                        self.rope_pos_gpu = Some(buf);
                    }
                } else if let Some(ref mut existing) = self.rope_pos_gpu {
                    let _ = reg.stream.memcpy_htod(&[pos_val], existing);
                }
                if let Some(ref buf) = self.rope_pos_gpu {
                    super::ROPE_POS_GPU_PTR.with(|p| p.set(buf as *const _ as u64));
                }
            }
        }

        // Size-pool activation is a one-shot opt-in handled in engine.rs
        // when TQ_SIZE_POOL=1 is set. The legacy decode-arena pool path
        // (hardcoded off for ~3 sprints) was removed 2026-04-13.

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        // Clone the span so its enter-guard doesn't pin a self borrow
        // (we need &mut self downstream for finalize_layer_loop_capture etc.).
        let span = self.span.clone();
        let _enter = span.enter();
        let backend = self.backend.clone();
        let backend = backend.as_ref();
        let mut layer_in = self.prepare_embedding(
            x, backend,
            #[cfg(feature = "cuda")] capturing,
            #[cfg(feature = "cuda")] recording,
        )?;

        // Begin graph capture AFTER embedding upload (H2D copy must be outside capture).
        #[cfg(feature = "cuda")]
        if capturing {
            // Restore pool buffers and set Pooled mode: kernel launches use
            // pre-allocated (Recording-pass) pointers that persist across graph replay.
            // Scratch-based capture: no pool needed, all pointers stable.
            if let Some(reg) = crate::cuda::kernels::global_registry() {
                match self.graph_manager.begin_capture(&reg.stream) {
                    Ok(()) => eprintln!("[cuda-graph] capture started (scratch-based)"),
                    Err(e) => eprintln!("[cuda-graph] begin_capture failed: {}", e),
                }
            }
        }

        // GPU vs CPU debug: print tensor stats at each layer boundary for divergence analysis.
        // Run with TQ_GPU_DEBUG=1, compare output between GPU and CPU runs.
        // Cache env vars ONCE before layer loop (avoid ~168 syscalls/token).
        let gpu_debug = {
            static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *C.get_or_init(|| std::env::var("TQ_GPU_DEBUG").map(|v| v == "1").unwrap_or(false))
        };
        let profiling = {
            static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *C.get_or_init(|| std::env::var("TQ_PROFILE").map(|v| v == "1").unwrap_or(false))
        };
        #[cfg(feature = "cuda")]
        let prof_stream = if profiling { crate::cuda::kernels::global_registry().map(|r| r.stream.clone()) } else { None };
        #[cfg(not(feature = "cuda"))]
        let prof_stream: Option<()> = None;
        let mut prof = ProfAccum { qkv_ns: 0, rope_ns: 0, attn_ns: 0, mlp_ns: 0, norm_ns: 0, other_ns: 0, n: 0 };
        macro_rules! prof_sync {
            ($stream:expr) => {
                #[cfg(feature = "cuda")]
                if profiling { if let Some(ref s) = $stream { let _ = s.synchronize(); } }
                #[cfg(not(feature = "cuda"))]
                let _ = &$stream;
            }
        }

        // Take scratch out of self to avoid borrow conflicts with self.layers.iter_mut().
        #[cfg(feature = "cuda")]
        let mut decode_scratch = self.decode_scratch.take();

        // Hybrid graph replayed: layer_in from last non-TQ layer's scratch slot.
        // TQ layers will run eagerly after the graph-replayed non-TQ portion.
        #[cfg(feature = "cuda")]
        if graph_replayed {
            if let Some(ref scratch) = decode_scratch {
                let skip_first = get_skip_layers(&self.layers[0].tq_config);
                let n_layers = self.layers.len();
                // Full graph (no TQ layers): last layer output. Hybrid: last non-TQ layer.
                let graph_last = if skip_first >= n_layers { n_layers - 1 } else { skip_first.saturating_sub(1) };
                let ci = graph_last & 1;
                layer_in = Tensor::from_cuda_arc(
                    Arc::clone(&scratch.combined_bufs[ci]),
                    vec![1, 1, scratch.hidden_dim], scratch.stream.clone(),
                );
            }
        }
        #[cfg(not(feature = "cuda"))]
        let graph_replayed = false;

        // Take layer_swap out of self so the inner loop can use it alongside
        // self.layers.iter_mut().
        #[cfg(feature = "cuda")]
        let mut layer_swap = self.layer_swap.take();
        #[cfg(feature = "cuda")]
        let layer_qweight_ptrs = std::mem::take(&mut self.layer_qweight_ptrs);
        #[cfg(feature = "cuda")]
        let n_layers_total = self.layers.len();

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Hybrid graph replay: skip non-TQ layers (already executed by graph)
            #[cfg(feature = "cuda")]
            if graph_replayed {
                let this_is_tq = get_layer_bits_at(
                    layer_idx, layer.tq_config.bits, &layer.tq_config, layer.n_layers,
                    index_pos + seq_len,
                ).is_some();
                if !this_is_tq {
                    continue;
                }
            }

            // ── Layer-swap pre-hook:
            //    (1) flush graveyard — drop any CudaSlices whose compute-
            //        completion event has fired (safe to cuMemFree now).
            //    (2) wait_for_layer — compute stream waits on the layer's
            //        H2D-ready event from the prior iteration's prefetch.
            #[cfg(feature = "cuda")]
            if let Some(ref mut sm) = layer_swap {
                sm.flush_graveyard();
                sm.wait_for_layer(layer_idx)
                    .expect("layer_swap wait_for_layer failed");
            }

            // Clone (Arc ref-count++ for GPU storage) so the fallback path
            // below can still read layer_in after the fused helper may
            // have rewritten it.
            let x = layer_in.clone();
            // prof_sync macro defined before loop

            // Debug checkpoint: layer input stats
            if gpu_debug && seq_len == 1 {
                if let Ok(data) = x.to_vec1() {
                    let n = data.len();
                    let sum: f64 = data.iter().map(|&v| v as f64).sum();
                    let mean = sum / n as f64;
                    let l2: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
                        |(mn, mx), &v| (mn.min(v), mx.max(v)));
                    eprintln!("[gpu-debug] L{} input: mean={:.6} l2={:.4} min={:.6} max={:.6} first5={:?}",
                        layer_idx, mean, l2, mn, mx, &data[..5.min(n)]);
                }
            }

            // Debug: unconditional capture status check
            #[cfg(feature = "cuda")]
            if capturing && layer_idx < 2 {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    let _ = reg.stream.context().check_err(); // clear stale from prev layer drops
                    match reg.stream.capture_status() {
                        Ok(s) => eprintln!("[cuda-graph] L{} entry: {:?}", layer_idx, s),
                        Err(e) => eprintln!("[cuda-graph] L{} entry ERROR: {}", layer_idx, e),
                    }
                }
            }

            // ── Fused kernel path: 5 launches replace ~13 per layer ──
            // Conditions: CUDA decode (seq_len=1), Q4K separate QKV, standard Mlp, no post-norms.
            // Skip for compressed layers (TQ KV path has CPU-side operations).
            // Phases are scoped to avoid borrow conflicts with &mut self in forward_attn.
            #[cfg(feature = "cuda")]
            let layer_uses_compression = get_layer_bits_at(
                layer_idx, layer.tq_config.bits, &layer.tq_config, layer.n_layers,
                index_pos + seq_len,
            ).is_some();

            // ── Hybrid graph: end capture at TQ boundary ──
            #[cfg(feature = "cuda")]
            maybe_end_hybrid_capture(
                &mut self.graph_manager, &mut capturing,
                layer_idx, layer_uses_compression,
            );

            // ── Megakernel wiring DISABLED pending v0.9.0 optimization ──
            // Sprint 4 (2026-04-19) validated end-to-end firing on Qwen2 7B
            // Q4_K_M with Q6K V + Q6K down. PPL bit-identical. BUT the
            // measured bench was -22% Std (58.7 → 45.8 tok/s) vs baseline.
            // Root causes for the regression (all known-addressable):
            //   1. phase_attn uses naive one-block-per-head; only 28/68 SMs
            //      active during attention. Needs flash_decode split-KV port.
            //   2. 95 KB shmem caps occupancy at 1 block/SM on sm_86. Phase-4
            //      shmem aliasing can drop this under 48 KB for 2 blocks/SM.
            //   3. Phase bodies use basic dp4a. Standalone path has
            //      dp4a_v2/v3/mrow8 variants that the persistent kernel
            //      doesn't pick up.
            // A user setting TQ_MEGAKERNEL=1 hoping for the originally
            // projected +10% would instead hit -22% — that's actively bad
            // UX. So the call site stays commented out until one of the
            // above optimization sprints lands.
            //
            // #[cfg(all(feature = "cuda", feature = "persistent-kernel"))]
            // {
            //     match super::megakernel::try_megakernel_layer(
            //         layer, &mut layer_in, &mut decode_scratch,
            //         seq_len, capturing, layer_uses_compression, index_pos,
            //     ) {
            //         Ok(true)  => continue,
            //         Ok(false) => {},
            //         Err(_)    => {}
            //     }
            // }

            #[cfg(feature = "cuda")]
            if try_fused_decode_layer(
                layer, layer_idx, &mut layer_in, mask.as_ref(), index_pos,
                backend, &mut decode_scratch,
                seq_len, capturing, graph_replayed, layer_uses_compression,
                gpu_debug, profiling, &mut prof,
                prof_stream.as_ref(),
            )? {
                continue;
            }

            // ── Fallback: original separate-kernel path ──
            let _t0 = if profiling { Some(std::time::Instant::now()) } else { None };

            let residual = &x;
            prof_sync!(prof_stream);
            let _t_norm = if profiling { Some(std::time::Instant::now()) } else { None };
            let x = layer.attention_norm.forward(&x, backend)?;
            prof_sync!(prof_stream);
            if let Some(t) = _t_norm { prof.norm_ns += t.elapsed().as_nanos() as u64; }

            let _t_attn = if profiling { Some(std::time::Instant::now()) } else { None };
            let attn = layer.forward_attn(&x, None, mask.as_ref(), index_pos, backend)?;
            #[cfg(feature = "cuda")]
            if false { // placeholder for future graph capture status checks
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    if let Ok(status) = reg.stream.capture_status() {
                        use cudarc::driver::sys::CUstreamCaptureStatus_enum::*;
                        match status {
                            CU_STREAM_CAPTURE_STATUS_ACTIVE => {},
                            s => eprintln!("[cuda-graph] L{} post-attn: capture status={:?}", layer_idx, s),
                        }
                    }
                }
            }
            // Optional post-attention norm (Gemma2)
            let attn = match &layer.post_attention_norm {
                Some(norm) => norm.forward(&attn, backend)?,
                None => attn,
            };
            prof_sync!(prof_stream);
            if let Some(t) = _t_attn { prof.attn_ns += t.elapsed().as_nanos() as u64; }

            // Fused residual add + FFN norm: residual += attn; x = rms_norm(residual)
            let _t_mlp = if profiling { Some(std::time::Instant::now()) } else { None };
            let _enter = layer.span_mlp.enter();
            let attn_f32 = attn.to_dtype(DType::F32)?;
            let residual_f32 = residual.to_dtype(DType::F32)?;

            #[cfg(feature = "cuda")]
            let (x, residual_owned) = if attn_f32.is_cuda() {
                let (normed, new_res) = attn_f32.fused_add_rms_norm_gpu(
                    &residual_f32, &layer.ffn_norm.weight, layer.ffn_norm.eps as f32,
                )?;
                (normed, new_res)
            } else {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_residual) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_residual, shape, attn_f32.device())?)
            };
            #[cfg(not(feature = "cuda"))]
            let (x, residual_owned) = {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_residual) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_residual, shape, attn_f32.device())?)
            };
            let residual = &residual_owned;
            let x = layer.mlp_or_moe.forward(&x, backend)?;
            // Optional post-FFN norm (Gemma2)
            let x = match &layer.post_ffn_norm {
                Some(norm) => norm.forward(&x, backend)?,
                None => x,
            };
            let x = (x + residual)?;
            prof_sync!(prof_stream);
            if let Some(t) = _t_mlp { prof.mlp_ns += t.elapsed().as_nanos() as u64; }
            prof.n += 1;
            layer_in = x;

            // Debug checkpoint: layer output stats
            if gpu_debug && seq_len == 1 {
                if let Ok(data) = layer_in.to_vec1() {
                    let n = data.len();
                    let sum: f64 = data.iter().map(|&v| v as f64).sum();
                    let mean = sum / n as f64;
                    let l2: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
                        |(mn, mx), &v| (mn.min(v), mx.max(v)));
                    eprintln!("[gpu-debug] L{} output: mean={:.6} l2={:.4} min={:.6} max={:.6} first5={:?}",
                        layer_idx, mean, l2, mn, mx, &data[..5.min(n)]);
                }
            }

            // ── Layer-swap post-hook: evict current, prefetch next.
            //    For pinned layers these are no-ops. The pointer cache lets us
            //    reach layers[i+1]'s QWeights without triggering a borrow
            //    conflict with the ongoing `self.layers.iter_mut()` loop.
            #[cfg(feature = "cuda")]
            if let Some(ref mut sm) = layer_swap {
                if layer_idx < layer_qweight_ptrs.len() {
                    unsafe {
                        sm.evict_layer_direct(layer_idx, &layer_qweight_ptrs[layer_idx]);
                    }
                }
                let next = layer_idx + 1;
                if next < n_layers_total && next < layer_qweight_ptrs.len() {
                    // SAFETY: pointers valid while self (and hence self.layers) lives.
                    if let Err(e) = unsafe {
                        sm.start_prefetch_direct(next, &layer_qweight_ptrs[next])
                    } {
                        eprintln!("[layer_swap] prefetch L{} failed: {}", next, e);
                    }
                }
            }
        }

        // Drain any remaining graveyard entries so CudaSlices from the last
        // 1-2 streamed layers don't accumulate across decode steps.
        #[cfg(feature = "cuda")]
        if let Some(ref mut sm) = layer_swap {
            sm.drain_graveyard();
        }

        // Restore layer_swap + pointer cache after the loop.
        #[cfg(feature = "cuda")]
        {
            self.layer_swap = layer_swap;
            self.layer_qweight_ptrs = layer_qweight_ptrs;
        }

        // Restore scratch buffers back into self after layer loop.
        #[cfg(feature = "cuda")]
        { self.decode_scratch = decode_scratch; }

        // End graph capture BEFORE norm/lm_head (they allocate → can't be in graph).
        // Graph captures layer loop only. Norm + lm_head run eagerly after replay.
        #[cfg(feature = "cuda")]
        if capturing {
            self.finalize_layer_loop_capture(&mut layer_in);
        }

        let output = self.apply_norm_and_lm_head(
            &layer_in, backend, seq_len, index_pos, gpu_debug, profiling,
            #[cfg(feature = "cuda")] prof_stream.as_ref(),
        )?;

        if profiling && prof.n > 0 {
            let total = prof.norm_ns + prof.attn_ns + prof.mlp_ns + prof.other_ns;
            if total > 0 {
                eprintln!("[profile] norm {:.1}% | attn {:.1}% | mlp {:.1}% | output {:.1}% | total {:.1}ms",
                    prof.norm_ns as f64 / total as f64 * 100.0,
                    prof.attn_ns as f64 / total as f64 * 100.0,
                    prof.mlp_ns as f64 / total as f64 * 100.0,
                    prof.other_ns as f64 / total as f64 * 100.0,
                    total as f64 / 1e6,
                );
            }
        }

        // Post-forward cleanup: the legacy size-pool deactivate and arena
        // mode-transition blocks (both hardcoded off upstream) were
        // removed 2026-04-13. Size-pool activation, when enabled via
        // TQ_SIZE_POOL=1, is a one-shot process-lifetime setup from
        // engine.rs and does not need per-forward teardown.

        Ok(output)
    }

    /// Run only the first `n_layers` layers + norm + LM head.
    /// Used as the "draft model" in self-speculative decoding: first 8 layers
    /// predict a candidate token at ~25% compute cost. The full model then
    /// verifies the prediction.
    ///
    /// IMPORTANT: this writes to layers[0..n_layers]'s KV caches. The caller
    /// must handle truncation on rejection. Layers beyond n_layers are untouched.
    /// Run all transformer layers and return the LAST-POSITION hidden state
    /// *before* the final norm + LM head. This is what EAGLE-style speculative
    /// decoding needs to drive its draft model.
    ///
    /// Modelled after `forward_partial` (single-token / prefill), skips the
    /// norm + lm_head + softcap tail. Returns a `[hidden_dim]` 1D tensor on
    /// whichever device the layer output lives on.
    ///
    /// Safe to call back-to-back with single-token inputs — uses the same KV
    /// cache state as `forward`, so caller pass the same `index_pos` they'd
    /// pass to `forward`.
    pub fn forward_last_hidden(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let backend = self.backend.clone();
        let backend = backend.as_ref();

        let mask = if seq_len == 1 { None } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };

        let mut layer_in = self.tok_embeddings.forward(x)?;
        if let Some(scale) = self.embed_scale {
            let data = layer_in.as_slice();
            let scaled: Vec<f32> = data.iter().map(|&v| v * scale).collect();
            layer_in = Tensor::from_vec(scaled, layer_in.shape().to_vec(), layer_in.device())?;
        }
        #[cfg(feature = "cuda")]
        if backend.is_gpu() {
            if let Ok(gpu_tensor) = layer_in.to_device_auto() {
                layer_in = gpu_tensor;
            }
        }

        let n = self.layers.len();
        for layer in self.layers[..n].iter_mut() {
            // Snapshot layer_in BEFORE the fused add-norm mutates its Arc'd
            // storage in-place. Without this copy-on-write boundary, the
            // fused_add_rms_norm_gpu write stomps the input buffer we also
            // read via `residual`, producing corrupt state on the 2nd+
            // forward_last_hidden call after a forward_hidden_all prefill.
            // Mirrors forward()'s pattern at the top of its layer loop.
            let x = layer_in.clone();
            let residual = &x;
            let x = layer.attention_norm.forward(&x, backend)?;
            let attn = layer.forward_attn(&x, None, mask.as_ref(), index_pos, backend)?;
            let attn = match &layer.post_attention_norm {
                Some(norm) => norm.forward(&attn, backend)?,
                None => attn,
            };

            let attn_f32 = attn.to_dtype(DType::F32)?;
            let residual_f32 = residual.to_dtype(DType::F32)?;

            #[cfg(feature = "cuda")]
            let (x, residual_owned) = if attn_f32.is_cuda() {
                attn_f32.fused_add_rms_norm_gpu(
                    &residual_f32, &layer.ffn_norm.weight, layer.ffn_norm.eps as f32,
                )?
            } else {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_res) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
            };
            #[cfg(not(feature = "cuda"))]
            let (x, residual_owned) = {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_res) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
            };
            let x = layer.mlp_or_moe.forward(&x, backend)?;
            let x = match &layer.post_ffn_norm {
                Some(norm) => norm.forward(&x, backend)?,
                None => x,
            };
            layer_in = (x + &residual_owned)?;
        }

        // Narrow to last position + squeeze → [hidden]. Matches the layout
        // EAGLE's draft_forward expects (`prev_hidden: &[f32]` of length hidden).
        let last = layer_in.narrow(1, seq_len - 1, 1)?.squeeze(1)?;   // [b, hidden]
        let last = last.squeeze(0)?;                                   // [hidden]
        Ok(last)
    }

    /// Variant of [`Self::forward_last_hidden`] that returns hidden states
    /// for ALL positions, not just the last one. Shape `[seq_len, hidden]`
    /// (batch squeezed out, assumes b_sz=1 which is the only case actually
    /// exercised in this engine).
    ///
    /// Sprint 3 Day 2: the EAGLE acceptance probe needs prompt prefill —
    /// the draft's KV cache must absorb `[hidden[0], hidden[1], ..., hidden[N-1]]`
    /// before generation starts so its attention sees the same context the
    /// target saw. The Day 1 `forward_last_hidden` returned only position
    /// N-1, which is insufficient for that prefill.
    pub fn forward_hidden_all(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let backend = self.backend.clone();
        let backend = backend.as_ref();

        let mask = if seq_len == 1 { None } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };

        let mut layer_in = self.tok_embeddings.forward(x)?;
        if let Some(scale) = self.embed_scale {
            let data = layer_in.as_slice();
            let scaled: Vec<f32> = data.iter().map(|&v| v * scale).collect();
            layer_in = Tensor::from_vec(scaled, layer_in.shape().to_vec(), layer_in.device())?;
        }
        #[cfg(feature = "cuda")]
        if backend.is_gpu() {
            if let Ok(gpu_tensor) = layer_in.to_device_auto() {
                layer_in = gpu_tensor;
            }
        }

        let n = self.layers.len();
        for layer in self.layers[..n].iter_mut() {
            // Snapshot layer_in BEFORE the fused add-norm mutates its Arc'd
            // storage in-place. See forward_last_hidden for the full rationale;
            // without this copy-on-write boundary, chained prefill+decode
            // corrupts the target's KV state.
            let x = layer_in.clone();
            let residual = &x;
            let x = layer.attention_norm.forward(&x, backend)?;
            let attn = layer.forward_attn(&x, None, mask.as_ref(), index_pos, backend)?;
            let attn = match &layer.post_attention_norm {
                Some(norm) => norm.forward(&attn, backend)?,
                None => attn,
            };

            let attn_f32 = attn.to_dtype(DType::F32)?;
            let residual_f32 = residual.to_dtype(DType::F32)?;

            #[cfg(feature = "cuda")]
            let (x, residual_owned) = if attn_f32.is_cuda() {
                attn_f32.fused_add_rms_norm_gpu(
                    &residual_f32, &layer.ffn_norm.weight, layer.ffn_norm.eps as f32,
                )?
            } else {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_res) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
            };
            #[cfg(not(feature = "cuda"))]
            let (x, residual_owned) = {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_res) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
            };
            let x = layer.mlp_or_moe.forward(&x, backend)?;
            let x = match &layer.post_ffn_norm {
                Some(norm) => norm.forward(&x, backend)?,
                None => x,
            };
            layer_in = (x + &residual_owned)?;
        }

        // Squeeze batch → [seq_len, hidden].
        let out = layer_in.squeeze(0)?;
        Ok(out)
    }

    /// Apply the base model's final RMSNorm (but NOT the LM head) to a
    /// hidden vector. Useful for EAGLE draft inputs: SafeAILab's cnets.py
    /// normalises `hidden_states` with the base model's final norm BEFORE
    /// concatenating with embeds and feeding to the fc fusion layer.
    /// Without this, fc_out overshoots by roughly sqrt(hidden) * sigma_W.
    pub fn apply_final_norm(&self, hidden: &Tensor) -> Result<Tensor> {
        let backend = self.backend.clone();
        let backend = backend.as_ref();
        let shape = hidden.shape().to_vec();
        let h = if shape.len() == 1 {
            hidden.unsqueeze(0)?.unsqueeze(0)?
        } else {
            hidden.clone()
        };
        let normed = self.norm.forward(&h, backend)?;
        // Squeeze back to the original rank.
        let normed = if shape.len() == 1 {
            normed.squeeze(0)?.squeeze(0)?
        } else {
            normed
        };
        Ok(normed)
    }

    /// Apply LM head (+ softcap) to a hidden vector, **SKIPPING the final
    /// RMSNorm**. Some EAGLE draft variants produce outputs in the same
    /// space the base model applies `lm_head` to directly, WITHOUT a
    /// preceding norm. Used by the acceptance probe for draft outputs.
    pub fn project_hidden_to_logits_no_norm(&self, hidden: &Tensor) -> Result<Tensor> {
        let backend = self.backend.clone();
        let backend = backend.as_ref();
        let shape = hidden.shape().to_vec();
        let h = if shape.len() == 1 {
            hidden.unsqueeze(0)?.unsqueeze(0)?
        } else {
            hidden.clone()
        };
        let (_b, s) = { let sh = h.shape(); (sh[0], sh[1]) };
        let h = h.narrow(1, s - 1, 1)?.squeeze(1)?;
        let logits = self.output.forward(&h, backend)?;
        let logits = if let Some(cap) = self.final_logit_softcap {
            apply_softcap(&logits, cap)?
        } else { logits };
        Ok(logits)
    }

    /// Apply final norm + LM head + softcap to a single hidden vector.
    /// Inverse of `forward_last_hidden`: takes `[hidden_dim]`, returns
    /// `[vocab_size]` logits. Used by the acceptance-rate probe to project
    /// a draft-predicted hidden back into token space.
    pub fn project_hidden_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let backend = self.backend.clone();
        let backend = backend.as_ref();
        // hidden is [hidden_dim]; expand to [1, 1, hidden] so `norm.forward`
        // and `output.forward` see the batch layout they expect.
        let shape = hidden.shape().to_vec();
        let h = if shape.len() == 1 {
            hidden.unsqueeze(0)?.unsqueeze(0)?
        } else {
            hidden.clone()
        };
        let normed = self.norm.forward(&h, backend)?;
        // Narrow to last position (no-op if seq_len already 1) + squeeze.
        let (_b, s) = {
            let sh = normed.shape();
            (sh[0], sh[1])
        };
        let normed = normed.narrow(1, s - 1, 1)?.squeeze(1)?;
        let logits = self.output.forward(&normed, backend)?;
        let logits = if let Some(cap) = self.final_logit_softcap {
            apply_softcap(&logits, cap)?
        } else { logits };
        Ok(logits)
    }

    pub fn forward_partial(&mut self, x: &Tensor, n_layers: usize, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let backend = self.backend.clone();
        let backend = backend.as_ref();

        let mask = if seq_len == 1 { None } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };

        let mut layer_in = self.tok_embeddings.forward(x)?;
        if let Some(scale) = self.embed_scale {
            let data = layer_in.as_slice();
            let scaled: Vec<f32> = data.iter().map(|&v| v * scale).collect();
            layer_in = Tensor::from_vec(scaled, layer_in.shape().to_vec(), layer_in.device())?;
        }
        #[cfg(feature = "cuda")]
        if backend.is_gpu() {
            if let Ok(gpu_tensor) = layer_in.to_device_auto() {
                layer_in = gpu_tensor;
            }
        }

        let n = n_layers.min(self.layers.len());
        for layer in self.layers[..n].iter_mut() {
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in, backend)?;
            let attn = layer.forward_attn(&x, None, mask.as_ref(), index_pos, backend)?;
            let attn = match &layer.post_attention_norm {
                Some(norm) => norm.forward(&attn, backend)?,
                None => attn,
            };

            let attn_f32 = attn.to_dtype(DType::F32)?;
            let residual_f32 = residual.to_dtype(DType::F32)?;

            #[cfg(feature = "cuda")]
            let (x, residual_owned) = if attn_f32.is_cuda() {
                attn_f32.fused_add_rms_norm_gpu(
                    &residual_f32, &layer.ffn_norm.weight, layer.ffn_norm.eps as f32,
                )?
            } else {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_res) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
            };
            #[cfg(not(feature = "cuda"))]
            let (x, residual_owned) = {
                let shape = attn_f32.shape().to_vec();
                let hidden = *shape.last().unwrap();
                let n_tokens = attn_f32.elem_count() / hidden;
                let (normed, new_res) = backend.fused_add_rms_norm(
                    attn_f32.as_slice(), residual_f32.as_slice(),
                    layer.ffn_norm.weight.as_slice(), layer.ffn_norm.eps as f32,
                    n_tokens, hidden,
                );
                (Tensor::from_vec(normed, shape.clone(), attn_f32.device())?,
                 Tensor::from_vec(new_res, shape, attn_f32.device())?)
            };
            let x = layer.mlp_or_moe.forward(&x, backend)?;
            let x = match &layer.post_ffn_norm {
                Some(norm) => norm.forward(&x, backend)?,
                None => x,
            };
            layer_in = (x + &residual_owned)?;
        }

        // Norm + LM head (same as full forward)
        let x = self.norm.forward(&layer_in, backend)?;
        let x = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let output = self.output.forward(&x, backend)?;
        if let Some(cap) = self.final_logit_softcap {
            apply_softcap(&output, cap)
        } else {
            Ok(output)
        }
    }

    /// Load from safetensors file(s) + config.json (FP16/BF16 models).
    ///
    /// Unlike GGUF, safetensors stores full-precision weights. No QMatMul quantization
    /// overhead — this is the ideal path for measuring TurboQuant's true quality impact
    /// (single quantization layer instead of compound Q4+TQ).
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing config.json + model*.safetensors files
    /// * `device` - Target device
    /// * `tq_config` - TurboQuant configuration
    pub fn from_safetensors(
        model_dir: &std::path::Path,
        device: &Device,
        tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        let src = crate::safetensors_src::SafetensorsContent::open(model_dir)?;
        Self::build(&src, device, tq_config)
    }
}
