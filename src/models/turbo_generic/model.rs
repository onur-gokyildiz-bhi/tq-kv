//! GenericTurboModel: GGUF loading, forward pass, CUDA graph management.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::ComputeBackend;
use crate::cuda::{TqTensor as Tensor, TqDevice as Device, TqDType as DType, TqError};
use crate::cuda::Result;
use crate::gguf::{GgufContent, GgmlDType};
use crate::qmatmul as qmm;
use tq_kv::TurboQuantConfig;

use super::primitives::{Embedding, RmsNorm, apply_softcap, softmax_last_dim, Module, MAX_SEQ_LEN};
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
    /// Cached output tensor from graph capture (for replay).
    #[cfg(feature = "cuda")]
    pub(crate) graph_output: Option<Tensor>,
    /// GPU buffers kept alive for graph replay (prevents ILLEGAL_ADDRESS from freed pointers).
    #[cfg(feature = "cuda")]
    pub(crate) graph_retained_buffers: Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// Pre-allocated intermediate buffers for graph capture (DecodeBufferPool).
    #[cfg(feature = "cuda")]
    pub(crate) graph_pool_buffers: Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// GPU buffer address of the input embedding (for updating before graph replay).
    #[cfg(feature = "cuda")]
    pub(crate) graph_input_buffer: Option<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// GPU scalar for RoPE position offset (updated before graph replay).
    #[cfg(feature = "cuda")]
    pub(crate) rope_pos_gpu: Option<cudarc::driver::CudaSlice<i32>>,
    /// Arena buffer pool for decode (reuses allocs across decode steps without CUDA Graph).
    #[cfg(feature = "cuda")]
    pub(crate) arena_pool_buffers: Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// Number of decode steps completed (0 = first decode records, 1+ = arena reuse).
    #[cfg(feature = "cuda")]
    pub(crate) arena_decode_count: usize,
    /// Pre-allocated scratch buffers for zero-alloc decode (fused kernel path).
    #[cfg(feature = "cuda")]
    pub(crate) decode_scratch: Option<DecodeScratch>,
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

impl GenericTurboModel {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: GgufContent,
        reader: &mut R,
        device: &Device,
        tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
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

        let rope_style = detect_rope_style(&arch);

        // Dump Gemma-relevant metadata
        if arch.contains("gemma") {
            for key in ["attention.value_length", "attention.sliding_window",
                        "attention.query_pre_attn_scalar"] {
                let full_key = format!("{arch}.{key}");
                if let Some(val) = ct.get(&full_key) {
                    eprintln!("  [gguf] {full_key} = {val:?}");
                }
            }
        }

        // Auto-detect features from GGUF tensors
        let has_bias = ct.tensor(reader, "blk.0.attn_q.bias", device).is_ok();
        let has_merged_qkv = ct.tensor(reader, "blk.0.attn_qkv.weight", device).is_ok();
        let has_ffn_gate = ct.tensor(reader, "blk.0.ffn_gate.weight", device).is_ok();
        let has_post_attn_norm = ct.tensor(reader, "blk.0.post_attention_norm.weight", device).is_ok();
        let has_post_ffn_norm = ct.tensor(reader, "blk.0.post_ffw_norm.weight", device).is_ok();

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
        // Note: Gemma 2 HF config says gelu_pytorch_tanh, but GGUF Q4K weights
        // seem to work better with SiLU. TODO: investigate further.
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
        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
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
            let dequant = crate::quant::dequantize(&emb_raw_data, emb_dtype,
                emb_shape.0 * emb_shape.1);
            let tensor = Tensor::from_vec(dequant, vec![emb_shape.0, emb_shape.1], device)?;
            (None, Some(tensor))
        };
        let norm = {
            let n = RmsNorm::from_qtensor(
                ct.tensor(reader, "output_norm.weight", device)?, rms_norm_eps, device,
            )?;
            if arch.contains("gemma") { n.with_add_unit() } else { n }
        };
        // Detect tie_word_embeddings: if output.weight is missing, reuse token embeddings
        let output = match ct.tensor(reader, "output.weight", device) {
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
                let wqkv = ct.tensor(reader, &format!("{prefix}.attn_qkv.weight"), device)?;
                QkvWeights::Merged { wqkv: QMatMul::from_qtensor(wqkv)? }
            } else {
                let wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
                let wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
                let wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
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
            let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            // Optional biases (Qwen2 has them, Llama/Phi/Gemma don't)
            let attention_bq = if has_bias {
                Some(ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?.dequantize_to_device(device)?)
            } else {
                None
            };
            let attention_bk = if has_bias {
                Some(ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?.dequantize_to_device(device)?)
            } else {
                None
            };
            let attention_bv = if has_bias {
                Some(ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?.dequantize_to_device(device)?)
            } else {
                None
            };

            // MLP: 3-gate (most models), 2-gate up/down (Phi-3.5), or MoE
            let mlp_or_moe = if n_expert > 1 {
                let gate_inp = ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let w1 = ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let w2 = ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let w3 = ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
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
                let w1 = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let w2 = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let w3 = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(w3)?,
                    activation: mlp_activation,
                })
            } else {
                // Phi-style: only ffn_up and ffn_down (no ffn_gate)
                let up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                let down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                MlpOrMoe::UpDown(MlpUpDown {
                    ffn_up: QMatMul::from_qtensor(up)?,
                    ffn_down: QMatMul::from_qtensor(down)?,
                })
            };

            let attention_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            // Optional post-norms (Gemma2)
            let post_attention_norm = if has_post_attn_norm {
                let t = ct.tensor(reader, &format!("{prefix}.post_attention_norm.weight"), device)?;
                Some(RmsNorm::from_qtensor(t, rms_norm_eps, device)?)
            } else {
                None
            };
            let post_ffn_norm = if has_post_ffn_norm {
                let t = ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?;
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
            graph_manager: crate::cuda::graph::CudaGraphManager::new(
                std::env::var("TQ_GRAPH").map(|v| v == "1").unwrap_or(false)
            ),
            #[cfg(feature = "cuda")]
            graph_output: None,
            #[cfg(feature = "cuda")]
            graph_retained_buffers: Vec::new(),
            #[cfg(feature = "cuda")]
            graph_pool_buffers: Vec::new(),
            #[cfg(feature = "cuda")]
            graph_input_buffer: None,
            #[cfg(feature = "cuda")]
            rope_pos_gpu: None,
            #[cfg(feature = "cuda")]
            arena_pool_buffers: Vec::new(),
            #[cfg(feature = "cuda")]
            arena_decode_count: 0,
            #[cfg(feature = "cuda")]
            decode_scratch: None,
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
        // Disable with TQ_NO_WARMUP=1 on low-memory systems.
        let do_warmup = std::env::var("TQ_NO_WARMUP").map(|v| v != "1").unwrap_or(true);
        if do_warmup {
        let b = model.backend.as_ref();
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
        // Upload persistent tensors to GPU (norm weights, cos/sin, biases).
        // Full broadcast GPU support enables this (stride-based kernels).
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

        eprintln!("  Weight caches warmed.");

        // Note: A.2 (raw_data release after GPU upload) was attempted but caused
        // crashes in CPU fallback paths (e.g. Q6K prefill dequant). Deferred until
        // all forward paths are fully GPU-resident with no CPU fallback.

        } // end if do_warmup

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

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;

        // Reset graph + arena on new sequence (prefill)
        #[cfg(feature = "cuda")]
        if seq_len > 1 {
            self.graph_manager.reset();
            self.graph_output = None;
            self.graph_retained_buffers.clear();
            self.graph_pool_buffers.clear();
            self.arena_pool_buffers.clear();
            self.arena_decode_count = 0;
        }

        // ── CUDA Graph replay (layer loop only — norm+lm_head run eagerly after) ──
        // Skip graph replay when seq_len > 256: fall through to eager mode
        // so flash_decode can be used (split-KV parallelism for long context).
        // Graph captures gqa_decode which is single-block — too slow past 256 tokens.
        #[cfg(feature = "cuda")]
        let mut graph_replayed = false;
        #[cfg(feature = "cuda")]
        let kv_len_for_graph = self.layers.first()
            .and_then(|l| l.gpu_kv_cache.as_ref())
            .map(|kv| kv.seq_len)
            .unwrap_or(0);
        #[cfg(feature = "cuda")]
        if seq_len == 1 && self.graph_manager.is_ready(1) && kv_len_for_graph <= 256 {
            let reg = crate::cuda::kernels::global_registry().unwrap();

            // Update embedding into dedicated buffer (same GPU address as capture)
            if let Some(ref mut input_buf) = self.graph_input_buffer {
                let new_emb = self.tok_embeddings.forward(x)?;
                let emb_data = new_emb.as_slice();
                let _ = reg.stream.context().check_err();
                let buf_mut = std::sync::Arc::get_mut(input_buf)
                    .expect("graph_input_buffer refcount > 1");
                let _ = reg.stream.memcpy_htod(emb_data, buf_mut);
            }

            // Update RoPE position GPU scalar
            if let Some(ref mut rope_buf) = self.rope_pos_gpu {
                let _ = reg.stream.memcpy_htod(&[index_pos as i32], rope_buf);
            }

            // Update KV cache valid_len (pre-append position)
            for layer in &mut self.layers {
                if let Some(ref mut gpu_kv) = layer.gpu_kv_cache {
                    let _ = reg.stream.memcpy_htod(&[gpu_kv.seq_len as i32], &mut gpu_kv.valid_len_gpu);
                }
            }

            // No sync needed: memcpy_htod + graph launch are stream-ordered.
            if let Ok(()) = self.graph_manager.replay(&reg.stream, 1) {
                // No post-replay sync: TQ layers launch on same stream after graph.
                // Post-replay: increment seq_len
                for layer in &mut self.layers {
                    if let Some(ref mut gpu_kv) = layer.gpu_kv_cache {
                        gpu_kv.seq_len += 1;
                    }
                }
                graph_replayed = true;
                // DON'T return — fall through to norm + lm_head
            }
        }

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

        // ── Size-based buffer pool for decode (replaces broken cursor-based arena) ──
        // Order-independent: buffers matched by size, not cursor position.
        // Arc::make_mut clones create new pool entries that get reused next token.
        // Size pool starts from decode step 1+ (step 0 discovers buffer pattern).
        #[cfg(feature = "cuda")]
        // Size pool disabled: cudarc CudaSlice internal state breaks on ptr::read.
        // CUDA's cuMemAllocAsync already uses an internal memory pool.
        // TODO: Revisit with cudarc raw pointer API or custom allocator.
        let size_pool_active = false;
        #[cfg(feature = "cuda")]
        if size_pool_active {
            crate::cuda::size_pool_activate();
        }

        // Legacy arena disabled.
        #[cfg(feature = "cuda")]
        let arena_active = false;
        #[cfg(not(feature = "cuda"))]
        let arena_active = false;
        #[cfg(not(feature = "cuda"))]
        let size_pool_active = false;
        #[cfg(feature = "cuda")]
        if arena_active {
            if self.arena_decode_count == 0 {
                // First decode: warm-up pass (GpuKvCache creation happens here).
                // Don't record — alloc pattern differs from steady state.
            } else if self.arena_decode_count == 1 {
                // Second decode: record steady-state alloc pattern.
                crate::cuda::decode_pool_set_mode(crate::cuda::PoolMode::Recording);
            } else {
                // Third+ decode: reuse recorded buffers.
                crate::cuda::decode_pool_restore(std::mem::take(&mut self.arena_pool_buffers));
                crate::cuda::decode_pool_set_mode(crate::cuda::PoolMode::Arena);
                crate::cuda::decode_pool_reset_cursor();
            }
        }

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let backend = self.backend.clone();
        let backend = backend.as_ref();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        // Gemma: scale embeddings by sqrt(hidden_dim)
        if let Some(scale) = self.embed_scale {
            let data = layer_in.as_slice();
            let scaled: Vec<f32> = data.iter().map(|&v| v * scale).collect();
            layer_in = Tensor::from_vec(scaled, layer_in.shape().to_vec(), layer_in.device())?;
        }

        // Phase 3: Upload embedding to GPU for GPU-resident forward pass.
        // All subsequent ops auto-dispatch to GPU when tensor is CUDA.
        #[cfg(feature = "cuda")]
        if backend.is_gpu() {
            if let Ok(gpu_tensor) = layer_in.to_device_auto() {
                layer_in = gpu_tensor;
            }
        }

        // Save input buffer for graph replay. Must be a SEPARATE allocation
        // (not Arc::clone of layer_in) so graph_input_buffer always has refcount=1
        // and Arc::get_mut succeeds on replay without cloning to a new address.
        #[cfg(feature = "cuda")]
        if capturing || recording {
            if layer_in.is_cuda() {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    let src = layer_in.cuda_data();
                    let n = layer_in.elem_count();
                    // Allocate dedicated buffer (refcount=1, never shared)
                    if self.graph_input_buffer.is_none() {
                        if let Ok(buf) = crate::cuda::gpu_alloc_zeros_pub(&reg.stream, n) {
                            self.graph_input_buffer = Some(std::sync::Arc::new(buf));
                        }
                    }
                    // Copy embedding data into the dedicated buffer
                    if let Some(ref mut buf) = self.graph_input_buffer {
                        let dst = std::sync::Arc::get_mut(buf).expect("graph_input_buffer shared");
                        let _ = reg.stream.memcpy_dtod(src, dst);
                        // Replace layer_in with tensor backed by graph_input_buffer
                        layer_in = Tensor::from_cuda_arc(
                            buf.clone(), layer_in.shape().to_vec(), reg.stream.clone(),
                        );
                    }
                }
            }
        }

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
        struct ProfAccum { qkv_ns: u64, rope_ns: u64, attn_ns: u64, mlp_ns: u64, norm_ns: u64, other_ns: u64, n: u32 }
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

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Hybrid graph replay: skip non-TQ layers (already executed by graph)
            #[cfg(feature = "cuda")]
            if graph_replayed {
                let this_is_tq = get_layer_bits(layer_idx, layer.tq_config.bits, &layer.tq_config, layer.n_layers).is_some();
                if !this_is_tq {
                    continue;
                }
            }

            let x = layer_in;
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
            let layer_uses_compression = get_layer_bits(layer_idx, layer.tq_config.bits, &layer.tq_config, layer.n_layers).is_some();

            // ── Hybrid graph: end capture at TQ boundary ──
            // Non-TQ layers are captured in the graph. When we hit the first TQ layer,
            // end capture and replay immediately. TQ layers run eagerly after.
            #[cfg(feature = "cuda")]
            if capturing && layer_uses_compression
                && matches!(self.graph_manager.status, crate::cuda::graph::GraphStatus::Capturing)
            {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    match self.graph_manager.end_capture(&reg.stream, 1) {
                        Ok(()) => {
                            eprintln!("[cuda-graph] hybrid: captured {} non-TQ layers", layer_idx);
                            // Capture only records — replay to actually execute
                            let _ = reg.stream.context().check_err();
                            match self.graph_manager.replay(&reg.stream, 1) {
                                Ok(()) => {
                                    let _ = reg.stream.synchronize();
                                    eprintln!("[cuda-graph] hybrid: initial replay OK");
                                }
                                Err(e) => {
                                    eprintln!("[cuda-graph] hybrid: replay FAILED: {}", e);
                                    self.graph_manager.reset();
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[cuda-graph] hybrid: end_capture failed: {}", e);
                            self.graph_manager.reset();
                        }
                    }
                }
                capturing = false;
            }

            #[cfg(feature = "cuda")]
            let fused_disabled = {
                static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *C.get_or_init(|| std::env::var("TQ_NO_FUSED").map(|v| v == "1").unwrap_or(false))
            };
            #[cfg(feature = "cuda")]
            if seq_len == 1 && x.is_cuda() && !layer_uses_compression && !fused_disabled
                && layer.post_attention_norm.is_none()
                && layer.post_ffn_norm.is_none()
                && matches!(&layer.qkv, QkvWeights::Separate { .. })
                && matches!(&layer.mlp_or_moe, MlpOrMoe::Mlp(_))
            {
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
                            let _tqkv = if pdet_qkv { let _ = reg.stream.synchronize(); Some(std::time::Instant::now()) } else { None };
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

                            // Step 2: Q matvec (Q4K) → scratch.q_buf
                            {
                                let oq = Arc::get_mut(&mut scratch.q_buf).expect("q aliased");
                                crate::cuda::kernels::q4km_matvec(
                                    reg, wq_gpu, normed_ptr, oq, q_out, hidden_dim,
                                ).map_err(|e| TqError::Msg(format!("scratch Q matvec: {}", e)))?;
                            }
                            // Step 3: K matvec (Q4K) → scratch.k_buf
                            {
                                let ok = Arc::get_mut(&mut scratch.k_buf).expect("k aliased");
                                crate::cuda::kernels::q4km_matvec(
                                    reg, wk_gpu, normed_ptr, ok, k_out, hidden_dim,
                                ).map_err(|e| TqError::Msg(format!("scratch K matvec: {}", e)))?;
                            }
                            // Step 4: V matvec (Q4K or Q6K fallback) → scratch.v_buf
                            {
                                let ov = Arc::get_mut(&mut scratch.v_buf).expect("v aliased");
                                let wv = if let QkvWeights::Separate { wv, .. } = &layer.qkv { wv } else { unreachable!() };
                                if let Some((wv_gpu, v_out, _)) = wv.q4k_gpu_data() {
                                    crate::cuda::kernels::q4km_matvec(
                                        reg, wv_gpu, normed_ptr, ov, v_out, hidden_dim,
                                    ).map_err(|e| TqError::Msg(format!("scratch V Q4K: {}", e)))?;
                                } else if let qmm::QMatMul::Quantized(qw) = &wv.inner {
                                    // Q6K: native fused dequant matvec (4.9x less bandwidth than F32)
                                    let w_raw = qw.gpu_cache_or_upload(&reg.stream);
                                    crate::cuda::kernels::q6k_matvec(
                                        reg, w_raw, normed_ptr, ov, qw.out_features(), qw.in_features(),
                                    ).map_err(|e| TqError::Msg(format!("scratch V Q6K: {}", e)))?;
                                } else {
                                    return Err(TqError::Msg("V weight: unsupported format".into()));
                                }
                            }
                            // Step 5: Add biases in-place (Qwen2 has Q/K/V biases)
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
                            if let Some(t) = _tqkv { let _ = reg.stream.synchronize(); eprintln!("[kernel] {:>12}: {:.1}μs", "norm+qkv", t.elapsed().as_nanos() as f64 / 1000.0); }
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

                        // Per-kernel detail profiling (TQ_PROFILE=1, layer 0 only)
                        let pdet = profiling && layer_idx == 0;
                        macro_rules! ptime {
                            ($label:expr, $body:expr) => {{
                                let _t = if pdet { let _ = reg.stream.synchronize(); Some(std::time::Instant::now()) } else { None };
                                let r = $body;
                                if let Some(t) = _t { let _ = reg.stream.synchronize(); eprintln!("[kernel] {:>12}: {:.1}μs", $label, t.elapsed().as_nanos() as f64 / 1000.0); }
                                r
                            }};
                        }

                        // 1. RoPE in-place (graph-safe: uses rope_pos_gpu when available)
                        let rope_gpu_pos = super::ROPE_POS_GPU_PTR.with(|p| p.get());
                        let rope_gpu_ref = if rope_gpu_pos != 0 {
                            Some(unsafe { &*(rope_gpu_pos as *const cudarc::driver::CudaSlice<i32>) })
                        } else { None };
                        ptime!("rope_q", {
                            let q_mut = Arc::get_mut(&mut scratch.q_buf).expect("q aliased");
                            let rope_fn = match layer.rope_style {
                                RopeStyle::Halved => crate::cuda::kernels::rope_halved_with_gpu_pos,
                                RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved_with_gpu_pos,
                            };
                            rope_fn(reg, q_mut, layer.cos.cuda_data(), layer.sin.cuda_data(),
                                1, scratch.n_head, scratch.head_dim, layer.rope_dim, index_pos, rope_gpu_ref,
                            ).map_err(|e| TqError::Msg(format!("scratch RoPE Q: {}", e)))
                        })?;
                        ptime!("rope_k", {
                            let k_mut = Arc::get_mut(&mut scratch.k_buf).expect("k aliased");
                            let rope_fn = match layer.rope_style {
                                RopeStyle::Halved => crate::cuda::kernels::rope_halved_with_gpu_pos,
                                RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved_with_gpu_pos,
                            };
                            rope_fn(reg, k_mut, layer.cos.cuda_data(), layer.sin.cuda_data(),
                                1, scratch.n_kv_head, scratch.head_dim, layer.rope_dim, index_pos, rope_gpu_ref,
                            ).map_err(|e| TqError::Msg(format!("scratch RoPE K: {}", e)))
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
                            let use_flash = !capturing && !graph_replayed
                                && gpu_kv.seq_len > 256;

                            if use_flash {
                                // Flash decode: split KV across blocks for parallelism
                                let actual_seq = gpu_kv.seq_len;
                                let split_size = 256usize;
                                let n_splits = (actual_seq + split_size - 1) / split_size;
                                let scale = 1.0 / (scratch.head_dim as f32).sqrt();

                                let mut partial_o: cudarc::driver::CudaSlice<f32> = reg.stream.alloc_zeros(
                                    scratch.n_head * n_splits * scratch.head_dim
                                ).map_err(|e| TqError::Msg(format!("flash partial_o: {}", e)))?;
                                let mut partial_max: cudarc::driver::CudaSlice<f32> = reg.stream.alloc_zeros(
                                    scratch.n_head * n_splits
                                ).map_err(|e| TqError::Msg(format!("flash partial_max: {}", e)))?;
                                let mut partial_sum: cudarc::driver::CudaSlice<f32> = reg.stream.alloc_zeros(
                                    scratch.n_head * n_splits
                                ).map_err(|e| TqError::Msg(format!("flash partial_sum: {}", e)))?;

                                crate::cuda::kernels::flash_decode_partial(
                                    reg, &*scratch.q_buf, &*gpu_kv.k_buf, &*gpu_kv.v_buf,
                                    &mut partial_o, &mut partial_max, &mut partial_sum,
                                    1, scratch.n_head, scratch.n_kv_head,
                                    actual_seq, scratch.head_dim, scale, split_size,
                                    gpu_kv.max_seq,
                                    0, // window_size: 0 = global (TODO: per-layer for Gemma 2)
                                ).map_err(|e| TqError::Msg(format!("flash_decode_partial: {}", e)))?;

                                crate::cuda::kernels::flash_decode_reduce(
                                    reg, &partial_o, &partial_max, &partial_sum,
                                    attn_mut,
                                    scratch.n_head, n_splits, scratch.head_dim, 1,
                                ).map_err(|e| TqError::Msg(format!("flash_decode_reduce: {}", e)))?;
                                Ok::<(), TqError>(())
                            } else {
                                // Graph-safe single-block attention
                                let attn_extra = if capturing { 1i32 } else { 0 };
                                crate::cuda::kernels::gqa_decode_attention_graph(
                                    reg, &*scratch.q_buf, &*gpu_kv.k_buf, &*gpu_kv.v_buf,
                                    attn_mut, &gpu_kv.valid_len_gpu,
                                    scratch.n_head, scratch.n_kv_head,
                                    gpu_kv.max_seq, scratch.head_dim,
                                    1.0 / (scratch.head_dim as f32).sqrt(),
                                    attn_extra,
                                    0, // window_size: 0 = global (TODO: per-layer sliding window for Gemma 2)
                                ).map_err(|e| TqError::Msg(format!("gqa_decode_attn: {}", e)))?;
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
                        layer.forward_attn(&x, Some((q_t, k_t, v_t)), mask.as_ref(), index_pos, backend)?
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
                            let _t = if pdet { let _ = reg.stream.synchronize(); Some(std::time::Instant::now()) } else { None };
                            {
                                let comb = Arc::get_mut(&mut scratch.combined_bufs[ci])
                                    .expect("scratch combined ping-pong aliased");
                                crate::cuda::kernels::add(
                                    reg, residual_f32.cuda_data(), attn_f32.cuda_data(),
                                    comb, hidden_dim,
                                ).map_err(|e| TqError::Msg(format!("scratch add: {}", e)))?;
                            }
                            if let Some(t) = _t { let _ = reg.stream.synchronize(); eprintln!("[kernel] {:>12}: {:.1}μs", "res+attn", t.elapsed().as_nanos() as f64 / 1000.0); }

                            // 2. fused gateup+silu
                            let _t = if pdet { Some(std::time::Instant::now()) } else { None };
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
                            if let Some(t) = _t { let _ = reg.stream.synchronize(); eprintln!("[kernel] {:>12}: {:.1}μs", "gateup", t.elapsed().as_nanos() as f64 / 1000.0); }

                            // 3. down projection + residual add
                            let _t = if pdet { Some(std::time::Instant::now()) } else { None };
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
                            if let Some(t) = _t { let _ = reg.stream.synchronize(); eprintln!("[kernel] {:>12}: {:.1}μs", "down+res", t.elapsed().as_nanos() as f64 / 1000.0); }

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
                            layer_in = result;
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
                            continue; // skip fallback path
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
                            layer_in = (mlp_out + &residual_owned)?;
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
                            continue;
                        }
                    }
                }
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
        }

        // Restore scratch buffers back into self after layer loop.
        #[cfg(feature = "cuda")]
        { self.decode_scratch = decode_scratch; }

        // End graph capture BEFORE norm/lm_head (they allocate → can't be in graph).
        // Graph captures layer loop only. Norm + lm_head run eagerly after replay.
        #[cfg(feature = "cuda")]
        if capturing && matches!(self.graph_manager.status, crate::cuda::graph::GraphStatus::Capturing) {
            if let Some(reg) = crate::cuda::kernels::global_registry() {
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
                                // Do NOT increment again here.
                                // Update layer_in from scratch combined buffer
                                if let Some(ref scratch) = self.decode_scratch {
                                    let last_ci = (self.layers.len() - 1) & 1;
                                    layer_in = Tensor::from_cuda_arc(
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
        }

        let x = self.norm.forward(&layer_in, backend)?;
        // For batched verify (seq_len > 1, index_pos > 0): keep ALL positions' logits.
        // For normal prefill/decode: only last position (next-token prediction).
        let x = if seq_len > 1 && index_pos > 0 {
            x.reshape(vec![seq_len, x.shape()[2]])?  // [seq_len, hidden]
        } else {
            x.narrow(1, seq_len - 1, 1)?.squeeze(1)?  // [1, hidden]
        };
        let _enter = self.span_output.enter();
        #[cfg(feature = "cuda")]
        let _t_lm = if profiling {
            if let Some(ref s) = prof_stream { let _ = s.synchronize(); }
            Some(std::time::Instant::now())
        } else { None };
        #[cfg(not(feature = "cuda"))]
        let _t_lm: Option<std::time::Instant> = None;
        // Debug: compare GPU vs CPU lm_head output
        if gpu_debug {
            if let Ok(h) = x.to_vec1() {
                let n = h.len();
                let l2: f64 = h.iter().map(|&v| (v as f64)*(v as f64)).sum::<f64>().sqrt();
                eprintln!("[gpu-debug] pre-lm_head: n={} l2={:.4}", n, l2);

                // CPU reference matmul for lm_head
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
        if let Some(t) = _t_lm {
            if let Some(ref s) = prof_stream { let _ = s.synchronize(); }
            eprintln!("[kernel] {:>12}: {:.1}μs", "lm_head", t.elapsed().as_nanos() as f64 / 1000.0);
        }

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

        // (End graph capture now handled before norm — see above)

        // After forward pass: deactivate size pool + track decode count
        #[cfg(feature = "cuda")]
        if seq_len == 1 && !capturing && !recording {
            if size_pool_active {
                crate::cuda::size_pool_deactivate();
                if self.arena_decode_count % 50 == 0 {
                    crate::cuda::size_pool_report();
                }
            }
            self.arena_decode_count += 1;
        }

        // After forward pass: handle arena mode transitions (legacy, disabled)
        #[cfg(feature = "cuda")]
        if arena_active {
            if self.arena_decode_count == 0 {
                // First decode done: warm-up pass, no recording. Just increment.
            } else if self.arena_decode_count == 1 {
                // Second decode done: drain recorded steady-state buffers for reuse.
                self.arena_pool_buffers = crate::cuda::decode_pool_drain();
                crate::cuda::decode_pool_set_mode(crate::cuda::PoolMode::Off);
                eprintln!("[arena] recording done: {} buffers saved for reuse", self.arena_pool_buffers.len());
            } else {
                // Arena reuse done: drain back to self, reset mode.
                self.arena_pool_buffers = crate::cuda::decode_pool_drain();
                crate::cuda::decode_pool_set_mode(crate::cuda::PoolMode::Off);
            }
            self.arena_decode_count += 1;
        }

        Ok(output)
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
        _model_dir: &std::path::Path,
        _device: &Device,
        _tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        bail!("safetensors loading not yet implemented for tq-cuda backend")
    }
}
