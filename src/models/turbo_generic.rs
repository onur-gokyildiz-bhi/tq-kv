//! Generic TurboQuant-enhanced model — replaces turbo_llama.rs and turbo_qwen2.rs.
//!
//! Reads GGUF metadata to auto-configure for any standard transformer architecture
//! (llama, qwen2, mistral, phi3, gemma, etc.). Supports optional attention biases,
//! MoE, and tie_word_embeddings — all detected from GGUF tensors at load time.
//!
//! Uses the same proven TurboQuant KV cache compression as turbo_qwen2.rs:
//! - Selective compression (first N layers uncompressed)
//! - Halved RoPE (rope_manual)
//! - f32 softmax in fused SIMD path
//! - f32 attention in decompress path (GPU)

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use rayon::prelude::*;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use tq_kv::TurboQuantConfig;

// ============================================================
// Shared primitives (CUDA-compatible)
// ============================================================

/// Device-aware RmsNorm — candle's version always dequantizes to CPU.
/// This version uses only primitive ops that have CUDA kernels.
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    fn from_qtensor(qtensor: candle_core::quantized::QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        Ok(Self { weight, eps, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(x.rank() - 1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)?.to_dtype(x_dtype)
    }
}

pub const MAX_SEQ_LEN: usize = 4096;

/// CUDA-compatible softmax (candle_nn::ops::softmax_last_dim has no CUDA kernel).
fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let last = x.rank() - 1;
    let max_val = x.max_keepdim(last)?;
    let exp = x.broadcast_sub(&max_val)?.exp()?;
    let sum = exp.sum_keepdim(last)?;
    exp.broadcast_div(&sum)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: candle_core::quantized::QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ============================================================
// MLP / MoE
// ============================================================

/// Standard 3-gate MLP: gate (w1) + up (w3) → silu(gate) * up → down (w2)
#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2.forward(&(silu(&w1)? * w3)?)
    }
}

/// Phi-style 2-gate MLP: up projects to 2×intermediate, split into gate+up halves,
/// apply silu(gate) * up, then down projects back.
#[derive(Debug, Clone)]
struct MlpUpDown {
    ffn_up: QMatMul,
    ffn_down: QMatMul,
}

impl Module for MlpUpDown {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up = self.ffn_up.forward(xs)?;
        let chunks = up.chunk(2, xs.rank() - 1)?;
        self.ffn_down.forward(&(silu(&chunks[0])? * &chunks[1])?)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    UpDown(MlpUpDown),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = softmax_last_dim(&router_logits)?;
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        sum_routing_weights += rw[expert_idx];
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        selected_rws[expert_idx].push(rw[expert_idx] / sum_routing_weights);
                    }
                }

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() { continue; }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws = Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                        .reshape(((), 1))?;
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }
                ys.reshape((b_size, seq_len, hidden_dim))
            }
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::UpDown(mlp) => mlp.forward(xs),
        }
    }
}

// ============================================================
// Utility functions
// ============================================================

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

// ============================================================
// TurboQuant KV Cache — Incremental + Fused Attention
// ============================================================

#[derive(Clone, Debug)]
struct CompressedKvCache {
    k_per_head: Vec<tq_kv::CompressedKeys>,
    v: Tensor,
    cached_len: usize,
    dtype: DType,
}

fn decompress_cache_to_tensor(
    k_per_head: &[tq_kv::CompressedKeys],
    n_kv_head: usize,
    total_len: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
    config: &TurboQuantConfig,
) -> Result<Tensor> {
    let mut all_data = Vec::with_capacity(n_kv_head * total_len * head_dim);
    for compressed in k_per_head.iter().take(n_kv_head) {
        let decompressed = tq_kv::decompress_keys(compressed, config);
        all_data.extend(decompressed);
    }
    Tensor::from_vec(all_data, (1, n_kv_head, total_len, head_dim), device)?.to_dtype(dtype)
}

// ============================================================
// RoPE variants
// ============================================================

/// Halved RoPE: first half / second half layout (Qwen2, most modern models).
fn rope_halved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = x.dims4()?;
    let half = d / 2;
    let x0 = x.narrow(3, 0, half)?;
    let x1 = x.narrow(3, half, half)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&r0, &r1], 3)
}

/// Interleaved RoPE: pairs (x0,x1), (x2,x3), ... layout (Llama).
fn rope_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let half = d / 2;
    let x = x.reshape((b, h, s, half, 2))?;
    let x0 = x.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = x.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let r0 = r0.unsqueeze(4)?;
    let r1 = r1.unsqueeze(4)?;
    Tensor::cat(&[&r0, &r1], 4)?.reshape((b, h, s, d))
}

/// Which RoPE layout to use.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RopeStyle {
    /// First-half / second-half (Qwen2, Mistral, Gemma, most modern models)
    Halved,
    /// Interleaved pairs (Llama)
    Interleaved,
}

// ============================================================
// Layer with TurboQuant KV cache
// ============================================================

/// Number of initial layers to keep uncompressed (fp16 KV cache).
/// Reduces error accumulation in deep models where early-layer errors
/// propagate through all subsequent layers.
const TQ_SKIP_FIRST_LAYERS: usize = 4;

/// QKV weight layout — separate tensors (most models) or merged single tensor (Phi-3.5).
#[derive(Debug, Clone)]
enum QkvWeights {
    Separate { wq: QMatMul, wk: QMatMul, wv: QMatMul },
    Merged { wqkv: QMatMul },
}

#[derive(Debug, Clone)]
struct LayerWeights {
    qkv: QkvWeights,
    attention_wo: QMatMul,
    // Optional biases (Qwen2 has them, Llama doesn't)
    attention_bq: Option<Tensor>,
    attention_bk: Option<Tensor>,
    attention_bv: Option<Tensor>,
    attention_norm: RmsNorm,
    post_attention_norm: Option<RmsNorm>,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    post_ffn_norm: Option<RmsNorm>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Padded head_dim (next power of 2) for Hadamard transform.
    /// Equal to head_dim when head_dim is already a power of 2.
    padded_head_dim: usize,
    rope_dim: usize,
    rope_style: RopeStyle,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    /// Layer index (0-based) — used for selective compression
    layer_idx: usize,
    /// Standard KV cache for uncompressed layers
    kv_cache: Option<(Tensor, Tensor)>,
    kv_compressed: Option<CompressedKvCache>,
    tq_config: TurboQuantConfig,
    signs: Vec<f32>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let x = x.contiguous()?;
        if self.rope_dim < self.head_dim {
            // Partial RoPE: apply rotation only to first rope_dim dimensions,
            // leave the remaining dimensions unchanged.
            let x_rope = x.narrow(3, 0, self.rope_dim)?;
            let x_pass = x.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
            let x_rotated = match self.rope_style {
                RopeStyle::Halved => rope_halved(&x_rope, &cos, &sin)?,
                RopeStyle::Interleaved => rope_interleaved(&x_rope, &cos, &sin)?,
            };
            Tensor::cat(&[&x_rotated, &x_pass], 3)
        } else {
            match self.rope_style {
                RopeStyle::Halved => rope_halved(&x, &cos, &sin),
                RopeStyle::Interleaved => rope_interleaved(&x, &cos, &sin),
            }
        }
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let (mut q, mut k, mut v) = match &self.qkv {
            QkvWeights::Separate { wq, wk, wv } => {
                (wq.forward(x)?, wk.forward(x)?, wv.forward(x)?)
            }
            QkvWeights::Merged { wqkv } => {
                let qkv = wqkv.forward(x)?;
                let q_size = self.n_head * self.head_dim;
                let kv_size = self.n_kv_head * self.head_dim;
                let q = qkv.narrow(2, 0, q_size)?;
                let k = qkv.narrow(2, q_size, kv_size)?;
                let v = qkv.narrow(2, q_size + kv_size, kv_size)?;
                (q, k, v)
            }
        };

        // Apply biases if present (Qwen2 has them, Llama/Phi/Gemma don't)
        if let Some(bq) = &self.attention_bq {
            q = q.broadcast_add(bq)?;
        }
        if let Some(bk) = &self.attention_bk {
            k = k.broadcast_add(bk)?;
        }
        if let Some(bv) = &self.attention_bv {
            v = v.broadcast_add(bv)?;
        }

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // Selective compression: first TQ_SKIP_FIRST_LAYERS use standard fp16 KV cache,
        // remaining layers use TurboQuant compression.
        let use_compression = self.layer_idx >= TQ_SKIP_FIRST_LAYERS;
        let n_rep = self.n_head / self.n_kv_head;

        if use_compression {
            // COMPRESSED PATH: TurboQuant KV cache
            let k_contiguous = k.contiguous()?;
            let k_flat = k_contiguous.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let cache_dtype = k.dtype();

            if self.kv_compressed.is_none() {
                let mut k_per_head = Vec::with_capacity(self.n_kv_head);
                for _ in 0..self.n_kv_head {
                    k_per_head.push(tq_kv::CompressedKeys::new_empty(
                        self.tq_config.bits, self.padded_head_dim, self.tq_config.rotation_seed,
                    ));
                }
                self.kv_compressed = Some(CompressedKvCache {
                    k_per_head,
                    v: v.clone(),
                    cached_len: 0,
                    dtype: cache_dtype,
                });
            }

            let cache = self.kv_compressed.as_mut().unwrap();

            for h in 0..self.n_kv_head {
                for s in 0..seq_len {
                    let offset = (h * seq_len + s) * self.head_dim;
                    let key_vec = &k_flat[offset..offset + self.head_dim];
                    if self.padded_head_dim > self.head_dim {
                        // Pad key vector with zeros to next power of 2 for Hadamard
                        let mut key_vec_padded = vec![0.0f32; self.padded_head_dim];
                        key_vec_padded[..self.head_dim].copy_from_slice(key_vec);
                        let (packed, norm) = tq_kv::compress_single_key_with_signs(
                            &key_vec_padded, self.padded_head_dim, &self.tq_config, &self.signs,
                        );
                        cache.k_per_head[h].append_raw(&packed, norm);
                    } else {
                        let (packed, norm) = tq_kv::compress_single_key_with_signs(
                            key_vec, self.head_dim, &self.tq_config, &self.signs,
                        );
                        cache.k_per_head[h].append_raw(&packed, norm);
                    }
                }
            }

            if cache.cached_len > 0 {
                cache.v = Tensor::cat(&[&cache.v, &v], 2)?;
            } else {
                cache.v = v.clone();
            }
            cache.cached_len += seq_len;
            let total_len = cache.cached_len;

            let y = if seq_len == 1 && total_len > 1 && q.device().is_cpu() {
                // FUSED PATH: Rayon parallel + SIMD (CPU generation)
                let q_flat = q.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                let centroids = tq_kv::codebook::get_centroids(self.tq_config.bits);
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                let need_pad = self.padded_head_dim > self.head_dim;
                let pdim = self.padded_head_dim;
                let hdim = self.head_dim;

                let head_scores: Vec<Vec<f32>> = (0..self.n_head).into_par_iter().map(|qh| {
                    let kv_h = qh / n_rep;
                    let q_vec = &q_flat[qh * hdim..(qh + 1) * hdim];
                    let q_for_rotate = if need_pad {
                        // Pad query vector with zeros to match padded compressed keys
                        let mut padded = vec![0.0f32; pdim];
                        padded[..hdim].copy_from_slice(q_vec);
                        padded
                    } else {
                        q_vec.to_vec()
                    };
                    let rotated_q = tq_kv::pre_rotate_query_with_signs(&q_for_rotate, &self.signs);
                    let compressed = &cache.k_per_head[kv_h];
                    tq_kv::fused_attention_scores(&rotated_q, compressed, centroids, scale)
                }).collect();

                let mut att_scores = Vec::with_capacity(self.n_head * total_len);
                for scores in &head_scores {
                    att_scores.extend_from_slice(scores);
                }

                // f32 softmax to prevent precision loss at long context
                let att = Tensor::from_vec(
                    att_scores, (b_sz, self.n_head, 1, total_len), q.device(),
                )?;
                let att = softmax_last_dim(&att)?.to_dtype(cache.dtype)?;
                let v_for_attn = repeat_kv(cache.v.clone(), n_rep)?;
                att.matmul(&v_for_attn.contiguous()?)?
            } else if seq_len == 1 && total_len > 1 {
                // GPU GENERATION: decompress + f32 attention
                // Decompress keys (padded_head_dim), then truncate to head_dim
                let k_full = decompress_cache_to_tensor(
                    &cache.k_per_head, self.n_kv_head, total_len,
                    self.padded_head_dim, DType::F32, q.device(), &self.tq_config,
                )?;
                let k_full = if self.padded_head_dim > self.head_dim {
                    k_full.narrow(3, 0, self.head_dim)?
                } else {
                    k_full
                };
                let q_f32 = q.to_dtype(DType::F32)?;
                let k_full = repeat_kv(k_full, n_rep)?;
                let v_f32 = repeat_kv(cache.v.to_dtype(DType::F32)?, n_rep)?;
                let att = (q_f32.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;
                let att = softmax_last_dim(&att)?;
                att.matmul(&v_f32.contiguous()?)?.to_dtype(cache.dtype)?
            } else {
                // PREFILL
                let k = repeat_kv(k, n_rep)?;
                let v_for_attn = repeat_kv(v, n_rep)?;
                let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                let att = match mask {
                    None => att,
                    Some(mask) => {
                        let mask = mask.broadcast_as(att.shape())?;
                        masked_fill(&att, &mask, &self.neg_inf)?
                    }
                };
                let att = softmax_last_dim(&att)?;
                att.matmul(&v_for_attn.contiguous()?)?
            };

            let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
            self.attention_wo.forward(&y)
        } else {
            // UNCOMPRESSED PATH: standard fp16 KV cache (first N layers)
            let (k, v) = match &self.kv_cache {
                None => (k, v),
                Some((prev_k, prev_v)) => {
                    if index_pos == 0 {
                        (k, v)
                    } else {
                        let k = Tensor::cat(&[prev_k, &k], 2)?;
                        let v = Tensor::cat(&[prev_v, &v], 2)?;
                        (k, v)
                    }
                }
            };
            self.kv_cache = Some((k.clone(), v.clone()));

            let k = repeat_kv(k, n_rep)?;
            let v = repeat_kv(v, n_rep)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = softmax_last_dim(&att)?;
            let y = att.matmul(&v.contiguous()?)?;

            let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
            self.attention_wo.forward(&y)
        }
    }
}

// ============================================================
// Generic TurboQuant Model
// ============================================================

#[derive(Debug, Clone)]
pub struct GenericTurboModel {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
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
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
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
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Read architecture from GGUF metadata
        let arch = md_get("general.architecture")
            .and_then(|v| v.to_string().map_err(|e| candle_core::Error::Msg(format!("{e:?}"))))
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
            candle_core::bail!(
                "TurboQuant requires power-of-2 head_dim, but this model has head_dim={}. \
                 Models with non-standard head dimensions (Phi-3.5, etc.) are not supported \
                 for KV compression. Run without --turbo-quant for these models.",
                head_dim,
            );
        }

        // RoPE dimension: some models (llama) specify it explicitly, others use head_dim
        let rope_dim = md_get(&format!("{arch}.rope.dimension_count"))
            .and_then(|m| m.to_u32())
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let rope_style = detect_rope_style(&arch);

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
        eprintln!(
            "  qkv={}, mlp={}, post_attn_norm={}, post_ffn_norm={}",
            qkv_style, mlp_style, has_post_attn_norm, has_post_ffn_norm,
        );

        // Pre-compute shared state
        let signs = tq_kv::hadamard::generate_signs(padded_head_dim, tq_config.rotation_seed);
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Embeddings + output
        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?, rms_norm_eps, device,
        )?;
        // Detect tie_word_embeddings: if output.weight is missing, reuse token embeddings
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => {
                eprintln!("  (tie_word_embeddings: reusing token_embd.weight for output)");
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
                QkvWeights::Separate {
                    wq: QMatMul::from_qtensor(wq)?,
                    wk: QMatMul::from_qtensor(wk)?,
                    wv: QMatMul::from_qtensor(wv)?,
                }
            };
            let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            // Optional biases (Qwen2 has them, Llama/Phi/Gemma don't)
            let attention_bq = if has_bias {
                Some(ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?.dequantize(device)?)
            } else {
                None
            };
            let attention_bk = if has_bias {
                Some(ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?.dequantize(device)?)
            } else {
                None
            };
            let attention_bv = if has_bias {
                Some(ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?.dequantize(device)?)
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
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps, device)?,
                post_attention_norm,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps, device)?,
                post_ffn_norm,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                padded_head_dim,
                rope_dim,
                rope_style,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                layer_idx,
                kv_cache: None,
                kv_compressed: None,
                tq_config: tq_config.clone(),
                signs: signs.clone(),
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            // Optional post-attention norm (Gemma2)
            let attn = match &layer.post_attention_norm {
                Some(norm) => norm.forward(&attn)?,
                None => attn,
            };
            let x = (attn + residual)?;

            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            // Optional post-FFN norm (Gemma2)
            let x = match &layer.post_ffn_norm {
                Some(norm) => norm.forward(&x)?,
                None => x,
            };
            let x = (x + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}
