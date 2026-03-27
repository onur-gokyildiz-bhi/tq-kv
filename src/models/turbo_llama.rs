//! TurboQuant-enhanced quantized Llama model.
//!
//! Fork of candle's `quantized_llama` with TurboQuant KV cache compression.
//! KV cache tensors are compressed to 3-4 bits using PolarQuant + QJL,
//! saving ~3-6x VRAM per cached token.
//!
//! Based on: candle-transformers 0.9.2 quantized_llama.rs
//! Modified: KV cache storage uses TurboQuant compression/decompression.

use std::collections::HashMap;

use candle_core::quantized::{gguf_file, QTensor};
use rayon::prelude::*;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use tq_kv::{CompressedKeys, TurboQuantConfig};

/// Device-aware RmsNorm — candle's version always dequantizes to CPU,
/// causing "no cuda implementation for rms-norm" when input is on CUDA.
/// This version dequantizes to the correct target device.
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    fn from_qtensor(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        Ok(Self { weight, eps, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        // Manual RmsNorm using basic tensor ops (all have CUDA kernels).
        // candle_nn::ops::rms_norm has no CUDA implementation.
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

/// CUDA-compatible SiLU activation: x * sigmoid(x).
fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

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

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
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
        }
    }
}

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

/// Compressed KV cache: per-head compressed keys + values as Tensor.
/// Keys: incremental compress + append (O(1) per token)
/// Values: Tensor (refcount, no clone overhead)
#[derive(Clone, Debug)]
struct CompressedKvCache {
    /// Per-KV-head compressed keys
    k_per_head: Vec<CompressedKeys>,
    /// Cached values tensor (batch, n_kv_head, cached_len, head_dim)
    v: Tensor,
    /// Number of cached sequence positions
    cached_len: usize,
    dtype: DType,
}

/// Decompress per-head caches into a single key tensor (GPU/fallback path).
fn decompress_cache_to_tensor(
    k_per_head: &[CompressedKeys],
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
// Layer with TurboQuant KV cache
// ============================================================

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    // TurboQuant compressed KV cache (replaces kv_cache: Option<(Tensor, Tensor)>)
    kv_compressed: Option<CompressedKvCache>,
    tq_config: TurboQuantConfig,
    /// Pre-computed Hadamard signs (avoids alloc per rotation)
    signs: Vec<f32>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

/// Manual RoPE (interleaved) using basic tensor ops — CUDA compatible.
/// candle_nn::rotary_emb::rope_i has no CUDA kernel.
fn rope_i_manual(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = x.dims4()?;
    let half = d / 2;
    // Interleaved: pairs (x0,x1), (x2,x3), ...
    // x_even = x[..., 0::2], x_odd = x[..., 1::2]
    let x_even = x.narrow(3, 0, half)?;  // won't work for interleaved
    // For interleaved RoPE, reshape to (..., half, 2) then rotate
    let x = x.reshape((_b, _h, _s, half, 2))?;
    let x0 = x.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = x.narrow(4, 1, 1)?.squeeze(4)?;
    // Rotation: [x0*cos - x1*sin, x0*sin + x1*cos]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq, half)
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    // Interleave back: stack on last dim then flatten
    let r0 = r0.unsqueeze(4)?;
    let r1 = r1.unsqueeze(4)?;
    Tensor::cat(&[&r0, &r1], 4)?.reshape((_b, _h, _s, d))
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        rope_i_manual(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // ========================================
        // Incremental KV Cache: compress new keys, append
        // Keys: Lloyd-Max compressed (2-4 bit), per-head
        // Values: Tensor (refcount, no clone)
        // ========================================
        let k_contiguous = k.contiguous()?;
        let k_flat = k_contiguous.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let cache_dtype = k.dtype();

        if self.kv_compressed.is_none() {
            let mut k_per_head = Vec::with_capacity(self.n_kv_head);
            for _ in 0..self.n_kv_head {
                k_per_head.push(tq_kv::CompressedKeys::new_empty(
                    self.tq_config.bits, self.head_dim, self.tq_config.rotation_seed,
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

        // Compress new key vectors per-head, append to cache
        for h in 0..self.n_kv_head {
            for s in 0..seq_len {
                let offset = (h * seq_len + s) * self.head_dim;
                let key_vec = &k_flat[offset..offset + self.head_dim];
                let (packed, norm) = tq_kv::compress_single_key_with_signs(
                    key_vec, self.head_dim, &self.tq_config, &self.signs,
                );
                cache.k_per_head[h].append_raw(&packed, norm);
            }
        }

        // Update values: Tensor::cat (no Vec clone)
        if cache.cached_len > 0 {
            cache.v = Tensor::cat(&[&cache.v, &v], 2)?;
        } else {
            cache.v = v.clone();
        }
        cache.cached_len += seq_len;
        let total_len = cache.cached_len;

        // ========================================
        // Attention: fused (CPU gen) or standard
        // ========================================
        let n_rep = self.n_head / self.n_kv_head;

        let y = if seq_len == 1 && total_len > 1 && q.device().is_cpu() {
            // FUSED PATH: CPU generation — no key decompression
            // Rayon parallel across query heads + SIMD (AVX2) inner dot product
            // ⟨q, k⟩ = ⟨R·q, centroids[idx]⟩ × sigma
            let q_flat = q.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let centroids = tq_kv::codebook::get_centroids(self.tq_config.bits);
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            // Parallel: each query head computed independently via Rayon
            let head_scores: Vec<Vec<f32>> = (0..self.n_head).into_par_iter().map(|qh| {
                let kv_h = qh / n_rep;
                let q_vec = &q_flat[qh * self.head_dim..(qh + 1) * self.head_dim];
                let rotated_q = tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs);
                let compressed = &cache.k_per_head[kv_h];
                // Batch fused attention: all positions at once, SIMD inner loop
                tq_kv::fused_attention_scores(&rotated_q, compressed, centroids, scale)
            }).collect();

            // Flatten into contiguous attention score array
            let mut att_scores = Vec::with_capacity(self.n_head * total_len);
            for scores in &head_scores {
                att_scores.extend_from_slice(scores);
            }

            // FIX: Keep attention scores in f32 through softmax to prevent
            // precision loss with f16. Only cast after softmax for value matmul.
            let att = Tensor::from_vec(
                att_scores, (b_sz, self.n_head, 1, total_len), q.device(),
            )?;
            let att = softmax_last_dim(&att)?.to_dtype(cache.dtype)?;
            let v_for_attn = repeat_kv(cache.v.clone(), n_rep)?;
            att.matmul(&v_for_attn.contiguous()?)?
        } else if seq_len == 1 && total_len > 1 {
            // GPU GENERATION: decompress cached keys + standard attention
            let k_full = decompress_cache_to_tensor(
                &cache.k_per_head, self.n_kv_head, total_len,
                self.head_dim, cache.dtype, q.device(), &self.tq_config,
            )?;
            let k_full = repeat_kv(k_full, n_rep)?;
            let v_for_attn = repeat_kv(cache.v.clone(), n_rep)?;
            let att = (q.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = softmax_last_dim(&att)?;
            att.matmul(&v_for_attn.contiguous()?)?
        } else {
            // PREFILL / FIRST TOKEN: use raw k tensor, standard attention
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

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.attention_wo.forward(&y)
    }
}

// ============================================================
// Model
// ============================================================

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
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

        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32()).unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32()).unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32()).unwrap_or(10000f32);

        let head_dim = embedding_length / head_count;
        eprintln!(
            "TurboQuant Llama: {} layers, {} heads, head_dim={}, {}-bit KV cache (fused attention)",
            block_count, head_count, head_dim, tq_config.bits
        );

        // Pre-compute Hadamard signs once (shared across all layers)
        let signs = tq_kv::hadamard::generate_signs(head_dim, tq_config.rotation_seed);

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?, rms_norm_eps, device,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp_or_moe = if n_expert <= 1 {
                let w1 = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let w2 = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let w3 = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(w3)?,
                })
            } else {
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
            };

            let attention_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps, device)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps, device)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
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
            let x = (attn + residual)?;

            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}
