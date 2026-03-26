//! TurboQuant-enhanced quantized Qwen2 model.
//!
//! Fork of candle's `quantized_qwen2` with TurboQuant KV cache compression.
//! Based on: candle-transformers 0.9.2 quantized_qwen2.rs

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use rayon::prelude::*;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use tq_kv::TurboQuantConfig;

/// Device-aware RmsNorm (candle's version always dequantizes to CPU → CUDA crash).
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
// Layer with TurboQuant KV cache
// ============================================================

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_compressed: Option<CompressedKvCache>,
    tq_config: TurboQuantConfig,
    signs: Vec<f32>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn rope_i_manual(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = x.dims4()?;
    let half = d / 2;
    let x = x.reshape((_b, _h, _s, half, 2))?;
    let x0 = x.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = x.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
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

        // Incremental KV Cache
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

        if cache.cached_len > 0 {
            cache.v = Tensor::cat(&[&cache.v, &v], 2)?;
        } else {
            cache.v = v.clone();
        }
        cache.cached_len += seq_len;
        let total_len = cache.cached_len;

        let n_rep = self.n_head / self.n_kv_head;

        let y = if seq_len == 1 && total_len > 1 && q.device().is_cpu() {
            // FUSED PATH: Rayon parallel + SIMD
            let q_flat = q.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let centroids = tq_kv::codebook::get_centroids(self.tq_config.bits);
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            let head_scores: Vec<Vec<f32>> = (0..self.n_head).into_par_iter().map(|qh| {
                let kv_h = qh / n_rep;
                let q_vec = &q_flat[qh * self.head_dim..(qh + 1) * self.head_dim];
                let rotated_q = tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs);
                let compressed = &cache.k_per_head[kv_h];
                tq_kv::fused_attention_scores(&rotated_q, compressed, centroids, scale)
            }).collect();

            let mut att_scores = Vec::with_capacity(self.n_head * total_len);
            for scores in &head_scores {
                att_scores.extend_from_slice(scores);
            }

            let att = Tensor::from_vec(
                att_scores, (b_sz, self.n_head, 1, total_len), q.device(),
            )?.to_dtype(cache.dtype)?;
            let att = softmax_last_dim(&att)?;
            let v_for_attn = repeat_kv(cache.v.clone(), n_rep)?;
            att.matmul(&v_for_attn.contiguous()?)?
        } else if seq_len == 1 && total_len > 1 {
            // GPU GENERATION: decompress + standard attention
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
            // PREFILL / FIRST TOKEN
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
    Ok((idx_theta.cos()?, idx_theta.sin()?))
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

        let head_count = md_get("qwen2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("qwen2.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("qwen2.embedding_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen2.rope.freq_base")
            .and_then(|m| m.to_f32()).unwrap_or(1000000f32);

        let head_dim = embedding_length / head_count;
        eprintln!(
            "TurboQuant Qwen2: {} layers, {} heads, head_dim={}, {}-bit KV cache (fused attention)",
            block_count, head_count, head_dim, tq_config.bits
        );

        let signs = tq_kv::hadamard::generate_signs(head_dim, tq_config.rotation_seed);

        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, device)?;
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
            let w1 = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let w2 = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            let w3 = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let attention_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps, device)?,
                mlp: Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(w3)?,
                },
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
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}
