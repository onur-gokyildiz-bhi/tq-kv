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
        normalized.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?.to_dtype(x_dtype)
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

    fn from_tensor(tensor: Tensor) -> Self {
        let inner = candle_core::quantized::QMatMul::Tensor(tensor);
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Self { inner, span }
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

/// Number of initial "sink" tokens whose keys are kept in FP16 (uncompressed).
/// Attention sink tokens receive disproportionate attention weight — quantizing them
/// causes up to 81% of total attention error. (KVSink, arXiv:2508.04257)
/// Override with TQ_SINK env var.
const TQ_SINK_TOKENS: usize = 4;

fn get_sink_tokens() -> usize {
    std::env::var("TQ_SINK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TQ_SINK_TOKENS)
}

#[derive(Clone, Debug)]
struct CompressedKvCache {
    /// Compressed keys per KV head (hot tier, recent tokens at original bit width)
    k_per_head: Vec<tq_kv::CompressedKeys>,
    /// Cold (decayed) keys per head — lower bit width, older tokens.
    /// Order: cold tokens come BEFORE hot tokens in sequence position.
    k_cold: Option<Vec<tq_kv::CompressedKeys>>,
    /// How many tokens are in the cold tier (same across all heads)
    cold_len: usize,
    /// Temporal decay config (None = disabled)
    decay_config: Option<tq_kv::TemporalDecayConfig>,
    /// Tokens since last decay check
    tokens_since_decay: usize,
    /// Uncompressed sink token keys: [1, n_kv_head, sink_len, head_dim]
    sink_k: Option<Tensor>,
    /// Number of sink tokens stored
    sink_len: usize,
    /// Value cache — uncompressed path (value_bits=0)
    v_raw: Option<Tensor>,
    /// Value cache — compressed path (value_bits=8), per KV head
    v_compressed: Option<Vec<tq_kv::CompressedValues>>,
    /// Value quantization bits (0=fp16, 8=absmax)
    value_bits: u8,
    /// Total cached length (sink + compressed + current)
    cached_len: usize,
    dtype: DType,
}

/// Decompress compressed keys to tensor. Only decompresses the compressed portion.
fn decompress_compressed_keys(
    k_per_head: &[tq_kv::CompressedKeys],
    n_kv_head: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
    config: &TurboQuantConfig,
) -> Result<Tensor> {
    let compressed_len = if k_per_head.is_empty() || k_per_head[0].count == 0 {
        return Tensor::zeros((1, n_kv_head, 0, head_dim), dtype, device);
    } else {
        k_per_head[0].count
    };
    let mut all_data = Vec::with_capacity(n_kv_head * compressed_len * head_dim);
    for compressed in k_per_head.iter().take(n_kv_head) {
        let decompressed = if compressed.group_size > 0 {
            tq_kv::decompress_keys_grouped(compressed, config)
        } else {
            tq_kv::decompress_keys(compressed, config)
        };
        all_data.extend(decompressed);
    }
    Tensor::from_vec(all_data, (1, n_kv_head, compressed_len, head_dim), device)?.to_dtype(dtype)
}

/// Decompress compressed values to F32 tensor: (1, n_kv_head, seq_len, head_dim).
fn decompress_values(
    v_per_head: &[tq_kv::CompressedValues],
    n_kv_head: usize,
    head_dim: usize,
    seq_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut all_data = Vec::with_capacity(n_kv_head * seq_len * head_dim);
    for compressed in v_per_head.iter().take(n_kv_head) {
        let decompressed = compressed.decompress();
        all_data.extend(decompressed);
    }
    Tensor::from_vec(all_data, (1, n_kv_head, seq_len, head_dim), device)
}

/// Sparse attention-value multiply on CPU tensors.
///
/// att shape: (1, n_heads, 1, seq_len)  — softmax weights for single query token
/// v shape:   (1, n_heads, seq_len, head_dim) — value cache
///
/// Returns: (1, n_heads, 1, head_dim) — same as att.matmul(&v) but skipping
/// V rows where the softmax weight < threshold.
fn sparse_attn_v(
    att: &Tensor,
    v: &Tensor,
    n_heads: usize,
    head_dim: usize,
    threshold: f32,
) -> Result<Tensor> {
    let att_flat = att.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let v_flat = v.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let seq_len = att.dim(3)?;

    let mut output = Vec::with_capacity(n_heads * head_dim);
    for h in 0..n_heads {
        let att_row = &att_flat[h * seq_len..(h + 1) * seq_len];
        let v_block = &v_flat[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let head_out = tq_kv::sparse_attn_v_mul(att_row, v_block, head_dim, threshold);
        output.extend_from_slice(&head_out);
    }

    Tensor::from_vec(output, (1, n_heads, 1, head_dim), att.device())
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
/// Override with TQ_SKIP env var (e.g. TQ_SKIP=8 for first 8 layers uncompressed).
const TQ_SKIP_FIRST_LAYERS: usize = 4;

fn get_skip_layers() -> usize {
    std::env::var("TQ_SKIP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TQ_SKIP_FIRST_LAYERS)
}

/// Layer-adaptive bitwidth: assign different bit widths to different layer ranges.
/// Format: "start-end:bits[,start-end:bits]" e.g. "4-15:2,16-27:4"
/// Unspecified layers use the default TQ bits. Layers below TQ_SKIP are uncompressed.
/// Override with TQ_LAYER_BITS env var.
fn get_layer_bits(layer_idx: usize, default_bits: u8) -> Option<u8> {
    let skip = get_skip_layers();
    if layer_idx < skip {
        return None; // uncompressed
    }

    if let Ok(val) = std::env::var("TQ_LAYER_BITS") {
        for part in val.split(',') {
            let parts: Vec<&str> = part.trim().split(':').collect();
            if parts.len() == 2 {
                let range_parts: Vec<&str> = parts[0].split('-').collect();
                if range_parts.len() == 2 {
                    if let (Ok(start), Ok(end), Ok(bits)) = (
                        range_parts[0].parse::<usize>(),
                        range_parts[1].parse::<usize>(),
                        parts[1].parse::<u8>(),
                    ) {
                        if layer_idx >= start && layer_idx <= end {
                            return Some(bits);
                        }
                    }
                }
            }
        }
    }

    Some(default_bits)
}

/// Sparse V threshold. Softmax weights below this are skipped in V multiply.
/// Set TQ_SPARSE_V=0 to disable. Default: 1e-6.
/// Override with TQ_SPARSE_V env var (e.g. TQ_SPARSE_V=1e-5).
const TQ_SPARSE_V_DEFAULT: f32 = 1e-6;

fn get_sparse_v_threshold() -> f32 {
    std::env::var("TQ_SPARSE_V")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TQ_SPARSE_V_DEFAULT)
}

/// Fused attention: compute attention scores directly from compressed indices
/// instead of decompressing keys first. Saves memory bandwidth on CPU.
/// Set TQ_FUSED=1 to enable. Default: off (decompress path).
fn get_use_fused() -> bool {
    std::env::var("TQ_FUSED")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Softmax bias correction: pre-compensate quantization-induced attention drift.
/// Set TQ_BIAS_CORRECT=1 to enable. Default: off.
fn get_bias_correction() -> bool {
    std::env::var("TQ_BIAS_CORRECT")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Value cache quantization bits. 0 = uncompressed fp16, 8 = 8-bit absmax.
/// Override with TQ_VBITS env var (e.g. TQ_VBITS=8 for 2x value savings).
const TQ_VBITS_DEFAULT: u8 = 0;

fn get_value_bits() -> u8 {
    std::env::var("TQ_VBITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TQ_VBITS_DEFAULT)
}

/// Temporal decay: demote old tokens to lower bit widths.
/// Format: "age:bits[,age:bits]" e.g. "512:2" or "256:3,1024:2"
/// Set TQ_DECAY=off to disable. Default: off.
fn get_decay_config() -> Option<tq_kv::TemporalDecayConfig> {
    let val = std::env::var("TQ_DECAY").ok()?;
    if val == "off" || val == "0" || val.is_empty() { return None; }
    let mut tiers = Vec::new();
    for part in val.split(',') {
        let parts: Vec<&str> = part.trim().split(':').collect();
        if parts.len() == 2 {
            if let (Ok(age), Ok(bits)) = (parts[0].parse::<usize>(), parts[1].parse::<u8>()) {
                tiers.push(tq_kv::DecayTier { age_threshold: age, bits });
            }
        }
    }
    if tiers.is_empty() { return None; }
    tiers.sort_by_key(|t| t.age_threshold);
    Some(tq_kv::TemporalDecayConfig { tiers, decay_interval: 128 })
}

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
        let layer_bits = get_layer_bits(self.layer_idx, self.tq_config.bits);
        let use_compression = layer_bits.is_some();
        let n_rep = self.n_head / self.n_kv_head;

        if use_compression {
            // Apply per-layer bit width if different from default
            let effective_bits = layer_bits.unwrap();
            let layer_tq_config = if effective_bits != self.tq_config.bits {
                tq_kv::TurboQuantConfig { bits: effective_bits, ..self.tq_config.clone() }
            } else {
                self.tq_config.clone()
            };
            // COMPRESSED PATH: TurboQuant KV cache with 3-fix quality enhancement
            //
            // Three paper-backed techniques to handle compound error (W4+KV4):
            //   Fix 1 — Sink token preservation: first N tokens' keys stay FP16 (KVSink)
            //   Fix 2 — Past-Only Quantization: current token's key is FP16 during
            //           attention, compressed into cache AFTER (WKVQuant)
            //   Fix 3 — Per-channel scaling: (future — placeholder for now)

            // Reset cache on new sequence
            if index_pos == 0 {
                self.kv_compressed = None;
            }

            let cache_dtype = k.dtype();
            let sink_n = get_sink_tokens();

            // Initialize cache on first call
            let vbits = get_value_bits();
            if self.kv_compressed.is_none() {
                let mut k_per_head = Vec::with_capacity(self.n_kv_head);
                let gs = layer_tq_config.group_size;
                for _ in 0..self.n_kv_head {
                    k_per_head.push(tq_kv::CompressedKeys::new_empty_grouped(
                        effective_bits, self.padded_head_dim, layer_tq_config.rotation_seed, gs,
                    ));
                }
                let (v_raw, v_compressed) = if vbits == 0 {
                    (None, None)
                } else {
                    let mut v_heads: Vec<tq_kv::CompressedValues> = Vec::with_capacity(self.n_kv_head);
                    for _ in 0..self.n_kv_head {
                        v_heads.push(tq_kv::CompressedValues::new_empty(self.head_dim));
                    }
                    (None, Some(v_heads))
                };
                self.kv_compressed = Some(CompressedKvCache {
                    k_per_head,
                    k_cold: None,
                    cold_len: 0,
                    decay_config: get_decay_config(),
                    tokens_since_decay: 0,
                    sink_k: None,
                    sink_len: 0,
                    v_raw,
                    v_compressed,
                    value_bits: vbits,
                    cached_len: 0,
                    dtype: cache_dtype,
                });
            }

            let cache = self.kv_compressed.as_mut().unwrap();
            let prev_total = cache.cached_len;

            // Determine which tokens in this batch are sink vs compressible
            // Sink tokens: positions [0, sink_n) in the sequence
            // POQ: during generation (seq_len=1), current token is NOT compressed yet
            let global_start = prev_total; // first position in this batch

            // --- Store values ---
            if cache.value_bits == 0 {
                // Uncompressed fp16 path
                if cache.cached_len > 0 {
                    cache.v_raw = Some(match &cache.v_raw {
                        Some(prev) => Tensor::cat(&[prev, &v], 2)?,
                        None => v.clone(),
                    });
                } else {
                    cache.v_raw = Some(v.clone());
                }
            } else {
                // Compressed 8-bit path: quantize per head × position
                let v_f32 = v.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                let v_comp = cache.v_compressed.as_mut().unwrap();
                for h in 0..self.n_kv_head {
                    for s in 0..seq_len {
                        let offset = (h * seq_len + s) * self.head_dim;
                        v_comp[h].append(&v_f32[offset..offset + self.head_dim]);
                    }
                }
            }

            // --- Handle sink tokens (FP16, uncompressed) ---
            let sink_end = sink_n.min(global_start + seq_len);
            let new_sink_count = if global_start < sink_n {
                // Some tokens in this batch are sink tokens
                let n_sink_in_batch = sink_end - global_start;
                // Extract sink portion of k: k is [1, n_kv_head, seq_len, head_dim]
                let sink_k_batch = if n_sink_in_batch < seq_len {
                    k.narrow(2, 0, n_sink_in_batch)?
                } else {
                    k.clone()
                };
                cache.sink_k = Some(match &cache.sink_k {
                    Some(prev) => Tensor::cat(&[prev, &sink_k_batch], 2)?,
                    None => sink_k_batch,
                });
                cache.sink_len += n_sink_in_batch;
                n_sink_in_batch
            } else {
                0
            };

            // --- Compress non-sink tokens into cache ---
            // All tokens after sink position get compressed into cache.
            // POQ twist: during generation (seq_len=1), the current token is ALSO
            // compressed into cache, but during attention we use the FP16 original
            // instead of the decompressed version (Fix 2: Past-Only Quantization).
            let compress_start = if global_start < sink_n { sink_end - global_start } else { 0 };
            let tokens_to_compress = seq_len.saturating_sub(compress_start);

            if tokens_to_compress > 0 {
                let k_to_compress = k.narrow(2, compress_start, tokens_to_compress)?;
                let k_flat = k_to_compress.contiguous()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let hdim = self.head_dim;
                let use_grouped = layer_tq_config.group_size > 0 && hdim % layer_tq_config.group_size == 0;
                for h in 0..self.n_kv_head {
                    for s in 0..tokens_to_compress {
                        let offset = (h * tokens_to_compress + s) * hdim;
                        let key_vec = &k_flat[offset..offset + hdim];
                        if use_grouped {
                            let (packed, gnorms, residual, outliers) = tq_kv::compress_single_key_grouped(
                                key_vec, hdim, &layer_tq_config, &self.signs,
                            );
                            // Set residual/outlier bits on first append
                            if cache.k_per_head[h].residual_bits == 0 && residual.is_some() {
                                cache.k_per_head[h].residual_bits = layer_tq_config.residual_bits;
                            }
                            if cache.k_per_head[h].outlier_k == 0 && outliers.is_some() {
                                cache.k_per_head[h].outlier_k = layer_tq_config.outlier_k;
                            }
                            // Append outliers
                            if let Some((oi, ov)) = outliers {
                                if cache.k_per_head[h].outlier_indices.is_none() {
                                    cache.k_per_head[h].outlier_indices = Some(Vec::new());
                                    cache.k_per_head[h].outlier_values = Some(Vec::new());
                                }
                                cache.k_per_head[h].outlier_indices.as_mut().unwrap().extend_from_slice(&oi);
                                cache.k_per_head[h].outlier_values.as_mut().unwrap().extend_from_slice(&ov);
                            }
                            cache.k_per_head[h].append_raw_grouped(&packed, &gnorms, residual);
                        } else if self.padded_head_dim > hdim {
                            let mut padded = vec![0.0f32; self.padded_head_dim];
                            padded[..hdim].copy_from_slice(key_vec);
                            let (packed, norm) = tq_kv::compress_single_key_with_signs(
                                &padded, self.padded_head_dim, &layer_tq_config, &self.signs,
                            );
                            cache.k_per_head[h].append_raw(&packed, norm);
                        } else {
                            let (packed, norm) = tq_kv::compress_single_key_with_signs(
                                key_vec, hdim, &layer_tq_config, &self.signs,
                            );
                            cache.k_per_head[h].append_raw(&packed, norm);
                        }
                    }
                }
            }

            cache.cached_len += seq_len;
            cache.tokens_since_decay += seq_len;
            let total_len = cache.cached_len;

            // --- Temporal decay: demote old hot tokens to lower bit width ---
            if let Some(ref decay_cfg) = cache.decay_config.clone() {
                if cache.tokens_since_decay >= decay_cfg.decay_interval {
                    cache.tokens_since_decay = 0;
                    // Find the lowest tier that applies (smallest bits)
                    // Use the first (and typically only) tier for simplicity
                    if let Some(tier) = decay_cfg.tiers.first() {
                        let hot_count = cache.k_per_head[0].count;
                        // Tokens eligible for decay: those older than age_threshold
                        // hot tokens span positions [sink_len .. sink_len + hot_count]
                        // "age" of the oldest hot token = total_len - sink_len - 0 = hot_count + cold_len
                        // We want to decay tokens whose age > threshold
                        // That means the first (hot_count + cold_len - threshold) hot tokens
                        let total_compressed = hot_count + cache.cold_len;
                        if total_compressed > tier.age_threshold && hot_count > 0 {
                            let to_decay = total_compressed - tier.age_threshold;
                            let to_decay = to_decay.min(hot_count); // can't decay more than we have
                            if to_decay > 0 {
                                // Initialize cold tier if needed
                                if cache.k_cold.is_none() {
                                    let mut cold = Vec::with_capacity(self.n_kv_head);
                                    for _ in 0..self.n_kv_head {
                                        cold.push(tq_kv::CompressedKeys::new_empty(
                                            tier.bits, self.padded_head_dim, layer_tq_config.rotation_seed,
                                        ));
                                    }
                                    cache.k_cold = Some(cold);
                                }
                                // Split front of hot, remap to cold bits, append to cold
                                let cold = cache.k_cold.as_mut().unwrap();
                                for h in 0..self.n_kv_head {
                                    let front = cache.k_per_head[h].split_off_front(to_decay);
                                    let remapped = front.remap_bits(tier.bits);
                                    cold[h].append_from(&remapped);
                                }
                                cache.cold_len += to_decay;
                            }
                        }
                    }
                }
            }

            // --- Build full key tensor for attention ---
            // Concatenate: [sink_keys (FP16) | cold_keys (decayed) | hot_keys (original bits) | current_key (FP16)]
            // This is the POQ + Sink + Decay approach: highest-impact tokens are lossless

            let y = if seq_len == 1 && total_len > 1 {
                // GENERATION: single token attention against full cache
                //
                // POQ (Past-Only Quantization): for the current token's key, we use
                // the FP16 original in attention instead of the compressed version.
                // The compressed version IS in the cache (for future tokens), but
                // we replace the last position with the lossless original.
                let q_f32 = q.to_dtype(DType::F32)?;

                let use_fused = get_use_fused() && q.device().is_cpu();
                let n_compressed = cache.k_per_head[0].count;
                let n_past_compressed = if n_compressed > 0 { n_compressed - 1 } else { 0 };

                // --- Compute attention scores ---
                let att = if use_fused {
                    // FUSED PATH: compute scores directly from compressed indices.
                    // No key decompression — saves memory bandwidth.
                    // Scores are computed per query-head using pre-rotated query
                    // and centroid table lookup (AVX2 SIMD when available).
                    let q_flat = q_f32.flatten_all()?.to_vec1::<f32>()?;
                    let scale = 1.0 / (self.head_dim as f32).sqrt();
                    let cold_centroids = cache.k_cold.as_ref()
                        .map(|c| tq_kv::codebook::get_centroids(c[0].bits));

                    use rayon::prelude::*;
                    let head_scores: Vec<Vec<f32>> = (0..self.n_head)
                        .into_par_iter()
                        .map(|qh| {
                            let kv_h = qh / n_rep;
                            let q_vec = &q_flat[qh * self.head_dim..(qh + 1) * self.head_dim];
                            let rotated_q = tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs);
                            let mut scores = Vec::with_capacity(total_len);

                            // Segment 1: Sink keys (standard dot product, not compressed)
                            if let Some(ref sink) = cache.sink_k {
                                let sink_f32 = sink.to_dtype(DType::F32).unwrap();
                                let sink_flat = sink_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                let sink_count = cache.sink_len;
                                for s in 0..sink_count {
                                    let offset = (kv_h * sink_count + s) * self.head_dim;
                                    let k_vec = &sink_flat[offset..offset + self.head_dim];
                                    let dot: f32 = q_vec.iter().zip(k_vec.iter())
                                        .map(|(&qi, &ki)| qi * ki).sum();
                                    scores.push(dot * scale);
                                }
                            }

                            // Segment 2: Cold (decayed) keys — fused at cold bit width
                            if let Some(ref cold) = cache.k_cold {
                                let cold_cb = cold_centroids.unwrap();
                                let cold_scores = tq_kv::fused_attention_scores(
                                    &rotated_q, &cold[kv_h], cold_cb, scale,
                                );
                                scores.extend_from_slice(&cold_scores);
                            }

                            // Segment 3: Hot compressed keys (excluding last = current)
                            if n_past_compressed > 0 {
                                let hot_cb = tq_kv::codebook::get_centroids(effective_bits);
                                let hot = &cache.k_per_head[kv_h];
                                let dim = hot.dim;
                                let bpv = hot.bytes_per_vector();
                                let mut idx_buf = vec![0u8; dim];
                                for pos in 0..n_past_compressed {
                                    let norm = hot.norms[pos];
                                    if norm < 1e-10 {
                                        scores.push(0.0);
                                        continue;
                                    }
                                    let start = pos * bpv;
                                    let end = start + bpv;
                                    tq_kv::codebook::unpack_indices_into(
                                        &hot.packed_indices[start..end], &mut idx_buf, hot.bits,
                                    );
                                    let score = tq_kv::fused_dot_product_with_centroids(
                                        &rotated_q, &idx_buf, norm, hot_cb, dim,
                                    ) * scale;
                                    scores.push(score);
                                }
                            }

                            // Segment 4: Current token key (FP16 original — POQ)
                            // Only if current token was compressed (not still in sink range)
                            if n_compressed > 0 {
                                let k_f32 = k.to_dtype(DType::F32).unwrap();
                                let k_flat = k_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                                let k_vec = &k_flat[kv_h * self.head_dim..(kv_h + 1) * self.head_dim];
                                let dot: f32 = q_vec.iter().zip(k_vec.iter())
                                    .map(|(&qi, &ki)| qi * ki).sum();
                                scores.push(dot * scale);
                            }

                            scores
                        })
                        .collect();

                    let mut all_scores = Vec::with_capacity(self.n_head * total_len);
                    for s in &head_scores {
                        all_scores.extend_from_slice(s);
                    }
                    let att = Tensor::from_vec(
                        all_scores, (1, self.n_head, 1, total_len), q.device(),
                    )?;
                    softmax_last_dim(&att)?
                } else {
                    // DECOMPRESS PATH: decompress all keys, standard matmul
                    let mut k_parts: Vec<Tensor> = Vec::new();

                    // Part 1: Sink keys (FP16, lossless)
                    if let Some(ref sink) = cache.sink_k {
                        k_parts.push(sink.to_dtype(DType::F32)?);
                    }

                    // Part 1.5: Cold (decayed) keys
                    if let Some(ref cold) = cache.k_cold {
                        if cold[0].count > 0 {
                            let k_cold = decompress_compressed_keys(
                                cold, self.n_kv_head,
                                self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                            )?;
                            let k_cold = if self.padded_head_dim > self.head_dim {
                                k_cold.narrow(3, 0, self.head_dim)?
                            } else {
                                k_cold
                            };
                            k_parts.push(k_cold);
                        }
                    }

                    // Part 2: Hot compressed keys (excluding last = current token)
                    if n_past_compressed > 0 {
                        let k_decomp = decompress_compressed_keys(
                            &cache.k_per_head, self.n_kv_head,
                            self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                        )?;
                        let k_decomp = if self.padded_head_dim > self.head_dim {
                            k_decomp.narrow(3, 0, self.head_dim)?
                        } else {
                            k_decomp
                        };
                        let k_past = k_decomp.narrow(2, 0, n_past_compressed)?;
                        k_parts.push(k_past);
                    }

                    // Part 3: Current token key (FP16 original — POQ lossless)
                    // Only add if current token was compressed (not still in sink range)
                    if n_compressed > 0 {
                        k_parts.push(k.to_dtype(DType::F32)?);
                    }

                    let k_full = if k_parts.len() == 1 {
                        k_parts.remove(0)
                    } else {
                        Tensor::cat(&k_parts, 2)?
                    };

                    let k_full = repeat_kv(k_full, n_rep)?;
                    let mut att = (q_f32.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;

                    // Softmax bias correction: compensate quantization-induced attention drift
                    if get_bias_correction() && n_past_compressed > 0 {
                        let bias = tq_kv::softmax_bias_correction(
                            &cache.k_per_head[0], self.head_dim,
                        );
                        // Build full bias vector: [sink=0, cold=0, hot_bias, current=0]
                        let mut full_bias = vec![0.0f32; total_len];
                        let hot_start = cache.sink_len + cache.cold_len;
                        for (i, &b) in bias.iter().take(n_past_compressed).enumerate() {
                            full_bias[hot_start + i] = b;
                        }
                        let bias_tensor = Tensor::from_vec(
                            full_bias, (1, 1, 1, total_len), q.device(),
                        )?;
                        att = att.broadcast_add(&bias_tensor)?;
                    }

                    softmax_last_dim(&att)?
                };

                // --- Compute attention output: att @ V ---
                let v_f32 = if cache.value_bits == 0 {
                    repeat_kv(cache.v_raw.as_ref().unwrap().to_dtype(DType::F32)?, n_rep)?
                } else {
                    let v_tensor = decompress_values(
                        cache.v_compressed.as_ref().unwrap(),
                        self.n_kv_head, self.head_dim, total_len, q.device(),
                    )?;
                    repeat_kv(v_tensor, n_rep)?
                };

                // Sparse V: skip V rows where softmax weight < threshold (CPU only)
                let sparse_thresh = get_sparse_v_threshold();
                if sparse_thresh > 0.0 && att.device().is_cpu() {
                    sparse_attn_v(&att, &v_f32, self.n_head, self.head_dim, sparse_thresh)?
                        .to_dtype(cache.dtype)?
                } else {
                    att.matmul(&v_f32.contiguous()?)?.to_dtype(cache.dtype)?
                }
            } else {
                // PREFILL: use original uncompressed keys for attention (standard path)
                // Keys are already compressed into cache above, but attention uses originals
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
        use candle_core::DType;

        // 1. Read config.json
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read config.json: {e}")))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Invalid config.json: {e}")))?;

        let head_count = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let head_count_kv = config["num_key_value_heads"].as_u64().unwrap_or(head_count as u64) as usize;
        let block_count = config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
        let embedding_length = config["hidden_size"].as_u64().unwrap_or(4096) as usize;
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let rope_freq_base = config["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
        let context_length = config["max_position_embeddings"].as_u64().unwrap_or(4096) as usize;
        let head_dim = config["head_dim"].as_u64()
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);

        if !head_dim.is_power_of_two() {
            candle_core::bail!("TurboQuant requires power-of-2 head_dim, got {}", head_dim);
        }

        let arch = config["model_type"].as_str().unwrap_or("llama").to_string();
        let rope_style = detect_rope_style(&arch);
        let rope_dim = head_dim; // safetensors models typically use full head_dim for RoPE

        eprintln!(
            "TurboQuant FP16 [{}]: {} layers, {} heads (kv={}), head_dim={}, emb={}, {}-bit KV cache",
            arch, block_count, head_count, head_count_kv, head_dim, embedding_length, tq_config.bits,
        );

        // 2. Load all safetensors files
        let mut tensors = std::collections::HashMap::new();
        let entries: Vec<_> = std::fs::read_dir(model_dir)
            .map_err(|e| candle_core::Error::Msg(format!("Cannot read model dir: {e}")))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
            .collect();

        if entries.is_empty() {
            candle_core::bail!("No .safetensors files found in {}", model_dir.display());
        }

        for entry in &entries {
            let loaded = candle_core::safetensors::load(entry.path(), device)?;
            tensors.extend(loaded);
        }
        eprintln!("  Loaded {} tensors from {} safetensors file(s)", tensors.len(), entries.len());

        // Helper to get a tensor by name, casting BF16→F32 for CPU compat
        let get = |name: &str| -> Result<Tensor> {
            let t = tensors.get(name)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing tensor: {name}")))?;
            // BF16 has no CPU matmul — cast to F16 (CUDA) or F32 (CPU)
            if t.dtype() == DType::BF16 && device.is_cpu() {
                t.to_dtype(DType::F32)
            } else {
                Ok(t)
            }
        };

        // 3. Pre-compute shared state
        let padded_head_dim = head_dim;
        let signs = tq_kv::hadamard::generate_signs(padded_head_dim, tq_config.rotation_seed);
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // 4. Embeddings + output
        let tok_embeddings = get("model.embed_tokens.weight")?;
        let norm_w = get("model.norm.weight")?;
        let norm = RmsNorm { weight: norm_w, eps: rms_norm_eps, span: tracing::span!(tracing::Level::TRACE, "rms-norm") };

        let output = if let Ok(t) = get("lm_head.weight") {
            QMatMul::from_tensor(t)
        } else {
            eprintln!("  (tie_word_embeddings: reusing embed_tokens.weight for lm_head)");
            QMatMul::from_tensor(tok_embeddings.clone())
        };

        // 5. Load layers
        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("model.layers.{i}");

            let wq = get(&format!("{p}.self_attn.q_proj.weight"))?;
            let wk = get(&format!("{p}.self_attn.k_proj.weight"))?;
            let wv = get(&format!("{p}.self_attn.v_proj.weight"))?;
            let wo = get(&format!("{p}.self_attn.o_proj.weight"))?;

            let qkv = QkvWeights::Separate {
                wq: QMatMul::from_tensor(wq),
                wk: QMatMul::from_tensor(wk),
                wv: QMatMul::from_tensor(wv),
            };

            // Biases (optional — Qwen2 has them)
            let bq = get(&format!("{p}.self_attn.q_proj.bias")).ok();
            let bk = get(&format!("{p}.self_attn.k_proj.bias")).ok();
            let bv = get(&format!("{p}.self_attn.v_proj.bias")).ok();

            // MLP
            let mlp_or_moe = if let Ok(gate) = get(&format!("{p}.mlp.gate_proj.weight")) {
                let down = get(&format!("{p}.mlp.down_proj.weight"))?;
                let up = get(&format!("{p}.mlp.up_proj.weight"))?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_tensor(gate),
                    feed_forward_w2: QMatMul::from_tensor(down),
                    feed_forward_w3: QMatMul::from_tensor(up),
                })
            } else {
                let up = get(&format!("{p}.mlp.up_proj.weight"))?;
                let down = get(&format!("{p}.mlp.down_proj.weight"))?;
                MlpOrMoe::UpDown(MlpUpDown {
                    ffn_up: QMatMul::from_tensor(up),
                    ffn_down: QMatMul::from_tensor(down),
                })
            };

            let attn_norm_w = get(&format!("{p}.input_layernorm.weight"))?;
            let ffn_norm_w = get(&format!("{p}.post_attention_layernorm.weight"))?;

            let attn_norm = RmsNorm { weight: attn_norm_w, eps: rms_norm_eps, span: tracing::span!(tracing::Level::TRACE, "rms-norm") };
            let ffn_norm = RmsNorm { weight: ffn_norm_w, eps: rms_norm_eps, span: tracing::span!(tracing::Level::TRACE, "rms-norm") };

            // Post norms (Gemma2)
            let post_attn_norm = get(&format!("{p}.post_attention_layernorm_2.weight")).ok()
                .map(|w| RmsNorm { weight: w, eps: rms_norm_eps, span: tracing::span!(tracing::Level::TRACE, "rms-norm") });
            let post_ffn_norm = get(&format!("{p}.post_feedforward_layernorm.weight")).ok()
                .map(|w| RmsNorm { weight: w, eps: rms_norm_eps, span: tracing::span!(tracing::Level::TRACE, "rms-norm") });

            layers.push(LayerWeights {
                qkv,
                attention_wo: QMatMul::from_tensor(wo),
                attention_bq: bq,
                attention_bk: bk,
                attention_bv: bv,
                attention_norm: attn_norm,
                ffn_norm,
                mlp_or_moe,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                rope_style,
                rope_dim,
                padded_head_dim,
                post_attention_norm: post_attn_norm,
                post_ffn_norm,
                layer_idx: i,
                kv_cache: None,
                kv_compressed: None,
                tq_config: tq_config.clone(),
                signs: signs.clone(),
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "mlp"),
            });
        }

        eprintln!("FP16 model loaded! {} layers", block_count);

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }
}
