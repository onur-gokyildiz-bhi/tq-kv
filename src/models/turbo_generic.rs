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

use crate::cuda::{TqTensor as Tensor, TqDevice as Device, TqDType as DType, TqError};
use crate::cuda::Result;
use crate::gguf::{GgufContent, GgmlDType};
use crate::qmatmul as qmm;
use tq_kv::TurboQuantConfig;

/// Bail macro compatible with our error type.
macro_rules! bail {
    ($($arg:tt)*) => { return Err(TqError::Msg(format!($($arg)*))) };
}

// ============================================================
// Traits and primitives (replaces candle_nn)
// ============================================================

/// Module trait — replaces candle_nn::Module.
trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// Embedding lookup — replaces candle_nn::Embedding.
#[derive(Debug, Clone)]
struct Embedding {
    weight: Tensor,
    hidden_size: usize,
}

impl Embedding {
    fn new(weight: Tensor, hidden_size: usize) -> Self {
        Self { weight, hidden_size }
    }

    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let shape = ids.shape().to_vec();
        let ids_flat = ids.to_vec1()?;
        let w = self.weight.as_slice();
        let n_tokens = ids_flat.len();
        let mut output = Vec::with_capacity(n_tokens * self.hidden_size);
        for &id in &ids_flat {
            let idx = id as usize;
            let start = idx * self.hidden_size;
            let end = start + self.hidden_size;
            if end <= w.len() {
                output.extend_from_slice(&w[start..end]);
            } else {
                output.extend(std::iter::repeat(0.0f32).take(self.hidden_size));
            }
        }
        // Preserve input shape: [batch, seq_len] → [batch, seq_len, hidden_size]
        let mut out_shape = shape;
        out_shape.push(self.hidden_size);
        Tensor::from_vec(output, out_shape, ids.device())
    }
}

// ============================================================
// Shared primitives (CUDA-compatible)
// ============================================================

/// Device-aware RmsNorm — uses primitive ops that work on both CPU and GPU.
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    fn from_qweight(raw: &[u8], dtype: GgmlDType, n_elements: usize, eps: f64, device: &Device) -> Result<Self> {
        let data = crate::quant::dequantize(raw, dtype, n_elements);
        let weight = Tensor::from_vec(data, vec![n_elements], device)?;
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        Ok(Self { weight, eps, span })
    }

    /// Create from QWeight (candle API compat: `RmsNorm::from_qtensor(qt, eps, device)?`).
    fn from_qtensor(qw: qmm::QWeight, eps: f64, device: &Device) -> Result<Self> {
        let n_elements = qw.shape.0 * qw.shape.1;
        Self::from_qweight(&qw.raw_data, qw.dtype, n_elements, eps, device)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        #[cfg(feature = "cuda")]
        if x.is_cuda() {
            return self.forward_gpu(x);
        }

        // CPU path
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(x.rank() - 1)?;
        let eps_tensor = Tensor::new(self.eps as f32, x.device())?;
        let rms = variance.broadcast_add(&eps_tensor)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }

    #[cfg(feature = "cuda")]
    fn forward_gpu(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().to_vec();
        let hidden_dim = *shape.last().unwrap();
        let n_tokens = x.elem_count() / hidden_dim;
        let stream = x.cuda_stream();

        // Ensure norm weight is on GPU
        let weight_gpu = self.weight.ensure_gpu(
            &crate::cuda::TqDevice::Cuda {
                context: stream.context().clone(),
                ordinal: 0,
            }
        )?;

        let mut out_gpu = stream.alloc_zeros::<f32>(x.elem_count())
            .map_err(|e| TqError::Msg(format!("rms_norm alloc: {}", e)))?;
        let reg = crate::cuda::kernels::KernelRegistry::new(&stream.context(), &stream)
            .map_err(|e| TqError::Msg(format!("kernel init: {}", e)))?;

        crate::cuda::kernels::rms_norm(
            &reg, x.cuda_data(), weight_gpu.cuda_data(), &mut out_gpu,
            n_tokens, hidden_dim, self.eps as f32,
        ).map_err(|e| TqError::Msg(format!("rms_norm kernel: {}", e)))?;

        Ok(Tensor::from_cuda(out_gpu, shape, stream.clone()))
    }
}

pub const MAX_SEQ_LEN: usize = 4096;

/// Softmax along last dimension — GPU-accelerated when tensor is on CUDA.
fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.is_cuda() {
        return softmax_last_dim_gpu(x);
    }
    // CPU path
    let last = x.rank() - 1;
    let max_val = x.max_keepdim(last)?;
    let exp = x.broadcast_sub(&max_val)?.exp()?;
    let sum = exp.sum_keepdim(last)?;
    exp.broadcast_div(&sum)
}

#[cfg(feature = "cuda")]
fn softmax_last_dim_gpu(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let n_cols = *shape.last().unwrap();
    let n_rows = x.elem_count() / n_cols;
    let stream = x.cuda_stream();

    let mut out_gpu = stream.alloc_zeros::<f32>(x.elem_count())
        .map_err(|e| TqError::Msg(format!("softmax alloc: {}", e)))?;
    let reg = crate::cuda::kernels::KernelRegistry::new(&stream.context(), &stream)
        .map_err(|e| TqError::Msg(format!("kernel init: {}", e)))?;
    crate::cuda::kernels::softmax_last_dim(&reg, x.cuda_data(), &mut out_gpu, n_rows, n_cols)
        .map_err(|e| TqError::Msg(format!("softmax kernel: {}", e)))?;

    Ok(Tensor::from_cuda(out_gpu, shape, stream.clone()))
}

/// SiLU activation — GPU-accelerated when tensor is on CUDA.
fn silu(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.is_cuda() {
        return silu_gpu(x);
    }
    x.silu()
}

#[cfg(feature = "cuda")]
fn silu_gpu(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let n = x.elem_count();
    let stream = x.cuda_stream();

    let mut out_gpu = stream.alloc_zeros::<f32>(n)
        .map_err(|e| TqError::Msg(format!("silu alloc: {}", e)))?;
    let reg = crate::cuda::kernels::KernelRegistry::new(&stream.context(), &stream)
        .map_err(|e| TqError::Msg(format!("kernel init: {}", e)))?;
    crate::cuda::kernels::silu(&reg, x.cuda_data(), &mut out_gpu, n)
        .map_err(|e| TqError::Msg(format!("silu kernel: {}", e)))?;

    Ok(Tensor::from_cuda(out_gpu, shape, stream.clone()))
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: qmm::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qweight(w: qmm::QWeight) -> Result<Self> {
        let inner = qmm::QMatMul::from_qweight(w);
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// Alias for from_qweight (candle API compat).
    fn from_qtensor(w: qmm::QWeight) -> Result<Self> {
        Self::from_qweight(w)
    }

    fn from_tensor(tensor: Tensor) -> Self {
        let inner = qmm::QMatMul::from_tensor(tensor);
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
                let xs = xs.reshape(vec![b_size * seq_len, hidden_dim])?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = softmax_last_dim(&router_logits)?;
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2()?;

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
                    let n_tokens_for_expert = top_x.len();
                    let top_x = Tensor::from_vec(top_x.iter().map(|&x| x as f32).collect(), vec![n_tokens_for_expert], xs.device())?;
                    let selected_rws = Tensor::from_slice(&selected_rws[expert_idx], vec![n_tokens_for_expert], xs.device())?
                        .reshape(vec![n_tokens_for_expert, 1])?;
                    let current_state = xs.index_select(&top_x, 0)?.reshape(vec![n_tokens_for_expert, hidden_dim])?;
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }
                ys.reshape(vec![b_size, seq_len, hidden_dim])
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
            .expand(vec![b_sz, n_kv_head, n_rep, seq_len, head_dim])?
            .reshape(vec![b_sz, n_kv_head * n_rep, seq_len, head_dim])
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    // Expand on_true to match mask shape (handles scalar → N-D)
    let n = mask.elem_count();
    let val = on_true.to_vec1()?[0];
    let expanded = Tensor::from_vec(vec![val; n], mask.shape().to_vec(), mask.device())?;
    mask.where_cond(&expanded, on_false)
}

// ============================================================
// TurboQuant KV Cache — Incremental + Fused Attention
// ============================================================

/// Number of initial "sink" tokens whose keys are kept in FP16 (uncompressed).
/// Attention sink tokens receive disproportionate attention weight — quantizing them
/// causes up to 81% of total attention error. (KVSink, arXiv:2508.04257)
/// Override with TQ_SINK env var.
const TQ_SINK_TOKENS: usize = 4;

fn get_sink_tokens(config: &tq_kv::TurboQuantConfig) -> usize {
    if let Some(sink) = config.sink_tokens {
        return sink;
    }
    std::env::var("TQ_SINK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TQ_SINK_TOKENS)
}

/// Compressed value store — either 8-bit per-vector or 4-bit per-group absmax.
#[derive(Clone, Debug)]
enum CompressedValueStore {
    Bits8(Vec<tq_kv::CompressedValues>),
    Bits4(Vec<tq_kv::CompressedValues4Bit>),
}

/// Per-head compacted KV cache segment: selected keys + synthetic values + beta biases.
#[derive(Debug, Clone)]
struct CompactedCacheHead {
    /// Compacted keys [t * head_dim] (f32, post-RoPE)
    keys: Vec<f32>,
    /// Per-key attention bias (log scale, added to logits before softmax) [t]
    beta: Vec<f32>,
    /// Synthetic values [t * head_dim] (f32)
    values: Vec<f32>,
    /// Number of compacted tokens
    t: usize,
    head_dim: usize,
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
    /// Compacted cache: attention-matching token reduction (Zweiger 2026).
    /// Sits between cold and hot: [sink | cold | compacted | hot | current]
    compacted: Option<Vec<CompactedCacheHead>>,
    /// Number of original tokens that were compacted away
    compacted_original_len: usize,
    /// Value cache — uncompressed path (value_bits=0)
    v_raw: Option<Tensor>,
    /// Value cache — compressed path (value_bits=4 or 8), per KV head
    v_compressed: Option<CompressedValueStore>,
    /// Value quantization bits (0=fp16, 4=4-bit per-group, 8=8-bit absmax)
    value_bits: u8,
    /// Total cached length (sink + compressed + current)
    cached_len: usize,
    dtype: DType,
    /// Pre-RoPE mode: compressed keys are stored BEFORE RoPE application.
    /// At decode time, keys must be decompressed and RoPE applied dynamically.
    pre_rope: bool,
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
        return Tensor::zeros(vec![1, n_kv_head, 0, head_dim], device);
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
    Tensor::from_vec(all_data, vec![1, n_kv_head, compressed_len, head_dim], device)?.to_dtype(dtype)
}

/// Decompress pre-RoPE compressed keys and apply RoPE dynamically.
/// `start_pos` is the sequence position of the first compressed key.
/// Returns post-RoPE keys ready for attention computation.
fn decompress_and_apply_rope(
    k_per_head: &[tq_kv::CompressedKeys],
    n_kv_head: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
    config: &TurboQuantConfig,
    cos: &Tensor,
    sin: &Tensor,
    start_pos: usize,
    rope_style: RopeStyle,
    rope_dim: usize,
) -> Result<Tensor> {
    // First decompress to get pre-RoPE keys
    let pre_rope = decompress_compressed_keys(k_per_head, n_kv_head, head_dim, DType::F32, device, config)?;
    let compressed_len = if k_per_head.is_empty() || k_per_head[0].count == 0 {
        return Tensor::zeros(vec![1, n_kv_head, 0, head_dim], device);
    } else {
        k_per_head[0].count
    };

    // Apply RoPE: slice cos/sin for the correct positions
    let cos_slice = cos.narrow(0, start_pos, compressed_len)?;
    let sin_slice = sin.narrow(0, start_pos, compressed_len)?;

    let rotated = if rope_dim < head_dim {
        let x_rope = pre_rope.narrow(3, 0, rope_dim)?;
        let x_pass = pre_rope.narrow(3, rope_dim, head_dim - rope_dim)?;
        let x_rotated = match rope_style {
            RopeStyle::Halved => rope_halved(&x_rope, &cos_slice, &sin_slice)?,
            RopeStyle::Interleaved => rope_interleaved(&x_rope, &cos_slice, &sin_slice)?,
        };
        Tensor::cat(&[&x_rotated, &x_pass], 3)?
    } else {
        match rope_style {
            RopeStyle::Halved => rope_halved(&pre_rope, &cos_slice, &sin_slice)?,
            RopeStyle::Interleaved => rope_interleaved(&pre_rope, &cos_slice, &sin_slice)?,
        }
    };
    rotated.to_dtype(dtype)
}

/// Decompress compressed values to F32 tensor: (1, n_kv_head, seq_len, head_dim).
fn decompress_values_store(
    store: &CompressedValueStore,
    n_kv_head: usize,
    head_dim: usize,
    seq_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut all_data = Vec::with_capacity(n_kv_head * seq_len * head_dim);
    match store {
        CompressedValueStore::Bits8(v_per_head) => {
            for compressed in v_per_head.iter().take(n_kv_head) {
                all_data.extend(compressed.decompress());
            }
        }
        CompressedValueStore::Bits4(v_per_head) => {
            for compressed in v_per_head.iter().take(n_kv_head) {
                all_data.extend(compressed.decompress());
            }
        }
    }
    Tensor::from_vec(all_data, vec![1, n_kv_head, seq_len, head_dim], device)
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
    let att_flat = att.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let v_flat = v.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()?;
    let seq_len = att.dim(3)?;

    let mut output = Vec::with_capacity(n_heads * head_dim);
    for h in 0..n_heads {
        let att_row = &att_flat[h * seq_len..(h + 1) * seq_len];
        let v_block = &v_flat[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let head_out = tq_kv::sparse_attn_v_mul(att_row, v_block, head_dim, threshold);
        output.extend_from_slice(&head_out);
    }

    Tensor::from_vec(output, vec![1, n_heads, 1, head_dim], att.device())
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
    let x = x.reshape(vec![b, h, s, half, 2])?;
    let x0 = x.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = x.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let r0 = r0.unsqueeze(4)?;
    let r1 = r1.unsqueeze(4)?;
    Tensor::cat(&[&r0, &r1], 4)?.reshape(vec![b, h, s, d])
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

fn get_skip_layers(config: &tq_kv::TurboQuantConfig) -> usize {
    // Config field takes priority, then env var, then default
    if let Some(skip) = config.skip_layers {
        return skip;
    }
    std::env::var("TQ_SKIP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TQ_SKIP_FIRST_LAYERS)
}

/// Number of final layers to keep uncompressed (fp16 KV cache).
/// turboquant_plus found last layers are disproportionately sensitive to
/// quantization: last 8 layers account for ALL quality loss on some models.
/// Boundary protection (first N + last M) recovers 37-91% of quality gap.
/// Override with TQ_PROTECT_LAST env var (e.g. TQ_PROTECT_LAST=2).
fn get_protect_last_layers(config: &tq_kv::TurboQuantConfig) -> usize {
    if let Some(n) = config.protect_last_layers {
        return n;
    }
    std::env::var("TQ_PROTECT_LAST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Parsed layer bit ranges, cached at first use.
static LAYER_BITS_CACHE: std::sync::OnceLock<Vec<(usize, usize, u8)>> = std::sync::OnceLock::new();

fn parse_layer_bits() -> &'static Vec<(usize, usize, u8)> {
    LAYER_BITS_CACHE.get_or_init(|| {
        let mut ranges = Vec::new();
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
                            ranges.push((start, end, bits));
                        }
                    }
                }
            }
        }
        ranges
    })
}

/// Layer-adaptive bitwidth: assign different bit widths to different layer ranges.
/// Format: "start-end:bits[,start-end:bits]" e.g. "4-15:2,16-27:4"
/// Unspecified layers use the default TQ bits. Layers below TQ_SKIP or within
/// TQ_PROTECT_LAST of the final layer are uncompressed (fp16).
/// Override with TQ_LAYER_BITS env var.
fn get_layer_bits(layer_idx: usize, default_bits: u8, config: &tq_kv::TurboQuantConfig, n_layers: usize) -> Option<u8> {
    let skip = get_skip_layers(config);
    if layer_idx < skip {
        return None; // uncompressed — boundary protection (first N)
    }

    let protect_last = get_protect_last_layers(config);
    if protect_last > 0 && n_layers > 0 && layer_idx >= n_layers - protect_last {
        return None; // uncompressed — boundary protection (last M)
    }

    let ranges = parse_layer_bits();
    for &(start, end, bits) in ranges {
        if layer_idx >= start && layer_idx <= end {
            return Some(bits);
        }
    }

    Some(default_bits)
}

/// Parsed head bit ranges, cached at first use.
static HEAD_BITS_CACHE: std::sync::OnceLock<Vec<(usize, usize, u8)>> = std::sync::OnceLock::new();

/// Parse TQ_HEAD_BITS env var. Format: "0-3:4,4-7:2" (same syntax as TQ_LAYER_BITS).
fn parse_head_bits() -> &'static Vec<(usize, usize, u8)> {
    HEAD_BITS_CACHE.get_or_init(|| {
        let mut ranges = Vec::new();
        if let Ok(val) = std::env::var("TQ_HEAD_BITS") {
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
                            ranges.push((start, end, bits));
                        }
                    }
                }
            }
        }
        ranges
    })
}

/// Resolve per-head bit widths from TQ_HEAD_BITS env var.
/// Returns None if no TQ_HEAD_BITS is set (all heads use default_bits).
fn resolve_per_head_bits(n_kv_head: usize, default_bits: u8) -> Option<Vec<u8>> {
    let ranges = parse_head_bits();
    if ranges.is_empty() {
        return None;
    }
    let mut bits = vec![default_bits; n_kv_head];
    for &(start, end, b) in ranges {
        for h in start..=end.min(n_kv_head.saturating_sub(1)) {
            bits[h] = b;
        }
    }
    Some(bits)
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

/// Pre-RoPE quantization: compress keys BEFORE RoPE for position-independent statistics.
/// Enable with TQ_PRE_ROPE=1. Incompatible with fused attention (auto-disabled).
fn get_pre_rope() -> bool {
    std::env::var("TQ_PRE_ROPE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Compaction threshold: run compact_head() when hot tokens exceed this count.
/// 0 = disabled. Set TQ_COMPACT=N to enable (e.g. TQ_COMPACT=512).
fn get_compact_threshold() -> usize {
    std::env::var("TQ_COMPACT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Compaction target ratio: compact to N% of original tokens.
/// Default: 5 (5% = 20x reduction). Set TQ_COMPACT_RATIO=N.
fn get_compact_ratio() -> usize {
    std::env::var("TQ_COMPACT_RATIO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// Value cache quantization bits. 0 = uncompressed fp16, 4 = 4-bit per-group (~3.2x),
/// 8 = 8-bit per-vector absmax (~1.9x).
/// Override with TQ_VBITS env var (e.g. TQ_VBITS=4 for 3.2x value savings).
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

/// Compute SmoothAttention scales from calibration channel_scales.
///
/// SmoothAttention migrates K outliers to Q: K /= lambda, Q *= lambda.
/// When `for_query=true`: returns lambda (multiply Q by this).
/// When `for_query=false`: returns 1/lambda (multiply K by this — shrinks outliers).
///
/// If no channel_scales in config, returns None (SmoothAttention disabled).
/// Compute SmoothAttention scales from calibration channel_scales.
///
/// SmoothAttention replaces the old channel_scales-in-compression approach.
/// Instead of scaling K inside the compression pipeline (which changes K and
/// requires inverse on decompress), we migrate outliers from K to Q at the
/// tensor level. Q stays fp32 so this is lossless. K becomes smoother,
/// improving quantization quality.
///
/// Returns None if no channel_scales in config (SmoothAttention disabled).
fn compute_smooth_scales(
    config: &TurboQuantConfig,
    head_dim: usize,
    device: &Device,
    for_query: bool,
) -> Option<Tensor> {
    // Only enable SmoothAttention if we have calibrated channel_scales
    // AND compression is active (skip_layers != 999)
    let scales = config.channel_scales.as_ref()?;
    if scales.len() != head_dim || config.skip_layers == Some(999) {
        return None;
    }

    // For Q: multiply by sqrt(scale) — absorb half the outlier
    // For K: multiply by 1/sqrt(scale) — smooth down outliers
    // Invariance: (Q*sqrt(s)) * (K/sqrt(s))^T = Q*K^T
    let vals: Vec<f32> = if for_query {
        scales.iter().map(|&s| s.max(0.01).sqrt()).collect()
    } else {
        scales.iter().map(|&s| 1.0 / s.max(0.01).sqrt()).collect()
    };

    Tensor::from_vec(vals, vec![head_dim], device).ok()
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
    /// Total number of layers — needed for boundary protection (TQ_PROTECT_LAST)
    n_layers: usize,
    /// Standard KV cache for uncompressed layers
    kv_cache: Option<(Tensor, Tensor)>,
    kv_compressed: Option<CompressedKvCache>,
    tq_config: TurboQuantConfig,
    signs: Vec<f32>,
    /// SmoothAttention: per-channel scales to migrate K outliers to Q.
    /// K is divided by these scales (reducing outliers), Q is multiplied (lossless since Q stays fp32).
    /// Computed during calibration or from running statistics.
    smooth_k_scales: Option<Tensor>,  // [1, 1, 1, head_dim]
    smooth_q_scales: Option<Tensor>,  // [1, 1, 1, head_dim]
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

        // Calibration hook: collect RAW key vectors (before attention bias, before RoPE).
        // These have consistent per-channel statistics — no positional or bias contamination.
        // Ideal for computing per-channel bias and scale calibration.
        if crate::calibrate::CALIBRATION_COLLECTOR.get().is_some() {
            let k_for_cal = k.reshape(vec![b_sz, seq_len, self.n_kv_head, self.head_dim])?
                .transpose(1, 2)?.contiguous()?;
            if let Ok(k_f32) = k_for_cal.to_dtype(DType::F32)?.flatten_all()?.to_vec1() {
                crate::calibrate::maybe_collect(&k_f32, self.n_kv_head, seq_len, self.head_dim);
            }
        }

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

        let q = q.reshape(vec![b_sz, seq_len, self.n_head, self.head_dim])?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(vec![b_sz, seq_len, self.n_kv_head, self.head_dim])?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(vec![b_sz, seq_len, self.n_kv_head, self.head_dim])?.transpose(1, 2)?.contiguous()?;

        // SmoothAttention: migrate K outliers to Q BEFORE RoPE.
        // K *= 1/sqrt(s), Q *= sqrt(s). Invariance: (Q*sqrt(s)) * (K/sqrt(s))^T = Q * K^T
        let (q, k) = if let (Some(ref q_scales), Some(ref k_scales)) = (&self.smooth_q_scales, &self.smooth_k_scales) {
            (q.broadcast_mul(q_scales)?, k.broadcast_mul(k_scales)?)
        } else {
            (q, k)
        };

        // Pre-RoPE quantization: save k BEFORE RoPE for compression.
        // Pre-RoPE keys have position-independent per-channel stats → better quantization.
        let pre_rope_mode = self.tq_config.pre_rope || get_pre_rope();
        let k_pre_rope = if pre_rope_mode { Some(k.clone()) } else { None };

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        // Selective compression: first TQ_SKIP_FIRST_LAYERS use standard fp16 KV cache,
        // remaining layers use TurboQuant compression.
        let layer_bits = get_layer_bits(self.layer_idx, self.tq_config.bits, &self.tq_config, self.n_layers);
        let use_compression = layer_bits.is_some();
        let n_rep = self.n_head / self.n_kv_head;

        if use_compression {
            // Apply per-layer bit width if different from default
            let effective_bits = layer_bits.unwrap();
            let mut layer_tq_config = if effective_bits != self.tq_config.bits {
                tq_kv::TurboQuantConfig { bits: effective_bits, ..self.tq_config.clone() }
            } else {
                self.tq_config.clone()
            };
            // When SmoothAttention is active, disable channel_scales in compression
            // to avoid double-application (SmoothAttention already handled the scaling)
            if self.smooth_k_scales.is_some() {
                layer_tq_config.channel_scales = None;
            }
            // Per-head adaptive bitwidth: resolve from TQ_HEAD_BITS env var or config
            let per_head_bits = resolve_per_head_bits(self.n_kv_head, effective_bits)
                .or_else(|| self.tq_config.per_head_bits.clone());
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
            let sink_n = get_sink_tokens(&self.tq_config);

            // Initialize cache on first call
            let vbits = get_value_bits();
            if self.kv_compressed.is_none() {
                let mut k_per_head = Vec::with_capacity(self.n_kv_head);
                let gs = layer_tq_config.group_size;
                for h in 0..self.n_kv_head {
                    let hbits = per_head_bits.as_ref()
                        .map(|phb| phb[h])
                        .unwrap_or(effective_bits);
                    k_per_head.push(tq_kv::CompressedKeys::new_empty_grouped(
                        hbits, self.padded_head_dim, layer_tq_config.rotation_seed, gs,
                    ));
                }
                let (v_raw, v_compressed) = match vbits {
                    0 => (None, None),
                    4 => {
                        let gs = layer_tq_config.group_size.max(32);
                        let mut v_heads: Vec<tq_kv::CompressedValues4Bit> = Vec::with_capacity(self.n_kv_head);
                        for _ in 0..self.n_kv_head {
                            v_heads.push(tq_kv::CompressedValues4Bit::new_empty(self.head_dim, gs));
                        }
                        (None, Some(CompressedValueStore::Bits4(v_heads)))
                    }
                    _ => {
                        // 8-bit (default for any non-zero, non-4 value)
                        let mut v_heads: Vec<tq_kv::CompressedValues> = Vec::with_capacity(self.n_kv_head);
                        for _ in 0..self.n_kv_head {
                            v_heads.push(tq_kv::CompressedValues::new_empty(self.head_dim));
                        }
                        (None, Some(CompressedValueStore::Bits8(v_heads)))
                    }
                };
                self.kv_compressed = Some(CompressedKvCache {
                    k_per_head,
                    k_cold: None,
                    cold_len: 0,
                    decay_config: get_decay_config(),
                    tokens_since_decay: 0,
                    sink_k: None,
                    sink_len: 0,
                    compacted: None,
                    compacted_original_len: 0,
                    v_raw,
                    v_compressed,
                    value_bits: vbits,
                    cached_len: 0,
                    dtype: cache_dtype,
                    pre_rope: pre_rope_mode,
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
                // Compressed path: quantize per head × position
                let v_f32 = v.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()?;
                let v_store = cache.v_compressed.as_mut().unwrap();
                match v_store {
                    CompressedValueStore::Bits8(v_comp) => {
                        for h in 0..self.n_kv_head {
                            for s in 0..seq_len {
                                let offset = (h * seq_len + s) * self.head_dim;
                                v_comp[h].append(&v_f32[offset..offset + self.head_dim]);
                            }
                        }
                    }
                    CompressedValueStore::Bits4(v_comp) => {
                        for h in 0..self.n_kv_head {
                            for s in 0..seq_len {
                                let offset = (h * seq_len + s) * self.head_dim;
                                v_comp[h].append(&v_f32[offset..offset + self.head_dim]);
                            }
                        }
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
                // Pre-RoPE mode: compress pre-RoPE keys (position-independent stats).
                // Post-RoPE mode (default): compress post-RoPE keys.
                let k_source = if pre_rope_mode {
                    k_pre_rope.as_ref().unwrap()
                } else {
                    &k
                };
                let k_to_compress = k_source.narrow(2, compress_start, tokens_to_compress)?;
                let k_flat = k_to_compress.contiguous()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let hdim = self.head_dim;
                let use_grouped = layer_tq_config.group_size > 0 && hdim % layer_tq_config.group_size == 0;
                for h in 0..self.n_kv_head {
                    // Per-head adaptive bitwidth: use head-specific config if assigned
                    let head_config = match per_head_bits {
                        Some(ref phb) if phb[h] != layer_tq_config.bits => {
                            // Clear calibrated codebook if bit width differs — it was
                            // calibrated for layer_tq_config.bits, not for this head's bits.
                            tq_kv::TurboQuantConfig {
                                bits: phb[h],
                                calibrated_codebook: None,
                                ..layer_tq_config.clone()
                            }
                        }
                        _ => layer_tq_config.clone(),
                    };
                    for s in 0..tokens_to_compress {
                        let offset = (h * tokens_to_compress + s) * hdim;
                        let key_vec = &k_flat[offset..offset + hdim];
                        if use_grouped {
                            let (packed, gnorms, residual, outliers) = tq_kv::compress_single_key_grouped(
                                key_vec, hdim, &head_config, &self.signs,
                            );
                            // Set residual/outlier bits on first append
                            if cache.k_per_head[h].residual_bits == 0 && residual.is_some() {
                                cache.k_per_head[h].residual_bits = head_config.residual_bits;
                            }
                            if cache.k_per_head[h].outlier_k == 0 && outliers.is_some() {
                                cache.k_per_head[h].outlier_k = head_config.outlier_k;
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
                                &padded, self.padded_head_dim, &head_config, &self.signs,
                            );
                            cache.k_per_head[h].append_raw(&packed, norm);
                        } else {
                            let (packed, norm) = tq_kv::compress_single_key_with_signs(
                                key_vec, hdim, &head_config, &self.signs,
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

            // --- KV Compaction: reduce token count when hot cache exceeds threshold ---
            // Compacts the oldest hot tokens into a smaller set of synthetic tokens
            // with attention biases (beta) to preserve attention behavior.
            // Only triggers once (no repeated compaction) to avoid cascading quality loss.
            let compact_threshold = get_compact_threshold();
            if compact_threshold > 0
                && cache.k_per_head[0].count > compact_threshold
                && seq_len == 1
                && cache.compacted.is_none()  // Only compact once
            {
                let hot_count = cache.k_per_head[0].count;
                // Keep the most recent compact_threshold/2 tokens uncompacted
                let keep_recent = compact_threshold / 2;
                let to_compact = hot_count.saturating_sub(keep_recent);
                if to_compact > 8 {
                    let ratio = get_compact_ratio();
                    let target_size = (to_compact * ratio / 100).max(4);

                    // Decompress hot keys for compaction.
                    // Pre-RoPE mode: decompress + apply RoPE so compaction works on post-RoPE keys.
                    let k_decomp_full = if cache.pre_rope {
                        let hot_start = cache.sink_len + cache.cold_len + cache.compacted_original_len;
                        decompress_and_apply_rope(
                            &cache.k_per_head, self.n_kv_head,
                            self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                            &self.cos, &self.sin, hot_start, self.rope_style, self.rope_dim,
                        )?
                    } else {
                        decompress_compressed_keys(
                            &cache.k_per_head, self.n_kv_head,
                            self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                        )?
                    };
                    let k_flat = k_decomp_full.flatten_all()?.to_vec1()?;

                    // Use ALL query heads mapped to each KV head as reference queries.
                    // More reference queries = better compaction quality.
                    // For GQA: n_rep queries share each KV head.
                    let q_flat = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                    // Collect recent queries: use current query (all heads)
                    // Each KV head gets n_rep reference queries from its mapped Q heads
                    let n_ref_queries = n_rep;

                    // Get values for compaction
                    let v_for_compact = if cache.value_bits == 0 {
                        let v_tensor = cache.v_raw.as_ref().unwrap().to_dtype(DType::F32)?;
                        // v_tensor is [1, n_kv_head, total_len, head_dim]
                        v_tensor.flatten_all()?.to_vec1()?
                    } else {
                        let v_tensor = decompress_values_store(
                            cache.v_compressed.as_ref().unwrap(),
                            self.n_kv_head, self.head_dim, total_len, q.device(),
                        )?;
                        v_tensor.flatten_all()?.to_vec1()?
                    };

                    let hdim = self.head_dim;
                    let padded = self.padded_head_dim;
                    let mut compacted_heads = Vec::with_capacity(self.n_kv_head);
                    for h in 0..self.n_kv_head {
                        // Extract this head's oldest keys [to_compact * padded_head_dim]
                        let k_head_offset = h * hot_count * padded;
                        let k_head = &k_flat[k_head_offset..k_head_offset + to_compact * padded];
                        // Trim padding to head_dim
                        let mut k_trimmed = Vec::with_capacity(to_compact * hdim);
                        for i in 0..to_compact {
                            k_trimmed.extend_from_slice(&k_head[i * padded..i * padded + hdim]);
                        }

                        // Collect reference queries for this KV head (all n_rep mapped Q heads)
                        let mut q_refs = Vec::with_capacity(n_ref_queries * hdim);
                        for qh_offset in 0..n_rep {
                            let qh = h * n_rep + qh_offset;
                            q_refs.extend_from_slice(&q_flat[qh * hdim..(qh + 1) * hdim]);
                        }

                        // Extract values: v_for_compact layout is [n_kv_head, total_len, head_dim]
                        // Hot values start at position (sink_len + cold_len)
                        let v_start = cache.sink_len + cache.cold_len;
                        let v_head_base = h * total_len * hdim;
                        let v_head_offset = v_head_base + v_start * hdim;
                        let v_head = &v_for_compact[v_head_offset..v_head_offset + to_compact * hdim];

                        let compacted = tq_kv::compaction::compact_head(
                            &k_trimmed, v_head, &q_refs,
                            to_compact, n_ref_queries, hdim, target_size,
                        );
                        compacted_heads.push(CompactedCacheHead {
                            keys: compacted.keys,
                            beta: compacted.beta,
                            values: compacted.values,
                            t: compacted.t,
                            head_dim: hdim,
                        });
                    }

                    // Remove compacted tokens from hot tier
                    for h in 0..self.n_kv_head {
                        let _front = cache.k_per_head[h].split_off_front(to_compact);
                    }
                    cache.compacted = Some(compacted_heads);
                    cache.compacted_original_len += to_compact;
                }
            }

            // --- Build full key tensor for attention ---
            // Concatenate: [sink | cold | compacted | hot | current]
            // This is the POQ + Sink + Decay + Compaction approach

            let y = if seq_len == 1 && total_len > 1 {
                // GENERATION: single token attention against full cache
                //
                // POQ (Past-Only Quantization): for the current token's key, we use
                // the FP16 original in attention instead of the compressed version.
                // The compressed version IS in the cache (for future tokens), but
                // we replace the last position with the lossless original.
                let q_f32 = q.to_dtype(DType::F32)?;

                // Fused attention incompatible with pre-RoPE and compaction
                let has_compacted = cache.compacted.is_some();
                let use_fused = get_use_fused() && q.device().is_cpu()
                    && !cache.pre_rope && !has_compacted;
                let n_compressed = cache.k_per_head[0].count;
                let n_past_compressed = if n_compressed > 0 { n_compressed - 1 } else { 0 };
                let compacted_t = cache.compacted.as_ref()
                    .map(|c| c[0].t).unwrap_or(0);

                // --- Compute attention scores ---
                let att = if use_fused {
                    // FUSED PATH: compute scores directly from compressed indices.
                    // No key decompression — saves memory bandwidth.
                    // Scores are computed per query-head using pre-rotated query
                    // and centroid table lookup (AVX2 SIMD when available).
                    let q_flat = q_f32.flatten_all()?.to_vec1()?;
                    let scale = 1.0 / (self.head_dim as f32).sqrt();
                    // Use calibrated centroids if available, else standard Gaussian
                    let cal_centroids_owned: Option<Vec<f32>> = layer_tq_config.calibrated_codebook
                        .as_ref().map(|cb| cb.centroids.clone());
                    // Note: cold centroids looked up per-head inside the loop (mixed bits)

                    use rayon::prelude::*;
                    let head_scores: Vec<Vec<f32>> = (0..self.n_head)
                        .into_par_iter()
                        .map(|qh| {
                            let kv_h = qh / n_rep;
                            let q_vec = &q_flat[qh * self.head_dim..(qh + 1) * self.head_dim];
                            let rotated_q = if let Some(ref matrix) = self.tq_config.rotation_matrix {
                                tq_kv::pre_rotate_query_with_matrix(q_vec, matrix)
                            } else {
                                tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs)
                            };
                            let mut scores = Vec::with_capacity(total_len);

                            // Segment 1: Sink keys (standard dot product, not compressed)
                            if let Some(ref sink) = cache.sink_k {
                                let sink_f32 = sink.to_dtype(DType::F32).expect("sink to_dtype");
                                let sink_flat = sink_f32.flatten_all().expect("sink flatten").to_vec1().expect("sink to_vec1");
                                let sink_count = cache.sink_len;
                                for s in 0..sink_count {
                                    let offset = (kv_h * sink_count + s) * self.head_dim;
                                    let k_vec = &sink_flat[offset..offset + self.head_dim];
                                    let dot: f32 = q_vec.iter().zip(k_vec.iter())
                                        .map(|(&qi, &ki)| qi * ki).sum();
                                    scores.push(dot * scale);
                                }
                            }

                            // Segment 2: Cold (decayed) keys — fused at cold bit width (per-head)
                            if let Some(ref cold) = cache.k_cold {
                                let cold_cb = tq_kv::codebook::get_centroids(cold[kv_h].bits);
                                let cold_scores = tq_kv::fused_attention_scores(
                                    &rotated_q, &cold[kv_h], cold_cb, scale,
                                );
                                scores.extend_from_slice(&cold_scores);
                            }

                            // Segment 3: Hot compressed keys (excluding last = current)
                            if n_past_compressed > 0 {
                                // Use calibrated centroids only if they match the head's bit width
                                let head_bits = cache.k_per_head[kv_h].bits;
                                let hot_cb: &[f32] = match cal_centroids_owned.as_deref() {
                                    Some(cal) if head_bits == layer_tq_config.bits => cal,
                                    _ => tq_kv::codebook::get_centroids(head_bits),
                                };
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
                                let k_f32 = k.to_dtype(DType::F32).expect("k to_dtype");
                                let k_flat = k_f32.flatten_all().expect("k flatten").to_vec1().expect("k to_vec1");
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
                        all_scores, vec![1, self.n_head, 1, total_len], q.device(),
                    )?;
                    softmax_last_dim(&att)?
                } else {
                    // DECOMPRESS PATH: decompress all keys, standard matmul
                    let mut k_parts: Vec<Tensor> = Vec::new();

                    // Part 1: Sink keys (FP16, lossless — always post-RoPE)
                    if let Some(ref sink) = cache.sink_k {
                        k_parts.push(sink.to_dtype(DType::F32)?);
                    }

                    // Part 1.5: Cold (decayed) keys
                    if let Some(ref cold) = cache.k_cold {
                        if cold[0].count > 0 {
                            let k_cold = if cache.pre_rope {
                                // Pre-RoPE: decompress + apply RoPE dynamically
                                let cold_start = cache.sink_len;
                                decompress_and_apply_rope(
                                    cold, self.n_kv_head,
                                    self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                                    &self.cos, &self.sin, cold_start, self.rope_style, self.rope_dim,
                                )?
                            } else {
                                decompress_compressed_keys(
                                    cold, self.n_kv_head,
                                    self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                                )?
                            };
                            let k_cold = if self.padded_head_dim > self.head_dim {
                                k_cold.narrow(3, 0, self.head_dim)?
                            } else {
                                k_cold
                            };
                            k_parts.push(k_cold);
                        }
                    }

                    // Part 2: Compacted keys (attention-matching reduced tokens + beta bias)
                    if let Some(ref comp) = cache.compacted {
                        // Build compacted keys tensor: [1, n_kv_head, compacted_t, head_dim]
                        let ct = comp[0].t;
                        let mut comp_data = Vec::with_capacity(self.n_kv_head * ct * self.head_dim);
                        for h in 0..self.n_kv_head {
                            comp_data.extend_from_slice(&comp[h].keys);
                        }
                        let k_comp = Tensor::from_vec(
                            comp_data, vec![1, self.n_kv_head, ct, self.head_dim], q.device(),
                        )?;
                        k_parts.push(k_comp);
                    }

                    // Part 3: Hot compressed keys (excluding last = current token)
                    if n_past_compressed > 0 {
                        let k_decomp = if cache.pre_rope {
                            // Pre-RoPE: decompress + apply RoPE dynamically
                            let hot_start = cache.sink_len + cache.cold_len + cache.compacted_original_len;
                            decompress_and_apply_rope(
                                &cache.k_per_head, self.n_kv_head,
                                self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                                &self.cos, &self.sin, hot_start, self.rope_style, self.rope_dim,
                            )?
                        } else {
                            decompress_compressed_keys(
                                &cache.k_per_head, self.n_kv_head,
                                self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                            )?
                        };
                        let k_decomp = if self.padded_head_dim > self.head_dim {
                            k_decomp.narrow(3, 0, self.head_dim)?
                        } else {
                            k_decomp
                        };
                        let k_past = k_decomp.narrow(2, 0, n_past_compressed)?;
                        k_parts.push(k_past);
                    }

                    // Part 4: Current token key (FP16 original — POQ lossless, always post-RoPE)
                    // Only add if current token was compressed (not still in sink range)
                    if n_compressed > 0 {
                        k_parts.push(k.to_dtype(DType::F32)?);
                    }

                    let k_full = if k_parts.len() == 1 {
                        k_parts.remove(0)
                    } else {
                        let k_refs: Vec<&Tensor> = k_parts.iter().collect();
                        Tensor::cat(&k_refs, 2)?
                    };

                    // Effective attention length: sink + cold + compacted_t + hot + current
                    let attn_len = cache.sink_len + cache.cold_len + compacted_t
                        + n_past_compressed + if n_compressed > 0 { 1 } else { 0 };

                    let k_full = repeat_kv(k_full, n_rep)?;
                    let mut att = (q_f32.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;

                    // Apply compaction beta biases to compacted segment logits
                    if let Some(ref comp) = cache.compacted {
                        let ct = comp[0].t;
                        let beta_start = cache.sink_len + cache.cold_len;
                        // Build bias vector: [sink=0, cold=0, compacted=beta, hot=0, current=0]
                        let mut full_bias = vec![0.0f32; attn_len];
                        // For simplicity, use first head's beta (GQA: same across heads)
                        for (i, &b) in comp[0].beta.iter().enumerate() {
                            full_bias[beta_start + i] = b;
                        }
                        let bias_tensor = Tensor::from_vec(
                            full_bias, vec![1, 1, 1, attn_len], q.device(),
                        )?;
                        att = att.broadcast_add(&bias_tensor)?;
                    }

                    // Softmax bias correction: compensate quantization-induced attention drift
                    if get_bias_correction() && n_past_compressed > 0 {
                        let bias = tq_kv::softmax_bias_correction(
                            &cache.k_per_head[0], self.head_dim,
                        );
                        // Build full bias vector: [..., hot_bias, ...]
                        let mut full_bias = vec![0.0f32; attn_len];
                        let hot_start = cache.sink_len + cache.cold_len + compacted_t;
                        for (i, &b) in bias.iter().take(n_past_compressed).enumerate() {
                            full_bias[hot_start + i] = b;
                        }
                        let bias_tensor = Tensor::from_vec(
                            full_bias, vec![1, 1, 1, attn_len], q.device(),
                        )?;
                        att = att.broadcast_add(&bias_tensor)?;
                    }

                    softmax_last_dim(&att)?
                };

                // --- Compute attention output: att @ V ---
                let sparse_thresh = get_sparse_v_threshold();

                // When compaction is active, build value tensor with compacted values spliced in
                if has_compacted {
                    // Build V tensor: [sink_v | cold_v | compacted_v | hot_v | current_v]
                    let mut v_parts: Vec<Tensor> = Vec::new();
                    let device = q.device();
                    let head_dim = self.head_dim;

                    // Get original value tensor (all positions including removed ones)
                    let v_all_f32 = if cache.value_bits == 0 {
                        cache.v_raw.as_ref().unwrap().to_dtype(DType::F32)?
                    } else {
                        decompress_values_store(
                            cache.v_compressed.as_ref().unwrap(),
                            self.n_kv_head, head_dim, total_len, device,
                        )?
                    };

                    // Sink values: positions [0, sink_len)
                    if cache.sink_len > 0 {
                        v_parts.push(v_all_f32.narrow(2, 0, cache.sink_len)?);
                    }

                    // Cold values: positions [sink_len, sink_len + cold_len)
                    if cache.cold_len > 0 {
                        v_parts.push(v_all_f32.narrow(2, cache.sink_len, cache.cold_len)?);
                    }

                    // Compacted synthetic values
                    let comp = cache.compacted.as_ref().unwrap();
                    let ct = comp[0].t;
                    let mut comp_v_data = Vec::with_capacity(self.n_kv_head * ct * head_dim);
                    for h in 0..self.n_kv_head {
                        comp_v_data.extend_from_slice(&comp[h].values);
                    }
                    let v_comp = Tensor::from_vec(
                        comp_v_data, vec![1, self.n_kv_head, ct, head_dim], device,
                    )?;
                    v_parts.push(v_comp);

                    // Hot values: skip compacted_original_len, take remaining
                    let hot_v_start = cache.sink_len + cache.cold_len + cache.compacted_original_len;
                    let hot_v_count = n_past_compressed + if n_compressed > 0 { 1 } else { 0 };
                    if hot_v_count > 0 {
                        v_parts.push(v_all_f32.narrow(2, hot_v_start, hot_v_count)?);
                    }

                    let v_full = if v_parts.len() == 1 {
                        v_parts.remove(0)
                    } else {
                        let v_refs: Vec<&Tensor> = v_parts.iter().collect();
                        Tensor::cat(&v_refs, 2)?
                    };
                    let v_full = repeat_kv(v_full, n_rep)?;
                    att.matmul(&v_full.contiguous()?)?.to_dtype(cache.dtype)?

                } else if sparse_thresh > 0.0 && att.device().is_cpu() && cache.value_bits > 0 {
                    // Fused sparse-decompress path: decompress only active V rows
                    let att_flat = att.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                    let v_store = cache.v_compressed.as_ref().unwrap();
                    let head_dim = self.head_dim;
                    let mut output = Vec::with_capacity(self.n_head * head_dim);

                    for h in 0..self.n_head {
                        let kv_h = h / n_rep; // GQA head mapping
                        let att_row = &att_flat[h * total_len..(h + 1) * total_len];

                        let head_out = match v_store {
                            CompressedValueStore::Bits4(ref v) =>
                                tq_kv::sparse_attn_v_mul_compressed_4bit(att_row, &v[kv_h], sparse_thresh),
                            CompressedValueStore::Bits8(ref v) =>
                                tq_kv::sparse_attn_v_mul_compressed_8bit(att_row, &v[kv_h], sparse_thresh),
                        };
                        output.extend_from_slice(&head_out);
                    }

                    Tensor::from_vec(output, vec![1, self.n_head, 1, head_dim], att.device())?
                        .to_dtype(cache.dtype)?
                } else {
                    // Standard path: decompress all values, matmul or sparse multiply
                    let v_f32 = if cache.value_bits == 0 {
                        repeat_kv(cache.v_raw.as_ref().unwrap().to_dtype(DType::F32)?, n_rep)?
                    } else {
                        let v_tensor = decompress_values_store(
                            cache.v_compressed.as_ref().unwrap(),
                            self.n_kv_head, self.head_dim, total_len, q.device(),
                        )?;
                        repeat_kv(v_tensor, n_rep)?
                    };

                    if sparse_thresh > 0.0 && att.device().is_cpu() {
                        sparse_attn_v(&att, &v_f32, self.n_head, self.head_dim, sparse_thresh)?
                            .to_dtype(cache.dtype)?
                    } else {
                        att.matmul(&v_f32.contiguous()?)?.to_dtype(cache.dtype)?
                    }
                }
            } else {
                // PREFILL: use original uncompressed keys for attention (standard path)
                let k = repeat_kv(k, n_rep)?;
                let v_for_attn = repeat_kv(v, n_rep)?;

                // Standard attention for prefill (no candle flash attention dependency)
                if false {
                    // placeholder branch — flash attention will be re-added when custom CUDA kernels land
                    unreachable!()
                } else {
                    let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                    let att = match mask {
                        None => att,
                        Some(mask) => {
                            // mask is [seq, seq], att is [batch, heads, seq, seq]
                            let mask4d = mask.unsqueeze(0)?.unsqueeze(0)?;
                            let mask4d = mask4d.broadcast_as(att.shape())?;
                            masked_fill(&att, &mask4d, &self.neg_inf)?
                        }
                    };
                    let att = softmax_last_dim(&att)?;
                    att.matmul(&v_for_attn.contiguous()?)?
                }
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
                    let mask4d = mask.unsqueeze(0)?.unsqueeze(0)?;
                    let mask4d = mask4d.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask4d, &self.neg_inf)?
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
        let protect_last = get_protect_last_layers(&tq_config);
        let skip_first = get_skip_layers(&tq_config);
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

        // Embeddings + output
        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize_to_device(device)?;
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
                n_layers: block_count,
                kv_cache: None,
                kv_compressed: None,
                tq_config: tq_config.clone(),
                signs: signs.clone(),
                smooth_k_scales: compute_smooth_scales(&tq_config, head_dim, device, false),
                smooth_q_scales: compute_smooth_scales(&tq_config, head_dim, device, true),
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
            let mask: Vec<f32> = (0..t)
                .flat_map(|i| (0..t).map(move |j| if j > i { 1.0f32 } else { 0.0f32 }))
                .collect();
            let mask = Tensor::from_slice(&mask, vec![t, t], device)?;
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
        let x = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
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
        _model_dir: &std::path::Path,
        _device: &Device,
        _tq_config: TurboQuantConfig,
    ) -> Result<Self> {
        bail!("safetensors loading not yet implemented for tq-cuda backend")
    }
}
