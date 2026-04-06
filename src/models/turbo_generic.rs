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
use std::sync::Arc;

// Thread-local raw device pointer for RoPE position GPU scalar.
// Points to the model's rope_pos_gpu CudaSlice. Updated before capture/replay.
#[cfg(feature = "cuda")]
std::thread_local! {
    static ROPE_POS_GPU_PTR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

use crate::backend::ComputeBackend;
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
    fn forward(&self, x: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor>;
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

    fn forward(&self, x: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_f32 = x.to_dtype(DType::F32)?;

        // GPU-resident path: use TqTensor::rms_norm_gpu directly
        #[cfg(feature = "cuda")]
        if x_f32.is_cuda() {
            return x_f32.rms_norm_gpu(&self.weight, self.eps as f32);
        }

        // CPU path via backend
        let shape = x_f32.shape().to_vec();
        let hidden = *shape.last().unwrap();
        let n_tokens = x_f32.elem_count() / hidden;
        let result = backend.rms_norm(x_f32.as_slice(), self.weight.as_slice(), self.eps as f32, n_tokens, hidden);
        Tensor::from_vec(result, shape, x.device())
    }
}

pub const MAX_SEQ_LEN: usize = 4096;

/// Softmax along last dimension — dispatched via compute backend.
fn softmax_last_dim(x: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.is_cuda() {
        return x.softmax_gpu();
    }
    let shape = x.shape().to_vec();
    let cols = *shape.last().unwrap();
    let rows = x.elem_count() / cols;
    let result = backend.softmax(x.as_slice(), rows, cols);
    Tensor::from_vec(result, shape, x.device())
}

/// Fused SiLU gate × up: silu(gate) * up — single pass, no intermediate tensor.
fn fused_silu_mul(gate: &Tensor, up: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.is_cuda() && up.is_cuda() {
        return gate.fused_silu_mul_gpu(up);
    }
    let result = backend.fused_silu_mul(gate.as_slice(), up.as_slice());
    Tensor::from_vec(result, gate.shape().to_vec(), gate.device())
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

    /// Pre-warm weight cache: dequant on CPU + upload to GPU via backend.
    /// Call during model load to eliminate first-forward latency.
    fn warmup(&self, backend: &dyn ComputeBackend) {
        match &self.inner {
            qmm::QMatMul::Quantized(qw) => {
                qw.warmup_cpu();
                backend.warmup_qweight(qw);
            }
            qmm::QMatMul::Full(_) => {},
        }
    }

    /// Get inner QWeight if it's Q4_K_M quantized (for fused kernel launches).
    fn q4k_weight(&self) -> Option<&qmm::QWeight> {
        match &self.inner {
            qmm::QMatMul::Quantized(qw) if matches!(qw.dtype, crate::gguf::GgmlDType::Q4K) => Some(qw),
            _ => None,
        }
    }

    /// Get GPU-resident raw Q4_K_M weight bytes for fused kernel launches.
    /// Lazily uploads on first call, cached for all subsequent calls.
    #[cfg(feature = "cuda")]
    fn q4k_gpu_data(&self) -> Option<(&cudarc::driver::CudaSlice<u8>, usize, usize)> {
        let qw = self.q4k_weight()?;
        let reg = crate::cuda::kernels::global_registry()?;
        let gpu = qw.gpu_cache_or_upload(&reg.stream);
        Some((gpu, qw.out_features(), qw.in_features()))
    }

    fn forward(&self, xs: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        let _enter = self.span.enter();
        match &self.inner {
            qmm::QMatMul::Quantized(qw) => {
                // GPU-resident path: keep data on GPU (decode only, batch=1)
                #[cfg(feature = "cuda")]
                if xs.is_cuda() {
                    let x_shape = xs.shape().to_vec();
                    let batch: usize = x_shape[..x_shape.len() - 1].iter().product();
                    if batch == 1 && matches!(qw.dtype, crate::gguf::GgmlDType::Q4K | crate::gguf::GgmlDType::Q8_0) {
                        let w_gpu = qw.gpu_cache_or_upload(
                            &crate::cuda::kernels::global_registry()
                                .expect("no GPU registry").stream
                        );
                        return xs.qmatmul_gpu(w_gpu, qw.dtype, qw.out_features(), qw.in_features());
                    }
                    // Prefill or Q6K on GPU: streaming GPU dequant + SGEMM
                    return self.inner.forward(xs);
                }

                let x_shape = xs.shape().to_vec();
                let in_features = qw.in_features();
                let out_features = qw.out_features();
                let last_dim = *x_shape.last()
                    .ok_or_else(|| TqError::Msg("empty input".into()))?;
                if last_dim != in_features {
                    return Err(TqError::Msg(format!(
                        "QMatMul: input last dim {} != weight in_features {}",
                        last_dim, in_features
                    )));
                }
                let batch_elements: usize = x_shape[..x_shape.len() - 1].iter().product();
                // If tensor is on GPU but we're in CPU path (prefill batch>1), download first
                let x_data = if xs.is_cuda() { xs.to_vec1()? } else { xs.as_slice().to_vec() };
                let result = backend.qmatmul(&x_data, qw, batch_elements, in_features, out_features);
                let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
                out_shape.push(out_features);
                Tensor::from_vec(result, out_shape, xs.device())
            }
            qmm::QMatMul::Full(w) => {
                // GPU: both on GPU → cuBLAS matmul (stays on GPU)
                #[cfg(feature = "cuda")]
                if xs.is_cuda() {
                    let w_gpu = w.to_device_auto()?;
                    let wt = w_gpu.t()?;
                    return xs.matmul(&wt);
                }
                // CPU: x @ W^T via backend
                let x_shape = xs.shape().to_vec();
                let w_shape = w.shape().to_vec();
                let k = *x_shape.last().unwrap();
                let n = w_shape[0];
                let m: usize = x_shape[..x_shape.len() - 1].iter().product();
                let result = backend.matmul(xs.as_slice(), w.as_slice(), m, k, n);
                let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
                out_shape.push(n);
                Tensor::from_vec(result, out_shape, xs.device())
            }
        }
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
    fn forward(&self, xs: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs, backend)?;
        let w3 = self.feed_forward_w3.forward(xs, backend)?;
        let activated = fused_silu_mul(&w1, &w3, backend)?;
        self.feed_forward_w2.forward(&activated, backend)
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
    fn forward(&self, xs: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        let up = self.ffn_up.forward(xs, backend)?;
        let chunks = up.chunk(2, xs.rank() - 1)?;
        let activated = fused_silu_mul(&chunks[0], &chunks[1], backend)?;
        self.ffn_down.forward(&activated, backend)
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
    fn forward(&self, xs: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(vec![b_size * seq_len, hidden_dim])?;
                let router_logits = feed_forward_gate_inp.forward(&xs, backend)?;
                let routing_weights = softmax_last_dim(&router_logits, backend)?;
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
                    let current_hidden_states = expert_layer.forward(&current_state, backend)?;
                    let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }
                ys.reshape(vec![b_size, seq_len, hidden_dim])
            }
            Self::Mlp(mlp) => mlp.forward(xs, backend),
            Self::UpDown(mlp) => mlp.forward(xs, backend),
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

/// Persistent GPU buffers for compressed KV data.
/// Pre-allocated to max_seq, incrementally appended at compression time.
/// Eliminates per-step CPU→GPU upload overhead for fused attention.
#[cfg(feature = "cuda")]
struct GpuCompressedKv {
    /// Packed indices: [n_kv_head, max_seq, bytes_per_key] (flat)
    packed_indices: cudarc::driver::CudaSlice<u8>,
    /// Per-vector norms: [n_kv_head, max_seq]
    norms: cudarc::driver::CudaSlice<f32>,
    /// Codebook centroids: [n_centroids] — uploaded once
    centroids: cudarc::driver::CudaSlice<f32>,
    /// V data (f32): [n_kv_head, max_seq, head_dim]
    v_data: cudarc::driver::CudaSlice<f32>,
    /// Scratch buffers (reused across calls)
    rotated_q: cudarc::driver::CudaSlice<f32>,  // [n_heads, head_dim]
    output_buf: cudarc::driver::CudaSlice<f32>,  // [n_heads, head_dim]
    /// Current count of compressed keys per head
    count: usize,
    max_seq: usize,
    n_kv_head: usize,
    n_heads: usize,
    head_dim: usize,
    bytes_per_key: usize,
    bits: u8,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuCompressedKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuCompressedKv").field("count", &self.count).field("max_seq", &self.max_seq).finish()
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuCompressedKv {
    fn clone(&self) -> Self { panic!("GpuCompressedKv cannot be cloned") }
}

#[cfg(feature = "cuda")]
impl GpuCompressedKv {
    fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        n_kv_head: usize,
        n_heads: usize,
        head_dim: usize,
        bits: u8,
        max_seq: usize,
        centroids: &[f32],
    ) -> std::result::Result<Self, crate::cuda::TqError> {
        let bytes_per_key = (head_dim * bits as usize + 7) / 8;
        let total_indices = n_kv_head * max_seq * bytes_per_key;
        let total_norms = n_kv_head * max_seq;
        let total_v = n_kv_head * max_seq * head_dim;

        let _ = stream.context().check_err();
        let packed_indices = stream.alloc_zeros::<u8>(total_indices)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv indices: {}", e)))?;
        let norms = stream.alloc_zeros::<f32>(total_norms)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv norms: {}", e)))?;
        let v_data = stream.alloc_zeros::<f32>(total_v)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv v: {}", e)))?;
        let gpu_centroids = stream.clone_htod(centroids)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv centroids: {}", e)))?;
        let rotated_q = stream.alloc_zeros::<f32>(n_heads * head_dim)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv rq: {}", e)))?;
        let output_buf = stream.alloc_zeros::<f32>(n_heads * head_dim)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv out: {}", e)))?;

        eprintln!("[gpu-tq] pre-allocated {}×{}×{} = {:.1}MB compressed KV",
            n_kv_head, max_seq, bytes_per_key,
            (total_indices + total_norms * 4 + total_v * 4) as f64 / 1e6);

        Ok(Self {
            packed_indices, norms, centroids: gpu_centroids, v_data,
            rotated_q, output_buf,
            count: 0, max_seq, n_kv_head, n_heads, head_dim, bytes_per_key, bits,
            stream,
        })
    }

    /// Append one compressed token's data for all heads.
    /// packed: [n_kv_head * bytes_per_key], head_norms: [n_kv_head], v_flat: [n_kv_head * head_dim]
    fn append(
        &mut self,
        packed: &[u8],
        head_norms: &[f32],
        v_flat: &[f32],
    ) -> std::result::Result<(), crate::cuda::TqError> {
        if self.count >= self.max_seq {
            return Err(crate::cuda::TqError::Msg("GpuCompressedKv overflow".into()));
        }
        let bpk = self.bytes_per_key;
        let pos = self.count;

        // Upload packed indices per head at correct offset
        for h in 0..self.n_kv_head {
            let src = &packed[h * bpk..(h + 1) * bpk];
            let dst_off = (h * self.max_seq + pos) * bpk;
            let _ = self.stream.memcpy_htod(src, &mut self.packed_indices.slice_mut(dst_off..dst_off + bpk));
        }

        // Upload norms per head
        for h in 0..self.n_kv_head {
            let dst_off = h * self.max_seq + pos;
            let _ = self.stream.memcpy_htod(&head_norms[h..h+1], &mut self.norms.slice_mut(dst_off..dst_off + 1));
        }

        // Upload V data per head
        let hd = self.head_dim;
        for h in 0..self.n_kv_head {
            let src = &v_flat[h * hd..(h + 1) * hd];
            let dst_off = (h * self.max_seq + pos) * hd;
            let _ = self.stream.memcpy_htod(src, &mut self.v_data.slice_mut(dst_off..dst_off + hd));
        }

        self.count += 1;
        Ok(())
    }

    fn reset(&mut self) {
        self.count = 0;
    }
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
    // GPU fused attention is triggered dynamically (no persistent state needed).
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
    if let Some(skip) = config.skip_layers {
        return skip;
    }
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TQ_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(TQ_SKIP_FIRST_LAYERS)
    })
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
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TQ_PROTECT_LAST").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    })
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
    static C: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_SPARSE_V").ok().and_then(|v| v.parse().ok()).unwrap_or(TQ_SPARSE_V_DEFAULT))
}

/// Fused attention: compute attention scores directly from compressed indices
/// instead of decompressing keys first. Saves memory bandwidth on CPU.
/// Set TQ_FUSED=1 to enable. Default: off (decompress path).
fn get_use_fused() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_FUSED").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false))
}

fn get_bias_correction() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_BIAS_CORRECT").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false))
}

fn get_pre_rope() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_PRE_ROPE").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false))
}

fn get_compact_threshold() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_COMPACT").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

fn get_compact_ratio() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_COMPACT_RATIO").ok().and_then(|v| v.parse().ok()).unwrap_or(5))
}

const TQ_VBITS_DEFAULT: u8 = 0;

fn get_value_bits() -> u8 {
    static C: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_VBITS").ok().and_then(|v| v.parse().ok()).unwrap_or(TQ_VBITS_DEFAULT))
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

// ============================================================
// Pre-allocated GPU KV Cache (for CUDA Graph replay)
// ============================================================

/// Maximum sequence length for pre-allocated KV cache.
/// Maximum pre-allocated KV cache sequence length (for graph-compatible padded attention).
/// Configurable via TQ_MAX_SEQ env var. Default 128 is efficient for benchmarks.
/// For production: set higher (e.g., 2048 or 4096).
fn get_max_kv_seq() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_MAX_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(128))
}

/// Pre-allocated scratch buffers for zero-alloc decode (seq_len=1).
///
/// During decode, each layer's fused kernels write into these fixed buffers
/// instead of allocating fresh GPU memory. Only one layer executes at a time,
/// so buffers are safely reused across all 28 layers.
///
/// Buffers are `Arc<CudaSlice<f32>>` so we can:
/// - `Arc::get_mut` → `&mut CudaSlice<f32>` for kernel writes (refcount==1 between layers)
/// - `Arc::clone` → cheap tensor view for downstream ops (refcount==2 during layer)
/// - After layer completes, tensor drops → refcount back to 1
#[cfg(feature = "cuda")]
struct DecodeScratch {
    /// QKV output buffers (fused norm+QKV kernel writes here)
    q_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    k_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    v_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// MLP intermediate buffer (fused gateup+silu kernel writes here)
    intermediate_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Fused attention output (gqa_decode_attention writes here)
    attn_out: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Wo projection output (qmatmul_gpu_into writes here)
    wo_out: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Combined residual + attn — ping-pong pair.
    /// Even layers write to [0], odd to [1]. Previous layer's output is read from the other.
    combined_bufs: [std::sync::Arc<cudarc::driver::CudaSlice<f32>>; 2],
    /// Cached stream for tensor wrapping
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// Dimension metadata
    q_out: usize,
    k_out: usize,
    v_out: usize,
    intermediate_dim: usize,
    hidden_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

#[cfg(feature = "cuda")]
impl DecodeScratch {
    fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        q_out: usize,
        k_out: usize,
        v_out: usize,
        intermediate_dim: usize,
        hidden_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> std::result::Result<Self, crate::cuda::TqError> {
        let alloc = |name: &str, size: usize| -> std::result::Result<cudarc::driver::CudaSlice<f32>, crate::cuda::TqError> {
            crate::cuda::gpu_alloc_zeros_pub(&stream, size)
                .map_err(|e| crate::cuda::TqError::Msg(format!("DecodeScratch {} alloc: {}", name, e)))
        };
        let q_buf = alloc("q", q_out)?;
        let k_buf = alloc("k", k_out)?;
        let v_buf = alloc("v", v_out)?;
        let intermediate_buf = alloc("intermediate", intermediate_dim)?;
        let attn_out = alloc("attn_out", q_out)?;
        let wo_out = alloc("wo_out", hidden_dim)?;
        let combined_a = alloc("combined_a", hidden_dim)?;
        let combined_b = alloc("combined_b", hidden_dim)?;
        let total_kb = (q_out * 2 + k_out + v_out + intermediate_dim + hidden_dim * 3) * 4 / 1024;
        eprintln!("  DecodeScratch allocated: qkv={}+{}+{} attn={} wo={} comb={}x2 inter={} ({}KB)",
            q_out, k_out, v_out, q_out, hidden_dim, hidden_dim, intermediate_dim, total_kb);
        Ok(Self {
            q_buf: std::sync::Arc::new(q_buf),
            k_buf: std::sync::Arc::new(k_buf),
            v_buf: std::sync::Arc::new(v_buf),
            intermediate_buf: std::sync::Arc::new(intermediate_buf),
            attn_out: std::sync::Arc::new(attn_out),
            wo_out: std::sync::Arc::new(wo_out),
            combined_bufs: [std::sync::Arc::new(combined_a), std::sync::Arc::new(combined_b)],
            stream,
            q_out,
            k_out,
            v_out,
            intermediate_dim,
            hidden_dim,
            n_head,
            n_kv_head,
            head_dim,
        })
    }
}

/// Pre-allocated GPU KV cache for CUDA Graph compatible inference.
///
/// Layout: K and V are flat `[n_kv_head * max_seq * head_dim]` buffers.
/// Per-head data is contiguous: head h starts at `h * max_seq * head_dim`.
/// New tokens are appended at `h * max_seq * head_dim + seq_len * head_dim`.
///
/// For attention, the full buffer is used as `[1, n_kv_head, max_seq, head_dim]`
/// with positions >= seq_len masked to -inf in the attention score.
#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuKvCache")
            .field("seq_len", &self.seq_len)
            .field("max_seq", &self.max_seq)
            .field("n_kv_head", &self.n_kv_head)
            .field("head_dim", &self.head_dim)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuKvCache {
    fn clone(&self) -> Self {
        // Shallow clone: Arc bump (zero-copy GPU buffers)
        Self {
            k_buf: self.k_buf.clone(),
            v_buf: self.v_buf.clone(),
            mask_buf: self.mask_buf.clone(),
            valid_len_gpu: self.stream.clone_htod(&[self.seq_len as i32]).unwrap(),
            seq_len: self.seq_len,
            max_seq: self.max_seq,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
            stream: self.stream.clone(),
        }
    }
}

#[cfg(feature = "cuda")]
struct GpuKvCache {
    k_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    v_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Attention padding mask: [max_seq]. 0.0 for valid, -1e10 for padding.
    mask_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// GPU scalar for current valid length (read by mask generation kernel).
    /// Updated via clone_htod before graph replay — graph-safe pointer.
    valid_len_gpu: cudarc::driver::CudaSlice<i32>,
    seq_len: usize,
    max_seq: usize,
    n_kv_head: usize,
    head_dim: usize,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        n_kv_head: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, crate::cuda::TqError> {
        let total = n_kv_head * max_seq * head_dim;
        let k_buf = crate::cuda::gpu_alloc_zeros_pub(&stream, total)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache k alloc: {}", e)))?;
        let v_buf = crate::cuda::gpu_alloc_zeros_pub(&stream, total)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache v alloc: {}", e)))?;
        let mask_buf = crate::cuda::gpu_alloc_zeros_pub(&stream, max_seq)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache mask alloc: {}", e)))?;
        let valid_len_gpu = stream.clone_htod(&[0i32])
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache len alloc: {}", e)))?;
        Ok(Self {
            k_buf: std::sync::Arc::new(k_buf),
            v_buf: std::sync::Arc::new(v_buf),
            mask_buf: std::sync::Arc::new(mask_buf),
            valid_len_gpu,
            seq_len: 0,
            max_seq,
            n_kv_head,
            head_dim,
            stream,
        })
    }

    fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Append new K/V tokens. Input shape: [1, n_kv_head, n_new, head_dim].
    fn append(&mut self, new_k: &Tensor, new_v: &Tensor, n_new: usize) -> Result<()> {
        if self.seq_len + n_new > self.max_seq {
            bail!("GpuKvCache: overflow {} + {} > {}", self.seq_len, n_new, self.max_seq);
        }
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;
        let k_src = new_k.cuda_data();
        let v_src = new_v.cuda_data();

        // During graph capture, skip memcpy_htod — it would be captured as a
        // graph node with baked host address. The pre-replay code updates valid_len_gpu.
        #[cfg(feature = "cuda")]
        let capturing = {
            match self.stream.capture_status() {
                Ok(cudarc::driver::sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_ACTIVE) => true,
                _ => false,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let capturing = false;

        if !capturing {
            let pos_val = self.seq_len as i32;
            let _ = self.stream.memcpy_htod(&[pos_val], &mut self.valid_len_gpu);
        }

        // Single kernel per K and V: handles all heads, reads seq_pos from GPU scalar.
        let k_refs = std::sync::Arc::strong_count(&self.k_buf);
        if k_refs > 1 {
            eprintln!("[WARN] GpuKvCache::append k_buf Arc ref_count={} — external alias will see stale data", k_refs);
        }
        let k_dst = std::sync::Arc::make_mut(&mut self.k_buf);
        crate::cuda::kernels::kv_cache_append(
            reg, k_src, k_dst, &self.valid_len_gpu,
            self.n_kv_head, self.max_seq, self.head_dim, n_new,
        ).map_err(|e| TqError::Msg(format!("kv append k: {}", e)))?;

        let v_refs = std::sync::Arc::strong_count(&self.v_buf);
        if v_refs > 1 {
            eprintln!("[WARN] GpuKvCache::append v_buf Arc ref_count={} — external alias will see stale data", v_refs);
        }
        let v_dst = std::sync::Arc::make_mut(&mut self.v_buf);
        crate::cuda::kernels::kv_cache_append(
            reg, v_src, v_dst, &self.valid_len_gpu,
            self.n_kv_head, self.max_seq, self.head_dim, n_new,
        ).map_err(|e| TqError::Msg(format!("kv append v: {}", e)))?;

        self.seq_len += n_new;

        if !capturing {
            // Normal mode: update valid_len to post-append total, generate mask
            let new_len = self.seq_len as i32;
            let _ = self.stream.memcpy_htod(&[new_len], &mut self.valid_len_gpu);
            let mask_mut = std::sync::Arc::make_mut(&mut self.mask_buf);
            crate::cuda::kernels::generate_kv_mask(reg, mask_mut, &self.valid_len_gpu, self.max_seq, 0)
                .map_err(|e| TqError::Msg(format!("kv mask gen: {}", e)))?;
        } else {
            // Graph capture: valid_len_gpu has pre_pos (set by pre-replay code).
            // Generate mask with extra=n_new so valid_len = pre_pos + n_new = post_pos.
            // This kernel launch IS captured in the graph.
            let mask_mut = std::sync::Arc::make_mut(&mut self.mask_buf);
            crate::cuda::kernels::generate_kv_mask(reg, mask_mut, &self.valid_len_gpu, self.max_seq, n_new)
                .map_err(|e| TqError::Msg(format!("kv mask gen: {}", e)))?;
        }
        Ok(())
    }

    /// Get K as a Tensor wrapping the pre-allocated buffer.
    /// Shape: [1, n_kv_head, max_seq, head_dim] (full buffer, mask unused positions).
    fn k_tensor(&self) -> Tensor {
        Tensor::from_cuda_arc(
            self.k_buf.clone(),
            vec![1, self.n_kv_head, self.max_seq, self.head_dim],
            self.stream.clone(),
        )
    }

    /// Get V as a Tensor wrapping the pre-allocated buffer.
    fn v_tensor(&self) -> Tensor {
        Tensor::from_cuda_arc(
            self.v_buf.clone(),
            vec![1, self.n_kv_head, self.max_seq, self.head_dim],
            self.stream.clone(),
        )
    }

    /// Get K narrowed to valid tokens only: [1, n_kv_head, seq_len, head_dim].
    /// No padding waste — attention only computes over actual positions.
    fn k_tensor_valid(&self) -> Result<Tensor> {
        let full = self.k_tensor();
        full.narrow(2, 0, self.seq_len)
    }

    /// Get V narrowed to valid tokens: [1, n_kv_head, seq_len, head_dim].
    fn v_tensor_valid(&self) -> Result<Tensor> {
        let full = self.v_tensor();
        full.narrow(2, 0, self.seq_len)
    }

    /// Get attention mask as tensor [1, 1, 1, max_seq] for broadcasting.
    fn mask_tensor(&self) -> Tensor {
        Tensor::from_cuda_arc(
            self.mask_buf.clone(),
            vec![1, 1, 1, self.max_seq],
            self.stream.clone(),
        )
    }
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
    /// Pre-allocated GPU KV cache (used when CUDA Graph is enabled).
    #[cfg(feature = "cuda")]
    gpu_kv_cache: Option<GpuKvCache>,
    /// Persistent GPU compressed KV buffers for fused TQ attention.
    #[cfg(feature = "cuda")]
    gpu_tq_cache: Option<GpuCompressedKv>,
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
        let (_b_sz, n_head, seq_len, _head_dim) = x.dims4()?;

        // GPU path: single kernel launch, no shape/stride metadata uploads.
        #[cfg(feature = "cuda")]
        if x.is_cuda() && self.cos.is_cuda() {
            if let Some(reg) = crate::cuda::kernels::global_registry() {
                let mut x_out = x.clone();
                // Check if graph-mode GPU position scalar is available
                let gpu_pos_ref = ROPE_POS_GPU_PTR.with(|p| p.get());
                if gpu_pos_ref != 0 {
                    // Graph mode: read position from GPU scalar
                    // SAFETY: pointer set by forward() before capture, valid for model lifetime
                    let gpu_slice = unsafe { &*(gpu_pos_ref as *const cudarc::driver::CudaSlice<i32>) };
                    let rope_with_gpu = match self.rope_style {
                        RopeStyle::Halved => crate::cuda::kernels::rope_halved_with_gpu_pos,
                        RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved_with_gpu_pos,
                    };
                    rope_with_gpu(
                        reg, x_out.cuda_data_mut(),
                        self.cos.cuda_data(), self.sin.cuda_data(),
                        seq_len, n_head, self.head_dim, self.rope_dim, index_pos,
                        Some(gpu_slice),
                    ).map_err(|e| TqError::Msg(format!("rope GPU: {}", e)))?;
                } else {
                    let rope_fn = match self.rope_style {
                        RopeStyle::Halved => crate::cuda::kernels::rope_halved,
                        RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved,
                    };
                    rope_fn(
                        reg, x_out.cuda_data_mut(),
                        self.cos.cuda_data(), self.sin.cuda_data(),
                        seq_len, n_head, self.head_dim, self.rope_dim, index_pos,
                    ).map_err(|e| TqError::Msg(format!("rope GPU: {}", e)))?;
                }
                return Ok(x_out);
            }
        }

        // CPU fallback: tensor-op RoPE
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let x = x.contiguous()?;
        if self.rope_dim < self.head_dim {
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
        pre_qkv: Option<(Tensor, Tensor, Tensor)>,
        mask: Option<&Tensor>,
        index_pos: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let used_fused_qkv = pre_qkv.is_some();
        let (mut q, mut k, mut v) = if let Some(qkv) = pre_qkv {
            qkv
        } else {
            match &self.qkv {
            QkvWeights::Separate { wq, wk, wv } => {
                (wq.forward(x, backend)?, wk.forward(x, backend)?, wv.forward(x, backend)?)
            }
            QkvWeights::Merged { wqkv } => {
                let qkv = wqkv.forward(x, backend)?;
                let q_size = self.n_head * self.head_dim;
                let kv_size = self.n_kv_head * self.head_dim;
                let q = qkv.narrow(2, 0, q_size)?;
                let k = qkv.narrow(2, q_size, kv_size)?;
                let v = qkv.narrow(2, q_size + kv_size, kv_size)?;
                (q, k, v)
            }
        }
        }; // end if let Some(qkv) / else

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

        // Apply biases if present (Qwen2 has them, Llama/Phi/Gemma don't).
        // Skip if pre_qkv was provided — fused kernel already includes bias.
        if !used_fused_qkv {
            if let Some(bq) = &self.attention_bq {
                q = q.broadcast_add(bq)?;
            }
            if let Some(bk) = &self.attention_bk {
                k = k.broadcast_add(bk)?;
            }
            if let Some(bv) = &self.attention_bv {
                v = v.broadcast_add(bv)?;
            }
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
                // Compressed path: quantize per head × position.
                // Use as_slice() for CPU or direct download for GPU (avoids extra allocs).
                let v_f32: Vec<f32> = if v.is_cuda() {
                    v.to_vec1()?
                } else {
                    v.as_slice().to_vec()
                };
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
            let _new_sink_count = if global_start < sink_n {
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
                // Direct download: skip contiguous+to_dtype+flatten chain for decode (already f32 contiguous)
                let k_flat: Vec<f32> = if k_to_compress.is_cuda() {
                    k_to_compress.contiguous()?.to_vec1()?
                } else {
                    k_to_compress.contiguous()?.as_slice().to_vec()
                };
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

            // --- GPU append: upload newly compressed tokens to persistent GPU buffers ---
            #[cfg(feature = "cuda")]
            if tokens_to_compress > 0 && seq_len == 1 {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    // Initialize GPU compressed KV on first use
                    if self.gpu_tq_cache.is_none() {
                        let head_bits = cache.k_per_head[0].bits;
                        let cal_cb: Option<Vec<f32>> = layer_tq_config.calibrated_codebook
                            .as_ref().map(|cb| cb.centroids.clone());
                        let cb: &[f32] = match cal_cb.as_deref() {
                            Some(cal) => cal,
                            None => tq_kv::codebook::get_centroids(head_bits),
                        };
                        let max_seq = std::env::var("TQ_MAX_SEQ").ok()
                            .and_then(|v| v.parse().ok()).unwrap_or(2048usize);
                        match GpuCompressedKv::new(
                            reg.stream.clone(), self.n_kv_head, self.n_head,
                            self.head_dim, head_bits, max_seq, cb,
                        ) {
                            Ok(mut gpu) => {
                                // Seed with all existing compressed keys
                                let existing = cache.k_per_head[0].count;
                                let bpk = gpu.bytes_per_key;
                                for pos in 0..existing {
                                    let mut packed_all = Vec::with_capacity(self.n_kv_head * bpk);
                                    let mut norms_all = Vec::with_capacity(self.n_kv_head);
                                    for h in 0..self.n_kv_head {
                                        let start = pos * bpk;
                                        packed_all.extend_from_slice(&cache.k_per_head[h].packed_indices[start..start + bpk]);
                                        norms_all.push(cache.k_per_head[h].norms[pos]);
                                    }
                                    // V data placeholder (zeros for now — V is handled separately)
                                    let v_zeros = vec![0.0f32; self.n_kv_head * self.head_dim];
                                    let _ = gpu.append(&packed_all, &norms_all, &v_zeros);
                                }
                                self.gpu_tq_cache = Some(gpu);
                            }
                            Err(e) => eprintln!("[gpu-tq] init failed: {}", e),
                        }
                    }

                    // Append the newly compressed token(s)
                    if let Some(ref mut gpu) = self.gpu_tq_cache {
                        let count = cache.k_per_head[0].count;
                        let bpk = gpu.bytes_per_key;
                        // The last token is at index count-1
                        let pos = count - 1;
                        let mut packed_all = Vec::with_capacity(self.n_kv_head * bpk);
                        let mut norms_all = Vec::with_capacity(self.n_kv_head);
                        for h in 0..self.n_kv_head {
                            let start = pos * bpk;
                            packed_all.extend_from_slice(&cache.k_per_head[h].packed_indices[start..start + bpk]);
                            norms_all.push(cache.k_per_head[h].norms[pos]);
                        }
                        let v_zeros = vec![0.0f32; self.n_kv_head * self.head_dim];
                        let _ = gpu.append(&packed_all, &norms_all, &v_zeros);
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
                let use_fused = get_use_fused() && q.is_cpu()
                    && !cache.pre_rope && !has_compacted;
                let n_compressed = cache.k_per_head[0].count;
                let n_past_compressed = if n_compressed > 0 { n_compressed - 1 } else { 0 };
                let compacted_t = cache.compacted.as_ref()
                    .map(|c| c[0].t).unwrap_or(0);

                // --- GPU Fused TQ Attention (compressed KV, no decompression) ---
                // GPU fused TQ attention: works even if q is CPU (we upload it).
                // Requirements: no pre-rope, no compacted, no cold, no sink, has past keys.
                #[cfg(feature = "cuda")]
                let gpu_fused_ok = crate::cuda::kernels::global_registry().is_some()
                    && !cache.pre_rope && !has_compacted
                    && cache.cold_len == 0 && cache.sink_len == 0
                    && n_past_compressed > 0 && seq_len == 1;
                #[cfg(not(feature = "cuda"))]
                let gpu_fused_ok = false;

                #[cfg(feature = "cuda")]
                if gpu_fused_ok {
                    if let (Some(reg), Some(ref mut gpu)) = (crate::cuda::kernels::global_registry(), &mut self.gpu_tq_cache) {
                        let n_keys = n_past_compressed;

                        // Pre-rotate query (Hadamard signs) — all heads
                        let q_flat = q_f32.flatten_all()?.to_vec1()?;
                        let mut rotated_q = Vec::with_capacity(self.n_head * self.head_dim);
                        for qh in 0..self.n_head {
                            let qstart = qh * self.head_dim;
                            let q_vec = &q_flat[qstart..qstart + self.head_dim];
                            let rq = tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs);
                            rotated_q.extend_from_slice(&rq);
                        }

                        // Upload only the rotated query (small: n_heads * head_dim * 4 bytes)
                        let _ = reg.stream.context().check_err();
                        let _ = reg.stream.memcpy_htod(&rotated_q, &mut gpu.rotated_q);

                        let scale = 1.0 / (self.head_dim as f32).sqrt();
                        crate::cuda::kernels::tq_fused_decode_attention(
                            reg, &gpu.rotated_q, &gpu.packed_indices, &gpu.norms,
                            &gpu.centroids, &gpu.v_data, &mut gpu.output_buf,
                            self.n_head, self.n_kv_head, n_keys, self.head_dim,
                            gpu.bits as usize, scale,
                        ).map_err(|e| TqError::Msg(format!("fused decode attention: {}", e)))?;

                        // Download output
                        let _ = reg.stream.context().check_err();
                        let attn_out = reg.stream.clone_dtoh(&gpu.output_buf)
                            .map_err(|e| TqError::Msg(format!("fused attn download: {}", e)))?;

                        let y = Tensor::from_vec(attn_out, vec![1, self.n_head, 1, self.head_dim], q.device())?;
                        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
                        return self.attention_wo.forward(&y, backend);
                    }
                }

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
                    softmax_last_dim(&att, backend)?
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
                        let _ct = comp[0].t;
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

                    softmax_last_dim(&att, backend)?
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

                } else if sparse_thresh > 0.0 && att.is_cpu() && cache.value_bits > 0 {
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

                    if sparse_thresh > 0.0 && att.is_cpu() {
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
                            // Additive mask: 0 for valid, -1e10 for masked
                            let mask4d = mask.unsqueeze(0)?.unsqueeze(0)?;
                            let mask4d = mask4d.broadcast_as(att.shape())?;
                            (att + mask4d)?
                        }
                    };
                    let att = softmax_last_dim(&att, backend)?;
                    att.matmul(&v_for_attn.contiguous()?)?
                }
            };

            let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
            self.attention_wo.forward(&y, backend)
        } else {
            // UNCOMPRESSED PATH: standard fp16 KV cache (first N layers)
            //
            // GPU path: pre-allocated GpuKvCache with in-place append + narrow.
            //   Eliminates Tensor::cat (which copies entire history per step).
            //   Uses narrow() for exact-length attention (no padding waste).
            // CPU path: Tensor::cat fallback.
            #[cfg(feature = "cuda")]
            let gpu_kv_disabled = {
                static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *C.get_or_init(|| std::env::var("TQ_NO_GPU_KV").map(|v| v == "1").unwrap_or(false))
            };
            #[cfg(feature = "cuda")]
            if k.is_cuda() && seq_len == 1 && !gpu_kv_disabled {
                // Initialize GPU KV cache on first decode step
                if self.gpu_kv_cache.is_none() {
                    if let Some(reg) = crate::cuda::kernels::global_registry() {
                        match GpuKvCache::new(reg.stream.clone(), self.n_kv_head, self.head_dim, get_max_kv_seq()) {
                            Ok(mut cache) => {
                                // Seed with prefill KV data from the CPU-path kv_cache
                                if let Some((prev_k, prev_v)) = &self.kv_cache {
                                    let prefill_len = prev_k.shape()[2];
                                    let gk = if prev_k.is_cuda() { prev_k.clone() } else {
                                        prev_k.to_device_auto().unwrap_or_else(|_| prev_k.clone())
                                    };
                                    let gv = if prev_v.is_cuda() { prev_v.clone() } else {
                                        prev_v.to_device_auto().unwrap_or_else(|_| prev_v.clone())
                                    };
                                    if gk.is_cuda() && gv.is_cuda() {
                                        if let Err(e) = cache.append(&gk, &gv, prefill_len) {
                                            eprintln!("[gpu-kv] L{}: seed failed: {}", self.layer_idx, e);
                                        }
                                    }
                                }
                                self.gpu_kv_cache = Some(cache);
                            }
                            Err(e) => eprintln!("[gpu-kv] L{}: alloc failed: {}", self.layer_idx, e),
                        }
                    }
                }
                if let Some(ref mut gpu_kv) = self.gpu_kv_cache {
                    if index_pos == 0 {
                        gpu_kv.reset();
                    }
                    // Append new K/V to pre-allocated buffers (in-place, zero-copy)
                    gpu_kv.append(&k, &v, seq_len)?;

                    // Padded attention with mask (cuBLAS matmul is faster than custom kernel).
                    let k_full = gpu_kv.k_tensor();
                    let v_full = gpu_kv.v_tensor();
                    let kv_mask = gpu_kv.mask_tensor();

                    let k_full = repeat_kv(k_full, n_rep)?;
                    let v_full = repeat_kv(v_full, n_rep)?;

                    let att = (q.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;
                    let kv_mask = kv_mask.broadcast_as(att.shape())?;
                    let att = (att + kv_mask)?;

                    let gpu_debug = {
                        static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        *C.get_or_init(|| std::env::var("TQ_GPU_DEBUG").map(|v| v == "1").unwrap_or(false))
                    };
                    if gpu_debug && self.layer_idx < 2 {
                        if let Ok(scores) = att.to_vec1() {
                            let max_seq_show = gpu_kv.seq_len.min(10);
                            let head0_scores: Vec<f32> = scores[..max_seq_show].to_vec();
                            eprintln!("[gpu-debug] L{} GpuKV att-pre-softmax head0[..{}]: {:?}",
                                self.layer_idx, max_seq_show, head0_scores);
                            // Show mask values too
                            let n_pad = scores.len() / self.n_head; // max_seq
                            let n_masked = scores[gpu_kv.seq_len..n_pad].iter()
                                .filter(|&&v| v < -1e9).count();
                            eprintln!("[gpu-debug] L{} GpuKV valid={} max_seq={} masked_positions={}",
                                self.layer_idx, gpu_kv.seq_len, gpu_kv.max_seq, n_masked);
                        }
                    }

                    let att = softmax_last_dim(&att, backend)?;

                    // Debug: attention weights after softmax (head 0)
                    if gpu_debug && self.layer_idx < 2 {
                        if let Ok(weights) = att.to_vec1() {
                            let max_seq_show = gpu_kv.seq_len.min(10);
                            let head0_weights: Vec<f32> = weights[..max_seq_show].to_vec();
                            let head0_sum: f32 = weights[..gpu_kv.max_seq].iter().sum();
                            eprintln!("[gpu-debug] L{} GpuKV att-post-softmax head0[..{}]: {:?} sum={:.6}",
                                self.layer_idx, max_seq_show, head0_weights, head0_sum);
                        }
                    }

                    let y = att.matmul(&v_full.contiguous()?)?;

                    let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
                    return self.attention_wo.forward(&y, backend);
                }
            }

            // CPU fallback / prefill: original cat()-based KV cache
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
                    // Additive mask: 0 for valid, -1e10 for masked
                    let mask4d = mask.unsqueeze(0)?.unsqueeze(0)?;
                    let mask4d = mask4d.broadcast_as(att.shape())?;
                    (att + mask4d)?
                }
            };
            let att = softmax_last_dim(&att, backend)?;
            let y = att.matmul(&v.contiguous()?)?;

            let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
            self.attention_wo.forward(&y, backend)
        }
    }
}

// ============================================================
// Generic TurboQuant Model
// ============================================================

pub struct GenericTurboModel {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    backend: Arc<dyn ComputeBackend>,
    span: tracing::Span,
    span_output: tracing::Span,
    /// CUDA Graph manager for decode acceleration.
    #[cfg(feature = "cuda")]
    graph_manager: crate::cuda::graph::CudaGraphManager,
    /// Cached output tensor from graph capture (for replay).
    #[cfg(feature = "cuda")]
    graph_output: Option<Tensor>,
    /// GPU buffers kept alive for graph replay (prevents ILLEGAL_ADDRESS from freed pointers).
    #[cfg(feature = "cuda")]
    graph_retained_buffers: Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// Pre-allocated intermediate buffers for graph capture (DecodeBufferPool).
    #[cfg(feature = "cuda")]
    graph_pool_buffers: Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// GPU buffer address of the input embedding (for updating before graph replay).
    #[cfg(feature = "cuda")]
    graph_input_buffer: Option<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// GPU scalar for RoPE position offset (updated before graph replay).
    #[cfg(feature = "cuda")]
    rope_pos_gpu: Option<cudarc::driver::CudaSlice<i32>>,
    /// Arena buffer pool for decode (reuses allocs across decode steps without CUDA Graph).
    #[cfg(feature = "cuda")]
    arena_pool_buffers: Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    /// Number of decode steps completed (0 = first decode records, 1+ = arena reuse).
    #[cfg(feature = "cuda")]
    arena_decode_count: usize,
    /// Pre-allocated scratch buffers for zero-alloc decode (fused kernel path).
    #[cfg(feature = "cuda")]
    decode_scratch: Option<DecodeScratch>,
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
                #[cfg(feature = "cuda")]
                gpu_kv_cache: None,
                #[cfg(feature = "cuda")]
                gpu_tq_cache: None,
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

        let backend = crate::backend::create_backend();

        let mut model = Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
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
        if crate::cuda::kernels::global_registry().is_some() {
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
        } // end if do_warmup

        Ok(model)
    }

    fn mask(&mut self, t: usize, _device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            // Additive causal mask: 0.0 for valid (j <= i), -1e10 for masked (j > i).
            // Can be added directly to attention scores — no masked_fill needed.
            let mask: Vec<f32> = (0..t)
                .flat_map(|i| (0..t).map(move |j| if j > i { -1e10f32 } else { 0.0f32 }))
                .collect();
            let mut mask = Tensor::from_slice(&mask, vec![t, t], _device)?;
            // Upload to GPU if available
            #[cfg(feature = "cuda")]
            if let Ok(gpu) = mask.to_device_auto() {
                mask = gpu;
            }
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
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
        #[cfg(feature = "cuda")]
        let mut graph_replayed = false;
        #[cfg(feature = "cuda")]
        if seq_len == 1 && self.graph_manager.is_ready(1) {
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

            let _ = reg.stream.synchronize();
            if let Ok(()) = self.graph_manager.replay(&reg.stream, 1) {
                let _ = reg.stream.synchronize();
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
        let capturing = seq_len == 1 && self.graph_manager.should_capture(1);
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
                    ROPE_POS_GPU_PTR.with(|p| p.set(buf as *const _ as u64));
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
            Some(self.mask(seq_len, x.device())?)
        };
        let _enter = self.span.enter();
        let backend = self.backend.clone();
        let backend = backend.as_ref();
        let mut layer_in = self.tok_embeddings.forward(x)?;

        // Phase 3: Upload embedding to GPU for GPU-resident forward pass.
        // All subsequent ops auto-dispatch to GPU when tensor is CUDA.
        #[cfg(feature = "cuda")]
        if crate::cuda::kernels::global_registry().is_some() {
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

        // Graph replayed: skip layer loop, set layer_in from scratch combined buffer.
        #[cfg(feature = "cuda")]
        if graph_replayed {
            if let Some(ref scratch) = decode_scratch {
                let last_ci = (self.layers.len() - 1) & 1;
                layer_in = Tensor::from_cuda_arc(
                    Arc::clone(&scratch.combined_bufs[last_ci]),
                    vec![1, 1, scratch.hidden_dim], scratch.stream.clone(),
                );
            }
        }
        #[cfg(not(feature = "cuda"))]
        let graph_replayed = false;

        if !graph_replayed {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
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
                        let rope_gpu_pos = ROPE_POS_GPU_PTR.with(|p| p.get());
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
        } // if !graph_replayed

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
        let x = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let _enter = self.span_output.enter();
        #[cfg(feature = "cuda")]
        let _t_lm = if profiling {
            if let Some(ref s) = prof_stream { let _ = s.synchronize(); }
            Some(std::time::Instant::now())
        } else { None };
        #[cfg(not(feature = "cuda"))]
        let _t_lm: Option<std::time::Instant> = None;
        let output = self.output.forward(&x, backend)?;
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
