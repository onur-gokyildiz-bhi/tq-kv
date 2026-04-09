//! Shared primitives: Module trait, Embedding, RmsNorm, softmax, softcap, fused SiLU.

use crate::backend::ComputeBackend;
use crate::cuda::{TqTensor as Tensor, TqDevice as Device, TqDType as DType, TqError};
use crate::cuda::Result;
use crate::gguf::GgmlDType;
use crate::qmatmul as qmm;

/// Bail macro compatible with our error type.
macro_rules! bail {
    ($($arg:tt)*) => { return Err(TqError::Msg(format!($($arg)*))) };
}

// ============================================================
// Traits and primitives (replaces candle_nn)
// ============================================================

/// Module trait — replaces candle_nn::Module.
pub(crate) trait Module {
    fn forward(&self, x: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor>;
}

/// Embedding lookup — replaces candle_nn::Embedding.
#[derive(Debug, Clone)]
pub(crate) struct Embedding {
    /// Full f32 weight (when dequantized at load) or None (lazy mode)
    pub(crate) weight: Option<Tensor>,
    /// Quantized raw data + dtype for lazy per-row dequant (saves ~2 GB)
    raw_data: Option<Vec<u8>>,
    raw_dtype: Option<crate::gguf::GgmlDType>,
    pub(crate) hidden_size: usize,
}

impl Embedding {
    pub(crate) fn new(weight: Tensor, hidden_size: usize) -> Self {
        Self { weight: Some(weight), raw_data: None, raw_dtype: None, hidden_size }
    }

    /// Lazy embedding: keep quantized data, dequant only needed rows on lookup.
    /// Saves ~2 GB for Qwen2 7B (151K vocab × 3584 dim × 4 bytes = 2.1 GB).
    pub(crate) fn new_lazy(raw_data: Vec<u8>, dtype: crate::gguf::GgmlDType, hidden_size: usize) -> Self {
        Self { weight: None, raw_data: Some(raw_data), raw_dtype: Some(dtype), hidden_size }
    }

    pub(crate) fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let shape = ids.shape().to_vec();
        let ids_flat = ids.to_vec1()?;
        let n_tokens = ids_flat.len();

        if let Some(ref w_tensor) = self.weight {
            // Full f32 path (original)
            let w = w_tensor.as_slice();
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
            let mut out_shape = shape;
            out_shape.push(self.hidden_size);
            Tensor::from_vec(output, out_shape, ids.device())
        } else if let (Some(ref raw), Some(dtype)) = (&self.raw_data, self.raw_dtype) {
            // Lazy dequant: only dequant requested rows
            let block_numel = dtype.block_numel();
            let block_bytes = dtype.block_size_bytes();
            let blocks_per_row = (self.hidden_size + block_numel - 1) / block_numel;
            let bytes_per_row = blocks_per_row * block_bytes;

            let mut output = Vec::with_capacity(n_tokens * self.hidden_size);
            for &id in &ids_flat {
                let idx = id as usize;
                let row_start = idx * bytes_per_row;
                let row_end = row_start + bytes_per_row;
                if row_end <= raw.len() {
                    let row_f32 = crate::quant::dequantize(&raw[row_start..row_end], dtype, self.hidden_size);
                    output.extend_from_slice(&row_f32);
                } else {
                    output.extend(std::iter::repeat(0.0f32).take(self.hidden_size));
                }
            }
            let mut out_shape = shape;
            out_shape.push(self.hidden_size);
            Tensor::from_vec(output, out_shape, &Device::Cpu)
        } else {
            bail!("Embedding: no weight data")
        }
    }
}

// ============================================================
// Shared primitives (CUDA-compatible)
// ============================================================

/// Device-aware RmsNorm — uses primitive ops that work on both CPU and GPU.
#[derive(Debug, Clone)]
pub(crate) struct RmsNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f64,
    /// Gemma: use (1 + weight) instead of weight
    pub(crate) add_unit: bool,
    span: tracing::Span,
}

impl RmsNorm {
    pub(crate) fn from_qweight(raw: &[u8], dtype: GgmlDType, n_elements: usize, eps: f64, device: &Device) -> Result<Self> {
        let data = crate::quant::dequantize(raw, dtype, n_elements);
        let weight = Tensor::from_vec(data, vec![n_elements], device)?;
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        Ok(Self { weight, eps, add_unit: false, span })
    }

    /// Create from QWeight (candle API compat).
    pub(crate) fn from_qtensor(qw: qmm::QWeight, eps: f64, device: &Device) -> Result<Self> {
        let n_elements = qw.shape.0 * qw.shape.1;
        Self::from_qweight(&qw.raw_data, qw.dtype, n_elements, eps, device)
    }

    /// Set Gemma-style weight offset: bake (1 + w) into stored weight
    pub(crate) fn with_add_unit(mut self) -> Self {
        let w = self.weight.as_slice();
        let w_plus_one: Vec<f32> = w.iter().map(|&v| 1.0 + v).collect();
        self.weight = Tensor::from_vec(w_plus_one, self.weight.shape().to_vec(), self.weight.device())
            .expect("add_unit weight creation");
        self.add_unit = true;
        self
    }

    pub(crate) fn forward(&self, x: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_f32 = x.to_dtype(DType::F32)?;

        // GPU-resident path: use TqTensor::rms_norm_gpu directly
        // (add_unit already baked into self.weight at construction time)
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
pub(crate) fn softmax_last_dim(x: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
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

/// Logit soft-capping: cap * tanh(logits / cap). Used by Gemma2.
pub(crate) fn apply_softcap(x: &Tensor, cap: f32) -> Result<Tensor> {
    // GPU path: in-place softcap kernel
    #[cfg(feature = "cuda")]
    if x.is_cuda() {
        if let Some(reg) = crate::cuda::kernels::global_registry() {
            let n = x.elem_count();
            let mut out = crate::cuda::gpu_alloc_zeros_pub(&reg.stream, n)
                .map_err(|e| TqError::Msg(format!("softcap alloc: {e}")))?;
            // Copy x → out, then softcap in-place
            let _ = reg.stream.memcpy_dtod(x.cuda_data(), &mut out);
            crate::cuda::kernels::softcap_inplace(reg, &mut out, cap, n)
                .map_err(|e| TqError::Msg(format!("softcap kernel: {e}")))?;
            return Ok(Tensor::from_cuda(out, x.shape().to_vec(), reg.stream.clone()));
        }
    }
    // CPU fallback
    let data = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let inv_cap = 1.0 / cap;
    let capped: Vec<f32> = data.iter().map(|&v| cap * (v * inv_cap).tanh()).collect();
    Tensor::from_vec(capped, x.shape().to_vec(), x.device())
}

/// Fused SiLU gate × up: silu(gate) * up — single pass, no intermediate tensor.
pub(crate) fn fused_silu_mul(gate: &Tensor, up: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if gate.is_cuda() && up.is_cuda() {
        return gate.fused_silu_mul_gpu(up);
    }
    let result = backend.fused_silu_mul(gate.as_slice(), up.as_slice());
    Tensor::from_vec(result, gate.shape().to_vec(), gate.device())
}
