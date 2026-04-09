//! QMatMul wrapper, MLP variants (standard, Phi-style, MoE), utility functions.

use crate::backend::ComputeBackend;
use crate::cuda::{TqTensor as Tensor, TqDType as DType, TqError};
use crate::cuda::Result;
use crate::qmatmul as qmm;

use super::primitives::{Module, softmax_last_dim, fused_silu_mul};

#[derive(Debug, Clone)]
pub(crate) struct QMatMul {
    pub(crate) inner: qmm::QMatMul,
    pub(crate) span: tracing::Span,
}

impl QMatMul {
    pub(crate) fn from_qweight(w: qmm::QWeight) -> Result<Self> {
        let inner = qmm::QMatMul::from_qweight(w);
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// Alias for from_qweight (candle API compat).
    pub(crate) fn from_qtensor(w: qmm::QWeight) -> Result<Self> {
        Self::from_qweight(w)
    }

    pub(crate) fn from_tensor(tensor: Tensor) -> Self {
        let inner = qmm::QMatMul::from_tensor(tensor);
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Self { inner, span }
    }

    /// Pre-warm weight cache: upload to GPU if available, CPU dequant only as fallback.
    pub(crate) fn warmup(&self, backend: &dyn ComputeBackend) {
        match &self.inner {
            qmm::QMatMul::Quantized(qw) => {
                if backend.is_gpu() {
                    // GPU available: upload raw quantized bytes, skip CPU f32 dequant
                    backend.warmup_qweight(qw);
                    return;
                }
                // CPU-only: dequant to f32 for SGEMM fallback
                qw.warmup_cpu();
                backend.warmup_qweight(qw);
            }
            qmm::QMatMul::Full(_) => {},
        }
    }

    /// Get inner QWeight if it's Q4_K_M quantized (for fused kernel launches).
    pub(crate) fn q4k_weight(&self) -> Option<&qmm::QWeight> {
        match &self.inner {
            qmm::QMatMul::Quantized(qw) if matches!(qw.dtype, crate::gguf::GgmlDType::Q4K) => Some(qw),
            _ => None,
        }
    }

    /// Get GPU-resident raw Q4_K_M weight bytes for fused kernel launches.
    #[cfg(feature = "cuda")]
    pub(crate) fn q4k_gpu_data(&self) -> Option<(&cudarc::driver::CudaSlice<u8>, usize, usize)> {
        let qw = self.q4k_weight()?;
        let reg = crate::cuda::kernels::global_registry()?;
        let gpu = qw.gpu_cache_or_upload(&reg.stream);
        Some((gpu, qw.out_features(), qw.in_features()))
    }

    pub(crate) fn forward(&self, xs: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        let _enter = self.span.enter();
        match &self.inner {
            qmm::QMatMul::Quantized(qw) => {
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
                let x_data = if xs.is_cuda() { xs.to_vec1()? } else { xs.as_slice().to_vec() };
                let result = backend.qmatmul(&x_data, qw, batch_elements, in_features, out_features);
                let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
                out_shape.push(out_features);
                Tensor::from_vec(result, out_shape, xs.device())
            }
            qmm::QMatMul::Full(w) => {
                #[cfg(feature = "cuda")]
                if xs.is_cuda() {
                    let w_gpu = w.to_device_auto()?;
                    let wt = w_gpu.t()?;
                    return xs.matmul(&wt);
                }
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

/// Activation type for gated MLP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum GateActivation {
    SiLU,  // Llama, Qwen, Mistral
    GELU,  // Gemma 2
}

/// Standard 3-gate MLP: gate (w1) + up (w3) → act(gate) * up → down (w2)
#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    pub(crate) feed_forward_w1: QMatMul,
    pub(crate) feed_forward_w2: QMatMul,
    pub(crate) feed_forward_w3: QMatMul,
    pub(crate) activation: GateActivation,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs, backend)?;
        let w3 = self.feed_forward_w3.forward(xs, backend)?;
        let activated = match self.activation {
            GateActivation::SiLU => fused_silu_mul(&w1, &w3, backend)?,
            GateActivation::GELU => {
                let w1_f32 = w1.to_dtype(DType::F32)?;
                let data = w1_f32.flatten_all()?.to_vec1()?;
                let sqrt_2_over_pi: f32 = 0.7978845608;
                let gelu: Vec<f32> = data.iter().map(|&x| {
                    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
                }).collect();
                let mut g = Tensor::from_vec(gelu, w1.shape().to_vec(), w1.device())?;
                #[cfg(feature = "cuda")]
                if w3.is_cuda() && !g.is_cuda() {
                    g = g.to_device_auto()?;
                }
                let w3_f32 = w3.to_dtype(DType::F32)?;
                (g * w3_f32)?
            }
        };
        self.feed_forward_w2.forward(&activated, backend)
    }
}

/// Phi-style 2-gate MLP: up projects to 2×intermediate, split into gate+up halves.
#[derive(Debug, Clone)]
pub(crate) struct MlpUpDown {
    pub(crate) ffn_up: QMatMul,
    pub(crate) ffn_down: QMatMul,
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
pub(crate) enum MlpOrMoe {
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

pub(crate) fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand(vec![b_sz, n_kv_head, n_rep, seq_len, head_dim])?
            .reshape(vec![b_sz, n_kv_head * n_rep, seq_len, head_dim])
    }
}

pub(crate) fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let n = mask.elem_count();
    let val = on_true.to_vec1()?[0];
    let expanded = Tensor::from_vec(vec![val; n], mask.shape().to_vec(), mask.device())?;
    mask.where_cond(&expanded, on_false)
}
