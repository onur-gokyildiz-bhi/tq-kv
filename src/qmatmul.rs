//! Quantized matrix multiplication — replacement for candle's QMatMul.
//!
//! Stores quantized weights in their raw GGML block format and dequantizes
//! on-the-fly during matmul. Phase 3 adds a fused CUDA kernel (q4km_matvec)
//! that avoids the intermediate f32 buffer entirely.
//!
//! Future: LoRA adapter support via `output = W*x + alpha * B*A*x`.

use crate::cuda::{TqTensor, TqDevice, Result, TqError};
use crate::gguf::GgmlDType;
use crate::quant;

#[cfg(feature = "cuda")]
use std::sync::OnceLock;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Quantized weight matrix stored in GGML block format.
///
/// When CUDA is enabled, raw weight bytes are lazily uploaded to GPU
/// on first use and cached for subsequent forward passes.
pub struct QWeight {
    /// Raw quantized bytes (GGML block layout).
    pub raw_data: Vec<u8>,
    /// Quantization type (Q4_K_M, Q6_K, Q8_0, F16, F32, ...).
    pub dtype: GgmlDType,
    /// Weight shape: (out_features, in_features) — row-major.
    pub shape: (usize, usize),
    /// Lazily uploaded GPU copy of raw_data (avoids re-upload per forward).
    #[cfg(feature = "cuda")]
    gpu_cache: OnceLock<CudaSlice<u8>>,
}

impl Clone for QWeight {
    fn clone(&self) -> Self {
        Self {
            raw_data: self.raw_data.clone(),
            dtype: self.dtype,
            shape: self.shape,
            #[cfg(feature = "cuda")]
            gpu_cache: OnceLock::new(), // fresh cache — will re-upload on first use
        }
    }
}

impl std::fmt::Debug for QWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QWeight")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("raw_bytes", &self.raw_data.len())
            .finish()
    }
}

impl QWeight {
    /// Create from raw GGUF tensor data.
    pub fn new(raw_data: Vec<u8>, dtype: GgmlDType, shape: (usize, usize)) -> Self {
        Self {
            raw_data, dtype, shape,
            #[cfg(feature = "cuda")]
            gpu_cache: OnceLock::new(),
        }
    }

    /// Dequantize entire weight matrix to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        let n_elements = self.shape.0 * self.shape.1;
        quant::dequantize(&self.raw_data, self.dtype, n_elements)
    }

    /// Dequantize to TqTensor.
    pub fn to_tensor(&self, device: &TqDevice) -> Result<TqTensor> {
        let data = self.dequantize();
        TqTensor::from_vec(data, vec![self.shape.0, self.shape.1], device)
    }

    /// Dequantize to TqTensor on target device (candle-compat: `.dequantize(device)?`).
    pub fn dequantize_to_device(&self, device: &TqDevice) -> Result<TqTensor> {
        self.to_tensor(device)
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize { self.shape.0 }

    /// Number of input features (columns).
    pub fn in_features(&self) -> usize { self.shape.1 }
}

/// Quantized matrix multiplication operator.
///
/// Wraps a QWeight and provides `forward(x) -> y` where:
/// - x: [batch..., in_features]
/// - y: [batch..., out_features]
///
/// For decode (seq_len=1), this is a single matvec.
/// For prefill (seq_len>1), this is a batched matmul.
#[derive(Debug, Clone)]
pub enum QMatMul {
    /// Quantized weight — dequant on-the-fly.
    Quantized(QWeight),
    /// Already dequantized (for norms, biases, embeddings).
    Full(TqTensor),
}

impl QMatMul {
    /// Create from a quantized weight.
    pub fn from_qweight(w: QWeight) -> Self {
        QMatMul::Quantized(w)
    }

    /// Alias for from_qweight (candle API compat).
    pub fn from_qtensor(w: QWeight) -> Self {
        Self::from_qweight(w)
    }

    /// Create from a full-precision tensor (non-quantized weights like biases).
    pub fn from_tensor(t: TqTensor) -> Self {
        QMatMul::Full(t)
    }

    /// Forward pass: x @ W^T.
    ///
    /// Input x: [batch..., in_features]
    /// Output:  [batch..., out_features]
    ///
    /// The weight is stored as [out_features, in_features], so we compute
    /// x @ W^T which gives [batch..., out_features].
    ///
    /// When CUDA is available and input is on GPU:
    /// - Q4_K_M/Q8_0: fused dequant+matvec kernel (no intermediate f32 buffer)
    /// - Other dtypes: dequant on CPU, upload, cuBLAS SGEMM (future)
    pub fn forward(&self, x: &TqTensor) -> Result<TqTensor> {
        match self {
            QMatMul::Full(w) => {
                // Standard matmul: x @ W^T
                let wt = w.t()?;
                x.matmul(&wt)
            }
            QMatMul::Quantized(qw) => {
                // Try GPU path first
                #[cfg(feature = "cuda")]
                if x.is_cuda() {
                    return Self::forward_gpu(qw, x);
                }

                // CPU path: dequantize + matmul
                Self::forward_cpu(qw, x)
            }
        }
    }

    /// CPU forward: dequantize weight, then naive matmul.
    fn forward_cpu(qw: &QWeight, x: &TqTensor) -> Result<TqTensor> {
        let x_shape = x.shape().to_vec();
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
        let x_data = x.as_slice();
        let w_data = qw.dequantize();

        let mut output = Vec::with_capacity(batch_elements * out_features);

        for b in 0..batch_elements {
            let x_row = &x_data[b * in_features..(b + 1) * in_features];
            for o in 0..out_features {
                let w_row = &w_data[o * in_features..(o + 1) * in_features];
                let dot: f32 = x_row.iter().zip(w_row.iter())
                    .map(|(&xi, &wi)| xi * wi)
                    .sum();
                output.push(dot);
            }
        }

        let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.push(out_features);
        TqTensor::from_vec(output, out_shape, x.device())
    }

    /// GPU forward: fused dequant+matvec for Q4_K_M / Q8_0.
    ///
    /// Raw quantized weight bytes are lazily uploaded to GPU on first call
    /// and cached in `QWeight::gpu_cache` for all subsequent calls.
    /// The fused kernel reads packed data directly — no intermediate
    /// f32 weight buffer needed. This is 3-4x faster than dequant+SGEMM.
    #[cfg(feature = "cuda")]
    fn forward_gpu(qw: &QWeight, x: &TqTensor) -> Result<TqTensor> {
        let x_shape = x.shape().to_vec();
        let in_features = qw.in_features();
        let out_features = qw.out_features();
        let batch_elements: usize = x_shape[..x_shape.len() - 1].iter().product();

        let stream = x.cuda_stream();

        // Lazy GPU cache: upload weight bytes once, reuse for all subsequent calls
        let w_gpu = qw.gpu_cache.get_or_init(|| {
            stream.clone_htod(&qw.raw_data)
                .expect("QWeight GPU upload failed")
        });

        // For decode (batch=1), use fused matvec kernel
        if batch_elements == 1 {
            let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_features)
                .map_err(|e| TqError::Msg(format!("output alloc: {}", e)))?;

            let reg = crate::cuda::kernels::KernelRegistry::new(
                &stream.context(), &stream,
            ).map_err(|e| TqError::Msg(format!("kernel init: {}", e)))?;

            let x_gpu = x.cuda_data();

            match qw.dtype {
                GgmlDType::Q4K => {
                    crate::cuda::kernels::q4km_matvec(
                        &reg, w_gpu, x_gpu, &mut out_gpu, out_features, in_features,
                    ).map_err(|e| TqError::Msg(format!("q4km_matvec: {}", e)))?;
                }
                GgmlDType::Q8_0 => {
                    crate::cuda::kernels::q8_0_matvec(
                        &reg, w_gpu, x_gpu, &mut out_gpu, out_features, in_features,
                    ).map_err(|e| TqError::Msg(format!("q8_0_matvec: {}", e)))?;
                }
                _ => {
                    return Self::forward_gpu_fallback(qw, x);
                }
            }

            let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
            out_shape.push(out_features);
            Ok(TqTensor::from_cuda(out_gpu, out_shape, stream.clone()))
        } else {
            // Prefill: fall back to dequant + standard matmul (future: cuBLAS)
            Self::forward_gpu_fallback(qw, x)
        }
    }

    /// GPU fallback: dequant on CPU, upload to GPU, standard matmul.
    #[cfg(feature = "cuda")]
    fn forward_gpu_fallback(qw: &QWeight, x: &TqTensor) -> Result<TqTensor> {
        // Dequant on CPU then upload
        let w_f32 = qw.dequantize();
        let stream = x.cuda_stream();
        let w_gpu = stream.clone_htod(&w_f32)
            .map_err(|e| TqError::Msg(format!("weight upload: {}", e)))?;

        let w_tensor = TqTensor::from_cuda(
            w_gpu,
            vec![qw.out_features(), qw.in_features()],
            stream.clone(),
        );

        // x @ W^T
        let wt = w_tensor.t()?;
        x.matmul(&wt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qmatmul_full() {
        // W = [[1,2],[3,4],[5,6]] (3x2), x = [1,1] (1x2)
        // x @ W^T = [1*1+1*2, 1*3+1*4, 1*5+1*6] = [3, 7, 11]
        let w = TqTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            &TqDevice::Cpu,
        ).unwrap();
        let x = TqTensor::from_vec(vec![1.0, 1.0], vec![1, 2], &TqDevice::Cpu).unwrap();

        let qmm = QMatMul::from_tensor(w);
        let y = qmm.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
        let data = y.to_vec1().unwrap();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 7.0).abs() < 1e-6);
        assert!((data[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_qmatmul_quantized_f32() {
        // QWeight with F32 dtype (trivial dequant) — same test as above
        let w_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let qw = QWeight::new(w_data, GgmlDType::F32, (3, 2));
        let x = TqTensor::from_vec(vec![1.0, 1.0], vec![1, 2], &TqDevice::Cpu).unwrap();

        let qmm = QMatMul::from_qweight(qw);
        let y = qmm.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
        let data = y.to_vec1().unwrap();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 7.0).abs() < 1e-6);
        assert!((data[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_qmatmul_batched() {
        // W = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1]] (5x4)
        // x = [2, 3, 4] batch of ones → each row is [1,1,1,1]
        // x @ W^T: each row → [1, 1, 1, 1, 4]
        let w = TqTensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
            ],
            vec![5, 4],
            &TqDevice::Cpu,
        ).unwrap();
        let x = TqTensor::from_vec(vec![1.0; 2 * 3 * 4], vec![2, 3, 4], &TqDevice::Cpu).unwrap();

        let qmm = QMatMul::from_tensor(w);
        let y = qmm.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3, 5]);
        let data = y.to_vec1().unwrap();
        // Every group of 5 should be [1, 1, 1, 1, 4]
        for i in 0..6 {
            assert!((data[i * 5] - 1.0).abs() < 1e-6);
            assert!((data[i * 5 + 4] - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_qweight_clone() {
        let w_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let qw = QWeight::new(w_data.clone(), GgmlDType::F32, (2, 2));
        let cloned = qw.clone();
        assert_eq!(cloned.dtype, GgmlDType::F32);
        assert_eq!(cloned.shape, (2, 2));
        assert_eq!(cloned.raw_data, w_data);
        // Dequantized values should match
        assert_eq!(qw.dequantize(), cloned.dequantize());
    }
}
