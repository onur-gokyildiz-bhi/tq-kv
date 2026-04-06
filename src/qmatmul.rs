//! Quantized matrix multiplication — replacement for candle's QMatMul.
//!
//! Stores quantized weights in their raw GGML block format and dequantizes
//! on-the-fly during matmul. Phase 3 adds a fused CUDA kernel (q4km_matvec)
//! that avoids the intermediate f32 buffer entirely.
//!
//! Future: LoRA adapter support via `output = W*x + alpha * B*A*x`.

use std::sync::OnceLock;
use crate::cuda::{TqTensor, TqDevice, Result, TqError};
use crate::gguf::GgmlDType;
use crate::quant;
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
    /// Lazily dequantized f32 weight (avoids re-dequant per forward).
    cpu_cache: OnceLock<Vec<f32>>,
    /// Lazily uploaded GPU copy of raw_data (avoids re-upload per forward).
    #[cfg(feature = "cuda")]
    gpu_cache: OnceLock<CudaSlice<u8>>,
    /// Lazily dequantized + uploaded f32 weights on GPU (for Q6K and other dtypes
    /// without fused GPU kernels — avoids re-dequant + re-upload per forward).
    #[cfg(feature = "cuda")]
    gpu_f32_cache: OnceLock<CudaSlice<f32>>,
    /// Lazily dequantized + converted to f16 + uploaded weights on GPU.
    /// Used for cuBLAS HGEMM prefill path (tensor core acceleration).
    #[cfg(feature = "cuda")]
    gpu_f16_cache: OnceLock<CudaSlice<half::f16>>,
}

impl Clone for QWeight {
    fn clone(&self) -> Self {
        Self {
            raw_data: self.raw_data.clone(),
            dtype: self.dtype,
            shape: self.shape,
            cpu_cache: OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_cache: OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_f32_cache: OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_f16_cache: OnceLock::new(),
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
            cpu_cache: OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_cache: OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_f32_cache: OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_f16_cache: OnceLock::new(),
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

    /// Eagerly upload raw quantized bytes to GPU.
    /// Call during model load to avoid first-forward latency.
    #[cfg(feature = "cuda")]
    pub fn upload_to_gpu(&self, device: &TqDevice) -> Result<()> {
        if let TqDevice::Cuda { .. } = device {
            let stream = device.cuda_stream()?;
            self.gpu_cache.get_or_init(|| {
                stream.clone_htod(&self.raw_data)
                    .expect("QWeight GPU upload failed")
            });
        }
        Ok(())
    }

    /// Eagerly dequantize and cache on CPU.
    /// Call during model load to avoid first-forward dequant latency.
    pub fn warmup_cpu(&self) {
        self.cpu_cache.get_or_init(|| self.dequantize());
    }

    /// Eagerly dequantize, convert to f16, and upload to GPU.
    /// Call during model load to avoid first-prefill HGEMM latency.
    #[cfg(feature = "cuda")]
    pub fn warmup_f16_gpu(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>) {
        self.gpu_f16_cache.get_or_init(|| {
            let w_f32 = self.cpu_cache.get_or_init(|| self.dequantize());
            let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
            stream.clone_htod(&w_f16).expect("weight f16 upload failed")
        });
    }

    /// Get cached dequantized weights, or dequantize and cache on first call.
    pub fn cpu_cache_or_dequant(&self) -> &Vec<f32> {
        self.cpu_cache.get_or_init(|| self.dequantize())
    }

    /// Get or initialize GPU cache of raw quantized bytes.
    /// Returns a reference to the CudaSlice<u8> for kernel launch.
    /// Returns the cached slice, or uploads on first call. Panics on OOM are
    /// avoided by returning a fallback empty slice on upload failure.
    #[cfg(feature = "cuda")]
    pub fn gpu_cache_or_upload(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> &CudaSlice<u8> {
        self.gpu_cache.get_or_init(|| {
            match stream.clone_htod(&self.raw_data) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("WARNING: QWeight GPU upload failed ({}), using empty placeholder", e);
                    stream.alloc_zeros::<u8>(1).expect("cannot alloc 1 byte on GPU")
                }
            }
        })
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize { self.shape.0 }

    /// Number of input features (columns).
    pub fn in_features(&self) -> usize { self.shape.1 }

    /// Get or lazily create dequantized F32 weight cache on GPU.
    /// Used by DecodeScratch for Q6K V weight f32_matvec.
    #[cfg(feature = "cuda")]
    pub fn gpu_f32_or_upload(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> &CudaSlice<f32> {
        self.gpu_f32_cache.get_or_init(|| {
            let w = self.dequantize();
            stream.clone_htod(&w).expect("weight f32 upload failed")
        })
    }
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

    /// CPU forward: dequantize weight, then optimized matmul.
    ///
    /// Decode (M=1): rayon parallel dot products (memory-bandwidth bound).
    /// Prefill (M>1): matrixmultiply sgemm (AVX2 cache-tiled micro-kernels).
    fn forward_cpu(qw: &QWeight, x: &TqTensor) -> Result<TqTensor> {
        use rayon::prelude::*;

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
        let w_data = qw.cpu_cache.get_or_init(|| qw.dequantize());
        let w_slice = w_data.as_slice();
        let mut output = vec![0.0f32; batch_elements * out_features];

        if batch_elements <= 4 {
            // Decode path: M=1 (or small batch).
            // Rayon parallel over output rows — each dot product independent.
            // Memory-bandwidth bound: each thread reads different w_rows.
            for b in 0..batch_elements {
                let x_row = &x_data[b * in_features..(b + 1) * in_features];
                let out_chunk = &mut output[b * out_features..(b + 1) * out_features];
                out_chunk.par_iter_mut().enumerate().for_each(|(o, out)| {
                    let w_row = &w_slice[o * in_features..(o + 1) * in_features];
                    *out = x_row.iter().zip(w_row.iter())
                        .map(|(&xi, &wi)| xi * wi)
                        .sum::<f32>();
                });
            }
        } else {
            // Prefill path: M>4.
            // matrixmultiply sgemm: AVX2 cache-tiled, single-threaded but
            // much better cache utilization than rayon for large M.
            let m = batch_elements;
            let k = in_features;
            let n = out_features;
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n,
                    1.0,
                    x_data.as_ptr(), k as isize, 1,
                    w_slice.as_ptr(), 1, k as isize,
                    0.0,
                    output.as_mut_ptr(), n as isize, 1,
                );
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
            let _ = stream.context().check_err(); // clear stale errors for graph capture
            let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_features)
                .map_err(|e| TqError::Msg(format!("output alloc: {}", e)))?;

            let reg = crate::cuda::kernels::global_registry()
                .ok_or_else(|| TqError::Msg("no GPU kernel registry".into()))?;

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
            // Prefill (batch>1): dequant + cuBLAS SGEMM
            Self::forward_gpu_fallback(qw, x)
        }
    }

    /// GPU fallback: dequant on CPU, upload to GPU, f32 matvec kernel.
    /// Used for Q6K and other dtypes without fused dequant GPU kernels.
    /// Weight is dequantized + uploaded ONCE (row-major, not transposed),
    /// then custom f32_matvec kernel (1 block/row, no cuBLAS overhead).
    #[cfg(feature = "cuda")]
    pub fn forward_gpu_fallback(qw: &QWeight, x: &TqTensor) -> Result<TqTensor> {
        let out_f = qw.out_features();
        let in_f = qw.in_features();

        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;
        let stream = &reg.stream;

        // Lazily dequant + upload to GPU (row-major, cached for all subsequent calls).
        let w_gpu = qw.gpu_f32_cache.get_or_init(|| {
            let w_f32 = qw.dequantize();
            stream.clone_htod(&w_f32).expect("weight f32 upload failed")
        });

        let x_shape = x.shape().to_vec();
        let batch: usize = x_shape[..x_shape.len() - 1].iter().product::<usize>().max(1);

        if batch == 1 {
            // Decode: custom f32_matvec kernel (zero cuBLAS overhead)
            let _ = stream.context().check_err();
            let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(out_f)
                .map_err(|e| TqError::Msg(format!("f32_matvec alloc: {}", e)))?;
            crate::cuda::kernels::f32_matvec(
                reg, w_gpu, x.cuda_data(), &mut out_gpu, out_f, in_f,
            ).map_err(|e| TqError::Msg(format!("f32_matvec: {}", e)))?;
            let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
            out_shape.push(out_f);
            Ok(TqTensor::from_cuda(out_gpu, out_shape, stream.clone()))
        } else {
            // Prefill (batch>1): cuBLAS SGEMM via TqTensor::matmul
            let w_tensor = TqTensor::from_cuda(
                w_gpu.clone(), vec![out_f, in_f], stream.clone(),
            );
            let wt = w_tensor.t()?;
            x.matmul(&wt)
        }
    }

    /// Prefill SGEMM with TF32 tensor core acceleration.
    /// Uses cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 on SM80+.
    /// Falls back to regular f32 SGEMM on older GPUs.
    #[cfg(feature = "cuda")]
    fn forward_gpu_sgemm_tf32(qw: &QWeight, x: &TqTensor, w_gpu: &CudaSlice<f32>) -> Result<TqTensor> {
        use cudarc::cublas::sys;

        let out_f = qw.out_features();
        let in_f = qw.in_features();
        let x_shape = x.shape().to_vec();
        let m = x_shape[..x_shape.len() - 1].iter().product::<usize>().max(1);

        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;
        let stream = &reg.stream;
        let blas = reg.get_cublas();

        let x_gpu = x.cuda_data();
        let mut out_gpu: CudaSlice<f32> = stream.alloc_zeros(m * out_f)
            .map_err(|e| TqError::Msg(format!("sgemm alloc: {}", e)))?;

        // cublasGemmEx: C = alpha * op(A) * op(B) + beta * C
        // Row-major X[M,K] @ W[N,K]^T = C[M,N]
        // cuBLAS col-major: C^T[N,M] = W[N,K] @ X[M,K]^T
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // Scope the device_ptr borrows so out_gpu can be moved after
        let result = {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (w_ptr, _g1) = w_gpu.device_ptr(stream.as_ref());
            let (x_ptr, _g2) = x_gpu.device_ptr(stream.as_ref());
            let (c_ptr, _g3) = out_gpu.device_ptr_mut(stream.as_ref());

            unsafe {
                sys::cublasGemmEx(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,  // W transposed
                    sys::cublasOperation_t::CUBLAS_OP_N,  // X not transposed (row-major)
                    out_f as i32,   // M (rows of result in col-major = N)
                    m as i32,       // N (cols of result = M batch)
                    in_f as i32,    // K
                    &alpha as *const f32 as *const _,
                    w_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    in_f as i32,    // lda (leading dim of W = K)
                    x_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    in_f as i32,    // ldb (leading dim of X = K)
                    &beta as *const f32 as *const _,
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    out_f as i32,   // ldc (leading dim of C = N)
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
        };

        if result != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            // Fallback: regular f32 matmul if TF32 not supported
            let w_tensor = TqTensor::from_cuda(
                w_gpu.clone(), vec![out_f, in_f], stream.clone(),
            );
            let wt = w_tensor.t()?;
            return x.matmul(&wt);
        }

        let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.push(out_f);
        Ok(TqTensor::from_cuda(out_gpu, out_shape, stream.clone()))
    }

    /// FP16 HGEMM prefill: dequant → f16 → cuBLAS GemmEx (tensor core accelerated).
    ///
    /// Weight: dequant to f32 → convert to f16 → upload once (cached in gpu_f16_cache).
    /// Activation: f32 → f16 on GPU (per-call, small relative to weight).
    /// Output: f16 → f32 conversion.
    ///
    /// Uses CUBLAS_COMPUTE_32F accumulation for accuracy with FP16 inputs.
    /// This engages FP16 tensor cores on SM75+ (Turing, Ampere, Ada, Hopper).
    #[cfg(feature = "cuda")]
    fn forward_gpu_hgemm(qw: &QWeight, x: &TqTensor) -> Result<TqTensor> {
        use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
        use cudarc::cublas::sys::cublasOperation_t;

        let out_f = qw.out_features(); // N (rows of W = cols of output)
        let in_f = qw.in_features();   // K
        let x_shape = x.shape().to_vec();
        let m = x_shape[..x_shape.len() - 1].iter().product::<usize>().max(1); // M (batch * seq_len)

        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;
        let stream = &reg.stream;

        // Lazily dequant → f16 → upload (cached for all subsequent calls).
        let w_f16_gpu = qw.gpu_f16_cache.get_or_init(|| {
            let w_f32 = qw.dequantize();
            let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
            stream.clone_htod(&w_f16).expect("weight f16 upload failed")
        });

        // Convert f32 activation → f16 on GPU
        let x_gpu = x.cuda_data();
        let x_n = x.elem_count();
        let mut x_f16: CudaSlice<half::f16> = stream.alloc_zeros(x_n)
            .map_err(|e| TqError::Msg(format!("x f16 alloc: {}", e)))?;
        crate::cuda::kernels::f32_to_f16(reg, x_gpu, &mut x_f16, x_n)
            .map_err(|e| TqError::Msg(format!("f32→f16: {}", e)))?;

        // Allocate f16 output
        let mut out_f16: CudaSlice<half::f16> = stream.alloc_zeros(m * out_f)
            .map_err(|e| TqError::Msg(format!("out f16 alloc: {}", e)))?;

        // cuBLAS HGEMM: C = X @ W^T
        // X is [M, K] row-major, W is [N, K] row-major → C = X @ W^T = [M, N]
        // cuBLAS is column-major: C^T[N, M] = W @ X^T
        let blas = reg.get_cublas();
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,  // W^T in col-major = W transposed
            transb: cublasOperation_t::CUBLAS_OP_N,  // X in col-major = X^T (row-major X)
            m: out_f as i32,    // rows of output (cols of C^T = N)
            n: m as i32,        // cols of output (rows of C^T = M)
            k: in_f as i32,     // inner dim
            alpha: half::f16::from_f32(1.0),
            lda: in_f as i32,   // leading dim of W (stored row-major, so K)
            ldb: in_f as i32,   // leading dim of X (stored row-major, so K)
            beta: half::f16::from_f32(0.0),
            ldc: out_f as i32,  // leading dim of C (col-major, so N)
        };

        unsafe {
            blas.gemm(cfg, w_f16_gpu, &x_f16, &mut out_f16)
                .map_err(|e| TqError::Msg(format!("cuBLAS HGEMM: {}", e)))?;
        }

        // Convert f16 output → f32
        let mut out_f32: CudaSlice<f32> = stream.alloc_zeros(m * out_f)
            .map_err(|e| TqError::Msg(format!("out f32 alloc: {}", e)))?;
        crate::cuda::kernels::f16_to_f32(reg, &out_f16, &mut out_f32, m * out_f)
            .map_err(|e| TqError::Msg(format!("f16→f32: {}", e)))?;

        let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.push(out_f);
        Ok(TqTensor::from_cuda(out_f32, out_shape, stream.clone()))
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
