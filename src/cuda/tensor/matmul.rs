//! Matmul ops on TqTensor: dense cuBLAS, cached-weight matvec, quantized matvec.
//! Extracted from tensor/mod.rs on 2026-04-13.

use super::{TqTensor, TqStorage, gpu_alloc_zeros};
use crate::cuda::{Result, TqError, tq_bail};

impl TqTensor {
    /// Dense matmul. Supports batched matmul for rank >= 2.
    /// GPU: uses cuBLAS SGEMM when both tensors are on CUDA.
    pub fn matmul(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.matmul_cublas(other);
        }
        crate::cuda::ops::TqOps::matmul(self, other)
    }
}

#[cfg(feature = "cuda")]
impl TqTensor {
    /// GPU matmul via cuBLAS SGEMM: C = A @ B.
    ///
    /// For 2D: standard SGEMM.
    /// For higher rank: strided batched SGEMM (batch dims broadcast).
    fn matmul_cublas(&self, other: &TqTensor) -> Result<Self> {
        use cudarc::cublas::{Gemm, GemmConfig};
        use cudarc::cublas::sys::cublasOperation_t;

        let a_shape = self.shape().to_vec();
        let b_shape = other.shape().to_vec();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        if a_rank < 2 || b_rank < 2 {
            tq_bail!("matmul_cublas: need rank >= 2, got {} and {}", a_rank, b_rank);
        }

        let m = a_shape[a_rank - 2]; // rows of A
        let k = a_shape[a_rank - 1]; // cols of A = rows of B
        let n = b_shape[b_rank - 1]; // cols of B

        if b_shape[b_rank - 2] != k {
            tq_bail!("matmul_cublas: inner dims mismatch {} vs {}", k, b_shape[b_rank - 2]);
        }

        let stream = self.cuda_stream().clone();
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU kernel registry for matmul".into()))?;
        let blas = reg.get_cublas();

        let batch: usize = a_shape[..a_rank - 2].iter().product();

        let mut out_gpu = gpu_alloc_zeros::<f32>(&stream, batch * m * n)
            .map_err(|e| TqError::Msg(format!("matmul alloc: {}", e)))?;

        // cuBLAS is column-major. To compute row-major C = A @ B,
        // we compute C^T = B^T @ A^T in column-major, which gives us
        // C in row-major layout. So we swap A and B and transpose flags.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        if batch == 1 {
            unsafe {
                blas.gemm(cfg, other.cuda_data(), self.cuda_data(), &mut out_gpu)
                    .map_err(|e| TqError::Msg(format!("cuBLAS sgemm: {}", e)))?;
            }
        } else {
            use cudarc::cublas::StridedBatchedConfig;
            let strided_cfg = StridedBatchedConfig {
                gemm: cfg,
                batch_size: batch as i32,
                stride_a: (k * n) as i64,
                stride_b: (m * k) as i64,
                stride_c: (m * n) as i64,
            };
            unsafe {
                blas.gemm_strided_batched(strided_cfg, other.cuda_data(), self.cuda_data(), &mut out_gpu)
                    .map_err(|e| TqError::Msg(format!("cuBLAS batched sgemm: {}", e)))?;
            }
        }

        let mut out_shape = a_shape[..a_rank - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Ok(Self::from_cuda(out_gpu, out_shape, stream))
    }

    /// GPU matvec with pre-transposed cached weight: output = x @ W^T.
    /// W^T is a borrowed &CudaSlice<f32> [in_features, out_features] — no clone needed.
    /// Used for Q6K and other dtypes where weight is dequantized + pre-transposed once.
    pub fn matvec_with_cached_wt(
        &self,
        wt: &cudarc::driver::CudaSlice<f32>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        use cudarc::cublas::{Gemm, GemmConfig};
        use cudarc::cublas::sys::cublasOperation_t;

        let stream = self.cuda_stream().clone();
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry for matvec".into()))?;
        let blas = reg.get_cublas();

        let x_shape = self.shape().to_vec();
        let batch: usize = x_shape[..x_shape.len() - 1].iter().product::<usize>().max(1);
        let m = batch;
        let k = in_features;
        let n = out_features;

        let mut out_gpu = gpu_alloc_zeros::<f32>(&stream, m * n)
            .map_err(|e| TqError::Msg(format!("matvec alloc: {}", e)))?;

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            blas.gemm(cfg, wt, self.cuda_data(), &mut out_gpu)
                .map_err(|e| TqError::Msg(format!("cuBLAS matvec: {}", e)))?;
        }

        let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(Self::from_cuda(out_gpu, out_shape, stream))
    }

    /// GPU quantized matvec: output = x @ W^T where W is Q4_K_M, Q6_K, or Q8_0.
    /// Weight bytes must already be on GPU (via QWeight::gpu_cache_or_upload).
    pub fn qmatmul_gpu(
        &self,
        w_gpu: &cudarc::driver::CudaSlice<u8>,
        dtype: crate::gguf::GgmlDType,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        let TqStorage::Cuda { data: x, stream } = &self.storage else {
            tq_bail!("qmatmul_gpu: input not on GPU");
        };
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let x_shape = self.shape.clone();
        let batch: usize = x_shape[..x_shape.len() - 1].iter().product();

        // Only decode (batch=1) uses fused kernel; prefill uses cuBLAS via dequant.
        if batch == 1 {
            let mut out = gpu_alloc_zeros::<f32>(stream, out_features)
                .map_err(|e| TqError::Msg(format!("qmatmul alloc: {}", e)))?;

            match dtype {
                crate::gguf::GgmlDType::Q4K => {
                    crate::cuda::kernels::q4km_matvec(reg, w_gpu, x, &mut out, out_features, in_features)
                        .map_err(|e| TqError::Msg(format!("q4km_matvec: {}", e)))?;
                }
                crate::gguf::GgmlDType::Q6K => {
                    crate::cuda::kernels::q6k_matvec(reg, w_gpu, x, &mut out, out_features, in_features)
                        .map_err(|e| TqError::Msg(format!("q6k_matvec: {}", e)))?;
                }
                crate::gguf::GgmlDType::Q8_0 => {
                    crate::cuda::kernels::q8_0_matvec(reg, w_gpu, x, &mut out, out_features, in_features)
                        .map_err(|e| TqError::Msg(format!("q8_0_matvec: {}", e)))?;
                }
                _ => tq_bail!("qmatmul_gpu: unsupported dtype {:?}", dtype),
            }

            let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
            out_shape.push(out_features);
            Ok(Self::from_cuda(out, out_shape, stream.clone()))
        } else {
            tq_bail!("qmatmul_gpu: batch>1 not yet supported (use cuBLAS path)");
        }
    }

    /// Like qmatmul_gpu but writes into a pre-allocated output buffer (zero alloc).
    /// Used by DecodeScratch to avoid GPU allocation during decode.
    pub fn qmatmul_gpu_into(
        &self,
        w_gpu: &cudarc::driver::CudaSlice<u8>,
        dtype: crate::gguf::GgmlDType,
        out_features: usize,
        in_features: usize,
        output: &mut cudarc::driver::CudaSlice<f32>,
    ) -> Result<()> {
        let TqStorage::Cuda { data: x, .. } = &self.storage else {
            tq_bail!("qmatmul_gpu_into: input not on GPU");
        };
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;
        match dtype {
            crate::gguf::GgmlDType::Q4K => {
                crate::cuda::kernels::q4km_matvec(reg, w_gpu, x, output, out_features, in_features)
                    .map_err(|e| TqError::Msg(format!("q4km_matvec: {}", e)))?;
            }
            crate::gguf::GgmlDType::Q8_0 => {
                crate::cuda::kernels::q8_0_matvec(reg, w_gpu, x, output, out_features, in_features)
                    .map_err(|e| TqError::Msg(format!("q8_0_matvec: {}", e)))?;
            }
            _ => tq_bail!("qmatmul_gpu_into: unsupported dtype {:?}", dtype),
        }
        Ok(())
    }
}
