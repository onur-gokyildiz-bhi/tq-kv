//! Compute backend abstraction — CPU, CUDA, Metal (future).
//!
//! The ComputeBackend trait decouples heavy math (matmul, norm, softmax, activations)
//! from the model graph. Tensor shape ops (narrow, cat, reshape, RoPE) stay in
//! turbo_generic.rs; only compute-bound kernels go through the backend.
//!
//! Data contract: &[f32] in, Vec<f32> out. No tensor type dependency in the trait.

use crate::qmatmul::QWeight;

/// Compute backend for heavy math operations.
///
/// All methods take raw f32 slices and return owned Vec<f32>.
/// Weight data is referenced by `&QWeight` — backends can cache GPU copies internally.
pub trait ComputeBackend: Send + Sync {
    /// Quantized matrix multiply: x @ W^T.
    ///
    /// - `x`: [m, k] row-major
    /// - `weight`: quantized [n, k] (out_features × in_features)
    /// - Returns: [m, n] row-major
    fn qmatmul(&self, x: &[f32], weight: &QWeight, m: usize, k: usize, n: usize) -> Vec<f32>;

    /// Full-precision matrix multiply: x @ W^T.
    ///
    /// - `x`: [m, k] row-major
    /// - `w`: [n, k] row-major (transposed internally)
    /// - Returns: [m, n] row-major
    fn matmul(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;

    /// RMS normalization: x * weight / rms(x).
    ///
    /// - `x`: [n_tokens, hidden] row-major
    /// - `weight`: [hidden] — learned scale
    /// - Returns: [n_tokens, hidden]
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, n_tokens: usize, hidden: usize) -> Vec<f32>;

    /// Softmax along last dimension.
    ///
    /// - `x`: [rows, cols] row-major
    /// - Returns: [rows, cols]
    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32>;

    /// SiLU activation: x * sigmoid(x).
    ///
    /// - `x`: flat f32 slice
    /// - Returns: same length
    fn silu(&self, x: &[f32]) -> Vec<f32>;

    /// Fused SiLU gate × up projection: silu(gate) * up.
    ///
    /// - `gate`, `up`: same length
    /// - Returns: same length
    fn fused_silu_mul(&self, gate: &[f32], up: &[f32]) -> Vec<f32>;

    /// Fused residual add + RMS normalization.
    ///
    /// Computes: residual += input; output = rms_norm(residual, weight, eps)
    /// Returns: (normalized_output, updated_residual)
    /// Both are [n_tokens, hidden] row-major.
    fn fused_add_rms_norm(
        &self, input: &[f32], residual: &[f32], weight: &[f32], eps: f32,
        n_tokens: usize, hidden: usize,
    ) -> (Vec<f32>, Vec<f32>);

    /// Halved RoPE: apply rotary position embedding in-place (Qwen2, Mistral style).
    ///
    /// - `x`: [n_tokens * n_heads * head_dim] flat, modified in-place
    /// - `cos`, `sin`: [max_seq_len, rope_dim/2] precomputed frequency tables
    /// - Returns: modified x
    fn rope_halved(
        &self, x: &mut Vec<f32>, cos: &[f32], sin: &[f32],
        n_tokens: usize, n_heads: usize, head_dim: usize, rope_dim: usize, pos_offset: usize,
    );

    /// Interleaved RoPE: apply rotary position embedding in-place (Llama style).
    fn rope_interleaved(
        &self, x: &mut Vec<f32>, cos: &[f32], sin: &[f32],
        n_tokens: usize, n_heads: usize, head_dim: usize, rope_dim: usize, pos_offset: usize,
    );

    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;

    /// Pre-upload a quantized weight to GPU (if applicable). No-op on CPU.
    fn warmup_qweight(&self, _weight: &QWeight) {}

    /// Pre-upload a persistent f32 weight (norm, bias) to GPU cache. No-op on CPU.
    fn warmup_f32(&self, _data: &[f32]) {}
}

// ============================================================
// CPU Backend — wraps existing rayon + matrixmultiply logic
// ============================================================

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl ComputeBackend for CpuBackend {
    fn qmatmul(&self, x: &[f32], weight: &QWeight, m: usize, k: usize, n: usize) -> Vec<f32> {
        use rayon::prelude::*;

        let w_data = weight.cpu_cache_or_dequant();
        let w = w_data.as_slice();
        let mut output = vec![0.0f32; m * n];

        if m <= 4 {
            // Decode path: rayon parallel dot products
            for b in 0..m {
                let x_row = &x[b * k..(b + 1) * k];
                let out_chunk = &mut output[b * n..(b + 1) * n];
                out_chunk.par_iter_mut().enumerate().for_each(|(o, out)| {
                    let w_row = &w[o * k..(o + 1) * k];
                    *out = x_row.iter().zip(w_row.iter())
                        .map(|(&xi, &wi)| xi * wi)
                        .sum::<f32>();
                });
            }
        } else {
            // Prefill path: matrixmultiply sgemm (AVX2 cache-tiled)
            unsafe {
                matrixmultiply::sgemm(
                    m, k, n,
                    1.0,
                    x.as_ptr(), k as isize, 1,
                    w.as_ptr(), 1, k as isize,
                    0.0,
                    output.as_mut_ptr(), n as isize, 1,
                );
            }
        }

        output
    }

    fn matmul(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];
        // w is [n, k] row-major, we need x @ w^T
        // sgemm: C = alpha * A * B + beta * C
        // A = x [m, k], B = w [n, k] but we want w^T [k, n]
        // With strides: B ptr = w, row_stride = 1, col_stride = k => reads as [k, n]
        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                1.0,
                x.as_ptr(), k as isize, 1,
                w.as_ptr(), 1, k as isize,
                0.0,
                output.as_mut_ptr(), n as isize, 1,
            );
        }
        output
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, n_tokens: usize, hidden: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; n_tokens * hidden];
        for t in 0..n_tokens {
            let row = &x[t * hidden..(t + 1) * hidden];
            let out_row = &mut output[t * hidden..(t + 1) * hidden];

            // variance = mean(x^2)
            let variance: f32 = row.iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
            let rms = (variance + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for i in 0..hidden {
                out_row[i] = row[i] * inv_rms * weight[i];
            }
        }
        output
    }

    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let out_row = &mut output[r * cols..(r + 1) * cols];

            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for i in 0..cols {
                out_row[i] = (row[i] - max_val).exp();
                sum += out_row[i];
            }
            let inv_sum = 1.0 / sum;
            for i in 0..cols {
                out_row[i] *= inv_sum;
            }
        }
        output
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| {
            let clamped = v.clamp(-20.0, 20.0); // avoid exp overflow
            clamped / (1.0 + (-clamped).exp())
        }).collect()
    }

    fn fused_silu_mul(&self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter().zip(up.iter())
            .map(|(&g, &u)| {
                let gc = g.clamp(-20.0, 20.0);
                (gc / (1.0 + (-gc).exp())) * u
            })
            .collect()
    }

    fn fused_add_rms_norm(
        &self, input: &[f32], residual: &[f32], weight: &[f32], eps: f32,
        n_tokens: usize, hidden: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut new_residual = vec![0.0f32; n_tokens * hidden];
        let mut output = vec![0.0f32; n_tokens * hidden];
        for t in 0..n_tokens {
            let row_start = t * hidden;
            let row_end = row_start + hidden;
            // residual += input
            for i in row_start..row_end {
                new_residual[i] = residual[i] + input[i];
            }
            // rms_norm(residual)
            let row = &new_residual[row_start..row_end];
            let variance: f32 = row.iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
            let inv_rms = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                output[row_start + i] = row[i] * inv_rms * weight[i];
            }
        }
        (output, new_residual)
    }

    fn rope_halved(
        &self, x: &mut Vec<f32>, cos: &[f32], sin: &[f32],
        n_tokens: usize, n_heads: usize, head_dim: usize, rope_dim: usize, pos_offset: usize,
    ) {
        let half = rope_dim / 2;
        let cos_stride = half;
        for t in 0..n_tokens {
            let pos = pos_offset + t;
            let cos_row = &cos[pos * cos_stride..pos * cos_stride + half];
            let sin_row = &sin[pos * cos_stride..pos * cos_stride + half];
            for h in 0..n_heads {
                let base = (t * n_heads + h) * head_dim;
                for i in 0..half {
                    let x0 = x[base + i];
                    let x1 = x[base + half + i];
                    x[base + i] = x0 * cos_row[i] - x1 * sin_row[i];
                    x[base + half + i] = x0 * sin_row[i] + x1 * cos_row[i];
                }
            }
        }
    }

    fn rope_interleaved(
        &self, x: &mut Vec<f32>, cos: &[f32], sin: &[f32],
        n_tokens: usize, n_heads: usize, head_dim: usize, rope_dim: usize, pos_offset: usize,
    ) {
        let half = rope_dim / 2;
        let cos_stride = half;
        for t in 0..n_tokens {
            let pos = pos_offset + t;
            let cos_row = &cos[pos * cos_stride..pos * cos_stride + half];
            let sin_row = &sin[pos * cos_stride..pos * cos_stride + half];
            for h in 0..n_heads {
                let base = (t * n_heads + h) * head_dim;
                for i in 0..half {
                    let x0 = x[base + i * 2];
                    let x1 = x[base + i * 2 + 1];
                    x[base + i * 2] = x0 * cos_row[i] - x1 * sin_row[i];
                    x[base + i * 2 + 1] = x0 * sin_row[i] + x1 * cos_row[i];
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

// ============================================================
// CUDA Backend — GPU-accelerated compute via cudarc
// ============================================================

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    registry: crate::cuda::kernels::KernelRegistry,
    /// GPU cache for small persistent weights (norm weights, biases).
    weight_cache: std::sync::Mutex<std::collections::HashMap<usize, cudarc::driver::CudaSlice<f32>>>,
    cpu: CpuBackend,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create a new CUDA backend. Initializes the GPU context and loads all PTX kernels.
    pub fn new() -> Result<Self, String> {
        let ctx = cudarc::driver::CudaContext::new(0)
            .map_err(|e| format!("CUDA init failed: {}", e))?;
        let stream = ctx.default_stream();
        let registry = crate::cuda::kernels::KernelRegistry::new(&ctx, &stream)
            .map_err(|e| format!("kernel load failed: {}", e))?;
        Ok(Self {
            stream, registry, cpu: CpuBackend::new(),
            weight_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Get a reference to the CUDA stream (for pre-uploading weights).
    pub fn stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        &self.stream
    }

    /// Get or upload a persistent weight slice to GPU. Cached by pointer address.
    fn cached_weight(&self, data: &[f32]) -> Option<cudarc::driver::CudaSlice<f32>> {
        let key = data.as_ptr() as usize;
        let mut cache = self.weight_cache.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            // Clone the CudaSlice handle (cheap — reference counted)
            Some(cached.clone())
        } else {
            match self.stream.clone_htod(data) {
                Ok(gpu_data) => {
                    let result = gpu_data.clone();
                    cache.insert(key, gpu_data);
                    Some(result)
                }
                Err(_) => None,
            }
        }
    }

    /// Pre-upload a weight slice to GPU cache. Call during model load.
    pub fn warmup_weight(&self, data: &[f32]) {
        let _ = self.cached_weight(data);
    }

}

#[cfg(feature = "cuda")]
impl ComputeBackend for CudaBackend {
    fn qmatmul(&self, x: &[f32], weight: &QWeight, m: usize, k: usize, n: usize) -> Vec<f32> {
        use crate::gguf::GgmlDType;

        // GPU fused kernel: only for decode (m=1) with supported quant types.
        // Prefill (m>1) falls back to CPU (future: cuBLAS SGEMM).
        if m == 1 && matches!(weight.dtype, GgmlDType::Q4K | GgmlDType::Q8_0) {
            let x_gpu = match self.stream.clone_htod(x) {
                Ok(v) => v,
                Err(_) => return self.cpu.qmatmul(x, weight, m, k, n),
            };

            // Get or upload weight bytes to GPU (cached in QWeight)
            let w_gpu = weight.gpu_cache_or_upload(&self.stream);

            // Allocate output
            let mut out_gpu: cudarc::driver::CudaSlice<f32> = match self.stream.alloc_zeros(n) {
                Ok(v) => v,
                Err(_) => return self.cpu.qmatmul(x, weight, m, k, n),
            };

            // Launch kernel
            let result = match weight.dtype {
                GgmlDType::Q4K => crate::cuda::kernels::q4km_matvec(
                    &self.registry, w_gpu, &x_gpu, &mut out_gpu, n, k,
                ),
                GgmlDType::Q8_0 => crate::cuda::kernels::q8_0_matvec(
                    &self.registry, w_gpu, &x_gpu, &mut out_gpu, n, k,
                ),
                _ => unreachable!(),
            };

            if result.is_err() {
                return self.cpu.qmatmul(x, weight, m, k, n);
            }

            match self.stream.clone_dtoh(&out_gpu) {
                Ok(v) => v,
                Err(_) => self.cpu.qmatmul(x, weight, m, k, n),
            }
        } else {
            self.cpu.qmatmul(x, weight, m, k, n)
        }
    }

    fn matmul(&self, x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        if m * k < 1024 {
            return self.cpu.matmul(x, w, m, k, n);
        }
        use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
        use cudarc::cublas::sys::cublasOperation_t;

        let x_gpu = match self.stream.clone_htod(x) { Ok(v) => v, Err(_) => return self.cpu.matmul(x, w, m, k, n) };
        let w_gpu = match self.cached_weight(w) { Some(v) => v, None => return self.cpu.matmul(x, w, m, k, n) };
        let mut out_gpu = match self.stream.alloc_zeros::<f32>(m * n) { Ok(v) => v, Err(_) => return self.cpu.matmul(x, w, m, k, n) };
        let blas = match CudaBlas::new(self.stream.clone()) { Ok(v) => v, Err(_) => return self.cpu.matmul(x, w, m, k, n) };

        // x @ W^T: x[m,k], W[n,k] → C[m,n]. cuBLAS col-major: C^T[n,m] = W @ x^T
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32, n: m as i32, k: k as i32,
            alpha: 1.0f32, lda: k as i32, ldb: k as i32, beta: 0.0f32, ldc: n as i32,
        };
        if unsafe { blas.gemm(cfg, &w_gpu, &x_gpu, &mut out_gpu) }.is_err() {
            return self.cpu.matmul(x, w, m, k, n);
        }
        self.stream.clone_dtoh(&out_gpu).unwrap_or_else(|_| self.cpu.matmul(x, w, m, k, n))
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, n_tokens: usize, hidden: usize) -> Vec<f32> {
        let n = n_tokens * hidden;
        if n < 512 { return self.cpu.rms_norm(x, weight, eps, n_tokens, hidden); }
        let x_gpu = match self.stream.clone_htod(x) { Ok(v) => v, Err(_) => return self.cpu.rms_norm(x, weight, eps, n_tokens, hidden) };
        let w_gpu = match self.cached_weight(weight) { Some(v) => v, None => return self.cpu.rms_norm(x, weight, eps, n_tokens, hidden) };
        let mut out_gpu = match self.stream.alloc_zeros::<f32>(n) { Ok(v) => v, Err(_) => return self.cpu.rms_norm(x, weight, eps, n_tokens, hidden) };
        if crate::cuda::kernels::rms_norm(&self.registry, &x_gpu, &w_gpu, &mut out_gpu, n_tokens, hidden, eps).is_err() {
            return self.cpu.rms_norm(x, weight, eps, n_tokens, hidden);
        }
        self.stream.clone_dtoh(&out_gpu).unwrap_or_else(|_| self.cpu.rms_norm(x, weight, eps, n_tokens, hidden))
    }

    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let n = rows * cols;
        if n < 512 { return self.cpu.softmax(x, rows, cols); }
        let x_gpu = match self.stream.clone_htod(x) { Ok(v) => v, Err(_) => return self.cpu.softmax(x, rows, cols) };
        let mut out_gpu = match self.stream.alloc_zeros::<f32>(n) { Ok(v) => v, Err(_) => return self.cpu.softmax(x, rows, cols) };
        if crate::cuda::kernels::softmax_last_dim(&self.registry, &x_gpu, &mut out_gpu, rows, cols).is_err() {
            return self.cpu.softmax(x, rows, cols);
        }
        self.stream.clone_dtoh(&out_gpu).unwrap_or_else(|_| self.cpu.softmax(x, rows, cols))
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        if n < 512 { return self.cpu.silu(x); }
        let x_gpu = match self.stream.clone_htod(x) { Ok(v) => v, Err(_) => return self.cpu.silu(x) };
        let mut out_gpu = match self.stream.alloc_zeros::<f32>(n) { Ok(v) => v, Err(_) => return self.cpu.silu(x) };
        if crate::cuda::kernels::silu(&self.registry, &x_gpu, &mut out_gpu, n).is_err() { return self.cpu.silu(x); }
        self.stream.clone_dtoh(&out_gpu).unwrap_or_else(|_| self.cpu.silu(x))
    }

    fn fused_silu_mul(&self, gate: &[f32], up: &[f32]) -> Vec<f32> {
        let n = gate.len();
        if n < 512 { return self.cpu.fused_silu_mul(gate, up); }
        let gate_gpu = match self.stream.clone_htod(gate) { Ok(v) => v, Err(_) => return self.cpu.fused_silu_mul(gate, up) };
        let up_gpu = match self.stream.clone_htod(up) { Ok(v) => v, Err(_) => return self.cpu.fused_silu_mul(gate, up) };
        let mut out_gpu = match self.stream.alloc_zeros::<f32>(n) { Ok(v) => v, Err(_) => return self.cpu.fused_silu_mul(gate, up) };
        if crate::cuda::kernels::fused_silu_mul(&self.registry, &gate_gpu, &up_gpu, &mut out_gpu, n).is_err() {
            return self.cpu.fused_silu_mul(gate, up);
        }
        self.stream.clone_dtoh(&out_gpu).unwrap_or_else(|_| self.cpu.fused_silu_mul(gate, up))
    }

    fn fused_add_rms_norm(
        &self, input: &[f32], residual: &[f32], weight: &[f32], eps: f32,
        n_tokens: usize, hidden: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = n_tokens * hidden;
        if n < 512 { return self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden); }
        let input_gpu = match self.stream.clone_htod(input) { Ok(v) => v, Err(_) => return self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden) };
        let mut res_gpu = match self.stream.clone_htod(residual) { Ok(v) => v, Err(_) => return self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden) };
        let w_gpu = match self.cached_weight(weight) { Some(v) => v, None => return self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden) };
        let mut out_gpu = match self.stream.alloc_zeros::<f32>(n) { Ok(v) => v, Err(_) => return self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden) };
        if crate::cuda::kernels::fused_add_rms_norm(&self.registry, &input_gpu, &mut res_gpu, &w_gpu, &mut out_gpu, n_tokens, hidden, eps).is_err() {
            return self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden);
        }
        let output = self.stream.clone_dtoh(&out_gpu).unwrap_or_else(|_| { self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden).0 });
        let new_residual = self.stream.clone_dtoh(&res_gpu).unwrap_or_else(|_| { self.cpu.fused_add_rms_norm(input, residual, weight, eps, n_tokens, hidden).1 });
        (output, new_residual)
    }

    fn rope_halved(
        &self, x: &mut Vec<f32>, cos: &[f32], sin: &[f32],
        n_tokens: usize, n_heads: usize, head_dim: usize, rope_dim: usize, pos_offset: usize,
    ) {
        let n = n_tokens * n_heads * head_dim;
        if n < 512 {
            return self.cpu.rope_halved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset);
        }
        let mut x_gpu = match self.stream.clone_htod(x.as_slice()) {
            Ok(v) => v,
            Err(_) => return self.cpu.rope_halved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset),
        };
        let cos_gpu = match self.cached_weight(cos) {
            Some(v) => v,
            None => return self.cpu.rope_halved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset),
        };
        let sin_gpu = match self.cached_weight(sin) {
            Some(v) => v,
            None => return self.cpu.rope_halved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset),
        };
        if crate::cuda::kernels::rope_halved(
            &self.registry, &mut x_gpu, &cos_gpu, &sin_gpu,
            n_tokens, n_heads, head_dim, rope_dim, pos_offset,
        ).is_err() {
            return self.cpu.rope_halved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset);
        }
        if let Ok(result) = self.stream.clone_dtoh(&x_gpu) {
            *x = result;
        } else {
            self.cpu.rope_halved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset);
        }
    }

    fn rope_interleaved(
        &self, x: &mut Vec<f32>, cos: &[f32], sin: &[f32],
        n_tokens: usize, n_heads: usize, head_dim: usize, rope_dim: usize, pos_offset: usize,
    ) {
        let n = n_tokens * n_heads * head_dim;
        if n < 512 {
            return self.cpu.rope_interleaved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset);
        }
        let mut x_gpu = match self.stream.clone_htod(x.as_slice()) {
            Ok(v) => v,
            Err(_) => return self.cpu.rope_interleaved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset),
        };
        let cos_gpu = match self.cached_weight(cos) {
            Some(v) => v,
            None => return self.cpu.rope_interleaved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset),
        };
        let sin_gpu = match self.cached_weight(sin) {
            Some(v) => v,
            None => return self.cpu.rope_interleaved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset),
        };
        if crate::cuda::kernels::rope_interleaved(
            &self.registry, &mut x_gpu, &cos_gpu, &sin_gpu,
            n_tokens, n_heads, head_dim, rope_dim, pos_offset,
        ).is_err() {
            return self.cpu.rope_interleaved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset);
        }
        if let Ok(result) = self.stream.clone_dtoh(&x_gpu) {
            *x = result;
        } else {
            self.cpu.rope_interleaved(x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset);
        }
    }

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn warmup_qweight(&self, weight: &QWeight) {
        // Upload raw quantized bytes to GPU cache
        weight.gpu_cache_or_upload(&self.stream);
    }

    fn warmup_f32(&self, data: &[f32]) {
        // Upload to GPU weight cache
        self.warmup_weight(data);
    }
}

/// Create the best available backend for the current system.
pub fn create_backend() -> std::sync::Arc<dyn ComputeBackend> {
    #[cfg(feature = "cuda")]
    {
        match CudaBackend::new() {
            Ok(backend) => {
                eprintln!("  Compute backend: cuda (GPU-accelerated qmatmul)");
                return std::sync::Arc::new(backend);
            }
            Err(e) => {
                eprintln!("  CUDA init failed ({}), falling back to CPU", e);
            }
        }
    }
    eprintln!("  Compute backend: cpu");
    std::sync::Arc::new(CpuBackend::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "index {}: {} vs {} (diff {})", i, x, y, (x - y).abs());
        }
    }

    #[test]
    fn test_cpu_rms_norm() {
        let backend = CpuBackend::new();
        // x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 1e-5
        // variance = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5 + 1e-5) ≈ 2.7386
        // normalized = [0.3651, 0.7303, 1.0954, 1.4606]
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let out = backend.rms_norm(&x, &w, 1e-5, 1, 4);
        let rms = (7.5f32 + 1e-5).sqrt();
        let expected: Vec<f32> = (1..=4).map(|i| i as f32 / rms).collect();
        approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    fn test_cpu_softmax() {
        let backend = CpuBackend::new();
        let x = vec![1.0, 2.0, 3.0];
        let out = backend.softmax(&x, 1, 3);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax doesn't sum to 1: {}", sum);
        assert!(out[2] > out[1] && out[1] > out[0], "softmax order wrong");
    }

    #[test]
    fn test_cpu_silu() {
        let backend = CpuBackend::new();
        let x = vec![0.0, 1.0, -1.0];
        let out = backend.silu(&x);
        // silu(0) = 0, silu(1) = 1/(1+e^-1) ≈ 0.7311, silu(-1) = -1/(1+e^1) ≈ -0.2689
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1] - 0.7311).abs() < 0.001);
        assert!((out[2] + 0.2689).abs() < 0.001);
    }

    #[test]
    fn test_cpu_fused_silu_mul() {
        let backend = CpuBackend::new();
        let gate = vec![1.0, 2.0];
        let up = vec![3.0, 4.0];
        let out = backend.fused_silu_mul(&gate, &up);
        let silu_gate = backend.silu(&gate);
        let expected: Vec<f32> = silu_gate.iter().zip(up.iter()).map(|(&s, &u)| s * u).collect();
        approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    fn test_cpu_matmul() {
        let backend = CpuBackend::new();
        // x = [1,1] (1×2), w = [[1,2],[3,4],[5,6]] (3×2)
        // x @ w^T = [3, 7, 11]
        let x = vec![1.0, 1.0];
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = backend.matmul(&x, &w, 1, 2, 3);
        approx_eq(&out, &[3.0, 7.0, 11.0], 1e-5);
    }
}
