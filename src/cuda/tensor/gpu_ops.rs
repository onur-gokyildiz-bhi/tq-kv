//! GPU-native compute methods on TqTensor (RMS norm, softmax, fused ops).
//! Extracted from tensor/mod.rs on 2026-04-13.
//!
//! These bypass ComputeBackend and keep data on GPU. Used directly from
//! turbo_generic when tensors are CUDA-resident.

use super::{TqTensor, TqStorage, TqDType, gpu_alloc_zeros};
use crate::cuda::{Result, TqError, tq_bail};

#[cfg(feature = "cuda")]
impl TqTensor {
    /// GPU RMS normalization: output = x * weight / rms(x).
    /// weight must be a CPU tensor (norm weights are small, cached on GPU via registry).
    pub fn rms_norm_gpu(&self, weight: &TqTensor, eps: f32) -> Result<Self> {
        let TqStorage::Cuda { data: x, stream } = &self.storage else {
            tq_bail!("rms_norm_gpu: tensor not on GPU");
        };
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let shape = self.shape.clone();
        let hidden = *shape.last().unwrap();
        let n_tokens = self.elem_count() / hidden;
        let n = n_tokens * hidden;

        let mut out = gpu_alloc_zeros::<f32>(stream, n)
            .map_err(|e| TqError::Msg(format!("rms_norm alloc: {}", e)))?;

        // Weight: use GPU data directly (no clone) or upload once.
        if weight.is_cuda() {
            crate::cuda::kernels::rms_norm(reg, x, weight.cuda_data(), &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("rms_norm kernel: {}", e)))?;
        } else {
            let w_gpu = stream.clone_htod(&weight.to_vec1()?)
                .map_err(|e| TqError::Msg(format!("rms_norm weight: {}", e)))?;
            crate::cuda::kernels::rms_norm(reg, x, &w_gpu, &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("rms_norm kernel: {}", e)))?;
        }

        Ok(Self::from_cuda(out, shape, stream.clone()))
    }

    /// GPU softmax along last dimension.
    pub fn softmax_gpu(&self) -> Result<Self> {
        let TqStorage::Cuda { data: x, stream } = &self.storage else {
            tq_bail!("softmax_gpu: tensor not on GPU");
        };
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let shape = self.shape.clone();
        let cols = *shape.last().unwrap();
        let rows = self.elem_count() / cols;

        let mut out = gpu_alloc_zeros::<f32>(stream, rows * cols)
            .map_err(|e| TqError::Msg(format!("softmax alloc: {}", e)))?;

        crate::cuda::kernels::softmax_last_dim(reg, x, &mut out, rows, cols)
            .map_err(|e| TqError::Msg(format!("softmax kernel: {}", e)))?;

        Ok(Self::from_cuda(out, shape, stream.clone()))
    }

    /// GPU fused SiLU × multiply: output = silu(self) * up.
    pub fn fused_silu_mul_gpu(&self, up: &TqTensor) -> Result<Self> {
        let TqStorage::Cuda { data: gate, stream } = &self.storage else {
            tq_bail!("fused_silu_mul_gpu: gate not on GPU");
        };
        let TqStorage::Cuda { data: up_data, .. } = &up.storage else {
            tq_bail!("fused_silu_mul_gpu: up not on GPU");
        };
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let n = self.elem_count();
        let mut out = gpu_alloc_zeros::<f32>(stream, n)
            .map_err(|e| TqError::Msg(format!("fused_silu_mul alloc: {}", e)))?;

        crate::cuda::kernels::fused_silu_mul(reg, gate, up_data, &mut out, n)
            .map_err(|e| TqError::Msg(format!("fused_silu_mul kernel: {}", e)))?;

        Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()))
    }

    /// GPU fused residual add + RMS norm.
    /// Returns (normalized_output, updated_residual).
    pub fn fused_add_rms_norm_gpu(&self, residual: &TqTensor, weight: &TqTensor, eps: f32) -> Result<(Self, Self)> {
        let TqStorage::Cuda { data: input, stream } = &self.storage else {
            tq_bail!("fused_add_rms_norm_gpu: input not on GPU");
        };
        let TqStorage::Cuda { data: res, .. } = &residual.storage else {
            tq_bail!("fused_add_rms_norm_gpu: residual not on GPU");
        };
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let shape = self.shape.clone();
        let hidden = *shape.last().unwrap();
        let n_tokens = self.elem_count() / hidden;
        let n = n_tokens * hidden;

        // Residual: need a mutable copy (kernel modifies in-place).
        // Arc::make_mut ensures copy-on-write: clones only if ref count > 1.
        let mut res_arc = res.clone();
        let ref_count = std::sync::Arc::strong_count(&res_arc);
        if ref_count > 1 {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!("[WARN] fused_add_rms_norm_gpu: Arc ref_count={} — copy-on-write will clone GPU buffer (this warning shown once)", ref_count);
            }
        }
        let res_mut = std::sync::Arc::make_mut(&mut res_arc);
        let mut out = gpu_alloc_zeros::<f32>(stream, n)
            .map_err(|e| TqError::Msg(format!("fused norm alloc: {}", e)))?;

        if weight.is_cuda() {
            crate::cuda::kernels::fused_add_rms_norm(reg, input.as_ref(), res_mut, weight.cuda_data(), &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("fused_add_rms_norm kernel: {}", e)))?;
        } else {
            let w_gpu = stream.clone_htod(&weight.to_vec1()?)
                .map_err(|e| TqError::Msg(format!("fused norm weight: {}", e)))?;
            crate::cuda::kernels::fused_add_rms_norm(reg, input.as_ref(), res_mut, &w_gpu, &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("fused_add_rms_norm kernel: {}", e)))?;
        }

        Ok((
            Self::from_cuda(out, shape.clone(), stream.clone()),
            Self { storage: TqStorage::Cuda { data: res_arc, stream: stream.clone() }, shape, dtype: TqDType::F32 },
        ))
    }
}
