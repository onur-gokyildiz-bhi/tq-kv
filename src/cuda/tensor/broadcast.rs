//! Broadcasting and shape-expanding ops on TqTensor.
//! Extracted from tensor/mod.rs on 2026-04-13 to shrink the 2012-line file.

use super::{TqTensor, TqStorage, gpu_alloc_zeros};
use crate::cuda::{Result, TqError, tq_bail};

impl TqTensor {
    /// Compute broadcast output shape + padded strides for two tensors.
    /// Returns (out_shape, a_strides, b_strides) with 0-stride for broadcast dims.
    #[cfg(feature = "cuda")]
    pub(super) fn broadcast_strides(a_shape: &[usize], b_shape: &[usize]) -> Result<(Vec<usize>, Vec<i32>, Vec<i32>)> {
        let rank = a_shape.len().max(b_shape.len());
        let mut out = vec![0usize; rank];
        let mut a_str = vec![0i32; rank];
        let mut b_str = vec![0i32; rank];

        // Pad shapes with 1s on the left
        let a_pad: Vec<usize> = (0..rank).map(|i| {
            if i < rank - a_shape.len() { 1 } else { a_shape[i - (rank - a_shape.len())] }
        }).collect();
        let b_pad: Vec<usize> = (0..rank).map(|i| {
            if i < rank - b_shape.len() { 1 } else { b_shape[i - (rank - b_shape.len())] }
        }).collect();

        for i in 0..rank {
            out[i] = a_pad[i].max(b_pad[i]);
            if a_pad[i] != 1 && b_pad[i] != 1 && a_pad[i] != b_pad[i] {
                tq_bail!("broadcast: shapes incompatible at dim {}", i);
            }
        }

        // Compute strides (row-major, 0 for broadcast dims)
        let mut a_real_stride = vec![1i32; rank];
        let mut b_real_stride = vec![1i32; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            a_real_stride[i] = a_real_stride[i + 1] * a_pad[i + 1] as i32;
            b_real_stride[i] = b_real_stride[i + 1] * b_pad[i + 1] as i32;
        }
        for i in 0..rank {
            a_str[i] = if a_pad[i] == 1 { 0 } else { a_real_stride[i] };
            b_str[i] = if b_pad[i] == 1 { 0 } else { b_real_stride[i] };
        }

        Ok((out, a_str, b_str))
    }

    /// GPU broadcast binary op helper (kernel-arg version, no GPU buffer uploads).
    #[cfg(feature = "cuda")]
    fn gpu_broadcast_binop(&self, other: &TqTensor, op: &str) -> Result<Self> {
        let (TqStorage::Cuda { data: a, stream }, TqStorage::Cuda { data: b, .. }) = (&self.storage, &other.storage) else {
            tq_bail!("gpu_broadcast_binop: both must be CUDA");
        };
        let reg = crate::cuda::kernels::global_registry().ok_or_else(|| TqError::Msg("no registry".into()))?;
        let (out_shape, a_str, b_str) = Self::broadcast_strides(&self.shape, &other.shape)?;
        let rank = out_shape.len();
        let n: usize = out_shape.iter().product();

        let mut out_strides = vec![1i32; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1] as i32;
        }

        let _ = stream.context().check_err();
        let mut out = gpu_alloc_zeros::<f32>(stream, n).map_err(|e| TqError::Msg(format!("{}", e)))?;
        crate::cuda::kernels::broadcast_binop_args(
            reg, a.as_ref(), b.as_ref(), &mut out, n, rank,
            &out_strides, &a_str, &b_str, op,
        ).map_err(|e| TqError::Msg(format!("broadcast {} kernel: {}", op, e)))?;

        Ok(Self::from_cuda(out, out_shape, stream.clone()))
    }

    /// Element-wise multiply with broadcasting.
    pub fn broadcast_mul(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.gpu_broadcast_binop(other, "mul");
        }
        crate::cuda::ops::TqOps::broadcast_binop(self, other, |a, b| a * b)
    }

    /// Element-wise add with broadcasting.
    pub fn broadcast_add(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.gpu_broadcast_binop(other, "add");
        }
        crate::cuda::ops::TqOps::broadcast_binop(self, other, |a, b| a + b)
    }

    /// Element-wise sub with broadcasting.
    pub fn broadcast_sub(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.gpu_broadcast_binop(other, "sub");
        }
        crate::cuda::ops::TqOps::broadcast_binop(self, other, |a, b| a - b)
    }

    /// Element-wise div with broadcasting.
    pub fn broadcast_div(&self, other: &TqTensor) -> Result<Self> {
        crate::cuda::ops::TqOps::broadcast_binop(self, other, |a, b| a / b)
    }
}
