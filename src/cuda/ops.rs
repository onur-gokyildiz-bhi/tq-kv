//! Tensor operations — CPU implementations.
//!
//! CUDA dispatch will be added in Phase 3.
//! Each op checks storage type and dispatches accordingly.

use super::{TqTensor, Result, tq_bail};

/// Tensor operation implementations.
pub struct TqOps;

impl TqOps {
    /// Matrix multiply: A[..., M, K] @ B[..., K, N] -> C[..., M, N].
    /// Supports batched matmul (leading dimensions must match or be 1).
    pub fn matmul(a: &TqTensor, b: &TqTensor) -> Result<TqTensor> {
        if a.rank() < 2 || b.rank() < 2 {
            tq_bail!("matmul: need rank >= 2, got {} and {}", a.rank(), b.rank());
        }

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[a.rank() - 2];
        let k = a_shape[a.rank() - 1];
        let k2 = b_shape[b.rank() - 2];
        let n = b_shape[b.rank() - 1];

        if k != k2 {
            tq_bail!("matmul: inner dims mismatch {} vs {}", k, k2);
        }

        // Compute batch dimensions
        let a_batch: Vec<usize> = a_shape[..a.rank() - 2].to_vec();
        let b_batch: Vec<usize> = b_shape[..b.rank() - 2].to_vec();

        // Broadcast batch dims
        let batch_dims = broadcast_shapes(&a_batch, &b_batch)?;
        let batch_size: usize = batch_dims.iter().product();

        let a_data = a.as_slice();
        let b_data = b.as_slice();
        let a_batch_stride = m * k;
        let b_batch_stride = k * n;
        let c_batch_stride = m * n;

        let a_batch_size: usize = a_batch.iter().product();
        let b_batch_size: usize = b_batch.iter().product();

        let mut result = vec![0.0f32; batch_size * m * n];

        for batch in 0..batch_size {
            let a_batch_idx = if a_batch_size == 1 { 0 } else { batch };
            let b_batch_idx = if b_batch_size == 1 { 0 } else { batch };

            let a_off = a_batch_idx * a_batch_stride;
            let b_off = b_batch_idx * b_batch_stride;
            let c_off = batch * c_batch_stride;

            // Naive matmul — replaced with cuBLAS in Phase 3
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
                    }
                    result[c_off + i * n + j] = sum;
                }
            }
        }

        let mut out_shape = batch_dims;
        out_shape.push(m);
        out_shape.push(n);
        TqTensor::from_vec(result, out_shape, a.device())
    }

    /// Reduce along a dimension, keeping the dimension as size 1.
    pub fn reduce_keepdim(
        t: &TqTensor,
        dim: usize,
        f: impl Fn(&[f32]) -> f32,
    ) -> Result<TqTensor> {
        if dim >= t.rank() {
            tq_bail!("reduce: dim {} >= rank {}", dim, t.rank());
        }

        let data = t.as_slice();
        let shape = t.shape();
        let dim_size = shape[dim];
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();

        let mut result = Vec::with_capacity(outer * inner);

        for o in 0..outer {
            for i in 0..inner {
                let mut slice = Vec::with_capacity(dim_size);
                for d in 0..dim_size {
                    slice.push(data[o * dim_size * inner + d * inner + i]);
                }
                result.push(f(&slice));
            }
        }

        let mut new_shape = shape.to_vec();
        new_shape[dim] = 1;
        TqTensor::from_vec(result, new_shape, t.device())
    }

    /// Element-wise binary op with broadcasting.
    pub fn broadcast_binop(
        a: &TqTensor,
        b: &TqTensor,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<TqTensor> {
        let a_data = a.as_slice();
        let b_data = b.as_slice();
        let a_shape = a.shape();
        let b_shape = b.shape();

        // Pad shapes to same rank
        let max_rank = a_shape.len().max(b_shape.len());
        let mut a_padded = vec![1usize; max_rank];
        let mut b_padded = vec![1usize; max_rank];
        let a_offset = max_rank - a_shape.len();
        let b_offset = max_rank - b_shape.len();
        a_padded[a_offset..].copy_from_slice(a_shape);
        b_padded[b_offset..].copy_from_slice(b_shape);

        // Compute output shape
        let mut out_shape = vec![0usize; max_rank];
        for i in 0..max_rank {
            if a_padded[i] == b_padded[i] {
                out_shape[i] = a_padded[i];
            } else if a_padded[i] == 1 {
                out_shape[i] = b_padded[i];
            } else if b_padded[i] == 1 {
                out_shape[i] = a_padded[i];
            } else {
                tq_bail!("broadcast: incompatible shapes {:?} vs {:?}", a_shape, b_shape);
            }
        }

        let out_n: usize = out_shape.iter().product();
        let mut result = Vec::with_capacity(out_n);

        // Compute strides for broadcasting
        let a_strides = compute_broadcast_strides(&a_padded, &out_shape);
        let b_strides = compute_broadcast_strides(&b_padded, &out_shape);
        let out_strides = compute_strides(&out_shape);

        for flat_idx in 0..out_n {
            let mut a_idx = 0usize;
            let mut b_idx = 0usize;
            let mut remaining = flat_idx;
            for d in 0..max_rank {
                let coord = remaining / out_strides[d];
                remaining %= out_strides[d];
                a_idx += coord * a_strides[d];
                b_idx += coord * b_strides[d];
            }
            result.push(f(a_data[a_idx], b_data[b_idx]));
        }

        TqTensor::from_vec(result, out_shape, a.device())
    }
}

// ─── Helpers ───────────────────────────────────────────────────

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn compute_broadcast_strides(src_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let src_strides = compute_strides(src_shape);
    src_strides.iter().zip(src_shape.iter()).zip(out_shape.iter())
        .map(|((&stride, &src_dim), &out_dim)| {
            if src_dim == 1 && out_dim > 1 { 0 } else { stride }
        })
        .collect()
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = vec![1usize; max_len];
    for i in 0..max_len {
        let ai = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
        let bi = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
        if ai == bi {
            result[i] = ai;
        } else if ai == 1 {
            result[i] = bi;
        } else if bi == 1 {
            result[i] = ai;
        } else {
            tq_bail!("broadcast: incompatible dims {} vs {} at position {}", ai, bi, i);
        }
    }
    Ok(result)
}
