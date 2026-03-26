//! CUDA-compatible softmax over the last dimension for candle-nn.
//!
//! # Problem
//!
//! `candle_nn::ops::softmax_last_dim` has no CUDA kernel. Calling it on a
//! CUDA tensor panics with:
//!
//!     "no cuda implementation for softmax"
//!
//! # Solution
//!
//! Implement softmax using the numerically-stable log-sum-exp decomposition
//! with only primitive ops that have CUDA support:
//!
//!     max_val   = max(x, dim=-1, keepdim=True)
//!     exp_x     = exp(x - max_val)
//!     sum_exp   = sum(exp_x, dim=-1, keepdim=True)
//!     softmax   = exp_x / sum_exp
//!
//! Subtracting the max before exponentiation prevents overflow.
//!
//! # Intended upstream location
//!
//! Replace or augment `softmax_last_dim` in `candle-nn/src/ops.rs`.

use candle_core::{Result, Tensor};

/// Numerically-stable softmax over the last dimension.
///
/// Works on any device (CPU, CUDA, Metal) because it only uses ops that
/// have universal backend support: `max_keepdim`, `broadcast_sub`, `exp`,
/// `sum_keepdim`, `broadcast_div`.
///
/// Replaces `candle_nn::ops::softmax_last_dim` which lacks a CUDA kernel.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let last = x.rank() - 1;
    // Subtract max for numerical stability (prevents exp overflow).
    let max_val = x.max_keepdim(last)?;
    let shifted = x.broadcast_sub(&max_val)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(last)?;
    exp.broadcast_div(&sum)
}

/// Softmax over an arbitrary dimension.
///
/// Generalized version for cases where the softmax dimension is not the last.
pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let max_val = x.max_keepdim(dim)?;
    let shifted = x.broadcast_sub(&max_val)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    exp.broadcast_div(&sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_softmax_1d() -> Result<()> {
        let device = &Device::Cpu;
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], device)?;
        let out = softmax_last_dim(&x)?;
        let vals = out.to_vec1::<f32>()?;

        // softmax([1,2,3]) = exp([1,2,3]) / sum(exp([1,2,3]))
        let exps: Vec<f32> = [1.0, 2.0, 3.0].iter().map(|v| v.exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (i, &v) in vals.iter().enumerate() {
            let expected = exps[i] / sum;
            assert!(
                (v - expected).abs() < 1e-6,
                "index {i}: got {v}, expected {expected}"
            );
        }

        // Probabilities should sum to 1.
        let total: f32 = vals.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "sum = {total}, expected 1.0");
        Ok(())
    }

    #[test]
    fn test_softmax_2d_last_dim() -> Result<()> {
        let device = &Device::Cpu;
        // Two rows -- softmax should be applied independently per row.
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 1.0]], device)?;
        let out = softmax_last_dim(&x)?;
        let vals = out.to_vec2::<f32>()?;

        for row in &vals {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "row sum = {sum}, expected 1.0"
            );
        }

        // Row 0: softmax([1,2]) -> [e^1/(e^1+e^2), e^2/(e^1+e^2)]
        // Second element should be larger.
        assert!(vals[0][1] > vals[0][0]);
        // Row 1: softmax([3,1]) -> first should be larger.
        assert!(vals[1][0] > vals[1][1]);
        Ok(())
    }

    #[test]
    fn test_softmax_numerical_stability() -> Result<()> {
        let device = &Device::Cpu;
        // Large values that would overflow naive exp().
        let x = Tensor::new(&[1000.0f32, 1001.0, 1002.0], device)?;
        let out = softmax_last_dim(&x)?;
        let vals = out.to_vec1::<f32>()?;

        // Should still produce valid probabilities (no NaN/Inf).
        let total: f32 = vals.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "sum = {total}, expected 1.0 (stability test)"
        );
        for &v in &vals {
            assert!(v.is_finite(), "got non-finite value {v}");
        }
        Ok(())
    }
}
