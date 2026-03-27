//! CUDA-compatible RMS Normalization for candle-nn.
//!
//! # Problem
//!
//! `candle_nn::ops::rms_norm` has no CUDA kernel dispatch. When the input tensor
//! lives on a CUDA device, calling `rms_norm` panics with:
//!
//!     "no cuda implementation for rms-norm"
//!
//! # Solution
//!
//! Decompose RMS normalization into primitive ops that all have CUDA support:
//! `sqr`, `mean_keepdim`, `sqrt`, `broadcast_div`, `broadcast_mul`, `to_dtype`.
//!
//! The math is identical to the standard RMS norm:
//!
//!     rms = sqrt(mean(x^2) + eps)
//!     output = (x / rms) * weight
//!
//! Computation is performed in f32 for numerical stability, then cast back to
//! the original dtype.
//!
//! # Intended upstream location
//!
//! Replace or augment `RmsNorm` in `candle-nn/src/ops.rs`.

use candle_core::{DType, Device, Result, Tensor};
use candle_core::quantized::QTensor;

/// RMS normalization layer that works on any device (CPU, CUDA, Metal).
///
/// Unlike the current `candle_nn::RmsNorm`, this implementation uses only
/// primitive tensor operations, all of which have CUDA kernels.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learned scale parameter (gamma), shape `[hidden_size]`.
    weight: Tensor,
    /// Small constant added to variance for numerical stability.
    eps: f64,
}

impl RmsNorm {
    /// Create from a float tensor (already on the target device).
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Create from a quantized tensor, dequantizing onto `device`.
    ///
    /// This is the common path for GGUF quantized models where norm weights
    /// are stored as QTensors but need to be used as float for computation.
    pub fn from_qtensor(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        Ok(Self { weight, eps })
    }
}

impl candle_nn::Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Promote to f32 for stable variance computation.
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        // variance = mean(x^2) over the last dimension
        let variance = x.sqr()?.mean_keepdim(x.rank() - 1)?;

        // rms = sqrt(variance + eps)
        let rms = (variance + self.eps)?.sqrt()?;

        // normalized = x / rms
        let normalized = x.broadcast_div(&rms)?;

        // Apply learned scale and cast back to original dtype.
        normalized
            .broadcast_mul(&self.weight)?
            .to_dtype(x_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_rms_norm_basic() -> Result<()> {
        let device = &Device::Cpu;
        let eps = 1e-5;

        // Weight = all ones (identity scale)
        let weight = Tensor::ones(&[4], DType::F32, device)?;
        let norm = RmsNorm::new(weight, eps);

        // Input: [1, 2, 3, 4]
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?.unsqueeze(0)?;
        let out = norm.forward(&x)?;
        let out_vec = out.squeeze(0)?.to_vec1::<f32>()?;

        // Manual: rms = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.50001) ~ 2.7386
        // normalized = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
        //            ~ [0.3651, 0.7303, 1.0954, 1.4606]
        let rms = (7.5f32 + eps as f32).sqrt();
        for (i, &v) in out_vec.iter().enumerate() {
            let expected = (i + 1) as f32 / rms;
            assert!(
                (v - expected).abs() < 1e-4,
                "index {i}: got {v}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rms_norm_with_weight() -> Result<()> {
        let device = &Device::Cpu;
        let eps = 1e-6;

        // Weight = [2.0, 0.5, 1.0]
        let weight = Tensor::new(&[2.0f32, 0.5, 1.0], device)?;
        let norm = RmsNorm::new(weight, eps);

        let x = Tensor::new(&[3.0f32, 6.0, 9.0], device)?.unsqueeze(0)?;
        let out = norm.forward(&x)?;
        let out_vec = out.squeeze(0)?.to_vec1::<f32>()?;

        // rms = sqrt((9 + 36 + 81)/3) = sqrt(42) ~ 6.4807
        // norm = [3/6.4807, 6/6.4807, 9/6.4807] ~ [0.4629, 0.9258, 1.3887]
        // scaled = [0.9258, 0.4629, 1.3887]
        let rms = 42.0f32.sqrt();
        let expected = [3.0 / rms * 2.0, 6.0 / rms * 0.5, 9.0 / rms * 1.0];
        for (i, (&got, &exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "index {i}: got {got}, expected {exp}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rms_norm_preserves_dtype() -> Result<()> {
        let device = &Device::Cpu;
        let weight = Tensor::ones(&[4], DType::F32, device)?;
        let norm = RmsNorm::new(weight, 1e-5);

        // Input in f16 -- output should also be f16.
        let x = Tensor::ones(&[1, 4], DType::F16, device)?;
        let out = norm.forward(&x)?;
        assert_eq!(out.dtype(), DType::F16);
        Ok(())
    }
}
