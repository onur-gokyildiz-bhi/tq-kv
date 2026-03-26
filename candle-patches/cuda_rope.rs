//! CUDA-compatible interleaved Rotary Position Embeddings (RoPE) for candle-nn.
//!
//! # Problem
//!
//! `candle_nn::rotary_emb::rope_i` (interleaved RoPE) has no CUDA kernel.
//! Calling it on a CUDA tensor panics with:
//!
//!     "no cuda implementation for rope_i"
//!
//! # Solution
//!
//! Implement interleaved RoPE using primitive tensor ops that all have CUDA
//! support: `reshape`, `narrow`, `squeeze`, `broadcast_mul`, `unsqueeze`, `cat`.
//!
//! ## Rotation Math
//!
//! RoPE applies a 2D rotation to consecutive pairs of dimensions. For
//! interleaved layout, dimension pairs are `(x[0], x[1])`, `(x[2], x[3])`, etc.
//!
//! Given a pair `(x_even, x_odd)` and precomputed `(cos_theta, sin_theta)`:
//!
//!     rotated_even = x_even * cos_theta - x_odd  * sin_theta
//!     rotated_odd  = x_even * sin_theta + x_odd  * cos_theta
//!
//! This is equivalent to multiplying by the 2x2 rotation matrix:
//!
//!     | cos  -sin |   | x_even |   | rotated_even |
//!     | sin   cos | * | x_odd  | = | rotated_odd  |
//!
//! ## Layout
//!
//! Input `x` has shape `(batch, n_heads, seq_len, head_dim)`.
//! `cos` and `sin` have shape `(seq_len, head_dim/2)` -- one angle per pair.
//!
//! The implementation:
//! 1. Reshapes `x` to `(B, H, S, half, 2)` to expose even/odd pairs.
//! 2. Splits along the last dim to get `x0` (even) and `x1` (odd).
//! 3. Applies the rotation formula.
//! 4. Interleaves the results back and reshapes to `(B, H, S, D)`.
//!
//! # Intended upstream location
//!
//! Replace or augment `rope_i` in `candle-nn/src/rotary_emb.rs`.

use candle_core::{Result, Tensor};

/// Apply interleaved Rotary Position Embeddings (RoPE).
///
/// Works on any device (CPU, CUDA, Metal) because it only uses ops with
/// universal backend support.
///
/// # Arguments
///
/// * `x`   - Input tensor of shape `(batch, n_heads, seq_len, head_dim)`.
///           Must be contiguous. `head_dim` must be even.
/// * `cos` - Cosine table, shape `(seq_len, head_dim / 2)`.
/// * `sin` - Sine table, shape `(seq_len, head_dim / 2)`.
///
/// # Returns
///
/// Tensor of the same shape as `x` with rotary embeddings applied.
///
/// Replaces `candle_nn::rotary_emb::rope_i` which lacks a CUDA kernel.
pub fn rope_i(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let half = d / 2;

    // Reshape to expose interleaved pairs: (B, H, S, half, 2)
    let x = x.reshape((b, h, s, half, 2))?;

    // Split into even and odd components.
    let x0 = x.narrow(4, 0, 1)?.squeeze(4)?; // shape: (B, H, S, half)
    let x1 = x.narrow(4, 1, 1)?.squeeze(4)?; // shape: (B, H, S, half)

    // Broadcast cos/sin from (S, half) to (1, 1, S, half) for batch/head dims.
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Apply 2D rotation per pair:
    //   r0 = x0 * cos - x1 * sin
    //   r1 = x0 * sin + x1 * cos
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

    // Interleave back: stack on dim 4 then flatten to head_dim.
    let r0 = r0.unsqueeze(4)?; // (B, H, S, half, 1)
    let r1 = r1.unsqueeze(4)?; // (B, H, S, half, 1)
    Tensor::cat(&[&r0, &r1], 4)?.reshape((b, h, s, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Build cos/sin tables for `seq_len` positions and `half` frequency pairs.
    fn build_cos_sin(
        seq_len: usize,
        half: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let base = 10000.0f32;
        let mut cos_data = Vec::with_capacity(seq_len * half);
        let mut sin_data = Vec::with_capacity(seq_len * half);
        for pos in 0..seq_len {
            for i in 0..half {
                let freq = 1.0 / base.powf(2.0 * i as f32 / (2 * half) as f32);
                let angle = pos as f32 * freq;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos_data, (seq_len, half), device)?;
        let sin = Tensor::from_vec(sin_data, (seq_len, half), device)?;
        Ok((cos, sin))
    }

    #[test]
    fn test_rope_i_identity_at_position_zero() -> Result<()> {
        let device = &Device::Cpu;
        let (b, h, s, d) = (1, 1, 1, 4);
        let half = d / 2;

        // At position 0, angles are all 0 -> cos=1, sin=0 -> identity rotation.
        let cos = Tensor::ones(&[s, half], DType::F32, device)?;
        let sin = Tensor::zeros(&[s, half], DType::F32, device)?;

        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?
            .reshape((b, h, s, d))?;
        let out = rope_i(&x, &cos, &sin)?;
        let out_vec = out.flatten_all()?.to_vec1::<f32>()?;

        // Output should equal input when sin=0, cos=1.
        assert_eq!(out_vec, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_rope_i_90_degree_rotation() -> Result<()> {
        let device = &Device::Cpu;
        let (b, h, s, d) = (1, 1, 1, 4);
        let half = d / 2;

        // 90-degree rotation: cos=0, sin=1
        //   r0 = x0*0 - x1*1 = -x1
        //   r1 = x0*1 + x1*0 =  x0
        // Input pairs: (1,2), (3,4) -> (-2,1), (-4,3)
        // Interleaved output: [-2, 1, -4, 3]
        let cos = Tensor::zeros(&[s, half], DType::F32, device)?;
        let sin = Tensor::ones(&[s, half], DType::F32, device)?;

        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?
            .reshape((b, h, s, d))?;
        let out = rope_i(&x, &cos, &sin)?;
        let out_vec = out.flatten_all()?.to_vec1::<f32>()?;

        let expected = vec![-2.0f32, 1.0, -4.0, 3.0];
        for (i, (&got, &exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "index {i}: got {got}, expected {exp}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rope_i_preserves_norm() -> Result<()> {
        // Rotation should preserve the L2 norm of each pair.
        let device = &Device::Cpu;
        let (b, h, s, d) = (1, 2, 3, 8);
        let half = d / 2;

        let (cos, sin) = build_cos_sin(s, half, device)?;

        // Random-ish input.
        let x_data: Vec<f32> = (0..(b * h * s * d))
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let x = Tensor::from_vec(x_data.clone(), (b, h, s, d), device)?;

        let out = rope_i(&x, &cos, &sin)?;
        let out_flat = out.flatten_all()?.to_vec1::<f32>()?;

        // Check norm preservation for each (position, head) vector.
        for idx in 0..(b * h * s) {
            let base = idx * d;
            let in_norm: f32 = x_data[base..base + d]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            let out_norm: f32 = out_flat[base..base + d]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!(
                (in_norm - out_norm).abs() < 1e-4,
                "vector {idx}: in_norm={in_norm}, out_norm={out_norm}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rope_i_multi_position() -> Result<()> {
        // Verify that different positions get different rotations.
        let device = &Device::Cpu;
        let (b, h, s, d) = (1, 1, 2, 4);
        let half = d / 2;

        let (cos, sin) = build_cos_sin(s, half, device)?;

        // Same vector at two different positions.
        let x = Tensor::new(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], device)?
            .reshape((b, h, s, d))?;
        let out = rope_i(&x, &cos, &sin)?;
        let vals = out.flatten_all()?.to_vec1::<f32>()?;

        // Position 0 and position 1 should produce different outputs.
        let pos0 = &vals[0..d];
        let pos1 = &vals[d..2 * d];
        let differ = pos0.iter().zip(pos1).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "different positions should yield different rotations");
        Ok(())
    }
}
