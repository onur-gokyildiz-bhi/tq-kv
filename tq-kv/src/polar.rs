//! PolarQuant: Quantization via Cartesian → Polar coordinate transform.
//!
//! Standard quantization quantizes each component independently.
//! PolarQuant separates the vector into norm + direction:
//!   - Norm: single scalar, stored at high precision
//!   - Direction (unit vector): uniformly quantized at low bit width
//!
//! After Hadamard rotation, components are approximately Gaussian,
//! which provides ideal conditions for uniform quantization.

/// A single vector compressed with PolarQuant.
#[derive(Clone, Debug)]
pub struct PolarQuantized {
    /// L2 norm of the original vector (stored at f32 precision)
    pub norm: f32,
    /// Quantized unit vector components (each `bits` bits)
    pub quantized_unit: Vec<u8>,
    /// Scale for dequantization
    pub scale: f32,
    /// Zero point for dequantization
    pub zero_point: f32,
}

/// Quantization configuration
#[derive(Clone, Debug)]
pub struct PolarConfig {
    /// Bit width (3 or 4)
    pub bits: u8,
}

impl Default for PolarConfig {
    fn default() -> Self {
        Self { bits: 4 }
    }
}

impl PolarConfig {
    /// Maximum quantized value
    #[inline]
    fn max_val(&self) -> u8 {
        (1u16 << self.bits) as u8 - 1
    }
}

/// Compress a single f32 vector with PolarQuant.
///
/// Input: Hadamard-rotated vector (components ~ Gaussian)
/// Output: norm + quantized unit vector
pub fn quantize(vector: &[f32], config: &PolarConfig) -> PolarQuantized {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Zero vector
    if norm < 1e-10 {
        return PolarQuantized {
            norm: 0.0,
            quantized_unit: vec![0; vector.len()],
            scale: 1.0,
            zero_point: 0.0,
        };
    }

    // Unit vector: x / ||x||
    let unit: Vec<f32> = vector.iter().map(|x| x / norm).collect();

    // Min-max quantization
    let min_val = unit.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = unit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    let max_q = config.max_val() as f32;
    let (scale, zero_point) = if range < 1e-10 {
        (1.0, min_val)
    } else {
        (range / max_q, min_val)
    };

    let quantized_unit: Vec<u8> = unit
        .iter()
        .map(|&u| {
            let q = ((u - zero_point) / scale).round();
            q.clamp(0.0, max_q) as u8
        })
        .collect();

    PolarQuantized {
        norm,
        quantized_unit,
        scale,
        zero_point,
    }
}

/// Decompress (dequantize) PolarQuant data.
///
/// Output: approximate original vector (in Hadamard domain)
pub fn dequantize(pq: &PolarQuantized) -> Vec<f32> {
    pq.quantized_unit
        .iter()
        .map(|&q| {
            let u = q as f32 * pq.scale + pq.zero_point;
            u * pq.norm
        })
        .collect()
}

/// Batch quantize: compress multiple vectors at once.
/// Each vector is `dim`-dimensional.
pub fn quantize_batch(data: &[f32], dim: usize, config: &PolarConfig) -> Vec<PolarQuantized> {
    assert_eq!(data.len() % dim, 0, "Data length must be evenly divisible by dim");
    data.chunks_exact(dim)
        .map(|chunk| quantize(chunk, config))
        .collect()
}

/// Batch dequantize.
pub fn dequantize_batch(quantized: &[PolarQuantized], dim: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(quantized.len() * dim);
    for pq in quantized {
        result.extend_from_slice(&dequantize(pq));
    }
    result
}

/// Compute quantization error (MSE).
pub fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    let sum_sq_err: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    sum_sq_err / original.len() as f32
}

/// Compute compression ratio.
/// Original: dim * 32 bits (f32)
/// Compressed: 32 (norm) + 32 (scale) + 32 (zero) + dim * bits
pub fn compression_ratio(dim: usize, bits: u8) -> f32 {
    let original_bits = dim as f32 * 32.0;
    let compressed_bits = 96.0 + dim as f32 * bits as f32; // 3x32 metadata + quantized values
    original_bits / compressed_bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() {
        let config = PolarConfig { bits: 4 };
        let vector = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.6, 0.4, 0.7];
        let pq = quantize(&vector, &config);
        let reconstructed = dequantize(&pq);

        // Norm should be preserved (approximately)
        let orig_norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (orig_norm - pq.norm).abs() < 1e-5,
            "Norm difference: {} vs {}",
            orig_norm,
            pq.norm
        );

        // Reconstruction error should be low
        let mse = compute_mse(&vector, &reconstructed);
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn test_compression_ratio_4bit() {
        let ratio = compression_ratio(128, 4);
        // 128 * 32 / (96 + 128 * 4) = 4096 / 608 ≈ 6.7x
        assert!(ratio > 6.0, "4-bit compression ratio expected: >6x, actual: {:.1}x", ratio);
    }

    #[test]
    fn test_compression_ratio_3bit() {
        let ratio = compression_ratio(128, 3);
        // 128 * 32 / (96 + 128 * 3) = 4096 / 480 ≈ 8.5x
        assert!(ratio > 8.0, "3-bit compression ratio expected: >8x, actual: {:.1}x", ratio);
    }

    #[test]
    fn test_batch_roundtrip() {
        let config = PolarConfig { bits: 4 };
        let data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let batch = quantize_batch(&data, 8, &config);
        let reconstructed = dequantize_batch(&batch, 8);
        let mse = compute_mse(&data, &reconstructed);
        assert!(mse < 0.01, "Batch MSE too high: {}", mse);
    }

    #[test]
    fn test_zero_vector() {
        let config = PolarConfig { bits: 4 };
        let vector = vec![0.0; 8];
        let pq = quantize(&vector, &config);
        assert_eq!(pq.norm, 0.0);
        let reconstructed = dequantize(&pq);
        for v in &reconstructed {
            assert_eq!(*v, 0.0);
        }
    }
}
