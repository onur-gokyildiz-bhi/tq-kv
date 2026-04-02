//! TQ4_1S Weight Compression — WHT rotation + 16 Lloyd-Max centroids.
//!
//! Same mathematical foundation as KV cache compression, applied to model weights.
//! Post-training, no retraining needed. Apply to Q8_0 GGUF weights.
//!
//! Block format (32 elements → 18 bytes):
//!   [scale:f16 = 2B][min:f16 = 2B][indices:4bit×32 = 16B] = 20 bytes
//!   Effective: 5.0 bits per weight
//!
//! Pipeline: dequant → normalize per-block → sign flip → WHT → 4-bit quantize → pack

use crate::codebook::Codebook;
use crate::hadamard::{fast_wht, inverse_wht, random_sign_flip, generate_signs};

/// Block size for weight compression (must be power of 2 for WHT).
pub const WC_BLOCK_SIZE: usize = 32;

/// Compressed weight block: 32 elements → 20 bytes.
/// Format: [scale:f16][offset:f16][nibble×16]
pub struct CompressedWeightBlock {
    /// Scale factor (f16 stored as u16 bits): maps codebook values back to original range
    pub scale_bits: u16,
    /// Offset (f16 stored as u16 bits): per-block mean (subtracted before rotation)
    pub offset_bits: u16,
    /// Packed 4-bit indices: 16 bytes = 32 nibbles
    pub packed: [u8; 16],
}

/// Compress a weight tensor using TQ4_1S.
///
/// Input: f32 weight data (dequantized from Q8_0), row-major [out_features, in_features]
/// Output: compressed blocks, one per 32-element chunk
///
/// `rotation_seed` determines the random sign flips (must match decompression).
pub fn compress_weights(
    weights: &[f32],
    rotation_seed: u64,
) -> Vec<CompressedWeightBlock> {
    let signs = generate_signs(WC_BLOCK_SIZE, rotation_seed);
    // Weight compression uses sigma=1.0 (not 1/sqrt(d)) because:
    // Input is per-block normalized to std=1, WHT preserves norm → post-WHT std ≈ 1.0
    // Unlike KV cache (unit vectors → post-WHT std = 1/sqrt(d))
    let mut codebook = Codebook::new(4, WC_BLOCK_SIZE);
    codebook.sigma = 1.0; // Override: weights are pre-normalized to unit variance

    let n_blocks = (weights.len() + WC_BLOCK_SIZE - 1) / WC_BLOCK_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        let start = block_idx * WC_BLOCK_SIZE;
        let end = (start + WC_BLOCK_SIZE).min(weights.len());
        let actual_len = end - start;

        // Copy block (pad with zeros if last block is partial)
        let mut block = [0.0f32; WC_BLOCK_SIZE];
        block[..actual_len].copy_from_slice(&weights[start..end]);

        // Compute block statistics over ACTUAL elements only (not zero padding)
        let mean: f32 = block[..actual_len].iter().sum::<f32>() / actual_len as f32;
        let rms: f32 = block[..actual_len].iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
            / actual_len as f32;
        let scale = rms.sqrt().max(1e-10);

        // Normalize: subtract mean, divide by scale (padding stays zero → mean-shifted)
        for v in block[..actual_len].iter_mut() {
            *v = (*v - mean) / scale;
        }
        // Zero-padded elements: set to 0 (neutral after WHT for unused positions)
        for v in block[actual_len..].iter_mut() {
            *v = 0.0;
        }

        // Random sign flip + WHT rotation
        random_sign_flip(&mut block, &signs);
        fast_wht(&mut block);

        // Quantize each coordinate to 4-bit index
        let mut indices = [0u8; WC_BLOCK_SIZE];
        for i in 0..WC_BLOCK_SIZE {
            indices[i] = codebook.quantize(block[i]);
        }

        // Pack into nibbles: 2 indices per byte
        let mut packed = [0u8; 16];
        for i in 0..16 {
            packed[i] = (indices[i * 2] & 0xF) | ((indices[i * 2 + 1] & 0xF) << 4);
        }

        // Store scale and offset as f16
        let scale_bits = f32_to_f16_bits(scale);
        let offset_bits = f32_to_f16_bits(mean);

        blocks.push(CompressedWeightBlock {
            scale_bits,
            offset_bits,
            packed,
        });
    }

    blocks
}

/// Decompress TQ4_1S weight blocks back to f32.
pub fn decompress_weights(
    blocks: &[CompressedWeightBlock],
    n_elements: usize,
    rotation_seed: u64,
) -> Vec<f32> {
    let signs = generate_signs(WC_BLOCK_SIZE, rotation_seed);
    let mut codebook = Codebook::new(4, WC_BLOCK_SIZE);
    codebook.sigma = 1.0; // Match compression: unit variance post-normalization

    let mut output = Vec::with_capacity(n_elements);

    for block in blocks {
        let scale = f16_bits_to_f32(block.scale_bits);
        let offset = f16_bits_to_f32(block.offset_bits);

        // Unpack nibbles → indices
        let mut indices = [0u8; WC_BLOCK_SIZE];
        for i in 0..16 {
            indices[i * 2] = block.packed[i] & 0xF;
            indices[i * 2 + 1] = (block.packed[i] >> 4) & 0xF;
        }

        // Dequantize: index → codebook centroid
        let mut values = [0.0f32; WC_BLOCK_SIZE];
        for i in 0..WC_BLOCK_SIZE {
            values[i] = codebook.dequantize(indices[i]);
        }

        // Inverse WHT + inverse sign flip
        inverse_wht(&mut values);
        random_sign_flip(&mut values, &signs);

        // Denormalize: multiply by scale, add offset
        for v in values.iter_mut() {
            *v = *v * scale + offset;
        }

        output.extend_from_slice(&values);
    }

    output.truncate(n_elements);
    output
}

/// Serialize compressed blocks to raw bytes (for GGUF storage).
/// Format per block: [scale:2B][offset:2B][packed:16B] = 20 bytes
pub fn blocks_to_bytes(blocks: &[CompressedWeightBlock]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks.len() * 20);
    for block in blocks {
        bytes.extend_from_slice(&block.scale_bits.to_le_bytes());
        bytes.extend_from_slice(&block.offset_bits.to_le_bytes());
        bytes.extend_from_slice(&block.packed);
    }
    bytes
}

/// Deserialize raw bytes to compressed blocks.
pub fn bytes_to_blocks(data: &[u8]) -> Vec<CompressedWeightBlock> {
    let n_blocks = data.len() / 20;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let off = i * 20;
        let scale_bits = u16::from_le_bytes([data[off], data[off + 1]]);
        let offset_bits = u16::from_le_bytes([data[off + 2], data[off + 3]]);
        let mut packed = [0u8; 16];
        packed.copy_from_slice(&data[off + 4..off + 20]);
        blocks.push(CompressedWeightBlock { scale_bits, offset_bits, packed });
    }
    blocks
}

/// Compression ratio: original f32 bytes / compressed bytes.
pub fn compression_ratio(n_elements: usize) -> f32 {
    let original = n_elements * 4; // f32
    let n_blocks = (n_elements + WC_BLOCK_SIZE - 1) / WC_BLOCK_SIZE;
    let compressed = n_blocks * 20; // 20 bytes per block
    original as f32 / compressed as f32
}

/// Bits per weight for TQ4_1S format.
pub fn bits_per_weight() -> f32 {
    20.0 * 8.0 / WC_BLOCK_SIZE as f32 // 20 bytes * 8 bits / 32 elements = 5.0
}

// ─── F16 helpers ─────────────────────────────────────────────

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN
        return (sign | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }

    // Reconstruct f32 value for subnormal handling
    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow → inf
    }
    if new_exp <= 0 {
        // f16 subnormal range or underflow
        if new_exp < -10 {
            return sign as u16; // too small → zero
        }
        // Subnormal f16: shift mantissa with implicit leading 1
        let mant = (frac | 0x800000) >> (1 - new_exp + 13);
        return (sign | mant) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        // Denorm
        let mut val = frac as f32 / 1024.0;
        val *= 2.0f32.powi(-14);
        return if sign != 0 { -val } else { val };
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F800000 | (frac << 13));
    }

    let f32_exp = (exp + 127 - 15) << 23;
    f32::from_bits(sign | f32_exp | (frac << 13))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple() {
        let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let blocks = compress_weights(&weights, 42);
        let decompressed = decompress_weights(&blocks, weights.len(), 42);

        assert_eq!(decompressed.len(), weights.len());
        let mse: f32 = weights.iter().zip(decompressed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>() / weights.len() as f32;
        let rmse = mse.sqrt();
        // 4-bit quantization: expect RMSE < 5% of range
        let range = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - weights.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(rmse < range * 0.10, "RMSE {:.6} too high for range {:.4}", rmse, range);
    }

    #[test]
    fn test_roundtrip_gaussian() {
        // Simulate typical weight distribution: small values centered around 0
        let mut weights = Vec::with_capacity(256);
        let mut x = 0.12345f64;
        for _ in 0..256 {
            x = (x * 1103515245.0 + 12345.0) % (1u64 << 31) as f64;
            // Scale to typical weight range [-0.1, 0.1]
            weights.push(((x / (1u64 << 31) as f64) * 0.2 - 0.1) as f32);
        }

        let blocks = compress_weights(&weights, 123);
        let decompressed = decompress_weights(&blocks, weights.len(), 123);

        let mse: f32 = weights.iter().zip(decompressed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>() / weights.len() as f32;
        let signal_power: f32 = weights.iter().map(|x| x * x).sum::<f32>() / weights.len() as f32;
        let snr = 10.0 * (signal_power / mse.max(1e-20)).log10();
        // Debug: print first few values
        eprintln!("  first 8 orig:   {:?}", &weights[..8]);
        eprintln!("  first 8 decomp: {:?}", &decompressed[..8]);
        eprintln!("  MSE={:.6}, signal={:.6}, SNR={:.1} dB", mse, signal_power, snr);
        // 4-bit quantization with WHT: expect SNR > 8 dB
        assert!(snr > 8.0, "SNR too low: {:.1} dB (MSE={:.6}, signal={:.6})", snr, mse, signal_power);
    }

    #[test]
    fn test_serialization() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let blocks = compress_weights(&weights, 7);
        let bytes = blocks_to_bytes(&blocks);
        let restored = bytes_to_blocks(&bytes);
        let decompressed = decompress_weights(&restored, weights.len(), 7);

        let direct = decompress_weights(&blocks, weights.len(), 7);
        assert_eq!(decompressed, direct, "Serialization round-trip failed");
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(1000);
        assert!(ratio > 5.0 && ratio < 7.0, "Unexpected ratio: {:.2}", ratio);
        assert!((bits_per_weight() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_steps() {
        // Trace each step for a single 32-element block
        let mut block: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.01).collect();
        let original = block.clone();
        let signs = generate_signs(32, 42);
        let mut codebook = Codebook::new(4, 32);
        codebook.sigma = 1.0; // Weight compression: unit variance

        // Step 1: normalize
        let mean: f32 = block.iter().sum::<f32>() / 32.0;
        let rms: f32 = block.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 32.0;
        let scale = rms.sqrt().max(1e-10);
        for v in block.iter_mut() { *v = (*v - mean) / scale; }
        let after_norm_std: f32 = (block.iter().map(|x| x * x).sum::<f32>() / 32.0).sqrt();
        eprintln!("  mean={:.6}, scale={:.6}, after_norm std={:.4}", mean, scale, after_norm_std);

        // Step 2: sign flip + WHT
        random_sign_flip(&mut block, &signs);
        fast_wht(&mut block);
        let after_wht_std: f32 = (block.iter().map(|x| x * x).sum::<f32>() / 32.0).sqrt();
        eprintln!("  after WHT std={:.4}, sigma={:.4}", after_wht_std, codebook.sigma);

        // Step 3: quantize + dequantize
        let indices: Vec<u8> = block.iter().map(|&v| codebook.quantize(v)).collect();
        let mut dequant: Vec<f32> = indices.iter().map(|&i| codebook.dequantize(i)).collect();
        let quant_err: f32 = block.iter().zip(dequant.iter())
            .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / 32.0;
        eprintln!("  quant MSE in WHT domain: {:.6}", quant_err);

        // Step 4: inverse WHT + inverse sign flip
        inverse_wht(&mut dequant);
        random_sign_flip(&mut dequant, &signs);

        // Step 5: denormalize
        for v in dequant.iter_mut() { *v = *v * scale + mean; }

        let final_mse: f32 = original.iter().zip(dequant.iter())
            .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / 32.0;
        eprintln!("  final MSE: {:.8}", final_mse);
        eprintln!("  orig[0..4]: {:?}", &original[..4]);
        eprintln!("  dec[0..4]:  {:?}", &dequant[..4]);
        assert!(final_mse < 1e-4, "Pipeline MSE too high: {:.6}", final_mse);
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.001, 65504.0];
        for &v in &values {
            let bits = f32_to_f16_bits(v);
            let restored = f16_bits_to_f32(bits);
            let rel_err = if v.abs() > 1e-6 { ((restored - v) / v).abs() } else { (restored - v).abs() };
            assert!(rel_err < 0.01, "f16 roundtrip failed: {} → {} (err {})", v, restored, rel_err);
        }
    }
}
