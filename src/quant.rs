//! GGML quantized format dequantization — Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32.
//!
//! Block-level dequantization for all GGUF quantization types.
//! Each function takes raw bytes and produces f32 output.

use crate::gguf::{GgmlDType, f16_to_f32};

// ─── Q8_0: 34 bytes → 32 f32 values ──────────────────────────

/// Dequantize Q8_0 blocks: [f16 d][i8 x 32] → f32.
pub fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 34;
    const BLOCK_NUMEL: usize = 32;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for i in 0..BLOCK_NUMEL {
            let qs = block[2 + i] as i8;
            output.push(qs as f32 * d);
        }
    }
    output
}

// ─── Q4_K (Q4_K_M): 144 bytes → 256 f32 values ──────────────

/// Extract per-sub-block (scale, min) from Q4_K/Q5_K scales array.
///
/// The 12-byte scales array packs 8 sub-block scales and 8 sub-block mins
/// using a 6-bit encoding with 2-bit shifts for sub-blocks 4-7.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q4_K_M blocks: [f16 d][f16 dmin][u8 scales x12][u8 qs x128] → 256 f32.
pub fn dequantize_q4k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 144;
    const BLOCK_NUMEL: usize = 256;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];   // 12 bytes
        let qs = &block[16..144];     // 128 bytes

        for sub in 0..8 {
            let (sc, m) = get_scale_min_k4(sub, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            for j in 0..16 {
                let byte = qs[sub * 16 + j];
                let lo = (byte & 0xF) as f32;
                let hi = (byte >> 4) as f32;
                output.push(d1 * lo - m1);
                output.push(d1 * hi - m1);
            }
        }
    }
    output
}

// ─── Q5_K (Q5_K_M): 176 bytes → 256 f32 values ──────────────

/// Dequantize Q5_K_M blocks: [f16 d][f16 dmin][u8 scales x12][u8 qh x32][u8 qs x128] → 256 f32.
pub fn dequantize_q5k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 176;
    const BLOCK_NUMEL: usize = 256;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];   // 12 bytes
        let qh = &block[16..48];      // 32 bytes — high bits
        let qs = &block[48..176];     // 128 bytes — low 4 bits

        for sub in 0..8 {
            let (sc, m) = get_scale_min_k4(sub, scales);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            for j in 0..16 {
                let byte = qs[sub * 16 + j];
                let lo = (byte & 0xF) as u32;
                let hi = (byte >> 4) as u32;

                // 5th bit from qh
                let qh_byte_idx = sub * 4 + j / 2;
                let qh_byte = if qh_byte_idx < 32 { qh[qh_byte_idx] } else { 0 };
                let bit_lo = ((qh_byte >> (2 * (j % 2))) & 1) as u32;
                let bit_hi = ((qh_byte >> (2 * (j % 2) + 1)) & 1) as u32;

                output.push(d1 * (lo | (bit_lo << 4)) as f32 - m1);
                output.push(d1 * (hi | (bit_hi << 4)) as f32 - m1);
            }
        }
    }
    output
}

// ─── Q6_K: 210 bytes → 256 f32 values ────────────────────────

/// Dequantize Q6_K blocks: [u8 ql x128][u8 qh x64][i8 scales x16][f16 d] → 256 f32.
pub fn dequantize_q6k(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 210;
    const BLOCK_NUMEL: usize = 256;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let ql = &block[0..128];       // 4-bit lower
        let qh = &block[128..192];     // 2-bit upper
        let scales = &block[192..208]; // i8 scales x16
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        for sub in 0..16 {
            let sc = scales[sub] as i8 as f32;
            let base = sub * 16;

            for j in 0..16 {
                let idx = base + j;
                let ql_byte = ql[idx / 2];
                let lo = if idx % 2 == 0 { ql_byte & 0xF } else { ql_byte >> 4 };

                let qh_idx = idx / 4;
                let qh_shift = (idx % 4) * 2;
                let hi = if qh_idx < 64 { (qh[qh_idx] >> qh_shift) & 3 } else { 0 };

                let val = ((lo as u32) | ((hi as u32) << 4)) as i32 - 32;
                output.push(d * sc * val as f32);
            }
        }
    }
    output
}

// ─── F16: 2 bytes → 1 f32 ────────────────────────────────────

pub fn dequantize_f16(data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        output.push(f16_to_f32(bits));
    }
    output
}

// ─── F32: passthrough ─────────────────────────────────────────

pub fn dequantize_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
        output.push(f32::from_le_bytes(bytes));
    }
    output
}

// ─── BF16: 2 bytes → 1 f32 ───────────────────────────────────

pub fn dequantize_bf16(data: &[u8], n_elements: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        // BF16 → F32: just shift left by 16
        output.push(f32::from_bits((bits as u32) << 16));
    }
    output
}

// ─── Dispatch ─────────────────────────────────────────────────

/// Dequantize raw tensor data to f32 based on GGML dtype.
pub fn dequantize(data: &[u8], dtype: GgmlDType, n_elements: usize) -> Vec<f32> {
    match dtype {
        GgmlDType::F32 => dequantize_f32(data, n_elements),
        GgmlDType::F16 => dequantize_f16(data, n_elements),
        GgmlDType::BF16 => dequantize_bf16(data, n_elements),
        GgmlDType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgmlDType::Q4K => dequantize_q4k(data, n_elements),
        GgmlDType::Q5K => dequantize_q5k(data, n_elements),
        GgmlDType::Q6K => dequantize_q6k(data, n_elements),
        _ => {
            eprintln!("WARNING: unsupported dequant for {:?}, returning zeros", dtype);
            vec![0.0; n_elements]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_roundtrip() {
        // Construct a Q8_0 block: d=0.5 (f16), values [1, -1, 2, -2, ...]
        let d_f16: u16 = 0x3800; // 0.5 in f16
        let mut block = vec![0u8; 34];
        block[0] = d_f16 as u8;
        block[1] = (d_f16 >> 8) as u8;
        // Set qs[0] = 2 (i8), qs[1] = -4 (i8)
        block[2] = 2u8;      // 2 as i8
        block[3] = (-4i8) as u8; // -4 as i8

        let result = dequantize_q8_0(&block, 32);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 1e-3); // 2 * 0.5 = 1.0
        assert!((result[1] - (-2.0)).abs() < 1e-3); // -4 * 0.5 = -2.0
    }

    #[test]
    fn test_q4k_basic() {
        // Minimal Q4K block: all zeros except d=1.0
        let mut block = vec![0u8; 144];
        let d_f16: u16 = 0x3C00; // 1.0 in f16
        block[0] = d_f16 as u8;
        block[1] = (d_f16 >> 8) as u8;
        // dmin = 0, scales = 0 → all outputs should be 0
        let result = dequantize_q4k(&block, 256);
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_f16_dequant() {
        let data = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0 in f16
        let result = dequantize_f16(&data, 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_dequant() {
        // BF16 1.0 = 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000)
        let data = vec![0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0 in bf16
        let result = dequantize_bf16(&data, 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_scale_min_k4() {
        // Simple test: scales [1,2,3,4, 5,6,7,8, 0,0,0,0]
        let scales = [1u8, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0];
        // sub-block 0: d = 1 & 63 = 1, m = 5 & 63 = 5
        assert_eq!(get_scale_min_k4(0, &scales), (1, 5));
        // sub-block 3: d = 4 & 63 = 4, m = 8 & 63 = 8
        assert_eq!(get_scale_min_k4(3, &scales), (4, 8));
    }
}
