//! GGML quantized format dequantization — Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32.
//!
//! Block-level dequantization for all GGUF quantization types.
//! Each function takes raw bytes and produces f32 output.

use crate::gguf::{GgmlDType, f16_to_f32};

/// Global TQ4_1S rotation seed. Set from GGUF metadata "tq.weight_compress_seed"
/// at model load time. Default: 42.
pub static TQ4_1S_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(42);

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

        // Canonical GGML Q4_K layout: 128 qs bytes in 4 groups of 32 bytes.
        // Each group covers 2 sub-blocks: lo nibbles → sub 2*j, hi nibbles → sub 2*j+1.
        // Each sub-block has its own (scale, min) pair from get_scale_min_k4.
        for j in 0..4 {
            let (sc_lo, m_lo) = get_scale_min_k4(2 * j, scales);
            let (sc_hi, m_hi) = get_scale_min_k4(2 * j + 1, scales);
            let d_lo = d * sc_lo as f32;
            let d_hi = d * sc_hi as f32;
            let dm_lo = dmin * m_lo as f32;
            let dm_hi = dmin * m_hi as f32;

            // 32 bytes → 64 values: lo nibbles first (32 values), then hi nibbles (32 values)
            let q_off = j * 32;
            for l in 0..32 {
                output.push(d_lo * (qs[q_off + l] & 0xF) as f32 - dm_lo);
            }
            for l in 0..32 {
                output.push(d_hi * (qs[q_off + l] >> 4) as f32 - dm_hi);
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

        // Canonical GGML Q5_K layout: 128 qs bytes in 4 groups of 32 bytes.
        // Each group covers 2 sub-blocks: lo nibbles → sub 2*j, hi nibbles → sub 2*j+1.
        // The 5th bit comes from qh[l]: bit (2*j) for lo, bit (2*j+1) for hi.
        for j in 0..4usize {
            let (sc_lo, m_lo) = get_scale_min_k4(2 * j, scales);
            let (sc_hi, m_hi) = get_scale_min_k4(2 * j + 1, scales);
            let d_lo = d * sc_lo as f32;
            let d_hi = d * sc_hi as f32;
            let dm_lo = dmin * m_lo as f32;
            let dm_hi = dmin * m_hi as f32;

            let q_off = j * 32;
            // lo nibbles first (32 values for sub-block 2*j)
            for l in 0..32 {
                let lo = (qs[q_off + l] & 0xF) as u32;
                let bit5 = ((qh[l] >> (2 * j)) & 1) as u32;
                output.push(d_lo * (lo | (bit5 << 4)) as f32 - dm_lo);
            }
            // hi nibbles next (32 values for sub-block 2*j+1)
            for l in 0..32 {
                let hi = (qs[q_off + l] >> 4) as u32;
                let bit5 = ((qh[l] >> (2 * j + 1)) & 1) as u32;
                output.push(d_hi * (hi | (bit5 << 4)) as f32 - dm_hi);
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

        // Canonical GGML Q6_K layout: 128 ql bytes + 64 qh bytes → 256 values.
        // Processed in 2 groups of 128 values. Each group uses 64 ql bytes + 32 qh bytes + 8 scales.
        // Within each group of 128:
        //   positions  0..31: lo nibble of ql[0..32],  qh bits 0-1
        //   positions 32..63: lo nibble of ql[32..64], qh bits 2-3
        //   positions 64..95: hi nibble of ql[0..32],  qh bits 4-5
        //   positions 96..127: hi nibble of ql[32..64], qh bits 6-7
        for n in 0..2usize {
            let ql_off = n * 64;
            let qh_off = n * 32;
            let sc_off = n * 8;
            let mut buf = [0.0f32; 128];

            for l in 0..32usize {
                let is = l / 16; // 0 or 1

                let q1 = (ql[ql_off + l] & 0xF) as i32
                    | (((qh[qh_off + l] >> 0) & 3) as i32) << 4;
                let q2 = (ql[ql_off + l + 32] & 0xF) as i32
                    | (((qh[qh_off + l] >> 2) & 3) as i32) << 4;
                let q3 = (ql[ql_off + l] >> 4) as i32
                    | (((qh[qh_off + l] >> 4) & 3) as i32) << 4;
                let q4 = (ql[ql_off + l + 32] >> 4) as i32
                    | (((qh[qh_off + l] >> 6) & 3) as i32) << 4;

                let sc0 = scales[sc_off + is] as i8 as f32;
                let sc2 = scales[sc_off + is + 2] as i8 as f32;
                let sc4 = scales[sc_off + is + 4] as i8 as f32;
                let sc6 = scales[sc_off + is + 6] as i8 as f32;

                buf[l]      = d * sc0 * (q1 - 32) as f32;
                buf[l + 32] = d * sc2 * (q2 - 32) as f32;
                buf[l + 64] = d * sc4 * (q3 - 32) as f32;
                buf[l + 96] = d * sc6 * (q4 - 32) as f32;
            }
            output.extend_from_slice(&buf);
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
// ─── Q8_1: 40 bytes → 32 f32 values ─────────────────────────
// Block: [f32 d][f32 s][i8 qs × 32] — signed 8-bit with scale + sum

pub fn dequantize_q8_1(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 40; // 4 (f32 d) + 4 (f32 s) + 32 (i8 qs)
    const BLOCK_NUMEL: usize = 32;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        // s (sum) is stored but not needed for dequant — it's used for dot product optimization
        for i in 0..BLOCK_NUMEL {
            let qs = block[8 + i] as i8;
            output.push(qs as f32 * d);
        }
    }
    output
}

// ─── Q4_0: 18 bytes → 32 f32 values ─────────────────────────
// Block: [f16 d][u8 qs × 16] — each byte holds 2 nibbles (4-bit unsigned, offset by 8)

pub fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 18; // 2 (f16 d) + 16 (4-bit × 32 / 2)
    const BLOCK_NUMEL: usize = 32;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let out = &mut output[b * BLOCK_NUMEL..];
        for j in 0..BLOCK_NUMEL / 2 {
            let byte = block[2 + j];
            let x0 = (byte & 0x0F) as i32 - 8;
            let x1 = ((byte >> 4) & 0x0F) as i32 - 8;
            out[j] = x0 as f32 * d;               // low nibble → first half
            out[j + BLOCK_NUMEL / 2] = x1 as f32 * d; // high nibble → second half
        }
    }
    output
}

// ─── Q4_1: 20 bytes → 32 f32 values ─────────────────────────
// Block: [f16 d][f16 m][u8 qs × 16] — 4-bit unsigned + min offset

pub fn dequantize_q4_1(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 20; // 2 (f16 d) + 2 (f16 m) + 16
    const BLOCK_NUMEL: usize = 32;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let out = &mut output[b * BLOCK_NUMEL..];
        for j in 0..BLOCK_NUMEL / 2 {
            let byte = block[4 + j];
            let x0 = (byte & 0x0F) as f32;
            let x1 = ((byte >> 4) & 0x0F) as f32;
            out[j] = x0 * d + m;
            out[j + BLOCK_NUMEL / 2] = x1 * d + m;
        }
    }
    output
}

// ─── Q5_0: 22 bytes → 32 f32 values ─────────────────────────
// Block: [f16 d][u32 qh][u8 qs × 16] — 4-bit nibbles + 1 high bit from qh

pub fn dequantize_q5_0(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 22; // 2 (f16 d) + 4 (qh) + 16 (qs)
    const BLOCK_NUMEL: usize = 32;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let out = &mut output[b * BLOCK_NUMEL..];
        for j in 0..BLOCK_NUMEL / 2 {
            let byte = block[6 + j];
            // Low nibble → first half of block (positions 0..15)
            let xh_0 = ((qh >> j) & 1) as i32;
            let x0 = ((byte & 0x0F) as i32) | (xh_0 << 4);
            out[j] = (x0 as f32 - 16.0) * d;
            // High nibble → second half of block (positions 16..31)
            let xh_1 = ((qh >> (j + 16)) & 1) as i32;
            let x1 = (((byte >> 4) & 0x0F) as i32) | (xh_1 << 4);
            out[j + BLOCK_NUMEL / 2] = (x1 as f32 - 16.0) * d;
        }
    }
    output
}

// ─── Q5_1: 24 bytes → 32 f32 values ─────────────────────────
// Block: [f16 d][f16 m][u32 qh][u8 qs × 16] — 5-bit unsigned + min offset

pub fn dequantize_q5_1(data: &[u8], n_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 24; // 2 (d) + 2 (m) + 4 (qh) + 16 (qs)
    const BLOCK_NUMEL: usize = 32;
    let n_blocks = n_elements / BLOCK_NUMEL;
    let mut output = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block = &data[b * BLOCK_SIZE..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let out = &mut output[b * BLOCK_NUMEL..];
        for j in 0..BLOCK_NUMEL / 2 {
            let byte = block[8 + j];
            // Low nibble → first half, qh bit at position j
            let xh_0 = ((qh >> j) & 1) as f32;
            let x0 = (byte & 0x0F) as f32 + xh_0 * 16.0;
            out[j] = x0 * d + m;
            // High nibble → second half, qh bit at position j+16
            let xh_1 = ((qh >> (j + 16)) & 1) as f32;
            let x1 = ((byte >> 4) & 0x0F) as f32 + xh_1 * 16.0;
            out[j + BLOCK_NUMEL / 2] = x1 * d + m;
        }
    }
    output
}

pub fn dequantize(data: &[u8], dtype: GgmlDType, n_elements: usize) -> Vec<f32> {
    match dtype {
        GgmlDType::F32 => dequantize_f32(data, n_elements),
        GgmlDType::F16 => dequantize_f16(data, n_elements),
        GgmlDType::BF16 => dequantize_bf16(data, n_elements),
        GgmlDType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgmlDType::Q4_1 => dequantize_q4_1(data, n_elements),
        GgmlDType::Q5_0 => dequantize_q5_0(data, n_elements),
        GgmlDType::Q5_1 => dequantize_q5_1(data, n_elements),
        GgmlDType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgmlDType::Q8_1 => dequantize_q8_1(data, n_elements),
        GgmlDType::Q4K => dequantize_q4k(data, n_elements),
        GgmlDType::Q5K => dequantize_q5k(data, n_elements),
        GgmlDType::Q6K => dequantize_q6k(data, n_elements),
        GgmlDType::TQ4_1S => {
            let blocks = tq_kv::weight_compress::bytes_to_blocks(data);
            // Seed is stored in GGUF metadata "tq.weight_compress_seed".
            // Default 42 matches compress_weights() default. Caller must ensure
            // seed consistency — dequantize_tq4_1s() provides seeded variant.
            tq_kv::weight_compress::decompress_weights(&blocks, n_elements, TQ4_1S_SEED.load(std::sync::atomic::Ordering::Relaxed))
        }
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
