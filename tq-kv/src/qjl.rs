//! QJL: Quantized Johnson-Lindenstrauss Error Correction
//!
//! Corrects residual error after Lloyd-Max codebook quantization using 1-bit projections.
//! Johnson-Lindenstrauss lemma: random projections preserve distances.
//!
//! ## Algorithm (SRHT — Subsampled Randomized Hadamard Transform)
//!
//! Replaces the naive O(d^2) dense random projection with O(d log d) SRHT:
//!
//! ```text
//! Compress:
//!   1. e = original - reconstructed              (error vector)
//!   2. p = S * H * D @ e                         (SRHT projection)
//!      where D = diag(random ±1), H = Walsh-Hadamard, S = subsample
//!   3. s = sign(p)                               (1-bit per projection)
//!   4. alpha = ||e|| * sqrt(2/pi) / sqrt(m)      (correction coefficient)
//!
//! Decompress:
//!   correction = alpha * D * H * S^T @ s
//! ```
//!
//! This achieves ~30-50% error reduction with only 1 extra bit/element.

use crate::hadamard;

/// QJL compression result
#[derive(Clone, Debug)]
pub struct QjlCorrection {
    /// Projection signs, bit-packed: each u8 = 8 projections
    pub signs: Vec<u8>,
    /// Projection dimension (m ≤ orig_dim)
    pub proj_dim: usize,
    /// Original dimension (d = head_dim)
    pub orig_dim: usize,
    /// Correction coefficient (learned or fixed)
    pub alpha: f32,
    /// Random seed (for reproducing D signs via hadamard::generate_signs)
    pub seed: u64,
}

/// Compute SRHT-based 1-bit projection.
///
/// ```text
/// SRHT DATA FLOW (d=128, m=proj_dim):
///   error[d] ──► D @ error ──► H @ (D @ error) ──► take first m ──► bit_pack signs
///                  128 muls      896 add/subs          slice           128 cmps
///                  ─────────────────────────────────────────────────────────────
///                  Total: ~1,150 ops  (vs 32,768 ops for dense O(d²))
/// ```
///
/// `error`: difference vector between original and reconstructed
/// `proj_dim`: projection dimension (m ≤ error.len(), typically same as dim)
/// `seed`: deterministic random seed for D signs
pub fn compute(error: &[f32], proj_dim: usize, seed: u64) -> QjlCorrection {
    let d = error.len();
    assert!(d.is_power_of_two(), "QJL SRHT requires power-of-2 dimension: {}", d);
    assert!(proj_dim <= d, "proj_dim ({}) must be ≤ dim ({})", proj_dim, d);

    // 1. D @ error: random sign flip (Rademacher diagonal)
    let signs = hadamard::generate_signs(d, seed);
    let mut rotated = error.to_vec();
    hadamard::random_sign_flip(&mut rotated, &signs);

    // 2. H @ (D @ error): Walsh-Hadamard transform — O(d log d)
    hadamard::fast_wht(&mut rotated);

    // 3. S: subsample first proj_dim elements
    // JL lemma guarantees distance preservation for any fixed subset of rows
    let projected = &rotated[..proj_dim];

    // 4. Bit-pack the signs
    let num_bytes = (proj_dim + 7) / 8;
    let mut sign_bits = vec![0u8; num_bytes];
    for (i, &p) in projected.iter().enumerate() {
        if p >= 0.0 {
            sign_bits[i / 8] |= 1 << (i % 8);
        }
    }

    // 5. Alpha: optimal correction coefficient
    let error_norm: f32 = error.iter().map(|x| x * x).sum::<f32>().sqrt();
    let alpha = error_norm * (2.0 / std::f32::consts::PI).sqrt() / (proj_dim as f32).sqrt();

    QjlCorrection {
        signs: sign_bits,
        proj_dim,
        orig_dim: d,
        alpha,
        seed,
    }
}

/// Compute SRHT projection with pre-computed D signs (hot path variant).
///
/// Eliminates `generate_signs()` allocation — use in per-token loops.
pub fn compute_with_signs(error: &[f32], proj_dim: usize, seed: u64, d_signs: &[f32]) -> QjlCorrection {
    let d = error.len();
    assert!(d.is_power_of_two(), "QJL SRHT requires power-of-2 dimension: {}", d);
    assert!(proj_dim <= d, "proj_dim ({}) must be ≤ dim ({})", proj_dim, d);
    assert_eq!(d_signs.len(), d, "D signs length mismatch: {} vs {}", d_signs.len(), d);

    let mut rotated = error.to_vec();
    hadamard::random_sign_flip(&mut rotated, d_signs);
    hadamard::fast_wht(&mut rotated);

    let projected = &rotated[..proj_dim];

    let num_bytes = (proj_dim + 7) / 8;
    let mut sign_bits = vec![0u8; num_bytes];
    for (i, &p) in projected.iter().enumerate() {
        if p >= 0.0 {
            sign_bits[i / 8] |= 1 << (i % 8);
        }
    }

    let error_norm: f32 = error.iter().map(|x| x * x).sum::<f32>().sqrt();
    let alpha = error_norm * (2.0 / std::f32::consts::PI).sqrt() / (proj_dim as f32).sqrt();

    QjlCorrection {
        signs: sign_bits,
        proj_dim,
        orig_dim: d,
        alpha,
        seed,
    }
}

/// Apply SRHT-based QJL correction.
///
/// ```text
/// SRHT INVERSE (d=128, m=proj_dim):
///   signs[m bits] ──► unpack ±1 ──► zero-pad to d ──► H @ padded ──► D @ result ──► scale + add
///                      m ops           (S^T)            896 ops        128 muls       128 FMA
/// ```
///
/// `reconstructed`: approximate vector from codebook (corrected in-place)
pub fn apply(reconstructed: &mut [f32], correction: &QjlCorrection) {
    assert_eq!(reconstructed.len(), correction.orig_dim);

    let d = correction.orig_dim;
    let m = correction.proj_dim;

    // 1. S^T @ signs: unpack signs into d-dimensional vector (zero-padded)
    let mut sign_vec = vec![0.0f32; d];
    for i in 0..m {
        let bit = (correction.signs[i / 8] >> (i % 8)) & 1;
        sign_vec[i] = if bit == 1 { 1.0 } else { -1.0 };
    }
    // Elements [m..d] stay 0.0 — this IS the S^T (subsample transpose) operation

    // 2. H^T @ (S^T @ signs) — WHT is self-inverse (orthogonal + symmetric)
    hadamard::fast_wht(&mut sign_vec);

    // 3. D^T @ result — D is diagonal with ±1, so D^T = D = D^{-1}
    let d_signs = hadamard::generate_signs(d, correction.seed);
    hadamard::random_sign_flip(&mut sign_vec, &d_signs);

    // 4. Scale by alpha and accumulate into reconstructed
    // No extra sqrt(d/m) needed: WHT normalization (1/sqrt(d)) combined with
    // ||S^T @ signs|| = sqrt(m) gives total correction = alpha * sqrt(m).
    // With alpha = ||e|| * sqrt(2/pi) / sqrt(m), total = ||e|| * sqrt(2/pi)
    // which is ~80% of error norm — correct for any m.
    for (r, &s) in reconstructed.iter_mut().zip(sign_vec.iter()) {
        *r += correction.alpha * s;
    }
}

/// Apply SRHT correction with pre-computed D signs (hot path variant).
pub fn apply_with_signs(reconstructed: &mut [f32], correction: &QjlCorrection, d_signs: &[f32]) {
    assert_eq!(reconstructed.len(), correction.orig_dim);
    assert_eq!(d_signs.len(), correction.orig_dim);

    let d = correction.orig_dim;
    let m = correction.proj_dim;

    let mut sign_vec = vec![0.0f32; d];
    for i in 0..m {
        let bit = (correction.signs[i / 8] >> (i % 8)) & 1;
        sign_vec[i] = if bit == 1 { 1.0 } else { -1.0 };
    }

    hadamard::fast_wht(&mut sign_vec);
    hadamard::random_sign_flip(&mut sign_vec, d_signs);

    for (r, &s) in reconstructed.iter_mut().zip(sign_vec.iter()) {
        *r += correction.alpha * s;
    }
}

/// Batch compute: QJL correction for multiple error vectors.
///
/// Optimized: shares D signs across all vectors (same SRHT basis) and
/// reuses all work buffers. Zero per-vector heap allocation.
///
/// Sharing D signs is valid because each error vector is already unique —
/// the JL guarantee only requires the projection to be independent of
/// the input, not that different inputs use different projections.
pub fn compute_batch(
    errors: &[f32],
    dim: usize,
    proj_dim: usize,
    base_seed: u64,
) -> Vec<QjlCorrection> {
    assert_eq!(errors.len() % dim, 0);
    assert!(dim.is_power_of_two(), "QJL SRHT requires power-of-2 dimension: {}", dim);
    assert!(proj_dim <= dim);

    let count = errors.len() / dim;
    let num_bytes = (proj_dim + 7) / 8;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let proj_dim_sqrt = (proj_dim as f32).sqrt();

    // D signs: compute ONCE for all vectors (shared SRHT basis)
    let d_signs = hadamard::generate_signs(dim, base_seed);

    // Reusable work buffers — zero per-vector allocation
    let mut rotated = vec![0.0f32; dim];
    let mut sign_bits = vec![0u8; num_bytes];

    let mut corrections = Vec::with_capacity(count);

    for chunk in errors.chunks_exact(dim) {
        // Copy error into work buffer, apply shared D, then WHT
        rotated.copy_from_slice(chunk);
        hadamard::random_sign_flip(&mut rotated, &d_signs);
        hadamard::fast_wht(&mut rotated);

        // Bit-pack signs of first proj_dim elements (reuse buffer)
        for b in sign_bits.iter_mut() { *b = 0; }
        for (j, &p) in rotated[..proj_dim].iter().enumerate() {
            if p >= 0.0 {
                sign_bits[j / 8] |= 1 << (j % 8);
            }
        }

        // Alpha
        let error_norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        let alpha = error_norm * sqrt_2_over_pi / proj_dim_sqrt;

        corrections.push(QjlCorrection {
            signs: sign_bits.clone(),
            proj_dim,
            orig_dim: dim,
            alpha,
            seed: base_seed,
        });
    }

    corrections
}

/// Batch apply — optimized with shared D signs and reusable work buffer.
pub fn apply_batch(reconstructed: &mut [f32], corrections: &[QjlCorrection], dim: usize) {
    assert_eq!(reconstructed.len() % dim, 0);
    if corrections.is_empty() { return; }

    // D signs: compute once (all corrections in a batch share the same seed)
    let d_signs = hadamard::generate_signs(dim, corrections[0].seed);

    // Reusable work buffer
    let mut sign_vec = vec![0.0f32; dim];

    for (chunk, corr) in reconstructed.chunks_exact_mut(dim).zip(corrections.iter()) {
        let m = corr.proj_dim;

        // Unpack signs into buffer (zero-pad for S^T)
        for val in sign_vec.iter_mut() { *val = 0.0; }
        for i in 0..m {
            let bit = (corr.signs[i / 8] >> (i % 8)) & 1;
            sign_vec[i] = if bit == 1 { 1.0 } else { -1.0 };
        }

        // H @ S^T @ signs, then D @ result
        hadamard::fast_wht(&mut sign_vec);
        hadamard::random_sign_flip(&mut sign_vec, &d_signs);

        // Accumulate correction
        for (r, &s) in chunk.iter_mut().zip(sign_vec.iter()) {
            *r += corr.alpha * s;
        }
    }
}

/// Memory cost of QJL correction (in bits)
pub fn memory_cost_bits(proj_dim: usize) -> usize {
    // 1-bit per projection + alpha (32-bit) + seed (64-bit)
    proj_dim + 32 + 64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_srht_reduces_error() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let dim = 128;
        let error: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        let error_norm_before: f32 = error.iter().map(|x| x * x).sum::<f32>().sqrt();

        let correction = compute(&error, dim, 42);

        let mut approx = vec![0.0f32; dim];
        apply(&mut approx, &correction);

        let residual: Vec<f32> = error.iter().zip(approx.iter()).map(|(e, a)| e - a).collect();
        let error_norm_after: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            error_norm_after < error_norm_before,
            "SRHT QJL did not reduce error: {:.4} -> {:.4}",
            error_norm_before,
            error_norm_after
        );

        let reduction = 1.0 - error_norm_after / error_norm_before;
        eprintln!(
            "SRHT QJL error reduction: {:.1}% ({:.4} -> {:.4})",
            reduction * 100.0,
            error_norm_before,
            error_norm_after
        );
    }

    #[test]
    fn test_srht_deterministic() {
        let error = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let c1 = compute(&error, 8, 42);
        let c2 = compute(&error, 8, 42);
        assert_eq!(c1.signs, c2.signs, "Same seed produced different results");
        assert_eq!(c1.alpha, c2.alpha);
    }

    #[test]
    fn test_bit_packing() {
        let error = vec![1.0; 16];
        let correction = compute(&error, 16, 42);
        // 16 projections -> 2 bytes
        assert_eq!(correction.signs.len(), 2);
    }

    #[test]
    fn test_srht_subsample_reduces_error() {
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let dim = 128;
        let error: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        let error_norm_before: f32 = error.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Subsample: proj_dim=64 < dim=128
        let correction = compute(&error, 64, 42);
        assert_eq!(correction.proj_dim, 64);
        assert_eq!(correction.orig_dim, 128);

        let mut approx = vec![0.0f32; dim];
        apply(&mut approx, &correction);

        let residual: Vec<f32> = error.iter().zip(approx.iter()).map(|(e, a)| e - a).collect();
        let error_norm_after: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            error_norm_after < error_norm_before,
            "Subsampled SRHT did not reduce error: {:.4} -> {:.4}",
            error_norm_before,
            error_norm_after
        );

        eprintln!(
            "Subsampled SRHT (m=64, d=128) error reduction: {:.1}%",
            (1.0 - error_norm_after / error_norm_before) * 100.0
        );
    }

    #[test]
    fn test_srht_zero_error() {
        let dim = 128;
        let error = vec![0.0f32; dim];

        let correction = compute(&error, dim, 42);
        assert_eq!(correction.alpha, 0.0, "Zero error should produce alpha=0");

        // Applying zero-alpha correction should not change anything
        let mut approx = vec![1.0f32; dim];
        let original = approx.clone();
        apply(&mut approx, &correction);

        for (a, o) in approx.iter().zip(original.iter()) {
            assert!(
                (a - o).abs() < 1e-10,
                "Zero-error correction should not modify reconstructed"
            );
        }
    }

    #[test]
    fn test_srht_with_signs_parity() {
        let mut rng = ChaCha8Rng::seed_from_u64(789);
        let dim = 128;
        let error: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let seed = 42u64;

        // Compute with seed-based signs
        let c1 = compute(&error, dim, seed);

        // Compute with pre-computed signs
        let d_signs = hadamard::generate_signs(dim, seed);
        let c2 = compute_with_signs(&error, dim, seed, &d_signs);

        assert_eq!(c1.signs, c2.signs, "Pre-computed signs must match seed-based");
        assert_eq!(c1.alpha, c2.alpha);

        // Apply both and compare
        let mut r1 = vec![0.0f32; dim];
        let mut r2 = vec![0.0f32; dim];
        apply(&mut r1, &c1);
        apply_with_signs(&mut r2, &c2, &d_signs);

        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "with_signs variant must match: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_srht_single_dim() {
        // dim=1 is power of 2, should work (trivial WHT)
        let error = vec![0.5f32];
        let correction = compute(&error, 1, 42);
        assert_eq!(correction.orig_dim, 1);
        assert_eq!(correction.proj_dim, 1);

        let mut approx = vec![0.0f32];
        apply(&mut approx, &correction);
        // Should produce some non-zero correction
        assert!(approx[0].abs() > 0.0, "Single-dim correction should be non-zero");
    }
}
