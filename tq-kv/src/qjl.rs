//! QJL: Quantized Johnson-Lindenstrauss Error Correction
//!
//! PolarQuant sonrası kalan hatayı 1-bit ile düzeltir.
//! Johnson-Lindenstrauss lemması: rastgele projeksiyon uzaklıkları korur.
//!
//! Algoritma:
//! 1. Hata vektörünü hesapla: e = original - reconstructed
//! 2. Rastgele matris R ile projekte et: p = R @ e
//! 3. Sadece işaretleri sakla: s = sign(p)  → 1-bit per projeksiyon
//! 4. Geri çözerken: correction = alpha * R^T @ s
//!
//! Bu, toplamda sadece 1 ek bit/eleman ile ~30-50% hata azaltması sağlar.

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// QJL sıkıştırma sonucu
#[derive(Clone, Debug)]
pub struct QjlCorrection {
    /// Projeksiyon işaretleri, bit-packed: her u8 = 8 projeksiyon
    pub signs: Vec<u8>,
    /// Projeksiyon boyutu
    pub proj_dim: usize,
    /// Orijinal boyut
    pub orig_dim: usize,
    /// Düzeltme katsayısı (öğrenilmiş veya sabit)
    pub alpha: f32,
    /// Random seed (projeksiyon matrisini yeniden üretmek için)
    pub seed: u64,
}

/// 1-bit projeksiyon üret.
///
/// `error`: orijinal ile reconstruct arasındaki fark vektörü
/// `proj_dim`: projeksiyon boyutu (genellikle orig_dim ile aynı veya yarısı)
/// `seed`: deterministik random seed
pub fn compute(error: &[f32], proj_dim: usize, seed: u64) -> QjlCorrection {
    let orig_dim = error.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Rastgele projeksiyon: R[i] @ error
    // R'nin her satırı ±1/sqrt(proj_dim) değerleri (Rademacher dağılımı)
    let scale = 1.0 / (proj_dim as f32).sqrt();
    let mut projected = Vec::with_capacity(proj_dim);

    for _ in 0..proj_dim {
        let mut dot = 0.0f32;
        for &e in error.iter() {
            let r = if rng.gen::<bool>() { scale } else { -scale };
            dot += r * e;
        }
        projected.push(dot);
    }

    // İşaretleri bit-pack
    let num_bytes = (proj_dim + 7) / 8;
    let mut signs = vec![0u8; num_bytes];
    for (i, &p) in projected.iter().enumerate() {
        if p >= 0.0 {
            signs[i / 8] |= 1 << (i % 8);
        }
    }

    // Alpha: optimal düzeltme katsayısı
    // Teorik olarak alpha = ||error|| * sqrt(2/pi) / sqrt(proj_dim)
    let error_norm: f32 = error.iter().map(|x| x * x).sum::<f32>().sqrt();
    let alpha = error_norm * (2.0 / std::f32::consts::PI).sqrt() / (proj_dim as f32).sqrt();

    QjlCorrection {
        signs,
        proj_dim,
        orig_dim,
        alpha,
        seed,
    }
}

/// QJL düzeltmesini uygula.
///
/// `reconstructed`: PolarQuant'tan gelen yaklaşık vektör (in-place düzeltilir)
pub fn apply(reconstructed: &mut [f32], correction: &QjlCorrection) {
    assert_eq!(reconstructed.len(), correction.orig_dim);

    let mut rng = ChaCha8Rng::seed_from_u64(correction.seed);
    let scale = 1.0 / (correction.proj_dim as f32).sqrt();

    // R^T @ signs: her orijinal boyut için tüm projeksiyonların katkısını topla
    // Bu O(proj_dim * orig_dim) ama orig_dim = head_dim = 128 için hızlı
    //
    // Optimization: R'yi satır satır üreterek bellek tasarrufu
    // R[i,j] aynı seed ile aynı sırada üretilir

    // Önce tüm R elemanlarını üretmemiz gerekiyor (aynı sırada)
    // R shape: (proj_dim, orig_dim)
    // Projeksiyonda: projected[i] = sum_j R[i,j] * error[j]
    // Geri çözmede: correction[j] = sum_i R[i,j] * sign[i]

    // R'yi transpose olarak uygulamamız lazım.
    // R satır-major üretildi, şimdi sütun-major okumamız gerekiyor.
    // En temiz yol: tüm R'yi üret, sonra transpose multiply.
    // head_dim=128, proj_dim=128 → 128*128 = 16K f32 = 64KB. OK.

    let mut r_matrix = Vec::with_capacity(correction.proj_dim * correction.orig_dim);
    for _ in 0..correction.proj_dim {
        for _ in 0..correction.orig_dim {
            let r = if rng.gen::<bool>() { scale } else { -scale };
            r_matrix.push(r);
        }
    }

    // R^T @ signs
    for j in 0..correction.orig_dim {
        let mut sum = 0.0f32;
        for i in 0..correction.proj_dim {
            let sign_bit = (correction.signs[i / 8] >> (i % 8)) & 1;
            let sign_val = if sign_bit == 1 { 1.0f32 } else { -1.0f32 };
            sum += r_matrix[i * correction.orig_dim + j] * sign_val;
        }
        reconstructed[j] += correction.alpha * sum;
    }
}

/// Batch compute: birden fazla hata vektörü için QJL düzeltmesi
pub fn compute_batch(
    errors: &[f32],
    dim: usize,
    proj_dim: usize,
    base_seed: u64,
) -> Vec<QjlCorrection> {
    assert_eq!(errors.len() % dim, 0);
    errors
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, chunk)| {
            // Her vektör için farklı seed (ama deterministik)
            compute(chunk, proj_dim, base_seed.wrapping_add(i as u64))
        })
        .collect()
}

/// Batch apply
pub fn apply_batch(reconstructed: &mut [f32], corrections: &[QjlCorrection], dim: usize) {
    assert_eq!(reconstructed.len() % dim, 0);
    for (chunk, corr) in reconstructed.chunks_exact_mut(dim).zip(corrections.iter()) {
        apply(chunk, corr);
    }
}

/// QJL düzeltmesinin bellek maliyeti (bit cinsinden)
pub fn memory_cost_bits(proj_dim: usize) -> usize {
    // 1-bit per projection + alpha (32-bit) + seed (64-bit)
    proj_dim + 32 + 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qjl_reduces_error() {
        // Rastgele bir hata vektörü oluştur
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let dim = 128;
        let error: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        let error_norm_before: f32 = error.iter().map(|x| x * x).sum::<f32>().sqrt();

        // QJL düzeltmesi hesapla
        let correction = compute(&error, dim, 42);

        // Sıfır vektöre düzeltme uygula (sadece düzeltmeyi görmek için)
        let mut approx = vec![0.0f32; dim];
        apply(&mut approx, &correction);

        // Kalan hata
        let residual: Vec<f32> = error.iter().zip(approx.iter()).map(|(e, a)| e - a).collect();
        let error_norm_after: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Düzeltme hata normunu azaltmalı
        assert!(
            error_norm_after < error_norm_before,
            "QJL hata azaltmadı: {:.4} → {:.4}",
            error_norm_before,
            error_norm_after
        );

        let reduction = 1.0 - error_norm_after / error_norm_before;
        eprintln!(
            "QJL hata azaltma: {:.1}% ({:.4} → {:.4})",
            reduction * 100.0,
            error_norm_before,
            error_norm_after
        );
    }

    #[test]
    fn test_qjl_deterministic() {
        let error = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let c1 = compute(&error, 8, 42);
        let c2 = compute(&error, 8, 42);
        assert_eq!(c1.signs, c2.signs, "Aynı seed farklı sonuç verdi");
        assert_eq!(c1.alpha, c2.alpha);
    }

    #[test]
    fn test_bit_packing() {
        let error = vec![1.0; 16];
        let correction = compute(&error, 16, 42);
        // 16 projeksiyon → 2 byte
        assert_eq!(correction.signs.len(), 2);
    }
}
