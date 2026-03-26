//! Fast Walsh-Hadamard Transform (WHT)
//!
//! TurboQuant'ın ilk adımı: vektörleri rastgele döndürmek.
//! Hadamard matrisi ile çarpma O(n log n) karmaşıklıkta yapılır.
//! Bu, outlier değerleri dağıtarak quantization hatasını minimuma indirir.

/// In-place Fast Walsh-Hadamard Transform.
/// Girdi uzunluğu 2'nin kuvveti olmalı. Değilse en yakın kuvvete pad'lenir.
///
/// Butterfly pattern:
/// ```text
///   h=1:  [a b] -> [a+b, a-b]
///   h=2:  [a b c d] -> [a+c, b+d, a-c, b-d] (paired)
///   ...
/// ```
pub fn fast_wht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "WHT girdi boyutu 2'nin kuvveti olmalı: {}", n);

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Normalize: 1/sqrt(n) ile çarp (ortogonal dönüşüm)
    let scale = 1.0 / (n as f32).sqrt();
    for val in x.iter_mut() {
        *val *= scale;
    }
}

/// WHT'nin tersi. Ortogonal olduğu için kendisi ile aynı (self-inverse).
/// Sadece tekrar çağırmak yeterli.
#[inline]
pub fn inverse_wht(x: &mut [f32]) {
    fast_wht(x);
}

/// Vektörü rastgele işaret değişikliği ile döndür (randomized Hadamard).
/// Bu, outlier'ları dağıtmak için WHT öncesi uygulanan standart teknik.
///
/// `signs[i]` = +1 veya -1, seed'den deterministik üretilir.
pub fn random_sign_flip(x: &mut [f32], signs: &[f32]) {
    debug_assert_eq!(x.len(), signs.len());
    for (val, &sign) in x.iter_mut().zip(signs.iter()) {
        *val *= sign;
    }
}

/// Seed'den deterministik rastgele işaretler üret.
/// Requires `std` feature (uses ChaCha8 RNG).
#[cfg(feature = "std")]
pub fn generate_signs(dim: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dim)
        .map(|_| if rng.gen::<bool>() { 1.0f32 } else { -1.0f32 })
        .collect()
}

/// Pad vector to next power of 2 (if needed).
pub fn pad_to_power_of_two(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n.is_power_of_two() {
        return x.to_vec();
    }
    let next_pow2 = n.next_power_of_two();
    let mut padded = vec![0.0f32; next_pow2];
    padded[..n].copy_from_slice(x);
    padded
}

/// Randomized Hadamard Transform: sign flip + WHT
/// Tam pipeline: x → diag(signs) * x → WHT(x)
/// Requires `std` feature.
#[cfg(feature = "std")]
pub fn randomized_hadamard(x: &mut [f32], seed: u64) {
    let signs = generate_signs(x.len(), seed);
    random_sign_flip(x, &signs);
    fast_wht(x);
}

/// Randomized Hadamard Transform with pre-computed signs.
/// Signs alloc'u kaydeder — hot loop'larda kullanın.
pub fn randomized_hadamard_with_signs(x: &mut [f32], signs: &[f32]) {
    random_sign_flip(x, signs);
    fast_wht(x);
}

/// Inverse Randomized Hadamard: WHT → inverse sign flip
/// Requires `std` feature.
#[cfg(feature = "std")]
pub fn inverse_randomized_hadamard(x: &mut [f32], seed: u64) {
    let signs = generate_signs(x.len(), seed);
    inverse_wht(x);
    random_sign_flip(x, &signs);
}

/// Inverse Randomized Hadamard with pre-computed signs.
pub fn inverse_randomized_hadamard_with_signs(x: &mut [f32], signs: &[f32]) {
    inverse_wht(x);
    random_sign_flip(x, signs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wht_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = original.clone();
        fast_wht(&mut x);
        inverse_wht(&mut x);
        for (a, b) in original.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-5, "WHT roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_randomized_hadamard_roundtrip() {
        let original = vec![1.0, -0.5, 2.3, 0.1, -1.2, 0.7, 3.0, -2.0];
        let mut x = original.clone();
        let seed = 42;
        randomized_hadamard(&mut x, seed);
        inverse_randomized_hadamard(&mut x, seed);
        for (a, b) in original.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "Roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_wht_orthogonality() {
        // WHT preserves L2 norm (Parseval's theorem)
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut transformed = x;
        fast_wht(&mut transformed);
        let norm_after: f32 = transformed.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-5,
            "Norm korunmadı: {} vs {}",
            norm_before,
            norm_after
        );
    }
}
