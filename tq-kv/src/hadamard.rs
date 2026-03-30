//! Fast Walsh-Hadamard Transform (WHT)
//!
//! The first step of TurboQuant: randomly rotating vectors.
//! Multiplication with the Hadamard matrix is done in O(n log n) complexity.
//! This minimizes quantization error by spreading out outlier values.

/// In-place Fast Walsh-Hadamard Transform.
/// Input length must be a power of 2. If not, it is padded to the nearest power.
///
/// Butterfly pattern:
/// ```text
///   h=1:  [a b] -> [a+b, a-b]
///   h=2:  [a b c d] -> [a+c, b+d, a-c, b-d] (paired)
///   ...
/// ```
pub fn fast_wht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "WHT input size must be a power of 2: {}", n);

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

    // Normalize: multiply by 1/sqrt(n) (orthogonal transform)
    let scale = 1.0 / (n as f32).sqrt();
    for val in x.iter_mut() {
        *val *= scale;
    }
}

/// Inverse of WHT. Since it is orthogonal, it is self-inverse.
/// Simply calling it again is sufficient.
#[inline]
pub fn inverse_wht(x: &mut [f32]) {
    fast_wht(x);
}

/// Rotate vector with random sign flipping (randomized Hadamard).
/// This is the standard technique applied before WHT to spread outliers.
///
/// `signs[i]` = +1 or -1, deterministically generated from seed.
pub fn random_sign_flip(x: &mut [f32], signs: &[f32]) {
    debug_assert_eq!(x.len(), signs.len());
    for (val, &sign) in x.iter_mut().zip(signs.iter()) {
        *val *= sign;
    }
}

/// Generate deterministic random signs from seed.
/// Requires `std` feature (uses ChaCha8 RNG).
#[cfg(feature = "std")]
pub fn generate_signs(dim: usize, seed: u64) -> Vec<f32> {
    let mut signs = vec![0.0f32; dim];
    generate_signs_into(&mut signs, seed);
    signs
}

/// Generate deterministic random signs into pre-allocated buffer.
/// Zero-alloc variant for hot paths (batch QJL).
#[cfg(feature = "std")]
pub fn generate_signs_into(output: &mut [f32], seed: u64) {
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for val in output.iter_mut() {
        *val = if rng.gen::<bool>() { 1.0f32 } else { -1.0f32 };
    }
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
/// Full pipeline: x → diag(signs) * x → WHT(x)
/// Requires `std` feature.
#[cfg(feature = "std")]
pub fn randomized_hadamard(x: &mut [f32], seed: u64) {
    let signs = generate_signs(x.len(), seed);
    random_sign_flip(x, &signs);
    fast_wht(x);
}

/// Randomized Hadamard Transform with pre-computed signs.
/// Saves signs allocation — use in hot loops.
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

/// Apply a dense orthogonal rotation matrix to a vector (matrix-vector multiply).
/// `matrix`: row-major [dim × dim] orthogonal matrix.
/// `x`: vector of length dim (modified in-place).
pub fn apply_rotation(x: &mut [f32], matrix: &[f32]) {
    let dim = x.len();
    debug_assert_eq!(matrix.len(), dim * dim);
    let input = x.to_vec();
    for i in 0..dim {
        let mut sum = 0.0f32;
        let row = &matrix[i * dim..(i + 1) * dim];
        for (j, &val) in input.iter().enumerate() {
            sum += row[j] * val;
        }
        x[i] = sum;
    }
}

/// Apply inverse (transpose) of an orthogonal rotation matrix.
/// For orthogonal R: R^{-1} = R^T.
pub fn apply_inverse_rotation(x: &mut [f32], matrix: &[f32]) {
    let dim = x.len();
    debug_assert_eq!(matrix.len(), dim * dim);
    let input = x.to_vec();
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += matrix[j * dim + i] * input[j]; // column of R = row of R^T
        }
        x[i] = sum;
    }
}

/// Generate a random orthogonal matrix using Cayley transform.
///
/// Cayley(A) = (I - A)(I + A)^{-1} where A is skew-symmetric.
/// This is the SpinQuant parametrization — can be optimized via gradient descent
/// on the skew-symmetric parameters.
///
/// For now: generates a random orthogonal matrix as a starting point.
#[cfg(feature = "std")]
pub fn random_orthogonal(dim: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    // Simple approach: QR decomposition of random matrix
    // We use a simpler method: random Householder reflections
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Start with identity
    let mut q = vec![0.0f32; dim * dim];
    for i in 0..dim {
        q[i * dim + i] = 1.0;
    }

    // Apply dim random Householder reflections: Q = H_1 * H_2 * ... * H_dim
    for k in 0..dim {
        // Random unit vector
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for val in &mut v { *val /= norm; }
        }

        // Apply Householder: Q = (I - 2*v*v^T) * Q
        let mut new_q = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0f32;
                // (I - 2*v*v^T)[i,k] * Q[k,j]
                for m in 0..dim {
                    let h_im = if i == m { 1.0 } else { 0.0 } - 2.0 * v[i] * v[m];
                    sum += h_im * q[m * dim + j];
                }
                new_q[i * dim + j] = sum;
            }
        }
        q = new_q;
    }

    q
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
    fn test_gaussianity_after_rotation() {
        // TheTom found: kurtosis 900.4 → 2.9 after rotation (Gaussian=3.0)
        // Verify our rotation produces near-Gaussian coordinates
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha8Rng;

        let dim = 128;
        let n_vectors = 256;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Generate non-Gaussian vectors (sparse, with outliers)
        let mut raw_coords = Vec::new();
        let mut rotated_coords = Vec::new();

        for _ in 0..n_vectors {
            let mut v: Vec<f32> = (0..dim).map(|_| {
                // Mix of small values + rare outliers (non-Gaussian)
                if rng.gen::<f32>() < 0.05 { rng.gen::<f32>() * 10.0 - 5.0 }
                else { rng.gen::<f32>() * 0.2 - 0.1 }
            }).collect();

            raw_coords.extend_from_slice(&v);

            randomized_hadamard(&mut v, 42);
            rotated_coords.extend_from_slice(&v);
        }

        // Compute kurtosis: E[(x-mu)^4] / (E[(x-mu)^2])^2
        let kurtosis = |data: &[f32]| -> f32 {
            let n = data.len() as f32;
            let mean = data.iter().sum::<f32>() / n;
            let m2 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
            let m4 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f32>() / n;
            if m2 > 0.0 { m4 / (m2 * m2) } else { 0.0 }
        };

        let k_before = kurtosis(&raw_coords);
        let k_after = kurtosis(&rotated_coords);

        eprintln!("Gaussianity verification:");
        eprintln!("  Before rotation: kurtosis = {:.1} (Gaussian=3.0)", k_before);
        eprintln!("  After rotation:  kurtosis = {:.1} (Gaussian=3.0)", k_after);

        // After rotation, kurtosis should be close to 3.0 (Gaussian)
        assert!(k_after > 2.0 && k_after < 5.0,
            "Post-rotation kurtosis {:.1} too far from Gaussian (3.0)", k_after);
        // Before rotation should be non-Gaussian (outlier-heavy)
        assert!(k_before > 5.0,
            "Pre-rotation kurtosis {:.1} should be non-Gaussian", k_before);
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
            "Norm not preserved: {} vs {}",
            norm_before,
            norm_after
        );
    }
}
