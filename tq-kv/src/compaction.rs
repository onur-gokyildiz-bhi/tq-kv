//! KV Cache Compaction — reduce token count while preserving attention behavior.
//!
//! Based on "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026).
//! Selects the most important keys, fits attention biases (beta) to preserve
//! softmax partition function, and solves for synthetic values via least squares.
//!
//! This is ORTHOGONAL to TurboQuant quantization:
//! - Compaction reduces NUMBER of tokens (T → t)
//! - TurboQuant reduces BITS per token (16 → 4)
//! - Combined: 50x token × 8x bits = 400x total compression

/// Result of KV cache compaction for a single attention head.
#[derive(Clone, Debug)]
pub struct CompactedHead {
    /// Compacted key vectors [t, head_dim]
    pub keys: Vec<f32>,
    /// Per-key attention bias (added to logits before softmax) [t]
    pub beta: Vec<f32>,
    /// Compacted value vectors (synthetic, fitted via regression) [t, head_dim]
    pub values: Vec<f32>,
    /// Indices of selected keys from original cache
    pub indices: Vec<usize>,
    /// Number of compacted tokens
    pub t: usize,
    /// Head dimension
    pub head_dim: usize,
}

/// Compact a single head's KV cache.
///
/// # Arguments
/// * `keys` - Original key matrix [T * head_dim] (flat, row-major)
/// * `values` - Original value matrix [T * head_dim] (flat, row-major)
/// * `queries` - Reference query vectors [n * head_dim] (from prefill or generated)
/// * `seq_len` - T (number of original tokens)
/// * `n_queries` - n (number of reference queries)
/// * `head_dim` - d
/// * `target_size` - t (desired compacted size)
///
/// Returns CompactedHead with t key-value pairs + beta biases.
pub fn compact_head(
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    seq_len: usize,
    n_queries: usize,
    head_dim: usize,
    target_size: usize,
) -> CompactedHead {
    let t = target_size.min(seq_len);
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    // Phase 1: Compute attention scores [n, T]
    let mut scores = vec![0.0f32; n_queries * seq_len];
    for qi in 0..n_queries {
        for ki in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += queries[qi * head_dim + d] * keys[ki * head_dim + d];
            }
            scores[qi * seq_len + ki] = dot * inv_sqrt_d;
        }
    }

    // Stable softmax per query row
    let mut exp_scores = vec![0.0f32; n_queries * seq_len];
    let mut row_sums = vec![0.0f32; n_queries]; // partition function per query

    for qi in 0..n_queries {
        let row = &scores[qi * seq_len..(qi + 1) * seq_len];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for ki in 0..seq_len {
            let e = (row[ki] - max_val).exp();
            exp_scores[qi * seq_len + ki] = e;
            sum_exp += e;
        }
        row_sums[qi] = sum_exp;
        // Normalize to get attention weights
        for ki in 0..seq_len {
            exp_scores[qi * seq_len + ki] /= sum_exp;
        }
    }

    // Phase 2: Score each key by mean attention weight across queries.
    // Mean is more robust than max — captures keys important to ALL queries, not just one.
    let mut key_scores = vec![0.0f32; seq_len];
    for ki in 0..seq_len {
        let mut sum_w = 0.0f32;
        for qi in 0..n_queries {
            sum_w += exp_scores[qi * seq_len + ki];
        }
        key_scores[ki] = sum_w / n_queries as f32;
    }

    // Select top-t keys
    let mut indexed: Vec<(usize, f32)> = key_scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected: Vec<usize> = indexed.iter().take(t).map(|&(i, _)| i).collect();

    // Extract selected keys
    let mut c1 = vec![0.0f32; t * head_dim];
    for (j, &idx) in selected.iter().enumerate() {
        c1[j * head_dim..(j + 1) * head_dim]
            .copy_from_slice(&keys[idx * head_dim..(idx + 1) * head_dim]);
    }

    // Phase 3: Fit beta via clamped least squares (NNLS)
    // Target: partition function per query (before normalization)
    // We need un-normalized exp scores, so recompute
    let mut exp_raw = vec![0.0f32; n_queries * seq_len];
    let mut targets = vec![0.0f32; n_queries];
    for qi in 0..n_queries {
        let row = &scores[qi * seq_len..(qi + 1) * seq_len];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for ki in 0..seq_len {
            let e = (row[ki] - max_val).exp();
            exp_raw[qi * seq_len + ki] = e;
            sum_exp += e;
        }
        targets[qi] = sum_exp;
    }

    // Design matrix M [n, t]: exp scores for selected keys
    let mut m_mat = vec![0.0f32; n_queries * t];
    for qi in 0..n_queries {
        for (j, &idx) in selected.iter().enumerate() {
            m_mat[qi * t + j] = exp_raw[qi * seq_len + idx];
        }
    }

    // Solve M @ B ≈ target, B >= 0 (clamped least squares)
    let beta_weights = solve_nnls(&m_mat, &targets, n_queries, t);
    let beta: Vec<f32> = beta_weights.iter().map(|&b| b.max(1e-12).ln()).collect();

    // Phase 4: Fit C2 (compacted values) via least squares
    // Compute compacted attention weights with beta
    let mut compact_scores = vec![0.0f32; n_queries * t];
    for qi in 0..n_queries {
        for j in 0..t {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += queries[qi * head_dim + d] * c1[j * head_dim + d];
            }
            compact_scores[qi * t + j] = dot * inv_sqrt_d + beta[j];
        }
    }

    // Softmax on compact scores → X [n, t]
    let mut x_mat = vec![0.0f32; n_queries * t];
    for qi in 0..n_queries {
        let row = &compact_scores[qi * t..(qi + 1) * t];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for j in 0..t {
            let e = (row[j] - max_val).exp();
            x_mat[qi * t + j] = e;
            sum_exp += e;
        }
        for j in 0..t {
            x_mat[qi * t + j] /= sum_exp;
        }
    }

    // Target Y [n, d]: original attention output
    let mut y_mat = vec![0.0f32; n_queries * head_dim];
    for qi in 0..n_queries {
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for ki in 0..seq_len {
                val += exp_scores[qi * seq_len + ki] * values[ki * head_dim + d];
            }
            y_mat[qi * head_dim + d] = val;
        }
    }

    // Solve X @ C2 = Y via ridge regression (Cholesky)
    // Lambda scales with 1/n_queries: more queries = more data = less regularization needed
    let lambda = if n_queries > 1 { 1e-4 / (n_queries as f32) } else { 1e-3 };
    let c2 = solve_ridge(&x_mat, &y_mat, n_queries, t, head_dim, lambda);

    CompactedHead {
        keys: c1,
        beta,
        values: c2,
        indices: selected,
        t,
        head_dim,
    }
}

/// Solve M @ x ≈ b, x >= 0 (clamped least squares).
fn solve_nnls(m: &[f32], b: &[f32], n: usize, t: usize) -> Vec<f32> {
    // M^T M [t, t]
    let mut mtm = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            let mut dot = 0.0f32;
            for qi in 0..n {
                dot += m[qi * t + i] * m[qi * t + j];
            }
            mtm[i * t + j] = dot;
        }
    }
    // Regularize
    for i in 0..t {
        mtm[i * t + i] += 1e-6;
    }
    // M^T b [t]
    let mut mtb = vec![0.0f32; t];
    for i in 0..t {
        let mut dot = 0.0f32;
        for qi in 0..n {
            dot += m[qi * t + i] * b[qi];
        }
        mtb[i] = dot;
    }
    // Solve via Cholesky
    let x = solve_cholesky(&mtm, &mtb, t);
    // Clamp non-negative
    x.iter().map(|&v| v.max(1e-12)).collect()
}

/// Solve X @ C2 = Y via ridge regression.
/// X [n, t], Y [n, d], returns C2 [t, d].
fn solve_ridge(x: &[f32], y: &[f32], n: usize, t: usize, d: usize, lambda: f32) -> Vec<f32> {
    // X^T X [t, t]
    let mut xtx = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            let mut dot = 0.0f32;
            for qi in 0..n {
                dot += x[qi * t + i] * x[qi * t + j];
            }
            xtx[i * t + j] = dot;
        }
    }
    // Add ridge: X^T X + lambda * I
    for i in 0..t {
        xtx[i * t + i] += lambda;
    }
    // X^T Y [t, d]
    let mut xty = vec![0.0f32; t * d];
    for i in 0..t {
        for di in 0..d {
            let mut dot = 0.0f32;
            for qi in 0..n {
                dot += x[qi * t + i] * y[qi * d + di];
            }
            xty[i * d + di] = dot;
        }
    }
    // Solve (X^T X + lambda I) @ C2 = X^T Y column by column
    let mut c2 = vec![0.0f32; t * d];
    for di in 0..d {
        let col: Vec<f32> = (0..t).map(|i| xty[i * d + di]).collect();
        let sol = solve_cholesky(&xtx, &col, t);
        for i in 0..t {
            c2[i * d + di] = sol[i];
        }
    }
    c2
}

/// Solve A @ x = b via Cholesky decomposition (A must be symmetric positive definite).
fn solve_cholesky(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    // L L^T = A (lower triangular)
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let val = a[i * n + i] - sum;
                l[i * n + j] = if val > 0.0 { val.sqrt() } else { 1e-10 };
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    // Forward substitution: L @ y = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..i {
            sum += l[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }
    // Back substitution: L^T @ x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = 0.0f32;
        for j in (i + 1)..n {
            sum += l[j * n + i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_basic() {
        let head_dim = 4;
        let seq_len = 10;
        let n_queries = 3;
        let target = 3;

        // Random-ish keys, values, queries
        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 11 + 5) % 13) as f32 * 0.1 - 0.6)
            .collect();
        let queries: Vec<f32> = (0..n_queries * head_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 * 0.1 - 0.9)
            .collect();

        let result = compact_head(&keys, &values, &queries, seq_len, n_queries, head_dim, target);

        assert_eq!(result.t, target);
        assert_eq!(result.keys.len(), target * head_dim);
        assert_eq!(result.values.len(), target * head_dim);
        assert_eq!(result.beta.len(), target);
        assert_eq!(result.indices.len(), target);

        // Beta should be finite
        for &b in &result.beta {
            assert!(b.is_finite(), "beta should be finite, got {}", b);
        }

        // Values should be finite
        for &v in &result.values {
            assert!(v.is_finite(), "value should be finite, got {}", v);
        }
    }

    #[test]
    fn test_compact_preserves_attention() {
        let head_dim = 8;
        let seq_len = 20;
        let n_queries = 5;
        let target = 10; // 50% compaction

        let keys: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let values: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 11 + 5) % 13) as f32 * 0.1 - 0.6)
            .collect();
        let queries: Vec<f32> = (0..n_queries * head_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 * 0.1 - 0.9)
            .collect();

        let result = compact_head(&keys, &values, &queries, seq_len, n_queries, head_dim, target);
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

        // Compare attention output: original vs compacted
        for qi in 0..n_queries {
            let q = &queries[qi * head_dim..(qi + 1) * head_dim];

            // Original attention output
            let mut orig_scores: Vec<f32> = (0..seq_len).map(|ki| {
                let k = &keys[ki * head_dim..(ki + 1) * head_dim];
                q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_d
            }).collect();
            let max_s = orig_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_s: Vec<f32> = orig_scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e: f32 = exp_s.iter().sum();
            let orig_out: Vec<f32> = (0..head_dim).map(|d| {
                (0..seq_len).map(|ki| exp_s[ki] / sum_e * values[ki * head_dim + d]).sum::<f32>()
            }).collect();

            // Compacted attention output (with beta)
            let mut comp_scores: Vec<f32> = (0..target).map(|j| {
                let k = &result.keys[j * head_dim..(j + 1) * head_dim];
                q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_d + result.beta[j]
            }).collect();
            let max_c = comp_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_c: Vec<f32> = comp_scores.iter().map(|&s| (s - max_c).exp()).collect();
            let sum_c: f32 = exp_c.iter().sum();
            let comp_out: Vec<f32> = (0..head_dim).map(|d| {
                (0..target).map(|j| exp_c[j] / sum_c * result.values[j * head_dim + d]).sum::<f32>()
            }).collect();

            // Cosine similarity between outputs
            let dot: f32 = orig_out.iter().zip(&comp_out).map(|(a, b)| a * b).sum();
            let norm_o: f32 = orig_out.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_c: f32 = comp_out.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos_sim = if norm_o > 1e-10 && norm_c > 1e-10 { dot / (norm_o * norm_c) } else { 0.0 };

            assert!(cos_sim > 0.8, "Compacted attention output should be similar to original, got cos_sim={}", cos_sim);
        }
    }

    #[test]
    fn test_cholesky_identity() {
        // A = I, b = [1,2,3] → x = [1,2,3]
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = solve_cholesky(&a, &b, 3);
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-5, "Cholesky identity: expected {}, got {}", b[i], x[i]);
        }
    }
}
