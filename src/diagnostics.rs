//! Per-layer diagnostic tool for TurboQuant fused attention quality.
//!
//! Designed to investigate Qwen 72B long-context degradation:
//! 80 layers, GQA-8 (8 KV heads, 64 query heads), head_dim=128.
//!
//! The hypothesis: compression error accumulates across layers because each
//! layer's compressed KV cache introduces small errors that compound through
//! the residual stream.
//!
//! Usage from engine code:
//! ```ignore
//! let diag = diagnostics::diagnose_layer(&original_keys, 8, seq_len, 128, &config, layer_idx);
//! diagnostics::print_diagnostic_report(&[diag]);
//! ```

use tq_kv::{
    codebook, hadamard, CompressedKeys, TurboQuantConfig,
};

// ============================================================
// Data types
// ============================================================

/// Per-layer compression quality report.
pub struct LayerDiagnostic {
    /// Transformer layer index (0-based).
    pub layer_idx: usize,
    /// Cosine similarity between compressed and original keys (averaged over heads).
    pub cos_sim_keys: f32,
    /// Cosine similarity between fused and standard attention scores (averaged over heads).
    pub cos_sim_scores: f32,
    /// Maximum absolute error in attention scores across all heads and positions.
    pub max_error: f32,
    /// Number of KV heads in this layer.
    pub n_kv_heads: usize,
    /// Number of cached key vectors per head.
    pub n_cached: usize,
}

/// Summary of error accumulation across all layers.
pub struct AccumulationReport {
    /// Layer indices where cos_sim_keys dropped below the threshold.
    pub flagged_layers_keys: Vec<(usize, f32)>,
    /// Layer indices where cos_sim_scores dropped below the threshold.
    pub flagged_layers_scores: Vec<(usize, f32)>,
    /// Threshold used for flagging.
    pub threshold: f32,
}

// ============================================================
// Core diagnostic functions
// ============================================================

/// Cosine similarity between two flat f32 slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        norm_a += (x as f64) * (x as f64);
        norm_b += (y as f64) * (y as f64);
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-30 {
        return 0.0;
    }
    (dot / denom) as f32
}

/// Compress keys for a single head and return the CompressedKeys struct.
fn compress_head_keys(
    head_keys: &[f32],
    head_dim: usize,
    seq_len: usize,
    config: &TurboQuantConfig,
    signs: &[f32],
) -> CompressedKeys {
    let mut cache = CompressedKeys::new_empty(config.bits, head_dim, config.rotation_seed);
    for s in 0..seq_len {
        let offset = s * head_dim;
        let key_vec = &head_keys[offset..offset + head_dim];
        let (packed, norm) =
            tq_kv::compress_single_key_with_signs(key_vec, head_dim, config, signs);
        cache.append_raw(&packed, norm);
    }
    cache
}

/// Decompress a CompressedKeys back to flat f32 vectors.
fn decompress_head_keys(compressed: &CompressedKeys, config: &TurboQuantConfig) -> Vec<f32> {
    tq_kv::decompress_keys(compressed, config)
}

/// Compute standard (uncompressed) attention scores: q @ k.T / sqrt(d).
/// `query`: single query vector [head_dim].
/// `keys`: flat key matrix [seq_len * head_dim].
/// Returns: [seq_len] scores.
fn standard_attention_scores(query: &[f32], keys: &[f32], head_dim: usize) -> Vec<f32> {
    let seq_len = keys.len() / head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let offset = s * head_dim;
        let k_vec = &keys[offset..offset + head_dim];
        let dot: f32 = query.iter().zip(k_vec.iter()).map(|(&q, &k)| q * k).sum();
        scores.push(dot * scale);
    }
    scores
}

/// Run diagnostics on key vectors for one transformer layer.
///
/// # Arguments
/// * `original_keys` - flat f32 slice: [n_kv_heads * seq_len * head_dim]
/// * `n_kv_heads` - number of KV heads (8 for Qwen 72B)
/// * `seq_len` - sequence length (number of cached positions)
/// * `head_dim` - dimension per head (128 for Qwen 72B)
/// * `config` - TurboQuant compression config
/// * `layer_idx` - transformer layer index (0-based)
///
/// # Returns
/// A `LayerDiagnostic` with per-layer quality metrics.
pub fn diagnose_layer(
    original_keys: &[f32],
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    config: &TurboQuantConfig,
    layer_idx: usize,
) -> LayerDiagnostic {
    assert_eq!(
        original_keys.len(),
        n_kv_heads * seq_len * head_dim,
        "diagnose_layer: original_keys length mismatch: expected {} got {}",
        n_kv_heads * seq_len * head_dim,
        original_keys.len(),
    );

    let signs = hadamard::generate_signs(head_dim, config.rotation_seed);
    let centroids = codebook::get_centroids(config.bits);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let head_size = seq_len * head_dim;
    let mut cos_sim_keys_sum = 0.0f32;
    let mut cos_sim_scores_sum = 0.0f32;
    let mut global_max_error = 0.0f32;

    for h in 0..n_kv_heads {
        let head_offset = h * head_size;
        let head_keys = &original_keys[head_offset..head_offset + head_size];

        // Compress and decompress keys for this head
        let compressed = compress_head_keys(head_keys, head_dim, seq_len, config, &signs);
        let decompressed = decompress_head_keys(&compressed, config);

        // Key cosine similarity (across entire head)
        let cos_keys = cosine_similarity(head_keys, &decompressed);
        cos_sim_keys_sum += cos_keys;

        // Attention score comparison using the last key position as a synthetic query.
        // In real usage, queries come from a different projection, but for diagnostics
        // this tests the score computation pipeline end-to-end.
        let query_pos = seq_len - 1;
        let query_offset = query_pos * head_dim;
        let query = &head_keys[query_offset..query_offset + head_dim];

        // Standard scores: q @ K^T / sqrt(d) using original keys
        let std_scores = standard_attention_scores(query, head_keys, head_dim);

        // Fused scores: pre-rotate query, use compressed key cache
        let rotated_q = tq_kv::pre_rotate_query_with_signs(query, &signs);
        let fused_scores =
            tq_kv::fused_attention_scores(&rotated_q, &compressed, centroids, scale);

        // Score cosine similarity
        let cos_scores = cosine_similarity(&std_scores, &fused_scores);
        cos_sim_scores_sum += cos_scores;

        // Max absolute error in scores
        for (&s, &f) in std_scores.iter().zip(fused_scores.iter()) {
            let err = (s - f).abs();
            if err > global_max_error {
                global_max_error = err;
            }
        }
    }

    LayerDiagnostic {
        layer_idx,
        cos_sim_keys: cos_sim_keys_sum / n_kv_heads as f32,
        cos_sim_scores: cos_sim_scores_sum / n_kv_heads as f32,
        max_error: global_max_error,
        n_kv_heads,
        n_cached: seq_len,
    }
}

// ============================================================
// GQA mapping verification
// ============================================================

/// Verify GQA query-to-KV head mapping for correctness.
///
/// For GQA-N (n_kv_heads KV heads, n_query_heads query heads):
///   n_rep = n_query_heads / n_kv_heads
///   kv_head[qh] = qh / n_rep
///
/// Returns a list of (query_head, expected_kv_head, actual_kv_head) tuples
/// for any mismatched mappings. Empty vec means all correct.
pub fn verify_gqa_mapping(n_query_heads: usize, n_kv_heads: usize) -> Vec<(usize, usize, usize)> {
    assert!(
        n_query_heads >= n_kv_heads,
        "n_query_heads ({}) must be >= n_kv_heads ({})",
        n_query_heads,
        n_kv_heads,
    );
    assert_eq!(
        n_query_heads % n_kv_heads,
        0,
        "n_query_heads ({}) must be divisible by n_kv_heads ({})",
        n_query_heads,
        n_kv_heads,
    );

    let n_rep = n_query_heads / n_kv_heads;
    let mut mismatches = Vec::new();

    for qh in 0..n_query_heads {
        let actual_kv = qh / n_rep;
        // The expected KV head for query head qh: each KV head serves n_rep consecutive query heads.
        // Query heads [0..n_rep) -> KV head 0, [n_rep..2*n_rep) -> KV head 1, etc.
        let expected_kv = qh / n_rep;

        // Bounds check: kv head must be < n_kv_heads
        if actual_kv >= n_kv_heads {
            mismatches.push((qh, expected_kv.min(n_kv_heads - 1), actual_kv));
        } else if actual_kv != expected_kv {
            mismatches.push((qh, expected_kv, actual_kv));
        }
    }

    mismatches
}

// ============================================================
// Error accumulation detection
// ============================================================

/// Detect layers where compression quality drops below a threshold.
///
/// Default threshold: 0.99 cosine similarity.
pub fn detect_error_accumulation(
    diagnostics: &[LayerDiagnostic],
    threshold: f32,
) -> AccumulationReport {
    let mut flagged_keys = Vec::new();
    let mut flagged_scores = Vec::new();

    for diag in diagnostics {
        if diag.cos_sim_keys < threshold {
            flagged_keys.push((diag.layer_idx, diag.cos_sim_keys));
        }
        if diag.cos_sim_scores < threshold {
            flagged_scores.push((diag.layer_idx, diag.cos_sim_scores));
        }
    }

    AccumulationReport {
        flagged_layers_keys: flagged_keys,
        flagged_layers_scores: flagged_scores,
        threshold,
    }
}

// ============================================================
// Reporting
// ============================================================

/// Print a formatted diagnostic report for multiple layers.
pub fn print_diagnostic_report(diagnostics: &[LayerDiagnostic]) {
    if diagnostics.is_empty() {
        eprintln!("[diagnostics] No layer diagnostics to report.");
        return;
    }

    let n_kv_heads = diagnostics[0].n_kv_heads;
    let n_cached = diagnostics[0].n_cached;

    eprintln!("====================================================================");
    eprintln!("  TurboQuant Per-Layer Diagnostic Report");
    eprintln!("====================================================================");
    eprintln!("  Layers:    {}", diagnostics.len());
    eprintln!("  KV heads:  {}", n_kv_heads);
    eprintln!("  Cached:    {} positions", n_cached);
    eprintln!("--------------------------------------------------------------------");
    eprintln!(
        "  {:>5}  {:>12}  {:>12}  {:>12}",
        "Layer", "CosSim(keys)", "CosSim(attn)", "MaxError"
    );
    eprintln!("--------------------------------------------------------------------");

    for diag in diagnostics {
        let key_flag = if diag.cos_sim_keys < 0.99 { " !" } else { "  " };
        let score_flag = if diag.cos_sim_scores < 0.99 { " !" } else { "  " };
        eprintln!(
            "  {:>5}  {:>12.6}{} {:>12.6}{} {:>12.6}",
            diag.layer_idx,
            diag.cos_sim_keys,
            key_flag,
            diag.cos_sim_scores,
            score_flag,
            diag.max_error,
        );
    }

    eprintln!("--------------------------------------------------------------------");

    // Summary statistics
    let avg_keys: f32 =
        diagnostics.iter().map(|d| d.cos_sim_keys).sum::<f32>() / diagnostics.len() as f32;
    let avg_scores: f32 =
        diagnostics.iter().map(|d| d.cos_sim_scores).sum::<f32>() / diagnostics.len() as f32;
    let min_keys = diagnostics
        .iter()
        .map(|d| d.cos_sim_keys)
        .fold(f32::INFINITY, f32::min);
    let min_scores = diagnostics
        .iter()
        .map(|d| d.cos_sim_scores)
        .fold(f32::INFINITY, f32::min);
    let max_err = diagnostics
        .iter()
        .map(|d| d.max_error)
        .fold(0.0f32, f32::max);

    eprintln!("  Avg cos_sim(keys):   {:.6}", avg_keys);
    eprintln!("  Min cos_sim(keys):   {:.6}", min_keys);
    eprintln!("  Avg cos_sim(scores): {:.6}", avg_scores);
    eprintln!("  Min cos_sim(scores): {:.6}", min_scores);
    eprintln!("  Max absolute error:  {:.6}", max_err);

    // Error accumulation detection
    let report = detect_error_accumulation(diagnostics, 0.99);
    if !report.flagged_layers_keys.is_empty() {
        eprintln!();
        eprintln!(
            "  WARNING: {} layers with cos_sim(keys) < {:.2}:",
            report.flagged_layers_keys.len(),
            report.threshold,
        );
        for &(layer, sim) in &report.flagged_layers_keys {
            eprintln!("    Layer {:>3}: {:.6}", layer, sim);
        }
    }
    if !report.flagged_layers_scores.is_empty() {
        eprintln!();
        eprintln!(
            "  WARNING: {} layers with cos_sim(scores) < {:.2}:",
            report.flagged_layers_scores.len(),
            report.threshold,
        );
        for &(layer, sim) in &report.flagged_layers_scores {
            eprintln!("    Layer {:>3}: {:.6}", layer, sim);
        }
    }

    if report.flagged_layers_keys.is_empty() && report.flagged_layers_scores.is_empty() {
        eprintln!();
        eprintln!("  All layers above {:.2} threshold. No degradation detected.", report.threshold);
    }

    eprintln!("====================================================================");
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_gqa_mapping_qwen72b() {
        // Qwen 72B: 64 query heads, 8 KV heads, GQA-8
        let mismatches = verify_gqa_mapping(64, 8);
        assert!(mismatches.is_empty(), "GQA-8 mapping should be correct");
    }

    #[test]
    fn test_gqa_mapping_no_gqa() {
        // MHA: equal heads
        let mismatches = verify_gqa_mapping(32, 32);
        assert!(mismatches.is_empty(), "MHA mapping should be correct");
    }

    #[test]
    fn test_diagnose_layer_basic() {
        let head_dim = 128;
        let n_kv_heads = 8;
        let seq_len = 16;
        let config = TurboQuantConfig::balanced(); // 4-bit

        // Generate random-ish keys (deterministic via simple formula)
        let total = n_kv_heads * seq_len * head_dim;
        let keys: Vec<f32> = (0..total)
            .map(|i| ((i as f32 * 0.0137).sin() * 0.5))
            .collect();

        let diag = diagnose_layer(&keys, n_kv_heads, seq_len, head_dim, &config, 0);

        assert_eq!(diag.layer_idx, 0);
        assert_eq!(diag.n_kv_heads, n_kv_heads);
        assert_eq!(diag.n_cached, seq_len);
        // 4-bit should give decent quality
        assert!(
            diag.cos_sim_keys > 0.90,
            "4-bit cos_sim_keys should be > 0.90, got {}",
            diag.cos_sim_keys,
        );
    }

    #[test]
    fn test_detect_error_accumulation() {
        let diags = vec![
            LayerDiagnostic {
                layer_idx: 0,
                cos_sim_keys: 0.998,
                cos_sim_scores: 0.995,
                max_error: 0.01,
                n_kv_heads: 8,
                n_cached: 100,
            },
            LayerDiagnostic {
                layer_idx: 1,
                cos_sim_keys: 0.985,
                cos_sim_scores: 0.980,
                max_error: 0.05,
                n_kv_heads: 8,
                n_cached: 100,
            },
        ];

        let report = detect_error_accumulation(&diags, 0.99);
        assert_eq!(report.flagged_layers_keys.len(), 1);
        assert_eq!(report.flagged_layers_keys[0].0, 1);
        assert_eq!(report.flagged_layers_scores.len(), 1);
        assert_eq!(report.flagged_layers_scores[0].0, 1);
    }
}
