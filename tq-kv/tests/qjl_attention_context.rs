//! QJL Adaptive Context Test
//!
//! Measures ATTENTION SCORE accuracy (not just MSE) at different context lengths.
//! This simulates the real failure mode: softmax amplifies small key errors into
//! large attention distribution shifts. Tests whether QJL helps at long context.

use tq_kv::*;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

fn gaussian_vectors(count: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count * dim).map(|_| {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.5
    }).collect()
}

/// Compute softmax attention distribution for a query against N keys.
/// Returns the attention weights (sum = 1.0).
fn softmax_attention(query: &[f32], keys: &[f32], dim: usize) -> Vec<f32> {
    let n_keys = keys.len() / dim;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut scores: Vec<f32> = keys.chunks_exact(dim)
        .map(|k| {
            query.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() * scale
        })
        .collect();

    // Numerically stable softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
    scores
}

/// KL divergence between two attention distributions.
/// Measures how different the compressed attention is from the original.
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    p.iter().zip(q).map(|(&pi, &qi)| {
        if pi > 1e-10 && qi > 1e-10 {
            (pi as f64) * ((pi as f64) / (qi as f64)).ln()
        } else {
            0.0
        }
    }).sum()
}

/// Top-k accuracy: does the compressed attention pick the same top-k tokens?
fn topk_match(original: &[f32], compressed: &[f32], k: usize) -> f32 {
    let mut orig_indexed: Vec<(usize, f32)> = original.iter().cloned().enumerate().collect();
    let mut comp_indexed: Vec<(usize, f32)> = compressed.iter().cloned().enumerate().collect();
    orig_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    comp_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let orig_topk: Vec<usize> = orig_indexed.iter().take(k).map(|(i, _)| *i).collect();
    let comp_topk: Vec<usize> = comp_indexed.iter().take(k).map(|(i, _)| *i).collect();

    let matches = orig_topk.iter().filter(|i| comp_topk.contains(i)).count();
    matches as f32 / k as f32
}

#[test]
fn qjl_attention_context_scaling() {
    let dim = 128;
    let n_queries = 8; // simulate 8 attention heads

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  QJL ATTENTION ACCURACY vs CONTEXT LENGTH                                    ║");
    eprintln!("║  Metric: KL divergence of attention distribution (lower = better)             ║");
    eprintln!("║  + Top-5 accuracy (does compression pick the same important tokens?)          ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║ {:>8} │ {:>12} {:>12} │ {:>10} {:>10} │ {:>8} ║",
        "Context", "KL(no QJL)", "KL(QJL)", "Top5(noQJL)", "Top5(QJL)", "Winner");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════════════╣");

    let mut qjl_wins_above = None;

    for &n_keys in &[64, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        let keys = gaussian_vectors(n_keys, dim, 42);
        let queries = gaussian_vectors(n_queries, dim, 99);

        // Compress keys: QJL OFF
        let config_off = TurboQuantConfig { bits: 4, use_qjl: false, ..Default::default() };
        let compressed_off = compress_keys(&keys, dim, &config_off);
        let decompressed_off = decompress_keys(&compressed_off, &config_off);

        // Compress keys: QJL ON
        let config_on = TurboQuantConfig { bits: 4, use_qjl: true, ..Default::default() };
        let compressed_on = compress_keys(&keys, dim, &config_on);
        let decompressed_on = decompress_keys(&compressed_on, &config_on);

        // Compute attention accuracy for each query
        let mut total_kl_off = 0.0f64;
        let mut total_kl_on = 0.0f64;
        let mut total_top5_off = 0.0f32;
        let mut total_top5_on = 0.0f32;

        for q_idx in 0..n_queries {
            let query = &queries[q_idx * dim..(q_idx + 1) * dim];

            let attn_orig = softmax_attention(query, &keys, dim);
            let attn_off = softmax_attention(query, &decompressed_off, dim);
            let attn_on = softmax_attention(query, &decompressed_on, dim);

            total_kl_off += kl_divergence(&attn_orig, &attn_off);
            total_kl_on += kl_divergence(&attn_orig, &attn_on);

            let top_k = 5.min(n_keys);
            total_top5_off += topk_match(&attn_orig, &attn_off, top_k);
            total_top5_on += topk_match(&attn_orig, &attn_on, top_k);
        }

        let avg_kl_off = total_kl_off / n_queries as f64;
        let avg_kl_on = total_kl_on / n_queries as f64;
        let avg_top5_off = total_top5_off / n_queries as f32;
        let avg_top5_on = total_top5_on / n_queries as f32;

        let winner = if avg_kl_on < avg_kl_off * 0.95 {
            if qjl_wins_above.is_none() { qjl_wins_above = Some(n_keys); }
            "QJL"
        } else if avg_kl_off < avg_kl_on * 0.95 {
            "no-QJL"
        } else {
            "tie"
        };

        eprintln!("║ {:>8} │ {:>12.6} {:>12.6} │ {:>10.1}% {:>9.1}% │ {:>8} ║",
            n_keys, avg_kl_off, avg_kl_on,
            avg_top5_off * 100.0, avg_top5_on * 100.0,
            winner);
    }

    eprintln!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    match qjl_wins_above {
        Some(threshold) => {
            eprintln!("║  FINDING: QJL becomes beneficial at {} tokens                   ║", threshold);
            eprintln!("║  RECOMMENDATION: QjlMode::Adaptive {{ threshold: {} }}              ║", threshold);
        }
        None => {
            eprintln!("║  FINDING: QJL did not improve attention accuracy at any tested length     ║");
            eprintln!("║  RECOMMENDATION: QjlMode::Off (4-bit MSE-only is sufficient)              ║");
        }
    }
    eprintln!("╚═══════════════════════════════════════════════════════════════════════════════╝");
}

#[test]
fn test_adaptive_qjl_mode() {
    let config = TurboQuantConfig::balanced_adaptive();
    assert!(!config.should_use_qjl(0));
    assert!(!config.should_use_qjl(2048));
    assert!(!config.should_use_qjl(4095));
    assert!(config.should_use_qjl(4096));
    assert!(config.should_use_qjl(16384));
}

#[test]
fn test_qjl_mode_off() {
    let config = TurboQuantConfig::balanced();
    assert!(!config.should_use_qjl(0));
    assert!(!config.should_use_qjl(100000));
}

#[test]
fn test_qjl_mode_on() {
    let mut config = TurboQuantConfig::balanced();
    config.qjl_mode = QjlMode::On;
    assert!(config.should_use_qjl(0));
    assert!(config.should_use_qjl(100000));
}

#[test]
fn test_qjl_legacy_compat() {
    // Legacy use_qjl field should work when qjl_mode is Off
    let config = TurboQuantConfig {
        use_qjl: true,
        qjl_mode: QjlMode::Off,
        ..Default::default()
    };
    assert!(config.should_use_qjl(0)); // legacy use_qjl=true overrides
}
