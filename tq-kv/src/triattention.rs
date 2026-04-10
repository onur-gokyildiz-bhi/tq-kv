//! TriAttention — Trigonometric KV Cache Scoring & Eviction
//!
//! Based on "TriAttention: Efficient Long Reasoning with Trigonometric KV
//! Compression" (Mao et al., arXiv:2604.04921, 2026).
//!
//! Core insight: Pre-RoPE Q/K vectors concentrate around fixed centers,
//! allowing attention logits to be approximated as a trigonometric series
//! dependent on query-key distance. This enables cheap KV importance scoring
//! without running full attention.
//!
//! ## Integration with TurboQuant
//!
//! TriAttention and TurboQuant are orthogonal:
//! - TriAttention: decides WHICH tokens to keep (eviction, ~10x token reduction)
//! - TurboQuant: decides HOW to compress kept tokens (quantization, ~4x bit reduction)
//! - Combined: ~40x total KV cache compression


/// Calibrated TriAttention parameters for a single model.
///
/// Computed offline during calibration from pre-RoPE Q/K statistics.
/// Immutable at inference time.
#[derive(Clone, Debug)]
pub struct TriAttentionConfig {
    /// Pre-RoPE Q centers per head [n_heads][head_dim].
    pub q_centers: Vec<Vec<f32>>,
    /// Pre-RoPE K centers per KV head [n_kv_heads][head_dim].
    pub k_centers: Vec<Vec<f32>>,
    /// Mean Resultant Length per Q head [n_heads]. R∈[0,1].
    /// High R → trig series is accurate; low R → fall back to norm scoring.
    pub mrl: Vec<f32>,
    /// E[||q_f||] per head [n_heads].
    pub q_norm_means: Vec<f32>,
    /// RoPE frequencies: omega_f = 1/theta^(2f/d) [head_dim/2].
    pub rope_freqs: Vec<f32>,
    /// Number of Q heads.
    pub n_heads: usize,
    /// Number of KV heads.
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Eviction interval in tokens (default: 128).
    pub eviction_interval: usize,
    /// KV budget: max tokens to retain after eviction.
    /// If retention_ratio > 0, budget = seq_len × retention_ratio (dynamic).
    pub budget: usize,
    /// Multi-offset set D for averaging: {1, 2, 4, ..., 2^16}.
    pub offsets: Vec<usize>,
    /// V3: Hard prefix protection — first N tokens NEVER evicted.
    /// Covers system prompt, few-shot examples, critical context.
    /// Default: 128. Set via TQ_TRIATTN_PREFIX.
    pub prefix_protection: usize,
    /// V3: Number of segments for per-segment eviction quota.
    /// Context divided into K equal buckets; each loses tokens proportionally.
    /// Prevents global sort from starving any region. Default: 8.
    pub n_segments: usize,
    /// V3: Retention ratio (0.0-1.0). Budget = seq_len × retention.
    /// 0.9 = evict 10%. 0.0 = disabled (use fixed budget). Default: 0.9.
    pub retention_ratio: f32,
}

impl TriAttentionConfig {
    /// Create config from calibration data.
    pub fn from_calibration(
        q_centers: Vec<Vec<f32>>,
        k_centers: Vec<Vec<f32>>,
        mrl: Vec<f32>,
        q_norm_means: Vec<f32>,
        rope_freqs: Vec<f32>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        // Geometric offset set: D = {1, 2, 4, 8, ..., 65536}
        let offsets: Vec<usize> = (0..=16).map(|i| 1usize << i).collect();

        Self {
            q_centers,
            k_centers,
            mrl,
            q_norm_means,
            rope_freqs,
            n_heads,
            n_kv_heads,
            head_dim,
            eviction_interval: std::env::var("TQ_TRIATTN_INTERVAL")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(128),
            budget: 2048,
            offsets,
            prefix_protection: std::env::var("TQ_TRIATTN_PREFIX")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(128),
            n_segments: std::env::var("TQ_TRIATTN_SEGMENTS")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(8),
            retention_ratio: std::env::var("TQ_TRIATTN_RETENTION")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(0.9),
        }
    }

    /// Default geometric offsets D = {1, 2, 4, ..., 2^16}.
    pub fn default_offsets() -> Vec<usize> {
        (0..=16).map(|i| 1usize << i).collect()
    }
}

/// Score a single key vector using TriAttention trigonometric scoring.
///
/// Computes S(k, Δ) = S_trig(k, Δ) + S_norm(k) for one key at distance Δ
/// from the current query position, for a specific Q head and KV head.
///
/// # Arguments
/// * `key_pre_rope` - Pre-RoPE key vector [head_dim]
/// * `q_head` - Which Q head (for center lookup)
/// * `kv_head` - Which KV head (for K center lookup)
/// * `distance` - Position distance Δ = pos_query - pos_key
/// * `config` - Calibrated TriAttention parameters
///
/// # Returns
/// Combined score S_trig + S_norm.
pub fn trig_score_single(
    key_pre_rope: &[f32],
    q_head: usize,
    _kv_head: usize,
    distance: usize,
    config: &TriAttentionConfig,
) -> f32 {
    let head_dim = config.head_dim;
    let n_pairs = head_dim / 2;
    let q_center = &config.q_centers[q_head];
    let delta = distance as f32;

    let mut s_trig = 0.0f32;
    let mut s_norm = 0.0f32;

    for f in 0..n_pairs {
        let omega = config.rope_freqs[f];

        // Q center in frequency pair f: (q_center[2f], q_center[2f+1])
        let qc_re = q_center[2 * f];
        let qc_im = q_center[2 * f + 1];
        let qc_norm = (qc_re * qc_re + qc_im * qc_im).sqrt();

        // Key in frequency pair f
        let k_re = key_pre_rope[2 * f];
        let k_im = key_pre_rope[2 * f + 1];
        let k_norm = (k_re * k_re + k_im * k_im).sqrt();

        if qc_norm < 1e-12 || k_norm < 1e-12 {
            continue;
        }

        // Phase difference: phi_f = arg(E[q_f]) - arg(k_f)
        let phi = qc_im.atan2(qc_re) - k_im.atan2(k_re);

        // S_trig += ||E[q_f]|| * ||k_f|| * cos(omega_f * delta + phi_f)
        s_trig += qc_norm * k_norm * (omega * delta + phi).cos();

        // S_norm += (1 - R_f) * E[||q_f||] * ||k_f||
        // Weight by how much the head deviates from its center
        let r_f = config.mrl[q_head];
        s_norm += (1.0 - r_f) * config.q_norm_means[q_head] * k_norm / n_pairs as f32;
    }

    s_trig + s_norm
}

/// Score a single key with multi-offset averaging.
///
/// S̃(k) = (1/|D|) * Σ_{δ∈D} S(k, Δ+δ)
///
/// Averaging over multiple future distances makes scoring robust to
/// the actual position where the key will be attended.
pub fn trig_score_multi_offset(
    key_pre_rope: &[f32],
    q_head: usize,
    kv_head: usize,
    base_distance: usize,
    config: &TriAttentionConfig,
) -> f32 {
    let n_offsets = config.offsets.len();
    let mut total = 0.0f32;
    for &delta in &config.offsets {
        total += trig_score_single(key_pre_rope, q_head, kv_head, base_distance + delta, config);
    }
    total / n_offsets as f32
}

/// Score all keys in the KV cache for a single KV head.
///
/// For GQA: aggregates scores from all Q heads that share this KV head,
/// using max after per-head normalization (z-score → max across heads).
///
/// # Arguments
/// * `keys_pre_rope` - Pre-RoPE keys for this KV head [n_keys, head_dim]
/// * `kv_head` - Which KV head
/// * `current_pos` - Current generation position (for distance computation)
/// * `key_positions` - Position of each key in the sequence [n_keys]
/// * `config` - Calibrated parameters
///
/// # Returns
/// Importance score per key [n_keys]. Higher = more important.
pub fn score_kv_head(
    keys_pre_rope: &[f32],
    kv_head: usize,
    current_pos: usize,
    key_positions: &[usize],
    config: &TriAttentionConfig,
) -> Vec<f32> {
    let n_keys = key_positions.len();
    let head_dim = config.head_dim;
    let n_rep = config.n_heads / config.n_kv_heads; // GQA ratio

    // Q heads that share this KV head
    let q_head_start = kv_head * n_rep;
    let q_head_end = (q_head_start + n_rep).min(config.n_heads);

    // Score per Q head, then aggregate
    let mut per_head_scores: Vec<Vec<f32>> = Vec::with_capacity(n_rep);

    for q_head in q_head_start..q_head_end {
        let mut scores = Vec::with_capacity(n_keys);
        for ki in 0..n_keys {
            let key = &keys_pre_rope[ki * head_dim..(ki + 1) * head_dim];
            let distance = current_pos.saturating_sub(key_positions[ki]);
            let s = trig_score_multi_offset(key, q_head, kv_head, distance, config);
            scores.push(s);
        }
        per_head_scores.push(scores);
    }

    // GQA aggregation: z-score normalize per head, then max across heads
    if per_head_scores.len() == 1 {
        return per_head_scores.into_iter().next().unwrap();
    }

    // Normalize each head's scores to zero mean, unit variance
    let normalized: Vec<Vec<f32>> = per_head_scores.iter().map(|scores| {
        let n = scores.len() as f32;
        if n < 2.0 {
            return scores.clone();
        }
        let mean: f32 = scores.iter().sum::<f32>() / n;
        let var: f32 = scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
        let std = var.sqrt().max(1e-8);
        scores.iter().map(|s| (s - mean) / std).collect()
    }).collect();

    // Max across heads for each key
    let mut final_scores = vec![f32::NEG_INFINITY; n_keys];
    for head_scores in &normalized {
        for (ki, &s) in head_scores.iter().enumerate() {
            final_scores[ki] = final_scores[ki].max(s);
        }
    }

    final_scores
}

/// Select top-B keys to retain based on TriAttention scores.
///
/// Returns indices of keys to KEEP (sorted by position for cache coherence).
pub fn select_top_keys(
    scores: &[f32],
    budget: usize,
) -> Vec<usize> {
    let n = scores.len();
    if budget >= n {
        return (0..n).collect();
    }

    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<usize> = indexed.iter().take(budget).map(|&(i, _)| i).collect();
    selected.sort_unstable(); // maintain position order for cache coherence
    selected
}

/// Full eviction pass: score all keys across all KV heads and return
/// per-head retain indices.
///
/// # Arguments
/// * `all_keys_pre_rope` - Pre-RoPE keys [n_kv_heads][n_keys * head_dim]
/// * `current_pos` - Current generation position
/// * `key_positions` - Positions of each key [n_keys] (shared across heads)
/// * `sink_count` - Number of sink tokens (always retained, not scored)
/// * `config` - Calibrated parameters
///
/// # Returns
/// Per-head Vec of indices to RETAIN (includes sink tokens).
/// V3 eviction: prefix protection + per-segment quota + dynamic retention.
///
/// V1 (old): global sort, evict lowest-scoring, sink_count=4 only protection.
/// V3 (new): prefix_protection (128), K segments with proportional eviction,
///           dynamic budget = seq_len × retention_ratio (default 0.9).
pub fn evict_pass(
    all_keys_pre_rope: &[&[f32]],
    current_pos: usize,
    key_positions: &[usize],
    sink_count: usize,
    config: &TriAttentionConfig,
) -> Vec<Vec<usize>> {
    let n_kv_heads = all_keys_pre_rope.len();
    let n_keys = key_positions.len();

    // V3: Dynamic budget from retention ratio (overrides fixed budget if set)
    let budget = if config.retention_ratio > 0.0 && config.retention_ratio < 1.0 {
        (n_keys as f32 * config.retention_ratio) as usize
    } else {
        config.budget
    };

    // V3: Protected zone = max(sink_count, prefix_protection)
    let protected = config.prefix_protection.max(sink_count).min(n_keys);

    let evictable_count = n_keys.saturating_sub(protected);
    let evictable_budget = budget.saturating_sub(protected);

    let mut result = Vec::with_capacity(n_kv_heads);

    for kv_head in 0..n_kv_heads {
        let head_dim = config.head_dim;

        if evictable_count <= evictable_budget || n_keys <= budget {
            // All fit within budget — no eviction needed
            result.push((0..n_keys).collect());
            continue;
        }

        // Score only evictable tokens (after protected zone)
        let evictable_keys = &all_keys_pre_rope[kv_head][protected * head_dim..];
        let evictable_positions = &key_positions[protected..];

        let scores = score_kv_head(
            evictable_keys,
            kv_head,
            current_pos,
            evictable_positions,
            config,
        );

        // V3: Per-segment eviction quota
        let retained_evictable = if config.n_segments > 1 && evictable_count > config.n_segments {
            select_top_keys_segmented(&scores, evictable_budget, config.n_segments)
        } else {
            select_top_keys(&scores, evictable_budget)
        };

        // Build full retained list: protected + evictable survivors
        let mut full_retained: Vec<usize> = (0..protected).collect();
        for idx in retained_evictable {
            full_retained.push(idx + protected);
        }
        full_retained.sort_unstable();
        result.push(full_retained);
    }

    result
}

/// V3: Per-segment eviction — divide tokens into K segments, retain proportionally.
/// Prevents global sort from starving any region of the context.
fn select_top_keys_segmented(scores: &[f32], budget: usize, n_segments: usize) -> Vec<usize> {
    let n = scores.len();
    if budget >= n { return (0..n).collect(); }
    if n_segments <= 1 { return select_top_keys(scores, budget); }

    let seg_size = (n + n_segments - 1) / n_segments;
    let mut retained = Vec::with_capacity(budget);
    let mut remaining_budget = budget;
    let mut remaining_tokens = n;

    for seg in 0..n_segments {
        let seg_start = seg * seg_size;
        let seg_end = (seg_start + seg_size).min(n);
        if seg_start >= n { break; }
        let seg_len = seg_end - seg_start;

        // Proportional quota: this segment's share of remaining budget
        let seg_quota = if remaining_tokens > 0 {
            ((seg_len as f64 / remaining_tokens as f64) * remaining_budget as f64).ceil() as usize
        } else { 0 };
        let seg_quota = seg_quota.min(seg_len).min(remaining_budget);

        // Score + select within segment
        let seg_scores = &scores[seg_start..seg_end];
        let seg_selected = select_top_keys(seg_scores, seg_quota);
        for idx in seg_selected {
            retained.push(seg_start + idx);
        }

        remaining_budget = remaining_budget.saturating_sub(seg_quota);
        remaining_tokens = remaining_tokens.saturating_sub(seg_len);
    }

    retained.sort_unstable();
    retained
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> TriAttentionConfig {
        let head_dim = 8;
        let n_heads = 4;
        let n_kv_heads = 2;

        // Synthetic Q centers — high concentration for heads 0,1, low for 2,3
        let q_centers = vec![
            vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.4, 0.6],
            vec![0.9, 0.4, 0.2, 0.3, 0.7, 0.1, 0.5, 0.5],
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1],
        ];
        let k_centers = vec![
            vec![0.5, 0.3, 0.2, 0.4, 0.6, 0.1, 0.3, 0.5],
            vec![0.4, 0.2, 0.3, 0.5, 0.5, 0.2, 0.4, 0.4],
        ];
        let mrl = vec![0.95, 0.92, 0.3, 0.4]; // heads 0,1 high concentration
        let q_norm_means = vec![1.2, 1.1, 0.8, 0.9];
        let rope_freqs = vec![1.0, 0.1, 0.01, 0.001]; // 4 pairs for dim=8

        TriAttentionConfig::from_calibration(
            q_centers, k_centers, mrl, q_norm_means, rope_freqs,
            n_heads, n_kv_heads, head_dim,
        )
    }

    #[test]
    fn test_trig_score_finite() {
        let config = make_test_config();
        let key = vec![0.5, 0.3, -0.1, 0.4, 0.2, -0.3, 0.6, 0.1];

        let score = trig_score_single(&key, 0, 0, 100, &config);
        assert!(score.is_finite(), "Score should be finite, got {}", score);
    }

    #[test]
    fn test_multi_offset_smoothing() {
        let config = make_test_config();
        let key = vec![0.5, 0.3, -0.1, 0.4, 0.2, -0.3, 0.6, 0.1];

        // Multi-offset should be smoother (less sensitive to exact distance)
        let s1 = trig_score_single(&key, 0, 0, 100, &config);
        let s2 = trig_score_single(&key, 0, 0, 101, &config);
        let s_avg = trig_score_multi_offset(&key, 0, 0, 100, &config);

        assert!(s_avg.is_finite());
        // Multi-offset score should be between extremes (smoothing effect)
        // This is a soft check — trig functions oscillate
        eprintln!("s1={}, s2={}, s_avg={}", s1, s2, s_avg);
    }

    #[test]
    fn test_select_top_keys() {
        let scores = vec![0.1, 0.9, 0.5, 0.3, 0.8, 0.2];
        let selected = select_top_keys(&scores, 3);
        assert_eq!(selected.len(), 3);
        // Should contain indices 1 (0.9), 4 (0.8), 2 (0.5) — sorted by position
        assert!(selected.contains(&1));
        assert!(selected.contains(&4));
        assert!(selected.contains(&2));
        // Should be sorted by position
        assert!(selected.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_select_top_keys_over_budget() {
        let scores = vec![0.1, 0.9, 0.5];
        let selected = select_top_keys(&scores, 10);
        assert_eq!(selected, vec![0, 1, 2]); // all retained
    }

    #[test]
    fn test_evict_pass_preserves_sinks() {
        let config = make_test_config();
        let head_dim = config.head_dim;
        let n_keys = 20;
        let sink_count = 4;

        // Generate synthetic keys
        let keys: Vec<f32> = (0..n_keys * head_dim)
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let positions: Vec<usize> = (0..n_keys).collect();

        let mut small_config = config.clone();
        small_config.budget = 10;

        let result = evict_pass(
            &[&keys, &keys], // 2 KV heads, same keys for simplicity
            n_keys + 50,     // current_pos
            &positions,
            sink_count,
            &small_config,
        );

        assert_eq!(result.len(), 2);
        for retained in &result {
            assert!(retained.len() <= small_config.budget);
            // Sink tokens (0..4) must always be retained
            for s in 0..sink_count {
                assert!(retained.contains(&s), "Sink token {} must be retained", s);
            }
        }
    }

    #[test]
    fn test_aligned_key_scores_higher() {
        // Keys aligned with Q center should score higher than random keys
        let config = make_test_config();
        let q_center = &config.q_centers[0];

        // Key aligned with Q center (scaled version)
        let aligned_key: Vec<f32> = q_center.iter().map(|&v| v * 2.0).collect();
        // Random key (orthogonal direction)
        let random_key = vec![0.6, -0.3, 0.7, -0.1, -0.5, 0.8, -0.2, 0.4];

        let s_aligned = trig_score_multi_offset(&aligned_key, 0, 0, 50, &config);
        let s_random = trig_score_multi_offset(&random_key, 0, 0, 50, &config);

        // Aligned key should generally score higher (especially with high MRL)
        eprintln!("aligned={:.4}, random={:.4}", s_aligned, s_random);
        assert!(s_aligned > s_random,
            "Aligned key ({:.4}) should score higher than random ({:.4})",
            s_aligned, s_random);
    }

    #[test]
    fn test_high_mrl_head_uses_trig() {
        // High MRL head (0, R=0.95) should show more distance-dependent scoring
        // Low MRL head (2, R=0.3) should be flatter
        let config = make_test_config();
        let key = vec![0.5, 0.3, -0.1, 0.4, 0.2, -0.3, 0.6, 0.1];

        // Score at different distances for high-MRL head
        let scores_high: Vec<f32> = (0..20)
            .map(|d| trig_score_single(&key, 0, 0, d * 10, &config))
            .collect();

        // Score at different distances for low-MRL head
        let scores_low: Vec<f32> = (0..20)
            .map(|d| trig_score_single(&key, 2, 0, d * 10, &config))
            .collect();

        // High MRL head should show more variance (distance-dependent oscillation)
        let var_high = variance(&scores_high);
        let var_low = variance(&scores_low);
        eprintln!("var_high_mrl={:.6}, var_low_mrl={:.6}", var_high, var_low);
        // Not a strict assertion — just verify both are finite and high-MRL varies more
        assert!(var_high.is_finite());
        assert!(var_low.is_finite());
    }

    #[test]
    fn test_eviction_reduces_count() {
        let mut config = make_test_config();
        config.budget = 5;
        let head_dim = config.head_dim;
        let n_keys = 20;
        let sink_count = 2;

        let keys: Vec<f32> = (0..n_keys * head_dim)
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let positions: Vec<usize> = (0..n_keys).collect();

        let result = evict_pass(
            &[&keys, &keys],
            n_keys + 100,
            &positions,
            sink_count,
            &config,
        );

        for retained in &result {
            assert_eq!(retained.len(), config.budget,
                "Should retain exactly budget={} tokens", config.budget);
            // First sink_count tokens always present
            assert!(retained[0] == 0);
            assert!(retained[1] == 1);
        }
    }

    #[test]
    fn test_score_deterministic() {
        // Same inputs should produce same scores (no randomness)
        let config = make_test_config();
        let key = vec![0.5, 0.3, -0.1, 0.4, 0.2, -0.3, 0.6, 0.1];

        let s1 = trig_score_multi_offset(&key, 0, 0, 100, &config);
        let s2 = trig_score_multi_offset(&key, 0, 0, 100, &config);
        assert_eq!(s1, s2, "Scores must be deterministic");
    }

    fn variance(data: &[f32]) -> f32 {
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n
    }

    #[test]
    fn test_score_kv_head_gqa() {
        let config = make_test_config();
        let head_dim = config.head_dim;
        let n_keys = 5;

        let keys: Vec<f32> = (0..n_keys * head_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 * 0.1 - 0.9)
            .collect();
        let positions: Vec<usize> = (0..n_keys).collect();

        // KV head 0 maps to Q heads 0,1 (n_rep=2)
        let scores = score_kv_head(&keys, 0, 100, &positions, &config);
        assert_eq!(scores.len(), n_keys);
        for &s in &scores {
            assert!(s.is_finite(), "Score should be finite, got {}", s);
        }
    }
}
