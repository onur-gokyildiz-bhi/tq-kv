//! Calibration pipeline: collect real KV activations and compute optimal
//! channel scales, codebook centroids, and rotation matrices.
//!
//! Usage: `tq calibrate <model> [--text file.txt]`
//!
//! Produces `~/.tq/models/{name}-{tag}/calibration.json` which is auto-loaded
//! by the engine at startup to improve compression quality.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::hub;

// ---------------------------------------------------------------------------
// Calibration data structures
// ---------------------------------------------------------------------------

/// Per-bitwidth codebook calibration data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodebookCalibration {
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
    pub bits: u8,
}

impl CodebookCalibration {
    /// Convert to tq_kv CalibratedCodebook.
    pub fn to_calibrated_codebook(&self) -> tq_kv::codebook::CalibratedCodebook {
        tq_kv::codebook::CalibratedCodebook {
            centroids: self.centroids.clone(),
            boundaries: self.boundaries.clone(),
            bits: self.bits,
        }
    }
}

/// Full calibration data for a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationData {
    pub model: String,
    pub head_dim: usize,
    pub n_samples: usize,
    pub channel_scales: Vec<f32>,
    pub codebook_2bit: Option<CodebookCalibration>,
    pub codebook_3bit: Option<CodebookCalibration>,
    pub codebook_4bit: Option<CodebookCalibration>,
    pub rotation_matrix: Vec<f32>,
    /// Per-head importance scores (key norm std-dev). Higher = more important head.
    #[serde(default)]
    pub head_importance: Option<Vec<f32>>,
    /// Auto-assigned per-head bit widths from importance scoring.
    #[serde(default)]
    pub auto_head_bits: Option<Vec<u8>>,
    /// Per-channel key bias — mean of post-RoPE key coordinates across calibration tokens.
    /// On GGUF models, weight quantization creates a systematic per-channel bias that
    /// breaks the zero-mean Gaussian assumption of Lloyd-Max codebook.
    /// Subtracting this bias before Hadamard rotation restores the Gaussian assumption.
    #[serde(default)]
    pub key_channel_bias: Option<Vec<f32>>,
    /// Eigenvalue spectrum from PCA (descending order).
    /// SpectralQuant insight: only d_eff ≈ 4 dimensions carry signal.
    #[serde(default)]
    pub eigenvalues: Option<Vec<f32>>,
    /// Effective dimension at 95% variance threshold.
    #[serde(default)]
    pub d_eff: Option<usize>,

    // -- TriAttention fields (arXiv:2604.04921) --

    /// Pre-RoPE query centers per head: E[q_f] in frequency-pair space.
    /// Shape: [n_heads][head_dim]. Stable across positions and contexts.
    #[serde(default)]
    pub tri_q_centers: Option<Vec<Vec<f32>>>,
    /// Pre-RoPE key centers per head: E[k_f] in frequency-pair space.
    #[serde(default)]
    pub tri_k_centers: Option<Vec<Vec<f32>>>,
    /// Mean Resultant Length per head: R_f = ||E[q_f/||q_f||]||.
    /// R→1 means high concentration (trig series accurate), R→0 means dispersed.
    #[serde(default)]
    pub tri_mrl: Option<Vec<f32>>,
    /// Pre-RoPE query norm means per head: E[||q_f||].
    #[serde(default)]
    pub tri_q_norm_means: Option<Vec<f32>>,
    /// RoPE frequencies per frequency pair: omega_f = 1/theta^(2f/d).
    #[serde(default)]
    pub tri_rope_freqs: Option<Vec<f32>>,
    /// Number of query heads (for GQA head mapping).
    #[serde(default)]
    pub tri_n_heads: Option<usize>,
    /// Number of KV heads.
    #[serde(default)]
    pub tri_n_kv_heads: Option<usize>,
}

impl CalibrationData {
    /// Get calibrated codebook for the given bit width.
    pub fn codebook_for_bits(&self, bits: u8) -> Option<tq_kv::codebook::CalibratedCodebook> {
        match bits {
            2 => self.codebook_2bit.as_ref().map(|c| c.to_calibrated_codebook()),
            3 => self.codebook_3bit.as_ref().map(|c| c.to_calibrated_codebook()),
            4 => self.codebook_4bit.as_ref().map(|c| c.to_calibrated_codebook()),
            _ => None,
        }
    }

    /// Build a TriAttentionConfig from calibration data, if tri_* fields are present.
    pub fn build_triattention_config(&self, head_dim: usize) -> Option<tq_kv::triattention::TriAttentionConfig> {
        let qc = self.tri_q_centers.as_ref()?;
        let kc = self.tri_k_centers.as_ref()?;
        let mrl = self.tri_mrl.as_ref()?;
        let qnm = self.tri_q_norm_means.as_ref()?;
        let rf = self.tri_rope_freqs.as_ref()?;
        let n_h = self.tri_n_heads.unwrap_or(0);
        let n_kvh = self.tri_n_kv_heads.unwrap_or(0);
        if n_h == 0 || n_kvh == 0 { return None; }
        Some(tq_kv::triattention::TriAttentionConfig::from_calibration(
            qc.clone(), kc.clone(), mrl.clone(), qnm.clone(), rf.clone(),
            n_h, n_kvh, head_dim,
        ))
    }

    /// Apply calibration data to a TurboQuantConfig.
    ///
    /// Calibration features are currently experimental. Channel_scales enables
    /// SmoothAttention (Q/K outlier migration). Codebook and rotation are
    /// disabled by default — enable with TQ_CAL_CODEBOOK=1 and TQ_CAL_ROTATION=1.
    pub fn apply_to_config(&self, config: &mut tq_kv::TurboQuantConfig) {
        // Key channel bias: Pre-Rotation Centering.
        // Only apply if bias is small relative to typical key norms.
        // Large bias (>1.0 mean) indicates RoPE contamination — skip.
        if let Some(ref bias) = self.key_channel_bias {
            let mean_abs_bias: f32 = bias.iter().map(|b| b.abs()).sum::<f32>() / bias.len() as f32;
            if mean_abs_bias < 1.0 {
                config.key_channel_bias = Some(bias.clone());
                eprintln!("  Pre-Rotation Centering enabled (mean |bias|={:.4})", mean_abs_bias);
            } else {
                eprintln!("  Pre-Rotation Centering SKIPPED (mean |bias|={:.4} — too large, likely RoPE contamination)", mean_abs_bias);
            }
        }
        // Channel scales: reserved for SmoothAttention (currently disabled for Q4_K_M quality)
        // config.channel_scales = Some(self.channel_scales.clone());
        // Calibrated codebook: experimental, disabled by default
        if std::env::var("TQ_CAL_CODEBOOK").ok().map_or(false, |v| v == "1") {
            if let Some(cb) = self.codebook_for_bits(config.bits) {
                config.calibrated_codebook = Some(cb);
            }
        }
        // PCA rotation: experimental, disabled by default
        if std::env::var("TQ_CAL_ROTATION").ok().map_or(false, |v| v == "1") {
            if !self.rotation_matrix.is_empty() {
                config.rotation_matrix = Some(self.rotation_matrix.clone());
            }
        }
        // Spectral d_eff: apply if calibration computed it
        if let Some(d_eff) = self.d_eff {
            config.spectral_d_eff = d_eff;
            eprintln!("  Spectral d_eff={} (QJL will target signal dimensions only)", d_eff);
        }
        // Apply auto-assigned per-head bits from calibration (env var TQ_HEAD_BITS overrides)
        if config.per_head_bits.is_none() {
            if let Some(ref ahb) = self.auto_head_bits {
                config.per_head_bits = Some(ahb.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Calibration collector — shared across layers during prefill
// ---------------------------------------------------------------------------

/// Collects raw post-RoPE key vectors during prefill for calibration.
///
/// Thread-safe: wrapped in Arc<Mutex<>> and shared across layers.
pub struct CalibrationCollector {
    /// Flat f32 key vectors, each of length head_dim
    pub samples: Vec<f32>,
    pub head_dim: usize,
    pub max_samples: usize,
    pub count: usize,
    /// Per-head key norm accumulators for importance scoring
    pub head_norm_sums: Vec<f64>,
    pub head_norm_sq_sums: Vec<f64>,
    pub head_sample_counts: Vec<usize>,
    pub n_kv_heads: usize,
    // -- TriAttention accumulators --
    pub tri_q_sums: Vec<Vec<f64>>,
    pub tri_k_sums: Vec<Vec<f64>>,
    pub tri_q_norm_sums: Vec<f64>,
    pub tri_q_unit_sums: Vec<Vec<f64>>,
    pub tri_q_counts: Vec<usize>,
    pub tri_k_counts: Vec<usize>,
    pub n_heads: usize,
}

impl CalibrationCollector {
    pub fn new(head_dim: usize, max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples * head_dim),
            head_dim,
            max_samples,
            count: 0,
            head_norm_sums: Vec::new(),
            head_norm_sq_sums: Vec::new(),
            head_sample_counts: Vec::new(),
            n_kv_heads: 0,
            tri_q_sums: Vec::new(),
            tri_k_sums: Vec::new(),
            tri_q_norm_sums: Vec::new(),
            tri_q_unit_sums: Vec::new(),
            tri_q_counts: Vec::new(),
            tri_k_counts: Vec::new(),
            n_heads: 0,
        }
    }

    /// Collect key vectors from a k tensor [batch, n_kv_heads, seq_len, head_dim].
    /// Collects samples from first KV head for codebook calibration,
    /// and per-head norm stats from ALL heads for importance scoring.
    pub fn collect_from_tensor(&mut self, k_flat: &[f32], n_kv_heads: usize, seq_len: usize, head_dim: usize) {
        // Initialize per-head accumulators on first call
        if self.n_kv_heads == 0 && n_kv_heads > 0 {
            self.n_kv_heads = n_kv_heads;
            self.head_norm_sums = vec![0.0; n_kv_heads];
            self.head_norm_sq_sums = vec![0.0; n_kv_heads];
            self.head_sample_counts = vec![0; n_kv_heads];
        }

        // Collect per-head norm stats from ALL heads (cheap, always runs)
        for h in 0..n_kv_heads.min(self.n_kv_heads) {
            for s in 0..seq_len {
                let offset = (h * seq_len + s) * head_dim;
                if offset + head_dim <= k_flat.len() {
                    let norm = k_flat[offset..offset + head_dim]
                        .iter()
                        .map(|x| (*x as f64) * (*x as f64))
                        .sum::<f64>()
                        .sqrt();
                    self.head_norm_sums[h] += norm;
                    self.head_norm_sq_sums[h] += norm * norm;
                    self.head_sample_counts[h] += 1;
                }
            }
        }

        // Collect samples from first KV head for codebook calibration
        if self.count >= self.max_samples {
            return;
        }
        let remaining = self.max_samples - self.count;
        let to_collect = seq_len.min(remaining);
        for s in 0..to_collect {
            let offset = s * head_dim; // first head, position s
            if offset + head_dim <= k_flat.len() {
                self.samples.extend_from_slice(&k_flat[offset..offset + head_dim]);
                self.count += 1;
            }
        }
    }

    pub fn is_full(&self) -> bool {
        self.count >= self.max_samples
    }

    /// Collect pre-RoPE Q and K vectors for TriAttention calibration.
    pub fn collect_pre_rope(&mut self, q_flat: &[f32], k_flat: &[f32],
                            n_heads: usize, n_kv_heads: usize,
                            seq_len: usize, head_dim: usize) {
        if self.n_heads == 0 && n_heads > 0 {
            self.n_heads = n_heads;
            self.tri_q_sums = vec![vec![0.0f64; head_dim]; n_heads];
            self.tri_q_norm_sums = vec![0.0f64; n_heads];
            self.tri_q_unit_sums = vec![vec![0.0f64; head_dim]; n_heads];
            self.tri_q_counts = vec![0; n_heads];
        }
        if self.tri_k_sums.is_empty() && n_kv_heads > 0 {
            self.tri_k_sums = vec![vec![0.0f64; head_dim]; n_kv_heads];
            self.tri_k_counts = vec![0; n_kv_heads];
        }
        for h in 0..n_heads.min(self.n_heads) {
            for s in 0..seq_len {
                let offset = (h * seq_len + s) * head_dim;
                if offset + head_dim > q_flat.len() { continue; }
                let q_vec = &q_flat[offset..offset + head_dim];
                for (i, &v) in q_vec.iter().enumerate() {
                    self.tri_q_sums[h][i] += v as f64;
                }
                let norm: f64 = q_vec.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                self.tri_q_norm_sums[h] += norm;
                if norm > 1e-12 {
                    let inv_norm = 1.0 / norm;
                    for (i, &v) in q_vec.iter().enumerate() {
                        self.tri_q_unit_sums[h][i] += (v as f64) * inv_norm;
                    }
                }
                self.tri_q_counts[h] += 1;
            }
        }
        for h in 0..n_kv_heads.min(self.tri_k_sums.len()) {
            for s in 0..seq_len {
                let offset = (h * seq_len + s) * head_dim;
                if offset + head_dim > k_flat.len() { continue; }
                let k_vec = &k_flat[offset..offset + head_dim];
                for (i, &v) in k_vec.iter().enumerate() {
                    self.tri_k_sums[h][i] += v as f64;
                }
                self.tri_k_counts[h] += 1;
            }
        }
    }
}

/// Global TriAttention config, set at startup from calibration data.
/// Read by turbo_generic.rs cache init when TQ_TRIATTN=1.
pub static TRIATTENTION_CONFIG: std::sync::OnceLock<tq_kv::triattention::TriAttentionConfig> =
    std::sync::OnceLock::new();

/// Initialize the global TriAttention config from calibration data.
/// Called from main.rs after loading calibration.
pub fn init_triattention(cal: &CalibrationData, head_dim: usize) -> bool {
    if let Some(mut cfg) = cal.build_triattention_config(head_dim) {
        cfg.budget = std::env::var("TQ_TRIATTN_BUDGET")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        cfg.eviction_interval = std::env::var("TQ_TRIATTN_INTERVAL")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let n_h = cfg.n_heads;
        let n_kvh = cfg.n_kv_heads;
        let mrl_mean: f32 = cfg.mrl.iter().sum::<f32>() / cfg.mrl.len().max(1) as f32;
        if TRIATTENTION_CONFIG.set(cfg).is_ok() {
            eprintln!("  TriAttention enabled: {} Q heads, {} KV heads, MRL={:.3}, budget={}, interval={}",
                n_h, n_kvh, mrl_mean,
                std::env::var("TQ_TRIATTN_BUDGET").unwrap_or("2048".into()),
                std::env::var("TQ_TRIATTN_INTERVAL").unwrap_or("128".into()));
            return true;
        }
    }
    false
}

/// Global calibration collector, set during `tq calibrate` runs.
pub static CALIBRATION_COLLECTOR: std::sync::OnceLock<Arc<Mutex<CalibrationCollector>>> =
    std::sync::OnceLock::new();

/// Initialize the global calibration collector.
pub fn init_collector(head_dim: usize, max_samples: usize) -> Arc<Mutex<CalibrationCollector>> {
    let collector = Arc::new(Mutex::new(CalibrationCollector::new(head_dim, max_samples)));
    let _ = CALIBRATION_COLLECTOR.set(collector.clone());
    collector
}

/// Check if calibration collection is active and collect from k tensor if so.
/// Called from turbo_generic.rs forward_attn().
#[inline]
pub fn maybe_collect(k_flat: &[f32], n_kv_heads: usize, seq_len: usize, head_dim: usize) {
    if let Some(collector) = CALIBRATION_COLLECTOR.get() {
        if let Ok(mut c) = collector.lock() {
            if !c.is_full() {
                c.collect_from_tensor(k_flat, n_kv_heads, seq_len, head_dim);
            }
        }
    }
}

/// Collect pre-RoPE Q and K vectors for TriAttention calibration.
#[inline]
pub fn maybe_collect_pre_rope(
    q_flat: &[f32], k_flat: &[f32],
    n_heads: usize, n_kv_heads: usize,
    seq_len: usize, head_dim: usize,
) {
    if let Some(collector) = CALIBRATION_COLLECTOR.get() {
        if let Ok(mut c) = collector.lock() {
            c.collect_pre_rope(q_flat, k_flat, n_heads, n_kv_heads, seq_len, head_dim);
        }
    }
}

// ---------------------------------------------------------------------------
// TriAttention statistics — pre-RoPE Q/K center extraction
// ---------------------------------------------------------------------------

/// Compute TriAttention calibration statistics from collected pre-RoPE samples.
pub fn compute_triattention_stats(collector: &CalibrationCollector)
    -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>)
{
    let head_dim = collector.head_dim;
    let n_heads = collector.n_heads;
    let n_kv_heads = collector.tri_k_sums.len();

    let mut q_centers = Vec::with_capacity(n_heads);
    let mut mrl = Vec::with_capacity(n_heads);
    let mut q_norm_means = Vec::with_capacity(n_heads);

    for h in 0..n_heads {
        let count = collector.tri_q_counts[h] as f64;
        if count < 1.0 {
            q_centers.push(vec![0.0f32; head_dim]);
            mrl.push(0.0);
            q_norm_means.push(1.0);
            continue;
        }
        let center: Vec<f32> = collector.tri_q_sums[h].iter()
            .map(|&s| (s / count) as f32).collect();
        let mean_norm = collector.tri_q_norm_sums[h] / count;
        let unit_mean_norm: f64 = collector.tri_q_unit_sums[h].iter()
            .map(|s| (s / count) * (s / count)).sum::<f64>().sqrt();
        q_centers.push(center);
        mrl.push(unit_mean_norm.min(1.0) as f32);
        q_norm_means.push(mean_norm as f32);
    }

    let mut k_centers = Vec::with_capacity(n_kv_heads);
    for h in 0..n_kv_heads {
        let count = collector.tri_k_counts[h] as f64;
        if count < 1.0 {
            k_centers.push(vec![0.0f32; head_dim]);
            continue;
        }
        let center: Vec<f32> = collector.tri_k_sums[h].iter()
            .map(|&s| (s / count) as f32).collect();
        k_centers.push(center);
    }

    if !mrl.is_empty() {
        let mean_mrl: f32 = mrl.iter().sum::<f32>() / mrl.len() as f32;
        eprintln!("  TriAttention MRL: mean={:.4} (higher = better trig approx)", mean_mrl);
    }

    (q_centers, k_centers, mrl, q_norm_means)
}

/// Compute RoPE frequency table.
pub fn compute_rope_freqs(head_dim: usize, theta: f32) -> Vec<f32> {
    let n_pairs = head_dim / 2;
    (0..n_pairs).map(|f| 1.0 / theta.powf(2.0 * f as f32 / head_dim as f32)).collect()
}
>>>>>>> 5d79b08 (TriAttention × TurboQuant mix: scoring engine + calibration + eviction infra)

// ---------------------------------------------------------------------------
// Per-head importance scoring
// ---------------------------------------------------------------------------

/// Compute head importance scores based on key norm variance.
/// Higher std-dev = head uses attention more selectively (global) = more important.
pub fn compute_head_importance(collector: &CalibrationCollector) -> Vec<f32> {
    let mut scores = Vec::with_capacity(collector.n_kv_heads);
    for h in 0..collector.n_kv_heads {
        let n = collector.head_sample_counts[h] as f64;
        if n < 2.0 {
            scores.push(1.0);
            continue;
        }
        let mean = collector.head_norm_sums[h] / n;
        let variance = (collector.head_norm_sq_sums[h] / n) - mean * mean;
        scores.push(variance.max(0.0).sqrt() as f32);
    }
    scores
}

/// Auto-assign per-head bit widths based on importance scores.
/// Top `high_frac` fraction of heads (by importance) get `high_bits`,
/// the rest get `low_bits`.
pub fn auto_assign_head_bits(
    scores: &[f32],
    high_bits: u8,
    low_bits: u8,
    high_frac: f32,
) -> Vec<u8> {
    let n = scores.len();
    let n_high = ((n as f32) * high_frac).ceil() as usize;
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut bits = vec![low_bits; n];
    for &(idx, _) in indexed.iter().take(n_high) {
        bits[idx] = high_bits;
    }
    bits
}

// ---------------------------------------------------------------------------
// Pre-Rotation Centering — key channel bias
// ---------------------------------------------------------------------------

/// Compute per-channel mean of key vectors.
///
/// On GGUF Q4_K_M models, quantized weight matrices produce keys with a
/// systematic per-channel bias (non-zero mean). This bias breaks the N(0, σ)
/// assumption that makes Lloyd-Max codebook optimal.
///
/// Subtracting this bias before Hadamard rotation restores the assumption,
/// improving quantization quality. On FP16 models, bias ≈ 0 (no effect).
pub fn compute_key_channel_bias(data: &[f32], head_dim: usize) -> Vec<f32> {
    let count = data.len() / head_dim;
    if count == 0 {
        return vec![0.0; head_dim];
    }

    let mut sums = vec![0.0f64; head_dim];
    for chunk in data.chunks_exact(head_dim) {
        for (i, &v) in chunk.iter().enumerate() {
            sums[i] += v as f64;
        }
    }

    sums.iter().map(|&s| (s / count as f64) as f32).collect()
}

// ---------------------------------------------------------------------------
// Compute calibration from collected samples
// ---------------------------------------------------------------------------

/// Run the full calibration pipeline on collected samples.
pub fn compute_calibration(
    collector: &CalibrationCollector,
    model_name: &str,
    rotation_seed: u64,
) -> CalibrationData {
    let head_dim = collector.head_dim;
    let data = &collector.samples;
    let n_samples = collector.count;

    eprintln!("Computing calibration from {} samples (head_dim={})...", n_samples, head_dim);

    // 0. Per-channel key bias (Pre-Rotation Centering)
    // On GGUF Q4_K_M models, weight quantization creates a systematic per-channel
    // bias in key vectors. Subtracting this before Hadamard rotation restores the
    // zero-mean Gaussian assumption that Lloyd-Max codebook requires.
    eprintln!("  Key channel bias (Pre-Rotation Centering)...");
    let key_channel_bias = compute_key_channel_bias(data, head_dim);
    let bias_magnitude: f32 = key_channel_bias.iter().map(|b| b.abs()).sum::<f32>() / head_dim as f32;
    eprintln!("    Mean |bias|: {:.6} (higher = more weight quant artifacts)", bias_magnitude);

    // 1. Channel scales
    eprintln!("  Channel scales...");
    let channel_scales = tq_kv::calibrate_channel_scales(data, head_dim);

    // 2. Apply channel scales to data before computing rotation + codebook
    // This matches the runtime order: scale → rotate → quantize
    let scaled_data: Vec<f32> = data.chunks_exact(head_dim)
        .flat_map(|chunk| {
            chunk.iter().zip(channel_scales.iter())
                .map(|(&v, &s)| v * s)
        })
        .collect();

    // 3. PCA rotation matrix + eigenvalue spectrum (computed on scaled data)
    eprintln!("  PCA rotation...");
    let (rotation_matrix, eigenvalues) = tq_kv::hadamard::calibrate_pca_rotation(&scaled_data, head_dim);

    // Compute effective dimension (SpectralQuant insight: d_eff ≈ 4 for keys)
    let d_eff_95 = tq_kv::hadamard::compute_d_eff(&eigenvalues, 0.95);
    let d_eff_99 = tq_kv::hadamard::compute_d_eff(&eigenvalues, 0.99);
    eprintln!("  Spectral analysis: d_eff(95%)={}, d_eff(99%)={} out of {} dims",
        d_eff_95, d_eff_99, head_dim);
    if eigenvalues.len() >= 4 {
        eprintln!("  Top eigenvalues: {:.4}, {:.4}, {:.4}, {:.4} (total={:.4})",
            eigenvalues[0], eigenvalues[1], eigenvalues[2], eigenvalues[3],
            eigenvalues.iter().sum::<f32>());
    }

    // 4. Calibrated codebooks (2, 3, 4 bit)
    // Use the PCA rotation matrix for codebook calibration so that
    // the codebook is fitted to the same rotation used at runtime.
    let rot_ref = if rotation_matrix.is_empty() { None } else { Some(rotation_matrix.as_slice()) };
    let mut codebook_2bit = None;
    let mut codebook_3bit = None;
    let mut codebook_4bit = None;

    for bits in [2u8, 3, 4] {
        eprintln!("  Codebook {}bit...", bits);
        let cb = tq_kv::calibrate_codebook_with_rotation(&scaled_data, head_dim, bits, rotation_seed, rot_ref);
        let cal = CodebookCalibration {
            centroids: cb.centroids.clone(),
            boundaries: cb.boundaries.clone(),
            bits,
        };
        match bits {
            2 => codebook_2bit = Some(cal),
            3 => codebook_3bit = Some(cal),
            4 => codebook_4bit = Some(cal),
            _ => {}
        }
    }

    // 5. Per-head importance scoring (if per-head stats were collected)
    let (head_importance, auto_head_bits) = if collector.n_kv_heads > 0 {
        let importance = compute_head_importance(collector);
        eprintln!("  Head importance scores: {:?}", importance);
        // Default: top 50% get 4-bit, rest get 2-bit
        let auto_bits = auto_assign_head_bits(&importance, 4, 2, 0.5);
        eprintln!("  Auto head bits: {:?}", auto_bits);
        (Some(importance), Some(auto_bits))
    } else {
        (None, None)
    };

    // 6. TriAttention pre-RoPE statistics
    let (tri_q_centers, tri_k_centers, tri_mrl, tri_q_norm_means) =
        if collector.n_heads > 0 && !collector.tri_q_sums.is_empty() {
            eprintln!("  TriAttention pre-RoPE statistics...");
            compute_triattention_stats(collector)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
    let rope_freqs = compute_rope_freqs(head_dim, 10000.0);

    CalibrationData {
        model: model_name.to_string(),
        head_dim,
        n_samples,
        channel_scales,
        codebook_2bit,
        codebook_3bit,
        codebook_4bit,
        rotation_matrix,
        head_importance,
        auto_head_bits,
        key_channel_bias: Some(key_channel_bias),
        eigenvalues: if eigenvalues.is_empty() { None } else { Some(eigenvalues) },
        d_eff: Some(d_eff_95),
        tri_q_centers: if tri_q_centers.is_empty() { None } else { Some(tri_q_centers) },
        tri_k_centers: if tri_k_centers.is_empty() { None } else { Some(tri_k_centers) },
        tri_mrl: if tri_mrl.is_empty() { None } else { Some(tri_mrl) },
        tri_q_norm_means: if tri_q_norm_means.is_empty() { None } else { Some(tri_q_norm_means) },
        tri_rope_freqs: Some(rope_freqs),
        tri_n_heads: if collector.n_heads > 0 { Some(collector.n_heads) } else { None },
        tri_n_kv_heads: if collector.tri_k_sums.is_empty() { None } else { Some(collector.tri_k_sums.len()) },
    }
}

// ---------------------------------------------------------------------------
// Save / Load calibration
// ---------------------------------------------------------------------------

/// Get the calibration file path for a model.
pub fn calibration_path(name: &str, tag: &str) -> PathBuf {
    hub::model_dir(name, tag).join("calibration.json")
}

/// Save calibration data to JSON.
pub fn save_calibration(data: &CalibrationData, path: &std::path::Path) -> Result<()> {
    let json = serde_json::to_string_pretty(data)
        .context("Failed to serialize calibration data")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, json)
        .with_context(|| format!("Failed to write calibration to {}", path.display()))?;
    let size = std::fs::metadata(path)?.len();
    eprintln!("Calibration saved: {} ({:.0} KB)", path.display(), size as f64 / 1024.0);
    Ok(())
}

/// Load calibration data from JSON, if it exists.
pub fn load_calibration(name: &str, tag: &str) -> Option<CalibrationData> {
    let path = calibration_path(name, tag);
    let json = std::fs::read_to_string(&path).ok()?;
    let data: CalibrationData = serde_json::from_str(&json).ok()?;
    eprintln!("Loaded calibration: {} ({} samples, head_dim={})", path.display(), data.n_samples, data.head_dim);
    Some(data)
}

/// Try to load calibration for a model query (handles "name:tag" and other formats).
pub fn load_calibration_for_model(model_query: &str) -> Option<CalibrationData> {
    // Try catalog lookup first
    if let Some(entry) = crate::catalog::find(model_query) {
        return load_calibration(entry.name, entry.tag);
    }
    // Try splitting "name:tag"
    if let Some((name, tag)) = model_query.split_once(':') {
        return load_calibration(name, tag);
    }
    None
}
