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

    /// Apply calibration data to a TurboQuantConfig.
    pub fn apply_to_config(&self, config: &mut tq_kv::TurboQuantConfig) {
        config.channel_scales = Some(self.channel_scales.clone());
        if let Some(cb) = self.codebook_for_bits(config.bits) {
            config.calibrated_codebook = Some(cb);
        }
        if !self.rotation_matrix.is_empty() {
            config.rotation_matrix = Some(self.rotation_matrix.clone());
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

    // 3. PCA rotation matrix (computed on scaled data)
    eprintln!("  PCA rotation...");
    let rotation_matrix = tq_kv::hadamard::calibrate_pca_rotation(&scaled_data, head_dim);

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
