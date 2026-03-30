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
}

impl CalibrationCollector {
    pub fn new(head_dim: usize, max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples * head_dim),
            head_dim,
            max_samples,
            count: 0,
        }
    }

    /// Collect key vectors from a k tensor [batch, n_kv_heads, seq_len, head_dim].
    /// Only collects from the first KV head to avoid redundancy across GQA groups.
    pub fn collect_from_tensor(&mut self, k_flat: &[f32], n_kv_heads: usize, seq_len: usize, head_dim: usize) {
        if self.count >= self.max_samples {
            return;
        }
        // Collect from first KV head only (representative of the distribution)
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

    // 2. PCA rotation matrix
    eprintln!("  PCA rotation...");
    let rotation_matrix = tq_kv::hadamard::calibrate_pca_rotation(data, head_dim);

    // 3. Calibrated codebooks (2, 3, 4 bit)
    let mut codebook_2bit = None;
    let mut codebook_3bit = None;
    let mut codebook_4bit = None;

    for bits in [2u8, 3, 4] {
        eprintln!("  Codebook {}bit...", bits);
        let cb = tq_kv::calibrate_codebook(data, head_dim, bits, rotation_seed);
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

    CalibrationData {
        model: model_name.to_string(),
        head_dim,
        n_samples,
        channel_scales,
        codebook_2bit,
        codebook_3bit,
        codebook_4bit,
        rotation_matrix,
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
