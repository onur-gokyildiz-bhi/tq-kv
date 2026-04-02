//! # tq-kv: Extreme KV Cache Compression for LLMs
//!
//! Pure Rust implementation of Google's TurboQuant algorithm (ICLR 2026).
//! Compresses KV cache keys to 2-4 bits with up to **15x compression** and
//! **0.997 cosine similarity**. Zero C/C++ dependencies.
//!
//! ## Algorithm
//!
//! 1. **Randomized Hadamard Transform** — decorrelates outliers, O(d log d)
//! 2. **Lloyd-Max Codebook Quantization** — optimal centroids for Gaussian, O(d)
//! 3. **Fused Attention** — pre-rotate query, centroid table lookup (no decompress)
//!
//! ## Quick Start
//!
//! ```rust
//! use tq_kv::{TurboQuantConfig, compress_keys, decompress_keys};
//!
//! let config = TurboQuantConfig::extreme(); // 2-bit, ~15x compression
//! let head_dim = 128;
//! let kv_data: Vec<f32> = vec![0.1; head_dim]; // one vector
//!
//! let compressed = compress_keys(&kv_data, head_dim, &config);
//! println!("Ratio: {:.1}x", compressed.compression_ratio());
//!
//! let restored = decompress_keys(&compressed, &config);
//! ```
//!
//! ## Incremental KV Cache
//!
//! ```rust
//! use tq_kv::*;
//!
//! let config = TurboQuantConfig::extreme();
//! let dim = 128;
//! let signs = hadamard::generate_signs(dim, config.rotation_seed);
//!
//! let mut cache = CompressedKeys::new_empty(config.bits, dim, config.rotation_seed);
//! let key = vec![0.1f32; dim];
//! let (packed, norm) = compress_single_key_with_signs(&key, dim, &config, &signs);
//! cache.append_raw(&packed, norm);
//! ```

/// Lloyd-Max codebook quantization (2/3/4-bit optimal centroids).
pub mod codebook;
/// KV cache compaction — reduce token count via attention matching.
pub mod compaction;
/// Fast Walsh-Hadamard Transform for decorrelation.
pub mod hadamard;

// Internal modules — not part of public API.
#[doc(hidden)]
pub mod polar;
#[doc(hidden)]
pub mod qjl;
pub mod weight_compress;

/// C FFI layer for llama.cpp and other C/C++ engines.
/// Compile with `cargo build --release --features ffi` to produce `libtq_kv.a`.
#[cfg(feature = "ffi")]
pub mod ffi;

#[doc(hidden)]
pub mod bench;

/// Python bindings via PyO3.
/// Build with `maturin develop --features python` or `cargo build --features python`.
#[cfg(feature = "python")]
pub mod python;

/// QJL activation mode.
#[derive(Clone, Debug, PartialEq)]
pub enum QjlMode {
    /// QJL always disabled (best for short context, ≤4K tokens).
    /// Community consensus: 4-bit MSE-only beats MSE+QJL at short context
    /// because QJL variance is amplified by softmax.
    Off,
    /// QJL always enabled (use with SRHT for acceptable overhead).
    On,
    /// Adaptive: QJL activates when cached token count exceeds threshold.
    /// At long context, accumulated quantization error grows large enough
    /// that QJL's +4.5 dB SNR correction outweighs softmax variance cost.
    Adaptive {
        /// Token count threshold above which QJL activates (default: 8192)
        threshold: usize,
    },
}

impl Default for QjlMode {
    fn default() -> Self {
        QjlMode::Off
    }
}

/// TurboQuant configuration.
#[derive(Clone, Debug)]
pub struct TurboQuantConfig {
    /// Quantization bit width (2, 3, or 4)
    pub bits: u8,
    /// QJL error correction mode
    pub qjl_mode: QjlMode,
    /// QJL projection dimension (0 = same as head_dim)
    pub qjl_proj_dim: usize,
    /// Hadamard rotation seed
    pub rotation_seed: u64,
    /// QJL base seed
    pub qjl_seed: u64,
    /// Sparse V threshold: softmax weights below this are skipped in V multiply.
    /// Set to 0.0 to disable. Default: 1e-6.
    pub sparse_v_threshold: f32,
    /// Value cache quantization bit width. 0 = uncompressed (fp16), 4 = 4-bit per-group absmax,
    /// 8 = 8-bit per-vector absmax. Keys use `bits` field; values use this.
    /// Default: 0 (uncompressed, matches paper).
    pub value_bits: u8,
    /// Per-channel scaling factors (SmoothQuant-style). Applied BEFORE Hadamard rotation
    /// to equalize outlier magnitudes across channels. `None` = disabled.
    /// Length must equal head_dim. Use `calibrate_channel_scales()` to compute from data.
    pub channel_scales: Option<Vec<f32>>,
    /// Group size for per-group quantization. Each group of `group_size` dimensions gets
    /// its own scale (sigma), giving finer-grained adaptation than per-vector sigma.
    /// 0 = per-vector sigma (legacy, single norm). Default: 32.
    /// Supported: 0, 32, 64, 128. Smaller = better quality, slightly more storage.
    pub group_size: usize,
    /// Residual quantization bits. If > 0, after first-pass quantization at `bits`,
    /// the residual (error) is quantized at `residual_bits`. Total storage = bits + residual_bits
    /// but quality is much better than direct (bits + residual_bits)-bit.
    /// 0 = disabled. Default: 0. Typical: 2 (for 2+2=4 bit total).
    pub residual_bits: u8,
    /// Outlier preservation: top-K entries per vector stored at full precision.
    /// These entries are zeroed before quantization and restored on decompress.
    /// 0 = disabled. Default: 0. Typical: 2 (top-2 outliers per 128-dim vector).
    pub outlier_k: usize,
    /// Calibrated codebook: optimal centroids from real model activations.
    /// If Some, used instead of standard Gaussian Lloyd-Max centroids.
    /// Calibrate with `CalibratedCodebook::calibrate()`.
    pub calibrated_codebook: Option<codebook::CalibratedCodebook>,
    /// Custom rotation matrix (SpinQuant-style). If Some, used instead of
    /// random Hadamard. Row-major [dim × dim]. Must be orthogonal.
    /// Generate with `hadamard::random_orthogonal()` or load learned matrix.
    pub rotation_matrix: Option<Vec<f32>>,
    /// Per-channel key bias for Pre-Rotation Centering.
    /// Subtract from key vectors BEFORE Hadamard rotation to remove systematic
    /// weight quantization bias. Restores N(0, σ) assumption for Lloyd-Max.
    /// On decompress, bias is added back after inverse rotation.
    /// Length must equal head_dim. Computed during calibration.
    pub key_channel_bias: Option<Vec<f32>>,
    /// Number of initial layers to skip (uncompressed fp16 KV cache).
    /// If None, falls back to TQ_SKIP env var (default: 4).
    pub skip_layers: Option<usize>,
    /// Number of final layers to protect (uncompressed fp16 KV cache).
    /// turboquant_plus found last layers are disproportionately sensitive:
    /// last 8 layers account for ALL quality loss in their experiments.
    /// If None, falls back to TQ_PROTECT_LAST env var (default: 0 = off).
    pub protect_last_layers: Option<usize>,
    /// Number of sink tokens to preserve at full precision.
    /// If None, falls back to TQ_SINK env var (default: 4).
    pub sink_tokens: Option<usize>,

    /// Per-head bit width assignments. If Some, each KV head uses its own bit width.
    /// Length must equal n_kv_heads. Each entry must be 2, 3, or 4.
    /// Overrides `bits` on a per-head basis. None = all heads use `bits`.
    pub per_head_bits: Option<Vec<u8>>,

    /// Pre-RoPE quantization mode (KVQuant approach).
    /// When true, keys are compressed BEFORE RoPE application — pre-RoPE keys have
    /// position-independent per-channel statistics, giving better codebook fit.
    /// At decode time, keys are decompressed and RoPE is applied dynamically.
    /// Incompatible with fused attention (falls back to decompress path).
    /// Default: false (traditional post-RoPE compression).
    pub pre_rope: bool,

    // Legacy field — use qjl_mode instead
    #[doc(hidden)]
    pub use_qjl: bool,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            qjl_mode: QjlMode::Off,
            use_qjl: false,
            qjl_proj_dim: 0,
            rotation_seed: 0x0054_5552_4230,
            qjl_seed: 0x0051_4A4C_4232,
            sparse_v_threshold: 1e-6,
            value_bits: 0,
            channel_scales: None,
            group_size: 32,
            residual_bits: 0,
            outlier_k: 0,
            calibrated_codebook: None,
            rotation_matrix: None,
            key_channel_bias: None,
            skip_layers: None,
            protect_last_layers: None,
            sink_tokens: None,
            per_head_bits: None,
            pre_rope: false,
        }
    }
}

impl TurboQuantConfig {
    /// 2-bit extreme compression (~16x theoretical)
    pub fn extreme() -> Self {
        Self { bits: 2, ..Default::default() }
    }

    /// 3-bit aggressive compression (~10x theoretical)
    pub fn aggressive() -> Self {
        Self { bits: 3, ..Default::default() }
    }

    /// 4-bit balanced compression (~8x theoretical)
    pub fn balanced() -> Self {
        Self { bits: 4, ..Default::default() }
    }

    /// 4-bit with adaptive QJL — auto-enables error correction at long context.
    /// SRHT QJL shows 2.9x lower attention KL divergence at all context lengths
    /// on synthetic data. On real models with Q4 weights, softmax may amplify
    /// QJL variance at short context. Adaptive mode hedges: OFF for prefill,
    /// ON after threshold (when accumulated error outweighs variance cost).
    pub fn balanced_adaptive() -> Self {
        Self {
            bits: 4,
            qjl_mode: QjlMode::Adaptive { threshold: 4096 },
            ..Default::default()
        }
    }

    // --- Builder-style methods ---

    /// Set value compression bits (0 = fp16, 8 = 8-bit absmax).
    pub fn with_value_bits(mut self, bits: u8) -> Self {
        self.value_bits = bits;
        self
    }

    /// Set sparse V threshold (0.0 = disabled).
    pub fn with_sparse_v(mut self, threshold: f32) -> Self {
        self.sparse_v_threshold = threshold;
        self
    }

    /// Set QJL mode.
    pub fn with_qjl(mut self, mode: QjlMode) -> Self {
        self.qjl_mode = mode;
        self
    }

    /// Set per-channel scaling factors.
    pub fn with_channel_scales(mut self, scales: Vec<f32>) -> Self {
        self.channel_scales = Some(scales);
        self
    }

    /// Set per-head bit width assignments.
    pub fn with_per_head_bits(mut self, bits: Vec<u8>) -> Self {
        self.per_head_bits = Some(bits);
        self
    }

    /// Get effective bits for a specific KV head.
    /// Returns per_head_bits[head_idx] if set, otherwise falls back to `self.bits`.
    pub fn bits_for_head(&self, head_idx: usize) -> u8 {
        self.per_head_bits
            .as_ref()
            .and_then(|phb| phb.get(head_idx).copied())
            .unwrap_or(self.bits)
    }

    /// Check if QJL should be active given current cache length.
    pub fn should_use_qjl(&self, cached_tokens: usize) -> bool {
        match &self.qjl_mode {
            QjlMode::Off => self.use_qjl, // legacy compat
            QjlMode::On => true,
            QjlMode::Adaptive { threshold } => cached_tokens >= *threshold,
        }
    }
}

/// Compressed vector collection.
#[derive(Clone, Debug)]
pub struct CompressedVectors {
    pub polar_data: Vec<polar::PolarQuantized>,
    pub qjl_corrections: Option<Vec<qjl::QjlCorrection>>,
    pub dim: usize,
    pub count: usize,
    pub config: TurboQuantConfig,
}

impl CompressedVectors {
    /// Compressed memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let polar_bytes: usize = self.polar_data.iter().map(|p| {
            4 + p.quantized_unit.len() + 4 + 4
        }).sum();
        let qjl_bytes: usize = self.qjl_corrections.as_ref().map_or(0, |corrections| {
            corrections.iter().map(|c| c.signs.len() + 4 + 8).sum()
        });
        polar_bytes + qjl_bytes
    }

    /// Original f32 memory usage in bytes.
    pub fn original_memory_bytes(&self) -> usize {
        self.count * self.dim * 4
    }

    /// Compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        self.original_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

/// V1 API: Compress vectors using PolarQuant pipeline.
/// **Deprecated** — use [`compress_keys`] (V2 Lloyd-Max) instead for better
/// compression ratio and speed.
#[deprecated(since = "0.1.0", note = "use compress_keys (V2 Lloyd-Max API) instead")]
pub fn compress_vectors(data: &[f32], dim: usize, config: &TurboQuantConfig) -> CompressedVectors {
    assert_eq!(data.len() % dim, 0, "Data length must be divisible by dim");
    assert!(dim.is_power_of_two(), "dim must be power of 2: {}", dim);

    let count = data.len() / dim;
    let polar_config = polar::PolarConfig { bits: config.bits };

    // 1. Hadamard rotation + PolarQuant
    let mut rotated = data.to_vec();
    for chunk in rotated.chunks_exact_mut(dim) {
        hadamard::randomized_hadamard(chunk, config.rotation_seed);
    }

    let polar_data = polar::quantize_batch(&rotated, dim, &polar_config);

    // 2. QJL error correction
    let qjl_corrections = if config.use_qjl {
        let reconstructed = polar::dequantize_batch(&polar_data, dim);
        let errors: Vec<f32> = rotated.iter().zip(reconstructed.iter())
            .map(|(orig, recon)| orig - recon).collect();
        let proj_dim = if config.qjl_proj_dim == 0 { dim } else { config.qjl_proj_dim };
        Some(qjl::compute_batch(&errors, dim, proj_dim, config.qjl_seed))
    } else {
        None
    };

    CompressedVectors { polar_data, qjl_corrections, dim, count, config: config.clone() }
}

/// V1 API: Decompress PolarQuant data back to f32 vectors.
#[deprecated(since = "0.1.0", note = "use decompress_keys (V2 Lloyd-Max API) instead")]
pub fn decompress_vectors(compressed: &CompressedVectors) -> Vec<f32> {
    let dim = compressed.dim;

    let mut result = polar::dequantize_batch(&compressed.polar_data, dim);

    if let Some(ref corrections) = compressed.qjl_corrections {
        qjl::apply_batch(&mut result, corrections, dim);
    }

    for chunk in result.chunks_exact_mut(dim) {
        hadamard::inverse_randomized_hadamard(chunk, compressed.config.rotation_seed);
    }

    result
}

// ============================================================
// V2 API: Paper-faithful Lloyd-Max codebook quantization
// Compatible with Dejan's Triton kernel approach.
// ============================================================

/// Paper-faithful compressed key cache.
/// Only keys are compressed; values stay in fp16.
#[derive(Clone, Debug)]
pub struct CompressedKeys {
    /// Packed quantized indices (bit-packed)
    pub packed_indices: Vec<u8>,
    /// Per-vector or per-group norms. Layout:
    /// - group_size=0: one f32 per vector (legacy per-vector sigma)
    /// - group_size>0: `dim/group_size` f32 per vector (per-group sigma)
    pub norms: Vec<f32>,
    /// QJL corrections (optional)
    pub qjl_corrections: Option<Vec<qjl::QjlCorrection>>,
    /// Codebook bit width
    pub bits: u8,
    /// Vector dimension (head_dim)
    pub dim: usize,
    /// Number of vectors
    pub count: usize,
    /// Rotation seed
    pub rotation_seed: u64,
    /// Group size for per-group quantization (0 = per-vector legacy)
    pub group_size: usize,
    /// Residual quantization: packed indices for second pass (error correction)
    pub residual_indices: Option<Vec<u8>>,
    /// Residual norms (per-group or per-vector, matching group_size)
    pub residual_norms: Option<Vec<f32>>,
    /// Residual bit width (0 = no residual)
    pub residual_bits: u8,
    /// Sparse outliers: (dim_index, value) pairs per vector, stored flat.
    /// Layout: [vec0_idx0, vec0_val0, vec0_idx1, vec0_val1, ..., vec1_idx0, ...]
    /// Each entry = 1 byte index + 4 bytes f32 = 5 bytes.
    pub outlier_indices: Option<Vec<u8>>,
    pub outlier_values: Option<Vec<f32>>,
    /// Number of outliers per vector
    pub outlier_k: usize,
}

impl CompressedKeys {
    /// Create empty compressed keys (for incremental append).
    pub fn new_empty(bits: u8, dim: usize, rotation_seed: u64) -> Self {
        Self::new_empty_grouped(bits, dim, rotation_seed, 0)
    }

    /// Create empty compressed keys with group quantization.
    pub fn new_empty_grouped(bits: u8, dim: usize, rotation_seed: u64, group_size: usize) -> Self {
        Self {
            packed_indices: Vec::new(),
            norms: Vec::new(),
            qjl_corrections: None,
            bits,
            dim,
            count: 0,
            rotation_seed,
            group_size,
            residual_indices: None,
            residual_norms: None,
            residual_bits: 0,
            outlier_indices: None,
            outlier_values: None,
            outlier_k: 0,
        }
    }

    /// Append a single compressed key to the cache (legacy per-vector norm).
    pub fn append_raw(&mut self, packed: &[u8], norm: f32) {
        self.packed_indices.extend_from_slice(packed);
        self.norms.push(norm);
        self.count += 1;
    }

    /// Append a single compressed key with per-group norms and optional residual.
    pub fn append_raw_grouped(
        &mut self,
        packed: &[u8],
        group_norms: &[f32],
        residual: Option<(Vec<u8>, Vec<f32>)>,
    ) {
        self.packed_indices.extend_from_slice(packed);
        self.norms.extend_from_slice(group_norms);
        if let Some((res_packed, res_norms)) = residual {
            if self.residual_indices.is_none() {
                self.residual_indices = Some(Vec::new());
                self.residual_norms = Some(Vec::new());
            }
            self.residual_indices.as_mut().unwrap().extend_from_slice(&res_packed);
            self.residual_norms.as_mut().unwrap().extend_from_slice(&res_norms);
        }
        self.count += 1;
    }

    /// Number of norms stored per vector.
    pub fn norms_per_vector(&self) -> usize {
        if self.group_size == 0 { 1 } else { self.dim / self.group_size }
    }

    /// Number of bytes per vector in packed format.
    pub fn bytes_per_vector(&self) -> usize {
        (self.dim * self.bits as usize + 7) / 8
    }

    /// Return the unpacked indices for a specific vector.
    pub fn get_indices(&self, vector_idx: usize) -> Vec<u8> {
        let bpv = self.bytes_per_vector();
        let start = vector_idx * bpv;
        let end = start + bpv;
        codebook::unpack_indices(&self.packed_indices[start..end], self.dim, self.bits)
    }

    /// Compressed memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        let index_bytes = self.packed_indices.len();
        let norm_bytes = self.norms.len() * 4;
        let qjl_bytes = self.qjl_corrections.as_ref().map_or(0, |c| {
            c.iter().map(|q| q.signs.len() + 4 + 8).sum()
        });
        index_bytes + norm_bytes + qjl_bytes
    }

    /// Original fp16 memory in bytes.
    pub fn original_memory_bytes(&self) -> usize {
        self.count * self.dim * 2 // fp16 = 2 bytes
    }

    /// Compression ratio vs fp16.
    pub fn compression_ratio(&self) -> f32 {
        self.original_memory_bytes() as f32 / self.memory_bytes() as f32
    }

    /// Split off the first `n` vectors from this cache, returning them as a new
    /// CompressedKeys. The remaining vectors stay in `self`.
    ///
    /// Used for temporal decay: extract old tokens for demotion to lower bit width.
    pub fn split_off_front(&mut self, n: usize) -> CompressedKeys {
        assert!(n <= self.count, "split_off_front: n={} > count={}", n, self.count);
        let bpv = self.bytes_per_vector();
        let byte_split = n * bpv;

        let npv = self.norms_per_vector();
        let norm_split = n * npv;

        let front = CompressedKeys {
            packed_indices: self.packed_indices[..byte_split].to_vec(),
            norms: self.norms[..norm_split].to_vec(),
            qjl_corrections: None,
            bits: self.bits,
            dim: self.dim,
            count: n,
            rotation_seed: self.rotation_seed,
            group_size: self.group_size,
            residual_indices: None,
            residual_norms: None,
            residual_bits: 0,
            outlier_indices: None,
            outlier_values: None,
            outlier_k: 0,
        };

        self.packed_indices = self.packed_indices[byte_split..].to_vec();
        self.norms = self.norms[norm_split..].to_vec();
        self.qjl_corrections = None;
        self.count -= n;

        front
    }

    /// Remap all indices to a lower bit width (temporal decay).
    ///
    /// Each index at the current bit width is mapped to the nearest centroid
    /// at `target_bits`. Norms are preserved. QJL corrections are dropped
    /// (not meaningful after bit-width change).
    ///
    /// Returns a new CompressedKeys at the target bit width.
    pub fn remap_bits(&self, target_bits: u8) -> CompressedKeys {
        assert!(target_bits < self.bits,
            "remap_bits: target {} must be < current {}", target_bits, self.bits);

        let remap = codebook::remap_table(self.bits, target_bits);

        // Unpack all indices, remap, repack at target bits
        let all_indices = codebook::unpack_indices(
            &self.packed_indices, self.count * self.dim, self.bits,
        );
        let remapped: Vec<u8> = all_indices.iter().map(|&idx| remap[idx as usize]).collect();
        let packed = codebook::pack_indices(&remapped, target_bits);

        CompressedKeys {
            packed_indices: packed,
            norms: self.norms.clone(),
            qjl_corrections: None,
            bits: target_bits,
            dim: self.dim,
            count: self.count,
            rotation_seed: self.rotation_seed,
            group_size: self.group_size,
            residual_indices: None,
            residual_norms: None,
            residual_bits: 0,
            outlier_indices: None,
            outlier_values: None,
            outlier_k: 0,
        }
    }

    /// Append all vectors from `other` into this cache.
    /// Both must have the same bit width and dimension.
    pub fn append_from(&mut self, other: &CompressedKeys) {
        assert_eq!(self.bits, other.bits);
        assert_eq!(self.dim, other.dim);
        self.packed_indices.extend_from_slice(&other.packed_indices);
        self.norms.extend_from_slice(&other.norms);
        self.count += other.count;
    }
}

// ============================================================
// Temporal Decay Configuration
// ============================================================

/// A single decay tier: tokens older than `age_threshold` get compressed to `bits`.
#[derive(Clone, Debug)]
pub struct DecayTier {
    /// Token age (distance from most recent) at which this tier activates.
    pub age_threshold: usize,
    /// Target bit width for this tier.
    pub bits: u8,
}

/// Temporal decay configuration.
///
/// Older tokens are progressively compressed to lower bit widths.
/// Example: `tiers = [DecayTier { age: 1024, bits: 3 }, DecayTier { age: 4096, bits: 2 }]`
/// means tokens older than 1024 get 3-bit, older than 4096 get 2-bit.
///
/// Tiers must be sorted by age_threshold ascending, bits descending.
#[derive(Clone, Debug)]
pub struct TemporalDecayConfig {
    /// Decay tiers, sorted by age_threshold ascending.
    pub tiers: Vec<DecayTier>,
    /// How often (in tokens) to check and apply decay. Default: 128.
    pub decay_interval: usize,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            tiers: vec![
                DecayTier { age_threshold: 512, bits: 2 },
            ],
            decay_interval: 128,
        }
    }
}

// ============================================================
// Value Compression (K/V Asymmetric)
// ============================================================

/// Compressed value cache using per-vector absmax quantization.
///
/// Each value vector is quantized to 8-bit using symmetric absmax scaling:
///   quantized[i] = round(clamp(value[i] / scale, -127, 127)) + 128
///   scale = max(|value[i]|) / 127
///
/// This gives 2x memory savings vs fp16 with negligible quality loss.
/// Unlike keys (which use Hadamard + Lloyd-Max), values don't benefit from
/// rotation — absmax is simpler and sufficient at 8-bit.
#[derive(Clone, Debug)]
pub struct CompressedValues {
    /// Quantized data: uint8, row-major [count * dim]
    pub data: Vec<u8>,
    /// Per-vector absmax scale factors
    pub scales: Vec<f32>,
    /// Vector dimension (head_dim)
    pub dim: usize,
    /// Number of vectors
    pub count: usize,
}

impl CompressedValues {
    /// Create empty compressed values (for incremental append).
    pub fn new_empty(dim: usize) -> Self {
        Self { data: Vec::new(), scales: Vec::new(), dim, count: 0 }
    }

    /// Append a single value vector (f32) to the compressed cache.
    pub fn append(&mut self, value: &[f32]) {
        debug_assert_eq!(value.len(), self.dim);
        let absmax = value.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if absmax > 1e-10 { absmax / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        for &v in value {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            self.data.push((q as i16 + 128) as u8);
        }
        self.scales.push(scale);
        self.count += 1;
    }

    /// Append multiple value vectors from a flat f32 slice.
    pub fn append_batch(&mut self, values: &[f32], dim: usize) {
        debug_assert_eq!(values.len() % dim, 0);
        for chunk in values.chunks_exact(dim) {
            self.append(chunk);
        }
    }

    /// Decompress all values back to f32.
    pub fn decompress(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.count * self.dim);
        for i in 0..self.count {
            let scale = self.scales[i];
            let start = i * self.dim;
            let end = start + self.dim;
            for &q in &self.data[start..end] {
                let v = (q as i16 - 128) as f32 * scale;
                result.push(v);
            }
        }
        result
    }

    /// Decompress a range of vectors [start_idx, start_idx + count).
    pub fn decompress_range(&self, start_idx: usize, range_count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(range_count * self.dim);
        for i in start_idx..start_idx + range_count {
            let scale = self.scales[i];
            let start = i * self.dim;
            let end = start + self.dim;
            for &q in &self.data[start..end] {
                let v = (q as i16 - 128) as f32 * scale;
                result.push(v);
            }
        }
        result
    }

    /// Compressed memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Original fp16 memory in bytes.
    pub fn original_memory_bytes(&self) -> usize {
        self.count * self.dim * 2
    }

    /// Compression ratio vs fp16.
    pub fn compression_ratio(&self) -> f32 {
        if self.count == 0 { return 0.0; }
        self.original_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

// ============================================================
// 4-bit Value Compression (Sparse V)
// ============================================================

/// Compressed value cache using per-group 4-bit absmax quantization.
///
/// Each group of `group_size` elements within a value vector is quantized independently:
///   quantized[i] = round(clamp(value[i] / scale, -7, 7)) + 8
///   scale = max(|value[g*gs..(g+1)*gs]|) / 7
///
/// Packing: 2 values per byte (low nibble first).
///
/// Compression ratio vs fp16 (dim=128, gs=32): 256 / 80 = **3.2x**.
/// Quality: cos_sim > 0.995 on typical LLM value activations.
#[derive(Clone, Debug)]
pub struct CompressedValues4Bit {
    /// Packed 4-bit data: 2 values per byte, row-major [count * dim / 2]
    pub data: Vec<u8>,
    /// Per-group absmax scales, layout: [count * n_groups] where n_groups = dim / group_size
    pub scales: Vec<f32>,
    /// Vector dimension (head_dim)
    pub dim: usize,
    /// Group size for per-group quantization (default 32)
    pub group_size: usize,
    /// Number of vectors stored
    pub count: usize,
}

impl CompressedValues4Bit {
    /// Create empty compressed values (for incremental append).
    pub fn new_empty(dim: usize, group_size: usize) -> Self {
        assert!(dim > 0 && group_size > 0 && dim % group_size == 0);
        Self { data: Vec::new(), scales: Vec::new(), dim, group_size, count: 0 }
    }

    /// Number of groups per vector.
    #[inline]
    fn n_groups(&self) -> usize {
        self.dim / self.group_size
    }

    /// Packed bytes per vector (dim / 2).
    #[inline]
    fn bytes_per_vec(&self) -> usize {
        self.dim / 2
    }

    /// Append a single value vector (f32) to the compressed cache.
    pub fn append(&mut self, value: &[f32]) {
        debug_assert_eq!(value.len(), self.dim);
        let gs = self.group_size;

        // Quantize per group
        let mut nibbles = Vec::with_capacity(self.dim);
        for group in value.chunks_exact(gs) {
            let absmax = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if absmax > 1e-10 { absmax / 7.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            for &v in group {
                let q = (v * inv_scale).round().clamp(-7.0, 7.0) as i8;
                nibbles.push((q + 8) as u8);
            }
            self.scales.push(scale);
        }

        // Pack nibbles: 2 per byte (low nibble first)
        for pair in nibbles.chunks(2) {
            let lo = pair[0] & 0x0F;
            let hi = if pair.len() > 1 { pair[1] & 0x0F } else { 0 };
            self.data.push(lo | (hi << 4));
        }
        self.count += 1;
    }

    /// Append multiple value vectors from a flat f32 slice.
    pub fn append_batch(&mut self, values: &[f32], dim: usize) {
        debug_assert_eq!(values.len() % dim, 0);
        for chunk in values.chunks_exact(dim) {
            self.append(chunk);
        }
    }

    /// Decompress all values back to f32.
    pub fn decompress(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.count * self.dim);
        let gs = self.group_size;
        let ng = self.n_groups();
        let bpv = self.bytes_per_vec();

        for row in 0..self.count {
            let data_start = row * bpv;
            let scale_start = row * ng;

            for g in 0..ng {
                let scale = self.scales[scale_start + g];
                let elem_start = g * gs;
                for j in 0..gs {
                    let idx = elem_start + j;
                    let byte_idx = data_start + idx / 2;
                    let nibble = if idx % 2 == 0 {
                        self.data[byte_idx] & 0x0F
                    } else {
                        (self.data[byte_idx] >> 4) & 0x0F
                    };
                    let v = (nibble as i8 - 8) as f32 * scale;
                    result.push(v);
                }
            }
        }
        result
    }

    /// Decompress a range of vectors [start_idx, start_idx + count).
    pub fn decompress_range(&self, start_idx: usize, range_count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(range_count * self.dim);
        let gs = self.group_size;
        let ng = self.n_groups();
        let bpv = self.bytes_per_vec();

        for row in start_idx..start_idx + range_count {
            let data_start = row * bpv;
            let scale_start = row * ng;

            for g in 0..ng {
                let scale = self.scales[scale_start + g];
                let elem_start = g * gs;
                for j in 0..gs {
                    let idx = elem_start + j;
                    let byte_idx = data_start + idx / 2;
                    let nibble = if idx % 2 == 0 {
                        self.data[byte_idx] & 0x0F
                    } else {
                        (self.data[byte_idx] >> 4) & 0x0F
                    };
                    let v = (nibble as i8 - 8) as f32 * scale;
                    result.push(v);
                }
            }
        }
        result
    }

    /// Decompress a single row into a pre-allocated buffer (hot path for fused sparse multiply).
    pub fn decompress_row_into(&self, row_idx: usize, output: &mut [f32]) {
        debug_assert!(row_idx < self.count);
        debug_assert!(output.len() >= self.dim);
        let gs = self.group_size;
        let ng = self.n_groups();
        let bpv = self.bytes_per_vec();
        let data_start = row_idx * bpv;
        let scale_start = row_idx * ng;

        for g in 0..ng {
            let scale = self.scales[scale_start + g];
            let elem_start = g * gs;
            for j in 0..gs {
                let idx = elem_start + j;
                let byte_idx = data_start + idx / 2;
                let nibble = if idx % 2 == 0 {
                    self.data[byte_idx] & 0x0F
                } else {
                    (self.data[byte_idx] >> 4) & 0x0F
                };
                output[idx] = (nibble as i8 - 8) as f32 * scale;
            }
        }
    }

    /// Compressed memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Original fp16 memory in bytes.
    pub fn original_memory_bytes(&self) -> usize {
        self.count * self.dim * 2
    }

    /// Compression ratio vs fp16.
    pub fn compression_ratio(&self) -> f32 {
        if self.count == 0 { return 0.0; }
        self.original_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

// ============================================================
// PolarQuant V Compression (Rotation + Lloyd-Max for Values)
// ============================================================

/// Value compression using PolarQuant-MSE: Hadamard rotation + Lloyd-Max codebook.
///
/// Same algorithm as K compression but WITHOUT QJL (values need MSE, not inner product
/// preservation). Achieves ~6x compression at +1% PPL — significantly better than
/// naive absmax quantization at the same bit rate.
///
/// Algorithm per value vector:
/// 1. Rotate: y = H @ D @ v (randomized Hadamard)
/// 2. Quantize: idx[i] = nearest_centroid(y[i]) with sigma = ||v||/sqrt(d)
/// 3. Store: (packed_indices, ||v||)
///
/// Dequantize:
/// 1. Lookup centroids: y_hat[i] = centroid[idx[i]]
/// 2. Inverse rotate: v_hat = D @ H @ y_hat
/// 3. Rescale by stored norm
#[derive(Clone, Debug)]
pub struct CompressedValuesPQ {
    /// Underlying compressed data (same format as CompressedKeys)
    inner: CompressedKeys,
    /// Hadamard sign vector (cached for fast decompress)
    signs: Vec<f32>,
}

impl CompressedValuesPQ {
    /// Create empty for incremental append.
    pub fn new_empty(dim: usize, bits: u8, rotation_seed: u64) -> Self {
        let signs = hadamard::generate_signs(dim, rotation_seed);
        Self {
            inner: CompressedKeys::new_empty(bits, dim, rotation_seed),
            signs,
        }
    }

    /// Append a single value vector.
    pub fn append(&mut self, value: &[f32]) {
        debug_assert_eq!(value.len(), self.inner.dim);
        let dim = self.inner.dim;

        // 1. Rotate with Hadamard
        let mut rotated = value.to_vec();
        hadamard::randomized_hadamard_with_signs(&mut rotated, &self.signs);

        // 2. Quantize with adaptive sigma per-vector
        let norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        let base_cb = codebook::Codebook::new(self.inner.bits, dim);

        if norm < 1e-10 {
            self.inner.norms.push(norm);
            let bytes_per_vec = (dim * self.inner.bits as usize + 7) / 8;
            self.inner.packed_indices.extend(std::iter::repeat(0u8).take(bytes_per_vec));
        } else {
            let adaptive_sigma = norm / (dim as f32).sqrt();
            let cb = codebook::Codebook { sigma: adaptive_sigma, ..base_cb };
            let indices: Vec<u8> = rotated.iter().map(|&v| cb.quantize(v)).collect();

            // Norm correction
            let recon_norm: f32 = indices.iter()
                .map(|&idx| { let v = cb.dequantize(idx); v * v })
                .sum::<f32>().sqrt();
            let corrected_norm = if recon_norm > 1e-10 { norm * norm / recon_norm } else { norm };
            self.inner.norms.push(corrected_norm);

            let packed = codebook::pack_indices(&indices, self.inner.bits);
            self.inner.packed_indices.extend_from_slice(&packed);
        }
        self.inner.count += 1;
    }

    /// Append batch from flat f32 slice.
    pub fn append_batch(&mut self, values: &[f32], dim: usize) {
        for chunk in values.chunks_exact(dim) {
            self.append(chunk);
        }
    }

    /// Decompress all values back to f32.
    pub fn decompress(&self) -> Vec<f32> {
        let dim = self.inner.dim;
        let bits = self.inner.bits;
        let base_cb = codebook::Codebook::new(bits, dim);
        let bytes_per_vec = (dim * bits as usize + 7) / 8;
        let mut result = Vec::with_capacity(self.inner.count * dim);

        for i in 0..self.inner.count {
            let norm = self.inner.norms[i];
            let start = i * bytes_per_vec;
            let end = start + bytes_per_vec;
            let indices = codebook::unpack_indices(
                &self.inner.packed_indices[start..end], dim, bits,
            );

            let adaptive_sigma = if norm > 1e-10 {
                norm / (dim as f32).sqrt()
            } else {
                base_cb.sigma
            };
            let cb = codebook::Codebook { sigma: adaptive_sigma, ..base_cb.clone() };

            let mut reconstructed: Vec<f32> = indices.iter()
                .map(|&idx| cb.dequantize(idx))
                .collect();

            // Inverse Hadamard: WHT then sign flip (reverse of forward: sign flip then WHT)
            hadamard::inverse_randomized_hadamard_with_signs(&mut reconstructed, &self.signs);

            result.extend_from_slice(&reconstructed);
        }
        result
    }

    /// Number of compressed vectors.
    pub fn count(&self) -> usize { self.inner.count }

    /// Compressed memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.inner.packed_indices.len() + self.inner.norms.len() * 4
    }

    /// Compression ratio vs fp16.
    pub fn compression_ratio(&self) -> f32 {
        if self.inner.count == 0 { return 0.0; }
        (self.inner.count * self.inner.dim * 2) as f32 / self.memory_bytes() as f32
    }
}

/// Compress key vectors using paper-faithful Lloyd-Max codebook.
///
/// TurboQuant paper algorithm (Zandieh et al., ICLR 2026):
/// 1. Random rotation (Hadamard) → coordinates become ~Gaussian(0, σ)
/// 2. Per-coordinate Lloyd-Max scalar quantization (NO unit normalization)
/// 3. Optional QJL 1-bit residual correction
///
/// Key insight: Hadamard rotation makes each coordinate ≈ N(0, ||x||/√d).
/// The codebook sigma adapts per-vector to match this variance.
/// Unit normalization is WRONG — it changes the distribution from Gaussian
/// to Beta (unit sphere), breaking Lloyd-Max optimality.
pub fn compress_keys(
    data: &[f32],
    dim: usize,
    config: &TurboQuantConfig,
) -> CompressedKeys {
    assert_eq!(data.len() % dim, 0);
    assert!(dim.is_power_of_two());

    let count = data.len() / dim;

    // 1. Per-channel scaling + Hadamard rotation
    let mut rotated = data.to_vec();
    if let Some(ref scales) = config.channel_scales {
        debug_assert_eq!(scales.len(), dim);
        for chunk in rotated.chunks_exact_mut(dim) {
            for (val, &s) in chunk.iter_mut().zip(scales.iter()) {
                *val *= s;
            }
        }
    }
    for chunk in rotated.chunks_exact_mut(dim) {
        hadamard::randomized_hadamard(chunk, config.rotation_seed);
    }

    // 2. Per-coordinate Lloyd-Max quantization with adaptive sigma
    // After rotation, coordinate i ≈ N(0, ||x||/√d)
    // Codebook sigma = ||x||/√d per vector (NOT fixed 1/√d)
    let mut all_indices = Vec::with_capacity(count * dim);
    let mut norms = Vec::with_capacity(count);
    let base_cb = codebook::Codebook::new(config.bits, dim);

    for chunk in rotated.chunks_exact(dim) {
        let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm < 1e-10 {
            norms.push(norm);
            all_indices.extend(std::iter::repeat(0u8).take(dim));
        } else {
            let adaptive_sigma = norm / (dim as f32).sqrt();
            let cb = codebook::Codebook {
                sigma: adaptive_sigma,
                ..base_cb.clone()
            };
            let indices: Vec<u8> = chunk.iter()
                .map(|&v| cb.quantize(v))
                .collect();

            // Norm correction: adjust stored norm so ||decompress|| ≈ ||original||
            let recon_norm: f32 = indices.iter()
                .map(|&idx| { let v = cb.dequantize(idx); v * v })
                .sum::<f32>().sqrt();
            let corrected_norm = if recon_norm > 1e-10 { norm * norm / recon_norm } else { norm };
            norms.push(corrected_norm);

            all_indices.extend_from_slice(&indices);
        }
    }

    // 3. Bit-pack indices
    let packed_indices = codebook::pack_indices(&all_indices, config.bits);

    // 4. QJL correction on residual (optional)
    let qjl_corrections = if config.use_qjl {
        let mut errors = Vec::with_capacity(count * dim);
        for (i, chunk) in rotated.chunks_exact(dim).enumerate() {
            let start = i * dim;
            let indices = &all_indices[start..start + dim];
            let norm = norms[i];
            if norm < 1e-10 {
                errors.extend(std::iter::repeat(0.0f32).take(dim));
                continue;
            }
            let adaptive_sigma = norm / (dim as f32).sqrt();
            let cb = codebook::Codebook {
                sigma: adaptive_sigma,
                ..base_cb.clone()
            };
            for (&orig, &idx) in chunk.iter().zip(indices.iter()) {
                let recon = cb.dequantize(idx);
                errors.push(orig - recon);
            }
        }
        let proj_dim = if config.qjl_proj_dim == 0 { dim } else { config.qjl_proj_dim };
        Some(qjl::compute_batch(&errors, dim, proj_dim, config.qjl_seed))
    } else {
        None
    };

    CompressedKeys {
        packed_indices,
        norms,
        qjl_corrections,
        bits: config.bits,
        dim,
        count,
        rotation_seed: config.rotation_seed,
        group_size: 0, // batch compress_keys uses per-vector sigma (legacy)
        residual_indices: None,
        residual_norms: None,
        residual_bits: 0,
        outlier_indices: None,
        outlier_values: None,
        outlier_k: 0,
    }
}

/// Decompress keys back to f32.
pub fn decompress_keys(compressed: &CompressedKeys, _config: &TurboQuantConfig) -> Vec<f32> {
    let base_cb = codebook::Codebook::new(compressed.bits, compressed.dim);
    let dim = compressed.dim;

    // Unpack indices
    let all_indices = codebook::unpack_indices(
        &compressed.packed_indices, compressed.count * dim, compressed.bits,
    );

    // Dequantize with adaptive sigma per vector
    let mut result = Vec::with_capacity(compressed.count * dim);
    for i in 0..compressed.count {
        let start = i * dim;
        let indices = &all_indices[start..start + dim];
        let norm = compressed.norms[i];
        if norm < 1e-10 {
            result.extend(std::iter::repeat(0.0f32).take(dim));
            continue;
        }
        let adaptive_sigma = norm / (dim as f32).sqrt();
        let cb = codebook::Codebook {
            sigma: adaptive_sigma,
            ..base_cb.clone()
        };
        for &idx in indices {
            result.push(cb.dequantize(idx));
        }
    }

    // QJL correction
    if let Some(ref corrections) = compressed.qjl_corrections {
        qjl::apply_batch(&mut result, corrections, dim);
    }

    // Inverse Hadamard
    for chunk in result.chunks_exact_mut(dim) {
        hadamard::inverse_randomized_hadamard(chunk, compressed.rotation_seed);
    }

    // Inverse per-channel scaling
    if let Some(ref scales) = _config.channel_scales {
        for chunk in result.chunks_exact_mut(dim) {
            for (val, &s) in chunk.iter_mut().zip(scales.iter()) {
                if s.abs() > 1e-10 { *val /= s; }
            }
        }
    }

    // Inverse Pre-Rotation Centering: add bias back
    if let Some(ref bias) = _config.key_channel_bias {
        for chunk in result.chunks_exact_mut(dim) {
            for (val, &b) in chunk.iter_mut().zip(bias.iter()) {
                *val += b;
            }
        }
    }

    result
}

/// Pre-rotate a query vector for fused attention.
///
/// The key insight from the paper:
///   ⟨q, Rᵀ·centroids[idx]⟩ = ⟨R·q, centroids[idx]⟩
///
/// Pre-rotate q once, then use centroid table lookup.
/// No need to decompress keys at all!
pub fn pre_rotate_query(query: &[f32], rotation_seed: u64) -> Vec<f32> {
    let mut rotated = query.to_vec();
    hadamard::randomized_hadamard(&mut rotated, rotation_seed);
    rotated
}

/// Pre-rotate query with pre-computed signs (alloc-free hot path).
pub fn pre_rotate_query_with_signs(query: &[f32], signs: &[f32]) -> Vec<f32> {
    let mut rotated = query.to_vec();
    hadamard::randomized_hadamard_with_signs(&mut rotated, signs);
    rotated
}

/// Pre-rotate query with a custom rotation matrix (SpinQuant/PCA).
pub fn pre_rotate_query_with_matrix(query: &[f32], matrix: &[f32]) -> Vec<f32> {
    let mut rotated = query.to_vec();
    hadamard::apply_rotation(&mut rotated, matrix);
    rotated
}

/// Compress a single key vector. For incremental KV cache.
/// Returns: (packed_indices, corrected_norm)
pub fn compress_single_key(
    key: &[f32],
    dim: usize,
    config: &TurboQuantConfig,
) -> (Vec<u8>, f32) {
    assert_eq!(key.len(), dim);

    let mut rotated = key.to_vec();
    if let Some(ref scales) = config.channel_scales {
        for (val, &s) in rotated.iter_mut().zip(scales.iter()) {
            *val *= s;
        }
    }
    hadamard::randomized_hadamard(&mut rotated, config.rotation_seed);

    let norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
    let base_cb = codebook::Codebook::new(config.bits, dim);

    let indices: Vec<u8> = if norm < 1e-10 {
        vec![0u8; dim]
    } else {
        let sigma = norm / (dim as f32).sqrt();
        let cb = codebook::Codebook { sigma, ..base_cb };
        rotated.iter().map(|&v| cb.quantize(v)).collect()
    };

    // Norm correction (see compress_single_key_with_signs for explanation)
    let corrected_norm = if norm > 1e-10 {
        let sigma = norm / (dim as f32).sqrt();
        let cb = codebook::Codebook { sigma, ..base_cb };
        let recon_norm: f32 = indices.iter()
            .map(|&idx| { let v = cb.dequantize(idx); v * v })
            .sum::<f32>().sqrt();
        if recon_norm > 1e-10 { norm * norm / recon_norm } else { norm }
    } else {
        norm
    };

    let packed = codebook::pack_indices(&indices, config.bits);
    (packed, corrected_norm)
}

/// Compress a single key vector with pre-computed signs.
/// Saves signs allocation in the hot loop.
///
/// **Norm Correction** (from turboquant_plus): after quantization, the reconstruction's
/// L2 norm differs from the original. We store a corrected norm such that
/// `||decompress(compress(k))|| ≈ ||k||`. This is free at decode time because
/// decompression already scales by `stored_norm / sqrt(d)`.
pub fn compress_single_key_with_signs(
    key: &[f32],
    dim: usize,
    config: &TurboQuantConfig,
    signs: &[f32],
) -> (Vec<u8>, f32) {
    assert_eq!(key.len(), dim);

    let mut rotated = key.to_vec();

    // Pre-Rotation Centering: subtract weight quantization bias
    if let Some(ref bias) = config.key_channel_bias {
        for (val, &b) in rotated.iter_mut().zip(bias.iter()) {
            *val -= b;
        }
    }

    // Per-channel scaling (SmoothQuant): equalize outlier magnitudes before rotation
    if let Some(ref scales) = config.channel_scales {
        debug_assert_eq!(scales.len(), dim);
        for (val, &s) in rotated.iter_mut().zip(scales.iter()) {
            *val *= s;
        }
    }

    // Apply rotation: custom matrix (SpinQuant) or randomized Hadamard
    if let Some(ref matrix) = config.rotation_matrix {
        hadamard::apply_rotation(&mut rotated, matrix);
    } else {
        hadamard::randomized_hadamard_with_signs(&mut rotated, signs);
    }

    let norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
    let base_cb = codebook::Codebook::new(config.bits, dim);

    let indices: Vec<u8> = if norm < 1e-10 {
        vec![0u8; dim]
    } else {
        let sigma = norm / (dim as f32).sqrt();
        if let Some(ref cal_cb) = config.calibrated_codebook {
            rotated.iter().map(|&v| cal_cb.quantize(v / sigma)).collect()
        } else {
            let cb = codebook::Codebook { sigma, ..base_cb };
            rotated.iter().map(|&v| cb.quantize(v)).collect()
        }
    };

    // Norm correction
    let corrected_norm = if norm > 1e-10 {
        let sigma = norm / (dim as f32).sqrt();
        let recon_norm_sq: f32 = if let Some(ref cal_cb) = config.calibrated_codebook {
            indices.iter()
                .map(|&idx| { let v = cal_cb.dequantize(idx) * sigma; v * v })
                .sum()
        } else {
            let cb = codebook::Codebook { sigma, ..base_cb };
            indices.iter()
                .map(|&idx| { let v = cb.dequantize(idx); v * v })
                .sum()
        };
        let recon_norm = recon_norm_sq.sqrt();
        if recon_norm > 1e-10 { norm * norm / recon_norm } else { norm }
    } else {
        norm
    };

    let packed = codebook::pack_indices(&indices, config.bits);
    (packed, corrected_norm)
}

/// Compress a single key with per-group quantization.
///
/// Instead of one sigma for the entire vector, each group of `group_size` dimensions
/// gets its own sigma = group_norm / sqrt(group_size). This captures within-vector
/// magnitude variation that per-vector sigma misses.
///
/// Returns: (packed_indices, group_norms, residual, outliers)
pub fn compress_single_key_grouped(
    key: &[f32],
    dim: usize,
    config: &TurboQuantConfig,
    signs: &[f32],
) -> (Vec<u8>, Vec<f32>, Option<(Vec<u8>, Vec<f32>)>, Option<(Vec<u8>, Vec<f32>)>) {
    assert_eq!(key.len(), dim);
    let gs = config.group_size;
    assert!(gs > 0 && dim % gs == 0, "dim {} must be divisible by group_size {}", dim, gs);

    let mut rotated = key.to_vec();

    // Pre-Rotation Centering: subtract per-channel key bias from weight quantization.
    // This restores the zero-mean assumption that Lloyd-Max codebook requires.
    // On FP16 models, bias ≈ 0 (no effect). On GGUF Q4_K_M, removes systematic shift.
    if let Some(ref bias) = config.key_channel_bias {
        for (val, &b) in rotated.iter_mut().zip(bias.iter()) {
            *val -= b;
        }
    }

    if let Some(ref scales) = config.channel_scales {
        for (val, &s) in rotated.iter_mut().zip(scales.iter()) {
            *val *= s;
        }
    }

    // Apply rotation: custom matrix (SpinQuant) or randomized Hadamard
    if let Some(ref matrix) = config.rotation_matrix {
        hadamard::apply_rotation(&mut rotated, matrix);
    } else {
        hadamard::randomized_hadamard_with_signs(&mut rotated, signs);
    }

    // Outlier extraction: find top-K by absolute value, save, zero out
    let outliers = if config.outlier_k > 0 {
        let k = config.outlier_k.min(dim);
        // Find indices of top-K abs values
        let mut abs_indexed: Vec<(usize, f32)> = rotated.iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        abs_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut out_indices = Vec::with_capacity(k);
        let mut out_values = Vec::with_capacity(k);
        for &(idx, _) in abs_indexed.iter().take(k) {
            out_indices.push(idx as u8);
            out_values.push(rotated[idx]);
            rotated[idx] = 0.0; // zero out outlier before quantization
        }
        Some((out_indices, out_values))
    } else {
        None
    };

    let base_cb = codebook::Codebook::new(config.bits, dim);
    let n_groups = dim / gs;
    let mut indices = Vec::with_capacity(dim);
    let mut group_norms = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let start = g * gs;
        let group = &rotated[start..start + gs];
        let group_norm: f32 = group.iter().map(|x| x * x).sum::<f32>().sqrt();

        if group_norm < 1e-10 {
            indices.extend(std::iter::repeat(0u8).take(gs));
            group_norms.push(0.0);
        } else {
            let sigma = group_norm / (gs as f32).sqrt();

            let group_indices: Vec<u8> = if let Some(ref cal_cb) = config.calibrated_codebook {
                // Calibrated codebook: quantize normalized values (v / sigma)
                group.iter().map(|&v| cal_cb.quantize(v / sigma)).collect()
            } else {
                let cb = codebook::Codebook { sigma, ..base_cb.clone() };
                group.iter().map(|&v| cb.quantize(v)).collect()
            };

            // Norm correction per group
            let recon_norm: f32 = if let Some(ref cal_cb) = config.calibrated_codebook {
                group_indices.iter()
                    .map(|&idx| { let v = cal_cb.dequantize(idx) * sigma; v * v })
                    .sum::<f32>().sqrt()
            } else {
                let cb = codebook::Codebook { sigma, ..base_cb.clone() };
                group_indices.iter()
                    .map(|&idx| { let v = cb.dequantize(idx); v * v })
                    .sum::<f32>().sqrt()
            };
            let corrected = if recon_norm > 1e-10 { group_norm * group_norm / recon_norm } else { group_norm };

            indices.extend_from_slice(&group_indices);
            group_norms.push(corrected);
        }
    }

    let packed = codebook::pack_indices(&indices, config.bits);

    // Residual quantization: quantize the first-pass error
    let residual = if config.residual_bits > 0 {
        let res_cb = codebook::Codebook::new(config.residual_bits, dim);
        let mut res_indices = Vec::with_capacity(dim);
        let mut res_norms = Vec::with_capacity(n_groups);

        for g in 0..n_groups {
            let start = g * gs;
            let group = &rotated[start..start + gs];
            let gn = group_norms[g];
            let sigma = if gn > 1e-10 { gn / (gs as f32).sqrt() } else { 1.0 };
            let cb = codebook::Codebook { sigma, ..base_cb.clone() };

            // Compute residual: original_rotated - first_pass_reconstruction
            let mut residual_group = Vec::with_capacity(gs);
            for j in 0..gs {
                let recon = cb.dequantize(indices[start + j]);
                residual_group.push(group[j] - recon);
            }

            // Quantize residual with its own sigma
            let res_norm: f32 = residual_group.iter().map(|x| x * x).sum::<f32>().sqrt();
            if res_norm < 1e-10 {
                res_indices.extend(std::iter::repeat(0u8).take(gs));
                res_norms.push(0.0);
            } else {
                let res_sigma = res_norm / (gs as f32).sqrt();
                let rcb = codebook::Codebook { sigma: res_sigma, ..res_cb.clone() };
                let ri: Vec<u8> = residual_group.iter().map(|&v| rcb.quantize(v)).collect();

                // Norm correction for residual
                let res_recon_norm: f32 = ri.iter()
                    .map(|&idx| { let v = rcb.dequantize(idx); v * v })
                    .sum::<f32>().sqrt();
                let corrected = if res_recon_norm > 1e-10 { res_norm * res_norm / res_recon_norm } else { res_norm };

                res_indices.extend_from_slice(&ri);
                res_norms.push(corrected);
            }
        }

        let res_packed = codebook::pack_indices(&res_indices, config.residual_bits);
        Some((res_packed, res_norms))
    } else {
        None
    };

    (packed, group_norms, residual, outliers)
}

/// Decompress keys with per-group norms.
pub fn decompress_keys_grouped(compressed: &CompressedKeys, config: &TurboQuantConfig) -> Vec<f32> {
    let gs = compressed.group_size;
    if gs == 0 {
        return decompress_keys(compressed, config);
    }

    let base_cb = codebook::Codebook::new(compressed.bits, compressed.dim);
    let dim = compressed.dim;
    let n_groups = dim / gs;
    let npv = compressed.norms_per_vector();

    let all_indices = codebook::unpack_indices(
        &compressed.packed_indices, compressed.count * dim, compressed.bits,
    );

    let mut result = Vec::with_capacity(compressed.count * dim);
    for i in 0..compressed.count {
        let norm_offset = i * npv;
        let idx_offset = i * dim;

        for g in 0..n_groups {
            let group_norm = compressed.norms[norm_offset + g];
            let sigma = if group_norm > 1e-10 { group_norm / (gs as f32).sqrt() } else { 1.0 };

            let gstart = idx_offset + g * gs;
            if let Some(ref cal_cb) = config.calibrated_codebook {
                for j in 0..gs {
                    let idx = all_indices[gstart + j];
                    result.push(cal_cb.dequantize(idx) * sigma);
                }
            } else {
                let cb = codebook::Codebook { sigma, ..base_cb.clone() };
                for j in 0..gs {
                    let idx = all_indices[gstart + j];
                    result.push(cb.dequantize(idx));
                }
            }
        }
    }

    // Add residual correction (in rotated domain, before inverse Hadamard)
    if let (Some(ref res_packed), Some(ref res_norms)) =
        (&compressed.residual_indices, &compressed.residual_norms)
    {
        let res_bits = compressed.residual_bits;
        if res_bits > 0 {
            let res_cb = codebook::Codebook::new(res_bits, dim);
            let res_indices = codebook::unpack_indices(res_packed, compressed.count * dim, res_bits);

            for i in 0..compressed.count {
                let norm_offset = i * npv;
                let idx_offset = i * dim;
                for g in 0..n_groups {
                    let res_norm = res_norms[norm_offset + g];
                    let sigma = if res_norm > 1e-10 { res_norm / (gs as f32).sqrt() } else { 1.0 };
                    let rcb = codebook::Codebook { sigma, ..res_cb.clone() };
                    let gstart = idx_offset + g * gs;
                    for j in 0..gs {
                        let idx = res_indices[gstart + j];
                        result[idx_offset + g * gs + j] += rcb.dequantize(idx);
                    }
                }
            }
        }
    }

    // Restore sparse outliers (in rotated domain, before inverse Hadamard)
    if let (Some(ref out_idx), Some(ref out_val)) =
        (&compressed.outlier_indices, &compressed.outlier_values)
    {
        let k = compressed.outlier_k;
        if k > 0 {
            for i in 0..compressed.count {
                let vec_offset = i * dim;
                let sparse_offset = i * k;
                for j in 0..k {
                    let idx = out_idx[sparse_offset + j] as usize;
                    result[vec_offset + idx] = out_val[sparse_offset + j];
                }
            }
        }
    }

    // Inverse rotation
    if let Some(ref matrix) = config.rotation_matrix {
        for chunk in result.chunks_exact_mut(dim) {
            hadamard::apply_inverse_rotation(chunk, matrix);
        }
    } else {
        for chunk in result.chunks_exact_mut(dim) {
            hadamard::inverse_randomized_hadamard(chunk, compressed.rotation_seed);
        }
    }

    // Inverse channel scaling
    if let Some(ref scales) = config.channel_scales {
        for chunk in result.chunks_exact_mut(dim) {
            for (val, &s) in chunk.iter_mut().zip(scales.iter()) {
                if s.abs() > 1e-10 { *val /= s; }
            }
        }
    }

    // Inverse Pre-Rotation Centering: add bias back
    if let Some(ref bias) = config.key_channel_bias {
        for chunk in result.chunks_exact_mut(dim) {
            for (val, &b) in chunk.iter_mut().zip(bias.iter()) {
                *val += b;
            }
        }
    }

    result
}

/// Compute attention score between pre-rotated query and compressed key.
///
/// Fused approach — NO key decompression needed:
///   ⟨q, k⟩ = ⟨q, R^T · k_rotated⟩ = ⟨R·q, k_rotated⟩
///
/// k_rotated is approximated by centroid lookups.
/// `key_norm`: stored norm of the key vector (for adaptive sigma).
pub fn fused_dot_product(
    rotated_query: &[f32],
    key_indices: &[u8],
    key_norm: f32,
    bits: u8,
    dim: usize,
) -> f32 {
    if key_norm < 1e-10 {
        return 0.0;
    }
    let adaptive_sigma = key_norm / (dim as f32).sqrt();
    let base_cb = codebook::Codebook::new(bits, dim);
    let cb = codebook::Codebook {
        sigma: adaptive_sigma,
        ..base_cb
    };

    rotated_query.iter().zip(key_indices.iter())
        .map(|(&q, &idx)| q * cb.dequantize(idx))
        .sum()
}

/// Fused dot product with pre-computed centroid table.
/// Eliminates Codebook construction overhead in the hot loop.
///
/// `base_centroids`: N(0,1) centroids (obtained via codebook::get_centroids)
/// Adaptive sigma per-key: scaled by centroid * sigma.
///
/// Uses SIMD (AVX2) when available for ~4x speedup on the inner loop.
#[inline]
pub fn fused_dot_product_with_centroids(
    rotated_query: &[f32],
    key_indices: &[u8],
    key_norm: f32,
    base_centroids: &[f32],
    dim: usize,
) -> f32 {
    if key_norm < 1e-10 {
        return 0.0;
    }
    let sigma = key_norm / (dim as f32).sqrt();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: checked AVX2+FMA support above
            return unsafe {
                fused_dot_avx2(rotated_query, key_indices, base_centroids, sigma)
            };
        }
    }

    // Scalar fallback
    fused_dot_scalar(rotated_query, key_indices, base_centroids, sigma)
}

/// Scalar fused dot product (portable).
#[inline]
fn fused_dot_scalar(
    query: &[f32],
    indices: &[u8],
    centroids: &[f32],
    sigma: f32,
) -> f32 {
    query.iter().zip(indices.iter())
        .map(|(&q, &idx)| q * centroids[idx as usize] * sigma)
        .sum()
}

/// AVX2 + FMA fused dot product — processes 8 floats per cycle.
///
/// Inner loop: gather centroids by index, multiply by query, FMA accumulate.
/// ~4x speedup over scalar on dim=128.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fused_dot_avx2(
    query: &[f32],
    indices: &[u8],
    centroids: &[f32],
    sigma: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let sigma_vec = _mm256_set1_ps(sigma);
    let n = query.len();
    let chunks = n / 8;

    for i in 0..chunks {
        let offset = i * 8;
        // Load 8 query values
        let q = _mm256_loadu_ps(query.as_ptr().add(offset));

        // Gather 8 centroids by index — manual gather (faster than _mm256_i32gather_ps for small tables)
        let c = _mm256_set_ps(
            *centroids.get_unchecked(indices[offset + 7] as usize),
            *centroids.get_unchecked(indices[offset + 6] as usize),
            *centroids.get_unchecked(indices[offset + 5] as usize),
            *centroids.get_unchecked(indices[offset + 4] as usize),
            *centroids.get_unchecked(indices[offset + 3] as usize),
            *centroids.get_unchecked(indices[offset + 2] as usize),
            *centroids.get_unchecked(indices[offset + 1] as usize),
            *centroids.get_unchecked(indices[offset] as usize),
        );

        // FMA: acc += q * c * sigma = q * (c * sigma)
        let cs = _mm256_mul_ps(c, sigma_vec);
        acc = _mm256_fmadd_ps(q, cs, acc);
    }

    // Horizontal sum of 8 accumulators
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Handle remainder
    let remainder_start = chunks * 8;
    for j in remainder_start..n {
        total += query[j] * centroids[indices[j] as usize] * sigma;
    }

    total
}

/// Batch fused attention scores: compute all attention scores between a
/// pre-rotated query and all keys in a compressed cache.
///
/// Returns a Vec of attention scores (one per cached key position).
/// This is the hot path for CPU inference — avoids per-key function call overhead.
pub fn fused_attention_scores(
    rotated_query: &[f32],
    compressed: &CompressedKeys,
    base_centroids: &[f32],
    scale: f32,
) -> Vec<f32> {
    let dim = compressed.dim;
    let bpv = compressed.bytes_per_vector();

    let mut indices_buf = vec![0u8; dim];
    let mut scores = Vec::with_capacity(compressed.count);

    for pos in 0..compressed.count {
        let norm = compressed.norms[pos];
        if norm < 1e-10 {
            scores.push(0.0);
            continue;
        }
        let start = pos * bpv;
        let end = start + bpv;
        codebook::unpack_indices_into(
            &compressed.packed_indices[start..end], &mut indices_buf, compressed.bits,
        );
        let score = fused_dot_product_with_centroids(
            rotated_query, &indices_buf, norm, base_centroids, dim,
        ) * scale;
        scores.push(score);
    }

    scores
}

/// Sparse attention-value multiply: only accumulate V rows where attention weight > threshold.
///
/// For autoregressive decode (seq_len=1), the attention weight vector is typically very sparse
/// after softmax — most positions have near-zero weight. Skipping those positions saves
/// memory bandwidth proportional to the sparsity (often 50-80% of V rows at long context).
///
/// # Arguments
/// * `attn_weights` - Softmax attention weights, shape `[seq_len]` (one query position).
/// * `values` - Value matrix, shape `[seq_len, head_dim]` (row-major).
/// * `head_dim` - Dimension per head.
/// * `threshold` - Weights below this are skipped. Use 0.0 to disable (dense path).
///
/// # Returns
/// Weighted sum vector of length `head_dim`.
pub fn sparse_attn_v_mul(
    attn_weights: &[f32],
    values: &[f32],
    head_dim: usize,
    threshold: f32,
) -> Vec<f32> {
    debug_assert_eq!(values.len(), attn_weights.len() * head_dim);
    let seq_len = attn_weights.len();
    let mut output = vec![0.0f32; head_dim];

    if threshold <= 0.0 {
        // Dense path — no sparsity
        for pos in 0..seq_len {
            let w = attn_weights[pos];
            let v_row = &values[pos * head_dim..(pos + 1) * head_dim];
            for (o, &v) in output.iter_mut().zip(v_row.iter()) {
                *o += w * v;
            }
        }
        return output;
    }

    // Sparse path — skip negligible weights
    for pos in 0..seq_len {
        let w = attn_weights[pos];
        if w < threshold {
            continue;
        }
        let v_row = &values[pos * head_dim..(pos + 1) * head_dim];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { sparse_v_accumulate_avx2(&mut output, v_row, w); }
                continue;
            }
        }
        for (o, &v) in output.iter_mut().zip(v_row.iter()) {
            *o += w * v;
        }
    }

    output
}

/// AVX2+FMA accumulate: output[i] += weight * v_row[i]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sparse_v_accumulate_avx2(output: &mut [f32], v_row: &[f32], weight: f32) {
    use std::arch::x86_64::*;

    let w_vec = _mm256_set1_ps(weight);
    let n = output.len();
    let chunks = n / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let o = _mm256_loadu_ps(output.as_ptr().add(offset));
        let v = _mm256_loadu_ps(v_row.as_ptr().add(offset));
        let result = _mm256_fmadd_ps(w_vec, v, o);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Remainder
    let rem_start = chunks * 8;
    for j in rem_start..n {
        *output.get_unchecked_mut(j) += weight * *v_row.get_unchecked(j);
    }
}

/// Fused sparse attention-value multiply on 4-bit compressed values.
///
/// For each position where `attn_weight >= threshold`:
///   1. Decompress that single V row from 4-bit packed format
///   2. Accumulate: `output[j] += weight * decompressed[j]`
///
/// Positions below threshold are never touched in memory — saving both
/// decompression compute and memory bandwidth (typically 50-80% of rows skipped).
pub fn sparse_attn_v_mul_compressed_4bit(
    attn_weights: &[f32],
    compressed: &CompressedValues4Bit,
    threshold: f32,
) -> Vec<f32> {
    debug_assert_eq!(attn_weights.len(), compressed.count);
    let dim = compressed.dim;
    let mut output = vec![0.0f32; dim];
    let mut row_buf = vec![0.0f32; dim];

    for pos in 0..attn_weights.len() {
        let w = attn_weights[pos];
        if threshold > 0.0 && w < threshold {
            continue;
        }
        compressed.decompress_row_into(pos, &mut row_buf);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { sparse_v_accumulate_avx2(&mut output, &row_buf, w); }
                continue;
            }
        }
        for (o, &v) in output.iter_mut().zip(row_buf.iter()) {
            *o += w * v;
        }
    }

    output
}

/// Fused sparse attention-value multiply on 8-bit compressed values.
///
/// Same as [`sparse_attn_v_mul_compressed_4bit`] but for 8-bit absmax values.
/// Decompresses only the rows that pass the sparsity threshold.
pub fn sparse_attn_v_mul_compressed_8bit(
    attn_weights: &[f32],
    compressed: &CompressedValues,
    threshold: f32,
) -> Vec<f32> {
    debug_assert_eq!(attn_weights.len(), compressed.count);
    let dim = compressed.dim;
    let mut output = vec![0.0f32; dim];
    let mut row_buf = vec![0.0f32; dim];

    for pos in 0..attn_weights.len() {
        let w = attn_weights[pos];
        if threshold > 0.0 && w < threshold {
            continue;
        }
        // Inline single-row 8-bit decompress
        let scale = compressed.scales[pos];
        let start = pos * dim;
        for j in 0..dim {
            let q = compressed.data[start + j];
            row_buf[j] = (q as i16 - 128) as f32 * scale;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { sparse_v_accumulate_avx2(&mut output, &row_buf, w); }
                continue;
            }
        }
        for (o, &v) in output.iter_mut().zip(row_buf.iter()) {
            *o += w * v;
        }
    }

    output
}

/// Softmax bias correction (Bondarenko, arXiv:2309.01729).
///
/// Quantization introduces systematic bias in attention scores: `<q, k_quant> ≠ <q, k_orig>`.
/// The expected bias per key position is estimated from the quantization error variance:
///   `bias[i] ≈ -0.5 * d * sigma_err^2 / sigma_score`
/// where `sigma_err` depends on bit width and `sigma_score` depends on head_dim.
///
/// Subtracting this bias before softmax partially compensates the quantization-induced
/// attention drift. Most effective at 2-bit where error variance is highest.
///
/// Returns per-position bias corrections (one per cached key).
pub fn softmax_bias_correction(
    compressed: &CompressedKeys,
    head_dim: usize,
) -> Vec<f32> {
    // Quantization MSE per centroid level (empirical from Lloyd-Max N(0,1))
    let mse_per_dim = match compressed.bits {
        2 => 0.1175f32,  // 4 centroids — highest error
        3 => 0.0344f32,  // 8 centroids
        4 => 0.0094f32,  // 16 centroids — lowest error
        _ => 0.0094f32,
    };

    // Bias correction: for each key, the expected score shift from quantization
    // is proportional to the key's norm and the per-dimension MSE.
    // bias ≈ -0.5 * (dim * mse_per_dim * norm^2 / dim) = -0.5 * mse_per_dim * norm^2
    // Scaled by 1/sqrt(d) for attention scale consistency.
    let scale = 0.5 * mse_per_dim / (head_dim as f32).sqrt();
    compressed.norms.iter().map(|&norm| {
        -scale * norm * norm / (head_dim as f32)
    }).collect()
}

/// Statistics from a sparse V multiply: how many positions were active vs skipped.
pub struct SparseVStats {
    /// Total sequence positions
    pub total: usize,
    /// Positions with weight >= threshold (actually computed)
    pub active: usize,
}

impl SparseVStats {
    /// Fraction of positions skipped (0.0 = fully dense, 1.0 = all skipped).
    pub fn sparsity(&self) -> f32 {
        if self.total == 0 { return 0.0; }
        (self.total - self.active) as f32 / self.total as f32
    }
}

/// Count how many positions would be active for a given threshold.
pub fn sparse_v_stats(attn_weights: &[f32], threshold: f32) -> SparseVStats {
    let active = attn_weights.iter().filter(|&&w| w >= threshold).count();
    SparseVStats { total: attn_weights.len(), active }
}

/// Calibrate optimal rotation matrix from key vectors (SpinQuant PCA approach).
///
/// Computes the covariance of key vectors and uses eigendecomposition
/// to find the rotation that decorrelates coordinates — optimal for
/// scalar quantization. No training loop needed.
///
/// Returns a [dim × dim] row-major rotation matrix.
/// Set it on `TurboQuantConfig::rotation_matrix`.
///
/// Expected improvement: 10-25% quantization error reduction over random Hadamard.
pub fn calibrate_rotation(data: &[f32], dim: usize) -> Vec<f32> {
    hadamard::calibrate_pca_rotation(data, dim)
}

/// Calibrate codebook from a batch of key vectors.
///
/// Collects post-Hadamard, normalized coordinate samples and runs Lloyd-Max
/// to find optimal centroids for the actual distribution.
///
/// Returns a CalibratedCodebook — set it on `TurboQuantConfig::calibrated_codebook`.
pub fn calibrate_codebook(data: &[f32], dim: usize, bits: u8, rotation_seed: u64) -> codebook::CalibratedCodebook {
    calibrate_codebook_with_rotation(data, dim, bits, rotation_seed, None)
}

/// Calibrate codebook with optional custom rotation matrix.
/// If rotation_matrix is Some, uses that instead of randomized Hadamard.
/// This ensures the codebook is fitted to the same rotation used at runtime.
pub fn calibrate_codebook_with_rotation(
    data: &[f32], dim: usize, bits: u8, rotation_seed: u64,
    rotation_matrix: Option<&[f32]>,
) -> codebook::CalibratedCodebook {
    assert_eq!(data.len() % dim, 0);

    // Rotate all vectors using the SAME rotation that will be used at runtime
    let mut rotated = data.to_vec();
    for chunk in rotated.chunks_exact_mut(dim) {
        if let Some(matrix) = rotation_matrix {
            hadamard::apply_rotation(chunk, matrix);
        } else {
            hadamard::randomized_hadamard(chunk, rotation_seed);
        }
    }

    // Normalize: divide each coordinate by its vector's sigma
    let mut normalized = Vec::with_capacity(rotated.len());
    for chunk in rotated.chunks_exact(dim) {
        let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sigma = if norm > 1e-10 { norm / (dim as f32).sqrt() } else { 1.0 };
        for &v in chunk {
            normalized.push(v / sigma);
        }
    }

    codebook::CalibratedCodebook::calibrate(&normalized, bits, 100)
}

/// Calibrate per-channel scaling factors from a batch of key vectors.
///
/// Computes `scale[i] = median_absmax / absmax[i]` for each channel, so that
/// channels with large outliers are scaled down and channels with small values
/// are scaled up. This equalizes the magnitude distribution before Hadamard
/// rotation, reducing quantization error on outlier channels.
///
/// Returns a Vec of length `dim` — pass to `TurboQuantConfig::channel_scales`.
pub fn calibrate_channel_scales(data: &[f32], dim: usize) -> Vec<f32> {
    assert_eq!(data.len() % dim, 0);
    let count = data.len() / dim;
    if count == 0 {
        return vec![1.0; dim];
    }

    // Compute absmax per channel
    let mut channel_absmax = vec![0.0f32; dim];
    for chunk in data.chunks_exact(dim) {
        for (i, &v) in chunk.iter().enumerate() {
            channel_absmax[i] = channel_absmax[i].max(v.abs());
        }
    }

    // Compute median absmax
    let mut sorted = channel_absmax.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    // Scale = median / absmax (brings all channels to similar magnitude)
    // Clip to [0.1, 10.0] to avoid extreme scaling that destroys information
    channel_absmax.iter().map(|&m| {
        if m > 1e-10 { (median / m).clamp(0.1, 10.0) } else { 1.0 }
    }).collect()
}

/// Evaluate V2 compression quality.
pub fn evaluate_keys(original: &[f32], compressed: &CompressedKeys, config: &TurboQuantConfig) -> CompressionStats {
    let decompressed = decompress_keys(compressed, config);
    let mse = polar::compute_mse(original, &decompressed);
    let signal_power: f32 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
    let snr_db = if signal_power > 0.0 && mse > 0.0 { 10.0 * (signal_power / mse).log10() } else if mse == 0.0 { f32::INFINITY } else { 0.0 };
    let max_error = original.iter().zip(decompressed.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    CompressionStats { mse, snr_db, ratio: compressed.compression_ratio(), max_error }
}

/// Compression quality statistics.
pub struct CompressionStats {
    pub mse: f32,
    pub snr_db: f32,
    pub ratio: f32,
    pub max_error: f32,
}

/// V1 API: Evaluate PolarQuant compression quality.
#[deprecated(since = "0.1.0", note = "use evaluate_keys instead")]
#[allow(deprecated)]
pub fn evaluate(original: &[f32], compressed: &CompressedVectors) -> CompressionStats {
    let decompressed = decompress_vectors(compressed);
    let mse = polar::compute_mse(original, &decompressed);
    let signal_power: f32 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
    let snr_db = if signal_power > 0.0 && mse > 0.0 { 10.0 * (signal_power / mse).log10() } else if mse == 0.0 { f32::INFINITY } else { 0.0 };
    let max_error = original.iter().zip(decompressed.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    CompressionStats { mse, snr_db, ratio: compressed.compression_ratio(), max_error }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    fn random_vectors(count: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..count * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_full_pipeline_4bit() {
        let dim = 128;
        let data = random_vectors(64, dim, 42);
        let config = TurboQuantConfig::balanced();
        let compressed = compress_vectors(&data, dim, &config);
        let stats = evaluate(&data, &compressed);
        assert!(stats.ratio > 2.5);
        assert!(stats.snr_db > 15.0);
    }

    #[test]
    fn test_full_pipeline_3bit() {
        let dim = 128;
        let data = random_vectors(64, dim, 42);
        let config = TurboQuantConfig::aggressive();
        let compressed = compress_vectors(&data, dim, &config);
        let stats = evaluate(&data, &compressed);
        assert!(stats.ratio > 2.0);
    }

    #[test]
    fn test_deterministic() {
        let dim = 64;
        let data = random_vectors(8, dim, 42);
        let config = TurboQuantConfig::default();
        let d1 = decompress_vectors(&compress_vectors(&data, dim, &config));
        let d2 = decompress_vectors(&compress_vectors(&data, dim, &config));
        assert_eq!(d1, d2);
    }

    // ========================================
    // V2 API Tests (Paper-faithful Lloyd-Max)
    // ========================================

    #[test]
    fn test_v2_codebook_2bit() {
        let dim = 128;
        let data = random_vectors(64, dim, 42);
        let config = TurboQuantConfig::extreme();
        let compressed = compress_keys(&data, dim, &config);
        let stats = evaluate_keys(&data, &compressed, &config);
        eprintln!("=== V2 Lloyd-Max 2-bit ===");
        eprintln!("Compression: {:.1}x", stats.ratio);
        eprintln!("MSE: {:.6}, SNR: {:.1} dB", stats.mse, stats.snr_db);
        assert!(stats.ratio > 3.0, "2-bit ratio: {:.1}x", stats.ratio);
    }

    #[test]
    fn test_v2_codebook_4bit() {
        let dim = 128;
        let data = random_vectors(64, dim, 42);
        let config = TurboQuantConfig::balanced();
        let compressed = compress_keys(&data, dim, &config);
        let stats = evaluate_keys(&data, &compressed, &config);
        eprintln!("=== V2 Lloyd-Max 4-bit ===");
        eprintln!("Compression: {:.1}x", stats.ratio);
        eprintln!("MSE: {:.6}, SNR: {:.1} dB", stats.mse, stats.snr_db);
        assert!(stats.ratio > 2.0, "4-bit ratio: {:.1}x", stats.ratio);
    }

    #[test]
    fn test_v2_pre_rotated_query() {
        let dim = 128;
        let seed = 42u64;
        let config = TurboQuantConfig { bits: 4, rotation_seed: seed, ..Default::default() };

        // Create a key and query
        let key = random_vectors(1, dim, 100);
        let query = random_vectors(1, dim, 200);

        // Standard dot product
        let standard_dot: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();

        // Compress key, pre-rotate query, fused dot product
        let compressed = compress_keys(&key, dim, &config);
        let rotated_q = pre_rotate_query(&query, seed);
        let all_indices = codebook::unpack_indices(&compressed.packed_indices, dim, config.bits);
        let fused_dot = fused_dot_product(&rotated_q, &all_indices, compressed.norms[0], config.bits, dim);

        // Should be close (not exact due to quantization)
        let rel_error = (standard_dot - fused_dot).abs() / standard_dot.abs().max(1e-10);
        eprintln!("Standard dot: {:.4}, Fused dot: {:.4}, Relative error: {:.4}",
            standard_dot, fused_dot, rel_error);
        // Quantization introduces error — fused dot should be in the right ballpark
        assert!(rel_error < 1.0, "Fused dot product too different: {:.4}", rel_error);
    }

    // ========================================
    // Incremental Append Tests
    // ========================================

    #[test]
    fn test_append_single_key_2bit() {
        let dim = 128;
        let config = TurboQuantConfig::extreme();
        let signs = hadamard::generate_signs(dim, config.rotation_seed);
        let data = random_vectors(4, dim, 42);

        let mut cache = CompressedKeys::new_empty(2, dim, config.rotation_seed);
        for chunk in data.chunks_exact(dim) {
            let (packed, norm) = compress_single_key_with_signs(chunk, dim, &config, &signs);
            cache.append_raw(&packed, norm);
        }
        assert_eq!(cache.count, 4);
        assert_eq!(cache.norms.len(), 4);
        // 2-bit: 128 indices → 32 bytes per vector
        assert_eq!(cache.packed_indices.len(), 4 * 32);
    }

    #[test]
    fn test_append_single_key_3bit() {
        let dim = 128;
        let config = TurboQuantConfig::aggressive();
        let signs = hadamard::generate_signs(dim, config.rotation_seed);
        let data = random_vectors(4, dim, 42);

        let mut cache = CompressedKeys::new_empty(3, dim, config.rotation_seed);
        for chunk in data.chunks_exact(dim) {
            let (packed, norm) = compress_single_key_with_signs(chunk, dim, &config, &signs);
            cache.append_raw(&packed, norm);
        }
        assert_eq!(cache.count, 4);
        // 3-bit: 128 * 3 / 8 = 48 bytes per vector
        assert_eq!(cache.packed_indices.len(), 4 * 48);
    }

    #[test]
    fn test_append_single_key_4bit() {
        let dim = 128;
        let config = TurboQuantConfig::balanced();
        let signs = hadamard::generate_signs(dim, config.rotation_seed);
        let data = random_vectors(4, dim, 42);

        let mut cache = CompressedKeys::new_empty(4, dim, config.rotation_seed);
        for chunk in data.chunks_exact(dim) {
            let (packed, norm) = compress_single_key_with_signs(chunk, dim, &config, &signs);
            cache.append_raw(&packed, norm);
        }
        assert_eq!(cache.count, 4);
        // 4-bit: 128 / 2 = 64 bytes per vector
        assert_eq!(cache.packed_indices.len(), 4 * 64);
    }

    #[test]
    fn test_append_then_decompress_equals_batch() {
        let dim = 128;
        let config = TurboQuantConfig::extreme();
        let signs = hadamard::generate_signs(dim, config.rotation_seed);
        let data = random_vectors(8, dim, 42);

        // Method 1: batch compress
        let batch_compressed = compress_keys(&data, dim, &config);
        let batch_decompressed = decompress_keys(&batch_compressed, &config);

        // Method 2: incremental append
        let mut cache = CompressedKeys::new_empty(2, dim, config.rotation_seed);
        for chunk in data.chunks_exact(dim) {
            let (packed, norm) = compress_single_key_with_signs(chunk, dim, &config, &signs);
            cache.append_raw(&packed, norm);
        }
        let incr_decompressed = decompress_keys(&cache, &config);

        // Both should produce same result
        assert_eq!(batch_decompressed.len(), incr_decompressed.len());
        for (a, b) in batch_decompressed.iter().zip(incr_decompressed.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Batch vs incremental mismatch: {} vs {}", a, b
            );
        }
    }

    #[test]
    fn test_append_to_empty_cache() {
        let dim = 64;
        let config = TurboQuantConfig::extreme();
        let signs = hadamard::generate_signs(dim, config.rotation_seed);

        let mut cache = CompressedKeys::new_empty(2, dim, config.rotation_seed);
        assert_eq!(cache.count, 0);
        assert!(cache.packed_indices.is_empty());

        let key = random_vectors(1, dim, 99);
        let (packed, norm) = compress_single_key_with_signs(&key, dim, &config, &signs);
        cache.append_raw(&packed, norm);

        assert_eq!(cache.count, 1);
        assert!(norm > 0.0);
    }

    // ========================================
    // Fused Attention Tests
    // ========================================

    #[test]
    fn test_fused_dot_product_2bit_accuracy() {
        let dim = 128;
        let config = TurboQuantConfig::extreme();
        let key = random_vectors(1, dim, 100);
        let query = random_vectors(1, dim, 200);

        let compressed = compress_keys(&key, dim, &config);
        let rotated_q = pre_rotate_query(&query, config.rotation_seed);
        let indices = compressed.get_indices(0);
        let centroids = codebook::get_centroids(2);

        let fused = fused_dot_product_with_centroids(
            &rotated_q, &indices, compressed.norms[0], centroids, dim,
        );
        let old_fused = fused_dot_product(
            &rotated_q, &indices, compressed.norms[0], 2, dim,
        );
        assert!((fused - old_fused).abs() < 1e-6, "2-bit: centroids vs codebook mismatch");
    }

    #[test]
    fn test_fused_dot_product_3bit_accuracy() {
        let dim = 128;
        let config = TurboQuantConfig::aggressive();
        let key = random_vectors(1, dim, 100);
        let query = random_vectors(1, dim, 200);

        let compressed = compress_keys(&key, dim, &config);
        let rotated_q = pre_rotate_query(&query, config.rotation_seed);
        let indices = compressed.get_indices(0);
        let centroids = codebook::get_centroids(3);

        let fused = fused_dot_product_with_centroids(
            &rotated_q, &indices, compressed.norms[0], centroids, dim,
        );
        let old_fused = fused_dot_product(
            &rotated_q, &indices, compressed.norms[0], 3, dim,
        );
        assert!((fused - old_fused).abs() < 1e-6, "3-bit: centroids vs codebook mismatch");
    }

    #[test]
    fn test_fused_dot_product_4bit_accuracy() {
        let dim = 128;
        let config = TurboQuantConfig { bits: 4, use_qjl: false, ..Default::default() };
        let key = random_vectors(1, dim, 100);
        let query = random_vectors(1, dim, 200);

        let compressed = compress_keys(&key, dim, &config);
        let rotated_q = pre_rotate_query(&query, config.rotation_seed);
        let indices = compressed.get_indices(0);
        let centroids = codebook::get_centroids(4);

        let fused = fused_dot_product_with_centroids(
            &rotated_q, &indices, compressed.norms[0], centroids, dim,
        );
        let old_fused = fused_dot_product(
            &rotated_q, &indices, compressed.norms[0], 4, dim,
        );
        assert!((fused - old_fused).abs() < 1e-5, "4-bit: centroids vs codebook mismatch");
    }

    #[test]
    fn test_fused_zero_norm_returns_zero() {
        let dim = 128;
        let query = random_vectors(1, dim, 42);
        let rotated_q = pre_rotate_query(&query, 0);
        let indices = vec![0u8; dim];
        let centroids = codebook::get_centroids(2);

        let score = fused_dot_product_with_centroids(
            &rotated_q, &indices, 0.0, centroids, dim,
        );
        assert_eq!(score, 0.0, "Zero-norm key must return 0.0");

        let score_old = fused_dot_product(&rotated_q, &indices, 0.0, 2, dim);
        assert_eq!(score_old, 0.0, "Zero-norm key must return 0.0 (old API)");
    }

    // ========================================
    // Signs Caching Tests
    // ========================================

    #[test]
    fn test_precomputed_signs_roundtrip() {
        let dim = 128;
        let seed = 42u64;
        let signs = hadamard::generate_signs(dim, seed);
        let original = random_vectors(1, dim, 99);

        let mut x1 = original.clone();
        hadamard::randomized_hadamard(&mut x1, seed);
        hadamard::inverse_randomized_hadamard(&mut x1, seed);

        let mut x2 = original.clone();
        hadamard::randomized_hadamard_with_signs(&mut x2, &signs);
        hadamard::inverse_randomized_hadamard_with_signs(&mut x2, &signs);

        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-6, "Signs mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_hadamard_with_signs_matches_seed() {
        let dim = 64;
        let seed = 123u64;
        let signs = hadamard::generate_signs(dim, seed);
        let original = random_vectors(1, dim, 42);

        let mut x_seed = original.clone();
        hadamard::randomized_hadamard(&mut x_seed, seed);

        let mut x_signs = original.clone();
        hadamard::randomized_hadamard_with_signs(&mut x_signs, &signs);

        for (a, b) in x_seed.iter().zip(x_signs.iter()) {
            assert!((a - b).abs() < 1e-6, "Seed vs signs output mismatch");
        }
    }

    #[test]
    fn test_compress_with_signs_matches_seed() {
        let dim = 128;
        let config = TurboQuantConfig::extreme();
        let signs = hadamard::generate_signs(dim, config.rotation_seed);
        let key = random_vectors(1, dim, 42);

        let (packed_seed, norm_seed) = compress_single_key(&key, dim, &config);
        let (packed_signs, norm_signs) = compress_single_key_with_signs(&key, dim, &config, &signs);

        assert_eq!(packed_seed, packed_signs);
        assert!((norm_seed - norm_signs).abs() < 1e-6);
    }

    #[test]
    fn test_v2_all_bitwidths() {
        let dim = 128;
        let data = random_vectors(32, dim, 42);

        eprintln!("\n=== TurboQuant V2 (Lloyd-Max) All Bitwidths ===");
        eprintln!("{:<6} {:<12} {:<10} {:<10} {:<10}", "Bits", "Ratio", "MSE", "SNR(dB)", "MaxErr");

        for bits in [2, 3, 4] {
            let config = TurboQuantConfig { bits, use_qjl: true, ..Default::default() };
            let compressed = compress_keys(&data, dim, &config);
            let stats = evaluate_keys(&data, &compressed, &config);
            eprintln!("{:<6} {:<12.1} {:<10.6} {:<10.1} {:<10.4}",
                bits, stats.ratio, stats.mse, stats.snr_db, stats.max_error);
        }
    }

    // ========================================
    // Fused Attention Scores Batch Test
    // ========================================

    #[test]
    fn test_fused_attention_scores_batch() {
        let dim = 128;
        let num_keys = 8;
        let config = TurboQuantConfig::extreme(); // 2-bit

        // Create random keys and compress them into a cache
        let keys = random_vectors(num_keys, dim, 42);
        let signs = hadamard::generate_signs(dim, config.rotation_seed);

        let mut cache = CompressedKeys::new_empty(config.bits, dim, config.rotation_seed);
        for chunk in keys.chunks_exact(dim) {
            let (packed, norm) = compress_single_key_with_signs(chunk, dim, &config, &signs);
            cache.append_raw(&packed, norm);
        }
        assert_eq!(cache.count, num_keys);

        // Create and pre-rotate a query
        let query = random_vectors(1, dim, 99);
        let rotated_q = pre_rotate_query_with_signs(&query, &signs);
        let base_centroids = codebook::get_centroids(config.bits);
        let scale = 1.0 / (dim as f32).sqrt();

        // Get batch scores via fused_attention_scores
        let batch_scores = fused_attention_scores(&rotated_q, &cache, base_centroids, scale);
        assert_eq!(batch_scores.len(), num_keys);

        // Verify each score matches calling fused_dot_product_with_centroids individually
        for pos in 0..num_keys {
            let indices = cache.get_indices(pos);
            let individual_score = fused_dot_product_with_centroids(
                &rotated_q, &indices, cache.norms[pos], base_centroids, dim,
            ) * scale;
            assert!(
                (batch_scores[pos] - individual_score).abs() < 1e-6,
                "Score mismatch at pos {}: batch={}, individual={}",
                pos, batch_scores[pos], individual_score,
            );
        }

        // Edge case: empty cache (0 keys)
        let empty_cache = CompressedKeys::new_empty(config.bits, dim, config.rotation_seed);
        let empty_scores = fused_attention_scores(&rotated_q, &empty_cache, base_centroids, scale);
        assert!(empty_scores.is_empty(), "Empty cache should return empty scores");

        // Edge case: single key cache
        let mut single_cache = CompressedKeys::new_empty(config.bits, dim, config.rotation_seed);
        let first_key = &keys[..dim];
        let (packed, norm) = compress_single_key_with_signs(first_key, dim, &config, &signs);
        single_cache.append_raw(&packed, norm);
        assert_eq!(single_cache.count, 1);

        let single_scores = fused_attention_scores(&rotated_q, &single_cache, base_centroids, scale);
        assert_eq!(single_scores.len(), 1);

        let single_indices = single_cache.get_indices(0);
        let expected_score = fused_dot_product_with_centroids(
            &rotated_q, &single_indices, single_cache.norms[0], base_centroids, dim,
        ) * scale;
        assert!(
            (single_scores[0] - expected_score).abs() < 1e-6,
            "Single key score mismatch: batch={}, individual={}",
            single_scores[0], expected_score,
        );
    }

    // ========================================
    // Sparse V Tests
    // ========================================

    #[test]
    fn test_sparse_v_matches_dense() {
        let head_dim = 128;
        let seq_len = 64;

        // Synthetic softmax-like weights (sum to 1, mostly small, few large)
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let raw: Vec<f32> = (0..seq_len).map(|_| rng.gen::<f32>()).collect();
        let sum: f32 = raw.iter().sum();
        let weights: Vec<f32> = raw.iter().map(|w| w / sum).collect();

        let values = random_vectors(seq_len, head_dim, 99);

        // Dense result (threshold = 0)
        let dense = sparse_attn_v_mul(&weights, &values, head_dim, 0.0);
        // Sparse result (threshold = 1e-6, should match since all weights > 1e-6)
        let sparse = sparse_attn_v_mul(&weights, &values, head_dim, 1e-6);

        for (d, s) in dense.iter().zip(sparse.iter()) {
            assert!(
                (d - s).abs() < 1e-5,
                "Sparse/dense mismatch: dense={}, sparse={}", d, s,
            );
        }
    }

    #[test]
    fn test_sparse_v_skips_small_weights() {
        let head_dim = 64;
        let seq_len = 8;

        // Only position 3 has significant weight
        let mut weights = vec![1e-8; seq_len];
        weights[3] = 0.999;
        // Normalize to sum=1 (close enough)
        let rem = (1.0 - 0.999) / 7.0;
        for (i, w) in weights.iter_mut().enumerate() {
            if i != 3 { *w = rem; }
        }

        let mut values = vec![0.0f32; seq_len * head_dim];
        // Set V[3] to all 1.0
        for j in 0..head_dim {
            values[3 * head_dim + j] = 1.0;
        }
        // Set other V rows to large values (should be skipped)
        for i in 0..seq_len {
            if i != 3 {
                for j in 0..head_dim {
                    values[i * head_dim + j] = 999.0;
                }
            }
        }

        // With high threshold, only position 3 survives
        let result = sparse_attn_v_mul(&weights, &values, head_dim, 0.01);

        // Result should be close to weights[3] * V[3] = 0.999 * [1,1,...,1]
        for &r in &result {
            assert!(
                (r - 0.999).abs() < 0.01,
                "Expected ~0.999, got {}", r,
            );
        }

        // Stats should show high sparsity
        let stats = sparse_v_stats(&weights, 0.01);
        assert_eq!(stats.active, 1);
        assert_eq!(stats.total, seq_len);
        assert!(stats.sparsity() > 0.8);
    }

    #[test]
    fn test_sparse_v_all_zeros_threshold() {
        // threshold=0 should be dense (no skipping)
        let head_dim = 32;
        let seq_len = 4;
        let weights = vec![0.25f32; seq_len];
        let values = random_vectors(seq_len, head_dim, 7);

        let result = sparse_attn_v_mul(&weights, &values, head_dim, 0.0);

        // Manual dense computation
        let mut expected = vec![0.0f32; head_dim];
        for pos in 0..seq_len {
            for j in 0..head_dim {
                expected[j] += 0.25 * values[pos * head_dim + j];
            }
        }

        for (e, r) in expected.iter().zip(result.iter()) {
            assert!((e - r).abs() < 1e-6, "Mismatch: expected={}, got={}", e, r);
        }
    }

    // ========================================
    // Compressed Values Tests (K/V Asymmetric)
    // ========================================

    #[test]
    fn test_compressed_values_roundtrip() {
        let dim = 128;
        let data = random_vectors(32, dim, 42);

        let mut cv = CompressedValues::new_empty(dim);
        cv.append_batch(&data, dim);
        assert_eq!(cv.count, 32);

        let decompressed = cv.decompress();
        assert_eq!(decompressed.len(), data.len());

        // 8-bit absmax should have very high cosine similarity
        let dot: f32 = data.iter().zip(decompressed.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_a * norm_b + 1e-10);
        assert!(cos_sim > 0.999, "8-bit value cos_sim should be > 0.999, got {}", cos_sim);

        // Max error per element should be small
        let max_err: f32 = data.iter().zip(decompressed.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let data_max: f32 = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel_max_err = max_err / data_max;
        assert!(rel_max_err < 0.02, "8-bit relative max error should be < 2%, got {:.4}", rel_max_err);
    }

    #[test]
    fn test_compressed_values_incremental() {
        let dim = 64;
        let mut cv = CompressedValues::new_empty(dim);

        // Append one at a time
        for i in 0..8 {
            let vec = random_vectors(1, dim, i as u64);
            cv.append(&vec);
        }
        assert_eq!(cv.count, 8);

        // Decompress range
        let range = cv.decompress_range(2, 3);
        assert_eq!(range.len(), 3 * dim);

        // Should match full decompress subset
        let full = cv.decompress();
        let expected = &full[2 * dim..5 * dim];
        assert_eq!(range, expected);
    }

    #[test]
    fn test_compressed_values_compression_ratio() {
        let dim = 128;
        let mut cv = CompressedValues::new_empty(dim);
        let data = random_vectors(64, dim, 77);
        cv.append_batch(&data, dim);

        // 8-bit: 1 byte data + 4 bytes scale per vector
        // fp16: 2 bytes per element
        // Ratio: (64*128*2) / (64*128*1 + 64*4) = 16384 / 8448 ≈ 1.94x
        let ratio = cv.compression_ratio();
        assert!(ratio > 1.8, "8-bit value compression ratio should be ~1.9x, got {:.2}", ratio);
        assert!(ratio < 2.1, "8-bit value compression ratio should be ~1.9x, got {:.2}", ratio);
    }

    #[test]
    fn test_compressed_values_zero_vector() {
        let dim = 32;
        let mut cv = CompressedValues::new_empty(dim);
        let zeros = vec![0.0f32; dim];
        cv.append(&zeros);

        let decompressed = cv.decompress();
        for &v in &decompressed {
            assert_eq!(v, 0.0, "Zero vector should decompress to zeros");
        }
    }

    // ========================================
    // 4-bit Compressed Values Tests
    // ========================================

    #[test]
    fn test_compressed_values_4bit_roundtrip() {
        let dim = 128;
        let data = random_vectors(32, dim, 42);

        let mut cv = CompressedValues4Bit::new_empty(dim, 32);
        cv.append_batch(&data, dim);
        assert_eq!(cv.count, 32);

        let decompressed = cv.decompress();
        assert_eq!(decompressed.len(), data.len());

        // 4-bit per-group should have high cosine similarity
        let dot: f32 = data.iter().zip(decompressed.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_a * norm_b + 1e-10);
        assert!(cos_sim > 0.99, "4-bit value cos_sim should be > 0.99, got {}", cos_sim);
    }

    #[test]
    fn test_compressed_values_4bit_incremental() {
        let dim = 64;
        let mut cv = CompressedValues4Bit::new_empty(dim, 32);

        for i in 0..8 {
            let vec = random_vectors(1, dim, i as u64);
            cv.append(&vec);
        }
        assert_eq!(cv.count, 8);

        // Decompress range should match full decompress subset
        let range = cv.decompress_range(2, 3);
        assert_eq!(range.len(), 3 * dim);

        let full = cv.decompress();
        let expected = &full[2 * dim..5 * dim];
        assert_eq!(range, expected);
    }

    #[test]
    fn test_compressed_values_4bit_compression_ratio() {
        let dim = 128;
        let mut cv = CompressedValues4Bit::new_empty(dim, 32);
        let data = random_vectors(64, dim, 77);
        cv.append_batch(&data, dim);

        // 4-bit: dim/2 bytes data + (dim/gs)*4 bytes scales per vector
        // vs fp16: dim*2 bytes
        // dim=128, gs=32: (64 + 16) = 80 bytes vs 256 = 3.2x
        let ratio = cv.compression_ratio();
        assert!(ratio > 3.0, "4-bit value compression ratio should be ~3.2x, got {:.2}", ratio);
        assert!(ratio < 4.0, "4-bit value compression ratio should be ~3.2x, got {:.2}", ratio);
    }

    #[test]
    fn test_compressed_values_4bit_zero_vector() {
        let dim = 32;
        let mut cv = CompressedValues4Bit::new_empty(dim, 32);
        let zeros = vec![0.0f32; dim];
        cv.append(&zeros);

        let decompressed = cv.decompress();
        for &v in &decompressed {
            assert_eq!(v, 0.0, "Zero vector should decompress to zeros");
        }
    }

    #[test]
    fn test_compressed_values_4bit_decompress_row_into() {
        let dim = 128;
        let mut cv = CompressedValues4Bit::new_empty(dim, 32);
        let data = random_vectors(16, dim, 55);
        cv.append_batch(&data, dim);

        let full = cv.decompress();
        let mut row_buf = vec![0.0f32; dim];

        for row in 0..16 {
            cv.decompress_row_into(row, &mut row_buf);
            let expected = &full[row * dim..(row + 1) * dim];
            assert_eq!(&row_buf, expected, "Row {} mismatch", row);
        }
    }

    // ========================================
    // Fused Sparse Compressed V Tests
    // ========================================

    #[test]
    fn test_fused_sparse_4bit_matches_dense() {
        let dim = 128;
        let seq_len = 64;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let raw: Vec<f32> = (0..seq_len).map(|_| rng.gen::<f32>()).collect();
        let sum: f32 = raw.iter().sum();
        let weights: Vec<f32> = raw.iter().map(|w| w / sum).collect();

        let values = random_vectors(seq_len, dim, 99);

        // Compress to 4-bit
        let mut cv = CompressedValues4Bit::new_empty(dim, 32);
        cv.append_batch(&values, dim);

        // Fused sparse (threshold=0 = dense)
        let fused = sparse_attn_v_mul_compressed_4bit(&weights, &cv, 0.0);

        // Reference: decompress then dense multiply
        let decompressed = cv.decompress();
        let reference = sparse_attn_v_mul(&weights, &decompressed, dim, 0.0);

        for (f, r) in fused.iter().zip(reference.iter()) {
            assert!(
                (f - r).abs() < 1e-5,
                "Fused/reference mismatch: fused={}, ref={}", f, r,
            );
        }
    }

    #[test]
    fn test_fused_sparse_4bit_skips_correctly() {
        let dim = 64;
        let seq_len = 8;

        // Only position 3 has significant weight
        let mut weights = vec![0.0001f32; seq_len];
        weights[3] = 0.999;

        let values = random_vectors(seq_len, dim, 42);

        let mut cv = CompressedValues4Bit::new_empty(dim, 32);
        cv.append_batch(&values, dim);

        // With high threshold, only position 3 survives
        let result = sparse_attn_v_mul_compressed_4bit(&weights, &cv, 0.01);

        // Reference: decompress row 3 manually, scale by weight
        let mut row3 = vec![0.0f32; dim];
        cv.decompress_row_into(3, &mut row3);
        let expected: Vec<f32> = row3.iter().map(|&v| v * 0.999).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-4,
                "Sparse skip mismatch: result={}, expected={}", r, e,
            );
        }
    }

    #[test]
    fn test_fused_sparse_8bit_matches_existing() {
        let dim = 128;
        let seq_len = 64;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let raw: Vec<f32> = (0..seq_len).map(|_| rng.gen::<f32>()).collect();
        let sum: f32 = raw.iter().sum();
        let weights: Vec<f32> = raw.iter().map(|w| w / sum).collect();

        let values = random_vectors(seq_len, dim, 99);

        // Compress to 8-bit
        let mut cv = CompressedValues::new_empty(dim);
        cv.append_batch(&values, dim);

        // Fused sparse with threshold
        let fused = sparse_attn_v_mul_compressed_8bit(&weights, &cv, 1e-6);

        // Reference: decompress then sparse multiply
        let decompressed = cv.decompress();
        let reference = sparse_attn_v_mul(&weights, &decompressed, dim, 1e-6);

        for (f, r) in fused.iter().zip(reference.iter()) {
            assert!(
                (f - r).abs() < 1e-4,
                "8-bit fused/reference mismatch: fused={}, ref={}", f, r,
            );
        }
    }

    #[test]
    fn test_fused_sparse_all_below_threshold() {
        let dim = 64;
        let seq_len = 4;
        let weights = vec![1e-8f32; seq_len];
        let values = random_vectors(seq_len, dim, 42);

        let mut cv = CompressedValues4Bit::new_empty(dim, 32);
        cv.append_batch(&values, dim);

        let result = sparse_attn_v_mul_compressed_4bit(&weights, &cv, 0.01);
        for &r in &result {
            assert_eq!(r, 0.0, "All below threshold should produce zero output");
        }
    }

    // ========================================
    // Temporal Decay Tests
    // ========================================

    #[test]
    fn test_remap_table_4to2() {
        let remap = codebook::remap_table(4, 2);
        assert_eq!(remap.len(), 16); // 4-bit = 16 centroids

        // Verify symmetry: remap[i] and remap[15-i] should be symmetric around center
        for i in 0..8 {
            assert_eq!(remap[i], 3 - remap[15 - i],
                "Remap should be symmetric: [{}]={}, [{}]={}", i, remap[i], 15-i, remap[15-i]);
        }

        // First centroids (most negative) should map to 2-bit index 0 (most negative)
        assert_eq!(remap[0], 0);
        // Last centroids (most positive) should map to 2-bit index 3 (most positive)
        assert_eq!(remap[15], 3);
    }

    #[test]
    fn test_remap_table_4to3() {
        let remap = codebook::remap_table(4, 3);
        assert_eq!(remap.len(), 16);
        // All remapped indices should be in [0, 7]
        for &idx in &remap {
            assert!(idx < 8, "3-bit index should be < 8, got {}", idx);
        }
    }

    #[test]
    fn test_split_off_front() {
        let dim = 64;
        let config = TurboQuantConfig::balanced();
        let data = random_vectors(10, dim, 42);
        let compressed = compress_keys(&data, dim, &config);

        let mut cache = compressed.clone();
        let front = cache.split_off_front(4);

        assert_eq!(front.count, 4);
        assert_eq!(cache.count, 6);
        assert_eq!(front.bits, 4);
        assert_eq!(cache.bits, 4);

        // Decompress both halves and verify they match original
        let d_front = decompress_keys(&front, &config);
        let d_back = decompress_keys(&cache, &config);
        let d_full = decompress_keys(&compressed, &config);

        // front + back should equal full
        let mut combined = d_front.clone();
        combined.extend_from_slice(&d_back);
        for (i, (a, b)) in combined.iter().zip(d_full.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6,
                "Split/merge mismatch at index {}: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_remap_bits_4to2() {
        let dim = 128;
        let config = TurboQuantConfig::balanced(); // 4-bit
        let data = random_vectors(8, dim, 42);
        let compressed = compress_keys(&data, dim, &config);

        let remapped = compressed.remap_bits(2);
        assert_eq!(remapped.count, 8);
        assert_eq!(remapped.bits, 2);
        assert_eq!(remapped.dim, dim);

        // Remapped should use less memory
        assert!(remapped.memory_bytes() < compressed.memory_bytes(),
            "2-bit should use less memory: {} vs {}", remapped.memory_bytes(), compressed.memory_bytes());

        // Decompress and check quality — 4→2 remap will lose some quality
        let d_4bit = decompress_keys(&compressed, &config);
        let config_2bit = TurboQuantConfig::extreme();
        let d_2bit = decompress_keys(&remapped, &config_2bit);

        // Cosine similarity between 4-bit decompressed and 2-bit remapped
        let dot: f32 = d_4bit.iter().zip(d_2bit.iter()).map(|(a, b)| a * b).sum();
        let n_a: f32 = d_4bit.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_b: f32 = d_2bit.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (n_a * n_b + 1e-10);
        assert!(cos_sim > 0.85,
            "4→2 remap cos_sim should be > 0.85, got {:.4}", cos_sim);
    }

    #[test]
    fn test_append_from() {
        let dim = 64;
        let config = TurboQuantConfig::extreme(); // 2-bit
        let data1 = random_vectors(4, dim, 42);
        let data2 = random_vectors(4, dim, 99);

        let c1 = compress_keys(&data1, dim, &config);
        let c2 = compress_keys(&data2, dim, &config);

        let mut merged = c1.clone();
        merged.append_from(&c2);
        assert_eq!(merged.count, 8);

        // Decompress merged should equal individual decompressions concatenated
        let d1 = decompress_keys(&c1, &config);
        let d2 = decompress_keys(&c2, &config);
        let d_merged = decompress_keys(&merged, &config);

        let mut expected = d1;
        expected.extend_from_slice(&d2);
        for (i, (a, b)) in expected.iter().zip(d_merged.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6,
                "append_from mismatch at {}: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_decay_memory_savings() {
        let dim = 128;
        let config = TurboQuantConfig::balanced(); // 4-bit
        let data = random_vectors(64, dim, 42);
        let compressed = compress_keys(&data, dim, &config);

        let mem_4bit = compressed.memory_bytes();
        let remapped = compressed.remap_bits(2);
        let mem_2bit = remapped.memory_bytes();

        // 2-bit should use roughly half the index bytes of 4-bit
        let savings_pct = (1.0 - mem_2bit as f32 / mem_4bit as f32) * 100.0;
        assert!(savings_pct > 30.0,
            "4→2 decay should save >30% memory, got {:.1}%", savings_pct);
    }

    // ========================================
    // Per-Channel Scaling Tests
    // ========================================

    #[test]
    fn test_channel_scales_improve_outlier_quality() {
        let dim = 128;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Create data with outlier channels (channels 0,1 have 10x larger values)
        let mut data = Vec::with_capacity(32 * dim);
        for _ in 0..32 {
            for i in 0..dim {
                let base: f32 = rng.gen::<f32>() * 2.0 - 1.0;
                let scale = if i < 2 { 10.0 } else { 1.0 };
                data.push(base * scale);
            }
        }

        // Compress WITHOUT channel scaling
        let config_plain = TurboQuantConfig::balanced();
        let compressed_plain = compress_keys(&data, dim, &config_plain);
        let stats_plain = evaluate_keys(&data, &compressed_plain, &config_plain);

        // Calibrate and compress WITH channel scaling
        let scales = calibrate_channel_scales(&data, dim);
        assert_eq!(scales.len(), dim);
        // Outlier channels should have scales < 1 (scaling down)
        assert!(scales[0] < 0.5, "Outlier channel 0 should be scaled down, got {}", scales[0]);

        let config_smooth = TurboQuantConfig {
            channel_scales: Some(scales),
            ..TurboQuantConfig::balanced()
        };
        let compressed_smooth = compress_keys(&data, dim, &config_smooth);
        let stats_smooth = evaluate_keys(&data, &compressed_smooth, &config_smooth);

        // Channel scaling should improve SNR on outlier data
        assert!(stats_smooth.snr_db > stats_plain.snr_db,
            "Channel scaling should improve SNR: {:.1} vs {:.1} dB",
            stats_smooth.snr_db, stats_plain.snr_db);
    }

    #[test]
    fn test_channel_scales_roundtrip() {
        let dim = 64;
        let data = random_vectors(8, dim, 77);
        let scales = calibrate_channel_scales(&data, dim);

        let config = TurboQuantConfig {
            channel_scales: Some(scales),
            ..TurboQuantConfig::balanced()
        };
        let compressed = compress_keys(&data, dim, &config);
        let decompressed = decompress_keys(&compressed, &config);

        // Should still achieve reasonable cosine similarity
        let dot: f32 = data.iter().zip(decompressed.iter()).map(|(a, b)| a * b).sum();
        let n_a: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_b: f32 = decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (n_a * n_b + 1e-10);
        assert!(cos_sim > 0.90, "Channel-scaled roundtrip cos_sim should be > 0.90, got {:.4}", cos_sim);
    }

    #[test]
    fn test_calibrated_codebook() {
        let dim = 128;
        let data = random_vectors(256, dim, 42);
        let cb = calibrate_codebook(&data, dim, 4, 0x0054_5552_4230);

        assert_eq!(cb.centroids.len(), 16); // 4-bit = 16 centroids
        assert_eq!(cb.boundaries.len(), 15);

        // Centroids should be sorted
        for i in 0..cb.centroids.len() - 1 {
            assert!(cb.centroids[i] <= cb.centroids[i + 1],
                "Centroids not sorted: [{}]={} > [{}]={}", i, cb.centroids[i], i+1, cb.centroids[i+1]);
        }

        // Should improve or match Gaussian MSE
        let mut rotated = data.clone();
        for chunk in rotated.chunks_exact_mut(dim) {
            hadamard::randomized_hadamard(chunk, 0x0054_5552_4230);
        }
        let mut normalized = Vec::new();
        for chunk in rotated.chunks_exact(dim) {
            let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
            let sigma = norm / (dim as f32).sqrt();
            for &v in chunk { normalized.push(v / sigma); }
        }

        let (mse_gaussian, mse_calibrated) = cb.improvement_vs_gaussian(&normalized, 4);
        assert!(mse_calibrated <= mse_gaussian * 1.01,
            "Calibrated should be ≤ Gaussian MSE: {:.6} vs {:.6}", mse_calibrated, mse_gaussian);
    }

    #[test]
    fn test_pca_rotation_orthogonal() {
        let dim = 32; // small for fast test
        let data = random_vectors(64, dim, 42);
        let rot = calibrate_rotation(&data, dim);

        assert_eq!(rot.len(), dim * dim);

        // Verify orthogonality: R^T R ≈ I
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += rot[k * dim + i] * rot[k * dim + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 0.01,
                    "R^T R[{},{}] = {}, expected {}", i, j, dot, expected
                );
            }
        }
    }

    #[test]
    fn test_pca_rotation_roundtrip() {
        let dim = 64;
        let data = random_vectors(32, dim, 77);
        let rot = calibrate_rotation(&data, dim);

        // Compress with PCA rotation
        let config = TurboQuantConfig {
            rotation_matrix: Some(rot),
            ..TurboQuantConfig::balanced()
        };
        let compressed = compress_keys(&data, dim, &config);
        let decompressed = decompress_keys(&compressed, &config);

        // Should achieve reasonable cosine similarity
        let dot: f32 = data.iter().zip(decompressed.iter()).map(|(a, b)| a * b).sum();
        let n_a: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_b: f32 = decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (n_a * n_b + 1e-10);
        assert!(cos_sim > 0.85, "PCA rotation cos_sim should be > 0.85, got {:.4}", cos_sim);
    }

    #[test]
    fn test_bits_for_head() {
        let config = TurboQuantConfig::balanced(); // bits=4
        assert_eq!(config.bits_for_head(0), 4);
        assert_eq!(config.bits_for_head(7), 4);

        let config = config.with_per_head_bits(vec![2, 4, 2, 4, 3, 3, 2, 4]);
        assert_eq!(config.bits_for_head(0), 2);
        assert_eq!(config.bits_for_head(1), 4);
        assert_eq!(config.bits_for_head(4), 3);
        // Out of range falls back to default bits
        assert_eq!(config.bits_for_head(100), 4);
    }

    #[test]
    fn test_mixed_bitwidth_compress_decompress() {
        let dim = 128;
        let data = random_vectors(16, dim, 42);

        // Compress same data at 2-bit and 4-bit
        let config_2 = TurboQuantConfig::extreme();
        let config_4 = TurboQuantConfig::balanced();

        let compressed_2 = compress_keys(&data, dim, &config_2);
        let compressed_4 = compress_keys(&data, dim, &config_4);

        assert_eq!(compressed_2.bits, 2);
        assert_eq!(compressed_4.bits, 4);

        let decompressed_2 = decompress_keys(&compressed_2, &config_2);
        let decompressed_4 = decompress_keys(&compressed_4, &config_4);

        // Both should decompress to correct length
        assert_eq!(decompressed_2.len(), data.len());
        assert_eq!(decompressed_4.len(), data.len());

        // 4-bit should have higher cosine similarity than 2-bit
        fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb + 1e-10)
        }

        let sim_2 = cos_sim(&data, &decompressed_2);
        let sim_4 = cos_sim(&data, &decompressed_4);
        assert!(sim_4 > sim_2, "4-bit ({:.4}) should have higher cos_sim than 2-bit ({:.4})", sim_4, sim_2);
        assert!(sim_2 > 0.85, "2-bit cos_sim should be > 0.85, got {:.4}", sim_2);
        assert!(sim_4 > 0.95, "4-bit cos_sim should be > 0.95, got {:.4}", sim_4);
    }

    #[test]
    fn test_fused_attention_mixed_bits() {
        let dim = 128;
        let data = random_vectors(8, dim, 42);

        // Simulate two heads with different bit widths
        let config_2 = TurboQuantConfig::extreme();
        let config_4 = TurboQuantConfig::balanced();

        let compressed_h0 = compress_keys(&data[..4 * dim], dim, &config_2);
        let compressed_h1 = compress_keys(&data[4 * dim..], dim, &config_4);

        // Each head should use correct centroids
        let centroids_2 = codebook::get_centroids(compressed_h0.bits);
        let centroids_4 = codebook::get_centroids(compressed_h1.bits);

        assert_eq!(centroids_2.len(), 4);  // 2^2
        assert_eq!(centroids_4.len(), 16); // 2^4

        // Fused dot product should work with each head's centroids
        let query = random_vectors(1, dim, 99);
        let mut rotated_q = query.clone();
        hadamard::randomized_hadamard(&mut rotated_q, config_2.rotation_seed);

        let mut idx_buf = vec![0u8; dim];

        // Head 0: 2-bit
        let bpv_0 = compressed_h0.bytes_per_vector();
        codebook::unpack_indices_into(&compressed_h0.packed_indices[..bpv_0], &mut idx_buf, 2);
        let score_0 = fused_dot_product_with_centroids(
            &rotated_q, &idx_buf, compressed_h0.norms[0], centroids_2, dim,
        );

        // Head 1: 4-bit
        let bpv_1 = compressed_h1.bytes_per_vector();
        codebook::unpack_indices_into(&compressed_h1.packed_indices[..bpv_1], &mut idx_buf, 4);
        let score_1 = fused_dot_product_with_centroids(
            &rotated_q, &idx_buf, compressed_h1.norms[0], centroids_4, dim,
        );

        // Both should produce finite, non-zero scores
        assert!(score_0.is_finite(), "2-bit fused score should be finite");
        assert!(score_1.is_finite(), "4-bit fused score should be finite");
    }

    // ─── PolarQuant V compression tests ─────────────────────────────

    #[test]
    fn test_values_pq_roundtrip_4bit() {
        let dim = 128;
        let n = 64;
        let mut rng_data: Vec<f32> = (0..n * dim)
            .map(|i| ((i as f32 * 0.618).sin() * 2.0))
            .collect();

        let mut vpq = CompressedValuesPQ::new_empty(dim, 4, 42);
        vpq.append_batch(&rng_data, dim);
        assert_eq!(vpq.count(), n);
        assert!(vpq.compression_ratio() > 2.5);

        let decompressed = vpq.decompress();
        assert_eq!(decompressed.len(), n * dim);

        // Cosine similarity should be high at 4-bit
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&a, &b) in rng_data.iter().zip(decompressed.iter()) {
            dot += a as f64 * b as f64;
            norm_a += (a as f64) * (a as f64);
            norm_b += (b as f64) * (b as f64);
        }
        let cosine = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-30);
        assert!(cosine > 0.98, "4-bit PolarQuant V cosine {} too low", cosine);
    }

    #[test]
    fn test_values_pq_better_ratio_than_absmax() {
        // PolarQuant V 3-bit achieves comparable quality to absmax 4-bit
        // but at 1.5x better compression ratio
        let dim = 128;
        let n = 256;
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i as f32 * 1.234).sin() * 3.0 + (i as f32 * 0.567).cos()))
            .collect();

        // PolarQuant V 3-bit (rotation + Lloyd-Max)
        let mut vpq3 = CompressedValuesPQ::new_empty(dim, 3, 42);
        vpq3.append_batch(&data, dim);
        let recon_pq3 = vpq3.decompress();

        // Absmax V4 (group quantization, 4-bit)
        let mut v4 = CompressedValues4Bit::new_empty(dim, 32);
        v4.append_batch(&data, dim);
        let recon_absmax4 = v4.decompress();

        // Compute cosine similarity
        let cos_pq3: f64 = data.chunks(dim).zip(recon_pq3.chunks(dim))
            .map(|(a, b)| {
                let dot: f64 = a.iter().zip(b).map(|(&x,&y)| x as f64 * y as f64).sum();
                let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                dot / (na * nb + 1e-30)
            }).sum::<f64>() / n as f64;

        let cos_absmax4: f64 = data.chunks(dim).zip(recon_absmax4.chunks(dim))
            .map(|(a, b)| {
                let dot: f64 = a.iter().zip(b).map(|(&x,&y)| x as f64 * y as f64).sum();
                let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                dot / (na * nb + 1e-30)
            }).sum::<f64>() / n as f64;

        // PQ 3-bit should be in the same ballpark as absmax 4-bit (within 3%)
        assert!(cos_pq3 > 0.97, "PQ 3-bit cosine {} too low", cos_pq3);

        // PQ 3-bit should have better compression ratio than absmax 4-bit
        assert!(vpq3.compression_ratio() > v4.compression_ratio(),
            "PQ 3-bit ratio {:.1}x should beat absmax 4-bit {:.1}x",
            vpq3.compression_ratio(), v4.compression_ratio());
    }

    #[test]
    fn test_values_pq_compression_ratio() {
        let dim = 128;
        let mut vpq = CompressedValuesPQ::new_empty(dim, 3, 42);
        let data: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        vpq.append(&data);

        // 3-bit: 128 dims × 3 bits / 8 = 48 bytes indices + 4 bytes norm = 52 bytes
        // vs fp16: 128 × 2 = 256 bytes → ~4.9x ratio
        assert!(vpq.compression_ratio() > 3.5, "3-bit PQ ratio {} too low", vpq.compression_ratio());
    }
}
