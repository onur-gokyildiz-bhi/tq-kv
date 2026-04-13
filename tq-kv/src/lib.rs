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
//! let (packed, norm, _token_mean) = compress_single_key_with_signs(&key, dim, &config, &signs);
//! cache.append_raw(&packed, norm);
//! ```

/// Lloyd-Max codebook quantization (2/3/4-bit optimal centroids).
pub mod codebook;
/// KV cache compaction — reduce token count via attention matching.
pub mod compaction;
/// TriAttention — trigonometric KV cache scoring & eviction (arXiv:2604.04921).
pub mod triattention;
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
    /// Effective spectral dimension (SpectralQuant).
    /// When set, QJL error correction is applied only to the top d_eff
    /// dimensions (signal), skipping noise dimensions. Reduces QJL memory
    /// cost by d_eff/dim factor and improves quality by avoiding noise injection.
    /// Set by calibration (eigenvalue spectrum analysis). 0 = disabled (full dim).
    pub spectral_d_eff: usize,
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
    /// Empirical finding: last layers are disproportionately sensitive —
    /// last 8 layers account for ~all quality loss in our experiments.
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

    /// Per-token mean removal: exploit softmax shift-invariance.
    /// Subtracts each key's scalar mean before quantization, stores it for decompress.
    /// Removes ~57% of key variance that attention ignores (softmax(x+c) = softmax(x)),
    /// giving the codebook more precision for the dimensions that matter.
    /// 2-bit cosine sim: 0.635 → 0.887 (+40%). 3-bit match rate: 26% → 64%.
    /// Storage: 4 bytes per token per head (~3% overhead). Default: true.
    pub center_keys: bool,

    /// Per-channel sigma in the rotated domain (KIVI-style adaptive codebook).
    /// After Hadamard rotation, each dimension still has different variance.
    /// Using per-channel sigma instead of per-vector sigma gives the codebook
    /// better per-dimension adaptation — critical for 2-bit quality.
    /// Computed from calibration. Length = head_dim. None = use per-vector sigma (legacy).
    pub rotated_channel_sigma: Option<Vec<f32>>,

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
            spectral_d_eff: 0,
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
            center_keys: true,
            rotated_channel_sigma: None,
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

    /// 3-bit balanced compression (~10.7x theoretical).
    /// Sprint 3: calibration shows 3-bit gives identical quality to 4-bit on
    /// Qwen2.5-7B Q4_K_M (4/5 on 5-prompt suite, PPL 4.974 exact match).
    /// 25% less K storage than 4-bit. Override with TQ_BITS=4 for conservative.
    pub fn balanced() -> Self {
        Self { bits: 3, ..Default::default() }
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

    // 2. QJL error correction (SpectralQuant: selective on signal dimensions)
    let qjl_corrections = if config.use_qjl {
        let reconstructed = polar::dequantize_batch(&polar_data, dim);
        let mut errors: Vec<f32> = rotated.iter().zip(reconstructed.iter())
            .map(|(orig, recon)| orig - recon).collect();
        // SpectralQuant: zero out noise dimension errors
        if config.spectral_d_eff > 0 && config.spectral_d_eff < dim {
            let d_eff = config.spectral_d_eff;
            for chunk in errors.chunks_exact_mut(dim) {
                for i in d_eff..dim { chunk[i] = 0.0; }
            }
        }
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
    /// Per-token key means (scalar per token per head).
    /// Stored when center_keys=true. Used to restore keys during decompress.
    /// Fused attention ignores this (softmax shift-invariant).
    pub key_means: Option<Vec<f32>>,
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
            key_means: None,
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
            key_means: self.key_means.as_ref().map(|m| m[..n].to_vec()),
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
            key_means: self.key_means.clone(),
        }
    }

    /// Select a subset of vectors by index, returning a new CompressedKeys.
    /// Indices must be sorted ascending and within [0, count).
    /// Used by TriAttention eviction to splice the cache.
    pub fn select_indices(&self, indices: &[usize]) -> CompressedKeys {
        let bpk = if self.count > 0 { self.packed_indices.len() / self.count } else { 0 };
        // Grouped format: norms_per_vector = dim / group_size (or 1 if non-grouped)
        let npv = if self.group_size > 0 && self.dim > 0 {
            self.dim / self.group_size
        } else {
            1
        };
        let mut packed = Vec::with_capacity(indices.len() * bpk);
        let mut norms = Vec::with_capacity(indices.len() * npv);
        for &idx in indices {
            if idx < self.count {
                let start = idx * bpk;
                let end = start + bpk;
                if end <= self.packed_indices.len() {
                    packed.extend_from_slice(&self.packed_indices[start..end]);
                }
                let norm_start = idx * npv;
                let norm_end = (norm_start + npv).min(self.norms.len());
                norms.extend_from_slice(&self.norms[norm_start..norm_end]);
            }
        }
        CompressedKeys {
            packed_indices: packed,
            norms,
            count: indices.len(),
            bits: self.bits,
            dim: self.dim,
            rotation_seed: self.rotation_seed,
            group_size: self.group_size,
            qjl_corrections: None,
            residual_indices: None,
            residual_norms: None,
            residual_bits: 0,
            outlier_indices: None,
            outlier_values: None,
            outlier_k: 0,
            key_means: self.key_means.as_ref().map(|m| {
                indices.iter().map(|&i| m[i]).collect()
            }),
        }
    }

    /// Append all vectors from `other` into this cache.
    /// Both must have the same bit width and dimension.
    pub fn append_from(&mut self, other: &CompressedKeys) {
        assert_eq!(self.bits, other.bits);
        assert_eq!(self.dim, other.dim);
        self.packed_indices.extend_from_slice(&other.packed_indices);
        self.norms.extend_from_slice(&other.norms);
        if let (Some(ref mut self_means), Some(ref other_means)) = (&mut self.key_means, &other.key_means) {
            self_means.extend_from_slice(other_means);
        }
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
        // SpectralQuant selective QJL: only correct error in signal dimensions.
        // Noise dimensions (d_eff..dim) get zeroed out — QJL on noise is harmful.
        if config.spectral_d_eff > 0 && config.spectral_d_eff < dim {
            let d_eff = config.spectral_d_eff;
            for chunk in errors.chunks_exact_mut(dim) {
                for i in d_eff..dim {
                    chunk[i] = 0.0; // zero out noise dimension errors
                }
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
        key_means: None, // batch compress path — mean-removal TODO
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

    // Dequantize with per-channel sigma (KIVI) or per-vector sigma (legacy)
    let mut result = Vec::with_capacity(compressed.count * dim);
    for i in 0..compressed.count {
        let start = i * dim;
        let indices = &all_indices[start..start + dim];
        let norm = compressed.norms[i];
        if norm < 1e-10 {
            result.extend(std::iter::repeat(0.0f32).take(dim));
            continue;
        }
        if let Some(ref pcs) = _config.rotated_channel_sigma {
            // KIVI per-channel: per-vector norm × per-channel ratio
            let per_vector_sigma = norm / (dim as f32).sqrt();
            let mean_sigma: f32 = pcs.iter().sum::<f32>() / dim as f32;
            for (d, &idx) in indices.iter().enumerate() {
                let ratio = pcs[d] / mean_sigma;
                let effective_sigma = per_vector_sigma * ratio;
                let cb = codebook::Codebook { sigma: effective_sigma, ..base_cb.clone() };
                result.push(cb.dequantize(idx));
            }
        } else {
            let adaptive_sigma = norm / (dim as f32).sqrt();
            let cb = codebook::Codebook { sigma: adaptive_sigma, ..base_cb.clone() };
            for &idx in indices {
                result.push(cb.dequantize(idx));
            }
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

    // Inverse per-token mean removal: add back stored means
    // (reverse of compress: mean subtracted after channel_bias, so restore before channel_bias)
    if let Some(ref means) = compressed.key_means {
        for (i, chunk) in result.chunks_exact_mut(dim).enumerate() {
            if i < means.len() {
                let mean = means[i];
                for val in chunk.iter_mut() {
                    *val += mean;
                }
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
/// **Norm Correction**: after quantization, the reconstruction's
/// L2 norm differs from the original. We store a corrected norm such that
/// `||decompress(compress(k))|| ≈ ||k||`. This is free at decode time because
/// decompression already scales by `stored_norm / sqrt(d)`.
pub fn compress_single_key_with_signs(
    key: &[f32],
    dim: usize,
    config: &TurboQuantConfig,
    signs: &[f32],
) -> (Vec<u8>, f32, Option<f32>) {
    assert_eq!(key.len(), dim);

    let mut rotated = key.to_vec();

    // Pre-Rotation Centering: subtract weight quantization bias
    if let Some(ref bias) = config.key_channel_bias {
        for (val, &b) in rotated.iter_mut().zip(bias.iter()) {
            *val -= b;
        }
    }

    // Per-token mean removal: exploit softmax shift-invariance.
    // Removes ~57% of key variance that attention ignores, giving the codebook
    // more precision for dimensions that matter. Mean stored for decompress.
    let token_mean = if config.center_keys {
        let mean = rotated.iter().sum::<f32>() / dim as f32;
        for val in rotated.iter_mut() {
            *val -= mean;
        }
        Some(mean)
    } else {
        None
    };

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
    } else if let Some(ref pcs) = config.rotated_channel_sigma {
        // KIVI per-channel: per-vector norm × per-channel ratio
        // per_vector_sigma captures this token's magnitude
        // relative_sigma[d] captures per-dimension variance differences
        let per_vector_sigma = norm / (dim as f32).sqrt();
        let mean_sigma: f32 = pcs.iter().sum::<f32>() / dim as f32;
        rotated.iter().enumerate().map(|(d, &v)| {
            let ratio = pcs[d] / mean_sigma;
            let effective_sigma = per_vector_sigma * ratio;
            let cb = codebook::Codebook { sigma: effective_sigma, ..base_cb };
            cb.quantize(v)
        }).collect()
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
        let recon_norm_sq: f32 = if let Some(ref pcs) = config.rotated_channel_sigma {
            let per_vector_sigma = norm / (dim as f32).sqrt();
            let mean_sigma: f32 = pcs.iter().sum::<f32>() / dim as f32;
            indices.iter().enumerate()
                .map(|(d, &idx)| {
                    let ratio = pcs[d] / mean_sigma;
                    let effective_sigma = per_vector_sigma * ratio;
                    let cb = codebook::Codebook { sigma: effective_sigma, ..base_cb };
                    let v = cb.dequantize(idx); v * v
                })
                .sum()
        } else {
            let sigma = norm / (dim as f32).sqrt();
            if let Some(ref cal_cb) = config.calibrated_codebook {
                indices.iter()
                    .map(|&idx| { let v = cal_cb.dequantize(idx) * sigma; v * v })
                    .sum()
            } else {
                let cb = codebook::Codebook { sigma, ..base_cb };
                indices.iter()
                    .map(|&idx| { let v = cb.dequantize(idx); v * v })
                    .sum()
            }
        };
        let recon_norm = recon_norm_sq.sqrt();
        if recon_norm > 1e-10 { norm * norm / recon_norm } else { norm }
    } else {
        norm
    };

    let packed = codebook::pack_indices(&indices, config.bits);
    (packed, corrected_norm, token_mean)
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
    let (rotation, _eigenvalues) = hadamard::calibrate_pca_rotation(data, dim);
    rotation
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
mod tests;
