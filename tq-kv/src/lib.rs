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
/// Fast Walsh-Hadamard Transform for decorrelation.
pub mod hadamard;

// Internal modules — not part of public API.
#[doc(hidden)]
pub mod polar;
#[doc(hidden)]
pub mod qjl;

/// C FFI layer for llama.cpp and other C/C++ engines.
/// Compile with `cargo build --release --features ffi` to produce `libtq_kv.a`.
#[cfg(feature = "ffi")]
pub mod ffi;

/// TurboQuant KV cache — drop-in replacement for candle_nn::kv_cache::KvCache.
/// Compile with `cargo build --features candle` to enable.
#[cfg(feature = "candle")]
pub mod candle_kv;
#[doc(hidden)]
pub mod bench;

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
    /// Value cache quantization bit width. 0 = uncompressed (fp16), 8 = 8-bit absmax.
    /// Keys use `bits` field; values use this. Default: 0 (uncompressed, matches paper).
    pub value_bits: u8,
    /// Per-channel scaling factors (SmoothQuant-style). Applied BEFORE Hadamard rotation
    /// to equalize outlier magnitudes across channels. `None` = disabled.
    /// Length must equal head_dim. Use `calibrate_channel_scales()` to compute from data.
    pub channel_scales: Option<Vec<f32>>,

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
    /// Per-vector norms (for QJL correction)
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
}

impl CompressedKeys {
    /// Create empty compressed keys (for incremental append).
    pub fn new_empty(bits: u8, dim: usize, rotation_seed: u64) -> Self {
        Self {
            packed_indices: Vec::new(),
            norms: Vec::new(),
            qjl_corrections: None,
            bits,
            dim,
            count: 0,
            rotation_seed,
        }
    }

    /// Append a single compressed key to the cache.
    /// `packed`: pack_indices output (for a single vector)
    /// `norm`: L2 norm of the vector (in rotated domain)
    pub fn append_raw(&mut self, packed: &[u8], norm: f32) {
        self.packed_indices.extend_from_slice(packed);
        self.norms.push(norm);
        self.count += 1;
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

        let front = CompressedKeys {
            packed_indices: self.packed_indices[..byte_split].to_vec(),
            norms: self.norms[..n].to_vec(),
            qjl_corrections: None, // QJL corrections are dropped during decay
            bits: self.bits,
            dim: self.dim,
            count: n,
            rotation_seed: self.rotation_seed,
        };

        self.packed_indices = self.packed_indices[byte_split..].to_vec();
        self.norms = self.norms[n..].to_vec();
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
            let adaptive_sigma = norms[i] / (dim as f32).sqrt();
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

    // Per-channel scaling (SmoothQuant): equalize outlier magnitudes before rotation
    if let Some(ref scales) = config.channel_scales {
        debug_assert_eq!(scales.len(), dim);
        for (val, &s) in rotated.iter_mut().zip(scales.iter()) {
            *val *= s;
        }
    }

    hadamard::randomized_hadamard_with_signs(&mut rotated, signs);

    let norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
    let base_cb = codebook::Codebook::new(config.bits, dim);

    let indices: Vec<u8> = if norm < 1e-10 {
        vec![0u8; dim]
    } else {
        let sigma = norm / (dim as f32).sqrt();
        let cb = codebook::Codebook { sigma, ..base_cb };
        rotated.iter().map(|&v| cb.quantize(v)).collect()
    };

    // Norm correction: compute reconstruction norm from indices, then adjust stored norm
    // so that decompress produces a vector with the ORIGINAL norm.
    // Decompression scales each centroid by (stored_norm / sqrt(d)), so:
    //   ||recon|| = stored_norm * sqrt(sum(centroid[idx_i]^2)) / sqrt(d)
    // We want ||recon|| = norm, so:
    //   corrected_norm = norm * norm / recon_norm_from_indices
    let corrected_norm = if norm > 1e-10 {
        let sigma = norm / (dim as f32).sqrt();
        let cb = codebook::Codebook { sigma, ..base_cb };
        let recon_norm_sq: f32 = indices.iter()
            .map(|&idx| { let v = cb.dequantize(idx); v * v })
            .sum();
        let recon_norm = recon_norm_sq.sqrt();
        if recon_norm > 1e-10 { norm * norm / recon_norm } else { norm }
    } else {
        norm
    };

    let packed = codebook::pack_indices(&indices, config.bits);
    (packed, corrected_norm)
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
    channel_absmax.iter().map(|&m| {
        if m > 1e-10 { median / m } else { 1.0 }
    }).collect()
}

/// Evaluate V2 compression quality.
pub fn evaluate_keys(original: &[f32], compressed: &CompressedKeys, config: &TurboQuantConfig) -> CompressionStats {
    let decompressed = decompress_keys(compressed, config);
    let mse = polar::compute_mse(original, &decompressed);
    let signal_power: f32 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
    let snr_db = if mse > 0.0 { 10.0 * (signal_power / mse).log10() } else { f32::INFINITY };
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
    let snr_db = if mse > 0.0 { 10.0 * (signal_power / mse).log10() } else { f32::INFINITY };
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
}
