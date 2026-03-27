//! TurboQuant KV cache — drop-in replacement for `candle_nn::kv_cache::KvCache`.
//!
//! Keys are compressed using TurboQuant (Lloyd-Max codebook + Hadamard rotation).
//! Values stay at full precision. When keys are needed for attention, they are
//! decompressed on the fly.
//!
//! # Usage
//!
//! Replace `KvCache::new(dim, max_seq_len)` with
//! `TurboKvCache::new(dim, n_kv_heads, head_dim, tq_config)`.
//!
//! ```ignore
//! use tq_kv::candle_kv::TurboKvCache;
//! use tq_kv::TurboQuantConfig;
//!
//! let config = TurboQuantConfig::balanced(); // 4-bit
//! let mut cache = TurboKvCache::new(2, 8, 128, config);
//!
//! // In your attention layer:
//! let (all_keys, all_values) = cache.append(&new_k, &new_v)?;
//! ```

use candle_core::{DType, Result, Tensor};

use crate::{codebook, hadamard, CompressedKeys, TurboQuantConfig};

/// TurboQuant-compressed KV cache.
///
/// Keys are quantized to 2-4 bits using the TurboQuant algorithm. Values are
/// stored at full precision. This provides 4-15x memory savings on the key
/// cache with minimal quality loss (cos_sim > 0.94).
pub struct TurboKvCache {
    /// Compressed key storage -- one CompressedKeys per KV head.
    k_compressed: Vec<CompressedKeys>,
    /// Standard value cache (uncompressed, fp16/f32).
    v_data: Option<Tensor>,
    /// Number of KV heads.
    n_kv_heads: usize,
    /// Head dimension (must be power of 2).
    head_dim: usize,
    /// Which tensor dimension is the sequence dim (typically 2 for (batch, heads, seq, dim)).
    dim: usize,
    /// Current total sequence length in cache.
    current_seq_len: usize,
    /// TurboQuant configuration (bits, rotation seed, QJL settings).
    config: TurboQuantConfig,
    /// Pre-computed Hadamard signs for fast rotation.
    signs: Vec<f32>,
    /// Base centroids for the configured bit width (N(0,1) normalized).
    centroids: &'static [f32],
}

impl TurboKvCache {
    /// Create a new TurboQuant KV cache.
    ///
    /// # Arguments
    /// * `dim` - Which tensor dimension is the sequence dimension (usually 2).
    /// * `n_kv_heads` - Number of key-value heads.
    /// * `head_dim` - Dimension per head (must be power of 2).
    /// * `config` - TurboQuant compression configuration.
    pub fn new(dim: usize, n_kv_heads: usize, head_dim: usize, config: TurboQuantConfig) -> Self {
        assert!(head_dim.is_power_of_two(), "head_dim must be power of 2, got {}", head_dim);
        let signs = hadamard::generate_signs(head_dim, config.rotation_seed);
        let centroids = codebook::get_centroids(config.bits);
        let k_compressed = (0..n_kv_heads)
            .map(|_| CompressedKeys::new_empty(config.bits, head_dim, config.rotation_seed))
            .collect();
        Self {
            k_compressed,
            v_data: None,
            n_kv_heads,
            head_dim,
            dim,
            current_seq_len: 0,
            config,
            signs,
            centroids,
        }
    }

    /// Append new key-value pairs to the cache.
    ///
    /// # Arguments
    /// * `k` - New keys, shape `(batch, n_kv_heads, seq_len, head_dim)`.
    /// * `v` - New values, shape `(batch, n_kv_heads, seq_len, head_dim)`.
    ///
    /// # Returns
    /// `(all_keys_decompressed, all_values)` covering the full sequence so far,
    /// ready for attention computation.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // 1. Extract new keys as f32 for compression
        let k_f32 = k.to_dtype(DType::F32)?.contiguous()?;
        let k_flat = k_f32.flatten_all()?.to_vec1::<f32>()?;
        let seq_len = k.dim(self.dim)?;

        // Compress each head x position independently
        for h in 0..self.n_kv_heads {
            for s in 0..seq_len {
                let offset = (h * seq_len + s) * self.head_dim;
                let key_vec = &k_flat[offset..offset + self.head_dim];
                let (packed, norm) = crate::compress_single_key_with_signs(
                    key_vec,
                    self.head_dim,
                    &self.config,
                    &self.signs,
                );
                self.k_compressed[h].append_raw(&packed, norm);
            }
        }

        // 2. Append values (uncompressed) along the sequence dimension
        self.v_data = Some(match &self.v_data {
            None => v.clone(),
            Some(prev_v) => Tensor::cat(&[prev_v, v], self.dim)?,
        });

        self.current_seq_len += seq_len;

        // 3. Decompress all keys for attention
        let total_len = self.current_seq_len;
        let device = k.device();
        let dtype = k.dtype();

        let mut all_k_data = Vec::with_capacity(self.n_kv_heads * total_len * self.head_dim);
        for compressed in &self.k_compressed {
            let decompressed = crate::decompress_keys(compressed, &self.config);
            all_k_data.extend(decompressed);
        }
        let k_tensor = Tensor::from_vec(
            all_k_data,
            (1, self.n_kv_heads, total_len, self.head_dim),
            device,
        )?
        .to_dtype(dtype)?;

        Ok((k_tensor, self.v_data.clone().unwrap()))
    }

    /// Get the current total sequence length stored in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Reset the cache, discarding all stored keys and values.
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.v_data = None;
        self.k_compressed = (0..self.n_kv_heads)
            .map(|_| {
                CompressedKeys::new_empty(self.config.bits, self.head_dim, self.config.rotation_seed)
            })
            .collect();
    }

    /// Compute fused attention scores directly from compressed keys (CPU SIMD path).
    ///
    /// This skips full key decompression entirely -- instead it pre-rotates the query
    /// and uses centroid table lookups (AVX2 when available). Much faster for
    /// autoregressive generation where `q` has `seq_len=1`.
    ///
    /// # Arguments
    /// * `q` - Query tensor, shape `(batch, n_query_heads, 1, head_dim)`.
    /// * `n_query_heads` - Number of query heads (may differ from `n_kv_heads` for GQA).
    /// * `scale` - Attention scale factor (typically `1/sqrt(head_dim)`).
    ///
    /// # Returns
    /// Attention scores tensor, shape `(batch, n_query_heads, 1, total_seq_len)`.
    pub fn fused_attention(
        &self,
        q: &Tensor,
        n_query_heads: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let q_f32 = q.to_dtype(DType::F32)?.contiguous()?;
        let q_flat = q_f32.flatten_all()?.to_vec1::<f32>()?;
        let n_rep = n_query_heads / self.n_kv_heads;
        let total_len = self.current_seq_len;

        use rayon::prelude::*;
        let head_scores: Vec<Vec<f32>> = (0..n_query_heads)
            .into_par_iter()
            .map(|qh| {
                let kv_h = qh / n_rep;
                let q_vec = &q_flat[qh * self.head_dim..(qh + 1) * self.head_dim];
                let rotated_q = crate::pre_rotate_query_with_signs(q_vec, &self.signs);
                crate::fused_attention_scores(
                    &rotated_q,
                    &self.k_compressed[kv_h],
                    self.centroids,
                    scale,
                )
            })
            .collect();

        let mut scores = Vec::with_capacity(n_query_heads * total_len);
        for s in &head_scores {
            scores.extend_from_slice(s);
        }

        Tensor::from_vec(scores, (1, n_query_heads, 1, total_len), q.device())
    }

    /// Compression ratio for the key cache (vs fp16 baseline).
    ///
    /// Returns 0.0 if the cache is empty.
    pub fn compression_ratio(&self) -> f32 {
        if self.current_seq_len == 0 {
            return 0.0;
        }
        let compressed_bytes: usize = self.k_compressed.iter().map(|c| c.memory_bytes()).sum();
        let original_bytes = self.n_kv_heads * self.current_seq_len * self.head_dim * 2; // fp16
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Memory saved in bytes compared to fp16 key storage.
    pub fn memory_saved_bytes(&self) -> usize {
        if self.current_seq_len == 0 {
            return 0;
        }
        let compressed: usize = self.k_compressed.iter().map(|c| c.memory_bytes()).sum();
        let original = self.n_kv_heads * self.current_seq_len * self.head_dim * 2;
        original.saturating_sub(compressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn make_random_tensor(shape: &[usize], seed: u64) -> Tensor {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_turbo_kv_cache_basic() {
        let config = TurboQuantConfig::balanced(); // 4-bit
        let n_kv_heads = 4;
        let head_dim = 64;
        let mut cache = TurboKvCache::new(2, n_kv_heads, head_dim, config);

        assert_eq!(cache.current_seq_len(), 0);

        // Simulate prefill: batch=1, heads=4, seq=8, dim=64
        let k = make_random_tensor(&[1, n_kv_heads, 8, head_dim], 42);
        let v = make_random_tensor(&[1, n_kv_heads, 8, head_dim], 43);

        let (all_k, all_v) = cache.append(&k, &v).unwrap();
        assert_eq!(cache.current_seq_len(), 8);
        assert_eq!(all_k.dims(), &[1, n_kv_heads, 8, head_dim]);
        assert_eq!(all_v.dims(), &[1, n_kv_heads, 8, head_dim]);

        // Verify compression ratio is reasonable (4-bit should be ~3-4x)
        let ratio = cache.compression_ratio();
        assert!(ratio > 2.0, "Expected ratio > 2.0, got {}", ratio);
        assert!(cache.memory_saved_bytes() > 0);
    }

    #[test]
    fn test_turbo_kv_cache_incremental_append() {
        let config = TurboQuantConfig::extreme(); // 2-bit
        let n_kv_heads = 2;
        let head_dim = 128;
        let mut cache = TurboKvCache::new(2, n_kv_heads, head_dim, config);

        // Prefill
        let k1 = make_random_tensor(&[1, n_kv_heads, 4, head_dim], 10);
        let v1 = make_random_tensor(&[1, n_kv_heads, 4, head_dim], 11);
        let (all_k1, all_v1) = cache.append(&k1, &v1).unwrap();
        assert_eq!(all_k1.dims(), &[1, n_kv_heads, 4, head_dim]);
        assert_eq!(all_v1.dims(), &[1, n_kv_heads, 4, head_dim]);

        // Generate token 1
        let k2 = make_random_tensor(&[1, n_kv_heads, 1, head_dim], 20);
        let v2 = make_random_tensor(&[1, n_kv_heads, 1, head_dim], 21);
        let (all_k2, all_v2) = cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.current_seq_len(), 5);
        assert_eq!(all_k2.dims(), &[1, n_kv_heads, 5, head_dim]);
        assert_eq!(all_v2.dims(), &[1, n_kv_heads, 5, head_dim]);

        // Generate token 2
        let k3 = make_random_tensor(&[1, n_kv_heads, 1, head_dim], 30);
        let v3 = make_random_tensor(&[1, n_kv_heads, 1, head_dim], 31);
        let (all_k3, all_v3) = cache.append(&k3, &v3).unwrap();
        assert_eq!(cache.current_seq_len(), 6);
        assert_eq!(all_k3.dims(), &[1, n_kv_heads, 6, head_dim]);
        assert_eq!(all_v3.dims(), &[1, n_kv_heads, 6, head_dim]);

        // 2-bit should give high compression
        let ratio = cache.compression_ratio();
        assert!(ratio > 5.0, "Expected ratio > 5.0 for 2-bit, got {}", ratio);
    }

    #[test]
    fn test_turbo_kv_cache_reset() {
        let config = TurboQuantConfig::balanced();
        let n_kv_heads = 2;
        let head_dim = 64;
        let mut cache = TurboKvCache::new(2, n_kv_heads, head_dim, config);

        let k = make_random_tensor(&[1, n_kv_heads, 4, head_dim], 42);
        let v = make_random_tensor(&[1, n_kv_heads, 4, head_dim], 43);
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.current_seq_len(), 4);

        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        assert_eq!(cache.compression_ratio(), 0.0);
        assert_eq!(cache.memory_saved_bytes(), 0);

        // Should work again after reset
        let (all_k, _) = cache.append(&k, &v).unwrap();
        assert_eq!(all_k.dims(), &[1, n_kv_heads, 4, head_dim]);
    }

    #[test]
    fn test_turbo_kv_cache_decompress_quality() {
        // Verify that decompressed keys have reasonable cosine similarity to originals
        let config = TurboQuantConfig::balanced(); // 4-bit
        let n_kv_heads = 2;
        let head_dim = 128;
        let mut cache = TurboKvCache::new(2, n_kv_heads, head_dim, config);

        let k = make_random_tensor(&[1, n_kv_heads, 4, head_dim], 99);
        let v = make_random_tensor(&[1, n_kv_heads, 4, head_dim], 100);

        let (all_k, _) = cache.append(&k, &v).unwrap();

        // Compare original vs decompressed
        let orig = k.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let decompressed = all_k.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Compute cosine similarity
        let dot: f32 = orig.iter().zip(decompressed.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_a * norm_b + 1e-10);

        assert!(
            cos_sim > 0.90,
            "Expected cos_sim > 0.90 for 4-bit, got {}",
            cos_sim
        );
    }

    #[test]
    fn test_turbo_kv_cache_fused_attention() {
        let config = TurboQuantConfig::balanced();
        let n_kv_heads = 2;
        let n_query_heads = 4; // GQA: 2 KV heads, 4 query heads
        let head_dim = 64;
        let mut cache = TurboKvCache::new(2, n_kv_heads, head_dim, config);

        // Fill cache with some keys
        let k = make_random_tensor(&[1, n_kv_heads, 8, head_dim], 42);
        let v = make_random_tensor(&[1, n_kv_heads, 8, head_dim], 43);
        cache.append(&k, &v).unwrap();

        // Query: single token
        let q = make_random_tensor(&[1, n_query_heads, 1, head_dim], 50);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let scores = cache.fused_attention(&q, n_query_heads, scale).unwrap();
        assert_eq!(scores.dims(), &[1, n_query_heads, 1, 8]);

        // Scores should be finite
        let scores_vec = scores.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for s in &scores_vec {
            assert!(s.is_finite(), "Score should be finite, got {}", s);
        }
    }
}
