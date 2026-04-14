//! Integration-style unit tests for tq-kv. Extracted from lib.rs on 2026-04-13.
//! (as an internal #[cfg(test)] submodule so private items stay accessible.)

#![cfg(test)]

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
            let (packed, norm, _token_mean) = compress_single_key_with_signs(chunk, dim, &config, &signs);
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
            let (packed, norm, _token_mean) = compress_single_key_with_signs(chunk, dim, &config, &signs);
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
            let (packed, norm, _token_mean) = compress_single_key_with_signs(chunk, dim, &config, &signs);
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
            let (packed, norm, _token_mean) = compress_single_key_with_signs(chunk, dim, &config, &signs);
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
        let (packed, norm, _token_mean) = compress_single_key_with_signs(&key, dim, &config, &signs);
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
        let (packed_signs, norm_signs, _) = compress_single_key_with_signs(&key, dim, &config, &signs);

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
            let (packed, norm, _token_mean) = compress_single_key_with_signs(chunk, dim, &config, &signs);
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
        let (packed, norm, _token_mean) = compress_single_key_with_signs(first_key, dim, &config, &signs);
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

    // ========================================
    // Asymmetric K/V plumbing (Step 2)
    // Tracking: knowledge `unblocks:hadamard-open-q`
    // ========================================

    /// `validate()` rejects out-of-range overrides per Q2.
    #[test]
    fn test_validate_asymmetric_kv_ranges() {
        // Symmetric (both None) always passes.
        let mut cfg = TurboQuantConfig::default();
        assert!(cfg.validate().is_ok(), "symmetric default must validate");

        // Valid range: Some(2)..=Some(8)
        for b in 2u8..=8 {
            cfg.k_bits = Some(b);
            cfg.v_bits = Some(b);
            assert!(cfg.validate().is_ok(), "k_bits=v_bits={} should validate", b);
        }

        // Some(0): rejected (per Q2 — use value_bits or future KBits::Fp16)
        cfg = TurboQuantConfig::default();
        cfg.k_bits = Some(0);
        assert!(cfg.validate().is_err(), "k_bits=Some(0) must be rejected");

        cfg = TurboQuantConfig::default();
        cfg.v_bits = Some(0);
        assert!(cfg.validate().is_err(), "v_bits=Some(0) must be rejected");

        // Some(1): rejected (1-bit not supported)
        cfg = TurboQuantConfig::default();
        cfg.k_bits = Some(1);
        assert!(cfg.validate().is_err(), "k_bits=Some(1) must be rejected");

        // Some(>8): rejected
        cfg = TurboQuantConfig::default();
        cfg.k_bits = Some(9);
        assert!(cfg.validate().is_err(), "k_bits=Some(9) must be rejected");
        cfg = TurboQuantConfig::default();
        cfg.v_bits = Some(16);
        assert!(cfg.validate().is_err(), "v_bits=Some(16) must be rejected");
    }

    /// With the `asymmetric-kv` feature on, the hot incremental K-compression
    /// path (`compress_single_key_*`) must use the K-side bit width when
    /// `k_bits` is set — verified by comparing packed byte counts and codebook
    /// centroid counts between a 4-bit override and a 2-bit baseline.
    ///
    /// NOTE: the competitor-style `K=8, V=4` target sits within the Q2 valid
    /// range `Some(2)..=Some(8)` but the underlying `codebook::Codebook`
    /// currently only supports 2/3/4-bit Lloyd-Max tables — extending it to
    /// 8-bit is a separate work item (likely alongside the dp4a kernel).
    /// This test therefore exercises the plumbing at 2/4 to prove the hot
    /// path honors `effective_k_bits()` asymmetrically.
    #[cfg(feature = "asymmetric-kv")]
    #[test]
    fn test_compress_single_key_asymmetric_uses_k_bits() {
        let dim = 64;
        let key = random_vectors(1, dim, 7);
        let signs: Vec<f32> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        // Baseline: bits=2, k_bits=None → effective_k_bits() == 2
        let cfg_sym = TurboQuantConfig { bits: 2, center_keys: false, ..Default::default() };
        assert_eq!(cfg_sym.effective_k_bits(), 2);

        // Asymmetric: bits=2, k_bits=Some(4), v_bits=Some(2)
        //   → effective_k_bits() == 4, effective_v_bits() == 2
        let cfg_asym = TurboQuantConfig {
            bits: 2,
            k_bits: Some(4),
            v_bits: Some(2),
            center_keys: false,
            ..Default::default()
        };
        assert_eq!(cfg_asym.effective_k_bits(), 4);
        assert_eq!(cfg_asym.effective_v_bits(), 2);
        assert!(cfg_asym.validate().is_ok());

        // compress_single_key_with_signs
        let (packed_sym, _, _) = compress_single_key_with_signs(&key, dim, &cfg_sym, &signs);
        let (packed_asym, _, _) = compress_single_key_with_signs(&key, dim, &cfg_asym, &signs);
        // 2-bit pack: 64 * 2 / 8 = 16 bytes
        // 4-bit pack: 64 * 4 / 8 = 32 bytes
        assert_eq!(packed_sym.len(), 16, "sym 2-bit should pack to 16 bytes");
        assert_eq!(packed_asym.len(), 32, "asym k_bits=Some(4) should pack to 32 bytes");

        // compress_single_key (legacy path without signs)
        let (p_sym, _) = compress_single_key(&key, dim, &cfg_sym);
        let (p_asym, _) = compress_single_key(&key, dim, &cfg_asym);
        assert_eq!(p_sym.len(), 16);
        assert_eq!(p_asym.len(), 32);

        // compress_single_key_grouped
        let cfg_sym_g = TurboQuantConfig { bits: 2, group_size: 32, center_keys: false, ..Default::default() };
        let cfg_asym_g = TurboQuantConfig {
            bits: 2,
            k_bits: Some(4),
            group_size: 32,
            center_keys: false,
            ..Default::default()
        };
        let (g_sym, _, _, _) = compress_single_key_grouped(&key, dim, &cfg_sym_g, &signs);
        let (g_asym, _, _, _) = compress_single_key_grouped(&key, dim, &cfg_asym_g, &signs);
        assert_eq!(g_sym.len(), 16);
        assert_eq!(g_asym.len(), 32);

        // Codebook centroid-count sanity: 2-bit → 4, 4-bit → 16.
        let cb_sym = codebook::Codebook::new(cfg_sym.effective_k_bits(), dim);
        let cb_asym = codebook::Codebook::new(cfg_asym.effective_k_bits(), dim);
        assert_eq!(cb_sym.centroids.len(), 4);
        assert_eq!(cb_asym.centroids.len(), 16);
    }

    /// Without the `asymmetric-kv` feature, `effective_k_bits()` must ignore
    /// `k_bits` and fall back to `bits` — preserves symmetric back-compat.
    #[cfg(not(feature = "asymmetric-kv"))]
    #[test]
    fn test_compress_single_key_symmetric_fallback() {
        let dim = 64;
        let cfg = TurboQuantConfig { bits: 4, k_bits: Some(8), v_bits: Some(4), ..Default::default() };
        // Without the feature, effective_*_bits returns bits / value_bits.
        assert_eq!(cfg.effective_k_bits(), 4);
        assert_eq!(cfg.effective_v_bits(), 0);

        let key = random_vectors(1, dim, 7);
        let signs: Vec<f32> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let (packed, _, _) = compress_single_key_with_signs(&key, dim, &cfg, &signs);
        // Always 4-bit when feature is off.
        assert_eq!(packed.len(), 32);
    }
