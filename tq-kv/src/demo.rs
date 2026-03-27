//! tq-kv demo binary: demonstrates TurboQuant compression with generated test data.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use tq_kv::codebook;
use tq_kv::hadamard;
use tq_kv::{
    compress_keys, compress_single_key_with_signs, decompress_keys, fused_attention_scores,
    pre_rotate_query_with_signs, CompressedKeys, TurboQuantConfig,
};

/// Generate `count` Gaussian vectors of dimension `dim` (Box-Muller, seed-deterministic).
fn gaussian_vectors(count: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sigma = 1.0 / (dim as f32).sqrt();
    (0..count * dim)
        .map(|_| {
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * sigma
        })
        .collect()
}

/// Cosine similarity between two equal-length slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// SNR in dB between original and reconstructed.
fn snr_db(original: &[f32], reconstructed: &[f32]) -> f32 {
    let signal_power: f32 =
        original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
    let mse: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / original.len() as f32;
    if mse > 0.0 {
        10.0 * (signal_power / mse).log10()
    } else {
        f32::INFINITY
    }
}

struct BitResult {
    bits: u8,
    ratio: f32,
    cos_sim: f32,
    snr: f32,
    compress_us: u128,
    decompress_us: u128,
}

fn run_compression_test(data: &[f32], dim: usize, config: &TurboQuantConfig) -> BitResult {
    let t0 = Instant::now();
    let compressed = compress_keys(data, dim, config);
    let compress_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let decompressed = decompress_keys(&compressed, config);
    let decompress_us = t1.elapsed().as_micros();

    let ratio = compressed.compression_ratio();
    let cos = cosine_similarity(data, &decompressed);
    let snr = snr_db(data, &decompressed);

    BitResult {
        bits: config.bits,
        ratio,
        cos_sim: cos,
        snr,
        compress_us,
        decompress_us,
    }
}

fn main() {
    let num_vectors = 512;
    let dim = 128;
    let seed = 42u64;

    println!("tq-kv v0.3.0 -- TurboQuant KV Cache Compression Demo");
    println!("=====================================================");
    println!();

    // -- Section 1: Compression quality --
    let data = gaussian_vectors(num_vectors, dim, seed);

    println!("Compression Quality ({} vectors, dim={})", num_vectors, dim);
    println!("------------------------------------------");

    let configs = [
        TurboQuantConfig::extreme(),   // 2-bit
        TurboQuantConfig::aggressive(), // 3-bit
        TurboQuantConfig::balanced(),   // 4-bit
    ];

    let results: Vec<BitResult> = configs
        .iter()
        .map(|cfg| run_compression_test(&data, dim, cfg))
        .collect();

    for r in &results {
        let compress_ms = r.compress_us as f64 / 1000.0;
        let decompress_ms = r.decompress_us as f64 / 1000.0;
        println!(
            "  {}-bit: ratio={:.1}x  cos_sim={:.3}  SNR={:.1} dB  compress={:.1}ms  decompress={:.1}ms",
            r.bits, r.ratio, r.cos_sim, r.snr, compress_ms, decompress_ms
        );
    }

    // -- Section 2: NIAH test --
    println!();
    println!("NIAH Test (needle at position {}/{})", num_vectors / 2, num_vectors);
    println!("--------------------------------------");

    let signs = hadamard::generate_signs(dim, TurboQuantConfig::default().rotation_seed);
    let needle_idx = num_vectors / 2;

    // The needle key is the vector at index needle_idx
    let needle_key = &data[needle_idx * dim..(needle_idx + 1) * dim];

    // Build a query that matches the needle: q = needle_key (perfect match)
    let query: Vec<f32> = needle_key.to_vec();

    let scale = 1.0 / (dim as f32).sqrt();

    for &bits in &[4u8, 2u8] {
        let config = match bits {
            2 => TurboQuantConfig::extreme(),
            4 => TurboQuantConfig::balanced(),
            _ => unreachable!(),
        };

        // Compress all keys
        let compressed = compress_keys(&data, dim, &config);

        // Pre-rotate query
        let rotated_q = pre_rotate_query_with_signs(&query, &signs);

        // Get base centroids for this bit width
        let base_centroids = codebook::get_centroids(bits);

        // Compute fused attention scores
        let scores = fused_attention_scores(&rotated_q, &compressed, base_centroids, scale);

        // Find max position
        let (max_pos, max_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let status = if max_pos == needle_idx { "PASS" } else { "FAIL" };
        println!(
            "  {}-bit fused attention: needle score={:.2}, max_pos={} -- {}",
            bits, max_score, max_pos, status
        );
    }

    // -- Section 3: Fused attention speedup --
    println!();
    println!("Fused Attention Speedup ({} keys)", num_vectors);
    println!("----------------------------------");

    let config_4bit = TurboQuantConfig::balanced();

    // Build compressed cache incrementally (to match incremental API usage)
    let mut cache = CompressedKeys::new_empty(4, dim, config_4bit.rotation_seed);
    for i in 0..num_vectors {
        let key = &data[i * dim..(i + 1) * dim];
        let (packed, norm) = compress_single_key_with_signs(key, dim, &config_4bit, &signs);
        cache.append_raw(&packed, norm);
    }

    let query2 = gaussian_vectors(1, dim, 99);
    let rotated_q2 = pre_rotate_query_with_signs(&query2, &signs);
    let base_centroids = codebook::get_centroids(4);

    // Warm up
    let _ = fused_attention_scores(&rotated_q2, &cache, base_centroids, scale);

    // Benchmark fused path
    let iters = 100;
    let t_fused = Instant::now();
    for _ in 0..iters {
        let _ = fused_attention_scores(&rotated_q2, &cache, base_centroids, scale);
    }
    let fused_total_us = t_fused.elapsed().as_micros() as f64;
    let fused_per_key_us = fused_total_us / (iters as f64 * num_vectors as f64);

    // Benchmark decompress + dot path
    let compressed_batch = compress_keys(&data, dim, &config_4bit);
    let t_decomp = Instant::now();
    for _ in 0..iters {
        let decompressed = decompress_keys(&compressed_batch, &config_4bit);
        // Dot product of query with each decompressed key
        let mut _scores = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let key_slice = &decompressed[i * dim..(i + 1) * dim];
            let dot: f32 = query2
                .iter()
                .zip(key_slice.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            _scores.push(dot);
        }
    }
    let decomp_total_us = t_decomp.elapsed().as_micros() as f64;
    let decomp_per_key_us = decomp_total_us / (iters as f64 * num_vectors as f64);

    let speedup = decomp_per_key_us / fused_per_key_us;

    println!("  Fused (centroid lookup):   {:.1} us/key", fused_per_key_us);
    println!(
        "  Decompress + dot product: {:.1} us/key",
        decomp_per_key_us
    );
    println!("  Speedup: {:.1}x", speedup);
}
