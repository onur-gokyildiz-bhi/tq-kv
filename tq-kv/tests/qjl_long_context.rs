//! QJL long context test: does QJL help at longer sequences?
//! Tanay's feedback: QJL might be needed at 16K+ to prevent error accumulation.

use tq_kv::*;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

fn gaussian_vectors(count: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count * dim).map(|_| {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.5
    }).collect()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}

#[test]
fn qjl_long_context_comparison() {
    let dim = 128;

    eprintln!();
    eprintln!("QJL Long Context Test");
    eprintln!("=====================");
    eprintln!("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Vectors", "QJL OFF cos", "QJL ON cos", "QJL OFF SNR", "QJL ON SNR", "QJL helps?");

    // Simulate different "context lengths" by compressing more vectors
    for n_vectors in [64, 256, 1024, 4096, 8192] {
        let data = gaussian_vectors(n_vectors, dim, 42);

        // 4-bit QJL OFF
        let config_off = TurboQuantConfig { bits: 4, use_qjl: false, ..Default::default() };
        let compressed_off = compress_keys(&data, dim, &config_off);
        let decompressed_off = decompress_keys(&compressed_off, &config_off);

        // 4-bit QJL ON
        let config_on = TurboQuantConfig { bits: 4, use_qjl: true, ..Default::default() };
        let compressed_on = compress_keys(&data, dim, &config_on);
        let decompressed_on = decompress_keys(&compressed_on, &config_on);

        let cos_off = cosine_sim(&data, &decompressed_off);
        let cos_on = cosine_sim(&data, &decompressed_on);

        let mse_off: f32 = data.iter().zip(&decompressed_off)
            .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / data.len() as f32;
        let mse_on: f32 = data.iter().zip(&decompressed_on)
            .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / data.len() as f32;

        let signal: f32 = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
        let snr_off = 10.0 * (signal / mse_off).log10();
        let snr_on = 10.0 * (signal / mse_on).log10();

        let helps = if cos_on > cos_off + 0.0001 { "YES" } else { "NO" };

        eprintln!("{:<12} {:<12.6} {:<12.6} {:<12.1} {:<12.1} {:<12}",
            n_vectors, cos_off, cos_on, snr_off, snr_on, helps);
    }

    // Also test 2-bit where error accumulation is more severe
    eprintln!();
    eprintln!("2-bit (more aggressive compression):");
    eprintln!("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Vectors", "QJL OFF cos", "QJL ON cos", "QJL OFF SNR", "QJL ON SNR", "QJL helps?");

    for n_vectors in [64, 256, 1024, 4096, 8192] {
        let data = gaussian_vectors(n_vectors, dim, 42);

        let config_off = TurboQuantConfig { bits: 2, use_qjl: false, ..Default::default() };
        let compressed_off = compress_keys(&data, dim, &config_off);
        let decompressed_off = decompress_keys(&compressed_off, &config_off);

        let config_on = TurboQuantConfig { bits: 2, use_qjl: true, ..Default::default() };
        let compressed_on = compress_keys(&data, dim, &config_on);
        let decompressed_on = decompress_keys(&compressed_on, &config_on);

        let cos_off = cosine_sim(&data, &decompressed_off);
        let cos_on = cosine_sim(&data, &decompressed_on);

        let mse_off: f32 = data.iter().zip(&decompressed_off)
            .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / data.len() as f32;
        let mse_on: f32 = data.iter().zip(&decompressed_on)
            .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / data.len() as f32;

        let signal: f32 = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
        let snr_off = 10.0 * (signal / mse_off).log10();
        let snr_on = 10.0 * (signal / mse_on).log10();

        let helps = if cos_on > cos_off + 0.0001 { "YES" } else { "NO" };

        eprintln!("{:<12} {:<12.6} {:<12.6} {:<12.1} {:<12.1} {:<12}",
            n_vectors, cos_off, cos_on, snr_off, snr_on, helps);
    }
}
