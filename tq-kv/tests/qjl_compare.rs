use tq_kv::*;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

fn gaussian_data(count: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count * dim).map(|_| {
        let u1: f32 = rng.gen::<f32>().max(1e-10);
        let u2: f32 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.1
    }).collect()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
fn qjl_vs_no_qjl_4bit() {
    let dim = 128;
    let n = 8 * 4096; // 8 heads * 4096 context = realistic
    let data = gaussian_data(n, dim, 42);

    // 4-bit WITH QJL
    let config_qjl = TurboQuantConfig { bits: 4, use_qjl: true, ..Default::default() };
    let t0 = std::time::Instant::now();
    let comp_qjl = compress_keys(&data, dim, &config_qjl);
    let compress_qjl_ms = t0.elapsed().as_millis();
    let t0 = std::time::Instant::now();
    let dec_qjl = decompress_keys(&comp_qjl, &config_qjl);
    let decompress_qjl_ms = t0.elapsed().as_millis();

    // 4-bit WITHOUT QJL
    let config_no = TurboQuantConfig { bits: 4, use_qjl: false, ..Default::default() };
    let t0 = std::time::Instant::now();
    let comp_no = compress_keys(&data, dim, &config_no);
    let compress_no_ms = t0.elapsed().as_millis();
    let t0 = std::time::Instant::now();
    let dec_no = decompress_keys(&comp_no, &config_no);
    let decompress_no_ms = t0.elapsed().as_millis();

    let mse_qjl: f32 = data.iter().zip(&dec_qjl).map(|(a,b)| (a-b)*(a-b)).sum::<f32>() / data.len() as f32;
    let mse_no: f32 = data.iter().zip(&dec_no).map(|(a,b)| (a-b)*(a-b)).sum::<f32>() / data.len() as f32;
    let sig: f32 = data.iter().map(|x| x*x).sum::<f32>() / data.len() as f32;
    let snr_qjl = 10.0 * (sig / mse_qjl).log10();
    let snr_no = 10.0 * (sig / mse_no).log10();
    let cos_qjl = cosine_sim(&data, &dec_qjl);
    let cos_no = cosine_sim(&data, &dec_no);

    let ratio_qjl = comp_qjl.original_memory_bytes() as f32 / comp_qjl.memory_bytes() as f32;
    let ratio_no = comp_no.original_memory_bytes() as f32 / comp_no.memory_bytes() as f32;

    eprintln!();
    eprintln!("================================================================");
    eprintln!("  4-BIT: QJL vs QJL'SIZ KARSILASTIRMA");
    eprintln!("  {} vectors x {} dim (Llama-3 8B full context)", n, dim);
    eprintln!("================================================================");
    eprintln!("                    QJL ON          QJL OFF");
    eprintln!("  --------------------------------------------------------");
    eprintln!("  Ratio             {:>7.1}x         {:>7.1}x", ratio_qjl, ratio_no);
    eprintln!("  SNR (dB)          {:>7.1}          {:>7.1}", snr_qjl, snr_no);
    eprintln!("  Cosine Sim        {:>10.6}     {:>10.6}", cos_qjl, cos_no);
    eprintln!("  Compress          {:>7} ms       {:>7} ms", compress_qjl_ms, compress_no_ms);
    eprintln!("  Decompress        {:>7} ms       {:>7} ms", decompress_qjl_ms, decompress_no_ms);
    eprintln!("  Memory (bytes)    {:>10}      {:>10}", comp_qjl.memory_bytes(), comp_no.memory_bytes());
    eprintln!("  --------------------------------------------------------");
    eprintln!("  SNR farki:    +{:.1} dB (QJL avantaji)", snr_qjl - snr_no);
    eprintln!("  Hiz farki:    {:.0}x compress, {:.0}x decompress (QJL'siz DAHA HIZLI)",
        compress_qjl_ms as f32 / compress_no_ms.max(1) as f32,
        decompress_qjl_ms as f32 / decompress_no_ms.max(1) as f32);
    eprintln!("  Ratio farki:  {:.1}x vs {:.1}x (QJL'siz DAHA IYI ratio)", ratio_qjl, ratio_no);
    eprintln!("================================================================");
}
