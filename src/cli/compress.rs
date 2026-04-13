//! `tq compress` -- TQ4_1S weight compression (WHT rotation + 4-bit quantization).

use anyhow::{Context, Result};

use crate::{gguf, Cli, Commands};

pub(crate) fn cmd_compress(cli: &Cli) -> Result<()> {
    let (input, output, seed, boundary) = match &cli.command {
        Some(Commands::Compress { input, output, seed, boundary }) => {
            (input.clone(), output.clone(), *seed, *boundary)
        }
        _ => unreachable!(),
    };

    let output = output.unwrap_or_else(|| {
        let stem = input.file_stem().unwrap().to_string_lossy();
        input.with_file_name(format!("{}_tq4.gguf", stem))
    });

    eprintln!("TQ4_1S Weight Compression");
    eprintln!("  Input:    {}", input.display());
    eprintln!("  Output:   {}", output.display());
    eprintln!("  Seed:     {}", seed);
    eprintln!("  Boundary: {} layers (first/last kept original)", boundary);
    eprintln!();

    // Read source GGUF
    let mut reader = std::io::BufReader::new(
        std::fs::File::open(&input).with_context(|| format!("Cannot open {}", input.display()))?
    );
    let ct = gguf::GgufContent::read(&mut reader)?;

    // Detect architecture and layer count
    let arch = ct.metadata.get("general.architecture")
        .and_then(|v| v.to_string_val().ok())
        .unwrap_or_else(|| "llama".to_string());
    let block_count = ct.metadata.get(&format!("{}.block_count", arch))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(32) as usize;

    eprintln!("  Architecture: {}, {} layers", arch, block_count);

    // Decide which tensors to compress:
    // - Boundary layers (first N, last N): keep original
    // - Middle layers: attn + ffn_gate + ffn_up → TQ4_1S, ffn_down → keep
    // - Norms, embeddings, output head: keep original
    let should_compress = |name: &str| -> bool {
        if !name.starts_with("blk.") { return false; }
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() < 3 { return false; }
        let layer_idx: usize = match parts[1].parse() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if layer_idx < boundary || layer_idx >= block_count.saturating_sub(boundary) {
            return false;
        }
        if name.contains("ffn_down") { return false; }
        name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v") ||
        name.contains("attn_output") || name.contains("attn_qkv") ||
        name.contains("ffn_gate") || name.contains("ffn_up")
    };

    let mut tensor_names: Vec<String> = ct.tensor_infos.keys().cloned().collect();
    tensor_names.sort();

    let mut compressed_count = 0usize;
    let mut original_bytes = 0usize;
    let mut compressed_bytes_total = 0usize;
    let mut out_tensors: Vec<(String, Vec<usize>, gguf::GgmlDType, Vec<u8>)> = Vec::new();

    for name in &tensor_names {
        let info = ct.tensor_infos.get(name).unwrap();
        let (_, raw_data) = ct.tensor_data(&mut reader, name)?;

        // Only compress if source is Q8_0 or higher precision (F16/F32/BF16).
        // Q4_K_M is already 4-bit — TQ4_1S (5.0 BPW) would make it bigger.
        let source_compressible = matches!(info.dtype,
            gguf::GgmlDType::Q8_0 | gguf::GgmlDType::Q8_1 | gguf::GgmlDType::Q8K |
            gguf::GgmlDType::F16 | gguf::GgmlDType::F32 | gguf::GgmlDType::BF16 |
            gguf::GgmlDType::Q6K | gguf::GgmlDType::Q5K
        );

        if should_compress(name) && source_compressible && info.n_elements >= 64 {
            let f32_data = crate::quant::dequantize(&raw_data, info.dtype, info.n_elements);
            let blocks = tq_kv::weight_compress::compress_weights(&f32_data, seed);
            let compressed = tq_kv::weight_compress::blocks_to_bytes(&blocks);

            let orig_size = raw_data.len();
            let comp_size = compressed.len();
            eprintln!("  [TQ4_1S] {}: {} → {} ({:.1}x)",
                name, orig_size, comp_size, orig_size as f32 / comp_size as f32);

            original_bytes += orig_size;
            compressed_bytes_total += comp_size;
            compressed_count += 1;
            out_tensors.push((name.clone(), info.shape.clone(), gguf::GgmlDType::TQ4_1S, compressed));
        } else {
            original_bytes += raw_data.len();
            compressed_bytes_total += raw_data.len();
            out_tensors.push((name.clone(), info.shape.clone(), info.dtype, raw_data));
        }
    }

    eprintln!();
    eprintln!("  Compressed {}/{} tensors", compressed_count, tensor_names.len());
    eprintln!("  Original:   {:.1} MB", original_bytes as f64 / 1048576.0);
    eprintln!("  Compressed: {:.1} MB", compressed_bytes_total as f64 / 1048576.0);
    if compressed_bytes_total < original_bytes {
        eprintln!("  Ratio:      {:.2}x ({:.0}% reduction)",
            original_bytes as f64 / compressed_bytes_total as f64,
            (1.0 - compressed_bytes_total as f64 / original_bytes as f64) * 100.0);
    }

    // Build metadata
    let mut meta_pairs: Vec<(String, gguf::GgufValue)> = ct.metadata.into_iter().collect();
    meta_pairs.sort_by(|a, b| a.0.cmp(&b.0));
    meta_pairs.push(("tq.weight_compress_seed".to_string(), gguf::GgufValue::U64(seed)));
    meta_pairs.push(("tq.weight_compress_boundary".to_string(), gguf::GgufValue::U32(boundary as u32)));
    meta_pairs.push(("tq.weight_compress_format".to_string(), gguf::GgufValue::String("TQ4_1S".to_string())));

    eprintln!("  Writing {}...", output.display());
    let mut writer = std::io::BufWriter::new(
        std::fs::File::create(&output).with_context(|| format!("Cannot create {}", output.display()))?
    );

    let meta_refs: Vec<(&str, &gguf::GgufValue)> = meta_pairs.iter()
        .map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &[usize], gguf::GgmlDType, &[u8])> = out_tensors.iter()
        .map(|(n, s, d, data)| (n.as_str(), s.as_slice(), *d, data.as_slice())).collect();

    gguf::write_gguf(&mut writer, &meta_refs, &tensor_refs)?;

    eprintln!("  Done! Output: {}", output.display());
    Ok(())
}
