//! `tq debug-*` -- diagnostic commands for Bug #2 (chat template gibberish).

use anyhow::{Context, Result};

use crate::{gguf, hub, quant, Cli, Commands};

/// Dump embedding row statistics for specific token IDs.
/// Used to diagnose Bug #2 (chat template inference gibberish).
pub(crate) fn cmd_debug_embedding(cli: &Cli) -> Result<()> {
    let (model_name, tokens_str) = match &cli.command {
        Some(Commands::DebugEmbedding { model, tokens }) => (model.as_str(), tokens.as_str()),
        _ => unreachable!(),
    };

    let token_ids: Vec<u32> = tokens_str
        .split(',')
        .map(|s| s.trim().parse::<u32>().with_context(|| format!("invalid token id: {}", s)))
        .collect::<Result<_>>()?;

    // Resolve model path
    let (gguf_path, _) = hub::resolve(model_name)?;
    eprintln!("Loading GGUF: {}", gguf_path.display());

    let file = std::fs::File::open(&gguf_path)
        .with_context(|| format!("cannot open {}", gguf_path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let ct = gguf::GgufContent::read(&mut reader)
        .map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;

    // Look up token_embd.weight metadata
    let info = ct.tensor_infos.get("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found in GGUF"))?
        .clone();

    let shape = &info.shape;
    let dtype = info.dtype;

    eprintln!("\n=== token_embd.weight Info ===");
    eprintln!("Shape: {:?}", shape);
    eprintln!("dtype: {:?}", dtype);
    eprintln!("n_elements: {}", info.n_elements);
    eprintln!("data_size_bytes: {}", info.data_size_bytes());

    // GGUF stores token_embd as [vocab_size, hidden_size]
    // shape[0] = vocab_size, shape[1] = hidden_size
    let vocab_size = shape[0];
    let hidden_size = if shape.len() >= 2 { shape[1] } else { 1 };

    eprintln!("hidden_size: {}", hidden_size);
    eprintln!("vocab_size:  {}", vocab_size);

    let block_numel = dtype.block_numel();
    let block_bytes = dtype.block_size_bytes();
    let blocks_per_row = (hidden_size + block_numel - 1) / block_numel;
    let bytes_per_row = blocks_per_row * block_bytes;
    eprintln!("block_numel: {}  block_bytes: {}", block_numel, block_bytes);
    eprintln!("blocks_per_row: {}  bytes_per_row: {}", blocks_per_row, bytes_per_row);
    eprintln!("expected total: {} bytes", vocab_size * bytes_per_row);

    // Load raw bytes
    let (_, raw) = ct.tensor_data(&mut reader, "token_embd.weight")
        .map_err(|e| anyhow::anyhow!("read error: {}", e))?;
    eprintln!("actual raw len: {} bytes", raw.len());

    // Verify consistency
    let actual_rows = raw.len() / bytes_per_row;
    eprintln!("actual rows (from raw len): {}", actual_rows);
    if actual_rows != vocab_size {
        eprintln!("\n⚠️  MISMATCH: shape says vocab_size={} but raw bytes imply {} rows",
                  vocab_size, actual_rows);
    }

    // Per-token stats
    println!("\n{:<8} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10} {}",
             "TokenID", "L2_norm", "Mean", "Min", "Max", "Zeros%", "Near0%", "Status");
    println!("{}", "-".repeat(110));

    for &token_id in &token_ids {
        let idx = token_id as usize;
        let row_start = idx * bytes_per_row;
        let row_end = row_start + bytes_per_row;

        if row_end > raw.len() {
            println!("{:<8} {:<60} OOB (row_end={} > raw_len={})",
                     token_id, "—", row_end, raw.len());
            continue;
        }

        let row_data = &raw[row_start..row_end];
        let row_f32 = quant::dequantize(row_data, dtype, hidden_size);

        let l2_norm: f32 = row_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mean: f32 = row_f32.iter().sum::<f32>() / hidden_size as f32;
        let min = row_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = row_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let zeros = row_f32.iter().filter(|&&x| x == 0.0).count();
        let near_zero = row_f32.iter().filter(|&&x| x.abs() < 1e-6).count();
        let zeros_pct = 100.0 * zeros as f32 / hidden_size as f32;
        let near_zero_pct = 100.0 * near_zero as f32 / hidden_size as f32;

        // Heuristic: if L2 norm << expected → damaged
        let status = if l2_norm < 0.01 {
            "❌ DAMAGED (near-zero)"
        } else if zeros_pct > 50.0 {
            "⚠️  HIGH ZEROS"
        } else if near_zero_pct > 90.0 {
            "⚠️  NEAR-ZERO"
        } else {
            "✅ OK"
        };

        println!("{:<8} {:<12.6} {:<12.6} {:<12.6} {:<12.6} {:<10.2} {:<10.2} {}",
                 token_id, l2_norm, mean, min, max, zeros_pct, near_zero_pct, status);
    }

    // Dump first 10 values for chat template tokens
    println!("\n=== First 10 values per token ===");
    for &token_id in &token_ids {
        let idx = token_id as usize;
        let row_start = idx * bytes_per_row;
        let row_end = row_start + bytes_per_row;
        if row_end > raw.len() { continue; }

        let row_data = &raw[row_start..row_end];
        let row_f32 = quant::dequantize(row_data, dtype, hidden_size);
        let preview: Vec<String> = row_f32[..10.min(row_f32.len())]
            .iter()
            .map(|v| format!("{:+.4}", v))
            .collect();
        println!("Token {:>6}: [{}]", token_id, preview.join(", "));
    }

    // Silence unused-variable lint when cli is only used via match above
    let _ = cli;

    Ok(())
}

/// Tokenize a string and dump the resulting token IDs.
/// Used to verify chat template tokens are correctly recognized.
pub(crate) fn cmd_debug_tokenize(cli: &Cli) -> Result<()> {
    let (model_name, text) = match &cli.command {
        Some(Commands::DebugTokenize { model, text }) => (model.as_str(), text.as_str()),
        _ => unreachable!(),
    };

    let (_, tokenizer_path) = hub::resolve(model_name)?;
    let tok_path = tokenizer_path
        .ok_or_else(|| anyhow::anyhow!("no tokenizer found for '{}'", model_name))?;

    eprintln!("Loading tokenizer: {}", tok_path.display());
    let tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {}", e))?;

    // Test both with and without add_special_tokens
    for add_special in [false, true] {
        let enc = tok.encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("tokenize error: {}", e))?;
        let ids = enc.get_ids();
        let tokens = enc.get_tokens();

        println!("\n=== add_special_tokens = {} ===", add_special);
        println!("Input: {:?}", text);
        println!("Token count: {}", ids.len());
        println!();

        println!("{:<8} {:<10} {}", "Index", "ID", "Token");
        println!("{}", "-".repeat(60));
        for (i, (id, tok)) in ids.iter().zip(tokens.iter()).enumerate() {
            let marker = if *id >= 151640 { "  ← SPECIAL" } else { "" };
            println!("{:<8} {:<10} {:?}{}", i, id, tok, marker);
        }
    }

    // Round-trip: decode to verify
    let enc = tok.encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize error: {}", e))?;
    let decoded = tok.decode(enc.get_ids(), false)
        .map_err(|e| anyhow::anyhow!("decode error: {}", e))?;
    println!("\n=== Round-trip ===");
    println!("Original: {:?}", text);
    println!("Decoded:  {:?}", decoded);
    println!("Match:    {}", if text == decoded { "✅" } else { "❌ DIFFERENT" });

    // Check special token IDs
    println!("\n=== Special Token Lookup ===");
    for (name, expected) in &[
        ("<|im_start|>", 151644u32),
        ("<|im_end|>", 151645),
        ("<|endoftext|>", 151643),
    ] {
        let id = tok.token_to_id(name);
        match id {
            Some(got) if got == *expected => println!("{:<20} ✅ {} (expected {})", name, got, expected),
            Some(got) => println!("{:<20} ⚠️  got {} expected {}", name, got, expected),
            None => println!("{:<20} ❌ NOT IN VOCAB", name),
        }
    }

    let _ = cli;

    Ok(())
}
