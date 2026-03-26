//! tq-diagnose: Per-layer diagnostic tool for TurboQuant fused attention quality.
//!
//! Designed to investigate Qwen 72B long-context degradation where compression
//! error may accumulate across 80 transformer layers with GQA-8 mapping.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --bin tq-diagnose -- --model-path /path/to/qwen2-72b-q4_k_m.gguf
//! ```
//!
//! # STUB STATUS
//!
//! This binary is a **stub**. Running it requires downloading the 45 GB Qwen 72B
//! GGUF model file. The code below documents the full diagnostic pipeline but
//! will exit with an error if the model file is not provided or not found.
//!
//! To run actual diagnostics without the model, use the unit tests in
//! `src/diagnostics.rs` which exercise the same logic on synthetic data.

mod diagnostics;

use std::path::PathBuf;

/// Qwen 72B architecture constants.
const QWEN72B_LAYERS: usize = 80;
const QWEN72B_N_QUERY_HEADS: usize = 64;
const QWEN72B_N_KV_HEADS: usize = 8;
const QWEN72B_HEAD_DIM: usize = 128;
const QWEN72B_GQA_REP: usize = QWEN72B_N_QUERY_HEADS / QWEN72B_N_KV_HEADS; // 8

fn main() {
    eprintln!("====================================================================");
    eprintln!("  tq-diagnose: TurboQuant Per-Layer Diagnostic Tool");
    eprintln!("====================================================================");
    eprintln!();
    eprintln!("  Target model:  Qwen2-72B");
    eprintln!("  Architecture:  {} layers, {} query heads, {} KV heads (GQA-{})",
        QWEN72B_LAYERS, QWEN72B_N_QUERY_HEADS, QWEN72B_N_KV_HEADS, QWEN72B_GQA_REP);
    eprintln!("  Head dim:      {}", QWEN72B_HEAD_DIM);
    eprintln!();

    // ----------------------------------------------------------------
    // Step 0: Verify GQA mapping
    // ----------------------------------------------------------------
    eprintln!("[Step 0] Verifying GQA-{} query-to-KV head mapping...", QWEN72B_GQA_REP);
    let mismatches = diagnostics::verify_gqa_mapping(QWEN72B_N_QUERY_HEADS, QWEN72B_N_KV_HEADS);
    if mismatches.is_empty() {
        eprintln!("  OK: All {} query heads map correctly to {} KV heads (n_rep={}).",
            QWEN72B_N_QUERY_HEADS, QWEN72B_N_KV_HEADS, QWEN72B_GQA_REP);
    } else {
        eprintln!("  ERROR: {} mapping mismatches detected!", mismatches.len());
        for (qh, expected, actual) in &mismatches {
            eprintln!("    query_head {} -> kv_head {} (expected {})", qh, actual, expected);
        }
    }
    eprintln!();

    // ----------------------------------------------------------------
    // Step 1: Parse CLI args for model path
    // ----------------------------------------------------------------
    let model_path: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from).or_else(|| {
        // Also check --model-path flag
        let args: Vec<String> = std::env::args().collect();
        for i in 0..args.len() {
            if args[i] == "--model-path" || args[i] == "-m" {
                return args.get(i + 1).map(PathBuf::from);
            }
        }
        None
    });

    // ----------------------------------------------------------------
    // Step 2: Run synthetic diagnostic (always available)
    // ----------------------------------------------------------------
    eprintln!("[Step 1] Running synthetic per-layer diagnostic (no model required)...");
    eprintln!();

    let seq_len = 64; // synthetic sequence length
    let config_4bit = tq_kv::TurboQuantConfig::balanced();
    let config_2bit = tq_kv::TurboQuantConfig::extreme();

    // Simulate 80 layers with synthetic keys
    for (label, config) in [("4-bit", &config_4bit), ("2-bit", &config_2bit)] {
        eprintln!("--- {} compression ---", label);

        let mut all_diags = Vec::with_capacity(QWEN72B_LAYERS);
        for layer_idx in 0..QWEN72B_LAYERS {
            // Deterministic synthetic keys: slight variation per layer to simulate
            // different activation distributions at different depths.
            let total = QWEN72B_N_KV_HEADS * seq_len * QWEN72B_HEAD_DIM;
            let layer_scale = 1.0 + (layer_idx as f32 * 0.01); // layers get slightly larger norms
            let keys: Vec<f32> = (0..total)
                .map(|i| {
                    let base = ((i as f32 * 0.0137 + layer_idx as f32 * 0.73).sin()) * 0.5;
                    base * layer_scale
                })
                .collect();

            let diag = diagnostics::diagnose_layer(
                &keys,
                QWEN72B_N_KV_HEADS,
                seq_len,
                QWEN72B_HEAD_DIM,
                config,
                layer_idx,
            );
            all_diags.push(diag);
        }

        diagnostics::print_diagnostic_report(&all_diags);
        eprintln!();
    }

    // ----------------------------------------------------------------
    // Step 3: Real model diagnostic (requires GGUF file)
    // ----------------------------------------------------------------
    match model_path {
        Some(path) => {
            if !path.exists() {
                eprintln!("[Step 2] Model file not found: {}", path.display());
                eprintln!("         Skipping real model diagnostic.");
                eprintln!();
                print_model_instructions();
                return;
            }

            eprintln!("[Step 2] Real model diagnostic with: {}", path.display());
            eprintln!();
            eprintln!("  STUB: Full model diagnostic is not yet implemented.");
            eprintln!("  The following steps would be performed:");
            eprintln!();
            eprintln!("  1. Load Qwen 72B GGUF model (candle quantized loader)");
            eprintln!("  2. Run a test prompt through the model (e.g., 'The capital of France is')");
            eprintln!("  3. During forward pass, intercept key vectors BEFORE compression at each layer");
            eprintln!("  4. For each of the {} layers:", QWEN72B_LAYERS);
            eprintln!("     a. Capture original key vectors: [{} heads x seq_len x {}]",
                QWEN72B_N_KV_HEADS, QWEN72B_HEAD_DIM);
            eprintln!("     b. Run diagnostics::diagnose_layer() to compress, decompress, compare");
            eprintln!("     c. Record per-layer cosine similarity and error metrics");
            eprintln!("  5. Print full diagnostic report with error accumulation detection");
            eprintln!("  6. Test at increasing context lengths: 128, 256, 512, 1024, 2048, 4096");
            eprintln!("     to detect length-dependent degradation");
            eprintln!();
            eprintln!("  To implement: modify turbo_qwen2.rs LayerWeights::forward_attn()");
            eprintln!("  to capture k_flat before compression and pass to diagnose_layer().");
            eprintln!();

            // Implementation sketch (commented out — requires candle + model loading):
            //
            // use candle_core::{Device, DType};
            // use candle_core::quantized::gguf_file;
            //
            // let device = Device::Cpu;
            // let mut file = std::fs::File::open(&path).unwrap();
            // let content = gguf_file::Content::read(&mut file).unwrap();
            // let tq_config = tq_kv::TurboQuantConfig::balanced();
            // let model = turbo_qwen2::ModelWeights::from_gguf(content, &mut file, &device, tq_config).unwrap();
            //
            // // Forward pass with diagnostic hooks would go here.
            // // Each layer's k_flat (captured in forward_attn at line 214 of turbo_qwen2.rs)
            // // needs to be extracted and passed to diagnose_layer().
        }
        None => {
            eprintln!("[Step 2] No model path provided. Skipping real model diagnostic.");
            eprintln!();
            print_model_instructions();
        }
    }
}

fn print_model_instructions() {
    eprintln!("  To run with the real Qwen 72B model:");
    eprintln!();
    eprintln!("    cargo run --release --bin tq-diagnose -- --model-path /path/to/qwen2-72b-instruct-q4_k_m.gguf");
    eprintln!();
    eprintln!("  The model file is ~45 GB. Download from HuggingFace:");
    eprintln!("    https://huggingface.co/Qwen/Qwen2-72B-Instruct-GGUF");
    eprintln!();
    eprintln!("  Synthetic diagnostics (Step 1) ran successfully above.");
    eprintln!("  Those results validate the diagnostic pipeline itself.");
}
