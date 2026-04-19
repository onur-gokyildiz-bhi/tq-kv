//! `tq bench` -- performance benchmarks (Standard / TQ / TQ+TriAttention).

use anyhow::{Context, Result};
use std::time::Instant;

use crate::engine::{Engine, GenerationParams};
use crate::{auto_tq, calibrate, catalog, config, hub, model, models, Cli, Commands};

pub(crate) fn cmd_bench(cli: &Cli) -> Result<()> {
    let (model_name, tokens, json_output, custom_prompt, prompt_file, tq_only, draft_model, spec_k) = match &cli.command {
        Some(Commands::Bench { model, tokens, json, prompt, prompt_file, tq_only, draft, speculate }) => {
            (model.as_str(), *tokens, *json, prompt.as_deref(), prompt_file.as_ref(), *tq_only, draft.as_deref(), *speculate)
        }
        _ => unreachable!(),
    };

    // Resolve prompt: --prompt-file wins over --prompt; both fall back to default.
    let file_prompt = match prompt_file {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .with_context(|| format!("reading --prompt-file {}", path.display()))?
        ),
        None => None,
    };
    let bench_prompt: &str = file_prompt.as_deref().or(custom_prompt).unwrap_or(
        "Explain the theory of relativity in simple terms. Include examples."
    );

    let display_name = catalog::find(model_name)
        .map(|e| e.display.to_string())
        .unwrap_or_else(|| model_name.to_string());

    if !json_output {
        eprintln!("TurboQuant Benchmark -- {}", display_name);
        eprintln!("Generating {} tokens per run...\n", tokens);
    }

    let gen_params = GenerationParams {
        max_tokens: tokens,
        temperature: 0.0, // deterministic for benchmarking
        ..Default::default()
    };

    // Helper to resolve and load engine
    let load_engine = |tq_config: Option<tq_kv::TurboQuantConfig>| -> Result<Engine> {
        if let Some(model_config) = config::get_model(model_name) {
            model::load_engine(model_config, None, tq_config, None, cli.cpu)
        } else {
            let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
            let tok_path = tokenizer_path.with_context(|| {
                format!("No tokenizer found for model '{}'", model_name)
            })?;
            let mf = gguf_path.to_string_lossy().to_string();
            let arch = config::detect_arch(&mf);
            Engine::load_with_device(&gguf_path, &tok_path, arch, tq_config, cli.cpu)
        }
    };

    // Detect chat template for proper prompt formatting.
    // TQ_BENCH_SYSTEM=1 adds the system prompt block (disabled by default because
    // commit 32c2dc5 showed it regressed sentence-completion prompts; worth
    // re-testing after Bug A/B fixes).
    let use_system = std::env::var("TQ_BENCH_SYSTEM").ok().map_or(false, |v| v == "1");
    let raw_prompt = std::env::var("TQ_RAW_PROMPT").ok().map_or(false, |v| v == "1");
    let formatted_prompt = if raw_prompt {
        bench_prompt.to_string()
    } else if let Some(entry) = catalog::find(model_name) {
        let lower = entry.arch.to_lowercase();
        if lower.contains("qwen") {
            if use_system {
                format!(
                    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    bench_prompt
                )
            } else {
                format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", bench_prompt)
            }
        } else if lower.contains("llama") {
            format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                bench_prompt
            )
        } else {
            bench_prompt.to_string()
        }
    } else {
        bench_prompt.to_string()
    };

    let n_runs = if tq_only { 2 } else { 3 };  // TQ + TQ+TriAttn (or Standard + TQ + TQ+TriAttn)

    // --- Run 1: Standard (no TQ) ---
    // Explicitly disable TriAttention for standard + TQ runs
    models::turbo_generic::set_triattention_override(false);

    let std_result = if tq_only {
        None
    } else {
        if !json_output {
            eprintln!("[1/{}] Loading model (standard)...", n_runs);
        }
        let mut engine_std = load_engine(None)?;
        let result = bench_run(&mut engine_std, &formatted_prompt, &gen_params)?;
        drop(engine_std);
        if !json_output {
            eprintln!("  Standard: {:.1} tok/s, {:.2}s total, TTFT {:.3}s\n",
                result.tok_per_sec, result.total_secs, result.ttft_secs);
        }
        Some(result)
    };

    // --- Run 2: TurboQuant 4-bit (no TriAttention) ---
    let run_idx = if tq_only { 1 } else { 2 };
    if !json_output {
        eprintln!("[{}/{}] Loading model (TQ 4-bit)...", run_idx, n_runs);
    }
    let bench_bits: u8 = std::env::var("TQ_BITS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(4);
    let mut tq_config = match bench_bits {
        2 => tq_kv::TurboQuantConfig::extreme(),
        3 => tq_kv::TurboQuantConfig::aggressive(),
        _ => tq_kv::TurboQuantConfig::balanced(),
    };
    if let Ok(val) = std::env::var("TQ_GROUP") {
        if let Ok(gs) = val.parse::<usize>() { tq_config.group_size = gs; }
    }
    let mut engine_tq = load_engine(Some(tq_config.clone()))?;
    let tq_result = bench_run(&mut engine_tq, &formatted_prompt, &gen_params)?;
    if !json_output {
        eprintln!("  TQ 4-bit: {:.1} tok/s, {:.2}s total, TTFT {:.3}s\n",
            tq_result.tok_per_sec, tq_result.total_secs, tq_result.ttft_secs);
    }
    drop(engine_tq);

    // --- Run 3: TQ 4-bit + TriAttention ---
    let run_idx = if tq_only { 2 } else { 3 };
    if !json_output {
        eprintln!("[{}/{}] Loading model (TQ+TriAttn)...", run_idx, n_runs);
    }
    // Enable TriAttention and load calibration
    models::turbo_generic::set_triattention_override(true);
    if let Some(cal_data) = calibrate::load_calibration_for_model(model_name) {
        calibrate::init_triattention(&cal_data, cal_data.head_dim);
    }
    let mut engine_tri = load_engine(Some(tq_config))?;
    let tri_result = bench_run(&mut engine_tri, &formatted_prompt, &gen_params)?;
    if !json_output {
        eprintln!("  TQ+TriAttn: {:.1} tok/s, {:.2}s total, TTFT {:.3}s\n",
            tri_result.tok_per_sec, tri_result.total_secs, tri_result.ttft_secs);
    }

    // --- Self-speculative decoding (TQ_SELF_SPEC=<n_layers>, e.g. TQ_SELF_SPEC=8) ---
    if let Ok(draft_layers_str) = std::env::var("TQ_SELF_SPEC") {
        if let Ok(draft_layers) = draft_layers_str.parse::<usize>() {
            let spec_k_self = std::env::var("TQ_SPEC_K").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(3usize);
            if !json_output {
                eprintln!("Self-speculative decode: draft_layers={}, K={}...", draft_layers, spec_k_self);
            }
            engine_tri.clear_cache();
            engine_tri.model.0.clear_kv_cache();
            let t0 = std::time::Instant::now();
            let (spec_output, avg_accepted) = engine_tri.self_speculative_generate(
                &formatted_prompt, &gen_params, draft_layers, spec_k_self,
                |_| {},
            )?;
            let elapsed = t0.elapsed().as_secs_f64();
            let spec_tokens = engine_tri.tokenizer.encode(&*spec_output, false)
                .map(|e| e.get_ids().len() as u32).unwrap_or(0);
            if !json_output {
                eprintln!("  Self-spec (L={}): {:.1} tok/s, {:.1} accepted/step, {} tokens in {:.2}s",
                    draft_layers, spec_tokens as f64 / elapsed, avg_accepted, spec_tokens, elapsed);
            }
        }
    }

    // --- Speculative decoding (optional, uses TQ+TriAttn engine) ---
    let _spec_result = if let Some(draft_name) = draft_model {
        if !json_output {
            eprintln!("Loading draft model ({}) for speculative decoding (K={})...", draft_name, spec_k);
        }
        let mut draft_engine = if let Some(model_config) = config::get_model(draft_name) {
            model::load_engine(model_config, None, None, None, cli.cpu)?
        } else {
            let (gguf_path, tokenizer_path) = hub::resolve(draft_name)?;
            let tok_path = tokenizer_path.with_context(|| {
                format!("No tokenizer found for draft model '{}'", draft_name)
            })?;
            let arch = config::detect_arch(&gguf_path.to_string_lossy().as_ref());
            Engine::load_with_device(&gguf_path, &tok_path, arch, None, cli.cpu)?
        };

        engine_tri.clear_cache();
        engine_tri.model.0.clear_kv_cache();

        let t0 = std::time::Instant::now();
        let (spec_output, avg_accepted) = Engine::speculative_generate(
            &mut engine_tri, &mut draft_engine,
            &formatted_prompt, &gen_params, spec_k,
            |_| {},
        )?;
        let elapsed = t0.elapsed().as_secs_f64();
        let spec_tokens = engine_tri.tokenizer.encode(&*spec_output, false)
            .map(|e| e.get_ids().len() as u32)
            .unwrap_or(0);

        if !json_output {
            eprintln!("  Speculative (K={}): {:.1} tok/s, {:.1} accepted/step, {} tokens in {:.2}s",
                spec_k, spec_tokens as f64 / elapsed, avg_accepted, spec_tokens, elapsed);
        }
        drop(draft_engine);
        Some((spec_tokens as f64 / elapsed, avg_accepted))
    } else {
        None
    };
    drop(engine_tri);
    // Restore TriAttention override
    models::turbo_generic::clear_triattention_override();

    // --- KV cache estimates ---
    let (n_layers, n_kv_heads, head_dim) = if let Some(entry) = catalog::find(model_name) {
        auto_tq::estimate_arch_params(entry.arch, entry.size_gb)
    } else {
        (32, 8, 128) // defaults
    };
    let kv_bytes = 2 * n_layers * n_kv_heads * head_dim * 2 * (tokens as usize);
    let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);
    let tq_compression = 3.8;
    let kv_tq_mb = kv_mb / tq_compression;
    // TriAttention: budget limits stored tokens (default 2048, but for 100-token bench ~all kept)
    let tri_budget = models::turbo_generic::get_triattention_budget();
    let tri_effective_tokens = std::cmp::min(tokens as usize, tri_budget);
    let kv_tri_mb = kv_mb * (tri_effective_tokens as f64 / tokens as f64) / tq_compression;

    // --- Output ---
    if json_output {
        let mut json = serde_json::json!({
            "model": display_name,
            "tokens": tokens,
            "tq_4bit": {
                "total_secs": tq_result.total_secs,
                "tok_per_sec": tq_result.tok_per_sec,
                "ttft_secs": tq_result.ttft_secs,
                "tokens_generated": tq_result.tokens_generated,
            },
            "tq_triattn": {
                "total_secs": tri_result.total_secs,
                "tok_per_sec": tri_result.tok_per_sec,
                "ttft_secs": tri_result.ttft_secs,
                "tokens_generated": tri_result.tokens_generated,
            },
            "kv_cache_mb": kv_mb,
            "kv_tq_mb": kv_tq_mb,
            "kv_triattn_mb": kv_tri_mb,
        });
        if let Some(ref s) = std_result {
            json["standard"] = serde_json::json!({
                "total_secs": s.total_secs,
                "tok_per_sec": s.tok_per_sec,
                "ttft_secs": s.ttft_secs,
                "tokens_generated": s.tokens_generated,
            });
        }
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        // Helper: format delta percentage
        let fmt_delta = |a: f64, b: f64, higher_is_better: bool| -> String {
            if b == 0.0 { return String::new(); }
            let ratio = a / b;
            if higher_is_better {
                if ratio > 1.0 { format!("+{:.0}%", (ratio - 1.0) * 100.0) }
                else { format!("-{:.0}%", (1.0 - ratio) * 100.0) }
            } else {
                if ratio < 1.0 { format!("-{:.0}%", (1.0 - ratio) * 100.0) }
                else { format!("+{:.0}%", (ratio - 1.0) * 100.0) }
            }
        };

        println!();
        println!("+==============================================================================+");
        println!("|  TurboQuant Benchmark -- {:<51}|", display_name);
        println!("+==============================================================================+");

        if let Some(ref s) = std_result {
            println!("| {:<16} | {:<11} | {:<11} | {:<11} | {:<7} |",
                "Metric", "Standard", "TQ 4-bit", "TQ+TriAttn", "vs Std");
            println!("+------------------+-------------+-------------+-------------+---------+");
            println!("| {:<16} | {:<11} | {:<11} | {:<11} | {:<7} |",
                "Tokens", s.tokens_generated, tq_result.tokens_generated, tri_result.tokens_generated, "");
            println!("| {:<16} | {:<11} | {:<11} | {:<11} | {:<7} |",
                "Total time",
                format!("{:.2}s", s.total_secs),
                format!("{:.2}s", tq_result.total_secs),
                format!("{:.2}s", tri_result.total_secs),
                fmt_delta(tri_result.total_secs, s.total_secs, false));
            println!("| {:<16} | {:<11} | {:<11} | {:<11} | {:<7} |",
                "tok/s",
                format!("{:.1}", s.tok_per_sec),
                format!("{:.1}", tq_result.tok_per_sec),
                format!("{:.1}", tri_result.tok_per_sec),
                fmt_delta(tri_result.tok_per_sec, s.tok_per_sec, true));
            println!("| {:<16} | {:<11} | {:<11} | {:<11} | {:<7} |",
                "TTFT",
                format!("{:.3}s", s.ttft_secs),
                format!("{:.3}s", tq_result.ttft_secs),
                format!("{:.3}s", tri_result.ttft_secs),
                fmt_delta(tri_result.ttft_secs, s.ttft_secs, false));
            println!("| {:<16} | {:<11} | {:<11} | {:<11} | {:<7} |",
                "KV cache (est.)",
                format!("{:.0} MB", kv_mb),
                format!("{:.1} MB", kv_tq_mb),
                format!("{:.1} MB", kv_tri_mb),
                fmt_delta(kv_tri_mb, kv_mb, false));
            println!("| {:<16} | {:<11} | {:<11} | {:<11} | {:<7} |",
                "Compression",
                "1.0x",
                format!("{:.1}x", tq_compression),
                format!("{:.0}x", kv_mb / kv_tri_mb.max(0.001)),
                "");
        } else {
            println!("| {:<16} | {:<11} | {:<11} |",
                "Metric", "TQ 4-bit", "TQ+TriAttn");
            println!("+------------------+-------------+-------------+");
            println!("| {:<16} | {:<11} | {:<11} |",
                "Tokens", tq_result.tokens_generated, tri_result.tokens_generated);
            println!("| {:<16} | {:<11} | {:<11} |",
                "Total time", format!("{:.2}s", tq_result.total_secs), format!("{:.2}s", tri_result.total_secs));
            println!("| {:<16} | {:<11} | {:<11} |",
                "tok/s", format!("{:.1}", tq_result.tok_per_sec), format!("{:.1}", tri_result.tok_per_sec));
            println!("| {:<16} | {:<11} | {:<11} |",
                "TTFT", format!("{:.3}s", tq_result.ttft_secs), format!("{:.3}s", tri_result.ttft_secs));
        }

        println!("+==============================================================================+");
    }

    Ok(())
}

pub(crate) struct BenchResult {
    pub tokens_generated: u32,
    pub total_secs: f64,
    pub tok_per_sec: f64,
    pub ttft_secs: f64,
}

pub(crate) fn bench_run(
    engine: &mut Engine,
    prompt: &str,
    params: &GenerationParams,
) -> Result<BenchResult> {
    engine.clear_cache();

    let start = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut token_count = 0u32;

    let show_output = std::env::var("TQ_BUG2_DEBUG").ok().map_or(false, |v| v == "1");
    let mut output_buf = String::new();
    let output = engine.generate(prompt, params, |token_text| {
        if first_token_time.is_none() {
            first_token_time = Some(Instant::now());
        }
        token_count += 1;
        if show_output {
            output_buf.push_str(token_text);
        }
    })?;
    if show_output {
        eprintln!("\n[BUG2] Full output: {:?}", output);
    }

    let total_elapsed = start.elapsed();
    let total_secs = total_elapsed.as_secs_f64();
    let ttft_secs = first_token_time
        .map(|t| t.duration_since(start).as_secs_f64())
        .unwrap_or(total_secs);
    let tok_per_sec = if total_secs > 0.0 {
        token_count as f64 / total_secs
    } else {
        0.0
    };

    Ok(BenchResult {
        tokens_generated: token_count,
        total_secs,
        tok_per_sec,
        ttft_secs,
    })
}
