mod auto_tq;
mod catalog;
mod chat;
mod config;
mod diagnostics;
mod download;
mod engine;
mod hub;
mod inference;
mod model;
mod models;
mod serve;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use engine::{Engine, GenerationParams};

#[derive(Parser)]
#[command(
    name = "tq",
    about = "TurboQuant -- Local LLM inference with KV cache compression",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    // Legacy: if no subcommand, treat first arg as prompt (backward compat)
    /// Prompt text (legacy mode -- use `tq chat` instead)
    prompt: Option<String>,

    /// Force CPU inference
    #[arg(long, global = true)]
    cpu: bool,

    /// TurboQuant bit width (2, 3, or 4). 0 = auto
    #[arg(long, default_value = "4", global = true)]
    tq_bits: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive chat with a model
    Chat {
        /// Model name or path (e.g., qwen72b or /path/to/model.gguf)
        model: String,
        /// System prompt
        #[arg(short, long, default_value = config::DEFAULT_SYSTEM_PROMPT)]
        system: String,
        /// Max tokens per response
        #[arg(short = 'n', long, default_value = "512")]
        max_tokens: u32,
        /// Temperature (0.0 - 2.0)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,
        /// Top-p sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Top-k sampling
        #[arg(long, default_value = "40")]
        top_k: usize,
        /// Repeat penalty
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,
        /// Enable TurboQuant KV cache compression
        #[arg(long)]
        turbo_quant: bool,
        /// Disable auto-TQ (VRAM-aware automatic compression, enabled by default on GPU)
        #[arg(long)]
        no_auto_tq: bool,
        /// Custom GGUF model path
        #[arg(long)]
        model_path: Option<PathBuf>,
        /// HuggingFace repo for tokenizer (overrides default)
        #[arg(long)]
        tokenizer_repo: Option<String>,
    },
    /// Start OpenAI-compatible API server
    Serve {
        /// Model name (e.g., qwen72b, llama3-8b)
        #[arg(short, long, default_value = "llama3-8b")]
        model: String,
        /// Custom GGUF model path
        #[arg(long)]
        model_path: Option<PathBuf>,
        /// HuggingFace repo for tokenizer (overrides default)
        #[arg(long)]
        tokenizer: Option<String>,
        /// Port number
        #[arg(short, long, default_value = "11435")]
        port: u16,
        /// Enable TurboQuant KV cache compression
        #[arg(long)]
        turbo_quant: bool,
        /// Disable auto-TQ (VRAM-aware automatic compression, enabled by default on GPU)
        #[arg(long)]
        no_auto_tq: bool,
        /// Enable web UI
        #[arg(long)]
        ui: bool,
    },
    /// Download a model from HuggingFace
    Pull {
        /// Model name (e.g., qwen2:7b, llama:8b, mistral:7b)
        model: String,
    },
    /// List downloaded or available models
    List {
        /// Show all available models in catalog (not just downloaded)
        #[arg(short, long)]
        available: bool,
    },
    /// Remove a downloaded model
    Rm {
        /// Model name (e.g., qwen2:7b)
        model: String,
    },
    /// Run performance benchmarks (with and without TurboQuant)
    Bench {
        /// Model name or path (e.g., qwen2:7b, llama:8b)
        model: String,
        /// Number of tokens to generate per run
        #[arg(short = 'n', long, default_value = "100")]
        tokens: u32,
        /// Output results as JSON
        #[arg(long)]
        json: bool,
        /// Custom prompt for benchmark
        #[arg(long)]
        prompt: Option<String>,
        /// Skip standard (non-TQ) run — use when CUDA lacks stock model support
        #[arg(long)]
        tq_only: bool,
    },
    /// Check system compatibility
    Doctor,
    /// Run perplexity evaluation
    Perplexity {
        /// Model name (e.g., qwen72b, llama3-8b)
        #[arg(short, long, default_value = "llama3-8b")]
        model: String,
        /// Custom GGUF model path
        #[arg(long)]
        model_path: Option<PathBuf>,
        /// HuggingFace repo for tokenizer (overrides default)
        #[arg(long)]
        tokenizer_repo: Option<String>,
        /// Text file for evaluation
        file: PathBuf,
        /// Chunk size
        #[arg(long, default_value = "512")]
        chunk: usize,
        /// Enable TurboQuant KV cache compression
        #[arg(long)]
        turbo_quant: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Serve { .. }) => cmd_serve(&cli).await,
        Some(Commands::Chat { .. }) => cmd_chat(&cli),
        Some(Commands::Doctor) => cmd_doctor(),
        Some(Commands::Perplexity { .. }) => cmd_perplexity(&cli),
        Some(Commands::Pull { ref model }) => cmd_pull(model),
        Some(Commands::List { available }) => cmd_list(available),
        Some(Commands::Rm { ref model }) => cmd_rm(model),
        Some(Commands::Bench { .. }) => cmd_bench(&cli),
        None => {
            // Legacy mode: if prompt given, run like old tq-engine
            if cli.prompt.is_some() {
                cmd_legacy(&cli)
            } else {
                eprintln!("Use `tq chat <model>` or `tq serve` or `tq --help`");
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn resolve_tq_config(turbo_quant: bool, tq_bits: u8) -> Option<tq_kv::TurboQuantConfig> {
    if turbo_quant {
        let mut config = match tq_bits {
            2 => tq_kv::TurboQuantConfig::extreme(),
            3 => tq_kv::TurboQuantConfig::aggressive(),
            _ => tq_kv::TurboQuantConfig::balanced(),
        };
        // TQ_RESIDUAL=2 enables 2-bit residual quantization
        if let Ok(val) = std::env::var("TQ_RESIDUAL") {
            if let Ok(bits) = val.parse::<u8>() {
                config.residual_bits = bits;
            }
        }
        // TQ_OUTLIER=2 preserves top-2 outlier entries per vector at full precision
        if let Ok(val) = std::env::var("TQ_OUTLIER") {
            if let Ok(k) = val.parse::<usize>() {
                config.outlier_k = k;
            }
        }
        Some(config)
    } else {
        None
    }
}

/// Resolve TQ config with auto-TQ support.
///
/// Priority: --turbo-quant (explicit on) > --no-auto-tq (explicit off) > auto-decide
fn resolve_tq_config_with_auto(
    turbo_quant: bool,
    no_auto_tq: bool,
    tq_bits: u8,
    force_cpu: bool,
    model_name: &str,
) -> Option<tq_kv::TurboQuantConfig> {
    // If user explicitly enabled TQ, use that
    if turbo_quant {
        return resolve_tq_config(true, tq_bits);
    }

    // If user explicitly disabled auto-TQ, return None
    if no_auto_tq {
        return None;
    }

    // Auto-TQ: check if we should enable compression
    if force_cpu {
        // Auto-TQ only applies on GPU
        return None;
    }

    let device = match candle_core::Device::cuda_if_available(0) {
        Ok(dev) if dev.is_cuda() => dev,
        _ => return None,
    };

    // Look up model in catalog for size/arch info
    let entry = catalog::find(model_name);
    let (model_size_bytes, arch, size_gb) = if let Some(e) = entry {
        ((e.size_gb * 1024.0 * 1024.0 * 1024.0) as u64, e.arch, e.size_gb)
    } else {
        // Unknown model -- can't auto-decide, skip
        return None;
    };

    let (n_layers, n_kv_heads, head_dim) = auto_tq::estimate_arch_params(arch, size_gb);
    let max_context = 4096; // default context window

    let result = auto_tq::decide(&device, model_size_bytes, n_layers, n_kv_heads, head_dim, max_context);
    auto_tq::print_decision(&result);
    auto_tq::to_tq_config(&result)
}

fn cmd_pull(model_query: &str) -> Result<()> {
    let entry = catalog::find(model_query).with_context(|| {
        format!(
            "Unknown model: '{}'\n\nAvailable models:\n{}",
            model_query,
            catalog::list_available()
                .iter()
                .map(|e| format!("  {}:{:<6} {}", e.name, e.tag, e.display))
                .collect::<Vec<_>>()
                .join("\n")
        )
    })?;

    hub::download(entry)?;
    Ok(())
}

fn cmd_list(show_available: bool) -> Result<()> {
    if show_available {
        println!("Available models:\n");
        println!(
            "{:<16} {:<30} {:>8}  {}",
            "NAME", "DESCRIPTION", "SIZE", "ARCH"
        );
        println!("{}", "-".repeat(72));
        for entry in catalog::list_available() {
            let pulled = if hub::is_downloaded(entry.name, entry.tag) {
                " (pulled)"
            } else {
                ""
            };
            println!(
                "{:<16} {:<30} {:>6.1} GB  {}{}",
                format!("{}:{}", entry.name, entry.tag),
                entry.display,
                entry.size_gb,
                entry.arch,
                pulled,
            );
        }
        return Ok(());
    }

    let downloaded = hub::list_downloaded();
    if downloaded.is_empty() {
        eprintln!("No models downloaded yet.");
        eprintln!("\nRun `tq pull <model>` to download a model.");
        eprintln!("Run `tq list --available` to see all available models.");
        return Ok(());
    }

    println!("Downloaded models:\n");
    println!(
        "{:<16} {:<30} {:>8}  {}",
        "NAME", "DESCRIPTION", "SIZE", "STATUS"
    );
    println!("{}", "-".repeat(72));
    for dm in &downloaded {
        let status = if dm.gguf_exists { "ready" } else { "missing" };
        println!(
            "{:<16} {:<30} {:>6.1} GB  {}",
            format!("{}:{}", dm.meta.name, dm.meta.tag),
            dm.meta.display,
            dm.meta.size_gb,
            status,
        );
    }

    if downloaded.iter().any(|d| d.gguf_exists) {
        println!();
        println!("Use `tq chat <name:tag>` to start chatting.");
    }

    Ok(())
}

fn cmd_rm(model_query: &str) -> Result<()> {
    // Parse name:tag
    let (name, tag) = if let Some(entry) = catalog::find(model_query) {
        (entry.name, entry.tag)
    } else {
        anyhow::bail!(
            "Unknown model: '{}'. Use `tq list` to see downloaded models.",
            model_query
        );
    };

    hub::remove(name, tag)
}

fn cmd_chat(cli: &Cli) -> Result<()> {
    let (model_name, system, max_tokens, temperature, top_p, top_k, repeat_penalty,
         turbo_quant, no_auto_tq, model_path_override, tokenizer_repo_override) = match &cli.command {
        Some(Commands::Chat {
            model, system, max_tokens, temperature, top_p, top_k, repeat_penalty,
            turbo_quant, no_auto_tq, model_path, tokenizer_repo,
        }) => (
            model.as_str(), system.as_str(), *max_tokens, *temperature, *top_p,
            *top_k, *repeat_penalty, *turbo_quant, *no_auto_tq, model_path.as_deref(),
            tokenizer_repo.as_deref(),
        ),
        _ => unreachable!(),
    };

    let tq_config = resolve_tq_config_with_auto(
        turbo_quant, no_auto_tq, cli.tq_bits, cli.cpu, model_name,
    );

    // Try legacy config first, then fall back to hub resolution
    let (mut engine, model_file) = if let Some(model_config) = config::get_model(model_name) {
        eprintln!("Model: {}", model_config.display_name);
        let eng = model::load_engine(
            model_config, model_path_override,
            tq_config, tokenizer_repo_override, cli.cpu,
        )?;
        let mf = model_path_override
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| model_config.gguf_filename.to_string());
        (eng, mf)
    } else if model_path_override.is_none() {
        // Try hub resolution (catalog-based)
        let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
        let tok_path = match (tokenizer_repo_override, tokenizer_path) {
            (Some(repo), _) => {
                // User override
                let local = std::path::Path::new(repo).join("tokenizer.json");
                if local.exists() {
                    local
                } else if std::path::Path::new(repo).exists() && repo.ends_with(".json") {
                    PathBuf::from(repo)
                } else {
                    let api = hf_hub::api::sync::Api::new()?;
                    api.model(repo.to_string()).get("tokenizer.json")?
                }
            }
            (None, Some(tp)) => tp,
            (None, None) => anyhow::bail!(
                "No tokenizer found for model. Use --tokenizer-repo to specify one."
            ),
        };

        let mf = gguf_path.to_string_lossy().to_string();
        let arch = config::detect_arch(&mf);

        eprintln!("Model: {}", model_name);
        let eng = Engine::load_with_device(
            &gguf_path, &tok_path, arch, tq_config, cli.cpu,
        )?;
        (eng, mf)
    } else {
        anyhow::bail!(
            "Unknown model: '{}'. Use `tq list --available` to see available models.",
            model_name
        );
    };

    let template = chat::ChatTemplate::detect(&model_file);

    let gen_params = GenerationParams {
        max_tokens,
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        ..Default::default()
    };

    run_interactive(&mut engine, system, &gen_params, &template)
}

async fn cmd_serve(cli: &Cli) -> Result<()> {
    let (model_name, model_path_override, tokenizer_override, port, turbo_quant, no_auto_tq) =
        match &cli.command {
            Some(Commands::Serve { model, model_path, tokenizer, port, turbo_quant, no_auto_tq, .. }) => {
                (model.as_str(), model_path.as_deref(), tokenizer.as_deref(), *port, *turbo_quant, *no_auto_tq)
            }
            _ => unreachable!(),
        };

    let tq_config = resolve_tq_config_with_auto(
        turbo_quant, no_auto_tq, cli.tq_bits, cli.cpu, model_name,
    );

    // Try legacy config first, then fall back to hub resolution
    let (engine, model_file, display_name) = if let Some(model_config) = config::get_model(model_name) {
        eprintln!("Model: {}", model_config.display_name);
        let eng = model::load_engine(
            model_config, model_path_override,
            tq_config.clone(), tokenizer_override, cli.cpu,
        )?;
        let mf = model_path_override
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| model_config.gguf_filename.to_string());
        (eng, mf, model_config.display_name.to_string())
    } else if model_path_override.is_none() {
        // Try hub resolution
        let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
        let tok_path = match (tokenizer_override, tokenizer_path) {
            (Some(repo), _) => {
                let local = std::path::Path::new(repo).join("tokenizer.json");
                if local.exists() {
                    local
                } else if std::path::Path::new(repo).exists() && repo.ends_with(".json") {
                    PathBuf::from(repo)
                } else {
                    let api = hf_hub::api::sync::Api::new()?;
                    api.model(repo.to_string()).get("tokenizer.json")?
                }
            }
            (None, Some(tp)) => tp,
            (None, None) => anyhow::bail!(
                "No tokenizer found for model. Use --tokenizer to specify one."
            ),
        };

        let mf = gguf_path.to_string_lossy().to_string();
        let arch = config::detect_arch(&mf);
        let display = catalog::find(model_name)
            .map(|e| e.display.to_string())
            .unwrap_or_else(|| model_name.to_string());

        eprintln!("Model: {}", display);
        let eng = Engine::load_with_device(
            &gguf_path, &tok_path, arch, tq_config.clone(), cli.cpu,
        )?;
        (eng, mf, display)
    } else {
        anyhow::bail!(
            "Unknown model: '{}'. Use `tq list --available` to see available models.",
            model_name
        );
    };

    let template = chat::ChatTemplate::detect(&model_file);

    serve::run_server(engine, template, display_name, port, tq_config, cli.cpu).await
}

fn cmd_doctor() -> Result<()> {
    println!("tq doctor -- System Compatibility Check");
    println!("======================================");

    // CPU features
    println!("\nCPU:");
    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "  AVX2:  {}",
            if is_x86_feature_detected!("avx2") { "yes" } else { "no" }
        );
        println!(
            "  FMA:   {}",
            if is_x86_feature_detected!("fma") { "yes" } else { "no" }
        );
        println!(
            "  AVX512: {}",
            if is_x86_feature_detected!("avx512f") { "yes" } else { "no" }
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("  (non-x86 platform, SIMD detection skipped)");
    }

    // CUDA
    println!("\nGPU:");
    match candle_core::Device::cuda_if_available(0) {
        Ok(dev) if dev.is_cuda() => println!("  CUDA:  yes (device 0)"),
        _ => println!("  CUDA:  no (CPU only)"),
    }

    // Models directory
    let models_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".tq")
        .join("models");
    println!("\nModels directory: {}", models_dir.display());

    // Known models (legacy config)
    println!("\nLegacy models (config.rs):");
    for m in config::ALL_MODELS {
        println!("  {} ({})", m.name, m.display_name);
    }

    // Catalog models
    println!("\nModel Hub catalog:");
    for e in catalog::list_available() {
        let pulled = if hub::is_downloaded(e.name, e.tag) { " [pulled]" } else { "" };
        println!("  {}:{} — {}{}", e.name, e.tag, e.display, pulled);
    }

    // Downloaded models
    let downloaded = hub::list_downloaded();
    if !downloaded.is_empty() {
        println!("\nPulled models:");
        for dm in &downloaded {
            let status = if dm.gguf_exists { "ready" } else { "cache missing" };
            println!("  {}:{} — {} ({})", dm.meta.name, dm.meta.tag, dm.meta.display, status);
        }
    }

    Ok(())
}

fn cmd_perplexity(cli: &Cli) -> Result<()> {
    let (model_name, model_path_override, tokenizer_repo_override, file, chunk, turbo_quant) =
        match &cli.command {
            Some(Commands::Perplexity {
                model, model_path, tokenizer_repo, file, chunk, turbo_quant,
            }) => (
                model.as_str(), model_path.as_deref(), tokenizer_repo.as_deref(),
                file, *chunk, *turbo_quant,
            ),
            _ => unreachable!(),
        };

    let tq_config = resolve_tq_config(turbo_quant, cli.tq_bits);

    // Try legacy config first, then fall back to hub resolution
    let mut engine = if let Some(model_config) = config::get_model(model_name) {
        model::load_engine(
            model_config, model_path_override,
            tq_config, tokenizer_repo_override, cli.cpu,
        )?
    } else if model_path_override.is_none() {
        let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
        let tok_path = match (tokenizer_repo_override, tokenizer_path) {
            (Some(repo), _) => {
                let local = std::path::Path::new(repo).join("tokenizer.json");
                if local.exists() {
                    local
                } else if std::path::Path::new(repo).exists() && repo.ends_with(".json") {
                    PathBuf::from(repo)
                } else {
                    let api = hf_hub::api::sync::Api::new()?;
                    api.model(repo.to_string()).get("tokenizer.json")?
                }
            }
            (None, Some(tp)) => tp,
            (None, None) => anyhow::bail!(
                "No tokenizer found for model. Use --tokenizer-repo to specify one."
            ),
        };
        let mf = gguf_path.to_string_lossy().to_string();
        let arch = config::detect_arch(&mf);
        Engine::load_with_device(&gguf_path, &tok_path, arch, tq_config, cli.cpu)?
    } else {
        anyhow::bail!(
            "Unknown model: '{}'. Use `tq list --available` to see available models.",
            model_name
        );
    };

    let text = std::fs::read_to_string(file)
        .with_context(|| format!("Cannot read perplexity file: {}", file.display()))?;
    let ppl = engine.compute_perplexity(&text, chunk)?;
    println!("Perplexity: {:.3}", ppl);
    Ok(())
}

fn cmd_bench(cli: &Cli) -> Result<()> {
    let (model_name, tokens, json_output, custom_prompt, tq_only) = match &cli.command {
        Some(Commands::Bench { model, tokens, json, prompt, tq_only }) => {
            (model.as_str(), *tokens, *json, prompt.as_deref(), *tq_only)
        }
        _ => unreachable!(),
    };

    let bench_prompt = custom_prompt.unwrap_or(
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

    // Detect chat template for proper prompt formatting
    let formatted_prompt = if let Some(entry) = catalog::find(model_name) {
        let lower = entry.arch.to_lowercase();
        if lower.contains("qwen") {
            format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", bench_prompt)
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

    // --- Run 1: Standard (no TQ) ---
    let std_result = if tq_only {
        None
    } else {
        if !json_output {
            eprintln!("[1/2] Loading model (standard)...");
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

    // --- Run 2: TurboQuant 4-bit ---
    if !json_output {
        if tq_only { eprintln!("[1/1] Loading model (TQ 4-bit)..."); }
        else { eprintln!("[2/2] Loading model (TQ 4-bit)..."); }
    }
    let tq_config = tq_kv::TurboQuantConfig::balanced(); // 4-bit
    let mut engine_tq = load_engine(Some(tq_config))?;

    let tq_result = bench_run(&mut engine_tq, &formatted_prompt, &gen_params)?;
    drop(engine_tq);

    if !json_output {
        eprintln!("  TQ 4-bit: {:.1} tok/s, {:.2}s total, TTFT {:.3}s\n",
            tq_result.tok_per_sec, tq_result.total_secs, tq_result.ttft_secs);
    }

    // --- KV cache estimates ---
    let (n_layers, n_kv_heads, head_dim) = if let Some(entry) = catalog::find(model_name) {
        auto_tq::estimate_arch_params(entry.arch, entry.size_gb)
    } else {
        (32, 8, 128) // defaults
    };
    let kv_bytes = 2 * n_layers * n_kv_heads * head_dim * 2 * (tokens as usize);
    let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);
    let kv_compressed_mb = kv_mb / 3.8;
    let compression_ratio = 3.8;

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
            "kv_cache_mb": kv_mb,
            "kv_compressed_mb": kv_compressed_mb,
            "compression_ratio": compression_ratio,
        });
        if let Some(ref s) = std_result {
            json["standard"] = serde_json::json!({
                "total_secs": s.total_secs,
                "tok_per_sec": s.tok_per_sec,
                "ttft_secs": s.ttft_secs,
                "tokens_generated": s.tokens_generated,
            });
            json["speedup_pct"] = serde_json::json!(
                ((tq_result.tok_per_sec / s.tok_per_sec) - 1.0) * 100.0
            );
        }
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        println!();
        println!("+============================================================+");
        println!("|  TurboQuant Benchmark -- {:<33}|", display_name);
        println!("+============================================================+");

        if let Some(ref s) = std_result {
            let time_delta = if tq_result.total_secs < s.total_secs {
                format!("-{:.0}%", (1.0 - tq_result.total_secs / s.total_secs) * 100.0)
            } else {
                format!("+{:.0}%", (tq_result.total_secs / s.total_secs - 1.0) * 100.0)
            };
            let toks_delta = if tq_result.tok_per_sec > s.tok_per_sec {
                format!("+{:.0}%", (tq_result.tok_per_sec / s.tok_per_sec - 1.0) * 100.0)
            } else {
                format!("-{:.0}%", (1.0 - tq_result.tok_per_sec / s.tok_per_sec) * 100.0)
            };
            let ttft_delta = if tq_result.ttft_secs > s.ttft_secs {
                format!("+{:.0}%", (tq_result.ttft_secs / s.ttft_secs - 1.0) * 100.0)
            } else {
                format!("-{:.0}%", (1.0 - tq_result.ttft_secs / s.ttft_secs) * 100.0)
            };

            println!("| {:<20} | {:<11} | {:<11} | {:<7} |", "Metric", "Standard", "TQ 4-bit", "Delta");
            println!("+----------------------+-------------+-------------+---------+");
            println!("| {:<20} | {:<11} | {:<11} | {:<7} |",
                "Tokens generated", s.tokens_generated, tq_result.tokens_generated, "");
            println!("| {:<20} | {:<11} | {:<11} | {:<7} |",
                "Total time", format!("{:.2}s", s.total_secs), format!("{:.2}s", tq_result.total_secs), time_delta);
            println!("| {:<20} | {:<11} | {:<11} | {:<7} |",
                "tok/s", format!("{:.1}", s.tok_per_sec), format!("{:.1}", tq_result.tok_per_sec), toks_delta);
            println!("| {:<20} | {:<11} | {:<11} | {:<7} |",
                "TTFT", format!("{:.3}s", s.ttft_secs), format!("{:.3}s", tq_result.ttft_secs), ttft_delta);
        } else {
            println!("| {:<20} | {:<11} |", "Metric", "TQ 4-bit");
            println!("+----------------------+-------------+");
            println!("| {:<20} | {:<11} |", "Tokens generated", tq_result.tokens_generated);
            println!("| {:<20} | {:<11} |", "Total time", format!("{:.2}s", tq_result.total_secs));
            println!("| {:<20} | {:<11} |", "tok/s", format!("{:.1}", tq_result.tok_per_sec));
            println!("| {:<20} | {:<11} |", "TTFT", format!("{:.3}s", tq_result.ttft_secs));
        }

        println!("| {:<20} | {:<11} | {:<11} | {:<7} |",
            "KV cache (est.)", format!("{:.0} MB", kv_mb), format!("{:.0} MB", kv_compressed_mb),
            format!("-{:.0}%", (1.0 - 1.0 / compression_ratio) * 100.0));
        println!("| {:<20} | {:<11} | {:<11} | {:<7} |",
            "Compression ratio", "1.0x", format!("{:.1}x", compression_ratio), "");
        println!("+============================================================+");
    }

    Ok(())
}

struct BenchResult {
    tokens_generated: u32,
    total_secs: f64,
    tok_per_sec: f64,
    ttft_secs: f64,
}

fn bench_run(
    engine: &mut Engine,
    prompt: &str,
    params: &GenerationParams,
) -> Result<BenchResult> {
    engine.clear_cache();

    let start = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut token_count = 0u32;

    let _output = engine.generate(prompt, params, |_token_text| {
        if first_token_time.is_none() {
            first_token_time = Some(Instant::now());
        }
        token_count += 1;
    })?;

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

fn cmd_legacy(cli: &Cli) -> Result<()> {
    eprintln!("Note: Legacy mode. Consider using `tq chat <model>` instead.");

    let prompt = cli.prompt.as_deref().unwrap();

    // Legacy defaults to llama3-8b
    let model_config = config::get_model("llama3-8b").unwrap();
    eprintln!("Model: {}", model_config.display_name);

    let tq_config = resolve_tq_config(false, cli.tq_bits);

    let mut engine = model::load_engine(
        model_config, None, tq_config, None, cli.cpu,
    )?;

    let template = chat::ChatTemplate::detect(model_config.gguf_filename);
    let formatted = chat::format_chat(&template, config::DEFAULT_SYSTEM_PROMPT, prompt);

    engine.clear_cache();
    inference::generate(&mut engine, &formatted, &GenerationParams::default())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive chat (reused by cmd_chat)
// ---------------------------------------------------------------------------

fn run_interactive(
    engine: &mut Engine,
    system_prompt: &str,
    gen_params: &GenerationParams,
    template: &chat::ChatTemplate,
) -> Result<()> {
    eprintln!("\n--- Interactive Chat (TurboQuant Engine) ---");
    eprintln!("Type 'q' or 'quit' to exit.\n");

    let stdin = io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        print!("You > ");
        io::stdout().flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if matches!(input, "q" | "quit" | "exit") {
            eprintln!("Goodbye!");
            break;
        }

        let formatted = chat::format_multi_turn(template, system_prompt, &history, input);

        print!("Assistant > ");
        io::stdout().flush()?;

        engine.clear_cache();
        let response = inference::generate(engine, &formatted, gen_params)?;
        println!();

        history.push((input.to_string(), response));
    }

    Ok(())
}
