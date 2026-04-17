mod auto_tq;
pub mod autocalib;
pub mod backend;
mod calibrate;
mod catalog;
mod chat;
mod cli;
mod config;
pub mod cuda;
pub mod gguf;
pub mod quant;
pub mod qmatmul;
pub mod safetensors_src;
pub mod sampling;
#[allow(dead_code)]
mod diagnostics;
mod download;
#[allow(dead_code)]
mod engine;
mod hub;
mod inference;
#[cfg(feature = "cuda")]
pub mod layer_swap;
mod model;
#[allow(dead_code)]
mod models;
mod serve;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use engine::{Engine, GenerationParams};

#[derive(Parser)]
#[command(
    name = "tq",
    about = "TurboQuant -- Local LLM inference with KV cache compression",
    version
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Commands>,

    // Legacy: if no subcommand, treat first arg as prompt (backward compat)
    /// Prompt text (legacy mode -- use `tq chat` instead)
    pub(crate) prompt: Option<String>,

    /// Force CPU inference
    #[arg(long, global = true)]
    pub(crate) cpu: bool,

    /// TurboQuant bit width (2, 3, or 4). 0 = auto
    #[arg(long, default_value = "4", global = true)]
    pub(crate) tq_bits: u8,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
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
        /// Draft model for speculative decoding (e.g., qwen2:0.5b)
        #[arg(long)]
        draft: Option<String>,
        /// Number of speculative tokens per step (default: 5)
        #[arg(long, default_value = "5")]
        speculate: usize,
    },
    /// NIAH (Needle-In-A-Haystack) test for TriAttention V3 validation
    Niah {
        /// Model name (e.g., qwen2:7b)
        model: String,
        /// Haystack token count (filler)
        #[arg(long, default_value = "1024")]
        context: usize,
        /// Needle position as fraction of context (0.5 = middle, 0.9 = end)
        #[arg(long, default_value = "0.9")]
        position: f32,
        /// Max tokens to generate for retrieval
        #[arg(long, default_value = "20")]
        max_tokens: u32,
        /// Use TQ + TriAttention (default: on)
        #[arg(long)]
        no_triattn: bool,
    },
    /// Check system compatibility
    Doctor,
    /// Auto-tune kernel dispatch: benchmarks each TQ_* variant and writes winners to per-machine cache
    Autotune {
        /// Model name (e.g., qwen2:7b, llama:8b)
        model: String,
        /// Tokens per variant mini-bench (default: 80)
        #[arg(short = 'n', long, default_value = "80")]
        tokens: u32,
        /// Quick mode — try fewer variants per dispatch (fast default-sanity check)
        #[arg(long)]
        quick: bool,
        /// Which dispatches to include (comma-separated subset of q4km,down,gateup,qkv,q6k).
        /// Default: all.
        #[arg(long)]
        only: Option<String>,
        /// Dry-run — print the plan without executing subprocesses
        #[arg(long)]
        dry_run: bool,
        /// Primary metric to maximise: "std" (default), "tq", or "tq+ta"
        #[arg(long, default_value = "std")]
        metric: String,
    },
    /// Calibrate TurboQuant for a model (computes optimal codebook, rotation, scales)
    Calibrate {
        /// Model name (e.g., qwen2:7b, llama:8b)
        model: String,
        /// Text file for calibration data (default: embedded wikitext sample)
        #[arg(long)]
        text: Option<PathBuf>,
        /// Number of key vectors to collect (default: 4096)
        #[arg(long, default_value = "4096")]
        samples: usize,
    },
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
        /// Run 3-way comparison: Standard vs TQ vs TQ+TriAttention
        #[arg(long)]
        compare: bool,
    },
    /// Compress model weights using TQ4_1S (WHT rotation + 4-bit quantization)
    Compress {
        /// Input GGUF model path
        input: PathBuf,
        /// Output compressed GGUF path (default: input_tq4.gguf)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Rotation seed (must match at load time)
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Boundary layers to skip (keep original quant for first/last N layers)
        #[arg(long, default_value = "2")]
        boundary: usize,
    },
    /// Run ablation study — sweep TQ configs and measure PPL for each
    Ablate {
        /// Model name (e.g., qwen2:7b, llama:8b)
        model: String,
        /// Text file for evaluation
        #[arg(long)]
        file: PathBuf,
        /// Quick mode (fewer configs)
        #[arg(long)]
        quick: bool,
        /// Output CSV file
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Debug: dump embedding row statistics (Bug #2 investigation)
    DebugEmbedding {
        /// Model name (e.g., qwen2:7b)
        model: String,
        /// Token IDs to inspect (comma-separated)
        #[arg(long, default_value = "1,1000,151643,151644,151645,151646")]
        tokens: String,
    },
    /// Debug: tokenize a string and dump token IDs (Bug #2 investigation)
    DebugTokenize {
        /// Model name (e.g., qwen2:7b)
        model: String,
        /// Text to tokenize
        text: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Serve { .. }) => cmd_serve(&cli).await,
        Some(Commands::Chat { .. }) => cmd_chat(&cli),
        Some(Commands::Calibrate { .. }) => cli::calibrate::cmd_calibrate(&cli),
        Some(Commands::Doctor) => cmd_doctor(),
        Some(Commands::Niah { .. }) => cli::niah::cmd_niah(&cli),
        Some(Commands::Perplexity { .. }) => cli::perplexity::cmd_perplexity(&cli),
        Some(Commands::Pull { ref model }) => cmd_pull(model),
        Some(Commands::List { available }) => cmd_list(available),
        Some(Commands::Rm { ref model }) => cmd_rm(model),
        Some(Commands::Bench { .. }) => cli::bench::cmd_bench(&cli),
        Some(Commands::Compress { .. }) => cli::compress::cmd_compress(&cli),
        Some(Commands::Ablate { .. }) => cli::ablate::cmd_ablate_study(&cli),
        Some(Commands::Autotune { .. }) => cli::autotune::cmd_autotune(&cli),
        Some(Commands::DebugEmbedding { .. }) => cli::debug::cmd_debug_embedding(&cli),
        Some(Commands::DebugTokenize { .. }) => cli::debug::cmd_debug_tokenize(&cli),
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
// Subcommand implementations (remaining in-line; heavier ones live in src/cli/)
// ---------------------------------------------------------------------------

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

    let tq_config = cli::resolve_tq_config_with_auto(
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

    let tq_config = cli::resolve_tq_config_with_auto(
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
    match crate::cuda::TqDevice::cuda_if_available(0).map_err(|e| anyhow::anyhow!("{}", e)) {
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

fn cmd_legacy(cli: &Cli) -> Result<()> {
    eprintln!("Note: Legacy mode. Consider using `tq chat <model>` instead.");

    let prompt = cli.prompt.as_deref().unwrap();

    // Legacy defaults to llama3-8b
    let model_config = config::get_model("llama3-8b").unwrap();
    eprintln!("Model: {}", model_config.display_name);

    let tq_config = cli::resolve_tq_config(false, cli.tq_bits);

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
