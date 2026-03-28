mod chat;
mod config;
mod diagnostics;
mod download;
mod engine;
mod inference;
mod model;
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
        /// Enable web UI
        #[arg(long)]
        ui: bool,
    },
    /// Download a model from HuggingFace
    Pull {
        /// Model name (e.g., qwen72b)
        model: String,
    },
    /// List downloaded models
    List,
    /// Remove a downloaded model
    Rm {
        /// Model name
        model: String,
    },
    /// Run performance benchmarks
    Bench {
        /// Model name or path
        model: String,
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
        Some(Commands::Pull { ref model }) => {
            eprintln!("tq pull {} -- coming in Phase 2", model);
            Ok(())
        }
        Some(Commands::List) => {
            eprintln!("tq list -- coming in Phase 2");
            Ok(())
        }
        Some(Commands::Rm { ref model }) => {
            eprintln!("tq rm {} -- coming in Phase 2", model);
            Ok(())
        }
        Some(Commands::Bench { ref model }) => {
            eprintln!("tq bench {} -- coming in Phase 3", model);
            Ok(())
        }
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
        let config = match tq_bits {
            2 => tq_kv::TurboQuantConfig::extreme(),
            3 => tq_kv::TurboQuantConfig::aggressive(),
            _ => tq_kv::TurboQuantConfig::balanced(),
        };
        Some(config)
    } else {
        None
    }
}

fn cmd_chat(cli: &Cli) -> Result<()> {
    let (model_name, system, max_tokens, temperature, top_p, top_k, repeat_penalty,
         turbo_quant, model_path_override, tokenizer_repo_override) = match &cli.command {
        Some(Commands::Chat {
            model, system, max_tokens, temperature, top_p, top_k, repeat_penalty,
            turbo_quant, model_path, tokenizer_repo,
        }) => (
            model.as_str(), system.as_str(), *max_tokens, *temperature, *top_p,
            *top_k, *repeat_penalty, *turbo_quant, model_path.as_deref(),
            tokenizer_repo.as_deref(),
        ),
        _ => unreachable!(),
    };

    let model_config = config::get_model(model_name).context(format!(
        "Unknown model: '{}'. Supported: {}",
        model_name,
        config::ALL_MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ")
    ))?;

    eprintln!("Model: {}", model_config.display_name);

    let tq_config = resolve_tq_config(turbo_quant, cli.tq_bits);

    let mut engine = model::load_engine(
        model_config, model_path_override,
        tq_config, tokenizer_repo_override, cli.cpu,
    )?;

    let model_file = model_path_override
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| model_config.gguf_filename.to_string());
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
    let (model_name, model_path_override, tokenizer_override, port, turbo_quant) =
        match &cli.command {
            Some(Commands::Serve { model, model_path, tokenizer, port, turbo_quant, .. }) => {
                (model.as_str(), model_path.as_deref(), tokenizer.as_deref(), *port, *turbo_quant)
            }
            _ => unreachable!(),
        };

    let model_config = config::get_model(model_name).context(format!(
        "Unknown model: '{}'. Supported: {}",
        model_name,
        config::ALL_MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ")
    ))?;

    eprintln!("Model: {}", model_config.display_name);

    let tq_config = resolve_tq_config(turbo_quant, cli.tq_bits);

    let engine = model::load_engine(
        model_config, model_path_override,
        tq_config, tokenizer_override, cli.cpu,
    )?;

    let model_file = model_path_override
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| model_config.gguf_filename.to_string());
    let template = chat::ChatTemplate::detect(&model_file);

    serve::run_server(engine, template, model_name.to_string(), port).await
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

    // Known models
    println!("\nSupported models:");
    for m in config::ALL_MODELS {
        println!("  {} ({})", m.name, m.display_name);
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

    let model_config = config::get_model(model_name).context(format!(
        "Unknown model: '{}'. Supported: {}",
        model_name,
        config::ALL_MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ")
    ))?;

    let tq_config = resolve_tq_config(turbo_quant, cli.tq_bits);

    let mut engine = model::load_engine(
        model_config, model_path_override,
        tq_config, tokenizer_repo_override, cli.cpu,
    )?;

    let text = std::fs::read_to_string(file)
        .with_context(|| format!("Cannot read perplexity file: {}", file.display()))?;
    let ppl = engine.compute_perplexity(&text, chunk)?;
    println!("Perplexity: {:.3}", ppl);
    Ok(())
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
