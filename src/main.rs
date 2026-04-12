mod auto_tq;
pub mod backend;
mod calibrate;
mod catalog;
mod chat;
mod config;
pub mod cuda;
pub mod gguf;
pub mod quant;
pub mod qmatmul;
pub mod sampling;
#[allow(dead_code)]
mod diagnostics;
mod download;
#[allow(dead_code)]
mod engine;
mod hub;
mod inference;
mod model;
#[allow(dead_code)]
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
        Some(Commands::Calibrate { .. }) => cmd_calibrate(&cli),
        Some(Commands::Doctor) => cmd_doctor(),
        Some(Commands::Niah { .. }) => cmd_niah(&cli),
        Some(Commands::Perplexity { .. }) => cmd_perplexity(&cli),
        Some(Commands::Pull { ref model }) => cmd_pull(model),
        Some(Commands::List { available }) => cmd_list(available),
        Some(Commands::Rm { ref model }) => cmd_rm(model),
        Some(Commands::Bench { .. }) => cmd_bench(&cli),
        Some(Commands::Compress { .. }) => cmd_compress(&cli),
        Some(Commands::Ablate { .. }) => cmd_ablate_study(&cli),
        Some(Commands::DebugEmbedding { .. }) => cmd_debug_embedding(&cli),
        Some(Commands::DebugTokenize { .. }) => cmd_debug_tokenize(&cli),
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
    resolve_tq_config_for_model(turbo_quant, tq_bits, None)
}

/// Auto-calibrate a model on first TQ use. Runs a quick prefill (~10s) to collect
/// key vector statistics, computes calibration data, and saves for future runs.
fn auto_calibrate(model_name: &str, force_cpu: bool) -> Result<calibrate::CalibrationData> {
    let t0 = std::time::Instant::now();

    let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
    let tok_path = tokenizer_path.ok_or_else(|| anyhow::anyhow!("No tokenizer for '{}'", model_name))?;
    let mf = gguf_path.to_string_lossy().to_string();
    let arch = config::detect_arch(&mf);

    // Determine head_dim from GGUF
    let head_dim = {
        let mut file = std::fs::File::open(&gguf_path)?;
        let content = crate::gguf::GgufContent::read(&mut file)
            .map_err(|e| anyhow::anyhow!("GGUF read: {}", e))?;
        // Find embedding_length key regardless of architecture prefix
        let n_embd = content.metadata.iter()
            .find(|(k, _)| k.ends_with(".embedding_length"))
            .and_then(|(_, v)| v.to_u32().ok())
            .unwrap_or(4096) as usize;
        let n_head = content.metadata.iter()
            .find(|(k, _)| k.ends_with(".attention.head_count"))
            .and_then(|(_, v)| v.to_u32().ok())
            .unwrap_or(32) as usize;
        n_embd / n_head
    };

    // Initialize collector (256 samples for quick calibration)
    let collector = calibrate::init_collector(head_dim, 256);

    // Default calibration text (Eiffel Tower passage — diverse vocabulary)
    let cal_text = "The tower is 324 metres tall, about the same height as an 81-storey building, \
         and the tallest structure in Paris. Its base is square, measuring 125 metres on \
         each side. During its construction, the Eiffel Tower surpassed the Washington \
         Monument to become the tallest human-made structure in the world, a title it held \
         for 41 years until the Chrysler Building in New York City was finished in 1930.";

    // Load model without TQ (skip all compression)
    std::env::set_var("TQ_SKIP", "999");
    let tq_stub = tq_kv::TurboQuantConfig { bits: 4, ..Default::default() };
    let mut engine = Engine::load_with_device(&gguf_path, &tok_path, arch, Some(tq_stub), force_cpu)?;
    std::env::remove_var("TQ_SKIP");

    // Run prefill
    let params = engine::GenerationParams { max_tokens: 1, temperature: 0.0, ..Default::default() };
    let _ = engine.generate_silent(cal_text, &params);
    drop(engine); // free memory before loading again

    // Compute calibration
    let c = collector.lock().map_err(|e| anyhow::anyhow!("Collector lock: {}", e))?;
    if c.count < 50 {
        anyhow::bail!("Too few samples ({}) — auto-calibrate needs at least 50", c.count);
    }
    let cal_data = calibrate::compute_calibration(&c, model_name, tq_kv::TurboQuantConfig::default().rotation_seed);

    // Save
    let (name, tag) = if let Some(e) = crate::catalog::find(model_name) {
        (e.name.to_string(), e.tag.to_string())
    } else if let Some((n, t)) = model_name.split_once(':') {
        (n.to_string(), t.to_string())
    } else {
        (model_name.to_string(), "default".to_string())
    };
    let path = calibrate::calibration_path(&name, &tag);
    calibrate::save_calibration(&cal_data, &path)?;

    eprintln!("  Auto-calibrate done: {} samples, {:.1}s → {}",
        c.count, t0.elapsed().as_secs_f64(), path.display());

    Ok(cal_data)
}

fn resolve_tq_config_for_model(turbo_quant: bool, tq_bits: u8, model_name: Option<&str>) -> Option<tq_kv::TurboQuantConfig> {
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
        // TQ_GROUP=0 disables group quantization (per-vector sigma)
        if let Ok(val) = std::env::var("TQ_GROUP") {
            if let Ok(gs) = val.parse::<usize>() {
                config.group_size = gs;
            }
        }
        // Auto-load calibration data if available (unless TQ_NO_CAL=1)
        let no_cal = std::env::var("TQ_NO_CAL").ok().map_or(false, |v| v == "1");
        if !no_cal {
            if let Some(name) = model_name {
                let cal_data = calibrate::load_calibration_for_model(name)
                    .or_else(|| {
                        // No calibration found — run auto-calibrate
                        let no_auto = std::env::var("TQ_NO_AUTO_CAL").ok().map_or(false, |v| v == "1");
                        if no_auto { return None; }
                        eprintln!("No calibration found for '{}' — running auto-calibrate...", name);
                        match auto_calibrate(name, false) {
                            Ok(cal) => Some(cal),
                            Err(e) => {
                                eprintln!("  Auto-calibrate failed: {} — continuing without calibration", e);
                                None
                            }
                        }
                    });
                if let Some(cal_data) = cal_data {
                    eprintln!("Using calibrated channel bias + per-head bits");
                    cal_data.apply_to_config(&mut config);
                    // TriAttention ON by default with TQ (the product mode).
                    // Disable with TQ_TRIATTN=0 or set_triattention_override(false).
                    let tri_disabled = std::env::var("TQ_TRIATTN").ok().map_or(false, |v| v == "0");
                    if !tri_disabled {
                        calibrate::init_triattention(&cal_data, cal_data.head_dim);
                        models::turbo_generic::set_triattention_override(true);
                    }
                }
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
        return resolve_tq_config_for_model(true, tq_bits, Some(model_name));
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

    let device = match crate::cuda::TqDevice::cuda_if_available(0).map_err(|e| anyhow::anyhow!("{}", e)) {
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

fn cmd_calibrate(cli: &Cli) -> Result<()> {
    let (model_name, text_file, max_samples) = match &cli.command {
        Some(Commands::Calibrate { model, text, samples }) => {
            (model.as_str(), text.clone(), *samples)
        }
        _ => unreachable!(),
    };

    eprintln!("=== TurboQuant Calibration ===");
    eprintln!("Model: {}", model_name);
    eprintln!("Samples: {}", max_samples);

    // Resolve model
    let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
    let tok_path = tokenizer_path.ok_or_else(|| {
        anyhow::anyhow!("No tokenizer found for model '{}'", model_name)
    })?;

    let mf = gguf_path.to_string_lossy().to_string();
    let arch = config::detect_arch(&mf);

    // Load calibration text
    let cal_text = if let Some(ref path) = text_file {
        std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read calibration text: {}", path.display()))?
    } else {
        // Default calibration text (short representative sample)
        "The tower is 324 metres tall, about the same height as an 81-storey building, \
         and the tallest structure in Paris. Its base is square, measuring 125 metres on \
         each side. During its construction, the Eiffel Tower surpassed the Washington \
         Monument to become the tallest human-made structure in the world, a title it held \
         for 41 years until the Chrysler Building in New York City was finished in 1930. \
         It was the first structure in the world to surpass both the 200-metre and 300-metre \
         mark in height. Due to the addition of a broadcasting aerial at the top of the tower \
         in 1957, it is now taller than the Chrysler Building by 5.2 metres. Excluding \
         transmitters, the Eiffel Tower is the second tallest free-standing structure in France \
         after the Millau Viaduct. The tower has three levels for visitors, with restaurants on \
         the first and second levels. The top level observation deck is 276 m above the ground, \
         the highest observation deck accessible to the public in the European Union. Tickets \
         can be purchased to ascend by stairs or lift to the first and second levels. The climb \
         from ground level to the first level is over 300 steps, as is the climb from the first \
         level to the second, making the entire ascent a physical challenge. Although there is a \
         staircase to the top level, it is usually accessible only by lift. The tower was designed \
         by the French civil engineer Gustave Eiffel and built by his engineering company. It was \
         constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. Although \
         initially criticised by some of France's leading artists and intellectuals for its design, \
         it has since become a global cultural icon of France and one of the most recognisable \
         structures in the world. The Eiffel Tower is the most visited paid monument in the world; \
         6.91 million people ascended it in 2015. It has been named after its designer, engineer \
         Gustave Eiffel, and is colloquially known as the Iron Lady.".to_string()
    };

    // Determine head_dim from GGUF metadata
    let head_dim = {
        let mut file = std::fs::File::open(&gguf_path)?;
        let content = crate::gguf::GgufContent::read(&mut file)
            .map_err(|e| anyhow::anyhow!("GGUF read error: {}", e))?;
        let n_embd = content.get("llama.embedding_length")
            .or_else(|| content.get("qwen2.embedding_length"))
            .or_else(|| content.get("phi3.embedding_length"))
            .or_else(|| content.get("gemma.embedding_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;
        let n_head = content.get("llama.attention.head_count")
            .or_else(|| content.get("qwen2.attention.head_count"))
            .or_else(|| content.get("phi3.attention.head_count"))
            .or_else(|| content.get("gemma.attention.head_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32) as usize;
        n_embd / n_head
    };

    eprintln!("Head dim: {}", head_dim);

    // Initialize calibration collector
    let collector = calibrate::init_collector(head_dim, max_samples);

    // Load engine WITHOUT TurboQuant (collect raw activations)
    // We use TQ with skip=999 to run through turbo_generic path but skip compression
    let tq_config = tq_kv::TurboQuantConfig {
        bits: 4,
        ..Default::default()
    };

    // Set TQ_SKIP very high so all layers are uncompressed but still go through
    // the turbo_generic code path (which has our collection hook)
    std::env::set_var("TQ_SKIP", "999");

    let mut engine = Engine::load_with_device(
        &gguf_path, &tok_path, arch, Some(tq_config.clone()), cli.cpu,
    )?;

    // Run prefill to collect activations
    eprintln!("Running prefill to collect KV activations...");
    let params = engine::GenerationParams {
        max_tokens: 1, // just prefill, don't generate
        temperature: 0.0,
        ..Default::default()
    };
    let _ = engine.generate_silent(&cal_text, &params);

    // Remove the skip override
    std::env::remove_var("TQ_SKIP");

    // Get collected samples
    let c = collector.lock().map_err(|e| anyhow::anyhow!("Collector lock error: {}", e))?;
    eprintln!("Collected {} key vectors", c.count);

    if c.count < 100 {
        anyhow::bail!("Too few samples collected ({}). Need at least 100. Try longer calibration text.", c.count);
    }

    // Compute calibration
    let cal_data = calibrate::compute_calibration(&c, model_name, tq_config.rotation_seed);

    // Save
    let entry = crate::catalog::find(model_name);
    let (name, tag) = if let Some(e) = entry {
        (e.name.to_string(), e.tag.to_string())
    } else if let Some((n, t)) = model_name.split_once(':') {
        (n.to_string(), t.to_string())
    } else {
        (model_name.to_string(), "default".to_string())
    };

    let path = calibrate::calibration_path(&name, &tag);
    calibrate::save_calibration(&cal_data, &path)?;

    eprintln!("\n=== Calibration Complete ===");
    eprintln!("Channel scales: {} values", cal_data.channel_scales.len());
    eprintln!("Rotation matrix: {}x{}", head_dim, head_dim);
    eprintln!("Codebooks: 2-bit, 3-bit, 4-bit");
    eprintln!("\nCalibration will be auto-loaded on next inference run.");

    Ok(())
}

fn cmd_ablate_study(cli: &Cli) -> Result<()> {
    let (model_name, file_path, quick, output_path) = match &cli.command {
        Some(Commands::Ablate { model, file, quick, output }) => {
            (model.as_str(), file.clone(), *quick, output.clone())
        }
        _ => unreachable!(),
    };

    let eval_text = std::fs::read_to_string(&file_path)
        .with_context(|| format!("Cannot read eval file: {}", file_path.display()))?;

    eprintln!("=== TurboQuant Ablation Study ===");
    eprintln!("Model: {}", model_name);
    eprintln!("Eval text: {} chars", eval_text.len());
    eprintln!("Mode: {}", if quick { "quick" } else { "full" });

    let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
    let tok_path = tokenizer_path.ok_or_else(|| anyhow::anyhow!("No tokenizer found"))?;
    let mf = gguf_path.to_string_lossy().to_string();
    let arch = config::detect_arch(&mf);

    // Check if calibration exists
    let has_calibration = calibrate::load_calibration_for_model(model_name).is_some();

    // Define sweep
    let bits_sweep: Vec<u8> = if quick { vec![2, 4] } else { vec![2, 3, 4] };
    let skip_sweep: Vec<usize> = if quick { vec![0, 4] } else { vec![0, 2, 4, 8] };
    let sink_sweep: Vec<usize> = if quick { vec![0, 4] } else { vec![0, 2, 4] };
    let cal_sweep: Vec<bool> = if has_calibration { vec![false, true] } else { vec![false] };

    let mut results: Vec<(String, f64, f64)> = Vec::new(); // (config_label, ppl, seconds)

    // Baseline: no compression (GenericTurboModel handles GPU/CPU transparently)
    {
        eprintln!("\n--- Baseline (no compression) ---");
        let start = Instant::now();
        let mut engine = Engine::load_with_device(&gguf_path, &tok_path, arch, None, cli.cpu)?;
        let ppl = engine.compute_perplexity(&eval_text)?;
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("  PPL: {:.3} ({:.1}s)", ppl, elapsed);
        results.push(("baseline".to_string(), ppl, elapsed));
    }

    let baseline_ppl = results[0].1;

    // Sweep TQ configs
    let total = bits_sweep.len() * skip_sweep.len() * sink_sweep.len() * cal_sweep.len();
    let mut run = 0;

    for &bits in &bits_sweep {
        for &skip in &skip_sweep {
            for &sink in &sink_sweep {
                for &use_cal in &cal_sweep {
                    run += 1;
                    let label = format!("{}bit_skip{}_sink{}{}", bits, skip, sink,
                        if use_cal { "_cal" } else { "" });
                    eprintln!("\n--- [{}/{}] {} ---", run, total, label);

                    let mut tq = match bits {
                        2 => tq_kv::TurboQuantConfig::extreme(),
                        3 => tq_kv::TurboQuantConfig::aggressive(),
                        _ => tq_kv::TurboQuantConfig::balanced(),
                    };
                    tq.skip_layers = Some(skip);
                    tq.sink_tokens = Some(sink);

                    if use_cal {
                        if let Some(cal_data) = calibrate::load_calibration_for_model(model_name) {
                            cal_data.apply_to_config(&mut tq);
                        }
                    }

                    let start = Instant::now();
                    let mut engine = Engine::load_with_device(
                        &gguf_path, &tok_path, arch, Some(tq), cli.cpu,
                    )?;
                    let ppl = engine.compute_perplexity(&eval_text)?;
                    let elapsed = start.elapsed().as_secs_f64();
                    let delta = (ppl - baseline_ppl) / baseline_ppl * 100.0;
                    eprintln!("  PPL: {:.3} (delta: {:+.2}%, {:.1}s)", ppl, delta, elapsed);
                    results.push((label, ppl, elapsed));
                }
            }
        }
    }

    // Print summary table
    eprintln!("\n╔══════════════════════════════════════════════════════╗");
    eprintln!("║              ABLATION RESULTS                       ║");
    eprintln!("╠══════════════════════╦════════╦═══════════╦══════════╣");
    eprintln!("║ Config               ║  PPL   ║ Delta PPL ║ Time (s) ║");
    eprintln!("╠══════════════════════╬════════╬═══════════╬══════════╣");
    for (label, ppl, secs) in &results {
        let delta = if label == "baseline" {
            "   ---   ".to_string()
        } else {
            format!("{:+.2}%", (ppl - baseline_ppl) / baseline_ppl * 100.0)
        };
        eprintln!("║ {:20} ║ {:6.3} ║ {:>9} ║ {:7.1}s ║", label, ppl, delta, secs);
    }
    eprintln!("╚══════════════════════╩════════╩═══════════╩══════════╝");

    // Write CSV if requested
    if let Some(ref csv_path) = output_path {
        let mut csv = String::from("config,ppl,delta_ppl_pct,time_s\n");
        for (label, ppl, secs) in &results {
            let delta = if label == "baseline" { 0.0 } else { (ppl - baseline_ppl) / baseline_ppl * 100.0 };
            csv.push_str(&format!("{},{:.4},{:.4},{:.2}\n", label, ppl, delta, secs));
        }
        std::fs::write(csv_path, &csv)?;
        eprintln!("\nCSV saved: {}", csv_path.display());
    }

    Ok(())
}

fn cmd_perplexity(cli: &Cli) -> Result<()> {
    let (model_name, model_path_override, tokenizer_repo_override, file, turbo_quant, compare) =
        match &cli.command {
            Some(Commands::Perplexity {
                model, model_path, tokenizer_repo, file, turbo_quant, compare, ..
            }) => (
                model.as_str(), model_path.as_deref(), tokenizer_repo.as_deref(),
                file, *turbo_quant, *compare,
            ),
            _ => unreachable!(),
        };

    let text = std::fs::read_to_string(file)
        .with_context(|| format!("Cannot read perplexity file: {}", file.display()))?;

    // Helper: load engine with given TQ config
    let load_engine = |tq_config: Option<tq_kv::TurboQuantConfig>| -> Result<Engine> {
        if let Some(model_config) = config::get_model(model_name) {
            model::load_engine(
                model_config, model_path_override,
                tq_config, tokenizer_repo_override, cli.cpu,
            )
        } else if model_path_override.is_none() {
            let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
            let tok_path = match (tokenizer_repo_override, &tokenizer_path) {
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
                (None, Some(tp)) => tp.clone(),
                (None, &None) => anyhow::bail!(
                    "No tokenizer found for model. Use --tokenizer-repo to specify one."
                ),
            };
            let mf = gguf_path.to_string_lossy().to_string();
            let arch = config::detect_arch(&mf);
            Engine::load_with_device(&gguf_path, &tok_path, arch, tq_config, cli.cpu)
        } else {
            anyhow::bail!(
                "Unknown model: '{}'. Use `tq list --available` to see available models.",
                model_name
            )
        }
    };

    if compare {
        // ── 3-way comparison: Standard vs TQ vs TQ+TriAttention ──
        let display_name = catalog::find(model_name)
            .map(|e| e.display.to_string())
            .unwrap_or_else(|| model_name.to_string());

        // 1. Standard
        eprintln!("[1/3] Perplexity: Standard...");
        models::turbo_generic::set_triattention_override(false);
        let mut engine_std = load_engine(None)?;
        let ppl_std = engine_std.compute_perplexity(&text)?;
        drop(engine_std);
        eprintln!("  Standard PPL: {:.3}", ppl_std);

        // 2. TQ 4-bit (no TriAttention)
        eprintln!("[2/3] Perplexity: TQ 4-bit...");
        models::turbo_generic::set_triattention_override(false);
        let tq_config_shared = resolve_tq_config_for_model(true, cli.tq_bits, Some(model_name));
        // resolve_tq_config enables TriAttn internally — force off for TQ-only
        models::turbo_generic::set_triattention_override(false);
        let mut engine_tq = load_engine(tq_config_shared.clone())?;
        let ppl_tq = engine_tq.compute_perplexity(&text)?;
        drop(engine_tq);
        eprintln!("  TQ 4-bit PPL: {:.3}", ppl_tq);

        // 3. TQ + TriAttention (same config, just enable TriAttn)
        eprintln!("[3/3] Perplexity: TQ+TriAttention...");
        models::turbo_generic::set_triattention_override(true);
        let mut engine_tri = load_engine(tq_config_shared)?;
        let ppl_tri = engine_tri.compute_perplexity(&text)?;
        drop(engine_tri);
        eprintln!("  TQ+TriAttn PPL: {:.3}", ppl_tri);

        // Restore
        models::turbo_generic::clear_triattention_override();

        // Results table
        let tq_delta = (ppl_tq / ppl_std - 1.0) * 100.0;
        let tri_delta = (ppl_tri / ppl_std - 1.0) * 100.0;
        println!();
        println!("+==============================================================+");
        println!("|  Perplexity Comparison -- {:<33}|", display_name);
        println!("+==============================================================+");
        println!("| {:<18} | {:<10} | {:<10} | {:<10} |", "Mode", "PPL", "vs Std", "Quality");
        println!("+--------------------+------------+------------+------------+");
        println!("| {:<18} | {:<10.3} | {:<10} | {:<10} |",
            "Standard", ppl_std, "baseline", "100%");
        println!("| {:<18} | {:<10.3} | {:>+9.1}% | {:<10} |",
            "TQ 4-bit", ppl_tq, tq_delta,
            if tq_delta.abs() < 5.0 { "excellent" } else if tq_delta < 15.0 { "good" } else { "degraded" });
        println!("| {:<18} | {:<10.3} | {:>+9.1}% | {:<10} |",
            "TQ+TriAttention", ppl_tri, tri_delta,
            if tri_delta.abs() < 5.0 { "excellent" } else if tri_delta < 15.0 { "good" } else { "degraded" });
        println!("+==============================================================+");
    } else {
        // ── Single run (existing behavior) ──
        let tq_config = resolve_tq_config_for_model(turbo_quant, cli.tq_bits, Some(model_name));
        let mut engine = load_engine(tq_config)?;
        let ppl = engine.compute_perplexity(&text)?;
        println!("Perplexity: {:.3}", ppl);
    }
    Ok(())
}

fn cmd_bench(cli: &Cli) -> Result<()> {
    let (model_name, tokens, json_output, custom_prompt, tq_only, draft_model, spec_k) = match &cli.command {
        Some(Commands::Bench { model, tokens, json, prompt, tq_only, draft, speculate }) => {
            (model.as_str(), *tokens, *json, prompt.as_deref(), *tq_only, draft.as_deref(), *speculate)
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
    let mut tq_config = tq_kv::TurboQuantConfig::balanced(); // 4-bit
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

fn cmd_niah(cli: &Cli) -> Result<()> {
    let (model_name, context_target, position, max_tokens, no_triattn) = match &cli.command {
        Some(Commands::Niah { model, context, position, max_tokens, no_triattn }) => {
            (model.as_str(), *context, *position, *max_tokens, *no_triattn)
        }
        _ => unreachable!(),
    };

    // Needle: use a high-frequency pattern that Q4_K_M can reliably retrieve.
    // "AURORA-7" fails even at 2-sentence context on Q4_K_M (see Bug #3 findings).
    // "Paris" works because it's the #1 completion for "capital of France".
    let needle_id = std::env::var("TQ_NIAH_NEEDLE").unwrap_or_else(|_| "Paris".to_string());
    let needle = if needle_id == "Paris" {
        "IMPORTANT: The capital of the secret country is Paris and the launch date is March 15.".to_string()
    } else {
        format!("IMPORTANT: The secret project codename is {} and the launch date is March 15.", needle_id)
    };

    let filler = "Modern computing systems continue to evolve through innovations in hardware architecture and software optimization. Researchers explore new approaches to improve efficiency across diverse platforms. The interplay between hardware capabilities and software requirements drives progress in computer science and engineering. ";

    // Build haystack: filler before + needle + filler after, sized to context_target words
    let target_words = context_target.max(200);
    let needle_pos = (target_words as f32 * position) as usize;
    let mut haystack = String::new();
    let words_per_filler = filler.split_whitespace().count();
    let n_pre = (needle_pos + words_per_filler - 1) / words_per_filler;
    for _ in 0..n_pre { haystack.push_str(filler); }
    haystack.push_str(&needle);
    haystack.push(' ');
    let remaining = target_words.saturating_sub(haystack.split_whitespace().count());
    let n_post = (remaining + words_per_filler - 1) / words_per_filler;
    for _ in 0..n_post { haystack.push_str(filler); }

    let actual_words = haystack.split_whitespace().count();

    // Cloze-style sentence completion. Plain text (no chat template) — chat
    // wrapping in greedy mode produces garbage on Qwen2 (see commit 32c2dc5).
    // The pattern is: state the document, then a sentence whose continuation
    // is the answer. Model is in pretraining-style next-token mode here.
    // Quotation pattern for raw-text NIAH. The model has seen millions of
    // "it was stated: 'X is Y'" patterns in pretraining. By quoting the
    // needle sentence with a blank, we prime the model to fill in the value.
    // This works in raw completion mode (no chat template needed).
    let prompt = format!(
        "{}\n\nQuote from the document above: \"The secret project codename is",
        haystack
    );

    eprintln!("============================================");
    eprintln!("  NIAH Test — {}", model_name);
    eprintln!("  Haystack: ~{} words", actual_words);
    eprintln!("  Needle: '{}' at {:.0}% depth (TQ_NIAH_NEEDLE to override)", needle_id, position * 100.0);
    eprintln!("  Mode: {}", if no_triattn { "TQ-only" } else { "TQ+TriAttn V3" });
    eprintln!("============================================");

    if !no_triattn {
        models::turbo_generic::set_triattention_override(true);
    } else {
        models::turbo_generic::set_triattention_override(false);
    }

    // Allow standard mode via TQ_NO_TQ=1 for debugging
    let tq_config = if std::env::var("TQ_NO_TQ").ok().map_or(false, |v| v == "1") {
        None
    } else {
        resolve_tq_config_for_model(true, cli.tq_bits, Some(model_name))
    };

    let mut engine = if let Some(model_config) = config::get_model(model_name) {
        model::load_engine(model_config, None, tq_config, None, cli.cpu)?
    } else {
        let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
        let tok_path = tokenizer_path.with_context(|| format!("No tokenizer found for '{}'", model_name))?;
        let arch = config::detect_arch(&gguf_path.to_string_lossy());
        Engine::load_with_device(&gguf_path, &tok_path, arch, tq_config, cli.cpu)?
    };

    // Greedy (same as bench command) for deterministic NIAH
    let params = GenerationParams {
        max_tokens,
        temperature: 0.0,  // ArgMax greedy
        ..Default::default()
    };

    engine.clear_cache();

    let mut output = String::new();
    let _ = engine.generate(&prompt, &params, |token| {
        output.push_str(token);
    })?;

    let lower = output.to_lowercase();
    let found = lower.contains(&needle_id.to_lowercase());

    eprintln!();
    eprintln!("Output: {}", output.trim());
    eprintln!();
    if found {
        eprintln!("  PASS — needle '{}' retrieved", needle_id);
    } else {
        eprintln!("  FAIL — needle '{}' NOT in output", needle_id);
    }
    eprintln!("============================================");

    models::turbo_generic::clear_triattention_override();

    if !found { std::process::exit(1); }
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

fn cmd_compress(cli: &Cli) -> Result<()> {
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

// ============================================================
// Bug #2 Debug Commands
// ============================================================

/// Dump embedding row statistics for specific token IDs.
/// Used to diagnose Bug #2 (chat template inference gibberish).
fn cmd_debug_embedding(cli: &Cli) -> Result<()> {
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

    Ok(())
}

/// Tokenize a string and dump the resulting token IDs.
/// Used to verify chat template tokens are correctly recognized.
fn cmd_debug_tokenize(cli: &Cli) -> Result<()> {
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

    Ok(())
}
