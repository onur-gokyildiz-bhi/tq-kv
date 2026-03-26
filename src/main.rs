mod chat;
mod config;
mod download;
mod engine;
mod inference;
mod model;
mod models;
mod serve;

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use engine::{Engine, GenerationParams};

#[derive(Parser, Debug)]
#[command(
    name = "tq-engine",
    about = "TurboQuant-powered LLM inference engine — Pure Rust, compressed KV cache",
    version
)]
struct Args {
    /// Prompt text (leave empty for interactive mode)
    prompt: Option<String>,

    /// Model to use: "llama3-8b", "qwen72b", "gemma3-4b"
    #[arg(short, long, default_value = "llama3-8b")]
    model: String,

    /// System prompt
    #[arg(short, long, default_value = config::DEFAULT_SYSTEM_PROMPT)]
    system: String,

    /// Maximum tokens to generate
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

    /// Custom GGUF model path
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// HuggingFace repo for tokenizer (overrides default)
    #[arg(long)]
    tokenizer_repo: Option<String>,

    /// Interactive chat mode
    #[arg(short, long)]
    interactive: bool,

    /// Read text from file and combine with prompt
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// HTTP daemon mode
    #[arg(long)]
    serve: bool,

    /// Daemon port
    #[arg(long, default_value = "8088")]
    port: u16,

    /// Force CPU inference (skip CUDA)
    #[arg(long)]
    cpu: bool,

    /// Enable TurboQuant KV cache compression
    #[arg(long)]
    turbo_quant: bool,

    /// TurboQuant bit width (2, 3, or 4)
    #[arg(long, default_value = "4")]
    tq_bits: u8,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_config = config::get_model(&args.model).context(format!(
        "Unknown model: '{}'. Supported: {}",
        args.model,
        config::ALL_MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", ")
    ))?;

    eprintln!("Model: {}", model_config.display_name);

    let tq_config = if args.turbo_quant {
        let config = match args.tq_bits {
            2 => tq_kv::TurboQuantConfig::extreme(),
            3 => tq_kv::TurboQuantConfig::aggressive(),
            _ => tq_kv::TurboQuantConfig::balanced(),
        };
        Some(config)
    } else {
        None
    };

    let mut engine = model::load_engine(
        model_config, args.model_path.as_deref(),
        tq_config, args.tokenizer_repo.as_deref(), args.cpu,
    )?;

    let model_name = args
        .model_path
        .as_deref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| model_config.gguf_filename.to_string());
    let template = chat::ChatTemplate::detect(&model_name);

    if args.serve {
        return serve::run_daemon(engine, &template, args.port);
    }

    let gen_params = GenerationParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repeat_penalty: args.repeat_penalty,
        ..Default::default()
    };

    if let Some(file_path) = &args.file {
        let file_content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Cannot read file: {}", file_path.display()))?;
        let prompt = match &args.prompt {
            Some(p) => format!("{}\n\n---\n\n{}", p, file_content),
            None => file_content,
        };
        let formatted = chat::format_chat(&template, &args.system, &prompt);
        engine.clear_cache();
        inference::generate(&mut engine, &formatted, &gen_params)?;
        return Ok(());
    }

    if args.interactive || args.prompt.is_none() {
        run_interactive(&mut engine, &args.system, &gen_params, &template)?;
    } else {
        let prompt = args.prompt.unwrap();
        let formatted = chat::format_chat(&template, &args.system, &prompt);
        engine.clear_cache();
        inference::generate(&mut engine, &formatted, &gen_params)?;
    }

    Ok(())
}

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

        if input.is_empty() { continue; }
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
