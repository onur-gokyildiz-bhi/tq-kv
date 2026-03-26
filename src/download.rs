use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::config::ModelConfig;

/// Ensure GGUF model file exists locally, download if needed.
pub fn ensure_model(config: &ModelConfig, models_dir: &std::path::Path) -> Result<PathBuf> {
    let local_path = models_dir.join(config.gguf_filename);

    if local_path.exists() {
        eprintln!("Model found: {}", local_path.display());
        return Ok(local_path);
    }

    eprintln!("Downloading model: {} -> {}", config.display_name, config.gguf_filename);

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
    pb.set_message(format!("Downloading {}...", config.gguf_filename));

    let api = Api::new().context("HuggingFace API init failed")?;
    let repo = api.model(config.hf_repo.to_string());
    let downloaded_path = repo.get(config.gguf_filename).context("Model download failed")?;

    pb.finish_with_message(format!("{} downloaded!", config.gguf_filename));
    Ok(downloaded_path)
}

/// Ensure tokenizer.json exists locally, download if needed.
pub fn ensure_tokenizer(config: &ModelConfig, models_dir: &std::path::Path) -> Result<PathBuf> {
    let tokenizer_filename = format!("{}-tokenizer.json", config.name);
    let local_path = models_dir.join(&tokenizer_filename);

    if local_path.exists() {
        return Ok(local_path);
    }

    eprintln!("Downloading tokenizer from {}", config.tokenizer_repo);

    let api = Api::new().context("HuggingFace API init failed")?;
    let repo = api.model(config.tokenizer_repo.to_string());
    let downloaded_path = repo.get("tokenizer.json").context("Tokenizer download failed")?;

    Ok(downloaded_path)
}
