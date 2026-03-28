//! Model Hub — download manager for tq models.
//!
//! Uses the HuggingFace Hub API to download GGUF models and tokenizers.
//! Models are tracked via metadata files in `~/.tq/models/{name}-{tag}/`.
//! Actual files live in the HF cache (`~/.cache/huggingface/`) to avoid
//! duplicating multi-GB files.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::catalog::{self, CatalogEntry};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Get the tq models metadata directory (~/.tq/models/).
pub fn models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tq")
        .join("models")
}

/// Get model metadata directory for a specific model.
pub fn model_dir(name: &str, tag: &str) -> PathBuf {
    models_dir().join(format!("{}-{}", name, tag))
}

fn metadata_path(name: &str, tag: &str) -> PathBuf {
    model_dir(name, tag).join("metadata.json")
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub tag: String,
    pub display: String,
    pub gguf_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub size_gb: f32,
    pub arch: String,
    pub pulled_at: String,
}

fn read_metadata(name: &str, tag: &str) -> Option<ModelMetadata> {
    let path = metadata_path(name, tag);
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

fn write_metadata(meta: &ModelMetadata) -> Result<()> {
    let dir = model_dir(&meta.name, &meta.tag);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("metadata.json");
    let json = serde_json::to_string_pretty(meta)?;
    std::fs::write(&path, json)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// HF API helper
// ---------------------------------------------------------------------------

fn make_api() -> Result<Api> {
    // hf_hub respects HF_TOKEN env var automatically, but we can also
    // use ApiBuilder for more control if needed.
    Api::new().context("HuggingFace API init failed (check network / HF_TOKEN)")
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

/// Check if a model has been pulled (metadata exists and gguf file is present).
pub fn is_downloaded(name: &str, tag: &str) -> bool {
    match read_metadata(name, tag) {
        Some(meta) => meta.gguf_path.exists(),
        None => false,
    }
}

/// Download (pull) a model from HuggingFace.
///
/// Returns the metadata for the downloaded model.
pub fn download(entry: &CatalogEntry) -> Result<ModelMetadata> {
    let dir = model_dir(entry.name, entry.tag);
    std::fs::create_dir_all(&dir)?;

    // Check if already downloaded
    if let Some(meta) = read_metadata(entry.name, entry.tag) {
        if meta.gguf_path.exists() {
            eprintln!(
                "{} ({}) is already downloaded.",
                entry.display,
                format!("{}:{}", entry.name, entry.tag)
            );
            eprintln!("  GGUF: {}", meta.gguf_path.display());
            if let Some(ref tp) = meta.tokenizer_path {
                eprintln!("  Tokenizer: {}", tp.display());
            }
            return Ok(meta);
        }
    }

    eprintln!(
        "Pulling {} ({:.1} GB)...",
        entry.display, entry.size_gb
    );

    let api = make_api()?;

    // Download GGUF model
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(format!("Downloading {}...", entry.filename));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let repo = api.model(entry.hf_repo.to_string());
    let gguf_path = repo
        .get(entry.filename)
        .with_context(|| format!("Failed to download {} from {}", entry.filename, entry.hf_repo))?;

    pb.finish_with_message(format!("Downloaded {}", entry.filename));

    // Download tokenizer
    let tokenizer_path = {
        let pb2 = ProgressBar::new_spinner();
        pb2.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb2.set_message(format!("Downloading tokenizer from {}...", entry.tokenizer_repo));
        pb2.enable_steady_tick(std::time::Duration::from_millis(100));

        let tok_repo = api.model(entry.tokenizer_repo.to_string());
        match tok_repo.get("tokenizer.json") {
            Ok(path) => {
                pb2.finish_with_message("Tokenizer downloaded.");
                Some(path)
            }
            Err(e) => {
                pb2.finish_with_message(format!(
                    "Warning: tokenizer download failed: {}. Use --tokenizer flag.",
                    e
                ));
                None
            }
        }
    };

    // Record metadata (points to HF cache paths, no file duplication)
    let now = {
        use std::time::SystemTime;
        let d = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        // Simple ISO-ish timestamp
        let secs = d.as_secs();
        format!("{}", secs)
    };

    let meta = ModelMetadata {
        name: entry.name.to_string(),
        tag: entry.tag.to_string(),
        display: entry.display.to_string(),
        gguf_path,
        tokenizer_path,
        size_gb: entry.size_gb,
        arch: entry.arch.to_string(),
        pulled_at: now,
    };

    write_metadata(&meta)?;

    eprintln!("\nPulled {}:{}", entry.name, entry.tag);
    eprintln!("  GGUF: {}", meta.gguf_path.display());
    if let Some(ref tp) = meta.tokenizer_path {
        eprintln!("  Tokenizer: {}", tp.display());
    }

    Ok(meta)
}

// ---------------------------------------------------------------------------
// List downloaded models
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DownloadedModel {
    pub meta: ModelMetadata,
    pub gguf_exists: bool,
}

/// List all models that have been pulled.
pub fn list_downloaded() -> Vec<DownloadedModel> {
    let dir = models_dir();
    let mut results = Vec::new();

    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return results,
    };

    for entry in entries.flatten() {
        let meta_path = entry.path().join("metadata.json");
        if meta_path.exists() {
            if let Ok(data) = std::fs::read_to_string(&meta_path) {
                if let Ok(meta) = serde_json::from_str::<ModelMetadata>(&data) {
                    let gguf_exists = meta.gguf_path.exists();
                    results.push(DownloadedModel { meta, gguf_exists });
                }
            }
        }
    }

    results.sort_by(|a, b| a.meta.name.cmp(&b.meta.name).then(a.meta.tag.cmp(&b.meta.tag)));
    results
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

/// Remove a downloaded model's metadata.
///
/// This removes the metadata in ~/.tq/models/{name}-{tag}/ but does NOT
/// delete the cached files in ~/.cache/huggingface/ (those are managed
/// by hf_hub and shared across tools).
pub fn remove(name: &str, tag: &str) -> Result<()> {
    let dir = model_dir(name, tag);
    if !dir.exists() {
        bail!("Model {}:{} is not downloaded.", name, tag);
    }

    std::fs::remove_dir_all(&dir)
        .with_context(|| format!("Failed to remove {}", dir.display()))?;

    eprintln!("Removed {}:{} metadata.", name, tag);
    eprintln!(
        "Note: cached files in HuggingFace cache (~/.cache/huggingface/) were not deleted."
    );
    eprintln!("To reclaim disk space, run: huggingface-cli delete-cache");

    Ok(())
}

// ---------------------------------------------------------------------------
// Resolve — unified model resolution
// ---------------------------------------------------------------------------

/// Resolve a model query to (gguf_path, tokenizer_path).
///
/// Handles:
/// - "name:tag" — catalog lookup + hub resolution
/// - "name" — catalog lookup (smallest variant) + hub resolution
/// - Legacy config.rs names like "qwen72b", "llama3-8b"
/// - Absolute/relative file paths to .gguf files
///
/// If the model is in the catalog but not yet pulled, it will be
/// auto-downloaded.
pub fn resolve(query: &str) -> Result<(PathBuf, Option<PathBuf>)> {
    // Check if query is a file path
    let as_path = Path::new(query);
    if as_path.exists() && as_path.is_file() {
        // Look for tokenizer.json next to the model file
        let tok = as_path.parent().map(|p| p.join("tokenizer.json")).filter(|p| p.exists());
        return Ok((as_path.to_path_buf(), tok));
    }

    // Try catalog lookup
    if let Some(entry) = catalog::find(query) {
        // Check if already downloaded
        if let Some(meta) = read_metadata(entry.name, entry.tag) {
            if meta.gguf_path.exists() {
                return Ok((meta.gguf_path, meta.tokenizer_path));
            }
        }

        // Auto-download
        eprintln!("Model {}:{} not found locally. Pulling...", entry.name, entry.tag);
        let meta = download(entry)?;
        return Ok((meta.gguf_path, meta.tokenizer_path));
    }

    bail!(
        "Unknown model: '{}'\n\
         \n\
         Available models (use `tq list --available` or `tq pull <name>`):\n\
         {}\n\
         \n\
         Or pass a path to a GGUF file directly.",
        query,
        catalog::list_available()
            .iter()
            .map(|e| format!("  {}:{:<6} {}", e.name, e.tag, e.display))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
