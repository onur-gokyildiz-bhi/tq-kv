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

    // Try as HuggingFace repo name (e.g. "Qwen/Qwen2.5-7B-Instruct")
    if query.contains('/') {
        eprintln!("Resolving HuggingFace repo: {}", query);
        let safetensors_dir = resolve_hf_safetensors(query)?;
        // Return directory path — engine will detect safetensors format
        let tok = safetensors_dir.join("tokenizer.json");
        let tok_path = if tok.exists() { Some(tok) } else { None };
        return Ok((safetensors_dir, tok_path));
    }

    bail!(
        "Unknown model: '{}'\n\
         \n\
         Available models (use `tq list --available` or `tq pull <name>`):\n\
         {}\n\
         \n\
         Or pass a GGUF file path, model dir, or HF repo name (e.g. Qwen/Qwen2.5-7B).",
        query,
        catalog::list_available()
            .iter()
            .map(|e| format!("  {}:{:<6} {}", e.name, e.tag, e.display))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Download safetensors model from HuggingFace hub.
///
/// Downloads config.json, tokenizer.json, and all model*.safetensors files.
/// Returns the local directory containing all files.
pub fn resolve_hf_safetensors(repo_name: &str) -> Result<PathBuf> {
    let api = make_api()?;
    let repo = api.model(repo_name.to_string());

    // Download config.json (required)
    let config_path = repo.get("config.json")
        .with_context(|| format!("Cannot download config.json from {}", repo_name))?;
    let model_dir = config_path.parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid config.json path"))?
        .to_path_buf();

    eprintln!("  Config: {}", config_path.display());

    // Download tokenizer.json (optional)
    match repo.get("tokenizer.json") {
        Ok(p) => eprintln!("  Tokenizer: {}", p.display()),
        Err(_) => eprintln!("  Warning: no tokenizer.json found"),
    }

    // Find and download safetensors files
    // HF hub API: list repo files, filter *.safetensors
    let info = api.model(repo_name.to_string());

    // Try common single-file pattern first
    let single = repo.get("model.safetensors");
    if single.is_ok() {
        eprintln!("  Model: model.safetensors");
        return Ok(model_dir);
    }

    // Try sharded pattern: model-00001-of-NNNNN.safetensors
    let mut found = false;
    for i in 1..=100 {
        let name = format!("model-{:05}-of-", i);
        // We don't know the total, so just try to get the index file first
        let idx_file = format!("model.safetensors.index.json");
        if !found {
            if let Ok(idx_path) = repo.get(&idx_file) {
                eprintln!("  Index: {}", idx_path.display());
                // Read index to find all shard files
                if let Ok(idx_str) = std::fs::read_to_string(&idx_path) {
                    if let Ok(idx_json) = serde_json::from_str::<serde_json::Value>(&idx_str) {
                        if let Some(map) = idx_json.get("weight_map").and_then(|m| m.as_object()) {
                            let mut shard_files: Vec<String> = map.values()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect();
                            shard_files.sort();
                            shard_files.dedup();
                            eprintln!("  Downloading {} shard(s)...", shard_files.len());
                            for shard in &shard_files {
                                let pb = ProgressBar::new_spinner();
                                pb.set_style(
                                    ProgressStyle::default_spinner()
                                        .template("{spinner:.green} {msg}")
                                        .unwrap(),
                                );
                                pb.set_message(format!("  {}", shard));
                                pb.enable_steady_tick(std::time::Duration::from_millis(100));
                                repo.get(shard)
                                    .with_context(|| format!("Failed to download {}", shard))?;
                                pb.finish_with_message(format!("  {} ✓", shard));
                            }
                            found = true;
                        }
                    }
                }
            }
            break;
        }
    }

    if !found {
        bail!(
            "No safetensors files found in {}. \
             The repo may not contain safetensors format weights.",
            repo_name
        );
    }

    Ok(model_dir)
}
