use anyhow::Result;
use std::path::{Path, PathBuf};
use tq_kv::TurboQuantConfig;

use crate::config::{self, ModelConfig};
use crate::download;
use crate::engine::Engine;

/// Resolve model + tokenizer paths and create Engine.
pub fn load_engine(
    model_config: &ModelConfig,
    model_path_override: Option<&Path>,
    tq_config: Option<TurboQuantConfig>,
    tokenizer_repo_override: Option<&str>,
    force_cpu: bool,
) -> Result<Engine> {
    let models_dir = get_models_dir();
    std::fs::create_dir_all(&models_dir)?;

    let model_path = match model_path_override {
        Some(path) => path.to_path_buf(),
        None => download::ensure_model(model_config, &models_dir)?,
    };

    let tokenizer_path = match tokenizer_repo_override {
        Some(repo) => {
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| anyhow::anyhow!("HF API init failed: {}", e))?;
            let hf_repo = api.model(repo.to_string());
            hf_repo.get("tokenizer.json")
                .map_err(|e| anyhow::anyhow!("Tokenizer download from {} failed: {}", repo, e))?
        }
        None => download::ensure_tokenizer(model_config, &models_dir)?,
    };
    let arch = config::detect_arch(&model_path.to_string_lossy());

    Engine::load_with_device(&model_path, &tokenizer_path, arch, tq_config, force_cpu)
}

fn get_models_dir() -> PathBuf {
    std::env::current_exe()
        .unwrap_or_else(|_| PathBuf::from("."))
        .parent()
        .unwrap_or(Path::new("."))
        .join("models")
}
