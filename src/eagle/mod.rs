//! EAGLE speculative decoding integration (draft + tree verify).
//!
//! # Status: Sprint 1 Day 1 — scaffolding only
//!
//! This module lays the structural foundation for EAGLE-3 integration but does
//! NOT yet run speculation. Sprint 1 Day 1 deliverables:
//! - Draft model config type that mirrors `config.json` of EAGLE checkpoints
//! - Loader that reads a safetensors directory containing the draft weights
//!   (expects a one-time offline conversion from `pytorch_model.bin` via
//!   `scripts/convert_eagle_weights.py`)
//! - CLI plumbing (`--eagle-draft-model PATH`) that loads the draft alongside
//!   the target model but does nothing with it yet
//!
//! # Roadmap
//! - Sprint 1: draft load + standalone forward pass
//! - Sprint 2: tree_attention kernel (or extend flash_decode with general mask)
//! - Sprint 3: verify/rollback loop in engine.rs
//! - Sprint 4: PPL + quality validation
//!
//! Reference: `memory/project_eagle3_integration_plan.md`.

use std::path::{Path, PathBuf};

/// Config of an EAGLE draft model. Mirrors the subset of `config.json` fields
/// that actually influence forward-pass tensor layout. Numbers MUST match the
/// target model's corresponding field (hidden_size, intermediate_size,
/// vocab_size, rope_theta) — if they diverge, the draft's predicted logits
/// cannot be re-projected into the target vocabulary.
#[derive(Debug, Clone)]
pub struct EagleDraftConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,      // EAGLE-1: 1, EAGLE-3: varies
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,    // often equals num_attention_heads (no GQA in draft)
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub qkv_bias: bool,
    /// Draft stores its weights in this precision. Loader may promote to FP16
    /// at load time to match the rest of the engine.
    pub native_dtype: DraftDType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftDType {
    BFloat16,
    Float16,
    Float32,
}

/// The concrete shape of EAGLE-Qwen2-7B-Instruct — matches
/// https://huggingface.co/yuhuili/EAGLE-Qwen2-7B-Instruct/config.json
///
/// When the target model is Qwen2 7B (hidden=3584, vocab=152064) these values
/// align with the target exactly, so the draft's logits drop into the target's
/// vocabulary without projection.
pub fn eagle_qwen2_7b_config() -> EagleDraftConfig {
    EagleDraftConfig {
        hidden_size:              3584,
        intermediate_size:        18944,
        num_hidden_layers:        1,
        num_attention_heads:      28,
        num_key_value_heads:      28,
        vocab_size:               152064,
        rope_theta:               1_000_000.0,
        rms_norm_eps:             1e-6,
        max_position_embeddings:  2048,
        qkv_bias:                 true,
        native_dtype:             DraftDType::BFloat16,
    }
}

/// Handle to a loaded EAGLE draft model. The inner model is not yet runnable
/// end-to-end (tree decode + verify unimplemented) — Sprint 1 just proves the
/// weights load and shapes match expectations.
#[derive(Debug)]
pub struct EagleDraft {
    pub config: EagleDraftConfig,
    pub weights_path: PathBuf,
    /// Will hold the loaded inner model once Sprint 1 finishes. For now None.
    pub model: Option<DraftModelStub>,
}

/// Stub for the draft inner model. Sprint 1 replaces this with a real
/// `GenericTurboModel` instance configured for 1 transformer layer.
#[derive(Debug)]
pub struct DraftModelStub {
    pub loaded: bool,
    pub tensor_count: usize,
    pub total_bytes: u64,
}

impl EagleDraft {
    /// Minimal scaffolding load — opens the weights file (safetensors or a
    /// directory containing `model.safetensors`) and validates that the file
    /// exists and has non-zero size. Actual tensor parsing lives in Sprint 1
    /// when we wire through `GenericTurboModel::build_from_source`.
    pub fn load(weights_path: &Path, config: EagleDraftConfig) -> Result<Self, EagleError> {
        // Accept either a directory (containing model.safetensors) or a direct
        // .safetensors / .bin path — same convention as target model loading.
        let canonical = if weights_path.is_dir() {
            weights_path.join("model.safetensors")
        } else {
            weights_path.to_path_buf()
        };

        if !canonical.exists() {
            return Err(EagleError::Msg(format!(
                "EAGLE weights not found: {}\n\
                 Convert the upstream pytorch_model.bin first with:\n  \
                 python scripts/convert_eagle_weights.py <path/to/pytorch_model.bin>",
                canonical.display()
            )));
        }

        let size = std::fs::metadata(&canonical)
            .map_err(|e| EagleError::Msg(format!("stat({}): {}", canonical.display(), e)))?
            .len();

        eprintln!(
            "[eagle] scaffold load: {} ({:.2} GB), native {:?}, {} layers",
            canonical.display(),
            size as f64 / 1e9,
            config.native_dtype,
            config.num_hidden_layers,
        );

        Ok(Self {
            config,
            weights_path: canonical,
            // Stub: Sprint 1 replaces with real tensor load.
            model: Some(DraftModelStub {
                loaded: true,
                tensor_count: 0,
                total_bytes: size,
            }),
        })
    }

    /// Report whether the draft is ready for speculation. Returns false during
    /// Sprint 1 Day 1 (scaffold-only).
    pub fn is_runnable(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub enum EagleError {
    Msg(String),
}

impl std::fmt::Display for EagleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EagleError::Msg(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for EagleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eagle_qwen2_7b_config_matches_hf() {
        let c = eagle_qwen2_7b_config();
        assert_eq!(c.hidden_size, 3584);
        assert_eq!(c.num_hidden_layers, 1);
        assert_eq!(c.vocab_size, 152064);
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert!(c.qkv_bias);
        assert_eq!(c.native_dtype, DraftDType::BFloat16);
    }

    #[test]
    fn load_missing_weights_is_err() {
        let cfg = eagle_qwen2_7b_config();
        let result = EagleDraft::load(Path::new("/nonexistent/model.safetensors"), cfg);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"));
        assert!(err.contains("convert_eagle_weights.py"));
    }

    #[test]
    fn draft_not_runnable_in_sprint1_day1() {
        // Scaffold is explicit about its incomplete state.
        let cfg = eagle_qwen2_7b_config();
        // Don't call load() (no real path), just verify API contract.
        let stub = EagleDraft {
            config: cfg,
            weights_path: PathBuf::from("/dev/null"),
            model: None,
        };
        assert!(!stub.is_runnable());
    }
}
