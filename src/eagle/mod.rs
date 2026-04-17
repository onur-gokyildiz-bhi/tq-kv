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
    /// Inventory of tensor (name, shape, bf16_bytes) actually present in the
    /// safetensors file. Populated once tensors parse successfully.
    pub tensors: Vec<TensorMeta>,
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name:  String,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub dtype: String,
}

/// Expected tensor shape spec for the EAGLE-1 Qwen2-7B draft. Derived from
/// running `scripts/convert_eagle_weights.py` once and reading the output
/// inventory. Used by `EagleDraft::validate_shapes` to catch mismatched
/// checkpoints at load time rather than at first forward pass.
///
/// Notable architectural quirks (why this list isn't a generic Qwen2 spec):
/// - Only ONE RMSNorm per layer (`post_attention_layernorm`) — pre-attention
///   norm is absent because input is already fc-projected from fused features.
/// - `fc.weight` shape (hidden, 2*hidden): the EAGLE fusion layer, takes
///   concat(prev_layer_hidden, embed(token)) and projects back to hidden.
/// - K/V are full-rank (hidden, hidden), NOT GQA-reduced. Target Qwen2 7B uses
///   28/4 GQA; draft uses 28/28.
/// - No lm_head weight — at decode time the draft clones the target's.
pub fn eagle_qwen2_7b_expected_tensors() -> Vec<(&'static str, Vec<usize>)> {
    let hidden = 3584_usize;
    let intermediate = 18944_usize;
    let vocab = 152064_usize;
    vec![
        ("layers.0.self_attn.q_proj.weight",             vec![hidden, hidden]),
        ("layers.0.self_attn.q_proj.bias",               vec![hidden]),
        ("layers.0.self_attn.k_proj.weight",             vec![hidden, hidden]),
        ("layers.0.self_attn.k_proj.bias",               vec![hidden]),
        ("layers.0.self_attn.v_proj.weight",             vec![hidden, hidden]),
        ("layers.0.self_attn.v_proj.bias",               vec![hidden]),
        ("layers.0.self_attn.o_proj.weight",             vec![hidden, hidden]),
        ("layers.0.mlp.gate_proj.weight",                vec![intermediate, hidden]),
        ("layers.0.mlp.up_proj.weight",                  vec![intermediate, hidden]),
        ("layers.0.mlp.down_proj.weight",                vec![hidden, intermediate]),
        ("layers.0.post_attention_layernorm.weight",     vec![hidden]),
        ("embed_tokens.weight",                          vec![vocab, hidden]),
        ("fc.weight",                                    vec![hidden, 2 * hidden]),
        ("fc.bias",                                      vec![hidden]),
    ]
}

impl EagleDraft {
    /// Parse the safetensors file and validate tensor inventory against the
    /// expected EAGLE draft layout.
    ///
    /// This runs actual tensor metadata inspection (safetensors header parse,
    /// zero tensor data read — so it's fast even for multi-GB files). The
    /// result is compared against `eagle_qwen2_7b_expected_tensors` so shape
    /// mismatches are surfaced immediately instead of triggering obscure
    /// kernel-arg errors later during forward.
    ///
    /// Sprint 1 Day 2: validation only, no GPU upload / kernel dispatch yet.
    /// Sprint 2 will extend this into a full `GenericTurboModel` instance.
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
            "[eagle] loading {} ({:.2} GB), native {:?}, {} layers",
            canonical.display(),
            size as f64 / 1e9,
            config.native_dtype,
            config.num_hidden_layers,
        );

        // Memory-map the safetensors file and parse its header. SafeTensors
        // puts a JSON header at the start listing every tensor's dtype, shape,
        // and byte offsets; reading the header is a few KB of IO regardless of
        // file size.
        let file = std::fs::File::open(&canonical)
            .map_err(|e| EagleError::Msg(format!("open({}): {}", canonical.display(), e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| EagleError::Msg(format!("mmap({}): {}", canonical.display(), e)))?;

        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| EagleError::Msg(format!("safetensors parse: {:?}", e)))?;

        let mut tensors: Vec<TensorMeta> = Vec::new();
        for name in st.names() {
            let view = st.tensor(name)
                .map_err(|e| EagleError::Msg(format!("tensor({}): {:?}", name, e)))?;
            tensors.push(TensorMeta {
                name:  name.to_string(),
                shape: view.shape().to_vec(),
                bytes: view.data().len(),
                dtype: format!("{:?}", view.dtype()),
            });
        }
        tensors.sort_by(|a, b| a.name.cmp(&b.name));

        // Validate against the EAGLE-Qwen2-7B expected inventory. Missing
        // tensors are hard errors; unexpected extras are warnings (upstream
        // may ship auxiliary keys we don't need).
        let mut errors: Vec<String> = Vec::new();
        let expected = eagle_qwen2_7b_expected_tensors();
        for (name, shape) in &expected {
            match tensors.iter().find(|t| t.name == *name) {
                None => errors.push(format!("missing tensor '{}'", name)),
                Some(t) => {
                    if t.shape != *shape {
                        errors.push(format!(
                            "tensor '{}' shape mismatch: expected {:?}, got {:?}",
                            name, shape, t.shape
                        ));
                    }
                }
            }
        }
        if !errors.is_empty() {
            return Err(EagleError::Msg(format!(
                "EAGLE weight validation failed ({} errors):\n  {}",
                errors.len(),
                errors.join("\n  "),
            )));
        }

        // Warn (don't fail) on extras — future EAGLE revisions might add fields.
        let expected_names: std::collections::HashSet<&str> =
            expected.iter().map(|(n, _)| *n).collect();
        let extras: Vec<&str> = tensors.iter()
            .map(|t| t.name.as_str())
            .filter(|n| !expected_names.contains(n))
            .collect();
        if !extras.is_empty() {
            eprintln!(
                "[eagle] note: {} unexpected tensor(s) in checkpoint (safe to ignore): {}",
                extras.len(),
                extras.join(", ")
            );
        }

        eprintln!(
            "[eagle] validated {}/{} expected tensors, total {:.2} GB",
            expected.len(),
            tensors.len(),
            size as f64 / 1e9,
        );

        Ok(Self {
            config,
            weights_path: canonical,
            model: Some(DraftModelStub {
                loaded: true,
                tensor_count: tensors.len(),
                total_bytes: size,
                tensors,
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

    #[test]
    fn expected_tensors_match_hf_inventory() {
        // Ordered sanity check — the 14 tensor names + shapes derived from
        // running `scripts/convert_eagle_weights.py` against
        // yuhuili/EAGLE-Qwen2-7B-Instruct on 2026-04-17.
        let t = eagle_qwen2_7b_expected_tensors();
        assert_eq!(t.len(), 14, "EAGLE-1 Qwen2 7B draft has exactly 14 tensors");

        // fc layer: the EAGLE fusion projection, hidden ← 2*hidden
        let fc_w = t.iter().find(|(n, _)| *n == "fc.weight").unwrap();
        assert_eq!(fc_w.1, vec![3584, 7168], "fc.weight projects 2*hidden to hidden");

        // K/V are full rank (no GQA in draft, unlike target Qwen2 7B which uses 28/4 GQA).
        let k_w = t.iter().find(|(n, _)| *n == "layers.0.self_attn.k_proj.weight").unwrap();
        assert_eq!(k_w.1, vec![3584, 3584], "draft K is NOT GQA-reduced");

        // Only ONE norm: post_attention_layernorm. No input_layernorm because
        // input is already fc-projected from fused features.
        let norm_count = t.iter().filter(|(n, _)| n.contains("layernorm")).count();
        assert_eq!(norm_count, 1, "EAGLE draft has only post_attention_layernorm");
    }

    #[test]
    fn expected_tensor_names_are_unique() {
        let t = eagle_qwen2_7b_expected_tensors();
        let names: std::collections::HashSet<&str> = t.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), t.len(), "no duplicate tensor names in spec");
    }
}
