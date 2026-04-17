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

#[cfg(feature = "cuda")]
pub mod forward;
#[cfg(feature = "cuda")]
pub use forward::DraftRuntime;

/// Sprint 2 Day 1: tree data structure + CPU reference attention. Pure
/// compute; always compiled (no cuda feature needed).
pub mod tree;
pub use tree::{TreeSpec, TreeError, build_ancestor_mask, tree_attention_cpu, MAX_TREE_NODES};

/// Config of an EAGLE draft model. Mirrors the subset of `config.json` fields
/// that actually influence forward-pass tensor layout, plus three
/// EAGLE-specific architecture flags discovered 2026-04-17 by inspecting the
/// yuhuili/EAGLE-Qwen2-7B-Instruct weight file:
///
/// 1. `has_fc_fusion`: a `fc` layer projects `[prev_hidden, embed(new_token)]`
///    (width `2 * hidden_size`) back down to `hidden_size` before attention.
///    Target models have no such layer.
/// 2. `has_pre_attention_norm`: draft has ONLY `post_attention_layernorm`,
///    no `input_layernorm`. The fc-projected input goes straight into attention.
/// 3. `num_key_value_heads == num_attention_heads`: draft uses full-rank K/V,
///    not GQA. Even when the target model uses GQA (e.g. Qwen2 7B 28/4),
///    the draft stays at 28/28. Draft KV cache is 7× per-token bigger but
///    draft runs ≤1 layer so total cache footprint stays small.
///
/// Numbers that MUST match the target model (for logits to drop into the
/// target's vocab): `hidden_size`, `intermediate_size`, `vocab_size`,
/// `rope_theta`, `rms_norm_eps`.
#[derive(Debug, Clone)]
pub struct EagleDraftConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,      // EAGLE-1: 1, EAGLE-3: varies
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,    // for EAGLE usually equals num_attention_heads
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub qkv_bias: bool,

    /// EAGLE has a `fc` layer that fuses target's previous-layer hidden state
    /// with the embedding of the most recently sampled token. When true, the
    /// draft builder must emit an extra dense + bias call point before attention.
    /// Input width: `2 * hidden_size`. Output width: `hidden_size`.
    pub has_fc_fusion: bool,

    /// True if the draft has an `input_layernorm` BEFORE attention (like a
    /// normal transformer block). False for EAGLE — the fc-projected input
    /// goes straight into attention, and only `post_attention_layernorm`
    /// applies before the MLP.
    pub has_pre_attention_norm: bool,

    /// Draft stores its weights in this precision. Loader promotes to FP16
    /// at upload time to match the rest of the engine (FP16 intermediate).
    pub native_dtype: DraftDType,
}

impl EagleDraftConfig {
    /// Width of the fc fusion layer's input vector. Only meaningful when
    /// `has_fc_fusion` is true.
    pub fn fc_in_dim(&self) -> usize {
        2 * self.hidden_size
    }

    /// KV cache per-layer bytes per token at FP16 (2 bytes per element).
    /// Useful for VRAM budgeting when attaching the draft to a running target.
    pub fn kv_bytes_per_token(&self) -> usize {
        // 2 (K and V) × hidden_size × 2 bytes
        2 * self.hidden_size * 2
    }
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
        num_key_value_heads:      28,        // no GQA in draft (target is 28/4)
        vocab_size:               152064,
        rope_theta:               1_000_000.0,
        rms_norm_eps:             1e-6,
        max_position_embeddings:  2048,
        qkv_bias:                 true,
        // EAGLE-specific flags (see struct doc for rationale)
        has_fc_fusion:            true,      // fc.weight [hidden, 2*hidden]
        has_pre_attention_norm:   false,     // only post_attention_layernorm present
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
    /// Scaffold + shape-validation metadata. Populated by [`EagleDraft::load`].
    pub model: Option<DraftModelStub>,
    /// GPU-resident weights, populated by [`EagleDraft::upload_to_gpu`].
    /// `None` before upload, CPU-only usage (tests, probe).
    #[cfg(feature = "cuda")]
    pub gpu: Option<DraftWeights>,
    /// FP32 runtime (scratch + KV cache + dequantised weight shadows),
    /// populated by [`EagleDraft::init_runtime`] the first time a forward is
    /// requested. Kept `Option` so CPU-only tests can still construct an
    /// `EagleDraft`.
    #[cfg(feature = "cuda")]
    pub runtime: Option<DraftRuntime>,
}

/// GPU-resident draft weights. All tensors promoted from BF16 (native) to FP16
/// at upload time — engine's intermediate precision is FP16 and keeping the
/// draft on the same dtype avoids per-kernel promote overhead. The one-shot
/// conversion cost is bounded (~1.65 GB of BF16 → ~825 MB of FP16 on GPU).
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DraftWeights {
    pub embed_tokens:             cudarc::driver::CudaSlice<half::f16>,  // [vocab, hidden]
    pub fc_weight:                cudarc::driver::CudaSlice<half::f16>,  // [hidden, 2*hidden]
    pub fc_bias:                  cudarc::driver::CudaSlice<half::f16>,  // [hidden]
    pub q_proj_weight:            cudarc::driver::CudaSlice<half::f16>,  // [hidden, hidden]
    pub q_proj_bias:              cudarc::driver::CudaSlice<half::f16>,  // [hidden]
    pub k_proj_weight:            cudarc::driver::CudaSlice<half::f16>,  // [hidden, hidden]
    pub k_proj_bias:              cudarc::driver::CudaSlice<half::f16>,  // [hidden]
    pub v_proj_weight:            cudarc::driver::CudaSlice<half::f16>,  // [hidden, hidden]
    pub v_proj_bias:              cudarc::driver::CudaSlice<half::f16>,  // [hidden]
    pub o_proj_weight:            cudarc::driver::CudaSlice<half::f16>,  // [hidden, hidden]
    pub mlp_gate_weight:          cudarc::driver::CudaSlice<half::f16>,  // [intermediate, hidden]
    pub mlp_up_weight:            cudarc::driver::CudaSlice<half::f16>,  // [intermediate, hidden]
    pub mlp_down_weight:          cudarc::driver::CudaSlice<half::f16>,  // [hidden, intermediate]
    pub post_attn_norm_weight:    cudarc::driver::CudaSlice<half::f16>,  // [hidden]

    /// Sum of all tensor bytes on GPU (post-FP16 promotion).
    pub total_bytes: u64,
}

#[cfg(feature = "cuda")]
impl DraftWeights {
    /// Summarised per-tensor allocation — used by `eagle-probe` UX only.
    pub fn allocation_report(&self, cfg: &EagleDraftConfig) -> Vec<(&'static str, usize)> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        vec![
            ("embed_tokens",         v * h * 2),
            ("fc.weight",            h * 2 * h * 2),
            ("fc.bias",              h * 2),
            ("q_proj.weight",        h * h * 2),
            ("q_proj.bias",          h * 2),
            ("k_proj.weight",        h * h * 2),
            ("k_proj.bias",          h * 2),
            ("v_proj.weight",        h * h * 2),
            ("v_proj.bias",          h * 2),
            ("o_proj.weight",        h * h * 2),
            ("gate_proj.weight",     i * h * 2),
            ("up_proj.weight",       i * h * 2),
            ("down_proj.weight",     h * i * 2),
            ("post_attn_norm",       h * 2),
        ]
    }
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
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            runtime: None,
        })
    }

    /// Upload the 14 validated draft tensors to GPU, promoting BF16 → FP16
    /// on the fly. Expensive (~1-3 seconds on 1.65 GB checkpoint) but one-shot.
    ///
    /// Precondition: `load` must have succeeded so `self.model` is populated.
    /// After this returns OK, `self.gpu` is `Some(DraftWeights)` with every
    /// tensor sitting in device memory ready for the Sprint 2 forward kernel.
    #[cfg(feature = "cuda")]
    pub fn upload_to_gpu(&mut self) -> Result<(), EagleError> {
        if self.model.is_none() {
            return Err(EagleError::Msg("upload_to_gpu called before load()".into()));
        }

        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| EagleError::Msg("no GPU registry available".into()))?;
        let stream = &reg.stream;

        // Re-mmap the file (the first load took ownership via a local scope).
        let file = std::fs::File::open(&self.weights_path)
            .map_err(|e| EagleError::Msg(format!("open({}): {}", self.weights_path.display(), e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| EagleError::Msg(format!("mmap: {}", e)))?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| EagleError::Msg(format!("safetensors parse: {:?}", e)))?;

        let up = |name: &str| -> Result<cudarc::driver::CudaSlice<half::f16>, EagleError> {
            upload_bf16_as_fp16(&st, name, stream)
        };

        let t_start = std::time::Instant::now();
        let weights = DraftWeights {
            embed_tokens:          up("embed_tokens.weight")?,
            fc_weight:             up("fc.weight")?,
            fc_bias:                up("fc.bias")?,
            q_proj_weight:         up("layers.0.self_attn.q_proj.weight")?,
            q_proj_bias:            up("layers.0.self_attn.q_proj.bias")?,
            k_proj_weight:         up("layers.0.self_attn.k_proj.weight")?,
            k_proj_bias:            up("layers.0.self_attn.k_proj.bias")?,
            v_proj_weight:         up("layers.0.self_attn.v_proj.weight")?,
            v_proj_bias:            up("layers.0.self_attn.v_proj.bias")?,
            o_proj_weight:         up("layers.0.self_attn.o_proj.weight")?,
            mlp_gate_weight:       up("layers.0.mlp.gate_proj.weight")?,
            mlp_up_weight:         up("layers.0.mlp.up_proj.weight")?,
            mlp_down_weight:       up("layers.0.mlp.down_proj.weight")?,
            post_attn_norm_weight: up("layers.0.post_attention_layernorm.weight")?,
            total_bytes: 0, // set below
        };
        let _ = stream.synchronize();
        let elapsed = t_start.elapsed();

        // Compute allocation summary.
        let total_bytes: usize = weights.allocation_report(&self.config).iter()
            .map(|(_, b)| *b).sum();

        eprintln!(
            "[eagle] uploaded 14 tensors to GPU, {:.2} GB FP16 in {:.1}s",
            total_bytes as f64 / 1e9,
            elapsed.as_secs_f32(),
        );
        self.gpu = Some(DraftWeights { total_bytes: total_bytes as u64, ..weights });
        Ok(())
    }

    /// Report whether the draft is ready for speculation. Returns true only
    /// after `load()` + `upload_to_gpu()` have both succeeded AND the FP32
    /// runtime has been initialised.
    ///
    /// Sprint 1 Day 4: `runtime` populated = "can run a single-token forward".
    /// Sprint 3 flips this true only after tree + verify land too.
    pub fn is_runnable(&self) -> bool {
        #[cfg(feature = "cuda")]
        { self.runtime.is_some() }
        #[cfg(not(feature = "cuda"))]
        { false }
    }

    /// Sprint 1 Day 4: build the FP32 runtime (dequantise FP16 weights, build
    /// RoPE tables, allocate scratch + KV cache). Idempotent — second call is
    /// a no-op. Requires prior `upload_to_gpu()`.
    #[cfg(feature = "cuda")]
    pub fn init_runtime(&mut self) -> Result<(), EagleError> {
        if self.runtime.is_some() { return Ok(()); }
        let gpu = self.gpu.as_ref().ok_or_else(||
            EagleError::Msg("init_runtime called before upload_to_gpu()".into()))?;
        let rt = DraftRuntime::from_weights(gpu, &self.config)?;
        eprintln!(
            "[eagle] FP32 runtime ready: ~{:.2} GB weight shadows + KV cache + scratch",
            rt.total_bytes() as f64 / 1e9,
        );
        self.runtime = Some(rt);
        Ok(())
    }

    /// Single-step draft forward pass. See [`DraftRuntime::forward`] for the
    /// contract. Call `init_runtime()` first.
    #[cfg(feature = "cuda")]
    pub fn draft_forward(
        &mut self,
        prev_hidden: &[f32],
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, EagleError> {
        let rt = self.runtime.as_mut().ok_or_else(||
            EagleError::Msg("draft_forward called before init_runtime()".into()))?;
        rt.forward(prev_hidden, token_id, position)
    }
}

/// Read a BF16 tensor from the safetensors view, convert every element to
/// FP16 via an f32 round-trip (bf16 and f16 have different mantissa/exponent
/// splits, so truncation alone doesn't work), and upload to GPU.
///
/// Worst-case cost: O(numel) CPU conversion, then one H2D copy. For a 1.65 GB
/// draft this is ~1 B elements which takes under 2 seconds on modern CPUs —
/// acceptable as a one-shot load overhead.
#[cfg(feature = "cuda")]
fn upload_bf16_as_fp16(
    st: &safetensors::SafeTensors<'_>,
    name: &str,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<cudarc::driver::CudaSlice<half::f16>, EagleError> {
    let view = st.tensor(name)
        .map_err(|e| EagleError::Msg(format!("tensor({}): {:?}", name, e)))?;
    if view.dtype() != safetensors::Dtype::BF16 {
        return Err(EagleError::Msg(format!(
            "tensor '{}' dtype {:?}, expected BF16", name, view.dtype()
        )));
    }
    let raw = view.data();
    let numel = raw.len() / 2;

    // BF16 → F32 → F16. Doing it in one pass via iter to let the compiler
    // vectorize as much as possible.
    let mut buf: Vec<half::f16> = Vec::with_capacity(numel);
    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        let f32v = half::bf16::from_bits(bits).to_f32();
        buf.push(half::f16::from_f32(f32v));
    }

    let slice = stream.memcpy_stod(&buf)
        .map_err(|e| EagleError::Msg(format!("H2D upload '{}': {}", name, e)))?;
    Ok(slice)
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
    fn draft_not_runnable_without_runtime() {
        // Scaffold is explicit about its incomplete state: is_runnable is
        // false until init_runtime() populates the FP32 runtime. Day 4 wires
        // the runtime but the stub here represents the pre-init state.
        let cfg = eagle_qwen2_7b_config();
        let stub = EagleDraft {
            config: cfg,
            weights_path: PathBuf::from("/dev/null"),
            model: None,
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            runtime: None,
        };
        assert!(!stub.is_runnable());
    }

    #[test]
    fn config_reflects_eagle_specific_architecture() {
        let cfg = eagle_qwen2_7b_config();
        // Three flags that diverge from a standard Qwen2 block:
        assert!(cfg.has_fc_fusion, "fc layer fuses prev_hidden + embed(token)");
        assert!(!cfg.has_pre_attention_norm, "only post_attention_layernorm present");
        assert_eq!(cfg.num_key_value_heads, cfg.num_attention_heads,
                   "draft uses full-rank KV (no GQA)");
        // fc input width is always 2*hidden
        assert_eq!(cfg.fc_in_dim(), 7168);
        // KV cache budget: 2 * hidden * 2 bytes = 14,336 B per token per layer
        assert_eq!(cfg.kv_bytes_per_token(), 14336);
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
