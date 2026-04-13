//! Safetensors weight source.
//!
//! Implements [`crate::gguf::WeightSource`] for HuggingFace safetensors directories
//! (single-file `model.safetensors` or sharded via `model.safetensors.index.json`).
//!
//! Responsibilities:
//!   1. Parse `config.json` and synthesise GGUF-style metadata keys
//!      (`general.architecture`, `<arch>.block_count`, etc.) so that the
//!      format-agnostic `GenericTurboModel::build` code path can consume it.
//!   2. Parse each shard's safetensors header once and record
//!      `(shard_path, data_offsets, dtype, shape)` per tensor.
//!   3. On `tensor()`: seek into the correct shard, read only that tensor's
//!      bytes, and return a `QWeight` with `raw_data: Vec<u8>`. No quantisation
//!      — safetensors carries full-precision (F16/BF16/F32) weights.
//!
//! Phase 1 keeps everything RAM-simple: one open+seek+read per tensor, no
//! mmap. Phase 2 switches to mmap-backed `RawBytes::Mmap` ranges so we don't
//! copy tensor bytes out of the file at all.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::cuda::{Result, TqDevice, TqError};
use crate::gguf::{GgmlDType, GgufValue, WeightSource};
use crate::qmatmul::QWeight;

/// Per-tensor header entry — where the raw bytes live on disk.
#[derive(Debug, Clone)]
struct TensorLoc {
    shard: PathBuf,
    /// Absolute byte offset (into the shard file) where this tensor's raw bytes start.
    abs_start: u64,
    /// Number of bytes.
    nbytes: u64,
    dtype: GgmlDType,
    /// Logical shape (first two dims; safetensors weights are 1-D bias or 2-D matrix).
    shape: (usize, usize),
}

pub struct SafetensorsContent {
    /// Synthesised GGUF-style metadata. Populated from `config.json`.
    metadata: HashMap<String, GgufValue>,
    /// GGUF-name → shard location. GGUF is the lingua franca inside the model
    /// builder, so we key by GGUF names even though the underlying file uses HF names.
    tensors: HashMap<String, TensorLoc>,
}

impl SafetensorsContent {
    /// Open a model directory containing `config.json` + safetensors files.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|e| {
            TqError::Msg(format!("cannot read {}: {}", config_path.display(), e))
        })?;
        let config: Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| TqError::Msg(format!("config.json parse error: {}", e)))?;

        let arch = detect_arch(&config)?;
        if arch == "gemma4" || arch == "gemma3" {
            reject_unsupported_gemma4_features(&config)?;
        }
        let metadata = synthesise_metadata(&arch, &config)?;

        // Determine shard files:
        //   - single:   model.safetensors
        //   - sharded:  model.safetensors.index.json → weight_map: {tensor: shard_file}
        let mut tensors = HashMap::new();
        let single = model_dir.join("model.safetensors");
        let index = model_dir.join("model.safetensors.index.json");
        if single.exists() {
            index_shard(&single, &mut tensors)?;
        } else if index.exists() {
            let idx_bytes = std::fs::read(&index).map_err(|e| {
                TqError::Msg(format!("cannot read {}: {}", index.display(), e))
            })?;
            let idx: Value = serde_json::from_slice(&idx_bytes)
                .map_err(|e| TqError::Msg(format!("index.json parse error: {}", e)))?;
            let map = idx
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| TqError::Msg("safetensors index missing weight_map".into()))?;
            let mut shard_files: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();
            for sf in shard_files {
                index_shard(&model_dir.join(&sf), &mut tensors)?;
            }
        } else {
            return Err(TqError::Msg(format!(
                "no model.safetensors or model.safetensors.index.json in {}",
                model_dir.display()
            )));
        }

        eprintln!(
            "Safetensors [{}]: {} tensors across shard(s)",
            arch,
            tensors.len()
        );
        Ok(Self { metadata, tensors })
    }
}

impl WeightSource for SafetensorsContent {
    fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    fn tensor(&self, name: &str, _device: &TqDevice) -> Result<QWeight> {
        let loc = self
            .tensors
            .get(name)
            .ok_or_else(|| TqError::Msg(format!("tensor not found: {}", name)))?;
        let mut f = File::open(&loc.shard).map_err(|e| {
            TqError::Msg(format!("open {}: {}", loc.shard.display(), e))
        })?;
        f.seek(SeekFrom::Start(loc.abs_start)).map_err(|e| {
            TqError::Msg(format!("seek {}: {}", loc.shard.display(), e))
        })?;
        let mut buf = vec![0u8; loc.nbytes as usize];
        f.read_exact(&mut buf).map_err(|e| {
            TqError::Msg(format!("read {}: {}", loc.shard.display(), e))
        })?;
        Ok(QWeight::new(buf, loc.dtype, loc.shape))
    }
}

// ────────────────────────────────────────────────────────────
// Metadata synthesis: config.json → GGUF-style metadata keys
// ────────────────────────────────────────────────────────────

fn detect_arch(config: &Value) -> Result<String> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TqError::Msg("config.json missing model_type".into()))?;
    // Map HF model_type → GGUF architecture name. For unknowns, pass through.
    let arch = match model_type {
        "qwen2" | "qwen2_moe" => "qwen2",
        "qwen3" | "qwen3_moe" => "qwen3",
        "llama" => "llama",
        "mistral" => "llama", // mistral uses llama arch family in GGUF
        "gemma" => "gemma",
        "gemma2" => "gemma2",
        // Gemma 4 lineage — same family dispatch (arch.contains("gemma") in
        // the builder catches Gemma-specific handling: with_add_unit norms,
        // embed_scale=sqrt(hidden), softcap, etc.). Layer-level sliding/full
        // alternation and partial RoPE are Sprint 1 follow-ups (text-only
        // path). Vision tower, sparse MoE, multimodal token routing are
        // Sprints 3-4.
        "gemma3" => "gemma3",
        "gemma4" => "gemma4",
        "phi3" => "phi3",
        other => other,
    };
    Ok(arch.to_string())
}

/// Early-out for features Sprint 1 doesn't support yet. Returns a clean error
/// instead of a late crash deep in the forward path.
fn reject_unsupported_gemma4_features(config: &Value) -> Result<()> {
    if config.get("vision_config").is_some() {
        return Err(TqError::Msg(
            "Gemma 4 vision_config detected — multimodal support is Sprint 3-4. \
             For Sprint 1 use a text-only variant or strip vision weights. \
             Memory: project_gemma4_backlog.md".into()
        ));
    }
    if config.get("moe_config").is_some() {
        return Err(TqError::Msg(
            "Gemma 4 moe_config detected (128 experts × top-8) — sparse MoE \
             support is Sprint 2. Current MoE path is Mixtral-style dense \
             loop and won't scale to 128 experts. Memory: \
             project_gemma4_backlog.md".into()
        ));
    }
    // Partial RoPE: Gemma 4 uses rope_type="proportional" with
    // partial_rotary_factor=0.25 on the full_attention regime. We don't have a
    // partial-RoPE kernel yet — warn loudly rather than silently apply full
    // RoPE and hope for the best.
    if let Some(rp) = config.get("rope_parameters") {
        if let Some(full_attn) = rp.get("full_attention") {
            if let Some(ptype) = full_attn.get("rope_type").and_then(|v| v.as_str()) {
                if ptype != "default" {
                    eprintln!(
                        "[gemma4] WARNING: full_attention rope_type='{}' — \
                         partial RoPE kernel not yet implemented (Sprint 1 \
                         TODO). Loading will proceed with default RoPE; \
                         expect incorrect attention for the global layers.",
                        ptype
                    );
                }
            }
        }
    }
    Ok(())
}

fn synthesise_metadata(arch: &str, c: &Value) -> Result<HashMap<String, GgufValue>> {
    let mut m = HashMap::new();
    m.insert(
        "general.architecture".into(),
        GgufValue::String(arch.to_string()),
    );

    fn u32_of(v: &Value, key: &str) -> Option<u32> {
        v.get(key).and_then(|x| x.as_u64()).map(|x| x as u32)
    }
    fn f32_of(v: &Value, key: &str) -> Option<f32> {
        v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32)
    }

    // Required
    let block_count = u32_of(c, "num_hidden_layers").ok_or_else(|| {
        TqError::Msg("config.json missing num_hidden_layers".into())
    })?;
    let emb = u32_of(c, "hidden_size").ok_or_else(|| {
        TqError::Msg("config.json missing hidden_size".into())
    })?;
    let n_head = u32_of(c, "num_attention_heads").ok_or_else(|| {
        TqError::Msg("config.json missing num_attention_heads".into())
    })?;
    // GQA: optional, defaults to n_head
    let n_kv_head = u32_of(c, "num_key_value_heads").unwrap_or(n_head);
    // eps: HF names vary
    let eps = f32_of(c, "rms_norm_eps")
        .or_else(|| f32_of(c, "layer_norm_epsilon"))
        .or_else(|| f32_of(c, "layer_norm_eps"))
        .unwrap_or(1e-5);

    m.insert(format!("{arch}.block_count"), GgufValue::U32(block_count));
    m.insert(format!("{arch}.embedding_length"), GgufValue::U32(emb));
    m.insert(format!("{arch}.attention.head_count"), GgufValue::U32(n_head));
    m.insert(
        format!("{arch}.attention.head_count_kv"),
        GgufValue::U32(n_kv_head),
    );
    m.insert(
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        GgufValue::F32(eps),
    );

    // Optional: rope theta, context length, head_dim, moe counts
    if let Some(theta) = f32_of(c, "rope_theta") {
        m.insert(format!("{arch}.rope.freq_base"), GgufValue::F32(theta));
    }
    // Gemma 4 splits rope across attention regimes: full_attention (theta=1e6,
    // proportional, partial=0.25) vs sliding_attention (theta=1e4, default).
    // Synthesise both bases so the builder can pick per-layer once the
    // alternation scheduler lands. For Sprint 1 we emit only the base values;
    // partial_rotary_factor and sliding alternation are TODO.
    if let Some(rp) = c.get("rope_parameters") {
        if let Some(f) = rp.get("full_attention") {
            if let Some(t) = f.get("rope_theta").and_then(|v| v.as_f64()) {
                m.insert(
                    format!("{arch}.rope.freq_base_full"),
                    GgufValue::F32(t as f32),
                );
            }
            if let Some(pf) = f.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                m.insert(
                    format!("{arch}.rope.partial_factor_full"),
                    GgufValue::F32(pf as f32),
                );
            }
        }
        if let Some(f) = rp.get("sliding_attention") {
            if let Some(t) = f.get("rope_theta").and_then(|v| v.as_f64()) {
                m.insert(
                    format!("{arch}.rope.freq_base_sliding"),
                    GgufValue::F32(t as f32),
                );
            }
        }
    }
    if let Some(sw) = u32_of(c, "sliding_window") {
        m.insert(
            format!("{arch}.attention.sliding_window"),
            GgufValue::U32(sw),
        );
    }
    if let Some(ctx) = u32_of(c, "max_position_embeddings") {
        m.insert(format!("{arch}.context_length"), GgufValue::U32(ctx));
    }
    // head_dim: HF sometimes includes it, sometimes implicit
    if let Some(hd) = u32_of(c, "head_dim") {
        m.insert(format!("{arch}.attention.key_length"), GgufValue::U32(hd));
    }
    // MoE: num_local_experts + num_experts_per_tok (Mixtral/Qwen-MoE style)
    if let Some(ne) = u32_of(c, "num_local_experts") {
        m.insert(format!("{arch}.expert_count"), GgufValue::U32(ne));
    }
    if let Some(neu) = u32_of(c, "num_experts_per_tok") {
        m.insert(format!("{arch}.expert_used_count"), GgufValue::U32(neu));
    }
    // Gemma2 soft-capping (read directly by from_gguf)
    if let Some(sc) = f32_of(c, "attn_logit_softcapping") {
        m.insert(format!("{arch}.attn_logit_softcapping"), GgufValue::F32(sc));
    }
    if let Some(sc) = f32_of(c, "final_logit_softcapping") {
        m.insert(format!("{arch}.final_logit_softcapping"), GgufValue::F32(sc));
    }

    // Tie word embeddings: we don't put it in metadata; it's implicit — if
    // lm_head.weight is absent, the builder's fallback handles it.
    Ok(m)
}

// ────────────────────────────────────────────────────────────
// Safetensors header parsing
// ────────────────────────────────────────────────────────────

/// Parse one shard's safetensors header and add every tensor to `out` keyed by
/// its GGUF-canonical name. The header format is:
///   [u64 LE: header_size][header_size bytes of JSON][tensor data...]
fn index_shard(
    path: &Path,
    out: &mut HashMap<String, TensorLoc>,
) -> Result<()> {
    let mut f = File::open(path).map_err(|e| {
        TqError::Msg(format!("open {}: {}", path.display(), e))
    })?;

    // Header size (u64 LE)
    let mut hs_buf = [0u8; 8];
    f.read_exact(&mut hs_buf).map_err(|e| {
        TqError::Msg(format!("read header size {}: {}", path.display(), e))
    })?;
    let header_size = u64::from_le_bytes(hs_buf);
    if header_size > 100 * 1024 * 1024 {
        return Err(TqError::Msg(format!(
            "implausible safetensors header size {} in {}",
            header_size,
            path.display()
        )));
    }

    // Header JSON
    let mut header_bytes = vec![0u8; header_size as usize];
    f.read_exact(&mut header_bytes).map_err(|e| {
        TqError::Msg(format!("read header {}: {}", path.display(), e))
    })?;
    let header: Value = serde_json::from_slice(&header_bytes).map_err(|e| {
        TqError::Msg(format!("header json {}: {}", path.display(), e))
    })?;
    let header_obj = header
        .as_object()
        .ok_or_else(|| TqError::Msg(format!("header not object in {}", path.display())))?;

    // Data region starts right after the header.
    let data_base: u64 = 8 + header_size;

    for (hf_name, info) in header_obj {
        if hf_name == "__metadata__" {
            continue;
        }
        let dtype_str = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TqError::Msg(format!("tensor {} missing dtype", hf_name)))?;
        let dtype = match dtype_str {
            "F16" => GgmlDType::F16,
            "BF16" => GgmlDType::BF16,
            "F32" => GgmlDType::F32,
            other => {
                return Err(TqError::Msg(format!(
                    "unsupported safetensors dtype {} for tensor {}",
                    other, hf_name
                )))
            }
        };

        let shape_arr = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TqError::Msg(format!("tensor {} missing shape", hf_name)))?;
        let shape: Vec<usize> = shape_arr
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let logical_shape = match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            _ => (shape[0], shape[1..].iter().product()),
        };

        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TqError::Msg(format!("tensor {} missing data_offsets", hf_name)))?;
        let start = offsets.get(0).and_then(|v| v.as_u64()).ok_or_else(|| {
            TqError::Msg(format!("tensor {} bad data_offsets", hf_name))
        })?;
        let end = offsets.get(1).and_then(|v| v.as_u64()).ok_or_else(|| {
            TqError::Msg(format!("tensor {} bad data_offsets", hf_name))
        })?;

        let gguf_name = match hf_to_gguf_name(hf_name) {
            Some(n) => n,
            None => continue, // silently skip unknown tensors (e.g. optional RoPE cache)
        };

        out.insert(
            gguf_name,
            TensorLoc {
                shard: path.to_path_buf(),
                abs_start: data_base + start,
                nbytes: end - start,
                dtype,
                shape: logical_shape,
            },
        );
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────
// HF → GGUF tensor name remap
// ────────────────────────────────────────────────────────────
//
// Canonical HF layout for decoder-only transformers (llama/qwen/mistral/gemma/phi3):
//   model.embed_tokens.weight                              -> token_embd.weight
//   model.norm.weight                                      -> output_norm.weight
//   lm_head.weight                                         -> output.weight
//   model.layers.{i}.input_layernorm.weight                -> blk.{i}.attn_norm.weight
//   model.layers.{i}.post_attention_layernorm.weight       -> blk.{i}.ffn_norm.weight
//   model.layers.{i}.self_attn.q_proj.{weight,bias}        -> blk.{i}.attn_q.{weight,bias}
//   model.layers.{i}.self_attn.k_proj.{weight,bias}        -> blk.{i}.attn_k.{weight,bias}
//   model.layers.{i}.self_attn.v_proj.{weight,bias}        -> blk.{i}.attn_v.{weight,bias}
//   model.layers.{i}.self_attn.o_proj.weight               -> blk.{i}.attn_output.weight
//   model.layers.{i}.mlp.gate_proj.weight                  -> blk.{i}.ffn_gate.weight
//   model.layers.{i}.mlp.up_proj.weight                    -> blk.{i}.ffn_up.weight
//   model.layers.{i}.mlp.down_proj.weight                  -> blk.{i}.ffn_down.weight

fn hf_to_gguf_name(hf: &str) -> Option<String> {
    // Top-level tensors
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }

    // Per-layer: parse "model.layers.{N}.<tail>"
    let rest = hf.strip_prefix("model.layers.")?;
    let (idx_str, tail) = rest.split_once('.')?;
    let idx: usize = idx_str.parse().ok()?;
    let blk = format!("blk.{idx}");

    let mapped = match tail {
        "input_layernorm.weight" => format!("{blk}.attn_norm.weight"),
        "post_attention_layernorm.weight" => format!("{blk}.ffn_norm.weight"),
        // Gemma2-style post-norms
        "pre_feedforward_layernorm.weight" => format!("{blk}.ffn_norm.weight"),
        "post_feedforward_layernorm.weight" => format!("{blk}.post_ffw_norm.weight"),

        "self_attn.q_proj.weight" => format!("{blk}.attn_q.weight"),
        "self_attn.k_proj.weight" => format!("{blk}.attn_k.weight"),
        "self_attn.v_proj.weight" => format!("{blk}.attn_v.weight"),
        "self_attn.o_proj.weight" => format!("{blk}.attn_output.weight"),
        "self_attn.q_proj.bias" => format!("{blk}.attn_q.bias"),
        "self_attn.k_proj.bias" => format!("{blk}.attn_k.bias"),
        "self_attn.v_proj.bias" => format!("{blk}.attn_v.bias"),

        "mlp.gate_proj.weight" => format!("{blk}.ffn_gate.weight"),
        "mlp.up_proj.weight" => format!("{blk}.ffn_up.weight"),
        "mlp.down_proj.weight" => format!("{blk}.ffn_down.weight"),

        _ => return None,
    };
    Some(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_basics() {
        assert_eq!(
            hf_to_gguf_name("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            hf_to_gguf_name("lm_head.weight").as_deref(),
            Some("output.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.27.mlp.gate_proj.weight").as_deref(),
            Some("blk.27.ffn_gate.weight")
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.0.self_attn.k_proj.bias").as_deref(),
            Some("blk.0.attn_k.bias")
        );
        assert!(hf_to_gguf_name("some.random.tensor").is_none());
    }
}
