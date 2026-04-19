//! Generic TurboQuant-enhanced model — replaces turbo_llama.rs and turbo_qwen2.rs.
//!
//! Reads GGUF metadata to auto-configure for any standard transformer architecture
//! (llama, qwen2, mistral, phi3, gemma, etc.). Supports optional attention biases,
//! MoE, and tie_word_embeddings — all detected from GGUF tensors at load time.
//!
//! Uses the same proven TurboQuant KV cache compression as turbo_qwen2.rs:
//! - Selective compression (first N layers uncompressed)
//! - Halved RoPE (rope_manual)
//! - f32 softmax in fused SIMD path
//! - f32 attention in decompress path (GPU)
//!
//! ## Module structure (R.1 refactor)
//!
//! - `primitives` — Embedding, RmsNorm, softmax, softcap, fused SiLU
//! - `mlp` — QMatMul wrapper, Mlp, MoE, repeat_kv
//! - `kv_cache` — TQ compression, GPU KV buffers, config helpers, RoPE, TriAttention
//! - `layer` — LayerWeights, forward_attn (attention + TQ compression dispatch)
//! - `model` — GenericTurboModel, from_gguf, forward, CUDA graph management

// Thread-local raw device pointer for RoPE position GPU scalar.
// Points to the model's rope_pos_gpu CudaSlice. Updated before capture/replay.
#[cfg(feature = "cuda")]
std::thread_local! {
    static ROPE_POS_GPU_PTR: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

mod primitives;
pub use primitives::MAX_SEQ_LEN;

mod mlp;
mod kv_cache;
mod layer;
mod model;
#[cfg(all(feature = "cuda", feature = "persistent-kernel"))]
mod megakernel;

pub use model::GenericTurboModel;
pub(crate) use kv_cache::{set_triattention_override, clear_triattention_override, get_triattention_budget, set_auto_layer_bits};
