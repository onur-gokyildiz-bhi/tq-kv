//! C FFI layer for tq-kv.
//!
//! Exposes TurboQuant compression as a C static library (libtq_kv.a).
//! Designed for integration with llama.cpp and other C/C++ inference engines.
//!
//! ## Usage from C
//!
//! ```c
//! #include "tq_kv.h"
//!
//! TqContext *ctx = tq_init(2, 128, 0);  // 2-bit, head_dim=128
//! tq_compress_and_append(ctx, key_data, 128);
//! float scores[n_keys];
//! tq_fused_attention(ctx, query_data, 128, scores, n_keys);
//! tq_free(ctx);
//! ```

use std::slice;

use crate::codebook;
use crate::hadamard;
use crate::{
    compress_single_key_with_signs, fused_dot_product_with_centroids, CompressedKeys,
    TurboQuantConfig,
};

/// Opaque context holding compression state for one KV head.
pub struct TqContext {
    config: TurboQuantConfig,
    cache: CompressedKeys,
    signs: Vec<f32>,
    centroids: &'static [f32],
    dim: usize,
}

/// Opaque context holding per-layer state (multiple KV heads).
pub struct TqLayerContext {
    heads: Vec<TqContext>,
    n_kv_heads: usize,
    head_dim: usize,
    bits: u8,
}

// ============================================================
// Single-head API
// ============================================================

/// Initialize a TurboQuant context for one KV head.
///
/// `bits`: quantization bit width (2, 3, or 4)
/// `head_dim`: dimension per attention head (must be power of 2)
/// `rotation_seed`: Hadamard rotation seed (0 for default)
///
/// Returns opaque pointer. Must be freed with `tq_free`.
#[no_mangle]
pub extern "C" fn tq_init(bits: u8, head_dim: u32, rotation_seed: u64) -> *mut TqContext {
    let dim = head_dim as usize;
    let config = TurboQuantConfig {
        bits,
        use_qjl: false,
        rotation_seed: if rotation_seed == 0 {
            0x0054_5552_4230
        } else {
            rotation_seed
        },
        ..TurboQuantConfig::default()
    };
    let signs = hadamard::generate_signs(dim, config.rotation_seed);
    let centroids = codebook::get_centroids(bits);
    let cache = CompressedKeys::new_empty(bits, dim, config.rotation_seed);

    let ctx = Box::new(TqContext {
        config,
        cache,
        signs,
        centroids,
        dim,
    });
    Box::into_raw(ctx)
}

/// Compress a single key vector and append to the cache.
///
/// `ctx`: context from `tq_init`
/// `key_data`: pointer to f32 array of length `head_dim`
/// `len`: number of f32 elements (must equal head_dim)
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn tq_compress_and_append(
    ctx: *mut TqContext,
    key_data: *const f32,
    len: u32,
) -> i32 {
    if ctx.is_null() || key_data.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;
    let len = len as usize;
    if len != ctx.dim {
        return -1;
    }

    let key = slice::from_raw_parts(key_data, len);
    let (packed, norm) = compress_single_key_with_signs(key, ctx.dim, &ctx.config, &ctx.signs);
    ctx.cache.append_raw(&packed, norm);
    0
}

/// Compute fused attention scores between a query and all cached keys.
///
/// `ctx`: context with cached keys
/// `query_data`: pointer to f32 query vector of length `head_dim`
/// `query_len`: number of f32 elements (must equal head_dim)
/// `scores_out`: output buffer, must have space for at least `tq_cached_count(ctx)` floats
/// `scale`: attention scale factor (typically 1/sqrt(head_dim))
///
/// Returns number of scores written, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn tq_fused_attention(
    ctx: *const TqContext,
    query_data: *const f32,
    query_len: u32,
    scores_out: *mut f32,
    scale: f32,
) -> i32 {
    if ctx.is_null() || query_data.is_null() || scores_out.is_null() {
        return -1;
    }
    let ctx = &*ctx;
    let query_len = query_len as usize;
    if query_len != ctx.dim {
        return -1;
    }
    if ctx.cache.count == 0 {
        return 0;
    }

    let query = slice::from_raw_parts(query_data, query_len);

    // Pre-rotate query: R*q so that <R*q, centroids[idx]> = <q, R^T*centroids[idx]>
    let rotated_q = crate::pre_rotate_query_with_signs(query, &ctx.signs);

    let scores = slice::from_raw_parts_mut(scores_out, ctx.cache.count);
    let bpv = ctx.cache.bytes_per_vector();

    for pos in 0..ctx.cache.count {
        let norm = ctx.cache.norms[pos];
        if norm < 1e-10 {
            scores[pos] = 0.0;
            continue;
        }
        let start = pos * bpv;
        let end = start + bpv;
        let indices =
            codebook::unpack_indices(&ctx.cache.packed_indices[start..end], ctx.dim, ctx.cache.bits);
        scores[pos] =
            fused_dot_product_with_centroids(&rotated_q, &indices, norm, ctx.centroids, ctx.dim)
                * scale;
    }

    ctx.cache.count as i32
}

/// Get number of cached key positions.
#[no_mangle]
pub unsafe extern "C" fn tq_cached_count(ctx: *const TqContext) -> u32 {
    if ctx.is_null() {
        return 0;
    }
    (*ctx).cache.count as u32
}

/// Get compressed memory usage in bytes.
#[no_mangle]
pub unsafe extern "C" fn tq_memory_bytes(ctx: *const TqContext) -> u64 {
    if ctx.is_null() {
        return 0;
    }
    (*ctx).cache.memory_bytes() as u64
}

/// Get original (uncompressed fp16) memory usage in bytes.
#[no_mangle]
pub unsafe extern "C" fn tq_original_memory_bytes(ctx: *const TqContext) -> u64 {
    if ctx.is_null() {
        return 0;
    }
    (*ctx).cache.original_memory_bytes() as u64
}

/// Reset the cache (clear all cached keys).
#[no_mangle]
pub unsafe extern "C" fn tq_clear(ctx: *mut TqContext) {
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    ctx.cache = CompressedKeys::new_empty(ctx.config.bits, ctx.dim, ctx.config.rotation_seed);
}

/// Free context. Must be called for every `tq_init`.
#[no_mangle]
pub unsafe extern "C" fn tq_free(ctx: *mut TqContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

// ============================================================
// Multi-head (layer) API — convenience for llama.cpp
// ============================================================

/// Initialize a full layer context (multiple KV heads).
///
/// `bits`: 2, 3, or 4
/// `n_kv_heads`: number of KV attention heads
/// `head_dim`: dimension per head
/// `rotation_seed`: 0 for default
#[no_mangle]
pub extern "C" fn tq_layer_init(
    bits: u8,
    n_kv_heads: u32,
    head_dim: u32,
    rotation_seed: u64,
) -> *mut TqLayerContext {
    let n_kv_heads = n_kv_heads as usize;
    let dim = head_dim as usize;
    let seed = if rotation_seed == 0 {
        0x0054_5552_4230
    } else {
        rotation_seed
    };

    let signs = hadamard::generate_signs(dim, seed);
    let centroids = codebook::get_centroids(bits);

    let heads: Vec<TqContext> = (0..n_kv_heads)
        .map(|_| {
            let config = TurboQuantConfig {
                bits,
                use_qjl: false,
                rotation_seed: seed,
                ..TurboQuantConfig::default()
            };
            TqContext {
                config: config.clone(),
                cache: CompressedKeys::new_empty(bits, dim, seed),
                signs: signs.clone(),
                centroids,
                dim,
            }
        })
        .collect();

    let ctx = Box::new(TqLayerContext {
        heads,
        n_kv_heads,
        head_dim: dim,
        bits,
    });
    Box::into_raw(ctx)
}

/// Compress key vectors for all KV heads and append to cache.
///
/// `key_data`: pointer to f32 array of shape [n_kv_heads * head_dim]
///             (all heads concatenated, one token's keys)
/// `len`: total number of f32 elements
#[no_mangle]
pub unsafe extern "C" fn tq_layer_compress_and_append(
    ctx: *mut TqLayerContext,
    key_data: *const f32,
    len: u32,
) -> i32 {
    if ctx.is_null() || key_data.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;
    let len = len as usize;
    let expected = ctx.n_kv_heads * ctx.head_dim;
    if len != expected {
        return -1;
    }

    let data = slice::from_raw_parts(key_data, len);
    for h in 0..ctx.n_kv_heads {
        let offset = h * ctx.head_dim;
        let key = &data[offset..offset + ctx.head_dim];
        let (packed, norm) =
            compress_single_key_with_signs(key, ctx.head_dim, &ctx.heads[h].config, &ctx.heads[h].signs);
        ctx.heads[h].cache.append_raw(&packed, norm);
    }
    0
}

/// Compute fused attention scores for one query head against one KV head's cache.
///
/// `kv_head_idx`: which KV head to attend to
/// `query_data`: f32 query vector [head_dim]
/// `scores_out`: output buffer [cached_count]
/// `scale`: 1/sqrt(head_dim)
#[no_mangle]
pub unsafe extern "C" fn tq_layer_fused_attention(
    ctx: *const TqLayerContext,
    kv_head_idx: u32,
    query_data: *const f32,
    query_len: u32,
    scores_out: *mut f32,
    scale: f32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let ctx = &*ctx;
    let kv_head_idx = kv_head_idx as usize;
    if kv_head_idx >= ctx.n_kv_heads {
        return -1;
    }

    tq_fused_attention(
        &ctx.heads[kv_head_idx] as *const TqContext,
        query_data,
        query_len,
        scores_out,
        scale,
    )
}

/// Get cached key count (same for all heads).
#[no_mangle]
pub unsafe extern "C" fn tq_layer_cached_count(ctx: *const TqLayerContext) -> u32 {
    if ctx.is_null() {
        return 0;
    }
    let ctx = &*ctx;
    if ctx.heads.is_empty() {
        return 0;
    }
    ctx.heads[0].cache.count as u32
}

/// Get total compressed memory across all heads.
#[no_mangle]
pub unsafe extern "C" fn tq_layer_memory_bytes(ctx: *const TqLayerContext) -> u64 {
    if ctx.is_null() {
        return 0;
    }
    let ctx = &*ctx;
    ctx.heads.iter().map(|h| h.cache.memory_bytes() as u64).sum()
}

/// Clear all heads' caches.
#[no_mangle]
pub unsafe extern "C" fn tq_layer_clear(ctx: *mut TqLayerContext) {
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    for head in &mut ctx.heads {
        head.cache =
            CompressedKeys::new_empty(head.config.bits, head.dim, head.config.rotation_seed);
    }
}

/// Free layer context.
#[no_mangle]
pub unsafe extern "C" fn tq_layer_free(ctx: *mut TqLayerContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}
