/**
 * tq-kv: TurboQuant KV Cache Compression for LLMs
 *
 * Pure Rust library exposed via C FFI.
 * Link with: -ltq_kv (or tq_kv.lib on Windows)
 *
 * Usage:
 *   TqContext *ctx = tq_init(2, 128, 0);
 *   tq_compress_and_append(ctx, key_data, 128);
 *   float scores[n];
 *   tq_fused_attention(ctx, query_data, 128, scores, scale);
 *   tq_free(ctx);
 *
 * https://github.com/onur-gokyildiz-bhi/tq-kv
 * License: MIT OR Apache-2.0
 */

#ifndef TQ_KV_H
#define TQ_KV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque types */
typedef struct TqContext TqContext;
typedef struct TqLayerContext TqLayerContext;

/* ============================================================
 * Single-head API
 * ============================================================ */

/**
 * Initialize a TurboQuant context for one KV head.
 *
 * @param bits          Quantization bit width (2, 3, or 4)
 * @param head_dim      Dimension per attention head (must be power of 2)
 * @param rotation_seed Hadamard rotation seed (0 for default)
 * @return Opaque pointer. Must be freed with tq_free().
 */
TqContext *tq_init(uint8_t bits, uint32_t head_dim, uint64_t rotation_seed);

/**
 * Compress a single key vector and append to cache.
 *
 * @param ctx      Context from tq_init()
 * @param key_data Pointer to float array of length head_dim
 * @param len      Number of floats (must equal head_dim)
 * @return 0 on success, -1 on error
 */
int32_t tq_compress_and_append(TqContext *ctx, const float *key_data, uint32_t len);

/**
 * Compute fused attention scores (no key decompression).
 *
 * Uses pre-rotated query trick: <R*q, centroids[idx]> * sigma.
 * AVX2+FMA SIMD accelerated on x86_64.
 *
 * @param ctx        Context with cached keys
 * @param query_data Float query vector [head_dim]
 * @param query_len  Length (must equal head_dim)
 * @param scores_out Output buffer [tq_cached_count(ctx)]
 * @param scale      Attention scale (typically 1/sqrt(head_dim))
 * @return Number of scores written, or -1 on error
 */
int32_t tq_fused_attention(const TqContext *ctx, const float *query_data,
                           uint32_t query_len, float *scores_out, float scale);

/** Get number of cached key positions. */
uint32_t tq_cached_count(const TqContext *ctx);

/** Get compressed memory usage in bytes. */
uint64_t tq_memory_bytes(const TqContext *ctx);

/** Get original (uncompressed fp16) memory in bytes. */
uint64_t tq_original_memory_bytes(const TqContext *ctx);

/** Clear cache (remove all cached keys). */
void tq_clear(TqContext *ctx);

/** Free context. Must call for every tq_init(). */
void tq_free(TqContext *ctx);

/* ============================================================
 * Multi-head (layer) API — convenience for llama.cpp
 * ============================================================ */

/**
 * Initialize a full layer context (all KV heads).
 *
 * @param bits          2, 3, or 4
 * @param n_kv_heads    Number of KV attention heads
 * @param head_dim      Dimension per head
 * @param rotation_seed 0 for default
 */
TqLayerContext *tq_layer_init(uint8_t bits, uint32_t n_kv_heads,
                              uint32_t head_dim, uint64_t rotation_seed);

/**
 * Compress one token's keys across all KV heads.
 *
 * @param key_data Float array [n_kv_heads * head_dim]
 * @param len      Total floats (must equal n_kv_heads * head_dim)
 */
int32_t tq_layer_compress_and_append(TqLayerContext *ctx,
                                     const float *key_data, uint32_t len);

/**
 * Fused attention for one query head against one KV head.
 *
 * For GQA: map query_head -> kv_head, then call this.
 */
int32_t tq_layer_fused_attention(const TqLayerContext *ctx,
                                 uint32_t kv_head_idx,
                                 const float *query_data, uint32_t query_len,
                                 float *scores_out, float scale);

uint32_t tq_layer_cached_count(const TqLayerContext *ctx);
uint64_t tq_layer_memory_bytes(const TqLayerContext *ctx);
void tq_layer_clear(TqLayerContext *ctx);
void tq_layer_free(TqLayerContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TQ_KV_H */
