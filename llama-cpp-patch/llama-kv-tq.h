/**
 * llama.cpp TurboQuant KV cache backend
 *
 * Integrates tq-kv (Rust) via C FFI for 2-4 bit KV cache compression.
 * Requires: libtq_kv.a linked, LLAMA_TQ_KV defined.
 */

#ifndef LLAMA_KV_TQ_H
#define LLAMA_KV_TQ_H

#include "tq_kv.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TurboQuant cache type identifiers */
enum llama_tq_type {
    LLAMA_TQ_NONE = 0,
    LLAMA_TQ_2BIT = 2,
    LLAMA_TQ_3BIT = 3,
    LLAMA_TQ_4BIT = 4,
};

/* Per-layer TurboQuant state */
struct llama_tq_layer {
    TqLayerContext *ctx;       /* Rust FFI context */
    enum llama_tq_type type;
    uint32_t n_kv_heads;
    uint32_t head_dim;
};

/* Full model TurboQuant state */
struct llama_tq_state {
    struct llama_tq_layer *layers;
    uint32_t n_layers;
    enum llama_tq_type type;
    bool enabled;
};

/**
 * Initialize TurboQuant state for a model.
 *
 * Call after model loading, before inference.
 *
 * @param n_layers   Number of transformer layers
 * @param n_kv_heads Number of KV attention heads per layer
 * @param head_dim   Dimension per head
 * @param type       LLAMA_TQ_2BIT, LLAMA_TQ_3BIT, or LLAMA_TQ_4BIT
 * @return Allocated state, or NULL on error
 */
struct llama_tq_state * llama_tq_init(
    uint32_t n_layers,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    enum llama_tq_type type
);

/**
 * Compress and store key vectors for one layer, one token.
 *
 * Called during KV cache update (llama_kv_cache_update or equivalent).
 *
 * @param state    TurboQuant state
 * @param layer_id Layer index
 * @param key_data Float key data [n_kv_heads * head_dim] (after RoPE)
 */
int llama_tq_compress_keys(
    struct llama_tq_state *state,
    uint32_t layer_id,
    const float *key_data
);

/**
 * Compute fused attention scores for one query head.
 *
 * Replaces Q @ K^T for the TurboQuant-compressed keys.
 * Returns attention scores (before softmax).
 *
 * @param state      TurboQuant state
 * @param layer_id   Layer index
 * @param kv_head_id KV head to attend to
 * @param query_data Float query vector [head_dim] (after RoPE)
 * @param scores_out Output buffer [n_cached_tokens]
 * @param scale      1.0 / sqrt(head_dim)
 * @return Number of scores, or -1 on error
 */
int llama_tq_fused_attention(
    const struct llama_tq_state *state,
    uint32_t layer_id,
    uint32_t kv_head_id,
    const float *query_data,
    float *scores_out,
    float scale
);

/** Get number of cached tokens for a layer. */
uint32_t llama_tq_cached_count(
    const struct llama_tq_state *state,
    uint32_t layer_id
);

/** Get total compressed memory across all layers (bytes). */
uint64_t llama_tq_total_memory(const struct llama_tq_state *state);

/** Clear all caches (e.g., for new conversation). */
void llama_tq_clear(struct llama_tq_state *state);

/** Free all TurboQuant state. */
void llama_tq_free(struct llama_tq_state *state);

/**
 * Parse cache type string from CLI.
 * "tq2" -> LLAMA_TQ_2BIT, "tq3" -> LLAMA_TQ_3BIT, etc.
 * Returns LLAMA_TQ_NONE if not a TurboQuant type.
 */
enum llama_tq_type llama_tq_parse_type(const char *str);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_KV_TQ_H */
