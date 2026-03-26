/**
 * llama.cpp TurboQuant KV cache backend — implementation
 *
 * Bridges llama.cpp's KV cache system with tq-kv's Rust compression
 * via the C FFI (tq_kv.h).
 */

#include "llama-kv-tq.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

struct llama_tq_state * llama_tq_init(
    uint32_t n_layers,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    enum llama_tq_type type
) {
    if (type == LLAMA_TQ_NONE) return nullptr;
    if (n_layers == 0 || n_kv_heads == 0 || head_dim == 0) return nullptr;

    auto *state = new (std::nothrow) llama_tq_state;
    if (!state) return nullptr;

    state->n_layers = n_layers;
    state->type = type;
    state->enabled = true;
    state->layers = new (std::nothrow) llama_tq_layer[n_layers];

    if (!state->layers) {
        delete state;
        return nullptr;
    }

    uint8_t bits = static_cast<uint8_t>(type);

    for (uint32_t i = 0; i < n_layers; i++) {
        state->layers[i].type = type;
        state->layers[i].n_kv_heads = n_kv_heads;
        state->layers[i].head_dim = head_dim;
        state->layers[i].ctx = tq_layer_init(bits, n_kv_heads, head_dim, 0);

        if (!state->layers[i].ctx) {
            // Cleanup on failure
            for (uint32_t j = 0; j < i; j++) {
                tq_layer_free(state->layers[j].ctx);
            }
            delete[] state->layers;
            delete state;
            return nullptr;
        }
    }

    fprintf(stderr, "tq-kv: initialized %u layers, %u KV heads, dim=%u, %u-bit\n",
            n_layers, n_kv_heads, head_dim, bits);

    return state;
}

int llama_tq_compress_keys(
    struct llama_tq_state *state,
    uint32_t layer_id,
    const float *key_data
) {
    if (!state || !state->enabled || layer_id >= state->n_layers) return -1;
    if (!key_data) return -1;

    auto &layer = state->layers[layer_id];
    uint32_t len = layer.n_kv_heads * layer.head_dim;

    return tq_layer_compress_and_append(layer.ctx, key_data, len);
}

int llama_tq_fused_attention(
    const struct llama_tq_state *state,
    uint32_t layer_id,
    uint32_t kv_head_id,
    const float *query_data,
    float *scores_out,
    float scale
) {
    if (!state || !state->enabled || layer_id >= state->n_layers) return -1;

    auto &layer = state->layers[layer_id];
    if (kv_head_id >= layer.n_kv_heads) return -1;

    return tq_layer_fused_attention(
        layer.ctx, kv_head_id,
        query_data, layer.head_dim,
        scores_out, scale
    );
}

uint32_t llama_tq_cached_count(
    const struct llama_tq_state *state,
    uint32_t layer_id
) {
    if (!state || layer_id >= state->n_layers) return 0;
    return tq_layer_cached_count(state->layers[layer_id].ctx);
}

uint64_t llama_tq_total_memory(const struct llama_tq_state *state) {
    if (!state) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < state->n_layers; i++) {
        total += tq_layer_memory_bytes(state->layers[i].ctx);
    }
    return total;
}

void llama_tq_clear(struct llama_tq_state *state) {
    if (!state) return;
    for (uint32_t i = 0; i < state->n_layers; i++) {
        tq_layer_clear(state->layers[i].ctx);
    }
}

void llama_tq_free(struct llama_tq_state *state) {
    if (!state) return;
    for (uint32_t i = 0; i < state->n_layers; i++) {
        if (state->layers[i].ctx) {
            tq_layer_free(state->layers[i].ctx);
        }
    }
    delete[] state->layers;
    delete state;
}

enum llama_tq_type llama_tq_parse_type(const char *str) {
    if (!str) return LLAMA_TQ_NONE;
    if (strcmp(str, "tq2") == 0) return LLAMA_TQ_2BIT;
    if (strcmp(str, "tq3") == 0) return LLAMA_TQ_3BIT;
    if (strcmp(str, "tq4") == 0) return LLAMA_TQ_4BIT;
    return LLAMA_TQ_NONE;
}
