/**
 * tq-kv C FFI test — validates the static library works from C.
 *
 * Build (Windows MSVC):
 *   cl /I ../include test_ffi.c /link /LIBPATH:../../target/release tq_kv.lib
 *       ws2_32.lib advapi32.lib userenv.lib bcrypt.lib ntdll.lib
 *
 * Build (Linux/macOS):
 *   cc -I ../include test_ffi.c -L ../../target/release -ltq_kv -lpthread -lm -ldl -o test_ffi
 *
 * Run:
 *   ./test_ffi
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "tq_kv.h"

#define HEAD_DIM 128
#define N_KEYS 64
#define N_KV_HEADS 8

/* Simple deterministic "random" data */
static void fill_data(float *buf, int len, float seed) {
    for (int i = 0; i < len; i++) {
        buf[i] = sinf(seed * (i + 1) * 0.1f) * 0.5f;
    }
}

static int test_single_head(void) {
    printf("=== Single-Head API ===\n");

    TqContext *ctx = tq_init(2, HEAD_DIM, 0);
    if (!ctx) {
        printf("FAIL: tq_init returned NULL\n");
        return 1;
    }
    printf("  tq_init(2-bit, dim=%d): OK\n", HEAD_DIM);

    /* Compress and append N_KEYS keys */
    float key[HEAD_DIM];
    for (int k = 0; k < N_KEYS; k++) {
        fill_data(key, HEAD_DIM, (float)(k + 1));
        int rc = tq_compress_and_append(ctx, key, HEAD_DIM);
        if (rc != 0) {
            printf("FAIL: tq_compress_and_append returned %d at key %d\n", rc, k);
            tq_free(ctx);
            return 1;
        }
    }
    printf("  Compressed %d keys: OK\n", N_KEYS);

    /* Check count */
    uint32_t count = tq_cached_count(ctx);
    if (count != N_KEYS) {
        printf("FAIL: expected %d cached, got %u\n", N_KEYS, count);
        tq_free(ctx);
        return 1;
    }
    printf("  Cached count: %u (correct)\n", count);

    /* Check memory */
    uint64_t compressed = tq_memory_bytes(ctx);
    uint64_t original = tq_original_memory_bytes(ctx);
    float ratio = (float)original / (float)compressed;
    printf("  Memory: %llu bytes compressed, %llu original (%.1fx ratio)\n",
           (unsigned long long)compressed, (unsigned long long)original, ratio);

    if (ratio < 5.0f) {
        printf("FAIL: 2-bit ratio should be > 5x, got %.1f\n", ratio);
        tq_free(ctx);
        return 1;
    }

    /* Fused attention */
    float query[HEAD_DIM];
    fill_data(query, HEAD_DIM, 99.0f);

    float *scores = (float *)malloc(N_KEYS * sizeof(float));
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    int n = tq_fused_attention(ctx, query, HEAD_DIM, scores, scale);
    if (n != N_KEYS) {
        printf("FAIL: fused_attention returned %d, expected %d\n", n, N_KEYS);
        free(scores);
        tq_free(ctx);
        return 1;
    }

    /* Verify scores are finite and non-zero */
    int all_finite = 1;
    int any_nonzero = 0;
    for (int i = 0; i < N_KEYS; i++) {
        if (!isfinite(scores[i])) all_finite = 0;
        if (fabsf(scores[i]) > 1e-10f) any_nonzero = 1;
    }

    if (!all_finite || !any_nonzero) {
        printf("FAIL: scores invalid (finite=%d, nonzero=%d)\n", all_finite, any_nonzero);
        free(scores);
        tq_free(ctx);
        return 1;
    }
    printf("  Fused attention (%d scores): OK (range [%.4f, %.4f])\n",
           n, scores[0], scores[N_KEYS - 1]);

    /* Clear and verify */
    tq_clear(ctx);
    if (tq_cached_count(ctx) != 0) {
        printf("FAIL: clear didn't reset count\n");
        free(scores);
        tq_free(ctx);
        return 1;
    }
    printf("  Clear: OK\n");

    free(scores);
    tq_free(ctx);
    printf("  PASS\n\n");
    return 0;
}

static int test_layer_api(void) {
    printf("=== Layer (Multi-Head) API ===\n");

    TqLayerContext *ctx = tq_layer_init(4, N_KV_HEADS, HEAD_DIM, 0);
    if (!ctx) {
        printf("FAIL: tq_layer_init returned NULL\n");
        return 1;
    }
    printf("  tq_layer_init(4-bit, %d heads, dim=%d): OK\n", N_KV_HEADS, HEAD_DIM);

    /* Append 32 tokens (each token = N_KV_HEADS * HEAD_DIM floats) */
    int n_tokens = 32;
    float *token_keys = (float *)malloc(N_KV_HEADS * HEAD_DIM * sizeof(float));

    for (int t = 0; t < n_tokens; t++) {
        fill_data(token_keys, N_KV_HEADS * HEAD_DIM, (float)(t + 1));
        int rc = tq_layer_compress_and_append(ctx, token_keys, N_KV_HEADS * HEAD_DIM);
        if (rc != 0) {
            printf("FAIL: layer compress at token %d\n", t);
            free(token_keys);
            tq_layer_free(ctx);
            return 1;
        }
    }
    printf("  Compressed %d tokens x %d heads: OK\n", n_tokens, N_KV_HEADS);

    uint32_t count = tq_layer_cached_count(ctx);
    if (count != (uint32_t)n_tokens) {
        printf("FAIL: expected %d, got %u\n", n_tokens, count);
        free(token_keys);
        tq_layer_free(ctx);
        return 1;
    }

    uint64_t mem = tq_layer_memory_bytes(ctx);
    printf("  Total memory: %llu bytes across %d heads\n",
           (unsigned long long)mem, N_KV_HEADS);

    /* Fused attention per head */
    float query[HEAD_DIM];
    fill_data(query, HEAD_DIM, 42.0f);
    float *scores = (float *)malloc(n_tokens * sizeof(float));
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int h = 0; h < N_KV_HEADS; h++) {
        int n = tq_layer_fused_attention(ctx, h, query, HEAD_DIM, scores, scale);
        if (n != n_tokens) {
            printf("FAIL: head %d returned %d scores\n", h, n);
            free(scores);
            free(token_keys);
            tq_layer_free(ctx);
            return 1;
        }
    }
    printf("  Fused attention (%d heads x %d positions): OK\n", N_KV_HEADS, n_tokens);

    tq_layer_clear(ctx);
    if (tq_layer_cached_count(ctx) != 0) {
        printf("FAIL: layer clear didn't reset\n");
        free(scores);
        free(token_keys);
        tq_layer_free(ctx);
        return 1;
    }
    printf("  Clear: OK\n");

    free(scores);
    free(token_keys);
    tq_layer_free(ctx);
    printf("  PASS\n\n");
    return 0;
}

static int test_all_bitwidths(void) {
    printf("=== All Bitwidths ===\n");
    uint8_t bits_list[] = {2, 3, 4};

    for (int b = 0; b < 3; b++) {
        uint8_t bits = bits_list[b];
        TqContext *ctx = tq_init(bits, HEAD_DIM, 0);

        float key[HEAD_DIM];
        for (int k = 0; k < 16; k++) {
            fill_data(key, HEAD_DIM, (float)(k + 1));
            tq_compress_and_append(ctx, key, HEAD_DIM);
        }

        uint64_t compressed = tq_memory_bytes(ctx);
        uint64_t original = tq_original_memory_bytes(ctx);
        float ratio = (float)original / (float)compressed;
        printf("  %d-bit: %.1fx compression (%llu -> %llu bytes)\n",
               bits, ratio, (unsigned long long)original, (unsigned long long)compressed);

        tq_free(ctx);
    }

    printf("  PASS\n\n");
    return 0;
}

int main(void) {
    printf("tq-kv C FFI Test Suite\n");
    printf("======================\n\n");

    int failures = 0;
    failures += test_single_head();
    failures += test_layer_api();
    failures += test_all_bitwidths();

    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TEST(S) FAILED\n", failures);
    }

    return failures;
}
