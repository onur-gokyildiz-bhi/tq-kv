# TODOS

## Completed

- [x] Incremental KV cache — O(1) per-token compress + append
- [x] Fused attention — pre-rotated query, centroid table lookup, no decompress
- [x] AVX2+FMA SIMD vectorization — 8.9x speedup in fused_dot_product
- [x] Rayon parallel multi-head attention
- [x] CUDA support — custom RmsNorm, RoPE, softmax (candle GGUF fix)
- [x] QJL removed from defaults — 29x faster compress, better ratio
- [x] C FFI layer — tq_kv.h + libtq_kv.a, 4 integration tests passing
- [x] llama.cpp integration patch — llama-kv-tq.cpp/.h, test_integration.c ALL PASSED
- [x] Perplexity benchmark — PPL 9.594 (+0.8%) on wikitext-2
- [x] NIAH benchmark — 9/9 pass at all depths
- [x] Context scaling — 4-bit flat +0.4-1.9% across 256-2048 tokens
- [x] no_std support — std feature flag for RNG
- [x] Batch fused_attention_scores() API
- [x] crates.io v0.2.0 published
- [x] Qwen 72B bug fix — attention Q/K/V bias + context_length from GGUF
- [x] QJL long context test — error does NOT accumulate, QJL removal confirmed
- [x] Project rename — turbo-quant → tq-kv
- [x] Directory rename — turbo-quant/ → tq-kv/
- [x] LICENSE-MIT + LICENSE-APACHE
- [x] V1 API deprecated, internal modules #[doc(hidden)]
- [x] README killer — PPL, NIAH, context scaling, C FFI, verified models
- [x] Design doc approved — hybrid engine (candle + custom kernels)

## Open — Phase 1 (This Week)

- [ ] candle PR: upstream RmsNorm CUDA implementation
- [ ] candle PR: upstream RoPE CUDA implementation
- [ ] candle PR: upstream softmax CUDA implementation

## Open — Phase 2 (Week 2-3)

- [ ] Q4_K_M AVX2 SIMD matmul kernel — close the gap to llama.cpp
- [ ] Benchmark: target within 3x of llama.cpp on short context
- [ ] Benchmark: FASTER than llama.cpp on 16K+ context

## Open — Phase 3 (Week 3-4)

- [ ] `.with_turbo_quant(bits)` API — builder pattern on candle models
- [ ] Auto quality gate — if PPL > threshold, auto-promote to next bit width
- [ ] tok/s, TTFT, memory profiling benchmarks

## Open — Phase 4 (After Performance Gates Met)

- [ ] HN post — focus on 4-bit PPL +0.8%, CUDA, AVX2 SIMD
- [ ] Reddit r/rust + r/LocalLLaMA posts
- [ ] Blog: technical deep-dive

## Future / Exploratory

- [ ] Claude Code local integration — TQ as MCP tool or local inference helper
- [ ] Chat tool — interactive TQ compression demo
- [ ] Metal shader (Apple Silicon) — deferred, needs Apple hardware
- [ ] PyO3 Python binding — deferred, no demand evidence yet
- [ ] Adaptive QJL — context-length-based threshold (if layer-level accumulation found)
- [ ] Layer-adaptive bitwidth — early layers 2-bit, late layers 4-bit
- [ ] Flash Attention tiling for fused TQ attention
