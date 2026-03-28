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
- [x] candle CUDA ops: RmsNorm, RoPE, softmax — upstream already in candle 0.9.2
- [x] candle 0.9.2 dependency update — tested, working
- [x] Metric audit — version + benchmark numbers aligned, BENCHMARK.md = single source
- [x] fused_attention_scores() dedicated test + buffer reuse optimization
- [x] Inline comments translated to English (5 source files)
- [x] TurboQuant comparison table (README section)
- [x] tq-demo CLI — bundled NIAH test + compression stats
- [x] HN post draft (docs/hn-launch.md)
- [x] Qwen 72B RoPE fix — halved vs interleaved layout (root cause of long-context bug)
- [x] TurboKvCache — generic drop-in replacement for candle KvCache
- [x] GenericTurboModel — unified engine for ALL GGUF architectures (llama, qwen2, mistral, gemma, phi3)
- [x] Qwen 72B 4-bit validated: 150 token coherent output (CPU + GPU)
- [x] Qwen2.5 7B GPU validated: correct output on CUDA RTX 3080
- [x] Selective layer compression — first N layers uncompressed for deep models
- [x] f32 softmax fix — prevents precision loss at long context

## Open — Phase 2: Performance (This Week)

- [ ] llama.cpp Q4_K_M baseline — record tok/s on same hardware (RTX 3080, Qwen 7B)
- [ ] Q4_K_M AVX2 SIMD matmul kernel — close the gap to llama.cpp
- [ ] Benchmark: target within 3x of llama.cpp on short context (≤4K)
- [ ] Benchmark: FASTER than llama.cpp on 16K+ context (TQ advantage)
- [ ] Blog: technical deep-dive (SIMD performance story)
- [ ] "v2 update" HN/Reddit post with performance results

## Open — Phase 3: Developer Experience (Week 3-4)

- [ ] `.with_turbo_quant(bits)` API — builder pattern on candle models
- [ ] Auto quality gate — if PPL > +2% threshold, auto-promote to next bit width
- [ ] tok/s, TTFT, memory profiling benchmarks

## Open — Launch

- [ ] HN post (docs/hn-launch.md ready — submit when timing is right)
- [ ] Reddit posts: r/rust + r/LocalLLaMA
- [ ] candle-examples PR (TurboQuant KV cache example)

## Future / Exploratory

- [ ] GitHub Actions CI/CD
- [ ] Web playground demo (after CLI demo validates UX)
- [ ] Claude Code local integration — TQ as MCP tool or local inference helper
- [ ] Metal shader (Apple Silicon) — deferred, needs Apple hardware
- [ ] PyO3 Python binding — deferred, no demand evidence yet
- [ ] Re-evaluate QJL as default — SRHT brings overhead to 2x (was 29x), +4.5 dB SNR. Consider enabling by default at 2-bit.
- [ ] Adaptive QJL — context-length-based threshold
- [ ] Layer-adaptive bitwidth — early layers 2-bit, late layers 4-bit
- [ ] Flash Attention tiling for fused TQ attention
- [ ] Qwen3.5 architecture support (GGUF arch = "qwen35", needs testing)
