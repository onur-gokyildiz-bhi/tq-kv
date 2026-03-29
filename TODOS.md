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
- [x] LICENSE-MIT + LICENSE-APACHE
- [x] V1 API deprecated, internal modules #[doc(hidden)]
- [x] README killer — PPL, NIAH, context scaling, C FFI, verified models
- [x] Design doc approved — hybrid engine (candle + custom kernels)
- [x] candle 0.9.2 dependency update
- [x] Metric audit — BENCHMARK.md = single source
- [x] TurboQuant comparison table (README section)
- [x] tq-demo CLI — bundled NIAH test + compression stats
- [x] HN post draft (docs/hn-launch.md)
- [x] Qwen 72B RoPE fix — halved vs interleaved layout
- [x] TurboKvCache — generic drop-in replacement for candle KvCache
- [x] GenericTurboModel — unified engine for ALL GGUF architectures
- [x] Selective layer compression — first N layers uncompressed for deep models
- [x] f32 softmax fix — prevents precision loss at long context

### v0.5.0 Sprint (Mar 28-29, 2026)
- [x] 3-Fix Framework — sink tokens + POQ + cache reset (GGUF compound error fix)
- [x] SRHT QJL — O(d²) → O(d log d), 115x speedup, +4.5 dB SNR
- [x] Adaptive QJL — QjlMode::Off/On/Adaptive{threshold}, context-aware
- [x] Norm Correction — ||decompress|| matches ||original||, zero decode cost
- [x] Gaussianity Verification — kurtosis 35.3 → 3.3 (confirms Lloyd-Max assumptions)
- [x] Dead code cleanup — -1,198 LOC, turbo_llama/turbo_qwen2 removed
- [x] Attention KL divergence test — SRHT QJL 2.9x better at all context lengths
- [x] crates.io v0.5.0 published
- [x] Paper draft — "TurboQuant on Quantized Models" (docs/arxiv/)
- [x] HN + X launch content (docs/hn-launch.md, docs/x-launch-tweets.md)

### Post-v0.5.0 Sprint (Mar 29, 2026)
- [x] Sparse V — skip V rows when softmax < threshold (TQ_SPARSE_V, AVX2 SIMD)
- [x] K/V Asymmetric — values 8-bit absmax, keys 4-bit (TQ_VBITS=8, ~1.9x V savings)
- [x] Temporal Decay — older tokens demoted to 2-bit via index remap (TQ_DECAY=512:2)
- [x] Fused attention path — TQ_FUSED=1, scores from compressed indices
- [x] Multi-turn chat bug fix — build_prompt() replaces extract_prompts()
- [x] Sink token shape mismatch fix — POQ current key only when n_compressed > 0
- [x] README + BENCHMARK + version bump (tq-engine 0.4.0)
- [x] PPL benchmark — Qwen 2.5 7B Q4_K_M (baseline 5.088, 4-bit 6.042 +18.7%)
- [x] CUDA throughput — Qwen 28.2 tok/s, Llama 19.2 tok/s
- [x] --tq-only bench flag
- [x] Per-channel key scaling — SmoothQuant-style (calibrate_channel_scales)
- [x] Engine::builder() API — .with_turbo_quant(bits), .with_quality_gate()
- [x] Auto quality gate — QualityGate monitors running PPL during generation
- [x] Layer-adaptive bitwidth — TQ_LAYER_BITS per-layer bit width ranges
- [x] Softmax bias correction — Bondarenko pre-compensation (TQ_BIAS_CORRECT=1)
- [x] Qwen3.5 9B support — catalog entry qwen3:9b

## Open — Next Sprint: Safetensors/FP16 Support

> Goal: tq-kv works on FP16 models, not just GGUF. Eliminates compound error,
> enables fair comparison with RotorQuant/Alican/TheTom, opens HF ecosystem.

- [ ] Safetensors model loader — candle safetensors reader for FP16/BF16 models
- [ ] FP16 TurboModel — GenericTurboModel variant using FP16 weights (no QMatMul)
- [ ] FP16 PPL benchmark — Qwen 2.5 7B FP16 vs TQ 4-bit (expect PPL <+1%)
- [ ] HuggingFace model hub integration — load by repo name (e.g. "Qwen/Qwen2.5-7B")

## Open — Launch (en son)

- [ ] arXiv submission — LaTeX compile + upload
- [ ] HN Show post — submit docs/hn-launch.md
- [ ] X thread launch — docs/x-launch-tweets.md + twitter-visuals.html
- [ ] Reddit posts: r/rust + r/LocalLLaMA
- [ ] candle-examples PR (TurboQuant KV cache example)
- [ ] Blog: technical deep-dive (SIMD performance story)

## Open — Performance

- [ ] llama.cpp Q4_K_M baseline — record tok/s on same hardware
- [ ] Benchmark: target within 3x of llama.cpp on short context (≤4K)
- [ ] Benchmark: FASTER than llama.cpp on 16K+ context (TQ advantage)
- [ ] tok/s, TTFT, memory profiling benchmarks
- [ ] GitHub Actions CI/CD

## Future / Exploratory

- [ ] Calibrated codebook — Lloyd-Max from model activations (not assumed Gaussian)
- [ ] Learned rotation — SpinQuant-style (up to 45% better than random Hadamard)
- [ ] Flash Attention tiling for fused TQ attention
- [ ] Metal shader (Apple Silicon) — deferred, needs Apple hardware
- [ ] PyO3 Python binding — deferred, no demand evidence yet
- [ ] Web playground demo
- [ ] Claude Code local integration — TQ as MCP tool
