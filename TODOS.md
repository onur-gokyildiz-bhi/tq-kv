# TODOS

## Completed

*(see git log for full history)*

### Post-v0.5.0 Sprint (Mar 29-30, 2026)
- [x] Sparse V, K/V Asymmetric, Temporal Decay, Fused Attention
- [x] Multi-turn chat fix, sink token shape mismatch fix
- [x] README + BENCHMARK + version bump (tq-engine 0.4.0)
- [x] PPL benchmark — Qwen 2.5 7B Q4_K_M + Qwen 0.5B FP16
- [x] CUDA throughput — Qwen 28.2 tok/s, Llama 19.2 tok/s
- [x] Per-channel scaling, builder API, quality gate, layer-adaptive, softmax bias, Qwen3.5
- [x] Safetensors/FP16 model loading (from_safetensors + HF repo name)
- [x] Group quantization (g=32) — per-group sigma
- [x] Residual quantization (TQ_RESIDUAL) — two-pass error correction
- [x] Dense-and-sparse outlier preservation (TQ_OUTLIER=K)
- [x] PPL optimized: +11.5% → +0.32% (4-bit K=8 RES=3 on FP16 Qwen 0.5B)

## Open — Performance

- [ ] llama.cpp Q4_K_M baseline — record tok/s on same hardware
- [ ] Benchmark: target within 3x of llama.cpp on short context (≤4K)
- [ ] Benchmark: FASTER than llama.cpp on 16K+ context (TQ advantage)
- [ ] tok/s, TTFT, memory profiling benchmarks
- [ ] GitHub Actions CI/CD

## Open — Launch (en son)

- [ ] arXiv submission — LaTeX compile + upload
- [ ] HN Show post — submit docs/hn-launch.md
- [ ] X thread launch
- [ ] Reddit posts: r/rust + r/LocalLLaMA
- [ ] candle-examples PR
- [ ] Blog: SIMD deep-dive

## Future / Exploratory

- [ ] Calibrated codebook — Lloyd-Max from model activations
- [ ] Learned rotation — SpinQuant-style
- [ ] Flash Attention tiling for fused TQ attention
- [ ] Metal shader (Apple Silicon)
- [ ] PyO3 Python binding
- [ ] Web playground demo
- [ ] Claude Code local integration — TQ as MCP tool
