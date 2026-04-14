# Changelog

All notable changes to **tq-engine** and the **tq-kv** library are documented
here. Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0-alpha] — in progress, 2026-04

### Added
- **mrow ladder** kernel family in `kernels/fused_mlp.cu` and `kernels/qmatmul.cu`:
  cooperative-per-superblock thread layout, 256/256 threads active per block.
- `fused_addnorm_q4km_gateup_silu_cpasync_f32` — double-buffered shared memory
  (2×288 B) with `cp.async` prefetch of the next superblock's gate+up weights.
- `fused_q4km_down_residual_cpasync_f32` — cooperative MLP down kernel with
  `cp.async` W-prefetch (+14% end-to-end).
- `q4km_matvec_wx_cpasync_f32` — `cp.async` double-buffer for both X and W in
  the decode matvec (+18–25% kernel time on LM head / Wo).
- `fused_norm_q4km_qkv_bias_cpasync_f32` — opt-in QKV variant with `cp.async`
  W-prefetch. Default OFF (kernel −17% but end-to-end flat due to sync cost).
- `rope_halved_qk_f32` / `rope_interleaved_qk_f32` — single-launch fused
  RoPE covering Q and K heads in one kernel.
- `BENCHMARKS.md` — reproducible end-to-end performance matrix with exact
  CLI invocations, methodology, and raw-log policy.
- `CHANGELOG.md` — this file.

### Changed
- Default kernel dispatch: `TQ_GATEUP=mrow8`, `TQ_Q4KM=mrow8`,
  `TQ_MLP_DOWN=cpasync`. Baselines remain selectable via env var for A/B.
- **TriAttention V3** hybrid threshold: switches from brute-force scoring to
  the fused compressed-KV path once budget × ctx exceeds the measured crossover.
- README "Current performance" section moved to the top of the document and
  linked to `BENCHMARKS.md`.

### Performance (RTX 3080, Qwen2.5-7B Q4_K_M, `-n 100 -p 32`, three-run mean)
- Standard: 23.3 → **28.0 tok/s** (+20%).
- TurboQuant 4-bit: 17.9 → **19.7 tok/s** (+10%).
- TurboQuant 4-bit + TriAttention: 18.2 → **19.4 tok/s** (+7%).
- PPL unchanged across all kernel swaps (bit-identical outputs).

### Fixed
- Pre-RoPE GPU path: incremental dst-offset bug and inverted attention
  scale (both latent). Slow-path quality now 4/5 on spot-check prompts.
- Gemma 2: missing BOS token during perplexity encode. Affects Gemma 2
  (PPL ~150k → expected range) and Llama (~4× lift). Qwen unchanged.

### Planned for 0.7.0 stable
- `dp4a` Q4_K matvec (close the upstream llama.cpp gap, target ≥70 tok/s).
- Sparse-V decode kernel (skip rows below softmax threshold).
- Asymmetric K/V bitwidth (4-bit K / 2-bit V default).
- Metal backend kickoff (kernel interface + skeleton).
- Multi-context benchmark table (1 K / 4 K / 16 K / 32 K).

## [0.6.0] — 2026-04-10

### Added
- TurboQuant + TriAttention default ON with `--turbo-quant`.
- Multi-arch CUDA build: sm_75, sm_80, sm_86, sm_89, sm_90 per-arch PTX,
  runtime selection.
- v2 common kernel helpers (`common_v2.cuh`): butterfly reduce,
  `float4` loads, streaming-cache hints, OnlineSoftmax.
- Auto-calibration on first use (256 samples, 2–3 s warmup).
- Three-way benchmark harness: Standard vs TQ vs TQ+TriAttention.
- Validated models: Qwen2.5 7B/0.5B, Llama 3.1 8B, Mistral 7B Instruct v0.3.

### Changed
- KIVI-style per-channel sigma is the default key quantizer path.
- Per-token mean removal enabled by default (+40% attention quality at 2-bit).

### Fixed
- 0.5B head_dim=64 dispatch bug.
- Llama 3 GQA GPU/CPU count desynchronization.
- Q5_0 dequant edge case.

## [0.5.0] — 2026-03

### Added
- Custom CUDA backend, candle framework dropped.
- `ComputeBackend` trait (`CpuBackend` / `CudaBackend`).
- CUDA Graph capture/replay for short-context decode.
- GPU prefill path with cuBLAS SGEMM after GPU-side dequant.
- `TqTensor` — own zero-copy tensor runtime over `Arc<CudaSlice<f32>>`.
- OpenAI-compatible `tq serve` (axum, SSE streaming).

### Performance
- Qwen2.5-7B Q4_K_M Standard: 1.3 → 17.8 tok/s across the release.

[Unreleased]: https://github.com/BHI-AI/tq-kv/compare/v0.7.0-alpha...HEAD
[0.7.0-alpha]: https://github.com/BHI-AI/tq-kv/compare/v0.6.0...v0.7.0-alpha
[0.6.0]: https://github.com/BHI-AI/tq-kv/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/BHI-AI/tq-kv/releases/tag/v0.5.0
