# Changelog

All notable changes to **tq-engine** and the **tq-kv** library are documented
here. Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] — 2026-04-19

### Added — since the 0.7.0-alpha draft

- **dp4a Q4_K matvec family** (sprint, Apr 15–17). `__dp4a`-based INT8
  pipeline for QKV, gateup, down, LM head, and Wo. Std Qwen2 7B went
  from 28 → **68 tok/s** (+143% from the alpha baseline). llama.cpp
  gap closed from 4.0× to **1.83×**.
- **`dp4a_v2` / `dp4a_v3` variants** (Apr 17). `__vsub4` + byte-wise
  unaligned loads for Q6K. Cumulative +33% on the down path; **67.1
  tok/s** headline with autocalibration.
- **Autocalibration (Phase 1–4)**. Arch+hidden-dim-keyed JSON cache at
  `~/.cache/tq/autocalib-<arch>-h<hidden>-<hash>-sm<NN>.json`, applied
  at model load. `tq autotune` CLI runs an in-process sweep (99 s vs
  360 s subprocess) and writes the winning `TQ_*` vars. Median-of-N
  with `--reps` + `--validate-ppl 1%` guard. Commit `a29349d`.
- **`EventTimer` profiling primitive**. `cudaEvent`-based per-kernel
  timing replaces `Instant+synchronize`. Previous profiling overstated
  rope / res+attn by ~5×. Commit `507a331`.
- **Megakernel Phase 3 plumbing** (commits `d93e5cc` → `a13a360`, Apr
  17–18). Persistent-kernel-lite: one block per SM walks a 9-phase
  micro-ISA for an entire decoder layer in a single launch. All 6
  phase stubs (phase_res_add / phase_kv_append / phase_wo / phase_down
  / phase_rmsnorm_gateup / phase_attn) now have real device-function
  bodies. Shmem opt-in via `cuFuncSetAttribute` supports the ~95 KB
  Qwen2-dim footprint (commit `e32d341`). Smoke + Qwen2-dim plumbing
  tests gate the wiring work. **The kernel is NOT yet wired into main
  decode** — integration into `model.rs::forward` is the first item on
  the 0.8.0 slate.
- **Long-context benchmark infrastructure**:
  - `tq bench --prompt-file <path>` CLI flag (Windows argv length limit
    made long prompts impossible without this).
  - `scripts/bench-long-context.sh` — sweeps {512, 2 K, 8 K, 16 K}
    contexts, handles Std-OOM by retrying with `--tq-only`, emits a
    markdown table.
  - See commit `0711532` for the script's captured findings section.
- **Residual-alias preventive fix** in `forward_hidden_all` /
  `forward_last_hidden`. Matches the `layer_in.clone()` snapshot
  pattern from production `forward()` to avoid a latent
  copy-on-write hazard in the fused add-norm path. Commit `c260851`.
- **EAGLE speculative-decode scaffolding** (Sprint 1–3, Apr 17–18).
  Safetensors loader + shape validator, BF16→FP16→FP32 GPU upload,
  tree-attention CPU reference (10/10 parity tests) + GPU kernel
  `tree_decode_partial_f32` (<1 e-4 vs CPU), acceptance probe CLI.
  Draft forward is bit-correct against a numpy reference but
  acceptance sits at **~6.7 %** (vs ~70 % for a healthy EAGLE on the
  same checkpoint) — deeper draft-forward convention investigation
  parked pending a week-scale harness effort.

### Added — from the 0.7.0-alpha draft

- **mrow ladder** kernel family in `kernels/fused_mlp.cu` and
  `kernels/qmatmul.cu`: cooperative-per-superblock thread layout,
  256/256 threads active per block.
- `fused_addnorm_q4km_gateup_silu_cpasync_f32` — double-buffered
  shared memory (2×288 B) with `cp.async` prefetch of the next
  superblock's gate+up weights.
- `fused_q4km_down_residual_cpasync_f32` — cooperative MLP down
  kernel with `cp.async` W-prefetch (+14 % end-to-end).
- `q4km_matvec_wx_cpasync_f32` — `cp.async` double-buffer for both X
  and W in the decode matvec (+18–25 % kernel time on LM head / Wo).
- `fused_norm_q4km_qkv_bias_cpasync_f32` — opt-in QKV variant with
  `cp.async` W-prefetch. Default OFF (kernel −17 % but end-to-end
  flat due to sync cost).
- `rope_halved_qk_f32` / `rope_interleaved_qk_f32` — single-launch
  fused RoPE covering Q and K heads in one kernel.
- `BENCHMARKS.md` — reproducible end-to-end performance matrix with
  exact CLI invocations, methodology, and raw-log policy.
- `CHANGELOG.md` — this file.

### Changed
- Default kernel dispatch flipped to the dp4a winners on Qwen2 7B:
  `TQ_DOWN=dp4a_v3`, `TQ_GATEUP=dp4a`, `TQ_Q4KM=dp4a_v2`,
  `TQ_Q6K=dp4a_v2`, `TQ_QKV=dp4a`. All vars remain user-overridable.
- **TriAttention V3** hybrid threshold: switches from brute-force
  scoring to the fused compressed-KV path once budget × ctx exceeds
  the measured crossover.
- README "Current performance" section moved to the top of the
  document and linked to `BENCHMARKS.md`.

### Performance (RTX 3080, Qwen2 7B Q4_K_M, `-n 100 -p 32`, three-run mean)

End-of-cycle measurement on 2026-04-19:

- Standard: **63.6 tok/s** (the 68 tok/s figure shown at Apr 17 close
  was a cherry-picked peak; three-run mean in the same commit range
  reproduces at 63.6 ± 0.5 — the deflated number is the one to trust).
- TurboQuant 4-bit: **39.3 tok/s**.
- TurboQuant 4-bit + TriAttention: **37.6 tok/s**.
- PPL (Qwen2 7B, `/tmp/ppl_bench.txt`): Standard **11.508**,
  TQ 4-bit **12.473**. Both well inside the regression thresholds
  (12.66 / 13.72) that the pre-push hook enforces.

Relative to the 0.7.0-alpha numbers (23.3 / 17.9 / 18.2 tok/s) this is
**+173 % / +120 % / +107 %**. The llama.cpp head-to-head gap closed
from ~4.0× to ~1.83× over the same window.

### Fixed
- Pre-RoPE GPU path: incremental dst-offset bug and inverted attention
  scale (both latent). Slow-path quality now 4/5 on spot-check prompts.
- Gemma 2: missing BOS token during perplexity encode. Affects Gemma 2
  (PPL ~150 k → expected range) and Llama (~4× lift). Qwen unchanged.
- Residual-alias hazard in `forward_hidden_all` / `forward_last_hidden`
  (preventive — did not manifest on real prompts but matches
  production `forward()`'s snapshot pattern).

### Honest refutations (v0.7.0 cycle)

- **"TurboQuant wins at long context" is NOT true on 10 GB cards
  today.** The bench sweep shows both Standard and TQ hit OOM at 8 K
  ctx on RTX 3080 10 GB — the bottleneck is the prefill attention
  N×N score tensor, not KV storage. A flash-attention prefill port
  (we only have flash_decode) is the real long-context unlock. The
  sweep script remains in-repo as a forcing function for that work.

### Planned-for-0.7.0 status
- [x] `dp4a` Q4_K matvec family → shipped, Std 28 → 68 tok/s.
- [x] Sparse-V decode kernel → shipped.
- [x] Asymmetric K/V bit-width → shipped.
- [ ] Metal backend kickoff → deferred to 0.8.0 (needs a collaborator
  with Apple Silicon hardware).
- [~] Multi-context benchmark table → infra shipped
  (`scripts/bench-long-context.sh`); numbers only go to 2 K because
  prefill attention OOMs past that on 10 GB.

### Planned for 0.8.0
- **Megakernel wiring**: route `TQ_MEGAKERNEL=1` through
  `model.rs::forward` for single-token decode. Plumbing is done; the
  wiring work itself (weight extraction, KV-cache interop, CUDA graph
  capture disable, PPL gate) is the v0.8.0 priority.
- **Flash-attention PREFILL port** — close the long-context OOM cliff.
  Unlocks ≥ 4 K contexts on 10 GB cards and the TurboQuant value
  proposition at real context lengths.
- **EAGLE acceptance debug** — torch reference harness covering
  non-zero inputs (numpy ref only tests zero-input smoke). Week-scale
  effort; only resume once flash-prefill is in.
- **Metal backend kickoff** if a collaborator picks it up.

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
