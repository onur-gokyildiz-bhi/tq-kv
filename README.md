# tq-kv

**Pure Rust TurboQuant KV cache compression. CUDA. AVX2 SIMD. C FFI. crates.io.**

[![Crates.io](https://img.shields.io/crates/v/tq-kv)](https://crates.io/crates/tq-kv)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Tests](https://img.shields.io/badge/tests-96%20passing-brightgreen)]()
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![no\_std](https://img.shields.io/badge/no__std-compatible-blue)]()

Implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with the **3-Fix framework** that enables aggressive key compression (4-bit, 7.5x) on GGUF quantized models -- where symmetric K compression produces catastrophic output.

Now with **Pre-RoPE key quantization** (34-59% less PPL gap), **KV Compaction** (up to 25x token reduction), **TriAttention** eviction for constant-memory KV cache, **per-token mean removal** (+40% attention quality at 2-bit), and **multi-arch CUDA** (Turing → Hopper).

<p align="center">
  <img src="docs/tq-demo.gif" alt="tq-engine Web UI demo — Qwen2.5 7B with TurboQuant 4-bit KV compression" width="720">
</p>

---

## Current performance

> RTX 3080 10GB, Qwen2.5-7B-Instruct Q4_K_M, own CUDA kernels, no external inference dependency. Three-run average, warmup discarded, TTFT separated.

| Mode | tok/s | TTFT | PPL (wikitext-2) | KV memory |
|:---|---:|---:|---:|:---|
| Standard | **28.0** | 0.19 s | 4.136 | grows linearly |
| TQ 4-bit | **19.7** | 0.10 s | 4.457 (+7.8%) | 3.8× smaller |
| TQ 4-bit + TriAttention | **19.4** | 0.11 s | 4.574 (+10.6%) | **constant** |

Full matrix (Llama 3.1 8B, Mistral 7B, multi-context, reproducible CLIs): [BENCHMARKS.md](BENCHMARKS.md).

## What's next (v0.7.0)

Sprint in progress. Targets:

- **dp4a Q4_K matvec** — close the Q4K instruction-efficiency gap against upstream llama.cpp (target ≥70 tok/s Qwen2 7B Standard on RTX 3080).
- **Sparse V** — skip value rows where softmax weight falls below threshold in the decode attention kernel.
- **Asymmetric K/V bitwidth** — 4-bit keys, 2-bit values by default; per-layer overrides.
- **Metal port kickoff** — shared kernel interface, Metal backend skeleton, MPS-only fallbacks for the ops without Metal kernels yet.

Full plan: see the `status:planned` nodes in the project knowledge graph.

---

## The Compound Error Problem

GGUF quantized models (Q4_K_M) already have weight quantization noise. Compressing KV cache on top introduces compound error through softmax. tq-kv solves this with a multi-stage pipeline:

| Config | Qwen 7B Q4_K_M PPL | Status |
|:-------|--------------------:|:-------|
| No compression | 4.136 | Baseline |
| **tq-kv 4-bit** | **4.457 (+8%)** | **Production** |
| **tq-kv 4-bit + TriAttention** | **4.574 (+11%)** | **Constant memory** |
| tq-kv 4-bit (no calibration) | ~15+ | Needs auto-calibrate |

Key innovations for GGUF compound error:
- **3-Fix framework**: sink tokens (FP16), current token (lossless POQ), cache reset
- **Pre-RoPE quantization**: position-independent per-channel stats, better codebook fit
- **Per-token mean removal**: softmax shift-invariance exploited, +40% attention quality at 2-bit
- **Auto-calibration**: zero-config — calibrates on first use (256 samples, 2-3 seconds)

---

## Measured Quality (Honest Numbers)

> All PPL measured with automated `tq perplexity` / `tq ablate`.

### GGUF Q4_K_M Models (compound error)

> Qwen 2.5 7B Q4_K_M, modern English, skip=4, sink=4

| Config | PPL | vs Baseline | Compression |
|:-------|----:|:-----------:|:-----------:|
| Baseline (no TQ) | 1.823 | -- | 1x |
| **Pre-RoPE 4-bit** | **1.890** | **+3.7%** | ~7.5x |
| Standard 4-bit | 1.925 | +5.6% | ~7.5x |
| TQ + Compact (500t/30%) | 2.227 | +22.2% | ~25x |
| Pre-RoPE + Compact (500t/30%) | 2.281 | +25.1% | ~25x |
| Full stack (Pre-RoPE+V4+Compact) | 2.84 | +55.8% | ~100x |

### Pre-RoPE Improvement Across Context Lengths

| Tokens | Baseline | Standard TQ (delta) | Pre-RoPE (delta) | Gap Reduction |
|:------:|:--------:|:-------------------:|:----------------:|:-------------:|
| 475 | 5.117 | 5.820 (+13.7%) | **5.403 (+5.6%)** | **59%** |
| 793 | 4.901 | 5.250 (+7.1%) | **5.125 (+4.6%)** | **35%** |
| 2106 | 1.823 | 1.925 (+5.6%) | **1.890 (+3.7%)** | **34%** |

### Value Compression is Nearly Free

| Value Config | Extra PPL vs K-only | Value Savings |
|:-------------|:-------------------:|:-------------:|
| V-fp16 (default) | -- | 1.0x |
| V-8bit (`TQ_VBITS=8`) | +0.2% | 2.0x |
| V-4bit (`TQ_VBITS=4`) | +1.3% | 3.2x |

### FP16 Models (no compound error)

> Qwen 2.5 0.5B FP16 safetensors, wikitext-2

| Bits | PPL | vs Baseline | Compression |
|:----:|----:|:-----------:|:-----------:|
| Baseline | 10.740 | -- | 1.0x |
| 4-bit | 11.967 | +11.4% | 7.5x |
| 2-bit | 27.696 | +157.9% | 14.2x |

### Compression Quality (per-layer)

| Model | Bits | Ratio | SNR (dB) | Cosine Sim |
|:------|:----:|------:|---------:|-----------:|
| Llama-3 8B | 2 | **14.2x** | 9.2 | 0.943 |
| Llama-3 8B | 4 | **7.5x** | 20.4 | 0.996 |
| Gemma 3 4B | 2 | **15.1x** | 9.2 | 0.943 |

### NIAH (Needle-In-A-Haystack)

| Bit Width | 10% | 25% | 50% | 75% | 90% |
|:---------:|:---:|:---:|:---:|:---:|:---:|
| 4-bit | PASS | PASS | PASS | PASS | PASS |
| 2-bit | PASS | PASS | PASS | PASS | PASS |

---

## Quick Start

### Install

```toml
[dependencies]
tq-kv = "0.6"
```

### CLI (tq-engine)

```bash
# Pull a model
tq pull qwen2:7b

# Chat with TurboQuant compression
tq chat qwen2:7b --turbo-quant

# Chat with Pre-RoPE (best quality)
TQ_PRE_ROPE=1 tq chat qwen2:7b --turbo-quant

# Start OpenAI-compatible API server
tq serve --model qwen2:7b --turbo-quant --port 11435

# Evaluate perplexity
tq perplexity --model qwen2:7b eval.txt --turbo-quant

# Calibrate (optimal codebook + rotation from real activations)
tq calibrate qwen2:7b --text calibration_data.txt

# Run ablation study
tq ablate qwen2:7b --file eval.txt --quick --output results.csv
```

### Library API

```rust
use tq_kv::*;

let config = TurboQuantConfig::balanced(); // 4-bit, 7.5x compression
let dim = 128;

// Batch compress
let compressed = compress_keys(&kv_data, dim, &config);
println!("Ratio: {:.1}x", compressed.compression_ratio());

// Fused attention -- no decompression, AVX2+FMA SIMD
let signs = hadamard::generate_signs(dim, config.rotation_seed);
let centroids = codebook::get_centroids(config.bits);
let rotated_q = pre_rotate_query_with_signs(&query, &signs);
let scores = fused_attention_scores(&rotated_q, &compressed, centroids, scale);

// KV Compaction -- reduce token count
let compacted = compaction::compact_head(&keys, &values, &queries,
    seq_len, n_queries, dim, target_size);
// compacted.keys, compacted.beta, compacted.values
```

---

## How It Works

```
Input KV vector (from GGUF Q4_K_M model)
    |
[0] Pre-RoPE capture (optional)            O(1)
    |  Compress BEFORE RoPE for position-independent stats
    |
[1] Channel bias subtraction (calibrated)   O(d)
    |  Remove weight-quantization artifacts
    |
[2] Per-token mean removal                   O(d)
    |  Softmax shift-invariant: attention ignores mean
    |  Frees codebook precision (+40% cosine sim @ 2-bit)
    |
[3] Randomized Hadamard Transform            O(d log d)
    |  Decorrelates outliers -> coordinates ~ Gaussian
    |
[4] Per-channel adaptive sigma (KIVI-style)  O(d)
    |  Per-dimension variance from calibration
    |  Lloyd-Max codebook + norm correction
    |
[5] SRHT QJL Error Correction (optional)     O(d log d)
    |  Structured Hadamard projection (+4.5 dB SNR)
    |
Output: packed indices + corrected norm + mean
        7.5x compression at 4-bit (keys only)
```

### 3-Fix for GGUF Models

```
Fix 1: Sink tokens (first 4) stay FP16     -> -81% attention error
Fix 2: Current token = lossless (POQ)       -> highest-impact position protected
Fix 3: Cache reset per conversation         -> prevents cross-contamination
```

### 6-Segment Attention

```
[sink FP16] [cold decayed] [compacted + beta] [hot compressed] [current FP16]
     |            |                |                   |                |
  Always       Temporal        Attention-           Per-head          POQ
  lossless     decay           matching             adaptive         lossless
  (Fix 1)                      reduction            bitwidth         (Fix 2)
                                    ^
                              TriAttention eviction scores all segments,
                              keeps top-B tokens, maintains fixed budget
```

### TriAttention: Constant-Memory KV Cache

Based on [TriAttention](https://arxiv.org/abs/2604.04921) (Mao et al., 2026). Pre-RoPE Q/K vectors concentrate around fixed centers, enabling cheap trigonometric importance scoring without full attention computation.

**Orthogonal to TurboQuant**: TriAttention decides *which* tokens to keep (eviction), TurboQuant decides *how* to compress them (quantization). Combined: fixed-size KV cache at any context length.

```bash
# TQ+TriAttention is ON by default with --turbo-quant (requires calibration)
tq chat qwen2:7b --turbo-quant

# Disable TriAttention (TQ-only mode)
TQ_TRIATTN=0 tq chat qwen2:7b --turbo-quant

# Custom budget
TQ_TRIATTN_BUDGET=256 tq chat qwen2:7b --turbo-quant
```

#### Memory Projection (Qwen2.5-7B, TQ 4-bit + TriAttention, budget=128)

| Context Length | FP16 KV | TQ 4-bit | TQ + TriAttn | Total Compression |
|:---------------|--------:|---------:|-------------:|------------------:|
| 4K tokens | 235 MB | 154 MB | **4.8 MB** | **49x** |
| 32K tokens | 1,879 MB | 1,233 MB | **4.8 MB** | **390x** |
| 128K tokens | 7,516 MB | 4,933 MB | **4.8 MB** | **1,560x** |

#### Speed vs Budget (RTX 3080, Qwen2.5-7B, 100 tokens)

| Budget | tok/s | vs TQ only | Fixed KV Memory |
|:------:|------:|:----------:|:---------------:|
| 512 | 15.3 | -6% | 19.3 MB |
| 256 | 15.3 | -6% | 9.6 MB |
| 128 | 13.8 | -15% | 4.8 MB |
| 64 | 7.9 | -46% | 2.4 MB |
| (none) | 16.2 | -- | grows linearly |

#### What Fits in 10GB VRAM (5GB available for KV cache)

| Method | Max Context Length |
|:-------|:------------------:|
| FP16 KV | ~87,000 tokens |
| TQ 4-bit | ~133,000 tokens |
| **TQ + TriAttention** | **unlimited** |

---

## Configuration Guide

### Recommended Configs by Use Case

| Use Case | Key Config | PPL Impact | Compression |
|:---------|:-----------|:----------:|:-----------:|
| **Best quality** | `TQ_PRE_ROPE=1` | +3.7% | ~7.5x |
| **Balanced** | `--turbo-quant` (default) | +5.6% | ~7.5x |
| **Maximum savings** | `TQ_PRE_ROPE=1 TQ_VBITS=4` | +5.0% | ~24x |
| **Long context** | `TQ_PRE_ROPE=1 TQ_COMPACT=1000 TQ_COMPACT_RATIO=30` | +30% | ~50x |
| **Extreme** | Pre-RoPE + V4 + Compact | +56% | ~100x |
| **Unlimited context** | `TQ_TRIATTN=1 TQ_TRIATTN_BUDGET=256` | TBD | **constant 10 MB** |

### Environment Variables

| Variable | Default | Description |
|:---------|:-------:|:------------|
| `TQ_SKIP` | 4 | Initial layers kept uncompressed (fp16 KV) |
| `TQ_PROTECT_LAST` | 0 | Final layers kept uncompressed (boundary protection) |
| `TQ_SINK` | 4 | Initial tokens preserved at fp16 (attention sinks) |
| `TQ_PRE_ROPE` | 0 | Pre-RoPE key quantization (1=enabled, best quality) |
| `TQ_COMPACT` | 0 | Compaction threshold (0=off, e.g. 500=compact when >500 hot tokens) |
| `TQ_COMPACT_RATIO` | 5 | Compaction target (% of original tokens to keep) |
| `TQ_VBITS` | 0 | Value compression bits (0=fp16, 4=4-bit, 8=8-bit) |
| `TQ_SPARSE_V` | 1e-6 | Skip V rows where softmax weight < threshold |
| `TQ_FUSED` | 0 | Fused attention from compressed indices (CPU only) |
| `TQ_DECAY` | off | Temporal decay (format: "age:bits" e.g. "512:2") |
| `TQ_LAYER_BITS` | -- | Per-layer bit width (format: "start-end:bits") |
| `TQ_HEAD_BITS` | -- | Per-head bit width (format: "0-3:4,4-7:2") |
| `TQ_GROUP` | 32 | Group size for per-group sigma |
| `TQ_BIAS_CORRECT` | 0 | Softmax bias correction (experimental) |
| `TQ_NO_CAL` | 0 | Disable calibration auto-loading |
| `TQ_TRIATTN` | **on** | TriAttention eviction (on by default with --turbo-quant, 0=disable) |
| `TQ_TRIATTN_BUDGET` | 2048 | Max KV tokens to retain (lower = more aggressive eviction, e.g. 128/256/512) |
| `TQ_TRIATTN_INTERVAL` | 128 | Eviction check interval in tokens |
| `TQ_CENTER_KEYS` | 1 | Per-token mean removal (softmax shift-invariant, +40% @ 2-bit) |
| `TQ_MAX_SEQ` | 2048 | Maximum KV cache sequence length |
| `TQ_NO_PER_CHANNEL` | 0 | Disable per-channel sigma (KIVI-style, for A/B testing) |
| `TQ_CUDA_ARCHES` | all | CUDA target arches (e.g. "86" for dev, "75,80,86,89,90" for release) |

---

## VRAM Savings

| Model | Context | FP16 KV | TQ 4-bit | TQ 2-bit | Savings |
|:------|:-------:|:-------:|:--------:|:--------:|:-------:|
| Qwen 2.5 7B | 4K | 256 MB | 34 MB | 18 MB | 7.5-14.2x |
| Qwen 2.5 72B | 4K | 640 MB | 85 MB | 45 MB | 7.5-14.2x |
| Llama 3.1 70B | 32K | 20 GB | 2.7 GB | 1.4 GB | 7.5-14.2x |

With KV Compaction: effective compression reaches 100-400x.
With TriAttention: **constant-memory KV cache** -- up to 1,560x at 128K context.

---

## SRHT QJL Performance (32K vectors, d=128, release)

| Metric | Dense QJL (paper) | SRHT QJL (ours) | No QJL |
|:-------|:-----------------:|:---------------:|:------:|
| Compress overhead | 29x | **1.45x** | 1.0x |
| SNR improvement | +1.2 dB | **+4.5 dB** | -- |
| Attention KL div. | -- | **2.9x lower** | -- |

---

## Full Product: tq-engine

tq-kv powers tq-engine -- "Rust's Ollama" with TurboQuant compression:

```bash
tq pull qwen2:7b          # download from HuggingFace
tq serve --turbo-quant     # OpenAI-compatible API (SSE streaming)
tq chat qwen2:7b           # terminal chat
```

Web UI at localhost:11435. Works with ChatBox and Open WebUI.

4 validated models: Qwen2.5 7B/0.5B, Llama 3.1 8B, Mistral 7B. Auto-detected from GGUF metadata.

### 3-Way Benchmark

```bash
tq bench qwen2:7b                    # Standard vs TQ vs TQ+TriAttention
tq perplexity -m qwen2:7b --compare eval.txt  # 3-way PPL comparison
scripts/ppl-check.sh                  # Regression CI (9 checks, tight thresholds)
```

---

## GPU Inference Performance

> RTX 3080 10GB, Qwen 2.5 7B Q4_K_M, own CUDA kernels (no candle dependency).
> Full reproducible matrix in [BENCHMARKS.md](BENCHMARKS.md).

| Mode | tok/s | TTFT | PPL | KV Memory |
|:-----|------:|-----:|----:|:---------:|
| **Standard** | **28.0** | 0.19s | 4.136 | grows linearly |
| **TQ 4-bit** | **19.7** | 0.10s | 4.457 (+7.8%) | 3.8x smaller |
| **TQ+TriAttention** | **19.4** | 0.11s | 4.574 (+10.6%) | **constant** |

4 validated models: Qwen2.5 7B/0.5B, Llama 3.1 8B, Mistral 7B. Auto-calibration on first use.

### Custom CUDA Kernel Stack (78 kernels, 16 files, 4.4K lines)

- **Multi-arch**: sm_75 (Turing) → sm_90 (Hopper), runtime GPU detect, per-arch PTX
- **Butterfly reduce**: `__shfl_xor_sync` broadcast to all threads (no broadcast hacks)
- **cp.async pipeline**: double-buffered qmatmul on Ampere+ (sm_80+), sync fallback on Turing
- **Fused layer kernels**: RmsNorm + Q4K QKV + bias, gateup + SiLU, down + residual
- **Q4K/Q6K/Q8_0 fused matvec**: `__ldg` read-only cache, dequant + dot in single kernel
- **Flash decode v2**: split-KV parallelism for long context (>256 tokens), online softmax
- **TQ fused attention**: compressed KV → attention score without decompression (moat kernel)
- **CUDA Graph replay**: short context (<256) as single GPU operation, auto-fallback to eager mode

### Optimization History

| Phase | tok/s | Key Change |
|:------|------:|:-----------|
| Baseline (candle) | 1.3 | candle framework |
| Custom CUDA kernels | 1.9 | Own matvec, RoPE, attention |
| DecodeScratch + fused | 10.1 | Zero-alloc decode, fused kernels |
| GPU prefill + CUDA Graph | 17.8 | Graph replay, Q6K lm_head |
| v2 kernels + multi-arch | 18.5 | `__ldg`, warp-reduce, cp.async, butterfly reduce |
| cp.async weight pipeline | 23.3 | MLP gateup + MLP down + qmatmul cp.async W-prefetch |
| **mrow ladder (v0.6, shipped 2026-04-14)** | **28.0** | cooperative-per-superblock gateup/down, RoPE Q+K fuse, bias-fused LM head |

---

## Benchmark

```bash
cargo run --release -p tq-kv --bin tq-kv-bench
```

Full results: [BENCHMARK.md](BENCHMARK.md)

## Paper

**Our work:**
- "TurboQuant on Quantized Models: Solving Compound Quantization Error with Pre-RoPE Compression and KV Compaction" -- BHI Research (2026)
- 3-Fix framework, SRHT QJL (115x speedup), Pre-RoPE quantization, KV Compaction, Adaptive QJL

**Original:**
- Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Mao, Lin, Huang et al. "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression." 2026. [arXiv:2604.04921](https://arxiv.org/abs/2604.04921)

## License

MIT OR Apache-2.0
