# tq-kv

**Pure Rust TurboQuant KV cache compression. CUDA. AVX2 SIMD. C FFI. crates.io.**

[![Crates.io](https://img.shields.io/crates/v/tq-kv)](https://crates.io/crates/tq-kv)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Tests](https://img.shields.io/badge/tests-86%20passing-brightgreen)]()
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![no\_std](https://img.shields.io/badge/no__std-compatible-blue)]()

Implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with the **3-Fix framework** that enables aggressive key compression (4-bit, 7.5x) on GGUF quantized models -- where symmetric K compression produces catastrophic output.

Now with **Pre-RoPE key quantization** (34-59% less PPL gap) and **KV Compaction** (up to 25x token reduction).

<p align="center">
  <img src="docs/tq-demo.gif" alt="tq-engine Web UI demo — Qwen2.5 7B with TurboQuant 4-bit KV compression" width="720">
</p>

---

## The Compound Error Problem

Most TurboQuant implementations assume FP16 model weights. In practice, everyone runs GGUF quantized models (Q4_K_M). When you compress **both** K and V symmetrically on GGUF, errors compound through softmax:

| Implementation | K compression | Qwen 7B Q4_K_M PPL | Status |
|:---------------|:-------------|--------------------:|:-------|
| No compression | -- | 5.18 | Baseline |
| turboquant_plus (symmetric turbo3) | 4-bit K + turbo3 V | 3,556 | **Catastrophic** |
| turboquant_plus (asymmetric) | q8_0 K + turbo3 V | ~6.64 | **Working (+2%)** |
| **tq-kv (3-Fix)** | **4-bit K** + V-fp16 | **6.07** | **Working (+17%)** |
| **tq-kv (Pre-RoPE)** | **4-bit K** + V-fp16 | **5.40** | **Working (+4.2%)** |

Two solutions exist for compound error on GGUF:
- **Asymmetric K/V** ([turboquant_plus](https://github.com/TheTom/turboquant_plus)): keep K at q8_0 (2x), compress V aggressively. Simple, effective, validated across 7 model families and 50+ testers.
- **3-Fix + Pre-RoPE** (tq-kv): compress K to 4-bit (7.5x) with sink preservation, POQ, and pre-RoPE quantization. More complex, but 3.75x more K compression.

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
    |  +34-59% PPL gap reduction
    |
[1] Randomized Hadamard Transform           O(d log d)
    |  Decorrelates outliers -> coordinates ~ Gaussian
    |  Verified: kurtosis 35.3 -> 3.3 (Gaussian=3.0)
    |
[2] Lloyd-Max Codebook + Adaptive Sigma      O(d)
    |  Per-vector sigma = ||x|| / sqrt(d)
    |  Norm correction: ||decompress|| matches ||original||
    |
[3] SRHT QJL Error Correction (optional)     O(d log d)
    |  Structured Hadamard projection (not dense random)
    |  Adaptive: auto-enables at long context
    |
Output: packed indices + corrected norm
        7.5x compression at 4-bit (keys only)
```

### 3-Fix for GGUF Models

```
Fix 1: Sink tokens (first 4) stay FP16     -> -81% attention error
Fix 2: Current token = lossless (POQ)       -> highest-impact position protected
Fix 3: Cache reset per conversation         -> prevents cross-contamination
```

### 5-Segment Attention

```
[sink FP16] [cold decayed] [compacted + beta] [hot compressed] [current FP16]
     |            |                |                   |                |
  Always       Temporal        Attention-           Per-head          POQ
  lossless     decay           matching             adaptive         lossless
  (Fix 1)                      reduction            bitwidth         (Fix 2)
```

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

---

## VRAM Savings

| Model | Context | FP16 KV | TQ 4-bit | TQ 2-bit | Savings |
|:------|:-------:|:-------:|:--------:|:--------:|:-------:|
| Qwen 2.5 7B | 4K | 256 MB | 34 MB | 18 MB | 7.5-14.2x |
| Qwen 2.5 72B | 4K | 640 MB | 85 MB | 45 MB | 7.5-14.2x |
| Llama 3.1 70B | 32K | 20 GB | 2.7 GB | 1.4 GB | 7.5-14.2x |

With KV Compaction: effective compression reaches 100-400x.

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

5 architectures: Qwen2, Llama, Mistral, Phi3, Gemma2 -- auto-detected from GGUF.

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

## License

MIT OR Apache-2.0
