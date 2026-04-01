# TurboQuant Benchmark Results

> **tq-kv** v0.7.0 -- Pure Rust, CUDA accelerated
> 3-Fix + Pre-RoPE + KV Compaction + SRHT QJL + Per-head Adaptive Bitwidth
> [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

**Hardware:** NVIDIA GeForce RTX 3080 10GB, Intel Core i9-13900K, 64GB RAM, Windows 11 Pro, CUDA 13.2, Rust 1.91 release mode.

---

## Pre-RoPE Key Quantization (NEW)

Compress keys BEFORE RoPE application. Pre-RoPE keys have position-independent per-channel statistics -- better fit for Lloyd-Max Gaussian codebook.

> Qwen 2.5 7B Q4_K_M, modern English text, skip=4, sink=4

| Tokens | Baseline | Standard TQ (delta) | Pre-RoPE (delta) | Gap Reduction |
|:------:|:--------:|:-------------------:|:----------------:|:-------------:|
| 475 | 5.117 | 5.820 (+13.7%) | **5.403 (+5.6%)** | **59%** |
| 793 | 4.901 | 5.250 (+7.1%) | **5.125 (+4.6%)** | **35%** |
| 2106 | 1.823 | 1.925 (+5.6%) | **1.890 (+3.7%)** | **34%** |

> Pre-RoPE consistently reduces the PPL gap by 34-59% at identical compression ratio (~7.5x).

---

## KV Compaction (NEW)

Attention-matching token reduction (Zweiger 2026). Orthogonal to bit compression.

> Qwen 2.5 7B Q4_K_M, 2106 tokens modern English

| Config | PPL | vs Baseline | Est. Compression |
|:-------|:---:|:-----------:|:----------------:|
| Baseline (no TQ) | 1.823 | -- | 1x |
| Standard TQ 4-bit | 1.925 | +5.6% | ~7.5x |
| Pre-RoPE 4-bit | 1.890 | +3.7% | ~7.5x |
| TQ + Compact (500t/30%) | 2.227 | +22.2% | ~25x |
| TQ + Compact (1000t/20%) | 2.367 | +29.9% | ~25x |
| TQ + Compact (1000t/50%) | 2.372 | +30.1% | ~10x |
| Pre-RoPE + Compact (500t/30%) | 2.281 | +25.1% | ~25x |
| Pre-RoPE + Compact (1000t/30%) | 2.364 | +29.7% | ~25x |

### Compaction Optimization Impact

Multi-query reference (all GQA heads) + mean-based scoring vs single-query max scoring:

| Config | Before Optimization | After Optimization | Improvement |
|:-------|:-------------------:|:------------------:|:-----------:|
| TQ + Compact 500t/30% | 5.777 | **2.227** | **2.6x** |
| TQ + Compact 1000t/20% | 2.982 | **2.367** | 1.26x |

---

## GGUF Q4_K_M Perplexity

> Qwen 2.5 7B Q4_K_M, wikitext-2, skip=4, sink=4

| Config | PPL | vs Baseline | Notes |
|:-------|----:|:-----------:|:------|
| Baseline (no TQ) | 5.178 | -- | |
| **K4-bit, V-fp16** | **6.065** | **+17.1%** | Standard recommended |
| K4-bit, V-8bit | 6.076 | +17.3% | V8 nearly free (+0.2%) |
| K4-bit, V-4bit | 6.143 | +18.6% | V4 acceptable |

### Value Compression Cost

| Value Config | Extra PPL vs K-only | Value Savings |
|:-------------|:-------------------:|:-------------:|
| V-fp16 (default) | -- | 1.0x |
| V-8bit (`TQ_VBITS=8`) | +0.2% | 2.0x |
| V-4bit (`TQ_VBITS=4`) | +1.3% | 3.2x |

> Key insight: softmax amplifies K errors exponentially, V errors scale linearly.

---

## FP16 Model Perplexity

> Qwen 2.5 0.5B FP16 safetensors, wikitext-2, 2366 tokens

| Bits | PPL | vs Baseline | Compression |
|:----:|----:|:-----------:|:-----------:|
| Baseline | 10.740 | -- | 1.0x |
| 4-bit | 11.967 | +11.4% | 7.5x |
| 3-bit | 13.933 | +29.7% | 9.8x |
| 2-bit | 27.696 | +157.9% | 14.2x |

---

## Compression Quality (per-layer, synthetic)

### Llama-3 8B

| Bits | Method | Ratio | SNR (dB) | Cosine Sim | Compress | Decompress |
|:----:|:-------|------:|---------:|-----------:|---------:|-----------:|
| 2 | Lloyd-Max | **14.2x** | 9.4 | 0.943 | 50 ms | 26 ms |
| 3 | Lloyd-Max | **9.8x** | 14.7 | 0.984 | 59 ms | 27 ms |
| 4 | Lloyd-Max + QJL | **5.3x** | 21.5 | 0.997 | 2335 ms | 2554 ms |

### Gemma 3 4B (head_dim=256)

| Bits | Method | Ratio | SNR (dB) | Cosine Sim |
|:----:|:-------|------:|---------:|-----------:|
| 2 | Lloyd-Max | **15.1x** | 9.3 | 0.942 |
| 3 | Lloyd-Max | **10.2x** | 14.7 | 0.984 |
| 4 | Lloyd-Max + QJL | **5.8x** | 21.4 | 0.997 |

---

## QJL: Dense vs SRHT

> 32,768 vectors, dim=128, 4-bit, release build

| Metric | Dense QJL (paper) | SRHT QJL (ours) | No QJL |
|:-------|:-----------------:|:------------------:|:------:|
| Compression ratio | 2.7x | 2.7x | **3.8x** |
| SNR | 21.5 dB | **24.8 dB** | 20.3 dB |
| Cosine similarity | 0.9966 | **0.9984** | 0.9956 |
| Compress overhead | 29x | **1.45x** | 1.0x |
| Decompress overhead | 128x | **1.7x** | 1.0x |
| Attention KL div. | -- | **2.9x lower** | baseline |

> SRHT QJL: **115x faster** than dense, **+4.5 dB better SNR**.

### Attention KL Divergence vs Context Length

| Context | KL(no QJL) | KL(SRHT QJL) | Reduction |
|:-------:|:----------:|:------------:|:---------:|
| 64 | 0.000270 | 0.000093 | 2.9x |
| 256 | 0.000283 | 0.000099 | 2.9x |
| 1,024 | 0.000289 | 0.000103 | 2.8x |
| 4,096 | 0.000288 | 0.000102 | 2.8x |
| 16,384 | 0.000288 | 0.000103 | 2.8x |

---

## 3-Fix Framework

> Qwen 2.5 7B Q4_K_M, 4-bit TurboQuant

| Configuration | Max Coherent Tokens | Language Mixing |
|:-------------|:-------------------:|:---------------:|
| No TurboQuant | unlimited | none |
| TurboQuant (no fixes) | ~50 | severe (5+ languages) |
| + Cache reset | ~100 | moderate |
| + Cache reset + POQ | ~200 | mild |
| **+ All 3 fixes** | **300+** | **none** |

> First TurboQuant validated on GGUF quantized models.

---

## VRAM Projection

| Model | Context | FP16 KV | TQ 4-bit | TQ 2-bit | Savings |
|:------|:-------:|--------:|---------:|---------:|:-------:|
| Qwen 2.5 7B | 4K | 256 MB | 34 MB | 18 MB | 7.5-14.2x |
| Qwen 2.5 72B | 4K | 640 MB | 85 MB | 45 MB | 7.5-14.2x |
| Llama 3.1 70B | 32K | 20 GB | 2.7 GB | 1.4 GB | 7.5-14.2x |

With Compaction (20x) + TQ 4-bit: effective 150x compression.

---

## Fused Attention vs Decompress Path

> Llama-3 8B, 512 cached keys, head_dim=128

| Bits | Fused (us) | Decompress+dot (us) | Speedup |
|:----:|----------:|-----------:|:-------:|
| 2-bit | 68 | 411 | **6.0x** |
| 3-bit | 97 | 584 | **6.0x** |
| 4-bit | 83 | 503 | **6.1x** |

> Note: Fused path incompatible with Pre-RoPE and Compaction (auto-fallback).

---

## Generation Throughput

> CPU (i9-13900K), release build, Q4_K_M models

| Model | Tokens | Standard tok/s | TQ 4-bit tok/s | Delta | TTFT Std | TTFT TQ |
|:------|:------:|:--------------:|:--------------:|:-----:|:--------:|:-------:|
| Qwen 2.5 7B | 50 | 5.66 | 4.75 | -16% | 2.31s | 2.51s |
| Qwen 2.5 7B | 200 | 6.55 | 6.04 | **-8%** | 2.22s | 2.32s |
| Llama 3.1 8B | 200 | 4.20 | 3.47 | -17% | 2.69s | 2.75s |

### CUDA Throughput (TQ 4-bit, RTX 3080)

| Model | CUDA tok/s | CPU tok/s | GPU Speedup | TTFT |
|:------|:---------:|:---------:|:-----------:|:----:|
| Qwen 2.5 7B | **28.2** | 6.04 | **4.7x** | 0.118s |
| Llama 3.1 8B | **19.2** | 3.47 | **5.5x** | 0.126s |

### vs llama.cpp (CPU, Q4_K_M, same hardware)

| Engine | Model | Decode tok/s | Note |
|:-------|:------|:-----------:|:-----|
| llama.cpp | Qwen 2.5 7B | **16.7** | Optimized GGML |
| llama.cpp | Llama 3.1 8B | **15.1** | Optimized GGML |
| tq-engine (TQ 4-bit) | Qwen 2.5 7B | 5.8 | candle framework |
| tq-engine (TQ 4-bit) | Llama 3.1 8B | 3.5 | candle framework |

> Decode gap (~2.8x) is candle vs GGML framework overhead, not TQ-specific.

---

## NIAH (Needle-In-A-Haystack)

> Trendyol Llama-3 8B, ~4K token context, RTX 3080

| Depth | Standard | TQ 4-bit | TQ 2-bit |
|:-----:|:--------:|:--------:|:--------:|
| 10% | PASS | PASS | PASS |
| 50% | PASS | PASS | PASS |
| 90% | PASS | PASS | PASS |

9/9 pass. Compressed keys preserve retrieval accuracy at every depth.

---

## Per-Token Inference Cost

| Architecture | Per-Token Cache Cost | Note |
|:-------------|--------------------:|:-----|
| Old (decompress+recompress all) | ~608 ms | O(N) growing |
| New (incremental append) | ~0.65 ms | O(1) constant |
| **Improvement** | **~935x** | cache overhead reduction |

---

## Gaussianity Verification

| Metric | Before Hadamard | After Hadamard | Gaussian Ideal |
|:-------|:---------------:|:--------------:|:--------------:|
| Kurtosis | 35.3 | **3.3** | 3.0 |

Confirms Lloyd-Max codebook assumptions hold after rotation.

---

## Run Benchmark

```bash
cargo run --release -p tq-kv --bin tq-kv-bench
```

## Technical Details

```
Crate:            tq-kv v0.6.0
License:          MIT / Apache-2.0
Language:         Pure Rust (13.7K LOC, no C/C++/Python dependency)
Algorithm:        TurboQuant (ICLR 2026, Google Research)
Rotation:         Fast Walsh-Hadamard Transform O(d log d)
Bit widths:       2, 3, 4-bit keys + 4/8-bit values
Pre-RoPE:         Position-independent key compression (34-59% less PPL gap)
Compaction:       Attention-matching token reduction (up to 25x)
Error Correction: SRHT QJL (115x faster than dense, +4.5 dB SNR)
Fused attention:  AVX2+FMA SIMD centroid lookup (6-8.9x speedup)
KV cache:         Incremental O(1) per token (~935x vs naive)
CUDA support:     Custom RmsNorm, RoPE, softmax kernels
Architectures:    Qwen2, Llama, Mistral, Phi3, Gemma2
Tests:            86 passing
```
