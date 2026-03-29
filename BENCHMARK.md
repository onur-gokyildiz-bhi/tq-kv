# TurboQuant Benchmark Results

> **tq-kv** v0.5.0 — Pure Rust, zero C/C++ dependency, CUDA accelerated
> First TurboQuant on GGUF quantized models — 3-Fix + SRHT QJL + Norm Correction + Adaptive QJL
> [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

**Hardware:** NVIDIA GeForce RTX 3080 10GB, Intel Core i9-13900K, 64GB RAM, Windows 11 Pro, CUDA 13.2, Rust 1.91 release mode.

---

## Algorithm

```
Input KV vector (fp16/fp32)
    |
    v
[1] Randomized Hadamard Transform         O(d log d)
    |  Fast Walsh-Hadamard + random sign flip
    |  Eliminates outliers, coordinates → Gaussian(0, ||x||/√d)
    |
    v
[2] Adaptive Sigma Lloyd-Max Quantization  O(d)
    |  Per-vector sigma = ||x|| / √d (NOT fixed 1/√d)
    |  Pre-computed optimal centroids for Gaussian
    |  Bit-packed storage: 2-bit=4 per byte, 4-bit=2 per byte
    |
    v
[3] QJL 1-bit Error Correction (optional)  O(d²)
    |  Rademacher random projection on residual
    |  Stores only sign bits — 1 extra bit per dimension
    |
    v
Output: packed_indices (uint8) + norms (fp32)
        Compression: up to 15.1x vs fp16
```

---

## Llama-3 8B — Trendyol Turkish LLM

> 32 layers | 8 KV heads | head_dim=128 | context=4096

### Compression Quality (per layer)

| Bits | Method | Ratio | SNR (dB) | Cosine Sim | Compress | Decompress |
|:----:|:-------|------:|---------:|-----------:|---------:|-----------:|
| 2 | **Lloyd-Max Codebook** | **14.2x** | 9.4 | 0.943 | 50 ms | 26 ms |
| 3 | **Lloyd-Max Codebook** | **9.8x** | 14.7 | 0.984 | 59 ms | 27 ms |
| 3 | PolarQuant + QJL | 3.0x | 14.6 | 0.984 | 2042 ms | 2872 ms |
| 4 | **Lloyd-Max + QJL** | **5.3x** | 21.5 | 0.997 | 2335 ms | 2554 ms |
| 4 | PolarQuant + QJL | 3.0x | 21.2 | 0.996 | 2051 ms | 2544 ms |

### VRAM Projection (full model, 4096 context, Keys only)

| Config | KV Cache fp16 | TurboQuant | Saved | Ratio |
|:------:|--------------:|-----------:|------:|------:|
| 2-bit | 256 MB | 18 MB | **238 MB** | 14.2x |
| 3-bit | 256 MB | 26 MB | **230 MB** | 9.8x |
| 4-bit | 256 MB | 48 MB | **208 MB** | 5.3x |

---

## Qwen2.5 72B Instruct

> 80 layers | 8 KV heads | head_dim=128 | context=4096

### Compression Quality

| Bits | Method | Ratio | SNR (dB) | Cosine Sim |
|:----:|:-------|------:|---------:|-----------:|
| 2 | Lloyd-Max Codebook | **14.2x** | 9.4 | 0.943 |
| 3 | Lloyd-Max Codebook | **9.8x** | 14.7 | 0.984 |
| 4 | Lloyd-Max + QJL | **5.3x** | 21.5 | 0.997 |

### VRAM Projection (full model, Keys only)

| Config | KV fp16 | TurboQuant | Saved | Ratio |
|:------:|--------:|-----------:|------:|------:|
| 2-bit | 640 MB | **45 MB** | **595 MB** | 14.2x |
| 3-bit | 640 MB | 65 MB | **575 MB** | 9.8x |
| 4-bit | 640 MB | 120 MB | **520 MB** | 5.3x |

> **595 MB saved** on Qwen 72B = 5-6 extra transformer layers fit on a GTX 3080 (10GB)

---

## Gemma 3 4B

> 26 layers | 4 KV heads | head_dim=256 | context=4096

| Bits | Method | Ratio | SNR (dB) | Cosine Sim | Compress | Decompress |
|:----:|:-------|------:|---------:|-----------:|---------:|-----------:|
| 2 | Lloyd-Max Codebook | **15.1x** | 9.3 | 0.942 | 56 ms | 25 ms |
| 3 | Lloyd-Max Codebook | **10.2x** | 14.7 | 0.984 | 65 ms | 26 ms |
| 4 | Lloyd-Max + QJL | **5.8x** | 21.4 | 0.997 | 4432 ms | 5817 ms |

---

## Method Comparison: Lloyd-Max vs PolarQuant

| Metric | Lloyd-Max 3-bit | PolarQuant 3-bit | Improvement |
|:-------|:---------------:|:----------------:|:-----------:|
| Compression ratio | **9.8x** | 3.0x | **+227%** |
| SNR | 14.7 dB | 14.6 dB | +0.7% |
| Cosine similarity | 0.984 | 0.984 | same |
| Compress speed | **59 ms** | 2042 ms | **35x faster** |
| Decompress speed | **27 ms** | 2872 ms | **106x faster** |

> Lloyd-Max without QJL is **35-106x faster** than PolarQuant+QJL at the same quality,
> with **3.3x better compression ratio**. QJL overhead dominates at low bit widths.

### QJL: Dense vs SRHT (v0.5.0)

> 32,768 vectors, dim=128, 4-bit, release build

| Metric | Dense QJL (paper) | SRHT QJL (v0.5.0) | No QJL |
|:-------|:-----------------:|:------------------:|:------:|
| Compression ratio | 2.7x | 2.7x | **3.8x** |
| SNR | 21.5 dB | **24.8 dB** | 20.3 dB |
| Cosine similarity | 0.9966 | **0.9984** | 0.9956 |
| Compress overhead | 29x | **1.45x** | 1.0x |
| Decompress overhead | 128x | **1.7x** | 1.0x |
| Attention KL divergence | — | **2.9x lower** | baseline |

> SRHT QJL: **115x faster** than dense, **+4.5 dB better SNR**. Structured Hadamard projection has lower variance than random Rademacher.
> Dense QJL hurts quality per 5 independent groups (ikawrakow, spiritbuun, scos-lab, Arclabs001, paper ablation).
> SRHT QJL improves attention KL divergence 2.9x at all context lengths (synthetic data; real-model validation pending).
> Default: QJL OFF. Adaptive mode available: `QjlMode::Adaptive { threshold: 4096 }`.

### Attention KL Divergence vs Context Length

| Context | KL(no QJL) | KL(SRHT QJL) | Reduction |
|:-------:|:----------:|:------------:|:---------:|
| 64 | 0.000270 | 0.000093 | 2.9x |
| 256 | 0.000283 | 0.000099 | 2.9x |
| 1,024 | 0.000289 | 0.000103 | 2.8x |
| 4,096 | 0.000288 | 0.000102 | 2.8x |
| 16,384 | 0.000288 | 0.000103 | 2.8x |

### 3-Fix Framework (v0.5.0 — GGUF Compound Error Fix)

> Qwen 2.5 7B Q4_K_M, 4-bit TurboQuant, CPU

| Configuration | Max Coherent Tokens | Language Mixing |
|:-------------|:-------------------:|:---------------:|
| No TurboQuant | unlimited | none |
| TurboQuant (no fixes) | ~50 | severe (5+ languages) |
| + Cache reset | ~100 | moderate |
| + Cache reset + POQ | ~200 | mild |
| **+ All 3 fixes** | **300+** | **none** |

> First TurboQuant implementation validated on GGUF quantized models.

### Norm Correction (v0.5.0)

Stores `corrected_norm = norm^2 / ||reconstruction||` so decompressed vector norm matches original.
Zero decode cost. Applied to all compression paths (batch + single key).

### Gaussianity Verification (v0.5.0)

| Metric | Before Hadamard | After Hadamard | Gaussian Ideal |
|:-------|:---------------:|:--------------:|:--------------:|
| Kurtosis | 35.3 | **3.3** | 3.0 |

Confirms Lloyd-Max codebook assumptions hold after rotation.

### Perplexity — Qwen 2.5 7B Instruct Q4_K_M (v0.5.0+)

> wikitext-2, 2328 tokens, CUDA (RTX 3080)

| Mode | PPL | vs Baseline |
|:-----|----:|:----------:|
| Baseline (no TQ) | 5.088 | — |
| TQ 4-bit, SINK=4 | **6.042** | **+18.7%** |
| TQ 3-bit, SINK=4 | 6.322 | +24.2% |
| TQ 2-bit, SINK=4 | 6.816 | +33.9% |
| TQ 4-bit, SINK=0 | 6.831 | +34.3% |
| TQ 2-bit, SINK=0 | 7.004 | +37.6% |

> Sink token preservation (Fix 1 of 3-Fix) reduces PPL delta by ~16 points.
> Higher delta vs Llama-3 FP16 (+0.8%) due to compound quantization error (Q4_K_M × TQ).

---

## Fused Attention vs Decompress Path

> Llama-3 8B, 512 cached keys, head_dim=128

| Bits | Fused (us) | Decompress+dot (us) | Speedup | Score Divergence |
|:----:|----------:|-----------:|:-------:|:----------------:|
| 2-bit | 68 | 411 | **6.0x** | 0.000001 |
| 3-bit | 97 | 584 | **6.0x** | 0.000001 |
| 4-bit | 83 | 503 | **6.1x** | 0.000001 |

Fused attention computes attention scores directly from compressed key indices via centroid table lookup. No key decompression needed. Pre-rotate query once, then per-key cost is a dot product with scaled centroids.

---

## Generation Throughput (v0.5.0+)

> CPU (i9-13900K), release build, Q4_K_M models

| Model | Tokens | Standard tok/s | TQ 4-bit tok/s | Delta | TTFT Std | TTFT TQ |
|:------|:------:|:--------------:|:--------------:|:-----:|:--------:|:-------:|
| Qwen 2.5 7B | 50 | 5.66 | 4.75 | -16% | 2.31s | 2.51s |
| Qwen 2.5 7B | 200 | 6.55 | 6.04 | **-8%** | 2.22s | 2.32s |
| Llama 3.1 8B | 200 | 4.20 | 3.47 | -17% | 2.69s | 2.75s |

> TQ overhead decreases with longer generation (amortized compression cost).
> At 200 tokens, Qwen gap is only 8%. Memory bandwidth savings offset compression cost.

---

## Qwen2.5 72B — Real Model Validation

> 80 layers, 8 KV heads (GQA-8), 64 query heads, head_dim=128, Q4_K_M

| Mode | Output |
|:-----|:-------|
| Standard (no TQ) | "Turkiye'nin baskenti Ankara'dir." |
| TurboQuant 4-bit | "Turkiye'nin baskenti Ankara'dir." |

Qwen2 requires attention Q/K/V bias (attn_q.bias, attn_k.bias, attn_v.bias) and context_length from GGUF metadata. Missing bias was the root cause of initial failures — not compression quality.

---

## Per-Token Inference Cost

Old architecture: decompress ALL keys + recompress ALL keys per token = O(N) growing cost.
New architecture: compress 1 new key + append + fused attention = O(1) cache + O(N) attention.

> At 4096 context, 2-bit:

| Architecture | Per-Token Cache Cost | Note |
|:-------------|--------------------:|:-----|
| Old (decompress+recompress all) | ~608 ms | O(N) growing |
| New (incremental append) | ~0.65 ms | O(1) constant |
| **Improvement** | **~935x** | cache overhead reduction |

---

## CUDA vs CPU

> Trendyol Llama-3 8B, TurboQuant 2-bit, 150 token generation

| Backend | Time | Speedup |
|:--------|-----:|:-------:|
| CUDA GPU | **7.4 s** | **3.2x** |
| CPU | 23.4 s | 1.0x |

> CUDA support required custom implementations of RmsNorm, RoPE, and softmax because candle's
> quantized_nn ops lack CUDA kernels.

## Needle-In-A-Haystack (NIAH)

> Trendyol Llama-3 8B, ~4K token context, RTX 3080 10GB

Needle: "The secret code for the treasure vault is GOLDEN-PHOENIX-7742."
Question: "What is the secret code for the treasure vault?"

| Depth | Standard | TurboQuant 4-bit | TurboQuant 2-bit |
|:-----:|:--------:|:----------------:|:----------------:|
| 10% | PASS | PASS | PASS |
| 50% | PASS | PASS | PASS |
| 90% | PASS | PASS | PASS |

9/9 pass. Compressed keys preserve retrieval accuracy at every depth.

---

## Context Scaling

> Trendyol Llama-3 8B, wikitext-2, CPU inference

PPL at different context lengths:

| Context | Standard | 4-bit | vs Baseline | 2-bit | vs Baseline |
|:-------:|---------:|------:|:----------:|------:|:----------:|
| 256 | 13.863 | 14.076 | +1.5% | 15.906 | +14.7% |
| 512 | 11.368 | 11.585 | +1.9% | 13.364 | +17.6% |
| 1024 | 7.441 | 7.474 | +0.4% | 9.041 | +21.5% |
| 2048 | 9.651 | 9.718 | +0.7% | 12.946 | +34.1% |

4-bit PPL remains flat across all context lengths (+0.4% to +1.9%).

---

## Codebook Quality (isolated, ideal Gaussian data)

| Bits | Centroids | MSE | SNR (dB) | Theoretical Ratio |
|:----:|:---------:|:---:|:--------:|:-----------------:|
| 2 | 4 | 0.000700 | 9.8 | 14.2x |
| 3 | 8 | 0.000237 | 14.5 | 10.7x |
| 4 | 16 | 0.000065 | 20.1 | 7.5x |

---

## Key Innovations

### 1. Adaptive Sigma (our contribution)

The paper assumes fixed sigma = 1/sqrt(d). We use **per-vector adaptive sigma**:

```
sigma_i = ||x_i|| / sqrt(d)
```

This matches the actual coordinate variance after Hadamard rotation, regardless of input scale. Standard approach uses fixed sigma which causes quality loss on vectors with varying norms.

### 2. Pre-Rotated Query Trick

```
dot(q, k) = dot(q, R^T * R * k)     // R is Hadamard rotation
           = dot(R*q, R*k)            // R is orthogonal
           = dot(R*q, centroids[idx]) // R*k ≈ codebook lookup
```

Rotate query once, then each key is just a table lookup. No decompression.

### 3. Selective QJL (removed from defaults)

- **All bit widths:** QJL disabled by default — at 4-bit, adds only +1.2 dB SNR but costs 29x compress and 128x decompress
- QJL still available as opt-in for maximum quality when latency is not a concern

---

## Technical Details

```
Crate:            tq-kv v0.4.0
License:          MIT / Apache-2.0
Language:         Pure Rust (no C/C++/Python)
Algorithm:        TurboQuant (ICLR 2026, Google Research)
Paper:            arxiv.org/abs/2504.19874
Quantization:     Lloyd-Max optimal codebook + adaptive sigma
Error Correction: QJL 1-bit (opt-in, removed from defaults)
Rotation:         Fast Walsh-Hadamard Transform O(d log d)
Bit widths:       2, 3, 4-bit
Bit packing:      yes (2-bit: 4/byte, 3-bit: 8/3bytes, 4-bit: 2/byte)
Pre-rotated Q:    yes (fused attention ready)
Fused attention:  yes (6x speedup vs decompress path)
KV cache:         incremental (O(1) per token, ~935x vs naive)
CUDA support:     yes (custom RmsNorm, RoPE, softmax kernels)
Platforms:        CPU (any) + CUDA GPU
Tested with:      Trendyol Llama-3 8B real model inference
```

---

## Run Benchmark

```bash
cargo run --release -p tq-kv --bin tq-kv-bench
```

## Use in Your Project

```toml
[dependencies]
tq-kv = "0.4"
```

```rust
use tq_kv::{TurboQuantConfig, compress_keys, decompress_keys, pre_rotate_query};

let config = TurboQuantConfig::extreme(); // 2-bit, 15x compression
let compressed = compress_keys(&kv_data, 128, &config);
let restored = decompress_keys(&compressed, &config);

// Fused attention: skip decompression
let rq = pre_rotate_query(&query, config.rotation_seed);
let score = tq_kv::fused_dot_product(&rq, &indices, norm, 2, 128);
```
