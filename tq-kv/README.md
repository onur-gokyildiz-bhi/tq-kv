# tq-kv

**TurboQuant: Extreme KV Cache Compression for LLMs**

Pure Rust implementation of Google's TurboQuant algorithm (ICLR 2026).
Compresses LLM key-value cache to 2-4 bits with up to 15x memory reduction.

**First TurboQuant that works on GGUF quantized models** — all other implementations fail on Q4_K_M.

## What's New in v0.5.0

- **3-Fix Framework**: Sink token preservation + Past-Only Quantization + cache reset — solves compound error on GGUF models (300+ token coherent output where others produce gibberish)
- **SRHT QJL**: O(d log d) structured projection replaces O(d^2) dense — 115x faster, +4.5 dB SNR
- **Adaptive QJL**: Context-length-aware error correction (QjlMode::Off/On/Adaptive)
- **Norm Correction**: Reconstruction norm matching — zero decode cost quality improvement
- **Gaussianity Verified**: Kurtosis 35.3 -> 3.3 after rotation (Gaussian=3.0)

## Results

| Bits | Compression | SNR (dB) | Cosine Sim | NIAH |
|:----:|:-----------:|:--------:|:----------:|:----:|
| 2 | **15.1x** | 9.3 | 0.942 | 9/9 pass |
| 3 | **10.2x** | 14.7 | 0.984 | — |
| 4 | **3.8x** | 20.3 | 0.996 | 9/9 pass |
| 4+QJL | **2.7x** | 24.8 | 0.998 | — |

### SRHT QJL Performance (32K vectors, d=128, release)

| Metric | Dense QJL (paper) | SRHT QJL (ours) | No QJL |
|:-------|:-----------------:|:---------------:|:------:|
| Compress overhead | 29x | **1.45x** | 1.0x |
| SNR improvement | +1.2 dB | **+4.5 dB** | — |
| Attention KL div. | — | **2.9x lower** | — |

## How It Works

```
Input KV vector (from GGUF Q4_K_M model)
    |
[1] Randomized Hadamard Transform           O(d log d)
    |  Decorrelates outliers → coordinates ~ Gaussian
    |  Verified: kurtosis 35.3 → 3.3 (Gaussian=3.0)
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
        3.8x compression at 4-bit (keys only)
```

### 3-Fix for GGUF Models

TurboQuant fails on quantized models due to compound error (W4 weights + KV4 cache).
Our 3-Fix framework solves this:

```
Fix 1: Sink tokens (first 4) stay FP16     → -81% attention error
Fix 2: Current token = lossless (POQ)       → highest-impact position protected
Fix 3: Cache reset per conversation         → prevents cross-contamination

K_attention = [K_sink_FP16 | K_compressed | K_current_FP16]
```

## Quick Start

```toml
[dependencies]
tq-kv = "0.5"
```

```rust
use tq_kv::{TurboQuantConfig, compress_keys, decompress_keys};

let head_dim = 128;
let kv_data: Vec<f32> = vec![0.1; head_dim];

// 4-bit balanced compression (3.8x, cos_sim 0.996)
let config = TurboQuantConfig::balanced();
let compressed = compress_keys(&kv_data, head_dim, &config);
println!("Ratio: {:.1}x", compressed.compression_ratio());

let restored = decompress_keys(&compressed, &config);
```

### Adaptive QJL

```rust
use tq_kv::{TurboQuantConfig, QjlMode};

// Auto-enable QJL at long context (4K+ tokens)
let config = TurboQuantConfig::balanced_adaptive();

// Check if QJL should activate
let use_qjl = config.should_use_qjl(current_cache_length);
```

### Fused Attention (Pre-Rotated Query)

Skip decompression entirely — 6x faster attention:

```rust
use tq_kv::{pre_rotate_query, fused_attention_scores, codebook};

let rotated_q = pre_rotate_query(&query, config.rotation_seed);
let centroids = codebook::get_centroids(config.bits);
let scores = fused_attention_scores(&rotated_q, &compressed, centroids, scale);
```

## VRAM Savings

| Model | Context | FP16 KV | TQ 4-bit | TQ 2-bit | Savings |
|:------|:-------:|:-------:|:--------:|:--------:|:-------:|
| Qwen 2.5 7B | 4K | 256 MB | 48 MB | 18 MB | 5.3-14.2x |
| Qwen 2.5 72B | 4K | 640 MB | 120 MB | 45 MB | 5.3-14.2x |
| Llama 3.1 70B | 32K | 20 GB | 5.3 GB | 1.4 GB | 3.8-14.2x |

## Full Product: tq-engine

tq-kv powers tq-engine — "Rust's Ollama" with TurboQuant compression:

```bash
tq pull qwen2:7b          # download from HuggingFace
tq serve --turbo-quant     # OpenAI-compatible API (SSE streaming)
tq chat qwen2:7b           # terminal chat
```

Web UI at localhost:11435. Works with ChatBox and Open WebUI.

5 architectures: Qwen2, Llama, Mistral, Phi3, Gemma2 — auto-detected from GGUF.

## Benchmark

```bash
cargo run --release -p tq-kv --bin tq-kv-bench
```

Full results: [BENCHMARK.md](../BENCHMARK.md)

## Paper

**Our work:**
- "TurboQuant on Quantized Models: Solving Compound Quantization Error in GGUF LLMs" — BHI Research (2026)
- 3-Fix framework, SRHT QJL (115x speedup), Adaptive QJL, Norm Correction

**Original:**
- Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## License

MIT OR Apache-2.0
