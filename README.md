# tq-kv

**Pure Rust TurboQuant -- extreme KV cache compression for LLM inference.**

[![Crates.io](https://img.shields.io/crates/v/tq-kv)](https://crates.io/crates/tq-kv)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)]()
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![no_std](https://img.shields.io/badge/no__std-compatible-blue)]()

Implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with architectural improvements. Compresses LLM KV cache keys to 2-4 bits. Zero C/C++ dependencies. NVIDIA CUDA + CPU.

```
15.1x compression | 0.997 cosine similarity | 8.9x fused attention speedup | CUDA 3.2x vs CPU
```

## Why tq-kv?

| | tq-kv | Other TurboQuant impls |
|:--|:------|:----------------------|
| **Language** | Pure Rust | Python + C + Metal shaders |
| **GPU** | NVIDIA CUDA (RTX 3080+) | Apple Metal only |
| **Compression** | **14.2x** (2-bit) | 4.6x (turbo3) |
| **Fused attention** | AVX2+FMA SIMD, 8.9x speedup | Metal kernel |
| **KV cache** | O(1) incremental append | Streaming |
| **Dependencies** | Zero C/C++ | CUDA runtime / llama.cpp |
| **no_std** | Yes (feature flag) | No |

## Key Results

> RTX 3080 10GB, i9-13900K, 64GB RAM, CUDA 13.2, Rust 1.91

| Bits | Compression | Cosine Sim | Fused vs Decompress | VRAM Saved (Llama-3 8B) |
|:----:|:----------:|:----------:|:-------------------:|:-----------------------:|
| 2 | **14.2x** | 0.943 | **8.9x faster** | 238 MB |
| 3 | **9.8x** | 0.984 | **8.6x faster** | 230 MB |
| 4 | **3.8x** | 0.996 | **8.6x faster** | 208 MB |

### Perplexity (wikitext-2, 2249 tokens, Trendyol Llama-3 8B)

| Mode | PPL | vs Baseline |
|:-----|----:|:----------:|
| Standard (fp32 KV) | **9.515** | baseline |
| TurboQuant 4-bit | **9.594** | +0.8% |
| TurboQuant 3-bit | **9.899** | +4.0% |
| TurboQuant 2-bit | **12.730** | +33.8% |

4-bit compression at **3.8x** with only **0.8% perplexity increase**.

### Per-Token Inference Cost (4096 context)

| | Old Architecture | tq-kv |
|:--|:---:|:---:|
| Cache overhead/token | ~608 ms (growing) | **~0.65 ms (constant)** |
| Strategy | decompress ALL + recompress ALL | compress 1 + append |
| Improvement | | **935x** |

## Algorithm

```
Input key vector (fp16/fp32)
    |
    v
[1] Randomized Hadamard Transform              O(d log d)
    |  Walsh-Hadamard + random sign flip
    |  Decorrelates outliers --> coordinates ~ Gaussian(0, sigma)
    v
[2] Adaptive Lloyd-Max Quantization             O(d)
    |  sigma = ||x|| / sqrt(d) per vector (not fixed -- our improvement)
    |  Pre-computed optimal centroids for Gaussian
    |  Bit-packed: 2-bit=4/byte, 3-bit=8/3bytes, 4-bit=2/byte
    v
Output: packed_indices (u8) + norms (f32)

--- Fused Attention (zero decompression) ---

  dot(q, k) = <R*q, centroids[idx]> * sigma
  Pre-rotate query once, then per-key = table lookup + SIMD dot.
  AVX2+FMA: 8 floats/cycle, gather + fused multiply-add.
```

## Quick Start

### Library

```toml
[dependencies]
tq-kv = "0.1"
```

```rust
use tq_kv::*;

let config = TurboQuantConfig::extreme(); // 2-bit, ~15x compression
let dim = 128;

// Compress keys
let compressed = compress_keys(&kv_data, dim, &config);

// Fused attention (no decompression -- SIMD accelerated)
let signs = hadamard::generate_signs(dim, config.rotation_seed);
let centroids = codebook::get_centroids(config.bits);
let rotated_q = pre_rotate_query_with_signs(&query, &signs);
let scores = fused_attention_scores(&rotated_q, &compressed, centroids, scale);

// Incremental KV cache (O(1) per token)
let mut cache = CompressedKeys::new_empty(config.bits, dim, config.rotation_seed);
let (packed, norm) = compress_single_key_with_signs(&new_key, dim, &config, &signs);
cache.append_raw(&packed, norm);
```

**Presets:**

| Preset | Bits | Ratio | Use Case |
|:-------|:----:|:-----:|:---------|
| `TurboQuantConfig::extreme()` | 2 | 14.2x | Maximum compression, edge deployment |
| `TurboQuantConfig::aggressive()` | 3 | 9.8x | Good balance for most models |
| `TurboQuantConfig::balanced()` | 4 | 3.8x | Highest quality, minimal PPL impact |

### Engine (inference binary)

```bash
# Build (Windows MSVC + CUDA)
build.bat release

# Single prompt with 2-bit TurboQuant
tq-engine "Explain quantum computing" --turbo-quant --tq-bits 2

# Interactive chat
tq-engine --interactive --turbo-quant --tq-bits 4

# CUDA GPU inference
tq-engine "Hello" --turbo-quant --tq-bits 2 --model-path model.gguf --tokenizer-repo org/model

# CPU only
tq-engine "Hello" --turbo-quant --tq-bits 2 --cpu

# Perplexity evaluation
tq-engine --perplexity wikitext2.txt --turbo-quant --tq-bits 4

# HTTP API
tq-engine --serve --port 8088 --turbo-quant --tq-bits 2
```

Supported architectures: Llama-3, Qwen2.5

### no_std

The core compression library works without `std`:

```toml
[dependencies]
tq-kv = { version = "0.1", default-features = false }
```

RNG-dependent functions (`generate_signs`, `randomized_hadamard`) require the `std` feature. The `_with_signs` variants work in `no_std` with pre-computed sign arrays.

## Performance Deep Dive

### Fused Attention: Why 8.9x Faster

Standard approach decompresses every cached key before computing attention:
```
For each key: unpack indices -> dequantize -> inverse Hadamard -> dot product
```

tq-kv's fused approach uses the orthogonality of the Hadamard transform:
```
<q, k> = <q, R^T * R*k> = <R*q, R*k> = <R*q, centroids[idx]> * sigma
```

Pre-rotate the query once. Then per-key cost is just a centroid table lookup and a dot product -- no decompression. The inner loop uses AVX2 gather + FMA instructions processing 8 floats per cycle.

### Incremental Cache: Why 935x Less Overhead

Old: decompress N cached keys + concatenate + recompress N+1 keys = O(N) per token.
New: compress 1 new key + append packed bytes = O(1) per token.

At 4096 context length, the old approach spends ~608ms on cache management alone. The new approach: ~0.65ms.

### CUDA Support

GGUF quantized models run on NVIDIA GPUs with custom CUDA-compatible implementations of:
- **RmsNorm** -- candle's `quantized_nn::RmsNorm` dequantizes to CPU; ours dequantizes to the target device
- **RoPE** -- interleaved rotary embeddings via basic tensor ops (candle's `rope_i` lacks CUDA kernel)
- **Softmax** -- numerically stable log-sum-exp via primitive CUDA-supported ops

Result: **3.2x speedup** over CPU on RTX 3080 with Llama-3 8B.

## Benchmarks

> RTX 3080 10GB, i9-13900K, 64GB RAM, Windows 11, CUDA 13.2, Rust 1.91

### Compression Quality

| Model | Bits | Ratio | SNR (dB) | Cosine Sim | Compress | Decompress |
|:------|:----:|------:|---------:|-----------:|---------:|-----------:|
| Llama-3 8B | 2 | **14.2x** | 9.4 | 0.943 | 55 ms | 23 ms |
| Llama-3 8B | 3 | **9.8x** | 14.7 | 0.984 | 65 ms | 25 ms |
| Llama-3 8B | 4 | **3.8x** | 20.3 | 0.996 | 75 ms | 23 ms |
| Gemma 3 4B | 2 | **15.1x** | 9.3 | 0.942 | 57 ms | 30 ms |
| Gemma 3 4B | 3 | **10.2x** | 14.7 | 0.984 | 65 ms | 25 ms |

### Fused Attention (512 cached keys, SIMD AVX2+FMA)

| Bits | Fused | Decompress+dot | Speedup |
|:----:|------:|---------------:|--------:|
| 2 | 82 us | 736 us | **8.9x** |
| 3 | 133 us | 723 us | **5.4x** |
| 4 | 84 us | 723 us | **8.6x** |

### VRAM Savings (Keys only, 4096 context)

| Model | KV fp16 | 2-bit | Saved |
|:------|--------:|------:|------:|
| Llama-3 8B (32L) | 256 MB | 18 MB | **238 MB** |
| Qwen 72B (80L) | 640 MB | 45 MB | **595 MB** |
| Gemma 3 4B (26L) | 208 MB | 14 MB | **194 MB** |

Run benchmarks: `cargo run --release -p tq-kv --bin tq-kv-bench`

## Our Contributions Beyond the Paper

| Innovation | Detail |
|:-----------|:-------|
| **Adaptive sigma** | Per-vector `sigma = \|\|x\|\| / sqrt(d)` instead of fixed `1/sqrt(d)`. Matches actual post-rotation variance. |
| **QJL removal** | QJL disabled at all bit widths. +1.2 dB SNR not worth 29x slower compress, 128x slower decompress, worse ratio. |
| **Fused SIMD attention** | AVX2+FMA centroid gather + dot product. 8 floats/cycle. Zero decompression. |
| **Rayon parallel heads** | Multi-head attention computed in parallel across CPU cores. |
| **O(1) incremental cache** | Append-only packed indices. No recompression. 935x overhead reduction. |
| **CUDA GGUF support** | Custom RmsNorm, RoPE, softmax for candle quantized models on NVIDIA GPUs. |
| **no_std core** | Compression/decompression works without allocator (pre-computed signs). |

## Architecture

```
tq-kv/                         Compression library (crates.io)
  lib.rs                       compress_keys, fused_dot_product, fused_attention_scores
  codebook.rs                  Lloyd-Max 2/3/4-bit optimal centroids
  hadamard.rs                  Fast Walsh-Hadamard Transform + sign caching

src/                           tq-engine inference binary
  engine.rs                    Dual-mode: stock candle or TurboQuant
  models/
    turbo_llama.rs             Llama-3 + compressed KV, Rayon parallel fused attention
    turbo_qwen2.rs             Qwen2 + compressed KV, same architecture
  config.rs                    Model configs (Llama-3 8B, Qwen 72B, Gemma 4B)
  serve.rs                     HTTP daemon (/health, /infer)
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

## Citation

If you use tq-kv in your research, please cite the original paper:

```bibtex
@inproceedings{ashkboos2026turboquant,
  title={TurboQuant: Online Vector Quantization for Efficient KV-Cache Compression},
  author={Ashkboos, Saleh and Zandieh, Amir and others},
  booktitle={ICLR},
  year={2026}
}
```
