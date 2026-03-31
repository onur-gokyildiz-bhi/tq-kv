# tq-kv

**Pure Rust TurboQuant KV cache compression. CUDA. AVX2 SIMD. C FFI. crates.io.**

[![Crates.io](https://img.shields.io/crates/v/tq-kv)](https://crates.io/crates/tq-kv)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE-MIT)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)]()
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![no\_std](https://img.shields.io/badge/no__std-compatible-blue)]()

Implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with the **3-Fix framework** that makes it work on GGUF quantized models -- where every other implementation produces catastrophic output.

---

## The Compound Error Problem

Every TurboQuant implementation assumes FP16 model weights. In practice, everyone runs GGUF quantized models (Q4_K_M). When you stack weight quantization + KV cache quantization, errors compound through softmax:

| Implementation | Qwen 7B Q4_K_M PPL | Status |
|:---------------|--------------------:|:-------|
| No compression | 5.18 | Baseline |
| turboquant_plus (symmetric) | 3,556 | **Catastrophic** |
| **tq-kv (3-Fix)** | **6.07** | **Working (+17%)** |
| turboquant_plus (asymmetric K=q8_0) | 6.64 | Working (+1%) but K uncompressed |

**tq-kv is the only implementation that compresses keys on GGUF and produces coherent output.** Others either crash (symmetric turbo) or avoid K compression entirely (asymmetric K=q8_0).

---

## Measured Quality (Honest Numbers)

> All PPL measured on wikitext-2 with automated `tq perplexity` / `tq ablate`.

### FP16 Models (no compound error)

> Qwen 2.5 0.5B FP16 safetensors, wikitext-2, 2366 tokens

| Bits | PPL | vs Baseline | Compression |
|:----:|----:|:-----------:|:-----------:|
| Baseline | 10.740 | -- | 1.0x |
| 4-bit | 11.967 | +11.4% | 7.5x |
| 3-bit | 13.933 | +29.7% | 9.8x |
| 2-bit | 27.696 | +157.9% | 14.2x |

### GGUF Q4_K_M Models (compound error)

> Qwen 2.5 7B Q4_K_M, wikitext-2, 2366 tokens, skip=4, sink=4

| Config | PPL | vs Baseline | Notes |
|:-------|----:|:-----------:|:------|
| Baseline (no TQ) | 5.178 | -- | |
| **K4-bit, V-fp16** | **6.065** | **+17.1%** | Default recommended |
| K4-bit, V-8bit | 6.076 | +17.3% | V8 nearly free (+0.2%) |
| K4-bit, V-4bit | 6.143 | +18.6% | V4 acceptable |
| K4-bit, skip=16 | 6.005 | +16.0% | Fewer compressed layers |

### Value Compression is Nearly Free

Key insight matching turboquant_plus findings: softmax amplifies K errors exponentially, V errors scale linearly. Compressing values has minimal quality impact:

| Value Config | Extra PPL vs K-only | Value Savings |
|:-------------|:-------------------:|:-------------:|
| V-fp16 (default) | -- | 1.0x |
| V-8bit (`TQ_VBITS=8`) | +0.2% | 2.0x |
| V-4bit (`TQ_VBITS=4`) | +1.3% | 3.2x |

### Compression Quality (synthetic, per-layer)

| Model | Bits | Ratio | SNR (dB) | Cosine Sim |
|:------|:----:|------:|---------:|-----------:|
| Llama-3 8B | 2 | **14.2x** | 9.2 | 0.943 |
| Llama-3 8B | 3 | **9.8x** | 14.7 | 0.984 |
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
tq-kv = "0.5"
```

### CLI (tq-engine)

```bash
# Pull a model
tq pull qwen2:7b

# Chat with TurboQuant compression
tq chat qwen2:7b --turbo-quant

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

// Incremental KV cache -- O(1) per token
let mut cache = CompressedKeys::new_empty(config.bits, dim, config.rotation_seed);
let (packed, norm) = compress_single_key_with_signs(&new_key, dim, &config, &signs);
cache.append_raw(&packed, norm);
```

---

## Configuration Guide

### Recommended Configs by Use Case

| Use Case | Key Bits | Value Bits | Skip | Sink | PPL Impact | Memory Savings |
|:---------|:--------:|:----------:|:----:|:----:|:----------:|:--------------:|
| **Quality-first (GGUF)** | 4 | fp16 | 4 | 4 | +17% | ~3.5x keys |
| **Balanced (GGUF)** | 4 | 8 | 4 | 4 | +17.3% | ~3.5x K + 2x V |
| **Maximum savings** | 4 | 4 | 4 | 4 | +18.6% | ~3.5x K + 3.2x V |
| **FP16 models** | 4 | fp16 | 4 | 4 | +11.4% | ~3.5x keys |
| **Long context** | 4 | 8 | 4 | 4 | ~+17% | Enables 2-4x longer ctx |

### Environment Variables

All TurboQuant behavior is runtime-configurable via environment variables:

| Variable | Default | Description |
|:---------|:-------:|:------------|
| `TQ_SKIP` | 4 | Number of initial layers to keep uncompressed (fp16 KV). Higher = better quality, less savings. |
| `TQ_SINK` | 4 | Number of initial tokens preserved at fp16. Attention sinks get disproportionate weight. |
| `TQ_VBITS` | 0 | Value compression bits. 0=fp16, 4=4-bit, 8=8-bit. 8 recommended (nearly free). |
| `TQ_SPARSE_V` | 1e-6 | Skip V rows where softmax weight < threshold. 0=disabled. CPU only. |
| `TQ_FUSED` | 0 | Fused attention from compressed indices. 1=enabled. CPU only, 6-8.9x speedup. |
| `TQ_DECAY` | off | Temporal decay. Format: "age:bits" e.g. "512:2". Old tokens auto-demoted. |
| `TQ_LAYER_BITS` | -- | Per-layer bit width. Format: "start-end:bits" e.g. "4-15:2,16-27:4". |
| `TQ_HEAD_BITS` | -- | Per-head bit width. Format: "0-3:4,4-7:2". |
| `TQ_BIAS_CORRECT` | 0 | Softmax bias correction. 1=enabled. Experimental. |
| `TQ_GROUP` | 32 | Group size for per-group sigma. 0=per-vector (legacy). 32 recommended. |
| `TQ_RESIDUAL` | 0 | Residual quantization bits. e.g. 2 for 4+2=6 effective bits. |
| `TQ_OUTLIER` | 0 | Top-K outlier entries preserved at full precision per vector. |
| `TQ_NO_CAL` | 0 | 1=disable calibration auto-loading. |

### Tuning Tips

**For GGUF Q4_K_M models:**
- Start with defaults: `--turbo-quant --tq-bits 4` (skip=4, sink=4, group=32)
- Add value compression: `TQ_VBITS=8` for ~2x extra savings at +0.2% PPL
- On CPU, enable fused attention: `TQ_FUSED=1` for 6-8.9x attention speedup
- For long context, enable sparse V: `TQ_SPARSE_V=1e-5` (skips 50-80% of V rows)

**For FP16 safetensors models:**
- Same config works, lower PPL impact (+11% vs +17%)
- Run `tq calibrate <model>` first for optimal codebook + rotation

**For maximum quality:**
- Increase skip: `TQ_SKIP=8` or `TQ_SKIP=16` (fewer layers compressed)
- Add residual: `TQ_RESIDUAL=2` (second-pass error correction)
- Add outlier preservation: `TQ_OUTLIER=4` (top-4 outliers kept at fp32)
- Calibrate: `tq calibrate <model>` (learned codebook from real activations)

---

## The 3-Fix Framework

TurboQuant fails on GGUF models due to compound quantization error (W4 weights + KV4 cache). Our 3-Fix framework solves this:

### Fix 1: Sink Token Preservation (`TQ_SINK`)

First N tokens' keys stay fp16. Attention sinks receive disproportionate weight regardless of content -- quantizing them causes most of the attention distribution error.

### Fix 2: Past-Only Quantization (POQ)

During generation, the current token's key uses its fp16 original in attention. It gets compressed into cache for future tokens. This protects the highest-impact position (most recent context).

```
Time t:   [sink_FP16 | past_compressed | current_t_FP16]
Time t+1: [sink_FP16 | past_compressed + current_t_compressed | current_{t+1}_FP16]
```

### Fix 3: Cache State Management

Hard reset of compressed KV state on new conversations. Prevents cross-conversation contamination from stale RoPE-mismatched keys.

---

## System Architecture

```
tq-kv/                              Compression library (crates.io v0.5.0)
  src/lib.rs                         compress_keys, fused_attention, sparse_attn_v_mul
  src/codebook.rs                    Lloyd-Max 2/3/4-bit + CalibratedCodebook
  src/hadamard.rs                    Fast Walsh-Hadamard + PCA rotation calibration
  src/qjl.rs                         SRHT QJL error correction (adaptive)
  src/candle_kv.rs                   TurboKvCache (candle drop-in)
  src/ffi.rs                         C FFI: single-head + multi-head layer API
  include/tq_kv.h                    C header for llama.cpp integration

src/                                 tq-engine inference binary ("tq")
  engine.rs                          Unified GenericTurboModel backend (GPU/CPU)
  models/turbo_generic.rs            5 GGUF architectures, auto-detected
  calibrate.rs                       Calibration pipeline (codebook + rotation + scales)
  serve.rs                           Hardened OpenAI-compatible HTTP API + Web UI
  chat.rs                            Multi-turn templates (Llama3, Qwen, Phi3, Mistral, Gemma)
  hub.rs                             HuggingFace model hub (pull/list/rm)
```

### Feature Summary

| Feature | Status | Details |
|:--------|:------:|:--------|
| 2/3/4-bit Lloyd-Max quantization | Production | Per-group sigma (g=32), norm correction |
| Fused attention (AVX2+FMA SIMD) | Production | 6-8.9x speedup, zero decompression |
| 3-Fix Framework (GGUF) | Production | Sink + POQ + cache reset |
| K/V asymmetric compression | Production | Keys 2-4 bit, values 4/8-bit or fp16 |
| Sparse V multiply | Production | Fused decompress, skips inactive rows |
| Temporal decay | Production | Auto-demote old tokens to lower bits |
| Per-head adaptive bitwidth | Production | Static (env var) or calibration-based |
| Calibration pipeline | Production | Codebook + PCA rotation + channel scales |
| SRHT QJL | Available | 115x faster, +4.5 dB SNR (off by default) |
| Residual quantization | Available | Second-pass error correction |
| Outlier preservation | Available | Top-K entries kept at full precision |
| O(1) incremental cache | Production | 935x vs naive recompression |
| CUDA GPU | Production | Custom RmsNorm, RoPE, softmax |
| C FFI | Production | `tq_kv.h` + `libtq_kv.a` for llama.cpp |
| PyO3 Python bindings | Available | `--features python` |
| `no_std` core | Available | For embedded/bare-metal targets |
| 5 GGUF architectures | Production | Qwen2, Llama, Mistral, Phi3, Gemma2 |
| Safetensors FP16 loading | Production | BF16 auto-cast |
| OpenAI-compatible API | Production | SSE streaming, hardened server |
| Web UI | Production | Embedded in binary |

---

## C FFI

Build: `cargo build --release --features ffi` produces `libtq_kv.a` + `tq_kv.h`.

```c
#include "tq_kv.h"

// Single-head API
TqContext *ctx = tq_init(4, 128, 0);              // 4-bit, head_dim=128
tq_compress_and_append(ctx, key_data, 128);
float scores[seq_len];
tq_fused_attention(ctx, query_data, 128, scores, scale);
tq_free(ctx);

// Multi-head layer API (designed for llama.cpp GQA)
TqLayerContext *layer = tq_layer_init(4, 8, 128, 0);  // 4-bit, 8 KV heads
tq_layer_compress_and_append(layer, all_heads_key_data, 8 * 128);
tq_layer_fused_attention(layer, kv_head_idx, query, 128, scores, scale);
tq_layer_free(layer);
```

Link: `-ltq_kv -lpthread -ldl -lm` (Linux) or `tq_kv.lib` (Windows MSVC).

---

## Benchmarks

```bash
cargo run --release -p tq-kv --bin tq-kv-bench   # compression quality
tq bench qwen2:7b                                  # tok/s, TTFT
tq ablate qwen2:7b --file eval.txt                 # PPL sweep
```

### Fused Attention (512 cached keys, AVX2+FMA)

| Bits | Fused | Decompress+dot | Speedup |
|:----:|------:|---------------:|--------:|
| 2 | 46 us | 515 us | **8.9x** |
| 4 | 46 us | 389 us | **8.6x** |

### Incremental Cache

| Method | Per-token cost | Scaling |
|:-------|---------------:|:-------:|
| Naive (decompress + recompress) | ~608 ms | O(N) |
| tq-kv (compress + append) | ~0.65 ms | **O(1)** |

---

## Known Limitations

- **K compression on GGUF Q4_K_M adds +17% PPL.** This is the compound quantization error problem. No implementation has solved it at the algorithm level. tq-kv's 3-Fix framework is the only one that produces working output (others get PPL 3556+). Asymmetric K=uncompressed + V=turbo avoids the issue but doesn't compress keys.
- **FP16 4-bit adds +11.4% PPL on small models (0.5B).** Larger models (7B+, 70B+) are more tolerant. TheTom reports +0.23% on 35B FP16.
- **No Metal/Apple Silicon.** CUDA and CPU only.
- **Calibration pipeline experimental.** Channel scales and PCA rotation can degrade quality on some models. Use `TQ_NO_CAL=1` to disable.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

## Citation

```bibtex
@software{gokyildiz2026tqkv,
  title={tq-kv: Production-Grade TurboQuant KV Cache Compression in Rust},
  author={Gokyildiz, Onur},
  year={2026},
  url={https://github.com/onur-gokyildiz-bhi/tq-kv}
}

@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization for Efficient KV-Cache Compression},
  author={Zandieh, Amir and Daliri, Majid and others},
  booktitle={ICLR},
  year={2026}
}
```
