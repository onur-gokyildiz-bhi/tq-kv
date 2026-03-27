# tq-kv

**TurboQuant: Extreme KV Cache Compression for LLMs**

Pure Rust implementation of Google's TurboQuant algorithm (ICLR 2026).
Compresses LLM key-value cache to 2-4 bits with up to 15x memory reduction and near-zero quality loss.

## Results

| Bits | Compression | SNR (dB) | Cosine Sim | Speed (128-dim) |
|:----:|:-----------:|:--------:|:----------:|:---------------:|
| 2 | **15.1x** | 9.3 | 0.942 | 56 ms compress |
| 3 | **10.2x** | 14.7 | 0.984 | 65 ms compress |
| 4 | **5.8x** | 21.4 | 0.997 | 5 ms compress |

## How It Works

```
Input KV vector
    |
[1] Randomized Hadamard Transform    O(d log d)
    |  Eliminates outliers, makes coordinates ~ Gaussian
    |
[2] Lloyd-Max Codebook Quantization  O(d)
    |  Optimal scalar quantizer for Gaussian distribution
    |  Pre-computed centroids: 4 (2-bit), 8 (3-bit), 16 (4-bit)
    |
[3] QJL 1-bit Error Correction       O(d) optional
    |  Johnson-Lindenstrauss residual projection
    |
Compressed: packed indices + norm + optional QJL signs
```

## Quick Start

```toml
[dependencies]
tq-kv = "0.4"
```

```rust
use tq_kv::{TurboQuantConfig, compress_keys, decompress_keys};

// Simulate a KV cache vector (head_dim must be power of 2)
let head_dim = 128;
let kv_data: Vec<f32> = vec![0.1; head_dim];

// 2-bit extreme compression (15x)
let config = TurboQuantConfig::extreme();
let compressed = compress_keys(&kv_data, head_dim, &config);
println!("Ratio: {:.1}x", compressed.compression_ratio());

let restored = decompress_keys(&compressed, &config);
```

## Fused Attention (Pre-Rotated Query Trick)

Skip decompression entirely during attention computation:

```rust
use tq_kv::{pre_rotate_query, fused_dot_product};

// Pre-rotate query ONCE
let rotated_q = pre_rotate_query(&query, config.rotation_seed);

// For each cached key: centroid lookup, no decompression
let score = fused_dot_product(&rotated_q, &key_indices, key_norm, bits, dim);
```

## VRAM Savings

| Model | KV Cache fp16 | 2-bit TurboQuant | Saved |
|:------|:-------------:|:----------------:|:-----:|
| Llama-3 8B | 256 MB | 18 MB | **238 MB** |
| Qwen2.5 72B | 640 MB | 45 MB | **595 MB** |
| Gemma 3 4B | 208 MB | 14 MB | **194 MB** |

## Benchmark

```bash
cargo run --release -p tq-kv --bin tq-kv-bench
```

## Paper

Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026.
[arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

## License

MIT OR Apache-2.0
