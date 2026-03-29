# CLAUDE.md

## Project

**tq-kv** — Pure Rust implementation of Google's TurboQuant (ICLR 2026) KV cache compression algorithm. Workspace with two crates:

- `tq-kv/` — standalone compression library (crates.io v0.5.0)
- `tq-engine` (root) — "Rust's Ollama" inference engine with TurboQuant + candle backend

## Build

Windows + MSVC + CUDA 13.2. Use `build.bat` which sets up vcvarsall, LIBCLANG_PATH, CUDA_PATH, and NVCC_PREPEND_FLAGS.

```bash
build.bat          # debug build
build.bat release  # release build
build.bat check    # type check only
build.bat test     # run tests
```

Direct cargo commands require MSVC env — always use build.bat or the PowerShell env setup pattern.

## Testing

```bash
cargo test --workspace
```

Framework: Rust built-in `#[test]`. 27 tests across tq-kv crate. No test infra for tq-engine (requires model download).

## Architecture

```
Cargo.toml (workspace)
├── turbo-quant/          # Standalone crate (tq-kv)
│   ├── lib.rs            # compress_keys / decompress_keys / fused_dot_product
│   ├── codebook.rs       # Lloyd-Max 2/3/4-bit optimal centroids
│   ├── hadamard.rs       # Fast Walsh-Hadamard Transform
│   ├── polar.rs          # PolarQuant (v1 legacy)
│   ├── qjl.rs            # QJL 1-bit error correction
│   └── bench.rs          # Benchmark suite
│
└── src/                  # tq-engine binary
    ├── engine.rs          # Dual-mode: stock candle or TurboQuant models
    ├── models/
    │   ├── turbo_llama.rs # candle quantized_llama fork + compressed KV
    │   └── turbo_qwen2.rs # candle quantized_qwen2 fork + compressed KV
    ├── config.rs          # Model configs (Llama-3 8B, Qwen 72B, Gemma 4B)
    ├── serve.rs           # HTTP daemon (/health, /infer)
    └── chat.rs            # Llama3 + Qwen chat templates
```

## Key Design Decisions

- **Adaptive sigma**: per-vector `sigma = ||x|| / sqrt(d)` instead of fixed `1/sqrt(d)`. Our contribution, not in paper.
- **Selective QJL**: QJL disabled by default at all bit widths (overhead > benefit). Available as opt-in for maximum quality.
- **Keys only**: Only keys compressed, values stay fp16/f32. Matches paper.
- **No unit normalization**: Codebook works directly on Gaussian coordinates, not unit sphere.

## Dependencies

- `candle-core` 0.9 (CUDA feature) — tensor ops, GGUF loading
- `candle-transformers` 0.9 — stock quantized models
- `tokenizers` 0.20 — HF tokenizer
- `tq-kv` (path) — our compression crate

## Benchmark

```bash
cargo run --release -p tq-kv --bin tq-kv-bench
```

Results in BENCHMARK.md. Key numbers: 2-bit = 15.1x compression, 4-bit = 3.8x (QJL off, default) to 5.8x (QJL on, Gemma 4B), cos_sim 0.942-0.997.
