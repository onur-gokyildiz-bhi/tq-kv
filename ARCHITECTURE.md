# tq-kv Architecture Documentation

**Version:** 0.5.0+ (March-April 2026)
**Author:** Onur Gokyildiz, BHI Research
**Total:** ~13.3K LOC Rust, 32 source files, 114 tests

---

## 1. Project Overview

tq-kv is a two-crate Rust workspace:

```
tq-kv/                          Library crate (crates.io: tq-kv v0.5.0)
  6,771 LOC                     Compression algorithms, C FFI, Python bindings

src/                            Binary crate (tq-engine)
  6,578 LOC                     Inference engine, HTTP server, model hub
```

### What It Does

Compresses LLM Key-Value cache to 2-4 bits using Google's TurboQuant algorithm (ICLR 2026), with the **3-Fix Framework** that makes it work on GGUF quantized models where all other implementations fail catastrophically (PPL 3556+).

### Key Numbers

| Metric | Value |
|--------|-------|
| FP16 4-bit PPL delta | +11.4% (Qwen 0.5B) |
| GGUF Q4_K_M 4-bit PPL delta | +17.1% (Qwen 7B) |
| Key compression ratio | 7.5-14.2x |
| Value compression (V8) extra PPL | +0.2% (nearly free) |
| Fused attention speedup (AVX2) | 8.9x |
| Incremental cache overhead | O(1), 0.65ms/token |
| Tests | 114 passing |

---

## 2. Library Crate: tq-kv (tq-kv/)

### 2.1 Core Algorithm (lib.rs — 2,500+ lines)

**TurboQuant Compression Pipeline:**

```
Input key vector [head_dim] (fp32)
    │
    ├─ [optional] Pre-Rotation Centering: subtract per-channel bias
    │              (removes weight quantization artifacts on GGUF)
    │
    ├─ [optional] Channel scaling: multiply by per-channel scales
    │              (SmoothAttention outlier migration)
    │
    ├─ Randomized Hadamard Transform (or PCA rotation matrix)
    │     O(d log d), decorrelates outliers → coordinates ≈ N(0, σ²)
    │
    ├─ Per-group Lloyd-Max quantization (group_size=32 default)
    │     σ = group_norm / √group_size (adaptive per-group)
    │     Codebook: pre-computed optimal centroids for Gaussian
    │     Supports: standard N(0,1) or calibrated from real data
    │
    ├─ Norm correction: stored_norm = norm² / recon_norm
    │     Ensures ||decompress|| ≈ ||original||
    │
    ├─ [optional] Residual quantization (second-pass error correction)
    ├─ [optional] Outlier preservation (top-K entries at full precision)
    ├─ [optional] QJL error correction (SRHT, 115x faster than dense)
    │
    └─ Output: packed_indices [u8] + group_norms [f32]
               Bit-packed: 2-bit=4/byte, 3-bit=8/3 bytes, 4-bit=2/byte
```

**Key Public Functions:**

| Function | Purpose |
|----------|---------|
| `compress_keys()` | Batch compress multiple key vectors |
| `compress_single_key_grouped()` | Compress one key with per-group sigma + outliers + residual |
| `compress_single_key_with_signs()` | Compress one key with pre-computed Hadamard signs |
| `decompress_keys()` / `decompress_keys_grouped()` | Restore compressed keys to f32 |
| `fused_attention_scores()` | Compute attention from compressed indices (no decompress) |
| `fused_dot_product_with_centroids()` | Single key-query dot product from indices (AVX2+FMA) |
| `pre_rotate_query_with_signs()` | Pre-rotate query for fused attention |
| `sparse_attn_v_mul()` | Sparse value multiply (skip near-zero weights) |
| `calibrate_channel_scales()` | Compute per-channel SmoothQuant scales |
| `calibrate_codebook()` | Learn optimal centroids from real activations |
| `calibrate_rotation()` | PCA-based optimal rotation matrix |

**Data Structures:**

```rust
CompressedKeys {
    packed_indices: Vec<u8>,    // Bit-packed quantized indices
    norms: Vec<f32>,            // Per-vector or per-group norms
    bits: u8,                   // 2, 3, or 4
    dim: usize,                 // head_dim
    count: usize,               // Number of vectors
    group_size: usize,          // 0=per-vector, 32=per-group
    // Optional enhancements:
    residual_indices/norms,     // Second-pass error correction
    outlier_indices/values,     // Top-K preserved entries
    qjl_corrections,            // SRHT QJL sign corrections
}

TurboQuantConfig {
    bits: u8,                   // Quantization width (2/3/4)
    group_size: usize,          // Per-group sigma (default 32)
    qjl_mode: QjlMode,         // Off / On / Adaptive{threshold}
    rotation_seed: u64,         // Deterministic Hadamard signs
    sparse_v_threshold: f32,    // Skip V rows below this weight
    value_bits: u8,             // 0=fp16, 4=4-bit, 8=8-bit
    // Calibration fields:
    channel_scales: Option<Vec<f32>>,
    calibrated_codebook: Option<CalibratedCodebook>,
    rotation_matrix: Option<Vec<f32>>,
    key_channel_bias: Option<Vec<f32>>,
    // Runtime config:
    skip_layers: Option<usize>,
    sink_tokens: Option<usize>,
    per_head_bits: Option<Vec<u8>>,
}
```

### 2.2 Codebook (codebook.rs — 450 lines)

Pre-computed Lloyd-Max optimal centroids for Gaussian N(0,1):
- **2-bit**: 4 centroids, 3 boundaries
- **3-bit**: 8 centroids, 7 boundaries
- **4-bit**: 16 centroids, 15 boundaries

Also: `CalibratedCodebook` — learned centroids from real model activations via iterative Lloyd-Max (100 iterations). Remap tables for temporal decay (bit-width demotion without decompression).

### 2.3 Hadamard Transform (hadamard.rs — 350 lines)

- `fast_wht()` — In-place Fast Walsh-Hadamard Transform, O(d log d)
- `randomized_hadamard()` / `randomized_hadamard_with_signs()` — WHT + random sign flip
- `generate_signs()` — Deterministic ±1 signs from seed (reproducible)
- `apply_rotation()` / `apply_inverse_rotation()` — Custom rotation matrix (SpinQuant/PCA)
- `calibrate_pca_rotation()` — Jacobi eigendecomposition for learned rotation

### 2.4 QJL Error Correction (qjl.rs — 200 lines)

SRHT-based Quantized Johnson-Lindenstrauss transform:
- 115x faster than dense random projection (O(d log d) vs O(d²))
- +4.5 dB SNR improvement over dense QJL
- Adaptive mode: auto-enables at configurable context length threshold

### 2.5 Compaction (compaction.rs — 400 lines)

KV cache token reduction via attention matching (Zweiger et al., 2026):
- Select top-t keys by max attention weight
- Fit per-key bias (beta) via NNLS to preserve softmax partition function
- Fit synthetic values via ridge regression
- Cholesky-based solvers (no external LAPACK dependency)
- Orthogonal to quantization: 50x token reduction × 8x bit reduction = 400x

### 2.6 Value Compression (in lib.rs)

- `CompressedValues` — 8-bit absmax per-group quantization
- `CompressedValues4Bit` — 4-bit symmetric per-group quantization
- `sparse_attn_v_mul_compressed_4bit()` / `_8bit()` — Fused sparse decompress+multiply

### 2.7 C FFI (ffi.rs + include/tq_kv.h)

Opaque handle API for C/C++ integration:
- `tq_init()` / `tq_free()` — Single head context
- `tq_compress_and_append()` — Incremental cache update
- `tq_fused_attention()` — Score computation from compressed indices
- `tq_layer_*()` — Multi-head layer API (designed for llama.cpp GQA)

### 2.8 Candle Integration (candle_kv.rs)

`TurboKvCache` — Drop-in replacement for candle's `KvCache`. Transparently compresses keys and optionally values during forward pass.

### 2.9 Python Bindings (python.rs, feature="python")

PyO3 bindings: `compress_keys`, `decompress_keys`, `fused_attention_scores` accessible from Python.

---

## 3. Engine Crate: tq-engine (src/)

### 3.1 Unified Model Backend (engine.rs — 550 lines)

`GenericTurboModel` handles ALL model formats (GGUF + safetensors) on ALL devices (CPU + CUDA). No stock candle model paths — single code path with CUDA-compatible ops.

```rust
ModelWeights(GenericTurboModel)  // Single variant, unified
```

When TQ is disabled, all layers use uncompressed fp16 KV cache through the same code path (skip_layers=999).

### 3.2 GenericTurboModel (models/turbo_generic.rs — 1,800+ lines)

**The heart of the engine.** Auto-detects architecture from GGUF metadata.

**Supported Architectures:**
- Qwen2 (with attention biases)
- Llama (GQA)
- Mistral (sliding window attention)
- Phi-3/3.5 (merged QKV, padded head_dim)
- Gemma 2 (head_dim=256)

**CUDA-Compatible Custom Ops:**
- `RmsNorm` — primitive ops only (no missing CUDA kernels)
- `rope_halved()` / `rope_interleaved()` — both RoPE styles
- `softmax_last_dim()` — manual exp/sum/div (no candle_nn dependency)

**Forward Attention Path (forward_attn):**

```
x [batch, seq_len, embed_dim]
    │
    ├─ Q, K, V = W_q(x), W_k(x), W_v(x)    (QMatMul, quantized matmul)
    │
    ├─ [calibration hook: collect raw pre-bias pre-RoPE keys]
    │
    ├─ Apply attention biases (Qwen2)
    │
    ├─ Reshape to [batch, n_heads, seq_len, head_dim]
    │
    ├─ [SmoothAttention: Q *= √s, K /= √s]
    │
    ├─ RoPE (halved or interleaved)
    │
    ├─ if compressed layer:
    │     ├─ Sink tokens (first N) → stored FP16
    │     ├─ POQ: current token → FP16 in attention, compressed for future
    │     ├─ Compress K per head (per-head adaptive bitwidth)
    │     ├─ Store V (fp16 / 8-bit / 4-bit)
    │     │
    │     ├─ if FUSED + CPU:
    │     │     Pre-rotate Q, centroid lookup, AVX2+FMA SIMD
    │     ├─ else:
    │     │     Decompress all keys, standard matmul
    │     │
    │     ├─ Softmax (f32)
    │     ├─ Sparse V multiply (skip near-zero weights)
    │     └─ Output [batch, n_heads, seq_len, head_dim]
    │
    └─ if uncompressed layer:
          Standard candle-style KV cache + matmul
```

**Runtime Configuration (all via env vars or config struct):**

| Var | Default | Purpose |
|-----|---------|---------|
| TQ_SKIP | 4 | Uncompressed initial layers |
| TQ_SINK | 4 | FP16 sink tokens |
| TQ_VBITS | 0 | Value compression (0/4/8) |
| TQ_FUSED | 0 | Fused attention (CPU only) |
| TQ_SPARSE_V | 1e-6 | Sparse V threshold |
| TQ_DECAY | off | Temporal decay config |
| TQ_LAYER_BITS | — | Per-layer bit width |
| TQ_HEAD_BITS | — | Per-head bit width |
| TQ_BIAS_CORRECT | 0 | Softmax bias correction |
| TQ_GROUP | 32 | Group size for sigma |
| TQ_RESIDUAL | 0 | Residual quantization bits |
| TQ_OUTLIER | 0 | Outlier preservation K |
| TQ_NO_CAL | 0 | Disable calibration |

### 3.3 Calibration Pipeline (calibrate.rs — 450 lines)

`tq calibrate <model>` — collects real key activations and computes:

1. **Key channel bias** (Pre-Rotation Centering) — per-channel mean from raw W_k*x output
2. **Channel scales** — SmoothQuant-style outlier equalization
3. **PCA rotation matrix** — Jacobi eigendecomposition for learned rotation
4. **Calibrated codebooks** (2/3/4-bit) — Lloyd-Max from real activations
5. **Per-head importance** — key norm std-dev for adaptive bitwidth
6. **Auto head bits** — top 50% heads → 4-bit, rest → 2-bit

Saved as `~/.tq/models/{name}-{tag}/calibration.json`. Auto-loaded at engine startup.

**Calibration hook** collects pre-bias, pre-RoPE key vectors from `forward_attn()` for clean per-channel statistics.

### 3.4 HTTP Server (serve.rs — 600 lines)

Hardened axum-based OpenAI-compatible API:
- `POST /v1/chat/completions` — streaming SSE + non-streaming
- `GET /v1/models` — list loaded/available models
- `POST /v1/models/load` — atomic model swap (old model stays during load)
- `GET /v1/models/status` — ready/loading status
- `GET /health` — health check
- `POST /infer` — legacy endpoint

**Safety:**
- All mutex locks use `.map_err()` (no panic on poison)
- Client disconnect stops generation (`tx.is_closed()`)
- Message count validation (max 1000)
- Model path restricted to .gguf/.safetensors/.bin

### 3.5 Model Hub (hub.rs + catalog.rs)

```bash
tq pull qwen2:7b      # Download from HuggingFace
tq list                 # Show downloaded models
tq rm qwen2:7b          # Remove model metadata
```

- Catalog: const array of {name, tag, hf_repo, filename, arch, size_gb}
- Metadata stored in `~/.tq/models/{name}-{tag}/metadata.json`
- Auto-download on first use
- Supports: GGUF files + safetensors FP16/BF16

### 3.6 CLI (main.rs — 900 lines)

```
tq chat <model>         Interactive chat with streaming
tq serve                OpenAI-compatible API server (port 11435)
tq perplexity           PPL evaluation on text file
tq calibrate            Compute optimal codebook/rotation/scales
tq ablate               Automated PPL sweep across configs
tq bench                tok/s + TTFT benchmarks
tq doctor               System compatibility check
tq pull/list/rm         Model management
```

### 3.7 Chat Templates (chat.rs)

Multi-turn formatting for:
- Llama 3 (`<|begin_of_text|><|start_header_id|>`)
- Qwen / Qwen2 (`<|im_start|>system`)
- Mistral (`[INST]`)
- Phi-3 (`<|system|>`)
- Gemma (`<start_of_turn>`)

---

## 4. The 3-Fix Framework

Solves compound quantization error (W4 weights + KV4 cache) that breaks all other TurboQuant implementations:

**Fix 1: Sink Token Preservation** — First N tokens' keys stay FP16. Attention sinks get disproportionate weight; quantizing them causes most attention distribution error.

**Fix 2: Past-Only Quantization (POQ)** — Current token's key uses FP16 original in attention. Compressed into cache for future tokens. Protects highest-impact position.

**Fix 3: Cache State Management** — Hard reset of compressed KV state per conversation. Prevents cross-conversation contamination from stale RoPE-mismatched keys.

---

## 5. Experimental Techniques (Research)

### 5.1 Pre-Rotation Centering
Subtract per-channel key bias before Hadamard rotation. Addresses weight quantization artifacts on GGUF models. Currently limited by pre/post RoPE mismatch — full benefit requires pre-RoPE quantization.

### 5.2 SmoothAttention
Migrate K outliers to Q: `Q *= √s, K /= √s`. Mathematically invariant (`Q*K^T` preserved). Currently disabled for Q4_K_M — channel scales from calibration don't improve quality on already-quantized models.

### 5.3 KV Compaction
Token-count reduction via attention matching. Orthogonal to quantization: Compaction 50x × TurboQuant 8x = 400x theoretical. Library implemented, engine integration pending.

---

## 6. Build System

```bash
build.bat              # Debug build (Windows + MSVC + CUDA 13.2)
build.bat release      # Release build
build.bat check        # Type check only
build.bat test         # Run tests
```

**Dependencies:**
- `candle-core` 0.9 — tensor ops, GGUF loading (CUDA feature)
- `candle-transformers` 0.9 — LogitsProcessor, Sampling only
- `tokenizers` 0.20 — HuggingFace tokenizer
- `axum` + `tokio` — async HTTP server
- `hf-hub` — HuggingFace model download
- `tq-kv` (path) — our compression crate

**Feature Flags (tq-kv):**
- `default` = `["std"]` — standard library features
- `candle` — candle tensor integration + rayon parallelism
- `ffi` — C FFI layer (`libtq_kv.a` + `tq_kv.h`)
- `python` — PyO3 Python bindings

---

## 7. Test Coverage

| Category | Count | Description |
|----------|------:|-------------|
| tq-kv unit tests | 77 | Compression, codebook, hadamard, compaction |
| Engine unit tests | 16 | Auto-TQ, catalog, diagnostics |
| Integration tests | 12 | QJL attention context, long context, compare |
| Doc tests | 2 | API usage examples |
| **Total** | **114** | **0 failures** |

---

## 8. File Map

```
tq-kv/src/
  lib.rs              Core compression/decompression pipeline, fused attention
  codebook.rs         Lloyd-Max centroids, CalibratedCodebook, remap tables
  hadamard.rs         Fast Walsh-Hadamard, PCA rotation, sign generation
  qjl.rs              SRHT QJL error correction
  compaction.rs       KV compaction via attention matching
  polar.rs            PolarQuant (V1 legacy, deprecated)
  candle_kv.rs        TurboKvCache (candle drop-in)
  ffi.rs              C FFI (single-head + multi-head layer)
  python.rs           PyO3 Python bindings
  bench.rs            Benchmark suite
  demo.rs             Demo binary

tq-kv/include/
  tq_kv.h             C header for llama.cpp integration

tq-kv/tests/
  qjl_attention_context.rs  Attention KL divergence at varying context
  qjl_compare.rs            QJL on vs off quality comparison
  qjl_long_context.rs       QJL scaling with context length
  test_ffi.rs               C FFI validation (requires feature)

src/
  main.rs             CLI entry point, subcommands, TQ config resolution
  engine.rs           Unified model backend (GenericTurboModel wrapper)
  models/
    turbo_generic.rs  Auto-detecting GGUF model + TQ compression
    mod.rs            Module declaration
  calibrate.rs        Calibration pipeline (codebook, rotation, scales, bias)
  serve.rs            Hardened axum HTTP server (OpenAI API)
  chat.rs             Multi-turn chat templates
  hub.rs              HuggingFace model hub (pull/list/rm)
  catalog.rs          Model catalog (name:tag → HF repo mapping)
  config.rs           Model architecture detection
  auto_tq.rs          VRAM-aware automatic TQ configuration
  diagnostics.rs      Layer diagnostic utilities
  diagnose.rs         Diagnostic binary
  download.rs         Download progress display
  inference.rs        Legacy inference utilities
  model.rs            Legacy model loading
  web/index.html      Embedded Web UI
```
