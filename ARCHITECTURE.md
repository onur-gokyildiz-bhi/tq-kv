# tq-kv Architecture Documentation

**Version:** 0.7.0 (April 2026)
**Author:** Onur Gokyildiz, BHI Research
**Total:** ~13.7K LOC Rust, 32 source files, 86 tests

---

## 1. Project Overview

tq-kv is a two-crate Rust workspace:

```
tq-kv/                          Library crate (crates.io: tq-kv v0.6.0)
  6,783 LOC                     Compression algorithms, compaction, C FFI, Python bindings

src/                            Binary crate (tq-engine)
  6,905 LOC                     Inference engine, HTTP server, model hub
```

### What It Does

Compresses LLM Key-Value cache to 2-4 bits using Google's TurboQuant algorithm (ICLR 2026), with the **3-Fix Framework** that makes it work on GGUF quantized models where all other implementations fail catastrophically (PPL 3556+). Includes **Pre-RoPE quantization** for position-independent key compression and **KV Compaction** for attention-matching token reduction.

### Key Numbers

| Metric | Value |
|--------|-------|
| GGUF Q4_K_M 4-bit PPL delta | +5.6% (Pre-RoPE) / +13.7% (standard) |
| FP16 4-bit PPL delta | +11.4% (Qwen 0.5B) |
| Key compression ratio | 7.5-14.2x |
| Value compression (V4) extra PPL | +1.3% |
| Value compression (V8) extra PPL | +0.2% (nearly free) |
| Compaction token reduction | up to 20x |
| Combined compression potential | 100-400x (TQ + compaction + V4) |
| Fused attention speedup (AVX2) | 8.9x |
| Incremental cache overhead | O(1), 0.65ms/token |
| Tests | 86 passing |

---

## 2. Library Crate: tq-kv (tq-kv/)

### 2.1 Core Algorithm (lib.rs -- 2,500+ lines)

**TurboQuant Compression Pipeline:**

```
Input key vector [head_dim] (fp32)
    |
    +-- [optional] Pre-Rotation Centering: subtract per-channel bias
    |              (removes weight quantization artifacts on GGUF)
    |
    +-- [optional] Channel scaling: multiply by per-channel scales
    |              (SmoothAttention outlier migration)
    |
    +-- Randomized Hadamard Transform (or PCA rotation matrix)
    |     O(d log d), decorrelates outliers -> coordinates ~ N(0, sigma^2)
    |
    +-- Per-group Lloyd-Max quantization (group_size=32 default)
    |     sigma = group_norm / sqrt(group_size) (adaptive per-group)
    |     Codebook: pre-computed optimal centroids for Gaussian
    |     Supports: standard N(0,1) or calibrated from real data
    |
    +-- Norm correction: stored_norm = norm^2 / recon_norm
    |     Ensures ||decompress|| ~ ||original||
    |
    +-- [optional] Residual quantization (second-pass error correction)
    +-- [optional] Outlier preservation (top-K entries at full precision)
    +-- [optional] QJL error correction (SRHT, 115x faster than dense)
    |
    +-- Output: packed_indices [u8] + group_norms [f32]
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
| `pre_rotate_query_with_matrix()` | Pre-rotate query with custom rotation matrix |
| `sparse_attn_v_mul()` | Sparse value multiply (skip near-zero weights) |
| `sparse_attn_v_mul_compressed_4bit()` | Fused sparse decompress+multiply (4-bit V) |
| `sparse_attn_v_mul_compressed_8bit()` | Fused sparse decompress+multiply (8-bit V) |
| `calibrate_channel_scales()` | Compute per-channel SmoothQuant scales |
| `calibrate_codebook()` | Learn optimal centroids from real activations |
| `calibrate_pca_rotation()` | PCA-based optimal rotation matrix |

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
    pre_rope: bool,             // Pre-RoPE quantization mode
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

### 2.2 Codebook (codebook.rs -- 450 lines)

Pre-computed Lloyd-Max optimal centroids for Gaussian N(0,1):
- **2-bit**: 4 centroids, 3 boundaries
- **3-bit**: 8 centroids, 7 boundaries
- **4-bit**: 16 centroids, 15 boundaries

Also: `CalibratedCodebook` -- learned centroids from real model activations via iterative Lloyd-Max (100 iterations). Remap tables for temporal decay (bit-width demotion without decompression).

### 2.3 Hadamard Transform (hadamard.rs -- 350 lines)

- `fast_wht()` -- In-place Fast Walsh-Hadamard Transform, O(d log d)
- `randomized_hadamard()` / `randomized_hadamard_with_signs()` -- WHT + random sign flip
- `generate_signs()` -- Deterministic +-1 signs from seed (reproducible)
- `apply_rotation()` / `apply_inverse_rotation()` -- Custom rotation matrix (SpinQuant/PCA)
- `calibrate_pca_rotation()` -- Jacobi eigendecomposition for learned rotation

### 2.4 QJL Error Correction (qjl.rs -- 200 lines)

SRHT-based Quantized Johnson-Lindenstrauss transform:
- 115x faster than dense random projection (O(d log d) vs O(d^2))
- +4.5 dB SNR improvement over dense QJL
- Adaptive mode: auto-enables at configurable context length threshold

### 2.5 Compaction (compaction.rs -- 420 lines)

KV cache token reduction via attention matching (Zweiger et al., 2026):
- Select top-t keys by **mean** attention weight across all reference queries
- Fit per-key bias (beta) via NNLS to preserve softmax partition function
- Fit synthetic values via ridge regression (adaptive lambda = 1e-4/n_queries)
- Cholesky-based solvers (no external LAPACK dependency)
- Orthogonal to quantization: 50x token reduction x 8x bit reduction = 400x

### 2.6 Value Compression (in lib.rs)

- `CompressedValues` -- 8-bit absmax per-group quantization
- `CompressedValues4Bit` -- 4-bit symmetric per-group quantization
- `sparse_attn_v_mul_compressed_4bit()` / `_8bit()` -- Fused sparse decompress+multiply

### 2.7 C FFI (ffi.rs + include/tq_kv.h)

Opaque handle API for C/C++ integration:
- `tq_init()` / `tq_free()` -- Single head context
- `tq_compress_and_append()` -- Incremental cache update
- `tq_fused_attention()` -- Score computation from compressed indices
- `tq_layer_*()` -- Multi-head layer API (designed for llama.cpp GQA)

### 2.8 Candle Integration (candle_kv.rs)

`TurboKvCache` -- Drop-in replacement for candle's `KvCache`. Transparently compresses keys and optionally values during forward pass.

### 2.9 Python Bindings (python.rs, feature="python")

PyO3 bindings: `compress_keys`, `decompress_keys`, `fused_attention_scores` accessible from Python.

---

## 3. Engine Crate: tq-engine (src/)

### 3.1 Unified Model Backend (engine.rs -- 550 lines)

`GenericTurboModel` handles ALL model formats (GGUF + safetensors) on ALL devices (CPU + CUDA). No stock candle model paths -- single code path with CUDA-compatible ops.

```rust
ModelWeights(GenericTurboModel)  // Single variant, unified
```

When TQ is disabled, all layers use uncompressed fp16 KV cache through the same code path (skip_layers=999).

### 3.2 GenericTurboModel (models/turbo_generic.rs -- 2,180+ lines)

**The heart of the engine.** Auto-detects architecture from GGUF metadata.

**Supported Architectures:**
- Qwen2 (with attention biases)
- Llama (GQA)
- Mistral (sliding window attention)
- Phi-3/3.5 (merged QKV, padded head_dim)
- Gemma 2 (head_dim=256)

**CUDA-Compatible Custom Ops:**
- `RmsNorm` -- primitive ops only (no missing CUDA kernels)
- `rope_halved()` / `rope_interleaved()` -- both RoPE styles
- `softmax_last_dim()` -- manual exp/sum/div (no candle_nn dependency)

**Forward Attention Path (forward_attn):**

```
x [batch, seq_len, embed_dim]
    |
    +-- Q, K, V = W_q(x), W_k(x), W_v(x)    (QMatMul, quantized matmul)
    |
    +-- [calibration hook: collect raw pre-bias pre-RoPE keys]
    |
    +-- Apply attention biases (Qwen2)
    |
    +-- Reshape to [batch, n_heads, seq_len, head_dim]
    |
    +-- [SmoothAttention: Q *= sqrt(s), K /= sqrt(s)]
    |
    +-- [Pre-RoPE: save k_pre_rope for compression]
    |
    +-- RoPE (halved or interleaved)
    |
    +-- if compressed layer:
    |     +-- Compress K (pre-RoPE or post-RoPE based on config)
    |     +-- Sink tokens (first N) -> stored FP16
    |     +-- POQ: current token -> FP16 in attention, compressed for future
    |     +-- Per-head adaptive bitwidth compression
    |     +-- Store V (fp16 / 8-bit / 4-bit)
    |     +-- Temporal decay (demote old tokens to lower bits)
    |     +-- KV Compaction (when hot cache exceeds threshold)
    |     |
    |     +-- Attention: [sink | cold | compacted | hot | current]
    |     |     +-- if FUSED + CPU (not pre-rope, not compacted):
    |     |     |     Pre-rotate Q, centroid lookup, AVX2+FMA SIMD
    |     |     +-- else:
    |     |           Decompress keys (+apply RoPE if pre-RoPE), matmul
    |     |           Apply beta biases for compacted segment
    |     |
    |     +-- Softmax (f32)
    |     +-- V multiply (standard / sparse / fused-compressed)
    |     +-- Output [batch, n_heads, seq_len, head_dim]
    |
    +-- if uncompressed layer:
          Standard candle-style KV cache + matmul
```

**5-Segment Attention Architecture:**

```
[sink FP16] [cold decayed] [compacted + beta] [hot compressed] [current FP16 (POQ)]
     |            |                |                   |                |
  Always       Lower bits     Synthetic KV       Original bits     Lossless
  lossless    (temporal        + attention        per-head         original
  (Fix 1)      decay)          biases            adaptive          (Fix 2)
```

**KV Compaction Integration:**

When hot tokens exceed `TQ_COMPACT` threshold:
1. Decompress all hot keys (apply RoPE if pre-RoPE mode)
2. Use ALL GQA-mapped query heads as reference (n_rep queries per KV head)
3. Run `compact_head()` per KV head: select top keys, fit beta via NNLS, fit synthetic values via ridge regression
4. Replace oldest hot tokens with compacted representation
5. During attention: compacted keys get beta added to logits, synthetic values used in V multiply

**Runtime Configuration (all via env vars or config struct):**

| Var | Default | Purpose |
|-----|---------|---------|
| TQ_SKIP | 4 | Uncompressed initial layers |
| TQ_SINK | 4 | FP16 sink tokens |
| TQ_VBITS | 0 | Value compression (0/4/8) |
| TQ_FUSED | 0 | Fused attention (CPU only) |
| TQ_SPARSE_V | 1e-6 | Sparse V threshold |
| TQ_PRE_ROPE | 0 | Pre-RoPE key quantization |
| TQ_COMPACT | 0 | Compaction threshold (0=off) |
| TQ_COMPACT_RATIO | 5 | Compaction target (% of original) |
| TQ_DECAY | off | Temporal decay config |
| TQ_LAYER_BITS | -- | Per-layer bit width |
| TQ_HEAD_BITS | -- | Per-head bit width |
| TQ_BIAS_CORRECT | 0 | Softmax bias correction |
| TQ_GROUP | 32 | Group size for sigma |
| TQ_RESIDUAL | 0 | Residual quantization bits |
| TQ_OUTLIER | 0 | Outlier preservation K |
| TQ_NO_CAL | 0 | Disable calibration |

### 3.3 Pre-RoPE Key Quantization

**The Problem:** Standard TurboQuant compresses keys after RoPE application. Post-RoPE keys have position-dependent per-channel statistics (each position gets a different sinusoidal rotation), making codebook fit suboptimal.

**The Solution:** Compress keys BEFORE RoPE. Pre-RoPE keys have consistent, position-independent per-channel statistics -- better match for Lloyd-Max Gaussian assumption.

**Trade-off:** At decode time, keys must be decompressed and RoPE applied dynamically per position. Fused attention is incompatible (RoPE is position-dependent, can't be absorbed into centroid table). Falls back to decompress path automatically.

**Implementation:**
- `k_pre_rope` tensor saved before `apply_rotary_emb()`
- Compression uses `k_pre_rope` instead of post-RoPE `k`
- `decompress_and_apply_rope()` function: decompress + slice cos/sin for stored positions + apply rotation
- Sink keys and current key (POQ) remain post-RoPE (FP16, no decompression needed)

**Results (Qwen2.5 7B Q4_K_M, 2K modern text):**
- Standard TQ: +5.6% PPL delta
- **Pre-RoPE: +3.7% PPL delta** (34% less quality loss, same compression ratio)

### 3.4 Calibration Pipeline (calibrate.rs -- 450 lines)

`tq calibrate <model>` -- collects real key activations and computes:

1. **Key channel bias** (Pre-Rotation Centering) -- per-channel mean from raw W_k*x output
2. **Channel scales** -- SmoothQuant-style outlier equalization
3. **PCA rotation matrix** -- Jacobi eigendecomposition for learned rotation
4. **Calibrated codebooks** (2/3/4-bit) -- Lloyd-Max from real activations
5. **Per-head importance** -- key norm std-dev for adaptive bitwidth
6. **Auto head bits** -- top 50% heads -> 4-bit, rest -> 2-bit

Saved as `~/.tq/models/{name}-{tag}/calibration.json`. Auto-loaded at engine startup.

**Calibration hook** collects pre-bias, pre-RoPE key vectors from `forward_attn()` for clean per-channel statistics.

### 3.5 HTTP Server (serve.rs -- 600 lines)

Hardened axum-based OpenAI-compatible API:
- `POST /v1/chat/completions` -- streaming SSE + non-streaming
- `GET /v1/models` -- list loaded/available models
- `POST /v1/models/load` -- atomic model swap (old model stays during load)
- `GET /v1/models/status` -- ready/loading status
- `GET /health` -- health check
- `POST /infer` -- legacy endpoint

**Safety:**
- All mutex locks use `.map_err()` (no panic on poison)
- Client disconnect stops generation (`tx.is_closed()`)
- Message count validation (max 1000)
- Model path restricted to .gguf/.safetensors/.bin

### 3.6 Model Hub (hub.rs + catalog.rs)

```bash
tq pull qwen2:7b      # Download from HuggingFace
tq list                # Show downloaded models
tq rm qwen2:7b         # Remove model metadata
```

- Catalog: const array of {name, tag, hf_repo, filename, arch, size_gb}
- Metadata stored in `~/.tq/models/{name}-{tag}/metadata.json`
- Auto-download on first use
- Supports: GGUF files + safetensors FP16/BF16

### 3.7 CLI (main.rs -- 900 lines)

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

### 3.8 Chat Templates (chat.rs)

Multi-turn formatting for:
- Llama 3 (`<|begin_of_text|><|start_header_id|>`)
- Qwen / Qwen2 (`<|im_start|>system`)
- Mistral (`[INST]`)
- Phi-3 (`<|system|>`)
- Gemma (`<start_of_turn>`)

---

## 4. The 3-Fix Framework

Solves compound quantization error (W4 weights + KV4 cache) that breaks all other TurboQuant implementations:

**Fix 1: Sink Token Preservation** -- First N tokens' keys stay FP16. Attention sinks get disproportionate weight; quantizing them causes most attention distribution error.

**Fix 2: Past-Only Quantization (POQ)** -- Current token's key uses FP16 original in attention. Compressed into cache for future tokens. Protects highest-impact position.

**Fix 3: Cache State Management** -- Hard reset of compressed KV state per conversation. Prevents cross-conversation contamination from stale RoPE-mismatched keys.

---

## 5. Advanced Techniques

### 5.1 Pre-RoPE Key Quantization (KVQuant approach)

Compress keys BEFORE RoPE application. Pre-RoPE keys have position-independent per-channel statistics -- better Gaussian assumption fit for Lloyd-Max codebook. Reduces PPL delta by 34% on modern text at the same compression ratio. Incompatible with fused attention (auto-fallback to decompress path).

### 5.2 KV Compaction (Zweiger 2026)

Attention-matching token reduction, fully integrated into the inference pipeline:
- Triggers when hot tokens exceed `TQ_COMPACT` threshold (single pass, no cascading)
- Uses ALL GQA-mapped query heads as reference (not just 1)
- Mean-based key scoring (robust across all queries)
- Adaptive ridge regression lambda (1e-4/n_queries)
- Compacted tokens stored with beta biases and synthetic values
- 5-segment attention: [sink | cold | compacted+beta | hot | current]

### 5.3 Pre-Rotation Centering

Subtract per-channel key bias before Hadamard rotation. Addresses weight quantization artifacts on GGUF models. Bias computed from pre-RoPE keys during calibration. Guarded: skips if mean |bias| > 1.0 (RoPE contamination).

### 5.4 SmoothAttention

Migrate K outliers to Q: `Q *= sqrt(s), K /= sqrt(s)`. Mathematically invariant (`Q*K^T` preserved). Currently disabled for Q4_K_M -- channel scales from calibration don't improve quality on already-quantized models.

### 5.5 Per-Head Adaptive Bitwidth

Different KV heads get different bit widths based on attention pattern sensitivity. Sparse heads (single token dominant) -> 2-bit. Diffuse heads -> 4-bit. Calibrated automatically from key norm variance.

### 5.6 Temporal Decay

Auto-demote old tokens to lower bit widths via centroid index remapping. Configurable tiers (e.g., "512:2" = tokens older than 512 -> 2-bit). No decompression needed -- index-level bit remapping.

---

## 6. Measured Results

### 6.1 Perplexity (Qwen2.5 7B Q4_K_M)

**Modern English text (2106 tokens):**

| Configuration | PPL | Delta | Compression |
|:-------------|:---:|:-----:|:-----------:|
| Baseline (no TQ) | 1.823 | -- | 1x |
| Standard TQ 4-bit | 1.925 | +5.6% | ~7.5x |
| **Pre-RoPE 4-bit** | **1.890** | **+3.7%** | ~7.5x |
| TQ + Compact (500t/30%) | 2.227 | +22.2% | ~25x |
| Pre-RoPE + Compact (500t/30%) | 2.281 | +25.1% | ~25x |
| Pre-RoPE + Compact (1000t/20%) | 2.364 | +29.7% | ~25x |

**Multi-length validation (modern English):**

| Tokens | Baseline | Standard TQ (delta) | Pre-RoPE (delta) |
|:------:|:--------:|:-------------------:|:----------------:|
| 475 | 5.117 | 5.820 (+13.7%) | **5.403 (+5.6%)** |
| 793 | 4.901 | 5.250 (+7.1%) | **5.125 (+4.6%)** |
| 2106 | 1.823 | 1.925 (+5.6%) | **1.890 (+3.7%)** |

### 6.2 SRHT QJL Performance (32K vectors, d=128, release)

| Metric | Dense QJL (paper) | SRHT QJL (ours) | No QJL |
|:-------|:-----------------:|:---------------:|:------:|
| Compress overhead | 29x | **1.45x** | 1.0x |
| SNR improvement | +1.2 dB | **+4.5 dB** | -- |
| Attention KL div. | -- | **2.9x lower** | -- |

### 6.3 VRAM Impact

| Model | Context | FP16 KV | TQ 4-bit | TQ 2-bit | Savings |
|:------|:-------:|:-------:|:--------:|:--------:|:-------:|
| Qwen 2.5 7B | 4K | 256 MB | 34 MB | 18 MB | 7.5-14.2x |
| Qwen 2.5 72B | 4K | 640 MB | 85 MB | 45 MB | 7.5-14.2x |
| Llama 3.1 70B | 32K | 20 GB | 2.7 GB | 1.4 GB | 7.5-14.2x |

---

## 7. Build System

```bash
build.bat              # Debug build (Windows + MSVC + CUDA 13.2)
build.bat release      # Release build
build.bat check        # Type check only
build.bat test         # Run tests
```

**Dependencies:**
- `candle-core` 0.9 -- tensor ops, GGUF loading (CUDA feature)
- `candle-transformers` 0.9 -- LogitsProcessor, Sampling only
- `tokenizers` 0.20 -- HuggingFace tokenizer
- `axum` + `tokio` -- async HTTP server
- `hf-hub` -- HuggingFace model download
- `tq-kv` (path) -- our compression crate

**Feature Flags (tq-kv):**
- `default` = `["std"]` -- standard library features
- `candle` -- candle tensor integration + rayon parallelism
- `ffi` -- C FFI layer (`libtq_kv.a` + `tq_kv.h`)
- `python` -- PyO3 Python bindings

---

## 8. Test Coverage

| Category | Count | Description |
|----------|------:|-------------|
| tq-kv unit tests | 77 | Compression, codebook, hadamard, compaction |
| Integration tests | 6 | QJL attention context, long context, compare |
| Doc tests | 2 | API usage examples |
| **Total** | **86** | **0 failures** |

---

## 9. File Map

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
    turbo_generic.rs  Auto-detecting GGUF model + TQ compression + compaction
    mod.rs            Module declaration
  calibrate.rs        Calibration pipeline (codebook, rotation, scales, bias)
  serve.rs            Hardened axum HTTP server (OpenAI API)
  chat.rs             Multi-turn chat templates
  hub.rs              HuggingFace model hub (pull/list/rm)
  catalog.rs          Model catalog (name:tag -> HF repo mapping)
  config.rs           Model architecture detection
  auto_tq.rs          VRAM-aware automatic TQ configuration
  diagnostics.rs      Layer diagnostic utilities
  diagnose.rs         Diagnostic binary
  download.rs         Download progress display
  inference.rs        Legacy inference utilities
  model.rs            Legacy model loading
  web/index.html      Embedded Web UI
```
