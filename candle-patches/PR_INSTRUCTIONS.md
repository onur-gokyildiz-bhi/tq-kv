# candle Upstream PR Instructions

## How to submit (manual — gh CLI not available)

### Step 1: Fork candle
1. Go to https://github.com/huggingface/candle
2. Click "Fork" → create `onur-gokyildiz-bhi/candle`

### Step 2: Clone and branch
```bash
git clone https://github.com/onur-gokyildiz-bhi/candle.git
cd candle
git checkout -b cuda-compatible-ops
```

### Step 3: Apply patches

**RmsNorm** → `candle-nn/src/ops.rs`
- Copy `RmsNorm` struct + `Module` impl from `cuda_rms_norm.rs`
- Add as `RmsNormCuda` or replace existing `RmsNorm` with CUDA-compatible version
- Add tests to `candle-nn/tests/ops_test.rs`

**Softmax** → `candle-nn/src/ops.rs`
- Copy `softmax_last_dim` from `cuda_softmax.rs`
- Replace existing implementation or add as `softmax_last_dim_cuda`
- Add tests

**RoPE** → `candle-nn/src/rotary_emb.rs`
- Copy `rope_i` from `cuda_rope.rs`
- Replace existing `rope_i` or add as `rope_i_cuda`
- Add tests

### Step 4: Test
```bash
cargo test --workspace
```

### Step 5: PR
Title: `feat: CUDA-compatible RmsNorm, softmax, and RoPE implementations`

Body:
```
## Problem

Three ops in candle-nn panic when called on CUDA tensors:
- `RmsNorm::forward` → "no cuda implementation for rms-norm"
- `softmax_last_dim` → "no cuda implementation for softmax"
- `rope_i` → "no cuda implementation for rope_i"

This blocks GGUF quantized model inference on GPU (e.g., Llama-3, Qwen2).

## Solution

Decompose each op into primitive tensor operations that all have CUDA kernels:
- sqr, mean_keepdim, sqrt, broadcast_div, broadcast_mul (RmsNorm)
- max_keepdim, broadcast_sub, exp, sum_keepdim (softmax)
- reshape, narrow, squeeze, broadcast_mul, unsqueeze, cat (RoPE)

Math is identical to the originals. Computation in f32 for stability, cast back.

## Evidence

These implementations have been running in production in [tq-kv](https://github.com/onur-gokyildiz-bhi/tq-kv) with:
- Llama-3 8B: 3.2x CUDA speedup over CPU
- Qwen2.5 72B: correct output on 80-layer model
- Benchmarked on RTX 3080, CUDA 13.2
```

## Fallback: candle-tq crate

If PRs are not accepted within 2 weeks, publish `candle-tq` on crates.io:
- Wraps `candle-core` + `candle-nn` + `candle-transformers`
- Re-exports everything with CUDA-compatible ops patched in
- Users: `cargo add candle-tq` instead of `candle-nn`

The candle-tq crate code is in the tq-engine's model files (turbo_llama.rs, turbo_qwen2.rs).
