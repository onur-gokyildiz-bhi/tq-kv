# Candle CUDA Workaround Patches

## Problem

Several `candle-nn` operations lack CUDA kernel implementations as of candle 0.9.x.
When running quantized LLM inference on GPU, the following functions panic with
"no cuda implementation for ..." errors:

1. **`candle_nn::ops::rms_norm`** -- RMS normalization used in every transformer layer
2. **`candle_nn::ops::softmax_last_dim`** -- softmax used in attention scoring
3. **`candle_nn::rotary_emb::rope_i`** -- interleaved rotary position embeddings

## Solution

Each patch file contains a pure-Rust reimplementation using only basic tensor
operations that already have CUDA dispatch (`sqr`, `mean_keepdim`, `sqrt`,
`broadcast_div`, `broadcast_mul`, `exp`, `max_keepdim`, `sum_keepdim`, `narrow`,
`reshape`, `cat`, etc.). No custom CUDA kernels are needed -- the existing
elementwise and reduction kernels compose to give correct results on any device.

## Files

| File | Replaces | Location in candle |
|------|----------|--------------------|
| `cuda_rms_norm.rs` | `candle_nn::ops::rms_norm` | `candle-nn/src/ops.rs` |
| `cuda_softmax.rs` | `candle_nn::ops::softmax_last_dim` | `candle-nn/src/ops.rs` |
| `cuda_rope.rs` | `candle_nn::rotary_emb::rope_i` | `candle-nn/src/rotary_emb.rs` |

## How to Apply

1. Fork `huggingface/candle` and check out the latest `main` branch.
2. For each patch, integrate the function body into the corresponding candle-nn
   module, replacing or augmenting the existing implementation.
3. The key insight: these are not new CUDA kernels. They decompose each operation
   into primitives that candle already supports on CUDA. The existing functions
   should be updated to use these decompositions as a fallback when no native
   CUDA kernel is available.
4. Run `cargo test --workspace` in the candle repo to verify nothing breaks.
5. Open a PR per patch (or one combined PR) with a clear description referencing
   the missing CUDA dispatch.

## Origin

Discovered while building [tq-kv](https://github.com/onur-gokyildiz/tq-kv), a
TurboQuant KV cache compression engine. The workarounds are duplicated in
`src/models/turbo_llama.rs` and `src/models/turbo_qwen2.rs`.
