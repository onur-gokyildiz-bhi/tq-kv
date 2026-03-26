# Candle Upstream PR Checklist

## Prerequisites

- [ ] Fork `huggingface/candle` on GitHub
- [ ] Clone fork and check out latest `main`
- [ ] Verify local build: `cargo build --workspace`
- [ ] Verify local tests: `cargo test --workspace`

## RmsNorm (cuda_rms_norm.rs)

- [ ] Add CUDA-compatible `RmsNorm::forward` to `candle-nn/src/ops.rs`
- [ ] Keep existing API (`RmsNorm::new`, `Module` impl) -- only change internals
- [ ] Add `from_qtensor` constructor that dequantizes to target device
- [ ] Add unit tests from `cuda_rms_norm.rs` to candle-nn test suite
- [ ] Verify: `cargo test -p candle-nn`

## Softmax (cuda_softmax.rs)

- [ ] Replace `softmax_last_dim` body in `candle-nn/src/ops.rs` with max-subtract-exp-div pattern
- [ ] Add generalized `softmax(x, dim)` variant
- [ ] Add unit tests from `cuda_softmax.rs` to candle-nn test suite
- [ ] Verify: `cargo test -p candle-nn`

## RoPE (cuda_rope.rs)

- [ ] Replace `rope_i` body in `candle-nn/src/rotary_emb.rs` with reshape-narrow-broadcast pattern
- [ ] Preserve existing function signature
- [ ] Add unit tests from `cuda_rope.rs` to candle-nn test suite
- [ ] Verify: `cargo test -p candle-nn`

## Final Steps

- [ ] Run full candle test suite: `cargo test --workspace`
- [ ] Test with a CUDA device if available (the whole point of these patches)
- [ ] Open PR(s) with clear description referencing missing CUDA dispatch
- [ ] Link to tq-kv as real-world evidence of the issue
