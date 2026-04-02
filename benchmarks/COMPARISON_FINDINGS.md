# tq-kv vs turboquant_plus: Comparative Analysis

Date: 2026-04-02
Test script: `benchmarks/compare_tqplus.py`

## QJL Disagreement — RESOLVED

The central scientific disagreement: 5 independent groups (including turboquant_plus) say "QJL hurts quality." tq-kv says SRHT QJL helps (+4.5 dB SNR).

**Both sides are correct.** The issue is projection type, not QJL itself.

| Method | 3-bit SNR | 4-bit SNR | vs No QJL |
|:-------|----------:|----------:|:---------:|
| No QJL (PolarQuant only) | 13.9 dB | 18.4 dB | baseline |
| Dense QJL (turboquant_plus) | 7.1 dB | 12.0 dB | **-6.5 dB** |
| SRHT QJL (tq-kv) | 14.7 dB | 19.2 dB | **+0.8 dB** |

Dense random Gaussian projection (d x d matrix) adds high variance to sign quantization.
Structured Hadamard projection distributes energy uniformly, reducing variance.

**Impact on attention (softmax):**
- Dense QJL: 3.5x worse KL divergence, top-1 accuracy 56% -> 34%
- PolarQuant only: baseline KL divergence, 56% top-1

turboquant_plus correctly dropped dense QJL. tq-kv correctly replaced it with SRHT.
The optimal approach is SRHT QJL, not no QJL.

## Rotation: Single-Sign vs Double-Sign

turboquant_plus uses D2 @ H @ D1 (two random sign vectors).
tq-kv uses (1/sqrt(d)) * H @ D (one sign vector).

**Result: No measurable difference.**
- Kurtosis: -0.010 vs -0.010 (identical, both near Gaussian ideal of 0)
- SNR: identical at 4-bit
- Cosine: identical

Single-sign is sufficient. Double-sign adds no quality benefit.

## Norm Correction

turboquant_plus normalizes to unit norm, stores norms, and renormalizes after decompression.
tq-kv uses adaptive sigma per vector (sigma_i = ||k_i|| / sqrt(d)).

**Result: Equivalent.**
- Norm correction gives perfect norm preservation but marginally lower SNR at 2-bit
- Without norm correction: higher SNR but ~6% norm error at 2-bit
- Cosine similarity: identical in both cases
- Both approaches are valid

## Boundary Layer Protection — ADOPTED from turboquant_plus

turboquant_plus discovered that last layers are disproportionately sensitive:
- Last 8 layers account for ALL quality loss on Qwen3.5-35B MoE
- Boundary V (first 2 + last 2 layers at q8_0): recovers 55-62% of quality gap
- Both first AND last layers matter
- Gains dilute at 16K+ context

tq-kv previously only protected first N layers (TQ_SKIP=4). Now added:
- `TQ_PROTECT_LAST` env var (default: 0)
- `protect_last_layers` config field
- Boundary protection: first N + last M layers uncompressed

**Recommended config:** `TQ_SKIP=4 TQ_PROTECT_LAST=2` for pure-attention models.

## Integration Opportunities

| From turboquant_plus | Status | Priority |
|:---------------------|:-------|:---------|
| Boundary layer protection | **IMPLEMENTED** (`TQ_PROTECT_LAST`) | Done |
| Sparse V dequant (+22.8% decode) | Exists as `TQ_SPARSE_V` | Compare |
| Asymmetric q8_0-K + compaction | Not tested | High |
| Layer-adaptive gradient (turbo2 early -> turbo3 mid -> q8_0 late) | Possible via `TQ_LAYER_BITS` | Medium |

| From tq-kv to turboquant_plus | Status | Priority |
|:-------------------------------|:-------|:---------|
| SRHT QJL (replace dropped dense QJL) | Available in tq-kv crate | High |
| KV Compaction (token reduction) | Not in turboquant_plus | High |
| Pre-RoPE key quantization | Not applicable (they don't compress K) | N/A |

## Key Numbers

| Metric | turboquant_plus | tq-kv |
|:-------|:----------------|:------|
| K compression | q8_0 (2x) | 4-bit (7.5x) |
| V compression | turbo2-turbo4 (3.8-6.4x) | V4/V8 (2-3.2x) |
| QJL | Dropped | SRHT (+0.8 dB) |
| Boundary protection | First 2 + last 2 | First 4 + last N (new) |
| Platform | Metal (Mac) | CUDA (NVIDIA) |
| Language | Python + C (llama.cpp fork) | Rust |
| Stars | 4,785 | — |
| Models tested | 1.5B - 122B | 0.5B - 72B |
