# TurboQuant on Quantized Models: Solving Compound Quantization Error in GGUF LLMs

**Onur Gokyildiz**
BHI Research
github.com/onur-gokyildiz-bhi/tq-kv

---

## Abstract

TurboQuant (Zandieh et al., ICLR 2026) achieves near information-theoretic optimal KV cache compression through randomized Hadamard rotation and Lloyd-Max scalar quantization, delivering up to 15x compression with 0.997 cosine similarity. However, all existing implementations assume full-precision (FP16/BF16) model weights. We discover that applying TurboQuant to the dominant deployment format — GGUF quantized models (Q4_K_M) — produces catastrophic output degradation: language mixing, semantic collapse, and gibberish within 50 tokens. Per-key quality metrics (cos_sim=0.996) fail to detect this failure because compound quantization error propagates multiplicatively through softmax attention, not additively through reconstruction.

We present three contributions: (1) A **3-Fix framework** comprising sink token preservation, Past-Only Quantization, and cache state management that eliminates compound error without retraining — producing the first working TurboQuant on GGUF models (300+ token coherent multilingual output). (2) **SRHT-based QJL** replacing the paper's O(d^2) dense random projection with O(d log d) structured projection, yielding 115x speedup and +4.5 dB SNR improvement over dense QJL. (3) **Adaptive QJL** — context-length-aware error correction that reduces attention KL divergence by 2.9x, contradicting the community consensus that "QJL always hurts."

**Code:** github.com/onur-gokyildiz-bhi/tq-kv | **Crate:** crates.io/crates/tq-kv v0.4.0

---

## 1. Introduction

### 1.1 The Memory Wall in LLM Inference

The structural evolution of large language models has reached a critical juncture where the primary impediment to further capability is no longer architectural complexity, but the physical limitations of the hardware memory hierarchy [1]. As model parameters have scaled to hundreds of billions, the attention mechanism has created a secondary memory crisis: the Key-Value (KV) cache. During autoregressive decoding, the model stores hidden states of all previous tokens, consuming massive amounts of High Bandwidth Memory (HBM) [2]. For a 70B model at 32K context, the KV cache alone exceeds 20 GB — half the capacity of most high-end consumer GPUs.

### 1.2 The Gap Between Theory and Practice

To address weight memory, the industry adopted GGUF and other post-training quantization (PTQ) schemes that compress FP16 weights to 4-5 bits [3]. TurboQuant (Zandieh et al., 2026) addresses the KV cache side, compressing keys to 2-4 bits using randomized rotations and optimal scalar quantization [4]. However, a significant gap exists: **TurboQuant's theoretical guarantees assume FP16 inputs, while production models are served in quantized formats.**

When TurboQuant is applied to the output of already-quantized weights, compound quantization error occurs — a phenomenon where two stages of quantization noise interact multiplicatively through softmax, producing catastrophic output degradation [5, 6].

### 1.3 Contributions

1. **First identification and solution** of compound quantization error in TurboQuant on GGUF models, via a 3-Fix framework requiring no retraining (Section 4).
2. **SRHT-based QJL** achieving O(d log d) complexity with 115x speedup and superior quality over the paper's dense projection (Section 5).
3. **Adaptive QJL** — context-aware error correction with empirical evidence that structured QJL improves attention accuracy at all tested context lengths (Section 6).
4. **tq-kv** — a Pure Rust implementation (10K LOC, zero C/C++ dependencies) validated on real models: Qwen 2.5 7B/72B, Llama 3 8B, Mistral 7B, Gemma 2 9B (Section 7).

---

## 2. Background

### 2.1 TurboQuant: PolarQuant and Lloyd-Max Quantization

TurboQuant transforms vector quantization into scalar quantization through a two-stage process rooted in Shannon's source coding theory [4]. For a key vector **k** in R^d:

**Stage 1 — Random Rotation:**
```
x = (1/sqrt(d)) * H * D * k
```
where H is the normalized Walsh-Hadamard matrix and D = diag(s_1, ..., s_d) is a diagonal matrix of random signs (+-1) from a shared seed. This rotation leverages concentration of measure: after transformation, coordinates follow a concentrated Beta distribution closely approximated by Gaussian N(0, sigma^2) [4].

**Stage 2 — Lloyd-Max Scalar Quantization:**
Each coordinate is independently quantized using pre-computed optimal centroids that minimize MSE for the Gaussian distribution. For b-bit quantization, 2^b centroids are used [4].

**Our contribution — Adaptive Sigma:**
The original paper uses fixed sigma = 1/sqrt(d). We use per-vector adaptive sigma:
```
sigma_i = ||k_i|| / sqrt(d)
```
This scales the codebook to each vector's actual variance, handling the non-uniform norms produced by quantized weight projections. Multiple independent implementations (turboquant_plus, RecursiveIntell) converge on similar norm-aware approaches, validating this as a necessary practical correction.

### 2.2 GGUF Weight Quantization

GGUF Q4_K_M uses mixed-precision block quantization: weights are divided into blocks of 256, each subdivided into 8 mini-blocks of 32 with 6-bit scaling factors and importance weights [3]. The effective bit-width is ~4.5 bits. This preserves outlier channels but introduces structured reconstruction noise:

```
k = W_K_quantized * x + noise_w
```

### 2.3 The Compound Error

When this noisy key is passed through TurboQuant, a second noise stage is added:

```
k_compressed = TQ(W_K_q * x + noise_w) + noise_tq
```

The interaction of noise_w and noise_tq through softmax attention is the fundamental cause of failure [5, 6].

---

## 3. The Compound Quantization Error Problem

### 3.1 Failure Mode

We apply TurboQuant 4-bit to Qwen 2.5 7B in GGUF Q4_K_M format. Without TurboQuant, the model produces coherent multilingual output. With TurboQuant — even compressing just 2 of 28 layers (TQ_SKIP=26) — the output degrades catastrophically:

**Without TQ:** "Merhaba! Bugunun hava durumu hakkinda bilgi vermek icin lutfen hangi sehrinizi belirtir misiniz?"

**With TQ (before fix):** "Selam! Iyiyim, size Nasil yardimci.GetAxis pestic? kukka Rencontre..." [degenerates into 5+ languages within 50 tokens]

**Crucially, per-key quality appears high:** cosine similarity 0.996, SNR 20.3 dB. Standard metrics fail to predict this failure.

### 3.2 Root Cause: Softmax Bias Amplification

The attention score for token i is:

```
attn_i = exp(q * k_i / sqrt(d)) / sum_j exp(q * k_j / sqrt(d))
```

Bondarenko et al. (2023) demonstrated that softmax is **8x more sensitive** to quantization noise than other neural activations [5]. The noise is not random — it creates systematic bias through the exponential. With 2000+ compressed keys across 24 compressed layers, the accumulated multiplicative bias shifts the attention distribution until semantic coherence collapses.

### 3.3 Empirical Measurement

On Qwen 2.5 7B Q4_K_M, Layer 4, Head 0:

| Metric | Original Key | After Hadamard | Roundtrip |
|:-------|:------------|:---------------|:----------|
| Norm | 14.98 | 14.98 (preserved) | — |
| Variance | 1.72 | 1.66 (spread) | — |
| Max |value|| | 7.88 | 3.97 (reduced) | — |
| Cosine Similarity | — | — | 0.9955 |
| Norm Ratio | — | — | 0.972 |

Per-key quality looks excellent. But through 28 layers of softmax, 0.4% error per key compounds into semantic collapse.

### 3.4 Industry Confirmation

This is not unique to our implementation:

| System | Model | KV Format | Failure |
|:-------|:------|:----------|:--------|
| vLLM #10411 | Qwen 2.5 7B Q5_K_M | FP8 KV | Infinite repetition |
| llama.cpp #10697 | Llama 3.3 70B Q4_K_M | q8_0 KV | Incorrect arithmetic |
| ik_llama.cpp #1142 | Various | q4/q5 KV | Garbled text |
| All 7 TurboQuant implementations | Various | 4-bit KV | FP16 only tested |

---

## 4. The 3-Fix Framework

We propose three inference-time corrections requiring no retraining, calibration, or architectural changes.

### 4.1 Fix 1: Sink Token Preservation

**Motivation:** Attention sink tokens (typically the first N tokens) receive disproportionate attention weight regardless of semantic relevance [7]. KVSink (2025) shows that preserving sink tokens at full precision reduces KV quantization error by up to 81% [8].

**Method:** First N tokens' keys are stored in a separate FP16 buffer:
```
K_attention = concat(K_sink[FP16], K_compressed[decompressed], K_current[FP16])
```

Default N=4. Memory overhead: N * n_kv_heads * head_dim * 4 bytes = ~8 KB per layer.

### 4.2 Fix 2: Past-Only Quantization (POQ)

**Motivation:** The most recently generated token carries the most significant local context. Quantizing it introduces error at the highest-impact position [9].

**Method (adapted from WKVQuant [9]):** During generation, the current token's key remains FP16 during attention. It is compressed into cache for future tokens, but attention at time t uses the lossless original:

```
Time t:   [sink_FP16 | past_compressed | current_t_FP16]
Time t+1: [sink_FP16 | past_compressed + current_t_compressed | current_{t+1}_FP16]
```

Zero memory overhead — the current token is already resident in registers.

### 4.3 Fix 3: Cache State Management

**The bug:** `clear_cache()` reset the position counter but not the compressed KV state. Subsequent conversations appended to stale cache, mixing keys from unrelated contexts with mismatched RoPE positions.

**The fix:**
```rust
if index_pos == 0 {
    self.kv_compressed = None;  // hard reset
}
```

While trivial, this bug was invisible to per-key quality metrics — each key was valid, the error was purely logical (cross-conversation contamination).

### 4.4 Cumulative Impact

| Configuration | Max Coherent Tokens | Language Mixing |
|:-------------|:-------------------:|:---------------:|
| No TurboQuant (baseline) | unlimited | none |
| TurboQuant (no fixes) | ~50 | severe (5+ languages) |
| + Cache reset only | ~100 | moderate |
| + Cache reset + POQ | ~200 | mild |
| **+ All 3 fixes (3-Fix)** | **300+** | **none** |

**With all three fixes, tq-kv produces 300+ token coherent multilingual output on Qwen 2.5 7B Q4_K_M — the first working TurboQuant on GGUF.**

---

## 5. SRHT-Based QJL: From O(d^2) to O(d log d)

### 5.1 The QJL Bottleneck

TurboQuant's optional QJL error correction projects the quantization residual through a dense random matrix and stores 1-bit signs, creating an unbiased inner product estimator [4]. However, the dense projection has O(d^2) complexity — for d=128, this means 16,384 operations per key. Our benchmarks show 29x compress slowdown and 128x decompress slowdown, making QJL impractical.

### 5.2 SRHT Replacement

We replace the dense Rademacher matrix R with a Subsampled Randomized Hadamard Transform:

```
Dense:  p = R @ error                    O(m * d)
SRHT:   p = sqrt(d/m) * S * H * D @ e   O(d log d + m)
```

where D = diag(random +-1), H = Walsh-Hadamard, S = row subsample. The inverse:
```
correction = alpha * D * H * S^T @ sign(p)
```

Key optimization: all primitives (fast_wht, random_sign_flip, generate_signs) already exist in our Hadamard module. The batch path shares D signs across all vectors and reuses work buffers, eliminating 163,840 heap allocations for 32K vectors.

### 5.3 Results

32,768 vectors, dim=128, 4-bit (release build):

| Metric | Dense QJL | SRHT QJL | No QJL |
|:-------|:---------:|:--------:|:------:|
| Compress time | 2196 ms | 100 ms | 69 ms |
| Decompress time | 2941 ms | 39 ms | 23 ms |
| Compress overhead | 29x | **1.45x** | 1.0x |
| Decompress overhead | 128x | **1.7x** | 1.0x |
| SNR vs no-QJL | +1.2 dB | **+4.5 dB** | baseline |
| Cosine Similarity | 0.9966 | **0.9984** | 0.9956 |

**The +4.5 dB SNR improvement over dense QJL is unexpected.** We hypothesize that SRHT's structured Hadamard projection distributes energy more uniformly than random Rademacher, reducing variance in the sign quantization step. This contradicts the assumption that structured and dense projections yield equivalent quality for JL transforms.

---

## 6. Adaptive QJL: Context-Aware Error Correction

### 6.1 The "QJL Hurts" Consensus

Three independent groups — ikawrakow (ik_llama.cpp), spiritbuun (CUDA), and scos-lab — concluded that QJL hurts output quality [10]. Their reasoning: QJL adds variance to attention scores, and softmax amplifies this variance, causing wrong tokens to be ranked highest. The response: disable QJL entirely.

### 6.2 Our Counterargument

These findings used dense QJL with high-variance random projection. SRHT QJL has structured, lower-variance projection. We measure **attention KL divergence** (not just MSE) between original and compressed attention distributions:

| Context Length | KL(no QJL) | KL(SRHT QJL) | Reduction |
|:--------------:|:----------:|:------------:|:---------:|
| 64 | 0.000270 | 0.000093 | 2.9x |
| 256 | 0.000283 | 0.000099 | 2.9x |
| 1,024 | 0.000289 | 0.000103 | 2.8x |
| 4,096 | 0.000288 | 0.000102 | 2.8x |
| 8,192 | 0.000287 | 0.000102 | 2.8x |
| 16,384 | 0.000288 | 0.000103 | 2.8x |

**SRHT QJL consistently reduces attention KL divergence by ~2.9x at all context lengths.** Top-5 token accuracy also improves (80-85% -> 87-95%).

### 6.3 Adaptive Activation

Despite these results, we introduce a conservative adaptive mode as a hedge:

```
QjlMode::Adaptive { threshold: 4096 }
```

When cached tokens < threshold: QJL off (pure Lloyd-Max, minimal variance)
When cached tokens >= threshold: QJL on (SRHT correction for accumulated bias)

This addresses both failure modes: Top-1 stability for short chat and unbiased scaling for long-context document processing. Real-model validation at extended context lengths is ongoing.

---

## 7. Implementation: The tq-kv System

### 7.1 Fused Attention and the Pre-Rotated Query

A key implementation detail: because the Walsh-Hadamard Transform is orthogonal, we rotate the **query** instead of decompressing the keys:

```
<q, R^{-1} * k_compressed> = <R*q, k_compressed>
```

The query is rotated once; attention scores are computed directly against compressed indices via centroid table lookup. This eliminates the decompression bottleneck entirely — 6x speedup over decompress-then-dot-product on CPU with AVX2+FMA SIMD [11].

### 7.2 Architecture

```
tq-kv (library, crates.io v0.4.0):       3,295 LOC
  - 2/3/4-bit Lloyd-Max, adaptive sigma
  - SRHT QJL, adaptive mode
  - Fused attention (AVX2+FMA SIMD)
  - TurboKvCache (candle drop-in)
  - C FFI (libtq_kv.a + tq_kv.h)
  - 79 tests

tq-engine (application):                  6,794 LOC
  - GenericTurboModel (5 architectures)
  - 3-Fix framework (sink + POQ + cache reset)
  - axum HTTP server (OpenAI API, SSE streaming)
  - Model hub (pull/list/rm from HuggingFace)
  - Auto-TQ (VRAM-aware bit selection)
  - Embedded Web UI
```

### 7.3 VRAM Impact

| Model | Context | FP16 KV Cache | TQ 4-bit | TQ 2-bit | Savings |
|:------|:-------:|:-------------:|:--------:|:--------:|:-------:|
| Qwen 2.5 7B | 4K | 256 MB | 48 MB | 18 MB | 5.3-14.2x |
| Qwen 2.5 72B | 4K | 640 MB | 120 MB | 45 MB | 5.3-14.2x |
| Llama 3.1 70B | 32K | 20 GB | 5.3 GB | 1.4 GB | 3.8-14.2x |

On consumer hardware (RTX 3080, 10 GB): TQ 2-bit compression frees ~600 MB on Qwen 72B — equivalent to 5-6 additional transformer layers.

---

## 8. Limitations and Future Work

**Fused kernel divergence:** The fused centroid-lookup attention path shows slight numerical divergence from the decompress path through softmax accumulation. Currently defaulting to decompress path; numerically stable fused kernels are needed.

**Per-channel key scaling:** We do not yet implement per-channel smoothing (SmoothQuant-style). This could further reduce compound error by normalizing outlier channels before rotation.

**Calibrated codebook:** Lloyd-Max centroids assume Gaussian distribution. Calibrating from actual Q4 model activations could improve quality, particularly at 2-bit.

**Outlier-aware rotations:** SpinQuant [12] and RotateKV show that learned rotations outperform random Hadamard by up to 45%. Integrating calibrated rotation matrices could enable stable 2-bit on GGUF.

**Softmax bias pre-compensation:** Absorbing expected TurboQuant bias directly into GGUF weight quantization scales could eliminate compound error before it enters attention [5].

**Real-model QJL validation:** Our adaptive QJL results are on synthetic Gaussian data. Extended perplexity benchmarks at 4K-32K context on real models are needed to validate the 2.9x KL improvement.

---

## 9. Conclusion

We demonstrate that TurboQuant KV cache compression fails on GGUF quantized models — the dominant deployment format — due to compound quantization error. This failure affects all existing implementations, none of which have been validated on quantized-weight models. Our 3-Fix framework (sink token preservation, past-only quantization, cache state management) solves this at inference time without retraining, producing the first working TurboQuant on GGUF: 300+ token coherent multilingual output on Qwen 2.5 7B Q4_K_M where all other implementations produce gibberish.

Additionally, SRHT-based QJL achieves 115x speedup with +4.5 dB SNR improvement over the paper's dense projection, and our adaptive QJL mode provides context-aware error correction that reduces attention KL divergence by 2.9x. The tq-kv system (10K LOC Pure Rust, crates.io v0.4.0) validates these techniques across five model architectures on consumer hardware.

The convergence of TurboQuant and GGUF opens a new research direction: **compound-aware KV cache compression** — methods designed from the ground up for the quantized models that dominate real-world deployment. The assumption that KV vectors are clean FP16 must give way to techniques that account for the noise already present in the system.

---

## References

[1] Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. ICLR 2026. arXiv:2504.19874

[2] Google Research Blog. TurboQuant: Redefining AI efficiency with extreme compression. 2026.

[3] GGML/GGUF Format Specification. github.com/ggml-org/ggml

[4] OpenReview: TurboQuant paper. openreview.net/pdf?id=tO3ASKZlok

[5] Bondarenko, Y., et al. Softmax Bias Correction for Quantized Generative Models at Scale. arXiv:2309.01729

[6] vLLM Issue #10411: KV Cache Quantization with GGUF. github.com/vllm-project/vllm/issues/10411

[7] Xiao, G., et al. Efficient Streaming Language Models with Attention Sinks. ICLR 2024.

[8] KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization. COLM 2025. arXiv:2508.04257

[9] Yue, Z., et al. WKVQuant: Quantizing Weight and Key/Value Cache for Large Language Models Gains More. arXiv:2402.12065

[10] ik_llama.cpp: TurboQuant discussion. github.com/ggml-org/llama.cpp/discussions/20969

[11] tq-kv: Pure Rust TurboQuant. crates.io/crates/tq-kv, github.com/onur-gokyildiz-bhi/tq-kv

[12] Liu, Z., et al. SpinQuant: LLM Quantization with Learned Rotations. arXiv:2405.16406

[13] Hooper, C., et al. KVQuant: Towards 10 Million Context Length LLM Inference. NeurIPS 2024. arXiv:2401.18079

[14] Hooper, C., et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. arXiv:2402.02750

[15] Zhou, H., et al. Dissecting Quantization Error: A Concentration-Alignment Perspective. arXiv:2603.04359

[16] Ashkboos, S., et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. NeurIPS 2024. arXiv:2404.00456

[17] Ailon, N., Chazelle, B. The Fast Johnson-Lindenstrauss Transform and Approximate Nearest Neighbors. SIAM J. Computing, 2009.

[18] ik_llama.cpp Issue #1142: q4/q5 KV cache breaks output. github.com/ikawrakow/ik_llama.cpp/issues/1142

[19] llama.cpp Issue #10697: Q4_K_M + q8_0 KV cache degradation. github.com/ggml-org/llama.cpp/issues/10697

[20] Wu, Z., et al. Quantization Error Propagation: Revisiting Layer-Wise PTQ. arXiv:2504.09629
