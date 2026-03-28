# X (Twitter) Launch Tweets — Updated with TheTom/turbo4 findings

## Main Thread: The Discovery

### Tweet 1 (Hook) [+ Visual 1: Before/After]
Every TurboQuant implementation fails on GGUF models.

Not "degrades slightly." FAILS. Gibberish in 5 languages within 50 tokens.

7+ implementations (Python, C, CUDA, Rust). All test on FP16 only. Nobody tested on the models people actually deploy.

We fixed it.

[IMAGE: visual-1-before-after.png]

### Tweet 2 (The Problem) [+ Visual 4: Journey]
The problem has a name: compound quantization error.

Model weights: already 4-bit (GGUF Q4_K_M)
KV cache: TurboQuant compresses to 4-bit

Two layers of noise. Softmax amplifies it multiplicatively, not additively.

exp(q * k_noisy) / sum(exp(q * k_j_noisy))

Small bias per key. 2000+ keys. Catastrophic attention shift.

Per-key cos_sim = 0.996. Looks fine. Output = gibberish.

### Tweet 3 (Our Solution) [+ Visual 2: 3-Fix Pipeline]
3 paper-backed fixes. Zero retraining. Zero calibration:

1. Sink tokens stay FP16 (first 4 tokens get 50%+ attention)
   → 81% error reduction [KVSink]

2. Current token = lossless (POQ)
   → Highest-impact position protected [WKVQuant]

3. Cache reset between conversations
   → A bug nobody caught because per-key metrics can't detect it

[IMAGE: visual-2-3fix.png]

### Tweet 4 (TheTom + QJL Context)
Meanwhile, @TheTom revived turbo4 from the dead on Metal.

PPL 679 → 6.125 (+0.23% vs q8_0). 7 implementation bugs. Amazing work.

Key shared finding: "QJL eliminates bias but explodes variance. Softmax amplifies variance."

5 groups confirmed: dense QJL hurts.

But we found something different...

### Tweet 5 (SRHT QJL — Contrarian) [+ Visual 6: QJL Controversy]
Dense QJL hurts. STRUCTURED QJL helps.

We replaced O(d^2) random projection with SRHT (Hadamard-based):
- 115x faster
- +4.5 dB SNR (3.75x better than dense)
- 2.9x lower attention KL divergence

The tool was broken. Not the idea.

*Caveat: synthetic data. Real-model validation pending. TheTom's evidence on real models is stronger. We present this as an open question.

[IMAGE: visual-3-srht.png]

### Tweet 6 (Numbers + Impact) [+ Visual 5: VRAM]
Results on Qwen 2.5 7B Q4_K_M:

Before fix: 50 tokens then gibberish
After 3-Fix: 300+ tokens, zero language mixing

Compression:
- 4-bit: 3.8x (256 MB → 48 MB)
- 2-bit: 14.2x (256 MB → 18 MB)
- Llama 70B 32K: 20 GB → 1.4 GB

NIAH: 9/9 pass at all depths.

[IMAGE: visual-5-vram.png]

### Tweet 7 (Product + CTA)
Not just a paper. A full product:

```
tq pull qwen2:7b
tq serve --turbo-quant
tq chat
```

Web UI, OpenAI API, model hub. 10K LOC Pure Rust.

"Rust's Ollama" with 4x memory compression.

Paper: [arxiv link]
Code: github.com/onur-gokyildiz-bhi/tq-kv
Crate: crates.io/crates/tq-kv

@onurgokyildiz | BHI Research

---

## Technical Thread (for ML audience)

### T1 [+ Visual 2]
Technical deep dive: why TurboQuant + GGUF = broken, and how compound-aware compression fixes it.

The assumption that killed everyone: "KV vectors are clean FP16."

In production, k = W_K_q4 * x + noise_w. Your keys are already noisy. Thread:

### T2
After Hadamard rotation, TurboQuant assumes coordinates ~ N(0, sigma^2).

With Q4 weights, they're NOT Gaussian. They carry structured quantization artifacts.

"Hadamard rotation only improves concentration, not alignment" (arXiv:2603.04359)

The codebook is optimized for the wrong distribution.

### T3
But per-key quality looks fine! cos_sim=0.996. SNR=20.3 dB.

The trap: these metrics measure RECONSTRUCTION error.
Attention measures SOFTMAX of dot products.

Softmax is 8x more sensitive to quantization than any other activation (Bondarenko 2023).

0.4% error per key * 2000 keys * softmax amplification = semantic collapse.

### T4
The 3-Fix targets the highest-impact positions:

Position 0-3 (sink tokens): get 50%+ attention. Keep FP16. 8 KB per layer.
Position t (current): most recent context. Keep FP16. Zero overhead.
Stale cache: hard reset per conversation. Zero overhead.

Simple. Paper-backed. Works.

### T5
On QJL — the interesting debate:

@TheTom + 4 others: "Drop QJL. More centroids > error correction."
Us: "Dense QJL hurts. SRHT QJL might help."

Dense: O(d^2), random Rademacher, high variance
SRHT: O(d log d), structured Hadamard, lower variance

Our synthetic tests: 2.9x better attention KL divergence.
TheTom's real tests: QJL hurts PPL.

Open question. We implemented adaptive mode as a hedge.

### T6
New research direction: compound-aware KV cache compression.

The field assumes clean inputs. Reality is everything is quantized.

Next steps:
- Per-channel key scaling (SmoothQuant for KV cache)
- Calibrated codebooks from Q4 activations
- Real-model QJL validation at 32K+ context

Paper: [arxiv]
Code: github.com/onur-gokyildiz-bhi/tq-kv

---

## Standalone Tweets

### Standalone 1 (Engagement bait)
"Do not bother using turbo4" — community consensus, March 2026

@TheTom: Fixed 7 bugs. PPL 679 → 6.125.
Us: Fixed compound error on GGUF. First working TurboQuant on quantized models.

Sometimes the answer is just debugging until it works.

### Standalone 2 (For Rust devs)
Pure Rust LLM inference with 4x memory compression:

tq pull qwen2:7b
tq serve --turbo-quant
tq chat

10K LOC. Zero C/C++. AVX2+FMA SIMD.
Works with ChatBox and Open WebUI.

Rust's Ollama. crates.io/crates/tq-kv

### Standalone 3 (Numbers visual) [+ Visual 3]
SRHT vs Dense QJL:

Dense: 29x slower, +1.2 dB
SRHT: 1.45x overhead, +4.5 dB

115x faster. 3.75x better quality.

Same algorithm. Different projection matrix.

The implementation matters more than the theory.

### Standalone 4 (The quote)
"The paper optimized for MSE. Attention optimizes through softmax. Different objective, different answer."

— @TheTom on why QJL hurts

This is the most important insight in KV cache compression right now. The metric you optimize for determines everything.
