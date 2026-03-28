# X (Twitter) Launch Tweets

## Thread 1: The Discovery (Ana Tweet + Thread)

### Tweet 1 (Hook)
We found a problem that affects EVERY TurboQuant implementation.

None of them work on quantized models (GGUF Q4_K_M).

We fixed it. First working TurboQuant on real deployed models.

Paper + code: [link]

1/7

### Tweet 2 (Problem)
The problem: TurboQuant paper assumes FP16 weights.

But nobody runs FP16 anymore. Everyone uses GGUF Q4_K_M.

When you apply TurboQuant to already-quantized models:
- 50 tokens of OK output
- Then gibberish: Turkish + Chinese + French mixed together
- Complete semantic collapse

2/7

### Tweet 3 (Why)
Root cause: compound quantization error.

Weight quant noise + KV quant noise = not additive.

It's MULTIPLICATIVE through softmax attention.

exp(q * k_noisy) / sum(exp(q * k_j_noisy))

Small systematic bias in every key -> softmax amplifies -> attention distribution shifts -> wrong tokens generated.

3/7

### Tweet 4 (Solution)
Our fix -- 3 paper-backed techniques, zero retraining:

1. Sink token preservation (first 4 tokens stay FP16)
   -> 81% error reduction [KVSink, arXiv:2508.04257]

2. Past-Only Quantization (current token = lossless)
   -> highest-impact position protected [WKVQuant]

3. Cache reset (bug nobody caught)
   -> stale conversations contaminating attention

4/7

### Tweet 5 (QJL Insight)
Bonus discovery: everyone says "QJL hurts" (ikawrakow, spiritbuun, scos-lab).

Wrong. DENSE QJL hurts. STRUCTURED QJL helps.

We replaced O(d^2) random projection with SRHT:
- 115x faster
- +4.5 dB SNR (BETTER, not just faster)
- 2.9x lower attention KL divergence at ALL context lengths

The tool was broken, not the idea.

5/7

### Tweet 6 (Numbers)
Results on Qwen 2.5 7B Q4_K_M:

Before fix: "Ataturk'dur pesticale Cumhuri Rencontre"
After fix: "Turkiye Cumhuriyeti'ni kurmus olan kisi Mustafa Kemal Ataturk'tur"

300+ tokens, zero language mixing.

4-bit: 3.8x compression, 0.996 cosine sim
2-bit: 14.2x compression, NIAH 9/9 pass
10K LOC pure Rust, zero C/C++ deps

6/7

### Tweet 7 (CTA)
Paper: [arxiv link]
Code: github.com/onur-gokyildiz-bhi/tq-kv
Crate: crates.io/crates/tq-kv

Full product -- not just a library:
  tq pull qwen2:7b
  tq serve --turbo-quant
  tq chat

"Rust's Ollama" with TurboQuant KV compression.

BHI Research -- @onaborkyildiz

7/7

---

## Thread 2: Technical Deep Dive (for ML audience)

### Tweet 1
Deep dive: why TurboQuant fails on GGUF and how we fixed it.

Key insight from 17 papers surveyed:

"Hadamard rotation only improves concentration, not alignment" (arXiv:2603.04359)

When inputs are already noisy from weight quant, rotation alone isn't enough.

Thread:

### Tweet 2
The paper assumes: k = W_K * x (clean FP16)

Reality: k = W_K_q4 * x + noise_w (quantized weight noise)

After Hadamard: not quite Gaussian. Outlier channels survive.

Lloyd-Max codebook is optimal for Gaussian.
For non-Gaussian from Q4: suboptimal boundaries -> systematic bias.

### Tweet 3
Softmax is the amplifier:

attn_i = exp(q * k_i / sqrt(d)) / sum(exp(q * k_j / sqrt(d)))

Quantization bias in k is NOT random noise.
It's systematic: every key shifts in the same direction.

exp(bias) is multiplicative, not additive.
2000 keys * small bias = large attention shift.

### Tweet 4
The fix is elegant:

Sink tokens (positions 0-3): FP16
  -> These get 50%+ of attention weight in many models
  -> Quantizing them = 81% of total error

Current token: FP16
  -> Most attended recent token stays lossless

Past tokens: TurboQuant 4-bit
  -> Compound error exists but is diluted

### Tweet 5
On QJL -- the contrarian finding:

Community: "QJL adds variance that softmax amplifies"
Us: "That's because you're using DENSE random projection"

SRHT (Hadamard-based) projection:
  - Structured -> lower variance
  - O(d log d) instead of O(d^2)
  - +4.5 dB SNR (dense was +1.2 dB)

The projection quality matters more than the projection speed.

### Tweet 6
We think this opens a new research direction:

"Compound-aware KV cache compression"

The assumption that KV vectors are clean needs to die.
In production, EVERYTHING is quantized.

Next: per-channel key scaling, calibrated codebooks, adaptive QJL threshold tuning.

Paper: [arxiv link]

---

## Standalone Tweets (for different days)

### Standalone 1 (Controversial/Engagement)
Hot take: QJL doesn't hurt. You're just implementing it wrong.

Every TurboQuant implementer: "QJL adds noise, remove it"

Us: Replace dense O(d^2) with structured SRHT O(d log d).

Result: 115x faster AND better quality (+4.5 dB vs +1.2 dB).

The tool was broken. Not the concept.

### Standalone 2 (Practical/Developer)
TIL: No TurboQuant implementation works on GGUF quantized models.

All 7+ implementations (Python, C, CUDA, Rust) only test on FP16.

But everyone deploys Q4_K_M.

We fixed it. 3 lines of code principle, 17 papers of research.

github.com/onur-gokyildiz-bhi/tq-kv

### Standalone 3 (Visual/Numbers)
TurboQuant on GGUF Q4_K_M -- before and after:

BEFORE:
"Ataturk'dur. pesticale Cumhuri Rencontre kukka"

AFTER (our 3-fix):
"Turkiye Cumhuriyeti'ni kurmus olan kisi
 Mustafa Kemal Ataturk'tur. Ataturk'un reformlari
 Turkiye'nin modernlesmesinde onemli rol oynamistir."

300+ tokens. Zero language mixing. Pure Rust.

### Standalone 4 (For Rust community)
Rust + ML: tq-kv ships as a full LLM platform

tq pull qwen2:7b
tq serve --turbo-quant
tq chat

10K LOC. Zero C/C++. SIMD AVX2+FMA.
candle backend. crates.io v0.4.0.

"Rust's Ollama" with 4x memory compression.

@rustlang
