# HN Launch Post

## Title

Show HN: tq-kv -- First TurboQuant that works on GGUF quantized models (Pure Rust)

## URL

https://github.com/onur-gokyildiz-bhi/tq-kv

## First Comment

Author here. tq-kv is a Pure Rust implementation of Google's TurboQuant (ICLR 2026) -- KV cache compression that shrinks transformer memory 4-15x.

**The problem nobody talks about:** Every TurboQuant implementation (Python, C, CUDA, Rust -- there are 7+) assumes FP16 base models. But in practice, everyone runs GGUF quantized models (Q4_K_M). When you apply TurboQuant to already-quantized models, you get garbled output -- language mixing, gibberish within 50 tokens. The compound error from double quantization (W4 weights + KV4 cache) gets amplified through softmax attention.

We found and solved this. Three inference-time fixes, no retraining:

1. **Sink token preservation** -- keep first N tokens' keys at FP16. Attention sinks get disproportionate weight; quantizing them causes 81% of the error (arXiv:2508.04257).

2. **Past-Only Quantization** -- the current token's key stays FP16 during attention. Only past tokens are compressed. Zero memory overhead (adapted from WKVQuant, arXiv:2402.12065).

3. **Cache state management** -- a bug where compressed KV cache was never cleared between conversations. Stale keys from previous contexts contaminated attention. Simple fix, but invisible to per-key quality metrics.

Result: 300+ token coherent multilingual output on Qwen 2.5 7B Q4_K_M, where every other implementation produces gibberish.

**Bonus -- SRHT QJL:** The paper's optional error correction (QJL) uses O(d^2) random projection. We replaced it with SRHT (Subsampled Randomized Hadamard Transform) -- 115x speedup AND +4.5 dB SNR improvement. The community consensus is "QJL hurts" (ikawrakow, spiritbuun, scos-lab all independently found this). Our finding: with SRHT, QJL actually reduces attention KL divergence by 2.9x at all context lengths. Dense QJL hurts; structured QJL helps.

**Numbers:**
- 4-bit: 3.8x compression, 0.996 cosine similarity, +0.8% PPL
- 2-bit: 14.2x compression, NIAH 9/9 pass at all depths
- SRHT QJL: 1.24x overhead (was 29x), +4.5 dB SNR
- Fused attention: AVX2+FMA SIMD, 6x over decompress path
- 10K LOC pure Rust, zero C/C++ deps, crates.io v0.4.0

**Full product (not just a library):**
```
tq pull qwen2:7b          # download from HuggingFace
tq serve --turbo-quant     # OpenAI-compatible API
tq chat qwen2:7b           # terminal chat
```
Web UI at localhost:11435, works with ChatBox and Open WebUI.

Paper draft with all details: [link to arxiv when published]

---

## Anticipated Questions

**Q: How does this compare to llama.cpp's built-in q4/q8 KV cache?**
Different approach. llama.cpp uses simple per-token uniform quantization. TurboQuant uses randomized rotation + optimal codebook, which preserves inner products (attention scores) better. Our 4-bit beats llama.cpp's q4_0 on perplexity while matching compression ratio.

**Q: Why not just use q8_0 KV cache?**
q8_0 gives 2x compression. TurboQuant 4-bit gives 3.8x. At 2-bit, 14.2x. The gap matters for long context: a 32K context on a 70B model needs ~20 GB KV cache at FP16. q8_0 → 10 GB. TQ 4-bit → 5.3 GB. TQ 2-bit → 1.4 GB.

**Q: Does this work on GPU?**
Yes, CUDA support via candle. But the 3-fix currently uses the decompress path (not fused SIMD). GPU fused path is next.

**Q: What's the catch?**
At 600+ tokens, compound error still causes some sentence structure degradation (not language mixing, but repetitive phrasing). Per-channel key scaling (SmoothQuant-style) is the next fix. Also, values are not compressed (keys only, matching the paper).
