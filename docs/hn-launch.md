# HN Launch Post Draft

## Title Options

1. tq-kv: Pure Rust implementation of TurboQuant KV cache compression (ICLR 2026)
2. Show HN: TurboQuant KV cache compression in Rust -- 4-bit with +0.8% PPL
3. Show HN: tq-kv -- Compress LLM KV caches 3.8x at 4-bit with near-zero quality loss
4. tq-kv: 4-bit KV cache compression for LLMs in pure Rust, +0.8% perplexity

## Post Text

URL field: `https://github.com/[org]/tq-kv`

(No body text needed if submitting as a URL post. First comment carries the context.)

## First Comment Draft

Author here. Some context on what this is and why.

**What:** tq-kv is a pure Rust implementation of Google Research's TurboQuant algorithm for compressing the KV cache in transformer inference. The paper was published at ICLR 2026. The library compresses keys using Lloyd-Max optimal codebooks and a fast Walsh-Hadamard transform to decorrelate attention head dimensions before quantization.

**The 4-bit story:** The headline number is 4-bit quantization: 3.8x compression with only +0.8% perplexity increase (9.594 vs 9.515 baseline on wikitext-2, Llama-3 8B) and cosine similarity of 0.996 between original and decompressed keys. This is the mode you'd actually use in production. At 2-bit you get 14.2x compression and it still passes Needle-in-a-Haystack at all 9 tested depths, but perplexity takes a 33.8% hit -- useful for long-context scenarios where you'd otherwise OOM, not for general use.

**Why Rust:** The candle ML framework gives us CUDA tensor ops without a C++ dependency chain. The compression library itself is `no_std` compatible with no allocator requirement for the core path. It builds on Linux, macOS, and Windows without a CMake step. We also ship a C FFI so it can be integrated into existing C/C++ inference engines -- there's a working llama.cpp integration patch.

**Performance:**
- CUDA kernel: 3.2x over CPU for compress/decompress
- AVX2+FMA SIMD: 8.9x speedup in fused dot-product attention
- O(1) incremental cache updates -- new tokens compress independently, no recompress-all
- ~3K lines of library code, MIT/Apache-2.0 dual licensed, on crates.io as v0.3.0

**Comparison to other approaches:** Most KV cache compression work either requires fine-tuning (KIVI, Gear) or operates at the framework level (vLLM's built-in quantization). TurboQuant is training-free and operates at the per-vector level, so it slots in as a library call. We're not aware of another standalone implementation outside Google's internal codebase.

**Honest limitations:**
- The inference engine bundled in the repo (tq-engine) is noticeably slower than llama.cpp for end-to-end generation. It's a demo, not a production server. The compression library is the product.
- 2-bit mode has a real quality cost (33.8% PPL increase). It's there for memory-constrained long-context use, not as a default.
- Values are not compressed (keys only, matching the paper). This caps theoretical max compression.

**What's next:** The main bottleneck now is that decompressed keys go through a generic matmul. A fused SIMD kernel that does quantized attention directly (dot product against compressed keys without full decompression) would close the gap to stock inference speed. That's the current focus.

Happy to answer questions about the algorithm, benchmarks, or integration.
