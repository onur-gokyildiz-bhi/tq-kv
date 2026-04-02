# tq-kv Roadmap — Benchmark-Driven Priorities

Date: 2026-04-02
Based on: `benchmarks/full_comparison.py` (10 tests, tq-kv vs turboquant_plus)

---

## Benchmark Evaluation

### What the numbers say

**Algoritmik avantajlarımız kanıtlandı:**
- SRHT QJL: +7.7 dB SNR, 4.4x daha iyi attention KL — rakipsiz
- 4-bit K on GGUF: +3.7% PPL vs onların +6.64% — daha fazla sıkıştırma, daha iyi kalite
- Non-integer bits: +6.3 dB at 2.5-bit — OutlierQuant'ı eziyor
- Fused attention: 8.9x speedup — hiçbir framework'te yok

**Ama algoritmik avantaj ≠ ürün avantajı:**
- 2 model test ettik, onlar 7+ (122B'ye kadar)
- GPU benchmark'ımız teorik — PTX compile oluyor ama end-to-end tok/s yok
- V compression ratio'da gerideyiz (4.9x vs 6.4x)
- Onlar production'da, biz lab'da

**Kritik insight:** turboquant_plus'ın QJL'siz PolarQuant SNR'ı 19.3 dB — bizim
SRHT'li 20.1 dB'den sadece 0.8 dB düşük. Asıl avantajımız SRHT QJL değil,
**4-bit K on GGUF** (3-Fix) + **KV Compaction** + **fused attention**. Bu üçlü
kombine olunca rakipsiz bir paket oluşuyor.

---

## Roadmap

### Phase 4: Production Validation (Öncelik: YÜKSEK)

**Hedef:** Algoritmik avantajı gerçek modellerde kanıtla.

| # | İş | Neden | Effort | Impact |
|:-:|:---|:------|:------:|:------:|
| 4.1 | **End-to-end GPU inference benchmark** | PTX compile oluyor ama gerçek tok/s yok. Bu olmadan hiçbir iddia geçerli değil. | M | **KRİTİK** |
| 4.2 | **Qwen2.5-72B Q4_K_M PPL testi** | 7B'de çalışıyor, ama 72B'de compound error farklı davranabilir. tq+ 122B'de validated. | L | KRİTİK |
| 4.3 | **Llama 3.1 70B PPL testi** | Farklı architecture (interleaved RoPE). GQA 8 head. Cross-architecture validation. | L | Yüksek |
| 4.4 | **NIAH (Needle-in-a-Haystack) 8K/32K** | 4K'da pass, ama long-context'te kalite korunuyor mu? tq+ 32K'da 9/9 validated. | M | Yüksek |
| 4.5 | **Automated PPL regression suite** | Her commit'te PPL regresyon olmasın. CI'da wikitext-2 subset. | S | Orta |

**Validation gate:** Qwen 72B + Llama 70B'de +5% PPL içinde kalmalı.

---

### Phase 5: V Compression Gap Kapatma (Öncelik: YÜKSEK)

**Hedef:** V compression ratio'yu 4.9x → 6x+ çıkar.

| # | İş | Neden | Effort |
|:-:|:---|:------|:------:|
| 5.1 | **PolarQuant V 2-bit mode** | tq+'ın turbo2-V'si 6.4x at +6.48% PPL. Bizim PQ V 2-bit'i test et. | S |
| 5.2 | **PolarQuant V + SRHT QJL** | K'da +0.8 dB veriyor, V'de de verir mi? V'de MSE matters, QJL inner product optimize eder — denemeli. | M |
| 5.3 | **Asymmetric K4+V-PQ3 combo** | Onların en iyi formülü: q8_0-K + turbo3-V. Bizimki: TQ4-K + PQ3-V. Karşılaştır. | S |
| 5.4 | **Sparse V + PolarQuant V fused** | Decompress sırasında sparse skip — +22% decode gibi. | M |

---

### Phase 6: Performance (Öncelik: YÜKSEK)

**Hedef:** RTX 3080'de >16 tok/s validated. llama.cpp'yi geç.

| # | İş | Neden | Effort |
|:-:|:---|:------|:------:|
| 6.1 | **GPU model weight upload at load** | Şu an her forward'da lazy upload. Model load'da bir kez yapılmalı. | M |
| 6.2 | **CUDA Graph capture for decode** | Graph manager var ama forward loop'a bağlı değil. Bağla. | L |
| 6.3 | **cuBLAS prefill optimization** | SGEMM wired ama batched multi-head attention optimize edilmedi. | M |
| 6.4 | **Kernel fusion audit** | fused_add_rms_norm, fused_swiglu_gemv — bunlar turbo_generic'te henüz dispatch edilmiyor. | M |
| 6.5 | **Memory pool integration** | GpuMemoryPool var ama forward loop kullanmıyor. Alloc overhead'i kaldır. | M |
| 6.6 | **Benchmark: tok/s vs llama.cpp** | Qwen 7B Q4_K_M — aynı model, aynı prompt, head-to-head. | S |

**Performance gate:** Decode >16 tok/s, prefill >500 tok/s on RTX 3080.

---

### Phase 7: Platform (Öncelik: ORTA)

**Hedef:** Mac + Windows + Linux, GPU agnostic.

| # | İş | Neden | Effort |
|:-:|:---|:------|:------:|
| 7.1 | **Metal kernel'ler** | Apple Silicon pazar payı. tq+'ın Metal kernel'leri referans. | XL |
| 7.2 | **Vulkan/WebGPU backend** | wgpu crate ile cross-platform GPU. Metal+CUDA+Vulkan tek API. | XL |
| 7.3 | **ARM NEON SIMD** | M1/M2/M3 CPU path'te auto-vectorization yetmez, explicit NEON. | M |
| 7.4 | **WASM build** | Browser'da inference — edge deployment. | L |

**Tavsiye:** 7.2 (wgpu) en stratejik. Metal+CUDA+Vulkan tek seferde çözülür.

---

### Phase 8: Ecosystem (Öncelik: ORTA)

**Hedef:** Kullanıcı kazanımı, community building.

| # | İş | Neden | Effort |
|:-:|:---|:------|:------:|
| 8.1 | **llama.cpp entegrasyon guide** | C FFI var, kullanım dokümanı yok. Step-by-step patch guide yaz. | S |
| 8.2 | **Python bindings (PyO3)** | Researcher'lar Python kullanıyor. pip install tq-kv. | M |
| 8.3 | **crates.io publish (tq-cuda)** | src/cuda/ standalone crate olarak yayınla. candle alternatifi. | M |
| 8.4 | **Benchmark reproducibility** | Docker container + scripts. "Run this, get these numbers." | S |
| 8.5 | **Blog post: SRHT QJL explained** | Bilimsel avantajı açıkla. Community attention. | S |
| 8.6 | **HuggingFace integration** | Model card'larda tq-kv destegi. Auto-download + compress. | L |

---

### Phase 9: Research (Öncelik: DÜŞÜK — ama moat)

**Hedef:** Algoritmik üstünlüğü korumak.

| # | İş | Neden | Effort |
|:-:|:---|:------|:------:|
| 9.1 | **Speculative decoding + TQ** | Draft model'in KV cache'i de compress edilebilir. 2x+ throughput. | L |
| 9.2 | **LoRA/QLoRA adapter loading** | Fine-tuned model serving. W*x + alpha*B*A*x fused kernel. | L |
| 9.3 | **Multimodal embeddings** | Vision/audio encoder output → KV cache. Raziel vizyonu. | XL |
| 9.4 | **Multi-GPU tensor parallelism** | NCCL via cudarc. 70B+ modeller için zorunlu. | XL |
| 9.5 | **Adaptive SRHT QJL** | Context length'e göre QJL on/off. Short: off, long: on. Zaten var ama tune et. | S |

---

## Öncelik Sıralaması (6 aylık plan)

```
Ay 1-2: VALIDATION (Phase 4)
  ├─ 4.1 End-to-end GPU benchmark (tok/s)
  ├─ 4.2 Qwen 72B PPL
  ├─ 4.3 Llama 70B PPL
  └─ 4.4 NIAH 32K

Ay 2-3: PERFORMANCE (Phase 6)
  ├─ 6.1 GPU weight upload at load
  ├─ 6.2 CUDA Graph in forward loop
  ├─ 6.4 Kernel fusion dispatch
  └─ 6.6 Benchmark vs llama.cpp

Ay 3-4: V COMPRESSION + ECOSYSTEM (Phase 5 + 8)
  ├─ 5.1 PQ V 2-bit mode
  ├─ 5.3 K4+V-PQ3 combo test
  ├─ 8.1 llama.cpp patch guide
  ├─ 8.2 Python bindings
  └─ 8.3 crates.io tq-cuda

Ay 4-6: PLATFORM + RESEARCH (Phase 7 + 9)
  ├─ 7.2 wgpu cross-platform backend
  ├─ 9.1 Speculative decoding
  ├─ 9.2 LoRA adapter loading
  └─ 8.5 Blog + community

```

---

## Success Metrics

| Milestone | Metric | Target | Current |
|:----------|:-------|:------:|:-------:|
| **GPU inference works** | Qwen 7B decode tok/s | >16 | 0 (untested) |
| **Large model validated** | Qwen 72B PPL | <+5% vs baseline | untested |
| **Beat llama.cpp** | Decode tok/s ratio | >1.0x | untested |
| **V compression gap closed** | V ratio at +1% PPL | >5x | 4.9x |
| **Community traction** | GitHub stars | >100 | 0 |
| **crates.io publish** | tq-cuda downloads | >500/month | 0 |
| **Cross-platform** | Platforms with GPU | 3+ | 1 (CUDA) |

---

## Anti-Patterns (Yapma)

1. **Feature ekleyip test etmeme** — tq+'ın 511 testi var, biz 156. Her feature = test.
2. **Algoritmayı optimize edip ürünü ihmal etme** — +0.3 dB SNR'dan çok end-to-end tok/s önemli.
3. **Metal'e dalıp CUDA'yı bitirmeme** — önce bir platform'da mükemmel ol.
4. **llama.cpp'yi taklit etme** — biz onun yerini alıyoruz, fork'u değil.
5. **SRHT QJL'yi upstream'e verme** — bu bizim moat. Sadece binary (libtq_kv.a).

---

## Sonuç

Algoritmik olarak **tq-kv üstün** (K quality, fused attention, compaction, 3-Fix).
Production olarak **turboquant_plus önde** (model validation, speed benchmarks, community).

Strateji: **Ay 1-2'de validation gap'i kapat.** 72B model PPL + gerçek GPU tok/s —
bu iki sayı kanıtlanınca, algoritmik avantaj ürün avantajına dönüşür.

"Show, don't tell." Benchmark sayıları her şeyi söyler.
