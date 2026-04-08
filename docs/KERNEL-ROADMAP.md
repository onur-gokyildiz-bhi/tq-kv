# CUDA Kernel Geliştirme Planı — tq-kv

**Mevcut:** 16 .cu dosya, ~3.8K CUDA + ~3.9K Rust wrapper = 7.6K satır
**Hedef:** llama.cpp paritesi + TQ benzersiz avantajı

---

## Phase 1: Doğruluk & Stabilite (1-2 hafta)

### 1.1 Flash Decode Kernel Debug
- **Dosya:** `kernels/flash_decode.cu`
- **Sorun:** seq_len > 256'da yanlış attention output → PPL 30K
- **Root cause:** Muhtemelen partial max/sum reduction'da off-by-one veya warp sync eksik
- **Test:** `TQ_SKIP=27 TQ_MAX_SEQ=2048` ile 1290-token PPL = 2.548 olmalı
- **Etki:** Long-context decode 2-3x hızlanır (split-KV parallelism)

### 1.2 Gemma 2 Kernel Desteği
- **Softcap GPU kernel:** `apply_softcap` şu an CPU (vec map). GPU elementwise kernel lazım
- **Dosya:** `kernels/elementwise.cu` → `softcap_f32(x, cap, out, n)`
- **Etki:** Gemma 2 decode 2-3x hızlanır (CPU→GPU bottleneck kalkar)

### 1.3 Non-256-Aligned GEMV Robustness
- **Dosya:** `kernels/qmatmul.cu`
- **Durum:** Bounds check eklendi ama Qwen2 0.5B (896-dim) hala bozuk
- **Sorun:** Muhtemelen shared memory veya warp reduction'da alignment issue
- **Test:** Qwen2 0.5B PPL reasonable olmalı (~15-20)

---

## Phase 2: Performans (2-4 hafta)

### 2.1 Flash Decode v2 (Correct + Fast)
- Split-KV with configurable split_size (128/256/512)
- Warp-level reduction (no shared memory bank conflicts)
- Online softmax (numerically stable, single pass)
- **Hedef:** seq_len=2048'de gqa_decode'a göre 3-4x hızlı

### 2.2 Tensor Core WMMA Matmul
- **Dosya:** Yeni `kernels/wmma_gemv.cu`
- FP16 input × FP16 weight → FP32 accumulate → FP16 output
- RTX 3080: 8x8x4 WMMA (sm_86), 119 TOPS FP16
- **Kullanım:** Prefill matmul (büyük batch), MLP intermediate
- **Etki:** Prefill 2-4x, MLP decode ~1.5x
- **Not:** Q4K dequant → FP16 → WMMA pipeline gerekir

### 2.3 Fused Q4K→FP16→GEMV Pipeline
- Dequant + matmul tek kernel'da (dequant zaten register'larda)
- Mevcut `q4km_matvec` zaten bunu yapıyor ama FP32
- FP16 versiyonu: yarı bandwidth, WMMA uyumlu
- **Etki:** Decode matvec ~1.3x

### 2.4 KV Cache Quantized Attention
- **Dosya:** Yeni `kernels/q4_attention.cu`
- Q4 compressed KV üzerinde doğrudan dot product (dequant-free)
- Mevcut `tq_fused_attention` bunu codebook-based yapıyor
- Genel Q4_K_M KV cache için de benzer kernel
- **Etki:** KV attention bandwidth 4x azalır

---

## Phase 3: Multi-Model & Yeni Arch (2-4 hafta)

### 3.1 Sliding Window Attention Kernel
- **Dosya:** `kernels/flash_attention.cu` veya yeni
- Gemma 2 alternating: even layers window=4096, odd layers global
- GQA decode kernel'ına `window_size` parametresi ekle
- Mask generation: position-aware window boundary

### 3.2 Grouped Query Attention Variants
- MQA (n_kv_head=1): broadcast optimization
- Llama 3.2 1B (n_kv_head=8, head_dim=64): küçük dim optimize
- DeepSeek MLA: multi-latent attention (farklı KV projection)

### 3.3 MoE Dispatch Kernel
- **Dosya:** Yeni `kernels/moe_dispatch.cu`
- Top-k expert seçimi + token routing GPU'da
- Mixtral, DBRX, Qwen-MoE desteği için
- **Etki:** MoE overhead minimize

### 3.4 Speculative Decode Kernels
- Draft model parallel eval
- Verification + acceptance GPU'da
- Token tree attention

---

## Phase 4: İleri Optimizasyon (4-8 hafta)

### 4.1 Persistent Thread Block Attention
- vLLM PagedAttention tarzı: persistent kernel, dynamic scheduling
- Variable-length sequence batch
- **Etki:** Serving throughput 2-3x

### 4.2 FP8 / INT8 Kernels (Ampere+)
- INT8 tensor core GEMM (sm_80+)
- FP8 matmul (sm_89+ / Hopper)
- **Etki:** W8A8 inference 2x decode hızı

### 4.3 Multi-GPU (Tensor Parallelism)
- NCCL ring allreduce
- Attention head split across GPUs
- **Etki:** 70B+ modeller 2+ GPU'da

### 4.4 Custom Memory Allocator
- Pool-based CUDA malloc (cuMemPool)
- Decode-step buffer reuse (mevcut arena system'in production versiyonu)
- **Etki:** Allocation overhead → 0

---

## Kernel Boyut Hedefleri

| Phase | Ek Satır | Toplam | llama.cpp Parite |
|---|---|---|---|
| Mevcut | — | 3.8K | %65 |
| Phase 1 | +500 | 4.3K | %70 |
| Phase 2 | +2K | 6.3K | %85 |
| Phase 3 | +1.5K | 7.8K | %90 |
| Phase 4 | +3K | 10.8K | %95+ |
| Phase 7 | +2.5K | 13.3K | %100+ (llama.cpp'de yok) |

---

## TQ Benzersiz Kernel Avantajı

Bu kernel'lar hiçbir açık kaynak projede yok:

| Kernel | Ne Yapar | Rakiplerde |
|---|---|---|
| `tq_compress.cu` | GPU Hadamard + codebook quantize | ❌ |
| `fused_attention.cu` (TQ) | Compressed KV'den direkt attention score | ❌ |
| `hadamard.cu` | Batched inverse WHT for decompress | ❌ |
| `sparse_v.cu` | Sparse attention × compressed V | ❌ |
| `trig_score.cu` | TriAttention scoring (eviction) | ❌ |

**Bu 5 kernel tq-kv'nin moat'ı.** Rakipler tüm altyapıyı sıfırdan yazmalı.

---

## Phase 5: Developer Experience & Tooling (2-3 hafta)

### 5.1 Benchmark Dashboard (Web UI)
- **İlham:** turboquant-scribe dashboard konsepti
- React/Tailwind web UI: `tq dashboard` komutu ile local server
- Model karşılaştırma: TQ vs Standard, farklı budget'lar, farklı modeller
- Gerçek zamanlı GPU monitoring (VRAM, utilization, temperature)
- 2K PNG export (paylaşılabilir benchmark raporu)
- **Etki:** Projenin görünürlüğü ve demo-ability artışı

### 5.2 HuggingFace Model Uyumluluk Matrisi
- **İlham:** turboquant-scribe HF trending entegrasyonu
- `tq compat <model>` komutu: modelin TQ ile çalışıp çalışmayacağını raporla
- head_dim, arch, VRAM gereksinimi, TQ uyumluluğu, tahmini PPL kaybı
- HF API'den model metadata çekme (parametre sayısı, config.json)
- Auto-calibrate: model çekilince otomatik kalibrasyon öner
- **Etki:** Kullanıcı onboarding süresi azalır

### 5.3 GPU Auto-Detection & Profile
- **İlham:** turboquant-scribe multi-platform GPU detection
- `tq info` komutu: GPU capabilities, VRAM, compute capability, tensor core desteği
- Otomatik optimal config: GPU'ya göre TQ_MAX_SEQ, budget, batch_size öneri
- Platform desteği: Windows (nvidia-smi), Linux (sysfs), macOS (Metal — gelecek)
- **Etki:** Sıfır-config kullanım deneyimi

### 5.4 Safetensors Model Loading
- `from_safetensors()` stub'ı implement et
- FP16/BF16 weight loading (GGUF dönüşüm gereksiz)
- On-the-fly Q4K quantization (load FP16 → quantize → inference)
- **Etki:** Tüm HuggingFace modelleri doğrudan çalışır

### 5.5 Env Var → Config File
- 25+ TQ_* env var → `tq.toml` veya `tq.json` config dosyası
- `tq config set max_seq 2048` CLI komutu
- Per-model config profilleri: `tq config profile qwen2:7b --budget 256`
- **Etki:** Production-ready konfigürasyon yönetimi

---

## Phase 6: Competitive Gap Close — turboquant_plus Parity (3-6 hafta)

> Tom'un bizden iyi olduğu 2 alan: platform coverage ve large-model validation.
> Bu phase her ikisini de kapatır.

### 6.1 Metal Backend (Apple Silicon)
- **Neden:** turboquant_plus en güçlü Metal'de (M1/M2/M5 validated). Biz CUDA-only.
- **Yaklaşım:** `ComputeBackend` trait zaten var → `MetalBackend` impl
- **Metal Shading Language:** Q4K matvec, RoPE, RmsNorm, softmax, attention
- **metal-rs** veya **wgpu** (WebGPU): cross-platform fallback
- **TQ kernels:** Hadamard + codebook WHT rotation (Tom'un fp16 vectorized approach'u referans)
- **Hedef:** M1+ Mac'lerde Qwen2 7B çalıştır, tok/s benchmark
- **Etki:** Apple kullanıcı tabanı açılır (%40+ local LLM kullanıcısı Mac)

### 6.2 Vulkan Backend (AMD/Intel/Qualcomm)
- **Neden:** Tom Vulkan desteği var (AMD RDNA4 validated). Biz sıfır.
- **Yaklaşım:** `VulkanBackend` impl via `vulkano` veya `wgpu` compute shaders
- **Öncelik:** Decode matvec + attention (prefill cuBLAS olmadan yavaş olabilir)
- **Min viable:** Q4K matvec + GQA attention + RoPE + RmsNorm
- **Etki:** AMD GPU kullanıcıları (özellikle Linux gaming community)

### 6.3 Large Model Validation Suite
- **Neden:** Tom 70B, 104B'de 128K context'te validated. Biz sadece 7B test ettik.
- **Plan:**
  - Llama 3.1 70B Q4_K_M: PPL + NIAH @ 4K/16K/32K
  - Qwen2.5 72B Q4_K_M: PPL + NIAH @ 4K/32K
  - Mistral 24B / Mixtral MoE: PPL + decode tok/s
  - Command-R+ 104B (multi-GPU ile): PPL @ 128K
- **Altyapı:** `tq ablate` genişlet → multi-context-length sweep
- **Raporlama:** Otomatik markdown tablo çıktısı (README'ye eklenebilir)
- **Etki:** "Büyük modellerde test etmediler" argümanı kapanır

### 6.4 Automated Benchmark CI
- **Neden:** Tom 511 Python test + reproducible benchmark. Biz 96 test, manual bench.
- **Plan:**
  - GitHub Actions: her PR'da `cargo test` + PPL regression check
  - Benchmark artifact: tok/s + PPL + VRAM → JSON → trend tracking
  - NIAH test suite: automated needle-in-a-haystack across context lengths
  - Comparison badge: "TQ 4-bit: +X% PPL, Y tok/s" otomatik güncellenir
- **Etki:** Her commit'in kalitesi ölçülü, regresyon yakalanır

### 6.5 Community Validation Program
- **Neden:** Tom 50+ tester, çoklu GPU/model raporları. Biz tek geliştirici.
- **Plan:**
  - `tq doctor` genişlet: GPU uyumluluk + model uyumluluk raporu
  - `tq report` komutu: tek tuşla PPL/tok/s/VRAM raporu → GitHub issue template
  - Discord/Matrix channel: kullanıcı raporları toplama
  - Bounty program: yeni GPU/model raporları için credit
- **Etki:** Validation coverage genişler, community ownership başlar

---

## Competitive Moat Summary

```
TQ-KV Benzersiz (Tom'da YOK):
  TriAttention     → CONSTANT memory KV (128K = 4.8 MB)
  Pre-RoPE         → Symmetric 4-bit on Q4KM WORKS (Tom: catastrophic)
  SRHT QJL         → Error correction (+4.5 dB SNR)
  KV Compaction    → 25x token reduction
  Own CUDA stack   → Zero dependencies, full control

Tom'un Avantajı (Phase 6 ile kapatılacak):
  Metal            → Phase 6.1 (MetalBackend)
  Vulkan           → Phase 6.2 (VulkanBackend)
  Large models     → Phase 6.3 (70B/104B validation)
  Community        → Phase 6.5 (validation program)
  Test coverage    → Phase 6.4 (CI + NIAH suite)
```

---

## Phase 7: Speculative Decoding — Block Diffusion (4-8 hafta)

> **İlham:** z-lab/dflash — diffusion drafter ile 6x lossless speedup iddiası.
> Onların kodu Python/HuggingFace, biz kendi CUDA kernel'larımızı yazacağız.
> TQ KV compression + speculative decoding stacklenirse multiplicative gain potansiyeli var.

### 7.1 Draft Model Altyapısı
- **Neden:** Speculative decoding 2 model gerektirir (target + draft). Mevcut runtime tek model.
- **Plan:**
  - Dual-model memory layout: target (Q4K) + draft (Q8/FP16, ~0.5B param) aynı GPU'da
  - VRAM bütçeleme: RTX 3080 10GB'da 7B target Q4K (~4GB) + 0.5B draft FP16 (~1GB) + KV cache (~2GB) = sığar
  - Draft model loader: ayrı GGUF veya safetensors'tan küçük model yükle
  - Shared KV cache: target model'in KV cache'ini draft model okuyabilsin (read-only view)
- **Dosya:** `src/models/draft_model.rs`, `src/speculative.rs`

### 7.2 Autoregressive Draft + Verify Kernels
- **Neden:** Klasik speculative decoding (Leviathan et al. 2023) baseline olarak lazım
- **Plan:**
  - Draft kernel: küçük model N token üretir (N=4-8), tek CUDA stream
  - Verify kernel: target model N+1 token'ı batch forward pass ile doğrular
  - Acceptance sampling: GPU'da rejection sampling (draft vs target logit karşılaştırma)
  - **Dosya:** Yeni `kernels/spec_verify.cu` — batch logit compare + accept/reject
- **Hedef:** Klasik speculative decode ile 2-3x decode hızlanma

### 7.3 Block Diffusion Drafter (DFlash-style)
- **Neden:** DFlash'ın core insight'ı — diffusion model autoregressive'den hızlı draft üretir
- **Plan:**
  - Mini diffusion model: 2-4 layer transformer + noise schedule
  - Block generation: tek forward pass'te 8-16 token paralel üret
  - Target model conditioning: target hidden states'ten feature extraction → draft model KV injection
  - Feature fusion kernel: `kernels/feature_fuse.cu` — multi-layer hidden state projection + concat
  - **Avantaj:** Draft latency token sayısından bağımsız (flat), autoregressive'de linear
- **Dosya:** `kernels/diffusion_step.cu`, `src/models/diffusion_drafter.rs`

### 7.4 TQ × Speculative Decode Fusion
- **Neden:** Bizim moat — TQ compressed KV cache üzerinde speculative decode yapan başka proje yok
- **Plan:**
  - Draft model TQ-compressed KV'den okuyan attention kernel (mevcut `fused_attention.cu` genişlet)
  - Verify step'te TQ decompress bypass: draft accept edilen token'ların KV'si zaten compressed
  - TriAttention entegrasyonu: eviction + speculative decode birlikte çalışsın
    - Evict edilen token'lar speculative draft'ta da skip edilsin
    - Accept edilen yeni token'lar otomatik TQ compress + append
  - **Hedef:** TQ 4-bit KV + speculative decode = 2x mem savings × 3x speed = 6x effective gain
- **Dosya:** `kernels/tq_spec_attention.cu`

### 7.5 Token Tree Attention
- **Neden:** Medusa/EAGLE tarzı tree-based verification daha yüksek acceptance rate verir
- **Plan:**
  - Tree attention mask kernel: `kernels/tree_attention.cu`
  - Draft model birden fazla branch üretir (top-k sampling at each position)
  - Tree verify: target model tek batch'te tüm branch'leri verify eder
  - Tree pruning GPU'da: accepted path selection + KV cache update
- **Etki:** Acceptance rate %60 → %80+, effective speedup 3x → 4-5x

### 7.6 Self-Speculative Decode (Draft Model-Free)
- **Neden:** İkinci model yüklenmeden, target model kendi early layer'ları ile draft yapabilir
- **Plan:**
  - Early-exit head: target model'in N. layer'ından (N=8) küçük LM head ile token üret
  - Skip-layer draft: sadece ilk K layer çalıştır, kalan layer'lar verify'da çalışır
  - **Avantaj:** Ekstra VRAM = ~0, sadece küçük bir LM head (~10MB)
  - **Trade-off:** Acceptance rate target-aware draft'tan düşük olabilir
- **Dosya:** `src/models/self_speculative.rs`, `kernels/early_exit.cu`

---

## Öncelik Özeti

```
Hemen:    Flash decode fix → long-context unlock
Kısa:     Tensor Core WMMA → %50 hız artışı potansiyeli
Orta:     Sliding window + MoE → yeni model ailesi desteği
Uzun:     FP8 + Multi-GPU → production serving
Paralel:  Dashboard + HF compat + safetensors → DX & görünürlük
Rekabet:  Metal + Vulkan + 70B validation → turboquant_plus parity
Gelecek:  Speculative decode → TQ×spec fusion = multiplicative gain
```

---

## Integration Validation Backlog

> Kernel altyapısı hazır ama henüz production'da aktif test edilmemiş özellikler.
> Her biri blocker fix + model indirme + PPL/speed validation gerektiriyor.

### V.1 Speculative Decode Validation
- **Durum:** engine.rs'de tam impl (draft+verify+rollback+streaming)
- **Blocker:** Draft model (Qwen2 0.5B) bozuk — Q5_0 dequant + head_dim=64 base inference bug
- **Fix:** Q5_0 dequant implementasyonu + head_dim=64 inference fix
- **Test:** `tq bench qwen2:7b -n 100 --draft qwen2:0.5b --speculate 5`
- **Beklenen:** ~1.5-2x speedup (K=5, acceptance rate ~60%)

### V.2 MoE Dispatch Validation
- **Durum:** turbo_generic.rs'de tam impl (CPU routing + GPU expert MLPs)
- **Blocker:** Katalogda MoE model yok
- **Fix:** Mixtral 8x7B Q4_K_M kataloga ekle + pull + test
- **Test:** `tq perplexity --model mixtral:8x7b /tmp/ppl_short.txt`
- **Beklenen:** PPL ~5-6, 2-of-8 expert dispatch çalışır

### V.3 Sliding Window Activation
- **Durum:** Kernel window_size param hazır, caller'lar hep 0 geçiyor
- **Blocker:** Gemma 2 fallback path kullanıyor (post_attention_norm yüzünden fused path skip)
- **Fix:** Per-layer window_size config (layer_idx % 2 == 0 → 4096, else → 0) + Gemma 2 fused path unlock
- **Test:** `tq perplexity --model gemma:9b` PPL < 20 olmalı
- **Beklenen:** Gemma 2 PPL fix'in son parçası

### V.4 head_dim > 128 Fused Path
- **Durum:** acc[] overflow fixed, kernel supports head_dim 256
- **Blocker:** Gemma 2 post_attention_norm → fused path skipped
- **Fix:** Fused path'i post_attention_norm destekleyecek şekilde genişlet (norm çağrısı ekle)
- **Test:** Gemma 2 fused path aktif iken PPL + speed
- **Beklenen:** Gemma 2 decode 2-3x hızlanır (CPU fallback → GPU fused)

