# CUDA Kernel Geliştirme Planı — tq-kv

**Mevcut:** 16 .cu dosya, ~3.8K CUDA + ~3.9K Rust wrapper = 7.6K satır
**Hedef:** llama.cpp paritesi + TQ benzersiz avantajı

---

## Phase 1: Doğruluk & Stabilite (1-2 hafta) ✅ TAMAMLANDI

### 1.1 Flash Decode Kernel Debug ✅
- **Commit:** `0b8978d` — KV stride bug (max_seq vs seq_kv). PPL 29672 → 2.539
- **Root cause:** Head 1+ yanlış adres okuyor (stride = seq_kv yerine max_seq olmalıydı)

### 1.2 Gemma 2 Kernel Desteği ✅
- **Commit:** `a0069f2` — GPU `softcap_inplace_f32` kernel
- Gemma 2 PPL hala 418K — kalan sorun fallback path, kernel değil

### 1.3 Non-256-Aligned GEMV Robustness ✅ (önceki session)
- Ceiling division + bounds check eklendi
- Qwen2 0.5B hala bozuk — daha derin inference bug (head_dim=64)

---

## Phase 2: Performans (2-4 hafta) ✅ TAMAMLANDI

### 2.1 Flash Decode v2 (Correct + Fast) ✅
- **Commit:** `a5ae04b` — 128 threads/block (4 warps), block_reduce_sum
- **Sonuç:** +14% decode speed (15.5 → 17.8 tok/s at 500 tokens)

### 2.2 Tensor Core WMMA Matmul — Deferred
- Decode batch=1 bandwidth-bound. WMMA sadece prefill/batch'te etkili.
- Prefill optimizasyonu gerektiğinde implement edilecek.

### 2.3 Fused Q4K→FP16→GEMV Pipeline — Assessed
- 8 rows/block denendi, -16% yavaşladı (register pressure).
- Mevcut 4 rows/block RTX 3080 için optimal.

### 2.4 KV Cache Quantized Attention — Deferred
- tq_fused_attention zaten codebook-based Q4 attention yapıyor.
- Genel Q4_K_M KV için ayrı kernel düşük öncelik.

---

## Phase 3: Multi-Model & Yeni Arch (2-4 hafta) ✅ TAMAMLANDI

### 3.1 Sliding Window Attention Kernel ✅
- **Commit:** `fed6b70` — window_size param GQA decode + flash decode kernels
- window=0 global, >0 sliding. Gemma 2 per-layer wiring bekliyor (V.3)

### 3.2 Grouped Query Attention Variants ✅
- **Commit:** `a312fc1` — acc[4]→acc[9] dynamic, head_dim 256'ya kadar destekli
- MQA/GQA mapping zaten generic. DeepSeek MLA ayrı mimari (deferred)

### 3.3 MoE Dispatch Kernel — Zaten var ✅
- CPU routing + GPU expert MLPs turbo_generic.rs'de implement edilmiş
- Katalogda MoE model yok (Mixtral eklenince test edilecek)
- GPU top-k dispatch batched inference'ta lazım olacak

### 3.4 Speculative Decode Kernels — Zaten var ✅
- engine.rs: draft+verify+rollback+streaming tam impl
- **Blocker:** Draft model (Qwen2 0.5B) head_dim=64 bozuk
- Token tree attention (Phase 7) ayrı planlı

---

## Phase 4: İleri Optimizasyon (4-8 hafta) — Assessed, Deferred

### 4.1 Persistent Thread Block Attention — Deferred
- Serving altyapısı (multi-request, paged KV) lazım. Tek-request engine.

### 4.2 FP8 / INT8 Kernels (Ampere+) — Deferred
- Q8_0 fused matvec zaten var. INT8 tensor core batch/prefill'de etkili.
- Decode batch=1 bandwidth-bound.

### 4.3 Multi-GPU (Tensor Parallelism) — Deferred
- Tek GPU setup. NCCL altyapısı büyük iş.
- **Etki:** 70B+ modeller 2+ GPU'da

### 4.4 Custom Memory Allocator — Deferred
- CUDA internal pool (cuMemAllocAsync) zaten aktif. DecodeScratch pre-allocate ediyor.
- Profilde allocation overhead ölçülemiyor. Mevcut sistem yeterli.

### 4.5 RAM Optimization ✅ (REFACTOR-PLAN'dan)
- **A.1:** `826a781` — Skip warmup_cpu on GPU: 32 GB → 6.5 GB
- **A.3:** `71de975` — Lazy embedding dequant: 6.5 GB → 4.8 GB
- **A.2:** Deferred — raw_data release CPU fallback'i kırar. R.3 refactor sonrası.

### 4.6 Multi-Arch Build + Kernel Optimizations ✅
- **Commit:** `6a399a2` — Multi-arch CUDA build (sm_75/80/86/89/90)
- build.rs: per-arch PTX compilation (80 PTX files), TQ_CUDA_ARCHES override
- Runtime GPU detection: cuDeviceGetAttribute → best_compiled_arch()
- common.cuh: TQ_HAS_CP_ASYNC, TQ_HAS_FP8, TQ_HAS_TMA, TQ_SMEM_MAX_KB
- **__ldg() read-only cache:** flash_decode, qmatmul (x loads, Q8_0), fused_attention (GQA decode)
- **Warp-reduce broadcast:** flash_decode saves 1 __syncthreads per KV token
- **Sonuç:** 17.6 → 18.2 tok/s (+3.4%), PPL unchanged
- **Bank conflict audit:** No actionable conflicts (stride-1 access, broadcast-safe)

### 4.7 cp.async Double-Buffer (sm_80+ Tiered) ✅
- **Commit:** `2d02d3b` — cp.async pipeline for qmatmul x prefetch
- common.cuh: tq_cp_async_f32/commit/wait_all/wait_one via inline PTX
- q4km_matvec + q6k_matvec: double-buffered s_x[2][QK_K], async prefetch next superblock
- sm_75 fallback: synchronous __ldg() (unchanged behavior)
- Verified: sm_75=0, sm_80+=10 cp.async instructions per kernel
- **Sonuç:** 18.2 → 18.4 tok/s (+1.1%), PPL unchanged

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

## Phase 6: Platform & Scale Parity (3-6 hafta)

> İki cephe açık: platform coverage (CUDA-only) ve large-model validation (sadece 7B).
> Bu phase her ikisini de kapatır.

### 6.1 Metal Backend (Apple Silicon)
- **Neden:** Local LLM kullanıcısının %40+'ı Mac. Biz CUDA-only.
- **Yaklaşım:** `ComputeBackend` trait zaten var → `MetalBackend` impl
- **Metal Shading Language:** Q4K matvec, RoPE, RmsNorm, softmax, attention
- **metal-rs** veya **wgpu** (WebGPU): cross-platform fallback
- **TQ kernels:** Hadamard + codebook WHT rotation (fp16 vectorized referans)
- **Hedef:** M1+ Mac'lerde Qwen2 7B çalıştır, tok/s benchmark
- **Etki:** Apple kullanıcı tabanı açılır

### 6.2 Vulkan Backend (AMD/Intel/Qualcomm)
- **Neden:** AMD RDNA4 ve Intel Arc kullanıcıları için sıfır desteğimiz var.
- **Yaklaşım:** `VulkanBackend` impl via `vulkano` veya `wgpu` compute shaders
- **Öncelik:** Decode matvec + attention (prefill cuBLAS olmadan yavaş olabilir)
- **Min viable:** Q4K matvec + GQA attention + RoPE + RmsNorm
- **Etki:** AMD GPU kullanıcıları (özellikle Linux gaming community)

### 6.3 Large Model Validation Suite
- **Neden:** Sadece 7B test ettik. 70B+ ve 128K context iddiaları doğrulanmadı.
- **Plan:**
  - Llama 3.1 70B Q4_K_M: PPL + NIAH @ 4K/16K/32K
  - Qwen2.5 72B Q4_K_M: PPL + NIAH @ 4K/32K
  - Mistral 24B / Mixtral MoE: PPL + decode tok/s
  - Command-R+ 104B (multi-GPU ile): PPL @ 128K
- **Altyapı:** `tq ablate` genişlet → multi-context-length sweep
- **Raporlama:** Otomatik markdown tablo çıktısı (README'ye eklenebilir)
- **Etki:** "Büyük modellerde test etmediler" argümanı kapanır

### 6.4 Automated Benchmark CI
- **Neden:** 96 test, manual bench. Regresyonu yakalayacak otomatik bir şey yok.
- **Plan:**
  - GitHub Actions: her PR'da `cargo test` + PPL regression check
  - Benchmark artifact: tok/s + PPL + VRAM → JSON → trend tracking
  - NIAH test suite: automated needle-in-a-haystack across context lengths
  - Comparison badge: "TQ 4-bit: +X% PPL, Y tok/s" otomatik güncellenir
- **Etki:** Her commit'in kalitesi ölçülü, regresyon yakalanır

### 6.5 Community Validation Program
- **Neden:** Tek geliştirici. Çoklu GPU/model raporu için bağımsız test yok.
- **Plan:**
  - `tq doctor` genişlet: GPU uyumluluk + model uyumluluk raporu
  - `tq report` komutu: tek tuşla PPL/tok/s/VRAM raporu → GitHub issue template
  - Discord/Matrix channel: kullanıcı raporları toplama
  - Bounty program: yeni GPU/model raporları için credit
- **Etki:** Validation coverage genişler, community ownership başlar

---

## Competitive Moat Summary

```
TQ-KV Benzersiz Özellikler:
  TriAttention     → CONSTANT memory KV (128K = 4.8 MB)
  Pre-RoPE         → Symmetric 4-bit on Q4KM WORKS
  SRHT QJL         → Error correction (+4.5 dB SNR)
  KV Compaction    → 25x token reduction
  Own CUDA stack   → Zero dependencies, full control

Açık Cepheler (Phase 6 ile kapatılacak):
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

## Phase 8: Research-Driven Advances (RESEARCH-2026.md'den)

> Araştırma taramasından çıkan, mevcut stack'e eklenebilecek teknikler.
> Her biri bağımsız, sprint olarak planlanabilir.

### 8.0 TriAttention V3 Upgrade — YÜKSEK ÖNCELİK

**Durum:** Mevcut TriAttention = naive V1. V1 NIAH end FAIL. V3 hedefi: +0.006% PPL, NIAH all-pass.

**V3'ün üç eklentisi:**

1. **Prefix protection (128 token):** İlk 128 token ASLA evict edilmez.
   - Bizim sink_tokens=4 yetersiz — system prompt + initial context korunmalı
   - `tq-kv/src/triattention.rs` — protected set'e prefix ekle
   - `TQ_TRIATTN_PREFIX=128` env var

2. **Per-segment eviction quota (K=8):** Context K eşit segment'e bölünür.
   Her segment'ten orantılı evict. Global sort eviction'ı tek bölgeye yoğunlaştırıyor → o
   bölgedeki bilgi kayboluyor → NIAH fail. Segment quota bunu önler.
   - `tq-kv/src/triattention.rs` — segment-aware eviction
   - `kernels/trig_score.cu` — segment boundary parameters
   - `TQ_TRIATTN_SEGMENTS=8` env var

3. **Conservative retention default:** Budget = seq_len × 0.9 (90% retention).
   Mevcut budget=128 çok agresif (%94 eviction → +10.6% PPL).
   Beklenti: %10 evict = +0.006%, %15 evict = +0.48%, %30+ evict = kötü.

**NIAH test protocol:**
- 32K/64K context, 3 pozisyon (start/mid/end)
- Strict checker: exact string match only
- `-b 512` batch size (eviction'ın gerçekten fire etmesi için)
- Min 3 chunk PPL (tek chunk noise'dan ayırt edilemez)

**V3'ün bile çözemediği (bizim fırsat):**
- Qwen3.5 hybrid Mamba+Attention: NIAH mid/end FAIL
  - Sadece rotated K dims'i score'luyor, %75 unrotated dims görünmüyor
  - Çözüm: trig score + direct cosine over unrotated dims
- 85% retention NIAH middle PARTIAL → Quest offload ile geri retrieve (Phase 8.4)
- Multi-needle test yok → bizim Sprint E'de yapılacak

**Etki:** PPL +10.6% → +0.5%, NIAH 32K all-pass, TQ+V3 full stack ~4.2x total compression

---

### 8.1 Entropy Coding Layer (KVTC insight, ICLR 2026)

**Kaynak:** arxiv 2511.01815 — KVTC: 20-40x compression via PCA + entropy coding
**Bizde:** Quantized indices (2/3/4-bit) direkt packed olarak saklanıyor.
**Yenilik:** Indices üzerine ANS (Asymmetric Numeral Systems) veya Huffman coding.
Lloyd-Max codebook sonrası index dağılımı uniform DEĞİL (merkez centroid'ler daha sık).
Entropy coding bu non-uniformity'yi exploit eder.

- **Dosya:** Yeni `tq-kv/src/entropy.rs` — ANS encode/decode
- **CUDA:** `kernels/entropy_decode.cu` — GPU parallel ANS decode (attention sırasında)
- **Etki:** Compression 5x → 8-10x, **sıfır kalite kaybı** (lossless layer)
- **Storage:** ~%30-40 ek sıkıştırma (3-bit indices entropy ~2.1 bit effective)
- **Risk:** Decode latency ekler (ANS decode serial). GPU parallel ANS gerekli.
- **Referans:** NVIDIA nvCOMP kütüphanesi ANS implementasyonu var

### 8.2 Per-Channel Adaptive Codebook (KIVI insight)

**Kaynak:** arxiv 2402.02750 — KIVI: per-channel asymmetric 2-bit, <1% PPL
**Bizde:** Per-vector norm + global codebook. Tüm dimension'lar aynı dağılım varsayılıyor.
**Yenilik:** Her dimension'a özel scale + zero-point. Rotation sonrası bile per-dim variance farklı.

- **Dosya:** `tq-kv/src/codebook.rs` — `PerChannelCodebook` struct
- **CUDA:** `kernels/tq_compress.cu` — per-dim scale/zero shared memory'de
- **Etki:** 2-bit PPL dramatik iyileşme (KIVI <1% vs bizim >>5%)
- **Storage:** 2 × head_dim × f16 per layer (scale + zero) = 512 byte/layer, ihmal edilebilir
- **Mean-removal ile birleşir:** center → per-channel scale → rotate → quantize

### 8.3 Cross-Layer KV Sharing (CommonKV / XQuant insight)

**Kaynak:** arxiv 2508.16134 (CommonKV), 2510.11236 (XQuant sub-1.4 bit)
**Bizde:** Her layer bağımsız compress ediliyor.
**Yenilik:** Adjacent layer'ların KV cache'i %80-98 benzer (SVD analizi).
Delta coding: layer N'i base olarak sakla, layer N+1'i delta olarak.

- **Dosya:** `tq-kv/src/cross_layer.rs` — delta compression between layers
- **Etki:** Per-layer storage %50-80 azalma. 3-bit → effective ~1.5-bit.
- **Risk:** Decode'da layer N'i decompress edip N+1'i reconstruct etmek latency ekler.
- **Strateji:** 4-layer group'lar (base + 3 delta). Base full quality, delta'lar 2-bit.

### 8.4 Quest-Style Offload + Retrieve (TriAttention v2)

**Kaynak:** Quest (Tang et al. 2024) — page-level KV retrieval, near-lossless
**Bizde:** TriAttention evict = kalıcı silme. NIAH'ta bilgi kaybı riski.
**Yenilik:** Evict etme, TQ-compressed olarak CPU'ya offload et. FAISS index tut. Gerektiğinde retrieve.

- **Dosya:** `src/models/turbo_generic/kv_cache.rs` — `OffloadedKvCache` struct
- **CPU side:** FAISS-rs veya custom ANN index (quantized key'ler üzerinde)
- **PCIe transfer:** top-32 retrieve = 32 × 128 × 4 = 16 KB, ~5 µs
- **Etki:** Constant GPU memory + near-lossless long-context recall
- **Risk:** PCIe latency per-token. Batch retrieve ile amortize edilir.
- **TriAttention ile synergy:** Score < threshold → offload (bugün: delete). Score needed → retrieve.

### 8.5 Marlin lop3 Register Dequant

**Kaynak:** IST-DASLab/marlin — W4A16 near-peak tensor core throughput
**Bizde:** Q4K dequant scalar loop (shift + mask + cast + scale). Functional ama slow path.
**Yenilik:** `lop3` PTX instruction: 3-input LUT, tek instruction'da nibble extract + zero-extend.
Register'larda dequant → FP16 → WMMA/tensor core MMA.

- **Dosya:** `kernels/qmatmul.cu` — `q4km_matvec` inner loop değişiklik
- **Etki:** Decode matvec %15-20 hızlanma (dequant compute → ~0, bandwidth utilization ↑)
- **Arch:** sm_75+ (lop3 tüm arch'larda var)
- **Risk:** Q4_K_M format Marlin'in INT4'ünden farklı — nibble layout adaptation gerekli

### 8.6 Fused RoPE + Softcap in Flash Decode

**Kaynak:** FlashDecoding v2+ iterations (Tri Dao)
**Bizde:** RoPE ayrı kernel, softcap ayrı kernel, flash_decode ayrı kernel = 3 launch/layer.
**Yenilik:** Hepsini tek kernel'da fuse et. K load edildiğinde RoPE anında uygula. Score'a softcap uygula.

- **Dosya:** `kernels/flash_decode.cu` — K load loop'una RoPE inline
- **Etki:** 32 layer × 2 extra launch = 64 kernel launch eliminasyonu, %10-15 decode hız
- **Risk:** Kernel complexity artar, register pressure. Profiling ile doğrula.

### 8.7 Learned Linear Adapter (KVLinC alternative to QJL)

**Kaynak:** arxiv 2510.05373 — KVLinC: Hadamard + learned correction
**Bizde:** QJL random projection ile error correction (+4.5 dB SNR).
**Yenilik:** Random projection yerine **learned linear adapter**. Per-layer small matrix (128×128).
Calibration sırasında öğrenilir, inference'da sabit.

- **Dosya:** `tq-kv/src/linear_adapter.rs` — calibration + apply
- **Eğitim:** MSE minimize: `adapter(compressed_key) ≈ original_key`. 5 dakika calibration.
- **Etki:** QJL'den daha iyi error correction, daha az storage (no signs vector)
- **Risk:** Per-layer 128×128 matrix = 64 KB/layer × 32 layer = 2 MB. Kabul edilebilir.
- **QJL ile karşılaştır:** A/B test, hangisi PPL'de daha iyi?

---

## Öncelik Özeti

```
✅ Done:  Phase 1-3 (doğruluk, performans, multi-model kernels)
✅ Done:  Phase 4.5-4.7 (RAM fix, multi-arch, cp.async, common v2)
Sonraki:  Mean-removal (MEAN-REMOVAL-PLAN.md) → +0.33 attention quality
          Entropy coding (8.1) → 5x→8-10x compression, sıfır kalite kaybı
          Per-channel codebook (8.2) → 2-bit viability
          Fused RoPE+softcap (8.6) → %10-15 hız
          Marlin lop3 dequant (8.5) → %15-20 hız
Orta:     Quest offload+retrieve (8.4) → near-lossless long-context
          Self-speculative decode (7.6) → 2x hız, 0 VRAM
          Cross-layer sharing (8.3) → effective sub-2-bit
          KVLinC learned adapter (8.7) → QJL replacement candidate
Uzun:     FP8 + Multi-GPU → production serving
Paralel:  Dashboard + HF compat + safetensors → DX & görünürlük
Rekabet:  Metal + Vulkan + 70B validation
Gelecek:  Block-diagonal rotation research + spec fusion
```

---

## Integration Validation Backlog

> Kernel altyapısı hazır ama henüz production'da aktif test edilmemiş özellikler.
> Her biri blocker fix + model indirme + PPL/speed validation gerektiriyor.

### V.1 Speculative Decode Validation ✅ FUNCTIONAL (perf suboptimal)
- **Fix:** `55f9e2c` Q5_0 split-half dequant fix (0.5B PPL 2.8M→6.9)
- **Fix:** `4f5dcc0` Sequential verify (batched verify mask shape mismatch)
- **Sonuç:** 2.7 tok/s, 1.3 accepted/step (K=5). Çalışıyor ama yavaş.
- **Kalan:** Batched verify (non-square causal mask) + draft KV truncate (full re-prefill yerine)
- **Hedef:** ~1.5-2x speedup (batched verify + KV truncate ile)

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

