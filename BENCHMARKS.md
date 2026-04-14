# BENCHMARKS

End-to-end inference benchmarks for **tq-engine** (the binary built from this
repo). Quality numbers for the **tq-kv** compression library alone live in
[BENCHMARK.md](BENCHMARK.md).

All numbers are reproducible from a clean checkout. Every tok/s has the exact
CLI invocation next to it. Hype-free: three-run averages, standard deviation,
TTFT separated from steady-state decode, warmup always discarded.

---

## Setup

### Software

| Component | Version |
|:---|:---|
| Rust | 1.91 (stable), release profile |
| CUDA Toolkit | 13.2 |
| NVIDIA Driver | 566.36 (Windows) / 560.x (Linux) |
| OS (primary) | Windows 11 Pro 26200 |
| Host compiler | MSVC cl.exe 19.41 (nvcc host) |

### Hardware tested

| GPU | Arch | VRAM | Status |
|:---|:---|:---|:---|
| RTX 3080 | sm_86 (Ampere) | 10 GB | Primary — all tables below |
| RTX 4090 | sm_89 (Ada) | 24 GB | PTX builds, results pending |
| RTX 2080 Ti | sm_75 (Turing) | 11 GB | PTX builds, no cp.async |
| H100 | sm_90 (Hopper) | 80 GB | PTX builds, results pending |

Multi-arch binary (`TQ_CUDA_ARCHES=75,80,86,89,90`) selects the fattest PTX at
load time; per-arch dispatch inside kernels uses `__CUDA_ARCH__` guards
(cp.async on sm_80+, FP8/TMA hooks on sm_89/sm_90).

### Model files

All GGUFs are the reference `Q4_K_M` quants from unsloth/bartowski. SHA256 is
captured by `tq pull` on first download (see `~/.tq/blobs/`).

| Model | Source | File size |
|:---|:---|:---|
| Qwen2.5-7B-Instruct | `unsloth/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` | 4.68 GB |
| Meta-Llama-3.1-8B-Instruct | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M` | 4.92 GB |
| Mistral-7B-Instruct-v0.3 | `bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M` | 4.37 GB |

```bash
tq pull qwen2:7b
tq pull llama3:8b
tq pull mistral:7b-v0.3
```

### Evaluation text (PPL)

Perplexity uses a deterministic wikitext-2 test slice shipped at
`scripts/eval/wikitext2-raw-v1.test.txt` (2,048 contiguous tokens, UTF-8). The
harness rejects any mismatched SHA256 to prevent silent baseline drift.

---

## Methodology

- **Decode tok/s**: steady-state decode of `-n 100` new tokens on a
  prompt of `-p 32` tokens. First token is discarded (TTFT). Reported
  number is `tokens / (wall_time - TTFT)`.
- **TTFT**: reported separately; includes prompt prefill, graph capture,
  and first-token sampling.
- **Warmup**: one full `-n 100` run is executed and thrown away before the
  measured runs begin. CUDA Graph is captured during warmup.
- **Averaging**: three measured runs, mean ± standard deviation.
- **VRAM**: peak `cudaMemGetInfo` delta between program start and decode
  completion. Driver overhead (~420 MB) subtracted.
- **PPL**: `tq perplexity` over the wikitext-2 slice. Sliding window 512,
  stride 256, teacher-forced.

Reproduce: `scripts/bench-all.sh` runs the full matrix below (~35 min on
RTX 3080).

---

## Qwen2.5 7B Q4_K_M — RTX 3080

Prompt: 32 tokens, generate 100 tokens. CUDA Graph enabled.

| Mode | tok/s (mean ± σ) | TTFT | VRAM | PPL (wikitext-2) |
|:---|---:|---:|---:|---:|
| Standard | **28.0 ± 0.3** | 0.19 s | 5.1 GB | 4.136 |
| TQ 4-bit | **19.7 ± 0.2** | 0.10 s | 4.7 GB | 4.457 (+7.8%) |
| TQ 4-bit + TriAttention (budget 256) | **19.4 ± 0.3** | 0.11 s | 4.6 GB | 4.574 (+10.6%) |

```bash
# Standard
tq bench qwen2:7b -n 100 -p 32

# TQ 4-bit
TQ_BUDGET=0 tq bench qwen2:7b -n 100 -p 32 --turbo-quant

# TQ + TriAttention
TQ_TRIATTN=1 TQ_TRIATTN_BUDGET=256 tq bench qwen2:7b -n 100 -p 32 --turbo-quant

# PPL (any of the above configs)
tq perplexity --model qwen2:7b scripts/eval/wikitext2-raw-v1.test.txt
```

**After dp4a q4km (v0.7.0 target):** TBD (pending Sprint 0.7.0 P0).

---

## Llama 3.1 8B Q4_K_M — RTX 3080

| Mode | tok/s (mean ± σ) | TTFT | VRAM | PPL (wikitext-2) |
|:---|---:|---:|---:|---:|
| Standard | **22.8 ± 0.4** | 0.22 s | 5.6 GB | 26.5 |
| TQ 4-bit | **16.4 ± 0.2** | 0.12 s | 5.2 GB | TBD (pending auto-calibrate re-run) |
| TQ 4-bit + TriAttention (budget 256) | **16.1 ± 0.3** | 0.13 s | 5.1 GB | TBD |

```bash
tq bench llama3:8b -n 100 -p 32
TQ_BUDGET=0 tq bench llama3:8b -n 100 -p 32 --turbo-quant
TQ_TRIATTN=1 TQ_TRIATTN_BUDGET=256 tq bench llama3:8b -n 100 -p 32 --turbo-quant
tq perplexity --model llama3:8b scripts/eval/wikitext2-raw-v1.test.txt
```

> Llama 3 baseline PPL is elevated vs. Qwen2 because the wikitext-2 slice
> is out-of-distribution for the Instruct tune. PPL deltas (TQ vs Std)
> are the load-bearing signal.

**After dp4a q4km (v0.7.0 target):** TBD.

---

## Mistral 7B Instruct v0.3 Q4_K_M — RTX 3080

| Mode | tok/s (mean ± σ) | TTFT | VRAM | PPL (wikitext-2) |
|:---|---:|---:|---:|---:|
| Standard | TBD (pending 0.7.0 re-run) | TBD | TBD | TBD |
| TQ 4-bit | TBD | TBD | TBD | TBD |
| TQ 4-bit + TriAttention (budget 256) | TBD | TBD | TBD | TBD |

```bash
tq bench mistral:7b-v0.3 -n 100 -p 32
```

Last session measured ~19 tok/s Standard on the pre-mrow kernel stack; the
mrow ladder (see CHANGELOG) is expected to carry the same +20% lift over, but
we will not publish a Mistral row until it is re-measured against this harness.

---

## Multi-context scaling (Qwen2.5 7B Q4_K_M)

| Context | Standard tok/s | TQ 4-bit tok/s | TQ+TriAttn tok/s | TQ+TriAttn KV VRAM |
|:---:|---:|---:|---:|---:|
| 1 K | TBD | TBD | TBD | 4.8 MB (budget 128) |
| 4 K | TBD | TBD | TBD | 4.8 MB |
| 16 K | TBD | TBD | TBD | 4.8 MB |
| 32 K | TBD | TBD | TBD | 4.8 MB |

Planned for v0.7.0 release. The constant-KV column is the intended headline:
TriAttention holds a fixed token budget regardless of context, so the memory
number does not grow.

```bash
# scripts/bench-context.sh (lands with 0.7.0)
for N in 1024 4096 16384 32768; do
  tq bench qwen2:7b -n 100 -p $N --turbo-quant
done
```

---

## Comparing to llama.cpp (informational)

On identical hardware and GGUF, llama.cpp's CUDA backend decodes Qwen2.5-7B
Q4_K_M at roughly 90–100 tok/s. tq-engine is currently ~3× slower in
Standard mode; this is a known gap dominated by Q4_K matvec instruction
efficiency (we do not yet emit dp4a). Closing it is the P0 item for v0.7.0
— see `docs/KERNEL-ROADMAP.md` Phase 2.

tq-engine's differentiators are the TurboQuant and TriAttention modes,
which llama.cpp does not ship. The moat is kernel-fused compressed-KV
attention plus the SRHT QJL error correction path.

---

## Raw run logs

Every measured run writes `benchmarks/runs/<date>-<model>-<mode>.jsonl`
with per-token latencies, cudaEvent timings, and env var snapshot.
Attach these to any regression report; do not report aggregate tok/s
without the underlying run log.
