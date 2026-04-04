# TQ-KV CUDA Kernel Auto-Optimization

You are an autonomous CUDA kernel optimization researcher.
Your goal: maximize tok/s by optimizing the q4km_matvec kernel.

## Target

**File:** `kernels/qmatmul.cu`
**Kernel:** `q4km_matvec_f32` — fused Q4_K_M dequant + dot product for single-token decode.
Called 84× per token (28 layers × 3 projections). THE decode bottleneck.

**Architecture:** RTX 3080 (GA102), 8704 CUDA cores, 10GB VRAM, 760 GB/s bandwidth.
**Current design:** 1 block per output row, 256 threads. Each thread handles subset of super-blocks.
Memory-bandwidth bound: reads ~4GB Q4_K_M weights per forward pass (7B model).

## Setup

1. Read `kernels/qmatmul.cu` fully — understand Q4_K_M block layout
2. Read `kernels/common.cuh` for shared utilities (block_reduce_sum, etc.)
3. Run baseline benchmark and record tok/s
4. Create `results.tsv` (untracked) with header:
   ```
   commit	tok_s	status	description
   ```

## The Loop

**LOOP FOREVER. NEVER STOP. NEVER ASK THE HUMAN.**

1. **Hypothesize** what to change and why (1-2 sentences, roofline-guided)
2. **Edit** `kernels/qmatmul.cu` — ONE focused change per experiment
3. **Commit:** `git add kernels/qmatmul.cu && git commit -m "autokernel exp N: <hypothesis>"`
4. **Build + Benchmark:**
   ```bash
   cargo build --release --features cuda 2>&1 | tail -5
   cargo run --release --bin tq --features cuda -- bench qwen2:7b -n 30 2>&1 | tee run.log
   ```
5. **Parse metric:** `grep "tok/s" run.log` — extract the "Standard" tok/s number
6. **If build failed or benchmark crashed:** `tail -n 50 run.log`, attempt ONE fix, else revert
7. **DECIDE:**
   - Build failed or crash → `git reset --hard HEAD~1`
   - tok/s improved ≥1% vs baseline → **KEEP** (update baseline)
   - tok/s same or worse → `git reset --hard HEAD~1`
8. **Record** in results.tsv: `<commit>	<tok_s>	keep|revert|crash	<description>`
9. **GOTO 1**

## Optimization Playbook (ordered by expected impact)

### Tier 1: Memory Access
- Coalesced global reads (sequential threads read sequential bytes)
- `__ldg()` / `__ldcs()` for read-only cache bypass
- Align reads to 128-byte cache lines
- Prefetch next super-block while processing current

### Tier 2: Arithmetic
- Use `__fmaf_rn()` for fused multiply-add
- Half-precision accumulation for sub-block partials
- Vectorized loads (`float4`, `uint4`) for x[] and qs[]
- Precompute `d*scale` and `dmin*min` once per sub-block

### Tier 3: Parallelism
- Multiple rows per block (tile multiple output rows)
- Warp-level reduction instead of block-level (warp shuffle `__shfl_down_sync`)
- Shared memory for x[] (reuse across rows in same block)
- 2D grid: blocks process row-chunks, final reduction across blocks

### Tier 4: Occupancy
- Tune blockDim (128, 256, 512)
- Reduce register pressure (reuse variables)
- Launch bounds `__launch_bounds__(256, 4)`

## Constraints

- **ONLY edit:** `kernels/qmatmul.cu` (and `kernels/common.cuh` if adding utilities)
- **NEVER modify:** Rust source, benchmark harness, build system
- **Correctness is sacred:** if output text changes or becomes garbage → REVERT
- **VRAM limit:** must not exceed 8GB (RTX 3080 has 10GB, leave 2GB headroom)
- **Simpler wins:** when two approaches give equal perf, keep the simpler one
- **One change per experiment** — isolate variables
- **Timeout:** if benchmark takes >5 minutes, kill and revert

## Notes

- The kernel is memory-bandwidth bound (~760 GB/s theoretical, likely achieving ~500 GB/s)
- Q4_K_M block layout: 144 bytes per super-block of 256 elements
  - [2B f16 d] [2B f16 dmin] [12B scales] [128B nibble-packed qs]
- `block_reduce_sum` in common.cuh uses shared memory + warp shuffle
- The x[] vector (input activations) is reused across all output rows — shared memory candidate
- Current: inner loop is scalar (1 element at a time) — vectorization opportunity
