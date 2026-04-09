// Common CUDA helpers v2 — improved primitives from llm.c + CUTLASS analysis.
// Multi-arch: feature flags adapt code paths per GPU generation.
//
// Changes vs v1:
//   - Butterfly warp reduce (__shfl_xor_sync) — result broadcast to ALL lanes
//   - cooperative_groups reduce — compiler-optimized, cleaner API
//   - Streaming cache hints (__ldcs/__stcs) — avoid L2 pollution on write-once data
//   - Templated block reduce — single function for sum/max/min
//   - float4 vectorized load/store helpers
//   - Multi-arch cp.async pipeline with sync fallback

#pragma once

#include <cuda_fp16.h>

// ─── Architecture feature flags ───────────────────────────────
// __CUDA_ARCH__ is set by nvcc at compile time (e.g., 860 for sm_86).
// Each kernel is compiled once per target arch, so these resolve statically.

#if __CUDA_ARCH__ >= 800
  #define TQ_HAS_CP_ASYNC 1       // Ampere+ async global→shared memcpy pipeline
#else
  #define TQ_HAS_CP_ASYNC 0
#endif

#if __CUDA_ARCH__ >= 890
  #define TQ_HAS_FP8 1            // Ada/Hopper FP8 E4M3 tensor core
#else
  #define TQ_HAS_FP8 0
#endif

#if __CUDA_ARCH__ >= 900
  #define TQ_HAS_TMA 1            // Hopper tensor memory accelerator
  #define TQ_HAS_WGMMA 1          // Hopper warpgroup MMA
#else
  #define TQ_HAS_TMA 0
  #define TQ_HAS_WGMMA 0
#endif

// Shared memory limits per architecture (conservative defaults)
#if __CUDA_ARCH__ >= 900
  #define TQ_SMEM_MAX_KB 228      // H100/H200
#elif __CUDA_ARCH__ >= 800
  #define TQ_SMEM_MAX_KB 100      // RTX 30xx/40xx, A100 (opt-in up to 164 KB)
#else
  #define TQ_SMEM_MAX_KB 48       // Turing default (opt-in up to 96 KB)
#endif

// ─── Utility ─────────────────────────────────────────────────

#define WARP_SIZE 32

template<class T>
__host__ __device__ __forceinline__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// ─── Vectorized load/store ───────────────────────────────────
// float4 = 16-byte aligned transaction. 4x fewer memory ops than scalar.
// Works on all architectures (sm_35+).

__device__ __forceinline__ float4 ld_vec(const float* addr) {
    return *reinterpret_cast<const float4*>(addr);
}

__device__ __forceinline__ void st_vec(float* addr, float4 val) {
    *reinterpret_cast<float4*>(addr) = val;
}

// Access individual element of float4 by index (0-3)
__device__ __forceinline__ float& vec_at(float4& v, int i) {
    return reinterpret_cast<float*>(&v)[i];
}

__device__ __forceinline__ float vec_at(const float4& v, int i) {
    return reinterpret_cast<const float*>(&v)[i];
}

// ─── Streaming cache hints ───────────────────────────────────
// __ldcs: load with cache-streaming — marks data as low-reuse, avoids L2 pollution.
// __stcs: store with cache-streaming — bypasses L1, writes directly to L2.
// Use for write-once output or read-once input (e.g., partial_O, attention output).

__device__ __forceinline__ float ld_stream(const float* addr) {
    return __ldcs(addr);
}

__device__ __forceinline__ void st_stream(float* addr, float val) {
    __stcs(addr, val);
}

// Read-only texture cache path — L1 bypass, better L2 hit rate for broadcast patterns
__device__ __forceinline__ float ld_readonly(const float* addr) {
    return __ldg(addr);
}

// ─── Warp-level butterfly reduce ─────────────────────────────
// __shfl_xor_sync butterfly pattern: result available on ALL lanes.
// Unlike __shfl_down_sync (result only on lane 0), eliminates broadcast step.
// Source: llm.c (Karpathy) common.h pattern.

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ─── Block-level reduce with ALL-thread broadcast ────────────
// Key difference from v1: result available in ALL threads, not just tid==0.
// Eliminates broadcast hacks at call sites (saves 1 __syncthreads per call).
//
// v1: float r = block_reduce_sum(x);         // only tid==0 correct
//     __shared__ float s; if(!tid) s=r; sync; // broadcast hack
//     r = s;
//
// v2: float r = block_reduce_sum(x);         // ALL threads correct
//
// Internal: 2 __syncthreads (same count as v1 + hack, but encapsulated).

__device__ float block_reduce_sum(float val) {
    __shared__ float smem[WARP_SIZE];
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;

    // Butterfly warp reduce: result in ALL lanes
    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    // ALL threads sum across warps → broadcast
    float total = 0.0f;
    for (int w = 0; w < n_warps; ++w) total += smem[w];
    __syncthreads();  // Protect smem for next reduce call
    return total;
}

__device__ float block_reduce_max(float val) {
    __shared__ float smem[WARP_SIZE];
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    float best = -1e10f;
    for (int w = 0; w < n_warps; ++w) best = fmaxf(best, smem[w]);
    __syncthreads();
    return best;
}

// ─── Async copy helpers (sm_80+ / Ampere+) ────────────────────
// cp.async bypasses registers: global → shared directly.
// Frees register file for compute, enables load/compute overlap.
// On sm_75 and below, use synchronous fallback via __ldg.

#if TQ_HAS_CP_ASYNC

__device__ __forceinline__ void tq_cp_async_f32(float* dst_shared, const float* src_global) {
    uint32_t smem_addr = __cvta_generic_to_shared(dst_shared);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(src_global)
    );
}

// 16-byte (float4) async copy — 4x bandwidth per instruction
__device__ __forceinline__ void tq_cp_async_f32x4(float* dst_shared, const float* src_global) {
    uint32_t smem_addr = __cvta_generic_to_shared(dst_shared);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(src_global)
    );
}

__device__ __forceinline__ void tq_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void tq_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Wait until at most N groups outstanding (for double/triple buffer overlap)
__device__ __forceinline__ void tq_cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

#else // sm_75 and below — synchronous fallback

__device__ __forceinline__ void tq_cp_async_f32(float* dst_shared, const float* src_global) {
    *dst_shared = __ldg(src_global);
}

__device__ __forceinline__ void tq_cp_async_f32x4(float* dst_shared, const float* src_global) {
    *reinterpret_cast<float4*>(dst_shared) = *reinterpret_cast<const float4*>(src_global);
}

__device__ __forceinline__ void tq_cp_async_commit() {}
__device__ __forceinline__ void tq_cp_async_wait_all() { __syncthreads(); }
__device__ __forceinline__ void tq_cp_async_wait_one() { __syncthreads(); }

#endif // TQ_HAS_CP_ASYNC

// ─── Unified async-or-sync shared memory load ────────────────
// Use this in kernel loops for double-buffered pipelines.
// sm_80+: true async (compute overlaps load)
// sm_75:  sync fallback (__ldg + __syncthreads)

__device__ __forceinline__ void tq_load_tile_f32(
    float* smem, const float* gmem, int n_elements
) {
    for (int i = threadIdx.x; i < n_elements; i += blockDim.x) {
        tq_cp_async_f32(&smem[i], &gmem[i]);
    }
    tq_cp_async_commit();
}

__device__ __forceinline__ void tq_load_tile_f32x4(
    float* smem, const float* gmem, int n_float4s
) {
    for (int i = threadIdx.x; i < n_float4s; i += blockDim.x) {
        tq_cp_async_f32x4(&smem[i * 4], &gmem[i * 4]);
    }
    tq_cp_async_commit();
}

// ─── Online softmax helpers ──────────────────────────────────
// Numerically stable single-pass softmax (Milakov & Gimelshein 2018).
// Used in flash_decode and fused_attention.

// Usage:
//   OnlineSoftmax sm;
//   for each key:
//     float score = block_reduce_sum(dot) * scale;  // ALL threads have score
//     float rescale = sm.update(score);              // rescale factor for accumulators
//     for (d) acc[d] = acc[d] * rescale + sm.weight(score) * v[d];
//   float inv = sm.finalize();
//   for (d) output[d] = acc[d] * inv;

struct OnlineSoftmax {
    float max_val;
    float sum;

    __device__ __forceinline__ OnlineSoftmax() : max_val(-1e10f), sum(0.0f) {}

    // Update with new score. Returns rescale factor for existing accumulators.
    __device__ __forceinline__ float update(float score) {
        float old_max = max_val;
        max_val = fmaxf(max_val, score);
        float rescale = expf(old_max - max_val);
        sum = sum * rescale + expf(score - max_val);
        return rescale;
    }

    // Attention weight for current score (call after update).
    __device__ __forceinline__ float weight(float score) const {
        return expf(score - max_val);
    }

    // Finalize: return 1/sum for output normalization.
    __device__ __forceinline__ float finalize() const {
        return (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    }
};
