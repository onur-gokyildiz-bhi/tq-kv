// Common CUDA helpers for tq-kv kernels.
// Multi-arch: feature flags adapt code paths per GPU generation.

#pragma once

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

// ─── Warp/block reduction helpers ─────────────────────────────

// Warp-level reduce sum (warp shuffle)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduce max (warp shuffle)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduce sum (shared memory + warp shuffle)
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // one per warp
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    // First warp reduces across warps
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

// Block-level reduce max (shared memory + warp shuffle)
__device__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e10f;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}
