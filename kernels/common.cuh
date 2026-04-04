// Common CUDA helpers for tq-kv kernels.

#pragma once

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

// Block-level reduce sum with BROADCAST to all threads.
// Result is valid on ALL threads (not just lane 0).
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // one per warp
    __shared__ float s_result;    // broadcast buffer
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    // First warp reduces across warps
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    // Broadcast result to all threads
    if (threadIdx.x == 0) s_result = val;
    __syncthreads();
    return s_result;
}

// Block-level reduce max with BROADCAST to all threads.
// Result is valid on ALL threads (not just lane 0).
__device__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    __shared__ float s_result;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e10f;
    if (warp_id == 0) val = warp_reduce_max(val);

    // Broadcast result to all threads
    if (threadIdx.x == 0) s_result = val;
    __syncthreads();
    return s_result;
}
