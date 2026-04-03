// GPU tensor shape operations: narrow, transpose, cat, broadcast elementwise.
//
// These enable tensors to stay on GPU between compute kernels,
// eliminating the CPU↔GPU transfer bottleneck.

#include "common.cuh"

// ─── Strided Copy (Narrow, Transpose) ────────────────────────
// Generic gather: output[i] = input[compute_src_offset(i)]
// Used for narrow (contiguous subslice) and transpose (permuted strides).

extern "C" __global__ void strided_copy_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n,                      // total output elements
    const int rank,                   // number of dimensions
    const int* __restrict__ out_shape,    // [rank] output shape
    const int* __restrict__ out_strides,  // [rank] output strides (for linear index decomp)
    const int* __restrict__ src_strides,  // [rank] source strides (for source offset calc)
    const int src_offset               // base offset in source
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Decompose linear index into multi-dim coordinates using output strides
    int remaining = idx;
    int src_idx = src_offset;
    for (int d = 0; d < rank; d++) {
        int coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * src_strides[d];
    }

    output[idx] = input[src_idx];
}

// ─── Concatenate Along Dimension ─────────────────────────────
// Copies N source buffers into one output buffer along a given dimension.
// Each source has the same shape except at the cat dimension.

extern "C" __global__ void concat_copy_f32(
    const float* __restrict__ src,     // source buffer
    float* __restrict__ dst,           // destination buffer
    const int n,                       // elements to copy from this source
    const int dst_offset               // offset in destination
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[dst_offset + idx] = src[idx];
}

// ─── Broadcast Binary Operations ─────────────────────────────
// output[i] = a[broadcast_a(i)] OP b[broadcast_b(i)]
// Supports broadcasting via stride-0 dimensions.

extern "C" __global__ void broadcast_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n,
    const int rank,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    const int* __restrict__ a_strides,
    const int* __restrict__ b_strides
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int remaining = idx;
    int a_idx = 0, b_idx = 0;
    for (int d = 0; d < rank; d++) {
        int coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        a_idx += coord * a_strides[d];
        b_idx += coord * b_strides[d];
    }
    output[idx] = a[a_idx] + b[b_idx];
}

extern "C" __global__ void broadcast_mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n,
    const int rank,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    const int* __restrict__ a_strides,
    const int* __restrict__ b_strides
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int remaining = idx;
    int a_idx = 0, b_idx = 0;
    for (int d = 0; d < rank; d++) {
        int coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        a_idx += coord * a_strides[d];
        b_idx += coord * b_strides[d];
    }
    output[idx] = a[a_idx] * b[b_idx];
}

extern "C" __global__ void broadcast_sub_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n,
    const int rank,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    const int* __restrict__ a_strides,
    const int* __restrict__ b_strides
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int remaining = idx;
    int a_idx = 0, b_idx = 0;
    for (int d = 0; d < rank; d++) {
        int coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        a_idx += coord * a_strides[d];
        b_idx += coord * b_strides[d];
    }
    output[idx] = a[a_idx] - b[b_idx];
}

// ─── Elementwise Unary Operations ────────────────────────────

extern "C" __global__ void exp_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = expf(input[idx]);
}

extern "C" __global__ void sqrt_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = sqrtf(input[idx]);
}

extern "C" __global__ void cos_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = cosf(input[idx]);
}

extern "C" __global__ void sin_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = sinf(input[idx]);
}

extern "C" __global__ void sqr_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = input[idx];
    output[idx] = v * v;
}

extern "C" __global__ void scalar_add_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float scalar,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = input[idx] + scalar;
}

extern "C" __global__ void div_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = a[idx] / b[idx];
}

// ─── Reduction ───────────────────────────────────────────────
// Sum or Max reduction along last dimension.

extern "C" __global__ void reduce_sum_last_f32(
    const float* __restrict__ input,   // [rows, cols]
    float* __restrict__ output,        // [rows, 1]
    const int rows,
    const int cols
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    const float* r = input + row * cols;
    for (int i = 0; i < cols; i++) {
        sum += r[i];
    }
    output[row] = sum;
}

extern "C" __global__ void reduce_max_last_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float mx = -1e30f;
    const float* r = input + row * cols;
    for (int i = 0; i < cols; i++) {
        mx = fmaxf(mx, r[i]);
    }
    output[row] = mx;
}

// ─── Copy with offsets (graph-capture safe cat) ─────────────
// dst[dst_offset + i] = src[src_offset + i] for i in 0..n.
// Replaces strided_copy + concat_copy pair in Tensor::cat() —
// no GPU metadata uploads (clone_htod) needed.

extern "C" __global__ void copy_with_offsets_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int n,
    const int src_offset,
    const int dst_offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[dst_offset + idx] = src[src_offset + idx];
}

// ─── KV Cache Attention Mask ────────────────────────────────
// Generates padding mask for pre-allocated KV cache.
// Positions 0..valid_len → 0.0, positions valid_len..max_seq → -1e10.
// Used with padded Q@K^T to ignore unfilled positions in softmax.
// Graph-safe: no allocations, reads valid_len from a GPU scalar buffer.

extern "C" __global__ void generate_kv_mask_f32(
    float* __restrict__ mask,
    const int* __restrict__ valid_len_ptr,   // GPU scalar: number of valid positions
    const int max_seq
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seq) return;
    const int valid_len = *valid_len_ptr;
    mask[idx] = (idx < valid_len) ? 0.0f : -1e10f;
}

// ─── F32 Matvec ─────────────────────────────────────────────
// Single-vector matrix-vector multiply: output = W @ x
// W: [out_features, in_features] row-major f32 (pre-dequantized, cached on GPU)
// x: [in_features] f32
// output: [out_features] f32
// Grid: out_features blocks, 256 threads
// Used for Q6K and other dtypes without fused dequant kernels.

extern "C" __global__ void f32_matvec(
    const float* __restrict__ W,      // [out_features, in_features]
    const float* __restrict__ x,      // [in_features]
    float* __restrict__ output,       // [out_features]
    const int out_features,
    const int in_features
) {
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;

    const float* w_row = W + row * in_features;
    float sum = 0.0f;
    for (int i = tid; i < in_features; i += blockDim.x) {
        sum += w_row[i] * x[i];
    }

    sum = block_reduce_sum(sum);
    if (tid == 0) {
        output[row] = sum;
    }
}
