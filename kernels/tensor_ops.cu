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

// ─── Strided Copy (kernel-arg version, no GPU buffer uploads) ──
// Same as strided_copy_f32 but shape/strides passed as kernel args.
// Supports rank 1-6 (covers all LLM tensor shapes).
// Eliminates 3 clone_htod calls per invocation.

extern "C" __global__ void strided_copy_args_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n,
    const int rank,
    // shape (padded to 6)
    const int s0, const int s1, const int s2, const int s3, const int s4, const int s5,
    // out_strides (padded to 6)
    const int os0, const int os1, const int os2, const int os3, const int os4, const int os5,
    // src_strides (padded to 6)
    const int ss0, const int ss1, const int ss2, const int ss3, const int ss4, const int ss5,
    const int src_offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int out_strides[6] = {os0, os1, os2, os3, os4, os5};
    const int src_strides[6] = {ss0, ss1, ss2, ss3, ss4, ss5};

    int remaining = idx;
    int src_idx = src_offset;
    for (int d = 0; d < rank; d++) {
        int coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * src_strides[d];
    }
    output[idx] = input[src_idx];
}

// ─── Broadcast Binary Ops (kernel-arg version) ─────────────────
// Eliminates 4 clone_htod calls per invocation.

#define BROADCAST_BINOP_ARGS(NAME, OP) \
extern "C" __global__ void NAME##_args_f32( \
    const float* __restrict__ a, \
    const float* __restrict__ b, \
    float* __restrict__ output, \
    const int n, const int rank, \
    const int os0, const int os1, const int os2, const int os3, const int os4, const int os5, \
    const int as0, const int as1, const int as2, const int as3, const int as4, const int as5, \
    const int bs0, const int bs1, const int bs2, const int bs3, const int bs4, const int bs5  \
) { \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    const int ost[6] = {os0, os1, os2, os3, os4, os5}; \
    const int ast[6] = {as0, as1, as2, as3, as4, as5}; \
    const int bst[6] = {bs0, bs1, bs2, bs3, bs4, bs5}; \
    int remaining = idx; \
    int a_idx = 0, b_idx = 0; \
    for (int d = 0; d < rank; d++) { \
        int coord = remaining / ost[d]; \
        remaining %= ost[d]; \
        a_idx += coord * ast[d]; \
        b_idx += coord * bst[d]; \
    } \
    output[idx] = a[a_idx] OP b[b_idx]; \
}

BROADCAST_BINOP_ARGS(broadcast_add, +)
BROADCAST_BINOP_ARGS(broadcast_mul, *)
BROADCAST_BINOP_ARGS(broadcast_sub, -)
BROADCAST_BINOP_ARGS(broadcast_div, /)

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

// ─── KV Cache Append (graph-replay-safe) ────────────────────
// Copy new K/V tokens into pre-allocated KV cache at dynamic offset.
// Reads base_dst_offset from a GPU scalar (updated before graph replay).
// Per-head: dst_off = head_idx * head_stride + *base_offset_ptr + intra_offset
//
// This replaces per-head copy_with_offsets calls with a single kernel launch
// that handles all heads, and reads the sequence position from GPU memory.

extern "C" __global__ void kv_cache_append_f32(
    const float* __restrict__ src,       // [n_kv_head * n_new * head_dim]
    float* __restrict__ dst,             // [n_kv_head * max_seq * head_dim]
    const int* __restrict__ seq_pos_ptr, // GPU scalar: current seq position (tokens already in cache)
    const int n_kv_head,
    const int max_seq,
    const int head_dim,
    const int n_new                      // number of new tokens (usually 1 for decode)
) {
    // Grid: (n_kv_head * n_new * head_dim) threads total
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_kv_head * n_new * head_dim;
    if (idx >= total) return;

    const int seq_pos = *seq_pos_ptr;

    // Decompose flat idx → (head, token_in_new, dim)
    const int dim_idx = idx % head_dim;
    const int token_idx = (idx / head_dim) % n_new;
    const int head_idx = idx / (n_new * head_dim);

    // Source: contiguous [n_kv_head, n_new, head_dim]
    // (idx is already the linear index into src)

    // Dest: [n_kv_head, max_seq, head_dim] — head h at offset h*max_seq*head_dim
    const int dst_idx = head_idx * max_seq * head_dim
                      + (seq_pos + token_idx) * head_dim
                      + dim_idx;

    dst[dst_idx] = src[idx];
}

// ─── KV Cache Attention Mask ────────────────────────────────
// Generates padding mask for pre-allocated KV cache.
// Positions 0..valid_len → 0.0, positions valid_len..max_seq → -1e10.
// Used with padded Q@K^T to ignore unfilled positions in softmax.
// Graph-safe: no allocations, reads valid_len from a GPU scalar buffer.

extern "C" __global__ void generate_kv_mask_f32(
    float* __restrict__ mask,
    const int* __restrict__ valid_len_ptr,   // GPU scalar: number of valid positions (pre-append)
    const int max_seq,
    const int extra                          // tokens being appended (usually 1), added to *valid_len_ptr
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seq) return;
    const int valid_len = *valid_len_ptr + extra;
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
