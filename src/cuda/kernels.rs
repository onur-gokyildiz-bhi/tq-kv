//! CUDA kernel launcher — loads PTX at init, provides type-safe Rust wrappers.
//!
//! PTX files are compiled by build.rs (nvcc) and embedded via include_str!.
//! Each kernel is loaded once into a CudaModule, then launched via cudarc's
//! launch_builder API.
//!
//! Only compiled when `--features cuda`.

use std::sync::Arc;
use cudarc::driver::{
    CudaContext, CudaStream, CudaModule, CudaFunction, CudaSlice, LaunchConfig,
    result::DriverError,
};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::Ptx;

/// Global kernel registry — initialized once, accessible from any GPU tensor op.
static GLOBAL_REGISTRY: std::sync::OnceLock<Arc<KernelRegistry>> = std::sync::OnceLock::new();

/// Get or initialize the global kernel registry.
pub fn global_registry() -> Option<&'static Arc<KernelRegistry>> {
    GLOBAL_REGISTRY.get()
}

/// Set the global kernel registry (called during device init).
pub fn set_global_registry(reg: Arc<KernelRegistry>) {
    let _ = GLOBAL_REGISTRY.set(reg);
}

// ─── PTX sources (embedded by build.rs) ────────────────────────

// Include the auto-generated PTX module from build.rs
include!(concat!(env!("OUT_DIR"), "/ptx_generated.rs"));

// ─── Kernel registry ───────────────────────────────────────────

/// Global kernel registry — lazily loads PTX modules and caches compiled kernels.
pub struct KernelRegistry {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    modules: std::collections::HashMap<&'static str, Arc<CudaModule>>,
    /// Cached cuBLAS handle (expensive to create, reuse across all matmul calls).
    pub cublas: std::sync::OnceLock<cudarc::cublas::CudaBlas>,
}

impl KernelRegistry {
    /// Initialize the kernel registry — loads all 11 compiled PTX modules.
    pub fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self, DriverError> {
        let mut reg = Self {
            ctx: ctx.clone(),
            stream: stream.clone(),
            modules: std::collections::HashMap::new(),
            cublas: std::sync::OnceLock::new(),
        };
        reg.load_all_ptx()?;
        Ok(reg)
    }

    /// Get or create the cached cuBLAS handle.
    pub fn get_cublas(&self) -> &cudarc::cublas::CudaBlas {
        self.cublas.get_or_init(|| {
            cudarc::cublas::CudaBlas::new(self.stream.clone())
                .expect("cuBLAS handle creation failed")
        })
    }

    /// Load all compiled PTX modules from embedded strings.
    fn load_all_ptx(&mut self) -> Result<(), DriverError> {
        let ptx_sources: &[(&str, &str)] = &[
            ("elementwise", PTX_ELEMENTWISE),
            // flash_attention: deferred (PTX JIT issue with mma.h on sm_86)
            // ("flash_attention", PTX_FLASH_ATTENTION),
            ("flash_decode", PTX_FLASH_DECODE),
            ("fused_attention", PTX_FUSED_ATTENTION),
            ("fused_layer", PTX_FUSED_LAYER),
            ("tensor_ops", PTX_TENSOR_OPS),
            ("fused_mlp", PTX_FUSED_MLP),
            ("fused_norm", PTX_FUSED_NORM),
            ("hadamard", PTX_HADAMARD),
            ("qmatmul", PTX_QMATMUL),
            ("rope", PTX_ROPE),
            ("sampling", PTX_SAMPLING),
            ("softmax", PTX_SOFTMAX),
            ("sparse_v", PTX_SPARSE_V),
            ("tq_compress", PTX_TQ_COMPRESS),
        ];
        let mut loaded = 0;
        for &(name, ptx_src) in ptx_sources {
            match self.load_ptx_module(name, ptx_src) {
                Ok(()) => loaded += 1,
                Err(e) => eprintln!("[cuda] WARNING: failed to load {} PTX: {:?}", name, e),
            }
        }
        eprintln!("[cuda] Loaded {}/{} PTX modules", loaded, ptx_sources.len());
        if loaded == 0 {
            return Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_PTX));
        }
        Ok(())
    }

    /// Load a PTX string into the registry under the given name.
    pub fn load_ptx_module(&mut self, name: &'static str, ptx_src: &str) -> Result<(), DriverError> {
        let ptx = Ptx::from_src(ptx_src);
        let module = self.ctx.load_module(ptx)?;
        self.modules.insert(name, module);
        Ok(())
    }

    /// Get a kernel function from a loaded module.
    pub fn get_fn(&self, module: &str, kernel: &str) -> Result<CudaFunction, DriverError> {
        let m = self.modules.get(module)
            .ok_or_else(|| {
                eprintln!("[cuda] Module '{}' not loaded (needed for kernel '{}')", module, kernel);
                DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_FOUND)
            })?;
        m.load_function(kernel)
    }
}

/// Pre-bind CUDA context to current thread.
/// Call once at the start of each forward pass to make subsequent
/// bind_to_thread() calls a cheap no-op (context is already current).
pub fn bind_context(reg: &KernelRegistry) {
    let _ = reg.stream.context().bind_to_thread();
}

// ─── Launch configuration helpers ──────────────────────────────

/// Standard 1D launch: n elements, 256 threads per block.
pub fn launch_1d(n: usize) -> LaunchConfig {
    let threads = 256u32;
    let blocks = ((n as u32) + threads - 1) / threads;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// 1D launch: 1 block per row, block_size threads.
/// block_size must be a multiple of 32 (warp size) for correct warp shuffle.
pub fn launch_per_row(n_rows: usize, block_size: usize) -> LaunchConfig {
    debug_assert!(block_size % 32 == 0, "block_size must be multiple of 32, got {}", block_size);
    LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch with shared memory.
pub fn launch_with_shmem(grid: u32, block: u32, shmem: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: shmem,
    }
}

// ─── Type-safe kernel wrappers ─────────────────────────────────
//
// cudarc 0.19 API: stream.launch_builder(&func).arg(&a).arg(&b).launch(cfg)
//
// Priority order for wiring:
//  1. softmax      — attention + logits
//  2. rms_norm     — normalization (2× per layer)
//  3. silu_mul     — MLP activation
//  4. rope         — position encoding
//  5. q4km_matvec  — fused decode matmul
//  6. elementwise  — add, mul, scalar
//  7. flash_attn   — prefill
//  8. tq_fused_attn — compressed KV
//  9. hadamard     — key decompress
// 10. sparse_v     — compressed V

/// Launch softmax_last_dim_f32: 1 block per row.
pub fn softmax_last_dim(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("softmax", "softmax_last_dim_f32")?;
    // Block size must be multiple of 32 (warp size) — warp shuffle reads all 32 lanes.
    // Round up to next multiple of 32, cap at 256.
    let block = (((n_cols.max(32) + 31) / 32) * 32).min(256) as u32;
    let cfg = launch_with_shmem(n_rows as u32, block, block * 4);
    let nr = n_rows as i32;
    let nc = n_cols as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input)
            .arg(output)
            .arg(&nr)
            .arg(&nc)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch rms_norm_f32: 1 block per token.
pub fn rms_norm(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n_tokens: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_norm", "rms_norm_f32")?;
    let block = 256.min(hidden_dim) as u32;
    let cfg = launch_with_shmem(n_tokens as u32, block, block * 4);
    let hd = hidden_dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input)
            .arg(weight)
            .arg(output)
            .arg(&hd)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch fused_add_rms_norm_f32: residual += input; output = norm(residual).
pub fn fused_add_rms_norm(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,
    residual: &mut CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n_tokens: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_norm", "fused_add_rms_norm_f32")?;
    let block = 256.min(hidden_dim) as u32;
    let cfg = launch_with_shmem(n_tokens as u32, block, block * 4);
    let hd = hidden_dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input)
            .arg(residual)
            .arg(weight)
            .arg(output)
            .arg(&hd)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch fused_silu_mul_f32: output = silu(gate) * up.
pub fn fused_silu_mul(
    reg: &KernelRegistry,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n_elements: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_mlp", "fused_silu_mul_f32")?;
    let cfg = launch_1d(n_elements);
    let n = n_elements as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(gate)
            .arg(up)
            .arg(output)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch q4km_matvec_f32: fused Q4_K_M dequant + matvec.
pub fn q4km_matvec(
    reg: &KernelRegistry,
    w_packed: &CudaSlice<u8>,
    x: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_features: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("qmatmul", "q4km_matvec_f32")?;
    // Multi-row: 4 rows per block → ceil(out_features / 4) blocks
    let rows_per_block = 4u32;
    let n_blocks = (out_features as u32 + rows_per_block - 1) / rows_per_block;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let of = out_features as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_packed)
            .arg(x)
            .arg(output)
            .arg(&of)
            .arg(&inf)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch q8_0_matvec_f32: fused Q8_0 dequant + matvec.
/// Q6K fused dequant + matvec. 210 bytes/256 values = 4.9x less bandwidth than F32.
pub fn q6k_matvec(
    reg: &KernelRegistry,
    w_packed: &CudaSlice<u8>,
    x: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_features: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("qmatmul", "q6k_matvec_f32")?;
    let rows_per_block = 4u32;
    let n_blocks = (out_features as u32 + rows_per_block - 1) / rows_per_block;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let of = out_features as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_packed).arg(x).arg(output)
            .arg(&of).arg(&inf)
            .launch(cfg)?;
    }
    Ok(())
}

pub fn q8_0_matvec(
    reg: &KernelRegistry,
    w_packed: &CudaSlice<u8>,
    x: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_features: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("qmatmul", "q8_0_matvec_f32")?;
    let cfg = launch_per_row(out_features, 256);
    let of = out_features as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_packed)
            .arg(x)
            .arg(output)
            .arg(&of)
            .arg(&inf)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch elementwise add_f32.
pub fn add(
    reg: &KernelRegistry,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "add_f32")?;
    let cfg = launch_1d(n);
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(output)
            .arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// GPU argmax: find index of maximum value. Single block, 256 threads.
/// Returns u32 index via out_idx GPU buffer. Avoids 600KB D2H copy.
pub fn argmax_gpu(
    reg: &KernelRegistry,
    data: &CudaSlice<f32>,
    out_idx: &mut CudaSlice<u32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "argmax_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(data).arg(out_idx).arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// In-place bias add: data[i] += bias[i].
pub fn bias_add_inplace(
    reg: &KernelRegistry,
    data: &mut CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "bias_add_f32")?;
    let cfg = launch_1d(n);
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(data as &mut CudaSlice<f32>)
            .arg(bias)
            .arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch elementwise mul_f32.
pub fn mul(
    reg: &KernelRegistry,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "mul_f32")?;
    let cfg = launch_1d(n);
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(output)
            .arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch scalar_mul_f32.
pub fn scalar_mul(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    scalar: f32,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "scalar_mul_f32")?;
    let cfg = launch_1d(n);
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input)
            .arg(output)
            .arg(&scalar)
            .arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch silu_f32: output = x * sigmoid(x).
pub fn silu(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "silu_f32")?;
    let cfg = launch_1d(n);
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input)
            .arg(output)
            .arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// Convert f32 → f16 on GPU.
pub fn f32_to_f16(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<half::f16>,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "f32_to_f16_kernel")?;
    let cfg = launch_1d(n);
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input)
            .arg(output)
            .arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// Convert f16 → f32 on GPU.
pub fn f16_to_f32(
    reg: &KernelRegistry,
    input: &CudaSlice<half::f16>,
    output: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "f16_to_f32_kernel")?;
    let cfg = launch_1d(n);
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input)
            .arg(output)
            .arg(&ni)
            .launch(cfg)?;
    }
    Ok(())
}

/// GPU TurboQuant key compression: Hadamard + codebook quantize + pack.
/// One block per KV head. Replaces CPU-side compress_single_key_with_signs.
pub fn tq_compress_key(
    reg: &KernelRegistry,
    key_vectors: &CudaSlice<f32>,  // [n_kv_heads, head_dim]
    signs: &CudaSlice<f32>,        // [head_dim]
    boundaries: &CudaSlice<f32>,   // [n_centroids - 1]
    centroids: &CudaSlice<f32>,    // [n_centroids]
    packed_out: &mut CudaSlice<u8>,  // [n_kv_heads, bytes_per_key]
    norms_out: &mut CudaSlice<f32>,  // [n_kv_heads]
    n_kv_heads: usize,
    head_dim: usize,
    n_centroids: usize,
    bytes_per_key: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tq_compress", "tq_compress_key_f32")?;
    let n_boundaries = n_centroids - 1;
    let shmem = ((head_dim + n_boundaries + n_centroids) * 4) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: shmem,
    };
    let hd = head_dim as i32;
    let nc = n_centroids as i32;
    let bpk = bytes_per_key as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(key_vectors).arg(signs).arg(boundaries).arg(centroids)
            .arg(packed_out).arg(norms_out)
            .arg(&hd).arg(&nc).arg(&bpk)
            .launch(cfg)?;
    }
    Ok(())
}

/// D2D scatter: append compressed token to GpuCompressedKv at correct position.
/// Zero CPU round-trip — all data stays on GPU.
pub fn tq_cache_scatter(
    reg: &KernelRegistry,
    packed_src: &CudaSlice<u8>,
    norms_src: &CudaSlice<f32>,
    v_src: &CudaSlice<f32>,
    packed_dst: &mut CudaSlice<u8>,
    norms_dst: &mut CudaSlice<f32>,
    v_dst: &mut CudaSlice<f32>,
    pos: usize,
    n_kv_heads: usize,
    max_seq: usize,
    head_dim: usize,
    bytes_per_key: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tq_compress", "tq_cache_scatter")?;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let p = pos as i32;
    let ms = max_seq as i32;
    let hd = head_dim as i32;
    let bpk = bytes_per_key as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(packed_src).arg(norms_src).arg(v_src)
            .arg(packed_dst).arg(norms_dst).arg(v_dst)
            .arg(&p).arg(&ms).arg(&hd).arg(&bpk)
            .launch(cfg)?;
    }
    Ok(())
}

/// Flash decode partial: split KV across blocks, each computes partial attention.
/// Grid: (n_splits, n_heads, batch), Block: 32 threads.
/// Call flash_decode_reduce after to combine partial results.
pub fn flash_decode_partial(
    reg: &KernelRegistry,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    partial_o: &mut CudaSlice<f32>,
    partial_max: &mut CudaSlice<f32>,
    partial_sum: &mut CudaSlice<f32>,
    batch_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    split_size: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("flash_decode", "flash_decode_partial")?;
    let n_splits = (seq_kv + split_size - 1) / split_size;
    let cfg = LaunchConfig {
        grid_dim: (n_splits as u32, n_heads as u32, batch_size as u32),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let bs = batch_size as i32;
    let nh = n_heads as i32;
    let nkv = n_kv_heads as i32;
    let skv = seq_kv as i32;
    let hd = head_dim as i32;
    let ss = split_size as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(q).arg(k).arg(v)
            .arg(partial_o).arg(partial_max).arg(partial_sum)
            .arg(&bs).arg(&nh).arg(&nkv).arg(&skv).arg(&hd).arg(&scale).arg(&ss)
            .launch(cfg)?;
    }
    Ok(())
}

/// Flash decode reduce: combine partial results from split-KV decode.
/// Grid: (1, n_heads, batch), Block: 32 threads.
pub fn flash_decode_reduce(
    reg: &KernelRegistry,
    partial_o: &CudaSlice<f32>,
    partial_max: &CudaSlice<f32>,
    partial_sum: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n_heads: usize,
    n_splits: usize,
    head_dim: usize,
    batch_size: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("flash_decode", "flash_decode_reduce")?;
    let cfg = LaunchConfig {
        grid_dim: (1, n_heads as u32, batch_size as u32),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let nh = n_heads as i32;
    let ns = n_splits as i32;
    let hd = head_dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(partial_o).arg(partial_max).arg(partial_sum).arg(output)
            .arg(&nh).arg(&ns).arg(&hd)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch flash_attention_prefill_f32.
pub fn flash_attention_prefill(
    reg: &KernelRegistry,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    o: &mut CudaSlice<f32>,
    batch_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Result<(), DriverError> {
    let f = reg.get_fn("flash_attention", "flash_attention_prefill_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32 * n_heads as u32, seq_q as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: (2 * 64 * head_dim * 4) as u32,
    };
    let bs = batch_size as i32;
    let nh = n_heads as i32;
    let nkv = n_kv_heads as i32;
    let sq = seq_q as i32;
    let skv = seq_kv as i32;
    let hd = head_dim as i32;
    let c = causal as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(q).arg(k).arg(v).arg(o)
            .arg(&bs).arg(&nh).arg(&nkv)
            .arg(&sq).arg(&skv).arg(&hd)
            .arg(&scale).arg(&c)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch tq_fused_attention_f32 for compressed KV attention.
pub fn tq_fused_attention(
    reg: &KernelRegistry,
    query: &CudaSlice<f32>,
    packed_indices: &CudaSlice<u8>,
    norms: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    scores_out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    bits: usize,
    scale: f32,
    max_seq: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_attention", "tq_fused_attention_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: ((1 << bits) * 4) as u32,
    };
    let nh = n_heads as i32;
    let nkv = n_kv_heads as i32;
    let nk = n_keys as i32;
    let hd = head_dim as i32;
    let b = bits as i32;
    let ms = max_seq as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(query).arg(packed_indices).arg(norms).arg(centroids).arg(scores_out)
            .arg(&nh).arg(&nkv).arg(&nk).arg(&hd).arg(&b).arg(&scale).arg(&ms)
            .launch(cfg)?;
    }
    Ok(())
}

/// GPU key decompression: packed indices → full FP32 keys.
/// Decompresses TQ-compressed keys directly on GPU for standard Q@K matmul.
/// Grid: (n_kv_heads, n_keys), Block: (head_dim or 128)
pub fn tq_decompress_keys(
    reg: &KernelRegistry,
    packed_indices: &CudaSlice<u8>,
    norms: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    signs: &CudaSlice<f32>,
    keys_out: &mut CudaSlice<f32>,
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    bits: usize,
    max_seq: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_attention", "tq_decompress_keys_f32")?;
    let block = head_dim.min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, n_keys as u32, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nkv = n_kv_heads as i32;
    let nk = n_keys as i32;
    let hd = head_dim as i32;
    let b = bits as i32;
    let ms = max_seq as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(packed_indices).arg(norms).arg(centroids).arg(signs)
            .arg(keys_out)
            .arg(&nkv).arg(&nk).arg(&hd).arg(&b).arg(&ms)
            .launch(cfg)?;
    }
    Ok(())
}

/// Two-step GPU key decompression: centroid lookup → hadamard inverse.
/// Replaces the fused tq_decompress_keys which had a subtle WHT bug.
///
/// Step 1: tq_centroid_lookup_f32 — unpack indices, lookup centroids, scale by sigma
/// Step 2: hadamard_inverse_batch_f32 — proven-correct inverse WHT + sign flip
///
/// Output layout: [n_kv_heads, n_keys, head_dim] (same as tq_decompress_keys)
pub fn tq_decompress_keys_v2(
    reg: &KernelRegistry,
    packed_indices: &CudaSlice<u8>,
    norms: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    signs: &CudaSlice<f32>,
    keys_out: &mut CudaSlice<f32>,
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    bits: usize,
    max_seq: usize,
) -> Result<(), DriverError> {
    // Contiguous output: key_offset=0, out_stride=n_keys
    tq_decompress_keys_range(
        reg, packed_indices, norms, centroids, signs, keys_out,
        n_kv_heads, n_keys, head_dim, bits, max_seq, 0, n_keys,
    )
}

/// Decompress a range of keys with configurable source offset and output stride.
///
/// key_offset: start reading from this position in packed_indices/norms
/// out_stride: per-head stride in output buffer (in vectors, not bytes)
///   - For contiguous output: out_stride = n_keys
///   - For strided cache:     out_stride = max_seq (or cache stride)
///
/// Writes n_kv_heads × n_keys vectors of head_dim floats.
/// Output position for head h, key i: keys_out[(h * out_stride + i) * head_dim]
pub fn tq_decompress_keys_range(
    reg: &KernelRegistry,
    packed_indices: &CudaSlice<u8>,
    norms: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    signs: &CudaSlice<f32>,
    keys_out: &mut CudaSlice<f32>,
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    bits: usize,
    max_seq: usize,
    key_offset: usize,
    out_stride: usize,
) -> Result<(), DriverError> {
    if n_keys == 0 { return Ok(()); }

    // Step 1: Centroid lookup → temp buffer (contiguous)
    // We need contiguous vectors for hadamard_inverse_batch, so always write
    // to a contiguous region first, then scatter to strided output if needed.
    let contiguous = out_stride == n_keys && key_offset == 0;

    if contiguous {
        // Fast path: output is already contiguous
        let f = reg.get_fn("fused_attention", "tq_centroid_lookup_f32")?;
        let block = head_dim.min(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_heads as u32, n_keys as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nkv, nk, hd, b, ms) = (n_kv_heads as i32, n_keys as i32, head_dim as i32, bits as i32, max_seq as i32);
        let ko = 0i32;
        let os = n_keys as i32;
        unsafe {
            reg.stream.launch_builder(&f)
                .arg(packed_indices).arg(norms).arg(centroids)
                .arg(&mut *keys_out)
                .arg(&nkv).arg(&nk).arg(&hd).arg(&b).arg(&ms).arg(&ko).arg(&os)
                .launch(cfg)?;
        }
        hadamard_inverse_batch(reg, keys_out, signs, n_kv_heads * n_keys, head_dim)?;
    } else {
        // Strided path: centroid lookup to temp, hadamard, then scatter
        let total = n_kv_heads * n_keys * head_dim;
        let mut temp = reg.stream.alloc_zeros::<f32>(total)?;
        let f = reg.get_fn("fused_attention", "tq_centroid_lookup_f32")?;
        let block = head_dim.min(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_heads as u32, n_keys as u32, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nkv, nk, hd, b, ms) = (n_kv_heads as i32, n_keys as i32, head_dim as i32, bits as i32, max_seq as i32);
        let ko = key_offset as i32;
        let os = n_keys as i32; // temp is contiguous
        unsafe {
            reg.stream.launch_builder(&f)
                .arg(packed_indices).arg(norms).arg(centroids)
                .arg(&mut temp)
                .arg(&nkv).arg(&nk).arg(&hd).arg(&b).arg(&ms).arg(&ko).arg(&os)
                .launch(cfg)?;
        }
        hadamard_inverse_batch(reg, &mut temp, signs, n_kv_heads * n_keys, head_dim)?;

        // Scatter from contiguous temp to strided output
        for h in 0..n_kv_heads {
            let src_off = h * n_keys * head_dim;
            let dst_off = h * out_stride * head_dim;
            copy_with_offsets(reg, &temp, keys_out, n_keys * head_dim, src_off, dst_off)?;
        }
    }

    Ok(())
}

/// Full fused TQ decode attention: compressed score + online softmax + V accumulation.
/// Single kernel replaces: decompress → matmul → softmax → matmul chain.
pub fn tq_fused_decode_attention(
    reg: &KernelRegistry,
    query: &CudaSlice<f32>,          // [n_heads, head_dim] pre-rotated
    packed_indices: &CudaSlice<u8>,  // [n_kv_heads, max_seq, bytes_per_key]
    norms: &CudaSlice<f32>,          // [n_kv_heads, max_seq]
    centroids: &CudaSlice<f32>,      // [n_centroids]
    v: &CudaSlice<f32>,              // [n_kv_heads, max_seq, head_dim]
    output: &mut CudaSlice<f32>,     // [n_heads, head_dim]
    n_heads: usize,
    n_kv_heads: usize,
    n_keys: usize,                   // actual number of compressed keys
    head_dim: usize,
    bits: usize,
    scale: f32,
    max_seq: usize,                  // buffer stride (pre-allocated size)
    // Sink token support (set n_sink=0 to disable)
    sink_k: Option<&CudaSlice<f32>>,
    sink_v: Option<&CudaSlice<f32>>,
    raw_query: Option<&CudaSlice<f32>>,
    n_sink: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_attention", "tq_fused_decode_attention_f32")?;
    let block_dim = 32u32.min(head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let nh = n_heads as i32;
    let nkv = n_kv_heads as i32;
    let nk = n_keys as i32;
    let hd = head_dim as i32;
    let b = bits as i32;
    let ms = max_seq as i32;
    let ns = n_sink as i32;
    static EMPTY_BUF: std::sync::OnceLock<CudaSlice<f32>> = std::sync::OnceLock::new();
    let empty = EMPTY_BUF.get_or_init(|| {
        reg.stream.alloc_zeros(1).expect("TQ empty buf alloc")
    });
    let sk = sink_k.unwrap_or(empty);
    let sv = sink_v.unwrap_or(empty);
    let rq = raw_query.unwrap_or(empty);
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(query).arg(packed_indices).arg(norms).arg(centroids)
            .arg(v).arg(output)
            .arg(&nh).arg(&nkv).arg(&nk).arg(&hd).arg(&b).arg(&scale).arg(&ms)
            .arg(sk).arg(sv).arg(rq).arg(&ns)
            .launch(cfg)?;
    }
    Ok(())
}

/// Graph-safe GQA decode attention: reads seq_len from GPU scalar.
/// Use for CUDA Graph replay — update seq_len_ptr via memcpy_htod before launch.
pub fn gqa_decode_attention_graph(
    reg: &KernelRegistry,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    seq_len_ptr: &CudaSlice<i32>,     // GPU scalar (pre-append value)
    n_heads: usize,
    n_kv_heads: usize,
    max_seq: usize,
    head_dim: usize,
    scale: f32,
    extra: i32,  // tokens appended after valid_len was set (typically 1)
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_attention", "gqa_decode_attention_graph_f32")?;
    let block_dim = 32u32.min(head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let nh = n_heads as i32;
    let nkv = n_kv_heads as i32;
    let ms = max_seq as i32;
    let hd = head_dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(q).arg(k).arg(v).arg(output).arg(seq_len_ptr)
            .arg(&nh).arg(&nkv).arg(&ms).arg(&hd).arg(&scale).arg(&extra)
            .launch(cfg)?;
    }
    Ok(())
}

/// GQA decode attention: fused Q@K^T + scale + softmax + @V.
/// Handles GQA head mapping internally — no repeat_kv copy needed.
/// K/V are pre-allocated padded buffers [n_kv_heads, max_seq, head_dim].
pub fn gqa_decode_attention(
    reg: &KernelRegistry,
    q: &CudaSlice<f32>,           // [n_heads, head_dim] (flattened [1, n_heads, 1, head_dim])
    k: &CudaSlice<f32>,           // [n_kv_heads, max_seq, head_dim]
    v: &CudaSlice<f32>,           // [n_kv_heads, max_seq, head_dim]
    output: &mut CudaSlice<f32>,  // [n_heads, head_dim]
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    max_seq: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_attention", "gqa_decode_attention_f32")?;
    let block_dim = 32u32.min(head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let nh = n_heads as i32;
    let nkv = n_kv_heads as i32;
    let sl = seq_len as i32;
    let ms = max_seq as i32;
    let hd = head_dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(q).arg(k).arg(v).arg(output)
            .arg(&nh).arg(&nkv).arg(&sl).arg(&ms).arg(&hd).arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch hadamard_inverse_batch_f32 for key decompression.
pub fn hadamard_inverse_batch(
    reg: &KernelRegistry,
    data: &mut CudaSlice<f32>,
    signs: &CudaSlice<f32>,
    n_vectors: usize,
    dim: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("hadamard", "hadamard_inverse_batch_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (n_vectors as u32, 1, 1),
        block_dim: (dim as u32, 1, 1),
        shared_mem_bytes: (dim * 4) as u32,
    };
    let nv = n_vectors as i32;
    let d = dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(data).arg(signs)
            .arg(&nv).arg(&d)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch hadamard_forward_batch_f32: forward randomized Hadamard for query pre-rotation.
/// y = (1/√d) * H * D * x where D = diag(signs).
/// Input `data` is [n_vectors × dim], modified in-place.
/// Signs are applied internally (no separate mul step needed).
pub fn hadamard_forward_batch(
    reg: &KernelRegistry,
    data: &mut CudaSlice<f32>,
    signs: &CudaSlice<f32>,
    n_vectors: usize,
    dim: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("hadamard", "hadamard_forward_batch_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (n_vectors as u32, 1, 1),
        block_dim: (dim as u32, 1, 1),
        shared_mem_bytes: (dim * 4) as u32,
    };
    let nv = n_vectors as i32;
    let d = dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(data).arg(signs)
            .arg(&nv).arg(&d)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch rope_halved_f32: in-place halved RoPE (Qwen2, Mistral).
/// x: [n_tokens * n_heads * head_dim] flat — modified in-place.
pub fn rope_halved(
    reg: &KernelRegistry,
    x: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos_offset: usize,
) -> Result<(), DriverError> {
    rope_halved_with_gpu_pos(reg, x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset, None)
}

/// Launch rope_halved_f32 with optional GPU position scalar (graph-replay-safe).
pub fn rope_halved_with_gpu_pos(
    reg: &KernelRegistry,
    x: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos_offset: usize,
    pos_offset_gpu: Option<&CudaSlice<i32>>,
) -> Result<(), DriverError> {
    let f = reg.get_fn("rope", "rope_halved_f32")?;
    // Kernel uses 2D grid: blockIdx.x = token, blockIdx.y = head.
    // Each block needs enough threads to cover rope_dim/2 elements.
    let half = rope_dim / 2;
    let threads = ((half.max(32) + 31) / 32 * 32).min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, n_heads as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nt = n_tokens as i32;
    let nh = n_heads as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let po = pos_offset as i32;
    let null_positions: u64 = 0;
    let null_gpu_pos: u64 = 0;
    unsafe {
        if let Some(gpu_ptr) = pos_offset_gpu {
            reg.stream.launch_builder(&f)
                .arg(x).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nh).arg(&hd).arg(&rd).arg(&po)
                .arg(gpu_ptr)
                .launch(cfg)?;
        } else {
            reg.stream.launch_builder(&f)
                .arg(x).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nh).arg(&hd).arg(&rd).arg(&po)
                .arg(&null_gpu_pos)
                .launch(cfg)?;
        }
    }
    Ok(())
}

/// Launch rope_interleaved_f32: in-place interleaved RoPE (Llama).
pub fn rope_interleaved(
    reg: &KernelRegistry,
    x: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos_offset: usize,
) -> Result<(), DriverError> {
    rope_interleaved_with_gpu_pos(reg, x, cos, sin, n_tokens, n_heads, head_dim, rope_dim, pos_offset, None)
}

/// Launch rope_interleaved_f32 with optional GPU position scalar.
pub fn rope_interleaved_with_gpu_pos(
    reg: &KernelRegistry,
    x: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos_offset: usize,
    pos_offset_gpu: Option<&CudaSlice<i32>>,
) -> Result<(), DriverError> {
    let f = reg.get_fn("rope", "rope_interleaved_f32")?;
    // Kernel uses 2D grid: blockIdx.x = token, blockIdx.y = head.
    let half = rope_dim / 2;
    let threads = ((half.max(32) + 31) / 32 * 32).min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, n_heads as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nt = n_tokens as i32;
    let nh = n_heads as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let po = pos_offset as i32;
    let null_positions: u64 = 0;
    let null_gpu_pos: u64 = 0;
    unsafe {
        if let Some(gpu_ptr) = pos_offset_gpu {
            reg.stream.launch_builder(&f)
                .arg(x).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nh).arg(&hd).arg(&rd).arg(&po)
                .arg(gpu_ptr)
                .launch(cfg)?;
        } else {
            reg.stream.launch_builder(&f)
                .arg(x).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nh).arg(&hd).arg(&rd).arg(&po)
                .arg(&null_gpu_pos)
                .launch(cfg)?;
        }
    }
    Ok(())
}

// ─── Tensor shape/elementwise ops (GPU-native) ──────────────

/// GPU strided copy: narrow + transpose via stride remapping.
pub fn strided_copy(
    reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
    n: usize, rank: usize,
    out_shape: &CudaSlice<i32>, out_strides: &CudaSlice<i32>,
    src_strides: &CudaSlice<i32>, src_offset: i32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "strided_copy_f32")?;
    let ni = n as i32; let ri = rank as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input).arg(output).arg(&ni).arg(&ri)
            .arg(out_shape).arg(out_strides).arg(src_strides).arg(&src_offset)
            .launch(launch_1d(n))?;
    }
    Ok(())
}

/// Strided copy with shape/strides as kernel args (no GPU buffer uploads).
/// Eliminates 3 clone_htod calls per invocation.
pub fn strided_copy_args(
    reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
    n: usize, rank: usize,
    out_shape: &[i32], out_strides: &[i32], src_strides: &[i32], src_offset: i32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "strided_copy_args_f32")?;
    let ni = n as i32; let ri = rank as i32;
    let pad = |s: &[i32]| -> [i32; 6] {
        let mut a = [0i32; 6];
        for (i, &v) in s.iter().enumerate().take(6) { a[i] = v; }
        a
    };
    let sh = pad(out_shape);
    let os = pad(out_strides);
    let ss = pad(src_strides);
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input).arg(output).arg(&ni).arg(&ri)
            .arg(&sh[0]).arg(&sh[1]).arg(&sh[2]).arg(&sh[3]).arg(&sh[4]).arg(&sh[5])
            .arg(&os[0]).arg(&os[1]).arg(&os[2]).arg(&os[3]).arg(&os[4]).arg(&os[5])
            .arg(&ss[0]).arg(&ss[1]).arg(&ss[2]).arg(&ss[3]).arg(&ss[4]).arg(&ss[5])
            .arg(&src_offset)
            .launch(launch_1d(n))?;
    }
    Ok(())
}

/// Broadcast binary op with strides as kernel args (no GPU buffer uploads).
/// Eliminates 4 clone_htod calls per invocation. `op`: "add", "mul", "sub", "div".
pub fn broadcast_binop_args(
    reg: &KernelRegistry, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>, n: usize, rank: usize,
    out_strides: &[i32], a_strides: &[i32], b_strides: &[i32],
    op: &str,
) -> Result<(), DriverError> {
    let kernel_name = format!("broadcast_{}_args_f32", op);
    let f = reg.get_fn("tensor_ops", &kernel_name)?;
    let ni = n as i32; let ri = rank as i32;
    let pad = |s: &[i32]| -> [i32; 6] {
        let mut a = [0i32; 6];
        for (i, &v) in s.iter().enumerate().take(6) { a[i] = v; }
        a
    };
    let os = pad(out_strides);
    let ast = pad(a_strides);
    let bs = pad(b_strides);
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(a).arg(b).arg(output).arg(&ni).arg(&ri)
            .arg(&os[0]).arg(&os[1]).arg(&os[2]).arg(&os[3]).arg(&os[4]).arg(&os[5])
            .arg(&ast[0]).arg(&ast[1]).arg(&ast[2]).arg(&ast[3]).arg(&ast[4]).arg(&ast[5])
            .arg(&bs[0]).arg(&bs[1]).arg(&bs[2]).arg(&bs[3]).arg(&bs[4]).arg(&bs[5])
            .launch(launch_1d(n))?;
    }
    Ok(())
}

pub fn gpu_exp(reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>, n: usize) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "exp_f32")?;
    unsafe { reg.stream.launch_builder(&f).arg(input).arg(output).arg(&(n as i32)).launch(launch_1d(n))?; }
    Ok(())
}

pub fn gpu_sqrt(reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>, n: usize) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "sqrt_f32")?;
    unsafe { reg.stream.launch_builder(&f).arg(input).arg(output).arg(&(n as i32)).launch(launch_1d(n))?; }
    Ok(())
}

pub fn gpu_cos(reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>, n: usize) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "cos_f32")?;
    unsafe { reg.stream.launch_builder(&f).arg(input).arg(output).arg(&(n as i32)).launch(launch_1d(n))?; }
    Ok(())
}

pub fn gpu_sin(reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>, n: usize) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "sin_f32")?;
    unsafe { reg.stream.launch_builder(&f).arg(input).arg(output).arg(&(n as i32)).launch(launch_1d(n))?; }
    Ok(())
}

pub fn gpu_sqr(reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>, n: usize) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "sqr_f32")?;
    unsafe { reg.stream.launch_builder(&f).arg(input).arg(output).arg(&(n as i32)).launch(launch_1d(n))?; }
    Ok(())
}

pub fn gpu_broadcast_add(
    reg: &KernelRegistry, a: &CudaSlice<f32>, b: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
    n: usize, rank: usize,
    out_shape: &CudaSlice<i32>, out_strides: &CudaSlice<i32>,
    a_strides: &CudaSlice<i32>, b_strides: &CudaSlice<i32>,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "broadcast_add_f32")?;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(a).arg(b).arg(output).arg(&(n as i32)).arg(&(rank as i32))
            .arg(out_shape).arg(out_strides).arg(a_strides).arg(b_strides)
            .launch(launch_1d(n))?;
    }
    Ok(())
}

pub fn gpu_broadcast_mul(
    reg: &KernelRegistry, a: &CudaSlice<f32>, b: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
    n: usize, rank: usize,
    out_shape: &CudaSlice<i32>, out_strides: &CudaSlice<i32>,
    a_strides: &CudaSlice<i32>, b_strides: &CudaSlice<i32>,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "broadcast_mul_f32")?;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(a).arg(b).arg(output).arg(&(n as i32)).arg(&(rank as i32))
            .arg(out_shape).arg(out_strides).arg(a_strides).arg(b_strides)
            .launch(launch_1d(n))?;
    }
    Ok(())
}

pub fn gpu_broadcast_sub(
    reg: &KernelRegistry, a: &CudaSlice<f32>, b: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
    n: usize, rank: usize,
    out_shape: &CudaSlice<i32>, out_strides: &CudaSlice<i32>,
    a_strides: &CudaSlice<i32>, b_strides: &CudaSlice<i32>,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "broadcast_sub_f32")?;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(a).arg(b).arg(output).arg(&(n as i32)).arg(&(rank as i32))
            .arg(out_shape).arg(out_strides).arg(a_strides).arg(b_strides)
            .launch(launch_1d(n))?;
    }
    Ok(())
}

pub fn gpu_reduce_sum_last(
    reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
    rows: usize, cols: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "reduce_sum_last_f32")?;
    unsafe { reg.stream.launch_builder(&f).arg(input).arg(output).arg(&(rows as i32)).arg(&(cols as i32)).launch(launch_1d(rows))?; }
    Ok(())
}

pub fn gpu_reduce_max_last(
    reg: &KernelRegistry, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
    rows: usize, cols: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "reduce_max_last_f32")?;
    unsafe { reg.stream.launch_builder(&f).arg(input).arg(output).arg(&(rows as i32)).arg(&(cols as i32)).launch(launch_1d(rows))?; }
    Ok(())
}

/// GPU concat copy: dst[dst_offset + i] = src[i] for i in 0..n.
pub fn concat_copy(
    reg: &KernelRegistry,
    src: &CudaSlice<f32>,
    dst: &CudaSlice<f32>,  // written to via kernel (logically mutable)
    n: usize,
    dst_offset: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "concat_copy_f32")?;
    let ni = n as i32;
    let di = dst_offset as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(src).arg(dst).arg(&ni).arg(&di)
            .launch(launch_1d(n))?;
    }
    Ok(())
}

/// Copy with source and destination offsets: dst[dst_off+i] = src[src_off+i].
/// Graph-capture safe: no clone_htod, no temp alloc. Replaces strided_copy+concat_copy in cat().
pub fn copy_with_offsets(
    reg: &KernelRegistry,
    src: &CudaSlice<f32>,
    dst: &CudaSlice<f32>,  // written via kernel
    n: usize,
    src_offset: usize,
    dst_offset: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "copy_with_offsets_f32")?;
    let ni = n as i32;
    let so = src_offset as i32;
    let do_ = dst_offset as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(src).arg(dst).arg(&ni).arg(&so).arg(&do_)
            .launch(launch_1d(n))?;
    }
    Ok(())
}

/// KV cache append: copy new K/V tokens into pre-allocated cache at dynamic GPU offset.
/// Single kernel launch for all heads (replaces per-head copy_with_offsets loop).
/// seq_pos_ptr is a GPU i32 scalar: updated before graph replay.
pub fn kv_cache_append(
    reg: &KernelRegistry,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    seq_pos_ptr: &CudaSlice<i32>,
    n_kv_head: usize,
    max_seq: usize,
    head_dim: usize,
    n_new: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "kv_cache_append_f32")?;
    let total = n_kv_head * n_new * head_dim;
    let nkv = n_kv_head as i32;
    let ms = max_seq as i32;
    let hd = head_dim as i32;
    let nn = n_new as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(src).arg(dst).arg(seq_pos_ptr)
            .arg(&nkv).arg(&ms).arg(&hd).arg(&nn)
            .launch(launch_1d(total))?;
    }
    Ok(())
}

/// GPU-side Q4_K_M dequantize: packed Q4K → full F32 matrix.
/// Grid: (n_superblocks, n_rows), Block: 128 threads.
/// Used for prefill streaming: temp alloc → dequant → cuBLAS SGEMM → free.
pub fn q4km_dequant_f32(
    reg: &KernelRegistry,
    w_packed: &CudaSlice<u8>,
    w_f32: &mut CudaSlice<f32>,
    n_rows: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("qmatmul", "q4km_dequant_f32")?;
    let n_sb = (in_features + 255) / 256; // QK_K = 256, ceiling division
    let cfg = LaunchConfig {
        grid_dim: (n_sb as u32, n_rows as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let nr = n_rows as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_packed)
            .arg(w_f32)
            .arg(&nr)
            .arg(&inf)
            .launch(cfg)?;
    }
    Ok(())
}

/// GPU-side Q4_K_M dequantize to FP16: packed Q4K → half matrix.
/// Grid: (n_superblocks, n_rows), Block: 128 threads.
/// Half the scratch size of F32 dequant — feeds directly into cuBLAS HGEMM.
pub fn q4km_dequant_f16(
    reg: &KernelRegistry,
    w_packed: &CudaSlice<u8>,
    w_f16: &mut CudaSlice<half::f16>,
    n_rows: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("qmatmul", "q4km_dequant_f16")?;
    let n_sb = (in_features + 255) / 256; // QK_K = 256, ceiling division
    let cfg = LaunchConfig {
        grid_dim: (n_sb as u32, n_rows as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let nr = n_rows as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_packed)
            .arg(w_f16)
            .arg(&nr)
            .arg(&inf)
            .launch(cfg)?;
    }
    Ok(())
}

/// GPU-side Q6_K dequantize: packed Q6K → full F32 matrix.
/// Grid: (n_superblocks, n_rows), Block: 256 threads (1 thread per value).
pub fn q6k_dequant_f32(
    reg: &KernelRegistry,
    w_packed: &CudaSlice<u8>,
    w_f32: &mut CudaSlice<f32>,
    n_rows: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("qmatmul", "q6k_dequant_f32")?;
    let n_sb = (in_features + 255) / 256; // QK_K = 256, ceiling division
    let cfg = LaunchConfig {
        grid_dim: (n_sb as u32, n_rows as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let nr = n_rows as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_packed)
            .arg(w_f32)
            .arg(&nr)
            .arg(&inf)
            .launch(cfg)?;
    }
    Ok(())
}

/// F32 matvec: output = W @ x. No dequant — for pre-dequantized cached weights.
/// 1 block per output row, 256 threads. Replaces cuBLAS SGEMM for decode (m=1).
pub fn f32_matvec(
    reg: &KernelRegistry,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    out_features: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "f32_matvec")?;
    let cfg = launch_per_row(out_features, 256);
    let of = out_features as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w).arg(x).arg(output)
            .arg(&of).arg(&inf)
            .launch(cfg)?;
    }
    Ok(())
}

/// Generate KV cache attention mask: 0.0 for valid positions, -1e10 for padding.
/// Reads valid_len from a GPU scalar buffer (graph-replay-safe: update the scalar before replay).
/// Generate KV cache attention mask. `extra` = tokens being appended (added to *valid_len_ptr).
pub fn generate_kv_mask(
    reg: &KernelRegistry,
    mask: &CudaSlice<f32>,
    valid_len_ptr: &CudaSlice<i32>,
    max_seq: usize,
    extra: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "generate_kv_mask_f32")?;
    let ms = max_seq as i32;
    let ex = extra as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(mask).arg(valid_len_ptr).arg(&ms).arg(&ex)
            .launch(launch_1d(max_seq))?;
    }
    Ok(())
}

// ─── Fused layer kernels (fused_layer.cu) ──────────────────────
//
// These replace 13 separate kernel launches per layer with 3:
//   Kernel 1: norm + QKV projection + bias
//   Kernel 2: residual add + norm + gate/up projection + silu*mul
//   Kernel 3: down projection + residual add

/// Fused RmsNorm + Q4_K_M QKV projection + bias.
/// Replaces: rms_norm + 3× q4km_matvec + 3× add (7 launches → 1).
/// Grid: (q_out + k_out + v_out) blocks, 256 threads.
pub fn fused_norm_q4km_qkv_bias(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,
    norm_weight: &CudaSlice<f32>,
    w_q: &CudaSlice<u8>,
    w_k: &CudaSlice<u8>,
    w_v: &CudaSlice<u8>,
    bias_q: Option<&CudaSlice<f32>>,
    bias_k: Option<&CudaSlice<f32>>,
    bias_v: Option<&CudaSlice<f32>>,
    out_q: &mut CudaSlice<f32>,
    out_k: &mut CudaSlice<f32>,
    out_v: &mut CudaSlice<f32>,
    hidden_dim: usize,
    q_out: usize,
    k_out: usize,
    v_out: usize,
    eps: f32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_layer", "fused_norm_q4km_qkv_bias_f32")?;
    let total_rows = (q_out + k_out + v_out) as u32;
    let block = 256u32;
    let shmem = (hidden_dim as u32) * 4; // f32 per element
    let cfg = launch_with_shmem(total_rows, block, shmem);
    let hd = hidden_dim as i32;
    let qo = q_out as i32;
    let ko = k_out as i32;
    let vo = v_out as i32;
    let null: u64 = 0;
    unsafe {
        let mut builder = reg.stream.launch_builder(&f);
        builder.arg(input).arg(norm_weight)
            .arg(w_q).arg(w_k).arg(w_v);
        // Bias pointers: pass null (0u64) when None
        if let Some(b) = bias_q { builder.arg(b); } else { builder.arg(&null); }
        if let Some(b) = bias_k { builder.arg(b); } else { builder.arg(&null); }
        if let Some(b) = bias_v { builder.arg(b); } else { builder.arg(&null); }
        builder.arg(out_q).arg(out_k).arg(out_v)
            .arg(&hd).arg(&qo).arg(&ko).arg(&vo).arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Fused RmsNorm + gate/up Q4_K_M projection + SiLU*mul.
/// Input must be pre-combined (residual + attn_out) by caller.
/// Replaces: rms_norm + 2× q4km_matvec + silu_mul (4 launches → 1).
/// Grid: intermediate_dim blocks, 256 threads.
pub fn fused_addnorm_q4km_gateup_silu(
    reg: &KernelRegistry,
    input: &CudaSlice<f32>,          // pre-combined: residual + attn_out
    norm_weight: &CudaSlice<f32>,
    w_gate: &CudaSlice<u8>,
    w_up: &CudaSlice<u8>,
    intermediate_out: &mut CudaSlice<f32>,
    hidden_dim: usize,
    intermediate_dim: usize,
    eps: f32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_layer", "fused_addnorm_q4km_gateup_silu_f32")?;
    let block = 256u32;
    let shmem = (hidden_dim as u32) * 4;
    let cfg = launch_with_shmem(intermediate_dim as u32, block, shmem);
    let hd = hidden_dim as i32;
    let id = intermediate_dim as i32;
    let null: u64 = 0; // _unused param (ABI compat)
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(input).arg(&null).arg(norm_weight)
            .arg(w_gate).arg(w_up)
            .arg(intermediate_out)
            .arg(&hd).arg(&id).arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Fused Q4_K_M down projection + residual add.
/// Replaces: q4km_matvec + add (2 launches → 1).
/// Grid: hidden_dim blocks, 256 threads.
/// NOTE: residual is updated in-place (residual += W_down @ intermediate).
pub fn fused_q4km_down_residual(
    reg: &KernelRegistry,
    w_down: &CudaSlice<u8>,
    intermediate: &CudaSlice<f32>,
    residual: &mut CudaSlice<f32>,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_layer", "fused_q4km_down_residual_f32")?;
    let cfg = launch_per_row(hidden_dim, 256);
    let hd = hidden_dim as i32;
    let id = intermediate_dim as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_down).arg(intermediate).arg(residual)
            .arg(&hd).arg(&id)
            .launch(cfg)?;
    }
    Ok(())
}
