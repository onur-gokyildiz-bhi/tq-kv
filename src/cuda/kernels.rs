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
            // flash_attention: deferred (PTX JIT issue on sm_86)
            // ("flash_attention", PTX_FLASH_ATTENTION),
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

/// Launch q8_0_matvec_f32: fused Q8_0 dequant + matvec.
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
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(query).arg(packed_indices).arg(norms).arg(centroids).arg(scores_out)
            .arg(&nh).arg(&nkv).arg(&nk).arg(&hd).arg(&b).arg(&scale)
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
    let f = reg.get_fn("rope", "rope_halved_f32")?;
    let total_half_pairs = n_tokens * n_heads * (rope_dim / 2);
    let cfg = launch_1d(total_half_pairs);
    let nt = n_tokens as i32;
    let nh = n_heads as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let po = pos_offset as i32;
    let null_positions: u64 = 0;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(x)
            .arg(cos)
            .arg(sin)
            .arg(&null_positions)
            .arg(&nt).arg(&nh).arg(&hd).arg(&rd).arg(&po)
            .launch(cfg)?;
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
    let f = reg.get_fn("rope", "rope_interleaved_f32")?;
    let total_half_pairs = n_tokens * n_heads * (rope_dim / 2);
    let cfg = launch_1d(total_half_pairs);
    let nt = n_tokens as i32;
    let nh = n_heads as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let po = pos_offset as i32;
    let null_positions: u64 = 0;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(x)
            .arg(cos)
            .arg(sin)
            .arg(&null_positions)
            .arg(&nt).arg(&nh).arg(&hd).arg(&rd).arg(&po)
            .launch(cfg)?;
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
pub fn generate_kv_mask(
    reg: &KernelRegistry,
    mask: &CudaSlice<f32>,
    valid_len_ptr: &CudaSlice<i32>,
    max_seq: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("tensor_ops", "generate_kv_mask_f32")?;
    let ms = max_seq as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(mask).arg(valid_len_ptr).arg(&ms)
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
