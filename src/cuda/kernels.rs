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
    /// Initialize the kernel registry — detects GPU arch, loads matching PTX modules.
    pub fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, sm_major: u32, sm_minor: u32) -> Result<Self, DriverError> {
        let mut reg = Self {
            ctx: ctx.clone(),
            stream: stream.clone(),
            modules: std::collections::HashMap::new(),
            cublas: std::sync::OnceLock::new(),
        };
        reg.load_all_ptx(sm_major, sm_minor)?;
        Ok(reg)
    }

    /// Get or create the cached cuBLAS handle.
    pub fn get_cublas(&self) -> &cudarc::cublas::CudaBlas {
        self.cublas.get_or_init(|| {
            cudarc::cublas::CudaBlas::new(self.stream.clone())
                .expect("cuBLAS handle creation failed")
        })
    }

    /// Load PTX modules for the best matching GPU architecture.
    fn load_all_ptx(&mut self, sm_major: u32, sm_minor: u32) -> Result<(), DriverError> {
        let arch = best_compiled_arch(sm_major, sm_minor);
        let ptx_sources = ptx_sources_for_arch(arch);
        eprintln!("[cuda] GPU sm_{}{} → loading PTX for sm_{} ({} compiled arches: {:?})",
            sm_major, sm_minor, arch, COMPILED_ARCHES.len(), COMPILED_ARCHES);

        if ptx_sources.is_empty() {
            eprintln!("[cuda] ERROR: no PTX available for arch sm_{}", arch);
            return Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_PTX));
        }

        let mut loaded = 0;
        let mut skipped = Vec::new();
        for &(name, ptx_src) in ptx_sources {
            // flash_attention: deferred (PTX JIT issue with mma.h)
            if name == "flash_attention" {
                skipped.push(name);
                continue;
            }
            match self.load_ptx_module(name, ptx_src) {
                Ok(()) => loaded += 1,
                Err(e) => eprintln!("[cuda] WARNING: failed to load {} PTX (sm_{}): {:?}", name, arch, e),
            }
        }
        if !skipped.is_empty() {
            eprintln!("[cuda] Skipped: {:?}", skipped);
        }
        eprintln!("[cuda] Loaded {}/{} PTX modules (sm_{})", loaded, ptx_sources.len() - skipped.len(), arch);
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
    // Q4_K_M matvec variants:
    //   default (mrow8):      8 rows/block cpasync X+W pipeline
    //   wx_cpasync:           4 rows/block cpasync X+W (TQ_Q4KM=wx)
    //   baseline:             original (X cp.async only)  (TQ_Q4KM=baseline)
    //   dp4a:                 INT8 tensor-pipe via __dp4a (TQ_Q4KM=dp4a)
    //                         — requires on-the-fly q8_1 activation quantize
    //                         — in_features must be multiple of QK8_1 (32)
    static USE_DP4A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let use_dp4a = *USE_DP4A.get_or_init(||
        std::env::var("TQ_Q4KM").ok().as_deref() == Some("dp4a")
    );
    if use_dp4a && in_features % QK8_1 == 0 {
        let n_blocks = in_features / QK8_1;
        let bytes_needed = n_blocks * Q8_1_BLOCK_BYTES;
        let mut x_q8_1: CudaSlice<u8> = reg.stream.alloc_zeros::<u8>(bytes_needed)?;
        quantize_f32_to_q8_1(reg, x, &mut x_q8_1, in_features)?;
        return q4km_matvec_dp4a(reg, w_packed, &x_q8_1, output, out_features, in_features);
    }
    static VARIANT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let kernel_name = *VARIANT.get_or_init(|| {
        match std::env::var("TQ_Q4KM").ok().as_deref() {
            Some("baseline") => "q4km_matvec_f32",
            Some("wx")       => "q4km_matvec_wx_cpasync_f32",
            Some("mrow16")   => "q4km_matvec_mrow16_f32",
            _                => "q4km_matvec_mrow8_f32",
        }
    });
    let f = reg.get_fn("qmatmul", kernel_name)?;
    let rows_per_block = if *kernel_name == *"q4km_matvec_mrow8_f32" {
        8u32
    } else if *kernel_name == *"q4km_matvec_mrow16_f32" {
        16u32
    } else {
        4u32
    };
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
    // Q6K matvec variants:
    //   default:   4 rows/block — best on Llama3 (Q6K dequant heavier than Q4K,
    //              mrow8 spilled registers and regressed -8..19%)
    //   mrow8:     8 rows/block (TQ_Q6K=mrow8) — opt-in for future HW
    static VARIANT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let kernel_name = *VARIANT.get_or_init(|| {
        match std::env::var("TQ_Q6K").ok().as_deref() {
            Some("mrow8") => "q6k_matvec_mrow8_f32",
            _             => "q6k_matvec_f32",
        }
    });
    let f = reg.get_fn("qmatmul", kernel_name)?;
    let rows_per_block = if *kernel_name == *"q6k_matvec_mrow8_f32" { 8u32 } else { 4u32 };
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

/// Block size of GGUF q8_1 (must equal sizeof(block_q8_1) in ggml-common.h).
/// Layout: half2 ds (4B) + int8 qs[32] (32B) = 36 bytes. No padding.
pub const Q8_1_BLOCK_BYTES: usize = 36;
pub const QK8_1: usize = 32;

/// Launch quantize_f32_to_q8_1_f32: convert FP32 activations to GGUF q8_1
/// blocks (prerequisite for the dp4a q4km×q8_1 matvec — MMVQ Step 1).
///
/// `n_elements` must be a multiple of `QK8_1` (32). The output buffer must
/// hold at least `(n_elements / QK8_1) * Q8_1_BLOCK_BYTES` bytes.
///
/// One CUDA block = one warp = one q8_1 block (32 elements).
pub fn quantize_f32_to_q8_1(
    reg: &KernelRegistry,
    x: &CudaSlice<f32>,
    out_q8_1: &mut CudaSlice<u8>,
    n_elements: usize,
) -> Result<(), DriverError> {
    debug_assert!(
        n_elements % QK8_1 == 0,
        "quantize_f32_to_q8_1: n_elements ({}) must be multiple of {}",
        n_elements, QK8_1
    );
    let n_blocks = n_elements / QK8_1;
    debug_assert!(
        out_q8_1.len() >= n_blocks * Q8_1_BLOCK_BYTES,
        "quantize_f32_to_q8_1: out buffer {} bytes < required {}",
        out_q8_1.len(), n_blocks * Q8_1_BLOCK_BYTES
    );
    let f = reg.get_fn("qmatmul", "quantize_f32_to_q8_1_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks as u32, 1, 1),
        block_dim: (32, 1, 1),     // one warp per block
        shared_mem_bytes: 0,
    };
    let n_el = n_elements as i32;
    let n_bl = n_blocks as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(x)
            .arg(out_q8_1)
            .arg(&n_el)
            .arg(&n_bl)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch `q4km_matvec_dp4a_f32`: Q4_K_M × q8_1 matvec using `__dp4a`.
///
/// MMVQ Step 2 — INT8 tensor-pipe decode path. Pre-quantize the activation to
/// q8_1 via [`quantize_f32_to_q8_1`] first, then call this launcher. Requires
/// sm_61+ (Pascal) for the `__dp4a` intrinsic; assumed available on Ampere+.
/// No env-gate: main session decides when to flip the default.
///
/// `in_features` must be a multiple of 256 (QK_K). The `x_q8_1` buffer must
/// hold `(in_features / 32) * Q8_1_BLOCK_BYTES` bytes.
pub fn q4km_matvec_dp4a(
    reg: &KernelRegistry,
    w_packed: &CudaSlice<u8>,
    x_q8_1: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    out_features: usize,
    in_features: usize,
) -> Result<(), DriverError> {
    debug_assert!(
        in_features % 256 == 0,
        "q4km_matvec_dp4a: in_features ({}) must be multiple of 256",
        in_features
    );
    let expected_q8_1 = (in_features / QK8_1) * Q8_1_BLOCK_BYTES;
    debug_assert!(
        x_q8_1.len() >= expected_q8_1,
        "q4km_matvec_dp4a: x_q8_1 buffer {} bytes < required {}",
        x_q8_1.len(), expected_q8_1
    );
    debug_assert!(
        output.len() >= out_features,
        "q4km_matvec_dp4a: output buffer {} elems < out_features {}",
        output.len(), out_features
    );
    let f = reg.get_fn("qmatmul", "q4km_matvec_dp4a_f32")?;
    let n_blocks = ((out_features as u32) + 7) / 8;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (256, 1, 1),       // 8 warps × 32 lanes
        shared_mem_bytes: 0,
    };
    let of = out_features as i32;
    let inf = in_features as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(w_packed)
            .arg(x_q8_1)
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

/// Softcap in-place: data = cap * tanh(data / cap). Gemma 2 attention/final logit capping.
pub fn softcap_inplace(
    reg: &KernelRegistry,
    data: &mut CudaSlice<f32>,
    cap: f32,
    n: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("elementwise", "softcap_inplace_f32")?;
    let cfg = launch_1d(n);
    let inv_cap = 1.0f32 / cap;
    let ni = n as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(data)
            .arg(&inv_cap)
            .arg(&cap)
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
    means_out: Option<&mut CudaSlice<f32>>,  // [n_kv_heads] per-token means
    channel_sigma: Option<&CudaSlice<f32>>,  // [head_dim] per-channel sigma (KIVI)
    n_kv_heads: usize,
    head_dim: usize,
    n_centroids: usize,
    bytes_per_key: usize,
    center_keys: bool,
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
    let ck = if center_keys { 1i32 } else { 0i32 };
    let null_ptr: u64 = 0;
    // 4 optional pointer args: means_out × channel_sigma = 4 combinations
    unsafe {
        match (means_out, channel_sigma) {
            (Some(means), Some(sigma)) => {
                reg.stream.launch_builder(&f)
                    .arg(key_vectors).arg(signs).arg(boundaries).arg(centroids)
                    .arg(packed_out).arg(norms_out).arg(means).arg(sigma)
                    .arg(&hd).arg(&nc).arg(&bpk).arg(&ck)
                    .launch(cfg)?;
            }
            (Some(means), None) => {
                reg.stream.launch_builder(&f)
                    .arg(key_vectors).arg(signs).arg(boundaries).arg(centroids)
                    .arg(packed_out).arg(norms_out).arg(means).arg(&null_ptr)
                    .arg(&hd).arg(&nc).arg(&bpk).arg(&ck)
                    .launch(cfg)?;
            }
            (None, Some(sigma)) => {
                reg.stream.launch_builder(&f)
                    .arg(key_vectors).arg(signs).arg(boundaries).arg(centroids)
                    .arg(packed_out).arg(norms_out).arg(&null_ptr).arg(sigma)
                    .arg(&hd).arg(&nc).arg(&bpk).arg(&ck)
                    .launch(cfg)?;
            }
            (None, None) => {
                reg.stream.launch_builder(&f)
                    .arg(key_vectors).arg(signs).arg(boundaries).arg(centroids)
                    .arg(packed_out).arg(norms_out).arg(&null_ptr).arg(&null_ptr)
                    .arg(&hd).arg(&nc).arg(&bpk).arg(&ck)
                    .launch(cfg)?;
            }
        }
    }
    Ok(())
}

/// GPU TurboQuant key compression with PER-GROUP sigma.
/// Mirrors CPU compress_single_key_grouped: head_dim is split into n_groups
/// blocks of `group_size` each, every block carries its own L2 norm /
/// sigma, and the centroid lookup uses the per-group sigma. Output norms
/// are arranged as [n_kv_heads, n_groups] (row-major over heads).
///
/// Constraints: head_dim % group_size == 0. Center-keys + per-channel sigma
/// (KIVI) are not supported on this path yet — they layer on top of a single
/// per-vector sigma and will be redesigned in a later sprint.
pub fn tq_compress_key_grouped(
    reg: &KernelRegistry,
    key_vectors: &CudaSlice<f32>,    // [n_kv_heads, head_dim]
    signs: &CudaSlice<f32>,          // [head_dim]
    boundaries: &CudaSlice<f32>,     // [n_centroids - 1]
    centroids: &CudaSlice<f32>,      // [n_centroids]
    packed_out: &mut CudaSlice<u8>,  // [n_kv_heads, bytes_per_key]
    gnorms_out: &mut CudaSlice<f32>, // [n_kv_heads, n_groups]
    n_kv_heads: usize,
    head_dim: usize,
    group_size: usize,
    n_centroids: usize,
    bytes_per_key: usize,
) -> Result<(), DriverError> {
    assert!(group_size > 0, "group_size must be > 0");
    assert!(head_dim % group_size == 0,
        "head_dim {} must be divisible by group_size {}", head_dim, group_size);
    let f = reg.get_fn("tq_compress", "tq_compress_key_grouped_f32")?;
    let n_boundaries = n_centroids - 1;
    let n_groups = head_dim / group_size;
    // Shared memory layout: s_data + s_bounds + s_cents + s_gsig + s_gnorm
    let shmem = ((head_dim + n_boundaries + n_centroids + 2 * n_groups) * 4) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: shmem,
    };
    let hd = head_dim as i32;
    let gs = group_size as i32;
    let nc = n_centroids as i32;
    let bpk = bytes_per_key as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(key_vectors).arg(signs).arg(boundaries).arg(centroids)
            .arg(packed_out).arg(gnorms_out)
            .arg(&hd).arg(&gs).arg(&nc).arg(&bpk)
            .launch(cfg)?;
    }
    Ok(())
}

/// TriAttention: score all keys for all KV heads using trigonometric series.
/// Grid: (n_kv_heads, 1, 1), Block: 256 threads.
/// Scores pre-RoPE keys without full attention computation.
pub fn trig_score_keys_batched(
    reg: &KernelRegistry,
    q_centers: &CudaSlice<f32>,       // [n_kv_heads, head_dim]
    keys_pre_rope: &CudaSlice<f32>,   // [n_kv_heads, n_keys, head_dim]
    rope_freqs: &CudaSlice<f32>,      // [n_pairs]
    key_positions: &CudaSlice<i32>,   // [n_keys]
    scores_out: &mut CudaSlice<f32>,  // [n_kv_heads, n_keys]
    mrl_per_head: &CudaSlice<f32>,    // [n_kv_heads]
    q_norm_per_head: &CudaSlice<f32>, // [n_kv_heads]
    current_pos: usize,
    n_keys: usize,
    n_kv_heads: usize,
    head_dim: usize,
    offsets: &CudaSlice<i32>,         // [n_offsets]
    n_offsets: usize,
) -> Result<(), DriverError> {
    let f = reg.get_fn("trig_score", "trig_score_keys_batched_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0, // shared memory statically allocated in kernel
    };
    let cp = current_pos as i32;
    let nk = n_keys as i32;
    let hd = head_dim as i32;
    let no = n_offsets as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(q_centers).arg(keys_pre_rope).arg(rope_freqs)
            .arg(key_positions).arg(scores_out)
            .arg(mrl_per_head).arg(q_norm_per_head)
            .arg(&cp).arg(&nk).arg(&hd).arg(&no).arg(offsets)
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

/// Flash decode v2: split KV across blocks, each computes partial attention.
/// Grid: (n_splits, n_heads, batch), Block: 128 threads (4 warps).
/// 4x memory bandwidth vs v1 (32 threads). Call flash_decode_reduce after.
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
    max_seq: usize,
    window_size: i32,
) -> Result<(), DriverError> {
    let f = reg.get_fn("flash_decode", "flash_decode_partial")?;
    let n_splits = (seq_kv + split_size - 1) / split_size;
    let cfg = LaunchConfig {
        grid_dim: (n_splits as u32, n_heads as u32, batch_size as u32),
        block_dim: (128, 1, 1),  // v2: 4 warps for better memory throughput
        shared_mem_bytes: 0,
    };
    let bs = batch_size as i32;
    let nh = n_heads as i32;
    let nkv = n_kv_heads as i32;
    let skv = seq_kv as i32;
    let hd = head_dim as i32;
    let ss = split_size as i32;
    let ms = max_seq as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(q).arg(k).arg(v)
            .arg(partial_o).arg(partial_max).arg(partial_sum)
            .arg(&bs).arg(&nh).arg(&nkv).arg(&skv).arg(&hd).arg(&scale).arg(&ss).arg(&ms).arg(&window_size)
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
        block_dim: (128, 1, 1),  // v2: match partial kernel thread count
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

        // Scatter from contiguous temp to strided output.
        // Per-head destination row is `key_offset` (not 0) — incremental decode
        // writes a single new row at index `key_offset` while preserving prior rows.
        // Bug fixed 2026-04-11: previously dst_off omitted the key_offset shift,
        // so step-2+ decodes silently overwrote row 0 of every head.
        for h in 0..n_kv_heads {
            let src_off = h * n_keys * head_dim;
            let dst_off = (h * out_stride + key_offset) * head_dim;
            copy_with_offsets(reg, &temp, keys_out, n_keys * head_dim, src_off, dst_off)?;
        }
    }

    Ok(())
}

/// Per-group decompress (Sprint 1B). Mirror of tq_decompress_keys_v2 but
/// reads gnorms[n_kv_heads, max_seq, n_groups] and runs the grouped centroid
/// lookup. The hadamard inverse step is unchanged because the rotation
/// itself is independent of how sigma was computed. Output layout matches
/// the per-vector variant: [n_kv_heads, n_keys, head_dim] when contiguous.
pub fn tq_decompress_keys_grouped(
    reg: &KernelRegistry,
    packed_indices: &CudaSlice<u8>,
    gnorms: &CudaSlice<f32>,             // [n_kv_heads, max_seq, n_groups]
    centroids: &CudaSlice<f32>,
    signs: &CudaSlice<f32>,
    keys_out: &mut CudaSlice<f32>,
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    group_size: usize,
    bits: usize,
    max_seq: usize,
    key_offset: usize,
    out_stride: usize,
) -> Result<(), DriverError> {
    if n_keys == 0 { return Ok(()); }
    assert!(group_size > 0 && head_dim % group_size == 0,
        "tq_decompress_keys_grouped: head_dim {} not divisible by group_size {}",
        head_dim, group_size);

    let f = reg.get_fn("fused_attention", "tq_centroid_lookup_grouped_f32")?;
    let block = head_dim.min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, n_keys as u32, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let nkv = n_kv_heads as i32;
    let nk = n_keys as i32;
    let hd = head_dim as i32;
    let gs = group_size as i32;
    let b = bits as i32;
    let ms = max_seq as i32;
    let ko = key_offset as i32;
    let os = out_stride as i32;

    let contiguous = out_stride == n_keys && key_offset == 0;
    if contiguous {
        unsafe {
            reg.stream.launch_builder(&f)
                .arg(packed_indices).arg(gnorms).arg(centroids)
                .arg(&mut *keys_out)
                .arg(&nkv).arg(&nk).arg(&hd).arg(&gs).arg(&b).arg(&ms).arg(&ko).arg(&os)
                .launch(cfg)?;
        }
        hadamard_inverse_batch(reg, keys_out, signs, n_kv_heads * n_keys, head_dim)?;
    } else {
        // Strided write: lookup into a contiguous temp first, hadamard, then scatter.
        let total = n_kv_heads * n_keys * head_dim;
        let mut temp = reg.stream.alloc_zeros::<f32>(total)?;
        let temp_os = n_keys as i32;
        unsafe {
            reg.stream.launch_builder(&f)
                .arg(packed_indices).arg(gnorms).arg(centroids)
                .arg(&mut temp)
                .arg(&nkv).arg(&nk).arg(&hd).arg(&gs).arg(&b).arg(&ms).arg(&ko).arg(&temp_os)
                .launch(cfg)?;
        }
        hadamard_inverse_batch(reg, &mut temp, signs, n_kv_heads * n_keys, head_dim)?;
        // Same bug + fix as tq_decompress_keys_range above: dst_off must add
        // key_offset, otherwise step-2+ incremental decodes scribble row 0.
        for h in 0..n_kv_heads {
            let src_off = h * n_keys * head_dim;
            let dst_off = (h * out_stride + key_offset) * head_dim;
            copy_with_offsets(reg, &temp, keys_out, n_keys * head_dim, src_off, dst_off)?;
        }
    }

    Ok(())
}

/// Per-group fused attention launcher (Sprint 1B). Wires the existing
/// tq_fused_attention_grouped_f32 kernel which has been dead code until
/// now. Reads gnorms[n_kv_heads, n_keys, n_groups] and produces the
/// pre-softmax score row [n_heads, n_keys].
pub fn tq_fused_attention_grouped(
    reg: &KernelRegistry,
    query: &CudaSlice<f32>,
    packed_indices: &CudaSlice<u8>,
    gnorms: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    scores_out: &mut CudaSlice<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    n_keys: usize,
    head_dim: usize,
    group_size: usize,
    bits: usize,
    scale: f32,
) -> Result<(), DriverError> {
    assert!(group_size > 0 && head_dim % group_size == 0,
        "tq_fused_attention_grouped: head_dim {} not divisible by group_size {}",
        head_dim, group_size);
    let f = reg.get_fn("fused_attention", "tq_fused_attention_grouped_f32")?;
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
    let gs = group_size as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(query).arg(packed_indices).arg(gnorms).arg(centroids).arg(scores_out)
            .arg(&nh).arg(&nkv).arg(&nk).arg(&hd).arg(&b).arg(&gs).arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Block size for fused decode attention kernels.
/// Adaptive to head_dim: round up to multiple of 32, cap at 128 (4 warps).
/// 128 saturates an SM's V-accumulate bandwidth at head_dim=128 (one dim/thread).
/// Smaller head_dim shrinks the block to avoid dead lanes in the dot-product stride.
fn decode_attn_block_dim(head_dim: usize) -> u32 {
    let hd = head_dim as u32;
    let bd = ((hd.min(128) + 31) / 32) * 32;
    bd.max(32)
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
    let block_dim = decode_attn_block_dim(head_dim);
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
    window_size: i32,  // sliding window: 0 = global, >0 = attend last N tokens
) -> Result<(), DriverError> {
    let f = reg.get_fn("fused_attention", "gqa_decode_attention_graph_f32")?;
    let block_dim = decode_attn_block_dim(head_dim);
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
            .arg(&nh).arg(&nkv).arg(&ms).arg(&hd).arg(&scale).arg(&extra).arg(&window_size)
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
    let block_dim = decode_attn_block_dim(head_dim);
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

/// Launch rope_halved_qk_f32: fused Q+K halved RoPE (one kernel replaces two).
/// `q` and `k` each in-place; cos/sin tables shared.
pub fn rope_halved_qk_with_gpu_pos(
    reg: &KernelRegistry,
    q: &mut CudaSlice<f32>,
    k: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_tokens: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos_offset: usize,
    pos_offset_gpu: Option<&CudaSlice<i32>>,
) -> Result<(), DriverError> {
    let f = reg.get_fn("rope", "rope_halved_qk_f32")?;
    let half = rope_dim / 2;
    let threads = ((half.max(32) + 31) / 32 * 32).min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, (n_q_heads + n_kv_heads) as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nt = n_tokens as i32;
    let nq = n_q_heads as i32;
    let nkv = n_kv_heads as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let po = pos_offset as i32;
    let null_positions: u64 = 0;
    let null_gpu_pos: u64 = 0;
    unsafe {
        if let Some(gpu_ptr) = pos_offset_gpu {
            reg.stream.launch_builder(&f)
                .arg(q).arg(k).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nq).arg(&nkv).arg(&hd).arg(&rd).arg(&po)
                .arg(gpu_ptr)
                .launch(cfg)?;
        } else {
            reg.stream.launch_builder(&f)
                .arg(q).arg(k).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nq).arg(&nkv).arg(&hd).arg(&rd).arg(&po)
                .arg(&null_gpu_pos)
                .launch(cfg)?;
        }
    }
    Ok(())
}

/// Launch rope_interleaved_qk_f32: fused Q+K interleaved RoPE (Llama style).
pub fn rope_interleaved_qk_with_gpu_pos(
    reg: &KernelRegistry,
    q: &mut CudaSlice<f32>,
    k: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_tokens: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos_offset: usize,
    pos_offset_gpu: Option<&CudaSlice<i32>>,
) -> Result<(), DriverError> {
    let f = reg.get_fn("rope", "rope_interleaved_qk_f32")?;
    let half = rope_dim / 2;
    let threads = ((half.max(32) + 31) / 32 * 32).min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, (n_q_heads + n_kv_heads) as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nt = n_tokens as i32;
    let nq = n_q_heads as i32;
    let nkv = n_kv_heads as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let po = pos_offset as i32;
    let null_positions: u64 = 0;
    let null_gpu_pos: u64 = 0;
    unsafe {
        if let Some(gpu_ptr) = pos_offset_gpu {
            reg.stream.launch_builder(&f)
                .arg(q).arg(k).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nq).arg(&nkv).arg(&hd).arg(&rd).arg(&po)
                .arg(gpu_ptr)
                .launch(cfg)?;
        } else {
            reg.stream.launch_builder(&f)
                .arg(q).arg(k).arg(cos).arg(sin).arg(&null_positions)
                .arg(&nt).arg(&nq).arg(&nkv).arg(&hd).arg(&rd).arg(&po)
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

/// Strided in-place RoPE for the GPU TQ decompress cache.
/// Buffer layout: [n_kv_head, max_seq, head_dim] (head-outer, token-inner).
/// Applies RoPE to the sub-range [start_token .. start_token + n_tokens) for
/// every kv head in a single launch. Used by the GPU Pre-RoPE attention path
/// after decompressing freshly added compressed keys.
pub fn rope_halved_strided(
    reg: &KernelRegistry,
    x: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_kv_head: usize,
    max_seq: usize,
    head_dim: usize,
    rope_dim: usize,
    start_token: usize,
    n_tokens: usize,
    pos_offset: usize,
) -> Result<(), DriverError> {
    if n_tokens == 0 { return Ok(()); }
    let f = reg.get_fn("rope", "rope_halved_strided_f32")?;
    let half = rope_dim / 2;
    let threads = ((half.max(32) + 31) / 32 * 32).min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_head as u32, n_tokens as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nkv = n_kv_head as i32;
    let ms = max_seq as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let st = start_token as i32;
    let nt = n_tokens as i32;
    let po = pos_offset as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(x).arg(cos).arg(sin)
            .arg(&nkv).arg(&ms).arg(&hd).arg(&rd).arg(&st).arg(&nt).arg(&po)
            .launch(cfg)?;
    }
    Ok(())
}

/// Strided in-place interleaved RoPE for the GPU TQ decompress cache.
/// Same shape contract as rope_halved_strided but uses the Llama-style
/// (2i, 2i+1) pair convention.
pub fn rope_interleaved_strided(
    reg: &KernelRegistry,
    x: &mut CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    n_kv_head: usize,
    max_seq: usize,
    head_dim: usize,
    rope_dim: usize,
    start_token: usize,
    n_tokens: usize,
    pos_offset: usize,
) -> Result<(), DriverError> {
    if n_tokens == 0 { return Ok(()); }
    let f = reg.get_fn("rope", "rope_interleaved_strided_f32")?;
    let half = rope_dim / 2;
    let threads = ((half.max(32) + 31) / 32 * 32).min(256) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_head as u32, n_tokens as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let nkv = n_kv_head as i32;
    let ms = max_seq as i32;
    let hd = head_dim as i32;
    let rd = rope_dim as i32;
    let st = start_token as i32;
    let nt = n_tokens as i32;
    let po = pos_offset as i32;
    unsafe {
        reg.stream.launch_builder(&f)
            .arg(x).arg(cos).arg(sin)
            .arg(&nkv).arg(&ms).arg(&hd).arg(&rd).arg(&st).arg(&nt).arg(&po)
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
    // QKV kernel dispatch. cpasync W-prefetch variant exists (faster per
    // kernel call, ~17% on norm+qkv) but end-to-end parity/regression on
    // Qwen2 7B (2026-04-14 bench: TQ -3% vs baseline). Default stays on the
    // proven baseline kernel; set TQ_QKV=cpasync to try on other workloads.
    static VARIANT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let kernel_name = *VARIANT.get_or_init(|| {
        match std::env::var("TQ_QKV").ok().as_deref() {
            Some("cpasync") => "fused_norm_q4km_qkv_bias_cpasync_f32",
            _               => "fused_norm_q4km_qkv_bias_f32",
        }
    });
    let f = reg.get_fn("fused_layer", kernel_name)?;
    let total_rows = (q_out + k_out + v_out) as u32;
    let block = 256u32;
    // cpasync needs 288 extra bytes for s_wbuf[2][36].
    let extra_shmem = if *kernel_name == *"fused_norm_q4km_qkv_bias_cpasync_f32" { 288 } else { 0 };
    let shmem = (hidden_dim as u32) * 4 + extra_shmem;
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
    // Gateup kernel dispatch. Seven variants live in fused_layer.cu:
    //   default (mrow8):   8 rows/block cpasync variant — best ROI RTX 3080
    //   mrow16:            16 rows/block (register spill on SM86 — opt-in)
    //   mrow4:             4 rows/block cpasync variant  (TQ_GATEUP=mrow4)
    //   mrow2:             2 rows/block cpasync variant  (TQ_GATEUP=mrow2)
    //   cpasync:           single-row cp.async pipeline  (TQ_GATEUP=cpasync)
    //   baseline:          original no-pipeline version  (TQ_GATEUP=baseline)
    //   lut:               warp-shuffle dequant LUT       (TQ_GATEUP=lut)
    //   dp4a:              INT8 dp4a matvec + inline q8_1 quantize (TQ_GATEUP=dp4a)
    static GATEUP_VARIANT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let kernel_name = *GATEUP_VARIANT.get_or_init(|| {
        match std::env::var("TQ_GATEUP").ok().as_deref() {
            Some("baseline") => "fused_addnorm_q4km_gateup_silu_f32",
            Some("lut")      => "fused_addnorm_q4km_gateup_silu_lut_f32",
            Some("cpasync")  => "fused_addnorm_q4km_gateup_silu_cpasync_f32",
            Some("mrow2")    => "fused_addnorm_q4km_gateup_silu_mrow2_f32",
            Some("mrow4")    => "fused_addnorm_q4km_gateup_silu_mrow4_f32",
            Some("mrow16")   => "fused_addnorm_q4km_gateup_silu_mrow16_f32",
            Some("dp4a")     => "fused_addnorm_q4km_gateup_silu_dp4a_f32",
            _                => "fused_addnorm_q4km_gateup_silu_mrow8_f32",
        }
    });
    let f = reg.get_fn("fused_layer", kernel_name)?;
    let block = 256u32;
    // cpasync: 576 B; mrow2: 1152 B; mrow4: 2304 B; baseline/lut: 0.
    let extra_shmem: u32 = if *kernel_name == *"fused_addnorm_q4km_gateup_silu_cpasync_f32" {
        576
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow2_f32" {
        1152
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow4_f32" {
        2304
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow8_f32" {
        4608
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow16_f32" {
        9216
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_dp4a_f32" {
        // s_x_q8_1: (hidden_dim / 32) * 36 B. e.g. 3584 → 4032 B.
        ((hidden_dim as u32) / 32) * 36
    } else {
        0
    };
    let shmem = (hidden_dim as u32) * 4 + extra_shmem;
    let grid = if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow2_f32" {
        (intermediate_dim as u32 + 1) / 2
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow4_f32" {
        (intermediate_dim as u32 + 3) / 4
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow8_f32" {
        (intermediate_dim as u32 + 7) / 8
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_mrow16_f32" {
        (intermediate_dim as u32 + 15) / 16
    } else if *kernel_name == *"fused_addnorm_q4km_gateup_silu_dp4a_f32" {
        (intermediate_dim as u32 + 7) / 8
    } else {
        intermediate_dim as u32
    };
    let cfg = launch_with_shmem(grid, block, shmem);
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
    // Down kernel dispatch:
    //   default (cpasync): single-row cp.async cooperative (Plan #9)
    //   mrow2:             2 rows/block (Std +4% but TQ -3% on RTX 3080 — opt-in)
    //   dp4a:              MMVQ Step 4: inline q8_1 quantize + INT8 dp4a matvec (opt-in)
    //   baseline:          thread-per-superblock original
    static VARIANT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    let kernel_name = *VARIANT.get_or_init(|| {
        match std::env::var("TQ_DOWN").ok().as_deref() {
            Some("baseline") => "fused_q4km_down_residual_f32",
            Some("mrow2")    => "fused_q4km_down_residual_mrow2_cpasync_f32",
            Some("dp4a")     => "fused_q4km_down_residual_dp4a_f32",
            _                => "fused_q4km_down_residual_cpasync_f32",
        }
    });
    let f = reg.get_fn("fused_layer", kernel_name)?;
    let grid = if *kernel_name == *"fused_q4km_down_residual_mrow2_cpasync_f32" {
        (hidden_dim as u32 + 1) / 2
    } else if *kernel_name == *"fused_q4km_down_residual_dp4a_f32" {
        (hidden_dim as u32 + 7) / 8
    } else {
        hidden_dim as u32
    };
    let cfg = if *kernel_name == *"fused_q4km_down_residual_dp4a_f32" {
        // s_x_q8_1: (intermediate_dim / 32) × 36 B. Qwen2 7B: 18944/32 * 36 = 21312 B.
        let shmem = ((intermediate_dim as u32) / 32) * 36;
        launch_with_shmem(grid, 256, shmem)
    } else {
        launch_per_row(grid as usize, 256)
    };
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

// ─── GPU smoke tests ─────────────────────────────────────────
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// MMVQ Step 1 round-trip: f32 → q8_1 → f32, max abs error ≤ d/2 (≈ amax/254).
    /// Uses a deterministic input so the bound is tight and reproducible.
    #[test]
    fn quantize_f32_to_q8_1_roundtrip() {
        let ctx = match cudarc::driver::CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => { eprintln!("[skip] no CUDA device: {:?}", e); return; }
        };
        let stream = ctx.default_stream();
        let (sm_major, sm_minor) = {
            use cudarc::driver::sys;
            let mut device: sys::CUdevice = 0;
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            unsafe {
                sys::cuDeviceGet(&mut device, 0);
                sys::cuDeviceGetAttribute(&mut major,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
                sys::cuDeviceGetAttribute(&mut minor,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
            }
            (major as u32, minor as u32)
        };
        let reg = KernelRegistry::new(&ctx, &stream, sm_major, sm_minor)
            .expect("kernel registry init");

        // 4 blocks × 32 elements; mix of magnitudes so each block has a different scale.
        const N_BLOCKS: usize = 4;
        const N: usize = N_BLOCKS * QK8_1;
        let mut x_host = vec![0.0f32; N];
        for b in 0..N_BLOCKS {
            // Per-block amax escalates: 1.0, 2.5, 0.125, 10.0
            let amax = [1.0_f32, 2.5, 0.125, 10.0][b];
            for i in 0..QK8_1 {
                // sawtooth in [-amax, +amax]
                let t = (i as f32 / (QK8_1 - 1) as f32) * 2.0 - 1.0;
                x_host[b * QK8_1 + i] = t * amax;
            }
        }

        let x_dev = stream.memcpy_stod(&x_host).expect("upload x");
        let mut out_dev: cudarc::driver::CudaSlice<u8> =
            stream.alloc_zeros(N_BLOCKS * Q8_1_BLOCK_BYTES).expect("alloc out");

        quantize_f32_to_q8_1(&reg, &x_dev, &mut out_dev, N).expect("launch");
        stream.synchronize().expect("sync");

        let out_host = stream.memcpy_dtov(&out_dev).expect("download");

        // Decode each block and verify round-trip error ≤ d/2 (with small float slack).
        for b in 0..N_BLOCKS {
            let off = b * Q8_1_BLOCK_BYTES;
            let block = &out_host[off..off + Q8_1_BLOCK_BYTES];
            // half2 ds
            let d_bits = u16::from_le_bytes([block[0], block[1]]);
            let s_bits = u16::from_le_bytes([block[2], block[3]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            let _s = half::f16::from_bits(s_bits).to_f32();

            // qs[32]
            let mut max_err = 0.0f32;
            let mut sum_q: i32 = 0;
            for i in 0..QK8_1 {
                let q = block[4 + i] as i8;
                sum_q += q as i32;
                let dq = (q as f32) * d;
                let err = (dq - x_host[b * QK8_1 + i]).abs();
                if err > max_err { max_err = err; }
            }

            // Upper bound: |x - q*d| ≤ d/2 (rounding) + half-precision error on d.
            // For d up to ~10/127 ≈ 0.079, slack of 1e-3 is generous.
            let bound = d * 0.5 + 1e-3;
            assert!(max_err <= bound,
                "block {}: round-trip max_err {:.6} > bound {:.6} (d={:.6})",
                b, max_err, bound, d);

            // Also verify max_err < 1/127 of the per-block amax (spec from agent brief).
            let amax_block = [1.0_f32, 2.5, 0.125, 10.0][b];
            let per_elem_bound = amax_block / 127.0;
            assert!(max_err <= per_elem_bound + 1e-4,
                "block {}: max_err {:.6} > 1/127 * amax = {:.6}",
                b, max_err, per_elem_bound);

            // s should equal d * sum_q (within fp16 slack).
            let s_expected = d * (sum_q as f32);
            let s_stored = _s;
            assert!((s_stored - s_expected).abs() <= 1e-2 * (s_expected.abs() + 1.0),
                "block {}: s mismatch — stored {:.6}, expected {:.6}",
                b, s_stored, s_expected);
        }
    }

    /// MMVQ Step 2: q4km_matvec_dp4a_f32 (INT8 pipe) vs. q4km_matvec_mrow8_f32
    /// (FP32 pipe) on synthetic data. Both should agree within INT8 activation
    /// quantization tolerance (|diff|/|ref| ≤ 3%).
    ///
    /// We fabricate Q4_K super-blocks with known (d, dmin, scales, nibbles) and
    /// a deterministic FP32 activation, then run both kernels and compare.
    #[test]
    fn q4km_matvec_dp4a_vs_fp32_reference() {
        const OUT_FEATURES: usize = 16;
        const IN_FEATURES:  usize = 256;   // exactly 1 super-block per row
        const N_SB:         usize = IN_FEATURES / 256;
        const Q4K_BYTES:    usize = 144;
        const BYTES_PER_ROW: usize = N_SB * Q4K_BYTES;

        let ctx = match cudarc::driver::CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => { eprintln!("[skip] no CUDA device: {:?}", e); return; }
        };
        let stream = ctx.default_stream();
        let (sm_major, sm_minor) = {
            use cudarc::driver::sys;
            let mut device: sys::CUdevice = 0;
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            unsafe {
                sys::cuDeviceGet(&mut device, 0);
                sys::cuDeviceGetAttribute(&mut major,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
                sys::cuDeviceGetAttribute(&mut minor,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
            }
            (major as u32, minor as u32)
        };
        if sm_major < 6 || (sm_major == 6 && sm_minor < 1) {
            eprintln!("[skip] dp4a requires sm_61+, have sm_{}{}", sm_major, sm_minor);
            return;
        }
        let reg = KernelRegistry::new(&ctx, &stream, sm_major, sm_minor)
            .expect("kernel registry init");

        // ── Build synthetic Q4_K weights: [OUT_FEATURES × BYTES_PER_ROW] ──
        // Seeded LCG for reproducibility; each row uses its own d/dmin/scales
        // so we exercise the full scale/min unpack path.
        let mut w_bytes = vec![0u8; OUT_FEATURES * BYTES_PER_ROW];
        let mut state: u32 = 0xC0DE_FACE;
        let mut next = || -> u32 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };

        // Helper: write fp16 from f32.
        fn f32_to_f16_bits(x: f32) -> u16 { half::f16::from_f32(x).to_bits() }

        for row in 0..OUT_FEATURES {
            for sb in 0..N_SB {
                let off = row * BYTES_PER_ROW + sb * Q4K_BYTES;
                let block = &mut w_bytes[off..off + Q4K_BYTES];
                // d ∈ [0.002, 0.01], dmin ∈ [0.001, 0.004] — small scales keep
                // dequantized values bounded so FP32 math stays well-behaved.
                let d_f    = 0.002_f32 + ((next() & 0xFF) as f32 / 255.0) * 0.008;
                let dmin_f = 0.001_f32 + ((next() & 0xFF) as f32 / 255.0) * 0.003;
                let d_bits  = f32_to_f16_bits(d_f).to_le_bytes();
                let dm_bits = f32_to_f16_bits(dmin_f).to_le_bytes();
                block[0] = d_bits[0]; block[1] = d_bits[1];
                block[2] = dm_bits[0]; block[3] = dm_bits[1];
                // Scales: 8 sub-blocks. Use the j<4 packing for simplicity:
                //   scales[0..4] = sc[0..4] (6-bit, top 2 bits zero)
                //   scales[4..8] = m[0..4]
                //   scales[8..12]: sub-blocks 4..7 packed. We'll set sc4..7 and m4..7 all
                //   to small values via the secondary packing.
                // To make get_scale_min_k4 decode cleanly for j >= 4 we must respect its layout.
                // Easiest: write a compatible 12-byte scales array by encoding all 8 sub-block
                // pairs via the inverse of get_scale_min_k4.
                let mut sc = [0u8; 8];
                let mut mn = [0u8; 8];
                for i in 0..8 {
                    sc[i] = ((next() & 0x1F) as u8) + 1; // 1..32 (fits 6-bit)
                    mn[i] = ((next() & 0x1F) as u8) + 1;
                }
                // Encode (j<4) directly.
                for j in 0..4 {
                    block[4 + j]     = sc[j] & 0x3F;
                    block[4 + 4 + j] = mn[j] & 0x3F;
                }
                // Encode (j>=4): scales[j+4] low nibble = sc[j] low 4, top 2 bits of
                // scales[j-4] and scales[j] hold high 2 bits of sc[j+4] and mn[j+4].
                // Following inverse of get_scale_min_k4:
                //   scales[j+4]&0x0F = sc[j]&0x0F
                //   scales[j-4]>>6   = sc[j]>>4
                //   scales[j+4]>>4   = mn[j]&0x0F
                //   scales[j]>>6     = mn[j]>>4
                for j in 4..8 {
                    let s = sc[j];
                    let m = mn[j];
                    // scales[j+4] byte = (mn[j]&0x0F)<<4 | (sc[j]&0x0F)
                    block[4 + (j + 4)] = ((m & 0x0F) << 4) | (s & 0x0F);
                    // Need to set top 2 bits of scales[j-4] and scales[j].
                    let lo_idx = 4 + (j - 4);  // scales[j-4]
                    let hi_idx = 4 + j;        // scales[j]
                    block[lo_idx] = (block[lo_idx] & 0x3F) | ((s >> 4) << 6);
                    block[hi_idx] = (block[hi_idx] & 0x3F) | ((m >> 4) << 6);
                }
                // qs: 128 bytes of packed nibbles. Deterministic pattern.
                for l in 0..128 {
                    let lo = (next() & 0x0F) as u8;
                    let hi = (next() & 0x0F) as u8;
                    block[16 + l] = lo | (hi << 4);
                }
            }
        }

        // ── Build FP32 activation ──
        let mut x_f32 = vec![0f32; IN_FEATURES];
        for i in 0..IN_FEATURES {
            let t = (i as f32) * 0.0123;
            x_f32[i] = t.sin() * 1.5 + (t * 0.37).cos() * 0.5;
        }

        // ── Dequantize weights on CPU for FP32 reference path AND for ground-truth ──
        // Use crate::quant::dequantize_q4k to get the exact same weight values the
        // FP32 kernel would see after unpacking.
        let w_f32_all = crate::quant::dequantize_q4k(&w_bytes, OUT_FEATURES * IN_FEATURES);

        // CPU ground-truth: plain FP32 matmul.
        let mut ref_host = vec![0f32; OUT_FEATURES];
        for row in 0..OUT_FEATURES {
            let mut acc = 0.0f32;
            let row_w = &w_f32_all[row * IN_FEATURES..(row + 1) * IN_FEATURES];
            for k in 0..IN_FEATURES {
                acc += row_w[k] * x_f32[k];
            }
            ref_host[row] = acc;
        }

        // ── GPU: FP32 path (q4km_matvec_mrow8_f32 via q4km_matvec launcher) ──
        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let x_dev_f32 = stream.memcpy_stod(&x_f32).expect("upload x_f32");
        let mut out_fp32: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(OUT_FEATURES).expect("alloc out_fp32");
        // Force the default (mrow8) variant — no env manipulation needed, it's
        // the default selected when TQ_Q4KM is unset.
        super::q4km_matvec(&reg, &w_dev, &x_dev_f32, &mut out_fp32, OUT_FEATURES, IN_FEATURES)
            .expect("launch fp32");
        stream.synchronize().expect("sync fp32");
        let out_fp32_host = stream.memcpy_dtov(&out_fp32).expect("download fp32");

        // ── GPU: dp4a path (activation → q8_1 → dp4a matvec) ──
        let n_q8_1_blocks = IN_FEATURES / QK8_1;
        let mut x_q8_1: cudarc::driver::CudaSlice<u8> =
            stream.alloc_zeros(n_q8_1_blocks * Q8_1_BLOCK_BYTES).expect("alloc x_q8_1");
        super::quantize_f32_to_q8_1(&reg, &x_dev_f32, &mut x_q8_1, IN_FEATURES)
            .expect("launch q8_1 quantize");
        let mut out_dp4a: cudarc::driver::CudaSlice<f32> =
            stream.alloc_zeros(OUT_FEATURES).expect("alloc out_dp4a");
        super::q4km_matvec_dp4a(&reg, &w_dev, &x_q8_1, &mut out_dp4a, OUT_FEATURES, IN_FEATURES)
            .expect("launch dp4a");
        stream.synchronize().expect("sync dp4a");
        let out_dp4a_host = stream.memcpy_dtov(&out_dp4a).expect("download dp4a");

        // ── Compare ──
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut max_rel  = 0.0f32;
        for row in 0..OUT_FEATURES {
            let r  = ref_host[row];
            let f  = out_fp32_host[row];
            let dq = out_dp4a_host[row];

            // Sanity: FP32 GPU path must agree with CPU reference closely.
            let fp32_err = (f - r).abs();
            assert!(fp32_err <= 1e-3 * (r.abs() + 1.0),
                "row {}: FP32 GPU mismatches CPU — gpu {:.6}, cpu {:.6}",
                row, f, r);

            let diff = (dq - r).abs();
            let rel  = diff / (r.abs() + 1e-6);
            if diff > max_diff { max_diff = diff; }
            sum_diff += diff;
            if rel > max_rel { max_rel = rel; }
        }
        let mean_diff = sum_diff / (OUT_FEATURES as f32);

        eprintln!(
            "[dp4a-vs-fp32] max_abs={:.6}, mean_abs={:.6}, max_rel={:.4}",
            max_diff, mean_diff, max_rel
        );

        let verdict = if max_rel <= 0.03 { "PASS" } else { "FAIL" };
        eprintln!("[dp4a-vs-fp32] verdict: {} (tol 3%)", verdict);

        assert!(max_rel <= 0.03,
            "dp4a vs fp32 reference: max rel error {:.4} > 0.03 (max_abs={:.6})",
            max_rel, max_diff);
    }
}

// ─── GQA Shared-K decode attention (skeleton launcher) ───────────────────
// Owner: attention-researcher  Plan: docs/gqa-shared-k-design.md
//
// Feature-gated stub. The CUDA kernel itself
// (`gqa_decode_attention_shared_k_f32` in kernels/fused_attention.cu)
// always compiles into the PTX, but this Rust-side launcher is only built
// when the `gqa-shared-k` cargo feature is enabled. That keeps the default
// build surface unchanged while letting CI exercise the new path on
// demand.
//
// Grid shape: one block per KV group (down from one per Q head). See the
// design doc for the full algorithm + register-pressure analysis.
//
// NOT WIRED INTO RUNTIME DISPATCH. Caller code does not exist yet — this
// signature is the contract that future implementation work targets.
#[cfg(feature = "gqa-shared-k")]
pub fn gqa_decode_attention_shared_k(
    reg: &KernelRegistry,
    q: &CudaSlice<f32>,           // [n_heads, head_dim]
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
    let f = reg.get_fn("fused_attention", "gqa_decode_attention_shared_k_f32")?;
    // Per design doc §3: 128 threads/block keeps register pressure under
    // the spill threshold for n_qh up to 7 (Qwen2 7B). Larger groups will
    // need warp-chunking — revisit when we hit that case.
    let block_dim: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: (n_kv_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0, // static __shared__ in the kernel
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

