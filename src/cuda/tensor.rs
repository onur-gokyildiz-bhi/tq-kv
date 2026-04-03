//! Minimal tensor type — CPU Vec<f32> or CUDA device memory.
//!
//! Implements only the ~25 operations used by turbo_generic.rs.
//! No autograd, no dynamic dispatch overhead.

use super::{TqDevice, TqDType, Result, TqError, tq_bail};

// ── CUDA Graph Decode Buffer Pool ────────────────────────────────
// For CUDA Graph replay, ALL GPU memory must be allocated BEFORE capture.
// During capture, cuMemAllocAsync creates graph-scoped virtual memory that's
// only valid during graph execution — not persistent between launches.
//
// Solution: DecodeBufferPool with two phases:
//   1. RECORDING (warm-up pass): alloc normally + save each buffer
//   2. POOLED (capture + replay passes): return saved buffers, no new alloc
//
// Same pointers every pass → graph bakes persistent addresses.

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolMode {
    /// Normal mode: gpu_alloc_zeros allocates fresh buffers
    Off,
    /// Recording: alloc normally + push to pool (warm-up pass before capture)
    Recording,
    /// Pooled: return pool[cursor++] instead of allocating (capture + replay passes)
    Pooled,
}

#[cfg(feature = "cuda")]
std::thread_local! {
    static DECODE_POOL_MODE: std::cell::Cell<PoolMode> = const { std::cell::Cell::new(PoolMode::Off) };
    static DECODE_POOL_BUFFERS: std::cell::RefCell<Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>>
        = const { std::cell::RefCell::new(Vec::new()) };
    static DECODE_POOL_CURSOR: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Set pool mode.
#[cfg(feature = "cuda")]
pub fn decode_pool_set_mode(mode: PoolMode) {
    DECODE_POOL_MODE.with(|m| m.set(mode));
    if mode == PoolMode::Recording {
        DECODE_POOL_BUFFERS.with(|b| b.borrow_mut().clear());
    }
    DECODE_POOL_CURSOR.with(|c| c.set(0));
}

/// Reset pool cursor to 0 (call at start of each forward pass in Pooled mode).
#[cfg(feature = "cuda")]
pub fn decode_pool_reset_cursor() {
    DECODE_POOL_CURSOR.with(|c| c.set(0));
}

/// Get recorded pool buffers (for storing in model struct).
#[cfg(feature = "cuda")]
pub fn decode_pool_drain() -> Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>> {
    DECODE_POOL_BUFFERS.with(|b| {
        let mut v = b.borrow_mut();
        std::mem::take(&mut *v)
    })
}

/// Restore pool buffers (from model struct into thread-local).
#[cfg(feature = "cuda")]
pub fn decode_pool_restore(buffers: Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>) {
    DECODE_POOL_BUFFERS.with(|b| *b.borrow_mut() = buffers);
}

/// Allocate or reuse a GPU buffer depending on pool mode.
/// In Recording mode: alloc + save to pool.
/// In Pooled mode: return pool[cursor++] (same pointer every time).
/// In Off mode: normal alloc.
#[cfg(feature = "cuda")]
fn pool_alloc_f32(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    len: usize,
) -> std::result::Result<std::sync::Arc<cudarc::driver::CudaSlice<f32>>, cudarc::driver::result::DriverError> {
    let mode = DECODE_POOL_MODE.with(|m| m.get());
    match mode {
        PoolMode::Off => {
            let buf = gpu_alloc_zeros::<f32>(stream, len)?;
            Ok(std::sync::Arc::new(buf))
        }
        PoolMode::Recording => {
            let buf = gpu_alloc_zeros::<f32>(stream, len)?;
            let arc = std::sync::Arc::new(buf);
            DECODE_POOL_BUFFERS.with(|b| b.borrow_mut().push(arc.clone()));
            Ok(arc)
        }
        PoolMode::Pooled => {
            let idx = DECODE_POOL_CURSOR.with(|c| {
                let i = c.get();
                c.set(i + 1);
                i
            });
            let arc = DECODE_POOL_BUFFERS.with(|b| {
                let bufs = b.borrow();
                if idx < bufs.len() {
                    Some(bufs[idx].clone())
                } else {
                    None
                }
            });
            match arc {
                Some(a) => {
                    // Verify size matches (sanity check)
                    if a.len() != len {
                        eprintln!("[pool] WARNING: size mismatch at idx {}: pool={} requested={}", idx, a.len(), len);
                    }
                    Ok(a)
                }
                None => {
                    eprintln!("[pool] WARNING: pool exhausted at idx {} (pool has {} buffers), allocating fresh",
                        idx, DECODE_POOL_BUFFERS.with(|b| b.borrow().len()));
                    let buf = gpu_alloc_zeros::<f32>(stream, len)?;
                    Ok(std::sync::Arc::new(buf))
                }
            }
        }
    }
}

// Keep graph_retention_start/drain as no-ops for backward compat
#[cfg(feature = "cuda")]
pub fn graph_retention_start() {}
#[cfg(feature = "cuda")]
pub fn graph_retention_drain() -> Vec<std::sync::Arc<cudarc::driver::CudaSlice<f32>>> { Vec::new() }
#[cfg(feature = "cuda")]
fn graph_retention_keep(_data: &std::sync::Arc<cudarc::driver::CudaSlice<f32>>) {}

/// Public wrapper for gpu_alloc_zeros (used by GpuKvCache).
#[cfg(feature = "cuda")]
pub fn gpu_alloc_zeros_pub(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    len: usize,
) -> std::result::Result<cudarc::driver::CudaSlice<f32>, cudarc::driver::result::DriverError> {
    gpu_alloc_zeros::<f32>(stream, len)
}

/// GPU alloc that clears cudarc's stale error_state first.
/// During CUDA Graph capture, CudaSlice::drop → cuMemFreeAsync on pre-capture memory
/// sets error_state to INVALID_VALUE. This stale error blocks subsequent alloc_zeros
/// via bind_to_thread → check_err. Clearing before alloc prevents the cascade.
///
#[cfg(feature = "cuda")]
fn gpu_alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    len: usize,
) -> std::result::Result<cudarc::driver::CudaSlice<T>, cudarc::driver::result::DriverError> {
    let _ = stream.context().check_err();
    stream.alloc_zeros(len)
}

/// Storage backend for tensor data.
///
/// CUDA variant uses `Arc<CudaSlice<f32>>` for zero-copy clone semantics:
/// - `TqTensor::clone()` = Arc ref-count increment (NOT GPU memory copy)
/// - `unsqueeze`, `reshape`, `contiguous` = zero-copy (only shape metadata changes)
/// - `cuda_data_mut()` = copy-on-write via `Arc::make_mut()` (only copies if shared)
#[derive(Debug, Clone)]
pub enum TqStorage {
    /// CPU: contiguous f32 data in row-major order.
    Cpu(Vec<f32>),
    /// CUDA: device memory on GPU (reference-counted, zero-copy clone).
    #[cfg(feature = "cuda")]
    Cuda {
        data: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    },
}

/// A minimal tensor: shape + dtype + storage.
///
/// Always row-major (C-contiguous). Shape is e.g. [batch, heads, seq_len, head_dim].
#[derive(Debug, Clone)]
pub struct TqTensor {
    storage: TqStorage,
    shape: Vec<usize>,
    dtype: TqDType,
}

impl TqTensor {
    // ─── Construction ──────────────────────────────────────────

    /// Create from a Vec<f32> and shape.
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<usize>>, _device: &TqDevice) -> Result<Self> {
        let shape = shape.into();
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            tq_bail!("from_vec: data len {} != shape product {}", data.len(), expected);
        }
        Ok(Self {
            storage: TqStorage::Cpu(data),
            shape,
            dtype: TqDType::F32,
        })
    }

    /// Create from a slice.
    pub fn from_slice(data: &[f32], shape: impl Into<Vec<usize>>, device: &TqDevice) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape, device)
    }

    /// Create a zeros tensor.
    pub fn zeros(shape: impl Into<Vec<usize>>, device: &TqDevice) -> Result<Self> {
        let shape = shape.into();
        let n: usize = shape.iter().product();
        Self::from_vec(vec![0.0; n], shape, device)
    }

    /// Scalar tensor (shape []).
    pub fn new(val: f32, _device: &TqDevice) -> Result<Self> {
        Ok(Self {
            storage: TqStorage::Cpu(vec![val]),
            shape: vec![],
            dtype: TqDType::F32,
        })
    }

    // ─── Accessors ─────────────────────────────────────────────

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn dtype(&self) -> TqDType { self.dtype }
    pub fn elem_count(&self) -> usize { self.shape.iter().product() }

    pub fn dim(&self, d: usize) -> Result<usize> {
        let d = self.resolve_dim(d)?;
        Ok(self.shape[d])
    }

    pub fn dims2(&self) -> Result<(usize, usize)> {
        if self.shape.len() != 2 { tq_bail!("dims2: rank {} != 2", self.rank()); }
        Ok((self.shape[0], self.shape[1]))
    }

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        if self.shape.len() != 3 { tq_bail!("dims3: rank {} != 3", self.rank()); }
        Ok((self.shape[0], self.shape[1], self.shape[2]))
    }

    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        if self.shape.len() != 4 { tq_bail!("dims4: rank {} != 4", self.rank()); }
        Ok((self.shape[0], self.shape[1], self.shape[2], self.shape[3]))
    }

    pub fn device(&self) -> &TqDevice {
        match &self.storage {
            TqStorage::Cpu(_) => &TqDevice::Cpu,
            #[cfg(feature = "cuda")]
            TqStorage::Cuda { .. } => &TqDevice::Cpu, // TODO: return actual device
        }
    }

    /// Get underlying f32 data.
    pub fn to_vec1(&self) -> Result<Vec<f32>> {
        match &self.storage {
            TqStorage::Cpu(data) => Ok(data.clone()),
            #[cfg(feature = "cuda")]
            TqStorage::Cuda { data, stream } => {
                // Sync + clear stale error_state before dtoh (graph capture/replay may leave benign errors)
                if let Err(e) = stream.synchronize() {
                    eprintln!("[dtoh] sync failed: {:?}", e);
                }
                if let Err(e) = stream.context().check_err() {
                    eprintln!("[dtoh] stale error cleared: {:?}", e);
                }
                // First try normal clone_dtoh. If it fails (e.g., stale error from graph capture),
                // fall back to raw CUDA memcpy which bypasses cudarc's error_state mechanism.
                match stream.clone_dtoh(data.as_ref()) {
                    Ok(v) => Ok(v),
                    Err(_) => {
                        let _ = stream.context().check_err(); // clear error_state
                        let _ = stream.synchronize();
                        // Raw memcpy: bypass cudarc's bind_to_thread/check_err
                        use cudarc::driver::{DevicePtr, DeviceSlice};
                        let n = data.len();
                        let mut dst = vec![0.0f32; n];
                        let (src_ptr, _guard) = data.as_ref().device_ptr(stream.as_ref());
                        let res = unsafe {
                            cudarc::driver::sys::cuMemcpyDtoH_v2(
                                dst.as_mut_ptr() as *mut _,
                                src_ptr,
                                n * std::mem::size_of::<f32>(),
                            )
                        };
                        if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                            return Err(TqError::Msg(format!("dtoh raw fallback (n={}): {:?}", n, res)));
                        }
                        Ok(dst)
                    }
                }
            }
        }
    }

    /// Borrow underlying f32 slice.
    /// For CUDA tensors: downloads to CPU transparently (allocates).
    pub fn as_slice(&self) -> &[f32] {
        match &self.storage {
            TqStorage::Cpu(data) => data,
            #[cfg(feature = "cuda")]
            TqStorage::Cuda { data, stream } => {
                // Transitional auto-download for ops without GPU dispatch yet.
                let cpu_data = stream.clone_dtoh(data.as_ref()).expect("as_slice GPU download failed");
                let leaked: &'static [f32] = Box::leak(cpu_data.into_boxed_slice());
                leaked
            }
        }
    }

    // ─── Shape operations ──────────────────────────────────────

    /// Reshape to new shape. Total elements must match.
    pub fn reshape(&self, new_shape: impl Into<Vec<usize>>) -> Result<Self> {
        let new_shape = new_shape.into();
        let old_n: usize = self.shape.iter().product();
        let new_n: usize = new_shape.iter().product();
        if old_n != new_n {
            tq_bail!("reshape: {} elements -> {} elements", old_n, new_n);
        }
        Ok(Self { storage: self.storage.clone(), shape: new_shape, dtype: self.dtype })
    }

    /// Flatten all dimensions into a single dimension.
    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(vec![self.elem_count()])
    }

    /// Add a dimension of size 1 at the given position.
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_shape = self.shape.clone();
        if dim > new_shape.len() {
            tq_bail!("unsqueeze: dim {} > rank {}", dim, self.rank());
        }
        new_shape.insert(dim, 1);
        Ok(Self { storage: self.storage.clone(), shape: new_shape, dtype: self.dtype })
    }

    /// Remove a dimension of size 1 at the given position.
    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let dim = self.resolve_dim(dim)?;
        if self.shape[dim] != 1 {
            tq_bail!("squeeze: dim {} has size {} != 1", dim, self.shape[dim]);
        }
        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);
        Ok(Self { storage: self.storage.clone(), shape: new_shape, dtype: self.dtype })
    }

    /// Transpose the last two dimensions.
    pub fn t(&self) -> Result<Self> {
        if self.rank() < 2 {
            tq_bail!("t: need rank >= 2, got {}", self.rank());
        }
        let r = self.rank();
        self.transpose(r - 2, r - 1)
    }

    /// Transpose two dimensions.
    pub fn transpose(&self, d1: usize, d2: usize) -> Result<Self> {
        let d1 = self.resolve_dim(d1)?;
        let d2 = self.resolve_dim(d2)?;
        if d1 == d2 { return Ok(self.clone()); }

        let rank = self.rank();
        let n = self.elem_count();

        // Source strides
        let mut old_strides = vec![1i32; rank];
        for i in (0..rank - 1).rev() {
            old_strides[i] = old_strides[i + 1] * self.shape[i + 1] as i32;
        }

        // New shape and strides (output layout)
        let mut new_shape = self.shape.clone();
        new_shape.swap(d1, d2);
        let mut new_strides = vec![1i32; rank];
        for i in (0..rank - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1] as i32;
        }

        // Source strides remapped: for the output's coordinate system,
        // source stride at dim d1 maps to original stride at d2 and vice versa
        let mut src_strides = old_strides.clone();
        src_strides.swap(d1, d2);

        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let _ = stream.context().check_err();
                let mut out_gpu = gpu_alloc_zeros::<f32>(stream,n)
                    .map_err(|e| TqError::Msg(format!("transpose alloc: {}", e)))?;
                let out_shape_gpu = stream.clone_htod(&new_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())
                    .map_err(|e| TqError::Msg(format!("transpose: {}", e)))?;
                let out_strides_gpu = stream.clone_htod(&new_strides)
                    .map_err(|e| TqError::Msg(format!("transpose: {}", e)))?;
                let src_strides_gpu = stream.clone_htod(&src_strides)
                    .map_err(|e| TqError::Msg(format!("transpose: {}", e)))?;

                super::kernels::strided_copy(
                    reg, src.as_ref(), &mut out_gpu, n, rank,
                    &out_shape_gpu, &out_strides_gpu, &src_strides_gpu, 0,
                ).map_err(|e| TqError::Msg(format!("transpose kernel: {}", e)))?;

                // Leak metadata to prevent cuMemFreeAsync during graph capture
                std::mem::forget(out_shape_gpu);
                std::mem::forget(out_strides_gpu);
                std::mem::forget(src_strides_gpu);

                return Ok(Self::from_cuda(out_gpu, new_shape, stream.clone()));
            }
        }

        // CPU path
        let data = self.as_slice();
        let mut result = vec![0.0f32; n];
        let old_strides_u: Vec<usize> = old_strides.iter().map(|&s| s as usize).collect();
        let new_strides_u: Vec<usize> = new_strides.iter().map(|&s| s as usize).collect();

        for flat_idx in 0..n {
            let mut remaining = flat_idx;
            let mut old_coords = vec![0usize; rank];
            for d in 0..rank {
                old_coords[d] = remaining / old_strides_u[d];
                remaining %= old_strides_u[d];
            }
            old_coords.swap(d1, d2);
            let new_flat: usize = old_coords.iter().zip(new_strides_u.iter())
                .map(|(&c, &s)| c * s).sum();
            result[new_flat] = data[flat_idx];
        }

        Self::from_vec(result, new_shape, self.device())
    }

    /// Narrow (slice) along a dimension.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dim = self.resolve_dim(dim)?;
        if start + len > self.shape[dim] {
            tq_bail!("narrow: {}..{} out of bounds for dim {} size {}",
                start, start + len, dim, self.shape[dim]);
        }

        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            let rank = self.rank();
            let mut new_shape = self.shape.clone();
            new_shape[dim] = len;
            let new_n: usize = new_shape.iter().product();
            let outer: usize = self.shape[..dim].iter().product();
            let inner: usize = self.shape[dim + 1..].iter().product();

            // Build source strides and output strides for strided_copy kernel
            let mut src_strides_v = vec![0i32; rank];
            let mut out_strides_v = vec![1i32; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                src_strides_v[i] = src_strides_v[i + 1] * self.shape[i + 1] as i32;
                out_strides_v[i] = out_strides_v[i + 1] * new_shape[i + 1] as i32;
            }
            // Source strides are same as output strides (same layout) but with original dim sizes
            for i in (0..rank.saturating_sub(1)).rev() {
                src_strides_v[i] = if i + 1 < rank { src_strides_v[i + 1] * self.shape[i + 1] as i32 } else { 1 };
            }
            let src_offset = (start * inner) as i32;

            // Simple contiguous copy approach: for contiguous narrow, use sliced memcpy
            let _ = stream.context().check_err();
            let mut out_gpu = gpu_alloc_zeros::<f32>(stream,new_n)
                .map_err(|e| TqError::Msg(format!("narrow alloc: {}", e)))?;

            if let Some(reg) = super::kernels::global_registry() {
                let out_shape_gpu = stream.clone_htod(&new_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())
                    .map_err(|e| TqError::Msg(format!("narrow shape upload: {}", e)))?;
                let out_strides_gpu = stream.clone_htod(&out_strides_v)
                    .map_err(|e| TqError::Msg(format!("narrow strides upload: {}", e)))?;

                // For narrow: source strides match original tensor layout
                let mut real_src_strides = vec![1i32; rank];
                for i in (0..rank.saturating_sub(1)).rev() {
                    real_src_strides[i] = real_src_strides[i + 1] * self.shape[i + 1] as i32;
                }
                // Adjust: the dim we're narrowing has offset applied via src_offset
                // Actually for narrow, src and dst have same strides except at narrow dim
                // Simplest: use the general strided_copy with correct offset
                let src_strides_gpu = stream.clone_htod(&real_src_strides)
                    .map_err(|e| TqError::Msg(format!("narrow src strides: {}", e)))?;

                // Source offset = start position in the narrowed dimension
                let base_offset = if dim == 0 {
                    (start * inner) as i32
                } else {
                    // General: start * stride_of_dim_in_source
                    (start as i32) * real_src_strides[dim]
                };

                super::kernels::strided_copy(
                    reg, src.as_ref(), &mut out_gpu, new_n, rank,
                    &out_shape_gpu, &out_strides_gpu, &src_strides_gpu, base_offset,
                ).map_err(|e| TqError::Msg(format!("narrow kernel: {}", e)))?;

                std::mem::forget(out_shape_gpu);
                std::mem::forget(out_strides_gpu);
                std::mem::forget(src_strides_gpu);

                return Ok(Self::from_cuda(out_gpu, new_shape, stream.clone()));
            }
            // Fallback: download, narrow on CPU, re-upload
        }

        // CPU path
        let data = self.as_slice();
        let rank = self.rank();
        let mut new_shape = self.shape.clone();
        new_shape[dim] = len;
        let outer: usize = self.shape[..dim].iter().product();
        let inner: usize = self.shape[dim + 1..].iter().product();
        let mut result = Vec::with_capacity(new_shape.iter().product());

        for o in 0..outer {
            for s in start..start + len {
                let src_offset = o * (self.shape[dim] * inner) + s * inner;
                result.extend_from_slice(&data[src_offset..src_offset + inner]);
            }
        }

        Self::from_vec(result, new_shape, self.device())
    }

    /// Concatenate tensors along a dimension.
    pub fn cat(tensors: &[&TqTensor], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            tq_bail!("cat: empty tensor list");
        }
        let first = tensors[0];
        let dim = first.resolve_dim(dim)?;

        // Validate shapes match on all non-cat dimensions
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.rank() != first.rank() {
                tq_bail!("cat: rank mismatch at tensor {}", i);
            }
            for d in 0..first.rank() {
                if d != dim && t.shape[d] != first.shape[d] {
                    tq_bail!("cat: shape mismatch at dim {} (tensor {})", d, i);
                }
            }
        }

        let mut new_shape = first.shape.clone();
        new_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();
        let inner: usize = first.shape[dim + 1..].iter().product();
        let outer: usize = first.shape[..dim].iter().product();

        let total_dim: usize = tensors.iter().map(|t| t.shape[dim]).sum();
        let new_n = outer * total_dim * inner;

        // GPU path: all tensors on GPU → GPU-to-GPU concat via copy_with_offsets.
        // Graph-capture safe: no clone_htod, no temp alloc — just kernel launches.
        #[cfg(feature = "cuda")]
        {
            let all_cuda = tensors.iter().all(|t| t.is_cuda());
            if all_cuda {
                if let TqStorage::Cuda { stream, .. } = &first.storage {
                    if let Some(reg) = super::kernels::global_registry() {
                        let out_gpu = gpu_alloc_zeros::<f32>(stream,new_n)
                            .map_err(|e| TqError::Msg(format!("cat alloc: {}", e)))?;

                        let mut dst_offset: usize = 0;
                        for o in 0..outer {
                            for t in tensors {
                                let chunk = t.shape[dim] * inner;
                                let src_off = o * chunk;
                                super::kernels::copy_with_offsets(
                                    reg, t.cuda_data(), &out_gpu, chunk, src_off, dst_offset,
                                ).map_err(|e| TqError::Msg(format!("cat copy: {}", e)))?;
                                dst_offset += chunk;
                            }
                        }

                        return Ok(Self::from_cuda(out_gpu, new_shape, stream.clone()));
                    }
                }
            }
        }

        // CPU path
        let mut result = Vec::with_capacity(new_n);
        let data_vecs: Vec<Vec<f32>> = tensors.iter()
            .map(|t| t.to_vec1())
            .collect::<Result<Vec<_>>>()?;
        for o in 0..outer {
            for (i, t) in tensors.iter().enumerate() {
                let t_dim = t.shape[dim];
                let src_offset = o * t_dim * inner;
                result.extend_from_slice(&data_vecs[i][src_offset..src_offset + t_dim * inner]);
            }
        }

        Self::from_vec(result, new_shape, first.device())
    }

    /// Ensure tensor is contiguous (no-op for our layout, always contiguous).
    pub fn contiguous(&self) -> Result<Self> {
        Ok(self.clone())
    }

    /// Convert dtype (currently only F32 supported, so this is a no-op).
    pub fn to_dtype(&self, dtype: TqDType) -> Result<Self> {
        if dtype == self.dtype { return Ok(self.clone()); }
        // TODO: actual f16/bf16 conversion when needed
        Ok(Self { storage: self.storage.clone(), shape: self.shape.clone(), dtype })
    }

    // ─── Math operations ───────────────────────────────────────

    /// Matrix multiply: self @ other.
    /// Supports batched matmul for rank >= 2.
    /// GPU: uses cuBLAS SGEMM when both tensors are on CUDA.
    pub fn matmul(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.matmul_cublas(other);
        }
        super::ops::TqOps::matmul(self, other)
    }

    /// Compute broadcast output shape + strides for two tensors.
    /// Returns (out_shape, a_strides, b_strides) with 0-stride for broadcast dims.
    #[cfg(feature = "cuda")]
    fn broadcast_strides(a_shape: &[usize], b_shape: &[usize]) -> Result<(Vec<usize>, Vec<i32>, Vec<i32>)> {
        let rank = a_shape.len().max(b_shape.len());
        let mut out = vec![0usize; rank];
        let mut a_str = vec![0i32; rank];
        let mut b_str = vec![0i32; rank];

        // Pad shapes with 1s on the left
        let a_pad: Vec<usize> = (0..rank).map(|i| {
            if i < rank - a_shape.len() { 1 } else { a_shape[i - (rank - a_shape.len())] }
        }).collect();
        let b_pad: Vec<usize> = (0..rank).map(|i| {
            if i < rank - b_shape.len() { 1 } else { b_shape[i - (rank - b_shape.len())] }
        }).collect();

        for i in 0..rank {
            out[i] = a_pad[i].max(b_pad[i]);
            if a_pad[i] != 1 && b_pad[i] != 1 && a_pad[i] != b_pad[i] {
                tq_bail!("broadcast: shapes incompatible at dim {}", i);
            }
        }

        // Compute strides (row-major, 0 for broadcast dims)
        let mut a_real_stride = vec![1i32; rank];
        let mut b_real_stride = vec![1i32; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            a_real_stride[i] = a_real_stride[i + 1] * a_pad[i + 1] as i32;
            b_real_stride[i] = b_real_stride[i + 1] * b_pad[i + 1] as i32;
        }
        for i in 0..rank {
            a_str[i] = if a_pad[i] == 1 { 0 } else { a_real_stride[i] };
            b_str[i] = if b_pad[i] == 1 { 0 } else { b_real_stride[i] };
        }

        Ok((out, a_str, b_str))
    }

    /// GPU broadcast binary op helper.
    #[cfg(feature = "cuda")]
    fn gpu_broadcast_binop(
        &self, other: &TqTensor,
        kernel_fn: fn(&super::kernels::KernelRegistry, &cudarc::driver::CudaSlice<f32>, &cudarc::driver::CudaSlice<f32>,
            &mut cudarc::driver::CudaSlice<f32>, usize, usize,
            &cudarc::driver::CudaSlice<i32>, &cudarc::driver::CudaSlice<i32>,
            &cudarc::driver::CudaSlice<i32>, &cudarc::driver::CudaSlice<i32>) -> std::result::Result<(), cudarc::driver::result::DriverError>,
    ) -> Result<Self> {
        let (TqStorage::Cuda { data: a, stream }, TqStorage::Cuda { data: b, .. }) = (&self.storage, &other.storage) else {
            tq_bail!("gpu_broadcast_binop: both must be CUDA");
        };
        let reg = super::kernels::global_registry().ok_or_else(|| TqError::Msg("no registry".into()))?;
        let (out_shape, a_str, b_str) = Self::broadcast_strides(&self.shape, &other.shape)?;
        let rank = out_shape.len();
        let n: usize = out_shape.iter().product();

        let mut out_strides = vec![1i32; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1] as i32;
        }

        let os_gpu = stream.clone_htod(&out_shape.iter().map(|&x| x as i32).collect::<Vec<_>>()).map_err(|e| TqError::Msg(format!("{}", e)))?;
        let ost_gpu = stream.clone_htod(&out_strides).map_err(|e| TqError::Msg(format!("{}", e)))?;
        let ast_gpu = stream.clone_htod(&a_str).map_err(|e| TqError::Msg(format!("{}", e)))?;
        let bst_gpu = stream.clone_htod(&b_str).map_err(|e| TqError::Msg(format!("{}", e)))?;

        let _ = stream.context().check_err(); // clear stale errors from CudaSlice drops
        let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
        kernel_fn(reg, a.as_ref(), b.as_ref(), &mut out, n, rank, &os_gpu, &ost_gpu, &ast_gpu, &bst_gpu)
            .map_err(|e| TqError::Msg(format!("broadcast kernel: {}", e)))?;

        // Leak metadata GPU buffers to prevent cuMemFreeAsync during graph capture.
        // cuMemFreeAsync of graph-internal allocations can set error_state, which causes
        // subsequent bind_to_thread → check_err to surface stale INVALID_VALUE errors.
        // Memory is small (16-32 bytes per call) and reclaimed on context destroy.
        std::mem::forget(os_gpu);
        std::mem::forget(ost_gpu);
        std::mem::forget(ast_gpu);
        std::mem::forget(bst_gpu);

        Ok(Self::from_cuda(out, out_shape, stream.clone()))
    }

    /// Element-wise multiply with broadcasting.
    pub fn broadcast_mul(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.gpu_broadcast_binop(other, super::kernels::gpu_broadcast_mul);
        }
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a * b)
    }

    /// Element-wise add with broadcasting.
    pub fn broadcast_add(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.gpu_broadcast_binop(other, super::kernels::gpu_broadcast_add);
        }
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a + b)
    }

    /// Element-wise sub with broadcasting.
    pub fn broadcast_sub(&self, other: &TqTensor) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() && other.is_cuda() {
            return self.gpu_broadcast_binop(other, super::kernels::gpu_broadcast_sub);
        }
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a - b)
    }

    /// Element-wise div with broadcasting.
    pub fn broadcast_div(&self, other: &TqTensor) -> Result<Self> {
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a / b)
    }

    /// Scalar multiply.
    pub fn mul_scalar(&self, s: f32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let n = self.elem_count();
                let s_gpu = stream.clone_htod(&[s]).map_err(|e| TqError::Msg(format!("{}", e)))?;
                // Use broadcast_mul with scalar (broadcast s to all elements)
                // Simpler: use scalar_mul kernel from elementwise.cu
                let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
                super::kernels::scalar_mul(reg, src, &mut out, s, n)
                    .map_err(|e| TqError::Msg(format!("scalar_mul: {}", e)))?;
                return Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()));
            }
        }
        let data: Vec<f32> = self.as_slice().iter().map(|&v| v * s).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Scalar divide (self / scalar).
    pub fn div_scalar(&self, s: f64) -> Result<Self> {
        self.mul_scalar(1.0 / s as f32)
    }

    /// Element-wise exp.
    pub fn exp(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let n = self.elem_count();
                let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
                super::kernels::gpu_exp(reg, src, &mut out, n).map_err(|e| TqError::Msg(format!("exp: {}", e)))?;
                return Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()));
            }
        }
        let data: Vec<f32> = self.as_slice().iter().map(|v| v.exp()).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise sqrt.
    pub fn sqrt(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let n = self.elem_count();
                let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
                super::kernels::gpu_sqrt(reg, src, &mut out, n).map_err(|e| TqError::Msg(format!("sqrt: {}", e)))?;
                return Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()));
            }
        }
        let data: Vec<f32> = self.as_slice().iter().map(|v| v.sqrt()).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise square.
    pub fn sqr(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let n = self.elem_count();
                let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
                super::kernels::gpu_sqr(reg, src, &mut out, n).map_err(|e| TqError::Msg(format!("sqr: {}", e)))?;
                return Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()));
            }
        }
        let data: Vec<f32> = self.as_slice().iter().map(|v| v * v).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise SiLU (x * sigmoid(x)).
    pub fn silu(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let n = self.elem_count();
                let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
                super::kernels::silu(reg, src, &mut out, n).map_err(|e| TqError::Msg(format!("silu: {}", e)))?;
                return Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()));
            }
        }
        let data: Vec<f32> = self.as_slice().iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let n = self.elem_count();
                let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
                super::kernels::gpu_cos(reg, src, &mut out, n).map_err(|e| TqError::Msg(format!("cos: {}", e)))?;
                return Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()));
            }
        }
        let data: Vec<f32> = self.as_slice().iter().map(|v| v.cos()).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                let n = self.elem_count();
                let mut out = gpu_alloc_zeros::<f32>(stream,n).map_err(|e| TqError::Msg(format!("{}", e)))?;
                super::kernels::gpu_sin(reg, src, &mut out, n).map_err(|e| TqError::Msg(format!("sin: {}", e)))?;
                return Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()));
            }
        }
        let data: Vec<f32> = self.as_slice().iter().map(|v| v.sin()).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Sum along a dimension, keeping the dimension as size 1.
    pub fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        super::ops::TqOps::reduce_keepdim(self, dim, |slice| slice.iter().sum())
    }

    /// Max along a dimension, keeping the dimension as size 1.
    pub fn max_keepdim(&self, dim: usize) -> Result<Self> {
        super::ops::TqOps::reduce_keepdim(self, dim, |slice| {
            slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        })
    }

    /// Mean along a dimension, keeping the dimension as size 1.
    pub fn mean_keepdim(&self, dim: usize) -> Result<Self> {
        super::ops::TqOps::reduce_keepdim(self, dim, |slice| {
            slice.iter().sum::<f32>() / slice.len() as f32
        })
    }

    /// Argmax along the last dimension, returns indices.
    pub fn argmax_last(&self) -> Result<Vec<u32>> {
        let data = self.as_slice();
        let last_dim = *self.shape.last().ok_or("argmax: empty shape")?;
        let n_rows = self.elem_count() / last_dim;
        let mut indices = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let start = row * last_dim;
            let (max_idx, _) = data[start..start + last_dim].iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            indices.push(max_idx as u32);
        }
        Ok(indices)
    }

    /// Arange: [0, 1, 2, ..., n-1] as f32.
    pub fn arange(start: usize, end: usize, device: &TqDevice) -> Result<Self> {
        let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
        let len = data.len();
        Self::from_vec(data, vec![len], device)
    }

    // ─── Additional methods (candle compatibility) ───────────

    /// Split tensor into `n` chunks along `dim`.
    pub fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Self>> {
        let dim = self.resolve_dim(dim)?;
        let size = self.shape[dim];
        let chunk_size = (size + n - 1) / n;
        let mut chunks = Vec::with_capacity(n);
        let mut start = 0;
        while start < size {
            let len = chunk_size.min(size - start);
            chunks.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(chunks)
    }

    /// Create a zero tensor with the same shape.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape.clone(), self.device())
    }

    /// Get data as Vec<Vec<f32>> (rank-2 only).
    pub fn to_vec2(&self) -> Result<Vec<Vec<f32>>> {
        let (rows, cols) = self.dims2()?;
        let data = self.as_slice();
        let mut result = Vec::with_capacity(rows);
        for r in 0..rows {
            result.push(data[r * cols..(r + 1) * cols].to_vec());
        }
        Ok(result)
    }

    /// Select rows/slices by index along a dimension.
    pub fn index_select(&self, indices: &TqTensor, dim: usize) -> Result<Self> {
        let dim = self.resolve_dim(dim)?;
        let idx = indices.as_slice();
        let data = self.as_slice();
        let outer: usize = self.shape[..dim].iter().product();
        let dim_size = self.shape[dim];
        let inner: usize = self.shape[dim + 1..].iter().product();
        let n_idx = idx.len();

        let mut result = Vec::with_capacity(outer * n_idx * inner);
        for o in 0..outer {
            for &i in idx {
                let i = i as usize;
                let src = o * dim_size * inner + i * inner;
                result.extend_from_slice(&data[src..src + inner]);
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = n_idx;
        Self::from_vec(result, new_shape, self.device())
    }

    /// Scatter-add: self[indices[i]] += src[i] along dimension.
    pub fn index_add(&self, indices: &TqTensor, src: &TqTensor, dim: usize) -> Result<Self> {
        let dim = self.resolve_dim(dim)?;
        let idx = indices.as_slice();
        let mut result = self.to_vec1()?;
        let src_data = src.as_slice();
        let inner: usize = self.shape[dim + 1..].iter().product();

        for (src_i, &target_i) in idx.iter().enumerate() {
            let target_i = target_i as usize;
            for j in 0..inner {
                result[target_i * inner + j] += src_data[src_i * inner + j];
            }
        }

        Self::from_vec(result, self.shape.clone(), self.device())
    }

    /// Expand (broadcast) size-1 dimensions to target shape.
    pub fn expand(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let target = shape.into();
        if target.len() != self.rank() {
            tq_bail!("expand: target rank {} != source rank {}", target.len(), self.rank());
        }

        let rank = self.rank();

        // Source strides (row-major), zeroed for size-1 dims (broadcast)
        let mut src_strides = vec![1i32; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            src_strides[i] = src_strides[i + 1] * self.shape[i + 1] as i32;
        }
        for i in 0..rank {
            if self.shape[i] == 1 {
                src_strides[i] = 0; // broadcast dimension
            } else if self.shape[i] != target[i] {
                tq_bail!("expand: dim {} size {} != target {}", i, self.shape[i], target[i]);
            }
        }

        // Target strides
        let mut tgt_strides = vec![1i32; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            tgt_strides[i] = tgt_strides[i + 1] * target[i + 1] as i32;
        }

        let n = target.iter().product::<usize>();

        // GPU path: use strided_copy with broadcast strides (0 = repeat)
        #[cfg(feature = "cuda")]
        if let TqStorage::Cuda { data: src, stream } = &self.storage {
            if let Some(reg) = super::kernels::global_registry() {
                // Clear stale errors from CudaSlice::drop → cuMemFreeAsync during graph capture
                let _ = stream.context().check_err();
                let mut out = gpu_alloc_zeros::<f32>(stream,n)
                    .map_err(|e| TqError::Msg(format!("expand alloc (n={}): {}", n, e)))?;
                let shape_gpu = stream.clone_htod(&target.iter().map(|&x| x as i32).collect::<Vec<_>>())
                    .map_err(|e| TqError::Msg(format!("expand: {}", e)))?;
                let tgt_gpu = stream.clone_htod(&tgt_strides)
                    .map_err(|e| TqError::Msg(format!("expand: {}", e)))?;
                let src_gpu = stream.clone_htod(&src_strides)
                    .map_err(|e| TqError::Msg(format!("expand: {}", e)))?;

                super::kernels::strided_copy(reg, src.as_ref(), &mut out, n, rank, &shape_gpu, &tgt_gpu, &src_gpu, 0)
                    .map_err(|e| TqError::Msg(format!("expand kernel: {}", e)))?;

                std::mem::forget(shape_gpu);
                std::mem::forget(tgt_gpu);
                std::mem::forget(src_gpu);

                return Ok(Self::from_cuda(out, target, stream.clone()));
            }
        }

        // CPU path
        let data = self.as_slice();
        let mut result = Vec::with_capacity(n);
        for flat in 0..n {
            let mut src_idx = 0;
            let mut remaining = flat;
            for d in 0..rank {
                let coord = remaining / tgt_strides[d] as usize;
                remaining %= tgt_strides[d] as usize;
                src_idx += coord * src_strides[d] as usize;
            }
            result.push(data[src_idx]);
        }

        Self::from_vec(result, target, self.device())
    }

    /// Broadcast self to target shape (alias for expand).
    pub fn broadcast_as(&self, shape: &[usize]) -> Result<Self> {
        self.expand(shape.to_vec())
    }

    /// Conditional select: where self != 0 pick on_true, else on_false.
    pub fn where_cond(&self, on_true: &TqTensor, on_false: &TqTensor) -> Result<TqTensor> {
        let mask = self.as_slice();
        let t = on_true.as_slice();
        let f = on_false.as_slice();
        let n = self.elem_count();
        if t.len() != n || f.len() != n {
            tq_bail!("where_cond: shape mismatch");
        }
        let result: Vec<f32> = (0..n)
            .map(|i| if mask[i] != 0.0 { t[i] } else { f[i] })
            .collect();
        Self::from_vec(result, self.shape.clone(), self.device())
    }

    // ─── Internal helpers ──────────────────────────────────────

    fn resolve_dim(&self, dim: usize) -> Result<usize> {
        if dim >= self.rank() {
            return Err(TqError::DimOutOfBounds { dim, rank: self.rank() });
        }
        Ok(dim)
    }

    /// True if this tensor lives on a CUDA device.
    pub fn is_cuda(&self) -> bool {
        match &self.storage {
            TqStorage::Cpu(_) => false,
            #[cfg(feature = "cuda")]
            TqStorage::Cuda { .. } => true,
        }
    }
}

// ─── GPU dispatch (cuda feature only) ─────────────────────────
#[cfg(feature = "cuda")]
impl TqTensor {
    /// Create from a Vec<f32> and shape, uploading to GPU when device is CUDA.
    ///
    /// Use this instead of `from_vec` when you want the tensor on the target device.
    pub fn from_vec_on_device(data: Vec<f32>, shape: impl Into<Vec<usize>>, device: &TqDevice) -> Result<Self> {
        let shape = shape.into();
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            tq_bail!("from_vec_on_device: data len {} != shape product {}", data.len(), expected);
        }
        match device {
            TqDevice::Cpu => Ok(Self {
                storage: TqStorage::Cpu(data),
                shape,
                dtype: TqDType::F32,
            }),
            TqDevice::Cuda { .. } => {
                let stream = device.cuda_stream()?;
                let cuda_data = stream.clone_htod(&data)
                    .map_err(|e| TqError::Cuda(e))?;
                let arc = std::sync::Arc::new(cuda_data);
                graph_retention_keep(&arc);
                Ok(Self {
                    storage: TqStorage::Cuda { data: arc, stream },
                    shape,
                    dtype: TqDType::F32,
                })
            }
        }
    }

    /// Create a zeros tensor on the target device (GPU-allocated when CUDA).
    pub fn zeros_on_device(shape: impl Into<Vec<usize>>, device: &TqDevice) -> Result<Self> {
        let shape = shape.into();
        let n: usize = shape.iter().product();
        match device {
            TqDevice::Cpu => Self::from_vec(vec![0.0; n], shape, device),
            TqDevice::Cuda { .. } => {
                let stream = device.cuda_stream()?;
                let cuda_data = gpu_alloc_zeros::<f32>(&stream, n)
                    .map_err(|e| TqError::Cuda(e))?;
                let arc = std::sync::Arc::new(cuda_data);
                graph_retention_keep(&arc);
                Ok(Self {
                    storage: TqStorage::Cuda { data: arc, stream },
                    shape,
                    dtype: TqDType::F32,
                })
            }
        }
    }

    /// Transfer tensor to a different device (CPU<->GPU).
    ///
    /// - CPU->GPU: uploads via `stream.clone_htod`
    /// - GPU->CPU: downloads via `stream.clone_dtoh`
    /// - Same device: returns a clone
    pub fn to_device(&self, device: &TqDevice) -> Result<Self> {
        match (&self.storage, device) {
            // CPU -> CPU: clone
            (TqStorage::Cpu(_), TqDevice::Cpu) => Ok(self.clone()),

            // CPU -> GPU: upload
            (TqStorage::Cpu(data), TqDevice::Cuda { .. }) => {
                let stream = device.cuda_stream()?;
                let cuda_data = stream.clone_htod(data)
                    .map_err(|e| TqError::Cuda(e))?;
                let arc = std::sync::Arc::new(cuda_data);
                graph_retention_keep(&arc);
                Ok(Self {
                    storage: TqStorage::Cuda { data: arc, stream },
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                })
            }

            // GPU -> CPU: download
            (TqStorage::Cuda { data, stream }, TqDevice::Cpu) => {
                let host_data = stream.clone_dtoh(data.as_ref())
                    .map_err(|e| TqError::Cuda(e))?;
                Ok(Self {
                    storage: TqStorage::Cpu(host_data),
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                })
            }

            // GPU -> GPU (same ordinal): clone
            (TqStorage::Cuda { .. }, TqDevice::Cuda { .. }) => {
                // TODO: cross-device transfer when multi-GPU is wired up
                Ok(self.clone())
            }
        }
    }

    /// Get a reference to the underlying CudaSlice for kernel launches.
    ///
    /// Panics if the tensor is on CPU.
    pub fn cuda_data(&self) -> &cudarc::driver::CudaSlice<f32> {
        match &self.storage {
            TqStorage::Cuda { data, .. } => data.as_ref(),
            _ => panic!("cuda_data() called on CPU tensor — use to_device() first"),
        }
    }

    /// Get a mutable reference to the underlying CudaSlice (copy-on-write).
    ///
    /// If the Arc has other references, this clones the GPU buffer first.
    /// Panics if the tensor is on CPU.
    pub fn cuda_data_mut(&mut self) -> &mut cudarc::driver::CudaSlice<f32> {
        match &mut self.storage {
            TqStorage::Cuda { data, .. } => std::sync::Arc::make_mut(data),
            _ => panic!("cuda_data_mut() called on CPU tensor — use to_device() first"),
        }
    }

    /// Get the CUDA stream associated with this tensor.
    ///
    /// Panics if the tensor is on CPU.
    pub fn cuda_stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        match &self.storage {
            TqStorage::Cuda { stream, .. } => stream,
            _ => panic!("cuda_stream() called on CPU tensor"),
        }
    }

    /// Ensure the tensor lives on GPU, uploading if needed.
    ///
    /// If already on GPU, returns a clone. If on CPU, uploads to the given device.
    pub fn ensure_gpu(&self, device: &TqDevice) -> Result<Self> {
        if self.is_cuda() {
            Ok(self.clone())
        } else {
            self.to_device(device)
        }
    }

    /// Upload to GPU if global registry is available. No-op if already on GPU.
    pub fn to_device_auto(&self) -> Result<Self> {
        if self.is_cuda() { return Ok(self.clone()); }
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry for auto-upload".into()))?;
        let stream = reg.stream.clone();
        match &self.storage {
            TqStorage::Cpu(data) => {
                let gpu = stream.clone_htod(data).map_err(|e| TqError::Msg(format!("auto-upload: {}", e)))?;
                let arc = std::sync::Arc::new(gpu);
                graph_retention_keep(&arc);
                Ok(Self { storage: TqStorage::Cuda { data: arc, stream }, shape: self.shape.clone(), dtype: self.dtype })
            }
            _ => Ok(self.clone()),
        }
    }

    /// Create a TqTensor directly from a CudaSlice (already on GPU).
    ///
    /// Used by kernel launchers to wrap output buffers as tensors.
    /// Create a TqTensor directly from a CudaSlice (already on GPU).
    /// Wraps in Arc for zero-copy clone semantics.
    pub fn from_cuda(
        data: cudarc::driver::CudaSlice<f32>,
        shape: Vec<usize>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Self {
        let arc = std::sync::Arc::new(data);
        Self { storage: TqStorage::Cuda { data: arc, stream }, shape, dtype: TqDType::F32 }
    }

    /// Create a TqTensor from an existing Arc<CudaSlice> (zero-copy view).
    /// Used by GpuKvCache to wrap pre-allocated buffers as tensors.

    /// Create a TqTensor from an existing Arc<CudaSlice> (zero-copy view).
    /// Used by GpuKvCache to wrap pre-allocated buffers as tensors.
    #[cfg(feature = "cuda")]
    pub fn from_cuda_arc(
        data: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
        shape: Vec<usize>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Self {
        graph_retention_keep(&data);
        Self {
            storage: TqStorage::Cuda { data, stream },
            shape,
            dtype: TqDType::F32,
        }
    }

    /// GPU matmul via cuBLAS SGEMM: C = A @ B.
    ///
    /// For 2D: standard SGEMM.
    /// For higher rank: strided batched SGEMM (batch dims broadcast).
    fn matmul_cublas(&self, other: &TqTensor) -> Result<Self> {
        use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
        use cudarc::cublas::sys::cublasOperation_t;

        let a_shape = self.shape().to_vec();
        let b_shape = other.shape().to_vec();
        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        if a_rank < 2 || b_rank < 2 {
            tq_bail!("matmul_cublas: need rank >= 2, got {} and {}", a_rank, b_rank);
        }

        let m = a_shape[a_rank - 2]; // rows of A
        let k = a_shape[a_rank - 1]; // cols of A = rows of B
        let n = b_shape[b_rank - 1]; // cols of B

        if b_shape[b_rank - 2] != k {
            tq_bail!("matmul_cublas: inner dims mismatch {} vs {}", k, b_shape[b_rank - 2]);
        }

        let stream = self.cuda_stream().clone();
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU kernel registry for matmul".into()))?;
        let blas = reg.get_cublas();

        let batch: usize = a_shape[..a_rank - 2].iter().product();

        let mut out_gpu = gpu_alloc_zeros::<f32>(&stream, batch * m * n)
            .map_err(|e| TqError::Msg(format!("matmul alloc: {}", e)))?;

        // cuBLAS is column-major. To compute row-major C = A @ B,
        // we compute C^T = B^T @ A^T in column-major, which gives us
        // C in row-major layout. So we swap A and B and transpose flags.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N, // B (no transpose in col-major = transpose in row-major)
            transb: cublasOperation_t::CUBLAS_OP_N, // A
            m: n as i32,     // cols of C (= cols of B in row-major)
            n: m as i32,     // rows of C (= rows of A in row-major)
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,   // leading dim of B in col-major
            ldb: k as i32,   // leading dim of A in col-major
            beta: 0.0f32,
            ldc: n as i32,   // leading dim of C in col-major
        };

        if batch == 1 {
            unsafe {
                blas.gemm(cfg, other.cuda_data(), self.cuda_data(), &mut out_gpu)
                    .map_err(|e| TqError::Msg(format!("cuBLAS sgemm: {}", e)))?;
            }
        } else {
            use cudarc::cublas::StridedBatchedConfig;
            let strided_cfg = StridedBatchedConfig {
                gemm: cfg,
                batch_size: batch as i32,
                stride_a: (k * n) as i64,
                stride_b: (m * k) as i64,
                stride_c: (m * n) as i64,
            };
            unsafe {
                blas.gemm_strided_batched(strided_cfg, other.cuda_data(), self.cuda_data(), &mut out_gpu)
                    .map_err(|e| TqError::Msg(format!("cuBLAS batched sgemm: {}", e)))?;
            }
        }

        let mut out_shape = a_shape[..a_rank - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Ok(Self::from_cuda(out_gpu, out_shape, stream))
    }

    /// GPU matvec with pre-transposed cached weight: output = x @ W^T.
    /// W^T is a borrowed &CudaSlice<f32> [in_features, out_features] — no clone needed.
    /// Used for Q6K and other dtypes where weight is dequantized + pre-transposed once.
    pub fn matvec_with_cached_wt(
        &self,
        wt: &cudarc::driver::CudaSlice<f32>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        use cudarc::cublas::{Gemm, GemmConfig};
        use cudarc::cublas::sys::cublasOperation_t;

        let stream = self.cuda_stream().clone();
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry for matvec".into()))?;
        let blas = reg.get_cublas();

        let x_shape = self.shape().to_vec();
        let batch: usize = x_shape[..x_shape.len() - 1].iter().product::<usize>().max(1);
        let m = batch;          // rows of x
        let k = in_features;    // cols of x = rows of W^T
        let n = out_features;   // cols of W^T

        let mut out_gpu = gpu_alloc_zeros::<f32>(&stream, m * n)
            .map_err(|e| TqError::Msg(format!("matvec alloc: {}", e)))?;

        // cuBLAS col-major: C^T = B^T @ A^T gives row-major C = A @ B
        // A = x [m, k], B = W^T [k, n]
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            blas.gemm(cfg, wt, self.cuda_data(), &mut out_gpu)
                .map_err(|e| TqError::Msg(format!("cuBLAS matvec: {}", e)))?;
        }

        let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(Self::from_cuda(out_gpu, out_shape, stream))
    }
}

// ─── Operator overloads ────────────────────────────────────────

impl std::ops::Add<TqTensor> for TqTensor {
    type Output = Result<TqTensor>;
    fn add(self, rhs: TqTensor) -> Result<TqTensor> { self.broadcast_add(&rhs) }
}

impl std::ops::Add<&TqTensor> for TqTensor {
    type Output = Result<TqTensor>;
    fn add(self, rhs: &TqTensor) -> Result<TqTensor> { self.broadcast_add(rhs) }
}

impl std::ops::Add<TqTensor> for &TqTensor {
    type Output = Result<TqTensor>;
    fn add(self, rhs: TqTensor) -> Result<TqTensor> { self.broadcast_add(&rhs) }
}

impl std::ops::Sub<TqTensor> for TqTensor {
    type Output = Result<TqTensor>;
    fn sub(self, rhs: TqTensor) -> Result<TqTensor> { self.broadcast_sub(&rhs) }
}

impl std::ops::Sub<&TqTensor> for TqTensor {
    type Output = Result<TqTensor>;
    fn sub(self, rhs: &TqTensor) -> Result<TqTensor> { self.broadcast_sub(rhs) }
}

impl std::ops::Mul<TqTensor> for TqTensor {
    type Output = Result<TqTensor>;
    fn mul(self, rhs: TqTensor) -> Result<TqTensor> { self.broadcast_mul(&rhs) }
}

impl std::ops::Mul<&TqTensor> for TqTensor {
    type Output = Result<TqTensor>;
    fn mul(self, rhs: &TqTensor) -> Result<TqTensor> { self.broadcast_mul(rhs) }
}

impl std::ops::Div<f64> for TqTensor {
    type Output = Result<TqTensor>;
    fn div(self, rhs: f64) -> Result<TqTensor> { self.div_scalar(rhs) }
}

impl std::ops::Div<f64> for &TqTensor {
    type Output = Result<TqTensor>;
    fn div(self, rhs: f64) -> Result<TqTensor> { self.div_scalar(rhs) }
}

// ─── GPU-native compute methods (Phase 2) ─────────────────────
//
// These bypass ComputeBackend and keep data on GPU.
// Used directly from turbo_generic.rs when tensors are CUDA-resident.

#[cfg(feature = "cuda")]
impl TqTensor {
    /// GPU RMS normalization: output = x * weight / rms(x).
    /// weight must be a CPU tensor (norm weights are small, cached on GPU via registry).
    pub fn rms_norm_gpu(&self, weight: &TqTensor, eps: f32) -> Result<Self> {
        let TqStorage::Cuda { data: x, stream } = &self.storage else {
            tq_bail!("rms_norm_gpu: tensor not on GPU");
        };
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let shape = self.shape.clone();
        let hidden = *shape.last().unwrap();
        let n_tokens = self.elem_count() / hidden;
        let n = n_tokens * hidden;

        let mut out = gpu_alloc_zeros::<f32>(stream,n)
            .map_err(|e| TqError::Msg(format!("rms_norm alloc: {}", e)))?;

        // Weight: use GPU data directly (no clone) or upload once
        if weight.is_cuda() {
            super::kernels::rms_norm(reg, x, weight.cuda_data(), &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("rms_norm kernel: {}", e)))?;
        } else {
            let w_gpu = stream.clone_htod(&weight.to_vec1()?)
                .map_err(|e| TqError::Msg(format!("rms_norm weight: {}", e)))?;
            super::kernels::rms_norm(reg, x, &w_gpu, &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("rms_norm kernel: {}", e)))?;
        }

        Ok(Self::from_cuda(out, shape, stream.clone()))
    }

    /// GPU softmax along last dimension.
    pub fn softmax_gpu(&self) -> Result<Self> {
        let TqStorage::Cuda { data: x, stream } = &self.storage else {
            tq_bail!("softmax_gpu: tensor not on GPU");
        };
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let shape = self.shape.clone();
        let cols = *shape.last().unwrap();
        let rows = self.elem_count() / cols;

        let mut out = gpu_alloc_zeros::<f32>(stream,rows * cols)
            .map_err(|e| TqError::Msg(format!("softmax alloc: {}", e)))?;

        super::kernels::softmax_last_dim(reg, x, &mut out, rows, cols)
            .map_err(|e| TqError::Msg(format!("softmax kernel: {}", e)))?;

        Ok(Self::from_cuda(out, shape, stream.clone()))
    }

    /// GPU fused SiLU × multiply: output = silu(self) * up.
    pub fn fused_silu_mul_gpu(&self, up: &TqTensor) -> Result<Self> {
        let TqStorage::Cuda { data: gate, stream } = &self.storage else {
            tq_bail!("fused_silu_mul_gpu: gate not on GPU");
        };
        let TqStorage::Cuda { data: up_data, .. } = &up.storage else {
            tq_bail!("fused_silu_mul_gpu: up not on GPU");
        };
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let n = self.elem_count();
        let mut out = gpu_alloc_zeros::<f32>(stream,n)
            .map_err(|e| TqError::Msg(format!("fused_silu_mul alloc: {}", e)))?;

        super::kernels::fused_silu_mul(reg, gate, up_data, &mut out, n)
            .map_err(|e| TqError::Msg(format!("fused_silu_mul kernel: {}", e)))?;

        Ok(Self::from_cuda(out, self.shape.clone(), stream.clone()))
    }

    /// GPU fused residual add + RMS norm.
    /// Returns (normalized_output, updated_residual).
    pub fn fused_add_rms_norm_gpu(&self, residual: &TqTensor, weight: &TqTensor, eps: f32) -> Result<(Self, Self)> {
        let TqStorage::Cuda { data: input, stream } = &self.storage else {
            tq_bail!("fused_add_rms_norm_gpu: input not on GPU");
        };
        let TqStorage::Cuda { data: res, .. } = &residual.storage else {
            tq_bail!("fused_add_rms_norm_gpu: residual not on GPU");
        };
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let shape = self.shape.clone();
        let hidden = *shape.last().unwrap();
        let n_tokens = self.elem_count() / hidden;
        let n = n_tokens * hidden;

        // Residual: need a mutable copy (kernel modifies in-place).
        // Arc::make_mut ensures copy-on-write: clones only if ref count > 1.
        let mut res_arc = res.clone();
        let res_mut = std::sync::Arc::make_mut(&mut res_arc);
        let mut out = gpu_alloc_zeros::<f32>(stream,n)
            .map_err(|e| TqError::Msg(format!("fused norm alloc: {}", e)))?;

        // Weight: use GPU data directly (no clone) or upload
        if weight.is_cuda() {
            super::kernels::fused_add_rms_norm(reg, input.as_ref(), res_mut, weight.cuda_data(), &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("fused_add_rms_norm kernel: {}", e)))?;
        } else {
            let w_gpu = stream.clone_htod(&weight.to_vec1()?)
                .map_err(|e| TqError::Msg(format!("fused norm weight: {}", e)))?;
            super::kernels::fused_add_rms_norm(reg, input.as_ref(), res_mut, &w_gpu, &mut out, n_tokens, hidden, eps)
                .map_err(|e| TqError::Msg(format!("fused_add_rms_norm kernel: {}", e)))?;
        }

        Ok((
            Self::from_cuda(out, shape.clone(), stream.clone()),
            { graph_retention_keep(&res_arc); Self { storage: TqStorage::Cuda { data: res_arc, stream: stream.clone() }, shape, dtype: TqDType::F32 } },
        ))
    }

    /// GPU quantized matvec: output = x @ W^T where W is Q4_K_M or Q8_0.
    /// Weight bytes must already be on GPU (via QWeight::gpu_cache_or_upload).
    pub fn qmatmul_gpu(
        &self,
        w_gpu: &cudarc::driver::CudaSlice<u8>,
        dtype: crate::gguf::GgmlDType,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        let TqStorage::Cuda { data: x, stream } = &self.storage else {
            tq_bail!("qmatmul_gpu: input not on GPU");
        };
        let reg = super::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;

        let x_shape = self.shape.clone();
        let batch: usize = x_shape[..x_shape.len() - 1].iter().product();

        // Only decode (batch=1) uses fused kernel; prefill uses cuBLAS via dequant
        if batch == 1 {
            let mut out = gpu_alloc_zeros::<f32>(stream,out_features)
                .map_err(|e| TqError::Msg(format!("qmatmul alloc: {}", e)))?;

            match dtype {
                crate::gguf::GgmlDType::Q4K => {
                    super::kernels::q4km_matvec(reg, w_gpu, x, &mut out, out_features, in_features)
                        .map_err(|e| TqError::Msg(format!("q4km_matvec: {}", e)))?;
                }
                crate::gguf::GgmlDType::Q8_0 => {
                    super::kernels::q8_0_matvec(reg, w_gpu, x, &mut out, out_features, in_features)
                        .map_err(|e| TqError::Msg(format!("q8_0_matvec: {}", e)))?;
                }
                _ => tq_bail!("qmatmul_gpu: unsupported dtype {:?}", dtype),
            }

            let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
            out_shape.push(out_features);
            Ok(Self::from_cuda(out, out_shape, stream.clone()))
        } else {
            tq_bail!("qmatmul_gpu: batch>1 not yet supported (use cuBLAS path)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec_and_shape() {
        let t = TqTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &TqDevice::Cpu).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.elem_count(), 6);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn test_reshape() {
        let t = TqTensor::from_vec(vec![1.0; 24], vec![2, 3, 4], &TqDevice::Cpu).unwrap();
        let r = t.reshape(vec![6, 4]).unwrap();
        assert_eq!(r.shape(), &[6, 4]);
    }

    #[test]
    fn test_narrow() {
        let t = TqTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &TqDevice::Cpu
        ).unwrap();
        let n = t.narrow(1, 1, 2).unwrap();
        assert_eq!(n.shape(), &[2, 2]);
        assert_eq!(n.to_vec1().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cat() {
        let a = TqTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &TqDevice::Cpu).unwrap();
        let b = TqTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], &TqDevice::Cpu).unwrap();
        let c = TqTensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[4, 2]);
        assert_eq!(c.to_vec1().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_transpose() {
        let t = TqTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &TqDevice::Cpu
        ).unwrap();
        let tr = t.t().unwrap();
        assert_eq!(tr.shape(), &[3, 2]);
        assert_eq!(tr.to_vec1().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul_2d() {
        // [2,3] @ [3,2] = [2,2]
        let a = TqTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &TqDevice::Cpu).unwrap();
        let b = TqTensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], &TqDevice::Cpu).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        let data = c.to_vec1().unwrap();
        assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_silu() {
        let t = TqTensor::from_vec(vec![0.0, 1.0, -1.0], vec![3], &TqDevice::Cpu).unwrap();
        let s = t.silu().unwrap();
        let data = s.to_vec1().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_broadcast_add() {
        let a = TqTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &TqDevice::Cpu).unwrap();
        let b = TqTensor::from_vec(vec![10.0, 20.0], vec![1, 2], &TqDevice::Cpu).unwrap();
        let c = a.broadcast_add(&b).unwrap();
        assert_eq!(c.to_vec1().unwrap(), vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_chunk() {
        let t = TqTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], &TqDevice::Cpu
        ).unwrap();
        let chunks = t.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[2]);
        assert_eq!(chunks[0].to_vec1().unwrap(), vec![1.0, 2.0]);
        assert_eq!(chunks[1].to_vec1().unwrap(), vec![3.0, 4.0]);
        assert_eq!(chunks[2].to_vec1().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_zeros_like() {
        let t = TqTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3], &TqDevice::Cpu).unwrap();
        let z = t.zeros_like().unwrap();
        assert_eq!(z.shape(), &[1, 3]);
        assert_eq!(z.to_vec1().unwrap(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_expand() {
        // Expand size-1 dim: [1, 3] -> [4, 3]
        let t = TqTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3], &TqDevice::Cpu).unwrap();
        let e = t.expand(vec![4, 3]).unwrap();
        assert_eq!(e.shape(), &[4, 3]);
        let data = e.to_vec1().unwrap();
        // Each row should be [1, 2, 3] repeated 4 times
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_index_select() {
        // 3x2 matrix, select rows 0 and 2
        let t = TqTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &TqDevice::Cpu
        ).unwrap();
        let indices = TqTensor::from_vec(vec![0.0, 2.0], vec![2], &TqDevice::Cpu).unwrap();
        let s = t.index_select(&indices, 0).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.to_vec1().unwrap(), vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_where_cond() {
        let mask = TqTensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4], &TqDevice::Cpu).unwrap();
        let on_true = TqTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4], &TqDevice::Cpu).unwrap();
        let on_false = TqTensor::from_vec(vec![100.0, 200.0, 300.0, 400.0], vec![4], &TqDevice::Cpu).unwrap();
        let result = mask.where_cond(&on_true, &on_false).unwrap();
        assert_eq!(result.to_vec1().unwrap(), vec![10.0, 200.0, 30.0, 400.0]);
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let t = TqTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &TqDevice::Cpu).unwrap();
        // unsqueeze at dim 0: [3] -> [1, 3]
        let u = t.unsqueeze(0).unwrap();
        assert_eq!(u.shape(), &[1, 3]);
        // squeeze dim 0: [1, 3] -> [3]
        let s = u.squeeze(0).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec1().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_operator_add() {
        let a = TqTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &TqDevice::Cpu).unwrap();
        let b = TqTensor::from_vec(vec![4.0, 5.0, 6.0], vec![3], &TqDevice::Cpu).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.to_vec1().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_operator_sub() {
        let a = TqTensor::from_vec(vec![10.0, 20.0, 30.0], vec![3], &TqDevice::Cpu).unwrap();
        let b = TqTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &TqDevice::Cpu).unwrap();
        let c = (a - b).unwrap();
        assert_eq!(c.to_vec1().unwrap(), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_operator_mul() {
        let a = TqTensor::from_vec(vec![2.0, 3.0, 4.0], vec![3], &TqDevice::Cpu).unwrap();
        let b = TqTensor::from_vec(vec![5.0, 6.0, 7.0], vec![3], &TqDevice::Cpu).unwrap();
        let c = (a * b).unwrap();
        assert_eq!(c.to_vec1().unwrap(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_scalar() {
        let t = TqTensor::from_vec(vec![10.0, 20.0, 30.0], vec![3], &TqDevice::Cpu).unwrap();
        let d = (t / 2.0).unwrap();
        assert_eq!(d.to_vec1().unwrap(), vec![5.0, 10.0, 15.0]);
    }
}
