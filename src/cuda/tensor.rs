//! Minimal tensor type — CPU Vec<f32> or CUDA device memory.
//!
//! Implements only the ~25 operations used by turbo_generic.rs.
//! No autograd, no dynamic dispatch overhead.

use super::{TqDevice, TqDType, Result, TqError, tq_bail};

/// Storage backend for tensor data.
#[derive(Debug, Clone)]
pub enum TqStorage {
    /// CPU: contiguous f32 data in row-major order.
    Cpu(Vec<f32>),
    /// CUDA: device memory on GPU.
    #[cfg(feature = "cuda")]
    Cuda {
        data: cudarc::driver::CudaSlice<f32>,
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

    /// Get underlying f32 data (CPU only).
    pub fn to_vec1(&self) -> Result<Vec<f32>> {
        match &self.storage {
            TqStorage::Cpu(data) => Ok(data.clone()),
            #[cfg(feature = "cuda")]
            TqStorage::Cuda { data, stream } => {
                stream.clone_dtoh(data)
                    .map_err(|e| TqError::Msg(format!("dtoh: {}", e)))
            }
        }
    }

    /// Borrow underlying f32 slice (CPU only, panics on CUDA).
    pub fn as_slice(&self) -> &[f32] {
        match &self.storage {
            TqStorage::Cpu(data) => data,
            #[cfg(feature = "cuda")]
            _ => panic!("as_slice() called on CUDA tensor — use to_vec1()"),
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

        let data = self.as_slice();
        let rank = self.rank();
        let n = self.elem_count();
        let mut result = vec![0.0f32; n];

        // Compute strides for old layout
        let mut old_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            old_strides[i] = old_strides[i + 1] * self.shape[i + 1];
        }

        // New shape and strides
        let mut new_shape = self.shape.clone();
        new_shape.swap(d1, d2);
        let mut new_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        // Copy with transposed indexing
        for flat_idx in 0..n {
            let mut remaining = flat_idx;
            let mut old_coords = vec![0usize; rank];
            for d in 0..rank {
                old_coords[d] = remaining / old_strides[d];
                remaining %= old_strides[d];
            }
            // Swap coordinates
            old_coords.swap(d1, d2);
            let new_flat: usize = old_coords.iter().zip(new_strides.iter())
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

        let data = self.as_slice();
        let rank = self.rank();

        // Compute strides
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = len;
        let new_n: usize = new_shape.iter().product();
        let mut result = Vec::with_capacity(new_n);

        // Iterate over all elements in the new tensor
        let outer: usize = self.shape[..dim].iter().product();
        let inner: usize = self.shape[dim + 1..].iter().product();

        for o in 0..outer {
            for s in start..start + len {
                let base = o * strides[dim] * self.shape[dim] + s * inner;
                // Wait, simpler approach for contiguous:
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
        let mut result = Vec::with_capacity(new_n);

        for o in 0..outer {
            for t in tensors {
                let t_data = t.as_slice();
                let t_dim = t.shape[dim];
                let src_offset = o * t_dim * inner;
                result.extend_from_slice(&t_data[src_offset..src_offset + t_dim * inner]);
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
    pub fn matmul(&self, other: &TqTensor) -> Result<Self> {
        super::ops::TqOps::matmul(self, other)
    }

    /// Element-wise multiply with broadcasting.
    pub fn broadcast_mul(&self, other: &TqTensor) -> Result<Self> {
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a * b)
    }

    /// Element-wise add with broadcasting.
    pub fn broadcast_add(&self, other: &TqTensor) -> Result<Self> {
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a + b)
    }

    /// Element-wise sub with broadcasting.
    pub fn broadcast_sub(&self, other: &TqTensor) -> Result<Self> {
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a - b)
    }

    /// Element-wise div with broadcasting.
    pub fn broadcast_div(&self, other: &TqTensor) -> Result<Self> {
        super::ops::TqOps::broadcast_binop(self, other, |a, b| a / b)
    }

    /// Scalar multiply.
    pub fn mul_scalar(&self, s: f32) -> Result<Self> {
        let data: Vec<f32> = self.as_slice().iter().map(|&v| v * s).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Scalar divide (self / scalar).
    pub fn div_scalar(&self, s: f64) -> Result<Self> {
        self.mul_scalar(1.0 / s as f32)
    }

    /// Element-wise exp.
    pub fn exp(&self) -> Result<Self> {
        let data: Vec<f32> = self.as_slice().iter().map(|v| v.exp()).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise sqrt.
    pub fn sqrt(&self) -> Result<Self> {
        let data: Vec<f32> = self.as_slice().iter().map(|v| v.sqrt()).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise square.
    pub fn sqr(&self) -> Result<Self> {
        let data: Vec<f32> = self.as_slice().iter().map(|v| v * v).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise SiLU (x * sigmoid(x)).
    pub fn silu(&self) -> Result<Self> {
        let data: Vec<f32> = self.as_slice().iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Result<Self> {
        let data: Vec<f32> = self.as_slice().iter().map(|v| v.cos()).collect();
        Self::from_vec(data, self.shape.clone(), self.device())
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Self> {
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
        let data = self.as_slice();

        // Source strides (row-major), zeroed for size-1 dims (broadcast)
        let mut src_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            src_strides[i] = src_strides[i + 1] * self.shape[i + 1];
        }
        for i in 0..rank {
            if self.shape[i] == 1 {
                src_strides[i] = 0;
            } else if self.shape[i] != target[i] {
                tq_bail!("expand: dim {} size {} != target {}", i, self.shape[i], target[i]);
            }
        }

        // Target strides
        let mut tgt_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            tgt_strides[i] = tgt_strides[i + 1] * target[i + 1];
        }

        let n = target.iter().product::<usize>();
        let mut result = Vec::with_capacity(n);
        for flat in 0..n {
            let mut src_idx = 0;
            let mut remaining = flat;
            for d in 0..rank {
                let coord = remaining / tgt_strides[d];
                remaining %= tgt_strides[d];
                src_idx += coord * src_strides[d];
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
                Ok(Self {
                    storage: TqStorage::Cuda { data: cuda_data, stream },
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
                let cuda_data = stream.alloc_zeros::<f32>(n)
                    .map_err(|e| TqError::Cuda(e))?;
                Ok(Self {
                    storage: TqStorage::Cuda { data: cuda_data, stream },
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
                Ok(Self {
                    storage: TqStorage::Cuda { data: cuda_data, stream },
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                })
            }

            // GPU -> CPU: download
            (TqStorage::Cuda { data, stream }, TqDevice::Cpu) => {
                let host_data = stream.clone_dtoh(data)
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
            TqStorage::Cuda { data, .. } => data,
            _ => panic!("cuda_data() called on CPU tensor — use to_device() first"),
        }
    }

    /// Get a mutable reference to the underlying CudaSlice for kernel launches.
    ///
    /// Panics if the tensor is on CPU.
    pub fn cuda_data_mut(&mut self) -> &mut cudarc::driver::CudaSlice<f32> {
        match &mut self.storage {
            TqStorage::Cuda { data, .. } => data,
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

    /// Create a TqTensor directly from a CudaSlice (already on GPU).
    ///
    /// Used by kernel launchers to wrap output buffers as tensors.
    pub fn from_cuda(
        data: cudarc::driver::CudaSlice<f32>,
        shape: Vec<usize>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Self {
        Self {
            storage: TqStorage::Cuda { data, stream },
            shape,
            dtype: TqDType::F32,
        }
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
}
