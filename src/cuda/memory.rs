//! GPU memory management — pool allocator, pinned memory, mmap GGUF.
//!
//! Avoids per-layer cudaMalloc overhead during inference.
//! Supports async H2D transfers for model loading.

use super::{Result, TqError};

/// Pre-allocated GPU memory pool for inference workspace.
///
/// Allocates one large buffer at startup, sub-allocates from it.
/// No fragmentation since allocations are strictly LIFO (layer-by-layer).
pub struct GpuMemoryPool {
    /// Total pool size in bytes.
    pub total_bytes: usize,
    /// Current offset (next free byte).
    pub offset: usize,
    /// Device ordinal.
    pub device_ordinal: usize,
    #[cfg(feature = "cuda")]
    pub device_ptr: Option<cudarc::driver::CudaSlice<u8>>,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool.
    pub fn new(size_bytes: usize, _device_ordinal: usize) -> Result<Self> {
        Ok(Self {
            total_bytes: size_bytes,
            offset: 0,
            device_ordinal: _device_ordinal,
            #[cfg(feature = "cuda")]
            device_ptr: None, // Allocated lazily on first use
        })
    }

    /// Sub-allocate from the pool.
    pub fn alloc(&mut self, size_bytes: usize, alignment: usize) -> Result<usize> {
        // Align offset
        let aligned = (self.offset + alignment - 1) / alignment * alignment;
        if aligned + size_bytes > self.total_bytes {
            return Err(TqError::Msg(format!(
                "GPU pool OOM: need {} bytes at offset {}, pool size {}",
                size_bytes, aligned, self.total_bytes
            )));
        }
        let ptr_offset = aligned;
        self.offset = aligned + size_bytes;
        Ok(ptr_offset)
    }

    /// Reset pool (free all sub-allocations). Called between forward passes.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Used bytes.
    pub fn used(&self) -> usize { self.offset }

    /// Free bytes.
    pub fn free(&self) -> usize { self.total_bytes - self.offset }
}

/// Pinned (page-locked) host memory for async H2D transfers.
///
/// Allocated with `cuMemHostAlloc` (via `CudaContext::alloc_pinned`) using the
/// `CU_MEMHOSTALLOC_WRITECOMBINED` flag — optimal for H2D-only staging buffers.
///
/// Two variants:
///   - `Pinned { .. }`   — real DMA-capable host memory (requires a CUDA context)
///   - `Pageable { .. }` — plain `Vec<u8>` fallback when no CUDA context is active
///
/// Only the pinned variant enables truly asynchronous H2D on a non-default
/// stream; pageable staging forces synchronous memcpy.
pub enum PinnedBuffer {
    #[cfg(feature = "cuda")]
    Pinned {
        /// Owns the pinned allocation; Drop frees via `cuMemFreeHost`.
        inner: cudarc::driver::PinnedHostSlice<u8>,
        size: usize,
    },
    Pageable {
        data: Vec<u8>,
        size: usize,
    },
}

impl PinnedBuffer {
    /// Allocate a pinned buffer from the given CUDA context. Returns `Pageable`
    /// fallback if the pinned allocation fails (e.g. OOM — pinned memory is a
    /// scarce resource on the host).
    #[cfg(feature = "cuda")]
    pub fn new_on_context(
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        size: usize,
    ) -> Self {
        // SAFETY: `alloc_pinned` leaves memory uninitialised; callers must write
        // before reading. Our usage always writes (loading from file/mmap)
        // before calling memcpy_htod.
        match unsafe { ctx.alloc_pinned::<u8>(size) } {
            Ok(inner) => PinnedBuffer::Pinned { inner, size },
            Err(e) => {
                eprintln!(
                    "[cuda] alloc_pinned({} bytes) failed: {} — falling back to pageable",
                    size, e
                );
                PinnedBuffer::Pageable { data: vec![0u8; size], size }
            }
        }
    }

    /// CPU-only constructor (no CUDA available — plain heap Vec).
    pub fn new(size: usize) -> Self {
        PinnedBuffer::Pageable { data: vec![0u8; size], size }
    }

    pub fn size(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            PinnedBuffer::Pinned { size, .. } => *size,
            PinnedBuffer::Pageable { size, .. } => *size,
        }
    }

    pub fn is_pinned(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            PinnedBuffer::Pinned { .. } => true,
            PinnedBuffer::Pageable { .. } => false,
        }
    }

    /// Read-only view of the host-side bytes. Synchronises with any pending
    /// stream work on pinned allocations (safe if no transfer is in flight).
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(feature = "cuda")]
            PinnedBuffer::Pinned { inner, size } => {
                let ptr = inner.as_ptr().expect("pinned event sync");
                unsafe { std::slice::from_raw_parts(ptr, *size) }
            }
            PinnedBuffer::Pageable { data, size } => &data[..*size],
        }
    }

    /// Mutable view for staging bytes before an H2D copy. Synchronises on
    /// pinned allocations — callers must not hold this across a memcpy_htod.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            #[cfg(feature = "cuda")]
            PinnedBuffer::Pinned { inner, size } => {
                let ptr = inner.as_mut_ptr().expect("pinned event sync");
                unsafe { std::slice::from_raw_parts_mut(ptr, *size) }
            }
            PinnedBuffer::Pageable { data, size } => &mut data[..*size],
        }
    }

    /// Access the underlying `PinnedHostSlice` if this buffer is pinned.
    /// Used by the LayerSwapManager for async `memcpy_htod` on the transfer
    /// stream.
    #[cfg(feature = "cuda")]
    pub fn as_pinned_slice(&mut self) -> Option<&mut cudarc::driver::PinnedHostSlice<u8>> {
        match self {
            PinnedBuffer::Pinned { inner, .. } => Some(inner),
            PinnedBuffer::Pageable { .. } => None,
        }
    }
}

/// KV cache memory planner.
///
/// Pre-computes max KV cache size at startup to avoid runtime allocation.
pub struct KvCachePlan {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Bytes per element in KV cache (2 for FP16, 4 for FP32, <1 for compressed).
    pub bytes_per_element: f32,
}

impl KvCachePlan {
    /// Calculate total KV cache size in bytes.
    pub fn total_bytes(&self) -> usize {
        let elements = self.n_layers * 2 * self.n_kv_heads * self.max_seq_len * self.head_dim;
        (elements as f64 * self.bytes_per_element as f64) as usize
    }

    /// With TurboQuant compression at given bit width.
    pub fn compressed_bytes(&self, bits: u8) -> usize {
        let k_elements = self.n_layers * self.n_kv_heads * self.max_seq_len * self.head_dim;
        let k_bytes = k_elements * bits as usize / 8;
        // V cache assumed fp16 by default
        let v_bytes = self.n_layers * self.n_kv_heads * self.max_seq_len * self.head_dim * 2;
        k_bytes + v_bytes
    }

    pub fn savings_ratio(&self, bits: u8) -> f32 {
        self.total_bytes() as f32 / self.compressed_bytes(bits) as f32
    }
}
