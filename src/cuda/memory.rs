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
/// Allocated with cudaMallocHost for DMA-capable access.
/// Used during model loading to overlap transfer with GPU compute.
pub struct PinnedBuffer {
    pub data: Vec<u8>,
    pub size: usize,
    /// Whether this buffer is actually pinned (requires CUDA runtime).
    pub is_pinned: bool,
}

impl PinnedBuffer {
    /// Allocate a pinned buffer (falls back to regular malloc without CUDA).
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            size,
            is_pinned: false, // TODO Phase 4: cudaMallocHost
        }
    }

    pub fn as_slice(&self) -> &[u8] { &self.data[..self.size] }
    pub fn as_mut_slice(&mut self) -> &mut [u8] { &mut self.data[..self.size] }
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
