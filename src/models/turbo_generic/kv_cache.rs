//! TurboQuant KV Cache: compression, GPU buffers, config helpers, RoPE, TriAttention.

use crate::backend::ComputeBackend;
use crate::cuda::{TqTensor as Tensor, TqDevice as Device, TqDType as DType, TqError};
use crate::cuda::Result;
use crate::qmatmul as qmm;
use tq_kv::TurboQuantConfig;

// ============================================================
// TurboQuant KV Cache — Incremental + Fused Attention
// ============================================================

/// Number of initial "sink" tokens whose keys are kept in FP16 (uncompressed).
/// Attention sink tokens receive disproportionate attention weight — quantizing them
/// causes up to 81% of total attention error. (KVSink, arXiv:2508.04257)
/// Override with TQ_SINK env var.
pub(crate) const TQ_SINK_TOKENS: usize = 4;

pub(crate) fn get_sink_tokens(config: &tq_kv::TurboQuantConfig) -> usize {
    if let Some(sink) = config.sink_tokens {
        return sink;
    }
    std::env::var("TQ_SINK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TQ_SINK_TOKENS)
}

/// Compressed value store — either 8-bit per-vector or 4-bit per-group absmax.
#[derive(Clone, Debug)]
pub(crate) enum CompressedValueStore {
    Bits8(Vec<tq_kv::CompressedValues>),
    Bits4(Vec<tq_kv::CompressedValues4Bit>),
}

/// Per-head compacted KV cache segment: selected keys + synthetic values + beta biases.
#[derive(Debug, Clone)]
pub(crate) struct CompactedCacheHead {
    /// Compacted keys [t * head_dim] (f32, post-RoPE)
    pub(crate) keys: Vec<f32>,
    /// Per-key attention bias (log scale, added to logits before softmax) [t]
    pub(crate) beta: Vec<f32>,
    /// Synthetic values [t * head_dim] (f32)
    pub(crate) values: Vec<f32>,
    /// Number of compacted tokens
    pub(crate) t: usize,
    pub(crate) head_dim: usize,
}

/// Persistent GPU buffers for compressed KV data.
/// Pre-allocated to max_seq, incrementally appended at compression time.
/// Eliminates per-step CPU→GPU upload overhead for fused attention.
#[cfg(feature = "cuda")]
pub(crate) struct GpuCompressedKv {
    /// Packed indices: [n_kv_head, max_seq, bytes_per_key] (flat)
    pub(crate) packed_indices: cudarc::driver::CudaSlice<u8>,
    /// Per-vector norms: [n_kv_head, max_seq] — legacy per-vector path.
    /// Always allocated so the existing fused_attention/decompress kernels keep
    /// working. When the grouped path takes over (Sprint 1D) this can be
    /// retired.
    pub(crate) norms: cudarc::driver::CudaSlice<f32>,
    /// Per-group norms: [n_kv_head, max_seq, n_groups]. Allocated only when
    /// `n_groups > 1` (i.e. grouped compression is enabled). The grouped
    /// compress kernel writes here; grouped attention/decompress kernels read
    /// here. Sprint 1A: structure exists but isn't wired into the active path
    /// until Sprint 1C flips on `TQ_GPU_GROUPED`.
    pub(crate) gnorms: Option<cudarc::driver::CudaSlice<f32>>,
    /// Codebook centroids: [n_centroids] — uploaded once
    pub(crate) centroids: cudarc::driver::CudaSlice<f32>,
    /// V data (f32): [n_kv_head, max_seq, head_dim]
    pub(crate) v_data: cudarc::driver::CudaSlice<f32>,
    /// Scratch buffers (reused across calls)
    pub(crate) rotated_q: cudarc::driver::CudaSlice<f32>,  // [n_heads, head_dim]
    pub(crate) output_buf: cudarc::driver::CudaSlice<f32>,  // [n_heads, head_dim]
    /// Decompressed key cache: [n_kv_head, max_seq, head_dim] (strided)
    /// Incrementally updated — only new keys decompressed each step.
    pub(crate) decomp_cache: cudarc::driver::CudaSlice<f32>,
    /// How many keys in decomp_cache are valid (matches count after initial fill)
    pub(crate) decomp_count: usize,
    /// Pre-allocated contiguous K gather buffer: avoids per-token alloc in narrow()
    pub(crate) k_contig: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Pre-allocated contiguous V gather buffer: avoids per-token alloc in narrow()
    pub(crate) v_contig: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Current count of compressed keys per head
    pub(crate) count: usize,
    pub(crate) max_seq: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) n_heads: usize,
    pub(crate) head_dim: usize,
    /// Per-token group count for the grouped path. 1 means per-vector (legacy).
    pub(crate) n_groups: usize,
    /// Group size in dims. 0 means per-vector mode (n_groups == 1).
    pub(crate) group_size: usize,
    pub(crate) bytes_per_key: usize,
    pub(crate) bits: u8,
    pub(crate) stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuCompressedKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuCompressedKv").field("count", &self.count).field("max_seq", &self.max_seq).finish()
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuCompressedKv {
    fn clone(&self) -> Self { panic!("GpuCompressedKv cannot be cloned") }
}

#[cfg(feature = "cuda")]
impl GpuCompressedKv {
    pub(crate) fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        n_kv_head: usize,
        n_heads: usize,
        head_dim: usize,
        bits: u8,
        max_seq: usize,
        centroids: &[f32],
        group_size: usize,
    ) -> std::result::Result<Self, crate::cuda::TqError> {
        let bytes_per_key = (head_dim * bits as usize + 7) / 8;
        let total_indices = n_kv_head * max_seq * bytes_per_key;
        let total_norms = n_kv_head * max_seq;
        let total_v = n_kv_head * max_seq * head_dim;

        // Sprint 1A: only allocate per-group norms when grouped mode is
        // requested (group_size > 0 and head_dim divides cleanly). Per-vector
        // norms are still allocated unconditionally so the existing kernels
        // keep working until the grouped path is wired in Sprint 1C/1D.
        let n_groups = if group_size > 0 && head_dim % group_size == 0 {
            head_dim / group_size
        } else {
            1
        };

        let _ = stream.context().check_err();
        let packed_indices = stream.alloc_zeros::<u8>(total_indices)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv indices: {}", e)))?;
        let norms = stream.alloc_zeros::<f32>(total_norms)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv norms: {}", e)))?;
        let gnorms = if n_groups > 1 {
            let total_gnorms = n_kv_head * max_seq * n_groups;
            Some(stream.alloc_zeros::<f32>(total_gnorms)
                .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv gnorms: {}", e)))?)
        } else {
            None
        };
        let v_data = stream.alloc_zeros::<f32>(total_v)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv v: {}", e)))?;
        let gpu_centroids = stream.clone_htod(centroids)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv centroids: {}", e)))?;
        let rotated_q = stream.alloc_zeros::<f32>(n_heads * head_dim)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv rq: {}", e)))?;
        let output_buf = stream.alloc_zeros::<f32>(n_heads * head_dim)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv out: {}", e)))?;
        let decomp_cache = stream.alloc_zeros::<f32>(total_v)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv decomp: {}", e)))?;
        let k_contig = std::sync::Arc::new(stream.alloc_zeros::<f32>(total_v)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv k_contig: {}", e)))?);
        let v_contig = std::sync::Arc::new(stream.alloc_zeros::<f32>(total_v)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuCompressedKv v_contig: {}", e)))?);
        let gnorm_bytes = n_groups.saturating_sub(1) * n_kv_head * max_seq * 4;
        eprintln!("[gpu-tq] pre-allocated {}×{}×{} = {:.1}MB compressed + {:.1}MB decomp cache (grouped: n_groups={}, +{:.2}MB gnorms)",
            n_kv_head, max_seq, bytes_per_key,
            (total_indices + total_norms * 4 + total_v * 4) as f64 / 1e6,
            (total_v * 4) as f64 / 1e6,
            n_groups,
            gnorm_bytes as f64 / 1e6);

        Ok(Self {
            packed_indices, norms, gnorms, centroids: gpu_centroids, v_data,
            rotated_q, output_buf, decomp_cache, decomp_count: 0,
            k_contig, v_contig,
            count: 0, max_seq, n_kv_head, n_heads, head_dim,
            n_groups, group_size,
            bytes_per_key, bits,
            stream,
        })
    }

    /// Append one compressed token's data for all heads.
    /// packed: [n_kv_head * bytes_per_key], head_norms: [n_kv_head], v_flat: [n_kv_head * head_dim]
    pub(crate) fn append(
        &mut self,
        packed: &[u8],
        head_norms: &[f32],
        v_flat: &[f32],
    ) -> std::result::Result<(), crate::cuda::TqError> {
        if self.count >= self.max_seq {
            pub(crate) static WARN_CPU: std::sync::Once = std::sync::Once::new();
            WARN_CPU.call_once(|| eprintln!("[WARN] GpuCompressedKv: max_seq={} reached, TQ context truncated", self.max_seq));
            return Ok(());
        }
        let bpk = self.bytes_per_key;
        let pos = self.count;

        // Upload packed indices per head at correct offset
        for h in 0..self.n_kv_head {
            let src = &packed[h * bpk..(h + 1) * bpk];
            let dst_off = (h * self.max_seq + pos) * bpk;
            let _ = self.stream.memcpy_htod(src, &mut self.packed_indices.slice_mut(dst_off..dst_off + bpk));
        }

        // Upload norms per head
        for h in 0..self.n_kv_head {
            let dst_off = h * self.max_seq + pos;
            let _ = self.stream.memcpy_htod(&head_norms[h..h+1], &mut self.norms.slice_mut(dst_off..dst_off + 1));
        }

        // Upload V data per head
        let hd = self.head_dim;
        for h in 0..self.n_kv_head {
            let src = &v_flat[h * hd..(h + 1) * hd];
            let dst_off = (h * self.max_seq + pos) * hd;
            let _ = self.stream.memcpy_htod(src, &mut self.v_data.slice_mut(dst_off..dst_off + hd));
        }

        self.count += 1;
        Ok(())
    }

    /// Append one token's compressed K and raw V directly from GPU buffers.
    /// Pure D2D scatter — no CPU round-trip. Uses cuMemcpyDtoDAsync per head.
    /// packed_gpu: [n_kv_head * bytes_per_key], norms_gpu: [n_kv_head], v_gpu: [n_kv_head * head_dim]
    pub(crate) fn append_gpu(
        &mut self,
        packed_gpu: &cudarc::driver::CudaSlice<u8>,
        norms_gpu: &cudarc::driver::CudaSlice<f32>,
        v_gpu: &cudarc::driver::CudaSlice<f32>,
    ) -> std::result::Result<(), crate::cuda::TqError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use cudarc::driver::sys;

        if self.count >= self.max_seq {
            pub(crate) static WARN_GPU: std::sync::Once = std::sync::Once::new();
            WARN_GPU.call_once(|| eprintln!("[WARN] GpuCompressedKv: max_seq={} reached, TQ context truncated (D2D)", self.max_seq));
            return Ok(());
        }
        let bpk = self.bytes_per_key;
        let pos = self.count;
        let hd = self.head_dim;
        let ms = self.max_seq;
        let n_kv = self.n_kv_head;
        let raw_stream = self.stream.cu_stream();

        // D2D scatter: packed indices (u8) per head
        {
            let (src_base, _g1) = packed_gpu.device_ptr(self.stream.as_ref());
            let (dst_base, _g2) = self.packed_indices.device_ptr_mut(self.stream.as_ref());
            for h in 0..n_kv {
                let src_off = (h * bpk) as u64;
                let dst_off = ((h * ms + pos) * bpk) as u64;
                let res = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(dst_base + dst_off, src_base + src_off, bpk as usize, raw_stream)
                };
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(crate::cuda::TqError::Msg(format!("d2d packed h={}: {:?}", h, res)));
                }
            }
        }

        // D2D scatter: norms (f32) per head
        {
            let (src_base, _g1) = norms_gpu.device_ptr(self.stream.as_ref());
            let (dst_base, _g2) = self.norms.device_ptr_mut(self.stream.as_ref());
            for h in 0..n_kv {
                let src_off = (h * 4) as u64;
                let dst_off = ((h * ms + pos) * 4) as u64;
                let res = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(dst_base + dst_off, src_base + src_off, 4, raw_stream)
                };
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(crate::cuda::TqError::Msg(format!("d2d norms h={}: {:?}", h, res)));
                }
            }
        }

        // D2D scatter: V data (f32) per head
        {
            let (src_base, _g1) = v_gpu.device_ptr(self.stream.as_ref());
            let (dst_base, _g2) = self.v_data.device_ptr_mut(self.stream.as_ref());
            for h in 0..n_kv {
                let src_off = (h * hd * 4) as u64;
                let dst_off = ((h * ms + pos) * hd * 4) as u64;
                let res = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(dst_base + dst_off, src_base + src_off, hd * 4, raw_stream)
                };
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(crate::cuda::TqError::Msg(format!("d2d v h={}: {:?}", h, res)));
                }
            }
        }

        self.count += 1;
        Ok(())
    }

    /// CPU-side append for the grouped norms storage. Mirror of append() but
    /// takes per-group norms (n_kv_head * n_groups floats) and writes them to
    /// self.gnorms via memcpy_htod. Used by the layer.rs seed loop and the
    /// CPU-compress→GPU-cache bridge when pre_rope_mode forces CPU compression.
    pub(crate) fn append_grouped(
        &mut self,
        packed: &[u8],
        group_norms: &[f32],   // [n_kv_head * n_groups]
        v_flat: &[f32],
    ) -> std::result::Result<(), crate::cuda::TqError> {
        if self.count >= self.max_seq {
            pub(crate) static WARN_CPU_G: std::sync::Once = std::sync::Once::new();
            WARN_CPU_G.call_once(|| eprintln!(
                "[WARN] GpuCompressedKv: max_seq={} reached, TQ context truncated (CPU grouped)",
                self.max_seq));
            return Ok(());
        }
        let gnorms_dst = match self.gnorms.as_mut() {
            Some(g) => g,
            None => return Err(crate::cuda::TqError::Msg(
                "append_grouped called but gnorms not allocated".to_string())),
        };
        let bpk = self.bytes_per_key;
        let pos = self.count;
        let hd = self.head_dim;
        let ng = self.n_groups;
        let ms = self.max_seq;

        // Packed indices per head
        for h in 0..self.n_kv_head {
            let src = &packed[h * bpk..(h + 1) * bpk];
            let dst_off = (h * ms + pos) * bpk;
            let _ = self.stream.memcpy_htod(src, &mut self.packed_indices.slice_mut(dst_off..dst_off + bpk));
        }

        // Per-group norms per head: write n_groups floats per (head, pos)
        for h in 0..self.n_kv_head {
            let src = &group_norms[h * ng..(h + 1) * ng];
            let dst_off = (h * ms + pos) * ng;
            let _ = self.stream.memcpy_htod(src, &mut gnorms_dst.slice_mut(dst_off..dst_off + ng));
        }

        // V data per head
        for h in 0..self.n_kv_head {
            let src = &v_flat[h * hd..(h + 1) * hd];
            let dst_off = (h * ms + pos) * hd;
            let _ = self.stream.memcpy_htod(src, &mut self.v_data.slice_mut(dst_off..dst_off + hd));
        }

        self.count += 1;
        Ok(())
    }

    /// Grouped variant of append_gpu (Sprint 1C). Same D2D scatter for
    /// packed indices and V, but writes per-group norms into self.gnorms
    /// instead of the per-vector self.norms. Caller is responsible for
    /// ensuring self.gnorms is allocated (i.e. n_groups > 1).
    ///
    /// gnorms_gpu layout: [n_kv_head, n_groups]
    pub(crate) fn append_gpu_grouped(
        &mut self,
        packed_gpu: &cudarc::driver::CudaSlice<u8>,
        gnorms_gpu: &cudarc::driver::CudaSlice<f32>,
        v_gpu: &cudarc::driver::CudaSlice<f32>,
    ) -> std::result::Result<(), crate::cuda::TqError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use cudarc::driver::sys;

        if self.count >= self.max_seq {
            pub(crate) static WARN_GPU_G: std::sync::Once = std::sync::Once::new();
            WARN_GPU_G.call_once(|| eprintln!(
                "[WARN] GpuCompressedKv: max_seq={} reached, TQ context truncated (D2D grouped)",
                self.max_seq));
            return Ok(());
        }
        let gnorms_dst = match self.gnorms.as_mut() {
            Some(g) => g,
            None => {
                return Err(crate::cuda::TqError::Msg(
                    "append_gpu_grouped called but gnorms not allocated (n_groups=1?)".to_string()));
            }
        };
        let bpk = self.bytes_per_key;
        let pos = self.count;
        let hd = self.head_dim;
        let ms = self.max_seq;
        let ng = self.n_groups;
        let n_kv = self.n_kv_head;
        let raw_stream = self.stream.cu_stream();

        // D2D scatter: packed indices (u8) per head
        {
            let (src_base, _g1) = packed_gpu.device_ptr(self.stream.as_ref());
            let (dst_base, _g2) = self.packed_indices.device_ptr_mut(self.stream.as_ref());
            for h in 0..n_kv {
                let src_off = (h * bpk) as u64;
                let dst_off = ((h * ms + pos) * bpk) as u64;
                let res = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(dst_base + dst_off, src_base + src_off, bpk as usize, raw_stream)
                };
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(crate::cuda::TqError::Msg(format!("d2d packed h={}: {:?}", h, res)));
                }
            }
        }

        // D2D scatter: per-group norms (f32 × n_groups) per head.
        // gnorms storage layout is [n_kv_head, max_seq, n_groups], so the
        // destination offset for (head h, token pos) is (h*max_seq + pos)*n_groups.
        {
            let (src_base, _g1) = gnorms_gpu.device_ptr(self.stream.as_ref());
            let (dst_base, _g2) = gnorms_dst.device_ptr_mut(self.stream.as_ref());
            let bytes_per_head = (ng * 4) as usize;
            for h in 0..n_kv {
                let src_off = (h * ng * 4) as u64;
                let dst_off = ((h * ms + pos) * ng * 4) as u64;
                let res = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(dst_base + dst_off, src_base + src_off, bytes_per_head, raw_stream)
                };
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(crate::cuda::TqError::Msg(format!("d2d gnorms h={}: {:?}", h, res)));
                }
            }
        }

        // D2D scatter: V data (f32) per head
        {
            let (src_base, _g1) = v_gpu.device_ptr(self.stream.as_ref());
            let (dst_base, _g2) = self.v_data.device_ptr_mut(self.stream.as_ref());
            for h in 0..n_kv {
                let src_off = (h * hd * 4) as u64;
                let dst_off = ((h * ms + pos) * hd * 4) as u64;
                let res = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(dst_base + dst_off, src_base + src_off, hd * 4, raw_stream)
                };
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(crate::cuda::TqError::Msg(format!("d2d v h={}: {:?}", h, res)));
                }
            }
        }

        self.count += 1;
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.count = 0;
        self.decomp_count = 0;
    }

    /// Compact GPU cache in-place: keep only `retained` indices (sorted ascending).
    /// Uses D2D copies to gather retained positions to buffer start.
    /// Much faster than full re-seed (copies only retained tokens, not all).
    pub(crate) fn compact(&mut self, retained: &[usize]) -> std::result::Result<(), crate::cuda::TqError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use cudarc::driver::sys;

        let n_retain = retained.len();
        if n_retain == 0 || n_retain >= self.count {
            return Ok(()); // nothing to do
        }

        let raw_stream = self.stream.cu_stream();
        let bpk = self.bytes_per_key;
        let hd = self.head_dim;
        let ms = self.max_seq;

        // For each head, gather retained positions to the front of the buffer.
        // We iterate in order (new_pos < old_pos for moved elements), so D2D
        // copies don't overlap destructively.
        // Use raw device_ptr for in-place D2D (same buffer src and dst).

        let packed_base = self.packed_indices.device_ptr(self.stream.as_ref()).0;
        let norms_base = self.norms.device_ptr(self.stream.as_ref()).0;
        let v_base = self.v_data.device_ptr(self.stream.as_ref()).0;
        // gnorms is optional — only present when grouped storage is allocated.
        let gnorms_base = self.gnorms.as_ref()
            .map(|g| g.device_ptr(self.stream.as_ref()).0);
        let ng = self.n_groups;

        for h in 0..self.n_kv_head {
            for (new_pos, &old_pos) in retained.iter().enumerate() {
                if new_pos == old_pos { continue; }
                // Packed indices
                let src = packed_base + ((h * ms + old_pos) * bpk) as u64;
                let dst = packed_base + ((h * ms + new_pos) * bpk) as u64;
                unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bpk, raw_stream); }

                // Norms (per-vector legacy)
                let src = norms_base + ((h * ms + old_pos) as u64) * 4;
                let dst = norms_base + ((h * ms + new_pos) as u64) * 4;
                unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, 4, raw_stream); }

                // Per-group norms (Sprint 1C). Layout: [n_kv, max_seq, n_groups]
                if let Some(base) = gnorms_base {
                    let src = base + ((h * ms + old_pos) * ng * 4) as u64;
                    let dst = base + ((h * ms + new_pos) * ng * 4) as u64;
                    unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, ng * 4, raw_stream); }
                }

                // V data
                let src = v_base + ((h * ms + old_pos) * hd) as u64 * 4;
                let dst = v_base + ((h * ms + new_pos) * hd) as u64 * 4;
                unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, hd * 4, raw_stream); }
            }
        }

        // Also compact decomp_cache if populated
        if self.decomp_count > 0 {
            let dc_base = self.decomp_cache.device_ptr(self.stream.as_ref()).0;
            for h in 0..self.n_kv_head {
                for (new_pos, &old_pos) in retained.iter().enumerate() {
                    if new_pos == old_pos || old_pos >= self.decomp_count { continue; }
                    let src = dc_base + ((h * ms + old_pos) * hd) as u64 * 4;
                    let dst = dc_base + ((h * ms + new_pos) * hd) as u64 * 4;
                    unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, hd * 4, raw_stream); }
                }
            }
            self.decomp_count = n_retain.min(self.decomp_count);
        }

        let _ = self.stream.synchronize();
        self.count = n_retain;
        Ok(())
    }
}

// ============================================================
// GPU Cold Tier — evicted tokens kept on GPU for fast retrieval
// ============================================================

/// GPU-resident storage for evicted compressed tokens (Sprint F).
///
/// Instead of downloading evicted tokens to CPU (slow PCIe round-trip),
/// we D2D-copy them to a separate "cold" GPU buffer. During attention,
/// `tq_fused_attention_grouped` scores cold tokens directly from their
/// compressed form — no decompress, no CPU involvement.
///
/// Cold tier grows with each eviction. When GPU memory is tight,
/// the oldest cold tokens can be spilled to the CPU OffloadedKvCache.
#[cfg(feature = "cuda")]
pub(crate) struct GpuColdKv {
    /// Post-RoPE decompressed keys: [n_kv_head, max_cold, head_dim].
    /// Stored as full f32 (not compressed) from decomp_cache at eviction time.
    /// No decompress or RoPE needed on retrieval — just read and cat.
    pub(crate) k_data: cudarc::driver::CudaSlice<f32>,
    /// Packed indices: [n_kv_head, max_cold, bytes_per_key] (kept for fused scoring)
    pub(crate) packed_indices: cudarc::driver::CudaSlice<u8>,
    /// Per-group norms: [n_kv_head, max_cold, n_groups]
    pub(crate) gnorms: cudarc::driver::CudaSlice<f32>,
    /// V data (f32): [n_kv_head, max_cold, head_dim]
    pub(crate) v_data: cudarc::driver::CudaSlice<f32>,
    /// Sequence positions for RoPE reconstruction: [max_cold] (CPU-side, cheap)
    pub(crate) seq_positions: Vec<usize>,
    /// Current count of cold tokens
    pub(crate) count: usize,
    pub(crate) max_cold: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) n_groups: usize,
    pub(crate) group_size: usize,
    pub(crate) bytes_per_key: usize,
    pub(crate) bits: u8,
    pub(crate) stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl GpuColdKv {
    pub(crate) fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        n_kv_head: usize,
        head_dim: usize,
        bits: u8,
        max_cold: usize,
        group_size: usize,
    ) -> std::result::Result<Self, crate::cuda::TqError> {
        let bytes_per_key = (head_dim * bits as usize + 7) / 8;
        let n_groups = if group_size > 0 && head_dim % group_size == 0 {
            head_dim / group_size
        } else { 1 };

        let k_data = stream.alloc_zeros::<f32>(n_kv_head * max_cold * head_dim)
            .map_err(|e| crate::cuda::TqError::Msg(format!("cold k_data: {}", e)))?;
        let packed = stream.alloc_zeros::<u8>(n_kv_head * max_cold * bytes_per_key)
            .map_err(|e| crate::cuda::TqError::Msg(format!("cold packed: {}", e)))?;
        let gnorms = stream.alloc_zeros::<f32>(n_kv_head * max_cold * n_groups)
            .map_err(|e| crate::cuda::TqError::Msg(format!("cold gnorms: {}", e)))?;
        let v_data = stream.alloc_zeros::<f32>(n_kv_head * max_cold * head_dim)
            .map_err(|e| crate::cuda::TqError::Msg(format!("cold v_data: {}", e)))?;

        let cold_mb = (n_kv_head * max_cold * head_dim * 4 * 2  // k_data + v_data
            + n_kv_head * max_cold * bytes_per_key
            + n_kv_head * max_cold * n_groups * 4) as f64 / (1024.0 * 1024.0);
        eprintln!("[gpu-cold] allocated {:.1}MB for {} cold slots (post-RoPE K + V + compressed)", cold_mb, max_cold);

        Ok(Self {
            k_data,
            packed_indices: packed,
            gnorms,
            v_data,
            seq_positions: Vec::with_capacity(max_cold),
            count: 0,
            max_cold,
            n_kv_head,
            head_dim,
            n_groups,
            group_size,
            bytes_per_key,
            bits,
            stream,
        })
    }

    /// D2D copy a single token from the hot GPU cache to cold tier.
    /// `hot_idx` is the position in the hot cache being evicted.
    pub(crate) fn push_from_hot(
        &mut self,
        hot: &GpuCompressedKv,
        hot_idx: usize,
        seq_pos: usize,
    ) -> std::result::Result<(), crate::cuda::TqError> {
        if self.count >= self.max_cold {
            // Cold tier full — oldest tokens could be spilled to CPU here.
            // For now, silently drop (circular buffer behavior).
            return Ok(());
        }
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use cudarc::driver::sys;

        let cold_pos = self.count;
        let bpk = self.bytes_per_key;
        let hd = self.head_dim;
        let ng = self.n_groups;
        let ms_hot = hot.max_seq;
        let ms_cold = self.max_cold;
        let raw_stream = self.stream.cu_stream();

        // D2D: packed indices per head
        {
            let (src_base, _) = hot.packed_indices.device_ptr(self.stream.as_ref());
            let (dst_base, _) = self.packed_indices.device_ptr(self.stream.as_ref());
            for h in 0..self.n_kv_head {
                let src = src_base + ((h * ms_hot + hot_idx) * bpk) as u64;
                let dst = dst_base + ((h * ms_cold + cold_pos) * bpk) as u64;
                unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bpk, raw_stream); }
            }
        }
        // D2D: gnorms per head
        {
            let (src_base, _) = hot.gnorms.as_ref()
                .ok_or_else(|| crate::cuda::TqError::Msg("hot gnorms missing".into()))?
                .device_ptr(self.stream.as_ref());
            let (dst_base, _) = self.gnorms.device_ptr(self.stream.as_ref());
            for h in 0..self.n_kv_head {
                let src = src_base + ((h * ms_hot + hot_idx) * ng * 4) as u64;
                let dst = dst_base + ((h * ms_cold + cold_pos) * ng * 4) as u64;
                unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, ng * 4, raw_stream); }
            }
        }
        // D2D: V data per head
        {
            let (src_base, _) = hot.v_data.device_ptr(self.stream.as_ref());
            let (dst_base, _) = self.v_data.device_ptr(self.stream.as_ref());
            for h in 0..self.n_kv_head {
                let src = src_base + ((h * ms_hot + hot_idx) * hd * 4) as u64;
                let dst = dst_base + ((h * ms_cold + cold_pos) * hd * 4) as u64;
                unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, hd * 4, raw_stream); }
            }
        }

        self.seq_positions.push(seq_pos);
        self.count += 1;
        Ok(())
    }

    /// D2D copy a post-RoPE decompressed key from hot decomp_cache to cold k_data.
    /// `hot_decomp` is `[n_kv_head, max_seq_hot, head_dim]` strided buffer.
    /// This stores the DECOMPRESSED post-RoPE K so retrieval needs zero work.
    pub(crate) fn push_decomp_key(
        &mut self,
        hot_decomp: &cudarc::driver::CudaSlice<f32>,
        hot_idx: usize,
        max_seq_hot: usize,
    ) -> std::result::Result<(), crate::cuda::TqError> {
        if self.count == 0 { return Ok(()); } // push_from_hot must be called first
        let cold_pos = self.count - 1; // last pushed position
        let hd = self.head_dim;
        let mc = self.max_cold;
        let raw_stream = self.stream.cu_stream();

        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use cudarc::driver::sys;

        let (src_base, _) = hot_decomp.device_ptr(self.stream.as_ref());
        let (dst_base, _) = self.k_data.device_ptr(self.stream.as_ref());
        for h in 0..self.n_kv_head {
            let src = src_base + ((h * max_seq_hot + hot_idx) * hd * 4) as u64;
            let dst = dst_base + ((h * mc + cold_pos) * hd * 4) as u64;
            unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, hd * 4, raw_stream); }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuColdKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuColdKv")
            .field("count", &self.count)
            .field("max_cold", &self.max_cold)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuColdKv {
    fn clone(&self) -> Self {
        panic!("GpuColdKv cannot be cloned (GPU buffers are not trivially copyable)")
    }
}

/// GPU-accelerated TriAttention scoring + selection.
/// Uploads pre-RoPE keys to GPU, runs trig_score_keys_batched kernel,
/// downloads scores, selects top-B on CPU.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_tri_score_and_select(
    reg: &crate::cuda::kernels::KernelRegistry,
    tri_keys: &[Vec<f32>],      // per KV head: [n_keys * head_dim]
    key_positions: &[usize],
    current_pos: usize,
    sink_count: usize,
    config: &tq_kv::triattention::TriAttentionConfig,
) -> std::result::Result<Vec<Vec<usize>>, crate::cuda::TqError> {
    let n_kv_heads = tri_keys.len();
    let n_keys = key_positions.len();
    let head_dim = config.head_dim;
    // Use dynamic budget from retention_ratio (same as evict_pass in triattention.rs).
    // The static config.budget (default 2048) is a hard ceiling; retention_ratio (0.9)
    // gives the per-eviction target. Without this, tokens are never actually evicted
    // because budget=2048 > n_keys at typical context lengths.
    let budget = if config.retention_ratio > 0.0 && config.retention_ratio < 1.0 {
        (n_keys as f32 * config.retention_ratio) as usize
    } else {
        config.budget
    };
    if n_keys == 0 || n_kv_heads == 0 {
        return Ok(vec![(0..n_keys).collect(); n_kv_heads]);
    }

    // Map Q centers to KV heads: for each KV head, use first mapped Q head's center
    let n_rep = config.n_heads / config.n_kv_heads;
    let mut q_centers_mapped: Vec<f32> = Vec::with_capacity(n_kv_heads * head_dim);
    let mut mrl_mapped: Vec<f32> = Vec::with_capacity(n_kv_heads);
    let mut qnorm_mapped: Vec<f32> = Vec::with_capacity(n_kv_heads);
    for kv_h in 0..n_kv_heads {
        let q_h = kv_h * n_rep; // first Q head for this KV head
        q_centers_mapped.extend_from_slice(&config.q_centers[q_h]);
        mrl_mapped.push(config.mrl[q_h]);
        qnorm_mapped.push(config.q_norm_means[q_h]);
    }

    // Flatten keys: [n_kv_heads, n_keys, head_dim]
    // Only score non-sink keys
    let non_sink_n = n_keys.saturating_sub(sink_count);
    let mut keys_flat: Vec<f32> = Vec::with_capacity(n_kv_heads * non_sink_n * head_dim);
    for h in 0..n_kv_heads {
        let start = sink_count * head_dim;
        if start < tri_keys[h].len() {
            keys_flat.extend_from_slice(&tri_keys[h][start..]);
        }
        // Pad if needed
        let expected = non_sink_n * head_dim;
        let actual = keys_flat.len() - h * expected;
        if actual < expected {
            keys_flat.extend(std::iter::repeat(0.0f32).take(expected - actual));
        }
    }

    let positions_i32: Vec<i32> = key_positions[sink_count..].iter()
        .map(|&p| p as i32).collect();
    let offsets_i32: Vec<i32> = config.offsets.iter().map(|&o| o as i32).collect();

    // Upload to GPU
    let gpu_q_centers = reg.stream.clone_htod(&q_centers_mapped)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu q_centers: {}", e)))?;
    let gpu_keys = reg.stream.clone_htod(&keys_flat)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu keys: {}", e)))?;
    let gpu_freqs = reg.stream.clone_htod(&config.rope_freqs)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu freqs: {}", e)))?;
    let gpu_positions = reg.stream.clone_htod(&positions_i32)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu positions: {}", e)))?;
    let gpu_mrl = reg.stream.clone_htod(&mrl_mapped)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu mrl: {}", e)))?;
    let gpu_qnorm = reg.stream.clone_htod(&qnorm_mapped)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu qnorm: {}", e)))?;
    let gpu_offsets = reg.stream.clone_htod(&offsets_i32)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu offsets: {}", e)))?;
    let mut gpu_scores = reg.stream.alloc_zeros::<f32>(n_kv_heads * non_sink_n)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu scores: {}", e)))?;

    // Launch kernel
    crate::cuda::kernels::trig_score_keys_batched(
        reg,
        &gpu_q_centers, &gpu_keys, &gpu_freqs, &gpu_positions,
        &mut gpu_scores, &gpu_mrl, &gpu_qnorm,
        current_pos, non_sink_n, n_kv_heads, head_dim,
        &gpu_offsets, config.offsets.len(),
    ).map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu kernel: {:?}", e)))?;

    // Download scores
    let scores_flat: Vec<f32> = reg.stream.clone_dtoh(&gpu_scores)
        .map_err(|e| crate::cuda::TqError::Msg(format!("tri-gpu dtoh: {}", e)))?;

    // Select top-B per head (on CPU)
    let non_sink_budget = budget.saturating_sub(sink_count);
    let mut result = Vec::with_capacity(n_kv_heads);
    for kv_h in 0..n_kv_heads {
        let head_scores = &scores_flat[kv_h * non_sink_n..(kv_h + 1) * non_sink_n];
        let selected = tq_kv::triattention::select_top_keys(head_scores, non_sink_budget);
        // Shift back to global indices + prepend sinks
        let mut full: Vec<usize> = (0..sink_count).collect();
        full.extend(selected.iter().map(|&i| i + sink_count));
        result.push(full);
    }

    Ok(result)
}

#[derive(Clone, Debug)]
pub(crate) struct CompressedKvCache {
    /// Compressed keys per KV head (hot tier, recent tokens at original bit width)
    pub(crate) k_per_head: Vec<tq_kv::CompressedKeys>,
    /// Cold (decayed) keys per head — lower bit width, older tokens.
    /// Order: cold tokens come BEFORE hot tokens in sequence position.
    pub(crate) k_cold: Option<Vec<tq_kv::CompressedKeys>>,
    /// How many tokens are in the cold tier (same across all heads)
    pub(crate) cold_len: usize,
    /// Temporal decay config (None = disabled)
    pub(crate) decay_config: Option<tq_kv::TemporalDecayConfig>,
    /// Tokens since last decay check
    pub(crate) tokens_since_decay: usize,
    /// Uncompressed sink token keys: [1, n_kv_head, sink_len, head_dim]
    pub(crate) sink_k: Option<Tensor>,
    /// Number of sink tokens stored
    pub(crate) sink_len: usize,
    /// Compacted cache: attention-matching token reduction (Zweiger 2026).
    /// Sits between cold and hot: [sink | cold | compacted | hot | current]
    pub(crate) compacted: Option<Vec<CompactedCacheHead>>,
    /// Number of original tokens that were compacted away
    pub(crate) compacted_original_len: usize,
    /// Value cache — uncompressed path (value_bits=0)
    pub(crate) v_raw: Option<Tensor>,
    /// Value cache — compressed path (value_bits=4 or 8), per KV head
    pub(crate) v_compressed: Option<CompressedValueStore>,
    /// Value quantization bits (0=fp16, 4=4-bit per-group, 8=8-bit absmax)
    pub(crate) value_bits: u8,
    /// Total cached length (sink + compressed + current)
    pub(crate) cached_len: usize,
    pub(crate) dtype: DType,
    /// Pre-RoPE mode: compressed keys are stored BEFORE RoPE application.
    /// At decode time, keys must be decompressed and RoPE applied dynamically.
    pub(crate) pre_rope: bool,
    // GPU fused attention is triggered dynamically (no persistent state needed).

    // -- TriAttention eviction state --
    pub(crate) tri_config: Option<tq_kv::triattention::TriAttentionConfig>,
    pub(crate) tri_keys_pre_rope: Option<Vec<Vec<f32>>>,
    pub(crate) tri_key_positions: Vec<usize>,
    pub(crate) tri_tokens_since_eviction: usize,
}

/// Decompress compressed keys to tensor. Only decompresses the compressed portion.
pub(crate) fn decompress_compressed_keys(
    k_per_head: &[tq_kv::CompressedKeys],
    n_kv_head: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
    config: &TurboQuantConfig,
) -> Result<Tensor> {
    let compressed_len = if k_per_head.is_empty() || k_per_head[0].count == 0 {
        return Tensor::zeros(vec![1, n_kv_head, 0, head_dim], device);
    } else {
        k_per_head[0].count
    };
    let mut all_data = Vec::with_capacity(n_kv_head * compressed_len * head_dim);
    for compressed in k_per_head.iter().take(n_kv_head) {
        let decompressed = if compressed.group_size > 0 {
            tq_kv::decompress_keys_grouped(compressed, config)
        } else {
            tq_kv::decompress_keys(compressed, config)
        };
        all_data.extend(decompressed);
    }
    Tensor::from_vec(all_data, vec![1, n_kv_head, compressed_len, head_dim], device)?.to_dtype(dtype)
}

/// Decompress pre-RoPE compressed keys and apply RoPE dynamically.
/// `start_pos` is the sequence position of the first compressed key.
/// Returns post-RoPE keys ready for attention computation.
pub(crate) fn decompress_and_apply_rope(
    k_per_head: &[tq_kv::CompressedKeys],
    n_kv_head: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
    config: &TurboQuantConfig,
    cos: &Tensor,
    sin: &Tensor,
    start_pos: usize,
    rope_style: RopeStyle,
    rope_dim: usize,
) -> Result<Tensor> {
    // First decompress to get pre-RoPE keys
    let pre_rope = decompress_compressed_keys(k_per_head, n_kv_head, head_dim, DType::F32, device, config)?;
    let compressed_len = if k_per_head.is_empty() || k_per_head[0].count == 0 {
        return Tensor::zeros(vec![1, n_kv_head, 0, head_dim], device);
    } else {
        k_per_head[0].count
    };

    // Apply RoPE: slice cos/sin for the correct positions
    let cos_slice = cos.narrow(0, start_pos, compressed_len)?;
    let sin_slice = sin.narrow(0, start_pos, compressed_len)?;

    let rotated = if rope_dim < head_dim {
        let x_rope = pre_rope.narrow(3, 0, rope_dim)?;
        let x_pass = pre_rope.narrow(3, rope_dim, head_dim - rope_dim)?;
        let x_rotated = match rope_style {
            RopeStyle::Halved => rope_halved(&x_rope, &cos_slice, &sin_slice)?,
            RopeStyle::Interleaved => rope_interleaved(&x_rope, &cos_slice, &sin_slice)?,
        };
        Tensor::cat(&[&x_rotated, &x_pass], 3)?
    } else {
        match rope_style {
            RopeStyle::Halved => rope_halved(&pre_rope, &cos_slice, &sin_slice)?,
            RopeStyle::Interleaved => rope_interleaved(&pre_rope, &cos_slice, &sin_slice)?,
        }
    };
    rotated.to_dtype(dtype)
}

/// Decompress compressed values to F32 tensor: (1, n_kv_head, seq_len, head_dim).
pub(crate) fn decompress_values_store(
    store: &CompressedValueStore,
    n_kv_head: usize,
    head_dim: usize,
    seq_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut all_data = Vec::with_capacity(n_kv_head * seq_len * head_dim);
    match store {
        CompressedValueStore::Bits8(v_per_head) => {
            for compressed in v_per_head.iter().take(n_kv_head) {
                all_data.extend(compressed.decompress());
            }
        }
        CompressedValueStore::Bits4(v_per_head) => {
            for compressed in v_per_head.iter().take(n_kv_head) {
                all_data.extend(compressed.decompress());
            }
        }
    }
    Tensor::from_vec(all_data, vec![1, n_kv_head, seq_len, head_dim], device)
}

/// Sparse attention-value multiply on CPU tensors.
///
/// att shape: (1, n_heads, 1, seq_len)  — softmax weights for single query token
/// v shape:   (1, n_heads, seq_len, head_dim) — value cache
///
/// Returns: (1, n_heads, 1, head_dim) — same as att.matmul(&v) but skipping
/// V rows where the softmax weight < threshold.
pub(crate) fn sparse_attn_v(
    att: &Tensor,
    v: &Tensor,
    n_heads: usize,
    head_dim: usize,
    threshold: f32,
) -> Result<Tensor> {
    let att_flat = att.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let v_flat = v.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()?;
    let seq_len = att.dim(3)?;

    let mut output = Vec::with_capacity(n_heads * head_dim);
    for h in 0..n_heads {
        let att_row = &att_flat[h * seq_len..(h + 1) * seq_len];
        let v_block = &v_flat[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let head_out = tq_kv::sparse_attn_v_mul(att_row, v_block, head_dim, threshold);
        output.extend_from_slice(&head_out);
    }

    Tensor::from_vec(output, vec![1, n_heads, 1, head_dim], att.device())
}

// ============================================================
// RoPE variants
// ============================================================

/// Halved RoPE: first half / second half layout (Qwen2, most modern models).
pub(crate) fn rope_halved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = x.dims4()?;
    let half = d / 2;
    let x0 = x.narrow(3, 0, half)?;
    let x1 = x.narrow(3, half, half)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&r0, &r1], 3)
}

/// Interleaved RoPE: pairs (x0,x1), (x2,x3), ... layout (Llama).
pub(crate) fn rope_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let half = d / 2;
    let x = x.reshape(vec![b, h, s, half, 2])?;
    let x0 = x.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = x.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let r0 = r0.unsqueeze(4)?;
    let r1 = r1.unsqueeze(4)?;
    Tensor::cat(&[&r0, &r1], 4)?.reshape(vec![b, h, s, d])
}

/// Which RoPE layout to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RopeStyle {
    /// First-half / second-half (Qwen2, Mistral, Gemma, most modern models)
    Halved,
    /// Interleaved pairs (Llama)
    Interleaved,
}

// ============================================================
// Layer with TurboQuant KV cache
// ============================================================

/// Number of initial layers to keep uncompressed (fp16 KV cache).
/// Reduces error accumulation in deep models where early-layer errors
/// propagate through all subsequent layers.
/// Override with TQ_SKIP env var (e.g. TQ_SKIP=8 for first 8 layers uncompressed).
pub(crate) const TQ_SKIP_FIRST_LAYERS: usize = 4;

pub(crate) fn get_skip_layers(config: &tq_kv::TurboQuantConfig) -> usize {
    if let Some(skip) = config.skip_layers {
        return skip;
    }
    pub(crate) static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TQ_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(TQ_SKIP_FIRST_LAYERS)
    })
}

/// Number of final layers to keep uncompressed (fp16 KV cache).
/// Last layers are disproportionately sensitive to quantization — they feed
/// directly into the LM head and small KV errors cause large logit errors.
/// Boundary protection (last N layers uncompressed) breaks the compound error
/// chain and recovers generation quality at modest compression cost.
///
/// Bug #2 investigation (2026-04-10): combined with Pre-RoPE (default ON),
/// protect_last=8 gives 4-5/5 acceptable on the validation prompt suite while
/// keeping high compression ratio.
///
/// Validation results (Pre-RoPE ON):
///   protect=0:  3/5 acceptable
///   protect=8:  4/5 acceptable — production default
///   protect=16: 5/5 acceptable (very conservative)
///
/// Default 8: balances generation quality and compression ratio.
/// Combined with TQ_PRE_ROPE=1 (default), 20 layers compressed for Qwen2.5-7B.
/// See docs/BUG2-FINDINGS.md for full investigation.
/// Override with TQ_PROTECT_LAST=0 to disable (research/ablation).
pub(crate) fn get_protect_last_layers(config: &tq_kv::TurboQuantConfig) -> usize {
    if let Some(n) = config.protect_last_layers {
        return n;
    }
    pub(crate) static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        // Sprint 2: reduced from 8 to 2 based on per-layer sensitivity calibration.
        // All 28 layers have cos_sim ≥ 0.9954 at 4-bit group_size=32. The compound
        // error from compressing ALL layers (protect=0) regresses quality, but
        // protecting just the last 2 (LM-head-adjacent) layers preserves 4/5 on
        // the 5-prompt suite while compressing 22/28 layers (was 16/28 with protect=8).
        std::env::var("TQ_PROTECT_LAST").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
    })
}

/// Parsed layer bit ranges, cached at first use.
pub(crate) static LAYER_BITS_CACHE: std::sync::OnceLock<Vec<(usize, usize, u8)>> = std::sync::OnceLock::new();

pub(crate) fn parse_layer_bits() -> &'static Vec<(usize, usize, u8)> {
    LAYER_BITS_CACHE.get_or_init(|| {
        let mut ranges = Vec::new();
        if let Ok(val) = std::env::var("TQ_LAYER_BITS") {
            for part in val.split(',') {
                let parts: Vec<&str> = part.trim().split(':').collect();
                if parts.len() == 2 {
                    let range_parts: Vec<&str> = parts[0].split('-').collect();
                    if range_parts.len() == 2 {
                        if let (Ok(start), Ok(end), Ok(bits)) = (
                            range_parts[0].parse::<usize>(),
                            range_parts[1].parse::<usize>(),
                            parts[1].parse::<u8>(),
                        ) {
                            ranges.push((start, end, bits));
                        }
                    }
                }
            }
        }
        ranges
    })
}

/// H1 Hybrid: TQ_HYBRID_THRESHOLD=N — below N total KV tokens, run all
/// normally-compressed layers as uncompressed (Std path). Above, normal TQ.
/// Default 0 = hybrid disabled (existing behavior, pure TQ as configured).
/// Typical value: 1024 (short-context ≤ 1024 tokens stays Std-fast).
pub(crate) fn get_hybrid_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TQ_HYBRID_THRESHOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    })
}

/// Calibrated per-layer bits (Sprint 3). Set from CalibrationData.auto_layer_bits
/// at model load time. 0 = uncompressed (skip/protect), 2/3/4 = bit width.
pub(crate) static AUTO_LAYER_BITS: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

/// Set auto_layer_bits from calibration data. Called once at model load.
pub(crate) fn set_auto_layer_bits(bits: Vec<u8>) {
    let _ = AUTO_LAYER_BITS.set(bits);
}

/// Layer-adaptive bitwidth: assign different bit widths to different layer ranges.
/// Format: "start-end:bits[,start-end:bits]" e.g. "4-15:2,16-27:4"
/// Unspecified layers use the default TQ bits. Layers below TQ_SKIP or within
/// TQ_PROTECT_LAST of the final layer are uncompressed (fp16).
/// Override with TQ_LAYER_BITS env var.
pub(crate) fn get_layer_bits(layer_idx: usize, default_bits: u8, config: &tq_kv::TurboQuantConfig, n_layers: usize) -> Option<u8> {
    get_layer_bits_at(layer_idx, default_bits, config, n_layers, usize::MAX)
}

/// H1 variant: considers `total_kv_len` for hybrid threshold. When hybrid
/// threshold is set AND total_kv_len is below it, returns None for layers
/// that would otherwise be compressed (full Std regime until threshold).
/// Pass usize::MAX to disable the hybrid check (existing callers).
pub(crate) fn get_layer_bits_at(
    layer_idx: usize,
    default_bits: u8,
    config: &tq_kv::TurboQuantConfig,
    n_layers: usize,
    total_kv_len: usize,
) -> Option<u8> {
    let skip = get_skip_layers(config);
    if layer_idx < skip {
        return None; // uncompressed — boundary protection (first N)
    }

    let protect_last = get_protect_last_layers(config);
    if protect_last > 0 && n_layers > 0 && layer_idx >= n_layers - protect_last {
        return None; // uncompressed — boundary protection (last M)
    }

    // H1 hybrid: below threshold, non-boundary layers also stay uncompressed.
    // Boundary (skip/protect) layers already returned None above.
    let hybrid_threshold = get_hybrid_threshold();
    if hybrid_threshold > 0 && total_kv_len < hybrid_threshold {
        return None;
    }

    // Priority 1: explicit TQ_LAYER_BITS env var (manual override)
    let ranges = parse_layer_bits();
    for &(start, end, bits) in ranges {
        if layer_idx >= start && layer_idx <= end {
            return Some(bits);
        }
    }

    // Priority 2: calibrated auto_layer_bits (Sprint 3)
    if let Some(auto_bits) = AUTO_LAYER_BITS.get() {
        if layer_idx < auto_bits.len() && auto_bits[layer_idx] > 0 {
            return Some(auto_bits[layer_idx]);
        }
    }

    Some(default_bits)
}

/// Parsed head bit ranges, cached at first use.
pub(crate) static HEAD_BITS_CACHE: std::sync::OnceLock<Vec<(usize, usize, u8)>> = std::sync::OnceLock::new();

/// Parse TQ_HEAD_BITS env var. Format: "0-3:4,4-7:2" (same syntax as TQ_LAYER_BITS).
pub(crate) fn parse_head_bits() -> &'static Vec<(usize, usize, u8)> {
    HEAD_BITS_CACHE.get_or_init(|| {
        let mut ranges = Vec::new();
        if let Ok(val) = std::env::var("TQ_HEAD_BITS") {
            for part in val.split(',') {
                let parts: Vec<&str> = part.trim().split(':').collect();
                if parts.len() == 2 {
                    let range_parts: Vec<&str> = parts[0].split('-').collect();
                    if range_parts.len() == 2 {
                        if let (Ok(start), Ok(end), Ok(bits)) = (
                            range_parts[0].parse::<usize>(),
                            range_parts[1].parse::<usize>(),
                            parts[1].parse::<u8>(),
                        ) {
                            ranges.push((start, end, bits));
                        }
                    }
                }
            }
        }
        ranges
    })
}

/// Resolve per-head bit widths from TQ_HEAD_BITS env var.
/// Returns None if no TQ_HEAD_BITS is set (all heads use default_bits).
pub(crate) fn resolve_per_head_bits(n_kv_head: usize, default_bits: u8) -> Option<Vec<u8>> {
    let ranges = parse_head_bits();
    if ranges.is_empty() {
        return None;
    }
    let mut bits = vec![default_bits; n_kv_head];
    for &(start, end, b) in ranges {
        for h in start..=end.min(n_kv_head.saturating_sub(1)) {
            bits[h] = b;
        }
    }
    Some(bits)
}

/// Sparse V threshold. Softmax weights below this are skipped in V multiply.
/// Set TQ_SPARSE_V=0 to disable. Default: 1e-6.
/// Override with TQ_SPARSE_V env var (e.g. TQ_SPARSE_V=1e-5).
pub(crate) const TQ_SPARSE_V_DEFAULT: f32 = 1e-6;

pub(crate) fn get_sparse_v_threshold() -> f32 {
    pub(crate) static C: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_SPARSE_V").ok().and_then(|v| v.parse().ok()).unwrap_or(TQ_SPARSE_V_DEFAULT))
}

/// Fused attention: compute attention scores directly from compressed indices
/// instead of decompressing keys first. Saves memory bandwidth on CPU.
/// Set TQ_FUSED=1 to enable. Default: off (decompress path).
pub(crate) fn get_use_fused() -> bool {
    pub(crate) static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_FUSED").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false))
}

pub(crate) fn get_bias_correction() -> bool {
    pub(crate) static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_BIAS_CORRECT").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false))
}

/// Pre-RoPE quantization: compress keys BEFORE rotary position embedding.
/// Pre-RoPE keys have position-independent per-channel statistics, giving
/// dramatically better codebook fit and recovering 34-59% of the quality gap
/// vs post-RoPE quantization (paper: BUG2-FINDINGS confirmed).
///
/// Default: ON (paper-published configuration). Without it, post-RoPE keys
/// produce compound generation errors at >8 compressed layers.
/// Override with TQ_PRE_ROPE=0 for ablation/research mode.
pub(crate) fn get_pre_rope() -> bool {
    pub(crate) static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_PRE_ROPE").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(true))
}

pub(crate) fn get_compact_threshold() -> usize {
    pub(crate) static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_COMPACT").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// Grouped GPU TQ path enabled. Default true (was opt-in via TQ_GPU_GROUPED=1
/// during Sprint 1C until parity was confirmed). Setting `TQ_GPU_GROUPED=0`
/// forces the legacy per-vector path — kept as an escape hatch for ablation
/// or in case future kernels expose a regression.
pub(crate) fn get_gpu_grouped_enabled() -> bool {
    pub(crate) static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_GPU_GROUPED")
        .ok()
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true))
}

pub(crate) fn get_compact_ratio() -> usize {
    pub(crate) static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_COMPACT_RATIO").ok().and_then(|v| v.parse().ok()).unwrap_or(5))
}

/// Quest-style CPU offload enabled. When on, evicted tokens are saved to CPU
/// instead of deleted, and can be retrieved on cache miss during attention.
/// Default: off (Sprint F experimental). Enable with TQ_OFFLOAD=1.
pub(crate) fn get_offload_enabled() -> bool {
    pub(crate) static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_OFFLOAD").ok().map(|v| v == "1").unwrap_or(false))
}

/// Override for TriAttention enable state. -1 = use env var, 0 = disabled, 1 = enabled.
/// Allows bench to toggle TriAttention between runs without process restart.
pub(crate) static TRIATTENTION_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

pub(crate) fn get_triattention_enabled() -> bool {
    let ov = TRIATTENTION_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if ov >= 0 { return ov == 1; }
    // Default: check env var once
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_TRIATTN").ok().map(|v| v == "1").unwrap_or(false))
}

/// Set triattention enabled/disabled override (for bench multi-run).
pub(crate) fn set_triattention_override(enabled: bool) {
    TRIATTENTION_OVERRIDE.store(if enabled { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
}

/// Clear triattention override (fall back to env var).
pub(crate) fn clear_triattention_override() {
    TRIATTENTION_OVERRIDE.store(-1, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn get_triattention_budget() -> usize {
    pub(crate) static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_TRIATTN_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(2048))
}

pub(crate) fn get_triattention_interval() -> usize {
    pub(crate) static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_TRIATTN_INTERVAL").ok().and_then(|v| v.parse().ok()).unwrap_or(128))
}

pub(crate) const TQ_VBITS_DEFAULT: u8 = 0;

pub(crate) fn get_value_bits() -> u8 {
    pub(crate) static C: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_VBITS").ok().and_then(|v| v.parse().ok()).unwrap_or(TQ_VBITS_DEFAULT))
}

/// Temporal decay: demote old tokens to lower bit widths.
/// Format: "age:bits[,age:bits]" e.g. "512:2" or "256:3,1024:2"
/// Set TQ_DECAY=off to disable. Default: off.
pub(crate) fn get_decay_config() -> Option<tq_kv::TemporalDecayConfig> {
    let val = std::env::var("TQ_DECAY").ok()?;
    if val == "off" || val == "0" || val.is_empty() { return None; }
    let mut tiers = Vec::new();
    for part in val.split(',') {
        let parts: Vec<&str> = part.trim().split(':').collect();
        if parts.len() == 2 {
            if let (Ok(age), Ok(bits)) = (parts[0].parse::<usize>(), parts[1].parse::<u8>()) {
                tiers.push(tq_kv::DecayTier { age_threshold: age, bits });
            }
        }
    }
    if tiers.is_empty() { return None; }
    tiers.sort_by_key(|t| t.age_threshold);
    Some(tq_kv::TemporalDecayConfig { tiers, decay_interval: 128 })
}

/// Compute SmoothAttention scales from calibration channel_scales.
///
/// SmoothAttention migrates K outliers to Q: K /= lambda, Q *= lambda.
/// When `for_query=true`: returns lambda (multiply Q by this).
/// When `for_query=false`: returns 1/lambda (multiply K by this — shrinks outliers).
///
/// If no channel_scales in config, returns None (SmoothAttention disabled).
/// Compute SmoothAttention scales from calibration channel_scales.
///
/// SmoothAttention replaces the old channel_scales-in-compression approach.
/// Instead of scaling K inside the compression pipeline (which changes K and
/// requires inverse on decompress), we migrate outliers from K to Q at the
/// tensor level. Q stays fp32 so this is lossless. K becomes smoother,
/// improving quantization quality.
///
/// Returns None if no channel_scales in config (SmoothAttention disabled).
pub(crate) fn compute_smooth_scales(
    config: &TurboQuantConfig,
    head_dim: usize,
    device: &Device,
    for_query: bool,
) -> Option<Tensor> {
    // Only enable SmoothAttention if we have calibrated channel_scales
    // AND compression is active (skip_layers != 999)
    let scales = config.channel_scales.as_ref()?;
    if scales.len() != head_dim || config.skip_layers == Some(999) {
        return None;
    }

    // For Q: multiply by sqrt(scale) — absorb half the outlier
    // For K: multiply by 1/sqrt(scale) — smooth down outliers
    // Invariance: (Q*sqrt(s)) * (K/sqrt(s))^T = Q*K^T
    let vals: Vec<f32> = if for_query {
        scales.iter().map(|&s| s.max(0.01).sqrt()).collect()
    } else {
        scales.iter().map(|&s| 1.0 / s.max(0.01).sqrt()).collect()
    };

    Tensor::from_vec(vals, vec![head_dim], device).ok()
}

// ============================================================
// Pre-allocated GPU KV Cache (for CUDA Graph replay)
// ============================================================

/// Maximum sequence length for pre-allocated KV cache.
/// Maximum pre-allocated KV cache sequence length (for graph-compatible padded attention).
/// Configurable via TQ_MAX_SEQ env var. Default 2048 for practical inference.
/// For production: set higher (e.g., 2048 or 4096).
pub(crate) fn get_max_kv_seq() -> usize {
    pub(crate) static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TQ_MAX_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(2048))
}

/// Pre-allocated scratch buffers for zero-alloc decode (seq_len=1).
///
/// During decode, each layer's fused kernels write into these fixed buffers
/// instead of allocating fresh GPU memory. Only one layer executes at a time,
/// so buffers are safely reused across all 28 layers.
///
/// Buffers are `Arc<CudaSlice<f32>>` so we can:
/// - `Arc::get_mut` → `&mut CudaSlice<f32>` for kernel writes (refcount==1 between layers)
/// - `Arc::clone` → cheap tensor view for downstream ops (refcount==2 during layer)
/// - After layer completes, tensor drops → refcount back to 1
#[cfg(feature = "cuda")]
pub(crate) struct DecodeScratch {
    /// QKV output buffers (fused norm+QKV kernel writes here)
    pub(crate) q_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    pub(crate) k_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    pub(crate) v_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// MLP intermediate buffer (fused gateup+silu kernel writes here)
    pub(crate) intermediate_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Fused attention output (gqa_decode_attention writes here)
    pub(crate) attn_out: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Wo projection output (qmatmul_gpu_into writes here)
    pub(crate) wo_out: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Combined residual + attn — ping-pong pair.
    /// Even layers write to [0], odd to [1]. Previous layer's output is read from the other.
    pub(crate) combined_bufs: [std::sync::Arc<cudarc::driver::CudaSlice<f32>>; 2],
    /// flash_decode partial buffers — persistent across decode steps. Sized
    /// for max n_splits at TQ_MAX_SEQ to avoid per-step alloc_zeros in the
    /// hot attention path.
    pub(crate) flash_partial_o: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    pub(crate) flash_partial_max: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    pub(crate) flash_partial_sum: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    pub(crate) flash_max_splits: usize,
    /// Cached stream for tensor wrapping
    pub(crate) stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// Dimension metadata
    pub(crate) q_out: usize,
    pub(crate) k_out: usize,
    pub(crate) v_out: usize,
    pub(crate) intermediate_dim: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
}

#[cfg(feature = "cuda")]
impl DecodeScratch {
    pub(crate) fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        q_out: usize,
        k_out: usize,
        v_out: usize,
        intermediate_dim: usize,
        hidden_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> std::result::Result<Self, crate::cuda::TqError> {
        let alloc = |name: &str, size: usize| -> std::result::Result<cudarc::driver::CudaSlice<f32>, crate::cuda::TqError> {
            crate::cuda::gpu_alloc_zeros_pub(&stream, size)
                .map_err(|e| crate::cuda::TqError::Msg(format!("DecodeScratch {} alloc: {}", name, e)))
        };
        let q_buf = alloc("q", q_out)?;
        let k_buf = alloc("k", k_out)?;
        let v_buf = alloc("v", v_out)?;
        let intermediate_buf = alloc("intermediate", intermediate_dim)?;
        let attn_out = alloc("attn_out", q_out)?;
        let wo_out = alloc("wo_out", hidden_dim)?;
        let combined_a = alloc("combined_a", hidden_dim)?;
        let combined_b = alloc("combined_b", hidden_dim)?;
        // flash_decode persistent partial buffers — sized for max n_splits at
        // TQ_MAX_SEQ with split_size chosen to match the runtime-smallest value.
        // Default split_size=64 (matches model.rs TQ_FLASH_SPLIT default); env
        // override consulted here so buffer is big enough for user's choice.
        let max_seq = get_max_kv_seq();
        let flash_split_size: usize = std::env::var("TQ_FLASH_SPLIT")
            .ok().and_then(|v| v.parse().ok())
            .unwrap_or(32)
            .max(16);
        let flash_max_splits = (max_seq + flash_split_size - 1) / flash_split_size;
        let flash_partial_o = alloc("flash_partial_o", n_head * flash_max_splits * head_dim)?;
        let flash_partial_max = alloc("flash_partial_max", n_head * flash_max_splits)?;
        let flash_partial_sum = alloc("flash_partial_sum", n_head * flash_max_splits)?;
        let total_kb = (q_out * 2 + k_out + v_out + intermediate_dim + hidden_dim * 3
            + n_head * flash_max_splits * (head_dim + 2)) * 4 / 1024;
        eprintln!("  DecodeScratch allocated: qkv={}+{}+{} attn={} wo={} comb={}x2 inter={} flash={}×{} ({}KB)",
            q_out, k_out, v_out, q_out, hidden_dim, hidden_dim, intermediate_dim,
            n_head, flash_max_splits, total_kb);
        Ok(Self {
            q_buf: std::sync::Arc::new(q_buf),
            k_buf: std::sync::Arc::new(k_buf),
            v_buf: std::sync::Arc::new(v_buf),
            intermediate_buf: std::sync::Arc::new(intermediate_buf),
            attn_out: std::sync::Arc::new(attn_out),
            wo_out: std::sync::Arc::new(wo_out),
            combined_bufs: [std::sync::Arc::new(combined_a), std::sync::Arc::new(combined_b)],
            flash_partial_o: std::sync::Arc::new(flash_partial_o),
            flash_partial_max: std::sync::Arc::new(flash_partial_max),
            flash_partial_sum: std::sync::Arc::new(flash_partial_sum),
            flash_max_splits,
            stream,
            q_out,
            k_out,
            v_out,
            intermediate_dim,
            hidden_dim,
            n_head,
            n_kv_head,
            head_dim,
        })
    }
}

/// Pre-allocated GPU KV cache for CUDA Graph compatible inference.
///
/// Layout: K and V are flat `[n_kv_head * max_seq * head_dim]` buffers.
/// Per-head data is contiguous: head h starts at `h * max_seq * head_dim`.
/// New tokens are appended at `h * max_seq * head_dim + seq_len * head_dim`.
///
/// For attention, the full buffer is used as `[1, n_kv_head, max_seq, head_dim]`
/// with positions >= seq_len masked to -inf in the attention score.
#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuKvCache")
            .field("seq_len", &self.seq_len)
            .field("max_seq", &self.max_seq)
            .field("n_kv_head", &self.n_kv_head)
            .field("head_dim", &self.head_dim)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuKvCache {
    fn clone(&self) -> Self {
        // Shallow clone: Arc bump (zero-copy GPU buffers)
        Self {
            k_buf: self.k_buf.clone(),
            v_buf: self.v_buf.clone(),
            mask_buf: self.mask_buf.clone(),
            valid_len_gpu: self.stream.clone_htod(&[self.seq_len as i32]).unwrap(),
            seq_len: self.seq_len,
            max_seq: self.max_seq,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
            stream: self.stream.clone(),
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct GpuKvCache {
    pub(crate) k_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    pub(crate) v_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// Attention padding mask: [max_seq]. 0.0 for valid, -1e10 for padding.
    pub(crate) mask_buf: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
    /// GPU scalar for current valid length (read by mask generation kernel).
    /// Updated via clone_htod before graph replay — graph-safe pointer.
    pub(crate) valid_len_gpu: cudarc::driver::CudaSlice<i32>,
    pub(crate) seq_len: usize,
    pub(crate) max_seq: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    pub(crate) fn new(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        n_kv_head: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, crate::cuda::TqError> {
        let total = n_kv_head * max_seq * head_dim;
        let k_buf = crate::cuda::gpu_alloc_zeros_pub(&stream, total)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache k alloc: {}", e)))?;
        let v_buf = crate::cuda::gpu_alloc_zeros_pub(&stream, total)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache v alloc: {}", e)))?;
        let mask_buf = crate::cuda::gpu_alloc_zeros_pub(&stream, max_seq)
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache mask alloc: {}", e)))?;
        let valid_len_gpu = stream.clone_htod(&[0i32])
            .map_err(|e| crate::cuda::TqError::Msg(format!("GpuKvCache len alloc: {}", e)))?;
        Ok(Self {
            k_buf: std::sync::Arc::new(k_buf),
            v_buf: std::sync::Arc::new(v_buf),
            mask_buf: std::sync::Arc::new(mask_buf),
            valid_len_gpu,
            seq_len: 0,
            max_seq,
            n_kv_head,
            head_dim,
            stream,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Append new K/V tokens. Input shape: [1, n_kv_head, n_new, head_dim].
    pub(crate) fn append(&mut self, new_k: &Tensor, new_v: &Tensor, n_new: usize) -> Result<()> {
        if self.seq_len + n_new > self.max_seq {
            pub(crate) static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| eprintln!("[WARN] GpuKvCache: max_seq={} reached, context truncated", self.max_seq));
            return Ok(());
        }
        let reg = crate::cuda::kernels::global_registry()
            .ok_or_else(|| TqError::Msg("no GPU registry".into()))?;
        let k_src = new_k.cuda_data();
        let v_src = new_v.cuda_data();

        // During graph capture, skip memcpy_htod — it would be captured as a
        // graph node with baked host address. The pre-replay code updates valid_len_gpu.
        #[cfg(feature = "cuda")]
        let capturing = {
            match self.stream.capture_status() {
                Ok(cudarc::driver::sys::CUstreamCaptureStatus_enum::CU_STREAM_CAPTURE_STATUS_ACTIVE) => true,
                _ => false,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let capturing = false;

        if !capturing {
            let pos_val = self.seq_len as i32;
            let _ = self.stream.memcpy_htod(&[pos_val], &mut self.valid_len_gpu);
        }

        // Single kernel per K and V: handles all heads, reads seq_pos from GPU scalar.
        let k_refs = std::sync::Arc::strong_count(&self.k_buf);
        if k_refs > 1 {
            eprintln!("[WARN] GpuKvCache::append k_buf Arc ref_count={} — external alias will see stale data", k_refs);
        }
        let k_dst = std::sync::Arc::make_mut(&mut self.k_buf);
        crate::cuda::kernels::kv_cache_append(
            reg, k_src, k_dst, &self.valid_len_gpu,
            self.n_kv_head, self.max_seq, self.head_dim, n_new,
        ).map_err(|e| TqError::Msg(format!("kv append k: {}", e)))?;

        let v_refs = std::sync::Arc::strong_count(&self.v_buf);
        if v_refs > 1 {
            eprintln!("[WARN] GpuKvCache::append v_buf Arc ref_count={} — external alias will see stale data", v_refs);
        }
        let v_dst = std::sync::Arc::make_mut(&mut self.v_buf);
        crate::cuda::kernels::kv_cache_append(
            reg, v_src, v_dst, &self.valid_len_gpu,
            self.n_kv_head, self.max_seq, self.head_dim, n_new,
        ).map_err(|e| TqError::Msg(format!("kv append v: {}", e)))?;

        self.seq_len += n_new;

        if !capturing {
            // Normal mode: update valid_len to post-append total, generate mask
            let new_len = self.seq_len as i32;
            let _ = self.stream.memcpy_htod(&[new_len], &mut self.valid_len_gpu);
            let mask_mut = std::sync::Arc::make_mut(&mut self.mask_buf);
            crate::cuda::kernels::generate_kv_mask(reg, mask_mut, &self.valid_len_gpu, self.max_seq, 0)
                .map_err(|e| TqError::Msg(format!("kv mask gen: {}", e)))?;
        } else {
            // Graph capture: valid_len_gpu has pre_pos (set by pre-replay code).
            // Generate mask with extra=n_new so valid_len = pre_pos + n_new = post_pos.
            // This kernel launch IS captured in the graph.
            let mask_mut = std::sync::Arc::make_mut(&mut self.mask_buf);
            crate::cuda::kernels::generate_kv_mask(reg, mask_mut, &self.valid_len_gpu, self.max_seq, n_new)
                .map_err(|e| TqError::Msg(format!("kv mask gen: {}", e)))?;
        }
        Ok(())
    }

    /// Get K as a Tensor wrapping the pre-allocated buffer.
    /// Shape: [1, n_kv_head, max_seq, head_dim] (full buffer, mask unused positions).
    pub(crate) fn k_tensor(&self) -> Tensor {
        Tensor::from_cuda_arc(
            self.k_buf.clone(),
            vec![1, self.n_kv_head, self.max_seq, self.head_dim],
            self.stream.clone(),
        )
    }

    /// Get V as a Tensor wrapping the pre-allocated buffer.
    pub(crate) fn v_tensor(&self) -> Tensor {
        Tensor::from_cuda_arc(
            self.v_buf.clone(),
            vec![1, self.n_kv_head, self.max_seq, self.head_dim],
            self.stream.clone(),
        )
    }

    /// Get K narrowed to valid tokens only: [1, n_kv_head, seq_len, head_dim].
    /// No padding waste — attention only computes over actual positions.
    pub(crate) fn k_tensor_valid(&self) -> Result<Tensor> {
        let full = self.k_tensor();
        full.narrow(2, 0, self.seq_len)
    }

    /// Get V narrowed to valid tokens: [1, n_kv_head, seq_len, head_dim].
    pub(crate) fn v_tensor_valid(&self) -> Result<Tensor> {
        let full = self.v_tensor();
        full.narrow(2, 0, self.seq_len)
    }

    /// Get attention mask as tensor [1, 1, 1, max_seq] for broadcasting.
    pub(crate) fn mask_tensor(&self) -> Tensor {
        Tensor::from_cuda_arc(
            self.mask_buf.clone(),
            vec![1, 1, 1, self.max_seq],
            self.stream.clone(),
        )
    }
}

// ============================================================
// Quest-style CPU Offload Cache (Sprint F)
// ============================================================

/// Single offloaded compressed KV entry — one token's worth of compressed K + raw V.
/// Stored on CPU after TriAttention eviction. Can be retrieved on cache miss.
#[derive(Debug, Clone)]
pub(crate) struct OffloadedKvEntry {
    /// Original sequence position (for RoPE reconstruction)
    pub(crate) seq_pos: usize,
    /// Packed TQ indices: [n_kv_head * bytes_per_key]
    pub(crate) packed_k: Vec<u8>,
    /// Per-group norms: [n_kv_head * n_groups] (grouped path)
    pub(crate) gnorms: Vec<f32>,
    /// V data (f32): [n_kv_head * head_dim]
    pub(crate) v_data: Vec<f32>,
}

/// CPU-side storage for tokens evicted by TriAttention.
/// Instead of deleting evicted tokens, we save them here for potential retrieval
/// when a future query has high attention to an evicted position ("cache miss").
///
/// V1 uses simple Vec storage with linear scan for retrieval.
/// Future: FAISS index for O(log n) search at >32K offloaded tokens.
#[derive(Debug, Clone)]
pub(crate) struct OffloadedKvCache {
    /// Per-layer storage of offloaded entries.
    pub(crate) layers: Vec<Vec<OffloadedKvEntry>>,
    /// Config for decompression on retrieval.
    pub(crate) bits: u8,
    pub(crate) head_dim: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) group_size: usize,
    pub(crate) bytes_per_key: usize,
    pub(crate) n_groups: usize,
}

impl OffloadedKvCache {
    pub(crate) fn new(
        n_layers: usize,
        bits: u8,
        head_dim: usize,
        n_kv_head: usize,
        group_size: usize,
    ) -> Self {
        let bytes_per_key = (head_dim * bits as usize + 7) / 8;
        let n_groups = if group_size > 0 && head_dim % group_size == 0 {
            head_dim / group_size
        } else { 1 };
        Self {
            layers: (0..n_layers).map(|_| Vec::new()).collect(),
            bits,
            head_dim,
            n_kv_head,
            group_size,
            bytes_per_key,
            n_groups,
        }
    }

    /// Store an evicted token's compressed K + V data.
    pub(crate) fn offload(&mut self, layer_idx: usize, entry: OffloadedKvEntry) {
        if layer_idx < self.layers.len() {
            self.layers[layer_idx].push(entry);
        }
    }

    /// Total number of offloaded tokens across all layers.
    pub(crate) fn total_offloaded(&self) -> usize {
        self.layers.iter().map(|l| l.len()).max().unwrap_or(0)
    }

    /// Retrieve and decompress offloaded keys for a given layer.
    /// Returns (decompressed_keys, v_data, seq_positions) for attention scoring.
    /// Keys are returned in Hadamard domain (pre-inverse-WHT) for fused scoring,
    /// or fully decompressed depending on the caller's needs.
    pub(crate) fn retrieve_all_for_layer(&self, layer_idx: usize) -> Option<&[OffloadedKvEntry]> {
        self.layers.get(layer_idx).filter(|l| !l.is_empty()).map(|l| l.as_slice())
    }

    /// Memory usage in bytes.
    pub(crate) fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| {
            l.iter().map(|e| {
                e.packed_k.len() + e.gnorms.len() * 4 + e.v_data.len() * 4 + 8
            }).sum::<usize>()
        }).sum()
    }
}
