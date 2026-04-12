//! Layer weight streaming for VRAM-constrained inference.
//!
//! When the full model doesn't fit in GPU memory, `LayerSwapManager` streams
//! non-pinned layers on/off the device using a dedicated transfer stream and
//! a small set of rotating slots (double-buffered by default). Pinned layers
//! stay resident for the full run.
//!
//! # High-level flow (Phase 5 will wire this into the forward loop)
//!
//! ```text
//! for layer_idx in 0..n_layers {
//!     if !swap.is_pinned(layer_idx) {
//!         swap.wait_for_layer(layer_idx)?;   // compute stream waits on H2D ready event
//!     }
//!     // … forward pass for this layer …
//!     if !swap.is_pinned(layer_idx) {
//!         swap.evict_layer(layer_idx);        // drop GPU slot, free the slot buffer
//!         swap.start_prefetch(layer_idx + 1)?; // kick next layer's H2D on transfer stream
//!     }
//! }
//! ```
//!
//! # Safety / invariants
//!
//! - Single-threaded: the manager assumes one caller drives the forward loop.
//!   Matches the existing `QWeight` / model assumptions.
//! - Transfer stream is distinct from compute stream; synchronisation is via
//!   `CudaEvent`s recorded on the transfer stream and waited on by the compute
//!   stream (`CudaStream::wait(&event)`).
//! - CUDA Graph capture is incompatible with swap mode (slot buffer device
//!   addresses change between prefetches). `engine.rs` should disable
//!   `TQ_GRAPH` whenever swap is active.
//!
//! # Status (Phase 4)
//!
//! Infrastructure only — manager construction, slot allocation, pin/prefetch
//! plumbing. **Not yet wired into the forward loop** (Phase 5). Phase 5 also
//! refactors `QWeight::gpu_cache` to hold `Arc<CudaSlice<u8>>` so a slot
//! buffer can be shared with the layer's QWeights via `inject_gpu`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

use crate::cuda::{memory::PinnedBuffer, Result, TqError};

/// Send+Sync wrapper around a raw pointer to a `QWeight`.
///
/// Needed because the forward loop stashes `*const QWeight` in
/// `GenericTurboModel::layer_qweight_ptrs`, and the whole `Engine` must be
/// moveable across threads (e.g. `tokio::task::spawn_blocking` in `serve.rs`).
/// Pointers are valid as long as the owning `LayerWeights` Vec isn't
/// reallocated — which never happens after model load.
#[derive(Copy, Clone)]
pub struct QWeightPtr(pub *const crate::qmatmul::QWeight);

// SAFETY: QWeight is Sync (SwapCell / OnceLock / UnsafeCell invariants held by
// single-threaded forward). The raw pointer doesn't add any extra aliasing
// hazards beyond what a `&QWeight` already has.
unsafe impl Send for QWeightPtr {}
unsafe impl Sync for QWeightPtr {}

// ─────────────────────────────────────────────────────────────
// Per-layer metadata: which bytes (in the host-side source) belong
// to this layer, and how much GPU space the layer needs.
// ─────────────────────────────────────────────────────────────

/// Where a single weight lives in the host-side source (file, mmap, or Vec).
#[derive(Clone, Debug)]
pub struct WeightLoc {
    /// Logical name (e.g. "blk.3.attn_q.weight") — informational.
    pub name: String,
    /// Byte offset into the source.
    pub src_offset: usize,
    /// Number of bytes.
    pub nbytes: usize,
    /// Offset into the per-layer packed GPU buffer.
    pub gpu_offset: usize,
}

/// All weights belonging to a single transformer layer.
#[derive(Clone, Debug)]
pub struct LayerMeta {
    pub layer_idx: usize,
    pub weights: Vec<WeightLoc>,
    /// Total bytes for the layer (sum of weights + alignment padding).
    pub total_bytes: usize,
}

// ─────────────────────────────────────────────────────────────
// Slot: a pre-allocated GPU buffer large enough for the largest
// non-pinned layer, plus an event recorded after H2D completes.
// ─────────────────────────────────────────────────────────────

pub struct LayerGpuSlot {
    /// Pre-allocated GPU buffer (bytes_per_layer).
    pub buffer: Arc<CudaSlice<u8>>,
    /// Which layer currently occupies this slot (None = free).
    pub loaded: Option<usize>,
    /// Recorded on the transfer stream after the H2D memcpy for `loaded`.
    pub ready: CudaEvent,
}

// ─────────────────────────────────────────────────────────────
// Manager
// ─────────────────────────────────────────────────────────────

pub struct LayerSwapManager {
    #[allow(dead_code)] // Held for lifetime; Phase 5 may need for event creation
    ctx: Arc<CudaContext>,
    transfer_stream: Arc<CudaStream>,
    compute_stream: Arc<CudaStream>,

    /// Double-buffered (or N-buffered) GPU slots. Reserved for the future
    /// optimized prefetch path; Phase 5 uses per-weight `clone_htod` instead.
    #[allow(dead_code)]
    slots: Vec<LayerGpuSlot>,
    /// Staging buffer in pinned host memory. Optional — falls back to pageable.
    #[allow(dead_code)]
    staging: PinnedBuffer,

    /// Always-resident layers (never evicted).
    pinned: std::collections::HashSet<usize>,

    /// Ready events per active (non-pinned) layer — the compute stream waits
    /// on these before launching that layer's kernels.
    ready_events: HashMap<usize, CudaEvent>,

    /// Per-layer metadata (indexed by layer_idx). Reserved for the future
    /// optimized path; the direct-ptr prefetch uses QWeight.raw_data directly.
    #[allow(dead_code)]
    layer_meta: Vec<LayerMeta>,
    /// Max bytes across all non-pinned layers. Used to size each slot.
    #[allow(dead_code)]
    bytes_per_layer: usize,
}

impl LayerSwapManager {
    /// Construct a manager.
    ///
    /// Arguments
    /// * `ctx` — CUDA context (usually from `TqDevice::cuda_context`)
    /// * `compute_stream` — the forward-pass stream (from `TqDevice::cuda_stream`)
    /// * `layer_meta` — per-layer byte layout (built by the caller)
    /// * `pinned_layers` — layer indices to keep always resident
    /// * `n_slots` — number of rotating slots (2 = double-buffer; use 1 for
    ///   strict serial behaviour when testing)
    pub fn new(
        ctx: Arc<CudaContext>,
        compute_stream: Arc<CudaStream>,
        layer_meta: Vec<LayerMeta>,
        pinned_layers: &[usize],
        n_slots: usize,
    ) -> Result<Self> {
        if n_slots == 0 {
            return Err(TqError::Msg("LayerSwapManager: n_slots must be ≥ 1".into()));
        }
        let transfer_stream = ctx
            .new_stream()
            .map_err(|e| TqError::Msg(format!("transfer stream: {}", e)))?;

        // Determine bytes_per_layer from the largest non-pinned layer.
        let pinned_set: std::collections::HashSet<usize> = pinned_layers.iter().copied().collect();
        let bytes_per_layer = layer_meta
            .iter()
            .filter(|lm| !pinned_set.contains(&lm.layer_idx))
            .map(|lm| lm.total_bytes)
            .max()
            .unwrap_or(0);

        // Allocate N slots.
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            let buf = transfer_stream
                .alloc_zeros::<u8>(bytes_per_layer.max(1))
                .map_err(|e| TqError::Msg(format!("slot alloc: {}", e)))?;
            let event = ctx
                .new_event(None)
                .map_err(|e| TqError::Msg(format!("slot event: {}", e)))?;
            slots.push(LayerGpuSlot {
                buffer: Arc::new(buf),
                loaded: None,
                ready: event,
            });
        }

        // Pinned layers tracked by index — QWeights hold their own GPU caches.
        let pinned = pinned_set;

        // Host staging buffer — one layer at a time. Sized to the largest
        // (pinned + non-pinned) layer so either case fits.
        let staging_size = layer_meta
            .iter()
            .map(|lm| lm.total_bytes)
            .max()
            .unwrap_or(0)
            .max(1);
        let staging = PinnedBuffer::new_on_context(&ctx, staging_size);

        Ok(Self {
            ctx,
            transfer_stream,
            compute_stream,
            slots,
            staging,
            pinned,
            ready_events: HashMap::new(),
            layer_meta,
            bytes_per_layer,
        })
    }

    /// Is this layer always resident?
    pub fn is_pinned(&self, layer_idx: usize) -> bool {
        self.pinned.contains(&layer_idx)
    }

    /// Mark a layer as always resident.
    pub fn mark_pinned(&mut self, layer_idx: usize) {
        self.pinned.insert(layer_idx);
    }

    /// Upload the given layer's QWeight bytes to GPU (via `inject_gpu`) on the
    /// transfer stream, then block the CPU until the copy completes. Used for
    /// always-resident ("pinned") layers during model load.
    ///
    /// SAFETY: `qweight_ptrs` must remain valid for the duration of this call.
    /// In practice: pointers into `self.layers[i].qweights()` — valid while
    /// the model exists.
    pub unsafe fn pin_layer_direct(
        &mut self,
        layer_idx: usize,
        qweight_ptrs: &[QWeightPtr],
    ) -> Result<()> {
        for qw_ptr in qweight_ptrs {
            let qw = &*qw_ptr.0;
            let gpu = self
                .transfer_stream
                .clone_htod(qw.raw_data.as_slice())
                .map_err(|e| TqError::Msg(format!("pin H2D layer {}: {}", layer_idx, e)))?;
            qw.inject_gpu(gpu);
        }
        self.transfer_stream
            .synchronize()
            .map_err(|e| TqError::Msg(format!("pin sync layer {}: {}", layer_idx, e)))?;
        self.pinned.insert(layer_idx);
        Ok(())
    }

    /// Async H2D for all QWeights of `layer_idx`. Each weight gets a fresh
    /// CudaSlice via `clone_htod` (no slot reuse for now). Event recorded on
    /// the transfer stream; `wait_for_layer` hooks the compute stream on it.
    ///
    /// No-op if the layer is pinned.
    ///
    /// SAFETY: same as `pin_layer_direct`.
    pub unsafe fn start_prefetch_direct(
        &mut self,
        layer_idx: usize,
        qweight_ptrs: &[QWeightPtr],
    ) -> Result<()> {
        if self.is_pinned(layer_idx) {
            return Ok(());
        }
        for qw_ptr in qweight_ptrs {
            let qw = &*qw_ptr.0;
            let gpu = self
                .transfer_stream
                .clone_htod(qw.raw_data.as_slice())
                .map_err(|e| TqError::Msg(format!("prefetch H2D layer {}: {}", layer_idx, e)))?;
            qw.inject_gpu(gpu);
        }
        let event = self
            .transfer_stream
            .record_event(None)
            .map_err(|e| TqError::Msg(format!("prefetch event layer {}: {}", layer_idx, e)))?;
        self.ready_events.insert(layer_idx, event);
        Ok(())
    }

    /// Block the compute stream (not the CPU) on the layer's H2D-ready event.
    /// Pinned layers are a no-op (already resident).
    pub fn wait_for_layer(&self, layer_idx: usize) -> Result<()> {
        if self.is_pinned(layer_idx) {
            return Ok(());
        }
        match self.ready_events.get(&layer_idx) {
            Some(event) => self
                .compute_stream
                .wait(event)
                .map_err(|e| TqError::Msg(format!("wait_for_layer {}: {}", layer_idx, e))),
            // No event yet (not prefetched). This is legitimate on first entry
            // for a non-pinned layer — the forward loop's initial iteration
            // has no prior prefetch. Skip and rely on the existing
            // `gpu_cache_or_upload` lazy-init path.
            None => Ok(()),
        }
    }

    /// Drop the GPU caches of every QWeight in the layer (swap-mode QWeights
    /// release their `CudaSlice<u8>`; pinned layers are not evicted).
    ///
    /// SAFETY: same as `pin_layer_direct`.
    pub unsafe fn evict_layer_direct(
        &mut self,
        layer_idx: usize,
        qweight_ptrs: &[QWeightPtr],
    ) {
        if self.is_pinned(layer_idx) {
            return;
        }
        // CRITICAL: block until all compute-stream kernels that consumed this
        // layer's weights are complete. `evict_gpu` drops the CudaSlice, which
        // calls cuMemFree synchronously — freeing memory that's still
        // referenced by a queued kernel would trigger ILLEGAL_ADDRESS.
        //
        // A future optimization (Phase 7): record a compute-stream event at
        // end of each layer and defer eviction until the event is signalled,
        // so the sync cost can overlap with the next prefetch.
        if let Err(e) = self.compute_stream.synchronize() {
            eprintln!("[layer_swap] compute sync before evict L{} failed: {}", layer_idx, e);
        }
        for qw_ptr in qweight_ptrs {
            let qw = &*qw_ptr.0;
            qw.evict_gpu();
        }
        self.ready_events.remove(&layer_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_loc_debug_roundtrip() {
        let wl = WeightLoc {
            name: "blk.0.attn_q.weight".into(),
            src_offset: 128,
            nbytes: 256,
            gpu_offset: 0,
        };
        let s = format!("{:?}", wl);
        assert!(s.contains("attn_q"));
    }

    #[test]
    fn layer_meta_total_consistency() {
        let lm = LayerMeta {
            layer_idx: 3,
            weights: vec![
                WeightLoc {
                    name: "a".into(),
                    src_offset: 0,
                    nbytes: 100,
                    gpu_offset: 0,
                },
                WeightLoc {
                    name: "b".into(),
                    src_offset: 100,
                    nbytes: 200,
                    gpu_offset: 128, // aligned
                },
            ],
            total_bytes: 128 + 200,
        };
        assert_eq!(lm.layer_idx, 3);
        assert_eq!(lm.weights.len(), 2);
    }
}
