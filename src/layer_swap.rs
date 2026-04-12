//! Layer weight streaming for VRAM-constrained inference.
//!
//! When the full model doesn't fit in GPU memory, `LayerSwapManager` streams
//! non-pinned layers on/off the device on a dedicated transfer stream. Pinned
//! layers stay resident for the full run.
//!
//! # Lane pool (Phase 7 real)
//!
//! Earlier phases did a `clone_htod` per QWeight per prefetch — 7 cudaMalloc
//! per layer per decode token on dense transformers, which on Qwen2.5-32B
//! (52 streamed layers) collapsed decode to 0.2 tok/s.
//!
//! The pool-of-lanes design eliminates that malloc thrash:
//!   * At startup the manager allocates `N` **lanes**, each containing one
//!     `Arc<CudaSlice<u8>>` per QWeight slot (sized to that weight's packed
//!     bytes). Total: `n_lanes * Σ QWeight-bytes` — e.g. 2×280 MB = 560 MB
//!     for 32B.
//!   * On prefetch: pick a free lane, `memcpy_htod` each weight's bytes
//!     **in place** into the lane's slot (no cudaMalloc), then `Arc::clone`
//!     each slot and `inject_gpu` it into the corresponding QWeight.
//!   * On eviction: QWeight drops its Arc. Lane keeps its Arc, so the
//!     buffer stays alive and is ready for the next prefetch — back to
//!     `memcpy_htod` with zero alloc.
//!
//! # Sync model
//!
//! Three primitives coordinate:
//!   * **Transfer stream** does async H2D into lane slots. After each
//!     prefetch records `ready_events[layer]`.
//!   * **Compute stream** (forward pass) waits on that event before
//!     launching the layer's kernels (`wait_for_layer`).
//!   * **Compute-completion events** — on eviction we record an event on
//!     the compute stream. The lane is not freed for re-prefetch until
//!     that event fires (tracked via `pending_eviction`), so the memcpy
//!     into the slot can't race with an in-flight kernel reading it.
//!
//! # Safety / invariants
//!
//! - Single-threaded forward (matches QWeight invariant).
//! - CUDA Graph + swap are mutually exclusive (slot device addresses change
//!   between prefetches). `engine.rs` disables `TQ_GRAPH` under swap.
//! - `QWeightPtr` wraps `*const QWeight` with `Send + Sync`; required so
//!   the owning `Engine` stays `Send` for `tokio::spawn_blocking`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

use crate::cuda::{Result, TqError};

/// Send+Sync wrapper around a raw pointer to a `QWeight`.
#[derive(Copy, Clone)]
pub struct QWeightPtr(pub *const crate::qmatmul::QWeight);

unsafe impl Send for QWeightPtr {}
unsafe impl Sync for QWeightPtr {}

// ─────────────────────────────────────────────────────────────
// Lane: one set of pre-allocated slot buffers matching the
// per-QWeight structure of every non-pinned layer.
// ─────────────────────────────────────────────────────────────

struct Lane {
    /// One Arc<CudaSlice> per QWeight of a layer, in the order produced by
    /// `LayerWeights::qweight_ptrs`. All layers are assumed to share the same
    /// QWeight structure (count + sizes). True for all decoder-only
    /// transformers we target (qwen2/3, llama, mistral, gemma, phi3).
    slots: Vec<Arc<CudaSlice<u8>>>,
    /// Currently-loaded layer_idx, None if empty or pending-evict-cleared.
    loaded: Option<usize>,
    /// Compute-stream event marking when prior kernels that read this lane
    /// have finished. The lane cannot be re-prefetched until the event
    /// fires, otherwise `memcpy_htod` would race the kernel. `None` means
    /// the lane has never been used OR the prior eviction's event has
    /// already been cleared.
    pending_evict_event: Option<CudaEvent>,
}

// ─────────────────────────────────────────────────────────────
// Manager
// ─────────────────────────────────────────────────────────────

pub struct LayerSwapManager {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    transfer_stream: Arc<CudaStream>,
    compute_stream: Arc<CudaStream>,

    pinned: std::collections::HashSet<usize>,
    /// H2D completion events per layer currently loaded in a lane.
    ready_events: HashMap<usize, CudaEvent>,

    /// Pool of pre-allocated lanes. Indexed by lane id (0..n_lanes).
    lanes: Vec<Lane>,
    /// Per-layer → lane id mapping for layers currently loaded in a lane.
    layer_lane: HashMap<usize, usize>,
}

impl LayerSwapManager {
    pub fn new(
        ctx: Arc<CudaContext>,
        compute_stream: Arc<CudaStream>,
        pinned_layers: &[usize],
    ) -> Result<Self> {
        let transfer_stream = ctx
            .new_stream()
            .map_err(|e| TqError::Msg(format!("transfer stream: {}", e)))?;

        Ok(Self {
            ctx,
            transfer_stream,
            compute_stream,
            pinned: pinned_layers.iter().copied().collect(),
            ready_events: HashMap::new(),
            lanes: Vec::new(),
            layer_lane: HashMap::new(),
        })
    }

    pub fn is_pinned(&self, layer_idx: usize) -> bool {
        self.pinned.contains(&layer_idx)
    }

    pub fn mark_pinned(&mut self, layer_idx: usize) {
        self.pinned.insert(layer_idx);
    }

    /// Pre-allocate `n_lanes` worth of slot buffers, sized from the template
    /// layer's QWeight sizes. Call once after `mark_pinned`/`pin_layer_direct`
    /// have finished, before the first `start_prefetch_direct`.
    ///
    /// SAFETY: pointers must outlive the manager (same invariant as
    /// `pin_layer_direct`). Only reads `QWeight::raw_data.len()`.
    pub unsafe fn allocate_lanes(
        &mut self,
        n_lanes: usize,
        template_ptrs: &[QWeightPtr],
    ) -> Result<()> {
        assert!(
            self.lanes.is_empty(),
            "allocate_lanes called twice — pool already initialised"
        );
        let mut total_bytes: usize = 0;
        for _ in 0..n_lanes {
            let mut slots: Vec<Arc<CudaSlice<u8>>> = Vec::with_capacity(template_ptrs.len());
            for p in template_ptrs {
                let qw = &*p.0;
                let nbytes = qw.raw_data.len();
                let slice = self
                    .transfer_stream
                    .alloc_zeros::<u8>(nbytes)
                    .map_err(|e| TqError::Msg(format!("lane slot alloc ({} bytes): {}", nbytes, e)))?;
                slots.push(Arc::new(slice));
                total_bytes += nbytes;
            }
            self.lanes.push(Lane {
                slots,
                loaded: None,
                pending_evict_event: None,
            });
        }
        eprintln!(
            "  Layer-swap lanes: {} × {} slots ({:.1} MB total)",
            n_lanes,
            template_ptrs.len(),
            total_bytes as f64 / 1_048_576.0,
        );
        Ok(())
    }

    /// Upload the given layer's QWeight bytes to GPU and block until the
    /// copy completes. Used for always-resident layers.
    ///
    /// SAFETY: `qweight_ptrs` must remain valid for the duration of this call.
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
            qw.inject_gpu(Arc::new(gpu));
        }
        self.transfer_stream
            .synchronize()
            .map_err(|e| TqError::Msg(format!("pin sync layer {}: {}", layer_idx, e)))?;
        self.pinned.insert(layer_idx);
        Ok(())
    }

    /// Pick an available lane: prefer empty/cleared, else one whose
    /// `pending_evict_event` has fired. Blocks on the oldest lane's event
    /// if none are ready.
    fn claim_lane(&mut self) -> Result<usize> {
        // First pass: find a fully free lane (no loaded + no pending event).
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            if lane.loaded.is_none() {
                // If there's a pending event, query it non-blocking first.
                if let Some(ev) = &lane.pending_evict_event {
                    if ev.is_complete() {
                        lane.pending_evict_event = None;
                        return Ok(i);
                    }
                } else {
                    return Ok(i);
                }
            }
        }
        // Second pass: no lane is free. Block on the first lane's event.
        let lane = &mut self.lanes[0];
        if let Some(ev) = lane.pending_evict_event.take() {
            ev.synchronize()
                .map_err(|e| TqError::Msg(format!("lane claim sync: {}", e)))?;
        }
        lane.loaded = None;
        Ok(0)
    }

    /// Async H2D (memcpy into pre-allocated lane slot; NO alloc) and inject
    /// the lane's Arc<CudaSlice> into each QWeight.
    ///
    /// No-op if `layer_idx` is pinned or already loaded in a lane.
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
        if self.layer_lane.contains_key(&layer_idx) {
            return Ok(()); // already in a lane
        }
        if self.lanes.is_empty() {
            return Err(TqError::Msg(
                "start_prefetch_direct before allocate_lanes".into(),
            ));
        }
        let lane_idx = self.claim_lane()?;

        // In-place memcpy_htod into each slot of the lane.
        let lane = &mut self.lanes[lane_idx];
        if lane.slots.len() != qweight_ptrs.len() {
            return Err(TqError::Msg(format!(
                "lane slot count {} != layer QWeight count {}",
                lane.slots.len(),
                qweight_ptrs.len()
            )));
        }
        for (slot_arc, qw_ptr) in lane.slots.iter_mut().zip(qweight_ptrs.iter()) {
            let qw = &*qw_ptr.0;
            let src = qw.raw_data.as_slice();
            // The Arc has 2+ strong refs during active layer + graveyard;
            // we need &mut CudaSlice. Get it via Arc::get_mut which succeeds
            // only if no other clones exist (i.e. on the first prefetch OR
            // after all prior QWeights in this lane have evict_gpu'd).
            match Arc::get_mut(slot_arc) {
                Some(buf) => {
                    let mut view = buf.slice_mut(0..src.len());
                    self.transfer_stream
                        .memcpy_htod(src, &mut view)
                        .map_err(|e| TqError::Msg(format!(
                            "prefetch memcpy_htod L{}: {}",
                            layer_idx, e
                        )))?;
                }
                None => {
                    return Err(TqError::Msg(format!(
                        "lane slot still referenced by QWeight (strong_count = {}) — \
                         prior layer not yet evicted when prefetching L{}",
                        Arc::strong_count(slot_arc),
                        layer_idx
                    )));
                }
            }
            // Share the Arc with the QWeight for this forward.
            qw.inject_gpu(Arc::clone(slot_arc));
        }
        // Record H2D-ready event for cross-stream sync.
        let event = self
            .transfer_stream
            .record_event(None)
            .map_err(|e| TqError::Msg(format!("prefetch event L{}: {}", layer_idx, e)))?;
        self.ready_events.insert(layer_idx, event);
        lane.loaded = Some(layer_idx);
        self.layer_lane.insert(layer_idx, lane_idx);
        Ok(())
    }

    /// Block the compute stream on the layer's H2D-ready event.
    pub fn wait_for_layer(&self, layer_idx: usize) -> Result<()> {
        if self.is_pinned(layer_idx) {
            return Ok(());
        }
        match self.ready_events.get(&layer_idx) {
            Some(event) => self
                .compute_stream
                .wait(event)
                .map_err(|e| TqError::Msg(format!("wait_for_layer {}: {}", layer_idx, e))),
            // No event: legitimate on first iteration of a non-pinned layer
            // that wasn't pre-prefetched. Lazy upload fills it synchronously.
            None => Ok(()),
        }
    }

    /// Drop each QWeight's Arc<CudaSlice>. The lane retains its Arc, so the
    /// slot buffer stays alive and can be `memcpy_htod`'d over next time.
    /// Record a compute-stream event on the lane so subsequent prefetches
    /// wait for the pending kernel work to complete before reusing the slot.
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
        for qw_ptr in qweight_ptrs {
            let qw = &*qw_ptr.0;
            let _ = qw.take_gpu();
        }
        self.ready_events.remove(&layer_idx);
        if let Some(lane_idx) = self.layer_lane.remove(&layer_idx) {
            // Record event NOW so the lane can't be reused until compute
            // finishes with the slice. With Arc we don't need a graveyard —
            // the slot Arc in the lane keeps the buffer alive.
            match self.compute_stream.record_event(None) {
                Ok(event) => {
                    self.lanes[lane_idx].pending_evict_event = Some(event);
                }
                Err(e) => {
                    eprintln!(
                        "[layer_swap] evict event record L{}: {} — falling back to blocking sync",
                        layer_idx, e
                    );
                    let _ = self.compute_stream.synchronize();
                }
            }
            self.lanes[lane_idx].loaded = None;
        }
    }

    /// Kept for forward-loop compatibility. No-op in lane design (lane
    /// pending_evict_event replaces the graveyard).
    pub fn flush_graveyard(&mut self) {}

    /// Kept for forward-loop compatibility. Block on every lane's pending
    /// eviction event so all prior kernel work is drained before the next
    /// forward call.
    pub fn drain_graveyard(&mut self) {
        for lane in &mut self.lanes {
            if let Some(ev) = lane.pending_evict_event.take() {
                let _ = ev.synchronize();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qweight_ptr_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QWeightPtr>();
    }
}
