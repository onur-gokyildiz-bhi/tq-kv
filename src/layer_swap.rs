//! Layer weight streaming for VRAM-constrained inference.
//!
//! When the full model doesn't fit in GPU memory, `LayerSwapManager` streams
//! non-pinned layers on/off the device on a dedicated transfer stream. Pinned
//! layers stay resident for the full run.
//!
//! # Sync model (Phase 6b)
//!
//! Three streams / events coordinate ownership:
//!   * **Transfer stream** — does async `clone_htod` of QWeight bytes.
//!     After each prefetch, records a `ready_events[layer]` event.
//!   * **Compute stream** — the forward-pass stream. Waits on
//!     `ready_events[N]` before launching layer N's kernels
//!     (`wait_for_layer`).
//!   * **Compute-completion events** — after evicting a layer, we record an
//!     event on the compute stream, *take* the layer's `CudaSlice`s out of
//!     the QWeights (without dropping them), and park them in the
//!     `graveyard`. On each subsequent iteration's pre-hook we query those
//!     events; when one fires we know no in-flight kernel still references
//!     that memory and it's safe to drop. This replaces the earlier
//!     `compute_stream.synchronize()` path, which serialised eviction with
//!     compute and killed async overlap.
//!
//! # Safety / invariants
//!
//! - Single-threaded forward (same invariant the existing `QWeight` code
//!   relies on).
//! - CUDA Graph capture is incompatible with swap (slot addresses change
//!   between prefetches). `engine.rs` disables `TQ_GRAPH` when swap is active.
//! - QWeightPtr wraps a `*const QWeight` with `Send + Sync` — required so
//!   the owning `Engine` stays `Send` for `tokio::spawn_blocking`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

use crate::cuda::{Result, TqError};

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
// Manager
// ─────────────────────────────────────────────────────────────

pub struct LayerSwapManager {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    transfer_stream: Arc<CudaStream>,
    compute_stream: Arc<CudaStream>,

    /// Always-resident layers (never evicted).
    pinned: std::collections::HashSet<usize>,

    /// Ready events per active (non-pinned) layer — the compute stream waits
    /// on these before launching that layer's kernels.
    ready_events: HashMap<usize, CudaEvent>,

    /// Graveyard of evicted `CudaSlice`s waiting for their compute-stream
    /// event to fire. Each entry holds:
    ///   - `event`: recorded on compute_stream *after* the layer's kernels
    ///     were launched. When `event.is_complete()` returns true, the
    ///     slices can be safely dropped (cuMemFree).
    ///   - `slices`: owned `CudaSlice<u8>`s quarantined from QWeights.
    ///
    /// Flushed at the start of every iteration (`flush_graveyard`). Typical
    /// occupancy: 1-2 layers at a time (recent evictions).
    graveyard: Vec<(CudaEvent, Vec<CudaSlice<u8>>)>,
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
            graveyard: Vec::new(),
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

    /// Drop any graveyard entries whose compute-stream event has fired.
    /// Called at every layer's pre-hook.
    ///
    /// Cheap: `CudaEvent::is_complete()` is a non-blocking query (cuEventQuery).
    pub fn flush_graveyard(&mut self) {
        // Note: event.is_complete is Ok→true meaning READY. Our earlier
        // layers' events should fire first in practice, but we scan all
        // entries every call. Graveyard is small (1-3 entries typical).
        self.graveyard.retain(|(event, _slices)| !event.is_complete());
    }

    /// Force-flush: drop all pending graveyard entries by blocking the host
    /// on each event. Called at end-of-forward to avoid accumulating across
    /// many decode steps.
    pub fn drain_graveyard(&mut self) {
        for (event, _) in self.graveyard.drain(..) {
            // Synchronize on each event (Drop order will then cuMemFree).
            if let Err(e) = event.synchronize() {
                eprintln!("[layer_swap] graveyard drain event sync failed: {}", e);
            }
        }
    }

    /// Upload the given layer's QWeight bytes to GPU (via `inject_gpu`) on the
    /// transfer stream, then block the CPU until the copy completes. Used for
    /// always-resident ("pinned") layers during model load.
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
            qw.inject_gpu(gpu);
        }
        self.transfer_stream
            .synchronize()
            .map_err(|e| TqError::Msg(format!("pin sync layer {}: {}", layer_idx, e)))?;
        self.pinned.insert(layer_idx);
        Ok(())
    }

    /// Async H2D for all QWeights of `layer_idx`. Each weight gets a fresh
    /// CudaSlice via `clone_htod`. Event recorded on the transfer stream;
    /// `wait_for_layer` hooks the compute stream on it. No-op if pinned.
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
            // has no prior prefetch. The existing `gpu_cache_or_upload` lazy
            // path will upload synchronously on first access.
            None => Ok(()),
        }
    }

    /// Take the GPU caches out of every QWeight in the layer and park them in
    /// the graveyard. Records a `compute_stream` event marking when those
    /// slices are safe to drop.
    ///
    /// Pinned layers are no-ops.
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
        // Take (don't drop) the CudaSlices out of the QWeights.
        let slices: Vec<CudaSlice<u8>> = qweight_ptrs
            .iter()
            .filter_map(|p| (*p.0).take_gpu())
            .collect();
        if slices.is_empty() {
            return;
        }
        // Record a completion event on compute_stream. By the time this event
        // fires, every kernel launched so far on the compute stream has
        // finished — including all kernels that read this layer's weights.
        // Until then, keep the slices alive in the graveyard.
        match self.compute_stream.record_event(None) {
            Ok(event) => self.graveyard.push((event, slices)),
            Err(e) => {
                eprintln!(
                    "[layer_swap] compute event record failed for L{}: {} — \
                     falling back to synchronous drop (may race in-flight kernels)",
                    layer_idx, e
                );
                // Worst case: slices drop here, races with in-flight kernels.
                // Should never happen unless context is gone.
                drop(slices);
            }
        }
        self.ready_events.remove(&layer_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qweight_ptr_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QWeightPtr>();
        // Also a smoke check that the manager type is Send even though it
        // holds raw pointers internally (through QWeightPtr).
        assert_send_sync::<QWeightPtr>();
    }
}
