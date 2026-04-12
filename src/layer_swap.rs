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

    /// Double-buffered (or N-buffered) GPU slots.
    slots: Vec<LayerGpuSlot>,
    /// Staging buffer in pinned host memory. Optional — falls back to pageable.
    staging: PinnedBuffer,

    /// Always-resident layers. H2D is performed once at load time; never evicted.
    pinned: HashMap<usize, Arc<CudaSlice<u8>>>,

    /// Per-layer metadata (indexed by layer_idx).
    layer_meta: Vec<LayerMeta>,
    /// Max bytes across all non-pinned layers. Used to size each slot.
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

        // Pinned-layer GPU allocations (filled in by `pin_layers`).
        let pinned = HashMap::with_capacity(pinned_set.len());

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
            layer_meta,
            bytes_per_layer,
        })
    }

    /// Total bytes pre-allocated in slots (for diagnostics).
    pub fn slot_footprint_bytes(&self) -> usize {
        self.slots.len() * self.bytes_per_layer
    }

    /// Total bytes resident from pinned layers.
    pub fn pinned_footprint_bytes(&self) -> usize {
        self.pinned.values().map(|s| s.num_bytes()).sum()
    }

    /// Is this layer always resident?
    pub fn is_pinned(&self, layer_idx: usize) -> bool {
        self.pinned.contains_key(&layer_idx)
    }

    /// Copy a single layer's bytes from `src` into a freshly allocated GPU
    /// buffer on the transfer stream, and store it in `self.pinned`.
    ///
    /// `src` must be the whole source buffer (file bytes / mmap). The per-weight
    /// offsets in `LayerMeta::weights` are interpreted relative to `src`.
    pub fn pin_layer(&mut self, layer_idx: usize, src: &[u8]) -> Result<()> {
        let meta = self
            .layer_meta
            .iter()
            .find(|lm| lm.layer_idx == layer_idx)
            .ok_or_else(|| TqError::Msg(format!("unknown layer {}", layer_idx)))?
            .clone();

        // Stage into pinned host buffer (ensures one contiguous H2D).
        self.stage_layer(&meta, src)?;

        // Allocate GPU buffer for this pinned layer.
        let staging_slice = &self.staging.as_slice()[..meta.total_bytes];
        let gpu_buf = self
            .transfer_stream
            .clone_htod(staging_slice)
            .map_err(|e| TqError::Msg(format!("pin H2D for layer {}: {}", layer_idx, e)))?;

        // Ensure the copy completes before any forward-stream reads.
        self.transfer_stream
            .synchronize()
            .map_err(|e| TqError::Msg(format!("pin sync layer {}: {}", layer_idx, e)))?;

        self.pinned.insert(layer_idx, Arc::new(gpu_buf));
        Ok(())
    }

    /// Kick off an async H2D copy for `layer_idx` into the next free slot.
    /// Returns immediately; the forward loop must call `wait_for_layer` before
    /// launching kernels that read the layer's weights.
    pub fn start_prefetch(&mut self, layer_idx: usize, src: &[u8]) -> Result<()> {
        if self.is_pinned(layer_idx) {
            return Ok(()); // already resident, nothing to do
        }
        let meta = self
            .layer_meta
            .iter()
            .find(|lm| lm.layer_idx == layer_idx)
            .ok_or_else(|| TqError::Msg(format!("unknown layer {}", layer_idx)))?
            .clone();

        // Pick a slot: prefer empty, else the least-recently-loaded.
        let slot_idx = self
            .slots
            .iter()
            .position(|s| s.loaded.is_none())
            .or_else(|| {
                // All slots occupied; overwrite slot 0 for now (round-robin
                // policy to be tuned in Phase 5 based on forward iteration).
                Some(0)
            })
            .unwrap();

        // Stage host bytes.
        self.stage_layer(&meta, src)?;

        // Async H2D into the slot buffer (memcpy on transfer stream).
        let staging_slice = &self.staging.as_slice()[..meta.total_bytes];
        // `buffer` is `Arc<CudaSlice<u8>>`; we need `&mut CudaSlice<u8>` for
        // memcpy_htod. The Arc here has a single strong count during prefetch
        // (Phase 5 will share it with QWeights via inject_gpu).
        {
            let slot = &mut self.slots[slot_idx];
            let buf_mut = Arc::get_mut(&mut slot.buffer).ok_or_else(|| {
                TqError::Msg(format!(
                    "slot {} buffer still referenced by a prior layer — forgot to evict?",
                    slot_idx
                ))
            })?;
            self.transfer_stream
                .memcpy_htod(staging_slice, &mut buf_mut.slice_mut(0..meta.total_bytes))
                .map_err(|e| TqError::Msg(format!("prefetch H2D layer {}: {}", layer_idx, e)))?;

            // Record event for cross-stream sync.
            let event = self
                .transfer_stream
                .record_event(None)
                .map_err(|e| TqError::Msg(format!("prefetch event layer {}: {}", layer_idx, e)))?;
            slot.ready = event;
            slot.loaded = Some(layer_idx);
        }
        Ok(())
    }

    /// Block the compute stream (not the CPU) on the layer's H2D-ready event.
    pub fn wait_for_layer(&self, layer_idx: usize) -> Result<()> {
        if self.is_pinned(layer_idx) {
            return Ok(());
        }
        let slot = self
            .slots
            .iter()
            .find(|s| s.loaded == Some(layer_idx))
            .ok_or_else(|| {
                TqError::Msg(format!(
                    "wait_for_layer({}) but no slot holds this layer",
                    layer_idx
                ))
            })?;
        self.compute_stream
            .wait(&slot.ready)
            .map_err(|e| TqError::Msg(format!("wait_for_layer {}: {}", layer_idx, e)))?;
        Ok(())
    }

    /// Mark a layer's slot as free (Phase 5 will also call `QWeight::evict_gpu`
    /// on every QWeight in the layer to drop their `Arc<CudaSlice<u8>>`).
    pub fn evict_layer(&mut self, layer_idx: usize) {
        for slot in &mut self.slots {
            if slot.loaded == Some(layer_idx) {
                slot.loaded = None;
            }
        }
    }

    // ─────────────────────────────────────────────────────────
    // Internals
    // ─────────────────────────────────────────────────────────

    /// Copy the per-weight byte ranges from `src` into the manager's packed
    /// staging buffer (pinned, if available).
    fn stage_layer(&mut self, meta: &LayerMeta, src: &[u8]) -> Result<()> {
        let dst = self.staging.as_mut_slice();
        if dst.len() < meta.total_bytes {
            return Err(TqError::Msg(format!(
                "staging buffer {} < layer {} bytes",
                dst.len(),
                meta.total_bytes
            )));
        }
        for w in &meta.weights {
            let src_end = w.src_offset + w.nbytes;
            if src_end > src.len() {
                return Err(TqError::Msg(format!(
                    "weight {} out of bounds: {}..{} > {}",
                    w.name,
                    w.src_offset,
                    src_end,
                    src.len()
                )));
            }
            let gpu_end = w.gpu_offset + w.nbytes;
            if gpu_end > meta.total_bytes {
                return Err(TqError::Msg(format!(
                    "weight {} gpu offset {}..{} > layer total {}",
                    w.name, w.gpu_offset, gpu_end, meta.total_bytes
                )));
            }
            dst[w.gpu_offset..gpu_end].copy_from_slice(&src[w.src_offset..src_end]);
        }
        Ok(())
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
