//! CUDA Graph capture and replay — eliminates per-kernel launch overhead.
//!
//! Captures the entire decode step (all layers) as a single CUDA graph.
//! On replay, the full forward pass runs as one GPU operation with zero
//! CPU-side dispatch. Measured 2.3x decode speedup on typical models.
//!
//! Usage:
//! 1. First token after prefill: run in eager mode (capture disabled)
//! 2. Second token: begin_capture → run forward → end_capture → graph is ready
//! 3. All subsequent tokens: graph.launch() — bypasses entire CPU dispatch
//!
//! Reference: vLLM captures 35 batch sizes. Without graphs: 30→69 tok/s.

/// Status of a captured CUDA graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphStatus {
    /// Not captured yet — running in eager mode.
    NotCaptured,
    /// Currently capturing operations.
    Capturing,
    /// Captured and ready to replay.
    Ready,
}

/// CUDA Graph manager for decode step acceleration.
///
/// Pre-captures the decode forward pass for different batch sizes.
/// On replay, the entire decode step runs as a single GPU operation.
pub struct CudaGraphManager {
    /// Map from batch_size → captured graph.
    #[cfg(feature = "cuda")]
    captured_graphs: std::collections::HashMap<usize, cudarc::driver::CudaGraph>,
    #[cfg(not(feature = "cuda"))]
    captured_graphs: std::collections::HashMap<usize, ()>,
    /// Supported batch sizes (pre-captured at model load).
    supported_batch_sizes: Vec<usize>,
    /// Current capture status.
    pub status: GraphStatus,
    /// Whether graph capture is enabled.
    pub enabled: bool,
    /// Warmup count: how many eager runs before capture (default: 1).
    pub warmup_runs: usize,
    /// Current eager run count.
    eager_count: usize,
    /// Non-default stream for graph capture (created on demand).
    /// Default/null stream doesn't support capture.
    #[cfg(feature = "cuda")]
    capture_stream: Option<std::sync::Arc<cudarc::driver::CudaStream>>,
}

// SAFETY: CudaGraph is bound to a single GPU context. We only access
// CudaGraphManager from the inference thread (single-threaded model forward).
// The raw CUDA pointers don't cross threads in practice.
unsafe impl Send for CudaGraphManager {}
unsafe impl Sync for CudaGraphManager {}

impl CudaGraphManager {
    /// Create a new graph manager.
    pub fn new(enabled: bool) -> Self {
        let supported = vec![1, 2, 4, 8, 16, 32, 64];
        Self {
            captured_graphs: std::collections::HashMap::new(),
            supported_batch_sizes: supported,
            status: GraphStatus::NotCaptured,
            enabled,
            warmup_runs: 1,
            eager_count: 0,
            #[cfg(feature = "cuda")]
            capture_stream: None,
        }
    }

    /// Find the nearest supported batch size (pad up).
    pub fn nearest_batch_size(&self, actual: usize) -> usize {
        for &s in &self.supported_batch_sizes {
            if s >= actual { return s; }
        }
        *self.supported_batch_sizes.last().unwrap_or(&1)
    }

    /// Check if a graph is ready for the given batch size.
    pub fn is_ready(&self, batch_size: usize) -> bool {
        self.captured_graphs.contains_key(&batch_size) && self.status == GraphStatus::Ready
    }

    /// Should we begin capture on this forward pass?
    pub fn should_capture(&mut self, batch_size: usize) -> bool {
        if !self.enabled || self.is_ready(batch_size) || self.status == GraphStatus::Capturing {
            return false;
        }
        self.eager_count += 1;
        self.eager_count > self.warmup_runs
    }

    /// Begin graph capture on a CUDA stream.
    #[cfg(feature = "cuda")]
    pub fn begin_capture(&mut self, stream: &cudarc::driver::CudaStream) -> Result<(), super::TqError> {
        use cudarc::driver::sys::CUstreamCaptureMode;
        // Sync stream and disable event tracking during capture.
        // Event tracking creates cross-stream dependencies that break capture.
        stream.synchronize()
            .map_err(|e| super::TqError::Msg(format!("graph pre-sync: {}", e)))?;
        unsafe { stream.context().disable_event_tracking(); }
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| super::TqError::Msg(format!("graph begin_capture: {}", e)))?;
        self.status = GraphStatus::Capturing;
        Ok(())
    }

    /// End graph capture, store the captured graph.
    #[cfg(feature = "cuda")]
    pub fn end_capture(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        batch_size: usize,
    ) -> Result<(), super::TqError> {
        use cudarc::driver::sys::CUgraphInstantiate_flags_enum;
        let result = stream.end_capture(CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        // Re-enable event tracking after capture
        unsafe { stream.context().enable_event_tracking(); }
        let graph = result.map_err(|e| super::TqError::Msg(format!("graph end_capture: {}", e)))?;
        if let Some(graph) = graph {
            // Pre-upload to avoid first-launch overhead
            graph.upload()
                .map_err(|e| super::TqError::Msg(format!("graph upload: {}", e)))?;
            self.captured_graphs.insert(batch_size, graph);
            self.status = GraphStatus::Ready;
            eprintln!("[cuda-graph] Captured decode graph for batch_size={}", batch_size);
        } else {
            self.status = GraphStatus::NotCaptured;
            eprintln!("[cuda-graph] Capture produced empty graph");
        }
        Ok(())
    }

    /// Replay a captured graph — runs the entire decode step as one GPU op.
    #[cfg(feature = "cuda")]
    pub fn replay(&self, batch_size: usize) -> Result<(), super::TqError> {
        let graph = self.captured_graphs.get(&batch_size)
            .ok_or_else(|| super::TqError::Msg(format!("no graph for batch_size={}", batch_size)))?;
        graph.launch()
            .map_err(|e| super::TqError::Msg(format!("graph launch: {}", e)))
    }

    /// Reset graph state (e.g., after KV cache clear).
    pub fn reset(&mut self) {
        self.captured_graphs.clear();
        self.status = GraphStatus::NotCaptured;
        self.eager_count = 0;
    }
}
