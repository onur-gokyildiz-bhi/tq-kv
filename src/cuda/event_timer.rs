//! Accurate per-section GPU timing via CUDA events.
//!
//! # Why this exists
//!
//! The `Instant::now() + stream.synchronize()` pattern used elsewhere in the
//! profile path measures host wall-time from one sync point to the next —
//! which silently absorbs any GPU work queued earlier but not yet drained.
//! A section timed that way reports `enqueue_latency + tail_drain_of_prior +
//! current_kernels_gpu_time`. On the 2026-04-17 `fused_q4km_down_residual_v3`
//! investigation this artifact reported a 360 μs section whose actual GPU
//! cost was ~100 μs — two other dispatches were pipelined behind it.
//!
//! `EventTimer` wraps the CUDA event API to record `start` and `end` events
//! in stream order. `cuEventElapsedTime` then returns the GPU-time elapsed
//! between the events — everything launched before `start` has already
//! drained by the time `start` itself executes, so the measurement is
//! isolated to what was launched between the two `.record` calls.
//!
//! # Usage
//!
//! ```ignore
//! let timer = EventTimer::new()?;
//! timer.start(stream)?;
//! // ... launch kernels ...
//! timer.stop(stream)?;
//! let ms = timer.elapsed_ms()?;   // blocks until `stop` completes
//! ```
//!
//! # Cost
//!
//! Event create + record is sub-microsecond on the host. `elapsed_ms` does
//! block on the `stop` event (that's the unavoidable sync). The overall
//! profile overhead is negligible compared to the kernels being measured.

use cudarc::driver::{sys, CudaStream, DriverError};

/// RAII-managed start/end CUDA event pair for one-shot GPU-time measurement.
///
/// The pair can be reused within the same context: each call to `start`
/// re-records on the given stream, and `stop` / `elapsed_ms` follow suit.
pub struct EventTimer {
    start_evt: sys::CUevent,
    stop_evt:  sys::CUevent,
}

// CUevent handles are process-wide and safe to send between threads as long
// as the CUDA context is current. Our inference runs on a single thread, so
// this impl is trivially satisfied.
unsafe impl Send for EventTimer {}
unsafe impl Sync for EventTimer {}

impl EventTimer {
    /// Create a fresh event pair. Uses the default flag (`CU_EVENT_DEFAULT`) so
    /// events keep timing information — `CU_EVENT_DISABLE_TIMING` would be
    /// faster but `cuEventElapsedTime` requires timing-enabled events.
    pub fn new() -> Result<Self, DriverError> {
        let mut start_evt: sys::CUevent = std::ptr::null_mut();
        let mut stop_evt:  sys::CUevent = std::ptr::null_mut();
        unsafe {
            let r1 = sys::cuEventCreate(&mut start_evt, 0);
            if r1 != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(DriverError(r1));
            }
            let r2 = sys::cuEventCreate(&mut stop_evt, 0);
            if r2 != sys::cudaError_enum::CUDA_SUCCESS {
                let _ = sys::cuEventDestroy_v2(start_evt);
                return Err(DriverError(r2));
            }
        }
        Ok(Self { start_evt, stop_evt })
    }

    /// Enqueue the `start` event into `stream`. Non-blocking on host.
    pub fn start(&self, stream: &CudaStream) -> Result<(), DriverError> {
        let raw = stream.cu_stream();
        let r = unsafe { sys::cuEventRecord(self.start_evt, raw) };
        if r != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(DriverError(r));
        }
        Ok(())
    }

    /// Enqueue the `stop` event into `stream`. Non-blocking on host.
    pub fn stop(&self, stream: &CudaStream) -> Result<(), DriverError> {
        let raw = stream.cu_stream();
        let r = unsafe { sys::cuEventRecord(self.stop_evt, raw) };
        if r != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(DriverError(r));
        }
        Ok(())
    }

    /// Block on `stop` event and return elapsed GPU-time in milliseconds.
    /// Only meaningful if both `start` and `stop` have been recorded.
    pub fn elapsed_ms(&self) -> Result<f32, DriverError> {
        // Wait for the stop event to complete — cuEventElapsedTime would
        // return an error otherwise.
        let r_sync = unsafe { sys::cuEventSynchronize(self.stop_evt) };
        if r_sync != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(DriverError(r_sync));
        }
        let mut ms: f32 = 0.0;
        // cudarc gates the v1 symbol `cuEventElapsedTime` to CUDA 11.4–12.9
        // and ships `_v2` for 12.8+. Project targets CUDA 13.2+ per
        // CLAUDE.md, so v2 is the only variant exposed on our builds.
        // If we ever downshift to CUDA 12.0–12.7, swap back to v1.
        let r = unsafe { sys::cuEventElapsedTime_v2(&mut ms, self.start_evt, self.stop_evt) };
        if r != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(DriverError(r));
        }
        Ok(ms)
    }

    /// Convenience: elapsed time in microseconds (what the existing profile
    /// log format expects).
    pub fn elapsed_us(&self) -> Result<f64, DriverError> {
        self.elapsed_ms().map(|ms| ms as f64 * 1000.0)
    }
}

impl Drop for EventTimer {
    fn drop(&mut self) {
        unsafe {
            if !self.start_evt.is_null() {
                let _ = sys::cuEventDestroy_v2(self.start_evt);
            }
            if !self.stop_evt.is_null() {
                let _ = sys::cuEventDestroy_v2(self.stop_evt);
            }
        }
    }
}

/// Ergonomic one-shot helper: record start → run closure → record stop →
/// return elapsed microseconds. Avoids boilerplate at the call site.
///
/// Returns `(result_of_f, elapsed_us)`; on event failure, elapsed is `None`.
///
/// ```ignore
/// let (result, us) = time_section(&stream, || {
///     crate::cuda::kernels::q6k_matvec(...)
/// });
/// eprintln!("q6k_matvec: {:.1}μs", us.unwrap_or(0.0));
/// ```
pub fn time_section<F, R>(stream: &CudaStream, f: F) -> (R, Option<f64>)
where
    F: FnOnce() -> R,
{
    let timer = match EventTimer::new() {
        Ok(t) => t,
        Err(_) => return (f(), None),
    };
    if timer.start(stream).is_err() {
        return (f(), None);
    }
    let result = f();
    if timer.stop(stream).is_err() {
        return (result, None);
    }
    let us = timer.elapsed_us().ok();
    (result, us)
}
