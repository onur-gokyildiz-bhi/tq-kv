//! Device abstraction — CPU or CUDA GPU.
//!
//! TqDevice::Cuda holds both the CUDA context and a shared KernelRegistry,
//! so any GPU tensor can access kernel launchers through its device.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;

/// Compute device for tensor operations.
#[derive(Clone)]
pub enum TqDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda {
        context: std::sync::Arc<CudaContext>,
        ordinal: usize,
        /// Shared kernel registry — every GPU tensor can launch kernels via this.
        registry: std::sync::Arc<super::kernels::KernelRegistry>,
    },
}

impl TqDevice {
    /// CPU device.
    pub fn cpu() -> Self {
        TqDevice::Cpu
    }

    /// Select CUDA device if available, otherwise fall back to CPU.
    /// Detects GPU compute capability and loads arch-matched PTX kernels.
    #[cfg(feature = "cuda")]
    pub fn cuda_if_available(ordinal: usize) -> super::Result<Self> {
        match CudaContext::new(ordinal) {
            Ok(ctx) => {
                // Detect GPU compute capability for multi-arch PTX selection
                let (sm_major, sm_minor) = get_compute_capability(ordinal);
                eprintln!("CUDA device {} initialized (sm_{}{})", ordinal, sm_major, sm_minor);

                // Non-default stream for CUDA Graph capture.
                // Event tracking disabled: single-stream inference, no cross-stream sync.
                let stream = ctx.new_stream()
                    .map_err(|e| super::TqError::Msg(format!("stream create: {}", e)))?;
                unsafe { ctx.disable_event_tracking(); }
                let registry = super::kernels::KernelRegistry::new(&ctx, &stream, sm_major, sm_minor)
                    .map_err(|e| super::TqError::Msg(format!("kernel init: {}", e)))?;
                let registry = std::sync::Arc::new(registry);
                // Set global registry so GPU tensor ops can access kernels
                super::kernels::set_global_registry(registry.clone());
                Ok(TqDevice::Cuda {
                    context: ctx, ordinal, registry,
                })
            }
            Err(e) => {
                eprintln!("CUDA not available ({}), falling back to CPU", e);
                Ok(TqDevice::Cpu)
            }
        }
    }

    /// Select CUDA device if available, otherwise fall back to CPU.
    #[cfg(not(feature = "cuda"))]
    pub fn cuda_if_available(_ordinal: usize) -> super::Result<Self> {
        eprintln!("CUDA support not compiled (build with --features cuda)");
        Ok(TqDevice::Cpu)
    }

    /// True if this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        match self {
            TqDevice::Cpu => false,
            #[cfg(feature = "cuda")]
            TqDevice::Cuda { .. } => true,
        }
    }

    /// True if this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        !self.is_cuda()
    }

    /// Get the cudarc context handle (panics on CPU).
    #[cfg(feature = "cuda")]
    pub fn cuda_context(&self) -> &std::sync::Arc<CudaContext> {
        match self {
            TqDevice::Cuda { context, .. } => context,
            _ => panic!("cuda_context() called on CPU device"),
        }
    }

    /// Get the compute stream (non-default, supports graph capture).
    #[cfg(feature = "cuda")]
    pub fn cuda_stream(&self) -> super::Result<std::sync::Arc<cudarc::driver::CudaStream>> {
        match self {
            TqDevice::Cuda { registry, .. } => Ok(registry.stream.clone()),
            _ => panic!("cuda_stream() called on CPU device"),
        }
    }

    /// Create a fresh CUDA stream dedicated to H2D transfers, independent of
    /// the compute stream. The manager records events on this stream after
    /// each prefetch so the compute stream can wait before launching the
    /// corresponding layer's kernels.
    ///
    /// Unlike the compute stream (which has event tracking disabled for
    /// CUDA-Graph replay), this stream keeps event tracking on — the
    /// `CudaStream::record_event` / `wait` pair depends on it.
    #[cfg(feature = "cuda")]
    pub fn new_transfer_stream(&self) -> super::Result<std::sync::Arc<cudarc::driver::CudaStream>> {
        match self {
            TqDevice::Cuda { context, .. } => context
                .new_stream()
                .map_err(|e| super::TqError::Msg(format!("transfer stream create: {}", e))),
            _ => Err(super::TqError::Msg("new_transfer_stream on CPU device".into())),
        }
    }
}

impl std::fmt::Debug for TqDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TqDevice::Cpu => write!(f, "Cpu"),
            #[cfg(feature = "cuda")]
            TqDevice::Cuda { ordinal, .. } => write!(f, "Cuda({})", ordinal),
        }
    }
}

#[cfg(feature = "cuda")]
impl TqDevice {
    /// Get the kernel registry for GPU tensor operations.
    pub fn registry(&self) -> &super::kernels::KernelRegistry {
        match self {
            TqDevice::Cuda { registry, .. } => registry,
            _ => panic!("registry() called on CPU device"),
        }
    }
}

impl PartialEq for TqDevice {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TqDevice::Cpu, TqDevice::Cpu) => true,
            #[cfg(feature = "cuda")]
            (TqDevice::Cuda { ordinal: a, .. }, TqDevice::Cuda { ordinal: b, .. }) => a == b,  // registry ignored for equality
            #[cfg(feature = "cuda")]
            _ => false,
        }
    }
}

/// Query GPU compute capability via CUDA driver API.
/// Returns (sm_major, sm_minor), e.g. (8, 6) for RTX 3080.
/// Falls back to (8, 6) if the query fails.
#[cfg(feature = "cuda")]
fn get_compute_capability(ordinal: usize) -> (u32, u32) {
    use cudarc::driver::sys;
    let mut device: sys::CUdevice = 0;
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    unsafe {
        let res = sys::cuDeviceGet(&mut device, ordinal as i32);
        if res != sys::cudaError_enum::CUDA_SUCCESS {
            eprintln!("[cuda] WARNING: cuDeviceGet failed ({:?}), assuming sm_86", res);
            return (8, 6);
        }
        let res = sys::cuDeviceGetAttribute(
            &mut major,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        );
        if res != sys::cudaError_enum::CUDA_SUCCESS {
            eprintln!("[cuda] WARNING: cuDeviceGetAttribute(major) failed ({:?}), assuming sm_86", res);
            return (8, 6);
        }
        let res = sys::cuDeviceGetAttribute(
            &mut minor,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        );
        if res != sys::cudaError_enum::CUDA_SUCCESS {
            eprintln!("[cuda] WARNING: cuDeviceGetAttribute(minor) failed ({:?}), assuming sm_86", res);
            return (8, 6);
        }
    }
    (major as u32, minor as u32)
}
