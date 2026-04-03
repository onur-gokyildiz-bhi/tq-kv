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
    /// Initializes the kernel registry for GPU tensor operations.
    #[cfg(feature = "cuda")]
    pub fn cuda_if_available(ordinal: usize) -> super::Result<Self> {
        match CudaContext::new(ordinal) {
            Ok(ctx) => {
                eprintln!("CUDA device {} initialized", ordinal);
                let stream = ctx.default_stream();
                let registry = super::kernels::KernelRegistry::new(&ctx, &stream)
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

    /// Get the default stream from the CUDA context (panics on CPU).
    #[cfg(feature = "cuda")]
    pub fn cuda_stream(&self) -> super::Result<std::sync::Arc<cudarc::driver::CudaStream>> {
        match self {
            TqDevice::Cuda { context, .. } => Ok(context.default_stream()),
            _ => panic!("cuda_stream() called on CPU device"),
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
