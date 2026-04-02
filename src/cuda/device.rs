//! Device abstraction — CPU or CUDA GPU.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;

/// Compute device for tensor operations.
#[derive(Clone, Debug)]
pub enum TqDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda {
        context: std::sync::Arc<CudaContext>,
        ordinal: usize,
    },
}

impl TqDevice {
    /// CPU device.
    pub fn cpu() -> Self {
        TqDevice::Cpu
    }

    /// Select CUDA device if available, otherwise fall back to CPU.
    #[cfg(feature = "cuda")]
    pub fn cuda_if_available(ordinal: usize) -> super::Result<Self> {
        match CudaContext::new(ordinal) {
            Ok(ctx) => {
                eprintln!("CUDA device {} initialized", ordinal);
                Ok(TqDevice::Cuda { context: ctx, ordinal })
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

impl PartialEq for TqDevice {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TqDevice::Cpu, TqDevice::Cpu) => true,
            #[cfg(feature = "cuda")]
            (TqDevice::Cuda { ordinal: a, .. }, TqDevice::Cuda { ordinal: b, .. }) => a == b,
            #[cfg(feature = "cuda")]
            _ => false,
        }
    }
}
