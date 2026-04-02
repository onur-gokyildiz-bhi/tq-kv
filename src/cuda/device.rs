//! Device abstraction — CPU or CUDA GPU.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice as CudarcDevice;

/// Compute device for tensor operations.
#[derive(Clone, Debug)]
pub enum TqDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda {
        device: std::sync::Arc<CudarcDevice>,
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
        match CudarcDevice::new(ordinal) {
            Ok(dev) => {
                let name = dev.name().unwrap_or_else(|_| "unknown".into());
                let (free, total) = dev.mem_info().unwrap_or((0, 0));
                eprintln!(
                    "CUDA device {}: {} ({:.1} GB free / {:.1} GB total)",
                    ordinal, name,
                    free as f64 / 1e9, total as f64 / 1e9,
                );
                Ok(TqDevice::Cuda { device: dev, ordinal })
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

    /// Get the cudarc device handle (panics on CPU).
    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> &std::sync::Arc<CudarcDevice> {
        match self {
            TqDevice::Cuda { device, .. } => device,
            _ => panic!("cuda_device() called on CPU device"),
        }
    }
}

impl PartialEq for TqDevice {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TqDevice::Cpu, TqDevice::Cpu) => true,
            #[cfg(feature = "cuda")]
            (TqDevice::Cuda { ordinal: a, .. }, TqDevice::Cuda { ordinal: b, .. }) => a == b,
            _ => false,
        }
    }
}
