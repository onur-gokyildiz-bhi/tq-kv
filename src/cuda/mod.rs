//! Custom CUDA backend for tq-kv inference engine.
//!
//! Replaces candle with direct CUDA control via cudarc.
//! CPU fallback for all operations.

mod device;
mod dtype;
mod tensor;
mod ops;
pub mod memory;
pub mod graph;
pub mod paged_kv;
#[cfg(feature = "cuda")]
pub mod kernels;

pub use device::TqDevice;
pub use dtype::TqDType;
pub use tensor::{TqTensor, TqStorage};
pub use ops::TqOps;

/// Result type for tensor operations.
pub type Result<T> = std::result::Result<T, TqError>;

/// Error type for the CUDA backend.
#[derive(Debug)]
pub enum TqError {
    /// Shape mismatch in tensor operation.
    ShapeMismatch { op: &'static str, expected: Vec<usize>, got: Vec<usize> },
    /// Dimension out of bounds.
    DimOutOfBounds { dim: usize, rank: usize },
    /// CUDA driver error.
    #[cfg(feature = "cuda")]
    Cuda(cudarc::driver::DriverError),
    /// Generic error with message.
    Msg(String),
}

impl std::fmt::Display for TqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TqError::ShapeMismatch { op, expected, got } => {
                write!(f, "{}: shape mismatch, expected {:?}, got {:?}", op, expected, got)
            }
            TqError::DimOutOfBounds { dim, rank } => {
                write!(f, "dimension {} out of bounds for rank {}", dim, rank)
            }
            #[cfg(feature = "cuda")]
            TqError::Cuda(e) => write!(f, "CUDA error: {}", e),
            TqError::Msg(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for TqError {}

impl From<String> for TqError {
    fn from(s: String) -> Self { TqError::Msg(s) }
}

impl From<&str> for TqError {
    fn from(s: &str) -> Self { TqError::Msg(s.to_string()) }
}

impl From<std::io::Error> for TqError {
    fn from(e: std::io::Error) -> Self { TqError::Msg(e.to_string()) }
}

impl From<serde_json::Error> for TqError {
    fn from(e: serde_json::Error) -> Self { TqError::Msg(e.to_string()) }
}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for TqError {
    fn from(e: cudarc::driver::DriverError) -> Self { TqError::Cuda(e) }
}

/// Bail macro for TqError (mirrors anyhow::bail).
macro_rules! tq_bail {
    ($($arg:tt)*) => { return Err($crate::cuda::TqError::Msg(format!($($arg)*))) };
}
pub(crate) use tq_bail;
