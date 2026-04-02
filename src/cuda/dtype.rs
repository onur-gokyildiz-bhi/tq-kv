//! Data type enum for tensor storage.

/// Supported tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TqDType {
    F16,
    BF16,
    F32,
    U8,
    U32,
}

impl TqDType {
    /// Size in bytes per element.
    pub fn size_in_bytes(self) -> usize {
        match self {
            TqDType::F16 | TqDType::BF16 => 2,
            TqDType::F32 | TqDType::U32 => 4,
            TqDType::U8 => 1,
        }
    }
}

impl std::fmt::Display for TqDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TqDType::F16 => write!(f, "f16"),
            TqDType::BF16 => write!(f, "bf16"),
            TqDType::F32 => write!(f, "f32"),
            TqDType::U8 => write!(f, "u8"),
            TqDType::U32 => write!(f, "u32"),
        }
    }
}
