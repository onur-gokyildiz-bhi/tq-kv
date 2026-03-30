//! PyO3 Python bindings for tq-kv.
//!
//! Build with: `maturin develop --features python`
//! or: `cargo build --release --features python`
//!
//! ```python
//! import tq_kv
//!
//! config = tq_kv.TurboQuantConfig(bits=4)
//! compressed = tq_kv.compress_keys(key_data, dim=128, config=config)
//! decompressed = tq_kv.decompress_keys(compressed, config=config)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyList;

/// TurboQuant configuration.
#[pyclass]
#[derive(Clone)]
struct TurboQuantConfig {
    inner: crate::TurboQuantConfig,
}

#[pymethods]
impl TurboQuantConfig {
    #[new]
    #[pyo3(signature = (bits=4, group_size=32, residual_bits=0, outlier_k=0))]
    fn new(bits: u8, group_size: usize, residual_bits: u8, outlier_k: usize) -> Self {
        let mut inner = match bits {
            2 => crate::TurboQuantConfig::extreme(),
            3 => crate::TurboQuantConfig::aggressive(),
            _ => crate::TurboQuantConfig::balanced(),
        };
        inner.group_size = group_size;
        inner.residual_bits = residual_bits;
        inner.outlier_k = outlier_k;
        Self { inner }
    }

    /// 2-bit extreme compression preset.
    #[staticmethod]
    fn extreme() -> Self {
        Self { inner: crate::TurboQuantConfig::extreme() }
    }

    /// 4-bit balanced preset.
    #[staticmethod]
    fn balanced() -> Self {
        Self { inner: crate::TurboQuantConfig::balanced() }
    }

    fn __repr__(&self) -> String {
        format!("TurboQuantConfig(bits={}, group_size={}, residual_bits={}, outlier_k={})",
            self.inner.bits, self.inner.group_size, self.inner.residual_bits, self.inner.outlier_k)
    }
}

/// Compressed keys container.
#[pyclass]
struct CompressedKeys {
    inner: crate::CompressedKeys,
}

#[pymethods]
impl CompressedKeys {
    /// Number of compressed vectors.
    #[getter]
    fn count(&self) -> usize { self.inner.count }

    /// Vector dimension.
    #[getter]
    fn dim(&self) -> usize { self.inner.dim }

    /// Bit width.
    #[getter]
    fn bits(&self) -> u8 { self.inner.bits }

    /// Compression ratio vs fp16.
    fn compression_ratio(&self) -> f32 { self.inner.compression_ratio() }

    /// Compressed size in bytes.
    fn memory_bytes(&self) -> usize { self.inner.memory_bytes() }

    fn __repr__(&self) -> String {
        format!("CompressedKeys(count={}, dim={}, bits={}, ratio={:.1}x)",
            self.inner.count, self.inner.dim, self.inner.bits, self.inner.compression_ratio())
    }
}

/// Compress key vectors.
///
/// Args:
///     data: flat f32 list/array of key vectors (length = count * dim)
///     dim: vector dimension (must be power of 2)
///     config: TurboQuantConfig
///
/// Returns:
///     CompressedKeys object
#[pyfunction]
fn compress_keys(data: Vec<f32>, dim: usize, config: &TurboQuantConfig) -> PyResult<CompressedKeys> {
    if data.len() % dim != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("data length {} not divisible by dim {}", data.len(), dim)
        ));
    }
    let inner = crate::compress_keys(&data, dim, &config.inner);
    Ok(CompressedKeys { inner })
}

/// Decompress keys back to f32 list.
#[pyfunction]
fn decompress_keys(compressed: &CompressedKeys, config: &TurboQuantConfig) -> Vec<f32> {
    if compressed.inner.group_size > 0 {
        crate::decompress_keys_grouped(&compressed.inner, &config.inner)
    } else {
        crate::decompress_keys(&compressed.inner, &config.inner)
    }
}

/// Evaluate compression quality (MSE, SNR, ratio, max_error).
#[pyfunction]
fn evaluate_keys(original: Vec<f32>, compressed: &CompressedKeys, config: &TurboQuantConfig) -> PyResult<(f32, f32, f32, f32)> {
    let stats = crate::evaluate_keys(&original, &compressed.inner, &config.inner);
    Ok((stats.mse, stats.snr_db, stats.ratio, stats.max_error))
}

/// Calibrate codebook from sample data.
///
/// Returns centroids as a list of f32.
#[pyfunction]
fn calibrate_codebook(data: Vec<f32>, dim: usize, bits: u8) -> Vec<f32> {
    let cb = crate::calibrate_codebook(&data, dim, bits, 0x0054_5552_4230);
    cb.centroids
}

/// Python module definition.
#[pymodule]
fn tq_kv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantConfig>()?;
    m.add_class::<CompressedKeys>()?;
    m.add_function(wrap_pyfunction!(compress_keys, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_keys, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_keys, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_codebook, m)?)?;
    Ok(())
}
