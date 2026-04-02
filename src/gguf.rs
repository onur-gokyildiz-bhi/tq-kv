//! GGUF v3 file parser — standalone replacement for candle's gguf_file.
//!
//! Reads .gguf binary files: header, metadata, tensor descriptors, quantized tensor data.
//! Compatible with llama.cpp GGUF v3 format (magic 0x46554747).

use std::collections::HashMap;
use std::io::{Read, Write, Seek, SeekFrom};

use crate::cuda::{TqDevice, TqError, Result};
use crate::qmatmul::QWeight;

// ─── Constants ─────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32;

// ─── GGML Quantization Types ──────────────────────────────────

/// GGML tensor data type (quantization format).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    BF16 = 30,
    /// TQ4_1S: TurboQuant 4-bit weight compression (WHT + 16 Lloyd-Max centroids).
    /// 32 elements → 20 bytes (5.0 BPW). Custom type, not in upstream GGML.
    /// Uses high range (50000+) to avoid collision with future GGML type IDs.
    TQ4_1S = 50001,
}

impl GgmlDType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(GgmlDType::F32),
            1 => Ok(GgmlDType::F16),
            2 => Ok(GgmlDType::Q4_0),
            3 => Ok(GgmlDType::Q4_1),
            6 => Ok(GgmlDType::Q5_0),
            7 => Ok(GgmlDType::Q5_1),
            8 => Ok(GgmlDType::Q8_0),
            9 => Ok(GgmlDType::Q8_1),
            10 => Ok(GgmlDType::Q2K),
            11 => Ok(GgmlDType::Q3K),
            12 => Ok(GgmlDType::Q4K),
            13 => Ok(GgmlDType::Q5K),
            14 => Ok(GgmlDType::Q6K),
            15 => Ok(GgmlDType::Q8K),
            30 => Ok(GgmlDType::BF16),
            50001 => Ok(GgmlDType::TQ4_1S),
            _ => Err(TqError::Msg(format!("unknown GGML dtype: {}", v))),
        }
    }

    /// Block size (number of elements per quantized block).
    pub fn block_numel(self) -> usize {
        match self {
            GgmlDType::F32 => 1,
            GgmlDType::F16 | GgmlDType::BF16 => 1,
            GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 |
            GgmlDType::Q5_1 | GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::TQ4_1S => 32,
            GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K |
            GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8K => 256,
        }
    }

    /// Bytes per block.
    pub fn block_size_bytes(self) -> usize {
        match self {
            GgmlDType::F32 => 4,
            GgmlDType::F16 | GgmlDType::BF16 => 2,
            GgmlDType::Q4_0 => 18,     // 2 (f16 d) + 16 (4-bit data)
            GgmlDType::Q4_1 => 20,     // 2+2 (f16 d,m) + 16 (4-bit data)
            GgmlDType::Q5_0 => 22,     // 2 + 4 (qh) + 16
            GgmlDType::Q5_1 => 24,     // 2+2 + 4 + 16
            GgmlDType::Q8_0 => 34,     // 2 (f16 d) + 32 (i8)
            GgmlDType::Q8_1 => 40,     // 4+4 (f32 d,s) + 32 (i8)
            GgmlDType::Q2K => 84,
            GgmlDType::Q3K => 110,
            GgmlDType::Q4K => 144,     // 2+2+12+128 = 144
            GgmlDType::Q5K => 176,     // 2+2+12+32+128 = 176
            GgmlDType::Q6K => 210,     // 128+64+16+2 = 210
            GgmlDType::Q8K => 292,     // 4+256+32 = 292
            GgmlDType::TQ4_1S => 20,  // 2+2+16 = 20 (scale:f16 + offset:f16 + nibbles:16B)
        }
    }

    /// Total bytes for `n_elements` values.
    pub fn tensor_size_bytes(self, n_elements: usize) -> usize {
        let numel = self.block_numel();
        let n_blocks = (n_elements + numel - 1) / numel;
        n_blocks * self.block_size_bytes()
    }
}

// ─── Metadata Value ───────────────────────────────────────────

/// GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn to_u32(&self) -> Result<u32> {
        match self {
            GgufValue::U32(v) => Ok(*v),
            GgufValue::U8(v) => Ok(*v as u32),
            GgufValue::U16(v) => Ok(*v as u32),
            GgufValue::I32(v) => Ok(*v as u32),
            GgufValue::U64(v) => Ok(*v as u32),
            _ => Err(TqError::Msg(format!("cannot convert {:?} to u32", self))),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            GgufValue::F32(v) => Ok(*v),
            GgufValue::F64(v) => Ok(*v as f32),
            _ => Err(TqError::Msg(format!("cannot convert {:?} to f32", self))),
        }
    }

    pub fn to_string_val(&self) -> Result<String> {
        match self {
            GgufValue::String(v) => Ok(v.clone()),
            _ => Err(TqError::Msg(format!("cannot convert {:?} to string", self))),
        }
    }

    pub fn to_u8(&self) -> Result<u8> {
        match self {
            GgufValue::U8(v) => Ok(*v),
            GgufValue::U16(v) => Ok(*v as u8),
            GgufValue::U32(v) => Ok(*v as u8),
            _ => Err(TqError::Msg(format!("cannot convert {:?} to u8", self))),
        }
    }
}

// ─── Tensor Info ──────────────────────────────────────────────

/// Descriptor for a tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Shape dimensions (standard order: outermost first).
    pub shape: Vec<usize>,
    pub dtype: GgmlDType,
    /// Byte offset from start of tensor data section.
    pub offset: u64,
    /// Total number of elements.
    pub n_elements: usize,
}

impl TensorInfo {
    pub fn data_size_bytes(&self) -> usize {
        self.dtype.tensor_size_bytes(self.n_elements)
    }
}

// ─── GGUF Content ─────────────────────────────────────────────

/// Parsed GGUF file content: metadata + tensor descriptors.
///
/// Call `tensor()` to load individual tensor data from the file.
pub struct GgufContent {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    /// Byte offset where tensor data begins (after header + metadata + descriptors + alignment).
    tensor_data_offset: u64,
}

impl GgufContent {
    /// Parse a GGUF file header, metadata, and tensor descriptors.
    ///
    /// Does NOT read tensor data — call `tensor()` for lazy loading.
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // ─── Header ───
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(TqError::Msg(format!(
                "not a GGUF file: magic 0x{:08x} != 0x{:08x}", magic, GGUF_MAGIC
            )));
        }

        let version = read_u32(reader)?;
        if version < 1 || version > 3 {
            return Err(TqError::Msg(format!("unsupported GGUF version: {}", version)));
        }

        let tensor_count = read_u64_or_u32(reader, version)? as usize;
        let metadata_kv_count = read_u64_or_u32(reader, version)? as usize;

        // ─── Metadata ───
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_string(reader, version)?;
            let value = read_value(reader, version)?;
            metadata.insert(key, value);
        }

        // ─── Tensor descriptors ───
        let mut tensor_infos = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(reader, version)?;
            let n_dims = read_u32(reader)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64_or_u32(reader, version)? as usize);
            }
            let dtype = GgmlDType::from_u32(read_u32(reader)?)?;
            let offset = read_u64(reader)?;

            // GGUF stores dimensions in reverse order (innermost first)
            dims.reverse();

            let n_elements: usize = dims.iter().product();

            tensor_infos.insert(name.clone(), TensorInfo {
                name,
                shape: dims,
                dtype,
                offset,
                n_elements,
            });
        }

        // ─── Alignment ───
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(DEFAULT_ALIGNMENT as u32) as usize;

        let current_pos = reader.stream_position()
            .map_err(|e| TqError::Msg(format!("seek error: {}", e)))? as usize;
        let tensor_data_offset = ((current_pos + alignment - 1) / alignment * alignment) as u64;

        Ok(GgufContent {
            version,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    /// Load raw tensor bytes from the file.
    pub fn tensor_data<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
    ) -> Result<(TensorInfo, Vec<u8>)> {
        let info = self.tensor_infos.get(name)
            .ok_or_else(|| TqError::Msg(format!("tensor not found: {}", name)))?
            .clone();

        let abs_offset = self.tensor_data_offset + info.offset;
        reader.seek(SeekFrom::Start(abs_offset))
            .map_err(|e| TqError::Msg(format!("seek error for {}: {}", name, e)))?;

        let size = info.data_size_bytes();
        let mut buf = vec![0u8; size];
        reader.read_exact(&mut buf)
            .map_err(|e| TqError::Msg(format!("read error for {} ({} bytes): {}", name, size, e)))?;

        Ok((info, buf))
    }

    /// Load a tensor as QWeight (candle-compatible API).
    ///
    /// This is the primary tensor loading method — matches candle's
    /// `content.tensor(reader, name, device)?` but returns QWeight.
    /// The `_device` parameter is accepted for API compat but ignored
    /// (QWeight is device-agnostic; transfer happens at compute time).
    pub fn tensor<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
        _device: &TqDevice,
    ) -> Result<QWeight> {
        let (info, data) = self.tensor_data(reader, name)?;
        let shape = if info.shape.len() >= 2 {
            (info.shape[0], info.shape[1])
        } else if info.shape.len() == 1 {
            (1, info.shape[0])
        } else {
            (1, 1)
        };
        Ok(QWeight::new(data, info.dtype, shape))
    }

    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a metadata value, returning error if missing.
    pub fn require(&self, key: &str) -> Result<&GgufValue> {
        self.metadata.get(key)
            .ok_or_else(|| TqError::Msg(format!("missing metadata: {}", key)))
    }
}

// ─── GGUF Writer ─────────────────────────────────────────────

/// Write a complete GGUF v3 file.
///
/// `tensors`: Vec of (name, dims_innermost_first, dtype, data_bytes).
/// Metadata and tensor data are written sequentially.
pub fn write_gguf<W: Write + Seek>(
    writer: &mut W,
    metadata: &[(&str, &GgufValue)],
    tensors: &[(&str, &[usize], GgmlDType, &[u8])],
) -> Result<()> {
    let alignment = DEFAULT_ALIGNMENT;

    // ─── Header ───
    writer.write_all(&GGUF_MAGIC.to_le_bytes()).map_err(io_err)?;
    writer.write_all(&3u32.to_le_bytes()).map_err(io_err)?;  // version 3
    writer.write_all(&(tensors.len() as u64).to_le_bytes()).map_err(io_err)?;
    writer.write_all(&(metadata.len() as u64).to_le_bytes()).map_err(io_err)?;

    // ─── Metadata ───
    for &(key, value) in metadata {
        write_string(writer, key)?;
        write_value(writer, value)?;
    }

    // ─── Tensor descriptors ───
    // Pre-compute offsets: tensors are packed sequentially with alignment
    let mut data_offset = 0u64;
    let mut tensor_offsets = Vec::with_capacity(tensors.len());
    for &(_, _, _, data) in tensors {
        let aligned = (data_offset as usize + alignment - 1) / alignment * alignment;
        tensor_offsets.push(aligned as u64);
        data_offset = aligned as u64 + data.len() as u64;
    }

    for (i, &(name, dims, dtype, _)) in tensors.iter().enumerate() {
        write_string(writer, name)?;
        writer.write_all(&(dims.len() as u32).to_le_bytes()).map_err(io_err)?;
        // GGUF stores dims innermost-first (reverse of our shape convention)
        for &d in dims.iter().rev() {
            writer.write_all(&(d as u64).to_le_bytes()).map_err(io_err)?;
        }
        writer.write_all(&(dtype as u32).to_le_bytes()).map_err(io_err)?;
        writer.write_all(&tensor_offsets[i].to_le_bytes()).map_err(io_err)?;
    }

    // ─── Alignment padding before tensor data ───
    let pos = writer.stream_position().map_err(io_err)? as usize;
    let aligned_pos = (pos + alignment - 1) / alignment * alignment;
    let padding = aligned_pos - pos;
    if padding > 0 {
        writer.write_all(&vec![0u8; padding]).map_err(io_err)?;
    }

    // ─── Tensor data ───
    let data_start = writer.stream_position().map_err(io_err)?;
    for (i, &(_, _, _, data)) in tensors.iter().enumerate() {
        // Pad to alignment
        let target = data_start + tensor_offsets[i];
        let current = writer.stream_position().map_err(io_err)?;
        if current < target {
            writer.write_all(&vec![0u8; (target - current) as usize]).map_err(io_err)?;
        }
        writer.write_all(data).map_err(io_err)?;
    }

    Ok(())
}

fn io_err(e: std::io::Error) -> TqError {
    TqError::Msg(format!("GGUF write: {}", e))
}

fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes()).map_err(io_err)?;
    w.write_all(s.as_bytes()).map_err(io_err)?;
    Ok(())
}

fn write_value<W: Write>(w: &mut W, val: &GgufValue) -> Result<()> {
    match val {
        GgufValue::U8(v) => { w.write_all(&0u32.to_le_bytes()).map_err(io_err)?; w.write_all(&[*v]).map_err(io_err)?; }
        GgufValue::I8(v) => { w.write_all(&1u32.to_le_bytes()).map_err(io_err)?; w.write_all(&[*v as u8]).map_err(io_err)?; }
        GgufValue::U16(v) => { w.write_all(&2u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::I16(v) => { w.write_all(&3u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::U32(v) => { w.write_all(&4u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::I32(v) => { w.write_all(&5u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::F32(v) => { w.write_all(&6u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::String(v) => { w.write_all(&8u32.to_le_bytes()).map_err(io_err)?; write_string(w, v)?; }
        GgufValue::U64(v) => { w.write_all(&10u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::I64(v) => { w.write_all(&11u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::F64(v) => { w.write_all(&12u32.to_le_bytes()).map_err(io_err)?; w.write_all(&v.to_le_bytes()).map_err(io_err)?; }
        GgufValue::Bool(v) => { w.write_all(&7u32.to_le_bytes()).map_err(io_err)?; w.write_all(&[if *v {1} else {0}]).map_err(io_err)?; }
        GgufValue::Array(arr) => {
            w.write_all(&9u32.to_le_bytes()).map_err(io_err)?; // GGUF_TYPE_ARRAY
            // Detect element type from first element
            let elem_type = match arr.first() {
                Some(GgufValue::U8(_)) => 0u32,
                Some(GgufValue::I8(_)) => 1u32,
                Some(GgufValue::U16(_)) => 2u32,
                Some(GgufValue::I16(_)) => 3u32,
                Some(GgufValue::U32(_)) => 4u32,
                Some(GgufValue::I32(_)) => 5u32,
                Some(GgufValue::F32(_)) => 6u32,
                Some(GgufValue::F64(_)) => 12u32,
                Some(GgufValue::String(_)) => 8u32,
                Some(GgufValue::U64(_)) => 10u32,
                Some(GgufValue::I64(_)) => 11u32,
                Some(GgufValue::Bool(_)) => 13u32,
                _ => 4u32,
            };
            w.write_all(&elem_type.to_le_bytes()).map_err(io_err)?;
            w.write_all(&(arr.len() as u64).to_le_bytes()).map_err(io_err)?;
            for elem in arr {
                write_value_payload(w, elem)?;
            }
        }
    }
    Ok(())
}

fn write_value_payload<W: Write>(w: &mut W, val: &GgufValue) -> Result<()> {
    match val {
        GgufValue::U8(v) => w.write_all(&[*v]).map_err(io_err)?,
        GgufValue::I8(v) => w.write_all(&[*v as u8]).map_err(io_err)?,
        GgufValue::U16(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::I16(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::U32(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::I32(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::F32(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::F64(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::U64(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::I64(v) => w.write_all(&v.to_le_bytes()).map_err(io_err)?,
        GgufValue::Bool(v) => w.write_all(&[if *v {1} else {0}]).map_err(io_err)?,
        GgufValue::String(v) => write_string(w, v)?,
        GgufValue::Array(_) => return Err(TqError::Msg("nested arrays not supported".into())),
    }
    Ok(())
}

// ─── Binary reading helpers ───────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read u8: {}", e)))?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read u16: {}", e)))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read i16: {}", e)))?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read u32: {}", e)))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read i32: {}", e)))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read u64: {}", e)))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read i64: {}", e)))?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read f32: {}", e)))?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read f64: {}", e)))?;
    Ok(f64::from_le_bytes(buf))
}

/// V1 uses u32 counts, V2/V3 uses u64.
fn read_u64_or_u32<R: Read>(r: &mut R, version: u32) -> Result<u64> {
    if version == 1 {
        Ok(read_u32(r)? as u64)
    } else {
        read_u64(r)
    }
}

/// Read a GGUF string: [len][bytes] (len is u32 for V1, u64 for V2/V3).
fn read_string<R: Read>(r: &mut R, version: u32) -> Result<String> {
    let len = read_u64_or_u32(r, version)? as usize;
    if len > 1_000_000 {
        return Err(TqError::Msg(format!("string too long: {} bytes", len)));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| TqError::Msg(format!("read string: {}", e)))?;
    String::from_utf8(buf).map_err(|e| TqError::Msg(format!("invalid utf8: {}", e)))
}

/// Read a typed GGUF metadata value.
fn read_value<R: Read>(r: &mut R, version: u32) -> Result<GgufValue> {
    let type_id = read_u32(r)?;
    match type_id {
        0 => Ok(GgufValue::U8(read_u8(r)?)),
        1 => Ok(GgufValue::I8(read_i8(r)?)),
        2 => Ok(GgufValue::U16(read_u16(r)?)),
        3 => Ok(GgufValue::I16(read_i16(r)?)),
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8 => Ok(GgufValue::String(read_string(r, version)?)),
        9 => {
            // Array: [elem_type:u32][count:u64_or_u32][values...]
            let elem_type = read_u32(r)?;
            let count = read_u64_or_u32(r, version)? as usize;
            if count > 10_000_000 {
                return Err(TqError::Msg(format!("array too large: {}", count)));
            }
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_typed_value(r, elem_type, version)?);
            }
            Ok(GgufValue::Array(values))
        }
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => Ok(GgufValue::I64(read_i64(r)?)),
        12 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Err(TqError::Msg(format!("unknown metadata type: {}", type_id))),
    }
}

/// Read a value of a known type (used for array elements).
fn read_typed_value<R: Read>(r: &mut R, type_id: u32, version: u32) -> Result<GgufValue> {
    match type_id {
        0 => Ok(GgufValue::U8(read_u8(r)?)),
        1 => Ok(GgufValue::I8(read_i8(r)?)),
        2 => Ok(GgufValue::U16(read_u16(r)?)),
        3 => Ok(GgufValue::I16(read_i16(r)?)),
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8 => Ok(GgufValue::String(read_string(r, version)?)),
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => Ok(GgufValue::I64(read_i64(r)?)),
        12 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Err(TqError::Msg(format!("unknown array element type: {}", type_id))),
    }
}

// ─── f16 conversion ───────────────────────────────────────────

/// Convert IEEE 754 half-precision (f16) to f32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let val = mant as f32 / 1024.0 * (1.0 / 16384.0); // 2^-14
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | 0x7f800000)
        } else {
            f32::NAN
        }
    } else {
        // Normal
        let new_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (new_exp << 23) | (mant << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_dtype_sizes() {
        assert_eq!(GgmlDType::Q4K.block_numel(), 256);
        assert_eq!(GgmlDType::Q4K.block_size_bytes(), 144);
        assert_eq!(GgmlDType::Q8_0.block_numel(), 32);
        assert_eq!(GgmlDType::Q8_0.block_size_bytes(), 34);
        assert_eq!(GgmlDType::F32.block_numel(), 1);
        assert_eq!(GgmlDType::F32.block_size_bytes(), 4);
    }

    #[test]
    fn test_ggml_dtype_tensor_size() {
        // Q4_K_M: 256 elements per block, 144 bytes per block
        // 4096 elements = 16 blocks = 16 * 144 = 2304 bytes
        assert_eq!(GgmlDType::Q4K.tensor_size_bytes(4096), 2304);
        // Q8_0: 32 elements per block, 34 bytes per block
        // 4096 elements = 128 blocks = 128 * 34 = 4352 bytes
        assert_eq!(GgmlDType::Q8_0.tensor_size_bytes(4096), 4352);
    }

    #[test]
    fn test_f16_to_f32() {
        // 0x3C00 = 1.0 in f16
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0x0000 = 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0xBC00 = -1.0
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0x4000 = 2.0
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gguf_magic() {
        let bytes: Vec<u8> = vec![
            0x47, 0x47, 0x55, 0x46, // magic "GGUF"
            0x03, 0x00, 0x00, 0x00, // version 3
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // tensor_count = 0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // metadata_count = 0
        ];
        let mut cursor = std::io::Cursor::new(bytes);
        let content = GgufContent::read(&mut cursor).unwrap();
        assert_eq!(content.version, 3);
        assert!(content.metadata.is_empty());
        assert!(content.tensor_infos.is_empty());
    }

    #[test]
    fn test_gguf_value_conversions() {
        // to_u32: from U32
        assert_eq!(GgufValue::U32(42).to_u32().unwrap(), 42);
        // to_u32: from U8
        assert_eq!(GgufValue::U8(7).to_u32().unwrap(), 7);
        // to_u32: from U16
        assert_eq!(GgufValue::U16(1000).to_u32().unwrap(), 1000);
        // to_u32: from I32
        assert_eq!(GgufValue::I32(99).to_u32().unwrap(), 99);
        // to_u32: from U64
        assert_eq!(GgufValue::U64(123).to_u32().unwrap(), 123);
        // to_u32: from F32 should fail
        assert!(GgufValue::F32(1.0).to_u32().is_err());

        // to_f32: from F32
        assert!((GgufValue::F32(3.14).to_f32().unwrap() - 3.14).abs() < 1e-6);
        // to_f32: from F64
        assert!((GgufValue::F64(2.718).to_f32().unwrap() - 2.718).abs() < 1e-3);
        // to_f32: from U32 should fail
        assert!(GgufValue::U32(1).to_f32().is_err());

        // to_string_val: from String
        assert_eq!(GgufValue::String("hello".into()).to_string_val().unwrap(), "hello");
        // to_string_val: from U32 should fail
        assert!(GgufValue::U32(1).to_string_val().is_err());
    }

    #[test]
    fn test_ggml_dtype_all_variants() {
        // All valid variant round-trips
        let cases: &[(u32, GgmlDType)] = &[
            (0, GgmlDType::F32),
            (1, GgmlDType::F16),
            (2, GgmlDType::Q4_0),
            (3, GgmlDType::Q4_1),
            (6, GgmlDType::Q5_0),
            (7, GgmlDType::Q5_1),
            (8, GgmlDType::Q8_0),
            (9, GgmlDType::Q8_1),
            (10, GgmlDType::Q2K),
            (11, GgmlDType::Q3K),
            (12, GgmlDType::Q4K),
            (13, GgmlDType::Q5K),
            (14, GgmlDType::Q6K),
            (15, GgmlDType::Q8K),
            (30, GgmlDType::BF16),
        ];
        for &(id, expected) in cases {
            assert_eq!(GgmlDType::from_u32(id).unwrap(), expected);
        }
        // Invalid values should error
        assert!(GgmlDType::from_u32(4).is_err());
        assert!(GgmlDType::from_u32(5).is_err());
        assert!(GgmlDType::from_u32(16).is_err());
        assert!(GgmlDType::from_u32(255).is_err());
    }
}
