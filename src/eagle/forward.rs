//! Sprint 1 Day 4 — EAGLE draft forward pass (FP32 smoke path).
//!
//! # Scope
//!
//! One-token draft forward. Builds an FP32 runtime (dequantising the FP16
//! weights uploaded in Day 3) plus scratch + KV cache buffers, then runs:
//!
//! ```text
//!   concat(prev_hidden [hidden], embed(token_id) [hidden])
//!     -> fc.weight @ _ + fc.bias          [hidden]   (EAGLE fusion; no pre-norm)
//!     -> q/k/v proj (+ biases)            [hidden each]
//!     -> RoPE(q, k) at position
//!     -> flash_decode over K/V cache     [hidden]
//!     -> o.weight @ _                    [hidden]
//!     -> residual (fc_out + attn_out)
//!     -> post_attention_layernorm         [hidden]
//!     -> gate_proj / up_proj / silu_mul   [intermediate]
//!     -> down_proj                        [hidden]
//!     -> residual2 = residual + mlp_out
//! ```
//!
//! The output hidden vector is copied back to the host and returned — the
//! caller (engine.rs in Sprint 3) will feed it into the target's LM head to
//! produce the speculated logits.
//!
//! # Why FP32 throughout
//!
//! Every other kernel in the engine (rms_norm, rope, flash_decode, silu_mul)
//! is FP32. Running the draft in FP32 means no fresh kernel variants and lets
//! us reuse the already-validated path. The one-time cost is ~1.17 GB of FP32
//! weight shadows on GPU (embedding kept as FP16 → FP32 once at init). The
//! originals FP16 slices are *not* freed in Day 4; Sprint 2 can drop them once
//! the forward is stable and we want to reclaim VRAM.
//!
//! # Not yet done (deferred to Sprint 2+)
//! - Tree attention (variable N queries)
//! - Verify / rollback loop in engine.rs
//! - KV cache truncation on mispredict

#![cfg(feature = "cuda")]

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys as drv_sys;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;

use super::{DraftWeights, EagleDraftConfig, EagleError};
use crate::cuda::kernels;

/// Partial-KV split size for the draft's `flash_decode` call. Small value
/// because the draft only attends over seq_len <= max_position_embeddings
/// (2048) and runs per-step, so launch overhead dominates anyway.
const FLASH_SPLIT_SIZE: usize = 32;

/// FP32 runtime for the EAGLE draft. Built lazily on first forward pass from
/// the Day 3 FP16 upload — see [`DraftRuntime::from_weights`].
#[derive(Debug)]
pub struct DraftRuntime {
    // ── Dequantised (FP32) weight shadows ──────────────────────────────────
    // All shape comments are row-major as stored by safetensors.
    pub embed_tokens: CudaSlice<f32>,     // [vocab, hidden]
    pub fc_weight:    CudaSlice<f32>,     // [hidden, 2*hidden]
    pub fc_bias:      CudaSlice<f32>,     // [hidden]
    pub q_weight:     CudaSlice<f32>,     // [hidden, hidden]
    pub q_bias:       CudaSlice<f32>,     // [hidden]
    pub k_weight:     CudaSlice<f32>,     // [hidden, hidden]
    pub k_bias:       CudaSlice<f32>,     // [hidden]
    pub v_weight:     CudaSlice<f32>,     // [hidden, hidden]
    pub v_bias:       CudaSlice<f32>,     // [hidden]
    pub o_weight:     CudaSlice<f32>,     // [hidden, hidden]
    pub gate_weight:  CudaSlice<f32>,     // [intermediate, hidden]
    pub up_weight:    CudaSlice<f32>,     // [intermediate, hidden]
    pub down_weight:  CudaSlice<f32>,     // [hidden, intermediate]
    pub post_norm:    CudaSlice<f32>,     // [hidden]

    // ── RoPE tables [max_seq, head_dim/2] ─────────────────────────────────
    pub rope_cos:     CudaSlice<f32>,
    pub rope_sin:     CudaSlice<f32>,

    // ── KV cache [n_kv_heads, max_seq, head_dim] ──────────────────────────
    pub k_cache:      CudaSlice<f32>,
    pub v_cache:      CudaSlice<f32>,

    // ── Per-forward scratch ───────────────────────────────────────────────
    concat_buf:   CudaSlice<f32>,         // [2*hidden]
    fc_out:       CudaSlice<f32>,         // [hidden]  — fc projection output
    q_buf:        CudaSlice<f32>,         // [hidden]
    k_buf:        CudaSlice<f32>,         // [hidden]
    v_buf:        CudaSlice<f32>,         // [hidden]
    attn_out:     CudaSlice<f32>,         // [hidden]
    o_out:        CudaSlice<f32>,         // [hidden]
    norm_out:     CudaSlice<f32>,         // [hidden]
    mlp_gate:     CudaSlice<f32>,         // [intermediate]
    mlp_up:       CudaSlice<f32>,         // [intermediate]
    mlp_silu:     CudaSlice<f32>,         // [intermediate]
    mlp_down:     CudaSlice<f32>,         // [hidden]

    partial_o:    CudaSlice<f32>,         // [n_heads * max_splits * head_dim]
    partial_max:  CudaSlice<f32>,         // [n_heads * max_splits]
    partial_sum:  CudaSlice<f32>,         // [n_heads * max_splits]

    pub config:   EagleDraftConfig,
    /// Number of tokens currently in K/V cache (starts at 0, grows monotonically
    /// until the verify/rollback path lands in Sprint 3 and truncates on reject).
    pub seq_len:  usize,
}

impl DraftRuntime {
    /// Dequantise the FP16 GPU weights and allocate scratch + KV + RoPE
    /// tables. Expensive (~1 GB of kernel work + 1-2 s) but one-shot.
    pub fn from_weights(w: &DraftWeights, cfg: &EagleDraftConfig) -> Result<Self, EagleError> {
        let reg = kernels::global_registry()
            .ok_or_else(|| EagleError::Msg("no GPU kernel registry available".into()))?;
        // Keep the Arc — cudarc's memcpy_stod/alloc_zeros etc. are defined on
        // `Arc<CudaStream>`, not on `&CudaStream`, so we can't bind through a
        // bare `&CudaStream`.
        let stream = reg.stream.clone();

        let hidden       = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let vocab        = cfg.vocab_size;
        let n_heads      = cfg.num_attention_heads;
        let n_kv_heads   = cfg.num_key_value_heads;
        let head_dim     = hidden / n_heads;
        let max_seq      = cfg.max_position_embeddings;
        let rope_dim     = head_dim;   // Qwen2 rotates the full head

        // f16 -> f32 dequant helper
        let dequant = |src: &CudaSlice<half::f16>, n: usize, label: &str|
            -> Result<CudaSlice<f32>, EagleError>
        {
            let mut out = stream.alloc_zeros::<f32>(n)
                .map_err(|e| EagleError::Msg(format!("alloc {} f32: {}", label, e)))?;
            kernels::f16_to_f32(reg, src, &mut out, n)
                .map_err(|e| EagleError::Msg(format!("f16->f32 {}: {}", label, e)))?;
            Ok(out)
        };

        let embed_tokens = dequant(&w.embed_tokens,      vocab * hidden,     "embed_tokens")?;
        let fc_weight    = dequant(&w.fc_weight,         hidden * 2 * hidden, "fc_weight")?;
        let fc_bias      = dequant(&w.fc_bias,            hidden,             "fc_bias")?;
        let q_weight     = dequant(&w.q_proj_weight,     hidden * hidden,    "q_weight")?;
        let q_bias       = dequant(&w.q_proj_bias,        hidden,             "q_bias")?;
        let k_weight     = dequant(&w.k_proj_weight,     hidden * hidden,    "k_weight")?;
        let k_bias       = dequant(&w.k_proj_bias,        hidden,             "k_bias")?;
        let v_weight     = dequant(&w.v_proj_weight,     hidden * hidden,    "v_weight")?;
        let v_bias       = dequant(&w.v_proj_bias,        hidden,             "v_bias")?;
        let o_weight     = dequant(&w.o_proj_weight,     hidden * hidden,    "o_weight")?;
        let gate_weight  = dequant(&w.mlp_gate_weight,   intermediate * hidden, "gate_weight")?;
        let up_weight    = dequant(&w.mlp_up_weight,     intermediate * hidden, "up_weight")?;
        let down_weight  = dequant(&w.mlp_down_weight,   hidden * intermediate, "down_weight")?;
        let post_norm    = dequant(&w.post_attn_norm_weight, hidden,         "post_norm")?;

        // RoPE tables (cos, sin). CPU build → H2D copy. Shape [max_seq, rope_dim/2].
        // The draft uses full-rank K/V (no GQA), so this matches the standard
        // `rope_halved_f32` layout the engine uses everywhere else.
        let half = rope_dim / 2;
        let mut cos_host = vec![0f32; max_seq * half];
        let mut sin_host = vec![0f32; max_seq * half];
        for p in 0..max_seq {
            for i in 0..half {
                let theta = (p as f32)
                    / cfg.rope_theta.powf(2.0 * i as f32 / rope_dim as f32);
                cos_host[p * half + i] = theta.cos();
                sin_host[p * half + i] = theta.sin();
            }
        }
        let rope_cos = stream.memcpy_stod(&cos_host)
            .map_err(|e| EagleError::Msg(format!("up rope_cos: {}", e)))?;
        let rope_sin = stream.memcpy_stod(&sin_host)
            .map_err(|e| EagleError::Msg(format!("up rope_sin: {}", e)))?;

        // KV cache: full-rank (draft is 28/28, not GQA).
        let k_cache = stream.alloc_zeros::<f32>(n_kv_heads * max_seq * head_dim)
            .map_err(|e| EagleError::Msg(format!("alloc k_cache: {}", e)))?;
        let v_cache = stream.alloc_zeros::<f32>(n_kv_heads * max_seq * head_dim)
            .map_err(|e| EagleError::Msg(format!("alloc v_cache: {}", e)))?;

        let alloc = |n: usize, label: &str| -> Result<CudaSlice<f32>, EagleError> {
            stream.alloc_zeros::<f32>(n)
                .map_err(|e| EagleError::Msg(format!("alloc {}: {}", label, e)))
        };
        let max_splits = (max_seq + FLASH_SPLIT_SIZE - 1) / FLASH_SPLIT_SIZE;

        let rt = DraftRuntime {
            embed_tokens, fc_weight, fc_bias,
            q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, o_weight,
            gate_weight, up_weight, down_weight, post_norm,
            rope_cos, rope_sin, k_cache, v_cache,
            concat_buf:  alloc(2 * hidden, "concat_buf")?,
            fc_out:      alloc(hidden,    "fc_out")?,
            q_buf:       alloc(hidden,    "q_buf")?,
            k_buf:       alloc(hidden,    "k_buf")?,
            v_buf:       alloc(hidden,    "v_buf")?,
            attn_out:    alloc(hidden,    "attn_out")?,
            o_out:       alloc(hidden,    "o_out")?,
            norm_out:    alloc(hidden,    "norm_out")?,
            mlp_gate:    alloc(intermediate, "mlp_gate")?,
            mlp_up:      alloc(intermediate, "mlp_up")?,
            mlp_silu:    alloc(intermediate, "mlp_silu")?,
            mlp_down:    alloc(hidden,    "mlp_down")?,
            partial_o:   alloc(n_heads * max_splits * head_dim, "partial_o")?,
            partial_max: alloc(n_heads * max_splits, "partial_max")?,
            partial_sum: alloc(n_heads * max_splits, "partial_sum")?,
            config:      cfg.clone(),
            seq_len:     0,
        };
        stream.synchronize()
            .map_err(|e| EagleError::Msg(format!("sync after runtime build: {}", e)))?;
        Ok(rt)
    }

    /// Approximate VRAM footprint of this runtime (weights + cache + scratch).
    pub fn total_bytes(&self) -> usize {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let v = cfg.vocab_size;
        let n_kv = cfg.num_key_value_heads;
        let hd = h / cfg.num_attention_heads;
        let ms = cfg.max_position_embeddings;
        let half = hd / 2;
        let max_splits = (ms + FLASH_SPLIT_SIZE - 1) / FLASH_SPLIT_SIZE;
        let weights_f32 = (v * h) + (h * 2 * h) + h + 4 * (h * h) + 3 * h + 3 * (i * h) + h;
        let kv_cache = 2 * n_kv * ms * hd;
        let rope = 2 * ms * half;
        let scratch = 2 * h + 4 * h + 3 * h + 4 * i + cfg.num_attention_heads * max_splits * (hd + 2);
        4 * (weights_f32 + kv_cache + rope + scratch)
    }

    /// Run one draft decode step. Returns the output hidden vector on the host.
    ///
    /// `prev_hidden` : target-model's previous-layer hidden state  [hidden]
    /// `token_id`    : ID of the just-sampled token (for embed lookup)
    /// `position`    : 0-indexed position of this step (feeds into RoPE and
    ///                 the KV cache append). Must be strictly monotonically
    ///                 non-decreasing across successive calls (Sprint 3 adds
    ///                 rollback).
    ///
    /// Contract:
    /// - input length == `self.config.hidden_size`
    /// - `token_id < self.config.vocab_size`
    /// - `position < self.config.max_position_embeddings`
    /// - `position == self.seq_len` (Sprint 1 — append-only)
    pub fn forward(
        &mut self,
        prev_hidden: &[f32],
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, EagleError> {
        let cfg = self.config.clone();
        let hidden       = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let n_heads      = cfg.num_attention_heads;
        let n_kv_heads   = cfg.num_key_value_heads;
        let head_dim     = hidden / n_heads;
        let rope_dim     = head_dim;
        let max_seq      = cfg.max_position_embeddings;

        // Input validation — the kernels below would either silently write
        // out-of-bounds or launch with bogus grid dims otherwise.
        if prev_hidden.len() != hidden {
            return Err(EagleError::Msg(format!(
                "prev_hidden len {} != hidden {}", prev_hidden.len(), hidden)));
        }
        if (token_id as usize) >= cfg.vocab_size {
            return Err(EagleError::Msg(format!(
                "token_id {} >= vocab {}", token_id, cfg.vocab_size)));
        }
        if position >= max_seq {
            return Err(EagleError::Msg(format!(
                "position {} >= max_seq {}", position, max_seq)));
        }
        if position != self.seq_len {
            return Err(EagleError::Msg(format!(
                "position {} != seq_len {} (Sprint 1 is append-only)",
                position, self.seq_len)));
        }

        let reg = kernels::global_registry()
            .ok_or_else(|| EagleError::Msg("no GPU registry in forward".into()))?;
        // Keep an Arc<CudaStream> for memcpy_stod/memcpy_dtov (which are
        // defined on the Arc type), plus a bare &CudaStream for device_ptr.
        let stream_arc = reg.stream.clone();
        let stream: &cudarc::driver::CudaStream = stream_arc.as_ref();
        let raw_stream = stream.cu_stream();
        let blas = reg.get_cublas();

        // Day 3 debug: when TQ_EAGLE_DUMP=<dir> is set, write every
        // intermediate to <dir>/rust_<stage>.npy for element-wise comparison
        // against the Python reference in scripts/eagle_ref_numpy.py.
        let dump_dir: Option<std::path::PathBuf> = std::env::var("TQ_EAGLE_DUMP")
            .ok().filter(|s| !s.is_empty()).map(|s| s.into());
        let dump = |slice: &CudaSlice<f32>, stage: &str| -> Result<(), EagleError> {
            let Some(ref dir) = dump_dir else { return Ok(()); };
            let _ = std::fs::create_dir_all(dir);
            stream_arc.synchronize().ok();
            let v = stream_arc.memcpy_dtov(slice)
                .map_err(|e| EagleError::Msg(format!("dump {} d2h: {}", stage, e)))?;
            let path = dir.join(format!("rust_{}.npy", stage));
            write_npy_f32(&path, &v)
                .map_err(|e| EagleError::Msg(format!("dump {} write: {}", stage, e)))?;
            let l2: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("  [rust] {:<18} len={} L2={:.3}", stage, v.len(), l2);
            Ok(())
        };

        // ── 1. concat_buf = [prev_hidden | embed_tokens[token_id, :]] ──────
        // Original SafeAILab cnets.py convention: torch.cat([hidden, embed], dim=-1).
        // Reverted Day 2 after the swap experiment collapsed draft output.
        let prev_slice = stream_arc.memcpy_stod(prev_hidden)
            .map_err(|e| EagleError::Msg(format!("H2D prev_hidden: {}", e)))?;
        let f32_bytes = std::mem::size_of::<f32>();
        {
            let (src_ptr, _g1) = prev_slice.device_ptr(stream);
            let (dst_ptr, _g2) = self.concat_buf.device_ptr_mut(stream);
            let bytes = hidden * f32_bytes;
            let res = unsafe { drv_sys::cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, bytes, raw_stream) };
            if res != drv_sys::cudaError_enum::CUDA_SUCCESS {
                return Err(EagleError::Msg(format!("D2D prev→concat: {:?}", res)));
            }
        }
        {
            let (emb_ptr, _g1) = self.embed_tokens.device_ptr(stream);
            let (dst_ptr,  _g2) = self.concat_buf.device_ptr_mut(stream);
            let row_off_bytes = (token_id as usize) * hidden * f32_bytes;
            let dst_off_bytes = hidden * f32_bytes;
            let res = unsafe {
                drv_sys::cuMemcpyDtoDAsync_v2(
                    dst_ptr + dst_off_bytes as u64,
                    emb_ptr + row_off_bytes as u64,
                    hidden * f32_bytes,
                    raw_stream,
                )
            };
            if res != drv_sys::cudaError_enum::CUDA_SUCCESS {
                return Err(EagleError::Msg(format!("D2D embed→concat: {:?}", res)));
            }
        }

        dump(&self.concat_buf, "01_concat")?;

        // ── 2. fc_out = fc_weight @ concat_buf + fc_bias ──────────────────
        // fc_weight shape [hidden, 2*hidden], input [2*hidden], output [hidden].
        matvec_row_major(&blas, &self.fc_weight, &self.concat_buf,
                         &mut self.fc_out, 2 * hidden, hidden)?;
        kernels::bias_add_inplace(reg, &mut self.fc_out, &self.fc_bias, hidden)
            .map_err(|e| EagleError::Msg(format!("fc bias: {}", e)))?;
        dump(&self.fc_out, "02_fc_out")?;

        // ── 3. q/k/v projections (fc_out -> {q,k,v}_buf) ──────────────────
        matvec_row_major(&blas, &self.q_weight, &self.fc_out,
                         &mut self.q_buf, hidden, hidden)?;
        kernels::bias_add_inplace(reg, &mut self.q_buf, &self.q_bias, hidden)
            .map_err(|e| EagleError::Msg(format!("q bias: {}", e)))?;
        matvec_row_major(&blas, &self.k_weight, &self.fc_out,
                         &mut self.k_buf, hidden, hidden)?;
        kernels::bias_add_inplace(reg, &mut self.k_buf, &self.k_bias, hidden)
            .map_err(|e| EagleError::Msg(format!("k bias: {}", e)))?;
        matvec_row_major(&blas, &self.v_weight, &self.fc_out,
                         &mut self.v_buf, hidden, hidden)?;
        kernels::bias_add_inplace(reg, &mut self.v_buf, &self.v_bias, hidden)
            .map_err(|e| EagleError::Msg(format!("v bias: {}", e)))?;
        dump(&self.q_buf, "03_q")?;
        dump(&self.k_buf, "04_k")?;
        dump(&self.v_buf, "05_v")?;

        // ── 4. RoPE(q, k) at `position` ───────────────────────────────────
        kernels::rope_halved(reg, &mut self.q_buf, &self.rope_cos, &self.rope_sin,
                             1, n_heads, head_dim, rope_dim, position)
            .map_err(|e| EagleError::Msg(format!("rope q: {}", e)))?;
        kernels::rope_halved(reg, &mut self.k_buf, &self.rope_cos, &self.rope_sin,
                             1, n_kv_heads, head_dim, rope_dim, position)
            .map_err(|e| EagleError::Msg(format!("rope k: {}", e)))?;
        dump(&self.q_buf, "06_q_rope")?;
        dump(&self.k_buf, "07_k_rope")?;

        // ── 5. Scatter k, v into cache at slot `position` ─────────────────
        // K cache layout is [n_kv_heads, max_seq, head_dim]. Our k_buf is
        // [n_kv_heads, head_dim] contiguous. For each head h, copy
        // k_buf[h * head_dim .. (h+1) * head_dim] into
        // k_cache[(h * max_seq + position) * head_dim ..].
        // Use a loop of D2D copies per head (n_kv_heads=28, small; kernel
        // fusion deferred to Sprint 2).
        let head_bytes = head_dim * f32_bytes;
        // K scatter
        {
            let (k_src, _g1) = self.k_buf.device_ptr(stream);
            let (k_dst, _g2) = self.k_cache.device_ptr_mut(stream);
            for h in 0..n_kv_heads {
                let src_off = (h * head_dim * f32_bytes) as u64;
                let dst_off = ((h * max_seq + position) * head_dim * f32_bytes) as u64;
                let res = unsafe {
                    drv_sys::cuMemcpyDtoDAsync_v2(k_dst + dst_off, k_src + src_off, head_bytes, raw_stream)
                };
                if res != drv_sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(EagleError::Msg(format!("k scatter h={}: {:?}", h, res)));
                }
            }
        }
        // V scatter
        {
            let (v_src, _g1) = self.v_buf.device_ptr(stream);
            let (v_dst, _g2) = self.v_cache.device_ptr_mut(stream);
            for h in 0..n_kv_heads {
                let src_off = (h * head_dim * f32_bytes) as u64;
                let dst_off = ((h * max_seq + position) * head_dim * f32_bytes) as u64;
                let res = unsafe {
                    drv_sys::cuMemcpyDtoDAsync_v2(v_dst + dst_off, v_src + src_off, head_bytes, raw_stream)
                };
                if res != drv_sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(EagleError::Msg(format!("v scatter h={}: {:?}", h, res)));
                }
            }
        }
        self.seq_len = position + 1;

        // ── 6. flash_decode — attention over K/V cache  ────────────────────
        let seq_kv = self.seq_len;
        let n_splits = (seq_kv + FLASH_SPLIT_SIZE - 1) / FLASH_SPLIT_SIZE;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        kernels::flash_decode_partial(
            reg, &self.q_buf, &self.k_cache, &self.v_cache,
            &mut self.partial_o, &mut self.partial_max, &mut self.partial_sum,
            1,                    // batch_size
            n_heads, n_kv_heads,
            seq_kv, head_dim, scale,
            FLASH_SPLIT_SIZE, max_seq,
            -1,                   // window_size (-1 = full attention)
        ).map_err(|e| EagleError::Msg(format!("flash_decode_partial: {}", e)))?;
        kernels::flash_decode_reduce(
            reg, &self.partial_o, &self.partial_max, &self.partial_sum,
            &mut self.attn_out, n_heads, n_splits, head_dim, 1,
        ).map_err(|e| EagleError::Msg(format!("flash_decode_reduce: {}", e)))?;
        dump(&self.attn_out, "08_attn_out")?;

        // ── 7. o_out = o_weight @ attn_out ─────────────────────────────────
        matvec_row_major(&blas, &self.o_weight, &self.attn_out,
                         &mut self.o_out, hidden, hidden)?;
        dump(&self.o_out, "09_o_out")?;

        // ── 8. post-attention block:
        //        residual <- fc_out + o_out    (residual path)
        //        norm_out <- rms_norm(residual, post_norm)
        // `fused_add_rms_norm` does exactly this in one kernel:
        //     residual += input; output = norm(residual)
        // So residual = fc_out (in-place accumulator), input = o_out.
        kernels::fused_add_rms_norm(
            reg,
            &self.o_out,          // input (added to residual)
            &mut self.fc_out,     // residual (in-place accumulator)
            &self.post_norm,      // norm weight
            &mut self.norm_out,   // output (normalised)
            1, hidden, cfg.rms_norm_eps,
        ).map_err(|e| EagleError::Msg(format!("post_attn norm: {}", e)))?;
        dump(&self.fc_out,   "10_residual_1")?;
        dump(&self.norm_out, "11_norm_out")?;

        // ── 9. MLP: gate, up, silu_mul, down ──────────────────────────────
        matvec_row_major(&blas, &self.gate_weight, &self.norm_out,
                         &mut self.mlp_gate, hidden, intermediate)?;
        matvec_row_major(&blas, &self.up_weight, &self.norm_out,
                         &mut self.mlp_up, hidden, intermediate)?;
        dump(&self.mlp_gate, "12_gate")?;
        dump(&self.mlp_up,   "13_up")?;
        kernels::fused_silu_mul(reg, &self.mlp_gate, &self.mlp_up,
                                &mut self.mlp_silu, intermediate)
            .map_err(|e| EagleError::Msg(format!("silu_mul: {}", e)))?;
        dump(&self.mlp_silu, "14_silu_mul")?;
        matvec_row_major(&blas, &self.down_weight, &self.mlp_silu,
                         &mut self.mlp_down, intermediate, hidden)?;
        dump(&self.mlp_down, "15_down")?;

        // ── 10. residual2 = residual + mlp_down  (in-place into fc_out) ───
        // bias_add_inplace is "data += bias" which is elementwise add over
        // arbitrary [n]-vectors — exactly what we need here.
        kernels::bias_add_inplace(reg, &mut self.fc_out, &self.mlp_down, hidden)
            .map_err(|e| EagleError::Msg(format!("mlp residual: {}", e)))?;
        dump(&self.fc_out, "16_final")?;

        // ── 11. D2H → return ───────────────────────────────────────────────
        stream_arc.synchronize()
            .map_err(|e| EagleError::Msg(format!("sync pre-d2h: {}", e)))?;
        let out = stream_arc.memcpy_dtov(&self.fc_out)
            .map_err(|e| EagleError::Msg(format!("D2H output: {}", e)))?;
        Ok(out)
    }
}

/// Compute `out[out_dim] = W[out_dim, in_dim] @ x[in_dim]` via cuBLAS SGEMM.
///
/// # Row-major / column-major
///
/// Weight `W` is stored row-major `[out_dim, in_dim]`. Input `x` is a
/// contiguous `[in_dim]` vector. Because cuBLAS is column-major, we use the
/// "swap operands and dims" trick from `matmul_cublas` in `src/cuda/tensor/matmul.rs`:
///
/// ```text
///   Row-major:  out[1, out_dim] = x[1, in_dim] @ W_T[in_dim, out_dim]
///   Call cuBLAS SGEMM:  m'=1, n'=out_dim, k=in_dim, A=x, B=W, both OP_N.
/// ```
///
/// lda=1, ldb=in_dim, ldc=1. The kernel computes the transpose in col-major
/// which is our row-major answer.
fn matvec_row_major(
    blas: &cudarc::cublas::CudaBlas,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    in_dim: usize,
    out_dim: usize,
) -> Result<(), EagleError> {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m:      1 as i32,
        n:      out_dim as i32,
        k:      in_dim as i32,
        alpha:  1.0f32,
        lda:    1 as i32,
        ldb:    in_dim as i32,
        beta:   0.0f32,
        ldc:    1 as i32,
    };
    unsafe {
        blas.gemm(cfg, x, w, out)
            .map_err(|e| EagleError::Msg(format!("matvec gemm: {}", e)))?;
    }
    Ok(())
}

/// Minimal NPY v1 writer for 1-D f32 arrays. Avoids adding a numpy crate
/// dependency for what's essentially a debug-only dump path.
/// Format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
fn write_npy_f32(path: &std::path::Path, data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let magic = b"\x93NUMPY";
    let version_major = 1u8;
    let version_minor = 0u8;
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );
    // Header must be padded so the total (magic+version+header_len+header) is a
    // multiple of 64; the header string is padded with spaces + trailing '\n'.
    let prefix_len = 6 + 2 + 2; // magic + version bytes + header_len u16
    let mut header_bytes = header.into_bytes();
    header_bytes.push(b'\n');
    while (prefix_len + header_bytes.len()) % 64 != 0 {
        header_bytes.insert(header_bytes.len() - 1, b' ');
    }
    let header_len = header_bytes.len() as u16;

    let mut f = std::fs::File::create(path)?;
    f.write_all(magic)?;
    f.write_all(&[version_major, version_minor])?;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(&header_bytes)?;
    // Data as little-endian f32 bytes
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    f.write_all(bytes)?;
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_dims_sanity() {
        // Pure data-structure test — verify cfg const would be sane.
        // Does not exercise GPU; just documents lda/ldb/ldc derivation.
        let out_dim = 8_usize;
        let in_dim  = 4_usize;
        assert_eq!(out_dim * in_dim, 32); // [out, in] flat size
        // The cuBLAS swap trick expects m'=1, n'=out, k=in.
        let m: i32 = 1;
        let n: i32 = out_dim as i32;
        let k: i32 = in_dim as i32;
        assert!(m > 0 && n > 0 && k > 0);
    }
}
