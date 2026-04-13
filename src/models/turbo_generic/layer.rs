//! LayerWeights: per-layer weights, QKV, RoPE, forward_attn (attention + TQ compression).

use std::sync::Arc;

use crate::backend::ComputeBackend;
use crate::cuda::{TqTensor as Tensor, TqDevice as Device, TqDType as DType, TqError};
use crate::cuda::Result;
use crate::qmatmul as qmm;
use tq_kv::TurboQuantConfig;

use super::primitives::{Module, RmsNorm, softmax_last_dim, apply_softcap};
use super::mlp::{QMatMul, MlpOrMoe, repeat_kv, masked_fill};
use super::kv_cache::*;

/// QKV weight layout — separate tensors (most models) or merged single tensor (Phi-3.5).
#[derive(Debug, Clone)]
pub(crate) enum QkvWeights {
    Separate { wq: QMatMul, wk: QMatMul, wv: QMatMul },
    Merged { wqkv: QMatMul },
}

#[derive(Debug, Clone)]
pub(crate) struct LayerWeights {
    pub(crate) qkv: QkvWeights,
    pub(crate) attention_wo: QMatMul,
    // Optional biases (Qwen2 has them, Llama doesn't)
    pub(crate) attention_bq: Option<Tensor>,
    pub(crate) attention_bk: Option<Tensor>,
    pub(crate) attention_bv: Option<Tensor>,
    pub(crate) attention_norm: RmsNorm,
    pub(crate) post_attention_norm: Option<RmsNorm>,
    pub(crate) mlp_or_moe: MlpOrMoe,
    pub(crate) ffn_norm: RmsNorm,
    pub(crate) post_ffn_norm: Option<RmsNorm>,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    /// Padded head_dim (next power of 2) for Hadamard transform.
    /// Equal to head_dim when head_dim is already a power of 2.
    pub(crate) padded_head_dim: usize,
    pub(crate) rope_dim: usize,
    pub(crate) rope_style: RopeStyle,
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
    pub(crate) neg_inf: Tensor,
    /// Gemma2 attention logit soft-capping: cap * tanh(logits / cap)
    pub(crate) attn_logit_softcap: Option<f32>,
    /// Pre-computed attention denominator. Normally `sqrt(head_dim)`; Gemma 2
    /// overrides with `sqrt(query_pre_attn_scalar)` (9B = 224, 27B = 144).
    /// Set at build time from `<arch>.attention.query_pre_attn_scalar` when
    /// present in GGUF metadata, else defaults to `sqrt(head_dim)`.
    pub(crate) attn_scale_denom: f64,
    /// Layer index (0-based) — used for selective compression
    pub(crate) layer_idx: usize,
    /// Total number of layers — needed for boundary protection (TQ_PROTECT_LAST)
    pub(crate) n_layers: usize,
    /// Standard KV cache for uncompressed layers
    pub(crate) kv_cache: Option<(Tensor, Tensor)>,
    /// Pre-allocated GPU KV cache (used when CUDA Graph is enabled).
    #[cfg(feature = "cuda")]
    pub(crate) gpu_kv_cache: Option<GpuKvCache>,
    /// Persistent GPU compressed KV buffers for fused TQ attention.
    #[cfg(feature = "cuda")]
    pub(crate) gpu_tq_cache: Option<GpuCompressedKv>,
    pub(crate) kv_compressed: Option<CompressedKvCache>,
    pub(crate) tq_config: TurboQuantConfig,
    pub(crate) signs: Vec<f32>,
    /// GPU-cached TQ compress buffers (uploaded lazily on first use)
    #[cfg(feature = "cuda")]
    pub(crate) signs_gpu: Option<cudarc::driver::CudaSlice<f32>>,
    #[cfg(feature = "cuda")]
    pub(crate) boundaries_gpu: Option<cudarc::driver::CudaSlice<f32>>,
    #[cfg(feature = "cuda")]
    pub(crate) centroids_gpu: Option<cudarc::driver::CudaSlice<f32>>,
    /// KIVI per-channel sigma [head_dim] — uploaded from calibration
    #[cfg(feature = "cuda")]
    pub(crate) channel_sigma_gpu: Option<cudarc::driver::CudaSlice<f32>>,
    /// Expanded signs [n_head × head_dim] for GPU Q pre-rotation
    #[cfg(feature = "cuda")]
    pub(crate) signs_expanded_gpu: Option<cudarc::driver::CudaSlice<f32>>,
    /// Cached sink K/V on GPU (uploaded once, reused every decode step)
    #[cfg(feature = "cuda")]
    pub(crate) sink_k_gpu: Option<cudarc::driver::CudaSlice<f32>>,
    #[cfg(feature = "cuda")]
    pub(crate) sink_v_gpu: Option<cudarc::driver::CudaSlice<f32>>,
    /// Quest-style CPU offload cache: stores evicted tokens for retrieval (Sprint F).
    /// Initialized lazily on first TriAttention eviction when TQ_OFFLOAD=1.
    pub(crate) offload_cache: Option<super::kv_cache::OffloadedKvCache>,
    /// GPU cold tier: evicted tokens kept on GPU for fast fused-attention retrieval.
    /// D2D copy (no PCIe) from hot → cold on eviction. Fused scoring on cold buffer
    /// during attention (no decompress needed). Falls back to CPU offload when GPU full.
    /// Note: not Clone/Debug (CudaSlice). Excluded from LayerWeights derive via manual field.
    #[cfg(feature = "cuda")]
    pub(crate) gpu_cold_cache: Option<Box<super::kv_cache::GpuColdKv>>,
    /// SmoothAttention: per-channel scales to migrate K outliers to Q.
    /// K is divided by these scales (reducing outliers), Q is multiplied (lossless since Q stays fp32).
    /// Computed during calibration or from running statistics.
    pub(crate) smooth_k_scales: Option<Tensor>,  // [1, 1, 1, head_dim]
    pub(crate) smooth_q_scales: Option<Tensor>,  // [1, 1, 1, head_dim]
    pub(crate) span_attn: tracing::Span,
    pub(crate) span_rot: tracing::Span,
    pub(crate) span_mlp: tracing::Span,
}

// ---------------------------------------------------------------------------
// Free helpers extracted from forward_attn_compressed (2026-04-13).
// Kept as free functions because they operate on CompressedKvCache directly
// without needing &mut LayerWeights.
// ---------------------------------------------------------------------------

/// TriAttention key storage: append this batch's pre-RoPE keys to the
/// per-head buffer and bump `tri_tokens_since_eviction`. Called at the
/// start of each forward pass when tri_config is Some.
fn store_tri_keys(
    cache: &mut CompressedKvCache,
    k_pre_rope: Option<&Tensor>,
    n_kv_head: usize,
    head_dim: usize,
    seq_len: usize,
    index_pos: usize,
) {
    if let Some(k_pr) = k_pre_rope {
        if let Ok(k_pr_flat) = k_pr.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
        {
            let tri_keys = cache.tri_keys_pre_rope.as_mut().unwrap();
            for h in 0..n_kv_head {
                for s in 0..seq_len {
                    let offset = (h * seq_len + s) * head_dim;
                    if offset + head_dim <= k_pr_flat.len() {
                        tri_keys[h].extend_from_slice(&k_pr_flat[offset..offset + head_dim]);
                    }
                }
            }
            for s in 0..seq_len {
                cache.tri_key_positions.push(index_pos + s);
            }
        }
    }
    cache.tri_tokens_since_eviction += seq_len;
}

/// TriAttention eviction pass: score per-head pre-RoPE keys, drop the
/// bottom N by importance, and keep the cache consistent (tri_keys,
/// tri_key_positions, k_per_head, v_raw, and — under cuda — the GPU hot
/// and cold tiers). Fires only on decode steps (seq_len==1) once the
/// token count exceeds the effective budget.
#[allow(clippy::too_many_arguments)]
fn maybe_evict_tri(
    cache: &mut CompressedKvCache,
    #[cfg(feature = "cuda")]
    gpu_tq_cache: &mut Option<GpuCompressedKv>,
    #[cfg(feature = "cuda")]
    gpu_cold_cache: &mut Option<Box<super::kv_cache::GpuColdKv>>,
    n_kv_head: usize,
    head_dim: usize,
    layer_idx: usize,
    #[cfg(feature = "cuda")]
    tq_bits: u8,
    #[cfg(feature = "cuda")]
    tq_group_size: usize,
    index_pos: usize,
    seq_len: usize,
) {
    let tri_cfg = match cache.tri_config.clone() {
        Some(c) => c,
        None => return,
    };
    let n_tri_keys = cache.tri_key_positions.len();
    // V3: dynamic budget from retention_ratio (overrides fixed budget)
    let effective_budget = if tri_cfg.retention_ratio > 0.0 && tri_cfg.retention_ratio < 1.0 {
        (n_tri_keys as f32 * tri_cfg.retention_ratio) as usize
    } else {
        tri_cfg.budget
    };
    if cache.tri_tokens_since_eviction < tri_cfg.eviction_interval
        || seq_len != 1
        || n_tri_keys <= effective_budget
    {
        return;
    }

    let current_pos = index_pos + seq_len;
    let tri_keys_ref = cache.tri_keys_pre_rope.as_ref().unwrap();

    // Try GPU scoring path, fall back to CPU
    let retain_per_head = 'scoring: {
        #[cfg(feature = "cuda")]
        if let Some(reg) = crate::cuda::kernels::global_registry() {
            if let Ok(retain) = gpu_tri_score_and_select(
                reg, tri_keys_ref, &cache.tri_key_positions,
                current_pos, cache.sink_len, &tri_cfg,
            ) {
                break 'scoring retain;
            }
        }
        // CPU fallback
        let key_refs: Vec<&[f32]> = tri_keys_ref.iter().map(|v| v.as_slice()).collect();
        tq_kv::triattention::evict_pass(
            &key_refs, current_pos, &cache.tri_key_positions,
            cache.sink_len, &tri_cfg,
        )
    };

    // #4: Apply eviction — splice tri_keys + compressed cache
    if let Some(retained) = retain_per_head.first() {
        if retained.len() < n_tri_keys {
            let evicted = n_tri_keys - retained.len();

            // Rebuild tri_key_positions (shared)
            let new_positions: Vec<usize> = retained.iter()
                .map(|&i| cache.tri_key_positions[i]).collect();

            // Rebuild per-head pre-RoPE keys
            let tri_keys = cache.tri_keys_pre_rope.as_mut().unwrap();
            for (h, head_retain) in retain_per_head.iter().enumerate() {
                if h >= tri_keys.len() { break; }
                let old = tri_keys[h].clone();
                let mut new_k = Vec::with_capacity(head_retain.len() * head_dim);
                for &idx in head_retain {
                    let start = idx * head_dim;
                    if start + head_dim <= old.len() {
                        new_k.extend_from_slice(&old[start..start + head_dim]);
                    }
                }
                tri_keys[h] = new_k;
            }

            // Splice compressed cache: keep only retained indices in k_per_head
            // tri positions [0..sink_len) are sink (not in k_per_head)
            // tri positions [sink_len..] map to k_per_head positions [0..]
            let sink_n = cache.sink_len;
            let compressed_retain: Vec<usize> = retained.iter()
                .filter(|&&i| i >= sink_n)
                .map(|&i| i - sink_n)
                .collect();
            let compressed_count = cache.k_per_head[0].count;
            if !compressed_retain.is_empty() && compressed_retain.len() < compressed_count {
                for h in 0..n_kv_head {
                    cache.k_per_head[h] = cache.k_per_head[h].select_indices(&compressed_retain);
                }
                // Splice v_raw if present.
                // v_raw: [1, n_kv_head, total_len, head_dim]
                // Rebuild by selecting sink + retained compressed positions
                if let Some(ref v_tensor) = cache.v_raw {
                    let total_v_len = v_tensor.dim(2).unwrap_or(0);
                    if total_v_len > 0 && !compressed_retain.is_empty() {
                        let mut slices = Vec::new();
                        if sink_n > 0 {
                            if let Ok(s) = v_tensor.narrow(2, 0, sink_n) {
                                slices.push(s);
                            }
                        }
                        for &ci in &compressed_retain {
                            let vi = ci + sink_n;
                            if vi < total_v_len {
                                if let Ok(s) = v_tensor.narrow(2, vi, 1) {
                                    slices.push(s);
                                }
                            }
                        }
                        if slices.len() > 1 {
                            let slice_refs: Vec<&Tensor> = slices.iter().collect();
                            if let Ok(new_v) = Tensor::cat(&slice_refs, 2) {
                                cache.v_raw = Some(new_v);
                            }
                        } else if slices.len() == 1 {
                            cache.v_raw = Some(slices.into_iter().next().unwrap());
                        }
                    }
                }
                // Sprint F: save evicted tokens before GPU compact.
                // Priority 1: GPU cold tier (D2D, fast, no PCIe).
                // Priority 2: CPU offload (fallback when GPU cold tier is full).
                #[cfg(feature = "cuda")]
                if super::kv_cache::get_offload_enabled() {
                    if let Some(ref gpu) = *gpu_tq_cache {
                        let retain_set: std::collections::HashSet<usize> =
                            compressed_retain.iter().copied().collect();
                        let evicted_indices: Vec<usize> = (0..compressed_count)
                            .filter(|i| !retain_set.contains(i))
                            .collect();

                        if !evicted_indices.is_empty() {
                            // Lazy-init GPU cold tier
                            if gpu_cold_cache.is_none() {
                                let max_cold = std::env::var("TQ_MAX_COLD")
                                    .ok().and_then(|v| v.parse().ok()).unwrap_or(512usize);
                                if let Ok(cold) = super::kv_cache::GpuColdKv::new(
                                    gpu.stream.clone(), n_kv_head,
                                    head_dim, tq_bits,
                                    max_cold, tq_group_size,
                                ) {
                                    *gpu_cold_cache = Some(Box::new(cold));
                                }
                            }

                            // D2D copy evicted tokens to cold tier.
                            if let Some(ref mut cold) = *gpu_cold_cache {
                                let ms_hot = gpu.max_seq;
                                let has_decomp = gpu.decomp_count > 0;
                                for &idx in &evicted_indices {
                                    let seq_pos = cache.tri_key_positions
                                        .get(sink_n + idx).copied().unwrap_or(0);
                                    let _ = cold.push_from_hot(gpu, idx, seq_pos);
                                    if has_decomp && idx < gpu.decomp_count {
                                        let _ = cold.push_decomp_key(
                                            &gpu.decomp_cache, idx, ms_hot,
                                        );
                                    }
                                }
                            }

                            {
                                use std::sync::atomic::{AtomicBool, Ordering};
                                static COLD_PRINTED: AtomicBool = AtomicBool::new(false);
                                if !COLD_PRINTED.swap(true, Ordering::Relaxed) {
                                    let n_cold = gpu_cold_cache.as_ref()
                                        .map(|c| c.count).unwrap_or(0);
                                    eprintln!("[gpu-cold] L{}: {} tokens evicted → GPU cold tier ({} total cold)",
                                        layer_idx, evicted_indices.len(), n_cold);
                                }
                            }
                        }
                    }
                }

                // Compact GPU TQ cache in-place (D2D gather, no re-seed)
                #[cfg(feature = "cuda")]
                if let Some(ref mut gpu) = *gpu_tq_cache {
                    let _ = gpu.compact(&compressed_retain);
                }
            }

            cache.tri_key_positions = new_positions;

            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static EVICT_PRINTED: AtomicBool = AtomicBool::new(false);
                if !EVICT_PRINTED.swap(true, Ordering::Relaxed) {
                    eprintln!("[tri-evict] L{}: {}/{} retained (evicted {})",
                        layer_idx, retained.len(), n_tri_keys, evicted);
                }
            }
        }
    }
    cache.tri_tokens_since_eviction = 0;
}

/// KV Compaction pass: when the hot tier exceeds `TQ_COMPACT_THRESHOLD`,
/// condense the oldest hot tokens into a smaller set of synthetic keys
/// + attention biases (beta) to preserve attention behaviour. Fires at
/// most once per sequence (gated on `cache.compacted.is_none()`) to
/// avoid cascading quality loss.
#[allow(clippy::too_many_arguments)]
fn maybe_compact_cache(
    cache: &mut CompressedKvCache,
    q: &Tensor,
    n_rep: usize,
    seq_len: usize,
    total_len: usize,
    index_pos: usize,
    n_kv_head: usize,
    head_dim: usize,
    padded_head_dim: usize,
    layer_tq_config: &tq_kv::TurboQuantConfig,
    cos: &Tensor,
    sin: &Tensor,
    rope_style: RopeStyle,
    rope_dim: usize,
) -> Result<()> {
    let compact_threshold = get_compact_threshold();
    if compact_threshold == 0
        || cache.k_per_head[0].count <= compact_threshold
        || seq_len != 1
        || cache.compacted.is_some()
    {
        return Ok(());
    }
    let hot_count = cache.k_per_head[0].count;
    // Keep the most recent compact_threshold/2 tokens uncompacted.
    let keep_recent = compact_threshold / 2;
    let to_compact = hot_count.saturating_sub(keep_recent);
    if to_compact <= 8 {
        return Ok(());
    }
    let ratio = get_compact_ratio();
    let target_size = (to_compact * ratio / 100).max(4);

    // Decompress hot keys for compaction.
    // Pre-RoPE mode: decompress + apply RoPE so compaction works on post-RoPE keys.
    let k_decomp_full = if cache.pre_rope {
        let hot_start = cache.sink_len + cache.cold_len + cache.compacted_original_len;
        decompress_and_apply_rope(
            &cache.k_per_head, n_kv_head,
            padded_head_dim, DType::F32, q.device(), layer_tq_config,
            cos, sin, hot_start, rope_style, rope_dim,
        )?
    } else {
        decompress_compressed_keys(
            &cache.k_per_head, n_kv_head,
            padded_head_dim, DType::F32, q.device(), layer_tq_config,
        )?
    };
    let k_flat = k_decomp_full.flatten_all()?.to_vec1()?;

    // Use ALL query heads mapped to each KV head as reference queries.
    // For GQA: n_rep queries share each KV head.
    let q_flat = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let n_ref_queries = n_rep;

    // Get values for compaction
    let v_for_compact = if cache.value_bits == 0 {
        let v_tensor = cache.v_raw.as_ref().unwrap().to_dtype(DType::F32)?;
        // v_tensor is [1, n_kv_head, total_len, head_dim]
        v_tensor.flatten_all()?.to_vec1()?
    } else {
        let v_tensor = decompress_values_store(
            cache.v_compressed.as_ref().unwrap(),
            n_kv_head, head_dim, total_len, q.device(),
        )?;
        v_tensor.flatten_all()?.to_vec1()?
    };

    let hdim = head_dim;
    let padded = padded_head_dim;
    let mut compacted_heads = Vec::with_capacity(n_kv_head);
    for h in 0..n_kv_head {
        // Extract this head's oldest keys [to_compact * padded_head_dim]
        let k_head_offset = h * hot_count * padded;
        let k_head = &k_flat[k_head_offset..k_head_offset + to_compact * padded];
        // Trim padding to head_dim
        let mut k_trimmed = Vec::with_capacity(to_compact * hdim);
        for i in 0..to_compact {
            k_trimmed.extend_from_slice(&k_head[i * padded..i * padded + hdim]);
        }

        // Collect reference queries for this KV head (all n_rep mapped Q heads)
        let mut q_refs = Vec::with_capacity(n_ref_queries * hdim);
        for qh_offset in 0..n_rep {
            let qh = h * n_rep + qh_offset;
            q_refs.extend_from_slice(&q_flat[qh * hdim..(qh + 1) * hdim]);
        }

        // Extract values: v_for_compact layout is [n_kv_head, total_len, head_dim]
        // Hot values start at position (sink_len + cold_len)
        let v_start = cache.sink_len + cache.cold_len;
        let v_head_base = h * total_len * hdim;
        let v_head_offset = v_head_base + v_start * hdim;
        let v_head = &v_for_compact[v_head_offset..v_head_offset + to_compact * hdim];

        // Use TriAttention scoring for key selection if available,
        // otherwise fall back to mean-attention scoring.
        let pruning_method = if let Some(ref tri_cfg) = cache.tri_config {
            let hot_start = cache.sink_len + cache.cold_len + cache.compacted_original_len;
            let key_positions: Vec<usize> = (0..to_compact).map(|i| hot_start + i).collect();
            tq_kv::compaction::PruningMethod::TriAttention {
                config: tri_cfg.clone(),
                current_pos: index_pos + seq_len,
                key_positions,
                kv_head: h,
            }
        } else {
            tq_kv::compaction::PruningMethod::MeanAttention
        };
        let compacted = tq_kv::compaction::compact_head_with_method(
            &k_trimmed, v_head, &q_refs,
            to_compact, n_ref_queries, hdim, target_size,
            &pruning_method,
        );
        compacted_heads.push(CompactedCacheHead {
            keys: compacted.keys,
            beta: compacted.beta,
            values: compacted.values,
            t: compacted.t,
            head_dim: hdim,
        });
    }

    // Remove compacted tokens from hot tier.
    for h in 0..n_kv_head {
        let _front = cache.k_per_head[h].split_off_front(to_compact);
    }
    cache.compacted = Some(compacted_heads);
    cache.compacted_original_len += to_compact;
    Ok(())
}

/// Temporal decay pass: once per `decay_interval` tokens, move the
/// oldest hot-tier keys to the lower-bit cold tier. Preserves recent
/// tokens at full hot-bit precision while compressing aged ones.
fn apply_temporal_decay(
    cache: &mut CompressedKvCache,
    n_kv_head: usize,
    padded_head_dim: usize,
    rotation_seed: u64,
) {
    let Some(decay_cfg) = cache.decay_config.clone() else { return; };
    if cache.tokens_since_decay < decay_cfg.decay_interval {
        return;
    }
    cache.tokens_since_decay = 0;
    // Use the first (and typically only) tier for simplicity.
    let Some(tier) = decay_cfg.tiers.first() else { return; };

    let hot_count = cache.k_per_head[0].count;
    // Decay tokens whose age exceeds the tier threshold.
    let total_compressed = hot_count + cache.cold_len;
    if total_compressed <= tier.age_threshold || hot_count == 0 {
        return;
    }
    let to_decay = (total_compressed - tier.age_threshold).min(hot_count);
    if to_decay == 0 {
        return;
    }

    // Initialize cold tier on first decay.
    if cache.k_cold.is_none() {
        let mut cold = Vec::with_capacity(n_kv_head);
        for _ in 0..n_kv_head {
            cold.push(tq_kv::CompressedKeys::new_empty(
                tier.bits, padded_head_dim, rotation_seed,
            ));
        }
        cache.k_cold = Some(cold);
    }
    // Split front of hot, remap to cold bits, append to cold.
    let cold = cache.k_cold.as_mut().unwrap();
    for h in 0..n_kv_head {
        let front = cache.k_per_head[h].split_off_front(to_decay);
        let remapped = front.remap_bits(tier.bits);
        cold[h].append_from(&remapped);
    }
    cache.cold_len += to_decay;
}

/// Append the sink portion of the incoming `k` to the uncompressed
/// `sink_k` tensor. Returns the in-batch offset where compressible
/// (non-sink) tokens begin.
fn append_sink_keys(
    cache: &mut CompressedKvCache,
    k: &Tensor,
    global_start: usize,
    sink_n: usize,
    seq_len: usize,
) -> Result<usize> {
    if global_start >= sink_n {
        return Ok(0);
    }
    let sink_end = sink_n.min(global_start + seq_len);
    let n_sink_in_batch = sink_end - global_start;
    // k is [1, n_kv_head, seq_len, head_dim]
    let sink_k_batch = if n_sink_in_batch < seq_len {
        k.narrow(2, 0, n_sink_in_batch)?
    } else {
        k.clone()
    };
    cache.sink_k = Some(match &cache.sink_k {
        Some(prev) => Tensor::cat(&[prev, &sink_k_batch], 2)?,
        None => sink_k_batch,
    });
    cache.sink_len += n_sink_in_batch;
    Ok(n_sink_in_batch)
}

/// Append this batch's value tensor to the cache (either FP16 `v_raw` or
/// 4/8-bit compressed per-head). Called once per forward pass before
/// sink/compress key handling.
fn append_values_to_cache(
    cache: &mut CompressedKvCache,
    v: &Tensor,
    n_kv_head: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<()> {
    if cache.value_bits == 0 {
        // Uncompressed fp16 path
        if cache.cached_len > 0 {
            cache.v_raw = Some(match &cache.v_raw {
                Some(prev) => Tensor::cat(&[prev, v], 2)?,
                None => v.clone(),
            });
        } else {
            cache.v_raw = Some(v.clone());
        }
        return Ok(());
    }

    // Compressed path: quantize per head × position.
    // Use as_slice() for CPU or direct download for GPU (avoids extra allocs).
    let v_f32: Vec<f32> = if v.is_cuda() {
        v.to_vec1()?
    } else {
        v.as_slice().to_vec()
    };
    let v_store = cache.v_compressed.as_mut().unwrap();
    match v_store {
        CompressedValueStore::Bits8(v_comp) => {
            for h in 0..n_kv_head {
                for s in 0..seq_len {
                    let offset = (h * seq_len + s) * head_dim;
                    v_comp[h].append(&v_f32[offset..offset + head_dim]);
                }
            }
        }
        CompressedValueStore::Bits4(v_comp) => {
            for h in 0..n_kv_head {
                for s in 0..seq_len {
                    let offset = (h * seq_len + s) * head_dim;
                    v_comp[h].append(&v_f32[offset..offset + head_dim]);
                }
            }
        }
    }
    Ok(())
}

impl LayerWeights {
    /// Enumerate every `QWeight` owned by this layer — used by
    /// `LayerSwapManager` to inject/evict GPU caches per prefetched layer.
    ///
    /// Order: attention (q,k,v,o or qkv,o) then MLP (gate,up,down) or
    /// UpDown (up,down) or MoE (gate_inp,experts…). Biases and RmsNorms
    /// are small and stay resident — not included here.
    pub(crate) fn qweights(&self) -> Vec<&crate::qmatmul::QWeight> {
        let mut out = Vec::with_capacity(8);
        match &self.qkv {
            QkvWeights::Separate { wq, wk, wv } => {
                if let Some(qw) = wq.inner.as_qweight() { out.push(qw); }
                if let Some(qw) = wk.inner.as_qweight() { out.push(qw); }
                if let Some(qw) = wv.inner.as_qweight() { out.push(qw); }
            }
            QkvWeights::Merged { wqkv } => {
                if let Some(qw) = wqkv.inner.as_qweight() { out.push(qw); }
            }
        }
        if let Some(qw) = self.attention_wo.inner.as_qweight() { out.push(qw); }
        match &self.mlp_or_moe {
            MlpOrMoe::Mlp(m) => {
                if let Some(qw) = m.feed_forward_w1.inner.as_qweight() { out.push(qw); }
                if let Some(qw) = m.feed_forward_w2.inner.as_qweight() { out.push(qw); }
                if let Some(qw) = m.feed_forward_w3.inner.as_qweight() { out.push(qw); }
            }
            MlpOrMoe::UpDown(ud) => {
                if let Some(qw) = ud.ffn_up.inner.as_qweight() { out.push(qw); }
                if let Some(qw) = ud.ffn_down.inner.as_qweight() { out.push(qw); }
            }
            MlpOrMoe::MoE { feed_forward_gate_inp, experts, .. } => {
                if let Some(qw) = feed_forward_gate_inp.inner.as_qweight() { out.push(qw); }
                for exp in experts {
                    if let Some(qw) = exp.feed_forward_w1.inner.as_qweight() { out.push(qw); }
                    if let Some(qw) = exp.feed_forward_w2.inner.as_qweight() { out.push(qw); }
                    if let Some(qw) = exp.feed_forward_w3.inner.as_qweight() { out.push(qw); }
                }
            }
        }
        out
    }

    /// Same layout as `qweights()` but returns `Send+Sync` raw-pointer wrappers
    /// (`QWeightPtr`). Used to escape the borrow checker in the forward loop
    /// when we need to access `self.layers[i+1]`'s QWeights while holding
    /// `&mut self.layers[i]`, while still permitting the owning `Engine` to
    /// move across threads (serve.rs' tokio blocking tasks).
    ///
    /// SAFETY: caller must ensure the owning `Vec<LayerWeights>` is not
    /// reallocated while pointers are in use, and only `&self` methods are
    /// invoked through them.
    #[cfg(feature = "cuda")]
    pub(crate) fn qweight_ptrs(&self) -> Vec<crate::layer_swap::QWeightPtr> {
        self.qweights()
            .into_iter()
            .map(|p| crate::layer_swap::QWeightPtr(p as *const _))
            .collect()
    }

    /// After warmup has uploaded every per-layer QWeight to GPU, release the
    /// CPU-side raw bytes for Owned (non-mmap) weights. Saves roughly the
    /// packed-model size of heap RAM on non-swap runs. Mmap weights are
    /// already zero-copy and stay. Swap mode is gated off inside
    /// `release_cpu_after_gpu` itself.
    #[cfg(feature = "cuda")]
    pub(crate) fn release_qweight_cpu(&mut self) {
        use crate::qmatmul as qmm;
        let process = |qm: &mut QMatMul| {
            if let qmm::QMatMul::Quantized(qw) = &mut qm.inner {
                qw.release_cpu_after_gpu();
            }
        };
        match &mut self.qkv {
            QkvWeights::Separate { wq, wk, wv } => { process(wq); process(wk); process(wv); }
            QkvWeights::Merged { wqkv } => process(wqkv),
        }
        process(&mut self.attention_wo);
        match &mut self.mlp_or_moe {
            MlpOrMoe::Mlp(m) => {
                process(&mut m.feed_forward_w1);
                process(&mut m.feed_forward_w2);
                process(&mut m.feed_forward_w3);
            }
            MlpOrMoe::UpDown(ud) => {
                process(&mut ud.ffn_up);
                process(&mut ud.ffn_down);
            }
            MlpOrMoe::MoE { feed_forward_gate_inp, experts, .. } => {
                process(feed_forward_gate_inp);
                for e in experts {
                    process(&mut e.feed_forward_w1);
                    process(&mut e.feed_forward_w2);
                    process(&mut e.feed_forward_w3);
                }
            }
        }
    }

    pub(crate) fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, n_head, seq_len, _head_dim) = x.dims4()?;

        // GPU path: single kernel launch, no shape/stride metadata uploads.
        #[cfg(feature = "cuda")]
        if x.is_cuda() && self.cos.is_cuda() {
            if let Some(reg) = crate::cuda::kernels::global_registry() {
                let mut x_out = x.clone();
                // Check if graph-mode GPU position scalar is available
                let gpu_pos_ref = super::ROPE_POS_GPU_PTR.with(|p| p.get());
                if gpu_pos_ref != 0 {
                    // Graph mode: read position from GPU scalar
                    // SAFETY: pointer set by forward() before capture, valid for model lifetime
                    let gpu_slice = unsafe { &*(gpu_pos_ref as *const cudarc::driver::CudaSlice<i32>) };
                    let rope_with_gpu = match self.rope_style {
                        RopeStyle::Halved => crate::cuda::kernels::rope_halved_with_gpu_pos,
                        RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved_with_gpu_pos,
                    };
                    rope_with_gpu(
                        reg, x_out.cuda_data_mut(),
                        self.cos.cuda_data(), self.sin.cuda_data(),
                        seq_len, n_head, self.head_dim, self.rope_dim, index_pos,
                        Some(gpu_slice),
                    ).map_err(|e| TqError::Msg(format!("rope GPU: {}", e)))?;
                } else {
                    let rope_fn = match self.rope_style {
                        RopeStyle::Halved => crate::cuda::kernels::rope_halved,
                        RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved,
                    };
                    rope_fn(
                        reg, x_out.cuda_data_mut(),
                        self.cos.cuda_data(), self.sin.cuda_data(),
                        seq_len, n_head, self.head_dim, self.rope_dim, index_pos,
                    ).map_err(|e| TqError::Msg(format!("rope GPU: {}", e)))?;
                }
                return Ok(x_out);
            }
        }

        // CPU fallback: tensor-op RoPE
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let x = x.contiguous()?;
        if self.rope_dim < self.head_dim {
            let x_rope = x.narrow(3, 0, self.rope_dim)?;
            let x_pass = x.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?;
            let x_rotated = match self.rope_style {
                RopeStyle::Halved => rope_halved(&x_rope, &cos, &sin)?,
                RopeStyle::Interleaved => rope_interleaved(&x_rope, &cos, &sin)?,
            };
            Tensor::cat(&[&x_rotated, &x_pass], 3)
        } else {
            match self.rope_style {
                RopeStyle::Halved => rope_halved(&x, &cos, &sin),
                RopeStyle::Interleaved => rope_interleaved(&x, &cos, &sin),
            }
        }
    }

    /// Compute QKV projections, apply biases, reshape to head layout, SmoothAttention,
    /// calibration hooks, and RoPE. Returns all intermediates needed by attention paths.
    #[allow(clippy::type_complexity)]
    fn compute_qkv_and_rope(
        &self,
        x: &Tensor,
        pre_qkv: Option<(Tensor, Tensor, Tensor)>,
        index_pos: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<(
        Tensor, Tensor, Tensor,     // q, k, v (post-RoPE)
        Option<Tensor>,             // k_pre_rope (for TQ/TriAttention)
        bool,                       // pre_rope_mode
        bool,                       // tri_enabled
        Option<u8>,                 // layer_bits
        usize,                      // n_rep
        usize,                      // b_sz
        usize,                      // seq_len
    )> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let used_fused_qkv = pre_qkv.is_some();
        let (mut q, mut k, mut v) = if let Some(qkv) = pre_qkv {
            qkv
        } else {
            match &self.qkv {
                QkvWeights::Separate { wq, wk, wv } => {
                    (wq.forward(x, backend)?, wk.forward(x, backend)?, wv.forward(x, backend)?)
                }
                QkvWeights::Merged { wqkv } => {
                    let qkv = wqkv.forward(x, backend)?;
                    let q_size = self.n_head * self.head_dim;
                    let kv_size = self.n_kv_head * self.head_dim;
                    let q = qkv.narrow(2, 0, q_size)?;
                    let k = qkv.narrow(2, q_size, kv_size)?;
                    let v = qkv.narrow(2, q_size + kv_size, kv_size)?;
                    (q, k, v)
                }
            }
        };

        // Calibration hook: collect RAW key vectors (before attention bias, before RoPE).
        if crate::calibrate::CALIBRATION_COLLECTOR.get().is_some() {
            let k_for_cal = k.reshape(vec![b_sz, seq_len, self.n_kv_head, self.head_dim])?
                .transpose(1, 2)?.contiguous()?;
            if let Ok(k_f32) = k_for_cal.to_dtype(DType::F32)?.flatten_all()?.to_vec1() {
                crate::calibrate::maybe_collect(&k_f32, self.n_kv_head, seq_len, self.head_dim, self.layer_idx);
            }
        }

        // Apply biases if present (Qwen2 has them, Llama/Phi/Gemma don't).
        if !used_fused_qkv {
            if let Some(bq) = &self.attention_bq {
                q = q.broadcast_add(bq)?;
            }
            if let Some(bk) = &self.attention_bk {
                k = k.broadcast_add(bk)?;
            }
            if let Some(bv) = &self.attention_bv {
                v = v.broadcast_add(bv)?;
            }
        }
        let q = q.reshape(vec![b_sz, seq_len, self.n_head, self.head_dim])?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(vec![b_sz, seq_len, self.n_kv_head, self.head_dim])?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(vec![b_sz, seq_len, self.n_kv_head, self.head_dim])?.transpose(1, 2)?.contiguous()?;

        // SmoothAttention: migrate K outliers to Q BEFORE RoPE.
        let (q, k) = if let (Some(ref q_scales), Some(ref k_scales)) = (&self.smooth_q_scales, &self.smooth_k_scales) {
            (q.broadcast_mul(q_scales)?, k.broadcast_mul(k_scales)?)
        } else {
            (q, k)
        };

        // TriAttention calibration: collect pre-RoPE Q and K for center statistics.
        if crate::calibrate::CALIBRATION_COLLECTOR.get().is_some() {
            if let (Ok(q_f32), Ok(k_f32)) = (
                q.to_dtype(DType::F32).and_then(|t| t.flatten_all()?.to_vec1()),
                k.to_dtype(DType::F32).and_then(|t| t.flatten_all()?.to_vec1()),
            ) {
                crate::calibrate::maybe_collect_pre_rope(
                    &q_f32, &k_f32, self.n_head, self.n_kv_head, seq_len, self.head_dim,
                );
            }
        }

        // Pre-RoPE quantization: save k BEFORE RoPE for compression.
        let pre_rope_mode = self.tq_config.pre_rope || get_pre_rope();
        let tri_enabled = get_triattention_enabled();
        let k_pre_rope = if pre_rope_mode || tri_enabled { Some(k.clone()) } else { None };

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let layer_bits = get_layer_bits(self.layer_idx, self.tq_config.bits, &self.tq_config, self.n_layers);
        let n_rep = self.n_head / self.n_kv_head;

        Ok((q, k, v, k_pre_rope, pre_rope_mode, tri_enabled, layer_bits, n_rep, b_sz, seq_len))
    }

    /// Uncompressed attention path: standard fp16 KV cache (first N layers).
    /// GPU path uses pre-allocated GpuKvCache, CPU path uses Tensor::cat.
    fn uncompressed_attention(
        &mut self,
        q: &Tensor,
        k: Tensor,
        v: Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        n_rep: usize,
        b_sz: usize,
        seq_len: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        let gpu_kv_disabled = {
            static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *C.get_or_init(|| std::env::var("TQ_NO_GPU_KV").map(|v| v == "1").unwrap_or(false))
        };
        #[cfg(feature = "cuda")]
        if k.is_cuda() && seq_len == 1 && !gpu_kv_disabled {
            // Initialize GPU KV cache on first decode step
            if self.gpu_kv_cache.is_none() {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    match GpuKvCache::new(reg.stream.clone(), self.n_kv_head, self.head_dim, get_max_kv_seq()) {
                        Ok(mut cache) => {
                            if let Some((prev_k, prev_v)) = &self.kv_cache {
                                let prefill_len = prev_k.shape()[2];
                                let gk = if prev_k.is_cuda() { prev_k.clone() } else {
                                    prev_k.to_device_auto().unwrap_or_else(|_| prev_k.clone())
                                };
                                let gv = if prev_v.is_cuda() { prev_v.clone() } else {
                                    prev_v.to_device_auto().unwrap_or_else(|_| prev_v.clone())
                                };
                                if gk.is_cuda() && gv.is_cuda() {
                                    if let Err(e) = cache.append(&gk, &gv, prefill_len) {
                                        eprintln!("[gpu-kv] L{}: seed failed: {}", self.layer_idx, e);
                                    }
                                }
                            }
                            self.gpu_kv_cache = Some(cache);
                        }
                        Err(e) => eprintln!("[gpu-kv] L{}: alloc failed: {}", self.layer_idx, e),
                    }
                }
            }
            if let Some(ref mut gpu_kv) = self.gpu_kv_cache {
                if index_pos == 0 {
                    gpu_kv.reset();
                }
                gpu_kv.append(&k, &v, seq_len)?;

                let k_full = gpu_kv.k_tensor();
                let v_full = gpu_kv.v_tensor();
                let kv_mask = gpu_kv.mask_tensor();

                let k_full = repeat_kv(k_full, n_rep)?;
                let v_full = repeat_kv(v_full, n_rep)?;

                let att = (q.matmul(&k_full.t()?)? / self.attn_scale_denom)?;
                let att = if let Some(cap) = self.attn_logit_softcap {
                    apply_softcap(&att, cap)?
                } else { att };
                let kv_mask = kv_mask.broadcast_as(att.shape())?;
                let att = (att + kv_mask)?;

                let gpu_debug = {
                    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    *C.get_or_init(|| std::env::var("TQ_GPU_DEBUG").map(|v| v == "1").unwrap_or(false))
                };
                if gpu_debug && self.layer_idx < 2 {
                    if let Ok(scores) = att.to_vec1() {
                        let max_seq_show = gpu_kv.seq_len.min(10);
                        let head0_scores: Vec<f32> = scores[..max_seq_show].to_vec();
                        eprintln!("[gpu-debug] L{} GpuKV att-pre-softmax head0[..{}]: {:?}",
                            self.layer_idx, max_seq_show, head0_scores);
                        let n_pad = scores.len() / self.n_head;
                        let n_masked = scores[gpu_kv.seq_len..n_pad].iter()
                            .filter(|&&v| v < -1e9).count();
                        eprintln!("[gpu-debug] L{} GpuKV valid={} max_seq={} masked_positions={}",
                            self.layer_idx, gpu_kv.seq_len, gpu_kv.max_seq, n_masked);
                    }
                }

                let att = softmax_last_dim(&att, backend)?;

                if gpu_debug && self.layer_idx < 2 {
                    if let Ok(weights) = att.to_vec1() {
                        let max_seq_show = gpu_kv.seq_len.min(10);
                        let head0_weights: Vec<f32> = weights[..max_seq_show].to_vec();
                        let head0_sum: f32 = weights[..gpu_kv.max_seq].iter().sum();
                        eprintln!("[gpu-debug] L{} GpuKV att-post-softmax head0[..{}]: {:?} sum={:.6}",
                            self.layer_idx, max_seq_show, head0_weights, head0_sum);
                    }
                }

                let y = att.matmul(&v_full.contiguous()?)?;
                let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
                return self.attention_wo.forward(&y, backend);
            }
        }

        // CPU fallback / prefill: original cat()-based KV cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[prev_k, &k], 2)?;
                    let v = Tensor::cat(&[prev_v, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;
        let att = (q.matmul(&k.t()?)? / self.attn_scale_denom)?;
        // Build causal mask from actual attention dimensions (handles batched verify
        // where seq_len < total_kv_len due to existing KV cache).
        let att = if seq_len > 1 {
            let total_kv_len = att.dim(3)?;
            let offset = total_kv_len - seq_len;
            let mask_data: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    let qpos = offset + i;
                    (0..total_kv_len).map(move |j| if j <= qpos { 0.0f32 } else { -1e10f32 })
                })
                .collect();
            let mut mask = Tensor::from_slice(&mask_data, vec![seq_len, total_kv_len], &Device::Cpu)?;
            #[cfg(feature = "cuda")]
            if att.is_cuda() {
                if let Ok(gpu) = mask.to_device_auto() { mask = gpu; }
            }
            let mask4d = mask.unsqueeze(0)?.unsqueeze(0)?;
            let mask4d = mask4d.broadcast_as(att.shape())?;
            (att + mask4d)?
        } else {
            att
        };
        let att = softmax_last_dim(&att, backend)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
        self.attention_wo.forward(&y, backend)
    }

    /// Entry point: 15-line dispatch to either the compressed (TQ KV cache)
    /// path or the uncompressed path. The compressed body — previously a
    /// 1830-line inline block — lives in `forward_attn_compressed`.
    ///
    /// Split per codescope audit (2026-04-13): the original monolith had
    /// 3285 callees and was the #1 structural debt flagged as a prerequisite
    /// for `gemma2-fix` and `gemma4-support` (both add their own attention
    /// variants and would otherwise compound the dispatch hell).
    pub(crate) fn forward_attn(
        &mut self,
        x: &Tensor,
        pre_qkv: Option<(Tensor, Tensor, Tensor)>,
        mask: Option<&Tensor>,
        index_pos: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<Tensor> {
        let span = self.span_attn.clone();
        let _enter = span.enter();
        let (q, k, v, k_pre_rope, pre_rope_mode, tri_enabled, layer_bits, n_rep, b_sz, seq_len) =
            self.compute_qkv_and_rope(x, pre_qkv, index_pos, backend)?;

        match layer_bits {
            Some(effective_bits) => self.forward_attn_compressed(
                q, k, v, k_pre_rope, pre_rope_mode, tri_enabled,
                effective_bits, n_rep, b_sz, seq_len,
                mask, index_pos, backend,
            ),
            None => self.uncompressed_attention(&q, k, v, mask, index_pos, n_rep, b_sz, seq_len, backend),
        }
    }

    /// Lazy first-call init for the TQ compressed KV cache. Called from
    /// `forward_attn_compressed` once per sequence (gated on
    /// `self.kv_compressed.is_none()`). Split out 2026-04-13 to shrink
    /// the 1857-line monolith.
    #[allow(clippy::too_many_arguments)]
    fn ensure_compressed_cache(
        &mut self,
        layer_tq_config: &tq_kv::TurboQuantConfig,
        per_head_bits: &Option<Vec<u8>>,
        effective_bits: u8,
        vbits: u8,
        cache_dtype: DType,
        pre_rope_mode: bool,
    ) {
        if self.kv_compressed.is_some() {
            return;
        }

        let mut k_per_head = Vec::with_capacity(self.n_kv_head);
        let gs = layer_tq_config.group_size;
        for h in 0..self.n_kv_head {
            let hbits = per_head_bits.as_ref()
                .map(|phb| phb[h])
                .unwrap_or(effective_bits);
            k_per_head.push(tq_kv::CompressedKeys::new_empty_grouped(
                hbits, self.padded_head_dim, layer_tq_config.rotation_seed, gs,
            ));
        }
        let (v_raw, v_compressed) = match vbits {
            0 => (None, None),
            4 => {
                let gs = layer_tq_config.group_size.max(32);
                let mut v_heads: Vec<tq_kv::CompressedValues4Bit> = Vec::with_capacity(self.n_kv_head);
                for _ in 0..self.n_kv_head {
                    v_heads.push(tq_kv::CompressedValues4Bit::new_empty(self.head_dim, gs));
                }
                (None, Some(CompressedValueStore::Bits4(v_heads)))
            }
            _ => {
                // 8-bit (default for any non-zero, non-4 value)
                let mut v_heads: Vec<tq_kv::CompressedValues> = Vec::with_capacity(self.n_kv_head);
                for _ in 0..self.n_kv_head {
                    v_heads.push(tq_kv::CompressedValues::new_empty(self.head_dim));
                }
                (None, Some(CompressedValueStore::Bits8(v_heads)))
            }
        };
        self.kv_compressed = Some(CompressedKvCache {
            k_per_head,
            k_cold: None,
            cold_len: 0,
            decay_config: get_decay_config(),
            tokens_since_decay: 0,
            sink_k: None,
            sink_len: 0,
            compacted: None,
            compacted_original_len: 0,
            v_raw,
            v_compressed,
            value_bits: vbits,
            cached_len: 0,
            dtype: cache_dtype,
            pre_rope: pre_rope_mode,
            tri_config: if get_triattention_enabled() {
                crate::calibrate::TRIATTENTION_CONFIG.get().cloned()
            } else { None },
            tri_keys_pre_rope: if get_triattention_enabled() {
                Some(vec![Vec::new(); self.n_kv_head])
            } else { None },
            tri_key_positions: Vec::new(),
            tri_tokens_since_eviction: 0,
        });
    }

    /// TQ-compressed KV cache attention path (sink tokens + past-only
    /// quantisation + GPU fused compress + grouped / per-head bit variants
    /// + TriAttention eviction). Extracted from forward_attn on
    /// 2026-04-13.
    #[allow(clippy::too_many_arguments)]
    fn forward_attn_compressed(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        k_pre_rope: Option<Tensor>,
        pre_rope_mode: bool,
        tri_enabled: bool,
        effective_bits: u8,
        n_rep: usize,
        b_sz: usize,
        seq_len: usize,
        mask: Option<&Tensor>,
        index_pos: usize,
        backend: &dyn ComputeBackend,
    ) -> Result<Tensor> {
        // Body lifted verbatim from the original forward_attn's
        // `if use_compression { ... }` branch — locals referenced here
        // were all either (a) returned from compute_qkv_and_rope (now
        // parameters), (b) `&mut self` fields, or (c) function parameters.
        {
            // Apply per-layer bit width if different from default.
            // `effective_bits` is now a function parameter (was
            // `layer_bits.unwrap()` in the original inline block).
            let mut layer_tq_config = if effective_bits != self.tq_config.bits {
                tq_kv::TurboQuantConfig { bits: effective_bits, ..self.tq_config.clone() }
            } else {
                self.tq_config.clone()
            };
            // When SmoothAttention is active, disable channel_scales in compression
            // to avoid double-application (SmoothAttention already handled the scaling)
            if self.smooth_k_scales.is_some() {
                layer_tq_config.channel_scales = None;
            }
            // Per-head adaptive bitwidth: resolve from TQ_HEAD_BITS env var or config
            let per_head_bits = resolve_per_head_bits(self.n_kv_head, effective_bits)
                .or_else(|| self.tq_config.per_head_bits.clone());
            // COMPRESSED PATH: TurboQuant KV cache with 3-fix quality enhancement
            //
            // Three paper-backed techniques to handle compound error (W4+KV4):
            //   Fix 1 — Sink token preservation: first N tokens' keys stay FP16 (KVSink)
            //   Fix 2 — Past-Only Quantization: current token's key is FP16 during
            //           attention, compressed into cache AFTER (WKVQuant)
            //   Fix 3 — Per-channel scaling: (future — placeholder for now)

            // Reset cache on new sequence
            if index_pos == 0 {
                self.kv_compressed = None;
            }

            let cache_dtype = k.dtype();
            let sink_n = get_sink_tokens(&self.tq_config);

            // Initialize cache on first call
            let vbits = get_value_bits();
            self.ensure_compressed_cache(
                &layer_tq_config, &per_head_bits, effective_bits, vbits,
                cache_dtype, pre_rope_mode,
            );

            let cache = self.kv_compressed.as_mut().unwrap();
            let prev_total = cache.cached_len;

            // Determine which tokens in this batch are sink vs compressible
            // Sink tokens: positions [0, sink_n) in the sequence
            // POQ: during generation (seq_len=1), current token is NOT compressed yet
            let global_start = prev_total; // first position in this batch

            // --- Store values ---
            append_values_to_cache(cache, &v, self.n_kv_head, seq_len, self.head_dim)?;

            // --- Handle sink tokens (FP16, uncompressed) ---
            // Returns the in-batch offset where compressed (non-sink) tokens
            // begin — i.e. how many of this batch's tokens went into sink_k.
            let compress_start = append_sink_keys(cache, &k, global_start, sink_n, seq_len)?;

            // --- Compress non-sink tokens into cache ---
            // All tokens after sink position get compressed into cache.
            // POQ twist: during generation (seq_len=1), the current token is ALSO
            // compressed into cache, but during attention we use the FP16 original
            // instead of the decompressed version (Fix 2: Past-Only Quantization).
            let tokens_to_compress = seq_len.saturating_sub(compress_start);

            // Check GPU compress eligibility once (used by both compress and skip logic)
            #[cfg(feature = "cuda")]
            let gpu_compress_ok = tokens_to_compress > 0 && seq_len == 1
                && k.is_cuda()
                && !pre_rope_mode
                && layer_tq_config.calibrated_codebook.is_none()
                && layer_tq_config.residual_bits == 0
                && layer_tq_config.outlier_k == 0
                && per_head_bits.is_none()
                && self.padded_head_dim == self.head_dim
                && self.gpu_tq_cache.is_some();
            #[cfg(not(feature = "cuda"))]
            let gpu_compress_ok = false;

            if tokens_to_compress > 0 && seq_len == 1 && gpu_compress_ok {
                // ── GPU fast path: single-token decode compression ──
                {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static C: AtomicU64 = AtomicU64::new(0);
                    if C.fetch_add(1, Ordering::Relaxed) == 0 {
                        eprintln!("[gpu-tq-compress] GPU fast path ACTIVE");
                    }
                }

                #[cfg(feature = "cuda")]
                if gpu_compress_ok {
                    if let Some(reg) = crate::cuda::kernels::global_registry() {
                        // Pre-RoPE bug fix: previously this site always used &k
                        // (the post-RoPE tensor), even when pre_rope_mode was on.
                        // The CPU fallback at L720+ already honored k_pre_rope;
                        // the GPU path didn't, so the GPU TQ cache silently stored
                        // post-RoPE keys regardless of TQ_PRE_ROPE. The all-GPU
                        // decode then re-RoPEd them on read, producing a double
                        // rotation that quietly degraded generation quality.
                        let k_source = if pre_rope_mode {
                            k_pre_rope.as_ref().expect("pre_rope_mode requires k_pre_rope")
                        } else {
                            &k
                        };
                        // K is [1, n_kv_head, 1, head_dim] for decode
                        let k_flat = k_source.reshape(vec![self.n_kv_head, self.head_dim])?;
                        let k_gpu = k_flat.cuda_data();

                        let bits = layer_tq_config.bits;
                        let n_centroids = 1usize << bits;
                        let bytes_per_key = (self.head_dim * bits as usize + 7) / 8;

                        // Signs already on GPU? Upload once if not.
                        if self.signs_gpu.is_none() {
                            self.signs_gpu = Some(reg.stream.clone_htod(&self.signs)
                                .map_err(|e| TqError::Msg(format!("signs upload: {}", e)))?);
                        }
                        if self.boundaries_gpu.is_none() {
                            let boundaries = tq_kv::codebook::get_boundaries(bits);
                            self.boundaries_gpu = Some(reg.stream.clone_htod(boundaries)
                                .map_err(|e| TqError::Msg(format!("boundaries upload: {}", e)))?);
                        }
                        if self.centroids_gpu.is_none() {
                            let centroids = tq_kv::codebook::get_centroids(bits);
                            self.centroids_gpu = Some(reg.stream.clone_htod(centroids)
                                .map_err(|e| TqError::Msg(format!("centroids upload: {}", e)))?);
                        }
                        // KIVI per-channel sigma upload (once)
                        if self.channel_sigma_gpu.is_none() {
                            if let Some(ref sigma) = self.tq_config.rotated_channel_sigma {
                                self.channel_sigma_gpu = Some(reg.stream.clone_htod(sigma)
                                    .map_err(|e| TqError::Msg(format!("channel_sigma upload: {}", e)))?);
                            }
                        }

                        // Sprint 1C: grouped GPU compress path. When the
                        // GpuCompressedKv was allocated with n_groups>1, run
                        // tq_compress_key_grouped and scatter to gpu_tq.gnorms
                        // instead of the legacy per-vector path. center_keys
                        // + KIVI per-channel sigma are ignored on the grouped
                        // path for now (see Sprint 1A commit), so we fall back
                        // to per-vector if either is on.
                        // Sprint 1D: flipped to default-enabled after parity
                        // verification (5-prompt suite identical, perf within
                        // run-to-run noise). TQ_GPU_GROUPED=0 disables it.
                        let want_grouped = super::kv_cache::get_gpu_grouped_enabled();
                        let gpu_tq_ref = self.gpu_tq_cache.as_ref().unwrap();
                        let can_grouped = want_grouped
                            && gpu_tq_ref.n_groups > 1
                            && !self.tq_config.center_keys
                            && self.channel_sigma_gpu.is_none();

                        // V data: already on GPU in scratch or k tensor
                        let v_flat = v.reshape(vec![self.n_kv_head, self.head_dim])?;
                        let v_gpu = v_flat.cuda_data();

                        if can_grouped {
                            let n_groups = gpu_tq_ref.n_groups;
                            let group_size = gpu_tq_ref.group_size;
                            // Alloc temp GPU buffers (grouped path)
                            let mut packed_gpu: cudarc::driver::CudaSlice<u8> = reg.stream.alloc_zeros(
                                self.n_kv_head * bytes_per_key
                            ).map_err(|e| TqError::Msg(format!("packed alloc: {}", e)))?;
                            let mut gnorms_gpu: cudarc::driver::CudaSlice<f32> = reg.stream.alloc_zeros(
                                self.n_kv_head * n_groups
                            ).map_err(|e| TqError::Msg(format!("gnorms alloc: {}", e)))?;

                            crate::cuda::kernels::tq_compress_key_grouped(
                                reg, k_gpu,
                                self.signs_gpu.as_ref().unwrap(),
                                self.boundaries_gpu.as_ref().unwrap(),
                                self.centroids_gpu.as_ref().unwrap(),
                                &mut packed_gpu, &mut gnorms_gpu,
                                self.n_kv_head, self.head_dim, group_size,
                                n_centroids, bytes_per_key,
                            ).map_err(|e| TqError::Msg(format!("tq_compress_key_grouped: {}", e)))?;

                            let gpu_tq = self.gpu_tq_cache.as_mut().unwrap();
                            gpu_tq.append_gpu_grouped(&packed_gpu, &gnorms_gpu, v_gpu)?;
                        } else {
                            // Legacy per-vector GPU compress path
                            let mut packed_gpu: cudarc::driver::CudaSlice<u8> = reg.stream.alloc_zeros(
                                self.n_kv_head * bytes_per_key
                            ).map_err(|e| TqError::Msg(format!("packed alloc: {}", e)))?;
                            let mut norms_gpu: cudarc::driver::CudaSlice<f32> = reg.stream.alloc_zeros(
                                self.n_kv_head
                            ).map_err(|e| TqError::Msg(format!("norms alloc: {}", e)))?;

                            crate::cuda::kernels::tq_compress_key(
                                reg, k_gpu,
                                self.signs_gpu.as_ref().unwrap(),
                                self.boundaries_gpu.as_ref().unwrap(),
                                self.centroids_gpu.as_ref().unwrap(),
                                &mut packed_gpu, &mut norms_gpu,
                                None,
                                self.channel_sigma_gpu.as_ref(),
                                self.n_kv_head, self.head_dim, n_centroids, bytes_per_key,
                                self.tq_config.center_keys,
                            ).map_err(|e| TqError::Msg(format!("tq_compress_key: {}", e)))?;

                            let gpu_tq = self.gpu_tq_cache.as_mut().unwrap();
                            gpu_tq.append_gpu(&packed_gpu, &norms_gpu, v_gpu)?;
                        }

                        // Don't increment cache.cached_len here — it's incremented
                        // after the compress section at line ~1917.

                        // V raw: only needed for sink token positions (first N tokens).
                        // After sink phase, V is stored in GpuCompressedKv.v_data directly.
                        // Skip CPU V cat during decode — eliminates GPU→CPU download overhead.
                        // (Sink V was already captured during prefill via CPU compress path)
                    }
                }
            }

            if tokens_to_compress > 0 && !(seq_len == 1 && gpu_compress_ok) {
                // ── CPU fallback: original compress path ──
                // Pre-RoPE mode: compress pre-RoPE keys (position-independent stats).
                let k_source = if pre_rope_mode {
                    k_pre_rope.as_ref().unwrap()
                } else {
                    &k
                };
                let k_to_compress = k_source.narrow(2, compress_start, tokens_to_compress)?;
                // Direct download: skip contiguous+to_dtype+flatten chain for decode (already f32 contiguous)
                let k_flat: Vec<f32> = if k_to_compress.is_cuda() {
                    k_to_compress.contiguous()?.to_vec1()?
                } else {
                    k_to_compress.contiguous()?.as_slice().to_vec()
                };
                let hdim = self.head_dim;
                let use_grouped = layer_tq_config.group_size > 0 && hdim % layer_tq_config.group_size == 0;
                for h in 0..self.n_kv_head {
                    // Per-head adaptive bitwidth: use head-specific config if assigned
                    let head_config = match per_head_bits {
                        Some(ref phb) if phb[h] != layer_tq_config.bits => {
                            // Clear calibrated codebook if bit width differs — it was
                            // calibrated for layer_tq_config.bits, not for this head's bits.
                            tq_kv::TurboQuantConfig {
                                bits: phb[h],
                                calibrated_codebook: None,
                                ..layer_tq_config.clone()
                            }
                        }
                        _ => layer_tq_config.clone(),
                    };
                    for s in 0..tokens_to_compress {
                        let offset = (h * tokens_to_compress + s) * hdim;
                        let key_vec = &k_flat[offset..offset + hdim];
                        if use_grouped {
                            let (packed, gnorms, residual, outliers) = tq_kv::compress_single_key_grouped(
                                key_vec, hdim, &head_config, &self.signs,
                            );
                            // Set residual/outlier bits on first append
                            if cache.k_per_head[h].residual_bits == 0 && residual.is_some() {
                                cache.k_per_head[h].residual_bits = head_config.residual_bits;
                            }
                            if cache.k_per_head[h].outlier_k == 0 && outliers.is_some() {
                                cache.k_per_head[h].outlier_k = head_config.outlier_k;
                            }
                            // Append outliers
                            if let Some((oi, ov)) = outliers {
                                if cache.k_per_head[h].outlier_indices.is_none() {
                                    cache.k_per_head[h].outlier_indices = Some(Vec::new());
                                    cache.k_per_head[h].outlier_values = Some(Vec::new());
                                }
                                cache.k_per_head[h].outlier_indices.as_mut().unwrap().extend_from_slice(&oi);
                                cache.k_per_head[h].outlier_values.as_mut().unwrap().extend_from_slice(&ov);
                            }
                            cache.k_per_head[h].append_raw_grouped(&packed, &gnorms, residual);
                        } else if self.padded_head_dim > hdim {
                            let mut padded = vec![0.0f32; self.padded_head_dim];
                            padded[..hdim].copy_from_slice(key_vec);
                            let (packed, norm, _token_mean) = tq_kv::compress_single_key_with_signs(
                                &padded, self.padded_head_dim, &head_config, &self.signs,
                            );
                            cache.k_per_head[h].append_raw(&packed, norm);
                        } else {
                            let (packed, norm, _token_mean) = tq_kv::compress_single_key_with_signs(
                                key_vec, hdim, &head_config, &self.signs,
                            );
                            cache.k_per_head[h].append_raw(&packed, norm);
                        }
                    }
                }
            }

            // --- GPU append: upload newly compressed tokens to persistent GPU buffers ---
            #[cfg(feature = "cuda")]
            if tokens_to_compress > 0 && seq_len == 1 {
                if let Some(reg) = crate::cuda::kernels::global_registry() {
                    // Initialize GPU compressed KV on first use
                    if self.gpu_tq_cache.is_none() {
                        let head_bits = cache.k_per_head[0].bits;
                        let cal_cb: Option<Vec<f32>> = layer_tq_config.calibrated_codebook
                            .as_ref().map(|cb| cb.centroids.clone());
                        let cb: &[f32] = match cal_cb.as_deref() {
                            Some(cal) => cal,
                            None => tq_kv::codebook::get_centroids(head_bits),
                        };
                        let max_seq = std::env::var("TQ_MAX_SEQ").ok()
                            .and_then(|v| v.parse().ok()).unwrap_or(2048usize);
                        // Sprint 1A: pass through layer_tq_config.group_size so the
                        // grouped norms buffer is allocated when grouped mode is on.
                        // The grouped path still has to be flipped on by Sprint 1C
                        // (TQ_GPU_GROUPED) before this buffer is actually written.
                        let group_size = layer_tq_config.group_size;
                        match GpuCompressedKv::new(
                            reg.stream.clone(), self.n_kv_head, self.n_head,
                            self.head_dim, head_bits, max_seq, cb, group_size,
                        ) {
                            Ok(mut gpu) => {
                                // Seed with all existing compressed keys + actual V data.
                                // Detect grouped CPU compression: if cache.k_per_head[h].norms
                                // has count*n_groups entries instead of count, the CPU side
                                // ran the per-group path (compress_single_key_grouped) and
                                // every token contributes n_groups floats to .norms. Reading
                                // .norms[pos] in that case picks the wrong float and silently
                                // ruins the dequant sigma. Bridge through gnorms instead.
                                let existing = cache.k_per_head[0].count;
                                let v_offset = cache.sink_len;
                                let gpu_bpk = gpu.bytes_per_key;
                                let cpu_n_groups = if existing > 0 {
                                    cache.k_per_head[0].norms.len() / existing
                                } else { 1 };
                                let use_grouped_seed = cpu_n_groups > 1
                                    && cpu_n_groups == gpu.n_groups
                                    && gpu.gnorms.is_some();

                                for pos in 0..existing {
                                    let mut packed_all = Vec::with_capacity(self.n_kv_head * gpu_bpk);
                                    let mut norms_all: Vec<f32> = Vec::with_capacity(
                                        self.n_kv_head * if use_grouped_seed { cpu_n_groups } else { 1 });
                                    for h in 0..self.n_kv_head {
                                        let h_bpk = cache.k_per_head[h].packed_indices.len() / existing.max(1);
                                        let start = pos * h_bpk;
                                        let end = (start + h_bpk).min(cache.k_per_head[h].packed_indices.len());
                                        let chunk = &cache.k_per_head[h].packed_indices[start..end];
                                        packed_all.extend_from_slice(chunk);
                                        if chunk.len() < gpu_bpk {
                                            packed_all.extend(std::iter::repeat(0u8).take(gpu_bpk - chunk.len()));
                                        }
                                        if use_grouped_seed {
                                            let n_off = pos * cpu_n_groups;
                                            norms_all.extend_from_slice(
                                                &cache.k_per_head[h].norms[n_off..n_off + cpu_n_groups]);
                                        } else {
                                            norms_all.push(cache.k_per_head[h].norms[pos]);
                                        }
                                    }
                                    let v_data = if let Some(ref v_raw) = cache.v_raw {
                                        let v_pos = v_raw.narrow(2, v_offset + pos, 1)
                                            .and_then(|t| t.to_dtype(DType::F32))
                                            .and_then(|t| t.flatten_all())
                                            .and_then(|t| t.to_vec1());
                                        match v_pos {
                                            Ok(d) => d,
                                            Err(_) => vec![0.0f32; self.n_kv_head * self.head_dim],
                                        }
                                    } else if let Some(ref v_store) = cache.v_compressed {
                                        // value_bits > 0: decompress V for this position
                                        let v_all = super::kv_cache::decompress_values_store(
                                            v_store, self.n_kv_head, self.head_dim,
                                            cache.cached_len, q.device(),
                                        ).ok();
                                        if let Some(ref vt) = v_all {
                                            vt.narrow(2, v_offset + pos, 1)
                                                .and_then(|t| t.to_dtype(DType::F32))
                                                .and_then(|t| t.flatten_all())
                                                .and_then(|t| t.to_vec1())
                                                .unwrap_or_else(|_| vec![0.0f32; self.n_kv_head * self.head_dim])
                                        } else {
                                            vec![0.0f32; self.n_kv_head * self.head_dim]
                                        }
                                    } else {
                                        vec![0.0f32; self.n_kv_head * self.head_dim]
                                    };
                                    if use_grouped_seed {
                                        let _ = gpu.append_grouped(&packed_all, &norms_all, &v_data);
                                    } else {
                                        let _ = gpu.append(&packed_all, &norms_all, &v_data);
                                    }
                                }
                                self.gpu_tq_cache = Some(gpu);
                            }
                            Err(e) => eprintln!("[gpu-tq] init failed: {}", e),
                        }
                    }
                    // Append the newly CPU-compressed token(s) to GPU cache.
                    // Skip if GPU compress already appended directly.
                    // `else if`: skip when GPU cache was just initialized (seed already includes all tokens).
                    else if !gpu_compress_ok {
                        if let Some(ref mut gpu) = self.gpu_tq_cache {
                            let count = cache.k_per_head[0].count;
                            let gpu_bpk = gpu.bytes_per_key;
                            let pos = count - 1;
                            // Same grouped detection as the seed loop above.
                            let cpu_n_groups = if count > 0 {
                                cache.k_per_head[0].norms.len() / count
                            } else { 1 };
                            let use_grouped_bridge = cpu_n_groups > 1
                                && cpu_n_groups == gpu.n_groups
                                && gpu.gnorms.is_some();

                            let mut packed_all = Vec::with_capacity(self.n_kv_head * gpu_bpk);
                            let mut norms_all: Vec<f32> = Vec::with_capacity(
                                self.n_kv_head * if use_grouped_bridge { cpu_n_groups } else { 1 });
                            for h in 0..self.n_kv_head {
                                let h_bpk = cache.k_per_head[h].packed_indices.len() / count.max(1);
                                let start = pos * h_bpk;
                                let end = (start + h_bpk).min(cache.k_per_head[h].packed_indices.len());
                                let chunk = &cache.k_per_head[h].packed_indices[start..end];
                                packed_all.extend_from_slice(chunk);
                                if chunk.len() < gpu_bpk {
                                    packed_all.extend(std::iter::repeat(0u8).take(gpu_bpk - chunk.len()));
                                }
                                if use_grouped_bridge {
                                    let n_off = pos * cpu_n_groups;
                                    norms_all.extend_from_slice(
                                        &cache.k_per_head[h].norms[n_off..n_off + cpu_n_groups]);
                                } else {
                                    norms_all.push(cache.k_per_head[h].norms[pos]);
                                }
                            }
                            // Extract V data: prefer v_raw, fall back to live v tensor.
                            // When value_bits > 0, v_raw is None (V stored in v_compressed).
                            // The live v tensor has the current token's V in fp32.
                            let v_data = if let Some(ref v_raw) = cache.v_raw {
                                let vlen = v_raw.dim(2)?;
                                let last_v = v_raw.narrow(2, vlen - 1, 1)?;
                                last_v.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?
                            } else {
                                // v_raw is None (value_bits > 0). Use live v tensor directly.
                                v.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?
                            };
                            if use_grouped_bridge {
                                let _ = gpu.append_grouped(&packed_all, &norms_all, &v_data);
                            } else {
                                let _ = gpu.append(&packed_all, &norms_all, &v_data);
                            }
                        }
                    }

                    // Ensure signs are on GPU for all-GPU decompress attention path.
                    // Uploaded here (not in GPU compress block) because GPU compress
                    // may not run on the first compression (gpu_tq_cache was None).
                    if self.signs_gpu.is_none() {
                        if let Ok(buf) = reg.stream.clone_htod(&self.signs) {
                            self.signs_gpu = Some(buf);
                        }
                    }
                }
            }

            cache.cached_len += seq_len;
            cache.tokens_since_decay += seq_len;

            // --- TriAttention: accumulate pre-RoPE keys + periodic eviction ---
            if cache.tri_config.is_some() {
                // #2: Store pre-RoPE keys for scoring
                store_tri_keys(
                    cache, k_pre_rope.as_ref(),
                    self.n_kv_head, self.head_dim, seq_len, index_pos,
                );

                // #3: Periodic eviction — score and remove low-importance tokens
                maybe_evict_tri(
                    cache,
                    #[cfg(feature = "cuda")] &mut self.gpu_tq_cache,
                    #[cfg(feature = "cuda")] &mut self.gpu_cold_cache,
                    self.n_kv_head, self.head_dim, self.layer_idx,
                    #[cfg(feature = "cuda")] self.tq_config.bits,
                    #[cfg(feature = "cuda")] self.tq_config.group_size,
                    index_pos, seq_len,
                );
            }

            let total_len = cache.cached_len;

            // --- Temporal decay: demote old hot tokens to lower bit width ---
            apply_temporal_decay(
                cache, self.n_kv_head, self.padded_head_dim,
                layer_tq_config.rotation_seed,
            );

            // --- KV Compaction: reduce token count when hot cache exceeds threshold ---
            maybe_compact_cache(
                cache, &q, n_rep, seq_len, total_len, index_pos,
                self.n_kv_head, self.head_dim, self.padded_head_dim,
                &layer_tq_config, &self.cos, &self.sin,
                self.rope_style, self.rope_dim,
            )?;

            // --- Build full key tensor for attention ---
            // Concatenate: [sink | cold | compacted | hot | current]
            // This is the POQ + Sink + Decay + Compaction approach

            let y = if seq_len == 1 && total_len > 1 {
                // GENERATION: single token attention against full cache
                //
                // POQ (Past-Only Quantization): for the current token's key, we use
                // the FP16 original in attention instead of the compressed version.
                // The compressed version IS in the cache (for future tokens), but
                // we replace the last position with the lossless original.
                let q_f32 = q.to_dtype(DType::F32)?;

                // Fused attention incompatible with pre-RoPE and compaction
                let has_compacted = cache.compacted.is_some();
                let use_fused = get_use_fused() && q.is_cpu()
                    && !cache.pre_rope && !has_compacted;
                // Use GPU TQ cache count when available (GPU compress path
                // doesn't update CPU cache.k_per_head[h].count).
                let n_compressed_cpu = cache.k_per_head[0].count;
                #[cfg(feature = "cuda")]
                let n_compressed = self.gpu_tq_cache.as_ref()
                    .map(|g| g.count).unwrap_or(n_compressed_cpu);
                #[cfg(not(feature = "cuda"))]
                let n_compressed = n_compressed_cpu;
                let n_past_compressed = if n_compressed > 0 { n_compressed - 1 } else { 0 };
                let compacted_t = cache.compacted.as_ref()
                    .map(|c| c[0].t).unwrap_or(0);

                // --- GPU Fused TQ Attention (compressed KV, no decompression) ---
                // GPU fused TQ attention: works even if q is CPU (we upload it).
                // Requirements: no pre-rope, no compacted, no cold, no sink, has past keys.
                // GPU fused TQ attention: compressed keys + optional sink tokens.
                // Requirements: no pre-rope, no compacted, no cold tier, has past keys.
                // GPU full fused TQ attention — enabled with TQ_GPU_FUSED=1
                #[cfg(feature = "cuda")]
                let gpu_fused_ok = std::env::var("TQ_GPU_FUSED").ok().map(|v| v == "1").unwrap_or(false)
                    && crate::cuda::kernels::global_registry().is_some()
                    && !cache.pre_rope && !has_compacted
                    && cache.cold_len == 0
                    && n_past_compressed > 0 && seq_len == 1;
                #[cfg(not(feature = "cuda"))]
                let gpu_fused_ok = false;

                // GPU fused attention with CPU V multiply (no online softmax in kernel)
                #[cfg(feature = "cuda")]
                if gpu_fused_ok {
                    if let (Some(reg), Some(ref gpu)) = (crate::cuda::kernels::global_registry(), &self.gpu_tq_cache) {
                        // Pre-rotate query for compressed scores
                        let q_flat = q_f32.flatten_all()?.to_vec1()?;
                        let mut rotated_q = Vec::with_capacity(self.n_head * self.head_dim);
                        for qh in 0..self.n_head {
                            let qstart = qh * self.head_dim;
                            let q_vec = &q_flat[qstart..qstart + self.head_dim];
                            let rq = tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs);
                            rotated_q.extend_from_slice(&rq);
                        }

                        // KIVI per-channel: pre-scale rotated query by sigma ratios
                        // so fused attention centroid lookup accounts for per-dim variance.
                        // score = sum(q[d]*ratio[d] * centroids[idx[d]]) * norm/sqrt(dim) * scale
                        if let Some(ref pcs) = self.tq_config.rotated_channel_sigma {
                            let mean_sigma: f32 = pcs.iter().sum::<f32>() / self.head_dim as f32;
                            for qh in 0..self.n_head {
                                let base = qh * self.head_dim;
                                for d in 0..self.head_dim {
                                    rotated_q[base + d] *= pcs[d] / mean_sigma;
                                }
                            }
                        }

                        // Upload rotated query to temp buffer
                        let rq_gpu = reg.stream.clone_htod(&rotated_q)
                            .map_err(|e| TqError::Msg(format!("rq upload: {}", e)))?;

                        // GPU: compute compressed scores [n_heads, n_past_compressed]
                        let mut scores_gpu = reg.stream.alloc_zeros::<f32>(self.n_head * n_past_compressed)
                            .map_err(|e| TqError::Msg(format!("scores alloc: {}", e)))?;
                        let scale = (1.0 / self.attn_scale_denom) as f32;
                        // Sprint 1C: route to grouped attention launcher when
                        // gnorms is populated. Both per-vector and grouped
                        // kernels write the same scores layout, so the
                        // downstream softmax/V path stays unchanged.
                        // pre_rope on => CPU grouped bridge filled gnorms; per-vector
                        // gpu.norms is stale and unsafe to read.
                        // Sprint 1D: flipped grouped to default-enabled.
                        let env_grouped = super::kv_cache::get_gpu_grouped_enabled();
                        let use_grouped_attn = (cache.pre_rope || env_grouped)
                            && gpu.gnorms.is_some()
                            && gpu.n_groups > 1;
                        if use_grouped_attn {
                            let gnorms_ref = gpu.gnorms.as_ref().unwrap();
                            crate::cuda::kernels::tq_fused_attention_grouped(
                                reg, &rq_gpu, &gpu.packed_indices, gnorms_ref,
                                &gpu.centroids, &mut scores_gpu,
                                self.n_head, self.n_kv_head, n_past_compressed, self.head_dim,
                                gpu.group_size, gpu.bits as usize, scale,
                            ).map_err(|e| TqError::Msg(format!("tq_fused_attention_grouped: {}", e)))?;
                        } else {
                            crate::cuda::kernels::tq_fused_attention(
                                reg, &rq_gpu, &gpu.packed_indices, &gpu.norms,
                                &gpu.centroids, &mut scores_gpu,
                                self.n_head, self.n_kv_head, n_past_compressed, self.head_dim,
                                gpu.bits as usize, scale, gpu.max_seq,
                            ).map_err(|e| TqError::Msg(format!("tq_fused_attention: {}", e)))?;
                        }

                        // Download compressed scores
                        let comp_scores: Vec<f32> = reg.stream.clone_dtoh(&scores_gpu)
                            .map_err(|e| TqError::Msg(format!("scores download: {}", e)))?;

                        // Download V data for compressed keys
                        let v_gpu_data: Vec<f32> = reg.stream.clone_dtoh(&gpu.v_data)
                            .map_err(|e| TqError::Msg(format!("v download: {}", e)))?;

                        // CPU: build full scores and compute attention (using GPU scores for compressed keys)
                        let sink_k_flat: Vec<f32> = cache.sink_k.as_ref()
                            .map(|sk| sk.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap())
                            .unwrap_or_default();
                        let sink_v_flat: Vec<f32> = cache.v_raw.as_ref()
                            .and_then(|vr| vr.narrow(2, 0, cache.sink_len).ok())
                            .map(|sv| sv.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap())
                            .unwrap_or_default();
                        let k_flat: Vec<f32> = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                        let v_flat_cur: Vec<f32> = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

                        // Use v_raw directly for V data (avoids GPU cache V issues)
                        let v_raw_flat: Vec<f32> = cache.v_raw.as_ref()
                            .map(|vr| vr.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap())
                            .unwrap_or_default();
                        let v_raw_len = cache.v_raw.as_ref().map(|vr| vr.dim(2).unwrap_or(0)).unwrap_or(0);

                        let mut attn_out = Vec::with_capacity(self.n_head * self.head_dim);
                        for qh in 0..self.n_head {
                            let kv_h = qh / n_rep;
                            let q_vec = &q_flat[qh * self.head_dim..(qh + 1) * self.head_dim];
                            let mut scores = Vec::with_capacity(total_len);

                            // Sink scores
                            for s in 0..cache.sink_len {
                                let k_off = kv_h * cache.sink_len * self.head_dim + s * self.head_dim;
                                let k_vec = &sink_k_flat[k_off..k_off + self.head_dim];
                                let dot: f32 = q_vec.iter().zip(k_vec).map(|(a,b)| a * b).sum();
                                scores.push(dot * scale);
                            }
                            // Compressed scores from GPU
                            for i in 0..n_past_compressed {
                                scores.push(comp_scores[qh * n_past_compressed + i]);
                            }
                            // Current token (POQ lossless)
                            if n_compressed > 0 {
                                let k_vec = &k_flat[kv_h * self.head_dim..(kv_h + 1) * self.head_dim];
                                let dot: f32 = q_vec.iter().zip(k_vec).map(|(a,b)| a * b).sum();
                                scores.push(dot * scale);
                            }

                            // Softmax + V weighted sum using v_raw directly
                            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let exp_s: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                            let sum_exp: f32 = exp_s.iter().sum();

                            let mut out = vec![0.0f32; self.head_dim];
                            for (si, &w_exp) in exp_s.iter().enumerate() {
                                let w = w_exp / sum_exp;
                                // V position in v_raw: 0..total_len
                                let v_pos = si; // direct position mapping
                                let v_off = kv_h * v_raw_len * self.head_dim + v_pos * self.head_dim;
                                if v_off + self.head_dim <= v_raw_flat.len() {
                                    for d in 0..self.head_dim { out[d] += w * v_raw_flat[v_off + d]; }
                                }
                            }
                            attn_out.extend_from_slice(&out);
                        }
                        let y = Tensor::from_vec(attn_out, vec![1, self.n_head, 1, self.head_dim], q.device())?;
                        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
                        return self.attention_wo.forward(&y, backend);
                    }
                }

                // ─── ALL-GPU TQ DECOMPRESS ATTENTION ──────────────────
                // GPU decompress keys + GPU matmul + GPU softmax + GPU V matmul.
                // No CPU round-trip. Uses proven two-step decompress (centroid lookup + hadamard inverse).
                // Pre-RoPE supported via rope_halved/interleaved_strided applied
                // in-place to freshly decompressed keys (since 2026-04-11).
                // Falls through to CPU paths when conditions aren't met.
                #[cfg(feature = "cuda")]
                if seq_len == 1
                    && !has_compacted
                    && cache.cold_len == 0
                    && n_compressed > 0
                    && self.padded_head_dim == self.head_dim
                    && cache.k_per_head[0].qjl_corrections.is_none()
                    && layer_tq_config.channel_scales.is_none()
                    && layer_tq_config.key_channel_bias.is_none()
                    && self.cos.is_cuda() && self.sin.is_cuda()
                {
                    if let (Some(reg), Some(ref mut gpu), Some(ref signs)) = (
                        crate::cuda::kernels::global_registry(),
                        &mut self.gpu_tq_cache,
                        &self.signs_gpu,
                    ) {
                        let gpu_result: Result<Tensor> = (|| {
                            let hdim = self.head_dim;
                            let stream = reg.stream.clone();

                            // Step 1: Incremental GPU decompress + K hot gather
                            let ms = gpu.max_seq;
                            let k_hot = if n_past_compressed > 0 {
                                if gpu.decomp_count < n_past_compressed {
                                    let new_start = gpu.decomp_count;
                                    let new_count = n_past_compressed - new_start;
                                    // Use grouped decompress when:
                                    //   (a) cache.pre_rope is on — CPU grouped compress was
                                    //       used and the bridge filled gpu.gnorms (gpu.norms
                                    //       is stale in this case), OR
                                    //   (b) Sprint 1D default — grouped on by default,
                                    //       TQ_GPU_GROUPED=0 to disable.
                                    let env_grouped = super::kv_cache::get_gpu_grouped_enabled();
                                    let use_grouped_decomp = (cache.pre_rope || env_grouped)
                                        && gpu.gnorms.is_some()
                                        && gpu.n_groups > 1;
                                    if use_grouped_decomp {
                                        let gnorms_ref = gpu.gnorms.as_ref().unwrap();
                                        crate::cuda::kernels::tq_decompress_keys_grouped(
                                            reg, &gpu.packed_indices, gnorms_ref,
                                            &gpu.centroids, signs, &mut gpu.decomp_cache,
                                            self.n_kv_head, new_count, hdim,
                                            gpu.group_size, gpu.bits as usize, ms,
                                            new_start, ms,
                                        ).map_err(|e| TqError::Msg(format!("decompress_grouped: {e}")))?;
                                    } else {
                                        crate::cuda::kernels::tq_decompress_keys_range(
                                            reg, &gpu.packed_indices, &gpu.norms,
                                            &gpu.centroids, signs, &mut gpu.decomp_cache,
                                            self.n_kv_head, new_count, hdim,
                                            gpu.bits as usize, ms,
                                            new_start, ms,
                                        ).map_err(|e| TqError::Msg(format!("decompress_range: {e}")))?;
                                    }

                                    // Pre-RoPE TQ: keys are stored in the cache pre-rotation,
                                    // so newly decompressed rows must have RoPE applied in-place
                                    // before they enter the gather/matmul. Subsequent decode
                                    // steps see them as post-RoPE. Positions are sink_len-relative
                                    // because the compressed cache only holds non-sink tokens.
                                    if cache.pre_rope {
                                        let pos_offset = cache.sink_len + new_start;
                                        let rope_strided = match self.rope_style {
                                            RopeStyle::Halved => crate::cuda::kernels::rope_halved_strided,
                                            RopeStyle::Interleaved => crate::cuda::kernels::rope_interleaved_strided,
                                        };
                                        rope_strided(
                                            reg, &mut gpu.decomp_cache,
                                            self.cos.cuda_data(), self.sin.cuda_data(),
                                            self.n_kv_head, ms, hdim, self.rope_dim,
                                            new_start, new_count, pos_offset,
                                        ).map_err(|e| TqError::Msg(format!("pre_rope strided: {e}")))?;
                                    }

                                    gpu.decomp_count = n_past_compressed;
                                }
                                // K hot: gather from strided decomp_cache → pre-allocated k_contig
                                {
                                    let n_kv = self.n_kv_head;
                                    let count = n_past_compressed;
                                    let dst = std::sync::Arc::get_mut(&mut gpu.k_contig)
                                        .expect("k_contig aliased");
                                    crate::cuda::kernels::strided_copy_args(
                                        reg, &gpu.decomp_cache, dst,
                                        n_kv * count * hdim, 3,
                                        &[(n_kv as i32), (count as i32), (hdim as i32)],
                                        &[((count * hdim) as i32), (hdim as i32), 1],
                                        &[((ms * hdim) as i32), (hdim as i32), 1],
                                        0,
                                    ).map_err(|e| TqError::Msg(format!("k_hot gather: {e}")))?;
                                }
                                Some(Tensor::from_cuda_arc(
                                    std::sync::Arc::clone(&gpu.k_contig),
                                    vec![1, self.n_kv_head, n_past_compressed, hdim],
                                    stream.clone(),
                                ))
                            } else { None };

                            // Cache sink K/V on GPU (upload once)
                            let sink_len = cache.sink_len;
                            if sink_len > 0 && self.sink_k_gpu.is_none() {
                                if let Some(ref sink_k) = cache.sink_k {
                                    let data: Vec<f32> = sink_k.to_dtype(DType::F32)?.to_vec1()?;
                                    if let Ok(buf) = stream.clone_htod(&data) {
                                        self.sink_k_gpu = Some(buf);
                                    }
                                }
                                if let Some(ref v_raw) = cache.v_raw {
                                    let data: Vec<f32> = v_raw.narrow(2, 0, sink_len)?.to_dtype(DType::F32)?.to_vec1()?;
                                    if let Ok(buf) = stream.clone_htod(&data) {
                                        self.sink_v_gpu = Some(buf);
                                    }
                                }
                            }

                            // Step 2: Build K on GPU [sink | hot | current]
                            let mut k_parts_gpu: Vec<Tensor> = Vec::new();
                            if let Some(ref sk) = self.sink_k_gpu {
                                k_parts_gpu.push(Tensor::from_cuda_arc(
                                    std::sync::Arc::new(sk.clone()),
                                    vec![1, self.n_kv_head, sink_len, hdim],
                                    stream.clone(),
                                ));
                            }
                            if let Some(kh) = k_hot { k_parts_gpu.push(kh); }

                            // Sprint F: retrieve from GPU cold tier — ZERO CPU involvement.
                            // Cold tier stores post-RoPE decompressed K (from decomp_cache
                            // at eviction time). Gather contiguously and cat into K_full.
                            // V2: cap at TQ_MAX_RETRIEVE (default 16) most-recent cold tokens.
                            // Keeps attention cost bounded regardless of cold tier size.
                            let mut n_retrieved = 0usize;
                            #[cfg(feature = "cuda")]
                            if super::kv_cache::get_offload_enabled() {
                                if let Some(ref cold) = self.gpu_cold_cache {
                                    if cold.count > 0 {
                                        let max_retrieve = {
                                            static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                                            *C.get_or_init(|| std::env::var("TQ_MAX_RETRIEVE")
                                                .ok().and_then(|v| v.parse().ok()).unwrap_or(16))
                                        };
                                        // Take the most recent cold tokens (highest seq_pos = most informative)
                                        let n_cold = cold.count.min(max_retrieve);
                                        let cold_offset = cold.count.saturating_sub(n_cold);
                                        let mc = cold.max_cold;

                                        // Gather contiguous K from strided cold.k_data
                                        // (post-RoPE, no decompress or RoPE needed)
                                        // cold_offset skips oldest tokens when cold.count > max_retrieve
                                        let mut cold_k_buf = reg.stream.alloc_zeros::<f32>(
                                            self.n_kv_head * n_cold * hdim
                                        ).map_err(|e| TqError::Msg(format!("cold k buf: {}", e)))?;
                                        for h in 0..self.n_kv_head {
                                            let src_off = (h * mc + cold_offset) * hdim;
                                            let dst_off = h * n_cold * hdim;
                                            crate::cuda::kernels::copy_with_offsets(
                                                reg, &cold.k_data, &cold_k_buf,
                                                n_cold * hdim, src_off, dst_off,
                                            ).map_err(|e| TqError::Msg(format!("cold k gather: {e}")))?;
                                        }

                                        k_parts_gpu.push(Tensor::from_cuda(
                                            cold_k_buf, vec![1, self.n_kv_head, n_cold, hdim], stream.clone(),
                                        ));
                                        n_retrieved = n_cold;
                                    }
                                }
                            }

                            if n_compressed > 0 {
                                let k_cur = k.to_dtype(DType::F32)?;
                                debug_assert!(k_cur.is_cuda(), "current K should be GPU");
                                k_parts_gpu.push(k_cur);
                            }

                            let k_full = if k_parts_gpu.len() == 1 {
                                k_parts_gpu.remove(0)
                            } else {
                                let k_refs: Vec<&Tensor> = k_parts_gpu.iter().collect();
                                Tensor::cat(&k_refs, 2)?
                            };
                            let k_full = repeat_kv(k_full, n_rep)?;

                            // Step 3: Q @ K^T on GPU (q_f32 already CUDA from attention_wo)
                            // BUG fix 2026-04-11: was `/ scale` which is `* sqrt(d)` (off by d=128).
                            // The correct scaling for attention logits is `(q @ k.T) / sqrt(d)`,
                            // i.e. multiply by `scale = 1/sqrt(d)`. Has been wrong since 3888d39 —
                            // PPL never hit this path (seq_len > 1) so the bug went unnoticed.
                            debug_assert!(q_f32.is_cuda(), "Q should be GPU in all-GPU path");
                            let mut att = (q_f32.matmul(&k_full.t()?)? / (hdim as f64).sqrt())?;
                            if let Some(cap) = self.attn_logit_softcap {
                                att = apply_softcap(&att, cap)?;
                            }
                            let att = softmax_last_dim(&att, backend)?;

                            // Step 4: V hot from strided gpu.v_data (single strided_copy kernel)
                            let mut v_parts_gpu: Vec<Tensor> = Vec::new();
                            if let Some(ref sv) = self.sink_v_gpu {
                                v_parts_gpu.push(Tensor::from_cuda_arc(
                                    std::sync::Arc::new(sv.clone()),
                                    vec![1, self.n_kv_head, sink_len, hdim],
                                    stream.clone(),
                                ));
                            }
                            if n_past_compressed > 0 {
                                // V hot: gather from strided v_data → pre-allocated v_contig (0 allocs)
                                {
                                    let n_kv = self.n_kv_head;
                                    let count = n_past_compressed;
                                    let dst = std::sync::Arc::get_mut(&mut gpu.v_contig)
                                        .expect("v_contig aliased");
                                    crate::cuda::kernels::strided_copy_args(
                                        reg, &gpu.v_data, dst,
                                        n_kv * count * hdim, 3,
                                        &[(n_kv as i32), (count as i32), (hdim as i32)],
                                        &[((count * hdim) as i32), (hdim as i32), 1],
                                        &[((ms * hdim) as i32), (hdim as i32), 1],
                                        0,
                                    ).map_err(|e| TqError::Msg(format!("v_hot gather: {e}")))?;
                                }
                                let v_hot = Tensor::from_cuda_arc(
                                    std::sync::Arc::clone(&gpu.v_contig),
                                    vec![1, self.n_kv_head, n_past_compressed, hdim],
                                    stream.clone(),
                                );
                                v_parts_gpu.push(v_hot);
                            }
                            // Sprint F: insert cold V (GPU cold tier, strided → contiguous gather)
                            #[cfg(feature = "cuda")]
                            if n_retrieved > 0 {
                                if let Some(ref cold) = self.gpu_cold_cache {
                                    let n_cold = n_retrieved;
                                    let mc = cold.max_cold;
                                    let cold_offset = cold.count.saturating_sub(n_cold);
                                    let mut cold_v_buf = reg.stream.alloc_zeros::<f32>(
                                        self.n_kv_head * n_cold * hdim
                                    ).map_err(|e| TqError::Msg(format!("cold v buf: {}", e)))?;
                                    for h in 0..self.n_kv_head {
                                        let src_off = (h * mc + cold_offset) * hdim;
                                        let dst_off = h * n_cold * hdim;
                                        crate::cuda::kernels::copy_with_offsets(
                                            reg, &cold.v_data, &cold_v_buf,
                                            n_cold * hdim, src_off, dst_off,
                                        ).map_err(|e| TqError::Msg(format!("cold v gather: {e}")))?;
                                    }
                                    v_parts_gpu.push(Tensor::from_cuda(
                                        cold_v_buf, vec![1, self.n_kv_head, n_cold, hdim], stream.clone(),
                                    ));
                                }
                            }
                            if n_compressed > 0 {
                                let v_cur = v.to_dtype(DType::F32)?;
                                debug_assert!(v_cur.is_cuda(), "current V should be GPU");
                                v_parts_gpu.push(v_cur);
                            }

                            let v_full = if v_parts_gpu.len() == 1 {
                                v_parts_gpu.remove(0)
                            } else {
                                let v_refs: Vec<&Tensor> = v_parts_gpu.iter().collect();
                                Tensor::cat(&v_refs, 2)?
                            };
                            let v_full = repeat_kv(v_full, n_rep)?;
                            att.matmul(&v_full.contiguous()?)?.to_dtype(cache.dtype)
                        })();

                        match gpu_result {
                            Ok(y) => {
                                let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
                                return self.attention_wo.forward(&y, backend);
                            }
                            Err(ref e) if n_compressed > n_compressed_cpu => {
                                // GPU compress active but all-GPU attention failed.
                                // CPU fallback has incomplete data — propagate error.
                                eprintln!("[gpu-tq-attn] FAILED (GPU compress active): {e}");
                                return Err(TqError::Msg(format!("all-GPU TQ attn: {e}")));
                            }
                            _ => {} // Fall through to CPU paths
                        }
                    }
                }

                // --- Compute attention scores ---
                let att = if use_fused {
                    // FUSED PATH: compute scores directly from compressed indices.
                    // No key decompression — saves memory bandwidth.
                    // Scores are computed per query-head using pre-rotated query
                    // and centroid table lookup (AVX2 SIMD when available).
                    let q_flat = q_f32.flatten_all()?.to_vec1()?;
                    let scale = (1.0 / self.attn_scale_denom) as f32;
                    // Use calibrated centroids if available, else standard Gaussian
                    let cal_centroids_owned: Option<Vec<f32>> = layer_tq_config.calibrated_codebook
                        .as_ref().map(|cb| cb.centroids.clone());
                    // Note: cold centroids looked up per-head inside the loop (mixed bits)

                    // GPU-accelerated compressed score computation (if available)
                    // GPU-accelerated score computation disabled — overhead too high for benefit
                    // TODO: pre-allocate GPU buffers and batch across layers
                    #[cfg(feature = "cuda")]
                    let gpu_comp_scores: Option<Vec<f32>> = if false && n_past_compressed >= 8 {
                        if let (Some(reg), Some(ref gpu)) = (crate::cuda::kernels::global_registry(), &self.gpu_tq_cache) {
                            // Upload rotated query
                            let mut rotated_q_all = Vec::with_capacity(self.n_head * self.head_dim);
                            for qh in 0..self.n_head {
                                let qstart = qh * self.head_dim;
                                let q_vec = &q_flat[qstart..qstart + self.head_dim];
                                let rq = if let Some(ref matrix) = self.tq_config.rotation_matrix {
                                    tq_kv::pre_rotate_query_with_matrix(q_vec, matrix)
                                } else {
                                    tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs)
                                };
                                rotated_q_all.extend_from_slice(&rq);
                            }
                            let rq_gpu = reg.stream.clone_htod(&rotated_q_all).ok();
                            if let Some(ref rq) = rq_gpu {
                                let mut scores_gpu = reg.stream.alloc_zeros::<f32>(self.n_head * n_past_compressed).ok();
                                if let Some(ref mut sg) = scores_gpu {
                                    let env_grouped = super::kv_cache::get_gpu_grouped_enabled();
                                    let use_grouped_attn = (cache.pre_rope || env_grouped)
                                        && gpu.gnorms.is_some()
                                        && gpu.n_groups > 1;
                                    let ok = if use_grouped_attn {
                                        let gnorms_ref = gpu.gnorms.as_ref().unwrap();
                                        crate::cuda::kernels::tq_fused_attention_grouped(
                                            reg, rq, &gpu.packed_indices, gnorms_ref,
                                            &gpu.centroids, sg,
                                            self.n_head, self.n_kv_head, n_past_compressed, self.head_dim,
                                            gpu.group_size, gpu.bits as usize, scale,
                                        )
                                    } else {
                                        crate::cuda::kernels::tq_fused_attention(
                                            reg, rq, &gpu.packed_indices, &gpu.norms,
                                            &gpu.centroids, sg,
                                            self.n_head, self.n_kv_head, n_past_compressed, self.head_dim,
                                            gpu.bits as usize, scale, gpu.max_seq,
                                        )
                                    };
                                    if ok.is_ok() {
                                        reg.stream.clone_dtoh(sg).ok()
                                    } else { None }
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None };
                    #[cfg(not(feature = "cuda"))]
                    let gpu_comp_scores: Option<Vec<f32>> = None;

                    use rayon::prelude::*;
                    let head_scores: Vec<Vec<f32>> = (0..self.n_head)
                        .into_par_iter()
                        .map(|qh| {
                            let kv_h = qh / n_rep;
                            let q_vec = &q_flat[qh * self.head_dim..(qh + 1) * self.head_dim];
                            let mut rotated_q = if let Some(ref matrix) = self.tq_config.rotation_matrix {
                                tq_kv::pre_rotate_query_with_matrix(q_vec, matrix)
                            } else {
                                tq_kv::pre_rotate_query_with_signs(q_vec, &self.signs)
                            };
                            // KIVI per-channel: pre-scale rotated query
                            if let Some(ref pcs) = self.tq_config.rotated_channel_sigma {
                                let mean_sigma: f32 = pcs.iter().sum::<f32>() / pcs.len() as f32;
                                for (d, v) in rotated_q.iter_mut().enumerate() {
                                    *v *= pcs[d] / mean_sigma;
                                }
                            }
                            let mut scores = Vec::with_capacity(total_len);

                            // Segment 1: Sink keys (standard dot product, not compressed)
                            if let Some(ref sink) = cache.sink_k {
                                let sink_f32 = sink.to_dtype(DType::F32).expect("sink to_dtype");
                                let sink_flat = sink_f32.flatten_all().expect("sink flatten").to_vec1().expect("sink to_vec1");
                                let sink_count = cache.sink_len;
                                for s in 0..sink_count {
                                    let offset = (kv_h * sink_count + s) * self.head_dim;
                                    let k_vec = &sink_flat[offset..offset + self.head_dim];
                                    let dot: f32 = q_vec.iter().zip(k_vec.iter())
                                        .map(|(&qi, &ki)| qi * ki).sum();
                                    scores.push(dot * scale);
                                }
                            }

                            // Segment 2: Cold (decayed) keys — fused at cold bit width (per-head)
                            if let Some(ref cold) = cache.k_cold {
                                let cold_cb = tq_kv::codebook::get_centroids(cold[kv_h].bits);
                                let cold_scores = tq_kv::fused_attention_scores(
                                    &rotated_q, &cold[kv_h], cold_cb, scale,
                                );
                                scores.extend_from_slice(&cold_scores);
                            }

                            // Segment 3: Hot compressed keys — GPU-accelerated when available
                            if n_past_compressed > 0 {
                                if let Some(ref gpu_scores) = gpu_comp_scores {
                                    // Use pre-computed GPU scores
                                    let start = qh * n_past_compressed;
                                    scores.extend_from_slice(&gpu_scores[start..start + n_past_compressed]);
                                } else {
                                    // CPU fallback
                                    let head_bits = cache.k_per_head[kv_h].bits;
                                    let hot_cb: &[f32] = match cal_centroids_owned.as_deref() {
                                        Some(cal) if head_bits == layer_tq_config.bits => cal,
                                        _ => tq_kv::codebook::get_centroids(head_bits),
                                    };
                                    let hot = &cache.k_per_head[kv_h];
                                    let dim = hot.dim;
                                    let bpv = hot.bytes_per_vector();
                                    let mut idx_buf = vec![0u8; dim];
                                    for pos in 0..n_past_compressed {
                                        let norm = hot.norms[pos];
                                        if norm < 1e-10 {
                                            scores.push(0.0);
                                            continue;
                                        }
                                        let start = pos * bpv;
                                        let end = start + bpv;
                                        tq_kv::codebook::unpack_indices_into(
                                            &hot.packed_indices[start..end], &mut idx_buf, hot.bits,
                                        );
                                        let score = tq_kv::fused_dot_product_with_centroids(
                                            &rotated_q, &idx_buf, norm, hot_cb, dim,
                                        ) * scale;
                                        scores.push(score);
                                    }
                                }
                            }

                            // Segment 4: Current token key (FP16 original — POQ)
                            // Only if current token was compressed (not still in sink range)
                            if n_compressed > 0 {
                                let k_f32 = k.to_dtype(DType::F32).expect("k to_dtype");
                                let k_flat = k_f32.flatten_all().expect("k flatten").to_vec1().expect("k to_vec1");
                                let k_vec = &k_flat[kv_h * self.head_dim..(kv_h + 1) * self.head_dim];
                                let dot: f32 = q_vec.iter().zip(k_vec.iter())
                                    .map(|(&qi, &ki)| qi * ki).sum();
                                scores.push(dot * scale);
                            }

                            scores
                        })
                        .collect();

                    let mut all_scores = Vec::with_capacity(self.n_head * total_len);
                    for s in &head_scores {
                        all_scores.extend_from_slice(s);
                    }
                    let att = Tensor::from_vec(
                        all_scores, vec![1, self.n_head, 1, total_len], q.device(),
                    )?;
                    softmax_last_dim(&att, backend)?
                } else {
                    // DECOMPRESS PATH: decompress all keys, standard matmul
                    let mut k_parts: Vec<Tensor> = Vec::new();

                    // Part 1: Sink keys (FP16, lossless — always post-RoPE)
                    if let Some(ref sink) = cache.sink_k {
                        k_parts.push(sink.to_dtype(DType::F32)?);
                    }

                    // Part 1.5: Cold (decayed) keys
                    if let Some(ref cold) = cache.k_cold {
                        if cold[0].count > 0 {
                            let k_cold = if cache.pre_rope {
                                // Pre-RoPE: decompress + apply RoPE dynamically
                                let cold_start = cache.sink_len;
                                decompress_and_apply_rope(
                                    cold, self.n_kv_head,
                                    self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                                    &self.cos, &self.sin, cold_start, self.rope_style, self.rope_dim,
                                )?
                            } else {
                                decompress_compressed_keys(
                                    cold, self.n_kv_head,
                                    self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                                )?
                            };
                            let k_cold = if self.padded_head_dim > self.head_dim {
                                k_cold.narrow(3, 0, self.head_dim)?
                            } else {
                                k_cold
                            };
                            k_parts.push(k_cold);
                        }
                    }

                    // Part 2: Compacted keys (attention-matching reduced tokens + beta bias)
                    if let Some(ref comp) = cache.compacted {
                        // Build compacted keys tensor: [1, n_kv_head, compacted_t, head_dim]
                        let ct = comp[0].t;
                        let mut comp_data = Vec::with_capacity(self.n_kv_head * ct * self.head_dim);
                        for h in 0..self.n_kv_head {
                            comp_data.extend_from_slice(&comp[h].keys);
                        }
                        let k_comp = Tensor::from_vec(
                            comp_data, vec![1, self.n_kv_head, ct, self.head_dim], q.device(),
                        )?;
                        k_parts.push(k_comp);
                    }

                    // Part 3: Hot compressed keys (excluding last = current token)
                    if n_past_compressed > 0 {
                        let k_decomp = if cache.pre_rope {
                            // Pre-RoPE: decompress + apply RoPE dynamically
                            let hot_start = cache.sink_len + cache.cold_len + cache.compacted_original_len;
                            decompress_and_apply_rope(
                                &cache.k_per_head, self.n_kv_head,
                                self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                                &self.cos, &self.sin, hot_start, self.rope_style, self.rope_dim,
                            )?
                        } else {
                            decompress_compressed_keys(
                                &cache.k_per_head, self.n_kv_head,
                                self.padded_head_dim, DType::F32, q.device(), &layer_tq_config,
                            )?
                        };
                        let k_decomp = if self.padded_head_dim > self.head_dim {
                            k_decomp.narrow(3, 0, self.head_dim)?
                        } else {
                            k_decomp
                        };
                        let k_past = k_decomp.narrow(2, 0, n_past_compressed)?;
                        k_parts.push(k_past);
                    }

                    // Part 4: Current token key (FP16 original — POQ lossless, always post-RoPE)
                    // Only add if current token was compressed (not still in sink range)
                    if n_compressed > 0 {
                        k_parts.push(k.to_dtype(DType::F32)?);
                    }

                    let k_full = if k_parts.len() == 1 {
                        k_parts.remove(0)
                    } else {
                        let k_refs: Vec<&Tensor> = k_parts.iter().collect();
                        Tensor::cat(&k_refs, 2)?
                    };

                    // Effective attention length: sink + cold + compacted_t + hot + current
                    let attn_len = cache.sink_len + cache.cold_len + compacted_t
                        + n_past_compressed + if n_compressed > 0 { 1 } else { 0 };

                    let k_full = repeat_kv(k_full, n_rep)?;
                    let mut att = (q_f32.matmul(&k_full.t()?)? / self.attn_scale_denom)?;
                    if let Some(cap) = self.attn_logit_softcap {
                        att = apply_softcap(&att, cap)?;
                    }

                    // Apply compaction beta biases to compacted segment logits
                    if let Some(ref comp) = cache.compacted {
                        let _ct = comp[0].t;
                        let beta_start = cache.sink_len + cache.cold_len;
                        // Build bias vector: [sink=0, cold=0, compacted=beta, hot=0, current=0]
                        let mut full_bias = vec![0.0f32; attn_len];
                        // For simplicity, use first head's beta (GQA: same across heads)
                        for (i, &b) in comp[0].beta.iter().enumerate() {
                            full_bias[beta_start + i] = b;
                        }
                        let bias_tensor = Tensor::from_vec(
                            full_bias, vec![1, 1, 1, attn_len], q.device(),
                        )?;
                        att = att.broadcast_add(&bias_tensor)?;
                    }

                    // Softmax bias correction: compensate quantization-induced attention drift
                    if get_bias_correction() && n_past_compressed > 0 {
                        let bias = tq_kv::softmax_bias_correction(
                            &cache.k_per_head[0], self.head_dim,
                        );
                        // Build full bias vector: [..., hot_bias, ...]
                        let mut full_bias = vec![0.0f32; attn_len];
                        let hot_start = cache.sink_len + cache.cold_len + compacted_t;
                        for (i, &b) in bias.iter().take(n_past_compressed).enumerate() {
                            full_bias[hot_start + i] = b;
                        }
                        let bias_tensor = Tensor::from_vec(
                            full_bias, vec![1, 1, 1, attn_len], q.device(),
                        )?;
                        att = att.broadcast_add(&bias_tensor)?;
                    }

                    softmax_last_dim(&att, backend)?
                };

                // --- Compute attention output: att @ V ---
                let sparse_thresh = get_sparse_v_threshold();

                // When compaction is active, build value tensor with compacted values spliced in
                if has_compacted {
                    // Build V tensor: [sink_v | cold_v | compacted_v | hot_v | current_v]
                    let mut v_parts: Vec<Tensor> = Vec::new();
                    let device = q.device();
                    let head_dim = self.head_dim;

                    // Get original value tensor (all positions including removed ones)
                    let v_all_f32 = if cache.value_bits == 0 {
                        cache.v_raw.as_ref().unwrap().to_dtype(DType::F32)?
                    } else {
                        decompress_values_store(
                            cache.v_compressed.as_ref().unwrap(),
                            self.n_kv_head, head_dim, total_len, device,
                        )?
                    };

                    // Sink values: positions [0, sink_len)
                    if cache.sink_len > 0 {
                        v_parts.push(v_all_f32.narrow(2, 0, cache.sink_len)?);
                    }

                    // Cold values: positions [sink_len, sink_len + cold_len)
                    if cache.cold_len > 0 {
                        v_parts.push(v_all_f32.narrow(2, cache.sink_len, cache.cold_len)?);
                    }

                    // Compacted synthetic values
                    let comp = cache.compacted.as_ref().unwrap();
                    let ct = comp[0].t;
                    let mut comp_v_data = Vec::with_capacity(self.n_kv_head * ct * head_dim);
                    for h in 0..self.n_kv_head {
                        comp_v_data.extend_from_slice(&comp[h].values);
                    }
                    let v_comp = Tensor::from_vec(
                        comp_v_data, vec![1, self.n_kv_head, ct, head_dim], device,
                    )?;
                    v_parts.push(v_comp);

                    // Hot values: skip compacted_original_len, take remaining
                    let hot_v_start = cache.sink_len + cache.cold_len + cache.compacted_original_len;
                    let hot_v_count = n_past_compressed + if n_compressed > 0 { 1 } else { 0 };
                    if hot_v_count > 0 {
                        v_parts.push(v_all_f32.narrow(2, hot_v_start, hot_v_count)?);
                    }

                    let v_full = if v_parts.len() == 1 {
                        v_parts.remove(0)
                    } else {
                        let v_refs: Vec<&Tensor> = v_parts.iter().collect();
                        Tensor::cat(&v_refs, 2)?
                    };
                    let v_full = repeat_kv(v_full, n_rep)?;
                    att.matmul(&v_full.contiguous()?)?.to_dtype(cache.dtype)?

                } else if sparse_thresh > 0.0 && att.is_cpu() && cache.value_bits > 0 {
                    // Fused sparse-decompress path: decompress only active V rows
                    let att_flat = att.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                    let v_store = cache.v_compressed.as_ref().unwrap();
                    let head_dim = self.head_dim;
                    let mut output = Vec::with_capacity(self.n_head * head_dim);

                    for h in 0..self.n_head {
                        let kv_h = h / n_rep; // GQA head mapping
                        let att_row = &att_flat[h * total_len..(h + 1) * total_len];

                        let head_out = match v_store {
                            CompressedValueStore::Bits4(ref v) =>
                                tq_kv::sparse_attn_v_mul_compressed_4bit(att_row, &v[kv_h], sparse_thresh),
                            CompressedValueStore::Bits8(ref v) =>
                                tq_kv::sparse_attn_v_mul_compressed_8bit(att_row, &v[kv_h], sparse_thresh),
                        };
                        output.extend_from_slice(&head_out);
                    }

                    Tensor::from_vec(output, vec![1, self.n_head, 1, head_dim], att.device())?
                        .to_dtype(cache.dtype)?
                } else {
                    // Standard path: decompress all values, matmul or sparse multiply
                    let v_f32 = if cache.value_bits == 0 {
                        repeat_kv(cache.v_raw.as_ref().unwrap().to_dtype(DType::F32)?, n_rep)?
                    } else {
                        let v_tensor = decompress_values_store(
                            cache.v_compressed.as_ref().unwrap(),
                            self.n_kv_head, self.head_dim, total_len, q.device(),
                        )?;
                        repeat_kv(v_tensor, n_rep)?
                    };

                    if sparse_thresh > 0.0 && att.is_cpu() {
                        sparse_attn_v(&att, &v_f32, self.n_head, self.head_dim, sparse_thresh)?
                            .to_dtype(cache.dtype)?
                    } else {
                        att.matmul(&v_f32.contiguous()?)?.to_dtype(cache.dtype)?
                    }
                }
            } else {
                // PREFILL: use original uncompressed keys for attention (standard path)
                let k = repeat_kv(k, n_rep)?;
                let v_for_attn = repeat_kv(v, n_rep)?;

                {
                    let att = (q.matmul(&k.t()?)? / self.attn_scale_denom)?;
                    let att = if seq_len > 1 {
                        let total_kv_len = att.dim(3)?;
                        let offset = total_kv_len - seq_len;
                        let mask_data: Vec<f32> = (0..seq_len)
                            .flat_map(|i| {
                                let qpos = offset + i;
                                (0..total_kv_len).map(move |j| if j <= qpos { 0.0f32 } else { -1e10f32 })
                            })
                            .collect();
                        let mut mask = Tensor::from_slice(&mask_data, vec![seq_len, total_kv_len], &Device::Cpu)?;
                        #[cfg(feature = "cuda")]
                        if att.is_cuda() {
                            if let Ok(gpu) = mask.to_device_auto() { mask = gpu; }
                        }
                        let mask4d = mask.unsqueeze(0)?.unsqueeze(0)?;
                        let mask4d = mask4d.broadcast_as(att.shape())?;
                        (att + mask4d)?
                    } else { att };
                    let att = softmax_last_dim(&att, backend)?;
                    att.matmul(&v_for_attn.contiguous()?)?
                }
            };

            let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
            self.attention_wo.forward(&y, backend)
        }
    }
}

