//! Unified inference engine — TqTensor-based (no candle dependency).
//!
//! Two modes:
//! - Standard: GGUF quantized models (f32 KV cache)
//! - TurboQuant: compressed KV cache (2-4 bit)

use anyhow::{Context, Result};
use tokenizers::Tokenizer;
use tq_kv::TurboQuantConfig;

use crate::cuda::{TqTensor as Tensor, TqDevice as Device, TqError};
use crate::gguf::GgufContent;
use crate::sampling::{Sampler, SamplingMode};
use crate::config::ModelArch;
use crate::models::turbo_generic;

/// Unified model backend — GenericTurboModel handles all GGUF architectures
/// with CUDA-compatible ops (RmsNorm, RoPE, softmax). When TQ is disabled,
/// all layers use uncompressed fp16 KV cache through the same code path.
pub struct ModelWeights(pub turbo_generic::GenericTurboModel);

impl ModelWeights {
    fn forward(&mut self, x: &Tensor, pos: usize) -> crate::cuda::Result<Tensor> {
        self.0.forward(x, pos)
    }

    /// Delegates to [`GenericTurboModel::forward_last_hidden`]. Added 2026-04-17
    /// for Sprint 3 Day 1 acceptance probe — returns the pre-norm, pre-lm_head
    /// hidden state EAGLE needs as its draft input.
    pub fn forward_last_hidden(&mut self, x: &Tensor, pos: usize) -> crate::cuda::Result<Tensor> {
        self.0.forward_last_hidden(x, pos)
    }

    /// Delegates to [`GenericTurboModel::project_hidden_to_logits`].
    pub fn project_hidden_to_logits(&self, hidden: &Tensor) -> crate::cuda::Result<Tensor> {
        self.0.project_hidden_to_logits(hidden)
    }
    fn forward_partial(&mut self, x: &Tensor, n_layers: usize, pos: usize) -> crate::cuda::Result<Tensor> {
        self.0.forward_partial(x, n_layers, pos)
    }
    fn truncate_kv_cache(&mut self, target_len: usize) {
        self.0.truncate_kv_cache(target_len);
    }
}

/// Generation parameters.
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: 0,
        }
    }
}

/// Runtime quality monitoring for TurboQuant compressed inference.
///
/// Tracks running perplexity estimate during generation. If PPL exceeds
/// the threshold, logs a warning suggesting higher bit width.
#[derive(Clone, Debug)]
pub struct QualityGate {
    /// PPL threshold — if exceeded, emit warning. Default: baseline * 1.5.
    pub ppl_threshold: f64,
    /// Check every N tokens (lightweight). Default: 50.
    pub check_interval: usize,
    /// Running NLL accumulator
    nll_sum: f64,
    /// Tokens evaluated
    nll_count: usize,
    /// Has already warned (avoid spam)
    warned: bool,
}

impl QualityGate {
    /// Create a quality gate with the given PPL threshold.
    pub fn new(ppl_threshold: f64) -> Self {
        Self {
            ppl_threshold,
            check_interval: 50,
            nll_sum: 0.0,
            nll_count: 0,
            warned: false,
        }
    }

    /// Update with a new token's NLL and check threshold.
    /// Returns `Some(current_ppl)` if threshold exceeded (first time only).
    pub fn update(&mut self, logits: &[f32], next_token: u32) -> Option<f64> {
        self.nll_count += 1;
        let nll = compute_nll(logits, next_token as usize);
        self.nll_sum += nll;

        if self.nll_count % self.check_interval == 0 && !self.warned {
            let current_ppl = (self.nll_sum / self.nll_count as f64).exp();
            if current_ppl > self.ppl_threshold {
                self.warned = true;
                return Some(current_ppl);
            }
        }
        None
    }

    /// Current running PPL estimate.
    pub fn current_ppl(&self) -> f64 {
        if self.nll_count == 0 { return 0.0; }
        (self.nll_sum / self.nll_count as f64).exp()
    }

    /// Reset for a new generation.
    pub fn reset(&mut self) {
        self.nll_sum = 0.0;
        self.nll_count = 0;
        self.warned = false;
    }
}

/// Unified inference engine.
pub struct Engine {
    pub model: ModelWeights,
    pub tokenizer: Tokenizer,
    device: Device,
    position: usize,
    pub(crate) eos_token_ids: Vec<u32>,
    /// Optional quality gate for runtime PPL monitoring.
    pub quality_gate: Option<QualityGate>,
}

/// Fluent builder for Engine configuration.
///
/// ```ignore
/// let engine = Engine::builder()
///     .model("models/qwen2-7b.gguf")
///     .tokenizer("models/tokenizer.json")
///     .with_turbo_quant(4)  // 4-bit KV cache compression
///     .build()?;
/// ```
pub struct EngineBuilder {
    model_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
    arch: Option<ModelArch>,
    tq_config: Option<TurboQuantConfig>,
    force_cpu: bool,
    quality_gate_threshold: Option<f64>,
}

impl EngineBuilder {
    /// Set GGUF model file path.
    pub fn model(mut self, path: impl AsRef<std::path::Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set tokenizer JSON file path.
    pub fn tokenizer(mut self, path: impl AsRef<std::path::Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set model architecture (auto-detected from GGUF if not set).
    pub fn arch(mut self, arch: ModelArch) -> Self {
        self.arch = Some(arch);
        self
    }

    /// Enable TurboQuant with specified bit width (2, 3, or 4).
    pub fn with_turbo_quant(mut self, bits: u8) -> Self {
        self.tq_config = Some(match bits {
            2 => TurboQuantConfig::extreme(),
            3 => TurboQuantConfig::aggressive(),
            _ => TurboQuantConfig::balanced(),
        });
        self
    }

    /// Enable TurboQuant with full config.
    pub fn with_turbo_quant_config(mut self, config: TurboQuantConfig) -> Self {
        self.tq_config = Some(config);
        self
    }

    /// Force CPU inference (disable CUDA).
    pub fn cpu(mut self) -> Self {
        self.force_cpu = true;
        self
    }

    /// Enable quality gate — warn if PPL exceeds threshold during generation.
    pub fn with_quality_gate(mut self, ppl_threshold: f64) -> Self {
        self.quality_gate_threshold = Some(ppl_threshold);
        self
    }

    /// Build the Engine.
    pub fn build(self) -> Result<Engine> {
        let model_path = self.model_path
            .ok_or_else(|| anyhow::anyhow!("model path not set — call .model()"))?;
        let tokenizer_path = self.tokenizer_path
            .ok_or_else(|| anyhow::anyhow!("tokenizer path not set — call .tokenizer()"))?;
        let arch = self.arch.unwrap_or(
            crate::config::detect_arch(&model_path.to_string_lossy())
        );
        let mut engine = Engine::load_with_device(
            &model_path, &tokenizer_path, arch, self.tq_config, self.force_cpu,
        )?;
        if let Some(threshold) = self.quality_gate_threshold {
            engine.enable_quality_gate(threshold);
        }
        Ok(engine)
    }
}

impl Engine {
    /// Create a fluent builder for Engine configuration.
    pub fn builder() -> EngineBuilder {
        EngineBuilder {
            model_path: None,
            tokenizer_path: None,
            arch: None,
            tq_config: None,
            force_cpu: false,
            quality_gate_threshold: None,
        }
    }

    /// Load GGUF model and tokenizer.
    /// Pass `tq_config` to enable TurboQuant compressed KV cache.
    pub fn load(
        model_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        arch: ModelArch,
        tq_config: Option<TurboQuantConfig>,
    ) -> Result<Self> {
        Self::load_with_device(model_path, tokenizer_path, arch, tq_config, false)
    }

    pub fn load_with_device(
        model_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        _arch: ModelArch,
        tq_config: Option<TurboQuantConfig>,
        force_cpu: bool,
    ) -> Result<Self> {
        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };
        eprintln!("Device: {}", if device.is_cuda() { "CUDA GPU" } else { "CPU" });

        // TQ_LAYER_SWAP: opt-in layer streaming. Must be processed BEFORE weight
        // construction so SwapCell picks swap mode at QWeight::new time.
        //   =0 (default) → standard resident weights
        //   =force       → enable swap; Phase 5 pins all layers (validates plumbing)
        //   =1           → enable swap; Phase 6 will compute pin_count from VRAM
        #[cfg(feature = "cuda")]
        let layer_swap_mode = std::env::var("TQ_LAYER_SWAP").ok();
        #[cfg(feature = "cuda")]
        let swap_active = matches!(layer_swap_mode.as_deref(), Some("force") | Some("1"));
        #[cfg(feature = "cuda")]
        if swap_active && device.is_cuda() {
            eprintln!(
                "Layer-swap mode: {} (note: incompatible with CUDA Graph)",
                layer_swap_mode.as_deref().unwrap_or("?")
            );
            crate::qmatmul::enable_layer_swap();
        }

        eprintln!("Loading model: {}", model_path.display());

        // Detect format: safetensors directory or GGUF file
        let is_safetensors = model_path.is_dir()
            || model_path.extension().map_or(false, |ext| ext == "safetensors");

        #[allow(unused_mut)]
        let mut model = if is_safetensors {
            // Safetensors (FP16/BF16) path
            let model_dir = if model_path.is_dir() {
                model_path.to_path_buf()
            } else {
                model_path.parent().unwrap_or(model_path).to_path_buf()
            };
            let tq = tq_config.unwrap_or_else(|| {
                let mut cfg = TurboQuantConfig::balanced();
                cfg.skip_layers = Some(999);
                cfg.sink_tokens = Some(0);
                cfg
            });
            let compressed = tq.skip_layers != Some(999);
            if compressed {
                eprintln!("TurboQuant FP16: {}-bit KV cache", tq.bits);
            }
            let w = turbo_generic::GenericTurboModel::from_safetensors(&model_dir, &device, tq)
                .map_err(|e| anyhow::anyhow!("Safetensors load error: {}", e))?;
            ModelWeights(w)
        } else {
            // GGUF path
            let tq = tq_config.unwrap_or_else(|| {
                let mut cfg = TurboQuantConfig::balanced();
                cfg.skip_layers = Some(999);
                cfg.sink_tokens = Some(0);
                cfg
            });
            let compressed = tq.skip_layers != Some(999);
            if compressed {
                eprintln!("TurboQuant: {}-bit KV cache", tq.bits);
            }
            // Memory-mapped GGUF path: zero-copy into GPU, no CPU Vec<u8>.
            //
            // Enabled explicitly via TQ_MMAP=1, OR implicitly when layer-swap
            // is active. Under swap we MUST keep raw_data around (for prefetch
            // re-upload), and an Owned Vec<u8> costs ~full model size on CPU
            // RAM — e.g. 18.5 GB for Qwen2.5-32B Q4_K_M, which caused a 40 GB
            // RSS regression observed on 2026-04-13. The mmap-backed RawBytes
            // variant keeps raw_data as zero-copy slices into the file, so
            // the OS pages in/out on demand and peak RSS stays around a few
            // GB instead of model-size.
            #[cfg(feature = "cuda")]
            let swap_active_mmap = matches!(
                std::env::var("TQ_LAYER_SWAP").ok().as_deref(),
                Some("1") | Some("force")
            ) && device.is_cuda();
            #[cfg(not(feature = "cuda"))]
            let swap_active_mmap = false;

            let use_mmap = std::env::var("TQ_MMAP").ok().as_deref() == Some("1")
                || swap_active_mmap;
            let w = if use_mmap {
                if std::env::var("TQ_MMAP").ok().as_deref() == Some("1") {
                    eprintln!("GGUF mmap mode (TQ_MMAP=1)");
                } else {
                    eprintln!("GGUF mmap mode (auto-enabled for TQ_LAYER_SWAP)");
                }
                let src = crate::gguf::GgufMmapSource::open(model_path)
                    .map_err(|e| anyhow::anyhow!("GGUF mmap error: {}", e))?;
                turbo_generic::GenericTurboModel::build_from_source(&src, &device, tq)
                    .map_err(|e| anyhow::anyhow!("Model load error: {}", e))?
            } else {
                let mut file = std::fs::File::open(model_path)
                    .with_context(|| format!("Cannot open model: {}", model_path.display()))?;
                let content = GgufContent::read(&mut file)
                    .map_err(|e| anyhow::anyhow!("GGUF read error: {}", e))?;
                turbo_generic::GenericTurboModel::from_gguf(content, &mut file, &device, tq)
                    .map_err(|e| anyhow::anyhow!("Model load error: {}", e))?
            };
            ModelWeights(w)
        };
        eprintln!("Model loaded!");

        // Size-pool activation is opt-in via TQ_SIZE_POOL=1. Empirical A/B
        // on qwen2:7b showed pool ON hurt Standard decode 17.1 → 13.2 tok/s —
        // the codescope audit's predicted 88-caller relief did not
        // materialise on small models, likely because shape variance per
        // layer spreads the pool across many thin buckets (~480 distinct
        // shapes for Qwen2-7B over a decode step) so hit-rate stays low
        // while bookkeeping dominates. Kept off by default; future sprint
        // to investigate whether the pool's per-bucket lookup overhead or
        // its memory retention is the real cost.
        #[cfg(feature = "cuda")]
        if std::env::var("TQ_SIZE_POOL").ok().as_deref() == Some("1") && device.is_cuda() {
            crate::cuda::size_pool_activate();
            eprintln!("  Size pool: ENABLED (opt-in via TQ_SIZE_POOL=1)");
        }

        // Install LayerSwapManager if swap mode was requested.
        //
        // TQ_LAYER_SWAP semantics:
        //   unset / 0 → off (default)
        //   1         → on only when model doesn't fit in VRAM (opt-in)
        //   force     → always on, all layers pinned (validation path)
        //
        // Per feedback (user preference): swap is NEVER auto-enabled on a
        // fitting model — the tok/s regression would break the default
        // experience. `1` turns itself off if the model fits.
        #[cfg(feature = "cuda")]
        if swap_active && device.is_cuda() {
            use crate::auto_tq::{decide_layer_swap, pinned_indices};
            use crate::layer_swap::LayerSwapManager;

            let mut inner = model;
            let n_layers = inner.0.layers.len();

            // Estimate model bytes: sum of per-layer QWeight raw_data lengths.
            let model_bytes: u64 = inner.0
                .layers
                .iter()
                .flat_map(|l| l.qweights())
                .map(|qw| qw.raw_data.len() as u64)
                .sum();
            // output.weight (lm_head) and token_embd are NOT layer weights —
            // they stay resident for the full run. Account for them in the
            // kv/fixed side so the layer budget doesn't over-commit.
            let lm_head_bytes: u64 = match &inner.0.output.inner {
                crate::qmatmul::QMatMul::Quantized(qw) => qw.raw_data.len() as u64,
                _ => 0,
            };
            // KV per-token-bytes (fp16 K + V across all layers).
            let kv_per_token_bytes: u64 = if let Some(l0) = inner.0.layers.first() {
                // 2 = K+V, 2 bytes/element (fp16), × n_kv_head × head_dim
                2u64 * l0.n_kv_head as u64 * l0.head_dim as u64 * 2 * n_layers as u64
            } else {
                0
            };
            // Worst-case context: TQ_MAX_SEQ env override, else 32K conservative
            // default (matches common long-context configs and prevents swap
            // policy from under-planning VRAM on 27B+ runs).
            let max_context: usize = std::env::var("TQ_MAX_SEQ")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32_768);

            // Fold lm_head into the "kv/fixed" account so the layer pin budget
            // leaves headroom for the always-resident lm_head (and its dequant
            // scratch during logit computation — reserve 2× the packed size
            // as a conservative estimate for cuBLAS workspace).
            let fixed_lm_head: u64 = lm_head_bytes * 2;
            let kv_per_token_bytes_adj = kv_per_token_bytes
                + (fixed_lm_head / max_context.max(1) as u64);

            let plan = decide_layer_swap(
                &device, model_bytes, n_layers, kv_per_token_bytes_adj, max_context,
            );
            eprintln!("  Layer-swap plan: {}", plan.reason);

            if !plan.enabled {
                // e.g. TQ_LAYER_SWAP=1 but model fits. Don't install manager.
                model = inner;
            } else {
                let ctx = device.cuda_context().clone();
                let compute_stream = device.cuda_stream()
                    .map_err(|e| anyhow::anyhow!("layer_swap compute stream: {}", e))?;
                let pinned_layers = pinned_indices(n_layers, plan.pin_count);
                let mut manager = LayerSwapManager::new(
                    ctx,
                    compute_stream,
                    &pinned_layers,
                ).map_err(|e| anyhow::anyhow!("layer_swap manager: {}", e))?;

                let qws_per_layer: Vec<Vec<crate::layer_swap::QWeightPtr>> = inner.0
                    .layers
                    .iter()
                    .map(|l| l.qweight_ptrs())
                    .collect();

                // Pin the selected layers at load time (synchronous H2D).
                for &layer_idx in &pinned_layers {
                    unsafe {
                        manager.pin_layer_direct(layer_idx, &qws_per_layer[layer_idx])
                            .map_err(|e| anyhow::anyhow!("pin layer {}: {}", layer_idx, e))?;
                    }
                }

                // Pre-allocate the streaming lane pool once (no per-prefetch
                // cudaMalloc). Uses the first non-pinned layer as the
                // structural template — same QWeight count + sizes across
                // layers holds for all decoder-only transformers we target.
                if n_layers > pinned_layers.len() {
                    let template_idx = (0..n_layers)
                        .find(|i| !manager.is_pinned(*i))
                        .unwrap_or(0);
                    let n_lanes: usize = std::env::var("TQ_SWAP_LANES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(2);
                    unsafe {
                        manager.allocate_lanes(n_lanes, &qws_per_layer[template_idx])
                            .map_err(|e| anyhow::anyhow!("allocate_lanes: {}", e))?;
                    }
                }

                // Kick initial prefetch for the first un-pinned layer so the
                // compute stream has something to wait on at layer 0.
                for layer_idx in 0..n_layers {
                    if !manager.is_pinned(layer_idx) {
                        unsafe {
                            manager.start_prefetch_direct(
                                layer_idx,
                                &qws_per_layer[layer_idx],
                            ).map_err(|e| anyhow::anyhow!("initial prefetch L{}: {}", layer_idx, e))?;
                        }
                        break;
                    }
                }

                // Explicitly upload lm_head to GPU now — it's queried lazily
                // by the final matmul and would otherwise OOM mid-forward on
                // VRAM-tight models (Qwen2.5-32B's lm_head alone is ~380MB
                // Q4K + 1-1.5GB dequant scratch at logit time).
                if let crate::qmatmul::QMatMul::Quantized(qw) = &inner.0.output.inner {
                    if let Err(e) = qw.upload_to_gpu(&device) {
                        eprintln!("  Layer-swap: lm_head upload failed: {} — may OOM mid-forward", e);
                    }
                }

                eprintln!(
                    "  Layer-swap installed: {} pinned, {} streamed (lm_head resident)",
                    pinned_layers.len(),
                    n_layers.saturating_sub(pinned_layers.len()),
                );

                inner.0.layer_swap = Some(manager);
                inner.0.layer_qweight_ptrs = qws_per_layer;
                model = inner;
            }
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load error: {}", e))?;

        let stop_tokens = [
            "<|eot_id|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|end|>",
            "<end_of_turn>",
            "</s>",
        ];
        let mut eos_token_ids: Vec<u32> = stop_tokens
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        if eos_token_ids.is_empty() {
            eos_token_ids.push(2); // fallback
        }
        eprintln!("EOS token IDs: {:?}", eos_token_ids);

        Ok(Self { model, tokenizer, device, position: 0, eos_token_ids, quality_gate: None })
    }

    /// Clear KV cache (both position counter and per-layer KV buffers).
    pub fn clear_cache(&mut self) {
        self.position = 0;
        self.model.0.clear_kv_cache();
    }

    /// Enable quality gate — monitors running PPL during generation.
    pub fn enable_quality_gate(&mut self, ppl_threshold: f64) {
        self.quality_gate = Some(QualityGate::new(ppl_threshold));
    }

    /// Create a token input tensor from token IDs.
    fn make_input(&self, tokens: &[u32]) -> crate::cuda::Result<Tensor> {
        let data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let len = data.len();
        Tensor::from_vec(data, vec![1, len], &self.device)
    }

    /// Generate text with streaming callback.
    pub fn generate<F>(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        // add_special_tokens: controls whether the tokenizer's post-processor
        // (BOS/EOS insertion) runs. Chat template special tokens like <|im_start|>
        // are always recognized via the vocabulary regardless of this flag.
        // TQ_NO_SPECIAL=1 disables post-processor for debugging.
        let add_special = !std::env::var("TQ_NO_SPECIAL").ok().map_or(false, |v| v == "1");
        let encoding = self.tokenizer
            .encode(prompt, add_special)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            anyhow::bail!("Empty prompt");
        }

        eprintln!("Prompt tokens: {}", prompt_tokens.len());

        // Debug: dump tokenization when TQ_DEBUG_TOKENS=1
        if std::env::var("TQ_DEBUG_TOKENS").ok().map_or(false, |v| v == "1") {
            let show = prompt_tokens.len().min(40);
            eprintln!("[DEBUG-TOK] first {} IDs: {:?}", show, &prompt_tokens[..show]);
            // Decode each token individually to show what it maps to
            let mut tok_strs = Vec::new();
            for &id in &prompt_tokens[..show] {
                let s = self.tokenizer.decode(&[id], false).unwrap_or_else(|_| format!("?{}", id));
                tok_strs.push(format!("{}:{:?}", id, s));
            }
            eprintln!("[DEBUG-TOK] tokens: {}", tok_strs.join(" | "));
            // Also show last 10
            if prompt_tokens.len() > 40 {
                let tail = &prompt_tokens[prompt_tokens.len()-10..];
                let mut tail_strs = Vec::new();
                for &id in tail {
                    let s = self.tokenizer.decode(&[id], false).unwrap_or_else(|_| format!("?{}", id));
                    tail_strs.push(format!("{}:{:?}", id, s));
                }
                eprintln!("[DEBUG-TOK] last 10: {}", tail_strs.join(" | "));
            }
        }

        let mut sampler = if params.temperature <= 0.0 {
            Sampler::new(SamplingMode::ArgMax, params.seed)
        } else {
            Sampler::new(
                SamplingMode::TopKTopP {
                    k: params.top_k,
                    p: params.top_p as f64,
                    temperature: params.temperature as f64,
                },
                params.seed,
            )
        };

        self.position = 0;
        let input = self.make_input(&prompt_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let logits = self.model.forward(&input, self.position)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        self.position += prompt_tokens.len();

        // Debug: dump top-10 logits after prefill when TQ_DEBUG_TOKENS=1
        if std::env::var("TQ_DEBUG_TOKENS").ok().map_or(false, |v| v == "1") {
            let logits_last = extract_last_logits(&logits).ok();
            let logits_vec: Vec<f32> = logits_last
                .and_then(|l| l.to_vec1().ok())
                .unwrap_or_default();
            if !logits_vec.is_empty() {
                let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate()
                    .map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top10: Vec<String> = indexed.iter().take(10)
                    .map(|(id, score)| {
                        let s = self.tokenizer.decode(&[*id as u32], false).unwrap_or_default();
                        format!("{}:{:?}({:.2})", id, s, score)
                    }).collect();
                eprintln!("[DEBUG-LOGITS] top-10 after prefill: {}", top10.join(" | "));
            }
        }

        let logits = extract_last_logits(&logits)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut next_token = sampler.sample(&logits)
            .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

        let mut output = String::new();
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut prev_decoded_len = 0;
        let mut n_generated = 0u32;

        while n_generated < params.max_tokens {
            if self.eos_token_ids.contains(&next_token) { break; }

            all_tokens.push(next_token);
            let full_text = self.tokenizer.decode(&all_tokens, true).unwrap_or_default();
            if full_text.len() > prev_decoded_len {
                let new_text = &full_text[prev_decoded_len..];
                on_token(new_text);
                output.push_str(new_text);
            }
            prev_decoded_len = full_text.len();

            let input = self.make_input(&[next_token])
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let logits = self.model.forward(&input, self.position)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            self.position += 1;

            let logits = extract_last_logits(&logits)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            // Download logits once — sampler needs CPU data anyway.
            // Apply repeat penalty on CPU to avoid GPU re-upload round-trip.
            let mut logits_vec = logits.to_vec1()
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            if params.repeat_penalty != 1.0 && !all_tokens.is_empty() {
                for &token_id in &all_tokens {
                    let idx = token_id as usize;
                    if idx < logits_vec.len() {
                        let score = logits_vec[idx];
                        logits_vec[idx] = if score > 0.0 {
                            score / params.repeat_penalty
                        } else {
                            score * params.repeat_penalty
                        };
                    }
                }
            }
            let logits_cpu = Tensor::from_vec(logits_vec.clone(), logits.shape().to_vec(), &Device::Cpu)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            next_token = sampler.sample(&logits_cpu)
                .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

            // Quality gate: reuse already-downloaded logits (no second DtoH)
            if let Some(ref mut gate) = self.quality_gate {
                if let Some(ppl) = gate.update(&logits_vec, next_token) {
                    eprintln!(
                        "Warning: Quality gate: PPL {:.1} exceeds threshold {:.1} at token {}. \
                         Consider using higher bit width (--tq-bits 4) or fewer compressed layers (TQ_SKIP).",
                        ppl, gate.ppl_threshold, n_generated,
                    );
                }
            }

            n_generated += 1;
        }

        // Reset quality gate for next generation
        if let Some(ref mut gate) = self.quality_gate {
            gate.reset();
        }

        Ok(output)
    }

    /// Generate silently (no stdout output).
    pub fn generate_silent(&mut self, prompt: &str, params: &GenerationParams) -> Result<String> {
        self.generate(prompt, params, |_| {})
    }

    /// Generate with streaming to stdout.
    pub fn generate_streaming(&mut self, prompt: &str, params: &GenerationParams) -> Result<String> {
        use std::io::Write;
        self.generate(prompt, params, |token| {
            print!("{}", token);
            let _ = std::io::stdout().flush();
        })
    }

    /// Self-speculative decoding: first N layers as draft, full model verifies.
    /// No separate draft model — reuses the same model's early layers.
    ///
    /// Draft: forward_partial(draft_layers) produces candidate token at ~25% compute.
    /// Verify: full forward on candidate. If argmax matches, accept + advance.
    /// If mismatch: use verify's token, truncate draft KV, retry.
    ///
    /// Returns (output_text, avg_accepted_per_step).
    pub fn self_speculative_generate<F>(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        draft_layers: usize,
        spec_k: usize,
        mut on_token: F,
    ) -> Result<(String, f64)>
    where
        F: FnMut(&str),
    {
        let add_special = !std::env::var("TQ_NO_SPECIAL").ok().map_or(false, |v| v == "1");
        let encoding = self.tokenizer.encode(prompt, add_special)
            .map_err(|e| anyhow::anyhow!("Tokenize: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() { anyhow::bail!("Empty prompt"); }

        eprintln!("Self-spec: draft_layers={}, K={}, prompt={} tokens",
            draft_layers, spec_k, prompt_tokens.len());

        let mut sampler = Sampler::new(SamplingMode::ArgMax, params.seed);

        // Prefill with full model
        self.position = 0;
        let input = self.make_input(&prompt_tokens).map_err(|e| anyhow::anyhow!("{}", e))?;
        let logits = self.model.forward(&input, self.position).map_err(|e| anyhow::anyhow!("{}", e))?;
        self.position += prompt_tokens.len();
        let logits = extract_last_logits(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut next_token = sampler.sample(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut output = String::new();
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut prev_decoded_len = 0;
        let mut n_generated = 0u32;
        let mut total_accepted = 0u64;
        let mut total_steps = 0u64;

        while n_generated < params.max_tokens {
            if self.eos_token_ids.contains(&next_token) { break; }

            // Accept the token from previous verify (or prefill)
            all_tokens.push(next_token);
            let full_text = self.tokenizer.decode(&all_tokens, true).unwrap_or_default();
            if full_text.len() > prev_decoded_len {
                on_token(&full_text[prev_decoded_len..]);
                output.push_str(&full_text[prev_decoded_len..]);
            }
            prev_decoded_len = full_text.len();
            n_generated += 1;

            // Draft phase: generate spec_k candidates using partial forward
            let mut draft_tokens = Vec::with_capacity(spec_k);
            let mut draft_token = next_token;
            let draft_start_pos = self.position;
            for _ in 0..spec_k {
                let input = self.make_input(&[draft_token]).map_err(|e| anyhow::anyhow!("{}", e))?;
                let logits = self.model.forward_partial(&input, draft_layers, self.position)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                self.position += 1;
                let logits = extract_last_logits(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;
                draft_token = sampler.sample(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;
                draft_tokens.push(draft_token);
                if self.eos_token_ids.contains(&draft_token) { break; }
            }

            // Verify phase: run full model on each draft token sequentially
            // (batch verify would be faster but requires careful KV management)
            self.position = draft_start_pos;
            // Truncate draft-layer KV back to draft_start_pos
            self.model.truncate_kv_cache(draft_start_pos);

            let mut accepted = 0usize;
            let mut verify_token = next_token; // the already-accepted token
            for &dt in &draft_tokens {
                let input = self.make_input(&[verify_token]).map_err(|e| anyhow::anyhow!("{}", e))?;
                let logits = self.model.forward(&input, self.position)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                self.position += 1;
                let logits = extract_last_logits(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;
                let target_token = sampler.sample(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;

                if target_token == dt {
                    // Match — accept draft token
                    all_tokens.push(dt);
                    let full_text = self.tokenizer.decode(&all_tokens, true).unwrap_or_default();
                    if full_text.len() > prev_decoded_len {
                        on_token(&full_text[prev_decoded_len..]);
                        output.push_str(&full_text[prev_decoded_len..]);
                    }
                    prev_decoded_len = full_text.len();
                    n_generated += 1;
                    accepted += 1;
                    verify_token = dt;
                    if self.eos_token_ids.contains(&dt) { break; }
                } else {
                    // Mismatch — use target's token, discard remaining drafts
                    next_token = target_token;
                    break;
                }
            }

            // If all draft tokens accepted, get next from last verify
            if accepted == draft_tokens.len() {
                let input = self.make_input(&[verify_token]).map_err(|e| anyhow::anyhow!("{}", e))?;
                let logits = self.model.forward(&input, self.position)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                self.position += 1;
                let logits = extract_last_logits(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;
                next_token = sampler.sample(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;
            }

            // Truncate KV to current accepted position
            self.model.truncate_kv_cache(self.position);

            total_accepted += accepted as u64;
            total_steps += 1;
        }

        let avg_accepted = if total_steps > 0 { total_accepted as f64 / total_steps as f64 } else { 0.0 };
        eprintln!("Self-spec: {} tokens, {:.1} accepted/step", n_generated, avg_accepted);
        Ok((output, avg_accepted))
    }

    /// Speculative decoding: draft model proposes K tokens, target model verifies in one pass.
    /// Returns accepted tokens per step on average.
    pub fn speculative_generate<F>(
        target: &mut Engine,
        draft: &mut Engine,
        prompt: &str,
        params: &GenerationParams,
        spec_k: usize,  // number of draft tokens per speculation step
        mut on_token: F,
    ) -> Result<(String, f64)>  // (output, avg_accepted_per_step)
    where
        F: FnMut(&str),
    {
        let encoding = target.tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            anyhow::bail!("Empty prompt");
        }

        eprintln!("Speculative decoding: K={}, prompt tokens: {}", spec_k, prompt_tokens.len());

        let mut sampler = Sampler::new(SamplingMode::ArgMax, params.seed);

        // Prefill both models with prompt
        target.position = 0;
        draft.position = 0;

        let input = target.make_input(&prompt_tokens).map_err(|e| anyhow::anyhow!("{}", e))?;
        let target_logits = target.model.forward(&input, 0).map_err(|e| anyhow::anyhow!("{}", e))?;
        target.position = prompt_tokens.len();

        let draft_input = draft.make_input(&prompt_tokens).map_err(|e| anyhow::anyhow!("{}", e))?;
        let _ = draft.model.forward(&draft_input, 0).map_err(|e| anyhow::anyhow!("{}", e))?;
        draft.position = prompt_tokens.len();

        let target_logits = extract_last_logits(&target_logits).map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut next_token = sampler.sample(&target_logits).map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut output = String::new();
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut prev_decoded_len = 0;
        let mut n_generated = 0u32;
        let mut total_accepted = 0usize;
        let mut total_steps = 0usize;

        while n_generated < params.max_tokens {
            if target.eos_token_ids.contains(&next_token) { break; }

            // ── Draft phase: generate K candidate tokens with draft model ──
            let mut draft_tokens = Vec::with_capacity(spec_k);
            let mut draft_token = next_token;

            for _ in 0..spec_k {
                let d_input = draft.make_input(&[draft_token]).map_err(|e| anyhow::anyhow!("{}", e))?;
                let d_logits = draft.model.forward(&d_input, draft.position).map_err(|e| anyhow::anyhow!("{}", e))?;
                draft.position += 1;

                let d_logits = extract_last_logits(&d_logits).map_err(|e| anyhow::anyhow!("{}", e))?;
                draft_token = sampler.sample(&d_logits).map_err(|e| anyhow::anyhow!("{}", e))?;
                draft_tokens.push(draft_token);

                if target.eos_token_ids.contains(&draft_token) { break; }
            }

            // ── Verify phase: run target model on [next_token, draft_tokens...] ──
            // Sequential verify: one token at a time. Batched verify needs the compressed
            // attention path to support seq_len > 1 with existing cache (future work).
            let mut verify_tokens = vec![next_token];
            verify_tokens.extend_from_slice(&draft_tokens);

            let vocab_size = target.tokenizer.get_vocab_size(false);
            let n_verify = verify_tokens.len();
            // Batched verify: single forward pass for all K+1 tokens.
            // Inline causal mask handles non-square attention (seq_len × cache+seq_len).
            let v_input = target.make_input(&verify_tokens).map_err(|e| anyhow::anyhow!("{}", e))?;
            let v_logits = target.model.forward(&v_input, target.position)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let v_logits_flat = v_logits.to_vec1().map_err(|e| anyhow::anyhow!("{}", e))?;

            // ── Accept phase: find longest matching prefix ──
            // Position i in v_logits predicts token at position i+1.
            // v_logits[0] predicts what comes after next_token → should match draft_tokens[0]
            let mut n_accepted = 0;
            for i in 0..draft_tokens.len().min(n_verify - 1) {
                let logit_offset = i * vocab_size;
                let logit_slice = &v_logits_flat[logit_offset..logit_offset + vocab_size];
                let target_pred = logit_slice.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);

                if target_pred == draft_tokens[i] {
                    n_accepted += 1;
                } else {
                    break;
                }
            }

            // Accept: next_token + n_accepted draft tokens
            // Emit next_token (already known)
            all_tokens.push(next_token);
            for i in 0..n_accepted {
                all_tokens.push(draft_tokens[i]);
            }

            // Decode and stream new text
            let full_text = target.tokenizer.decode(&all_tokens, true).unwrap_or_default();
            if full_text.len() > prev_decoded_len {
                let new_text = &full_text[prev_decoded_len..];
                on_token(new_text);
                output.push_str(new_text);
            }
            prev_decoded_len = full_text.len();

            n_generated += 1 + n_accepted as u32;
            total_accepted += 1 + n_accepted; // 1 for next_token + n_accepted drafts
            total_steps += 1;

            // Update target position (accepted next_token + n_accepted drafts)
            target.position += 1 + n_accepted;

            // Roll back draft model to match target position.
            // Truncate KV cache instead of full re-prefill (O(1) vs O(n)).
            // Draft generated spec_k tokens but only n_accepted were accepted.
            // The draft's KV cache has entries for rejected tokens — truncate them.
            if n_accepted < draft_tokens.len() {
                // Draft KV has: prompt + [prev accepted] + next_token + n_accepted + (spec_k - n_accepted rejected)
                // Target position is: prompt_len + all_accepted + 1 + n_accepted
                // Truncate draft KV to target.position
                draft.model.0.truncate_kv_cache(target.position);
                draft.position = target.position;
            }

            // Next token: sample from last position's target logits
            let last_logit_offset = n_accepted * vocab_size;
            if last_logit_offset + vocab_size <= v_logits_flat.len() {
                let last_logits = &v_logits_flat[last_logit_offset..last_logit_offset + vocab_size];
                let last_logits_t = Tensor::from_vec(
                    last_logits.to_vec(), vec![vocab_size], &Device::Cpu,
                ).map_err(|e| anyhow::anyhow!("{}", e))?;
                next_token = sampler.sample(&last_logits_t).map_err(|e| anyhow::anyhow!("{}", e))?;
            } else {
                break;
            }
        }

        let avg_accepted = if total_steps > 0 { total_accepted as f64 / total_steps as f64 } else { 0.0 };
        eprintln!("Speculative decode: {:.1} tokens/step avg ({} accepted / {} steps)",
            avg_accepted, total_accepted, total_steps);

        Ok((output, avg_accepted))
    }

    /// Compute perplexity on a text.
    ///
    /// Token-by-token evaluation: at each position, the model predicts the next token.
    /// PPL = exp(average negative log-likelihood).
    pub fn compute_perplexity(&mut self, text: &str) -> Result<f64> {
        // `add_special_tokens=true` so the tokenizer's TemplateProcessing
        // (e.g. Gemma 2's <bos> prefix) is honoured. Without BOS, Gemma 2
        // operates out-of-distribution and PPL explodes (~150k on 52-token
        // prose). Qwen2 tokenizer has a plain ByteLevel post-processor with
        // no BOS, so this flag is a no-op there.
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
        let tokens = encoding.get_ids().to_vec();
        let n_tokens = tokens.len();

        if n_tokens < 2 {
            anyhow::bail!("Text too short for perplexity (need >= 2 tokens)");
        }

        eprintln!("Perplexity eval: {} tokens", n_tokens);

        let mut total_nll = 0.0f64;
        let mut n_evaluated = 0usize;
        self.position = 0;

        // Process first token (no prediction possible)
        let first = self.make_input(&tokens[0..1])
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let logits = self.model.forward(&first, 0)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        self.position = 1;

        // Get prediction for token[1] from the first forward pass
        let logits = extract_last_logits(&logits)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let logit_vec = logits.to_vec1()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let nll = compute_nll(&logit_vec, tokens[1] as usize);
        total_nll += nll;
        n_evaluated += 1;

        // Token-by-token evaluation
        for t in 1..n_tokens - 1 {
            let input = self.make_input(&tokens[t..t + 1])
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let logits = self.model.forward(&input, self.position)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            self.position += 1;

            let logits = extract_last_logits(&logits)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let logit_vec = logits.to_vec1()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            let nll = compute_nll(&logit_vec, tokens[t + 1] as usize);
            total_nll += nll;
            n_evaluated += 1;

            if n_evaluated % 100 == 0 {
                let current_ppl = (total_nll / n_evaluated as f64).exp();
                eprintln!("  [{}/{}] PPL: {:.3}", n_evaluated, n_tokens - 1, current_ppl);
            }
        }

        let avg_nll = total_nll / n_evaluated as f64;
        let perplexity = avg_nll.exp();

        eprintln!("Final perplexity: {:.3} ({} tokens)", perplexity, n_evaluated);
        Ok(perplexity)
    }
}

/// Compute negative log-likelihood for a target token given logits.
fn compute_nll(logits: &[f32], target_id: usize) -> f64 {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f64 = logits.iter()
        .map(|&x| ((x - max_val) as f64).exp())
        .sum::<f64>()
        .ln() + max_val as f64;
    let target_logit = if target_id < logits.len() {
        logits[target_id] as f64
    } else {
        logits[0] as f64
    };
    log_sum_exp - target_logit
}

/// Extract the last token's logits from model output.
/// Model output may be [batch, seq_len, vocab] or [batch, vocab].
fn extract_last_logits(logits: &Tensor) -> crate::cuda::Result<Tensor> {
    let shape = logits.shape();
    match shape.len() {
        3 => {
            // [batch, seq_len, vocab] → take last seq position, squeeze batch
            let seq_len = shape[1];
            logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)
        }
        2 => {
            // [batch, vocab] → squeeze batch
            logits.squeeze(0)
        }
        1 => {
            // Already [vocab]
            Ok(logits.clone())
        }
        _ => Err(TqError::Msg(format!("unexpected logits shape: {:?}", shape))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::{TqTensor as Tensor, TqDevice as Device};

    #[test]
    fn test_extract_last_logits_3d() {
        // [1, 5, 100] — batch=1, seq_len=5, vocab=100
        let data: Vec<f32> = (0..500).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, vec![1, 5, 100], &Device::Cpu).unwrap();
        let result = extract_last_logits(&t).unwrap();
        assert_eq!(result.shape(), &[100]);
        let vals = result.to_vec1().unwrap();
        // Last seq position is row 4 (indices 400..500)
        assert!((vals[0] - 400.0).abs() < 1e-6);
        assert!((vals[99] - 499.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_last_logits_2d() {
        // [1, 100] — batch=1, vocab=100
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, vec![1, 100], &Device::Cpu).unwrap();
        let result = extract_last_logits(&t).unwrap();
        assert_eq!(result.shape(), &[100]);
        let vals = result.to_vec1().unwrap();
        assert!((vals[0] - 0.0).abs() < 1e-6);
        assert!((vals[99] - 99.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_last_logits_1d() {
        // [100] — already flat
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data.clone(), vec![100], &Device::Cpu).unwrap();
        let result = extract_last_logits(&t).unwrap();
        assert_eq!(result.shape(), &[100]);
        assert_eq!(result.to_vec1().unwrap(), data);
    }

    #[test]
    fn test_compute_nll() {
        // Known logits: [2.0, 1.0, 0.1] with target=0
        // softmax(2.0, 1.0, 0.1):
        //   exp(2)=7.389, exp(1)=2.718, exp(0.1)=1.105
        //   sum = 11.212
        //   p(0) = 7.389/11.212 = 0.659
        //   NLL = -ln(0.659) ≈ 0.417
        let logits = vec![2.0f32, 1.0, 0.1];
        let nll = compute_nll(&logits, 0);
        assert!((nll - 0.417).abs() < 0.01, "NLL was {}", nll);

        // Target with low probability should have high NLL
        let nll_low = compute_nll(&logits, 2);
        assert!(nll_low > nll, "low-prob target should have higher NLL");

        // NLL should always be non-negative for valid targets
        assert!(nll >= 0.0);
        assert!(nll_low >= 0.0);
    }
}
