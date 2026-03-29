//! Candle-based unified inference engine.
//!
//! Two modes:
//! - Standard: candle's stock quantized models (f32 KV cache)
//! - TurboQuant: forked models with compressed KV cache (2-4 bit)

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama as qlm;
use candle_transformers::models::quantized_qwen2 as qqw;
use tokenizers::Tokenizer;
use tq_kv::TurboQuantConfig;

use crate::config::ModelArch;
use crate::models::turbo_generic;

/// Model variants — standard or TurboQuant enhanced.
enum ModelWeights {
    Llama(qlm::ModelWeights),
    Qwen2(qqw::ModelWeights),
    TurboGeneric(turbo_generic::GenericTurboModel),
}

impl ModelWeights {
    fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, pos),
            Self::Qwen2(m) => m.forward(x, pos),
            Self::TurboGeneric(m) => m.forward(x, pos),
        }
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
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    position: usize,
    eos_token_id: u32,
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
        arch: ModelArch,
        tq_config: Option<TurboQuantConfig>,
        force_cpu: bool,
    ) -> Result<Self> {
        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).context("Device init failed")?
        };
        eprintln!("Device: {}", if device.is_cuda() { "CUDA GPU" } else { "CPU" });

        eprintln!("Loading model: {}", model_path.display());
        let mut file = std::fs::File::open(model_path)
            .with_context(|| format!("Cannot open model: {}", model_path.display()))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("GGUF read error: {}", e))?;

        let model = match (arch, tq_config) {
            (_, Some(tq)) => {
                // TurboQuant: use generic model for ANY architecture
                eprintln!("TurboQuant Generic: {}-bit KV cache (auto-detecting architecture from GGUF)", tq.bits);
                let w = turbo_generic::GenericTurboModel::from_gguf(content, &mut file, &device, tq)
                    .map_err(|e| anyhow::anyhow!("TurboGeneric load error: {}", e))?;
                ModelWeights::TurboGeneric(w)
            }
            (ModelArch::Llama, None) => {
                let w = qlm::ModelWeights::from_gguf(content, &mut file, &device)
                    .map_err(|e| anyhow::anyhow!("Llama load error: {}", e))?;
                ModelWeights::Llama(w)
            }
            (ModelArch::Qwen2, None) => {
                let w = qqw::ModelWeights::from_gguf(content, &mut file, &device)
                    .map_err(|e| anyhow::anyhow!("Qwen2 load error: {}", e))?;
                ModelWeights::Qwen2(w)
            }
        };
        eprintln!("Model loaded!");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load error: {}", e))?;

        let eos_token_id = tokenizer
            .token_to_id("<|eot_id|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        Ok(Self { model, tokenizer, device, position: 0, eos_token_id, quality_gate: None })
    }

    /// Clear KV cache.
    pub fn clear_cache(&mut self) {
        self.position = 0;
    }

    /// Enable quality gate — monitors running PPL during generation.
    /// If PPL exceeds `threshold`, logs a warning suggesting higher bit width.
    pub fn enable_quality_gate(&mut self, ppl_threshold: f64) {
        self.quality_gate = Some(QualityGate::new(ppl_threshold));
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
        let encoding = self.tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            anyhow::bail!("Empty prompt");
        }

        eprintln!("Prompt tokens: {}", prompt_tokens.len());

        let sampling = if params.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::TopKThenTopP {
                k: params.top_k,
                p: params.top_p as f64,
                temperature: params.temperature as f64,
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(params.seed, sampling);

        self.position = 0;
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.position)?;
        self.position += prompt_tokens.len();

        let logits = logits.squeeze(0)?;
        let logits = extract_last_logits(&logits)?.to_device(&Device::Cpu)?;
        let mut next_token = logits_processor
            .sample(&logits)
            .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

        let mut output = String::new();
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut prev_decoded_len = 0;
        let mut n_generated = 0u32;

        while n_generated < params.max_tokens {
            if next_token == self.eos_token_id { break; }

            // Incremental decode: decode ALL generated tokens, then diff with previous.
            // This preserves spaces that sentencepiece tokenizers encode as part of
            // the token (e.g., "▁Hello" → " Hello"). Single-token decode loses the space.
            all_tokens.push(next_token);
            let full_text = self.tokenizer.decode(&all_tokens, true).unwrap_or_default();
            if full_text.len() > prev_decoded_len {
                let new_text = &full_text[prev_decoded_len..];
                on_token(new_text);
                output.push_str(new_text);
            }
            prev_decoded_len = full_text.len();

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, self.position)?;
            self.position += 1;

            let logits = logits.squeeze(0)?;
            let logits = extract_last_logits(&logits)?.to_device(&Device::Cpu)?;

            // Apply repetition penalty: reduce logits for tokens already generated
            let logits = if params.repeat_penalty != 1.0 && !all_tokens.is_empty() {
                let mut logits_vec = logits.to_vec1::<f32>()?;
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
                Tensor::from_vec(logits_vec, logits.shape(), logits.device())?
            } else {
                logits
            };

            next_token = logits_processor
                .sample(&logits)
                .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

            // Quality gate: monitor running PPL
            if let Some(ref mut gate) = self.quality_gate {
                let logits_vec = logits.to_vec1::<f32>()?;
                if let Some(ppl) = gate.update(&logits_vec, next_token) {
                    eprintln!(
                        "⚠ Quality gate: PPL {:.1} exceeds threshold {:.1} at token {}. \
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

    /// Compute perplexity on a text.
    ///
    /// Token-by-token evaluation: at each position, the model predicts the next token.
    /// PPL = exp(average negative log-likelihood).
    /// `stride`: number of tokens to process in the prefill before switching to per-token eval.
    pub fn compute_perplexity(&mut self, text: &str, _stride: usize) -> Result<f64> {
        let encoding = self.tokenizer
            .encode(text, false)
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
        let first = Tensor::new(&tokens[0..1], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&first, 0)?;
        self.position = 1;

        // Get prediction for token[1] from the first forward pass
        let logits = logits.squeeze(0)?.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
        let logit_vec = if logits.dims().len() == 2 {
            logits.get(logits.dims()[0] - 1)?.to_vec1::<f32>()?
        } else {
            logits.to_vec1::<f32>()?
        };
        let nll = compute_nll(&logit_vec, tokens[1] as usize);
        total_nll += nll;
        n_evaluated += 1;

        // Token-by-token evaluation
        for t in 1..n_tokens - 1 {
            let input = Tensor::new(&tokens[t..t + 1], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, self.position)?;
            self.position += 1;

            let logits = logits.squeeze(0)?.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
            let logit_vec = if logits.dims().len() == 2 {
                logits.get(logits.dims()[0] - 1)?.to_vec1::<f32>()?
            } else {
                logits.to_vec1::<f32>()?
            };

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

fn extract_last_logits(logits: &Tensor) -> candle_core::Result<Tensor> {
    let dims = logits.dims();
    if dims.len() == 2 {
        logits.get(dims[0] - 1)
    } else {
        Ok(logits.clone())
    }
}
