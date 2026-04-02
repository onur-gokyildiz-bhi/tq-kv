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
struct ModelWeights(turbo_generic::GenericTurboModel);

impl ModelWeights {
    fn forward(&mut self, x: &Tensor, pos: usize) -> crate::cuda::Result<Tensor> {
        self.0.forward(x, pos)
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
    eos_token_ids: Vec<u32>,
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

        eprintln!("Loading model: {}", model_path.display());

        // Detect format: safetensors directory or GGUF file
        let is_safetensors = model_path.is_dir()
            || model_path.extension().map_or(false, |ext| ext == "safetensors");

        let model = if is_safetensors {
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
            let mut file = std::fs::File::open(model_path)
                .with_context(|| format!("Cannot open model: {}", model_path.display()))?;

            let content = GgufContent::read(&mut file)
                .map_err(|e| anyhow::anyhow!("GGUF read error: {}", e))?;

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
            let w = turbo_generic::GenericTurboModel::from_gguf(content, &mut file, &device, tq)
                .map_err(|e| anyhow::anyhow!("Model load error: {}", e))?;
            ModelWeights(w)
        };
        eprintln!("Model loaded!");

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

    /// Clear KV cache.
    pub fn clear_cache(&mut self) {
        self.position = 0;
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
        let encoding = self.tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            anyhow::bail!("Empty prompt");
        }

        eprintln!("Prompt tokens: {}", prompt_tokens.len());

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
            // Apply repetition penalty
            let logits = if params.repeat_penalty != 1.0 && !all_tokens.is_empty() {
                let mut logits_vec = logits.to_vec1()
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
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
                Tensor::from_vec(logits_vec, logits.shape().to_vec(), &self.device)
                    .map_err(|e| anyhow::anyhow!("{}", e))?
            } else {
                logits
            };

            next_token = sampler.sample(&logits)
                .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

            // Quality gate: monitor running PPL
            if let Some(ref mut gate) = self.quality_gate {
                let logits_vec = logits.to_vec1()
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
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

    /// Compute perplexity on a text.
    ///
    /// Token-by-token evaluation: at each position, the model predicts the next token.
    /// PPL = exp(average negative log-likelihood).
    pub fn compute_perplexity(&mut self, text: &str) -> Result<f64> {
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
