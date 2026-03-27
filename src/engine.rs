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
use crate::models::{turbo_generic, turbo_llama, turbo_qwen2};

/// Model variants — standard or TurboQuant enhanced.
enum ModelWeights {
    Llama(qlm::ModelWeights),
    Qwen2(qqw::ModelWeights),
    TurboLlama(turbo_llama::ModelWeights),
    TurboQwen2(turbo_qwen2::ModelWeights),
    TurboGeneric(turbo_generic::GenericTurboModel),
}

impl ModelWeights {
    fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, pos),
            Self::Qwen2(m) => m.forward(x, pos),
            Self::TurboLlama(m) => m.forward(x, pos),
            Self::TurboQwen2(m) => m.forward(x, pos),
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

/// Unified inference engine.
pub struct Engine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    position: usize,
    eos_token_id: u32,
}

impl Engine {
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

        Ok(Self { model, tokenizer, device, position: 0, eos_token_id })
    }

    /// Clear KV cache.
    pub fn clear_cache(&mut self) {
        self.position = 0;
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
        let mut n_generated = 0u32;

        while n_generated < params.max_tokens {
            if next_token == self.eos_token_id { break; }

            let token_text = self.tokenizer.decode(&[next_token], true).unwrap_or_default();
            on_token(&token_text);
            output.push_str(&token_text);

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, self.position)?;
            self.position += 1;

            let logits = logits.squeeze(0)?;
            let logits = extract_last_logits(&logits)?.to_device(&Device::Cpu)?;
            next_token = logits_processor
                .sample(&logits)
                .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;

            n_generated += 1;
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
    pub fn compute_perplexity(&mut self, text: &str, stride: usize) -> Result<f64> {
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
