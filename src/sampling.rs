//! Token sampling — replacement for candle_transformers LogitsProcessor.
//!
//! Supports ArgMax, TopK, TopP (nucleus), and temperature scaling.

use crate::cuda::{TqTensor, Result, TqError};

/// Sampling strategy.
#[derive(Debug, Clone)]
pub enum SamplingMode {
    /// Greedy: always pick highest probability token.
    ArgMax,
    /// Top-K then Top-P with temperature.
    TopKTopP {
        k: usize,
        p: f64,
        temperature: f64,
    },
}

/// Token sampler with RNG state.
pub struct Sampler {
    mode: SamplingMode,
    rng_state: u64,
}

impl Sampler {
    pub fn new(mode: SamplingMode, seed: u64) -> Self {
        Self { mode, rng_state: seed }
    }

    pub fn argmax() -> Self {
        Self::new(SamplingMode::ArgMax, 0)
    }

    /// Sample a token from logits tensor.
    ///
    /// logits: [vocab_size] or [1, vocab_size]
    /// Returns: token id (u32)
    pub fn sample(&mut self, logits: &TqTensor) -> Result<u32> {
        let data = logits.as_slice();
        let vocab_size = *logits.shape().last()
            .ok_or_else(|| TqError::Msg("empty logits".into()))?;

        // Get the last row of logits (for batch=1)
        let logits_row = if data.len() > vocab_size {
            &data[data.len() - vocab_size..]
        } else {
            data
        };

        match &self.mode {
            SamplingMode::ArgMax => {
                let (max_idx, _) = logits_row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                Ok(max_idx as u32)
            }
            SamplingMode::TopKTopP { k, p, temperature } => {
                self.sample_top_k_top_p(logits_row, *k, *p, *temperature)
            }
        }
    }

    fn sample_top_k_top_p(
        &mut self,
        logits: &[f32],
        k: usize,
        p: f64,
        temperature: f64,
    ) -> Result<u32> {
        let n = logits.len();

        // Apply temperature
        let temp = temperature.max(1e-7) as f32;
        let scaled: Vec<f32> = logits.iter().map(|&v| v / temp).collect();

        // Create (index, logit) pairs and sort descending
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Top-K: keep only top k candidates
        let top_k = k.min(n);
        indexed.truncate(top_k);

        // Softmax over candidates
        let max_logit = indexed[0].1;
        let mut probs: Vec<(usize, f32)> = indexed.iter()
            .map(|&(idx, logit)| (idx, (logit - max_logit).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, prob) in probs.iter_mut() {
            *prob /= sum;
        }

        // Top-P (nucleus): keep smallest set with cumulative prob >= p
        let mut cumsum = 0.0f64;
        let mut cutoff = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumsum += prob as f64;
            if cumsum >= p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);

        // Re-normalize after top-p cut
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, prob) in probs.iter_mut() {
            *prob /= sum;
        }

        // Sample from distribution using xorshift64
        let r = self.rand_f32();
        let mut cumsum = 0.0f32;
        for &(idx, prob) in &probs {
            cumsum += prob;
            if r < cumsum {
                return Ok(idx as u32);
            }
        }

        // Fallback to last candidate
        Ok(probs.last().unwrap().0 as u32)
    }

    /// Xorshift64 PRNG — fast, good enough for sampling.
    fn rand_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        // Convert to [0, 1) float
        (self.rng_state >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::TqDevice;

    #[test]
    fn test_argmax() {
        let logits = TqTensor::from_vec(
            vec![0.1, 0.5, 0.3, 0.9, 0.2],
            vec![5],
            &TqDevice::Cpu,
        ).unwrap();

        let mut sampler = Sampler::argmax();
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 3); // index of 0.9
    }

    #[test]
    fn test_top_k_deterministic() {
        let logits = TqTensor::from_vec(
            vec![10.0, 0.0, 0.0, 0.0, 0.0], // overwhelming first token
            vec![5],
            &TqDevice::Cpu,
        ).unwrap();

        let mut sampler = Sampler::new(
            SamplingMode::TopKTopP { k: 40, p: 0.9, temperature: 0.1 },
            42,
        );
        // With very low temperature and one dominant logit, should always pick 0
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sampling_distribution() {
        // With equal logits, sampling should produce diverse tokens
        let logits = TqTensor::from_vec(
            vec![1.0; 100],
            vec![100],
            &TqDevice::Cpu,
        ).unwrap();

        let mut sampler = Sampler::new(
            SamplingMode::TopKTopP { k: 100, p: 1.0, temperature: 1.0 },
            12345,
        );

        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            seen.insert(sampler.sample(&logits).unwrap());
        }
        // With uniform distribution and 50 samples from 100 tokens, should see variety
        assert!(seen.len() > 5);
    }

    #[test]
    fn test_argmax_tie_breaking() {
        // Two equal max values — argmax should pick the first one
        let logits = TqTensor::from_vec(
            vec![0.1, 0.9, 0.5, 0.9, 0.2],
            vec![5],
            &TqDevice::Cpu,
        ).unwrap();

        let mut sampler = Sampler::argmax();
        let token = sampler.sample(&logits).unwrap();
        // Rust's max_by picks the last max with partial_cmp, but the iterator
        // returns the element with the greatest value; with ties the last wins
        assert!(token == 1 || token == 3); // either tie-break is acceptable
    }

    #[test]
    fn test_temperature_effect() {
        // With very high temperature, distribution should be nearly uniform
        // even with non-uniform logits
        let logits = TqTensor::from_vec(
            vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![10],
            &TqDevice::Cpu,
        ).unwrap();

        // Low temperature: should almost always pick token 0
        let mut low_temp = Sampler::new(
            SamplingMode::TopKTopP { k: 10, p: 1.0, temperature: 0.01 },
            42,
        );
        let mut token0_count = 0;
        for _ in 0..20 {
            if low_temp.sample(&logits).unwrap() == 0 {
                token0_count += 1;
            }
        }
        assert!(token0_count >= 18, "low temp should almost always pick dominant token");

        // High temperature: should see more diversity
        let mut high_temp = Sampler::new(
            SamplingMode::TopKTopP { k: 10, p: 1.0, temperature: 100.0 },
            42,
        );
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            seen.insert(high_temp.sample(&logits).unwrap());
        }
        assert!(seen.len() >= 3, "high temp should produce diverse tokens, got {}", seen.len());
    }
}
