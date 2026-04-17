//! Sprint 3 Day 1 — EAGLE draft acceptance-rate probe.
//!
//! Runs a fixed prompt through the target (greedy decode) and at each step
//! asks the draft what IT would have predicted. Reports the acceptance rate.
//!
//! # Why this exists
//!
//! Before investing in target-side tree attention + KV rollback (Sprint 3
//! days 2+), we want a cheap signal that the draft actually agrees with the
//! target on a reasonable fraction of tokens. Industry EAGLE-3 implementations
//! report ~85% acceptance on in-distribution prompts; anything < 30% is a
//! draft-load or projection bug worth debugging first.
//!
//! # Probe loop per step `i` at position `pos`
//!
//! ```text
//!   1. target_hidden  = target.forward_last_hidden(next_input, pos)   // [hidden]
//!   2. target_logits  = target.project_hidden_to_logits(target_hidden)
//!   3. target_token   = argmax(target_logits)
//!   4. if i > 0:
//!        draft_hidden = draft.draft_forward(prev_hidden, prev_token, pos - prompt_len)
//!        draft_logits = target.project_hidden_to_logits(tensor(draft_hidden))
//!        draft_token  = argmax(draft_logits)
//!        accepted[i] = (draft_token == target_token)
//!   5. prev_hidden, prev_token = target_hidden, target_token
//! ```
//!
//! The input at step 0 is the full prompt (prefill); subsequent steps feed
//! just the single sampled token (single-token decode). `pos` advances by
//! prompt_len on step 0 and by 1 thereafter — same convention as the normal
//! decode loop.
//!
//! # Known approximation
//!
//! This mirrors EAGLE's online decode closely but not identically. EAGLE's
//! real verify step feeds the draft the hidden state of the TARGET's
//! previous forward, not the draft's own output. We do the same (target
//! hidden in, draft output never feeds back), so the probe reflects the
//! quality the real verify loop will see.

#![cfg(feature = "cuda")]

use crate::cuda::TqTensor as Tensor;
use crate::engine::Engine;
use super::{EagleDraft, EagleError};

/// Outcome of a single probe step. `None` for step 0 where no draft
/// prediction exists yet (no prev hidden).
#[derive(Debug)]
pub struct ProbeStep {
    pub position: usize,
    pub target_token: u32,
    pub draft_token: Option<u32>,
}

impl ProbeStep {
    pub fn is_accepted(&self) -> Option<bool> {
        self.draft_token.map(|d| d == self.target_token)
    }
}

/// End-to-end acceptance-rate probe. Consumes the engine + draft (neither
/// can be safely reused because the target's KV cache is advanced).
pub fn run_acceptance_probe(
    engine: &mut Engine,
    draft: &mut EagleDraft,
    prompt: &str,
    n_steps: usize,
) -> Result<Vec<ProbeStep>, EagleError> {
    // Tokenize.
    let enc = engine.tokenizer.encode(prompt, true)
        .map_err(|e| EagleError::Msg(format!("tokenize: {:?}", e)))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(EagleError::Msg("empty prompt after tokenize".into()));
    }
    eprintln!("[probe] prompt = {} tokens", prompt_ids.len());

    // Initialise draft runtime once (idempotent).
    draft.init_runtime()?;

    let eos = engine.eos_token_ids.clone();
    let model = &mut engine.model;

    let mut steps: Vec<ProbeStep> = Vec::with_capacity(n_steps);
    let mut pos: usize = 0;

    // Previous-step hidden + token; None until after step 0 completes.
    let mut prev_hidden_host: Option<Vec<f32>> = None;
    let mut prev_token: Option<u32> = None;

    for i in 0..n_steps {
        // Build the next input tensor.
        let input_ids: Vec<u32> = if i == 0 { prompt_ids.clone() } else {
            vec![prev_token.expect("prev_token set after step 0")]
        };
        let seq_len = input_ids.len();
        let input_i64: Vec<i64> = input_ids.iter().map(|&t| t as i64).collect();
        let x = Tensor::from_vec(
            input_i64.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            vec![1, seq_len],
            &crate::cuda::TqDevice::Cpu,
        ).map_err(|e| EagleError::Msg(format!("build input: {}", e)))?;

        // ── Target forward — capture hidden (pre-norm/pre-lm-head) ──
        let tgt_hidden = model.forward_last_hidden(&x, pos)
            .map_err(|e| EagleError::Msg(format!("target forward_last_hidden: {}", e)))?;
        let tgt_logits = model.project_hidden_to_logits(&tgt_hidden)
            .map_err(|e| EagleError::Msg(format!("target project: {}", e)))?;
        let tgt_logits_host = tgt_logits.to_vec1()
            .map_err(|e| EagleError::Msg(format!("target logits to_vec: {}", e)))?;
        let tgt_token = argmax(&tgt_logits_host);

        // Extract target hidden as host Vec<f32> for the next step's draft call.
        let tgt_hidden_host: Vec<f32> = tgt_hidden.to_vec1()
            .map_err(|e| EagleError::Msg(format!("target hidden to_vec: {}", e)))?;

        // ── Draft forward (only if we have a prev step) ──
        let draft_token = if let (Some(prev_h), Some(prev_t)) = (prev_hidden_host.as_ref(), prev_token) {
            // Draft position is the within-generation index (0..n_steps-1).
            let draft_pos = i - 1;  // for the transition between step i-1 and i
            let d_hidden = draft.draft_forward(prev_h, prev_t, draft_pos)?;
            // Wrap draft hidden as a tensor and project through target's norm + lm_head.
            let d_tensor = Tensor::from_vec(d_hidden, vec![draft.config.hidden_size],
                                            &crate::cuda::TqDevice::Cpu)
                .map_err(|e| EagleError::Msg(format!("wrap draft hidden: {}", e)))?;
            let d_logits = model.project_hidden_to_logits(&d_tensor)
                .map_err(|e| EagleError::Msg(format!("draft project: {}", e)))?;
            let d_logits_host = d_logits.to_vec1()
                .map_err(|e| EagleError::Msg(format!("draft logits to_vec: {}", e)))?;
            Some(argmax(&d_logits_host))
        } else { None };

        steps.push(ProbeStep {
            position: pos + seq_len - 1,
            target_token: tgt_token,
            draft_token,
        });

        // Advance.
        pos += seq_len;
        prev_hidden_host = Some(tgt_hidden_host);
        prev_token = Some(tgt_token);

        // Early stop on EOS.
        if eos.contains(&tgt_token) {
            eprintln!("[probe] EOS at step {}, stopping early", i);
            break;
        }
    }

    Ok(steps)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v { best_v = x; best_i = i; }
    }
    best_i as u32
}

/// Human-friendly summary over a probe run.
pub fn summarise(steps: &[ProbeStep]) {
    let (mut compared, mut accepted) = (0usize, 0usize);
    for s in steps {
        if let Some(a) = s.is_accepted() {
            compared += 1;
            if a { accepted += 1; }
        }
    }
    let rate = if compared > 0 { 100.0 * accepted as f32 / compared as f32 } else { 0.0 };
    println!();
    println!("Acceptance-rate probe:");
    println!("  steps compared : {}", compared);
    println!("  draft == target: {} ({:.1}%)", accepted, rate);
    if compared > 0 {
        println!();
        println!("  step  pos   target   draft    match");
        for (idx, s) in steps.iter().enumerate() {
            let (dt, mk) = match s.draft_token {
                Some(d) => (format!("{:6}", d), if d == s.target_token { "YES" } else { "   " }),
                None    => (String::from("   -  "), "   "),
            };
            println!("  {:4}  {:3}   {:6}   {}   {}", idx, s.position, s.target_token, dt, mk);
        }
    }
    if compared >= 10 {
        let range = if rate >= 70.0 { "healthy - proceed to tree decode"
        } else if rate >= 30.0 { "plausible - spec-decode will help but margins thinner"
        } else if rate >= 10.0 { "low - check draft weights / projection path"
        } else { "failing - likely correctness bug (wrong layer hidden, bad load, bad projection)"
        };
        println!();
        println!("  verdict: {}", range);
    }
}
