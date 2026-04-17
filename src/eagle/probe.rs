//! Sprint 3 Day 1-2 — EAGLE draft acceptance-rate probe.
//!
//! Runs a fixed prompt through the target (greedy decode) and at each step
//! asks the draft what IT would have predicted. Reports the acceptance rate.
//!
//! # Why this exists
//!
//! Before investing in target-side tree attention + KV rollback (Sprint 3
//! days 3+), we want a cheap signal that the draft actually agrees with
//! the target on a reasonable fraction of tokens. Industry EAGLE-3
//! implementations report ~85% acceptance on in-distribution prompts;
//! anything < 30% is a draft-load or projection bug worth debugging first.
//!
//! # Day 2 fix: draft prompt prefill
//!
//! Day 1 ran the probe without warming the draft's KV cache on the prompt,
//! so the draft's attention saw only a single K/V entry when it fired for
//! the first generated token. Acceptance was 0%. Day 2 prefills the draft
//! by running `draft_forward` on every prompt position (except position 0
//! which has no `prev_hidden`).
//!
//! # Probe flow (Day 2 layout)
//!
//! ```text
//!   PREFILL
//!     all_h, gen0_logits = target.forward_hidden_all(prompt)    // target KV +=prompt_len
//!     gen0_token         = argmax(gen0_logits)
//!     for i in 1..prompt_len:
//!         draft.forward(all_h[i-1], prompt[i], draft_pos=i-1)   // draft KV +=prompt_len-1
//!
//!   DECODE LOOP (n_steps)
//!     # Draft predicts next token FROM previous target hidden + sampled token
//!     draft_h    = draft.forward(all_h[i-1], gen[i], draft_pos=prompt_len-1+i)
//!     draft_pred = argmax(project(draft_h))
//!
//!     # Target forward for the actual ground truth
//!     tgt_h      = target.forward_last_hidden([gen[i]], pos=prompt_len+i)
//!     tgt_token  = argmax(project(tgt_h))
//!
//!     accepted[i] = (draft_pred == tgt_token)
//!     all_h[i-1]  ← tgt_h      // feed target hidden into next iteration
//!     gen[i+1]    ← tgt_token
//! ```

#![cfg(feature = "cuda")]

use crate::cuda::TqTensor as Tensor;
use crate::engine::Engine;
use super::{EagleDraft, EagleError};

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

pub fn run_acceptance_probe(
    engine: &mut Engine,
    draft: &mut EagleDraft,
    prompt: &str,
    n_steps: usize,
) -> Result<Vec<ProbeStep>, EagleError> {
    let enc = engine.tokenizer.encode(prompt, true)
        .map_err(|e| EagleError::Msg(format!("tokenize: {:?}", e)))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(EagleError::Msg("empty prompt after tokenize".into()));
    }
    let prompt_len = prompt_ids.len();
    eprintln!("[probe] prompt = {} tokens", prompt_len);

    draft.init_runtime()?;

    let eos = engine.eos_token_ids.clone();
    let model = &mut engine.model;
    let hidden_dim = draft.config.hidden_size;

    // ── PREFILL: target processes full prompt, capture ALL positions' hidden ──
    let prompt_i64: Vec<f32> = prompt_ids.iter().map(|&t| t as f32).collect();
    let x_prompt = Tensor::from_vec(prompt_i64, vec![1, prompt_len], &crate::cuda::TqDevice::Cpu)
        .map_err(|e| EagleError::Msg(format!("build prompt input: {}", e)))?;
    let all_h = model.forward_hidden_all(&x_prompt, 0)
        .map_err(|e| EagleError::Msg(format!("target forward_hidden_all: {}", e)))?;
    // all_h shape: [prompt_len, hidden]. Flatten + pull to host so we can slice.
    let all_h_flat = all_h.reshape(vec![prompt_len * hidden_dim])
        .map_err(|e| EagleError::Msg(format!("reshape all_h: {}", e)))?;
    let all_h_host: Vec<f32> = all_h_flat.to_vec1()
        .map_err(|e| EagleError::Msg(format!("all_h to_vec: {}", e)))?;

    // Project last-position hidden → first generated token.
    let last_h_slice: Vec<f32> = all_h_host[(prompt_len - 1) * hidden_dim ..].to_vec();
    let last_h_tensor = Tensor::from_vec(last_h_slice.clone(), vec![hidden_dim],
                                          &crate::cuda::TqDevice::Cpu)
        .map_err(|e| EagleError::Msg(format!("wrap last_h: {}", e)))?;
    let first_logits = model.project_hidden_to_logits(&last_h_tensor)
        .map_err(|e| EagleError::Msg(format!("project first logits: {}", e)))?;
    let first_logits_host = first_logits.to_vec1()
        .map_err(|e| EagleError::Msg(format!("first logits to_vec: {}", e)))?;
    let first_gen = argmax(&first_logits_host);
    eprintln!("[probe] first generated token = {}", first_gen);

    // (Helper moved out of scope — see `apply_base_norm` near the bottom of
    // this file. A closure would capture `&model` and conflict with the
    // later `&mut model` calls to forward_last_hidden.)

    // ── PREFILL draft's KV cache with prompt positions 1..prompt_len ──
    // At position p >= 1, draft sees (norm(hidden[p-1]), prompt[p]) → predicts prompt[p+1].
    // Apply base's final RMSNorm to each hidden before handing to draft (see `normed` doc).
    eprintln!("[probe] prefilling draft with {} prompt positions ...", prompt_len - 1);
    for p in 1..prompt_len {
        let prev_h = &all_h_host[(p - 1) * hidden_dim .. p * hidden_dim];
        let prev_h_n = apply_base_norm(model, prev_h)?;
        let _ = draft.draft_forward(&prev_h_n, prompt_ids[p], p - 1)?;
    }

    // ── DECODE LOOP — probe each step ──
    let mut steps: Vec<ProbeStep> = Vec::with_capacity(n_steps);

    // Rolling state:
    //   prev_hidden_host: target hidden at position (prompt_len - 1 + i) — i.e., the
    //                     hidden that PRODUCED the currently-pending token
    //   pending_token   : the already-sampled token the draft must consume NEXT
    //   target_pos      : the absolute target position where the NEXT target
    //                     forward will write its output (≥ prompt_len)
    let mut prev_hidden_host: Vec<f32> = apply_base_norm(model, &last_h_slice)?;
    let mut pending_token: u32 = first_gen;
    let mut target_pos: usize = prompt_len;
    let mut draft_pos: usize = prompt_len - 1;

    for i in 0..n_steps {
        // ── Draft predicts NEXT token (i.e., what target will sample at target_pos+1) ──
        let d_hidden_host = draft.draft_forward(&prev_hidden_host, pending_token, draft_pos)?;
        let d_tensor = Tensor::from_vec(d_hidden_host, vec![hidden_dim],
                                        &crate::cuda::TqDevice::Cpu)
            .map_err(|e| EagleError::Msg(format!("wrap draft hidden: {}", e)))?;
        // Day 2 debug: try passing draft output through base's final norm +
        // lm_head (same projection target uses). If acceptance jumps from 0%
        // we've confirmed the draft is in pre-norm residual space.
        let d_logits = model.project_hidden_to_logits(&d_tensor)
            .map_err(|e| EagleError::Msg(format!("project draft: {}", e)))?;
        let d_logits_host = d_logits.to_vec1()
            .map_err(|e| EagleError::Msg(format!("draft logits to_vec: {}", e)))?;
        let draft_pred = argmax(&d_logits_host);

        // ── Target forward for ground truth at target_pos ──
        let x_token = Tensor::from_vec(vec![pending_token as f32], vec![1, 1],
                                       &crate::cuda::TqDevice::Cpu)
            .map_err(|e| EagleError::Msg(format!("build token input: {}", e)))?;
        let tgt_h = model.forward_last_hidden(&x_token, target_pos)
            .map_err(|e| EagleError::Msg(format!("target forward: {}", e)))?;
        let tgt_h_host: Vec<f32> = tgt_h.to_vec1()
            .map_err(|e| EagleError::Msg(format!("target hidden to_vec: {}", e)))?;
        let tgt_logits = model.project_hidden_to_logits(&tgt_h)
            .map_err(|e| EagleError::Msg(format!("project target: {}", e)))?;
        let tgt_logits_host = tgt_logits.to_vec1()
            .map_err(|e| EagleError::Msg(format!("target logits to_vec: {}", e)))?;
        let tgt_next_token = argmax(&tgt_logits_host);

        steps.push(ProbeStep {
            position: target_pos + 1,  // position of the token being predicted
            target_token: tgt_next_token,
            draft_token: Some(draft_pred),
        });

        if eos.contains(&tgt_next_token) {
            eprintln!("[probe] EOS at step {}, stopping early", i);
            break;
        }

        // Advance state. The target's hidden at position `target_pos` is now
        // prev_hidden for the NEXT draft call — after applying base's final norm.
        prev_hidden_host = apply_base_norm(model, &tgt_h_host)?;
        pending_token = tgt_next_token;
        target_pos += 1;
        draft_pos += 1;
    }

    Ok(steps)
}

/// Apply base model's final RMSNorm to a host vec `[hidden]`. EAGLE's draft
/// was trained to consume NORMALISED hidden states, not raw residual stream —
/// without this, `fc_out` overshoots by `sqrt(hidden) * sigma_W` ≈ 10×.
fn apply_base_norm(model: &crate::engine::ModelWeights, raw: &[f32])
    -> Result<Vec<f32>, EagleError>
{
    let t = Tensor::from_vec(raw.to_vec(), vec![raw.len()],
                             &crate::cuda::TqDevice::Cpu)
        .map_err(|e| EagleError::Msg(format!("wrap for norm: {}", e)))?;
    let n = model.apply_final_norm(&t)
        .map_err(|e| EagleError::Msg(format!("apply_final_norm: {}", e)))?;
    n.to_vec1().map_err(|e| EagleError::Msg(format!("norm to_vec: {}", e)))
}

fn argmax(v: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v { best_v = x; best_i = i; }
    }
    best_i as u32
}

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
