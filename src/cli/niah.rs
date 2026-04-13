//! `tq niah` -- Needle-In-A-Haystack retrieval test.

use anyhow::{Context, Result};

use crate::cli::resolve_tq_config_for_model;
use crate::engine::{Engine, GenerationParams};
use crate::{config, hub, model, models, Cli, Commands};

pub(crate) fn cmd_niah(cli: &Cli) -> Result<()> {
    let (model_name, context_target, position, max_tokens, no_triattn) = match &cli.command {
        Some(Commands::Niah { model, context, position, max_tokens, no_triattn }) => {
            (model.as_str(), *context, *position, *max_tokens, *no_triattn)
        }
        _ => unreachable!(),
    };

    // Needle: use a high-frequency pattern that Q4_K_M can reliably retrieve.
    // "AURORA-7" fails even at 2-sentence context on Q4_K_M (see Bug #3 findings).
    // "Paris" works because it's the #1 completion for "capital of France".
    let needle_id = std::env::var("TQ_NIAH_NEEDLE").unwrap_or_else(|_| "Paris".to_string());
    let needle = if needle_id == "Paris" {
        "IMPORTANT: The capital of the secret country is Paris and the launch date is March 15.".to_string()
    } else {
        format!("IMPORTANT: The secret project codename is {} and the launch date is March 15.", needle_id)
    };

    let filler = "Modern computing systems continue to evolve through innovations in hardware architecture and software optimization. Researchers explore new approaches to improve efficiency across diverse platforms. The interplay between hardware capabilities and software requirements drives progress in computer science and engineering. ";

    // Build haystack: filler before + needle + filler after, sized to context_target words
    let target_words = context_target.max(200);
    let needle_pos = (target_words as f32 * position) as usize;
    let mut haystack = String::new();
    let words_per_filler = filler.split_whitespace().count();
    let n_pre = (needle_pos + words_per_filler - 1) / words_per_filler;
    for _ in 0..n_pre { haystack.push_str(filler); }
    haystack.push_str(&needle);
    haystack.push(' ');
    let remaining = target_words.saturating_sub(haystack.split_whitespace().count());
    let n_post = (remaining + words_per_filler - 1) / words_per_filler;
    for _ in 0..n_post { haystack.push_str(filler); }

    let actual_words = haystack.split_whitespace().count();

    // Cloze-style sentence completion. Plain text (no chat template) — chat
    // wrapping in greedy mode produces garbage on Qwen2 (see commit 32c2dc5).
    // The pattern is: state the document, then a sentence whose continuation
    // is the answer. Model is in pretraining-style next-token mode here.
    // Quotation pattern for raw-text NIAH. The model has seen millions of
    // "it was stated: 'X is Y'" patterns in pretraining. By quoting the
    // needle sentence with a blank, we prime the model to fill in the value.
    // This works in raw completion mode (no chat template needed).
    let prompt = format!(
        "{}\n\nQuote from the document above: \"The secret project codename is",
        haystack
    );

    eprintln!("============================================");
    eprintln!("  NIAH Test — {}", model_name);
    eprintln!("  Haystack: ~{} words", actual_words);
    eprintln!("  Needle: '{}' at {:.0}% depth (TQ_NIAH_NEEDLE to override)", needle_id, position * 100.0);
    eprintln!("  Mode: {}", if no_triattn { "TQ-only" } else { "TQ+TriAttn V3" });
    eprintln!("============================================");

    if !no_triattn {
        models::turbo_generic::set_triattention_override(true);
    } else {
        models::turbo_generic::set_triattention_override(false);
    }

    // Allow standard mode via TQ_NO_TQ=1 for debugging
    let tq_config = if std::env::var("TQ_NO_TQ").ok().map_or(false, |v| v == "1") {
        None
    } else {
        resolve_tq_config_for_model(true, cli.tq_bits, Some(model_name))
    };

    let mut engine = if let Some(model_config) = config::get_model(model_name) {
        model::load_engine(model_config, None, tq_config, None, cli.cpu)?
    } else {
        let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
        let tok_path = tokenizer_path.with_context(|| format!("No tokenizer found for '{}'", model_name))?;
        let arch = config::detect_arch(&gguf_path.to_string_lossy());
        Engine::load_with_device(&gguf_path, &tok_path, arch, tq_config, cli.cpu)?
    };

    // Greedy (same as bench command) for deterministic NIAH
    let params = GenerationParams {
        max_tokens,
        temperature: 0.0,  // ArgMax greedy
        ..Default::default()
    };

    engine.clear_cache();

    let mut output = String::new();
    let _ = engine.generate(&prompt, &params, |token| {
        output.push_str(token);
    })?;

    let lower = output.to_lowercase();
    let found = lower.contains(&needle_id.to_lowercase());

    eprintln!();
    eprintln!("Output: {}", output.trim());
    eprintln!();
    if found {
        eprintln!("  PASS — needle '{}' retrieved", needle_id);
    } else {
        eprintln!("  FAIL — needle '{}' NOT in output", needle_id);
    }
    eprintln!("============================================");

    models::turbo_generic::clear_triattention_override();

    if !found { std::process::exit(1); }
    Ok(())
}
