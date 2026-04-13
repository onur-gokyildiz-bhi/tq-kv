//! `tq perplexity` -- perplexity evaluation (with optional 3-way comparison).

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::cli::resolve_tq_config_for_model;
use crate::engine::Engine;
use crate::{catalog, config, hub, model, models, Cli, Commands};

pub(crate) fn cmd_perplexity(cli: &Cli) -> Result<()> {
    let (model_name, model_path_override, tokenizer_repo_override, file, turbo_quant, compare) =
        match &cli.command {
            Some(Commands::Perplexity {
                model, model_path, tokenizer_repo, file, turbo_quant, compare, ..
            }) => (
                model.as_str(), model_path.as_deref(), tokenizer_repo.as_deref(),
                file, *turbo_quant, *compare,
            ),
            _ => unreachable!(),
        };

    let text = std::fs::read_to_string(file)
        .with_context(|| format!("Cannot read perplexity file: {}", file.display()))?;

    // Helper: load engine with given TQ config
    let load_engine = |tq_config: Option<tq_kv::TurboQuantConfig>| -> Result<Engine> {
        if let Some(model_config) = config::get_model(model_name) {
            model::load_engine(
                model_config, model_path_override,
                tq_config, tokenizer_repo_override, cli.cpu,
            )
        } else if model_path_override.is_none() {
            let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
            let tok_path = match (tokenizer_repo_override, &tokenizer_path) {
                (Some(repo), _) => {
                    let local = std::path::Path::new(repo).join("tokenizer.json");
                    if local.exists() {
                        local
                    } else if std::path::Path::new(repo).exists() && repo.ends_with(".json") {
                        PathBuf::from(repo)
                    } else {
                        let api = hf_hub::api::sync::Api::new()?;
                        api.model(repo.to_string()).get("tokenizer.json")?
                    }
                }
                (None, Some(tp)) => tp.clone(),
                (None, &None) => anyhow::bail!(
                    "No tokenizer found for model. Use --tokenizer-repo to specify one."
                ),
            };
            let mf = gguf_path.to_string_lossy().to_string();
            let arch = config::detect_arch(&mf);
            Engine::load_with_device(&gguf_path, &tok_path, arch, tq_config, cli.cpu)
        } else {
            anyhow::bail!(
                "Unknown model: '{}'. Use `tq list --available` to see available models.",
                model_name
            )
        }
    };

    if compare {
        // ── 3-way comparison: Standard vs TQ vs TQ+TriAttention ──
        let display_name = catalog::find(model_name)
            .map(|e| e.display.to_string())
            .unwrap_or_else(|| model_name.to_string());

        // 1. Standard
        eprintln!("[1/3] Perplexity: Standard...");
        models::turbo_generic::set_triattention_override(false);
        let mut engine_std = load_engine(None)?;
        let ppl_std = engine_std.compute_perplexity(&text)?;
        drop(engine_std);
        eprintln!("  Standard PPL: {:.3}", ppl_std);

        // 2. TQ 4-bit (no TriAttention)
        eprintln!("[2/3] Perplexity: TQ 4-bit...");
        models::turbo_generic::set_triattention_override(false);
        let tq_config_shared = resolve_tq_config_for_model(true, cli.tq_bits, Some(model_name));
        // resolve_tq_config enables TriAttn internally — force off for TQ-only
        models::turbo_generic::set_triattention_override(false);
        let mut engine_tq = load_engine(tq_config_shared.clone())?;
        let ppl_tq = engine_tq.compute_perplexity(&text)?;
        drop(engine_tq);
        eprintln!("  TQ 4-bit PPL: {:.3}", ppl_tq);

        // 3. TQ + TriAttention (same config, just enable TriAttn)
        eprintln!("[3/3] Perplexity: TQ+TriAttention...");
        models::turbo_generic::set_triattention_override(true);
        let mut engine_tri = load_engine(tq_config_shared)?;
        let ppl_tri = engine_tri.compute_perplexity(&text)?;
        drop(engine_tri);
        eprintln!("  TQ+TriAttn PPL: {:.3}", ppl_tri);

        // Restore
        models::turbo_generic::clear_triattention_override();

        // Results table
        let tq_delta = (ppl_tq / ppl_std - 1.0) * 100.0;
        let tri_delta = (ppl_tri / ppl_std - 1.0) * 100.0;
        println!();
        println!("+==============================================================+");
        println!("|  Perplexity Comparison -- {:<33}|", display_name);
        println!("+==============================================================+");
        println!("| {:<18} | {:<10} | {:<10} | {:<10} |", "Mode", "PPL", "vs Std", "Quality");
        println!("+--------------------+------------+------------+------------+");
        println!("| {:<18} | {:<10.3} | {:<10} | {:<10} |",
            "Standard", ppl_std, "baseline", "100%");
        println!("| {:<18} | {:<10.3} | {:>+9.1}% | {:<10} |",
            "TQ 4-bit", ppl_tq, tq_delta,
            if tq_delta.abs() < 5.0 { "excellent" } else if tq_delta < 15.0 { "good" } else { "degraded" });
        println!("| {:<18} | {:<10.3} | {:>+9.1}% | {:<10} |",
            "TQ+TriAttention", ppl_tri, tri_delta,
            if tri_delta.abs() < 5.0 { "excellent" } else if tri_delta < 15.0 { "good" } else { "degraded" });
        println!("+==============================================================+");
    } else {
        // ── Single run (existing behavior) ──
        let tq_config = resolve_tq_config_for_model(turbo_quant, cli.tq_bits, Some(model_name));
        let mut engine = load_engine(tq_config)?;
        let ppl = engine.compute_perplexity(&text)?;
        println!("Perplexity: {:.3}", ppl);
    }
    Ok(())
}
