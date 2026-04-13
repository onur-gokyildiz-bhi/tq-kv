//! `tq ablate` -- sweep TQ configs and measure PPL for each.

use anyhow::{Context, Result};
use std::time::Instant;

use crate::engine::Engine;
use crate::{calibrate, config, hub, Cli, Commands};

pub(crate) fn cmd_ablate_study(cli: &Cli) -> Result<()> {
    let (model_name, file_path, quick, output_path) = match &cli.command {
        Some(Commands::Ablate { model, file, quick, output }) => {
            (model.as_str(), file.clone(), *quick, output.clone())
        }
        _ => unreachable!(),
    };

    let eval_text = std::fs::read_to_string(&file_path)
        .with_context(|| format!("Cannot read eval file: {}", file_path.display()))?;

    eprintln!("=== TurboQuant Ablation Study ===");
    eprintln!("Model: {}", model_name);
    eprintln!("Eval text: {} chars", eval_text.len());
    eprintln!("Mode: {}", if quick { "quick" } else { "full" });

    let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
    let tok_path = tokenizer_path.ok_or_else(|| anyhow::anyhow!("No tokenizer found"))?;
    let mf = gguf_path.to_string_lossy().to_string();
    let arch = config::detect_arch(&mf);

    // Check if calibration exists
    let has_calibration = calibrate::load_calibration_for_model(model_name).is_some();

    // Define sweep
    let bits_sweep: Vec<u8> = if quick { vec![2, 4] } else { vec![2, 3, 4] };
    let skip_sweep: Vec<usize> = if quick { vec![0, 4] } else { vec![0, 2, 4, 8] };
    let sink_sweep: Vec<usize> = if quick { vec![0, 4] } else { vec![0, 2, 4] };
    let cal_sweep: Vec<bool> = if has_calibration { vec![false, true] } else { vec![false] };

    let mut results: Vec<(String, f64, f64)> = Vec::new(); // (config_label, ppl, seconds)

    // Baseline: no compression (GenericTurboModel handles GPU/CPU transparently)
    {
        eprintln!("\n--- Baseline (no compression) ---");
        let start = Instant::now();
        let mut engine = Engine::load_with_device(&gguf_path, &tok_path, arch, None, cli.cpu)?;
        let ppl = engine.compute_perplexity(&eval_text)?;
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("  PPL: {:.3} ({:.1}s)", ppl, elapsed);
        results.push(("baseline".to_string(), ppl, elapsed));
    }

    let baseline_ppl = results[0].1;

    // Sweep TQ configs
    let total = bits_sweep.len() * skip_sweep.len() * sink_sweep.len() * cal_sweep.len();
    let mut run = 0;

    for &bits in &bits_sweep {
        for &skip in &skip_sweep {
            for &sink in &sink_sweep {
                for &use_cal in &cal_sweep {
                    run += 1;
                    let label = format!("{}bit_skip{}_sink{}{}", bits, skip, sink,
                        if use_cal { "_cal" } else { "" });
                    eprintln!("\n--- [{}/{}] {} ---", run, total, label);

                    let mut tq = match bits {
                        2 => tq_kv::TurboQuantConfig::extreme(),
                        3 => tq_kv::TurboQuantConfig::aggressive(),
                        _ => tq_kv::TurboQuantConfig::balanced(),
                    };
                    tq.skip_layers = Some(skip);
                    tq.sink_tokens = Some(sink);

                    if use_cal {
                        if let Some(cal_data) = calibrate::load_calibration_for_model(model_name) {
                            cal_data.apply_to_config(&mut tq);
                        }
                    }

                    let start = Instant::now();
                    let mut engine = Engine::load_with_device(
                        &gguf_path, &tok_path, arch, Some(tq), cli.cpu,
                    )?;
                    let ppl = engine.compute_perplexity(&eval_text)?;
                    let elapsed = start.elapsed().as_secs_f64();
                    let delta = (ppl - baseline_ppl) / baseline_ppl * 100.0;
                    eprintln!("  PPL: {:.3} (delta: {:+.2}%, {:.1}s)", ppl, delta, elapsed);
                    results.push((label, ppl, elapsed));
                }
            }
        }
    }

    // Print summary table
    eprintln!("\n╔══════════════════════════════════════════════════════╗");
    eprintln!("║              ABLATION RESULTS                       ║");
    eprintln!("╠══════════════════════╦════════╦═══════════╦══════════╣");
    eprintln!("║ Config               ║  PPL   ║ Delta PPL ║ Time (s) ║");
    eprintln!("╠══════════════════════╬════════╬═══════════╬══════════╣");
    for (label, ppl, secs) in &results {
        let delta = if label == "baseline" {
            "   ---   ".to_string()
        } else {
            format!("{:+.2}%", (ppl - baseline_ppl) / baseline_ppl * 100.0)
        };
        eprintln!("║ {:20} ║ {:6.3} ║ {:>9} ║ {:7.1}s ║", label, ppl, delta, secs);
    }
    eprintln!("╚══════════════════════╩════════╩═══════════╩══════════╝");

    // Write CSV if requested
    if let Some(ref csv_path) = output_path {
        let mut csv = String::from("config,ppl,delta_ppl_pct,time_s\n");
        for (label, ppl, secs) in &results {
            let delta = if label == "baseline" { 0.0 } else { (ppl - baseline_ppl) / baseline_ppl * 100.0 };
            csv.push_str(&format!("{},{:.4},{:.4},{:.2}\n", label, ppl, delta, secs));
        }
        std::fs::write(csv_path, &csv)?;
        eprintln!("\nCSV saved: {}", csv_path.display());
    }

    Ok(())
}
