//! `tq calibrate` -- manual TurboQuant calibration pass.

use anyhow::{Context, Result};

use crate::engine::{self, Engine};
use crate::{calibrate, config, hub, Cli, Commands};

pub(crate) fn cmd_calibrate(cli: &Cli) -> Result<()> {
    let (model_name, text_file, max_samples) = match &cli.command {
        Some(Commands::Calibrate { model, text, samples }) => {
            (model.as_str(), text.clone(), *samples)
        }
        _ => unreachable!(),
    };

    eprintln!("=== TurboQuant Calibration ===");
    eprintln!("Model: {}", model_name);
    eprintln!("Samples: {}", max_samples);

    // Resolve model
    let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
    let tok_path = tokenizer_path.ok_or_else(|| {
        anyhow::anyhow!("No tokenizer found for model '{}'", model_name)
    })?;

    let mf = gguf_path.to_string_lossy().to_string();
    let arch = config::detect_arch(&mf);

    // Load calibration text
    let cal_text = if let Some(ref path) = text_file {
        std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read calibration text: {}", path.display()))?
    } else {
        // Default calibration text (short representative sample)
        "The tower is 324 metres tall, about the same height as an 81-storey building, \
         and the tallest structure in Paris. Its base is square, measuring 125 metres on \
         each side. During its construction, the Eiffel Tower surpassed the Washington \
         Monument to become the tallest human-made structure in the world, a title it held \
         for 41 years until the Chrysler Building in New York City was finished in 1930. \
         It was the first structure in the world to surpass both the 200-metre and 300-metre \
         mark in height. Due to the addition of a broadcasting aerial at the top of the tower \
         in 1957, it is now taller than the Chrysler Building by 5.2 metres. Excluding \
         transmitters, the Eiffel Tower is the second tallest free-standing structure in France \
         after the Millau Viaduct. The tower has three levels for visitors, with restaurants on \
         the first and second levels. The top level observation deck is 276 m above the ground, \
         the highest observation deck accessible to the public in the European Union. Tickets \
         can be purchased to ascend by stairs or lift to the first and second levels. The climb \
         from ground level to the first level is over 300 steps, as is the climb from the first \
         level to the second, making the entire ascent a physical challenge. Although there is a \
         staircase to the top level, it is usually accessible only by lift. The tower was designed \
         by the French civil engineer Gustave Eiffel and built by his engineering company. It was \
         constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. Although \
         initially criticised by some of France's leading artists and intellectuals for its design, \
         it has since become a global cultural icon of France and one of the most recognisable \
         structures in the world. The Eiffel Tower is the most visited paid monument in the world; \
         6.91 million people ascended it in 2015. It has been named after its designer, engineer \
         Gustave Eiffel, and is colloquially known as the Iron Lady.".to_string()
    };

    // Determine head_dim from GGUF metadata
    let head_dim = {
        let mut file = std::fs::File::open(&gguf_path)?;
        let content = crate::gguf::GgufContent::read(&mut file)
            .map_err(|e| anyhow::anyhow!("GGUF read error: {}", e))?;
        let n_embd = content.get("llama.embedding_length")
            .or_else(|| content.get("qwen2.embedding_length"))
            .or_else(|| content.get("phi3.embedding_length"))
            .or_else(|| content.get("gemma.embedding_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;
        let n_head = content.get("llama.attention.head_count")
            .or_else(|| content.get("qwen2.attention.head_count"))
            .or_else(|| content.get("phi3.attention.head_count"))
            .or_else(|| content.get("gemma.attention.head_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32) as usize;
        n_embd / n_head
    };

    eprintln!("Head dim: {}", head_dim);

    // Initialize calibration collector
    let collector = calibrate::init_collector(head_dim, max_samples);

    // Load engine WITHOUT TurboQuant (collect raw activations)
    // We use TQ with skip=999 to run through turbo_generic path but skip compression
    let tq_config = tq_kv::TurboQuantConfig {
        bits: 4,
        ..Default::default()
    };

    // Set TQ_SKIP very high so all layers are uncompressed but still go through
    // the turbo_generic code path (which has our collection hook)
    std::env::set_var("TQ_SKIP", "999");

    let mut engine = Engine::load_with_device(
        &gguf_path, &tok_path, arch, Some(tq_config.clone()), cli.cpu,
    )?;

    // Run prefill to collect activations
    eprintln!("Running prefill to collect KV activations...");
    let params = engine::GenerationParams {
        max_tokens: 1, // just prefill, don't generate
        temperature: 0.0,
        ..Default::default()
    };
    let _ = engine.generate_silent(&cal_text, &params);

    // Remove the skip override
    std::env::remove_var("TQ_SKIP");

    // Get collected samples
    let c = collector.lock().map_err(|e| anyhow::anyhow!("Collector lock error: {}", e))?;
    eprintln!("Collected {} key vectors", c.count);

    if c.count < 100 {
        anyhow::bail!("Too few samples collected ({}). Need at least 100. Try longer calibration text.", c.count);
    }

    // Compute calibration
    let cal_data = calibrate::compute_calibration(&c, model_name, tq_config.rotation_seed);

    // Save
    let entry = crate::catalog::find(model_name);
    let (name, tag) = if let Some(e) = entry {
        (e.name.to_string(), e.tag.to_string())
    } else if let Some((n, t)) = model_name.split_once(':') {
        (n.to_string(), t.to_string())
    } else {
        (model_name.to_string(), "default".to_string())
    };

    let path = calibrate::calibration_path(&name, &tag);
    calibrate::save_calibration(&cal_data, &path)?;

    eprintln!("\n=== Calibration Complete ===");
    eprintln!("Channel scales: {} values", cal_data.channel_scales.len());
    eprintln!("Rotation matrix: {}x{}", head_dim, head_dim);
    eprintln!("Codebooks: 2-bit, 3-bit, 4-bit");
    eprintln!("\nCalibration will be auto-loaded on next inference run.");

    Ok(())
}
