//! Shared CLI helpers: TQ config resolution, auto-calibration, VRAM-aware auto-TQ.

use anyhow::Result;

use crate::engine::{self, Engine};
use crate::{auto_tq, calibrate, catalog, config, hub, models};

pub(crate) fn resolve_tq_config(turbo_quant: bool, tq_bits: u8) -> Option<tq_kv::TurboQuantConfig> {
    resolve_tq_config_for_model(turbo_quant, tq_bits, None)
}

/// Auto-calibrate a model on first TQ use. Runs a quick prefill (~10s) to collect
/// key vector statistics, computes calibration data, and saves for future runs.
pub(crate) fn auto_calibrate(model_name: &str, force_cpu: bool) -> Result<calibrate::CalibrationData> {
    let t0 = std::time::Instant::now();

    let (gguf_path, tokenizer_path) = hub::resolve(model_name)?;
    let tok_path = tokenizer_path.ok_or_else(|| anyhow::anyhow!("No tokenizer for '{}'", model_name))?;
    let mf = gguf_path.to_string_lossy().to_string();
    let arch = config::detect_arch(&mf);

    // Determine head_dim from GGUF
    let head_dim = {
        let mut file = std::fs::File::open(&gguf_path)?;
        let content = crate::gguf::GgufContent::read(&mut file)
            .map_err(|e| anyhow::anyhow!("GGUF read: {}", e))?;
        // Find embedding_length key regardless of architecture prefix
        let n_embd = content.metadata.iter()
            .find(|(k, _)| k.ends_with(".embedding_length"))
            .and_then(|(_, v)| v.to_u32().ok())
            .unwrap_or(4096) as usize;
        let n_head = content.metadata.iter()
            .find(|(k, _)| k.ends_with(".attention.head_count"))
            .and_then(|(_, v)| v.to_u32().ok())
            .unwrap_or(32) as usize;
        n_embd / n_head
    };

    // Initialize collector (256 samples for quick calibration)
    let collector = calibrate::init_collector(head_dim, 256);

    // Default calibration text (Eiffel Tower passage — diverse vocabulary)
    let cal_text = "The tower is 324 metres tall, about the same height as an 81-storey building, \
         and the tallest structure in Paris. Its base is square, measuring 125 metres on \
         each side. During its construction, the Eiffel Tower surpassed the Washington \
         Monument to become the tallest human-made structure in the world, a title it held \
         for 41 years until the Chrysler Building in New York City was finished in 1930.";

    // Load model without TQ (skip all compression)
    std::env::set_var("TQ_SKIP", "999");
    let tq_stub = tq_kv::TurboQuantConfig { bits: 4, ..Default::default() };
    let mut engine = Engine::load_with_device(&gguf_path, &tok_path, arch, Some(tq_stub), force_cpu)?;
    std::env::remove_var("TQ_SKIP");

    // Run prefill
    let params = engine::GenerationParams { max_tokens: 1, temperature: 0.0, ..Default::default() };
    let _ = engine.generate_silent(cal_text, &params);
    drop(engine); // free memory before loading again

    // Compute calibration
    let c = collector.lock().map_err(|e| anyhow::anyhow!("Collector lock: {}", e))?;
    if c.count < 50 {
        anyhow::bail!("Too few samples ({}) — auto-calibrate needs at least 50", c.count);
    }
    let cal_data = calibrate::compute_calibration(&c, model_name, tq_kv::TurboQuantConfig::default().rotation_seed);

    // Save
    let (name, tag) = if let Some(e) = crate::catalog::find(model_name) {
        (e.name.to_string(), e.tag.to_string())
    } else if let Some((n, t)) = model_name.split_once(':') {
        (n.to_string(), t.to_string())
    } else {
        (model_name.to_string(), "default".to_string())
    };
    let path = calibrate::calibration_path(&name, &tag);
    calibrate::save_calibration(&cal_data, &path)?;

    eprintln!("  Auto-calibrate done: {} samples, {:.1}s → {}",
        c.count, t0.elapsed().as_secs_f64(), path.display());

    Ok(cal_data)
}

pub(crate) fn resolve_tq_config_for_model(turbo_quant: bool, tq_bits: u8, model_name: Option<&str>) -> Option<tq_kv::TurboQuantConfig> {
    if turbo_quant {
        let mut config = match tq_bits {
            2 => tq_kv::TurboQuantConfig::extreme(),
            3 => tq_kv::TurboQuantConfig::aggressive(),
            _ => tq_kv::TurboQuantConfig::balanced(),
        };
        // TQ_RESIDUAL=2 enables 2-bit residual quantization
        if let Ok(val) = std::env::var("TQ_RESIDUAL") {
            if let Ok(bits) = val.parse::<u8>() {
                config.residual_bits = bits;
            }
        }
        // TQ_OUTLIER=2 preserves top-2 outlier entries per vector at full precision
        if let Ok(val) = std::env::var("TQ_OUTLIER") {
            if let Ok(k) = val.parse::<usize>() {
                config.outlier_k = k;
            }
        }
        // TQ_GROUP=0 disables group quantization (per-vector sigma)
        if let Ok(val) = std::env::var("TQ_GROUP") {
            if let Ok(gs) = val.parse::<usize>() {
                config.group_size = gs;
            }
        }
        // Auto-load calibration data if available (unless TQ_NO_CAL=1)
        let no_cal = std::env::var("TQ_NO_CAL").ok().map_or(false, |v| v == "1");
        if !no_cal {
            if let Some(name) = model_name {
                let cal_data = calibrate::load_calibration_for_model(name)
                    .or_else(|| {
                        // No calibration found — run auto-calibrate
                        let no_auto = std::env::var("TQ_NO_AUTO_CAL").ok().map_or(false, |v| v == "1");
                        if no_auto { return None; }
                        eprintln!("No calibration found for '{}' — running auto-calibrate...", name);
                        match auto_calibrate(name, false) {
                            Ok(cal) => Some(cal),
                            Err(e) => {
                                eprintln!("  Auto-calibrate failed: {} — continuing without calibration", e);
                                None
                            }
                        }
                    });
                if let Some(cal_data) = cal_data {
                    eprintln!("Using calibrated channel bias + per-head bits");
                    cal_data.apply_to_config(&mut config);
                    // Sprint 3: apply auto_layer_bits from calibration
                    if let Some(ref alb) = cal_data.auto_layer_bits {
                        let n_2 = alb.iter().filter(|&&b| b == 2).count();
                        let n_3 = alb.iter().filter(|&&b| b == 3).count();
                        let n_4 = alb.iter().filter(|&&b| b == 4).count();
                        let n_skip = alb.iter().filter(|&&b| b == 0).count();
                        if n_2 > 0 || n_3 > 0 {
                            eprintln!("  Auto layer bits: {} skip, {} @2-bit, {} @3-bit, {} @4-bit",
                                n_skip, n_2, n_3, n_4);
                        }
                        models::turbo_generic::set_auto_layer_bits(alb.clone());
                    }
                    // TriAttention ON by default with TQ (the product mode).
                    // Disable with TQ_TRIATTN=0 or set_triattention_override(false).
                    let tri_disabled = std::env::var("TQ_TRIATTN").ok().map_or(false, |v| v == "0");
                    if !tri_disabled {
                        calibrate::init_triattention(&cal_data, cal_data.head_dim);
                        models::turbo_generic::set_triattention_override(true);
                    }
                }
            }
        }
        Some(config)
    } else {
        None
    }
}

/// Resolve TQ config with auto-TQ support.
///
/// Priority: --turbo-quant (explicit on) > --no-auto-tq (explicit off) > auto-decide
pub(crate) fn resolve_tq_config_with_auto(
    turbo_quant: bool,
    no_auto_tq: bool,
    tq_bits: u8,
    force_cpu: bool,
    model_name: &str,
) -> Option<tq_kv::TurboQuantConfig> {
    // If user explicitly enabled TQ, use that
    if turbo_quant {
        return resolve_tq_config_for_model(true, tq_bits, Some(model_name));
    }

    // If user explicitly disabled auto-TQ, return None
    if no_auto_tq {
        return None;
    }

    // Auto-TQ: check if we should enable compression
    if force_cpu {
        // Auto-TQ only applies on GPU
        return None;
    }

    let device = match crate::cuda::TqDevice::cuda_if_available(0).map_err(|e| anyhow::anyhow!("{}", e)) {
        Ok(dev) if dev.is_cuda() => dev,
        _ => return None,
    };

    // Look up model in catalog for size/arch info
    let entry = catalog::find(model_name);
    let (model_size_bytes, arch, size_gb) = if let Some(e) = entry {
        ((e.size_gb * 1024.0 * 1024.0 * 1024.0) as u64, e.arch, e.size_gb)
    } else {
        // Unknown model -- can't auto-decide, skip
        return None;
    };

    let (n_layers, n_kv_heads, head_dim) = auto_tq::estimate_arch_params(arch, size_gb);
    let max_context = 4096; // default context window

    let result = auto_tq::decide(&device, model_size_bytes, n_layers, n_kv_heads, head_dim, max_context);
    auto_tq::print_decision(&result);
    auto_tq::to_tq_config(&result)
}
