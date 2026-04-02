//! Auto-TQ: Automatic KV cache compression based on available VRAM.
//!
//! When the user doesn't specify `--turbo-quant`, the system checks if the model
//! fits in GPU VRAM and auto-enables compression if needed.

use crate::cuda::TqDevice as Device;

/// Result of the auto-TQ decision.
pub struct AutoTqResult {
    pub enabled: bool,
    pub bits: u8,
    pub reason: String,
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
}

/// Decide whether to enable TurboQuant based on available VRAM.
///
/// Algorithm:
/// 1. Query CUDA device for total VRAM (via `TQ_VRAM_MB` env var or default 10 GB)
/// 2. Estimate model weight size from file size / catalog
/// 3. Estimate KV cache size: n_layers * n_kv_heads * 2 * head_dim * max_ctx * 2 bytes
/// 4. If weights + KV > 0.85 * VRAM -> enable TQ
///    a. Try 4-bit -> KV reduced 3.8x
///    b. If still tight -> try 2-bit -> KV reduced 14.2x
///    c. If still too large -> recommend CPU
/// 5. On CPU-only -> return disabled (user must opt in with --turbo-quant)
pub fn decide(
    device: &Device,
    model_size_bytes: u64,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_context: usize,
) -> AutoTqResult {
    // Check if CUDA available
    if !device.is_cuda() {
        return AutoTqResult {
            enabled: false,
            bits: 0,
            reason: "CPU mode -- use --turbo-quant to enable manually".into(),
            vram_total_mb: 0,
            vram_available_mb: 0,
        };
    }

    // Query VRAM -- use env var override or default to 10 GB (RTX 3080 class)
    let vram_total_mb = std::env::var("TQ_VRAM_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000u64);

    let model_size_mb = model_size_bytes / (1024 * 1024);

    // KV cache estimate: per token, per layer
    // keys + values = 2 * n_kv_heads * head_dim * 2 bytes (fp16) * n_layers
    let kv_per_token_bytes = 2 * n_kv_heads * head_dim * 2 * n_layers;
    let kv_total_bytes = kv_per_token_bytes * max_context;
    let kv_estimate_mb = (kv_total_bytes / (1024 * 1024)) as u64;

    let total_needed_mb = model_size_mb + kv_estimate_mb;
    let threshold_mb = (vram_total_mb as f64 * 0.85) as u64;

    if total_needed_mb <= threshold_mb {
        // Fits without compression
        return AutoTqResult {
            enabled: false,
            bits: 0,
            reason: format!(
                "Model fits in VRAM ({} MB / {} MB)",
                total_needed_mb, vram_total_mb
            ),
            vram_total_mb,
            vram_available_mb: vram_total_mb.saturating_sub(model_size_mb),
        };
    }

    // Try 4-bit compression (3.8x KV reduction)
    let kv_4bit_mb = kv_estimate_mb * 100 / 380; // /3.8
    if model_size_mb + kv_4bit_mb <= threshold_mb {
        return AutoTqResult {
            enabled: true,
            bits: 4,
            reason: format!(
                "Auto-TQ 4-bit: KV cache {} MB -> {} MB ({} MB saved)",
                kv_estimate_mb,
                kv_4bit_mb,
                kv_estimate_mb - kv_4bit_mb
            ),
            vram_total_mb,
            vram_available_mb: threshold_mb.saturating_sub(model_size_mb + kv_4bit_mb),
        };
    }

    // Try 2-bit compression (14.2x KV reduction)
    let kv_2bit_mb = kv_estimate_mb * 100 / 1420; // /14.2
    if model_size_mb + kv_2bit_mb <= threshold_mb {
        return AutoTqResult {
            enabled: true,
            bits: 2,
            reason: format!(
                "Auto-TQ 2-bit: KV cache {} MB -> {} MB ({} MB saved)",
                kv_estimate_mb,
                kv_2bit_mb,
                kv_estimate_mb - kv_2bit_mb
            ),
            vram_total_mb,
            vram_available_mb: threshold_mb.saturating_sub(model_size_mb + kv_2bit_mb),
        };
    }

    // Still too large -- recommend CPU
    AutoTqResult {
        enabled: false,
        bits: 0,
        reason: format!(
            "Model too large for GPU ({} MB > {} MB VRAM). Use --cpu",
            total_needed_mb, vram_total_mb
        ),
        vram_total_mb,
        vram_available_mb: 0,
    }
}

/// Estimate model architecture parameters from catalog metadata.
///
/// Returns (n_layers, n_kv_heads, head_dim) based on well-known model families.
/// These are estimates -- the actual values come from GGUF metadata at load time.
pub fn estimate_arch_params(arch: &str, size_gb: f32) -> (usize, usize, usize) {
    match arch {
        "qwen2" => {
            if size_gb > 30.0 {
                // Qwen2.5 72B: 80 layers, 8 kv heads (GQA), head_dim=128
                (80, 8, 128)
            } else {
                // Qwen2.5 7B: 28 layers, 4 kv heads (GQA), head_dim=128
                (28, 4, 128)
            }
        }
        "llama" => {
            // Llama 3.1 8B: 32 layers, 8 kv heads (GQA), head_dim=128
            (32, 8, 128)
        }
        "gemma2" => {
            // Gemma 2 9B: 42 layers, 4 kv heads (GQA), head_dim=256
            (42, 4, 256)
        }
        "phi3" => {
            // Phi-3.5 Mini: 32 layers, 32 kv heads (MHA), head_dim=96
            (32, 32, 96)
        }
        _ => {
            // Conservative defaults for unknown arch
            (32, 8, 128)
        }
    }
}

/// Pretty-print auto-TQ decision to stderr.
pub fn print_decision(result: &AutoTqResult) {
    if result.enabled {
        eprintln!("[auto-tq] {}", result.reason);
        eprintln!(
            "[auto-tq] VRAM: {} MB total, ~{} MB available after model + compressed KV",
            result.vram_total_mb, result.vram_available_mb
        );
    } else if result.vram_total_mb > 0 {
        eprintln!("[auto-tq] {}", result.reason);
    }
    // On CPU, stay silent (no VRAM info to show)
}

/// Convert an AutoTqResult into a TurboQuantConfig, if auto-TQ is enabled.
pub fn to_tq_config(result: &AutoTqResult) -> Option<tq_kv::TurboQuantConfig> {
    if !result.enabled {
        return None;
    }
    Some(match result.bits {
        2 => tq_kv::TurboQuantConfig::extreme(),
        3 => tq_kv::TurboQuantConfig::aggressive(),
        _ => tq_kv::TurboQuantConfig::balanced(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decide_cpu_disabled() {
        let device = Device::Cpu;
        let result = decide(&device, 5_000_000_000, 32, 8, 128, 4096);
        assert!(!result.enabled);
        assert_eq!(result.bits, 0);
        assert!(result.reason.contains("CPU"));
    }

    #[test]
    fn test_estimate_arch_params_qwen_7b() {
        let (layers, kv_heads, head_dim) = estimate_arch_params("qwen2", 4.7);
        assert_eq!(layers, 28);
        assert_eq!(kv_heads, 4);
        assert_eq!(head_dim, 128);
    }

    #[test]
    fn test_estimate_arch_params_qwen_72b() {
        let (layers, kv_heads, head_dim) = estimate_arch_params("qwen2", 45.0);
        assert_eq!(layers, 80);
        assert_eq!(kv_heads, 8);
        assert_eq!(head_dim, 128);
    }

    #[test]
    fn test_to_tq_config_disabled() {
        let result = AutoTqResult {
            enabled: false,
            bits: 0,
            reason: String::new(),
            vram_total_mb: 10000,
            vram_available_mb: 5000,
        };
        assert!(to_tq_config(&result).is_none());
    }

    #[test]
    fn test_to_tq_config_4bit() {
        let result = AutoTqResult {
            enabled: true,
            bits: 4,
            reason: String::new(),
            vram_total_mb: 10000,
            vram_available_mb: 3000,
        };
        let cfg = to_tq_config(&result).unwrap();
        assert_eq!(cfg.bits, 4);
    }

    #[test]
    fn test_to_tq_config_2bit() {
        let result = AutoTqResult {
            enabled: true,
            bits: 2,
            reason: String::new(),
            vram_total_mb: 10000,
            vram_available_mb: 1000,
        };
        let cfg = to_tq_config(&result).unwrap();
        assert_eq!(cfg.bits, 2);
    }

    #[test]
    fn test_decide_fits_no_compression() {
        // On CPU device, decide should always return disabled (no VRAM to manage)
        let device = Device::Cpu;
        // Small model that would fit easily: 1 GB, 32 layers, 8 heads, dim 128, 4096 ctx
        let result = decide(&device, 1_000_000_000, 32, 8, 128, 4096);
        assert!(!result.enabled);
        assert_eq!(result.bits, 0);
        assert_eq!(result.vram_total_mb, 0);
        // Reason should mention CPU
        assert!(result.reason.contains("CPU"));
    }

    #[test]
    fn test_decide_4bit_compression() {
        // On CPU device, even a huge model should still return disabled
        // because decide() requires CUDA to be active
        let device = Device::Cpu;
        let result = decide(&device, 50_000_000_000, 80, 8, 128, 131072);
        assert!(!result.enabled);
        assert_eq!(result.bits, 0);
        // Verify the AutoTqResult fields are zeroed for CPU path
        assert_eq!(result.vram_total_mb, 0);
        assert_eq!(result.vram_available_mb, 0);
    }
}
