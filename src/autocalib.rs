//! Model-specific kernel dispatch presets.
//!
//! Automatically selects the best kernel variant for a given (model_arch,
//! hidden_dim, sm_arch) combination, based on empirical benchmarks.
//!
//! The preset is applied by setting env vars BEFORE any kernel dispatch
//! OnceLock has initialized. User-set env vars take precedence — we only
//! populate the gaps.
//!
//! ## Why this exists
//!
//! Our dispatch has 5+ envs (TQ_Q4KM, TQ_DOWN, TQ_GATEUP, TQ_QKV, TQ_Q6K)
//! and each has 3-8 variants. The optimal combination differs per model
//! (Qwen2 7B vs Llama3 8B have different hidden/intermediate dims, affecting
//! which kernel wins). Without presets, users must hand-tune env vars per
//! model, which is tedious and discoverable only via trial.
//!
//! ## How it works
//!
//! 1. At model load, we detect (arch_name, hidden_dim, sm_major, sm_minor).
//! 2. We look up `PRESETS` for a match.
//! 3. For each dispatch env var, if unset in the environment, we set it to
//!    the preset's recommended variant via `std::env::set_var`.
//! 4. Subsequent `std::env::var("TQ_X")` lookups in kernel dispatch code
//!    see the preset values.
//!
//! ## Extending
//!
//! Add entries to `PRESETS` after measuring with `tq bench`. Each entry
//! includes the shipped-date and the bench result that justified the choice.
//!
//! ## Autotune future
//!
//! The preset table is hand-curated. A future `tq autotune <model>` subcommand
//! will replace this with per-machine measured choices, writing to
//! `~/.cache/tq/autocalib-<model_hash>.json`. The lookup order will then be:
//! 1. User-set env var
//! 2. On-disk autotune cache for this machine
//! 3. Hard-coded preset for this model/arch
//! 4. Built-in defaults (current OnceLock fallbacks)

use std::collections::HashMap;

/// A recommended dispatch for a specific (model, SM arch) combination.
#[derive(Debug, Clone, Default)]
pub struct DispatchPreset {
    pub tq_q4km:   Option<&'static str>,
    pub tq_down:   Option<&'static str>,
    pub tq_gateup: Option<&'static str>,
    pub tq_qkv:    Option<&'static str>,
    pub tq_q6k:    Option<&'static str>,
    /// Flash-decode threshold (token count; 0 = always flash).
    pub tq_flash_threshold: Option<&'static str>,
    /// Flash-decode split size (16/24/32).
    pub tq_flash_split:     Option<&'static str>,
    /// One-line provenance string for logging.
    pub provenance: &'static str,
}

/// Key for the preset table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PresetKey {
    pub arch: String,
    pub hidden_dim: usize,
    pub sm_major: u8,
    pub sm_minor: u8,
}

/// Build the preset table. Called once at first lookup.
fn build_presets() -> HashMap<PresetKey, DispatchPreset> {
    let mut m = HashMap::new();

    // Qwen2 7B (hidden=3584) on sm_86 (RTX 3080/3090/A40 etc)
    // Session 2026-04-17 shipped defaults.
    m.insert(
        PresetKey { arch: "qwen2".into(), hidden_dim: 3584, sm_major: 8, sm_minor: 6 },
        DispatchPreset {
            tq_q4km:   Some("dp4a_v2"),
            tq_down:   Some("dp4a_v3"),
            tq_gateup: Some("dp4a"),
            tq_qkv:    Some("dp4a"),
            tq_q6k:    Some("dp4a_v2"),
            tq_flash_threshold: Some("0"),
            tq_flash_split:     Some("32"),
            provenance: "Qwen2 7B sm_86 (Std 68 tok/s, commit d3ce20c, 2026-04-17)",
        },
    );

    // Qwen2.5 7B — same hidden_dim, inherit Qwen2 7B preset until measured.
    m.insert(
        PresetKey { arch: "qwen2.5".into(), hidden_dim: 3584, sm_major: 8, sm_minor: 6 },
        DispatchPreset {
            tq_q4km:   Some("dp4a_v2"),
            tq_down:   Some("dp4a_v3"),
            tq_gateup: Some("dp4a"),
            tq_qkv:    Some("dp4a"),
            tq_q6k:    Some("dp4a_v2"),
            tq_flash_threshold: Some("0"),
            tq_flash_split:     Some("32"),
            provenance: "Inherited from Qwen2 7B sm_86 (unmeasured on Qwen2.5)",
        },
    );

    // Llama3 8B (hidden=4096) on sm_86 — extrapolated from Qwen2 pattern,
    // needs empirical validation (different intermediate_dim=14336 vs 18944).
    m.insert(
        PresetKey { arch: "llama".into(), hidden_dim: 4096, sm_major: 8, sm_minor: 6 },
        DispatchPreset {
            tq_q4km:   Some("dp4a_v2"),
            tq_down:   Some("dp4a_v3"),
            tq_gateup: Some("dp4a"),
            tq_qkv:    Some("dp4a"),
            tq_q6k:    Some("dp4a_v2"),
            tq_flash_threshold: Some("0"),
            tq_flash_split:     Some("32"),
            provenance: "Extrapolated from Qwen2 sm_86 (Llama3 validation pending)",
        },
    );

    m
}

static PRESETS: std::sync::OnceLock<HashMap<PresetKey, DispatchPreset>> =
    std::sync::OnceLock::new();

/// Apply the preset for a given model+arch, if one exists and the user
/// hasn't already set these env vars.
///
/// Must be called BEFORE any kernel dispatch `OnceLock` reads the env vars.
/// Typical call site: just after model metadata is parsed from GGUF and
/// compute capability is detected, before `Engine::load_with_device` builds
/// the kernel registry.
///
/// Returns the preset applied (for logging), or `None` if no match.
pub fn apply_preset(
    arch: &str,
    hidden_dim: usize,
    sm_major: u8,
    sm_minor: u8,
) -> Option<&'static DispatchPreset> {
    let table = PRESETS.get_or_init(build_presets);
    let key = PresetKey {
        arch: arch.to_ascii_lowercase(),
        hidden_dim,
        sm_major,
        sm_minor,
    };
    let preset = table.get(&key)?;

    let mut applied: Vec<(&str, &str)> = Vec::new();
    let mut set_if_unset = |key: &'static str, val_opt: Option<&'static str>| {
        if let Some(val) = val_opt {
            if std::env::var(key).is_err() {
                // SAFETY: single-threaded at startup; no concurrent env access.
                std::env::set_var(key, val);
                applied.push((key, val));
            }
        }
    };
    set_if_unset("TQ_Q4KM",            preset.tq_q4km);
    set_if_unset("TQ_DOWN",            preset.tq_down);
    set_if_unset("TQ_GATEUP",          preset.tq_gateup);
    set_if_unset("TQ_QKV",             preset.tq_qkv);
    set_if_unset("TQ_Q6K",             preset.tq_q6k);
    set_if_unset("TQ_FLASH_THRESHOLD", preset.tq_flash_threshold);
    set_if_unset("TQ_FLASH_SPLIT",     preset.tq_flash_split);

    if applied.is_empty() {
        eprintln!(
            "[autocalib] Preset matched ({}): all env vars pre-set by user — no override.",
            preset.provenance
        );
    } else {
        eprintln!(
            "[autocalib] Applied preset: {} -> {}",
            preset.provenance,
            applied.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(" ")
        );
    }
    Some(preset)
}

/// Lookup only — returns the preset without applying it. Useful for docs/UI.
#[cfg(test)]
pub fn lookup(
    arch: &str,
    hidden_dim: usize,
    sm_major: u8,
    sm_minor: u8,
) -> Option<&'static DispatchPreset> {
    let table = PRESETS.get_or_init(build_presets);
    table.get(&PresetKey {
        arch: arch.to_ascii_lowercase(),
        hidden_dim,
        sm_major,
        sm_minor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_7b_sm86_has_preset() {
        let p = lookup("qwen2", 3584, 8, 6).expect("Qwen2 7B sm_86 preset missing");
        assert_eq!(p.tq_q6k, Some("dp4a_v2"));
        assert_eq!(p.tq_down, Some("dp4a_v3"));
    }

    #[test]
    fn case_insensitive_arch() {
        assert!(lookup("Qwen2", 3584, 8, 6).is_some());
        assert!(lookup("QWEN2", 3584, 8, 6).is_some());
    }

    #[test]
    fn missing_model_returns_none() {
        assert!(lookup("unknown_arch", 1234, 8, 6).is_none());
    }
}
