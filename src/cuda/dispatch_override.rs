//! Runtime overrides for kernel dispatch choice.
//!
//! Each dispatch site in `kernels.rs` reads its variant from an `OnceLock<V>`
//! that pins on first call. That's right for production runs (env var → lock →
//! done) but blocks in-process autotuning, where we want to cycle variants
//! without spawning a subprocess per variant.
//!
//! The override layer works like this:
//!   - Per-dispatch `AtomicI8`, sentinel `-1` means "no override, fall through to
//!     the OnceLock-cached env default".
//!   - Dispatch reads the atomic first: if >= 0, interpret as variant index;
//!     else resolve via the usual OnceLock path.
//!   - External callers (autotune) set the atomic via `set_*_override_by_name`
//!     or clear with `clear_all`.
//!
//! Variant indices MUST match the numeric discriminants of the corresponding
//! enums in `kernels.rs`. These are kept stable via `#[repr(i8)]`. If you add
//! a new variant to one of those enums, append it (don't reorder) and extend
//! the `*_override_by_name` matcher below.

use std::sync::atomic::{AtomicI8, Ordering};

pub static Q4KM_OVERRIDE:   AtomicI8 = AtomicI8::new(-1);
pub static DOWN_OVERRIDE:   AtomicI8 = AtomicI8::new(-1);
pub static GATEUP_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
pub static QKV_OVERRIDE:    AtomicI8 = AtomicI8::new(-1);
pub static Q6K_OVERRIDE:    AtomicI8 = AtomicI8::new(-1);

/// Clear every dispatch override — next kernel call falls through to the
/// env-derived OnceLock default.
pub fn clear_all() {
    Q4KM_OVERRIDE.store(-1,   Ordering::Relaxed);
    DOWN_OVERRIDE.store(-1,   Ordering::Relaxed);
    GATEUP_OVERRIDE.store(-1, Ordering::Relaxed);
    QKV_OVERRIDE.store(-1,    Ordering::Relaxed);
    Q6K_OVERRIDE.store(-1,    Ordering::Relaxed);
}

/// Set Q4KM dispatch override. Returns true on recognised name.
///
/// Variant index ↔ name map (must mirror `Dp4aVariant` in q4km_matvec):
///   0 = Off, 1 = V1 (dp4a), 2 = V2 (dp4a_v2), 3 = Mrow4 (dp4a_mrow4)
/// Legacy env names (baseline/mrow8/wx/mrow16) map to Off so the fallback
/// branch drives the legacy kernel table.
pub fn set_q4km_override_by_name(name: &str) -> bool {
    let idx: i8 = match name {
        "dp4a"       => 1,
        "dp4a_v2"    => 2,
        "dp4a_mrow4" => 3,
        "baseline" | "mrow8" | "wx" | "mrow16" => 0,
        _ => return false,
    };
    Q4KM_OVERRIDE.store(idx, Ordering::Relaxed);
    true
}

/// Set TQ_DOWN dispatch override. Variant map matches `DownVariant` in
/// `fused_q4km_down_residual`.
///   0 = Baseline, 1 = Mrow2, 2 = Cpasync, 3 = Dp4aV1, 4 = Dp4aV2, 5 = Dp4aV3
pub fn set_down_override_by_name(name: &str) -> bool {
    let idx: i8 = match name {
        "baseline" => 0,
        "mrow2"    => 1,
        "cpasync"  => 2,
        "dp4a"     => 3,
        "dp4a_v2"  => 4,
        "dp4a_v3"  => 5,
        _ => return false,
    };
    DOWN_OVERRIDE.store(idx, Ordering::Relaxed);
    true
}

/// Gateup dispatch stays string-addressed in `fused_addnorm_q4km_gateup_silu`,
/// so we key this by a separate static table of &'static str instead of index.
/// Leave as-is for now — the in-process autotune for gateup will fall back
/// to subprocess mode until a structured enum lands.
pub fn set_gateup_override_by_name(_name: &str) -> bool { false }

/// QKV dispatch matches `QkvVariant` in `fused_norm_q4km_qkv_bias`.
///   0 = Baseline, 1 = Cpasync, 2 = Dp4a
pub fn set_qkv_override_by_name(name: &str) -> bool {
    let idx: i8 = match name {
        "baseline" => 0,
        "cpasync"  => 1,
        "dp4a"     => 2,
        _ => return false,
    };
    QKV_OVERRIDE.store(idx, Ordering::Relaxed);
    true
}

/// TQ_Q6K dispatch. Matches `Q6kVariant` in `q6k_matvec`.
///   0 = Baseline, 1 = Mrow8, 2 = Dp4aV2
pub fn set_q6k_override_by_name(name: &str) -> bool {
    let idx: i8 = match name {
        "baseline" => 0,
        "mrow8"    => 1,
        "dp4a_v2"  => 2,
        _ => return false,
    };
    Q6K_OVERRIDE.store(idx, Ordering::Relaxed);
    true
}

/// Generic dispatcher: route (env_key, variant_name) → the matching setter.
/// Returns `true` if the key is recognised AND the name is valid.
pub fn set_override(env_key: &str, variant: &str) -> bool {
    match env_key {
        "TQ_Q4KM"   => set_q4km_override_by_name(variant),
        "TQ_DOWN"   => set_down_override_by_name(variant),
        "TQ_GATEUP" => set_gateup_override_by_name(variant),
        "TQ_QKV"    => set_qkv_override_by_name(variant),
        "TQ_Q6K"    => set_q6k_override_by_name(variant),
        _ => false,
    }
}

/// True iff the dispatch identified by `env_key` supports runtime override
/// via `set_override`. Autotune uses this to decide subprocess-vs-in-process
/// per dispatch.
pub fn supports_override(env_key: &str) -> bool {
    matches!(env_key, "TQ_Q4KM" | "TQ_DOWN" | "TQ_QKV" | "TQ_Q6K")
}
