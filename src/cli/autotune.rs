//! `tq autotune` — measure the best kernel dispatch variant per dispatch for
//! the current (model, GPU) combination, and write the winners to a
//! per-machine cache that `autocalib` will consume on next model load.
//!
//! ## Strategy
//!
//! One-variable-at-a-time sweep via subprocess invocations of `tq bench`.
//! For each dispatch env (TQ_Q4KM, TQ_DOWN, TQ_GATEUP, TQ_QKV, TQ_Q6K), we
//! try each candidate variant by spawning a child `tq bench --tq-only` with
//! only that env var forced; autocalib preset handles the other dispatches
//! in the child. We parse the child's tok/s from stdout and pick the winner.
//!
//! This pattern has three properties that matter for robustness:
//!
//! 1. **Fresh process per variant** — no OnceLock state carry-over, no need
//!    to refactor the dispatch internals into runtime-switchable atomics.
//! 2. **Preset-as-baseline** — the non-target dispatches use the current
//!    hand-curated preset, so we're measuring marginal gain vs. preset, not
//!    vs. a worst-case cold default.
//! 3. **Same binary path** — the child is just a fresh invocation of the
//!    currently-running tq executable, so there's no ABI mismatch.
//!
//! ## Output
//!
//! JSON cache at `~/.cache/tq/autocalib-<model_hash>-sm<major><minor>.json`.
//! On next model load, `autocalib::apply_preset` reads this first and applies
//! its winners before falling back to the hand-curated preset table.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::{Cli, Commands};

/// Dispatch keys we sweep over, with their candidate variants per mode.
/// First variant is used as the initial "best" (preset default).
struct DispatchSweep {
    env_key: &'static str,
    // Full set of variants tried in normal mode.
    full: &'static [&'static str],
    // Reduced set for --quick mode (just sanity-check top 2).
    quick: &'static [&'static str],
}

const SWEEPS: &[DispatchSweep] = &[
    DispatchSweep {
        env_key: "TQ_Q4KM",
        full:  &["dp4a_v2", "dp4a", "mrow8", "baseline"],
        quick: &["dp4a_v2", "mrow8"],
    },
    DispatchSweep {
        env_key: "TQ_DOWN",
        full:  &["dp4a_v3", "dp4a_v2", "dp4a", "cpasync", "baseline"],
        quick: &["dp4a_v3", "dp4a"],
    },
    DispatchSweep {
        env_key: "TQ_GATEUP",
        full:  &["dp4a", "mrow8", "mrow4", "cpasync", "baseline"],
        quick: &["dp4a", "mrow8"],
    },
    DispatchSweep {
        env_key: "TQ_QKV",
        full:  &["dp4a", "baseline"],
        quick: &["dp4a"],
    },
    DispatchSweep {
        env_key: "TQ_Q6K",
        full:  &["dp4a_v2", "baseline", "mrow8"],
        quick: &["dp4a_v2", "baseline"],
    },
];

/// JSON on-disk format for a calibration result.
///
/// Filename encodes the minimum fields needed to route the cache to the
/// right (arch, hidden_dim, sm) triple. JSON body adds provenance +
/// per-variant raw measurements so `tq autotune` output stays debuggable.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AutotuneCache {
    pub model_name:  String,
    pub model_hash:  String,
    /// Model architecture string from GGUF `general.architecture` (e.g. "qwen2", "llama").
    pub arch:        String,
    /// Embedding dim (hidden_size) — distinguishes 7B vs 13B even within same arch.
    pub hidden_dim:  usize,
    pub sm_major:    u8,
    pub sm_minor:    u8,
    pub tokens_per_run: u32,
    pub created_utc: String,
    pub winners:     std::collections::BTreeMap<String, String>,
    pub all_results: Vec<VariantResult>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VariantResult {
    pub env_key: String,
    pub variant: String,
    /// Median tok/s across `samples.len()` reps. NaN on all-fail.
    pub tok_per_sec: f32,
    /// Raw per-rep measurements (length == reps arg).
    pub samples: Vec<f32>,
    /// Whether at least one rep succeeded.
    pub ok: bool,
    /// If --validate-ppl ran: measured PPL. NaN if not run or failed.
    pub ppl: f32,
    /// True if the variant was disqualified by PPL tolerance (ppl > baseline * (1+tol)).
    pub ppl_disqualified: bool,
}

pub(crate) fn cmd_autotune(cli: &Cli) -> Result<()> {
    let (model, tokens, quick, only, dry_run, metric_str, reps, validate_ppl, ppl_tol) = match &cli.command {
        Some(Commands::Autotune { model, tokens, quick, only, dry_run, metric, reps, validate_ppl, ppl_tolerance }) => {
            (
                model.clone(), *tokens, *quick, only.clone(), *dry_run, metric.clone(),
                *reps, validate_ppl.clone(), *ppl_tolerance,
            )
        }
        _ => unreachable!(),
    };
    let metric = match metric_str.to_ascii_lowercase().as_str() {
        "std" | "standard"         => PrimaryMetric::Standard,
        "tq" | "tq4" | "tq4bit"    => PrimaryMetric::Tq4Bit,
        "tq+ta" | "tqta" | "triattn" | "tq+triattn" => PrimaryMetric::TqTriAttn,
        other => anyhow::bail!("unknown --metric '{}': use 'std', 'tq', or 'tq+ta'", other),
    };
    let reps = reps.max(1);

    // Filter sweeps by --only.
    let only_set: Option<std::collections::HashSet<String>> = only.map(|s| {
        s.split(',').map(|x| x.trim().to_ascii_lowercase()).collect()
    });
    let sweeps: Vec<&DispatchSweep> = SWEEPS.iter().filter(|d| {
        only_set.as_ref().map_or(true, |set| {
            // allow naming like "q4km" (strip TQ_ prefix) or "TQ_Q4KM" (full).
            let key_lower = d.env_key.to_ascii_lowercase();
            let short = key_lower.strip_prefix("tq_").unwrap_or(&key_lower);
            set.contains(short) || set.contains(&key_lower)
        })
    }).collect();

    // Detect SM + model hash up front.
    let (sm_major, sm_minor) = detect_sm();
    let (model_hash, arch, hidden_dim) = probe_model(&model)?;
    let exe = std::env::current_exe().context("current_exe()")?;

    eprintln!("=== tq autotune ===");
    eprintln!("  model       : {} (arch={} hidden={} hash {})", model, arch, hidden_dim, &model_hash[..12]);
    eprintln!("  gpu         : sm_{}{}", sm_major, sm_minor);
    eprintln!("  tokens/run  : {}", tokens);
    eprintln!("  mode        : {}", if quick { "quick" } else { "full" });
    eprintln!("  metric      : {:?}", metric);
    eprintln!("  reps        : {} (median aggregation)", reps);
    if let Some(ref p) = validate_ppl {
        eprintln!("  ppl-guard   : file={} tol={:.2}%", p.display(), ppl_tol * 100.0);
    } else {
        eprintln!("  ppl-guard   : off (perf-only)");
    }
    eprintln!("  dispatches  : {}", sweeps.iter().map(|s| s.env_key).collect::<Vec<_>>().join(", "));
    eprintln!();

    let n_variants: usize = sweeps.iter().map(|d| if quick { d.quick.len() } else { d.full.len() }).sum();
    let ppl_cost = if validate_ppl.is_some() { 1 } else { 0 };
    let runs_per_variant = (reps as usize) + ppl_cost;
    let n_runs = n_variants * runs_per_variant;
    eprintln!("Plan: {} variants × {} runs each = {} subprocess invocations (~20s each)", n_variants, runs_per_variant, n_runs);
    if dry_run {
        for d in &sweeps {
            let variants = if quick { d.quick } else { d.full };
            eprintln!("  {:<10} -> {}", d.env_key, variants.join(", "));
        }
        eprintln!("\n[dry-run] not executing.");
        return Ok(());
    }
    eprintln!();

    // Optional PPL baseline (computed once per run, used as the reference for tolerance).
    let ppl_baseline = if let Some(ref ppl_file) = validate_ppl {
        eprintln!("Measuring PPL baseline (no variant overrides)...");
        let t = Instant::now();
        let b = run_ppl(&exe, &model, ppl_file, None, None).ok();
        if let Some(v) = b {
            eprintln!("  baseline PPL = {:.4} ({:.1}s)\n", v, t.elapsed().as_secs_f32());
        } else {
            eprintln!("  baseline PPL = FAILED — PPL guard will be disabled for this run\n");
        }
        b
    } else {
        None
    };

    let mut all_results: Vec<VariantResult> = Vec::new();
    let mut winners: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();

    let total_start = Instant::now();
    for d in &sweeps {
        let variants = if quick { d.quick } else { d.full };
        eprintln!("--- {} ({} variants) ---", d.env_key, variants.len());
        let mut local: Vec<(f32, String)> = Vec::new();
        for v in variants {
            // Multi-rep perf sweep.
            let mut samples: Vec<f32> = Vec::with_capacity(reps as usize);
            for rep in 0..reps {
                let run_start = Instant::now();
                let result = run_variant(&exe, &model, d.env_key, v, tokens, metric);
                let elapsed = run_start.elapsed().as_secs_f32();
                match result {
                    Ok(tps) => {
                        if reps > 1 {
                            eprintln!("  {:<10} = {:<12}  rep {}/{}: {:.1} tok/s  ({:.1}s)",
                                d.env_key, v, rep + 1, reps, tps, elapsed);
                        }
                        samples.push(tps);
                    }
                    Err(e) => {
                        eprintln!("  {:<10} = {:<12}  rep {}/{} FAILED ({:.1}s): {}",
                            d.env_key, v, rep + 1, reps, elapsed, e);
                    }
                }
            }

            // Median of successful samples.
            let (median, ok) = if samples.is_empty() { (f32::NAN, false) } else {
                let mut s = samples.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                (s[s.len() / 2], true)
            };

            // Optional PPL guard.
            let (ppl, ppl_disq) = if let (Some(ref ppl_file), Some(base)) = (validate_ppl.as_ref(), ppl_baseline) {
                match run_ppl(&exe, &model, ppl_file, Some(d.env_key), Some(v)) {
                    Ok(p) => {
                        let disq = p > base * (1.0 + ppl_tol);
                        (p, disq)
                    }
                    Err(e) => {
                        eprintln!("  {:<10} = {:<12}  PPL check FAILED: {}", d.env_key, v, e);
                        (f32::NAN, false)
                    }
                }
            } else {
                (f32::NAN, false)
            };

            // Log summary line.
            let ppl_tag = if !ppl.is_nan() {
                if ppl_disq {
                    format!(" PPL={:.4} [DISQUALIFIED]", ppl)
                } else {
                    format!(" PPL={:.4}", ppl)
                }
            } else {
                String::new()
            };
            if ok {
                eprintln!("  {:<10} = {:<12}  median {:>6.1} tok/s  (n={}){}",
                    d.env_key, v, median, samples.len(), ppl_tag);
                if !ppl_disq {
                    local.push((median, v.to_string()));
                }
            } else {
                eprintln!("  {:<10} = {:<12}  ALL REPS FAILED{}", d.env_key, v, ppl_tag);
            }

            all_results.push(VariantResult {
                env_key: d.env_key.to_string(),
                variant: v.to_string(),
                tok_per_sec: median,
                samples,
                ok,
                ppl,
                ppl_disqualified: ppl_disq,
            });
        }
        // Pick winner (highest median tok/s, skipping disqualified variants).
        if let Some((tps, winner)) = local.into_iter().reduce(|a, b| if a.0 >= b.0 { a } else { b }) {
            eprintln!("  → winner: {} = {} ({:.1} tok/s)\n", d.env_key, winner, tps);
            winners.insert(d.env_key.to_string(), winner);
        } else {
            eprintln!("  ! no qualified winners for {} — leaving preset/default\n", d.env_key);
        }
    }

    let total_secs = total_start.elapsed().as_secs_f32();
    eprintln!("=== autotune complete in {:.1}s ===", total_secs);
    eprintln!();
    eprintln!("Winners:");
    for (k, v) in &winners {
        eprintln!("  {}={}", k, v);
    }

    // Write cache.
    let cache = AutotuneCache {
        model_name: model.clone(),
        model_hash: model_hash.clone(),
        arch: arch.clone(),
        hidden_dim,
        sm_major,
        sm_minor,
        tokens_per_run: tokens,
        created_utc: now_iso8601(),
        winners,
        all_results,
    };
    let cache_path = cache_path_for(&arch, hidden_dim, &model_hash, sm_major, sm_minor)?;
    std::fs::create_dir_all(cache_path.parent().unwrap()).ok();
    let json = serde_json::to_string_pretty(&cache).context("serialize cache")?;
    std::fs::write(&cache_path, json).with_context(|| format!("write {}", cache_path.display()))?;
    eprintln!("\nCache written: {}", cache_path.display());
    eprintln!("Next run will apply these winners automatically.");

    Ok(())
}

/// Selects which of Standard / TQ 4-bit / TQ+TriAttn is the primary metric
/// we optimise the variant for. Defaults to Standard (most common perf
/// reporting target; also the one that shows direct kernel improvements).
#[derive(Copy, Clone, Debug)]
enum PrimaryMetric { Standard, Tq4Bit, TqTriAttn }

fn run_variant(exe: &Path, model: &str, env_key: &str, variant: &str, tokens: u32, metric: PrimaryMetric) -> Result<f32> {
    // Run the full 3-way bench so we can see all metrics, but pick the one
    // that matches `metric` as the return value. Slightly slower than
    // --tq-only (extra Std run, ~20s) but gives users truthful winners for
    // their primary workload. TQ-only autotune historically picked DOWN=dp4a
    // which regressed Std -15% — we fix that here by defaulting to Std.
    let output = Command::new(exe)
        .arg("bench")
        .arg(model)
        .arg("-n").arg(tokens.to_string())
        .env(env_key, variant)
        .env("TQ_AUTOTUNE_CHILD", "1")      // suppresses [autocalib] preset log spam
        .output()
        .with_context(|| format!("spawn tq bench for {}={}", env_key, variant))?;

    // tq bench writes lines like:
    //   "  Standard: 68.0 tok/s, 2.94s total, TTFT 0.195s"
    //   "  TQ 4-bit: 36.0 tok/s, 5.55s total, TTFT 0.108s"
    //   "  TQ+TriAttn: 34.4 tok/s, 5.80s total, TTFT 0.107s"
    // We want the value matching `metric`.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let target_prefix = match metric {
        PrimaryMetric::Standard  => "Standard:",
        PrimaryMetric::Tq4Bit    => "TQ 4-bit:",
        PrimaryMetric::TqTriAttn => "TQ+TriAttn:",
    };
    for line in stderr.lines() {
        if line.trim_start().starts_with(target_prefix) {
            if let Some(tps) = parse_tok_per_sec(line) {
                return Ok(tps);
            }
        }
    }
    // Fallback: any tok/s line (useful if user picks metric not emitted).
    for line in stderr.lines() {
        if let Some(tps) = parse_tok_per_sec(line) { return Ok(tps); }
    }
    Err(anyhow!("no '{}' tok/s line found. Child exit: {:?}", target_prefix, output.status))
}

/// Run `tq perplexity --model <m> <file>` optionally with a dispatch env var
/// forced. Returns parsed final PPL value.
fn run_ppl(exe: &Path, model: &str, ppl_file: &Path, env_key: Option<&str>, variant: Option<&str>) -> Result<f32> {
    let mut cmd = Command::new(exe);
    cmd.arg("perplexity")
        .arg("--model").arg(model)
        .arg(ppl_file)
        .env("TQ_AUTOTUNE_CHILD", "1");
    if let (Some(k), Some(v)) = (env_key, variant) {
        cmd.env(k, v);
    }
    let output = cmd.output()
        .with_context(|| format!("spawn tq perplexity for {:?}={:?}", env_key, variant))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Look for "Perplexity: X.XXX" or "Final perplexity: X.XXX".
    for hay in [stderr.as_ref(), stdout.as_ref()] {
        for line in hay.lines() {
            let lower = line.to_ascii_lowercase();
            if let Some(idx) = lower.find("perplexity:") {
                let after = &line[idx + "perplexity:".len()..];
                // Grab first float token.
                let tok = after.split_whitespace().next().unwrap_or("");
                if let Ok(v) = tok.trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.').parse::<f32>() {
                    if v.is_finite() && v > 0.0 { return Ok(v); }
                }
            }
        }
    }
    Err(anyhow!("no 'Perplexity:' line in child output. Exit: {:?}", output.status))
}

fn parse_tok_per_sec(line: &str) -> Option<f32> {
    // matches "...: 36.0 tok/s"
    let trimmed = line.trim();
    let idx = trimmed.find(" tok/s")?;
    // walk back to find the preceding number
    let before = &trimmed[..idx];
    let num_start = before.rfind(|c: char| c.is_whitespace() || c == ':')? + 1;
    before[num_start..].trim().parse::<f32>().ok()
}

fn detect_sm() -> (u8, u8) {
    #[cfg(feature = "cuda")]
    {
        let (maj, min) = crate::cuda::device::compute_capability(0);
        (maj as u8, min as u8)
    }
    #[cfg(not(feature = "cuda"))]
    { (0, 0) }
}

/// Probe GGUF metadata + compute model fingerprint. Returns (hash16hex,
/// arch, hidden_dim). Fingerprint is FNV-1a over (length, first 4 KB) —
/// stable across file moves, changes on requantize. No crypto needed.
fn probe_model(model_name: &str) -> Result<(String, String, usize)> {
    use std::io::Read;

    let (gguf_path, _) = crate::hub::resolve(model_name)
        .with_context(|| format!("resolve model '{}'", model_name))?;
    let len = std::fs::metadata(&gguf_path)?.len();

    // Parse GGUF metadata to get arch + embedding_length.
    let mut reader = std::io::BufReader::new(std::fs::File::open(&gguf_path)?);
    let ct = crate::gguf::GgufContent::read(&mut reader)
        .with_context(|| format!("read GGUF metadata {}", gguf_path.display()))?;
    let arch = ct.metadata.get("general.architecture")
        .and_then(|v| v.to_string_val().ok())
        .unwrap_or_else(|| "llama".to_string());
    let hidden_dim = ct.metadata.get(&format!("{}.embedding_length", arch))
        .and_then(|v| v.to_u32().ok())
        .ok_or_else(|| anyhow!("no {}.embedding_length in metadata", arch))? as usize;

    // Fingerprint over (length, first 4 KB).
    let mut file = std::fs::File::open(&gguf_path)?;
    let head_size = len.min(4096) as usize;
    let mut head = vec![0u8; head_size];
    file.read_exact(&mut head)?;
    let mut h: u64 = 0xcbf29ce484222325;
    for b in len.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for b in head {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    Ok((format!("{:016x}", h), arch, hidden_dim))
}

/// Filename encoding: `autocalib-<arch>-h<hidden_dim>-<hash16>-sm<NN>.json`.
/// - `<arch>`: from GGUF, lowercased (e.g. `qwen2`, `llama`)
/// - `<hidden_dim>`: distinguishes 7B (3584) vs 13B vs 0.5B within same arch
/// - `<hash16>`: FNV-1a fingerprint, first 16 hex chars, for exact file match
/// - `<NN>`: sm_major*10 + sm_minor (e.g. 86 for sm_8.6)
///
/// The autocalib read path filters filenames on (arch, hidden_dim, sm) first
/// and only falls back to mtime when multiple caches survive. This prevents
/// the "newest overwrites all" bug that would hit a user who autotunes
/// Qwen2 then Llama3 on the same machine.
fn cache_path_for(arch: &str, hidden_dim: usize, model_hash: &str, sm_major: u8, sm_minor: u8) -> Result<PathBuf> {
    let base = dirs_cache_tq()?;
    Ok(base.join(format!(
        "autocalib-{}-h{}-{}-sm{}{}.json",
        arch.to_ascii_lowercase(),
        hidden_dim,
        &model_hash[..16],
        sm_major,
        sm_minor
    )))
}

pub fn dirs_cache_tq() -> Result<PathBuf> {
    // Follow the same convention as HuggingFace / other CLI tools:
    // XDG_CACHE_HOME/tq on Linux, %LOCALAPPDATA%\tq\cache on Windows,
    // ~/Library/Caches/tq on macOS. We fall back to ~/.cache/tq.
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg).join("tq"));
    }
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return Ok(PathBuf::from(local).join("tq").join("cache"));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(home).join(".cache").join("tq"));
    }
    if let Ok(up) = std::env::var("USERPROFILE") {
        return Ok(PathBuf::from(up).join(".cache").join("tq"));
    }
    anyhow::bail!("cannot determine cache directory (no HOME/USERPROFILE/XDG_CACHE_HOME)");
}

fn now_iso8601() -> String {
    // Avoid pulling chrono — use std::time with a tiny formatter.
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Very crude Y-M-D HH:MM:SS conversion via days-from-epoch.
    // Good enough for a debug timestamp in a cache file.
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    let day = secs / 86400;       // days since 1970-01-01 UTC
    let (y, mo, dy) = days_to_ymd(day as i64);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, dy, h, m, s)
}

fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    days += 719468;
    let era = if days >= 0 { days / 146097 } else { (days - 146096) / 146097 };
    let doe = (days - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tok_per_sec_from_bench_line() {
        let line = "  TQ 4-bit: 36.0 tok/s, 5.55s total, TTFT 0.108s";
        assert_eq!(parse_tok_per_sec(line), Some(36.0));
    }

    #[test]
    fn parse_tok_per_sec_standard() {
        let line = "  Standard: 68.0 tok/s, 2.94s total, TTFT 0.195s";
        assert_eq!(parse_tok_per_sec(line), Some(68.0));
    }

    #[test]
    fn parse_tok_per_sec_ignores_unrelated() {
        assert_eq!(parse_tok_per_sec("Loading model..."), None);
        assert_eq!(parse_tok_per_sec(""), None);
    }

    #[test]
    fn cache_filename_format() {
        let p = cache_path_for("qwen2", 3584, "deadbeefcafef00d", 8, 6)
            .expect("cache path constructed");
        let name = p.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, "autocalib-qwen2-h3584-deadbeefcafef00d-sm86.json");
    }

    #[test]
    fn cache_filename_lowercases_arch() {
        let p = cache_path_for("QWEN2", 3584, "deadbeefcafef00d", 8, 6).unwrap();
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("autocalib-qwen2-"));
    }
}
