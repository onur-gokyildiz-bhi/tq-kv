//! CLI subcommand implementations split per command.
//!
//! Each `cmd_*` module owns one subcommand's end-to-end logic.
//! Shared plumbing (TQ config resolution, auto-calibration) lives in `common`.

pub(crate) mod common;

pub(crate) mod ablate;
pub(crate) mod autotune;
pub(crate) mod bench;
pub(crate) mod calibrate;
pub(crate) mod compress;
pub(crate) mod debug;
pub(crate) mod niah;
pub(crate) mod perplexity;

pub(crate) use common::{
    resolve_tq_config, resolve_tq_config_for_model, resolve_tq_config_with_auto,
};
