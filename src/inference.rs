use anyhow::Result;
use crate::engine::{Engine, GenerationParams};

/// Generate text with streaming to stdout.
pub fn generate(engine: &mut Engine, prompt: &str, params: &GenerationParams) -> Result<String> {
    engine.generate_streaming(prompt, params)
}

/// Generate text silently (no stdout output).
pub fn generate_silent(engine: &mut Engine, prompt: &str, params: &GenerationParams) -> Result<String> {
    engine.generate_silent(prompt, params)
}
