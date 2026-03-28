use anyhow::Result;
use crate::engine::{Engine, GenerationParams};

/// Generate text with streaming to stdout.
pub fn generate(engine: &mut Engine, prompt: &str, params: &GenerationParams) -> Result<String> {
    engine.generate_streaming(prompt, params)
}

