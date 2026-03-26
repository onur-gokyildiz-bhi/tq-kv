/// Supported model definitions.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: &'static str,
    pub display_name: &'static str,
    pub hf_repo: &'static str,
    pub gguf_filename: &'static str,
    pub tokenizer_repo: &'static str,
    pub default_gpu_layers: u32,
    pub default_context_size: u32,
}

pub const MODEL_LLAMA3_8B: ModelConfig = ModelConfig {
    name: "llama3-8b",
    display_name: "Llama-3 8B Instruct",
    hf_repo: "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    gguf_filename: "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    tokenizer_repo: "meta-llama/Meta-Llama-3-8B-Instruct",
    default_gpu_layers: 33,
    default_context_size: 4096,
};

pub const MODEL_QWEN25_72B: ModelConfig = ModelConfig {
    name: "qwen72b",
    display_name: "Qwen2.5 72B Instruct",
    hf_repo: "bartowski/Qwen2.5-72B-Instruct-GGUF",
    gguf_filename: "Qwen2.5-72B-Instruct-Q4_K_M-00001-of-00002.gguf",
    tokenizer_repo: "Qwen/Qwen2.5-72B-Instruct",
    default_gpu_layers: 20,
    default_context_size: 4096,
};

pub const MODEL_GEMMA3_4B: ModelConfig = ModelConfig {
    name: "gemma3-4b",
    display_name: "Gemma 3 4B IT",
    hf_repo: "bartowski/gemma-3-4b-it-GGUF",
    gguf_filename: "gemma-3-4b-it-Q4_K_M.gguf",
    tokenizer_repo: "google/gemma-3-4b-it",
    default_gpu_layers: 33,
    default_context_size: 4096,
};

pub const ALL_MODELS: &[ModelConfig] = &[MODEL_LLAMA3_8B, MODEL_QWEN25_72B, MODEL_GEMMA3_4B];

pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Answer the user's questions clearly and concisely.";

pub fn get_model(name: &str) -> Option<&'static ModelConfig> {
    ALL_MODELS.iter().find(|m| m.name == name)
}

/// Detect model architecture from filename.
pub enum ModelArch {
    Llama,
    Qwen2,
}

pub fn detect_arch(model_name: &str) -> ModelArch {
    let lower = model_name.to_lowercase();
    if lower.contains("qwen") {
        ModelArch::Qwen2
    } else {
        ModelArch::Llama
    }
}
