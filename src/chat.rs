/// Chat template selector based on model type.
#[derive(Clone)]
pub enum ChatTemplate {
    Llama3,
    Qwen,
    Phi3,
    Mistral,
    Gemma,
    ChatML,  // Generic fallback — works with most models
}

impl ChatTemplate {
    /// Detect template from model path or architecture.
    pub fn detect(model_hint: &str) -> Self {
        let lower = model_hint.to_lowercase();
        if lower.contains("qwen") {
            ChatTemplate::Qwen
        } else if lower.contains("phi") {
            ChatTemplate::Phi3
        } else if lower.contains("mistral") {
            ChatTemplate::Mistral
        } else if lower.contains("gemma") {
            ChatTemplate::Gemma
        } else if lower.contains("llama") || lower.contains("meta") {
            ChatTemplate::Llama3
        } else {
            // Default to ChatML — most compatible
            ChatTemplate::ChatML
        }
    }
}

/// Format single-turn chat prompt.
pub fn format_chat(template: &ChatTemplate, system_prompt: &str, user_message: &str) -> String {
    match template {
        ChatTemplate::Llama3 => format_llama3(system_prompt, user_message),
        ChatTemplate::Qwen => format_chatml(system_prompt, user_message),  // Qwen uses ChatML
        ChatTemplate::Phi3 => format_phi3(system_prompt, user_message),
        ChatTemplate::Mistral => format_mistral(system_prompt, user_message),
        ChatTemplate::Gemma => format_gemma(system_prompt, user_message),
        ChatTemplate::ChatML => format_chatml(system_prompt, user_message),
    }
}

/// Format multi-turn chat prompt.
pub fn format_multi_turn(
    template: &ChatTemplate,
    system_prompt: &str,
    history: &[(String, String)],
    current_message: &str,
) -> String {
    match template {
        ChatTemplate::Llama3 => format_llama3_multi(system_prompt, history, current_message),
        ChatTemplate::Qwen => format_chatml_multi(system_prompt, history, current_message),
        ChatTemplate::Phi3 => format_phi3_multi(system_prompt, history, current_message),
        ChatTemplate::Mistral => format_mistral_multi(system_prompt, history, current_message),
        ChatTemplate::Gemma => format_gemma_multi(system_prompt, history, current_message),
        ChatTemplate::ChatML => format_chatml_multi(system_prompt, history, current_message),
    }
}

// --- Llama-3 ---
// <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>...

fn format_llama3(system_prompt: &str, user_message: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
         {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
         {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

fn format_llama3_multi(
    system_prompt: &str,
    history: &[(String, String)],
    current_message: &str,
) -> String {
    let mut prompt = format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
         {system_prompt}<|eot_id|>"
    );
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\n\
             {user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\
             {assistant_msg}<|eot_id|>"
        ));
    }
    prompt.push_str(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n\
         {current_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ));
    prompt
}

// --- ChatML (Qwen, gpt-oss, generic) ---
// <|im_start|>system\n{system}<|im_end|>\n...

fn format_chatml(system_prompt: &str, user_message: &str) -> String {
    format!(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n\
         <|im_start|>user\n{user_message}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn format_chatml_multi(
    system_prompt: &str,
    history: &[(String, String)],
    current_message: &str,
) -> String {
    let mut prompt = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!(
            "<|im_start|>user\n{user_msg}<|im_end|>\n\
             <|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
        ));
    }
    prompt.push_str(&format!(
        "<|im_start|>user\n{current_message}<|im_end|>\n\
         <|im_start|>assistant\n"
    ));
    prompt
}

// --- Phi-3 / Phi-3.5 ---
// <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n

fn format_phi3(system_prompt: &str, user_message: &str) -> String {
    format!(
        "<|system|>\n{system_prompt}<|end|>\n\
         <|user|>\n{user_message}<|end|>\n\
         <|assistant|>\n"
    )
}

fn format_phi3_multi(
    system_prompt: &str,
    history: &[(String, String)],
    current_message: &str,
) -> String {
    let mut prompt = format!("<|system|>\n{system_prompt}<|end|>\n");
    for (user_msg, assistant_msg) in history {
        prompt.push_str(&format!(
            "<|user|>\n{user_msg}<|end|>\n\
             <|assistant|>\n{assistant_msg}<|end|>\n"
        ));
    }
    prompt.push_str(&format!(
        "<|user|>\n{current_message}<|end|>\n\
         <|assistant|>\n"
    ));
    prompt
}

// --- Mistral ---
// [INST] {system}\n\n{user} [/INST]

fn format_mistral(system_prompt: &str, user_message: &str) -> String {
    format!("[INST] {system_prompt}\n\n{user_message} [/INST]")
}

fn format_mistral_multi(
    system_prompt: &str,
    history: &[(String, String)],
    current_message: &str,
) -> String {
    let mut prompt = String::new();
    let first_user = if !history.is_empty() {
        &history[0].0
    } else {
        current_message
    };
    prompt.push_str(&format!("[INST] {system_prompt}\n\n{first_user} [/INST]"));

    for (i, (user_msg, assistant_msg)) in history.iter().enumerate() {
        if i == 0 {
            prompt.push_str(&format!(" {assistant_msg}</s>"));
        } else {
            prompt.push_str(&format!(" [INST] {user_msg} [/INST] {assistant_msg}</s>"));
        }
    }
    if !history.is_empty() {
        prompt.push_str(&format!(" [INST] {current_message} [/INST]"));
    }
    prompt
}

// --- Gemma ---
// <start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n

fn format_gemma(system_prompt: &str, user_message: &str) -> String {
    let msg = if system_prompt.is_empty() {
        user_message.to_string()
    } else {
        format!("{system_prompt}\n\n{user_message}")
    };
    format!(
        "<start_of_turn>user\n{msg}<end_of_turn>\n\
         <start_of_turn>model\n"
    )
}

fn format_gemma_multi(
    system_prompt: &str,
    history: &[(String, String)],
    current_message: &str,
) -> String {
    let mut prompt = String::new();
    for (i, (user_msg, assistant_msg)) in history.iter().enumerate() {
        let user_text = if i == 0 && !system_prompt.is_empty() {
            format!("{system_prompt}\n\n{user_msg}")
        } else {
            user_msg.clone()
        };
        prompt.push_str(&format!(
            "<start_of_turn>user\n{user_text}<end_of_turn>\n\
             <start_of_turn>model\n{assistant_msg}<end_of_turn>\n"
        ));
    }
    let current = if history.is_empty() && !system_prompt.is_empty() {
        format!("{system_prompt}\n\n{current_message}")
    } else {
        current_message.to_string()
    };
    prompt.push_str(&format!(
        "<start_of_turn>user\n{current}<end_of_turn>\n\
         <start_of_turn>model\n"
    ));
    prompt
}
