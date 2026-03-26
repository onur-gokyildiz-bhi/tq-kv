/// Chat template selector based on model type.
#[derive(Clone)]
pub enum ChatTemplate {
    Llama3,
    Qwen,
}

impl ChatTemplate {
    /// Detect template from model path.
    pub fn detect(model_path: &str) -> Self {
        let lower = model_path.to_lowercase();
        if lower.contains("qwen") {
            ChatTemplate::Qwen
        } else {
            ChatTemplate::Llama3
        }
    }
}

/// Format single-turn chat prompt.
pub fn format_chat(template: &ChatTemplate, system_prompt: &str, user_message: &str) -> String {
    match template {
        ChatTemplate::Llama3 => format_llama3_chat(system_prompt, user_message),
        ChatTemplate::Qwen => format_qwen_chat(system_prompt, user_message),
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
        ChatTemplate::Llama3 => format_llama3_multi_turn(system_prompt, history, current_message),
        ChatTemplate::Qwen => format_qwen_multi_turn(system_prompt, history, current_message),
    }
}

// --- Llama-3 ---

pub fn format_llama3_chat(system_prompt: &str, user_message: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
         {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
         {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

fn format_llama3_multi_turn(
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

// --- Qwen2.5 ---

fn format_qwen_chat(system_prompt: &str, user_message: &str) -> String {
    format!(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n\
         <|im_start|>user\n{user_message}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn format_qwen_multi_turn(
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
