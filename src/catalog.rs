/// Built-in catalog of popular GGUF models.
///
/// Each entry maps a user-friendly name:tag to a HuggingFace repo + filename.

#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub name: &'static str,
    pub tag: &'static str,
    pub display: &'static str,
    pub hf_repo: &'static str,
    pub filename: &'static str,
    pub tokenizer_repo: &'static str,
    pub size_gb: f32,
    pub arch: &'static str,
}

pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        name: "qwen2",
        tag: "7b",
        display: "Qwen2.5 7B Instruct",
        hf_repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct",
        size_gb: 4.7,
        arch: "qwen2",
    },
    CatalogEntry {
        name: "qwen2",
        tag: "72b",
        display: "Qwen2.5 72B Instruct",
        hf_repo: "bartowski/Qwen2.5-72B-Instruct-GGUF",
        filename: "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-72B-Instruct",
        size_gb: 45.0,
        arch: "qwen2",
    },
    CatalogEntry {
        name: "llama",
        tag: "8b",
        display: "Llama 3.1 8B Instruct",
        hf_repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        tokenizer_repo: "NousResearch/Meta-Llama-3.1-8B-Instruct",
        size_gb: 4.9,
        arch: "llama",
    },
    CatalogEntry {
        name: "mistral",
        tag: "7b",
        display: "Mistral 7B Instruct v0.3",
        hf_repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        filename: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        tokenizer_repo: "mistralai/Mistral-7B-Instruct-v0.3",
        size_gb: 4.4,
        arch: "llama",
    },
    // Phi-3.5 removed: head_dim=96 (not power of 2) requires padding to 128,
    // which adds 33% zero coordinates and degrades compression quality significantly.
    // TurboQuant's Hadamard rotation spreads zeros across all dimensions, diluting signal.
    // Will re-add when chunk-based Hadamard (split 96 = 64+32) is implemented.
    CatalogEntry {
        name: "gemma",
        tag: "9b",
        display: "Gemma 2 9B IT",
        hf_repo: "bartowski/gemma-2-9b-it-GGUF",
        filename: "gemma-2-9b-it-Q4_K_M.gguf",
        tokenizer_repo: "google/gemma-2-9b-it",
        size_gb: 5.4,
        arch: "gemma2",
    },
];

/// Find catalog entry by "name:tag" or just "name" (defaults to smallest tag).
///
/// Also supports legacy config.rs names like "qwen72b" and "llama3-8b" by
/// fuzzy matching.
pub fn find(query: &str) -> Option<&'static CatalogEntry> {
    // Try exact "name:tag" match first
    if let Some((name, tag)) = query.split_once(':') {
        return CATALOG.iter().find(|e| e.name == name && e.tag == tag);
    }

    // Try exact name match (pick smallest/first entry for that name)
    if let Some(entry) = CATALOG.iter().find(|e| e.name == query) {
        return Some(entry);
    }

    // Legacy name mapping: "qwen72b" -> qwen2:72b, "llama3-8b" -> llama:8b, etc.
    let lower = query.to_lowercase();
    if lower.contains("qwen") && lower.contains("72") {
        return CATALOG.iter().find(|e| e.name == "qwen2" && e.tag == "72b");
    }
    if lower.contains("qwen") && lower.contains("7") {
        return CATALOG.iter().find(|e| e.name == "qwen2" && e.tag == "7b");
    }
    if lower.contains("llama") {
        return CATALOG.iter().find(|e| e.name == "llama" && e.tag == "8b");
    }
    if lower.contains("mistral") {
        return CATALOG.iter().find(|e| e.name == "mistral" && e.tag == "7b");
    }
    // Phi removed from catalog (head_dim=96 incompatible with TQ)
    if lower.contains("gemma") {
        if lower.contains("9") {
            return CATALOG.iter().find(|e| e.name == "gemma" && e.tag == "9b");
        }
        return CATALOG.iter().find(|e| e.name == "gemma");
    }

    None
}

/// List all available models in the catalog.
pub fn list_available() -> &'static [CatalogEntry] {
    CATALOG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_exact() {
        assert!(find("qwen2:7b").is_some());
        assert_eq!(find("qwen2:7b").unwrap().display, "Qwen2.5 7B Instruct");
    }

    #[test]
    fn test_find_name_only() {
        assert!(find("llama").is_some());
        assert_eq!(find("llama").unwrap().tag, "8b");
    }

    #[test]
    fn test_find_legacy() {
        let entry = find("qwen72b").unwrap();
        assert_eq!(entry.name, "qwen2");
        assert_eq!(entry.tag, "72b");
    }

    #[test]
    fn test_find_missing() {
        assert!(find("nonexistent").is_none());
    }
}
