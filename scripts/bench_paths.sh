#!/usr/bin/env bash
# bench_paths.sh — GGUF path mapping for head-to-head bench runner.
# Sourced by bench_vs_llamacpp.sh. Paths are MSYS/git-bash style (/c/...).
#
# Keep this file in sync with the HuggingFace cache on the dev box. When
# a model moves snapshots, `find ~/.cache/huggingface -name "*Q4_K_M.gguf"`
# is the quickest refresher.

# shellcheck disable=SC2034
declare -A GGUF_PATHS=(
  ["qwen2:7b"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--Qwen2.5-7B-Instruct-GGUF/snapshots/8911e8a47f92bac19d6f5c64a2e2095bd2f7d031/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
  ["qwen2:0.5b"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--Qwen2.5-0.5B-Instruct-GGUF/snapshots/41ba88dbac95fed2528c92514c131d73eb5a174b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
  ["qwen2:32b"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--Qwen2.5-32B-Instruct-GGUF/snapshots/2116cbb385b8ce3a4d28cf3bf1cd2039a55821a6/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
  ["llama:8b"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--Meta-Llama-3-8B-Instruct-GGUF/snapshots/4ebc4aa83d60a5d6f9e1e1e9272a4d6306d770c1/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
  ["llama3.1:8b"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  ["mistral:7b"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--Mistral-7B-Instruct-v0.3-GGUF/snapshots/61fd4167fff3ab01ee1cfe0da183fa27a944db48/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
  ["phi3.5:mini"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--Phi-3.5-mini-instruct-GGUF/snapshots/6d70da17e749a471ccb62ade694486011a75cda3/Phi-3.5-mini-instruct-Q4_K_M.gguf"
  ["gemma2:9b"]="/c/Users/onurg/.cache/huggingface/hub/models--bartowski--gemma-2-9b-it-GGUF/snapshots/d731033f3dc4018261fd39896e50984d398b4ac5/gemma-2-9b-it-Q4_K_M.gguf"
)
