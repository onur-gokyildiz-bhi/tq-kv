#!/bin/bash
# Model compatibility matrix — PPL + tok/s for all supported models
# Usage: ! scripts/tq-status.sh [ppl_file]
set -e

PPL_FILE="${1:-/tmp/ppl_short.txt}"
TQ_BIN="./target/release/tq-kv-drop-candle"

echo "=== Building ==="
cargo build --release --features cuda 2>&1 | tail -3

echo ""
echo "================================================================"
echo "  Model Compatibility Matrix — $(date '+%Y-%m-%d %H:%M')"
echo "================================================================"

run_test() {
    local model="$1"
    local label="$2"
    local env_prefix="$3"

    echo -n "  $label: "
    if result=$(eval "$env_prefix $TQ_BIN perplexity --model $model $PPL_FILE" 2>&1); then
        ppl=$(echo "$result" | grep -oP 'perplexity[:\s=]+\K[\d.]+' || echo "?")
        toks=$(echo "$result" | grep -oP '[\d.]+\s*tok/s' | head -1 || echo "? tok/s")
        echo "PPL=$ppl  $toks"
    else
        echo "FAILED"
    fi
}

echo ""
echo "--- Qwen2 7B Q4K ---"
run_test "qwen2:7b" "Standard    " ""
run_test "qwen2:7b" "TQ budget=128" "TQ_BUDGET=128"
run_test "qwen2:7b" "TQ budget=64 " "TQ_BUDGET=64"

echo ""
echo "--- Llama3 8B Q4K ---"
run_test "llama3:8b" "Standard    " ""
run_test "llama3:8b" "TQ budget=128" "TQ_BUDGET=128"

echo ""
echo "================================================================"
echo "  Expected baselines:"
echo "    Qwen2 7B: PPL ~7.5 (std), ~8.0-8.5 (TQ b128)"
echo "    Llama3 8B: PPL ~26.5 (std), ~28-30 (TQ b128)"
echo "    PPL > 50 = broken"
echo "================================================================"
