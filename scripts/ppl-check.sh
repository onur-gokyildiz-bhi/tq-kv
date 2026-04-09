#!/bin/bash
# PPL regression check — 3 model × 3 mode with strict thresholds.
# Usage: scripts/ppl-check.sh [test_file]
#
# Thresholds: baseline × 1.10 (max 10% headroom).
# Exit 1 on any regression. Designed for CI integration.
set -euo pipefail

PPL_FILE="${1:-/tmp/ppl_bench.txt}"
TQ_BIN="${TQ_BIN:-cargo run --release --features cuda --bin tq --}"

# Create test file if missing
if [ ! -f "$PPL_FILE" ]; then
    echo "Creating test file at $PPL_FILE ..."
    echo "The transformer architecture has fundamentally changed natural language processing. Large language models built on transformers demonstrate remarkable capabilities across tasks including text generation, summarization, question answering, and code synthesis. The key innovation is the self-attention mechanism which allows every token to attend to every other token in the sequence. Training requires massive datasets and significant computational resources. The KV cache optimization stores key-value projections to avoid redundant computation during autoregressive generation." > "$PPL_FILE"
fi

echo "============================================"
echo "  PPL Regression Check"
echo "  File: $PPL_FILE"
echo "============================================"
echo ""

FAIL=0

run_ppl() {
    local model="$1"
    local mode="$2"
    local threshold="$3"
    local extra_env="$4"

    local label
    case "$mode" in
        std)     label="Standard" ;;
        tq)      label="TQ 4-bit" ;;
        tq_ta)   label="TQ+TriAttn" ;;
    esac

    local cmd_args="perplexity -m $model"
    case "$mode" in
        std)     cmd_args="$cmd_args $PPL_FILE" ;;
        tq)      cmd_args="$cmd_args --turbo-quant $PPL_FILE" ;;
        tq_ta)   cmd_args="$cmd_args --turbo-quant $PPL_FILE" ;;
    esac

    local ppl
    if [ "$mode" = "tq" ]; then
        ppl=$(TQ_TRIATTN=0 $extra_env $TQ_BIN $cmd_args 2>/dev/null | grep "^Perplexity:" | awk '{print $2}')
    elif [ "$mode" = "tq_ta" ]; then
        ppl=$($extra_env $TQ_BIN $cmd_args 2>/dev/null | grep "^Perplexity:" | awk '{print $2}')
    else
        ppl=$($extra_env $TQ_BIN $cmd_args 2>/dev/null | grep "^Perplexity:" | awk '{print $2}')
    fi

    if [ -z "$ppl" ]; then
        echo "  FAIL  $model $label: no PPL output"
        FAIL=1
        return
    fi

    local ok
    ok=$(awk "BEGIN { print ($ppl <= $threshold) ? 1 : 0 }")
    if [ "$ok" = "1" ]; then
        printf "  %-12s %-10s %-12s PPL=%-8s threshold=%-8s  OK\n" "$model" "$label" "" "$ppl" "$threshold"
    else
        printf "  %-12s %-10s %-12s PPL=%-8s threshold=%-8s  FAIL\n" "$model" "$label" "" "$ppl" "$threshold"
        FAIL=1
    fi
}

# Thresholds: measured baseline × 1.10 (10% max headroom)
# Baseline (Apr 10, 2026, /tmp/ppl_bench.txt ~110 tokens):
#   Qwen2 7B:  Std=4.136, TQ=4.457, TQ+TA=4.457
#   Llama3 8B: Std=6.274, TQ=6.386, TQ+TA=6.386
#   Mistral 7B: Std=4.828, TQ=5.066, TQ+TA=5.066

echo "--- Qwen2.5 7B ---"
run_ppl "qwen2:7b" "std"   "4.55"  ""
run_ppl "qwen2:7b" "tq"    "4.90"  ""
run_ppl "qwen2:7b" "tq_ta" "4.90"  ""
echo ""

echo "--- Llama 3.1 8B ---"
run_ppl "llama:8b" "std"   "6.90" ""
run_ppl "llama:8b" "tq"    "7.02" ""
run_ppl "llama:8b" "tq_ta" "7.02" ""
echo ""

echo "--- Mistral 7B ---"
run_ppl "mistral:7b" "std"   "5.31" ""
run_ppl "mistral:7b" "tq"    "5.57" ""
run_ppl "mistral:7b" "tq_ta" "5.57" ""
echo ""

echo "============================================"
if [ "$FAIL" = "0" ]; then
    echo "  ALL PASSED"
    echo "============================================"
    exit 0
else
    echo "  REGRESSION DETECTED"
    echo "============================================"
    exit 1
fi
