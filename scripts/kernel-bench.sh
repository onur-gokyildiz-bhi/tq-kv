#!/bin/bash
# Kernel benchmark — tok/s before/after comparison
# Usage: ! scripts/kernel-bench.sh [n_tokens]
set -e

N="${1:-50}"
TQ_BIN="./target/release/tq-kv-drop-candle"

echo "=== Building ==="
cargo build --release --features cuda 2>&1 | tail -3

echo ""
echo "=== Standard decode (no TQ) ==="
$TQ_BIN bench qwen2:7b -n "$N" 2>&1 | grep -E "tok/s|speed|tokens"

echo ""
echo "=== TQ decode (budget=128) ==="
TQ_BUDGET=128 $TQ_BIN bench qwen2:7b -n "$N" 2>&1 | grep -E "tok/s|speed|tokens"

echo ""
echo "=== Baseline reference: ==="
echo "  Standard: ~17 tok/s (Qwen2 7B Q4K, RTX 3080)"
echo "  TQ b128:  ~15 tok/s"
