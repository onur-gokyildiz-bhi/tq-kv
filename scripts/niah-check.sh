#!/bin/bash
# NIAH (Needle-In-A-Haystack) via PPL degradation test.
#
# Strategy: embed a needle fact in long haystack text, followed by
# text that references the needle. If TriAttention evicts the needle,
# the reference section becomes unpredictable → PPL spikes.
#
# Usage: scripts/niah-check.sh [model]
# Requires TQ_MAX_SEQ >= 1024
set -euo pipefail

MODEL="${1:-qwen2:7b}"
TQ_BIN="${TQ_BIN:-cargo run --release --features cuda --bin tq --}"
NIAH_FILE="/tmp/niah_test.txt"

# Build haystack with needle in the middle
FILLER="Modern computing systems continue to evolve rapidly with innovations in hardware architecture, software optimization, and algorithmic design. Researchers explore new approaches to improve efficiency across different platforms and workloads. The interplay between hardware capabilities and software requirements drives progress in the field of computer science and engineering."

{
    # ~300 tokens of filler before needle
    for i in $(seq 1 6); do echo "$FILLER"; done

    # THE NEEDLE
    echo "IMPORTANT: The secret project codename is AURORA-7 and the launch date is March 15."

    # ~300 tokens of filler after needle
    for i in $(seq 1 6); do echo "$FILLER"; done

    # Reference section — depends on the needle being retained
    echo "As mentioned earlier in this document, the project known by its codename AURORA-7 is scheduled for launch. The March 15 date was confirmed by the team."
} > "$NIAH_FILE"

TOKEN_COUNT=$(wc -w "$NIAH_FILE" | awk '{print $1}')
echo "============================================"
echo "  NIAH PPL Test — $MODEL"
echo "  Haystack: ~$TOKEN_COUNT words"
echo "  Needle: 'AURORA-7' at ~50% depth"
echo "============================================"
echo ""

FAIL=0

# Standard baseline
PPL_STD=$(TQ_MAX_SEQ=2048 $TQ_BIN perplexity -m "$MODEL" "$NIAH_FILE" 2>/dev/null | grep "^Perplexity:" | awk '{print $2}')
printf "  %-16s  PPL=%-8s\n" "Standard" "$PPL_STD"

# TQ without TriAttention
PPL_TQ=$(TQ_MAX_SEQ=2048 TQ_TRIATTN=0 $TQ_BIN perplexity -m "$MODEL" --turbo-quant "$NIAH_FILE" 2>/dev/null | grep "^Perplexity:" | awk '{print $2}')
printf "  %-16s  PPL=%-8s\n" "TQ 4-bit" "$PPL_TQ"

# TQ + TriAttention (budget=512 — forces eviction on ~800 token text)
PPL_TA=$(TQ_MAX_SEQ=2048 TQ_TRIATTN_BUDGET=512 $TQ_BIN perplexity -m "$MODEL" --turbo-quant "$NIAH_FILE" 2>/dev/null | grep "^Perplexity:" | awk '{print $2}')
printf "  %-16s  PPL=%-8s  (budget=512)\n" "TQ+TriAttn" "$PPL_TA"

echo ""

# Check: TQ+TriAttn PPL should be within 2x of Standard
# If needle is evicted, PPL on reference section spikes → overall PPL >> 2x
if [ -n "$PPL_STD" ] && [ -n "$PPL_TA" ]; then
    RATIO=$(awk "BEGIN { printf \"%.1f\", $PPL_TA / $PPL_STD }")
    printf "  Ratio TQ+TA/Std: %sx\n" "$RATIO"
    OK=$(awk "BEGIN { print ($PPL_TA / $PPL_STD <= 2.0) ? 1 : 0 }")
    if [ "$OK" = "0" ]; then
        echo "  FAIL: PPL ratio > 2.0x — needle may have been evicted"
        FAIL=1
    fi
fi

echo ""
echo "============================================"
if [ "$FAIL" = "0" ]; then
    echo "  NIAH PASSED"
else
    echo "  NIAH FAILED — check TriAttention eviction policy"
fi
echo "============================================"
exit $FAIL
