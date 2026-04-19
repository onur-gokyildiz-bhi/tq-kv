#!/usr/bin/env bash
# bench-long-context.sh — measure tok/s across context lengths.
#
# WHAT WE LEARNED (2026-04-19 qwen2:7b RTX 3080 10GB, n_decode=50):
#
#   Context | Std tok/s | TQ tok/s | TriAttn tok/s | Notes
#   --------|-----------|----------|---------------|----------
#   512     |  44.4     | 18.2     | 11.3          | all ok
#   2K      |  27.1     |  6.7     |  4.7          | all ok
#   8K      |  OOM      | OOM      | OOM           | prefill attn
#   16K     |  OOM      | OOM      | OOM           | matmul alloc
#
# The original hypothesis ("TQ wins at long context because its KV is
# compressed") is REFUTED on this hardware: KV storage isn't the
# bottleneck — prefill attention is. Full-context attention builds an
# N×N score tensor per head per layer; at 8K that's ~3.7 GB per layer,
# overflowing a 10 GB card even before decode starts. Flash-attention
# for PREFILL (we only have flash_decode today) is the real unlock.
#
# This script still has value as a forcing function: it draws the
# attn-prefill VRAM cliff in black and white, and a flash-prefill port
# can be validated against it (the 8K/16K rows flip from OOM to numbers).
#
# Std and TQ are benched in SEPARATE processes so a Std OOM doesn't
# kill the TQ measurement. When Std OOMs we retry with --tq-only so
# TQ numbers still land in the table if attention doesn't choke it too.
#
# Usage:
#   ./scripts/bench-long-context.sh [model]          # default qwen2:7b
#   TQ_BENCH_LOG_DIR=/tmp/tqbench ./scripts/bench-long-context.sh
#   CONTEXTS="512 2K 8K" ./scripts/bench-long-context.sh   # custom brackets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TQ="$REPO_ROOT/target/release/tq.exe"
MODEL="${1:-qwen2:7b}"
N_DECODE="${N_DECODE:-50}"
CONTEXTS="${CONTEXTS:-512 2K 8K 16K}"

if ! [ -x "$TQ" ]; then
    echo "error: $TQ not found — run cargo build --release --features cuda first" >&2
    exit 1
fi

# Parallel-GPU guard: refuse if another tq bench is live. `pgrep` is
# missing on Git-Bash on Windows; fall back to tasklist.
is_parallel() {
    if command -v pgrep >/dev/null 2>&1; then
        pgrep -f "tq.exe bench" | grep -v "$$" >/dev/null 2>&1
    elif command -v tasklist >/dev/null 2>&1; then
        tasklist //FI "IMAGENAME eq tq.exe" 2>/dev/null | grep -qi 'tq.exe'
    else
        return 1
    fi
}
if is_parallel; then
    echo "error: another tq.exe is already running — aborting to avoid OOM" >&2
    exit 1
fi

TMP_DIR="${TQ_BENCH_LOG_DIR:-$(mktemp -d)}"
mkdir -p "$TMP_DIR"
if [ -z "${TQ_BENCH_LOG_DIR:-}" ]; then
    trap 'rm -rf "$TMP_DIR"' EXIT
fi

# ─── Seed passage (~90 Qwen2 tokens per repeat) ────────────────────────
SEED="The transformer architecture has fundamentally changed natural language processing. \
Large language models built on transformers demonstrate remarkable capabilities across \
tasks including text generation, summarization, question answering, and code synthesis. \
The key innovation is the self-attention mechanism which allows every token to attend \
to every other token in the sequence. Training requires massive datasets and compute."

declare -A REPEATS
REPEATS[512]=6
REPEATS[2K]=22
REPEATS[8K]=90
REPEATS[16K]=180
REPEATS[32K]=360

make_prompt() {
    local target_name="$1"
    local reps="${REPEATS[$target_name]:-}"
    if [ -z "$reps" ]; then
        echo "error: unknown context target '$target_name'" >&2
        return 1
    fi
    local out="$TMP_DIR/prompt_${target_name}.txt"
    if [ ! -s "$out" ]; then
        : >"$out"
        for ((i=0; i<reps; i++)); do
            printf '%s\n' "$SEED" >>"$out"
        done
    fi
    echo "$out"
}

# Extract a `Label: VALUE tok/s` match from bench stderr. Returns empty
# string if the label didn't produce a result (e.g. OOM).
# Trailing `|| true` matters: set -eo pipefail is active, and if grep
# finds no match the pipeline returns non-zero and would abort the
# whole script. The OOM path relies on an empty return here.
extract_tok_per_sec() {
    local label="$1"
    local text="$2"
    { printf '%s\n' "$text" | grep -oE "$label: *[0-9.]+" | awk '{print $NF}' | head -1; } || true
}

oom_seen() {
    printf '%s\n' "$1" | grep -qE "CUDA_ERROR_OUT_OF_MEMORY|OOM"
}

# Runs one `tq bench` invocation and stores results in globals:
#   RUN_STD / RUN_TQ / RUN_TRI — "-" if the variant didn't produce a tok/s
#   RUN_STATUS — ok | oom | err
# Avoids <<< heredoc / subshell parsing quirks on Git-Bash.
run_bench_once() {
    local prompt_file="$1"
    local tq_only_flag="$2"
    local tag="$3"
    local out=""
    local rc=0
    # Swallow exit code: set -e would otherwise abort the whole script
    # on the first Std OOM, which is exactly what we need to tolerate.
    set +e
    if [ -n "$tq_only_flag" ]; then
        out=$("$TQ" bench "$MODEL" -n "$N_DECODE" --prompt-file "$prompt_file" "$tq_only_flag" 2>&1)
    else
        out=$("$TQ" bench "$MODEL" -n "$N_DECODE" --prompt-file "$prompt_file" 2>&1)
    fi
    rc=$?
    set -e
    {
        echo "===== $(date -u +%FT%TZ) ctx=$tag tq_only='$tq_only_flag' rc=$rc ====="
        echo "$out"
    } >>"$TMP_DIR/log.txt"
    RUN_STATUS="ok"
    if [ "$rc" -ne 0 ]; then
        if oom_seen "$out"; then RUN_STATUS="oom"; else RUN_STATUS="err"; fi
    fi
    local std tq tri
    std=$(extract_tok_per_sec "Standard" "$out")
    tq=$(extract_tok_per_sec "TQ 4-bit"  "$out")
    tri=$(extract_tok_per_sec "TQ\+TriAttn" "$out")
    RUN_STD="${std:--}"
    RUN_TQ="${tq:--}"
    RUN_TRI="${tri:--}"
}

# ─── Header ────────────────────────────────────────────────────────────
printf '\n## Long-context bench — %s (n_decode=%d, RTX 3080 10GB)\n\n' "$MODEL" "$N_DECODE"
printf '| Context  | Std tok/s | TQ tok/s | TriAttn tok/s | Notes           |\n'
printf '|----------|-----------|----------|---------------|-----------------|\n'

# ─── Runs ──────────────────────────────────────────────────────────────
for tag in $CONTEXTS; do
    prompt_path=$(make_prompt "$tag")
    approx_words=$(wc -w <"$prompt_path")
    echo "[bench] $tag ctx (~${approx_words} words) full run..." >&2
    run_bench_once "$prompt_path" "" "$tag-full"
    std="$RUN_STD"; tq="$RUN_TQ"; tri="$RUN_TRI"; status="$RUN_STATUS"
    notes=""
    if [ "$status" = "oom" ] && [ "$tq" = "-" ]; then
        # Std OOM'd before TQ even ran. Retry with --tq-only so TQ/TriAttn
        # numbers still land in the table.
        echo "[bench] $tag ctx Std OOM, retrying --tq-only..." >&2
        run_bench_once "$prompt_path" "--tq-only" "$tag-tqonly"
        tq="$RUN_TQ"; tri="$RUN_TRI"
        notes="Std OOM"
        if [ "$RUN_STATUS" = "oom" ]; then notes="all OOM"; fi
    elif [ "$status" = "oom" ]; then
        notes="Std OOM"
    elif [ "$status" = "err" ]; then
        notes="run error"
    fi
    printf '| %-8s | %-9s | %-8s | %-13s | %-15s |\n' "$tag" "$std" "$tq" "$tri" "$notes"
done

echo ""
echo "Raw log: $TMP_DIR/log.txt"
