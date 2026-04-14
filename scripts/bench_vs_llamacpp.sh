#!/usr/bin/env bash
# bench_vs_llamacpp.sh — Apples-to-apples head-to-head runner.
#
# Runs the competitor llama.cpp fork (with turbo3/turbo4 KV) and our tq bench
# back-to-back on the same GGUF. Prints a comparison markdown table and appends
# it to docs/bench-history/<ISO-date>.md (gitignored).
#
# Usage:
#   ./scripts/bench_vs_llamacpp.sh qwen2:7b
#   ./scripts/bench_vs_llamacpp.sh llama:8b 3   # override repetitions
#
# Guardrails:
#   - Refuses to run if another llama-bench / tq.exe bench / llama-cli is live.
#   - Never launches in parallel with another GPU workload (feedback:no_parallel_gpu).
#
# Notes:
#   - llama-bench output is parsed from `-o csv`. We pick `avg_ts` (tok/s) for -n rows.
#   - tq bench output is parsed from stderr lines "Standard:", "TQ 4-bit:", "TQ+TriAttn:".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=./bench_paths.sh
source "$SCRIPT_DIR/bench_paths.sh"

MODEL="${1:-}"
REPS="${2:-3}"
NTOK="${NTOK:-100}"

LLAMA_BENCH="E:/Onur/BHI/external/llama-tq/build-cuda/bin/llama-bench.exe"
TQ_BIN="$REPO_ROOT/target/release/tq.exe"

if [[ -z "$MODEL" ]]; then
  echo "usage: $0 <model-shortname> [reps=3]"
  echo "known models: ${!GGUF_PATHS[*]}"
  exit 2
fi

GGUF="${GGUF_PATHS[$MODEL]:-}"
if [[ -z "$GGUF" ]]; then
  echo "ERROR: unknown model '$MODEL'. Add it to scripts/bench_paths.sh." >&2
  echo "known: ${!GGUF_PATHS[*]}" >&2
  exit 2
fi
if [[ ! -f "$GGUF" ]]; then
  echo "ERROR: GGUF not found: $GGUF" >&2
  exit 2
fi
if [[ ! -x "$LLAMA_BENCH" ]]; then
  echo "ERROR: llama-bench missing: $LLAMA_BENCH" >&2
  exit 2
fi
if [[ ! -x "$TQ_BIN" ]]; then
  echo "ERROR: tq.exe not built. Run: cargo build --release --features cuda" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Safeguard: refuse to run alongside any other GPU workload
# ---------------------------------------------------------------------------
check_no_gpu_conflict() {
  local running
  running="$(tasklist.exe 2>/dev/null | tr -d '\r' || true)"
  local conflict=0
  for proc in "llama-bench.exe" "llama-cli.exe" "llama-perplexity.exe" "tq.exe"; do
    if echo "$running" | grep -qi "^$proc"; then
      echo "ERROR: conflicting GPU process already running: $proc" >&2
      conflict=1
    fi
  done
  if [[ $conflict -ne 0 ]]; then
    echo "Refusing to launch (feedback:no_parallel_gpu)." >&2
    exit 1
  fi
}
check_no_gpu_conflict

DATE="$(date +%Y-%m-%d)"
TS="$(date +%H:%M:%S)"
HIST_DIR="$REPO_ROOT/docs/bench-history"
HIST_FILE="$HIST_DIR/${DATE}.md"
mkdir -p "$HIST_DIR"

echo "=== Head-to-Head: $MODEL ==="
echo "GGUF:  $GGUF"
echo "Reps:  $REPS     n-tokens: $NTOK"
echo "Date:  $DATE $TS"
echo ""

# ---------------------------------------------------------------------------
# llama.cpp runs — F16 / q8_0 / turbo3 / turbo4. Missing support = N/A.
# ---------------------------------------------------------------------------
run_llama_bench() {
  local label="$1"; shift
  local csv
  local out=""
  # shellcheck disable=SC2086
  if out="$("$LLAMA_BENCH" -m "$GGUF" -ngl 99 -n "$NTOK" -p 0 -r "$REPS" -o csv "$@" 2>&1)"; then
    # Parse last non-header CSV line. Column `avg_ts` is tok/s; `n_gen` > 0 for -n rows.
    csv="$(echo "$out" | grep -E '^"[^"]+",' | tail -n 1 || true)"
    if [[ -z "$csv" ]]; then
      echo "N/A"
      return
    fi
    # Extract avg_ts: llama-bench CSV columns include ...,n_gen,...,avg_ts,stddev_ts
    # Robust approach: grab the second-to-last numeric field.
    local avg
    avg="$(echo "$csv" | awk -F',' '{ gsub(/"/,"",$(NF-1)); print $(NF-1) }')"
    if [[ -z "$avg" ]]; then
      echo "N/A"
    else
      printf "%.2f" "$avg"
    fi
  else
    echo "N/A ($label unsupported)"
  fi
}

echo "--- llama.cpp baselines ---"
LL_F16="$(run_llama_bench "F16")"
echo "  F16 KV:    $LL_F16 tok/s"

LL_Q8="$(run_llama_bench "q8_0" -ctk q8_0 -ctv q8_0)"
echo "  q8_0 KV:   $LL_Q8 tok/s"

LL_TQ3="$(run_llama_bench "turbo3" -ctk turbo3 -ctv turbo3)"
echo "  turbo3 KV: $LL_TQ3 tok/s"

LL_TQ4="$(run_llama_bench "turbo4" -ctk turbo4 -ctv turbo4)"
echo "  turbo4 KV: $LL_TQ4 tok/s"

# Give the driver a breath to release context before we launch tq.
sleep 1

# ---------------------------------------------------------------------------
# tq bench runs — average REPS runs of Std / TQ / TQ+TriAttn
# ---------------------------------------------------------------------------
# Three accumulators
STD_SUM=0; STD_CNT=0
TQ_SUM=0;  TQ_CNT=0
TRI_SUM=0; TRI_CNT=0

echo ""
echo "--- tq bench runs ---"
for ((i=1; i<=REPS; i++)); do
  echo "  run $i/$REPS..."
  # tq prints to stderr; capture both streams.
  out="$("$TQ_BIN" bench "$MODEL" -n "$NTOK" 2>&1 || true)"

  std="$(echo "$out" | grep -E '^\s*Standard:'   | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -n1 | awk '{print $1}')"
  tq="$( echo "$out" | grep -E '^\s*TQ 4-bit:'   | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -n1 | awk '{print $1}')"
  tri="$(echo "$out" | grep -E '^\s*TQ\+TriAttn:' | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -n1 | awk '{print $1}')"

  if [[ -n "$std" ]]; then STD_SUM="$(awk -v a="$STD_SUM" -v b="$std" 'BEGIN{print a+b}')"; STD_CNT=$((STD_CNT+1)); fi
  if [[ -n "$tq"  ]]; then TQ_SUM="$( awk -v a="$TQ_SUM"  -v b="$tq"  'BEGIN{print a+b}')"; TQ_CNT=$((TQ_CNT+1)); fi
  if [[ -n "$tri" ]]; then TRI_SUM="$(awk -v a="$TRI_SUM" -v b="$tri" 'BEGIN{print a+b}')"; TRI_CNT=$((TRI_CNT+1)); fi

  echo "    Std=${std:-N/A}  TQ=${tq:-N/A}  TQ+TriAttn=${tri:-N/A}"
done

avg() {
  local sum="$1" cnt="$2"
  if [[ "$cnt" -eq 0 ]]; then echo "N/A"; else awk -v s="$sum" -v c="$cnt" 'BEGIN{printf "%.2f", s/c}'; fi
}
TQ_STD="$(avg "$STD_SUM" "$STD_CNT")"
TQ_TQ="$( avg "$TQ_SUM"  "$TQ_CNT")"
TQ_TRI="$(avg "$TRI_SUM" "$TRI_CNT")"

# ---------------------------------------------------------------------------
# Gap computation (llama / tq). "—" if either side is N/A.
# ---------------------------------------------------------------------------
gap() {
  local l="$1" t="$2"
  if [[ "$l" == "N/A"* || "$t" == "N/A"* ]]; then
    echo "—"
  else
    awk -v l="$l" -v t="$t" 'BEGIN{ if (t>0) printf "%.2fx", l/t; else print "—" }'
  fi
}

GAP_STD="$(gap "$LL_F16"  "$TQ_STD")"
GAP_Q8="$( gap "$LL_Q8"   "$TQ_STD")"
GAP_TQ3="$(gap "$LL_TQ3"  "$TQ_TQ")"
GAP_TQ4="$(gap "$LL_TQ4"  "$TQ_TQ")"
GAP_TRI="$(gap "$LL_TQ4"  "$TQ_TRI")"

# ---------------------------------------------------------------------------
# Emit markdown table (stdout + append to history)
# ---------------------------------------------------------------------------
TABLE=$(cat <<EOF
### $MODEL — $DATE $TS (reps=$REPS, n=$NTOK)

| Config                      | llama.cpp t/s | tq-kv t/s   | Gap     |
|-----------------------------|---------------|-------------|---------|
| F16 KV / Std                | $LL_F16       | $TQ_STD     | $GAP_STD |
| q8_0 KV / Std               | $LL_Q8        | $TQ_STD     | $GAP_Q8  |
| turbo3 KV / TQ 3-bit        | $LL_TQ3       | $TQ_TQ      | $GAP_TQ3 |
| turbo4 KV / TQ 4-bit        | $LL_TQ4       | $TQ_TQ      | $GAP_TQ4 |
| turbo4 KV / TQ+TriAttn      | $LL_TQ4       | $TQ_TRI     | $GAP_TRI |

GGUF: \`$GGUF\`
EOF
)

echo ""
echo "$TABLE"
echo ""

{
  echo ""
  echo "$TABLE"
} >> "$HIST_FILE"

echo "Appended to: $HIST_FILE"
