#!/usr/bin/env bash
# Launch-count comparison for the Coder MLA attention-graph change.
# Uses ENGINE_PIPE_PROF (which prints the cumulative launch counter once per
# run_pipeline call) and reports the steady-state delta between two decode tokens.
#   BIN=./build/test_gen.exe bash tools/mla_launch_count.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
M="${M:-${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf}"
BIN="${BIN:-./build/test_gen.exe}"
log=$(mktemp)
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_PIPE_PROF=1 \
  "$BIN" "$M" 549,6077,280,7239 6 >"$log" 2>&1
# Every decode token prints "[pipe] launches <n> = ..." (cumulative).  Take the last
# few deltas; they are the steady-state per-token kernel launch count.
grep -oE '^\[pipe\] launches [0-9]+' "$log" | awk '{print $3}' | tail -6 | \
  awk 'NR>1 {printf "token %d: %d launches\n", NR-1, $1-prev} {prev=$1}'
rm -f "$log"
