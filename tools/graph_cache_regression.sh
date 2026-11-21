#!/usr/bin/env bash
# Regression for docs/GRAPH-INVARIANTS.md invariant 1 (a graph replay must invalidate
# host caches that use the launch counter as a validity proof) and for the general
# "graph mode does not change the model output" property.
#
# Same prompt must give the same greedy tokens in every graph/head mode:
#   * default (whole-model graph where available / per-layer graphs)
#   * ENGINE_NO_GRAPH=1              (the --no-graph switch)
#   * ENGINE_NO_HEAD_TOPK=1          (the switch combination that exposed the MLA
#                                     quant-cache bug: it removes the two head launches
#                                     that used to keep cuda::g_launch_calls ahead of
#                                     the stale qc_stamp_)
#   * ENGINE_NO_MLA_ATTN_GRAPH=1     (ignored until the MLA attention graph lands)
#
# Configs are passed as environment assignments (prefix), never as CLI flags.
#
#   MODEL_DIR=D:/p0/models bash tools/graph_cache_regression.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR="${MODEL_DIR:-D:/p0/models}"
GEN=./build/test_gen.exe
[ -x "$GEN" ] || { echo "missing $GEN (run bash build.sh)"; exit 1; }
mkdir -p build/logs
fail=0
pass=0

# run_case <model> <ids> <n> <envs> -> prints the "gen:" line
run_case() {
  local out
  out=$(eval "ENGINE_SKIP_WARMUP=0 $4 \"$GEN\" \"$1\" \"$2\" \"$3\" --ctx 1024" 2>&1)
  printf '%s\n' "$out" | grep -E '^gen:'
}

# check <tag> <model> <ids> <n> <env1> <env2> ... (first config is the reference)
check() {
  local tag="$1" model="$2" ids="$3" n="$4"
  shift 4
  local ref="" bad_cfg="" line="" envs=""
  local first=1
  for envs in "$@"; do
    line=$(run_case "$model" "$ids" "$n" "$envs")
    if [ -z "$line" ]; then
      printf '%-46s FAIL (empty gen for [%s])\n' "$tag" "$envs"
      fail=$((fail + 1))
      return
    fi
    if [ $first -eq 1 ]; then
      ref="$line"; first=0
    elif [ "$line" != "$ref" ]; then
      bad_cfg="$envs"
    fi
    printf '    %-42s %s\n' "[${envs:-default}]" "${line:0:74}"
  done
  if [ -n "$bad_cfg" ]; then
    printf '%-46s FAIL (config [%s] produced a different sequence)\n' "$tag" "$bad_cfg"
    fail=$((fail + 1))
  else
    printf '%-46s PASS (%d configs identical)\n' "$tag" "$#"
    pass=$((pass + 1))
  fi
}

echo "== graph-cache invariant regression =="
check "9B: graph on/off x head topk on/off" \
  "$MODEL_DIR/Qwen3.5-9B-Q5_K_M.gguf" "760" 6 \
  "" "ENGINE_NO_GRAPH=1" "ENGINE_NO_HEAD_TOPK=1" "ENGINE_NO_GRAPH=1 ENGINE_NO_HEAD_TOPK=1"

check "Coder(MLA+MoE): graphs on/off x head topk" \
  "$MODEL_DIR/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf" "549,6077,280,7239" 8 \
  "" "ENGINE_NO_HEAD_TOPK=1" "ENGINE_NO_MLA_ATTN_GRAPH=1 ENGINE_NO_HEAD_TOPK=1" \
  "ENGINE_NO_GRAPH=1 ENGINE_NO_HEAD_TOPK=1"

check "0.5B: graph on/off" \
  "$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf" "760" 10 \
  "" "ENGINE_NO_GRAPH=1"

printf '\nresults: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && echo "GRAPH CACHE INVARIANT GREEN" || echo "GRAPH CACHE INVARIANT FAILURES"
exit "$((fail > 0 ? 1 : 0))"
