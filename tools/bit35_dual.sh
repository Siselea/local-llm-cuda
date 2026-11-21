#!/usr/bin/env bash
# Bit-exactness for the 35B CPU-layer work (shared-expert block graph + dual-stream
# overlap): greedy tokens must match in every config (new default / graph off /
# dual-stream off / pre-change binary).
#   bash build/bit35_dual.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
M="${M:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
NEW="${NEW:-./build/test_gen.exe}"
OLD="${OLD:-./build/test_gen_ek_base.exe}"
N="${N:-12}"
run() {  # <bin> <env> -> GEN
  local out
  out=$(eval "ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 $2 \"$1\" \"$M\" 760 \"$N\" --ctx 1024 2>&1")
  GEN=$(printf '%s\n' "$out" | grep -oE '^gen:.*')
}
run "$NEW" ""; g1="$GEN"
run "$NEW" ""; g2="$GEN"
run "$NEW" "ENGINE_NO_CPU_DUAL_STREAM=1"; g3="$GEN"
run "$NEW" "ENGINE_NO_CPU_FFN_GRAPH=1"; g4="$GEN"
run "$NEW" "ENGINE_NO_CPU_DUAL_STREAM=1 ENGINE_NO_CPU_FFN_GRAPH=1"; g5="$GEN"
run "$OLD" ""; g6="$GEN"
echo "gen: ${g1:0:100}"
fail=0
for pair in "2:$g2" "dual-off:$g3" "ffn-graph-off:$g4" "both-off:$g5" "old-binary:$g6"; do
  name="${pair%%:*}"; val="${pair#*:}"
  if [ -z "$val" ]; then echo "EMPTY GEN ($name)"; fail=1;
  elif [ "$val" != "$g1" ]; then echo "GEN MISMATCH ($name)"; fail=1; fi
done
[ -z "$g1" ] && fail=1
[ "$fail" -eq 0 ] && echo "35B CPU-LAYER WORK BIT-EXACT GREEN (6 runs identical)" || echo "35B BIT-EXACT FAILURES"
exit "$fail"
