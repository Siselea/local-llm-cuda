#!/usr/bin/env bash
# Bit-exactness for the 35B CPU-layer FFN graph: greedy tokens must match in every
# config (new default / new with the graph off / the pre-change binary), and the
# last-token logits must be byte-identical when both sides are self-deterministic
# (the engine has run-to-run nondeterminism from split-K atomics on some heads).
#
#   bash build/bit35_cpuffn.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
M="${M:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
NEW="${NEW:-./build/test_gen.exe}"
OLD="${OLD:-./build/test_gen_ek_base.exe}"
N="${N:-16}"
mkdir -p build/logs

run() {  # <bin> <env> <logits> -> GEN
  local out
  out=$(eval "ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_NO_HEAD_TOPK=1 ENGINE_LOGITS=\"$3\" $2 \"$1\" \"$M\" 760 \"$N\" --ctx 1024" 2>&1)
  GEN=$(printf '%s\n' "$out" | grep -oE '^gen:.*')
}
L=build/logs/bit35
run "$NEW" ""                       "${L}_new1.f32"
g_new1="$GEN"
run "$NEW" ""                       "${L}_new2.f32"
g_new2="$GEN"
run "$NEW" "ENGINE_NO_CPU_FFN_GRAPH=1" "${L}_off1.f32"
g_off1="$GEN"
run "$NEW" "ENGINE_NO_CPU_FFN_GRAPH=1" "${L}_off2.f32"
g_off2="$GEN"
run "$OLD" ""                       "${L}_old1.f32"
g_old1="$GEN"
run "$OLD" ""                       "${L}_old2.f32"
g_old2="$GEN"

fail=0
for name in new2 off1 off2 old1 old2; do
  eval "g=\$g_$name"
  if [ "$g" != "$g_new1" ]; then
    echo "GEN MISMATCH: $name vs new1"; fail=1
  fi
done
if [ -z "$g_new1" ]; then echo "EMPTY GEN"; fail=1; fi
echo "gen: ${g_new1:0:90}"

det() { cmp -s "$1" "$2"; }
if det "${L}_new1.f32" "${L}_new2.f32"; then det_new=1; else det_new=0; fi
if det "${L}_off1.f32" "${L}_off2.f32"; then det_off=1; else det_off=0; fi
if det "${L}_old1.f32" "${L}_old2.f32"; then det_old=1; else det_old=0; fi
echo "determinism: new=$det_new off=$det_off old=$det_old"
if [ "$det_new" = 1 ] && [ "$det_off" = 1 ]; then
  if det "${L}_new1.f32" "${L}_off1.f32"; then echo "logits new vs graph-off: BYTE-IDENTICAL";
  else echo "LOGITS DIFFER (new vs off): $(cmp -l "${L}_new1.f32" "${L}_off1.f32" | head -1)"; fail=1; fi
else
  echo "logits new vs off: n/a (nondeterministic within a binary)"
fi
if [ "$det_new" = 1 ] && [ "$det_old" = 1 ]; then
  if det "${L}_new1.f32" "${L}_old1.f32"; then echo "logits new vs pre-change binary: BYTE-IDENTICAL";
  else echo "LOGITS DIFFER (new vs old): $(cmp -l "${L}_new1.f32" "${L}_old1.f32" | head -1)"; fail=1; fi
else
  echo "logits new vs old: n/a (nondeterministic within a binary)"
fi
[ "$fail" -eq 0 ] && echo "35B CPU-FFN GRAPH BIT-EXACT GREEN" || echo "35B CPU-FFN GRAPH BIT-EXACT FAILURES"
exit "$fail"
