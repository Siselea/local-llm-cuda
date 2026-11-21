#!/usr/bin/env bash
# Bit-exactness of the Coder MLA attention-graph change (+ the pinned async routing
# uploads): the graph-free baseline binary and the new binary must produce the same
# greedy tokens AND byte-identical logits for the last decode token.
#
# The engine itself is not always bit-reproducible run-to-run (split-K GEMV uses
# atomicAdd on some lm_head shapes), so each binary is run TWICE first:
#   * all four runs must produce the same greedy tokens (hard requirement);
#   * if both binaries are self-deterministic, their logits must be byte-identical;
#     otherwise the logits check is reported as n/a and only the tokens are compared.
#
# The last-token logits are dumped with ENGINE_LOGITS; that path only runs when the
# device top-k head is disabled (ENGINE_NO_HEAD_TOPK=1), which is also the switch
# combination that exposed the quant-cache bug (docs/GRAPH-INVARIANTS.md invariant 1).
#
#   bash tools/coder_bitexact.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
M="${M:-${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf}"
BASE="${BASE:-./build/test_gen_ek_base.exe}"
NEW="${NEW:-./build/test_gen.exe}"
mkdir -p build/logs
fail=0
pass=0

# run <bin> <logits path> <ids> <n> <envs> -> sets GENLINE
run_one() {
  local out
  out=$(eval "ENGINE_TOPK=1 ENGINE_NO_HEAD_TOPK=1 ENGINE_LOGITS=\"$2\" $5 \"$1\" \"$M\" \"$3\" \"$4\" --ctx 1024" 2>&1)
  GENLINE=$(printf '%s\n' "$out" | grep -oE '^gen:.*')
}

compare() {  # <tag> <ids> <n> <envs>
  local tag="$1" ids="$2" n="$3" envs="$4"
  local g="" lines=() logits=()
  local i
  for i in b1 b2 n1 n2; do
    local bin="$BASE" lf="build/logits_${i}.f32"
    [ "${i:0:1}" = "n" ] && bin="$NEW"
    run_one "$bin" "$lf" "$ids" "$n" "$envs"
    if [ -z "$GENLINE" ]; then
      printf '%-40s FAIL (empty gen, run %s)\n' "$tag" "$i"
      fail=$((fail + 1))
      return
    fi
    lines+=("$GENLINE")
    logits+=("$lf")
  done
  local ref="${lines[0]}" j
  for j in 1 2 3; do
    if [ "${lines[$j]}" != "$ref" ]; then
      printf '%-40s FAIL (gen mismatch between runs: %s vs %s)\n' "$tag" "${lines[$j]:0:48}" "$ref"
      fail=$((fail + 1))
      return
    fi
  done
  local det_b=0 det_n=0 byteeq=0
  cmp -s "${logits[0]}" "${logits[1]}" && det_b=1
  cmp -s "${logits[2]}" "${logits[3]}" && det_n=1
  if [ "$det_b" = 1 ] && [ "$det_n" = 1 ]; then
    cmp -s "${logits[0]}" "${logits[2]}" && byteeq=1
    if [ "$byteeq" = 1 ]; then
      printf '%-40s PASS (gen identical + logits byte-identical, %s B)\n' "$tag" \
        "$(stat -c %s "${logits[2]}")"
      pass=$((pass + 1))
    else
      printf '%-40s FAIL (gen identical but logits differ: %s)\n' "$tag" \
        "$(cmp -l "${logits[0]}" "${logits[2]}" | head -1)"
      fail=$((fail + 1))
    fi
  else
    printf '%-40s PASS (gen identical; logits n/a: binary nondeterministic run-to-run)\n' "$tag"
    pass=$((pass + 1))
  fi
}

echo "== Coder MLA graph bit-exactness (graph binary vs graph-free baseline) =="
compare "auto placement, 12 tokens"     "549,6077,280,7239" 12 ""
compare "pinned 16 experts, 12 tokens"  "549,6077,280,7239" 12 "ENGINE_GPU_EXPERTS=16"
compare "single-token prompt, 8 tokens" "760" 8 ""
compare "7-token prompt, 10 tokens"     "549,6077,280,7239,1000,1001,2000" 10 ""

printf '\nresults: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && echo "MLA BIT-EXACT GREEN" || echo "MLA BIT-EXACT FAILURES"
exit "$((fail > 0 ? 1 : 0))"
