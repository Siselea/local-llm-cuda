#!/usr/bin/env bash
# Final acceptance for the HTTP service (run after the qwen35 prefill bug fix).
#
#   bash tools/final_acceptance.sh [--with-longctx]
#
# Stages (each prints its own PASS/FAIL and leaves no server running):
#   1. 0.5B  : tools/verify_serve.sh                       (7031, HTTP/分词/采样/SSE/错误码/greedy 对拍)
#   2. 9B    : tools/verify_serve.sh <9B> 7032 SKIP_PARITY=1 (真实中文/多轮/流式，单卡放不下两份 9B)
#   3. 35B   : tools/serve_35b_e2e.sh                      (7030, 中文单轮/多轮/流式/300·500 token/greedy 对拍/优雅退出)
#   4. 4k    : tools/serve_longctx_check.sh <35B> 7034 8192 4000  (--with-longctx)
#
# GPU must be idle; every stage refuses to start when >2 GB is already used.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
M05="<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf"
M9="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf"
M35="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
WITH_LONGCTX=0
[ "${1:-}" = "--with-longctx" ] && WITH_LONGCTX=1

mkdir -p build/logs
declare -a NAMES=()
declare -a RESULTS=()
stage_fail=0

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
wait_gpu_idle() {
  for _ in $(seq 1 300); do
    local u
    u=$(gpu_used)
    if [ -n "$u" ] && [ "$u" -lt 2000 ]; then return 0; fi
    echo "  waiting for GPU to become idle (used=${u:-?} MiB) ..."
    sleep 2
  done
  return 1
}
check_port_free() {
  if curl.exe -s -m 2 -o /dev/null "http://127.0.0.1:$1/health"; then
    echo "  port $1 is already in use"; return 1
  fi
  return 0
}

run_stage() {
  local name="$1"
  shift
  echo
  echo "############################################################################"
  echo "## STAGE: $name"
  echo "############################################################################"
  if ! wait_gpu_idle; then
    echo "## STAGE $name: SKIP (GPU busy)"
    NAMES+=("$name")
    RESULTS+=("SKIP(gpu busy)")
    return
  fi
  local rc=0
  "$@" || rc=$?
  NAMES+=("$name")
  if [ "$rc" -eq 0 ]; then RESULTS+=("PASS"); else RESULTS+=("FAIL(rc=$rc)"); fi
}

[ -x build/engine.exe ] || { echo "missing build/engine.exe (bash build.sh first)"; exit 1; }
[ -x build/test_gen.exe ] || { echo "missing build/test_gen.exe"; exit 1; }
[ -x build/test_tokenizer.exe ] || { echo "missing build/test_tokenizer.exe"; exit 1; }
[ -f "$M05" ] || { echo "missing $M05"; exit 1; }

echo "== final acceptance =="
echo "  GPU used: $(gpu_used) MiB"
if [ "$WITH_LONGCTX" = "1" ]; then
  echo "  including 4k long-context stage"
fi

check_port_free 7031 || exit 1
run_stage "0.5B http e2e (7031)" bash tools/verify_serve.sh

if [ -f "$M9" ]; then
  check_port_free 7032 || true
  run_stage "9B http e2e (7032, no test_gen parity)" \
    env SKIP_PARITY=1 bash tools/verify_serve.sh "$M9" 7032
else
  echo "skip 9B stage (model missing)"
fi

if [ -f "$M35" ]; then
  check_port_free 7030 || true
  run_stage "35B e2e (7030): Chinese / multi-turn / stream / 300+500 tok / parity / shutdown" \
    bash tools/serve_35b_e2e.sh
  if [ "$WITH_LONGCTX" = "1" ]; then
    check_port_free 7034 || true
    run_stage "35B 4k long-context (7034, ctx 8192)" \
      env EXTRA="${LONGCTX_EXTRA:-}" bash tools/serve_longctx_check.sh "$M35" 7034 8192 4000
  fi
else
  echo "skip 35B stage (model missing)"
fi

echo
echo "############################################################################"
echo "## FINAL SUMMARY"
echo "############################################################################"
fail=0
for i in "${!NAMES[@]}"; do
  printf '  %-78s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"
  case "${RESULTS[$i]}" in FAIL*) fail=$((fail + 1)) ;; esac
done
echo
if [ "$fail" -eq 0 ]; then
  echo "FINAL ACCEPTANCE GREEN"
else
  echo "FINAL ACCEPTANCE FAILURES: $fail stage(s)"
fi
echo
echo "leftover model processes (should be empty):"
tasklist 2>/dev/null | grep -i -E "engine.exe|test_gen|llama-server" | head -5 || true
echo "nvidia-smi memory.used = $(gpu_used) MiB"
exit "$fail"
