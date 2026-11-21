#!/usr/bin/env bash
# Acceptance for the logits-row cap (bigger prefill chunk, fewer dequant passes):
#   1. interleaved A/B: prefill_logits_rows=0 (old, chunk 1024) vs default (chunk 2048+)
#   2. greedy sequence identity: same prompt, old vs new chunking
#   3. five suites
# Needs an EXCLUSIVE GPU (~20 min).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
export MODEL_DIR
export PY="${PY:-python}"
SUM=/tmp/logitscap_summary.txt
: > "$SUM"
{
  echo "===== logits-row-cap acceptance ====="
  echo "date : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "HEAD : $(git rev-parse --short HEAD) (+worktree changes)"
  echo "gpu  : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
} | tee -a "$SUM"

echo "== 1. interleaved prefill A/B (9B ctx32768, prefix cache off) ==" | tee -a "$SUM"
ab() {  # tag env-var tag port
  local label="$1" envkv="$2" port="$3" cases="$4"
  env $envkv timeout 240 python tools/prefill_cliff_probe.py \
      --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" --ctx 32768 --rows 512 --rows-max 4096 \
      --port "$port" --prefix-cache 0 --tag "$label" --cases "$cases" 2>&1 |
      grep -E "req[0-9]|ERROR" | sed "s/^/$label: /" | tee -a "$SUM"
  grep -E "prefill chunk" "build/prefill_cliff_${label}_32768_${port}.log" 2>/dev/null |
      sed "s/^/  $label /" | tee -a "$SUM"
}
for r in 1 2; do
  ab "old$r" "ENGINE_PREFILL_LOGITS_ROWS=0" $((7130 + r)) "4394,9100,13363"
  ab "new$r" "" $((7140 + r)) "4394,9100,13363"
done

echo "== 2. greedy identity old vs new chunking ==" | tee -a "$SUM"
g_new=$(ENGINE_PREFILL_LOGITS_ROWS=1 ./build/test_gen.exe \
        "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" 760 5 --ctx 4096 --prefill-rows 512 2>&1 |
        grep '^gen:')
g_old=$(ENGINE_PREFILL_LOGITS_ROWS=0 ./build/test_gen.exe \
        "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" 760 5 --ctx 4096 --prefill-rows 512 2>&1 |
        grep '^gen:')
echo "  new: $g_new" | tee -a "$SUM"
echo "  old: $g_old" | tee -a "$SUM"
[ "$g_new" = "$g_old" ] && echo "  IDENTICAL (chunk-independent greedy) ✓" | tee -a "$SUM" ||
  echo "  DIFFERENT (chunk size changes the math at tie level) - reported" | tee -a "$SUM"

run() {  # name timeout script
  local name="$1" tmo="$2" script="$3"
  echo "== 3. $name ==" | tee -a "$SUM"
  timeout "$tmo" bash "$script" > "/tmp/lc_$name.log" 2>&1
  echo "  exit=$?" | tee -a "$SUM"
  grep -E "results:|GREEN|FAIL|expected top-1|ALL GREEN" "/tmp/lc_$name.log" | tail -4 |
      sed 's/^/    /' | tee -a "$SUM"
}
run genreg 600 tools/gen_regression.sh
run smoke 420 tools/smoke_models.sh
run verify 420 tools/verify.sh
run generality 420 tools/verify_generality.sh
run serve 300 tools/verify_serve.sh
echo "DONE" | tee -a "$SUM"
