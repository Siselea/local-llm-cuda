#!/usr/bin/env bash
# Acceptance for the ctx-relative prefill chunk cap:
#   1. ctx 4096 (launcher default): chunk must stay well below ctx, post-load free >=
#      margin, prewarm must actually run, and a 2.4k prompt must prefill normally
#   2. ctx 32768: the long-document gains from the logits cap must still be there
#      (interleaved old/new, old = ENGINE_PREFILL_LOGITS_ROWS=0 -> chunk 1024)
#   3. decode no regression (9B n=17)
#   4. five suites
# Needs an EXCLUSIVE GPU (~18 min).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
export MODEL_DIR
export PY="${PY:-python}"
SUM=/tmp/chunkcap_summary.txt
: > "$SUM"
{
  echo "===== ctx-relative prefill chunk cap acceptance ====="
  echo "date : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "HEAD : $(git rev-parse --short HEAD) (+working tree changes)"
  echo "gpu  : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
} | tee -a "$SUM"

probe() {  # tag ctx port cases env...
  local tag="$1" ctx="$2" port="$3" cases="$4"; shift 4
  env "$@" timeout 240 python tools/prefill_cliff_probe.py \
      --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" --ctx "$ctx" --rows 512 --rows-max 4096 \
      --port "$port" --prefix-cache 0 --tag "$tag" --cases "$cases" 2>&1 |
      grep -E "req[0-9]|ERROR" | sed "s/^/$tag: /" | tee -a "$SUM"
  grep -E "\[ctx\] prefill chunk|prewarm|check OK|ERROR" \
      "build/prefill_cliff_${tag}_${ctx}_${port}.log" 2>/dev/null |
      sed "s/^/  $tag /" | tee -a "$SUM"
}

echo "== 1. ctx 4096 (launcher default) ==" | tee -a "$SUM"
probe cap4096 4096 7161 "2441"

echo "== 2. ctx 32768 long-doc A/B (old = logits_rows=0) ==" | tee -a "$SUM"
for r in 1 2; do
  probe "old$r" 32768 $((7170 + r)) "4394,9100,13363" "ENGINE_PREFILL_LOGITS_ROWS=0"
  probe "new$r" 32768 $((7180 + r)) "4394,9100,13363"
done

echo "== 3. decode no regression (9B n=17) ==" | tee -a "$SUM"
ENGINE_DEC_SKIP=2 ./build/test_gen.exe "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" 760 17 2>&1 |
    grep -E "decode .* ms/token|^gen:" | sed 's/^/  /' | tee -a "$SUM"

run() {  # name timeout script
  local name="$1" tmo="$2" script="$3"
  echo "== 4. $name ==" | tee -a "$SUM"
  timeout "$tmo" bash "$script" > "/tmp/cc_$name.log" 2>&1
  echo "  exit=$?" | tee -a "$SUM"
  grep -E "results:|GREEN|FAIL|expected top-1|ALL GREEN" "/tmp/cc_$name.log" | tail -4 |
      sed 's/^/    /' | tee -a "$SUM"
}
run genreg 600 tools/gen_regression.sh
run smoke 420 tools/smoke_models.sh
run verify 420 tools/verify.sh
run generality 420 tools/verify_generality.sh
run serve 300 tools/verify_serve.sh
echo "DONE" | tee -a "$SUM"
