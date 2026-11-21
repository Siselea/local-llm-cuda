#!/usr/bin/env bash
# Acceptance run for fix(placement): validate post-load free VRAM and calibrate the
# ledger-invisible reserve.  Needs an EXCLUSIVE GPU (~15-20 min).
#
#   1. vram ledger probe on the three auto-placement models (calibration evidence)
#   2. verify_generality THREE times: the flaky check must now pass without relying on
#      the retry path (a retry is reported as "(retry 1/1 ...)" in the log, so grep for
#      that to prove the fix instead of the workaround)
#   3. gen_regression (10/10) and smoke_models (four models)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
export MODEL_DIR
export PY="${PY:-python}"
SUM=/tmp/place_fix_summary.txt
: > "$SUM"
{
  echo "===== placement fix acceptance ====="
  echo "date : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "HEAD : $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-60)"
  echo "gpu  : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
} | tee -a "$SUM"

echo "== 1. vram ledger calibration ==" | tee -a "$SUM"
if [ "${REUSE_LEDGER:-0}" = "1" ]; then
  echo "  (reused from the earlier probe run: Coder gap 2097, 9B 1254, 35B 1130 MiB)" | tee -a "$SUM"
else
  bash tools/vram_ledger_probe.sh 2>&1 | tee /tmp/place_ledger.log | grep -E "^==|run[0-9]" | tee -a "$SUM"
fi

echo "== 1b. decode impact of the reserve (A/B: extra reserve 0 vs 2048) ==" | tee -a "$SUM"
decode_ab() {  # name model
  local name="$1" model="$2"
  [ -f "$model" ] || { echo "  $name: missing, skipped" | tee -a "$SUM"; return; }
  local old new
  old=$(ENGINE_VRAM_EXTRA_RESERVE_MB=0 ENGINE_DEC_SKIP=2 \
        ./build/test_gen.exe "$model" 760 17 2>&1 | grep -oE 'decode [0-9.]+ ms/token' | tail -1)
  new=$(ENGINE_DEC_SKIP=2 \
        ./build/test_gen.exe "$model" 760 17 2>&1 | grep -oE 'decode [0-9.]+ ms/token' | tail -1)
  echo "  $name: reserve0=[$old] reserve2048=[$new]" | tee -a "$SUM"
}
decode_ab "Qwen3.6-35B-A3B" "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
decode_ab "Coder-V2-Lite"  "${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
decode_ab "Qwen3.5-9B"     "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"

for i in 1 2 3; do
  echo "== 2.$i verify_generality (no-retry expectation) ==" | tee -a "$SUM"
  timeout 420 bash tools/verify_generality.sh > "/tmp/place_gen_$i.log" 2>&1
  rc=$?
  echo "  generation $i: exit=$rc" | tee -a "$SUM"
  grep -E "results:|GENERALITY" "/tmp/place_gen_$i.log" | tee -a "$SUM"
  # prove the retry workaround was NOT needed
  if grep -q "(retry 1/1" "/tmp/place_gen_$i.log"; then
    echo "  NOTE: a case was saved by the retry path in run $i" | tee -a "$SUM"
  else
    echo "  clean: no retry was needed" | tee -a "$SUM"
  fi
  grep -B2 -A6 "FAIL" "/tmp/place_gen_$i.log" | head -20 | sed 's/^/    /' | tee -a "$SUM"
done

echo "== 3. gen_regression ==" | tee -a "$SUM"
timeout 600 bash tools/gen_regression.sh > /tmp/place_genreg.log 2>&1
echo "  exit=$?" | tee -a "$SUM"
grep -E "results:|GREEN|FAIL" /tmp/place_genreg.log | tail -3 | sed 's/^/    /' | tee -a "$SUM"

echo "== 4. smoke_models ==" | tee -a "$SUM"
timeout 420 bash tools/smoke_models.sh > /tmp/place_smoke.log 2>&1
echo "  exit=$?" | tee -a "$SUM"
grep -E "40716|185|2614|PASS|FAIL" /tmp/place_smoke.log | tail -6 | sed 's/^/    /' | tee -a "$SUM"

echo "DONE" | tee -a "$SUM"
