#!/usr/bin/env bash
# Acceptance for the x8 dequant kernels (DEQUANT-X8): end-to-end A/B and the suites.
#   1. micro bench: old vs x8 + bit-identical slab check
#   2. real-prefill A/B: 9B, prefix cache off, interleaved ENGINE_DQ_OLD / default
#   3. token equivalence + five suites
# Needs an EXCLUSIVE GPU (~15 min).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
export MODEL_DIR
export PY="${PY:-python}"
SUM=/tmp/dequant_x8_summary.txt
: > "$SUM"
{
  echo "===== x8 dequant acceptance ====="
  echo "date : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "HEAD : $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-60)"
  echo "gpu  : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
} | tee -a "$SUM"

echo "== 1. micro bench ==" | tee -a "$SUM"
./build/test_gen.exe --bench-dequant 2>&1 | tee /tmp/dq_bench.log | tail -5 | tee -a "$SUM"

echo "== 2. real-prefill A/B (9B, ctx 32768, prefix cache off, rows auto) ==" | tee -a "$SUM"
# interleaved: old, new, old, new on the same prompt sizes
for rnd in 1 2; do
  ENGINE_DQ_OLD=1 timeout 240 python tools/prefill_cliff_probe.py \
      --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" --ctx 32768 --rows 0 --rows-max 0 \
      --port $((7091 + rnd)) --prefix-cache 0 --tag "dqold$rnd" --cases 3611,11020 2>&1 |
      grep -E "req[0-9]" | sed "s/^/  old r$rnd: /" | tee -a "$SUM"
  timeout 240 python tools/prefill_cliff_probe.py \
      --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" --ctx 32768 --rows 0 --rows-max 0 \
      --port $((7093 + rnd)) --prefix-cache 0 --tag "dqnew$rnd" --cases 3611,11020 2>&1 |
      grep -E "req[0-9]" | sed "s/^/  new r$rnd: /" | tee -a "$SUM"
done

echo "== 3. token equivalence (must be identical) ==" | tee -a "$SUM"
new_gen=$(./build/test_gen.exe "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" 760 3 --ctx 512 2>&1 | grep '^gen:')
old_gen=$(ENGINE_DQ_OLD=1 ./build/test_gen.exe "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" 760 3 --ctx 512 2>&1 | grep '^gen:')
echo "  new: $new_gen" | tee -a "$SUM"
echo "  old: $old_gen" | tee -a "$SUM"
[ "$new_gen" = "$old_gen" ] && echo "  IDENTICAL ✓" | tee -a "$SUM" ||
  echo "  MISMATCH ✗" | tee -a "$SUM"

run() {   # name timeout script
  local name="$1" tmo="$2" script="$3"
  echo "== 4. $name ==" | tee -a "$SUM"
  timeout "$tmo" bash "$script" > "/tmp/dq_$name.log" 2>&1
  echo "  exit=$?" | tee -a "$SUM"
  grep -E "results:|GREEN|FAIL|expected top-1" "/tmp/dq_$name.log" | tail -4 | sed 's/^/    /' | tee -a "$SUM"
}
run genreg 600 tools/gen_regression.sh
run smoke 420 tools/smoke_models.sh
run verify 420 tools/verify.sh
run generality 420 tools/verify_generality.sh
run serve 300 tools/verify_serve.sh
echo "DONE" | tee -a "$SUM"
