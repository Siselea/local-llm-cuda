#!/usr/bin/env bash
# Generality checks (not part of tools/verify.sh, which stays at 51 items):
#   - unified configuration: priority CLI > config file > env > default, errors
#   - kernel calibration cache: generate -> read back -> shapes hit
#   - adaptive VRAM placement: margin sweep changes the layer count monotonically
#   - graceful failure: unsupported arch / quant type / tensor naming
#   - engine run subcommand + info validation on the local models
#
# Usage: bash tools/verify_generality.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ENGINE=./build/engine.exe
GEN=./build/test_gen.exe
PY="${PY:-python}"
TMP=build/generality-tmp
mkdir -p "$TMP"
pass=0
fail=0
note() { printf '%-58s %s\n' "$1" "$2"; }
ok() { pass=$((pass + 1)); note "$1" "PASS"; }
bad() {
  fail=$((fail + 1))
  note "$1" "FAIL"
}
# run <name> <expected substring> <command...>
has() {
  local name="$1" want="$2"
  shift 2
  local out
  out=$(eval "$@" 2>&1)
  if printf '%s' "$out" | grep -qF "$want"; then ok "$name"; else
    bad "$name"
    printf '%s\n' "$out" | tail -4 | sed 's/^/    /'
  fi
}
# Same assertion as has(), but retries once after a short pause.  On this machine the
# previous engine process can still be releasing its VRAM when the next one starts
# (WDDM), which is enough to flip the post-load margin check (headroom is only
# ~170 MiB over the 1200 MiB default margin).  The assertion itself is NOT relaxed:
# the engine must still report "check OK", we just allow the release latency to settle.
has_retry() {
  local name="$1" want="$2"
  shift 2
  local out
  out=$(eval "$@" 2>&1)
  if printf '%s' "$out" | grep -qF "$want"; then ok "$name"; return 0; fi
  sleep 2
  out=$(eval "$@" 2>&1)
  if printf '%s' "$out" | grep -qF "$want"; then
    ok "$name (retry 1/1 due to VRAM release latency)"
  else
    bad "$name (retry 1/1 due to VRAM release latency)"
    printf '%s\n' "$out" | tail -4 | sed 's/^/    /'
    # surface the VRAM ledger so a future flake can be told apart from a real regression
    printf '%s\n' "$out" | grep -E '\[vram\]' | sed 's/^/    /'
  fi
}

fails_with() {
  local name="$1" want="$2"
  shift 2
  local out rc
  out=$(eval "$@" 2>&1)
  rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qF "$want"; then ok "$name"; else
    bad "$name (rc=$rc)"
    printf '%s\n' "$out" | tail -4 | sed 's/^/    /'
  fi
}

[ -x "$ENGINE" ] || { echo "missing $ENGINE (run bash build.sh)"; exit 1; }
[ -x "$GEN" ] || { echo "missing $GEN (run bash build.sh)"; exit 1; }

echo "== configuration layer =="
has "print-config works" "effective configuration" "$ENGINE --print-config"
has "env layer visible" "111" "ENGINE_CTX=111 $ENGINE --print-config"
printf '[engine]\nctx = 333\nvram_margin_mb = 1500\nno_graph = true\n\n[kernel]\nQ6_K 100x200 = q6k_dp4a\n' \
  > "$TMP/test.cfg"
has "config file beats env" "333" \
  "ENGINE_CTX=111 $ENGINE --config $TMP/test.cfg --print-config"
has "config source shown" "[config]" \
  "ENGINE_CTX=111 $ENGINE --config $TMP/test.cfg --print-config"
has "cli beats config" "444" \
  "ENGINE_CTX=111 $ENGINE --config $TMP/test.cfg --ctx 444 --print-config"
has "kernel section parsed" "q6k_dp4a" \
  "$ENGINE --config $TMP/test.cfg --print-config"
printf 'ctx = abc\n' > "$TMP/bad.cfg"
fails_with "invalid int rejected" "expects an integer" \
  "$ENGINE --config $TMP/bad.cfg --print-config"
# NOTE: ctx = 0 is a documented valid value since the adaptive-ctx work ("0 = derive
# the limit from scratch"), so the out-of-range probe uses a value above the real
# upper bound (1 << 20) to keep the intent of the check.
printf 'ctx = 99999999\n' > "$TMP/range.cfg"
fails_with "out-of-range rejected" "out of range" \
  "$ENGINE --config $TMP/range.cfg --print-config"
printf 'nonexistent_key = 1\n' > "$TMP/unknown.cfg"
fails_with "unknown key rejected" "unknown config key" \
  "$ENGINE --config $TMP/unknown.cfg --print-config"
fails_with "unknown option rejected" "unknown option" "$ENGINE info --definitely-not-an-option"

echo
echo "== graceful failure =="
if [ ! -f build/fixtures/tiny_gguf_moe/model.gguf ]; then
  echo "  (tiny_gguf_moe fixture missing: run tools/verify.sh once)"
else
  [ -d build/fixtures/bad_models ] || $PY tools/make_bad_models.py >/dev/null
  has "unsupported arch explains families" "understand" \
    "$ENGINE info build/fixtures/bad_models/arch_mamba.gguf"
  has "unsupported quant lists supported" "unsupported quant type(s): Q4_0" \
    "$ENGINE info build/fixtures/bad_models/quant_q4_0.gguf"
  has "unknown layout names what is missing" "no embedding tensor recognized" \
    "$ENGINE info build/fixtures/bad_models/name_unknown.gguf"
  fails_with "loader refuses bad layout" "IR validation failed" \
    "$GEN build/fixtures/bad_models/name_unknown.gguf 1 1"
  has "info reports VALIDATION FAIL" "VALIDATION : FAIL" \
    "$ENGINE info build/fixtures/bad_models/quant_q4_0.gguf"
fi

echo
echo "== local models still parse + validate =="
for m in "${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
         "${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf" \
         "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" \
         "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"; do
  [ -f "$m" ] || continue
  has "info $(basename "$m")" "VALIDATION : PASS" "$ENGINE info '$m'"
done

echo
echo "== kernel calibration (--kernel-calibrate) =="
M_QWEN="${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
if [ ! -f "$M_QWEN" ]; then
  echo "  (qwen2.5-0.5b missing, skipped)"
else
  CACHE="$TMP/kernel-cache.txt"
  rm -f "$CACHE"
  out=$($GEN --model "$M_QWEN" --ids 1000 --n 1 --kernel-calibrate \
        --kernel-cache "$CACHE" --calib-iters 10 2>&1)
  if [ -f "$CACHE" ] && grep -q '^\[kernel\]' "$CACHE" && printf '%s' "$out" | grep -q 'wrote'; then
    ok "calibration writes cache ($(grep -c ' = ' "$CACHE") shapes)"
  else
    bad "calibration writes cache"
    printf '%s\n' "$out" | tail -5 | sed 's/^/    /'
  fi
  has "cache is loaded back" "kernel calibration cache" \
    "$GEN --model '$M_QWEN' --kernel-cache '$CACHE' --print-config"
  has "cache overrides dispatch" "calibrated override" \
    "RUNNER_DEBUG=1 ENGINE_TOPK=1 $GEN --model '$M_QWEN' --ids 1000,1001 --n 3 --kernel-cache '$CACHE'"
  has "cached run keeps prediction" "40716" \
    "ENGINE_TOPK=1 $GEN --model '$M_QWEN' --ids 1000,1001 --n 33 --kernel-cache '$CACHE'"
fi

echo
echo "== adaptive VRAM placement (margin sweep) =="
M_CODER="${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
if [ ! -f "$M_CODER" ]; then
  echo "  (Coder-V2-Lite missing, skipped)"
else
  layers_of() {
    $GEN --model "$M_CODER" --ids 549,6077 --n 1 --vram-margin "$1" 2>&1 |
      grep -oE 'gpu_moe_layers=[0-9]+' | head -1 | cut -d= -f2
  }
  lo=$(layers_of 600)
  hi=$(layers_of 2400)
  if [ -n "$lo" ] && [ -n "$hi" ] && [ "$lo" -gt "$hi" ] 2>/dev/null; then
    ok "smaller margin keeps more layers (600MiB:$lo > 2400MiB:$hi)"
  else
    bad "margin sweep (600MiB:$lo, 2400MiB:$hi)"
  fi
  # record device VRAM before the check: if this ever flakes again the log shows
  # whether the headroom was thin (driver/previous process) or a real regression.
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  (pre-check GPU mem used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"
  fi
  has_retry "post-load margin enforced" "check OK" \
    "ENGINE_TOPK=1 $GEN --model '$M_CODER' --ids 549,6077,280,7239 --n 2"
  has "explicit gpu-layers honoured" "gpu_moe_layers=8" \
    "ENGINE_TOPK=1 $GEN --model '$M_CODER' --ids 549,6077 --n 1 --gpu-layers 8"
fi

echo
echo "results: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "GENERALITY GREEN" || echo "GENERALITY FAILURES"
exit "$fail"
