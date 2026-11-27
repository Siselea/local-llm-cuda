#!/usr/bin/env bash
# Run the remaining acceptance suites with a self-contained environment.
# (Command-prefix env assignments proved unreliable through the ops wrapper, so the
# exports live inside this script and every suite is invoked from the same shell.)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
export MODEL_DIR
export PY="${PY:-python}"
OUT="${OUT:-/tmp}"
echo "[suites] MODEL_DIR=$MODEL_DIR PY=$PY"

run() {   # name timeout script
  local name="$1" tmo="$2" script="$3"
  echo "[suites] == $name ($script) =="
  timeout "$tmo" bash "$script" > "$OUT/pf_$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc"
  echo "${name^^}_EXIT=$rc" >> "$OUT/pf_all_status.txt"
}

: > "$OUT/pf_all_status.txt"
run genreg 600 tools/gen_regression.sh
run gen 420 tools/verify_generality.sh
run serve 300 tools/verify_serve.sh
echo "[suites] ALLDONE"
cat "$OUT/pf_all_status.txt"
