#!/usr/bin/env bash
# Wait for the shared build/test_gen.exe to be unlocked (other agents run it for the
# qwen35 correctness regression), then build and run the three acceptance suites.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=${LOG:-build/verify_prefill.log}
: > "$LOG"

try_build() {
  # a locked exe makes lld-link fail; retry for up to ~20 minutes
  for i in $(seq 1 60); do
    if bash build.sh >> "$LOG" 2>&1; then
      echo "[verify-prefill] build OK (attempt $i)" >> "$LOG"
      return 0
    fi
    echo "[verify-prefill] build locked, retry $i" >> "$LOG"
    sleep 20
  done
  return 1
}

{
  echo "===== $(date) ====="
  echo "== verify.sh =="
} >> "$LOG"
try_build || { echo "BUILD FAILED (locked)"; exit 2; }
bash tools/verify.sh >> "$LOG" 2>&1
echo "VERIFY_EXIT=$?" >> "$LOG"

echo "== smoke_models.sh ==" >> "$LOG"
bash tools/smoke_models.sh >> "$LOG" 2>&1
echo "SMOKE_EXIT=$?" >> "$LOG"

echo "== verify_generality.sh ==" >> "$LOG"
bash tools/verify_generality.sh >> "$LOG" 2>&1
echo "GENERALITY_EXIT=$?" >> "$LOG"
echo "[verify-prefill] done" >> "$LOG"
