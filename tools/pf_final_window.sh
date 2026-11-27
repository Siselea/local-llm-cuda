#!/usr/bin/env bash
# One-shot final verification for the PREFILL-CLIFF fix, designed for a single
# exclusive-GPU window (~10-12 min).  Records the exact tree state, runs the three
# remaining acceptance suites, then the small-batch/tail curve, and prints one summary.
#
#   bash tools/pf_final_window.sh
# Logs: /tmp/pfw_<name>.log ; summary at the end (also /tmp/pfw_summary.txt)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
export MODEL_DIR
export PY="${PY:-python}"

SUM=/tmp/pfw_summary.txt
: > "$SUM"
{
  echo "===== prefill-cliff final window ====="
  echo "date    : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "HEAD    : $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-70)"
  echo "dirty   : $(git status --porcelain | wc -l) modified/untracked paths"
  echo "gpu     : $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
} | tee -a "$SUM"

run() {   # name timeout command...
  local name="$1" tmo="$2"; shift 2
  echo "== $name ==" 
  timeout "$tmo" "$@" > "/tmp/pfw_$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc" | tee -a "$SUM"
  tail -3 "/tmp/pfw_$name.log" | sed 's/^/    /' | tee -a "$SUM"
}

# 1) the three acceptance suites that were blocked by GPU contention
run genreg  600 bash tools/gen_regression.sh
run gen     420 bash tools/verify_generality.sh
run serve   300 bash tools/verify_serve.sh

# 2) small-batch / tail curve + quant-GEMM comparison (~1.5 min)
run smallbatch 300 bash tools/pf_small_batch.sh

echo "== summary files ==" | tee -a "$SUM"
ls -la /tmp/pfw_*.log | awk '{print "   "$5" "$9}' | tee -a "$SUM"
echo "DONE" | tee -a "$SUM"
