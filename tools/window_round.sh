#!/usr/bin/env bash
# One-shot "window round": everything that needs the GPU/quiet machine, in the order
# agreed with the lead.  Stops on the first hard failure (each step logs separately).
#
#   1) bash build.sh                       (full relink; launcher.exe must not be running)
#   2) verify.sh                           (was blocked by a file lock, tests never ran)
#   3) gen_regression.sh                   (must be 10/10; the 35B flake must be retested)
#   4) thread e2e A/B                      (old-9-os vs new-default 12/pair vs 10/pair)
#   5) moe_gap_probe                       (llama-bench -ngl {0,6,15} vs ours {6,15})
#
# Usage: bash tools/window_round.sh [1|2|3|4|5|all]   (default all)
set -uo pipefail
cd "$(dirname "$0")/.."
export MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the directory containing the GGUF files}"
STEP="${1:-all}"
LOG="build/window-round.log"
mkdir -p build
run_step() {
  local n="$1" name="$2"; shift 2
  if [ "$STEP" != "all" ] && [ "$STEP" != "$n" ]; then return 0; fi
  echo "########## [$n] $name  $(date '+%F %T') ##########" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "########## [$n] rc=$rc ##########" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "[window-round] STOP at step $n ($name) rc=$rc; see $LOG" | tee -a "$LOG"
    exit "$rc"
  fi
}

gpu_free_check() {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
  if [ -z "$free" ] || [ "$free" -lt 11000 ]; then
    echo "[window-round] GPU not free (${free:-?} MiB) - another process is using it; aborting" | tee -a "$LOG"
    exit 3
  fi
}

# step 1: full build.  Windows locks the image file of a running exe, so:
#   - a running launcher.exe is renamed aside (the GUI keeps running from the renamed
#     image; the fresh build/launcher.exe is written normally) and a note is logged;
#   - a running engine.exe is NOT touched: abort and tell the operator to close it.
prepare_build_locks() {
  if powershell -NoProfile -Command "if (Get-Process engine -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
    echo "[window-round] engine.exe is RUNNING - close it before building (not touching it)" | tee -a "$LOG"
    exit 2
  fi
  if powershell -NoProfile -Command "if (Get-Process launcher -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
    local n=2
    while [ -e "build/launcher-running-$n.exe" ]; do n=$((n+1)); done
    mv build/launcher.exe "build/launcher-running-$n.exe" 2>/dev/null &&
      echo "[window-round] running launcher.exe renamed to launcher-running-$n.exe (image keeps running)" | tee -a "$LOG"
  fi
}
run_step 0 "prepare build locks" prepare_build_locks
run_step 1 "bash build.sh" bash build.sh

# step 2: verify (build inside verify re-runs build.sh; fine)
run_step 2 "verify.sh" bash tools/verify.sh

# step 3: gen_regression (needs GPU)
gpu_free_check
run_step 3 "gen_regression.sh" bash tools/gen_regression.sh

# step 4: affinity end-to-end (needs GPU; script has its own guards)
gpu_free_check
run_step 4 "thread e2e A/B" env N=20 REPS=2 bash tools/thread_e2e.sh

# step 5: CPU MoE gap probe (llama-bench + ours, needs GPU for -ngl>0)
gpu_free_check
run_step 5 "moe gap probe" bash tools/moe_gap_probe.sh

echo "[window-round] ALL DONE $(date '+%F %T')" | tee -a "$LOG"
