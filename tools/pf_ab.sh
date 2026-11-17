#!/usr/bin/env bash
# Interleaved A/B benchmark for prefill work.  Waits for a quiet GPU, then runs each
# configuration REPS times in an interleaved order and reports the min (VM/GPU noise
# on this machine is 10-20x when another process loads a model, so min-of-N with
# interleaving is the only trustworthy summary).
#
# usage: bash tools/pf_ab.sh "<model>" "<ids-file>" "<n_gen>" "<ctx>" "<prefill_rows>" \
#            "ENV1=VAL1,ENV2=VAL2" "ENV1=VAL1,..."
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?model}"
IDS="${2:?ids file}"
NGEN="${3:-4}"
CTX="${4:-4096}"
ROWS="${5:-2048}"
shift 5
REPS="${REPS:-3}"
BIN="${BIN:-./build/pf_gen.exe}"

# wait for a quiet GPU: <900 MiB used and <10% util sustained for 3 samples
wait_quiet() {
  for _ in $(seq 1 120); do
    local used util
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if [ "$used" -lt 900 ] && [ "$util" -lt 10 ]; then return 0; fi
    sleep 5
  done
  return 1
}

run_one() {   # $1 = env spec, prints ms
  local spec="$1"
  local out
  out=$(env $spec "$BIN" "$MODEL" "$(cat "$IDS")" "$NGEN" --ctx "$CTX" --prefill-rows "$ROWS" 2>&1)
  echo "$out" | sed -n 's/.*prefill [0-9]* tokens: \([0-9.]*\) ms.*/\1/p'
  echo "$out" | sed -n 's/^gen: /gen: /p' >&2
}

declare -A best
for r in $(seq 1 "$REPS"); do
  for spec in "$@"; do
    wait_quiet || { echo "GPU never went quiet, abort"; exit 2; }
    ms=$(run_one "$spec")
    [ -z "$ms" ] && { echo "[$spec] run failed"; continue; }
    cur="${best[$spec]:-}"
    if [ -z "$cur" ] || awk -v a="$ms" -v b="$cur" 'BEGIN{exit !(a<b)}'; then
      best[$spec]="$ms"
    fi
    echo "rep$r  $spec : ${ms} ms"
  done
done
echo "---- min of ${REPS} (interleaved) ----"
for spec in "$@"; do
  ms="${best[$spec]:-nan}"
  awk -v m="$ms" -v s="$spec" -v t="$IDS" 'BEGIN{printf "%-40s %9.1f ms  %6.0f tok/s\n", s, m, 1622*1000/m}'
done
