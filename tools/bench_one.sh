#!/usr/bin/env bash
# Dedicated per-model decode benchmark: N consecutive runs of ONE model on an idle
# GPU, reporting the median (the VM's per-run spread is ~5 %, and running the four
# models in one batch lets the previous 10 GB process teardown pollute the next one).
#   M=<model> IDS=<ids> N=<gen tokens> REPS=<count> ENVS="K=V ..." bash tools/bench_one.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M="${M:?model path required}"
IDS="${IDS:-760}"
N="${N:-9}"
REPS="${REPS:-5}"
ENVS="${ENVS:-}"
BIN="${BIN:-./build/test_gen.exe}"
vals=()
for i in $(seq 1 "$REPS"); do
  out=$(eval "ENGINE_TOPK=1 $ENVS \"$BIN\" \"$M\" \"$IDS\" \"$N\" 2>&1")
  ms=$(printf '%s\n' "$out" | grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2)
  top=$(printf '%s\n' "$out" | grep -oE 'topk: [0-9]+' | head -1 | cut -d' ' -f2)
  printf 'run %d: %s ms/token  top1=%s\n' "$i" "${ms:-ERR}" "${top:-?}"
  vals+=("${ms:-0}")
done
printf '%s\n' "${vals[@]}" | sort -n | awk -v n="$REPS" '
  {v[NR]=$1}
  END {
    m = (NR % 2) ? v[(NR+1)/2] : (v[NR/2] + v[NR/2+1]) / 2;
    printf "median: %.2f ms/token -> %.2f tok/s (min %.2f max %.2f)\n", m, 1000/m, v[1], v[NR];
  }'
