#!/usr/bin/env bash
# Interleaved multi-config 35B decode A/B: every rep runs every config back-to-back,
# so VM drift is shared. Usage:
#   REPS=2 N=14 bash tools/ab35.sh "base:" "gpufulllayer:ENGINE_GPU_FULL_LAYER_GRAPH=1" ...
set -uo pipefail
cd "$(dirname "$0")/.."
M35="${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
BIN="${BIN:-build/test_gen.exe}"
N="${N:-14}"
REPS="${REPS:-2}"
TAG="${TAG:-ab}"
OUT="build/ab35-$TAG.log"
: > "$OUT"
echo "== ab35 tag=$TAG bin=$BIN n=$N reps=$REPS ==" | tee -a "$OUT"
i=0
names=()
for spec in "$@"; do
  names[$i]="${spec%%:*}"
  envs[$i]="${spec#*:}"
  i=$((i+1))
done
for r in $(seq 1 "$REPS"); do
  k=0
  for spec in "$@"; do
    nm="${names[$k]}"; en="${envs[$k]}"; k=$((k+1))
    f="build/ab35-$TAG-$nm"
    env ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 $en "$BIN" "$M35" 760 "$N" > "$f.out" 2> "$f.err"
    dec=$(grep -oE 'decode [0-9.]+' "$f.err" | head -1 | cut -d' ' -f2)
    top=$(grep -E '^topk:' "$f.out" | head -1 | sed 's/topk: //;s/(.*)//g' | tr -d ' ')
    echo "rep$r $nm decode=$dec topk=$top" | tee -a "$OUT"
  done
done
echo "== summary ==" | tee -a "$OUT"
for nm in "${names[@]}"; do
  vals=($(grep -oE " $nm decode=[0-9.]+" "$OUT" | grep -oE '[0-9.]+$'))
  med=$(printf '%s\n' "${vals[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1)
  awk -v n="$nm" -v m="$med" -v a="$mn" 'BEGIN {printf "%-22s median %6.2f ms  %6.2f tok/s  (min %.2f)\n", n, m, 1000/m, a}' | tee -a "$OUT"
done
