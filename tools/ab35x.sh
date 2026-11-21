#!/usr/bin/env bash
# Interleaved 35B decode A/B over arbitrary env variants (n=40 -> 38 timed steps).
# usage: bash tools/ab35x.sh "name|ENV1=1 ENV2=2" ...   (empty env = baseline)
set -uo pipefail
cd "$(dirname "$0")/.."
M="${MODEL:-D:/p0/models/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
BIN="${BIN:-build/test_gen.exe}"
N="${N:-40}"
REPS="${REPS:-2}"
GL="${GL:-15}"
TAG="${TAG:-x}"
OUT="build/ab35x-$TAG.log"
: > "$OUT"
echo "== ab35x tag=$TAG bin=$BIN n=$N reps=$REPS gpu-layers=$GL ==" | tee -a "$OUT"
for r in $(seq 1 "$REPS"); do
  for spec in "$@"; do
    nm="${spec%%|*}"; en="${spec#*|}"
    f="build/f-$TAG-$nm"
    env ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 $en "$BIN" "$M" 760 "$N" --gpu-layers "$GL" \
        > "$f.out" 2> "$f.err"
    dec=$(grep -oE 'decode [0-9.]+' "$f.err" | head -1 | cut -d' ' -f2)
    top=$(grep -E '^topk:' "$f.out" | head -1 | sed 's/topk: //;s/(.*)//g;s/ //g')
    gpu=$(grep -oE 'gpu_moe_layers=[0-9]+/[0-9]+' "$f.err" | tail -1)
    echo "rep$r $nm decode=$dec top1=$top $gpu" | tee -a "$OUT"
  done
done
echo "== summary ==" | tee -a "$OUT"
for spec in "$@"; do
  nm="${spec%%|*}"
  vals=($(grep -oE " $nm decode=[0-9.]+" "$OUT" | grep -oE '[0-9.]+$'))
  med=$(printf '%s\n' "${vals[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1)
  awk -v n="$nm" -v m="$med" -v a="$mn" 'BEGIN {printf "%-16s median %6.2f ms  %6.2f tok/s  (min %.2f)\n", n, m, 1000/m, a}' | tee -a "$OUT"
done
