#!/usr/bin/env bash
# 35B decode bench: N gen tokens (DEC_SKIP=2), prints median decode ms/token.
# Usage: TAG=b0 REPS=2 N=14 bash tools/b35.sh [env assignments...]
#   e.g. TAG=x1 REPS=2 bash tools/b35.sh ENGINE_GPU_FULL_LAYER_GRAPH=1
# PWR=1 samples nvidia-smi power/util during the run (use for a dedicated power run).
# NOTE: native exe stderr does not survive a shell pipe in this MSYS env -> files.
set -uo pipefail
cd "$(dirname "$0")/.."
M35="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
BIN="${BIN:-build/test_gen.exe}"
N="${N:-14}"
REPS="${REPS:-2}"
TAG="${TAG:-run}"
PWR="${PWR:-0}"
OUT="build/b35-$TAG.log"
: > "$OUT"
echo "== b35 tag=$TAG bin=$BIN n=$N reps=$REPS pwr=$PWR env=[$*] ==" | tee -a "$OUT"
for r in $(seq 1 "$REPS"); do
  if [ "$PWR" = "1" ]; then
    ( while :; do
        nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null
        sleep 0.05
      done ) > "build/b35-$TAG.pwr" 2>/dev/null & PW=$!
  fi
  env ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 "$@" "$BIN" "$M35" 760 "$N" \
      > "build/b35-$TAG.out" 2> "build/b35-$TAG.err"
  [ "$PWR" = "1" ] && { kill $PW 2>/dev/null; wait $PW 2>/dev/null; }
  dec=$(grep -oE 'decode [0-9.]+' "build/b35-$TAG.err" | head -1 | cut -d' ' -f2)
  top=$(grep -E '^topk:' "build/b35-$TAG.out" | head -1)
  echo "rep$r decode=$dec  $top" | tee -a "$OUT"
done
vals=($(grep -oE 'decode=[0-9.]+' "$OUT" | cut -d= -f2))
med=$(printf '%s\n' "${vals[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1)
mx=$(printf '%s\n' "${vals[@]}" | sort -n | tail -1)
awk -v m="$med" -v a="$mn" -v b="$mx" 'BEGIN {printf "MEDIAN %.2f ms/token = %.2f tok/s (min %.2f max %.2f)\n", m, 1000/m, a, b}'
if [ "$PWR" = "1" ]; then
  awk -F', ' '{if($1+0>p) p=$1+0; if($2+0>u) u=$2+0} END {printf "power max %.1f W, util max %d %%\n", p, u}' "build/b35-$TAG.pwr"
fi
echo TAG=$TAG
