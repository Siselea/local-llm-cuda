#!/usr/bin/env bash
# Final 35B A/B: reference binary (build/test_gen_base.exe) vs current (build/test_gen.exe).
# Alternating order per rep cancels monotonic drift; n=40 gives 38 timed steps.
# Usage: REPS=3 bash tools/ab35_final.sh [extra args passed to both binaries]
set -uo pipefail
cd "$(dirname "$0")/.."
M="${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
A="build/test_gen_base.exe"
B="build/test_gen.exe"
N="${N:-40}"
REPS="${REPS:-3}"
OUT="build/ab35-final.log"
: > "$OUT"
run() { # $1 tag $2 bin $3 extra
  local tag="$1" bin="$2" extra="$3"
  env ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 $extra "$bin" "$M" 760 "$N" \
      > "build/f-$tag.out" 2> "build/f-$tag.err"
  local dec top gen
  dec=$(grep -oE 'decode [0-9.]+' "build/f-$tag.err" | head -1 | cut -d' ' -f2)
  top=$(grep -E '^topk:' "build/f-$tag.out" | head -1 | sed 's/topk: //;s/(.*)//g;s/ //g')
  gen=$(grep -E '^gen:' "build/f-$tag.out" | head -1 | md5sum | cut -c1-8)
  echo "$tag $dec $top $gen"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then order="base new"; else order="new base"; fi
  for nm in $order; do
    extra="${EXTRA:-}"
    if [ "$nm" = "base" ]; then bin="$A"; tag="base-r$r"; else bin="$B"; tag="new-r$r"; fi
    res=$(run "$tag" "$bin" "$extra")
    echo "$res" | tee -a "$OUT"
  done
done
summ() {
  local nm="$1" vals med mn mx
  vals=($(awk -v n="$nm" '$1 ~ "^"n" " {print $2}' "$OUT"))
  [ ${#vals[@]} -eq 0 ] && return
  med=$(printf '%s\n' "${vals[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1)
  mx=$(printf '%s\n' "${vals[@]}" | sort -n | tail -1)
  awk -v n="$nm" -v m="$med" -v a="$mn" -v b="$mx" \
    'BEGIN {printf "%-6s median %6.2f ms  %6.2f tok/s  (min %.2f max %.2f)\n", toupper(n), m, 1000/m, a, b}'
}
echo "== summary ==" | tee -a "$OUT"
{ summ base; summ new; } | tee -a "$OUT"
b=$(awk -v m=0 '/^base/ {s+=$2;n++} END {print (n?s/n:0)}' "$OUT")
n=$(awk -v m=0 '/^new/ {s+=$2;n++} END {print (n?s/n:0)}' "$OUT")
awk -v a="$b" -v b="$n" 'BEGIN {if (a>0) printf "mean base %.2f new %.2f  delta %+.2f ms (%+.1f%%)\n", a, b, b-a, 100*(b-a)/a}'
