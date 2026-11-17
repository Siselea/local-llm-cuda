#!/usr/bin/env bash
# Interleaved A/B: pre-change binary vs current binary, same model/ids/n.
set -uo pipefail
cd "$(dirname "$0")/.."
M="${M:?model}"
IDS="${IDS:-760}"
N="${N:-9}"
REPS="${REPS:-5}"
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }
ab=(); bb=(); at=(); bt=()
for r in $(seq 1 "$REPS"); do
  a=$(ENGINE_TOPK=1 ./build/test_gen_pre.exe "$M" "$IDS" "$N" 2>&1)
  b=$(ENGINE_TOPK=1 ./build/test_gen.exe     "$M" "$IDS" "$N" 2>&1)
  av=$(printf '%s\n' "$a" | dec); bv=$(printf '%s\n' "$b" | dec)
  at+=("$(printf '%s\n' "$a" | grep -oE 'topk: [0-9]+' | head -1)")
  bt+=("$(printf '%s\n' "$b" | grep -oE 'topk: [0-9]+' | head -1)")
  ab+=("$av"); bb+=("$bv")
  printf 'rep%d pre=%s new=%s\n' "$r" "$av" "$bv"
done
amed=$(printf '%s\n' "${ab[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
bmed=$(printf '%s\n' "${bb[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
printf 'median pre=%s new=%s\n' "$amed" "$bmed"
awk -v a="$amed" -v b="$bmed" 'BEGIN {printf "delta %+.3f ms (%+.2f%%), tok/s %.2f -> %.2f\n", b-a, 100*(b-a)/a, 1000/a, 1000/b}'
printf 'topk pre: %s\n' "${at[*]}"
printf 'topk new: %s\n' "${bt[*]}"
