#!/usr/bin/env bash
# MTP speculative decoding A/B: plain greedy vs spec (token identity + speed).
#   M=<model> IDS=<ids> N=<tokens> KS="1 2" bash tools/mtp_spec_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M="${M:?model path required}"
IDS="${IDS:-760}"
N="${N:-12}"
REPS="${REPS:-3}"
KS="${KS:-1 2}"
B=./build/mtp_run.exe
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }
gen() { grep -oE '^gen:.*' | head -1; }

echo "== $M  ids=$IDS n=$N =="
pl=(); ag=()
for r in $(seq 1 "$REPS"); do
  out=$($B --model "$M" --ids "$IDS" --mode plain --n "$N" 2>&1)
  pl+=("$(printf '%s\n' "$out" | dec)")
  ag+=("$(printf '%s\n' "$out" | gen)")
done
pmed=$(med "${pl[@]}")
PTOK=$(printf '%s\n' "${ag[0]}")
printf 'plain : %6.2f ms/token (%5.1f tok/s)  %s\n' "$pmed" "$(awk -v x="$pmed" 'BEGIN{print 1000/x}')" "$PTOK"

for k in $KS; do
  sl=(); sg=(); alpha=()
  for r in $(seq 1 "$REPS"); do
    out=$($B --model "$M" --ids "$IDS" --mode spec --mtp --mtp-k "$k" --n "$N" 2>&1)
    sl+=("$(printf '%s\n' "$out" | dec)")
    sg+=("$(printf '%s\n' "$out" | gen)")
    alpha+=("$(printf '%s\n' "$out" | grep -oE 'alpha=[0-9.]+' | head -1 | cut -d= -f2)")
  done
  smed=$(med "${sl[@]}")
  a=$(med "${alpha[@]}")
  same="IDENTICAL"
  [ "${sg[0]}" != "$PTOK" ] && same="DIFFERS"
  printf 'k=%s    : %6.2f ms/token (%5.1f tok/s)  alpha=%s  tokens=%s\n' \
    "$k" "$smed" "$(awk -v x="$smed" 'BEGIN{print 1000/x}')" "$a" "$same"
  if [ "$same" = "DIFFERS" ]; then
    printf '   plain: %s\n   spec : %s\n' "$PTOK" "${sg[0]}"
  fi
  awk -v a="$pmed" -v b="$smed" 'BEGIN {printf "   speedup %+.1f%%\n", 100*(a-b)/a}'
done
