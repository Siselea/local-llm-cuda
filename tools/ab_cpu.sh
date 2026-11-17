#!/usr/bin/env bash
# Interleaved A/B of two bench builds on the same shape; reports min and median.
# usage: bash tools/ab_cpu.sh <a.exe> <b.exe> <mode args...> [REPS]
set -uo pipefail
cd "$(dirname "$0")/.."
A="$1"; B="$2"; shift 2
REPS="${REPS:-7}"
a=(); b=()
for r in $(seq 1 "$REPS"); do
  av=$("$A" "$@" 2>&1 | grep -oE '\| [0-9.]+ GB/s' | grep -oE '[0-9.]+')
  bv=$("$B" "$@" 2>&1 | grep -oE '\| [0-9.]+ GB/s' | grep -oE '[0-9.]+')
  a+=("$av"); b+=("$bv")
done
min() { printf '%s\n' "$@" | sort -n | head -1; }
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
printf 'A: min %s  med %s   (%s)\n' "$(min "${a[@]}")" "$(med "${a[@]}")" "${a[*]}"
printf 'B: min %s  med %s   (%s)\n' "$(min "${b[@]}")" "$(med "${b[@]}")" "${b[*]}"
awk -v a="$(min "${a[@]}")" -v b="$(min "${b[@]}")" 'BEGIN {printf "min-based: %+.2f %%\n", 100*(b-a)/a}'
