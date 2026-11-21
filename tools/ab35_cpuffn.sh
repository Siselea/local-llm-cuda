#!/usr/bin/env bash
# 35B interleaved A/B for the CPU-layer shared-expert block graph (same binary,
# ENGINE_NO_CPU_FFN_GRAPH switch) + gen equality across all configs.
#   REPS=5 bash build/ab35_cpuffn.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M="${M:-D:/p0/models/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
BIN="${BIN:-./build/test_gen.exe}"
N="${N:-15}"
REPS="${REPS:-5}"
one() {  # <extra env> -> "ms|gen"
  local out ms gen
  out=$(eval "ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 $1 \"$BIN\" \"$M\" 760 \"$N\" --ctx 1024 2>&1")
  ms=$(printf '%s\n' "$out" | grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2)
  gen=$(printf '%s\n' "$out" | grep -oE '^gen:.*')
  printf '%s|%s' "${ms:-ERR}" "$gen"
}
A=(); B=(); GENREFA=""; GENREFB=""
printf '%-4s %10s %12s\n' rep plain cpu-ffn-graph
for r in $(seq 1 "$REPS"); do
  IFS='|' read -r a ga < <(one "ENGINE_NO_CPU_FFN_GRAPH=1")
  IFS='|' read -r b gb < <(one "")
  A+=("$a"); B+=("$b")
  [ -z "$GENREFA" ] && GENREFA="$ga"; [ -z "$GENREFB" ] && GENREFB="$gb"
  same="gen-same"; [ "$ga" = "$gb" ] || same="GEN-DIFF"
  printf '%-4s %10s %12s  %s\n' "$r" "$a" "$b" "$same"
done
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
ma=$(med "${A[@]}"); mb=$(med "${B[@]}")
printf '\nmedian: plain=%s  cpu-ffn-graph=%s  delta=%s ms\n' "$ma" "$mb" \
  "$(awk -v a="$ma" -v b="$mb" 'BEGIN{printf "%+.3f", b-a}')"
[ "$GENREFA" = "$GENREFB" ] && echo "gen identical across configs" || echo "GEN MISMATCH"
