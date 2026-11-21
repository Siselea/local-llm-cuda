#!/usr/bin/env bash
# Interleaved A/B decode benchmark that compares two *different binaries* (the MLA
# attention-graph change), because one side must not contain the new behaviour at all.
#
#   BASE=./build/test_gen_ek_base.exe NEW=./build/test_gen.exe bash tools/ab_mla.sh
#
# Each line: base ms | new ms | delta | top-1 | gen md5 of the two runs (must match).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
M="${M:-${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf}"
BASE="${BASE:-./build/test_gen_ek_base.exe}"
NEW="${NEW:-./build/test_gen.exe}"
IDS="${IDS:-549,6077,280,7239}"
N="${N:-9}"
REPS="${REPS:-3}"
ENVS="${ENVS:-ENGINE_TOPK=1 ENGINE_DEC_SKIP=2}"
CTX="${CTX:-}"

one() {  # $1 = binary; prints "ms|top1|gen"
  local out ms top gen
  out=$(eval "$ENVS \"$1\" \"$M\" \"$IDS\" \"$N\" $CTX 2>&1")
  ms=$(printf '%s\n' "$out" | grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2)
  top=$(printf '%s\n' "$out" | grep -oE 'topk: [0-9]+' | head -1 | cut -d' ' -f2)
  gen=$(printf '%s\n' "$out" | grep -oE '^gen:.*' | head -1)
  printf '%s|%s|%s\n' "${ms:-ERR}" "${top:-?}" "$gen"
}

printf 'base=%s\nnew =%s\nenvs=%s\n' "$BASE" "$NEW" "$ENVS"
printf '%-4s %9s %9s %9s  %s\n' rep base new delta gen
ab=(); nb=()
for r in $(seq 1 "$REPS"); do
  IFS='|' read -r a at ag < <(one "$BASE")
  IFS='|' read -r b bt bg < <(one "$NEW")
  ab+=("$a"); nb+=("$b")
  same="gen-same"
  [ "$ag" = "$bg" ] || same="GEN-DIFF"
  printf '%-4s %9s %9s %9s  %s (top %s/%s)\n' "$r" "$a" "$b" \
    "$(awk -v x="$a" -v y="$b" 'BEGIN{ if (x+0>0 && y+0>0) printf "%+.3f", y-x; else printf "n/a" }')" \
    "$same" "$at" "$bt"
done
amed=$(printf '%s\n' "${ab[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
nmed=$(printf '%s\n' "${nb[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
printf 'median base=%s new=%s delta=%s tok/s base=%.2f new=%.2f\n' "$amed" "$nmed" \
  "$(awk -v a="$amed" -v b="$nmed" 'BEGIN{ printf "%+.3f ms (%.2f%%)", b-a, 100*(b-a)/a }')" \
  "$(awk -v a="$amed" 'BEGIN{printf "%.2f", 1000/a}')" \
  "$(awk -v b="$nmed" 'BEGIN{printf "%.2f", 1000/b}')"
