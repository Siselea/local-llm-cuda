#!/usr/bin/env bash
# 4-way interleaved A/B ladder for the Coder MLA-attention-graph change.
# One measurement per configuration per round (rounds ordered base -> controls -> new
# to spread any thermal/VM drift evenly), then medians.
#
#   cfg base   : build/test_gen_ek_base.exe       (12:00 validated build, no MLA graph)
#   cfg nograph: new binary + ENGINE_NO_GRAPH=1   (disables every capture path)
#   cfg gaoff  : new binary + ENGINE_NO_MLA_ATTN_GRAPH=1 (only the MLA graph off)
#   cfg new    : new binary, default               (MLA graph + pinned routing uploads)
#
# Derivation: graph gain = new - gaoff (same binary, only the MLA graph differs);
#             upload gain = gaoff - nograph (same binary, capture paths off vs only MLA off);
#             total = new - base.
#
#   bash tools/ab_mla_ladder.sh            (REPS=5 by default)
set -uo pipefail
cd "$(dirname "$0")/.."
M="${M:-D:/p0/models/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf}"
BASE="${BASE:-./build/test_gen_ek_base.exe}"
NEW="${NEW:-./build/test_gen.exe}"
IDS="${IDS:-549,6077,280,7239}"
N="${N:-9}"
REPS="${REPS:-5}"
COMMON="ENGINE_TOPK=1 ENGINE_DEC_SKIP=2"

one() {  # <bin> <extra env> -> ms
  local out ms
  out=$(eval "$COMMON $2 \"$1\" \"$M\" \"$IDS\" \"$N\" 2>&1")
  ms=$(printf '%s\n' "$out" | grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2)
  printf '%s' "${ms:-ERR}"
}

declare -a SBASE=() SNG=() SGA=() SNEW=()
printf '%-4s %9s %9s %9s %9s\n' rep base no-graph mla-graph-off new
for r in $(seq 1 "$REPS"); do
  b=$(one "$BASE" "")
  ng=$(one "$NEW" "ENGINE_NO_GRAPH=1")
  ga=$(one "$NEW" "ENGINE_NO_MLA_ATTN_GRAPH=1")
  n=$(one "$NEW" "")
  SBASE+=("$b"); SNG+=("$ng"); SGA+=("$ga"); SNEW+=("$n")
  printf '%-4s %9s %9s %9s %9s\n' "$r" "$b" "$ng" "$ga" "$n"
done

med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END { if (NR==0) {print 0; exit} print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2 }'; }
mb=$(med "${SBASE[@]}"); mng=$(med "${SNG[@]}"); mga=$(med "${SGA[@]}"); mn=$(med "${SNEW[@]}")
printf '\nmedian: base=%s  no-graph=%s  mla-graph-off=%s  new=%s ms/token\n' "$mb" "$mng" "$mga" "$mn"
awk -v b="$mb" -v ng="$mng" -v ga="$mga" -v n="$mn" 'BEGIN{
  printf "total (new-base)        : %+.3f ms (%.2f%%), %.2f -> %.2f tok/s\n", n-b, 100*(n-b)/b, 1000/b, 1000/n;
  printf "MLA graph (new-gaoff)   : %+.3f ms\n", n-ga;
  printf "graph-capture paths (gaoff-nograph): %+.3f ms\n", ga-ng;
  printf "async routing uploads (gaoff-base, mixed with no-graph): %+.3f ms\n", ga-b;
}'
echo
echo "NOTE: 'mla-graph-off' is the clean control for the graph itself (same binary)."
