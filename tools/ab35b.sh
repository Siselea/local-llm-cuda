#!/usr/bin/env bash
# Interleaved A/B on the 35B: pre-change binary (test_gen_pre_opt.exe, built before the
# int8 host-GEMV work) vs the current binary.  Also captures the internal host-matvec
# time (ENGINE_DBG_CPUEXEC) and the per-layer CPU MoE phases (ENGINE_DBG_MOE_PHASES),
# which are the numbers that actually moved -- end-to-end decode is the judge.
set -uo pipefail
cd "$(dirname "$0")/.."
M35=${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
N="${N:-17}"
REPS="${REPS:-5}"
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }
tk() { grep -oE 'topk: [0-9]+' | head -1; }
A=build/test_gen_pre_opt.exe
B=build/test_gen.exe
a=(); b=()
for r in $(seq 1 "$REPS"); do
  outa=$(ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 "$A" "$M35" "760" "$N" 2>&1)
  outb=$(ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 "$B" "$M35" "760" "$N" 2>&1)
  av=$(printf '%s\n' "$outa" | dec); bv=$(printf '%s\n' "$outb" | dec)
  a+=("$av"); b+=("$bv")
  printf 'rep%d pre=%.2f new=%.2f  topk: %s / %s\n' "$r" "$av" "$bv" \
    "$(printf '%s\n' "$outa" | tk)" "$(printf '%s\n' "$outb" | tk)"
done
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
printf 'median pre=%.2f new=%.2f\n' "$(med "${a[@]}")" "$(med "${b[@]}")"
awk -v a="$(med "${a[@]}")" -v b="$(med "${b[@]}")" \
  'BEGIN {printf "delta %+.2f ms (%+.1f%%), tok/s %.2f -> %.2f\n", b-a, 100*(b-a)/a, 1000/a, 1000/b}'
echo "=== host matvec (cpuexec) + per-layer CPU phase split ==="
for tag in pre new; do
  bin="$A"; [ "$tag" = new ] && bin="$B"
  echo "--- $tag"
  ENGINE_DBG_CPUEXEC=1 ENGINE_DBG_MOE_PHASES=1 ENGINE_TOPK=1 "$bin" "$M35" "760" 2 2>&1 |
    grep -E "cpuexec|moeph" | tail -6
done
echo DONE-35B-AB
