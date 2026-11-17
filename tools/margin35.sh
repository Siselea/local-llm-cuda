#!/usr/bin/env bash
# 35B carry-one-more-GPU-layer probe: --vram-margin 1200 (default) vs 400.
#
# Answers: how much decode does one extra GPU-resident MoE layer buy, and what is the
# post-load free-VRAM safety boundary?  Outputs are diffed token-by-token (gen md5) so
# the placement change is proven numerically identical (or not) instead of assumed.
#
# Usage: REPS=4 bash tools/margin35.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
BIN="${BIN:-build/test_gen.exe}"
N="${N:-40}"
REPS="${REPS:-4}"
OUT="build/margin35.log"
: > "$OUT"
echo "== 35B margin probe (n=$N, reps=$REPS) ==" | tee -a "$OUT"
last_window() {
  grep -oE '\(last [0-9]+: [0-9.]+ ms\)' "$1" | tail -1 | grep -oE '[0-9.]+ ms' | awk '{print $1}'
}
run() { # tag margin
  local tag="$1" mg="$2"
  ( while :; do
      nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null
      sleep 0.5
    done ) > "build/pwrm-$tag.log" 2>/dev/null &
  local pw=$!
  env ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_DBG_CPUEXEC=1 ENGINE_WIN_PRIORITY=1 \
      "$BIN" "$M" 760 "$N" --vram-margin "$mg" > "build/m-$tag.out" 2> "build/m-$tag.err"
  kill $pw 2>/dev/null; wait $pw 2>/dev/null
  local dec gemv layers free_mb topk gen pwr
  dec=$(grep -oE 'decode [0-9.]+' "build/m-$tag.err" | head -1 | cut -d' ' -f2)
  gemv=$(last_window "build/m-$tag.err")
  layers=$(grep -oE 'gpu_moe_layers=[0-9]+' "build/m-$tag.err" | head -1 | cut -d= -f2)
  free_mb=$(grep -oE 'post-load used=[0-9]+ MiB free=[0-9]+' "build/m-$tag.err" | grep -oE 'free=[0-9]+' | cut -d= -f2)
  # minimum free VRAM observed by the sampler after load (drives the "safe floor" answer)
  minfree=$(awk -F', *' 'NF>=3 {if (mn=="" || $3+0<mn) mn=$3+0} END {printf "%d", mn}' "build/pwrm-$tag.log" 2>/dev/null)
  topk=$(grep -E '^topk:' "build/m-$tag.out" | head -1 | sed 's/topk: //;s/(.*)//g;s/ //g')
  gen=$(grep -E '^gen:' "build/m-$tag.out" | head -1 | md5sum | cut -c1-8)
  pwr=$(awk -F', *' '{if($1+0>p)p=$1+0} END {printf "%.1fW", p}' "build/pwrm-$tag.log" 2>/dev/null)
  printf '%-8s margin=%-5s layers=%-2s postfree=%-5s minfree=%-5s decode=%7s ms hostGEMV=%s ms/layer top=%s gen=%s gpu=%s\n' \
    "$tag" "$mg" "${layers:-?}" "${free_mb:-?}" "${minfree:-?}" "${dec:-ERR}" "${gemv:-ERR}" \
    "${topk:0:4}" "$gen" "$pwr" | tee -a "$OUT"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then order=("m1200" "m400"); else order=("m400" "m1200"); fi
  for arm in "${order[@]}"; do
    case "$arm" in
      m1200) run "m1200-r$r" 1200 ;;
      m400)  run "m400-r$r" 400 ;;
    esac
  done
done
echo "== summary ==" | tee -a "$OUT"
med_of() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
for arm in m1200 m400; do
  vals=(); gs=()
  for f in build/m-$arm-r*.err; do
    [ -f "$f" ] || continue
    v=$(grep -oE 'decode [0-9.]+' "$f" | head -1 | cut -d' ' -f2); [ -n "$v" ] && vals+=("$v")
    g=$(last_window "$f"); [ -n "$g" ] && gs+=("$g")
  done
  [ ${#vals[@]} -eq 0 ] && continue
  m=$(med_of "${vals[@]}"); gm=$(med_of "${gs[@]}")
  awk -v n="$arm" -v m="$m" -v g="$gm" \
    'BEGIN {printf "%-8s median decode %7.3f ms/token = %6.2f tok/s  hostGEMV %s ms/layer\n", n, m, 1000/m, g}' | tee -a "$OUT"
done
echo "== raw: $OUT =="
