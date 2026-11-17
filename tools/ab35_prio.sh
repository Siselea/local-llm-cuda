#!/usr/bin/env bash
# 35B decode A/B: Windows priority (win_priority normal vs HIGH) on the real desktop.
#
# Arms:
#   pre      build/test_gen_prepr.exe  (the pre-change binary, no priority code)
#   normal   build/test_gen.exe with ENGINE_WIN_PRIORITY=0
#   high     build/test_gen.exe with ENGINE_WIN_PRIORITY=1  (production default)
#
# Every run is one process (priority is a process property), so each arm reloads the
# 20 GB model; the arm order is reversed on even reps so the slow monotonic drift of
# this box cancels.  n=40 -> 37 timed decode steps after ENGINE_DEC_SKIP=2.
# Records: decode ms/token, in-engine host GEMV (ENGINE_DBG_CPUEXEC, cumulative avg
# over the last 25-layer window), GPU power/util (nvidia-smi sampler) and total CPU
# utilisation (typeperf) during the run.
#
# Usage: REPS=5 PRE_REPS=3 bash tools/ab35_prio.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
NEW="build/test_gen.exe"
PRE="build/test_gen_prepr.exe"
N="${N:-40}"
REPS="${REPS:-5}"
PRE_REPS="${PRE_REPS:-3}"
OUT="build/ab35-prio.log"
: > "$OUT"
echo "== 35B decode A/B: priority (n=$N, reps=$REPS, pre_reps=$PRE_REPS) ==" | tee -a "$OUT"
echo "ENV cpu% before: $(cpu_load)" | tee -a "$OUT"
echo "ENV gpu_used before: $(gpu_used_mib) MiB" | tee -a "$OUT"

last_window() { # exact average of the final 25-layer window from ENGINE_DBG_CPUEXEC
  grep -oE '\(last [0-9]+: [0-9.]+ ms\)' "$1" | tail -1 |
    grep -oE '[0-9.]+ ms' | awk '{print $1}'
}

# Total CPU% sampled in ONE PowerShell process (4 x 1 s).  The earlier per-run
# `cmd /c typeperf` sampler leaked its typeperf child (kill only caught the cmd wrapper)
# and those orphans ate ~8% CPU each -> it polluted a whole clean-room batch.  Now the
# environment load is sampled once before/after the batch, never during a timed run.
cpu_load() {
  powershell -NoProfile -Command \
    "(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 4).CounterSamples | ForEach-Object { [math]::Round(\$_.CookedValue,1) }" \
    2>/dev/null | tr '\n' ' '
}

# GPU memory in use right before the run: proves the GPU was free of other models
# (the clean-room window is shared with other agents otherwise).
gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

run() { # tag bin [env...]
  local tag="$1" bin="$2"; shift 2
  ( while :; do
      nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null
      sleep 0.5
    done ) > "build/pwr-$tag.log" 2>/dev/null &
  local pw=$!
  local t0 t1 gpu0
  gpu0=$(gpu_used_mib)
  t0=$(date +%s)
  env ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_DBG_CPUEXEC=1 "$@" "$bin" "$M" 760 "$N" \
      > "build/o-$tag.out" 2> "build/o-$tag.err"
  t1=$(date +%s)
  kill $pw 2>/dev/null; wait $pw 2>/dev/null
  local dec gemv topk gen pwr cls
  dec=$(grep -oE 'decode [0-9.]+' "build/o-$tag.err" | head -1 | cut -d' ' -f2)
  gemv=$(last_window "build/o-$tag.err")
  topk=$(grep -E '^topk:' "build/o-$tag.out" | head -1 | sed 's/topk: //;s/(.*)//g;s/ //g')
  gen=$(grep -E '^gen:' "build/o-$tag.out" | head -1 | md5sum | cut -c1-8)
  pwr=$(awk -F', *' '{if ($1+0 > p) p=$1+0; if ($2+0 > u) u=$2+0} END {printf "%.1fW/%d%%", p, u}' \
        "build/pwr-$tag.log" 2>/dev/null)
  cls=$(grep -o 'priority class = [A-Z_]*' "build/o-$tag.err" | head -1 | awk '{print $4}')
  printf '%-8s rep wall=%3ss decode=%7s ms  hostGEMV=%s ms/layer  %-22s top=%s gen=%s  gpu=%s  gpu_before=%sMiB\n' \
    "$tag" "$((t1 - t0))" "${dec:-ERR}" "${gemv:-ERR}" "${cls:-none}" "${topk:0:4}" "$gen" "$pwr" "${gpu0:-?}" \
    | tee -a "$OUT"
}

for r in $(seq 1 "$REPS"); do
  order=("normal" "high")
  [ "$r" -le "$PRE_REPS" ] && order=("pre" "normal" "high")
  if [ $((r % 2)) -eq 0 ]; then
    rev=(); for ((i = ${#order[@]} - 1; i >= 0; i--)); do rev+=("${order[i]}"); done
    order=("${rev[@]}")
  fi
  for arm in "${order[@]}"; do
    case "$arm" in
      pre)     run "pre-r$r" "$PRE" ;;
      normal)  run "normal-r$r" "$NEW" ENGINE_WIN_PRIORITY=0 ;;
      high)    run "high-r$r" "$NEW" ENGINE_WIN_PRIORITY=1 ;;
    esac
  done
done

echo "== summary ==" | tee -a "$OUT"
med_of() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
for arm in pre normal high; do
  vals=(); gs=()
  for f in build/o-$arm-r*.err; do
    [ -f "$f" ] || continue
    v=$(grep -oE 'decode [0-9.]+' "$f" | head -1 | cut -d' ' -f2)
    [ -n "$v" ] && vals+=("$v")
    g=$(last_window "$f"); [ -n "$g" ] && gs+=("$g")
  done
  [ ${#vals[@]} -eq 0 ] && continue
  m=$(med_of "${vals[@]}"); gm=$(med_of "${gs[@]}")
  awk -v n="$arm" -v m="$m" -v g="$gm" \
    'BEGIN {printf "%-8s median decode %7.3f ms/token = %6.2f tok/s   hostGEMV %s ms/layer\n", n, m, 1000/m, g}' \
    | tee -a "$OUT"
done
b=$(grep -oE 'decode=[0-9.]+' "$OUT" | head -0)
echo "ENV cpu% after: $(cpu_load)" | tee -a "$OUT"
echo "ENV gpu_used after: $(gpu_used_mib) MiB" | tee -a "$OUT"
echo "== raw: $OUT =="
