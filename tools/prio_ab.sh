#!/usr/bin/env bash
# Windows priority A/B for the CPU host-GEMV path — pure CPU, no GPU needed.
#
# Runs the isolated 35B-shape CPU layer benchmark (24 DRAM-resident copies of the
# expert slab, 9 pool threads) once per arm, alternating the arm order every rep so
# minute-scale machine drift cancels.  This exercises the *production* code path:
# config::apply_win_priority() sets the process class at startup and cpu::Pool
# applies win_thread_priority to the workers when it creates them.
#
# Usage: REPS=5 ITERS=200 bash tools/prio_ab.sh
#   Arms (override with ARMS="..." as "name|ENV=1 ENV2=2" list):
#     normal                 win_priority=0
#     high                   win_priority=1            (production default)
#     above-normal           win_priority=2
#     high+workers-inherit   win_priority=1 win_thread_priority=0
#     high+workers-highest   win_priority=1 win_thread_priority=2
#     normal+workers-highest win_priority=0 win_thread_priority=2
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BIN:-build/bench_cpu_layer.exe}"
COPIES="${COPIES:-24}"
ITERS="${ITERS:-200}"
REPS="${REPS:-5}"
OUT="build/prio-ab.log"
: > "$OUT"
DEFAULT_ARMS="normal|ENGINE_WIN_PRIORITY=0
high|ENGINE_WIN_PRIORITY=1
above-normal|ENGINE_WIN_PRIORITY=2
high+workers-inherit|ENGINE_WIN_PRIORITY=1 ENGINE_WIN_THREAD_PRIORITY=0
high+workers-highest|ENGINE_WIN_PRIORITY=1 ENGINE_WIN_THREAD_PRIORITY=2
normal+workers-highest|ENGINE_WIN_PRIORITY=0 ENGINE_WIN_THREAD_PRIORITY=2"
ARMS_TEXT="${ARMS:-$DEFAULT_ARMS}"
ARMS=()
while IFS= read -r line; do [ -n "$line" ] && ARMS+=("$line"); done <<< "$ARMS_TEXT"
NA=${#ARMS[@]}

echo "== prio A/B: $BIN layer $COPIES $ITERS, reps=$REPS, arms=$NA ==" | tee -a "$OUT"
for r in $(seq 1 "$REPS"); do
  idxs=($(seq 0 $((NA - 1))))
  if [ $((r % 2)) -eq 0 ]; then
    rev=()
    for ((i = NA - 1; i >= 0; i--)); do rev+=("${idxs[i]}"); done
    idxs=("${rev[@]}")
  fi
  for i in "${idxs[@]}"; do
    arm="${ARMS[$i]}"
    name="${arm%%|*}"
    envs="${arm#*|}"
    # shellcheck disable=SC2086
    env $envs "$BIN" layer "$COPIES" "$ITERS" \
        > "build/prio-$name-r$r.out" 2> "build/prio-$name-r$r.err"
    ms=$(grep -oE '\| [0-9.]+ ms/layer' "build/prio-$name-r$r.out" | head -1 | awk '{print $2}')
    gbs=$(grep -oE '\| [0-9.]+ GB/s' "build/prio-$name-r$r.out" | head -1 | awk '{print $2}')
    chk=$(grep -oE 'chk [-0-9.]+' "build/prio-$name-r$r.out" | head -1 | awk '{print $2}')
    tgt=$(grep -o 'priority class = [A-Z_]*' "build/prio-$name-r$r.err" | head -1 | awk '{print $4}')
    printf '%-26s rep%d %8s ms  %6s GB/s  class=%-12s chk=%s\n' \
      "$name" "$r" "${ms:-ERR}" "${gbs:-?}" "${tgt:-none}" "${chk:-?}" | tee -a "$OUT"
  done
done

echo "== summary (median over $REPS reps) ==" | tee -a "$OUT"
med_of() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
base=""
for ((i = 0; i < NA; i++)); do
  name="${ARMS[$i]%%|*}"
  vals=()
  for r in $(seq 1 "$REPS"); do
    f="build/prio-$name-r$r.out"
    v=$(grep -oE '\| [0-9.]+ ms/layer' "$f" | head -1 | awk '{print $2}')
    [ -n "$v" ] && vals+=("$v")
  done
  [ ${#vals[@]} -eq 0 ] && continue
  m=$(med_of "${vals[@]}")
  mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1)
  [ -z "$base" ] && base="$m"
  awk -v n="$name" -v m="$m" -v mn="$mn" -v b="$base" \
    'BEGIN {printf "%-26s median %7.4f ms/layer (%.2f GB/s)  min %7.4f  vs-normal %+6.1f%%\n", n, m, 15.56*1024/(m*1000), mn, 100*(m-b)/b}' | tee -a "$OUT"
done
echo "== raw: $OUT =="
