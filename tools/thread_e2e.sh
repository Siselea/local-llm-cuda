#!/usr/bin/env bash
# End-to-end 35B decode A/B for thread-count x affinity configs.
# Waits (in the background) for the GPU to become free (>= THRESH MiB free for 3
# consecutive checks), then runs the configs with build/test_gen_scan.exe.
#
# Each run: 35B, prompt 760, n=20 tokens, ENGINE_TOPK=1 ENGINE_DEC_SKIP=2
# ENGINE_DBG_CPUEXEC=1.  Records decode ms/token, the last-25-layer host-GEMV window
# (ENGINE_DBG_CPUEXEC), the placement line and the generated-token md5 (bit-exactness).
#
# Usage: N=20 REPS=1 WAIT=1 bash tools/thread_e2e.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory containing the GGUF files}"\nM="${MODEL:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
BIN="${BIN:-build/test_gen_scan.exe}"
N="${N:-20}"
REPS="${REPS:-1}"
THRESH="${THRESH:-11500}"
WAIT="${WAIT:-1}"
OUT="build/thread-e2e.log"
: > "$OUT"

gpu_free() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
    head -1 | tr -d ' \r'
}

last_window() {
  grep -oE '\(last [0-9]+: [0-9.]+ ms\)' "$1" | tail -1 | grep -oE '[0-9.]+ ms' | awk '{print $1}'
}

if [ "$WAIT" = "1" ]; then
  echo "[e2e] waiting for GPU free >= ${THRESH} MiB (background waiter) ..." | tee -a "$OUT"
  ok=0
  for _ in $(seq 1 720); do   # up to 1h at 5 s
    f=$(gpu_free)
    if [ -n "$f" ] && [ "$f" -ge "$THRESH" ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    if [ "$ok" -ge 3 ]; then break; fi
    sleep 5
  done
  echo "[e2e] GPU free at $(date '+%H:%M:%S') (free=$(gpu_free) MiB)" | tee -a "$OUT"
fi

run() { # tag threads aff   (threads=0 => engine default; aff=- => engine default)
  local tag="$1" thr="$2" aff="$3"
  local free
  free=$(gpu_free)
  if [ -z "$free" ] || [ "$free" -lt "$THRESH" ]; then
    echo "$tag SKIP (gpu free ${free:-?} MiB)" | tee -a "$OUT"
    return 3
  fi
  local t0 t1 extra_env=() extra_arg=()
  [ "$aff" != "-" ] && extra_env+=("ENGINE_WIN_AFFINITY=$aff")
  [ "$thr" != "0" ] && extra_arg+=(--threads "$thr")
  t0=$(date +%s)
  env "${extra_env[@]}" ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_DBG_CPUEXEC=1 \
      "$BIN" "$M" 760 "$N" "${extra_arg[@]}" \
      > "build/e2e-$tag.out" 2> "build/e2e-$tag.err"
  local rc=$?
  t1=$(date +%s)
  local dec gemv layers topk gen cls
  dec=$(grep -oE 'decode [0-9.]+' "build/e2e-$tag.err" | head -1 | cut -d' ' -f2)
  gemv=$(last_window "build/e2e-$tag.err")
  layers=$(grep -oE 'over [0-9]+ layers' "build/e2e-$tag.err" | tail -1 | grep -oE '[0-9]+')
  topk=$(grep -E '^topk:' "build/e2e-$tag.out" | head -1 | sed 's/topk: //;s/(.*)//g;s/ //g')
  gen=$(grep -E '^gen:' "build/e2e-$tag.out" | head -1 | md5sum | cut -c1-8)
  cls=$(grep -o 'priority class = [A-Z_]*' "build/e2e-$tag.err" | head -1 | awk '{print $4}')
  local cpl
  cpl=$(grep -cE '\[cpuexec\]' "build/e2e-$tag.err")
  printf '%-14s thr=%-2s aff=%-2s rc=%d wall=%3ss decode=%7s ms hostGEMV=%s ms/层 (cpuexec lines=%s) %s top=%s gen=%s\n' \
    "$tag" "$thr" "$aff" "$rc" "$((t1 - t0))" "${dec:-ERR}" "${gemv:-ERR}" "$cpl" \
    "${cls:-none}" "${topk:0:4}" "$gen" | tee -a "$OUT"
}

for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then
    order=("old-9-os:9:0" "new-default:0:-" "t10-pair:10:2")
  else
    order=("t10-pair:10:2" "new-default:0:-" "old-9-os:9:0")
  fi
  for cfg in "${order[@]}"; do
    IFS=':' read -r tag thr aff <<< "$cfg"
    run "$tag" "$thr" "$aff"
  done
done
echo "[e2e] done $(date '+%F %T')" | tee -a "$OUT"
