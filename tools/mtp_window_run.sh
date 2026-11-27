#!/usr/bin/env bash
# GPU-window driver for the MTP numeric-consistency investigation.
# Runs the minimal set of measurements and stores logs under build/mtp-win/.
#
#   bash tools/mtp_window_run.sh [stage]
#     stage=core   : probe three-stream comparison + plain/spec direct A/B (default)
#     stage=k nobs : numeric-knob scan with the probe (n_gen=180)
#     stage=cache  : serve prefix-cache cold/warm consistency (9B)
set -uo pipefail
cd "$(dirname "$0")/.."
M35="${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
M9="${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
OUT=build/mtp-win
mkdir -p "$OUT"
STAGE="${1:-core}"
N="${N:-300}"
NPE="${NPE:-180}"

case "$STAGE" in
core)
  echo "=== 1) probe: plain vs batchgreedy vs spec (${N} tokens) ==="
  ENGINE_MTP=1 ENGINE_MTP_TRACE=1 ENGINE_GPU_EXPERTS=16 \
    ./build/mtp_numprobe.exe "$M35" --ids 760,6511,314,9338,369 --n-gen "$N" \
    > "$OUT/probe35b.log" 2>&1
  echo "probe rc=$? (log $OUT/probe35b.log)"
  grep -E "model:|A decode|C seq|E decode|plain vs|spec vs|spec: k=|DIFFERS|IDENTICAL" \
    "$OUT/probe35b.log" | tail -20
  echo
  echo "=== 2) direct A/B: plain greedy vs --mtp greedy (${N} tokens) ==="
  ENGINE_GPU_EXPERTS=16 ./build/test_gen_mtpdbg.exe "$M35" \
    --ids 760,6511,314,9338,369 --n "$N" --ctx 512 \
    > "$OUT/plain.log" 2>"$OUT/plain.err"
  grep -oE '^gen:.*' "$OUT/plain.log" | head -1 | cut -d' ' -f2- > "$OUT/plain.ids"
  ENGINE_GPU_EXPERTS=16 ./build/test_gen_mtpdbg.exe "$M35" \
    --ids 760,6511,314,9338,369 --n "$N" --ctx 512 --mtp --mtp-k 2 \
    > "$OUT/spec.log" 2>"$OUT/spec.err"
  grep -oE '^gen:.*' "$OUT/spec.log" | head -1 | cut -d' ' -f2- > "$OUT/spec.ids"
  grep -E "^\[mtp\]" "$OUT/spec.err" | tail -8
  python - "$OUT/plain.ids" "$OUT/spec.ids" <<'PYEOF'
import sys
a = open(sys.argv[1]).read().split()
b = open(sys.argv[2]).read().split()
n = min(len(a), len(b))
d = next((i for i in range(n) if a[i] != b[i]), None)
print(f"plain tokens={len(a)} spec tokens={len(b)}")
if d is None:
    print(f"IDENTICAL on all {n} tokens")
else:
    print(f"DIVERGES at token {d}: plain={a[d]} spec={b[d]}")
    print("  plain:", " ".join(a[max(0,d-4):d+5]))
    print("  spec :", " ".join(b[max(0,d-4):d+5]))
PYEOF
  ;;
k)
  shift
  for knob in "$@"; do
    name="${knob%%:*}"; envs="${knob#*:}"
    echo "=== knob $name ($envs) probe ${NPE} tokens ==="
    # shellcheck disable=SC2086
    env ENGINE_MTP=1 ENGINE_GPU_EXPERTS=16 $envs \
      ./build/mtp_numprobe.exe "$M35" --ids 760,6511,314,9338,369 --n-gen "$NPE" \
      > "$OUT/probe_$name.log" 2>&1
    echo "rc=$?"
    grep -E "plain vs|spec vs|A decode|C seq" "$OUT/probe_$name.log" | tail -6
  done
  ;;
cache)
  echo "=== serve prefix-cache cold/warm (9B) ==="
  python tools/cache_cold_warm_check.py --model "$M9" \
    > "$OUT/cache_cold_warm.log" 2>&1
  echo "rc=$?"
  tail -20 "$OUT/cache_cold_warm.log"
  ;;
*)
  echo "unknown stage $STAGE"; exit 2 ;;
esac
