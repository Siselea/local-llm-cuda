#!/usr/bin/env bash
# vram_extra_reserve_mb tuning curve: for each candidate reserve, record the auto-placed
# GPU layer count, the post-load free VRAM (must stay >= the 1200 MiB margin) and the
# decode speed.  The goal is the smallest reserve that keeps free VRAM safely above the
# margin on an exclusive GPU, because every extra MiB of reserve can cost a whole GPU
# layer (~7-10% decode on the MoE models).
#
#   bash tools/vram_reserve_curve.sh            (exclusive GPU, ~3 min)
#   RESERVES="0 512 1024 2048" bash tools/vram_reserve_curve.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
RESERVES="${RESERVES:-0 512 1024 1536 2048}"
GEN=./build/test_gen.exe

curve() {  # name model
  local name="$1" model="$2"
  [ -f "$model" ] || { echo "== $name: missing, skipped"; return; }
  echo "== $name =="
  for r in $RESERVES; do
    out=$(ENGINE_VRAM_EXTRA_RESERVE_MB="$r" ENGINE_DEC_SKIP=2 \
          "$GEN" --model "$model" --ids 760 17 2>&1)
    layers=$(printf '%s' "$out" | grep -oE 'gpu_moe_layers=[0-9]+' | head -1 | cut -d= -f2)
    free=$(printf '%s' "$out" | grep -oE 'post-load used=[0-9]+ MiB free=[0-9]+' | head -1)
    verdict=$(printf '%s' "$out" | grep -oE '\[vram\] (check OK|ERROR)' | head -1)
    dec=$(printf '%s' "$out" | grep -oE 'decode [0-9.]+ ms/token' | tail -1)
    printf '  reserve=%-5s layers=%-3s %-38s %-16s %s\n' \
           "${r}" "${layers:-?}" "${free:-?}" "${verdict##*/ }" "${dec}"
  done
}

curve "Coder-V2-Lite" "${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
curve "Qwen3.6-35B-A3B" "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
curve "Qwen3.5-9B" "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
echo
echo "=> pick the smallest reserve whose free VRAM stays >= ~1500 MiB (margin 1200 + slack)"
echo "   for every auto-placed (MoE) model; dense models ignore the auto layer placement."
