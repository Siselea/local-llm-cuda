#!/usr/bin/env bash
# 35B: placement/margin experiment + baseline decode median
set -uo pipefail
cd "$(dirname "$0")/.."
M=<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
B=./build/test_gen.exe
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }

echo "=== placement lines at various margins ==="
for MG in 1200 900 700 500 400 300; do
  echo "--- margin=$MG ---"
  ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 ENGINE_VRAM_MARGIN_MB=$MG $B "$M" 760 2 2>&1 |
    grep -E '\[placement\]|\[vram\] post-load|\[vram\] check OK|\[vram\] free' | head -5
done

echo "=== decode median per margin (3 reps) ==="
for MG in 1200 700 500 400 300; do
  vals=""
  for i in 1 2 3; do
    v=$(ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 ENGINE_VRAM_MARGIN_MB=$MG $B "$M" 760 9 2>&1 | dec)
    vals="$vals ${v:-ERR}"
  done
  med=$(printf '%s\n' $vals | sort -n | sed -n 2p)
  echo "margin=$MG vals:$vals median=$med"
done
