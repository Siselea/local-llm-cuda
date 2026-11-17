#!/usr/bin/env bash
# 35B margin sweep, one config at a time, hard timeout per run.
# Prints the placement decision and the decode ms/token immediately.
set -uo pipefail
cd "$(dirname "$0")/.."
M=<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
B=./build/test_gen.exe
N="${N:-5}"
for MG in ${MARGINS:-1200 1000 900 800 700}; do
  printf '=== margin=%s ===\n' "$MG"
  for i in 1 2; do
    out=$(ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 ENGINE_VRAM_MARGIN_MB=$MG timeout -s KILL 120 $B "$M" 760 "$N" 2>&1)
    rc=$?
    pl=$(printf '%s\n' "$out" | grep -E '\[placement\]' | head -1)
    vp=$(printf '%s\n' "$out" | grep -E '\[vram\] post-load' | head -1)
    dec=$(printf '%s\n' "$out" | grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2)
    top=$(printf '%s\n' "$out" | grep -oE 'topk: [0-9]+' | head -1 | cut -d' ' -f2)
    printf '  run%d rc=%d decode=%s top1=%s\n' "$i" "$rc" "${dec:-HANG}" "${top:-?}"
    printf '    %s\n' "${pl:-<no placement line>}"
    printf '    %s\n' "${vp:-<no vram line>}"
  done
done
