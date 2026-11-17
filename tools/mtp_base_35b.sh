#!/usr/bin/env bash
# Baseline + cost-breakdown probe for the 35B (MTP feasibility).
set -uo pipefail
cd "$(dirname "$0")/.."
M35=<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
IDS="760,6511,314,9338,369"
echo "===== A: test_gen baseline (ctx 1024, 17 tokens) ====="
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ./build/test_gen.exe "$M35" "$IDS" 17 --ctx 1024 2>&1 |
  grep -E "^gen:|\[time\]|vram|placement"
echo "===== B: n=1 vs n=2 per-layer ms (ENGINE_DBG_LAYERMS) ====="
ENGINE_DBG_LAYERMS=1 ./build/mtp_probe.exe --model "$M35" --ids "$IDS" --ctx 1024 --rows 2 --reps 1 2>&1 |
  grep -E "layerms|probe"
echo "===== C: phase split (ENGINE_DBG_MOE_PHASES + KPROF) ====="
ENGINE_DBG_MOE_PHASES=1 ENGINE_KPROF=1 ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 \
  ./build/test_gen.exe "$M35" "$IDS" 6 --ctx 1024 2>&1 |
  grep -E "moeph|kprof|\[time\]" | tail -30
echo DONE
