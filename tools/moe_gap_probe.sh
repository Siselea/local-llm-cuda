#!/usr/bin/env bash
# Why is llama.cpp faster?  Same model, same GPU-layer count, decode-only comparison.
#
# A) llama-bench: 35B with -ngl {0,6,15}, -t 12, -p 0 -n 32  -> tg t/s (decode only)
# B) our engine: test_gen 35B --gpu-layers {6,15}, n=32      -> decode ms + per-layer
#    breakdown ([layerms]: sync / launch / host GEMV) + ENGINE_DBG_CPUEXEC last-25 window
#
# Usage: MODE=both bash tools/moe_gap_probe.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory containing the GGUF files}"\nM="${MODEL:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
LB="${LLAMA_BENCH:-llama-bench}"
GEN="${GEN:-build/test_gen.exe}"
OUT="build/moe-gap-probe.log"
: > "$OUT"
MODE="${MODE:-both}"

if [ "$MODE" = "both" ] || [ "$MODE" = "llama" ]; then
  echo "== llama-bench (decode-only, -p 0 -n 32, -t 12) ==" | tee -a "$OUT"
  for ngl in 6 15 0; do
    echo "--- llama -ngl $ngl ---" | tee -a "$OUT"
    "$LB" -m "$M" -ngl "$ngl" -t 12 -p 0 -n 32 -r 2 -o md 2>/dev/null | tee -a "$OUT"
  done
fi

if [ "$MODE" = "both" ] || [ "$MODE" = "ours" ]; then
  echo "== ours: test_gen 35B (decode-only, n=32) ==" | tee -a "$OUT"
  for gl in 6 15; do
    echo "--- ours --gpu-layers $gl ---" | tee -a "$OUT"
    env ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_NO_VRAM_EXTEND=1 ENGINE_DBG_CPUEXEC=1 \
        ENGINE_DBG_LAYERMS=1 "$GEN" "$M" 760 32 --gpu-layers "$gl" \
        > "build/gap-ours-gl$gl.out" 2> "build/gap-ours-gl$gl.err"
    echo "  $(grep -oE 'decode [0-9.]+ ms/token \([0-9]+ tok/s' "build/gap-ours-gl$gl.err" | head -1)" | tee -a "$OUT"
    echo "  $(grep -oE '\(last 25: [0-9.]+ ms\)' "build/gap-ours-gl$gl.err" | tail -1) (host GEMV/layer)" | tee -a "$OUT"
    grep -E "\[layerms\]" "build/gap-ours-gl$gl.err" | tail -6 | sed 's/^/  /' | tee -a "$OUT"
    echo "  placement: $(grep -oE 'gpu_moe_layers=[0-9]+/[0-9]+' "build/gap-ours-gl$gl.err" | head -1)" | tee -a "$OUT"
  done
fi
echo "[saved] $OUT"
