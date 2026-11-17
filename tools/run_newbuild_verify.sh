#!/usr/bin/env bash
# One-off verification of the dev-route + per-layer-CUDA-graph build (2026-11-15) with the
# same measurement protocol as UTILIZATION-REPORT.md.  Kept out of the deliverable tree.
set -uo pipefail
cd "$(dirname "$0")/.."
MON="./tools/util_bin/measure_utilization.exe"
BIN="./tools/util_bin/test_gen_new.exe"
R="tools/util_results_new"
M35="${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
MCO="${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
run() { local dir="$1" envs="$2" n="$3" model="$4" ids="$5"; shift 5
  mkdir -p "$R/$dir"; echo "=== $dir n=$n $* ==="
  env $envs "$MON" --out-prefix "$R/$dir/measurement" --interval-ms 10 \
      --wait-gpu-idle-sec 900 --idle-mem-max-mb 1200 --cpu-idle-max 45 \
      --during-check-ms 5000 --timeout-ms 420000 \
      -- "$BIN" "$model" "$ids" "$n" "$@" 2>&1 | tail -2; }

# 1) default execution (new: dev-route + per-layer CUDA graph on GPU layers)
run qwen3.6-35B-A3B/new_default "ENGINE_TOPK=1 ENGINE_DBG_FWD=1" 17 \
    "$M35" 760 --gpu-layers 15 --no-vram-check
# 2) probe: --no-graph (layer graph off) + kprof/layerms, to compare the GPU/CPU split
run qwen3.6-35B-A3B/new_probe \
    "ENGINE_TOPK=1 ENGINE_DBG_FWD=1 ENGINE_DBG_LAYERMS=1 ENGINE_KPROF=1 ENGINE_DBG_CPUEXEC=1" 17 \
    "$M35" 760 --no-graph --gpu-layers 15 --no-vram-check
# 3) long-decode hang verification (used to hang at token 25)
run qwen3.6-35B-A3B/new_long "ENGINE_TOPK=1 ENGINE_DBG_FWD=1" 257 \
    "$M35" 760 --gpu-layers 15 --no-vram-check
# 4) Coder long-decode hang verification (used to hang at token ~77)
run DeepSeek-Coder-V2-Lite/new_long "ENGINE_TOPK=1 ENGINE_DBG_FWD=1" 257 \
    "$MCO" "549,6077,280,7239" --gpu-layers 20 --no-vram-check
echo ALLDONE
