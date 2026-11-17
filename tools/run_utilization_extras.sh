#!/usr/bin/env bash
# run_utilization_extras.sh -- the extra measurements behind UTILIZATION-REPORT.md that the
# main suite does not cover:
#
#   * 35B `layerms` run: true per-layer wall clock for the 40 layers WITHOUT ENGINE_KPROF
#     (kprof's two CUDA events per kernel inflate the wall by ~30 % on this WDDM setup;
#     the layerms run is what the report's 40-row per-layer table is built from).
#   * 35B / Coder `probe` runs with an explicitly pinned placement.  On a machine whose
#     free VRAM wobbles, the engine's `[placement] adaptive:` reduction path can hang
#     (see UTILIZATION-REPORT.md §10); `--gpu-layers N --no-vram-check` pins the
#     clean-machine placement and avoids that path.
#   * dense models' CUDA-graph `--layers` differencing (same as
#     run_class_gpu_probe.sh, kept here for one-command reproduction).
#
# Every run is a short load -> sample -> exit window; monitor waits for a quiet GPU first.
#
# Usage:
#   bash tools/run_utilization_extras.sh                 # all of the below
#   EXTRA="35b_layerms" bash tools/run_utilization_extras.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MON="./tools/util_bin/measure_utilization.exe"
BIN="./tools/util_bin/test_gen_snap.exe"
R="tools/util_results"
M35="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
MCO="<MODEL_DIR>/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
M9B="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf"
MQ="<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROBE_ENV="ENGINE_TOPK=1 ENGINE_DBG_FWD=1 ENGINE_DBG_LAYERMS=1 ENGINE_KPROF=1 ENGINE_DBG_CPUEXEC=1 ENGINE_PIPE_PROF=1"
LAYERMS_ENV="ENGINE_TOPK=1 ENGINE_DBG_FWD=1 ENGINE_DBG_LAYERMS=1 ENGINE_DBG_CPUEXEC=1"

run() { # dir env n model ids extra...
  local dir="$1" envs="$2" n="$3" model="$4" ids="$5"; shift 5
  mkdir -p "$R/$dir"
  echo "=== $dir n=$n $* ==="
  env $envs "$MON" --out-prefix "$R/$dir/measurement" --interval-ms 10 \
      --wait-gpu-idle-sec 300 --idle-mem-max-mb 1200 --cpu-idle-max 45 \
      --during-check-ms 5000 --timeout-ms 180000 \
      -- "$BIN" "$model" "$ids" "$n" "$@" 2>&1 | tail -2
}

want() { [ -z "${EXTRA:-}" ] || echo " ${EXTRA} " | grep -q " $1 "; }

# n=17: keep well clear of the intermittent long-decode hang (UTILIZATION-REPORT §10)
want 35b_layerms && \
run qwen3.6-35B-A3B/layerms "$LAYERMS_ENV" 17 "$M35" 760 --no-graph --no-vram-check --gpu-layers 15

want 35b_probe && \
run qwen3.6-35B-A3B/probe "$PROBE_ENV" 17 "$M35" 760 --no-graph --no-vram-check --gpu-layers 15

want 35b_prefix && {
  run qwen3.6-35B-A3B/prefix1 "$PROBE_ENV" 17 "$M35" 760 --no-graph --no-vram-check --gpu-layers 15 --layers 1
  run qwen3.6-35B-A3B/prefixN "$PROBE_ENV" 17 "$M35" 760 --no-graph --no-vram-check --gpu-layers 15 --layers 15
}

want coder_prefix && {
  run DeepSeek-Coder-V2-Lite/probe   "$PROBE_ENV" 17 "$MCO" "549,6077,280,7239" --no-graph --no-vram-check --gpu-layers 20
  run DeepSeek-Coder-V2-Lite/prefix1 "$PROBE_ENV" 17 "$MCO" "549,6077,280,7239" --no-graph --no-vram-check --gpu-layers 20 --layers 1
  run DeepSeek-Coder-V2-Lite/prefixN "$PROBE_ENV" 17 "$MCO" "549,6077,280,7239" --no-graph --no-vram-check --gpu-layers 20 --layers 17
}

want dense_graph_layers && {
  run qwen3.5-9B/graph_layers1    "ENGINE_TOPK=1 ENGINE_DBG_FWD=1" 33 "$M9B" 760 --layers 1
  run qwen3.5-9B/graph_layers16   "ENGINE_TOPK=1 ENGINE_DBG_FWD=1" 33 "$M9B" 760 --layers 16
  run qwen2.5-0.5b/graph_layers1  "ENGINE_TOPK=1 ENGINE_DBG_FWD=1" 257 "$MQ" "1000,1001" --layers 1
  run qwen2.5-0.5b/graph_layers12 "ENGINE_TOPK=1 ENGINE_DBG_FWD=1" 257 "$MQ" "1000,1001" --layers 12
}

echo "extras done; regenerate summary with:"
echo "  python tools/utilization_report.py --root $R --md $R/summary.md --json $R/summary.json"
