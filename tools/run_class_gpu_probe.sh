#!/usr/bin/env bash
# run_class_gpu_probe.sh -- measure the *true* GPU time per layer for the dense models,
# free of host-submission bubbles, using CUDA-graph mode + --layers differencing.
#
# Why: ENGINE_KPROF (CUDA events) only works with --no-graph, and in --no-graph mode the
# per-kernel event elapsed times include the host submission gaps whenever the host is the
# bottleneck (0.5B: 0.55 ms host per layer vs ~0.04 ms of kernels).  For dense models the
# full CUDA graph removes every submission gap, the GPU is saturated (NVML util 96 %), so
# the graph-mode token wall IS the GPU time:
#     W(1)  = p + H          (1 layer + final norm + lm_head)
#     W(N)  = N*p + H        -> p = (W(N)-W(1))/(N-1),  H = W(1)-p
#
# Runs needed: --layers 1 and --layers N (N = 16 for 9B, 12 for 0.5B) in default (graph)
# mode, 257 decode tokens so that the CUDA-graph capture token is amortised away and the
# NVML 1 s-scale metrics are meaningful.  Outputs go next to the main suite results.
#
# Usage: bash tools/run_class_gpu_probe.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="tools/util_results"
MON="./tools/util_bin/measure_utilization.exe"
BIN="./tools/util_bin/test_gen_snap.exe"
WAIT_SEC="${WAIT_SEC:-600}"

run() { # model name path ids n layers
  local model="$1" path="$2" ids="$3" n="$4" layers="$5"
  local out="$ROOT/$model/graph_layers$layers"
  mkdir -p "$out"
  echo "=== $model graph --layers $layers ==="
  env ENGINE_TOPK=1 ENGINE_DBG_FWD=1 \
    "$MON" --out-prefix "$out/measurement" --interval-ms 10 --wait-gpu-idle-sec "$WAIT_SEC" \
           --idle-mem-max-mb 1200 --cpu-idle-max 45 --during-check-ms 5000 \
           -- "$BIN" "$path" "$ids" "$n" --layers "$layers" 2>&1 | tail -3
}

M_QWEN="<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf"
M_9B="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf"

run qwen2.5-0.5b "$M_QWEN" "1000,1001" 257 1
run qwen2.5-0.5b "$M_QWEN" "1000,1001" 257 12
run qwen3.5-9B "$M_9B" "760" 257 1
run qwen3.5-9B "$M_9B" "760" 257 16
echo "done"
