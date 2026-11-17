#!/usr/bin/env bash
# Four-model decode benchmark with a 35B-specific timing window.
#
# The 35B runs with ENGINE_DEC_SKIP=2 so the per-layer CUDA-graph capture (paid once
# per process, ~tens of ms) is excluded from the average -- this is the steady-state
# decode number the two binaries are compared on.  The other three models have no
# CPU-placed MoE layers, so their numbers are directly comparable with the historical
# tools/bench_perf.sh results.
#
# Usage: BIN=./build/test_gen.exe TAG=after bash tools/bench_ab_final.sh
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${TAG:-run}"
BIN="${BIN:-./build/test_gen.exe}"
OUT="build/ab-$TAG.txt"
: > "$OUT"
M_QWEN="<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf"
M_CODER="<MODEL_DIR>/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
M_9B="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf"
M_35B="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"

one() {
  local tag="$1" model="$2" ids="$3" n="$4" envs="$5"
  local out top dec tps gen
  if [ ! -f "$model" ]; then
    echo "$tag SKIP(model missing)" | tee -a "$OUT"
    return
  fi
  out=$(env ENGINE_TOPK=1 $envs "$BIN" "$model" "$ids" "$n" 2>&1)
  top=$(printf "%s" "$out" | grep -oE "topk: [0-9]+\([-0-9.]+\).*" | head -1)
  gen=$(printf "%s" "$out" | grep -oE "^gen:.*" | head -1)
  dec=$(printf "%s" "$out" | grep -oE "decode [0-9.]+ ms/token" | head -1)
  tps=$(printf "%s" "$out" | grep -oE "decode [0-9.]+ ms/token" | sed -E 's/decode ([0-9.]+) ms.*/\1/' | awk '{if ($1>0) printf "%.2f", 1000/$1}')
  echo "$tag | $top | $dec | ${tps:-?} tok/s" | tee -a "$OUT"
  echo "$tag | $gen" >> "$OUT"
}

echo "== ab tag=$TAG bin=$BIN ==" | tee -a "$OUT"
for rep in ${REPS:-1}; do
  echo "-- rep $rep --" | tee -a "$OUT"
  one "qwen0.5b " "$M_QWEN" "1000,1001" 33 "ENGINE_SKIP_WARMUP=0"
  one "coder     " "$M_CODER" "549,6077,280,7239" 9 ""
  one "qwen3.5-9B" "$M_9B" "760" 9 "ENGINE_SKIP_WARMUP=0"
  one "qwen3.6-35B" "$M_35B" "760" 14 "ENGINE_DEC_SKIP=2"
done
echo "wrote $OUT"
