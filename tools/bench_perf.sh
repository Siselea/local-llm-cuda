#!/usr/bin/env bash
# Deterministic 4-model decode benchmark used for before/after comparison.
# The reference configuration (matching the project's stated baselines) is:
#   - MoE models use the engine's automatic layer-granular placement (no pinning)
#   - 0.5B / 9B run with warmup enabled
# Usage: BIN=./build/test_gen.exe bash build/bench_perf.sh <tag>
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-run}"
BIN="${BIN:-./build/test_gen.exe}"
OUT="build/perf-$TAG.txt"
: > "$OUT"
M_QWEN="${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
M_CODER="${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
M_9B="${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
M_35B="${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"

one() {
  local tag="$1" model="$2" ids="$3" n="$4" envs="$5"
  local out top dec tps
  if [ ! -f "$model" ]; then
    echo "$tag SKIP(model missing)" | tee -a "$OUT"
    return
  fi
  out=$(env ENGINE_TOPK=1 $envs "$BIN" "$model" "$ids" "$n" 2>&1)
  top=$(printf "%s" "$out" | grep -oE "topk: [0-9]+\\([-0-9.]+\\)" | head -1)
  dec=$(printf "%s" "$out" | grep -oE "decode [0-9.]+ ms/token" | head -1)
  tps=$(printf "%s" "$out" | grep -oE "decode [0-9.]+ ms/token" | sed -E 's/decode ([0-9.]+) ms.*/\1/' | awk '{if ($1>0) printf "%.2f", 1000/$1}')
  echo "$tag | $top | $dec | ${tps:-?} tok/s" | tee -a "$OUT"
}

echo "== perf tag=$TAG bin=$BIN ==" | tee -a "$OUT"
for rep in 1 2; do
  echo "-- rep $rep --" | tee -a "$OUT"
  one "qwen0.5b " "$M_QWEN" "1000,1001" 33 "ENGINE_SKIP_WARMUP=0"
  one "coder     " "$M_CODER" "549,6077,280,7239" 9 ""
  one "qwen3.5-9B" "$M_9B" "760" 9 "ENGINE_SKIP_WARMUP=0"
  one "qwen3.6-35B" "$M_35B" "760" 9 ""
done
echo "wrote $OUT"
