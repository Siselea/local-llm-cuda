#!/usr/bin/env bash
# Re-run the one gen_regression case that flaked with an empty `gen:` line, with the
# full stderr visible, so a deterministic cause (OOM / loader error / timeout) is
# distinguishable from a real token mismatch.  Also prints GPU free before/after.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory containing the GGUF files}"\nM="${MODEL:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' \\r')
echo "[case] GPU free before: ${free0} MiB"
for i in 1 2; do
  echo "--- attempt $i ---"
  env ENGINE_GPU_EXPERTS=16 ./build/test_gen.exe --model "$M" \
      --ids 760,6511,314,9338,369 --n 12 --ctx 1024 2>&1 | tail -8
  echo "exit=$?"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | sed 's/^/  gpu free after: /'
done
