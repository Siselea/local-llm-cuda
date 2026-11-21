#!/usr/bin/env bash
# Calibrate vram_extra_reserve_mb: the VRAM the engine allocates that its own ledger
# (cuda::g_buf_bytes) cannot see -- CUDA context, cuBLAS workspaces, allocator padding.
# The auto layer placement reserves that amount, otherwise the post-load margin is
# optimistic and the driver pages (measured 25 tok/s -> 3 tok/s).
#
# Each line prints: model, gpu_moe_layers, post-load free, ledger gap, verdict.
# Run on an EXCLUSIVE GPU (the free-VRAM probe is device-wide).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
GEN=./build/test_gen.exe
ROUNDS="${ROUNDS:-2}"

probe() {   # name model [extra args...]
  local name="$1" model="$2"; shift 2
  [ -f "$model" ] || { echo "== $name: missing model, skipped"; return; }
  echo "== $name =="
  for i in $(seq 1 "$ROUNDS"); do
    out=$("$GEN" --model "$model" --ids 549,6077 --n 1 "$@" 2>&1)
    layers=$(printf '%s' "$out" | grep -oE 'gpu_moe_layers=[0-9]+' | head -1 | cut -d= -f2)
    vram=$(printf '%s' "$out" | grep -E '\[vram\] (check OK|ERROR):' | head -1)
    gap=$(printf '%s' "$out" | grep -oE 'ledger gap -?[0-9]+ MiB' | head -1)
    free=$(printf '%s' "$out" | grep -oE 'post-load used=[0-9]+ MiB free=[0-9]+' | head -1)
    echo "  run$i: layers=${layers:-?} | ${free:-?} | ${gap:-no-gap} | ${vram##*: }"
  done
}

probe "Coder-V2-Lite (MoE, the flaky case)" "${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
probe "Qwen3.5-9B (dense)"                "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
probe "Qwen3.6-35B-A3B (MoE, big)"        "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
echo
echo "=> the recommended vram_extra_reserve_mb is the largest |ledger gap| seen for the"
echo "   models that use automatic placement (MoE), rounded up with ~256 MiB headroom."
