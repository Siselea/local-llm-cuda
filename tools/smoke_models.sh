#!/usr/bin/env bash
# Smoke test on the four real local models: load, run, print the key numbers.
# Missing models are reported and skipped (the script never fails on absence).
#
#   bash tools/smoke_models.sh
set -uo pipefail
cd "$(dirname "$0")/.."
M_QWEN="${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
M_CODER="${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
M_9B="${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
M_35B="${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"

run() {
  local tag="$1" model="$2" ids="$3" n="$4" envs="$5" expect="$6"
  if [ ! -f "$model" ]; then
    printf "%-22s SKIP (model not found)\n" "$tag"
    return
  fi
  printf "%-22s " "$tag"
  local out
  out=$(env ENGINE_TOPK=1 $envs ./build/test_gen.exe "$model" "$ids" "$n" 2>&1)
  local top
  top=$(printf "%s" "$out" | grep -oE "topk: [0-9]+\([-0-9.]+\)" | head -1)
  local speed
  speed=$(printf "%s" "$out" | grep -oE "decode [0-9.]+ ms/token" | head -1)
  printf "%s | %s | expected top-1 %s\n" "${top:-no topk}" "${speed:-no decode}" "$expect"
}

echo "building (if needed)..."
[ -x build/test_gen.exe ] || bash build.sh >/dev/null || exit 1

run "qwen2.5-0.5b"    "$M_QWEN"  "1000,1001" 33 "ENGINE_SKIP_WARMUP=0" "40716"
run "Coder-V2-Lite"   "$M_CODER" "549,6077,280,7239" 1 "ENGINE_GPU_EXPERTS=16" "185"
run "Qwen3.5-9B"      "$M_9B"    "760" 1 "ENGINE_SKIP_WARMUP=0"       "2614"
run "Qwen3.6-35B-A3B" "$M_35B"   "760" 1 "ENGINE_GPU_EXPERTS=16"        "2614"

echo
echo "Reference points (all four re-measured against llama.cpp greedy decoding):"
echo "  qwen2.5-0.5b  : 40716 (greedy, own tokenizer)"
echo "  Coder-V2-Lite : 185   (matches the official deepseek_v2 reference)"
echo "  Qwen3.5-9B    : 2614  (llama.cpp, prompt id 760 'The')"
echo "  Qwen3.6-35B   : 2614  (llama.cpp, prompt id 760 'The')"
echo
echo "REMEMBER: a single-token top-1 check is NOT enough for the qwen35 hybrid stacks --"
echo "it stays green even when the SSM tensor semantics are wrong (position 0 is nearly"
echo "degenerate).  Run tools/gen_regression.sh, which prefills a multi-token prompt and"
echo "compares 8-16 greedy token ids with llama.cpp."

