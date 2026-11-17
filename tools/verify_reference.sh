#!/usr/bin/env bash
# Extended evidence: compare the engine against the OFFICIAL transformers reference
# implementations for the two hybrid/MLA models (slow, ~15 minutes).
#
#   bash tools/verify_reference.sh
#
# The quick suite (tools/verify.sh) is the one that must stay ALL GREEN; this script
# reproduces the numbers quoted in SPEC-COMPLIANCE.md section 4.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
M9="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf"
MOE="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
CD="<MODEL_DIR>/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"

echo "=== Qwen3.5-9B (GGUF): engine vs llama.cpp-semantics reference (layers 0-7) ==="
# IMPORTANT: a GGUF qwen35/qwen35moe file is written with the *llama.cpp* conventions
#   ssm_a = -A                    (not log A)
#   q/k -> v head broadcast = h % n_kh   (block repeat, not repeat_interleave)
# so matching the official HF modules (which implement log A + repeat_interleave) is
# the WRONG criterion for a GGUF file: it is exactly how the "single-token smoke is
# green, real text is prompt-copying" bug survived.  tools/q35_variants.py implements
# the GGUF convention; in this mode it is validated against the engine below.
# tools/q35_torch_chain.py stays the oracle for *safetensors* checkpoints only.
# ENGINE_PREFILL_F16=0 forces the exact-f32 prefill GEMM path so the comparison against
# the exact numpy reference is not polluted by the fp16 weight-dequant prefill path
# (which is within ~1e-3 relative by design).
ENGINE_SKIP_WARMUP=1 ENGINE_PREFILL_F16=0 ENGINE_DUMP_EVERY=build/ref9b ./build/test_gen.exe "$M9" \
  "760,6511,314,9338" 1 >/dev/null 2>&1
"$PY" tools/q35_variants.py --model "$M9" --dump build/ref9b --ids "760,6511,314,9338" \
  --a-fix 1 --bcast tile --layer 7 --tolerance 2e-4 2>/dev/null \
  | grep -E "^layer|DUMP CHECK"

echo
echo "=== Coder-V2-Lite: engine vs official deepseek_v2 reference (layer 0-2) ==="
ENGINE_GPU_EXPERTS=16 ENGINE_DUMP_EVERY=build/refcd ./build/test_gen.exe "$CD" \
  "549,6077,280,7239" 1 >/dev/null 2>&1
"$PY" tools/dsv2_torch_chain.py --dump build/refcd --ids "549,6077,280,7239" --layers 0-2 \
  --norm-plus1 2>/dev/null | grep -E "^layer|top5"

echo
echo "=== engine predictions for the same prompts ==="
ENGINE_SKIP_WARMUP=1 ENGINE_TOPK=1 ./build/test_gen.exe "$M9" "760,6511,314,9338" 1 2>&1 \
  | grep -oE "topk: [0-9]*\([-0-9.]*\)" | head -1
ENGINE_GPU_EXPERTS=16 ENGINE_TOPK=1 ./build/test_gen.exe "$CD" "549,6077,280,7239" 1 2>&1 \
  | grep -oE "topk: [0-9]*\([-0-9.]*\)" | head -1

rm -rf build/ref9b build/refcd
echo
echo "Reference: Qwen3.5-9B GGUF 1-token = llama.cpp 2614;"
echo "           Qwen3.5-9B GGUF 4-token 'The capital of France' = llama.cpp 369 (' is');"
echo "           Coder-V2-Lite 4-token = 185."
