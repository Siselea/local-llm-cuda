#!/usr/bin/env bash
# Greedy-generation regression against llama.cpp ground truth.
#
# WHY THIS EXISTS
# ---------------
# The old tests only checked the *first* token after a 1-token prompt.  On GGUF
# Qwen3.5/Qwen3.6 checkpoints this was blind to two semantic mismatches between the
# HF formulas the engine used to implement and the llama.cpp convention the GGUF
# files are actually written in:
#   * qwen35.ssm_a stores -A (not log A)  -> the SSM decay was ~exp(-softplus)
#   * the q/k heads broadcast to v heads by block repeat (h % n_kh), not by
#     repeat_interleave (h / rep)
# Both leave position 0 almost intact and destroy every later position, so the
# single-token smoke test stayed green while real prompts generated prompt-copying /
# symbol soup.  This script therefore prefills a *real* multi-token prompt and
# compares 8-16 greedy tokens, id by id, with llama.cpp.
#
# The expected ids below were captured from llama.cpp (U:/llama) via
#   POST /completion {"prompt": ..., "n_predict": n, "temperature": 0,
#                     "cache_prompt": false, "n_probs": 1}
# and read back from completion_probabilities[].id (llama-server, same GGUF file).
#
# Usage:  bash tools/gen_regression.sh
#         bash tools/gen_regression.sh 9b        # only cases whose tag starts with 9b
# Missing model files are reported and skipped (like smoke_models.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
GEN=./build/test_gen.exe
M_QWEN="${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
M_CODER="${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
M_9B="${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
M_35B="${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"

FILTER="${1:-}"
pass=0
fail=0

# check <tag> <model> <env> <ids> <n> <expected ids...>
check() {
  local tag="$1" model="$2" envs="$3" ids="$4" n="$5"
  shift 5
  local expect="$*"
  if [ -n "$FILTER" ] && [[ "$tag" != $FILTER* ]]; then return 0; fi
  if [ ! -f "$model" ]; then printf '%-28s SKIP (model missing)\n' "$tag"; return 0; fi
  if [ ! -x "$GEN" ]; then printf '%-28s FAIL (missing %s; run bash build.sh)\n' "$tag" "$GEN"; fail=$((fail + 1)); return 1; fi
  # shellcheck disable=SC2086
  local out gen
  out=$(env $envs "$GEN" --model "$model" --ids "$ids" --n "$n" --ctx 1024 2>&1)
  gen=$(printf '%s\n' "$out" | sed -n 's/^gen: //p' | head -1)
  if [ "$gen" = "$expect" ]; then
    printf '%-28s PASS (%d greedy tokens == llama.cpp)\n' "$tag" "$n"
    pass=$((pass + 1))
  else
    printf '%-28s FAIL\n    expected: %s\n    got     : %s\n' "$tag" "$expect" "$gen"
    printf '%s\n' "$out" | grep -E "^gen:" >/dev/null
    fail=$((fail + 1))
  fi
}

echo "== qwen35 GGUF semantics: greedy generation == llama.cpp =="
# The capital of France is   (llama.cpp: " Paris.\nThe capital of France is Paris.\nThe")
check 9b_capital_of_france "$M_9B" "" \
  "760,6511,314,9338,369" 12 \
  11751 13 198 760 6511 314 9338 369 11751 13 198 760
# The   (llama.cpp: " following are multiple choice questions (with answers) about")
check 9b_the "$M_9B" "" \
  "760" 10 \
  2614 513 5081 5576 4602 318 4058 10926 8 883
# France is   (llama.cpp: " a country in Western Europe. It is")
check 9b_france_is "$M_9B" "" \
  "47358,369" 8 \
  264 3046 303 10550 4357 13 1049 369
# 你好  (llama.cpp ids match; token ids from tokenizer --ids-file)
check 9b_zh_nihao "$M_9B" "" \
  "109266" 8 \
  3709 95815 111354 103801 3709 96846 96442 97235
# chat prompt: system "You are a helpful assistant." + user 你好, thinking on, 22 ids
# (llama.cpp: "Thinking Process:\n\n1.  **")
check 9b_chat22 "$M_9B" "" \
  "248045,8678,198,2523,513,264,10631,17313,13,248046,198,248045,846,198,109266,248046,198,248045,74455,198,248068,198" \
  8 \
  90700 8340 25 271 16 13 220 2972

echo
echo "== qwen36-MoE GGUF semantics: greedy generation == llama.cpp =="
# The capital of France is  (35B weights differ, hence its own ground truth)
check 35b_capital_of_france "$M_35B" "ENGINE_GPU_EXPERTS=16" \
  "760,6511,314,9338,369" 12 \
  11751 11 264 3177 34756 364 1141 25438 57902 1680 430 279
# chat prompt (system "You are a helpful assistant." + user 你好, thinking on)
check 35b_chat22 "$M_35B" "ENGINE_GPU_EXPERTS=16" \
  "248045,8678,198,2523,513,264,10631,17313,13,248046,198,248045,846,198,109266,248046,198,248045,74455,198,248068,198" \
  8 \
  40 2688 28378 310 279 1156 579 40719
# 你好 (35B tokenizer/weights differ from the 9B, so it needs its own ground truth)
check 35b_zh_nihao "$M_35B" "ENGINE_GPU_EXPERTS=16" \
  "109266" 8 \
  3709 95815 111354 98292 97432 3709 99443 97706

echo
echo "== non-qwen35 stacks must not regress =="
check coder_v2_lite "$M_CODER" "ENGINE_GPU_EXPERTS=16" "549,6077,280,7239" 1 185
check qwen2_5_0b "$M_QWEN" "" "1000,1001" 1 40716

echo
echo "results: $pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
  echo "GEN REGRESSION GREEN"
else
  echo "GEN REGRESSION FAILURES"
fi
exit "$((fail > 0 ? 1 : 0))"
