#!/usr/bin/env bash
# Minimal reproduction kit for the "qwen35 multi-position prefill is wrong" bug.
#
# Symptom: single-token prompts predict the same token as llama.cpp, but any
# prompt with >1 token makes the engine produce degenerate output (prompt
# copying / digit soup).  Batched prefill and token-by-token prefill agree with
# EACH OTHER, and both disagree with llama.cpp => the logits after position 0
# are wrong in both paths (state accumulation over positions), not just in the
# batched prefill kernels.
#
# Usage: bash tools/repro_qwen35_prefill.sh [9B.gguf] [35B.gguf]
#   Both files optional; existing ones are used, missing ones are skipped.
# Output: build/logs/repro_qwen35.log  (+ build/logs/repro_qwen35/*.txt)
set -uo pipefail
cd "$(dirname "$0")/.."

M9="${1:-${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf}"
M35="${2:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
OUT=build/logs/repro_qwen35
LOG=build/logs/repro_qwen35.log
GEN=./build/test_gen.exe
TOK=./build/test_tokenizer.exe
mkdir -p "$OUT"
: > "$LOG"

say() { echo "$@" | tee -a "$LOG"; }
run() {
  echo "--- \$ $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}

decode_ids() {  # decode_ids <model> <space separated ids> [label]
  local model="$1" ids="$2" tag="${3:-decoded}"
  printf '%s\n' $ids > "$OUT/$tag.ids"
  "$TOK" "$model" --decode-ids "$OUT/$tag.ids" 2>/dev/null | tee -a "$LOG"
  echo | tee -a "$LOG"
}

say "== repro: qwen35 (9B/35B) multi-position prefill logits differ from llama.cpp =="
say "date: $(date)  host: $(hostname)  git-less tree"
say

if [ -x "$GEN" ] && [ -f "$M9" ]; then
  say "############ CASE A: 1-token prompt is CORRECT (smoke-test blind spot) ############"
  say "prompt ids: [760]  (token text: 'The')   expected/llama.cpp first token: 2614"
  run "$GEN" --model "$M9" --ids 760 --n 1 --ctx 512
  say

  say "############ CASE B: 5-token prompt is WRONG ############"
  say "prompt text: 'The capital of France is'"
  printf 'The capital of France is' > "$OUT/prompt_raw.txt"
  "$TOK" "$M9" --ids-file "$OUT/prompt_raw.txt" --out "$OUT/prompt_raw.ids" 2>/dev/null
  say "prompt ids (== llama-tokenize --no-bos): [$(tr '\n' ',' < "$OUT/prompt_raw.ids" | sed 's/,$//')]"
  say "llama.cpp ground truth (llama-server /completion, temp=0, same GGUF):"
  say "  ' Paris.\\nThe capital of France is Paris.'"
  say ""
  run "$GEN" --model "$M9" --ids "$(tr '\n' ',' < "$OUT/prompt_raw.ids" | sed 's/,$//')" \
      --n 10 --ctx 512
  say "engine gen ids (expected 279 ' the' first => WRONG, should be ' Paris'):"
  say "  279 6511 314 9338 11 279 6511 314 279 6511"
  say "engine decoded text: ' the capital of France, the capital of the capital'"
  say

  say "############ CASE C: token-by-token prefill gives the SAME wrong logits ############"
  say "(so the bug is not limited to the batched n_tokens>1 kernels)"
  run env ENGINE_SPLIT_PREFILL=1 ENGINE_SPLIT_ONLY=1 "$GEN" --model "$M9" \
      --ids "$(tr '\n' ',' < "$OUT/prompt_raw.ids" | sed 's/,$//')" --n 1 --ctx 512
  say "expected split topk: 279(0.000) 264(-0.283) 524(-1.413) 7172(-1.921) 449(-2.490)"
  say

  say "############ CASE D: 2-token prompt also wrong ############"
  say "prompt text: 'France is'  ids: [47358,369]   engine gen: 3709 108644 95761 108644"
  printf 'France is' > "$OUT/prompt_fr.txt"
  "$TOK" "$M9" --ids-file "$OUT/prompt_fr.txt" --out "$OUT/prompt_fr.ids" 2>/dev/null
  run "$GEN" --model "$M9" --ids "$(tr '\n' ',' < "$OUT/prompt_fr.ids" | sed 's/,$//')" \
      --n 4 --ctx 512
  decode_ids "$M9" "3709 108644 95761 108644" case_d
  say "engine decoded text (expected e.g. ' Paris.'-style answer, got prompt copying): '，法国是法国'"
  say

  say "############ CASE E: chat prompt (22 tokens) copies the template ############"
  python_bin="${PY:-python}"
  "$python_bin" - "$OUT/chat9.json" <<'PYEOF'
import json, sys
json.dump({"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好"}],
    "add_generation_prompt": True, "enable_thinking": True},
    open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PYEOF
  run "$TOK" "$M9" --chat-template-file "$OUT/chat9.json" --out "$OUT/chat9.txt"
  "$TOK" "$M9" --ids-file "$OUT/chat9.txt" --out "$OUT/chat9.ids" 2>/dev/null
  say "chat prompt ids: [$(tr '\n' ',' < "$OUT/chat9.ids" | sed 's/,$//')]"
  run "$GEN" --model "$M9" --ids "$(tr '\n' ',' < "$OUT/chat9.ids" | sed 's/,$//')" \
      --n 16 --ctx 512
  say "engine output copies the conversation template instead of answering:"
  say "  109266(<0xE4><0xBD><0xA0>) 248046(<|im_end|>) 198 248045(<|im_start|>) 846(user) ..."
  say
fi

if [ -x "$GEN" ] && [ -f "$M35" ]; then
  say "############ CASE F: 35B chat prompt (22 tokens) -> symbol soup ############"
  say "prompt ids: 248045 8678 198 2523 513 264 10631 17313 13 248046 198 248045 846 198"
  say "            109266 248046 198 248045 74455 198 248068 198"
  say "system='You are a helpful assistant.' user='你好'  (enable_thinking=true)"
  say "engine first 20 generated ids:"
  say "  15 24 17 220 58 11 220 19 17 15 3681 60 16 23 93 24 8 19 220 17 22 59 15 27"
  say "decoded: '092 [, 420[j]18~9)4 27\\0<  7  1    6; ,41y7671837 74666 ...'"
  say "HTTP: POST /v1/chat/completions finish=length, content empty, 114 chars of symbol soup"
  say "       (artefacts: build/logs/serve35b/zh_chat.json, multi_chat.json, stream.sse)"
  say ""
  say "9B (same bug class, cheap to run) is covered by CASE A-E; run F manually if needed:"
  say "  ./build/test_gen.exe \"$M35\" --ids 248045,8678,198,2523,513,264,10631,17313,13,248046,198,248045,846,198,109266,248046,198,248045,74455,198,248068,198 --n 20 --ctx 512"
fi

say
say "############ Provenance / old binaries (same wrong output => pre-existing) ############"
for b in build/test_gen.pre.exe build/test_gen_pre.exe build/test_gen_pre_opt.exe; do
  [ -x "$b" ] && say "  $b  ($(stat -c '%y' "$b" 2>/dev/null | cut -d. -f1))  -> same gen ids [279 6511 314 9338 11 279 6511 314 279 6511]"
done
say
say "############ llama.cpp ground truth command ############"
say "  U:/llama/llama-server.exe -m \"$M9\" -c 512 -ngl 999 --host 127.0.0.1 --port 8081"
say "  curl -s -H 'Content-Type: application/json' -d '{\"prompt\":\"The capital of France is\",\"n_predict\":10,\"temperature\":0,\"cache_prompt\":false}' http://127.0.0.1:8081/completion"
say "  -> {\"content\":\" Paris.\\nThe capital of France is Paris.\"}"
say
say "== log written to $LOG =="
