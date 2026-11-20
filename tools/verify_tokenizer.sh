#!/usr/bin/env bash
# Tokenizer acceptance test (reproducible):
#   1. generated Unicode tables are up to date
#   2. C++ self-test on every local GGUF: decode(encode(s)) == s, special tokens
#      atomic, ChatML chat template shape + ids
#   3. token-id parity with llama.cpp's own tokenizer (llama-tokenize) on the
#      synthetic corpus AND on real repo documents (thousands of tokens)
#   4. round-trip of a chat-template prompt
#
# Usage: bash tools/verify_tokenizer.sh [--quick]
#   --quick: skip the llama-tokenize oracle comparison (self-test only)
#
# llama-tokenize is taken from $LLAMA_TOKENIZE (default: llama-tokenize.exe on PATH).
# Without it, oracle checks are skipped (self-test still runs).
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
EXE=build/test_tokenizer.exe
LLAMA_TOKENIZE="${LLAMA_TOKENIZE:-llama-tokenize.exe}"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

fail=0
pass=0
ok() { pass=$((pass + 1)); printf '%-74s PASS\n' "$1"; }
bad() {
  fail=$((fail + 1))
  printf '%-74s FAIL\n' "$1"
}

echo "== build artefacts =="
[ -x "$EXE" ] || { echo "missing $EXE (run bash build.sh)"; exit 1; }
[ -x "$PY" ] || { echo "missing python: $PY"; exit 1; }

echo
echo "== generated Unicode tables =="
if "$PY" tools/gen_unicode_tables.py --check; then
  ok "include/engine/unicode_tables.h up to date"
else
  bad "include/engine/unicode_tables.h up to date"
fi

MODELS=(
  "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
  "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
  "${MODEL_DIR}/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"
  "${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  "${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
)

for m in "${MODELS[@]}"; do
  [ -f "$m" ] || { echo "  skip (missing): $m"; continue; }
  name=$(basename "$m")
  echo
  echo "== $name =="
  if "$EXE" "$m" --self-test > build/logs/tok_selftest.log 2>&1; then
    ok "self-test: roundtrip + specials + chat template"
  else
    bad "self-test: roundtrip + specials + chat template"
    tail -20 build/logs/tok_selftest.log | sed 's/^/    /'
  fi
  if [ "$QUICK" = "1" ]; then
    echo "  (--quick: oracle comparison skipped)"
    continue
  fi
  if [ -x "$LLAMA_TOKENIZE" ]; then
    if "$PY" tools/check_tokenizer.py cpp --model "$m" --exe "$EXE" \
         --llama-tokenize "$LLAMA_TOKENIZE" > build/logs/tok_oracle.log 2>&1; then
      ok "llama.cpp tokenize parity (corpus + real documents + template)"
    else
      bad "llama.cpp tokenize parity (corpus + real documents + template)"
      tail -25 build/logs/tok_oracle.log | sed 's/^/    /'
    fi
  else
    echo "  (llama-tokenize not found at $LLAMA_TOKENIZE: oracle comparison skipped)"
  fi
done

echo
echo "== encode/decode CLI smoke (whichever model is present) =="
M=""
for m in "${MODELS[@]}"; do
  [ -f "$m" ] && { M="$m"; break; }
done
if [ -n "$M" ]; then
  printf 'hello world, 你好世界!\n' > build/logs/tok_smoke.txt
  if "$EXE" "$M" --ids-file build/logs/tok_smoke.txt --out build/logs/tok_smoke.ids &&
     "$EXE" "$M" --decode-ids build/logs/tok_smoke.ids --out build/logs/tok_smoke.out &&
     cmp -s build/logs/tok_smoke.txt build/logs/tok_smoke.out; then
    ok "ids-file encode + decode-ids round trip"
  else
    bad "ids-file encode + decode-ids round trip"
  fi
fi

echo
echo "results: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "TOKENIZER GREEN" || echo "TOKENIZER FAILURES"
exit "$fail"
