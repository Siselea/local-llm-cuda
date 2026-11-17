#!/usr/bin/env bash
# End-to-end HTTP server test (lightweight model, local port).
#
#   bash tools/verify_serve.sh [model.gguf] [port]
#
# Defaults: qwen2.5-0.5b on port 7031 (small/fast, does not need the 35B).
# Checks: startup line, /health, /v1/models, /props, browser UI, /tokenize +
# /detokenize, non-stream + SSE streaming chat/completions, llama.cpp
# /completion, 404/400 errors, greedy parity with build/test_gen.exe, and a
# graceful /shutdown that releases the port.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
MODEL="${1:-${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
PORT="${2:-7031}"
SKIP_PARITY="${SKIP_PARITY:-0}"  # set 1 for models that cannot fit twice in VRAM
ENGINE=./build/engine.exe
GEN=./build/test_gen.exe
LOG=build/logs/serve_verify.log
mkdir -p build/logs

fail=0
pass=0
ok() { pass=$((pass + 1)); printf '%-70s PASS\n' "$1"; }
bad() {
  fail=$((fail + 1))
  printf '%-70s FAIL\n' "$1"
  shift
  [ $# -gt 0 ] && printf '%s\n' "$@" | tail -8 | sed 's/^/    /'
}

[ -x "$ENGINE" ] || { echo "missing $ENGINE (run bash build.sh)"; exit 1; }
[ -x "$GEN" ] || { echo "missing $GEN (run bash build.sh)"; exit 1; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL"; exit 1; }

PYJSON() { "$PY" -c "$1" 2>/dev/null; }

echo "== starting engine serve ($(basename "$MODEL") on 127.0.0.1:$PORT) =="
rm -f "$LOG"
"$ENGINE" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx 1024 >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  if kill -0 "$SRV" 2>/dev/null; then
    curl.exe -s -m 5 -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    for _ in $(seq 1 50); do
      kill -0 "$SRV" 2>/dev/null || break
      sleep 0.2
    done
    kill -0 "$SRV" 2>/dev/null && taskkill //PID "$SRV" //F >/dev/null 2>&1
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 300); do
  if grep -q "listening on http" "$LOG" 2>/dev/null && \
     curl.exe -s -m 2 -o /dev/null -w '' "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SRV" 2>/dev/null; then break; fi
  sleep 0.2
done
if [ "$ready" != "1" ]; then
  echo "server did not start:"
  tail -30 "$LOG" | sed 's/^/    /'
  exit 1
fi

if grep -q "listening on http://127.0.0.1:$PORT (model=.*, ctx=1024)" "$LOG"; then
  ok "startup line: $(grep -m1 'listening on' "$LOG")"
else
  bad "startup line (model/ctx)" "$(grep -m1 'listening on' "$LOG")"
fi

BASE="http://127.0.0.1:$PORT"

echo
echo "== meta endpoints =="
R=$(curl.exe -s -m 10 "$BASE/health")
if printf '%s' "$R" | grep -q '"status": *"ok"'; then ok "GET /health"; else bad "GET /health" "$R"; fi
R=$(curl.exe -s -m 10 "$BASE/v1/models")
MODEL_EXPECT=$(basename "$MODEL" .gguf)
if printf '%s' "$R" | grep -qF "$MODEL_EXPECT"; then
  ok "GET /v1/models"
else
  bad "GET /v1/models (expected id '$MODEL_EXPECT')" "$R"
fi
R=$(curl.exe -s -m 10 "$BASE/props")
if printf '%s' "$R" | grep -q '"n_ctx"' && printf '%s' "$R" | grep -q '"tokenizer"'; then
  ok "GET /props"
else
  bad "GET /props" "$R"
fi
CODE=$(curl.exe -s -m 10 -o build/logs/serve_index.html -w '%{http_code}' "$BASE/")
if [ "$CODE" = "200" ] && grep -qE "local chat|llama\.cpp" build/logs/serve_index.html; then
  ok "GET / browser chat UI"
else
  bad "GET / browser chat UI (code=$CODE)"
fi

echo
echo "== tokenize / detokenize =="
TOK_BODY='{"content":"Hello world, 你好世界!"}'
R=$(curl.exe -s -m 10 -H 'Content-Type: application/json' -d "$TOK_BODY" "$BASE/tokenize")
IDS=$(printf '%s' "$R" | PYJSON 'import json,sys; print(",".join(str(t) for t in json.load(sys.stdin)["tokens"]))')
if [ -n "$IDS" ] && [ "$IDS" != "None" ]; then ok "POST /tokenize -> [$IDS]"; else bad "POST /tokenize" "$R"; fi
DET_BODY="{\"tokens\":[$IDS]}"
R=$(curl.exe -s -m 10 -H 'Content-Type: application/json' -d "$DET_BODY" "$BASE/detokenize")
if printf '%s' "$R" | grep -q 'Hello world, 你好世界!'; then
  ok "POST /detokenize round trip"
else
  bad "POST /detokenize round trip" "$R"
fi

echo
echo "== chat completions (non-stream) =="
# enable_thinking=false: Qwen3.5+ models otherwise spend small max_tokens budgets
# inside <think> and the answer would be empty (same issue as the 35B e2e).
BODY='{"model":"engine","messages":[{"role":"user","content":"Reply with exactly: OK"}],"temperature":0,"max_tokens":24,"stream":false,"enable_thinking":false}'
R=$(curl.exe -s -m 120 -H 'Content-Type: application/json' -d "$BODY" "$BASE/v1/chat/completions")
CHAT_CONTENT=$(printf '%s' "$R" | PYJSON 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))')
CHAT_FIN=$(printf '%s' "$R" | PYJSON 'import json,sys; print(json.load(sys.stdin)["choices"][0]["finish_reason"])')
printf '%s' "$R" > build/logs/serve_chat.json
if printf '%s' "$R" | grep -q '"object": *"chat.completion"' && [ -n "$CHAT_CONTENT" ] \
   && [ "$CHAT_CONTENT" != "''" ] && [ "$CHAT_CONTENT" != "None" ] \
   && "$PY" tools/check_output_quality.py --json build/logs/serve_chat.json --expect-ascii \
        > build/logs/serve_chat_quality.txt 2>&1; then
  ok "POST /v1/chat/completions -> content=$CHAT_CONTENT finish=$CHAT_FIN (quality OK)"
else
  bad "POST /v1/chat/completions" "$(head -c 500 "$R")" "$(tail -2 build/logs/serve_chat_quality.txt 2>/dev/null)"
fi

echo
echo "== chat completions (SSE stream) =="
curl.exe -s -N -m 120 -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count: 1 2 3"}],"temperature":0,"max_tokens":24,"stream":true,"enable_thinking":false}' \
  "$BASE/v1/chat/completions" > build/logs/serve_stream.txt
if grep -q '^data: ' build/logs/serve_stream.txt && grep -q 'data: \[DONE\]' build/logs/serve_stream.txt; then
  NCHUNKS=$(grep -c '^data: {' build/logs/serve_stream.txt)
  ok "SSE stream: $NCHUNKS chunks + [DONE]"
else
  bad "SSE stream" "$(head -5 build/logs/serve_stream.txt)"
fi

echo
echo "== completions endpoints =="
R=$(curl.exe -s -m 120 -H 'Content-Type: application/json' \
      -d '{"prompt":"The capital of France is","temperature":0,"max_tokens":8}' "$BASE/v1/completions")
if printf '%s' "$R" | grep -q '"object": *"text_completion"'; then
  ok "POST /v1/completions"
else
  bad "POST /v1/completions" "$R"
fi
R=$(curl.exe -s -m 120 -H 'Content-Type: application/json' \
      -d '{"prompt":"The capital of France is","temperature":0,"n_predict":8,"stream":false}' "$BASE/completion")
if printf '%s' "$R" | grep -q '"content"' && printf '%s' "$R" | grep -q '"tokens_predicted"'; then
  ok "POST /completion (llama.cpp style)"
else
  bad "POST /completion (llama.cpp style)" "$R"
fi
curl.exe -s -N -m 120 -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","temperature":0,"n_predict":4,"stream":true}' "$BASE/completion" \
  > build/logs/serve_llama_stream.txt
if grep -q '"stop": *true' build/logs/serve_llama_stream.txt; then
  ok "POST /completion SSE stream"
else
  bad "POST /completion SSE stream" "$(head -5 build/logs/serve_llama_stream.txt)"
fi

echo
echo "== error handling =="
CODE=$(curl.exe -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/no/such/path")
[ "$CODE" = "404" ] && ok "unknown path -> 404" || bad "unknown path -> 404 (got $CODE)"
CODE=$(curl.exe -s -m 10 -o build/logs/serve_err.txt -w '%{http_code}' \
        -d '{"messages":[]}' "$BASE/v1/chat/completions")
[ "$CODE" = "400" ] && ok "empty messages -> 400" || bad "empty messages -> 400 (got $CODE)"
CODE=$(curl.exe -s -m 10 -o /dev/null -w '%{http_code}' -d 'not json' "$BASE/v1/completions")
[ "$CODE" = "400" ] && ok "invalid JSON -> 400" || bad "invalid JSON -> 400 (got $CODE)"
LONGP=$(printf 'word %.0s' $(seq 1 1200))
CODE=$(curl.exe -s -m 20 -o build/logs/serve_long.txt -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$LONGP\",\"n_predict\":4}" "$BASE/completion")
if [ "$CODE" = "400" ] && grep -q "too long" build/logs/serve_long.txt; then
  ok "prompt > ctx -> 400 with clear error"
else
  bad "prompt > ctx -> 400 with clear error (got $CODE)"
fi
R=$(curl.exe -s -m 10 "$BASE/health")
if printf '%s' "$R" | grep -q '"status": *"ok"'; then
  ok "server still healthy after error responses"
else
  bad "server still healthy after error responses" "$R"
fi

echo
echo "== greedy parity with test_gen.exe (same seed / temperature 0) =="
if [ "$SKIP_PARITY" = "1" ]; then
  echo "  (SKIP_PARITY=1: skipped; use tools/serve_35b_e2e.sh for the two-phase 35B parity test)"
else
PROMPT="The capital of France is"
N=6
TOKS=$(curl.exe -s -m 10 -H 'Content-Type: application/json' \
        -d "{\"content\":\"$PROMPT\"}" "$BASE/tokenize" |
       PYJSON 'import json,sys; print(json.load(sys.stdin)["tokens"])')
IDS_LIST=$(printf '%s' "$TOKS" | PYJSON 'import json,sys; print(json.load(sys.stdin))')
IDS_CSV=$(printf '%s' "$IDS_LIST" | PYJSON 'import json,sys; print(",".join(str(t) for t in json.load(sys.stdin)))')
SRV_IDS=$(curl.exe -s -m 120 -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"temperature\":0,\"n_predict\":$N}" "$BASE/completion" |
       PYJSON 'import json,sys; print(",".join(str(t) for t in json.load(sys.stdin)["generated_token_ids"]))')
REF_OUT=$("$GEN" --model "$MODEL" --ids "$IDS_CSV" --n "$N" --ctx 1024 2>/dev/null |
          grep -oE 'gen:.*' | head -1 | sed 's/gen: *//' | tr -s ' ' | sed 's/^ //')
REF_IDS=$(printf '%s' "$REF_OUT" | tr ' ' ',')
if [ -n "$SRV_IDS" ] && [ "$SRV_IDS" = "$REF_IDS" ]; then
  ok "server greedy ids == test_gen ids [$SRV_IDS]"
else
  bad "server greedy ids == test_gen ids" "server=[$SRV_IDS] test_gen=[$REF_IDS]"
fi
fi

echo
echo "== graceful shutdown =="
CODE=$(curl.exe -s -m 10 -o build/logs/serve_shutdown.json -w '%{http_code}' -X POST "$BASE/shutdown")
[ "$CODE" = "200" ] && ok "POST /shutdown -> 200" || bad "POST /shutdown -> 200 (got $CODE)"
for _ in $(seq 1 100); do
  kill -0 "$SRV" 2>/dev/null || break
  sleep 0.2
done
if kill -0 "$SRV" 2>/dev/null; then
  bad "process exits after shutdown"
else
  wait "$SRV" 2>/dev/null
  RC=$?
  [ "$RC" = "0" ] && ok "process exits cleanly (rc=0)" || bad "process exits cleanly (rc=$RC)"
fi
if grep -q "stopped; VRAM released" "$LOG"; then
  ok "VRAM released message logged"
else
  bad "VRAM released message logged" "$(tail -5 "$LOG")"
fi
sleep 0.5
if curl.exe -s -m 2 -o /dev/null "$BASE/health"; then
  bad "port released after shutdown"
else
  ok "port released after shutdown"
fi

echo
echo "results: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "SERVE VERIFY GREEN" || echo "SERVE VERIFY FAILURES"
exit "$fail"
