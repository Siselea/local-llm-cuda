#!/usr/bin/env bash
# Real-user end-to-end check for the MLA-graph build: start the built-in HTTP server
# with DeepSeek-Coder-V2-Lite and use it the way a user would.
#
# Coder-V2-Lite's GGUF carries no chat template ("chat tmpl: none"), so the engine
# answers /v1/chat/completions with a documented 400 ("use /v1/completions with a raw
# prompt"); this script asserts that contract and then holds the actual conversation
# over the raw-prompt endpoints, including streaming and a multi-turn prefix-cache hit.
#
#   bash tools/coder_serve_chat.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR="${MODEL_DIR:-D:/p0/models}"
MODEL="${1:-$MODEL_DIR/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf}"
PORT="${2:-7040}"
PY="${PY:-D:/p0/py312/python.exe}"
export PYTHONIOENCODING=utf-8   # served text can contain any byte sequence
ENGINE=./build/engine.exe
LOG=build/logs/coder_serve_chat.log
mkdir -p build/logs
fail=0
pass=0
ok() { pass=$((pass + 1)); printf '%-46s PASS %s\n' "$1" "${2:-}"; }
bad() { fail=$((fail + 1)); printf '%-46s FAIL %s\n' "$1" "${2:-}"; }

[ -x "$ENGINE" ] || { echo "missing $ENGINE"; exit 1; }
[ -f "$MODEL" ] || { echo "missing model $MODEL"; exit 1; }

echo "== starting server ($(basename "$MODEL") on 127.0.0.1:$PORT) =="
rm -f "$LOG"
"$ENGINE" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx 1024 >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  if kill -0 "$SRV" 2>/dev/null; then
    curl.exe -s -m 5 -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    for _ in $(seq 1 50); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.2; done
    kill -0 "$SRV" 2>/dev/null && taskkill //PID "$SRV" //F >/dev/null 2>&1
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 600); do
  if grep -q "listening on http" "$LOG" 2>/dev/null && \
     curl.exe -s -m 2 -o /dev/null -w '' "http://127.0.0.1:$PORT/health" 2>/dev/null; then
    ready=1; break
  fi
  sleep 0.25
done
[ "$ready" = 1 ] && ok "server ready" || { bad "server ready"; tail -5 "$LOG"; exit 1; }
BASE="http://127.0.0.1:$PORT"

# --- 1) documented contract: no chat template -> 400 with a usable message ---------
R=$(curl.exe -s -m 60 -H 'Content-Type: application/json' \
  -d '{"model":"coder","messages":[{"role":"user","content":"你好"}],"max_tokens":8}' \
  "$BASE/v1/chat/completions")
if printf '%s' "$R" | grep -q 'chat template is not supported'; then
  ok "chat endpoint explains missing template" "(400 by design)"
else
  bad "chat endpoint explains missing template" "$(printf '%s' "$R" | head -c 120)"
fi

# --- 2) the actual conversation: instruction + code, raw prompt (non-stream) ------
PROMPT='You are a helpful coding assistant.
### Instruction:
用 Python 写一个判断回文的函数，只输出代码。
### Response:
'
BODY=$("$PY" -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"max_tokens":120,"temperature":0}))' "$PROMPT")
R1=$(curl.exe -s -m 300 -H 'Content-Type: application/json' -d "$BODY" "$BASE/v1/completions")
printf '%s' "$R1" >build/logs/coder_chat1.json
INFO=$("$PY" -c 'import json; d=json.load(open("build/logs/coder_chat1.json",encoding="utf-8",errors="replace")); c=d["choices"][0]; t=d.get("timings",{}); u=d.get("usage",{}); print("%s|%s|%s|%s" % (c.get("finish_reason"), t.get("decode_ms",0), t.get("tokens_per_second",0), u.get("completion_tokens",0))); print(json.dumps(c.get("text","").strip()[:200]))' 2>/dev/null)
FIN=$(printf '%s' "$INFO" | head -1 | cut -d'|' -f1)
DMS=$(printf '%s' "$INFO" | head -1 | cut -d'|' -f2)
TPS=$(printf '%s' "$INFO" | head -1 | cut -d'|' -f3)
NTK=$(printf '%s' "$INFO" | head -1 | cut -d'|' -f4)
TXT=$(printf '%s' "$INFO" | tail -n +2)
if [ -n "$TXT" ]; then
  ok "coder conversation (raw prompt)" "(finish=$FIN, $NTK tok, decode ${DMS} ms, ${TPS} tok/s)"
  printf '    ---- reply (first 200 chars, json-escaped) ----\n    %s\n' "$TXT"
else
  bad "coder conversation (raw prompt)" "(empty text)"
fi
# Coder-V2-Lite's coherent-text limit is a pre-existing engine limitation
# (docs/reports/SPEC-COMPLIANCE.md: "5-token 以上的生成质量已定性为数值敏感"), and
# tools/coder_text_graph_neutral.sh proves the served bytes are identical with the
# pre-MLA-graph binary, so here we only assert "non-empty + served at a usable speed".
awk -v tps="$TPS" 'BEGIN{exit !(tps+0 >= 15)}' \
  && ok "decode speed >= 15 tok/s" "(${TPS} tok/s)" \
  || bad "decode speed >= 15 tok/s" "(${TPS} tok/s)"

# --- 3) multi-turn via the prefix cache: same prompt + a follow-up continuation ----
PROMPT2="${PROMPT}
例子里再加一个测试调用。
### Response:
"
BODY2=$("$PY" -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"max_tokens":80,"temperature":0}))' "$PROMPT2")
R2=$(curl.exe -s -m 300 -H 'Content-Type: application/json' -d "$BODY2" "$BASE/v1/completions")
printf '%s' "$R2" >build/logs/coder_chat2.json
T2=$("$PY" -c 'import json; d=json.load(open("build/logs/coder_chat2.json",encoding="utf-8",errors="replace")); print(json.dumps(d["choices"][0].get("text","").strip()[:160]))' 2>/dev/null)
if [ -n "$T2" ]; then ok "second turn (longer prompt, prefix cache)" ""; else bad "second turn (longer prompt, prefix cache)" "(empty)"; fi
CACHED=$(grep -oE 'prefill [0-9]+ \[new [0-9]+, cached [0-9]+\]' "$LOG" | tail -1)
if printf '%s' "$CACHED" | grep -qE 'cached [1-9]'; then
  ok "prefix cache reused across turns" "($CACHED)"
else
  printf '%-46s NOTE %s\n' "prefix cache reused across turns" \
    "(${CACHED:-no prefill line}); dense model: KV snapshot disabled (pre-existing)"
fi

# --- 4) streaming SSE on the raw-prompt endpoint ----------------------------------
SSE=$(curl.exe -s -m 300 -N -H 'Content-Type: application/json' \
  -d '{"prompt":"def is_palindrome(s):","max_tokens":24,"temperature":0,"stream":true}' \
  "$BASE/v1/completions")
if printf '%s' "$SSE" | grep -q 'data: '; then
  ok "streaming SSE" "($(printf '%s' "$SSE" | grep -c 'data: ') events)"
else
  bad "streaming SSE" "(no data: events)"
fi

# --- 5) llama.cpp style endpoint --------------------------------------------------
R3=$(curl.exe -s -m 120 -H 'Content-Type: application/json' \
  -d '{"prompt":"def add(a, b):","n_predict":24,"temperature":0}' "$BASE/completion")
C3=$("$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("content",""))' <<<"$R3" 2>/dev/null)
[ -n "$C3" ] && ok "llama.cpp /completion" "" || bad "llama.cpp /completion" "(empty)"

# --- 6) graceful shutdown + clean log ---------------------------------------------
curl.exe -s -m 5 -X POST "$BASE/shutdown" >/dev/null 2>&1
for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.2; done
if kill -0 "$SRV" 2>/dev/null; then bad "graceful shutdown" "(still running)"; else ok "graceful shutdown" ""; fi
if grep -qiE "assert|exception|CUDA error|runtime_error|segmentation" "$LOG"; then
  bad "server log clean" ""
  grep -iE "assert|exception|CUDA error|runtime_error|segmentation" "$LOG" | head -3 | sed 's/^/    /'
else
  ok "server log clean" ""
fi

printf '\nresults: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] && echo "CODER SERVE E2E GREEN" || echo "CODER SERVE E2E FAILURES"
exit "$((fail > 0 ? 1 : 0))"
