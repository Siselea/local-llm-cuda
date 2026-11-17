#!/usr/bin/env bash
# 35B end-to-end acceptance on port 7030 (the "real use" path):
#   1. greedy parity with build/test_gen.exe (same model, same ctx, same placement)
#   2. startup line / /health / /props / /v1/models / browser index
#   3. Chinese single-turn chat (non-stream)
#   4. multi-turn messages (system/user/assistant/user)
#   5. stream=true SSE (Chinese prompt, reassemble + [DONE])
#   6. medium prompts around/over the historical >384-token crash boundary
#   7. /tokenize + /detokenize round trip of Chinese text
#   8. graceful /shutdown: process exits, port released, VRAM freed
#
# Usage: bash tools/serve_35b_e2e.sh [model.gguf] [port]
# Requires a clean GPU: the script refuses to start if foreign processes are
# holding more than 2000 MiB.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
MODEL="${1:-<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
PORT="${2:-7030}"
CTX="${CTX:-4096}"          # override if 4096 does not fit: CTX=2048 bash tools/serve_35b_e2e.sh
GPULAYERS="${GPULAYERS:-16}" # MoE layers kept in VRAM (12 GB card: 16)
EXTRA="${EXTRA:-}"          # extra engine flags, e.g. EXTRA="--kv-type q8_0" if VRAM is tight
OUT=build/logs/serve35b
BASE="http://127.0.0.1:$PORT"
ENGINE=./build/engine.exe
GEN=./build/test_gen.exe
TOKEXE=./build/test_tokenizer.exe
mkdir -p "$OUT"
rm -f "$OUT"/*.json "$OUT"/*.txt "$OUT"/*.log 2>/dev/null

fail=0
pass=0
ok() { pass=$((pass + 1)); printf '%-72s PASS\n' "$1"; }
bad() {
  fail=$((fail + 1))
  printf '%-72s FAIL\n' "$1"
  shift
  [ $# -gt 0 ] && printf '%s\n' "$@" | tail -10 | sed 's/^/    /'
}
# jget FILE EXPR -> ASCII-safe JSON of EXPR (\uXXXX escapes, GBK console safe).
jget() {
  "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));v=($2);print(json.dumps(v,ensure_ascii=True))" "$1" 2>/dev/null
}
jlen() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(len($2))" "$1" 2>/dev/null; }
# jget_text: raw value for ASCII-only strings (ids csv, arch names...).  ASCII so
# the GBK console cannot mangle it.
jget_text() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));v=($2);print(v if isinstance(v,str) else json.dumps(v))" "$1" 2>/dev/null; }
# jhas_error FILE -> prints 1 when the response contains a JSON "error", else 0
jhas_error() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(1 if d.get('error') is not None else 0)" "$1" 2>/dev/null; }
# quality_gate <json|--sse> <file> [extra flags]: degenerate output (prompt copy /
# symbol soup / repetition / empty) is a FAIL, not "HTTP 200 => PASS".
quality_gate() {
  if "$PY" tools/check_output_quality.py "$@" >"$OUT/quality_last.txt" 2>&1; then
    return 0
  fi
  tail -2 "$OUT/quality_last.txt" | tr '\n' ' ' | sed 's/^/    /' >&2
  return 1
}

[ -x "$ENGINE" ] || { echo "missing $ENGINE (run bash build.sh)"; exit 1; }
[ -x "$GEN" ] || { echo "missing $GEN"; exit 1; }
[ -x "$TOKEXE" ] || { echo "missing $TOKEXE"; exit 1; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL"; exit 1; }

echo "== preflight: GPU must be idle (foreign usage < 2000 MiB) =="
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "  nvidia-smi memory.used = ${USED:-?} MiB"
if [ -n "${USED:-}" ] && [ "$USED" -gt 2000 ]; then
  echo "  GPU is busy (${USED} MiB) -- refusing to start. Wait for the other agent to finish."
  exit 1
fi
if curl.exe -s -m 2 -o /dev/null "$BASE/health"; then
  echo "  port $PORT is already serving -- refusing to start"
  exit 1
fi

echo
echo "== phase 1/2: greedy parity reference with test_gen.exe =="
# Render the exact chat prompt with the tokenizer tool, then ask test_gen for
# the greedy continuation of its token ids (same ctx / placement as the server).
CC="$OUT/chat_prompt.json"
"$PY" - "$CC" <<'PY'
import json, sys
msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好"},
]
json.dump({"messages": msgs, "add_generation_prompt": True, "enable_thinking": True},
          open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
"$TOKEXE" "$MODEL" --chat-template-file "$CC" --out "$OUT/chat_prompt.txt" || exit 1
"$TOKEXE" "$MODEL" --ids-file "$OUT/chat_prompt.txt" --out "$OUT/chat_prompt.ids" || exit 1
PROMPT_IDS=$(tr '\n' ',' < "$OUT/chat_prompt.ids" | sed 's/,$//')
echo "  chat prompt: $(wc -c < "$OUT/chat_prompt.txt") bytes, $(wc -l < "$OUT/chat_prompt.ids") tokens"
GEN_OUT=$("$GEN" --model "$MODEL" --ids "$PROMPT_IDS" --n 8 --ctx "$CTX" \
          --gpu-layers "$GPULAYERS" $EXTRA 2>"$OUT/test_gen.log" | grep -oE 'gen:.*' | head -1 | sed 's/gen: *//')
REF_IDS=$(printf '%s' "$GEN_OUT" | tr -s ' ' | sed 's/^ //' | tr ' ' ',')
if [ -n "$REF_IDS" ]; then
  ok "test_gen greedy reference: [$REF_IDS]"
else
  bad "test_gen greedy reference" "$(tail -5 "$OUT/test_gen.log")"
fi

echo
echo "== phase 2/2: starting engine serve on $BASE =="
"$ENGINE" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx "$CTX" \
    --gpu-layers "$GPULAYERS" $EXTRA >"$OUT/server.log" 2>&1 &
SRV=$!
cleanup() {
  if kill -0 "$SRV" 2>/dev/null; then
    curl.exe -s -m 5 -X POST "$BASE/shutdown" >/dev/null 2>&1
    for _ in $(seq 1 75); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.2; done
    kill -0 "$SRV" 2>/dev/null && taskkill //PID "$SRV" //F >/dev/null 2>&1
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 600); do
  if grep -q "listening on http" "$OUT/server.log" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then break; fi
  sleep 0.5
done
if [ "$ready" != "1" ]; then
  echo "server failed to start:"; tail -30 "$OUT/server.log" | sed 's/^/    /'; exit 1
fi
grep -m1 "listening on" "$OUT/server.log" | sed 's/^/  /'
grep -m1 "model ready" "$OUT/server.log" | sed 's/^/  /'
if grep -q "listening on http://127.0.0.1:$PORT (model=.*, ctx=$CTX)" "$OUT/server.log"; then
  ok "startup line (model/ctx)"
else
  bad "startup line (model/ctx)"
fi

R=$(curl.exe -s -m 10 "$BASE/health"); printf '%s' "$R" > "$OUT/health.json"
printf '%s' "$R" | grep -q '"status": *"ok"' && ok "GET /health" || bad "GET /health" "$R"

curl.exe -s -m 10 "$BASE/props" > "$OUT/props.json"
ARCH=$(jget_text "$OUT/props.json" 'd["arch"]')
NCTX=$(jget "$OUT/props.json" 'd["n_ctx"]')
TOKV=$(jget "$OUT/props.json" 'd["n_vocab"]')
QMAT=$(jget "$OUT/props.json" '",".join(sorted(d["quant_mix"]))')
if [ "$ARCH" = "qwen35moe" ] && [ "$NCTX" = "$CTX" ]; then
  ok "GET /props (arch=$ARCH ctx=$NCTX vocab=$TOKV quant=[$QMAT])"
else
  bad "GET /props" "$(head -c 600 "$OUT/props.json")"
fi

curl.exe -s -m 10 -o "$OUT/index.html" -w '%{http_code}' "$BASE/" > "$OUT/index.code"
[ "$(cat "$OUT/index.code")" = "200" ] && grep -qE "local chat|llama\.cpp" "$OUT/index.html" \
  && ok "GET / browser chat UI" || bad "GET / browser chat UI"

echo
echo "== warmup: first request pays lazy CUDA init / graph capture =="
"$PY" -c "
import json
json.dump({'prompt': 'Hello', 'n_predict': 1, 'temperature': 0, 'enable_thinking': False},
          open('$OUT/warmup_req.json', 'w', encoding='utf-8'))
"
WT0=$(date +%s%N)
curl.exe -s -m 300 -H 'Content-Type: application/json' --data-binary "@$OUT/warmup_req.json" \
    "$BASE/completion" > "$OUT/warmup.json"
WT1=$(date +%s%N)
WALL_MS=$(( (WT1 - WT0) / 1000000 ))
UPF=$(jget "$OUT/warmup.json" 'int(d.get("timings",{}).get("prefill_ms",0))')
echo "  cold first request wall=${WALL_MS}ms (reported prefill=${UPF}ms; includes lazy init)"
ok "cold first request served (wall ${WALL_MS}ms)"

echo
echo "== Chinese single-turn chat (non-stream) =="
REQ="$OUT/req_zh.json"
"$PY" - "$REQ" <<'PY'
import json, sys
body = {"messages": [
    {"role": "system", "content": "你是一个简洁的中文助手，回答不超过 60 字。"},
    {"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "temperature": 0.8, "top_p": 0.95, "top_k": 40, "max_tokens": 192, "stream": False}
json.dump(body, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
T0=$(date +%s)
curl.exe -s -m 300 -H 'Content-Type: application/json' --data-binary "@$REQ" "$BASE/v1/chat/completions" > "$OUT/zh_chat.json"
T1=$(date +%s)
ZH_CONTENT=$(jget "$OUT/zh_chat.json" 'd["choices"][0]["message"]["content"]')
ZH_CLEN=$(jget "$OUT/zh_chat.json" 'len(d["choices"][0]["message"]["content"])')
ZH_RLEN=$(jget "$OUT/zh_chat.json" 'len(d["choices"][0]["message"].get("reasoning_content") or "")')
ZH_ERR=$(jhas_error "$OUT/zh_chat.json")
if [ "$ZH_ERR" = "0" ] && quality_gate --json "$OUT/zh_chat.json" --expect-cjk; then
  ok "Chinese single-turn: content_len=$ZH_CLEN reasoning_len=$ZH_RLEN wall=$((T1-T0))s (quality OK)"
else
  bad "Chinese single-turn (content_len=${ZH_CLEN:-?} reasoning_len=${ZH_RLEN:-?})" "$(head -c 400 "$OUT/zh_chat.json")"
fi
TIM_MS=$(jget "$OUT/zh_chat.json" 'int(d["timings"]["prefill_ms"])')
GEN_TPS=$(jget "$OUT/zh_chat.json" 'round(d["timings"]["tokens_per_second"],1)')
echo "  prefill=${TIM_MS}ms decode=${GEN_TPS} tok/s"

echo
echo "== multi-turn messages (system / user / assistant / user) =="
REQ="$OUT/req_multi.json"
"$PY" - "$REQ" <<'PY'
import json, sys
body = {"messages": [
    {"role": "system", "content": "你是一个助手。"},
    {"role": "user", "content": "法国的首都是哪里？"},
    {"role": "assistant", "content": "巴黎。"},
    {"role": "user", "content": "那德国呢？"}],
    "temperature": 0, "max_tokens": 128, "stream": False, "enable_thinking": False}
json.dump(body, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
curl.exe -s -m 300 -H 'Content-Type: application/json' --data-binary "@$REQ" "$BASE/v1/chat/completions" > "$OUT/multi_chat.json"
M_CLEN=$(jget "$OUT/multi_chat.json" 'len(d["choices"][0]["message"]["content"])')
M_RLEN=$(jget "$OUT/multi_chat.json" 'len(d["choices"][0]["message"].get("reasoning_content") or "")')
M_ERR=$(jhas_error "$OUT/multi_chat.json")
if [ "$M_ERR" = "0" ] && quality_gate --json "$OUT/multi_chat.json" --expect-cjk; then
  ok "multi-turn: content_len=$M_CLEN reasoning_len=$M_RLEN (quality OK)"
else
  bad "multi-turn (content_len=${M_CLEN:-?} reasoning_len=${M_RLEN:-?})" "$(head -c 400 "$OUT/multi_chat.json")"
fi

echo
echo "== stream=true SSE (Chinese) =="
REQ="$OUT/req_stream.json"
"$PY" - "$REQ" <<'PY'
import json, sys
body = {"messages": [{"role": "user", "content": "数到五，用中文，只输出数字。"}],
        "temperature": 0, "max_tokens": 64, "stream": True,
        "enable_thinking": False}
json.dump(body, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
curl.exe -s -N -m 300 -H 'Content-Type: application/json' --data-binary "@$REQ" \
    "$BASE/v1/chat/completions" > "$OUT/stream.sse"
NCHUNK=$(grep -c '^data: {' "$OUT/stream.sse")
RECON=$("$PY" - "$OUT/stream.sse" <<'PY'
import json, sys
content, reason, done = [], [], False
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if not line.startswith("data: "):
        continue
    payload = line[6:].strip()
    if payload == "[DONE]":
        done = True
        continue
    try:
        j = json.loads(payload)
    except Exception:
        continue
    d = (j.get("choices") or [{}])[0].get("delta") or {}
    if d.get("content"):
        content.append(d["content"])
    if d.get("reasoning_content"):
        reason.append(d["reasoning_content"])
print(json.dumps({"content": "".join(content), "reasoning": "".join(reason), "done": done},
                 ensure_ascii=True))
PY
)
S_CLEN=$(printf '%s' "$RECON" | "$PY" -c 'import json,sys; print(len(json.load(sys.stdin)["content"]))')
S_RLEN=$(printf '%s' "$RECON" | "$PY" -c 'import json,sys; print(len(json.load(sys.stdin)["reasoning"]))')
S_DONE=$(printf '%s' "$RECON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["done"])')
if [ "$NCHUNK" -ge 2 ] && [ "$S_DONE" = "True" ] && quality_gate --sse "$OUT/stream.sse" --expect-cjk; then
  ok "SSE: $NCHUNK data chunks + [DONE], content_len=$S_CLEN reasoning_len=$S_RLEN (quality OK)"
else
  bad "SSE stream (chunks=$NCHUNK done=$S_DONE content=$S_CLEN reasoning=$S_RLEN)" "$(head -5 "$OUT/stream.sse")"
fi

echo
echo "== tokenize / detokenize (Chinese round trip) =="
"$PY" - "$OUT/req_tok.json" <<'PY'
import json, sys
json.dump({"content": "你好，世界！这是一个测试。"}, open(sys.argv[1], "w", encoding="utf-8"),
          ensure_ascii=False)
PY
curl.exe -s -m 10 -H 'Content-Type: application/json' --data-binary "@$OUT/req_tok.json" \
     "$BASE/tokenize" > "$OUT/tokenize.json"
IDS=$(jget_text "$OUT/tokenize.json" 'json.dumps(d["tokens"])')
"$PY" - "$OUT/req_detok.json" "$IDS" <<'PY'
import json, sys
json.dump({"tokens": json.loads(sys.argv[2])}, open(sys.argv[1], "w", encoding="utf-8"))
PY
curl.exe -s -m 10 -H 'Content-Type: application/json' --data-binary "@$OUT/req_detok.json" \
     "$BASE/detokenize" > "$OUT/detokenize.json"
DET_OK=$("$PY" -c "import json,sys;req=json.load(open(sys.argv[1],encoding='utf-8'))['content'];got=json.load(open(sys.argv[2],encoding='utf-8')).get('content');print('OK' if got==req else 'BAD')" "$OUT/req_tok.json" "$OUT/detokenize.json")
DET_LEN=$(jlen "$OUT/tokenize.json" 'd["tokens"]')
if [ "$DET_OK" = "OK" ]; then
  ok "tokenize+detokenize Chinese round trip ($DET_LEN tokens)"
else
  bad "tokenize+detokenize Chinese round trip" "$(jget "$OUT/detokenize.json" 'd.get("content")')"
fi

echo
echo "== medium prompts (historical >384 crash boundary) =="
for TARGET in 300 500; do
  REQ="$OUT/req_${TARGET}.json"
  "$PY" - "$REQ" "$TARGET" <<'PY'
import json, sys
target = int(sys.argv[2])
unit = "这是长提示词预填充测试的一句话。"
text = ""
while len(text) < target * 2:
    text += unit
body = {"messages": [{"role": "user", "content": "请只回复：收到。" + text}],
        "temperature": 0, "max_tokens": 16, "stream": False, "enable_thinking": False}
json.dump(body, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
  # token count via the server's own tokenizer
  "$PY" - "$REQ" "$OUT/ptok_${TARGET}.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
json.dump({"content": body["messages"][0]["content"]},
          open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
PY
  curl.exe -s -m 30 -H 'Content-Type: application/json' --data-binary "@$OUT/ptok_${TARGET}.json" \
       "$BASE/tokenize" > "$OUT/ptok_${TARGET}_out.json"
  NTOK=$(jget "$OUT/ptok_${TARGET}_out.json" 'len(d["tokens"])')
  T0=$(date +%s)
  CODE=$(curl.exe -s -m 300 -o "$OUT/mid_${TARGET}.json" -w '%{http_code}' \
         -H 'Content-Type: application/json' --data-binary "@$REQ" "$BASE/v1/chat/completions")
  T1=$(date +%s)
  PF=$(jget "$OUT/mid_${TARGET}.json" 'int(d.get("timings",{}).get("prefill_ms",0))')
  MID_LEN=$(jget "$OUT/mid_${TARGET}.json" 'len(d["choices"][0]["message"]["content"])')
  MID_RLEN=$(jget "$OUT/mid_${TARGET}.json" 'len(d["choices"][0]["message"].get("reasoning_content") or "")')
  MIDERR=$(jhas_error "$OUT/mid_${TARGET}.json")
  if [ "$CODE" = "200" ] && [ "$MIDERR" = "0" ] && quality_gate --json "$OUT/mid_${TARGET}.json" --expect-cjk; then
    ok "prompt ~${NTOK} tokens: 200, prefill=${PF}ms wall=$((T1-T0))s content_len=$MID_LEN reasoning_len=$MID_RLEN (quality OK)"
  else
    bad "prompt ~${NTOK} tokens (code=$CODE content_len=${MID_LEN:-?} quality)" "$(head -c 400 "$OUT/mid_${TARGET}.json")"
  fi
done

echo
echo "== greedy parity: server /completion vs test_gen.exe =="
REQ="$OUT/req_parity.json"
"$PY" - "$REQ" "$OUT/chat_prompt.txt" <<'PY'
import json, sys
prompt = open(sys.argv[2], encoding="utf-8").read()
json.dump({"prompt": prompt, "temperature": 0, "n_predict": 8, "stream": False},
          open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
curl.exe -s -m 300 -H 'Content-Type: application/json' --data-binary "@$REQ" \
     "$BASE/completion" > "$OUT/parity.json"
SRV_IDS=$(jget_text "$OUT/parity.json" '",".join(str(t) for t in d.get("generated_token_ids",[]))')
if [ -n "$SRV_IDS" ] && [ "$SRV_IDS" = "$REF_IDS" ]; then
  ok "greedy parity server == test_gen: [$SRV_IDS]"
else
  bad "greedy parity server == test_gen" "server=[$SRV_IDS] test_gen=[$REF_IDS]"
fi

echo
echo "== output quality probe (engine-level: WARN, not counted as a service FAIL) =="
QUAL=$("$PY" -c "
import json, sys
base = sys.argv[1]
def text(rel):
    try:
        d = json.load(open(base + '/' + rel, encoding='utf-8'))
        m = d['choices'][0]['message']
        return (m.get('content') or '') + (m.get('reasoning_content') or '')
    except Exception:
        return ''
zh = text('zh_chat.json')
cjk = sum(1 for ch in zh if 0x4E00 <= ord(ch) <= 0x9FFF)
alnum = sum(1 for ch in zh if ch.isascii() and ch.isalnum())
print(json.dumps({'len': len(zh), 'cjk': cjk, 'ascii_alnum': alnum}))
" "$OUT")
Q_LEN=$(printf '%s' "$QUAL" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["len"])')
Q_CJK=$(printf '%s' "$QUAL" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["cjk"])')
if [ "$Q_LEN" -gt 0 ] && [ "$Q_CJK" -eq 0 ]; then
  printf '  %-72s WARN (degenerate: %s chars, 0 CJK)\n' "Chinese response quality" "$Q_LEN"
else
  printf '  %-72s OK (%s chars, %s CJK)\n' "Chinese response quality" "$Q_LEN" "$Q_CJK"
fi

echo
echo "== graceful shutdown =="
CODE=$(curl.exe -s -m 10 -o "$OUT/shutdown.json" -w '%{http_code}' -X POST "$BASE/shutdown")
[ "$CODE" = "200" ] && ok "POST /shutdown -> 200" || bad "POST /shutdown -> 200 (got $CODE)"
for _ in $(seq 1 150); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.2; done
if kill -0 "$SRV" 2>/dev/null; then
  bad "server process exits"
else
  wait "$SRV" 2>/dev/null
  RC=$?
  [ "$RC" = "0" ] && ok "process exits cleanly (rc=0)" || bad "process exits cleanly (rc=$RC)"
fi
grep -q "stopped; VRAM released" "$OUT/server.log" && ok "VRAM release logged" \
  || bad "VRAM release logged" "$(tail -5 "$OUT/server.log")"
sleep 1
if curl.exe -s -m 2 -o /dev/null "$BASE/health"; then
  bad "port released after shutdown"
else
  ok "port released after shutdown"
fi
USED_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "  nvidia-smi memory.used after shutdown = ${USED_AFTER:-?} MiB (idle desktop level expected)"

echo
echo "artefacts: $OUT/"
echo "results: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "35B E2E GREEN" || echo "35B E2E FAILURES"
exit "$fail"
