#!/usr/bin/env bash
# Long-context request check for `engine serve` (default: ~4000-token prompt).
#
#   bash tools/serve_longctx_check.sh [model.gguf] [port] [ctx] [target_tokens]
#
# Starts the server, builds a Chinese prompt of ~target_tokens tokens (measured
# with the server's own /tokenize), sends one /v1/chat/completions request,
# reports prefill latency + generated length, then shuts the server down.
# Exit 0 requires: HTTP 200, timings present, and a non-empty answer.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
MODEL="${1:-<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf}"
PORT="${2:-7034}"
CTX="${3:-8192}"
TARGET="${4:-4000}"
EXTRA="${EXTRA:-}"          # e.g. EXTRA="--kv-type q8_0" if 12 GB VRAM is tight
OUT=build/logs/serve_longctx
BASE="http://127.0.0.1:$PORT"
ENGINE=./build/engine.exe
mkdir -p "$OUT"
rm -f "$OUT"/* 2>/dev/null

jget() {
  "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));v=($2);print(json.dumps(v,ensure_ascii=True))" "$1" 2>/dev/null
}
jhas_error() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(1 if d.get('error') is not None else 0)" "$1" 2>/dev/null; }

fail=0
pass=0
ok() { pass=$((pass + 1)); printf '%-68s PASS\n' "$1"; }
bad() {
  fail=$((fail + 1))
  printf '%-68s FAIL\n' "$1"
  shift
  [ $# -gt 0 ] && printf '%s\n' "$@" | tail -8 | sed 's/^/    /'
}

[ -x "$ENGINE" ] || { echo "missing $ENGINE"; exit 1; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL"; exit 1; }

echo "== preflight =="
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "  nvidia-smi memory.used = ${USED:-?} MiB"
if [ -n "${USED:-}" ] && [ "$USED" -gt 2000 ]; then
  echo "  GPU busy; refuse to start (wait for the other agent)"
  exit 1
fi
if curl.exe -s -m 2 -o /dev/null "$BASE/health"; then
  echo "  port $PORT already serving"; exit 1
fi

echo "== starting serve (ctx=$CTX, target prompt ~$TARGET tokens) =="
"$ENGINE" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx "$CTX" $EXTRA >"$OUT/server.log" 2>&1 &
SRV=$!
cleanup() {
  if kill -0 "$SRV" 2>/dev/null; then
    curl.exe -s -m 5 -X POST "$BASE/shutdown" >/dev/null 2>&1
    for _ in $(seq 1 200); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.2; done
    kill -0 "$SRV" 2>/dev/null && MSYS_NO_PATHCONV=1 taskkill /PID "$SRV" /F >/dev/null 2>&1
  fi
}
trap cleanup EXIT
for _ in $(seq 1 900); do
  grep -q "listening on http" "$OUT/server.log" 2>/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "server died:"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 0.5
done
grep -m1 "listening on" "$OUT/server.log" | sed 's/^/  /'
grep -m1 "model ready" "$OUT/server.log" | sed 's/^/  /'
ok "server started"

echo "== building a ~$TARGET token Chinese prompt via /tokenize =="
"$PY" - "$OUT" "$TARGET" "$PORT" <<'PYEOF'
import json, sys, urllib.request
out, target, port = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base = "http://127.0.0.1:" + port
# NOTE: edits below are re-run with the port baked in by the shell via env
unit = "这是一段用于验证长上下文预填充与多轮对话稳定性的中文测试文本，请继续对话。"
text = ""
while True:
    text += unit
    body = json.dumps({"content": text}, ensure_ascii=False).encode()
    req = urllib.request.Request(base + "/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    n = len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"])
    if n >= target:
        break
json.dump({"content": text, "n_tokens": n}, open(out + "/long_prompt.json", "w", encoding="utf-8"),
          ensure_ascii=True)
print(f"  prompt tokens = {n} ({len(text)} chars)")
PYEOF
NTOK=$(jget "$OUT/long_prompt.json" 'd["n_tokens"]')
[ -n "$NTOK" ] && ok "prompt tokenized ($NTOK tokens)" || bad "prompt tokenized"

echo "== request =="
"$PY" - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
d = json.load(open(out + "/long_prompt.json", encoding="utf-8"))
body = {"messages": [{"role": "user", "content": d["content"] + "\n请只回复：收到。"}],
        "temperature": 0, "max_tokens": 32, "stream": False, "enable_thinking": False}
json.dump(body, open(out + "/req.json", "w", encoding="utf-8"), ensure_ascii=False)
PYEOF
T0=$(date +%s)
CODE=$(curl.exe -s -m 900 -o "$OUT/resp.json" -w '%{http_code}' \
       -H 'Content-Type: application/json' --data-binary "@$OUT/req.json" "$BASE/v1/chat/completions")
T1=$(date +%s)
PF=$(jget "$OUT/resp.json" 'int(d.get("timings",{}).get("prefill_ms",0))')
CLEN=$(jget "$OUT/resp.json" 'len(d["choices"][0]["message"]["content"])')
RLEN=$(jget "$OUT/resp.json" 'len(d["choices"][0]["message"].get("reasoning_content") or "")')
ERR=$(jhas_error "$OUT/resp.json")
if [ "$CODE" = "200" ] && [ "$ERR" = "0" ] && [ -n "$CLEN" ] && [ "$CLEN" -gt 0 ] 2>/dev/null; then
  ok "long-context chat 200: prefill=${PF}ms wall=$((T1-T0))s content_len=$CLEN reasoning_len=$RLEN"
  PP=$("$PY" -c "print(round($NTOK*1000.0/max($PF,1),1))" 2>/dev/null)
  echo "  prefill speed ≈ ${PP} tok/s over $NTOK tokens"
else
  bad "long-context chat (code=$CODE)" "$(head -c 500 "$OUT/resp.json")"
fi

echo "== shutdown =="
curl.exe -s -m 10 -o /dev/null -X POST "$BASE/shutdown"
for _ in $(seq 1 200); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.2; done
kill -0 "$SRV" 2>/dev/null && bad "server exits" || ok "server exits (port released)"
if grep -q "stopped; VRAM released" "$OUT/server.log"; then ok "VRAM release logged"; else bad "VRAM release logged"; fi

echo
echo "artefacts: $OUT/"
echo "results: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "LONGCTX GREEN" || echo "LONGCTX FAILURES"
exit "$fail"
