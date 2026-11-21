#!/usr/bin/env bash
# Is Coder's served text quality caused by the MLA-graph change, or pre-existing?
# Runs the same greedy request against three server configurations and compares the
# generated text byte-for-byte:
#   A) base binary (12:00, no invariant fix, no MLA graph), graph paths as they were
#   B) new binary with ENGINE_NO_MLA_ATTN_GRAPH=1 (everything else identical)
#   C) new binary default (MLA attention graph + pinned async routing uploads)
#
#   bash build/compare_coder_serve_text.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR="${MODEL_DIR:-D:/p0/models}"
MODEL="$MODEL_DIR/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
PY="${PY:-D:/p0/py312/python.exe}"
export PYTHONIOENCODING=utf-8
LOGD=build/logs
mkdir -p "$LOGD"

ask() {  # <binary> <extra env> <outfile>
  local bin="$1" envs="$2" out="$3" port=7041
  rm -f "build/logs/serve_cmp.log"
  eval "$envs \"$bin\" serve \"$MODEL\" --host 127.0.0.1 --port $port --ctx 1024" \
    >build/logs/serve_cmp.log 2>&1 &
  local srv=$!
  for _ in $(seq 1 600); do
    grep -q "listening on http" build/logs/serve_cmp.log 2>/dev/null && break
    sleep 0.25
  done
  local prompt='def is_palindrome(s):
    return'
  local body
  body=$("$PY" -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"max_tokens":48,"temperature":0,"top_k":1}))' "$prompt")
  curl.exe -s -m 300 -H 'Content-Type: application/json' -d "$body" \
    "http://127.0.0.1:$port/v1/completions" >"$out"
  curl.exe -s -m 5 -X POST "http://127.0.0.1:$port/shutdown" >/dev/null 2>&1
  for _ in $(seq 1 80); do kill -0 "$srv" 2>/dev/null || break; sleep 0.2; done
  kill -0 "$srv" 2>/dev/null && taskkill //PID "$srv" //F >/dev/null 2>&1
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8",errors="replace")); t=d["choices"][0].get("text",""); open(sys.argv[1]+".txt","wb").write(t.encode("utf-8","surrogateescape")); print(json.dumps(t[:160]))' "$out" 2>/dev/null
}

echo "== A) base binary =="
A=$(ask ./build/engine_ek_base.exe "" "$LOGD/coder_text_A.json"); echo "  $A"
echo "== B) new binary, MLA graph OFF =="
B=$(ask ./build/engine.exe "ENGINE_NO_MLA_ATTN_GRAPH=1" "$LOGD/coder_text_B.json"); echo "  $B"
echo "== C) new binary, MLA graph ON =="
C=$(ask ./build/engine.exe "" "$LOGD/coder_text_C.json"); echo "  $C"

echo
cmp -s "$LOGD/coder_text_B.json.txt" "$LOGD/coder_text_C.json.txt" \
  && echo "graph on/off text: IDENTICAL ($(md5sum "$LOGD/coder_text_C.json.txt" | cut -c1-16))" \
  || echo "graph on/off text: DIFFERS"
cmp -s "$LOGD/coder_text_A.json.txt" "$LOGD/coder_text_C.json.txt" \
  && echo "base vs new text  : IDENTICAL ($(md5sum "$LOGD/coder_text_A.json.txt" | cut -c1-16))" \
  || echo "base vs new text  : DIFFERS"
