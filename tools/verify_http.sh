#!/usr/bin/env bash
# HTTP-layer self-test (no model, no GPU, no runner.cpp dependency):
#   builds build/test_http.exe from src/http_server.cpp + tests/test_http.cpp,
#   starts it on a local port and drives it with tools/verify_http.py.
#
# Usage: bash tools/verify_http.sh [port]
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
PORT="${1:-7039}"
EXE=build/test_http.exe
LOG=build/logs/http_verify.log
mkdir -p build/logs

echo "== build test_http.exe (http_server + test only, no engine sources) =="
"$PY" -m ziglang c++ -std=c++17 -O2 -Wall -Iinclude src/http_server.cpp tests/test_http.cpp \
  -o "$EXE" || { echo "build failed"; exit 1; }
echo "  built $EXE"

rm -f "$LOG"
"$EXE" --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
cleanup() {
  if kill -0 "$SRV" 2>/dev/null; then
    curl.exe -s -m 3 -X POST "http://127.0.0.1:$PORT/shutdown" >/dev/null 2>&1
    for _ in $(seq 1 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.1; done
    kill -0 "$SRV" 2>/dev/null && taskkill //PID "$SRV" //F >/dev/null 2>&1
  fi
}
trap cleanup EXIT

for _ in $(seq 1 100); do
  grep -q "listening on" "$LOG" 2>/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "test_http died:"; cat "$LOG"; exit 1; }
  sleep 0.1
done

"$PY" tools/verify_http.py --host 127.0.0.1 --port "$PORT"
RC=$?

for _ in $(seq 1 50); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.1; done
if kill -0 "$SRV" 2>/dev/null; then
  echo "server did not exit after /shutdown"
  exit 1
fi
wait "$SRV" 2>/dev/null
SRV_RC=$?
if [ "$SRV_RC" -ne 0 ]; then
  echo "server exit code $SRV_RC"
  exit 1
fi

[ "$RC" -eq 0 ] && echo "HTTP SELF-TEST GREEN" || echo "HTTP SELF-TEST FAILURES"
exit "$RC"
