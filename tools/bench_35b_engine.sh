#!/usr/bin/env bash
# 35B two-turn comparison run: start `engine serve`, benchmark, shut down.
#
# usage: bash tools/bench_35b_engine.sh <tag> [mtp 0|1] [think_budget] [ctx] [kv_type]
#   e.g. bash tools/bench_35b_engine.sh q8mtpoff 0 512 262144 q8_0
#
# Writes:
#   build/logs/bench_<tag>.serve.log   engine startup/placement + request log
#   build/logs/bench_<tag>.json        per-turn metrics + answer/reasoning text
set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: bench_35b_engine.sh <tag> [mtp 0|1] [think_budget] [ctx] [kv_type] [max_tokens]}"
MTP="${2:-0}"
BUDGET="${3:-512}"
CTX="${4:-262144}"
KVTYPE="${5:-q8_0}"
MAXTOK="${6:-4096}"
: "${MODEL_DIR:?set MODEL_DIR to the directory containing the GGUF files}"\nMODEL="${MODEL:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
PORT="${PORT:-7030}"
SERVE_LOG="build/logs/bench_${TAG}.serve.log"
BENCH_JSON="build/logs/bench_${TAG}.json"
mkdir -p build/logs

# port + GPU guard (never fight another server)
if curl.exe -s -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: port ${PORT} already has a server"; exit 1
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
if [ "${USED:-0}" -gt 2000 ]; then
  echo "ERROR: GPU busy (${USED} MiB used)"; exit 2
fi

ARGS=(serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" --ctx "$CTX"
      --kv-type "$KVTYPE" --gpu-layers -1 --win-affinity 2 --cpu-int8 1 --win-priority 1
      --default-max-tokens 1048576 --dev-route --layer-graph-auto --think-budget "$BUDGET")
# Omit --threads by default so the engine uses its SMT-paired 2x physical default;
# set THREADS=9 (or any N) to reproduce the old 1.5x/OS-scheduled baseline.
if [ -n "${THREADS:-}" ]; then ARGS+=(--threads "$THREADS"); fi
if [ "$MTP" = "1" ]; then ARGS+=(--mtp --mtp-k 2); fi

echo "=== run ${TAG}: mtp=${MTP} think_budget=${BUDGET} ctx=${CTX} kv=${KVTYPE} max_tokens=${MAXTOK}"
rm -f "$SERVE_LOG"
"./build/engine.exe" "${ARGS[@]}" >"$SERVE_LOG" 2>&1 &
PID=$!
READY=0
for _ in $(seq 1 360); do
  if grep -q "listening on http" "$SERVE_LOG" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then break; fi
  sleep 0.5
done
if [ "$READY" != "1" ]; then
  echo "ERROR: server did not become ready; tail of log:"
  tail -30 "$SERVE_LOG"
  kill "$PID" 2>/dev/null
  exit 3
fi
grep -E "model ready|prefix-cache|sampling|placement\] (decision|gpu_moe_layers|vram|ledger)|\[ctx\]|vram\] (post-load|check)|\[mtp\]" "$SERVE_LOG" | tail -20

GPU_LOG="build/logs/bench_${TAG}.gpu.log"
: > "$GPU_LOG"
( while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null
    sleep 1
  done ) &
GPUSAMPLER=$!
python() { "${PY:-python}" "$@"; }
python tools/serve_twoturn_bench.py --port "$PORT" --tag "$TAG" --out "$BENCH_JSON" --max-tokens "$MAXTOK"
RC=$?
kill "$GPUSAMPLER" 2>/dev/null

curl.exe -s -m 10 -X POST "http://127.0.0.1:${PORT}/shutdown" >/dev/null 2>&1
for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
kill -9 "$PID" 2>/dev/null
echo "=== after shutdown: nvidia-smi = $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
exit "$RC"
