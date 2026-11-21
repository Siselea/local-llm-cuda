#!/usr/bin/env bash
# Clean interleaved A/B of the x8 dequant kernels on real prefills.
# Each round runs the OLD kernel path (ENGINE_DQ_OLD=1) and then the default (x8) on
# the same prompt sizes, so GPU drift affects both sides equally.  Exclusive GPU only.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the directory holding the .gguf models}"
ROUNDS="${ROUNDS:-2}"
CASES="${CASES:-3611,11020}"
for r in $(seq 1 "$ROUNDS"); do
  ENGINE_DQ_OLD=1 timeout 240 python tools/prefill_cliff_probe.py \
      --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" --ctx 32768 --rows 0 --rows-max 0 \
      --port $((7100 + r)) --prefix-cache 0 --tag "dqf_old$r" --cases "$CASES" 2>&1 |
      grep -E "req[0-9]" | sed "s/^/old r$r: /"
  timeout 240 python tools/prefill_cliff_probe.py \
      --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" --ctx 32768 --rows 0 --rows-max 0 \
      --port $((7110 + r)) --prefix-cache 0 --tag "dqf_new$r" --cases "$CASES" 2>&1 |
      grep -E "req[0-9]" | sed "s/^/new r$r: /"
done
