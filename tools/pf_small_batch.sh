#!/usr/bin/env bash
# Small-batch (tail) prefill curve + quantized-GEMM (no dequant) comparison.
#
# WHY: a cache-hit multi-turn request only prefills a short tail (e.g. 52 tokens) and
# measured ~233 ms => 4.5 ms/token, while a 128-token chunk runs ~1.5 ms/token.  The
# suspicion is that every forward pays a *fixed* cost (the whole-model dequant to fp16
# ~137 ms on a 9B + per-matmul launch overhead), not that small n is inherently slow.
# This probe measures prefill tok/s for target lengths 8/16/32/52/64/128/256 with:
#   rows=512  + prefill_f16=1 (default: fp16 tensor cores with dequant staging)
#   rows=512  + ENGINE_GEMM=1 (self-developed quantized GEMM, no fp16 mirror)
# so the fixed cost can be split from the per-token cost and the tail strategy decided.
#
# Needs an exclusive GPU (~1 min for both runs).
#   bash tools/pf_small_batch.sh [model.gguf]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf}
CASES=${CASES:-8,16,32,52,64,128,256}
CTX=${CTX:-2048}
PORT=${PORT:-7077}

run_case() {   # tag env...
  local tag="$1"; shift
  echo "===== $tag ====="
  env "$@" python tools/prefill_cliff_probe.py \
      --model "$MODEL" --ctx "$CTX" --rows 512 --rows-max 512 \
      --port "$PORT" --tag "smallbatch-$tag" --cases "$CASES" 2>&1 | grep -E "req|healthy"
}

run_case fp16_default ENGINE_PREFILL_F16=1
run_case quant_gemm  ENGINE_PREFILL_F16=0 ENGINE_GEMM=1
run_case f32_plain   ENGINE_PREFILL_F16=0 ENGINE_GEMM=0

echo "=== summary (prefill tok/s vs prompt tokens) ==="
python - <<'PY'
import glob, json, os
for f in sorted(glob.glob('build/prefill_cliff_smallbatch-*.jsonl')):
    tag = os.path.basename(f).replace('prefill_cliff_', '').replace('.jsonl', '')
    rows = []
    for ln in open(f, encoding='utf-8', errors='replace'):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if 'prompt_tokens' in d:
            rows.append((d['prompt_tokens'], d.get('prefill_ms', 0), d.get('prefill_tok_s', 0)))
    if not rows:
        continue
    print(f'-- {tag}')
    for n, ms, tps in rows:
        print(f'   n={n:5d}  prefill={ms:8.1f} ms  {tps:7.1f} tok/s  '
              f'({ms / max(n,1):5.2f} ms/token)')
    ns = [r[0] for r in rows]
    ms = [r[1] for r in rows]
    if len(ns) >= 2:
        # linear fit ms = fixed + slope*n  -> fixed cost and per-token cost
        mx = sum(ns) / len(ns)
        my = sum(ms) / len(ms)
        den = sum((x - mx) ** 2 for x in ns)
        slope = sum((x - mx) * (y - my) for x, y in zip(ns, ms)) / den if den else 0.0
        print(f'   fit: fixed ~ {my - slope * mx:.1f} ms, per-token ~ {slope:.2f} ms')
PY
