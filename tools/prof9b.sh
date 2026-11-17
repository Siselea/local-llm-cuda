#!/usr/bin/env bash
# 9B profiling: kprof table, layer-diff, lmhead contribution, head breakdown.
set -uo pipefail
cd "$(dirname "$0")/.."
M=<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf
B=./build/test_gen.exe
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }

echo "=== A. graph-mode layer diff (W(N)) ==="
for L in 1 2 8 16 32; do
  v=$(ENGINE_TOPK=1 ENGINE_LAYERS=$L ENGINE_SKIP_WARMUP=0 $B "$M" 760 9 2>&1 | dec)
  echo "layers=$L wall=$v"
done

echo "=== B. skip lmhead (graph, full) ==="
for i in 1 2; do
  v=$(ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 ENGINE_SKIP_LMHEAD=1 $B "$M" 760 9 2>&1 | dec)
  echo "skiplmhead=$v"
done

echo "=== C. ENGINE_DBG_FWD breakdown ==="
ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 ENGINE_DBG_FWD=1 $B "$M" 760 3 2>&1 | grep -iE 'fwd|head|logits|sample|lmhead|layer' | head -20

echo "=== D. kprof (--no-graph, 2 tokens) ==="
ENGINE_KPROF=1 ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 $B "$M" 760 2 --no-graph 2>&1 | grep -E '^\s*\[kprof\]|kprof' | head -40

echo "=== E. placement / vram ==="
ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 ENGINE_DBG_VRAM=1 $B "$M" 760 2 2>&1 | grep -iE 'placement|vram|free' | head -10
