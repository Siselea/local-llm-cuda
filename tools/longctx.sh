#!/usr/bin/env bash
# Long-context probe for the 9B (the model that is GEMV-bandwidth bound and fully
# GPU-resident): short prompt vs 2048-token prefill at --ctx 4096, same decode count,
# plus the top-5 to check correctness of the long-context path.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BIN:-"$BIN"}"
M9=${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf
IDS=$(cat build/ids2048.txt)
N="${N:-17}"
echo "=== 9B short prompt, ctx 4096 ==="
for r in 1 2 3; do
  ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 "$BIN" "$M9" "760" "$N" --ctx 4096 2>&1 |
    grep -E "^\[time\]|^topk:|^gen:" | tr '\n' ' '; echo
done
echo "=== 9B 2048-token prompt, ctx 4096 ==="
for r in 1 2 3; do
  ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 "$BIN" "$M9" "$IDS" "$N" --ctx 4096 2>&1 |
    grep -E "^\[time\]|^topk:|^gen:" | tr '\n' ' '; echo
done
echo "=== 9B 2048-token prompt, ctx 8192 (does a larger KV change anything?) ==="
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 "$BIN" "$M9" "$IDS" "$N" --ctx 8192 2>&1 |
  grep -E "^\[time\]|^topk:" | tr '\n' ' '; echo
echo "=== 9B short prompt, default ctx (no --ctx) for the pre-change comparison ==="
for r in 1 2; do
  ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 "$BIN" "$M9" "760" "$N" 2>&1 |
    grep -E "^\[time\]|^topk:" | tr '\n' ' '; echo
done
echo DONE-LONGCTX
