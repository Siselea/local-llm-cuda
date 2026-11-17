#!/usr/bin/env bash
# Final measurement matrix for PRE-FILL-REPORT.  Waits for a quiet GPU before every
# run and reports min-of-N; prints one CSV-ish line per configuration.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-./build/pf_gen.exe}
M9=<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf
M05=<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf
MCD=<MODEL_DIR>/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf
M35=<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
REPS=${REPS:-3}

wait_quiet() {
  for _ in $(seq 1 240); do
    local used util
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if [ "$used" -lt 900 ] && [ "$util" -lt 10 ]; then return 0; fi
    sleep 3
  done
  return 1
}

# one timed prefill: prints "<ms> <tok/s>"
prefill() {   # model ids n_gen ctx rows env...
  local model="$1" ids="$2" ngen="$3" ctx="$4" rows="$5"; shift 5
  local out
  out=$(env "$@" "$BIN" "$model" "$(cat "$ids")" "$ngen" --ctx "$ctx" --prefill-rows "$rows" 2>&1)
  local ms
  ms=$(echo "$out" | sed -n 's/.*prefill [0-9]* tokens: \([0-9.]*\) ms.*/\1/p')
  [ -z "$ms" ] && { echo "FAIL"; return 1; }
  awk -v m="$ms" -v n="$(awk -F, '{print NF}' "$ids")" 'BEGIN{printf "%.1f %.0f\n", m, n*1000/m}'
}

run() {   # label model ids ngen ctx rows env
  local label="$1" model="$2" ids="$3" ngen="$4" ctx="$5" rows="$6"; shift 6
  local best="" bestpp=""
  for r in $(seq 1 "$REPS"); do
    wait_quiet || { echo "$label,NO_QUIET_GPU"; return; }
    local res; res=$(prefill "$model" "$ids" "$ngen" "$ctx" "$rows" "$@")
    if [ "$res" = "FAIL" ]; then continue; fi
    local ms pp; ms=${res% *}; pp=${res#* }
    if [ -z "$best" ] || awk -v a="$ms" -v b="$best" 'BEGIN{exit !(a<b)}'; then best="$ms"; bestpp="$pp"; fi
  done
  echo "$label,ms=${best:-na},pp=${bestpp:-na}"
}

echo "=== 9B prefill 1622 tokens (ctx 4096) ==="
run "9B_default(all opts)"     "$M9" build/ids1622.txt 4 4096 512
run "9B_no_gqa"                "$M9" build/ids1622.txt 4 4096 512 ENGINE_ATTN_GQA=0
run "9B_no_gd_reg"             "$M9" build/ids1622.txt 4 4096 512 ENGINE_GD_REG=0
run "9B_no_attn_pre"           "$M9" build/ids1622.txt 4 4096 512 ENGINE_ATTN_PRE_ROWS=1 ENGINE_ATTN_GQA=0
run "9B_f32_gemm(baseline2)"   "$M9" build/ids1622.txt 4 4096 512 ENGINE_PREFILL_F16=0
run "9B_rows_max_512(4chunk)"  "$M9" build/ids1622.txt 4 4096 512 ENGINE_PREFILL_ROWS_MAX=512
run "9B_rows_max_1024(2chunk)" "$M9" build/ids1622.txt 4 4096 512 ENGINE_PREFILL_ROWS_MAX=1024
run "9B_f16_acc(experimental)" "$M9" build/ids1622.txt 4 4096 512 ENGINE_PREFILL_F16_ACC=1
run "9B_skip_gemm(non-gemm)"   "$M9" build/ids1622.txt 4 4096 512 ENGINE_SKIP_GEMM=1

echo "=== 9B prefill, ctx 131072 (llama.cpp comparison condition) ==="
run "9B_ctx128k"               "$M9" build/ids1622.txt 4 131072 512

echo "=== 9B decode (n=17, dec_skip=2) ==="
for r in 1 2 3; do
  wait_quiet
  ENGINE_DEC_SKIP=2 "$BIN" "$M9" 760 17 --ctx 512 2>&1 | sed -n 's/.*decode \([0-9.]*\) ms\/token.*/9B_decode_ms=\1/p'
done

echo "=== other models: prefill (short prompt) + decode ==="
run "0.5B_prefill_33tok"       "$M05" build/ids32.txt 33 512 512
run "Coder_prefill"            "$MCD" build/ids128.txt 9 512 512
run "35B_prefill_128tok"       "$M35" build/ids128.txt 9 512 512
for r in 1 2; do
  wait_quiet
  for pair in "0.5B|$M05|1000,1001|33" "Coder|$MCD|549,6077,280,7239|9"; do
    IFS='|' read -r nm mdl ids ng <<< "$pair"
    out=$("$BIN" "$mdl" "$ids" "$ng" --ctx 512 2>&1 | sed -n 's/.*decode \([0-9.]*\) ms\/token.*/'"$nm"'_decode_ms=\1/p')
    echo "$out"
  done
done
wait_quiet
"$BIN" "$M35" 760 9 --ctx 512 2>&1 | sed -n 's/.*decode \([0-9.]*\) ms\/token.*/35B_decode_ms=\1/p'
echo "=== done ==="
