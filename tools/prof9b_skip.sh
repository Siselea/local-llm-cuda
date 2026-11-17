#!/usr/bin/env bash
# 9B: wall-clock attribution by skipping whole kernel classes (graph mode).
# ENGINE_SKIP_KERNELS drops the launch entirely; wall time difference = that class'
# in-model cost (numerics are junk, that is fine for attribution).
set -uo pipefail
cd "$(dirname "$0")/.."
M=${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf
B=./build/test_gen.exe
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }
run() {
  local tag="$1" envs="$2" vals=""
  for i in 1 2 3; do
    v=$(eval "ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 $envs \"$B\" \"$M\" 760 9 2>&1" | dec)
    vals="$vals ${v:-ERR}"
  done
  echo "$tag :$vals"
}
GEMVS="gemv_q5k_x4_dp4a_s,gemv_q6k_dp4a,gemv_q6k_dp4a_sk,gemv_q5k_x4_dp4a,gemv_q8_0_sk"
EDGE_SMALL="rmsnorm,quant_x_i8,gd_delta_rule_128f,rope_strided,add_inplace,zero_fill,attn_cached_w,q35_gate_mul,silu_mul,gd_ssm_prep,q35_split_qgate,kv_write"
echo "=== 9B graph-mode skip attribution ==="
run "baseline     " ""
run "skip 5 gemv  " "ENGINE_SKIP_KERNELS=$GEMVS"
run "skip 12 edge " "ENGINE_SKIP_KERNELS=$EDGE_SMALL"
run "skip rmsnorm " "ENGINE_SKIP_KERNELS=rmsnorm"
run "skip quant    " "ENGINE_SKIP_KERNELS=quant_x_i8"
run "skip add      " "ENGINE_SKIP_KERNELS=add_inplace"
run "skip zerofill " "ENGINE_SKIP_KERNELS=zero_fill"
run "skip rope     " "ENGINE_SKIP_KERNELS=rope_strided"
run "skip qkvsplit " "ENGINE_SKIP_KERNELS=q35_split_qgate"
run "skip ssmprep  " "ENGINE_SKIP_KERNELS=gd_ssm_prep"
run "skip gddelta  " "ENGINE_SKIP_KERNELS=gd_delta_rule_128f"
