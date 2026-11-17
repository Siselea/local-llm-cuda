#!/usr/bin/env bash
# Attribute the single-pass numeric drift of the new decode paths to each feature
# by dumping the last-token logits for one forward and comparing with numpy.
set -uo pipefail
cd "$(dirname "$0")/.."
M="${M:-${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf}"
BIN=./build/test_gen.exe
one() {  # $1 = out file, rest = env assignments
  local out="$1"; shift
  env ENGINE_LOGITS="build/$out" ENGINE_SKIP_WARMUP=0 "$@" "$BIN" "$M" 760 1 >/dev/null 2>&1
}
one l_all_old.f32 ENGINE_NO_SSM_FUSE=1 ENGINE_NO_GD_DELTA_128=1 ENGINE_NO_QUANT_REUSE=1
one l_fuse.f32    ENGINE_NO_GD_DELTA_128=1 ENGINE_NO_QUANT_REUSE=1
one l_delta.f32   ENGINE_NO_SSM_FUSE=1 ENGINE_NO_QUANT_REUSE=1
one l_reuse.f32   ENGINE_NO_SSM_FUSE=1 ENGINE_NO_GD_DELTA_128=1
one l_all_new.f32
"python" - <<'PYEOF'
import numpy as np
old = np.fromfile('build/l_all_old.f32', dtype=np.float32).astype(np.float64)
for tag, name in [('l_fuse', 'ssm_fuse only'), ('l_delta', 'gd_delta_128 only'),
                  ('l_reuse', 'quant_reuse only'), ('l_all_new', 'all new')]:
    a = np.fromfile(f'build/{tag}.f32', dtype=np.float32).astype(np.float64)
    d = a - old
    print(f'{name:20s} rel_l2={np.sqrt((d*d).sum()/(old*old).sum()):.3e} '
          f'max|d|={np.abs(d).max():.3e} argmax={a.argmax()}/{old.argmax()}')
PYEOF
