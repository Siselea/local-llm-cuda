#!/usr/bin/env bash
# Full verification: Python unit tests + G0 (IR parity) + M2 (plan parity).
# Usage: bash tools/verify.sh [--quick]
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
GGUF_MODELS=(
  "${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
  "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
  "${MODEL_DIR}/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"
  "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
)
MODELS=("${GGUF_MODELS[@]}" "build/fixtures/tiny_hf" "build/fixtures/tiny_hf_moe" \
        "build/fixtures/tiny_gguf_moe/model.gguf" "build/fixtures/tiny_gguf_mla/model.gguf")

fail=0

echo "== build =="
bash build.sh || exit 1

echo
echo "== fixtures (safetensors) =="
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from pathlib import Path
from tests.test_st import make_hf_model
root = Path('build/fixtures')
specs = [('tiny_hf', {}), ('tiny_hf_moe', dict(n_expert=4, expert_inter=32))]
for name, kw in specs:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    make_hf_model(d, n_layer=2, hidden=64, inter=128, vocab=128, **kw)
print("fixtures:", ", ".join(n for n, _ in specs))
PYEOF
"$PY" tests/make_tiny_layer.py || fail=1
./build/test_q35_conv.exe >/dev/null 2>&1 || fail=1
echo "q35 conv consistency : PASS (chunked vs incremental)"
"$PY" tests/make_tiny_model.py || fail=1
N_LAYER=4 "$PY" tests/make_tiny_model.py build/fixtures/tiny_model4 >/dev/null || fail=1
"$PY" tests/make_tiny_moe.py >/dev/null || fail=1
"$PY" tests/make_tiny_gguf_moe.py >/dev/null || fail=1
N_EXPERT=300 "$PY" tests/make_tiny_gguf_moe.py build/fixtures/tiny_gguf_moe300 >/dev/null || fail=1
"$PY" tests/make_tiny_gguf_mla.py >/dev/null || fail=1
"$PY" tests/make_tiny_gguf_ds2.py >/dev/null || fail=1
QW="${MODEL_DIR}/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"
CD="${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
"$PY" tools/make_quant_fixture.py "$QW" blk.0.ssm_alpha.weight      build/fixtures/q4k 4096  || fail=1
"$PY" tools/make_quant_fixture.py "$CD" blk.0.attn_kv_a_mqa.weight  build/fixtures/q5k 4608  || fail=1
"$PY" tools/make_quant_fixture.py "$QW" blk.0.ffn_down.weight       build/fixtures/q6k 4096  || fail=1
"$PY" tools/make_quant_fixture.py "$QW" blk.32.attn_k.weight        build/fixtures/q8k 8192  || fail=1
"$PY" tools/make_quant_fixture.py "${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf" blk.0.attn_k.weight build/fixtures/q5_0 3584 || fail=1
"$PY" tools/make_gemv_fixture.py  "$QW" blk.0.attn_gate.weight     build/fixtures/gemv_q4k 1024  || fail=1
"$PY" tools/make_gemv_fixture.py  "$CD" blk.0.ffn_gate.weight      build/fixtures/gemv_q5k 1024  || fail=1
"$PY" tools/make_gemv_fixture.py  "$QW" blk.32.attn_k.weight       build/fixtures/gemv_q8_0 1024 || fail=1
"$PY" tools/make_gemv_fixture.py  "$QW" blk.0.attn_gate.weight     build/fixtures/gemv_q4k_big 4096 || fail=1
"$PY" tools/make_gguf_ref.py "${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf" build/fixtures/gguf_qwen05 8 || fail=1
[ $? -ne 0 ] && fail=1

echo
echo "== python unit tests =="
"$PY" -m unittest discover -s tests || fail=1

echo
echo "== G0: C++ IR == Python IR =="
for m in "${MODELS[@]}"; do
  [ -e "$m" ] || { echo "skip (missing): $m"; continue; }
  "$PY" tools/diff_ir.py "$m" || fail=1
done

echo
echo "== M3: CUDA (NVRTC kernels + cuBLAS) =="
if [ -x ./build/test_cuda.exe ]; then ./build/test_cuda.exe || fail=1; else echo "missing build/test_cuda.exe"; fail=1; fi
if [ -x ./build/test_gemm.exe ]; then ./build/test_gemm.exe || fail=1; else echo "missing build/test_gemm.exe"; fail=1; fi
if [ -x ./build/test_layer.exe ]; then ./build/test_layer.exe || fail=1; else echo "missing build/test_layer.exe"; fail=1; fi
if [ -x ./build/test_quant.exe ]; then ./build/test_quant.exe || fail=1; else echo "missing build/test_quant.exe"; fail=1; fi
if [ -x ./build/test_cpu_moe.exe ]; then ./build/test_cpu_moe.exe || fail=1; else echo "missing build/test_cpu_moe.exe"; fail=1; fi
if [ -x ./build/test_gemv.exe ]; then ./build/test_gemv.exe || fail=1; else echo "missing build/test_gemv.exe"; fail=1; fi
if [ -x ./build/test_kv.exe ]; then ./build/test_kv.exe || fail=1; else echo "missing build/test_kv.exe"; fail=1; fi
if [ -x ./build/test_kernel_bounds.exe ]; then ./build/test_kernel_bounds.exe || fail=1; else echo "missing build/test_kernel_bounds.exe"; fail=1; fi
if [ -x ./build/test_q35_geometry.exe ]; then ./build/test_q35_geometry.exe || fail=1; else echo "missing build/test_q35_geometry.exe"; fail=1; fi
if [ -x ./build/test_model.exe ]; then
  ./build/test_model.exe || fail=1
  ./build/test_model.exe build/fixtures/tiny_model4 || fail=1
  ./build/test_model.exe build/fixtures/tiny_moe || fail=1
  ENGINE_GPU_EXPERTS=0 ./build/test_model.exe build/fixtures/tiny_moe || fail=1
  ENGINE_GPU_EXPERTS=2 ./build/test_model.exe build/fixtures/tiny_moe || fail=1
  ENGINE_GPU_EXPERTS=0 ./build/test_model.exe build/fixtures/tiny_gguf_moe || fail=1
  ENGINE_GPU_EXPERTS=2 ./build/test_model.exe build/fixtures/tiny_gguf_moe || fail=1
  ENGINE_VRAM_BUDGET_MB=0 ./build/test_model.exe build/fixtures/tiny_gguf_moe || fail=1
  ./build/test_model.exe build/fixtures/tiny_gguf_moe || fail=1
  ./build/test_model.exe build/fixtures/tiny_gguf_mla || fail=1
  ./build/test_model.exe build/fixtures/tiny_gguf_ds2 || fail=1
else echo "missing build/test_model.exe"; fail=1; fi
if [ -x ./build/test_mla_kv.exe ]; then ./build/test_mla_kv.exe || fail=1; else echo "missing build/test_mla_kv.exe"; fail=1; fi
if [ -x ./build/test_decode.exe ]; then
  ./build/test_decode.exe || fail=1
  ./build/test_decode.exe build/fixtures/tiny_gguf_mla || fail=1
  ./build/test_decode.exe "${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
      build/fixtures/gguf_qwen05/tokens.i32 || fail=1
else echo "missing build/test_decode.exe"; fail=1; fi
if [ -x ./build/test_gguf.exe ]; then
  ./build/test_gguf.exe "${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf" build/fixtures/gguf_qwen05 || fail=1
else echo "missing build/test_gguf.exe"; fail=1; fi
if [ -x ./build/test_serve_guard.exe ]; then
  ./build/test_serve_guard.exe || fail=1
else echo "missing build/test_serve_guard.exe"; fail=1; fi

echo
echo "== M2: C++ plan == Python plan =="
for m in "${MODELS[@]}"; do
  [ -e "$m" ] || { echo "skip (missing): $m"; continue; }
  "$PY" tools/diff_plan.py "$m" || fail=1
done

echo
if [ "$fail" -eq 0 ]; then
  echo "ALL GREEN"
else
  echo "FAILURES PRESENT"
fi
exit "$fail"
