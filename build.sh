#!/usr/bin/env bash
# Build the engine with the pip-shipped Zig toolchain (no admin, no MSVC/CUDA toolkit).
set -euo pipefail
cd "$(dirname "$0")"

# Python with the pip-installed `ziglang` package (see README "依赖").
#   PY=/path/to/python bash build.sh
PY="${PY:-${PYTHON:-python}}"
CXX=("$PY" -m ziglang c++)

mkdir -p build
SRC="src/gguf.cpp src/st.cpp src/json_parse.cpp src/ir.cpp src/planner.cpp src/cuda.cpp src/cublas.cpp src/runner.cpp src/cpu_matvec.cpp src/qwen35.cpp src/config.cpp src/kernel_select.cpp src/tokenizer.cpp src/mtp.cpp src/sampler.cpp src/http_server.cpp src/model_scan.cpp src/serve.cpp"

build() {
  "${CXX[@]}" -std=c++17 -O2 -Wall -Iinclude $SRC "$1" -o "build/$2"
  echo "built build/$2"
}

# launcher.exe: standalone SUBSYSTEM:WINDOWS GUI (no console window).  It only
# talks to engine.exe through `models --tsv` + the HTTP API, so it does not need
# the engine sources / CUDA paths at all.
build_gui() {
  "${CXX[@]}" -std=c++17 -O2 -Wall -Iinclude -Wl,--subsystem,windows "$1" -o "build/$2" \
    -lwinhttp -ldwmapi -luxtheme -lgdi32 -luser32 -lshell32
  echo "built build/$2"
}

build src/main.cpp        engine.exe
build_gui src/launcher.cpp launcher.exe
build tests/test_cuda.cpp  test_cuda.exe
build tests/test_gemm.cpp  test_gemm.exe
build tests/test_layer.cpp test_layer.exe
build tests/test_quant.cpp test_quant.exe
build tests/test_gemv.cpp  test_gemv.exe
build tests/test_kv.cpp    test_kv.exe
build tests/test_model.cpp test_model.exe
build tests/test_gguf.cpp  test_gguf.exe
build tests/test_decode.cpp test_decode.exe
build tests/test_gen.cpp test_gen.exe
build tests/test_json_utf8.cpp test_json_utf8.exe
build tests/test_json_depth.cpp test_json_depth.exe
build tests/test_mla_kv.cpp test_mla_kv.exe
build tests/test_cpu_quant.cpp test_cpu_quant.exe
build tests/test_cpu_moe.cpp test_cpu_moe.exe
build tests/test_q35_conv.cpp test_q35_conv.exe
build tests/test_kernel_bounds.cpp test_kernel_bounds.exe
build tests/test_mla_cache_cmp.cpp test_mla_cmp.exe
build tests/test_matrix.cpp test_matrix.exe
build tests/test_tokenizer.cpp test_tokenizer.exe
build tests/test_sampler.cpp test_sampler.exe
build tests/test_serve_guard.cpp test_serve_guard.exe
