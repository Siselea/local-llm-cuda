#!/usr/bin/env bash
# Build the Rust-vs-C++ CPU kernel probe (rust-probe/).
#
# This machine has NO MSVC link.exe and NO MinGW gcc, so a plain `rustc foo.rs`
# cannot link (verified: link.exe not found / missing CRT).  Two workarounds:
#   * rust is installed with rustup into a user directory (no admin):
#       RUSTUP_HOME=$HOME/.rustup CARGO_HOME=$HOME/.cargo
#       RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
#       rustup-init.exe --yes --profile minimal --default-host x86_64-pc-windows-msvc
#   * the Rust side is built as a *staticlib* (no linker needed) and linked into
#     the C++ harness with the repo's zig toolchain, which bundles mingw-w64 + lld.
#
# usage: bash rust-probe/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
RUSTC="${RUSTC:-rustc}"
: "${MODEL_DIR:?set MODEL_DIR to the directory containing the GGUF files}"
OUT=build/rustprobe
mkdir -p "$OUT"

# 0) real weights: one CPU layer (experts 0..7) of the 35B, from the actual GGUF
if [ ! -f "$OUT/gate8.bin" ]; then
  python rust-probe/extract_weights.py \
    "${MODEL:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}" \
    16 "$OUT"
fi

# 1) Rust kernels -> staticlib (no link step)
"$RUSTC" --target x86_64-pc-windows-gnu --crate-type staticlib \
  -C opt-level=3 -C target-cpu=native -C panic=abort -C codegen-units=1 \
  rust-probe/rust_kernels.rs -o "$OUT/librust_kernels.a"

# 2) C++ harness (production kernels via #include) + Rust staticlib, zig links both
"$PY" -m ziglang c++ -std=c++17 -O2 -Iinclude -w \
  rust-probe/cpp_bench.cpp src/config.cpp "$OUT/librust_kernels.a" \
  -o "$OUT/ab_bench.exe"

echo "built $OUT/ab_bench.exe"
echo
echo "run:"
echo "  $OUT/ab_bench.exe verify            # bit-exactness Rust vs C++"
echo "  $OUT/ab_bench.exe kernel 80         # single-thread kernel A/B (min of 80)"
echo "  RP_THREADS=12 RP_ITERS=20 $OUT/ab_bench.exe ab 6   # 12-thread layer A/B"
