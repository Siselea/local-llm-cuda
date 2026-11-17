#!/usr/bin/env bash
# CPU-only interleaved A/B of the int8 host-GEMV kernels using the isolated layer
# benchmark (24 rotating copies of the 35B CPU-layer geometry -> DRAM streaming, same
# computation as cpu::moe_forward_mt inside the engine).  No GPU needed, so it can run
# while the GPU is busy.
#   A: ENGINE_CPU_INT8=0 (fp32 kernels, the current default)
#   B: ENGINE_CPU_INT8=1 (int8 for Q6_K rows only)
# Reports min and median (min is the more robust statistic on a shared VM).
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${REPS:-9}"
ITERS="${ITERS:-80}"
THR="${THR:-9}"
a=(); b=(); ba=(); bb=()
for r in $(seq 1 "$REPS"); do
  oa=$(ENGINE_CPU_INT8=0 ENGINE_THREADS=$THR ./build/bench_int8.exe layer 24 "$ITERS" 2>&1)
  ob=$(ENGINE_CPU_INT8=1 ENGINE_THREADS=$THR ./build/bench_int8.exe layer 24 "$ITERS" 2>&1)
  a+=("$(printf '%s\n' "$oa" | grep -oE '[0-9.]+ ms/layer' | grep -oE '[0-9.]+')")
  b+=("$(printf '%s\n' "$ob" | grep -oE '[0-9.]+ ms/layer' | grep -oE '[0-9.]+')")
  ba+=("$(printf '%s\n' "$oa" | grep -oE '[0-9.]+ GB/s' | grep -oE '[0-9.]+')")
  bb+=("$(printf '%s\n' "$ob" | grep -oE '[0-9.]+ GB/s' | grep -oE '[0-9.]+')")
done
min() { printf '%s\n' "$@" | sort -n | head -1; }
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
echo "layer ms   off: min $(min "${a[@]}")  med $(med "${a[@]}")   (${a[*]})"
echo "layer ms   on : min $(min "${b[@]}")  med $(med "${b[@]}")   (${b[*]})"
echo "bandwidth  off: min-bw-of-max $(printf '%s\n' "${ba[@]}" | sort -n | tail -1) GB/s | on: $(printf '%s\n' "${bb[@]}" | sort -n | tail -1) GB/s"
awk -v a="$(min "${a[@]}")" -v b="$(min "${b[@]}")" \
  'BEGIN {printf "min-based layer time: %+.2f%% (%s -> %s ms)\n", 100*(b-a)/a, a, b}'
awk -v a="$(med "${a[@]}")" -v b="$(med "${b[@]}")" \
  'BEGIN {printf "median-based layer time: %+.2f%% (%s -> %s ms)\n", 100*(b-a)/a, a, b}'
echo DONE-CPU-AB
