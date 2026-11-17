#!/usr/bin/env bash
# Final before/after benchmark: runs both binaries back-to-back with no other GPU
# work in between so thermal/clock state is comparable.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./build/test_gen.baseline.exe bash tools/bench_perf.sh baseline-clean
BIN=./build/test_gen.exe          bash tools/bench_perf.sh after-clean
echo "== combined =="
cat build/perf-baseline-clean.txt build/perf-after-clean.txt
