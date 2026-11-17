#!/usr/bin/env bash
# Final four-model table: interleaved pre (before this sprint) vs new, median of N runs.
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${REPS:-5}"
M05=${MODEL_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf
MC=${MODEL_DIR}/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf
M9=${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf
M35=${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }

one() {  # tag model ids n
  local tag="$1" m="$2" ids="$3" n="$4" e=""
  [ "$tag" = "Coder" ] || [ "$tag" = "35B" ] || e="ENGINE_SKIP_WARMUP=0"
  local ab=() bb=()
  for r in $(seq 1 "$REPS"); do
    ab+=("$(eval "ENGINE_TOPK=1 $e ./build/test_gen_pre.exe \"$m\" \"$ids\" $n 2>&1" | dec)")
    bb+=("$(eval "ENGINE_TOPK=1 $e ./build/test_gen.exe \"$m\" \"$ids\" $n 2>&1" | dec)")
  done
  local am bm
  am=$(med "${ab[@]}"); bm=$(med "${bb[@]}")
  awk -v t="$tag" -v a="$am" -v b="$bm" 'BEGIN {printf "%-6s pre %7.2f ms (%5.1f tok/s)  new %7.2f ms (%5.1f tok/s)  delta %+6.2f ms (%+.1f%%)\n", t, a, 1000/a, b, 1000/b, b-a, 100*(b-a)/a}'
}

one "0.5B"  "$M05" "1000,1001" 33
one "Coder" "$MC"  "549,6077,280,7239" 9
one "9B"    "$M9"  "760" 9
one "35B"   "$M35" "760" 9
