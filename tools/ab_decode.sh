#!/usr/bin/env bash
# Interleaved A/B decode benchmark: alternates a reference run and a variant run
# so VM drift cancels.  VARIANT holds extra env assignments, e.g.
#   VARIANT="ENGINE_SKIP_KERNELS=rmsnorm" REPS=5 bash tools/ab_decode.sh
# NOTE: runs must use bash prefix assignments (not `env`): the Windows sandbox
# does not deliver a program's output through a pipe when it is started via env.
set -uo pipefail
cd "$(dirname "$0")/.."
M="${M:-<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf}"
IDS="${IDS:-760}"
N="${N:-9}"
REPS="${REPS:-5}"
BASE="${BASE:-}"
VARIANT="${VARIANT:-}"
BIN="${BIN:-./build/test_gen.exe}"

one() {
  local extra="$1" out
  out=$(eval "ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 $extra \"$BIN\" \"$M\" \"$IDS\" \"$N\" 2>&1" |
    grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2)
  printf '%s\n' "${out:-ERR}"
}

printf 'variant: %s\n' "${VARIANT:-<none>}"
printf '%-4s %9s %9s %8s\n' rep base variant delta
ab=(); bb=()
for r in $(seq 1 "$REPS"); do
  a=$(one "$BASE")
  b=$(one "$BASE $VARIANT")
  ab+=("$a")
  bb+=("$b")
  d=$(awk -v a="$a" -v b="$b" 'BEGIN { if (a+0 > 0 && b+0 > 0) printf "%+.3f", b - a; else printf "n/a" }')
  printf '%-4s %9s %9s %8s\n' "$r" "$a" "$b" "$d"
done
amed=$(printf '%s\n' "${ab[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
bmed=$(printf '%s\n' "${bb[@]}" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
printf 'median base=%s variant=%s delta=%s\n' "$amed" "$bmed" \
  "$(awk -v a="$amed" -v b="$bmed" 'BEGIN { printf "%+.3f ms (%.2f%%)", b - a, 100.0 * (b - a) / a }')"
