#!/usr/bin/env bash
# MTP greedy-vs-plain divergence hunt (qwen35 hybrid, especially 35B).
#
#   bash tools/mtp_divergence.sh [model.gguf] [n_tokens] [ids]
#
# Runs plain greedy, then speculative greedy with k=1 and k=2 (and optionally a
# numerics-perturbation sweep via $ENVS_LIST), and reports for every variant the
# first token index where it differs from plain greedy, plus the token ids around it.
#
# Why this script exists: MTP (speculative decoding) must be "lossless": the emitted
# stream is defined as the target model's own arg-max at every position, so it can only
# diverge from plain greedy if either
#   (a) the batched verify forward computes slightly different numerics than the
#       single-token decode forward (f16 prefill GEMM vs int8-dp4a GEMV, f32 vs
#       quantized head, n>1 vs n=1 attention, separate vs fused SSM path), or
#   (b) the conv/rec state rollback after a rejected draft is not faithful.
# Running the same prompt under numerics knobs tells (a) apart from (b): a numerics
# cause moves the divergence point, a state/rollback bug tends to keep it.
set -uo pipefail
cd "$(dirname "$0")/.."

M="${1:-${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf}"
N="${2:-300}"
IDS="${3:-760,6511,314,9338,369}"
OUT="${OUT:-build/mtp-div}"
mkdir -p "$OUT"
TAG="$(basename "$M" .gguf)"
PY="${PY:-python}"

extract() { grep -oE '^gen:.*' | head -1 | cut -d' ' -f2-; }

# run_plain <tag> <extra env>
run_plain() {
  local tag="$1"; shift
  local out="$OUT/${tag}.log"
  env ENGINE_GPU_EXPERTS=16 "$@" ./build/test_gen.exe "$M" --ids "$IDS" --n "$N" \
      --ctx 512 2>"$out.err" | extract >"$OUT/${tag}.ids"
  printf '%s' "$(wc -w <"$OUT/${tag}.ids")"
}

# run_spec <tag> <k> <extra env>
run_spec() {
  local tag="$1" k="$2"; shift 2
  local out="$OUT/${tag}.log"
  env ENGINE_GPU_EXPERTS=16 "$@" ./build/test_gen.exe "$M" --ids "$IDS" --n "$N" \
      --ctx 512 --mtp --mtp-k "$k" 2>"$out.err" | extract >"$OUT/${tag}.ids"
  grep -hE '^\[mtp\]' "$out.err" | tail -1 >>"$OUT/${tag}.ids.stats"
  printf '%s' "$(wc -w <"$OUT/${tag}.ids")"
}

compare() {  # compare <base.ids> <test.ids> <label> <stats-file>
  local base="$1" test="$2" label="$3" stats="${4:-/dev/null}"
  if [ ! -s "$test" ]; then printf '%-34s NO OUTPUT\n' "$label"; return; fi
  "$PY" - "$base" "$test" "$label" "$stats" <<'PYEOF'
import sys
base = open(sys.argv[1]).read().split()
test = open(sys.argv[2]).read().split()
lab = sys.argv[3]
stats = ""
try:
    stats = open(sys.argv[4]).read().strip()
except Exception:
    pass
n = min(len(base), len(test))
first = next((i for i in range(n) if base[i] != test[i]), None)
extra = ""
if len(base) != len(test):
    extra = f"  len {len(base)} vs {len(test)}"
if first is None:
    print(f"{lab:<34} IDENTICAL ({n} tokens){extra}  {stats}")
else:
    lo, hi = max(0, first - 2), min(n, first + 3)
    ctx = " ".join(f"{j}:{base[j]}/{test[j]}" for j in range(lo, hi))
    print(f"{lab:<34} DIFFERS at token {first} "
          f"(plain {base[first] if first < len(base) else '?'} vs "
          f"spec {test[first] if first < len(test) else '?'}){extra}")
    print(f"{'':<34}   context plain/spec: {ctx}  {stats}")
PYEOF
}

echo "== $TAG  ids=$IDS  n=$N  (GPU=ENGINE_GPU_EXPERTS=16) =="
echo
echo "-- baseline: plain greedy, default numerics --"
run_plain plain >/dev/null
echo "   plain tokens: $(wc -w <"$OUT/plain.ids")"

echo
echo "-- spec variants, default numerics --"
KS="${KS:-2}"
for k in $KS; do run_spec "spec_k$k" "$k" >/dev/null; done
for k in $KS; do compare "$OUT/plain.ids" "$OUT/spec_k$k.ids" "spec k=$k (default)"; done

# ---------------------------------------------------------------------------
# Numerics-perturbation sweep.  Each knob changes the verify-vs-decode numeric
# relation; if the divergence point moves, the cause is numeric, not rollback.
# ---------------------------------------------------------------------------
# KNOBS=1 enables the sweep (each variant reloads the model: slow on 35B).
declare -a VARIANTS=(
  "nof16:ENGINE_PREFILL_F16=0"
  "nogemv:ENGINE_NO_GEMV=1"
  "nocpuexp:ENGINE_GPU_EXPERTS=0"
  "nodp4a:ENGINE_NO_DP4A_GEMV=1"
)
for v in "${VARIANTS[@]}"; do
  [ "${KNOBS:-0}" = "1" ] || break
  name="${v%%:*}"; envs="${v#*:}"
  echo
  echo "-- numerics knob: $name ($envs) --"
  # shellcheck disable=SC2086
  run_plain "plain_$name" $envs >/dev/null
  # shellcheck disable=SC2086
  run_spec "spec_k2_$name" 2 $envs >/dev/null
  compare "$OUT/plain_$name.ids" "$OUT/spec_k2_$name.ids" "spec k=2 [$name]" "$OUT/spec_k2_$name.ids.stats"
done

echo
echo "logs: $OUT/"
