#!/usr/bin/env bash
# Interleaved A/B of the CPU int8 host-GEMV kernels on the 35B: SAME binary, only the
# ENGINE_CPU_INT8 switch differs, so the comparison isolates the int8 path exactly.
# Metrics, in order of trust under VM noise:
#   1) [cpuexec] batched host matvec avg ms  (engine-internal, CPU-side numerator)
#   2) [moeph] phase A/B split (A = gate+up Q4_K, B = down Q6_K -- only B uses int8)
#   3) end-to-end decode ms/token (interleaved, median of REPS)
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${BIN:-./build/test_gen.exe}"
M=${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
N="${N:-17}"
REPS="${REPS:-6}"
dec() { grep -oE 'decode [0-9.]+' | head -1 | cut -d' ' -f2; }
cx()  { grep -oE '\[cpuexec\] batched host matvec avg [0-9.]+' | grep -oE '[0-9.]+$' | tail -1; }
mb()  { grep -oE '\[moeph\][^ ]* .*' | sed -E 's/.*A ([0-9.]+) B ([0-9.]+) merge.*/\1 \2/' | tail -1; }
plat(){ grep -oE 'post-load used=[0-9]+ MiB free=[0-9]+ MiB \(gpu_moe_layers=[0-9]+\)' | tail -1; }
a=(); b=(); ca=(); cb=(); ta=(); tb=()
for r in $(seq 1 "$REPS"); do
  outa=$(ENGINE_CPU_INT8=0 ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_DBG_CPUEXEC=1 ENGINE_DBG_MOE_PHASES=1 \
  "$BIN" "$M" "760" "$N" 2>&1)
  outb=$(ENGINE_CPU_INT8=1 ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_DBG_CPUEXEC=1 ENGINE_DBG_MOE_PHASES=1 \
  "$BIN" "$M" "760" "$N" 2>&1)
  av=$(printf '%s\n' "$outa" | dec); bv=$(printf '%s\n' "$outb" | dec)
  a+=("$av"); b+=("$bv")
  ca+=("$(printf '%s\n' "$outa" | cx)"); cb+=("$(printf '%s\n' "$outb" | cx)")
  ta+=("$(printf '%s\n' "$outa" | mb)");  tb+=("$(printf '%s\n' "$outb" | mb)")
  printf 'rep%d off: dec=%.2f matvec=%s moephA/B=%s | on: dec=%.2f matvec=%s moephA/B=%s | topk %s / %s\n' \
    "$r" "$av" "$(printf '%s\n' "$outa" | cx)" "$(printf '%s\n' "$outa" | mb)" \
    "$bv" "$(printf '%s\n' "$outb" | cx)" "$(printf '%s\n' "$outb" | mb)" \
    "$(printf '%s\n' "$outa" | grep -oE 'topk: [0-9]+' | head -1)" \
    "$(printf '%s\n' "$outb" | grep -oE 'topk: [0-9]+' | head -1)"
done
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
echo "----- medians (cpu_int8=0 vs 1) -----"
echo "decode  : $(med "${a[@]}") vs $(med "${b[@]}") ms"
echo "matvec  : $(med "${ca[@]}") vs $(med "${cb[@]}") ms/layer (cpuexec)"
echo "placement: $(printf '%s\n' "$outa" | plat) | $(printf '%s\n' "$outb" | plat)"
awk -v a="$(med "${ca[@]}")" -v b="$(med "${cb[@]}")" \
  'BEGIN {if (a>0) printf "matvec delta %+.2f%%\n", 100*(b-a)/a}'
awk -v a="$(med "${a[@]}")" -v b="$(med "${b[@]}")" \
  'BEGIN {printf "decode delta %+.2f ms (%+.1f%%), tok/s %.2f -> %.2f\n", b-a, 100*(b-a)/a, 1000/a, 1000/b}'
echo DONE-AB-INT8
