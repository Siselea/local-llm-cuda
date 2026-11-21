#!/usr/bin/env bash
# Decode budget audit measurements (one GPU window, serial runs only, exclusive GPU!).
#
# Two kinds of run per case:
#   wall   : default config (graphs on), no profiling env -> per-step decode ms only
#   kernel : ENGINE_NO_GRAPH=1 + ENGINE_KPROF=1 (+ PROF_ACC) -> per-kernel GPU times;
#            the wall number of these runs is NOT comparable (graphs are off, kprof
#            records events per launch), only the kernel table is used.
#
#   MODEL_DIR=/path/to/models bash build/audit_decode.sh <group> [reps]
#   groups: intro | ctx9b | kv9b | coder | b35 | head | overhead | all
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
mkdir -p build/logs
PROBE_SRC=tools/audit_decode_probe.cpp
OVER_SRC=tools/audit_overheads.cpp
P=./build/decode_probe.exe
group="${1:-all}"
SRC="src/gguf.cpp src/st.cpp src/json_parse.cpp src/ir.cpp src/planner.cpp src/cuda.cpp \
src/cublas.cpp src/runner.cpp src/cpu_matvec.cpp src/qwen35.cpp src/config.cpp \
src/kernel_select.cpp src/tokenizer.cpp src/mtp.cpp src/sampler.cpp src/http_server.cpp src/model_scan.cpp"
PYBIN="${PY:-python}"
if [ ! -x "$P" ]; then
  echo "building decode_probe + bench_overheads"
  "$PYBIN" -m ziglang c++ -std=c++17 -O2 -Wall -Iinclude $SRC "$PROBE_SRC" -o "$P" || exit 1
  "$PYBIN" -m ziglang c++ -std=c++17 -O2 -Wall -Iinclude src/cuda.cpp src/config.cpp \
    src/cublas.cpp "$OVER_SRC" -o ./build/bench_overheads.exe || exit 1
fi
STEPS="${2:-5}"

M9="$MODEL_DIR/Qwen3.5-9B-Q5_K_M.gguf"
MC="$MODEL_DIR/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
M35="$MODEL_DIR/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"

probe() {  # <log> <extra env> <args...>
  local log="$1" envs="$2"; shift 2
  eval "$envs $P $*" >"$log" 2>&1
}

case "$group" in
intro|all)
  echo "### exclusivity snapshot"
  nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
  tasklist 2>/dev/null | grep -icE "test_gen|engine\.exe|pfdiag" || true
  ;;
esac

case "$group" in
ctx9b|all)
  for plen in 32 757 9100; do
    ctx=$((plen + 128))
    tag="ctx${plen}"
    echo "### 9B $tag"
    # wall (graphs on, no profiling)
    probe "build/logs/audit_9b_${tag}_wall.log" "PROBE_CTX=$ctx" \
      "$M9" --prompt-len "$plen" --steps "$STEPS" --ctx "$ctx"
    grep -E "prefill|step|decode median" "build/logs/audit_9b_${tag}_wall.log" | tail -8
    # kernel table (graphs off + kprof)
    probe "build/logs/audit_9b_${tag}_kern.log" \
      "ENGINE_NO_GRAPH=1 ENGINE_KPROF=1 ENGINE_PROF_ACC=1 PROBE_CTX=$ctx" \
      "$M9" --prompt-len "$plen" --steps 3 --ctx "$ctx"
    grep -E "^\[kprof\]|^\[ctx\]" "build/logs/audit_9b_${tag}_kern.log" | head -18
  done
  ;;
esac

case "$group" in
kv9b|all)
  echo "### 9B @9100: kv precision / split sweep (wall, graphs on)"
  for envs in "ENGINE_KV_TYPE=q8_0" "ENGINE_KV_TYPE=q4_0" "ENGINE_KV_SPLIT=8" \
              "ENGINE_KV_SPLIT=16" "ENGINE_KV_SPLIT=64"; do
    tag=$(echo "$envs" | tr '=/' '__')
    probe "build/logs/audit_9b_${tag}.log" "PROBE_CTX=9230 $envs" \
      "$M9" --prompt-len 9100 --steps "$STEPS" --ctx 9230
    printf '%-24s ' "$envs"
    grep -E "decode median" "build/logs/audit_9b_${tag}.log" | tail -1
    grep -E "^\[ctx\]" "build/logs/audit_9b_${tag}.log" | sed 's/^/    /'
  done
  ;;
esac

case "$group" in
coder|all)
  echo "### Coder decode budget (wall + acc + kernel table)"
  probe build/logs/audit_coder_wall.log "PROBE_CTX=1024" \
    "$MC" --prompt-len 4 --steps "$STEPS" --ctx 1024
  grep -E "prefill|decode median" build/logs/audit_coder_wall.log | tail -3
  probe build/logs/audit_coder_kern.log "ENGINE_NO_GRAPH=1 ENGINE_KPROF=1 ENGINE_PROF_ACC=1 PROBE_CTX=1024" \
    "$MC" --prompt-len 4 --steps 3 --ctx 1024
  grep -E "^\[kprof\]|^\[acc\] (wall|layers|per token)|CPU sub|GPU layer parts|^\[acc\]   head" \
    build/logs/audit_coder_kern.log | head -34
  ;;
esac

case "$group" in
b35|all)
  echo "### 35B decode budget (graphs on, prof_acc)"
  probe build/logs/audit_35b_wall.log "PROBE_CTX=1024" \
    "$M35" --prompt-len 4 --steps "$STEPS" --ctx 1024
  grep -E "prefill|decode median" build/logs/audit_35b_wall.log | tail -3
  probe build/logs/audit_35b_kern.log "ENGINE_KPROF=1 ENGINE_PROF_ACC=1 PROBE_CTX=1024" \
    "$M35" --prompt-len 4 --steps 3 --ctx 1024
  grep -E "^\[kprof\]|^\[acc\]   layers|^\[acc\] per token|CPU sub" build/logs/audit_35b_kern.log | head -26
  ;;
esac

case "$group" in
head|all)
  echo "### head bandwidth (kprof, graphs off so lm_head is a bare launch)"
  for spec in "$M9 9b" "$M35 35b" "$MC coder"; do
    set -- $spec
    probe "build/logs/audit_head_$2.log" "ENGINE_NO_GRAPH=1 ENGINE_KPROF=1 PROBE_CTX=256" \
      "$1" --prompt-len 16 --steps 3 --ctx 256
    echo "-- $2"
    grep -E "^\[kprof\].*(gemv_q|lm|gemv_f32|gemv_q6|x4|dp4a)" "build/logs/audit_head_$2.log" | head -6
  done
  ;;
esac

case "$group" in
overhead|all)
  echo "### per-token fixed overhead microbench"
  ./build/bench_overheads.exe >build/logs/audit_overheads.log 2>&1
  cat build/logs/audit_overheads.log
  ;;
esac

echo
echo "done; logs in build/logs/audit_*.log"
