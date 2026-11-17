#!/usr/bin/env bash
# run_utilization_suite.sh -- measure GPU/CPU utilisation and per-layer bubbles for the four
# local models with tools/measure_utilization.exe + the engine's own profiling env vars.
#
# The engine binary is a *snapshot* (tools/util_bin/test_gen_snap.exe) so that a concurrent
# rebuild of build/test_gen.exe cannot invalidate the measurement mid-suite; the sha256 of
# the snapshot is written to tools/util_results/MANIFEST.md.
#
# Every run:
#   * waits (up to WAIT_SEC) for a quiet GPU (no test_gen*/engine*/llama* compute app,
#     VRAM below IDLE_MEM_MB) and a quiet CPU,
#   * samples NVML at 10 ms + the engine's CPU time + timestamped stderr,
#   * refuses to be trusted if the validity block shows foreign GPU work.
#
# Modes (all with ENGINE_DBG_FWD=1 so the analyser gets per-token boundaries):
#   graph    default execution (CUDA graph for dense models, per-layer graph for GPU MoE
#            layers), n=65 / 257 tokens -- true wall + true NVML utilisation
#   probe    --no-graph + ENGINE_KPROF=1 + ENGINE_DBG_LAYERMS=1 + ENGINE_DBG_CPUEXEC=1
#            + ENGINE_PIPE_PROF=1, n=17 -- per-layer wall, per-kernel GPU time, host matvec
#   prefix1  like probe with --layers 1   -> GPU time of layer 0 + run-level head
#   prefixN  like probe with --layers 15 (35B) / 16 (9B/0.5B) -> per-class GPU differencing
#   prof     --no-graph + ENGINE_PROFILE=1, n=5 -- MoE phase breakdown ([moe-prof])
#
# Usage:
#   bash tools/run_utilization_suite.sh                # all four models, all modes
#   MODELS="qwen3.5-9B" bash tools/run_utilization_suite.sh
#   MODES="graph probe" bash tools/run_utilization_suite.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="tools/util_results"
MON="./tools/util_bin/measure_utilization.exe"
BIN="./tools/util_bin/test_gen_snap.exe"
WAIT_SEC="${WAIT_SEC:-900}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1200}"
INTERVAL_MS="${INTERVAL_MS:-10}"
RETRIES="${RETRIES:-2}"
MODELS="${MODELS:-qwen2.5-0.5b DeepSeek-Coder-V2-Lite qwen3.5-9B qwen3.6-35B-A3B}"
MODES="${MODES:-graph probe prefix1 prefixN prof}"

M_QWEN="<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf"
M_CODER="<MODEL_DIR>/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
M_9B="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf"
M_35B="<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"

model_path() {
  case "$1" in
    qwen2.5-0.5b) echo "$M_QWEN" ;;
    DeepSeek-Coder-V2-Lite) echo "$M_CODER" ;;
    qwen3.5-9B) echo "$M_9B" ;;
    qwen3.6-35B-A3B) echo "$M_35B" ;;
    *) echo "" ;;
  esac
}
model_ids() {
  case "$1" in
    qwen2.5-0.5b) echo "1000,1001" ;;
    DeepSeek-Coder-V2-Lite) echo "549,6077,280,7239" ;;
    *) echo "760" ;;
  esac
}
graph_n() { case "$1" in qwen2.5-0.5b) echo 1025 ;; DeepSeek-Coder-V2-Lite) echo 129 ;; *) echo 257 ;; esac; }
pin_args() { # model mode -> extra args that avoid the adaptive-placement path
  case "$1:$2" in
    DeepSeek-Coder-V2-Lite:probe|DeepSeek-Coder-V2-Lite:prefix1|DeepSeek-Coder-V2-Lite:prefixN|DeepSeek-Coder-V2-Lite:prof)
      echo "--no-vram-check --gpu-layers 20" ;;
    qwen3.6-35B-A3B:probe|qwen3.6-35B-A3B:prefix1|qwen3.6-35B-A3B:prefixN|qwen3.6-35B-A3B:prof)
      echo "--no-vram-check --gpu-layers 15" ;;
    *) echo "" ;;
  esac
}
prefix_n() {
  case "$1" in
    35B|qwen3.6-35B-A3B) echo 15 ;;
    qwen3.5-9B) echo 16 ;;
    qwen2.5-0.5b) echo 12 ;;
    DeepSeek-Coder-V2-Lite) echo 17 ;;   # 1 dense + 16 GPU MoE layers (auto placement)
    *) echo 15 ;;
  esac
}

mkdir -p "$ROOT"
mkdir -p tools/util_bin
if [ ! -f "$BIN" ]; then
  echo "ERROR: $BIN missing; snapshot build/test_gen.exe into tools/util_bin/test_gen_snap.exe first" >&2
  exit 2
fi
if [ ! -f "$MON" ]; then
  echo "building $MON"
  PY="${PY:-python}"
  "$PY" -m ziglang c++ -std=c++17 -O2 tools/measure_utilization.cpp -o "$MON" || exit 2
fi

HASH_BEFORE=$(sha256sum "$BIN" 2>/dev/null | cut -d' ' -f1)
MANIFEST="$ROOT/MANIFEST.md"
{
  echo "# measurement manifest"
  echo
  echo "- date: $(date -Iseconds)"
  echo "- engine binary: \`$BIN\` sha256=$HASH_BEFORE mtime=$(date -r "$BIN" -Iseconds 2>/dev/null)"
  echo "- monitor: \`$MON\` sha256=$(sha256sum "$MON" | cut -d' ' -f1)"
  echo "- sampling: NVML every ${INTERVAL_MS} ms, idle wait ${WAIT_SEC}s, idle VRAM < ${IDLE_MEM_MB} MiB"
  echo
} >> "$MANIFEST"

json_get() { # json_get <file> <key>
  grep -o "\"$1\": *[^,}]*" "$2" 2>/dev/null | head -1 | sed -E 's/.*: *//' | tr -d '"'
}

run_one() { # model mode
  local model="$1" mode="$2"
  local path ids n extra_env extra_args out_dir prefix attempt timeout_ms
  path=$(model_path "$model"); ids=$(model_ids "$model")
  case "$mode" in
    graph)   n=$(graph_n "$model"); extra_env="ENGINE_TOPK=1 ENGINE_DBG_FWD=1"; extra_args="" ;;

    probe)   n=65
             extra_env="ENGINE_TOPK=1 ENGINE_DBG_FWD=1 ENGINE_DBG_LAYERMS=1 ENGINE_KPROF=1"
             extra_args="--no-graph"
             [ "$model" = "qwen2.5-0.5b" ] && extra_env="$extra_env ENGINE_DBG_CPUEXEC=1"
             [ "$model" != "qwen2.5-0.5b" ] && extra_env="$extra_env ENGINE_DBG_CPUEXEC=1" ;;
    prefix1) n=65
             extra_env="ENGINE_TOPK=1 ENGINE_DBG_FWD=1 ENGINE_DBG_LAYERMS=1 ENGINE_KPROF=1"
             extra_args="--no-graph --layers 1" ;;
    prefixN) n=65
             extra_env="ENGINE_TOPK=1 ENGINE_DBG_FWD=1 ENGINE_DBG_LAYERMS=1 ENGINE_KPROF=1"
             extra_args="--no-graph --layers $(prefix_n "$model")" ;;
    prof)    n=5
             extra_env="ENGINE_TOPK=1 ENGINE_DBG_FWD=1 ENGINE_PROFILE=1"
             extra_args="--no-graph" ;;
    *) echo "unknown mode $mode" >&2; return 1 ;;
  esac
  case "$mode" in
    graph) timeout_ms=300000 ;;
    *) timeout_ms=240000 ;;
  esac
  extra_args="$extra_args $(pin_args "$model" "$mode")"
  out_dir="$ROOT/$model/$mode"
  mkdir -p "$out_dir"
  prefix="$out_dir/measurement"
  for attempt in $(seq 1 "$RETRIES"); do
    echo "=== $model / $mode (attempt $attempt) n=$n ==="
    rm -f "$prefix".*.csv "$prefix".*.log "$prefix".*.json "$prefix".*.txt
    env $extra_env \
      "$MON" --out-prefix "$prefix" --interval-ms "$INTERVAL_MS" \
             --wait-gpu-idle-sec "$WAIT_SEC" --idle-mem-max-mb "$IDLE_MEM_MB" --cpu-idle-max 45 \
             --during-check-ms 5000 --timeout-ms "$timeout_ms" \
             -- "$BIN" "$path" "$ids" "$n" $extra_args \
      2>&1 | tee "$out_dir/monitor.stdout.txt"
    local quiet foreign_dur
    quiet=$(json_get quiet_before_start "$prefix.meta.json")
    foreign_dur=$(json_get foreign_during_run_hits "$prefix.meta.json")
    if [ "$quiet" = "true" ] && { [ "$foreign_dur" = "0" ] || [ -z "$foreign_dur" ]; }; then
      if [ "$(json_get timed_out "$prefix.meta.json")" = "true" ]; then
        if grep -q "\[time\] prefill" "$prefix.engine.log" 2>/dev/null; then
          echo "  note: child timed out (engine hang) but decode data exists -> kept as partial"
        else
          echo "  !! child timed out before prefill; retrying"
          continue
        fi
      fi
      break
    fi
    echo "  !! run marked contaminated (quiet=$quiet foreign_during=$foreign_dur); retrying"
  done
  # sanity: the engine must have produced a decode line
  if ! grep -q "\[time\] decode" "$prefix.engine.log" 2>/dev/null; then
    echo "  !! no [time] decode line in $prefix.engine.log"
    tail -5 "$prefix.engine.log" 2>/dev/null
  fi
}

for model in $MODELS; do
  p=$(model_path "$model")
  if [ ! -f "$p" ]; then
    echo "-- SKIP $model (missing $p)"
    continue
  fi
  for mode in $MODES; do
    run_one "$model" "$mode"
  done
done

HASH_AFTER=$(sha256sum "$BIN" 2>/dev/null | cut -d' ' -f1)
if [ "$HASH_BEFORE" != "$HASH_AFTER" ]; then
  echo "!! engine snapshot changed during the suite ($HASH_BEFORE -> $HASH_AFTER)" | tee -a "$MANIFEST"
fi
echo "done; now run: python tools/utilization_report.py --root $ROOT --md $ROOT/summary.md --json $ROOT/summary.json"
