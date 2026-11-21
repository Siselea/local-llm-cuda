#!/usr/bin/env bash
# One-shot cold-start forensics: run the prefill probe (2 identical-length requests:
# first = cold, second = warm) while sampling GPU clocks/power and disk read bytes.
# Answers: is the cold first prefill disk-bound (page-in) or GPU-clock/VRAM bound?
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-coldforensics}
ROWS=${2:-512}
CTX=${3:-32768}
CASES=${4:-2969,2969}
MODEL=${MODEL:-D:/p0/models/Qwen3.5-9B-Q5_K_M.gguf}
OUT=build/$TAG

: > "$OUT.gpu.csv"
: > "$OUT.disk.csv"
( while true; do nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader >> "$OUT.gpu.csv"; sleep 0.4; done ) &
GPU_SAMPLER=$!
( typeperf -sc 400 '\LogicalDisk(_Total)\Disk Read Bytes/sec' >> "$OUT.disk.csv" 2>/dev/null ) &
DISK_SAMPLER=$!

python tools/prefill_cliff_probe.py --model "$MODEL" --ctx "$CTX" \
  --rows "$ROWS" --rows-max "$ROWS" --tag "$TAG" --cases "$CASES" 2>&1 | \
  grep -E "healthy|req|wrote"

kill $GPU_SAMPLER $DISK_SAMPLER 2>/dev/null
wait 2>/dev/null
echo "--- GPU samples (clocks.sm, power, util, mem) ---"
sort "$OUT.gpu.csv" | uniq -c | sort -rn | head -6
echo "--- disk read during window (MiB/s max, mean of nonzero) ---"
python - "$OUT.disk.csv" <<'PY'
import sys
vals=[]
for ln in open(sys.argv[1], encoding='utf-8', errors='replace'):
    parts=[p.strip() for p in ln.strip('"\n\r ').split('","')]
    if len(parts)>=2:
        try: vals.append(float(parts[-1]))
        except ValueError: pass
if vals:
    nz=[v for v in vals if v>1e6]
    print(f"samples={len(vals)} max={max(vals)/1048576:.1f} MiB/s  "
          f"mean_nonzero={ (sum(nz)/len(nz)/1048576 if nz else 0):.1f} MiB/s  "
          f"samples>50MiB/s={sum(1 for v in nz if v>50*1048576)}")
else:
    print("no disk samples")
PY
