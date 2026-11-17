#!/usr/bin/env bash
# Keep a passive CPU capture hanging: whenever an engine.exe appears, attach the
# event-triggered watcher (tools/cpu_watch_trigger.py) to it and record the next
# generation(s).  Read-only counters; it never starts/kills/sends anything.
#
# Usage: DURATION=2700 bash tools/watch_when_engine.sh
set -uo pipefail
cd "$(dirname "$0")/.."
DURATION="${DURATION:-2700}"          # total wall time to keep watching (s)
PER_ENGINE_WATCH="${PER_ENGINE_WATCH:-600}"
PY="${PY:-python}"
LOG="build/passive-autocapture.log"
mkdir -p build
: >> "$LOG"
END=$(( $(date +%s) + DURATION ))
echo "[autocapture] start $(date '+%F %T')  duration=${DURATION}s" >> "$LOG"
seen_pid=""
while [ "$(date +%s)" -lt "$END" ]; do
  pid=$(powershell -NoProfile -Command \
        "(Get-Process engine -ErrorAction SilentlyContinue | Select-Object -First 1).Id" \
        2>/dev/null | tr -d '\r' | head -1)
  if [ -n "$pid" ] && [ "$pid" != "$seen_pid" ]; then
    echo "[autocapture] engine pid=$pid detected $(date '+%H:%M:%S') -> watching" >> "$LOG"
    "$PY" tools/cpu_watch_trigger.py --pid "$pid" --watch-seconds "$PER_ENGINE_WATCH" \
        --trigger-cores 2.5 --tag "live-$pid" >> "$LOG" 2>&1
    seen_pid="$pid"
    echo "[autocapture] engine pid=$pid watch ended $(date '+%H:%M:%S')" >> "$LOG"
  fi
  sleep 5
done
echo "[autocapture] done $(date '+%F %T')" >> "$LOG"
