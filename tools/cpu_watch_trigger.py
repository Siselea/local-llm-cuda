#!/usr/bin/env python3
"""Event-triggered passive CPU capture for a *live* engine process.

Waits (read-only) until the engine actually starts generating (process CPU rises above
~1 core for two consecutive 0.3 s probes), then records per-logical-core utilisation +
process CPU + GPU util/power at 50 ms until generation ends, and correlates the capture
with the engine's own `[req] ... decode ... tok/s` log line.

Nothing is started, killed or sent: this only reads performance counters, so it is safe
to run while the user is chatting with the app.

Usage:
  python tools/cpu_watch_trigger.py --pid <engine pid> \
      --watch-seconds 600 --tag live35b
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_probe import read_cpu_times, process_cpu_seconds, gpu_sample, _ncpu  # noqa: E402

LOG_DEFAULT = "build/launcher-engine.log"


def req_lines(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [ln.strip() for ln in f if ln.startswith("[req]")]
    except OSError:
        return []


class Recorder:
    def __init__(self, pid, period=0.05):
        self.pid = pid
        self.period = period
        self.samples = []

    def capture(self, max_seconds=40.0, idle_stop=4.0):
        prev_cpu = read_cpu_times()
        prev_proc = process_cpu_seconds(self.pid)
        prev_t = time.perf_counter()
        gpu = gpu_sample()
        last_gpu = 0.0
        start = prev_t
        idle = 0.0
        last_active = start
        while True:
            time.sleep(self.period)
            now = time.perf_counter()
            cur_cpu = read_cpu_times()
            cur_proc = process_cpu_seconds(self.pid)
            if cur_proc is None:
                return False
            dt = now - prev_t
            per_core = []
            for i in range(_ncpu):
                d_idle = cur_cpu[i][0] - prev_cpu[i][0]
                d_total = (cur_cpu[i][1] + cur_cpu[i][2]) - (prev_cpu[i][1] + prev_cpu[i][2])
                per_core.append(0.0 if d_total <= 0 else
                                max(0.0, min(1.0, 1.0 - d_idle / d_total)))
            proc_cores = (cur_proc - prev_proc) / dt
            if now - last_gpu >= 0.5:
                gpu = gpu_sample()
                last_gpu = now
            self.samples.append({
                "t": round(now - start, 3),
                "proc_cores": round(proc_cores, 2),
                "per_core": [round(v * 100, 1) for v in per_core],
                "gpu": gpu,
            })
            if proc_cores >= 0.5:
                idle = 0.0
                last_active = now
            else:
                idle += dt
            end = now - start
            if end >= max_seconds:
                break
            if end > 1.0 and idle >= idle_stop and now - last_active > idle_stop:
                break
            prev_cpu, prev_proc, prev_t = cur_cpu, cur_proc, now
        return True


def summarize(samples, req_line):
    if not samples:
        return {}
    pc = [s["proc_cores"] for s in samples]
    per_core = [round(sum(s["per_core"][i] for s in samples) / len(samples), 1)
                for i in range(_ncpu)]
    gp = [s["gpu"]["power_w"] for s in samples if s["gpu"]]
    gu = [s["gpu"]["util_pct"] for s in samples if s["gpu"]]
    free = [s["gpu"]["free_mib"] for s in samples if s["gpu"] and "free_mib" in s["gpu"]]

    def frac(th):
        return round(100.0 * sum(1 for v in pc if v >= th) / len(pc), 1)

    # "CPU burst" samples: process at >=70 % of its observed peak -> the host-GEMV bursts
    peak = max(pc)
    burst = [s for s in samples if s["proc_cores"] >= 0.7 * peak]
    burst_core = [round(sum(s["per_core"][i] for s in burst) / len(burst), 1)
                  for i in range(_ncpu)]
    # per-second sequence
    per_second = {}
    for s in samples:
        b = int(s["t"])
        per_second.setdefault(b, []).append(s)
    seq = []
    for b in sorted(per_second):
        grp = per_second[b]
        seq.append({"sec": b,
                    "proc_cores": round(sum(x["proc_cores"] for x in grp) / len(grp), 2),
                    "per_core_pct": [round(sum(x["per_core"][i] for x in grp) / len(grp), 1)
                                     for i in range(_ncpu)],
                    "gpu_util": round(sum((x["gpu"] or {}).get("util_pct", 0)
                                          for x in grp) / len(grp))})
    return {
        "samples": len(samples),
        "duration_s": round(samples[-1]["t"], 2),
        "proc_cores_median": round(statistics.median(pc), 2),
        "proc_cores_max": peak,
        "proc_cores_mean": round(sum(pc) / len(pc), 2),
        "duty_ge4_pct": frac(4.0),
        "duty_ge6_pct": frac(6.0),
        "duty_ge8_pct": frac(8.0),
        "per_core_mean_pct": per_core,
        "burst_samples": len(burst),
        "burst_per_core_mean_pct": burst_core,
        "gpu_power_median_w": statistics.median(gp) if gp else None,
        "gpu_util_median_pct": statistics.median(gu) if gu else None,
        "gpu_free_min_mib": min(free) if free else None,
        "per_second": seq,
        "req_line": req_line,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--watch-seconds", type=float, default=600)
    ap.add_argument("--period", type=float, default=0.05)
    ap.add_argument("--tag", default="live")
    ap.add_argument("--log", default=LOG_DEFAULT)
    ap.add_argument("--max-captures", type=int, default=5)
    ap.add_argument("--trigger-cores", type=float, default=2.2,
                    help="engine process CPU (in cores) that counts as 'generating'; "
                         "model load stays below ~2, a 25-CPU-layer decode is ~4")
    args = ap.parse_args()

    started = time.time()
    # idle baseline
    rec = Recorder(args.pid, args.period)
    probe_prev = process_cpu_seconds(args.pid)
    probe_t = time.perf_counter()
    hits = []
    warm = 0
    cap = 0
    print(f"[trigger] watching pid {args.pid} for {args.watch_seconds:.0f}s "
          f"(per-core @ {int(1000 * args.period)} ms)", flush=True)
    while time.time() - started < args.watch_seconds and cap < args.max_captures:
        time.sleep(0.3)
        cur = process_cpu_seconds(args.pid)
        if cur is None:
            print("[trigger] process exited")
            break
        now = time.perf_counter()
        cores = (cur - probe_prev) / max(now - probe_t, 1e-6)
        probe_prev, probe_t = cur, now
        if cores >= args.trigger_cores:
            warm += 1
        else:
            warm = 0
        if warm >= 2:
            warm = 0
            before = len(req_lines(args.log))
            print(f"[trigger] generation detected (eng cores {cores:.1f}) -> capture "
                  f"#{cap + 1}", flush=True)
            rec = Recorder(args.pid, args.period)
            rec.capture()
            entry = summarize(rec.samples, None)
            reqs = req_lines(args.log)
            if reqs:
                entry["req_line"] = reqs[-1]
            entry["capture_idx"] = cap
            hits.append(entry)
            cap += 1
            print(f"[trigger] capture #{cap}: {entry['duration_s']}s  proc "
                  f"{entry['proc_cores_median']} cores  duty>=6 {entry['duty_ge6_pct']}%  "
                  f"{entry['req_line'] or ''}", flush=True)
            time.sleep(1.0)
    if not hits:
        print("[trigger] no generation observed within the watch window")
        return 1
    with open(f"build/cpuwatch-{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(hits, f, ensure_ascii=False, indent=1)
    for h in hits:
        print(f"\n== capture {h['capture_idx']} ({h['duration_s']}s) ==")
        print(f"  process CPU: median {h['proc_cores_median']} cores, "
              f"mean {h['proc_cores_mean']}, max {h['proc_cores_max']}")
        print(f"  duty: >=4 cores {h['duty_ge4_pct']}%, >=6 cores {h['duty_ge6_pct']}%, "
              f">=8 cores {h['duty_ge8_pct']}%")
        print(f"  per-core mean %: {h['per_core_mean_pct']}")
        print(f"  CPU-burst samples (n={h['burst_samples']}): per-core mean % "
              f"{h['burst_per_core_mean_pct']}")
        print(f"  GPU: util {h['gpu_util_median_pct']}%, power {h['gpu_power_median_w']} W, "
              f"min free {h['gpu_free_min_mib']} MiB")
        print(f"  req: {h['req_line'] or '(none)'}")
        print("  per-second (proc cores | per-core % | gpu%):")
        for row in h["per_second"][:20]:
            print(f"    {row['sec']:3d}s {row['proc_cores']:5.2f} | {row['per_core_pct']} "
                  f"| {row['gpu_util']}%")
    print(f"\n[saved] build/cpuwatch-{args.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
