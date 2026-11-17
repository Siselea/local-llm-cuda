#!/usr/bin/env python3
"""Passive CPU watcher for a *running* engine process (the user's launcher session).

Samples the engine's per-core CPU, per-process CPU (in cores), and nvidia-smi GPU
util/power at ~50 ms.  No GPU work is issued, nothing is started or killed: this only
reads counters, so it can run while the user is chatting with the app.

Usage:
  python tools/cpu_watch.py --pid <engine pid> --seconds 120 --period 0.05 \
      --tag live35b

Analysis: contiguous "decode" windows = samples where GPU util >= 30 % or the engine
process burns >= 2 cores.  For each window: duration, process cores (median/max),
per-core mean %, GPU power/util.  Also prints the per-second sequence.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_probe import read_cpu_times, process_cpu_seconds, gpu_sample, _ncpu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--period", type=float, default=0.05)
    ap.add_argument("--tag", default="watch")
    args = ap.parse_args()

    proc = None
    samples = []
    t0 = time.perf_counter()
    prev_cpu = read_cpu_times()
    prev_proc = process_cpu_seconds(args.pid)
    prev_t = time.perf_counter()
    last_gpu = 0.0
    gpu = gpu_sample()
    while time.perf_counter() - t0 < args.seconds:
        time.sleep(args.period)
        now = time.perf_counter()
        cur_cpu = read_cpu_times()
        cur_proc = process_cpu_seconds(args.pid)
        if cur_proc is None:
            print("process exited")
            break
        dt = now - prev_t
        per_core = []
        for i in range(_ncpu):
            d_idle = cur_cpu[i][0] - prev_cpu[i][0]
            d_total = (cur_cpu[i][1] + cur_cpu[i][2]) - (prev_cpu[i][1] + prev_cpu[i][2])
            per_core.append(0.0 if d_total <= 0 else max(0.0, min(1.0, 1.0 - d_idle / d_total)))
        proc_cores = (cur_proc - prev_proc) / dt
        if now - last_gpu >= 0.5:
            gpu = gpu_sample()
            last_gpu = now
        samples.append({"t": round(now - t0, 3),
                        "proc_cores": round(proc_cores, 2),
                        "per_core": [round(v * 100, 1) for v in per_core],
                        "gpu": gpu})
        prev_cpu, prev_proc, prev_t = cur_cpu, cur_proc, now

    if not samples:
        print("no samples")
        return 1

    # decode window detection
    active = [s for s in samples
              if (s["gpu"] and s["gpu"]["util_pct"] >= 30) or s["proc_cores"] >= 2.0]
    # group contiguous active samples (gap <= 0.3 s)
    windows = []
    cur = []
    for s in samples:
        is_active = s in active
        if is_active:
            if cur and s["t"] - cur[-1]["t"] > 0.3:
                windows.append(cur)
                cur = []
            cur.append(s)
        elif cur:
            windows.append(cur)
            cur = []
    if cur:
        windows.append(cur)
    windows = [w for w in windows if w[-1]["t"] - w[0]["t"] >= 0.3]

    def med(vals):
        return statistics.median(vals) if vals else None

    summary = {"pid": args.pid, "seconds": args.seconds, "period": args.period,
               "n_samples": len(samples),
               "windows": [], "per_second": []}
    for k, w in enumerate(windows):
        pc = [s["proc_cores"] for s in w]
        per_core = [round(sum(s["per_core"][i] for s in w) / len(w), 1) for i in range(_ncpu)]
        gp = [s["gpu"]["power_w"] for s in w if s["gpu"]]
        gu = [s["gpu"]["util_pct"] for s in w if s["gpu"]]
        summary["windows"].append({
            "idx": k, "start_s": w[0]["t"], "dur_s": round(w[-1]["t"] - w[0]["t"], 2),
            "proc_cores_median": round(med(pc), 2), "proc_cores_max": max(pc),
            "per_core_mean_pct": per_core,
            "gpu_power_median_w": med(gp), "gpu_util_median_pct": med(gu),
        })
        print(f"[window {k}] {w[0]['t']:.1f}s +{w[-1]['t'] - w[0]['t']:.1f}s  "
              f"proc cores median {med(pc):.2f} max {max(pc):.2f}  "
              f"GPU util {med(gu) if med(gu) else 0:.0f}% / power "
              f"{med(gp) if med(gp) else 0:.0f}W")
        print(f"          per-core mean %: {per_core}")

    # per-second sequence
    sec = {}
    for s in samples:
        b = int(s["t"])
        sec.setdefault(b, []).append(s)
    for b in sorted(sec):
        grp = sec[b]
        pc = [x["proc_cores"] for x in grp]
        per_core = [round(sum(x["per_core"][i] for x in grp) / len(grp), 1)
                    for i in range(_ncpu)]
        gu = [x["gpu"]["util_pct"] for x in grp if x["gpu"]]
        row = {"sec": b, "proc_cores": round(sum(pc) / len(pc), 2),
               "per_core_pct": per_core, "gpu_util": round(sum(gu) / len(gu), 0) if gu else None}
        summary["per_second"].append(row)
        print(f"  {b:3d}s proc={row['proc_cores']:5.2f} cores gpu={row['gpu_util']}% "
              f"per-core {per_core}")

    idle = [s for s in samples if s not in active and not any(s in w for w in windows)]
    if idle:
        print(f"[idle baseline] {len(idle)} samples  proc cores median "
              f"{med([s['proc_cores'] for s in idle]):.2f}")
    with open(f"build/cpuwatch-{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(f"build/cpuwatch-{args.tag}.csv", "w", encoding="utf-8") as f:
        f.write("t,proc_cores," + ",".join(f"cpu{i}" for i in range(_ncpu)) +
                ",gpu_util,gpu_power\n")
        for s in samples:
            g = s["gpu"] or {}
            f.write(f"{s['t']},{s['proc_cores']}," +
                    ",".join(str(v) for v in s["per_core"]) +
                    f",{g.get('util_pct', '')},{g.get('power_w', '')}\n")
    print(f"[saved] build/cpuwatch-{args.tag}.json / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
