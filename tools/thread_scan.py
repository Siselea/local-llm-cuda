#!/usr/bin/env python3
"""Thread-count x affinity scan for the CPU host-GEMV layer (bench_cpu_layer layer mode).

Configs: threads {4,6,7,8,9,10,11,12} x win_affinity {0 = OS scheduling, 2 = SMT-paired}.
Every config is re-run REPS times; the run order is rotated one slot per rep so the
minutes-scale drift / background load of this box is shared by all configs instead of
penalising whichever config happened to run last.  Reports min / p25 / median / max
ms/layer and GB/s (median-based), sorted by median.

Usage: python tools/thread_scan.py [--reps 6] [--iters 150]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

THREADS = [4, 6, 7, 8, 9, 10, 11, 12]
AFFS = [0, 2]
AFF_NAME = {0: "os", 1: "phys-first", 2: "smt-pair"}


def run_one(binary, copies, iters, threads, aff):
    env = dict(os.environ)
    env["ENGINE_THREADS"] = str(threads)
    env["ENGINE_WIN_AFFINITY"] = str(aff)
    env.pop("ENGINE_DBG_AFFINITY", None)
    p = subprocess.run([binary, "layer", str(copies), str(iters)],
                       capture_output=True, text=True, env=env, timeout=120)
    ms = gbs = None
    for line in p.stdout.splitlines():
        if " ms/layer " in line:
            # | 0.752 ms/layer | 21.69 GB/s | chk ...
            for part in line.split("|"):
                part = part.strip()
                if part.endswith("ms/layer"):
                    ms = float(part.split()[0])
                elif part.endswith("GB/s"):
                    gbs = float(part.split()[0])
    if ms is None:
        raise RuntimeError(f"cannot parse output for t={threads} aff={aff}: {p.stdout!r} {p.stderr[-300:]!r}")
    return ms, gbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/bench_cpu_layer.exe")
    ap.add_argument("--copies", type=int, default=24)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", default="build/thread-scan")
    args = ap.parse_args()

    cfgs = [(t, a) for t in THREADS for a in AFFS]
    results = {c: [] for c in cfgs}
    gbs_res = {c: [] for c in cfgs}
    n = len(cfgs)
    print(f"scan: {n} configs x {args.reps} reps, iters={args.iters}, copies={args.copies}")
    for r in range(args.reps):
        off = r % n
        for k in range(n):
            cfg = cfgs[(k + off) % n]
            t, a = cfg
            ms, gbs = run_one(args.bin, args.copies, args.iters, t, a)
            results[cfg].append(ms)
            gbs_res[cfg].append(gbs)
            print(f"  rep{r + 1} t={t:2d} aff={AFF_NAME[a]:9s} {ms:7.4f} ms/layer {gbs:6.2f} GB/s",
                  flush=True)

    def stats(v):
        v = sorted(v)
        return {"min": v[0], "p25": v[len(v) // 4], "median": statistics.median(v),
                "max": v[-1], "mean": statistics.mean(v)}

    rows = []
    for cfg in cfgs:
        st = stats(results[cfg])
        st["threads"] = cfg[0]
        st["aff"] = AFF_NAME[cfg[1]]
        # bench's own GB/s = layer_bytes / (ms * 1e6); layer is 15.56 MiB = 16.31 MB
        st["gbs_median"] = 15.56 * 1.048576 / st["median"]
        rows.append(st)
    rows.sort(key=lambda r: r["median"])
    base_9 = next(r for r in rows if r["threads"] == 9 and r["aff"] == "os")
    print("\n== summary (sorted by median) ==")
    print(f"{'threads':>7} {'aff':9s} {'min':>8} {'p25':>8} {'median':>8} {'mean':>8} "
          f"{'max':>8} {'GB/s':>7} {'vs t9/os':>9}")
    for r in rows:
        delta = 100.0 * (r["median"] - base_9["median"]) / base_9["median"]
        print(f"{r['threads']:7d} {r['aff']:9s} {r['min']:8.4f} {r['p25']:8.4f} "
              f"{r['median']:8.4f} {r['mean']:8.4f} {r['max']:8.4f} "
              f"{r['gbs_median']:7.2f} {delta:+8.1f}%")
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "raw": {f"{k[0]}/{AFF_NAME[k[1]]}": v
                                         for k, v in results.items()}}, f, indent=1)
    with open(args.out + ".csv", "w", encoding="utf-8") as f:
        f.write("threads,aff,min,p25,median,mean,max,gbs_median\n")
        for r in rows:
            f.write(f"{r['threads']},{r['aff']},{r['min']:.4f},{r['p25']:.4f},"
                    f"{r['median']:.4f},{r['mean']:.4f},{r['max']:.4f},{r['gbs_median']:.3f}\n")
    print(f"[saved] {args.out}.json / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
