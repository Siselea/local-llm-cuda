#!/usr/bin/env python3
"""Decode-time CPU probe for the engine (test_gen direct measurement).

Per logical core utilisation + per-process CPU (in cores) + GPU power/util, sampled
during the *decode* window of one test_gen run.  Pure ctypes: reads
NtQuerySystemInformation(SystemProcessorPerformanceInformation) and GetProcessTimes,
so the sampler itself costs well under 1% CPU (no extra processes during the window).

Usage:
  python tools/cpu_probe.py --model <gguf> --ids 760 --n 300 \
      --threads 9 --tag 35b-t9

Writes build/cpuprobe-<tag>.csv (per sample) and build/cpuprobe-<tag>.json (summary).
"""
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("IdleTime", ctypes.c_longlong),
        ("KernelTime", ctypes.c_longlong),
        ("UserTime", ctypes.c_longlong),
        ("DpcTime", ctypes.c_longlong),
        ("InterruptTime", ctypes.c_longlong),
        ("InterruptCount", ctypes.c_ulong),
    ]


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]


def ft_to_i64(ft):
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


_ncpu = max(1, os.cpu_count() or 1)
_buf = (SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION * _ncpu)()
_ret = ctypes.c_ulong(0)


def read_cpu_times():
    """Return list of (idle, kernel, user) 100ns units per logical core."""
    ntdll.NtQuerySystemInformation.restype = ctypes.c_long
    st = ntdll.NtQuerySystemInformation(
        8, ctypes.byref(_buf), ctypes.sizeof(_buf), ctypes.byref(_ret))
    if st != 0:
        raise OSError(f"NtQuerySystemInformation failed status=0x{st & 0xffffffff:x}")
    return [(_buf[i].IdleTime, _buf[i].KernelTime, _buf[i].UserTime) for i in range(_ncpu)]


def process_cpu_seconds(pid):
    """kernel+user seconds of the process, or None when it is gone."""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        c, e, k, u = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        if not kernel32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                                        ctypes.byref(k), ctypes.byref(u)):
            return None
        return (ft_to_i64(k) + ft_to_i64(u)) / 1e7
    finally:
        kernel32.CloseHandle(h)


def gpu_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return out
    except Exception:
        return "?"


def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if not out:
            return None
        p, u, c, mf = [x.strip() for x in out[0].split(",")[:4]]
        return {"power_w": float(p), "util_pct": float(u), "sm_mhz": float(c),
                "free_mib": float(mf)}
    except Exception:
        return None


class Sampler(threading.Thread):
    def __init__(self, pid, period=0.1, gpu_period=1.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.period = period
        self.gpu_period = gpu_period
        self.samples = []
        self.stop = threading.Event()
        self.t0 = time.perf_counter()

    def run(self):
        prev_cpu = read_cpu_times()
        prev_proc = process_cpu_seconds(self.pid)
        prev_t = time.perf_counter()
        last_gpu_t = 0.0
        gpu = None
        while not self.stop.is_set():
            self.stop.wait(self.period)
            now = time.perf_counter()
            cur_cpu = read_cpu_times()
            cur_proc = process_cpu_seconds(self.pid)
            if cur_proc is None:
                break
            dt = now - prev_t
            per_core = []
            for i in range(_ncpu):
                d_idle = cur_cpu[i][0] - prev_cpu[i][0]
                d_total = (cur_cpu[i][1] + cur_cpu[i][2]) - \
                          (prev_cpu[i][1] + prev_cpu[i][2])
                per_core.append(0.0 if d_total <= 0 else
                                max(0.0, min(1.0, 1.0 - d_idle / d_total)))
            proc_cores = None
            if prev_proc is not None:
                proc_cores = (cur_proc - prev_proc) / dt
            if now - last_gpu_t >= self.gpu_period:
                gpu = gpu_sample()
                last_gpu_t = now
            self.samples.append({
                "t": now - self.t0,
                "per_core": [round(v * 100, 1) for v in per_core],
                "core_mean_pct": round(100.0 * sum(per_core) / _ncpu, 1),
                "busy_cores": round(sum(per_core), 2),
                "proc_cores": None if proc_cores is None else round(proc_cores, 2),
                "gpu": gpu,
            })
            prev_cpu, prev_proc, prev_t = cur_cpu, cur_proc, now


def thread_states(pid):
    """One-shot PowerShell snapshot of the process' thread states (mid-decode)."""
    ps = (
        "$p=Get-Process -Id %d -ErrorAction SilentlyContinue; if(-not $p){'gone';exit}; "
        "$p.Threads | Group-Object {$_.ThreadState} | ForEach-Object { "
        "'{0}={1}' -f $_.Name, $_.Count }; "
        "$p.Threads | Group-Object {$_.WaitReason} | ForEach-Object { "
        "'wait:{0}={1}' -f $_.Name, $_.Count }"
    ) % pid
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=20)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception as e:
        return [f"thread-state probe failed: {e}"]


def median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bin", default="build/test_gen.exe")
    ap.add_argument("--extra-env", default="")
    ap.add_argument("--period", type=float, default=0.1)
    ap.add_argument("--min-free-mib", type=int, default=0,
                    help="skip the run (rc=3) when free VRAM is below this")
    args = ap.parse_args()

    gpu_before = gpu_free_mib()
    if args.min_free_mib > 0:
        try:
            free_mib = int(gpu_before.split(",")[0])
        except Exception:
            free_mib = 0
        if free_mib < args.min_free_mib:
            print(f"[{args.tag}] SKIP: free VRAM {gpu_before.split(',')[0]} MiB "
                  f"< {args.min_free_mib} MiB (another process is using the GPU)")
            return 3
    cmd = [args.bin, args.model, args.ids, str(args.n)]
    if args.threads > 0:
        cmd += ["--threads", str(args.threads)]
    env = dict(os.environ)
    env.setdefault("ENGINE_TOPK", "0")
    for kv in args.extra_env.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v

    log = open(f"build/cpuprobe-{args.tag}.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, text=True, encoding="utf-8", errors="replace")
    sampler = Sampler(proc.pid, period=args.period)
    sampler.start()

    t_prefill = None
    t_decode_end = None
    pool_line = None
    stderr_lines = []
    stderr_t0 = time.perf_counter()

    def read_stderr():
        nonlocal t_prefill, t_decode_end, pool_line
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())
            log.write(line)
            log.flush()
            if "[time] prefill" in line:
                t_prefill = time.perf_counter()
            if "[time] decode" in line:
                t_decode_end = time.perf_counter()
            if "[cpu] pool threads" in line:
                pool_line = line.strip()

    readers = [threading.Thread(target=read_stderr, daemon=True)]
    for r in readers:
        r.start()
    # drain stdout in the background so the child never blocks on a full pipe
    out_lines = []

    def read_stdout():
        for line in proc.stdout:
            out_lines.append(line.rstrip())

    so = threading.Thread(target=read_stdout, daemon=True)
    so.start()

    # wait for the prefill line (decode start), then probe thread states mid-decode
    while t_prefill is None and proc.poll() is None:
        time.sleep(0.05)
    tsnap = []
    if proc.poll() is None:
        time.sleep(2.0)
        tsnap = thread_states(proc.pid)
        # also capture the pool-thread line if the engine printed it before/at prefill
    rc = proc.wait()
    sampler.stop.set()
    sampler.join(timeout=5)
    for r in readers:
        r.join(timeout=2)
    so.join(timeout=2)
    log.close()

    wall = time.perf_counter() - stderr_t0
    samples = sampler.samples
    if t_prefill is None:
        # fall back: last 70% of samples (generation is the tail of the run)
        k = int(len(samples) * 0.3)
        window = samples[k:]
        window_kind = "fallback-last70pct"
    else:
        t0 = t_prefill - sampler.t0
        t1 = ((t_decode_end or time.perf_counter()) - sampler.t0)
        window = [s for s in samples if t0 <= s["t"] <= t1]
        window_kind = "prefill-line-to-decode-line"

    per_core_med = []
    for i in range(_ncpu):
        per_core_med.append(round(median([s["per_core"][i] for s in window]) or 0, 1))
    per_core_mean = []
    for i in range(_ncpu):
        vals = [s["per_core"][i] for s in window]
        per_core_mean.append(round(sum(vals) / len(vals), 1) if vals else 0.0)
    proc_cores_med = median([s["proc_cores"] for s in window])
    proc_cores_max = max([s["proc_cores"] for s in window if s["proc_cores"] is not None],
                         default=None)
    busy_cores_med = median([s["busy_cores"] for s in window])
    gpus = [s["gpu"] for s in window if s["gpu"]]
    power = [g["power_w"] for g in gpus]
    util = [g["util_pct"] for g in gpus]
    freeg = [g["free_mib"] for g in gpus if "free_mib" in g]
    min_free = min(freeg) if freeg else None
    # contamination guard: a second CUDA context appearing mid-run shows up as a
    # sudden free-VRAM drop (and made the decode 3-6x slower in earlier attempts)
    contaminated = min_free is not None and min_free < 3000

    core_counts = [s["proc_cores"] for s in window if s["proc_cores"] is not None]
    max_proc = max(core_counts) if core_counts else 0.0

    def frac_ge(th):
        if not core_counts:
            return None
        return round(100.0 * sum(1 for v in core_counts if v >= th) / len(core_counts), 1)

    # per-second sequence (mean per core + process cores) inside the decode window
    per_second = []
    if window:
        t0w = window[0]["t"]
        buckets = {}
        for smp in window:
            b = int(smp["t"] - t0w)
            buckets.setdefault(b, []).append(smp)
        for b in sorted(buckets):
            grp = buckets[b]
            pc = [round(sum(x["per_core"][i] for x in grp) / len(grp), 1)
                  for i in range(_ncpu)]
            cc = [x["proc_cores"] for x in grp if x["proc_cores"] is not None]
            per_second.append({"sec": b,
                               "proc_cores": round(sum(cc) / len(cc), 2) if cc else None,
                               "per_core_pct": pc})

    # CPU-phase samples: those where the process runs at >=70 % of its observed peak
    # (the host-GEMV burst).  Their per-core vector is the "6 physical cores busy" view.
    phase = [s for s in window
             if s["proc_cores"] is not None and s["proc_cores"] >= 0.7 * max_proc]
    phase_cores = []
    for i in range(_ncpu):
        vals = [s["per_core"][i] for s in phase]
        phase_cores.append(round(sum(vals) / len(vals), 1) if vals else 0.0)

    timing = {}
    for line in stderr_lines:
        if "[time] decode" in line:
            timing["decode_line"] = line.strip()
        if "[time] prefill" in line:
            timing["prefill_line"] = line.strip()
        if "[cpu] pool threads" in line:
            timing["pool_line"] = line.strip()

    summary = {
        "tag": args.tag,
        "model": os.path.basename(args.model),
        "threads_arg": args.threads,
        "n": args.n,
        "rc": rc,
        "wall_s": round(wall, 1),
        "window_kind": window_kind,
        "window_s": round(window[-1]["t"] - window[0]["t"], 2) if window else 0,
        "n_samples": len(window),
        "sample_period_s": sampler.period,
        "proc_cores_median": None if proc_cores_med is None else round(proc_cores_med, 2),
        "proc_cores_max": None if proc_cores_max is None else round(proc_cores_max, 2),
        "busy_cores_median": None if busy_cores_med is None else round(busy_cores_med, 2),
        "per_core_median_pct": per_core_med,
        "per_core_mean_pct": per_core_mean,
        "gpu_power_median_w": median(power),
        "gpu_power_max_w": max(power) if power else None,
        "gpu_util_median_pct": median(util),
        "gpu_util_max_pct": max(util) if util else None,
        "pool_line": pool_line,
        "timing": timing,
        "thread_states_mid_decode": tsnap,
        "gpu_before": gpu_before.strip(),
        "gpu_free_min_mib": min_free,
        "contaminated": bool(contaminated),
        "duty_proc_ge_4cores_pct": frac_ge(4.0),
        "duty_proc_ge_6cores_pct": frac_ge(6.0),
        "duty_proc_ge_8cores_pct": frac_ge(8.0),
        "phase_threshold_cores": round(0.7 * max_proc, 2),
        "phase_samples": len(phase),
        "phase_per_core_mean_pct": phase_cores,
        "per_second": per_second,
    }
    # thread count right after the run is impossible (process gone); the mid-decode
    # snapshot strings carry e.g. "Running=10" / "Wait=..."
    with open(f"build/cpuprobe-{args.tag}.csv", "w", encoding="utf-8") as f:
        f.write("t,proc_cores,busy_cores,core_mean_pct," +
                ",".join(f"cpu{i}" for i in range(_ncpu)) + ",gpu_power_w,gpu_util\n")
        for s in samples:
            g = s["gpu"] or {}
            f.write(f"{s['t']:.3f},{s['proc_cores']},{s['busy_cores']},{s['core_mean_pct']}," +
                    ",".join(str(v) for v in s["per_core"]) +
                    f",{g.get('power_w','')},{g.get('util_pct','')}\n")
    with open(f"build/cpuprobe-{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(f"[{args.tag}] {summary['model']} threads_arg={args.threads} rc={rc} "
          f"wall={summary['wall_s']}s window={summary['window_s']}s ({window_kind})")
    print(f"  {timing.get('prefill_line', '')}")
    print(f"  {timing.get('decode_line', '')}")
    print(f"  {timing.get('pool_line', '(no [cpu] pool threads line)')}")
    print(f"  process CPU: median {summary['proc_cores_median']} cores, "
          f"max {summary['proc_cores_max']}  (busy_cores {summary['busy_cores_median']})")
    print(f"  per-core median %: {per_core_med}")
    print(f"  per-core mean   %: {per_core_mean}")
    print(f"  GPU: power median {fmt(summary['gpu_power_median_w'])} W "
          f"(max {fmt(summary['gpu_power_max_w'])}), util median "
          f"{fmt(summary['gpu_util_median_pct'])} % (max {fmt(summary['gpu_util_max_pct'])})")
    print(f"  duty cycle: proc CPU >=4 cores {fmt(summary['duty_proc_ge_4cores_pct'])}% of "
          f"decode samples, >=6 cores {fmt(summary['duty_proc_ge_6cores_pct'])}%, "
          f">=8 cores {fmt(summary['duty_proc_ge_8cores_pct'])}%")
    print(f"  CPU-phase samples (proc >= {summary['phase_threshold_cores']} cores, "
          f"n={len(phase)}): per-core mean % {phase_cores}")
    if per_second:
        print("  per-second (sec: proc_cores | per-core %):")
        for row in per_second[:12]:
            print(f"    {row['sec']:3d}s {row['proc_cores']:5.2f} | {row['per_core_pct']}")
    print(f"  GPU before: {summary['gpu_before']}  min free during decode: "
          f"{min_free if min_free is not None else '?'} MiB"
          f"{'  *** CONTAMINATED (another GPU process appeared) ***' if contaminated else ''}")
    if tsnap:
        print("  thread states (mid-decode): " + " ".join(tsnap))
    return 0 if rc == 0 else 1


def fmt(v):
    return "-" if v is None else f"{v:.1f}"


if __name__ == "__main__":
    sys.exit(main())
