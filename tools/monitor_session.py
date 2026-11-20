#!/usr/bin/env python3
"""Passive performance monitor for a running llama.cpp server.

Samples (no GPU memory, no interference):
  * nvidia-smi: util / mem / power / temp / sm clock
  * llama-server /slots: request id, processing flag, prompt/decode counters
  * our own per-request derivation: prefill tok/s, decode tok/s

Writes build/monitor/llama-<ts>.csv (raw samples) and on exit prints a summary.
Usage: python tools/monitor_session.py [--port 8070] [--secs 3600] [--interval 1.0]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request

NVSMI = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
    "--format=csv,noheader,nounits",
]


def gpu_sample():
    try:
        out = subprocess.run(NVSMI, capture_output=True, text=True, timeout=5).stdout.strip()
        util, mem, power, temp, clk = [x.strip() for x in out.split(",")]
        return float(util), int(mem), float(power), int(temp), int(clk)
    except Exception:
        return (float("nan"), 0, float("nan"), 0, 0)


def slots(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--secs", type=int, default=3600)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--outdir", default="build/monitor")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(a.outdir, f"llama-{ts}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["t", "util", "mem_mb", "power_w", "temp_c", "clk_mhz",
                "busy", "task", "n_prompt", "prompt_done", "n_decoded"])

    # per-task bookkeeping
    cur_task = None
    task_first = {}
    task_last = {}
    completed = []
    t0 = time.time()
    print(f"[monitor] port={a.port} csv={path} secs={a.secs}", flush=True)

    while time.time() - t0 < a.secs:
        now = time.time() - t0
        util, mem, power, temp, clk = gpu_sample()
        sl = slots(a.port)
        busy = 0
        task = -1
        n_prompt = prompt_done = n_dec = -1
        if sl:
            s = sl[0]
            busy = 1 if s.get("is_processing") else 0
            task = s.get("id_task", -1)
            n_prompt = s.get("n_prompt_tokens", -1)
            prompt_done = s.get("n_prompt_tokens_processed", -1)
            n_dec = s.get("n_decoded", s.get("n_past", -1))
        w.writerow([f"{now:.1f}", util, mem, f"{power:.1f}", temp, clk,
                    busy, task, n_prompt, prompt_done, n_dec])
        f.flush()

        if busy:
            if task != cur_task:
                cur_task = task
                task_first[task] = (now, n_prompt, prompt_done, n_dec)
            task_last[task] = (now, n_prompt, prompt_done, n_dec)
        elif cur_task is not None and task != cur_task:
            # task finished between samples
            if cur_task in task_last:
                completed.append(cur_task)
            cur_task = None

        time.sleep(max(0.2, a.interval))

    f.close()
    # summary for every task we saw
    print("\n[monitor] per-request summary (derived from /slots deltas)")
    print(f"{'task':>6} {'prefill_tok/s':>14} {'decode_tok/s':>13} {'prompt':>7} {'decoded':>8} {'wall_s':>7}")
    for task in sorted(task_first):
        t_start, np0, pd0, nd0 = task_first[task]
        t_end, np1, pd1, nd1 = task_last.get(task, (t_start, np0, pd0, nd0))
        wall = max(1e-3, t_end - t_start)
        d_prompt = max(0, (pd1 - pd0) if pd0 >= 0 and pd1 >= 0 else 0)
        d_dec = max(0, (nd1 - nd0) if nd0 >= 0 and nd1 >= 0 else 0)
        print(f"{task:>6} {d_prompt/wall:>14.1f} {d_dec/wall:>13.2f} {np1:>7} {d_dec:>8} {wall:>7.1f}")
    print(f"[monitor] csv: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
