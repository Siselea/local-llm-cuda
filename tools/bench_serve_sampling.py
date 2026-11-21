#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interleaved A/B of serve non-greedy sampling (perf(serve), 2026-11-21).

Runs the *before* and *after* engine binaries alternately (5 trials each, order
flipped every trial so machine drift cancels) on the same 9B model and measures
non-streaming decode tok/s for two request groups:

  group A: temperature=0.6, top_p=0.95, top_k=20   (webui default path)
  group B: temperature=0.6, top_p=0.95, top_k=0    (llama.cpp-style top_p only)

Each server does one warm-up request first (CUDA graph capture), then the two
groups.  Results (per-trial ms/token and tok/s) go to build/logs/serve_sampling_ab.json
and are printed as a median table.

Usage:
  python tools/bench_serve_sampling.py --before build/engine_before_sampler.exe \
      --after build/engine_after_sampler.exe --model D:/p0/models/Qwen3.5-9B-Q5_K_M.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT = "用三句话介绍杭州。"
GROUPS = [
    ("top_k20", {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
    ("top_k0", {"temperature": 0.6, "top_p": 0.95, "top_k": 0}),
]


def http_json(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    return json.loads(body) if body.strip() else {}


def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=20)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


OUR_SERVERS = ("engine_smp_before.exe", "engine_smp_after.exe")


def other_engine_procs():
    """Count engine/test_gen/llama processes that are NOT the A/B servers we started.

    Anything else with a GPU footprint can distort the measurement (other agents run
    35B/9B benchmarks on the same machine), so trials with foreign processes are
    flagged and excluded from the summary.
    """
    try:
        out = subprocess.check_output(["tasklist"], text=True, timeout=20, errors="replace")
    except Exception:
        return 0
    n = 0
    for line in out.splitlines():
        low = line.lower()
        if ".exe" not in low:
            continue
        if any(name in low for name in OUR_SERVERS):
            continue
        if any(k in low for k in ("engine", "test_gen", "llama")):
            n += 1
    return n


def wait_for_idle(deadline, sleep_s=15.0):
    """Block until the GPU is idle and no foreign engine process is running."""
    while time.time() < deadline:
        used = gpu_used_mib()
        procs = other_engine_procs()
        print(f"  [idle-wait] gpu={used} MiB foreign_procs={procs}")
        if 0 <= used < 600 and procs == 0:
            return True
        time.sleep(sleep_s)
    return False


def start_server(binary, model, port, ctx, gpu_layers, threads, log_path):
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    cmd = [binary, "serve", "--model", model, "--host", "127.0.0.1", "--port", str(port),
           "--ctx", str(ctx), "--gpu-layers", str(gpu_layers),
           "--threads", str(threads), "--default-max-tokens", "1048576"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (rc={proc.returncode}); see {log_path}")
        try:
            http_json(f"http://127.0.0.1:{port}/health", timeout=3)
            return proc, log
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not become ready in 240 s")


def stop_server(proc, log, port):
    try:
        http_json(f"http://127.0.0.1:{port}/shutdown", payload={}, timeout=10)
    except Exception:
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    finally:
        log.close()


def chat_once(port, params, max_tokens, prompt_override=None):
    body = {"messages": [{"role": "user", "content": prompt_override or PROMPT}],
            "stream": False,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(params)
    t0 = time.perf_counter()
    d = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    wall = time.perf_counter() - t0
    tm = d.get("timings", {})
    gen = len(d.get("generated_token_ids") or [])
    decode_ms = tm.get("decode_ms") or 0.0
    return {"wall_s": round(wall, 3), "gen": gen, "decode_ms": round(decode_ms, 2),
            "tok_s": round(gen * 1000.0 / decode_ms, 2) if decode_ms > 0 and gen else None,
            "ms_per_token": round(decode_ms / gen, 3) if decode_ms > 0 and gen else None,
            "finish": d["choices"][0].get("finish_reason")}


def measure(binary, tag, args, trial, order):
    port = args.port
    log_path = os.path.join(args.log_dir, f"sampling_ab_{tag}_t{trial}.log")
    if other_engine_procs() != 0:
        return {"tag": tag, "trial": trial, "order": order, "contaminated": True,
                "reason": "foreign process before start", "groups": {}}
    out = {"tag": tag, "trial": trial, "order": order, "groups": {},
           "contaminated": False}
    proc, log = start_server(binary, args.model, port, args.ctx, args.gpu_layers,
                             args.threads, log_path)
    foreign = 0
    try:
        # Warm up thoroughly: graph capture, first-token path, and the cold-start
        # prefill effect (the first long prefill after load can be seconds slower).
        chat_once(port, GROUPS[0][1], 8)
        filler = ("杭州是一座有深厚历史文化的城市。" * 90)  # ~1.4k chars, fits ctx 2048
        chat_once(port, GROUPS[0][1], 8, prompt_override=filler)
        chat_once(port, GROUPS[0][1], 8)
        for name, params in GROUPS:
            r = chat_once(port, params, args.max_tokens)
            out["groups"][name] = r
            print(f"  [{tag} t{trial}] {name:8s} gen={r['gen']:3d} "
                  f"decode={r['decode_ms']:8.1f} ms  {r['tok_s']} tok/s  "
                  f"({r['ms_per_token']} ms/token)")
        return out
    finally:
        stop_server(proc, log, port)
        foreign = other_engine_procs()
        if foreign:
            print(f"  [{tag} t{trial}] WARNING: {foreign} foreign process(es) during the trial")
            out["contaminated"] = True
            out["reason"] = f"{foreign} foreign process(es) during the trial"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--model", default="D:/p0/models/Qwen3.5-9B-Q5_K_M.gguf")
    ap.add_argument("--port", type=int, default=7041)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--gpu-layers", default="-1")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--log-dir", default="build/logs")
    ap.add_argument("--out", default="build/logs/serve_sampling_ab.json")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if another GPU test process is present")
    ap.add_argument("--wait-idle", type=int, default=1800,
                    help="seconds to wait for an idle GPU before giving up")
    ap.add_argument("--min-clean", type=int, default=3,
                    help="minimum number of uncontaminated trials per tag")
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    used = gpu_used_mib()
    procs = other_engine_procs()
    print(f"GPU used: {used} MiB, foreign engine processes: {procs}")
    if not args.allow_busy:
        if used > 2500 or procs > 0:
            print(f"waiting up to {args.wait_idle}s for the GPU to become idle ...")
            if not wait_for_idle(time.time() + args.wait_idle):
                print("REFUSING: GPU never became idle (other benchmark still running).")
                return 2
    if not os.path.isfile(args.before) or not os.path.isfile(args.after):
        print("ERROR: before/after binary missing")
        return 2

    results = []
    for trial in range(1, args.trials + 1):
        order = [("before", args.before), ("after", args.after)]
        if trial % 2 == 0:
            order.reverse()
        print(f"trial {trial}/{args.trials}: {[t for t, _ in order]}")
        if not args.allow_busy:
            if other_engine_procs() != 0 or gpu_used_mib() >= 600:
                print("  [trial] another GPU test appeared; waiting for it to finish ...")
                if not wait_for_idle(time.time() + args.wait_idle):
                    print("  [trial] GPU never became idle -> stopping trials")
                    break
        for tag, binary in order:
            r = measure(binary, tag, args, trial, [t for t, _ in order])
            results.append(r)
            if r.get("contaminated"):
                print(f"  [{tag} t{trial}] CONTAMINATED: {r.get('reason')} "
                      f"(excluded from the summary)")

    summary = {}
    clean = [r for r in results if not r.get("contaminated")]
    clean_count = {t: sum(1 for r in clean if r["tag"] == t) for t in ("before", "after")}
    print(f"\nclean trials: {clean_count} (need >= {args.min_clean} per tag)")
    ok = all(v >= args.min_clean for v in clean_count.values())
    for name, _ in GROUPS:
        summary[name] = {}
        for tag in ("before", "after"):
            toks = [r["groups"][name]["tok_s"] for r in clean
                    if r["tag"] == tag and r["groups"][name]["tok_s"]]
            per = [r["groups"][name]["ms_per_token"] for r in clean
                   if r["tag"] == tag and r["groups"][name]["ms_per_token"]]
            summary[name][tag] = {
                "tok_s_median": round(statistics.median(toks), 2) if toks else None,
                "tok_s_mean": round(statistics.mean(toks), 2) if toks else None,
                "ms_token_median": round(statistics.median(per), 3) if per else None,
                "n": len(toks),
            }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "trials": results, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    print("\n== median decode speed (5 interleaved trials) ==")
    for name, _ in GROUPS:
        b = summary[name]["before"]
        a = summary[name]["after"]
        gain = (a["ms_token_median"] - b["ms_token_median"]) if (a["ms_token_median"] and
                                                               b["ms_token_median"]) else None
        pct = (100.0 * gain / b["ms_token_median"]) if gain is not None and b["ms_token_median"] else None
        def fmt(v, suffix=""):
            return f"{v:.2f}{suffix}" if isinstance(v, (int, float)) else "n/a"
        print(f"{name:8s} before {fmt(b['tok_s_median'])} tok/s ({fmt(b['ms_token_median'])} ms/tok) | "
              f"after {fmt(a['tok_s_median'])} tok/s ({fmt(a['ms_token_median'])} ms/tok) | "
              f"delta {fmt(gain)} ms/token ({fmt(pct, '%')})")
    print(f"\nsaved {args.out}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
