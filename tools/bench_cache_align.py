# -*- coding: utf-8 -*-
"""A/B benchmark: prefix-cache alignment (serve_cache_align) 0 vs 128.

Scenario: a ~9k-token first turn, then a short second turn.  With the 512/1024
prefill chunk the cache point used to be aligned to the chunk (turn-2 re-prefills
up to ~1023 history tokens); align=128 snapshots on a finer grid so turn-2 only
re-prefills the new tokens (expected 1.2 s -> ~0.25 s).

Usage: python build/bench_cache_align.py [--trials 3] [--binary build/engine_v014_wip.exe]
Requires an idle GPU.
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

PORT = 7066
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
         "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]
FILLER = " ".join(WORDS[i % len(WORDS)] for i in range(4400))  # ~5.8k tokens for 9B


def http(url, payload=None, timeout=2400):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def gpu_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=20)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def chat(port, messages, max_tokens=16):
    body = {"messages": messages, "stream": False, "temperature": 0, "top_k": 1,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    d = http(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    tm = d.get("timings", {})
    usage = d.get("usage") or {}
    return {"wall_s": round(time.time() - t0, 3), "prefill_ms": tm.get("prefill_ms"),
            "prefilled": tm.get("prefilled_tokens"), "cached": tm.get("cache_hit_tokens"),
            "prompt_n": tm.get("prompt_n", usage.get("prompt_tokens")),
            "answer": d["choices"][0]["message"].get("content") or "",
            "ids": d.get("generated_token_ids") or []}


def run_trial(binary, model, align, args):
    log_path = f"build/logs/cache_align_srv_{align}_{int(time.time())}.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    cmd = [binary, "serve", "--model", model, "--host", "127.0.0.1", "--port", str(PORT),
           "--ctx", "32768", "--gpu-layers", "-1", "--threads", "12",
           "--default-max-tokens", "1048576", "--dev-route"]
    if align is not None:
        cmd += ["--cache-align", str(align)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    out = {"align": align, "turns": {}}
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                raise RuntimeError(f"server died: {log_path}")
            try:
                http(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        # turn 1: long prompt (cold)
        m1 = [{"role": "user", "content": FILLER}]
        t1 = chat(PORT, m1)
        out["turns"]["turn1"] = t1
        # turn 2: same long prompt + short follow-up (history is the long prompt)
        m2 = m1 + [{"role": "assistant", "content": t1["answer"]},
                   {"role": "user", "content": "继续"}]
        t2 = chat(PORT, m2)
        out["turns"]["turn2"] = t2
    finally:
        try:
            http(f"http://127.0.0.1:{PORT}/shutdown", payload={}, timeout=5)
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/engine_v014_wip.exe")
    ap.add_argument("--model",
                    default=(os.environ["MODEL_DIR"] + "/Qwen3.5-9B-Q5_K_M.gguf")
                    if os.environ.get("MODEL_DIR") else None)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    if not args.model:
        print("ERROR: pass --model or set MODEL_DIR")
        return 2
    used = gpu_used()
    print(f"GPU used: {used} MiB")
    if used > 2500:
        print("REFUSING: GPU busy")
        return 2
    results = []
    for trial in range(1, args.trials + 1):
        order = [0, 128]
        if trial % 2 == 0:
            order.reverse()
        print(f"trial {trial}: align {order}")
        for align in order:
            r = run_trial(args.binary, args.model, align, args)
            r["trial"] = trial
            results.append(r)
            t2 = r["turns"]["turn2"]
            print(f"  align={align:3d}: turn1 prompt={r['turns']['turn1']['prompt_n']} "
                  f"prefill={r['turns']['turn1']['prefill_ms']:.0f}ms | "
                  f"turn2 prompt={t2['prompt_n']} prefill={t2['prefill_ms']:.0f}ms "
                  f"new={t2['prefilled']} cached={t2['cached']} wall={t2['wall_s']}s")
    summary = {}
    for align in (0, 128):
        rs = [r for r in results if r["align"] == align]
        summary[align] = {
            "turn1_prefill_median": statistics.median(
                [r["turns"]["turn1"]["prefill_ms"] for r in rs]),
            "turn2_prefill_median": statistics.median(
                [r["turns"]["turn2"]["prefill_ms"] for r in rs]),
            "turn2_new_median": statistics.median(
                [r["turns"]["turn2"]["prefilled"] or 0 for r in rs]),
            "turn2_wall_median": statistics.median(
                [r["turns"]["turn2"]["wall_s"] for r in rs]),
        }
    with open("build/logs/cache_align_ab.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, ensure_ascii=False, indent=2)
    print("\n== medians ==")
    for align in (0, 128):
        s = summary[align]
        print(f"align={align:3d}: turn1 {s['turn1_prefill_median']:.0f} ms | "
              f"turn2 prefill {s['turn2_prefill_median']:.0f} ms "
              f"(new {s['turn2_new_median']:.0f}) wall {s['turn2_wall_median']:.2f} s")
    d = summary[0]["turn2_prefill_median"] - summary[128]["turn2_prefill_median"]
    print(f"turn2 prefill saved: {d:.0f} ms "
          f"({100.0 * d / summary[0]['turn2_prefill_median']:.0f}%)")
    print("saved build/logs/cache_align_ab.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
