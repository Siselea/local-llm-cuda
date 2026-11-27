#!/usr/bin/env python
"""Cold full-prefill vs warm prefix-cache-hit consistency check (user-visible).

For two prompts (one short enough that `reuse_pos == 0` i.e. the cache cannot engage,
and one long enough to engage it) the same greedy chat request is sent three times to
`engine serve` with the prefix cache ON:
  call 1 = cache miss: history chunk (up to the cache point) + generation-prompt tail
  call 2/3 = cache hit: restore the recurrent-state snapshot, prefill only the tail
The generated token ids of every call must be identical.

Then the server is restarted with `--prefix-cache=0` (every call is a single full
prefill) and its ids are compared against call 1 of the cached server.  That cross-path
comparison is the one that can expose "same question answered differently depending on
the cache configuration" -- the cache-on cold path splits the prompt into
`chunk(cache_align*k) + tail` while cache-off prefills it in `prefill_rows` chunks, so
the two paths only agree if the batch numerics agree.

Usage:
  python tools/cache_cold_warm_check.py --engine ./build/engine.exe \
      --model ${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf [--port 7041] [--max-tokens 48]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SHORT = "用一句话介绍杭州。"
# ~40 x 14 CJK chars = ~500 chars => several hundred tokens: long enough that the
# cache point (history tokens aligned down to --prefix-cache-align, default 128) > 0.
LONG = "这是一段用于测试前缀缓存的长文本，请只回答最后一个问题：" + \
       "杭州西湖位于浙江省杭州市区西部，是国家重点风景名胜区。" * 18 + \
       "问题：杭州西湖在哪个省？"


def wait_health(port, timeout=240.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read().decode()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def chat(port, prompt, max_tokens):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "cache_prompt": True,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.loads(resp.read().decode("utf-8", "replace"))
    tm = d.get("timings", {})
    return {
        "ids": d.get("generated_token_ids") or [],
        "prompt_n": tm.get("prompt_n"),
        "cache_hit": tm.get("cache_hit_tokens", 0),
        "prefilled": tm.get("prefilled_tokens"),
        "prefill_ms": tm.get("prefill_ms"),
    }


def start_server(engine, model, port, prefix_cache, log_path):
    args = [engine, "serve", "--model", model, "--host", "127.0.0.1", "--port",
            str(port), "--ctx", "2048", "--think-budget", "0"]
    if not prefix_cache:
        args.append("--prefix-cache=0")
    log = open(log_path, "wb")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def stop_server(proc, log):
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        log.close()
    except Exception:
        pass


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def run_phase(tag, engine, model, port, prefix_cache, max_tokens, out_dir, prompts):
    log_path = os.path.join(out_dir, f"cache_{tag}.server.log")
    proc, log = start_server(engine, model, port, prefix_cache, log_path)
    try:
        if not wait_health(port):
            print(f"[{tag}] server failed to become ready: {log_path}")
            return None
        out = {}
        for name, prompt in prompts.items():
            runs = [chat(port, prompt, max_tokens) for _ in range(3)]
            out[name] = runs
            for i, r in enumerate(runs):
                print(f"[{tag}/{name}] call {i + 1}: prompt_n={r['prompt_n']} "
                      f"cache_hit={r['cache_hit']} prefilled={r['prefilled']} "
                      f"prefill={r['prefill_ms']:.1f}ms ids={len(r['ids'])}")
        return out
    finally:
        stop_server(proc, log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./build/engine.exe")
    ap.add_argument("--model", default=os.path.join(os.environ.get("MODEL_DIR", ""), "Qwen3.5-9B-Q5_K_M.gguf") if os.environ.get("MODEL_DIR") else "")
    ap.add_argument("--port", type=int, default=7041)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out-dir", default="build/logs/cold-warm")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = {"short": SHORT, "long": LONG}

    print(f"== {os.path.basename(args.model)}  same greedy chat request x3 per prompt ==")
    on = run_phase("cache_on", args.engine, args.model, args.port, True, args.max_tokens,
                   args.out_dir, prompts)
    off = run_phase("cache_off", args.engine, args.model, args.port, False, args.max_tokens,
                    args.out_dir, prompts)
    if not on or not off:
        return 1

    bad = 0
    for name in prompts:
        print(f"\n-- [{name}] cache ON: cold (call 1) vs warm (calls 2,3) --")
        for warm_i in (1, 2):
            d = first_diff(on[name][0]["ids"], on[name][warm_i]["ids"])
            if d is None:
                print(f"   cold vs call {warm_i + 1}: IDENTICAL "
                      f"({len(on[name][0]['ids'])} tokens)")
            else:
                bad += 1
                print(f"   cold vs call {warm_i + 1}: DIFFERS at token {d} "
                      f"(cold {on[name][0]['ids'][d]} vs warm {on[name][warm_i]['ids'][d]})")
        print(f"-- [{name}] cache OFF (all full prefill) vs cache ON call 1 (cold) --")
        for i in (0, 1, 2):
            d = first_diff(off[name][i]["ids"], on[name][0]["ids"])
            if d is None:
                print(f"   off call {i + 1} vs on call 1: IDENTICAL")
            else:
                bad += 1
                print(f"   off call {i + 1} vs on call 1: DIFFERS at token {d} "
                      f"(off {off[name][i]['ids'][d]} vs on {on[name][0]['ids'][d]})")

    print("\nresult:", "CONSISTENT (cold/warm/cache-off all identical)" if bad == 0
          else f"INCONSISTENT ({bad} mismatches)")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
