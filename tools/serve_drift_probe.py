# -*- coding: utf-8 -*-
"""Multi-turn thinking-drift probe with llama.cpp webui request parameters.

Runs a 3-turn Chinese chat exactly like the webui does (streaming SSE,
cache_prompt:true, webui sampling defaults), records for every turn:
  * first-delta latency, prefill ms, decode tok/s
  * the thinking text and the most repeated sentences inside it
  * the final answer

Usage: python tools/serve_drift_probe.py [--turns 3] [--temp 0.8] [--top-k 40]
       [--top-p 0.95] [--min-p 0.05] [--max-tokens 1024] [--greedy]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://127.0.0.1:7030"  # overridden by --url

TURNS = [
    "你好，请用三句话介绍杭州。",
    "再详细一点，重点说说它的历史和文化。",
    "用同样的风格介绍一下苏州。",
    "那南京呢？也按这个风格来。",
    "最后说说上海吧。",
]


def send(messages, *, temperature, top_k, top_p, min_p, max_tokens, thinking=True):
    body = {
        "messages": messages,
        "stream": True,
        "cache_prompt": True,
        "samplers": "edkypmxt",
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "dynatemp_range": 0,
        "xtc_probability": 0,
        "typical_p": 1,
        "repeat_last_n": 64,
        "repeat_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "dry_multiplier": 0,
        "max_tokens": max_tokens,
        "timings_per_token": True,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    content = []
    reasoning = []
    timings = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            ch = (j.get("choices") or [{}])[0]
            d = ch.get("delta") or {}
            rt = d.get("reasoning_content") or ""
            if rt:
                reasoning.append(rt)
            t = d.get("content") or ""
            if (t or rt) and first is None:
                first = time.time() - t0
            if t:
                content.append(t)
            if j.get("timings"):
                timings = j["timings"]
    wall = time.time() - t0
    full = "".join(content)
    think, answer = "".join(reasoning), full
    i = full.find("<think>") if not reasoning else -1
    if i >= 0:
        rest = full[i + 7:]
        j = rest.find("</think>")
        if j >= 0:
            think, answer = rest[:j], rest[j + 8:]
        else:
            think, answer = rest, ""
    return {"first": first, "wall": wall, "timings": timings, "content": full,
            "think": think, "answer": answer.strip()}


def repeat_stats(think: str):
    # sentences (keeps CJK/EN punctuation), only count non-trivial ones
    sents = [s.strip() for s in re.split(r"(?<=[.!?。！？；;\n])\s*", think) if len(s.strip()) >= 6]
    counts: dict[str, int] = {}
    for s in sents:
        counts[s] = counts.get(s, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    return len(sents), top


def main() -> int:
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:7030")
    args = ap.parse_args()
    URL = args.url

    temperature = 0.0 if args.greedy else args.temp
    messages = []
    worst = 0
    print(f"== drift probe: temp={temperature} top_k={args.top_k} top_p={args.top_p} "
          f"min_p={args.min_p} max_tokens={args.max_tokens} ==")
    for turn in range(min(args.turns, len(TURNS))):
        messages.append({"role": "user", "content": TURNS[turn]})
        r = send(messages, temperature=temperature, top_k=args.top_k, top_p=args.top_p,
                 min_p=args.min_p, max_tokens=args.max_tokens)
        n_sents, top = repeat_stats(r["think"])
        worst = max(worst, top[0][1] if top else 0)
        t = r["timings"]
        print(f"\n--- turn {turn + 1}: {TURNS[turn]!r}")
        print(f"  first_delta={(r['first'] or -1):.2f}s wall={r['wall']:.1f}s "
              f"prompt_n={t.get('prompt_n')} predicted_n={t.get('predicted_n')} "
              f"prefill={t.get('prompt_ms')}ms cached={t.get('cache_hit_tokens')}")
        print(f"  think_len={len(r['think'])} sentences={n_sents} top_repeats={top}")
        print(f"  think(head)={r['think'][:160]!r}")
        print(f"  answer(head)={r['answer'][:120]!r}")
        # history uses the answer only (webui excludeThoughtOnReq=true default)
        messages.append({"role": "assistant", "content": r["answer"]})
    print(f"\nMAX repeated sentence count: {worst}")
    print("DRIFT DETECTED" if worst >= 5 else "NO DRIFT (max repeat < 5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
