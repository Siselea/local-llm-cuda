# -*- coding: utf-8 -*-
"""Probe the live engine serve for thinking-phase repetition with webui params.

Sends a /v1/chat/completions SSE request the way the llama.cpp webui does and
counts how often the most frequent sentence repeats inside the thinking part.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://127.0.0.1:7030"
BASE = {
    "stream": True,
    "cache_prompt": True,
    "samplers": "edkypmxt",
    "dynatemp_range": 0,
    "min_p": 0.05,
    "xtc_probability": 0,
    "xtc_threshold": 0.1,
    "typical_p": 1,
    "repeat_last_n": 64,
    "repeat_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "dry_multiplier": 0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "timings_per_token": True,
}


def send(name, messages, *, temperature, top_k, top_p, max_tokens, cache=True,
         thinking=True):
    body = dict(BASE)
    body.update({
        "messages": messages,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "cache_prompt": cache,
    })
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    content = []
    timings = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
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
            t = d.get("content") or ""
            if t:
                if first is None:
                    first = time.time() - t0
                content.append(t)
            if j.get("timings"):
                timings = j["timings"]
    wall = time.time() - t0
    full = "".join(content)
    think, answer = "", full
    i = full.find("<think>")
    if i >= 0:
        rest = full[i + 7:]
        j = rest.find("</think>")
        if j >= 0:
            think, answer = rest[:j], rest[j + 8:]
        else:
            think, answer = rest, ""
    # repetition detector: split thinking into sentences, count max repeats
    sents = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s*", think) if len(s.strip()) >= 8]
    counts = {}
    for s in sents:
        counts[s] = counts.get(s, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    print(f"\n### {name}")
    print(f"  first_delta={first:.2f}s wall={wall:.2f}s prompt_n={timings.get('prompt_n')} "
          f"predicted_n={timings.get('predicted_n')} cached={timings.get('cache_hit_tokens')}")
    print(f"  think_len={len(think)} answer_len={len(answer)} "
          f"think_sentences={len(sents)} top_repeated={[(c, s[:40]) for s, c in top]}")
    print("  think(head):", repr(think[:300]))
    if answer:
        print("  answer(head):", repr(answer[:160]))
    return {"think": think, "answer": answer, "repeats": top, "timings": timings}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    msgs = [{"role": "user", "content": "你好"}]
    if mode in ("all", "temp08"):
        send("temp0.8/top_k40/top_p0.95 (webui default), cache on", msgs,
             temperature=0.8, top_k=40, top_p=0.95, max_tokens=256)
    if mode in ("all", "nocache"):
        send("temp0.8 webui default, cache_prompt=false", msgs,
             temperature=0.8, top_k=40, top_p=0.95, max_tokens=256, cache=False)
    if mode in ("all", "greedy"):
        send("greedy temp=0 (cache off)", msgs,
             temperature=0, top_k=40, top_p=0.95, max_tokens=256, cache=False)
    if mode in ("all", "qwenrec"):
        send("Qwen3 rec: temp0.6/top_k20/top_p0.95", msgs,
             temperature=0.6, top_k=20, top_p=0.95, max_tokens=256, cache=False)
