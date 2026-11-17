#!/usr/bin/env python
"""Probe an `engine serve` instance the same way the llama.cpp webui does.

The llama.cpp b6000 webui:
  * POSTs /v1/chat/completions with stream:true, cache_prompt:true, max_tokens:-1,
    llama.cpp sampling fields (samplers/top_k/top_p/min_p/...);
  * renders ONLY choices[0].delta.content (it ignores reasoning_content!), and
    parses  thinking...<｜end▁of▁thinking｜> tags client-side into a collapsible thought block;
  * strips the thought from assistant history before the next request
    (excludeThoughtOnReq=true: content.split("</think>")[-1].trim()).

This tool measures:
  1. time to first visible content delta (the "spinner" latency),
  2. whether the first deltas carry the  thinking wrapper and the thought text,
  3. the assembled answer outside the think block,
  4. multi-turn prefix-cache behaviour (per-turn prefill time & cache hit tokens
     from the response `timings` + the server log lines if --log is given).

Usage:
  python tools/serve_webui_probe.py --url http://127.0.0.1:7030 [--turns 3]
                                   [--log build/logs/serve.log] [--max-tokens 1024]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def encode_body(messages, *, stream, max_tokens, thinking):
    # llama.cpp webui defaults (see build/webui/index.html config object)
    body = {
        "messages": messages,
        "stream": stream,
        "cache_prompt": True,
        "samplers": "edkypmxt",
        "temperature": 0.8,
        "dynatemp_range": 0,
        "top_k": 40,
        "top_p": 0.95,
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
        "max_tokens": max_tokens,
        "timings_per_token": True,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def stream_chat(url, messages, max_tokens, thinking):
    data = json.dumps(encode_body(messages, stream=True, max_tokens=max_tokens,
                                  thinking=thinking), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first_delta = None
    content, reasoning = [], []
    timings = None
    finish = None
    n_chunks = 0
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
            n_chunks += 1
            ch = (j.get("choices") or [{}])[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            txt = delta.get("content") or ""
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if txt:
                if first_delta is None:
                    first_delta = time.time() - t0
                content.append(txt)
            if j.get("timings"):
                timings = j["timings"]
    wall = time.time() - t0
    full = "".join(content)
    # webui-style split: content outside <think>...</think> is the answer
    think_part = ""
    answer = full
    i = full.find("<think>")
    if i >= 0:
        rest = full[i + len("<think>"):]
        j = rest.find("</think>")
        if j >= 0:
            think_part = rest[:j]
            answer = full[:i] + rest[j + len("</think>"):]
        else:  # thought ran into the token limit without closing
            think_part = rest
            answer = full[:i]
    answer = answer.strip()
    return {
        "first_delta_s": first_delta,
        "wall_s": wall,
        "n_chunks": n_chunks,
        "content": full,
        "think": think_part,
        "answer": answer,
        "finish_reason": finish,
        "timings": timings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7030")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=-1, help="-1 = webui default (no limit)")
    ap.add_argument("--thinking", type=int, default=1, help="1 = enable thinking (webui default)")
    ap.add_argument("--log", default="", help="server log file; prints [req] lines for the turns")
    ap.add_argument("--question", default="请用一句话介绍你自己。")
    args = ap.parse_args()

    print(f"== webui-style SSE probe: {args.url} (thinking={args.thinking}) ==")
    messages = [{"role": "user", "content": args.question}]
    rows = []
    for turn in range(1, args.turns + 1):
        r = stream_chat(args.url, messages, args.max_tokens, bool(args.thinking))
        rows.append(r)
        fd = r["first_delta_s"]
        t = r["timings"] or {}
        prefill = t.get("prompt_ms", t.get("prefill_ms"))
        pn = t.get("prompt_n")
        print(f"\n--- turn {turn}: messages={len(messages)}")
        print(f"  first visible content delta : {fd:.3f} s" if fd is not None else
              "  first visible content delta : NONE (empty stream!)")
        print(f"  wall                        : {r['wall_s']:.2f} s, chunks={r['n_chunks']}, "
              f"finish={r['finish_reason']}")
        print(f"  timings                     : prompt_n={pn} prompt_ms={prefill} "
              f"predicted_n={t.get('predicted_n')} predicted_ms={t.get('predicted_ms')}")
        print(f"  think tags present          : {'<think>' in r['content']} / "
              f"{'</think>' in r['content']}")
        print(f"  think text (first 160)      : {r['think'][:160]!r}")
        print(f"  answer (first 200)          : {r['answer'][:200]!r}")
        if args.log:
            try:
                lines = [ln for ln in open(args.log, encoding="utf-8", errors="replace")
                         if ln.startswith("[req]")]
                if lines:
                    last = lines[-1].rstrip()
                    m = re.search(r"prefill ([0-9.]+) \[new (\d+), cached (\d+)\]", last)
                    if m:
                        print(f"  server prefill              : {m.group(1)} ms "
                              f"(new={m.group(2)} tokens, cached={m.group(3)} tokens)")
                    else:
                        print(f"  server log                  : {last}")
            except FileNotFoundError:
                pass
        # next turn: assistant answer as history (webui excludes the thought before
        # sending when excludeThoughtOnReq=true, the default)
        messages.append({"role": "assistant", "content": r["answer"]})
        messages.append({"role": "user", "content": "再补充一句就好。"})

    print("\n== summary ==")
    for i, r in enumerate(rows, 1):
        t = r["timings"] or {}
        print(f"  turn {i}: first_delta={r['first_delta_s']:.3f}s "
              f"prefill_n={t.get('prompt_n')} prefill_ms={t.get('prompt_ms')} "
              f"answer_len={len(r['answer'])}")
    # hard checks for CI use
    bad = [i for i, r in enumerate(rows, 1)
           if r["first_delta_s"] is None or "<think>" not in r["content"]
           or (args.thinking and not r["think"])]
    if bad:
        print(f"\nFAIL: turns without visible thinking content: {bad}")
        return 1
    print("\nWEBUI PROBE OK (first deltas visible, thinking wrapped in content)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
