#!/usr/bin/env python
"""A/B check for the serve multi-turn prefix cache.

For a 5-turn synthetic conversation (each turn grows the history by ~45 tokens):
  * phase "off": /v1/chat/completions with cache_prompt=false  -> every request
    prefills the whole conversation from position 0 (the reference baseline);
  * phase "on" : same request with cache_prompt=true            -> only the new
    tokens are prefilled, the shared history prefix is reused (KV + hybrid SSM
    state snapshot).

Acceptance (the two hard requirements):
  1. generated_token_ids are IDENTICAL between the cached and the full-prefill
     request on every turn (greedy, temperature 0);
  2. on the last turn the cached prefill is close to "just the new tokens",
     i.e. far below the full-prefill time.

Usage:
  python tools/serve_prefix_cache_check.py --url http://127.0.0.1:7030 [--turns 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FILLERS = [
    "第一段记录讲述了项目的历史背景与目标设定。",
    "第二段记录列出了本周完成的三项主要工作与结果。",
    "第三段记录说明了测试环境、显卡型号和驱动的版本。",
    "第四段记录汇总了性能数据：吞吐、延迟以及显存占用。",
    "第五段记录描述了下一步的优化方向与验收标准。",
    "第六段记录保留了风险清单与回滚方案的具体步骤。",
]


FILLER_REPEAT = 2  # global, set from --filler-repeat


def canned_assistant(turn: int) -> str:
    # deterministic, *varied* history growth (~20 tokens per filler sentence);
    # varied text avoids the model degenerating on a repeated prompt, which would
    # make the A/B comparison meaningless.
    parts = []
    for j in range(FILLER_REPEAT):
        parts.append(f"第{turn}轮记录{j}号：{FILLERS[(turn * 7 + j) % len(FILLERS)]}")
    return "固定历史回答。" + "".join(parts)


def chat(url: str, messages, *, cache: bool, max_tokens: int) -> dict:
    body = {
        "messages": messages,
        "stream": False,
        "cache_prompt": cache,
        "temperature": 0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    ch = d["choices"][0]
    return {
        "ids": d.get("generated_token_ids") or [],
        "content": ch["message"].get("content") or "",
        "prefill_ms": d.get("timings", {}).get("prefill_ms", 0.0),
        "prompt_n": d.get("timings", {}).get("prompt_n", d.get("usage", {}).get("prompt_tokens", 0)),
        "cache_hit": d.get("timings", {}).get("cache_hit_tokens", 0),
        "prefilled": d.get("timings", {}).get("prefilled_tokens", 0),
        "finish": ch.get("finish_reason"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7030")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--filler-repeat", type=int, default=2,
                    help="history tokens added per turn is ~12x this (default 2 => ~45)")
    args = ap.parse_args()
    global FILLER_REPEAT
    FILLER_REPEAT = max(1, args.filler_repeat)

    rows = []
    messages = []
    token_ok = True
    for turn in range(1, args.turns + 1):
        user = f"这是第{turn}轮，请只回复：收到{turn}。"
        messages.append({"role": "user", "content": user})
        off = chat(args.url, messages, cache=False, max_tokens=args.max_tokens)
        on = chat(args.url, messages, cache=True, max_tokens=args.max_tokens)
        same = off["ids"] == on["ids"] and len(off["ids"]) > 0
        token_ok = token_ok and same
        speed = off["prefill_ms"] / on["prefill_ms"] if on["prefill_ms"] > 0 else 0.0
        rows.append({
            "turn": turn,
            "prompt_n": off["prompt_n"],
            "off_ms": off["prefill_ms"],
            "on_ms": on["prefill_ms"],
            "on_new": on["prefilled"],
            "on_cached": on["cache_hit"],
            "speedup": speed,
            "tokens_equal": same,
            "off_ids": off["ids"],
            "on_ids": on["ids"],
        })
        messages.append({"role": "assistant", "content": canned_assistant(turn)})

    print("== prefix-cache A/B (greedy, temperature 0) ==")
    print(f"{'turn':>4} {'prompt':>7} {'full_ms':>9} {'cached_ms':>10} "
          f"{'new':>5} {'reused':>7} {'speedup':>8} {'tokens':>8}")
    for r in rows:
        print(f"{r['turn']:>4} {r['prompt_n']:>7} {r['off_ms']:>9.0f} {r['on_ms']:>10.0f} "
              f"{r['on_new']:>5} {r['on_cached']:>7} {r['speedup']:>7.1f}x "
              f"{'SAME' if r['tokens_equal'] else 'DIFF':>8}")
        if not r["tokens_equal"]:
            print(f"      off ids: {r['off_ids']}")
            print(f"      on  ids: {r['on_ids']}")

    last = rows[-1]
    checks = [
        ("all turns token-identical to full prefill", token_ok),
        ("turn-1 has no cache hit (cold)", rows[0]["on_cached"] == 0),
        ("last turn reuses a history prefix", last["on_cached"] > 0),
        # the cache granularity is one prefill chunk (serve default 128 tokens), so the
        # cached prefill keeps the new tokens plus <chunk tokens of aligned history
        ("last turn prefill < 60% of full prefill",
         last["on_ms"] < 0.60 * last["off_ms"] if last["off_ms"] else False),
    ]
    print("\n== checks ==")
    fail = 0
    for name, ok in checks:
        print(f"  {name:<48} {'PASS' if ok else 'FAIL'}")
        fail += 0 if ok else 1
    print(f"\nlast turn: full prefill {last['off_ms']:.0f} ms ({last['prompt_n']} tokens) -> "
          f"cached prefill {last['on_ms']:.0f} ms ({last['on_new']} new tokens), "
          f"{last['speedup']:.1f}x faster")
    print("\nPREFIX CACHE GREEN" if fail == 0 else f"\nPREFIX CACHE FAILURES: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
