# -*- coding: utf-8 -*-
"""Real-machine checks for the serve stability fixes.

Runs against a live `engine serve`:
  A) two-turn conversation (你好 -> 帮我写1000字小说) at temp 0.7, expecting the
     repetition guard to stop the degenerate loop (finish_reason=repetition,
     repeat_guard_hit=True) instead of running to the token cap;
  B) three normal prompts that must NOT be stopped (finish=stop/length,
     repeat_guard_hit=False, no char/sentence loops);
  C) a thinking question: think budget hit -> forced close -> clean thinking-off
     re-answer must be fluent, with no planning text residue.

Usage: python tools/serve_stability_check.py --port 7030 --tag 35b [--json out.json]
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CJK = re.compile(r"[\u4e00-\u9fff]")


def stream(port, messages, *, temperature=0.7, max_tokens=2048, top_p=0.8, top_k=20,
           min_p=0.0, timeout=1800):
    body = {"messages": messages, "stream": True, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "min_p": min_p,
            "max_tokens": max_tokens}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    ttfb = None
    content, reasoning = [], []
    timings, finish = {}, None
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body, ensure_ascii=False).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    while True:
        line = resp.readline()
        if not line:
            break
        t = line.decode("utf-8", "replace").strip()
        if not t.startswith("data:"):
            continue
        d = t[5:].strip()
        if d == "[DONE]":
            break
        try:
            obj = json.loads(d)
        except json.JSONDecodeError:
            continue
        ch = (obj.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        delta = ch.get("delta") or {}
        c = delta.get("content") or ""
        r = delta.get("reasoning_content") or ""
        if (c or r) and ttfb is None:
            ttfb = (time.perf_counter() - t0) * 1000
        if c:
            content.append(c)
        if r:
            reasoning.append(r)
        if obj.get("timings"):
            timings = obj["timings"]
    wall = time.perf_counter() - t0
    conn.close()
    full = "".join(content)
    think = "".join(reasoning)
    shown = full
    if "<think>" in shown:
        j = shown.find("</think>")
        shown = shown[j + len("</think>"):] if j >= 0 else shown
    shown = shown.strip()
    return {"wall_s": round(wall, 2), "ttfb_ms": round(ttfb, 1) if ttfb else None,
            "finish": finish, "timings": timings, "raw": full,
            "reasoning": think, "answer": shown}


def sentence_loop(text):
    sents = [s.strip() for s in re.split(r"(?<=[。！？.!?\n])", text) if len(s.strip()) >= 8]
    counts = {}
    for s in sents:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values()) if counts else 0


def char_loop(text, maxp=96):
    """Longest run of 'same unit x5' anywhere in text (same idea as the guard)."""
    n = len(text)
    for p in range(4, maxp + 1):
        for start in range(0, n - 5 * p + 1, max(1, p // 2)):
            seg = text[start:start + p]
            if text[start:start + 5 * p] == seg * 5:
                return p
    return 0


PLANNING_MARKERS = ["Thinking Process", "Analyze the Request", "Analyze the User",
                    "Draft the", "Refine the", "**Analyze", "Let's produce",
                    "Let's write", "Final Selection", "Critique"]


def planning_residue(answer):
    hits = [m for m in PLANNING_MARKERS if m in answer]
    numbered = len(re.findall(r"(?m)^\s*\d+\.\s+\*\*", answer))
    return hits, numbered


def run_case(port, name, messages, expect_guard_stop, max_tokens=2048):
    r = stream(port, messages, temperature=0.7, max_tokens=max_tokens)
    tm = r["timings"]
    ans = r["answer"]
    sl, cl = sentence_loop(ans), char_loop(ans)
    markers, numbered = planning_residue(ans)
    r.update({"name": name, "prompt_tokens": tm.get("prompt_n"),
              "prefill_ms": tm.get("prefill_ms"), "decode_ms": tm.get("decode_ms"),
              "gen_tokens": tm.get("predicted_n"),
              "repeat_guard_hit": tm.get("repeat_guard_hit"),
              "think_budget_hit": tm.get("think_budget_hit"),
              "sentence_loop": sl, "char_loop_p": cl,
              "answer_cjk": len(CJK.findall(ans)), "answer_chars": len(ans),
              "planning_markers": markers, "numbered_plan_lines": numbered})
    print(f"\n=== {name}")
    print(f"  wall={r['wall_s']}s ttfb={r['ttfb_ms']}ms prompt={r['prompt_tokens']} "
          f"prefill={r['prefill_ms']}ms decode={r['decode_ms']}ms gen={r['gen_tokens']} "
          f"finish={r['finish']}")
    print(f"  guard_hit={r['repeat_guard_hit']} think_budget_hit={r['think_budget_hit']} "
          f"sentence_loop={sl} char_loop_p={cl} answer={r['answer_cjk']} CJK")
    print(f"  planning markers={markers} numbered_plan_lines={numbered}")
    print(f"  answer(head)={ans[:180]!r}")
    if expect_guard_stop:
        ok = (r["repeat_guard_hit"] is True and r["finish"] == "repetition"
              and sl < 5 and cl == 0)
    else:
        ok = (not r["repeat_guard_hit"]) and r["finish"] in ("stop", "length") and sl < 3 and cl == 0
    r["ok"] = bool(ok)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7030)
    ap.add_argument("--tag", default="35b")
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-think", action="store_true")
    args = ap.parse_args()
    port = args.port
    results = []

    # A) the two-turn real user conversation that used to loop
    msgs = [{"role": "user", "content": "你好"}]
    r1 = stream(port, msgs, temperature=0.7, max_tokens=512)
    print(f"=== turn1 你好: wall={r1['wall_s']}s finish={r1['finish']} "
          f"answer={r1['answer'][:60]!r}")
    msgs.append({"role": "assistant", "content": r1["answer"]})
    r2 = run_case(port, "turn2 帮我写1000字小说", msgs + [{"role": "user", "content": "帮我写1000字小说"}],
                  expect_guard_stop=True)
    with open(f"build/compare/live-turn2-{args.tag}.txt", "w", encoding="utf-8") as f:
        f.write(r2["answer"])
    results.append({"turn1": {k: r1[k] for k in ("wall_s", "finish", "answer")}})
    results.append(r2)

    # B) controls: normal prompts must not be stopped
    controls = [
        ("control 你好", [{"role": "user", "content": "你好"}], 256),
        ("control 介绍杭州", [{"role": "user", "content": "请用三句话介绍杭州。"}], 512),
        ("control 写 Python 代码", [{"role": "user", "content":
                                    "用 Python 写一个函数，判断一个整数是否为素数，并给一个例子。"}], 768),
    ]
    for name, messages, mt in controls:
        results.append(run_case(port, name, messages, expect_guard_stop=False, max_tokens=mt))

    # C) think budget: question that needs thinking -> budget hit -> clean re-answer
    if not args.skip_think:
        r = run_case(port, "think-budget question",
                     [{"role": "user", "content": "为什么天空是蓝色的？请用通俗的方式解释。"}],
                     expect_guard_stop=False, max_tokens=1024)
        clean = (r["think_budget_hit"] is True and r["answer_cjk"] >= 30 and
                 not r["planning_markers"] and r["char_loop_p"] == 0 and
                 r["sentence_loop"] < 3)
        print(f"  think-budget clean re-answer: {'PASS' if clean else 'FAIL'}")
        r["ok"] = r["ok"] or clean
        results.append(r)

    ok = all(x.get("ok", False) for x in results)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nsaved {args.json}")
    print("\nSTABILITY CHECKS " + ("GREEN" if ok else "RED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
