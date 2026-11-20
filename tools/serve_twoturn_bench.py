#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-turn "real conversation" benchmark for `engine serve`.

Matches the llama.cpp comparison run prepared by the lead:

  turn 1: 你好
  turn 2: 帮我写1000字小说            (same conversation: messages accumulate)

Sampling params are the user's real llama.cpp settings:
  temperature 0.7, top_p 0.8, top_k 20, min_p 0.0

For every turn this records (streaming request, so the first token is timed):
  wall_s, prompt_tokens, prefill_ms, decode_ms, gen_tokens, ttfb_ms,
  decode_tok_s, finish_reason, answer chars / CJK chars, reasoning chars,
  repetition metric (max identical consecutive sentences in the answer).

It also snapshots /props (model, n_ctx, prefix_cache stats) and writes a JSON +
a markdown table that can be pasted into docs/.

Usage:
  python tools/serve_twoturn_bench.py --port 7030 --tag "35b_q8kv_mtp-off" \
      --out build/logs/bench_35b.json [--max-tokens 2048]
"""
from __future__ import annotations

import argparse
import http.client
import os
import json
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TURNS = ["你好", "帮我写1000字小说"]


def post_json(port, path, payload, timeout=1800):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(payload, ensure_ascii=False)
    t0 = time.perf_counter()
    conn.request("POST", path, body=body.encode("utf-8"),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    dt = (time.perf_counter() - t0)
    conn.close()
    return resp.status, json.loads(raw.decode("utf-8", "replace")), dt


def stream_chat(port, payload, timeout=1800):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(payload, ensure_ascii=False)
    t0 = time.perf_counter()
    ttfb_any = None
    ttfb_content = None
    answer, reasoning = [], []
    timings, usage = None, None
    finish = None
    n_chunks = 0
    conn.request("POST", "/v1/chat/completions", body=body.encode("utf-8"),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    while True:
        line = resp.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace").strip()
        if not text.startswith("data:"):
            continue
        data = text[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        n_chunks += 1
        choices = obj.get("choices") or []
        if choices:
            ch = choices[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            d = ch.get("delta") or {}
            c = d.get("content") or ""
            r = d.get("reasoning_content") or d.get("reasoning") or ""
            if (c or r) and ttfb_any is None:
                ttfb_any = (time.perf_counter() - t0) * 1000.0
            if c:
                if ttfb_content is None:
                    ttfb_content = (time.perf_counter() - t0) * 1000.0
                answer.append(c)
            if r:
                reasoning.append(r)
        if "usage" in obj:
            usage = obj["usage"]
        if "timings" in obj:
            timings = obj["timings"]
    total = time.perf_counter() - t0
    conn.close()
    return {
        "status": resp.status,
        "wall_s": total,
        "ttfb_ms": ttfb_any,
        "first_content_ms": ttfb_content,
        "answer": "".join(answer),
        "reasoning": "".join(reasoning),
        "timings": timings or {},
        "usage": usage or {},
        "finish_reason": finish,
        "chunks": n_chunks,
    }


CJK = re.compile(r"[\u4e00-\u9fff]")


def quality(text: str):
    cjk = len(CJK.findall(text))
    sents = [s.strip() for s in re.split(r"(?<=[。！？.!?\n])", text) if len(s.strip()) >= 8]
    counts: dict[str, int] = {}
    for s in sents:
        counts[s] = counts.get(s, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    return {
        "chars": len(text),
        "cjk_chars": cjk,
        "sentences": len(sents),
        "max_sentence_repeat": top[0][1] if top else 0,
        "top_repeated": [(c, s[:40]) for s, c in top],
        "looks_truncated": (text and text.rstrip()[-1] not in "。！？.!?』”\"）)"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7030)
    ap.add_argument("--tag", default="engine")
    ap.add_argument("--out", default="build/logs/bench_twoturn.json")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    args = ap.parse_args()

    result = {
        "tag": args.tag,
        "port": args.port,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "top_k": args.top_k, "min_p": args.min_p,
                     "max_tokens": args.max_tokens},
        "turns": [],
    }
    try:
        _, props, _ = post_json(args.port, "/props", {})
        result["props"] = {
            "model_name": props.get("model_name"),
            "n_ctx": props.get("n_ctx"),
            "arch": props.get("arch"),
            "n_layer": props.get("n_layer"),
            "mtp_layers": props.get("mtp_layers"),
            "prefix_cache": props.get("prefix_cache"),
            "default_generation_settings": props.get("default_generation_settings"),
            "think_budget": props.get("think_budget"),
            "repeat_guard": props.get("repeat_guard"),
        }
    except Exception as e:  # noqa: BLE001
        result["props_error"] = str(e)

    messages = []
    for i, text in enumerate(TURNS):
        messages.append({"role": "user", "content": text})
        body = {
            "messages": messages,
            "stream": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_tokens": args.max_tokens,
        }
        r = stream_chat(args.port, body)
        tm = r["timings"]
        prompt_n = tm.get("prompt_n") or r["usage"].get("prompt_tokens")
        prefill_ms = tm.get("prefill_ms", tm.get("prompt_ms"))
        decode_ms = tm.get("decode_ms", tm.get("predicted_ms"))
        gen_n = tm.get("predicted_n") or r["usage"].get("completion_tokens")
        decode_tps = (gen_n / (decode_ms / 1000.0)) if (gen_n and decode_ms) else None
        raw_content = r["answer"]
        shown = raw_content
        if "<think>" in shown:
            j = shown.find("</think>")
            shown = shown[j + len("</think>"):] if j >= 0 else shown
        shown = shown.lstrip()
        qa = quality(shown)
        qr = quality(r["reasoning"]) if r["reasoning"] else None
        row = {
            "turn": i + 1,
            "user": text,
            "wall_s": round(r["wall_s"], 3),
            "ttfb_ms": None if r["ttfb_ms"] is None else round(r["ttfb_ms"], 1),
            "first_content_ms": (None if r["first_content_ms"] is None
                                 else round(r["first_content_ms"], 1)),
            "prompt_tokens": prompt_n,
            "prefill_ms": None if prefill_ms is None else round(prefill_ms, 1),
            "decode_ms": None if decode_ms is None else round(decode_ms, 1),
            "gen_tokens": gen_n,
            "decode_tok_s": None if decode_tps is None else round(decode_tps, 2),
            "finish_reason": r["finish_reason"],
            "cache_hit_tokens": tm.get("cache_hit_tokens"),
            "prefilled_tokens": tm.get("prefilled_tokens"),
            "think_budget_hit": tm.get("think_budget_hit"),
            "repeat_guard_hit": tm.get("repeat_guard_hit"),
            "answer": qa,
            "answer_text": shown,
            "raw_content_chars": len(raw_content),
            "reasoning": qr,
        }
        result["turns"].append(row)
        print(f"\n=== turn {i + 1}: {text!r}")
        print(f"  status={r['status']} wall={row['wall_s']}s ttfb={row['ttfb_ms']}ms "
              f"first_content={row['first_content_ms']}ms")
        print(f"  prompt={prompt_n} tok prefill={row['prefill_ms']}ms "
              f"decode={row['decode_ms']}ms gen={gen_n} tok "
              f"decode_speed={row['decode_tok_s']} tok/s finish={r['finish_reason']}")
        print(f"  cache: hit={row['cache_hit_tokens']} prefilled={row['prefilled_tokens']}")
        print(f"  answer: chars={qa['chars']} cjk={qa['cjk_chars']} sents={qa['sentences']} "
              f"max_repeat={qa['max_sentence_repeat']} truncated={qa['looks_truncated']}")
        print(f"  reasoning: {qr['chars'] if qr else 0} chars")
        print(f"  answer(head): {shown[:160]!r}")
        result.setdefault("_turn2_text", None)
        # webui default: only the answer is kept in history
        messages.append({"role": "assistant", "content": r["answer"]})

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    os.makedirs("build/compare", exist_ok=True)
    with open(f"build/compare/our-turn2-{args.tag}.txt", "w", encoding="utf-8") as f:
        t2 = result["turns"][1]
        f.write("USER: " + t2["user"] + "\n\n")
        f.write("REASONING (" + str((t2["reasoning"] or {}).get("chars")) + " chars):\n")
        f.write((t2.get("_reasoning_text") or "") + "\n\n--- ANSWER (" +
                str(t2["answer"]["cjk_chars"]) + " CJK chars) ---\n")
        f.write(t2["answer_text"])
    with open("build/compare/our-turn2.txt", "w", encoding="utf-8") as f:
        t2 = result["turns"][1]
        f.write(t2["answer_text"])
    print(f"\nsaved: {args.out}")

    # markdown table for docs
    print("\n| 轮次 | wall (s) | prompt tok | prefill ms | decode ms | gen tok | "
          "首 token ms | decode tok/s | answer CJK | max repeat |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in result["turns"]:
        print(f"| {t['turn']} | {t['wall_s']} | {t['prompt_tokens']} | {t['prefill_ms']} | "
              f"{t['decode_ms']} | {t['gen_tokens']} | {t['ttfb_ms']} | {t['decode_tok_s']} | "
              f"{t['answer']['cjk_chars']} | {t['answer']['max_sentence_repeat']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
