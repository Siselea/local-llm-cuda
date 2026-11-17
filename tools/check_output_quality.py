#!/usr/bin/env python
"""Output-quality gate for the HTTP service tests (pure CPU, no model).

The service e2e tests must not call a 200 response "PASS" when the model output
is degenerate (prompt copying / symbol soup / repetition).  This checker turns
"HTTP 200" into a real acceptance decision:

  python tools/check_output_quality.py --json resp.json [--expect-cjk|--expect-ascii]
        [--prompt-file rendered_prompt.txt] [--min-chars 1]
  python tools/check_output_quality.py --sse stream.sse --expect-cjk

Exit code 0 = output looks healthy, 1 = degenerate/empty (FAIL), 2 = usage error.
All diagnostics are ASCII so the GBK console can never corrupt them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load_json(path: Path):
    d = json.load(open(path, encoding="utf-8"))
    if "choices" not in d or not d["choices"]:
        return "", "", {"error": d.get("error")}
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if not content and not reasoning and "text" in ch:  # /v1/completions
        content = ch["text"]
    return content, reasoning, {"finish_reason": ch.get("finish_reason")}


def load_sse(path: Path):
    content, reasoning, chunks, done, finish = [], [], 0, False, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            done = True
            continue
        try:
            j = json.loads(payload)
        except Exception:
            continue
        chunks += 1
        ch = (j.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        delta = ch.get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if j.get("content"):  # llama.cpp style /completion stream
            content.append(j["content"])
    return "".join(content), "".join(reasoning), {"chunks": chunks, "done": done,
                                                "finish_reason": finish}


CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x3040, 0x30FF),
              (0xAC00, 0xD7AF))


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in CJK_RANGES)


def has_cjk(text: str) -> int:
    return sum(1 for c in text if is_cjk(c))


def analyse(text: str) -> dict:
    chars = [c for c in text if not c.isspace()]
    n = len(chars)
    cjk = sum(1 for c in chars if is_cjk(c))
    latin = sum(1 for c in chars if c.isascii() and c.isalpha())
    digit = sum(1 for c in chars if c.isdigit())
    punct = sum(1 for c in chars if ord(c) < 128 and not c.isalnum() and c not in "\t")
    # repetition: most common 8-char n-gram + distinct 6-gram ratio
    grams = Counter(text[i:i + 8] for i in range(max(0, len(text) - 8)))
    top_gram, top_count = (grams.most_common(1)[0] if grams else ("", 0))
    g6 = [text[i:i + 6] for i in range(max(0, len(text) - 6))]
    uniq6 = (len(set(g6)) / len(g6)) if g6 else 1.0
    meaningful = cjk + latin
    return {
        "len": len(text),
        "meaningful_ratio": round(meaningful / n, 3) if n else 0.0,
        "nonspace": n,
        "cjk": cjk,
        "cjk_ratio": round(cjk / n, 3) if n else 0.0,
        "latin": latin,
        "latin_ratio": round(latin / n, 3) if n else 0.0,
        "digit": digit,
        "punct": punct,
        "punct_ratio": round(punct / n, 3) if n else 0.0,
        "max_8gram_repeat": top_count,
        "top_8gram": top_gram,
        "uniq_6gram_ratio": round(uniq6, 3),
    }


def longest_common_chunk(text: str, prompt: str, cap: int = 200) -> int:
    """Longest common substring length between output and prompt (capped O(n*m))."""
    text = text[:2000]
    prompt = prompt[:4000]
    best = 0
    # binary search on length would be better, but outputs are short here
    for i in range(len(text)):
        if len(text) - i <= best:
            break
        for L in range(best + 1, min(len(text) - i, cap) + 1):
            if text[i:i + L] in prompt:
                best = L
            else:
                break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="response JSON from /v1/chat/completions, /v1/completions or /completion")
    src.add_argument("--sse", help="SSE stream capture")
    ap.add_argument("--expect-cjk", action="store_true", help="answer must contain Chinese text")
    ap.add_argument("--expect-ascii", action="store_true", help="answer must be English/ASCII text")
    ap.add_argument("--prompt-file", default="", help="rendered prompt text (prompt-copy detection)")
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--max-punct-ratio", type=float, default=0.45)
    ap.add_argument("--max-repeat", type=int, default=3)
    ap.add_argument("--min-cjk-ratio", type=float, default=0.30)
    ap.add_argument("--min-latin-ratio", type=float, default=0.40)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.json:
        content, reasoning, extra = load_json(Path(args.json))
    else:
        content, reasoning, extra = load_sse(Path(args.sse))
    text = content + reasoning
    m = analyse(text)

    problems = []
    if m["nonspace"] < args.min_chars:
        problems.append(f"empty/too-short output (nonspace={m['nonspace']})")
    if args.expect_cjk and m["nonspace"] > 0 and m["cjk_ratio"] < args.min_cjk_ratio:
        problems.append(f"expected Chinese, cjk_ratio={m['cjk_ratio']:.3f} < {args.min_cjk_ratio}")
    if args.expect_ascii and m["nonspace"] > 0 and m["latin_ratio"] < args.min_latin_ratio:
        problems.append(f"expected ASCII text, latin_ratio={m['latin_ratio']:.3f} < {args.min_latin_ratio}")
    if m["nonspace"] >= 16 and m["punct_ratio"] > args.max_punct_ratio:
        problems.append(f"symbol soup: punct_ratio={m['punct_ratio']:.3f} > {args.max_punct_ratio}")
    if m["nonspace"] >= 16 and m["meaningful_ratio"] < 0.30:
        problems.append(f"symbol soup: letters+CJK ratio={m['meaningful_ratio']:.3f} < 0.30")
    if m["nonspace"] >= 32 and m["max_8gram_repeat"] > args.max_repeat:
        problems.append(f"repetition: 8-gram {m['top_8gram']!r} x{m['max_8gram_repeat']} > {args.max_repeat}")
    if m["nonspace"] >= 24 and m["uniq_6gram_ratio"] < 0.50:
        problems.append(f"repetition: distinct 6-gram ratio={m['uniq_6gram_ratio']:.3f} < 0.50")
    if args.prompt_file:
        try:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8", errors="replace")
        except Exception:
            prompt = ""
        if prompt:
            lcs = longest_common_chunk(text, prompt)
            # a verbatim 40+ char copy of the rendered prompt is degeneration,
            # not an answer (short echoes like "OK" or a quoted word are fine)
            if lcs >= 40:
                problems.append(f"prompt copying: {lcs}-char verbatim prompt chunk in output")

    print(json.dumps({"metrics": m, "extra": extra,
                      "problems": problems}, ensure_ascii=True))
    if problems:
        print("QUALITY FAIL: " + "; ".join(problems))
        return 1
    print("QUALITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
