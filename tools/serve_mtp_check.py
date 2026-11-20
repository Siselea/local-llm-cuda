# -*- coding: utf-8 -*-
"""MTP serve verification: greedy token-id equality vs plain decoding + tok/s.

Two runs (separate server startups):
  python tools/serve_mtp_check.py --port 7030 --tag off --save build/logs/mtp_off.json
  ... restart server with --mtp --mtp-k 2 ...
  python tools/serve_mtp_check.py --port 7030 --tag on  --save build/logs/mtp_on.json \
      --baseline build/logs/mtp_off.json

The script sends the same greedy chat request (temperature=0, enable_thinking=false)
and compares generated_token_ids token-for-token; it also reports wall/prefill/decode
and the engine's tok/s.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT = "用一句话介绍杭州。"  # overridden by --prompt
MAX_TOKENS = 96


def chat(port, payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    wall = time.perf_counter() - t0
    ch = d["choices"][0]
    tm = d.get("timings", {})
    return {"wall_s": round(wall, 3), "ids": d.get("generated_token_ids") or [],
            "finish": ch.get("finish_reason"), "timings": tm,
            "content": ch["message"].get("content") or "",
            "reasoning": ch["message"].get("reasoning_content") or ""}


def props(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=30) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    return {"mtp": d.get("mtp"), "mtp_layers": d.get("mtp_layers"),
            "n_ctx": d.get("n_ctx"), "model": d.get("model_name")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7030)
    ap.add_argument("--tag", default="off")
    ap.add_argument("--save", default="build/logs/mtp_check.json")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    p = props(args.port)
    payload = {"messages": [{"role": "user", "content": args.prompt}], "stream": False,
               "temperature": 0, "max_tokens": args.max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    r = chat(args.port, payload)
    tm = r["timings"]
    out = {"tag": args.tag, "props": p, "prompt": PROMPT,
           "max_tokens": args.max_tokens,
           "wall_s": r["wall_s"], "finish": r["finish"],
           "prompt_n": tm.get("prompt_n"), "prefill_ms": tm.get("prefill_ms"),
           "decode_ms": tm.get("decode_ms"),
           "predicted_n": tm.get("predicted_n") or len(r["ids"]),
           "ids": r["ids"],
           "decode_tok_s": (round((tm.get("predicted_n") or len(r["ids"])) /
                                  ((tm.get("decode_ms") or 1) / 1000.0), 2)),
           "content": r["content"]}
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[{args.tag}] mtp={p['mtp']} wall={out['wall_s']}s gen={out['predicted_n']} "
          f"prefill={out['prefill_ms']}ms decode={out['decode_ms']}ms "
          f"{out['decode_tok_s']} tok/s finish={out['finish']}")
    print("content:", repr(out["content"][:100]))

    if args.baseline:
        base = json.load(open(args.baseline, encoding="utf-8"))
        same = base["ids"] == out["ids"]
        print(f"baseline={base['tag']} gen={base['predicted_n']} "
              f"{base['decode_tok_s']} tok/s")
        print(f"token-for-token identical: {same}  ({len(out['ids'])} tokens)")
        if not same:
            m = next((i for i in range(min(len(base['ids']), len(out['ids'])))
                      if base['ids'][i] != out['ids'][i]), None)
            print(f"  first mismatch at {m}: off={base['ids'][m:m+5] if m is not None else None} "
                  f"on={out['ids'][m:m+5] if m is not None else None}")
        out["baseline_same"] = same
        out["baseline"] = base
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
