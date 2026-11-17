#!/usr/bin/env python3
"""engine serve end-to-end benchmark (OpenAI-compatible HTTP API).

Three request shapes, each measured N times (default 3, median reported):
  a) non-stream  POST /v1/chat/completions  max_tokens=256
  b) streaming   same + "stream": true      -> TTFB (first content delta) + decode tok/s
  c) long prompt ~1500 tokens               -> prefill tok/s (max_tokens=8)

The engine returns an extra `timings` object ({prefill_ms, decode_ms, tokens_per_second})
in the non-streaming body and in the final SSE usage chunk, so the decode tok/s below is
the engine's own measured rate, not inferred from the wall clock.  Wall clock is reported
too so HTTP/JSON overhead stays visible.

Usage:
  python tools/serve_bench.py --port 7030 --tag 9b --reps 3
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import time

QUESTION = "用三句话介绍一下杭州，然后说说西湖为什么有名。"
BASE_PARA = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. Jinxed wizards pluck ivy from the big quilt. "
    "We promptly judged antique ivory buckles for the next prize. "
)


def post_json(port, path, payload, timeout=900):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(payload, ensure_ascii=False)
    t0 = time.perf_counter()
    conn.request("POST", path, body=body.encode("utf-8"),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    dt = (time.perf_counter() - t0) * 1000.0
    status = resp.status
    conn.close()
    return status, json.loads(raw.decode("utf-8", "replace")), dt


def stream_chat(port, payload, timeout=900):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(payload, ensure_ascii=False)
    t0 = time.perf_counter()
    ttfb = None
    usage = timings = None
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
            delta = choices[0].get("delta") or {}
            if ttfb is None and (delta.get("content") or delta.get("reasoning_content")
                                 or delta.get("reasoning")):
                ttfb = (time.perf_counter() - t0) * 1000.0
        if "usage" in obj:
            usage = obj["usage"]
        if "timings" in obj:
            timings = obj["timings"]
    total = (time.perf_counter() - t0) * 1000.0
    conn.close()
    return dict(status=resp.status, ttfb_ms=ttfb, total_ms=total, usage=usage,
                timings=timings, chunks=n_chunks)


def token_count(port, text):
    status, obj, _ = post_json(port, "/tokenize", {"content": text})
    return len(obj.get("tokens") or [])


def build_long_prompt(port, target):
    """Repeat a paragraph until /tokenize says we are close to `target` tokens."""
    reps = max(1, target // 60)
    for _ in range(8):
        text = ("Routine log line for the prefill benchmark. " * 0) + (BASE_PARA * reps)
        n = token_count(port, text)
        if abs(n - target) <= 40:
            return text, n
        reps = max(1, round(reps * target / max(n, 1)))
    return text, n


def median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7030)
    ap.add_argument("--tag", default="serve")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--long-target", type=int, default=1500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tc = http.client.HTTPConnection("127.0.0.1", args.port, timeout=30)
    tc.request("GET", "/health")
    health = tc.getresponse()
    health_raw = health.read()
    tc.close()
    if health.status != 200:
        print(f"ERROR health status {health.status}: {health_raw[:200]!r}")
        return 2
    print(f"[health] {health_raw.decode('utf-8', 'replace').strip()[:200]}")

    long_text, long_n = build_long_prompt(args.port, args.long_target)
    print(f"[long prompt] tokenized prompt_n={long_n} chars={len(long_text)}")

    chat_base = {
        "messages": [{"role": "user", "content": QUESTION}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    stream_payload = dict(chat_base, stream=True)
    long_payload = {
        "messages": [{"role": "user", "content": long_text}],
        "max_tokens": 8,
        "temperature": 0,
    }

    rows = {"nonstream": [], "stream": [], "longprefill": []}
    for rep in range(1, args.reps + 1):
        # a) non-streaming chat
        st, obj, wall = post_json(args.port, "/v1/chat/completions", chat_base)
        tim = obj.get("timings") or {}
        use = obj.get("usage") or {}
        row = dict(rep=rep, status=st, wall_ms=wall,
                   prompt_n=use.get("prompt_tokens"), gen_n=use.get("completion_tokens"),
                   prefill_ms=tim.get("prefill_ms"), decode_ms=tim.get("decode_ms"),
                   decode_tps=tim.get("tokens_per_second"),
                   wall_tps=(use.get("completion_tokens") or 0) * 1000.0 / wall if wall else None)
        rows["nonstream"].append(row)
        print(f"[nonstream r{rep}] status={st} prompt={row['prompt_n']} gen={row['gen_n']} "
              f"wall={fmt(wall,1)} ms prefill={fmt(row['prefill_ms'],1)} "
              f"decode={fmt(row['decode_ms'],1)} ms decode={fmt(row['decode_tps'],2)} tok/s "
              f"wall={fmt(row['wall_tps'],2)} tok/s")

        # b) streaming chat
        sr = stream_chat(args.port, stream_payload)
        tim = sr["timings"] or {}
        use = sr["usage"] or {}
        d_tps = tim.get("tokens_per_second")
        if d_tps is None and tim.get("predicted_ms"):
            d_tps = tim["predicted_n"] * 1000.0 / tim["predicted_ms"]
        row = dict(rep=rep, status=sr["status"], ttfb_ms=sr["ttfb_ms"],
                   total_ms=sr["total_ms"], n_chunks=sr["chunks"],
                   prompt_n=use.get("prompt_tokens") or tim.get("prompt_n"),
                   gen_n=use.get("completion_tokens") or tim.get("predicted_n"),
                   prefill_ms=tim.get("prefill_ms") or tim.get("prompt_ms"),
                   decode_ms=tim.get("decode_ms") or tim.get("predicted_ms"),
                   decode_tps=d_tps,
                   wall_tps=(use.get("completion_tokens") or 0) * 1000.0 / sr["total_ms"]
                   if sr["total_ms"] else None)
        rows["stream"].append(row)
        print(f"[stream    r{rep}] status={sr['status']} chunks={sr['chunks']} "
              f"ttfb={fmt(sr['ttfb_ms'],1)} ms total={fmt(sr['total_ms'],1)} ms "
              f"gen={row['gen_n']} decode={fmt(row['decode_tps'],2)} tok/s "
              f"wall={fmt(row['wall_tps'],2)} tok/s")

        # c) long-prompt prefill
        st, obj, wall = post_json(args.port, "/v1/chat/completions", long_payload)
        tim = obj.get("timings") or {}
        use = obj.get("usage") or {}
        pn = use.get("prompt_tokens")
        pm = tim.get("prefill_ms")
        row = dict(rep=rep, status=st, wall_ms=wall, prompt_n=pn,
                   prefill_ms=pm,
                   prefill_tps=(pn * 1000.0 / pm) if (pn and pm) else None,
                   gen_n=use.get("completion_tokens"), decode_tps=tim.get("tokens_per_second"))
        rows["longprefill"].append(row)
        print(f"[longpref  r{rep}] status={st} prompt={pn} prefill={fmt(pm,1)} ms "
              f"= {fmt(row['prefill_tps'],1)} tok/s wall={fmt(wall,1)} ms")

    def agg(key, field):
        return median([r.get(field) for r in rows[key]])

    summary = {
        "tag": args.tag,
        "port": args.port,
        "reps": args.reps,
        "nonstream": {
            "decode_tps_median": agg("nonstream", "decode_tps"),
            "wall_tps_median": agg("nonstream", "wall_tps"),
            "prefill_ms_median": agg("nonstream", "prefill_ms"),
            "decode_ms_median": agg("nonstream", "decode_ms"),
            "wall_ms_median": agg("nonstream", "wall_ms"),
            "gen_tokens": agg("nonstream", "gen_n"),
        },
        "stream": {
            "ttfb_ms_median": agg("stream", "ttfb_ms"),
            "decode_tps_median": agg("stream", "decode_tps"),
            "wall_tps_median": agg("stream", "wall_tps"),
            "total_ms_median": agg("stream", "total_ms"),
            "gen_tokens": agg("stream", "gen_n"),
        },
        "longprefill": {
            "prefill_tps_median": agg("longprefill", "prefill_tps"),
            "prefill_ms_median": agg("longprefill", "prefill_ms"),
            "prompt_n": agg("longprefill", "prompt_n"),
            "wall_ms_median": agg("longprefill", "wall_ms"),
        },
        "rows": rows,
    }
    print("\n== summary (median of %d) ==" % args.reps)
    print(f"non-stream : decode {fmt(summary['nonstream']['decode_tps_median'],2)} tok/s "
          f"(wall {fmt(summary['nonstream']['wall_tps_median'],2)}), "
          f"first-token(prefill) {fmt(summary['nonstream']['prefill_ms_median'],1)} ms, "
          f"wall {fmt(summary['nonstream']['wall_ms_median'],0)} ms")
    print(f"stream     : TTFB {fmt(summary['stream']['ttfb_ms_median'],1)} ms, "
          f"decode {fmt(summary['stream']['decode_tps_median'],2)} tok/s "
          f"(wall {fmt(summary['stream']['wall_tps_median'],2)}), "
          f"total {fmt(summary['stream']['total_ms_median'],0)} ms")
    print(f"long prompt: prefill {fmt(summary['longprefill']['prefill_tps_median'],1)} tok/s "
          f"({fmt(summary['longprefill']['prefill_ms_median'],1)} ms for "
          f"{summary['longprefill']['prompt_n']} tokens)")
    out = args.out or f"build/serve-bench-{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
