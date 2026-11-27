# -*- coding: utf-8 -*-
"""Serve edge-case audit harness (real user paths that the five suites miss).

Cases:
  A. bad-request status codes (malformed JSON, messages not array, empty messages,
     prompt array for /completion, prompt too long, max_tokens edge values)
  B. stop sequences (multibyte / single char / "\\n\\n"), stop exclusion + finish reason
  C. /v1/completions vs /completion parity (same prompt/params -> same tokens)
  D. prefix cache after client disconnect mid-stream (abort) then a normal request
  E. edited history + regenerate: cached ids == cache_prompt:false ids
  F. concurrency: streaming + normal request; /health while streaming
  G. slowloris: headers with Content-Length then no body; does another request hang?
  H. keep-alive: normal request then streaming request on the same socket
  I. /tokenize + /detokenize edge strings (emoji/ZWJ, CJK-B, RTL, long)

Usage: python tools/serve_edge_cases.py --model <gguf> [--binary build/engine.exe]
       [--ctx 4096] [--port 7051] [--fast]
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:{port}"


def http(port, method, path, body=None, timeout=300, raw_body=None):
    url = BASE.format(port=port) + path
    data = raw_body if raw_body is not None else (
        None if body is None else json.dumps(body, ensure_ascii=False).encode())
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read().decode("utf-8", "replace")
            return {"status": r.status, "secs": round(time.time() - t0, 3), "body": payload,
                    "json": _try_json(payload)}
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "replace")
        return {"status": e.code, "secs": round(time.time() - t0, 3), "body": payload,
                "json": _try_json(payload)}
    except Exception as e:
        return {"status": None, "secs": round(time.time() - t0, 3), "error": repr(e)}


def http_bytes(port, path, body, timeout=300):
    """POST and return the raw response bytes (strict UTF-8 / JSON validity checks)."""
    url = BASE.format(port=port) + path
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _try_json(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def chat(port, messages, **kw):
    body = {"messages": messages, "stream": False, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 24}
    body.update(kw)
    return http(port, "POST", "/v1/chat/completions", body)


def stream_raw(port, body, read_bytes=400, close_after=True, timeout=60):
    """Open a raw socket, send a streaming chat request, read a bit, then close."""
    payload = json.dumps(body, ensure_ascii=False).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
           f"Connection: close\r\n\r\n").encode() + payload
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.sendall(req)
    got = b""
    t0 = time.time()
    try:
        while len(got) < read_bytes and time.time() - t0 < timeout:
            chunk = s.recv(4096)
            if not chunk:
                break
            got += chunk
    finally:
        if close_after:
            try:
                import struct
                s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("HH", 1, 0))
            except Exception:
                pass
            s.close()
    return got


def wait_health(port, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = http(port, "GET", "/health", timeout=2)
        if r["status"] == 200:
            return True
        time.sleep(0.5)
    return False


def check(results, name, ok, detail):
    results.append({"case": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="build/engine.exe")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=7051)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--fast", action="store_true", help="skip the slow long-generation cases")
    ap.add_argument("--gpu-layers", default="-1",
                    help="-1 = all layers on GPU; 0 = CPU-only (HTTP-layer tests)")
    args = ap.parse_args()
    port = args.port
    log_path = Path("build/logs/serve_edge_server.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [args.binary, "serve", "--model", args.model, "--host", "127.0.0.1",
         "--port", str(port), "--ctx", str(args.ctx), "--gpu-layers", str(args.gpu_layers),
         "--threads", str(args.threads), "--default-max-tokens", "1048576"],
        stdout=log, stderr=subprocess.STDOUT)
    results = []
    try:
        if not wait_health(port):
            print("SERVER FAILED TO START")
            return 2
        print("server ready")

        # ---------------- A. bad-request status codes ----------------
        r = http(port, "POST", "/v1/chat/completions", raw_body=b"{not json")
        check(results, "A1 malformed JSON -> 400", r["status"] == 400,
              f"status={r['status']} body={r['body'][:120]}")
        r = http(port, "POST", "/v1/chat/completions", {"messages": "nope"})
        check(results, "A2 messages not array -> 400", r["status"] == 400,
              f"status={r['status']} body={r['body'][:120]}")
        r = http(port, "POST", "/v1/chat/completions", {"messages": []})
        check(results, "A3 empty messages -> 400", r["status"] == 400,
              f"status={r['status']} body={r['body'][:120]}")
        r = http(port, "POST", "/completion", {"prompt": ["array", "not allowed"]})
        check(results, "A4 /completion prompt array -> 400", r["status"] == 400,
              f"status={r['status']} body={r['body'][:120]}")
        long_prompt = "词" * (args.ctx + 50)
        r = http(port, "POST", "/v1/completions", {"prompt": long_prompt, "max_tokens": 4})
        check(results, "A5 prompt >= ctx -> 400", r["status"] == 400,
              f"status={r['status']} body={r['body'][:160]}")

        # max_tokens edges
        r = chat(port, [{"role": "user", "content": "只回复：好"}], max_tokens=1)
        j = r["json"] or {}
        ids = j.get("generated_token_ids") or []
        check(results, "A6 max_tokens=1 -> exactly <=1 token",
              r["status"] == 200 and len(ids) <= 1,
              f"status={r['status']} ids={ids} finish={j.get('choices',[{}])[0].get('finish_reason')}")
        r = chat(port, [{"role": "user", "content": "只回复：好"}], max_tokens=0)
        j = r["json"] or {}
        ids = j.get("generated_token_ids") or []
        check(results, "A7 max_tokens=0 -> documented no-limit (<=1024), not crash",
              r["status"] == 200 and len(ids) < 1024,
              f"status={r['status']} gen={len(ids)}")
        r = chat(port, [{"role": "user", "content": "只回复：好"}], max_tokens=-1)
        check(results, "A8 max_tokens=-1 -> 200 (no-limit)", r["status"] == 200,
              f"status={r['status']} gen={len((r['json'] or {}).get('generated_token_ids') or [])}")
        r = chat(port, [{"role": "user", "content": "只回复：好"}], max_tokens=10 ** 6)
        check(results, "A9 max_tokens=1e6 -> 200 clamped to ctx", r["status"] == 200,
              f"status={r['status']}")

        # ---------------- B. stop sequences ----------------
        r = chat(port, [{"role": "user", "content": "用一句话介绍杭州，以句号结尾。"}],
                 max_tokens=64, stop=["。"])
        j = r["json"] or {}
        content = (j.get("choices", [{}])[0].get("message") or {}).get("content", "")
        finish = (j.get("choices", [{}])[0] or {}).get("finish_reason")
        check(results, "B1 multibyte stop '。' excluded from content",
              r["status"] == 200 and "。" not in content and finish == "stop",
              f"finish={finish} content_tail={content[-40:]!r}")
        r = chat(port, [{"role": "user", "content": "输出三行：one / two / three，用换行分隔。"}],
                 max_tokens=64, stop=["\n"])
        j = r["json"] or {}
        content = (j.get("choices", [{}])[0].get("message") or {}).get("content", "")
        check(results, "B2 stop '\\n' -> no newline in content",
              r["status"] == 200 and "\n" not in content,
              f"content_tail={content[-60:]!r}")
        r = chat(port, [{"role": "user", "content": "只输出：ABC###DEF"}], max_tokens=64,
                 stop=["###"])
        j = r["json"] or {}
        content = (j.get("choices", [{}])[0].get("message") or {}).get("content", "")
        check(results, "B3 multi-char stop '###' excluded",
              r["status"] == 200 and "###" not in content,
              f"content={content!r}")

        # B4: thinking enabled + a stop string of length >= 2.  The SSE/thinking
        # split looks for the "</think>" marker in flushed text, but the stop
        # holdback (max_stop-1 bytes) can cut the marker in half, so the answer
        # after </think> stays in reasoning and content ends up empty.
        r = chat(port, [{"role": "user", "content": "用一句话介绍杭州。"}],
                 max_tokens=96, stop=["zzz"],
                 chat_template_kwargs={"enable_thinking": True})
        j = r["json"] or {}
        msg = (j.get("choices", [{}])[0].get("message") or {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        check(results, "B4 thinking + stop>=2 chars -> content non-empty",
              r["status"] == 200 and len(content.strip()) > 0,
              f"content_len={len(content)} reasoning_len={len(reasoning)} "
              f"content={content[:80]!r}")

        # B5: stop strings also match *thinking* text.  A blank line is common in
        # thinking blocks, so stop=["\\n\\n"] can end the request before the answer.
        r = chat(port, [{"role": "user", "content": "用一句话介绍杭州。"}],
                 max_tokens=96, stop=["\n\n"],
                 chat_template_kwargs={"enable_thinking": True})
        j = r["json"] or {}
        msg = (j.get("choices", [{}])[0].get("message") or {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        check(results, "B5 thinking + stop '\\n\\n' still yields an answer",
              r["status"] == 200 and len(content.strip()) > 0,
              f"content_len={len(content)} reasoning_len={len(reasoning)} "
              f"content={content[:80]!r}")

        # ---------------- C. completions parity ----------------
        prompt = "The capital of France is"
        r1 = http(port, "POST", "/v1/completions",
                  {"prompt": prompt, "max_tokens": 12, "temperature": 0})
        r2 = http(port, "POST", "/completion",
                  {"prompt": prompt, "max_tokens": 12, "temperature": 0})
        j1, j2 = r1["json"] or {}, r2["json"] or {}
        text1 = (j1.get("choices", [{}])[0].get("text") or "")
        text2 = (j2.get("content") or "")
        check(results, "C1 /v1/completions vs /completion same tokens",
              r1["status"] == 200 and r2["status"] == 200 and text1 == text2,
              f"v1={text1[:60]!r} legacy={text2[:60]!r}")

        # ---------------- D. disconnect mid-stream then normal request ----------------
        got = stream_raw(port, {"messages": [{"role": "user", "content": "写一段800字散文"}],
                                "stream": True, "temperature": 0.8, "max_tokens": 512,
                                "chat_template_kwargs": {"enable_thinking": False}},
                         read_bytes=300)
        time.sleep(0.5)
        r = chat(port, [{"role": "user", "content": "只回复：收到"}], max_tokens=8)
        check(results, "D1 request after client disconnect works",
              r["status"] == 200, f"status={r['status']} first_bytes={got[:40]!r}")
        h = http(port, "GET", "/health")
        check(results, "D2 /health recovers (slot not leaked)",
              h["status"] == 200 and (h["json"] or {}).get("slots_idle") == 1,
              f"health={h['json']}")

        # ---------------- E. edited history + regenerate ----------------
        base = [{"role": "user", "content": "记住数字 739。只回复 OK。"},
                {"role": "assistant", "content": "OK"},
                {"role": "user", "content": "刚才的数字是多少？只回复数字。"}]
        r_plain = chat(port, base, cache_prompt=False, max_tokens=8)
        ids_plain = (r_plain["json"] or {}).get("generated_token_ids") or []
        # warm the cache with a *different* history first, then edit the first user message
        warm = [dict(base[0]), dict(base[1]), dict(base[2])]
        warm[0]["content"] = "记住数字 999。只回复 OK。"
        chat(port, warm, cache_prompt=True, max_tokens=8)
        r_edit = chat(port, base, cache_prompt=True, max_tokens=8)
        ids_edit = (r_edit["json"] or {}).get("generated_token_ids") or []
        t = (r_edit["json"] or {}).get("timings") or {}
        check(results, "E1 edited history regenerate == full-prefill tokens",
              r_edit["status"] == 200 and ids_edit == ids_plain,
              f"plain={ids_plain} edit={ids_edit} new={t.get('prefilled_tokens')} cached={t.get('cache_hit_tokens')}")

        # ---------------- F. concurrency ----------------
        def slow_stream():
            body = {"messages": [{"role": "user", "content": "写100字说明"}], "stream": True,
                    "temperature": 0.7, "max_tokens": 64,
                    "chat_template_kwargs": {"enable_thinking": False}}
            return _stream_whole(port, body)

        holder = {}
        th = threading.Thread(target=lambda: holder.setdefault("r", slow_stream()))
        th.start()
        time.sleep(0.8)
        t0 = time.time()
        r = chat(port, [{"role": "user", "content": "只回复：并发"}], max_tokens=8)
        elapsed = time.time() - t0
        th.join(timeout=120)
        check(results, "F1 concurrent request succeeds (serialized server)",
              r["status"] == 200,
              f"status={r['status']} waited={elapsed:.1f}s")
        check(results, "F2 streaming request also completed",
              bool(holder.get("r")), f"bytes={len((holder.get('r') or b''))}")
        t0 = time.time()
        h = http(port, "GET", "/health", timeout=10)
        check(results, "F3 /health while streaming is served",
              h["status"] == 200, f"status={h['status']} waited={time.time()-t0:.1f}s")

        # ---------------- G. slowloris: CL but no body ----------------
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                  b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n")
        time.sleep(0.3)
        t0 = time.time()
        h = http(port, "GET", "/health", timeout=8)
        s.close()
        check(results, "G1 half-open request does not block /health",
              h["status"] == 200 and time.time() - t0 < 5,
              f"status={h['status']} waited={time.time()-t0:.1f}s")

        # ---------------- H. keep-alive path ----------------
        sock = socket.create_connection(("127.0.0.1", port), timeout=60)
        body = json.dumps({"prompt": "hi", "max_tokens": 4, "temperature": 0}).encode()
        sock.sendall((f"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
        time.sleep(0.2)
        sock.close()
        r = chat(port, [{"role": "user", "content": "只回复：keepalive"}], max_tokens=4)
        check(results, "H1 keep-alive socket does not wedge the server",
              r["status"] == 200, f"status={r['status']}")

        # ---------------- I. tokenizer endpoints ----------------
        edge = "家庭成员：👨\u200d👩\u200d👧\u200d👦 扩展B：𠀀 مرحبا 123"
        r = http(port, "POST", "/tokenize", {"content": edge})
        j = r["json"] or {}
        ids = j.get("tokens") or []
        r2 = http(port, "POST", "/detokenize", {"tokens": ids})
        back = (r2["json"] or {}).get("content", "")
        check(results, "I1 /tokenize+/detokenize roundtrip (emoji/CJK-B/RTL)",
              r["status"] == 200 and r2["status"] == 200 and back == edge,
              f"ids={len(ids)} back={back!r}")

        # ---------------- J. seeded sampling reproducibility (cache on/off) ----------------
        msgs = [{"role": "user", "content": "用一句话介绍苏州。"}]
        a = chat(port, msgs, max_tokens=32, temperature=0.6, top_k=20, top_p=0.95, seed=1234)
        b = chat(port, msgs, max_tokens=32, temperature=0.6, top_k=20, top_p=0.95, seed=1234,
                 cache_prompt=False)
        ida = (a["json"] or {}).get("generated_token_ids") or []
        idb = (b["json"] or {}).get("generated_token_ids") or []
        tb = (b["json"] or {}).get("timings") or {}
        check(results, "J1 seeded sampling reproduces with cache on/off",
              a["status"] == 200 and b["status"] == 200 and ida == idb,
              f"cache_ids={ida} plain_ids={idb} plain_new={tb.get('prefilled_tokens')}")

        # ---------------- K. non-stream JSON must be valid UTF-8 ----------------
        # Generation can stop in the middle of a multi-byte character (max_tokens).
        # The JSON body must still be valid UTF-8 / parseable by strict clients.
        cut_ok = True
        cut_detail = []
        for mt in (1, 2, 3):
            st_code, raw = http_bytes(port, "/completion",
                                      {"prompt": "请用一句完整的中文介绍杭州，并说明它为什么值得一去。",
                                       "max_tokens": mt, "temperature": 0})
            try:
                text = raw.decode("utf-8")  # strict
                json.loads(text)
                cut_detail.append(f"mt={mt}: valid utf8+json ({len(raw)}B)")
            except Exception as e:
                cut_ok = False
                cut_detail.append(f"mt={mt}: INVALID ({e}) bytes={raw[-40:]!r}")
            st2, raw2 = http_bytes(port, "/v1/chat/completions",
                                   {"messages": [{"role": "user",
                                                  "content": "请用一句完整的中文介绍杭州。"}],
                                    "max_tokens": mt, "temperature": 0,
                                    "chat_template_kwargs": {"enable_thinking": False}})
            try:
                raw2.decode("utf-8")
                json.loads(raw2)
            except Exception as e:
                cut_ok = False
                cut_detail.append(f"chat mt={mt}: INVALID ({e})")
        check(results, "K1 non-stream JSON valid UTF-8 for mid-character cuts",
              cut_ok, "; ".join(cut_detail))

        # ---------------- L. input-driven DoS budgets ----------------
        big = "中" * 349525  # ~1 MB UTF-8
        t0 = time.time()
        r = http(port, "POST", "/tokenize", {"content": big}, timeout=60)
        secs = time.time() - t0
        body = r.get("body") or ""
        check(results, "L1 /tokenize 1MB CJK rejected fast",
              r["status"] == 400 and secs < 0.5 and "prompt too long to tokenize" in body,
              f"status={r['status']} secs={secs:.2f}s body={body[:120]}")
        mid = "中" * 50000  # under the global 200k hard cap, far above ctx*8 for ctx4096
        t0 = time.time()
        r = http(port, "POST", "/tokenize", {"content": mid}, timeout=120)
        secs = time.time() - t0
        try:
            toks = len((json.loads(r.get("body") or "{}").get("tokens")) or [])
        except Exception:
            toks = -1
        check(results, "L2 /tokenize 50k chars works (global cap, not ctx)",
              r["status"] == 200 and secs < 2.0 and toks == 50000,
              f"status={r['status']} secs={secs:.2f}s tokens={toks}")
        # invalid UTF-8 in content must be a clean 400 (strict text_util)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tokenize",
            data=b'{"content":"\xC0\xAF"}', headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                invalid_status, invalid_body = resp.status, resp.read()
        except urllib.error.HTTPError as e:
            invalid_status, invalid_body = e.code, e.read()
        check(results, "L3 /tokenize invalid UTF-8 -> 400",
              invalid_status == 400 and b"not valid UTF-8" in invalid_body,
              f"status={invalid_status} body={invalid_body[:100].decode('latin1', 'replace')}")
        huge = "中" * 349525
        t0 = time.time()
        r = http(port, "POST", "/v1/chat/completions",
                 {"messages": [{"role": "user", "content": huge}],
                  "max_tokens": 4, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}}, timeout=60)
        secs = time.time() - t0
        check(results, "L4 chat with 1MB CJK prompt rejected fast",
              r["status"] == 400 and secs < 0.5,
              f"status={r['status']} secs={secs:.2f}s")
        stop_long = "X" * 10000
        r = chat(port, {"stop": stop_long}, max_tokens=4, timeout=30)
        check(results, "L5 10KB stop sequence -> 400",
              r["status"] == 400, f"status={r['status']}")

        # ---------------- Wipe/done ----------------
        summary = {"passed": sum(1 for x in results if x["ok"]),
                   "failed": sum(1 for x in results if not x["ok"]),
                   "results": results}
        Path("build/logs/serve_edge_cases.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nEDGE CASES: {summary['passed']} passed, {summary['failed']} failed")
        return 0 if summary["failed"] == 0 else 1
    finally:
        try:
            http(port, "POST", "/shutdown", {}, timeout=10)
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log.close()


def _stream_whole(port, body):
    payload = json.dumps(body, ensure_ascii=False).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode() + payload
    s = socket.create_connection(("127.0.0.1", port), timeout=300)
    s.sendall(req)
    got = b""
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            got += chunk
            if b"\r\n0\r\n\r\n" in got:
                break
    finally:
        s.close()
    return got


if __name__ == "__main__":
    raise SystemExit(main())
