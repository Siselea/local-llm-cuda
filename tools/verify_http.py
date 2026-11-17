#!/usr/bin/env python
"""HTTP-layer acceptance tests for build/test_http.exe (no model, no GPU).

Drives the real WinSock2 server through http.client and raw sockets:
keep-alive reuse, Content-Length bodies, chunked streaming, CORS preflight,
404 JSON errors, malformed request -> 400, HTTP/1.0 close semantics, pipelined
requests, and /shutdown.

Usage: python tools/verify_http.py [--port 7039]
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.request
import http.client

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

failures = 0
passes = 0


def ok(name: str, detail: str = "") -> None:
    global passes
    passes += 1
    print(f"  {name:70s} PASS{(' ' + detail) if detail else ''}")


def bad(name: str, detail: str = "") -> None:
    global failures
    failures += 1
    print(f"  {name:70s} FAIL{(' ' + detail) if detail else ''}")


def expect(name: str, cond: bool, detail: str = "") -> None:
    ok(name, detail) if cond else bad(name, detail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7039)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    host, port = args.host, args.port

    # 1. basic GET
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("GET", "/ping")
    resp = conn.getresponse()
    body = resp.read()
    expect("GET /ping -> 200 + JSON",
           resp.status == 200 and json.loads(body).get("pong") is True,
           f"status={resp.status} body={body!r}")

    # 2. keep-alive: two requests on the same connection
    conn.request("GET", "/ping")
    r2 = conn.getresponse()
    b2 = r2.read()
    conn.request("GET", "/latin1")
    r3 = conn.getresponse()
    b3 = r3.read()
    expect("keep-alive reuse (2 requests, 1 socket)",
           r2.status == 200 and r3.status == 200 and b3 == "你好\n".encode("utf-8"),
           f"status={r2.status}/{r3.status} body={b3!r}")

    # 3. large POST body (Content-Length handling)
    payload = ("0123456789abcdef" * 8192).encode()  # 128 KiB
    conn.request("POST", "/echo", body=payload,
                 headers={"Content-Type": "application/octet-stream"})
    r4 = conn.getresponse()
    j4 = json.loads(r4.read())
    expect("POST /echo 128 KiB body round trip",
           r4.status == 200 and j4["length"] == len(payload) and j4["body"].encode() == payload,
           f"status={r4.status} length={j4.get('length')}")

    # 4. chunked streaming response
    conn.request("POST", "/stream", body=b"")
    r5 = conn.getresponse()
    streamed = r5.read()
    hdrs = dict((k.lower(), v) for k, v in r5.getheaders())
    expect("POST /stream -> chunked, 5 chunks, complete body",
           r5.status == 200 and hdrs.get("transfer-encoding") == "chunked"
           and streamed == b"".join(b"chunk%d\n" % i for i in range(5)),
           f"status={r5.status} te={hdrs.get('transfer-encoding')} body={streamed!r}")

    # 5. status codes + JSON error body
    conn.request("GET", "/status/418")
    r6 = conn.getresponse()
    b6 = r6.read()
    expect("GET /status/418 -> 418 with JSON body",
           r6.status == 418 and b"\"status\": 418" in b6, f"status={r6.status}")

    conn.request("GET", "/nope")
    r7 = conn.getresponse()
    b7 = r7.read()
    err = json.loads(b7).get("error", {})
    expect("GET /nope -> 404 JSON error",
           r7.status == 404 and "no such" in err.get("message", ""),
           f"status={r7.status} body={b7[:80]!r}")

    # 6. CORS preflight
    conn.request("OPTIONS", "/v1/chat/completions",
                 headers={"Origin": "http://localhost",
                          "Access-Control-Request-Method": "POST"})
    r8 = conn.getresponse()
    r8.read()
    h8 = dict((k.lower(), v) for k, v in r8.getheaders())
    expect("OPTIONS preflight -> 204 + CORS headers",
           r8.status == 204 and h8.get("access-control-allow-origin") == "*"
           and "POST" in h8.get("access-control-allow-methods", ""),
           f"status={r8.status}")
    conn.close()

    # 7. malformed request line -> 400 (raw socket)
    s = socket.create_connection((host, port), timeout=10)
    s.sendall(b"GARBAGE\r\n\r\n")
    raw = s.recv(4096)
    s.close()
    expect("malformed request line -> 400",
           raw.startswith(b"HTTP/1.1 400"), repr(raw[:60]))

    # 8. HTTP/1.0 without keep-alive: response then close
    s = socket.create_connection((host, port), timeout=10)
    s.sendall(b"GET /ping HTTP/1.0\r\n\r\n")
    data = b""
    while True:
        piece = s.recv(4096)
        if not piece:
            break
        data += piece
    s.close()
    expect("HTTP/1.0 request: body then connection close",
           data.startswith(b"HTTP/1.1 200") and b'"pong": true' in data
           and b"Connection: close" in data, f"{len(data)} bytes")

    # 9. pipelined requests in one TCP write (pending-buffer path)
    s = socket.create_connection((host, port), timeout=10)
    s.sendall(b"GET /ping HTTP/1.1\r\nHost: x\r\n\r\nGET /latin1 HTTP/1.1\r\nHost: x\r\n\r\n")
    data = b""
    deadline = time.time() + 5
    while time.time() < deadline:
        if data.count(b"HTTP/1.1 200") >= 2 and "你好".encode() in data:
            break
        try:
            piece = s.recv(4096)
        except OSError:
            break
        if not piece:
            break
        data += piece
    s.close()
    expect("pipelined requests -> two responses",
           data.count(b"HTTP/1.1 200") == 2 and "你好".encode() in data,
           f"responses={data.count(b'HTTP/1.1 200')}")

    # 10. shutdown
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("POST", "/shutdown", body=b"")
    rs = conn.getresponse()
    bs = rs.read()
    expect("POST /shutdown -> 200",
           rs.status == 200 and b"shutting down" in bs, f"status={rs.status}")
    conn.close()

    # port should be released shortly
    released = False
    for _ in range(50):
        try:
            socket.create_connection((host, port), timeout=0.5).close()
        except OSError:
            released = True
            break
        time.sleep(0.1)
    expect("port released after shutdown", released)

    print()
    print(f"results: {passes} passed, {failures} failed")
    print("HTTP LAYER GREEN" if failures == 0 else "HTTP LAYER FAILURES")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
