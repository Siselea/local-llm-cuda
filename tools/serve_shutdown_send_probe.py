# -*- coding: utf-8 -*-
"""v0.1.9 probes for two serve edge questions (CPU-only, no model generation):

A. double /shutdown: first 200, second should get "503 shutting down" (best effort)
   instead of a TCP reset; then process/port cleanup.
B. slow reader: send a large /tokenize response, read only the headers, and measure
   how long /health is blocked while the client refuses to read the body.  This
   answers "is the send() blocking window acceptable for a single-user local
   server, or does it need non-blocking writes + timeout?".

Usage:
  MODEL_DIR=D:/p0/models python build/http_shutdown_send_probe.py --binary build/engine.exe
Writes build/logs/http_shutdown_send.json.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def http(port, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"status": r.status, "body": r.read().decode("utf-8", "replace")}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read().decode("utf-8", "replace")}
    except Exception as e:
        return {"status": None, "error": repr(e)}


def server_pid(port):
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process | Where-Object {{ $_.Name -eq 'engine.exe' -and "
             f"$_.CommandLine -like '*--port {port}*' }}).ProcessId"],
            text=True, timeout=20)
        ps = [int(x) for x in out.split() if x.strip().isdigit()]
        return ps[0] if ps else None
    except Exception:
        return None


def start_server(binary, port):
    model = str(Path(os.environ.get("MODEL_DIR") or "D:/p0/models") / MODEL)
    log = open(f"build/logs/http_probe_server_{port}.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [binary, "serve", model, "--host", "127.0.0.1", "--port", str(port),
         "--ctx", "256", "--gpu-layers", "0", "--threads", "4",
         "--default-max-tokens", "8"],
        stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 240:
        if proc.poll() is not None:
            raise RuntimeError(f"server died (see build/logs/http_probe_server_{port}.log)")
        if http(port, "/health", timeout=2)["status"] == 200:
            return proc, log
        time.sleep(0.4)
    proc.kill()
    raise RuntimeError("server health timeout")


def wait_port_closed(port, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            time.sleep(0.2)
        except Exception:
            return True
    return False


def stop_server(proc, log, port):
    if proc.poll() is None:
        http(port, "/shutdown", {}, timeout=10)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    log.close()


def double_shutdown(port):
    results = [None, None]

    def one(i):
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=10)
            c.sendall(b"POST /shutdown HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
                      b"Content-Type: application/json\r\nConnection: close\r\n\r\n{}")
            c.settimeout(30)
            data = c.recv(512)
            results[i] = (data.split(b"\r\n")[0].decode("latin1", "replace") if data else "EOF")
            c.close()
        except Exception as e:
            results[i] = f"error: {e!r}"

    ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return results


def slow_reader(port):
    """Return dict describing how long /health is blocked by a non-reading client."""
    body = json.dumps({"content": "中" * 50000}).encode()  # ~150 KB -> ~350 KB JSON reply
    req = (f"POST /tokenize HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\n\r\n").encode() + body
    slow = socket.create_connection(("127.0.0.1", port), timeout=10)
    slow.sendall(req)
    slow.settimeout(10)
    # read only the response headers (and at most a little body), then stop reading
    head = b""
    while b"\r\n\r\n" not in head and len(head) < 8192:
        head += slow.recv(1024)
    head_len = len(head)
    result = {"head_bytes": head_len,
              "head": head.split(b"\r\n")[0].decode("latin1", "replace")}

    health_result = {}

    def probe():
        t0 = time.time()
        r = http(port, "/health", timeout=40)
        health_result.update(r)
        health_result["latency_s"] = round(time.time() - t0, 2)

    th = threading.Thread(target=probe)
    th.start()
    time.sleep(3.0)
    result["health_started_while_blocked"] = "latency" not in health_result
    slow.close()  # unblock the server's send()
    th.join(timeout=40)
    result["health_after_close"] = health_result
    result["server_alive_after"] = http(port, "/health", timeout=5)["status"] == 200
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/engine.exe")
    ap.add_argument("--port", type=int, default=7027)
    args = ap.parse_args()
    port = args.port
    Path("build/logs").mkdir(parents=True, exist_ok=True)
    out = {}

    # A. double shutdown
    proc, log = start_server(args.binary, port)
    try:
        t0 = time.time()
        res = double_shutdown(port)
        out["double_shutdown"] = {"responses": res, "latency_s": round(time.time() - t0, 2),
                                  "port_closed": wait_port_closed(port, 30)}
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
        out["double_shutdown"]["process_rc"] = proc.poll()
        out["double_shutdown"]["pid_after"] = None
    finally:
        if proc.poll() is None:
            stop_server(proc, log, port)
        else:
            log.close()

    # B. slow reader / blocking send window
    proc2, log2 = start_server(args.binary, port)
    try:
        out["slow_reader"] = slow_reader(port)
        out["slow_reader"]["process_rc"] = proc2.poll()
    finally:
        stop_server(proc2, log2, port)

    Path("build/logs/http_shutdown_send.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("saved build/logs/http_shutdown_send.json")


if __name__ == "__main__":
    main()
