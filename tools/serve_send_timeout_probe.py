# -*- coding: utf-8 -*-
"""Upper bound of the blocking-send window: keep a non-reading client open and measure
how long /health (another connection) takes to get served.  Decides whether the
accepted socket inherits the listening socket's 30 s SO_SNDTIMEO or can block forever.

Usage: MODEL_DIR=D:/p0/models python build/send_timeout_bound.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
BIN = "build/wt_http503/build/engine.exe"
PORT = 7028


def health(timeout):
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=timeout) as r:
            code = r.status
            r.read()
        return {"status": code, "latency_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"status": None, "latency_s": round(time.time() - t0, 2), "error": repr(e)}


def main():
    env = dict(os.environ)
    model = str(Path(os.environ.get("MODEL_DIR") or "D:/p0/models") / MODEL)
    log = open("build/logs/send_timeout_server.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([BIN, "serve", model, "--host", "127.0.0.1", "--port", str(PORT),
                             "--ctx", "256", "--gpu-layers", "0", "--threads", "4",
                             "--default-max-tokens", "8"], stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            if health(2)["status"] == 200:
                break
            if proc.poll() is not None:
                raise SystemExit("server died")
            time.sleep(0.4)
        body = json.dumps({"content": "中" * 50000}).encode()
        req = (f"POST /tokenize HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        slow = socket.create_connection(("127.0.0.1", PORT), timeout=10)
        slow.sendall(req)
        slow.settimeout(10)
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 8192:
            head += slow.recv(1024)

        result = {}

        def probe():
            result.update(health(60))

        th = threading.Thread(target=probe)
        th.start()
        th.join(timeout=55)
        blocked_ms = None
        if th.is_alive():
            blocked_ms = ">55 s (no inherited send timeout?)"
        else:
            blocked_ms = result.get("latency_s")
        print(json.dumps({"health_while_client_not_reading": result,
                          "blocked_window_s": blocked_ms}, ensure_ascii=False, indent=2))
        slow.close()
        time.sleep(0.5)
        print(json.dumps({"health_after_client_close": health(10)}, ensure_ascii=False, indent=2))
        Path("build/logs/send_timeout_bound.json").write_text(
            json.dumps({"health_while_blocked": result, "blocked_window_s": blocked_ms},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/shutdown", data=b"{}",
                headers={"Content-Type": "application/json"}), timeout=5).read()
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
