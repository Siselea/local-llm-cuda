# -*- coding: utf-8 -*-
"""v0.1.9 runtime robustness checks for `engine serve` (0.5B, small GPU footprint).

A. 200 x "streaming request then immediate RST":
   - RSS / VRAM / slot counters every 25 cycles (leak or monotonic growth?)
   - after the storm: normal non-stream request OK, prefix cache on/off ids identical
B. shutdown races:
   - B1: two concurrent POST /shutdown while a stream generation is running
   - B2: two concurrent POST /shutdown on an idle server (fresh instance)
   - after shutdown: port closed, process exit code, VRAM released

Usage:
  MODEL_DIR=/path/to/models python build/rst_shutdown_test.py [--binary build/engine.exe]
Writes build/logs/rst_shutdown.json.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def model_path():
    d = os.environ.get("MODEL_DIR", "")
    if not d:
        raise SystemExit("set MODEL_DIR to your models directory")
    return str(Path(d) / DEFAULT_MODEL)


def http(port, path, body=None, timeout=60):
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


def json_body(r):
    try:
        return json.loads(r.get("body") or "{}")
    except Exception:
        return {}


def rss_mb(pid):
    if not pid:
        return None
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).WorkingSet64"],
            text=True, timeout=20).strip()
        return round(int(out) / (1 << 20), 1) if out else None
    except Exception:
        return None


def vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=20).strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def slots(port):
    # /health carries the busy/slot counters (single slot: idle 0/1, processing 0/1).
    h = json_body(http(port, "/health", timeout=5))
    return {"slots_idle": h.get("slots_idle"), "slots_processing": h.get("slots_processing"),
            "n_ctx": h.get("n_ctx"), "status": h.get("status")}


def health(port):
    return http(port, "/health", timeout=3)


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
    log_path = f"build/logs/rst_server_{port}.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [binary, "serve", model_path(), "--host", "127.0.0.1", "--port", str(port),
         "--ctx", "512", "--gpu-layers", "-1", "--threads", "8",
         "--default-max-tokens", "256"],
        stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            raise RuntimeError(f"server died; see {log_path}")
        if health(port)["status"] == 200:
            return proc, log
        time.sleep(0.4)
    proc.kill()
    raise RuntimeError("server health timeout")


def stop_server(proc, log, port):
    if proc.poll() is None:
        http(port, "/shutdown", {}, timeout=10)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    log.close()


def stream_socket(port, max_tokens=16):
    body = json.dumps({"messages": [{"role": "user", "content": "写一句话介绍杭州"}],
                       "stream": True, "temperature": 0.7, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
    s = socket.create_connection(("127.0.0.1", port), timeout=20)
    s.sendall(req)
    return s


def wait_port_closed(port, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=1)
            c.close()
            time.sleep(0.3)
        except Exception:
            return True
    return False


def shutdown_reqs(port, n=2):
    """Fire n /shutdown POSTs concurrently; return per-socket results."""
    results = [None] * n

    def one(i):
        try:
            c = socket.create_connection(("127.0.0.1", port), timeout=10)
            c.sendall(b"POST /shutdown HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
                      b"Content-Type: application/json\r\nConnection: close\r\n\r\n{}")
            c.settimeout(60)
            data = c.recv(256)
            results[i] = data.split(b"\r\n")[0].decode("latin1", "replace") if data else "EOF"
            c.close()
        except Exception as e:
            results[i] = f"error: {e!r}"

    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/engine.exe")
    ap.add_argument("--port", type=int, default=7025)
    ap.add_argument("--cycles", type=int, default=200)
    args = ap.parse_args()
    port = args.port
    Path("build/logs").mkdir(parents=True, exist_ok=True)
    out = {"model": model_path(), "cycles": args.cycles, "binary": args.binary}

    # ---------------- A. 200 x stream + immediate RST ----------------
    proc, log = start_server(args.binary, port)
    pid = server_pid(port)
    try:
        # Warm up the generation/prefill workspaces first: their one-time allocation
        # is not a leak and would otherwise look like a +100 MB baseline jump.
        for _ in range(2):
            http(port, "/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "hi"}], "temperature": 0,
                  "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}},
                 timeout=120)
        time.sleep(0.3)
        rss0, vr0 = rss_mb(pid), vram_mb()
        out["baseline"] = {"rss_mb": rss0, "vram_mb": vr0, "slots": slots(port),
                           "health": health(port)["status"], "warmup": True}
        samples, errors = [], 0
        t0 = time.time()
        for i in range(args.cycles):
            try:
                s = stream_socket(port)
                s.settimeout(10)
                s.recv(200)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("HH", 1, 0))
                s.close()
            except Exception:
                errors += 1
                try:
                    s.close()
                except Exception:
                    pass
            if i % 25 == 0 or i == args.cycles - 1:
                samples.append({"i": i, "rss_mb": rss_mb(pid), "vram_mb": vram_mb(),
                                "slots": slots(port)})
        elapsed = time.time() - t0
        after = {"rss_mb": rss_mb(pid), "vram_mb": vram_mb(), "slots": slots(port),
                 "health": health(port)["status"]}
        # correctness / prefix-cache state after the storm
        def chat(cache):
            return http(port, "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "记住数字 42，只回复 OK"}],
                         "temperature": 0, "max_tokens": 8, "cache_prompt": cache,
                         "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
        p_plain, p_cache = chat(False), chat(True)
        ids_plain = json_body(p_plain).get("generated_token_ids")
        ids_cache = json_body(p_cache).get("generated_token_ids")
        rss_vals = [s["rss_mb"] for s in samples if s["rss_mb"] is not None]
        monotonic = all(b >= a - 2.0 for a, b in zip(rss_vals, rss_vals[1:]))
        out["rst_storm"] = {
            "elapsed_s": round(elapsed, 2), "client_errors": errors,
            "rss_samples_mb": rss_vals,
            "rss_growth_mb": round(rss_vals[-1] - rss0, 1) if rss_vals and rss0 is not None else None,
            "rss_monotonic": monotonic,
            "vram_start_mb": vr0, "vram_end_mb": after["vram_mb"],
            "slots_samples": [s["slots"] for s in samples],
            "after": after,
            "plain_status": p_plain["status"], "cache_status": p_cache["status"],
            "ids_equal": ids_plain == ids_cache,
            "plain_content": json_body(p_plain).get("choices", [{}])[0].get("message", {}).get("content")
            if p_plain["status"] == 200 else None,
        }
    finally:
        stop_server(proc, log, port)
    out["after_phase_a"] = {"pid_alive": pid is not None and rss_mb(pid) is not None,
                            "port_closed": wait_port_closed(port, 15)}

    # ---------------- B1. concurrent /shutdown during generation ----------------
    proc2, log2 = start_server(args.binary, port)
    try:
        s = stream_socket(port, max_tokens=256)
        s.settimeout(10)
        try:
            s.recv(400)
        except Exception:
            pass
        t0 = time.time()
        results = shutdown_reqs(port, 2)
        latency = time.time() - t0
        try:
            s.close()
        except Exception:
            pass
        closed = wait_port_closed(port, 60)
        try:
            proc2.wait(timeout=30)
        except Exception:
            pass
        out["shutdown_during_generation"] = {
            "concurrent_results": results, "latency_s": round(latency, 2),
            "port_closed": closed, "process_rc": proc2.poll(),
            "vram_after_mb": vram_mb(),
        }
    finally:
        if proc2.poll() is None:
            stop_server(proc2, log2, port)
        else:
            log2.close()

    # ---------------- B2. two concurrent /shutdown on idle server --------------
    proc3, log3 = start_server(args.binary, port)
    try:
        t0 = time.time()
        results = shutdown_reqs(port, 2)
        latency = time.time() - t0
        closed = wait_port_closed(port, 60)
        try:
            proc3.wait(timeout=30)
        except Exception:
            pass
        out["shutdown_idle_double"] = {
            "concurrent_results": results, "latency_s": round(latency, 2),
            "port_closed": closed, "process_rc": proc3.poll(),
            "vram_after_mb": vram_mb(),
        }
    finally:
        if proc3.poll() is None:
            stop_server(proc3, log3, port)
        else:
            log3.close()

    Path("build/logs/rst_shutdown.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k in (
        "baseline", "rst_storm", "after_phase_a", "shutdown_during_generation",
        "shutdown_idle_double")}, ensure_ascii=False, indent=2))
    print("saved build/logs/rst_shutdown.json")


if __name__ == "__main__":
    main()
