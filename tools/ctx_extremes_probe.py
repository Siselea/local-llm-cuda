#!/usr/bin/env python
"""Arbitrary-context + extreme-condition probe for `engine serve`.

Task A (arbitrary ctx): load the model at every requested ctx and record requested vs
actual ctx, clamp reason, KV bytes, load time, first-token latency and VRAM/RAM peaks.
Task B (extreme matrix): tiny/huge prompts, tiny/huge ctx, tiny/big models, tight VRAM,
30-turn long session, back-to-back requests and empty input -- each cell records
pass/fail + speed + peak memory, so the report can say "acceptable / explained / fix".

The tool only needs stdlib + a working `engine.exe` (and nvidia-smi for VRAM peaks).
It never leaves a process behind: every server is shut down via /shutdown, then killed.

Examples:
  python tools/ctx_extremes_probe.py --scenario ctx_matrix --model <9b.gguf>
  python tools/ctx_extremes_probe.py --scenario all --model <9b.gguf> \
      --big-model <35b.gguf> --small-model <0.5b.gguf>
  python tools/ctx_extremes_probe.py --scenario perf --model <9b.gguf> --big-model ...
Outputs: build/ctx_extremes/results.json (+ a markdown table on stdout).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "build" / "ctx_extremes"
ENGINE = str(ROOT / "build" / "engine.exe")
START_PORT = 7300

# ---------------------------------------------------------------------------
# memory sampling (VRAM via nvidia-smi, RSS via psapi GetProcessMemoryInfo)
# ---------------------------------------------------------------------------
psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return int(float(out[0]))
    except Exception:
        return -1


def proc_rss_mib(pid: int) -> int:
    try:
        h = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
        if not h:
            return -1
        try:
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            if not psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return -1
            return int(pmc.WorkingSetSize // (1024 * 1024))
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        return -1


class Sampler(threading.Thread):
    """Polls global GPU used MiB + engine RSS while a server is alive."""

    def __init__(self, proc: subprocess.Popen, interval: float = 0.25):
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval
        self.stop = threading.Event()
        self.gpu_max = 0
        self.gpu_base = gpu_used_mib()
        self.rss_max = 0

    def run(self):
        while not self.stop.is_set():
            g = gpu_used_mib()
            if g > self.gpu_max:
                self.gpu_max = g
            r = proc_rss_mib(self.proc.pid) if self.proc.poll() is None else -1
            if r > self.rss_max:
                self.rss_max = r
            self.stop.wait(self.interval)

    def finish(self):
        self.stop.set()
        self.join(timeout=2)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def http_json(method: str, url: str, body=None, timeout=600):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            if not raw:
                return r.status, {}, time.time() - t0
            try:
                return r.status, json.loads(raw), time.time() - t0
            except UnicodeDecodeError as ue:
                # Do not fail the whole measurement if the server emits a JSON body
                # with an invalid UTF-8 tail (repro for the engine bug this probes):
                # decode leniently and remember the raw prefix for the report.
                text = raw.decode('utf-8', 'replace')
                try:
                    payload = json.loads(text)
                except Exception:
                    payload = {'_invalid_utf8': str(ue), '_raw_hex': raw[:96].hex()}
                return r.status, payload, time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"raw": raw.decode("utf-8", "replace")[:400]}
        return e.code, payload, time.time() - t0


def port_free(port: int) -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ---------------------------------------------------------------------------
# One engine serve process
# ---------------------------------------------------------------------------
class Server:
    def __init__(self, model: str, ctx: int, kv: str = "f32", port: int | None = None,
                 extra: list[str] | None = None, tag: str = "case"):
        self.model = model
        self.ctx = ctx
        self.kv = kv
        self.tag = tag
        self.port = port or self._find_port()
        self.log_path = OUTDIR / f"serve-{tag}-{ctx}-{kv}-{self.port}.log"
        self.log = None
        self.proc = None
        self.sampler = None
        self.extra = extra or []
        self.load_s = 0.0
        self.ok = False
        self.error = ""

    @staticmethod
    def _find_port() -> int:
        for p in range(START_PORT, START_PORT + 200):
            if port_free(p):
                return p
        raise RuntimeError("no free port")

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        OUTDIR.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        cmd = [ENGINE, "serve", self.model, "--host", "127.0.0.1", "--port", str(self.port),
               "--ctx", str(self.ctx), "--kv-type", self.kv, "--no-interactive"] + self.extra
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=str(ROOT))
        self.sampler = Sampler(self.proc)
        self.sampler.start()
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.error = "engine exited during load"
                break
            try:
                st, body, _ = http_json("GET", self.base + "/health", timeout=2)
                if st == 200:
                    self.ok = True
                    break
            except Exception:
                pass
            time.sleep(0.2)
        self.load_s = time.time() - t0
        if not self.ok and not self.error:
            self.error = "timeout waiting for /health"
        return self

    def props(self):
        return http_json("GET", self.base + "/props", timeout=10)[1]

    def post(self, path: str, body, timeout=600):
        return http_json("POST", self.base + path, body, timeout=timeout)

    def log_text(self) -> str:
        try:
            self.log.flush()
        except Exception:
            pass
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def __exit__(self, *exc):
        try:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    http_json("POST", self.base + "/shutdown", timeout=5)
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        if self.sampler is not None:
            self.sampler.finish()
        if self.log is not None:
            self.log.close()
        return False

    # ------------------------------- metrics -------------------------------
    def metrics(self) -> dict:
        text = self.log_text()
        m = {}
        mm = re.search(r"\[vram\] post-load used=(\d+) MiB free=(\d+) MiB", text)
        if mm:
            m["post_load_used_mib"] = int(mm.group(1))
            m["post_load_free_mib"] = int(mm.group(2))
        mm = re.search(r"\[ctx\] requested=(\d+)->(\d+) kv_type=(\w+) kv/token=(\d+) B", text)
        if mm:
            m["ctx_requested_log"] = int(mm.group(1))
            m["ctx_actual_log"] = int(mm.group(2))
            m["kv_type_log"] = mm.group(3)
            m["kv_per_token"] = int(mm.group(4))
        mm = re.search(r"\[ctx\] CLAMPED: (\d+) > max_ctx (\d+) for kv_type=(\w+)", text)
        if mm:
            m["clamp_log"] = {"requested": int(mm.group(1)), "actual": int(mm.group(2)),
                              "kv_type": mm.group(3)}
        if self.sampler is not None:
            m["gpu_peak_mib"] = self.sampler.gpu_max
            m["gpu_base_mib"] = self.sampler.gpu_base
            m["engine_gpu_peak_mib"] = (self.sampler.gpu_max - self.sampler.gpu_base
                                        if self.sampler.gpu_max > 0 else -1)
            m["rss_peak_mib"] = self.sampler.rss_max
        return m


def first_token_ms(timings: dict) -> float:
    p = timings.get("prefill_ms") or 0.0
    d = timings.get("decode_ms") or 0.0
    return round(float(p) + float(d), 1)


def completion(server: Server, prompt: str, n: int, timeout=900) -> dict:
    st, body, wall = server.post("/completion",
                                 {"prompt": prompt, "n_predict": n, "temperature": 0},
                                 timeout=timeout)
    out = {"status": st, "wall_s": round(wall, 3)}
    if st != 200 or "error" in body:
        out["error"] = body.get("error", body)
        return out
    t = body.get("timings", {})
    out.update({
        "tokens_evaluated": body.get("tokens_evaluated"),
        "tokens_predicted": body.get("tokens_predicted"),
        "prefill_ms": t.get("prefill_ms"),
        "decode_ms": t.get("decode_ms"),
        "first_token_ms": first_token_ms(t),
        "prefill_tok_s": round((body.get("tokens_evaluated") or 0) * 1000.0 / t["prefill_ms"], 1)
        if t.get("prefill_ms") else None,
        "decode_tok_s": round((body.get("tokens_predicted") or 0) * 1000.0 / t["decode_ms"], 2)
        if t.get("decode_ms") else None,
        "content_head": (body.get("content") or "")[:40],
    })
    return out


def ctx_case(model: str, ctx: int, kv: str, workdir_out: dict) -> dict:
    tag = Path(model).stem[:18]
    r = {"scenario": "ctx_matrix", "model": Path(model).name, "ctx_req": ctx, "kv": kv}
    with Server(model, ctx, kv, tag=f"ctx-{tag}") as s:
        r["ok"] = s.ok
        r["error"] = s.error
        r["load_s"] = round(s.load_s, 1)
        if s.ok:
            p = s.props()
            r["n_ctx"] = p.get("n_ctx")
            r["ctx_requested"] = p.get("n_ctx_requested")
            r["requested_ctx"] = p.get("requested_ctx")
            r["actual_ctx"] = p.get("actual_ctx")
            r["ctx_clamped"] = p.get("ctx_clamped")
            r["ctx_train"] = p.get("ctx_train")
            r["ctx_train_exceeded"] = p.get("ctx_train_exceeded")
            r["completion"] = completion(s, "Hello", 1)
            r["completion_prompt"] = "Hello"
            r.update(s.metrics())
        else:
            log = s.log_text()
            r["failure_tail"] = [ln for ln in log.splitlines()
                                 if "ERROR" in ln or "hint" in ln][-3:]
    return r


# ---------------------------------------------------------------------------
# prompt calibration through the server tokenizer
# ---------------------------------------------------------------------------
def build_prompt_with_tokens(server: Server, target: int) -> str:
    """Build a repeating-text prompt whose /tokenize count is just below `target`.

    Repeated text tokenizes very regularly, so iterate on the unit count with the
    measured tokens/unit ratio; keep the closest prompt that does not exceed target.
    """
    unit = "The quick brown fox jumps over the lazy dog. "
    units = max(1, target // 10)
    best, best_n = "", 0
    for _ in range(16):
        text = unit * units
        n = tokens_of(server, text)
        if n <= 0:
            return text
        if n <= target and n > best_n:
            best, best_n = text, n
        tol = max(2, target // 200)
        if abs(n - target) <= tol:
            return text if n <= target else (best or text)
        prev = units
        units = max(1, int(round(units * (target / n))))
        if units == prev:
            units += -1 if n > target else 1
        units = max(1, units)
    if best:
        return best
    return unit * units


def tokens_of(server: Server, text: str) -> int:
    st, body, _ = server.post("/tokenize", {"content": text}, timeout=120)
    return len(body.get("tokens", [])) if st == 200 else -1


# ---------------------------------------------------------------------------
# Task B scenarios
# ---------------------------------------------------------------------------
def scenario_short_prompt(model: str) -> list[dict]:
    rows = []
    with Server(model, 4096, "f32", tag="short") as s:
        if not s.ok:
            return [{"scenario": "short_prompt", "ok": False, "error": s.error}]
        for prompt, label in [("Hi", "1-2 tokens"), ("Hello", "1-2 tokens"),
                              ("Hi there", "2-4 tokens")]:
            ntok = tokens_of(s, prompt)
            r = completion(s, prompt, 8)
            r.update({"scenario": "short_prompt", "prompt": prompt, "prompt_tokens": ntok})
            rows.append(r)
        rows[-1].update(s.metrics())
    return rows


def scenario_long_prompt(model: str, ctx: int) -> list[dict]:
    rows = []
    with Server(model, ctx, "f32", tag="long") as s:
        if not s.ok:
            return [{"scenario": "long_prompt", "ctx": ctx, "ok": False, "error": s.error}]
        target = ctx - 256  # ~99% of ctx; leaves room for 8 generated tokens
        text = build_prompt_with_tokens(s, target)
        ntok = tokens_of(s, text)
        r = completion(s, text, 8)
        r.update({"scenario": "long_prompt", "ctx": ctx, "prompt_tokens_target": target,
                  "prompt_tokens_actual": ntok})
        r.update(s.metrics())
        rows.append(r)
    return rows


def scenario_tiny_ctx(model: str) -> list[dict]:
    rows = []
    for ctx in (16, 64, 256):
        with Server(model, ctx, "f32", tag=f"tiny-{ctx}") as s:
            r = {"scenario": "tiny_ctx", "ctx_req": ctx, "ok": s.ok, "error": s.error,
                 "load_s": round(s.load_s, 1)}
            if s.ok:
                p = s.props()
                r.update({"n_ctx": p.get("n_ctx"), "ctx_requested": p.get("n_ctx_requested")})
                r["short_completion"] = completion(s, "Hello", 8)
                # over-long prompt must be a clear 400, not a crash
                long_p = "word " * 200
                st, body, _ = s.post("/completion", {"prompt": long_p, "n_predict": 1,
                                                     "temperature": 0})
                r["overctx_status"] = st
                r["overctx_message"] = str(body.get("error", ""))[:180]
                r.update(s.metrics())
            rows.append(r)
    return rows


def scenario_huge_ctx(model: str) -> list[dict]:
    rows = []
    for ctx, kv in ((32768, "q8_0"), (131072, "q8_0"), (262144, "q4_0")):
        with Server(model, ctx, kv, tag=f"huge-{ctx}") as s:
            r = {"scenario": "huge_ctx", "ctx_req": ctx, "kv": kv, "ok": s.ok,
                 "error": s.error, "load_s": round(s.load_s, 1)}
            if s.ok:
                p = s.props()
                r.update({"n_ctx": p.get("n_ctx"), "ctx_requested": p.get("n_ctx_requested"),
                          "actual_ctx": p.get("actual_ctx"), "ctx_clamped": p.get("ctx_clamped")})
                r["completion"] = completion(s, "Hello", 8)
                r.update(s.metrics())
            else:
                r["failure_tail"] = [ln for ln in s.log_text().splitlines()
                                     if "ERROR" in ln or "hint" in ln][-3:]
            rows.append(r)
    return rows


def scenario_model_extremes(small_model: str, big_model: str) -> list[dict]:
    rows = []
    for label, model in (("small", small_model), ("big", big_model)):
        if not model or not Path(model).exists():
            rows.append({"scenario": "model_extremes", "kind": label, "ok": False,
                         "error": "model missing"})
            continue
        with Server(model, 4096, "f32", tag=f"model-{label}", extra=["--vram-margin", "1200"]) as s:
            r = {"scenario": "model_extremes", "kind": label, "model": Path(model).name,
                 "ok": s.ok, "error": s.error, "load_s": round(s.load_s, 1)}
            if s.ok:
                p = s.props()
                r.update({"n_ctx": p.get("n_ctx"), "actual_ctx": p.get("actual_ctx")})
                r["completion"] = completion(s, "Hello", 16)
                r.update(s.metrics())
            else:
                r["failure_tail"] = [ln for ln in s.log_text().splitlines()
                                     if "ERROR" in ln or "hint" in ln][-3:]
            rows.append(r)
    return rows


def scenario_vram_tight(hog_model: str, target_model: str) -> list[dict]:
    """Occupy 2-4 GiB with a small-model serve, then start the 9B target."""
    rows = []
    hog = Server(hog_model, 131072, "f32", tag="vram-hog")
    with hog:
        if not hog.ok:
            return [{"scenario": "vram_tight", "ok": False, "error": "hog failed: " + hog.error}]
        free_after_hog = gpu_used_mib()
        rows.append({"scenario": "vram_tight", "phase": "hog", "ok": True,
                     "gpu_used_mib": free_after_hog,
                     "hog_ctx": 131072, "hog_kv": "f32"})
        with Server(target_model, 4096, "f32", tag="vram-target") as t:
            r = {"scenario": "vram_tight", "phase": "target", "ok": t.ok, "error": t.error,
                 "load_s": round(t.load_s, 1)}
            if t.ok:
                p = t.props()
                r.update({"n_ctx": p.get("n_ctx"), "actual_ctx": p.get("actual_ctx")})
                r["completion"] = completion(t, "Hello", 8)
                r.update(t.metrics())
            else:
                r["failure_tail"] = [ln for ln in t.log_text().splitlines()
                                     if "ERROR" in ln or "hint" in ln or "WARN" in ln][-4:]
            rows.append(r)
    return rows


def parse_req_lines(log_text: str) -> list[dict]:
    """Parse the serve [req] lines for per-request prefill/cache numbers."""
    out = []
    for ln in log_text.splitlines():
        m = re.search(r"prompt=(\d+) gen=(\d+)\s+(\d+) ms .*prefill (\d+) \[new (\d+), cached (\d+)\]",
                      ln)
        if m:
            out.append({"prompt": int(m.group(1)), "gen": int(m.group(2)),
                        "wall_ms": int(m.group(3)), "prefill_ms": int(m.group(4)),
                        "new": int(m.group(5)), "cached": int(m.group(6))})
    return out


def scenario_session30(model: str) -> list[dict]:
    rows = []
    with Server(model, 8192, "f32", tag="session30") as s:
        if not s.ok:
            return [{"scenario": "session30", "ok": False, "error": s.error}]
        messages = []
        user_unit = "请根据下面的材料回答一个问题：" + ("关于本地推理引擎的上下文与显存。 " * 8)
        t_start = time.time()
        for turn in range(1, 31):
            messages.append({"role": "user", "content": user_unit + f"第 {turn} 轮。"})
            st, body, wall = s.post("/v1/chat/completions",
                                    {"messages": messages, "max_tokens": 16,
                                     "temperature": 0, "stream": False}, timeout=1200)
            row = {"scenario": "session30", "turn": turn, "status": st,
                   "wall_s": round(wall, 2)}
            if st == 200:
                t = body.get("timings", {})
                row.update({"prefill_ms": t.get("prefill_ms"),
                            "decode_ms": t.get("decode_ms"),
                            "prompt_tokens": body.get("usage", {}).get("prompt_tokens")})
                msg = body["choices"][0]["message"]
                messages.append({"role": "assistant",
                                 "content": msg.get("content") or ""})
            else:
                row["error"] = str(body.get("error", body))[:160]
            rows.append(row)
            if turn in (1, 10, 20, 30):
                rows[-1].update({k: v for k, v in s.metrics().items()
                                 if k in ("post_load_free_mib", "rss_peak_mib")})
        rows[-1]["total_wall_s"] = round(time.time() - t_start, 1)
        rows[-1]["req_lines"] = parse_req_lines(s.log_text())
    return rows


def scenario_back_to_back(model: str) -> list[dict]:
    rows = []
    with Server(model, 4096, "f32", tag="b2b") as s:
        if not s.ok:
            return [{"scenario": "back_to_back", "ok": False, "error": s.error}]
        completion(s, "Hello", 4)  # warmup
        for i in range(10):
            st, body, wall = s.post("/completion",
                                    {"prompt": f"Request number {i}: Hello", "n_predict": 8,
                                     "temperature": 0}, timeout=300)
            row = {"scenario": "back_to_back", "i": i, "status": st,
                   "wall_s": round(wall, 2)}
            if st == 200:
                t = body.get("timings", {})
                row.update({"prefill_ms": t.get("prefill_ms"), "decode_ms": t.get("decode_ms")})
            else:
                row["error"] = str(body.get("error", body))[:160]
            rows.append(row)
        rows.append({"scenario": "back_to_back", "summary": True,
                     "n_ok": sum(1 for r in rows if r.get("status") == 200),
                     "wall_s": [r.get("wall_s") for r in rows if "wall_s" in r]})
    return rows


def scenario_empty_input(model: str) -> list[dict]:
    rows = []
    with Server(model, 4096, "f32", tag="empty") as s:
        if not s.ok:
            return [{"scenario": "empty_input", "ok": False, "error": s.error}]
        cases = [
            ("completion_empty_prompt", "/completion", {"prompt": "", "n_predict": 4}),
            ("chat_empty_messages", "/v1/chat/completions",
             {"messages": [], "max_tokens": 4}),
            ("completion_whitespace_prompt", "/completion", {"prompt": "   ", "n_predict": 4}),
        ]
        for name, path, body in cases:
            st, resp, wall = s.post(path, body, timeout=120)
            rows.append({"scenario": "empty_input", "case": name, "status": st,
                         "error": resp.get("error") if isinstance(resp, dict) else None,
                         "wall_s": round(wall, 2)})
    return rows


def scenario_perf(model: str, label: str, warmup: int = 1, runs: int = 2) -> list[dict]:
    rows = []
    with Server(model, 4096, "f32", tag=f"perf-{label}") as s:
        if not s.ok:
            return [{"scenario": "perf", "model": label, "ok": False, "error": s.error}]
        completion(s, "Hello", 8)  # warmup / graph capture
        vals = []
        for i in range(runs):
            r = completion(s, "The capital of France is", 32)
            r["run"] = i
            if "decode_tok_s" in r:
                vals.append(r["decode_tok_s"])
            rows.append(r)
        rows.append({"scenario": "perf", "model": label, "runs": runs,
                     "decode_tok_s_median": sorted(vals)[len(vals) // 2] if vals else None,
                     "vram_peak_mib": s.metrics().get("gpu_peak_mib")})
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(ENGINE))
    ap.add_argument("--model", required=True, help="primary model (9B)")
    ap.add_argument("--small-model", default="")
    ap.add_argument("--big-model", default="")
    ap.add_argument("--scenario", default="all",
                    help="all|ctx_matrix|short_prompt|long_prompt|tiny_ctx|huge_ctx|"
                         "model_extremes|vram_tight|session30|back_to_back|empty_input|perf")
    ap.add_argument("--ctx-list",
                    default="1,2,16,100,1000,4096,12345,32768")
    ap.add_argument("--out", default=str(OUTDIR / "results.json"))
    args = ap.parse_args()

    ENGINE = args.engine
    os.makedirs(OUTDIR, exist_ok=True)
    results: list[dict] = []
    todo = (["ctx_matrix", "short_prompt", "long_prompt", "tiny_ctx", "huge_ctx",
             "model_extremes", "vram_tight", "session30", "back_to_back", "empty_input",
             "perf"] if args.scenario == "all" else args.scenario.split(","))

    def run(name, fn, *a):
        print(f"\n===== {name} =====", flush=True)
        try:
            rows = fn(*a)
        except Exception as e:  # keep the probe going
            rows = [{"scenario": name, "ok": False, "error": f"probe exception: {e}"}]
        for r in rows:
            print(json.dumps(r, ensure_ascii=False)[:400], flush=True)
        results.extend(rows)

    if "ctx_matrix" in todo:
        for item in [x.strip() for x in args.ctx_list.split(",") if x.strip()]:
            if ":" in item:
                ctx_s, kv = item.split(":", 1)
            else:
                ctx_s, kv = item, "f32"
            ctx = int(ctx_s)
            run(f"ctx_matrix ctx={ctx} kv={kv}",
                lambda ctx=ctx, kv=kv: [ctx_case(args.model, ctx, kv, {})])
    if "short_prompt" in todo:
        run("short_prompt", scenario_short_prompt, args.model)
    if "long_prompt" in todo:
        for ctx in (4096, 8192):
            run(f"long_prompt ctx={ctx}", scenario_long_prompt, args.model, ctx)
    if "tiny_ctx" in todo:
        run("tiny_ctx", scenario_tiny_ctx, args.model)
    if "huge_ctx" in todo:
        run("huge_ctx", scenario_huge_ctx, args.model)
    if "model_extremes" in todo:
        run("model_extremes", scenario_model_extremes, args.small_model, args.big_model)
    if "vram_tight" in todo:
        run("vram_tight", scenario_vram_tight, args.small_model, args.model)
    if "session30" in todo:
        run("session30", scenario_session30, args.model)
    if "back_to_back" in todo:
        run("back_to_back", scenario_back_to_back, args.model)
    if "empty_input" in todo:
        run("empty_input", scenario_empty_input, args.model)
    if "perf" in todo:
        run("perf 9B", scenario_perf, args.model, "9B")
        if args.big_model and Path(args.big_model).exists():
            run("perf big", scenario_perf, args.big_model, "big")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
