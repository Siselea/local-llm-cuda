# -*- coding: utf-8 -*-
"""Drive the llama.cpp webui in a real (headless) Edge browser via CDP.

This is the "real user" acceptance path for `engine serve`:

  * open http://127.0.0.1:PORT/ in Edge (headless=new, CDP on --debug-port)
  * type Chinese text into the chat textarea (Input.insertText -> real input events)
  * click the send button (Input.dispatchMouseEvent, not a raw POST)
  * measure time to first rendered assistant character (streaming)
  * read the thinking block + answer text from the DOM (with showThoughtInProgress)
  * check the sidebar conversation title
  * repeat for N turns, screenshotting the key moments

Usage:
  python tools/webui_e2e.py --url http://127.0.0.1:7030 --turns 3 \
      --messages "你好，请用三句话介绍杭州。|再详细一点，重点说说它的历史和文化。|用同样的风格介绍一下苏州。" \
      --out-dir build/logs/e2e
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import websocket  # websocket-client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def http_json(url, method="GET", timeout=10):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


class Cdp:
    def __init__(self, port: int):
        self.port = port
        self.next_id = 0
        self.ws = None
        self.pending = {}

    def wait_ready(self, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                v = http_json(f"http://127.0.0.1:{self.port}/json/version", timeout=2)
                if v.get("webSocketDebuggerUrl"):
                    return v
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError("Edge CDP endpoint did not come up")

    def open_page(self, url: str):
        # PUT /json/new?url=... creates a new target and returns its ws url
        q = urllib.parse.quote(url, safe="")
        target = http_json(f"http://127.0.0.1:{self.port}/json/new?{q}", method="PUT")
        self.ws = websocket.create_connection(target["webSocketDebuggerUrl"],
                                              timeout=120)
        self.send("Page.enable")
        self.send("Runtime.enable")
        return target

    def send(self, method, params=None):
        self.next_id += 1
        mid = self.next_id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result")
            # events are ignored (we poll DOM instead of waiting for events)

    def eval(self, expr, await_promise=False):
        r = self.send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": await_promise,
        })
        res = r.get("result", {})
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error: {r['exceptionDetails'].get('text')} :: {expr[:120]}")
        return res.get("value")

    def screenshot(self, path):
        r = self.send("Page.captureScreenshot", {"format": "png"})
        data = base64.b64decode(r["data"])
        with open(path, "wb") as f:
            f.write(data)
        return path


def _port_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def launch_edge(port: int, profile: str, window=(1280, 900)):
    exe = next((p for p in EDGE_CANDIDATES if os.path.exists(p)), None)
    if not exe:
        raise RuntimeError("msedge.exe not found")
    last_err = ""
    for attempt in range(3):
        if not _port_free(port):
            port += 1  # stale headless instance from an earlier run, pick another
        base = [
            exe,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--headless=new",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-sync",
            "--hide-scrollbars",
            f"--window-size={window[0]},{window[1]}",
            "about:blank",
        ]
        proc = subprocess.Popen(base, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        cdp = Cdp(port)
        try:
            cdp.wait_ready(timeout=20)
            return proc, cdp
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            proc.terminate()
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:
                proc.kill()
                out, err = "", ""
            print(f"[edge] launch attempt {attempt + 1} failed: {e}; "
                  f"rc={proc.returncode} stderr={err[-400:]!r}")
            time.sleep(0.5)
    raise RuntimeError(f"Edge CDP endpoint did not come up: {last_err}")


def wait_until(cdp, expr, timeout=20, interval=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cdp.eval(expr):
            return True
        time.sleep(interval)
    return False


JS_TEXTAREA = """(() => {
  const t = document.querySelector('#msg-input');
  if (!t) return false;
  t.focus();
  return document.activeElement === t;
})()"""

JS_SEND_RECT = """(() => {
  const b = document.querySelector('button[aria-label="Send message"]');
  if (!b) return null;
  b.scrollIntoView({block: 'center'});
  const r = b.getBoundingClientRect();
  return {x: r.left + r.width / 2, y: r.top + r.height / 2};
})()"""

JS_IS_GENERATING = """(() => {
  // while generating the send button is replaced by the stop button
  return !!document.querySelector('button[aria-label="Stop"], button[aria-label="Stop generating"]');
})()"""

JS_LIST_TEXT = """(() => {
  const el = document.querySelector('#messages-list');
  return el ? el.innerText : '';
})()"""

JS_TITLE = """(() => {
  // conversation names live in the left sidebar list
  const side = document.querySelector('aside') || document.querySelector('[class*="drawer-side"]');
  const txt = side ? side.innerText : document.body.innerText;
  const lines = txt.split('\\n').map(s => s.trim()).filter(Boolean);
  return lines.slice(0, 12);
})()"""


def click(cdp, x, y):
    for kind in ("mousePressed", "mouseReleased"):
        cdp.send("Input.dispatchMouseEvent", {
            "type": kind, "x": x, "y": y, "button": "left", "clickCount": 1,
        })
        time.sleep(0.05)


def run_turn(cdp, text, tag, out_dir, settle=2.0, timeout=600):
    """Type text, click send, wait for the streamed answer, screenshot."""
    # 1) focus + type (real input events, including the IME text path)
    if not wait_until(cdp, JS_TEXTAREA, timeout=20):
        raise RuntimeError("chat textarea #msg-input not found")
    cdp.eval("document.querySelector('#msg-input').select()")
    cdp.send("Input.insertText", {"text": text})
    typed = cdp.eval("document.querySelector('#msg-input').value")
    cdp.screenshot(os.path.join(out_dir, f"{tag}_typed.png"))

    # 2) click the send button (real mouse events); if the DOM is still busy from
    #    the previous turn, fall back to a real Enter key press (webui sends on Enter)
    before = cdp.eval(JS_LIST_TEXT)
    rect = None
    for _ in range(40):
        rect = cdp.eval(JS_SEND_RECT)
        if rect:
            break
        time.sleep(0.25)
    if not rect:
        print("  [webui] send button not found, falling back to Enter key")
        cdp.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter",
                                           "code": "Enter", "windowsVirtualKeyCode": 13})
        cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter",
                                           "code": "Enter", "windowsVirtualKeyCode": 13})
    t0 = time.time()
    if rect:
        click(cdp, rect["x"], rect["y"])
    # Some webui versions do not submit on the synthetic mouse click (focus/overlay
    # quirks).  If the textarea still holds the text, send real Enter key events.
    time.sleep(0.4)
    still_typed = cdp.eval("(() => { const t = document.querySelector('#msg-input');"
                           " return t ? t.value.trim().length > 0 : false; })()")
    if still_typed:
        print("  [webui] click did not submit, sending Enter key")
        for kind in ("keyDown", "keyUp"):
            cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter",
                                               "code": "Enter",
                                               "windowsVirtualKeyCode": 13})
        time.sleep(0.4)

    # 3) poll: wait for generation to start, first assistant character, completion.
    #    The turn is only "done" once the webui is idle AND we have seen assistant
    #    text: a slow prefill/think phase otherwise looks like a finished turn.
    first_char = None
    last = before
    stable_since = None
    while time.time() - t0 < timeout:
        time.sleep(0.15)
        gen = bool(cdp.eval(JS_IS_GENERATING))
        now = cdp.eval(JS_LIST_TEXT) or ""
        if first_char is None and len(now) > len(before) + len(text) + 2:
            first_char = time.time() - t0
        if now != last:
            last = now
            stable_since = time.time()
        elif (not gen and first_char is not None and stable_since and
              time.time() - stable_since > settle):
            break
    # 3b) a turn that never produced assistant text still has to finish generating
    if first_char is None:
        for _ in range(1200):
            if not cdp.eval(JS_IS_GENERATING):
                break
            time.sleep(0.2)
    wall = time.time() - t0
    # let the answer render completely, then scroll the transcript to the bottom so
    # the screenshot shows the final text instead of a mid-stream frame
    time.sleep(2.0)
    cdp.eval("(() => { const sc = document.querySelector('#chat-scroll, .overflow-y-auto, main');"
             " if (sc) sc.scrollTop = sc.scrollHeight; window.scrollTo(0, document.body.scrollHeight); })()")
    time.sleep(0.5)
    gen = bool(cdp.eval(JS_IS_GENERATING))
    # wait until the webui is really idle again (stop button gone, send button back)
    # so the next turn's typing/click cannot race with the previous generation.
    for _ in range(60):
        busy = bool(cdp.eval(JS_IS_GENERATING))
        has_send = bool(cdp.eval(JS_SEND_RECT))
        if not busy and has_send:
            break
        time.sleep(0.5)
    time.sleep(1.5)
    cdp.screenshot(os.path.join(out_dir, f"{tag}_done.png"))
    title = cdp.eval(JS_TITLE)
    return {"text": text, "first_char_s": first_char, "wall_s": wall,
            "still_generating": gen, "list_text": last, "title_lines": title}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7030")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--messages", default="你好，请用三句话介绍杭州。|再详细一点，重点说说它的历史和文化。|用同样的风格介绍一下苏州。")
    ap.add_argument("--out-dir", default="build/logs/e2e")
    ap.add_argument("--debug-port", type=int, default=9333)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--keep-browser", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    profile = os.path.abspath(os.path.join(out_dir, "edge-profile"))
    import shutil
    shutil.rmtree(profile, ignore_errors=True)
    msgs = [m for m in args.messages.split("|") if m.strip()]
    n_turns = min(args.turns, len(msgs))

    edge, cdp = launch_edge(args.debug_port, profile)
    report = {"url": args.url, "turns": []}
    try:
        cdp.open_page(args.url)
        time.sleep(2.0)
        # force the thought block to render while streaming (webui setting)
        cdp.eval("""(() => {
            const c = JSON.parse(localStorage.getItem('config') || '{}');
            c.showThoughtInProgress = true;
            localStorage.setItem('config', JSON.stringify(c));
            return true;
        })()""")
        cdp.send("Page.navigate", {"url": args.url})
        time.sleep(3.0)
        cdp.screenshot(os.path.join(out_dir, "00_start.png"))
        for i in range(n_turns):
            r = run_turn(cdp, msgs[i], f"{i + 1:02d}", out_dir, settle=args.settle,
                         timeout=args.timeout)
            r["turn"] = i + 1
            report["turns"].append(r)
            print(f"\n--- turn {i + 1}: typed {r['text']!r}")
            print(f"  first_char={r['first_char_s']} wall={r['wall_s']:.1f}s "
                  f"still_generating={r['still_generating']}")
            # show only the freshly added tail (assistant text + thinking)
            tail = r["list_text"]
            idx = tail.rfind(r["text"])
            tail = tail[idx + len(r["text"]):] if idx >= 0 else tail[-1200:]
            print(f"  rendered(tail, first 500): {tail[:500]!r}")
            print(f"  title_lines: {r['title_lines']}")
        report_path = os.path.join(out_dir, "e2e_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nreport: {report_path}")
        print(f"screenshots in {out_dir}")
    finally:
        if not args.keep_browser:
            subprocess.run(["taskkill", "/PID", str(edge.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
