# -*- coding: utf-8 -*-
"""Real-browser check: the llama.cpp webui page + a fetch() with thinking + stop.

The stock webui UI has no stop-sequence field, so this drives the *same origin* from
the real page in headless Edge:
  1. open the webui and wait for it to load,
  2. type one normal prompt through the UI (real input events),
  3. from the page itself do `fetch('/v1/chat/completions', {stop:[...],
     chat_template_kwargs:{enable_thinking:true}})` and consume the SSE stream,
  4. assert: HTTP 200, the `</think>` marker is present (thinking block closed by the
     model), the visible content after it is non-empty, the stop string was excluded,
  5. save a screenshot + JSON evidence.

Usage:
  python tools/serve_stop_e2e.py --url http://127.0.0.1:7030 --out-dir build/logs/e2e_stop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from webui_e2e import Cdp, launch_edge, run_turn, wait_until  # noqa: E402


FETCH_JS = r"""
(async () => {
  const body = {
    messages: [{role: 'user', content: '1+1等于几？只回答数字。'}],
    stream: true, temperature: 0,
    max_tokens: 512,
    stop: ['zzz'],
    chat_template_kwargs: {enable_thinking: true},
  };
  const r = await fetch('/v1/chat/completions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const raw = await r.text();
  let text = '';
  const lines = raw.split('\n');
  for (const line of lines) {
    const s = line.trim();
    if (!s.startsWith('data:')) continue;
    const payload = s.slice(5).trim();
    if (!payload || payload === '[DONE]') continue;
    try {
      const j = JSON.parse(payload);
      const ch = (j.choices || [{}])[0];
      const d = ch.delta || {};
      if (d.content) text += d.content;
    } catch (e) {}
  }
  return {status: r.status, text: text};
})()
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7030")
    ap.add_argument("--out-dir", default="build/logs/e2e_stop")
    ap.add_argument("--cdp-port", type=int, default=9341)
    ap.add_argument("--profile",
                    default=os.path.join(tempfile.gettempdir(), "engine_edge_e2e_profile"))
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    args.profile = os.path.abspath(args.profile)
    os.makedirs(args.out_dir, exist_ok=True)
    proc = None
    result = {"ok": False}
    try:
        # Reuse an already-running headless Edge when one listens on --cdp-port
        # (manual launch is more reliable in some environments than spawning Edge
        # from the test process); otherwise launch our own.
        cdp = Cdp(args.cdp_port)
        try:
            cdp.wait_ready(timeout=3)
            print(f"[edge] reusing CDP on {args.cdp_port}")
        except Exception:
            proc, cdp = launch_edge(args.cdp_port, args.profile)
            print(f"[edge] launched headless Edge on port {cdp.port}")
        cdp.open_page(args.url)
        wait_until(cdp, "document.readyState === 'complete'", timeout=60)
        wait_until(cdp, "!!document.querySelector('textarea')", timeout=60)
        cdp.screenshot(os.path.join(args.out_dir, "01_webui.png"))
        # 1) real UI turn (typing + clicking send), proving the page works
        run_turn(cdp, "你好，请只回复：收到", "ui_turn", args.out_dir, settle=2.0,
                 timeout=args.timeout)
        # 2) page-origin fetch with thinking + stop
        t0 = time.time()
        value = cdp.eval(FETCH_JS, await_promise=True)
        elapsed = time.time() - t0
        text = (value or {}).get("text") or ""
        status = (value or {}).get("status")
        after = text.split("</think>", 1)[1] if "</think>" in text else None
        result = {
            "ok": status == 200 and "</think>" in text and bool((after or "").strip())
                  and "zzz" not in text,
            "status": status,
            "elapsed_s": round(elapsed, 2),
            "total_text_len": len(text),
            "after_think_len": len(after or ""),
            "after_think": (after or "")[:80],
            "marker_present": "</think>" in text,
            "stop_excluded": "zzz" not in text,
            "tail": text[-80:],
        }
        cdp.screenshot(os.path.join(args.out_dir, "02_stop_thinking_done.png"))
        with open(os.path.join(args.out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
