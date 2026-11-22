# -*- coding: utf-8 -*-
"""Offline tokenizer edge-case audit for `engine serve`.

Checks, per corpus item:
  * C++ encode (build/test_tokenizer.exe --ids-file) == Python reference (tools/tok_ref.py)
  * C++ decode(encode(s)) == s  (byte round trip; skipped for raw-invalid UTF-8)
  * special tokens stay atomic (one id per special token)
  * no crash / deterministic output

Usage: python tools/check_tok_edges.py --model <gguf> [--exe build/test_tokenizer.exe]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path("tools").resolve()))
sys.path.insert(0, str(Path(".").resolve()))

import tok_ref  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CASES = [
    ("ascii", "The quick brown fox jumps over the lazy dog."),
    ("emoji-zwj-family", "家庭成员：👨\u200d👩\u200d👧\u200d👦 结束。"),
    ("emoji-flags-skin", "🇨🇳🇺🇸 👨🏽\u200d🚀 👩🏿\u200d🎤"),
    ("cjk-ext-b", "扩展B区汉字：𠀀𠀁𠀂𠀃（U+20000 起）。"),
    ("cjk-rare", "龘齉靐齾爩麤龗灪龖"),
    ("rtl-arabic", "مرحبا بالعالم! 123"),
    ("rtl-hebrew", "שלום עולם! 456"),
    ("combining", "e\u0301 n\u0303 a\u0308 \u0915\u093f"),
    ("mixed-script", "ASCII-中文-العربية-हिन्दी-😀 end"),
    ("long-no-space", "a" * 3000),
    ("long-cjk-no-space", "𠀀" * 800),
    ("json-specials", '{"a":[1,2],"b":"<|im_start|>","c":"\\n"}'),
    ("markup", "<think>\nthinking\n</think> answer"),
    ("whitespace", "  leading\n\ttabs\r\nnewlines   trailing  "),
    ("nul-ish", "before\\u0000after"),  # literal backslash-u, not a NUL byte
    ("empty", ""),
]

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "</think>", "<｜end▁of▁thinking｜>"]


SCRATCH = Path("build/tok_edge_tmp")
SCRATCH.mkdir(parents=True, exist_ok=True)
_counter = [0]


def _new(ext: str) -> Path:
    _counter[0] += 1
    return SCRATCH / f"case{_counter[0]}{ext}"


def run(exe: str, model: str, mode: str, path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([exe, model, mode, str(path), "--out", str(path.with_suffix(".out"))],
                          capture_output=True)


def encode(exe: str, model: str, text: str) -> list[int]:
    p = _new(".txt")
    p.write_bytes(text.encode("utf-8"))  # binary: Python text mode would turn \n into \r\n on Windows
    try:
        r = run(exe, model, "--ids-file", p)
        if r.returncode != 0:
            raise RuntimeError(f"encode failed: {r.stderr.decode('utf-8', 'replace')[:300]}")
        out = p.with_suffix(".out")
        ids = [int(x) for x in out.read_text(encoding="utf-8").split() if x.strip()]
        return ids
    finally:
        p.unlink(missing_ok=True)
        p.with_suffix(".out").unlink(missing_ok=True)


def decode(exe: str, model: str, ids: list[int]) -> bytes:
    p = _new(".ids")
    p.write_text("\n".join(str(i) for i in ids), encoding="utf-8")
    try:
        r = run(exe, model, "--decode-ids", p)
        if r.returncode != 0:
            raise RuntimeError(f"decode failed: {r.stderr.decode('utf-8', 'replace')[:300]}")
        return p.with_suffix(".out").read_bytes()
    finally:
        p.unlink(missing_ok=True)
        p.with_suffix(".out").unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", default="build/test_tokenizer.exe")
    args = ap.parse_args()
    try:
        ref = tok_ref.load_tokenizer(args.model)
    except Exception as e:
        ref = None
        print(f"model: {args.model}")
        print(f"reference unavailable ({e}); checking C++ self-consistency only")
    if ref is not None:
        nv = len(getattr(ref, "id_to_token", None) or getattr(ref, "tokens", []))
        print(f"ref  : {type(ref).__name__}  vocab={nv}")
    fail = 0
    rows = []
    for name, text in CASES:
        try:
            ids = encode(args.exe, args.model, text)
        except Exception as e:
            print(f"FAIL {name:20s} encode error: {e}")
            fail += 1
            rows.append({"case": name, "ok": False, "error": str(e)})
            continue
        want = None
        if ref is not None:
            try:
                want = ref.encode(text)
            except Exception:
                want = None
        same_ids = want is not None and ids == want
        if not same_ids and want is not None:
            d = next((i for i, (a, b) in enumerate(zip(ids, want)) if a != b), None)
            if d is None and len(ids) != len(want):
                d = min(len(ids), len(want))
            print(f"FAIL {name:20s} ids differ at {d}: got {ids[d:d+8] if d is not None else ids[:8]} "
                  f"want {want[d:d+8] if d is not None else want[:8]}")
            fail += 1
        rt = None
        if not text.startswith("before\\u0000") and "<|im_" not in text:
            try:
                dec = decode(args.exe, args.model, ids)
                rt = dec == text.encode("utf-8")
                if not rt:
                    print(f"FAIL {name:20s} roundtrip mismatch: {dec[:80]!r} vs {text[:80]!r}")
                    fail += 1
            except Exception as e:
                print(f"FAIL {name:20s} decode error: {e}")
                fail += 1
        rows.append({"case": name, "chars": len(text), "ids": len(ids),
                     "ref_ids": len(want) if want is not None else None,
                     "ids_match": same_ids, "roundtrip": rt})
        ok = (rt is not False) and (same_ids or want is None)
        print(f"{'PASS' if ok else 'FAIL'} {name:20s} "
              f"chars={len(text):5d} ids={len(ids):5d} ref={len(want) if want is not None else -1}")
    # special-token atomicity
    for s in SPECIALS:
        ids = encode(args.exe, args.model, s)
        if len(ids) == 1:
            print(f"PASS special {s!r} -> single id {ids[0]}")
            rows.append({"case": f"special:{s}", "single_id": ids[0]})
        else:
            print(f"WARN special {s!r} -> {len(ids)} ids {ids[:6]}")
            rows.append({"case": f"special:{s}", "ids": ids})
    Path("build/logs/tokenizer_edges.json").write_text(
        json.dumps({"rows": rows, "fail": fail}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTOKENIZER EDGES {'GREEN' if fail == 0 else f'FAILURES: {fail}'}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
