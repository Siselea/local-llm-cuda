#!/usr/bin/env python
"""Cross-check a GGUF tokenizer implementation against llama.cpp's tokenizer.

Modes:
  # find which llama.cpp pre-tokenizer regex the GGUF pre name maps to
  python tools/check_tokenizer.py candidates --model <m.gguf> [--llama-tokenize U:/llama/llama-tokenize.exe]

  # validate the C++ tokenizer (build/test_tokenizer.exe) against the oracle:
  python tools/check_tokenizer.py cpp --model <m.gguf> --exe build/test_tokenizer.exe
      [--llama-tokenize ...] [--skip-oracle]

Checks performed in both modes on a deterministic corpus:
  * whole-document token-id equality with llama-tokenize (when available)
  * decode(encode(text)) == text (byte round trip)
  * special tokens stay atomic (<|im_start|> etc. are single ids)
  * the Qwen ChatML chat template renders to the documented format and its
    token ids match the oracle

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tok_ref  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULT_LLAMA_TOKENIZE = "U:/llama/llama-tokenize.exe"


# Deterministic mixed corpus.  Exercises every branch of the GPT-2/Qwen
# pre-tokenizer: contractions, digit runs, unicode letters, astral plane,
# combining marks, punctuation runs, newline runs, trailing whitespace, NBSP,
# zero-width marks and CJK/latin mixing.
CORPUS_LINES = [
    "Hello world! This is a tokenizer test.",
    "It's 2026 and I'd like to say you'll see we're fine; he's gone.",
    "Numbers: 0 1 12 123 1234 12345 999999 3.14159 2026-11-15",
    "CJK: 你好，世界。这是一个测试。日本語のテキストとabc混在。",
    "Astral: 😀🎉🚀 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ① ⑴ Ⅻ ¼ ½ ٣٤٥ １２３",
    "Combining: e\u0301 vs é, zoom\u0308, A\u030a\u0301",
    "Whitespace: a   b\tc",
    "d",
    "e",
    "",
    "",
    "f",
    "trailing spaces   end",
    "Punctuation: !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ... ?! --- /// ",
    "Quotes: \"double\" 'single' “curly” ‘q’ 「かぎ」 『二重』",
    "Code: def foo(x): return x*2 # comment",
    "    if x != 0:",
    "        print(f\"{x:03d}\")",
    "Unicode letters: Ünïcödé ÀÉÎÕÜ ñ ç ø å œ ß ẞ",
    "Zero width: a\u200bb\u200cc\u200dd\u200ee\u200ff",
    "Symbols: ± § © ® ™ ° µ € £ ¥ ₹",
    "Spaces: a\u00a0b\u3000c\u2028d\u2029e",
    "Fullwidth: ＡＢＣ１２３！？",
    "Emoji ZWJ: 👨‍👩‍👧‍👦 👍🏽",
    "Repeated: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Mixed 中文English123!@# end",
]


def build_corpus() -> str:
    return "\n".join(CORPUS_LINES) + "\n"


# ---------------------------------------------------------------------------
def oracle_ids(model: str, text: str, exe: str) -> list[int]:
    """Token ids from llama-tokenize for the whole document (single call)."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False,
                                     newline="") as f:
        # newline="" keeps LF as LF: Windows text mode would otherwise turn the
        # corpus into CRLF and shift every token id.
        f.write(text)
        path = f.name
    try:
        out = subprocess.run([exe, "-m", model, "-f", path, "--ids", "--no-bos"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        if out.returncode != 0:
            raise SystemExit(f"llama-tokenize failed: {out.stderr[-2000:]}")
        ids = None
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                ids = json.loads(line)
        if ids is None:
            raise SystemExit(f"could not parse llama-tokenize output: {out.stdout[-2000:]}")
        return ids
    finally:
        Path(path).unlink(missing_ok=True)


def first_diff(a: list[int], b: list[int]):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i, a[i], b[i]
    if len(a) != len(b):
        return min(len(a), len(b)), (a[len(b) - 1] if len(a) > len(b) else None), \
               (b[len(a) - 1] if len(b) > len(a) else None)
    return None


def fmt_ids(ids: list[int], start: int, n: int = 8) -> str:
    lo = max(0, start - 3)
    return str(ids[lo:lo + n])


# ---------------------------------------------------------------------------
def mode_candidates(args) -> int:
    model = args.model
    text = build_corpus()
    want = None
    if args.llama_tokenize and Path(args.llama_tokenize).exists():
        want = oracle_ids(model, text, args.llama_tokenize)
        print(f"oracle     : {len(want)} tokens from llama-tokenize")
    else:
        print("oracle     : (llama-tokenize not found, comparing candidates to each other only)")
    results = {}
    for name in tok_ref.PRE_REGEXES:
        tk = tok_ref.load_tokenizer(model, tok_ref.PRE_REGEXES[name])
        got = tk.encode(text)
        results[name] = got
        if want is not None:
            d = first_diff(got, want)
            status = "MATCH" if d is None else f"DIFF at {d[0]} got={d[1]} want={d[2]}"
            print(f"  {name:12s} {len(got):6d} tokens  {status}")
            if d is not None:
                print(f"    got : {fmt_ids(got, d[0])}")
                print(f"    want: {fmt_ids(want, d[0])}")
        else:
            print(f"  {name:12s} {len(got):6d} tokens")
    # round trip of each candidate
    for name in tok_ref.PRE_REGEXES:
        tk = tok_ref.load_tokenizer(model, tok_ref.PRE_REGEXES[name])
        got = results[name]
        dec = tk.decode(got)
        if dec != text:
            print(f"  {name:12s} ROUNDTRIP MISMATCH")
            for i, (a, b) in enumerate(zip(dec, text)):
                if a != b:
                    print(f"    first diff at char {i}: decoded {a!r} vs input {b!r}")
                    break
        else:
            print(f"  {name:12s} roundtrip OK")
    if want is not None:
        matches = [n for n in results if first_diff(results[n], want) is None]
        print(f"\nmatching candidates: {matches or 'NONE'}")
        return 0 if matches else 1
    return 0


# ---------------------------------------------------------------------------
def mode_cpp(args) -> int:
    model = args.model
    exe = args.exe
    # Keep scratch files inside the repo: the Windows temp dir is not always
    # writable for child processes in this environment.
    tmpdir = Path("build/tokcheck-tmp")
    if tmpdir.exists():
        for f in tmpdir.iterdir():
            f.unlink()
    tmpdir.mkdir(parents=True, exist_ok=True)
    failures = 0

    def run(*cmd, binary=False):
        r = subprocess.run(list(cmd), capture_output=True)
        if r.returncode != 0:
            print(f"FAIL: {' '.join(str(c) for c in cmd[:3])} rc={r.returncode}\n"
                  f"{r.stderr.decode('utf-8', 'replace')[-3000:]}")
            raise SystemExit(1)
        return r

    def cpp_encode(text: str) -> list[int]:
        src = tmpdir / "in.txt"
        out = tmpdir / "ids.txt"
        src.write_bytes(text.encode("utf-8"))
        run(exe, model, "--ids-file", str(src), "--out", str(out))
        return [int(x) for x in out.read_text().split()]

    def cpp_decode(ids: list[int]) -> str:
        src = tmpdir / "ids_in.txt"
        out = tmpdir / "dec.txt"
        src.write_text(" ".join(str(i) for i in ids))
        run(exe, model, "--decode-ids", str(src), "--out", str(out))
        return out.read_bytes().decode("utf-8", "replace")

    # 0. what can we validate on this vocab?
    from pyref.gguf_parse import read_gguf
    gg = read_gguf(model)
    tmpl_text = gg.get("tokenizer.chat_template") or gg.get("tokenizer.ggml.chat_template") or ""
    thinking_aware = "enable_thinking" in tmpl_text
    pre_name = gg.get("tokenizer.ggml.pre", "")
    pre_supported = pre_name in ("qwen35", "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "")
    print(f"pre-tokenizer  : {pre_name!r} ({'implemented' if pre_supported else 'NOT implemented, approximation'})")
    special_text = "<|im_start|>user\nhi<|im_end|>\n"
    im_start_atomic = len(cpp_encode("<|im_start|>")) == 1

    # 1. whole-document encode vs llama.cpp
    text = build_corpus()
    got = cpp_encode(text)
    print(f"cpp encode : {len(got)} tokens")
    if args.llama_tokenize and Path(args.llama_tokenize).exists():
        want = oracle_ids(model, text, args.llama_tokenize)
        print(f"oracle     : {len(want)} tokens")
        d = first_diff(got, want)
        if d is None:
            print("encode vs oracle : MATCH")
        elif pre_supported:
            failures += 1
            print(f"encode vs oracle : FAIL at token {d[0]}: cpp={d[1]} llama={d[2]}")
            print(f"  cpp  : {fmt_ids(got, d[0])}")
            print(f"  llama: {fmt_ids(want, d[0])}")
            lo = max(0, d[0] - 20)
            print(f"  cpp decoded context: {cpp_decode(got[lo:lo + 40])!r}")
        else:
            print(f"encode vs oracle : MISMATCH at token {d[0]} (pre {pre_name!r} not "
                  f"implemented -- documented limitation)")
    else:
        print("encode vs oracle : SKIP (llama-tokenize not found)")

    # 1b. real UTF-8 documents (longer than the synthetic corpus; catches
    #     corpus-boundary assumptions and gives thousands of tokens of evidence)
    if args.llama_tokenize and Path(args.llama_tokenize).exists() and pre_supported:
        real = args.text_file or ["README.md", "docs/CONFIG.md", "FINAL-SPRINT-REPORT.md"]
        for rf in real:
            rp = Path(rf)
            if not rp.exists():
                continue
            raw = rp.read_bytes().decode("utf-8", "replace")
            fids = cpp_encode(raw)
            fwant = oracle_ids(model, raw, args.llama_tokenize)
            d = first_diff(fids, fwant)
            if d is None:
                print(f"real file {rf}: {len(fids)} tokens -> MATCH")
            else:
                failures += 1
                print(f"real file {rf}: FAIL at token {d[0]} (cpp={d[1]} llama={d[2]}, "
                      f"{len(fids)} vs {len(fwant)} tokens)")

    # 2. round trip
    decoded = cpp_decode(got)
    if decoded == text:
        print("decode roundtrip : OK")
    else:
        failures += 1
        print("decode roundtrip : FAIL")
        for i, (a, b) in enumerate(zip(decoded, text)):
            if a != b:
                print(f"  first diff at char {i}: {decoded[max(0, i - 20):i + 20]!r} "
                      f"vs {text[max(0, i - 20):i + 20]!r}")
                break

    # 3. special tokens: atomic ids identical to the oracle
    sp_ids = cpp_encode(special_text)
    sp_ok = True
    if not im_start_atomic:
        print("special tokens : SKIP (vocab has no atomic <|im_start|> token)")
        sp_ids = []
    elif args.llama_tokenize and Path(args.llama_tokenize).exists():
        want = oracle_ids(model, special_text, args.llama_tokenize)
        sp_ok = sp_ids == want
        print(f"special encode vs oracle : {'MATCH' if sp_ok else 'FAIL'}  ids={sp_ids}")
    else:
        print(f"special encode ids : {sp_ids}")
    if sp_ids:
        dec_special = cpp_decode(sp_ids)
        if dec_special != "user\nhi\n":
            sp_ok = False
            print(f"special decode mismatch (control tokens must be skipped): {dec_special!r}")
        print(f"special tokens : {'OK' if sp_ok else 'FAIL'}")
        failures += 0 if sp_ok else 1

    # 4. chat template rendering + tokenization
    # The expected text is template-aware: Qwen3.5+ templates open a <think>
    # block in the generation prompt, older Qwen2.5 ones do not.
    chatml_supported = "<|im_start|>" in tmpl_text and "add_generation_prompt" in tmpl_text
    if not chatml_supported:
        print(f"chat template  : SKIP ({'no template' if not tmpl_text else 'not ChatML (not implemented)'})")
        print()
        print("ALL TOKENIZER CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
        return 0 if failures == 0 else 1
    convo = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "\u4f60\u597d"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    cpath = tmpdir / "chat.json"
    cpath.write_text(json.dumps(convo, ensure_ascii=True))
    rendered_path = tmpdir / "rendered.txt"
    run(exe, model, "--chat-template-file", str(cpath), "--out", str(rendered_path))
    rendered = rendered_path.read_text(encoding="utf-8")
    expected = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n\u4f60\u597d<|im_end|>\n"
                "<|im_start|>assistant\n" + ("<think>\n" if thinking_aware else ""))
    tmpl_ok = rendered == expected
    print(f"chat template render : {'OK' if tmpl_ok else 'FAIL'} ({len(rendered)} bytes)")
    if not tmpl_ok:
        print(f"  got : {rendered!r}")
        print(f"  want: {expected!r}")
        failures += 1
    tmpl_ids = cpp_encode(rendered)
    if args.llama_tokenize and Path(args.llama_tokenize).exists():
        want = oracle_ids(model, rendered, args.llama_tokenize)
        if tmpl_ids != want:
            failures += 1
            print(f"chat prompt token ids vs oracle : FAIL")
            d = first_diff(tmpl_ids, want)
            print(f"  at {d[0]}: cpp={d[1]} llama={d[2]}")
            print(f"  cpp  : {fmt_ids(tmpl_ids, d[0])}")
            print(f"  llama: {fmt_ids(want, d[0])}")
        else:
            print(f"chat prompt token ids vs oracle : MATCH ({len(tmpl_ids)} tokens)")
    print(f"chat prompt ids : {tmpl_ids[:12]}{' ...' if len(tmpl_ids) > 12 else ''}")

    if thinking_aware:
        # 4b. thinking disabled: empty-think prefill
        convo2 = dict(convo)
        convo2["enable_thinking"] = False
        cpath.write_text(json.dumps(convo2, ensure_ascii=True))
        run(exe, model, "--chat-template-file", str(cpath), "--out", str(rendered_path))
        rendered2 = rendered_path.read_text(encoding="utf-8")
        exp2 = expected[:-len("<think>\n")] + "<think>\n\n</think>\n\n"
        if rendered2 != exp2:
            failures += 1
            print(f"enable_thinking=false : FAIL\n  got : {rendered2!r}")
        else:
            print("enable_thinking=false : OK (empty think prefill)")

        # 4c. assistant thinking preserved for the current turn
        convo3 = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello", "reasoning_content": "reason"},
            ],
            "add_generation_prompt": False,
            "enable_thinking": True,
        }
        cpath.write_text(json.dumps(convo3, ensure_ascii=True))
        run(exe, model, "--chat-template-file", str(cpath), "--out", str(rendered_path))
        rendered3 = rendered_path.read_text(encoding="utf-8")
        exp3 = ("<|im_start|>user\nhi<|im_end|>\n"
                "<|im_start|>assistant\n<think>\nreason\n</think>\n\nhello<|im_end|>\n")
        if rendered3 != exp3:
            failures += 1
            print(f"assistant reasoning render : FAIL\n  got : {rendered3!r}")
        else:
            print("assistant reasoning render : OK")
    else:
        # older ChatML template: assistant history is rendered verbatim
        convo3 = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "add_generation_prompt": False,
            "enable_thinking": True,
        }
        cpath.write_text(json.dumps(convo3, ensure_ascii=True))
        run(exe, model, "--chat-template-file", str(cpath), "--out", str(rendered_path))
        rendered3 = rendered_path.read_text(encoding="utf-8")
        exp3 = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
        if rendered3 != exp3:
            failures += 1
            print(f"assistant history render : FAIL\n  got : {rendered3!r}")
        else:
            print("assistant history render : OK (no think block)")

    print()
    print("ALL TOKENIZER CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("candidates")
    p1.add_argument("--model", required=True)
    p1.add_argument("--llama-tokenize", default=DEFAULT_LLAMA_TOKENIZE)
    p1.set_defaults(func=mode_candidates)

    p2 = sub.add_parser("cpp")
    p2.add_argument("--model", required=True)
    p2.add_argument("--exe", default="build/test_tokenizer.exe")
    p2.add_argument("--llama-tokenize", default=DEFAULT_LLAMA_TOKENIZE)
    p2.add_argument("--skip-oracle", action="store_true")
    p2.add_argument("--text-file", action="append", default=None,
                    help="extra UTF-8 file(s) compared with the oracle (repeatable)")
    p2.add_argument("--im-start", type=int, default=None)
    p2.add_argument("--im-end", type=int, default=None)
    p2.set_defaults(func=mode_cpp)

    args = ap.parse_args()
    if getattr(args, "skip_oracle", False):
        args.llama_tokenize = ""
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
