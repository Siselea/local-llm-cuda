#!/usr/bin/env python
"""Dump the tokenizer-relevant GGUF metadata of a model.

Usage:
  python tools/dump_tok.py <model.gguf> [--tokens N] [--json out.json]

Prints every tokenizer.* key (type, length, first values), the special-token
table (token_type != 1), the merge pair count, the pre-tokenizer regex name and
the chat template.  The JSON dump is the reference used by
tests/test_tokenizer.cpp (chat-template expected token ids are produced there).

Reuses pyref/gguf_parse.py, i.e. the same byte-level parser the C++ side
mirrors, so the dump is diff-able with the C++ tokenizer loader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyref.gguf_parse import read_gguf  # noqa: E402

# The Windows console defaults to GBK here; the token dumps contain byte-level
# markers such as U+0120, so force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

# gguf token_type values (llama.cpp enum llama_token_type)
TOKEN_TYPE_NAMES = {
    0: "UNDEFINED", 1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL",
    4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE",
}

TOK_KEYS = [
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.remove_extra_whitespaces",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.seperator_token_id",
    "tokenizer.ggml.separator_token_id",
    "tokenizer.ggml.chat_template",
    "tokenizer.chat_template",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=24, help="how many vocab entries to show")
    ap.add_argument("--json", default="", help="write a machine-readable dump here")
    args = ap.parse_args()

    g = read_gguf(args.model)
    print(f"file        : {g.path}")
    print(f"version     : {g.version}  tensors={len(g.tensors)}  kv={len(g.kv)}")
    print()
    print("== tokenizer keys ==")
    tok_keys = [k for k in g.kv if k.startswith("tokenizer.") or "vocab" in k]
    for k in tok_keys:
        v = g.kv[k]
        if isinstance(v, list):
            head = v[:args.tokens]
            print(f"{k:44s} list[{len(v)}] head={head!r}")
        elif isinstance(v, str):
            if len(v) > 400:
                print(f"{k:44s} str[{len(v)}] = {v[:400]!r} ...")
            else:
                print(f"{k:44s} str[{len(v)}] = {v!r}")
        else:
            print(f"{k:44s} {type(v).__name__} = {v!r}")
    for k in TOK_KEYS:
        if k not in g.kv:
            print(f"{k:44s} <absent>")
    print()

    tokens = g.kv.get("tokenizer.ggml.tokens", []) or []
    scores = g.kv.get("tokenizer.ggml.scores", []) or []
    types = g.kv.get("tokenizer.ggml.token_type", []) or []
    merges = g.kv.get("tokenizer.ggml.merges", []) or []

    print("== vocab ==")
    print(f"n_tokens    : {len(tokens)}")
    print(f"n_scores    : {len(scores)}")
    print(f"n_token_type: {len(types)}")
    print(f"n_merges    : {len(merges)}   head={merges[:8]!r}")
    print()
    print("== special / non-normal tokens (token_type != NORMAL) ==")
    shown = 0
    counts: dict[int, int] = {}
    for i, t in enumerate(types):
        counts[t] = counts.get(t, 0) + 1
        if t != 1 and shown < 40:
            tok = tokens[i] if i < len(tokens) else "<missing>"
            score = scores[i] if i < len(scores) else float("nan")
            print(f"  {i:7d} type={t}({TOKEN_TYPE_NAMES.get(t, '?')}) score={score:.4f} {tok!r}")
            shown += 1
    print(f"  token_type histogram: {dict(sorted(counts.items()))}")
    print()

    print("== first tokens ==")
    for i in range(min(args.tokens, len(tokens))):
        t = tokens[i]
        score = scores[i] if i < len(scores) else float("nan")
        tt = types[i] if i < len(types) else -1
        print(f"  {i:7d} score={score:12.6f} type={tt} {t!r}")
    print()

    # how many tokens contain the byte-level space marker
    print("== byte-level / SPM markers ==")
    print(f"tokens starting with ' ' (U+0120 g-dot) : {sum(t.startswith(chr(0x120)) for t in tokens)}")
    print(f"tokens starting with '▁' (U+2581)      : {sum(t.startswith(chr(0x2581)) for t in tokens)}")
    print(f"tokens '<0xNN>'                        : {sum(t.startswith('<0x') for t in tokens)}")
    for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|end|>", "<|startoftext|>"):
        idx = [i for i, t in enumerate(tokens) if t == marker]
        print(f"token {marker!r}: ids={idx}")
    print()

    tmpl = g.kv.get("tokenizer.chat_template", "")
    print("== chat template ==")
    print(tmpl if tmpl else "<absent>")

    if args.json:
        dump = {
            "path": str(g.path),
            "keys": {k: (v if not isinstance(v, list) or len(v) < 64 else
                         {"len": len(v), "head": v[:64]}) for k, v in g.kv.items()
                     if k.startswith("tokenizer.") or "vocab" in k},
            "n_tokens": len(tokens),
            "n_merges": len(merges),
            "bos": g.kv.get("tokenizer.ggml.bos_token_id"),
            "eos": g.kv.get("tokenizer.ggml.eos_token_id"),
            "token_type_histogram": {str(k): v for k, v in counts.items()},
        }
        Path(args.json).write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
