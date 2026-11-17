#!/usr/bin/env python
"""Reference GGUF tokenizer (Python) used to pin down the C++ implementation.

Two jobs:
  1. experiment with candidate pre-tokenizer regexes (`--candidates`) so the
     C++ splitter can be validated against llama-tokenize before it exists;
  2. provide the reference implementation `Tokenizer` imported by
     tools/check_tokenizer.py and tests/make_tiny_tokenizer.py.

Only the standard library plus `regex` (already installed for the pyref side)
are used.  The GGUF metadata is read with pyref/gguf_parse.py, i.e. the same
byte-level parser the C++ side mirrors.

Supported tokenizer families (tokenizer.ggml.model):
  gpt2  -> byte-level BPE (GPT-2 unicode byte mapping + merges)
  llama -> SentencePiece (▁ word marker, highest-score-first merges, byte fallback)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pyref.gguf_parse import read_gguf  # noqa: E402

TOKEN_TYPE_NAMES = {
    0: "UNDEFINED", 1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL",
    4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE",
}
# token types llama.cpp treats as "special" for encoding / detokenizing
SPECIAL_TYPES = {2, 3, 4}


# ---------------------------------------------------------------------------
# GPT-2 byte <-> unicode mapping (bytes_to_unicode in the GPT-2 BPE paper)
# ---------------------------------------------------------------------------
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE2UNI = _bytes_to_unicode()
UNI2BYTE = {v: k for k, v in BYTE2UNI.items()}


# ---------------------------------------------------------------------------
# candidate pre-tokenizer regexes (llama.cpp LLAMA_VOCAB_PRE_TYPE_*)
# The first one that reproduces llama-tokenize exactly for qwen35 wins; see
# tools/check_tokenizer.py --candidates.
# ---------------------------------------------------------------------------
PRE_REGEXES: dict[str, str] = {
    # llama.cpp "qwen2" / "qwen3": GPT-2 style, one digit per token
    "qwen2": ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
              " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
    # explicit-case contraction variant of the same regex (Qwen2.5 tokenizer.json)
    "qwen2_ci": ("(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
                 "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
                 " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
    # number groups of up to 3 digits
    "n3": ("(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
           "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
           " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
    "n3_ci": ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
              " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
    # deepseek-llm (single digit, explicit contractions)
    "deepseek": ("(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
                 "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
                 " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
}


def _compile(pattern: str):
    return re.compile(pattern)


class BPETokenizer:
    def __init__(self, tokens, token_types, merges, pre: str, pre_regex: str | None = None):
        self.id_to_token = list(tokens)
        self.token_types = list(token_types) if token_types else [1] * len(tokens)
        self.token_to_id: dict[str, int] = {}
        for i, t in enumerate(self.id_to_token):
            # first occurrence wins, like llama.cpp's vocab hash map
            self.token_to_id.setdefault(t, i)
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            left, right = m.split(" ", 1)
            self.ranks[(left, right)] = i
        self.pre = pre
        self.regex = _compile(pre_regex if pre_regex is not None else PRE_REGEXES[pre])

    # -- encode ------------------------------------------------------------
    def _bpe(self, word: str) -> list[str]:
        sym = list(word)
        while len(sym) > 1:
            best, best_rank = -1, None
            for i in range(len(sym) - 1):
                r = self.ranks.get((sym[i], sym[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best < 0:
                break
            sym[best:best + 2] = [sym[best] + sym[best + 1]]
        return sym

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for m in self.regex.finditer(text):
            chunk = m.group()
            for piece in self._bpe("".join(BYTE2UNI[b] for b in chunk.encode("utf-8"))):
                tid = self.token_to_id.get(piece)
                if tid is None:
                    # byte fallback: the single byte of each character
                    for b in piece.encode("utf-8"):
                        ids.append(self.token_to_id["".join(BYTE2UNI[b])])
                else:
                    ids.append(tid)
        return ids

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        if not parse_special:
            return self._encode_ordinary(text)
        specials = sorted(
            ((i, t) for i, t in enumerate(self.id_to_token)
             if self.token_types[i] in SPECIAL_TYPES),
            key=lambda x: -len(x[1]))
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            best_pos, best_id, best_len = -1, -1, 0
            for sid, stok in specials:
                p = text.find(stok, pos)
                if p < 0:
                    continue
                if best_pos < 0 or p < best_pos or (p == best_pos and len(stok) > best_len):
                    best_pos, best_id, best_len = p, sid, len(stok)
            if best_pos < 0:
                ids.extend(self._encode_ordinary(text[pos:]))
                break
            ids.extend(self._encode_ordinary(text[pos:best_pos]))
            ids.append(best_id)
            pos = best_pos + best_len
        return ids

    # -- decode ------------------------------------------------------------
    def _token_bytes(self, tid: int) -> bytes:
        tok = self.id_to_token[tid]
        if self.token_types[tid] in SPECIAL_TYPES:
            return tok.encode("utf-8")
        out = bytearray()
        for ch in tok:
            b = UNI2BYTE.get(ch)
            if b is None:
                out.extend(ch.encode("utf-8"))
            else:
                out.append(b)
        # <0xNN> byte tokens (SPM byte fallback)
        if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
            try:
                return bytes([int(tok[3:5], 16)])
            except ValueError:
                pass
        return bytes(out)

    def decode(self, ids: Iterable[int]) -> str:
        # Concatenate the raw bytes first: a byte-level BPE token may end in the
        # middle of a multi-byte UTF-8 sequence, so per-token decoding is wrong.
        return b"".join(self._token_bytes(i) for i in ids).decode("utf-8", "replace")


class SPMTokenizer:
    """SentencePiece (llama) tokenizer:  ▁ word marker + scores + byte fallback."""

    def __init__(self, tokens, token_types, scores, pre: str):
        self.id_to_token = list(tokens)
        self.token_types = list(token_types) if token_types else [1] * len(tokens)
        self.scores = list(scores) if scores else [0.0] * len(tokens)
        self.token_to_id: dict[str, int] = {}
        for i, t in enumerate(self.id_to_token):
            self.token_to_id.setdefault(t, i)
        self.pre = pre
        self.unk_id = None
        for i, t in enumerate(self.id_to_token):
            if self.token_types[i] == 2:
                self.unk_id = i
                break

    def _tokenize_fragment(self, frag: str) -> list[int]:
        # seed with the best-scoring prefix of every position, then greedy merge
        ids: list[int] = []
        pos = 0
        while pos < len(frag):
            best_id, best_len = None, 0
            for ln in range(len(frag) - pos, 0, -1):
                tid = self.token_to_id.get(frag[pos:pos + ln])
                if tid is not None:
                    best_id, best_len = tid, ln
                    break
            if best_id is None:
                # byte fallback: encode one UTF-8 byte as <0xNN>
                b = frag.encode("utf-8")[pos]
                bt = "<0x%02X>" % b
                if bt in self.token_to_id:
                    ids.append(self.token_to_id[bt])
                    pos += 1
                    continue
                if self.unk_id is not None:
                    ids.append(self.unk_id)
                    pos += 1
                    continue
                raise KeyError(f"cannot encode {frag[pos:pos+1]!r}")
            ids.append(best_id)
            pos += best_len
        return ids

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        ids: list[int] = []
        # naive SPM: split on spaces, prefix each word with ▁ (first word too)
        for i, word in enumerate(text.split(" ")):
            frag = "\u2581" + word if i == 0 else "\u2581" + word
            if frag in ("\u2581",) and not word:
                continue
            ids.extend(self._tokenize_fragment(frag))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        pieces = []
        for i in ids:
            tok = self.id_to_token[i]
            if self.token_types[i] in SPECIAL_TYPES:
                pieces.append(tok)
            elif len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                try:
                    pieces.append(bytes([int(tok[3:5], 16)]).decode("utf-8", "replace"))
                except ValueError:
                    pieces.append(tok)
            else:
                pieces.append(tok)
        return "".join(pieces).replace("\u2581", " ")


# ---------------------------------------------------------------------------
def load_tokenizer(path: str, pre_regex: str | None = None):
    g = read_gguf(path)
    kv = g.kv
    model = kv.get("tokenizer.ggml.model", "")
    tokens = kv.get("tokenizer.ggml.tokens") or []
    types = kv.get("tokenizer.ggml.token_type") or []
    pre = kv.get("tokenizer.ggml.pre", "")
    if model == "gpt2":
        merges = kv.get("tokenizer.ggml.merges") or []
        return BPETokenizer(tokens, types, merges, pre, pre_regex)
    if model in ("llama", "spm"):
        scores = kv.get("tokenizer.ggml.scores") or []
        return SPMTokenizer(tokens, types, scores, pre)
    raise SystemExit(f"unsupported tokenizer model {model!r}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--candidates", action="store_true",
                    help="print token ids for the sample corpus under every candidate regex")
    ap.add_argument("--text", default="")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if args.text:
        tk = load_tokenizer(args.model)
        ids = tk.encode(args.text)
        print("ids :", ids)
        print("dec :", repr(tk.decode(ids)))
        return 0
    corpus = [
        "Hello world!",
        "It's 2026.\n\n  Leading and trailing  ",
        "def f(x):\n    return x * 2  # comment\n",
        "你好，世界。这是一个测试。",
        "2026年11月15日 12:34:56",
        "a1b2c3 1 12 123 1234 12345",
        "    indent\t\ttab",
        "emoji 😀🎉 test",
        "「引用」『单引号』… ——",
    ]
    for name, pat in PRE_REGEXES.items():
        tk = load_tokenizer(args.model, pat)
        print(f"== {name}")
        for c in corpus:
            print("  ", tk.encode(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
