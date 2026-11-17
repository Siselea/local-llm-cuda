"""Compare our engine's greedy generation against llama-server (llama.cpp) on the
same token ids. llama-server provides the token ids for the prompt text, so this
harness needs no tokenizer implementation on our side."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyref.gguf_parse import read_gguf  # noqa: E402


def post(url: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte->unicode table used by byte-level BPE."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def detokenize(tokens: list[str], ids: list[int]) -> str:
    dec = byte_decoder()
    out = bytearray()
    for i in ids:
        if i < 0 or i >= len(tokens):
            continue
        for ch in tokens[i]:
            out.append(dec.get(ch, ord("?")))
    return bytes(out).decode("utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--engine", default="build/test_gen.exe")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    tok = post(base + "/tokenize", {"content": args.text, "add_special": True})
    ids = tok["tokens"]
    ref = post(base + "/completions",
               {"prompt": args.text, "max_tokens": args.n, "temperature": 0.0,
                "cache_prompt": False})
    ref_text = ref["content"]

    out = subprocess.run([args.engine, args.model, ",".join(map(str, ids)), str(args.n)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print("engine failed:", out.stderr.strip())
        return 2
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("gen:")][0]
    gen_ids = [int(x) for x in line.split()[1:]]

    g = read_gguf(args.model)
    vocab = g.get("tokenizer.ggml.tokens")
    our_text = detokenize(vocab, gen_ids)

    print(f"prompt ids : {ids}")
    print(f"llama.cpp  : {ref_text!r}")
    print(f"ours       : {our_text!r}")
    print(f"ours ids   : {gen_ids}")
    ok = our_text == ref_text
    print("RESULT     :", "MATCH" if ok else "MISMATCH")
    if not ok:
        n = min(len(our_text), len(ref_text))
        for i in range(n):
            if our_text[i] != ref_text[i]:
                print(f"  first diff at char {i}: ours={our_text[i]!r} ref={ref_text[i]!r}")
                break
        print(f"  lengths ours={len(our_text)} ref={len(ref_text)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
