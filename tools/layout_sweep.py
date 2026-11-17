"""Compare our engine's next-token choice against llama-server for several layouts.

For each prompt: llama-server gives the token ids and its greedy next token; we run
the engine with the same ids ([BOS] + ids) for every layout and report whether the
top-1 token matches llama's continuation text.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_compare import detokenize  # noqa: E402
from pyref.gguf_parse import read_gguf  # noqa: E402

PROMPTS = [
    "def fibonacci(n):",
    "The capital of France is",
    "import numpy as np",
    "def main():",
    "for i in range(10):",
]


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8793)
    ap.add_argument("--engine", default="build/test_gen.exe")
    ap.add_argument("--layouts", default="perhead/global,perhead/perhead,global/global,global/perhead")
    args = ap.parse_args()

    g = read_gguf(args.model)
    vocab = g.get("tokenizer.ggml.tokens")
    base = f"http://127.0.0.1:{args.port}"
    layouts = [l.split("/") for l in args.layouts.split(",")]

    rows = []
    for text in PROMPTS:
        ids = post(base + "/tokenize", {"content": text})["tokens"]
        ref = post(base + "/completions", {"prompt": text, "max_tokens": 1,
                                           "temperature": 0.0, "cache_prompt": False})
        ref_text = ref["content"]
        entry = {"prompt": text, "ids": ids, "ref": ref_text, "ours": {}}
        for q, k in layouts:
            env = {"ENGINE_MLA_Q_LAYOUT": q, "ENGINE_MLA_KV_LAYOUT": k, "ENGINE_TOPK": "1"}
            out = subprocess.run([args.engine, args.model,
                                  ",".join(["100000"] + [str(i) for i in ids]), "1"],
                                 capture_output=True, text=True,
                                 env={**__import__("os").environ, **env})
            if out.returncode != 0:
                entry["ours"][f"{q}/{k}"] = "ERR"
                continue
            line = [l for l in out.stdout.splitlines() if l.startswith("topk:")][0]
            top1 = int(line.split()[1].split("(")[0])
            entry["ours"][f"{q}/{k}"] = detokenize(vocab, [top1])
        rows.append(entry)

    keys = [f"{q}/{k}" for q, k in layouts]
    print(f"{'prompt':<28}{'llama':<14}" + "".join(f"{k:<22}" for k in keys))
    for r in rows:
        def safe(x: str) -> str:
            return x.encode("unicode_escape").decode("ascii")
        print(f"{r['prompt'][:27]:<28}{safe(r['ref']):<14}" +
              "".join(f"{safe(r['ours'][k]):<22}" for k in keys))
    print()
    for k in keys:
        hits = sum(1 for r in rows if r["ours"][k] == r["ref"])
        print(f"{k}: {hits}/{len(rows)} exact next-token match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
