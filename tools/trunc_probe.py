"""Layer-wise semantic probe on truncated real gemma4 checkpoints (CPU only).

Builds/uses a truncated real checkpoint (tools/make_trunc_gemma4.py, block_count=N,
only the first N real layers, global tensors kept) and compares llama.cpp CPU
/embedding (hidden after N layers) against our exact f32 reference
(tools/trunc_bisect.py:ref_forward, dequantised weights, f32 math).

At N=1 only the first layer (a sliding-attention layer) runs, so rope at position
> 0 and the proportional-frequency-factor rule are not involved: a mismatch here
is an embedding / RMSNorm / V-norm / attention-structure difference.
Later N include a full-attention layer (N>=6 for gemma4-12b) and then the rope
freq factors matter at positions > 0.

Usage: python tools/trunc_probe.py --model build/fixtures/gemma4_trunc1.gguf --n 1 \
           [--port 8091] [--ids 9259;3,5,7,9,11,13,4,6]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma4_ref import GGUFTensors, Config  # noqa: E402
from trunc_bisect import ref_forward  # noqa: E402

LLAMA = Path("U:/llama/llama-server.exe")


def post(port: int, path: str, obj: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}" + path,
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def start_server(port: int, model: Path) -> subprocess.Popen:
    p = subprocess.Popen(
        [str(LLAMA), "-m", str(model), "--ctx-size", "128", "-ngl", "0",
         "--host", "127.0.0.1", "--port", str(port), "--temp", "0",
         "--top-k", "1", "--parallel", "1", "--embeddings"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit("llama-server died")
            time.sleep(0.5)
    p.kill()
    raise SystemExit("llama-server not ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="build/fixtures/gemma4_trunc1.gguf")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ids", default="9259;3,5,7,9,11,13,4,6")
    opts = ap.parse_args()
    model = Path(opts.model)
    prompts = [[int(x) for x in grp.split(",") if x.strip()]
               for grp in opts.ids.split(";")]
    torch.set_num_threads(8)

    server = start_server(opts.port, model)
    try:
        oracle = []
        for ids in prompts:
            e = post(opts.port, "/embedding", {"content": ids})
            oracle.append(torch.from_numpy(np.array(e[0]["embedding"], dtype=np.float32)))
        try:
            gen = post(opts.port, "/completion", {"prompt": prompts[-1], "n_predict": 2,
                                                 "temperature": 0.0, "n_probs": 3,
                                                 "cache_prompt": False})
            steps = gen.get("completion_probabilities") or []
            print("llama gen (last prompt):", [s["id"] for s in steps],
                  "top0:", [(t["id"], round(t["logprob"], 4))
                            for t in (steps[0]["top_logprobs"][:3] if steps else [])])
        except Exception as exc:
            print("llama gen unavailable:", exc)
    finally:
        server.kill()
        server.wait()

    ts = GGUFTensors(model)
    cfg = Config(ts)
    print(f"trunc model: layers={cfg.n_layer} swa={cfg.is_swa} "
          f"kv={cfg.n_kv} d={cfg.head_dim} rope_n={cfg.rope_n}")
    fails = 0
    checks = 0
    for ids, target in zip(prompts, oracle):
        ref = ref_forward(ts, cfg, ids, cfg.n_layer)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        rel = float((ref - target).norm() / target.norm())
        rel_last = float((ref[-1] - target[-1]).norm() / target[-1].norm())
        print(f"ids={ids} hidden {tuple(target.shape)} relL2={rel:.4e} "
              f"last-row={rel_last:.4e} llama_norm={float(target[-1].norm()):.2f} "
              f"ref_norm={float(ref[-1].norm()):.2f}")
        checks += 1
        if not (rel < 2e-2):
            fails += 1
    print(f"TRUNC PROBE N={cfg.n_layer}: {checks} prompts, {fails} above 2e-2 relL2")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
