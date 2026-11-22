"""Layer-wise bisect against llama.cpp on the REAL gemma4-12b checkpoint.

The real checkpoint still differs from our implementation (hidden relL2 ~0.2),
while a tiny synthetic gemma4 matches llama.cpp to 6.7e-5.  To find the first
diverging layer without a per-layer oracle, build a "truncated" copy of the real
GGUF where every layer >= N is an exact identity:

  zero attn_output.weight and ffn_down.weight  -> that layer's attention/MLP
  output is 0 (the residual passes through untouched), and set
  layer_output_scale.weight = 1.0 so the stream is not rescaled.

Zeroing quantized blocks is exact: an all-zero K-quant block dequantises to 0.

Then llama.cpp's /embedding (content = token ids, so no BOS) returns the hidden
state after exactly N real layers; compare it with our python reference forward
stopped after N layers.  Repeat for growing N to localise the divergence.

Usage: python tools/trunc_bisect.py --ns 1,2,5,6,8 --ids 9259
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf
from gemma4_ref import GGUFTensors, Config, rms_norm, rope, gelu  # noqa: E402

SRC = Path("U:/AI/gemma-4-12b-it-Q4_K_M.gguf")
TRUNC = Path("U:/AI/_gemma4_trunc.gguf")
LLAMA = Path("U:/llama/llama-server.exe")
PORT = 8097


def tensor_map(path: Path):
    r = gguf.GGUFReader(str(path), "r")
    return r, {t.name: t for t in r.tensors}, r.data_offset


def zero_range(path: Path, off: int, nbytes: int):
    with open(path, "r+b") as f:
        f.seek(off)
        chunk = bytes(1 << 22)
        left = nbytes
        while left > 0:
            n = min(left, len(chunk))
            f.write(chunk[:n])
            left -= n


def patch_identity(path: Path, tensors: dict, data_offset: int, layer: int, identity: bool,
                   src_path: Path, src_offset: int, src_tensors: dict):
    """Zero (identity=True) or restore (False) the two output projections."""
    for name in (f"blk.{layer}.attn_output.weight", f"blk.{layer}.ffn_down.weight"):
        if name not in tensors:
            continue
        t = tensors[name]
        off = t.data_offset          # absolute file offset (gguf reader adds data section)
        nb = t.n_bytes
        if identity:
            zero_range(path, off, nb)
        else:
            st = src_tensors[name]
            with open(src_path, "rb") as fs, open(path, "r+b") as fd:
                fs.seek(st.data_offset)
                fd.seek(off)
                left = nb
                while left > 0:
                    b = fs.read(min(left, 1 << 22))
                    fd.write(b)
                    left -= len(b)
    # layer output scale: 1.0 when identity, original value otherwise
    name = f"blk.{layer}.layer_output_scale.weight"
    if name in tensors:
        t = tensors[name]
        off = t.data_offset
        if identity:
            with open(path, "r+b") as f:
                f.seek(off)
                f.write(struct.pack("<f", 1.0))
        else:
            with open(src_path, "rb") as fs, open(path, "r+b") as fd:
                fs.seek(src_tensors[name].data_offset)
                fd.seek(off)
                fd.write(fs.read(4))


def start_server(model: Path):
    p = subprocess.Popen(
        [str(LLAMA), "-m", str(model), "--ctx-size", "32", "-ngl", "0",
         "--host", "127.0.0.1", "--port", str(PORT), "--temp", "0", "--top-k", "1",
         "--parallel", "1", "--embeddings"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("llama-server died")
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("llama-server timeout")


def post(path: str, obj: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}" + path,
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))


def ref_forward(ts: GGUFTensors, cfg: Config, ids: list[int], n_layers: int) -> torch.Tensor:
    """Our best candidate math, stopped after n_layers."""
    x = ts.embed_rows(ids) * math.sqrt(cfg.hidden)
    T = len(ids)
    pos = torch.arange(T, dtype=torch.long)
    for i in range(n_layers):
        lp = cfg.per_layer(i)
        d, nh, nkv = lp["d"], cfg.n_head, lp["n_kv"]
        rep = nh // nkv
        pre = f"blk.{i}."
        Ts = ts.get_t
        xn = rms_norm(x, torch.from_numpy(ts.get(pre + "attn_norm.weight")), cfg.eps)
        q = (xn @ Ts(pre + "attn_q.weight").T).view(T, nh, d)
        q = rms_norm(q, Ts(pre + "attn_q_norm.weight"), cfg.eps)
        q = rope(q, pos, d, lp["theta"], True)
        inv = torch.arange(nh) // rep
        k = (xn @ Ts(pre + "attn_k.weight").T).view(T, nkv, d)
        k = rms_norm(k, Ts(pre + "attn_k_norm.weight"), cfg.eps)
        k = rope(k, pos, d, lp["theta"], True)
        if ts.has(pre + "attn_v.weight"):
            v = (xn @ Ts(pre + "attn_v.weight").T).view(T, nkv, d)
            v = rms_norm(v, None, cfg.eps)
        else:
            v = (xn @ Ts(pre + "attn_k.weight").T).view(T, nkv, d)
            v = rms_norm(v, None, cfg.eps)
        kk, vv = k[:, inv, :], v[:, inv, :]
        sc = 1.0
        att = torch.einsum("thd,shd->hts", q, kk) * sc
        idxs = torch.arange(T)
        causal = idxs[:, None] >= idxs[None, :]
        if lp["window"]:
            causal &= (idxs[:, None] - idxs[None, :]) < lp["window"]
        att = torch.softmax(att.masked_fill(~causal[None], float("-inf")).float(), -1)
        ctx = torch.einsum("hts,shd->thd", att, vv).reshape(T, nh * d)
        o = ctx @ Ts(pre + "attn_output.weight").T
        o = rms_norm(o, Ts(pre + "post_attention_norm.weight"), cfg.eps)
        x = x + o
        hn = rms_norm(x, torch.from_numpy(ts.get(pre + "ffn_norm.weight")), cfg.eps)
        g = hn @ Ts(pre + "ffn_gate.weight").T
        u = hn @ Ts(pre + "ffn_up.weight").T
        mlp = (gelu(g) * u) @ Ts(pre + "ffn_down.weight").T
        mlp = rms_norm(mlp, Ts(pre + "post_ffw_norm.weight"), cfg.eps)
        x = (x + mlp) * float(ts.get(pre + "layer_output_scale.weight")[0])
        ts.drop()
    return rms_norm(x, ts.get_t("output_norm.weight"), cfg.eps)


import math  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1,2,5,6,8")
    ap.add_argument("--ids", default="9259")
    opts = ap.parse_args()
    ids = [int(x) for x in opts.ids.split(",")]
    ns = [int(x) for x in opts.ns.split(",")]
    torch.set_num_threads(8)

    src_reader, src_tensors, src_do = tensor_map(SRC)
    # one raw copy, then mutate in place
    if not TRUNC.exists() or TRUNC.stat().st_size != SRC.stat().st_size:
        print(f"copying {SRC} -> {TRUNC} ...", flush=True)
        t0 = time.time()
        shutil.copyfile(SRC, TRUNC)
        print(f"  copied in {time.time()-t0:.1f}s", flush=True)
    tr_reader, tr_tensors, tr_do = tensor_map(TRUNC)
    ts = GGUFTensors(SRC)
    cfg = Config(ts)

    # start from "all layers >= first are identity", then restore upwards
    first = min(ns)
    for i in range(first, cfg.n_layer):
        patch_identity(TRUNC, tr_tensors, tr_do, i, True, SRC, src_do, src_tensors)
    print(f"identity layers {first}..{cfg.n_layer-1} patched", flush=True)

    refs = {n: ref_forward(ts, cfg, ids, n) for n in []}  # placeholder
    results = []
    for n in ns:
        # restore layers between previous n and this n (they are real again)
        for i in range(n):
            patch_identity(TRUNC, tr_tensors, tr_do, i, False, SRC, src_do, src_tensors)
        print(f"N={n}: layers 0..{n-1} real, {n}.. identity; starting llama.cpp", flush=True)
        p = start_server(TRUNC)
        try:
            e = post("/embedding", {"content": ids})
            target = torch.from_numpy(np.array(e[0]["embedding"][-1], dtype=np.float32))
        finally:
            p.kill()
            p.wait()
        ref = ref_forward(ts, cfg, ids, n)
        rel = float((ref - target).norm() / target.norm())
        cos = float(torch.dot(ref, target) / (ref.norm() * target.norm()))
        print(f"  N={n:2d} ref_norm={float(ref.norm()):8.2f} llama_norm={float(target.norm()):8.2f} "
              f"relL2={rel:.3e} cos={cos:.6f}", flush=True)
        results.append((n, rel, cos))
        # keep layers < n real for the next iteration; zero layer n.. again later
        for i in range(n, cfg.n_layer):
            patch_identity(TRUNC, tr_tensors, tr_do, i, True, SRC, src_do, src_tensors)
    print("\n== bisect summary ==")
    for n, rel, cos in results:
        print(f"N={n:2d} relL2={rel:.3e} cos={cos:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
