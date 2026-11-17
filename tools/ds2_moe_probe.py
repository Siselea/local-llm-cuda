"""Recompute one MoE layer of the real deepseek2 model in numpy from the engine's
dumped hidden state, and compare (python tools/ds2_moe_probe.py --dump build/ds2dump)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds2_probe import F  # noqa: E402


def silu(x):
    return x / (1.0 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", default="1558")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--dump", default="build/ds2dump")
    ap.add_argument("--router", default="sigmoid")
    ap.add_argument("--norm-topk", type=int, default=0)
    args = ap.parse_args()

    f = F(args.model)
    g = f.g
    hidden = int(g.get("deepseek2.embedding_length"))
    n_expert = int(g.get("deepseek2.expert_count"))
    topk = int(g.get("deepseek2.expert_used_count"))
    p = f"blk.{args.layer}."

    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)
    hn = np.fromfile(args.dump + f"_hn{args.layer}.f32", dtype=np.float32).reshape(T, hidden)
    ref = np.fromfile(args.dump + f"_moe{args.layer}.f32", dtype=np.float32).reshape(T, hidden)

    wr = f.t(p + "ffn_gate_inp.weight")
    logits = (hn @ wr.T).astype(np.float32)
    if args.router == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    else:
        e = np.exp(logits.astype(np.float64) - logits.max(axis=-1, keepdims=True))
        scores = e / e.sum(axis=-1, keepdims=True)

    ge = f.t(p + "ffn_gate_exps.weight")
    ue = f.t(p + "ffn_up_exps.weight")
    de = f.t(p + "ffn_down_exps.weight")
    sg = f.t(p + "ffn_gate_shexp.weight")
    su = f.t(p + "ffn_up_shexp.weight")
    sd = f.t(p + "ffn_down_shexp.weight")

    out = np.zeros((T, hidden), dtype=np.float64)
    route_info = []
    for t in range(T):
        idx = np.argsort(-scores[t])[:topk]
        w = scores[t][idx]
        if args.norm_topk:
            w = w / w.sum()
        route_info.append((idx.tolist(), w.tolist()))
        acc = np.zeros(hidden, dtype=np.float64)
        for k, e in enumerate(idx):
            gt = (hn[t] @ ge[e].T).astype(np.float64)
            ut = (hn[t] @ ue[e].T).astype(np.float64)
            act = silu(gt) * ut
            acc += w[k] * (act @ de[e].T)
        out[t] = acc

    # shared experts are added by the engine inside run_moe as well
    gsh = (hn @ sg.T).astype(np.float64)
    ush = (hn @ su.T).astype(np.float64)
    out += (silu(gsh) * ush) @ sd.T

    num = np.linalg.norm(out - ref)
    den = np.linalg.norm(ref)
    print("routes:", [(i, [round(x, 4) for x in w]) for i, w in route_info])
    print(f"moe layer{args.layer}: rel_l2={num/den:.3e} max_abs={np.abs(out-ref).max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
