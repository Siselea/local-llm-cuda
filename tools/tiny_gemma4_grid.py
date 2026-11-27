"""Micro-oracle semantics grid for gemma4 using the tiny synthetic GGUF.

llama.cpp loads `build/fixtures/tiny_gemma4/model.gguf` (same metadata structure
and tensor layout as the real gemma4-12b, but 64 hidden / 2 layers), so
architecture semantics can be tested in milliseconds instead of 2.5 minutes.
The oracle (final hidden state after output_norm, from llama-server /embedding
with token ids) is compared against candidate python implementations; the
verified rules are then ported to the engine.

Usage: python tools/tiny_gemma4_grid.py [--ids 5,7,9] [--all-single]
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_ref import GGUFTensors, Config, rms_norm, rope, gelu  # noqa: E402

ORACLE = Path("build/tiny_gemma4-oracle.npz")
MODEL = Path("build/fixtures/tiny_gemma4/model.gguf")


def rmsnorm_ops(x, w, eps, plus1_w):
    var = x.pow(2).mean(-1, keepdim=True) + eps
    y = x * torch.pow(var, -0.5)
    if w is not None:
        y = y * ((1.0 + w) if plus1_w else w)
    return y


def rope_style(x, pos, n_rot, theta, neox=True, ff=None):
    """x [T,H,D]; neox=rotate_half pairs, else GPTJ interleaved pairs.

    ff is the proportional-rope frequency divisor table (GGUF rope_freqs.weight):
    llama.cpp computes theta_i = pos * base^(-2i/n_rot) / ff[i] (the gemma4 full
    layers use it; sliding layers do not).  ff=None is plain full rotary.
    """
    d = x.shape[-1]
    n_rot = min(n_rot, d)
    inv = 1.0 / (theta ** (torch.arange(0, n_rot, 2, dtype=torch.float32) / n_rot))
    if ff is not None:
        n = min(int(ff.numel()), int(inv.numel()))
        inv[:n] = inv[:n] / ff[:n].float()
    ang = pos.float()[:, None] * inv[None, :]
    if neox:
        cos = torch.cat([ang.cos(), ang.cos()], dim=-1)[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], dim=-1)[:, None, :]
        xr = x[..., :n_rot]
        out = torch.cat([xr * cos + rotate_half_last(xr) * sin, x[..., n_rot:]], dim=-1)
        return out
    # GPTJ: pairs (0,1),(2,3),...
    xr = x[..., :n_rot]
    even = xr[..., 0::2]
    odd = xr[..., 1::2]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    new_even = even * c - odd * s
    new_odd = even * s + odd * c
    inter = torch.stack([new_even, new_odd], dim=-1).reshape(*xr.shape)
    return torch.cat([inter, x[..., n_rot:]], dim=-1)


def rotate_half_last(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def forward(ts, cfg, ids, v) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(ids)
    pos = torch.arange(T, dtype=torch.long)
    x = ts.embed_rows(ids)
    if v["embed"] == "sqrt":
        x = x * math.sqrt(cfg.hidden)
    x = x.float()
    for i in range(cfg.n_layer):
        lp = cfg.per_layer(i)
        d, nh, nkv = lp["d"], cfg.n_head, lp["n_kv"]
        rep = nh // nkv
        pre = f"blk.{i}."
        Ts = ts.get_t
        an = ts.get(pre + "attn_norm.weight")
        xn = rmsnorm_ops(x, torch.from_numpy(an), cfg.eps, v["norm_plus1"])
        Wq, Wk = Ts(pre + "attn_q.weight"), Ts(pre + "attn_k.weight")
        Wo = Ts(pre + "attn_output.weight")
        q = (xn @ Wq.T).view(T, nh, d)
        if ts.has(pre + "attn_q_norm.weight"):
            q = rmsnorm_ops(q, Ts(pre + "attn_q_norm.weight"), cfg.eps, v["qk_plus1"])
        ff_tab = None
        if ts.has("rope_freqs.weight") and v.get("rope_ff", "auto") != "ignore":
            cand = ts.get_t("rope_freqs.weight")
            # llama.cpp binds rope_freqs only to full-attention layers and only
            # when the table length matches that layer's n_rot/2.
            if lp["window"] == 0 and cand.numel() * 2 == lp["n_rot"]:
                ff_tab = cand
        q = rope_style(q, pos, lp["n_rot"], lp["theta"], v["rope_neox"], ff_tab)
        has_v = ts.has(pre + "attn_v.weight")
        Wv = Ts(pre + "attn_v.weight") if has_v else None
        if has_v:
            k = (xn @ Wk.T).view(T, nkv, d)
            if ts.has(pre + "attn_k_norm.weight"):
                k = rmsnorm_ops(k, Ts(pre + "attn_k_norm.weight"), cfg.eps, v["qk_plus1"])
            k = rope_style(k, pos, lp["n_rot"], lp["theta"], v["rope_neox"], ff_tab)
            vv = (xn @ Wv.T).view(T, nkv, d)
            if v["vnorm"] == "on":
                vv = rmsnorm_ops(vv, None, cfg.eps, False)
        else:
            k_raw = (xn @ Wk.T).view(T, nkv, d)
            if v["vkind"] == "raw":
                vv = k_raw
            elif v["vkind"] == "rms_raw":
                vv = rmsnorm_ops(k_raw, None, cfg.eps, False)
            else:
                kk = rmsnorm_ops(k_raw, Ts(pre + "attn_k_norm.weight"), cfg.eps,
                                 v["qk_plus1"])
                kk = rope_style(kk, pos, lp["n_rot"], lp["theta"], v["rope_neox"], ff_tab)
                vv = kk
            k = rmsnorm_ops(k_raw, Ts(pre + "attn_k_norm.weight"), cfg.eps, v["qk_plus1"])
            k = rope_style(k, pos, lp["n_rot"], lp["theta"], v["rope_neox"], ff_tab)
        if v["vmap"] == "repeat":
            idx = torch.arange(nh) // rep
        else:
            idx = torch.arange(nh) % nkv
        kk = k[:, idx, :]
        vvv = vv[:, idx, :]
        scale = 1.0 if v["scale"] == "one" else 1.0 / math.sqrt(d)
        scores = torch.einsum("thd,shd->hts", q, kk) * scale
        idxs = torch.arange(T)
        causal = idxs[:, None] >= idxs[None, :]
        if lp["window"]:
            w = lp["window"]
            causal &= (idxs[:, None] - idxs[None, :]) < w if v["win_mode"] == "lt" \
                else (idxs[:, None] - idxs[None, :]) <= w
        scores = scores.masked_fill(~causal[None], float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1)
        ctx = torch.einsum("hts,shd->thd", attn, vvv).reshape(T, nh * d)
        o = ctx @ Wo.T
        if v["sandwich"] and ts.has(pre + "post_attention_norm.weight"):
            o = rmsnorm_ops(o, Ts(pre + "post_attention_norm.weight"), cfg.eps, v["norm_plus1"])
        x = x + o
        fn = ts.get(pre + "ffn_norm.weight")
        hn = rmsnorm_ops(x, torch.from_numpy(fn), cfg.eps, v["norm_plus1"])
        g = hn @ Ts(pre + "ffn_gate.weight").T
        u = hn @ Ts(pre + "ffn_up.weight").T
        act = gelu(g) if v["act"] == "gelu" else torch.nn.functional.silu(g)
        mlp = (act * u) @ Ts(pre + "ffn_down.weight").T
        if v["sandwich"] and ts.has(pre + "post_ffw_norm.weight"):
            mlp = rmsnorm_ops(mlp, Ts(pre + "post_ffw_norm.weight"), cfg.eps, v["norm_plus1"])
        x = x + mlp
        if v["scalar"] == "final":
            x = x * float(ts.get(pre + "layer_output_scale.weight")[0])
        ts.drop()
    x = rmsnorm_ops(x, ts.get_t("output_norm.weight"), cfg.eps, v["norm_plus1"])
    logits = x @ ts.get_t("token_embd.weight").T
    if v["softcap"] and cfg.softcap:
        logits = torch.tanh(logits / cfg.softcap) * cfg.softcap
    return x, logits


def base_variant() -> dict:
    return dict(norm_plus1=False, qk_plus1=False, rope_neox=True, embed="sqrt",
                vnorm="on", vkind="rms_raw", vmap="repeat", scale="one", win_mode="lt",
                act="gelu", sandwich=True, scalar="final", softcap=True,
                rope_ff="auto")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="5,7,9")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--oracle", default=str(ORACLE))
    ap.add_argument("--all-single", action="store_true")
    opts = ap.parse_args()
    ids = [int(x) for x in opts.ids.split(",")]
    o = np.load(ORACLE)
    target = torch.from_numpy(o["hidden"]).float()
    torch.set_num_threads(4)
    ts = GGUFTensors(MODEL)
    cfg = Config(ts)

    variants = [("base", base_variant())]
    for key, vals in [("rope_neox", [False]), ("scale", ["rsqrt"]), ("win_mode", ["le"]),
                      ("qk_plus1", [True]), ("norm_plus1", [True]),
                      ("vnorm", ["off"]), ("vkind", ["raw", "roped"]),
                      ("vmap", ["block"]), ("embed", ["none"]), ("act", ["silu"]),
                      ("sandwich", [False]), ("scalar", ["sub"])]:
        for val in vals:
            v = base_variant()
            v[key] = val
            variants.append((f"{key}={val}", v))

    print(f"target hidden shape {tuple(target.shape)}; "
          f"{len(variants)} variants; ids={o['ids'].tolist()}", flush=True)
    rows = []
    for name, v in variants:
        ts.cache.clear()
        h, logits = forward(ts, cfg, [int(x) for x in o["ids"]], v)
        rel = float((h - target).norm() / target.norm())
        rows.append((rel, name))
    rows.sort()
    for rel, name in rows:
        print(f"relL2={rel:10.3e}  {name}", flush=True)

    # also check logits against llama's top-k logprobs for the best variant
    best = dict(rows[0]) if False else None
    name0 = rows[0][1]
    for n, v in variants:
        if n == name0:
            ts.cache.clear()
            h, logits = forward(ts, cfg, [int(x) for x in o["ids"]], v)
            lp = torch.log_softmax(logits[-1].float(), dim=-1)
            ids_ = o["prob_ids"].tolist()
            dl = [abs(float(lp[i]) - p) for i, p in zip(ids_, o["probs"])]
            print(f"\nbest variant {n}: max|dlogp| over llama top-k = {max(dl):.3e}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
