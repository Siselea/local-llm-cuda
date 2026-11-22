"""Single-token hypothesis grid for Gemma4 semantics (vs llama.cpp oracle).

With a ONE-token prompt the attention part is trivial (softmax over one key), so
the mismatch is in: embedding scale, RMSNorm convention, V construction for
k_eq_v layers, sandwich norms, layer_output_scale, MLP activation, final norm,
tied head and logit softcap.  This script runs a small grid of hypotheses in ONE
pass over the 48 layers (weights dequantised once per layer and reused) and ranks
each variant by how close its top-k log-probs are to llama.cpp.

Usage: python tools/gemma4_grid.py --prompt Hello
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma4_ref import GGUFTensors, Config, post, rms_norm, rope, gelu, default_gemma_model  # noqa: E402

SERVER = "http://127.0.0.1:8099"
TOP = 10


def oracle(prompt: str, n: int = 1) -> dict:
    ids = post("/tokenize", {"content": prompt})["tokens"]
    r = post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0,
                             "n_probs": TOP, "cache_prompt": False})
    return {"ids": ids, "step0": r["completion_probabilities"][0]["top_logprobs"]}


def variants_targeted() -> list[dict]:
    """Round 2: attention-output construction for T=1 (V source + head mapping)."""
    base = dict(norm="direct", embed="sqrt", act="gelu", vnorm="on",
                vkind="rms_raw", vmap="repeat", sandwich="on", scalar="on",
                softcap="on")
    out = []
    for vnorm in ("on", "off"):
        for vkind in ("rms_raw", "raw", "roped"):
            for vmap in ("repeat", "block"):
                v = dict(base)
                v.update(vnorm=vnorm, vkind=vkind, vmap=vmap)
                out.append(v)
    return out


def variants() -> list[dict]:
    base = dict(norm="direct", embed="sqrt", act="gelu", vnorm="on",
                vkind="rms_raw", sandwich="on", scalar="on", softcap="on")
    out = [base]
    for key, vals in [("norm", ["plus1"]), ("embed", ["none"]), ("act", ["silu"]),
                      ("vnorm", ["off"]), ("vkind", ["raw", "roped"]),
                      ("sandwich", ["off"]), ("scalar", ["off"]),
                      ("softcap", ["off"])]:
        v = dict(base)
        v[key] = vals[0]
        out.append(v)
    # a few promising combinations
    for combo in [
        dict(norm="plus1", embed="none"),
        dict(norm="plus1", vnorm="off"),
        dict(norm="direct", embed="none", vnorm="off"),
        dict(norm="plus1", embed="sqrt", act="silu"),
    ]:
        v = dict(base)
        v.update(combo)
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--model", default=str(default_gemma_model()))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--top", type=int, default=8)
    opts = ap.parse_args()
    torch.set_num_threads(opts.threads)

    o = oracle(opts.prompt)
    ids = o["ids"]
    print(f"oracle top ids {[t['id'] for t in o['step0'][:5]]} "
          f"logp {[round(t['logprob'],3) for t in o['step0'][:5]]}", flush=True)

    ts = GGUFTensors(Path(opts.model))
    cfg = Config(ts)
    vars_ = variants_targeted()[: opts.top]
    print(f"{len(vars_)} variants, arch={cfg.arch} layers={cfg.n_layer}", flush=True)

    # state per variant: [hidden]
    x = ts.embed_rows(ids)[0].float()  # [hidden] raw embedding
    states = []
    for v in vars_:
        e = x * math.sqrt(cfg.hidden) if v["embed"] == "sqrt" else x.clone()
        states.append(e)

    t0 = time.time()
    for i in range(cfg.n_layer):
        lp = cfg.per_layer(i)
        d, n_h, n_kv = lp["d"], cfg.n_head, lp["n_kv"]
        rep = n_h // n_kv
        pre = f"blk.{i}."
        Ts = ts.get_t
        Wq = Ts(pre + "attn_q.weight")
        Wk = Ts(pre + "attn_k.weight")
        Wo = Ts(pre + "attn_output.weight")
        Wg = Ts(pre + "gate.weight") if False else Ts(pre + "ffn_gate.weight")
        Wu = Ts(pre + "ffn_up.weight")
        Wd = Ts(pre + "ffn_down.weight")
        an = ts.get(pre + "attn_norm.weight")
        fn = ts.get(pre + "ffn_norm.weight")
        pan = ts.get(pre + "post_attention_norm.weight") if ts.has(pre + "post_attention_norm.weight") else None
        pfn = ts.get(pre + "post_ffw_norm.weight") if ts.has(pre + "post_ffw_norm.weight") else None
        qn = ts.get(pre + "attn_q_norm.weight") if ts.has(pre + "attn_q_norm.weight") else None
        kn = ts.get(pre + "attn_k_norm.weight") if ts.has(pre + "attn_k_norm.weight") else None
        lscale = (float(ts.get(pre + "layer_output_scale.weight")[0])
                  if ts.has(pre + "layer_output_scale.weight") else 1.0)
        has_v = ts.has(pre + "attn_v.weight")
        Wv = Ts(pre + "attn_v.weight") if has_v else None
        kd = n_kv * d
        new_states = []
        for v, xv in zip(vars_, states):
            plus1 = v["norm"] == "plus1"
            xn = rms_norm(xv.unsqueeze(0), torch.from_numpy(an), cfg.eps, plus1)[0]
            k_raw = (Wk @ xn).view(n_kv, d)
            if has_v:
                vv = (Wv @ xn).view(n_kv, d)
                if v["vnorm"] == "on":
                    vv = rms_norm(vv, None, cfg.eps)
            else:
                if v["vkind"] == "raw":
                    vv = k_raw
                elif v["vkind"] == "rms_raw":
                    vv = rms_norm(k_raw, None, cfg.eps)
                else:  # roped: k_norm + rope on raw K, then use as V
                    kk = rms_norm(k_raw, torch.from_numpy(kn), cfg.eps,
                                  v["norm"] == "plus1")
                    kk = rope(kk.unsqueeze(0), torch.tensor([0]), d, lp["theta"], True)[0]
                    vv = kk
            if v["vmap"] == "repeat":
                ctx = vv.repeat_interleave(rep, dim=0).reshape(-1)  # h/rep
            else:
                idx = torch.arange(n_h) % n_kv                        # block mapping
                ctx = vv[idx].reshape(-1)
            o_attn = Wo @ ctx
            if v["sandwich"] == "on" and pan is not None:
                o_attn = rms_norm(o_attn.unsqueeze(0), torch.from_numpy(pan), cfg.eps,
                                  plus1)[0]
            x2 = xv + o_attn
            hn = rms_norm(x2.unsqueeze(0), torch.from_numpy(fn), cfg.eps, plus1)[0]
            g = Wg @ hn
            u = Wu @ hn
            act = gelu(g) if v["act"] == "gelu" else torch.nn.functional.silu(g)
            mlp = Wd @ (act * u)
            if v["sandwich"] == "on" and pfn is not None:
                mlp = rms_norm(mlp.unsqueeze(0), torch.from_numpy(pfn), cfg.eps, plus1)[0]
            x3 = x2 + mlp
            if v["scalar"] == "on":
                x3 = x3 * lscale
            new_states.append(x3)
        states = new_states
        ts.drop()
        if (i + 1) % 8 == 0:
            print(f"  layer {i+1}/{cfg.n_layer} done ({time.time()-t0:.1f}s)", flush=True)

    on = ts.get("output_norm.weight")
    top = o["step0"]
    target_ids = [t["id"] for t in top]
    print("\n== ranking (lower max|dlogp| is better; argmax must match) ==", flush=True)
    rows = []
    for v, xv in zip(vars_, states):
        xf = rms_norm(xv.unsqueeze(0), torch.from_numpy(on), cfg.eps,
                      v["norm"] == "plus1")[0]
        logits = ts.embed_logits(xf.unsqueeze(0))[0]
        if v["softcap"] == "on" and cfg.softcap:
            logits = torch.tanh(logits / cfg.softcap) * cfg.softcap
        lp = torch.log_softmax(logits.float(), dim=-1)
        diffs = [abs(float(lp[t["id"]]) - t["logprob"]) for t in top]
        ref_top = int(torch.topk(lp, 1).indices[0])
        rows.append((max(diffs), ref_top == target_ids[0], v, ref_top))
    rows.sort(key=lambda r: (not r[1], r[0]))
    for maxdiff, match, v, ref_top in rows:
        print(f"max|dlogp|={maxdiff:7.3f} argmax={'MATCH' if match else 'miss'} "
              f"ref={ref_top} {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
