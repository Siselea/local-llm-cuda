"""Rank Gemma4 semantic hypotheses by matching llama.cpp's final hidden state.

llama-server `/embedding` (with `content` given as token ids, so no BOS is
prepended) returns the per-token final hidden state *after* output_norm for a
causal LM.  That vector is a much sharper oracle than the top-k logits: it
isolates the whole 48-layer stack (norms, V construction, residual scheme,
sandwich norms, layer scalar, embedding scale) from the tied lm_head/softcap.

For a 1-token prompt the attention output is exactly V, so this also pins down
the V construction rules quickly (the q/k path does not affect it).

Usage: python tools/gemma4_hidden_grid.py --prompt Hello
"""

from __future__ import annotations

import argparse
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


def fetch_oracle(prompt: str) -> dict:
    ids = post("/tokenize", {"content": prompt})["tokens"]
    c = post("/completion", {"prompt": ids, "n_predict": 4, "temperature": 0.0,
                             "n_probs": 10, "cache_prompt": False})
    e = post("/embedding", {"content": ids})
    hidden = np.array(e[0]["embedding"], dtype=np.float32)   # [T, hidden]
    return {"ids": ids, "hidden": hidden,
            "probs": c["completion_probabilities"][0]["top_logprobs"],
            "gen": [x["id"] for x in c["completion_probabilities"]]}


def variants() -> list[dict]:
    """Round 3: rotary dim per layer family (full layers may use partial rotary)."""
    out = []
    for rope_mode in ("full", "partial25", "partial50"):
        for vkind in ("rms_raw", "roped"):
            out.append(dict(norm="direct", embed="sqrt", act="gelu", vnorm="on",
                            vkind=vkind, rope_mode=rope_mode, scalar="final",
                            sandwich="on"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--model", default=str(default_gemma_model()))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--top", type=int, default=18)
    opts = ap.parse_args()
    torch.set_num_threads(opts.threads)

    o = fetch_oracle(opts.prompt)
    ids = o["ids"]
    target = torch.from_numpy(o["hidden"][-1]).float()
    print(f"prompt={opts.prompt!r} ids={ids} llama hidden norm={float(target.norm()):.3f}",
          flush=True)
    print(f"llama top ids {[t['id'] for t in o['probs'][:5]]} "
          f"logp {[round(t['logprob'],3) for t in o['probs'][:5]]}", flush=True)

    ts = GGUFTensors(Path(opts.model))
    cfg = Config(ts)
    vars_ = variants()[: opts.top]
    print(f"{len(vars_)} variants, layers={cfg.n_layer}", flush=True)

    x0 = ts.embed_rows(ids)[0].float()
    states = [(x0 * math.sqrt(cfg.hidden) if v["embed"] == "sqrt" else x0.clone())
              for v in vars_]
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
        Wg = Ts(pre + "ffn_gate.weight")
        Wu = Ts(pre + "ffn_up.weight")
        Wd = Ts(pre + "ffn_down.weight")
        an = ts.get(pre + "attn_norm.weight")
        fn = ts.get(pre + "ffn_norm.weight")
        pan = ts.get(pre + "post_attention_norm.weight")
        pfn = ts.get(pre + "post_ffw_norm.weight")
        kn = ts.get(pre + "attn_k_norm.weight")
        lscale = float(ts.get(pre + "layer_output_scale.weight")[0])
        has_v = ts.has(pre + "attn_v.weight")
        Wv = Ts(pre + "attn_v.weight") if has_v else None
        new_states = []
        for v, xv in zip(vars_, states):
            xn = rms_norm(xv.unsqueeze(0), torch.from_numpy(an), cfg.eps)[0]
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
                else:
                    kk = rms_norm(k_raw, torch.from_numpy(kn), cfg.eps)
                    nr = d
                    if v["rope_mode"] == "partial25":
                        nr = max(2, d // 4)
                    elif v["rope_mode"] == "partial50":
                        nr = max(2, d // 2)
                    kk = rope(kk.unsqueeze(0), torch.tensor([0]), nr, lp["theta"], True)[0]
                    vv = kk
            ctx = vv.repeat_interleave(rep, dim=0).reshape(-1)
            if v["scalar"] == "sub":
                o_attn = (Wo @ ctx) * lscale
            else:
                o_attn = Wo @ ctx
            o_attn = rms_norm(o_attn.unsqueeze(0), torch.from_numpy(pan), cfg.eps)[0]
            x2 = xv + o_attn
            hn = rms_norm(x2.unsqueeze(0), torch.from_numpy(fn), cfg.eps)[0]
            g = Wg @ hn
            u = Wu @ hn
            mlp = Wd @ (gelu(g) * u)
            mlp = rms_norm(mlp.unsqueeze(0), torch.from_numpy(pfn), cfg.eps)[0]
            if v["scalar"] == "sub":
                x3 = x2 + mlp * lscale
            else:
                x3 = x2 + mlp
                if v["scalar"] == "final":
                    x3 = x3 * lscale
            new_states.append(x3)
        states = new_states
        ts.drop()
        if (i + 1) % 16 == 0:
            print(f"  layer {i+1}/{cfg.n_layer} ({time.time()-t0:.1f}s)", flush=True)

    on = ts.get("output_norm.weight")
    rows = []
    for v, xv in zip(vars_, states):
        h = rms_norm(xv.unsqueeze(0), torch.from_numpy(on), cfg.eps)[0]
        rel = float((h - target).norm() / target.norm())
        cos = float(torch.dot(h, target) / (h.norm() * target.norm()))
        rows.append((rel, cos, v))
    rows.sort(key=lambda r: r[0])
    print("\n== ranked by relative L2 to llama.cpp final hidden ==", flush=True)
    for rel, cos, v in rows:
        print(f"relL2={rel:9.3e} cos={cos:.6f} {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
