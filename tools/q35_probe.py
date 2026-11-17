"""numpy reference for a qwen35 layer, following transformers' qwen3_5 implementation.

Compares against engine dumps (ENGINE_DUMP=<prefix>):
  <prefix>_q35_hn0.f32  input to the layer's mixer (after input_layernorm)
  <prefix>_q35_x0.f32   layer output

Usage: python tools/q35_probe.py --model <gguf> --ids "1,2,3" --dump build/q35d [--layer 0]
"""
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


def rmsnorm(x, w, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)


def l2norm(x, eps=1e-6):
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True) + eps)
    return (x * inv).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    f = F(args.model)
    g = f.g
    hidden = int(g.get("qwen35.embedding_length"))
    eps = float(g.get("qwen35.attention.layer_norm_rms_epsilon"))
    conv_k = int(g.get("qwen35.ssm.conv_kernel"))
    inner = int(g.get("qwen35.ssm.inner_size"))
    state = int(g.get("qwen35.ssm.state_size"))
    groups = int(g.get("qwen35.ssm.group_count"))
    tsr = int(g.get("qwen35.ssm.time_step_rank"))
    n_vh, n_kh = tsr, groups
    kd, vd = state, inner // tsr
    key_dim, value_dim = n_kh * kd, n_vh * vd
    conv_dim = 2 * key_dim + value_dim

    p = f"blk.{args.layer}."
    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)

    x = f.t("token_embd.weight")[ids].astype(np.float32)
    h = rmsnorm(x, f.t(p + "attn_norm.weight"), eps)
    hn = np.fromfile(args.dump + "_q35_hn0.f32", dtype=np.float32).reshape(T, hidden)
    print(f"hn0  : rel_l2={np.linalg.norm(hn - h)/np.linalg.norm(h):.3e}")

    # --- mixer ---
    wqkv = f.t(p + "attn_qkv.weight")          # [conv_dim, hidden]
    wz = f.t(p + "attn_gate.weight")           # [value_dim, hidden]
    wa = f.t(p + "ssm_alpha.weight")           # [n_vh, hidden]
    wb = f.t(p + "ssm_beta.weight")            # [n_vh, hidden]
    convw = f.t(p + "ssm_conv1d.weight")       # [conv_dim, K]
    a_log = f.t(p + "ssm_a")
    dt_bias = f.t(p + "ssm_dt.bias")
    normw = f.t(p + "ssm_norm.weight")
    wout = f.t(p + "ssm_out.weight")

    mixed = (h @ wqkv.T).astype(np.float32)    # [T, conv_dim]
    z = (h @ wz.T).astype(np.float32).reshape(T, n_vh, vd)
    beta = (1.0 / (1.0 + np.exp(-(h @ wb.T).astype(np.float64)))).astype(np.float64)
    a = (h @ wa.T).astype(np.float64)
    gdec = -np.exp(a_log.astype(np.float64))[None, :] * np.log1p(np.exp(a + dt_bias[None, :]))
    # conv1d (causal, zero padding) over the token axis, then silu
    padded = np.concatenate([np.zeros((conv_k - 1, conv_dim), np.float32), mixed], axis=0)
    conv = np.zeros((T, conv_dim), np.float32)
    for t in range(T):
        acc = np.zeros(conv_dim, np.float64)
        for k in range(conv_k):
            acc += convw[:, k].astype(np.float64) * padded[t + k].astype(np.float64)
        conv[t] = acc
    conv = silu(conv)

    q = conv[:, :key_dim].reshape(T, n_kh, kd).astype(np.float64)
    k = conv[:, key_dim:2 * key_dim].reshape(T, n_kh, kd).astype(np.float64)
    v = conv[:, 2 * key_dim:].reshape(T, n_vh, vd).astype(np.float64)
    q = l2norm(q) / np.sqrt(kd)
    k = l2norm(k)
    rep = n_vh // n_kh
    if rep > 1:
        q = np.repeat(q, rep, axis=1)
        k = np.repeat(k, rep, axis=1)

    S = np.zeros((n_vh, kd, vd), np.float64)
    o = np.zeros((T, n_vh, vd), np.float64)
    for t in range(T):
        S *= np.exp(gdec[t])[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - kv) * beta[t][:, None]
        S += k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", S, q[t])

    # gated RMSNorm (norm before gate) then out_proj
    o = o.reshape(T * n_vh, vd)
    ms = np.mean(o ** 2, axis=-1, keepdims=True)
    o = (o / np.sqrt(ms + eps) * normw) * silu(z.reshape(T * n_vh, vd))
    out = (o.reshape(T, value_dim).astype(np.float32) @ wout.T).astype(np.float32)
    x1 = (x + out).astype(np.float32)

    # --- MLP ---
    h2 = rmsnorm(x1, f.t(p + "post_attention_norm.weight"), eps)
    gg = (h2 @ f.t(p + "ffn_gate.weight").T).astype(np.float32)
    uu = (h2 @ f.t(p + "ffn_up.weight").T).astype(np.float32)
    act = (silu(gg.astype(np.float64)) * uu).astype(np.float32)
    d = (act @ f.t(p + "ffn_down.weight").T).astype(np.float32)
    x2 = (x1 + d).astype(np.float32)

    ref = np.fromfile(args.dump + "_q35_x0.f32", dtype=np.float32).reshape(T, hidden)
    print(f"x0   : rel_l2={np.linalg.norm(ref - x2)/np.linalg.norm(x2):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
