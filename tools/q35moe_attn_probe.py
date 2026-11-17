"""numpy reference for a qwen35 full-attention layer (qwen3-next style).

Needs engine dumps: ENGINE_DUMP=<p> ENGINE_DUMP_LAYER=<layer>
  <p>_q35_attn_in.f32   layer input
  <p>_q35_attn_hn.f32   after input_layernorm
  <p>_q35_attn_mid.f32  after mixer + residual
  <p>_q35_attn_x.f32    layer output
Usage: python tools/q35_attn_probe.py --model <gguf> --dump build/q35a --layer 3
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


def rope_half(x, pos, base, rope_dim):
    out = x.astype(np.float64).copy()
    half = rope_dim // 2
    freqs = 1.0 / (base ** (np.arange(half, dtype=np.float64) * 2.0 / rope_dim))
    for t, p in enumerate(pos):
        ang = p * freqs
        c, s = np.cos(ang), np.sin(ang)
        a = out[t, ..., :half].copy()
        b = out[t, ..., half:rope_dim].copy()
        out[t, ..., :half] = a * c - b * s
        out[t, ..., half:rope_dim] = a * s + b * c
    return out


def has_tensor(f, name):
    return name in f.info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"<MODEL_DIR>\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf")
    ap.add_argument("--dump", default="build/a35")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--ids", default="760,6511,314,9338")
    args = ap.parse_args()

    f = F(args.model)
    g = f.g
    hidden = int(g.get("qwen35moe.embedding_length") or g.get("qwen35.embedding_length"))
    eps = float(g.get("qwen35moe.attention.layer_norm_rms_epsilon") or g.get("qwen35.attention.layer_norm_rms_epsilon"))
    H = int(g.get("qwen35moe.attention.head_count") or g.get("qwen35.attention.head_count"))
    KV = int(g.get("qwen35moe.attention.head_count_kv") or g.get("qwen35.attention.head_count_kv"))
    D = int(g.get("qwen35moe.attention.key_length") or g.get("qwen35.attention.key_length"))
    rope_dim = int(g.get("qwen35moe.rope.dimension_count") or g.get("qwen35.rope.dimension_count", 64))
    base = float(g.get("qwen35moe.rope.freq_base") or g.get("qwen35.rope.freq_base", 10000000.0))
    p = f"blk.{args.layer}."
    T = len([x for x in args.ids.split(",")])
    pos = np.arange(T)

    x_in = np.fromfile(args.dump + "_q35_attn_in.f32", dtype=np.float32)
    T = x_in.size // hidden
    x_in = x_in.reshape(T, hidden)
    pos = np.arange(T)
    h = rmsnorm(x_in, f.t(p + "attn_norm.weight"), eps)
    h_eng = np.fromfile(args.dump + "_q35_attn_hn.f32", dtype=np.float32).reshape(T, hidden)
    print(f"hn      : {np.linalg.norm(h - h_eng)/np.linalg.norm(h):.3e}")

    qfull = (h @ f.t(p + "attn_q.weight").T).astype(np.float32).reshape(T, H, 2 * D)
    q = qfull[..., :D]
    gate = qfull[..., D:]
    q = rmsnorm(q, f.t(p + "attn_q_norm.weight"), eps)
    k = (h @ f.t(p + "attn_k.weight").T).astype(np.float32).reshape(T, KV, D)
    k = rmsnorm(k, f.t(p + "attn_k_norm.weight"), eps)
    v = (h @ f.t(p + "attn_v.weight").T).astype(np.float32).reshape(T, KV, D)
    q = rope_half(q, pos, base, rope_dim)
    k = rope_half(k, pos, base, rope_dim)

    rep = H // KV
    kx = np.repeat(k.astype(np.float64), rep, axis=1)
    vx = np.repeat(v.astype(np.float64), rep, axis=1)
    score = 1.0 / np.sqrt(D)
    outv = np.zeros((T, H, D), dtype=np.float64)
    for t in range(T):
        s = np.einsum("hd,jhd->hj", q[t], kx[: t + 1]) * score
        m = s.max(axis=-1, keepdims=True)
        e = np.exp(s - m)
        pr = e / e.sum(axis=-1, keepdims=True)
        outv[t] = np.einsum("hj,jhd->hd", pr, vx[: t + 1])
    attn = outv.reshape(T, H * D).astype(np.float32)
    attn = attn * (1.0 / (1.0 + np.exp(-gate.reshape(T, H * D))))
    o = (attn @ f.t(p + "attn_output.weight").T).astype(np.float32)
    mid = (x_in + o).astype(np.float32)
    mid_eng = np.fromfile(args.dump + "_q35_attn_mid.f32", dtype=np.float32).reshape(T, hidden)
    print(f"mixer   : {np.linalg.norm(mid - mid_eng)/np.linalg.norm(mid):.3e}")

    if not has_tensor(f, p + "ffn_gate.weight"):
        print("layer   : (MoE FFN, mixer comparison only)")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
