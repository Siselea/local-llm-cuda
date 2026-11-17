"""Full qwen35 numpy forward, compared layer by layer against engine dumps.

Engine: ENGINE_DUMP_EVERY=<p> build/test_gen.exe <model> "<ids>" 1
Then : python tools/q35_full_probe.py --model <gguf> --ids "..." --dump <p>
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


def rope_interleaved(x, pos, base, rope_dim):
    """GPT-NeoX interleaved rotation (pairs (0,1), (2,3), ...)."""
    out = x.astype(np.float64).copy()
    half = rope_dim // 2
    freqs = 1.0 / (base ** (np.arange(half, dtype=np.float64) * 2.0 / rope_dim))
    for t, p in enumerate(pos):
        ang = p * freqs
        c, s = np.cos(ang), np.sin(ang)
        a = out[t, ..., 0:rope_dim:2].copy()
        b = out[t, ..., 1:rope_dim:2].copy()
        out[t, ..., 0:rope_dim:2] = a * c - b * s
        out[t, ..., 1:rope_dim:2] = a * s + b * c
    return out


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


class Model:
    def __init__(self, path: str, swap_ab: bool = False, conv_rev: bool = False,
                 a_raw: bool = False, rope_mode: str = "half"):
        self.rope_mode = rope_mode
        self.swap_ab = swap_ab
        self.conv_rev = conv_rev
        self.a_raw = a_raw
        self.f = F(path)
        g = self.f.g
        self.hidden = int(g.get("qwen35.embedding_length"))
        self.eps = float(g.get("qwen35.attention.layer_norm_rms_epsilon"))
        self.H = int(g.get("qwen35.attention.head_count"))
        self.KV = int(g.get("qwen35.attention.head_count_kv"))
        self.D = int(g.get("qwen35.attention.key_length"))
        self.rope_dim = int(g.get("qwen35.rope.dimension_count", 64))
        self.base = float(g.get("qwen35.rope.freq_base", 10000000.0))
        self.n_layer = int(g.get("qwen35.block_count"))
        self.n_vh = int(g.get("qwen35.ssm.time_step_rank"))
        self.n_kh = int(g.get("qwen35.ssm.group_count"))
        self.kd = int(g.get("qwen35.ssm.state_size"))
        self.vd = int(g.get("qwen35.ssm.inner_size")) // self.n_vh
        self.conv_k = int(g.get("qwen35.ssm.conv_kernel"))
        self.interval = int(g.get("qwen35.full_attention_interval", 4))

    def kind(self, i: int) -> str:
        return "attn" if (i + 1) % self.interval == 0 else "linear"

    def linear_layer(self, i: int, x: np.ndarray, conv_state: np.ndarray):
        f = self.f
        p = f"blk.{i}."
        T = x.shape[0]
        h = rmsnorm(x, f.t(p + "attn_norm.weight"), self.eps)
        wqkv = f.t(p + "attn_qkv.weight")
        mixed = (h @ wqkv.T).astype(np.float32)
        key_dim = self.n_kh * self.kd
        value_dim = self.n_vh * self.vd
        conv_dim = 2 * key_dim + value_dim
        convw = f.t(p + "ssm_conv1d.weight")
        padded = np.concatenate([conv_state, mixed], axis=0)
        conv = np.zeros_like(mixed)
        for t in range(T):
            acc = np.zeros(conv_dim, np.float64)
            for k in range(self.conv_k):
                kk = (self.conv_k - 1 - k) if self.conv_rev else k
                acc += convw[:, kk].astype(np.float64) * padded[t + k].astype(np.float64)
            conv[t] = acc
        conv = silu(conv)
        new_state = padded[-self.conv_k + 1:].copy() if self.conv_k > 1 else conv_state
        q = l2norm(conv[:, :key_dim].reshape(T, self.n_kh, self.kd)) / np.sqrt(self.kd)
        k = l2norm(conv[:, key_dim:2 * key_dim].reshape(T, self.n_kh, self.kd))
        v = conv[:, 2 * key_dim:].reshape(T, self.n_vh, self.vd).astype(np.float64)
        rep = self.n_vh // self.n_kh
        if rep > 1:
            q = np.repeat(q, rep, axis=1)
            k = np.repeat(k, rep, axis=1)
        a = (h @ f.t(p + "ssm_alpha.weight").T).astype(np.float64)
        b = (h @ f.t(p + "ssm_beta.weight").T).astype(np.float64)
        if self.swap_ab:
            a, b = b, a
        beta = 1.0 / (1.0 + np.exp(-b))
        a_log = f.t(p + "ssm_a").astype(np.float64)
        dtb = f.t(p + "ssm_dt.bias").astype(np.float64)
        sp = np.log1p(np.exp(a + dtb[None, :]))
        gd = (a_log[None, :] * sp) if self.a_raw else (-np.exp(a_log)[None, :] * sp)
        z = (h @ f.t(p + "attn_gate.weight").T).astype(np.float32).reshape(T, self.n_vh, self.vd)
        S = np.zeros((self.n_vh, self.kd, self.vd), np.float64)
        o = np.zeros((T, self.n_vh, self.vd), np.float64)
        for t in range(T):
            S *= np.exp(gd[t])[:, None, None]
            kv = np.einsum("hkv,hk->hv", S, k[t])
            delta = (v[t] - kv) * beta[t][:, None]
            S += k[t][:, :, None] * delta[:, None, :]
            o[t] = np.einsum("hkv,hk->hv", S, q[t])
        o = o.reshape(T * self.n_vh, self.vd)
        ms = np.mean(o ** 2, axis=-1, keepdims=True)
        o = (o / np.sqrt(ms + self.eps) * f.t(p + "ssm_norm.weight")) * silu(
            z.reshape(T * self.n_vh, self.vd))
        out = (o.reshape(T, value_dim).astype(np.float32) @ f.t(p + "ssm_out.weight").T)
        x1 = (x + out).astype(np.float32)
        h2 = rmsnorm(x1, f.t(p + "post_attention_norm.weight"), self.eps)
        gg = (h2 @ f.t(p + "ffn_gate.weight").T).astype(np.float32)
        uu = (h2 @ f.t(p + "ffn_up.weight").T).astype(np.float32)
        act = (silu(gg.astype(np.float64)) * uu).astype(np.float32)
        d = (act @ f.t(p + "ffn_down.weight").T).astype(np.float32)
        return (x1 + d).astype(np.float32), new_state

    def attn_layer(self, i: int, x: np.ndarray, kv_cache):
        f = self.f
        p = f"blk.{i}."
        T = x.shape[0]
        h = rmsnorm(x, f.t(p + "attn_norm.weight"), self.eps)
        qfull = (h @ f.t(p + "attn_q.weight").T).astype(np.float32).reshape(T, self.H, 2 * self.D)
        q = rmsnorm(qfull[..., : self.D], f.t(p + "attn_q_norm.weight"), self.eps)
        gate = qfull[..., self.D:]
        k = (h @ f.t(p + "attn_k.weight").T).astype(np.float32).reshape(T, self.KV, self.D)
        k = rmsnorm(k, f.t(p + "attn_k_norm.weight"), self.eps)
        v = (h @ f.t(p + "attn_v.weight").T).astype(np.float32).reshape(T, self.KV, self.D)
        pos = np.arange(T)
        if self.rope_mode == "half":
            q = rope_half(q, pos, self.base, self.rope_dim)
            k = rope_half(k, pos, self.base, self.rope_dim)
        elif self.rope_mode == "full":
            q = rope_half(q, pos, self.base, self.D)
            k = rope_half(k, pos, self.base, self.D)
        elif self.rope_mode == "tail":
            q = rope_half(q[..., self.D - self.rope_dim:], pos, self.base, self.rope_dim)
            k = rope_half(k[..., self.D - self.rope_dim:], pos, self.base, self.rope_dim)
        elif self.rope_mode == "interleaved":
            q = rope_interleaved(q, pos, self.base, self.rope_dim)
            k = rope_interleaved(k, pos, self.base, self.rope_dim)
        rep = self.H // self.KV
        kx = np.repeat(k.astype(np.float64), rep, axis=1)
        vx = np.repeat(v.astype(np.float64), rep, axis=1)
        score = 1.0 / np.sqrt(self.D)
        outv = np.zeros((T, self.H, self.D), np.float64)
        for t in range(T):
            s = np.einsum("hd,jhd->hj", q[t], kx[: t + 1]) * score
            m = s.max(axis=-1, keepdims=True)
            e = np.exp(s - m)
            pr = e / e.sum(axis=-1, keepdims=True)
            outv[t] = np.einsum("hj,jhd->hd", pr, vx[: t + 1])
        attn = outv.reshape(T, self.H * self.D).astype(np.float32)
        attn = attn * (1.0 / (1.0 + np.exp(-gate.reshape(T, self.H * self.D))))
        o = (attn @ f.t(p + "attn_output.weight").T).astype(np.float32)
        x1 = (x + o).astype(np.float32)
        h2 = rmsnorm(x1, f.t(p + "post_attention_norm.weight"), self.eps)
        gg = (h2 @ f.t(p + "ffn_gate.weight").T).astype(np.float32)
        uu = (h2 @ f.t(p + "ffn_up.weight").T).astype(np.float32)
        act = (silu(gg.astype(np.float64)) * uu).astype(np.float32)
        d = (act @ f.t(p + "ffn_down.weight").T).astype(np.float32)
        return (x1 + d).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--max-layer", type=int, default=32)
    ap.add_argument("--swap-ab", action="store_true")
    ap.add_argument("--conv-rev", action="store_true")
    ap.add_argument("--a-raw", action="store_true")
    ap.add_argument("--rope-mode", default="half",
                    choices=["half", "full", "tail", "interleaved"])
    ap.add_argument("--topk", action="store_true")
    args = ap.parse_args()

    m = Model(args.model, swap_ab=args.swap_ab, conv_rev=args.conv_rev, a_raw=args.a_raw,
              rope_mode=args.rope_mode)
    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)
    x = m.f.t("token_embd.weight")[ids].astype(np.float32)
    for i in range(min(m.n_layer, args.max_layer)):
        if m.kind(i) == "linear":
            zero_state = np.zeros((m.conv_k - 1, 2 * m.n_kh * m.kd + m.n_vh * m.vd),
                                  np.float32)
            x, _ = m.linear_layer(i, x, zero_state)
        else:
            x = m.attn_layer(i, x, None)
        fp = Path(f"{args.dump}_L{i}.f32")
        if not fp.exists():
            continue
        ref = np.fromfile(fp, dtype=np.float32).reshape(T, m.hidden)
        rel = np.linalg.norm(ref - x) / np.linalg.norm(x)
        if i == min(m.n_layer, args.max_layer) - 1 and args.topk:
            print("last layer rel", f"{rel:.3e}")
    if args.topk:
        xf = rmsnorm(x, m.f.t("output_norm.weight"), m.eps)
        logits = (xf @ m.f.t("output.weight").T).astype(np.float32)
        order = np.argsort(-logits[-1])[:5]
        mx = logits[-1].max()
        lp = logits[-1] - mx - np.log(np.exp(logits[-1] - mx).sum())
        print("top5:", [(int(i), round(float(lp[i]), 3)) for i in order])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
