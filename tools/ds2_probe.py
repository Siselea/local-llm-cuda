"""Dequantise real GGUF tensors in numpy and recompute layer 0 of a deepseek2 model,
then compare against dumps written by the engine (ENGINE_DUMP=<prefix>)."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf.quants import dequantize as gq_dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyref.gguf_parse import read_gguf  # noqa: E402

TYPE_SIZE = {  # id -> (name, block_elems, block_bytes)
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
}


def f16(bits: np.ndarray) -> np.ndarray:
    return bits.view(np.float16).astype(np.float32)


def scale_min(sub: int, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sub shape (...,) -> (sc, mn); s shape (...,12) uint8"""
    if sub < 4:
        sc = s[..., sub] & 63
        mn = s[..., sub + 4] & 63
    else:
        sc = (s[..., sub + 4] & 0xF) | ((s[..., sub - 4] >> 6) << 4)
        mn = (s[..., sub] >> 4) | ((s[..., sub - 4] >> 2) & 0x30)
    return sc.astype(np.float32), mn.astype(np.float32)


def dequant(raw: bytes, type_id: int, n_elems: int) -> np.ndarray:
    name, be, bb = TYPE_SIZE[type_id]
    nb = n_elems // be
    buf = np.frombuffer(raw, dtype=np.uint8, count=nb * bb).reshape(nb, bb)
    if name == "F32":
        return buf.view(np.float32).reshape(-1)
    if name == "F16":
        return f16(buf.view(np.uint16))
    if name == "Q8_0":
        d = f16(buf[:, 0:2].copy().view(np.uint16).reshape(-1))
        q = buf[:, 2:].view(np.int8).astype(np.float32)
        return (d[:, None] * q).reshape(-1)
    if name in ("Q5_0", "Q5_1"):
        d = f16(buf[:, 0:2].copy().view(np.uint16).reshape(-1))
        if name == "Q5_0":
            qh = buf[:, 2:6].copy().view(np.uint32).reshape(-1)
            qs = buf[:, 6:22]
        else:
            m = f16(buf[:, 2:4].copy().view(np.uint16).reshape(-1))
            qh = buf[:, 4:8].copy().view(np.uint32).reshape(-1)
            qs = buf[:, 8:24]
        out = np.empty((nb, 32), dtype=np.float32)
        for i in range(32):
            low = np.where(i < 16, qs[:, i % 16] & 0xF, qs[:, i % 16] >> 4)
            hi = (qh >> i) & 1
            if name == "Q5_0":
                out[:, i] = d * (low | (hi << 4) - 16).astype(np.float32)
            else:
                out[:, i] = d * (low | (hi << 4)).astype(np.float32) + m
        return out.reshape(-1)
    if name in ("Q4_K", "Q5_K"):
        d = f16(buf[:, 0:2].copy().view(np.uint16).reshape(-1))
        dmin = f16(buf[:, 2:4].copy().view(np.uint16).reshape(-1))
        s = buf[:, 4:16]
        qs = buf[:, 48:176] if name == "Q5_K" else buf[:, 16:144]
        qh = buf[:, 16:48] if name == "Q5_K" else None
        out = np.empty((nb, 256), dtype=np.float32)
        for e in range(256):
            g, half, l = e >> 6, (e & 63) >> 5, e & 31
            sub = g * 2 + half
            qb = qs[:, g * 32 + l]
            nib = (qb >> 4) if half else (qb & 0xF)
            val = nib.astype(np.float32)
            if qh is not None:
                val += (((qh[:, l] >> sub) & 1) << 4).astype(np.float32)
            sc, mn = scale_min(sub, s)
            out[:, e] = d * sc * val - dmin * mn
        return out.reshape(-1)
    if name == "Q6_K":
        d = f16(buf[:, 208:210].copy().view(np.uint16).reshape(-1))
        ql = buf[:, 0:128]
        qh = buf[:, 128:192]
        sc = buf[:, 192:208].view(np.int8).astype(np.float32)
        out = np.empty((nb, 256), dtype=np.float32)
        for n2 in range(2):
            for l in range(32):
                is_ = l // 16
                qi, hi = n2 * 64, n2 * 32
                q1 = ((ql[:, qi + l] & 0xF) | (((qh[:, hi + l] >> 0) & 3) << 4)).astype(np.float32) - 32
                q2 = ((ql[:, qi + l + 32] & 0xF) | (((qh[:, hi + l] >> 2) & 3) << 4)).astype(np.float32) - 32
                q3 = ((ql[:, qi + l] >> 4) | (((qh[:, hi + l] >> 4) & 3) << 4)).astype(np.float32) - 32
                q4 = ((ql[:, qi + l + 32] >> 4) | (((qh[:, hi + l] >> 6) & 3) << 4)).astype(np.float32) - 32
                base = n2 * 128
                out[:, base + l + 0] = d * sc[:, n2 * 8 + is_ + 0] * q1
                out[:, base + l + 32] = d * sc[:, n2 * 8 + is_ + 2] * q2
                out[:, base + l + 64] = d * sc[:, n2 * 8 + is_ + 4] * q3
                out[:, base + l + 96] = d * sc[:, n2 * 8 + is_ + 6] * q4
        return out.reshape(-1)
    raise SystemExit(f"unsupported type {name}")


class F:
    def __init__(self, path: str):
        self.g = read_gguf(path)
        self.f = open(path, "rb")
        self.info = {t.name: t for t in self.g.tensors}

    def t(self, name: str) -> np.ndarray:
        ti = self.info[name]
        self.f.seek(self.g.data_offset + ti.offset)
        raw = self.f.read(ti.nbytes)
        if ti.type_name == "F32":
            v = np.frombuffer(raw, dtype=np.float32)
        elif ti.type_name == "F16":
            v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            v = gq_dequantize(np.frombuffer(raw, dtype=np.uint8),
                              gguf.GGMLQuantizationType(ti.type_id)).astype(np.float32)
        return v.reshape(ti.shape).astype(np.float32)


def rmsnorm(x, w, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", default="785,6722,315,9625,374")
    ap.add_argument("--dump", default="build/ds2dump")
    ap.add_argument("--yarn", action="store_true", help="apply YaRN frequencies + mscale")
    ap.add_argument("--no-yarn", dest="yarn", action="store_false")
    ap.set_defaults(yarn=True)
    args = ap.parse_args()

    f = F(args.model)
    g = f.g
    hidden = int(g.get("deepseek2.embedding_length"))
    n_head = int(g.get("deepseek2.attention.head_count"))
    kv_lora = int(g.get("deepseek2.attention.kv_lora_rank"))
    qk = int(g.get("deepseek2.attention.key_length"))
    v_dim = int(g.get("deepseek2.attention.value_length"))
    rope_dim = int(g.get("deepseek2.rope.dimension_count"))
    base = float(g.get("deepseek2.rope.freq_base"))
    eps = float(g.get("deepseek2.attention.layer_norm_rms_epsilon"))
    qk_nope = qk - rope_dim

    ids = [int(x) for x in args.ids.split(",")]
    L = len(ids)
    pos = np.arange(L)

    emb = f.t("token_embd.weight")
    x = emb[ids].astype(np.float32)

    ln1 = f.t("blk.0.attn_norm.weight")
    h = rmsnorm(x, ln1, eps)

    wq = f.t("blk.0.attn_q.weight")
    q = (h @ wq.T).reshape(L, n_head, qk)
    wkva = f.t("blk.0.attn_kv_a_mqa.weight")
    kv_a = (h @ wkva.T).astype(np.float32)
    wnorm = f.t("blk.0.attn_kv_a_norm.weight")
    c_kv = rmsnorm(kv_a[:, :kv_lora], wnorm, eps)
    k_pe = kv_a[:, kv_lora:]
    wkb = f.t("blk.0.attn_kv_b.weight")
    k_nope = (c_kv @ wkb[: n_head * qk_nope].T).reshape(L, n_head, qk_nope)
    v = (c_kv @ wkb[n_head * qk_nope:].T).reshape(L, n_head, v_dim)

    # rope frequencies
    half = rope_dim // 2
    freqs = base ** (-2.0 * np.arange(half) / rope_dim)
    mscale = 1.0
    if args.yarn:
        factor = float(g.get("deepseek2.rope.scaling.factor", 1.0))
        orig = float(g.get("deepseek2.rope.scaling.original_context_length", 4096))
        logmul = float(g.get("deepseek2.rope.scaling.yarn_log_multiplier", 0.0))
        bf = float(g.get("deepseek2.rope.scaling.beta_fast", 32.0))
        bs = float(g.get("deepseek2.rope.scaling.beta_slow", 1.0))

        def corr(beta):
            return rope_dim * np.log(orig / (beta * 2 * np.pi)) / (2 * np.log(base))

        c0 = max(0.0, np.floor(corr(bf)))
        c1 = min(rope_dim - 1.0, np.ceil(corr(bs)))
        for i in range(half):
            i0 = 2.0 * i
            extrap = base ** (-i0 / rope_dim)
            ramp = min(1.0, max(0.0, (i0 - c0) / max(1e-9, c1 - c0)))
            freqs[i] = extrap * (1 - ramp) / factor + extrap * ramp
        mscale = 1.0 if factor <= 1 else 0.1 * logmul * np.log(factor) + 1.0

    def rope(x_, offset, dim):
        out = x_.astype(np.float64).copy()
        hh = dim // 2
        for i, p in enumerate(pos):
            ang = p * freqs[:hh]
            c, s = np.cos(ang) * mscale, np.sin(ang) * mscale
            a = out[i, ..., offset:offset + hh].copy()
            b = out[i, ..., offset + hh:offset + dim].copy()
            out[i, ..., offset:offset + hh] = a * c - b * s
            out[i, ..., offset + hh:offset + dim] = a * s + b * c
        return out

    q = rope(q, qk_nope, rope_dim)
    k_pe_r = rope(k_pe[:, None, :], 0, rope_dim)
    k = np.concatenate([k_nope.astype(np.float64),
                        np.repeat(k_pe_r, n_head, axis=1)], axis=-1)
    scale = 1.0 / np.sqrt(qk)
    outv = np.zeros((L, n_head, v_dim), dtype=np.float64)
    for hh in range(n_head):
        s = q[:, hh, :] @ k[:, hh, :].T * scale
        s[np.arange(L)[None, :] > np.arange(L)[:, None]] = -np.inf
        m = s.max(axis=-1, keepdims=True)
        e = np.exp(s - m)
        pr = e / e.sum(axis=-1, keepdims=True)
        outv[:, hh, :] = pr @ v[:, hh, :].astype(np.float64)
    wo = f.t("blk.0.attn_output.weight")
    o = (outv.reshape(L, n_head * v_dim).astype(np.float32) @ wo.T).astype(np.float32)
    x1 = (x + o).astype(np.float32)

    ln2 = f.t("blk.0.ffn_norm.weight")
    h2 = rmsnorm(x1, ln2, eps)
    wg = f.t("blk.0.ffn_gate.weight")
    wu = f.t("blk.0.ffn_up.weight")
    wd = f.t("blk.0.ffn_down.weight")
    gg = (h2 @ wg.T).astype(np.float32)
    uu = (h2 @ wu.T).astype(np.float32)
    act = (gg / (1.0 + np.exp(-gg.astype(np.float64))) * uu).astype(np.float32)
    d = (act @ wd.T).astype(np.float32)
    x2 = (x1 + d).astype(np.float32)

    def cmp(name: str, ref: np.ndarray, path: str) -> None:
        try:
            got = np.fromfile(path, dtype=np.float32)
        except OSError:
            print(f"{name}: no dump at {path}")
            return
        ref = ref.reshape(-1)
        if got.size != ref.size:
            print(f"{name}: size {got.size} vs {ref.size}")
            return
        num = np.linalg.norm(got - ref)
        den = np.linalg.norm(ref)
        print(f"{name}: rel_l2={num/den:.3e} max_abs={np.abs(got-ref).max():.3e}")

    print(f"yarn={args.yarn} mscale={mscale:.5f}")
    cmp("layer0.hn ", h, args.dump + "_hn0.f32")
    cmp("layer0.x  ", x2, args.dump + "_x0.f32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
