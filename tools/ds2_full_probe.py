"""Full-model numpy forward of a deepseek2 GGUF, compared against the engine's logits.

Streams one layer at a time (dequantise in python via gguf.quants), so memory stays
at a few hundred MB.

Usage:
  ENGINE_LOGITS=build/eng_logits.f32 build/test_gen.exe <model> "<ids>" 1
  python tools/ds2_full_probe.py --model <model> --ids "..." --logits build/eng_logits.f32
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--yarn", action="store_true", default=True)
    ap.add_argument("--dump-every", default="")
    args = ap.parse_args()

    f = F(args.model)
    g = f.g
    hidden = int(g.get("deepseek2.embedding_length"))
    n_layer = int(g.get("deepseek2.block_count"))
    n_head = int(g.get("deepseek2.attention.head_count"))
    kv_lora = int(g.get("deepseek2.attention.kv_lora_rank"))
    qk = int(g.get("deepseek2.attention.key_length"))
    v_dim = int(g.get("deepseek2.attention.value_length"))
    rope_dim = int(g.get("deepseek2.rope.dimension_count"))
    base = float(g.get("deepseek2.rope.freq_base"))
    eps = float(g.get("deepseek2.attention.layer_norm_rms_epsilon"))
    qk_nope = qk - rope_dim
    n_expert = int(g.get("deepseek2.expert_count", 0))
    topk = int(g.get("deepseek2.expert_used_count", 0))
    leading_dense = int(g.get("deepseek2.leading_dense_block_count", 0))

    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)
    pos = np.arange(T)

    factor = float(g.get("deepseek2.rope.scaling.factor", 1.0))
    orig = float(g.get("deepseek2.rope.scaling.original_context_length", 4096))
    logmul = float(g.get("deepseek2.rope.scaling.yarn_log_multiplier", 0.0))
    beta_fast = float(g.get("deepseek2.rope.scaling.beta_fast", 32.0))
    beta_slow = float(g.get("deepseek2.rope.scaling.beta_slow", 1.0))
    half = rope_dim // 2
    freqs = base ** (-2.0 * np.arange(half) / rope_dim)
    mscale = 1.0
    if args.yarn and factor > 1:
        def corr(beta):
            return rope_dim * np.log(orig / (beta * 2 * np.pi)) / (2 * np.log(base))
        c0 = max(0.0, np.floor(corr(beta_fast)))
        c1 = min(rope_dim - 1.0, np.ceil(corr(beta_slow)))
        for i in range(half):
            i0 = 2.0 * i
            extrap = base ** (-i0 / rope_dim)
            ramp = min(1.0, max(0.0, (i0 - c0) / max(1e-9, c1 - c0)))
            freqs[i] = extrap * (1 - ramp) / factor + extrap * ramp
        mscale = 0.1 * logmul * np.log(factor) + 1.0

    cos_t = np.stack([np.cos(p * freqs) * mscale for p in pos])
    sin_t = np.stack([np.sin(p * freqs) * mscale for p in pos])

    def rope(x, offset, dim):
        hh = dim // 2
        out = x.astype(np.float64).copy()
        a = out[..., offset:offset + hh].copy()
        b = out[..., offset + hh:offset + dim].copy()
        c = cos_t[:, None, :]
        s = sin_t[:, None, :]
        out[..., offset:offset + hh] = a * c - b * s
        out[..., offset + hh:offset + dim] = a * s + b * c
        return out

    x = f.t("token_embd.weight")[ids].astype(np.float32)
    scale = 1.0 / np.sqrt(qk)
    for li in range(n_layer):
        p = f"blk.{li}."
        h = (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps) *
             f.t(p + "attn_norm.weight")).astype(np.float32)
        q = (h @ f.t(p + "attn_q.weight").T).reshape(T, n_head, qk)
        kv_a = (h @ f.t(p + "attn_kv_a_mqa.weight").T).astype(np.float32)
        wnorm = f.t(p + "attn_kv_a_norm.weight")
        c_kv = (kv_a[:, :kv_lora] /
                np.sqrt(np.mean(kv_a[:, :kv_lora].astype(np.float64) ** 2, axis=-1,
                                keepdims=True) + eps) * wnorm).astype(np.float32)
        k_pe = kv_a[:, kv_lora:]
        wkb = f.t(p + "attn_kv_b.weight")
        k_nope = (c_kv @ wkb[: n_head * qk_nope].T).reshape(T, n_head, qk_nope)
        v = (c_kv @ wkb[n_head * qk_nope:].T).reshape(T, n_head, v_dim)
        q = rope(q, qk_nope, rope_dim)
        k_pe_r = rope(k_pe[:, None, :], 0, rope_dim)
        k = np.concatenate([k_nope.astype(np.float64),
                            np.repeat(k_pe_r, n_head, axis=1)], axis=-1)
        outv = np.zeros((T, n_head, v_dim), dtype=np.float64)
        for hh in range(n_head):
            s = q[:, hh, :] @ k[:, hh, :].T * scale
            s[np.arange(T)[None, :] > np.arange(T)[:, None]] = -np.inf
            m = s.max(axis=-1, keepdims=True)
            e = np.exp(s - m)
            pr = e / e.sum(axis=-1, keepdims=True)
            outv[:, hh, :] = pr @ v[:, hh, :].astype(np.float64)
        o = (outv.reshape(T, n_head * v_dim).astype(np.float32) @ f.t(p + "attn_output.weight").T)
        x = (x + o).astype(np.float32)
        x_attn = x.copy()

        h2 = (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps) *
              f.t(p + "ffn_norm.weight")).astype(np.float32)
        if li < leading_dense:
            gg = (h2 @ f.t(p + "ffn_gate.weight").T).astype(np.float32)
            uu = (h2 @ f.t(p + "ffn_up.weight").T).astype(np.float32)
            act = silu(gg.astype(np.float64)) * uu
            ff = (act @ f.t(p + "ffn_down.weight").T).astype(np.float32)
        else:
            wr = f.t(p + "ffn_gate_inp.weight")
            scores = 1.0 / (1.0 + np.exp(-(h2 @ wr.T).astype(np.float64)))
            ge = f.t(p + "ffn_gate_exps.weight")
            ue = f.t(p + "ffn_up_exps.weight")
            de = f.t(p + "ffn_down_exps.weight")
            ff = np.zeros((T, hidden), dtype=np.float32)
            for t in range(T):
                idx = np.argsort(-scores[t])[:topk]
                w = scores[t][idx]
                acc = np.zeros(hidden, dtype=np.float64)
                for kk, e in enumerate(idx):
                    gt = (h2[t] @ ge[e].T).astype(np.float64)
                    ut = (h2[t] @ ue[e].T).astype(np.float64)
                    acc += w[kk] * ((silu(gt) * ut) @ de[e].T)
                ff[t] = acc.astype(np.float32)
            gsh = (h2 @ f.t(p + "ffn_gate_shexp.weight").T).astype(np.float64)
            ush = (h2 @ f.t(p + "ffn_up_shexp.weight").T).astype(np.float64)
            ff = (ff.astype(np.float64) + (silu(gsh) * ush) @ f.t(p + "ffn_down_shexp.weight").T
                  ).astype(np.float32)
        x = (x + ff).astype(np.float32)
        if args.dump_every:
            got = np.fromfile(f"{args.dump_every}_L{li}.f32", dtype=np.float32).reshape(x.shape)
            Lx = np.fromfile(f"{args.dump_every}_L{li}.f32", dtype=np.float32).reshape(x.shape)
            print(f"  layer {li:2d}: rel_l2={np.linalg.norm(Lx-x)/np.linalg.norm(x):.3e}",
                  flush=True)
        elif li % 5 == 0:
            print(f"  layer {li:2d} done", flush=True)

    xf = (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps) *
          f.t("output_norm.weight")).astype(np.float32)
    logits = (xf @ f.t("output.weight").T).astype(np.float32)
    eng = np.fromfile(args.logits, dtype=np.float32).reshape(logits.shape)
    num = np.linalg.norm(eng - logits)
    den = np.linalg.norm(logits)
    print(f"full model: rel_l2={num/den:.3e} argmax py={int(logits[-1].argmax())} "
          f"engine={int(eng[-1].argmax())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
