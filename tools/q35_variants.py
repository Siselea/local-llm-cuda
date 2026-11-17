"""Multi-variant numpy reference for Qwen3.5-9B: the GGUF-vs-HF semantics adjudicator.

This tool exists because a GGUF qwen35/qwen35moe file is NOT "HF semantics with a
different container": the llama.cpp converter writes

  * qwen35.ssm_a = -A            (not log A)   ->  g = ssm_a * softplus(a + dt_bias)
  * q/k -> v head broadcast by *block repeat*   ->  v_head h reads key head h % n_kh
    (llama.cpp ggml/src/ggml-cuda/gated_delta_net.cu: iq1 = fastmodulo(h_idx, neqk1);
     ggml.h marks the alternative as [TAG_GGML_GDN_BCAST])

while HF safetensors checkpoints use log(A) and repeat_interleave (h / rep).  Feeding a
GGUF file with HF semantics leaves position 0 almost correct and destroys every later
position -- the exact failure mode behind "single-token smoke is green, real text is
garbage".  This script reimplements the dense qwen35 stack in numpy and can switch both
conventions, so it is the referee whenever GGUF and HF conventions disagree.

Two uses
--------
1. Root-cause / adjudication table (all variants in one pass):
     python tools/q35_variants.py --model <gguf> --ids "760,6511,314,9338,369"
   Expected: only a-fix=1 + bcast=tile reproduces llama.cpp's greedy continuation.

2. Per-layer regression for the engine's GGUF path (engine must match a-fix=1+tile):
     ENGINE_DUMP_EVERY=build/ref9b ./build/test_gen.exe <gguf> "760,6511,314,9338" 1
     python tools/q35_variants.py --model <gguf> --ids "760,6511,314,9338" \
         --dump build/ref9b --a-fix 1 --bcast tile --layer 7 --tolerance 2e-4
   Exit code 1 if any dumped layer is farther than --tolerance (rel_l2).

Do not use the HF torch chain (tools/q35_torch_chain.py) as the oracle for a GGUF file:
it applies HF semantics.  It remains valid for safetensors checkpoints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds2_probe import F  # noqa: E402

EPS = 1e-6
ROPE_DIM = 64


def rmsnorm(x, w, eps=EPS):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)


def l2norm(x, eps=EPS):
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True) + eps)
    return (x * inv).astype(np.float64)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


class Ctx:
    def __init__(self, model: str):
        self.f = F(model)
        g = self.f.g
        self.hidden = int(g.get("qwen35.embedding_length"))
        self.n_layer = int(g.get("qwen35.block_count"))
        self.n_head = int(g.get("qwen35.attention.head_count"))
        self.n_kv = int(g.get("qwen35.attention.head_count_kv"))
        self.head_dim = int(g.get("qwen35.attention.key_length"))
        self.ffn = int(g.get("qwen35.feed_forward_length"))
        self.interval = int(g.get("qwen35.full_attention_interval", 4))
        self.rope_base = float(g.get("qwen35.rope.freq_base", 1e7))
        self.rope_dim = int(g.get("qwen35.rope.dimension_count", 64))
        self.eps = float(g.get("qwen35.attention.layer_norm_rms_epsilon"))
        self.n_kh = int(g.get("qwen35.ssm.group_count"))
        self.n_vh = int(g.get("qwen35.ssm.time_step_rank"))
        self.kd = int(g.get("qwen35.ssm.state_size"))
        self.vd = self.kd
        self.key_dim = self.n_kh * self.kd
        self.value_dim = self.n_vh * self.vd
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_k = int(g.get("qwen35.ssm.conv_kernel"))
        # rope frequencies: theta^(-2i/rope_dim), half split
        half = self.rope_dim // 2
        self.freqs = np.array(
            [self.rope_base ** (-2.0 * i / self.rope_dim) for i in range(half)],
            dtype=np.float64)
        self._cache = {}

    def t(self, name):
        v = self._cache.get(name)
        if v is None:
            v = self.f.t(name).astype(np.float32)
            self._cache[name] = v
        return v

    def drop(self, prefix):
        for k in [k for k in self._cache if k.startswith(prefix)]:
            del self._cache[k]


def rope(x, freqs, T):
    """x: [T, heads, D]; rotate the first 2*len(freqs) dims, half-split (HF rotate_half)."""
    half = len(freqs)
    d = 2 * half
    pos = np.arange(T, dtype=np.float64)[:, None]
    ang = pos * freqs[None, :]                     # [T, half]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    xr = x[..., :d]
    x1 = xr[..., :half]
    x2 = xr[..., half:]
    out = x.copy()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:d] = x1 * sin + x2 * cos
    return out


def attention_layer(ctx: Ctx, i: int, x: np.ndarray) -> np.ndarray:
    p = f"blk.{i}."
    T = x.shape[0]
    H, KV, D = ctx.n_head, ctx.n_kv, ctx.head_dim
    rep = H // KV
    hn = rmsnorm(x, ctx.t(p + "attn_norm.weight"), ctx.eps)
    qfull = (hn @ ctx.t(p + "attn_q.weight").T).astype(np.float32)   # [T, H*2D]
    qfull = qfull.reshape(T, H, 2 * D)
    q = qfull[..., :D].copy()
    gate = qfull[..., D:].copy()
    k = (hn @ ctx.t(p + "attn_k.weight").T).astype(np.float32).reshape(T, KV, D)
    v = (hn @ ctx.t(p + "attn_v.weight").T).astype(np.float32).reshape(T, KV, D)
    q = rmsnorm(q, ctx.t(p + "attn_q_norm.weight"), ctx.eps)
    k = rmsnorm(k, ctx.t(p + "attn_k_norm.weight"), ctx.eps)
    q = rope(q, ctx.freqs, T)
    k = rope(k, ctx.freqs, T)
    kx = np.repeat(k, rep, axis=1)
    vx = np.repeat(v, rep, axis=1)
    scale = 1.0 / np.sqrt(np.float64(D))
    out = np.zeros((T, H, D), dtype=np.float32)
    for t in range(T):
        s = np.einsum("hd,nhd->hn", q[t].astype(np.float64),
                      kx[:t + 1].astype(np.float64)) * scale
        s = s - s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        e = e / e.sum(axis=-1, keepdims=True)
        out[t] = np.einsum("hn,nhd->hd", e,
                           vx[:t + 1].astype(np.float64)).astype(np.float32)
    out = out * (1.0 / (1.0 + np.exp(-gate)))
    out = out.reshape(T, H * D)
    proj = (out @ ctx.t(p + "attn_output.weight").T).astype(np.float32)
    return x + proj


def mixer_layer(ctx: Ctx, i: int, x: np.ndarray, bcast: str, a_fix: bool) -> np.ndarray:
    p = f"blk.{i}."
    T = x.shape[0]
    n_kh, n_vh, kd, vd = ctx.n_kh, ctx.n_vh, ctx.kd, ctx.vd
    rep = n_vh // n_kh
    hn = rmsnorm(x, ctx.t(p + "attn_norm.weight"), ctx.eps)
    mixed = (hn @ ctx.t(p + "attn_qkv.weight").T).astype(np.float32)   # [T, conv_dim]
    z = (hn @ ctx.t(p + "attn_gate.weight").T).astype(np.float32)      # [T, value_dim]
    b = (hn @ ctx.t(p + "ssm_beta.weight").T).astype(np.float32)       # [T, n_vh]
    a = (hn @ ctx.t(p + "ssm_alpha.weight").T).astype(np.float32)      # [T, n_vh]
    convw = ctx.t(p + "ssm_conv1d.weight")                             # [conv_dim, K]
    # causal depthwise conv with zero left history (sequence starts at position 0)
    conv = np.zeros_like(mixed)
    for k in range(ctx.conv_k):
        j = np.arange(T) + k - (ctx.conv_k - 1)
        m = j >= 0
        conv[m] += mixed[j[m]] * convw[:, k]
    conv = silu(conv)
    q = conv[:, :ctx.key_dim].reshape(T, n_kh, kd).astype(np.float64)
    k = conv[:, ctx.key_dim:2 * ctx.key_dim].reshape(T, n_kh, kd).astype(np.float64)
    v = conv[:, 2 * ctx.key_dim:].reshape(T, n_vh, vd).astype(np.float64)
    q = l2norm(q) / np.sqrt(np.float64(kd))
    k = l2norm(k)
    ssm_a = ctx.t(p + "ssm_a").astype(np.float64)                      # stored -A
    if a_fix:
        g = ssm_a[None, :] * softplus(a.astype(np.float64) + ctx.t(p + "ssm_dt.bias").astype(np.float64))
    else:
        g = -np.exp(ssm_a)[None, :] * softplus(a.astype(np.float64) + ctx.t(p + "ssm_dt.bias").astype(np.float64))
    beta = 1.0 / (1.0 + np.exp(-b.astype(np.float64)))
    if bcast == "interleave":
        qe = np.repeat(q, rep, axis=1)     # v head h <- k head h//rep
        ke = np.repeat(k, rep, axis=1)
    elif bcast == "tile":
        qe = np.tile(q, (1, rep, 1))       # v head h <- k head h%n_kh
        ke = np.tile(k, (1, rep, 1))
    else:
        raise ValueError(bcast)
    S = np.zeros((n_vh, kd, vd), dtype=np.float64)
    o = np.zeros((T, n_vh, vd), dtype=np.float64)
    for t in range(T):
        S *= np.exp(g[t])[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, ke[t])
        delta = (v[t] - kv) * beta[t][:, None]
        S += ke[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", S, qe[t])
    o = o.reshape(T * n_vh, vd).astype(np.float32)
    nw = ctx.t(p + "ssm_norm.weight")
    ms = np.mean(o.astype(np.float64) ** 2, axis=-1, keepdims=True)
    o = (o / np.sqrt(ms + ctx.eps) * nw) * silu(z.reshape(T * n_vh, vd))
    out = (o.reshape(T, ctx.value_dim).astype(np.float32) @ ctx.t(p + "ssm_out.weight").T).astype(np.float32)
    return x + out


def ffn_layer(ctx: Ctx, i: int, x: np.ndarray) -> np.ndarray:
    p = f"blk.{i}."
    h2 = rmsnorm(x, ctx.t(p + "post_attention_norm.weight"), ctx.eps)
    g = (h2 @ ctx.t(p + "ffn_gate.weight").T).astype(np.float32)
    u = (h2 @ ctx.t(p + "ffn_up.weight").T).astype(np.float32)
    act = (silu(g.astype(np.float64)) * u.astype(np.float64)).astype(np.float32)
    d = (act @ ctx.t(p + "ffn_down.weight").T).astype(np.float32)
    return x + d


def top5(logits):
    mx = logits.max()
    lp = logits - mx - np.log(np.exp(logits - mx).sum())
    order = np.argsort(-logits)[:5]
    return [(int(i), round(float(lp[i]), 3)) for i in order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--layer", type=int, default=-1,
                    help="stop after this layer (0-based, debug/regression)")
    ap.add_argument("--a-fix", type=int, default=-1, choices=(-1, 0, 1),
                    help="ssm_a convention: 1 = GGUF/llama.cpp (-A), 0 = HF (log A); "
                         "-1 = run the full 3-variant adjudication table")
    ap.add_argument("--bcast", default="", choices=("", "tile", "interleave"),
                    help="q/k->v head broadcast: tile = GGUF/llama.cpp (h %% n_kh), "
                         "interleave = HF (h / rep)")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="if > 0 and --dump is given, fail when any layer rel_l2 exceeds it")
    args = ap.parse_args()
    ctx = Ctx(args.model)
    ids = [int(s) for s in args.ids.split(",")]
    T = len(ids)

    single = ""
    if args.a_fix in (0, 1) and args.bcast:
        single = f"a{args.a_fix}_{args.bcast}"
        variants = {single: (args.bcast, bool(args.a_fix))}
    elif args.a_fix in (0, 1) or args.bcast:
        ap.error("--a-fix and --bcast must be given together for a single variant")
    else:
        # default: the full adjudication table (root-cause evidence)
        variants = {
            "V0_engine_current(logA,interleave)": ("interleave", False),
            "V1_Afix_only(-A,interleave)": ("interleave", True),
            "V2_llama(-A,tile)": ("tile", True),
            "V3_bcast_only(logA,tile)": ("tile", False),
        }
    # load token rows without materialising the whole embedding matrix
    import gguf
    from gguf.quants import dequantize as gq_dequantize
    ti = ctx.f.info["token_embd.weight"]
    row_bytes = ti.nbytes // int(ti.shape[0])
    x0 = np.zeros((T, ctx.hidden), dtype=np.float32)
    for r, tok in enumerate(ids):
        ctx.f.f.seek(ctx.f.g.data_offset + ti.offset + tok * row_bytes)
        raw = ctx.f.f.read(row_bytes)
        if ti.type_name == "F32":
            v = np.frombuffer(raw, dtype=np.float32)
        elif ti.type_name == "F16":
            v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            v = gq_dequantize(np.frombuffer(raw, dtype=np.uint8),
                              gguf.GGMLQuantizationType(ti.type_id)).astype(np.float32)
        x0[r] = v.reshape(-1)[:ctx.hidden]
    xs = {name: x0.copy() for name in variants}
    worst = 0.0

    n_layer = ctx.n_layer if args.layer < 0 else args.layer + 1
    import time
    t_start = time.time()
    for i in range(n_layer):
        t_l0 = time.time()
        is_attn = (i + 1) % ctx.interval == 0
        for name, (bcast, a_fix) in variants.items():
            x = xs[name]
            mix = (attention_layer(ctx, i, x) if is_attn
                   else mixer_layer(ctx, i, x, bcast, a_fix))
            xs[name] = ffn_layer(ctx, i, mix)
        # dump comparison against the engine's ENGINE_DUMP_EVERY files
        if args.dump:
            fp = Path(f"{args.dump}_L{i}.f32")
            if fp.exists():
                ref = np.fromfile(fp, dtype=np.float32).reshape(T, ctx.hidden)
                errs = {n: float(np.linalg.norm(xs[n] - ref) / np.linalg.norm(ref))
                        for n in variants}
                max_err = max(errs.values())
                line = " ".join(f"{n}={e:.2e}" for n, e in errs.items())
                flag = ""
                if args.tolerance > 0:
                    flag = "  <= tol" if max_err <= args.tolerance else "  EXCEEDS tol"
                    if max_err > args.tolerance:
                        worst = max_err
                print(f"layer {i:2d} ({'attn' if is_attn else 'lin '}): {line}{flag}",
                      flush=True)
        if i % 4 == 3:
            print(f"[progress] layer {i + 1}/{n_layer}  {time.time() - t_l0:.1f}s "
                  f"(total {time.time() - t_start:.1f}s)", flush=True)
        ctx.drop(f"blk.{i}.")

    out_w = ctx.t("output.weight").astype(np.float32)
    norm_w = ctx.t("output_norm.weight")
    for name in variants:
        xf = rmsnorm(xs[name], norm_w, ctx.eps)
        logits = (xf[-1].astype(np.float64) @ out_w.T.astype(np.float64)).astype(np.float32)
        print(f"{name}: top5 = {top5(logits)}", flush=True)

    if args.dump and args.tolerance > 0:
        if worst > args.tolerance:
            print(f"DUMP CHECK FAILED: max per-layer rel_l2 = {worst:.3e} > {args.tolerance:.1e}")
            return 1
        print(f"DUMP CHECK PASSED: max per-layer rel_l2 = {worst:.3e} <= {args.tolerance:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
