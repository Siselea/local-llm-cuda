"""Numpy reference forward for a dense GGUF model (llama/qwen2 style, GQA + qkv bias).

Dequantizes with gguf.quants and runs a plain float32 forward, so the engine's
CUDA path can be compared against an implementation that shares no code with it.

Usage: python tools/make_gguf_ref.py <model.gguf> <out_dir> [n_tokens] [n_layer_cap]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyref.gguf_parse import read_gguf  # noqa: E402


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def main() -> int:
    model = sys.argv[1]
    out = Path(sys.argv[2])
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out.mkdir(parents=True, exist_ok=True)

    g = read_gguf(model)
    arch = str(g.get("general.architecture", ""))
    hidden = int(g.get(f"{arch}.embedding_length"))
    n_layer = int(g.get(f"{arch}.block_count"))
    n_head = int(g.get(f"{arch}.attention.head_count"))
    n_kv = int(g.get(f"{arch}.attention.head_count_kv", n_head))
    inter = int(g.get(f"{arch}.feed_forward_length"))
    eps = float(g.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5))
    base = float(g.get(f"{arch}.rope.freq_base", 1e4))
    head_dim = int(g.get(f"{arch}.attention.key_length", 0)) or hidden // n_head
    vocab = len(g.get("tokenizer.ggml.tokens"))

    ti = {t.name: t for t in g.tensors}
    cache: dict[str, np.ndarray] = {}

    def T(name: str) -> np.ndarray:
        if name in cache:
            return cache[name]
        t = ti[name]
        with open(model, "rb") as f:
            f.seek(g.data_offset + t.offset)
            raw = f.read(t.nbytes)
        if t.type_name == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        elif t.type_name == "BF16":
            arr = bf16_to_f32(raw)
        elif t.type_name == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            qtype = gguf.GGMLQuantizationType(t.type_id)
            block_elems, block_bytes = gguf.quants.GGML_QUANT_SIZES[qtype]
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, block_bytes)
            arr = dequantize(blocks, qtype).reshape(-1).astype(np.float32)
        arr = arr.reshape(t.shape)
        cache[name] = arr
        return arr

    rng = np.random.default_rng(11)
    tokens = rng.integers(0, vocab, size=n_tokens, dtype=np.int32)
    tokens.tofile(out / "tokens.i32")

    x = T("token_embd.weight")[tokens].astype(np.float32)
    pos = np.arange(n_tokens)

    def rmsnorm(v, w):
        ms = np.mean(v.astype(np.float64) ** 2, axis=-1, keepdims=True)
        return (v / np.sqrt(ms + eps) * w).astype(np.float32)

    half = head_dim // 2
    freqs = 1.0 / (base ** (np.arange(0, half, dtype=np.float64) * 2.0 / head_dim))

    def rope(v, nh):
        vv = v.reshape(n_tokens, nh, head_dim).astype(np.float64).copy()
        for i, p in enumerate(pos):
            ang = p * freqs
            c, s = np.cos(ang), np.sin(ang)
            a = vv[i, :, :half].copy()
            b = vv[i, :, half:].copy()
            vv[i, :, :half] = a * c - b * s
            vv[i, :, half:] = a * s + b * c
        return vv.reshape(n_tokens, nh * head_dim)

    rep = n_head // n_kv
    idx_q = np.arange(n_tokens)[:, None]
    idx_k = np.arange(n_tokens)[None, :]

    def attention(q, k, v):
        outv = np.zeros((n_tokens, n_head * head_dim), dtype=np.float64)
        for h in range(n_head):
            kv = h // rep
            qh = q[:, h * head_dim:(h + 1) * head_dim].astype(np.float64)
            kh = k[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
            vh = v[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
            s = qh @ kh.T / np.sqrt(head_dim)
            s[idx_k > idx_q] = -np.inf
            m = s.max(axis=-1, keepdims=True)
            e = np.exp(s - m)
            p = e / e.sum(axis=-1, keepdims=True)
            outv[:, h * head_dim:(h + 1) * head_dim] = p @ vh
        return outv.astype(np.float32)

    for i in range(n_layer):
        p = f"blk.{i}."
        h = rmsnorm(x, T(p + "attn_norm.weight"))
        q = (h @ T(p + "attn_q.weight").T + T(p + "attn_q.bias")).astype(np.float32)
        k = (h @ T(p + "attn_k.weight").T + T(p + "attn_k.bias")).astype(np.float32)
        v = (h @ T(p + "attn_v.weight").T + T(p + "attn_v.bias")).astype(np.float32)
        q = rope(q, n_head)
        k = rope(k, n_kv)
        a = attention(q, k, v)
        o = (a @ T(p + "attn_output.weight").T).astype(np.float32)
        x = (x + o).astype(np.float32)
        h2 = rmsnorm(x, T(p + "ffn_norm.weight"))
        gt = (h2 @ T(p + "ffn_gate.weight").T).astype(np.float32)
        ut = (h2 @ T(p + "ffn_up.weight").T).astype(np.float32)
        act = (gt / (1.0 + np.exp(-gt.astype(np.float64))) * ut).astype(np.float32)
        d = (act @ T(p + "ffn_down.weight").T).astype(np.float32)
        x = (x + d).astype(np.float32)

    xf = rmsnorm(x, T("output_norm.weight"))
    head = T("output.weight") if "output.weight" in ti else T("token_embd.weight")
    logits = (xf @ head.T).astype(np.float32)
    logits.tofile(out / "ref_logits.f32")
    meta = {"model": model, "arch": arch, "n_layer": n_layer, "hidden": hidden,
            "n_head": n_head, "n_kv": n_kv, "head_dim": head_dim, "vocab": vocab,
            "n_tokens": n_tokens, "eps": eps, "rope_base": base,
            "tokens": tokens.tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("arch", "n_layer", "hidden", "vocab", "n_tokens")},
                     ensure_ascii=False))
    print("argmax(last) =", int(logits[-1].argmax()), " std =", round(float(logits.std()), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
