"""Tiny MoE model in GGUF (fused 3-D expert tensors) + numpy reference logits.

Exercises the GGUF fused-expert path: blk.N.ffn_{gate,up,down}_exps.weight are
[n_expert, ...] tensors instead of one tensor per expert.

Usage: python tests/make_tiny_gguf_moe.py [out_dir]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

HIDDEN = 64
N_HEAD = 4
N_KV = 2
HEAD_DIM = 16
FFN = 128
EXPERT_INTER = 32
N_EXPERT = 4
TOP_K = 2
VOCAB = 128
N_LAYER = 2
L = 6
EPS = 1e-5
ROPE_BASE = 10000.0
SEED = 20261115

T_STRING, T_UINT32, T_FLOAT32, T_ARRAY = 8, 4, 6, 9
T_F32 = 0
T_Q8_0 = 8


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def quant_q8_0(arr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Per-32-element blocks: fp16 scale + int8 quants. Returns (bytes, dequantized)."""
    flat = arr.reshape(-1, 32).astype(np.float32)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1.0 / d)
    q = np.clip(np.rint(flat * inv), -127, 127).astype(np.int8)
    # the file stores fp16(d): the reference must use the same rounded scale
    d16 = d.astype(np.float16).astype(np.float32)
    dq = (d16 * q.astype(np.float32)).reshape(arr.shape)
    blob = b"".join(
        np.float16(d[i, 0]).tobytes() + q[i].tobytes() for i in range(flat.shape[0]))
    return blob, dq


deq: dict[str, np.ndarray] = {}


def write_gguf(path: Path, kv: list, tensors: list[tuple[str, np.ndarray]]) -> None:
    body = bytearray()
    body += struct.pack("<I", 0x46554747) + struct.pack("<I", 3)
    body += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for key, vtype, val in kv:
        body += _s(key) + struct.pack("<I", vtype)
        if vtype == T_STRING:
            body += _s(val)
        elif vtype == T_UINT32:
            body += struct.pack("<I", int(val))
        elif vtype == T_FLOAT32:
            body += struct.pack("<f", float(val))
        elif vtype == T_ARRAY:
            body += struct.pack("<I", T_STRING) + struct.pack("<Q", len(val))
            for token in val:
                body += _s(token)
        else:
            raise ValueError(f"kv type {vtype}")

    offset = 0
    infos, payloads = [], []
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dims = list(arr.shape[::-1])          # gguf stores innermost first
        if "_exps" in name:
            blob, dq = quant_q8_0(arr)
            infos.append((name, dims, offset, T_Q8_0))
            payloads.append(blob)
            deq[name] = dq
        else:
            infos.append((name, dims, offset, T_F32))
            payloads.append(arr.tobytes())
        offset = (offset + len(payloads[-1]) + 31) // 32 * 32
    for name, dims, off, tid in infos:
        body += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            body += struct.pack("<Q", d)
        body += struct.pack("<I", tid) + struct.pack("<Q", off)
    while len(body) % 32:
        body += b"\x00"
    for data in payloads:
        body += data
        body += b"\x00" * ((-len(data)) % 32)
    path.write_bytes(bytes(body))


def rmsnorm(x, w):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + EPS) * w).astype(np.float32)


def rope(x, positions, n_head, head_dim):
    half = head_dim // 2
    freqs = 1.0 / (ROPE_BASE ** (np.arange(0, half, dtype=np.float64) * 2.0 / head_dim))
    out = x.reshape(len(positions), n_head, head_dim).astype(np.float64).copy()
    for i, p in enumerate(positions):
        ang = p * freqs
        c, s = np.cos(ang), np.sin(ang)
        a = out[i, :, :half].copy()
        b = out[i, :, half:].copy()
        out[i, :, :half] = a * c - b * s
        out[i, :, half:] = a * s + b * c
    return out.reshape(len(positions), n_head * head_dim).astype(np.float32)


def attention(q, k, v, n_head, n_kv, head_dim):
    Lq, Lk = q.shape[0], k.shape[0]
    rep = n_head // n_kv
    outv = np.zeros((Lq, n_head * head_dim), dtype=np.float64)
    for h in range(n_head):
        kv = h // rep
        qh = q[:, h * head_dim:(h + 1) * head_dim].astype(np.float64)
        kh = k[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
        vh = v[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
        s = qh @ kh.T / np.sqrt(head_dim)
        s[np.arange(Lk)[None, :] > np.arange(Lq)[:, None]] = -np.inf
        m = s.max(axis=-1, keepdims=True)
        e = np.exp(s - m)
        p = e / e.sum(axis=-1, keepdims=True)
        outv[:, h * head_dim:(h + 1) * head_dim] = p @ vh
    return outv.astype(np.float32)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/fixtures/tiny_gguf_moe")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hd, kd = N_HEAD * HEAD_DIM, N_KV * HEAD_DIM

    def W(*shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    T: dict[str, np.ndarray] = {
        "token_embd.weight": W(VOCAB, HIDDEN),
        "output.weight": W(VOCAB, HIDDEN),
        "output_norm.weight": (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32),
    }
    for i in range(N_LAYER):
        p = f"blk.{i}."
        T[p + "attn_norm.weight"] = (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32)
        T[p + "ffn_norm.weight"] = (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32)
        T[p + "attn_q.weight"] = W(HIDDEN, HIDDEN)
        T[p + "attn_k.weight"] = W(kd, HIDDEN)
        T[p + "attn_v.weight"] = W(kd, HIDDEN)
        T[p + "attn_output.weight"] = W(HIDDEN, hd)
        T[p + "ffn_gate_inp.weight"] = W(N_EXPERT, HIDDEN, scale=0.05)
        T[p + "ffn_gate_exps.weight"] = W(N_EXPERT, EXPERT_INTER, HIDDEN)
        T[p + "ffn_up_exps.weight"] = W(N_EXPERT, EXPERT_INTER, HIDDEN)
        T[p + "ffn_down_exps.weight"] = W(N_EXPERT, HIDDEN, EXPERT_INTER)

    for i in range(N_LAYER):
        p = f"blk.{i}."
        for suffix in ("attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
                       "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            T[p + suffix + ".weight"] = T[p + suffix + ".weight"].astype(np.float32)

    tokens_list = [f"tok{i}" for i in range(VOCAB)]
    kv = [
        ("general.architecture", T_STRING, "llama"),
        ("general.alignment", T_UINT32, 32),
        ("llama.block_count", T_UINT32, N_LAYER),
        ("llama.embedding_length", T_UINT32, HIDDEN),
        ("llama.feed_forward_length", T_UINT32, FFN),
        ("llama.context_length", T_UINT32, 2048),
        ("llama.attention.head_count", T_UINT32, N_HEAD),
        ("llama.attention.head_count_kv", T_UINT32, N_KV),
        ("llama.attention.key_length", T_UINT32, HEAD_DIM),
        ("llama.attention.value_length", T_UINT32, HEAD_DIM),
        ("llama.attention.layer_norm_rms_epsilon", T_FLOAT32, EPS),
        ("llama.rope.freq_base", T_FLOAT32, ROPE_BASE),
        ("llama.expert_count", T_UINT32, N_EXPERT),
        ("llama.expert_used_count", T_UINT32, TOP_K),
        ("llama.expert_feed_forward_length", T_UINT32, EXPERT_INTER),
        ("llama.expert_weights_norm", T_UINT32, 1),
        ("tokenizer.ggml.model", T_STRING, "gpt2"),
        ("tokenizer.ggml.tokens", T_ARRAY, tokens_list),
        ("tokenizer.ggml.eos_token_id", T_UINT32, 1),
    ]
    write_gguf(out / "model.gguf", kv, list(T.items()))
    for k, v in deq.items():
        T[k] = v

    tokens_np = np.array([(SEED + i * 41) % VOCAB for i in range(L)], dtype=np.int32)
    tokens_np.tofile(out / "tokens.i32")

    # ---- reference forward ----
    x = T["token_embd.weight"][tokens_np].astype(np.float32)
    pos = np.arange(L)

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def moe(h, p):
        probs = softmax((h @ T[p + "ffn_gate_inp.weight"].T).astype(np.float64))
        outv = np.zeros_like(h, dtype=np.float64)
        for t in range(L):
            idx = np.argsort(-probs[t])[:TOP_K]
            w = probs[t][idx]
            w = w / w.sum()
            acc = np.zeros(HIDDEN, dtype=np.float64)
            for e, wt in zip(idx, w):
                g = h[t] @ T[p + "ffn_gate_exps.weight"][e].T
                u = h[t] @ T[p + "ffn_up_exps.weight"][e].T
                a = (g / (1.0 + np.exp(-g.astype(np.float64)))) * u
                acc += wt * (a @ T[p + "ffn_down_exps.weight"][e].T)
            outv[t] = acc
        return outv.astype(np.float32)

    for i in range(N_LAYER):
        p = f"blk.{i}."
        h = rmsnorm(x, T[p + "attn_norm.weight"])
        q = (h @ T[p + "attn_q.weight"].T).astype(np.float32)
        k = (h @ T[p + "attn_k.weight"].T).astype(np.float32)
        v = (h @ T[p + "attn_v.weight"].T).astype(np.float32)
        q = rope(q, pos, N_HEAD, HEAD_DIM)
        k = rope(k, pos, N_KV, HEAD_DIM)
        a = attention(q, k, v, N_HEAD, N_KV, HEAD_DIM)
        o = (a @ T[p + "attn_output.weight"].T).astype(np.float32)
        x = (x + o).astype(np.float32)
        h2 = rmsnorm(x, T[p + "ffn_norm.weight"])
        x = (x + moe(h2, p)).astype(np.float32)

    xf = rmsnorm(x, T["output_norm.weight"])
    logits = (xf @ T["output.weight"].T).astype(np.float32)
    logits.tofile(out / "ref_logits.f32")
    print(f"model    : {out}/model.gguf  layers={N_LAYER} experts={N_EXPERT} topk={TOP_K}")
    print(f"logits   : argmax(last)={int(logits[-1].argmax())} std={logits.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
