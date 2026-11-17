"""Generate a tiny random llama layer (safetensors) + numpy reference output.

The C++ test (tests/test_layer.cpp) runs the same layer through the CUDA path and
compares against ref_out.f32. Keeping the reference in numpy (not torch) means the
oracle has no dependency on the framework we are trying to replace.

Usage: python tests/make_tiny_layer.py [out_dir]
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
INTER = 128
VOCAB = 128
N_LAYER = 1
L = 6          # prompt length
EPS = 1e-5
ROPE_BASE = 10000.0
SEED = 20261112


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 -> bf16 (RNE) and back, so weights match what we upload."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb).astype(np.uint32)
    h = (u >> 16).astype(np.uint16)                 # bf16 bit pattern
    return (h.astype(np.uint32) << 16).view(np.float32)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    hdr, payloads, off = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        nbytes = arr.size * 2                      # bf16 payload
        hdr[name] = {"dtype": "BF16", "shape": list(arr.shape),
                     "data_offsets": [off, off + nbytes]}
        b = bf16_round(arr)                        # float32 with low 16 bits zero
        payloads.append((b.view(np.uint32) >> 16).astype(np.uint16).tobytes())
        off += nbytes
    js = json.dumps(hdr, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(js)) + js + b"".join(payloads))


def rmsnorm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + EPS) * w).astype(np.float32)


def rope(x: np.ndarray, positions: np.ndarray, n_head: int, head_dim: int) -> np.ndarray:
    half = head_dim // 2
    freqs = 1.0 / (ROPE_BASE ** (np.arange(0, half, dtype=np.float64) * 2.0 / head_dim))
    out = x.reshape(len(positions), n_head, head_dim).astype(np.float64).copy()
    for i, p in enumerate(positions):
        ang = p * freqs
        c, s = np.cos(ang), np.sin(ang)
        x1 = out[i, :, :half].copy()
        x2 = out[i, :, half:].copy()
        out[i, :, :half] = x1 * c - x2 * s
        out[i, :, half:] = x1 * s + x2 * c
    return out.reshape(len(positions), n_head * head_dim).astype(np.float32)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, n_head: int, n_kv: int,
              head_dim: int, causal_offset: int = 0) -> np.ndarray:
    Lq, Lk = q.shape[0], k.shape[0]
    rep = n_head // n_kv
    out = np.zeros((Lq, n_head * head_dim), dtype=np.float64)
    scale = 1.0 / np.sqrt(head_dim)
    idx_q = np.arange(Lq)[:, None] + causal_offset
    idx_k = np.arange(Lk)[None, :]
    for h in range(n_head):
        kv = h // rep
        qh = q[:, h * head_dim:(h + 1) * head_dim].astype(np.float64)
        kh = k[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
        vh = v[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
        scores = qh @ kh.T * scale
        scores[idx_k > idx_q] = -np.inf
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        p = e / e.sum(axis=-1, keepdims=True)
        out[:, h * head_dim:(h + 1) * head_dim] = p @ vh
    return out.astype(np.float32)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/fixtures/tiny_layer")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    def W(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    hd = N_HEAD * HEAD_DIM
    kd = N_KV * HEAD_DIM
    tensors = {
        "model.embed_tokens.weight": W(VOCAB, HIDDEN),
        "lm_head.weight": W(VOCAB, HIDDEN),
        "model.norm.weight": np.ones(HIDDEN, dtype=np.float32),
        "model.layers.0.input_layernorm.weight": (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32),
        "model.layers.0.post_attention_layernorm.weight": (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": W(hd, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": W(kd, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": W(kd, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": W(HIDDEN, hd),
        "model.layers.0.mlp.gate_proj.weight": W(INTER, HIDDEN),
        "model.layers.0.mlp.up_proj.weight": W(INTER, HIDDEN),
        "model.layers.0.mlp.down_proj.weight": W(HIDDEN, INTER),
    }
    # round weights to bf16 once, then use the rounded values for the reference too
    tensors = {k: bf16_round(v) for k, v in tensors.items()}
    write_safetensors(out / "model.safetensors", tensors)

    cfg = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": N_LAYER, "hidden_size": HIDDEN,
        "intermediate_size": INTER, "vocab_size": VOCAB,
        "num_attention_heads": N_HEAD, "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM, "rope_theta": ROPE_BASE,
        "max_position_embeddings": 2048, "eos_token_id": 1,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    x = (rng.standard_normal((L, HIDDEN)) * 0.5).astype(np.float32)
    pos = np.arange(L)

    ln1 = tensors["model.layers.0.input_layernorm.weight"]
    ln2 = tensors["model.layers.0.post_attention_layernorm.weight"]
    Wq = tensors["model.layers.0.self_attn.q_proj.weight"]
    Wk = tensors["model.layers.0.self_attn.k_proj.weight"]
    Wv = tensors["model.layers.0.self_attn.v_proj.weight"]
    Wo = tensors["model.layers.0.self_attn.o_proj.weight"]
    Wg = tensors["model.layers.0.mlp.gate_proj.weight"]
    Wu = tensors["model.layers.0.mlp.up_proj.weight"]
    Wd = tensors["model.layers.0.mlp.down_proj.weight"]

    h = rmsnorm(x, ln1)
    q = (h @ Wq.T).astype(np.float32)
    k = (h @ Wk.T).astype(np.float32)
    v = (h @ Wv.T).astype(np.float32)
    q = rope(q, pos, N_HEAD, HEAD_DIM)
    k = rope(k, pos, N_KV, HEAD_DIM)
    attn = attention(q, k, v, N_HEAD, N_KV, HEAD_DIM)
    o = (attn @ Wo.T).astype(np.float32)
    h1 = (x + o).astype(np.float32)
    n2 = rmsnorm(h1, ln2)
    gate = (n2 @ Wg.T).astype(np.float32)
    up = (n2 @ Wu.T).astype(np.float32)
    act = (gate / (1.0 + np.exp(-gate.astype(np.float64))) * up).astype(np.float32)
    d = (act @ Wd.T).astype(np.float32)
    y = (h1 + d).astype(np.float32)

    x.tofile(out / "input.f32")
    y.tofile(out / "ref_out.f32")

    # ---- KV cache path fixtures: prefill L tokens, then one decode step ----
    q.tofile(out / "q.f32")
    k.tofile(out / "k.f32")
    v.tofile(out / "v.f32")
    attn.tofile(out / "attn_ref.f32")

    x7 = (rng.standard_normal((1, HIDDEN)) * 0.5).astype(np.float32)
    h7 = rmsnorm(x7, ln1)
    q7 = (h7 @ Wq.T).astype(np.float32)
    k7 = (h7 @ Wk.T).astype(np.float32)
    v7 = (h7 @ Wv.T).astype(np.float32)
    q7 = rope(q7, np.array([L]), N_HEAD, HEAD_DIM)
    k7 = rope(k7, np.array([L]), N_KV, HEAD_DIM)
    k_all = np.concatenate([k, k7], axis=0)
    v_all = np.concatenate([v, v7], axis=0)
    attn7 = attention(q7, k_all, v_all, N_HEAD, N_KV, HEAD_DIM, causal_offset=L)
    x7.tofile(out / "x7.f32")
    q7.tofile(out / "q7.f32")
    k7.tofile(out / "k7.f32")
    v7.tofile(out / "v7.f32")
    attn7.tofile(out / "attn7_ref.f32")
    print(f"model    : {out}  (hidden={HIDDEN} heads={N_HEAD}/{N_KV} L={L})")
    print(f"output   : ref_out.f32  mean={y.mean():.5f} std={y.std():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
