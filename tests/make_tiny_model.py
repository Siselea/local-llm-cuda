"""Generate a 2-layer tiny llama (embedding + N layers + norm + lm_head) and the
numpy reference logits for a token sequence.

Usage: python tests/make_tiny_model.py [out_dir]
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
N_LAYER = int(__import__('os').environ.get('N_LAYER', '2'))
L = 6
EPS = 1e-5
ROPE_BASE = 10000.0
SEED = 20261113


def bf16_round(x: np.ndarray) -> np.ndarray:
    u = np.asarray(x, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb).astype(np.uint32)
    h = (u >> 16).astype(np.uint16)
    return (h.astype(np.uint32) << 16).view(np.float32)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    hdr, payloads, off = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        nbytes = arr.size * 2
        hdr[name] = {"dtype": "BF16", "shape": list(arr.shape),
                     "data_offsets": [off, off + nbytes]}
        b = bf16_round(arr)
        payloads.append((b.view(np.uint32) >> 16).astype(np.uint16).tobytes())
        off += nbytes
    js = json.dumps(hdr, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(js)) + js + b"".join(payloads))


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
        x1 = out[i, :, :half].copy()
        x2 = out[i, :, half:].copy()
        out[i, :, :half] = x1 * c - x2 * s
        out[i, :, half:] = x1 * s + x2 * c
    return out.reshape(len(positions), n_head * head_dim).astype(np.float32)


def attention(q, k, v, n_head, n_kv, head_dim):
    Lq, Lk = q.shape[0], k.shape[0]
    rep = n_head // n_kv
    out = np.zeros((Lq, n_head * head_dim), dtype=np.float64)
    idx_q = np.arange(Lq)[:, None]
    idx_k = np.arange(Lk)[None, :]
    for h in range(n_head):
        kv = h // rep
        qh = q[:, h * head_dim:(h + 1) * head_dim].astype(np.float64)
        kh = k[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
        vh = v[:, kv * head_dim:(kv + 1) * head_dim].astype(np.float64)
        scores = qh @ kh.T / np.sqrt(head_dim)
        scores[idx_k > idx_q] = -np.inf
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        p = e / e.sum(axis=-1, keepdims=True)
        out[:, h * head_dim:(h + 1) * head_dim] = p @ vh
    return out.astype(np.float32)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/fixtures/tiny_model")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hd, kd = N_HEAD * HEAD_DIM, N_KV * HEAD_DIM

    def W(*shape):
        return bf16_round((rng.standard_normal(shape) * 0.02).astype(np.float32))

    T: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": W(VOCAB, HIDDEN),
        "lm_head.weight": W(VOCAB, HIDDEN),
        "model.norm.weight": (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32),
    }
    for i in range(N_LAYER):
        p = f"model.layers.{i}."
        T[p + "input_layernorm.weight"] = (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32)
        T[p + "post_attention_layernorm.weight"] = (1.0 + rng.standard_normal(HIDDEN) * 0.01).astype(np.float32)
        T[p + "self_attn.q_proj.weight"] = W(hd, HIDDEN)
        T[p + "self_attn.k_proj.weight"] = W(kd, HIDDEN)
        T[p + "self_attn.v_proj.weight"] = W(kd, HIDDEN)
        T[p + "self_attn.o_proj.weight"] = W(HIDDEN, hd)
        T[p + "mlp.gate_proj.weight"] = W(INTER, HIDDEN)
        T[p + "mlp.up_proj.weight"] = W(INTER, HIDDEN)
        T[p + "mlp.down_proj.weight"] = W(HIDDEN, INTER)

    # the fixture is stored as bf16, so the reference must use the rounded values
    T = {k: bf16_round(v) for k, v in T.items()}
    write_safetensors(out / "model.safetensors", T)
    cfg = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": N_LAYER, "hidden_size": HIDDEN,
        "intermediate_size": INTER, "vocab_size": VOCAB,
        "num_attention_heads": N_HEAD, "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM, "rope_theta": ROPE_BASE,
        "max_position_embeddings": 2048, "eos_token_id": 1,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    tokens = np.array([(SEED + i * 37) % VOCAB for i in range(L)], dtype=np.int32)
    tokens.tofile(out / "tokens.i32")
    np.save(out / "tokens.npy", tokens)

    # ---- reference forward ----
    x = T["model.embed_tokens.weight"][tokens].astype(np.float32)
    pos = np.arange(L)
    for i in range(N_LAYER):
        p = f"model.layers.{i}."
        h = rmsnorm(x, T[p + "input_layernorm.weight"])
        q = (h @ T[p + "self_attn.q_proj.weight"].T).astype(np.float32)
        k = (h @ T[p + "self_attn.k_proj.weight"].T).astype(np.float32)
        v = (h @ T[p + "self_attn.v_proj.weight"].T).astype(np.float32)
        q = rope(q, pos, N_HEAD, HEAD_DIM)
        k = rope(k, pos, N_KV, HEAD_DIM)
        a = attention(q, k, v, N_HEAD, N_KV, HEAD_DIM)
        o = (a @ T[p + "self_attn.o_proj.weight"].T).astype(np.float32)
        x = (x + o).astype(np.float32)
        h2 = rmsnorm(x, T[p + "post_attention_layernorm.weight"])
        g = (h2 @ T[p + "mlp.gate_proj.weight"].T).astype(np.float32)
        u = (h2 @ T[p + "mlp.up_proj.weight"].T).astype(np.float32)
        act = (g / (1.0 + np.exp(-g.astype(np.float64))) * u).astype(np.float32)
        d = (act @ T[p + "mlp.down_proj.weight"].T).astype(np.float32)
        x = (x + d).astype(np.float32)

    xf = rmsnorm(x, T["model.norm.weight"])
    logits = (xf @ T["lm_head.weight"].T).astype(np.float32)
    logits.tofile(out / "ref_logits.f32")
    print(f"model    : {out}  layers={N_LAYER} L={L}")
    print(f"logits   : {logits.shape}  argmax(last)={int(logits[-1].argmax())}  "
          f"mean={logits.mean():.4f} std={logits.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
