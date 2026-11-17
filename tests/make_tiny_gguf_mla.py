"""Tiny deepseek2 (MLA) model + numpy reference logits.

Attention is the MLA form used by DeepSeek-V2:
  q      = x @ Wq^T                      -> per head [qk_nope | qk_rope]
  kv_a   = x @ Wkv_a^T                   -> [kv_lora | rope]
  c_kv   = rmsnorm(kv_a[..., :kv_lora])
  k_nope = c_kv @ Wkb[:H*qk_nope]^T      -> per head qk_nope
  v      = c_kv @ Wkb[H*qk_nope:]^T      -> per head v_dim
  k      = [k_nope | k_pe] per head (k_pe shared, rope applied)
  scores = (q · k) / sqrt(qk_nope + qk_rope)

Usage: python tests/make_tiny_gguf_mla.py [out_dir]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

HIDDEN = 64
N_HEAD = 4
QK_NOPE = 8
QK_ROPE = 4
V_DIM = 8
KV_LORA = 16
FFN = 128
VOCAB = 128
N_LAYER = 2
L = 5
EPS = 1e-6
ROPE_BASE = 10000.0
SEED = 20261116

T_STRING, T_UINT32, T_FLOAT32, T_ARRAY, T_F32 = 8, 4, 6, 9, 0


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


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
        else:
            body += struct.pack("<I", T_STRING) + struct.pack("<Q", len(val))
            for token in val:
                body += _s(token)
    offset = 0
    infos = []
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        infos.append((name, list(arr.shape[::-1]), offset))
        offset = (offset + arr.nbytes + 31) // 32 * 32
    for name, dims, off in infos:
        body += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            body += struct.pack("<Q", d)
        body += struct.pack("<I", T_F32) + struct.pack("<Q", off)
    while len(body) % 32:
        body += b"\x00"
    for name, arr in tensors:
        data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        body += data + b"\x00" * ((-len(data)) % 32)
    path.write_bytes(bytes(body))


def rmsnorm(x, w):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + EPS) * w).astype(np.float32)


def rope_part(x, positions, rope_dim, base):
    """Rotate the last rope_dim of the final axis (in place-ish)."""
    half = rope_dim // 2
    freqs = 1.0 / (base ** (np.arange(0, half, dtype=np.float64) * 2.0 / rope_dim))
    out = x.astype(np.float64).copy()
    for i, p in enumerate(positions):
        ang = p * freqs
        c, s = np.cos(ang), np.sin(ang)
        a = out[i, :, 0:rope_dim:2].copy()
        b = out[i, :, 1:rope_dim:2].copy()
        out[i, :, 0:rope_dim:2] = a * c - b * s
        out[i, :, 1:rope_dim:2] = a * s + b * c
    return out


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/fixtures/tiny_gguf_mla")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    QK = QK_NOPE + QK_ROPE

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
        T[p + "attn_q.weight"] = W(N_HEAD * QK, HIDDEN)
        T[p + "attn_kv_a_mqa.weight"] = W(KV_LORA + QK_ROPE, HIDDEN)
        T[p + "attn_kv_a_norm.weight"] = (1.0 + rng.standard_normal(KV_LORA) * 0.01).astype(np.float32)
        T[p + "attn_kv_b.weight"] = W(N_HEAD * (QK_NOPE + V_DIM), KV_LORA)
        T[p + "attn_output.weight"] = W(HIDDEN, N_HEAD * V_DIM)
        T[p + "ffn_gate.weight"] = W(FFN, HIDDEN)
        T[p + "ffn_up.weight"] = W(FFN, HIDDEN)
        T[p + "ffn_down.weight"] = W(HIDDEN, FFN)

    kv = [
        ("general.architecture", T_STRING, "deepseek2"),
        ("general.alignment", T_UINT32, 32),
        ("deepseek2.block_count", T_UINT32, N_LAYER),
        ("deepseek2.embedding_length", T_UINT32, HIDDEN),
        ("deepseek2.feed_forward_length", T_UINT32, FFN),
        ("deepseek2.context_length", T_UINT32, 2048),
        ("deepseek2.attention.head_count", T_UINT32, N_HEAD),
        ("deepseek2.attention.head_count_kv", T_UINT32, N_HEAD),
        ("deepseek2.attention.key_length", T_UINT32, QK),
        ("deepseek2.attention.value_length", T_UINT32, V_DIM),
        ("deepseek2.attention.kv_lora_rank", T_UINT32, KV_LORA),
        ("deepseek2.attention.layer_norm_rms_epsilon", T_FLOAT32, EPS),
        ("deepseek2.rope.dimension_count", T_UINT32, QK_ROPE),
        ("deepseek2.rope.freq_base", T_FLOAT32, ROPE_BASE),
        ("deepseek2.leading_dense_block_count", T_UINT32, N_LAYER),
        ("tokenizer.ggml.model", T_STRING, "gpt2"),
        ("tokenizer.ggml.tokens", T_ARRAY, [f"tok{i}" for i in range(VOCAB)]),
        ("tokenizer.ggml.eos_token_id", T_UINT32, 1),
    ]
    write_gguf(out / "model.gguf", kv, list(T.items()))

    tokens = np.array([(SEED + i * 29) % VOCAB for i in range(L)], dtype=np.int32)
    tokens.tofile(out / "tokens.i32")

    # ---- reference forward ----
    x = T["token_embd.weight"][tokens].astype(np.float32)
    pos = np.arange(L)
    scale = 1.0 / np.sqrt(QK)

    def attention_mla(h, p):
        q = (h @ T[p + "attn_q.weight"].T).reshape(L, N_HEAD, QK).astype(np.float32)
        kv_a = (h @ T[p + "attn_kv_a_mqa.weight"].T).astype(np.float32)
        c_kv = rmsnorm(kv_a[:, :KV_LORA], T[p + "attn_kv_a_norm.weight"])
        k_pe = kv_a[:, KV_LORA:]
        wb = T[p + "attn_kv_b.weight"].reshape(N_HEAD, QK_NOPE + V_DIM, KV_LORA)
        # per-head [k_nope | v] blocks, as in the DeepSeek-V2 reference
        k_nope = np.einsum("lk,hdk->lhd", c_kv, wb[:, :QK_NOPE, :])
        v = np.einsum("lk,hdk->lhd", c_kv, wb[:, QK_NOPE:, :])

        q_rope = rope_part(q[:, :, QK_NOPE:], pos, QK_ROPE, ROPE_BASE)
        q = np.concatenate([q[:, :, :QK_NOPE].astype(np.float64), q_rope], axis=-1)
        k_pe_r = rope_part(k_pe[:, None, :], pos, QK_ROPE, ROPE_BASE)  # [L,1,rope]
        k = np.concatenate([k_nope.astype(np.float64),
                            np.repeat(k_pe_r, N_HEAD, axis=1)], axis=-1)

        outv = np.zeros((L, N_HEAD, V_DIM), dtype=np.float64)
        for hh in range(N_HEAD):
            s = q[:, hh, :] @ k[:, hh, :].T * scale
            s[np.arange(L)[None, :] > np.arange(L)[:, None]] = -np.inf
            m = s.max(axis=-1, keepdims=True)
            e = np.exp(s - m)
            pr = e / e.sum(axis=-1, keepdims=True)
            outv[:, hh, :] = pr @ v[:, hh, :].astype(np.float64)
        o = outv.reshape(L, N_HEAD * V_DIM).astype(np.float32)
        return (o @ T[p + "attn_output.weight"].T).astype(np.float32)

    for i in range(N_LAYER):
        p = f"blk.{i}."
        h = rmsnorm(x, T[p + "attn_norm.weight"])
        o = attention_mla(h, p)
        x = (x + o).astype(np.float32)
        h2 = rmsnorm(x, T[p + "ffn_norm.weight"])
        g = (h2 @ T[p + "ffn_gate.weight"].T).astype(np.float32)
        u = (h2 @ T[p + "ffn_up.weight"].T).astype(np.float32)
        a = (g / (1.0 + np.exp(-g.astype(np.float64))) * u).astype(np.float32)
        d = (a @ T[p + "ffn_down.weight"].T).astype(np.float32)
        x = (x + d).astype(np.float32)

    xf = rmsnorm(x, T["output_norm.weight"])
    logits = (xf @ T["output.weight"].T).astype(np.float32)
    logits.tofile(out / "ref_logits.f32")
    print(f"model    : {out}/model.gguf  deepseek2 layers={N_LAYER} L={L}")
    print(f"logits   : argmax(last)={int(logits[-1].argmax())} std={logits.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
