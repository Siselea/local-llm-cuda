"""Tiny deepseek2 model with MLA attention + MoE (fused) + shared experts.

Layer 0 is dense, layer 1 is MoE (leading_dense_block_count = 1) so the fixture
exercises both FFN paths behind the MLA attention.

Routing (DeepSeek-V2): scores = softmax(x @ Wr^T) over all experts, top-k weights
are the raw softmax scores (norm_topk_prob = false, expert_weights_scale = 1.0) --
matches transformers' DeepseekV2TopkRouter.

Usage: python tests/make_tiny_gguf_ds2.py [out_dir]
"""

from __future__ import annotations

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
QK = QK_NOPE + QK_ROPE
FFN = 128
N_EXPERT = 4
TOP_K = 2
EXPERT_INTER = 32
N_SHARED = 2
SH_INTER = N_SHARED * EXPERT_INTER
VOCAB = 128
N_LAYER = 2
L = 5
EPS = 1e-6
ROPE_BASE = 10000.0
SEED = 20261127

T_STRING, T_UINT32, T_FLOAT32, T_ARRAY, T_F32, T_Q8_0 = 8, 4, 6, 9, 0, 8

deq: dict[str, np.ndarray] = {}


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def quant_q8_0(arr: np.ndarray):
    flat = arr.reshape(-1, 32).astype(np.float32)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d == 0, 0, 1.0 / np.where(d == 0, 1, d))
    q = np.clip(np.rint(flat * inv), -127, 127).astype(np.int8)
    d16 = d.astype(np.float16).astype(np.float32)
    blob = b"".join(np.float16(d[i, 0]).tobytes() + q[i].tobytes() for i in range(flat.shape[0]))
    return blob, (d16 * q.astype(np.float32)).reshape(arr.shape)


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
    infos, payloads = [], []
    offset = 0
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dims = list(arr.shape[::-1])
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
        body += data + b"\x00" * ((-len(data)) % 32)
    path.write_bytes(bytes(body))


def rmsnorm(x, w):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + EPS) * w).astype(np.float32)


def rope_part(x, positions, rope_dim, base):
    """Interleaved (complex) rotary, as in the DeepSeek-V2 reference."""
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
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/fixtures/tiny_gguf_ds2")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

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
        if i < 1:
            T[p + "ffn_gate.weight"] = W(FFN, HIDDEN)
            T[p + "ffn_up.weight"] = W(FFN, HIDDEN)
            T[p + "ffn_down.weight"] = W(HIDDEN, FFN)
        else:
            T[p + "ffn_gate_inp.weight"] = W(N_EXPERT, HIDDEN)
            T[p + "ffn_gate_exps.weight"] = W(N_EXPERT, EXPERT_INTER, HIDDEN)
            T[p + "ffn_up_exps.weight"] = W(N_EXPERT, EXPERT_INTER, HIDDEN)
            T[p + "ffn_down_exps.weight"] = W(N_EXPERT, HIDDEN, EXPERT_INTER)
            T[p + "ffn_gate_shexp.weight"] = W(SH_INTER, HIDDEN)
            T[p + "ffn_up_shexp.weight"] = W(SH_INTER, HIDDEN)
            T[p + "ffn_down_shexp.weight"] = W(HIDDEN, SH_INTER)

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
        ("deepseek2.leading_dense_block_count", T_UINT32, 1),
        ("deepseek2.expert_count", T_UINT32, N_EXPERT),
        ("deepseek2.expert_used_count", T_UINT32, TOP_K),
        ("deepseek2.expert_shared_count", T_UINT32, N_SHARED),
        ("deepseek2.expert_feed_forward_length", T_UINT32, EXPERT_INTER),
        ("deepseek2.expert_weights_scale", T_FLOAT32, 1.0),
        ("tokenizer.ggml.model", T_STRING, "gpt2"),
        ("tokenizer.ggml.tokens", T_ARRAY, [f"tok{i}" for i in range(VOCAB)]),
        ("tokenizer.ggml.eos_token_id", T_UINT32, 1),
    ]
    write_gguf(out / "model.gguf", kv, list(T.items()))
    for k, v in deq.items():
        T[k] = v

    tokens = np.array([(SEED + i * 31) % VOCAB for i in range(L)], dtype=np.int32)
    tokens.tofile(out / "tokens.i32")

    x = T["token_embd.weight"][tokens].astype(np.float32)
    pos = np.arange(L)
    scale = 1.0 / np.sqrt(QK)

    def attention_mla(h, p):
        q = (h @ T[p + "attn_q.weight"].T).reshape(L, N_HEAD, QK).astype(np.float32)
        kv_a = (h @ T[p + "attn_kv_a_mqa.weight"].T).astype(np.float32)
        c_kv = rmsnorm(kv_a[:, :KV_LORA], T[p + "attn_kv_a_norm.weight"])
        k_pe = kv_a[:, KV_LORA:]
        wb = T[p + "attn_kv_b.weight"].reshape(N_HEAD, QK_NOPE + V_DIM, KV_LORA)
        k_nope = np.einsum("lk,hdk->lhd", c_kv, wb[:, :QK_NOPE, :])
        v = np.einsum("lk,hdk->lhd", c_kv, wb[:, QK_NOPE:, :])
        q_rope = rope_part(q[:, :, QK_NOPE:], pos, QK_ROPE, ROPE_BASE)
        q = np.concatenate([q[:, :, :QK_NOPE].astype(np.float64), q_rope], axis=-1)
        k_pe_r = rope_part(k_pe[:, None, :], pos, QK_ROPE, ROPE_BASE)
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

    def dense_ffn(h, p):
        g = (h @ T[p + "ffn_gate.weight"].T).astype(np.float32)
        u = (h @ T[p + "ffn_up.weight"].T).astype(np.float32)
        a = (g / (1.0 + np.exp(-g.astype(np.float64))) * u).astype(np.float32)
        return (a @ T[p + "ffn_down.weight"].T).astype(np.float32)

    def moe(h, p):
        logits = (h @ T[p + "ffn_gate_inp.weight"].T).astype(np.float64)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        scores = e / e.sum(axis=-1, keepdims=True)
        outv = np.zeros((L, HIDDEN), dtype=np.float64)
        for t in range(L):
            idx = np.argsort(-scores[t])[:TOP_K]
            for e in idx:
                w = scores[t, e]
                g = (h[t] @ T[p + "ffn_gate_exps.weight"][e].T).astype(np.float64)
                u = (h[t] @ T[p + "ffn_up_exps.weight"][e].T).astype(np.float64)
                a = g / (1.0 + np.exp(-g)) * u
                outv[t] += w * (a @ T[p + "ffn_down_exps.weight"][e].T)
        g = (h @ T[p + "ffn_gate_shexp.weight"].T).astype(np.float64)
        u = (h @ T[p + "ffn_up_shexp.weight"].T).astype(np.float64)
        a = g / (1.0 + np.exp(-g)) * u
        outv += a @ T[p + "ffn_down_shexp.weight"].T
        return outv.astype(np.float32)

    for i in range(N_LAYER):
        p = f"blk.{i}."
        h = rmsnorm(x, T[p + "attn_norm.weight"])
        x = (x + attention_mla(h, p)).astype(np.float32)
        h2 = rmsnorm(x, T[p + "ffn_norm.weight"])
        x = (x + (dense_ffn(h2, p) if i < 1 else moe(h2, p))).astype(np.float32)

    xf = rmsnorm(x, T["output_norm.weight"])
    logits = (xf @ T["output.weight"].T).astype(np.float32)
    logits.tofile(out / "ref_logits.f32")
    print(f"model    : {out}/model.gguf  deepseek2 MLA+MoE layers={N_LAYER} L={L}")
    print(f"logits   : argmax(last)={int(logits[-1].argmax())} std={logits.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
