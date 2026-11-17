"""Compare transformers' Qwen3_5Attention on real GGUF weights against the engine's
dumped mixer output for one full-attention layer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds2_probe import F  # noqa: E402

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5Attention,
    Qwen3_5TextRotaryEmbedding,
)


def main() -> int:
    model_path = r"<MODEL_DIR>\Qwen3.5-9B-Q5_K_M.gguf"
    dump = Path("build")
    f = F(model_path)
    cfg = Qwen3_5TextConfig(
        hidden_size=4096, num_attention_heads=16, num_key_value_heads=4, head_dim=256,
        intermediate_size=12288, num_hidden_layers=1, layer_types=["full_attention"],
        rms_norm_eps=1e-6, rope_theta=10000000.0, partial_rotary_factor=0.25,
    )
    mod = Qwen3_5Attention(cfg, 0).float().eval()
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    p = "blk.3."
    sd = {
        "q_proj.weight": f.t(p + "attn_q.weight"),
        "k_proj.weight": f.t(p + "attn_k.weight"),
        "v_proj.weight": f.t(p + "attn_v.weight"),
        "o_proj.weight": f.t(p + "attn_output.weight"),
        # the reference module stores zero-centred weights (applied as 1+w), while
        # GGUF has the converted ones -> subtract 1 when loading into the module
        "q_norm.weight": f.t(p + "attn_q_norm.weight") - 1.0,
        "k_norm.weight": f.t(p + "attn_k_norm.weight") - 1.0,
    }
    for k, v in sd.items():
        mod.state_dict()[k].copy_(torch.from_numpy(np.ascontiguousarray(v)))

    x_in = np.fromfile(dump / "q35a_q35_attn_in.f32", dtype=np.float32)
    mid = np.fromfile(dump / "q35a_q35_attn_mid.f32", dtype=np.float32)
    T = x_in.size // 4096
    x_in = x_in.reshape(T, 4096)
    mid = mid.reshape(T, 4096)
    pos = torch.arange(T).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(torch.from_numpy(x_in).unsqueeze(0), pos)
    xt = torch.from_numpy(x_in).unsqueeze(0)
    with torch.no_grad():
        hh = mod.q_proj(xt).view(T, -1, 2 * 256)
        q_raw = hh[..., :256]
        qn = mod.q_norm(q_raw)
        k_raw = mod.k_proj(xt).view(T, -1, 256)
        kn = mod.k_norm(k_raw)
    w_q = f.t(p + "attn_q_norm.weight")
    q_np = q_raw.numpy()
    qn_np = (q_np / np.sqrt(np.mean(q_np.astype(np.float64) ** 2, -1, keepdims=True) + 1e-6)
             * w_q).astype(np.float32)
    print("q_norm torch vs numpy:", np.linalg.norm(qn.numpy() - qn_np) / np.linalg.norm(qn_np))
    print("cos shape:", tuple(cos.shape), "dim", cos.dim())
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    with torch.no_grad():
        qr_t, kr_t = apply_rotary_pos_emb(qn.transpose(0, 1), kn.transpose(0, 1), cos, sin)
    def rope_np(x, T):
        out = x.astype(np.float64).copy()
        half = 32
        for t in range(T):
            ang = t * (1.0 / (10000000.0 ** (np.arange(half) * 2.0 / 64)))
            c, s2 = np.cos(ang), np.sin(ang)
            a = out[t, ..., :half].copy()
            b = out[t, ..., half:64].copy()
            out[t, ..., :half] = a * c - b * s2
            out[t, ..., half:64] = a * s2 + b * c
        return out
    qr_np = rope_np(qn_np, T)
    print("rope torch vs numpy:", np.linalg.norm(qr_t.numpy() - qr_np) / np.linalg.norm(qr_np))
    with torch.no_grad():
        neg = torch.finfo(torch.float32).min
        tri = torch.triu(torch.ones(T, T), diagonal=1).bool()
        bias = torch.zeros(1, 1, T, T)
        bias.masked_fill_(tri, neg)
        out, _ = mod(xt, position_embeddings=(cos, sin), attention_mask=bias)
        out = out.squeeze(0).numpy()
    mixer_eng = mid - x_in
    print(f"torch attn vs engine mixer: {np.linalg.norm(out - mixer_eng)/np.linalg.norm(out):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
