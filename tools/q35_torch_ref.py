"""Run transformers' Qwen3_5GatedDeltaNet on the real GGUF weights for one layer and
compare with our numpy reference mixers (removes the circularity of engine==numpy)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds2_probe import F  # noqa: E402

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet  # noqa: E402


def main() -> int:
    model_path = r"<MODEL_DIR>\Qwen3.5-9B-Q5_K_M.gguf"
    dump = Path("build")
    f = F(model_path)
    g = f.g
    T = 4
    hidden = 4096
    cfg = Qwen3_5TextConfig(
        hidden_size=hidden,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        intermediate_size=12288,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=10000000.0,
    )
    torch.manual_seed(0)
    mod = Qwen3_5GatedDeltaNet(cfg, 0).float().eval()
    p = "blk.0."
    sd = {
        "in_proj_qkv.weight": f.t(p + "attn_qkv.weight"),
        "in_proj_z.weight": f.t(p + "attn_gate.weight"),
        "in_proj_a.weight": f.t(p + "ssm_alpha.weight"),
        "in_proj_b.weight": f.t(p + "ssm_beta.weight"),
        "conv1d.weight": f.t(p + "ssm_conv1d.weight").reshape(8192, 1, 4),
        "dt_bias": f.t(p + "ssm_dt.bias"),
        "A_log": f.t(p + "ssm_a"),
        "norm.weight": f.t(p + "ssm_norm.weight"),
        "out_proj.weight": f.t(p + "ssm_out.weight"),
    }
    for k, v in sd.items():
        mod.state_dict()[k].copy_(torch.from_numpy(np.ascontiguousarray(v)))
    hn = np.fromfile(dump / "q35d_q35_hn0.f32", dtype=np.float32).reshape(T, hidden)
    with torch.no_grad():
        out = mod(torch.from_numpy(hn).unsqueeze(0)).squeeze(0).numpy()
    ref = np.fromfile(dump / "q35d_q35_x0.f32", dtype=np.float32).reshape(T, hidden)

    # mixer-only reference: layer output minus MLP cannot be isolated here, so compare
    # this module output against the engine's projection by re-running the engine layer
    # with the MLP disabled is not supported; instead we compare against numpy mixer.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from q35_full_probe import Model  # noqa: E402

    m = Model(model_path)
    x = f.t("token_embd.weight")[[760, 6511, 314, 9338]].astype(np.float32)
    zero_state = np.zeros((m.conv_k - 1, 2 * m.n_kh * m.kd + m.n_vh * m.vd), np.float32)
    # recompute the numpy mixer output (attention part) by running the layer and
    # subtracting the MLP contribution is messy; instead compare the full layer output
    x_np, _ = m.linear_layer(0, x, zero_state)
    print(f"torch module out vs engine full layer (should differ by MLP): "
          f"{np.linalg.norm(out)/np.linalg.norm(ref):.4f} ratio")
    # The engine's layer-0 output came from the same input; the module output is the
    # mixer only, so compare with numpy mixer = x + mixer; recover it from numpy by
    # computing the layer output and subtracting the MLP is not needed: compare the
    # *difference* between torch module output and our engine mixer output. The engine
    # dumps both, so use q35_attn-style dumps unavailable for layer 0. Fall back to a
    # consistency check: torch module vs numpy layer-attention output computed inline.
    h = (x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + m.eps) *
         f.t(p + "attn_norm.weight")).astype(np.float32)
    # numpy mixer from q35_probe logic
    key_dim = m.n_kh * m.kd
    value_dim = m.n_vh * m.vd
    conv_dim = 2 * key_dim + value_dim
    mixed = (h @ f.t(p + "attn_qkv.weight").T).astype(np.float32)
    convw = f.t(p + "ssm_conv1d.weight")
    padded = np.concatenate([zero_state, mixed], axis=0)
    conv = np.zeros_like(mixed)
    for t in range(T):
        acc = np.zeros(conv_dim, np.float64)
        for k in range(m.conv_k):
            acc += convw[:, k].astype(np.float64) * padded[t + k].astype(np.float64)
        conv[t] = acc
    conv = conv / (1.0 + np.exp(-conv))
    q = conv[:, :key_dim].reshape(T, m.n_kh, m.kd)
    k = conv[:, key_dim:2 * key_dim].reshape(T, m.n_kh, m.kd)
    v = conv[:, 2 * key_dim:].reshape(T, m.n_vh, m.vd).astype(np.float64)
    q = q / np.sqrt((q.astype(np.float64) ** 2).sum(-1, keepdims=True) + 1e-6) / np.sqrt(m.kd)
    k = k / np.sqrt((k.astype(np.float64) ** 2).sum(-1, keepdims=True) + 1e-6)
    rep = m.n_vh // m.n_kh
    q = np.repeat(q, rep, axis=1)
    k = np.repeat(k, rep, axis=1)
    a = (h @ f.t(p + "ssm_alpha.weight").T).astype(np.float64)
    b = (h @ f.t(p + "ssm_beta.weight").T).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-b))
    gd = -np.exp(f.t(p + "ssm_a").astype(np.float64))[None, :] * np.log1p(
        np.exp(a + f.t(p + "ssm_dt.bias").astype(np.float64)[None, :]))
    z = (h @ f.t(p + "attn_gate.weight").T).astype(np.float32).reshape(T, m.n_vh, m.vd)
    S = np.zeros((m.n_vh, m.kd, m.vd), np.float64)
    o = np.zeros((T, m.n_vh, m.vd), np.float64)
    for t in range(T):
        S *= np.exp(gd[t])[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - kv) * beta[t][:, None]
        S += k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", S, q[t])
    o = o.reshape(T * m.n_vh, m.vd)
    ms = np.mean(o ** 2, axis=-1, keepdims=True)
    o = (o / np.sqrt(ms + m.eps) * f.t(p + "ssm_norm.weight")) * (
        z.reshape(T * m.n_vh, m.vd) / (1.0 + np.exp(-z.reshape(T * m.n_vh, m.vd))))
    out_np = (o.reshape(T, value_dim).astype(np.float32) @ f.t(p + "ssm_out.weight").T)
    print(f"torch module vs numpy mixer: {np.linalg.norm(out - out_np)/np.linalg.norm(out_np):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
