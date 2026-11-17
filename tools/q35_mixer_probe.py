"""Compare the engine's layer-0 SSM mixer against numpy for any qwen35 model."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds2_probe import F

def silu(x): return x / (1.0 + np.exp(-x))

def rmsnorm(x, w, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)

def l2norm(x, eps=1e-6):
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True) + eps)
    return (x * inv).astype(np.float64)

def main(model, dump, token):
    f = F(model); g = f.g
    hidden = int(g.get("qwen35moe.embedding_length") or g.get("qwen35.embedding_length")); eps = 1e-6
    n_vh = 32; n_kh = 16; kd = 128; vd = 128
    key_dim = n_kh * kd; value_dim = n_vh * vd; conv_dim = 2 * key_dim + value_dim
    conv_k = 4; p = "blk.0."
    x = f.t("token_embd.weight")[[token]].astype(np.float32)
    h = rmsnorm(x, f.t(p + "attn_norm.weight"), eps)
    hn = np.fromfile(dump + "_q35_hn.f32", dtype=np.float32).reshape(-1, hidden)
    print("hn :", np.linalg.norm(hn - h) / np.linalg.norm(h))
    mixed = (h @ f.t(p + "attn_qkv.weight").T).astype(np.float32)
    convw = f.t(p + "ssm_conv1d.weight")
    padded = np.concatenate([np.zeros((conv_k - 1, conv_dim), np.float32), mixed], axis=0)
    conv = np.zeros_like(mixed)
    for t in range(1):
        acc = np.zeros(conv_dim, np.float64)
        for k in range(conv_k):
            acc += convw[:, k].astype(np.float64) * padded[t + k].astype(np.float64)
        conv[t] = acc
    conv = silu(conv)
    ce = np.fromfile(dump + "_q35_conv.f32", dtype=np.float32).reshape(-1, conv_dim)
    print("conv:", np.linalg.norm(ce - conv) / np.linalg.norm(conv))
    q = l2norm(conv[:, :key_dim].reshape(1, n_kh, kd)) / np.sqrt(kd)
    k = l2norm(conv[:, key_dim:2 * key_dim].reshape(1, n_kh, kd))
    v = conv[:, 2 * key_dim:].reshape(1, n_vh, vd).astype(np.float64)
    q = np.repeat(q, 2, axis=1); k = np.repeat(k, 2, axis=1)
    a = (h @ f.t(p + "ssm_alpha.weight").T).astype(np.float64)
    b = (h @ f.t(p + "ssm_beta.weight").T).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-b))
    gd = -np.exp(f.t(p + "ssm_a").astype(np.float64))[None, :] * np.log1p(
        np.exp(a + f.t(p + "ssm_dt.bias").astype(np.float64)[None, :]))
    S = np.zeros((n_vh, kd, vd), np.float64)
    S *= np.exp(gd[0])[:, None, None]
    kv = np.einsum("hkv,hk->hv", S, k[0])
    delta = (v[0] - kv) * beta[0][:, None]
    S += k[0][:, :, None] * delta[:, None, :]
    o = np.einsum("hkv,hk->hv", S, q[0]).reshape(n_vh, vd).astype(np.float64)
    ms = np.mean(o ** 2, axis=-1, keepdims=True)
    z = (h @ f.t(p + "attn_gate.weight").T).astype(np.float32).reshape(n_vh, vd)
    o = (o / np.sqrt(ms + eps) * f.t(p + "ssm_norm.weight")) * silu(z)
    out = (o.reshape(1, value_dim).astype(np.float32) @ f.t(p + "ssm_out.weight").T)
    mid = np.fromfile(dump + "_q35_mid0.f32", dtype=np.float32).reshape(-1, hidden)
    eng = mid - x
    print("mixer:", np.linalg.norm(out - eng) / np.linalg.norm(eng))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
