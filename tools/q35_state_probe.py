import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
from ds2_probe import F

def silu(x): return x / (1.0 + np.exp(-x))

def rmsnorm(x, w, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)

def l2norm(x, eps=1e-6):
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True) + eps)
    return (x * inv).astype(np.float64)

f = F(r"<MODEL_DIR>\Qwen3.5-9B-Q5_K_M.gguf")
g = f.g
hidden = 4096; eps = 1e-6
n_vh = 32; n_kh = 16; kd = 128; vd = 128
key_dim = n_kh * kd; value_dim = n_vh * vd; conv_dim = 2 * key_dim + value_dim
conv_k = 4
p = "blk.0."
wqkv = f.t(p + "attn_qkv.weight"); wz = f.t(p + "attn_gate.weight")
wa = f.t(p + "ssm_alpha.weight"); wb = f.t(p + "ssm_beta.weight")
convw = f.t(p + "ssm_conv1d.weight")
a_log = f.t(p + "ssm_a"); dtb = f.t(p + "ssm_dt.bias")
normw = f.t(p + "ssm_norm.weight"); wout = f.t(p + "ssm_out.weight")
ln1 = f.t(p + "attn_norm.weight")
ids = [760, 6511]
x_all = f.t("token_embd.weight")[ids].astype(np.float32)

def mixer(x, conv_state, rec_state):
    T = x.shape[0]
    h = rmsnorm(x, ln1, eps)
    mixed = (h @ wqkv.T).astype(np.float32)
    padded = np.concatenate([conv_state, mixed], axis=0)
    conv = np.zeros_like(mixed)
    for t in range(T):
        acc = np.zeros(conv_dim, np.float64)
        for k in range(conv_k):
            acc += convw[:, k].astype(np.float64) * padded[t + k].astype(np.float64)
        conv[t] = acc
    conv = silu(conv)
    keep = padded[-conv_k + 1:]   # oldest-first, matching the reference conv cache
    q = l2norm(conv[:, :key_dim].reshape(T, n_kh, kd)) / np.sqrt(kd)
    k = l2norm(conv[:, key_dim:2 * key_dim].reshape(T, n_kh, kd))
    v = conv[:, 2 * key_dim:].reshape(T, n_vh, vd).astype(np.float64)
    q = np.repeat(q, 2, axis=1); k = np.repeat(k, 2, axis=1)
    a = (h @ wa.T).astype(np.float64); b = (h @ wb.T).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-b))
    gd = -np.exp(a_log.astype(np.float64))[None, :] * np.log1p(np.exp(a + dtb[None, :].astype(np.float64)))
    z = (h @ wz.T).astype(np.float32).reshape(T, n_vh, vd)
    S = rec_state.copy()
    o = np.zeros((T, n_vh, vd), np.float64)
    for t in range(T):
        S *= np.exp(gd[t])[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - kv) * beta[t][:, None]
        S += k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", S, q[t])
    o = o.reshape(T * n_vh, vd)
    ms = np.mean(o ** 2, axis=-1, keepdims=True)
    o = (o / np.sqrt(ms + eps) * normw) * silu(z.reshape(T * n_vh, vd))
    out = (o.reshape(T, value_dim).astype(np.float32) @ wout.T)
    return out, keep, S

zero_cs = np.zeros((conv_k - 1, conv_dim), np.float32)
zero_S = np.zeros((n_vh, kd, vd), np.float64)
out_chunk, _, _ = mixer(x_all, zero_cs, zero_S)
o0, cs, S = mixer(x_all[:1], zero_cs, zero_S)
o1, _, _ = mixer(x_all[1:], cs, S)
print("chunk[1] vs incremental[1]:", np.linalg.norm(o1[0] - out_chunk[1]) / np.linalg.norm(out_chunk[1]))
# compare against engine dumps (layer-0 output only; the layer also has the MLP, so we
# compare the mixer part by adding the residual here)
def layer0_out(x, conv_state, rec_state):
    mix, keep, S = mixer(x, conv_state, rec_state)
    x1 = (x + mix).astype(np.float32)
    h2 = rmsnorm(x1, f.t(p + "post_attention_norm.weight"), eps)
    gg = (h2 @ f.t(p + "ffn_gate.weight").T).astype(np.float32)
    uu = (h2 @ f.t(p + "ffn_up.weight").T).astype(np.float32)
    act = (silu(gg.astype(np.float64)) * uu).astype(np.float32)
    d = (act @ f.t(p + "ffn_down.weight").T).astype(np.float32)
    return (x1 + d).astype(np.float32), keep, S
import os
l0_chunk, _, _ = layer0_out(x_all, zero_cs, zero_S)
a, cs, S = layer0_out(x_all[:1], zero_cs, zero_S)
b, _, _ = layer0_out(x_all[1:], cs, S)
if os.path.exists("build/c_L0.f32"):
    c = np.fromfile("build/c_L0.f32", dtype=np.float32).reshape(-1, hidden)
    print("engine chunked L0 vs numpy:", np.linalg.norm(l0_chunk - c) / np.linalg.norm(c))
if os.path.exists("build/i_L0.f32"):
    i = np.fromfile("build/i_L0.f32", dtype=np.float32).reshape(-1, hidden)
    print("engine incremental L0 vs numpy:", np.linalg.norm(b - i) / np.linalg.norm(i))
    print("engine incremental L0 vs engine chunked:", np.linalg.norm(i - c[1:]) / np.linalg.norm(c[1:]))
