"""Probe: compare the expanded MLA V magnitude with / without the kv_a RMSNorm."""
import sys

import numpy as np

sys.path.insert(0, ".")
from tools.ds2_probe import F  # noqa: E402

f = F(r"<MODEL_DIR>\DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf")
g = f.g
hidden = int(g.get("deepseek2.embedding_length"))
eps = float(g.get("deepseek2.attention.layer_norm_rms_epsilon"))
ids = [549, 6077, 280, 7239]
T = 4
x = f.t("token_embd.weight")[ids].astype(np.float32)
p = "blk.0."
n1 = f.t(p + "attn_norm.weight")
n1 = n1 + 1.0 if n1.mean() < 0.8 else n1
ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
hn = (x / np.sqrt(ms + eps) * n1).astype(np.float32)
print("hn norm", round(float(np.linalg.norm(hn)), 3))
ckv = (hn @ f.t(p + "attn_kv_a_mqa.weight").T).astype(np.float32)
kv_lora = int(g.get("deepseek2.attention.kv_lora_rank"))
kvn_raw = ckv[:, :kv_lora]
wkn = f.t(p + "attn_kv_a_norm.weight")
print("kv_a_norm mean", round(float(wkn.mean()), 4))
wkn = wkn + 1.0 if wkn.mean() < 0.8 else wkn
m2 = np.mean(kvn_raw.astype(np.float64) ** 2, axis=-1, keepdims=True)
kvn = (kvn_raw / np.sqrt(m2 + eps) * wkn).astype(np.float32)
print("kv_nope raw norm", round(float(np.linalg.norm(kvn_raw)), 3),
      "| normalised", round(float(np.linalg.norm(kvn)), 3))
wkb = f.t(p + "attn_kv_b.weight")
n_head = int(g.get("deepseek2.attention.head_count"))
nope, vdim = 128, 128
for tag, inp in [("normalised", kvn), ("raw", kvn_raw)]:
    kvb = (inp @ wkb.T).astype(np.float32).reshape(T, n_head, nope + vdim)
    v = kvb[..., nope:]
    print(f"V ({tag}) norm", round(float(np.linalg.norm(v)), 3))
