"""Third-opinion numpy check of the MLA q vector (per-head [qk_nope | rope])."""
import sys

import numpy as np

sys.path.insert(0, ".")
from tools.ds2_probe import F  # noqa: E402

f = F(r"<MODEL_DIR>\DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf")
g = f.g
hidden = int(g.get("deepseek2.embedding_length"))
n_head = int(g.get("deepseek2.attention.head_count"))
nope, rope = 128, 64
base = float(g.get("deepseek2.rope.freq_base"))
ids = [549, 6077, 280, 7239]
T = len(ids)
p = "blk.0."

hn = np.fromfile("build/mla0_hn.f32", dtype=np.float32).reshape(T, hidden)
q = (hn @ f.t(p + "attn_q.weight").T).astype(np.float32).reshape(T, n_head, nope + rope)

half = rope // 2
freqs = 1.0 / (base ** (np.arange(half, dtype=np.float64) * 2.0 / rope))
out = q.astype(np.float64).copy()
for t in range(T):
    ang = t * freqs
    c, s = np.cos(ang), np.sin(ang)
    a = out[t, :, nope:][:, 0:rope:2].copy()
    b = out[t, :, nope:][:, 1:rope:2].copy()
    out[t, :, nope::2] = a * c - b * s
    out[t, :, nope + 1::2] = a * s + b * c
np.asarray(out, dtype=np.float32).tofile("build/np_q0.f32")

for tag, path in [("engine", "build/mla0_q.f32"), ("chain", "build/chain_q0.f32")]:
    try:
        x = np.fromfile(path, dtype=np.float32)
    except OSError:
        print(tag, "missing")
        continue
    if x.size != out.size:
        print(tag, "size", x.size, out.size)
        continue
    print(f"{tag}: rel_l2 vs numpy = {np.linalg.norm(x - out.ravel()) / np.linalg.norm(out):.3e}")
