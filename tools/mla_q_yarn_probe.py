"""Third-opinion numpy q with the exact HF YaRN parameters."""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from tools.ds2_probe import F  # noqa: E402

f = F(r"<MODEL_DIR>\DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf")
g = f.g
hidden = int(g.get("deepseek2.embedding_length"))
n_head = int(g.get("deepseek2.attention.head_count"))
nope, rope = 128, 64
base, factor, orig, dim = 10000.0, 40.0, 4096, 64
ids = [549, 6077, 280, 7239]
T = len(ids)

corr = lambda beta: dim * math.log(orig / (beta * 2 * math.pi)) / (2 * math.log(base))
low = max(0.0, math.floor(corr(32)))
high = min(dim - 1.0, math.ceil(corr(1)))
half = dim // 2
inv = np.zeros(half)
for i in range(half):
    pos_freq = base ** (2 * i / dim)
    extrap = 1.0 / pos_freq
    interp = 1.0 / (factor * pos_freq)
    ramp = min(1.0, max(0.0, (i - low) / max(1e-9, high - low)))
    ef = 1.0 - ramp
    inv[i] = interp * (1 - ef) + extrap * ef
mscale = (0.1 * 1.0 * math.log(factor) + 1.0) / (0.1 * 0.0707 * math.log(factor) + 1.0)

hn = np.fromfile("build/mla0_hn.f32", dtype=np.float32).reshape(T, hidden)
q = (hn @ f.t("blk.0.attn_q.weight").T).astype(np.float64).reshape(T, n_head, nope + rope)
out = q.copy()
for t in range(T):
    ang = t * inv
    c, s = np.cos(ang) * mscale, np.sin(ang) * mscale
    a = out[t, :, nope::2].copy()
    b = out[t, :, nope + 1::2].copy()
    out[t, :, nope::2] = a * c - b * s
    out[t, :, nope + 1::2] = a * s + b * c
ref = np.asarray(out, dtype=np.float32).ravel()
np.asarray(out, dtype=np.float32).tofile("build/np_q0_yarn.f32")
def rope_var(kind, ms):
    o = q.copy()
    for t in range(T):
        ang = t * inv
        c, s = np.cos(ang) * ms, np.sin(ang) * ms
        if kind == "inter":
            a = o[t, :, nope::2].copy(); b = o[t, :, nope + 1::2].copy()
            o[t, :, nope::2] = a * c - b * s
            o[t, :, nope + 1::2] = a * s + b * c
        else:
            a = o[t, :, nope:nope + half].copy(); b = o[t, :, nope + half:].copy()
            o[t, :, nope:nope + half] = a * c[None, :] - b * s[None, :]
            o[t, :, nope + half:] = a * s[None, :] + b * c[None, :]
    return np.asarray(o, dtype=np.float32).ravel()

def rope_at(offset, kind, ms):
    o = q.copy()
    for t in range(T):
        ang = t * inv
        c, s = np.cos(ang) * ms, np.sin(ang) * ms
        if kind == "inter":
            a = o[t, :, offset:offset + rope:2].copy(); b = o[t, :, offset + 1:offset + rope:2].copy()
            o[t, :, offset:offset + rope:2] = a * c - b * s
            o[t, :, offset + 1:offset + rope:2] = a * s + b * c
        else:
            a = o[t, :, offset:offset + half].copy(); b = o[t, :, offset + half:offset + rope].copy()
            o[t, :, offset:offset + half] = a * c[None, :] - b * s[None, :]
            o[t, :, offset + half:offset + rope] = a * s[None, :] + b * c[None, :]
    return np.asarray(o, dtype=np.float32).ravel()

eng = np.fromfile("build/mla0_q.f32", dtype=np.float32)
for off in (0, 64, 128):
    for kind in ("inter", "half"):
        for ms in (mscale, 1.0):
            v = rope_at(off, kind, ms)
            r = np.linalg.norm(eng - v) / np.linalg.norm(v)
            if r < 0.15:
                print(f"MATCH engine vs rope[off={off}, {kind}, ms={ms:.4f}]: {r:.3e}")
for kind in ("inter", "half"):
    for ms in (mscale, 1.0):
        v = rope_var(kind, ms)
        print(f"engine vs rope[{kind}, mscale={ms:.4f}]: {np.linalg.norm(eng - v)/np.linalg.norm(v):.3e}")

for tag, path in [("engine", "build/mla0_q.f32"), ("chain", "build/chain_q0.f32")]:
    try:
        x = np.fromfile(path, dtype=np.float32)
    except OSError:
        print(tag, "missing")
        continue
    print(f"{tag}: rel_l2 vs numpy(YaRN) = {np.linalg.norm(x - ref) / np.linalg.norm(ref):.3e}")
