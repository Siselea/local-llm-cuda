"""Numeric check of the Q5_1 device dot product used by the fused MoE kernels.

Compares the formula implemented in kernels.h (qrow_dot<4>/qrows4<4>) against the
CPU dequant reference on a real GGUF tensor.
"""
import struct
import sys

import numpy as np

sys.path.insert(0, ".")
from pyref.gguf_parse import read_gguf  # noqa: E402

PATH = r"<MODEL_DIR>\DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"


def half(bits):
    return float(np.frombuffer(struct.pack("<H", bits), dtype=np.float16)[0])


def ref_block(p):
    """llama.cpp / our cpu_matvec Q5_1 dequant."""
    d = half(struct.unpack_from("<H", p, 0)[0])
    m = half(struct.unpack_from("<H", p, 2)[0])
    qh = struct.unpack_from("<I", p, 4)[0]
    qs = p[8:24]
    out = np.zeros(32, dtype=np.float32)
    for i in range(32):
        low = (qs[i] & 0xF) if i < 16 else (qs[i - 16] >> 4)
        hi = (qh >> i) & 1
        out[i] = d * float(low | (hi << 4)) + m
    return out


def kernel_block(p, lane):
    """Exactly what qrows4<4> computes for element `lane` of the block."""
    d = half(struct.unpack_from("<H", p, 0)[0])
    m = half(struct.unpack_from("<H", p, 2)[0])
    qh = struct.unpack_from("<I", p, 4)[0]
    qs = p[8:24]
    low = (qs[lane] & 0xF) if lane < 16 else (qs[lane - 16] >> 4)
    hi = (qh >> lane) & 1
    return d * float(low | (hi << 4)) + m


def main():
    g = read_gguf(PATH)
    t = [x for x in g.tensors if x.name == "blk.3.ffn_down_exps.weight"][0]
    print("tensor:", t.name, t.shape, t.type_name)
    with open(PATH, "rb") as f:
        base = g.data_offset + t.offset
        bad = 0
        for blk in range(8):
            f.seek(base + blk * 24)
            p = f.read(24)
            ref = ref_block(p)
            ker = np.array([kernel_block(p, lane) for lane in range(32)], dtype=np.float32)
            rel = np.linalg.norm(ker - ref) / max(1e-12, np.linalg.norm(ref))
            if rel > 1e-6:
                bad += 1
                print(f"block {blk}: MISMATCH rel={rel:.3e}")
                print("  ref[:6]", np.round(ref[:6], 4), " ker[:6]", np.round(ker[:6], 4))
        print("blocks checked: 8, mismatched:", bad)


if __name__ == "__main__":
    main()
