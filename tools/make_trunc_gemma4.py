"""Write a truncated gemma4 GGUF (first N layers only) for fast llama.cpp anchors.

The full 7 GB checkpoint is too slow to copy/bisect repeatedly on this machine.
llama.cpp accepts a gemma4 file with block_count = N and exactly N layers, so
this tool re-serialises the first N layers (same metadata, arrays truncated) into
a ~1 GB file that loads quickly and yields the hidden state after N layers via
llama-server /embedding.  Zeroing a layer's attn_output.weight + ffn_down.weight
(and setting its layer_output_scale to 1) then makes that layer an exact
identity, which allows layer-by-layer bisection inside the small file.

Usage: python tools/make_trunc_gemma4.py --n 8 [--out _gemma4_trunc8.gguf]
The full checkpoint comes from GEMMA_MODEL or MODEL_DIR/gemma-4-12b-it-Q4_K_M.gguf.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import gguf
import numpy as np

import fuzz_gguf as F

def default_src() -> Path:
    explicit = os.environ.get("GEMMA_MODEL")
    if explicit:
        return Path(explicit)
    model_dir = os.environ.get("MODEL_DIR", "")
    name = "gemma-4-12b-it-Q4_K_M.gguf"
    return Path(model_dir) / name if model_dir else Path(name)


SRC = default_src()
BLOCK_KEYS = ("head_count_kv", "sliding_window_pattern")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--src", default=str(SRC))
    opts = ap.parse_args()
    n = opts.n
    src = Path(opts.src)
    out_dir = Path(os.environ.get("OUT_DIR", "."))
    out = Path(opts.out) if opts.out else out_dir / f"_gemma4_trunc{n}.gguf"
    r = gguf.GGUFReader(str(src), "r")
    fields = r.fields

    g = F.GGUF()
    for key, fld in fields.items():
        if key.startswith("GGUF."):            # synthetic header fields, not metadata
            continue
        tys = fld.types
        parts = fld.parts
        if len(tys) == 1:                     # scalar
            if key.endswith(".block_count"):
                g.add(key, 4, struct.pack("<I", n))
            elif tys[0].value == 8:            # STRING: len + bytes
                g.add(key, 8, bytes(parts[3]) + (bytes(parts[4]) if len(parts) > 4 else b""))
            else:
                g.add(key, tys[0].value, bytes(parts[3]))
        else:                                  # array
            elem_t = tys[1].value
            cnt = int.from_bytes(bytes(parts[4]), "little")
            if elem_t == 8:                    # STRING elements use (len, bytes) pairs
                elems = [bytes(parts[5 + 2 * i]) + bytes(parts[6 + 2 * i])
                         for i in range(cnt)]
            else:
                elems = [bytes(p) for p in parts[5:5 + cnt]]
            if any(key.endswith(k) for k in BLOCK_KEYS):
                elems = elems[:n]
            g.add(key, 9, struct.pack("<I", elem_t) + struct.pack("<Q", len(elems)) +
                  b"".join(elems))

    keep = ("token_embd.weight", "output_norm.weight", "rope_freqs.weight")
    n_t = 0
    for t in r.tensors:
        name = t.name
        if name in keep:
            pass
        elif name.startswith("blk."):
            layer = int(name.split(".")[1])
            if layer >= n:
                continue
        else:
            continue
        dims = list(reversed(t.shape))
        data = np.array(t.data).tobytes()
        g.tensor(name, dims, data, type_id=int(t.tensor_type.value))
        n_t += 1
    data = g.build()
    out.write_bytes(data)
    print(f"wrote {out} ({len(data)/1e6:.1f} MB, {n_t} tensors, block_count={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
