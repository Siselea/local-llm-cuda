"""GEMV fixtures: real GGUF quantized rows x random vector -> reference y = W @ x.

Usage: python tools/make_gemv_fixture.py <model.gguf> <tensor> <out_dir> [n_out] [seed]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf.quants import dequantize, GGML_QUANT_SIZES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyref.gguf_parse import read_gguf  # noqa: E402

QTYPE = {
    12: gguf.GGMLQuantizationType.Q4_K,
    13: gguf.GGMLQuantizationType.Q5_K,
    14: gguf.GGMLQuantizationType.Q6_K,
    8: gguf.GGMLQuantizationType.Q8_0,
}


def main() -> int:
    model, name, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n_out_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    out_dir.mkdir(parents=True, exist_ok=True)

    g = read_gguf(model)
    info = next((t for t in g.tensors if t.name == name), None)
    if info is None:
        raise SystemExit(f"tensor {name} not found")
    if info.type_id not in QTYPE:
        raise SystemExit(f"{name}: unsupported qtype {info.type_name}")

    qtype = QTYPE[info.type_id]
    block_elems, block_bytes = GGML_QUANT_SIZES[qtype]
    n_out, n_in = info.shape[0], info.shape[1]

    blocks_per_row = n_in // block_elems
    rows = min(n_out, n_out_cap)
    n_bytes = rows * blocks_per_row * block_bytes
    with open(model, "rb") as f:
        f.seek(g.data_offset + info.offset)
        raw = f.read(n_bytes)

    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_bytes).reshape(-1, block_bytes)
    W = dequantize(blocks, qtype).reshape(rows, n_in).astype(np.float32)

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n_in) * 0.5).astype(np.float32)
    y = (W @ x).astype(np.float32)

    (out_dir / "blocks.bin").write_bytes(raw)
    x.tofile(out_dir / "x.f32")
    y.tofile(out_dir / "y.f32")
    meta = {
        "model": str(model), "tensor": name, "qtype": info.type_name,
        "qtype_id": info.type_id, "n_out": rows, "n_in": int(n_in),
        "block_elems": int(block_elems), "block_bytes": int(block_bytes),
        "bytes": len(raw),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
