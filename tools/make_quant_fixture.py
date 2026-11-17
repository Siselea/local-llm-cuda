"""Extract a quantized tensor block from a GGUF and write a dequant fixture.

Reference comes from gguf.quants (the upstream ggml-mirroring implementation),
so the CUDA kernel is validated against an oracle we did not write.

Usage: python tools/make_quant_fixture.py <model.gguf> <tensor_name> <out_dir>
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
    6: gguf.GGMLQuantizationType.Q5_0,
    1: gguf.GGMLQuantizationType.F16,
}


def main() -> int:
    model, name, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    g = read_gguf(model)
    info = next((t for t in g.tensors if t.name == name), None)
    if info is None:
        raise SystemExit(f"tensor {name} not found in {model}")
    if info.type_id not in QTYPE:
        raise SystemExit(f"tensor {name}: qtype {info.type_name} not supported by fixture tool")

    with open(model, "rb") as f:
        f.seek(g.data_offset + info.offset)
        raw = f.read(info.nbytes)

    qtype = QTYPE[info.type_id]
    block_elems, block_bytes = GGML_QUANT_SIZES[qtype]
    n_blocks = len(raw) // block_bytes
    if len(sys.argv) > 4:                     # optional cap, keeps fixtures small
        n_blocks = min(n_blocks, int(sys.argv[4]))
        raw = raw[: n_blocks * block_bytes]
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * block_bytes).reshape(
        n_blocks, block_bytes)

    ref = dequantize(blocks, qtype).reshape(-1).astype(np.float32)
    (out_dir / "blocks.bin").write_bytes(raw)
    ref.tofile(out_dir / "ref.f32")
    meta = {
        "model": str(model), "tensor": name, "qtype": info.type_name,
        "qtype_id": info.type_id, "n_blocks": int(n_blocks),
        "block_elems": int(block_elems), "block_bytes": int(block_bytes),
        "n_elements": int(info.n_elements),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
