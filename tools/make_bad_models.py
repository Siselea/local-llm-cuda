"""Mutation tests for the engine's graceful-failure paths.

Takes the tiny GGUF fixtures and corrupts exactly one aspect each, then checks that
`engine info` reports a specific, actionable error instead of crashing or silently
producing garbage.

Usage: python tools/make_bad_models.py [fixtures_dir]
Writes build/fixtures/bad_*.gguf and prints the command lines to try.
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path


def find_kv_value_offset(data: bytes, key: bytes) -> tuple[int, int, int]:
    """Locate key's value: returns (value_offset, value_type, str_len_offset)."""
    klen = struct.pack("<Q", len(key))
    pos = data.find(klen + key)
    if pos < 0:
        raise SystemExit(f"metadata key {key!r} not found")
    p = pos + 8 + len(key)
    vtype = struct.unpack("<I", data[p:p + 4])[0]
    return p + 4, vtype, p


def patch_arch(src: Path, dst: Path, new_arch: bytes) -> None:
    data = bytearray(src.read_bytes())
    voff, vtype, _ = find_kv_value_offset(bytes(data), b"general.architecture")
    if vtype != 8:  # string
        raise SystemExit("unexpected metadata type for general.architecture")
    slen_off = voff
    slen = struct.unpack("<Q", data[slen_off:slen_off + 8])[0]
    assert slen == 5, slen
    data[slen_off + 8:slen_off + 8 + 5] = new_arch
    dst.write_bytes(bytes(data))


def patch_tensor_type(src: Path, dst: Path, tensor: bytes, new_type: int) -> None:
    data = bytearray(src.read_bytes())
    # tensor name length + name, then n_dims, dims, type id
    needle = struct.pack("<Q", len(tensor)) + tensor
    pos = data.find(needle)
    if pos < 0:
        raise SystemExit(f"tensor {tensor!r} not found")
    p = pos + len(needle)
    ndims = struct.unpack("<I", data[p:p + 4])[0]
    p += 4 + 8 * ndims
    old = struct.unpack("<I", data[p:p + 4])[0]
    data[p:p + 4] = struct.pack("<I", new_type)
    dst.write_bytes(bytes(data))
    print(f"  patched {tensor.decode()} type {old} -> {new_type}")


def patch_tensor_name(src: Path, dst: Path, old_name: bytes, new_name: bytes) -> None:
    assert len(old_name) == len(new_name)
    data = bytearray(src.read_bytes())
    needle = struct.pack("<Q", len(old_name)) + old_name
    pos = data.find(needle)
    if pos < 0:
        raise SystemExit(f"tensor {old_name!r} not found")
    data[pos + 8:pos + 8 + len(old_name)] = new_name
    dst.write_bytes(bytes(data))


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "build/fixtures")
    src = root / "tiny_gguf_moe" / "model.gguf"
    if not src.exists():
        raise SystemExit(f"{src} not found; run tools/verify.sh once to build fixtures")
    dst_dir = root / "bad_models"
    dst_dir.mkdir(parents=True, exist_ok=True)

    print("[bad models] unsupported architecture")
    patch_arch(src, dst_dir / "arch_mamba.gguf", b"mamba")

    print("[bad models] unsupported quant type (Q4_0 instead of F32)")
    patch_tensor_type(src, dst_dir / "quant_q4_0.gguf", b"blk.0.attn_q.weight", 2)

    print("[bad models] unknown tensor naming convention (embedding renamed)")
    patch_tensor_name(src, dst_dir / "name_unknown.gguf", b"token_embd.weight",
                      b"token_embx.weight")

    print("\nTry:")
    for f, what in [("arch_mamba.gguf", "architecture"),
                    ("quant_q4_0.gguf", "quant type"),
                    ("name_unknown.gguf", "tensor naming")]:
        print(f"  ./build/engine.exe info {dst_dir / f}    # {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
