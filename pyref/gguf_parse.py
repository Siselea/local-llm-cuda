"""Minimal GGUF reader (header + metadata + tensor table only).

Deliberately written from the spec, not using the `gguf` pip package, because
the C++ implementation has to do exactly the same byte-level work and the two
must stay diff-able.

GGUF v2/v3 layout:
  magic   u32  "GGUF"
  version u32
  n_tensors u64
  n_kv      u64
  kv pairs ...
  tensor infos ...
  pad to `general.alignment` (default 32)
  tensor data
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = 0x46554747  # "GGUF"

# gguf value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<?",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}

# tensor type id -> (name, block_size, bytes_per_block)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
}


class GGUFError(RuntimeError):
    pass


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    type_id: int
    offset: int
    n_elements: int
    nbytes: int
    type_name: str

    @property
    def bpp(self) -> float:
        return self.nbytes * 8.0 / max(self.n_elements, 1)


@dataclass
class GGUFModel:
    path: Path
    version: int
    kv: dict = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    data_offset: int = 0
    file_size: int = 0
    alignment: int = 32

    def get(self, key: str, default=None):
        return self.kv.get(key, default)

    def meta_int(self, key: str) -> int:
        v = self.kv.get(key)
        if not isinstance(v, int):
            raise GGUFError(f"missing/invalid int metadata: {key}")
        return v

    def meta_str(self, key: str) -> str:
        v = self.kv.get(key)
        if not isinstance(v, str):
            raise GGUFError(f"missing/invalid string metadata: {key}")
        return v


class _Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise GGUFError(f"truncated at {self.pos}+{n} (size {len(self.buf)})")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        n = self.scalar("<Q")
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALAR:
            return self.scalar(_SCALAR[vtype])
        if vtype == GGUF_TYPE_STRING:
            return self.string()
        if vtype == GGUF_TYPE_ARRAY:
            etype = self.scalar("<I")
            count = self.scalar("<Q")
            if count > 10_000_000:
                raise GGUFError(f"implausible array length {count}")
            if etype in _SCALAR:
                size = struct.calcsize(_SCALAR[etype]) * count
                raw = self.take(size)
                return list(struct.unpack("<" + _SCALAR[etype][1] * count, raw)) if etype != GGUF_TYPE_BOOL \
                    else [bool(b) for b in raw]
            return [self.value(etype) for _ in range(count)]
        raise GGUFError(f"unknown gguf value type {vtype}")


def tensor_nbytes(type_id: int, n_elements: int) -> int:
    if type_id not in GGML_TYPES:
        raise GGUFError(f"unsupported ggml tensor type id {type_id}")
    _, blck, size = GGML_TYPES[type_id]
    if n_elements % blck != 0:
        raise GGUFError(
            f"tensor element count {n_elements} not divisible by block {blck} "
            f"for {GGML_TYPES[type_id][0]}")
    return n_elements // blck * size


def read_gguf(path: str | Path, meta_only: bool = True) -> GGUFModel:
    path = Path(path)
    file_size = path.stat().st_size
    # header + metadata + tensor table live at the start; read a generous chunk
    # and re-read the whole file only if metadata overflows it.
    read_len = min(file_size, 64 << 20)
    with path.open("rb") as f:
        buf = f.read(read_len)

    r = _Reader(buf)
    magic = r.scalar("<I")
    if magic != MAGIC:
        raise GGUFError(f"bad magic 0x{magic:08x}, expected GGUF")
    version = r.scalar("<I")
    if version not in (2, 3):
        raise GGUFError(f"unsupported GGUF version {version}")
    n_tensors = r.scalar("<Q")
    n_kv = r.scalar("<Q")

    kv: dict = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.scalar("<I")
        kv[key] = r.value(vtype)

    tensors: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.scalar("<I")
        dims = [r.scalar("<Q") for _ in range(n_dims)]
        type_id = r.scalar("<I")
        offset = r.scalar("<Q")
        n_elem = 1
        for d in dims:
            n_elem *= d
        if type_id not in GGML_TYPES:
            raise GGUFError(f"tensor {name}: unsupported type id {type_id}")
        tname = GGML_TYPES[type_id][0]
        tensors.append(TensorInfo(
            name=name, shape=list(reversed(dims)), type_id=type_id, offset=offset,
            n_elements=n_elem, nbytes=tensor_nbytes(type_id, n_elem), type_name=tname))

    alignment = int(kv.get("general.alignment", 32) or 32)
    data_offset = (r.pos + alignment - 1) // alignment * alignment

    return GGUFModel(path=path, version=version, kv=kv, tensors=tensors,
                     data_offset=data_offset, file_size=file_size, alignment=alignment)
