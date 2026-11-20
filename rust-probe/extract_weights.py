#!/usr/bin/env python3
"""Extract one CPU-layer worth of *real* Q4_K/Q6_K expert weights from the 35B GGUF.

The Rust/C++ language comparison must run on identical bytes, and synthetic scales
are not quite identical (denormal/NaN scale values change timing).  This dumps the
raw quantized bytes of experts 0..7 of one MoE layer:

  gate8.bin : 8 experts x 512 rows x 8 Q4_K blocks x 144 B = 4,718,592 B
  up8.bin   : same
  down8.bin : 8 experts x 2048 rows x 2 Q6_K blocks x 210 B = 6,881,280 B
  total     : 16,318,464 B  (matches the engine's 16.3 MB / CPU layer)

usage: extract_weights.py <model.gguf> <layer> <outdir>
"""
import os
import struct
import sys

Q4_K, Q6_K = 12, 14
TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
             10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"}
# bytes per 256-value block (Q4_K/Q6_K) or per 32-value block (Q8_0/Q4_0/...)
BLOCK_ELEMS = {12: 256, 13: 256, 14: 256, 8: 32, 2: 32, 3: 32, 6: 32, 7: 32}
BLOCK_BYTES = {12: 144, 13: 176, 14: 210, 8: 34, 2: 18, 3: 20, 6: 22, 7: 24}


class R:
    def __init__(self, f):
        self.f = f

    def bytes(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.bytes(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.bytes(8))[0]

    def s(self):
        n = self.u64()
        if n > (1 << 24):
            raise ValueError("string too long")
        return self.bytes(n).decode("utf-8", "replace")

    def kv(self, t):
        if t == 0: return struct.unpack("<B", self.bytes(1))[0]
        if t == 1: return struct.unpack("<b", self.bytes(1))[0]
        if t == 2: return struct.unpack("<H", self.bytes(2))[0]
        if t == 3: return struct.unpack("<h", self.bytes(2))[0]
        if t == 4: return self.u32()
        if t == 5: return struct.unpack("<i", self.bytes(4))[0]
        if t == 6: return struct.unpack("<f", self.bytes(4))[0]
        if t == 7: return struct.unpack("<?", self.bytes(1))[0]
        if t == 8: return self.s()
        if t == 9:  # array
            at = self.u32()
            n = self.u64()
            return [self.kv(at) for _ in range(n)]
        if t == 10: return self.u64()
        if t == 11: return struct.unpack("<q", self.bytes(8))[0]
        if t == 12: return struct.unpack("<d", self.bytes(8))[0]
        raise ValueError("bad kv type %d" % t)


def main():
    model, layer, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    with open(model, "rb") as f:
        r = R(f)
        if r.bytes(4) != b"GGUF":
            raise SystemExit("not a gguf")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kvs = {}
        for _ in range(n_kv):
            k = r.s()
            t = r.u32()
            kvs[k] = r.kv(t)
        tensors = {}
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            off = r.u64()
            tensors[name] = (dims, ttype, off)
        align = int(kvs.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
        print(f"gguf v{version}: {n_tensors} tensors, {n_kv} kv, align={align}, "
              f"data_start={data_start}")

        def dump(tensor, n_experts, out_name):
            name = f"blk.{layer}.{tensor}"
            if name not in tensors:
                raise SystemExit(f"missing {name}")
            dims, ttype, off = tensors[name]
            if ttype not in BLOCK_ELEMS:
                raise SystemExit(f"{name}: unsupported type {TYPE_NAME.get(ttype, ttype)}")
            be, bb = BLOCK_ELEMS[ttype], BLOCK_BYTES[ttype]
            row_vals = dims[0]
            n_rows = 1
            for d in dims[1:]:
                n_rows *= d
            n_expert = dims[2] if len(dims) > 2 else 1
            rows_per_expert = n_rows // n_expert
            row_bytes = row_vals // be * bb
            slab = rows_per_expert * row_bytes
            print(f"  {name}: dims={dims} {TYPE_NAME[ttype]} row={row_vals} vals "
                  f"({row_bytes} B), rows/expert={rows_per_expert}, "
                  f"expert slab={slab} B")
            f.seek(data_start + off)
            n = n_experts * slab
            buf = f.read(n)
            if len(buf) != n:
                raise SystemExit("short tensor read")
            p = os.path.join(outdir, out_name)
            with open(p, "wb") as g:
                g.write(buf)
            print(f"  -> {p} {n} B")
            return n

        tot = 0
        tot += dump("ffn_gate_exps.weight", 8, "gate8.bin")
        tot += dump("ffn_up_exps.weight", 8, "up8.bin")
        tot += dump("ffn_down_exps.weight", 8, "down8.bin")
        print(f"total {tot} bytes")


if __name__ == "__main__":
    main()
