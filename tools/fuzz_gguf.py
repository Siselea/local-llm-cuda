"""Loader-robustness fuzzer for the engine's GGUF path.

Two sections:

  gguf : structural mutations of a valid tiny GGUF fixture (truncation, magic,
         version, header counts, metadata value types, tensor table fields,
         unsupported quant types, tensor data truncation) plus synthetic models
         for architecture / hyper-parameter boundaries.  Every generated file is
         then handed to `engine info` (parse + IR validation) and, for the cases
         that exercise the loader, to `test_gen` (actual weight loading).

  cli  : configuration / command line validation matrix (bad ints, out of range,
         missing config file, malformed config syntax, unknown keys/options ...).

Each subprocess is run with a hard timeout and classified:

  OK_ACCEPT   exit 0, no crash            (expected for valid models)
  OK_REJECT   non-zero exit, clean error  (expected for invalid models)
  CRASH       Windows exception code (0xC0000005 access violation, 0xC00000FD
              stack overflow, 0xC0000409 fast-fail ...)
  TIMEOUT     killed after --timeout
  SILENT      exit 0 although the input was invalid (classified by the case's
              expectation; not automatically a bug for data-section mutations
              because GGUF has no checksums)

Usage:
  python tools/fuzz_gguf.py                 # generate + run everything
  python tools/fuzz_gguf.py --section gguf
  python tools/fuzz_gguf.py --only trunc_50,offset_huge
  python tools/fuzz_gguf.py --list
  python tools/fuzz_gguf.py --no-run        # only generate fixtures
Exit code: 0 when every case meets its expectation and no crash/timeout, 1 otherwise.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "engine.exe"
GEN = ROOT / "build" / "test_gen.exe"
OUTDIR = ROOT / "build" / "fuzz"
BASE = ROOT / "build" / "fixtures" / "tiny_gguf_moe" / "model.gguf"
DEFAULT_TIMEOUT = 30

# Windows NTSTATUS values a crashed process can return from subprocess.
CRASH_CODES = {
    3221225477: "ACCESS_VIOLATION(0xC0000005)",
    3221225725: "STACK_OVERFLOW(0xC00000FD)",
    3221226505: "STACK_BUFFER_OVERRUN(0xC0000409)",
    3221225620: "INT_DIVIDE_BY_ZERO(0xC0000094)",
    3221225476: "INT_OVERFLOW(0xC0000095)",
    3221225595: "PRIV_INSTRUCTION(0xC0000096)",
    3221226356: "ILLEGAL_INSTRUCTION(0xC000001D)",
    3221225781: "DLL_NOT_FOUND(0xC0000135)",
    3221225775: "BAD_IMAGE(0xC000012F)",
}

# ---------------------------------------------------------------------------
# Minimal GGUF v3 writer (F32 tensors; lets cases control every header field)
# ---------------------------------------------------------------------------

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL = 0, 1, 2, 3, 4, 5, 6, 7
T_STRING, T_ARRAY, T_U64, T_I64, T_F64 = 8, 9, 10, 11, 12


def s8(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def bytes_array(elem_type: int, payload: bytes, length: int | None = None) -> bytes:
    n = length if length is not None else 0
    return struct.pack("<IQ", elem_type, n) + payload


def str_array(items: list[str]) -> bytes:
    return struct.pack("<IQ", T_STRING, len(items)) + b"".join(s8(i) for i in items)


def kv(key: str, vtype: int, payload: bytes) -> bytes:
    return s8(key) + struct.pack("<I", vtype) + payload


class GGUF:
    """Accumulates metadata + F32 tensors and serialises a GGUF v3 file."""

    def __init__(self):
        self.meta: list[bytes] = []
        self.tensors: list[dict] = []
        self.alignment = 32

    def add(self, key: str, vtype: int, payload: bytes):
        self.meta.append(kv(key, vtype, payload))
        if key == "general.alignment" and vtype == T_U32:
            self.alignment = max(1, struct.unpack("<I", payload)[0])
        return self

    def u32(self, key: str, v: int):
        return self.add(key, T_U32, struct.pack("<I", v & 0xFFFFFFFF))

    def i32(self, key: str, v: int):
        return self.add(key, T_I32, struct.pack("<i", v))

    def u64(self, key: str, v: int):
        return self.add(key, T_U64, struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF))

    def f32(self, key: str, v: float):
        return self.add(key, T_F32, struct.pack("<f", v))

    def string(self, key: str, v: str):
        return self.add(key, T_STRING, s8(v))

    def tensor(self, name: str, dims: list[int], data: bytes | None, type_id: int = 0,
               offset: int | None = None):
        """dims in *logical* order (shape as printed by the engine); the writer
        reverses them because GGUF stores ne[0] first."""
        self.tensors.append(
            {"name": name, "dims": dims, "data": data or b"", "type": type_id,
             "offset": offset})
        return self

    def build(self) -> bytes:
        # Tensor table first: compute offsets if not forced.
        infos = bytearray()
        data_blobs = bytearray()
        for t in self.tensors:
            offset = t["offset"]
            if offset is None:
                # GGUF tensor data must be aligned to general.alignment (32 by
                # default); llama.cpp rejects unaligned tensor offsets.
                offset = (len(data_blobs) + self.alignment - 1) // self.alignment * self.alignment
            t["offset"] = offset
            infos += s8(t["name"])
            infos += struct.pack("<I", len(t["dims"]))
            for d in reversed(t["dims"]):
                infos += struct.pack("<Q", d & 0xFFFFFFFFFFFFFFFF)
            infos += struct.pack("<I", t["type"])
            infos += struct.pack("<Q", offset & 0xFFFFFFFFFFFFFFFF)
            # place data at its offset (support gaps from forced offsets)
            if offset >= len(data_blobs):
                data_blobs += b"\x00" * (offset - len(data_blobs))
            data_blobs += t["data"]
        header = struct.pack("<IIQQ", 0x46554747, 3, len(self.tensors), len(self.meta))
        table = header + b"".join(self.meta) + bytes(infos)
        pad = (-len(table)) % self.alignment
        return table + b"\x00" * pad + bytes(data_blobs)


def f32_tensor(n: int, seed: int = 1) -> bytes:
    rng = random.Random(seed)
    return b"".join(struct.pack("<f", rng.uniform(-0.05, 0.05)) for _ in range(n))


# ---------------------------------------------------------------------------
# In-place locator for the base fixture (find metadata / tensor table fields)
# ---------------------------------------------------------------------------


class Layout:
    def __init__(self, data: bytes):
        self.data = data
        self.file_size = len(data)
        self.meta: dict[str, dict] = {}
        self.tensors: list[dict] = []
        pos = 0
        magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", data, pos)
        self.magic, self.version = magic, version
        self.n_tensors, self.n_kv = n_tensors, n_kv
        self.counts_pos = pos + 8
        pos += 24
        for _ in range(n_kv):
            klen = struct.unpack_from("<Q", data, pos)[0]
            key = data[pos + 8:pos + 8 + klen].decode("latin1")
            vtype_pos = pos + 8 + klen
            vtype = struct.unpack_from("<I", data, vtype_pos)[0]
            value_pos = vtype_pos + 4
            end = skip_value(data, value_pos, vtype)
            self.meta[key] = {"key_pos": pos, "klen": klen, "vtype_pos": vtype_pos,
                              "vtype": vtype, "value_pos": value_pos, "end": end}
            pos = end
        for i in range(n_tensors):
            name_pos = pos
            nlen = struct.unpack_from("<Q", data, pos)[0]
            name = data[pos + 8:pos + 8 + nlen].decode("latin1")
            p = pos + 8 + nlen
            ndims_pos = p
            ndims = struct.unpack_from("<I", data, p)[0]
            p += 4
            dims_pos = p
            p += 8 * ndims
            type_pos = p
            p += 4
            offset_pos = p
            p += 8
            self.tensors.append({"name": name, "name_pos": name_pos, "nlen": nlen,
                                 "ndims_pos": ndims_pos, "ndims": ndims,
                                 "dims_pos": dims_pos, "type_pos": type_pos,
                                 "offset_pos": offset_pos, "type": struct.unpack_from("<I", data, type_pos)[0],
                                 "offset": struct.unpack_from("<Q", data, offset_pos)[0]})
            pos = p
        align = 32
        if "general.alignment" in self.meta:
            m = self.meta["general.alignment"]
            align = max(1, struct.unpack_from("<I", data, m["value_pos"])[0])
        self.table_end = pos
        self.data_offset = (pos + align - 1) // align * align

    def patch_u32(self, off: int, v: int) -> bytes:
        b = bytearray(self.data)
        struct.pack_into("<I", b, off, v & 0xFFFFFFFF)
        return bytes(b)

    def patch_u64(self, off: int, v: int) -> bytes:
        b = bytearray(self.data)
        struct.pack_into("<Q", b, off, v & 0xFFFFFFFFFFFFFFFF)
        return bytes(b)


def skip_value(data: bytes, pos: int, vtype: int) -> int:
    if vtype in (T_U8, T_I8, T_BOOL):
        return pos + 1
    if vtype in (T_U16, T_I16):
        return pos + 2
    if vtype in (T_U32, T_I32, T_F32):
        return pos + 4
    if vtype in (T_U64, T_I64, T_F64):
        return pos + 8
    if vtype == T_STRING:
        n = struct.unpack_from("<Q", data, pos)[0]
        return pos + 8 + n
    if vtype == T_ARRAY:
        etype, n = struct.unpack_from("<IQ", data, pos)
        p = pos + 12
        for _ in range(n):
            p = skip_value(data, p, etype)
        return p
    raise ValueError(f"bad vtype {vtype} at {pos}")


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------


def build_dense_model(**over) -> bytes:
    """A tiny valid llama-like dense model; over-writable hyper parameters."""
    p = dict(hidden=32, n_head=4, key_dim=8, n_kv=2, ffn=64, vocab=16, n_layer=2,
             rope_base=10000.0, rope_dim=0, ctx=64, arch="llama",
             head_count_kv=None, n_expert=0, expert_used=0, expert_ffn=0,
             token_len=None, seq=None)
    p.update(over)
    hidden, n_head, head_dim = p["hidden"], p["n_head"], p["key_dim"]
    n_kv = p["head_count_kv"] if p["head_count_kv"] is not None else p["n_kv"]
    vocab, n_layer, ffn = p["vocab"], p["n_layer"], p["ffn"]
    g = GGUF()
    g.string("general.architecture", p["arch"])
    g.u32(f"{p['arch']}.block_count", n_layer)
    g.u32(f"{p['arch']}.embedding_length", hidden)
    g.u32(f"{p['arch']}.feed_forward_length", ffn)
    g.u32(f"{p['arch']}.context_length", p["ctx"])
    g.u32(f"{p['arch']}.attention.head_count", n_head)
    g.u32(f"{p['arch']}.attention.head_count_kv", n_kv)
    g.u32(f"{p['arch']}.attention.key_length", head_dim)
    g.u32(f"{p['arch']}.attention.value_length", head_dim)
    if p["rope_dim"]:
        g.u32(f"{p['arch']}.rope.dimension_count", p["rope_dim"])
    g.f32(f"{p['arch']}.rope.freq_base", p["rope_base"])
    g.f32(f"{p['arch']}.attention.layer_norm_rms_epsilon", 1e-5)
    g.string("tokenizer.ggml.model", "gpt2")
    g.add("tokenizer.ggml.tokens", T_ARRAY,
          str_array([f"t{i}" for i in range(p["token_len"] if p["token_len"] is not None else vocab)]))
    g.add("tokenizer.ggml.scores", T_ARRAY,
          struct.pack("<IQ", T_F32, vocab) + b"\x00" * (4 * vocab))
    g.add("tokenizer.ggml.token_type", T_ARRAY,
          struct.pack("<IQ", T_I32, vocab) + b"\x00" * (4 * vocab))
    g.i32("tokenizer.ggml.eos_token_id", 1)
    for i in range(n_layer):
        pre = f"blk.{i}."
        g.tensor(pre + "attn_norm.weight", [hidden], f32_tensor(hidden, i))
        g.tensor(pre + "attn_q.weight", [n_head * head_dim, hidden], f32_tensor(n_head * head_dim * hidden, i + 1))
        g.tensor(pre + "attn_k.weight", [n_kv * head_dim, hidden], f32_tensor(n_kv * head_dim * hidden, i + 2))
        g.tensor(pre + "attn_v.weight", [n_kv * head_dim, hidden], f32_tensor(n_kv * head_dim * hidden, i + 3))
        g.tensor(pre + "attn_output.weight", [hidden, n_head * head_dim], f32_tensor(hidden * n_head * head_dim, i + 4))
        g.tensor(pre + "ffn_norm.weight", [hidden], f32_tensor(hidden, i + 5))
        g.tensor(pre + "ffn_gate.weight", [ffn, hidden], f32_tensor(ffn * hidden, i + 6))
        g.tensor(pre + "ffn_up.weight", [ffn, hidden], f32_tensor(ffn * hidden, i + 7))
        g.tensor(pre + "ffn_down.weight", [hidden, ffn], f32_tensor(hidden * ffn, i + 8))
    g.tensor("token_embd.weight", [vocab, hidden], f32_tensor(vocab * hidden, 99))
    g.tensor("output_norm.weight", [hidden], f32_tensor(hidden, 100))
    if p.get("seq"):
        return p["seq"]
    return g.build()


class Case:
    def __init__(self, name: str, kind: str, expect: str, data: bytes | None = None,
                 note: str = "", stderr_has: str | None = None,
                 run_gen: bool = False, ids: str = "1", n: int = 1):
        self.name = name
        self.kind = kind
        self.expect = expect          # accept | reject
        self.data = data
        self.note = note
        self.stderr_has = stderr_has
        self.run_gen = run_gen
        self.ids = ids
        self.n = n


def gguf_cases() -> list[Case]:
    if not BASE.exists():
        raise SystemExit(f"base fixture missing: {BASE} (run tools/verify.sh once)")
    base = BASE.read_bytes()
    lay = Layout(base)
    cases: list[Case] = []
    A = cases.append

    def trunc(n: int, note=""):
        return Case(f"trunc_{note or n}", "gguf", "reject", base[:n],
                    note=f"file cut to {n} bytes")

    # ---- truncation / header ----
    A(Case("empty_file", "gguf", "reject", b""))
    A(trunc(4, "magic_only"))
    A(trunc(8, "magic_version"))
    A(trunc(24, "header_only"))
    A(trunc(len(base) // 100, "p1"))
    A(trunc(len(base) // 4, "p25"))
    A(trunc(len(base) // 2, "p50"))
    A(trunc(len(base) * 3 // 4, "p75"))
    A(trunc(len(base) * 9 // 10, "p90"))
    A(trunc(len(base) * 99 // 100, "p99"))
    A(trunc(len(base) - 1, "minus1"))
    A(Case("magic_zero", "gguf", "reject", lay.patch_u32(0, 0)))
    A(Case("magic_reversed", "gguf", "reject", b"FUGG" + base[4:]))
    A(Case("version_0", "gguf", "reject", lay.patch_u32(4, 0)))
    A(Case("version_1", "gguf", "reject", lay.patch_u32(4, 1)))
    A(Case("version_4", "gguf", "reject", lay.patch_u32(4, 4)))
    A(Case("version_max", "gguf", "reject", lay.patch_u32(4, 0xFFFFFFFF)))

    # ---- header counts ----
    hdr_n_tensors = lay.counts_pos
    hdr_n_kv = lay.counts_pos + 8
    A(Case("n_tensors_2p40", "gguf", "reject", lay.patch_u64(hdr_n_tensors, 1 << 40),
           note="reserve() on untrusted count"))
    A(Case("n_tensors_max", "gguf", "reject", lay.patch_u64(hdr_n_tensors, 0xFFFFFFFFFFFFFFFF)))
    A(Case("n_tensors_plus10", "gguf", "reject", lay.patch_u64(hdr_n_tensors, lay.n_tensors + 10)))
    A(Case("n_tensors_minus5", "gguf", "reject", lay.patch_u64(hdr_n_tensors, max(0, lay.n_tensors - 5)),
           note="table desync: extra tensor infos become 'data'"))
    A(Case("n_kv_2p40", "gguf", "reject", lay.patch_u64(hdr_n_kv, 1 << 40)))

    # ---- metadata value corruption ----
    first_meta = lay.meta[sorted(lay.meta)[0]]
    A(Case("meta_keylen_max", "gguf", "reject", lay.patch_u64(first_meta["key_pos"], 0xFFFFFFFFFFFFFFFF)))
    A(Case("meta_vtype_99", "gguf", "reject", lay.patch_u32(first_meta["vtype_pos"], 99)))
    if "general.alignment" in lay.meta:
        m = lay.meta["general.alignment"]
        A(Case("alignment_max", "gguf", "reject", lay.patch_u32(m["value_pos"], 0xFFFFFFF0),
               note="huge general.alignment"))
    if "llama.block_count" in lay.meta:
        m = lay.meta["llama.block_count"]
        A(Case("block_count_0", "gguf", "reject", lay.patch_u32(m["value_pos"], 0)))
        A(Case("block_count_1e6", "gguf", "reject", lay.patch_u32(m["value_pos"], 1_000_000),
               note="untrusted layer count -> error string / vector growth"))
        A(Case("block_count_1e8", "gguf", "reject", lay.patch_u32(m["value_pos"], 100_000_000),
               note="untrusted layer count 1e8"))

    # ---- tensor table corruption ----
    t0 = lay.tensors[0]
    t2d = next((t for t in lay.tensors if t["ndims"] == 2), t0)
    A(Case("tensor_ndims_9", "gguf", "reject", lay.patch_u32(t0["ndims_pos"], 9)))
    A(Case("tensor_dims_overflow", "gguf", "reject",
           _patch_dims(lay, lay.tensors.index(t2d), [1 << 32, 1 << 32]), note="n_elements multiplication overflow"))
    A(Case("tensor_dim_max", "gguf", "reject", _patch_dims(lay, lay.tensors.index(t2d), [0xFFFFFFFFFFFFFFFF, 2])))
    A(Case("tensor_offset_max", "gguf", "reject",
           lay.patch_u64(t0["offset_pos"], 0xFFFFFFFFFFFFFFF0),
           note="offset + nbytes u64 overflow in span check"))
    A(Case("tensor_offset_2p63", "gguf", "reject", lay.patch_u64(t0["offset_pos"], 1 << 63)))
    A(Case("tensor_type_999", "gguf", "reject", lay.patch_u32(t0["type_pos"], 999)))
    A(Case("tensor_type_Q3_K", "gguf", "reject", lay.patch_u32(t0["type_pos"], 11),
           stderr_has="unsupported quant"))
    A(Case("tensor_type_IQ2_XXS", "gguf", "reject", lay.patch_u32(t0["type_pos"], 16),
           stderr_has="unsupported quant"))
    A(Case("data_cut_90pct", "gguf", "reject", base[:lay.data_offset + (len(base) - lay.data_offset) // 10],
           note="header+table intact, tensor data truncated", run_gen=True))

    # ---- loader-stage (test_gen) on structurally invalid files ----
    A(Case("load_data_trunc", "gguf", "reject",
           base[:lay.data_offset + max(1, (len(base) - lay.data_offset) // 100)],
           run_gen=True))
    A(Case("load_offset_huge", "gguf", "reject",
           lay.patch_u64(t0["offset_pos"], 0xFFFFFFFFFFFFFFF0), run_gen=True))
    A(Case("load_quant_Q3_K", "gguf", "reject", lay.patch_u32(t0["type_pos"], 11),
           stderr_has="unsupported quant", run_gen=True))
    A(Case("load_type_999", "gguf", "reject", lay.patch_u32(t0["type_pos"], 999), run_gen=True))
    A(Case("n_tensors_0", "gguf", "reject", lay.patch_u64(hdr_n_tensors, 0),
           note="declared 0 tensors: layout info discarded"))
    A(Case("n_kv_0", "gguf", "reject", lay.patch_u64(hdr_n_kv, 0),
           note="metadata skipped: tensor table starts in metadata"))
    A(Case("tensor_name_max", "gguf", "reject",
           lay.patch_u64(t0["name_pos"], 0xFFFFFFFFFFFFFFFF)))
    A(Case("tensor_offsets_overlap", "gguf", "reject",
           lay.patch_u64(lay.tensors[1]["offset_pos"], lay.tensors[0]["offset"]),
           note="two tensors at the same data offset"))
    A(Case("ffn_0_load", "gguf", "reject", build_dense_model(ffn=0), run_gen=True,
           note="feed_forward_length=0 with 0-element FFN tensors"))

    # ---- synthetic hyper-parameter boundaries ----
    A(Case("syn_valid_dense", "gguf", "accept", build_dense_model(), run_gen=True,
           note="control: tiny dense F32 model must load and generate"))
    A(Case("head_dim_64", "gguf", "accept", build_dense_model(key_dim=64, hidden=256), run_gen=True,
           note="head_dim != 128, supported: 256/4"))
    A(Case("head_dim_96", "gguf", "accept", build_dense_model(key_dim=96, hidden=384), run_gen=True))
    A(Case("head_dim_128", "gguf", "accept", build_dense_model(key_dim=128, hidden=512), run_gen=True))
    A(Case("head_dim_256", "gguf", "accept", build_dense_model(key_dim=256, hidden=1024), run_gen=True))
    A(Case("head_dim_odd_65", "gguf", "reject", build_dense_model(key_dim=65, hidden=260),
           stderr_has="odd"))
    A(Case("n_head_kv_3_of_4", "gguf", "reject", build_dense_model(head_count_kv=3),
           stderr_has="not divisible"))
    A(Case("n_head_kv_0", "gguf", "reject", build_dense_model(head_count_kv=0)))
    A(Case("rope_base_0", "gguf", "reject", build_dense_model(rope_base=0.0), run_gen=True,
           note="freq_base=0 -> pow(0,-x) inf; must be rejected, not silently accepted"))
    A(Case("rope_base_negative", "gguf", "reject", build_dense_model(rope_base=-100.0), run_gen=True))
    A(Case("rope_dim_odd", "gguf", "reject", build_dense_model(rope_dim=5)))
    A(Case("rope_dim_gt_head", "gguf", "reject", build_dense_model(rope_dim=64)))
    A(Case("vocab_tokens_mismatch", "gguf", "reject", build_dense_model(token_len=32),
           note="tokenizer tokens 32 vs embedding rows/vocab_size 16"))
    A(Case("n_expert_300", "gguf", "reject", build_dense_model(n_expert=300, expert_used=4,
                                                                 expert_ffn=64),
           stderr_has="expert_count"))
    A(Case("n_expert_0_dense", "gguf", "accept", build_dense_model(n_expert=0)))
    A(Case("block_count_0_syn", "gguf", "reject", build_dense_model(n_layer=0)))
    A(Case("hidden_huge", "gguf", "reject",
           meta_only_model(block_count=2, embedding_length=0xFFFFFFFFFFFFFFFF),
           note="embedding_length 2^64-1 -> int truncation; must be rejected"))
    A(Case("ffn_0", "gguf", "reject", build_dense_model(ffn=0)))

    # ---- metadata value *type* mismatch (semantically valid GGUF encoding) ----
    g = GGUF()
    g.string("general.architecture", "llama")
    g.string("llama.block_count", "2")          # must be UINT32
    g.u32("llama.embedding_length", 64)
    A(Case("meta_block_count_string", "gguf", "reject", g.build(),
           note="n_layer declared as STRING instead of UINT32"))

    # ---- yarn scaling with missing / zero parameters ----
    def yarn_model(**over):
        g = GGUF()
        p = dict(hidden=64, n_head=4, key_dim=16, n_kv=2, ffn=128, vocab=16, n_layer=1)
        p.update(over)
        g.string("general.architecture", "llama")
        g.u32("llama.block_count", p["n_layer"])
        g.u32("llama.embedding_length", p["hidden"])
        g.u32("llama.feed_forward_length", p["ffn"])
        g.u32("llama.attention.head_count", p["n_head"])
        g.u32("llama.attention.head_count_kv", p["n_kv"])
        g.u32("llama.attention.key_length", p["key_dim"])
        g.u32("llama.attention.value_length", p["key_dim"])
        g.f32("llama.rope.freq_base", 10000.0)
        g.f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        g.string("llama.rope.scaling.type", "yarn")
        for k, v in over.get("scaling", {}).items():
            g.f32(f"llama.rope.scaling.{k}", v)
        g.string("tokenizer.ggml.model", "gpt2")
        g.add("tokenizer.ggml.tokens", T_ARRAY, str_array([f"t{i}" for i in range(p["vocab"])]))
        hidden, ffn, n_head, kd, n_kv, vocab = p["hidden"], p["ffn"], p["n_head"], p["key_dim"], p["n_kv"], p["vocab"]
        for i in range(p["n_layer"]):
            pre = f"blk.{i}."
            g.tensor(pre + "attn_norm.weight", [hidden], f32_tensor(hidden, i))
            g.tensor(pre + "attn_q.weight", [n_head * kd, hidden], f32_tensor(n_head * kd * hidden))
            g.tensor(pre + "attn_k.weight", [n_kv * kd, hidden], f32_tensor(n_kv * kd * hidden))
            g.tensor(pre + "attn_v.weight", [n_kv * kd, hidden], f32_tensor(n_kv * kd * hidden))
            g.tensor(pre + "attn_output.weight", [hidden, n_head * kd], f32_tensor(hidden * n_head * kd))
            g.tensor(pre + "ffn_norm.weight", [hidden], f32_tensor(hidden))
            g.tensor(pre + "ffn_gate.weight", [ffn, hidden], f32_tensor(ffn * hidden))
            g.tensor(pre + "ffn_up.weight", [ffn, hidden], f32_tensor(ffn * hidden))
            g.tensor(pre + "ffn_down.weight", [hidden, ffn], f32_tensor(hidden * ffn))
        g.tensor("token_embd.weight", [vocab, hidden], f32_tensor(vocab * hidden))
        g.tensor("output_norm.weight", [hidden], f32_tensor(hidden))
        return g.build()

    A(Case("yarn_missing_params", "gguf", "accept", yarn_model(), run_gen=True,
           note="yarn scaling with defaults; must not crash"))
    A(Case("yarn_factor_0", "gguf", "reject", yarn_model(scaling={"factor": 0.0}), run_gen=True,
           note="factor=0 -> 1/(factor*pos_freq) inf; must not be silently accepted"))
    A(Case("yarn_orig_0", "gguf", "reject", yarn_model(scaling={"factor": 4.0, "original_context_length": 0.0}), run_gen=True,
           note="original_context_length=0 -> log(0) -inf"))

    # ---- deeply nested metadata array (stack overflow probe) ----
    depth = 40000
    nested = b"\x00" * 0
    payload = b""
    for _ in range(depth):
        payload = struct.pack("<IQ", T_ARRAY, 1) + payload
    g = GGUF()
    g.add("general.fuzz_depth", T_ARRAY, struct.pack("<IQ", T_ARRAY, 1) + payload)
    A(Case("meta_array_depth_40k", "gguf", "reject", _prefix_with_meta(g),
           note="nested ARRAY metadata depth 40k: recursion depth DoS"))

    return cases


def meta_only_model(**kv_u32) -> bytes:
    g = GGUF()
    g.string("general.architecture", "llama")
    for k, v in kv_u32.items():
        g.add(f"llama.{k.replace('_', '.')}", T_U32 if v <= 0xFFFFFFFF else T_U64,
              struct.pack("<I", v & 0xFFFFFFFF) if v <= 0xFFFFFFFF else struct.pack("<Q", v))
    return g.build()


def _patch_dims(lay: Layout, idx: int, dims: list[int]) -> bytes:
    b = bytearray(lay.data)
    p = lay.tensors[idx]["dims_pos"]
    for d in dims:
        struct.pack_into("<Q", b, p, d & 0xFFFFFFFFFFFFFFFF)
        p += 8
    return bytes(b)


def _prefix_with_meta(meta_only: GGUF) -> bytes:
    """Valid header with no tensors and one arbitrary metadata key (for parser probes)."""
    header = struct.pack("<IIQQ", 0x46554747, 3, 0, len(meta_only.meta))
    table = header + b"".join(meta_only.meta)
    return table


# ---------------------------------------------------------------------------
# CLI / config validation cases (no GPU touched: engine info is parse-only)
# ---------------------------------------------------------------------------

def cli_cases(outdir: Path, base: Path) -> list[tuple]:
    """(name, argv, expect, stderr_has) — argv already resolved."""
    f = lambda n: str(outdir / n)
    eng = str(ENGINE)
    return [
        ("cli_ctx_zero", [eng, "info", "--ctx", "0", str(base)], "accept", None),
        ("cli_ctx_negative", [eng, "info", "--ctx", "-1", str(base)], "reject", "out of range"),
        ("cli_ctx_huge", [eng, "info", "--ctx", "99999999", str(base)], "reject", "out of range"),
        ("cli_ctx_not_int", [eng, "info", "--ctx", "abc", str(base)], "reject", "expects an integer"),
        ("cli_threads_negative", [eng, "info", "--threads", "-2", str(base)], "reject", "out of range"),
        ("cli_threads_huge", [eng, "info", "--threads", "99999", str(base)], "reject", "out of range"),
        ("cli_gpu_layers_huge", [eng, "info", "--gpu-layers", "99999", str(base)], "reject", "out of range"),
        ("cli_gpu_layers_negative", [eng, "info", "--gpu-layers", "-2", str(base)], "reject", "out of range"),
        ("cli_vram_margin_negative", [eng, "info", "--vram-margin", "-5", str(base)], "reject", "out of range"),
        ("cli_unknown_option", [eng, "info", str(base), "--definitely-not-an-option"], "reject", "unknown option"),
        ("cli_missing_value", [eng, "info", str(base), "--ctx"], "reject", "missing value"),
        ("cli_bad_config_missing", [eng, "info", "--config", f("no-such-config.cfg"), str(base)], "reject", "cannot open config"),
        ("cli_config_no_equals", [eng, "info", "--config", f("bad_syntax.cfg"), str(base)], "reject", "expected 'key = value'"),
        ("cli_config_unknown_key", [eng, "info", "--config", f("bad_key.cfg"), str(base)], "reject", "unknown config key"),
        ("cli_config_bad_int", [eng, "info", "--config", f("bad_int.cfg"), str(base)], "reject", "expects an integer"),
        ("cli_config_range", [eng, "info", "--config", f("bad_range.cfg"), str(base)], "reject", "out of range"),
        ("cli_config_unknown_section", [eng, "info", "--config", f("bad_section.cfg"), str(base)], "reject", "unknown section"),
        ("cli_config_dup_key", [eng, "info", "--config", f("dup.cfg"), str(base)], "accept", None),
        ("cli_config_cli_overrides", [eng, "info", "--config", f("good.cfg"), "--ctx", "64", str(base)], "accept", None),
        ("cli_engine_no_args", [eng], "reject", "usage"),
        ("cli_engine_unknown_cmd", [eng, "frobnicate"], "reject", "usage"),
        ("cli_info_no_model", [eng, "info"], "reject", "usage"),
    ]



# ---------------------------------------------------------------------------
# Runner / classifier
# ---------------------------------------------------------------------------


def classify(rc: int, stdout: str, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    if rc < 0:
        return f"CRASH({rc})"
    if rc in CRASH_CODES:
        return f"CRASH({CRASH_CODES[rc]})"
    if rc == 0:
        return "OK_ACCEPT"
    return "OK_REJECT"


def run(cmd: list[str], timeout: int) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        rc, out, err, timed = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, out, err, timed = -999, (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""), True
    dt = time.time() - t0
    return {"rc": rc, "out": out, "err": err, "timeout": timed, "sec": dt,
            "class": classify(rc, out, err, timed)}


def evaluate(case: Case, res: dict) -> dict:
    """Return {'verdict': 'OK'|'BUG'|'SILENT'|'CRASH'|'TIMEOUT', 'why': str}."""
    if res["class"].startswith("CRASH"):
        return {"verdict": "CRASH", "why": res["class"]}
    if res["class"] == "TIMEOUT":
        return {"verdict": "TIMEOUT", "why": f">{DEFAULT_TIMEOUT}s"}
    if case.expect == "reject":
        if res["class"] == "OK_REJECT":
            why = "clean reject"
            if case.stderr_has and case.stderr_has not in (res["err"] + res["out"]):
                return {"verdict": "BUG", "why": f"rejected but message lacks '{case.stderr_has}'"}
            return {"verdict": "OK", "why": why}
        return {"verdict": "SILENT", "why": f"exit 0 on invalid input ({case.note})"}
    if res["class"] == "OK_ACCEPT":
        return {"verdict": "OK", "why": "accepted"}
    if case.stderr_has and case.stderr_has in (res["err"] + res["out"]):
        return {"verdict": "OK", "why": "rejected with expected message"}
    # expected accept but rejected: sometimes a legitimate stricter validation is fine
    if "error:" in res["err"] or "error:" in res["out"]:
        return {"verdict": "OK", "why": "rejected with error (strict)"}
    return {"verdict": "BUG", "why": "expected accept, got unexpected failure"}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["gguf", "cli", "all"], default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases: list[Case] = []
    if args.section in ("gguf", "all"):
        cases += gguf_cases()
    if args.list:
        for c in cases:
            print(f"{c.name:28s} kind={c.kind:5s} expect={c.expect:6s} {c.note}")
        return 0

    # config syntax fixtures for the CLI section
    (outdir / "bad_syntax.cfg").write_text("ctx = 5\nthis line has no equals\n")
    (outdir / "bad_key.cfg").write_text("ctx = 5\nfrobnicate = 1\n")
    (outdir / "bad_int.cfg").write_text("ctx = abc\n")
    (outdir / "bad_range.cfg").write_text("ctx = 99999999\n")
    (outdir / "bad_section.cfg").write_text("[nonsense]\nctx = 5\n")
    (outdir / "dup.cfg").write_text("ctx = 32\nctx = 64\n")
    (outdir / "good.cfg").write_text("ctx = 32\nthreads = 2\n")

    results = []
    if cases:
        for c in cases:
            if only and c.name not in only:
                continue
            path = outdir / f"{c.name}.gguf"
            path.write_bytes(c.data if c.data is not None else b"")
            cmds = []
            base_cmd = [str(ENGINE), "info", str(path)]
            cmds.append(("info", base_cmd))
            if c.run_gen:
                cmds.append(("gen", [str(GEN), str(path), c.ids, str(c.n)]))
            for mode, cmd in cmds:
                res = run(cmd, args.timeout)
                ev = evaluate(c, res)
                results.append({"case": c.name, "mode": mode, "file": path.name,
                                "expect": c.expect, "note": c.note,
                                "result": res, "verdict": ev["verdict"], "why": ev["why"]})
                mark = {"OK": "OK  ", "BUG": "BUG ", "SILENT": "SILENT",
                        "CRASH": "CRASH", "TIMEOUT": "TIMEOUT"}[ev["verdict"]]
                print(f"{mark} {c.name:26s} [{mode:4s}] rc={res['rc']:>12} "
                      f"{res['sec']:5.2f}s {ev['why']}")

    if args.section in ("cli", "all") and not only:
        for name, argv, expect, sub in cli_cases(outdir, BASE):
            res = run(argv, args.timeout)
            c = Case(name, "cli", expect, stderr_has=sub)
            ev = evaluate(c, res)
            results.append({"case": name, "mode": "cli", "file": "", "expect": expect,
                            "note": " ".join(argv[1:]), "result": res,
                            "verdict": ev["verdict"], "why": ev["why"]})
            mark = {"OK": "OK  ", "BUG": "BUG ", "SILENT": "SILENT",
                    "CRASH": "CRASH", "TIMEOUT": "TIMEOUT"}[ev["verdict"]]
            print(f"{mark} {name:26s} [cli ] rc={res['rc']:>12} {res['sec']:5.2f}s {ev['why']}")

    bad = [r for r in results if r["verdict"] in ("BUG", "SILENT", "CRASH", "TIMEOUT")]
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": str(ENGINE), "gen": str(GEN), "base": str(BASE),
        "total": len(results), "ok": len(results) - len(bad),
        "findings": len(bad),
        "results": results,
    }
    (outdir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n== {len(results) - len(bad)}/{len(results)} cases as expected, "
          f"{len(bad)} finding(s) ==")
    print(f"wrote {outdir / 'results.json'}")
    if args.no_run:
        return 0
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
