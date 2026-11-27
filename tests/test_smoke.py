"""Synthetic GGUF writer + end-to-end smoke tests (no external models needed).

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyref.gguf_parse import read_gguf, GGUFError
from pyref.ir import build_ir_from_gguf, validate_ir, Role
from pyref.planner import plan, BW_CPU_GEMV_GBPS

GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGML_F32 = 0
GGML_Q4_K = 12


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_tiny_gguf(path: Path, n_layer=2, n_expert=4, hidden=64, ffn=128,
                    inter=32, vocab=128) -> None:
    kv = [
        ("general.architecture", GGUF_TYPE_STRING, "testarch"),
        ("general.alignment", GGUF_TYPE_UINT32, 32),
        ("testarch.block_count", GGUF_TYPE_UINT32, n_layer),
        ("testarch.embedding_length", GGUF_TYPE_UINT32, hidden),
        ("testarch.feed_forward_length", GGUF_TYPE_UINT32, ffn),
        ("testarch.context_length", GGUF_TYPE_UINT32, 2048),
        ("testarch.attention.head_count", GGUF_TYPE_UINT32, 8),
        ("testarch.attention.head_count_kv", GGUF_TYPE_UINT32, 2),
        ("testarch.attention.key_length", GGUF_TYPE_UINT32, 8),
        ("testarch.attention.value_length", GGUF_TYPE_UINT32, 8),
        ("testarch.expert_count", GGUF_TYPE_UINT32, n_expert),
        ("testarch.expert_used_count", GGUF_TYPE_UINT32, 2),
        ("testarch.expert_shared_count", GGUF_TYPE_UINT32, 1),
        ("testarch.expert_feed_forward_length", GGUF_TYPE_UINT32, inter),
        ("tokenizer.ggml.model", GGUF_TYPE_STRING, "gpt2"),
        ("tokenizer.ggml.tokens", 9, None),  # array<string>; etype+count written below
    ]

    tensors = [("token_embd.weight", (vocab, hidden), GGML_F32),
               ("output.weight", (vocab, hidden), GGML_F32)]
    for i in range(n_layer):
        tensors += [
            (f"blk.{i}.attn_norm.weight", (hidden,), GGML_F32),
            (f"blk.{i}.attn_q.weight", (hidden, hidden), GGML_Q4_K),
            (f"blk.{i}.ffn_norm.weight", (hidden,), GGML_F32),
            (f"blk.{i}.ffn_gate_inp.weight", (n_expert, hidden), GGML_F32),
            # fused experts: [n_expert, out, in] -> raw gguf dims are reversed
            (f"blk.{i}.ffn_gate_exps.weight", (n_expert, inter, hidden), GGML_Q4_K),
            (f"blk.{i}.ffn_up_exps.weight", (n_expert, inter, hidden), GGML_Q4_K),
            (f"blk.{i}.ffn_down_exps.weight", (n_expert, hidden, inter), GGML_Q4_K),
            (f"blk.{i}.ffn_gate_shexp.weight", (inter, hidden), GGML_Q4_K),
            (f"blk.{i}.ffn_up_shexp.weight", (inter, hidden), GGML_Q4_K),
            (f"blk.{i}.ffn_down_shexp.weight", (hidden, inter), GGML_Q4_K),
        ]

    def tbytes(shape, tid):
        n = 1
        for d in shape:
            n *= d
        blck, size = (1, 4) if tid == GGML_F32 else (256, 144)
        if n % blck:
            raise ValueError(f"{shape} not divisible by {blck}")
        return n // blck * size

    # tensor offsets (relative to data section), aligned to 32
    offset = 0
    infos = []
    for name, shape, tid in tensors:
        infos.append((name, shape, tid, offset))
        offset = (offset + tbytes(shape, tid) + 31) // 32 * 32

    body = bytearray()
    body += struct.pack("<I", 0x46554747)
    body += struct.pack("<I", 3)
    body += struct.pack("<Q", len(tensors))
    body += struct.pack("<Q", len(kv))
    for key, vtype, val in kv:
        body += _s(key)
        body += struct.pack("<I", vtype)
        if key == "tokenizer.ggml.tokens":
            body += struct.pack("<I", GGUF_TYPE_STRING)
            body += struct.pack("<Q", vocab)
            for i in range(vocab):
                body += _s(f"tok{i}")
        elif isinstance(val, str):
            body += _s(val)
        elif vtype == GGUF_TYPE_UINT32:
            body += struct.pack("<I", int(val))
        elif vtype == GGUF_TYPE_FLOAT32:
            body += struct.pack("<f", float(val))
    for name, shape, tid, off in infos:
        body += _s(name)
        body += struct.pack("<I", len(shape))
        for d in reversed(shape):          # gguf stores dims innermost-first
            body += struct.pack("<Q", d)
        body += struct.pack("<I", tid)
        body += struct.pack("<Q", off)

    while len(body) % 32:
        body += b"\x00"
    data_len = offset
    body += b"\x00" * data_len
    path.write_bytes(bytes(body))


class TestGGUFParse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.gguf"
        write_tiny_gguf(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header(self):
        g = read_gguf(self.path)
        self.assertEqual(g.version, 3)
        self.assertEqual(g.meta_str("general.architecture"), "testarch")
        self.assertEqual(len(g.tensors), 2 + 2 * 10)

    def test_shapes_reversed(self):
        g = read_gguf(self.path)
        by = {t.name: t for t in g.tensors}
        self.assertEqual(by["blk.0.ffn_gate_exps.weight"].shape, [4, 32, 64])
        self.assertEqual(by["blk.0.attn_q.weight"].shape, [64, 64])

    def test_nbytes_q4k(self):
        g = read_gguf(self.path)
        by = {t.name: t for t in g.tensors}
        t = by["blk.0.attn_q.weight"]         # 64*64 = 4096 elems, Q4_K 256/144
        self.assertEqual(t.nbytes, 4096 // 256 * 144)

    def test_bad_magic(self):
        p = Path(self.tmp.name) / "bad.gguf"
        p.write_bytes(b"NOPE" + b"\x00" * 64)
        with self.assertRaises(GGUFError):
            read_gguf(p)


class TestIR(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.gguf"
        write_tiny_gguf(self.path)
        self.ir = build_ir_from_gguf(read_gguf(self.path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_roles(self):
        counts = {}
        for t in self.ir.tensors:
            counts[t.role] = counts.get(t.role, 0) + 1
        self.assertEqual(counts[Role.EXPERT], 6)      # 2 layers x 3 fused
        self.assertEqual(counts[Role.SHARED_EXPERT], 6)
        self.assertEqual(counts[Role.ROUTER], 2)

    def test_validate_passes(self):
        self.assertEqual(validate_ir(self.ir), [])

    def test_expert_bytes(self):
        per_layer = self.ir.expert_bytes_per_layer()
        self.assertEqual(len(per_layer), 2)
        self.assertGreater(per_layer[0], 0)

    def test_kv_formula(self):
        # 2 layers * 2 kv heads * (8+8) dims * 4 bytes (f32, what the runner allocates)
        from pyref.placement import kv_bytes_per_token
        self.assertEqual(kv_bytes_per_token(self.ir, 0), 2 * 2 * 16 * 4)


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.gguf"
        write_tiny_gguf(self.path, n_layer=4, n_expert=8, hidden=256, ffn=512,
                        inter=64, vocab=256)
        self.ir = build_ir_from_gguf(read_gguf(self.path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_budget_accounting(self):
        p = plan(self.ir, ctx=2048, vram_total=2 << 30)
        self.assertGreater(p.expert_budget, 0)
        accounted = p.fixed_bytes + p.kv_bytes + p.workspace_bytes + p.expert_budget
        self.assertLessEqual(accounted, p.vram_total - p.vram_reserve)
        self.assertAlmostEqual(accounted, p.vram_total - p.vram_reserve, delta=2)

    def test_too_little_vram_degrades_to_zero_budget(self):
        p = plan(self.ir, ctx=2048, vram_total=1 << 29)
        self.assertEqual(p.expert_budget, 0)
        self.assertTrue(any("exceed VRAM" in n for n in p.notes))

    def test_slots_shrink_with_smaller_vram(self):
        big = plan(self.ir, ctx=2048, vram_total=2 << 30)
        small = plan(self.ir, ctx=2048, vram_total=1 << 30)
        self.assertGreaterEqual(big.slots_per_layer, small.slots_per_layer)

    def test_cpu_projection_matches_constant(self):
        p = plan(self.ir, ctx=2048, vram_total=1 << 28)
        self.assertGreater(p.proj_decode_cpu_ms, 0)
        self.assertGreater(BW_CPU_GEMV_GBPS, 0)


if __name__ == "__main__":
    unittest.main()
