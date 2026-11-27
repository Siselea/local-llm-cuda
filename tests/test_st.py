"""Synthetic HF safetensors fixtures + safetensors IR tests.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyref.ir import validate_ir, Role
from pyref.st_parse import build_ir_from_hf, read_safetensors

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "F8_E4M3": 1}


def write_safetensors(path: Path, tensors: dict) -> None:
    """tensors: name -> (dtype, shape)"""
    hdr, blobs, off = {}, [], 0
    for name, (dt, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        nb = n * DTYPE_BYTES[dt]
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        blobs.append(b"\x00" * nb)
        off += nb
    js = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(js)) + js + b"".join(blobs))


def make_hf_model(root: Path, n_layer=2, hidden=64, inter=128, vocab=128,
                  n_head=4, n_head_kv=2, n_expert=0, expert_inter=32) -> Path:
    head_dim = hidden // n_head
    cfg = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": n_layer,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head_kv,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
        "eos_token_id": 1,
    }
    if n_expert:
        cfg.update({"num_experts": n_expert, "num_experts_per_tok": 2,
                    "moe_intermediate_size": expert_inter})

    tensors = {
        "model.embed_tokens.weight": ("BF16", (vocab, hidden)),
        "lm_head.weight": ("BF16", (vocab, hidden)),
        "model.norm.weight": ("BF16", (hidden,)),
    }
    for i in range(n_layer):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = ("BF16", (hidden,))
        tensors[p + "self_attn.q_proj.weight"] = ("BF16", (n_head * head_dim, hidden))
        tensors[p + "self_attn.k_proj.weight"] = ("BF16", (n_head_kv * head_dim, hidden))
        tensors[p + "self_attn.v_proj.weight"] = ("BF16", (n_head_kv * head_dim, hidden))
        tensors[p + "self_attn.o_proj.weight"] = ("BF16", (hidden, n_head * head_dim))
        tensors[p + "post_attention_layernorm.weight"] = ("BF16", (hidden,))
        if n_expert:
            tensors[p + "mlp.gate.weight"] = ("BF16", (n_expert, hidden))
            for e in range(n_expert):
                ep = p + f"mlp.experts.{e}."
                tensors[ep + "gate_proj.weight"] = ("BF16", (expert_inter, hidden))
                tensors[ep + "up_proj.weight"] = ("BF16", (expert_inter, hidden))
                tensors[ep + "down_proj.weight"] = ("BF16", (hidden, expert_inter))
        else:
            tensors[p + "mlp.gate_proj.weight"] = ("BF16", (inter, hidden))
            tensors[p + "mlp.up_proj.weight"] = ("BF16", (inter, hidden))
            tensors[p + "mlp.down_proj.weight"] = ("BF16", (hidden, inter))

    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_safetensors(root / "model.safetensors", tensors)
    return root


class TestSafetensorsReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_offsets(self):
        p = self.root / "x.safetensors"
        write_safetensors(p, {"a": ("F32", (2, 3)), "b": ("BF16", (4,))})
        ts = read_safetensors(p)
        self.assertEqual(ts["a"].nbytes, 24)
        self.assertEqual(ts["b"].nbytes, 8)
        self.assertEqual(ts["a"].shape, [2, 3])

    def test_header_size_mismatch_rejected(self):
        p = self.root / "bad.safetensors"
        hdr = {"a": {"dtype": "F32", "shape": [10], "data_offsets": [0, 4]}}
        js = json.dumps(hdr).encode()
        p.write_bytes(struct.pack("<Q", len(js)) + js + b"\x00" * 4)
        with self.assertRaises(Exception):
            read_safetensors(p)


class TestHFDense(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ir = build_ir_from_hf(make_hf_model(Path(self.tmp.name)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation(self):
        self.assertEqual(validate_ir(self.ir), [])

    def test_fields(self):
        self.assertEqual(self.ir.container, "safetensors")
        self.assertEqual(self.ir.n_layer, 2)
        self.assertEqual(self.ir.hidden, 64)
        self.assertEqual(self.ir.attn.n_head, 4)
        self.assertEqual(self.ir.attn.n_head_kv, 2)
        self.assertEqual(self.ir.attn.variant, "gqa")
        self.assertEqual(self.ir.moe.n_expert, 0)

    def test_roles(self):
        counts = {}
        for t in self.ir.tensors:
            counts[t.role] = counts.get(t.role, 0) + 1
        self.assertEqual(counts[Role.EMBED], 1)
        self.assertEqual(counts[Role.OUTPUT], 1)
        self.assertEqual(counts[Role.FFN_DENSE], 6)   # 2 layers x 3
        self.assertEqual(counts[Role.ATTN], 8)

    def test_kv(self):
        # 2 layers * 2 kv heads * (16+16) * 4 bytes (f32, what the runner allocates)
        from pyref.placement import kv_bytes_per_token
        self.assertEqual(kv_bytes_per_token(self.ir, 0), 2 * 2 * 32 * 4)


class TestHFMoE(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_hf_model(Path(self.tmp.name), n_expert=4)
        self.ir = build_ir_from_hf(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout_and_validation(self):
        self.assertEqual(self.ir.moe.layout, "split_per_expert")
        self.assertEqual(self.ir.moe.n_expert, 4)
        self.assertEqual(validate_ir(self.ir), [])

    def test_expert_tensor_count(self):
        experts = [t for t in self.ir.tensors if t.role == Role.EXPERT]
        self.assertEqual(len(experts), 2 * 4 * 3)

    def test_router(self):
        routers = [t for t in self.ir.tensors if t.role == Role.ROUTER]
        self.assertEqual(len(routers), 2)
        self.assertEqual(routers[0].shape, [4, 64])


if __name__ == "__main__":
    unittest.main()
