"""Generate a tiny synthetic *gemma4* GGUF for llama.cpp-anchored semantics work.

The real 12B model is 7 GB and takes ~2.5 min per reference forward on CPU.
A tiny gemma4 model with the same metadata structure / tensor layout gives the
same code paths in llama.cpp but loads in milliseconds, so architecture
hypotheses can be tested exactly and cheaply (embedding/attention/norm/scalar/
softcap semantics), then ported to the engine.

Layout: 2 layers, layer0 sliding (head_dim 8, n_kv 2, has attn_v), layer1 full
(head_dim 16, n_kv 1, attention_k_eq_v -> no attn_v), tied LM head.

Usage: python tools/make_tiny_gemma4.py [out_dir]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuzz_gguf as F  # GGUF writer helpers

HIDDEN = 64
N_HEAD = 4
SWA_D = 8
FULL_D = 16
SWA_NKV = 2
FULL_NKV = 1
FFN = 128
VOCAB = 64
N_LAYER = 2
SWA_WINDOW = 4
EPS = 1e-6
SEED = 1234


def f32(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 0.05).astype(np.float32).tobytes()


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/fixtures/tiny_gemma4")
    out.mkdir(parents=True, exist_ok=True)
    g = F.GGUF()
    arch = "gemma4"
    g.string("general.architecture", arch)
    g.u32(f"{arch}.block_count", N_LAYER)
    g.u32(f"{arch}.context_length", 128)
    g.u32(f"{arch}.embedding_length", HIDDEN)
    g.u32(f"{arch}.feed_forward_length", FFN)
    g.u32(f"{arch}.attention.head_count", N_HEAD)
    g.add(f"{arch}.attention.head_count_kv", F.T_ARRAY,
          struct.pack("<IQ", F.T_U32, N_LAYER) + struct.pack("<II", SWA_NKV, FULL_NKV))
    g.f32(f"{arch}.rope.freq_base", 1000000.0)
    g.f32(f"{arch}.rope.freq_base_swa", 10000.0)
    g.f32(f"{arch}.attention.layer_norm_rms_epsilon", EPS)
    g.u32(f"{arch}.attention.key_length", FULL_D)
    g.u32(f"{arch}.attention.value_length", FULL_D)
    g.f32(f"{arch}.final_logit_softcapping", 30.0)
    g.u32(f"{arch}.attention.sliding_window", SWA_WINDOW)
    g.u32(f"{arch}.attention.shared_kv_layers", 0)
    g.u32(f"{arch}.embedding_length_per_layer_input", 0)
    g.add(f"{arch}.attention.sliding_window_pattern", F.T_ARRAY,
          struct.pack("<IQ", F.T_BOOL, N_LAYER) + bytes([1, 0]))
    g.u32(f"{arch}.attention.key_length_swa", SWA_D)
    g.u32(f"{arch}.attention.value_length_swa", SWA_D)
    g.u32(f"{arch}.rope.dimension_count", FULL_D)
    g.u32(f"{arch}.rope.dimension_count_swa", SWA_D)
    # tokenizer: minimal gemma4-style (ids are what we feed; strings are cosmetic)
    g.string("tokenizer.ggml.model", "gemma4")
    g.add("tokenizer.ggml.tokens", F.T_ARRAY, F.str_array([f"<t{i}>" for i in range(VOCAB)]))
    g.add("tokenizer.ggml.scores", F.T_ARRAY,
          struct.pack("<IQ", F.T_F32, VOCAB) + struct.pack("<%df" % VOCAB, *([0.0] * VOCAB)))
    g.add("tokenizer.ggml.token_type", F.T_ARRAY,
          struct.pack("<IQ", F.T_I32, VOCAB) + struct.pack("<%di" % VOCAB, *([1] * VOCAB)))
    g.add("tokenizer.ggml.merges", F.T_ARRAY, struct.pack("<IQ", F.T_STRING, 0))
    g.u32("tokenizer.ggml.bos_token_id", 2)
    g.u32("tokenizer.ggml.eos_token_id", 3)
    g.u32("tokenizer.ggml.unknown_token_id", 0)
    g.u32("tokenizer.ggml.padding_token_id", 0)
    g.add("tokenizer.ggml.add_bos_token", F.T_BOOL, bytes([1]))
    g.add("tokenizer.ggml.add_space_prefix", F.T_BOOL, bytes([0]))

    g.tensor("token_embd.weight", [VOCAB, HIDDEN], f32((VOCAB, HIDDEN), SEED + 1))
    g.tensor("output_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 2))
    # llama.cpp gemma4 requires this tensor (real model: [n_rot/2], 1.0 then 1e30);
    # for the text path it is a loader requirement, not a maths input.
    g.tensor("rope_freqs.weight", [FULL_D // 2],
             struct.pack("<%df" % (FULL_D // 2), *([1.0] * (FULL_D // 2))))
    for i in range(N_LAYER):
        swa = i == 0
        d = SWA_D if swa else FULL_D
        nkv = SWA_NKV if swa else FULL_NKV
        pre = f"blk.{i}."
        p = f"_{i}"
        g.tensor(pre + "attn_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 10 + i))
        g.tensor(pre + "attn_q.weight", [N_HEAD * d, HIDDEN], f32((N_HEAD * d, HIDDEN), SEED + 20 + i))
        g.tensor(pre + "attn_k.weight", [nkv * d, HIDDEN], f32((nkv * d, HIDDEN), SEED + 30 + i))
        if swa:
            g.tensor(pre + "attn_v.weight", [nkv * d, HIDDEN], f32((nkv * d, HIDDEN), SEED + 40 + i))
        g.tensor(pre + "attn_output.weight", [HIDDEN, N_HEAD * d], f32((HIDDEN, N_HEAD * d), SEED + 50 + i))
        g.tensor(pre + "attn_q_norm.weight", [d], f32((d,), SEED + 60 + i))
        g.tensor(pre + "attn_k_norm.weight", [d], f32((d,), SEED + 70 + i))
        g.tensor(pre + "ffn_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 80 + i))
        g.tensor(pre + "ffn_gate.weight", [FFN, HIDDEN], f32((FFN, HIDDEN), SEED + 90 + i))
        g.tensor(pre + "ffn_up.weight", [FFN, HIDDEN], f32((FFN, HIDDEN), SEED + 100 + i))
        g.tensor(pre + "ffn_down.weight", [HIDDEN, FFN], f32((HIDDEN, FFN), SEED + 110 + i))
        g.tensor(pre + "post_attention_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 120 + i))
        g.tensor(pre + "post_ffw_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 130 + i))
        g.tensor(pre + "layer_output_scale.weight", [1], struct.pack("<f", 0.5 + 0.1 * i))
    data = g.build()
    (out / "model.gguf").write_bytes(data)
    print(f"wrote {out / 'model.gguf'} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
