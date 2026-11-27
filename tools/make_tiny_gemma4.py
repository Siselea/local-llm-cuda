"""Generate a tiny synthetic *gemma4* GGUF for llama.cpp-anchored semantics work.

The real 12B model is 7 GB and takes ~2.5 min per reference forward on CPU.
A tiny gemma4 model with the same metadata structure / tensor layout gives the
same code paths in llama.cpp but loads in milliseconds, so architecture
hypotheses can be tested exactly and cheaply (embedding/attention/norm/scalar/
softcap semantics), then ported to the engine.

Layout: 2 layers, layer0 sliding (head_dim 8, n_kv 2, has attn_v), layer1 full
(head_dim 16, n_kv 1, attention_k_eq_v -> no attn_v), tied LM head.

The optional --rope-freqs table selects the gemma4 proportional-rope layout:
  ones      every pair rotates (plain full rotary)
  sentinel  25% of the pairs rotate, the rest use llama.cpp's 1e30 no-rotation
            sentinel (the real 12b file layout)

Usage: python tools/make_tiny_gemma4.py [out_dir] [--rope-freqs ones|sentinel]
"""

from __future__ import annotations

import argparse
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


def f32(shape, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 0.05 * scale).astype(np.float32).tobytes()


def build(out: Path, rope_mode: str = "ones", full_rope_base: float = 1000000.0,
          full_layer_weight_scale: float = 1.0,
          probe_full_dominant: bool = False) -> Path:
    """Write the tiny gemma4 GGUF into `out`; returns the model path.

    full_rope_base only affects the full-attention layers' rope base.  The
    default is the real gemma4 value (1e6); a smaller base makes the
    proportional-rope sentinel pairs much more visible at short prompt
    lengths, which is useful for the tiny semantics probe.
    """
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
    g.f32(f"{arch}.rope.freq_base", full_rope_base)
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
    # llama.cpp gemma4 binds this table (shape n_rot/2) to the full-attention
    # layers as per-pair frequency divisors: theta_i = pos*base^(-2i/n_rot)/ff[i].
    # The real 12b table is 64 x 1.0 + 192 x 1e30, i.e. only the first 25% of the
    # pairs rotate; 1e30 underflows theta to 0 => cos=1,sin=0 (no rotation).
    half = FULL_D // 2
    freqs = [1.0] * half
    if rope_mode == "sentinel":
        for i in range(half):
            if i >= max(1, half // 4):
                freqs[i] = 1e30
    elif rope_mode == "none":
        # degenerate proportional table: no pair rotates on the full layers
        freqs = [1e30] * half
    elif rope_mode != "ones":
        raise SystemExit(f"unknown rope_mode {rope_mode!r}")
    g.tensor("rope_freqs.weight", [half], struct.pack("<%df" % half, *freqs))
    for i in range(N_LAYER):
        swa = i == 0
        d = SWA_D if swa else FULL_D
        nkv = SWA_NKV if swa else FULL_NKV
        pre = f"blk.{i}."
        p = f"_{i}"
        # Probe sensitivity knob: amplifying the full layer's projections makes the
        # rope-frequency-factor effect visible on the final hidden state (the real
        # weights make that layer a small part of the residual stream).
        ws = full_layer_weight_scale if not swa else 1.0
        g.tensor(pre + "attn_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 10 + i))
        g.tensor(pre + "attn_q.weight", [N_HEAD * d, HIDDEN], f32((N_HEAD * d, HIDDEN), SEED + 20 + i, ws))
        g.tensor(pre + "attn_k.weight", [nkv * d, HIDDEN], f32((nkv * d, HIDDEN), SEED + 30 + i, ws))
        if swa:
            g.tensor(pre + "attn_v.weight", [nkv * d, HIDDEN], f32((nkv * d, HIDDEN), SEED + 40 + i))
        g.tensor(pre + "attn_output.weight", [HIDDEN, N_HEAD * d], f32((HIDDEN, N_HEAD * d), SEED + 50 + i, ws))
        g.tensor(pre + "attn_q_norm.weight", [d], f32((d,), SEED + 60 + i))
        g.tensor(pre + "attn_k_norm.weight", [d], f32((d,), SEED + 70 + i))
        g.tensor(pre + "ffn_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 80 + i))
        g.tensor(pre + "ffn_gate.weight", [FFN, HIDDEN], f32((FFN, HIDDEN), SEED + 90 + i, ws))
        g.tensor(pre + "ffn_up.weight", [FFN, HIDDEN], f32((FFN, HIDDEN), SEED + 100 + i, ws))
        g.tensor(pre + "ffn_down.weight", [HIDDEN, FFN], f32((HIDDEN, FFN), SEED + 110 + i, ws))
        g.tensor(pre + "post_attention_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 120 + i))
        g.tensor(pre + "post_ffw_norm.weight", [HIDDEN], f32((HIDDEN,), SEED + 130 + i))
        # Probe knob: with a tiny first-layer output scale the full layer's output
        # dominates the residual stream, which makes its rope changes visible on
        # the final hidden state (real weights hide them behind the residual).
        lscale = (0.02 if swa else 0.7) if probe_full_dominant else (0.5 + 0.1 * i)
        g.tensor(pre + "layer_output_scale.weight", [1], struct.pack("<f", lscale))
    path = out / "model.gguf"
    path.write_bytes(g.build())
    print(f"wrote {path} rope_freqs={rope_mode} ({path.stat().st_size} bytes)")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="build/fixtures/tiny_gemma4")
    ap.add_argument("--rope-freqs", choices=["ones", "sentinel", "none"], default="ones")
    ap.add_argument("--full-rope-base", type=float, default=1000000.0)
    ap.add_argument("--full-layer-weight-scale", type=float, default=1.0)
    ap.add_argument("--probe-full-dominant", action="store_true")
    args = ap.parse_args()
    build(Path(args.out_dir), args.rope_freqs, args.full_rope_base,
          args.full_layer_weight_scale, args.probe_full_dominant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
