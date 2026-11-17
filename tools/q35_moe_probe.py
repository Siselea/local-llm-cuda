"""MoE probe for qwen3.5-MoE (35B): compare the engine's layer-0 FFN output against a
numpy implementation of Qwen3_5MoeSparseMoeBlock semantics.

Engine dumps (ENGINE_DUMP=build/m35 ENGINE_DUMP_Q35=1):
  _q35_mid0.f32   hidden after mixer + residual (MoE block input)
  _q35_ffn_in.f32 hidden after post_attention_norm (router/expert input)
  _q35_x0.f32     layer output
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds2_probe import F  # noqa: E402

MODEL = r"<MODEL_DIR>\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"
HIDDEN = 2048
INTER = 512
N_EXP = 256
TOP_K = 8


def silu(x):
    return x / (1.0 + np.exp(-x))


def main() -> int:
    f = F(MODEL)
    dump = Path("build")
    p = "blk.0."
    router = f.t(p + "ffn_gate_inp.weight")            # [256, 2048] f32
    sh_gate = f.t(p + "ffn_gate_shexp.weight")         # [512, 2048]
    sh_up = f.t(p + "ffn_up_shexp.weight")
    sh_down = f.t(p + "ffn_down_shexp.weight")
    sh_gate_inp = f.t(p + "ffn_gate_inp_shexp.weight")  # [2048]

    x_mid = np.fromfile(dump / "m35_q35_mid0.f32", dtype=np.float32).reshape(-1, HIDDEN)
    hn = np.fromfile(dump / "m35_q35_ffn_in.f32", dtype=np.float32).reshape(-1, HIDDEN)
    x_out = np.fromfile(dump / "m35_q35_x0.f32", dtype=np.float32).reshape(-1, HIDDEN)
    T = x_mid.shape[0]
    target = (x_out - x_mid).astype(np.float64)   # MoE contribution

    logits = hn.astype(np.float64) @ router.T.astype(np.float64)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    idx = np.argsort(-probs, axis=-1)[:, :TOP_K]
    w = np.take_along_axis(probs, idx, axis=-1)
    w /= w.sum(-1, keepdims=True)

    gate_exps = f.t(p + "ffn_gate_exps.weight")   # [256, 512, 2048]
    up_exps = f.t(p + "ffn_up_exps.weight")
    down_exps = f.t(p + "ffn_down_exps.weight")

    out = np.zeros((T, HIDDEN), np.float64)
    for t in range(T):
        h = hn[t].astype(np.float64)
        for j, e in enumerate(idx[t]):
            g = gate_exps[e].astype(np.float64) @ h
            u = up_exps[e].astype(np.float64) @ h
            act = silu(g) * u
            out[t] += w[t, j] * (down_exps[e].astype(np.float64) @ act)
        # shared expert with sigmoid gate
        sg = silu(sh_gate.astype(np.float64) @ h)
        su = sh_up.astype(np.float64) @ h
        shared = sh_down.astype(np.float64) @ (sg * su)
        gate_scalar = 1.0 / (1.0 + np.exp(-(sh_gate_inp.astype(np.float64) @ h)))
        out[t] += gate_scalar * shared

    print(f"moe out  : rel={np.linalg.norm(out - target)/np.linalg.norm(target):.3e}")
    print(f"norms    : numpy={np.linalg.norm(out):.3f} engine={np.linalg.norm(target):.3f}")
    print("top experts for token 0:", idx[0][:5].tolist(), "weights", np.round(w[0][:5], 3).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
