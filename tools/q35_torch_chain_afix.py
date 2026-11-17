"""Chained torch reference for Qwen3.5: runs transformers' own modules layer by layer
with the real GGUF weights and compares against the engine's per-layer dumps.

Engine: ENGINE_DUMP_EVERY=build/q35N build/test_gen.exe <model> "<ids>" 1
Usage : python tools/q35_torch_chain.py --model <gguf> --dump build/q35N --ids "760,6511,314,9338"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds2_probe import F  # noqa: E402

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--max-layer", type=int, default=32)
    args = ap.parse_args()

    f = F(args.model)
    g = f.g
    hidden = int(g.get("qwen35.embedding_length"))
    ffn = int(g.get("qwen35.feed_forward_length"))
    eps = float(g.get("qwen35.attention.layer_norm_rms_epsilon"))
    interval = int(g.get("qwen35.full_attention_interval", 4))
    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)
    cfg = Qwen3_5TextConfig(
        hidden_size=hidden, num_attention_heads=16, num_key_value_heads=4, head_dim=256,
        intermediate_size=ffn, num_hidden_layers=1, layer_types=["linear_attention"],
        rms_norm_eps=eps, rope_theta=10000000.0, partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4, linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=128, linear_value_head_dim=128,
    )
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    x = torch.from_numpy(f.t("token_embd.weight")[ids].astype(np.float32)).unsqueeze(0)
    pos = torch.arange(T).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(x, pos)

    def load_norm(mod, name):
        w = f.t(name)
        mod.weight.copy_(torch.from_numpy(np.ascontiguousarray(w - 1.0)))

    for i in range(min(int(g.get("qwen35.block_count")), args.max_layer)):
        p = f"blk.{i}."
        is_linear = (i + 1) % interval != 0
        with torch.no_grad():
            h = Qwen3_5RMSNorm(hidden, eps=eps)
            load_norm(h, p + "attn_norm.weight")
            hn = h(x)
            if is_linear:
                mix = Qwen3_5GatedDeltaNet(cfg, 0).float().eval()
                sd = mix.state_dict()
                sd["in_proj_qkv.weight"].copy_(torch.from_numpy(f.t(p + "attn_qkv.weight")))
                sd["in_proj_z.weight"].copy_(torch.from_numpy(f.t(p + "attn_gate.weight")))
                sd["in_proj_a.weight"].copy_(torch.from_numpy(f.t(p + "ssm_alpha.weight")))
                sd["in_proj_b.weight"].copy_(torch.from_numpy(f.t(p + "ssm_beta.weight")))
                sd["conv1d.weight"].copy_(
                    torch.from_numpy(f.t(p + "ssm_conv1d.weight").reshape(-1, 1, 4)))
                sd["dt_bias"].copy_(torch.from_numpy(f.t(p + "ssm_dt.bias")))
                _a = f.t(p + "ssm_a").astype(np.float32)
                # GGUF ssm_a stores -A (llama.cpp LLM_TENSOR_SSM_A_NOSCAN); HF wants
                # A_log = log(A) so that g = -exp(A_log)*softplus = ssm_a*softplus.
                sd["A_log"].copy_(torch.from_numpy(np.log(np.maximum(-_a, 1e-8)).astype(np.float32)))
                sd["norm.weight"].copy_(torch.from_numpy(f.t(p + "ssm_norm.weight")))
                sd["out_proj.weight"].copy_(torch.from_numpy(f.t(p + "ssm_out.weight")))
                mix_out = mix(hn)
            else:
                att = Qwen3_5Attention(cfg, 0).float().eval()
                sd = att.state_dict()
                sd["q_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_q.weight")))
                sd["k_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_k.weight")))
                sd["v_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_v.weight")))
                sd["o_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_output.weight")))
                sd["q_norm.weight"].copy_(
                    torch.from_numpy(f.t(p + "attn_q_norm.weight") - 1.0))
                sd["k_norm.weight"].copy_(
                    torch.from_numpy(f.t(p + "attn_k_norm.weight") - 1.0))
                neg = torch.finfo(torch.float32).min
                tri = torch.triu(torch.ones(T, T), diagonal=1).bool()
                bias = torch.zeros(1, 1, T, T)
                bias.masked_fill_(tri, neg)
                mix_out, _ = att(hn, position_embeddings=(cos, sin), attention_mask=bias)
            x = x + mix_out
            h2 = Qwen3_5RMSNorm(hidden, eps=eps)
            load_norm(h2, p + "post_attention_norm.weight")
            mlp = Qwen3_5MLP(cfg, ffn).float().eval()
            sd = mlp.state_dict()
            sd["gate_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_gate.weight")))
            sd["up_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_up.weight")))
            sd["down_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_down.weight")))
            x = x + mlp(h2(x))
        xn = x.squeeze(0).numpy()
        fp = Path(f"{args.dump}_L{i}.f32")
        if fp.exists():
            ref = np.fromfile(fp, dtype=np.float32).reshape(T, hidden)
            print(f"layer {i:2d} ({'linear' if is_linear else 'attn'}): "
                  f"{np.linalg.norm(ref - xn)/np.linalg.norm(xn):.3e}", flush=True)
    with torch.no_grad():
        fn = Qwen3_5RMSNorm(hidden, eps=eps)
        fn.weight.copy_(torch.from_numpy(np.ascontiguousarray(
            f.t("output_norm.weight") - 1.0)))
        xf = fn(x)
        logits = (xf @ torch.from_numpy(f.t("output.weight").T)).squeeze(0).numpy()[-1]
    mx = logits.max()
    lp = logits - mx - np.log(np.exp(logits - mx).sum())
    order = np.argsort(-logits)[:5]
    print("torch-chain top5:", [(int(i), round(float(lp[i]), 3)) for i in order])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
