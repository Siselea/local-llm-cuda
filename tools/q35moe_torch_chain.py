"""Chained torch reference for Qwen3.5-MoE: runs the official modules layer by layer on the
real GGUF weights and compares with the engine's per-layer dumps.

Engine: ENGINE_DUMP_EVERY=build/m35L build/test_gen.exe <35B> "760" 1
Usage : python tools/q35moe_torch_chain.py --ids "760" --dump build/m35L [--layers 0-5]
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

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeAttention,
    Qwen3_5MoeExperts,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextRotaryEmbedding,
)

MODEL = r"<MODEL_DIR>\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf"


def has_tensor(f, name):
    return name in f.info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="760")
    ap.add_argument("--dump", default="build/m35L")
    ap.add_argument("--layers", default="0-5")
    ap.add_argument("--init-dump", default="", help="start from this engine layer dump instead of the embedding")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.layers.split("-"))
    f = F(args.model)
    g = f.g
    hidden = int(g.get("qwen35moe.embedding_length"))
    ffn = int(g.get("qwen35moe.feed_forward_length") or 0)
    moe_inter = int(g.get("qwen35moe.expert_feed_forward_length"))
    n_expert = int(g.get("qwen35moe.expert_count"))
    n_used = int(g.get("qwen35moe.expert_used_count"))
    sh_inter = int(g.get("qwen35moe.expert_shared_feed_forward_length") or 512)
    eps = float(g.get("qwen35moe.attention.layer_norm_rms_epsilon"))
    interval = int(g.get("qwen35moe.full_attention_interval") or 4)
    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=hidden, num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        intermediate_size=ffn or moe_inter, num_hidden_layers=1,
        layer_types=["linear_attention"], rms_norm_eps=eps, rope_theta=10000000.0,
        partial_rotary_factor=0.25, linear_conv_kernel_dim=4, linear_num_key_heads=16,
        linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128,
        num_experts=n_expert, num_experts_per_tok=n_used, moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=sh_inter, norm_topk_prob=True,
    )
    rot = Qwen3_5MoeTextRotaryEmbedding(cfg)
    if args.init_dump:
        init = np.fromfile(args.init_dump, dtype=np.float32)
        x = torch.from_numpy(init.reshape(1, -1, hidden))
    else:
        x = torch.from_numpy(f.t("token_embd.weight")[ids].astype(np.float32)).unsqueeze(0)
    pos = torch.arange(T).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(x, pos)

    def load_norm(mod, name):
        mod.weight.copy_(torch.from_numpy(np.ascontiguousarray(f.t(name) - 1.0)))

    for i in range(lo, min(hi + 1, 40)):
        p = f"blk.{i}."
        is_linear = (i + 1) % interval != 0
        with torch.no_grad():
            ln1 = Qwen3_5MoeRMSNorm(hidden, eps=eps)
            load_norm(ln1, p + "attn_norm.weight")
            hn = ln1(x)
            if is_linear:
                mix = Qwen3_5MoeGatedDeltaNet(cfg, 0).float().eval()
                sd = mix.state_dict()
                sd["in_proj_qkv.weight"].copy_(torch.from_numpy(f.t(p + "attn_qkv.weight")))
                sd["in_proj_z.weight"].copy_(torch.from_numpy(f.t(p + "attn_gate.weight")))
                sd["in_proj_a.weight"].copy_(torch.from_numpy(f.t(p + "ssm_alpha.weight")))
                sd["in_proj_b.weight"].copy_(torch.from_numpy(f.t(p + "ssm_beta.weight")))
                sd["conv1d.weight"].copy_(
                    torch.from_numpy(f.t(p + "ssm_conv1d.weight").reshape(-1, 1, 4)))
                sd["dt_bias"].copy_(torch.from_numpy(f.t(p + "ssm_dt.bias")))
                sd["A_log"].copy_(torch.from_numpy(f.t(p + "ssm_a")))
                sd["norm.weight"].copy_(torch.from_numpy(f.t(p + "ssm_norm.weight")))
                sd["out_proj.weight"].copy_(torch.from_numpy(f.t(p + "ssm_out.weight")))
                mix_out = mix(hn)
                if i == 3:
                    print("  [dbg] hn norm", float(hn.norm()), "mix norm", float(mix_out.norm()))
            else:
                att = Qwen3_5MoeAttention(cfg, 0).float().eval()
                sd = att.state_dict()
                sd["q_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_q.weight")))
                sd["k_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_k.weight")))
                sd["v_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_v.weight")))
                sd["o_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_output.weight")))
                sd["q_norm.weight"].copy_(torch.from_numpy(f.t(p + "attn_q_norm.weight") - 1.0))
                sd["k_norm.weight"].copy_(torch.from_numpy(f.t(p + "attn_k_norm.weight") - 1.0))
                neg = torch.finfo(torch.float32).min
                tri = torch.triu(torch.ones(T, T), diagonal=1).bool()
                bias = torch.zeros(1, 1, T, T)
                bias.masked_fill_(tri, neg)
                mix_out, _ = att(hn, position_embeddings=(cos, sin), attention_mask=bias)
                if i == 3:
                    print("  [dbg] hn norm", float(hn.norm()), "mix norm", float(mix_out.norm()))
            x = x + mix_out
            ln2 = Qwen3_5MoeRMSNorm(hidden, eps=eps)
            load_norm(ln2, p + "post_attention_norm.weight")
            h2 = ln2(x)
            # MoE FFN
            blk = Qwen3_5MoeSparseMoeBlock(cfg).float().eval()
            sd = blk.state_dict()
            sd["gate.weight"].copy_(torch.from_numpy(f.t(p + "ffn_gate_inp.weight")))
            gate_e = f.t(p + "ffn_gate_exps.weight")   # [E, inter, hidden]
            up_e = f.t(p + "ffn_up_exps.weight")
            down_e = f.t(p + "ffn_down_exps.weight")   # [E, hidden, inter]
            sd["experts.gate_up_proj"].copy_(
                torch.from_numpy(np.concatenate([gate_e, up_e], axis=1)))
            sd["experts.down_proj"].copy_(torch.from_numpy(down_e))
            sd["shared_expert.gate_proj.weight"].copy_(
                torch.from_numpy(f.t(p + "ffn_gate_shexp.weight")))
            sd["shared_expert.up_proj.weight"].copy_(
                torch.from_numpy(f.t(p + "ffn_up_shexp.weight")))
            sd["shared_expert.down_proj.weight"].copy_(
                torch.from_numpy(f.t(p + "ffn_down_shexp.weight")))
            sd["shared_expert_gate.weight"].copy_(
                torch.from_numpy(f.t(p + "ffn_gate_inp_shexp.weight").reshape(1, -1)))
            x = x + blk(h2)
        print(f"  [dbg] x norm {float(x.norm()):.4f}", flush=True)
        fp = Path(f"{args.dump}_L{i}.f32")
        xn = x.squeeze(0).numpy()
        if fp.exists():
            ref = np.fromfile(fp, dtype=np.float32).reshape(T, hidden)
            print(f"layer {i:2d} ({'linear' if is_linear else 'attn'}): "
                  f"{np.linalg.norm(ref - xn)/np.linalg.norm(xn):.3e}", flush=True)
        else:
            print(f"layer {i:2d} ({'linear' if is_linear else 'attn'}): (no dump)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
