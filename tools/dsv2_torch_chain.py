"""Chained torch reference for DeepSeek-V2 (Coder-V2-Lite): runs the official modules on
the real GGUF weights layer by layer and compares with the engine's per-layer dumps.

Engine: ENGINE_DUMP_EVERY=build/cd build/test_gen.exe <coder> "<ids>" 1
Usage : python tools/dsv2_torch_chain.py --dump build/cd --ids "549,6077,280,7239"
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

from transformers.models.deepseek_v2.configuration_deepseek_v2 import (  # noqa: E402
    DeepseekV2Config,
)
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (  # noqa: E402
    DeepseekV2Attention,
    DeepseekV2MLP,
    DeepseekV2Moe,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
)

MODEL = r"<MODEL_DIR>\DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"


def has_tensor(f, name):
    return name in f.info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="549,6077,280,7239")
    ap.add_argument("--dump", default="build/cd")
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--norm-plus1", action="store_true")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.layers.split("-"))
    f = F(args.model)
    g = f.g
    hidden = int(g.get("deepseek2.embedding_length"))
    n_head = int(g.get("deepseek2.attention.head_count"))
    n_kv = int(g.get("deepseek2.attention.head_count_kv"))
    qk_nope = 128
    qk_rope = int(g.get("deepseek2.rope.dimension_count") or 64)
    v_dim = 128
    kv_lora = int(g.get("deepseek2.attention.kv_lora_rank"))
    n_layer = int(g.get("deepseek2.block_count"))
    n_expert = int(g.get("deepseek2.expert_count"))
    n_used = int(g.get("deepseek2.expert_used_count"))
    n_shared = int(g.get("deepseek2.expert_shared_count") or 2)
    inter = int(g.get("deepseek2.expert_feed_forward_length"))
    sh_inter = int(g.get("deepseek2.expert_shared_feed_forward_length") or inter * n_shared)
    eps = float(g.get("deepseek2.attention.layer_norm_rms_epsilon"))
    rope_base = float(g.get("deepseek2.rope.freq_base"))
    scale_depth = float(g.get("deepseek2.attention.scale_depth") or 1.0)
    ids = [int(x) for x in args.ids.split(",")]
    T = len(ids)

    cfg = DeepseekV2Config(
        hidden_size=hidden, num_attention_heads=n_head, num_key_value_heads=n_kv,
        qk_nope_head_dim=qk_nope, qk_rope_head_dim=qk_rope, v_head_dim=v_dim,
        kv_lora_rank=kv_lora, num_hidden_layers=n_layer,
        n_routed_experts=n_expert, num_experts_per_tok=n_used,
        n_shared_experts=n_shared, moe_intermediate_size=inter,
        intermediate_size=int(g.get("deepseek2.feed_forward_length") or inter),
        rms_norm_eps=eps, rope_theta=rope_base, max_position_embeddings=163840,
        first_k_dense_replace=int(g.get("deepseek2.leading_dense_block_count") or 1),
        moe_layer_freq=1, topk_method="greedy", norm_topk_prob=True,
        attention_bias=False, q_lora_rank=None,
        # the GGUF metadata marks YaRN scaling (factor 40, original ctx 4096)
        rope_scaling={"type": "yarn", "factor": 40.0,
                      "original_max_position_embeddings": 4096,
                      "beta_fast": 32, "beta_slow": 1,
                      "mscale": 1.0, "mscale_all_dim": 0.0707},
    )
    rot = DeepseekV2RotaryEmbedding(cfg)
    x = torch.from_numpy(f.t("token_embd.weight")[ids].astype(np.float32)).unsqueeze(0)
    pos = torch.arange(T).unsqueeze(0)
    with torch.no_grad():
        freqs_cis = rot(x, pos)

    for i in range(lo, min(hi + 1, n_layer)):
        p = f"blk.{i}."
        with torch.no_grad():
            ln1 = DeepseekV2RMSNorm(hidden, eps=eps)
            n1 = f.t(p + "attn_norm.weight")
            ln1.weight.copy_(torch.from_numpy(n1 + 1.0 if args.norm_plus1 else n1))
            hn = ln1(x)
            if i == 0:
                import numpy as _np
                _np.asarray(hn.squeeze(0).numpy(), dtype=_np.float32).tofile("build/chain_hn0.f32")
            att = DeepseekV2Attention(cfg, i).float().eval()
            sd = att.state_dict()
            sd["q_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_q.weight")))
            sd["kv_a_proj_with_mqa.weight"].copy_(
                torch.from_numpy(f.t(p + "attn_kv_a_mqa.weight")))
            nk = f.t(p + "attn_kv_a_norm.weight")
            sd["kv_a_layernorm.weight"].copy_(
                torch.from_numpy(nk + 1.0 if args.norm_plus1 else nk))
            sd["kv_b_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_kv_b.weight")))
            sd["o_proj.weight"].copy_(torch.from_numpy(f.t(p + "attn_output.weight")))
            neg = torch.finfo(torch.float32).min
            tri = torch.triu(torch.ones(T, T), diagonal=1).bool()
            bias = torch.zeros(1, 1, T, T)
            bias.masked_fill_(tri, neg)
            if i == 0:
                import numpy as _np
                with torch.no_grad():
                    _ckv = att.kv_a_proj_with_mqa(hn)
                    _np.asarray(_ckv.squeeze(0).numpy(), dtype=_np.float32).tofile(
                        "build/chain_ckv0.f32")
                    _kvn = att.kv_a_layernorm(_ckv[..., : att.kv_lora_rank])
                    _np.asarray(_kvn.squeeze(0).numpy(), dtype=_np.float32).tofile(
                        "build/chain_kvn0.f32")
            if i == 0:
                import numpy as _np
                with torch.no_grad():
                    qproj = att.q_proj(hn).view(1, T, att.num_heads, -1)
                    q_nope_t, q_pe_t = torch.split(qproj, [att.qk_nope_head_dim,
                                                           att.qk_rope_head_dim], dim=-1)
                    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
                        apply_rotary_emb as _are)
                    q_pe_r, _ = _are(q_pe_t.transpose(1, 2), q_pe_t.transpose(1, 2),
                                     freqs_cis)
                    qfull = torch.cat([q_nope_t.transpose(1, 2), q_pe_r], dim=-1)
                    _np.asarray(qfull.squeeze(0).permute(1, 0, 2).contiguous().numpy(),
                                dtype=_np.float32).tofile("build/chain_q0.f32")
            if i == 0:
                import numpy as _np
                with torch.no_grad():
                    ckv = att.kv_a_proj_with_mqa(hn)
                    kvn = att.kv_a_layernorm(ckv[..., : att.kv_lora_rank])
                    kb = att.kv_b_proj(kvn).view(1, T, att.num_heads, -1)
                    kn = kb[..., : att.qk_nope_head_dim]
                    kpe = ckv[..., att.kv_lora_rank:].unsqueeze(1)   # [1,1,T,64]
                    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
                        apply_rotary_emb as _are2)
                    kpe_r, _ = _are2(kpe, kpe, freqs_cis)
                    kpe_b = kpe_r.transpose(1, 2).expand(-1, -1, att.num_heads, -1)
                    kfull = torch.cat([kn, kpe_b], dim=-1)
                    _np.asarray(kfull.squeeze(0).numpy(), dtype=_np.float32).tofile(
                        "build/chain_k0.f32")
            mix, _ = att(hn, position_embeddings=freqs_cis, attention_mask=bias)
            if i == 0:
                import numpy as _np
                _np.asarray(mix.squeeze(0).numpy(), dtype=_np.float32).tofile(
                    "build/chain_attn0.f32")
                # also record the expanded V so we can separate score vs value scaling
                with torch.no_grad():
                    ckv = att.kv_a_proj_with_mqa(hn)
                    kvn = att.kv_a_layernorm(ckv[..., : att.kv_lora_rank])
                    kvb = att.kv_b_proj(kvn).view(1, T, att.num_heads,
                                                  att.qk_nope_head_dim + att.v_head_dim)
                    vv = kvb[..., att.qk_nope_head_dim:]
                    _np.asarray(vv.squeeze(0).numpy(), dtype=_np.float32).tofile(
                        "build/chain_v0.f32")
            x = x + mix
            ln2 = DeepseekV2RMSNorm(hidden, eps=eps)
            n2 = f.t(p + "ffn_norm.weight")
            ln2.weight.copy_(torch.from_numpy(n2 + 1.0 if args.norm_plus1 else n2))
            h2 = ln2(x)
            if has_tensor(f, p + "ffn_gate_exps.weight"):
                moe = DeepseekV2Moe(cfg).float().eval()
                sd = moe.state_dict()
                sd["experts.gate_up_proj"].copy_(torch.from_numpy(np.concatenate(
                    [f.t(p + "ffn_gate_exps.weight"), f.t(p + "ffn_up_exps.weight")], axis=1)))
                sd["experts.down_proj"].copy_(torch.from_numpy(f.t(p + "ffn_down_exps.weight")))
                sd["gate.weight"].copy_(torch.from_numpy(f.t(p + "ffn_gate_inp.weight")))
                sd["shared_experts.gate_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_gate_shexp.weight")))
                sd["shared_experts.up_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_up_shexp.weight")))
                sd["shared_experts.down_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_down_shexp.weight")))
                if has_tensor(f, p + "ffn_gate_shexp.weight"):
                    sd["shared_experts.gate_proj.weight"].copy_(
                        torch.from_numpy(f.t(p + "ffn_gate_shexp.weight")))
                    sd["shared_experts.up_proj.weight"].copy_(
                        torch.from_numpy(f.t(p + "ffn_up_shexp.weight")))
                    sd["shared_experts.down_proj.weight"].copy_(
                        torch.from_numpy(f.t(p + "ffn_down_shexp.weight")))
                x = x + moe(h2)
            else:
                mlp = DeepseekV2MLP(cfg).float().eval()
                sd = mlp.state_dict()
                sd["gate_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_gate.weight")))
                sd["up_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_up.weight")))
                sd["down_proj.weight"].copy_(torch.from_numpy(f.t(p + "ffn_down.weight")))
                x = x + mlp(h2)
        if i == min(hi, n_layer - 1):
            with torch.no_grad():
                fn = DeepseekV2RMSNorm(hidden, eps=eps)
                nf = f.t("output_norm.weight")
                fn.weight.copy_(torch.from_numpy(nf + 1.0 if args.norm_plus1 else nf))
                xf = fn(x)
                logits = (xf @ torch.from_numpy(f.t("output.weight").T)).squeeze(0).numpy()[-1]
            mx = logits.max()
            lp = logits - mx - np.log(np.exp(logits - mx).sum())
            order = np.argsort(-logits)[:5]
            print("chain top5:", [(int(j), round(float(lp[j]), 3)) for j in order], flush=True)
        fp = Path(f"{args.dump}_L{i}.f32")
        xn = x.squeeze(0).numpy()
        if fp.exists():
            ref = np.fromfile(fp, dtype=np.float32).reshape(T, hidden)
            print(f"layer {i:2d}: {np.linalg.norm(ref - xn)/np.linalg.norm(xn):.3e}", flush=True)
        else:
            print(f"layer {i:2d}: (no dump)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
