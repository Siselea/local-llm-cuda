"""Model IR: the single intermediate representation every container maps into.

Scope (v0, per engine-v0-spec-and-plan.md §3.1): safetensors(+index) and GGUF,
architectures qwen35 / deepseek2 / llama-like, quant Q4_K/Q5_K/Q8_0/BF16/FP8.

This module is the *reference implementation*: it must agree field-by-field with
the C++ loader or the C++ loader is wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from .gguf_parse import GGUFModel, GGUFError, TensorInfo


class IRValidationError(RuntimeError):
    pass


class Role(str, Enum):
    EMBED = "embed"
    OUTPUT = "output"
    ATTN = "attn"
    FFN_DENSE = "ffn_dense"
    EXPERT = "expert"
    SHARED_EXPERT = "shared_expert"
    ROUTER = "router"
    NORM = "norm"
    OTHER = "other"


@dataclass
class QuantDesc:
    scheme: str          # "F32" | "F16" | "BF16" | "Q4_K" | "Q5_K" | "Q8_0" | ...
    block: int           # elements per block
    bytes_per_block: int

    @property
    def bpp(self) -> float:
        return self.bytes_per_block * 8.0 / self.block


@dataclass
class TensorDesc:
    name: str
    shape: list[int]              # logical [out, in] order (GGUF dims reversed)
    dtype: str                    # quant scheme name
    role: Role
    layer: Optional[int]
    file_offset: int              # absolute offset into the container file
    nbytes: int
    n_elements: int
    quant: QuantDesc

    @property
    def bpp(self) -> float:
        return self.quant.bpp


@dataclass
class AttnDesc:
    variant: str = "gqa"          # mha | gqa | mqa | mla
    n_head: int = 0
    n_head_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    rope_freq_base: float = 10000.0
    rope_dim: int = 0
    sliding_window: int = 0
    full_attention_interval: int = 0   # hybrid SSM models: 1 full-attn every N layers
    kv_lora_rank: int = 0              # MLA: compressed KV latent width
    leading_dense_blocks: int = 0      # DeepSeek-style: first N layers are dense


@dataclass
class SSMDesc:
    variant: str = "none"
    conv_kernel: int = 0
    n_key_head: int = 0
    n_value_head: int = 0
    key_head_dim: int = 0
    value_head_dim: int = 0
    key_dim: int = 0
    value_dim: int = 0
    conv_dim: int = 0
    time_step_rank: int = 0


@dataclass
class MoEDesc:
    n_expert: int = 0
    n_expert_used: int = 0
    n_shared: int = 0
    router_kind: str = "topk_softmax"
    norm_topk: bool = False
    expert_ffn_length: int = 0
    layout: str = "none"          # none | fused_3d (GGUF) | split_per_expert (HF)


@dataclass
class LayerDesc:
    index: int
    n_tensors: int = 0
    n_expert_tensors: int = 0
    kind: str = "attn"            # attn | linear (qwen35 hybrid)
    attn: AttnDesc = field(default_factory=AttnDesc)
    moe: MoEDesc = field(default_factory=MoEDesc)
    norm: str = "rmsnorm"
    act: str = "silu"


@dataclass
class TokenizerDesc:
    model: str = "unknown"        # gpt2 | llama (sentencepiece) | ...
    pre: str = ""
    n_vocab: int = 0
    eos_id: int = -1
    chat_template: bool = False


@dataclass
class ModelIR:
    arch: str
    container: str
    path: str
    file_size: int
    n_layer: int
    hidden: int
    ffn: int
    vocab: int
    ctx_train: int
    quant_mix: dict                # scheme -> bytes
    attn: AttnDesc
    moe: MoEDesc
    tokenizer: TokenizerDesc
    ssm: SSMDesc = field(default_factory=SSMDesc)
    layers: list[LayerDesc] = field(default_factory=list)
    tensors: list[TensorDesc] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mtp_layers: int = 0

    # ---- derived helpers -------------------------------------------------
    def by_role(self, role: Role) -> list[TensorDesc]:
        return [t for t in self.tensors if t.role == role]

    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def fixed_bytes(self) -> int:
        """Weights that must stay resident on GPU (attention/dense/embed/head/norm/router/shared)."""
        roles = {Role.EMBED, Role.OUTPUT, Role.ATTN, Role.FFN_DENSE,
                 Role.NORM, Role.ROUTER, Role.SHARED_EXPERT}
        return sum(t.nbytes for t in self.tensors if t.role in roles)

    def expert_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors if t.role == Role.EXPERT)

    def expert_bytes_per_layer(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for t in self.tensors:
            if t.role == Role.EXPERT and t.layer is not None:
                out[t.layer] = out.get(t.layer, 0) + t.nbytes
        return out

    def to_json(self, include_tensors: bool = True) -> dict:
        d = {
            "arch": self.arch,
            "container": self.container,
            "path": self.path,
            "file_size": self.file_size,
            "n_layer": self.n_layer,
            "hidden": self.hidden,
            "ffn": self.ffn,
            "vocab": self.vocab,
            "ctx_train": self.ctx_train,
            "mtp_layers": self.mtp_layers,
            "quant_mix": self.quant_mix,
            "attn": asdict(self.attn),
            "moe": asdict(self.moe),
            **({"ssm": asdict(self.ssm)} if self.ssm.variant != "none" else {}),
            "tokenizer": asdict(self.tokenizer),
            "bytes": {
                "total": self.total_bytes(),
                "fixed": self.fixed_bytes(),
                "experts": self.expert_bytes(),
            },
            "warnings": self.warnings,
            "layers": [asdict(l) for l in self.layers],
        }
        if include_tensors:
            d["tensors"] = [
                {
                    "name": t.name, "shape": t.shape, "dtype": t.dtype,
                    "role": t.role.value, "layer": t.layer,
                    "offset": t.file_offset, "nbytes": t.nbytes,
                    "bpp": round(t.bpp, 4),
                }
                for t in self.tensors
            ]
        return d


# --------------------------------------------------------------------------
# role classification
# --------------------------------------------------------------------------

_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def classify(name: str) -> Role:
    if name == "token_embd.weight" or name.endswith(".token_embd.weight"):
        return Role.EMBED
    if name == "output.weight" or name.endswith(".output.weight"):
        return Role.OUTPUT
    if "_exps" in name:
        return Role.EXPERT
    if "_shexp" in name:
        return Role.SHARED_EXPERT
    if "ffn_gate_inp" in name:
        return Role.ROUTER
    if "attn_" in name or "ssm_" in name or "time_mix" in name or "rope" in name:
        return Role.ATTN
    if "norm" in name:
        return Role.NORM
    if "ffn_" in name:
        return Role.FFN_DENSE
    return Role.OTHER


def layer_of(name: str) -> Optional[int]:
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# GGUF -> IR
# --------------------------------------------------------------------------

_QUANT_NAMES = {
    "F32": "F32", "F16": "F16", "BF16": "BF16",
    "Q4_0": "Q4_0", "Q4_1": "Q4_1", "Q5_0": "Q5_0", "Q5_1": "Q5_1",
    "Q8_0": "Q8_0", "Q8_1": "Q8_1",
    "Q2_K": "Q2_K", "Q3_K": "Q3_K", "Q4_K": "Q4_K", "Q5_K": "Q5_K",
    "Q6_K": "Q6_K", "Q8_K": "Q8_K",
    "IQ4_NL": "IQ4_NL", "IQ4_XS": "IQ4_XS",
    "IQ2_XXS": "IQ2_XXS", "IQ2_XS": "IQ2_XS", "IQ2_S": "IQ2_S",
    "IQ3_XXS": "IQ3_XXS", "IQ3_S": "IQ3_S", "IQ1_S": "IQ1_S", "IQ1_M": "IQ1_M",
    "TQ1_0": "TQ1_0", "TQ2_0": "TQ2_0",
}


def _meta_int(g: GGUFModel, arch: str, suffix: str, default: Optional[int] = None) -> Optional[int]:
    v = g.get(f"{arch}.{suffix}", default)
    if v is None:
        return default
    return int(v)


def _meta_float(g: GGUFModel, arch: str, suffix: str, default: Optional[float] = None) -> Optional[float]:
    v = g.get(f"{arch}.{suffix}", default)
    if v is None:
        return default
    return float(v)


def build_ir_from_gguf(g: GGUFModel) -> ModelIR:
    arch = str(g.get("general.architecture", ""))
    if not arch:
        raise IRValidationError("GGUF missing general.architecture")

    n_layer = _meta_int(g, arch, "block_count")
    hidden = _meta_int(g, arch, "embedding_length")
    if not n_layer or not hidden:
        raise IRValidationError(f"missing block_count/embedding_length for arch {arch}")

    warnings: list[str] = []
    attn = AttnDesc(
        n_head=_meta_int(g, arch, "attention.head_count", 0) or 0,
        n_head_kv=_meta_int(g, arch, "attention.head_count_kv", 0) or 0,
        key_length=_meta_int(g, arch, "attention.key_length", 0) or 0,
        value_length=_meta_int(g, arch, "attention.value_length", 0) or 0,
        rope_freq_base=_meta_float(g, arch, "rope.freq_base", 10000.0) or 10000.0,
        rope_dim=_meta_int(g, arch, "rope.dimension_count", 0) or 0,
        sliding_window=_meta_int(g, arch, "attention.sliding_window", 0) or 0,
        full_attention_interval=_meta_int(g, arch, "full_attention_interval", 0) or 0,
    )
    if attn.key_length == 0 and attn.n_head > 0:
        # some archs (qwen2) do not store head_dim; it is hidden / n_head
        attn.key_length = attn.value_length = hidden // attn.n_head
    if attn.n_head_kv and attn.n_head:
        if attn.n_head_kv == 1:
            attn.variant = "mqa"
        elif attn.n_head_kv < attn.n_head:
            attn.variant = "gqa"
        else:
            attn.variant = "mha"
    # MLA (deepseek2): kv_lora_rank present in metadata
    kv_lora = _meta_int(g, arch, "attention.kv_lora_rank", 0) or 0
    if kv_lora:
        attn.variant = "mla"
        attn.kv_lora_rank = kv_lora
    attn.leading_dense_blocks = _meta_int(g, arch, "leading_dense_block_count", 0) or 0

    router_kind = "sigmoid_topk" if arch == "deepseek3" else "topk_softmax"
    moe = MoEDesc(
        router_kind=router_kind,
        n_expert=_meta_int(g, arch, "expert_count", 0) or 0,
        n_expert_used=_meta_int(g, arch, "expert_used_count", 0) or 0,
        n_shared=_meta_int(g, arch, "expert_shared_count", 0) or 0,
        norm_topk=bool(_meta_int(g, arch, "expert_weights_norm", 0) or 0),
        expert_ffn_length=_meta_int(g, arch, "expert_feed_forward_length", 0) or 0,
        layout="fused_3d" if (_meta_int(g, arch, "expert_count", 0) or 0) else "none",
    )

    # qwen35 hybrid stacks: gated delta net parameters
    ssm = SSMDesc()
    conv_k = _meta_int(g, arch, "ssm.conv_kernel", 0) or 0
    inner = _meta_int(g, arch, "ssm.inner_size", 0) or 0
    state = _meta_int(g, arch, "ssm.state_size", 0) or 0
    groups = _meta_int(g, arch, "ssm.group_count", 0) or 0
    tsr = _meta_int(g, arch, "ssm.time_step_rank", 0) or 0
    if conv_k > 0 and inner > 0 and state > 0 and groups > 0:
        if (moe.n_expert or 0) > 0:
            moe.norm_topk = True
            if not moe.n_shared:
                moe.n_shared = 1
        ssm = SSMDesc(
            variant="gated_delta", conv_kernel=conv_k,
            n_key_head=groups, n_value_head=tsr, key_head_dim=state,
            value_head_dim=(inner // tsr) if tsr else 0,
            key_dim=groups * state, value_dim=inner,
            conv_dim=groups * state * 2 + inner, time_step_rank=tsr,
        )

    tok = TokenizerDesc(
        model=str(g.get("tokenizer.ggml.model", "unknown")),
        pre=str(g.get("tokenizer.ggml.pre", "")),
        n_vocab=len(g.get("tokenizer.ggml.tokens", []) or []),
        eos_id=int(g.get("tokenizer.ggml.eos_token_id", -1)),
        chat_template=bool(g.get("tokenizer.chat_template")),
    )

    tensors: list[TensorDesc] = []
    quant_mix: dict[str, int] = {}
    for t in g.tensors:
        scheme = _QUANT_NAMES.get(t.type_name, t.type_name)
        q = QuantDesc(scheme=scheme, block=1, bytes_per_block=0)
        # recover block/size from the parser table
        from .gguf_parse import GGML_TYPES
        _, blck, size = GGML_TYPES[t.type_id]
        q.block, q.bytes_per_block = blck, size
        tensors.append(TensorDesc(
            name=t.name,
            shape=t.shape,
            dtype=scheme,
            role=classify(t.name),
            layer=layer_of(t.name),
            file_offset=g.data_offset + t.offset,
            nbytes=t.nbytes,
            n_elements=t.n_elements,
            quant=q,
        ))
        quant_mix[scheme] = quant_mix.get(scheme, 0) + t.nbytes

    layers = [LayerDesc(index=i) for i in range(n_layer)]
    for t in tensors:
        if t.layer is None or t.layer >= n_layer:
            continue
        L = layers[t.layer]
        L.n_tensors += 1
        if t.role == Role.EXPERT:
            L.n_expert_tensors += 1
        L.attn = attn
        L.moe = moe
    if ssm.variant != "none":
        interval = _meta_int(g, arch, "full_attention_interval", 0) or 0
        for i in range(len(layers)):
            layers[i].kind = "attn" if (interval > 0 and (i + 1) % interval == 0) else "linear"

    # MTP / nextn head: extra blocks that are NOT transformer layers. They must be
    # excluded from layer count, KV accounting and expert placement (spec §1.2).
    mtp_layers = sorted({t.layer for t in tensors
                         if t.layer is not None and ".nextn." in t.name})
    if mtp_layers:
        n_layer = min(n_layer, min(mtp_layers))
        layers = layers[:n_layer]
        for t in tensors:
            if t.layer is not None and t.layer >= n_layer:
                t.role = Role.OTHER
        warnings.append(f"MTP head detected on block(s) {mtp_layers}; excluded from {n_layer} layers")
    ffn = _meta_int(g, arch, "feed_forward_length", 0) or 0
    ctx_train = _meta_int(g, arch, "context_length", 0) or 0
    vocab = tok.n_vocab or (_meta_int(g, arch, "vocab_size", 0) or 0)

    ir = ModelIR(
        arch=arch, container="gguf", path=str(g.path), file_size=g.file_size,
        n_layer=n_layer, hidden=hidden, ffn=ffn, vocab=vocab, ctx_train=ctx_train,
        quant_mix=quant_mix, attn=attn, moe=moe, ssm=ssm, tokenizer=tok,
        layers=layers, tensors=tensors, warnings=warnings,
        mtp_layers=len(mtp_layers),
    )

    # sanity warnings
    if ir.moe.n_expert and ir.moe.n_expert_used > ir.moe.n_expert:
        warnings.append("expert_used_count > expert_count")
    if attn.key_length == 0 and attn.n_head:
        warnings.append("key_length missing; KV estimate will be 0")
    return ir


# --------------------------------------------------------------------------
# validation (spec §3.3) — must reject, not silently proceed
# --------------------------------------------------------------------------

def validate_ir(ir: ModelIR) -> list[str]:
    errs: list[str] = []

    if ir.n_layer <= 0:
        errs.append("n_layer <= 0")
    if not ir.tensors:
        errs.append("no tensors")

    # 1) per-layer expected tensors
    seen_layers = {t.layer for t in ir.tensors if t.layer is not None}
    missing = [i for i in range(ir.n_layer) if i not in seen_layers]
    if missing:
        errs.append(f"layers with no tensors: {missing[:8]}")

    # 2) expert table completeness — two layouts must both validate:
    #    fused_3d (GGUF): 3 tensors/layer, leading dim == n_expert
    #    split_per_expert (HF): n_expert*3 tensors/layer
    if ir.moe.n_expert:
        router_layers = {t.layer for t in ir.tensors if t.role == Role.ROUTER}
        expert_layers: dict[int, list[TensorDesc]] = {}
        for t in ir.tensors:
            if t.role == Role.EXPERT and t.layer is not None:
                expert_layers.setdefault(t.layer, []).append(t)

        for layer in sorted(router_layers):
            ts = expert_layers.get(layer, [])
            fused = any(len(t.shape) == 3 for t in ts)
            if fused:
                if len(ts) != 3:
                    errs.append(
                        f"layer {layer}: fused MoE layer must have 3 tensors, got {len(ts)}")
                    continue
                for t in ts:
                    if t.shape[0] != ir.moe.n_expert:
                        errs.append(
                            f"{t.name}: expert dim {t.shape[0]} != n_expert {ir.moe.n_expert}")
            else:
                expected = ir.moe.n_expert * 3
                if len(ts) != expected:
                    errs.append(
                        f"layer {layer}: split MoE layer must have {expected} tensors "
                        f"(n_expert*3), got {len(ts)}")
                for t in ts:
                    if len(t.shape) != 2:
                        errs.append(f"{t.name}: expected 2-D expert matrix, got {t.shape}")

        for layer in sorted(set(expert_layers) - router_layers):
            errs.append(f"layer {layer}: expert tensors present but no ffn_gate_inp router")

        if not router_layers:
            errs.append("moe.n_expert > 0 but no router tensors found")

    # 3) shape consistency for dense FFN (gate/up/down)
    by_name = {t.name: t for t in ir.tensors}

    def check_triplet(suffix_gate: str, suffix_up: str, suffix_down: str, inter: int):
        for i in range(ir.n_layer):
            g = by_name.get(f"blk.{i}.{suffix_gate}")
            u = by_name.get(f"blk.{i}.{suffix_up}")
            d = by_name.get(f"blk.{i}.{suffix_down}")
            if g and u and len(g.shape) == 2 and len(u.shape) == 2:
                if g.shape != u.shape:
                    errs.append(f"layer {i}: gate/up shape mismatch {g.shape} vs {u.shape}")
                if d and len(d.shape) == 2 and d.shape[1] != g.shape[0]:
                    errs.append(f"layer {i}: down shape {d.shape} incompatible with gate {g.shape}")

    if ir.ffn:
        check_triplet("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", ir.ffn)

    # 4) quant block divisibility
    for t in ir.tensors:
        if t.quant.block > 1 and t.n_elements % t.quant.block != 0:
            errs.append(f"{t.name}: {t.n_elements} not divisible by block {t.quant.block}")

    # 5) attention invariants
    a = ir.attn
    if a.n_head and a.n_head_kv and a.n_head % a.n_head_kv != 0:
        errs.append(f"n_head {a.n_head} not divisible by n_head_kv {a.n_head_kv}")

    # 6) tensor table coverage (no holes / overlaps within the container)
    spans = sorted(((t.file_offset, t.file_offset + t.nbytes, t.name) for t in ir.tensors))
    for (s0, e0, n0), (s1, e1, n1) in zip(spans, spans[1:]):
        if s1 < e0:
            errs.append(f"tensor overlap: {n0} ends {e0} > {n1} starts {s1}")
        elif s1 > e0:
            # holes are allowed (padding) but flag large ones
            if s1 - e0 > 1 << 20:
                ir.warnings.append(f"hole of {s1 - e0} bytes after {n0}")

    # 7) vocab vs embedding rows
    emb = next((t for t in ir.tensors if t.role == Role.EMBED), None)
    if emb and ir.vocab and len(emb.shape) == 2 and emb.shape[0] != ir.vocab:
        errs.append(f"embedding rows {emb.shape[0]} != vocab {ir.vocab}")

    if ir.file_size and spans:
        last_end = spans[-1][1]
        if last_end > ir.file_size:
            errs.append(f"tensor data exceeds file size: {last_end} > {ir.file_size}")

    return errs
