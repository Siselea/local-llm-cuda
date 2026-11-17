"""safetensors (+ HF config.json) reader -> IR.

Two containers in v0: GGUF (fused experts) and safetensors/transformers
(split experts). The IR normalizes both; `MoEDesc.layout` records which one.

safetensors layout: u64 header_len | JSON header | tensor data
JSON: {"name": {"dtype","shape","data_offsets":[begin,end]}, "__metadata__": {...}}
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from .ir import (ModelIR, TensorDesc, QuantDesc, AttnDesc, MoEDesc, LayerDesc,
                 TokenizerDesc, Role, IRValidationError)


class SafetensorsError(RuntimeError):
    pass


DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

# quantization containers that need a scheme we do not support in v0
_UNSUPPORTED_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx", "weight_scale", "weight_zero_point")


@dataclass
class STTensor:
    name: str
    dtype: str
    shape: list[int]
    begin: int
    end: int
    file: str
    data_start: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


def read_safetensors(path: str | Path) -> dict[str, STTensor]:
    path = Path(path)
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SafetensorsError(f"{path}: file shorter than 8 bytes")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 << 20:
            raise SafetensorsError(f"{path}: implausible header length {n}")
        hdr_raw = f.read(n)
    try:
        hdr = json.loads(hdr_raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SafetensorsError(f"{path}: bad JSON header: {e}") from e

    data_start = 8 + n
    out: dict[str, STTensor] = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        if dtype not in DTYPE_BYTES:
            raise SafetensorsError(f"{path}: tensor {name}: unsupported dtype {dtype}")
        b, e = info["data_offsets"]
        shape = list(info["shape"])
        expect = DTYPE_BYTES[dtype]
        n_elem = 1
        for d in shape:
            n_elem *= d
        if (e - b) != n_elem * expect:
            raise SafetensorsError(
                f"{path}: tensor {name}: data size {e - b} != shape {shape} x {expect}")
        out[name] = STTensor(name, dtype, shape, b, e, str(path), data_start)
    return out


def load_shards(model_dir: str | Path) -> tuple[dict[str, STTensor], int]:
    """Returns (tensors, total container bytes on disk)."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    tensors: dict[str, STTensor] = {}
    if index.exists():
        doc = json.loads(index.read_text(encoding="utf-8"))
        weight_map = doc.get("weight_map")
        if not isinstance(weight_map, dict):
            raise SafetensorsError(f"{index}: missing weight_map")
        cache: dict[str, dict[str, STTensor]] = {}
        for name, shard in weight_map.items():
            if shard not in cache:
                cache[shard] = read_safetensors(model_dir / shard)
            if name not in cache[shard]:
                raise SafetensorsError(f"{index}: tensor {name} not in shard {shard}")
            tensors[name] = cache[shard][name]
        total = sum((model_dir / s).stat().st_size for s in cache)
        return tensors, total

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise SafetensorsError(f"{model_dir}: no .safetensors files")
    for f in files:
        tensors.update(read_safetensors(f))
    return tensors, sum(f.stat().st_size for f in files)


def _layer_of_hf(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _classify_hf(name: str) -> Role:
    n = name
    if n.endswith("embed_tokens.weight") or n == "model.embed_tokens.weight":
        return Role.EMBED
    if n.endswith("lm_head.weight"):
        return Role.OUTPUT
    if ".mlp.shared_experts." in n or ".shared_experts." in n:
        return Role.SHARED_EXPERT
    if ".mlp.experts." in n or ".experts." in n:
        return Role.EXPERT
    if ".mlp.gate.weight" in n or n.endswith(".mlp.gate.weight") or ".router." in n:
        return Role.ROUTER
    if ".self_attn." in n or ".attention." in n:
        return Role.ATTN
    if "norm" in n.lower():
        return Role.NORM
    if ".mlp." in n:
        return Role.FFN_DENSE
    return Role.OTHER


def _cfg(config: dict, *keys, default=None):
    for k in keys:
        if k in config:
            return config[k]
    return default


def build_ir_from_hf(model_dir: str | Path) -> ModelIR:
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise IRValidationError(f"{model_dir}: config.json not found")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tensors_raw, container_bytes = load_shards(model_dir)

    # reject weight-only quantization containers we do not model yet
    for name in tensors_raw:
        if any(name.endswith(s) for s in _UNSUPPORTED_SUFFIXES):
            raise SafetensorsError(
                f"{name}: this safetensors checkpoint is a quantized container "
                f"(AWQ/GPTQ-style). v0 supports BF16/F16/FP8 safetensors only.")

    arch = str(_cfg(cfg, "model_type", "architectures", default="unknown"))
    if isinstance(arch, list):
        arch = arch[0] if arch else "unknown"

    n_layer = int(_cfg(cfg, "num_hidden_layers", "n_layer", default=0))
    hidden = int(_cfg(cfg, "hidden_size", "n_embd", default=0))
    ffn = int(_cfg(cfg, "intermediate_size", "n_inner", default=0))
    vocab = int(_cfg(cfg, "vocab_size", default=0))
    ctx_train = int(_cfg(cfg, "max_position_embeddings", "n_positions", default=0))
    n_head = int(_cfg(cfg, "num_attention_heads", "n_head", default=0))
    n_head_kv = int(_cfg(cfg, "num_key_value_heads", "n_head_kv", default=n_head))
    head_dim = int(_cfg(cfg, "head_dim", default=hidden // n_head if n_head else 0))

    attn = AttnDesc(
        n_head=n_head, n_head_kv=n_head_kv,
        key_length=head_dim, value_length=head_dim,
        rope_freq_base=float(_cfg(cfg, "rope_theta", "rope_embedding_base", default=10000.0)),
        rope_dim=int(_cfg(cfg, "partial_rotary_factor", default=1.0) * head_dim)
        if head_dim else 0,
        sliding_window=int(_cfg(cfg, "sliding_window", default=0) or 0),
        kv_lora_rank=int(_cfg(cfg, "kv_lora_rank", default=0) or 0),
    )
    if attn.kv_lora_rank:
        attn.variant = "mla"
    elif n_head_kv and n_head:
        attn.variant = "mqa" if n_head_kv == 1 else ("gqa" if n_head_kv < n_head else "mha")

    n_expert = int(_cfg(cfg, "num_experts", "num_local_experts", "n_routed_experts", default=0) or 0)
    n_expert_used = int(_cfg(cfg, "num_experts_per_tok", "n_group_used",
                             "num_experts_per_token", default=0) or 0)
    n_shared = int(_cfg(cfg, "n_shared_experts", "num_shared_experts", default=0) or 0)
    expert_ffn = int(_cfg(cfg, "moe_intermediate_size", "expert_intermediate_size", default=0) or 0)
    moe = MoEDesc(n_expert=n_expert, n_expert_used=n_expert_used, n_shared=n_shared,
                  expert_ffn_length=expert_ffn,
                  layout="split_per_expert" if n_expert else "none")

    tok = TokenizerDesc(
        model="hf-bpe",
        pre=str(_cfg(cfg, "tokenizer_class", default="")),
        n_vocab=vocab,
        eos_id=int(_cfg(cfg, "eos_token_id", default=-1) if isinstance(
            _cfg(cfg, "eos_token_id", default=-1), int) else -1),
        chat_template=(model_dir / "chat_template.jinja").exists() or "chat_template" in cfg,
    )

    tensors: list[TensorDesc] = []
    quant_mix: dict[str, int] = {}
    for name, t in tensors_raw.items():
        bpp = DTYPE_BYTES[t.dtype]
        q = QuantDesc(scheme=t.dtype, block=1, bytes_per_block=bpp)
        role = _classify_hf(name)
        tensors.append(TensorDesc(
            name=name, shape=t.shape, dtype=t.dtype, role=role,
            layer=_layer_of_hf(name),
            file_offset=t.data_start + t.begin, nbytes=t.nbytes,
            n_elements=max(1, t.nbytes // bpp), quant=q))
        quant_mix[t.dtype] = quant_mix.get(t.dtype, 0) + t.nbytes

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

    ir = ModelIR(
        arch=arch, container="safetensors", path=str(model_dir),
        file_size=container_bytes,
        n_layer=n_layer, hidden=hidden, ffn=ffn, vocab=vocab, ctx_train=ctx_train,
        quant_mix=quant_mix, attn=attn, moe=moe, tokenizer=tok,
        layers=layers, tensors=tensors)
    if not tensors:
        ir.warnings.append("no tensors found")
    return ir
