"""Placement decider + memory accounting: Python mirror of the `placement` namespace in
`src/planner.cpp` (which is the authority).

Why this module exists: `engine plan` prints a budget plan followed by a placement
decision.  The budget numbers come from `planner::compute` (mirrored by pyref/planner.py)
but the placement decision (kind / gpu_moe_layers / host need / ctx recommendation) lives
in `placement::` and was previously C++ only, so the Python reference could not be
compared against it.  Everything here is a byte-for-byte mirror: same formulas, same
strings, no extra dependencies.

Units: bytes everywhere, `ctx` in tokens, `kv_type` 0 = f32 / 1 = q8_0 / 2 = q4_0 (the
layout the runner really allocates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .ir import ModelIR

# Same absolute ctx cap as DenseRunner::derive_ctx_limit().
K_MAX_CTX_POLICY = 1 << 20


def nb32(d: int) -> int:
    return (d + 31) // 32


def kv_layout_name(kv_type: int) -> str:
    return "q8_0" if kv_type == 1 else ("q4_0" if kv_type == 2 else "f32")


def kv_layer_count(ir: ModelIR) -> int:
    """Number of layers that hold a KV cache (hybrid SSM stacks keep KV only on the
    full-attention layers).  Mirrors placement::kv_layer_count."""
    n = 0
    layers = ir.layers
    for i in range(ir.n_layer):
        kind_attn = i >= len(layers) or layers[i].kind == "attn"
        has = not (ir.ssm.variant != "none" and not kind_attn)
        if has:
            n += 1
    if n <= 0:
        n = ir.n_layer
    return n


def kv_bytes_per_token(ir: ModelIR, kv_type: int) -> int:
    """KV cache bytes per token for the layout the engine actually allocates.

    NOTE: this is NOT `ModelIR.kv_bytes_per_token(bpe)` - that legacy helper takes an
    abstract bytes-per-value and ignores kv_type, which under-reported the default f32
    cache by 2x in the budget planner.
    """
    n = kv_layer_count(ir)
    if n <= 0:
        return 0
    a = ir.attn
    if a.variant == "mla":
        # MLA allocates the expanded per-head K/V cache, f32 only.
        return n * a.n_head * (a.key_length + a.value_length) * 4
    if a.n_head_kv <= 0 or a.key_length <= 0:
        return 0
    klen = a.key_length
    vlen = a.value_length if a.value_length else a.key_length
    if kv_type == 0:
        return n * a.n_head_kv * (klen + vlen) * 4
    bb = 34 if kv_type == 1 else 18
    return n * a.n_head_kv * (nb32(klen) + nb32(vlen)) * bb


def kv_bytes(ir: ModelIR, ctx: int, kv_type: int) -> int:
    c = ctx if ctx > 0 else 1
    return kv_bytes_per_token(ir, kv_type) * c


def activation_bytes(ir: ModelIR, rows: int) -> int:
    """Activation/scratch bytes for `rows` batch rows - mirror of
    DenseRunner::activation_bytes() (same fields, same formula)."""
    r = rows if rows > 0 else 1
    hidden = ir.hidden
    n_head = ir.attn.n_head
    head_dim = ir.attn.key_length
    n_kv = ir.attn.n_head_kv if ir.attn.n_head_kv else n_head
    inter = ir.ffn if ir.ffn > 0 else hidden * 4
    per_row = hidden * 4 * 6
    per_row += n_head * head_dim * 4 * 2
    per_row += n_kv * head_dim * 4 * 2
    per_row += inter * 4 * 3
    per_row += ir.vocab * 4
    return r * per_row + (2 << 20)


def default_host_reserve(host_total_bytes: int) -> int:
    return max(host_total_bytes // 16, 1 << 30)


def fmt_bytes(n: int) -> str:
    """C++ placement::fmt_bytes (binary units, 2 decimals)."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    v = float(n)
    u = 0
    while v >= 1024.0 and u < 4:
        v /= 1024.0
        u += 1
    return f"{v:.2f} {units[u]}"


KIND_NAMES = {"gpu": "gpu", "split": "vram+ram split", "ssd": "ssd stream"}


def kind_name(kind: str) -> str:
    return KIND_NAMES[kind]


@dataclass
class MemoryInfo:
    valid: bool = False
    total_bytes: int = 0
    free_bytes: int = 0


@dataclass
class CtxFit:
    kv_type: int = 0
    kv_name: str = "f32"
    kv_per_token: int = 0
    max_ctx: int = 0
    kv_at_max: int = 0


def estimate_ctx(ir: ModelIR, kv_type: int, free_vram: int, vram_reserve: int,
                 row_cap: int) -> CtxFit:
    """Pre-load ctx recommendation; mirrors placement::estimate_ctx (the load-time
    derivation in DenseRunner is still authoritative)."""
    f = CtxFit()
    f.kv_type = kv_type
    f.kv_name = kv_layout_name(kv_type)
    if ir.attn.variant == "mla" and kv_type != 0:
        f.kv_type = 0
        f.kv_name = "f32"
    f.kv_per_token = kv_bytes_per_token(ir, f.kv_type)
    model_ctx = int(min(ir.ctx_train, K_MAX_CTX_POLICY)) if ir.ctx_train > 0 else K_MAX_CTX_POLICY
    want = max(1, model_ctx)
    if f.kv_per_token == 0:
        f.max_ctx = want
        return f
    cap = row_cap if row_cap > 0 else 1
    weights = ir.fixed_bytes()
    reserve = vram_reserve
    seq = want
    fit = want
    # Same fixed-point iteration as DenseRunner::derive_ctx_limit(): longer ctx ->
    # bigger activation row buffer -> smaller KV budget.
    for _ in range(4):
        rows = max(1, min(seq, cap))
        act = activation_bytes(ir, rows)
        avail = free_vram - weights - act - reserve if free_vram > weights + act + reserve else 0
        fit = int(avail // f.kv_per_token)
        fit = min(fit, want)
        if fit >= seq:
            break
        seq = fit
        if seq < 1:
            seq = 1
    f.max_ctx = max(0, min(want, fit))
    f.kv_at_max = f.kv_per_token * f.max_ctx
    return f


@dataclass
class Decision:
    kind: str = "gpu"
    model_bytes: int = 0
    fixed_bytes: int = 0
    expert_bytes: int = 0
    kv_bytes: int = 0
    workspace_bytes: int = 0
    vram_total: int = 0
    vram_free: int = 0
    vram_reserve: int = 0
    vram_usable: int = 0
    vram_need: int = 0
    vram_left: int = 0
    host_total: int = 0
    host_free: int = 0
    host_reserve: int = 0
    host_usable: int = 0
    host_need: int = 0
    n_moe_layers: int = 0
    gpu_moe_layers: int = 0
    fits_vram: bool = False
    fits_all: bool = False
    dense_offload_unsupported: bool = False
    reason: str = ""
    notes: list = field(default_factory=list)


def decide(ir: ModelIR, ctx: int, kv_type: int, vram: MemoryInfo, host: MemoryInfo,
           vram_reserve: int, host_reserve: int, workspace: int,
           ssd_requested: bool = False) -> Decision:
    """Mirror of placement::decide().  Keep the strings byte-identical to C++."""
    d = Decision()
    total = ir.total_bytes()
    d.fixed_bytes = ir.fixed_bytes()
    d.expert_bytes = ir.expert_bytes()
    d.model_bytes = total if total else d.fixed_bytes + d.expert_bytes
    c = max(1, ctx)
    d.kv_bytes = kv_bytes(ir, c, kv_type)
    d.workspace_bytes = workspace

    d.vram_total = vram.total_bytes
    d.vram_free = vram.free_bytes
    d.vram_reserve = vram_reserve
    d.vram_usable = d.vram_free - d.vram_reserve if d.vram_free > d.vram_reserve else 0
    d.host_total = host.total_bytes
    d.host_free = host.free_bytes
    d.host_reserve = host_reserve
    d.host_usable = d.host_free - d.host_reserve if d.host_free > d.host_reserve else 0

    d.vram_need = d.fixed_bytes + d.kv_bytes + d.workspace_bytes
    per_layer = ir.expert_bytes_per_layer()
    d.n_moe_layers = len(per_layer)
    vram_need_all = d.vram_need + d.expert_bytes

    if not vram.valid:
        d.kind = "gpu"
        d.reason = "VRAM probe unavailable: the loader's measured free VRAM decides"
        d.notes.append("device memory probe failed (no GPU/driver); residency was not verified")
        return d
    if vram_need_all <= d.vram_usable:
        d.kind = "gpu"
        d.fits_vram = True
        d.fits_all = True
        d.gpu_moe_layers = d.n_moe_layers
        d.reason = (f"fixed + experts + KV + workspace ({fmt_bytes(vram_need_all)}) fit in "
                    f"usable VRAM ({fmt_bytes(d.vram_usable)})")
        if d.n_moe_layers:
            d.notes.append(f"all {d.n_moe_layers} MoE layers stay GPU-resident; "
                           f"no CPU/SSD path needed")
        return d
    if d.n_moe_layers > 0 and d.vram_need <= d.vram_usable:
        d.vram_left = d.vram_usable - d.vram_need
        d.host_need = d.expert_bytes - d.vram_left if d.expert_bytes > d.vram_left else 0
        d.fits_all = host.valid and d.host_need <= d.host_usable
        if d.fits_all:
            d.kind = "split"
            # Exact layer-granular estimate (same rule as the budget plan).
            d.gpu_moe_layers = 0
            used = 0
            for _layer, bytes_ in sorted(per_layer.items()):
                if used + bytes_ > d.vram_left:
                    break
                used += bytes_
                d.gpu_moe_layers += 1
            d.gpu_moe_layers = max(0, min(d.gpu_moe_layers, d.n_moe_layers))
            d.reason = "weights exceed VRAM but fit VRAM + host RAM: CPU expert offload path"
            d.notes.append(f"VRAM keeps {fmt_bytes(d.vram_left)} of experts -> about "
                           f"{d.gpu_moe_layers}/{d.n_moe_layers} MoE layers GPU-resident, "
                           f"the rest run on the host CPU path")
        else:
            d.kind = "ssd"
            d.reason = "weights do not fit VRAM + host RAM: only disk streaming remains"
            if not host.valid:
                d.notes.append("host RAM probe unavailable: treated as not fitting")
    else:
        d.kind = "ssd"
        d.dense_offload_unsupported = d.n_moe_layers == 0
        if d.dense_offload_unsupported:
            d.reason = ("dense model does not fit VRAM and this build has no dense "
                        "host/SSD weight path")
        else:
            d.reason = "non-expert weights + KV alone exceed usable VRAM"
    if ssd_requested and d.kind != "ssd":
        d.kind = "ssd"
        d.reason = ("SSD streaming forced by --ssd-stream / --placement ssd though the "
                    "weights would fit")
    if d.kind == "ssd":
        if d.host_need > 0:
            d.notes.append(f"would spill {fmt_bytes(d.host_need)} past host RAM "
                           f"({fmt_bytes(d.host_usable)} usable)")
        d.notes.append("SSD streaming is not implemented in this build; enable it explicitly "
                       "with --ssd-stream (interface placeholder) after lowering --ctx / using "
                       "a smaller quantisation, or pass --allow-oversubscribe to attempt the "
                       "load anyway")
    return d
