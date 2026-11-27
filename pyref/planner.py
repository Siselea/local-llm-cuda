"""Budget planner: measured constants -> VRAM placement plan (spec §4).

Authority: `planner::compute` in src/planner.cpp.  This reference mirrors it field for
field; `tools/planner_parity.py` asserts the two agree on real + synthetic models.

All bandwidth constants come from P0 measurements on this machine
(see default-workspace/p0-results-2026-11-11.md). Do not replace them with
theoretical specs without re-measuring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

from . import placement as _placement
from .ir import ModelIR

# --- measured constants (P0, 2026-11-11, RTX 3060 12 GB) -------------------
BW_PCIE_H2D_GBPS = 12.0
BW_PCIE_D2H_GBPS = 11.0
BW_RAM_COPY_GBPS = 22.5
BW_CPU_GEMV_GBPS = 4.4       # llama.cpp haswell AVX2, batch=1, Q5_K
BW_HBM_GBPS = 360.0          # spec sheet, not measured
T_COPY_LAT_US = 16.6
DEFAULT_VRAM_RESERVE = 1024 << 20   # 1 GiB left to the driver/display
DEFAULT_WORKSPACE = 256 << 20       # compute buffers, activations, cuBLAS ws


@dataclass
class PlacementPlan:
    ctx: int
    kv_type: int               # 0 f32 / 1 q8_0 / 2 q4_0 (the layout the runner allocates)
    kv_name: str
    vram_total: int
    vram_reserve: int
    fixed_bytes: int
    kv_bytes: int
    workspace_bytes: int
    expert_budget: int
    expert_total: int
    expert_per_layer: dict
    layer_placement: dict      # MoE block index -> "gpu" / "cpu"
    slots_per_layer: int       # advisory: uniform expert slots/layer that fit the budget
    layers_on_gpu: int
    layers_on_cpu: int
    fully_resident: bool
    # projections
    proj_decode_cpu_ms: float
    proj_decode_pcie_ms: float
    notes: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def plan(ir: ModelIR, ctx: int = 8192, kv_type: int = 0,
         vram_total: int = 12 * (1 << 30), vram_reserve: int = DEFAULT_VRAM_RESERVE,
         workspace: int = DEFAULT_WORKSPACE) -> PlacementPlan:
    notes: list[str] = []
    fixed = ir.fixed_bytes()
    # The budget uses the layout the runner really allocates (kv_type), not a
    # bytes-per-value guess: the two differ 2x for the default f32 cache.
    kv = _placement.kv_bytes(ir, ctx, kv_type)
    expert_total = ir.expert_bytes()
    per_layer = ir.expert_bytes_per_layer()
    n_layers_with_experts = len(per_layer)

    usable = vram_total - vram_reserve
    expert_budget = usable - fixed - kv - workspace
    if expert_budget < 0:
        notes.append("fixed weights + KV already exceed VRAM; no expert can be resident")
        expert_budget = 0

    layer_placement: dict[int, str] = {}
    if n_layers_with_experts == 0:
        slots = 0
        fully = True
        layers_on_gpu = 0
        layers_on_cpu = 0
    else:
        # Advisory uniform estimate (what --n-cpu-moe expresses), kept for display.
        avg_layer_bytes = expert_total / n_layers_with_experts
        slots = expert_budget // max(int(avg_layer_bytes), 1)
        fully = expert_budget >= expert_total
        if ir.moe.n_expert:
            slots = min(slots, ir.moe.n_expert)
        slots = max(0, int(slots))
        # Exact layer-granular placement: walk the real per-layer expert bytes instead
        # of assuming the average (#27).
        layers_on_gpu = 0
        used = 0
        for _layer, bytes_ in sorted(per_layer.items()):
            if used + bytes_ > expert_budget:
                break
            used += bytes_
            layers_on_gpu += 1
        layers_on_cpu = max(0, n_layers_with_experts - layers_on_gpu)
        for idx, layer in enumerate(sorted(per_layer)):
            layer_placement[layer] = "gpu" if idx < layers_on_gpu else "cpu"

    # projections: bytes per token touched by experts, split GPU/CPU by slot ratio
    active_expert_bytes = 0
    if ir.moe.n_expert and ir.moe.n_expert_used and n_layers_with_experts > 0:
        avg = expert_total / n_layers_with_experts
        share = ir.moe.n_expert_used / ir.moe.n_expert
        active_expert_bytes = int(avg * share * n_layers_with_experts)

    hit = (layers_on_gpu / n_layers_with_experts) if n_layers_with_experts else 1.0
    cpu_bytes = int(active_expert_bytes * (1 - hit))

    proj_cpu_ms = cpu_bytes / (BW_CPU_GEMV_GBPS * 1e9) * 1e3
    proj_pcie_ms = active_expert_bytes / (BW_PCIE_H2D_GBPS * 1e9) * 1e3

    if not fully and n_layers_with_experts:
        notes.append(
            f"expert placement: {layers_on_gpu}/{n_layers_with_experts} MoE layers fit; "
            f"llama.cpp equivalent is --n-cpu-moe {layers_on_cpu}")
    if ctx > ir.ctx_train and ir.ctx_train:
        notes.append(f"requested ctx {ctx} > trained ctx {ir.ctx_train}")
    if ir.attn.variant == "mla":
        notes.append(
            f"MLA KV: {ir.attn.kv_lora_rank} latent + {ir.attn.rope_dim} rope per layer/token")
    if ir.attn.full_attention_interval > 1:
        notes.append(
            f"hybrid model: only {math.ceil(ir.n_layer / ir.attn.full_attention_interval)} "
            f"of {ir.n_layer} layers hold KV")

    return PlacementPlan(
        ctx=ctx, kv_type=kv_type, kv_name=_placement.kv_layout_name(kv_type),
        vram_total=vram_total, vram_reserve=vram_reserve,
        fixed_bytes=fixed, kv_bytes=kv, workspace_bytes=workspace,
        expert_budget=int(expert_budget), expert_total=expert_total,
        expert_per_layer={str(k): v for k, v in sorted(per_layer.items())},
        layer_placement={str(k): v for k, v in sorted(layer_placement.items())},
        slots_per_layer=int(slots), layers_on_gpu=layers_on_gpu, layers_on_cpu=layers_on_cpu,
        fully_resident=fully,
        proj_decode_cpu_ms=round(proj_cpu_ms, 2),
        proj_decode_pcie_ms=round(proj_pcie_ms, 2),
        notes=notes)
