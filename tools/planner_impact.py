#!/usr/bin/env python
"""Quantify the two C++ planner estimate defects for the PLANNER-PARITY report:

1. KV under-report: legacy `kv_bpe` (2 bytes/value) vs the real kv_type layout (f32 = 4).
2. Average-vs-exact layer placement: uniform `budget // avg_layer` vs walking the real
   per-layer expert bytes.

Run from the repo root: python tools/planner_impact.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyref.cli import _load          # noqa: E402
from pyref import placement as pl    # noqa: E402
from pyref.planner import DEFAULT_VRAM_RESERVE, DEFAULT_WORKSPACE  # noqa: E402

MIB = 1 << 20
GIB = 1 << 30
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MODELS = [
    "DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf",
    "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf",
]


def old_layers(ir, expert_budget: int) -> int:
    """Old C++ planner::compute / placement::decide count (uniform average)."""
    per = ir.expert_bytes_per_layer()
    n = len(per)
    if not n:
        return 0
    avg = ir.expert_bytes() / n
    slots = expert_budget // max(int(avg), 1)
    if ir.moe.n_expert:
        slots = min(slots, ir.moe.n_expert)
    slots = max(0, slots)
    if expert_budget >= ir.expert_bytes():
        return n
    return min(n, slots)


def exact_layers(ir, budget: int) -> int:
    used = 0
    count = 0
    for _layer, bytes_ in sorted(ir.expert_bytes_per_layer().items()):
        if used + bytes_ > budget:
            break
        used += bytes_
        count += 1
    return count


def main() -> int:
    print(f"# planner estimate impact ({len(MODELS)} models)\n")
    for name in MODELS:
        path = MODEL_DIR / name
        if not path.exists():
            print(f"(missing) {path}")
            continue
        ir = _load(str(path))
        per = ir.expert_bytes_per_layer()
        n = len(per)
        total = ir.expert_bytes()
        avg = total / n if n else 0
        sizes = sorted(per.values())
        print(f"## {name}")
        print(f"   MoE layers={n}  expert_total={total / GIB:.3f} GiB  "
              f"avg/layer={avg / GIB:.3f} GiB  min={sizes[0] / GIB:.3f}  "
              f"max={sizes[-1] / GIB:.3f}  spread={sizes[-1] / max(sizes[0], 1):.2f}x")
        for ctx in (8192, 32768):
            kv_old = ir.kv_bytes_per_token(2) * ctx            # legacy bytes-per-value
            kv_new = pl.kv_bytes(ir, ctx, 0)                   # real f32 layout
            for vram in (4, 8, 12, 24):
                usable = vram * GIB - DEFAULT_VRAM_RESERVE
                fixed = ir.fixed_bytes()
                budget_old = max(0, usable - fixed - kv_old - DEFAULT_WORKSPACE)
                budget_new = max(0, usable - fixed - kv_new - DEFAULT_WORKSPACE)
                l_old = old_layers(ir, budget_old)
                l_avg_new_kv = old_layers(ir, budget_new)
                l_exact = exact_layers(ir, budget_new)
                flag = ""
                if l_old != l_exact:
                    flag = f"  <-- old plan off by {l_exact - l_old:+d} layer(s)"
                print(f"   ctx={ctx:<6} vram={vram:>2} GiB: kv {kv_old / MIB:7.1f} -> "
                      f"{kv_new / MIB:7.1f} MiB ({kv_new / max(kv_old, 1):.1f}x)  "
                      f"layers: avg+old-kv={l_old:2d} avg+new-kv={l_avg_new_kv:2d} "
                      f"exact+new-kv={l_exact:2d}{flag}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
