#!/usr/bin/env python
"""M2 gate: compare the C++ placement plan against the Python reference (JSON).

Single-model entry point used by tools/verify.sh; the multi-model matrix lives in
tools/planner_parity.py (same comparison core).  Memory inputs are simulated
(--vram-gib/--host-gib) so the two documents are byte-deterministic.

Usage: python tools/diff_plan.py <model.gguf> [--ctx N] [--vram-gib F] [--host-gib F]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner_parity as parity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--vram-gib", type=float, default=12.0)
    ap.add_argument("--host-gib", type=float, default=parity.HOST_GIB)
    ap.add_argument("--kv-type", default="f32")
    args = ap.parse_args()
    py_doc, cpp_doc = parity.run_one(args.model, args.ctx, args.vram_gib, args.kv_type,
                                     f"diff_{Path(args.model).stem[:20]}")
    diffs = parity.diff_doc(py_doc, cpp_doc)
    print(f"model      : {Path(args.model).name}")
    print(f"ctx        : {args.ctx}   vram: {args.vram_gib} GiB   host: {args.host_gib} GiB")
    if diffs:
        print(f"M2         : FAIL - {len(diffs)} field(s) differ")
        for d in diffs[:15]:
            print(f"  - {d}")
        return 1
    print(f"M2         : PASS - plan identical ({len(cpp_doc)} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
