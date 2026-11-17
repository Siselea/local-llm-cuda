#!/usr/bin/env python
"""M2 gate: compare the C++ placement plan against the Python reference (JSON).

Usage: python tools/diff_plan.py <model.gguf> [--ctx N] [--vram-gib F]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENGINE = ROOT / "build" / "engine.exe"
TMP = ROOT / "build" / "irdiff"


def norm(v):
    if isinstance(v, str):
        return v.replace("\\", "/")
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--vram-gib", type=float, default=12.0)
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    py, cpp = TMP / "py_plan.json", TMP / "cpp_plan.json"

    subprocess.run([PY, "-m", "pyref.cli", "plan", args.model, "--ctx", str(args.ctx),
                    "--vram-gib", str(args.vram_gib), "--json", str(py)],
                   cwd=ROOT, check=True, capture_output=True)
    subprocess.run([str(ENGINE), "plan", args.model, "--ctx", str(args.ctx),
                    "--vram-gib", str(args.vram_gib), "--json", str(cpp)],
                   cwd=ROOT, check=True, capture_output=True)

    a = norm(json.loads(py.read_text(encoding="utf-8")))
    b = norm(json.loads(cpp.read_text(encoding="utf-8")))
    diffs = [f"{k}: py={a.get(k)!r} cpp={b.get(k)!r}" for k in sorted(set(a) | set(b))
             if a.get(k) != b.get(k)]
    print(f"model      : {Path(args.model).name}")
    print(f"ctx        : {args.ctx}   vram: {args.vram_gib} GiB")
    if diffs:
        print(f"M2         : FAIL - {len(diffs)} field(s) differ")
        for d in diffs[:15]:
            print(f"  - {d}")
        return 1
    print(f"M2         : PASS - plan identical ({len(a)} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
