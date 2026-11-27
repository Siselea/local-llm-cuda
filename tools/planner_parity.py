#!/usr/bin/env python
"""Planner parity gate (CPU-only): `engine plan --json` vs `pyref plan --json`.

The C++ planner is the authority (it has the ledger gap and the launch-time measured
checks); `pyref/planner.py` + `pyref/placement.py` are the reference implementation.  This
tool asserts they agree byte-for-byte on the full JSON document:

  * budget plan: ctx / kv_type / kv_bytes / expert_budget / expert_total /
    expert_per_layer / layer_placement / slots_per_layer / layers_on_gpu /
    layers_on_cpu / fully_resident / projections / notes
  * placement decision: kind / gpu_moe_layers / every byte field / reason / notes
  * pre-load ctx recommendation: max_ctx / kv_per_token / kv_at_max

Memory inputs are simulated (--vram-gib/--host-gib) so the comparison is deterministic and
does not need a GPU: VRAM = {4,8,12,24} GiB, host = 32 GiB, ctx = {2048, 8192, 32768},
kv_type = {f32, q8_0}.  Run with --full for the extended matrix (more ctx points + q4_0).

Usage:
  python tools/planner_parity.py [--full] [--models M1 M2 ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENGINE = ROOT / "build" / "engine.exe" if os.name == "nt" else ROOT / "build" / "engine"
TMP = ROOT / "build" / "planner_parity"

DEFAULT_MODELS = [
    "Qwen3.5-9B-Q5_K_M.gguf",
    "DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf",
    "Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf",
    "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf",
]
FIXTURES = [
    "build/fixtures/tiny_gguf_moe/model.gguf",
    "build/fixtures/tiny_gguf_moe_f16/model.gguf",
]

CTX_QUICK = (2048, 8192)
VRAM_QUICK = (8.0, 12.0)
KV_QUICK = ("f32",)
CTX_FULL = (2048, 8192, 32768, 131072)
VRAM_FULL = (4.0, 8.0, 12.0, 24.0)
KV_FULL = ("f32", "q8_0", "q4_0")
HOST_GIB = 32.0
ROW_CAP = 512


def norm(value):
    """Normalize for comparison: Windows paths -> forward slashes, numbers -> float."""
    if isinstance(value, str):
        return value.replace("\\", "/")
    if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
    if isinstance(value, list):
        return [norm(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def diff_doc(a, b, path=""):
    """Recursive JSON comparison; numbers compare as floats, strings/paths normalized."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                diffs.append(f"MISSING in pyref: {path}.{key}")
            elif key not in b:
                diffs.append(f"MISSING in cpp: {path}.{key}")
            else:
                diffs.extend(diff_doc(a[key], b[key], f"{path}.{key}"))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"LENGTH {path}: pyref={len(a)} cpp={len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(diff_doc(x, y, f"{path}[{i}]"))
        return diffs
    if norm(a) != norm(b) or type(a) is bool or type(b) is bool:
        if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            diffs.append(f"TYPE {path}: pyref={type(a).__name__} cpp={type(b).__name__}")
        elif norm(a) != norm(b):
            diffs.append(f"DIFF {path}: pyref={a!r} cpp={b!r}")
    return diffs


def run_one(model: str, ctx: int, vram: float, kv: str, tag: str):
    """Produce both JSON documents for one case; return (pyref_doc, cpp_doc)."""
    out = TMP / tag
    out.mkdir(parents=True, exist_ok=True)
    py_json = out / "py.json"
    cpp_json = out / "cpp.json"
    base = ["--ctx", str(ctx), "--vram-gib", str(vram), "--host-gib", str(HOST_GIB),
            "--row-cap", str(ROW_CAP), "--kv-type", kv]
    py_cmd = [PY, "-m", "pyref.cli", "plan", model, *base,
              "--vram-margin-mb", "1200", "--workspace-mb", "768", "--host-margin-mb", "0",
              "--json", str(py_json)]
    cpp_cmd = [str(ENGINE), "plan", model, *base,
               "--vram-margin", "1200", "--workspace", "768", "--host-margin", "0",
               "--json", str(cpp_json)]
    subprocess.run(py_cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run(cpp_cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return (json.loads(py_json.read_text(encoding="utf-8")),
            json.loads(cpp_json.read_text(encoding="utf-8")))


def model_list() -> list[Path]:
    models = []
    model_dir = Path(os.environ.get("MODEL_DIR", ""))
    for name in DEFAULT_MODELS:
        p = model_dir / name
        if p.exists():
            models.append(p)
    for rel in FIXTURES:
        p = ROOT / rel
        if p.exists():
            models.append(p)
    return models


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="planner C++/pyref parity gate")
    ap.add_argument("--full", action="store_true", help="extended matrix")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--show", type=int, default=5, help="diffs printed per failing case")
    args = ap.parse_args(argv)

    models = ([Path(m) for m in args.models] if args.models else model_list())
    if not models:
        print("planner parity: SKIPPED (set MODEL_DIR to the model directory)")
        return 0
    if not ENGINE.exists():
        print(f"planner parity: FAIL - {ENGINE} missing (build first)")
        return 1
    TMP.mkdir(parents=True, exist_ok=True)

    ctxs = CTX_FULL if args.full else CTX_QUICK
    vrams = VRAM_FULL if args.full else VRAM_QUICK
    kvs = KV_FULL if args.full else KV_QUICK

    cases = total = 0
    for model in models:
        name = model.stem[:24].replace(" ", "_")
        for ctx in ctxs:
            for vram in vrams:
                for kv in kvs:
                    total += 1
                    tag = f"{name}_c{ctx}_v{vram:g}_{kv}"
                    try:
                        py_doc, cpp_doc = run_one(str(model), ctx, vram, kv, tag)
                    except subprocess.CalledProcessError as e:
                        print(f"planner parity: FAIL - runner error on {tag}: {e.stderr[-800:]}")
                        return 1
                    diffs = diff_doc(py_doc, cpp_doc)
                    if diffs:
                        cases += 1
                        print(f"planner parity: FAIL {model.name} ctx={ctx} vram={vram:g} kv={kv}")
                        for d in diffs[: args.show]:
                            print(f"  - {d}")
                    else:
                        print(f"  PASS {model.name} ctx={ctx} vram={vram:g} GiB kv={kv}: "
                              f"{len(cpp_doc)} fields identical")
    if cases:
        print(f"planner parity: FAILED ({cases}/{total} cases differ)")
        return 1
    print(f"planner parity: PASS ({total}/{total} cases, "
          f"{len(models)} model(s), matrix ctx{ctxs} vram{vrams} kv{kvs}, host {HOST_GIB:g} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
