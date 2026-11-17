#!/usr/bin/env python
"""G0 gate: compare the C++ IR dump against the Python reference, field by field.

Usage:
  python tools/diff_ir.py <model.gguf> [--keep]

Exit code 0 = identical on every compared field (G0 pass).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(Path(sys.executable))
ENGINE = ROOT / "build" / "engine.exe"

SCALARS = ["arch", "container", "file_size", "n_layer", "hidden", "ffn", "vocab",
           "ctx_train", "mtp_layers"]
DICTS = ["attn", "moe", "ssm", "tokenizer", "bytes", "quant_mix"]
TENSOR_FIELDS = ["dtype", "role", "layer", "nbytes", "offset"]


def dump_py(model: Path, out: Path) -> None:
    subprocess.run([PY, "-m", "pyref.cli", "ir", str(model), "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)


def dump_cpp(model: Path, out: Path) -> None:
    subprocess.run([str(ENGINE), "ir", str(model), "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    model = Path(args.model)

    # NOTE: outputs must stay inside the workspace — native binaries launched from
    # here are denied writes elsewhere (sandbox artifact), which surfaces as EACCES.
    tmp = ROOT / "build" / "irdiff"
    tmp.mkdir(parents=True, exist_ok=True)
    py_json, cpp_json = tmp / "py.json", tmp / "cpp.json"
    dump_py(model, py_json)
    dump_cpp(model, cpp_json)
    a = json.loads(py_json.read_text(encoding="utf-8"))
    b = json.loads(cpp_json.read_text(encoding="utf-8"))

    diffs: list[str] = []
    for k in SCALARS:
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: py={a.get(k)!r} cpp={b.get(k)!r}")
    for k in DICTS:
        da, db = a.get(k, {}), b.get(k, {})
        for kk in sorted(set(da) | set(db)):
            if da.get(kk) != db.get(kk):
                diffs.append(f"{k}.{kk}: py={da.get(kk)!r} cpp={db.get(kk)!r}")

    ta = {t["name"]: t for t in a.get("tensors", [])}
    tb = {t["name"]: t for t in b.get("tensors", [])}
    if set(ta) != set(tb):
        only_a = sorted(set(ta) - set(tb))[:5]
        only_b = sorted(set(tb) - set(ta))[:5]
        diffs.append(f"tensor name sets differ: only_py={only_a} only_cpp={only_b}")
    for name in sorted(set(ta) & set(tb)):
        for f in TENSOR_FIELDS:
            if ta[name].get(f) != tb[name].get(f):
                diffs.append(f"tensor {name}.{f}: py={ta[name].get(f)!r} cpp={tb[name].get(f)!r}")
        if ta[name].get("shape") != tb[name].get("shape"):
            diffs.append(f"tensor {name}.shape: py={ta[name].get('shape')} cpp={tb[name].get('shape')}")

    la = a.get("layers", [])
    lb = b.get("layers", [])
    if len(la) != len(lb):
        diffs.append(f"layer count: py={len(la)} cpp={len(lb)}")
    else:
        for x, y in zip(la, lb):
            for f in ("index", "n_tensors", "n_expert_tensors", "kind"):
                if x.get(f) != y.get(f):
                    diffs.append(f"layer {x.get('index')}.{f}: py={x.get(f)} cpp={y.get(f)}")

    print(f"model      : {model.name}")
    print(f"tensors    : {len(ta)} (py) vs {len(tb)} (cpp)")
    print(f"layers     : {len(la)} (py) vs {len(lb)} (cpp)")
    if diffs:
        print(f"G0         : FAIL - {len(diffs)} field(s) differ")
        for d in diffs[:20]:
            print(f"  - {d}")
        return 1
    print("G0         : PASS - every compared field identical")
    if not args.keep:
        py_json.unlink(missing_ok=True)
        cpp_json.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
