"""Scan a directory of real GGUF model files for loader compatibility.

Reads only the GGUF header/metadata/tensor table (no tensor data), then asks the
engine's parse+validate path (`engine info`) about the file.  This is the
"real user downloads a model" check: it finds architectures / quantisations /
tensor layouts the loader cannot handle.

Usage:
  python tools/audit_gguf_corpus.py [dirs...]
  python tools/audit_gguf_corpus.py --json build/audit/corpus.json
Outputs a table and a JSON file; exits non-zero when a model fails validation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "engine.exe"
DEFAULT_DIRS = [Path(os.environ.get("MODEL_DIR", ""))] if os.environ.get("MODEL_DIR") else []


def scan_header(path: Path) -> dict:
    """Header-only scan via the repo's python GGUF reader (no tensor data)."""
    sys.path.insert(0, str(ROOT))
    from pyref.gguf_parse import read_gguf  # type: ignore

    g = read_gguf(path, meta_only=True)
    tensors = g.tensors
    names = {t.name for t in tensors}
    arch = str(g.kv.get("general.architecture", ""))
    quant = {}
    for t in tensors:
        quant[t.type_name] = quant.get(t.type_name, 0) + 1
    return {
        "arch": arch,
        "n_tensors": len(tensors),
        "has_output": "output.weight" in names,
        "has_token_embd": "token_embd.weight" in names,
        "quant_types": sorted(quant),
        "has_experts": any("_exps" in n for n in names),
        "has_mtp": any(".nextn." in n for n in names),
        "n_layer_meta": int(g.kv.get(f"{arch}.block_count", 0) or 0),
        "n_kv": int(g.kv.get(f"{arch}.attention.head_count_kv", 0) or 0),
        "key_length": int(g.kv.get(f"{arch}.attention.key_length", 0) or 0),
        "n_head": int(g.kv.get(f"{arch}.attention.head_count", 0) or 0),
    }


def run_info(path: Path) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run([str(ENGINE), "info", str(path)], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=180,
                           stdin=subprocess.DEVNULL)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -999, "", "TIMEOUT>180s"
    dt = time.time() - t0
    validation = ""
    errors = []
    for line in out.splitlines():
        if line.startswith("VALIDATION"):
            validation = line.strip()
        elif line.strip().startswith("- "):
            errors.append(line.strip()[2:])
    arch = ""
    for line in out.splitlines():
        if line.startswith("arch"):
            arch = line.strip()
    return {"rc": rc, "seconds": round(dt, 2), "validation": validation,
            "errors": errors, "stderr": err.strip()[-400:], "arch_line": arch}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=[str(d) for d in DEFAULT_DIRS])
    ap.add_argument("--json", default=str(ROOT / "build" / "audit" / "corpus.json"))
    args = ap.parse_args()

    files: list[Path] = []
    for d in args.dirs:
        p = Path(d)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.gguf")))
    if not files:
        print("no *.gguf found")
        return 1

    rows = []
    print(f"{'model':52s} {'arch':12s} {'out':3s} {'OK':3s} {'val':12s} notes")
    print("-" * 110)
    for f in files:
        rec = {"path": str(f), "name": f.name}
        try:
            hdr = scan_header(f)
            rec.update(hdr)
        except Exception as e:  # noqa: BLE001 - report any header parse failure
            rec["header_error"] = f"{type(e).__name__}: {e}"
        info = run_info(f)
        rec["info"] = info
        rec["ok"] = info["validation"].startswith("VALIDATION : PASS")
        rows.append(rec)
        notes = ""
        if "header_error" in rec:
            notes += "HEADER:" + rec["header_error"][:60] + " "
        if not rec.get("has_output", True):
            notes += "NO-output.weight "
        if rec.get("has_mtp"):
            notes += "MTP "
        if rec.get("has_experts"):
            notes += "MoE "
        if info["errors"]:
            notes += "ERR:" + " | ".join(info["errors"])[:160]
        print(f"{f.name[:52]:52s} {str(rec.get('arch','?'))[:12]:12s} "
              f"{'y' if rec.get('has_output') else 'N':3s} "
              f"{'P' if rec['ok'] else 'F':3s} "
              f"{info['validation'][:12]:12s} {notes}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "models": rows}, indent=2, ensure_ascii=False))
    failed = [r for r in rows if not r["ok"]]
    missing_out = [r for r in rows if not r.get("has_output", True)]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} parse+validate PASS; "
          f"{len(failed)} FAIL; {len(missing_out)} without output.weight")
    print(f"wrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
