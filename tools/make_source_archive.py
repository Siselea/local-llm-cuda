#!/usr/bin/env python3
"""Build a clean, buildable source archive for the engine.

Why this exists
---------------
v0.1.7 shipped with an ad-hoc source zip assembled by hand.  It could not be built
(build.sh / README / LICENSE / engine.cfg.example were missing) and it carried 50
editor backups (`*.pre*`, `*.v1`) plus `__pycache__` directories, which makes a
source release look careless and impossible to reproduce.  This tool packs by
whitelist, so a source archive is reproducible and never picks up debris.

Usage
-----
    python tools/make_source_archive.py                 # -> dist/<name>-src.zip
    python tools/make_source_archive.py --out D:/x.zip  # explicit path
    python tools/make_source_archive.py --list          # show what would be packed

The archive root is `local-llm-cuda-<version>/` (version from git tag/describe, or
"HEAD" when untagged), matching the binary release layout.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# Files everyone needs to build and understand the project (must exist).
REQUIRED_ROOT_FILES = [
    "build.sh",
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "THIRD-PARTY.md",
    "engine.cfg.example",
]
# Whole trees that belong in a source release.
INCLUDE_DIRS = ["src", "include", "tests", "tools", "pyref", "docs", ".github"]
# Never pack these, even inside included trees.
EXCLUDE_SUFFIXES = (".pre", ".v1", ".bak", ".orig", ".rej", ".pyc", ".pyo", ".swp")
EXCLUDE_DIR_NAMES = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
EXCLUDE_FILES = {"nohup.out", "launcher-ctx-debug.log"}


def git_version(root: Path) -> str:
    try:
        tag = subprocess.run(["git", "describe", "--tags", "--abbrev=0"],
                             cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
        if tag:
            return tag
    except Exception:
        pass
    return "HEAD"


def wanted(rel: Path) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_DIR_NAMES for p in parts):
        return False
    name = rel.name
    if name in EXCLUDE_FILES:
        return False
    # Editor backups: foo.cpp.pre, foo.cpp.pre_ui2, foo.cpp.pre-17ms, foo.v1, ...
    for suf in EXCLUDE_SUFFIXES:
        if name.endswith(suf):
            return False
    if ".pre" in name or name.endswith(".pre_ui") or ".pre-prefix" in name:
        return False
    return True


def collect(root: Path) -> list[Path]:
    files: list[Path] = []
    for f in REQUIRED_ROOT_FILES:
        p = root / f
        if not p.is_file():
            print(f"ERROR: required root file missing: {f}", file=sys.stderr)
            sys.exit(2)
        files.append(Path(f))
    for d in INCLUDE_DIRS:
        base = root / d
        if not base.exists():
            print(f"WARN: optional tree missing: {d}", file=sys.stderr)
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and wanted(p.relative_to(root)):
                files.append(p.relative_to(root))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output zip path (default: dist/<name>-src.zip)")
    ap.add_argument("--root", default=".", help="project root (default: cwd)")
    ap.add_argument("--list", action="store_true", help="only list the files that would be packed")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = collect(root)
    if args.list:
        for f in files:
            print(f)
        print(f"-- {len(files)} files", file=sys.stderr)
        return 0

    version = git_version(root)
    name = f"local-llm-cuda-{version}"
    out = Path(args.out) if args.out else root / "dist" / f"{name}-src.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in files:
            z.write(root / rel, arcname=f"{name}/{rel.as_posix()}")

    size = out.stat().st_size
    print(f"wrote {out}  ({len(files)} files, {size} bytes)")
    # Sanity: nothing embarrassing slipped in.
    bad = [n for n in zipfile.ZipFile(out).namelist()
           if ".pre" in Path(n).name or "__pycache__" in n or n.endswith(".v1")]
    if bad:
        print(f"ERROR: archive still contains debris: {bad[:5]}", file=sys.stderr)
        return 3
    print("clean: no backups, no __pycache__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
