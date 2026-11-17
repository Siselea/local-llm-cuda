#!/usr/bin/env python3
"""Check that local links in README.md resolve to files/dirs in this repository.

External http(s)/mailto links and pure anchors are ignored (they are covered by
GitHub rendering).  Used by the `sanity` CI job so a moved file / renamed
screenshot does not leave a dead relative link in the README.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["README.md"]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def is_local(link: str) -> bool:
    low = link.lower()
    return not (
        low.startswith(("http://", "https://", "mailto:", "ftp:", "ssh:", "//", "#"))
        or low.startswith("data:")
    )


def main() -> int:
    missing: list[tuple[str, str, str]] = []
    checked = 0
    for rel in TARGETS:
        path = ROOT / rel
        if not path.is_file():
            print(f"missing target file: {rel}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            link = m.group(1).strip()
            if not is_local(link):
                continue
            # strip fragment / query; decode %20 etc. conservatively
            local = link.split("#", 1)[0].split("?", 1)[0]
            if not local:
                continue
            local = local.replace("%20", " ")
            target = (path.parent / local).resolve()
            try:
                target.relative_to(ROOT)
            except ValueError:
                missing.append((rel, link, "outside repository"))
                continue
            checked += 1
            if not target.exists():
                missing.append((rel, link, str(target)))
    print(f"README local links checked: {checked}")
    if missing:
        print("BROKEN local links:", file=sys.stderr)
        for rel, link, why in missing:
            print(f"  {rel}: {link}  -> {why}", file=sys.stderr)
        return 1
    print("all local README links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
