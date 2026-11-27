#!/usr/bin/env python3
"""Fetch the official llama.cpp single-file web UI (MIT) for `engine serve /`.

Tries a couple of llama.cpp tags (b6000 first — the version this engine was
developed against) from raw.githubusercontent.com, saves the original
`index.html.gz` plus its uncompressed `index.html` under `build/webui/`.

The bundle is git-ignored: we do not redistribute it in this repository.

Usage:
    python tools/fetch_webui.py [--tag b6000] [--out build/webui]

Exit codes: 0 = one tag downloaded and extracted; 1 = every download failed
(see the printed hints: network/proxy or download it manually).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# fallback order: the tag this engine was developed/tested against, then a
# stable later tag in case upstream moves/deletes the old commit.
TAGS = ["b6000", "b6001", "b6005"]
URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/tools/server/public/index.html.gz"
USER_AGENT = "engine-fetch-webui/1.0 (+https://github.com/Siselea)"


def download(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(url, resp.status, "unexpected status", resp.headers, None)
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="download llama.cpp official webui (MIT)")
    ap.add_argument("--tag", default="", help="llama.cpp tag, e.g. b6000 (default: try %s)"
                                           % TAGS[0])
    ap.add_argument("--out", default="build/webui", help="output dir (default: build/webui)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_path = out_dir / "index.html.gz"
    html_path = out_dir / "index.html"

    tags = [args.tag] if args.tag else TAGS
    last_error = ""
    for tag in tags:
        url = URL.format(tag=tag)
        print(f"fetching llama.cpp webui ({tag}) ...\n  {url}")
        try:
            data = download(url)
        except Exception as exc:  # noqa: BLE001 - report any network/proxy failure
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {last_error}")
            continue
        if len(data) < 1_000_000:
            last_error = f"downloaded file looks too small ({len(data)} bytes)"
            print(f"  FAILED: {last_error}")
            continue
        try:
            html = gzip.decompress(data)
        except OSError as exc:
            last_error = f"not a valid gzip bundle: {exc}"
            print(f"  FAILED: {last_error}")
            continue
        # Apply the documented offline edit (see docs/WEBUI.md / THIRD-PARTY.md):
        # the Python interpreter Web Worker was the only CDN dependency; point it
        # at a local path so the bundled page has zero network dependencies.
        html = re.sub(rb'https://cdn\.jsdelivr\.net/pyodide/[^"\']+/pyodide\.js',
                      b'/webui/pyodide.js', html)
        gz_path.write_bytes(data)
        html_path.write_bytes(html)
        sha = hashlib.sha256(data).hexdigest()[:16]
        print(f"  OK: {len(data):,} bytes gz -> {len(html):,} bytes html (sha256:{sha}...)")
        print(f"  saved: {gz_path}")
        print(f"  saved: {html_path}")
        print("  license: llama.cpp tools/server/webui (MIT) — see THIRD-PARTY.md")
        return 0

    print("\nCould not download the webui.", file=sys.stderr)
    if last_error:
        print(f"last error: {last_error}", file=sys.stderr)
    print(
        "Hints:\n"
        "  * check network/proxy (raw.githubusercontent.com reachable)\n"
        "  * manual download: https://raw.githubusercontent.com/ggml-org/llama.cpp/"
        "b6000/tools/server/public/index.html.gz\n"
        "  * put it at build/webui/index.html.gz and gunzip it to build/webui/index.html\n"
        "  * `engine serve` also works without it (small built-in fallback page).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
