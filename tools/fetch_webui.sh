#!/usr/bin/env bash
# Fetch the official llama.cpp single-file web UI (MIT) for `engine serve /`.
# Thin wrapper around tools/fetch_webui.py (Python 3.8+, standard library only).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-${PYTHON:-python}}"
exec "$PY" tools/fetch_webui.py "$@"
