# Chat UI (llama.cpp official web UI, MIT)

`engine serve` serves the official **llama.cpp single-file web UI** at
`http://127.0.0.1:<port>/`. The 4.4 MB HTML bundle is **not committed** to this
repository (it is third-party MIT content); fetch it once into the git-ignored
`build/webui/` directory:

```bash
python tools/fetch_webui.py        # or: bash tools/fetch_webui.sh
```

| item | value |
| --- | --- |
| source | `https://raw.githubusercontent.com/ggml-org/llama.cpp/b6000/tools/server/public/index.html.gz` |
| upstream tag | `b6000` (llama.cpp commit `4762ad7316`, 2025-07-27) |
| license | MIT (llama.cpp `tools/server/webui`) — see [`THIRD-PARTY.md`](../THIRD-PARTY.md) |
| size | 4,413,949 bytes uncompressed (single file; JS/CSS/fonts inlined) |
| `index.html.gz` | the original upstream gzip file, kept unmodified |

## Why a static file instead of a string in engine.exe

The bundle is ~4.4 MB: embedding it would grow `engine.exe` from ~2.3 MB to
~6.7 MB for no benefit. `engine serve` looks for the web UI **next to the exe**
(this is `build/webui/` in the normal layout) and serves the file verbatim; when
absent, a small built-in fallback page is served instead, so a copied exe never
shows a blank tab.

Search order (`engine serve`, first hit wins):

1. `$ENGINE_WEBUI_FILE`
2. `<exe-dir>/webui/index.html`  ← normal layout
3. `<exe-dir>/../webui/index.html`
4. `./webui/index.html`
5. compiled-in minimal fallback page

## Offline property (no CDN)

The only externally loaded resource in upstream was the optional "Python
interpreter" Web Worker script
`https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js`. In the version this
engine was developed against it was rewritten to `/webui/pyodide.js` (a path our
server does not provide), so the chat UI has **zero network dependencies**;
enabling the Python interpreter offline simply fails that optional feature.
Every other `http(s)://` string in the bundle is an XML namespace / documentation
link, not a fetch.

`tools/fetch_webui.py` downloads the **unmodified upstream gzip**, so if you use
a newer upstream tag the CDN reference may come back. If you need the offline
property, apply the one-line rewrite described above (or keep the b6000 bundle).

## Server compatibility notes

The web UI talks to `/props` and `/v1/chat/completions` (SSE). `src/serve.cpp`
implements:

* `/props` with `model_alias`, `modalities` (plus `model_path`, `n_ctx`, ...);
* `GET /slots` returns `[]` (llama.cpp compatibility stub);
* streamed final chunk carries llama.cpp's `prompt_n/prompt_ms/predicted_n/
  predicted_ms` aliases for t/s display;
* `max_tokens: -1` (web UI default) means "no cap beyond ctx" instead of 1 token.

To update the vendored UI later: download the same path from a newer upstream
tag, keep the file under `build/webui/` (git-ignored), and re-check the offline
rewrite if you enabled it.
