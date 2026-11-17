# Third-party notices / 第三方声明

This repository contains **only original source code and documentation** written
for this project, except where explicitly noted below. It does **not** contain
model weights, CUDA redistribution binaries, or the upstream llama.cpp web UI
bundle (the latter is downloaded by the user at build/setup time).

本仓库只包含本项目自研源码与文档；**不包含模型权重、CUDA 二进制分发、llama.cpp
webui 前端 bundle**（后者由用户首次运行时自行下载）。

## 1. llama.cpp official web UI

- Project: **llama.cpp** (`ggml-org/llama.cpp`), MIT License.
- Component used: `tools/server/public/index.html.gz` from tag **b6000**
  (upstream commit `4762ad7316`, 2025-07-27).
- Our usage: `engine serve` serves this single-file chat UI at `/`.
  The file is **not committed** to this repository (≈4.4 MB after gunzip); fetch
  it with `python tools/fetch_webui.py` (or `bash tools/fetch_webui.sh`), which
  stores it under `build/webui/` (git-ignored).
- Upstream project: <https://github.com/ggml-org/llama.cpp>
- Upstream license: <https://github.com/ggml-org/llama.cpp/blob/master/LICENSE>

## 2. NVIDIA runtime libraries (not redistributed)

- The build loads **NVRTC** (NVIDIA Runtime Compilation) and **cuBLAS/cuBLASLt**
  from the CUDA runtime installed by the NVIDIA driver / CUDA Python wheels.
  No NVIDIA binaries are vendored in this repository.
- Your use of those libraries is governed by the NVIDIA CUDA Toolkit / driver
  license you obtained them under.

## 3. Build toolchain

- **ziglang** (`python -m ziglang c++`) is used as a portable C++17 compiler
  without requiring Visual Studio / administrator rights. Zig is MIT-licensed.
- Python is used only for build/verification scripts.

## 4. Model weights and model licenses

- **No model weights are included.** The engine is a model runtime: users bring
  their own GGUF files (or safetensors directories) and must comply with the
  license of each model they download.
- Mentioned test models (Qwen / Qwen3.5 / Qwen3.6, DeepSeek-Coder-V2, Gemma,
  gpt-oss, etc.) are property of their respective authors and are covered by
  their own licenses; this repository's MIT license does **not** apply to them.

## 5. Other open-source mentions

- The launcher uses standard Win32 APIs only (`user32`, `gdi32`, `dwmapi`,
  `uxtheme`, `winhttp`, `iphlpapi` loaded dynamically). No third-party GUI
  toolkit is bundled.
- JSON/tokenizer/GGUF parsing code in this repository is original code.

---

本项目以 **MIT License** 发布（见 `LICENSE`），版权行 `Copyright (c) 2026 Siselea`。
本声明不构成任何商业用途限制或授权扩张；请遵守上述第三方组件的各自许可证。
