# Contributing to local-llm-cuda

感谢你愿意参与！本项目是一个**从零实现的 LLM 推理引擎**（C++17 + CUDA，Windows + 消费级
NVIDIA GPU），目标是可读、可复现、可验证，而不是堆功能。提交前请读完本文件。

> English speakers: the project is maintained in Chinese; English issues/PRs are welcome.
> Please include the same information (environment, model, exact command, logs).

---

## 1. 构建

环境：**Windows 11 x64**、NVIDIA 驱动、Python 3.10+（构建需要，运行不需要 Python）。

```bash
python -m pip install ziglang
python -m pip install nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
bash build.sh
```

- 不要求 MSVC / CUDA Toolkit / 管理员权限。
- 产物在 `build/`（git-ignored）：`engine.exe`、`launcher.exe`、各 `test_*.exe`。
- 如果 Python 不在 PATH：`PY=<python路径> bash build.sh`。
- **不要提交** `build/`、`.exe`、模型权重、日志、`*.pre*`/`*.v1` 历史快照。

### 1.1 共享 `build/` 只由「干净 HEAD worktree」构建（团队规则，2026-11-22）

`build/` 是所有验证脚本共用的产物目录。**绝不要在含未提交 WIP 的工作区直接跑 `bash build.sh`**
——它会把实验性 WIP 编进共享的 `engine.exe`/`test_gen.exe`/…，其他人拿到的就是“坏的共享二进制”
（实际踩过的坑：Gemma4 WIP 编进 test_gen 后 `gen_regression` 9/10 FAIL、`got` 全空，
qwen35 报 `missing tensor blk.N.ffn_norm.weight`；而干净 HEAD 构建 10/10 全绿）。

规则：

1. **共享 `build/` 的二进制只从已提交 HEAD 的独立 worktree 构建**：

   ```bash
   cd <repo>
   git worktree add ../engine_clean HEAD
   cd ../engine_clean
   PY=<python> bash build.sh
   # 确认主仓库没有验证进程占用后，复制回共享 build/
   cp build/*.exe ../<repo>/build/
   ```

2. **含 WIP 的实验编译**去自己的 worktree（如 `../engine_arch`），产物**不要**覆盖共享 `build/`；
   需要别人用你的实验二进制时，明确写清 worktree 路径/二进制名。
3. 覆盖共享 `build/` 前先看 `tasklist | grep -E "test_gen|engine.exe"` 与 `nvidia-smi`，
   确认没有验证进程在用（Windows 上正在运行的 exe 无法覆盖，会报 `Device or resource busy`）。
4. `bash tools/check_clean_build.sh` 用于检查“当前工作区是否 dirty”与“build/ 是否来自 HEAD”
   （该脚本由 server-mode 维护；没有脚本时按本节步骤人工检查）。

## 2. 跑验证（PR 必须全过）

五套引擎验证（需要自备测试模型，设置 `MODEL_DIR` 指向模型目录）：

```bash
export MODEL_DIR=<你的模型目录>
bash tools/verify.sh              # 单元测试 + G0 IR 对拍 + M2 plan 对拍 + 生成一致性 → ALL GREEN
bash tools/smoke_models.sh        # 四模型冒烟（0.5B / Coder / 9B / 35B）
bash tools/gen_regression.sh      # 多 token 生成回归，与 llama.cpp greedy 逐 token 一致
bash tools/verify_generality.sh   # 配置优先级 / 自适应放置 / 优雅失败 / 校准缓存
bash tools/verify_serve.sh        # HTTP 服务端到端（默认 0.5B@7031）：API/SSE/优雅退出
```

Launcher 相关改动（`src/launcher.cpp`、`src/model_scan.cpp`）还应跑：

```bash
MODEL_DIR=<你的模型目录> python tests/test_launcher.py   # 80 项 GUI/args/启动停止验收（需 GPU）
bash tools/verify_serve.sh        # HTTP 服务端到端
bash tools/webui_e2e.py ...       # 真实浏览器（可选）
```

`tests/test_serve_guard.cpp` 的 repeat-guard 单测已包含在 `verify.sh` 中（构建后自动执行）。
PR 描述里贴出五条命令的结果摘要（PASS/FAIL 行），不要只写“本地跑过了”。

## 3. 代码风格

- **C++17**，无第三方 C++ 依赖（不引入新的 header-only 库/构建系统）。
- 缩进 2 空格；文件内匿名 namespace 组织实现细节；遵循现有命名（snake_case 函数/变量，
  PascalCase 类型）。
- CUDA kernel **必须通过 NVRTC 运行时编译**（字符串放在 `include/engine/kernels.h`，
  由 `src/cuda.cpp` 编译），**不要**新增 `.cu` 文件 / nvcc 依赖。
- 量化/放置/内存账本改动必须在 PR 中给出对拍或实测数据，并说明对其它模型的影响。
- **禁止模型名特判**：不允许按模型文件名/名字硬编码行为；能力应从 GGUF/配置元数据推导。
- 新配置项必须走 `src/config.cpp` schema（cli/config/env/default + `--print-config --json`），
  让 launcher 自动生成控件。
- 文档/注释用中文或英文均可，保持与周围一致。

## 4. 提交 PR

1. Fork + feature branch，保持提交聚焦（一个 PR 一件事）。
2. 填写 `.github/pull_request_template.md`（改了什么、怎么验证、对性能/显存的影响）。
3. CI 必须通过：`build`（Windows 全量 `bash build.sh`）、`sanity`、`no-private-paths`。
4. 性能类 PR 必须附**可复现命令**与 `ENGINE_KPROF=1` 的逐内核数据（或前后对比表）。
5. 不要提交模型文件、生成的大日志、本机绝对路径（CI 会检查开发机绝对路径/用户名是否泄漏）。
6. 报告 bug 时请附：
   - Windows 版本、GPU 型号/显存、驱动版本；
   - 模型名称/量化/大小（**不要上传模型权重本身**）；
   - 复现命令（`engine.exe serve ...` / `launcher.exe` 操作步骤）；
   - `engine.exe --print-config --json` 输出；
   - `launcher-engine.log` / `serve` 控制台日志尾部；
   - 性能问题附 `ENGINE_KPROF=1` 与温度/占用，避免把后台 GPU 占用误判为回退。

## 5. 报告安全/隐私问题

不要在公开 issue 贴 API key、token、完整用户路径或模型权重。先发最小复现（去掉这些信息）。

## 6. 开发报告

设计/验证报告放在 `docs/reports/`（索引见 `docs/reports/INDEX.md`）。新增报告请在同目录添加
一句话说明到 INDEX。生成物（`*.log`、`.csv`、截图）不提交。

感谢贡献 🙏
