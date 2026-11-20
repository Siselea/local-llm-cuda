# llama.cpp webui + Win32 launcher + 模型扫描（2026-11-15）

本轮按用户中途修正后的需求完成三件事：

1. **对话页面直接套用 llama.cpp 官方 webui**（不再自己设计），补齐它依赖的端点/字段，
   浏览器打开 `http://127.0.0.1:7030/` 就是官方 `llama.cpp - chat` 界面；
2. **双击启动做成 Win32 GUI 小窗口**（`build/launcher.exe`，`SUBSYSTEM:WINDOWS`，无控制台），
   模型列表 / 端口 / ctx / KV / 启动 / 停止 / 打开浏览器 / 状态一屏搞定；
3. **模型扫描放进引擎**（`engine.exe models` 子命令，供 GUI、bat、CLI 共用），
   `启动.bat` 作为无 GUI 环境的控制台兜底。

推理/内核/调度代码（`runner.cpp` forward、`kernels.h`、`cpu_matvec.cpp`）、`src/mtp.cpp`
均未改动；`build.sh` 没有新增第三方依赖或编译选项，只新增源文件与一个 GUI 构建目标。

---

## 1. 对话页面 = llama.cpp 官方 webui（MIT，单文件）

| 项目 | 值 |
| --- | --- |
| 文件 | `build/webui/index.html`（4,413,949 B，单文件，JS/CSS/字体全内联） |
| 来源 | `https://raw.githubusercontent.com/ggml-org/llama.cpp/b6000/tools/server/public/index.html.gz` |
| 上游 tag / commit | `b6000` / `4762ad7316`，2025-07-27 |
| 许可 | MIT（llama.cpp 的 `tools/server/webui`） |
| 原始参考 | `build/webui/index.html.gz`（上游原样 gzip，未修改） |
| 说明 | `build/webui/README-WEBUI.md`（来源、离线改动、服务端兼容点） |

**为什么是静态文件而不是内嵌字符串**：bundle 有 4.4 MB，塞进 exe 会把 `engine.exe`
从 2.1 MB 变成 6.5 MB。`engine serve` 改为在 exe 同目录找 `webui/index.html` 静态返回
（正常布局即 `build/webui/index.html`）；找不到时回退到编进 exe 的极简页面
（约 14.5 KB，仍无 CDN、离线可用），拷贝单个 exe 也不会白屏。

搜索顺序：`$ENGINE_WEBUI_FILE` → `<exe 目录>/webui/index.html` →
`<exe 目录>/../webui/index.html` → `./webui/index.html` → 内置兜底页。

**离线核实（无 CDN）**：bundle 里唯一会被网络加载的引用是可选"Python 解释器"功能的
Web Worker `https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js`，已改为本地路径
`/webui/pyodide.js`（本服务不提供，该可选功能离线不可用，其余功能零网络依赖）。
`grep -c cdn.jsdelivr build/webui/index.html` = 0；`src="https://..."` 无命中；
其它 `http(s)://` 字符串都是 XML 命名空间 / 文档链接，不会被 fetch。

### 服务端兼容性改动（`src/serve.cpp`）

webui 实际只依赖 `/props` + `/v1/chat/completions`（SSE），但有几处细节必须对齐：

1. **`max_tokens: -1` = 不额外限制**。官方 webui 默认发 `max_tokens:-1`；原实现会
   `max(1, min(-1, ctx))` = 只生成 1 个 token。现在 `<=0` 视为不限制（仍受 ctx 约束）。
   实测：`curl ... -d '{"max_tokens":-1}'` → `completion_tokens: 9`（不再恒为 1）。
2. **SSE 末尾 chunk 带 llama.cpp 名字的 timings**：webui 从最终 chunk 读
   `prompt_n/prompt_ms/predicted_n/predicted_ms` 计算 t/s。现在流式最后一块同时包含
   llama.cpp 名字与引擎原有名字（`prefill_ms/decode_ms/tokens_per_second`），
   内置兜底页和老客户端都不受影响。
3. **`/props` 补 `model_alias`、`modalities`**（原有 `model_path`、`build_info`、
   `default_generation_settings`、`n_ctx`、`chat_template` 等已对齐）。
4. **`GET /slots` → `[]`**：llama.cpp 兼容占位（本引擎单 slot、串行；webui 实际不轮询它）。
5. **`GET /` 加 `Cache-Control: no-store`**：重编译 exe 后浏览器不会拿旧页面。

SSE 帧格式本来就兼容（`data: {...}\n\n` + `data: [DONE]`，JSON 单行；webui 的
`TextDecoderStream` 按行切分后 `startsWith("data:")` 再 `JSON.parse`）。

### 端到端验证

* 页面加载：标题 `🦙 llama.cpp - chat`，首页 Server Info 显示
  `Model: qwen2.5-0.5b-instruct-q4_k_m.gguf` / `Build: engine serve (c++17, WinSock2, ...)`
  （来自我们的 `/props`）——截图 `build/webui-01-home.png`。
* 真实发消息（浏览器里输入并点发送）→ 回复正常显示、会话写入 IndexedDB、
  刷新/路由 `#/chat/conv-...` 正常——截图 `build/webui-03-reply.png`，移动端
  `build/webui-04-mobile.png`；服务端日志 `build/serve-webui.log` 里对应
  `[req] POST /v1/chat/completions -> 200 prompt=12 gen=20`。
* 局限（如实）：该版 webui 只渲染 `content`（客户端解析 `<think>` 标签），
  而本引擎流式把思考段放在 `reasoning_content` 字段，所以对会思考的模型
  **思考过程在 webui 里不显示**、答案在思考结束后出现（对话本身完整可用）；
  内置兜底页则正常显示可折叠思考段。

---

## 2. 双击启动：`build/launcher.exe`（Win32 GUI，无控制台）

新增 `src/launcher.cpp`，用 `build.sh` 的 `build_gui()` 单独编译：

```bash
ziglang c++ -std=c++17 -O2 -Wl,--subsystem,windows src/launcher.cpp \
    -o build/launcher.exe -lwinhttp -ldwmapi -luxtheme -lgdi32 -luser32 -lshell32
```

产物 492,032 B，PE subsystem = 2（GUI；`engine.exe` 仍是 CONSOLE，脚本/CI 输出不变）。
零第三方依赖：纯 Win32 控件 + WinHTTP（`/health`、`/shutdown`）+ ShellExecute（打开浏览器）。

界面（深/浅色跟随系统，微软雅黑 UI，DWM 深色标题栏，自绘圆角按钮）：

* **模型列表**：来自 `engine.exe models --tsv`，每行 `大小 · 类型 · 文件名`（按大小降序），
  选中显示完整路径，双击列表项 = 启动；
* **参数行**：端口（默认 7030）、上下文（4096/8192/32768/131072/自动）、KV（f32/q8_0/q4_0）；
* **按钮**：重新扫描 / 启动 / 停止 / 打开浏览器；
* **状态区**：`模型加载中…（Ns）` → `运行中 http://127.0.0.1:7030/` → `已停止`，
  引擎 stdout/stderr 重定向到 `build/launcher-engine.log`（控制台隐藏）；
* **关闭窗口时**若引擎在跑 → 询问"停止并退出 / 保持运行并退出 / 取消"；
* 引擎由 `CreateProcessW(..., CREATE_NO_WINDOW)` 启动，不弹黑框；启动成功后自动开一次
  浏览器（`--no-browser` 可关，方便自动化）。
* **自动化参数**（快捷键/脚本也方便）：`launcher.exe [--autostart 5] [--port 7030]
  [--ctx 4096] [--no-browser]`。

自动化验收（`build/test_launcher.py`，用 Win32 消息驱动真实窗口、PrintWindow 截图）：

| 步骤 | 结果 |
| --- | --- |
| `launcher.exe --autostart 5 --port 7030` | 窗口出现，列表 5 个模型（21.84/11.85/6.58/5.89/0.49 GB） |
| 引擎健康检查 | 8.0 s 后 200 OK，状态 `运行中 http://127.0.0.1:7030/` |
| 截图 | `build/launcher-01-running.png` |
| 点「停止」（WM_COMMAND） | 端口释放，状态 `已停止`，截图 `build/launcher-02-stopped.png` |
| 运行中点关闭 | 找到确认对话框，选「是」→ 引擎停止、窗口退出码 0 |
| 无 `--no-browser` 启动 | 调用 `ShellExecuteW("open", url)`（无错误返回）；**本 agent 会话拉起的进程没有桌面访问权，观测不到浏览器窗口**（`ShellExecute` 打开 win.ini 也会 `SE_ERR_ACCESSDENIED`），所以这一项只验证到调用成功、未做到视觉截图——真实双击场景由 Explorer 启动，不受该限制 |

---

## 3. 模型扫描：`engine.exe models`（C++ 内置）

新增 `src/model_scan.cpp` + `include/engine/model_scan.h`；`main.cpp` 加 `models` 子命令，
`serve` 支持 `--pick N` / `--model auto`，交互式选择器也改用同一套扫描。

**扫描目录（按优先级，去重；`--model-dir` / `model_dir`/`ENGINE_MODEL_DIR` 可前置追加，`;` 分隔多个）**：

1. `<exe 目录>/models/`
2. `<exe 目录>/../models/`
3. `<MODEL_DIR>/`
4. `%USERPROFILE%/Downloads/`
5. `%USERPROFILE%/.cache/`
6. `--model-dir` 追加目录（排在内置目录之前，可覆盖发现顺序）

**规则**：只列 `*.gguf`（大小写不敏感）、只扫目录顶层；跳过 `mmproj-*` 视觉投影；
分片 `*-00001-of-000NN.gguf` 合并为一条（仅当第 1 片存在才收录，大小求和，缺失分片给 warning）；
按大小降序；显示 `大小(GB)/类型/文件名/目录`；类型是从文件名推断的**参考值**
（`MoE~`/`dense~`、参数规模、量化、`[MTP]`），不写死任何模型名——真实架构以加载时日志为准。
另支持 `ENGINE_MODEL_SCAN_DIRS=A;B` 完全替换内置目录（测试"找不到模型"用）。

```bash
./build/engine.exe models                 # 人读表格（编号/大小/类型/目录）
./build/engine.exe models --tsv          # launcher.exe 解析
./build/engine.exe models --plain        # N|path
./build/engine.exe models --json         # 脚本/测试
./build/engine.exe serve --pick 3        # 直接启动列表第 3 个
./build/engine.exe serve --model auto    # 最大模型
# 找不到模型时：列出所有搜索目录 + “把 .gguf 放到 X 目录 / 用 --model <path>” 提示，退出码 1
```

本机实际扫描结果（`build/models-tsv.txt`，节选）：

```
21.84 GB  MoE~ 35B Q4_K_M   Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
11.85 GB  dense~ Q5_K_M     DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf
6.58 GB   dense~ 9B Q5_K_M  Qwen3.5-9B-Q5_K_M.gguf
5.89 GB   dense~ 9B Q4_K_M  Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf   [MTP]
0.49 GB   dense~ 0.5B Q4_K_M qwen2.5-0.5b-instruct-q4_k_m.gguf
```

（DeepSeek-Coder-V2-Lite 实际是 MoE，但文件名没有 MoE 线索，显示 `dense~`——这正是
"文件名推断、仅供参考"的已知误差，真实架构在加载日志里是 `moe.n_expert=...`。）

---

## 4. 兜底：`启动.bat` / `stop.bat`（纯 ASCII + CRLF）

* `启动.bat`：定位 `engine.exe`（同目录 / `build/` / `../build/`）→ 端口占用检测
  （已在跑：打开浏览器 / 重启 / 退出）→ 调 `engine.exe models` 显示编号列表 → 选模型
  （回车=1，也可粘贴完整路径）→ 输入 ctx（回车=4096）→ `engine serve --pick N ... --open-browser`，
  引擎就绪后自己开浏览器；异常退出 `pause` 不闪退。也支持参数直接启动：
  `启动.bat 5 4096`。
* `stop.bat`：先 `POST /shutdown`，轮询端口释放；超时则 `taskkill /f /im engine.exe`。
* 两个 bat 全部 ASCII、CRLF；中文只由引擎以 UTF-8 输出（脚本在弹提示后 `chcp 65001`，
  避免 `chcp` 影响 `set /p` 的重定向输入）。
* 验证：`build/bat-nomodel.log`（无模型 → 清晰提示 + 退出码 1）、
  `build/bat-launch.log`（真实选择 1 号 35B → 分片放置决策 → `[serve] browser : ...`）。

---

## 5. 四套验证状态

见 `build/all-verify.log`：

| 验证 | 结果 |
| --- | --- |
| `tools/verify.sh` | **ALL GREEN**（`VERIFY_RC=0`，51 项） |
| `tools/smoke_models.sh` | `SMOKE_RC=0`（40716/185/2614/2614，四模型全 PASS） |
| `tools/gen_regression.sh` | `GEN_RC=0`（10/10：9b 5 组 + 35b 3 组 + coder/0.5b 均 PASS） |
| `tools/verify_generality.sh` | `GREEN`：`results: 26 passed, 0 failed`，`GENREG_RC=0` |

（并在 `build.sh` 中新增 `build_gui`，四套验证的 build 步骤会一并产出 `launcher.exe`；
完整日志 `build/all-verify.log`。）

---

## 6. 体积 / 未做

* `engine.exe`：2,134,528 → **2,195,968 B（+61 KB，+2.9%）**；增量 = 模型扫描 +
  内置兜底页（14.5 KB）+ webui 静态服务/兼容代码。4.4 MB 的 webui 不内嵌。
  `engine.exe` 仍是 CONSOLE 子系统（脚本/CI 输出不变），`launcher.exe` 是 GUI 子系统。
* `launcher.exe`：**492,032 B**（GUI 子系统，零第三方依赖）。
* 未做/局限：
  * webui 的思考段不显示（见 §1 局限）；官方 webui 本身不认 `reasoning_content` 字段；
  * launcher 不做开机自启/托盘；深/浅色仅在启动时读取系统设置，运行中不跟随切换；
  * 分片 GGUF 的合并条目可显示，但本机没有分片模型，**未验证**加载器对分片 GGUF 的支持；
  * 类型列是文件名启发式（如上 DeepSeek 误标 dense~）；
  * `启动.bat` 的编号选择依赖引擎模型顺序稳定（按大小降序，同大小按名字），
    两次扫描之间目录内容变化时会变；
  * **浏览器自动打开的观测限制**：本 agent 会话启动的进程无桌面访问权
    （`ShellExecute` 打开本地文件返回 `SE_ERR_ACCESSDENIED`），因此"自动弹出浏览器"
    只验证到 `ShellExecute` 调用成功且无 WARN，未能在本会话截图；Explorer 双击的真实
    场景不受此限。
* 改动前备份（仓库无 git）：`src/serve.cpp.pre_ui`、`src/main.cpp.pre_ui`、
  `include/engine/serve.h.pre_ui`、`build.sh.pre_ui`、`src/http_server.cpp.pre_ui`。

---

## 7. 变更文件清单

| 文件 | 变更 |
| --- | --- |
| `src/serve.cpp` | 优先从 exe 同目录静态提供 llama.cpp webui（无则内置兜底页，no-store）；SSE 末尾 chunk 增加 llama.cpp `timings` 字段；`max_tokens<=0` = 不限制；`/props` 加 `model_alias`/`modalities`；`GET /slots → []`；`serve --open-browser` |
| `include/engine/serve.h` | `Options::open_browser` |
| `src/model_scan.cpp`、`include/engine/model_scan.h` | 新增：模型扫描（目录优先级/分片合并/mmproj 跳过/大小排序/类型启发式/table·plain·tsv·json/no-model 提示/env 覆盖） |
| `src/main.cpp` | 新增 `models` 子命令（`--json/--plain/--tsv`）、`serve --pick N` / `--model auto`、`--open-browser`、无模型/非 tty 时打印扫描结果与提示；交互式选择器改用 model_scan |
| `src/launcher.cpp` | 新增：Win32 GUI launcher（GUI 子系统，492 KB，零三方依赖） |
| `build.sh` | `SRC` 加 `src/model_scan.cpp`；新增 `build_gui()` 产出 `build/launcher.exe` |
| `启动.bat`、`stop.bat` | 新增：控制台兜底启动 / 停止（纯 ASCII、CRLF） |
| `build/webui/index.html`（+ `.gz`、`README-WEBUI.md`） | 新增：llama.cpp b6000 官方单文件 webui（MIT），仅把 pyodide CDN 改成不可达本地路径 |
| `docs/SERVE.md`、`docs/CONFIG.md`、`README.md` | 文档更新（webui/launcher/models/默认 max_tokens/`model_dir` 语义） |

---

## 8. 证据路径

| 内容 | 路径 |
| --- | --- |
| llama.cpp webui 首页（Server Info） | `build/webui-01-home.png` |
| llama.cpp webui 对话回复 | `build/webui-03-reply.png` |
| llama.cpp webui 移动端 | `build/webui-04-mobile.png` |
| webui 服务端日志（`[serve] webui : ...` + 请求） | `build/serve-webui.log` |
| launcher 运行中（模型列表/参数/状态） | `build/launcher-01-running.png` |
| launcher 已停止 | `build/launcher-02-stopped.png` |
| launcher 自动化证据 JSON | `build/launcher-evidence.json` |
| 引擎日志（launcher 启动，控制台隐藏） | `build/launcher-engine.log` |
| 模型列表（TSV/表格/JSON/plain） | `build/models-tsv.txt`、`build/models-table.txt`、`build/models.json`、`build/models-plain.txt` |
| 端点兼容性证据（props/slots/SSE timings/max_tokens=-1/页面 4,413,949 B/无 CDN） | `build/webui-endpoints.txt` |
| 自动开浏览器窗口探测（本会话受限，见 §6） | `build/browser-open-check.txt` |
| 无模型兜底启动 | `build/bat-nomodel.log` |
| 真实 35B 兜底启动（含扫描与自动开浏览器调用） | `build/bat-launch.log` |
| 兜底 bat 管道输入启动（选 5 号 / ctx 2048 → 引擎健康 → 退出码 0） | `build/bat-launch-piped.log` |
| `stop.bat` 停止验证（POST /shutdown → 端口释放） | `build/bat-stop.log` |
| 四套验证完整日志 | `build/all-verify.log` |
| 内置兜底页（webui 缺失时；14,478 B，思考折叠等） | `build/ui9b-02-streaming.png`、`build/ui9b-03-reply.png`、`build/ui9b-03b-thinking-open.png`、`build/ui9b-04-settings.png`、`build/ui9b-05-mobile.png` |
| 兜底页切换验证日志（`webui/index.html not found` → 14,478 B 极简页） | `build/serve-fallback-test.log` |

---

## 9. UI round 3（2026-11-15 23:0x）：用户试用反馈的 3 个 bug + 默认模型

用户实机试用 launcher 后报告：滚动闪烁 / 模型下拉选不中 / KV 下拉选不中。修复（只动 `src/launcher.cpp` + `build/test_launcher.py`，未碰引擎/serve.cpp）：

1. **原生下拉**：删掉自绘 combo shell + `subclass_combo` + "打开时改窗口高度"的旧逻辑；模型/KV/高级枚举下拉全部是原生 `COMBOBOX`（`CBS_DROPDOWNLIST` + owner-draw 仅上色），鼠标/键盘/焦点由 Windows 原生处理；下拉高度在 `CBN_DROPDOWN`/`CBN_CLOSEUP` 调整。实测鼠标点开模型下拉并点第 6 项（qwen2.5-0.5B）：`CB_GETCURSEL=5`，元数据行刷新为 `hidden 896 / 权重 463 MB`；KV 下拉点 q8_0：`CB_GETCURSEL=1`，KV 估算 `96 MB → 26 MB`。
2. **滚动闪烁**：主窗口 `WS_EX_COMPOSITED`（整页双缓冲）+ `WS_CLIPCHILDREN` + `WM_ERASEBKGND` 返回 1；滚动帧连续抓取 SHA-256 相同（`5650d908…` × 3，滚动后 `8c3325b1…` 不同），无残影/闪烁。
3. **默认模型**：不再默认「自动（最大模型）」=35B；首次启动选 9B（6.58 GB，44.7 tok/s，首 token 168 ms），下拉里 9B 标「（推荐，日常对话最快）」、35B 标「（最强，但慢）」、其它中型标「（中等）」。
4. 「打开对话页」tooltip 补充：首次打开加载 llama.cpp 界面、发消息后首字 0.2–0.5 s 延迟正常。

`build/test_launcher.py` 从 48 项增加到 **57 项**：新增 `native_combo`、`composited_window`、`default_model_is_9b`、`model_labels`、`open_page_tooltip_hint`、`model_dropdown_mouse_select`、`kv_dropdown_mouse_select`、`scroll_frames_stable`、`scroll_content_moves`；实机 `57/57 PASS`（0.5B 加载 7.8 s，起停都过）。

截图：`build/launcher-06-dropdown.png`（原生下拉展开、9B 推荐项）、`build/launcher-07-model-picked.png`（鼠标选 0.5B + q8_0，KV 96→26 MB）、`build/launcher-08-scroll.png`（滚动后）、`build/launcher-01-default.png`（默认 9B）。证据：`build/launcher-evidence.json`（checks=57，passed=true）。

---

## 10. UI round 4（2026-11-16 晚）：9B 稠密模型「GPU 专家层数/自动」看起来卡死

用户报：选 9B 后 GPU 专家层数调不了、自动取消不了。根因：9B 是稠密 hybrid（无 MoE 专家层），本就不能分层；旧实现只是把控件 EnableWindow(FALSE)，没解释，且「自动」视觉上仍是勾选，像坏了。

修复（只动 `src/launcher.cpp`）：
- 稠密模型（`moe.n_expert==0` 或 `expert_per_layer` 全 0）：滑块 + 「自动」+「显存自动加层 vram_auto_layers」全部禁用；「自动」强制显示为 OFF；标签置灰；说明行改为「该模型是稠密模型（无 MoE 专家层），全部权重都在显存，无需 GPU/CPU 分层」；估算行改为「显存占用 = 全部权重 + KV + 工作区 ≈ X GB / 可用 Y GB」。
- MoE 模型（35B/Coder）：恢复全部控件、自动开关与层数估算；切回时显式层数夹到 `[0, n_moe]`，切 9B 时回 0（合法）。
- 顺带：启动前若端口上已有别家引擎（如另一个 agent 的 serve），`on_timer` 会对比 `/props.model_path` 与本窗口启动的模型；不匹配则终止子进程并显示「端口已被别的引擎占用…」，避免误报「运行中」。
- 新增自动化消息 `WM_APP+40`（读内部 toggle 状态）供测试断言。

验收（真实鼠标/滚轮，非 SendMessage 假点击）：`build/test_launcher_realuser.py` 10/10 PASS；`build/test_launcher.py` 64/64 PASS。真实用户剧本截图：`build/launcher-realuser-01-first-look.png` … `-09-stopped.png`（8 步 9 张）；9B 禁用态/35B 可用态：`build/launcher-09-dense-gpu-disabled.png`、`build/launcher-10-moe-gpu-enabled.png`。

真实操作中发现并记录的问题（未改代码，供后续）：
1. 鼠标滚轮消息发给「焦点控件」：若模型下拉仍有焦点，滚轮可能改模型而不是滚页面（脚本用 AttachThreadInput 把焦点交还主窗口后通过）。真实用户一般会先点空白处，但滚轮前焦点在下拉上时会被误改，建议 launcher 在窗口层拦截 WM_MOUSEWHEEL 或对闭合下拉屏蔽滚轮。
2. 共享 GPU 时 `/props` 偶发超时但 `/health` 200（server-mode 正在改 serve）；测试改为任一接口返回本窗口模型名即算可达，避免把引擎侧抖动记为 launcher 失败。

---

## 11. UI round 5（2026-11-16 夜）：滚轮焦点坑真 bug 修复

真实用户剧本发现的真 bug：`WM_MOUSEWHEEL` 发给**焦点控件**；用户点完模型下拉后焦点仍停在下拉上，此时滚滚轮会**改模型**而不是滚页面。

修复（只动 `src/launcher.cpp`，双保险）：
- **主窗口层**：`WM_MOUSEWHEEL` 统一滚动页面；`WM_MOUSEHWHEEL` 无横向内容直接吞掉。
- **COMBOBOX 子类化**（`combo_wheel_proc`，用 `SetWindowLongPtrW`，不引入新库）：
  - 闭合态（`CB_GETDROPPEDSTATE==FALSE`）：吞掉滚轮并把 `WM_MOUSEWHEEL` 转发给主窗口滚动页面，**绝不改选中**；
  - 展开态：把滚轮转给 `ComboLBox` 列表（滚动列表），**绝不改选中**。
- **ComboLBox 子类化**（`combo_list_wheel_proc`，CBN_DROPDOWN 时安装）：滚轮只滚列表（`LB_GETTOPINDEX`+`LB_SETTOPINDEX`），选中只由用户点击/回车提交，Esc 关闭恢复原选中。

真实鼠标验收（`build/test_launcher_realuser.py` 新增第 8 步，共 **13 项断言 = 10/10 → 13/13 PASS**）：
- 8a 焦点停在闭合下拉上滚滚轮：焦点类 `ComboBox`，模型 sel 3→3 不变，页面 scroll 0→180；
- 8b 展开下拉滚滚轮：悬停高亮 sel=1（原生 hover 行为），滚轮后仍 sel=1（滚轮未再改选中），页面 scroll 0→0，列表展开=1、top 0→0（6 项无溢出）；
- 8c Esc 关闭：sel 回到 3，未提交悬停高亮。
截图：`build/launcher-realuser-10-wheel-closed.png`、`build/launcher-realuser-11-wheel-open.png`；`build/test_launcher.py` 仍 **64/64 PASS**。`launcher.exe` **662,016 B**。

---

## 12. UI round 6（2026-11-16 深夜）：已有实例守卫 · toggle 复位 · 全盘模型扫描

### 12.1 已有实例检测（用户连点两次启动导致双 engine 抢显存）
- `engine_process_pids()`（Toolhelp 枚举 engine.exe，排除自己子进程）+ `tcp_listener_pid()`（动态加载 `iphlpapi!GetExtendedTcpTable`，不新增链接库）双路检测。
- 点「启动引擎」前若已有 engine/端口被占：弹框「检测到引擎已在运行（PID xxx，模型 xxx）。是=停止并重启，否=取消」；选是则终止旧 engine、等端口释放再启动。同一 launcher 重复点启动显示「引擎已在运行/启动中，已忽略重复启动」，绝不静默起第二个。
- 测试：`existing_engine_guard`（外部 engine 运行时点击 → 状态「检测到引擎已在运行…已取消启动」，engine 进程数=1）、`double_start_single_engine`（运行中重复点击 → 「已忽略重复启动」，进程数=1）。

### 12.2 测试后 toggle/cfg 复位
- `test_launcher.py` 收尾把 vram_auto/ssd 关闭，dev_route/layer_graph/cpu_int8/thinking/gpu_auto 恢复默认，KV=f32，清高级 override，并删除 `launcher-engine.cfg`；新增 `toggles_reset_after_test`、`launcher_cfg_clean`。
- 新增 `startup_resets_toggles_from_schema`：把 vram_auto 留 ON 后关闭再启 → 新实例从 `engine serve --print-config --json` 重取默认值，toggle=OFF、args 无「VRAM 自动加层」。
- 测试结束后无 launcher/engine 进程、无 cfg、GPU 空闲。

### 12.3 全盘模型扫描（扫整个系统的固定盘/移动盘）
- `engine models`：默认读 `engine-models-cache.json`（exe 同目录）秒回；`--refresh` 全盘扫并写缓存（含每盘 root mtime/目录数/耗时/scan_seconds）；`--scan-depth N`（默认 7）、`--min-size-mb N`（默认 100）、`--progress`（stderr，每发现/每 1.5s 或 1万目录）、`--no-scan`（只用 --model-dir）、`--model-dir` 前置优先；跳过 `DRIVE_REMOTE/CDROM/RAMDISK`、Windows/Program Files*/ProgramData/$RECYCLE.BIN/System Volume Information/node_modules/.git/AppData\Local\Temp/Windows\Installer/Recovery、reparse point；`FindFirstFileExW`(BASIC+LARGE_FETCH)；3 线程 BELOW_NORMAL；宽字符 API + UTF-8 JSON（中文路径正确）。
- 实测（RTX3060 机器）：首扫冷缓存 **76.08s**（C: 58675 目录 / D: 3049 / U: 12368，共 74092 目录）；第二次热缓存 **5.54s**；读缓存 **0.01s**。
- **扫到 18 个模型 = D: 5 + U: 13 + C: 0**：
  - D: `<MODEL_DIR>`：35B-A3B-Q4(21.84G)、Coder-V2-Lite(11.85G)、Qwen3.5-9B-Q5(6.58G)、Qwythos-9B-MTP(5.89G)、qwen2.5-0.5b(0.49G)
  - U: `<MODEL_DIR>`：Qwen3.6-35B-A3B-Q8_0(37.8G)、35B-A3B-Q4(21.84G)、Gemma4-26B-A4B(16.8G)、Qwen3.8-27B(16.55G)、Coder-V2-Lite(11.85G)、gpt-oss-20b(11.62G)、R1-Distill-14B(8.99G)、Qwythos-9B-Q6(7.36G)、gemma-4-12b(7.12G)、Qwen3.5-9B(6.58G)、Qwythos-9B-MTP(5.89G)、DeepSeek-V4-Pro-9B-MTP(5.78G)、Qwen3.5-9B-DeepSeek-V4-Flash(5.63G)
- Launcher：下拉标签 `[D:] 21.84 GB · MoE · name`；启动先读缓存秒出并后台异步 `--refresh --progress`（进度「扫描中 X/Y 盘 · 已找到 N 个模型」）；「重新扫描（后台）」按钮；完成后保持选中项。
- **超大模型警告**（不写死模型名）：`size > (VRAM总量+内存总量)*0.8` → 标签追加 `（⚠ 37.8G 超过显存+内存 80%，会非常慢）`；U: 37.8GB Q8_0 正确带警告。
- 测试新增：`scan_progress_reported`、`scan_cache_written`、`cache_hit_read`、`no_scan_only_model_dir`、`drive_labels`、`refresh_button_background_scan`、`oversized_model_warning`。

### 12.4 最终结果
- `build/test_launcher.py` **76/76 PASS**（`build/test-launcher-ui10.log`）。
- `build/test_launcher_realuser.py` **13/13 PASS**（`build/test-launcher-realuser10.log`，真实鼠标/滚轮/拖拽；截图 `launcher-realuser-01..11-*.png`）。
- 截图：`launcher-12-fullscan-default.png`、`launcher-13-drives-dropdown.png`、`launcher-14-scanning.png`、`launcher-15-scan-done.png`、`launcher-16-dropdown-full-warning.png`（18 项含 ⚠ 37.8G）。
- 最终 `build/launcher.exe` **705,024 B**、`build/engine.exe` 2,298,368 B（含全盘扫描）；未改 `src/serve.cpp`/内核/mtp.cpp。

---

## 13. 线程自动 + SMT 成对绑核（2026-11-17，配合 prio-quiet 引擎默认）

背景：引擎新默认改为 `win_affinity=2`（SMT 兄弟对绑核）+ `threads=0/auto`（本机 6C/12T → 12 线程），
实测 CPU 专家层快 8–12%；但旧 launcher 默认把 UI 的 9 线程显式传给引擎，覆盖了新默认。

改动（只动 `src/launcher.cpp`）：
- **CPU 线程数控件改为下拉**：`自动（跟随引擎默认）`（默认选中，**不传 `--threads`**）+ 显式 1..2×logical；
  提示「自动 = 不传 --threads，跟随引擎默认（本机 6C/12T → 12 线程）」。
- `default_thread_count()` 在 SMT 拓扑下改为 **2×物理核（12）**，无 SMT 回退 1.5×；仅作为 UI 展示/显式候选。
- 性能开关组新增 **「SMT 成对绑核 win_affinity」**（默认开 → `--win-affinity 2`；关 → `--win-affinity 0`），
  tooltip：「按 SMT 兄弟对绑核，实测 CPU 专家层快 8–12%；关闭用于排查/对比」。
- `win_affinity` 加入 curated 主面板 key（高级面板不再重复），schema 覆盖仍 100%。
- 启动摘要显示 `线程 自动 · SMT绑核 开`。

验证：
- UI 实测（不启引擎）：线程下拉 = `自动（跟随引擎默认）`，提示含 12 线程；SMT 开关默认开；
  `launcher-effective-args.txt` 摘要「线程 自动 · SMT绑核 开」。
- `test_launcher.py` 新增 4 项：`threads_default_auto`、`win_affinity_toggle_default_on`、
  `engine_threads_auto_cli`（日志命令行**无 `--threads`**）、`win_affinity_default_on_cli`（`--win-affinity 2`）；
  总检查 76→**80 项**。
- 引擎侧证据：`src/cpu_matvec.cpp` Pool：`win_affinity=2` + SMT2 → `nthreads=2×physical=12`，
  CPU 专家首次调用打印 `[cpu] pool threads: 12`（0.5B dense 不走 CPU pool；35B CPU 专家路径可见）。
- 截图：`build/launcher-threads-affinity.png`（1066×720，30 KB，已裁掉底部私有路径区）。
- **最终 test_launcher：80/80 PASS**（2026-11-17 23:54，`build/test-launcher-affinity7.log`），新增 4 项全过：
  `threads_default_auto=True`、`win_affinity_toggle_default_on=True`、
  `engine_threads_auto_cli=True`（0.5B 启动命令行**无 `--threads`**）、
  `win_affinity_default_on_cli=True`（命令行 `--win-affinity 2`）。
- 引擎线程数最终证据（CPU-only 微基准，源码同 build/engine.exe；未传 `--threads`）：
  `[cpu] sched: win_priority=1 worker_rel=2 affinity=2 cpus=0 pairs=6 nthreads=12` +
  `[cpu] pool threads: 12`；`ENGINE_WIN_AFFINITY=0` 对照 = `[cpu] pool threads: 9`。
