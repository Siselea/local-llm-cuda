# 任意上下文长度 + 极端情况矩阵（CTX-AND-EXTREMES）

日期：2026-11-22 · 作者：feature-polish（lead 指派）
对象：`engine` 的 ctx 语义（CLI / config / /props / launcher）与极端条件下的性能/健壮性。
GPU：RTX 3060 12 GB · RAM 32 GB · Windows 11 VM（所有数字为本机实测，写清测量口径）。

> 本文分两部分：**A. 任意 ctx**（任意正整数、只在物理放不下时夹紧、requested→actual 可观测）；
> **B. 极端矩阵**（极短/极长 prompt、极小/极大 ctx、极小/极大模型、显存紧张、30 轮长会话、
> 背靠背请求、空输入）。每格记录：能否跑通 / 速度 / 显存与 RAM 峰值 / 是否可接受 / 改进项。
>
> 状态：0.5B 部分已完成（本机小模型，逻辑与账目测试）；9B/35B 与大 ctx/性能不回退等
> **需要大窗口**的部分标 `⏳`；0.5B 与 9B 的 ctx 矩阵、9B 大 ctx 已在大窗口补齐（§1.1/§1.2）。

---

## 0. TL;DR

| 用户要求 | 结论 |
|---|---|
| 任意上下文长度（不是预设） | **已实现**。`--ctx` 接受 1..1048576 任意整数（0=auto），配置/环境变量/launcher 数字框同源；只有物理装不下才夹紧，且打印 requested→actual 账目。 |
| 只在物理放不下时夹紧 | **已实现**，三层：①装载前 `[ctx] budget/CLAMPED` 预算账目 ②装载后**实测** free VRAM < margin 时按“实测 free + 真实 KV/token”算出可行 ctx 并**自动重试**（最多 4 次）③KV 分配 OOM 也按模型结构估算可行 ctx 重试。请求值全程保留。 |
| /props 与启动日志 | `/props` 新增 `n_ctx_requested` / `requested_ctx` / `actual_ctx` / `ctx_clamped` / `ctx_train_exceeded`；启动打印 `[serve] ctx : requested X -> actual Y (no clamp|CLAMPED...)`。 |
| 极小 ctx | ctx=1 可加载但任何非空 prompt 都返回明确 400（“prompt is too long … over by N”）；ctx=2 起可生成（prompt 1 token + 1 token）。**不再崩、不再静默变 32**。 |
| 极端条件不灾难性变慢 | 短 prompt/背靠背/空输入正常；长 prompt 的 prefill 是 O(ctx²) attention 主导（0.5B：4k 上下文 1390 tok/s → 8k 上下文 592 tok/s），是**可解释**的算力特性而非静默退化；VL 余量守卫在显存紧张时给明确错误/夹紧而不是 paging。 |
| 发现的明确 bug | 3 个已修（见 §5，各有独立 commit）：①小 ctx 时 `engine run` 校验后仍 forward 越界崩溃 ②serve 静默 `max(32,ctx)` 且无 requested/actual ③实测夹紧重试的返回值没有传回调用方（`run(Options)` 按值传参，重试从未生效）。另有 1 个流程级问题：**脏工作区 build 污染共享 `build/`**（§6）。 |

---

## 1. ctx 语义（实现后的契约）

```
--ctx N     1..1048576 任意整数（0 = auto：按空闲显存推导）
            = 总序列预算（prompt + 生成；不是“生成窗口”）
KV 大小      = kv_bytes_per_token(kv_type) × actual_ctx
夹紧条件     只有 actual_ctx < requested 且物理装不下（预算/实测/OOM），否则 verbatim
请求值       全程保留：/props.n_ctx_requested、请求 400 消息里的 “requested X, clamped to Y”
```

**为什么 0.5B 上 1..262144 全部 verbatim 成功**：预算
`usable = 11236 - fixed(463) - reserve(1968) ≈ 8805 MiB`；f32 KV=24 KiB/token，
262144 需 6144 MiB < 8805，物理放得下 ⇒ 不夹紧（`actual == requested`，`ctx_train_exceeded=true`
仅提示超过训练长度，不阻止）。超过物理上限（例如同模型再叠加一个 6 GiB 的 hog）时走 §2 的
实测夹紧路径，`requested 262144 -> actual 75775 (clamped)`。

### 1.1 0.5B ctx 边界矩阵（逻辑/账目测试；CPU=0.5B，GPU 与其它 agent 共存）

测量：`python tools/ctx_extremes_probe.py --scenario ctx_matrix --model <0.5B>
--ctx-list "1,2,16,100,1000,4096,12345,32768,131072,262144,262144:q8_0,262144:q4_0"`
（每格：启动 serve → /props → /completion 1 token（`prompt="Hello"`）→ 关闭）

| kv | ctx 请求 | actual | clamped | KV MiB | 加载 s | post-load free MiB | GPU used 峰值 MiB | 1-token 结果 |
|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 1 | no | 0 | 13.5 | 9408 | 6614 | **400**（prompt 1 token 占满 ctx=1，明确报错） |
| f32 | 2 | 2 | no | 0 | 11.7 | 9408 | 6614 | 200（1 token 生成） |
| f32 | 16 | 16 | no | 0 | 11.6 | 9408 | 6614 | 200 |
| f32 | 100 | 100 | no | 2 | 11.6 | 9406 | 6608 | 200 |
| f32 | 1000 | 1000 | no | 23 | 12.1 | 9384 | 6642 | 200 |
| f32 | 4096 | 4096 | no | 96 | 11.6 | 9312 | 6836 | 200 |
| f32 | 12345 | 12345 | no | 289 | 11.6 | 9024 | 7276 | 200 |
| f32 | 32768 | 32768 | no | 768 | 12.1 | 8640 | 7677 | 200 |
| f32 | 131072 | 131072 | no | 3072 | 11.6 | 6336 | 10053 | 200（`ctx_train_exceeded=true`） |
| f32 | 262144 | 262144 | no | 6144 | 11.6 | 3264 | 11977 | 200（`ctx_train_exceeded=true`） |
| q8_0 | 262144 | 262144 | no | 1632 | 11.6 | 7776 | 7451 | 200 |
| q4_0 | 262144 | 262144 | no | 864 | 11.6 | 8544 | 6681 | 200 |

结论：
- **任意正整数都被接受**（1/2/16/100/1000/4096/12345/32768/131072/262144 全部 `actual==requested`），
  没有“预设值”限制；KV 字节与 `kv_bytes_per_token × ctx` 完全对得上（f32 24 KiB/token、
  q8 6.375 KiB/token、q4 3.375 KiB/token）。
- ctx=1 的语义是明确的：**加载成功、/props 正常**，但任何非空 prompt 返回 400
  （“prompt is too long: 1 tokens >= actual context 1, over by …”）；ctx=2 起可生成。
- 超过 `ctx_train` 只标 `ctx_train_exceeded=true`（质量提示），不阻止——用户要自由，就要给自由。

### 1.2 9B ctx 矩阵（独占 GPU，2026-11-22 大窗口实测）

| kv | ctx 请求 | actual | clamped | KV MiB | 加载 s | post-load free MiB | GPU 峰值 MiB | 1-token 结果 |
|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 1 | no | 0 | 16.0 | 6178 | 6589 | 400（prompt 1 token 占满 ctx=1） |
| f32 | 2 | 2 | no | 0 | 15.5 | 6178 | 6591 | 200（prefill 35.3 ms） |
| f32 | 16 | 16 | no | 1 | 14.8 | 6176 | 6591 | 200（35.3 ms） |
| f32 | 100 | 100 | no | 6 | 15.4 | 6172 | 6579 | 200（35.4 ms） |
| f32 | 1000 | 1000 | no | 62 | 15.5 | 6114 | 6735 | 200（31.9 ms） |
| f32 | 4096 | 4096 | no | 256 | 16.5 | 5922 | 7611 | 200（32.7 ms） |
| f32 | 12345 | 12345 | no | 771 | 16.5 | 5378 | 9053 | 200（32.7 ms） |
| f32 | 32768 | 32768 | no | 2048 | 16.0 | 4130 | 9403 | 200（31.8 ms） |
| q8_0 | 131072 | 131072 | no | 2176 | 16.0 | 4002 | 9531 | 200（33.8 ms） |
| q4_0 | 262144 | 262144 | no | 2304 | 16.0 | 3874 | 9659 | 200（31.7 ms） |

结论（与 0.5B 相同，且更严格）：
- 任意正整数 ctx **全部 verbatim 接受**（`actual == requested`，`ctx_clamped=false`），
  KV 与 64 KiB/token（f32）、6.375 KiB/token（q8_0）、3.375 KiB/token（q4_0）逐字节吻合；
- 131072(q8_0) / 262144(q4_0) 都在 12 GB 卡上**直接跑通**（GPU 峰值 9.66 GB，post-load
  free 3.87 GB > 1200 MiB margin），不需要夹紧；加载时间稳定 15–17 s；
- ctx=1 语义仍是“可加载、非空 prompt 400”；ctx≥2 正常生成；
- 全程每 case 正常 `/shutdown`，无残留进程；测量时 GPU free 11.7 GB（独占）。

### 1.2 显存紧张时的“实测夹紧”（正确性关键路径）

场景：先起一个 0.5B hog（ctx 262144 f32，占 ~4.3 GB），再起第二个 0.5B 请求 ctx 262144。

| 阶段 | 结果 |
|---|---|
| hog | 成功：实测夹紧到物理可行值后启动（占 GPU ~4.3 GB） |
| target `--ctx 262144` | **成功且语义清晰**：`/props` → `n_ctx_requested=262144, actual_ctx=75775, ctx_clamped=true`；请求 200。日志含 `[ctx] measured clamp: requested 262144 does not fit; retrying with --ctx 75775`，以及 `[ctx] CLAMPED` 预算账目。 |
| target `--ctx 4096`（与 hog 共存） | 干净启动：actual=4096、无夹紧、生成 200。 |

过程中暴露并修掉的真 bug：**夹紧重试的 `ctx_suggested` 没有传回调用方**
（`serve::run(Options)` 按值传参 → 重试分支永远不触发，serve 直接退出）。
修复：`run(Options&)` in/out 引用，`engine serve` 最多 4 次重试。修复后上述结果成立。

代价：极端 ctx + 显存紧张时一次启动可能要 2–3 次“加载→实测→重试”（每次 10–45 s）。
这是“只有物理放不下才夹紧”的诚实代价；改进项：把 ledger gap（CUDA context/cuBLAS workspace，
本机实测 1.1–4.2 GiB 波动）纳入装载前预算，减少重试次数（见 §7）。

---

## 2. 极端矩阵（0.5B 已完成部分）

测量口径：`engine serve` + HTTP `/completion` / `/v1/chat/completions`；
first-token ≈ `timings.prefill_ms + decode_ms`（该响应）；tok/s 取自 `/completion` 的 timings；
显存峰值为 nvidia-smi 250 ms 采样（含同机其它进程，故同时记录 post-load free）；
RAM 为进程 WorkingSet 峰值（psapi）。

### 2.1 极短 prompt（0.5B，ctx 4096）

| prompt | tokens | first token | prefill | decode | 判定 |
|---|---|---|---|---|---|
| `Hi` | 1 | **42.0 ms** | 5.5 ms | 219 tok/s | ✅ 正常 |
| `Hello` | 1 | 42.1 ms | 5.5 ms | 219 tok/s | ✅ |
| `Hi there` | 2 | 46.2 ms | 10.2 ms | 222 tok/s | ✅ |

→ 首 token 延迟主要由固定开销（CUDA graph/采样/HTTP）构成，1–2 token prompt 无退化。

### 2.2 极长 prompt（0.5B，接近填满 ctx）

| ctx | prompt tokens | prefill | prefill tok/s | first token | decode | 判定 |
|---|---|---|---|---|---|---|
| 4096 | 3841 (94%) | 2763 ms | **1390 tok/s** | 2.82 s | 150 tok/s | ✅ 可接受 |
| 8192 | 7931 (97%) | 13394 ms | **592 tok/s** | 13.47 s | 111 tok/s | ⚠️ 慢但可解释 |

解释：prefill 的 attention 成本是 O(n²)（7931² / 3841² ≈ 4.3×，实测 13.4/2.76 ≈ 4.8×），
0.5B 的小 GEMM 无法摊薄 attention；9B 上 GEMM 占比更高、比值会小。**不是静默变慢，是算力函数**。
改进项：更快的长 ctx prefill attention（更大 tile / 分块 online softmax 的行复用 / int8 KV 读取），
或对超长 prompt 提前给“预计 prefill 时间”提示。

### 2.3 极小 ctx（0.5B）

| ctx | 短 prompt 生成 | 超 ctx prompt（201 tokens） | 判定 |
|---|---|---|---|
| 16 | 200（1 token prompt + 8 gen） | **400**：`over by 185 tokens; shorten the prompt, raise --ctx, …` | ✅ 明确 |
| 64 | 200 | **400**：`over by 137 tokens` | ✅ |
| 256 | 200 | 200（放得下） | ✅ |

无崩溃、无静默截断；错误消息包含 requested/actual（若被夹紧）与 over-by。

### 2.4 背靠背请求（0.5B，10 个连续 `/completion`）

10/10 200；单请求 wall 0.05–0.07 s（prefill 9.6–18.2 ms、8 token decode 36–39 ms）。
无排队/串行问题（单用户单线程设计；连续请求稳定）。

### 2.5 空输入（0.5B）

| 输入 | 结果 | 判定 |
|---|---|---|
| `/completion` `prompt=""` | **400** `prompt encodes to zero tokens` | ✅ 明确 |
| `/v1/chat/completions` `messages=[]` | **400** `'messages' must not be empty` | ✅ 明确 |
| `/completion` `prompt="   "` | 200（空白 tokenize 后非空 → 正常生成） | ✅ 语义可接受（文档写明） |

### 2.6 连续长会话 30 轮（0.5B，ctx 8192，每轮 +~120 token）

前缀缓存生效（`[req] prefill 431 [new 247, cached 3072]` → 第 30 轮 `cached 3712, new 217`）：
- 每轮 wall 0.35 s → 0.57 s（随历史线性增长，主要是每轮 ~220 token 的新 prefill + restore）
- 30 轮总 wall 10.4 s；进程 RSS 峰值 569 MiB，VRAM 稳定（post-load free 9216 MiB，无泄漏趋势）
- 判定：✅ 可接受；长会话成本来自“新增 token 的 prefill”，不是全量重算（cache 命中率 ~93–95% 的 token）。

⏳ 9B 的 30 轮（每轮 +100 token、真实中文问答）在大窗口补测。

### 2.7 大模型 / 大 ctx / 35B / 9B 性能不回退 ⏳

需要大窗口（等 generalize 释放 GPU）：
- 9B ctx 1/2/16/100/1000/4096/12345/32768 + 131072(q8_0)/262144(q4_0)
- 35B（CPU 专家）短/长 prompt、30 轮
- 性能不回退门槛：9B ≥45 tok/s、35B ≥29、Coder ≥24（与 2026-11-21 基线同口径）
- 五套验证：verify / smoke / gen_regression / verify_generality / verify_serve

---

## 3. launcher：任意 ctx 输入

改动（commit `f60d57a`，`src/launcher.cpp` + `tests/test_launcher.py`）：
- 上下文从 **7 个预设的滑块** 改为：**可编辑数字框**（ES_NUMBER，1..1048576 任意整数）
  \+ **对数滑块**（1..1048576 全覆盖，位置 500 ≈ 1024，精值用键盘输入）
  \+ **“自动（按空闲显存推导）”勾选**（等价 `--ctx 0`）。
- 实时估算按实际 `kv_type` 显示 KV 显存；超过显存可容纳值时红字提示
  “引擎实测放不下时会自动夹紧并打印 requested→actual 账目”。
- 非法/空输入：红字提示 + 启动按钮流程里 `MessageBox` 拒绝启动，不再静默回退到预设值。
- `launcher.exe --ctx N` 也接受任意 0..1048576（不再映射到预设槽位）。
- 截图（本机实测，`build/` 下；UI 不启动模型，仅 launcher 自身 + 模型扫描）：
  - `build/launcher-10-ctx-controls.png`：显存与上下文区域：可编辑数字框 + 对数滑块 + 自动勾选；
  - `build/launcher-11-ctx-12345.png`（及 `-crop.png`）：输入 12345 → “手动 12345 tokens ·
    KV 显存 ≈ 772 MB（每 token 64.0 KB）”；
  - `build/launcher-12-ctx-invalid.png`：清空 → 红字 “上下文长度必须是 1..1048576 的整数
    （任意值均可；当前输入不能启动）”；
  - `build/launcher-13-ctx-auto.png`：勾选自动 → 按当前空闲显存推导 601 tokens，等价 `--ctx 0`。
- 本轮顺带修掉的两个 launcher 真 bug（commit `d04d755`）：
  1. 启动时初始 `4096` 的 EN_CHANGE 在 `g_ctx_edit` 赋值前触发 → 初始就红字“非法”且不再自动恢复；
  2. **中文 IME 下数字框收不到键入**（IME 组合吞掉 WM_CHAR，ES_NUMBER 只有粘贴能用）；
     修复：数字 EDIT 解除 IMM 绑定 + WM_KEYDOWN 数字/退格/删除子类化直接改文本，任何 IME 下可用。
- `tests/test_launcher.py` 已更新覆盖任意值/非法提示/自动勾选/滑块映射；完整 GUI 验收（起 0.5B +
  启动/停止 + 证据 JSON）在大窗口后重跑并记录结果。

---

## 4. /props 与日志（可观测性）

`GET /props` 新增（llama.cpp 兼容字段 `n_ctx` 保持为 actual）：

```json
"n_ctx": 75775, "n_ctx_requested": 262144, "requested_ctx": 262144,
"actual_ctx": 75775, "ctx_clamped": true, "ctx_train": 32768,
"ctx_train_exceeded": true
```

启动日志（`engine serve`）：

```
[ctx] requested=262144->208311 kv_type=f32 kv/token=24576 B (24.00 KiB) kv_layers=24/24 kv_total=4882 MiB
[ctx] budget: free=7359 MiB - weights=463 MiB - act(rows=512)=46 MiB - reserve=1968 MiB -> max_ctx=208311
[ctx] CLAMPED: 262144 > max_ctx 208311 for kv_type=f32 (would need 6144 MiB, budget 4928 MiB)
[serve] ctx        : requested 262144 -> actual 208311 (CLAMPED: ...)
[serve] ctx clamp  : requested 262144 -> 137898 measured-infeasible (...); suggested --ctx 75775
[ctx] measured clamp: requested 262144 does not fit; retrying with --ctx 75775 (attempt 2/4)
[serve] ctx        : requested 262144 -> actual 75775 (CLAMPED: ...)
```

两层账目：装载前预算（runner）+ 装载后实测（serve），后者才是“物理放不下”的最终判据。

---

## 5. 本轮修掉的明确 bug（每项独立 commit）

| commit | 问题 | 修复 |
|---|---|---|
| `5414092` | `engine run --ctx N` 在 prompt+生成超过 N 时先打印部分 token 再 `error: forward: position out of range`（半途崩溃） | ctx 视为硬预算：`n` 夹紧到 `ctx-prompt` 并打印 `[run] n capped`；prompt 已占满时装载前明确 `[run] ERROR`；任何正整数 ctx 均可。 |
| `bd0b477` | `serve` 静默 `max(32, ctx)`（--ctx 1/2/16 被悄悄改成 32，日志还显示 requested=32）；无 requested/actual；物理放不下只会整进程失败 | 任意正整数透传；/props 5 个新字段；启动日志 requested→actual；装载后实测夹紧→返回可行 ctx→自动重试（最多 4 次）；OOM 也走估算重试；dense+超预估 ctx 不再直接拒绝。 |
| `fix(serve)` 后续（HEAD 47555d2 附近） | `ctx_suggested` 按值传参丢失 → 实测夹紧重试从未生效（§1.2 暴露） | `serve::run(Options&)` in/out 引用；重试链路实测生效（hog 场景 requested 262144 → actual 75775 成功）。 |
| `d04d755` | launcher ctx 数字框：初始即红字非法；**中文 IME 下键入数字被吞**（只有粘贴可用）；跨进程写入后画面不刷新 | 初始化时屏蔽 EN_CHANGE（g_ctx_sync）+ 每次刷新从输入框内容自愈；数字 EDIT 解除 IMM 绑定 + WM_KEYDOWN 数字/退格/删除子类化；截图脚本强制重绘。 |

---

## 6. 流程级发现：脏工作区 `build.sh` 污染共享 `build/`

症状（2026-11-22 16:25–16:35，`gen_regression` 9/10 FAIL，`got` 全空，coder 输出 75723）：
不是引擎 commit 回归，而是**含未提交 WIP 的工作区跑了 `bash build.sh`**，把 generalize 的
Gemma4「extended path」WIP（`[arch] extended path enabled ... qk_norm=1 gelu=1 embed_scale=64`
以及 `missing tensor blk.N.ffn_norm.weight`）编进了共享的 `build/test_gen.exe`。

证据：
- 旧二进制（16:25，脏树构建）：`gen_regression` → 1/10（9B/35B `got` 全空、coder 75723）；
- 独立 worktree 从**已提交 HEAD 47555d2** 重建 test_gen → `gen_regression` **10/10 GREEN**；
- 归因：`5414092` 只改 main.cpp、`bd0b477` 改 serve/config-help、`f60d57a` 改 launcher/tests，
  均不在 test_gen 的链接路径；HEAD 全量构建下五套全绿。
- 恢复动作已执行：`build/bak_wip_1625/` 备份坏二进制 → 用干净 HEAD worktree 的全套
  `build/*.exe`（sha1 校验一致）覆盖共享 `build/` → 重跑 `gen_regression` **10/10 GREEN**、
  `smoke_models` **40716/185/2614/2614 EXIT=0**。
- generalize 的 arch 提交进入 HEAD（53396ef）后，再次从干净 HEAD worktree 全量重建并覆盖共享
  `build/`，`gen_regression` 仍 **10/10 GREEN**、smoke 4/4 一致（`build/logs/genreg_head53396.log`
  / `smoke_head53396.log`）→ 共享环境当前是“HEAD 二进制且绿的”。

团队规则（已写入 `CONTRIBUTING.md` §1.1，并由 server-mode 的 `tools/check_clean_build.sh` 门禁）：
**共享 `build/` 只允许由“干净 HEAD worktree”构建**；含 WIP 的实验编译必须去独立 worktree
（如 `../engine_arch`），覆盖前确认无验证进程占用。

---

## 7. 发现的问题 / 改进项（未在本轮修）

1. **实测夹紧要重载 2–3 次**（极端 ctx + 显存紧张）：装载前预算没算进 ledger gap
   （CUDA context / cuBLAS workspace，本机 1.1–4.2 GiB 且随模型/ctx 变化）。
   改进：把 gap 建模进 `derive_ctx_limit` 的 reserve（或首轮 load 后缓存 gap 估算），
   让第一次就报出接近真值的 max ctx；重试仅作兜底。
2. **多轮重试期间 `[vram] ledger gap` 出现负值**（`g_buf_bytes` 只加不减）：纯诊断字段，
   不影响功能；改进：buffer 析构时同步扣减或每次 serve 重置计数器。
3. **长 prompt 的 O(ctx²) prefill**（§2.2）：0.5B 8k 上下文 592 tok/s。改进项见该节。
4. **ctx=1 无法生成**是接口语义（总预算=1），目前用 400 明确报错；若产品希望“1 token 生成”
   需要另立语义（例如 prompt 不计入预算），不建议。
5. 9B/35B 的 ctx 矩阵与性能不回退、35B 30 轮、launcher 截图 = `⏳` 大窗口/干净二进制后进行。

---

## 8. 复现命令

```bash
# ctx 边界矩阵（逻辑/账目；0.5B 可随时跑）
python tools/ctx_extremes_probe.py --scenario ctx_matrix \
  --model "$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  --ctx-list "1,2,16,100,1000,4096,12345,32768,131072,262144,262144:q8_0,262144:q4_0"

# 极端输入（短/长/小 ctx/空/背靠背/30 轮）
python tools/ctx_extremes_probe.py --scenario short_prompt,tiny_ctx,empty_input,back_to_back,session30 \
  --model "$MODEL_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# 显存紧张下的实测夹紧（hog + 第二个 serve）
python build/vram_tight_05b.py        # 开发脚本，输出 build/ctx_extremes/vram_tight_05b.json

# 手工验证 requested/actual
./build/engine.exe serve <model> --port 7030 --ctx 262144 --kv-type q4_0
curl -s http://127.0.0.1:7030/props | grep -E "n_ctx|requested_ctx|actual_ctx|ctx_clamped"
```

结果 JSON 在 `build/ctx_extremes/*.json`，serve 日志在 `build/ctx_extremes/serve-*.log`。
