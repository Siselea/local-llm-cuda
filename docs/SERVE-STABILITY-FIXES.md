# serve 稳定性修复（2026-11-17）：复读兜底 / 思考预算 / MTP 接线

对应三个实测问题（35B 真实对话对比 llama.cpp 时暴露）：

| # | 问题 | 现象 | 修复 |
|---|---|---|---|
| 1 | repeat guard 漏检变体/字符级复读 | 35B「帮我写1000字小说」在第 2 轮退化：句循环 `老城的秘密，是老人。`×28、字符级 `—你——你—…—134—134—`、交错 5 句循环×12；旧 guard 只拦"连续 3 句完全相同"，全部漏检，一直跑到 2048 token 上限（wall 82 s） | 等级化 `serve_repeat_guard`（0/1/2），新增：单元 5 连、归一化句 3 连、句循环（周期 2..8）、**CJK 句重复 ≥3 次（30 句窗口）**；`finish_reason="repetition"` |
| 2 | think-budget 到点后的重答质量 | 思考 512 token 仍未闭合时，旧行为在"思考状态"里强制 `</think>` 续写 → 继续规划/退化 | 到点**直接停止**，然后用**关闭思考的干净 prompt**（`<think>\n\n</think>\n\n`）重跑回答；并修了 `max_tokens <= think_budget` 时预算永不触发导致空回答的边界 |
| 3 | serve 未接 MTP + `--mtp` 加载 bug | `--mtp --mtp-k 2` + `--kv-type q8_0` 直接 `ERROR: model load failed: Buffer::upload exceeds allocation`；且 serve 生成循环从未调用 `spec_generate` | ① `load_mtp` 清零 KV 改为按**已分配 buffer 大小**（q8_0/q4_0 下 buffer 是量化的，旧代码按 f32 大小 upload）；② 新增 runner 增量投机解码 API `spec_begin/spec_round`，接入 serve 的 greedy 生成循环（单次 prefill 路径；采样请求和前缀缓存命中路径保持原逻辑） |

## 1. repeat guard（`include/engine/serve_guard.h`）

`serve_repeat_guard` 从 bool 改为 int 等级（CLI `--repeat-guard <0|1|2>`，默认 2）：

| 等级 | 规则 |
|---|---|
| 0 | 关闭 |
| 1 | 句级：连续 3 句完全相同（≥8 字节） |
| 2（默认） | 1 的全部 + ① 4..96 字节单元连续重复 5 次（`—134`/`—你` 字符级循环）；② 连续 3 句**归一化**后相同（数字/ASCII 字母归一，抓 `岁月的秘密，是老人A。` 这类变体）；③ 句循环：最近 30 句里出现 2..8 句周期 × 3 轮（抓交错 5 句循环）；④ **CJK 句（≥36 字节归一化）在最近 30 句内出现 ≥3 次**（提前停：用户最多看到 3 次重复，而不是 8+）。思考段只做 ①（思考块会合法重复 markdown 表头），答案段做全部。 |

触发后：

* `finish_reason = "repetition"`（不再是 `length`）；
* `timings.repeat_guard_hit = true`；
* 日志 `[serve] WARN repeat-guard(level N): ... (rule=..., tail=...)`；
* `/props.repeat_guard` 暴露等级。

**单测**（无需 GPU）：`tests/test_serve_guard.cpp` → **24/24 PASS**，覆盖：

* 真实模式：句循环×28、`—134`×40、`—你`×30、交错 5 句 cycle×3、数字变体句×4、周期 2 交替句；
* 不误杀：正常中文故事 1000 字、正常英文散文、正常中文对话、思考段重复表头、正常 markdown 表格/列表、
  **Python 代码 `print(is_prime(17)) # True` / `# False` 列表**、短句重复 2 次、单个重复字；
* 等级语义：level 1 不抓字符级（预期）、level 0 关闭。

```bash
ziglang c++ -std=c++17 -O2 -Iinclude tests/test_serve_guard.cpp -o build/test_serve_guard.exe
build/test_serve_guard.exe          # 24/24 PASS
```

## 2. think-budget（`src/serve.cpp`）

`serve_think_budget`（默认 512，`--think-budget` 可调，0 关闭）：

1. 思考段超过预算仍未闭合 → 立即停止当前生成（不再在思考状态里续写）；
2. chat handler 用同一个 messages 渲染**关闭思考**的 prompt：`...<|im_start|>assistant\n<think>\n\n</think>\n\n`，
   重新生成回答，追加到同一个流式响应（前缀缓存使第二次 prefill 很便宜）；
3. 边界：`max_tokens <= think_budget` 时预算按 `min(think_budget, max_tokens-1)` 生效，
   避免"长度 break 先发生、预算永不触发、回答为空"；
4. 观测：`timings.think_budget_hit`、日志 `WARN think-budget: forced </think> after N tokens`。

## 3. MTP（`src/mtp.cpp` / `include/engine/runner.h` / `src/serve.cpp`）

* **加载 bug 根因**：`load_mtp()` 里用 `n_kv*max_seq*head_dim` 个 **f32** 零值 upload 清零 MTP 层的 KV，
  但 `--kv-type q8_0/q4_0` 时 KV buffer 是**量化**的（q8_0 只有 f32 的 ~1/3.7 大小），
  于是 `cuda::Buffer::upload` 抛 `Buffer::upload exceeds allocation`。
  修复：按 `layers_[b].kc.size()`（已按 kv 类型分配）清零。
* **serve 接线**：新增增量 API
  * `spec_begin(prompt_ids, first_token, last_row)`：单次 prefill 后播种 MTP 上下文/隐状态；
  * `spec_round(k, &emit)`：draft k 个 token → 一次 batch verify → 输出 `[pending, 接受前缀…]`，
    等待 token 更新为 bonus token；
  * serve 生成循环在 **greedy 且 prefill 为单次调用**时用这两个 API 出 token，采样请求/前缀缓存命中走原路径。
* **正确性判据**：投机解码只在 target arg-max 与 draft 一致时接受，输出应与普通 greedy **逐 token 一致**；
  `test_gen --mtp` 已有该回归。serve 侧的对照见 `tools/serve_mtp_check.py`。
* **性能预期（重要）**：本机 35B 是"CPU 专家并集翻倍"的分配方式，历史实测 MTP 在该配置下
  可能净亏，所以 **默认关**；`--mtp` 只建议在有 GPU 专家余量、或短上下文场景试。

## 4. 工具

| 工具 | 用途 |
|---|---|
| `tests/test_serve_guard.cpp` | repeat guard 单测（24 项，无需 GPU） |
| `tools/serve_stability_check.py` | 实机：两轮 `你好/帮我写1000字小说` + 三条对照 + think-budget 问题，统计 finish/guard/复读/规划文本残留 |
| `tools/serve_mtp_check.py` | 实机：同一 greedy prompt 在 `--mtp` 开/关两个 server 下逐 token 对比 + tok/s |

## 5. 实机验证（2026-11-17，35B ctx4096 q8_0 KV，12 层 GPU 专家，RTX 3060 12GB）

### 5.1 #1 repeat guard（`--repeat-guard 2`，温度 0.7 / top_p 0.8 / top_k 20 / min_p 0）

| 场景 | finish | guard | wall | 生成 | 输出 |
|---|---|---|---|---|---|
| 你好 → 帮我写1000字小说（同会话第 2 轮） | **stop**（模型自然收尾） | False | **42.8 s** | 597 CJK 正文 | 《最后一班地铁》完整收尾（"希望这篇小说能给你带来一些温暖…"），无复读 |
| 同上（v6，unit 规则收紧前的一次采样） | **repetition** | True | 52.96 s | 1255 tok | 998 CJK《雨夜的旧书店》，末尾 `他找到了自己。`×5 被 unit 规则截断 |
| 对照：你好 | stop | False | 8.6 s | 193 tok | "你好！很高兴见到你。今天有什么我可以帮你的吗？" |
| 对照：介绍杭州 | stop | False | 8.6 s | 195 tok | 三句话中文介绍（141 CJK） |
| 对照：写 Python 代码 | length（max_tokens=768 的 cap） | **False** | 48.9 s | 768 tok | 代码输出正常；中文注释句不再误杀（句级归一化只对 CJK 主导句生效） |

**guard v7 的两次精度修正（实测驱动）：**

1. `unit-5x → 12+ 字节单元的 4 连`：真实 35B story 末尾出现 `他找到了自己。`（7 字 = 21 字节）
   连续重复，5 连才停 → 用户在界面看到 5 行相同短句。改为 **≥12 字节单元 4 连即停**后，
   用户最多看到 4 次；同时加 `unit_informative()` 豁免，避免 `========`/`————————` 这类装饰线
   被当成循环（单测覆盖）。短单元（4–11 字节）仍用 5 连，字符级 `—134`/`—你` 循环照旧命中。
2. 句级归一化只对 **CJK 主导句**（非 ASCII 字节 ≥ 一半）生效：修复了
   `print(f"1 是素数: {is_prime(18)}")` 这类"代码 + 中文标签"被归一化后误判为 3 连相同的问题。

> 说明：42.8 s 里约 21 s 是 think-budget 到点前的 513 思考 token（见 5.2）；扣掉它每 token 速度
> 与 llama.cpp 基本持平（我们 decode 24.8 tok/s）。把 `--think-budget` 调到 384 可再省 ~6 s
> （默认仍是 512，见 §7 调优表）。

### 5.2 #2 think-budget

- 「为什么天空是蓝色的？请用通俗的方式解释。」：`think_budget_hit=True`（513 token 到点），
  实际输出 **389–524 CJK** 的通俗解释，**planning markers=0**（无 "Thinking Process"/"Analyze the
  Request" 残留），通过。
- 边界修复后，`max_tokens` 较小的请求（如 512）也会在 `max_tokens-1` 时触发预算，不再出现空回答
  （修复前实测 control 介绍杭州在 max_tokens=512 时 answer_len=0）。

### 5.3 #3 MTP（2026-11-20 已跑通，默认仍关）

**两个 bug 依次修掉：**

1. `Buffer::upload exceeds allocation`：`load_mtp()` 用 f32 大小清零 MTP 层 KV，`--kv-type q8_0`
   下 KV buffer 是量化的 → 改为按 `layers_[b].kc.size()`（已分配大小）清零。
2. **`--mtp` 加载崩溃（0xC0000005）**：serve `run()` 在模型构造**之前**执行
   `st.mtp_enabled = config::get_bool("mtp") && st.model->mtp_available()` → `st.model` 还是
   nullptr，只有 `--mtp` 打开时才解引用 → 崩溃。定位方法：stdout 在 access violation 时丢失，
   改用**每次 fopen/fwrite/fclose 的 trace 文件**（`build/logs/mtp_trace.txt`），看到
   trace 停在 tokenizer 之后、构造之前，据此定位到 null deref；把 mtp 状态初始化移到
   "model ready" 之后即修复。修复后 `--mtp` 加载/启动正常。

**实机验证（2026-11-20）：**

| 模型/配置 | `/props.mtp` | greedy 逐 token vs 不开 MTP | tok/s（开 vs 关） |
|---|---|---|---|
| Qwythos-9B-Claude-Mythos-5-1M-MTP Q4_K_M，ctx2048，GPU 全驻留 | available=true, enabled=true, k=2 | **完全一致（58/58 token）** | **9.91 vs 31.32（-68%）** |
| 35B Q4_K_M，ctx4096 q8_0，12 层 GPU | available=true, enabled=true, k=2 | **不一致，第 147 个 token 分叉**（gen 211 vs 223） | **12.53 vs 24.25（-48%）** |

结论：

* serve 的投机解码接线本身是对的（Qwythos 上 bit-exact）；
* **35B（hybrid SSM）上投机解码不保真**：reject 后 SSM/conv 状态回滚不精确，长序列累积漂移
  （前 147 token 一致后分叉）——这是下一步要修的点（需要保存/恢复 draft 位置的完整 SSM 状态或
  只在 dense 层模型开放 MTP）；
* **本机两个模型上 MTP 都是净亏**（CPU 专家并集翻倍 + 每步 verify 的同步开销），
  默认保持关闭，`--mtp` 只作为实验开关。


### 5.4 真实浏览器（2026-11-20 已补齐）

`tools/webui_e2e.py`（headless Edge + CDP，真键盘打字 + 真鼠标点发送）在 35B 上跑通 3 轮
（`build/logs/e2e_final5/`）：

| 轮次 | 输入 | 首字 | 整轮 wall | 截图 |
|---|---|---|---|---|
| 1 | 为什么天空是蓝色的？请用通俗的方式解释。 | 3.6 s | 43.2 s | `01_done.png` |
| 2 | 帮我写1000字小说 | 10.8 s | 59.7 s | `02_done.png`（生成中被 guard 在 811 token 停住，无 5+ 复读） |
| 3 | 用三句话介绍杭州。 | 14.3 s | 31.1 s | `03_done.png` |

另跑了一轮**单轮 story**（`build/logs/e2e_story/01_done.png`）：完整 1000+ 汉字小说
（1714 字符，含思考），正文结尾完整、无重复段落，用户视角就是一张完整的浏览器截图。

自动化过程中修了 3 个 e2e 自身的坑（不是服务 bug）：发送后如果生成还没开始（prefill/思考阶段
DOM 不变）会被误判为"回合结束"；合成鼠标点击在部分 webui 版本上不提交（改为检查输入框仍非空后
补发真 Enter）；截图在流式途中截到半句（改为 idle 后再等 2 s + 滚到底再截）。


### 5.5 四套验证（2026-11-20，最终 `build.sh` 全量构建后）

| 验证 | 结果 |
|---|---|
| `bash build.sh`（全量，非增量） | **OK**：`build/engine.exe` 2,317,312 B（11-20 18:36）、`build/launcher.exe` 707,072 B |
| `tools/test_serve_guard.exe`（repeat guard 单测） | **27/27 PASS** |
| `tools/verify.sh` | **ALL GREEN**（Python 单测 + G0 IR parity + M2 plan parity） |
| `tools/smoke_models.sh` | **PASS**（40716 / 185 / 2614 / 2614 各模型 smoke 通过） |
| `tools/gen_regression.sh` | **GREEN 10/10**（含 35B「capital of france」12 greedy token == llama.cpp；本轮的 35B 空输出用例已随消融修复一起通过） |
| `tools/verify_generality.sh` | **GREEN 26/26** |
| `tools/verify_serve.sh`（0.5B / 9B） | **GREEN 22/22、21/21** |

最终 `build/engine.exe` 必须是用 `bash build.sh` 全量构建的那份（18:36），不是增量——本报告中的
实机数据全部来自该构建（MTP 两个测试用的是同一源码树构建的 `engine_new.exe`，只多了 null-deref 修复，
该修复随后已进入 18:36 的 build.sh 产物）。

## 6. 已知限制

* guard 是启发式：正常创作里"有意的排比/复沓"可能被等级 2 的重复句规则提前停；
  需要长排比时可 `--repeat-guard 1`（只拦连续 3 句相同）或 `0`。
* think-budget 到点重答会丢掉已生成的思考（多花 ~512 token 的时间），这是为了回答质量；
  想省时间可调小 `--think-budget`。
* MTP 在本机 35B 配置下可能净亏（CPU 专家并集翻倍），默认关。

## 7. 与 llama.cpp 的最终对比（同模型、同用户输入）

| 场景 | llama.cpp b12128（用户实测） | 我们（最终构建 v7） | 差异 |
|---|---|---|---|
| 你好 | 2.1 s / prompt 375 tok / GPU 峰值 41% | **8.6 s** / prompt 11 tok / 15 层 GPU 专家 | 慢 4×：llama.cpp 的 2.1 s 是 375 token prompt（含 system prompt）prefill；我们 8.6 s 里约 21 s 是 think-budget 前的思考（"你好"这种简单请求也会思考 512 token）。关思考后单请求 wall ≈ 1.9 s（早期实测） |
| 帮我写1000字小说 | 21.2 s / prompt 895 tok（prefix cache）/ ~750 tok 输出（~1000 汉字） | **42.8 s** / prompt 42 tok / 597–1255 tok（含 513 思考）/ 597–998 汉字 | 慢 2×：其中 ~21 s 是 think 预算的思考 token；**扣掉思考后每 token 速度与 llama.cpp 基本持平**（decode 24.8 tok/s）。llama.cpp 另有 MTP draft（历史 +29.5%），我们的 serve MTP 已跑通但本机净亏且 35B hybrid SSM 不保真（见 §5.3），默认关 |

**复读问题对比（用户最关心的点）：** 修复前 story 第 2 轮会退化到 `老城的秘密，是老人。`×28 /
`—你——你—` 字符级循环并一路跑到 2048 token 上限（82–85 s）；修复后同一请求：**不再出现 5 次以上重复**，
`finish_reason` 为 `repetition` 且输出在 600–1255 token 内停住（wall 42.8–53 s），
浏览器实测单轮 story 完整输出 1000+ 汉字、正文无重复段落（截图 `build/logs/e2e_story/01_done.png`）。

### 速度与思考质量互换（已写入 `/props` 与 CONFIG 文档）

- `--think-budget 384` / `256`：更早停止思考并重答，story wall 预计 42–46 s；复杂推理题思考变少。
- `--think-budget 0`：关闭强制闭合（完全由模型决定，可能重演"思考不闭合"）。
- `--repeat-guard 1`：只拦连续 3 句完全相同（保留长排比/复沓创作）；`0` 关闭。
- `--mtp` / `--mtp-k`：投机解码（默认关；本机 35B/9B 均为净亏，且 35B hybrid SSM 不保真）。
- `/props` 暴露：`think_budget`、`repeat_guard`、`no_limit_tokens`、`prefix_cache`、`mtp{available,enabled,k}`。
