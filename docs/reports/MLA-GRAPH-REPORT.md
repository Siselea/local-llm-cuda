# Coder（DeepSeek-V2-Lite，MLA）decode 路径 CUDA Graph 化 — 实现与实测

日期：2026-11-21（下午）
范围：`src/runner.cpp`（MLA attention 段图化 + 路由表 pinned async 上传）、
`src/cuda.cpp` / `include/engine/cuda.h`（`Graph::launch()` 代数计数器、`Graph::capture_abort()`）、
`include/engine/runner.h`、`src/config.cpp`（新键 `mla_attn_graph`）、`docs/CONFIG.md`、
`docs/GRAPH-INVARIANTS.md`（新）、`tools/`（4 个新工具）。
**不碰**：CUDA 内核数学、`src/serve.cpp`、qwen35/35B 的既有图路径。

---

## 0. 结论（TL;DR）

| 指标 | 改动前（12:00 基线二进制） | 改动后 | 变化 |
|---|---|---|---|
| Coder decode（4-token prompt，9 步，交错 5 轮中位） | **48.92 ms/token** | **41.02 ms/token** | **−7.90 ms（−16.2%）** |
| tok/s | 20.44 | **24.38** | +19.3% |
| 每 token kernel 发射次数（`ENGINE_PIPE_PROF` 稳态） | **765** | **279** | **−486**（= 27 层 × 18 次，精确吻合） |
| 逐 token bit-exact | — | gen + logits **逐字节相同**（4/4 用例，409600 B） | ✔ |
| Coder serve e2e | — | **9/9 PASS**，decode 23.2 tok/s，输出与图无关 | ✔ |
| 五套验证 | — | **FIVE GREEN**（见 §6） | ✔ |

图化收益 −7.45 ms 来自 MLA attention 段图（同二进制对照 `ENGINE_NO_MLA_ATTN_GRAPH=1`），
不是测量漂移（对照档 48.5–48.9 ms 与基线一致）；pinned async 路由上传约 −0.45 ms（在噪声内）。

---

## 1. 为什么 Coder 之前是 0 张图

Coder 是 MoE 模型，`use_graph_` 被强制关闭（host routing 需要逐层 D2H + 同步，整模型图不可行）；
而 per-layer 自动图只覆盖 qwen35 层（`q35_layer_graph_ok` 对 MLA 返回 false）。
于是每个 decode token 有 ~765 次裸 `cuLaunchKernel`，几乎全是 MLA 的 attention 段
（27 层 × 18 次）与 MoE 段。

## 2. 实现

### 2.1 MLA attention 段图（`mla_attn_graph`，默认开）

段 = `rms(ln1) + run_mla_attention() + rms(ln2)`。整段在 decode 时是**纯 kernel 序列**：
四个 MLA 投影走 `mla_quant` 的 GEMV kernel；fp32 的 k/v 扩张走 `matvec_f32_ptr`，它在
`n_tokens<=4` 时选择 `gemv_f32` **kernel** 而不是 cuBLAS。因此可以整段录制（与 qwen35 的
mixer 图同构）：

- **懒捕获**：第一个 decode token 正常执行（本 token 结果由它产生），同一个调用里再录制；
  第二个 token 起重放（`[mla-cap] recorded 18 launches` × 27 层 = 一次性 486 次录制）。
- **录制内容**：18 个 kernel；位置、`max_seq_`、指针都通过设备内存/固定 buffer 传入，
  与 `pos` 无关，因此**不需要按 ctx 分桶**。
- **开关**：配置键 `mla_attn_graph`（默认 `true`）；`ENGINE_NO_MLA_ATTN_GRAPH=1` /
  `--no-mla-attn-graph` / `--no-graph` 关闭；`ENGINE_DUMP*` / `ENGINE_PROFILE` /
  `ENGINE_KPROF` 下自动退回普通路径（这些路径会在段内同步）。`ENGINE_PIPE_PROF`
  允许开着图（它只在 pipeline 调用之间同步/打印），用于发射计数对照。
- 诊断：`ENGINE_DBG_MLA_CAP=1` 打印每层录制次数与首次重放。
- 捕获失败（极端情况）时 `Graph::capture_abort()` 恢复流状态并永久关闭该优化，不留半录制的流。

### 2.2 路由表 pinned async 上传

Coder 的 GPU 驻留层用 host routing，每层原来有 2 次**同步** `cuMemcpyHtoD`
（`d_idx_`/`d_w_`，每次刷 WDDM 队列）。改为：成员 `cuda::PinnedBuffer d_idx_pin_/d_w_pin_`
暂存 + `upload_async`。源缓冲只在下一次 `run_moe` 覆写，而那次覆写前必有一次流同步，
所以源缓冲在拷贝完成前不会被改写。实测收益在噪声内（−0.45 ms），但去掉了 2 次队列刷写。

### 2.3 顺带修掉的通用不变量 bug（`quant_reuse_`）

`quant_reuse_`（dp4a GEMV 的 int8 激活暂存）用 `cuda::g_launch_calls` 证明"暂存写入后没有新
发射"，从而在 gate/up、qkv、q/k/v 之间复用 `qx_i8_`。**graph replay 执行 kernel 但不增加
该计数**，于是 replay 后紧接宿主 GEMV 时会误复用上一 token 的暂存。

实际触发：MLA 段图 + `ENGINE_NO_HEAD_TOPK=1`（head 少了 2 次发射，计数恰好停在旧 `qc_stamp_`）
→ Coder 第 3 个 token 起输出变成重复 token。默认 `head_topk` 有那 2 次发射，掩盖了它。

修法（通用）：新增 `cuda::g_graph_launches` 代数计数器（`Graph::launch()` 每次 +1），
所有"按发射计数判活"的缓存把代数一并纳入判活（`qc_graph_ == g_graph_launches`），
即**任何一次 replay 都使暂存失效**。不变量写在 `docs/GRAPH-INVARIANTS.md`，
回归测试 `tools/graph_cache_regression.sh`（9B/Coder/0.5B × 图开/关 × head-topk 开/关）。

---

## 3. `cuGraphLaunch` 单价（先测再决定值不值得做）

`tools/bench_graphlaunch.cpp`（RTX 3060，WDDM，200 iter × 7 批取中位）：

| 操作 | 耗时 |
|---|---|
| 裸 `cuda::launch`（1 kernel） | **10.41 µs** |
| `cuGraphLaunch`（1 节点图） | **10.02 µs**（≈1 次裸 launch，**远低于传言的 50–100 µs**） |
| 18 次裸 launch | **177.35 µs** |
| 一条 18 节点图 | **15.52 µs**（省 **161.8 µs/层**） |
| GPU 侧 elapsed：18 裸 vs 1 图 | 163.7 µs vs 16.3 µs |

⇒ 判定 WORTH IT：27 层 ≈ **4.4 ms/token** 提交成本可省；实测更多（−7.45 ms），
因为图还消除了裸提交在 GPU 上留下的队首空隙（GPU elapsed 数据佐证）。

## 4. 逐 token bit-exact

`tools/coder_bitexact.sh`：基线（无 MLA 图、同步上传）vs 新二进制，四类 prompt：
auto 放置 12 token、`ENGINE_GPU_EXPERTS=16` 混合放置 12 token、单 token prompt 8 token、
7 token prompt 10 token。每类每个二进制跑两遍（先证明"同一二进制两次结果一致"，
排除 split-K 原子加的固有非确定性），再跨二进制比较：

- 4/4 用例：**gen 完全一致 + logits 409600 B 逐字节相同**。
- 早先一轮出现的 1-byte logits 差异经此方法确认是引擎自身的 run-to-run 非确定性
  （同二进制两遍也不一致），不是图引入的差异；之后的重跑里两个二进制都自洽且跨二进制逐字节相同。

## 5. 交错 A/B（5 轮 × 4 档，`ENGINE_DEC_SKIP=2`）

| 轮 | base（12:00 基线） | `ENGINE_NO_GRAPH=1` | `ENGINE_NO_MLA_ATTN_GRAPH=1` | new（默认） |
|---|---|---|---|---|
| 1 | 48.54 | 48.77 | 49.54 | 40.86 |
| 2 | 49.41 | 49.21 | 48.42 | 41.60 |
| 3 | 48.89 | 49.39 | 49.72 | 41.60 |
| 4 | 49.64 | 48.81 | 47.85 | 40.74 |
| 5 | 48.92 | 48.30 | 48.47 | 41.02 |
| **中位** | **48.92** | **48.81** | **48.47** | **41.02** |

- **MLA 图 = new − mla-graph-off = −7.45 ms**（同二进制、只切图的干净对照）
- 关掉所有捕获路径（`ENGINE_NO_GRAPH=1`）与基线一致（48.81 vs 48.92，噪声内）
- 路由 async 上传 = mla-graph-off − base ≈ −0.45 ms（噪声内，声称"不退化"）
- 包含一次性捕获成本的"不 skip"口径：第一个 decode token 多 ~0.4–0.6 ms（486 次录制
  + 27 次 instantiate），在 9 步平均里约 +0.05 ms；真实用户每次会话只在首 token 承担一次。

## 6. 验证（全部在最终二进制上重跑）

- `tools/graph_cache_regression.sh`：**3/3 PASS**（9B 4 配置、Coder 4 配置、0.5B 2 配置逐 token 一致，
  其中 Coder 的 `ENGINE_NO_HEAD_TOPK=1` 正是当初触发 quant-cache bug 的组合）。
- **五套验证：FIVE GREEN** —— `verify` ALL GREEN / `smoke` 40716·185·2614·**2614** /
  `gen_regression` 10-0 / `generality`（22 PASS，含 adaptive placement）/ `verify_serve` 22-0。
- **Coder 真机对话 e2e：9 passed, 0 failed（GREEN）** —— `tools/coder_serve_chat.sh`：
  起服务 → 健康检查 → chat 端点"无模板"400 契约 → `/v1/completions` 中文指令对话
  （120 tok，decode 23.23 tok/s）→ 第二轮更长 prompt → SSE 流式（26 事件）→
  llama.cpp `/completion` → 优雅关服 → 日志无 error。
- **服务输出与图无关**：`tools/coder_text_graph_neutral.sh` 用同一个 greedy 请求对比
  A) 11-20 的 pre-MLA 二进制、B) 新二进制 + `ENGINE_NO_MLA_ATTN_GRAPH=1`、C) 新二进制默认：
  三份**输出文本逐字节完全相同**（md5 `d2eecae7ae6659c6`）⇒ 图化是文本中性的。

### 已知问题（预先存在，非本次引入）

Coder-V2-Lite 的**长文本质量**本身是已知限制（`docs/reports/SPEC-COMPLIANCE.md`：
"5-token 以上的生成质量已定性为数值敏感"）：greedy 多 token 输出会出现乱码/无效 UTF-8
字节；GGUF 也没有 chat template，所以 serve 的 `/v1/chat/completions` 按设计返回 400、
需要走 `/v1/completions` 原样 prompt。本次 e2e 只断言"服务可用 + 速度 + 与图无关"，
文本质量不在本任务范围（与 pre-MLA 二进制逐字节一致已证明不是本次改动引起）。

## 6b. 后续：35B 的两个 graph 机会（同一轮实测）

### A) 35B GPU 层整层图 —— **实测证伪，不做**

历史结论是"整层图更慢"，怀疑根因是 `run_gpu_layer_graph()` 置 `layer_graph_mode_=true`
关掉了融合专家内核（`can_fuse` 要求 `!layer_graph_mode_`）。实测证明**这不是真因**：

- 同二进制交错 3 轮（`ENGINE_GPU_FULL_LAYER_GRAPH=1` vs 默认，15 步中位）：
  **plain 37.81 → 整层图 40.00 ms（+2.19 ms，更慢）**，gen 完全一致。
- `ENGINE_PROF_ACC=1` 账本：整层图把 GPU 层提交从 12 次降到 0 次（launches 405→225，
  GPU 层 0.201→0.010 ms），但 **CPU 层 sync 从 0.322 涨到 0.512 ms/层（+4.75 ms/token）**，
  CPU 层总时间 1.284→1.511 ms/层。净结果 +2.4 ms。
- 原因：35B GPU 层走设备路由，`all_idx` 为空，`can_fuse` 在两模式下本来就是 false；
  整层图真正的作用是让 **host 不再受提交限制、跑到 GPU 前面**，省下的提交时间被
  CPU 层的 `stream_sync`（等 GPU + 队列排空）原样吃掉。**35B 已经从"提交受限"变成
  "CPU 层 sync / 宿主 GEMV 受限"，继续给 GPU 层加图只会把时间搬家。**

### B) 35B CPU 层的共享专家 GPU 段图（`cpu_layer_ffn_graph`，默认开）—— **采纳**

CPU 层每层有 ~8–9 次 kernel 发射（zero-fill + 共享 gate/up/silu/down + gate_inp + add），
这段只依赖 `hn_` 和固定权重、逐 token 不变，且**在 host 的关键路径上**（host 必须先提交完
才能 `stream_sync`）。按层录一张图（懒捕获：第一个 decode token 正常执行后录制）：

| 指标（同二进制交错 5 轮，15 步中位） | graph off | graph on | 变化 |
|---|---|---|---|
| 35B decode | 38.04 ms | **37.36 ms** | **−0.68 ms（−1.8%）** |
| launches/token（`[acc]`） | 405（CPU 225） | **230（CPU 50）** | −175（=25 层 × 7） |
| CPU 层 launch | 0.116 ms/层 | **0.049 ms/层** | −0.067 |
| CPU 层 sync | 0.332 ms/层 | 0.374 ms/层 | +0.042（另一半被 sync 吃掉） |

上限 −1.7 ms 只兑现了 −0.68 ms：**省下的提交时间约一半变成了 sync 等待**（与 A 同一机制），
但仍净正收益，且 kernel 序列/加法次序完全不变。开关 `ENGINE_NO_CPU_FFN_GRAPH=1` /
`--no-cpu-ffn-graph`。bit-exact：35B 16 token × 6 次运行（新默认×2 / 图关×2 / 改动前二进制×2）
gen 完全一致（35B logits 本身 run-to-run 非确定，split-K 原子加）；Coder 4/4 用例
gen + 409600 B logits 逐字节相同（Coder 的 CPU 层同样走这条图）。

顺带收益：Coder 也有 10 个 CPU 层，交错 4 轮实测 **38.80 → 38.26 ms（−0.54 ms）**。

> **测量口径提醒**：本报告的绝对 ms 只在同一次交错批次内可比。机器状态（GPU 时钟/
> 热状态）会让同一二进制的绝对值漂移 ±2 ms 量级（例如 Coder 默认档在 14:05 的批次
> 里是 41.0，在 15:47 的批次里是 38.3，代码相同）。所有结论都来自同批交错的中位差。

## 7. 失败/负面尝试记录

1. **`quant_reuse_` 不变量 bug**（§2.3）：先手动在 replay 后清 `qc_ptr_`，后改为
   `g_graph_launches` 代数计数器并写入 `docs/GRAPH-INVARIANTS.md`（更通用、单一机制）；
   回归脚本覆盖 `ENGINE_NO_HEAD_TOPK=1` 这个触发路径。
2. 早先按"每层 ~13 次发射"估算，实测是 **18 次**（文档/配置帮助文字已按实测更新）。
3. 早先一次 bit-exact 对照 run 出现"空 gen"（疑似瞬态），加了"空 gen 即 FAIL + 每二进制跑两遍"
   后未再复现，4/4 用例通过。
4. 回归脚本初版把 `--no-graph` 当 CLI 参数传（配置前缀只能是环境变量）导致空输出，
   已改为 `ENGINE_NO_GRAPH=1`；脚本行尾统一 LF（CRLF 会破坏 `\` 续行）。
5. `ENGINE_DBG_LAYERMS` 在本机每 token +11 ms，未用于算账；发射计数用 `ENGINE_PIPE_PROF`，
   时间用交错 A/B + 中位。

## 8. 文件与开关

- 代码：`src/runner.cpp`（`mla_attn_segment` / `mla_attn_graph_ok` / `run_mla_attn_graph` /
  `capture_mla_attn_graph`；`run_moe` 路由上传；`run_gemv_plan` 代数判活）、
  `src/cuda.cpp` + `include/engine/cuda.h`（`g_graph_launches`、`capture_abort`）、
  `include/engine/runner.h`、`src/config.cpp`、`docs/CONFIG.md`。
- 关闭：`ENGINE_NO_MLA_ATTN_GRAPH=1`（只要 MLA 图）、`--no-graph` / `ENGINE_NO_GRAPH=1`（全部图）。
- 文档：`docs/GRAPH-INVARIANTS.md`（后续任何图化必须遵守的 4 条不变量）。
- 工具：`tools/bench_graphlaunch.cpp|sh`、`tools/ab_mla_ladder.sh`、`tools/coder_bitexact.sh`、
  `tools/mla_launch_count.sh`、`tools/graph_cache_regression.sh`、`tools/coder_serve_chat.sh`。
