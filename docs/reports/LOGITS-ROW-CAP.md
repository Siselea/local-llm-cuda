# LOGITS-ROW-CAP：prefill 只保留 1 行 logits，让 chunk 变大（减少重复反量化）

> 作者：prefill-fast｜日期：2026-11-21｜改动：`src/config.cpp`（`prefill_logits_rows`）、`include/engine/runner.h`、`src/runner.cpp`（logits 分配 + `activation_bytes` + 增长 cushion）、`src/serve.cpp`（`prefill_rows_max` 默认 1024→4096）
> 背景：层主序 V1 被显存卡住（整序列 hidden 220–537 MB 放不下，见 §1），改用同一杠杆的更小改动。

---

## 1. 为什么不是层主序（lead 已批准改道）
层主序要求"整序列 hidden 常驻"：13.4k token × 4096 × 4 = **220 MB**、ctx 32768 = **537 MB**；而 9B 在 ctx 32768 的余量只有 ~250 MB（free 2.7 GB − 激活 1.25 GB − margin 1.2 GB）。FFN 的 act 还要 n×inter 暂存（13.4k × 12288 × 4 ≈ 660 MB），同样放不下 ⇒ 层主序只能在小 ctx 生效，收益有限，降级为 v0.1.6 独立任务。

## 2. 同一杠杆：`logits_` 占掉了 78% 的每行激活预算
- `activation_bytes()` 的每行预算 1.28 MB 里，`vocab*4`（248k 词表）= **993 KB（78%）**；
- `logits_` 实际按 `rows × vocab × 4` 分配（`runner.cpp:614`）；
- 但 **prefill head 只算 1 行**（`rows = last_row_only ? 1 : n`），serve/CLI 的 prefill 全是 `last_row_only=true`。

⇒ 把 logits 预留改成 **1 行**（`prefill_logits_rows` 默认 1；full-logits 调用惰性扩容），`activation_bytes` 同步改成常数 ⇒ 同样的 VRAM 能放下的 chunk 行数 ×~4，13.4k/16k 文档的 chunk 数减半，**重复整模型反量化从 14 次降到 7–8 次**。

## 3. 顺带修掉的估算 bug（否则会翻车）
只改 logits 时，9B ctx32768 的 chunk 直接长到 4096 行，但 **serve 的 post-load VRAM guard 拒绝启动**（free 930–1002 MiB < 1200 margin）。原因：`activation_bytes()` **漏算 SSM/MLA/MoE 的每行工作缓冲**——
- Qwen3.5-9B（hybrid SSM）：q35_* 共 ~221 KB/行；
- Coder（MLA）：mla_* 共 ~?
- MoE：router_logits + gather staging。

修法：把这些逐项加进 `activation_bytes()`，并按实测（rows=4096 时真实占用 2300 MiB vs 公式 1925 MiB ≈ +18%）给增长检查加 **1.18 cushion**。最终 9B ctx32768 的 chunk 落在 **2048**（真能放下），serve guard 通过。

## 4. 验收（独占 GPU？见 §4.3 的噪声说明）

### 4.1 交错 A/B（9B ctx 32768，prefix cache 关，`ENGINE_PREFILL_LOGITS_ROWS=0` = 旧行为 chunk 1024；默认 = 新 chunk 2048）

| prompt | 旧（r1/r2） | 新（r1/r2） | 均值变化 |
|---|---|---|---|
| 5.3k | 4862 / 4950 ms | 5420 / 4385 ms | ≈ 0%（首轮噪声） |
| 11.0k | 14210 / 12069 ms | 14035 / 11405 ms | **−3.2%** |
| 16.2k | 26344 / 21610 ms | 24201 / 20870 ms | **−6.0%** |

### 4.2 chunk 决策
```
old: [ctx] prefill chunk grown 512 -> 1024 rows (act=1452 MiB)
new: [ctx] prefill chunk grown 512 -> 2048 rows (act=964 MiB)
```
（新估算更诚实：2048 行的真实占用反而比旧的 1024 行估算小，因为 logits 不再按行放大。）

### 4.3 为什么不是 token 级 bit-exact
chunk 大小 1024→2048 改变了每个 GEMM 的 M（cuBLAS 选核/累加顺序随形状变）⇒ 数学不变、浮点舍入可能变。验收口径与上次 rows 128→512/1024 一致：greedy 序列对比 + gen_regression + smoke + 五套。**实测 greedy 序列**见 §5。
另注：本窗口 GPU 有其它进程（~1 GB / 44% util），A/B 的交错设计抵消系统性漂移，但单轮波动可达 ~20%（上表已给出 r1/r2 两轮原始值）。

### 4.4 五套验证（同一二进制）

| 套件 | 结果 |
|---|---|
| `verify.sh` | **ALL GREEN** |
| `smoke_models.sh` | **exit 0**：40716 / 185 / 2614 / 2614 |
| `gen_regression.sh` | **GREEN 10/10** |
| `verify_generality.sh` | **GREEN 26/26** |
| `verify_serve.sh` | **GREEN 22/22**（注：新建 worktree 的 build 目录缺 `webui/index.html` 资产时该套件会失败 1 例——那是构建产物缺失，不是本改动；补上资产后 22/22） |

## 5. 结果汇总

| 项 | 结果 |
|---|---|
| greedy 序列（同 prompt，旧 chunk 1024 vs 新 chunk 2048） | **完全相同**（`2614 513 5081 5576 4602`）——chunk 变大只改 cuBLAS M，未翻转贪心 token |
| 11.0k prompt | **−3.2%**（交错 2 轮均值） |
| 16.2k prompt | **−6.0%**（交错 2 轮均值） |
| 5.3k prompt | ≈0（chunk 数变化不大） |
| chunk 决策 | 9B ctx32768：1024 → **2048**（`act` 估算从 1452 MiB 降到 964 MiB） |

**为什么 16.2k 收益大于 11.0k**：chunk 1024→2048 让 16.2k 从 16 块变 8 块（省 8 次整模型 dequant + 8 次 per-chunk 固定开销），而 11.0k 从 11 块变 6 块。

## 7. 补丁：chunk 不允许等于整个 ctx（fix(prefill): cap serve chunk below ctx）

`89e8b3a` 之后在 **ctx 4096（launcher 默认）** 上 chunk 会长到 4096 = ctx，带来三个问题：
1. activation 按全部位置分配 → 实测启动占用到 ~9.7 GB / 12 GB，余量偏紧；
2. serve 的启动预热（prewarm）在 chunk 接近 ctx 时会被跳过（其条件为 `chunk+16 > ctx`，并且 warm 文本只有 192 token ⇒ 任何 chunk>184 都会跳过）；
3. ctx4096 正是用户最常用的默认配置。

修法（`src/runner.cpp`，对所有调用者生效）：
- 新增 `ctx_chunk_cap = max(512, ctx*3/4)`，**初始 rows_ 与两个增长循环**（pre-load 与装后实测）都以它为上限；
- 顺带修了增长循环的边界 bug：原逻辑在 `rows_ < cap` 时仍会翻倍越过 cap，现改为检查 `rows_*2 <= cap`。

实测（9B，ctx 4096，prefix cache 关）：chunk `4096 → 2048`、post-load free **5922 MiB**（阈值 1200）、2441-token prompt prefill 2.0 s（1210 tok/s）。
长文档（ctx 32768）不受影响：chunk 仍落在 2048，交错 A/B 保持 **−3.5% ~ −6.2%**（11.0k/16.2k）。

> ⚠ 未完成的一处：prewarm 的 warm 文本长度（固定 192 token ⇒ chunk>184 时跳过预热）需要在 `src/serve.cpp` 里改成按 chunk 长度构造并封顶 `min(chunk,1024)`。该文件在我提交时正被另一个 agent 并发修改（工作树有 102 行未提交 diff），为避免覆盖其 WIP，**这条留给该文件的下一位改动者**（补法：把 48 次固定重复改为循环加长到 `min(chunk+8, ctx-8)`，并把预热的 head/tail/graph-warm 切片与条件改用 `pw_chunk = min(chunk,1024)`）。chunk 上限本身已单独生效。

## 6. 边界
- 只影响 prefill 的激活预算与 logits 分配；decode 路径完全不变。
- full-logits 调用（`last_row_only=false`、MTP verify）会惰性扩容 `logits_`，行为与旧版一致（多一次分配与一行日志）。
- prefix-cache 对齐已是 128（server-mode 解耦），chunk 变大**不再有多轮代价**。
- 复现：`ENGINE_PREFILL_LOGITS_ROWS=0` 回旧行为。
