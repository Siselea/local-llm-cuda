# Windows 调度优先级 + auto-margin 换层 — 实测报告（WIN-PRIORITY）

**日期**：2026-11-15（真实物理机：i5-10400F 6C/12T · 32 GB DDR4 · RTX 3060 12 GB · Windows 11）
**动工范围**：`src/config.cpp` / `include/engine/config.h`（3 个调度键 + 4 个 auto-margin 键）、
`src/main.cpp`、`tests/test_gen.cpp`（启动时应用优先级）、`src/cpu_matvec.cpp`（`cpu::Pool` worker
线程优先级 / 亲和性）、`src/runner.cpp`（装载后 auto-margin 决策器）、`docs/CONFIG.md`、
`engine.cfg.example`；新增/扩展工具 `tools/prio_ab.sh`、`tools/ab35_prio.sh`、
`tools/margin35.sh`、`tools/bench_cpu_layer.cpp`（新增 `prio` 交错模式）。
**最终状态**：**四套验证全绿**（见 §7）；数值逐 token bit-exact；默认开启 `win_priority=1`（HIGH）；
35B 默认 **27.43 tok/s**（16 GPU 层 + HIGH），opt-in auto-margin 换层后 **28.14 tok/s**；
`vram_auto_layers` 默认关（§6 的安全性结论）。
**2026-11-15 22:04–22:20 补充**：清场对照（用户关掉 LANDrop/QQ/Edge/浏览器）与 serve 端到端已完成，
见 §10/§11——**清场本身只值 0.2–0.3 ms，优先级在清场后失效；35B serve decode 26.66 tok/s**。

---

## 0. 一句话结论

> **优先级路径做完了、有效但比预期小一个数量级。** 真实物理机 + 桌面后台（LANDrop 稳定占用
> **15.3% of 12 线程 ≈ 1.8 核**）确实在抢 CPU：隔离基准里 HIGH 比 normal 快 **3.7–5.2%**
> （0.655 → 0.631 ms/层，24.9 → 25.8 GB/s），35B 端到端 decode **37.30 → 36.46 ms
> （26.81 → 27.43 tok/s，+2.3%）**，宿主 GEMV 0.717 → 0.684 ms/层（−4.6%），且逐 token bit-exact。
>
> **但“0.96 → 0.62 ms/层 = 8 ms”的前提在当前机器状态下不成立**：本轮用 `ENGINE_DBG_CPUEXEC`
> 的“最后 25 层窗口”实测，引擎内宿主 GEMV 在 normal 下就是 **0.717 ms/层**，同代码隔离基准是
> **0.655–0.661 ms/层**，真实差距只有 **0.06 ms/层 ≈ 1.5 ms/token**。上一轮报告的 0.96 是
> **整轮累计平均**（前 25 层 1.36 ms，被 lazy CUDA-graph capture / page-in 污染，跑 975 层才
> 收敛到 0.73），且当时很可能有并发 agent 负载。因此**优先级最多能拿回 ~1.5 ms，不是 8 ms**；
> **30–32 tok/s 目标未达成**（不换层时 27.4 tok/s，opt-in 换层后 28.0–28.1 tok/s）。
>
> **margin 400 换 1 层是真的**（−0.67 ms，逐 token bit-exact），**但不安全**：装载后 free
> 908–941 MiB 只是表象，per-layer CUDA graph capture 在装载后还会再吃 **~850 MiB**，decode
> 期间实测 free 只剩 **79–82 MiB**（默认 margin 1200 时 592 MiB）——正好贴着“free=0 → 驱动
> 分页 → 3 tok/s”的坑边。因此 auto-margin 决策器实现为 **opt-in、默认关闭**，并在文档里写明
> 原因；这是本轮**主动没有采纳**的收益。
>
> **清场对照（22:04–22:20，LANDrop/QQ/Edge/浏览器已关）**：清场本身只让 35B decode 快
> **0.2–0.3 ms（−0.6%）**，而优先级在清场后从 −0.84 ms 掉到 **−0.20 ms（噪声内）**；
> 清场 high = **26.55 tok/s**，32–34 tok/s 不可达。结论：**优先级是“和桌面后台抢时间片”的补偿，
> 不是算力提升；35B 的瓶颈不在 CPU 时间片**（隔离 CPU 层清场后仍有 −5.4% 的优先级收益，端到端却
> 只剩 −0.5%，说明被 GPU/sync 流水吃掉）。

---

## 1. 任务前提的修正（重要）

上一轮给出的 35B decode 分解把最大单点定为“宿主 GEMV 引擎内 0.96 vs 隔离 0.62 ms/层，差 0.34
ms/层 × 24 = 8 ms”。本轮先复测这个前提：

| 测量 | 上一轮报告 | 本轮实测（同一台真机，LANDrop 15% 背景负载） |
|---|---|---|
| 引擎内宿主 GEMV（`ENGINE_DBG_CPUEXEC`） | 0.96 ms/层（累计平均） | **0.717 ms/层**（normal，最后 25 层窗口中位）；累计平均收敛值 0.73 |
| 同代码隔离基准（`bench_cpu_layer layer 24`，真 DRAM，9 线程） | 0.62–0.65 ms/层 | **0.655–0.661 ms/层**（normal；HIGH 下 0.631） |
| 差距 | 0.34 ms/层（8 ms/token） | **0.06 ms/层（~1.5 ms/token）** |

工程内首次出现的累计平均曲线（一次 35B n=40 运行，`ENGINE_DBG_CPUEXEC`）：

```
前 25 层累计平均 1.36 ms/层 → 50 层 1.04 → 75 层 0.93 → … → 950 层 0.73（最后 25 层窗口 0.730）
```

累计平均被开头的 lazy CUDA graph capture / 权重 page-in 拖高。上一轮引用的 0.96 只能是
**整轮累计 + 并发负载**的产物，不代表稳态。**结论：宿主 GEMV 已经贴着隔离基准，剩余可抢回
的只有 0.06 ms/层量级（≈1.5 ms/token）**，这一点决定了本轮所有数字的量级。

为此给 `ENGINE_DBG_CPUEXEC` 增加一项输出（纯诊断，默认关闭）：
`[cpuexec] batched host matvec avg X ms over N layers (last 25: Y ms)`，窗口值是本报告 A/B 的
主指标；累计平均保留以便和旧报告对齐。

---

## 2. 实现（win_priority）

### 2.1 配置面（注册进统一配置系统：`include/engine/config.h` + `src/config.cpp` + `docs/CONFIG.md` §3.5.1）

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `win_priority` | `--win-priority <n>` | `ENGINE_WIN_PRIORITY` | int [0,2] | **`1`** | 进程优先级类：`1`=`HIGH_PRIORITY_CLASS`（默认）、`2`=`ABOVE_NORMAL`、`0`=`NORMAL`（关闭/A-B）。**不用 REALTIME**（会饿死输入/IO/WDDM 提交线程且需额外权限） |
| `win_thread_priority` | `--win-thread-priority <n>` | `ENGINE_WIN_THREAD_PRIORITY` | int [-1,2] | `-1` | CPU 池 worker 线程的**相对**优先级：`-1`=auto（`win_priority>0` 时 `THREAD_PRIORITY_HIGHEST`，否则继承）、`0`=继承、`1`=ABOVE_NORMAL、`2`=HIGHEST（base+2） |
| `win_affinity` | `--win-affinity <n>` | `ENGINE_WIN_AFFINITY` | int [0,1] | `0` | 可选：worker i 硬绑到“先物理核、后 SMT 兄弟”顺序的第 i 个逻辑 CPU。实测与不绑持平（§3），默认关 |

生效位置（都在**加载模型之前**，所以 serve/run/test 工具一致受益）：

- `config::apply_win_priority()`（`src/config.cpp` 实现，`config.h` 声明）：`SetPriorityClass`。
  即使 `win_priority=0` 也显式设回 `NORMAL`——子进程会继承父进程的优先级类，不显式设置的话
  “从 HIGH shell 里启动的 benchmark”就没有真正的 normal 对照臂。启动日志输出一行证据：
  `[win] process priority class = HIGH (win_priority=1)`。
- 调用点：`src/main.cpp`（`run` 和 `serve` 共用同一个入口，**serve 因此同样生效**）、
  `tests/test_gen.cpp`（A/B 脚本用）、`tools/bench_cpu_layer.cpp`（隔离基准用）。
- `cpu::Pool` 构造（`src/cpu_matvec.cpp`）：每个 worker 在启动时
  `SetThreadPriority(THREAD_PRIORITY_HIGHEST)`（默认 `win_priority=1` → 13+2=**15**），
  可选 `SetThreadAffinityMask`。`ENGINE_DBG_SCHED=1` 打印最终的 worker 相对优先级/亲和性。

### 2.2 为什么不默认 REALTIME / 为什么主线程不提权

- REALTIME（24）会饿死鼠标/IO/WDDM 提交线程，且需要 `SeIncreaseBasePriorityPrivilege`；HIGH（13）
  仍低于 input 优先级带，实测无副作用（35B/9B/Coder 全部输出 bit-exact、验证全绿）。
- 隔离基准显示收益来自**进程类**而不是某个线程：主线程单独 HIGHEST、worker 单独 HIGHEST 都无效，
  甚至 worker 单独提权（10）在跨进程 A/B 里 **+2.9%（变慢）**——主线程作为并行区参与者反而成了
  拖尾。故默认策略 = 进程类 HIGH + 全体 worker HIGHEST（一并抬）。

### 2.3 bit-exact 论证

优先级只改调度顺序、不改任何计算。证据：

- 35B 三次优先级臂 × 5 reps 的 `gen:` 序列 md5 全部相同（`e9f0a5a6`），`topk: 2614(...)` 相同；
- 隔离基准 `chk` 累加和在三档优先级、六个 arm 下完全相同（`-10539247861760.000000`）；
- 9B / Coder 的 `topk:` 相同。

---

## 3. 隔离基准：三档优先级 + 主/worker 拆分（纯 CPU，不占 GPU）

工具：`tools/bench_cpu_layer.cpp`（24 个独立 DRAM 副本 × 15.56 MB 轮转，工作集 373 MB ≫ 12 MB
L3，9 线程生产内核 `cpu::moe_forward_mt`），新增 `prio` 模式：**单进程内 6 个 arm 交错**
（每轮反转顺序），消掉分钟级漂移；每 arm 每轮 200 次迭代（~130 ms）。

### 3.1 单进程交错（3 次独立运行，各 5 轮 × 6 arm × 200 iters，取“3 次运行中位数”的中位）

| arm | 每次运行的 median (ms/层) | 三次中位 | vs normal | GB/s |
|---|---|---|---|---|
| `normal`（进程 NORMAL，全继承） | 0.6600 / 0.6611 / 0.6677 | **0.6611** | — | 24.7 |
| `proc-high`（进程 HIGH，线程继承） | 0.6335 / 0.6325 / 0.6298 | **0.6325** | **−4.3%** | 25.8 |
| **`high+workers-highest`（= 生产默认）** | 0.6254 / 0.6320 / 0.6265 | **0.6265** | **−5.2%** | 26.0 |
| `above-normal+workers-highest` | 0.6425 / 0.6416 / 0.6522 | 0.6425 | −2.8% | 25.4 |
| `workers-highest-only`（进程 normal） | 0.6568 / 0.6697 / 0.6492 | 0.6568 | −0.6% | 24.8 |
| `main-highest-only`（进程 normal） | 0.6599 / 0.6598 / 0.6544 | 0.6598 | −0.2% | 24.7 |

**读法**：收益来自“把整个进程抬到 HIGH”；单独抬主线程或单独抬 worker 线程都无效；ABOVE_NORMAL
只有 HIGH 的 ~一半。这与“真实桌面后台在抢时间片”的假设一致——只有进程级优先级能改变调度结果。

### 3.2 生产代码路径（跨进程，`ENGINE_WIN_PRIORITY`/`ENGINE_WIN_THREAD_PRIORITY` 走
`apply_win_priority()` + `cpu::Pool`），`tools/prio_ab.sh`，5 reps 交错轮换：

| arm | median (ms/层) | min | vs normal |
|---|---|---|---|
| `normal` | 0.6550 | 0.6480 | — |
| `high` | **0.6310** | 0.6270 | **−3.7%** |
| `above-normal` | 0.6470 | 0.6400 | −1.2% |
| `high+workers-inherit` | 0.6350 | 0.6320 | −3.1% |
| `high+workers-highest`（生产默认） | 0.6380 | 0.6280 | −2.6% |
| `normal+workers-highest` | 0.6740 | 0.6580 | **+2.9%（变慢）** |

跨进程噪声更大（±1%），但方向与单进程交错一致：进程 HIGH ≈ −3~−5%，worker 单独提权无益甚至有害。
（GB/s 由 bench 自身口径 16.31 MB/层换算；normal 的 24.7 GB/s 与 `chk` 行打印的 24.4–24.7 一致。）

### 3.3 背景负载记录（同期实测）

- `LANDrop ×2` 进程：合计 **15.3% of 12 线程（≈1.8 核）**，CPU 时间各 13.6 万秒，长期稳定占用；
- 机器空闲总占用 28–30%；`DeepSeek Harness` agent 进程 ~3.5%；
- 35B decode 运行期间整机总 CPU 占用 37–41%（`typeperf \Processor(_Total)\% Processor Time`）；
- 35B 运行期间 GPU：util 峰值 100%（power 采样 100 ms 粒度偶尔回落到 42–54%），power 峰值
  72.5–83.8 W（RTX 3060 上限 170 W，decode 是 CPU 等待型负载，功耗不是瓶颈）。

---

## 4. 35B 端到端 A/B（真实环境，未清场）

脚本 `tools/ab35_prio.sh`：每个 arm 单独进程（优先级是进程属性），n=40 → 39 个计时步（`ENGINE_DEC_SKIP=2`），
`ENGINE_TOPK=1 ENGINE_DBG_CPUEXEC=1`；每 rep 反转 arm 顺序。13 次运行交错。

| 轮次 | pre（旧二进制） | normal（新，prio=0） | high（新，prio=1） |
|---|---|---|---|
| r1 | 37.78 | 37.35 | **36.49** |
| r2 | 37.22 | 37.02 | **37.14** |
| r3 | 37.45 | 37.66 | **36.24** |
| r4 | — | 37.30 | **36.38** |
| r5 | — | 37.18 | **36.46** |
| **中位** | **37.450 ms（26.70 tok/s）** | **37.300 ms（26.81 tok/s）** | **36.460 ms（27.43 tok/s）** |

- **normal → high：−0.84 ms（−2.25%），26.81 → 27.43 tok/s**；即 **35B 最终 tok/s = 27.43**（默认 16 个 GPU 层 + HIGH）；opt-in auto-margin 换层后 **28.14–28.31**；
- pre → normal：−0.15 ms（−0.4%，噪声内）→ **无回退**；
- 宿主 GEMV（最后 25 层窗口，逐 run）：normal `0.737/0.815/0.699/0.710/0.717`（中位 **0.717**），
  high `0.676/0.692/0.684/0.699/0.677`（中位 **0.684**）→ **−4.6%**；
- **逐 token bit-exact**：13 次运行 `topk: 2614(-2.582)…` 与 `gen:` md5（`e9f0a5a6`）全部一致。

端到端只体现 −0.84 ms 而不是 24 ×（0.717−0.684）= 0.79 ms 之外的更多，是因为 35B decode 是
CPU/GPU 流水（GPU 层与头部工作可以部分重叠），宿主 GEMV 的改善不能 100% 落到墙钟上。

**结论：HIGH 优先级在 35B 上真实有效但幅度 ~2.3%；把 GEMV 推向 0.62 的目标只走完 0.717→0.684，
30–32 tok/s（116–124%）不可达——物理上限就是宿主 GEMV 的算力（见 CPU-REPACK-REPORT：内核
24.8 GB/s 硬上限）。**

## 5. 9B / Coder 回归（硬性约束）

| 模型 | normal | HIGH | 约束 | 结论 |
|---|---|---|---|---|
| Qwen3.5-9B decode | 21.45 / 21.46 ms（**46.7 tok/s**） | 21.41 / 21.45 ms（**46.6 tok/s**） | ≥44 tok/s | ✅ 通过；dense 模型不用 CPU 池，优先级对它是纯噪声 |
| Coder-V2-Lite decode | 41.80 / 40.98 ms（24.2 tok/s） | 40.56 / 40.39 ms（**24.7 tok/s**） | ≥23 tok/s | ✅ 通过；−2.2%（同为 CPU 卸载模型，方向一致） |

0.5B（`smoke_models.sh` 的 40716）与全部真实模型 top-1 在优先级臂下不变（见 §7）。

---

## 6. `--vram-margin 400` 换 1 层：收益真实，但**不安全**

脚本 `tools/margin35.sh`（HIGH 优先级、n=40、3 reps 交错、每次运行用 nvidia-smi 每 500 ms 采样
`power.draw,utilization.gpu,memory.free`）：

| arm | 规划 GPU 层 | 装载后 free（实测） | **decode 期间最低 free（采样）** | decode 中位 | tok/s | gen md5 |
|---|---|---|---|---|---|---|
| margin=1200（默认） | 15 → 实测 16 | 1423 / 1445 / 1449 MiB | **592 / 592 / 594 MiB** | 36.21 ms | 27.62 | e9f0a5a6 |
| margin=400 | 17 | 911 / 941 / 908 MiB | **82 / 81 / 79 MiB** | 35.54 ms | 28.14 | e9f0a5a6 |

- **收益**：+1 个 GPU 驻留 MoE 层 → decode **−0.67 ms（−1.85%）**，与上一轮“同会话 −0.64 ms”一致；
  逐 token bit-exact（同一 `gen:` md5），说明“专家放 CPU 还是 GPU”在这套实现里逐位一致。
- **风险**：装载后 free 908–941 MiB 看起来满足“≥900”的安全线，但 **per-layer CUDA graph capture
  发生在装载后**，还会再吃掉 ~850 MiB（1200 臂：1445 → 592；400 臂：941 → 82）。decode 期间
  只剩 **79–82 MiB** 空闲——踩过的“free=0 → 驱动分页 → 3 tok/s”坑就在一步之外。这次没有分页
  （decode 仍 28 tok/s），但这是运气，不是余量。
- **结论：不默认采纳 margin=400 的换层。** 安全边界应表达为“**装载后 free ≥ ~1750 MiB**”
  （900 目标 + ~850 graph 预留），而不是 900；35B 默认 margin 1200 的装载后 1445 MiB 正好是
  “graph 预留后仍剩 592 MiB”的现实配置。

### 6.1 auto-margin 决策器（已实现，**opt-in，默认关**）

用户要求把“降 margin 换 1 层”做成决策器而非手动改 margin。实现方式避开了“猜 margin”：
**装载后用实测 free 决策**——`DenseRunner::apply_auto_vram_layers()`（`src/runner.cpp`）在
`check_post_load_vram()` 之后、CUDA graph capture 之前运行：

1. 只在“自动放置 + 存在 ≥2 个 CPU 专家层 + 用户没有显式给 `--vram-margin`/`--gpu-layers`/
   `--vram-budget`”时启用（显式旋钮是用户的契约，绝不二次猜测）；
2. 逐层把下一个 CPU MoE 层的专家权重从 host 量化字节上传进显存（`QTensor` + 刷新该层的
   `all_gp_/all_up_/all_dp_` 设备指针数组），把 `cpu_begin` 设为全部专家在 GPU，保持连续前缀；
3. 每抬一层都要求 **实测 free − 本层字节 ≥ `vram_margin_floor_mb`（默认 900）**；
4. 打印决策与数字，日志可直接审计。

实测（35B，`ENGINE_VRAM_AUTO_LAYERS=1`，其它默认）：

```
[placement] auto-vram: promoted MoE layer blk.16 to the GPU (498 MiB, free 1447 -> 935 MiB, floor 900 MiB) gpu_moe_layers -> 17
[vram] check OK after auto-vram: 934 MiB free >= 900 MiB floor (promoted 1 layer)
decode 36.00 -> 35.34 ms（−0.66 ms），gen md5 e9f0a5a6（bit-exact）
```

**但默认仍然关闭**：它和 margin=400 是同一个终局（decode 期 free ~82 MiB），因为 floor 900 是在
“graph capture 之前”测量的，不含 ~850 MiB 的 graph 预留。要默认开启，floor 必须抬到
~1750 MiB，而 35B 装载后只有 1445 MiB ⇒ 0 层可换 ⇒ 净收益为 0。**所以在当前实现下这个决策器
不能安全地默认生效**，保留为 `--vram-auto-layers`（opt-in，明确打印 floor 与每层收益），
把选择权交给确认过该模型/ctx graph 预留的用户。

> 附带验证：**graph capture 的 ~850 MiB 是装载后最大的隐性预留**。若以后要安全地多放层，
> 应该把 “capture 后实测 free” 纳入决策（例如把 auto-margin 移到 capture 之后并让被抬升层走
> 无 graph 路径），这是明确的后续项而不是本轮的低风险改动。

---

## 7. 四套验证（全绿）

| 套件 | 结果 | 证据 |
|---|---|---|
| `tools/verify.sh` | **ALL GREEN**（RC=0；含 fixtures、Python unittest、G0 IR parity、M3 CUDA/cuBLAS、M2 plan parity、真实模型 parse/diff） | `build/allsuites.log` |
| `tools/smoke_models.sh` | qwen2.5-0.5b **40716**（decode 4.97–5.12 ms = **195–201 tok/s**）/ Coder-V2-Lite **185** / Qwen3.5-9B **2614** / Qwen3.6-35B-A3B **2614**（expected 全部命中，RC=0） | 同上 |
| `tools/gen_regression.sh` | **10/10 PASS**（9B×5 + 35B×3 + Coder + 0.5B，与 llama.cpp greedy 逐 id 对照，RC=0） | 同上 |
| `tools/verify_generality.sh` | **26/26 PASS · GENERALITY GREEN**（配置层/校准缓存/自适应 margin sweep 600>2400/**post-load margin enforced**/explicit gpu-layers/优雅失败/4 个真实模型 parse） | 同上 |
| `bash build.sh` | 20 个二进制全部链接成功（无 error） | — |

四套验证是在 **`win_priority=1`（默认 HIGH）** 下跑的——数值全部不变，进一步确认优先级改动
不触碰计算路径。注意：跑完四套后只改过一句 `--vram-auto-layers` 的帮助文案（纯 printf 字符串），
随后重跑 `verify_generality`（GREEN 26/26）与 `smoke_models`（RC=0、四个 top-1 不变）复核最终二进制
（`build/allsuites-final.log`）；`verify.sh` / `gen_regression` 不涉及该字符串。默认值的取舍：数据支持 HIGH（隔离 −3.7~−5.2%、35B −2.3%、Coder −2.2%、9B 中性、
零副作用、bit-exact），因此默认设 `1`；想回退历史行为用 `--win-priority 0` / `ENGINE_WIN_PRIORITY=0`。

---

## 8. 失败 / 未达成（如实）

1. **30–32 tok/s（150% 目标）没有达到**，本轮也没有接近：优先级把 35B 从 26.81 推到 27.43 tok/s，
   opt-in 换层再推到 28.14。根因是任务书里“宿主 GEMV 0.96 vs 隔离 0.62”的前提在当前机器状态下
   不成立（真实差距 0.06 ms/层），优先级只能拿回其中一半左右。**宿主 GEMV 的算力上限（24.8 GB/s）
   没有变，CPU 层路径没有新的物理空间。**
2. **margin=400 换层没有默认采纳**：−0.67 ms 是真的，但 decode 期 free 82 MiB 不安全（§6）。
   auto-margin 决策器做成了 opt-in，默认收益为 0。
3. **“清场对照”（B 环境）没有做**：用户明确要求不碰任何非本项目进程、不做需要用户配合的清场
   窗口。所有 35B 数据都是在当前环境（LANDrop 15.3% 背景负载）下测的；空闲机器上绝对数字会更好
   （上一轮隔离基准 0.62 就是那种状态），但**相对收益**（优先级 3–5%）来自“和后台争抢”，清场后
   反而会变小甚至归零——这正是“优先级是桌面负载的补偿而不是算力提升”的应有结论。
4. `workers-highest-only`（只抬 worker 不提进程）实测 **+2.9% 变慢**；`main-highest-only` 无效。
5. **serve 路径没有做 HTTP 端到端 tok/s 实测**（7030 服务已按用户要求关闭，不在本轮重启）。
   优先级生效由同一入口 + 启动日志 `[win] process priority class = HIGH` 保证，但“远程 API 用户
   实际受益多少”待用户回家后开服务补测。
6. `win_affinity`（绑核）没有单独做完整 A/B：本机与上轮结论一致（绑核在真实层基准里是噪声内），
   保持 `0` 默认关闭的 opt-in 开关；隔离基准的 `read` 模式里 compact/smt 绑核历史数据也未复测。

## 9. 后续项（用户回家/确认后）

- ~~B 环境（清场）对照~~ → **已完成（§10）**：清场只值 0.2–0.3 ms（−0.6%），优先级在清场后失效。
- ~~serve 端到端~~ → **已完成（§11）**：HIGH 已在 serve 确认（日志 + PriorityClass），
  9B 44.69 / 35B 26.66 tok/s（非流式 decode），流式 TTFB 179 / 547 ms。
- 剩余：长时 serve 的稳定性/无副作用观察（IO/输入延迟）待用户实际使用验证。
- **若接受 82 MiB 风险**：`--vram-auto-layers`（或 `--vram-margin 400`）可拿 +0.67 ms / +0.5 tok/s；
  建议只在确认该模型/ctx 的 graph 预留后开。
- **真正往 30+ tok/s 走的唯一方向**仍然是宿主 GEMV 算力本身（CPU-REPACK 已证明 24.8 GB/s 是内核
  上限；下一步只能换算法/指令级重写，或把更多层放 GPU —— 后者受 12 GB 显存约束）。

---

## 10. 清场对照（2026-11-15 22:04–22:20，用户已关后台程序）

用户到家后关闭了 LANDrop / QQ / Edge / 浏览器（Steam 保留，实测仅 0.04% CPU）。这段窗口里重跑了
35B A/B（每臂 5 次、normal/high 交替），并做了两组控制实验。**之前那批 22:01 的 data 被两个原因
污染，已单独标记、不进入任何结论**：① 我旧版脚本的 per-run `cmd /c typeperf` 采样器泄漏了它的
typeperf 子进程（每个 ~8% CPU，6 个 run 后 ~50% 负载）；② ux-launcher 当时在跑 launcher+engine
（同 GPU、占 ~150 MiB）。污染证据：`build/ab35-prio-CONTAMINATED.log` + `build/contaminated/`。
修好的脚本 `tools/ab35_prio.sh` 已去掉 per-run CPU 采样（改为批量前后各采一次），并给每次 run
记录 `gpu_before`。

### 10.1 三列对比（35B，n=40，5 reps 中位）

| 指标 | 旧环境 normal（LANDrop 15%） | 旧环境 high | **清场 normal** | **清场 high** |
|---|---|---|---|---|
| decode ms/token | 37.30 | 36.46 | **37.86** | **37.66** |
| tok/s | 26.81 | 27.43 | **26.41** | **26.55** |
| 宿主 GEMV（last 25） | 0.717 | 0.684 | **0.739** | **0.745** |

清场后环境证据：整机 CPU 6.0 / 6.8 / 5.2 / 8.0 %（旧环境 28–30 %）；每次 run 前 GPU 仅
829–890 MiB 占用（无其它模型）；GPU 全程 100 % util、80–84 W；逐 token bit-exact（`gen` md5
`e9f0a5a6`）不变。

### 10.2 控制实验：清场到底贡献多少？

- **控制 1（同一旧 pre 二进制跨环境）**：旧环境 pre 中位 37.45 ms → 清场 37.13 / 37.33 ms，
  **清场贡献 ≈ −0.22 ms（−0.6 %）**。
- **控制 2（隔离 CPU 层基准，清场重测）**：normal 0.6995 ms/层 → high+workers-highest
  0.6616 ms/层（**−5.4 %**，纯 CPU 层的高优先级收益在清场环境里依然存在且比旧环境更大）。
- 但同一批数据里 35B 端到端 normal→high 只有 −0.20 ms（−0.53 %，噪声内）。

### 10.3 结论

1. **“后台程序抢走 11 % CPU”对 35B decode 墙钟的影响只有 0.2–0.3 ms（<1%）**——35B decode 的
   关键路径是 GPU/sync 流水，不是 CPU 时间片；清场不改变这一点。
2. **优先级只在“有同优先级竞争者”时有效**：旧环境 −0.84 ms（−2.25 %），清场后 −0.20 ms（噪声内）。
   这与“补偿后台争抢”的定性一致。默认保留 `win_priority=1` 的理由是它在隔离 CPU 层仍有 −5 %
   且零副作用，但**不要把它当算力优化**。
3. 清场 high = **26.55 tok/s**，旧环境 high = 27.43：清场后的绝对数反而略低 1 ms，属于跨时段
   漂移（同二进制跨批次 36.2–38.4 ms 的范围内），不能归因于清场；两个方向都远低于 30–32 tok/s
   目标。**124–132 % 目标不可达**。

---

## 11. serve 端到端（HTTP / OpenAI API，端口 7040）

在同一个清场窗口内测。启动命令：

```bash
./build/engine.exe serve <model.gguf> --port 7040 --ctx 4096 --yes
```

**优先级确认（两层证据）**：启动日志 `[win] process priority class = HIGH (win_priority=1)`；
PowerShell `Get-Process engine | Select PriorityClass` → **High**（15 线程）。serve 与 run 共用
`main.cpp` 入口，所以 `serve` 确实继承了 HIGH。

请求：`tools/serve_bench.py`，每类 3 次取中位；temperature=0，max_tokens=256；长 prompt 用
`/tokenize` 校准到 1501 tokens。

| 指标（3 次中位） | 9B | 35B |
|---|---|---|
| 非流式 decode | **44.69 tok/s**（wall 43.07） | **26.66 tok/s**（wall 25.22） |
| 非流式首 token（prefill） | 167.7 ms（首请求冷 553 ms） | 535.6 ms（首请求冷 973 ms） |
| 非流式端到端 wall（256 tok） | 5944 ms | 10150 ms |
| 流式 TTFB | **179.3 ms** | **547.0 ms** |
| 流式 decode | 44.42 tok/s（wall 43.08） | 26.42 tok/s（wall 25.00） |
| 长 prompt（1501 tok）prefill | **1095.4 tok/s**（1370 ms） | **55.6 tok/s**（27.0 s） |

读法：

- serve 的 35B decode（26.66）与 test_gen 直测（26.4–26.6）一致，**HTTP/JSON 开销 ~1–2 %**；
  流式 TTFB 比非流式 prefill 略高 ~10–12 ms（SSE 首包 + JSON 包装）。
- 35B 长 prompt prefill 只有 **55.6 tok/s（27 s）**：MoE 专家层在 CPU 上要逐 token 算 8 个专家，
  prefill 是它的算力瓶颈（9B dense 是 1095 tok/s）。这是 35B 的已知短板，不是本轮的优先级问题。
- **重要副作用发现**：35B serve 在默认 `margin=1200 / ctx=4096` 下装载后 free 1734 MiB，
  per-layer CUDA graph capture 后只剩 **74 MiB**，仍稳定跑 26.6 tok/s。这意味着
  `vram_auto_layers` 若在 serve ctx=4096 下自动 promote（1734−498=1236 MiB）会在 graph capture
  时 OOM（capture 实测要吃 ~1660 MiB）——**默认关是唯一安全选择**，比 §6 的 82 MiB 论证更硬。

---

## 附录 A：复现命令

```bash
# 隔离基准三档优先级（纯 CPU，不占 GPU；单进程交错 6 arms）
./build/bench_cpu_layer.exe prio 24 200 5
# 生产路径跨进程 A/B（5 reps 交错）
REPS=5 ITERS=200 bash tools/prio_ab.sh
# 35B 端到端（HIGH vs normal vs 旧二进制；13 次运行交错）
REPS=5 PRE_REPS=3 bash tools/ab35_prio.sh
# margin 换层 + decode 期 free 采样
REPS=3 bash tools/margin35.sh
# auto-margin 决策器（opt-in）
ENGINE_VRAM_AUTO_LAYERS=1 ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 \
  ./build/test_gen.exe <35B.gguf> 760 40
# serve 端到端（先起服务，再跑 bench：非流式/流式/长 prompt 各 3 次中位）
./build/engine.exe serve <model.gguf> --port 7040 --ctx 4096 --yes &
python tools/serve_bench.py --port 7040 --tag 35b --reps 3
# 四套验证
bash tools/verify.sh && bash tools/smoke_models.sh && \
  bash tools/gen_regression.sh && bash tools/verify_generality.sh
```

## 附录 B：证据文件

- `build/ab35-prio.log` + `build/o-{pre,normal,high}-r*.{out,err}` — 35B 端到端 13 次运行原始数据
  （含 `[cpuexec] … (last 25: X ms)`、`[win]` 行、功耗/CPU 采样）
- `build/prio-r{1,2,3}.txt` + `build/prio-r*.err` — 单进程交错 6 arm × 5 轮原始数据
- `build/prio-ab.log` — 生产路径跨进程 6 arm × 5 reps 原始数据
- `build/margin35.log` + `build/m-*.err` + `build/pwrm-*.log` — margin 实验（含 nvidia-smi free 轨迹）
- `build/av-def.err` / `build/av-on.err` — auto-margin 决策器 default / opt-in 两次运行日志
- `build/allsuites.log` — 四套验证完整输出
- `build/ab35-prio.log`（清场批次，22:04–22:14）+ `build/o-{normal,high}-r*.{out,err}` + `build/pwr-*.log`
- `build/ctl-pre-{1,2}.*` — 控制实验 1（旧二进制清场重测）
- `build/serve-9b.log` / `build/serve-35b.log` / `build/serve-bench-9b.json` / `build/serve-bench-35b.json`
  — serve 端到端（`[win]` 行、placement、请求日志、timings）
- `build/ab35-prio-CONTAMINATED.log` + `build/contaminated/` — **污染批次（不进入结论）**：
  旧脚本 typeperf 泄漏 + ux-launcher 同时占 GPU
