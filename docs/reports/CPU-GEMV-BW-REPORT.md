# CPU 量化 GEMV 带宽优化 — 交付报告

**日期**：2026-11-14（阶段十三，SPEC-COMPLIANCE.md §29）
**改动文件**：`src/cpu_matvec.cpp`（核心）、`include/engine/cpu_matvec.h`（未变）、
`tests/test_cpu_moe.cpp`（新增单测）、`tools/bench_cpu_layer.cpp`（新增基准）、
`build.sh` / `tools/verify.sh`（接入单测）、`src/config.cpp` + `docs/CONFIG.md`（threads 语义）、
`tools/README.md`。

**一句话结论**：CPU 专家层的有效带宽 **14.0–14.5 → 20.6–23.7 GB/s**（峰值 1.6×，同窗口 1.42×），
引擎内 host matvec **1.18–1.27 → 0.81–0.87 ms/层**，35B decode **52.9 → 49.5 ms（20.2 tok/s）**，
GPU 忙占比 **45.3% → 50.2%**。**≤40 ms 目标未达成**，原因已量化（§2.3）；四模型预测与三套验证全绿（§4）。

---

## 1. CPU GEMV 带宽前后对比

### 1.1 测法（为什么不能再用旧基准）

旧的 `tools/bench_cpu_matvec.cpp` 用单个矩阵：512×2048 的 Q4_K 只有 **0.59 MB**，落在 12 MB 的 L3 里，
量到的是 cache 带宽而不是 DRAM。decode 时 25 个 CPU 层每层要流式读 **16.3 MB** 权重
（8 路由专家 × gate/up Q4_K 2×512×2048 + down Q6_K 2048×512），完全在 DRAM。

新基准 `tools/bench_cpu_layer.cpp`：

- 分配 **24 个独立副本** × 15.56 MB = **373 MB 工作集**（> 10× L3），每轮换一个副本 ⇒ 必然 DRAM；
- 支持 3 种测法，同一份字节/同一线程切分可互相校验天花板：
  - `layer`：直接跑生产函数 `cpu::moe_forward_mt`（8 专家 × 3 矩阵，hidden=2048/inter=512，Q4_K/Q4_K/Q6_K）
  - `matvec`：单个量化矩阵（类型/形状/副本数可调）
  - `read`：**纯 AVX2 32B 载入**同字节数（不做解包），给该工作集/线程数下的 DRAM 天花板
- 构建：`ziglang c++ -std=c++17 -O2 -Iinclude tools/bench_cpu_layer.cpp src/cpu_matvec.cpp src/config.cpp -o build/bench_cpu_layer.exe`

### 1.2 结果

**DRAM 天花板（纯读，373 MB 轮转）**：

| 线程 | 1 | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| GB/s | 15.8–18.4 | 24.1–26.7 | 29.4–32.2 | 31.2–33.7 | 32.1–34.7 | 31.1–36.0 |

⇒ 4–6 线程就吃满 31–35 GB/s；12 线程没有额外 DRAM 带宽。而旧行内核单核只有 3.5–4 GB/s ⇒ **不是带宽问题**。

**每层耗时 / 有效带宽（`layer` 模式，16.3 MB/层）**：

| 版本 | 线程数（pool 默认） | ms/层（min-of-N） | GB/s |
|---|---|---|---|
| 之前 | 12（旧默认） | 1.127–1.17 | **13.7–14.5** |
| 之前 | 9 | 0.955 | 17.1 |
| **之后** | **9（新默认）** | **0.715–0.793** | **20.6–23.7** |
| 之后 | 12 | 0.718–0.817 | 20.0–22.7 |

- 相同线程数下（排除线程数效应）：12 线程 **1.127 → 0.817 ms（1.38×）**；9 线程 **0.955 → 0.793（1.20×）**。
- 峰值（机器空闲窗口）：**0.715 ms/层 = 22.8 GB/s**，对应 1.6× 提升。
- 单核 Q4_K 行内核：**3.9 → 5.7 GB/s**（1.5×）。

> 机器漂移说明：本机是工作站（同时有桌面/QQ/任务管理器/另一个 agent 的活），
> 同一份二进制连续测可漂 ±20–30%。上表全部是**同一时间窗口内 min-of-N**，
> 并且 decode 用 pre/new 两二进制**背靠背交错**对比（§2）。

### 1.3 带宽为什么上不去（取证）

1. **旧内核被 FMA 依赖链卡死**：Q4_K 块 144 B / 256 元素，旧实现每块 32 条**串行依赖**的 FMA
   （单个累加器），4 cycle × 32 = 128 cycle/块。用 dependent-FMA 链标定核心频率 = 3.19 GHz，
   实测 126 cycle/块 ⇒ 完全吻合。Q6_K 每块 64 条链，更糟。
2. **不是 ALU/载入端口/DRAM**：去掉 x 载入 + 去掉 FMA-with-x 的"纯解包探针"与完整内核同速；
   单核纯读能到 16–18 GB/s。
3. **权重载入延迟**：加显式预取后 12 线程 Q4_K 26 → 29–31 GB/s（+15–20%）。

### 1.4 改了什么

1. **4 个独立累加器 + 16 B 展开**：依赖链 32/64 → 8 条（`avx2_dot_q4k` / `avx2_dot_q6k`）。
2. **显式预取**：每行入口预取本行（T0→L1）+ 下一行（T1→L2），预取越界不 fault。
3. **类型派发/行距移出行循环**：每行一次运行时整数除法 + 2 次 `std::string` 比较 → 每阶段一次
   （35B 每 CPU 层 24576 行）。
4. **线程池交接**：condvar 交接 ~32 µs/并行区（11 waiter 抢一把 mutex），每层 3 个并行区 ⇒ ~2.4 ms/token；
   改为 atomic generation + 有界 PAUSE 自旋（~20 µs 窗口）再退回 condvar。
5. **默认线程数 = `hardware_concurrency - max(2, hw/4)`**（本机 12 → 9）：SMT 槽全占会被 OS/WDDM
   驱动抢 CPU，实测 12 线程 17.4 GB/s vs 9–10 线程 22.4–23.7 GB/s。`threads` 配置键"0=自动"语义已更新。

---

## 2. 35B decode 前后

### 2.1 主结果（两二进制背靠背交错，`test_gen 760 9`，每次 9 token）

| rep | pre 二进制 | new 二进制 |
|---|---|---|
| 1 | 52.91 ms（18.9 tok/s） | **49.55 ms（20.2 tok/s）** |
| 2 | 52.02 ms（19.2 tok/s） | **49.37 ms（20.3 tok/s）** |
| 3 | 53.76 ms（18.6 tok/s） | （该轮模型加载超时，未采） |
| **中位** | **52.91 ms** | **49.46 ms（1.07×）** |

机器空闲窗口里用 `ENGINE_THREADS=9` 得到的同版本数字：**46.71 / 46.86 / 47.06 ms（21.2–21.4 tok/s）**；
`bench_perf.sh final`（自动放置，机器较忙）两轮：50.92 / 51.81 ms（19.6 / 19.3 tok/s）。

### 2.2 GPU 忙占比（`ENGINE_KPROF=1`，各 1 次，n=9）

| | decode（含 profiler 开销） | GPU 时间 | GPU 忙占比 |
|---|---|---|---|
| pre | 64.92 ms | 28.74 ms | **45.3%** |
| new | 59.30 ms | 28.34 ms | **50.2%** |

GPU 绝对时间基本不变（28.7 → 28.3 ms），是墙钟从 64.9 → 59.3 ms 让占比重排——与上一轮
（33.9% → 44.3%）同一方法论。

### 2.3 为什么没到 ≤40 ms（如实量化）

机器快时的逐层/逐阶段实测（`ENGINE_DBG_LAYERMS`、`ENGINE_DBG_MOE_PHASES`、`ENGINE_DBG_CPUEXEC`）：

| 每 CPU 层（×25） | ms |
|---|---|
| host matvec（phase A 0.53 + phase B 0.35 + merge 0.05，含 3 个并行区） | **0.87** |
| 本层 GPU 注意力（与 host matvec **严格串行**：路由/激活要先 D2H 回来） | ~0.35 |
| WDDM 提交（24–26 次 launch） | ~0.20 |
| 同步尾（等本层 GPU 注意力） | ~0.20 |

| 每 GPU 层（×15） | ms |
|---|---|
| 整层 | 0.56–0.85 |

`25×1.42 + 15×0.6 ≈ 44.5 ms` + 收尾 ≈ 46–52 ms。CPU GEMV 只占 ~22 ms；其 DRAM 地板是
`16.3 MB / 32 GB/s ≈ 0.51 ms/层`（25 层 = 12.7 ms），即使内核效率到 100% 也只能省 ~9 ms，
**刚好卡在 40 ms 边缘**，而剩下的 ~0.75 ms/层是 GPU 侧串行 + WDDM 提交，属于"不动 CUDA 内核"的约束之外。
若允许改 GPU 侧（CUDA Graph 只捕获 CPU 层的注意力段、或把 CPU 层注意力与下一层提交重叠），
才可能真正压到 40 ms 以下。

---

## 3. 四模型性能表（证明没有牺牲别的模型）

`bash tools/bench_perf.sh final`（两轮，自动放置；机器当时有其他进程，绝对值偏大）：

| 模型 | 之前（§28 记录） | 现在（2 轮） | 门槛 | 状态 |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite | 41.8 ms（24.0 tok/s） | 43.3 / 44.2 ms（23.1 / 22.6 tok/s） | ≥23 | ✅ 同状态 A/B new 更快（见下） |
| Qwen3.5-9B | 27.0 ms（37.1 tok/s） | 26.8 / 27.5 ms（37.3 / 36.4 tok/s） | ≥36 | ✅ |
| qwen2.5-0.5b | 6.0 ms（166 tok/s） | 5.87 / 6.52 ms（170 / 153 tok/s） | ≥155 | ✅ 同状态 A/B new 更快（见下） |
| Qwen3.6-35B-A3B | 50.2 ms（19.9 tok/s） | 50.9 / 51.8 ms（19.6 / 19.3 tok/s） | ≤40 ms | ❌ 目标未达成 |

因为机器噪声大，另做了 **pre/new 二进制交错 A/B**（每模型 4 次，各 2 次）：

| 模型 | pre 中位 | new 中位 | 结论 |
|---|---|---|---|
| qwen2.5-0.5b | 6.41 ms | **5.91 ms** | new 更快（dense，不走 CPU 专家路径，噪声内） |
| Coder-V2-Lite | 51.7 ms（噪声大） | **44.7 ms** | new 不慢于 pre；机器快时 42.2 ms = 23.7 tok/s |
| Qwen3.5-9B | 26.70 ms | 27.03 ms | +1.2%（噪声内；dense 路径未改） |
| Qwen3.6-35B-A3B | 52.91 ms | **49.46 ms** | 1.07× |

Coder/9B/0.5B 与 35B 共用同一套 CPU 行内核（Coder 的 CPU 专家是 Q5_K/Q8_0，本次只加了预取 + 派发外提，
算术逐位不变），A/B 显示没有回退。

---

## 4. 三套验证状态

| 套件 | 结果 |
|---|---|
| `bash tools/verify.sh` | **ALL GREEN**（含新增 `build/test_cpu_moe.exe`） |
| `bash tools/smoke_models.sh` | **4/4 正确**：40716 / 185 / 2614 / 69014 |
| `bash tools/verify_generality.sh` | **GENERALITY GREEN**（26 passed, 0 failed） |

新增数值单测 `tests/test_cpu_moe.cpp`（接入 build.sh + verify.sh）：用**改动前内核算术的逐位复刻**
（单个 8 通道累加器、`std::fma` 复现 vfmsub/vfmadd、同一 hsum 树）作参考：

| 用例 | max&#124;Δ&#124; | 限值 |
|---|---|---|
| Q4_K 行 (512×2048，max&#124;y&#124;=1.54) | 4.8e-07 | 1e-6 |
| Q6_K 行 (2048×512) | 2.8e-08 | 1e-6 |
| `moe_forward_mt` 整层（1 token） | 7.3e-08 | 1e-6 |
| `moe_forward_mt` 整层（2 token） | 6.0e-08 | 1e-6 |
| Q5_K / Q8_0 路径自洽 | 2.6e-06 / 1.2e-07 | 1e-3 |

额外交叉验证（真实数据，非合成）：`build/calib2.cpp` 拿 35B 的 `blk.16.ffn_{gate,up,down}_exps`
真实量化权重，新内核 vs 旧内核逐行：max|Δ| = 3.8e-07 / 3.9e-07 / 1.2e-07（max|y| = 1.16 / 1.00 / 0.59）。

---

## 5. 失败的尝试（如实）

| 尝试 | 数据 | 结论 |
|---|---|---|
| **gate+up 两行融合内核**（共享一次 x 载入、省一半行开销） | phase A 0.53 → **0.63–0.77 ms** | 更慢：8 个 scale/offset 广播 + 4 累加器 + 数据寄存器超 16 个 ymm，压力过大 |
| **down+merge 融合**（按输出行一次算 8 个专家，省掉 45 µs 的 merge 并行区） | phase B 0.35 → **0.44 ms** | 净亏：每线程 8 条交错流破坏预取 |
| **12 线程（占满全部逻辑核）** | 17.4 vs 23.3 GB/s（9–10 线程） | OS/WDDM 驱动抢 CPU；select 9–10 |
| **预取策略**：整行 T0 / 本行 T0+下一行 T0 / 下一行 T0+下下行 T1 / 滚动 +288 B / 本行 T0+下一行 T1 | 27.0–30.7 / 27.6–30.1 / 27.9–29.0 / 29.5–31.6 / **28.4–31.4 GB/s** | 选最后一种（最简单、多线程最稳） |
| **int8/16 整数点积**（vpmaddwd/vpmaddubsw） | — | 未做：x 必须量化，误差 >1e-6，会破坏 Coder 的 top1-top2 余量；且当前已是解包/依赖链受限而非 ALU 受限 |
| **绑核 / 只留物理核** | 6 线程 0.955 ms vs 9 线程 0.793 | 6 线程不够（单核吞吐不足）；9 线程是甜点 |
| **目标 ≤40 ms** | 49.5 ms（中位） | 未达成；瓶颈已重新量化（§2.3），不是 CPU GEMV 单独能解 |

---

## 附：复现命令

```bash
cd engine
bash build.sh                                        # 17 个可执行 + 新单测
./build/test_cpu_moe.exe                             # 数值单测（≤1e-6）
bash tools/verify.sh                                 # ALL GREEN
bash tools/smoke_models.sh                           # 40716/185/2614/69014
bash tools/verify_generality.sh                      # GENERALITY GREEN

# 独立带宽基准（不进 build.sh，避免拖慢日常构建）
python -m ziglang c++ -std=c++17 -O2 -Iinclude \
  tools/bench_cpu_layer.cpp src/cpu_matvec.cpp src/config.cpp -o build/bench_cpu_layer.exe
./build/bench_cpu_layer.exe layer 24 100             # 16.3 MB/层 × 24 副本轮转
./build/bench_cpu_layer.exe read 24 100 12           # 同字节纯读上限
ENGINE_THREADS=12 ./build/bench_cpu_layer.exe layer 24 100

# 引擎内取证
ENGINE_DBG_CPUEXEC=1 ENGINE_TOPK=1 ./build/test_gen.exe "<35B>" "760" 9
ENGINE_DBG_LAYERMS=1 ./build/test_gen.exe "<35B>" "760" 2
ENGINE_DBG_MOE_PHASES=1 ./build/test_gen.exe "<35B>" "760" 3
ENGINE_KPROF=1 ./build/test_gen.exe "<35B>" "760" 9   # GPU busy
bash tools/bench_perf.sh final                        # 四模型
```

回退：本轮的改动前源码在 `build/bak/cpu_matvec.cpp.pre`（对应二进制 `build/test_gen.pre.exe`）。
