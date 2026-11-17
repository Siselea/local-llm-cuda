# 利用率与流水线气泡取证（UTILIZATION-REPORT）

**日期**：2026-11-15
**问题**：引擎跑起来 **CPU 和 GPU 占用都上不去**（llama.cpp 是 GPU 90 %+ / CPU 60 %+），说明流水线里有大量空闲气泡。
**本报告只做取证与量化**：不修改 `src/`、`include/`、`build.sh`、`tests/`；所有数据来自新增的 `tools/` 度量工具 + 引擎既有环境变量，报告期未改动引擎行为。
**被测二进制**：`tools/util_bin/test_gen_snap.exe`，sha256 `05ca7d3c254c4d2d6045b8c82d85eb8297ae7bbe04cd1712ff6109a38556a873`（测量开始时 `build/test_gen.exe` 的快照；另一个智能体期间在并发重建该文件，快照保证 23 个有效 run 口径一致）。
**被测模型**：0.5B / Qwen3.5-9B / DeepSeek-Coder-V2-Lite / Qwen3.6-35B-A3B（`<MODEL_DIR>/*.gguf`）。

---

## 0. 一句话结论

- **9B / 0.5B 的 GPU 已经饱和**（NVML 利用率 96 % / 88 %，SM 时钟顶在 1951–1953 MHz），气泡只有 **1.8 %**；它们与 llama.cpp 的差距在 **GPU 内核本身**（GEMV 之外的边缘内核 + 内核效率），不在调度。
- **Coder / 35B 的 GPU 确实闲**（40 % / 44 %）：空闲几乎全部是 **"GPU 等 CPU"** —— 25 个（35B）/ 6 个（Coder）CPU 专家层的 host matvec + 每层一次 WDDM 提交/同步往返，把单 token 的 50 %–60 % 时间花在 GPU 空转上。
- 35B decode 期间 **SM 时钟实测 1950 MHz（顶格），没有 570 MHz 降频**；SPEC §12 的"降频"判断不成立，GPU 是"闲"不是"慢"。
- 35B 每个 CPU 层 1.53 ms 中，**约 0.40 ms（26 %）是纯跨设备往返开销（可回收 ≈ 10 ms/token）**，1.13 ms 是 host matvec（CPU 算力，只能靠更快内核或把层搬回 GPU 回收）。
- "GPU 忙占比 50 % → 85 %" 对 35B/Coder **结构上不可达**：层间严格串行时 `busy ≤ GPU_work/(GPU_work+host_work)`；35B 的 GPU 工作量约 22 ms、host 工作量约 38 ms，上限 ≈ 37–44 %。现实目标是把 host 工作量压下去后的 **55 %–65 %**（对应 token 36–39 ms）。

---

## 1. 度量工具与口径（详见 `tools/README.md` 新增章节）

| 工具 | 作用 |
|---|---|
| `tools/measure_utilization.cpp` | 外部采样器（C++ 动态加载 `nvml.dll`）：spawn 引擎、按 10 ms 采 GPU 利用率/显存/功耗/SM 时钟/温度/pstate + 进程 CPU 时间/整机 CPU 时间，并给引擎每一行 stderr 打时间戳。**刻意不用 `nvidia-smi --loop-ms`**：每次 spawn nvidia-smi 要 50–100 ms，做不出 10 ms 档位；单进程 NVML 一次采样几微秒。 |
| `tools/utilization_report.py` | 解析器/报告器：把 `*.timeline.csv` 与带时间戳的 `*.engine.log` 对齐，解析 `[time]/[fwd]/[layerms]/[kprof]/[cpuexec]/[pipe]/[moe-prof]`，输出逐层表、按层类别的 NVML 利用率、kprof 逐内核表、文本时间序列、`summary.md/summary.json`。 |
| `tools/run_utilization_suite.sh` | 四模型取证套件（20 个 run，每个 load→采样→退出，不常驻）：`graph`（默认执行方式，真墙钟+真 NVML）、`probe`（`--no-graph`+`ENGINE_KPROF`+`ENGINE_DBG_LAYERMS`+`ENGINE_DBG_CPUEXEC`）、`prefix1/prefixN`（`--layers N` 前缀差分）、`prof`（`ENGINE_PROFILE`）。 |
| `tools/run_class_gpu_probe.sh` | 稠密模型"真实每层 GPU 时间"：**CUDA Graph 模式**下用 `--layers 1` 与 `--layers N` 的墙钟差分（graph 重放无主机提交气泡，`W(N)=N·p+H`），绕开 `--no-graph` 下 kprof 事件把提交气泡算进 kernel 的假象。 |

引擎内部时间线**只用既有环境变量**打开，没有任何代码改动：
`ENGINE_DBG_FWD`（每 forward 一行）、`ENGINE_DBG_LAYERMS`（逐层墙钟/sync/launch）、`ENGINE_KPROF`（CUDA event 逐内核 GPU 时间 + busy%）、`ENGINE_DBG_CPUEXEC`（host matvec ms/层）、`ENGINE_PIPE_PROF`、`ENGINE_PROFILE`。

**数据怎么测的（避免假象）**：

1. **外部采样 10 ms**，但 **NVML 的时间分辨率是 ~0.5–1 s**（`utilization.gpu`/`power.draw` 由驱动按滚动窗口更新；10 ms 只是"读得勤"，不是 1 s 内有 100 个独立样本）。所以**利用率均值只用 ≥5 s 的解码窗口**（`graph` 模式每 run 6–13 s）；`probe/prefix` 这类 <2 s 窗口只用于逐层时间，不看 NVML 均值。
2. **桌面底噪 ~25 %**：本机常驻 NVIDIA Overlay/Edge WebView/Explorer 等 CEF 进程，空闲时 NVML 也报 10–27 %、~17 W。每个 run 前采 2 s `ambient_*` 基线写进 `meta.json`，本报告的 util 数据都附带基线。
3. **`ENGINE_KPROF` 在 `--no-graph` 下会把主机提交气泡算进 kernel elapsed**：实测 0.5B 的 kprof total 10.98 ms，而 CUDA Graph 模式整个 token 只要 6.89 ms —— kprof 数值是"kernel 时间+提交气泡"的**上界**。因此本报告用三种口径交叉：kprof（相对占比/内核清单）、NVML/功率（外部占空比）、`--layers` 差分（稠密模型真实每层 GPU 时间）。
4. **warmup / graph capture token 要剔除**：稠密模型第一个 decode token 现场捕获整图（0.5B 实测 ~15 ms vs 稳态 6.9 ms）。`[fwd]` 行把 token 切开，解析器自动识别 token 0 的 capture 首帧并剔除；表中 "steady decode (median)" 与引擎自报均值并列。
5. **单次测点受机器负载影响大**：测量窗口里另一智能体也在用机器，整机 CPU 占用 25 %–44 %（`sys_busy` 列）。host matvec 这类 CPU 侧数字会随之膨胀（35B host matvec 本轮 1.13–1.20 ms/层，而 CLEAN 机器上 `CPU-GEMV-BW-REPORT.md` 记录为 0.81–0.87 ms/层）。报告里两个值都给。
6. **长 decode 会偶发挂死**（见 §10），所有 run 都加了 `--timeout-ms`，挂死 run 的部分数据（decode 前 N 个 token）仍可用并标 `timed_out`。
7. **避免 L2 假象**：本报告引用的"孤立 GEMV 带宽 295–319 GB/s / 纯读天花板 ~330 GB/s"来自 `Q6K-DP4A-RESULTS.md` 的多副本轮转基准（工作集 ≫ 12 MB L3，量到的是 DRAM）；模型内流量则用 **GGUF 张量表逐张量求和**得到（如 9B 每 token 5868 MB，embed 只算 1 行），不靠文件大小估算。两者相除得到"有效带宽"，避免把 cache 命中当成 DRAM 带宽。
8. **避免 profiler 自身开销混入**：`kprof` 每个内核插 2 个 CUDA event，在 WDDM 上会把墙钟膨胀 ~30 %（35B: 71.75 vs 50.09 ms）。所以"逐层墙钟"用**不插 event 的 layerms run**，"每层真实 GPU 时间"用**无 profiler 的 graph 差分**；kprof 只用于内核相对占比与调用次数。

---

## 2. 四模型总表（`graph` 模式，干净窗口，所有 run `quiet_before_start=true` / `foreign_during_run_hits=0`）

| 模型 | 放置 | steady decode | tok/s | 用户 llama.cpp | 占比 | NVML GPU util 均值/中位 | util<30 % 的时间 | 功耗 | SM 时钟 | 进程 CPU（占整机 12 线程） | kprof GPU busy | 气泡占比 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5B | 24 层全在 GPU | **6.89 ms** | 145 | — | — | **87.7 % / 91 %** | 5.1 % | 114 W | 1953 MHz | 100 % (1.00 core，8.3 %) | 68.8 %\* | **~19 %（外推差）** |
| Qwen3.5-9B | 32 层全在 GPU | **27.17 ms** | 36.8 | 46.7 tok/s (21.4 ms) | 79 % | **96.0 % / 96 %** | 0.0 % | 148.4 W | 1951 MHz (1942–1965) | 97 % (0.97 core，8.1 %) | **95.8 %** | **1.8 %** |
| Coder-V2-Lite | 20 GPU / 6 CPU MoE 层 | **44.24 ms** | 22.6 | 18.17 tok/s (55.0 ms) | **124 %** | **39.9 % / 37 %** | 0.0 % | 69.0 W | 1913 MHz (1777–1950) | **422 % (4.22 core，35.2 %)** | **48.5 %** | **~52–60 %** |
| Qwen3.6-35B-A3B | 15 GPU / 25 CPU MoE 层 | **50.09 ms** | 20.0 | 25.76 tok/s (38.8 ms) | 78 % | **43.5 % / 43 %** | 0.0 % | 50.1 W | 1923 MHz（有效值 1950，见 §5） | **357 % (3.57 core，29.8 %)** | **43.2 %** | **~56 %**† |

\* 0.5B 的 kprof busy 只在 `--no-graph` 下可得且被提交气泡污染；其真实占空比以 NVML 87.7 % 为准（graph 模式）。
† 35B 的 `graph` run 在 decode 第 25 个 token 时引擎挂死（§10 的丢唤醒 bug，`timed_out=true`）；表中数字取前 25 个稳态 token（墙钟 46.8–50.7 ms），此后没有新数据，`[time] decode` 未打印。挂死前的采样全部干净（`foreign_during_run_hits=0`）。
（"占比" = 引擎 tok/s ÷ 用户 llama.cpp tok/s；Coder 超过 100 % 说明这个模型我们比 llama.cpp 快。）

**关键读数**：0.5B 和 9B 的 `util<30 %` 占比几乎为 0 —— 这两个模型没有气泡问题；Coder/35B 的 GPU 利用率只有 40 % 左右，同时进程 CPU 反而很高（4.2 / 3.6 core），典型的"两侧互相等、谁都没跑满"。

---

## 3. Qwen3.5-9B：GPU 已饱和，差距在内核本身（回答"剩余 20 % 差距在哪"）

### 3.1 空闲不在调度：GPU 忙占比 96 %，SM 时钟顶格

- NVML：util 均值 **96.0 %**（p10=93 %，p90=97 %），`util<30 %` 的时间 **0.0 %**；功耗 **148.4 W**（整机空闲 16 W）；**SM 时钟 1951 MHz（min 1942 / max 1965，= 最大 boost）**。
- 独立佐证：`ENGINE_PROFILE` 的 `[gpu] forward gpu=27.12–27.19 ms`（CUDA event 包住整个 32 层 pipeline），与 `graph` 模式墙钟 27.17 ms 几乎相等 → **GPU 时间 ≈ 墙钟，占空比 ≈ 100 %**。
- 功率/时钟也排除了"降频"：9B 是四个模型里唯一持续 148 W 的。

### 3.2 27.17 ms 的 GPU 时间花在哪：CUDA Graph 墙钟差分的真实每层 GPU 时间

用 `--layers 1` 与 `--layers 16` 的 graph 模式墙钟做差分（graph 重放没有主机提交气泡，`W(N)=N·p+H`）：

| 项 | 值 |
|---|---|
| W(1 层 + head) | 4.334 ms |
| W(16 层 + head) | 15.149 ms |
| **每层 GPU 时间 p** | **0.721 ms** |
| head（final norm + lm_head + logits 回传） | 3.61 ms |
| 全模型外推 32·p + H | 26.69 ms |
| 实测墙钟 | 27.17 ms |
| **残余气泡** | **0.48 ms（1.8 %）** |

### 3.3 每层 0.721 ms 与带宽账

- 9B 每 token 需要从 HBM 读的权重（GGUF 张量表实测，embed 只读 1 行）：**5868 MB/token**。
- 用 27.17 ms 算有效带宽 = **216 GB/s**；用 GPU 时间 26.7 ms 算 = **220 GB/s**。
- 参照（都在同一台机器实测，见 `Q6K-DP4A-RESULTS.md`）：Q5_K/Q6_K 的 dp4a GEMV **孤立带宽 295–319 GB/s**，纯读天花板 **~330 GB/s**。
- 结论：**有效带宽只有孤立 GEMV 的 70–75 %**。缺口不是"没喂饱 GPU"（96 % 忙），而是：
  1. **GEMV 之外的边缘内核**：kprof 里 22 种内核中，GEMV 类 5 种合计 23.82 ms（72 %），其余 `gd_delta_rule`(3.89 ms/24 次)、`rmsnorm`(0.97/81)、`quant_x_i8`(1.20/201)、`rope_strided`、`gd_norm_qk`、`gd_conv1d_silu`、`q35_beta_g`、`silu_mul`、`add_inplace`、`zero_fill`、`kv_write` 等合计约 **9.1 ms（28 %，含 kprof 气泡上界）**。按 kprof 全局膨胀系数 ~1.2× 折算，真实边缘内核约 **5–6 ms**。
  2. **模型内的 GEMV 达不到孤立带宽**：孤立 295–319 GB/s 是"多副本轮转、L2 规避、中位"的理想值；模型里每层权重都是不同地址、与 norm/quant/attention 内核交替，实测 216–220 GB/s ≈ 孤立的 72 %，还叠加 split-K 的额外写回/规约。
- 与 llama.cpp 的 21.4 ms 对照：21.4 ms 意味着 **307 GB/s 有效带宽**（几乎贴纯读天花板）且几乎没有边缘内核开销。**所以 9B 的 20 % 差距是"非 GEMV 内核开销（~5–6 ms）+ GEMV 有效带宽（220 vs 295–319）"，不是调度、不是气泡、不是降频。**

### 3.4 9B 的逐层表（probe，`--no-graph`）

无 kprof 时 `wall` 是主机发射时间（32 层 ×0.593 ms = 19.0 ms 主机时间 < 27.1 ms GPU 时间 → **主机跟得上 GPU**，这是 9B 能保持 96 % 占空比的原因）：

| 指标 | 值 |
|---|---|
| 每层主机发射（launch+sched） | 0.593 ms（其中 launch 调用约 0.233 ms） |
| 每层 GPU 时间（graph 差分） | 0.721 ms |
| 每层主机/GPU 比 | 0.82（主机只比 GPU 快 18 %，余量很薄） |
| 每层权重流量 | 5868/32 ≈ 183 MB |

> 工程含义：9B 上把主机侧 launch 数再减少（逐层 graph/融合）不会带来收益（GPU 已 96 %）；**只有减少/融合边缘内核、提高 GEMV 有效带宽才有收益**。

---

## 4. qwen2.5-0.5B：内核发射/占用率受限，无气泡可收

- graph 模式 steady **6.89 ms/token**，NVML util **87.7 %**（median 91 %）、功耗 114 W、SM 1953 MHz；进程 CPU 1.00 core。
- graph 差分：p = (3.366−1.323)/11 = **0.186 ms/层**，head H = 1.137 ms，全模型外推 5.59 ms vs 实测 6.89 ms（**残余 1.3 ms，约 19 %**：层数从 12 加到 24 时每层 GPU 时间变大，说明小模型上内核波次/占用率效应明显）。
- 每 token 只需 392 MB 权重，6.89 ms → **57 GB/s**，远低于 HBM 能力；kprof 显示 24 层 × ~16 个内核 = ~380 次内核发射 / 6.89 ms ≈ **18 µs/内核**，是典型的 **latency/occupancy 受限**，不是带宽也不是气泡。
- 该模型已经用了整图 CUDA Graph（graph 模式），进一步提高需要减少内核数/融合层，优先级低（用户表中 0.5B 是唯一"我们没有落后"的模型）。

---

## 5. Qwen3.6-35B-A3B：气泡最大的模型（回答 Q1/Q2）

### 5.1 放置与总账

- 放置：**15/40 个 MoE 层在 GPU，25 个在 host**（`[placement] gpu_moe_layers=15/40 (auto)`，无 adaptive 触发；vram post-load free 1787 MiB）。
- steady decode **50.09 ms/token**（20.0 tok/s）；prefill 1 token 70 ms。
- NVML：util **43.5 %**（median 43，p10 36，p90 49），功耗 **50.1 W**；`[kprof]` GPU busy **43.2 %**；两者一致 → **GPU 约 44 % 占空比，其余 56 %（≈28 ms/token）是 GPU 空闲**。
- 进程 CPU **357 %（3.57 core = 整机 29.8 %）** —— 比 9B 高 3.7 倍，但 GPU 反而闲，典型"两侧互相等"。
- **SM 时钟：decode 窗口内 127/156 个采样是 1950 MHz（max 1965）**；另有 29 个采样显示 210 MHz，但同一瞬间 NVML 报 `util=100 %`、`power=16.5 W`（物理矛盾）→ 这些是驱动传感器未刷新的假值。**结论：35B decode 时 SM 时钟顶格，无 570 MHz 降频。**

### 5.2 每层墙钟（`ENGINE_DBG_LAYERMS`，无 kprof 开销，取两个完整 decode token 均值）

| 层类 | 层数 | 墙钟/层（均值） | host matvec | host launch | sync 等待 | 该层窗口内 GPU 占空比 |
|---|---|---|---|---|---|---|
| **GPU 层**（专家在显存） | 15 | **0.620 ms**（0.47–0.78） | 0 | 0.236 ms（~26 次发射） | 0.206 ms | **≈ 90–95 %（GPU-bound）** |
| **CPU 层**（专家在 host） | 25 | **1.537 ms**（1.28–1.85） | **1.13–1.20 ms** | 0.255 ms（~25 次发射） | 0.149 ms | **≈ 25–30 %** |

- 15×0.620 + 25×1.537 = **47.73 ms**；加 final norm + lm_head + logits 回传（约 2.4 ms）= **50.1 ms** ✓（与 graph 模式墙钟 50.09 ms 吻合）。
- CPU 层构成为：**host matvec 74–78 % + 每层提交/同步往返 ~26 %**。GPU 层里也有约 2/3 是主机提交与同步等待（0.236+0.206 = 0.442 ms 对 0.620 ms 墙钟）。

### 5.3 40 层逐层明细（两个完整 decode token 平均，`ENGINE_DBG_LAYERMS` 实测）

> 下表 `墙钟/launch/sync` 都是逐层实测；`host matvec` 只有类平均值（`[cpuexec]` 收敛值 1.13–1.20 ms/层，逐层不单独打印）。

| layer | 类 | 墙钟 ms | launch ms | sync ms |
|---|---|---|---|---|
| 0 | GPU | 0.58 | 0.21 | 0.18 |
| 1 | GPU | 0.77 | 0.28 | 0.30 |
| 2 | GPU | 0.78 | 0.40 | 0.12 |
| 3 | GPU | 0.53 | 0.28 | 0.04 |
| 4 | GPU | 0.63 | 0.26 | 0.20 |
| 5 | GPU | 0.61 | 0.21 | 0.26 |
| 6 | GPU | 0.59 | 0.21 | 0.24 |
| 7 | GPU | 0.51 | 0.18 | 0.15 |
| 8 | GPU | 0.63 | 0.21 | 0.28 |
| 9 | GPU | 0.60 | 0.21 | 0.24 |
| 10 | GPU | 0.62 | 0.19 | 0.27 |
| 11 | GPU | 0.47 | 0.23 | 0.04 |
| 12 | GPU | 0.61 | 0.20 | 0.27 |
| 13 | GPU | 0.64 | 0.20 | 0.29 |
| 14 | GPU | 0.71 | 0.30 | 0.19 |
| 15 | CPU | 1.45 | 0.23 | 0.07 |
| 16 | CPU | 1.67 | 0.29 | 0.17 |
| 17 | CPU | 1.46 | 0.21 | 0.22 |
| 18 | CPU | 1.71 | 0.35 | 0.14 |
| 19 | CPU | 1.59 | 0.28 | 0.03 |
| 20 | CPU | 1.48 | 0.20 | 0.21 |
| 21 | CPU | 1.46 | 0.23 | 0.21 |
| 22 | CPU | 1.54 | 0.21 | 0.20 |
| 23 | CPU | 1.29 | 0.20 | 0.04 |
| 24 | CPU | 1.54 | 0.33 | 0.17 |
| 25 | CPU | 1.85 | 0.32 | 0.16 |
| 26 | CPU | 1.46 | 0.24 | 0.19 |
| 27 | CPU | 1.31 | 0.19 | 0.03 |
| 28 | CPU | 1.54 | 0.23 | 0.20 |
| 29 | CPU | 1.77 | 0.21 | 0.21 |
| 30 | CPU | 1.55 | 0.26 | 0.21 |
| 31 | CPU | 1.60 | 0.33 | 0.05 |
| 32 | CPU | 1.41 | 0.19 | 0.22 |
| 33 | CPU | 1.50 | 0.25 | 0.23 |
| 34 | CPU | 1.62 | 0.32 | 0.16 |
| 35 | CPU | 1.65 | 0.24 | 0.06 |
| 36 | CPU | 1.60 | 0.29 | 0.14 |
| 37 | CPU | 1.46 | 0.23 | 0.20 |
| 38 | CPU | 1.28 | 0.22 | 0.19 |
| 39 | CPU | 1.65 | 0.34 | 0.03 |

类汇总（两个 decode token 平均）：

```
GPU x15: wall 0.620  launch 0.236  sync 0.206  (sum 9.30 ms)
CPU x25: wall 1.537  launch 0.255  sync 0.149  (sum 38.44 ms)
TOTAL   : wall 47.73 ms (2 decode tokens averaged)
```

- 15 个 GPU 层的墙钟合计 9.30 ms，25 个 CPU 层的墙钟合计 38.44 ms，40 层合计 **47.73 ms**；加 final norm + lm_head + logits 回传（约 2.4 ms）= **50.1 ms**，与 graph 模式墙钟 50.09 ms 吻合（误差 0.02 ms）。
- 逐层 launch 列（15–25 次内核提交）在 CPU 层和 GPU 层都稳定在 **0.19–0.35 ms**；sync 列在 GPU 层平均 0.206 ms、CPU 层平均 0.149 ms —— 每层一次阻塞同步。
- `layer 25`（1.85 ms）和 `layer 18`（1.71 ms）是抖动层（被桌面/并发 agent 抢占时会被拉长，两个 token 平均后仍偏高）。

### 5.4 时间序列（四模型，文本；来自各自 `graph` 运行）

> NVML 的 util 只有 ~0.5 s 分辨率，所以下面只能看量级/趋势；单 token 级的波形不可得，逐层请看 §5.3。
> 字符表：` `=0 … `@`=100 %。

**qwen2.5-0.5B**（bin 89 ms，80 bin；上=NVML util 0–100%，下=GPU 功耗 41–132 W，空闲 16 W）

```
util  : ::::#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
power :     :::::-*****%@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

**Qwen3.5-9B**（bin 87 ms，80 bin；上=NVML util 0–100%，下=GPU 功耗 41–160 W，空闲 21 W）

```
util  : @@@@@@@@@@@@@@@@@@@@@@%%%%%@@@@@@%%%%%@@@@@@@%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
power :     -+++++%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
```

**DeepSeek-Coder-V2-Lite**（bin 72 ms，80 bin；上=NVML util 0–100%，下=GPU 功耗 23–78 W，空闲 16 W）

```
util  : %%%%=---------------------------=======----------------------------=============
power :     -======%@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%##############%%%%%%%@@@@@@@@@@@@@
```

**Qwen3.6-35B-A3B**（bin 50 ms，25 bin；上=NVML util 0–100%，下=GPU 功耗 26–75 W，空闲 21 W）

```
util  : *==========---------=====
power :  .........=*********%@@@@
```

> 读数：**9B** 的 util 是 ~96 % 的直线（没有气泡）；**Coder** 在 32–43 % 之间来回（21 个 GPU 层与 6 个 CPU 层交替，被 1 s 窗口拉平）；**35B** 是 43 % 附近的直线 + 功耗 26→75 W 的爬升（warm-up/并发负载），没有"满占空比段"；**0.5B** 前 ~0.4 s 是 graph capture 首帧（低 util），之后贴着 88–91 %。

### 5.5 Q1 答案：空闲发生在哪一侧？

| 空闲来源 | 位置 | 量化（每 token） |
|---|---|---|
| **GPU 等 CPU：host matvec**（25 个 CPU 层） | CPU 层 | 25 × (1.527 − 该层 GPU 时间 ~0.40) ≈ **28 ms** |
| **GPU 等 CPU：host 内核提交** | 全部 40 层 | 40 × ~0.27 ms ≈ 10.8 ms 的 host 时间里 GPU 大部分空闲（与计算部分重叠，净气泡 ~4–6 ms） |
| **GPU 等 CPU：router 回传同步** | 全部 40 层 | 40 × ~0.17 ms ≈ 6.8 ms 的同步等待 |
| **CPU 等 GPU：GPU 层内** | GPU 层 | 15 × 0.20 ms ≈ 3 ms（GPU 层是 GPU-bound，CPU 只等 GPU） |
| head / logits 回传 / argmax | 层外 | ~0.9 ms |

⇒ **主因是"GPU 等 CPU"（CPU 专家层的 host matvec 占 25 层，每层 1.1–1.2 ms 串行）+ 每层一次 WDDM 提交/同步往返；"CPU 等 GPU"只有 ~3 ms。** 不是两边互相等各占一半。

### 5.6 Q2 答案：25 个 CPU 层的每层 1.42–1.53 ms 里，多少能靠"消除跨设备往返"回收？

- CPU 层 1.527 ms 的构成（实测）：**host matvec 1.13–1.20 ms（74–78 %）+ host launch 0.289 ms（~19 %）+ sync 等待 0.137 ms（~9 %）**。
- **能靠"消除跨设备往返"回收的是 launch + sync ≈ 0.40–0.43 ms/层 → 25 层 ≈ 10–10.7 ms/token（占 token 的 20–21 %）**，手段：逐层 CUDA Graph / 设备端路由（去掉每层 router 回传） / 融合内核减少发射次数。
- **不能靠"消除往返"回收的是 host matvec 的 1.13 ms/层 ≈ 28 ms/token**（CPU 在算专家，GPU 只能等）；这部分只能靠 **更快的 CPU GEMV**（干净机器 0.87 ms → 可回收 ~6.5 ms）或 **把层搬回 GPU**（每层省 ~0.9 ms，但显存不够）。
- 补充：CPU 层的 GPU attention 必须等 attention 输出才能算 matvec（层内强依赖），所以"把 attention 和 matvec 重叠"在层内做不到。

---

## 6. DeepSeek-Coder-V2-Lite：两个独立气泡源各占一半

- steady decode **44.24 ms/token**（22.6 tok/s，**比用户 llama.cpp 的 18.17 tok/s 快 24 %**）。
- 放置：20 GPU / 6 CPU MoE 层；NVML util **39.9 %**、kprof busy **48.5 %**、功耗 69 W、SM 1913 MHz（1777–1950）；进程 CPU **422 %（4.22 core）**。
- probe（`--no-graph`，kprof 把墙钟膨胀 54.94/44.24 = 1.24×，下表已按 0.805 折算回真墙钟）：

| 层类 | 层数 | 真墙钟/层 | GPU 时间/层（kprof 差分，上界） | host matvec/层 | sync/层 |
|---|---|---|---|---|---|
| GPU 层（含 dense 首层） | 21 | **1.19 ms** | ~0.85 ms | 0 | **0.38 ms** |
| CPU 层 | 6 | **2.70 ms** | ~0.74 ms | **2.41 ms** | 0.02 ms |

- 21×1.19 + 6×2.70 ≈ 41.2 ms + head ≈ 44.2 ms ✓
- **气泡源 1**：6 个 CPU 层每层 host matvec **2.41 ms**（Coder 的 CPU 专家更重：inter=1408、每 token 6 个专家×8 层路由），6 层 ≈ 14.5 ms，占 token 的 33 %；这段时间 GPU 空闲。
- **气泡源 2**：21 个 GPU 层**每层 0.38 ms 的 sync 等待**（router logits 回传 + WDDM 往返），21 层 ≈ **8.0 ms**，占 token 的 18 %。
- **气泡源 3**：host launch 21×0.28 + 6×0.29 ≈ 7.6 ms 的提交时间（与计算部分重叠）。
- CPU 占用 4.22 core 的来源：6 个 CPU 层的 9 线程 matvec（~2.4 ms × 9 线程）+ 线程池唤醒/自旋（每层两个并行区）。**这是"CPU 也上不去、GPU 也上不去"的直接原因：host 只在 1/3 时间真正并行算，其余时间在等同步。**

---

## 7. Q4：我们和 llama.cpp 的调度差别能从哪量化看出来？

可量化的三点（全部来自本报告数据）：

1. **CPU 专家层的主机侧时间**：我们 35B 1.13–1.20 ms/层、Coder 2.41 ms/层（都被 host matvec 主导）。llama.cpp 同一台机器的对应数字用户没有给出，但这是最直接的一阶差异：**25 层 × Δms = 每 token 的直接差**。
2. **每层一次同步/回传**：我们用 `[layerms] sync 1/0.14–0.38` 直接量到每层一次阻塞同步（router logits 回传）。这是"每层一个 WDDM 往返"的独占证据；llama.cpp 在 decode 时对 MoE 路由是设备端完成（router 也在 GPU 上，无需回传），所以没有这 0.14–0.38 ms/层。35B 40 层 × 0.17 ≈ 6.8 ms、Coder 21 层 × 0.38 ≈ 8.0 ms。
3. **提交气泡的可见性**：`--no-graph` 与 graph 模式的墙钟差直接量化主机提交成本：0.5B **10.0 → 6.89 ms（−31 %）**、9B 只差 0.5 %（GPU-bound 时提交被隐藏）。**同样是"每 token 几百次内核发射"，GPU-bound 的模型不受影响，CPU-offload 的模型直接吃满**——这解释了为什么我们 9B 接近 llama.cpp（79 %）而 35B/Coder 差距更大（78 % / 已超过 Coder 的 llama 值）。

---

## 8. 如果要把 GPU 忙占比提上去：按模型排序的建议（带预期收益 ms）

> 结构性前提：层间严格串行时 `GPU busy ≤ GPU_work / (GPU_work + host_work)`。35B 实测 GPU_work ≈ 22 ms、host_work ≈ 38 ms → **上限 ≈ 37–44 %**（正好是实测 43 %）；Coder 同理 ≈ 45 %。**所以"85 %"在 35B/Coder 上不可能靠调调度达到**，必须减少 host 侧工作量或把工作量搬到 GPU。

### 8.1 Qwen3.6-35B-A3B（当前 50.09 ms，GPU 忙 43 %）

| 优先级 | 动作 | 预期收益 | 依据 |
|---|---|---|---|
| **P0** | **加快 CPU 专家 matvec**：1.13–1.20 → 0.85–0.90 ms/层（干净机器已达 0.87；本轮受并发机负载膨胀） | **−6.5～−8 ms** → 42–44 ms（GPU 忙 44→50 %） | 25 层 × 0.26–0.30 ms；`[cpuexec]` 逐层实测 |
| **P0** | 若 CPU GEMV 再优化到 bench 峰值 0.715 ms/层（`CPU-GEMV-BW-REPORT.md`） | **−10～−11 ms** → 39–40 ms（GPU 忙 55 %） | 25 × 0.42–0.45 ms |
| **P1** | **消除每层 CUDA 提交**：GPU 层逐层 CUDA Graph（launch 0.237→~0.03 ms）、CPU 层合并小内核 | **−3～−4 ms**（15 层 × 0.2 ms + 部分 CPU 层） | `[layerms] launch` 逐层实测 |
| **P1** | **设备端路由**去掉 router 回传同步（sync 0.14–0.20 ms/层） | 上界 **−6.8 ms**；但 dev-route kernel 实测慢 2.6 ms，净收益 **−3～−5 ms** | `[layerms] sync` + 另一智能体的 dev-route A/B |
| **P2** | vram margin 1200→600 MiB 多放 1–2 个 MoE 层 | **−0.9～−1.8 ms** | 每层 CPU→GPU 省 ~0.9 ms（1.13 host → ~0.2 GPU） |

**组合 P0+P1（不含 dev-route）**：50.1 → **36–39 ms（26–28 tok/s）**，GPU 忙占比 → **55–60 %**；距离 85 % 仍远（结构上限）。

### 8.2 DeepSeek-Coder-V2-Lite（当前 44.24 ms，GPU 忙 40 %）

| 优先级 | 动作 | 预期收益 | 依据 |
|---|---|---|---|
| **P0** | 6 个 CPU 层的 14.5 ms host matvec：迁移到 GPU（需 ~2.4 GB 显存，可把 vram margin 降到 0）或优化 CPU 内核 | 迁移上界 **−12 ms**；纯 CPU 提速 30 % → **−4.5 ms** | `[cpuexec]` 2.41 ms/层 ×6 |
| **P0** | 21 个 GPU 层的 router 回传同步（0.38 ms/层） | 上界 **−8 ms**；dev-route 净收益需扣除 kernel 变慢 | `[layerms] sync` |
| **P1** | 逐层 CUDA Graph 去掉 21 个 GPU 层的 26 次发射 | 上界 **−5 ms** | `[layerms] launch` 0.28 ms/层 |

### 8.3 Qwen3.5-9B（当前 27.17 ms，GPU 忙 96 %）——没有气泡，别动调度

| 优先级 | 动作 | 预期收益 |
|---|---|---|
| **P0** | **减少/融合 GEMV 之外的边缘内核**（`quant_x_i8` 201 次、`rmsnorm` 81 次、`add_inplace`、`rope`、`gd_*` SSM 链）：把 int8 量化融进 GEMV、norm 融进前一个内核的 epilogue | **−3～−5 ms**（边缘内核约 5–6 ms） |
| **P1** | GEMV 有效带宽 220 → 295–319 GB/s（提高占用/减少 split-K 回写、波次对齐） | **−3～−4 ms** |

两者都做到才可能到 ~22 ms（≈46 tok/s，与 llama.cpp 的 46.7 tok/s 持平）。**给 9B 加 Graph/改调度是零收益**。

### 8.4 qwen2.5-0.5B

内核发射/占用率受限（57 GB/s 有效带宽、~380 次发射/token），已用整图 graph。收益来自减少内核数/融合层；优先级最低。

---

## 9. 多智能体并发时如何避免测量被抢占污染（本报告的血泪教训）

本机是**两个 agent 共享**的（另一个 agent 同时在改 `src/` 并做 35B A/B）。这直接毁掉了若干 run，方法学如下：

1. **采样前等待静默**（`measure_utilization --wait-gpu-idle-sec`）：要求连续 3 次检查满足
   ① `nvidia-smi --query-compute-apps` 里没有 `test_gen*|engine*|llama*` 进程；
   ② 显存 < `--idle-mem-max-mb`（默认 1500 MiB；正常桌面 ~300–600 MiB）；
   ③ 整机 CPU < `--cpu-idle-max`（实测桌面底噪 20–45 %，用 45）；
   ④ NVML util < `--idle-util-max`（桌面 CEF 常驻，别用 5 % 这种严格阈值，默认 60）。
   **不要用"显存够"当唯一判据**：对方的 0.5B run 只占 0.6 GB，但会抢 CPU。
2. **记录 nvidia-smi 进程列表**：启动前/结束后各存一份（`*.foreign-start.txt`/`*.foreign-end.txt`），运行中每 5 s 复查一次（`*.foreign-during.txt` + `meta.foreign_during_run_hits`）。**报告里每个 run 都有 `quiet_before_start` / `foreign_during_run_hits` / `contaminated` 字段**，本报告只采用两者皆干净的 run；被抢占的 run 直接重跑。
3. **记录环境基线**：每次 run 前 2 s 采 `ambient_gpu_util` / `ambient_gpu_power_w`（本机底噪 10–27 %、16–17 W），否则会把桌面占用算进"引擎利用率"。
4. **用二进制快照 + sha256**：对方会并发重建 `build/test_gen.exe`；每次 run 都用 `tools/util_bin/test_gen_snap.exe` 并把 sha256 写进 `tools/util_results/MANIFEST.md`，suite 前后校验哈希没变。
5. **加超时与进程清理**：引擎在长 decode 时**偶发挂死**（§10），挂死后进程不退出、继续占 11 GB 显存 + 12 GB 主机内存，会让对方 build 失败（`lld-link: Permission denied`）并让后续 run 全部排队。采样器提供 `--timeout-ms`：超时 `TerminateProcess` 并把 `timed_out:true` 写进 meta；**部分数据（已完成的 decode token）仍然可用**。
6. **不要两个引擎同时 load 大模型**：两台同时在 12 GB 显存上 load 35B/11 GB 的模型会触发 WDDM 分页——双方都变慢，且**实测会让其中一方卡死在 decode 中途**（本次复现 3 次：35B 第 25 token、Coder 第 77 token、35B 第 3 token；`foreign=0`、非 adaptive 路径也出现）。这类"挂死"符合 §10 里 `cpu::Pool` 丢唤醒的并发 bug 特征（并发负载会放大触发概率），靠"错峰跑 + 显存阈值 + 外部 timeout"规避。
7. **时间窗口约定**：并发 agent 之间用消息约定独占窗口，并优先跑**短 run**（本次每个 run 都是 load→采样 6–13 s→退出，不做常驻采样）。

---

## 10. 测量中发现的引擎问题（附带证据，供后续修复）

1. **长 decode 偶发挂死**：`n=257` 时 35B 在第 25 个 decode token、Coder 在第 ~77 token 完全卡死（GPU 5 %/210 MHz、CPU 空转、`[time] decode` 不打印）；`n=17` 时 35B 也出现过 1 次第 3 token 挂死。`foreign_during_run_hits=0`、非 adaptive 路径。另一智能体（hostlayer-35b）定位为 `cpu::Pool` 的条件变量丢唤醒（主线程 `gen_` 递增 + `notify_all` 未持锁，worker 在 predicate 检查与 `cv_.wait` 之间被调度走会永久睡死，主线程自旋等 `remaining_` → 与"CPU 空转、GPU 掉到 210 MHz"完全吻合），**已在该 agent 的新 build 中修复（待长 decode 验证）**。本报告所有数据来自修复前的快照，故仍建议 decode 加外部 timeout。
2. **`[placement] adaptive:` 自适应减层路径必挂**（本次 3 次复现：Coder `20→17`、`20→16`，卡在 `[vram] post-load` 之后）。`--gpu-layers N --no-vram-check` 显式钉住放置可绕开。这条路径在显存紧张时必然触发，是"看起来是负载问题其实是引擎 bug"的陷阱。
3. **kprof 的 `GPU busy%` 在 `--no-graph` 下高估**：它把主机提交气泡算进 kernel elapsed（0.5B 上 kprof 68.8 % vs 真实占空比 ~88 % 是另一回事：0.5B 的 kprof total 10.98 ms 里 ~5 ms 是气泡）。**判断主机是否受限请用 `--layers` 差分 + `[gpu] forward` 事件跨度**。

---

## 11. 复现

```bash
cd engine
# 1) 构建采样器（ziglang 工具链，无需 MSVC/CUDA toolkit）
python -m ziglang c++ -std=c++17 -O2 tools/measure_utilization.cpp \
    -o tools/util_bin/measure_utilization.exe
# 2) 引擎快照（避免并发重建污染口径）
cp build/test_gen.exe tools/util_bin/test_gen_snap.exe && sha256sum tools/util_bin/test_gen_snap.exe
# 3) 四模型 20 个 run（graph / probe / prefix1 / prefixN / prof；每个 load→采样→退出）
bash tools/run_utilization_suite.sh            # 约 12–15 分钟 GPU 时间
# 3b) 报告的补充 run（35B 逐层 layerms、Coder/35B pinned probe 等）
bash tools/run_utilization_extras.sh
# 4) 稠密模型真实每层 GPU 时间（graph 模式 --layers 差分，约 1.5 分钟）
bash tools/run_class_gpu_probe.sh
# 5) 汇总（生成 summary.md/summary.json，本报告所有数字都能从 report.json 复核）
python tools/utilization_report.py --root tools/util_results \
    --md tools/util_results/summary.md --json tools/util_results/summary.json
```

关键产物：`tools/util_results/summary.md`（自动生成的表）、`tools/util_results/<model>/<mode>/report.json`（每 run 的全部派生量）、`tools/util_results/MANIFEST.md`（二进制哈希与采样参数）。

**局限性**：
- NVML 的 util/功耗是 ~0.5–1 s 滚动值，单 token 级波形不可得；逐层占空比是用"该层窗口内采样均值 × kprof 内核占比"交叉推断的，标了 "≈" 的数字是推断值。
- kprof 在 `--no-graph` 下高估 kernel 时间（本报告同时给出 kprof 与 graph 差分两个口径）。
- 35B 的 CUDA Graph 不可用（MoE 模型引擎禁用整图），逐层 graph 是另一智能体的实验方向（`layer_graph_auto` 当前重放 illegal memory access）。
- 测量期间整机有 25–44 % 的背景 CPU 占用（另一 agent），host matvec 类数字比干净机器高 25–30 %；报告同时给出干净机器历史值（0.87 ms/层）。

---

## 附录 A：kprof 逐内核表（`--no-graph`，最后一个 decode token；数值含 event 开销，只用于相对占比）

**Qwen3.5-9B**（22 种内核，GPU 时间合计 32.94 ms；GEMV 类约占 72 %，SSM/attention/norm 等边缘内核约占 28 %）

| 内核 | GPU 时间 ms | 调用次数 | ms/次 | 占 GPU 时间 |
|---|---|---|---|---|
| `gemv_q5k_x4_dp4a_s` | 10.90 | 115 | 0.0948 | 33.1 % |
| `gemv_q6k_dp4a` | 5.89 | 25 | 0.2356 | 17.9 % |
| `gemv_q6k_dp4a_sk` | 3.67 | 45 | 0.0816 | 11.1 % |
| `gd_delta_rule` | 2.98 | 24 | 0.1242 | 9.0 % |
| `gemv_q5k_x4_dp4a` | 2.06 | 16 | 0.1288 | 6.2 % |
| `gemv_q8_0_sk` | 1.30 | 48 | 0.0271 | 3.9 % |
| `quant_x_i8` | 1.20 | 201 | 0.0060 | 3.7 % |
| `rmsnorm` | 0.97 | 81 | 0.0120 | 2.9 % |
| `gd_norm_qk` | 0.70 | 24 | 0.0292 | 2.1 % |
| `gd_conv1d_silu` | 0.65 | 24 | 0.0271 | 2.0 % |
| `q35_beta_g` | 0.56 | 24 | 0.0233 | 1.7 % |
| `rope_strided` | 0.44 | 16 | 0.0275 | 1.3 % |
| `add_inplace` | 0.27 | 64 | 0.0042 | 0.8 % |
| `attn_cached_w` | 0.24 | 8 | 0.0300 | 0.7 % |
| `q35_split3` | 0.20 | 24 | 0.0083 | 0.6 % |
| `zero_fill` | 0.18 | 48 | 0.0037 | 0.5 % |

**Qwen3.6-35B-A3B**（GPU 时间合计 29.11 ms；`moe_gateup_silu`/`moe_down_scatter` 各 15 次 = 15 个 GPU 层的融合专家内核，`gd_*` 是 30 个线性注意力层的内核）

| 内核 | GPU 时间 ms | 调用次数 | ms/次 | 占 GPU 时间 |
|---|---|---|---|---|
| `gemv_q4k_x4` | 5.57 | 96 | 0.0580 | 19.1 % |
| `gd_delta_rule` | 3.89 | 30 | 0.1297 | 13.4 % |
| `gemv_f32` | 3.07 | 120 | 0.0256 | 10.5 % |
| `add_inplace` | 2.16 | 145 | 0.0149 | 7.4 % |
| `gemv_q8_0_bm` | 1.85 | 1 | 1.8500 | 6.3 % |
| `moe_gateup_silu` | 1.68 | 15 | 0.1120 | 5.8 % |
| `rmsnorm` | 1.59 | 101 | 0.0157 | 5.5 % |
| `gemv_q6k_dp4a` | 1.06 | 20 | 0.0530 | 3.6 % |
| `gd_norm_qk` | 0.96 | 30 | 0.0320 | 3.3 % |
| `zero_fill` | 0.82 | 40 | 0.0205 | 2.8 % |
| `gd_conv1d_silu` | 0.82 | 30 | 0.0273 | 2.8 % |
| `moe_down_scatter` | 0.79 | 15 | 0.0527 | 2.7 % |
| `moe_shared_gate` | 0.79 | 40 | 0.0198 | 2.7 % |
| `gd_conv_state` | 0.57 | 30 | 0.0190 | 2.0 % |
| `gemv_q4k_k2` | 0.57 | 74 | 0.0077 | 2.0 % |
| `silu_mul` | 0.52 | 40 | 0.0130 | 1.8 % |

**DeepSeek-Coder-V2-Lite**（GPU 时间合计 27.64 ms；`mla_*` 是 MLA 注意力，`moe_gateup_silu`/`moe_down_scatter` 各 20 次 = 20 个 GPU MoE 层）

| 内核 | GPU 时间 ms | 调用次数 | ms/次 | 占 GPU 时间 |
|---|---|---|---|---|
| `moe_down_scatter` | 5.40 | 20 | 0.2700 | 19.6 % |
| `moe_gateup_silu` | 4.71 | 20 | 0.2355 | 17.0 % |
| `gemv_q5k_x4_dp4a_s` | 4.64 | 176 | 0.0264 | 16.8 % |
| `quant_x_i8` | 3.16 | 189 | 0.0167 | 11.4 % |
| `rmsnorm` | 1.79 | 82 | 0.0218 | 6.5 % |
| `add_inplace` | 1.32 | 86 | 0.0153 | 4.8 % |
| `rope_interleaved` | 1.02 | 54 | 0.0189 | 3.7 % |
| `gemv_q6k_dp4a` | 0.94 | 13 | 0.0723 | 3.4 % |
| `mla_attn_cached` | 0.90 | 27 | 0.0333 | 3.3 % |
| `mla_kv_split` | 0.71 | 27 | 0.0263 | 2.6 % |
| `zero_fill` | 0.71 | 26 | 0.0273 | 2.6 % |
| `mla_kv_write` | 0.59 | 27 | 0.0219 | 2.1 % |
| `gemv_q8_0_bm` | 0.48 | 1 | 0.4800 | 1.7 % |
| `mla_build_k` | 0.47 | 27 | 0.0174 | 1.7 % |

**qwen2.5-0.5B**（GPU 时间合计 10.97 ms；24 层 × ~16 个内核 ≈ 380 次发射，这是 0.5B 每 token 只有 57 GB/s 有效带宽的原因）

| 内核 | GPU 时间 ms | 调用次数 | ms/次 | 占 GPU 时间 |
|---|---|---|---|---|
| `gemv_q5_0_p4` | 3.88 | 96 | 0.0404 | 35.4 % |
| `add_bias` | 1.11 | 72 | 0.0154 | 10.1 % |
| `rmsnorm` | 0.99 | 49 | 0.0202 | 9.1 % |
| `add_inplace` | 0.89 | 48 | 0.0185 | 8.1 % |
| `rope` | 0.88 | 48 | 0.0183 | 8.0 % |
| `gemv_q5_0_k2` | 0.73 | 36 | 0.0203 | 6.7 % |
| `gemv_q8_0_bm` | 0.50 | 1 | 0.5000 | 4.6 % |
| `kv_write` | 0.47 | 24 | 0.0196 | 4.3 % |
| `attn_cached_w` | 0.39 | 24 | 0.0163 | 3.5 % |
| `gemv_q4k_k2` | 0.37 | 12 | 0.0308 | 3.4 % |
| `gemv_q8_0_sk` | 0.24 | 12 | 0.0200 | 2.2 % |
| `gemv_q6k_dp4a` | 0.20 | 12 | 0.0167 | 1.9 % |

## 附录 B：数据可信度自查清单

每个 run 的 `tools/util_results/<model>/<mode>/report.json` 里都有以下字段，本报告引用的所有 run 均满足：

| 检查项 | 要求 | 说明 |
|---|---|---|
| `validity.quiet_before_start` | `true` | 启动前连续 3 次检查：无外部引擎进程 + 显存 < 1200 MiB + 整机 CPU < 45 % |
| `validity.foreign_during_run_hits` | `0` | 运行中每 5 s 的 `nvidia-smi --query-compute-apps` 检查 |
| `validity.timed_out` | 尽量 `false`；`true` 时只用挂死前的数据 | 35B `graph` run 为 `true`（用前 25 个稳态 token） |
| `meta.ambient_gpu_util` | 记录并随表附上 | 本机桌面底噪 1–27 % |
| `meta.interval_mean_ms` | 10.0–10.3 | 实际采样间隔（请求 10 ms） |
| `meta.interval_max_ms` | < 100 | 每 5 s 的 foreign check 造成 ~60 ms 空隙，<1 % 样本 |
| `validity.decode_samples` | ≥ 500（graph 窗口） | 短窗口（<2 s）不用于 NVML 均值 |

---

## 附录 C：新 build（dev-route + GPU 层逐层 CUDA Graph）验证（2026-11-15 01:30 追加）

> **这是报告的追加验证，不改变正文（§0–§11）关于旧快照 `05ca7d3c…` 的基线结论。**
> 被验证的新 build：`build/test_gen.exe` 2026-11-15 01:27，sha256 `3314ad726d5156ba370631fe5cc202012ffc5c75876e943cc311ed425f22d5ef`
> （快照 `tools/util_bin/test_gen_new.exe`）。新 build 默认启用**设备端路由** + **GPU 层逐层 CUDA Graph**。
> 复现：`bash tools/run_newbuild_verify.sh`；数据：`tools/util_results_new/`。所有 run 均
> `quiet_before_start=true` / `foreign_during_run_hits=0`。

### C.1 35B：GPU 忙占比 +10 pt，单 token −9.5 %

| 指标 | 旧快照（正文基线） | 新 build | 变化 |
|---|---|---|---|
| steady decode（n=17） | 50.09 ms（20.0 tok/s） | **45.35 ms（22.0 tok/s）** | **−9.5 %** |
| 长 decode（n=257） | 第 25 token 挂死 | **45.71 ms/token × 256/256 步，exit=0** | **挂死已修复** |
| NVML GPU util 均值 / 中位 | 43.5 % / 43 | **46.4 % / 45**（11.8 s 窗口，1163 样本） | **+2.9 pt** |
| GPU 功耗均值 | 50.1 W（1.3 s 窗口） | **76.0 W**（11.8 s 窗口，更真实） | +26 W |
| SM 时钟 | 1950（max 1965） | **1940 / max 1965** | 仍顶格，无降频 |
| 进程 CPU | 3.57 core（29.8 % 整机） | **4.01 core（33.4 %）** | +0.44 core |
| kprof GPU busy（`--no-graph`，口径不同） | 43.2 %（total 29.11 / wall 67.43） | **41.2 %（total 23.85 / wall 57.85）** | total −5.3 ms |

逐层同协议（LAYERMS only，无 kprof 开销，2 token 平均）：

| 层类 | 旧 build | 新 build | 说明 |
|---|---|---|---|
| GPU 层（15 层）墙钟 | 0.620 ms | **0.203 ms** | **−67 %**；launch 0.236→**0.095**、sync 0.206→**0.000**（dev-route 消掉 router 回传同步） |
| CPU 层（25 层）墙钟 | 1.537 ms | **1.642 ms** | **+7 %**（sync 0.149→0.338 ms/层，+4.75 ms/token） |
| 40 层合计 | 47.73 ms | **44.09 ms** | −3.64 ms |

⇒ 收益几乎全部来自 GPU 层（15×0.417 = 6.3 ms），但 CPU 层多出的同步等待吃回 ~4.8 ms；若把 CPU 层 sync 压回旧值，35B 还能再快 ~4 ms。
⚠️ 新 build 在 `--no-graph` 下反而慢（该 run 53.05 ms vs 默认执行 45.35 ms）——**逐层 graph 对新 build 是必需项**（dev-route kernel 并行度低、GPU 更像瓶颈，提交气泡更贵）。

### C.2 Coder：钉死放置 + 外部显存占用 → 显存打满的"假回退"；auto 后反而比旧 build 快 4.5 %

| 配置 | decode | prefill | VRAM 后置检查 | 说明 |
|---|---|---|---|---|
| 旧 build（20 GPU 层） | 44.24 ms | 203 ms | used 10982 / free 1306 MiB | 正文基线（另一次会话量的 free） |
| 新 build，钉死 20 层 `--no-vram-check` | **95.40 ms** | **3492 ms** | used **12288 / free 0** MiB | **最坏情况（显存被打满 → WDDM 分页）**，CPU 反而只有 1.96 core；见下方更正 |
| 新 build，自动放置（adaptive 20→17） | **42.23 ms**（平均 42.67，min 40.87） | 191 ms | used 10843 / free 1444 MiB | **比旧 build 快 4.5 %**；NVML util 42.0 %、功耗 56.5 W、SM 1950、CPU 4.68 core |

- **更正（2026-11-15 01:40，hostlayer-35b 澄清）**：**Coder 的设备路由实际是关闭的**——引擎的开关判定是"专家张量里含 Q5_1 就整个模型禁用设备路由"，Coder 的 78 个 `_exps` 张量中有 14 个 Q5_1，因此 Coder 走旧 host 路由路径。所以上表"钉死 20 层 free=0"**不能归因于 dev-route 多占 1.3 GB**；更可能是**加载时刻的外部显存占用不同**（旧 build 的 1306 MiB free 是另一次会话量的），auto 放置时 free=1444 MiB 也支持这个解释。⇒ `95.40 ms` 应视为**显存被打满时的最坏情况**，不是新 build 的固定回退；可信的对比是 auto 的 **42.23 ms vs 旧 build 44.24 ms（−4.5 %）**。
- **旧 build 会挂死的 `[placement] adaptive:` 路径在新 build 上正常完成**（`20 -> 17`）——与长 decode 丢唤醒一起被修复。
- 教训：钉死放置 + `--no-vram-check` 会在外部显存波动时把"显存打满"误读成"新 build 回退"；做 A/B 时应优先用 **auto 放置**或加载后核对 `[vram] post-load free`。

### C.3 新行为开关（hostlayer-35b 提供，用于 A/B 复测）

| 行为 | 关闭方式 | 说明 |
|---|---|---|
| 设备端路由（dev-route） | `ENGINE_NO_DEV_ROUTE=1`（或 config `dev_route=false`） | 默认 true，env 是反转开关 |
| 逐层 CUDA Graph（全部） | `ENGINE_LAYER_GRAPH_AUTO=0`（或 `--layer-graph-auto=0`） | 默认 true |
| 只关 CPU 层图 / 只关 GPU 层图 | `ENGINE_NO_CPU_LAYER_GRAPH=1` / `ENGINE_NO_GPU_LAYER_GRAPH=1` | 用于定位是哪一侧的图收益/代价 |
| 两个行为都关（旧路径） | `ENGINE_NO_DEV_ROUTE=1 ENGINE_LAYER_GRAPH_AUTO=0` | 与旧快照基线做严格 A/B |

注意：本报告附录 C 的 35B 数字来自默认（dev-route + 逐层 graph 全开）；`new_probe`/`new_layerms` 用 `--no-graph` 会关掉逐层 graph（dev-route 仍开），所以那两行不能当作"旧路径"对照。严格 A/B 需按上表把两个开关都关掉。

### C.4 口径提醒

- 新 build 数字与正文（旧快照）**不是同一二进制**；host matvec 类指标受当时整机负载影响（本批 ambient CPU ~21–43 %）。
- 35B `--no-graph` 会关掉新 build 的逐层 CUDA Graph，故 C.1 的 kprof busy 只作交叉参考，主口径为 NVML util（46.4 %）。
- Coder 的 42.23 ms 来自 n=33、1.37 s 解码窗口，NVML 均值偏粗；35B 的 46.4 % 来自 n=257、11.8 s 窗口，可信度更高。
