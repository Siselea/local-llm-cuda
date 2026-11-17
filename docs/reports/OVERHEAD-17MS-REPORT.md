# 17 ms 可回收开销 — 取证、实现与结论报告（OVERHEAD-17MS）

日期：2026-11-15 · 目标：Qwen3.6-35B-A3B decode **38.6 ms → ≤25.9 ms（≥38.6 tok/s = llama.cpp 25.76 的 150%）**
范围：`src/runner.cpp`（`run_moe` 的 CPU 层执行序、MoE 尾部合并）、`include/engine/kernels.h`
（新增 `add_three_inplace`）、`include/engine/runner.h`（kernel 句柄），新增验证/测量工具
`tools/b35.sh`、`tools/ab35.sh`、`tools/ab35_final.sh`。

---

## 0. 一句话结论

> **任务书里"每 CPU 层 28–40 次 launch + 2 次 stream_sync"的现状已经不存在**：上一轮
> （FINAL-SPRINT + hostlayer）已经把 CPU 层压到 **10 次 launch / 1 次 sync**，16 个 GPU 层
> 也从 0.53 ms 降到 **0.19–0.20 ms/层**。任务书预估的 ~17 ms 可回收里，**约 13 ms 已经被花掉了**。
>
> 本轮真正找到并修掉的是**一个新的、此前没被定量的开销**：每个 CPU 层在 sync 之前排队的
> GPU 工作（zerofill + 共享专家 6 次 launch + router）会与紧随其后的宿主 GEMV 抢 CPU/内存带宽。
> 把这些 GPU 工作**移到 sync 之前**、让宿主 GEMV 面对一个**空的提交队列**，引擎内
> `moe_forward_mt` 从 **1.12–1.14 ms/层降到 0.96 ms/层**，端到端
> **38.67 → 37.22 ms（−1.22 ms，−3.2%，交错 A/B 中位）**，且**逐 token bit-exact**。
>
> **150% 目标未达成**（差 11.3 ms）。37.22 ms 里 **23.0 ms 是宿主 GEMV**（24 × 0.96），
> 这条路径已被前一轮证明是**算力受限**（内核上限 24.8–26.3 GB/s，隔离基准 0.62–0.65 ms/层），
> 剩下约 14 ms 是 24 次宿主往返 + 16 个 GPU 层 + head。**精确缺口分解见 §5。**

---

## 1. 现状复核（任务书的表已经过期）

用 `ENGINE_DBG_LAYERMS=1` 逐层实测（35B，默认配置，`ENGINE_DEC_SKIP=2`）：

| 任务书假设 | 实测现状（本轮改动前） |
|---|---|
| CPU 层 28–40 次 launch，0.22 ms | **10 次 launch，0.118 ms** |
| CPU 层 2 次 `stream_sync`，0.14 ms | **1 次，0.17 ms**（含等待本层 attention 真正算完） |
| 16 个 GPU 层 0.53 ms/层，合计 8.5 ms | **0.19–0.20 ms/层，合计 3.1 ms**（12 次 launch） |
| 收尾 1–2 ms | 约 2.5–3.5 ms（lm_head Q8_0 540 MB ≥1.6 ms 是硬底） |
| 宿主 GEMV 0.77 ms/层 | **1.12–1.14 ms/层**（`ENGINE_DBG_CPUEXEC`）← 比任务书更差 |

即：任务书表里"WDDM 提交 5.3 ms + sync 3.4 ms"的大部分，上一轮已经拿走了；
真正还剩下的大头是 **宿主 GEMV** 和 **每层一次的宿主往返延迟**。

实测每 token 计数（`ENGINE_PIPE_PROF`，改动前）：**432 次 launch、24 次 sync、宿主 GEMV 24 次**。
（`ENGINE_PIPE_PROF` 自己会每 MoE 层多插 1 次 `cuda::sync()`，所以它的 sync 计数是 50，不是真实值。）

---

## 2. 本轮改动（2 项，均 bit-exact）

### 2.1 A：把 CPU 层的 GPU 侧工作全部挪到唯一那次 sync 之前（主收益）

`run_moe` 的 CPU 层执行序原来是：

```
router gemm(GPU) → D2H×2 → stream_sync ─┬─ host softmax/top-k
                                        ├─ zerofill + 共享专家 gate/up/silu/down/gate/add (GPU×6)
                                        └─ host GEMV（moe_forward_mt，9 线程，0.62 ms 的活）
                                        → H2D + add + add
```

问题：**共享专家的 6 次 launch 和它们的 WDDM 提交/刷新，正好与宿主 GEMV 并行发生**。
宿主 GEMV 是 9 线程、算力受限的紧循环，提交线程来抢 CPU 和内存带宽时它就跑不满。
（这也解释了为什么"引擎内 0.77–1.14 ms"始终比隔离基准 0.62–0.65 ms 慢 20–80%。）

改成：

```
router gemm(GPU) → D2H×2 → zerofill + 共享专家(GPU×6) → stream_sync（唯一一次，把队列排空）
→ host softmax/top-k → host GEMV（此时 GPU 队列为空、无提交工作）
→ H2D + x += (xout + host_part)（融合成 1 次 launch，见 2.2）
```

- 数值：**同内核、同顺序**（共享专家对 `xout_` 的加法顺序不变，router 与共享专家互相独立）
  ⇒ **逐 token bit-exact**：35B n=40 的 `gen:` 序列 md5 与改动前完全相同（`e9f0a5a6`），
  `topk: 2614(...)` 也完全相同。
- 实测：`ENGINE_DBG_CPUEXEC` 的 `batched host matvec avg` **1.12–1.14 → 0.96 ms**（−0.16 ms/层，
  −14%）⇒ 24 层 −3.8 ms 的潜在收益（端到端体现 −1.2 ms，其余被 GPU 侧重叠吸收）。
- 对所有"专家卸载到 CPU"的模型生效，无模型名特判；GPU 驻留层走原来代码路径（不受影响）。

### 2.2 B：MoE 尾部两次 add 融合成一次（`add_three_inplace`）

原来尾部：`H2D(host_part)` → `xout += host_part` → `x += xout`（3 次提交）。
新增内核 `add_three_inplace(a,b,c,n): a += (b + c)`，等价于 `b += c; a += b`（中间量同样只舍入一次），
所以**逐位相同**。CPU 层 launch 从 **10 → 9**（每 token 少 24 次提交）。
`ENGINE_DUMP` / `ENGINE_DUMP_MOE_SPLIT` 调试路径保留原来的两步合并，dump 内容不变。

---

## 3. 结果

### 3.1 35B decode：交错 A/B（`tools/ab35_final.sh`，n=40 → 38 个计时步，`ENGINE_DEC_SKIP=2`，3 轮换序）

| 轮次 | 改动前 `test_gen_base.exe` | 改动后 `test_gen.exe` |
|---|---|---|
| rep1（base 先） | 38.77 ms | 37.80 ms |
| rep2（new 先） | 38.67 ms | 37.22 ms |
| rep3（base 先） | 37.92 ms | 36.69 ms |
| **中位** | **38.67 ms（25.86 tok/s）** | **37.22 ms（26.87 tok/s）** |
| 均值 | 38.45 ms | 37.24 ms |

**Δ = −1.22 ms（−3.2%），tok/s 25.86 → 26.87（llama.cpp 25.76 的 104.3%）。**
（上一轮报告记录的同机基线是 37.98 ms；本轮改动前实测 38.45–38.67，漂移在虚拟机正常范围内。）

### 3.2 逐项账本

| 项 | 前 | 后 | 说明 |
|---|---|---|---|
| CPU 层 launch 次数/token | 432 | **408** | 24 层 ×(10→9) |
| CPU 层 `stream_sync` 次数/token | 24 | **24** | 每层 1 次不可省（宿主需要激活） |
| CPU 层 sync 墙钟/token | 4.2 ms | 4.2 ms | 其中 ~0.10 ms/层是 attention 真算的时间 |
| CPU 层 launch 墙钟/token | 2.8 ms | **2.5 ms** | 0.118 → ~0.106 ms/层 |
| **引擎内宿主 GEMV** | **1.12–1.14 ms/层** | **0.96 ms/层** | −14%，本轮主要收益 |
| GPU 层/层 | 0.19–0.20 ms | 0.19–0.20 ms | 未动 |
| 35B decode | 38.67 ms | **37.22 ms** | −1.22 ms |

### 3.3 Coder（同为 CPU 卸载模型）

| 轮次 | 前 | 后 |
|---|---|---|
| rep1 | 41.38 ms（24.17 tok/s） | 41.36 ms（24.18 tok/s） |
| rep2 | 41.98 ms | 42.18 ms |

**无变化（噪声内）**。原因已取证：Coder 的 CPU 层（layers 18–26，9 层）**sync 只有 0.01–0.02 ms**
（它的设备/宿主数据流不同），本轮修的是"提交工作与 GEMV 抢资源"这条通路，它本来就不存在；
Coder 的 CPU 层是 **27 次 launch、2.5–3.1 ms/层**（arena 流式 + f32 宿主权重），瓶颈在别处。
⇒ **没有回退**（Coder 23.9 tok/s ≥ 23），但**也不能声称 Coder 受益**。

### 3.4 kprof 逐内核（最终二进制，`ENGINE_KPROF=1`，最后一次 decode token，13 步计时）

`[kprof] total GPU time 11.90 ms over 13 kernel kinds · GPU busy 28.5%`（该 run decode 41.88 ms；
kprof 的 CUDA event 对会在每对 launch 之间引入提交间隙，绝对值偏大，只用于**相对归因**）：

| 内核 | ms/token | calls | 归属 |
|---|---|---|---|
| `gemv_f32` | 3.60 | 120 | **24 个 CPU 层的 attention 投影**（5 次/层）← 这就是 sync 里等待的 GPU 工作 |
| `moe_gateup_silu_dev` | 2.08 | 16 | 16 个 GPU 层专家（0.13 ms/层） |
| `gemv_q8_0_bm` | 1.88 | 1 | **lm_head**（Q8_0 540 MB ⇒ 287 GB/s，head 的硬底） |
| `add_inplace` | 1.27 | 56 | 共享专家/尾部 add（含提交间隙） |
| `moe_down_scatter_dev` | 0.89 | 16 | GPU 层专家 |
| `moe_route` | 0.61 | 16 | GPU 层设备端路由 |
| `add_three_inplace` | 0.57 | 24 | **本轮新增**（替代原来的 2 次 add） |
| `moe_shared_gate` | 0.43 | 40 | 共享专家 sigmoid 门 |
| `rmsnorm4` | 0.22 | 41 | |
| `silu_mul` / `zero_fill` / `topk_*` | 0.35 | | |

⇒ **16 个 GPU 层本身每 token 约 3.6–4.3 ms**（0.22–0.27 ms/层，含提交间隙），
与任务书里的 8.5 ms 相比已经减半；本轮**没有改动** GPU 层路径（整层图实测更慢，见 §6.1）。

### 3.5 GPU 功耗（n=400 长 decode，NVML 采样 5/sec）

| 指标 | 值 |
|---|---|
| SM 时钟（decode 窗口） | **1777 MHz**（idle 210 MHz；未降频） |
| 功耗（decode 窗口） | **35–37 W**（mean 34.9） |
| GPU util（瞬时采样） | ~10%（每 token GPU 真正忙 ~8–10 ms / 37 ms） |

**这不是"GPU 没在算"**：GPU 层的 kprof 与墙钟都是实打实的，功耗低是因为 **40 层里 24 层在宿主
CPU 上执行**、GPU 两头等（同机 9B 全 GPU 时同一套代码功耗 148 W）。要让 35B 的 GPU 功耗上去，
必须减少 CPU 层数量 —— 而那被 12 GB 显存限死（16/40 层）。


### 3.6 显存余量实验（配置项，未改默认值）

`--vram-margin 400` 让 35B 多放 1 个 GPU 层（16 → 17）：同会话两次运行 **37.88 → 37.24 ms（−0.64 ms）**。
默认仍保留 1200 MiB 安全余量；要那 0.6 ms 可以显式传 `--vram-margin 400`（代价：运行时只剩 ~950 MiB 余量）。

### 3.7 最终二进制单次实测（安静窗口）

n=60：**36.60 ms/token（27.3 tok/s）**；n=400（399 步）：**36.92 ms/token（27.1 tok/s）**。
（A/B 那一组数字 38.67/37.22 是同会话换序交错的，受当时背景负载影响略慢；两者不矛盾。）

---

## 4. 四套验证（最终二进制 `build/test_gen.exe`，2026-11-15 17:52 重建）

| 套件 | 结果 |
|---|---|
| `tools/verify.sh` | **ALL GREEN**（build + Python 单测 + G0 IR parity + M2 plan parity，7 个模型全 PASS） |
| `tools/smoke_models.sh` | **40716 / 185 / 2614 / 2614**（4/4 模型原 top-1） |
| `tools/gen_regression.sh` | **10 passed, 0 failed — GEN REGRESSION GREEN**（35B 3 例 × 8–12 token 与 llama.cpp 逐 token 相同） |
| `tools/verify_generality.sh` | **26 passed, 0 failed — GENERALITY GREEN** |

数值证据：本轮两项改动都是**同内核、同舍入顺序**（2.1 只是提交顺序，2.2 是等价代数变换），
所以 35B n=40 的 `gen:` 序列 md5 与改动前**完全相同**（`e9f0a5a6`），全部 4 个模型的 top-1 不变。

---

## 5. 剩余缺口：为什么 150% 到不了（精确分解）

> **测量口径警告（本轮重要发现）**：`ENGINE_DBG_LAYERMS=1` 每层往 stderr 打一行，在本机
> 每行约 200–300 µs（stderr 无缓冲 + 文件在 D:），40 层/token ⇒ **该开关自己给每 token 增加
> 约 11 ms**。所以"逐层之和 32 ms + decode 43 ms"的差不是 head，而是这个打印开销
> （两台不同 run 的差值两次都是 11.0–11.1 ms，可复现）。逐层**墙钟**仍可用（打印发生在层边界），
> 但**不能用逐层和去反推端到端**；下面用干净 run 的端到端 + 计数 + kprof 来分解。

干净 run（**37.2 ms**，A/B 中位；安静窗口 36.6 ms）的构成：

| 项 | ms/token | 依据 | 可动性 |
|---|---|---|---|
| **宿主 GEMV** | **23.0** | `ENGINE_DBG_CPUEXEC` 0.96 × 24（本轮从 1.12 降下来） | 算力受限：内核算力上限 24.8–26.3 GB/s，隔离基准 0.62–0.65 ms/层 |
| CPU 层 attention（GPU 真实工作） | 3.6 | kprof `gemv_f32` 120 calls | 23 MB/层 ÷ ~300 GB/s ≈ 0.08 ms 是地板，现在 0.15 ms/层（含提交间隙） |
| 16 个 GPU 层（专家 + 路由 + 提交） | 3.6–4.3 | kprof dev 内核 3.58 + 提交 | 0.22–0.27 ms/层，已比任务书假设的 0.53 减半 |
| head（lm_head + norm + topk + D2H） | 2.0–2.5 | kprof `gemv_q8_0_bm` 1.88 ms | lm_head 540 MB ≥1.6 ms 是硬底 |
| 24 × 宿主往返（sync flush + 9 次提交/层） | ~3.5 | sync 0.17/层（含 0.10 等待 attention）、launch 0.106/层 | 每层 1 次 sync 不可省（宿主需要激活） |
| routing/merge/其他 | ~1 | | |

**物理下界**（本设计、不改数值精度）：

- 每 token 必须从 DRAM 读 **24 层 × 8 专家 × 2.04 MB = 391 MB** 专家权重。纯读 DRAM 天花板
  32.5 GB/s ⇒ **12.0 ms**；本内核算力上限 26.3 GB/s ⇒ **14.9 ms**。
- 24 层 attention + 16 个 GPU 层 + head 的 GPU 真实工作 ≈ **2.4 + 2.4 + 1.8 = 6.6 ms**，
  且每层 attention 的输出是下一层宿主 GEMV 的输入，**无法跨层重叠**（GPU 空闲 70% 是这个原因）。
- 24 次宿主往返 ≈ **3.6 ms**。

⇒ **绝对下界 ≈ 14.9 + 6.6 + 3.6 ≈ 25.1 ms**。25.9 ms 目标**恰好压在这条设计路径的物理地板上**：
必须"宿主 GEMV 跑到隔离速度 0.62（−8 ms）+ 往返开销再砍 2/3"才可能。
实测 37.2 ms 与下界之间的 12 ms 缺口里，**约 8 ms 是宿主 GEMV 的引擎内损耗（0.96 vs 0.62）**，
其余在 WDDM 往返与 GPU 层提交延迟。宿主机是 6C/12T 虚拟机且常驻 LANDrop/Steam/QQ/Edge 等后台
进程，9 线程 GEMV 只要被抢 1 个逻辑核就掉 ~11%；上一轮已证明 pinned/页表、预取、绑核、int8、
线程数（6/8/9/10/12）都无效或更差 ⇒ **在不改数值精度的前提下这条路径没有 150% 的空间**。

### 想再往前走，只剩这几条（都有硬代价，本轮没做）

1. **把专家 GEMV 分给 GPU（PCIe 流式）**：391 MB/token，PCIe 3.0 实测 ~12 GB/s ⇒ 光传输 32 ms，
   还要和 CPU 抢 DRAM 带宽；即使 50/50 拆分 + 完美重叠，也只能回收 2–3 ms，且破坏 bit-exact。
2. **CPU 层改用设备端路由**（省宿主 softmax/top-k ~0.02 ms/层 = 0.5 ms）：路由权重出现 ~1e-7 相对差。
3. **投机解码（MTP）**：已被 `MTP-IMPL-REPORT.md` 实测证伪（k=1 67.2 ms vs 41.8 ms；本引擎
   n>1 前向成本 ≈ 2.0×，MoE 的 8/256 路由让两个 token 几乎不共享权重）。
4. **17 个 GPU 层**：已验证 −0.64 ms（`--vram-margin 400`），但吃掉安全余量，未做默认。
5. **把 GPU 层的 12 次 launch 再压**（整层图更慢，见 §6.1）：需要新写"设备端路由 + 融合专家"
   的单一内核，预期 0.5–1.0 ms，工作量最大。

---

## 6. 失败的尝试（如实）

1. **`ENGINE_GPU_FULL_LAYER_GRAPH=1`（GPU 层整层录图）**：交错 A/B 中位 **43.15 vs 41.72 ms（更慢）**。
   原因清楚：整层图必须走设备端路由（`k_moe*gateupdev*`），而默认路径的 **host 路由 + 融合内核
   `moe_gateup_silu`/`moe_down_scatter` 更快**。默认保持关闭。
2. **把 CPU 层尾部（zerofill+共享+两个 add）录成 2 张 CUDA Graph**：没有实施，先做了成本核算——
   GPU 层实测"图替换 ~12 次 launch"只省 0.076 ms/层 ⇒ 本机 `cuGraphLaunch` 约 0.05–0.1 ms，
   而 CPU 层尾部是 9 次 launch（0.118 ms），换成 2 次图启动大概率是**负收益**；
   另外 `ENGINE_NO_CPU_LAYER_GRAPH` 的 A/B（42.44 vs 41.72）也说明图启动并不便宜。**放弃。**
3. **`ENGINE_NO_SHARED=1` 诊断**：该二进制在装载阶段**挂死**（pool 丢唤醒的另一种触发，见下）——
   属于诊断开关路径，未追（默认路径 n=40 复跑 6 次无挂死）。
4. **`ENGINE_NO_DEV_ROUTE=1`（43.53 vs 41.72）/ `ENGINE_NO_CPU_LAYER_GRAPH=1`（42.44 vs 41.72）**：
   都确认当前默认值是对的，没有回退空间。

---

## 7. 复现命令

```bash
cd engine
bash tools/verify.sh                      # ALL GREEN
bash tools/smoke_models.sh                # 40716 / 185 / 2614 / 2614
bash tools/gen_regression.sh              # 10/10（35B 3 例逐 token == llama.cpp）
bash tools/verify_generality.sh           # GENERALITY GREEN (26/26)

# 本轮 A/B（改动前二进制 = build/test_gen_base.exe，改动前源码快照 src/*.pre-17ms）
bash tools/b35.sh                         # 单配置多次（TAG=... REPS=... N=40）
bash tools/ab35.sh  "base:" "variant:ENV=1"   # 多配置交错
bash tools/ab35_final.sh                  # 最终 base vs new（3 轮换序，n=40）

# 逐层取证
M=<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_DBG_LAYERMS=1 build/test_gen.exe "$M" 760 12
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_DBG_CPUEXEC=1 build/test_gen.exe "$M" 760 24
```

改动文件：`src/runner.cpp`（run_moe 执行序 + 尾部融合）、`include/engine/kernels.h`（`add_three_inplace`）、
`include/engine/runner.h`（`k_add3_`）。改动前源码快照：`src/runner.cpp.pre-17ms`、
`include/engine/runner.h.pre-17ms`（kernels.h 的快照是 `build/bak/` 之前的版本 + §2.2 的 diff）。

## 8. 附：本轮环境的额外事实（供下一轮参考）

- **本机噪声很大**：同一二进制连续跑 35B，decode 可在 36.7–42.1 ms 之间漂移（±7%），
  所以所有结论都用**换序交错 A/B 的中位**，单次运行不可作为判据。测量窗口 ≥38 个计时步（n=40）时
  才稳定到 ±0.3 ms。
- **装载时间 ~40–55 s**（19 GB 专家权重从磁盘读入 + 6–7 GB 上传显存），是 A/B 成本的主要来源；
  页面缓存被挤掉时会涨到 2–4 min/次。
- **GPU 功耗**：35B decode 期间 GPU 只有 **~45–55 W**（空闲 210 MHz / 解码窗口 1.9–2.0 GHz，util 40–55%），
  因为 40 层里 24 层在宿主 CPU 上跑、GPU 每个 token 真正忙 ~7 ms。这与用户"功耗应 >90 W"
  的直觉不符，但**不是 GPU 没在算**：GPU 层的 kprof/墙钟是实打实的（每层 0.19 ms），
  功耗低是 **CPU 边界**造成的（9B 全 GPU 时同样代码功耗 148 W）。
- **`ENGINE_NO_SHARED=1` 会挂死**（装载后卡住，pool 丢唤醒的另一条触发路径）；默认配置未复现，
  属诊断开关路径的已知脆弱点。跑长 decode 时建议 `taskkill` 兜底。

