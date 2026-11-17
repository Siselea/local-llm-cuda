# CPU 专家路径「权重 repack / 带宽」方向 — 取证与结论报告

**日期**：2026-11-15（repack sprint）
**动工范围**：`src/cpu_matvec.cpp`（Q4_K/Q6_K 行内核、两级 int8 内核）、`src/runner.cpp`（CPU 层分段计时，临时）、
新增探针 `tools/bench_rp.cpp`
**最终状态**：**所有产品代码改动已回退，工作树与动工前逐字节等价**（见 §8 的 md5 证据）。
本轮产出是**认知与取证**，不是代码：任务书的核心前提被数据推翻。

---

## 0. 一句话结论

> **「35B 的 CPU 专家层是内存带宽受限（19.4 GB/s，可推到 30 GB/s）」是错的。**
> 把数据全部放进 L3（内存完全免费）时，生产 Q4_K 行内核本身只有 **24.8 GB/s**；
> DRAM 下 25–26.5 GB/s，**已经贴住它自己的计算上限**。纯读 DRAM 天花板确实有
> **32.5–34.8 GB/s**，但内核永远摸不到——**这条路径是计算受限，不是带宽受限**。
> 因此 repack（改内存布局）方向没有任何物理空间；预取/行交错/绑核/两级 int8 全部实测无效或负收益。

---

## 1. 为什么必须换测法（方法论）

旧结论「19.4 GB/s」有两个叠加误差：

1. **跨进程 A/B 不可用**：本机是 6 核 12 线程 VM，还跑着桌面/服务/其他 agent，同一份二进制的
   decode/wall 数字可以漂 **±20–30%**。几分钟前跑的二进制和几分钟后的二进制比，分辨不出 10% 的改动。
2. **引擎内 vs 隔离基准混用**：隔离基准是 0.65–0.68 ms/层（24 副本轮转），而引擎内
   `moe_forward_mt` 实测 0.77–0.84 ms/层（含线程池冷唤醒、WDDM 提交争用），两者差 15–25%。

新探针 **`tools/bench_rp.cpp`** 解决这两点：

- **单进程内交错**所有内核变体（每轮换一个变体），所有变体看到同一机器状态 ⇒ 漂移被消掉；
- **每轮换一个新的 9 MB 副本**（24 副本轮转 ⇒ 工作集 216 MB > 10×12 MB L3），否则同一轮里
  第一个变体把数据拉进 L3，后面的变体白嫖缓存（我第一版就踩了这个坑，测出 simd 变体"快 35%"
  是假的）；
- 报 **min / p25 / median**（本机噪声下 min 最稳，median 看持续性）；
- 直接 `#include "../src/cpu_matvec.cpp"`，**bench 的就是生产内核本身**，不是拷贝；
- 关键对照：**`copies=1`（9 MB 常驻 L3）** 用来分离「计算上限」和「内存上限」。

---

## 2. 决定性证据：计算上限 vs 内存上限

35B CPU 层几何：hidden=2048、inter=512、topk=8 ⇒ phase A（gate+up Q4_K）= 9.0 MB，
phase B（down Q6_K）= 6.6 MB，一层 16.3 MB。9 线程。

### 2.1 数据全在 L3（copies=1，内存免费）时，内核仍然只有 24.8 GB/s

| 变体（同一份字节、同一访问顺序） | L3 常驻 GB/s（纯计算上限） | DRAM 24 副本 GB/s |
|---|---|---|
| 纯 32B 读循环（`readA32`，不做运算） | 114–153 | **32.5–34.8** |
| **生产 Q4_K fp32 行内核** | **24.8** | **25.0–26.5** |
| 去掉 scale 解码（`noscale`，常数代替） | 33.4 | 26.5 |
| 去掉 32 次 x 载入（`nox`） | 28.4 | — |
| 只做 load + nibble 提取，不做 FP（`unpack`） | 63.8 | — |

**读法**：数据全在 L3 时读循环能跑 114–153 GB/s（说明不是 L3 带宽问题），而生产内核只有
24.8 GB/s。**内核的算力才是 25 GB/s 的那堵墙；DRAM 的 32.5 GB/s 从来没被摸到过。**

### 2.2 线程扩展曲线也是算力形状（L3 常驻，phase A）

| 线程 | 1 | 2 | 3 | 4 | 6 | 9 | 12 |
|---|---|---|---|---|---|---|---|
| 纯读 GB/s | 33 | 57 | 70 | 79 | 119 | 153 | 140 |
| **生产内核 GB/s** | **5.84** | 11.6 | 13.8 | 17.0 | 20.4 | 24.8 | 23.2 |

单线程 5.84 GB/s（与上一轮报告"单核 5.7 GB/s"一致，当时测的是 cache 常驻）。SMT 只贡献
~30%（9 线程 vs 6 线程），12 线程反而更慢（SMT 争用 p1 口）。

### 2.3 指令级解释（为什么 25 是硬的）

Q4_K 一个 144 B 块 = 256 值，生产内核 **208 条指令/块**，其中：

| 端口类别 | 操作 | 每块条数 |
|---|---|---|
| `vcvtdq2ps`（**Comet Lake 上只在 p1，1/cycle**） | int32→float | **32** |
| `vfmadd`/`vfmsub`（p0/p1，各 1/cycle） | 乘加 | 64 |
| `vpmovzxbd`（p5，1/cycle） | 8×u8→8×i32 | 16 |
| `vpand`/`vpsrld` | nibble 拆分 | 16 |
| x 载入（32B） | 激活 | 32 |
| 权重载入（8B） | 权重 | 8 |

p1 口每块至少要 **32（cvt）+ 32（一半 FMA）= 64 cycle**；6 物理核 × 3.19 GHz（VM 实测频率）
⇒ 理论 43 GB/s。但实测只有 25——剩下的是 SMT 争用、端口调度与内存延迟叠加的现实值。
**结论：要把 25 推高，必须砍掉 96 个 FP 操作中的一大块；改内存布局不改变这个数。**

---

## 3. 引擎内 CPU 层开销分解（35B，GPU 窗口实测）

`ENGINE_DBG_LAYERMS=1` + 临时分段计时（`d2h / route / moe / up`），GPU 空闲窗口、服务已停、9 线程：

**16 个 GPU 层**：0.46–0.60 ms/层（典型 0.53）⇒ 8.5 ms/token
**24 个 CPU 层**：1.29–1.96 ms/层（典型 1.40）⇒ 33.6 ms/token

单层 CPU 层的构成（典型值，ms）：

| 段 | 典型 | 范围 | 隔离基准对照 |
|---|---|---|---|
| **`moe_forward_mt`（宿主 GEMV）** | **0.77** | 0.62–1.04 | **0.62–0.65** |
| WDDM 提交（**28–40 次 launch**） | 0.22 | 0.17–0.56 | — |
| `cuda::stream_sync`（2 次） | 0.14 | 0.05–0.66 | — |
| 宿主路由（softmax/top-k/打包） | 0.08 | 0.07–0.30 | — |
| `xout_host` 上传 + add kernel | 0.03 | 0.02–0.06 | — |
| 本层 GPU 注意力/共享专家实际执行 | ~0.16 | — | — |
| D2H 同步 | **~0.01** | 0.00–0.21 | 不是瓶颈 |

⇒ decode 预算（38.6 ms 中位）：**宿主 GEMV 18.5 ms（48%）+ GPU 层 8.5 ms（22%）
+ CPU 层内 GPU/同步/提交 ~11.6 ms（30%）**。

**关键修正**：隔离 vs 引擎内的 moe 差是 **0.12–0.15 ms/层（+20%）**，不是任务书假设的
0.2–0.35。我把两个可疑原因都排除了：

- **pinned 内存假设**：`hn_host` 来自 `cuMemAllocHost`。加 `ENGINE_HOST_COPY=1` 先把激活拷到
  页表内存（可缓存）再进内核 ⇒ 3 轮 wall 和 112.88 vs 113.29 ms，**无差异**（差 -0.4%，噪声内）。
- **线程池冷唤醒假设**：`ENGINE_POOL_SPIN=20000`（~0.8 ms 自旋窗口，覆盖 GPU 阶段）⇒
  111.50 vs 113.29 ms，**-1.6%，噪声内**。

⇒ 这 0.15 ms/层是**虚拟机 SMT + CUDA 驱动线程争用**的固有代价，不是可以在 CPU 侧改掉的东西。

---

## 4. 试过且无效 / 负收益的（全部有交错 A/B 数据）

### 4.1 权重 repack（任务书方向 A）——**未落地，因为前提不成立**
诊断数据（§2.1）显示：`noscale`（去掉 scale 解码）能把 L3 计算上限从 24.8 推到 33.4，
但 **DRAM 下只从 25.0 变成 26.5**——因为一旦指令变少，瓶颈就滑到"内存延迟/MLP"，
26.5 是这个内核形态的新地板。也就是说 repack 可能带来的最大收益是 **+5%**，
而 repack 的代价是 +2.9%（Q6_K）额外主机内存、加载时间和一整条新代码路径。**负期望，不做。**

### 4.2 软件预取 / 多行交错（方向 B）
| 尝试 | 结果 |
|---|---|
| 预取深度 0 / 1 / 2 / 3 / 4 / 6 行 | 全部在噪声内（最好 +5%，最好一次是噪声） |
| 行内**滚动**预取流（每块发 2 条，288 / 576 / 1152 B 提前量，T0/T1） | 24.3 / 25.3 / 27.8 GB/s（基线 26.4）⇒ 无稳定收益 |
| 2 行交错（共享 x 载入，8 累加器） | **20.1 GB/s（-20%）**：寄存器溢出（16 ymm 不够放 8 acc + 8 scale + 4 x） |
| 同块 2 块交错（2 条独立权重流） | 20.2 GB/s（-19%）同样溢出 |

**这就是上一轮"gate+up 融合内核"失败的原因，本轮量化确认：AVX2 的 16 个 ymm 寄存器装不下
两行 Q4_K 的累加器 + scale 广播 + x 载入，任何行交错都溢出。**

### 4.3 双缓冲（方向 C）——由 4.1/4.2 直接排除
数据流是纯顺序流、没有"加载/计算重叠"的余地：MLP 已经由硬件预取器 + 行首软件预取撑满，
而计算上限（25）低于内存上限（32.5），双缓冲不会让计算变快。

### 4.4 SIMD scale 解码（**唯一实现完的优化，bit-exact，但性能中性**）
把 8 个 6-bit scale/min 的标量解码（`scale_min()` ×8，约 96 条标量指令/块）换成
一条 16B 载入 + pshufb/掩码移位，并按 8 块 hoist 到块循环外：

- 数值：`test_cpu_moe.exe` **全 PASS**（fp32 路径 max|Δ| = 4.77e-07 与改动前一致 ⇔ 逐位不变）；
- 性能：隔离 `layer` 模式 A/B **median 24.74 vs 24.83 GB/s（-0.4%，噪声内）**；
- L3 计算上限：25.5 vs 24.8 GB/s，也没变。
- **结论：回退。** 它只值 ~0.1 GB/s，不值得多一条 SIMD 解码路径。

### 4.5 线程绑核（SMT 成对 pack）
隔离探针里 `aff=3`（线程 2k→核 k、2k+1→核 k 的 SMT 兄弟）**median +15%**，
但在**真实 `moe_forward_mt` 层基准里回到噪声内**（24.9 vs 24.8），且 10/12 线程会因主线程
被挤掉而出现 5–7 GB/s 的灾难态。**不放。**

### 4.6 生产两级 int8 Q4_K（cpu_int8=2）
- 原实现 19.7 GB/s。查出原因：4 个 32B raw 向量在宏展开前就全部载入 + `__m256&` 引用 ⇒
  **寄存器溢出**；同时 `d*sc` 的标量乘 + `set1_ps` 每次 contrib 都重新广播。
- 修掉后 **22.7 GB/s（+15%）**，但**仍然低于 fp32 的 25.0**。
- 根因：两级 int8 需要 2 套激活（base + residual），每个 32 值子块要做
  2×(maddubs+madd+cvtps) + mul + fmadd ≈ 13 条指令，**并没有比 fp32 的 12 条 FP 少多少**；
  而单级 int8 的量化误差是 **3e-3 相对误差**（任务书要求 rel_l2 ≤ 1e-6），根本不可用。
- ⇒ **`cpu_int8_x` 档位最终没有落地**：两级 int8 更慢，单级 int8 精度不够。
  （因此"Coder top-1 是否仍 185"的实测也不需要了——没有要开的档位。）
- 现已默认开的 `cpu_int8=1`（Q6_K，只影响 down 矩阵）仍然保留：phase B 上 int8 24.8 vs
  fp32 22.9 GB/s，**+8%**，这是真实收益，本轮没有动它。

### 4.7 其他
| 尝试 | 结果 |
|---|---|
| 线程数 6 / 8 / 9 / 10 / 12 | 9 仍最优（隔离 24.8，6→20.4，12→23.2） |
| `ENGINE_HOST_COPY`（pinned→页表激活） | 无差异（§3） |
| `ENGINE_POOL_SPIN=20000` | 无差异（§3） |
| 页表大页/THP、NUMA | VM 内无控制权，未做 |

---

## 5. 目标现实性（数据结论）

任务目标：35B decode 25.7 → ≥30 tok/s（38.9 → ≤33.3 ms）。

| 假设 | matvec | decode | tok/s |
|---|---|---|---|
| 现状（隔离 0.65 / 引擎内 0.77） | 0.77 | 38.6 | 25.9 |
| **理想 A**：引擎内 matvec 追平隔离基准 | 0.65 | 35.7 | **28.0** |
| **理想 B**：matvec 达到 DRAM 地板 32.5 GB/s（0.50 ms） | 0.50 | 32.1 | **31.2** |
| **理想 B 的前提** | 计算上限必须 >32.5，实测只有 25 ⇒ **物理不可达**（不动数值精度） | | |

⇒ **在「不改 CUDA、不改数值精度」的约束下，35B 的现实上限是 ~28 tok/s（112%）。**
要到 30+ 必须从下面三块里拿（都不属于本轮授权范围）：

1. **CPU 层里的 28–40 次 launch（0.22 ms/层 = 5.3 ms/token）**：CUDA Graph 只捕获 CPU 层的
   注意力段（报告里试过整层图捕获，中性偏负；分段捕获没试过）；
2. **`stream_sync` 2 次（0.14 ms/层 = 3.4 ms/token）**：device-side routing 能省掉宿主往返
   （已知有实现但因为数值差异默认关）；
3. **16 个 GPU 层 8.5 ms（22%）**：纯 GPU 侧。

---

## 6. Coder / 9B / 0.5B

本轮**没有改动任何产品代码**，所以三个模型不存在回退（四套验证见 §7）。
Coder 的 CPU 专家是 Q5_K/Q8_0（走 `dot_row` 的通用路径，不在本轮内核实验范围），
它的 23.4 → 26 tok/s 目标同样受「phase A Q5_K 也是同一类计算受限内核」约束，
需要单独做 Q5_K 的取证（本轮未做，如实标注）。

---

## 7. 数值与验证

中间版本（SIMD scale 解码 + hoisted decode + int8 scale hoist，**已回退**）的数值证据：

```
Q4_K row dot (512x2048) vs old kernel        max|delta| = 4.768e-07  (limit 1e-06)  PASS
Q6_K row dot (2048x512) vs old kernel        max|delta| = 2.794e-08  (limit 1e-06)  PASS
moe_forward_mt (n_tokens=1) vs old path      max|delta| = 7.264e-08  (limit 1e-06)  PASS
moe_forward_mt (n_tokens=2) vs old path      max|delta| = 5.960e-08  (limit 1e-06)  PASS
int8 Q4_K row dot rel_l2 (cpu_int8=1)        max|delta| = 1.646e-07  (limit 5e-05)  PASS
int8 moe_forward_mt rel_l2 (cpu_int8=1)      max|delta| = 3.060e-05  (limit 5e-05)  PASS
CPU MOE TEST: PASS
```

最终回退后重跑（本报告提交时的实测，`build.sh` 全量重建）：

| 套件 | 结果 |
|---|---|
| `bash tools/verify.sh` | **ALL GREEN** |
| `bash tools/smoke_models.sh` | **4/4 正确：40716 / 185 / 2614 / 2614** |
| `bash tools/gen_regression.sh` | **10 passed, 0 failed — GEN REGRESSION GREEN** |
| `bash tools/verify_generality.sh` | **26 passed, 0 failed — GENERALITY GREEN** |

35B decode（回退后的二进制、GPU 空闲窗口、`test_gen "<35B>" 760 9`）：**38.59 ms/token（26 tok/s）**，
与动工前完全一致（本轮没有任何产品代码改动）。

---

## 8. 工作树状态（逐字节等价证明）

| 文件 | 动工前 md5 | 提交时 md5 |
|---|---|---|
| `src/cpu_matvec.cpp` | 18875e5b87b3f9eb23215f9ad5c383a4 | 18875e5b87b3f9eb23215f9ad5c383a4 |
| `src/runner.cpp` | b7384138038fa2c7d95d294b109bfedc | b7384138038fa2c7d95d294b109bfedc |
| `src/qwen35.cpp` | 2e3a8c3a9c8901abb6665971db4aa87e | 同左（未改） |
| `src/config.cpp` | 48a9d71a988ee0b265be3e7d4376e602 | 同左（未改） |
| `include/engine/cpu_matvec.h` | c4c2f7c04de3b9b77f38d41364386a25 | 同左（未改） |

唯一新增：**`tools/bench_rp.cpp`**（取证探针，不进 `build.sh`，不影响任何产品行为）。
备份：`build/bak/*.pre-repack`。

---

## 9. 复现命令

```bash
cd engine
# 取证探针（唯一需要构建的新工具）
python -m ziglang c++ -std=c++17 -O2 -Iinclude \
  tools/bench_rp.cpp src/config.cpp -o build/bench_rp.exe

./build/bench_rp.exe a 24 40 9 read prod        # DRAM：纯读 vs 生产内核（phase A）
./build/bench_rp.exe a 1  40 9 read prod        # L3 常驻：计算上限（决定性对照）
./build/bench_rp.exe a 1  30 9 read prod i8     # int8 vs fp32 计算上限
./build/bench_rp.exe b 24 40 9 read f32 i8      # phase B（down/Q6_K）
ENGINE_RP_AFF=3 ./build/bench_rp.exe a 24 40 10 read prod   # 绑核探针

# 生产层基准（原工具，改前/改后同状态对比）
python -m ziglang c++ -std=c++17 -O2 -Iinclude \
  tools/bench_cpu_layer.cpp src/cpu_matvec.cpp src/config.cpp -o build/bench_cpu_layer.exe
bash tools/ab_cpu.sh ./build/bench_a.exe ./build/bench_b.exe layer 24 20

# 引擎内分段（35B，需要空闲 GPU）
ENGINE_DBG_LAYERMS=1 ./build/test_gen.exe "<35B>.gguf" 760 1
ENGINE_DBG_MOE_PHASES=1 ./build/test_gen.exe "<35B>.gguf" 760 5
```
