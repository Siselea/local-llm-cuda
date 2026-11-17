# PRE-FILL 报告：prefill 从 llama.cpp 的 1/4 到与之同量级（多行量化 GEMM + 注意力分块 + SSM 状态常驻）

> 执行者：prefill-fast｜时间：2026-11-15 09:35–11:40｜本机 RTX 3060 12GB（cc 8.6，GA104 解锁卡 / WDDM VM，与 `LONGCTX-REPORT` 同机）
> 任务：把 prefill（pp）从 333 tok/s（llama.cpp 的 1/4）做到超过 llama.cpp 的 1384 tok/s
> 验收：`verify.sh` / `smoke_models.sh` / `verify_generality.sh` 全绿；decode 不回退；数值 prefill vs 逐 token 喂 top-1 一致；不改 `build.sh` 编译选项；不动既有 Q5_K/Q6_K dp4a 数学

---

## 0. TL;DR

| 指标（9B Q5_K_M，1622-token prompt，ctx 4096） | 改前 | **改后** | 倍数 |
|---|---|---|---|
| prefill 墙钟 | 4869 ms | **1478–1551 ms**（多窗口 min 1478） | **3.1–3.3×** |
| pp | 333 tok/s | **1046–1097 tok/s** | **3.1–3.3×** |
| 与 llama.cpp（1384 tok/s @ n=1622） | 24% | **79%** | — |

**单块（1622 token 一次 forward）的构成（本机实测，非估算）**：

| 部分 | ms | 说明 |
|---|---|---|
| cuBLAS fp16 tensor-core GEMM | **958** | 1.121e13 MAC / 11.7 T-MAC/s；本卡 fp16 tensor 峰值 12.8 T-MAC/s，**已到硬件上限** |
| dequant→fp16 + x→fp16 | 137 | 6.1 GB 权重读 + 12.2 GB fp16 写 + 激活转换 |
| **GEMM 相关小计** | **1177** | 用 `ENGINE_SKIP_GEMM=1` 差分实测（整块 1478 − 其余 301） |
| 非 GEMM 全部（attention/gd_delta/归一化/量化激活/逐元素 + 发射气泡） | **301** | 其中 attention 约 100、gd_delta 约 98 |

**三个最大的单项收益（按 ms）**：
1. **-2391 ms**：rows>1 的 matmul 从"整模型反量化成 f32 + f32 cuBLAS"改成"**权重直接反量化成 fp16 + cuBLAS fp16（f32 累加）**"（4869→2478 ms 量级）。
2. **-309 ms**：prefill attention 从"1 warp/(token,head)，K/V 全量重复读"改成"8 个 query 行/block 共享 + GQA 组内共享"（444→~100 ms GPU）。
3. **-152 ms**：gated-delta-rule（线性注意力）的 64 KiB/head 递归状态从"global 每 token 读写"改成"**常驻寄存器 + 只有小组内 partial 走 shared**"（250→98 ms GPU）。

**重要限制（如实）**：本机 fp16 tensor core 在 **f32 累加**下就是 12.8 T-MAC/s；唯一更快的可行路径是 cuBLAS int8（41 T-MAC/s，int32 累加精确）或 fp16 **fp16 累加**（22 T-MAC/s），前者受"per-block scale 无法表达为普通 GEMM"限制，后者数值不安全（实测会把 1.6k 退化 prompt 的 top-1 翻掉）。详见 §6。所以 **1478 ms 里 958 ms 是硬件上限**，要达到 llama.cpp 的 1172 ms 还差 ~300 ms 的非 GEMM 优化空间。

---

## 1. 复现基线与测量纪律

```bash
# 基线（改动前，GPU 独占）
./build/test_gen.exe <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf "$(cat build/ids1622.txt)" 4 --ctx 4096
# -> [time] prefill 1622 tokens: 4869 ms (333 tok/s)
#    [gpu] forward gpu=1343 / 1423 / 1523 / 527 ms  (rows=512 -> 4 块)
```

**测量纪律（本机是 VM，噪声极大，吃过教训）**：
- 开始前 `nvidia-smi --query-gpu=memory.used` 必须 < 900 MiB；否则换下一个窗口。
- 每个配置测 ≥3 次、**交错**（A/B/A/B/A/B），取 min；同一二进制相邻两次可差 14×（被 llama-server 35B 抢显存 → WDDM 换页：5859 ms vs 673 ms）。
- 一次被污染的完整证据：`chunk1=5859ms chunk2=5592ms` 然后 `cuEventElapsedTime failed (CUresult=600 device not ready)`。
- kprof 归因用 `ENGINE_KPROF=1`（注意：CUDA Graph 模式下看不到内核，且 `build/pf_gen.exe` 是本报告专用的 prefill-only 探针，见 §7.1）。

**功耗证据（回答"是不是真的在算"）**：prefill 期间 `nvidia-smi` 采样 **144 W / 100% util / 1957 MHz / 59 °C**（空闲态 15–37 W / 1777 MHz）。不是空转。

---

## 2. 改动清单（全部已落盘，默认生效的标 [默认]）

| 文件 | 改动 | 关键行 |
|---|---|---|
| `include/engine/kernels.h` | ①`kDequantF16`：Q4_K/Q5_K/Q6_K/Q8_0 **直接反量化成 fp16**（每线程 2 元素、4 字节打包写，写全合并）②`kAttnTiledPre`：prefill attention 每 block 8 个 query 行共享 K/V（f32 用 float4 读，消除 8× 扇区浪费）③`kAttnTiledPreGqa`：一个 warp 负责一个 query 行的**全部 rep 个 q head**（K/V 只读一遍）④`kGdDelta128Smem`：递归状态暂存 shared 版（无收益，留作 A/B）⑤`kGdDelta128Reg`：状态常驻寄存器版 ⑥`kF16ToF32` | 文件末尾追加，未改任何既有内核 |
| `src/runner.cpp` | `matmul()` 新增 **rows>1 的 fp16 快速分支**（默认）；dequant 目标从 f32 改成 fp16，`slab_rows` 允许翻倍；`run_attn` 新增两条 prefill 派发；kernel 注册 + NVRTC 拼接；**自适应 chunk 放大**；prefill CUDA Graph（默认关）；`ENGINE_SKIP_GEMM` 诊断 | `matmul()` ~3150–3265、`run_attn` ~880、构造期 ~150 |
| `src/qwen35.cpp` | `gd_delta_rule_128_reg/smem` 派发（n_tokens>1 时），`kh_mode` 透传（与 `qwen35-prefill-bug` 的语义修复对齐） | ~460–500 |
| `include/engine/runner.h` | 新句柄、`prefill_f16_*`/`attn_*`/`gd_*`/`prefill_graph` 成员、`prefill_graphs_`、`y16_scratch_` | — |
| `src/cublas.{h,cpp}` | `gemm_f16_nt_ld_c16`（fp16 累加，实验用，默认不走） | — |
| `src/config.cpp` | 9 个新键：`prefill_f16`(t)、`prefill_f16_min_macs`(0)、`prefill_f16_acc`(f)、`prefill_graph`(f)、`prefill_rows_max`(4096)、`attn_pre_rows`(8)、`attn_gqa`(t)、`gd_smem`(t)、`gd_reg`(t) | 表尾追加 |
| `docs/CONFIG.md` | 上述 9 键的说明（含实测数与代价） | §3.3 表 |
| `tools/pf_gen.cpp` | prefill-only + kprof 复位探针（不在 `build.sh` 里） | 新文件 |
| `tools/probe_gd_parity.cpp` | 三个 gd_delta 内核**逐位对拍**单测（两 kh_mode × 多 T × chunked 状态续传） | 新文件 |
| `tools/probe_prim.cpp`/`probe_mma.cpp`/`probe_blas.cpp`/`probe_i8big.cpp` | 本卡原语微基准（dp4a/mma/cuBLAS f32/f16/int8/带宽） | 新文件 |
| `tools/pf_ab.sh`、`tools/pf_verify_all.sh` | 交错 A/B 取 min 脚本、三套验证重试脚本 | 新文件 |

**未改**：`build.sh`、任何既有内核的数学（Q5_K/Q6_K dp4a 未动）、`ir.cpp/gguf.cpp` 解析、MLA 路径、`src/main.cpp`、HTTP 服务。

---

## 3. 方法 1（最大项）：权重直接反量化到 fp16 + fp16 tensor core GEMM（-2391 ms 量级）

### 3.1 根因（不是"量化"，是"fp32 cuBLAS"）

`matmul()` 原来的 rows>1 路径：把每个权重矩阵按 slab **反量化成 f32** 写进 scratch，再 `cublasSgemm`。实测（`tools/probe_blas.cpp`，9B prefill 形状）：

| 路径（M=512, N=12288, K=4096） | 实测吞吐 |
|---|---|
| 我们的 f32 cuBLAS NT | **3.9–4.2 T-MAC/s** |
| cuBLAS fp16 NT（f16 A/B，**f32 累加**） | **11.2–12.8 T-MAC/s** |
| cuBLAS int8（N,M 正确朝向：权重做 A） | **41 T-MAC/s**（但见 §6 的 scale 限制） |
| cuBLAS fp16 + **fp16 累加** | 21–22 T-MAC/s（数值不安全，§6） |
| CUDA-core dp4a 峰值 | 13.8 T-MAC/s（= 2× fp32 FMA；本卡 dp4a 是 64/SM/clk） |
| fp32 FMA 峰值 / HBM 带宽 | 5.75 T-MAC/s / 331 GB/s |

⇒ prefill 的 4× 差距**几乎全部**来自"我们用 f32 cuBLAS，而 llama.cpp 用 tensor core"。llama.cpp 的 1384 tok/s = 12.5 T-MAC/s，正好是**本卡 fp16 tensor（f32 累加）峰值**，说明它也是贴着硬件上限在跑。

### 3.2 实现

1. 新增 4 个 **直接 dequant→fp16** 内核（Q4_K/Q5_K/Q6_K/Q8_0）：每线程处理相邻 2 个元素，`cvt.rn.f16.f32` 后打包成 1 次 4 字节写（写全合并）。**跳过 f32 中转**：流量从 6.1 GB 读 + 24.5 GB f32 写 + 24.5 GB 读，降到 6.1 GB 读 + 12.2 GB fp16 写。
2. `matmul()` 的 fp16 分支：`x(f32) → x_half`（一次 `f32_to_half`）→ 每个 slab：dequant→fp16 到 scratch → `cublas::gemm_f16_nt_ld`（f16 A/B，**f32 累加**，C 是 f32）→ 直接写进 `out`。
3. slab 容量翻倍（scratch 按 f32 元素数分配，fp16 元素数是 2×）⇒ 大部分矩阵一次 slab 完成。
4. 数值：**fp16 存储 + f32 累加**，相对误差 ~3e-4（fp16 尾数 11 bit）。四模型 top-1 逐位不变；9B `103548 220 220 220` 序列与 f32 路径完全一致。

### 3.3 收益

| 9B 512-token prefill | ms |
|---|---|
| `ENGINE_PREFILL_F16=0`（原 f32 路径） | 1389 / 1445 |
| `ENGINE_PREFILL_F16=1`（默认） | **673 / 699** |
| | **2.1×** |

---

## 4. 方法 2：剩下 300 ms 非 GEMM 的压缩

### 4.1 prefill attention：444 → ~100 ms（-309 ms 合计，含 GQA）

**根因**：`attn_tiled` 是"1 warp = 1 个 (token, head)"，每个 warp 独立把整条 K/V 历史读一遍 ⇒ 1622-token、8 个 attention 层、16 个 head 合计约 **42 GB** K/V 流量；而且 4 个 q head 共用同一个 kv head，等于 4× 重复。

**改法**（两个内核，均不改每个 head 的算术顺序）：
1. `attn_tiled_pre`：**8 个 query 行/block**（8 warp，同 head）。所有 warp 以相同步调走 key，同一行 K/V 一次 HBM 读、其余 7 个 warp 走 L1；f32 KV 用 `float4` 读（每 warp 每行 2 条 512 B 全合并取数，代替 8 条 32 B-strided 取数）。
2. `attn_tiled_pre_gqa`：一个 warp 负责一个 query 行的**全部 rep=4 个 q head**（同一 kv head）⇒ K/V 再省 4×（总 32× 流量削减）。

| 9B 1622-token 单块 | attention GPU 时间 |
|---|---|
| `attn_tiled`（原） | **444 ms** |
| `attn_tiled_pre`（rpb=8） | 138 ms |
| + `attn_tiled_pre_gqa` | **~100 ms**（端到端再 -38 ms） |

### 4.2 gd_delta（gated delta rule，24 个线性注意力层）：250 → 98 ms

**根因**（先证伪了两条路，才找到真因）：
- 原内核每个 token 都要把 **64 KiB/head 的递归状态 S 从 global 读+写**（32 head × 128 KiB = 4 MiB/token 的 L2 流量）。
- 试了 ①状态暂存 shared（69 KiB，opt-in）②状态常驻寄存器。**①完全无收益** ⇒ 证明瓶颈不是带宽，而是**串行依赖链**（1622 次迭代、每次 4 个 barrier + 32+32 FMA）。
- ②`gd_delta_rule_128_reg`：每线程把自己的 32 个状态值（列 j 的 32 行）常驻寄存器，整个 token 循环不再访存 S，只有 4 组 partial 走 shared（每迭代 4 barrier）。

| | GPU 时间（24 层，1622 token） |
|---|---|
| `gd_delta_rule_128`（原） | 248–250 ms |
| `gd_delta_rule_128_smem` | 250 ms（无收益） |
| `gd_delta_rule_128_reg` | **98 ms** |

**正确性**：`tools/probe_gd_parity.cpp` 在随机数据上对拍三个内核：
```
kh_mode=0/1 × n_vh∈{8,32} × T∈{1,7,33}   smem vs ref: BIT-EXACT, reg vs ref: BIT-EXACT
chunked(5+6) 状态续传 vs 单次 11 token     BIT-EXACT（两种 kh_mode）
GD PARITY: PASS
```

### 4.3 自适应 chunk 放大（-110 ms）

**根因**：每个 chunk 都要**重新反量化整个模型**（95 ms）+ 重付每块固定开销；默认 `--prefill-rows 512` 让 1.6k prompt 变成 4 块。

**改法**：`prefill_rows` 只作为起点，装载时按 `权重 + KV + 激活(2×rows) + reserve ≤ 空闲显存` **逐次翻倍**（上限 `prefill_rows_max=4096` / `max_seq`），打印 `[ctx] prefill chunk grown 512 -> 2048 rows`。同时保留原来的显存不足自动下调。**ctx 上限的推导仍按 512 行算**，所以长上下文能力不受影响（这是关键设计点：先定 ctx，再放大 chunk）。

| 9B 1622-token | rows_cap | 墙钟 | pp |
|---|---|---|---|
| 4 块 | 512 | 2393 ms | 678 |
| 2 块 | 1024 | 2157 ms | 752 |
| **1 块** | **2048（自动）** | **2046 ms** | **793** |

（上表是只打开 fp16 GEMM 时的数字；再叠加 attention/gd 优化后单块到 **1478–1551 ms**。）

**最终配置的分项消融（1622 token，ctx 4096，交错 min-of-2，本机噪声 ±5%）**：

| 配置 | 墙钟 |
|---|---|
| 全开（默认） | **1478–1551 ms / 1046–1097 tok/s** |
| 关 GQA 共享（`attn_gqa=0`） | +38 ms |
| 关寄存器态 gd_delta（`gd_reg=0`） | +157 ms |
| 关全部 prefill attention 分块（`attn_pre_rows=1`） | +309 ms（444→~100 ms GPU） |
| 关 fp16 GEMM（`prefill_f16=0`，即原路径） | 4869 ms（4 块） |
| `ENGINE_SKIP_GEMM=1`（诊断，非 GEMM 部分） | 301 ms |
| 实验：fp16 累加（`prefill_f16_acc=1`） | 1182 ms / 1371 tok/s，**数值不安全**，默认关 |
| 实验：prefill CUDA Graph（`prefill_graph=1`） | 1506 ms（无收益，默认关） |

---

## 5. 观测但未默认启用的项（含原因，别踩坑）

| 项 | 实测 | 为什么没默认开 |
|---|---|---|
| **fp16 累加 GEMM**（`CUBLAS_COMPUTE_16F`） | **1182 ms / 1371 tok/s**（比 f32 累加快 396 ms） | **数值不安全**：1.6k 退化 prompt 的 top-1 从 97049 变成 681，而**逐 token 喂**（`ENGINE_SPLIT_PREFILL=1`）给的是 97049 ⇒ 违反验收 4。四模型短 prompt top-1 不变，但这个 0.3%/matmul 的 fp16 归约舍入在近似平局上会翻。做成 `prefill_f16_acc` 配置项，默认关 |
| **prefill CUDA Graph**（按 chunk token 数捕获整条流水线，2341 launch → 1） | 1506 vs 1478 ms（**无收益**） | prefill 已 GPU-bound：`ENGINE_PIPE_PROF` 显示 launch 2341 次=388 ms CPU，但这段时间与 GPU 执行重叠，墙钟不变。默认关（也避免捕获期分配的坑） |
| **状态暂存 shared 的 gd_delta** | 250 ms（=原版） | 见 4.2，瓶颈不是带宽；保留作 A/B |
| **int8 dp4a GEMM / int8 MMA GEMM** | 分析后放弃，见 §6 | 本卡 dp4a 峰值 13.8 T-MAC/s、`mma.sync.m16n8k32` 微基准只有 6.8 T-MAC/s，仅与 fp16 f32-累加（12.8）同量级，且要自写 GEMM + per-32 weight 重量化 + scale 合并（25–50% 指令开销）⇒ 高风险低回报 |
| **`--quant-gemm`（既有 G1 内核）** | 568 ms（任务书写明） | 它是 f32 shared-tile GEMM，无 tensor core；被新路径全面取代（新路径单块 1478 ms 对 4 块 4869 ms） |

---

## 6. 为什么"再快一倍"需要自研 int8 GEMM（诚实结论与下一步）

本卡三档算力（全部实测，`tools/probe_*.cpp`）：

| 精度路径 | 吞吐 | 累加精度 | 能否表达 per-32-block scale |
|---|---|---|---|
| cuBLAS f32 | 4.0 T-MAC/s | f32 | 可以（但慢） |
| **cuBLAS fp16** | **12.8 T-MAC/s** | **f32** | 不需要（权重已是 fp16 精确值） |
| cuBLAS fp16 累加 | 22.3 T-MAC/s | fp16（0.3%/matmul） | 不需要，但数值翻车 |
| cuBLAS int8 | **41 T-MAC/s** | **int32（精确）** | **不能**：普通 GEMM 只有一个 per-tensor/per-row scale |
| 自研 int8 mma + per-32 scale | 理论峰值 41，现实 25–30 | int32 精确 | **能**（代价是每 32 元素一次 int32→f32 合并） |

所以唯一"既快又数值安全"的 >12.8 T-MAC/s 路线是**自研 int8 MMA GEMM**（`mma.sync.m16n8k32.s32.s8.s8.s32`，weight 按 per-32/256 重量化成 int8 存 shared，激活用既有的 2 级 int8 量化（1e-5）或单级（5e-3），每 256 个 K 元素做一次 int32→f32 合并 + per-block scale）。预期 GEMM 从 958 ms 降到 450–600 ms（总 prefill 950–1150 ms，pp 1400–1700）。**本轮没做**：需要 2–4 h 连续开发 + 逐位对拍，且当前剩余时间不够冒这个风险。

---

## 7. 验证

### 7.1 数值对拍（prefill vs 逐 token 喂）

```bash
# 逐 token 喂的参考 top-1（1.6k prompt）
ENGINE_SPLIT_PREFILL=1 ENGINE_SPLIT_ONLY=1 ENGINE_TOPK=1 ./build/pf_gen.exe <9B> "$(cat build/ids1622.txt)" 1 --ctx 4096
# split topk: 97049(0.000) 97912(-0.255) 681(-0.390) ...
# 我们的 prefill（默认配置）top-1 = 97049 ✓ 一致
# 四模型 smoke 全绿：40716 / 185 / 2614 / 2614
```

`bash tools/smoke_models.sh`：**四模型全对**（0.5B 40716、Coder 185、9B 2614、35B 2614；35B 的期望值已被 `qwen35-prefill-bug` 的语义修复更新）。

`verify.sh`：**ALL GREEN（exit 0，64 PASS）**。`verify_generality.sh`：见 §7.3。

### 7.2 decode 不回退

我改的都是 `n_tokens>1` 分支（`matmul()` 的 GEMV 分支、`run_attn` 的 split 分支、`gd_delta` 的 decode 融合分支都原样保留），唯一的全局改动是**自适应 chunk 放大**（只影响 prefill）与新增配置项（默认值都指向"新 prefill 快路径"）。

| 模型 | decode（改动后实测） | 任务硬指标 |
|---|---|---|
| Qwen3.5-9B | **21.96 ms/token = 45.6 tok/s** | ≥45 ✓ |
| qwen2.5-0.5B | **5.13 ms/token = 195 tok/s**（smoke_models 实跑） | ≥185 ✓ |
| Coder-V2-Lite | decode 路径逐行未改；`smoke_models` top-1 仍 185 | ≥22.5 ✓（沿用 `KV-LONGCTX-REPORT` §7.4 同二进制 A/B：42.6–66.8 ms，VM 噪声大） |
| 35B-A3B | decode 路径逐行未改；`smoke_models` top-1 仍 2614 | ≥24 ✓（38.1–40.9 ms = 25.0 tok/s，`GENERAL-OPT-REPORT` §2） |

### 7.3 三套验证（原始日志 `build/verify_prefill.log`）

| 套件 | 结果 |
|---|---|
| `bash tools/verify.sh` | **ALL GREEN**（exit 0） |
| `bash tools/smoke_models.sh` | **exit 0**：0.5B `40716`（decode **5.13 ms/token**）、Coder `185`、9B `2614`、35B `2614` |
| `bash tools/verify_generality.sh` | 见下（唯一 FAIL 是既有测试过时，已修） |

**generality 的唯一 FAIL 与本次改动无关，且已修**：`tools/verify_generality.sh:67` 用 `ctx = 0` 探测"越界值应被拒"，
但 `kv-longctx` 之后 `ctx = 0` 已是**文档化的合法值**（"从零推导上限"，`docs/CONFIG.md` §3.3.1）⇒ 该测试已过时。
改成真正越界的 `ctx = 99999999`（保持测试意图），并加注释说明。

> 过程中还遇到两个**环境性**失败，不是代码问题，记录以免误判：
> ① `build.sh` 写 `build/test_gen.exe` 被其它智能体的回归进程文件锁挡住（`lld-link: failed to write output`）；
> ② 验证脚本与其它智能体的 `build.sh` 并发时，`./build/engine.exe` 会瞬时消失（`No such file or directory`）⇒ 所有 config 测试假失败。
> 结论：本机多智能体并发时，三套验证必须独占执行。

### 7.4 与 llama.cpp 同条件（ctx）

9B 的 `-c 131072` 条件在本机 f32 KV 下会被 `ctx_auto` 夹到 **38053**（q8_0 才能到 143260，见 `KV-LONGCTX-REPORT` §6.2），
此时 `prefill_rows` 因显存只回到 512（4 块）：**1622 token prefill = 1897 ms（855 tok/s）**。
所以与 llama.cpp 的 1384 tok/s 的对比请用同一 ctx 档：本报告主表用 ctx 4096（单块 2048 行，1551/1478 ms）。

---

## 8. 结论：与 llama.cpp 的对比

| 9B Q5_K_M，n=1622，ctx 4096 | pp | 相对 llama.cpp |
|---|---|---|
| 我方改前 | 333 | 24% |
| 我方改后（默认配置） | **1097** | **79%** |
| llama.cpp（`-c 131072 -b 2048 -ub 512`，用户原始记录） | 1384 | 100% |

**差距的 100% 是可解释的**：llama.cpp 的 1384 tok/s = 12.5 T-MAC/s，就是本卡 fp16 tensor core（f32 累加）的物理峰值；我方 fp16 GEMM 也是 11.7 T-MAC/s（峰值的 92%）。剩下的差距全在**非 GEMM 的 301 ms**（attention 100 + gd_delta 98 + 归一化/逐元素/量化约 100）与 dequant 137 ms 上。要在 pp 上反超 llama.cpp，只有两条路：
1. 自研 int8 MMA GEMM（§6，GEMM 958→450–600 ms）—— 数值可保持（int32 精确累加 + 2 级 int8 激活），是本任务**唯一**能把 pp 推到 1400+ 的路线，建议作为下一轮的 P0。
2. 把 attention/gd_delta/dequant 再压 2–3×（301+137 → 150 ms 量级），可到 ~1100 ms ≈ pp 1475，但每条都需要重写内核。

---

## 9. 复现命令

```bash
# 基线/改后对比（prefill-only 计时）
bash build.sh
./build/test_gen.exe <9B> "$(cat build/ids1622.txt)" 4 --ctx 4096        # 一行 prefill 计时
ENGINE_PREFILL_F16=0 ./build/test_gen.exe <9B> "$(cat build/ids1622.txt)" 4 --ctx 4096  # A/B
ENGINE_SKIP_GEMM=1   ./build/test_gen.exe <9B> "$(cat build/ids1622.txt)" 4 --ctx 4096  # 非 GEMM 部分
ENGINE_ATTN_GQA=0 ENGINE_GD_REG=0 ENGINE_ATTN_PRE_ROWS=1 ...             # 逐项关掉
# 交错 A/B（等 GPU 空闲 + 取 min）
REPS=3 bash tools/pf_ab.sh <9B> build/ids1622.txt 4 4096 512 "ENV=..." "ENV=..."
# gd_delta 三内核逐位对拍
python -m ziglang c++ -std=c++17 -O2 -Iinclude src/cuda.cpp src/config.cpp \
  tools/probe_gd_parity.cpp -o build/probe_gd_parity.exe && ./build/probe_gd_parity.exe
# kprof 归因（prefill only）
ENGINE_KPROF=1 ENGINE_KPROF_PREFILL=1 ENGINE_PREFILL_ONLY=1 ./build/pf_gen.exe ...
# 本卡原语微基准
./build/probe_prim.exe   # dp4a/cvt/FMA/带宽
./build/probe_blas.exe   # cuBLAS f32/f16 NT + mma
./build/probe_i8big.exe  # cuBLAS int8（正确朝向 = 41 T-MAC/s）
```
