# GENERAL-OPT 报告（2026-11-15，长上下文 + CPU 量化 GEMV + 去写死审计）

负责：general-opt · 任务书：A（35B CPU 量化 GEMV）> C（长上下文）> B（9B GEMV）> 去写死审计
产物：`LONGCTX-REPORT.md`、`docs/CPU-INT8.md`、`DEADCODE-AUDIT.md`、本文件

---

## 0. 一句话结论

1. **C 拿到了一个硬 bug 并修掉（默认生效）**：任何 **>384 token** 的上下文（预填或 decode）
   在 9B/35B 上都会 `cuLaunchKernel failed (invalid argument)` 直接崩。修完后 2048 token 预填 +
   4096 ctx 正常跑，贪心序列稳定。这是本轮唯一"到达用户价值"的改动。
2. **A 做了实现、拿到部分数据、但没达到 110%**：int8 host 行内核在 **Q6_K phase B 上快 6%**
   （0.346→0.326 ms/层，≈ −0.5 ms/token），端到端 A/B 被虚拟机噪声吃掉（同一对二进制先后测出
   +10.6% 和 −9.6% 两个相反结论）⇒ 按用户硬约束**默认关闭**，做成 `cpu_int8` 配置项。
3. **B（9B GEMV）没做**；**去写死审计完成**（`DEADCODE-AUDIT.md`：20 处常量逐条定性，本轮改掉 6 处，
   仍有 5 处写死并给了下一轮方案）。
4. **要求 1（每项 ≥110%）没有达成**：本轮测到的绝对水平是 9B 22.2–22.5 ms（44–45 tok/s，
   llama.cpp 的 95%）、35B 40.3 ms（24.8 tok/s，96%）——**注意这两个数字来自本机一个"较快窗口"
   （早前的报告口径是 21.3 / 37.98）**，虚拟机漂移 ±20%，绝对值不能当结论，只能用同窗口交错 A/B。

---

## 1. 改动清单（全部已落盘）

| 文件 | 改动 | 默认行为 | 状态 |
|---|---|---|---|
| `include/engine/cuda.h`、`src/cuda.cpp` | 新增 `cuda::max_dynamic_shared_mem()`：绑定 `cuFuncSetAttribute` + `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`，按 kernel 缓存；launch **失败路径**才查内核名并打印 `kernel/grid/block/shared`（热路径 0 成本） | 生效 | ✅ |
| `src/qwen35.cpp`、`src/runner.cpp` | attention launch 的 smem 预算：opt-in 设备上限 + `while (nwarp>1 && smem > cap) nwarp >>= 1`（qwen35 与 dense/MLA 两条路径都改） | 生效 | ✅（**长上下文崩溃修复**） |
| `src/cpu_matvec.cpp` | 两级 int8 激活量化 `XQ8`/`xq_build` + `avx2_dot_q4k_i8` / `avx2_dot_q6k_i8` + 按量化类型/宽度分派；`cpu_pool_spin` 改读配置 | 默认 **不启用** int8 | ✅ 数值 PASS，性能不足以默认开 |
| `src/config.cpp` | 追加 4 键（不重排）：`cpu_int8`(0)、`tiny_sk_max_nout`(128)、`tiny_sk_min_nin`(512)、`cpu_pool_spin`(512) | — | ✅ |
| `include/engine/kernel_select.h`、`src/kernel_select.cpp` | tiny-n_out split-K 的两个 constexpr 阈值变成 `GemvParams` 字段 + 配置（默认值不变，行为逐位不变） | 同前 | ✅ |
| `include/engine/runner.h`、`src/runner.cpp` | 上述两个新配置的成员与快照 | 同前 | ✅ |
| `tests/test_cpu_moe.cpp` | 新增 int8 数值段（Q4_K 2.19e-6 / Q6_K 1.06e-5 / 整层 3.06e-5，限 5e-5）；原 fp32 位精确性段固定 `cpu_int8=0` | — | ✅ 单测 PASS |
| `docs/CONFIG.md` | 追加 4 行（`cpu_int8` 指向 `docs/CPU-INT8.md`） | — | ✅ |
| `docs/CPU-INT8.md`、`LONGCTX-REPORT.md`、`DEADCODE-AUDIT.md` | 三份报告 | — | ✅ |
| `tools/longctx.sh`、`ab35b.sh`、`ab_cpu.sh`、`dbg_i8*.cpp` + `tools/README.md` 追加 | 诊断工具 | — | ✅ |

没有改 `build.sh`；没有按模型名特判（`DEADCODE-AUDIT.md` §4/§5）；没有碰 `src/main.cpp`/`serve*`。

## 2. C：长上下文（详见 `LONGCTX-REPORT.md`）

- **根因**：`attn_cached_w` 的 `sh = (max_seq + 32 + nwarp*head_dim)*4`；9B `head_dim=256`，
  `ctx_len>384 ⇒ nwarp=32 ⇒ 49280 B > 48 KiB` 默认上限。
- **修复后 9B（--ctx 4096）**：短上下文 decode 22.2–22.5 ms → 384/512 token 23.2–24.4 ms
  → 1k 27.3 ms → **2k 33.3 ms（−50%）**；prefill 384 token 1350 ms（284 tok/s）→ 2k 9882 ms（207 tok/s）。
- **正确性**：384 token 一次性 prefill vs 逐 token prefill：top-1 都是 `100465`，top-5 顺序一致，
  首生成 token 一致；2k 预填生成 `271 155648` 稳定；短上下文 9B 预测仍 `2614`。
- **瓶颈**（与 kv-longctx 的定位一致）：attention 内核 K/V 跨 lane stride 导致 sector 放大、
  GQA 4× 重复读、smem O(max_seq)、**KV cache 是 f32**（`runner.h:98`，kernel 形参 `const float*`）。
  预期 KV q8_0 能把 2k ctx 的 attention 访存减半。

## 3. A：CPU 量化 GEMV（详见 `docs/CPU-INT8.md`）

| 指标 | fp32 | int8 | Δ |
|---|---|---|---|
| 孤立行内核 Q4_K（L1 热） | 184–200 ns | 208–242 ns | **慢 12–21%** |
| 孤立行内核 Q6_K（L1 热） | 288–300 ns | **229 ns** | **快 1.31×** |
| 引擎内 phase B（down，Q6_K，24 层中位） | 0.346 ms/层 | **0.326 ms/层** | **−6%** |
| 引擎内 phase A（gate+up，Q4_K） | 0.491 ms/层 | 0.503 ms/层 | 噪声内（同一代码路径） |
| 端到端 35B（交错 A/B 5 次中位） | 40.30 ms | 44.56 ms | **不可信**（后测反向：48.50 vs 43.83） |

**结论**：Q6_K 的 int8 是真实的小赢（≈ −0.5 ms/token），但 35B 的 CPU 路径在 9 线程下**不是纯
指令受限**（19.4 GB/s vs 纯读 29.7 GB/s，且相位实测显示降指令只换来 6%），所以总收益 << 任务书
预期的 −7 ms。**默认关闭**（`cpu_int8=0`），`=1` 可 A/B。

## 4. B（9B GEMV）：没做

原因：整轮时间被"长上下文崩溃"（无预告、必须修）和测量窗口协调吃掉；9B 的 GEMV 已经 289 GB/s
（孤立 295–319、纯读 330），要继续必须重写 Q5_K/Q6_K dp4a 的 load 调度（寄存器分块/双缓冲/
`ld.global.nc`），属于需要独占 GPU 的深水区，本轮没有窗口。该项留给下一轮，建议排
C（KV 量化 + 分块 attention）之后。

## 5. 负结果（如实）

1. **端到端 A/B 在本机不可用**：同一对二进制交错 5 次得到 +10.6%，随后单独重测得到 −9.6%；
   另一智能体同期在编译/跑模型（整机 CPU 60%+、GPU 6 GB 被占）。**CPU 侧 10–20% 量级的改动
   在本机必须靠"孤立内核 + 相位级"数据判定**，端到端只能当回退判据。
2. Q4_K int8 行内核**比 fp32 慢**（花了半天才接受：Q4_K 的 fp32 本来就只要 0.875 指令/元素）。
3. 加深预取（当前行 → 前 1 行 T0 / 前 2 行 T1）：孤立 matvec min 从 13.2→19.0 GB/s 但同代码重跑
   6.3–23.9 GB/s，完全不可分辨 ⇒ **回退**（保留原来的本行 T0 + 下一行 T1）。
4. **`ENGINE_KPROF` 在 graph 模式下看不到内核**（`cuGraphLaunch` 整图重放，报 `0 kernel kinds`）：
   长上下文归因必须用 `--no-graph` + kprof，或者用 `--layers` 差分。已写进 `LONGCTX-REPORT.md` §5。
5. 修 bug 过程中发现的**错误归因链**：先只改了 `src/runner.cpp` 的 launch（dense 路径），
   9B 仍崩；9B 是 qwen35 架构，走 `src/qwen35.cpp` 的同款 launch。教训：同一内核可能有多个
   launch 站点，改 attention 要 grep `k_attnw_` 全部。

## 6. 需要 lead 在 kv-longctx 收工后统一验证的清单

> ⚠️ 现在工作树**编译不过**（kv-longctx 正在改 `runner.cpp`/`runner.h`：`kv_override_ctx_` 未声明）。
> 我最后一次成功编译是 08:26，之后只在 `src/cpu_matvec.cpp`/`config.cpp`/`kernel_select.*`/
> `tests/test_cpu_moe.cpp` 上有改动（这些文件不冲突）。

| # | 项 | 命令 | 期望 |
|---|---|---|---|
| 1 | 全量构建 | `bash build.sh` | 17 个 exe 全部 built（含 `test_gen`） |
| 2 | CPU 单测（含我的 int8 段） | `./build/test_cpu_moe.exe` | `CPU MOE TEST: PASS`（13 项） |
| 3 | 三套验证 | `bash tools/verify.sh && bash tools/smoke_models.sh && bash tools/verify_generality.sh` | ALL GREEN / 4 模型 40716·185·2614·69014 / GENERALITY GREEN 26/26 |
| 4 | 四模型最终表 | `REPS=3 bash tools/final_table.sh` | 记下 35B/9B/Coder/0.5B 中位与实际 tok/s（注意 final_table 的 pre 二进制是 06:29 快照） |
| 5 | 长上下文回归 | `bash tools/longctx.sh` | 384/512/1024/2048 全部不崩；decode 应随 kv-longctx 的 KV 量化下降 |
| 6 | 默认值不回归 | `ENGINE_CPU_INT8=1` 与默认各跑一次 35B n=9 | topk 都是 69014；默认路径应与改动前逐位一致（int8 关闭时 fp32 内核逐位不变） |
| 7 | 新增配置生效 | `./build/test_gen.exe --help \| grep -E "cpu-int8\|tiny-sk\|cpu-pool-spin"` | 三行都在 |
| 8 | kernel_select 默认不变 | 对照 `build/test_gen_pre_opt.exe` 的 `--print-config` | 除新增键外无差异 |

如果出现回归，优先怀疑我的两块：**(a) attention smem/nwarp 逻辑**（改 qwen35.cpp/runner.cpp，
kv-longctx 也在动这两处，merge 时要看 `max_dynamic_shared_mem` 是否还在）、**(b) kernel_select
的两个阈值字段**（默认值 128/512 必须与原 constexpr 一致）。

## 7. 回退方式（没有 git）

- 长上下文修复：把 `src/qwen35.cpp` / `src/runner.cpp` 里 attention launch 恢复成
  `const int nwarp = ...; const size_t sh = ...; cuda::launch(...)` 三行版本即可（`cuda.cpp` 的新
  函数留着无害）。
- int8：默认已关，不需要回退；删代码时注意 `tests/test_cpu_moe.cpp` 的 int8 段要一起去掉。
- 阈值字段：#6 表里把 `p.tiny_sk_max_nout/min_nin` 换回 `kTinySkMaxNout/kTinySkMinNin` 即可。

## 8. 复现命令（本轮全部数据）

```bash
# 长上下文
bash tools/longctx.sh
./build/test_gen.exe <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf "$(cat build/ids2048.txt)" 17 --ctx 4096
# 长上下文 bug 复现（回退到修复前）：n=512 预填会报 cuLaunchKernel(attn_cached_w ... shared 49280)
# int8 数值 / ALU / 相位
./build/test_cpu_moe.exe
./build/dbg_i8_alu.exe
ENGINE_CPU_INT8=1 ENGINE_DBG_MOE_PHASES=1 ./build/test_gen.exe <35B> "760" 2
# 35B A/B（需要安静窗口）
M=<35B> N=17 REPS=5 bash tools/ab35b.sh
```

---

# 第二轮（2026-11-15 09:35–10:20）：int8 默认决策 + 去写死清零 + 回归验证

## R2.1 CPU int8 的最终决策：**默认开启 `cpu_int8=1`**

按 lead 的优先级（**单层 host matvec > 孤立带宽 > 端到端**）做判定。隔离基准用
`tools/bench_cpu_layer.cpp` 的 `layer` 模式（24 个副本轮转 ⇒ 真 DRAM 流，计算与引擎内
`cpu::moe_forward_mt` 完全相同：hidden=2048 / inter=512 / top-8 / gate+up Q4_K / down Q6_K），
同二进制、9 线程、9 次交错配对：

| `cpu_int8` | min ms/层 | 中位 ms/层 | 配对中位（相对 off） |
|---|---|---|---|
| 0（fp32） | 0.762 | 0.811 | — |
| **1（Q6_K int8，新默认）** | **0.714** | **0.734** | **−10.1%**（min −30.3%，max +1.0%） |
| 2（Q4_K 也 int8） | 0.808 | 0.848 | +4.6%（更慢） |

第二次独立复测（默认 vs off，9 次配对）：**−7.47%**（min −11.0%，max +7.6%）。
次要证据：引擎内 `ENGINE_DBG_MOE_PHASES` phase B（down，Q6_K）0.346 → 0.326 ms/层（−6%）；
默认开启后 `--print-config` 显示 `cpu_int8 = 1`；`test_cpu_moe.exe` 13/13 PASS（rel_l2 ≤3.06e-5）；
35B top-1 `69014` 在 `ENGINE_CPU_INT8=1` 下复现；**Coder 的 CPU 专家是 Q5_K/Q5_1（没有 int8 内核）
⇒ 行为逐位不变**，0.5B/9B 是 dense 无 CPU 专家层 ⇒ 不受影响。

**开启条件（写进 `docs/CPU-INT8.md`）**：只有当 CPU 侧"指令受限"且该量化类型的 fp32 内核指令重时才划算。
本机实测：纯读天花板 29.7 GB/s，量化行内核 21 GB/s（不是 DRAM 受限）；Q6_K 的 fp32 内核 1.4 指令/元素
⇒ int8 赢；Q4_K 只要 0.875 指令/元素 ⇒ int8 输。GPU 侧同理（prefill-fast 实测 fp16 cuBLAS 12.8
T-MAC/s vs int8 12.0 vs f32 4.0 ⇒ GPU 的 int8 不是免费提速）。

## R2.2 去写死：第一轮"仍写死 5 处"全部处理（详见 `DEADCODE-AUDIT.md`）

新改掉 6 处：`kQ6kDp4aAutoSplitMaxNout`、`n_out<=256` → 配置键；attention warp 边界 96/384
（launch 与 CUDA-graph 桶）→ **单一来源函数 + 配置**（这同时消除了 capture/replay blockDim 不一致的隐患）；
CPU 线程池线程数 → **按物理核推导**（本机 6 核/12 逻辑 ⇒ 9 线程，与原经验值一致，已用
`tools/dbg_topo.cpp` 验证）；`rmsnorm4` 线程数 → 由宽度推导；`cpu_int8` 分派 → 实测选型。
外部已解决：KV dtype（kv-longctx 的 `kv_type`）、O(max_seq) smem（分块 attention）、prefill 精度
（prefill-fast 的 `prefill_f16`）。剩余 3 处是"实测默认值"（warp 边界、pool spin、Q6_K split 拐点），
物理上无解析推导，已全部配置化、可重标。

## R2.3 长上下文回归（`tools/longctx.sh`，9B，当前构建）

| prompt | prefill | decode |
|---|---|---|
| 短（1 token） | — | **22.1 / 22.5 / 22.6 ms**（44–45 tok/s） |
| 2048 token，`--ctx 4096` | **3226 / 9288 / 3140 ms（635 / 221 / 652 tok/s）** | **23.20 / 23.54 / 23.66 ms（42–43 tok/s）** |
| 2048 token，`--ctx 8192` | 3022 ms（678 tok/s） | 23.01 ms |

- **不再崩溃**（384/512/1024/2048 全部跑完；`--ctx 8192` 也正常）。
- **2k ctx 的 decode 只比短上下文慢 ~5%**（第一轮修复崩溃后是 33.3–40.3 ms / −50%；kv-longctx 的
  分块 attention + KV 量化把它救回来了）。
- prefill 也从 9882–11109 ms（207 tok/s）降到 3140–9288 ms（221–652 tok/s，视是否命中 fp16 路径/机器状态）。
- 正确性：短 prompt top-1 `2614` 不变；2k prompt top-1 `271` 稳定（第 3/4 名的 220/457 顺序在噪声内互换，
  属 split-K/图重放的求和顺序差异）；生成序列一致（271 连续）。

## R2.4 我验证过的清单（勾）

- [x] `./build/test_cpu_moe.exe` → **13/13 PASS**（含 int8 4 项，rel_l2 2.19e-6 / 1.06e-5 / 3.06e-5）
- [x] `tools/longctx.sh`（384/512/1024/2048 + ctx 8192）→ **全部不崩**，数据见 R2.3
      ⚠️ 该次测量与另一智能体的 GPU 任务重叠（prefill 有 3226/9288/3140 ms 三档离散），
      **decode 一栏（23.2–23.7 ms）可信度较高**（同批短上下文 22.1–22.6 一致），prefill 绝对值待安静窗口复核
- [x] `--print-config`：`cpu_int8=1`、`tiny_sk_*`、`cpu_pool_spin`、`attn_warp_ctx_*`、
      `q6k_dp4a_split_max_nout`、`small_nout_split_max` 全部可见且为预期默认值
- [x] 物理核推导：`physical_cores=6 / hw=12 → 9 线程`（`dbg_topo.exe`）
- [x] `kernel_select.cpp` 单独编译通过；`test_gen_gopt.exe` 全量编译通过
- [ ] 35B 端到端 int8 A/B（默认 vs off，同二进制交错，`BIN=./build/test_gen_gopt.exe N=11 REPS=5 bash tools/ab_int8.sh`）
      → **被 lead 叫停**（GPU 让给 prefill-fast 40 分钟），等窗口。默认开启的决定按 lead 规定的
      优先级用"单层 host matvec"判据做出（−10.1% / −7.47% 两次配对），端到端只作不回退的确认
- [ ] 官方 `build.sh` 全量 + `verify.sh`/`smoke_models.sh`/`verify_generality.sh` → 归 lead 统一验收
      （我改了 `cpu_matvec.cpp`/`config.cpp` 默认值/`runner.cpp` 的派生逻辑，需要一起跑）

## R2.5 文件归属与冲突披露（重要）

lead 已把 `src/runner.cpp`、`src/config.cpp` 的写入权交给 prefill-fast。我**已停手**，并在本轮
09:36–09:52 落盘了以下改动（都已在文件里，prefill-fast 的 09:55 编辑后仍然存在）：
- `runner.cpp`：`attn_warps_for_ctx`/`attn_warp_bucket_for_ctx`（新函数，~line 23-40）、nwarp 选择改用它、
  rmsnorm 线程数推导、`gemv_params()` +2 行、`load_config_snapshot` +2 行
- `config.cpp`：追加 8 个配置键（`cpu_int8` 默认 1、`tiny_sk_max_nout/min_nin`、`cpu_pool_spin`、
  `attn_warp_ctx_mid/hi`、`q6k_dp4a_split_max_nout`、`small_nout_split_max`），**未重排**
- `runner.h`：2 个 static 声明 + 4 个成员
- `qwen35.cpp`：`layer_graph_bucket()` 改调 `attn_warp_bucket_for_ctx()`
- 我后续只碰：`cpu_matvec.cpp`、`kernel_select.*`、`tests/test_cpu_moe.cpp`、`docs/*`、`tools/*`、`*REPORT*.md`

## R2.6 第二阶段新增/更新的产物

`docs/CPU-INT8.md`（默认开启的判据表）、`DEADCODE-AUDIT.md`（第二轮，含"为什么有些默认值物理上
不能解析推导"）、`GENERAL-OPT-REPORT.md`（本节）、`docs/CONFIG.md`（8 个键）、
`tools/ab_int8.sh`、`tools/ab_int8_cpu.sh`、`tools/dbg_topo.cpp`、`tools/longctx.sh`（支持 `BIN=`）。
