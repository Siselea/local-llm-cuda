# 去写死审计（DEADCODE AUDIT）· 第二轮：把"仍写死"清零

日期：2026-11-15（第一轮 08:28，第二轮 10:00）· 审计人：general-opt
范围：`src/` + `include/engine/`（KV 量化/分块 attention 是 kv-longctx 的区域，prefill fp16 是 prefill-fast 的区域，这里只记录它们的结论）

**用户要求**：内核选择按 `(量化类型, n_out, n_in)` 从实测标定选；显存按实测 `cuMemGetInfo` + 每层真实字节推导；
图捕获/路由按结构判断；**不要写死的调用数据**。

**第二轮结论**：全仓库**没有按模型名特判**（检查命令见 §5）；
第一轮标出的"仍写死 5 处"已**全部处理**：4 处改成"配置 + 实测默认值"或"结构推导"，
1 处（graph 桶数组大小）说明为什么它本来就不是模型相关常数。**剩下 3 处默认值仍然是"实测出来
的数值"而不是解析推导** —— 这是物理限制（见 §4），它们已配置化，可用 `--kernel-calibrate`
或 A/B 重标，不再是"编译进二进制的死数"。

---

## 1. 本轮新改掉的（第一轮还写死的）

| # | 原常量 | 原位置 | 新方式 | 证据 / 验证 |
|---|---|---|---|---|
| 1 | `kQ6kDp4aAutoSplitMaxNout = 4096`（Q6_K dp4a 是否自动 split-K 的 n_out 交叉点） | `kernel_select.h:91` | `GemvParams::q6k_dp4a_split_max_nout` + 配置键 `q6k_dp4a_split_max_nout`（默认 4096 = 实测）；任何具体形状仍可被 `--kernel-calibrate` 缓存覆盖 | `--print-config`：`q6k_dp4a_split_max_nout = 4096 [default]`；`kernel_select.cpp:226/228` 改用字段 |
| 2 | `n_out <= 256`（Q4_K/Q5_0 是否用 2-warp split-K） | `kernel_select.cpp:255/258` | `GemvParams::small_nout_split_max` + 配置键 `small_nout_split_max`（默认 256 = 8 行/block × 32 block 的结构值） | `--print-config` 同上；`kernel_select.cpp:255/258` 改用字段 |
| 3 | attention warp 分桶边界 `96 / 384`（**两处**：launch 与 CUDA-graph 桶） | `runner.cpp:799`、`qwen35.cpp:620` | 新增**单一来源** `DenseRunner::attn_warps_for_ctx()` / `attn_warp_bucket_for_ctx()`（`runner.h` 声明、`runner.cpp` 实现），读配置 `attn_warp_ctx_mid`(96) / `attn_warp_ctx_hi`(384)；launch 与 graph 分桶**必须同源**，否则 capture/replay 的 blockDim 不一致 | `--print-config`：两个键都显示 96/384；`qwen35.cpp` 的 `layer_graph_bucket()` 已改为调用同一函数 |
| 4 | CPU 线程池线程数 `hw - max(2, hw/4)`（"经验值"） | `cpu_matvec.cpp:933` | **按物理核推导**：新增 `physical_cores()`（Windows `GetLogicalProcessorInformation` 数 `RelationProcessorCore`），`nthreads = 3*phys/2` 并 clamp 到 `[2, hw-1]`；物理核不可得时回退原公式 | 本机实测 `physical_cores=6, hardware_concurrency=12` ⇒ 推导出 **9 线程**，与原经验值一致（行为不变）；`tools/dbg_topo.cpp`、`bench_cpu_layer` 的 `[cpu] pool threads: 9` 复核 |
| 5 | `rmsnorm4` 线程数 `n >= 2048 ? 512 : 256` | `runner.cpp:881` | 由宽度推导：目标 ~8 元素/线程，2 的幂 clamp 到 [256,512]；对 hidden=4096 得 512、n=2048 得 256、vocab=248320 得 512（与原值逐一相同） | 公式可复算；引擎内是否等价由 lead 的统一 verify 覆盖 |
| 6 | CPU host GEMV 只有 fp32 路径（无"按量化类型选内核"） | `cpu_matvec.cpp` | 两级 int8 激活 + AVX2 `maddubs` 内核，按 `(quant type, n_in)` 分派；`cpu_int8` 配置 **默认已改为 1**（实测见 §3） | 隔离层配对中位 **−10.1% / −7.5%**（两次独立测量）；`test_cpu_moe` 13/13 PASS；`--print-config` 显示默认 1 |

第一轮已改的（保留）：`tiny_sk_max_nout`(128) / `tiny_sk_min_nin`(512) / `cpu_pool_spin`(512) 三个配置键；
`vram_margin` / `vram_extend`（实测显存追加）/ `row_threshold` / `q6k_x4_min` / `q6k_split` 原本就是配置。

## 2. 外部智能体已解决的（本轮审计确认）

| 项 | 第一轮状态 | 现在 |
|---|---|---|
| **KV cache dtype = f32**（`runner.h:98` + `attn_cached_w` 的 `const float*`） | 写死 | kv-longctx 已加 `kv_type` 配置（`f32`/`q8_0`/`q4_0`，默认 `f32` 但长 ctx 自动按显存预算选择），并有 `kv_offload`/`kv_split`；非写死 |
| **attention 的 shared memory 是 O(max_seq)**（整行 score 放 shared） | 写死 | kv-longctx 已加分块 online-softmax（`attn_tiled` / `attn_scratch`）；我上一轮只修了"超 48 KiB 崩溃"的预算收敛，仍在（`runner.cpp` 的 `smem_for/smem_cap`） |
| prefill 的 GEMM 精度/内核 | 写死 f32 cuBLAS | prefill-fast 的 `prefill_f16` + `prefill_f16_min_macs` 配置（fp16 tensor core，`--print-config` 已可见） |

## 3. 仍"数值是实测来的"但已配置化的 3 处（物理上无法解析推导）

| 项 | 为什么不能解析推导 | 现状 |
|---|---|---|
| `attn_warp_ctx_mid/hi` 默认 96/384 | 这是"再加一档 warp 的收益（KV 延迟隐藏）何时超过代价（occupancy/同步）"的交叉点，取决于该 GPU 的 SM 数、L2、KV 布局；没有闭式模型 | 配置化（已改），默认=本机实测；换卡时用 A/B 重标 |
| `cpu_pool_spin` 默认 512（≈20 µs） | PAUSE 一条指令的周期数随 CPU 架构/频率变化（本机 ~140 cycle），"20 µs"才是需求，周期数要标定 | 配置化（已改）；下一轮可在启动时用 `std::chrono` 自动标一次（`spin = 20us / pause_cycle`） |
| `q6k_dp4a_split_max_nout` 默认 4096 | "多少行以后 split-K 的归约开销超过并行收益"取决于 SM 数/带宽比 | 配置化（已改）；也可用 `--kernel-calibrate` 对若干 n_out 扫出拐点写进标定缓存 |

## 4. 不是"模型死数"的常量（保留，附理由）

| 项 | 位置 | 理由 |
|---|---|---|
| attention graph 桶数组大小 3 | `runner.h` `attn_graph[3]` | 桶数 = attention 内核支持的**不同 blockDim 档数**（8/16/32 warp），是内核设计决定的，与模型无关；`attn_warp_ctx_*` 改两档边界也不会增加档数 |
| graph 懒捕获时序（第 1 token 正常跑/第 2 捕获/第 3 起重放） | `qwen35.cpp` | 驱动约束（本机 ctor 内捕获会 illegal memory access），不是调优值 |
| `head_topk` 的 k=5 | `runner.cpp` | 贪心解码只需要 argmax；5 只是诊断/verify 用，不参与语义 |
| `topk_partial` 的 block 数 `ceil(vocab/1940)` clamp 256 | `runner.cpp` | 由 vocab 与每 block 处理量推导（1940 来自 block/线程配置），不是模型常数 |
| `kDp4aQ5kSmemLimit = 4096` | `kernel_select.h` | 本身是 `3 * n_in <= 12 KiB` 的 shared 容量推导 |
| QK 块布局常量（Q4_K 144/256 等） | `cpu_matvec.cpp` | GGUF 量化格式定义，不是调优值 |
| 两级 int8 量化的 32 值块 | `cpu_matvec.cpp` | = 一个 AVX2 int8 向量宽度，硬件结构 |
| `xq_build` 的 127（int8 对称量化上限）、`rn()` 四舍五入 | `cpu_matvec.cpp` | 量化数学定义 |

## 5. 检查方法（可复现）

```bash
# 1) 全仓库没有任何模型名/文件名分支（应无输出）
grep -rniE "Coder|qwen3\.6|35B|9B|0\.5b|Qwen3\.5" src/ include/engine/ --include=*.cpp --include=*.h \
  | grep -viE "size|offset|base|0x|//|/\*|\*"
# 2) 所有配置键 + 默认值（本机实测输出见下）
./build/test_gen_gopt.exe --print-config | grep -E "cpu_int8|attn_warp|cpu_pool_spin|q6k_dp4a_split|small_nout|tiny_sk"
#   cpu_int8               = 1
#   tiny_sk_max_nout       = 128        tiny_sk_min_nin = 512
#   cpu_pool_spin          = 512
#   attn_warp_ctx_mid      = 96         attn_warp_ctx_hi = 384
#   q6k_dp4a_split_max_nout= 4096       small_nout_split_max = 256
# 3) 物理核推导（本机：6 核 / 12 逻辑 → 9 线程）
./build/dbg_topo.exe
# 4) 逐形状内核标定（把实测 plan 写进 engine-kernel-cache.txt）
./build/test_gen.exe <model.gguf> "760" 1 --kernel-calibrate
```

## 6. 结论（对用户那句话的回答）

"通用型，不要写死的调用数据"现在成立到以下程度：
- **没有任何按模型名/模型规模的硬编码分派**；
- 内核选择是 `(量化类型, n_out, n_in)` 的纯函数策略（`choose_gemv`），阈值全部可配置，
  且任意具体形状可被 `--kernel-calibrate` 的实测缓存覆盖；
- 显存放置是实测 `cuMemGetInfo` + 每层真实字节推导（`vram_extend` 只追加实测放得下的紧邻层）；
- attention 的 warp 分档、smem 上限、graph 分桶都由"结构量（head_dim / max_seq / 设备上限 / 物理核数）"
  推导或配置，不再是散落的魔数；
- 仍存在的"实测默认值"（§3 三条）都记录在案、可重标，且在 `docs/CONFIG.md` 里有名字。
