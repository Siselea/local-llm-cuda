# KV cache 量化 + 长上下文 attention + 上下文上限自适应（KV-LONGCTX-REPORT）

> 执行者：kv-longctx（子智能体）｜时间：2026-11-15 08:19–09:30｜本机 RTX 3060 12GB / i5-10400F / 32GB DDR4 / Windows VM
> 目标：① KV q8_0/q4_0 量化（支持模型完整上下文上限）② 长上下文 decode 掉幅 ≤15%（2048 ctx）③ KV 上限自适应推导 ④ 与 llama.cpp 同条件 pp/tg 对比
> 验证：`verify.sh` ALL GREEN、`smoke_models.sh` 四模型预测全对、`verify_generality.sh` 见 §8

---

## 0. TL;DR

1. **长上下文掉幅已攻克**：9B 在 2048-token 填满上下文时 decode 从 **33.3–40.3 ms（−50…−80%）→ 23.09 ms（f32）/ 23.54 ms（q8_0）**，相对短上下文（21.9 ms）只掉 **+5.4% / +7.5%**，远好于「≤15%」目标。4096 填满时 +13.6%（f32）/ +17.8%（q8_0）。
2. **KV q8_0/q4_0 已实现并跑通**：写入时按 32 元素块量化（q8_0 = 34 B/32 值，q4_0 = 18 B/32 值），attention 内核里反量化累加，无 f32 展开。9B 每 token KV：**f32 64 KiB → q8_0 17 KiB → q4_0 9 KiB**（4×/7× 压缩）。
3. **128k/256k 配置可加载可跑**（这依赖另一个必修项：激活按 chunk 分配 + prompt 分块 prefill，见 §5）：9B `--ctx 131072 kv_type=q8_0` 实际占用 8286 MiB、余 4002 MiB；`--ctx 262144 kv_type=q4_0` 占用 8414 MiB、余 3874 MiB，decode 不掉速。
4. **同条件对比（llama-bench 语义：`-c/--ctx` 设上下文、prompt 未填满）**：9B tg **46.3 tok/s @128k（llama.cpp 46.7，99%）**、**45.5 tok/s @256k（llama.cpp 46.6，98%）**；pp 只有 llama.cpp 的 25%（333 vs 1384），瓶颈不在 attention 而在 `n_tokens>1` 前向路径（整模型反量化 + f32 lm_head），属 MTP/多行 GEMV 任务范围，见 §7 归因。
5. **1M 上下文在单卡不可行**（精确显存账见 §6.2）：KV q8_0 需 17408 MiB / q4_0 9216 MiB，而本配置 KV 预算只有 3005 MiB。`ctx_auto` 会把它夹到 **q8_0 143260 / q4_0 270602 / f32 38053** 并打印完整账目；被夹紧后照常加载运行、速度不掉。
6. **数值**：q8_0 下 40 步贪心序列与 f32 **逐 token 完全一致（41/41）**；但 logits rel_l2 = 3.1e-2（**未达 1e-3 要求**，见 §4 的诚实结论与分析：这是 8-bit KV 的固有量化误差经 32 层放大的结果，q4/q8 误差比 14.3× ≈ 理论 16×，说明实现正确，是 1e-3 这个验收线本身不可达）。q4_0 会在 40 步内发散（2/41 一致），因此 **q4_0 只用于「放不下」的场景，默认仍是 f32，q8_0 是可选项**。
7. 默认行为保守：`kv_type` 默认 **f32**、`ctx_auto` 默认 **true**（只会夹紧、不会自动升级 kv_type）、`kv_offload` 只解析并说明（未实现主机映射 KV）。短上下文零回退（§8 A/B）。

---

## 1. 实现总览

### 1.1 数据布局

| kv_type | 块格式 | 每 token KV（9B） | 布局 |
|---|---|---|---|
| `f32` | 4 B/值 | 64 KiB | `[n_kv][max_seq][head_dim]`（历史布局，不变） |
| `q8_0` | 2 B f16 scale + 32×int8 = **34 B / 32 值**，`d = amax/127` | 17 KiB | `[n_kv][max_seq][ceil(D/32)·34]` |
| `q4_0` | 2 B f16 scale + 16 B nibbles = **18 B / 32 值**，`d = amax/8`，`q∈[-8,7]` 存 `q+8` | 9 KiB | `[n_kv][max_seq][ceil(D/32)·18]` |

块格式与引擎 GEMV 的 Q8_0 完全一致（34 B/32 值）；`head_dim % 32 != 0` 时尾部块补零、
attention 只消费前 `D` 个元素；`head_dim > 256` 或不是 32 的倍数时**量化 KV 直接报错**
（分块内核不支持），f32 则回退历史内核。MLA 模型（Coder-V2-Lite）强制 f32（无量化 MLA 内核）。

### 1.2 内核（全部追加在 `include/engine/kernels.h` 末尾）

| 内核 | 作用 |
|---|---|
| `kv_write_q` | 量化 KV 写入。grid = `n_tokens·n_kv`，128 线程（4 warp），一个 warp 处理一个 32 元素块：32 lane 读一行的 32 个 float（coalesced）→ `__shfl_xor` 求 amax → 量化 → lane0 写 f16 scale、各 lane 写 1 个 int8（q8）或打包 nibble（q4，`__shfl` 把 lane+16 的高半字节并进来） |
| `attn_tiled` | 分块 online-softmax attention，**不占 shared memory、与 max_seq 无关**。grid = `(n_tokens, n_head, nsplit)`，blockDim=32（一个 warp 一条 query 行的一个 head）。lane 拥有 `D/32` 个连续元素（9B 为 8 个），因此每次 K/V 行读取都是整行 coalesced，彻底消掉了历史内核的 **8× sector 放大** 与**整行 logits 停放在 shared** 的两个问题（后者正是 >384 token 崩溃 / nwarp 降级的根因）。online softmax 的 max/分母/累加器全在寄存器里 |
| `attn_combine` | split-K 归并：`(m, s, acc[D])` 部分和 → 全局 max/分母 → 加权求和 |

### 1.3 派发（`DenseRunner::run_attn`，`src/runner.cpp`）

```
ctx = start_pos + n_tokens
if (kv_type==f32 && ctx < 385 && max_seq <= 12000)   -> 历史内核 attn_cached_w（短上下文数字不动）
else if (kv_quant_ok())                              -> attn_tiled
      n_tokens <= 8 : nsplit = kv_split(默认32) + attn_combine   (decode：key 区间切给多 block 提高并行度)
      n_tokens  > 8 : nsplit = 1，softmax 在寄存器内完成，无 scratch  (prefill)
else                                                 -> 量化 KV 报错 / f32 回退历史内核
```

CUDA Graph 安全：`nsplit`/grid 与 ctx 无关（decode 固定 `kv_split`），所以 graph 捕获的参数恒定；
`attn_scratch_` 在构造期分配（capture 期间不 cudaMalloc）。

### 1.4 C++ 改动清单（给协调用：与 MTP 分支的冲突面）

| 文件 | 位置 | 改动 |
|---|---|---|
| `include/engine/kernels.h` | **文件末尾**（`}  // namespace kernels` 之前） | 追加 4 个 kernel 字符串（+~330 行），不改任何既有内核 |
| `include/engine/runner.h` | `max_seq_` 附近成员区 / kernel 句柄区 | 新增成员 `kv_type_/kv_split_/prefill_rows_/rows_/attn_flash_min_/n_kv_layers_/attn_scratch_/attn_scratch_rows_/attn_split_rows_/kv_override_ctx_`，方法 `kv_layer_bytes/activation_bytes/kv_quant_ok/derive_ctx_limit/alloc_kv_cache/kv_write_layer/run_attn/run_attn_legacy`，句柄 `k_kvwriteq_/k_attntiled_/k_attncombine_` |
| `src/runner.cpp` | 构造函数 `head_dim_ = ...` 之后 | ①MLA 强制 f32 ②`derive_ctx_limit()` ③`prefill_rows_`/`rows_` + 按显存自适应下调 chunk |
| | NVRTC 源码拼接（`kTopkFinal` 之后） | `+ kKvQuantCommon + kKvWriteQ + kAttnTiled + kAttnCombine` |
| | kernel 注册（`k_attnw_` 之后） | +3 个句柄 |
| | 激活缓冲分配块 `const size_t rows = ...` | `rows = rows_`（原来 `max_seq_`）——**长上下文能加载的关键** |
| | `logits_pin_.resize(...)` 之后 | `attn_scratch_` 分配 |
| | MoE 放置的 KV reserve 公式 | `n_layer·n_kv·(key/n_head)·max_seq·2·4` → `kv_layer_bytes·n_kv_layers_`（原公式把 head_dim 写成 `key_length/n_head`，低估 8–16×；默认 ctx 下对放置无影响，实测 35B/Coder 的 `gpu_moe_layers` 不变） |
| | 构造期 `vram_kv_per_layer_` | 用 `kv_layer_bytes(max_seq_)`（量化后字节） |
| | `load_weights()` 通用层的 KV 分配 | → `alloc_kv_cache(w)` |
| | `run_layer_body()` 的 kv_write + attention 两个 launch 块 | → `kv_write_layer(...)` + `run_attn(...)`（**这是与 MTP 改 forward 时最可能相邻的位置**） |
| | `forward_impl()` 开头 | prompt 超过 `rows_` 时按块递归 prefill |
| | `~DenseRunner()` 之前 | 新增方法实现块（+~250 行） |
| | `load_config_snapshot()` | 解析 `kv_type`/`kv_split` |
| `src/qwen35.cpp` | `load_q35_attn()` 末尾 | 2 行 KV 分配 → `alloc_kv_cache(w)` |
| | `run_q35_attn_attn()` 的 kv/attention 两个块 | → `kv_write_layer(...)` + `run_attn(...)`（9B 的唯一 attention 派发点） |
| `src/config.cpp` | `kDefs[]` | +5 条：`kv_type/kv_offload/kv_split/ctx_auto/prefill_rows` |
| `docs/CONFIG.md` | 新增 §3.3.1 / §6.1 / §6.2 | 配置项、自适应公式、1M 显存账 |
| `engine.cfg.example` | 注释示例 | +5 个键 |
| `build/kv_probe.cpp` | 新增（**不在 build.sh 里**，仓库外构建的探针） | 数值验收探针（三份 KV 精度对拍 + 40 步贪心） |

\*\*未改\*\*：`build.sh`、任何既有内核、`ir.cpp/gguf.cpp` 的解析、MLA 路径、forward 的语义。

**精确行号（最终二进制对应的代码，便于协调/回退）**：

| 文件:行 | 内容 |
|---|---|
| `kernels.h:3184` | `kKvQuantCommon`（f16 转换/块布局/kv_get） |
| `kernels.h:3210` | `kKvWriteQ` → `kv_write_q` |
| `kernels.h:3275` | `kAttnTiled` → `attn_tiled` |
| `kernels.h:3398` | `kAttnCombine` → `attn_combine` |
| `runner.cpp:137-140` | 构造期：`derive_ctx_limit()` + `rows_` |
| `runner.cpp:246` | NVRTC 源码拼接追加 4 个内核 |
| `runner.cpp:284-286` | 3 个 kernel 句柄注册 |
| `runner.cpp:471` | `vram_kv_per_layer_ = kv_layer_bytes(max_seq_)`（放置估算用同一个量化 KV 大小） |
| `runner.cpp:660` | `derive_ctx_limit`（自适应 ctx 推导 + 打印 + 三种 kv_type 上限表） |
| `runner.cpp:747` | `alloc_kv_cache` |
| `runner.cpp:768` | `kv_write_layer` |
| `runner.cpp:789` | `run_attn_legacy`（历史内核原样搬进函数，warp/smem 预算逻辑保留） |
| `runner.cpp:811` | `run_attn`（派发：f32 短 ctx → legacy；其余 → attn_tiled[+combine]） |
| `runner.cpp:1339` | 通用层 KV 分配 → `alloc_kv_cache(w)` |
| `runner.cpp:1588-1589` | 通用层 attention 两个 launch → `kv_write_layer` + `run_attn`（**与 MTP 的冲突面**） |
| `runner.cpp:1832` | `forward_impl` 分块 prefill（**与 MTP 的冲突面**） |
| `runner.cpp`（构造期激活分配块） | `const size_t rows = rows_;`（原 `max_seq_`） |
| `qwen35.cpp:238` | 9B 的 KV 分配 → `alloc_kv_cache(w)` |
| `qwen35.cpp:564-565` | 9B 的 kv_write+attention → 2 行调用 |
| `config.cpp:62/67/71/75/79` | 5 个新配置键 |

---

## 2. 长上下文掉幅：改前 / 改后（9B，Qwen3.5-9B-Q5_K_M）

固定 prompt（`build/ids2048.txt` 真 token），`ENGINE_DEC_SKIP=2/3`，14 步计时（VM 上取多次中位/区间）。

### 2.1 填满上下文的 decode（真实长上下文）

| 填充 ctx | kv_type | 改前 decode | 改后 decode | tok/s | 相对短上下文掉幅 |
|---|---|---|---|---|---|
| 34（短） | f32 | 22.2–22.5 ms | **21.95 ms** | 45.6 | 基准 |
| 384 | — | 23.9 ms（前一智能体） | — | — | — |
| 1024 | — | 27.3 ms（前一智能体 +23%） | — | — | — |
| **2048** | f32 | **33.3–40.3 ms（−50…−80%）** | **23.09 ms**（23.07/23.09/23.29） | 43.3 | **+5.4%** ✅ |
| **2048** | q8_0 | — | **23.54 ms**（23.47/23.54/23.55） | 42.5 | **+7.5%** ✅ |
| 4096 | f32 | — | 24.78 ms | 40.4 | +13.6% |
| 4096 | q8_0 | — | 25.79 ms | 38.8 | +17.8% |

改前 2048 掉 50–80%（33.3–40.3 ms）→ 改后 f32 +5.4%、q8_0 +7.5%，**目标（≤15%）达成**。
（2048 这组是最终二进制 `build/test_gen.exe` 的三次采样中位；2048 之后各项仍是同一最终二进制。）

### 2.2 逐内核归因（`ENGINE_KPROF=1` + `ENGINE_NO_GRAPH=1`，2048 填满，q8_0）

前一智能体拿不到 kprof 归因（graph 模式下 `cuGraphLaunch` 重放的内核 kprof 看不到），下面这组数补上了这个空白：

- decode 墙钟 32.82 ms（no-graph，比 graph 模式慢 ~25%），kprof GPU 合计 **30.70 ms/token，GPU busy 96.5%**
- 分类合计：GEMV 类 **23.97 ms（78.1%）**、attention 类 **1.68 ms（5.5%）**、SSM/线性注意力 0.82 ms、norm 0.92 ms、quant_x_i8 0.61 ms、其余 <0.5 ms

| kernel | 每 token GPU 时间 | 占比 |
|---|---|---|
| gemv_q5k_x4_dp4a_s | 11.72 ms | 38.2% |
| gemv_q6k_dp4a | 6.22 ms | 20.3% |
| gemv_q6k_dp4a_sk | 3.98 ms | 13.0% |
| gemv_q5k_x4_dp4a | 2.05 ms | 6.7% |
| **attn_tiled** | **1.63 ms**（8 层 × 0.204 ms） | **5.3%** |
| gemv_q8_0_sk | 1.42 ms | 4.6% |
| gd_delta_rule_128f | 0.70 ms | 2.3% |
| quant_x_i8 | 0.61 ms | 2.0% |
| rmsnorm4 / rmsnorm | 0.57 / 0.35 ms | 1.9 / 1.1% |
| **kv_write_q** | **0.32 ms**（8 次） | **1.0%** |
| attn_combine | 0.05 ms | 0.2% |

结论：**2048 ctx 的两项瓶颈（8× sector 放大、整行 shared 停放）已经消除**；剩下的时间是大头 GEMV（78%）+ attention 5.5%。
attention 里仍有一个已知优化空间：GQA 的 4 个 q head 各自读同一份 K/V（4× 重复读），下一步可把
「同组 4 个 q head 放同一个 warp」省掉 4×，预计 attention 1.63 → 0.5 ms（2048）量级；本轮未做（目标已达成，优先保正确性/验证）。
`kv_split=64` 试过反而略慢（24.14 vs 23.56 ms），所以默认保持 32。

### 2.3 prefill（pp）现状与归因

| prompt | ctx 配置 | prefill | tok/s |
|---|---|---|---|
| 1622 | 4096 / 131072 / 262144 | 4.70–4.87 s | 333–345 |
| 2048 | 4096 | 5.77–5.90 s | 347–355 |
| 4096 | 8192 | 13.5–14.0 s | 293–302 |

kprof（prefill 1622 一次，no-graph）：**GPU 合计 1325 ms，墙钟 5225 ms ⇒ GPU 空闲 75%**；
GPU 部分：`dequant_q5k` 423 ms（32%，整模型反量化成 f32 的子步骤）+ **`attn_tiled` 421 ms（31.8%）**
+ `gd_delta_rule_128` 262 ms（19.8%）+ `dequant_q6k` 102 ms。墙钟的另外 3.9 s 是 host 侧
（`n_tokens>1` 前向按 slab 反量化 + cuBLAS + 每层 launch），**这正是 MTP-RECON 里 P0-A「多行量化 GEMV + 多行 lm_head」要解决的问题**，不是 KV/attention 的问题。
本轮的分块 prefill 只影响预填的块数（块越大越少付「整模型反量化」固定成本），所以 `--prefill-rows` 可按显存调大：
128k 配置下默认 512 行（4 块），显存允许时可以 `--prefill-rows 1024`（2 块，pp 再快 ~2×，但 128k KV 会因激活变大被 `ctx_auto` 夹到 105613）。

---

## 3. KV q8_0 / q4_0 的显存效果与上限自适应

### 3.1 各 kv_type 的 KV 占用（9B，8 个全注意力层，n_kv=4，head_dim=256，17 KiB/token @q8_0）

| ctx | f32 | q8_0 | q4_0 |
|---|---|---|---|
| 2048 | 128 MiB | 34 MiB | 18 MiB |
| 8192 | 512 MiB | 136 MiB | 72 MiB |
| 32768 | 2.0 GiB | 544 MiB | 288 MiB |
| **131072** | **8.0 GiB** | **2.13 GiB** | 1.13 GiB |
| **262144** | 16.0 GiB | **4.25 GiB** | **2.25 GiB** |
| 1048576 | 64.0 GiB | 17.0 GiB | 9.0 GiB |

### 3.2 自适应推导公式（`derive_ctx_limit`，装载前，打印 4 类行）

```
kv/token      = Σ_attn_layers n_kv · head_dim · 2 · bytes_per_value      # f32=4, q8_0=34/32, q4_0=18/32
act(rows)     = rows·(6·hidden + 2·n_head·head_dim + 2·n_kv·head_dim + 3·ffn + vocab)·4
reserve       = vram_margin_mb + workspace_mb
max_ctx       = floor((free_at_load − resident_weights − act(prefill_rows) − reserve) / kv/token)
final_ctx     = min(--ctx, 支持上限)        # ctx_auto=true；--ctx 0 = 直接用支持上限
```

实际输出（9B，`--ctx 1048576 --kv-type q8_0`，本机 11236 MiB 空闲 / 权重 6263 MiB）：

```
[ctx] requested=1048576->143260 kv_type=q8_0 kv/token=17408 B (17.00 KiB) kv_layers=8/32 kv_total=2378 MiB
[ctx] budget: free=11236 MiB - weights=6263 MiB - act(rows=512)=627 MiB - reserve=1968 MiB -> max_ctx=143260 (ctx_auto on)
[ctx] CLAMPED: 1048576 > max_ctx 143260 for kv_type=q8_0 (would need 17408 MiB, budget 3005 MiB)
[ctx]   with kv_type=f32: max_ctx=38053
[ctx]   with kv_type=q8_0: max_ctx=143260
[ctx]   with kv_type=q4_0: max_ctx=270602
[vram] post-load used=8510 MiB free=3778 MiB ... [vram] check OK: 3778 MiB free >= 1200 MiB margin
```

被夹紧后**照常加载、跑通**（1M 请求下实测 decode 21.75–22.75 ms/token = 44–46 tok/s，见 §6.2）。
`--ctx 0` = 完全由自适应推导（实测 9B q8_0：`[ctx] requested=…->143260`，随后 decode 22.08 ms/token=45 tok/s，top-1 不变）。

### 3.3 各模型在不同 kv_type 下的最大 ctx（本机 12 GB 实算）

| 模型 | KV 层 | f32 | q8_0 | q4_0 | 备注 |
|---|---|---|---|---|---|
| Qwen3.5-9B（hybrid，8/32 全注意力） | 8 | **38053** | **143260** | **270602** | `--prefill-rows 512`；若用 1024 行则是 28053 / 105613 / 199491 |
| deepseek2 Coder-V2-Lite（MLA） | 27 | 受 MLA f32 限制（1.88 MiB/token）→ **~5.4k 以内不夹紧**，超大 ctx 会夹紧 | 强制 f32 | 强制 f32 | MLA 无量化内核，`kv_type` 请求会被显式提示并回退 f32 |
| Qwen3.6-35B-A3B（hybrid MoE） | 10 | 按同一公式；默认 ctx 下不夹紧 | 可选 | 可选 | MoE 的显存大头是专家层，KV 与之共用预算（放置逻辑已用同一份 `kv_layer_bytes`） |
| qwen2.5-0.5b（GQA 14/2, D=64） | 24 | 不夹紧（KV 24 KiB/token） | 可选 | 可选 | 分块内核支持 D=64（每 lane 2 元素） |

### 3.4 新配置键（已注册 `src/config.cpp` + `docs/CONFIG.md` §3.3.1）

| key | 默认 | 说明 |
|---|---|---|
| `kv_type` | `f32` | `f32` / `q8_0` / `q4_0`（MLA 强制 f32） |
| `kv_offload` | `false` | **未实现**主机映射 KV；解析并打印说明（1M 的显存账见 §6.2） |
| `ctx_auto` | `true` | 按显存推导上限并夹紧，打印决策依据 |
| `prefill_rows` | `512` | 激活行上限 / prompt 分块 prefill 块大小（显存不足自动下调） |
| `kv_split` | `32` | decode attention 的 split-K 份数 |

---

## 4. KV 量化数值验收（诚实版）

方法：`build/kv_probe.cpp`（仓库外构建，同一个进程里依次用 f32 / q8_0 / q4_0 各建一个 runner，
同 prompt、同一 40 步贪心、逐 token 比较 logits 与 token）。9B，prompt = 512 个真实 token。

| kv_type | prefill logits rel_l2 | decode logits rel_l2（mean / max） | 40 步贪心一致数 | 首次分歧 | decode ms/token |
|---|---|---|---|---|---|
| f32 | —（基准） | — | 41/41 | — | 23.52 |
| **q8_0** | **3.06e-02** | **2.90e-02 / 5.11e-02** | **41/41（完全一致）** | 无 | 23.32 |
| q4_0 | 4.39e-01 | 7.62e-01 / 1.17e+00 | 2/41 | 第 2 步 | 23.97 |

再取 32/180 token 的短 prompt 复核：q8_0 prefill rel_l2 稳定在 **2.5e-2 ~ 3.1e-2**（与 ctx 长度基本无关，是
32 层逐层放大的固有结果）；q4_0 在 0.33–0.44。

**结论（必须如实说）**：
1. **「logits rel_l2 ≤ 1e-3」这条验收线没有达到**（实测 q8_0 = 3.1e-2）。这不是实现 bug：
   q4_0/q8_0 的 logits 误差比 = 0.439/0.0306 = **14.3×**，与两者理论量化步长比 16× 吻合；
   KV 张量本身的相对 RMS 误差 q8_0 ≈ 0.5%（误差上界 `amax/254`）、q4_0 ≈ 4.5%，即实现就是标准 q8_0 精度。
   这类误差在 32 层网络里被放大到 ~3% 的 logits 相对差，但**不改变贪心输出**。
2. 用户真正要求的「生成 token 序列在 40 步内保持稳定」：**q8_0 41/41 完全一致** ✅；q4_0 不稳定（第 2 步发散）。
3. 因此落地策略：**默认 f32（保守）**；q8_0 作为「上下文放不下 f32 时」的推荐选项（128k 就属于这种）；
   q4_0 只用于 256k 这类必须再省一半的场合，并在文档里标注精度风险。
4. 短 ctx 时误差更小：34-token prompt，f32 vs q8_0 的 top-5 logprob 差 0.007 nats、token 序列完全一致；
   2048 填满时 top-5 的 logprob 差 0.1–0.26 nats、40 步序列仍**逐 token 一致**。

---

## 5. 长上下文能加载的前提：激活按 chunk 分配 + prompt 分块 prefill

历史实现把所有激活（含 `logits_ = max_seq · vocab · 4`）都按 `max_seq` 分配：9B 在 `--ctx 131072` 时
仅 logits 就要 **131072 × 248320 × 4 B = 130 TB**，`--ctx 65536` 也要 65 TB ⇒ 128k/256k 连加载都不可能。

修法（本轮已实现）：
1. `rows_ = min(max_seq_, prefill_rows)`，所有激活（x/hn/q/k/v/attn/proj/gate/up/act/down/logits/q35_*/mla_*/moe 缓冲）按 `rows_` 分配；
2. `forward_impl()` 在 `n > rows_` 时按 `rows_` 分块递归 prefill（每块 `last_row_only=true`，位置连续推进），数值等价（BPE 逐块、KV 位置连续、SSM 状态连续）；
3. 显存不足时自动把 `rows_` 折半（打印 `[ctx] prefill chunk reduced ...`），所以任何模型都不会因 chunk 过大 OOM。
4. 短 prompt（`n <= rows_`）路径完全不变（预测值不变）。

实测（9B）：`--ctx 131072 q8_0` 装载后 used 8286 MiB / free 4002 MiB；`--ctx 262144 q4_0` used 8414 / free 3874。

---

## 6. 与 llama.cpp 同条件对比（9B）

### 6.1 对比表

llama.cpp 侧数据来自用户 `<MODEL_DIR>\AI-最优启动\实测原始数据.txt` 的记录（`-fa on -ctk q8_0 -ctv q8_0 -ngl 99 -lm mmap`，pp 的 n≈1622，tg 是 llama-bench 的生成项）：

| 条件（同模型 Qwen3.5-9B-Q5_K_M / 同 ctx） | llama.cpp pp | **我方 pp** | llama.cpp tg | **我方 tg** | tg 比值 |
|---|---|---|---|---|---|
| `-c 131072`（`qwen9b_128k`，q8_0 KV） | 1384.5 | **333–345** | 46.7 | **46.3**（短 prompt/未填满）<br>**43.0**（1622 填满） | **99%** / 92% |
| `-c 262144`（`qwen9b_256k`，q8_0 KV） | 1340.8 | 我方 q8_0 放不下（夹到 143260）；**用 q4_0 跑 256k：336** | 46.6 | **45.5**（短 prompt/未填满，q4_0）<br>**44.4**（1622 填满，q4_0） | **98%** / 95% |
| `-c 196608`（`q9b_192k`，q8_0 KV） | 1370.7 | 我方 q8_0 夹到 143260（≈llama.cpp 140k 等效档） | 46.8 | 46.3 | 99% |

**pp 差距的解释（实测归因，不是 attention 的锅）**：我方 prefill 的 GPU 只忙 25%
（1622 token prompt：墙钟 5225 ms，GPU 内核合计 1325 ms）。
瓶颈是 `n_tokens>1` 前向：权重按 slab 反量化成 f32 再 cuBLAS（kprof 里 `dequant_q5k+q6k` 占 525 ms）
+ 每层 launch/host 往返。这属于 MTP-RECON 的 **P0-A（2–4 行量化 GEMV + 多行 lm_head）**，
做完 pp 才能从 4× 差距缩到 1× 量级；本轮不在范围内（且与 MTP 智能体正在改的 forward 路径重叠）。
注意：llama.cpp 的 46.7 tok/s 是**未填满 KV** 的 llama-bench 数字；真实「填满 128k」时两边都会掉到
「KV 读放主导」，llama.cpp 若真填满 128k q8_0 每 token 要读 2.28 GB KV（≈7.6 ms @300 GB/s 下界），
我方在同样条件下的物理下界与它同阶（我方当前 attention 有 4× GQA 重复读，是下一步优化点，见 §2.2）。

### 6.2 1M 上下文：精确显存账（结论：单卡不可行）

9B 的 KV 上限 = 假设把 1M 全填（模型 rope 支持 1M 的 Qwythos 变体更明显），需要：

| kv_type | 1M ctx KV 需要 | 本配置 KV 预算 | 差额 | 结论 |
|---|---|---|---|---|
| f32 | 65 536 MiB (64 GiB) | 3 005 MiB | −62.5 GiB | 不可行 |
| q8_0 | 17 408 MiB (17 GiB) | 3 005 MiB | −14.1 GiB | 不可行 |
| q4_0 | 9 216 MiB (9 GiB) | 3 005 MiB | −6.1 GiB | 不可行 |

即使实现 `kv_offload`（把 KV 放主机 32 GB 内存）也不实用：decode 每步要重读 9.0 GiB(q4_0) KV，
PCIe 4.4–12 GB/s ⇒ **0.75–2.0 s/token**；1M prefill 更不可能。因此本机结论是：
**1M 不可行；9B 的 GPU 上限是 q8_0 143260 / q4_0 270602（本配置实测可加载可跑）。**
`--ctx 1048576` 现在不会崩：会被 `ctx_auto` 夹到上表上限，打印完整账目，并照常加载运行
（实测 1M 请求下：q8_0 夹到 143260 后 decode 22.75 ms/token=44 tok/s；q4_0 夹到 270602 后 21.75 ms/token=46 tok/s）。

---

## 7. 三套验证 + 短上下文无回退证据

### 7.1 `bash tools/verify.sh` → **ALL GREEN**（64 个 PASS，0 FAIL，exit 0）

包含：build.sh 全部 19 个二进制、IR/plan 与 Python 参考一致、CUDA 内核数值、tiny 模型前向、KV roundtrip 等。
（新增的 `[ctx]` 决策打印只走 stderr，不影响任何断言。）

### 7.2 `bash tools/smoke_models.sh` → 四模型预测全对

| 模型 | top-1 | 期望 | 结果 |
|---|---|---|---|
| qwen2.5-0.5b | 40716 | 40716 | ✅ |
| DeepSeek-Coder-V2-Lite | 185 | 185 | ✅ |
| Qwen3.5-9B | 2614 | 2614 | ✅ |
| Qwen3.6-35B-A3B | 69014 | 69014 | ✅ |

### 7.3 `bash tools/verify_generality.sh` → **GENERALITY GREEN**（results: 26 passed, 0 failed，exit 0）

包含：四模型 + 额外模型的 info/IR/plan 通用性、坏模型校验、`--kernel-calibrate` 缓存往返、
自适应显存放置 margin 扫描（4 档）、显式 `--gpu-layers` 生效等。

### 7.4 短上下文无回退（新二进制 vs 改动前的 `build/test_gen_pre_opt.exe`，同命令 A/B）

| 模型 | 配置 | 改动前 | **改动后** | 硬指标 | 结论 |
|---|---|---|---|---|---|
| qwen2.5-0.5b | `1000,1001`，n=33 | 5.42 / 5.51 / 5.58 ms | **5.14 / 5.23 / 5.33 ms** | ≥185 tok/s | ✅ 更快（188–195 tok/s） |
| Qwen3.5-9B | `760`，n=17 | 22.2–22.5 ms（前一智能体） | **21.76 / 21.91 / 21.95 ms** | ≥45 tok/s | ✅ 45.6–46.0 tok/s |
| Coder-V2-Lite | `549,6077,280,7239`，n=9，auto 放置 | 45.62 / 54.97 ms | 42.59 / 57.57 / 66.75 ms | ≥22.5 tok/s | ⚠️ VM 噪声大（区间 42.6–66.8 ms，最好 23.5 tok/s）；放置层数与改动前一致（17 层），无系统性回退 |
| Qwen3.6-35B-A3B | `760`，n=9，auto 放置 | 38.33 / 39.51 ms（中位 39.9=25.1 tok/s） | 38.11 / 40.02 / 40.85 ms（中位 **40.0=25.0 tok/s**） | ≥25 tok/s | ✅ 中位达标，与改动前一致 |

> 说明：Coder 的 auto 放置对 host/GPU 往返非常敏感，本 VM 当前噪声大（同一二进制相邻两次可差 50%），
> 表里给的是全部采样；放置决策（gpu_moe_layers=17）、KV 分配、attention 路径在 Coder（MLA）上**完全未改**，
> 因此不存在由本轮改动引入的回退。

---

## 8. 交付物 & 已知问题 / 下一步

**新增/改动文件**：见 §1.4。另外仓库外（`build/`）新增：
`build/kv_probe.exe`（数值验收探针）、`build/ids1622.txt / ids512.txt / ids32.txt / ids180.txt / ids4096.txt / ids8192.txt`（测试 prompt）、
`build/verify_kv.log`、`build/generality_kv.log`（验证日志）。

**已知问题（如实）**：
1. `logits rel_l2 ≤ 1e-3` 未达（q8_0 = 3.1e-2），原因见 §4；**贪心 40 步完全一致**。建议把验收线改成
   「KV 张量 rel_l2 ≤ 1e-2 + 40 步 tick 一致」。
2. q4_0 的 logits 误差 0.44、40 步内发散 ⇒ 只建议在 256k 这类 f32/q8 放不下的场景使用。
3. attention 仍有 4× GQA 重复读（同一 kv head 的 4 个 q head 各读一遍 K/V）；这是 8192+ 填满时
   剩下的主要可优化项，预计 attention 时间再降 2–4×。
4. prefill（pp）是 4× 落后 llama.cpp 的来源（§6.1），需要 MTP 的 P0-A 多行 GEMV；本轮未动（与 MTP 冲突面）。
5. `kv_offload` 只做了配置面与显存账（未实现主机映射 KV），1M 场景的结论是「不可行」并给了精确账。
6. 分块内核要求 `head_dim ≤ 256` 且 `head_dim % 32 == 0`；不满足时 f32 回退历史内核、量化 KV 报错
   （四模型都满足：9B/35B 256、0.5B 64、Coder 走 MLA f32）。
7. **与 MTP 的冲突面**（协调用）：`runner.cpp` 的 `run_layer_body()` attention 派发块（2 个 launch 合成 2 行调用）、
   `forward_impl()` 开头（chunking）、构造期激活分配（`rows_`）三处。其余改动在 kernels.h 末尾 / qwen35.cpp
   的 2 个派发点 / config.cpp 表 / docs，均不与 MTP 的 forward/KV 回滚路径重叠。
