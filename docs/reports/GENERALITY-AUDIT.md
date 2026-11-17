# 泛化性审计：硬编码模型假设与魔法常量

> 审计目标：找出阻止本引擎成为 llama.cpp / vLLM 那类"通用推理引擎"的全部硬编码假设。
> 性质：**只读审计**，不修改任何源文件；本文件即全部产物。
> 结论：不是"再加几个模型"的问题，而是**模型描述（IR）、内核选择、放置预算、入口 API 四层都缺少可配置边界**。
> 最要命的一类不是"跑不起来"，而是**跑起来但静默给出错误数值**（P0 的 6 项里有 5 项属于此类）。

---

## 0. 验证方法（行号基准）

- 审计时间：**2026-11-14 19:28–19:52**（Asia/Shanghai）；源码快照为此区间内工作区状态。
- 行号基准文件（`wc -l`）：
  `src/runner.cpp` 2899、`src/qwen35.cpp` 517、`src/ir.cpp` 551、`src/planner.cpp` 82、`src/cpu_matvec.cpp` 658、`src/cuda.cpp` 606、`src/cublas.cpp` 154、`src/main.cpp` 356、`tests/test_gen.cpp` 137、`include/engine/kernels.h` 2758、`include/engine/planner.h` 42、`include/engine/runner.h` 354。
- 快照指纹（`md5sum`，便于行号漂移后确认是否同一版本）：
  `src/runner.cpp` d9407f4e…、`src/qwen35.cpp` 93acda57…、`src/ir.cpp` 51acbf8d…、`src/cuda.cpp` a877718b…、`src/cublas.cpp` de6883bf…、`include/engine/kernels.h` 9e11096d…、`src/cpu_matvec.cpp` c8b8eaed…、`tests/test_gen.cpp` 8cb73337…、`include/engine/planner.h` 0884478c…。
- 复核方式：`grep -n "<常量名>" src/runner.cpp`，或 `sed -n '309,311p' src/runner.cpp`。下文每条都可用"文件:行号"直接定位，必要时用该行上下 ±3 行确认。
- **并发改动说明（重要）**：审计期间工作区正在被并行重构，分两波：
  1. 19:44–19:47 新增 `include/engine/config.h`、`src/config.cpp`（约 48 个配置键）、`include/engine/kernel_select.h`、`src/kernel_select.cpp`；
  2. 19:48–19:50 `src/runner.cpp`（2899→3095 行）、`src/qwen35.cpp`（517→527 行）、`include/engine/runner.h`（354→360 行）被改写：kernel 选择从 `matmul()` 抽到 `kernel_select.*`、部分常量接入 config。
  本报告"现状"列描述的是**重构前快照**（行号以上表为准，md5 用来确认）；重构后若行号对不上，用附录 B 的 `grep` 重新定位。`include/engine/runner.h` 在 19:50 版本的对应位置：`max_seq` 默认 `20`/`208`、`rope_base_`/`eps_` `210`/`211`、`dev_route_default_` `309`。
  **注意：参数化 ≠ 泛化**。`config`/`kernel_select` 化能解决 C 类"阈值不可调"，但**不会**修掉 A/D 类逻辑缺陷（`ne > 256` 静默返回、`K != 4` 静默返回、SWA 未实现、safetensors 全 f32、路由语义缺字段）——这些仍是本报告 P0/P1 的主体。
- 统计口径：`getenv()` 调用 **129 处**、不同变量名 **87 个**（`grep -rn getenv src include tests --include=*.cpp --include=*.h | wc -l`；名去重 `grep -rhoE "(ENGINE|RUNNER|DSH|PROBE)_[A-Z0-9_]+" ... | sort -u | wc -l`）。

---

## 1. 摘要

### 1.1 计数

| 类别 | 条目数 | 说明 |
|---|---|---|
| A. 架构 / 模型名特判 | 12 | 靠 `arch` 字符串、variant 字符串、模型族私有的张量命名/排布做分支 |
| B. 维度常量 | 18 | 不来自 GGUF metadata / safetensors config，而是写死或静默兜底 |
| C. 调优常量 | 21 | VRAM 余量、kernel 选择阈值、split-K、warp 数等经验值 |
| D. 后端 / 环境硬编码 | 16 | DLL 路径、单 GPU、线程池、8 行/块、256 专家、Windows-only |
| E. 入口 / CLI 限制 | 16 | 只有 info/ir/plan，无 run；无 tokenizer；test_gen 是唯一推理入口 |
| **合计** | **83** | 其中 P0 6 项、P1 40 项、P2 37 项 |

### 1.2 五个最危险项

| 排名 | 位置 | 问题 | 为什么最危险 |
|---|---|---|---|
| 1 | `include/engine/kernels.h:2028-2036` | `moe_route`：`if (ne > 256) return;` + 栈数组 `float pr[256]` | 专家数 >256 时内核**直接返回**，`slot_exp/slot_w` 保持未初始化/上次残留，后续 gate/up/down kernel 用垃圾专家下标跑 → 静默错误结果（触发条件：启用设备端路由，见 `src/runner.cpp:1481/1817`）|
| 2 | `include/engine/kernels.h:1137-1156` | `gd_conv_state`：`if (K != 4) return;`（注释"K is always 4 for qwen35"）| 任何 `ssm.conv_kernel != 4` 的 hybrid 模型，卷积状态永不更新，**第 2 个 prefill chunk / 第 2 个 decode step 起数值即错**，且无任何报错 |
| 3 | `src/ir.cpp:155,159` + 全仓库无使用点 | `sliding_window`、`leading_dense_blocks` 被解析进 IR，runner/planner 从不读取 | 带滑窗注意力（Mistral / Qwen2.5-SWA / Gemma 系）的 GGUF 会静默按**全注意力**算：logits 错，且 KV 按全上下文分配（`src/runner.cpp:860`）→ 可能 OOM |
| 4 | `src/runner.cpp:599,812-858` | safetensors 容器只实现 llama-like dense / 普通 MoE；MoE 路径 **无条件把所有专家反量化成 f32 上传 GPU**（`838-853`），`gpu_experts_` 只影响 host 副本 | 任何真实 HF safetensors MoE（DeepSeek/Qwen3-MoE）必然爆显存；SSM/MLA 直接以"缺张量"失败 |
| 5 | `src/runner.cpp:1522-1548` 与 `kernels.h:2028-2071` | 路由语义写死两种：`softmax→topk[→renorm]` / `sigmoid→topk`；无 `routed_scaling_factor`、无 group-limited routing、无 `e_score_correction_bias`；host 与 device 是**两份独立的 top-k/归一化实现**（device 端 `2065-2070` 做 renorm，host 端 `1542-1547` 做 partial_sort 截断） | DeepSeek-V3 类模型静默算错 MoE 权重；两份实现各自演进、缺少共同的"路由契约"，新增路由特性必须改两处 |

> 第 6 名（本应并列 P0）：`src/runner.cpp:289-290` 的放置预算用 `hd = key_length / n_head` 估算 KV——`key_length` 语义是 **per-head dim**（见 `pyref/ir.py:183` 注释），这里再除 `n_head` 会把 KV 低估 1~2 个数量级，且没有 MLA / hybrid 分支 → 过度往 GPU 放专家 → 0 MiB free / 驱动分页。`src/ir.cpp:121-131` 与 `pyref/ir.py:170-184` 早已有正确的 `kv_bytes_per_token()`，只是 runner 没用。

### 1.3 一句话结论

引擎目前是"**四模型专用实现**"：模型描述层只覆盖 llama-like / deepseek2(MLA) / qwen3.5(hybrid) 三条硬编码分支；kernel 选择层有 10+ 个写死的形状阈值；放置层把整机常量（12 GiB、2560 MiB、768 MiB、360 GB/s）当事实；入口层只有"给 token id 出 token id"。要变成通用引擎，缺的不是性能优化，而是**契约**：IR 应成为唯一模型事实来源，配置层应成为唯一策略来源，入口应支持文本与采样。

---

## 2. 分类清单

格式：`编号 | 文件:行号 | 现状 | 风险 | 建议`

### A. 架构 / 模型名特判（12 项）

| # | 文件:行号 | 现状 | 风险 | 建议 |
|---|---|---|---|---|
| A1 | `src/runner.cpp:442-454` | `upload_norm()` 中 `ir_.arch == "deepseek2"` 才做"是否 ones-centred"的数据探测：`size>=64` 且均值 `< 0.8` 则给 norm 权重 +1；`ENGINE_NORM_PLUS1` 可覆盖 | 语义绑死在 arch 名上：其它 arch 的 zero-centred 权重不会被修正（静默小偏差）；deepseek2 权重统计异常时误判 | 把 `norm_plus1` 变成 IR 显式字段（由 converter metadata 或张量统计写入），移除 arch 字符串判断 |
| A2 | `src/ir.cpp:181` | `if (m.arch == "deepseek3") mo.router_kind = "sigmoid_topk";` | 全 IR 唯一的 arch 字符串特判；其它 sigmoid 门控架构（V3 派生、部分国产 MoE）会走 softmax，路由权重全错 | 从 GGUF metadata（`expert_gating_func` 等）推导；无 metadata 时由配置覆盖，而非 arch 名 |
| A3 | `src/ir.cpp:251-254` | 检测到 SSM 且 `moe.n_expert > 0` 时**强制** `norm_topk = true`、`n_shared == 0` 时置 1 | qwen3.5-MoE 专属假设；其它 hybrid-MoE 会静默改变路由归一化和 shared expert 数量 | 只从 metadata（`expert_weights_norm` / `expert_shared_count`）读取；缺失时告警，不猜测 |
| A4 | `src/ir.cpp:255-257` | hybrid 层类型：`kind = ((i+1) % interval == 0) ? "attn" : "linear"` | 只支持"每 interval 层一个 full attention"这一种排布；交错/前置 attention 排布静默错 | 由每层是否含 SSM 张量推断层类型（有 `ssm_*` 即 linear），interval 仅作一致性校验 |
| A5 | `src/ir.cpp:248` | `m.ssm.conv_dim = key_dim*2 + inner` | 假设 conv1d 输入是 [q,k] 拼接（2×key_dim）+ v；换成 [q,v] 或带 gate 的排布即错 | 由 `ssm_conv1d.weight` 的实际 shape 推导 conv_dim |
| A6 | `src/runner.cpp:85-90` | rope 频率表按 `variant == "mla"` / 非 mla 两分支取 `rope_dim` 或 `key_length` | 只有两种位置编码结构；无 rope、mrope（多模态）、任意 partial 比例（非 qwen35 路径）无法表达 | rope 结构（是否 partial、维度、是否 interleaved）进 IR 字段 |
| A7 | `src/runner.cpp:599-604` | `ssm.variant != "none" && gguf` → 所有层交给 `load_q35_linear/attn`；否则走 llama 命名 | 唯一 hybrid 实现是 qwen3.5 族；其它 SSM 架构只能报"缺张量"，错误信息不可诊断 | 抽象层加载器接口：按 IR 的 layer kind 选实现，加载器注册制 |
| A8 | `src/runner.cpp:912-919` | `run_layer_body`：`is_linear`→q35 linear；`q35.attn_kind`→q35 attn；否则 llama 路径 | 所有非 qwen35 的 attention 变体（QK-norm、partial rope、SWA、softcap）静默落到 llama 路径 → 数值错 | 同上；并在 validate() 中对"存在 `attn_q_norm` 但走 llama 路径"直接报错 |
| A9 | `src/runner.cpp:1522-1548`（host）与 `include/engine/kernels.h:2028-2071`（device） | 路由仅 `sigmoid_topk` / `softmax→topk→renorm` 两种；无 `routed_scaling_factor`、无 group-limited routing、无 `e_score_correction_bias`；两处独立实现 | DeepSeek-V3 / 分组路由模型静默算错；host/device 两份实现迟早漂移 | IR 增加 router 字段（gating func、scale、n_group、topk_group、bias），host 与 device 共用同一参数集 |
| A10 | `src/ir.cpp:155,159`（解析）；`src/runner.cpp`/`src/planner.cpp` 无使用 | `sliding_window`、`leading_dense_blocks` 进 IR 后无人读取 | 带 SWA 的模型静默按全上下文注意力运行：结果错 + KV cache 按 `max_seq` 全量分配 | 实现 SWA kv 窗口，或在 `validate()` 中对 `sliding_window > 0` 报 "unsupported" 并拒绝运行 |
| A11 | `src/runner.cpp:599,812-858` | safetensors 分支只认 `self_attn.*` / `mlp.*` 的 llama 命名；无 MLA/SSM；MoE 全专家 f32 上传 | 真实 HF 模型目录（DeepSeek-V2/V3、Qwen3.5）加载即失败；MoE 必 OOM | 显式 (container × arch variant) 支持矩阵；ST 路径接量化/分片 + expert placement |
| A12 | `src/qwen35.cpp:32-47,197-215` | 硬编码 GGUF 张量名：`ssm_conv1d`/`ssm_a`/`ssm_dt.bias`/`ssm_norm`/`attn_qkv`/`attn_gate`/`attn_q_norm`…；且要求这些张量**必须是量化格式**（否则 throw） | 换 converter 命名即失败；F16/BF16 的 qwen3.5 权重不能用 | 按 IR 张量 role/shape 查找，不按名字；允许 f16 权重 |

### B. 维度常量（18 项）

| # | 文件:行号 | 现状 | 风险 | 建议 |
|---|---|---|---|---|
| B1 | `src/runner.cpp:24`（使用点 `343-345`） | `constexpr size_t kDp4aMaxElems = 65536;` 写死 int8 激活 scratch 上限；注释："64K 覆盖受支持模型的所有 Q5_K n_in（最大 12288）" | 任何 `n_in > 32768` 的模型静默丢失 dp4a 路径（性能掉 ~2.3x，无日志）；这是**按四个已知模型标定的上限** | 由模型最大 `n_in` 在加载期计算并分配，或进 config |
| B2 | `src/runner.cpp:2465-2466,2501-2502` | dp4a 生效条件含 `static_cast<size_t>(w.n_in)*2 <= qx_i8_.size()` | 超限即静默退回 fp32 kernel，用户只看到变慢 | 同上；启动时打印一次 "dp4a disabled: n_in > X" |
| B3 | `src/runner.cpp:289-290` | 放置预算：`hd = key_length / n_head`；`kv = n_layer * n_head_kv * hd * max_seq * 2 * 4`。无 MLA 分支、无 `full_attention_interval` 分支 | `key_length` 是 per-head dim，再除 `n_head` → KV 低估 1~2 个数量级；MLA 模型 kv 估算完全错 → 过度 GPU 放置 → 0 MiB free / 驱动分页 | 直接调用 `ir_.kv_bytes_per_token()`（`src/ir.cpp:121-131` 已正确），runner 与 planner 共用一份 |
| B4 | `src/runner.cpp:363` | `logits_ = rows * vocab_ * 4`：按 `max_seq × vocab` 分配，且每个 token 都算全词表 | 248k vocab 模型白占 1 GiB+ 显存；`last_row_only` 只省下载不省计算/分配 | 只分配输出行；prefill 非末行用分块 lm_head |
| B5 | `src/runner.cpp:378` | `q35_qfull_ = rows * (n_head_ * head_dim_ * 2)` | 假设 q+gate 融合输出恰为 2×q 维度（qwen3-next 约定） | 用 `attn_q.weight` 的实际 n_out 推导 |
| B6 | `src/runner.cpp:1364-1365` | CPU expert slab：`max(inter,hidden)^2 / 8 + 4096` 字节 | 用"1/8 字节/元素"猜最大单 expert 张量；实际 Q4_K=144/256≈0.56 B/elem，是 1/8 的 4.5 倍 → 分配不足/越界隐患 | 由 expert 的 type + block_bytes 精确计算 |
| B7 | `src/runner.cpp:2336-2339` | f32 staging slab 上限 `cap = 8u << 20`（32 MiB） | 实现细节上限，大张量走分块；不限制泛化 | 保留（P2） |
| B8 | `src/ir.cpp:162-164` | `key_length == 0 && n_head > 0` → 静默用 `hidden / n_head` | 只对 head_dim==hidden/n_head 的模型成立；MLA/显式 head_dim 模型静默错 | 缺失时报错或要求 config 显式确认 |
| B9 | `src/runner.cpp:81` | `if (inter_ <= 0) inter_ = hidden_ * 4;` | 缺 `feed_forward_length` 时静默用 4x（现代模型多为 2.67x/3x） | 报错 |
| B10 | `include/engine/runner.h:20,202` | 构造函数默认 `max_seq = 256`，成员 `max_seq_ = 256` | 默认上下文 256；`forward()` 超过直接抛 "bad token count"（`src/runner.cpp:1281`） | 默认取 `ctx_train`，上限由 config 给 |
| B11 | `include/engine/runner.h:204-205` | `rope_base_ = 10000.0f`、`eps_ = 1e-5f` 成员默认值 | 缺 metadata 时静默退回 llama1 默认值 | 缺失告警 |
| B12 | `src/runner.cpp:58` | eps 只读 `<arch>.attention.layer_norm_rms_epsilon`，缺省 1e-5 | 不同 arch 的 key 名不同（`layer_norm_epsilon` 等）→ 静默 1e-5 | 多 key 回退 + 缺失告警 |
| B13 | `src/qwen35.cpp:21` | `constexpr float kQ35Eps = 1e-6f;`，用于所有 q35 RMSNorm（`388,447`） | qwen35 的 eps 完全不读 metadata；换版本静默错 | 从 GGUF 读 `attention.layer_norm_rms_epsilon` |
| B14 | `src/qwen35.cpp:427` | `rope_dim = ir_.attn.rope_dim > 0 ? ... : 64;`（文件头注释也写死"first 64 dims"） | 部分旋转维度写死 64 兜底 | metadata 缺失即报错 |
| B15 | `src/qwen35.cpp:299` | `rep = nh / nkh`（整数除法，无整除校验） | `n_value_head` 不整除 `n_head` 时静默截断，attention 覆盖错 KV 头 | `validate()` 增加整除检查 |
| B16 | `include/engine/planner.h:37-39`（及 `25`） | planner 默认 `ctx = 8192`、`kv_bpe = 2`、`vram_total = 12ull << 30`（12 GiB） | 12 GiB 是 RTX 3060 的常量；换卡即错 | 运行时查询 `cuda::mem_info().total_bytes`，ctx 取 metadata |
| B17 | `src/ir.cpp:492,496` | safetensors：`head_dim` 缺省 `hidden / n_head`；`rope_dim = partial_rotary_factor * head_dim`，`partial_rotary_factor` 缺省 1.0 | 缺字段静默假设"全头 rope / 常规 head_dim" | 缺失告警；`partial_rotary_factor` 进 IR 显式字段 |
| B18 | `src/runner.cpp:290` | KV 字节数写死 `* 2 * 4`（K、V 各 f32） | 无法用 fp16 KV；与 planner 的 `kv_bpe` 参数脱节 | KV dtype 进 config，runner/planner 共用 |

### C. 调优常量（21 项）

| # | 文件:行号 | 现状 | 风险 | 建议 |
|---|---|---|---|---|
| C1 | `src/runner.cpp:309-311` | MoE 放置安全余量默认 `2560ull << 20`（2.5 GiB），`ENGINE_VRAM_MARGIN_MB` 可覆盖；注释记录了在 Coder 上 768 MiB→2.5 GiB 的调参史 | 单机调参结果当默认值：小模型浪费 2.5 GiB（少放专家），大显存卡又可能不够 | 改成"按 (层数, 专家字节, KV) 比例的自动余量 + 标定命令"；默认值仅作 fallback |
| C2 | `src/runner.cpp:291` | `reserve = kv + (768ull << 20)`：workspace 常量 768 MiB | 同上；与 planner 的 `DEFAULT_WORKSPACE = 256 MiB`（`planner.h:21`）**互相矛盾**，两处不同数字 | 统一为 config 一项 |
| C3 | `src/runner.cpp:2428-2432` | `row_thr` 默认 2048（`ENGINE_ROW_THRESHOLD`）；`long_rows = n_in >= 2048 && n_out >= row_thr`（又一个写死 2048） | 阈值按 9B/0.5B 标定；不同 GPU 的 SM 数/HBM 带宽下最优阈值不同 | 标定表 or 运行时按 SMs 归一化 |
| C4 | `src/runner.cpp:2438-2441` | `q6k_x4_threshold` 默认 32768（`ENGINE_Q6K_X4_MIN`） | 同上（"<32K 行时 1-row 更快"来自本机实测） | 同上 |
| C5 | `src/runner.cpp:2503-2513` | Q6_K dp4a split-K：`n_out <= 4096 && sb_count >= 32 && %4==0` → split 4；`>=16 && %2==0` → 2；否则 1 | 三个魔数（4096/32/16）+ 整除条件，第 3 个 GPU 上很可能不再最优 | 进标定表；或按 grid 波次（waves）推导 |
| C6 | `src/runner.cpp:2481-2482` | Q5_K dp4a 共享内存 staging 条件 `shx = 3*n_in; shx <= 12*1024` | 12 KiB 是"共享内存不伤占用率"的经验拐点；n_in 阈值实为 4096 | 由 `cuda::device_info()` 的 shared mem/SM 与占用率公式推导 |
| C7 | `src/runner.cpp:2532-2535` | tiny-SK：`Q8_0 && n_out <= 128 && n_in >= 512`，`ksplit = n_out<=32?8:(<=64?4:2)`，`(n_in/32)%ksplit==0` | 6 个魔数专为"32×4096 SSM 投影"这类形状 | 抽象为"grid 太小则 split-K"的通用规则 |
| C8 | `src/runner.cpp:2546-2551` | `n_out <= 256` 的 Q5_0/Q4_K 走 k2（2 warp/行）kernel | 256 魔数；更宽的门控投影无法受益 | 同上 |
| C9 | `src/runner.cpp:2610-2620` | x4 split-K 默认关（`ENGINE_SPLITK` 才开）；开启后 `blocks_now>=256→1、>=128→2、否则 4`，`n_out<256→1` | 阈值绑本机 SM 数；默认关闭说明收益不稳定 | 由 SM 数推导 `ksplit`，或自动 A/B |
| C10 | `src/runner.cpp:1077`；`src/qwen35.cpp:492` | `nwarp = ctx_len > 384 ? 32 : (ctx_len > 96 ? 16 : 8)`（**同一行逻辑复制两份**） | 384/96 绑本机带宽/占用率；两份复制易漂移 | 提为函数（含 GPU 参数）；短期至少去重 |
| C11 | `include/engine/planner.h:14-21` | 带宽常量：`BW_PCIE_H2D_GBPS=12.0`、`D2H=11.0`、`RAM_COPY=22.5`、`CPU_GEMV=4.4`、`HBM=360.0`、`T_COPY_LAT_US=16.6`；`DEFAULT_VRAM_RESERVE=1 GiB`、`DEFAULT_WORKSPACE=256 MiB`；注释明说"P0 本机测量值，不要换成 spec 值" | 换机器后 planner 的 `proj_decode_*_ms` 与放置建议全错；且与 runner 的 2560/768 MiB 不一致 | 加 `engine calibrate` 子命令写标定文件；planner 读标定值 |
| C12 | `src/runner.cpp:2262` | fp16 TN 镜像预留 `512ull << 20` 头寸 | 魔数；显存小时直接跳过优化，无日志 | 按空闲显存比例 |
| C13 | `src/runner.cpp:268` | prefill f32 镜像门限：`dense_bytes * 4 + (1ull << 30) > free` 则关 | 1 GiB 头寸魔数；f32 镜像本身 = 4x dense 权重（大模型根本不会开） | 用"镜像 + workspace"的显式预算 |
| C14 | `src/runner.cpp:1397,1415` | 小 batch 走自研 GEMV：`n_tokens_cur_ <= 4`，否则 cuBLAS | 4 是经验拐点（cuBLAS 小 batch 效率差） | 由矩阵形状/GPU 推导，进配置 |
| C15 | `src/runner.cpp:1481,1817` | 设备端路由/融合 MoE 仅 `n_tokens <= 8` | 8 是"槽位少时收益大"的经验值；prefill 稍长即退回 host 路由 | 进配置；结合 D1 的 256 上限一起修 |
| C16 | `src/runner.cpp:2426` | 只有 `n_tokens == 1` 才用自研 GEMV kernel（且排除 Q5_1） | decode 专用；prefill 全走 cuBLAS f32/量化 GEMM（opt-in） | 长期实现"任意 M 的 quant GEMM"（自研 GEMM 目前 opt-in，见 C18） |
| C17 | `src/runner.cpp:2294-2297` | block-major 重排默认白名单只有 `Q8_0`；Q6_K 注释写"parity 测试不过，默认关闭" | 已知内核缺陷被静默绕过；Q5_0 要显式开 | 修复 Q6_K bm 或移除该路径；白名单进配置 |
| C18 | `src/runner.cpp:2643-2652` | 自研 quant GEMM：`ENGINE_GEMM` 才启用，且仅 Q8_0/Q4_K/Q5_K（type 3/0/1） | prefill 主力路径默认是 cuBLAS f32 → 量化权重在 prefill 被反量化，带宽/显存双输 | 扩到全部受支持 quant，默认开启并加数值回归 |
| C19 | `src/runner.cpp:2573-2575,2588-2590` | 激活量化缓冲按需分配并 `+4096` / `+256` 字节余量 | 实现细节，不影响泛化 | 保留（P2） |
| C20 | `src/cpu_matvec.cpp:612` | CPU 多线程条件：线程数 ≤1 或 `n_out < 64` 时退回单线程 | 64 行是经验值；小模型 CPU 层可能偏保守 | 进配置 |
| C21 | `src/runner.cpp:1078-1080`；`include/engine/kernels.h:1473,1486` | attention kernel 动态共享内存 = `(max_seq + 32 + nwarp*head_dim) * 4` 字节；`pr[32]` 隐含 `nwarp<=32` | 共享内存随 `max_seq_` 线性增长，**默认 48 KiB 上限**下 ctx 只能到 ~1.1 万 token；超限是 CUresult 报错而非友好提示 → 上下文能力被 kernel 形状静默封顶 | flash-decoding / 分块 KV kernel；短期在启动时检查 `48 KiB` 并给出明确上限 |

### D. 后端 / 环境硬编码（16 项）

| # | 文件:行号 | 现状 | 风险 | 建议 |
|---|---|---|---|---|
| D1 | `include/engine/kernels.h:2028-2036` | `moe_route`：`float pr[256]`；`if (ne > 256) return;` | >256 专家时内核静默返回 → 未初始化/残留的 slot 被 gate/up/down kernel 使用；调用侧（`src/runner.cpp:1481,1817`）不检查 `ne<=256` | 调用侧强制 `ne<=256` 走 host 路由，>256 直接报错或升 kernel 模板；`pr` 改动态 shared/上限检查 |
| D2 | `include/engine/kernels.h:2002,2019`（另 `254,291,330,579,618,963` 同类） | `row = blockIdx.x * 8 + warp`：**8 行/块**写死；`src/runner.cpp:2632-2637` 注释承认"launch shape 必须是 8 行/256 线程" | 任何想把 kernel 改成 4 行/16 行（不同 quant 的 x4 变体）都必须同时改 launch 侧，两处靠注释维系 | 把行数作为 kernel 描述字段（`rows_per_block`）随 kernel 名一起注册，launch 统一查表 |
| D3 | `src/cuda.cpp:100-104` | NVRTC 默认路径 `<PATH>\py312\Lib\site-packages\nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll`（可用 `DSH_NVRTC_PATH` 覆盖） | 换机器/换 Python 环境即失败；默认值与作者本机绑定 | 自动探测：`nvidia` 包目录搜索 → PATH → CUDA_HOME；失败信息给搜索路径列表 |
| D4 | `src/cublas.cpp:51-55` | cuBLAS 默认目录 `<PATH>\py312\...\nvidia\cublas\bin`（可用 `DSH_CUBLAS_DIR` 覆盖） | 同上 | 同上 |
| D5 | `src/cuda.cpp:237-243` | `cuDeviceGet(&dev, 0)` + `CtxCreate`：**永远 0 号卡、单 context** | 多卡机器无法选卡；无 `ENGINE_DEVICE`/`CUDA_VISIBLE_DEVICES` 语义（新增 config 层已在补 `device` 键，但代码未接入） | 设备索引配置化 + 枚举能力（名称/显存/cc）；多卡需 tensor/层切分策略 |
| D6 | `src/cuda.cpp:261` | `g_info.runtime_version = 12090;` 伪造 CUDA runtime 版本 | 版本相关判断/日志不可信；未来按 runtime 版本分支会出错 | 用 `nvrtcVersion()` 或 `cuDriverGetVersion` 真值 |
| D7 | `src/cuda.cpp:208-211` | 二次加载 fallback 仍写死 `nvrtc64_120_0.dll`；错误信息也固定该名 | CUDA 12.x 其它 minor / CUDA 13 直接失败 | 按 glob 搜索 `nvrtc64_*.dll` |
| D8 | `src/cublas.cpp:68-70` | 写死 `cublas64_12.dll` | 同上 | 按 `cublas64_*.dll` 搜索 |
| D9 | `src/cpu_matvec.cpp:556-561` | 线程池 `hardware_concurrency() - 1`，**无配置项**；`609-612` 打印一次 | 无法限制 CPU 占用（多实例/与 GPU 抢占）；超线程机器可能 -1 不利 | `threads` 配置（新 config 层已有键）+ 亲和性选项 |
| D10 | `build.sh:9` | `PY="${PY:-python}"` | 构建脚本硬编码本机解释器 | 探测 `python`/`python3`/`py -3` |
| D11 | `include/engine/kernels.h:1137-1156` | `gd_conv_state`：`if (K != 4) return;`（"K is always 4"），且 `1144` 的状态移位只处理 K=4 的 3 个 tap | conv_kernel≠4 的 hybrid 模型状态永不正确，且无报错 | kernel 支持任意 K（循环 + 状态拷贝），或加载期直接拒绝 K≠4 |
| D12 | `src/qwen35.cpp:106` | `if (gpu_moe_layers_ >= 0 && std::getenv("ENGINE_GPU_EXPERTS") == nullptr)`：层粒度的 GPU/CPU 专家放置**依赖环境变量是否设置** | 环境变量优先级混乱：一旦设了 `ENGINE_GPU_EXPERTS` 就绕过 runner 的自动放置（`src/runner.cpp:272-336`），两套放置策略并存 | 单一放置策略对象；显式优先级（CLI > config > env > 自动） |
| D13 | `src/qwen35.cpp:45` | `w.q35.a_log = upload(tensor_f32(p + "ssm_a"));` 直接当 `log(A)` 用；`359` `ENGINE_Q35_RAW_A` 才按 `-A` 解释 | converter 差异靠实验开关兜；错了解释就是数值全错而不报错 | 从 metadata 判定 A 的参数化（或校验值域），不用实验开关 |
| D14 | `include/engine/kernels.h:2145-2152`（调用 `src/runner.cpp:2230-2240`） | `clock_warmer`：靠空转 kernel 防 WDDM 空闲降频，`ENGINE_KEEP_GPU_BUSY` opt-in | Windows/WDDM 专属补丁；在 Linux/正常调度下是纯浪费 | 平台条件编译 + 默认关闭；定位为已知性能 workaround 而非"引擎能力" |
| D15 | `src/cuda.cpp:128-130,200-206` | 全程 `LoadLibraryA`/`SetDllDirectoryA`/`__stdcall` 函数指针：Windows + 动态 DLL 专用；无 Linux/dlopen 路径 | 无法在 Linux（vLLM 的主战场）编译运行；"通用引擎"门槛 | 后端抽象：CUDA 驱动 API 层 + 平台加载器（dlopen/LoadLibrary） |
| D16 | `include/engine/kernels.h:1388-1411,1453`（`__dp4a`） | 含 `__dp4a` 的 kernel 无编译期能力检查：NVRTC 目标 = 设备 cc（`src/cuda.cpp:282-283`），cc<6.1 时编译失败，错误是 NVRTC 日志 | 老卡（Pascal/Kepler）上用户体验是"PTX 编译失败"而非"不支持 dp4a" | 启动时检查 cc 并禁用相关 kernel（或给出明确错误） |

### E. 入口 / CLI 限制（16 项）

| # | 文件:行号 | 现状 | 风险 | 建议 |
|---|---|---|---|---|
| E1 | `src/main.cpp:302-308` | CLI 只有 `info` / `ir` / `plan` 三个子命令；无 `run` / `serve` / `bench` | 真正的"推理引擎"入口其实在 `tests/test_gen.cpp`；产品化无从谈起 | 增加 `run`（文本进文本出）与 `serve`（HTTP/OpenAI 兼容） |
| E2 | `src/main.cpp:320-321` | CLI 默认 `ctx = 8192`、`vram = 12.0` GiB | 默认值绑本机；`plan` 结果对别的显卡误导 | 取 metadata `context_length` + 实际显存 |
| E3 | `src/main.cpp:323-340` | 选项解析只有 `--out/--json/--no-tensors/--force/--ctx/--kv-bpe/--vram-gib`；未知选项报错退出 | 无采样参数、无设备/线程/专家放置参数 | 统一 CLI 表（新 config 层已有 48 个键，需接到 CLI） |
| E4 | `include/engine/runner.h:20-30` | 公共 API：`DenseRunner(path, max_seq, container)` + `forward(tokens, start_pos, last_row_only)`；无配置对象、无采样器、无会话 | 调用方要改行为只能改环境变量；无法多序列/多会话 | `EngineConfig` + `Session`（KV 归属 session）+ `Sampler` |
| E5 | `tests/test_gen.cpp:14-17` | 唯一生成入口用法：`test_gen <model.gguf> <ids> <n_generate>`，**只有 model/ids/n 三个参数** | 无法选 ctx、采样、stop、batch；CI/对拍只能靠环境变量 | 提升为正式 CLI 子命令，参数化 |
| E6 | `tests/test_gen.cpp:33-35` | 容器探测靠文件名子串 `find(".gguf") != npos`（大小写敏感、相对/绝对路径都靠猜） | 目录型 safetensors 与 `.GGUF`、无扩展名文件都会误判 → 走错 loader | 用"是目录 + config.json"判定，或魔数嗅探 |
| E7 | `tests/test_gen.cpp:32` | `max_seq = ids.size() + n_gen + 16` | 任意 magic 32/16 余量；长 prompt 静默截断风险 | 由 ctx 配置 |
| E8 | `tests/test_gen.cpp:59-60` | warmup 固定喂 2 个 token（`forward({ids[0], ids[1]})`） | 对 SSM 模型会污染 recurrent state（依赖后续 reset 逻辑）；对 batch 路径假定 2 即可 | 显式 warmup 策略/可关 |
| E9 | `tests/test_gen.cpp:80-96,97-111` | `ENGINE_TARGET` 只查 top-200；`ENGINE_TOPK` 固定打印 top-5 | 对拍诊断能力写死 | 参数化 |
| E10 | `tests/test_gen.cpp:112-122` | 生成是纯 greedy argmax；无 temperature/top-p/repetition penalty；**无 EOS 停止**（无条件跑满 n_gen） | 不是可用采样器；对齐参考实现时需依赖外部脚本 | 采样器模块 + EOS/stop 词表 |
| E11 | `src/ir.cpp:183-187`（仅 metadata）；全仓库无 tokenizer 实现 | `tokenizer.ggml.*` 只被读取和打印，**没有任何 encode/decode 代码** | 引擎只能吃 token id → 不是"能对话的引擎"；chat template 只记录布尔 | 实现 GGUF BPE/SentencePiece tokenizer + chat template 渲染 |
| E12 | 同 E11 + `src/ir.cpp:187,519-520` | 无采样、无 chat template 应用、无 stop strings、无 tool-call 协议 | 与 vLLM 类引擎的 API 面差距最大的一块 | 同 E10/E11 |
| E13 | `src/runner.cpp:1319`；`src/runner.cpp:1281` | CUDA Graph 只在 `n == 1` 捕获；`forward()` 单序列、有界 `max_seq`；无 paged KV / 连续批处理 / 前缀缓存 | 吞吐上不了"引擎"级别；多请求要串行 | paged KV + continuous batching（大改，明确排期） |
| E14 | `src/runner.cpp:2428-2431,2438-2441,2464,2495-2500,1440-1443` | 129 处 `getenv()`、87 个变量名；大量用 `static const` 缓存（**首次调用后不可再改**）；无配置文件、无优先级定义；`include/engine/runner.h:303` 的 `dev_route_default_` 从未被置 true（死开关） | 行为不可复现（脚本先 set 什么就是什么）；库调用方无法配置；死开关说明"默认值"与"实际默认"曾脱节 | 统一 config 层（正在新增）+ 启动时打印生效配置 + 删除死开关 |
| E15 | `src/ir.cpp:195`；`src/runner.cpp:537`；`src/runner.cpp:496`；`src/ir.cpp:463-469` | 错误信息缺"支持清单"：`"unsupported tensor type for X"`、`"unsupported gguf quant type X for Y"`、`"unsupported dtype X"`；其中 ST 的报错还宣称支持 FP8（`ir.cpp:467`），但 `tensor_f32` 只认 F32/BF16/F16（`runner.cpp:471-496`）→ FP8 必然失败 | 用户无法自诊断；错误信息与实现不一致 | 统一列出受支持类型；删掉误导性 FP8 文案或实现 FP8 |
| E16 | `src/runner.cpp:1281-1283`；`src/cuda.cpp:368` | 越界/显存不足只有通用异常（`"forward: bad token count"`、`"cuMemAlloc failed"`），无上下文（需要多少/可用多少/建议降低 ctx 或 n-cpu-moe） | 失败不可恢复、不可指导 | 结构化错误 + 建议动作；显存不足时回退 CPU 专家 |

---

## 3. 风险分级

### P0（6 项，必须最先修）

| 条目 | 位置 | 理由 |
|---|---|---|
| D1 | `kernels.h:2028-2036` | 静默错误路由（>256 专家），错得无感知；且是栈数组上限，属"越界即错"级别 |
| D11 | `kernels.h:1137-1156` | 静默不更新卷积状态（K≠4），hybrid 模型从第 2 个 chunk 起全错 |
| A10 | `ir.cpp:155,159` | SWA 元数据解析但不实现 → 静默全注意力 + KV 超算 |
| A11 | `runner.cpp:599,812-858` | safetensors 能力矩阵缺失 + 全专家 f32 上传 → 真实 ST 模型不可用（OOM） |
| A9 | `runner.cpp:1522-1548` + `kernels.h:2028-2071` | 路由语义写死（无 scaling/group/bias），host/device 两份实现 |
| B3 | `runner.cpp:289-290` | KV 估算用错语义 → 放置预算系统性低估 → 隐性 OOM/驱动分页，性能骤降且难归因 |

判据：**P0 = 静默给出错误数值，或在最常见的真实模型上必然失败**（不是"跑得慢"）。

### P1（40 项，影响"能跑什么模型 / 能不能换机器"）

- A 类：A1、A2、A3、A7、A8、A12 —— 决定"除这四个模型外还能不能加载/算对"。
- B 类：B1、B2、B4、B5、B6、B8、B10、B13、B14、B15、B16 —— 维度/预算常量，换模型即出问题。
- C 类：C1、C2、C3、C5、C10、C11、C21 —— 机器相关调优与 kernel 形状封顶（C21 限制上下文）。
- D 类：D2、D3、D4、D5、D7、D8、D12、D13、D15 —— 可移植性与环境硬编码（D15 Windows-only 是"通用引擎"的硬门槛）。
- E 类：E1、E5、E10、E11、E13、E14、E15 —— 入口/API 能力（无 run、无 tokenizer、无采样、无批处理、无配置、错误不可诊断）。

### P2（37 项，可排期）

- A4、A5、A6（排布/编码的第三、第四种变体）
- B7、B9、B11、B12、B17、B18（有兜底、影响面小）
- C4、C6、C7、C8、C9、C12、C13、C14、C15、C16、C17、C18、C19、C20（性能阈值/细节；C17 的 Q6_K bm parity 缺陷应单独建 issue）
- D6、D9、D10、D14、D16
- E2、E3、E4、E6、E7、E8、E9、E12、E16

> 分级原则：**静默错 > 必然失败 > 换机退化 > 可用性/性能**。P1 中 D3/D4/D7/D8 属于"换一台机器就 100% 起不来"，优先级实际上高于一般 P1，应在阶段 1 一并处理。

---

## 4. 修复建议优先级

### 阶段 0：堵住静默错误（1–2 天，最高收益/最低成本）

1. `ne > 256`：调用侧（`runner.cpp:1481,1817`）加硬检查，>256 时强制 host 路由；kernel 内 `pr[256]` 改为"超限即写错误标志并 return"或 `extern __shared__` 动态数组。
2. `K != 4`：`gd_conv_state` 改为通用循环；短期在 `ir.cpp` 的 SSM 检测处要求 `conv_kernel == 4` 并在 `validate()` 报错。
3. SWA/leading_dense：`validate()` 对 `sliding_window > 0 || leading_dense_blocks > 0` 报"unsupported"，先拒绝而不是算错。
4. KV 预算：删除 `runner.cpp:289-290` 的本地公式，改调 `ir::kv_bytes_per_token()`；`runner.h` 增加 `kv_bpe`。
5. 路由：IR 增加 `router_scale`、`n_group/topk_group`、`score_bias` 字段（至少先支持 `routed_scaling_factor`，DeepSeek-V3 类模型即可对拍），host/device 共用。

### 阶段 1：把"只能跑四个模型"变成"能跑一类模型"（1–2 周）

1. 架构能力矩阵：`(container × attn.variant × ssm.variant × moe)` → 加载器/执行器注册表；无实现时**入口期**报清晰错误（含缺哪个张量/哪个 variant）。
2. 移除 arch 字符串特判：A1/A2/A3 的全部逻辑改由 metadata + 张量统计 + 显式 IR 字段驱动。
3. 动态维度：B1/B2（dp4a 上限）、B4（logits 分配）、B5（qfull）、B6（expert slab）按实际 n_in/n_out 计算。
4. kernel 形状契约：`rows_per_block`、`nwarps`、共享内存需求、量化类型 id 全部写进 kernel 注册表（`kernel_select.*` 正在新增，应承接这部分），消除 D2/C10 的"注释维系"。
5. 上下文能力：C21 的 48 KiB 共享内存上限要先**显式检查并报错**，再排期 flash-decoding。

### 阶段 2：去掉机器常量（1 周，可与阶段 1 并行）

1. 新增 `engine calibrate` 子命令：测 HBM 带宽、PCIe H2D/D2H、CPU GEMV、拷贝延迟 → 写标定文件；planner 读标定（替换 C11 的 7 个常量），并统一 C1/C2 的 2560/768 与 planner 的 1 GiB/256 MiB 两组矛盾值。
2. 自动余量：VRAM margin 用 `kv + workspace + 专家字节的百分比` 推导，不用绝对 MB。
3. 设备/线程：`device`、`threads` 接入实际代码（config 层已有键），默认自动。
4. 环境可移植：DLL/python 目录自动探测（D3/D4/D7/D8），Windows 专属代码隔离（D14/D15）。

### 阶段 3：入口产品化（2–4 周）

1. `run`/`serve` 子命令 + `EngineConfig`/`Session`/`Sampler` API（E1/E3/E4）。
2. tokenizer + chat template（E11/E12）——没有它就没有"文本进文本出"。
3. paged KV + continuous batching + 前缀缓存（E13），decode CUDA Graph 支持 batch>1。
4. test_gen 的 5 个环境变量开关（E8/E9）升为正式参数，CI 脚本随之切换。

### 验收与回归建议

- 数值：沿用 `tools/verify.sh`（51 项）+ `pyref/diff_ir.py` / `diff_plan.py`；每修一项 P0 加一个夹具：
  - >256 专家的合成 GGUF（验证"拒绝或走 host 路由"，不再静默）；
  - `ssm.conv_kernel=3` 的合成模型（验证通用卷积状态）；
  - 带 `sliding_window` 的合成模型（验证拒绝/实现）；
  - 带 `routed_scaling_factor` 的合成 MoE（验证权重缩放）。
- 泛化：`tools/smoke_models.sh` 保持四模型；新增至少两个"非本机标定"的架构（如 Gemma3 的 QK-norm + SWA、Mixtral 的 softmax MoE）。
- 每个阶段结束跑一次 `grep -n "getenv" | wc -l`，目标是 129 → <40（调试类保留），并删除所有"首次调用缓存"的静态读取。

---

## 附录 A：最重要的 28 个调优环境变量（快照口径）

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| `ENGINE_VRAM_MARGIN_MB` | 2560 | MoE 放置安全余量（MiB） | `src/runner.cpp:310` |
| `ENGINE_VRAM_BUDGET_MB` | 当前空闲 | 放置预算上限（MiB） | `src/runner.cpp:283` |
| `ENGINE_GPU_EXPERTS` | -1（按层自动） | 每层上 GPU 的专家数；设置后绕过自动放置 | `src/runner.cpp:36`、`src/qwen35.cpp:106` |
| `ENGINE_PREFILL_F32` | 1（MoE 下自动 0） | prefill 保留 f32 镜像 | `src/runner.cpp:258,267` |
| `ENGINE_NO_GRAPH` | 0 | 关 decode CUDA Graph | `src/runner.cpp:35` |
| `ENGINE_LAYER_GRAPH` | 0 | 逐层 graph 捕获（隐含开启设备端路由） | `src/runner.cpp:882,1450` |
| `ENGINE_DEV_ROUTE` / `ENGINE_NO_DEV_ROUTE` | 关 / 关 | 设备端路由开关（`dev_route_default_` 从未置 true） | `src/runner.cpp:1477-1479,1813` |
| `ENGINE_ROUTER` | 取 metadata | 覆盖 `router_kind` | `src/runner.cpp:64` |
| `ENGINE_NORM_TOPK` | 取 metadata | 覆盖 `norm_topk` | `src/runner.cpp:65` |
| `ENGINE_NORM_PLUS1` | -1（自动探测） | q/k norm 是否 +1 | `src/runner.cpp:443` |
| `ENGINE_EXPERTS_TOPK` | 0 | 截断 top-k（调试） | `src/runner.cpp:1441` |
| `ENGINE_NO_MOE_FUSED` | 0 | 关融合 MoE kernel | `src/runner.cpp:1726` |
| `ENGINE_NO_SHARED` | 0 | 关 shared expert | `src/runner.cpp:1589`、`src/qwen35.cpp:68` |
| `ENGINE_ROW_THRESHOLD` | 2048 | x4 GEMV 的行数阈值 | `src/runner.cpp:2429` |
| `ENGINE_Q6K_X4_MIN` | 32768 | Q6_K 用 x4 的最小 n_out | `src/runner.cpp:2439` |
| `ENGINE_Q6K_SPLIT` | 0（自动） | Q6_K dp4a split-K 固定值 | `src/runner.cpp:2498` |
| `ENGINE_SPLITK` | 关 | x4 kernel 的 split-K 开关 | `src/runner.cpp:2614` |
| `ENGINE_NO_TINY_SK` | 0 | 关小 n_out 的 split-K | `src/runner.cpp:2533` |
| `ENGINE_NO_DP4A_GEMV` / `ENGINE_NO_DP4A_Q6K` | 0 | 关 int8 dp4a 路径 | `src/runner.cpp:2464,2496` |
| `ENGINE_GEMM` | 0 | 自研量化 GEMM（opt-in） | `src/runner.cpp:2642` |
| `ENGINE_HALF_T` | 0 | fp16 TN GEMM（实测更慢，opt-in） | `src/runner.cpp:2254` |
| `ENGINE_BM_TYPES` | `Q8_0` | block-major 重排白名单 | `src/runner.cpp:2294` |
| `ENGINE_MLA_Q_LAYOUT` / `ENGINE_MLA_KV_LAYOUT` | 空 | MLA q/kv 内存布局实验开关 | `src/runner.cpp:37,42` |
| `ENGINE_NO_YARN` | 0 | 关 YaRN 缩放 | `src/runner.cpp:97` |
| `ENGINE_LAYERS` | 0 | 只跑前 N 层（调试） | `src/runner.cpp:1186` |
| `ENGINE_KEEP_GPU_BUSY` | 0 | WDDM 空闲降频对抗（空转 kernel） | `src/runner.cpp:2231` |
| `ENGINE_Q35_RAW_A` | 0 | `ssm_a` 按 `-A` 解释的实验开关 | `src/qwen35.cpp:359` |
| `DSH_NVRTC_PATH` / `DSH_CUBLAS_DIR` | `<PATH>\py312\...` | NVRTC / cuBLAS DLL 目录 | `src/cuda.cpp:101`、`src/cublas.cpp:52` |

**分组说明**：其余约 60 个变量是调试/导出类，不改变主路径语义，可分为三组：
1. 诊断打印：`RUNNER_DEBUG`、`ENGINE_PROFILE`、`ENGINE_KPROF`、`ENGINE_DBG_*`（VRAM/LOAD/LOAD2/DEVROUTE/DEVWHY/BIGGEMV/MATVEC/CPULAYER/MOEHASH/LAYERMS/LG/LG2/SCRATCH/Q35LOAD/VOCT…）。
2. dump 对拍：`ENGINE_DUMP*`（13 处，含 `_LAYER`/`_MLA`/`_Q35`/`_MOE_SPLIT`/`_EVERY`/`_ROPE`）、`ENGINE_LOGITS`。
3. 内核实验：`ENGINE_DP4A`、`ENGINE_NO_GEMV`、`ENGINE_NO_BM`、`ENGINE_NO_AVX2`、`ENGINE_SH_SWAP_GU`、`ENGINE_MOE_SWAP_GU`、`ENGINE_SKIP_MLA_ATTN`、`ENGINE_SKIP_LMHEAD`、`ENGINE_MOE_FIXED_ROUTING`、`ENGINE_MLA_ROPE_NEOX`、`ENGINE_LG_RECAPTURE`、`ENGINE_LG_DENSE_ONLY`、`PROBE_DEV_KERNEL`。

---

## 附录 B：复核命令

```bash
# 1) 行号定位（示例：VRAM margin / 255 专家 / KV 估算）
grep -n "margin = 2560ull << 20" src/runner.cpp

# 2) 关键缺陷直接定位
grep -n "ne > 256" include/engine/kernels.h            # D1
grep -n "if (K != 4) return" include/engine/kernels.h   # D11
grep -n "sliding_window" src/ir.cpp src/runner.cpp      # A10：ir 解析，runner 无使用
grep -n "key_length /" src/runner.cpp   # B3：确认 hd = key_length / n_head
grep -n "sigmoid_topk\|router_kind" src/runner.cpp include/engine/kernels.h  # A9
grep -n "hd = ir_.attn.n_head" src/runner.cpp

# 3) 计数口径
grep -rn "getenv" src include tests --include=*.cpp --include=*.h | wc -l          # 快照 129
grep -rhoE "(ENGINE|RUNNER|DSH|PROBE)_[A-Z0-9_]+" src include tests --include=*.cpp --include=*.h | sort -u | wc -l  # 快照 87

# 4) 量化类型白名单（三处不一致，见 A/备注）
grep -n "quant_kernel\|type_id\|gemv_typed" src/runner_internal.h src/runner.cpp
```

**附注（审计中额外发现的白名单漂移）**：量化类型白名单在四处各写一份且互不一致——
`src/runner_internal.h:27-36`（Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1）、
`src/runner.cpp:1377-1382`（Q4_K/Q5_K/Q6_K/Q8_0/Q5_0，**没有 Q5_1**）、
`src/runner.cpp:1470-1475` 与 `1731-1736`、`1819-1824`（Q4_K/Q5_K/Q6_K/Q8_0/**Q5_1**，**没有 Q5_0**）、
`src/cpu_matvec.cpp:24-30`（含 Q5_0 与 Q5_1）。
结果：同一个 Q5_1 张量在 GEMV 路径会掉到 f32 分支，在融合 MoE 路径却能走 device kernel；Q5_0 反之。属 D 类缺注册表的直接后果，建议随阶段 1 的 kernel 注册表一并删除。

---

## 附录 C：明确**不算**泛化缺陷的常量（避免误伤）

以下常量是实现细节或格式规范，不应进入"硬编码模型假设"整改清单：

- GGUF 量化格式常量：144/176/210/34/22/24 字节/块、32/256 元素/块（`src/gguf.cpp:13-25`、`runner_internal.h:27-36`）——格式定义，非模型假设。
- kernel 并行形状：256 线程/块、32 lane/warp、4 行/warp 的 x4 变体（如 `kernels.h:663,872`）——性能实现细节；但"8 行/块"因耦合 launch 侧（D2）单独计入。
- 数学常量：`1e-30f` 初值、`rsqrtf(D)` 缩放、YaRN 的 6.283185307179586（`src/runner.cpp:110`）、sigmoid/softmax 实现——算法定义。
- `kDp4aMaxElems` 之外的一次性小缓冲余量 `+4096`/`+256` 字节（C19）——实现细节，仅记录不列为风险。
- 调试阈值如 `w.n_out > 50000`（`src/runner.cpp:2423`，仅 `ENGINE_DBG_BIGGEMV` 打印用）——不影响主路径。
- IR `validate()` 中的通用检查（`src/ir.cpp:346-353`：元素数整除块、`n_head % n_head_kv`）——正确方向；缺的是 SSM/rope/head_dim 相关检查（已计入 B15/A10）。

---

## 附录 D：v1 通用化改造后的状态（2026-11-14 20:15 追加）

> 本审计是**重构前快照**。v1 改造（配置面 + 自适应 + 内核选择集中化 + 优雅失败）合入后，
> 各条目的现状更新如下；未列到的条目状态不变。

### D.1 已解决 / 明显缓解

| 审计范围 | 改造后现状 |
|---|---|
| C 类（调优常量，21 项） | 18 项收进 `src/config.cpp` 配置注册表并可通过 `--config`/CLI/环境变量设置：`vram_margin_mb`(默认 1200)、`vram_budget_mb`、`workspace_mb`、`row_threshold`、`q6k_x4_min`、`q6k_split`、`dp4a_gemv`、`dp4a_q6k`、`tiny_sk`、`splitk`、`dp4a`、`no_gemv`、`half`、`half_t`、`bm_types`、`no_bm`、`keep_gpu_busy`、`no_moe_fused`。其余 3 项是纯诊断开关（`ENGINE_PROFILE`/`ENGINE_KPROF`/`ENGINE_PIPE_PROF`），仍为 env |
| C 类 kernel 分派阈值 | 全部集中到 `src/kernel_select.cpp:choose_gemv()`；`--kernel-calibrate` 在进程内对真实形状微基准、写 `engine-kernel-cache.txt`，之后自动加载（形状兼容性校验拒绝误用） |
| C 类 VRAM 余量 | 不再用固定 2560 MiB 一刀切：默认 `vram_margin_mb=1200`，装载每层专家前用 `cuMemGetInfo` + 剩余分配量实测校验，不足自动减层并打印依据；装载后再校验一次 |
| D1-D4（DLL 路径/单 GPU/线程池） | 新增配置：`nvrtc_path`（env `DSH_NVRTC_PATH`）、`cublas_dir`（env `DSH_CUBLAS_DIR`）、`device`、`threads` |
| E 类（入口/CLI） | 新增 `engine run` 子命令；统一 CLI（`--model/--ctx/--threads/--gpu-layers/--vram-margin/--no-graph/...`）、`--config` 配置文件、`--print-config`、`--kernel-calibrate`；`test_gen` 保留旧位置参数并支持同一套选项；容器检测改为读 GGUF magic（不再依赖扩展名） |
| 4 处量化类型白名单不一致 | 新增 IR 校验统一白名单（F32/F16/BF16/Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1），不支持的量化类型在**校验阶段**报错并列出支持列表 |
| `moe_route` `ne>256` 静默（P0 #1） | **不再静默**：`ir::validate()` 在加载前报错 `expert_count > 256 ... not supported`（内核仍只有 256 上限，但不会算错） |
| 滑窗 `sliding_window` 未实现（P0 #3） | 不再静默：校验期打印 warning 说明"按全注意力计算、KV 偏大" |
| 未知架构/张量命名 | 新增一致性校验：embed/attn/ffn 角色识别、n_head×key_length、rope 维度、MLA/SSM 几何、MoE 层完整性、embedding 行数等；缺张量时错误附带"支持的模型族清单"（见 §D.3） |

### D.2 未解决（内核/架构级，超出配置面范围）

- `moe_route` 的 256 专家上限、`gd_conv_state` 的 `K != 4` 静默返回、路由缺 `routed_scaling_factor`/group routing/bias —— **内核数学/模型语义**，按约束 4（不动 dp4a 内核数学）与投入产出比未改；前者已由校验期报错兜底。
- safetensors MoE 路径仍把所有专家反量化为 f32 上传（仅小夹具可用），未实现真实 HF MoE 的分片/量化驻留。
- 架构分支仍按 IR variant 分三条（dense / MLA / hybrid SSM）；`deepseek3` 路由特判、`deepseek2` norm 零中心判定仍是 arch 字符串特判（已有数据驱动兜底与覆盖开关）。
- `src/runner.cpp` 放置估算里的 KV 字节仍沿用 `key_length/n_head` 的旧公式（`ir::kv_bytes_per_token()` 是正确的但未接入）；由于 v1 的逐层实测校验会在装载期纠正，该偏差只影响规划线的显示值，不影响最终放置正确性。
- tokenizer / 文本输入 / 采样策略仍不在引擎内（入口给 token id、贪心输出）。

### D.3 错误信息样例（`tools/make_bad_models.py` 生成的三类坏模型）

```
# 未知架构
error: GGUF metadata for arch 'mamba' is missing block_count. This engine understands
llama-style metadata keys ... for these families: llama-like dense (...), MLA (deepseek2 /
deepseek3), hybrid SSM (qwen35 / qwen35moe). ...

# 不支持的量化类型
VALIDATION : FAIL (1 errors)
  - unsupported quant type(s): Q4_0. This engine implements: F32, F16, BF16, Q4_K, Q5_K,
    Q6_K, Q8_0, Q5_0, Q5_1. Convert the model with llama.cpp's quantize tool ...

# 未知张量命名
  - no embedding tensor recognized (expected token_embd.weight for GGUF or
    model.embed_tokens.weight for safetensors): unsupported model layout or wrong file
```

验证命令：`bash tools/verify_generality.sh`。
