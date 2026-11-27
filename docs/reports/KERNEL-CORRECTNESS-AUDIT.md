# CUDA 内核正确性 / 未定义行为审计（v0.1.7）

审计对象：**v0.1.7 源码归档 `D:/v017_src/lingshi/`**（与 v0.1.7 tag 一致；主树 HEAD `e4f8eab` 只多 JSON 深度修复与归档工具）。
范围：`include/engine/kernels.h`（4302 行、103 个内核）+ `src/cuda.cpp`（855 行驱动/资源封装）+
`src/kernel_select.cpp`（内核选择合法性）+ 这些内核在 `src/runner.cpp` / `src/qwen35.cpp` 里的**启动参数**
（grid/block/动态 shared memory/参数类型）。只做**正确性与未定义行为**，不看性能。

方法：静态逐内核审查 + 启动点逐一比对（grid/block/smem 与内核实际下标上界）+ 两个 NVRTC 小 harness
（合成张量，见 §4，已实机确认 2 项）。
> 说明：本机无 compute-sanitizer（无 CUDA Toolkit），shared-memory 越界类问题无法用工具捕获，
> 因此下面区分"静态确定"与"实机确认"。

---

## 1. 已确认缺陷（按严重度排序）

### D1 [高] `attn_combine` 的 `w[33]` 与 `kv_split` 配置上限 64 不匹配 —— 可配置触发的 shared 越界写
- 位置：`include/engine/kernels.h:3587` `__shared__ float w[33];`；写入循环 `kernels.h:3586-3598`
  `for (int sp = 0; sp < nsplit; ++sp) { ... w[sp] = wt; ... }`（`sp` 仅 thread 0 执行）。
- 触发：`kv_split`（`src/config.cpp:153-155`，范围 **1..64**，默认 32）> 33，且走 tiled/split-K 解码
  （f32 KV 且 ctx ≥ `attn_flash_min_`=385，或任意量化 KV）。`runner.cpp:1234` 用同一个 `nsplit` 启动
  `attn_combine`。即 `--kv-split 34..64` / `ENGINE_KV_SPLIT=64` 即可触发。
- 后果：**shared memory 数组越界写（UB）**。当前布局下写入落在 `inv_s` 与未使用的 shared 尾部，且
  `inv_s` 在越界写之后才赋值，实测**可能表现为"无害"**（值仍能读回），但这是未定义行为：换编译器/架构/
  寄存器共享量变化即可能静默破坏注意力输出。默认 32 不触发；校准缓存也不会选 >8，但用户配置可触发。
- 修复（10 分钟）：`__shared__ float w[64];`，并在 `choose/validate` 处把 `nsplit` 限制到数组长度
  （或 `static_assert(kv_split_max <= 64)`），另加一条配置越界断言。**建议顺手把 nsplit 上限从 64 降到 32**
  （GQA 之后 32 是最优，见 `docs/reports/DECODE-BUDGET-AUDIT.md`）。

### D2 [高] `moe_route` 对 `n_expert > 256` 静默返回，host 无守卫 —— 未初始化路由槽 → 野指针/越界 atomicAdd
- 位置：`kernels.h:2295` `moe_route(...)`，`kernels.h:2301` **`if (ne > 256) return;`**（不写任何输出）。
- host 侧：`runner.cpp:3589` 启动 `k_moeroute_`（grid=n_tokens, block=32），两个调用点
  `runner.cpp:3049`（layer-graph/pre 路径）与 `runner.cpp:3551`（device-route 路径）都**只检查**
  `L.all_ptr_idx >= 0 && n_tokens <= 8 && k_moeroute_`，**没有 `ne <= 256` 检查**。
- 路由槽缓冲 `slot_exp_/slot_tok_/slot_w_` 由 `cuda::Buffer`（`cuMemAlloc`，**不清零**）分配
  （`runner.cpp:817`，扩容在 3568-3572），内核不写时保留垃圾值。
- 下游：`moe_gateup_silu_dev`（`kernels.h:2346`）读 `slot_exp[slot]` 当专家下标 → 读 `gptrs[e]`
  指针数组越界 → 用垃圾指针取权重；`moe_down_scatter_dev`（`kernels.h:2371`）用 `slot_tok[slot]`
  做 `atomicAdd(&xout[slot_tok*hidden + row], ...)` → **越界写/非法地址/崩溃**。
- 触发条件：MoE 模型 `n_expert > 256` 且走 device-route。device route 默认值是
  **`dev_route_default_ = !q51_expert`（`runner.cpp:119`）**，即除 Q5_1 专家外的 MoE 模型默认开启。
  当前三个模型 ne = 256 / 64 / 0（35B 恰好 256，边界安全）。
  **修正（修复阶段核实）**：v0.1.7 的 IR 校验在加载期就硬拒绝 `n_expert > 256`
  （`src/ir.cpp:581`：`expert_count ... > 256: ... not supported`，`test_model` 以 exit 2 退出）
  ⇒ 这条静默返回路径**当前不可达**，也不会真的出现野指针；它属于"若未来放宽 256 限制就会踩"的
  **潜在缺陷/防御缺口**（Kimi-K2 384 类模型现在会被明确拒绝而不是崩溃）。
- 已实机确认（§4 harness）：ne=256 正常写槽；**ne=300 时 slot 保持哨兵值 `0x7f7f7f7f` 未写**。
- 修复（30 分钟）：host 侧 `if (ne > 256)` 回退到 host 路由（现有 host path 可用）；
  kernel 侧把固定 `pr[256]` 改为分块/动态上限，或用 `ne <= 256` 在 host 选择 kernel；
  并在 `slot_*` 分配后做一次 `zero_fill`（防御性，即使不修边界也能把"野指针"降级为"缺专家"）。

### D3 [中-高] `attn_cached_cfg` 动态 shared memory 超过 48 KiB 时没有 opt-in —— 长 ctx 的通用架构模型启动即失败
- 位置：`kernels.h:4257` `attn_cached_cfg`，`extern __shared__ float sh[]` 按位置使用 `sh[j]`（j ≤ max_seq-1）。
  host：`runner.cpp:1583` `const size_t sh = (max_seq_ + 32) * sizeof(float);`
  `cuda::launch(k_attncfg_, ...)`，**没有** `cuda::max_dynamic_shared_mem(k_attncfg_)`。
  对比：MLA 路径在 `runner.cpp:5228` 有 opt-in；legacy `attn_cached_w` 在 `runner.cpp:1149` 有 opt-in。
- 触发：通用架构（`arch_ext` 的 extended dense 路径，如 gemma4）`--ctx > 12256`
  （48*1024/4 − 32 = 12256）。默认 ctx 4096 不触发，用户设大 ctx 即触发。
- 后果：`cuLaunchKernel(attn_cached_cfg ... shared 60000)` 返回 **CUresult=1 invalid argument**，
  首 token 即抛异常，属"用户可感知的崩溃级启动失败"。
- 已实机确认（§4 harness）：60000 B 动态 smem 启动被拒绝；49152 B 正常。
- 修复（5 分钟）：`cuda::max_dynamic_shared_mem(k_attncfg_)` + 在 ctx 预算里按内核 smem 上限 clamp
  （或超限时回退到 tiled 内核）。**同类检查**：`mla_attn_cached` 已有 opt-in，但设备 opt-in 上限约
  101376 B（3060）⇒ **MLA 模型 ctx > 25344 时仍会启动失败**（Coder 在 12 GB 上 ctx 到不了这么高，
  属潜在；建议在 ctx 预算/启动时报明确错误而不是 cuLaunchKernel 失败）。

### D4 [中] `topk_partial` 的 bitmap `taken[256]` 假设每块 ≤ 8192 个索引，host 未断言
- 位置：`kernels.h:3112` `__shared__ unsigned int taken[256];`；`kernels.h:3117`
  `words = ((hi - lo) + 31) >> 5;`，初始化循环 `for (w = tid; w < words; ...) taken[w] = 0u;` ——
  **没有 `w < 256` 保护**；后续 `taken[off >> 5]` 读/写同理。
- host：`runner.cpp:757` `nblk = (vocab + 1939) / 1940`，clamp 到 ≤256 ⇒ chunk = ceil(vocab/nblk)。
  对 vocab ≤ 496640 时 chunk ≤ 1940 ⇒ words ≤ 61，安全；**vocab > 496640 时 nblk 被 clamp 到 256，
  chunk > 8192 ⇒ words > 256 ⇒ shared 越界写**。当前所有模型 vocab ≤ 248320，不触发（潜在）。
- 修复（15 分钟）：`nblk` 计算改为 `max(computed, ceil(vocab/8192))`（或把 `taken` 改成 512/1KB 并动态
  校验 chunk），加一条 host assert。另 `topk_final` 的 `taken[128]`（`kernels.h:3162`）恰好等于
  `kMaxTopK=128`（`head_topk_max` clamp 到 128），**k=128 时安全**，但同样建议 `static_assert(kMaxTopK<=128)`。

### D5 [中] 死代码 `mla_attn` 的 scores 数组硬编码 512，激活即越界
- 位置：`kernels.h:546-556`：`extern __shared__ float sh[]; float* scores = sh; float* acc = sh + 512;`
  `scores[j]` 对 `j ≤ i`（i = 块即 token 下标，prefill 时可达数千）——sh 的 scores 区只有 512 floats。
- 现状：`k_mlaattn_` 只在 `runner.cpp:377` 注册，**任何地方都没有启动**（见 §2 死代码清单）⇒ 当前无影响。
- 后果（若重新启用）：prefill > 512 token 时 shared 越界写，覆盖 `acc` 及之后。
- 修复（10 分钟）：删掉该内核，或 `scores` 改用 `max_seq` 动态 smem 并做 opt-in。

### D6 [低-中] `gd_ssm_prep` 的 `hist[4]` 与卷积核大小 K 解耦 —— K>5 时栈越界读
- 位置：`kernels.h:1365` `float hist[4];`；`kernels.h:1367-1371` 填充 `i < nh_hist && i < 4`；
  但卷积循环（`kernels.h:1373-1376`）读 `hist[-j - 1]`，j ≥ -(K-1) ⇒ 下标最大 K-2。
  **K ≥ 6 时读 hist[4..] 越界（栈上 UB）**；K=4（qwen35 当前所有模型）安全。
- 修复（10 分钟）：`if (K > 5) return;`（调用方回退非 fused 路径）或 `hist[K-1]` 动态长度（改为固定 8 + assert）。

### D7 [低] `gd_conv_state` 对 K != 4 静默不做事
- 位置：`kernels.h:1153` `if (K != 4) return;` —— 换一个 conv kernel size 的模型会**静默**保留旧 conv 历史，
  输出错误而不是报错。修复：改为通用循环或 host 断言/回退。

### D8 [低] 激活量化内核假设 `n % 32 == 0`，尾部越界写
- `kernels.h:2695` `quant_x_i8`：`dx[2*blk]`、`dx[2*blk+1]`、`tx[blk]` 按 `blocks=ceil(n/32)` 写，
  而 host 缓冲按 `n/32` 分配（`runner.cpp:4466` 启动 `(n+31)/32` 块）⇒ n 非 32 倍数时 `dx` 尾部
  最多越界 2 floats；`kernels.h:1637` `quantize_x_q8` 对最后不足 32 的块读 `x[b*32+i]` 越界。
  GGUF 量化格式保证 n_in % 32 == 0（Q8_0/Q5_0/Q5_1）或 % 256（K-quants），**当前不可达**；
  建议在 kernel_select/调用处加 assert，防未来 misc 形状。

### D9 [低] softmax 全 -inf 行在旧内核里没有 s==0 保护
- `attn_cached_w`（`kernels.h:1734`）、`attn_cached_cfg`（`4257`）、`mla_attn_cached`（`1044`）
  都做 `const float inv = 1.f / lsum;`（无 `s>0` 判断）；tiled/GQA/pre 内核已改为
  `const float inv = s > 0.f ? 1.f/s : 0.f;`。若某行所有分数为 -inf（数值异常/全零 q 之外的溢出），
  旧内核产生 inf/NaN 输出而不是 0。触发极难，属数值稳健性；修复：三处统一加守卫。

### D10 [信息] 设备内存账本计数器只增不减；`g_buf_log` 无上界
- `cuda.cpp:470-472` 在 `Buffer` 构造里累加 `g_buf_bytes/g_buf_count/g_buf_log`，
  **析构/移动赋值释放时没有递减**（`cuda.cpp:476-490`）。这些值仅用于打印
  （`runner.cpp:464`、`2017`），不影响放置决策，但会让"alloc_sum/ledger"日志虚高
  （本次审计期间就看到 `alloc_sum=5322 MiB ... in 6548 buffers` 之类的漂移）。修复（10 分钟）：
  析构里递减，`g_buf_log` 加容量上限或仅 debug 构建启用。

### D11 [信息] capture 期间 upload/sync 是**静默跳过**，没有断言
- `cuda.cpp:516` 的 `capturing_now()` 保护让 `Buffer::upload/upload_async/download/download_async/
  copy_d2d_async/stream_sync/sync`（`cuda.cpp:520-560, 575-605, 830-840`）在 capture 中直接 return
  （只打印前 8 次告警）。当前捕获体（`runner.cpp:2158/2800/2828/3227/5057`）都是"纯 launch"，
  upload/sync 都在 capture 之外，逻辑正确；但这是**靠代码约定而非断言**：将来任何人在捕获体中加一次
  upload 都会被静默跳过 → 固化出错误的 graph。建议 debug 构建里在捕获期把这些 API 变成断言失败
  （或至少每次打印，不只 8 次）。
- `Buffer::upload_at`（`cuda.cpp:495`）既没有 capture 保护也没有越界检查（目标是外部 `dev_dst`），
  当前只在 35B expert streaming 的 prefill 路径调用（`runner.cpp:3721`），不在 capture 内。

### D12 [低] `Event::record()` 记录在 legacy 默认流上
- `cuda.cpp:847` `Event::record()` 用 `EventRecord(ev_, nullptr)`（默认流），而引擎业务都在
  `g_stream` 上（`cuda.cpp:66` 创建、`set_stream`）。仅 `ENGINE_PIPE_PROF` 用（`prof_ev0_/prof_ev1_`），
  但默认流的隐式同步语义会串行化其它流并扭曲测量；若在 graph capture 期间开启，默认流事件还会
  引入隐式捕获依赖/报错。修复（5 分钟）：`record()` 改为 `EventRecord(ev_, g_stream)`。

### D13 [低] grid.y = n_tokens*topk 的 MoE 启动在超大 prefill chunk 下会超过 65535
- `runner.cpp:3516/3526`（host 专家路径）与 `3610/3625`（device 路由路径）：
  `cuda::launch(k_moegateup_, (expert_inter_+7)/8, nact, ...)`，`nact = n_tokens*topk`。
  `prefill_rows` 配置允许到 65536（`config.cpp:161`），35B topk=8 ⇒ `--prefill-rows 8192+` 时
  nact > 65535 ⇒ **cuLaunchKernel invalid**（启动失败）。默认 chunk 512/4096 下 nact ≤ 32768 安全。
  修复（15 分钟）：nact 超限时按 token 分块启动（外循环）或把 nact 放到 grid.x 的两维展开。

### D14 [低] 已死内核（未启动）保留着危险代码
- `attn`（`kernels.h:53`）根本没有 `mod_->kernel()` 注册；`attn_cached`（380）与 `mla_attn`（546）
  注册后从未启动（见 §2 清单）。三种内核分别有 `scores[L]`/`sh+512` 等老式 shared 用法。
  建议删除或标注 `[[maybe_unused]]//dead`，减少审查面与误用风险。

---

## 2. 已核对为正常（重点项）

| 检查点 | 结论 |
|---|---|
| **对齐假设**：所有 `float4/uint4/int4` 强转 | x4 GEMV 的 `float4 xa/xb` 从 `x + sb*256 + lane*8` 读（8 float=32B 对齐，`cuda::Buffer`=cuMemAlloc ≥256B 对齐）；Q4K/Q5K 的 `*reinterpret_cast<const unsigned long long*>(p + qoff)` 中 `qoff=16/48+g*32+l0` 且 l0 ∈{0,8,16,24}，行 stride 144/176 均为 8 的倍数 ⇒ 8B 对齐；Q6_K dp4a **只用 2B 读**（210B stride 无法 4B 对齐，这是历史 misaligned CUDA_ERROR 716 的正确修法）；attn 的 `float4` 只在 `head_dim==256` 且 e0=lane*8 时启用，`cuda::Buffer` 16B 对齐 ✓。 |
| **shared memory 尺寸**（除 D1/D3/D4/D5/D14 外） | `attn_cached_w` 的 `smem_for(nw)=max_seq+32+nw*D` 恰好覆盖 `align32(qpos+1)+nwarp*D`（`kernels.h:1734-1801`）；`attn_cached`/`mla_attn_cached` 的 `max_seq` floats 由 host 精确分配（`runner.cpp:1577`、`5229`）+ MLA 有 opt-in；`gd_delta_rule_128_smem` 69120 B 与 launch 一致且 host 有 `shbytes>=69120` 回退（`qwen35.cpp:512-517`）；`gemv_q5k_x4_dp4a_s` 的 3*n_in 与 staging 下标一致（`kernels.h:2826-2984`）；`rmsnorm/rmsnorm4` 的 smem=blockDim floats ✓；`attn_tiled`/`pre`/`gqa`/`combine`（除 w[33]）都用寄存器+p_dynamic=0。 |
| **GQA 内核 rep 假设** | host 三重保护：`gqa_rep = n_head/n_kv`；要求 `n_head_ % n_kv_ == 0`、`2 <= rep <= 8`、`block=32*rep ≤ 256`（`runner.cpp:1219-1227`）；kernel 另有 `if (h >= n_head) return;`（`kernels.h:3453`）⇒ 不变式成立。prefill GQA 要求 `rep<=4` 且 `head_dim==256`（`runner.cpp:1196`），kernel 只用前 4 个 head（`kernels.h:4145+`）。 |
| **量化 block 解析边界** | Q4_K(144B)/Q5_K(176B)/Q6_K(210B)/Q8_0(34B)/Q5_0(22B)/Q5_1(24B) 的 superblock 迭代上界由 GGUF 格式保证 `n_in%block_elems==0`；逐内核核对 `ql/qh/qs/sc` 最大下标均 ≤ block 尾（Q4K qs 最大 143/144；Q5K qh[31]、qs 最大 175/176；Q6K 单 block 索引 ≤ 209；x8 fp16 变体写 `out[b*32+lane]`（256 u16/block）✓）。 |
| **GEMV n_out 尾部** | 全部 family 都有 `if (row >= n_out) return;`；x4/dev 变体对无效行用 `row=n_out-1` 计算但**只在 valid 时写**；`n_out=0` 不会被请求。 |
| **topk k=128 边界** | `kMaxTopK=128`，`head_topk_max` clamp ≤128（`runner.cpp:752-754`），`topk_pv_/pi_` 按 nblk*128 分配，`topk_final taken[128]` 恰好够用（见 D4 注）。 |
| **dequant f16 / x8 线程数** | host `runner.cpp:4235-4260`：Q4K/Q5K/Q6K per-block 变体 `f16_threads=128`（内核 `t=threadIdx.x` 2 值/线程 ✓），Q8_0=32 ✓；x8 变体 block=256（8 warp×superblock）grid=ceil(nblk/8) ✓。 |
| **`f16_to_f32_ld`** | `i< n` 守卫，`r=i/rows, c=i%rows`，`rows` 来自 slab_rows≥1；写 `out[r*out_stride+c]` 与 host 指针偏移 r0 一致 ✓。 |
| **`transpose_f16`** | 标准 32×33 tile，读/写都有 `<rows/<cols` 守卫，blockDim=(32,8) 与 host `32,8` 一致 ✓。 |
| **`quant_gemm`** | A/W tile 索引 ≤ 32×33，M/N/K 全部有界检查；Q6_K 走 fallback（deq32_slice 对 Q6_K 返回 0 且 host 只对 0/1/3 类型调用）✓。 |
| **capture 安全** | 捕获体（layer graph / prefill chunk graph / whole-model decode graph / cpu-ffn graph）都是 launch-only；`stream_sync()` 在 `capture_begin` 之前；`max_dynamic_shared_mem`、`cuda::Buffer` 分配都在捕获外；`Graph::capture_end` 先清 guard 再检查 rc；`~Graph` 有 capture_abort 兜底；`capturing_active()` 在 shutdown 断言（`runner.cpp:` 析构）✓。`quant_reuse_` 失效条件同时比较 `g_launch_calls` 与 `g_graph_launches` ✓（`runner.cpp` qc_stamp_/qc_graph_）。 |
| **双流（shared block）** | `sh_ev_->record_on(current)` → `sh_ev_->stream_wait(sh_stream_)` → 切到 sh_stream 提交 → `cuda::stream_sync(sh_stream_)` 后再做 host 依赖；capture 期间 `!graph_capture_` 禁用双流 ✓（`runner.cpp:3249-3263, 3457`）。 |
| **资源 RAII** | `Buffer/PinnedBuffer/MappedBuffer` 自写 move ctor/assign，均置空源对象、释放旧资源；`Module` 析构 unload；`Graph` 析构 exec/graph destroy；`Event` 析构 destroy。异常路径无泄漏（构造抛异常时指针尚未保存）✓。计数器的账本问题见 D10。 |

---

## 3. 仍需运行时验证 / 建议补的 harness

1. **D1（attn_combine nsplit>33）**：无法用本机工具捕获 shared 越界；建议在 CI/有 sanitizer 的机器上
   用 `kv_split=64` + 9B ctx 512 跑一 token（或把 `w[33]` 改 64 后 A/B 结果必须逐位一致）。
2. **D2（moe_route ne>256）下游路径**：本 harness 已证明槽不写；要在真实模型上复现需构造 >256 专家的
   GGUF（大工程）。建议先加 host 回退 + slot zero_fill，再用单测覆盖 `ne=300` 的 host 选择逻辑。
3. **D3/D3b（attn_cached_cfg / mla_attn_cached 设备 opt-in 上限）**：用 arch_ext 模型 `--ctx 16384`
   复现 D3（预期抛 `cuLaunchKernel(attn_cached_cfg ... invalid argument)`）；Coder `--ctx 26624` 复现
   MLA opt-in 上限（3060 opt-in = 101376 B ⇒ ctx > 25344 失败）。建议在 ctx 预算里按内核 smem 上限 clamp。
4. **D13（nact > 65535）**：35B `--prefill-rows 8192` 复现 cudaLaunchKernel invalid argument。
5. **D5（死内核删除）**：纯代码清理，无运行时。

---

## 4. 实机确认的 harness（本机运行，`build/audit_kernel_harness.cpp` + `build/audit_kernel_gen.py`）

从 v0.1.7 归档中提取 `attn_cached_cfg` 与 `moe_route` 的原始源码，用引擎自己的 NVRTC/driver 封装
（`src/cuda.cpp`）编译并启动，合成张量、哨兵填充：

```
smem=60000 bytes -> launch rejected: cuLaunchKernel(attn_cached_cfg grid 1x1x1 block 32 shared 60000)
                    failed (CUresult=1, invalid argument)
attn_cached_cfg fails when dynamic smem > 48 KiB (no opt-in)   CONFIRMED
ne=256 -> slot_exp[0]=7 (written)                              CONFIRMED (control)
ne=300 -> slot_exp[0]=2139062143 (UNCHANGED sentinel)          CONFIRMED
harness: 3 confirmed, 0 not reproduced
```

* harness 源码：`build/audit_kernel_harness.cpp`、`build/audit_kernel_gen.py`、`build/audit_kernel_srcs.cpp`
  （均为审计工具，未改产品代码；harness 链接的是主树 HEAD 的 `src/cuda.cpp`，被测内核源码取自归档）。
  运行占用 GPU < 0.1 s，无大模型加载。

---

## 5. 修复清单（按严重度 / 工作量）

| # | 缺陷 | 修复 | 工作量 | 风险 |
|---|---|---|---|---|
| 1 | D1 `attn_combine w[33]` vs kv_split≤64（shared 越界写） | `w[64]` + 配置/选择处断言；顺势把 kv_split 上限 32 | 10 min | 极低 |
| 2 | D2 `moe_route ne>256` 静默（当前被 IR 校验挡住，潜在） | host `ne>256` 回退 host 路由 + slot `zero_fill`（已修，作为放宽 256 限制前的前置防御） | 30 min + 五套 | 低（默认模型不变） |
| 3 | D3 `attn_cached_cfg` 无 opt-in（长 ctx 崩溃） | call `max_dynamic_shared_mem` + ctx clamp | 5-15 min | 极低 |
| 4 | D3b MLA opt-in 超设备上限（ctx>25344） | ctx 预算按 kernel smem 上限 clamp + 明确报错 | 15 min | 低 |
| 5 | D4 `topk_partial taken[256]` host 无断言 | nblk 至少 ceil(vocab/8192) + assert | 15 min | 极低 |
| 6 | D5 死内核 `mla_attn`（sh+512） | 删除 | 5 min | 无 |
| 7 | D6 `gd_ssm_prep hist[4]` K>5 | `if (K>5) return` 或 hist[8] | 10 min | 极低 |
| 8 | D7 `gd_conv_state K!=4` 静默 | host 断言/回退 | 10 min | 极低 |
| 9 | D8 量化激活 n%32 假设 | 调用处 assert | 5 min | 无 |
| 10 | D9 旧 softmax 无 s==0 守卫 | 三处加 `s>0` 保护 | 5 min | 无 |
| 11 | D10 账本计数器只增不减 | 析构递减 + log 限长 | 10 min | 无 |
| 12 | D11 capture 期静默跳过 upload/sync | debug 构建断言 | 10 min | 无（debug 仅） |
| 13 | D12 Event::record 默认流 | 改 `g_stream` | 5 min | 无 |
| 14 | D13 nact>65535 grid.y | 分块启动或断言 | 15 min | 低 |
| 15 | D14 死内核清理（attn/attn_cached） | 删除/标注 | 5 min | 无 |

**结论**：三个发布模型（0.5B/9B/35B/Coder，n_expert ≤ 256、ctx 在默认范围内）走的路径**没有发现
会立即触发的生产缺陷**；确认的高危项都是"边界外推"型：D1 是用户可设配置（`--kv-split >33`），
D2 是 >256 专家的新模型，D3 是通用架构 + 大 ctx。建议 v0.1.8 前把 #1/#2/#3 修掉（合计约 1 小时），
其余按表排期。


---

## 6. 修复记录（v0.1.8，commit `fix(kernels): bounds + smem opt-in + >256-expert guard`）

| 缺陷 | 修复内容 | 回归测试 |
|---|---|---|
| D1 `attn_combine` w[33] / kv_split≤64 | `kernels.h`: `w[64]`；`config.cpp`: kv_split 上限 64→**32**（GQA 后 32 即最优）；`runner.cpp`: clamp 同步 32；`CONFIG.md` 更新。另：**非法 env 值现在直接报错**（此前被静默忽略，`ENGINE_KV_SPLIT=34` 会悄悄回退默认） | `tests/test_kernel_bounds.cpp`：默认 32；`ENGINE_KV_SPLIT=34/64` 被配置层拒绝；8/16/32 接受；`attn_combine` 在 nsplit=8/16/32 下与 host 参考合并一致（含一个空 split 的边界） |
| D2 `moe_route` ne>256 静默（潜在；当前 IR 校验会拒绝该模型） | `runner.h`: `kMaxDeviceRouteExperts=256` + `device_route_supported(ne)`；`runner.cpp`: 两个 device-route 启用条件都要求它（单点语义），`slot_exp_/slot_tok_/slot_w_` 在分配后由 `ensure_route_slots_zeroed()` 清零（含 reallocation 后），即使漏掉守卫也不会把垃圾下标喂给 `gptrs[e]`/`atomicAdd`。**目的**：在放宽 256 专家限制之前先有安全回退 | 守卫单测（256/64 可用，257/300/384 回退）；`tests/make_tiny_gguf_moe.py` 支持 `N_EXPERT` 环境变量，`verify.sh` 生成 **300 专家** fixture，测试断言加载被 **明确拒绝**（`expert_count 300 > 256 ... not supported`，exit≠0），把"拒绝而非崩溃/野指针"的契约钉住 |
| D3 `attn_cached_cfg` 无 smem opt-in | `runner.cpp`: 启动前 `cuda::max_dynamic_shared_mem(k_attncfg_)`；加载期按两个随 ctx 增长的 kernel（`attn_cached_cfg`、`mla_attn_cached`）的 smem 上限 **clamp ctx** 并打印 `[ctx] kernel-smem clamp`，不再在 `cuLaunchKernel` 里崩 | `test_kernel_bounds.cpp`：48 KiB 动态 smem 正常；opt-in 后 60 KB（原失败场景）成功 |
| D6 `gd_ssm_prep hist[4]`（K>5 栈越界） | `kernels.h`: `hist[8]` + `K>9` 防御性返回；`qwen35.cpp`: fused 路径 host 门限加 `K <= 8` | 由 `tools/verify.sh` 的 q35 conv 一致性测试与五套覆盖（K=4 路径） |

其余项（D4/D5/D7-D14）按 §5 待办保留，未在本次改动范围内。
