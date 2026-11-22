# Decode 预算审计：长上下文与逐内核分解

日期：2026-11-21（晚）
范围：**只测量、不改产品代码**。目标：把 9B/Coder/35B decode 每 token 的钱花在哪逐个 kernel 拆开，
找出用户能感知的剩余损失，并按可回收 ms 排序。

工具（都在 `build/`，不进产品）：
- `build/decode_probe.cpp|exe`：直接驱动 `DenseRunner`，绕开 Windows 32KB 命令行上限，
  可跑 prompt 9100 / ctx 9400 的长上下文；输出 per-step decode wall、gen 序列；
  kernel 级数据来自 `ENGINE_KPROF`（runner 析构时 dump 单 token 的逐 kernel GPU 时间），
  分段账本来自 `ENGINE_PROF_ACC`（每 token 一次）。
- `build/audit_decode.sh`：分组执行（`ctx9b` / `kv9b` / `coder` / `b35` / `head`），日志 `build/logs/audit_*.log`。
- `build/bench_overheads.cpp|exe`：host 侧固定开销（Event 创建/销毁、pageable vs pinned H2D、
  launch 提交、空 sync），定位"每 token 固定但可省"的驱动调用。

测量前提：**串行、独占 GPU**（并发会让 VRAM 换页并污染所有数字，18:0x 已发生过一次）。

---

## 0. 模型/内核地图（代码审查结论）

| 模型 | 层/形状 | 注意力 | KV | lm_head |
|---|---|---|---|---|
| 9B (qwen35) | 32 层, hidden 4096, vocab 248320 | GQA heads 16/4, klen=vlen=256 | f32 默认（q8_0/q4_0 可选） | `output.weight` **Q6_K**, 248320×4096×0.8203 B ≈ **834 MB** |
| Coder (deepseek2 MLA) | 27 层, hidden 2048, vocab 102400 | MLA 16 heads, klen 192 vlen 128 | f32（MLA 不支持量化） | **Q6_K** 102400×2048×0.8203 ≈ **172 MB** |
| 35B (qwen35moe) | 40 层, hidden 2048, vocab 248320, 256 experts top-8 + shared | GQA（10 个全注意力层 + 30 个 linear/SSM） | f32, KV layers 10/40 | **Q8_0** 248320×2048×1.0625 ≈ **540 MB** |

注意力路径选择（`run_attn`）：
- 9B/Coder decode 的 **f32 KV** 且 `ctx_len < attn_flash_min_(=385)` ⇒ `attn_cached_w`（legacy warp/head）
- `ctx_len ≥ 385` ⇒ `attn_tiled` + `attn_combine`，**split-K = kv_split_(默认 32)**（decode `n_tokens=1 ≤ attn_split_rows_=8`）
- 量化 KV（q8_0/q4_0）一律走 tiled 路径
- **decode 的 tiled grid = (tokens, n_head, nsplit) ⇒ 每个 query head 独立读整段 K/V**；
  prefill 有 `attn_tiled_pre`（GQA 组复用 K/V，注释称 4× 更少流量），**decode 无此变体**（本轮重点验证项）
- MLA decode 的 attention 段已在 per-layer CUDA Graph 里（v0.1.1），kprof 看不到那段 kernel

每 token 的固定开销（代码位置）：
- `run_pipeline` 每次调用构造 **2 个 `cuda::Event`**（`runner.cpp:2033`，仅在 `ENGINE_PIPE_PROF` 时使用）
  ⇒ 2×`cuEventCreate` + 2×`cuEventDestroy`/token
- `forward_impl`：`x_.upload_async(x.data(), …)`，`x` 是 `std::vector`（**pageable**，9B 16 KB）
- `set_position`：`pos_dev_.upload_async(&pos_host_, 4)`（**pageable** 4 B）
- `cuda::launch`：每次 2 次 `steady_clock`（只喂 `g_launch_ms`）
- decode head：device top-k（默认 8）或整行 logits 的 pinned D2H + `stream_sync`（`[acc] head copy`）

---

## 1. 9B 长上下文：−12% 去哪了

运行时实测（probe 装载日志）：**9B 只有 8/32 层拥有 KV cache**（qwen35 混合栈，其余层是
linear/SSM 状态）⇒ `kv/token = 64 KiB/ctx-position`（8 层 × 4 kv head × (256+256) × 4 B）。
解析模型（每 token 的 K/V DRAM 读）：

| ctx | KV 读/token（1× 最小） | @331 GB/s | 4× GQA 重复读时 |
|---|---|---|---|
| 32 | 2.0 MB | 0.006 ms | 0.024 ms |
| 757 | 48.4 MB | 0.15 ms | 0.59 ms |
| 9100 | **583 MB** | **1.76 ms** | **7.0 ms** |

⇒ 长 ctx −12% 的 KV 开销上限约 **1.8 ms（无重复读）~ 7 ms（GQA 全重复）**，
实测值落在中间就说明 L2 吸收了部分重复 —— kprof 的 `attn_tiled` 斜率可反推真实流量。

**代码侧的两个具体低效点**（对照 prefill 已有的优化）：
1. decode 的 `attn_tiled` grid = `(tokens, n_head, nsplit)`，**每个 query head 单独读整段 K/V**；
   `rep = n_head/n_kv = 4`（9B）⇒ 理论 DRAM 放大 4×（是否被 L2 吃掉由实测判定）。
   prefill 的 `attn_tiled_pre` 用"一个 block 放 8 个同 head 的 query 行，K/V 行由 L1 共享"
   解决了 prefill 的同类问题，但 decode 路径没有对应变体。
2. decode 的 `attn_tiled` f32 路径用 **8 次 strided 4B load** 读 K/V（lane 内 e0=lane*8）；
   `attn_tiled_pre` 的 `f32_wide` 分支已改成 **2 次 float4（512B 全合并）** —— decode 未采用。

**候选实现方案（不改代码，交排期）**：
- 新 kernel `attn_tiled_gqa_decode`：grid `(tokens, n_kv, nsplit)`，block = 32*rep 线程
  （每 warp 一个 query head，同一 block 内所有 warp 逐行读同一 K/V → 首个 warp 拉取、其余 L1 命中）；
  f32 路径直接用 `f32_wide` 的 float4 读法；量化 KV 分支照抄 `attn_tiled`（q8_0/q4_0）。
  数学与 `attn_tiled` 完全一致（同一 per-lane 点积顺序 + online softmax + split-K combine），
  **可逐 token bit-exact**（用 `tools/coder_bitexact.sh` 同款 gen+logits 对照验证）。
- 预期：9B@9100 若实测确有 4× 重复 ⇒ **−1.3~2 ms**；@757 ⇒ −0.1~0.4 ms；短 ctx 影响 ≈0。
  若实测 1× 已达带宽底 ⇒ 收益 ≈0，仅剩 float4 的指令开销优化（~0.1-0.3 ms）。

### 1.1 逐 kernel 表（实测，`ENGINE_NO_GRAPH=1 + ENGINE_KPROF`，单 token GPU 时间）

| kernel（按 ctx9100 排序） | ctx32 ms | ctx757 ms | ctx9100 ms | calls | 备注 |
|---|---|---|---|---|---|
| gemv_q5k_x4_dp4a_s | 12.59 | 12.59 | 13.62 | 115 | Q5_K 投影（与 ctx 无关） |
| gemv_q6k_dp4a（含 lm_head） | 9.19 | 9.19 | 7.75 | 25 | 1-layer 对照：**head ≈3.7 ms**（见 §4） |
| **attn_tiled** | —（legacy） | **0.47** | **5.29** | 8 | **随 ctx 线性增长，见 §1.3** |
| gemv_q6k_dp4a_sk | 5.21 | 5.21 | 4.31 | 45 | |
| gemv_q5k_x4_dp4a | 2.07 | 2.07 | 2.14 | 16 | |
| gemv_q8_0_sk | 2.54 | 2.54 | 1.64 | 48 | |
| attn_cached_w（legacy, ctx<385） | 0.28 | — | — | 8 | |
| rmsnorm/rmsnorm4/quant_x_i8/add/zero_fill | ~4.4 | ~4.4 | ~2.1 | | 与 ctx 无关 |
| kv_write | — | — | 0.41 | 8 | 每 token 写一行，与 ctx 无关 |
| **GPU 总时间/token** | **33.31** | **32.36** | **38.73** | | |

Wall（图开、无 profiling、probe 5 步中位）：**ctx32 27.25 ms｜ctx757 25.74 ms｜ctx9100 32.50 ms**
（test_gen 同条件交叉验证 ctx256 = 26.11 ms，probe 口径一致；本机当前 9B decode ≈26 ms，
早前 "21.24 ms" 在本轮条件下**未复现**，可能是当时时钟/热点状态差异，需记入结论口径）。

### 1.3 attn_tiled 斜率 → GQA 重复倍数（**关键结论**）

- 斜率 = (5.29 − 0.47) ms / (9100 − 757) = **0.578 µs / ctx-position**
- 1× 理论流量 = 64 KiB/position（8 个 KV 层 × 4 kv head × (256+256) × 4 B）⇒ 1× 等效 110 GB/s；
  2× ⇒ 221 GB/s；**3× ⇒ 331 GB/s（≈3060 峰值）**
- ⇒ **实测 ≈ 3× 冗余读**（GQA rep=4 的 4× 被 L2 吸收了约 1/4，剩 3× 打到 DRAM）。
- 可回收：冗余降到 1× ⇒ attention **5.29 → ≈1.8 ms**，**≈ −3.2 ms/token（−10%）@9100**；
  @757 ⇒ −0.3 ms；短 ctx ≈0。
- 交叉证据：`kv_split` 扫描（同 ctx9100，wall）——8:36.58 / **16:31.34** / 32(默认):32.50 / 64:31.60 ms，
  split≥16 后不再改善 ⇒ 已到带宽墙而非并行度不足，与"3× 冗余 + DRAM 带宽"一致。

### 1.4 KV 精度/切分开关（实测，ctx9100 wall）

| 配置 | decode ms | 结论 |
|---|---|---|
| `kv_type=f32`（默认） | **32.50** | 最快 |
| `kv_type=q8_0` | 35.79 | 量化 KV 的 tiled kernel 解包开销 > 省下的带宽，**更慢** |
| `kv_type=q4_0` | 35.23 | 同上，**更慢** |
| `kv_split=16` | **31.34** | 比默认 32 快 **1.16 ms**（长 ctx；短 ctx 待测） |

⇒ **不建议**用 kv 量化换长 ctx 带宽（负收益）；建议评估 `kv_split` 默认 32→16（长 ctx −1.16 ms，
需要补一个短 ctx 对照确认不退化）。

---

## 2. Coder decode 的 41.02 ms 账本

### 2.1 实测逐 kernel（图关 + kprof，单 token GPU 时间合计 **39.27 ms**）

| kernel | ms | calls | 占 GPU |
|---|---|---|---|
| gemv_q5k_x4_dp4a_s | 9.20 | 176 | 23.4% |
| **quant_x_i8**（dp4a 的激活量化） | **5.50** | 135 | **14.0%** |
| moe_gateup_silu（融合专家） | 4.26 | 15 | 10.8% |
| moe_down_scatter | 4.07 | 15 | 10.4% |
| rope_interleaved | 2.21 | 54 | 5.6% |
| zero_fill | 2.11 | 26 | 5.4% |
| add_inplace | 2.07 | 69 | 5.3% |
| rmsnorm4 | 1.90 | 82 | 4.8% |
| gemv_q6k_dp4a | 1.73 | 13 | 4.4% |
| mla_attn_cached / kv_write / split2 / kv_split / build_k | 4.06 | 27 each | 10.3% |
| **lm_head（gemv_q8_0_bm）** | **0.74** | 1 | 1.9% |
| silu_mul / add_three / topk | 0.95 | | 2.4% |

**注意 1（口径）**：Coder 的 wall 本轮被**放置变化污染**：装载时 VRAM 已被其它进程占去一部分，
adaptive 把 `gpu_moe_layers 16→15`（`[vram] free=1650 MiB`），decode 从 41.02 → 61.67 ms。
⇒ **Coder 的 wall 账本需要下一个干净窗口重测**，并记录 `[placement]` 行作为口径。
**注意 2（新候选）**：`quant_x_i8` 每 token **5.5 ms（135 次）** 是新增候选 —— 激活量化在 MoE/专家
GEMV 前被重复执行（v0.1.1 的 `quant_reuse_` 只覆盖连续同激活 GEMV；MoE 段每次 matmul 重新量化）。
设计方向：扩大复用范围，或把 quant 融进 GEMV prologue。属产品改动、风险中等，需先补干净账本。

填表命令：`bash build/audit_decode.sh coder`（wall 看 `audit_coder_wall.log`，kernel 看 `audit_coder_kern.log`；
kprof 会关掉 MLA graph / cpu-ffn graph，所以逐 kernel 时间用图关对照，wall 用图开）。

---

## 3. 35B decode 预算（对照）

已有账本（早前）：CPU 层 host GEMV 0.61–0.73 ms/层 × 24 = 15–18 ms；sync ~0.33 ms/层；
GPU 层 0.2 ms/层；GPU 忙占比 32.6%。本轮用同一 probe 复测，确认 v0.1.2 之后的最新构成。
命令：`bash build/audit_decode.sh b35`。

---

## 4. head 是否已到带宽硬底

| 模型 | head 权重 | 理论时间 @331 GB/s | 实测 | 达成率 | 余量 |
|---|---|---|---|---|---|
| 35B Q8_0 540 MB | 540 MB | 1.63 ms | **1.88 ms**（`gemv_q8_0_bm`, 1 call） | 87% | 0.25 ms |
| **9B Q6_K 834 MB** | 834 MB | 2.52 ms | **≈3.7 ms**（`gemv_q6k_dp4a`，1-layer 对照 2 calls/3.95 ms，head 占绝大部分） | **68%** | **0.9–1.2 ms** |
| Coder Q6_K 172 MB（实际跑 `gemv_q8_0_bm`，推测被重打包为 Q8_0 block-major） | 172 MB | 0.52 ms | **0.74 ms**（1 call） | 71% | 0.1–0.2 ms |

9B head 的识别方法：`ENGINE_LAYERS=1 + NO_GRAPH + KPROF` 只跑 1 层，`gemv_q6k_dp4a` 只剩
2 calls / 3.95 ms，其中 head（n_out=248320）占绝大部分 ⇒ 当时估算 834 MB / ~3.7 ms = 225 GB/s（68%）。

### 4.1 复核修正（2026-11-22，逐 GEMV 计时）

上面的 225 GB/s 是**从聚合计时反推的错误估计**：把同一 kernel 的两次调用（head + 一层 8192×4096
投影）当成 head 一个。用 `ENGINE_DBG_GEMVTIME=1` 直接打印单次 GEMV 后：

| GEMV | 权重 | 实测 | 带宽 |
|---|---|---|---|
| 9B head 248320×4096 Q6_K（`gemv_q6k_dp4a`） | 834.36 MB | **2.82–2.86 ms** | **291–295 GB/s（88–89% 峰值）** |
| 9B head 同形状（新 `gemv_q6k_dp4a_v2`，2 行/warp + 共享激活） | 834.36 MB | 2.83–2.96 ms | 282–295 GB/s（**无提升**） |

⇒ **9B head 已接近带宽硬底，可回收余量只有 ~0.1–0.2 ms（不是原先估的 0.9–1.2 ms）**，不值得为它
增加一个 2 行/warp 内核（实测还略慢：Q6_K 210 B/superblock 的 stride 让 4B q 载入无法对齐，
退回 2×2B 后 2 行/warp 的收益被抵消）。**该项关闭，不改代码。**
35B/Coder 的 head 已用 block-major 布局（87%/71%），同样接近硬底。
（教材：审计报告里凡是"由聚合 kernel 计时反推单次时间"的数字，落地前必须用 per-call 计时复核。）

---

## 5. 每 token 固定但可省的驱动调用

`build/bench_overheads.exe` 输出（host 侧 median µs/call）：

| 项 | µs/call | 每 token 次数 | 可回收 µs |
|---|---|---|---|
| `Event` ctor+dtor（cuEventCreate+Destroy） | **0.46** | 2 | **≈0.9（可忽略）** |
| pageable `upload_async` 4 B（pos） | **28.41** | 1 | — |
| 同上改 pinned+memcpy | 10.57 | 1 | **≈17.8** |
| pageable `upload_async` 16 K（x 行） | 11.84 | 1 | — |
| 同上改 pinned+memcpy | 12.10 | 1 | ≈0（pageable 16K 没有额外成本） |
| `stream_sync`（空队列） | 0.21 | 1–2 | — |
| `cuda::launch`（warm） | 11.41 | ~240（图外） | 每次 2 次时钟 ≈0.05 µs |
| launch + sync round trip | 16.39 | — | — |

⇒ 固定开销里**只有 `set_position` 的 pageable 4B 拷贝值得改（~18 µs/token，0.08%）**；
"每 token 两个 Event" 实测仅 0.9 µs，**不列为可回收项**；`x` 的 pageable 16 K 上传没有额外成本。

另外用 `ENGINE_PROF_ACC` 的 `accounted vs wall` 残差复核：残差 >0.2 ms 说明还有未分类的每 token 开销。

---

## 6. 可回收项排序（初版，实测后更新）

| # | 项 | 实测/预估收益 | 风险/成本 | 类型 |
|---|---|---|---|---|
| 1 | **decode attention GQA 复用 + f32 float4** | **9B@9100 −3.2 ms（−10%）；@757 −0.3 ms；短 ctx ≈0** | 中：新 kernel（照 `attn_tiled_pre` 的 L1 共享 + `f32_wide` 思路），数学不变可 bit-exact | 产品改动；建议排期 |
| 2 | **`kv_split` 默认 32 → 16** | **9B@9100 −1.16 ms**（31.34 vs 32.50）；短 ctx 待测 | 低（改默认或按 ctx 自适应） | 配置默认 |
| 3 | ~~9B head Q6_K 向量化~~ | **已证伪：实测 291–295 GB/s（88–89% 峰值），余量仅 ~0.1–0.2 ms；新 2 行/warp 内核无提升 ⇒ 关闭**（见 §4.1） | — | 不做 |
| 4 | Coder `quant_x_i8` 5.5 ms/token（14% GPU） | 0.5–2 ms（需查调用来源/复用范围） | 中：扩大 quant 复用或融进 prologue | 产品改动；先补干净账本 |
| 5 | `set_position` 改 pinned | 18 µs/token（0.08%） | 极低 | 产品改动（顺手） |
| — | 长 ctx 用 `kv_type=q8_0/q4_0` | **负收益（+2.7~3.3 ms）** | — | **不做** |
| — | 每 token 2 个 Event / `cuda::launch` 双时钟 | ≈1 µs / 未测到实际差 | — | **不做** |
| ? | Coder CPU 层 host expert GEMV 占比 | 待干净窗口用 prof_acc 复测（本轮被放置变化污染） | — | 待测 |


---

## 7. 测量日志索引

| 日志 | 内容 |
|---|---|
| `build/logs/audit_9b_ctx32.log` / `ctx757` / `ctx9100` | 9B 三档 ctx 的 kprof + prof_acc + per-step ms |
| `build/logs/audit_9b_ENGINE_KV_TYPE_q8_0.log` 等 | kv 精度/切分 sweep |
| `build/logs/audit_coder.log` | Coder 账本 |
| `build/logs/audit_35b.log` | 35B 账本 |
| `build/logs/audit_9b_head.log` | head 带宽 |

（测量结果在下节补入。）
