# MTP 投机解码 实施报告（MTP-IMPL）

日期：2026-11-15 · 目标：Qwen3.6-35B-A3B decode 25.7 → ≥38.6 tok/s（150% of llama.cpp 25.76）
结论先行：**机制 P0+P1 已完整实现并验证无损；150% 目标未达到，且用实测数据证明本轮实现路线下 35B 不可能达到**
（原因不是 draft 质量，而是本引擎 batch-verify 恒等于 k+1 倍单步成本）。见 §4 成本分解。

---

## 0. TL;DR（6 条）

1. **MTP 层成功装载并算对了**：`blk.40` 的 4 个 `nextn.*` 张量 + 完整 attention+MoE 层，走 `load_q35_attn`
   复用主模型层代码 ⇒ 数值语义与主模型同源。`[mtp] block blk.40 loaded: eh_proj=Q4_K 2048x4096,
   256 experts on CPU`。
2. **draft top-1 命中率（真实文本，35B，贪心）**：单 prompt 12 步 **10/12 = 83.3%**；三个 prompt 见 §2。
   这同时反证了 `eh_proj` 的拼接顺序 **`concat(enorm(emb), hnorm(h))`（emb 在前）是正确的**
   （顺序错则命中率会跌到 ~0）。
3. **单步投机解码（k=1..3）已实现且逐 token 无损**：`[t_next, d1..dk]` 批 verify + 最长一致前缀接受
   + gated-delta-net `conv_state/rec_state` 的**逐位置槽位快照回滚**（attention KV 位置索引，回滚免费）。
   实测 **输出 token 与纯贪心逐 token 完全一致**（35B / Qwythos-9B-MTP 全部 IDENTICAL）。
4. **35B 上 MTP 是负收益**：plain 41.8 ms/token → k=1 **67.2 ms（−61%）**、k=2 70.2 ms（−68%）。
   净吞吐 **23.9 → 14.9 tok/s**，距 38.6 tok/s 目标差 2.6×。
5. **根因（实测，非推测）**：本引擎 `n_tokens>1` 前向的成本 ≈ `n ×` 单步成本（35B：n=2 时 2.01×、
   n=3 时 2.43×、n=4 时 3.16×），因为 decode 的时间几乎全部花在 **每 token 独立的
   expert gather**（CPU 24 层 × 8 专家 × 2.04 MB = 392 MB/token）和 `n>1` 的 fp16 反量化暂存路径上。
   投机解码只能在"权重读一遍服务多行"上赚，而 **MoE 的 8/256 路由让两个 token 几乎不共享权重**
   （2 token 的专家并集 15.7/256 ⇒ CPU 侧工作量严格翻倍）。
   ⇒ 收益模型 `S = E/(c+k·d)` 中 **c≈2.0–2.6 > 1 且 E≤2**，数学上不可能 >1×。
6. **同一套机制在 Qwythos-9B-MTP（额外发现的带 nextn 头的模型）上同样因为 `n>1` 路径而负收益**
   （plain 30.96 ms → k=1 94.8 ms，α=0.778 但 c≈5.4：稠密模型 `n>1` 走"整模型反量化成 fp16 暂存"
   ⇒ 权重流量 6.13 GB → 30 GB）。⇒ **要拿到 MTP 的真收益，前置条件是 MTP-RECON §3.3 的 P0-A
   "2–4 行量化 GEMV + 多行 head"（读一遍权重算多行）**，否则任何模型都亏。该 kernel 本轮未实现。

---

## 1. MTP 层装载（P0）与显存账

| 项 | 数值 | 说明 |
|---|---|---|
| MTP block | `blk.40`（`ir.mtp_block=40`, `mtp_layers=1`） | 由 `.nextn.` 张量名 + `*.nextn_predict_layers` 元数据键交叉校验（不按模型名特判） |
| 装载方式 | `load_q35_attn(40, …)` | MTP 层恒为 full-attention 层（与主模型 `full_attention_interval=4` 无关），代码与主模型层完全同源 |
| nextn 张量 | `eh_proj` Q4_K[4096→2048]、`enorm/hnorm/shared_head_norm` F32 | 共用主模型 `token_embd` / `output.weight`（Q8_0 515 MiB，`lm_head_qt_` 已验证可用） |
| MTP 专家放置 | **CPU**（256 专家，522 MiB） | `[mtp] MTP experts: CPU (need 522 MiB, free 238 MiB, margin 256 MiB)`：35B 在 ctx=77 时装载完激活后**只剩 238 MiB**，装不下 522 MiB |
| MTP 非专家部分 | GPU（attn ~15.5 MB + shared expert 2 MB + router） | 与 MTP-RECON 的判断一致："35B 的 MTP 层放不进显存"——**但原因是激活/图占用把 free 吃到 238 MiB，不是 MTP-RECON 说的 1137 MiB**（那是在激活分配前测的） |
| MTP KV | 2 kv head × 256 × 2 × 4 B = 4 KiB/token ⇒ ctx 77 = 308 KiB | 独立 KV cache（`alloc_kv_cache`） |
| 回滚槽位 | 30 个线性层 × (rec 2.0 MiB + conv 0.1 MiB) × k | k=1 时 **63 MiB**，k=2 时 126 MiB，k=3 时 189 MiB；35B 当前 free 238 MiB ⇒ k≤3 可行（k=1 最稳） |

⚠️ 一个与 MTP-RECON 不同的实测事实：**MTP 层的非专家权重只有 ~20 MB，真正占地方的是 522 MB 路由专家**。
把 MTP 专家放 GPU 需要 522+256=778 MiB 空闲，而 35B 在激活/CUDA Graph 分配之后只剩 238 MiB。
想放 GPU 必须从主模型挪走一个 GPU MoE 层（`gpu_moe_layers 16→15`，主 decode 每 token +0.8 ms），
与省下的 draft host 往返（~2 ms/round）大致抵消，本轮不做。

## 2. draft top-1 命中率（真实文本）

35B，贪心，`tools/mtp_run.exe --mode acc --mtp --n 16`，逐位置对拍（draft vs 目标模型 arg-max）：

| 模型 | prompt（ids） | 文本 | 步数 | top-1 命中率 |
|---|---|---|---|---|
| **35B** | 760 | "The" | 12 | **10/12 = 83.3%** |
| **35B** | 760,6511,314,9338,369 | "The capital of France is" | 16 | **14/16 = 87.5%** |
| **35B** | 109266 | "你好" | 16 | **12/16 = 75.0%** |
| Qwythos-9B-MTP | 760,6511,314,9338,369 | "The capital of France is" | 16 | **15/16 = 93.8%** |

（`tools/mtp_run.exe --mode acc --mtp --n 16`；每一行都是"draft 与目标模型 arg-max 逐位置对拍"。）

对照 llama.cpp 同模型实测（用户数据）：temp 0.6/0.7 α=0.45–0.62、greedy α 未测。**本引擎 greedy 下
α=0.71–0.83（spec 实测 0.714 / 0.583，acc 实测 0.833），比 llama.cpp 的采样 α 更高**——draft 侧实现是对的。

链式 k 步的第二/第三步 α 明显下降（k=2 实测 α=0.583、k=1 α=0.714）：第一个 draft 错则后序全错，
且链式 draft 的 h 来自 MTP 自己的输出（累积误差）。这与 llama.cpp 的经验一致（n-max=2 最优）。

## 3. k=1/2/3 端到端 tok/s 与逐 token 一致性

`tools/mtp_spec_ab.sh`（同一二进制、同 prompt、median of 3；`decode` = 预热轮之后的稳态）：

### Qwen3.6-35B-A3B（ctx=65，gpu_moe_layers=16，MTP 专家在 CPU）

| 配置 | ms/token | tok/s | α | 输出 token |
|---|---|---|---|---|
| plain greedy | 41.77 | 23.9 | — | `2614 513 5081 5576 4602 318 4058 10926 8 883 10774 76467` |
| spec k=1 | 67.22 | 14.9 | 0.714 | **IDENTICAL** |
| spec k=2 | 70.16 | 14.2 | 0.583 | **IDENTICAL** |

单轮成本拆分（`--mode spec` 自带统计，prompt="The capital of France is"，7 轮）：
```
draft  5.43 ms/轮 = 0.13 个 decode（MTP attn+MoE 在 CPU ⇒ host 往返 ~3 ms + 共享 head 540 MB ≈ 2 ms）
verify 115.84 ms/轮（n=2）= 2.77 个 decode   ← 死因
E = 1+α = 1.857；T = 2.90 decode ⇒ 每 token = 1.56 decode = 71.8 ms（13.9 tok/s）
```

### Qwythos-9B-MTP（ctx≈81，MTP 层全 GPU，稠密 FFN）

| 配置 | ms/token | tok/s | α | 输出 token |
|---|---|---|---|---|
| plain greedy | 30.96 | 32.3 | — | `2614 513 5081 5576 4602 318 4058 10926 8 883 14963 18076 13 198 3710 369` |
| spec k=1 | 94.83 | 10.5 | 0.778 | **IDENTICAL** |
| spec k=2 | 87.95 | 11.4 | 0.562 | **IDENTICAL** |
| spec k=3 | 77.61 | 12.9 | 0.476 | **IDENTICAL** |

单轮成本拆分（prompt="The capital of France is"，k=2，6 轮，α=**1.000**——重复性文本把接受率打满）：
```
draft  10.81 ms/轮 = 0.35 个 decode（2 个 draft 步：2×(258 MB MTP 层 + 834 MB 共享 head)）
verify 179.64 ms/轮（n=3）= 5.80 个 decode   ← 稠密模型 n>1 走"整模型反量化成 fp16 暂存"的代价
E = 3.0；T = 6.15 decode ⇒ 每 token = 2.05 decode = 63.5 ms（实测 72.6）
```

→ **无损性成立**（投机解码的硬判据）：接受/拒绝规则是"目标模型自身 arg-max 与 draft 完全一致才保留"，
被拒绝的 draft 的 KV/SSM 状态全部回滚，所以最终 token 序列与纯贪心逐 token 相同。

## 4. 35B 为什么达不到 150%：精确成本分解（实测）

### 4.1 `c` = 一次 (k+1)-token 验证前向 / 一次 decode（`tools/mtp_probe.exe`，35B，ctx=65）

| 批大小 n | 时间 | c = T(n)/T(1) | 每 token |
|---|---|---|---|
| 1 | 40.35 ms | 1.000 | 40.35 |
| 2 | 81.06 ms | **2.009** | 40.53 |
| 3 | 98.22 ms | 2.434 | 32.74 |
| 4 | 127.63 ms | 3.163 | 31.91 |

**这是死因**：验证 2 个 token 的成本 = 2 × decode 成本，而一轮最多只能赚回 2 个 token
（E ≤ 1+α ≤ 2）⇒ `S = E/(c+k·d) ≤ 2/(2.0+d) < 1`。

### 4.2 每层分解（`ENGINE_DBG_LAYERMS`，35B）

| n | 24 个 CPU 层合计 | 16 个 GPU 层提交 | 合计 |
|---|---|---|---|
| 1 | 38.5 ms（中位 1.45 ms/层） | 3.7 ms | ≈42 ms |
| 2 | **80.1 ms（中位 2.85 ms/层）** | 8.4 ms | ≈88 ms |

CPU 层里 expert matvec（`ENGINE_DBG_MOE_PHASES`）：n=1 时 0.94 ms/层（8 个 (expert,token) 对），
n=2 时 **1.89 ms/层（16 对）**——严格线性。
⇒ 两个 token 的 8/256 路由专家并集 = 256·(1−(1−8/256)²) = **15.7 ≈ 16**，
**CPU 专家读取几乎不共享**，批验证省不到任何权重流量。

### 4.3 一轮的实测成本（`--mode spec`，35B，k=1，α=0.857，7 轮）

```
draft  5.43 ms/轮   = 0.130 decode
verify 115.84 ms/轮 = 2.773 decode   （独立 probe 测 n=2 = 2.01×，多出的 0.76 来自：
                                      SSM 逐 token 快照（30 层 × ~5 次额外发射）、
                                      多行 head 走 fp16 暂存（540 MB → 1.6 GB 流量 ≈ 6 ms）、
                                      per-row logits 下载 + host argmax）
E = 1.857 tokens/轮；T_round = 121.3 ms = 2.90 decode
⇒ 每 emitted token = 1.56 decode = 65.3 ms（实测 71.8，含轮首/轮尾不整齐）
S = 1.857 / 2.90 = 0.64×（−36%）
```

### 4.4 要达到 150% 需要什么

`S = E/(c+k·d)`；k=2、α=0.83 ⇒ E = 1+0.83+0.69 = 2.52。要 S≥1.5 需 `c+2d ≤ 1.68`。
当前 c≈2.0–2.6。⇒ 必须把 **n=2 的验证前向压到 ≤1.4× 单步**（现状 2.0×）。
唯一路径 = MTP-RECON §3.3 的 **P0-A：2–4 行量化 GEMV（读一遍权重算多行）+ 多行 head**，
把 GPU 侧权重流量摊薄；但即使做到 c=1.15，35B 的 CPU MoE 部分（24 层、占 38.5/42 ms）**不会摊薄**
⇒ 乐观上限 c≈1.55（CPU 部分 0.92 + GPU 部分摊薄 0.6）⇒ S(k=2, α=0.83) = 2.52/(1.55+0.2) = 1.44（+44%），
k=1 时只有 1.13。**150% 在本引擎的 35B 上即便补完 P0-A 也基本不可达；`≥32 tok/s`（P1 验收线）在 c≈1.55
时也达不到（需 c ≤ 1.4）。** 真正能吃到 MTP 收益的是**稠密全 GPU 模型**（Qwythos-9B-MTP）：
只需 P0-A 把 c 从 5.4 压到 1.15，α=0.78 ⇒ S(k=1)=1.78/1.29=**1.38（+38%）**。

## 5. 三套验证 + gen_regression 状态

见 §8 运行记录：

| 套件 | 结果 |
|---|---|
| `tools/verify.sh` | **ALL GREEN** |
| `tools/gen_regression.sh` | 见 §8 |
| `tools/smoke_models.sh` | 见 §8 |
| `tools/verify_generality.sh` | 见 §8 |

MTP 默认 **关闭**（`mtp=false`），所以以上套件的行为与改动前逐位相同；改动只新增
`Runner::load_mtp/ spec_generate / mtp_accuracy` 与两条 `if (flag)` 分支。

### 5.1 额外：MTP 路径对 llama.cpp golden ids 的逐 token 一致性（最强证据）

直接用 `gen_regression.sh` 里 35B 的三个 golden 用例跑 `--mtp`（env 相同：`ENGINE_GPU_EXPERTS=16`）：

| 用例 | golden（llama.cpp） | `--mtp` 输出 | 结果 |
|---|---|---|---|
| `35b_capital_of_france`（k=1，α=0.857） | `11751 11 264 3177 34756 364 1141 25438 57902 1680 430 279` | 完全一致 | ✅ |
| `35b_zh_nihao`（k=2，α=0.300） | `3709 95815 111354 98292 97432 3709 99443 97706` | 完全一致 | ✅ |

⇒ **投机解码路径不仅与纯贪心一致，还与 llama.cpp 的 golden 逐 token 一致**（即使 k=2 时 α 只有 0.30、
大量拒绝+回滚，token 仍完全正确）。这同时验证了 SSM 回滚（拒绝时必须回滚 conv/rec 状态）没有跑偏。

## 6. 代码改动清单

| 文件 | 改动 | 说明 |
|---|---|---|
| `include/engine/ir.h` | `mtp_block`, `mtp_desc` | MTP block 描述（不改 `layers` 尺寸，避免影响所有模型的预算/校验） |
| `src/ir.cpp` | 填 `mtp_block`/`mtp_desc`，读 `nextn_predict_layers` 交叉校验 | 能力判定只看"有没有 nextn 层" |
| `include/engine/runner.h` | MTP 私有成员 + `mtp_available/spec_generate/mtp_accuracy/mtp_draft` | |
| `src/mtp.cpp`（**新**） | MTP 装载、draft 步、MTP KV context 预填充、verify 批前向、接受/回滚、spec 主循环、draft 准确率 | ~600 行 |
| `src/runner.cpp` | ① 构造函数末尾 `load_mtp()` ② `run_pipeline` 多行量化 head（`verify_multi_head_`） ③ `forward_impl` 的 verify 分支（逐行 arg-max） | 都在 `if (verify_multi_head_)` 分支内 |
| `src/qwen35.cpp` | ① `load_q35_ffn` 支持 MTP block 的 LayerDesc 与显式放置 ② `run_q35_linear_attn` 的 `spec_snapshot_` 模式：conv/delta 按 token 逐个推进 + 逐位置快照 | 数值与批版逐位一致（同一递推，T=1 循环） |
| `src/cuda.cpp` + `include/engine/cuda.h` | `copy_d2d_async`（`cuMemcpyDtoDAsync_v2`） | 状态快照 / hidden 保存 |
| `src/config.cpp` | `mtp` / `mtp_k` / `mtp_gpu` / `mtp_vram_margin_mb` / `mtp_ctx_prefill` | 注册进统一配置面 |
| `tests/test_gen.cpp` | `--mtp` 时走 `spec_generate`（同一 `gen:` 输出格式 ⇒ 可逐 token 对拍） | |
| `docs/CONFIG.md` | 新增 §3.8 MTP / 投机解码 | |
| `tools/mtp_probe.cpp`（新） | 验证批 c=T(n)/T(1) 度量 | |
| `tools/mtp_run.cpp`（新） | `--mode acc/plain/spec` | |
| `tools/mtp_spec_ab.sh`（新） | plain vs spec 交错 A/B + 逐 token 一致性 | |
| `build.sh` | 源列表加 `src/mtp.cpp` | |

## 7. 失败/未完成项（如实记录）

1. **P2/P3 未做**：链式 k 只是"顺序 draft k 步"，没做自适应 k、没做 MTP KV 的精确回滚
   （被拒绝位置的 draft KV 由下一轮覆盖，不影响无损性，只影响 α）、没做 verify 的 CUDA Graph。
2. **MTP 专家没进显存**：35B 激活+CUDA Graph 后只剩 238 MiB（MTP-RECON 引用的 1137 MiB 是激活分配前的
   数值）。曾试过"把 MTP 专家放 GPU"，被显存拒绝并自动退回 CPU（设计如此）。
3. **第一次测量 c 时被污染**：早期 probe 先跑 `last_row_only=false`（物化 2 GB f32 head）再测
   last-only，导致 n=2 测出 112×；修正 probe（只测 last-only、不污染显存）后得到 2.01×。
4. **想用 serial verify（k 次 n=1 前向）绕过批成本**：等价于 T = E×decode + draft ⇒ 必然 <1，放弃。
5. **`main.cpp run` / `serve` 未接 MTP**：只接了 `test_gen`（官方基准与 gen_regression 用的就是它）
   和新工具 `mtp_run`；`server` 的 SSE 路径要另接流式接受逻辑，本轮时间不够。
6. **未做 P0-A 多行量化 GEMV**：这是让 MTP 真正变快的唯一前置，工作量大（4 种量化类型 × 2–4 行 kernel
   + 派发 + 数值对拍），本轮来不及；见 §4.4 的定量必要性。

---

## 8. 运行记录（原始数据）

### 8.1 三套验证 + gen_regression（`build/mtp_regression.log`）

```
===== verify.sh
... M2 : PASS - plan identical (18 fields) ... ALL GREEN
===== gen_regression
9b_capital_of_france         PASS (12 greedy tokens == llama.cpp)
9b_the                       PASS (10 greedy tokens == llama.cpp)
9b_france_is                 PASS (8 greedy tokens == llama.cpp)
9b_zh_nihao                  PASS (8 greedy tokens == llama.cpp)
9b_chat22                    PASS (8 greedy tokens == llama.cpp)
35b_capital_of_france        PASS (12 greedy tokens == llama.cpp)
35b_chat22                   PASS (8 greedy tokens == llama.cpp)
35b_zh_nihao                 PASS (8 greedy tokens == llama.cpp)
coder_v2_lite                PASS (1 greedy tokens == llama.cpp)
qwen2_5_0b                   PASS (1 greedy tokens == llama.cpp)
results: 10 passed, 0 failed        GEN REGRESSION GREEN
===== smoke_models
qwen2.5-0.5b  : 40716 | Coder-V2-Lite : 185 | Qwen3.5-9B : 2614 | Qwen3.6-35B : 2614
===== verify_generality
results: 26 passed, 0 failed        GENERALITY GREEN
```

### 8.2 c = T(n)/T(1)（`build/mtp_probe.exe`，35B，ctx=65）

```
[placement] gpu_moe_layers=16/40, post-load free=1450 MiB
[probe] n=1 decode 40.35 ms (24.8 tok/s)
[probe] n    ms         c=T(n)/T(1) per-token
[probe] 2    81.06      2.009      40.53 ms/token
[probe] 3    98.22      2.434      32.74 ms/token
[probe] 4    127.63     3.163      31.91 ms/token
```

### 8.3 draft 准确率（`build/mtp_accuracy_multi.log`）

```
35B  capital-of-France : mtp_draft_acc: 14/16 = 0.875
35B  zh 你好          : mtp_draft_acc: 12/16 = 0.750
35B  760(The)         : mtp_draft_acc: 10/12 = 0.833
9B-MTP capital        : mtp_draft_acc: 15/16 = 0.938
```

### 8.4 spec 单轮拆分

```
35B  k=1: draft 5.43 ms/轮, verify 115.84 ms/轮 (n=2), alpha=0.857, steady 71.78 ms/token (13.9 tok/s)
9B   k=2: draft 10.81 ms/轮, verify 179.64 ms/轮 (n=3), alpha=1.000, steady 72.55 ms/token (13.8 tok/s)
35B  k=1 + ENGINE_GPU_EXPERTS=16（MTP 专家上 GPU）: draft 7.56, verify 149.83, 89.49 ms/token
     —— 注意这条是"16/256 专家在 GPU"的慢放置，baseline 本身也慢；它只用来证明 **MTP 专家 GPU/CPU
     两种放置都能正确工作**（`[mtp] ... 256 experts on GPU` 与 `on CPU` 都跑通、token 都对）。
```

### 8.5 显存账（MTP 相关）

| 项 | 35B | Qwythos-9B-MTP |
|---|---|---|
| MTP 层非专家权重 | ~20 MB（attn 15.5 + shared 2 + router） | ~10 MB |
| MTP 路由专家 | 522 MB（**CPU**：装了激活后 free 只剩 233–238 MiB） | 无（稠密 FFN，258 MB 全 GPU） |
| MTP KV | 4 KiB/token（ctx 77 → 308 KiB） | 8 KiB/token |
| 回滚槽位（k=1 / k=2） | 63 MiB / 126 MiB | 50 MiB / 100 MiB |
