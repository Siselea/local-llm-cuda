# 层主序 prefill 设计（LAYER-MAJOR-PREFILL-PLAN）

> 作者：prefill-fast｜日期：2026-11-21｜性质：**纯设计文档（未上 GPU 验证）**，数据来自本轮已实测的 prefill 成本模型
> 背景：`PREFILL-CLIFF-REPORT.md` 修掉了 serve 的两个用户可感知慢点（长文档 1.85–2.0×、假冷启动）。本文件回答"下一步怎么把长文档与 tail 的剩余固定成本再压下去"。

---

## 0. TL;DR（结论先给）

1. **层主序（layer-major）prefill 的收益 = (chunk 数 − 1) × 单次整模型 dequant 成本**。用实测数：
   - 13.4k prompt、chunk=1024（14 块）：省 `13 × ~95 ms ≈ 1.24 s` / 16.0 s ⇒ **约 −7.7%**
   - 4.4k prompt、chunk=1024（5 块）：省 `4 × 95 ≈ 0.38 s` / 3.7 s ⇒ **约 −10%**
   - **52-token tail（1 块）：收益 0**（本来就只 dequant 一次）
2. 所以：**层主序不是 tail 的解**。tail 的 ~180–200 ms 固定成本要另找杠杆，其中最便宜的是**小 n 的 CUDA graph**（把 ~700 次 launch 变成 1 次；现有 `prefill_graph` 机制早就写好，只是默认关、且当时对大 chunk 无收益所以被搁置）。
3. 推荐顺序：**先做小 n CUDA graph（1–2 h，直接改善多轮首 token）→ 再做层主序 V1（FFN-only，2–3 h，长文档 −8~−10%）**。

---

## 1. 实测成本模型（本轮与 server-mode 的独立数据）

| 场景 / 配置 | 每 forward 固定成本 | 每 token 成本 | 来源 |
|---|---|---|---|
| rows=512，默认 fp16 路径 | **≈198 ms** | 0.67 ms | 本报告 `PREFILL-CLIFF-REPORT.md` §5.4.1 |
| rows=128，默认 fp16 路径 | **≈180 ms** | 0.124 ms | server-mode `build/logs/tail_probe_plain.json` |
| rows=128，`ENGINE_GEMM=1`（自研量化 GEMM） | ≈381 ms | 0.581 ms | server-mode `tail_probe_gemm.json`（更差） |
| rows=128，`ENGINE_PREFILL_F16=0`（原 f32 路径） | ≈180 ms | 0.124 ms | server-mode `tail_probe_nof16.json` |

固定成本 ≈ **180–200 ms/forward**，它与 chunk 大小、与 `prefill_f16` 单开关都无关 ⇒ 说明它不是"f16 权重量化"这一项，而是**每次 forward 都要为整个模型做一遍权重搬运 + 每矩阵一次 launch** 的合成：

| 组成（9B 估算，待 instrument 确认） | 量级 |
|---|---|
| 整模型权重 dequant → fp16（6.1 GB 读 + 12.2 GB 写） | 理论 55 ms / 旧内核实测 ~95 ms |
| 每层 ~7 个 matmul × 32 层 ≈ **700+ 次 kernel launch**（WDDM 提交 ~30–100 µs/次） | 20–70 ms |
| 逐 chunk 的 x→fp16 转换、attention/gd_delta 的小批处理、host 往返 | 数十 ms |

> 长文档时"整模型 dequant"被**每个 chunk 重复一次**：13.4k @chunk=1024 = 14 次；@chunk=128 = 105 次（这正是 `PREFILL-CLIFF-REPORT` §3 修掉的 1.85× 差距来源，改为 chunk=512/1024 后仍剩 14 次）。

---

## 2. 层主序 prefill（设计）

### 2.1 核心变换

现状（chunk-major，`forward_impl` 的递归分块）：外层是 chunk，内层是 32 个层 —— 每个 chunk 都把整模型权重 dequant 一遍。

```
for chunk in chunks:            # 外层
    for layer in layers:        # 内层
        run_layer(layer, chunk) # 每个 matmul 内部 dequant 该矩阵的权块
```

层主序（layer-major）：

```
for layer in layers:                      # 外层
    dequant_layer_weights(layer)          # 一次性：该层权重 → fp16 staging
    for chunk in chunks:                  # 内层：token 分块（顺序推进 KV/SSM 状态）
        run_layer_math(layer, chunk)      # GEMM 直接用已 dequant 的 fp16 权重
```

### 2.2 关键约束与可行性

1. **状态顺序**：KV cache 位置、SSM/conv 递归状态必须按 token 顺序推进 ⇒ chunk 循环必须保持**从小到大**，不能并行。层主序天然满足（每层内按 chunk 顺序跑）。
2. **hidden state 常驻**：层主序需要把全部 token 的 hidden state 留在显存（层与层之间传递）：9B 13.4k token × 4096 × 4 B = **220 MB**；1.6k = 26 MB。可接受（当前 ctx=32768 时装载后实测还有 4.1 GB 空闲）。
3. **bit-exact 可保持**：只要 **chunk 边界与大小不变**，每个 GEMM 的 M 参数就不变（cuBLAS 选核/累加顺序只依赖形状）⇒ 层主序只是"同一批 GEMM 换了个执行顺序"，结果应逐位一致。这是本方案最重要的性质：**不需要动 cache 对齐、不需要重新验证数值语义**，用现成的 `gen_regression`（10/10）+ 逐 token 对拍（`ENGINE_SPLIT_PREFILL`）即可验收。
4. **FFN 才有收益**：一层内部对 chunk 的批量化只对"无 token 间依赖"的部分成立。attention（q/k/v/o）在 token 间是因果依赖（但每个矩阵仍是独立的 GEMM，也可以在 chunk 之间批量化，因为 attention 的输入只依赖上一层输出，不依赖同层其它矩阵）⇒ 实际上**整层都可以层主序**，只要保持 chunk 之间的顺序。

### 2.3 两个落地变体

| | V1（per-matrix streaming） | V2（full-layer staging） |
|---|---|---|
| staging 大小 | **最大单矩阵**：9B ffn_down 12288×4096×2 B = **100 MB**（现有 `dequant_scratch_` 容量正好是这个量级，无需新增显存） | **整层**：9B ≈ 380 MB fp16（新增显存） |
| 改动面 | matmul 增加"一个矩阵 × 多 chunk"批量接口；层内计算顺序按矩阵重排 | 同上 + 层权重的整层 staging 生命周期管理 |
| 适用 | 所有 dense 模型；35B MoE 专家矩阵也可（每专家矩阵更小） | 仅 dense 大模型（35B 的 MoE + MLA 不建议） |
| 收益 | 与 V2 相同（收益只取决于"dequant 从每 chunk 一次变成每请求一次"） | 同 V1（V2 只多省一点启动开销） |

**推荐 V1**：复用现有 scratch，VRAM 风险最低，收益与 V2 相同。

### 2.4 与现有机制的关系

- **替换** `prefill_f16` 路径里"每 matmul 一次 dequant"：权重仍以 fp16 tensor core 计算（硬件上限 12.8 T-MAC/s 不变）。
- **不影响** `rows_`/`prefill_rows_max`/自适应 rows：chunk 仍按 VRAM 与 attention 工作集决定，只是不再乘以 dequant 成本。
- **不影响** cache 边界对齐（server-mode 的领域）：chunk 边界/大小不变 ⇒ `reuse_pos=(boundary/chunk)*chunk` 不变。
- **不影响** decode 路径（n=1 走 GEMV/融合内核，不进 prefill 这条线）。

### 2.5 内存账（9B，ctx=32768，实测装载后 free ≈ 4130 MiB）

| 项 | 增量 |
|---|---|
| 单矩阵 fp16 staging（V1） | 0（复用现有 ~100 MB scratch） |
| 全 token hidden state 常驻 | +26 MB（1.6k）/ +220 MB（13.4k） |
| 整层 staging（V2，可选） | +380 MB |
| 结论 | V1 在 ctx=32768 仍有 ~3.9 GB 余量，安全；V2 也可行但不必要 |

---

## 3. tail 的 ~180 ms：更便宜的解法（建议先做）

tail 是"1 个 chunk"，层主序救不了它。它的固定成本里有很大一块是 **launch 数量**（~700 次/forward），而 WDDM 下每次 launch 约 30–100 µs ⇒ 20–70 ms；再加上 dequant ~95 ms。

**小 n CUDA graph**：复用现成的 `prefill_graph` 机制（`src/runner.cpp` 里按 `n_tokens` 捕获整条 prefill 流水线，默认 `prefill_graph=false`；当初实测对大 chunk 无收益 1506 vs 1478 ms，因为大 chunk 是 GPU-bound）。对 n≤128 的 tail 它是 **launch-bound**，预期能把固定成本从 ~180 ms 降一个档次（dequant 仍在，launch 大部分消失）。

设计要点：
1. 新配置 `prefill_graph_small_max`（默认 128，0=关）：只对 `n_tokens ≤ max` 的 forward 捕获图；
2. 按 n 分桶（例如 8/16/…/128 或精确 n），图缓存上限 `prefill_graph_slots`（默认 16）避免无限内存；
3. 捕获前必须预分配 `x_half_`/`dequant_scratch_`（现有代码已这么做）；
4. 位置无关性已验证（位置在 `pos_dev_`，图捕获时不 baked；大 chunk 图当时实测无收益但没有正确性问题）；
5. 与 server-mode 的 cache 路径交互：tail 的 n 由 prompt 决定（变化多），桶化会造成"某个 n 首次请求捕获一次"的成本（捕获 ~5–20 ms），可接受；若不想承担，可只对**最常见桶**（如 n≤128 每 16 一桶）捕获。

---

## 4. 落地步骤与验收

| 步骤 | 工作量 | 验收 |
|---|---|---|
| ① 小 n CUDA graph（`prefill_graph_small_max`） | 1–2 h | 52-token tail prefill 从 ~200 ms 降到 ≤130 ms；`gen_regression` 10/10；逐 token 对拍一致 |
| ② 层主序 V1（dense 模型） | 2–3 h | 13.4k prompt −7~−10%（交错 A/B min-of-3）；`gen_regression` 10/10；`ENGINE_SPLIT_PREFILL` top-1 一致；bit-exact 与现路径对拍 |
| ③ （可选）层主序 V2 / 35B MoE 适配 | +1 天 | 35B 长 prompt prefill 不退化 |

**风险**：
- bit-exact 前提是"chunk 边界不变"——若将来 server-mode 的 cache 解耦需要改变前向 chunk，必须重新对拍（届时保持"同一次请求内 chunk 序列一致"即可）。
- MLA（Coder）与 MoE（35B）路径的中间张量与 dense 不同，V1 首版只对 dense（9B/0.5B）启用，其余走原路径（自动回退，由 `ir_.moe.n_expert==0 && ir_.attn.variant!="mla"` 门控）。
- 需要 instrument 确认固定成本的精确拆分（dequant vs launch vs 其他）：在 `matmul` 里加 `ENGINE_PROF_PREFILL_FIXED=1` 的累计计时（dequant 总时间 / cuBLAS 调用次数 / host 时间）——这一步很便宜，建议和 ① 一起做。

---

## 5. 附：为什么不是"更大 chunk"或"更多 KV 量化"

- 更大 chunk（2048+）在 ctx=32768 下不可行（实测 2048 行激活 2.5 GB + KV 2 GB + 权重 6.26 GB 会贴边/触发换页）；且见 `PREFILL-CLIFF-REPORT` §5.5：chunk 变大会让 prefix-cache 边界变粗（长多轮每轮 +~1 s）。
- KV 量化（q8_0/q4_0）只影响 attention 与显存占用，不改变本文件讨论的"每 forward 整模型 dequant + launch"固定成本。
- 自研量化 GEMM（`ENGINE_GEMM=1`）在小 n 下实测更差（fixed 381 ms），不作为 prefill 主路径。
