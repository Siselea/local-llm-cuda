# Planner ↔ pyref/planner.py 一致性（v0.1.10）

> 执行：prefill-fast｜日期：2026-11-27｜范围：`src/planner.cpp`、`src/main.cpp`（`engine plan`）、`pyref/planner.py`、新增 `pyref/placement.py`、`tools/planner_parity.py`
> 规则：C++ 为权威，pyref 对齐 C++；纯 CPU（不占 GPU）。

## 0. 结论摘要

| # | 结论 |
|---|---|
| **真实 C++ 估算错误（已修）** | **预算 planner 的 KV 记账用旧 `kv_bpe` 模型**（2 字节/值），而 runner 实际按 `kv_type`（默认 f32 = 4 字节/值）分配：**非 MLA 模型差 2×，MLA 模型（Coder-V2-Lite）差 17.8×**。同一条 `engine plan` 里 budget 段与 `[placement]` 段自相矛盾（9B：`kv cache 256 MiB` vs `KV 512 MiB`）。这会把 expert_budget 抬高、把 GPU 层数**大幅高估**（Coder @12 GiB/8k：旧 24/26 层 vs 真实 14/26 层）。 |
| **#27 的实质（已收口）** | plan 与 placement 的层数都由"平均专家层大小"改成**按真实逐层专家字节累加**。本机 4 个发布模型上两层大小差异仅 1.15×，修复前后层数恰好一致（所以 #27 之前判 REAL-but-MITIGATED 仍然成立），但现在不存在"平均法在层大小不均匀时估错"的理论缺口，且 plan/decide 用同一规则。 |
| **parity 收掉** | `engine plan --json` 新增 `layer_placement` / 完整 `placement` decision / `suggested_ctx`；`pyref` 新增 `placement.py` 完整镜像。**全矩阵 288/288 逐字段一致**（6 模型 × ctx{2048,8192,32768,131072} × vram{4,8,12,24} GiB × kv{f32,q8_0,q4_0}，host 32 GiB、row-cap 512）。无容差：两边是同一套整数公式，必须精确相等。 |
| **已知边界** | patch 前 plan 层 parity 本来就成立（72/72 组合全 PASS，见 §2），缺的是 placement/建议 ctx 两个"无法对照"的维度；launcher/GUI 用的 `engine info --json` 的 `kv_per_token{f32,q8_0,q4_0}` 本来就是 placement 的真实公式（**GUI 估算一直是对的，错的只有 `engine plan` 的 budget 段**）。 |

## 1. 差异表（修复前 → 修复后）

同一模型、`--ctx 8192 --vram-gib 12 --host-gib 32 --row-cap 512`：

| 字段 | 9B（GQA hybrid） | Coder-V2-Lite（MLA） | 35B-A3B（MoE） |
|---|---|---|---|
| `kv_bytes`（plan budget） | 256.00 MiB → **512.00 MiB**（2×） | 243.0 MiB → **4320.0 MiB**（**17.8×**） | 160.0 MiB → **320.0 MiB**（2×） |
| 与 `[placement]` 段一致性 | 256≠512（矛盾）→ **512=512** | 243≠4320（矛盾）→ **4320=4320** | 160≠320 → **一致** |
| `layers_on_gpu`（ctx 8192, vram 12） | dense 模型无专家 | **24/26 → 14/26** | 19/40 → 19/40（恰好不变） |
| `layers_on_gpu`（ctx 32768, vram 12） | — | **22/26 → 0/26** | 18/40 → 17/40 |
| `layer_placement`（每层） | —（无专家） | 不存在 → 26 项（前 14 gpu / 后 12 cpu） | 不存在 → 40 项 |
| `placement` decision | 仅文本 → JSON 完整对象 | 同左 | 同左 |
| `suggested_ctx` | 不存在 → `{f32, 65536 B/token, max_ctx=54885, …}` | 同左 | 同左 |

影响（`tools/planner_impact.py` 输出）：Coder 在 8 GiB/8k ctx 下旧 plan 建议 14 层 GPU、真实只有 4 层；32k ctx/12 GiB 下旧 plan 22 层、真实 0 层（KV 一项就把 usable VRAM 吃光）。这就是"`engine plan` 显示与实测不符"的实质内容。

## 2. 根因

1. **KV 记账分裂**：`planner::compute(m, ctx, kv_bpe=2)` 用 `Model::kv_bytes_per_token(bpe)`——
   - MLA：`n_kv_layers * (kv_lora_rank + rope_dim) * bpe`（**压缩 latent**，且 bpe=2）；
   - 非 MLA：`n_kv_layers * n_head_kv * (klen+vlen) * bpe`（bpe=2 = 2 字节/值）。
   而 runner 实际分配（`placement::kv_bytes_per_token`，也是 `engine info --json` 给 GUI 的公式）：MLA **per-head 展开 K/V × 4 字节**、非 MLA × `kv_type` 字节（默认 f32=4）。两者在 9B 差 2×、在 Coder（MLA）差 17.8×。
2. **平均 vs 逐层（#27）**：`compute` 与 `placement::decide` 都用 `budget // avg_layer` 估层数，层大小不均匀时会估错落点。
3. **pyref 覆盖缺口**：`pyref/planner.py` 只有 budget plan；没有 `placement::decide`、`estimate_ctx`（建议 ctx）的参考实现；`engine plan --json` 也不输出 placement/每层/建议 ctx ⇒ 这三类数字无法对照。

patch 前 plan 层 parity 已成立（6 模型 × 3 ctx × 4 vram = 72 组 `tools/diff_plan.py` 全 PASS），所以本轮不是"修分歧"，而是：修真实估算错误 + 把无法对照的维度补齐并锁死。

## 3. 修复（C++ 权威）

| 文件 | 改动 |
|---|---|
| `include/engine/planner.h` | `PlacementPlan`：`kv_bpe` → **`kv_type` + `kv_name`**；新增 `layer_placement`（MoE 层→gpu/cpu）；`compute(m, ctx, kv_type, …)` |
| `src/planner.cpp` | `compute`：KV 改用 **`placement::kv_bytes`**（单一来源）；`layers_on_gpu`/`layer_placement` 改为**逐层累加**；`slots_per_layer` 保留为"uniform advisory"并注释；note 文案 `uniform placement:` → `expert placement:`。`placement::decide` 的 `gpu_moe_layers` 同样逐层累加 |
| `src/main.cpp` | `engine plan`：新 flags `--kv-type {f32,q8_0,q4_0}`、`--host-gib`、`--row-cap`；`--kv-bpe` 移除并给出明确报错（其模型低报 2×）；`--json` 增加 `kv_type`/`kv_name`/`layer_placement`/`placement`（完整 decision：kind、gpu_moe_layers、全部字节字段、reason、notes）/`suggested_ctx`（estimate_ctx）；text 输出增加每层放置与 `[suggested]` 行 |

## 4. pyref 对齐

| 文件 | 改动 |
|---|---|
| `pyref/placement.py`（新） | `placement::` 命名空间的完整镜像：`kv_layer_count` / `kv_bytes_per_token` / `kv_bytes` / `activation_bytes` / `default_host_reserve` / `fmt_bytes` / `kind_name` / `MemoryInfo` / `CtxFit` / **`estimate_ctx`** / **`decide`**——公式与 `reason`/`notes` 字符串逐字节一致 |
| `pyref/planner.py` | `plan(ir, ctx, kv_type, …)`；KV 用 `placement.kv_bytes`；逐层累加 + `layer_placement`；`to_json` 含 `kv_type`/`kv_name`/`layer_placement` |
| `pyref/cli.py` | `plan` 同 C++ flags（`--kv-type/--host-gib/--row-cap/--vram-margin-mb/--workspace-mb/--host-margin-mb`）；JSON 与 C++ 同键（含 `placement`/`suggested_ctx`）；无 `--host-gib` 时用 ctypes 探 host RAM（无第三方依赖） |

## 5. Parity 测试

- **`tools/planner_parity.py`**（新）：矩阵跑 `engine plan --json` vs `pyref plan --json`，递归逐字段比较。
  - 默认 quick：6 模型 × ctx{2048,8192} × vram{8,12} × kv{f32} = **24/24 PASS**
  - `--full`：6 模型 × ctx{2048,8192,32768,131072} × vram{4,8,12,24} × kv{f32,q8_0,q4_0} = **288/288 PASS**
  - 6 模型 = 4 个发布模型（9B、Coder-V2-Lite、Qwythos-9B、35B-A3B）+ 2 个合成 tiny MoE（Q8_0 / F16 专家组，`build/fixtures/tiny_gguf_moe*`）
- **`tools/diff_plan.py`**：改为同一比较核心的单模型 M2 gate（模拟内存，确定性），4 个发布模型均 `PASS - plan identical (22 fields)`。
- **接入 `tools/verify.sh`**：M2 段保留逐模型 `diff_plan`，并新增 `planner_parity` quick 矩阵（`MODEL_DIR` 未设置/模型缺失时 skip；CI 无模型 → 不阻塞）。
- **比较口径**：数字统一按 float 比较（C++ JSON 输出 `0` 而 Python 输出 `0.0` 不算差异）；Windows 路径分隔符归一化；bool 严格（不参与数值比较）；列出两端缺键。
- **容差**：**0**。两边是同一套整数公式（KV/activation/expert 字节、逐层累加、ctx 拟合迭代），应精确相等；出现任何差异都按回归处理。运行时放置仍以装载时**实测 free VRAM** 为准（`expert_layer_fits_vram`、`388106a`），与 planner 的模拟数字允许不同——这是"计划 vs 实测"的设计边界，不在 parity 范围。

## 6. 验证

| 项 | 结果 |
|---|---|
| `bash build.sh`（含 C++ 改动） | ✅ `built build/engine.exe` |
| Python 单元测试 `unittest discover -s tests` | ✅ 21 tests OK |
| `planner_parity` quick | ✅ 24/24 |
| `planner_parity --full` | ✅ 288/288 |
| `diff_plan` × 4 发布模型 | ✅ 全部 PASS（22 字段） |
| 错误分支 sanity | ✅ 9B ctx32768 vram2 → `ssd stream`（dense 不支持 offload）；35B 4 GiB → `2/40` 层 GPU 且 `--n-cpu-moe 38`；`--kv-type q8_0` 正常 |
| 全量 `verify.sh` | 本轮按"CPU-only、不占 GPU"要求**未运行**；parity 已接入该脚本，发布 gate 会覆盖 |
| 修复前证据 | 同一条 `engine plan` 输出 `kv cache 256 MiB` + `[placement] … KV 512 MiB`（9B）与 `243 MiB` + `4320 MiB`（Coder），见 §1 |

## 7. 残余 / 后续

- **旧 KV 语义已在本版清理**（消除"两套 KV 记账"这个下一个 bug 的温床）：删除 C++ `Model::kv_bytes_per_token(bpe)`（`ir.h`/`ir.cpp`）与 `engine info --json` 的 `kv_per_token_bpe2` 字段，pyref 的同名方法与 JSON 字段一并删除；两个 Python 单测改为断言真实 f32 记账（`placement.kv_bytes_per_token(ir, 0)`），`tools/planner_impact.py` 为保留历史对比把旧公式内联（已注释）。无外部消费者：`tools/diff_ir.py` 的 G0 不比较该字段（复跑 G0 PASS），launcher/GUI 用的是 `engine info` 的 `kv_per_token{f32,q8_0,q4_0}`（= placement 真实公式）→ **GUI 估算一直正确**。
- `plan.layers_on_gpu` 与 `placement.gpu_moe_layers` 的 reserve 语义不同（plan 用 `DEFAULT_VRAM_RESERVE`=1024 MiB，placement 用 `vram_margin + workspace`=1968 MiB 默认），因此同一模型可能给出不同层数——**故意保留**（前者是预算估算，后者是启动决策），已在报告中注明。
- 运行时仍是最终权威：`expert_layer_fits_vram` 逐层实测 + margin/ledger-gap 检查（`8199927`/`388106a`）；planner 现在是"乐观上限"估计，误差被实测纠正。
- `tools/planner_impact.py` 是本次的量化脚本（保留，供复查数字）。

## 8. 变更文件

```
src/planner.cpp            include/engine/planner.h     src/main.cpp
pyref/placement.py (new)   pyref/planner.py             pyref/cli.py
tools/planner_parity.py (new)  tools/planner_impact.py (new)  tools/diff_plan.py
tools/verify.sh
docs/reports/PLANNER-PARITY.md (本文)
```

commit：本轮提交（`git log --grep planner` 可见）。
