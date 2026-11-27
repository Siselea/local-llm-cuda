# REVIEW-34 对应段落：Gated DeltaNet 几何（外部审查 #11 / #13）

日期：2026-11-27
范围：外部审查清单第 11 条（`kernels.h`：Gated DeltaNet 卷积状态内核只处理 K=4）与第 13 条
（`qwen35.cpp` 运行期：头几何缺整除性校验）的**核实、判定、修复与测试**。
对应内部审计 `docs/reports/KERNEL-CORRECTNESS-AUDIT.md` 的 D6/D7。

---

## #11 Gated DeltaNet 卷积状态内核只处理 K=4

### 核实（成立）
- `include/engine/kernels.h` `gd_conv_state`（v0.1.7：`if (K != 4) return;`）只实现 K=4 的移位；
  K≠4 时**静默返回**，conv 历史不更新 ⇒ 下一 chunk 的卷积读到旧状态，输出错误（不崩溃）。
- `gd_ssm_prep`（decode 融合路径）在 v0.1.7 里用 `hist[4]`，但卷积循环对 `j >= -(K-1)` 读 `hist[-j-1]`：
  **K ≥ 6 时栈上越界读（UB）**（D6）。
- 其它相关环节对 K 是通用的：`gd_conv1d_silu` 按 K 循环；`conv_state` 缓冲按 `(K-1)*C` 分配
  （`qwen35.cpp` load 处）。

### 支持的模型实际 K 值（GGUF metadata）
| 模型 | `ssm.conv_kernel` | `state_size` | `group_count` | `time_step_rank` | `inner_size` |
|---|---|---|---|---|---|
| Qwen3.5-9B-Q5_K_M | **4** | 128 | 16 | 32 | 4096 |
| Qwythos-9B-…-MTP-Q4_K_M | **4** | 128 | 16 | 32 | 4096 |
| Huihui-Qwen3.6-35B-A3B-…-Q4_K_M | **4** | 128 | 16 | 32 | 4096 |

⇒ 当前所有受支持模型 K 均为 4，**没有非 4 的真实模型**。

### 判定与修法（选择 b：显式拒绝；D6 提前已修）
- **D6**：v0.1.8 已修（commit `dd1c61d`）：`gd_ssm_prep` 用 `hist[8]` + `K>9` 防御返回，
  `qwen35.cpp` fused 路径 host 门限加 `K <= 8`。
- **#11 主体**：按审查建议选"显式拒绝"——在 `qwen35.cpp` 的**加载期**（`load_q35_linear`，
  每个线性层加载时调用，开销可忽略）验证 `conv_kernel == 4`，否则抛
  `Gated DeltaNet conv kernel K=N unsupported (this build implements K=4 only); refusing the
  model instead of computing a wrong recurrent state`。
  - 理由：K=4 是唯一实测模型；通用化（模板/switch 展开）虽可行（`gd_conv_state` 通用移位 ~15 行，
    `gd_ssm_prep` 的 hist 循环已支持 K≤9），但会引入未经任何真实模型验证的数值路径，
    而审查要求的底线是"不再静默错误"。显式拒绝 + 文档是当前风险最低的修法。
  - 若后续出现 K≠4 的 qwen35 家族模型：把 `gd_conv_state` 改成通用移位（new_state[a] =
    x[T-1-a] 若 ≥0 否则 old_state[a-T]），并把本校验放宽到 2..9；`conv_state` 分配与
    `gd_conv1d_silu` 已是通用的。

### 测试
`tests/test_q35_geometry.cpp`：K=2/3/5/6 均在加载期被拒绝（错误信息含 `K=<n>`）；K=4 通过；
非 gated_delta 层不触发校验。`tests/test_kernel_bounds.cpp` 的 D6 覆盖由 `verify.sh` 的 q35 conv 一致性
测试与五套模型验证承担（K=4 真实路径）。

---

## #13 qwen35 运行期头几何假设（缺整除性校验）

### 核实（成立，且有真实的风险点）
`qwen35.cpp` 运行期直接从 metadata 取几何并做整数除法/映射：
- `rep = n_value_head / n_key_head`，HF 容器用 `kh = h / rep`（**要求整除**；不整除时截断 ⇒ 头映射错）；
  GGUF 用 `kh = h % n_key_head`（block repeat，无需整除）。
- `kd = n_key_head * key_head_dim`、`vd = n_value_head * value_head_dim`、`conv_dim = 2*kd + vd`
  是 conv/q/k/v 切分依据；metadata 若 `inner_size % time_step_rank != 0`，IR 的
  `value_head_dim = inner / tsr` 会**截断**，随后 conv 切分整体错位。
- delta-rule 通用内核 `gd_delta_rule`：block=VD 线程、shared `sk[KD]`/`sq[KD]` 需 `KD <= VD`；
  `VD > 1024` 无法启动；专用 128/128 内核仅在 `KD==VD==128` 时启用（host 已 gated）。
- 全注意力层 `run_attn`/GQA 代码用 `rep = n_head / n_kv`（不整除则静默截断，少的头不计算/重复）。
- v0.1.7 的 `src/ir.cpp` 只校验了 `key/value head dim > 0` 与 rope 关系，**没有**校验上述整除/一致性
  （IR 加载期校验由另一路负责，本次按分工在运行期模块补齐）。

### 支持的模型实际几何（全部通过下列校验）
| 模型 | n_key_head | n_value_head | key/value head dim | key_dim | value_dim | conv_dim |
|---|---|---|---|---|---|---|
| 9B / Qwythos | 16 | 32 | 128 / 128 | 2048 | 4096 | 8192 |
| 35B | 16 | 32 | 128 / 128 | 2048 | 4096 | 8192 |
| 全注意力 | `head_count=16`, `head_count_kv=4`（9B/Qwythos）、`2`（35B） | — | 256 / 256 | — | — | — |

### 修复（加载期 fail-fast，不进入运行期算错）
`qwen35.cpp` 新增 `runner::q35_geometry`：
- `validate_ssm(S, gguf_container)`（`load_q35_linear` 调用）：K==4；`n_key_head/n_value_head > 0`；
  `key/value_head_dim > 0`；`value_head_dim <= 1024`；`key_head_dim <= value_head_dim`（通用内核 staging）；
  HF 容器要求 `n_value_head % n_key_head == 0`；`key_dim == n_kh*KD`、`value_dim == n_vh*VD`
  （抓 metadata 截断）；`conv_dim == 2*key_dim + value_dim`（conv 宽度与 q|k|v 头几何一致）。
- `validate_attention_heads(n_head, n_kv)`（`load_q35_attn` 调用）：两者 > 0；`n_head % n_kv == 0`
  （GQA 组映射整除，否则会静默丢头）。
- 全部抛 `std::runtime_error` 并带具体数值，加载期即失败（`test_model`/`engine run` 打印 `ERROR: ...` 并退出）。

### 测试
`tests/test_q35_geometry.cpp`（16 个断言）：
- shipped 几何（16/32、128/128、K=4、conv_dim 8192）GGUF 与 HF 都通过；
- HF 24/16 不整除被拒；GGUF block-repeat 允许不整除（与 llama.cpp 语义一致）；
- 0 头、0 head dim、VD>1024、KD>VD、value_dim/key_dim 与头数不一致、conv_dim 不一致、K 非 4 全部被拒；
- 注意力 16/4、16/2 通过，16/3、0/x 被拒。

---

## 复现方式

```
bash build.sh
./build/test_q35_geometry.exe      # #11/#13 的加载期校验
./build/test_kernel_bounds.exe     # v0.1.8 的 kv_split/attn smem/device-route 回归
```

## 残留风险 / 后续
- 若未来引入 K≠4 的 qwen35 家族模型，需要按 §#11 的通用化步骤实现并重新验证（当前明确拒绝）。
- 全注意力 `n_head % n_kv` 校验在 q35 加载路径上；若其他 arch 复用同一 run_attn 代码，
  建议把该校验提到 IR 层（`src/ir.cpp`，由对应负责人排期），本段按分工只覆盖 qwen35 运行期。
- 外部审查 #13 提到的"conv kernel 与 head_dim 关系"已用 `conv_dim == 2*key_dim+value_dim` 覆盖。
