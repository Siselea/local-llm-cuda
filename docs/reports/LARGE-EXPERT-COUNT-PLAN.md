# >256 专家（Kimi-K2 384 等）支持设计（v0.1.9，待排期）

状态：**设计文档，不改代码**。作者：edge-kernels；触发：v0.1.7 内核审计 D2 的后续。
相关现状：`src/ir.cpp:581` 在加载期硬拒绝 `expert_count > 256`；`kernels.h` 的 `moe_route`
（device-side router）用栈上 `float pr[256]` 且 `if (ne > 256) return;`；host-side 路由
路径本身对专家数没有硬上限。

---

## 1. 现状核实

| 环节 | 现状 | 是否受 256 限制 |
|---|---|---|
| IR 校验 | `src/ir.cpp:581` `expert_count > 256` → 报错 "not supported"（test_model exit 2） | **硬限制**，模型直接拒载 |
| `moe_route`（device router） | `float pr[256]` 栈数组；`if (ne > 256) return;` 不写 slot | **硬限制**（当前因 IR 拒载不可达） |
| 路由落盘 | `slot_exp_/slot_tok_/slot_w_` 大小 = n_tokens×topk（与 ne 无关） | 否 |
| `moe_gateup_*/moe_down_*` | blockIdx.y = (token,expert) slot，与 ne 无关 | 否（v0.1.9 已加 grid.y 分块） |
| host 路由 | host 侧遍历 ne 个 logits 选 topk；`d_idx_/d_w_` 大小 = rows×topk | 否 |
| `router_logits_` | rows × ne × 4 B（4096 行 × 384 专家 = 6 MB，可接受） | 否（内存随 ne 线性） |
| 放置/专家并集 | `gpu_n`/`all_gp_`/per-layer 指针表按 ne 循环；`gpu_experts` 选取子集 | 需逐处核对（未发现 256 常量，但要跑通） |
| `moe_gateup_silu_dev`/`down` | 用 slot_exp[slot] 查专家指针表 | 否（受 IR 可达性保护） |

结论：**功能上唯一的结构性限制是 `moe_route` 的 256**；其余是容量/校验/内存账与测试覆盖。

## 2. 改动方案

### 2.1 `moe_route` 内核通用化（核心，~2–3 小时）
- 把 `float pr[256]` 改为动态共享内存：`extern __shared__ float pr[];`（host 传 `ne*4` 字节）
  或分块（每块 256 个专家做 partial top-k + 合并）。**推荐前者**：ne ≤ 12288 时 smem ≤ 48 KiB，
  Kimi-K2 384 仅 1.5 KiB；top-k 选择是单线程 O(ne×topk) ≈ 384×8 ≈ 3k 次比较（~µs 级），
  不需要并行化。
- 删除 `if (ne > 256) return;`；保留 `ne > 1114112`（smem 上限）的明确错误。
- 注意 `moe_route` 当前的 `router_sigmoid`/`norm_topk` 分支都用 `pr[]`，改为传入的指针即可；
  kernel 仍在 block 0 thread 0 串行执行，保持 bit-exact（选择序不变）。
- 新增 `kMaxDeviceRouteExperts` 常量（`runner::` 现有 256 的守卫改为动态上限），并把
  `device_route_supported(ne)` 更新为 smem 上限（如 12288 或 4096）。

### 2.2 IR 校验放宽（`src/ir.cpp`，~30 分钟）
- `expert_count > 256` → `expert_count > kMaxDeviceRouteExperts`（或先放宽到 1024，
  再按 smem 上限动态判断）。
- 同时校验放置/并集容量：`gpu_experts` 子集、`all_gp_` 指针表、每层 host gate 数组
  的尺寸都按 ne 计算（现有代码用 ne 循环，预计只需跑通测试）。
- 建议对新上限加 warning：「expert_count=N > 256: device router 使用动态共享内存，
  host 路由回退仍可用；性能验证见 docs/reports/LARGE-EXPERT-COUNT-PLAN.md」。

### 2.3 内存/放置账（~2–4 小时）
- `router_logits_` = rows×ne×4（4096×384 = 6 MB）；
- per-layer `all_gp_/all_up_/all_dp_` 指针表 = ne×8 B（3 KB），可忽略；
- host 路由的 `host_gate`/`h_gate_bytes` 等若有按 256 的固定数组，需要改动态
  （grep `256` in placement/moe load paths 逐处确认）；
- 放置决策的 `gpu_experts` 子集并集逻辑（gpu/cpu 混合）要加 ne>256 的合成模型测试。

### 2.4 性能（可接受，需实测）
- device 路由的收益是省掉每层一次 host round-trip；ne 增大只增加 router 的 µs 级 O(ne)
  选择，不会影响 MoE GEMV（每 token 仍只跑 topk 个专家）。
- host 路由回退路径已存在，Kimi-K2 在 12 GB 上大概率大部分层走 CPU 专家 + host 路由，
  性能由 CPU GEMV/带宽决定，不由本改动决定。
- 35B（ne=256）必须**逐位不变**：moe_route 选择序与 slot 内容保持相同（仅数组来源从
  栈改共享内存）。

## 3. 验收方法

1. **合成模型**：`tests/make_tiny_gguf_moe.py` 已支持 `N_EXPERT` 环境变量
   （v0.1.8 引入），生成 `build/fixtures/tiny_gguf_moe300` / `tiny_gguf_moe384`
   （含 `ref_logits.f32`）。
2. **测试矩阵**（新增 `tests/test_large_experts.cpp` 或扩展 `test_gguf_moe_offload`）：
   - `ENGINE_GPU_EXPERTS=0`（全 host 专家）：加载 + 参考 logits 一致；
   - `ENGINE_GPU_EXPERTS=N`（全 GPU 专家）：device route 开关（`ENGINE_DEV_ROUTE=0/1`）
     两种都应与参考一致（选择序相同；若 device 路径数值语义未验证则先只开 host）；
   - 混合放置（GPU 一半）：host 路由 + GPU 专家 GEMV 路径；
   - `test_model.exe build/fixtures/tiny_gguf_moe384`：exit 0 + logits 校验通过；
   - 回归：现有 `tiny_gguf_moe`（4 专家）与 35B（256 专家）的 gen_regression / smoke 不变。
3. **五套验证**：`tools/check_clean_build.sh`（build/verify/smoke/gen_regression/generality/serve）。
4. **性能 A/B**：35B decode 同二进制前后（应完全一致）；Kimi-K2 类模型（若到手）测
   tok/s 与 llama.cpp 对比。

## 4. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 35B 边界回归 | ne=256 恰好是旧上限；改动态共享内存后 slot 内容必须完全一样 | 用 35B gen_regression + 逐位 A/B |
| smem 上限 | `ne*4` 动态共享内存 >48 KiB 时 launch 失败 | 上限校验 + 明确错误；opt-in 不必要（12288 专家内） |
| 内存账 | router_logits 与放置账可能假设 rows×256 | 逐处 grep 256/固定数组，合成 384 模型覆盖 |
| device 路由数值 | device 路由对某些量化类型与 host 路由有已知数值差异（默认只对已验证布局开启） | 保持 `dev_route_default_` 的按类型/布局开关，ne>256 默认 host 路由 |
| 放置/并集复杂度 | gpu/cpu 混合并集逻辑对 ne 无上限假设但需测试 | 384 专家混合放置测试矩阵 |

## 5. 工作量估算

| 阶段 | 内容 | 估时 |
|---|---|---|
| A | `moe_route` 动态 smem + 上限常量 + host `device_route_supported` 更新 | 2–3 h |
| B | IR 校验放宽 + 放置/并集逐处核对（含 256 常量清理） | 2–4 h |
| C | 合成 300/384 专家 fixture + test_model 测试矩阵 | 2 h |
| D | 35B/9B/Coder 回归 + 性能 A/B + 五套 gate | 2–3 h |
| 合计 | | **约 1–1.5 人日** |

建议排期为下个发布窗口的独立任务；改动不动 MoE GEMV/attention 内核，风险集中在路由/放置层。


---

## 6. 实施状态

### 已完成（v0.1.10，步骤 1：host-route fallback）
- **IR 校验放宽**（`src/ir.cpp`）：`expert_count > 256` 不再报错；**≤ 4096 的模型加载期给出 warning**
  （"device-side router is disabled ... host-side routing is used"），>4096 仍拒绝。
- **路由选择**：`runner::kMaxDeviceRouteExperts` 保持 256，`device_route_supported(ne)` 对 >256
  返回 false ⇒ 自动走 host 路由；host 路由本身对专家数无上限（只按 topk 扫描 ne 个 logits）。
- **测试**（`tools/verify.sh`）：
  - 生成 `build/fixtures/tiny_gguf_moe300` / `tiny_gguf_moe384`（`N_EXPERT` 环境变量）；
  - `test_gguf_moe_offload` 新增可选专家数参数，对 300/384 跑 all-GPU / all-CPU / mixed 放置矩阵，
    全部与 numpy 参考 logits 对拍；
  - `test_model.exe` 三种放置（默认 / 全 CPU / 全 GPU）加载 + 参考 logits；
  - `test_gen.exe` 30-token greedy 生成，在 `ENGINE_DEV_ROUTE=1` 与 `=0` 下逐位一致。
- **现有模型零回退**：ne=256（35B）仍走 device router（≤256），ne=64（Coder）同理；见五套 gate。

### 待做（步骤 2：device router 扩到 >256，性能优化）
按 §2.1 把 `moe_route` 的 `float pr[256]` 改为动态共享内存（`ne*4` 字节 ≤48 KiB，384 专家仅 1.5 KiB），
删除 256 硬返回，`kMaxDeviceRouteExperts` 随之提升；然后比较 ne=384 时 device vs host 路由的
decode 速度与 token 一致性，若 device 无优势则默认保持 host 并文档化。

### 步骤 1 实测结果（2026-11-27，worktree `feat/large-experts`）
- 合成 **300 / 384 专家** tiny MoE（Q8_0 experts，2 层 topk=2）：
  - `test_model` 三种放置（默认 auto / `ENGINE_GPU_EXPERTS=0` / `ENGINE_GPU_EXPERTS=ne ENGINE_DEV_ROUTE=1`）
    **全部 PASS**（rel_l2 ≤ 4.4e-05，argmax 一致；后两者 rel_l2 ≈2.4e-07）；
  - `test_gguf_moe_offload` 放置矩阵（all-GPU / all-CPU / mixed 1-GPU）**PASS**（host=300/384 计数正确）；
  - `test_gen` **30-token greedy** 在 `ENGINE_DEV_ROUTE=1` 与 `=0` 下**逐 token 相同**。
- 零回退：`smoke_models` 四模型官方 token 全对；`gen_regression` **10/10 GREEN**（9B×5、35B×3、Coder、0.5B）。
- `tools/verify.sh`（含新用例与全部 test_*、IR/M2 parity）**ALL GREEN**。
- `test_kernel_bounds` 单跑 GREEN 且无残留进程（新增 `KERNEL_BOUNDS_CHILD` 递归守卫；见 D11 fork-bomb 教训）。

### 下一步（步骤 2）
`moe_route` 动态共享内存 + 提高 device 上限，按 §2.1 实施；若 ne=384 时 device 路由无速度优势，
保持默认 host 路由并在此文档记录 A/B 数据。

---

## 7. 实施状态（步骤 2：device router 扩到 >256，v0.1.10）

### 改动
- `kernels.h` `moe_route`：`float pr[256]`（栈）→ `extern __shared__ float pr[]`（动态共享内存），
  删除 `if (ne > 256) return;`。host 按 `ne*sizeof(float)` 传 smem。
- `runner.h`：`kMaxDeviceRouteExperts` 256 → **4096**（= IR 专家数上限）；`device_route_supported(ne)`
  对 257..4096 返回 true。
- `src/runner.cpp`：`moe_route` launch 传入 `ne*4` 字节动态共享内存。
- `src/ir.cpp`：>256 的 warning 改为"device router 用动态共享内存；专家不全在 GPU 时仍走 host 路由"。
- `tools/verify.sh`：>256 用例增加 `ENGINE_NO_DEV_ROUTE=1`（host 路由）与 `ENGINE_DEV_ROUTE=1`
  （device 路由）两条 logits 对比；30-token 生成 parity 现在真正比较 device vs host 路由。

### 内存账（ne 个专家）
| 项 | 公式 | ne=384 | ne=4096（IR 上限） |
|---|---|---|---|
| `moe_route` 动态共享内存/block | `4*ne` B | 1.5 KiB | 16 KiB（默认 48 KiB 限制内，无需 opt-in） |
| `router_logits_` | `rows*ne*4` B | rows=4096 → 6 MiB | rows=4096 → 64 MiB |
| 路由槽位 `slot_exp_/slot_tok_/slot_w_` | `n_tokens*topk*4` B ×3 | 与 ne 无关 | 与 ne 无关 |
| host 路由回退 | 无额外 device 内存（D2H logits + host 向量） | — | — |

### 验收实测（合成 300/384 专家 tiny MoE，Q8_0 experts，2 层 topk=2）
- **输出一致性**：`ENGINE_DEV_ROUTE=1`（device）vs `ENGINE_NO_DEV_ROUTE=1`（host）
  **30-token greedy 逐 token 相同**（300/384 均 PASS）；两种路由单独 vs numpy 参考 logits 均 PASS
  （device rel_l2≈2.4e-07/2.6e-07，host rel_l2≈4.4e-05/2.6e-07）。
- **速度（decode ms/token，6 次交替取中位数；tiny 模型噪声大）**：
  | ne | device 中位数 | host 中位数 | device 优势 |
  |---|---|---|---|
  | 300 | 0.61 ms | 0.73 ms | ~16% |
  | 384 | 0.86 ms | 1.05 ms | ~18% |
  结构性收益 = 每 MoE 层省一次 host round-trip；tiny 模型绝对量小、噪声 ±30%，
  真实宽专家模型（若到手）应更明显。默认策略保持"全部专家 GPU 且量化布局已验证时用 device 路由"，
  否则 host 路由。
- `test_gguf_moe_offload` 300/384 放置矩阵 PASS（host 计数正确）。
- `test_kernel_bounds` GREEN（`device_route_supported` 新边界：257/300/384/4096 true、4097 false）。

### 剩余
- 真实 Kimi-K2 类模型（若到手）验证：加载 + 生成 + device/host 路由 A/B + 显存账。
- 若要在 IR 上限之上继续扩（>4096 专家），只需提高 `kMaxDeviceRouteExperts` 并选择 smem opt-in
  （>48 KiB 需 `max_dynamic_shared_mem`），host 路由已无上限。
