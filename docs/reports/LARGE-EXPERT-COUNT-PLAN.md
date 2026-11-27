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
