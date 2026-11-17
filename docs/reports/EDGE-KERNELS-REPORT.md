# 9B 边缘内核优化报告（EDGE-KERNELS）

日期：2026-11-15 · 目标：Qwen3.5-9B decode 26.4 ms/token → ≤23 ms/token（≥43 tok/s），
四模型预测与三套验证保持全绿，其它模型不回退。

**结论：9B decode 26.07 → 22.22 ms/token（38.36 → 45.00 tok/s，−14.8 %），超目标。
同一二进制上"新路径全开 vs 全关"的交错 A/B 是 25.70 → 22.22 ms（−3.48 ms，−13.5 %）。
四模型全部提速，预测全对，三套验证全绿。**

---

## 0. 一句话账本

| 项 | 省下 | 手段 |
|---|---|---|
| `gd_delta_rule` 寄存器分块重写（`gd_delta_rule_128`） | **−2.57 ms** | 状态布局改 `[n_vh][KD][VD]`（参考实现布局）+ 4 路寄存器分块，2 MB state 每 token 只读一次写一次 |
| SSM 链融合（`gd_ssm_prep` + `gd_delta_rule_128f`） | **−0.59 ms** | conv1d+silu+写 q/k/v+conv-state+beta/g 合成 1 发；q/k L2 norm 与 gated RMSNorm 折进 delta kernel |
| int8 激活 staging 复用（`quant_reuse`） | −0.04 ms | 连续 dp4a GEMV 共用同一份 `quant_x_i8`（201 → 129 次/token） |
| logits 回传走 pinned 暂存 | −0.06 ms | pageable `cuMemcpyDtoH` 1 MB → pinned + `DtoHAsync`（0.34 → 0.28 ms） |
| **合计（单项 A/B 中位之和）** | **−3.26 ms** | 同二进制全开 vs 全关交错 A/B：**−3.48 ms**；改动前/后二进制：**−3.85 ms** |

---

## 1. 9B decode 前后（每配置 5–7 次独立运行取中位；GPU 空闲、无并发任务）

对照用的是**改动前的同一二进制快照** `tools/util_bin/test_gen_new.exe`（sha256
`3314ad72…`，与 UTILIZATION-REPORT 附录 C 的 "new build" 同源）。

| 配置 | 每次 ms/token | 中位 | tok/s |
|---|---|---|---|
| 改动前二进制 `test_gen_new.exe`（5 次） | 26.08 / 25.64 / 26.19 / 26.07 / 25.79 | **26.07** | 38.36 |
| 改动后 `build/test_gen.exe`（7 次） | 22.28 / 22.19 / 22.54 / 22.17 / 22.21 / 22.22 / 22.28 | **22.22** | **45.00** |
| 改动后二进制，三条新路径全关（7 次） | 26.05 / 25.72 / 25.68 / 25.99 / 25.57 / 25.59 / 25.70 | **25.70** | 38.91 |

> 前两行是"改动前/后"的最终口径（−3.85 ms，−14.8 %）；第三行是**同一二进制**把
> `ssm_fuse`/`gd_delta_128`/`quant_reuse` 全关掉的结果，与全开对比 −3.48 ms（−13.5 %），
> 完全排除了二进制版本差异带来的干扰。

- 40 token 长解码：**22.15 ms/token（39 步）**，生成序列与"全部新路径关闭"逐 token 相同
  （`2614 513 279 1118 220 16 15×35`）。
- 目标 ≤23.0 ms/token（≥43 tok/s）达成，余量约 0.7 ms。

### 1.1 每层 / head 拆分（graph 模式 `--layers` 差分）

W(N) = N·p + H，W(1)=3.81 ms、W(16)=12.90 ms ⇒ p = **0.606 ms/层**、H = **3.20 ms**。
32·p + H = 22.6 ms ✓ 与实测 22.3 吻合。head 里 lm_head 自身约 2.6 ms（834 MB @ ~319 GB/s），
其余 ~0.6 ms 是 final norm + logits 回传 + 采样。层内 183 MB 权重 / 0.606 ms ≈ **303 GB/s**，
已贴近孤立的 Q5_K/Q6_K dp4a GEMV 实测 295–319 GB/s ⇒ **GEMV 侧已无空间，剩下的都在边缘内核**。

### 1.2 `ENGINE_KPROF` 边缘内核表（`--no-graph`，数值含 event/气泡膨胀，只看相对变化）

| 内核 | 改动前 ms / 次数 | 改动后 ms / 次数 | 说明 |
|---|---|---|---|
| `gd_delta_rule` → `gd_delta_rule_128f` | **3.28 / 24** | **0.43 / 24** | 7.6×（融合后含 q/k norm + gated norm） |
| `quant_x_i8` | 0.97 / **201** | 0.64 / **129** | 复用 + 融合后少 72 次 |
| `gd_conv1d_silu` | 0.69 / 24 | — | 并入 `gd_ssm_prep` |
| `gd_norm_qk` | 0.64 / 24 | — | 折进 delta kernel prologue |
| `q35_beta_g` | 0.48 / 24 | — | 并入 `gd_ssm_prep` 的额外 block |
| `gd_norm_gate` | 0.11 / 24 | — | 折进 delta kernel epilogue |
| `q35_split3` | 0.18 / 24 | — | conv 直接写 q/k/v |
| `gd_conv_state` | 0.09 / 24 | — | 并入 `gd_ssm_prep`（列分区，无竞争） |
| `gd_ssm_prep`（新） | — | **0.12 / 24** | 1 发替 4 发 |
| 内核种类 | **22** | **17** | |
| 每 token 内核发射数（kprof 计数） | **891** | **699** | −21.5 % |
| 边缘内核合计（kprof 口径） | **8.98 ms** | **3.78 ms** | −5.2 ms（kprof 膨胀 ≈1.3×，折算真实 ≈ −3.9 ms） |

---

## 2. 做了哪几项（每项实现要点 + 省多少）

### 2.1 `gd_delta_rule_128`：gated delta rule 重写（−2.57 ms，最大头）

**问题**：通用 kernel 状态布局 `[n_vh][VD][KD]`，一线程一 value dim，内层对 key dim 循环 ⇒
一个 warp 的 32 个 lane 每次读 32 条不同的 128 B 行（完全不合并）；2 MB state 反复穿过 L1。
隔离实测 **0.136 ms/次**（24 次/token），是最大的非 GEMV 内核。

**改法**（`include/engine/kernels.h::kGdDelta128`）：

- 状态布局改成 `[n_vh][KD][VD]`（= qwen35 参考实现 `einsum("hkv,hk->hv")` 的天然布局），
  内层对 key dim 循环时 value dim 连续 ⇒ 每 warp 每次读 128 B 整行；
- block = 4×128 = 512 线程：`(grp, j)` 线程把自己负责的 32 个状态值留在**寄存器**里，
  key dim 4 路分块 + shared 归约 ⇒ **state 只读一次、写一次**（4 MB 流量）；
- 只对 `KD==VD==128` 启用（本机三个 qwen35 模型都是 128/128），其余仍走通用 kernel；
  state 只由本 kernel 读写，初始只上传全零 ⇒ 布局切换是局部的。

**结果**：隔离 **0.136 → 0.0125 ms/次（10.9×，≈ 4 MB / 12.5 µs ≈ 320 GB/s 贴顶）**，
状态 rel_l2 = **6.8e-9**、输出 rel_l2 = **2.1e-7**（vs 原 kernel，微基准）。

### 2.2 SSM 链融合（−0.59 ms，内核数 24 层 × 6 → × 2）

每个线性注意力层原来跑：`gd_conv1d_silu → gd_conv_state → q35_split3 → q35_beta_g →
gd_norm_qk → gd_delta_rule → gd_norm_gate` 七个串行小内核（每层约 0.30 kprof ms）。

- **`gd_ssm_prep`**（新，仅 T==1 decode）：conv1d+silu **直接按 q|k|v 分区写入**三个 buffer
  （省掉 split3 的 16 KB 读写）、同一 kernel 里做 conv 历史移位（**每个 block 只碰自己那 256 列**
  ⇒ 读写同列无跨 block 竞争，T==1 下不需要额外同步）、`blockIdx.x == nb` 的那个 block 顺带算
  `beta = sigmoid(b)` 与 `g = -exp(A)·softplus(a+dt)`。顺带把 conv 的并行度从 1 个 block
  提到 32 个 block（C=8192 / 256）。**4 发 → 1 发**。
- **`gd_delta_rule_128f`**（新）：把 q/k 的 per-head L2 norm 放进 prologue（共享内存里保留
  归一化后的 sk/sq，不再回写 global），把 gated RMSNorm（`o·rsqrt(mean(o²)+eps)·w·sigmoid(z)`）
  放进 epilogue。**3 发 → 1 发**。
- T>1（prefill）保持原多内核路径，行为不变。

### 2.3 int8 激活 staging 复用（−0.04 ms；201 → 129 次 quant_x_i8）

`quant_x_i8` 原来在**每个** dp4a GEMV 前都跑一次，但同一层 gate/up 读同一份 `hn_`，qkv/z 也是。
在 `run_gemv_plan()` 里按 `(激活指针, n_in)` + `cuda::g_launch_calls` 缓存：**只有"上次量化之后
一个内核都没发射过"才复用**（这同时证明缓冲区不可能被改写，天然安全，且 CUDA Graph
capture/replay 下行为确定）。另把 attention 层的 `qp/kp/vp` 三次 GEMV 调成相邻（原来 `split_qgate`
夹在中间）。201 → 129 次/token。实测收益在噪声内（−0.04 ms），保留理由是少 72 次发射 + 少读一次激活。

### 2.4 logits 回传走 pinned（−0.06 ms）

decode 每 token 要回传 1 MB logits。原来 `cuMemcpyDtoH`（pageable）在 WDDM 上 **0.34 ms**；
改成 pinned 暂存 + `cuMemcpyDtoHAsync` + `stream_sync` + 一次 host memcpy ⇒ **0.28 ms**
（`ENGINE_DBG_FWD` 计时）。

---

## 3. 四模型性能（每模型 5 次独立运行取中位；GPU 空闲、无并发）

| 模型 | 改动前 | 改动后 | tok/s | 用户 llama.cpp | 占比 | 要求 |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 5.6 ms / 178 | **5.17 ms** | **193.4** | — | — | ≥170 ✓ |
| DeepSeek-Coder-V2-Lite | 42.9 ms / 23.3 | **41.24 ms** | **24.25** | 18.17 | 133 % | ≥22.5 ✓ |
| **Qwen3.5-9B** | **26.07 ms / 38.4**（同机同快照复测） | **22.22 ms** | **45.00** | 46.7 | **96 %** | **≤23.0 ms ✓** |
| Qwen3.6-35B-A3B | 47.7 ms / 21.0 | **40.66 ms** | **24.59** | 25.76 | 95 % | ≥19.5 ✓ |

- 0.5B/Coder/35B 的"改动前"取项目既有基线（`build/finalN_1.log`、UTILIZATION-REPORT 正文）。
- **Coder 注意**：若用 `ENGINE_GPU_EXPERTS=16` 钉死放置会掉到 66.9 ms（14.9 tok/s）——
  这与本次改动无关（显存打满的假回退，见 UTILIZATION-REPORT 附录 C.2），auto 放置是 41.2 ms。
- **35B 受益最大**：它 30 个线性注意力层也走 `gd_delta_rule_128` 与融合链，47.7 → 40.7 ms；
  60 token 长解码无挂死。
- 9B 与 llama.cpp 的差距从 79 % 拉到 **96 %**。

---

## 4. 验证状态（最终二进制）

| 套件 | 结果 |
|---|---|
| `tools/verify.sh` | **ALL GREEN**（全部 C++/Python 单测、G0 IR、M2 plan、M3 CUDA） |
| `tools/smoke_models.sh` | 4/4 预测正确：40716 / 185 / 2614 / 69014 |
| `tools/verify_generality.sh` | **GENERALITY GREEN**（26 passed, 0 failed） |

数值：

- 新内核 vs 原内核（微基准，单步）：`gd_delta_rule_128` 状态 rel_l2 **6.8e-9**、输出 **2.1e-7**。
- 端到端（单次 forward，最后 token logits）：rel_l2 **3.5e-5**。注意**引擎原路径本身**
  因 split-K GEMV 的 `atomicAdd` 求和顺序不确定，**同一配置重复跑**的 logits 相对差就有
  **3.5–4.0e-5**（实测 rep2/rep1=3.97e-5、rep3/rep1=3.58e-5）⇒ 新路径的端到端差异完全落在
  原引擎自身的运行间噪声内，逐内核口径远小于 1e-5。
- 贪心生成 token 序列：9B 40 步、12 步均与"全关旧路径"逐 token 相同；
  单次 forward logits 的 argmax 全一致（2614）。

---

## 5. 新增/变更的配置项（已注册 + 写进 `docs/CONFIG.md`）

| key | CLI | env（反转） | 默认 | 作用 |
|---|---|---|---|---|
| `gd_delta_128` | `--no-gd-delta-128` | `ENGINE_NO_GD_DELTA_128` | `true` | 寄存器分块 delta-rule kernel（head dim 128） |
| `ssm_fuse` | `--no-ssm-fuse` | `ENGINE_NO_SSM_FUSE` | `true` | `gd_ssm_prep` + `gd_delta_rule_128f` 融合链（仅 T==1） |
| `quant_reuse` | `--no-quant-reuse` | `ENGINE_NO_QUANT_REUSE` | `true` | 连续 dp4a GEMV 复用 int8 staging |

诊断开关（不注册进配置表）：

- `ENGINE_SKIP_KERNELS=name1,name2`：`cuda::launch` 层按内核名丢弃发射（只用于墙钟归因，
  数值无意义）。本次"边缘内核总共值多少"就是用它量的：一次跳过 570 个小内核 ⇒ **−1.62 ms**。
- `ENGINE_SSM_PREP_ONLY=1` / `ENGINE_SSM_DELTA_ONLY=1`：只开融合链的一半，用于二分定位。
- `ENGINE_KPROF` 现在在 **graph 模式**下也能打印（capture 时记录的 event 成为 graph node，
  重放后再取 elapsed）；数值仍只作相对参考。

---

## 6. 失败尝试（如实记录）

1. **rmsnorm + quant_x_i8 融合（负结果，已回退）**：把两级 int8 量化折进 `rmsnorm` 的 epilogue
   （同一 block 内 warp 分块），省一次发射并把 32 次 quant 变成 0 次。实测 **22.2 → 22.8 ms
   （慢 0.6 ms）**：`rmsnorm` 只有 1 个 block（hidden 归一化），而 `quant_x_i8` 原来用 128 个
   32 线程的 block 并行；融合后量化被压到 1 个 SM 上串行 16 组 ⇒ 得不偿失。
   代码已完整回退（kernel 也删掉），不留死路径。
2. **quant 复用一开始想做成"按 buffer 指针缓存 + 手动失效"**：太容易漏（rmsnorm/silu/add/
   delta 都会写 GEMV 输入），放弃；改成"期间零发射"判定后无失效点、graph 安全。
3. **修 SSM 融合时踩的坑**：第一次实现把旧的 split/beta/delta 代码整段套进了 `else` 分支，
   融合开启时整条链被跳过（预测变成 760 全同）。用 `ENGINE_SSM_PREP_ONLY` / `_DELTA_ONLY`
   二分定位后修好——这两个诊断开关保留下来。
4. **`gd_delta_rule` 的第一版候选（v2：KD-major 布局 + 两遍读）** 隔离 0.025 ms/次（5.4×），
   但比寄存器分块版（0.0125 ms）差一倍，未采用。
5. **正确性检查时发现候选 kernel 与原 kernel 差 2.5e-4**：原因是第一版把 `decay` 放在 kv 点积
   **之后**，而参考实现是 `S *= exp(g)` 之后才算 kv（decay-first）。修正后差 6.8e-9。
6. **想给"残差 add + 下一层 rmsnorm"做跨层融合**（64 次 `add_inplace`，理论 −0.2 ms）：
   需要延迟发射 + 在 run_pipeline/layer 边界 flush，会波及非 qwen35 模型（MLA/MoE/0.5B 的
   layer body 读 `x_` 的时机不同），风险不划算，未做。留给后续。
7. `ENGINE_KPROF` 在 graph 模式下拿不到数据的第一版：pairs 记录了但 `EventElapsedTime` 全失败
   （capture 里的 event 节点不能被回读），改为"capture 记事件、重放后取 elapsed"仍 0；
   最终用 `ENGINE_SKIP_KERNELS` 做归因，kprof 只用于 `--no-graph` 的相对对比。

---

## 7. 复现

```bash
bash build.sh
M="<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf"

# 9B decode（5 次中位）
M="$M" IDS=760 N=9 REPS=5 ENVS="ENGINE_SKIP_WARMUP=0" bash tools/bench_one.sh

# 单项 A/B（交错运行、抵消漂移）
REPS=5 BASE="ENGINE_NO_SSM_FUSE=1" VARIANT="ENGINE_NO_GD_DELTA_128=1" \
  bash tools/ab_decode.sh

# 边缘内核对账（--no-graph 相对口径）
ENGINE_KPROF=1 ENGINE_TOPK=1 ENGINE_SKIP_WARMUP=0 ./build/test_gen.exe "$M" 760 2 --no-graph

# 数值
bash tools/numcheck.sh          # 单次 forward logits：新/旧路径 + 各项归因
bash tools/verify.sh && bash tools/smoke_models.sh && bash tools/verify_generality.sh
```

工具与新增文件：`tools/ab_decode.sh`（交错 A/B）、`tools/bench_one.sh`（单模型中位）、
`tools/numcheck.sh`（logits 数值归因）、`tools/bench_gd.cpp` + `tools/bench_gd_kernels.h`
（delta-rule 隔离微基准）。

回退：本次改动的源文件快照为 `include/engine/kernels.h.v1|.v2`、`src/runner.cpp.v1|.v2|.v3`、
`src/qwen35.cpp.v1|.v3`、`include/engine/runner.h.v1|.v2|.v3`、`src/config.cpp.v1`、
`src/cuda.cpp.v1`、`docs/CONFIG.md.v1`。三个新行为都能用 env 单独关掉。
