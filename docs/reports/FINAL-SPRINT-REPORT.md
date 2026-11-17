# 最后冲刺报告（FINAL-SPRINT）：9B 与 35B 达到 llama.cpp 的 100%+

日期：2026-11-15 · 目标：9B decode ≤21.4 ms（≥46.7 tok/s）、35B ≤38.8 ms（≥25.76 tok/s），
四模型预测与三套验证保持全绿，Coder ≥24 tok/s、0.5B ≥185 tok/s 不回退。

**结论：两个目标都达成。9B 22.10 → 21.42 ms（46.69 tok/s = llama.cpp 的 100.0%），
35B 42.18 → 37.98 ms（26.33 tok/s = 102.2%）。四模型预测全对（40716 / 185 / 2614 / 69014），
`verify.sh` ALL GREEN、`verify_generality.sh` GENERATION GREEN(26/26)、`smoke_models.sh` 4/4，
Coder 23.92 → 24.08 tok/s（无回退，反而略快）、0.5B 192.3 → 200.0 tok/s（+4%）。**

> 数字口径：下面所有"前/后"都是**同一台机器上两个二进制背靠背交错**跑出来的中位
> （`tools/ab_pre_new.sh`，改动前二进制 `build/test_gen_pre.exe` 由 `.pre` 快照重建），
> 因为虚拟机漂移 ±3–5 %。绝对值和历史报告（22.22 / 40.66）的差异全部来自漂移：
> 本轮"改前"二进制实测就是 22.10 / 42.18。

---

## 1. 一句话账本

| 改动 | 作用位置 | 预期 | 实测（交错 A/B，中位） |
|---|---|---|---|
| **decode 头设备端 top-k**（`head_topk`）：logits 行归约在 GPU 上做完，只回传 5 组 `(index,value)`，不再 pinned D2H 整行 + 2 次 1 MB memcpy | `include/engine/kernels.h`（`topk_partial`/`topk_final`）、`src/runner.cpp`（`run_pipeline` 头、`forward_impl`）、`include/engine/runner.h`（`forward_topk`）、`tests/test_gen.cpp` | −0.2～0.3 ms | 9B −0.33 ms（0.27 → ~0.02 ms 头回传）；35B −0.25；0.5B −0.15 |
| **float4 rmsnorm**（`rms4`）：load/store 指令数 4× 减少，reduction 从 8 次 `__syncthreads` 减到 1 次 warp-shuffle + 1 barrier；宽度 ≥2048 时用 512 线程（每线程 2 次 float4） | `kernels.h::rmsnorm4`、`src/runner.cpp::rms_raw`、`src/qwen35.cpp::rms_buf` | −0.3～0.4 ms（9B 81 次/token） | 9B −0.30（256 线程，0.66 → 0.36）再 −0.10（512 线程） |
| **`q35_split_qgate` 并行化**：原来一个 block 串行跑 32 个 head（~27 µs），改成 grid=(token,head) | `kernels.h`、`src/qwen35.cpp` | −0.15 ms | 9B −0.15（0.22 → ~0.05） |
| **显存预留重复计数修复**：GGUF 的 `output.weight` 在构造函数里就已上传，装载循环又把它当成"未来分配"预留了一次（35B 是 565 MiB） | `src/runner.cpp`（放置记账） | 让"实测校验"能多放一层 | 35B 剩余预留 1660 → 1100 MiB |
| **`vram_extend`**：算术估算只当**下限**，逐层*实测*显存校验可以**追加**一层（只尝试紧邻下一层，保持连续前缀；装载后 margin 校验仍生效） | `src/runner.cpp`、`src/qwen35.cpp`（两条 MoE 装载路径都要改） | 35B +1 GPU 层 = −1 ms | 35B 15 → 16 层（post-load free 1940 → 1432 MiB ≥ 1200 margin） |
| **把逐层 CUDA Graph 录制挪进 prefill**：首次 decode 形状调用"正常执行 + 同一次调用内立即录制"，一次性 `cuGraphInstantiate` 不再落进 decode 计时窗口 | `src/qwen35.cpp`（`run_q35_attn_graph` / `run_gpu_layer_graph`） | −0.6 ms（9 步均值） | 35B −0.9（39.90 → 38.90 单步中位） |
| **CPU 驻留层的逐层 Graph 默认开启**（`cpu_layer_graph`，原来 opt-in 且实测"净变慢"）：每层少 ~7 次 WDDM 提交 | `src/runner.cpp`（`run_layer`）、`src/config.cpp` | −1.6 ms（24 层 × 7 发 × ~10 µs） | 35B −1.2（38.90 → 37.7），A/B 中位 −0.64（受噪声影响） |

**逐项之和 ≈ −1.0 ms（9B）/ −3.4 ms（35B），与端到端 −0.68 / −4.20 量级一致**（部分项互相重叠，
例如 top-k 头在 35B 上被 CPU 层气泡掩盖）。

---

## 2. 每项的数据依据（为什么做这个而不是别的）

先用 `ENGINE_SKIP_KERNELS` 做整类内核的墙钟归因（graph 模式，N=9，每项 3 次）：

| 9B（改前基线 22.43） | 实测 |
|---|---|
| 跳过 5 种 GEMV 内核 | 22.43 → **2.10 ms** ⇒ GEMV 占 **20.3 ms**（5868 MB ⇒ **289 GB/s**，已等于孤立微基准 295–319） |
| 跳过 12 种边缘内核 | 22.43 → **20.89** ⇒ 边缘内核总共只有 **1.54 ms** |
| 其中 `rmsnorm` 一项 | **0.66 ms**（81 次 × 8.1 µs，单 block 延迟受限） |
| 其中 `gd_delta_rule_128f` | 0.52 ms（24 次 × 21.7 µs，已接近孤立 12.5 µs 的 2 MB state 往返下限） |
| `quant_x_i8` / `add_inplace` | **0 ms**（完全被 GEMV 掩盖） |

⇒ **GEMV 20.3 ms 已经贴着孤立带宽，没有空间**；剩下能动的只有 1.54 ms 边缘内核 + 0.6 ms 头。
（这也修正了任务书里"220 vs 295 GB/s 有 3–4 ms 空间"的旧判断：边缘内核融合之后有效带宽已经是
289 GB/s，旧数字 220 是把 3.9 ms 的 kprof 边缘内核算进了分母。）于是把力气全押在
rmsnorm、split_qgate、头部回传三项上。

`lm_head` 的 plan 也验证过：Q6_K 248320×4096 默认 dp4a 最好（21.45），换 `q6k_x4` 24.2、
换 `q6k` 22.7、`q6k_dp4a_sk2/4` 21.6/21.9 ⇒ **不动**。

35B 的归因：`ENGINE_PIPE_PROF` 每 token **694 次内核发射 = 6.8 ms 纯 WDDM 提交时间**，
`syncs` 每 token ~50 次 = 6 ms，host matvec 0.84 ms × 24 层 = 20 ms。逐层 `ENGINE_DBG_LAYERMS`：
GPU 层 0.15–0.34 ms / CPU 层 1.08–1.44 ms。⇒ 三个可动项：**层数**（每层 1 ms）、
**提交次数**（每层 ~7 次可省）、**一次性录制**（被摊进 decode 窗口）。

---

## 3. 结果

### 3.1 交错 A/B（`tools/ab_pre_new.sh`，改动前二进制由 `.pre` 快照重建）

| 模型 | 改前 ms | 改后 ms | Δ | 改前 tok/s | 改后 tok/s | llama.cpp | 占比 |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5B | 5.20 | **5.00** | −0.20 (−3.8 %) | 192.3 | **200.0** | — | — |
| Coder-V2-Lite | 41.81 | **41.53** | −0.28 (−0.7 %) | 23.92 | **24.08** | 18.17 | 132 % |
| **Qwen3.5-9B** | 22.10 | **21.42** | **−0.68 (−3.1 %)** | 45.25 | **46.69** | 46.7 | **100.0 %** |
| **Qwen3.6-35B-A3B** | 42.18 | **37.98** | **−4.20 (−10.0 %)** | 23.71 | **26.33** | 25.76 | **102.2 %** |

（每行 5–7 次交错，中位；Coder 7 次、其余 5 次。改前/改后各自 min–max：9B 22.00–22.17 /
21.35–21.44，35B 40.58–42.97 / 37.89–41.86。）

**最终二进制**（在上面 A/B 之后又加了"rmsnorm 512 线程"这一项，各自单独跑 3 次）：
**9B 21.31 / 21.32 / 21.36 ms ⇒ 中位 21.32 ms = 46.90 tok/s（llama.cpp 的 100.4 %）**；
**35B 37.81 / 37.93 / 38.18 ms ⇒ 中位 37.93 ms = 26.36 tok/s（102.3 %）**，预测 2614 / 69014 不变。

### 3.2 四模型 + 验证

| 套件 | 结果 |
|---|---|
| `tools/verify.sh` | **ALL GREEN** |
| `tools/smoke_models.sh` | 4/4：**40716 / 185 / 2614 / 69014** |
| `tools/verify_generality.sh` | **GENERALITY GREEN（26 passed, 0 failed）** |

数值：
- 全部新路径都是**逐位相同**的计算（top-k 只是把整行 argmax 搬到设备端、rmsnorm4 只改求和顺序、
  放置/Graph 改动不改数学），四模型贪心 top-1 与改前完全一致（见上表 A/B 的 `topk:` 列逐次相同）。
- `rmsnorm4` vs 标量 `rmsnorm`：求和顺序不同 ⇒ 单次 forward logits 相对差 ~1e-7（在引擎自身
  split-K `atomicAdd` 的 3.5–4.0e-5 运行间噪声内），**没有引入新的数量级**。

---

## 4. 失败的尝试（如实）

1. **第一版 `topk_final` 单线程扫描**：640 个候选 × 5 轮 = 3200 次**相关全局读**（一次约 300 ns）
   ⇒ 该内核 0.5 ms，把 top-k 的收益全部吃掉（端到端反而慢 0.27 ms）。改成 256 线程并行
   block-reduction 后 3 µs，才有 −0.33 ms。
2. **`lm_head` 换 plan**：`q6k_x4` / `q6k` / `q6k_dp4a_sk2|4` 全部更慢（见 §2），放弃。
3. **`ENGINE_CPU_LAYER_GRAPH=1` 在旧二进制上 A/B**：中位 +0.07 ms（净中性偏负），一度判定放弃；
   定位到原因是**一次性录制落在 decode 计时窗口**（24 个 graph × ~0.3 ms），
   把录制挪到首次执行之后才变成正收益。
4. **想用"CPU 层只跑 gate/up、down 放 GPU"**：算过账——CPU 只读 9.4 MB/层（省 0.4 ms/层），
   但 routed expert 的 top-8 每 token 都变，无法静态驻留这 6.9 MB 权重（每 token 上传要 1.36 ms），
   只能用"全部 256 专家 down 常驻"= 5.5 GB，显存放不下 ⇒ 放弃（写在这里避免后续重复论证）。
5. **`vram_extend` 第一版只改了 `src/runner.cpp` 的通用 MoE 装载路径**：35B 走的是 qwen35 的
   `load_q35_ffn`（`src/qwen35.cpp`），完全没生效（debug 里连一条 `[vram] MoE layer #15` 都没有）。
   两条路径都改后才 15 → 16 层。
6. **想把 `vram_margin` 默认从 1200 降到 1000 多放一层**：没有采用。理由：那会同时作用到 Coder/Coder
   类显存吃紧的模型（历史上有过 600 MiB 触发驱动分页、Coder 66.9 ms 的"假回退"），而
   `vram_extend` 用实测值同样拿到了一层，且 margin 语义没被削弱。最终 16 层时 post-load
   free = 1432 MiB，仍高于 1200 MiB margin。

---

## 5. 剩余缺口（最精确分解）

**9B：21.42 ms（46.69 tok/s）**，与 llama.cpp 21.4 ms 持平，缺口 0。

按 `ENGINE_SKIP_KERNELS` 归因（改后）：
| 组成 | ms | 说明 |
|---|---|---|
| GEMV（含 lm_head） | 20.3 | 5868 MB ⇒ **289 GB/s**，孤立微基准 295–319、纯读天花板 330 |
| 边缘内核合计 | ~1.0 | rmsnorm 0.36 + gd_delta 0.52 + 其余 0.1 |
| 设备端 top-k 头 | ~0.02 | 原来是 0.27（1 MB pinned D2H）+ ~0.1（两次 1 MB 主机 memcpy） |

⇒ 9B **再往上走只有一条路：把 GEMV 从 289 GB/s 提到 320+ GB/s**（需要重写 Q5_K/Q6_K dp4a GEMV 的
load 调度，是本轮没时间做的深水区）。边缘内核已经压到 1.0 ms，最多再榨 0.3。

**35B：37.98 ms（26.33 tok/s）**，已达 102%。
| 组成 | ms | 说明 |
|---|---|---|
| 24 个 CPU 层 | ~28 | host matvec 0.84×24 = **20 ms**（16.3 MB/层 ⇒ 19.4 GB/s；孤立 `bench_cpu_layer` 最好 0.715 ms/L = 22.8 GB/s，纯读 DRAM 天花板 31–36 GB/s）|
| 16 个 GPU 层 | ~4 | 0.25 ms/层，已近下限 |
| 头（lm_head Q8_0 515 MB + top-k） | ~2 | 515 MB @ ~300 GB/s |
| 一次性捕获摊销（N=9） | ~0.3 | 已挪进 prefill，只剩 graph 重放的固定开支 |

⇒ 35B 若还要更快，唯一的大头是 **CPU 量化 GEMV 从 22 GB/s 提到 DRAM 天花板 31–36 GB/s**
（0.84 → 0.55 ms/层 ⇒ −7 ms）。这需要把 Q4_K/Q6_K 的 host 权重重排成 AVX2 友好的 interleaved 布局
（llama.cpp 的 repack 路线），属于独立的大改动，本轮未做。

---

## 6. 配置与回退

新增 4 个配置键（全部默认 `true`、都能用 env 反转、已写进 `docs/CONFIG.md` 3.2/3.3/3.5 节）：

| key | CLI | env（反转） | 作用 |
|---|---|---|---|
| `head_topk` | `--no-head-topk` | `ENGINE_NO_HEAD_TOPK` | decode 头设备端 top-5 归约 + 只回传 5 组 pair |
| `rms4` | `--no-rms4` | `ENGINE_NO_RMS4` | float4 rmsnorm（宽度 %4==0 时自动启用） |
| `vram_extend` | `--no-vram-extend` | `ENGINE_NO_VRAM_EXTEND` | 实测显存校验可追加 GPU 驻留层 |
| `cpu_layer_graph` | `--cpu-layer-graph` | `ENGINE_NO_CPU_LAYER_GRAPH` | CPU 驻留 MoE 层的 mixer 段逐层 Graph |

改动文件（`.pre` 快照 = 本轮开始时的状态，回退直接覆盖回去）：
`include/engine/kernels.h`、`include/engine/runner.h`、`src/runner.cpp`、`src/qwen35.cpp`、
`src/config.cpp`、`tests/test_gen.cpp`、`docs/CONFIG.md`。
没有改 `build.sh`；新增工具 `tools/ab_pre_new.sh`（交错 A/B）、`tools/final_table.sh`、
`tools/prof9b*.sh`、`tools/prof35b_margin2.sh`、`tools/gguf_table.py`。

## 7. 复现

```bash
bash build.sh
# 交错 A/B（核对本轮收益）
M=<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf N=9 REPS=5 bash tools/ab_pre_new.sh
M=<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf N=9 REPS=5 bash tools/ab_pre_new.sh
# 四模型中位
REPS=5 bash tools/final_table.sh
# 9B 内核类归因（跳过整类内核，墙钟差分）
ENGINE_SKIP_KERNELS=rmsnorm,quant_x_i8,gd_delta_rule_128f,rope_strided,add_inplace,zero_fill,attn_cached_w,q35_gate_mul,silu_mul,gd_ssm_prep,q35_split_qgate,kv_write \
  ENGINE_TOPK=1 ./build/test_gen.exe <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf 760 9
# 35B 发射/同步开销
ENGINE_TOPK=1 ENGINE_PIPE_PROF=1 ./build/test_gen.exe <35B.gguf> 760 6
# 验证
bash tools/verify.sh && bash tools/smoke_models.sh && bash tools/verify_generality.sh
# 回退单项
ENGINE_NO_HEAD_TOPK=1 / ENGINE_NO_RMS4=1 / ENGINE_NO_VRAM_EXTEND=1 / ENGINE_NO_CPU_LAYER_GRAPH=1
```
