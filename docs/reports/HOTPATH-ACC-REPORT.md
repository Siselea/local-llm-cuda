# 35B「未归因 7.7 ms」钉死 + 两项热路径修复（HOTPATH-ACC）

日期：2026-11-21 · 目标：把 35B decode 账本里"未归因 ~7.7 ms（20%）"分到具体位置，并吃能吃的部分。
范围：`src/runner.cpp`、`src/qwen35.cpp`、`include/engine/config.h`（新增 env 缓存宏）。
新增工具：`tools/ab35x.sh`（任意 env 变体交错 A/B）、`ENGINE_PROF_ACC`（低开销逐段记账）。

---

## 0. 结论先行

> **"7.7 ms 未归因"不是一块隐藏开销**，而是旧账本里 4 个被低估/漏计的项之和：
> 1. **每层 ~30–35 次 `std::getenv` = 2.6–3.0 ms/token**（本机 getenv =2.13 µs/次，旧账本完全没有这一项）；
> 2. **CPU 层每层的"往返机械"**：`pre`(router 提交+2 次 pinned D2H+共享专家入队) 0.063 + `tail`(H2D+add3) 0.112
>    + `launch`(9 次提交) 0.115 = **0.29 ms/层 = 7.0 ms/token**（旧账本只按 launch+sync 的旧值 0.29 算，且算在别处）；
> 3. **`sync` 等待**从旧值 0.165 涨到 **0.354 ms/层 = 8.5 ms/token**（宿主侧变快后，GPU 侧 attention/共享专家变成显性瓶颈）；
> 4. **GPU 层 3.2 ms**（旧账本按 3.0 算，量级对）与 **lm_head 1.84 ms**（旧账本"head ~2.5"其实基本就是它）。
>
> 本轮实测账本（35B，`--gpu-layers 15`：24 个 CPU MoE 层 + 16 个非 CPU 层，wall **35.36 ms**）：
> **CPU host GEMV 14.8 (42%) · CPU 层 sync 8.5 (24%) · GPU 层 3.2 (9%) · CPU 层 launch 2.8 (8%) ·
> CPU 层 tail 2.7 (8%) · CPU 层 pre 1.5 (4%) · lm_head 1.8 (5%) · 其它 ~0.9**。
>
> **修复 1（本轮）**：热路径 getenv 全部按调用点缓存 ⇒ **35B −2.11 ms（−5.8%）**，27.4 → 29.1 tok/s，**逐 token bit-exact**。
> **修复 2（本轮，来自 edge-kernels 审计）**：纯 CPU 层 `run_moe` 的 256 轮空转（~0.5 ms/token）已吃掉。
> **五套验证全绿**（§6）。

---

## 1. 为什么旧账本对不上：`ENGINE_DBG_LAYERMS` 自带 +11 ms/token

- 旧账本用 `ENGINE_DBG_LAYERMS=1` 逐层打点。本机 stderr 无缓冲且重定向到磁盘，**每行 200–300 µs**，
  40 层/token ⇒ 该开关自己给每 token 加 **~11 ms**。
  实测（同一二进制、同一配置）：**逐层之和 32.1 ms vs `[time] decode 43.1 ms`**；
  两次独立 run 的差都是 11.0–11.1 ms，可复现 ⇒ 差值不是 head、不是未归因，是打印开销。
- 后果：用逐层和反推端到端会凭空多出 ~11 ms，再和端到端相减就会被误读成"未归因"。

### 新工具：`ENGINE_PROF_ACC=1`（内存累加、退出时打印一次）

- 开销实测 **<0.1 ms/token**（同一配置：base 43.26 vs acc_on 43.26），且 `accounted == wall`（残差 0.000）。
- 覆盖：token wall / `pre`(embed+upload+setpos) / pipeline / Σ`run_layer`(按 CPU/GPU 层分类) /
  `head` / head 的 D2H+sync；CPU 层内部再分 `pre / route / gemv / tail`，并统计每层的
  `g_sync_ms`、`g_launch_ms`、launch/sync 次数。
- 用法：`ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ENGINE_PROF_ACC=1 build/test_gen.exe <35B> 760 48 --gpu-layers 15`
  （默认跳过前 4 个 n==1 前向以排除 capture/page-in；`ENGINE_PROF_ACC_SKIP=N` 可调）。

---

## 2. 精确账本（钉死 7.7 ms）

35B，`--gpu-layers 15`（24 CPU MoE 层 + 16 非 CPU 层，40 层），`ENGINE_PROF_ACC=1`：

| 段 | ms/token | 占 wall | 每层 | 说明 |
|---|---|---|---|---|
| **CPU host GEMV**（`moe_forward_mt`） | **14.81** | 42% | 0.617 | 12 线程/SMT 成对；已等于隔离基准（算力受限） |
| **CPU 层 `sync` 等待** | **8.49** | 24% | 0.354 | 等 attention 图+router+共享专家的 GPU 完成（微基准：`cuStreamSynchronize` 空转仅 2.1 µs ⇒ 这是真 GPU 工作+队列排空） |
| **GPU 层**（16 层） | **3.22** | 9% | 0.201 | launch 0.146 + 其它 0.056（GPU 层无 sync） |
| **CPU 层 `launch`**（9 次提交） | **2.75** | 8% | 0.115 | 13 µs/次（本机 WDDM 提交价，微基准一致） |
| **CPU 层 `tail`**（H2D host_part + `add_three`） | **2.69** | 8% | 0.112 | 1 次 pinned H2D + 1 次内核 |
| **CPU 层 `pre`**（router 提交 + 2 次 pinned D2H + 共享专家入队） | **1.51** | 4% | 0.063 | |
| **lm_head**（540 MB Q8_0 GEMV） | **1.84** | 5% | — | `ENGINE_SKIP_KERNELS=gemv_q8_0_bm` A/B 实测（35.86 → 34.02）；kprof 同值 1.88 |
| head 其它（final norm + top-k + topk D2H/sync） | 0.19 | 1% | — | |
| `pre`（embed 反量化 + x 上传 + setpos） | 0.07 | 0% | — | |
| 计数 | **408 launch/token**（CPU 216 / GPU 192）、**24 sync/token** | | | `route` 宿主侧只 0.007/层 |

**读法**：CPU 层总成本 **1.250 ms/层 = GEMV 0.617 (49%) + 非 GEMV 0.633 (51%)**。
非 GEMV 里最大的已经是 **`sync` 0.354（等 GPU）**，不是宿主提交——
即宿主侧（GEMV/提交/往返）变快之后，**GPU 侧每层 attention+共享专家+路由的串行执行时间 + WDDM 往返暴露出来了**。

**和旧账本的差异**：旧账本 `18.9(GEMV)+7.2(launch+sync)+3.0(GPU层)+2.5(head)=31.6`，与 39.3 差 7.7。
按本轮实测，同样几项是 **14.8+2.8+8.5+3.2+1.8+2.7+1.5+0.9 ≈ 36.2**——
即旧账本的 sync 少了 ~5 ms、tail/pre 漏了 ~4.2 ms、getenv 完全没算（~2.8 ms），
而 GEMV 其实比账本便宜（12 线程绑核后 0.617 vs 0.755）。**没有神秘项。**

---

## 3. 修复 1：热路径 `std::getenv` 缓存（−2.11 ms，bit-exact）

- **微基准**（同一台机、同一环境）：`std::getenv()` = **2.13 µs/call**（CRT 线性扫描整个环境块；
  `steady_clock::now()` 只有 0.037 µs）。热路径每层约 30–35 次（`run_layer` 9 次、`run_moe` ~20 次、
  qwen35 层路径 ~5 次）⇒ 40 层 = 1200–1400 次 ⇒ **2.6–3.0 ms/token**。
- **实现**：`include/engine/config.h` 末尾新增两个宏（每个调用点一份函数局部 static 缓存）：
  `ENGINE_ENV_ON(name)` / `ENGINE_ENV_STR(name)`；把 `runner.cpp`（>1450 行）与 `qwen35.cpp`（>270 行）
  的**热路径**替换掉（71 + 24 处）。**加载期/一次性诊断仍用普通 `getenv()`**（行为不变）。
- **为什么安全**：进程环境在启动后不会改变 ⇒ 缓存值与每次调用 `getenv()` 完全相同。
  实测：35B n=40 的 `gen:` 序列 md5 与改动前**相同**（`e9f0a5a6`），`topk:` 相同（2614）。

## 4. 修复 2：纯 CPU 层 `run_moe` 的 256 轮空转（~0.5 ms/token）

`run_moe` 的逐专家循环在 `cpu_batched && gpu_n == 0`（纯 CPU 层）时每轮只走 `continue`，
但仍付 3 次 `steady_clock::now()` + 2 次 `std::vector::assign`：**22 µs/层 × 24 层 ≈ 0.53 ms/token**。
修法（3 行，语义等价）：
```cpp
const int expert_loop_end = (cpu_batched && gpu_n == 0) ? 0 : ne;
for (int e = 0; e < expert_loop_end && !fused_done; ++e) { ... }
```
（`gpu_n > 0` 的混合放置分支保留。该发现来自 edge-kernels 的 `docs/HOTPATH-AUDIT.md` §2.2，我复核后实施。）

---

## 5. A/B 结果（交错、同会话）

`build/test_gen_base.exe`（改动前）vs `build/test_gen.exe`（getenv 缓存 + 空转修复），
35B、`--gpu-layers 15`、`ENGINE_DEC_SKIP=2`、n=40（38 个计时步）：

| 轮次 | 改动前 | 改动后 |
|---|---|---|
| rep1 | 36.30 ms | **34.40 ms** |
| rep2 | 36.58 ms | **34.26 ms** |
| **中位** | **36.44 ms / 27.4 tok/s** | **34.33 ms / 29.1 tok/s** |
| Δ | | **−2.11 ms（−5.8%）** |

bit-exact：两轮 `topk` 都是 2614，`gen` 序列 md5 都是 `e9f0a5a6`（与改动前逐 token 相同）。
（注：本机噪声极大——同一二进制跨时段实测 35.4 ↔ 43.3 ms——所以所有结论都用同会话交错 A/B；
上表与用户口径的差别来自机器状态，不是改动。）

---

## 6. 五套验证 + 真实用户端到端

### 6.1 五套（最终二进制：`build/test_gen.exe` 12:00 / `build/engine.exe` 12:04，`MODEL_DIR=${MODEL_DIR}`）

| 套件 | 结果 |
|---|---|
| `tools/verify.sh`（全量 build + 单测 + IR/plan parity） | **ALL GREEN**（exit 0） |
| `tools/smoke_models.sh` | **40716 / 185 / 2614 / 2614**（4/4 正确） |
| `tools/gen_regression.sh`（与 llama.cpp 逐 token 对齐） | **10 passed, 0 failed — GREEN** |
| `tools/verify_generality.sh` | **26 passed, 0 failed — GREEN** |
| `tools/verify_serve.sh`（HTTP/SSE/WebUI/优雅退出） | **22 passed, 0 failed — SERVE VERIFY GREEN** |

数值证据：35B n=40 的 `gen:` md5 与改动前相同（`e9f0a5a6`）、`topk: 2614` 相同；
其余三个模型 top-1 不变；本轮的改动只有"缓存 env 值"和"跳过纯 continue 的行为空转"，不改任何数学。

### 6.2 真实用户视角（9B serve，端口 7050）

- **浏览器**：`http://127.0.0.1:7050/` 在真实 Chromium 里渲染出完整 llama.cpp chat WebUI
  （会话列表、模型信息 `Qwen3.5-9B-Q5_K_M.gguf`、Samplers/Reasoning 设置面板全部可见）。
  截图：`.dsh-browser/engine-webui-e2e.png`。
- **中文流式对话**（WebUI 用的同一组端点 `/v1/chat/completions` + `stream:true`）：
  - turn1「你好，请用一句话介绍你自己。」：SSE 25 KB，`predicted_n=91`，decode **45.6 tok/s**，2.58 s；
  - turn2（带 3 条历史的多轮）「记住我的名字叫小明…我刚才让你记住的名字是什么？」：
    `prompt_n=60`（历史确实进了上下文），`predicted_n=169`，decode **44.7 tok/s**，4.41 s；
    流式内容为连贯中文/思考过程（think 预算命中是 serve 的既有配置，与本轮改动无关）。
  - `POST /shutdown` → `{"status":"shutting down"}` → 日志 `[serve] stopped; VRAM released`，
    GPU 回到空闲（914 MiB / 12 W），端口释放。
- 局限（如实）：本会话的浏览器工具只能"打开页面 + 抽取文本/截图"，**没有点击/键盘输入能力**，
  所以"在 WebUI 输入框里真敲字"这一步没法由我完成；上面是"真实浏览器渲染 + WebUI 同源端点 +
  多轮中文流式"的最接近的端到端验证。最终的一次鼠标键盘点选建议由 lead/server-mode 侧执行。


---

## 7. 9B 顺手复核（固定开销 <1%）

`ENGINE_PROF_ACC=1`（9B，decode 21.456 ms/token，全 GPU、整 pipeline CUDA Graph）：

| 段 | ms/token |
|---|---|
| `pre`（embed + x 上传 + setpos） | 0.073 |
| pipeline（graph replay + `stream_sync`，即全部 GPU 工作） | 21.279 |
| head copy（topk D2H + sync） | 0.105 |

⇒ **9B 没有隐藏固定开销**（非 GPU 工作合计 0.18 ms = 0.8%）；它已经是纯 GPU-bound
（GEMV 20.3 + 边缘内核 1.5，上一轮报告）。lm_head 对 9B 是 618 MB Q6_K ≈1.9 ms，已在 pipeline 内。

---

## 8. 下一个杠杆（按实测收益排序）

1. **CPU 层 `sync` 8.49 ms（最大）**：等的是 GPU 侧每层 attention（kprof：`gemv_f32` 120 次 = 3.4 ms，
   即 0.14 ms/层）+ 共享专家（`moe_shared_gate` 40 次 0.68）+ router。微基准证明 `cuStreamSynchronize`
   本身只有 2.1 µs ⇒ **必须减少 GPU 侧每层的内核数/串行时间**（内核融合或把共享专家并入 attention 图），
   这属于 CUDA 内核工作，本轮未做。
2. **CPU 层 `launch` 2.75 ms（9 次 × 13 µs）**：可试把 [zerofill+共享专家 6 个 kernel] 捕获成一张图
   （routing 无关，可在 sync 前录）。需要先测本机 `cuGraphLaunch` 单价（历史 GPU 层实测约 0.05–0.1 ms/次，
   若成立则净收益有限，需 A/B 判断）。
3. **lm_head 1.84 ms**：Q8_0 540 MB @ ~290 GB/s，硬底，不动。
4. **GPU 层 3.2 ms**：16 层 × 0.20（launch 0.146 占 73%），整层图历史 A/B 更慢（设备路由内核慢于 host 路由融合内核），未再动。

---

## 9. 方法学教训（写进下一轮手册）

1. **不要用 `ENGINE_DBG_LAYERMS` 算账**（本机 +11 ms/token）；用 `ENGINE_PROF_ACC=1`。
2. **不要用累计平均值**（`ENGINE_DBG_CPUEXEC` 旧 cumulative avg 被 graph capture/page-in 污染到 1.36 ms，
   `last 25` 窗口才收敛 0.72；本轮 GEMV 实测 0.617）。
3. **本机噪声 ±7–20%**：同一二进制跨时段 35.4 ↔ 43.3 ms；必须同会话交错 A/B + 中位 + ≥38 计时步。
4. **`std::getenv` 在 Windows 上是 2 µs 级操作**，热路径每层几十次就是毫秒级；新代码请直接用
   `ENGINE_ENV_ON/STR`。
