# hostlayer-35b 交付报告（2026-11-15）

任务：Qwen3.6-35B-A3B decode 49 ms → ≤40 ms（≥25 tok/s），四模型预测 + 三套验证保持全绿，
Coder ≥22 / 9B ≥36 / 0.5B ≥160 tok/s 不回退。

## 1. 结论先行

| 指标 | 旧构建 | 新构建 | 变化 |
|---|---|---|---|
| 35B decode（双二进制交错 A/B，n=17，稳态中位） | **51.9 ms**（19.3 tok/s） | **47.6 ms**（21.0 tok/s） | **-8.3%** |
| 35B decode（独立利用率 lab，n=17 median） | 50.09 ms（20.0 tok/s） | **45.35 ms（22.0 tok/s）** | **-9.5%** |
| 35B GPU 层墙钟 / 提交 / 同步 | 0.620 / 0.236 / 0.206 ms | **0.203 / 0.095 / 0.000 ms** | GPU 层 -67% |
| 35B GPU util（NVML 均值） | 43.5% | **46.4%** | +2.9 pt |
| 35B 进程 CPU | 3.57 core | **4.01 core** | 重活更多落在 CPU 提交/主机侧 |
| 长 decode n=257 | 第 25 token 挂死 | **256/256 步跑完** | 挂死修复 ✓ |
| Coder | 44.24 ms | **42.23 ms** | -4.5%（自动放置 17 层） |
| 9B | 26.53 ms（本机同会话） | 26.30 ms | 持平 |
| 0.5B | 6.99 ms | 5.55 ms | 持平/更快 |

**40 ms 目标未达成**（最好 45.35–47.6）。剩余瓶颈的精确分解见 §5。

四模型预测 top-1：**40716 / 185 / 2614 / 69014 全对**；生成序列与旧二进制**逐 token 相同**
（35B n=17 的 17 个 token md5 一致；Coder/9B/0.5B 同）。三套验证：
`tools/verify.sh` **ALL GREEN**、`tools/smoke_models.sh` 全对、`tools/verify_generality.sh`
**GENERALITY GREEN（26/26）**。

## 2. 路线选择（数据推翻原建议）

原任务推荐"把 CPU 层的注意力整个搬到主机"。用 `tools/bench_cpu_layer.exe matvec` 实测
35B 注意力投影的主机耗时（9 线程、16 副本轮转模拟 DRAM 压力）：

| 35B 单层注意力投影 | 主机耗时 |
|---|---|
| SSM attn_qkv 8192×2048 Q6_K（13.1 MB） | 0.687 ms |
| SSM attn_gate(z) 4096×2048 Q4_K | 0.244 ms |
| SSM ssm_out 4096×2048 Q4_K | 0.244 ms |
| **SSM 层合计（23 MB）** | **≈1.18 ms** |
| full-attn 层 q/k/v/o | ≈0.84 ms |

GPU 做同样 23 MB 只要 ~0.12 ms（+每层往返 ~0.35 ms 总开销）。**整层主机化净增 0.4–1.1 ms/层**，
25 层 = +10…+28 ms，比省下的 0.75 ms/层更贵 ⇒ **A1 路线否掉**。

改为"注意力留 GPU，只削 GPU 侧每层的固定开销"。逐层实测（`ENGINE_DBG_LAYERMS`）显示：
每层 20–40% 的墙钟是**把 kernel 递给 WDDM**（本机一次提交 10–20 µs、一层 15–26 个 kernel），
而 GPU 真正执行只有 ~0.17 ms/层。外部利用率取证：35B decode **SM 时钟 1950 MHz（无降频）、
GPU util 43.5–53%** ⇒ GPU 是"闲"不是"降频"，方向确定为**消同步 + 消提交**。

## 3. 关键实现（3 项 + 1 个 bug 修复，全部有回退开关）

1. **GPU 驻留层设备端路由 `dev_route`（默认开）**：`k_moeroute_` 在设备端算 top-k，
   `k_moegateupdev_`/`k_modeowndev_` 直接读设备端专家指针数组，替掉原来的
   "router D2H + hn D2H + cuStreamSynchronize + softmax/top-k + 路由表/指针数组 H2D"。
   **GPU 层 0.591 → 0.275 ms/层**。安全门：专家张量含 `Q5_1` 时整模型自动回退旧路径
   （Q5_1 的设备 GEMV 分支数值未验证；Coder 的 14 个 Q5_1 张量因此不受影响）。
2. **GPU 驻留层 attention/mixer 段逐层 CUDA Graph `layer_graph_auto`（默认开）**：
   该段 ~15 个小 kernel 录成一张图，decode 时 1 次 `cuGraphLaunch` 重放
   （**0.275 → 0.199 ms/层**）。位置无关性：position 全走设备缓冲 `pos_dev_`；full-attn 的
   warp 数按 ctx 分桶（≤96/≤384/其它）各存一张图。**懒捕获**：token 1 正常执行（顺带完成惰性
   分配）、token 2 录制、token 3 起重放（捕获开销不进稳态）。
   - 陷阱（如实记录）：**在构造函数里捕获的图，重放会 illegal memory access**——用 9B 模型
     最小复现到"单 kernel 图也崩"，同样代码在 forward 里捕获就正常。因此默认懒捕获，
     `ENGINE_GRAPH_PRECAPTURE` 只留作诊断。
   - CPU 驻留层的 attention 图默认**不开**（`ENGINE_CPU_LAYER_GRAPH=1` 才开）：提交时间能降
     0.24→0.14 ms，但 flush 时序被打散、同步从 0.15 涨到 0.30 ms，CPU 层净变慢，实测否掉。
3. **CPU 线程池丢唤醒修复**：`cpu::Pool::run()` 原来在 mutex 外 `gen_.fetch_add`+`notify_all`，
   worker 在"predicate 检查→cv_.wait"之间被调度走就永久睡死，主线程自旋等 `remaining_`
   ⇒ 整进程挂死（GPU 空闲/CPU 空转）。修复：gen 递增与 notify 放进同一把 mutex。
   外部取证确认：旧构建 35B n=257 第 25 token 必挂、Coder 第 ~77 token 必挂；新构建两个都跑完
   256/256 步，`[placement] adaptive:` 减层路径也不再挂。

## 4. 失败尝试（如实）

- **整层主机化（原建议 A1）**：被 §2 的实测否掉（主机注意力投影 1.18 ms/层 > 省下的 0.75）。
- **CPU 层 attention 段图**：提交省 0.1 ms/层，但同步涨 0.15 ms/层，净亏 ⇒ 默认关。
- **构造函数里预捕获图**：重放的图非法访存（最小复现），改懒捕获。
- **`--vram-margin 500` 多放 GPU 层**：adaptive 检查把 17 层退回 16 层，收益 <1 ms；`--no-bm`
  （省 515 MB）更慢。均未采用。
- **`ENGINE_GPU_FULL_LAYER_GRAPH`（GPU 层整层录图）**：实现保留为诊断开关，未作为默认
  （整层图需设备路由，且收益未超过仅 attention 段）。
- 早期两次挂死（GPU 空闲、CPU 空转）最终定位到丢唤醒，而不是 "WDDM 竞争/显存抢占"。

## 5. 未达成 40 ms 的精确分解（n=17 A/B，新构建 45.4–47.6 ms）

| 项 | 数值 | 说明 |
|---|---|---|
| 15 个 GPU 层 | 15 × 0.203 = **3.0 ms** | 已接近"1 图 + 设备路由 MoE"下限 |
| 25 个 CPU 层 | 25 × 1.64 = **41 ms**（本机同会话口径） | 主缺口 |
| 收尾：final norm + lm_head Q8_0 515 MB GEMV + logits D2H | ≈**3 ms** | lm_head 本身 ~1.5–2 ms（DRAM 地板） |
| CPU 层内部 | host matvec 0.87–1.0（60%+）、提交 0.24、同步 0.15–0.34 | matvec 已达 22–23.5 GB/s（峰值 23.7） |

CPU 层的 host matvec 已贴硬件带宽上限，没有余量；剩余是 WDDM 的提交/刷写固定成本。
继续接近 40 ms 只剩三条路（本轮未做）：
① 把 "ln2 rmsnorm + router + zerofill + shared expert" 并入同一张图（需把 run_moe 的 GPU 段
与 host 段拆开，预计 -1 ms 量级）；
② 更多 GPU 层（显存只剩 1.7 GB，`--vram-margin` 下探被 adaptive 检查挡回，收益 <1 ms）；
③ 主机 matvec 换算法/更大页（当前 DRAM 受限，需要 OS 层改动）。

## 6. 复现

```bash
bash build.sh && bash tools/verify.sh            # ALL GREEN
bash tools/smoke_models.sh                       # 40716 / 185 / 2614 / 69014
bash tools/verify_generality.sh                  # GENERALITY GREEN (26/26)

# 35B A/B（本机是虚拟机，长跑会漂移，必须双二进制背靠背交错）
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ./build/test_gen.pre.exe "<35B>" 760 17 --gpu-layers 15 --no-vram-check
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ./build/test_gen.exe     "<35B>" 760 17 --gpu-layers 15 --no-vram-check

# 逐层分解
ENGINE_DBG_LAYERMS=1 ENGINE_DBG_MOE_PHASES=1 ./build/test_gen.exe "<35B>" 760 3 --gpu-layers 15 --no-vram-check

# 关闭新行为（回到旧路径）
ENGINE_NO_DEV_ROUTE=1 ENGINE_LAYER_GRAPH_AUTO=0 ...
```

改动文件：`src/config.cpp`（2 个新配置键）、`src/runner.cpp`、`include/engine/runner.h`、
`src/qwen35.cpp`（attention 拆分 + 图机制）、`src/cpu_matvec.cpp`（丢唤醒修复）、
`src/cuda.cpp`/`include/engine/cuda.h`（图捕获诊断）、`tests/test_gen.cpp`（`ENGINE_DEC_SKIP`）、
`tools/bench_ab_final.sh`、`docs/CONFIG.md`、`SPEC-COMPLIANCE.md` §30。
`include/engine/kernels.h` **未改动**（dp4a GEMV 数学保持原样），`build.sh` 未改。
