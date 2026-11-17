# 复盘（Retrospective）— 自研本地推理引擎 v0

> 按 [engine-v0-spec-and-plan.md](https://github.com/Siselea) §4 的规则：G2 门槛未达成
> 即触发"停项目、归档、写复盘"。本文即该复盘。
>
> 关联文档：[FINAL-REPORT.md](FINAL-REPORT.md)（交付报告）、
> [SPEC-COMPLIANCE.md](SPEC-COMPLIANCE.md)（规格符合性与全部实验记录）、
> [README.md](README-v0-ARCHIVE.md)（用法）。

## 1. 一句话结论

**引擎做成了、验证了、也优化了 3.4–7.7 倍，但在性能上始终没有相对 llama.cpp 的入口**：
四个本机模型的速度是 llama.cpp 实测值的 **20%–33%**，远低于规格书改判后的
"不低于 llama.cpp 的 90%"判据。因此按 spec §11 的预设，本项目以**可维护性 / 学习资产**
的定位收官，性能目标不再推进。

## 2. 实际交付

| 维度 | 结果 |
|---|---|
| 代码 | 9,031 行 C++17 + CUDA（src 5,966 / include 3,065），测试 1,778 行 |
| 自研内核 | **67 个**（全部手写，经 NVRTC 运行时编译） |
| 可执行目标 | 16 个；工具脚本 32 个 |
| 架构覆盖 | llama-like dense、MoE（融合/分片/shared/CPU-GPU 分置）、MLA（DeepSeek-V2）、hybrid SSM（qwen35 / qwen35moe）、YaRN、CUDA Graph、safetensors + GGUF |
| 真实模型验证 | **4 个本机模型全部与官方参考实现一致**（40716 / 185 / 2614 / 69014） |
| 验证套件 | `bash tools/verify.sh` → **51 项 ALL GREEN**（含 torch 逐层对拍） |
| 工具链约束 | 无管理员权限：ziglang + NVRTC + 动态加载 cuBLAS/cudart |

## 3. 数字化的性能位置

| 模型 | llama.cpp（用户实测） | 本引擎 | 占比 |
|---|---|---|---|
| DeepSeek-Coder-V2-Lite | 18.17 tok/s | 3.4–4.3 tok/s | **21%** |
| Qwen3.6-35B-A3B | 25.76 tok/s | 3.3–5.1 tok/s | **13–20%** |
| Qwen3.5-9B | 46.7 tok/s | 15–16 tok/s | **33%** |
| qwen2.5-0.5b | （未测） | 143–164 tok/s | — |

本会话内的提升：Coder **1004 → 233–296 ms/token（3.4–4.3×）**、
35B **1497 → 195–302 ms/token（5–7.7×）**。

## 4. 什么起作用了（6 项，均有实测增益）

| # | 优化 | 增益 |
|---|---|---|
| 1 | MLA / 共享专家权重改为**按 GGUF 量化格式常驻**（原为 f32，每层 142 MB） | Coder −15% |
| 2 | **层粒度专家放置**（原为"每层均分专家"导致每层 host 往返） | 35B −52% |
| 3 | 修 qwen35 加载器绕过层粒度放置（19 GB 塞进 12 GB 卡） | 消除驱动 paging |
| 4 | 预算器不再无条件预留 f32 预填镜像（2 GB） | 多放 4 层专家 |
| 5 | 路由表**打包上传**（原为每专家 2 次同步拷贝，每 token 300+ 次） | 消除 0.3 ms/层 |
| 6 | **融合 MoE 内核** + **设备端路由**（消除每层 host 同步） | Coder −62% |

另有 AVX2 CPU 量化内核（Q4_K 0.23 → 28.6 GMAC/s，**124×**）——但实测表明
35B 的瓶颈不在 CPU 路径，所以未转化为端到端收益。

## 5. 什么没起作用（21 项负结果，全部记录在 SPEC）

按类别归纳：

- **量化/格式层**：dp4a、block-major 布局、专家流式（per-expert / arena）、fp16 NT/TN GEMM
- **并行/调度层**：CUDA 逐层图（正确但无收益）、GPU 时钟保温（WDDM 下挂死）、
  自研 GEMM（大模型慢 2×）
- **编译/工具层**：`-O3 -march=native`、`-march=alderlake`、CPU 自动向量化
- **测量层**：多次被 WDDM 提交批处理与 GPU 降频误导（见 §7）

## 6. 差距的根因（已量化）

1. **GPU 侧有效带宽只有 ~6.4 GB/s**（孤立 GEMV 测量 20–48 GB/s）→ 每层 61 MB / 9.6 ms。
   逐层图消除启动开销后**没有改善**，说明是执行效率问题，非调度问题。
2. **CPU 专家路径**（已修）：AVX2 后 28.6 GMAC/s，与 llama.cpp 同量级；
   但 35B 的瓶颈其实在 GPU 侧。
3. **GPU 降频**：35B 的 22/40 层在 CPU 上跑，GPU 长时间空闲 → 驱动降到 570 MHz
   （Coder 能维持 1965 MHz）。WDDM 下无法用自旋内核保温。
4. **cuBLAS 路径依赖**：prefill 仍走 dequant + cuBLAS；自研 GEMM 大模型慢 2×。

## 7. 过程教训（比性能数字更值钱）

1. **测量协议先于优化**。本会话至少有 3 次因为测量方法错误得出反向结论：
   WDDM 每 8 ms 强制提交一个批次、GPU 在 570–1965 MHz 之间漂移、探针里的
   `cuda::sync()` 把 0.1 ms 的操作测成 8 ms。加 `ENGINE_EXPERTS_TOPK` 扫描后，
   才把"每个额外活跃专家 +87 ms"与"孤立 GEMV 0.08 ms"对上，定位到驱动层。
2. **语义 bug 比性能 bug 更贵**。14 个真实缺陷里有 5 个是 DeepSeek-V2 语义
   （sigmoid/softmax、rope 半分裂、RMSNorm 零点、per-head kv_b、YaRN 参数），
   全靠与官方 transformers 逐层对拍才暴露。
3. **"看似正确"的指针/映射最容易错**。Q5_1 的 `type_id` 映射在"是否走设备路由"的
   判断里有、在"启动内核"的两处 lambda 里没有 → 设备路径被跳过而 host 路由也被跳过，
   结果无声地错。修好后 Coder 立刻快 2.6×。
4. **负结果必须立刻记账**。21 个负结果全部写进 SPEC，避免了重复尝试。

## 8. 如果继续，按性价比排序的下一步

1. **把 GPU 侧带宽从 6.4 GB/s 提上去**（唯一真正的瓶颈）。需要 Nsight/CUPTI 级别的
   profiling：确认是合并度、占用率还是 WDDM 批次问题。
2. **prefill 的量化 GEMM 换 tiled + smem 双缓冲**（当前自研版在 9B 上慢 2×，
   但小模型已快 8%，说明方向可行）。
3. **多槽并发 / vLLM 式连续批处理**（spec 的 G-E/G-F），这是 llama.cpp 单用户
   也不具备的能力，可能是唯一的"差异化入口"。
4. CPU/GPU 流水（让 35B 的 GPU 不再长时间空闲、避免降频）。

## 9. 判定

- **G0（IR 对拍）**：✅ 达成（8/8）
- **G1（自研 GEMM ≥ 85% cuBLAS）**：⚠️ 已实现且数值正确（四模型全对），
  性能未达标（大模型 44%，小模型 108%）
- **G2（性能 ≥ llama.cpp 90%）**：❌ 未达成（20%–33%）
- **结论**：按 spec §4 的规则，**v0 项目在 G2 止损点结束**。交付物是一套
  **可运行、可验证、文档完备**的推理引擎与一份**完整的性能归因报告**；
  没有交付"比 llama.cpp 更快的推理"。

---

*复盘完成日期：2026-11-14。全部数字可在本机复现：`bash tools/verify.sh`、
`bash tools/smoke_models.sh`、`bash tools/verify_reference.sh`。*
