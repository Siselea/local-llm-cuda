# 报告索引 / Report index

本目录归档引擎开发过程中的设计、验证与性能报告，按主题查阅即可。

- 所有数字均为报告写作时的本机实测（RTX 3060 12 GB + 32 GB RAM，Windows 11 VM）；
- 当前对外性能数字以根目录 `README.md` 为准；
- 历史报告中的路径/模型名是开发环境记录，不代表仓库依赖。

| 文件 | 标题 | 一句话说明 |
| --- | --- | --- |
| [CTX-AND-EXTREMES.md](CTX-AND-EXTREMES.md) | 任意上下文长度 + 极端情况矩阵（CTX-AND-EXTREMES） | `--ctx` 任意正整数、只在物理放不下时夹紧、requested/actual 可观测；极小/极大 ctx、极短/极长 prompt、显存紧张、30 轮长会话、空输入的实测矩阵与问题清单。 |
| [CPU-GEMV-BW-REPORT.md](CPU-GEMV-BW-REPORT.md) | CPU 量化 GEMV 带宽优化 — 交付报告 | CPU 量化 GEMV 的带宽/DRAM 天花板测量与优化（cpu_matvec 行内核）。 |
| [CPU-OCCUPANCY-REPORT.md](CPU-OCCUPANCY-REPORT.md) | decode 期间 CPU 占用实测报告（CPU-OCCUPANCY） | decode 期间 CPU 占用实测：为何用户看到 CPU 不忙、如何让专家层吃满 CPU。 |
| [CPU-REPACK-REPORT.md](CPU-REPACK-REPORT.md) | CPU 专家路径「权重 repack / 带宽」方向 — 取证与结论报告 | CPU 专家路径「权重 repack / 带宽」取证与结论（含两级 int8）。 |
| [DEADCODE-AUDIT.md](DEADCODE-AUDIT.md) | 去写死审计（DEADCODE AUDIT）· 第二轮：把"仍写死"清零 | 去写死审计：清理源码中硬编码模型假设与魔法常量。 |
| [EDGE-KERNELS-REPORT.md](EDGE-KERNELS-REPORT.md) | 9B 边缘内核优化报告（EDGE-KERNELS） | 9B 边缘内核优化：decode 26.4→目标 ≤23 ms/token，四模型不回退。 |
| [FEATURE-MATRIX.md](FEATURE-MATRIX.md) | FEATURE-MATRIX — 功能面审计 · 放置决策器 · 交互式启动 | 功能面审计：放置决策器、交互式启动、101+ 配置项映射。 |
| [FINAL-REPORT.md](FINAL-REPORT.md) | engine v0 — 最终交付报告 | engine v0 最终交付报告：M0–M5 里程碑、验证证据与 G 门槛结论。 |
| [FINAL-SPRINT-REPORT.md](FINAL-SPRINT-REPORT.md) | 最后冲刺报告（FINAL-SPRINT）：9B 与 35B 达到 llama.cpp 的 100%+ | 最后冲刺：9B/35B decode 达到 llama.cpp 100%+ 的优化记录。 |
| [GENERAL-OPT-REPORT.md](GENERAL-OPT-REPORT.md) | GENERAL-OPT 报告（2026-11-15，长上下文 + CPU 量化 GEMV + 去写死审计） | 通用优化汇总：长上下文、CPU 量化 GEMV、去写死审计。 |
| [GENERALITY-AUDIT.md](GENERALITY-AUDIT.md) | 泛化性审计：硬编码模型假设与魔法常量 | 泛化性审计：硬编码模型假设与魔法常量清单。 |
| [GENERALITY-REPORT.md](GENERALITY-REPORT.md) | 通用化改造交付报告（v1） | 通用化改造交付（v1）：配置面、自适应显存放置、优雅失败。 |
| [ARCH-GENERALITY.md](ARCH-GENERALITY.md) | 按 metadata 驱动的架构泛化：per-layer 几何 + 通用层块（gemma4） | 不按 arch 名特判：per-layer 注意力几何/可选算子从 metadata 推导；真实 gemma4-12B 可加载+生成，数值 parity 未完成（默认关闭）。 |
| [HOSTLAYER-35B-REPORT.md](HOSTLAYER-35B-REPORT.md) | hostlayer-35b 交付报告（2026-11-15） | 35B CPU 专家层调度与主机侧带宽分析。 |
| [KV-LONGCTX-REPORT.md](KV-LONGCTX-REPORT.md) | KV cache 量化 + 长上下文 attention + 上下文上限自适应（KV-LONGCTX-REPORT） | KV 量化与长上下文报告：q8_0/q4_0 KV、ctx 自适应与长上下文实测。 |
| [LONGCTX-REPORT.md](LONGCTX-REPORT.md) | 长上下文（4k / 2k 预填）报告：一个必然崩溃的 bug + 修复后的数据 | 长上下文（2048/4k/8k）填充与性能实测。 |
| [LOGITS-ROW-CAP.md](LOGITS-ROW-CAP.md) | prefill 只保留 1 行 logits，让 chunk 变大（LOGITS-ROW-CAP） | logits 从 rows×vocab 改为 1 行（prefill head 只算 1 行）+ 补齐 SSM/MLA/MoE 每行激活估算：9B ctx32768 chunk 1024→2048，11k −3.2%、16.2k −6.0%，greedy 序列不变，五套全绿。 |
| [DEQUANT-X8.md](DEQUANT-X8.md) | 反量化内核提速（DEQUANT-X8） | 一个 warp 一个 super-block + 16B 向量化 store：181–207 GB/s → **303–308 GB/s（1.48–1.70×）**，fp16 slab 与旧内核**逐字节一致**；真实 prefill 4.4k −4.4%、13.4k −2.6%；五套全绿。 |
| [FUSED-GEMM-PROTO.md](FUSED-GEMM-PROTO.md) | fused dequant+GEMM 最小原型（负结果 + 修正判断） | 单内核 fused（shared dequant + mma.m16n8k16）数值正确（rel_l2≤7e-6）但 naive 版慢 50×；实测 dequant 是计算瓶颈（115 GB/s）而非流量，修正"tail 190→85 ms"为"~190→140-150 ms"，建议先做层主序 + 反量化提速。 |
| [LAYER-MAJOR-PREFILL-PLAN.md](LAYER-MAJOR-PREFILL-PLAN.md) | 层主序 prefill 设计（LAYER-MAJOR-PREFILL-PLAN） | 设计文档：把 chunk-major 改成 layer-major（每权重矩阵 dequant 一次），预期长文档 −8~−10%；tail 用「小 n CUDA graph」而不是层主序。 |
| [MTP-IMPL-REPORT.md](MTP-IMPL-REPORT.md) | MTP 投机解码 实施报告（MTP-IMPL） | MTP 投机解码实现报告（实验特性，默认关闭）。 |
| [MTP-RECON.md](MTP-RECON.md) | MTP / 投机解码 可行性报告（MTP-RECON） | MTP 机制逆向分析：权重布局、草稿/验证流程与可行性评估。 |
| [OVERHEAD-17MS-REPORT.md](OVERHEAD-17MS-REPORT.md) | 17 ms 可回收开销 — 取证、实现与结论报告（OVERHEAD-17MS） | decode 17 ms 额外开销定位与优化报告。 |
| [PRE-FILL-REPORT.md](PRE-FILL-REPORT.md) | PRE-FILL 报告：prefill 从 llama.cpp 的 1/4 到与之同量级（多行量化 GEMM + 注意力分块 + SSM 状态常驻） | prefill fp16 tensor-core GEMM 优化：9B 1.6k prompt 333→1046-1097 tok/s。 |
| [PLACEMENT-VRAM-MARGIN.md](PLACEMENT-VRAM-MARGIN.md) | 自动分层显存余量修复（PLACEMENT-VRAM-MARGIN） | 自动分层漏算"账本看不见"的 1.1-2.1 GiB 开销（驱动/cuBLAS/分配器）→ 装载后可能 0 MiB free、驱动换页（3 tok/s）：新增可标定 vram_extra_reserve_mb=1024、逐层自适应检查纳入该预留、装载后不足改为明确 ERROR；3×generality + genreg + smoke 全绿。 |
| [PREFILL-CLIFF-REPORT.md](PREFILL-CLIFF-REPORT.md) | PREFILL-CLIFF 报告：用户可感知的两个 prefill 慢（首 token / 长文档） | serve prefill chunk 128→512/1024 自适应 + 装载期按实际显存定 rows + 超长 prompt 400：长文档 prefill 1.85–2.0×，冷启动惩罚消除。 |
| [Q6K-DP4A-RESULTS.md](Q6K-DP4A-RESULTS.md) | Q6_K dp4a GEMV 改造（2026-11-14） | Q6_K DP4A int8 内核的结果与边界分析。 |
| [QWEN35-SEMANTICS-FIX-REPORT.md](QWEN35-SEMANTICS-FIX-REPORT.md) | Qwen3.5-9B / Qwen3.6-35B-A3B 真实文本生成修复报告（GGUF 张量语义） | qwen35/qwen35moe GGUF 语义修复：llama.cpp 约定 vs HF 语义，逐 token 一致。 |
| [RETROSPECTIVE.md](RETROSPECTIVE.md) | 复盘（Retrospective）— 自研本地推理引擎 v0 | 项目复盘：G2 止损点、根因与经验。 |
| [SERVE-REPORT.md](SERVE-REPORT.md) | HTTP 服务 + 分词器 + 采样器 交付报告 | HTTP serve 收口报告：OpenAI 兼容、SSE、webui、前缀缓存、思考漂移与采样。 |
| [SPEC-COMPLIANCE.md](SPEC-COMPLIANCE.md) | engine v0 — 规格符合性与门槛实测报告 | 规格符合性逐条审计（G0/G1/G2 门槛与证据）。 |
| [UTILIZATION-REPORT.md](UTILIZATION-REPORT.md) | 利用率与流水线气泡取证（UTILIZATION-REPORT） | GPU/CPU 利用率测量报告（含 CONTAMINATED 标注的历史数据说明）。 |
| [WEBUI-LAUNCHER-REPORT.md](WEBUI-LAUNCHER-REPORT.md) | llama.cpp webui + Win32 launcher + 模型扫描（2026-11-15） | llama.cpp webui + Win32 launcher：全盘模型扫描/缓存、GUI 交互与真实用户验收。 |
| [WIN-PRIORITY-REPORT.md](WIN-PRIORITY-REPORT.md) | Windows 调度优先级 + auto-margin 换层 — 实测报告（WIN-PRIORITY） | Windows 进程/线程优先级实测报告（win_priority 的真实收益）。 |
| [THREAD-AFFINITY-REPORT.md](THREAD-AFFINITY-REPORT.md) | 线程数 × SMT 成对绑核 — 实测报告（THREAD-AFFINITY） | 12 线程 SMT 成对绑核 vs 9 线程：CPU 专家层 +8–12%，及 8×2 配置扫描。 |
| [MOE-GAP-REPORT.md](MOE-GAP-REPORT.md) | 35B MoE decode 与 llama.cpp 差距 — 取证报告（MOE-GAP） | 同模型同 GPU 层数下，llama.cpp 与自研引擎 decode 差距的逐层归因。 |

## 已清理 / removed during repository cleanup

- `README-v0-ARCHIVE.md` — 开发期旧 README（历史数字与里程碑），已被仓库根目录 README 取代；需要考古可在 git 历史中查看。
