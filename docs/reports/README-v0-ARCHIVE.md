# engine — 本地推理引擎 v0

> 最终交付报告：**[FINAL-REPORT.md](FINAL-REPORT.md)**　|　复盘（G2 止损点归档）：**[RETROSPECTIVE.md](RETROSPECTIVE.md)**
> 门槛与证据：**[SPEC-COMPLIANCE.md](SPEC-COMPLIANCE.md)**；工具索引：**[tools/README.md](../../tools/README.md)**
> 通用化改造：**[GENERALITY-AUDIT.md](GENERALITY-AUDIT.md)**（硬编码审计）、
> **[GENERALITY-REPORT.md](GENERALITY-REPORT.md)**（配置面/自适应交付报告）、
> **[docs/CONFIG.md](../CONFIG.md)**（配置手册）　|　**[FEATURE-MATRIX.md](FEATURE-MATRIX.md)**（功能面审计 / 放置决策器 / 交互式启动）

## 一句话状态

自研 C++17 + CUDA 推理引擎 v0（无管理员工具链：ziglang + NVRTC + 动态加载 cuBLAS）。

**四个本机真实模型全部加载、生成，并与官方参考实现对拍通过**：

| 模型 | 架构 | 验证基准 | 结果 |
|---|---|---|---|
| qwen2.5-0.5b Q4_K_M | llama-like dense | llama.cpp | 端到端 5.9e-06、短生成 6/6 一致 |
| DeepSeek-Coder-V2-Lite Q5_K_M | MLA + MoE | **官方 transformers `deepseek_v2`** | 预测完全一致（185 / -0.003） |
| Qwen3.5-9B Q5_K_M | hybrid SSM (Gated DeltaNet) | **llama.cpp（GGUF 语义）** | 7 组 prompt 逐 token 一致（含中文/chat 模板，8–12 token/组，见 `QWEN35-SEMANTICS-FIX-REPORT.md`） |
| Qwen3.6-35B-A3B Q4_K_M | hybrid SSM + MoE | **llama.cpp（GGUF 语义）** | 与 9B 同标准逐 token 一致；单 token 2614 |


关键数字（2026-11-15 收口验收；RTX 3060 12 GB + 32 GB RAM VM）：

- `bash tools/verify.sh` → **ALL GREEN**（单测 + G0 IR 对拍 + M2 plan 对拍 + 生成一致性）
- `bash tools/smoke_models.sh` → **40716 / 185 / 2614 / 2614**（四模型单 token top-1 与
  llama.cpp / 官方参考实现一致）
- `bash tools/gen_regression.sh` → **10/10**：真实多 token prompt + 8–16 个 greedy token
  与 llama.cpp **逐 id 一致**（0.5B / Coder / 9B / 35B 全部覆盖）
- `bash tools/verify_generality.sh` → **GREEN**（配置三通道 / 内核校准缓存 / 自适应显存放置 /
  优雅失败 / `engine run`）
- `bash tools/verify_serve.sh` → **PASS**（HTTP 端到端：SSE、tokenize、与 test_gen 贪心对拍、
  `/shutdown`；非交互启动行为与脚本兼容）
- decode 中位（本机）：**0.5B 192 / 9B 45 / 35B 25 / Coder 23.4 tok/s**（约本机 llama.cpp 的 99–129%）
- prefill：9B 1.6k prompt **1046–1097 tok/s**（llama.cpp 的 79%；fp16 tensor core GEMM）
- 架构覆盖：llama-like dense、MoE（融合/分片/shared/CPU-GPU 分置）、MLA（deepseek2）、
  hybrid SSM（qwen35 / qwen35moe）、CUDA Graph、YaRN

已知差异的正确认知（2026-11-15 修正）：**GGUF qwen35/qwen35moe 文件是按 llama.cpp
约定写入的** —— `qwen35.ssm_a` 存的是 `-A`（不是 log A），SSM 的 q/k→v head 广播是
**块重复**（value head h 读 key head h % n_key_heads），而官方 HF `qwen3_5` 模块用的是
log A + repeat_interleave。此前引擎忠实实现了 HF 公式却套在 llama.cpp 语义的数据上，
且拿 HF 参考当基准，所以"逐层 1e-6 一致"掩盖了真实文本生成崩溃（单 token 恰好仍对）。
现在 GGUF 路径按 llama.cpp 语义、safetensors 路径按 HF 语义（`src/qwen35.cpp` 的
`container_` 分路 + `ENGINE_Q35_LOG_A` / `ENGINE_Q35_HF_BCAST` 回退开关）。
G-A..G-F 性能门槛未达成（差距与根因见 FINAL-REPORT §3.9/§3.10 与 SPEC-COMPLIANCE.md §5.2b）。

按 [engine-v0-spec-and-plan.md](https://github.com/Siselea) 实施。

## 里程碑

| 里程碑 | 状态 | 验证 |
|---|---|---|
| M0 工程骨架 + 工具链 | ✅ | `bash build.sh` |
| M1 IR + GGUF/safetensors 解析 + 校验 | ✅ | G0 对拍 8/8 模型 |
| M2 预算器 + 放置计划 | ✅ | M2 对拍 8/8 模型；真实 35B 算出 91/256 专家 |
| M3 单层执行 | ✅ | 单层 2.1e-08；5 格式反量化逐位一致 |
| M4 分段专家 | ✅ | 5 种 CPU/GPU 切分全 2.7e-07 |
| M5 调度 / KV / CUDA Graph | ✅ | 增量 decode 1.4e-07 ~ 1.7e-06 |
| M6 hybrid SSM（超出原规格） | ✅ | Qwen3.5-9B 逐层 1.7e-06；35B 逐层 5e-06 |

一条命令验证全部：`bash tools/verify.sh`（51 项）。**真文本生成回归**：
`bash tools/gen_regression.sh`（多 token prompt + 8–12 greedy token vs llama.cpp，8 组）。
四个真实模型的冒烟测试：`bash tools/smoke_models.sh`（**注意：单 token smoke 无法发现
qwen35 GGUF 语义错误，必须跑 gen_regression**）。

## 快速开始

> 工具脚本清单见 [tools/README.md](../../tools/README.md)。**配置手册见 [docs/CONFIG.md](../CONFIG.md)**。

```bash
bash build.sh                                   # 构建（需要 ziglang + pip 的 CUDA 运行时包）
bash tools/verify.sh                            # 全量验证（51 项）
bash tools/gen_regression.sh                    # 多 token 生成回归 vs llama.cpp（8 组）
bash tools/verify_generality.sh                 # 通用化检查（配置/自适应/优雅失败/校准）

./build/engine.exe info  <model.gguf>           # 解析 + IR 摘要
./build/engine.exe plan  <model.gguf>           # 预算与放置计划
./build/engine.exe run --model <model.gguf> --ids 760 --n 8
./build/engine.exe --print-config               # 打印生效配置（cli > config > env > default）
./build/engine.exe --config engine.cfg --print-config
./build/test_gen.exe --model <model.gguf> --ids 760 --n 8 --ctx 512 --gpu-layers 12
./build/test_gen.exe --model <model.gguf> --ids 760 --n 1 --kernel-calibrate

# 双击/一键启动（构建后产出 build/launcher.exe：Win32 深色 GUI，无控制台窗口）
./build/launcher.exe                            # 模型 + GPU层/上下文滑块 + 全部开关（高级面板由 serve --print-config --json 生成）
python build/test_launcher.py                   # 验收：4 张截图 + 100% 选项覆盖 + 启动/停止验证
启动.bat                                        # 无 GUI 环境的控制台兜底（纯 ASCII，支持 启动.bat 5 4096）
./build/engine.exe models                      # 扫描模型（--tsv/--json/--plain），GUI/bat/CLI 共用
./build/engine.exe serve --pick 5 --port 7030  # 用列表第 5 个模型启动
./build/engine.exe serve --model auto --port 7030 --open-browser
```

浏览器打开 `http://127.0.0.1:7030/` 即 **llama.cpp 官方 webui**（MIT 单文件，静态放在
`build/webui/index.html`；无 CDN、离线可用；缺失时回退到内置极简页）。详见
[docs/SERVE.md](../SERVE.md) 与 `WEBUI-LAUNCHER-REPORT.md`。

### 配置与自适应（v1 新增）

- **统一配置**：所有运行时旋钮集中到 `src/config.cpp` 的注册表；`--config <file>`（key = value）、
  常用 CLI（`--ctx` / `--threads` / `--gpu-layers` / `--vram-margin` / `--no-graph` / …）、
  环境变量兼容层；优先级 **CLI > config > env > 默认**；`--print-config` 显示每项来源。
  完整键表、配置文件示例、环境变量清单：**[docs/CONFIG.md](../CONFIG.md)**。
- **自适应显存**：`vram_margin_mb`（默认 1200）在装载每层专家前用 `cuMemGetInfo` + 剩余分配量
  实测校验，不足时自动减少 GPU 层数并打印决策依据；装载后再校验一次。
- **内核选择集中化**：`choose_kernel(type, n_out, n_in)`（`src/kernel_select.cpp`）是唯一的
  kernel 分派策略；`--kernel-calibrate` 在本进程内对真实形状做微基准，把胜出方案写入
  `engine-kernel-cache.txt` 并自动加载（兼容性校验会拒绝形状不匹配的旧缓存）。
- **优雅失败**：不支持的架构/量化类型/张量命名给出"缺什么 + 支持列表"的错误；
  新增 shape/metadata 一致性校验（head×head_dim、rope 维度、专家数上界、MLA/SSM 几何、
  MoE 层完整性等），在校验阶段就明确报错而不是加载到一半崩溃。

```bash
# 配置文件示例（可直接复制 engine.cfg.example）
cat > engine.cfg <<'EOF'
model          = <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf
ctx            = 512
vram_margin_mb = 1200
gpu_layers     = auto
EOF
./build/engine.exe run --config engine.cfg --ids 760 --n 8
```

调试开关（v1 起统一走配置系统，环境变量仍兼容）：

| 变量 | 对应配置键 / CLI | 作用 |
|---|---|---|
| `ENGINE_GPU_EXPERTS=N` | `gpu_experts` / `--gpu-experts` | 每层前 N 个专家放显存，其余走 CPU |
| `ENGINE_GPU_LAYERS=N` | `gpu_layers` / `--gpu-layers` | 放显存的 MoE 层数（auto = 按显存自适应） |
| `ENGINE_VRAM_MARGIN_MB` | `vram_margin_mb` / `--vram-margin` | 装载后必须剩余的空闲显存（自适应放置依据） |
| `ENGINE_VRAM_BUDGET_MB=M` | `vram_budget_mb` / `--vram-budget` | 用模拟显存预算驱动专家放置 |
| `ENGINE_LAYERS=N` | `layers` / `--layers` | 只跑前 N 层（性能分解） |
| `ENGINE_NO_GRAPH=1` | `no_graph` / `--no-graph` | 禁用 CUDA Graph |
| `ENGINE_PREFILL_F32=0/1` | `prefill_f32` / `--prefill-f32` | 是否保留 f32 权重镜像（batch>1 用） |
| `ENGINE_NORM_PLUS1=0/1` | `norm_plus1` / `--norm-plus1` | 强制 RMSNorm 零中心解释（默认自动判定） |
| `ENGINE_ROUTER` / `ENGINE_NORM_TOPK` | `router` / `norm_topk` | 覆盖 MoE 路由方式与 top-k 归一化 |
| `ENGINE_NO_YARN=1` | `no_yarn` / `--no-yarn` | 关闭 YaRN rope |
| `RUNNER_DEBUG=1` | `debug` / `--debug` | 打印放置/内核选择/显存决策依据 |
| `ENGINE_MLA_Q_LAYOUT` / `ENGINE_MLA_KV_LAYOUT` | `mla_q_layout` / `mla_kv_layout` | MLA 布局实验 |

**完整配置键表、类型/默认值/范围、配置文件格式、优先级规则、全部兼容环境变量清单见
[docs/CONFIG.md](../CONFIG.md)。** 下面只列仍然只能用环境变量的诊断/转储开关：

实验性优化开关（实测收益为负或不确定，默认关闭，仅作记录）：
`ENGINE_DP4A`（int8 dp4a GEMV）、`ENGINE_HALF`（fp16 NT 预填）、
`ENGINE_HALF_T`（fp16 TN 预填）、`ENGINE_BM_TYPES`（block-major 权重）、
`ENGINE_NO_SHARED` / `ENGINE_MOE_SWAP_GU` / `ENGINE_SH_SWAP_GU`。

qwen35 GGUF 语义开关（**默认值已按 GGUF 文件修正；这些只用于 A/B 与 safetensors**）：
`ENGINE_Q35_RAW_A`（强制 ssm_a = -A，默认对 GGUF 已是）、`ENGINE_Q35_LOG_A`
（回退到 HF 的 log(A) 约定）、`ENGINE_Q35_HF_BCAST`（回退到 HF 的 repeat_interleave
q/k 广播）。裁判工具：`tools/q35_variants.py`（见其文件头说明）。

调试 dump：`ENGINE_DUMP=<prefix>`（关键张量）、`ENGINE_DUMP_EVERY=<prefix>`（逐层）、
`ENGINE_DUMP_LAYER=N`、`ENGINE_DUMP_MLA`、`ENGINE_DUMP_Q35`、`ENGINE_DUMP_ROPE`、
`ENGINE_LOGITS=<file>`、`ENGINE_TOPK=1`（打印 top-5）、`ENGINE_TARGET=<id>`（打印指定 token 的排名）。

## 放置策略：gpu / vram+ram split / ssd（v1.1 新增）

启动 `serve` / `run` / `plan` 时，引擎在**加载权重前**打印一份 `[placement]` 账目并给出放置决策：

| 判定 | 条件 | 路径 |
|---|---|---|
| `gpu` | fixed + experts + KV(ctx) + workspace ≤ 可用 VRAM | 全部权重常驻显存（最快） |
| `vram+ram split` | 放不下显存，但 fixed+KV+workspace 放得下，且剩余专家字节 ≤ 可用 RAM（仅 MoE） | 现有 CPU 专家卸载路径（层粒度，实际层数由装载时实测显存决定） |
| `ssd stream` | 连 VRAM+RAM 都装不下 | **候选**，默认关闭；本版本未实现（接口占位 + 确定性报错），必须显式 `--ssd-stream` |

原则与用户要求一致：**“显存+内存放得下就没有必要 SSD”**。SSD 只在装不下时才是候选，
且默认关闭（PCIe×8/机械盘上流式通常净负收益：本机实测 CPU lane 4.4 GB/s、PCIe H2D 12 GB/s，
一次专家 miss 就是几十~几百 MiB 的 I/O）。`--placement auto|gpu|split|ssd` 可显式指定策略；
`--host-margin MB` 控制保留的主机内存（0=自动 max(1 GiB, RAM/16)）；`--allow-oversubscribe`
才会在“装不下”时强行让驱动分页。完整账目字段、四模型实测与门控规则见
**[FEATURE-MATRIX.md](FEATURE-MATRIX.md) §1–§2** 与 [docs/CONFIG.md §6.3](../CONFIG.md)。

```bash
# 离线看账目（不加载权重；--vram-gib 可模拟显存预算）
./build/engine.exe plan <model.gguf> --ctx 4096
# -> [placement] ledger : model ... = fixed ... + experts ...
#    [placement] vram   : free ... MiB, reserve ... -> usable ...
#    [placement] decision: gpu | vram+ram split | ssd stream
```

## 交互式启动（v1.1 新增）

在终端里启动 serve/run 时，会在**加载模型之前**询问（全部 Enter 直过）：

1. **KV 量化类型**：`f32 / q8_0 / q4_0`，并打印三档 KV/token 与各自可支持的最大 ctx 推荐值；
   MLA 模型自动跳过（保持 f32）。默认 f32（保持历史/质量默认）。
2. **是否 CPU 专家卸载**（仅 MoE）：默认按放置决策器推荐；选 no 强制全部专家进显存并警告。
3. **ctx 长度**：`[Enter] = auto`（等价 `--ctx 0`，装载时按实测显存推导并打印 `[ctx]` 账目）。

`run` 还会问模型（未给路径时列出 `<MODEL_DIR>`/`model_dir` 下的 `.gguf`）与生成 token 数 n。

- **非 TTY（管道 / 服务 / CI）零提示**，行为与旧版一致：`printf "" | ./build/engine.exe serve ...`
  仍是默认 ctx=4096、逐 token 输出不变；
- `--no-interactive` / `--yes` 一键跳过；显式给出的 `--ctx`/`--kv-type`/`--gpu-layers`/`--n`
  对应项不再询问；
- 任意非法输入回退默认值，不会崩溃；选择通过统一配置注册表写回，与命令行完全同源。

```bash
./build/engine.exe serve <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf --port 7030   # 终端里会询问三项
printf "" | ./build/engine.exe serve <model.gguf> --port 7030 --yes        # 脚本：零询问
```

详见 **[FEATURE-MATRIX.md](FEATURE-MATRIX.md) §3** 与 [docs/CONFIG.md §8](../CONFIG.md)。

## 结构

```
engine/
  include/engine/   # 公共头：config.h / kernel_select.h / ir.h / runner.h / kernels.h ...
  src/              # 实现：config.cpp / kernel_select.cpp / runner.cpp / qwen35.cpp / ir.cpp ...
  docs/CONFIG.md    # 配置手册（配置键表 / 文件格式 / 优先级 / 环境变量 / 校准 / 自适应）
  pyref/
    gguf_parse.py   # 从零实现的 GGUF v2/v3 读取器（header/metadata/tensor table）
    ir.py           # ModelIR 数据模型 + role 分类 + GGUF→IR + 校验规则（spec §3.3）
    planner.py      # 预算器：实测带宽常量 → PlacementPlan（spec §4）
    cli.py          # info / ir / plan 三个子命令
  tests/
    test_smoke.py   # 合成 GGUF 写入器 + 12 个单测（无需外部模型）
```

## 用法

```bash
cd engine
PY=python

$PY -m pyref.cli info  <model.gguf>                 # 解析 + 校验报告
$PY -m pyref.cli ir    <model.gguf> --out ir.json   # 导出完整 IR
$PY -m pyref.cli plan  <model.gguf> --ctx 8192      # 预算与放置计划

$PY -m unittest discover -s tests -v                # 单测（12 项，<1 s）
```

## M1/M2 验证结果（2026-11-12，本机真实模型）

| 模型 | 容器 | 解析耗时 | 校验 | 规划结果 | 与 llama.cpp 实测对照 |
|---|---|---:|---|---|---|
| DeepSeek-Coder-V2-Lite Q5_K_M | GGUF | 240 ms | PASS | 24/26 MoE 层驻显存，KV 243 MiB | `-ngl 99` 实测 35.88 tok/s（全部驻留）→ 基本吻合 |
| Qwen3.5-9B Q5_K_M | GGUF | 555 ms | PASS | dense，驻留需求 6.62 GiB | 12 GB 卡可全驻留 ✅ |
| Qwythos-9B-MTP Q4_K_M | GGUF | 560 ms | PASS | **识别出 `blk.32.nextn.*` MTP 头并从 32 层中剔除** | llama.cpp 需 `--spec-type draft-mtp` |
| Qwen3.6-35B-A3B Q4_K_M | GGUF | — | PASS | 18/40 MoE 层驻显存（层粒度）⇒ `n_cpu_moe≈22` | **实测最优 `--n-cpu-moe 24`，偏差 2 层（5%）** |

最后一行是 M2 的关键验收：预算器仅用「固定权重 + KV + workspace + 每层专家字节」的算术，
就把 llama.cpp 扫参得到的最优点预测到 7.5% 以内。

## 已实现的语义细节（这些是踩过的坑）

- **GGUF 专家是融合的**：`blk.N.ffn_gate_exps.weight` 是 `[n_expert, inter, hidden]` 单张量，
  不是每专家一张。校验器按此判断，分片布局（v0 不支持）会显式报错。
- **MLA 的 KV 不是 `n_head_kv × head_dim`**：deepseek2 用 `kv_lora_rank(512)+rope_dim(64)`
  每层每 token 一份压缩 latent；按 GQA 公式会高估 ~16 倍。
- **混合 SSM 模型的 KV 层数**：`full_attention_interval=4` 时只有 8/32 层持有 KV。
- **MTP/nextn 块不是 transformer 层**：必须从层数、KV、专家放置中剔除，否则预算全错。
- **专家放置按层均匀**（与 llama.cpp `--n-cpu-moe` 同构，P0 实测该策略最优）。

## 工具链（已解决，无需管理员）

本机没有 MSVC / CUDA Toolkit，且当前会话无管理员权限（写 `C:\Program Files` 被拒）。
已改为**全 pip 免管理员工具链**，并已实测可用：

| 组件 | 来源 | 用途 |
|---|---|---|
| **zig c++ 0.17.1** | `pip install ziglang` | 完整 C++17 编译器（自带 clang + libc++，`python -m ziglang c++`），M0–M2 主机代码 |
| **NVRTC 12.9** | `pip install nvidia-cuda-nvrtc-cu12` | 运行时编译 CUDA kernel（替代 nvcc，免 Toolkit） |
| **cuBLAS 12.9** | `pip install nvidia-cublas-cu12` | dense GEMM |
| **cudart 12.9** | `pip install nvidia-cuda-runtime-cu12` | 运行时 API |

验证：

```bash
python -m ziglang c++ -std=c++17 hello.cpp -O2 -o hello.exe   # 已跑通
```

代价与约束：

- NVRTC 只能编译**设备代码**，主机侧必须是 C++（这正好符合我们的分层）；
- kernel 以源码字符串形式嵌入，首次启动编译（可缓存 cubin 到磁盘，符合规格书 §A5 的缓存设计）；
- 若日后要 `nvcc` 的 fatbin / `-arch` 全量预编译，仍需要一次管理员安装（可选，不阻塞）。

## 已知未完成项（2026-11-15 收口后更新）

按投入产出比排序：

1. **SSD 流式加载本体**：放置决策器、门控、接口占位与实现方案已就绪
   （`--ssd-stream` 当前确定性报“未实现”），真正的 `pread` 双缓冲 + 专家 LRU 未写。
   见 [FEATURE-MATRIX.md](FEATURE-MATRIX.md) §2。
2. **dense 模型的 host/SSD 权重路径**：引擎只有 MoE 专家可卸载；dense 装不下显存时决策器
   会明确报“dense 不能分片”，不会假装能 split。
3. **`kv_offload`（host-mapped KV）**：仅解析 + 说明，未实现。
4. **MTP 接 serve**：`src/mtp.cpp` 的投机解码已在 `test_gen`/`mtp_run` 生效，HTTP 服务未接；
   不属于本轮范围。
5. **交互式启动的模型选择**：只扫描本地 `model_dir`/`<MODEL_DIR>`，不联网、不记忆历史。
6. **决策器预算与装载实测的偏差**：决策器是“装载前预告”（固定 workspace 预算），真正逐层
   放置仍由 runner 装载时 `cuMemGetInfo` 实测决定；Coder/35B 的预计 GPU 层数可能与实际差 1–3 层。

详细的量化证据、尝试过的优化与负结果见 [SPEC-COMPLIANCE.md](SPEC-COMPLIANCE.md)。
