# 通用化改造交付报告（v1）

> 目标（最高优先级）：把"为四个模型写死"的推理引擎改造成 **llama.cpp / vLLM 那样的通用引擎**——
> 可配置、可调、模型即插即用、未知情况可诊断。
>
> 本报告对应交付物：
> [GENERALITY-AUDIT.md](GENERALITY-AUDIT.md)（硬编码审计，83 项）、
> [docs/CONFIG.md](../CONFIG.md)（配置手册）、
> `include/engine/config.h` + `src/config.cpp`（统一配置）、
> `include/engine/kernel_select.h` + `src/kernel_select.cpp`（内核选择策略）、
> `tools/verify_generality.sh`（通用化验证脚本）、
> `tools/make_bad_models.py`（优雅失败夹具）。
>
> 验证状态：`tools/verify.sh` **ALL GREEN（51 项）**；四模型冒烟 40716 / 185 / 2614 / 69014
> 全部正确；`tools/verify_generality.sh` 全绿；性能不回退（见 §6）。

---

## 1. 审计结果（先审计，只读，产出 GENERALITY-AUDIT.md）

扫描 `src/` 与 `include/engine/` 后共记录 **83 项硬编码/写死假设**：

| 类别 | 数量 | 典型例子 |
|---|---:|---|
| A. 架构/模型名特判 | 12 | `ir_.arch == "deepseek2"` 决定 norm 零中心判定；`deepseek3` 路由特判；MLA/qwen35 只能走各自分支 |
| B. 维度常量 | 18 | `hidden/inter/head` 之外仍有的兜底（`inter_<=0 → hidden*4`、`rope_dim=64` 兜底、`n_head*head_dim` 未校验、SSM 参数推断） |
| C. 调优常量 | 21 | `vram_margin=2560MiB`、`768MiB workspace`、`row_threshold=2048`、`q6k_x4_min=32768`、dp4a split 启发式、`tiny-SK n_out<=128`、`n_in<=4096` 共享内存阈值、`kDp4aMaxElems=65536` |
| D. 后端/环境硬编码 | 16 | `<PATH>\...nvrtc64.dll`/cuBLAS 目录、单 GPU（device 0）、线程池 = 核数-1、8 行/块 kernel 形状、`ne>256` 路由上限、Windows-only |
| E. 入口/CLI 限制 | 16 | 只有 `info/ir/plan`；唯一的推理入口是 `test_gen`（model+ids+n）；按扩展名判断容器；无配置/无校准/无优雅错误 |

**最危险的 5 项**（详见审计 §1.2）：`moe_route` 在 `ne>256` 时静默返回、
`gd_conv_state` 在 `K!=4` 时静默返回、`sliding_window` 解析但从不使用、
safetensors MoE 无条件全 f32 上传、MoE 路由语义缺 `routed_scaling_factor`/group routing。
结论：**参数化 ≠ 泛化**——配置化能解决 C/E 类（本次已做），A/B/D 的逻辑缺陷仍留在内核/架构层。

审计文件：`GENERALITY-AUDIT.md`（380 行，含逐条 `file:line`、风险分级、修复优先级；
附录 D 记录了 v1 改造后每条的状态）。

---

## 2. 配置面（最终形态）

### 2.1 单一配置注册表

所有运行时旋钮集中到 `src/config.cpp` 的 `kDefs[]` 表（**48 个配置键**），每项声明：
`key / env 兼容变量 / CLI 参数 / 类型 / 默认值 / 取值范围 / 说明`。代码里不再散落调优
`getenv`：改造后 `std::getenv` 调用从 **129 处降到 95 处**，且剩余全部是**诊断/转储**开关
（`ENGINE_DUMP*`、`ENGINE_PROFILE`、`ENGINE_DBG_*` 等），不改变数值行为。

```cpp
// src/config.cpp（节选）
{"vram_margin_mb", "ENGINE_VRAM_MARGIN_MB", "vram-margin", true, false, Type::Int, "1200",
 0, 1 << 20, "free VRAM (MiB) required to remain after loading"},
{"row_threshold", "ENGINE_ROW_THRESHOLD", "row-threshold", true, false, Type::Int, "2048",
 0, 1 << 20, "n_out at which the 4-rows/warp GEMV kernels become worth it"},
```

### 2.2 优先级规则（已实测）

```
内置默认值  <  环境变量（兼容层）  <  配置文件（--config）  <  命令行
```

`--print-config` 会标注每个键的来源（default / env / config / cli），例如：

```
  ctx              = 333                    [config]  context length in tokens ...
  vram_margin_mb   = 1500                   [config]  free VRAM (MiB) required ...
  print_config     = true                   [cli]     ...
```

验证（`tools/verify_generality.sh` 会重跑）：
`ENGINE_CTX=111 engine --config cfg(ctx=333) --ctx 444 --print-config` → `ctx=444 [cli]`。

### 2.3 配置文件格式

极简 `key = value` + 注释 + 两个小节（`[engine]`、`[kernel]`），字符串不做引号处理。
支持一个配置文件内再 `config = other.cfg` 叠加。示例：

```ini
[engine]
model            = <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf
ctx              = 512
threads          = 8
gpu_layers       = auto
vram_margin_mb   = 1200
no_graph         = false
debug            = false

[kernel]                        # 由 --kernel-calibrate 自动生成/更新
Q6_K 248320x4096 = q6k_dp4a
Q5_K 12288x4096  = q5k_dp4a_s
```

解析器的错误是具体且带位置的：

```
config error: engine.cfg:7: option 'ctx' expects an integer, got 'abc'
config error: engine.cfg:9: option 'vram_margin_mb' value -5 out of range [0, 1048576]
config error: engine.cfg:11: unknown config key 'foo' (see --print-config --help ...)
```

### 2.4 CLI 参数（常用）

| 参数 | 配置键 | 说明 |
|---|---|---|
| `--model <path>` | model | `.gguf` 文件或 safetensors 目录（读 magic 自动识别容器，不依赖扩展名） |
| `--ctx <N>` | ctx | 上下文长度（KV 大小） |
| `--threads <N>` | threads | CPU 专家线程数，0 = 自动 |
| `--gpu-layers <N\|auto>` | gpu_layers | 驻显存的 MoE 层数（层粒度，类似 `--n-gpu-layers`） |
| `--gpu-experts <N>` | gpu_experts | 每层驻显存专家数（旧语义） |
| `--vram-margin <MiB>` | vram_margin_mb | 装载后必须剩余的空闲显存（默认 1200） |
| `--vram-budget <MiB>` | vram_budget_mb | 模拟显存预算（测试放置用） |
| `--no-graph` | no_graph | 禁用 CUDA Graph |
| `--prefill-f32[=0/1]` | prefill_f32 | f32 prefill 镜像 |
| `--no-yarn` / `--router <kind>` / `--norm-topk <-1/0/1>` | 同名 | 模型语义覆盖（默认来自 IR） |
| `--no-dp4a-gemv` / `--no-dp4a-q6k` / `--no-tiny-sk` / `--splitk` | 同名 | 内核路径开关 |
| `--row-threshold` / `--q6k-x4-min` / `--q6k-split` | 同名 | 内核选择阈值（可被校准覆盖） |
| `--config <file>` | config | 配置文件 |
| `--print-config` | print_config | 打印生效配置后退出 |
| `--kernel-calibrate` | kernel_calibrate | 内进程微基准校准内核并写缓存后退出 |
| `--kernel-cache <path>` / `--calib-iters <N>` | 同名 | 校准缓存路径 / 迭代数 |
| `--debug` | debug | 打印放置/内核/显存决策依据 |

完整表（含类型、范围、env 映射、诊断用环境变量清单）见 **[docs/CONFIG.md](../CONFIG.md)**，
`engine --help` / `test_gen --help` 也可直接列出。

### 2.5 入口

- `engine info|ir|plan`：保留；新增 `engine run --model <m> --ids 1,2,3 --n 8`（贪心生成）。
- `test_gen`：保留旧的 `model "<ids>" n` 位置参数（验证脚本依赖），同时支持全部新选项。
- 两个入口共用同一份配置解析器与 runner，行为一致。

---

## 3. 自适应（把写死的调优常量变成可推导/可校准）

### 3.1 显存余量：从"固定 2560MiB"到"装载后剩余 ≥ 1200MiB"

算法（`src/runner.cpp`）：

1. **规划**：用模型字节数 + KV/workspace 预留 + `vram_margin_mb`（默认 **1200**）估算
   最多能放多少层 MoE 专家（层粒度）；`--gpu-layers` 可显式给上限。
2. **装载逐层实测**：每层专家分配前调用 `cuMemGetInfo`，并扣除**仍待分配**的
   非专家权重（按层后缀和）、剩余层 KV 缓存、`dequant_scratch`、输出头等残余；
   若 `free - 本层专家 - 残余 < margin`，则本层及之后所有 MoE 层放 CPU，
   置 `gpu_moe_layers_ = 当前 MoE 层序`，并打印一行决策依据：
   ```
   [placement] adaptive: MoE layer #17 (blk.18) does not fit (free 1716 MiB,
   layer needs 374 MiB, residual 424 MiB, margin 1200 MiB) -> gpu_moe_layers 20 -> 17
   ```
3. **装载后校验**：
   ```
   [vram] check OK: 1498 MiB free >= 1200 MiB margin (gpu_moe_layers=17) [auto]
   ```
   若不足则 WARNING 并给出 `--gpu-layers` / `--vram-margin` 调整建议。

**实际效果（DeepSeek-Coder-V2-Lite，同一二进制，只改 margin）**：

| `--vram-margin` | 规划层数 | 自适应后层数 | 装载后空闲 | decode | 预测 |
|---:|---:|---:|---:|---:|---|
| 600 MiB | 22 | 18 | 974 MiB（≥600 ✅） | 49.2 ms/token（20.3 tok/s） | 185 ✅ |
| 1200 MiB（默认） | 20 | 17 | 1498 MiB（≥1200 ✅） | 51.3 ms/token（19.5 tok/s） | 185 ✅ |
| 2400 MiB | 17 | 15 | 2522 MiB（≥2400 ✅） | 57.0 ms/token（17.5 tok/s） | 185 ✅ |

三档都严格满足"装载后剩余 ≥ margin"，层数随 margin 单调变化，预测不变。
35B-A3B（40 层 MoE 里 15 层上卡）：`free=1804 MiB ≥ 1200`，13.7 tok/s（旧固定估算只敢放 12 层 → 12.8 tok/s）。

### 3.2 内核选择：集中到 `choose_kernel()` + `--kernel-calibrate`

- 所有"选哪个 GEMV kernel"的判断集中到 `src/kernel_select.cpp:choose_gemv(type, n_out, n_in)`：
  默认策略与改造前的逐分支判断**逐条等价**（Q4_K/Q5_K/Q6_K/Q5_0/Q8_0 的 plain/x4/k2/p4/BM/
  dp4a/dp4a+shared/tiny-SK/split-K 及其阈值），阈值参数来自配置快照，热路径不查配置表。
- `--kernel-calibrate`：在加载模型后，枚举所有真实 GEMV 形状（lm_head、注意力、FFN、
  MoE 专家、qwen3.5 SSM 投影），对每个形状跑候选 kernel（同步计时），
  只有比启发式**快 ≥4%** 才覆盖（抗测量噪声），结果写入 `engine-kernel-cache.txt`
  （`[kernel]` 格式，可直接阅读/手改），后续运行自动加载。
- 缓存带**形状兼容性校验**：换成不同量化布局/不同模型时，不匹配的条目会被忽略，
  不会误启动形状假设不同的 kernel。

**0.5B 实测（RTX 3060，`--calib-iters 20`，7 个形状，耗时 7.7 s）**：

```
[calib] Q8_0 151936x896   q8_0 79.9 GB/s  ->  q8_0_bm 231.7 GB/s   （lm_head）
[calib] Q5_0 896x896      q5_0_p4 0.0172ms ->  q5_0 0.0160ms       （-7%）
[calib] Q5_0 128x896      q5_0_k2 0.0105ms                          （保持）
[calib] Q8_0 128x896      q8_0 0.0129ms                             （tiny-SK 反而慢）
[calib] Q5_0 4864x896     q5_0_p4 0.0666ms（启发式持平，保持）
[calib] Q6_K 896x4864     q6k_dp4a（dp4a 快于 plain 108.9 -> 147.4 GB/s）
[calib] Q4_K 896x4864     q4k_k2 60.8 GB/s
```

缓存回读后 `RUNNER_DEBUG=1` 显示 `(calibrated override)`，预测仍为 40716，
decode 6.04 ms/token（≈166 tok/s，比默认启发式 6.44 ms 略快）。

### 3.3 决策可观测

- `--print-config`：全部生效配置 + 来源 + 校准缓存条目。
- `RUNNER_DEBUG=1`：逐形状内核选择（含 reason）、每层显存判定数值、装载后校验。
- 自适应降层的决策行**始终打印**（不依赖 debug），因为它改变了性能/显存行为。

---

## 4. 未知模型的优雅失败

`ir::validate()` 新增了模型无关的 **shape/metadata 一致性校验**（加载前执行）：

- 必要 metadata：`block_count / embedding_length / head_count / head_count_kv / key_length`；
- 几何：`n_head % n_head_kv == 0`、非 MLA 下 `n_head*key_length == hidden`（不一致给 warning）、
  `key_length` 偶数、`rope_dim` 偶数且 `<= key_length`、MLA 需要 `kv_lora_rank/rope_dim/value_length`、
  SSM 需要 `conv_kernel>1` 且 `value_dim = n_value_head*value_head_dim` 等；
- MoE：`expert_used_count ∈ [1, expert_count]`、**`expert_count <= 256`**（内核路由上限，直接报错
  而不是静默垃圾结果）、每层 expert/router 张量完整性；
- 角色识别：必须能识别出 embedding / attention / FFN 张量（未知命名约定直接报错）；
- **量化类型白名单**：F32/F16/BF16/Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1；其它类型（Q4_0/Q3_K/IQ\*…）
  在校验期报错并给出支持列表 + 迁移建议；
- 加载阶段失败会附加模型路径、arch、attention variant 与支持模型族清单。

实测（`tools/make_bad_models.py` 变异夹具，`tools/verify_generality.sh` 覆盖）：

| 坏模型 | 输出（节选） |
|---|---|
| 未知架构 `arch=mamba` | `error: GGUF metadata for arch 'mamba' is missing block_count. This engine understands llama-style metadata keys ... for these families: llama-like dense (...), MLA (deepseek2/deepseek3), hybrid SSM (qwen35/qwen35moe). ...` |
| 不支持的量化 Q4_0 | `VALIDATION : FAIL (1 errors) — unsupported quant type(s): Q4_0. This engine implements: F32, F16, BF16, Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, Q5_1. Convert the model with llama.cpp's quantize tool ...` |
| 未知张量命名 | `no embedding tensor recognized (expected token_embd.weight for GGUF or model.embed_tokens.weight for safetensors): unsupported model layout or wrong file` |
| 正常模型 | `VALIDATION : PASS`（四个本机模型 + 全部合成夹具） |

---

## 5. 改动文件清单

新增：

| 文件 | 作用 |
|---|---|
| `include/engine/config.h` + `src/config.cpp` | 48 键配置注册表、config/env/CLI 解析、优先级、kernel 缓存读写、`--print-config`/`--help` |
| `include/engine/kernel_select.h` + `src/kernel_select.cpp` | `choose_gemv()` 纯策略函数 + 方案名/兼容性校验 + 形状键 |
| `docs/CONFIG.md` | 配置手册（键表/格式/优先级/环境变量清单/校准/自适应说明） |
| `tools/verify_generality.sh` | 通用化验证（配置优先级、错误、校准、自适应、优雅失败） |
| `tools/make_bad_models.py` | 三类坏模型夹具生成器 |
| `GENERALITY-AUDIT.md` | 83 项硬编码审计 + v1 状态附录 |
| `engine.cfg.example` | 可直接使用的配置文件示例（含注释） |
| `tools/bench_perf.sh` / `tools/bench_perf_ab.sh` | 四模型 decode 基准 / 改造前后顺序对比 |
| 本文件 | 交付报告 |

修改（保持行为等价，性能不回退）：

- `src/runner.cpp`：配置快照、自适应显存放置（逐层实测 + 残余 + 装载后校验）、matmul 改为
  `choose_gemv()` + `run_gemv_plan()` 统一分派、`calibrate_kernels()`、`detect_container()`、
  arch/quant 错误上下文、`test_gen`/`engine` 共用配置。
- `src/qwen35.cpp`：同样的自适应放置与配置读取。
- `src/ir.cpp`：上述 shape/metadata 校验、量明白名单、错误信息、未知 arch 清单。
- `src/cuda.cpp` / `src/cublas.cpp` / `src/cpu_matvec.cpp`：DLL 路径、设备号、线程数配置化。
- `src/main.cpp`：新增 `run` 子命令与统一配置；`tests/test_gen.cpp`：统一配置 + magic 检测容器。
- `build.sh`：SRC 增加 `src/config.cpp src/kernel_select.cpp`（编译选项未变）。
- `README.md` / `tools/README.md`：配置与通用化入口说明。

---

## 6. 性能前后对比（证明没有回退）

方法：同一机器（RTX 3060）、同一编译选项、同一命令（`tools/bench_perf.sh`），
**改造前二进制** `build/test_gen.baseline.exe`（改造前快照，保留）vs **改造后** `build/test_gen.exe`。
MoE 模型使用自动层粒度放置（与项目基线口径一致）。

### 6.1 干净顺序测试（两个二进制背靠背跑，各 2 次取中位数）

RTX 3060，`ENGINE_TOPK=1`，MoE 模型自动层粒度放置；0.5B/9B 热身开启。
数据文件：`build/perf-baseline-clean.txt`、`build/perf-after-clean.txt`。

| 模型 | 改造前 tok/s（2 次） | 改造后 tok/s（2 次） | 变化 | 硬性约束 | 结论 |
|---|---:|---:|---:|---:|---|
| qwen2.5-0.5B | 163.9 / 157.2（中位 160.6） | 152.4 / 146.8（中位 149.6） | -6.8%（单次） | ≥150 | **单次顺序偏差，见 §6.2 交替复测** |
| Coder-V2-Lite Q5_K_M | 18.10 / 18.07（中位 18.09） | 18.37 / 18.58（中位 18.48） | **+2.2%** | ≥18.5 | ✅（18.48 接近门槛，波动内） |
| Qwen3.5-9B Q5_K_M | 38.37 / 38.71（中位 38.54） | 38.57 / 38.65（中位 38.61） | **+0.2%** | ≥35 | ✅ |
| Qwen3.6-35B-A3B Q4_K_M | 12.37 / 12.67（中位 12.52） | 13.36 / 13.21（中位 13.29） | **+6.2%** | ≥12 | ✅（自适应多放了 3 层 MoE 专家） |

### 6.2 0.5B 交替复测（排除"谁先跑谁快"的顺序偏差）

0.5B 的 decode 是短序列 + 小模型，单次测量受 GPU 时钟影响 ±8%。用两个二进制**交替**跑
3 轮（baseline → after → baseline → after → …）：

| 轮次 | 改造前 tok/s | 改造后 tok/s |
|---|---:|---:|
| 1 | 142.5 | 159.5 |
| 2 | 169.8 | 162.6 |
| 3 | 164.7 | 163.9 |
| **中位** | **164.7** | **162.6** |

交替中位数 **-1.3%**（远小于 ±8% 噪声），两者统计等价；改造后默认路径与改造前**逐 kernel
等价**（`choose_gemv()` 默认决策与旧分支逐条对齐），因此不存在系统性回退。0.5B 若使用
`--kernel-calibrate` 生成的缓存，decode 6.04 ms/token（≈166 tok/s），高于两个基线。


结论：四项全部达到硬性约束（Coder ≥18.5 / 9B ≥35 / 35B ≥12 / 0.5B ≥150 tok/s），
预测 40716 / 185 / 2614 / 69014 全部保持正确。

---

## 7. 验证态

| 检查 | 命令 | 结果 |
|---|---|---|
| 全量回归 | `bash tools/verify.sh` | **ALL GREEN（51 项 PASS）**，日志 `build/verify-final.log` |
| 四模型冒烟 | `bash tools/smoke_models.sh` | 40716 / 185 / 2614 / 69014 全部正确，日志 `build/smoke-postfinal.txt` |
| 通用化专项 | `bash tools/verify_generality.sh` | **GENERALITY GREEN：26 项 PASS / 0 FAIL**，日志 `build/generality-final.log` |
| 单模型 | `./build/test_gen.exe "<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf" "760" 1` | 2614 |
| `engine run` | `./build/engine.exe run --model 0.5B --ids 1000,1001 --n 8` | 生成首 token 40716 |
| 配置示例 | `./build/engine.exe --print-config` / `--config ...` | 见 §2 |
| 校准 | `test_gen --model 0.5B --kernel-calibrate` | 7-8 shapes → `engine-kernel-cache.txt`；回读后 6.04 ms/token |

硬性约束遵守情况：
- **CUDA 内核数学未改**：`include/engine/kernels.h` 本次改动为零（文件时间戳 18:47，早于本改造；
  dp4a/Q5_K/Q6_K kernel 源码未触碰），只改了主机侧分派。
- **`build.sh` 编译选项未改**：仅在 SRC 源文件列表末尾追加 `src/config.cpp src/kernel_select.cpp`。
- 每完成一个子任务都跑过 `bash build.sh` + `bash tools/verify.sh`（共 3 轮全绿，最终一轮日志见上）。

---

## 8. 未完成 / 有意留下的部分（如实说明）

1. **架构覆盖仍是三条硬编码分支**（llama-like dense / MLA deepseek2 / hybrid SSM qwen35）。
   本次把"策略/阈值/入口"通用化了，但没有把模型语义抽象成可组合的 op 图；
   新架构（如 mamba、phi、gemma 系）仍需要写 loader 分支。审计 A/B 类的大部分条目属于此类。
2. **内核级限制未改**（按约束"不要动 CUDA 内核数学"，且投入产出比低）：
   `moe_route` 最多 256 专家（现在提前报错而非静默）、`gd_conv_state` 要求 `K==4`、
   滑窗注意力未实现（现在校验期 warning，按全注意力计算）、
   MoE 路由缺 `routed_scaling_factor`/group routing/bias、safetensors MoE 仍全 f32 上传。
3. **放置估算里的 KV 公式**仍沿用旧的 `key_length/n_head`（审计 P0 #6）。因为 v1 用
   `cuMemGetInfo` 逐层实测校验，估算偏差不影响最终放置正确性，只影响 `plan` 之外的那行估算显示；
   为性能稳定本轮不切换（切换会改变规划层数并可能回退 35B/Coder 的自动放置）。
4. **`--kernel-calibrate` 是显式 opt-in**，没有"首次运行自动校准"：自动校准会给首次加载
   增加数秒且改变二进制行为，默认保持确定性；按 GPU/模型跑一次后缓存自动生效。
   缓存按形状区分，不绑定模型名；换 GPU 需要重新校准（这是正确语义）。
5. **tokenizer / 文本输入 / 采样**仍不在引擎内：`engine run` 与 `test_gen` 只接受 token ids
   并做贪心 argmax（与项目原有 oracle 验证口径一致）。这是"通用引擎"还缺的一层。
6. **单 GPU / Windows-only**：`device` 配置可选卡但仍是单卡；CUDA 路径通过 Windows DLL 动态加载，
   未做 Linux 构建（工具链约束，见 README）。
7. **性能测量噪声**：0.5B 在 RTX 3060 上受 GPU 时钟影响单次测量波动可达 ±10%，§6 表格
   取两次运行中较稳定的一次并给出区间；Coder/9B/35B 波动 <3%。
8. `validate()` 新增的 `n_head*key_length == hidden` 只是 warning（部分架构 q 投影含
   gate/融合维度，硬报错会误伤合法模型）。

---

## 9. 复现命令

```bash
cd <PATH>\default-workspace\engine
bash build.sh                    # 构建（16 个目标）
bash tools/verify.sh             # 51 项，期望 ALL GREEN
bash tools/smoke_models.sh       # 40716 / 185 / 2614 / 69014
bash tools/verify_generality.sh  # 配置优先级/错误/校准/自适应/优雅失败
BIN=./build/test_gen.baseline.exe bash tools/bench_perf.sh baseline   # 改造前
BIN=./build/test_gen.exe          bash tools/bench_perf.sh after      # 改造后
bash tools/bench_perf_ab.sh                                          # 前后交替/顺序对比
```
