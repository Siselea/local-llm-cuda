# 配置手册（engine v1 通用化配置面）

本引擎的运行时参数全部收拢到**统一配置系统**：一份配置注册表（`src/config.cpp`）、
一个解析器、一套优先级规则。命令行、配置文件、兼容环境变量、内置默认值只有一个来源。

```
优先级（低 -> 高）：  内置默认值  <  环境变量  <  配置文件  <  命令行
```

- 每个选项都有：**类型**（int / float / bool / string）、**默认值**、**取值范围**；
  非法值在启动时报错并指出 `文件:行号` 或具体选项名，不会静默使用错误值。
- `--print-config` 打印全部生效值 + 每项来自哪一层（default / env / config / cli）。
- `--help` 打印全部命令行参数。
- 环境变量只作为**兼容层**保留，旧脚本无需修改；新代码请用配置文件或 CLI。

---

## 1. 快速开始

```bash
# 1) 查看当前生效配置（不加载模型）
./build/engine.exe --print-config

# 2) 用配置文件跑生成
./build/engine.exe run --config engine.cfg --ids 1000,1001 --n 32
./build/test_gen.exe --model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ids 1000,1001 --n 32

# 3) 临时覆盖 + 内核自动校准（首次部署新 GPU 时跑一次）
./build/test_gen.exe --model <model.gguf> --ids 760 --n 8 --vram-margin 1500 --ctx 512
./build/test_gen.exe --model <model.gguf> --ids 760 --n 8 --kernel-calibrate

# 4) 起 HTTP 服务（OpenAI 兼容 + llama.cpp 风格，浏览器聊天界面在 /）
./build/engine.exe serve <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf --port 7030 --ctx 4096
# -> listening on http://127.0.0.1:7030 (model=..., ctx=4096)

# 5) 交互式启动（终端里会询问 KV 精度 / 是否 CPU 专家卸载 / ctx）
./build/engine.exe serve <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf --port 7030
# 非交互（管道/脚本/CI 永不询问；--no-interactive / --yes 也可显式跳过）
printf "" | ./build/engine.exe serve <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf --port 7030 --yes

# 6) 放置决策账目（不加载权重，可离线/模拟显存预算）
./build/engine.exe plan <MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf --ctx 4096
# -> [placement] decision: vram+ram split ...
```

`engine.exe` 子命令：`info`（元数据/校验报告）、`ir`（导出 IR JSON）、`plan`（显存/专家放置规划）、
`run`（贪心生成）、`serve`（HTTP 服务，见 §7）。`test_gen.exe` 是同一套 runner 的精简生成入口
（输出格式被验证脚本使用）。

---

## 2. 配置文件格式

极简 `key = value`，行内 `#` 到行尾是注释，`[engine]`（可省略）和 `[kernel]` 两个小节。
键名即下表第一列；值不做引号处理（字符串原样取 `=` 后面的内容）。

```ini
# engine.cfg
[engine]
model            = <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf
ctx              = 512          # KV cache 长度
threads          = 8            # CPU 专家线程数，0 = 自动（约 3/4 逻辑核，留余量给 OS/驱动）
gpu_layers       = auto         # MoE 层保留在显存的数量，auto/-1 = 自动按空闲显存放置
vram_margin_mb   = 1200         # 装载完成后必须剩下的空闲显存（不足自动减 GPU 层）
workspace_mb     = 768
no_graph         = false
debug            = false        # 打印放置/内核选择决策

[kernel]
# 由 --kernel-calibrate 自动生成，手工可读可改：
# <quant type> <n_out>x<n_in> = <plan>
Q6_K 248320x4096 = q6k_dp4a
Q5_K 12288x4096  = q5k_dp4a_s
```

配置文件中的 `[kernel]` 小节与独立的校准缓存文件（默认 `engine-kernel-cache.txt`，
由 `kernel_cache` 选项指定）格式相同；独立缓存文件由 `--kernel-calibrate` 自动重写。

加载规则：

- `--config <file>` 在其余命令行选项之前生效（因此 **配置文件 > 环境变量** 成立）；
- 配置文件里再写 `config = another.cfg` 会叠加加载（后加载者覆盖前者）；
- 未知键、缺 `=`、非法值、未知小节都会报错，例如：
  `engine.cfg:7: option 'ctx' expects an integer, got 'abc'`。

---

## 3. 配置项总表

> CLI 列写出参数名（`<v>` 表示带值）；env 列是兼容环境变量；
> **覆盖顺序**：CLI > config > env > default。

### 3.1 模型与运行形状

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `model` | `--model <v>` | `ENGINE_MODEL` | — | string | `""` | 模型路径：`.gguf` 文件或 safetensors 目录（自动识别，不依赖扩展名） |
| `ctx` | `--ctx <v>` | `ENGINE_CTX` | int [1, 1M] | `256` | 上下文长度（决定 KV cache 大小）；不设置时生成入口按 prompt+n 自动估算 |
| `threads` | `--threads <v>` | `ENGINE_THREADS` | int [0,4096] | `0` | CPU 专家 GEMV 线程数，0 = 自动：能读到 SMT=2 拓扑时 = **2×物理核**（本机 6C/12T → 12，且默认 `win_affinity=2` 成对绑核），否则回退历史公式 `hardware_concurrency - max(2, hw/4)`（本机 9）。实测 12/11/10 成对比 9 不绑快 4–12%，见 `reports/THREAD-AFFINITY-REPORT.md` |
| `gpu_layers` | `--gpu-layers <v>` | `ENGINE_GPU_LAYERS` | int [-1,4096] | `-1` | 放显存的 MoE 层数（层粒度，类似 llama.cpp `--n-gpu-layers`）；`auto`/`-1` = 自动推算；实际装载时还会按实测空闲显存自适应下调 |
| `gpu_experts` | `--gpu-experts <v>` | `ENGINE_GPU_EXPERTS` | int [-1,1e5] | `-1` | 每层放显存的专家数（旧语义，固定切分；-1 = 自动/层粒度） |
| `layers` | `--layers <v>` | `ENGINE_LAYERS` | int [0,1M] | `0` | 只跑前 N 层（性能分解），0 = 全部 |

### 3.2 显存与放置（自适应）

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `vram_margin_mb` | `--vram-margin <v>` | `ENGINE_VRAM_MARGIN_MB` | int [0,1M] | `1200` | **装载后必须剩余的空闲显存**。估算阶段用于规划，装载阶段逐层用 `cuMemGetInfo` 实测校验，不足则自动减少 GPU 层并打印决策 |
| `vram_budget_mb` | `--vram-budget <v>` | `ENGINE_VRAM_BUDGET_MB` | int [0,1M] | `0` | 模拟显存预算（测试放置策略用），0 = 读真实空闲显存 |
| `workspace_mb` | `--workspace <v>` | `ENGINE_WORKSPACE_MB` | int [0,1M] | `768` | 放置估算时给 KV/激活/scratch 预留的显存 |
| `vram_check` | `--no-vram-check` | `ENGINE_NO_VRAM_CHECK` (present = off) | — | bool | `true` | 关闭装载期/装载后的自适应显存校验（调试用） |
| `vram_extend` | `--no-vram-extend` | `ENGINE_NO_VRAM_EXTEND` | bool | `true` | 让**实测**的逐层显存校验在算术估算之外**追加** GPU 驻留 MoE 层（只尝试紧邻的下一层，保持连续前缀；装载后的 margin 校验仍然生效）。35B 上因此多放 1 层（15→16），~1 ms/token |
| `vram_auto_layers` | `--vram-auto-layers` | `ENGINE_VRAM_AUTO_LAYERS` | bool | `false` | **可选（默认关）auto-margin 决策器**：装载完成后用**实测**空闲显存把还没上 GPU 的 CPU 专家层“抬升”到显存（最多 `vram_auto_layers_max` 层），并保证抬升后仍留下 `vram_margin_floor_mb`。本机 35B 实测 +1 层、decode 27.43→28.1 tok/s（−0.67 ms/token、bit-exact），但 decode 期间空闲显存只剩 ~80 MiB，贴着驱动分页线（free=0 → 3 tok/s、极端情况严重变慢），所以**默认关**；建议只在确认没有其他程序占用显存时开启（serve ctx=4096 下实测 35B 会在 CUDA graph 捕获时 OOM，serve 场景不建议开）。显式 `--vram-margin` / `--gpu-layers` / `--vram-budget` 时不生效。launcher 主面板「性能开关」里有对应开关（默认关，tooltip/红字写明风险） |
| `vram_margin_floor_mb` | `--vram-margin-floor <v>` | `ENGINE_VRAM_MARGIN_FLOOR_MB` | int [0,1M] | `900` | auto-margin 抬升后**必须留下**的空闲显存下限（MiB）。900 = 本机实测安全线（free=0 会让驱动分页 → 3 tok/s） |
| `vram_auto_layers_max` | `--vram-auto-layers-max <v>` | `ENGINE_VRAM_AUTO_LAYERS_MAX` | int [0,64] | `2` | auto-margin 一次装载最多抬升几层（每层约 500 MiB，实测 35B 只够 1 层） |
| `device` | `--device <v>` | `ENGINE_DEVICE` | int [0,64] | `0` | CUDA 设备序号（多卡时选择） |
| `host_margin_mb` | `--host-margin <v>` | `ENGINE_HOST_MARGIN_MB` | int [0,1M] | `0` | 放置决策器要求装载后保留的空闲**主机内存**（MiB）。0 = 自动：`max(1 GiB, 总内存/16)`（本机 32 GiB → 2 GiB） |
| `placement_policy` | `--placement <v>` | `ENGINE_PLACEMENT` | string | `auto` | 权重放置策略：`auto`（实测 VRAM/内存后三档决策，默认）/ `gpu`（要求全驻显存）/ `split`（要求 CPU 专家卸载）/ `ssd`（SSD 流式，**本版本未实现**，显式指定会报错） |
| `ssd_stream` | `--ssd-stream` | `ENGINE_SSD_STREAM` | bool | `false` | 仅当模型同时超过 VRAM+RAM 时才考虑 SSD 专家流式；**默认关闭**（PCIe×8/机械盘上流式通常净负收益），**本版本未实现**：显式开启会得到确定性错误 + 接口占位说明 |
| `ssd_cache` | `--ssd-cache <dir>` | `ENGINE_SSD_CACHE` | string | `""` | SSD 专家缓存目录（接口占位；决策日志会报告该路径，流式加载本体未实现） |
| `allow_oversubscribe` | `--allow-oversubscribe` | `ENGINE_ALLOW_OVERSUBSCRIBE` | bool | `false` | 决策器判定“VRAM+RAM 装不下”时仍尝试装载（驱动会分页，预期单数字 tok/s）；默认关闭 |

### 3.2.1 启动交互（本轮新增）

只影响**交互式启动**（是否询问 ctx / KV 精度 / CPU 专家卸载），不改变运行时数值行为；
非 TTY（管道/服务/CI）完全不询问。完整说明见 `FEATURE-MATRIX.md` §3 与本文 §8。

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `interactive` | `--no-interactive` | `ENGINE_INTERACTIVE` | bool | `true` | stdin 是终端时，serve/run 启动前询问 KV 精度 / 是否 CPU 专家卸载 / ctx（run 另问模型与 n）。`--no-interactive` 或 `ENGINE_INTERACTIVE=0` 关闭 |
| `assume_yes` | `--yes` | `ENGINE_ASSUME_YES` | bool | `false` | 跳过全部询问，直接使用计算出的默认值（脚本/CI 友好） |
| `model_dir` | `--model-dir <v>` | `ENGINE_MODEL_DIR` | string | `""` | 追加的模型扫描目录（`;` 分隔多个），排在内置目录之前；空 = 内置列表（exe 同目录 `models/`、`../models/`、`<MODEL_DIR>`、`%USERPROFILE%/Downloads`、`%USERPROFILE%/.cache`） |

### 3.3 执行 / CUDA Graph

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `no_graph` | `--no-graph` | `ENGINE_NO_GRAPH` | bool | `false` | 禁用 CUDA Graph capture |
| `layer_graph` | `--layer-graph` | `ENGINE_LAYER_GRAPH` | bool | `false` | 实验性逐层 Graph（默认关闭） |
| `layer_graph_auto` | `--layer-graph-auto` | `ENGINE_LAYER_GRAPH_AUTO` | bool | `true` | qwen35 MoE 层的 attention/mixer 段逐层 CUDA Graph：每层 ~15 个 kernel 压成 1 次 `cuGraphLaunch`（WDDM 上每次提交 10–20 µs，实测 GPU 层 0.275→0.199 ms）。懒捕获：首次 decode 形状调用正常执行**并在同一次调用内立即录制**（即录制发生在 prefill 里、不进 decode 计时窗口），之后重放 |
| `cpu_layer_graph` | `--cpu-layer-graph` | `ENGINE_NO_CPU_LAYER_GRAPH`（反转） | bool | `true` | CPU 驻留 MoE 层也录制 attention/mixer 段的逐层 Graph：每层少 ~7 次 WDDM 提交，35B 上 ~1.6 ms/token。早期实测"净变慢"是因为一次性录制落在 decode 计时窗口内；把录制挪到首次执行之后（prefill 内）后净收益为正 |
| `dev_route` | `--dev-route` | `ENGINE_NO_DEV_ROUTE`（反转）/ `ENGINE_DEV_ROUTE`（强制开） | bool | `true` | GPU 驻留层用设备端路由：省掉每层 router D2H + host 同步 + 路由表 H2D；专家的量化布局里含 `Q5_1` 时自动关闭（该分支数值未验证） |
| `mla_no_graph` | `--mla-no-graph` | `ENGINE_MLA_NO_GRAPH` | bool | `false` | MLA 模型禁用 Graph（保留历史规则） |
| `mla_attn_graph` | `--no-mla-attn-graph` | `ENGINE_NO_MLA_ATTN_GRAPH` | bool | `true` | deepseek2 MLA 的 decode attention 段（ln1 norm + MLA attention + ln2 norm）按层录一张 CUDA Graph：实测每层 18 次 WDDM 提交 → 1 次 cuGraphLaunch（本机 1 次图启动 ≈10 µs，18 次裸提交 ≈180 µs）。仅当 MLA 投影全量化且 `mla_q_layout` 非 global 时生效；`--no-graph` 也会整体关掉 |
| `prefill_f32` | `--prefill-f32` / `--prefill-f32=0` | `ENGINE_PREFILL_F32` | bool | `true` | 保留 f32 权重镜像用于 batch>1 prefill；MoE 或显存不足时自动关闭 |
| `gemm` | `--quant-gemm` | `ENGINE_GEMM` | bool | `false` | 自研量化 GEMM 用于 prefill（实验） |
| `no_moe_fused` | `--no-moe-fused` | `ENGINE_NO_MOE_FUSED` | bool | `false` | 禁用融合 MoE kernel |
| `keep_gpu_busy` | `--keep-gpu-busy` | `ENGINE_KEEP_GPU_BUSY` | bool | `false` | CPU 专家层执行时在副流跑时钟保温 kernel |

### 3.3.1 KV cache 精度 / 长上下文 / 上下文自适应

KV cache 是 decode 阶段唯一随上下文线性增长的显存与读放来源。默认 `f32` 保持历史布局
（短上下文数值与性能完全不变）；`q8_0` / `q4_0` 把每个注意力层的 K/V 按 32 元素一块量化
（块格式与 GEMV 用的 Q8_0 相同：2 B f16 scale + 32×int8 = 34 B；q4_0 为 2 B + 16 B = 18 B），
写入时在 `kv_write_q` 里量化，读取时由分块 online-softmax attention 内核（`attn_tiled`）
反量化并累加，不产生 f32 展开。

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `kv_type` | `--kv-type <v>` | `ENGINE_KV_TYPE` | string | `f32` | KV cache 精度：`f32`（默认/历史布局）、`q8_0`（34 B/32 值，推荐）、`q4_0`（18 B/32 值，最省显存，精度较低）。9B 上每 token KV：f32 64 KiB / q8_0 17 KiB / q4_0 9 KiB。MLA 模型（Coder-V2-Lite）强制 f32（无量化 MLA 内核，启动时会打印说明） |
| `kv_offload` | `--kv-offload[=0]` | `ENGINE_KV_OFFLOAD` | bool | `false` | 预留：把 KV 放主机内存。**当前版本未实现主机映射 KV**（会打印说明并回退为"KV 必须放显存"，由 `ctx_auto` 推导上限），1M 上下文的显存账见 §6.2 |
| `ctx_auto` | `--ctx-auto[=0]` | `ENGINE_CTX_AUTO` | bool | `true` | 装载前按 `空闲显存 − 权重 − 激活 − reserve` 与 `kv_type` 推导该配置能支持的最大 ctx，与 `--ctx` 取较小值并打印决策依据（`[ctx] budget ...` / `[ctx] CLAMPED ...` / 各 kv_type 的最大 ctx）。`--ctx 0` = 完全交给自适应推导 |
| `prefill_rows` | `--prefill-rows <v>` | `ENGINE_PREFILL_ROWS` | int [1,65536] | `512` | 激活缓冲的行上限：prompt 超过该长度时**分块 prefill**（每块 `prefill_rows` 个 token）。历史上激活（尤其 `logits_`）按 `max_seq × vocab × 4` 分配，`--ctx 131072` 时需要 130 TB，因此长上下文必须分块；显存不足时装载阶段会自动下调该值并打印 |
| `prefill_rows_max` | `--prefill-rows-max <v>` | `ENGINE_PREFILL_ROWS_MAX` | int [1,65536] | `4096` | 分块 prefill 的**自动放大上限**：`prefill_rows` 只是起点，装载时按 `权重+KV+激活(2×rows)+reserve ≤ 空闲显存` 逐次翻倍（最多到本值/`max_seq`）。块越大，整模型 dequant 与每块固定开销被摊得越薄（9B：4 块 → 1 块，pp +17%）。显存不足时仍会自动下调，决策打印 `[ctx] prefill chunk grown/reduced ...` |
| `prefill_f16` | `--prefill-f16[=0]` | `ENGINE_PREFILL_F16` | bool | `true` | **prefill（n_tokens>1）GEMM 走 fp16 tensor core**：权重直接反量化成 fp16（不经 f32 中转）+ cuBLAS fp16 + f32 累加。本机（RTX 3060）实测 12.8 T-MAC/s vs f32 cuBLAS 4.0 T-MAC/s（3.2×）；输出与 f32 路径相差 ~3e-4 相对（fp16 存储误差），四模型 top-1 逐位不变 |
| `prefill_f16_min_macs` | `--prefill-f16-min-macs <v>` | `ENGINE_PREFILL_F16_MIN_MACS` | int | `0` | 每次 matmul 的 MAC 数低于该值时保留原 f32/dequant 路径（0 = 全部走 fp16） |
| `prefill_f16_acc` | `--prefill-f16-acc[=0]` | `ENGINE_PREFILL_F16_ACC` | bool | `false` | **实验**：fp16 GEMM 用 fp16 累加（`CUBLAS_COMPUTE_16F`），再快 1.85×（22.3 T-MAC/s），代价是 K 方向归约里的 fp16 舍入（每 matmul ~0.3% 相对误差）。实测会把 1.6k 退化 prompt 的 top-1 翻掉（与逐 token 喂不一致），因此**默认关闭** |
| `prefill_graph` | `--prefill-graph[=0]` | `ENGINE_PREFILL_GRAPH` | bool | `false` | **实验**：按 chunk token 数捕获整条 prefill 流水线的 CUDA Graph（2341 次 launch → 1 次）。本机实测无收益（prefill 已是 GPU-bound，host 发射被完全掩盖），默认关闭 |
| `attn_pre_rows` | `--attn-pre-rows <v>` | `ENGINE_ATTN_PRE_ROWS` | int [1,32] | `8` | prefill attention 每 block 的 query 行数：同 block 的 8 个 warp 走同一份 K/V（L1 复用）。1 = 关（回到每 warp 一个 (token,head)）；实测 444 ms → 138 ms（1622 token 单块，8 层） |
| `attn_gqa` | `--attn-gqa[=0]` | `ENGINE_ATTN_GQA` | bool | `true` | prefill attention 的 GQA 组共享：一个 warp 负责一个 query 行的**全部 rep 个 q head**，K/V 只读一遍（4× 流量）。每个 head 的算术顺序不变 ⇒ 结果与 `attn_tiled_pre` 逐位一致。仅 f32 KV + head_dim 256（9B/35B） |
| `kv_split` | `--kv-split <v>` | `ENGINE_KV_SPLIT` | int [1,64] | `32` | decode attention 的 split-K 份数（一条 query 的 key 区间分给多少个 block）。prefill batch 固定用 1（每个 (token,head) 一个 block，寄存器内完成 softmax，不需要 scratch） |

**实现位置**：内核 `kv_write_q` / `attn_tiled` / `attn_combine`（`include/engine/kernels.h`，
新增在文件末尾；注入在 `src/runner.cpp` 的 NVRTC 源码拼接与 kernel 注册处）；
C++ 侧 `DenseRunner::kv_layer_bytes / alloc_kv_cache / kv_write_layer / run_attn /
run_attn_legacy / derive_ctx_limit / activation_bytes`（`src/runner.cpp`）；
两个调用点（通用 GQA 路径 `run_layer_body`、qwen35 9B 的 `run_q35_attn_attn`）统一改为
调用 `kv_write_layer` + `run_attn`。

**派发规则**（`run_attn`）：
- `kv_type == f32` 且 `ctx < 385` 且 `max_seq <= 12000` → 历史内核 `attn_cached_w`（短上下文数字不动）；
- 其余情况（量化 KV / 长上下文 / 大 `max_seq`）→ 分块 online-softmax 内核 `attn_tiled`：
  `n_tokens<=8` 时按 `kv_split` 做 split-K + `attn_combine` 合并（decode 并行度），
  `n_tokens>8` 时 `nsplit=1`、每个 (token,head) 一个 block、softmax 在寄存器内完成（prefill）。
  `head_dim` 不是 32 的倍数或大于 256 时无法走分块内核：f32 回退历史内核，量化 KV 直接报错。


### 3.4 模型语义覆盖（默认从 IR/GGUF metadata 推导）

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `norm_plus1` | `--norm-plus1 <v>` | `ENGINE_NORM_PLUS1` | int [-1,1] | `-1` | RMSNorm 零中心（+1）解释：-1 自动（仅 deepseek2 按数据判定），0/1 强制 |
| `no_yarn` | `--no-yarn` | `ENGINE_NO_YARN` | bool | `false` | 忽略 GGUF `rope.scaling.type=yarn` |
| `router` | `--router <v>` | `ENGINE_ROUTER` | string | `""` | MoE 路由覆盖：`topk_softmax` / `sigmoid_topk`（默认来自 IR） |
| `norm_topk` | `--norm-topk <v>` | `ENGINE_NORM_TOPK` | int [-1,1] | `-1` | top-k 权重归一化覆盖：-1 自动 |
| `experts_topk` | `--experts-topk <v>` | `ENGINE_EXPERTS_TOPK` | int [0,256] | `0` | 运行时覆盖 MoE top-k，0 = 按 IR |
| `no_shared` | `--no-shared` | `ENGINE_NO_SHARED` | bool | `false` | 跳过 shared expert（诊断） |
| `mla_q_layout` | `--mla-q-layout <v>` | `ENGINE_MLA_Q_LAYOUT` | string | `""` | MLA q 布局实验：`global` / `global_pe` / `perhead` |
| `mla_kv_layout` | `--mla-kv-layout <v>` | `ENGINE_MLA_KV_LAYOUT` | string | `""` | MLA kv 布局实验：`global` / `global_vk` / `perhead` / `perhead_vk` |

### 3.5 内核选择

`--kernel-calibrate` 会用本机真实权重形状微基准测出最优 GEMV 方案并写入校准缓存；
校准时每个形状的候选、耗时、GB/s 和最终选择都会打印。校准缓存命中时 `--print-config`
和 `RUNNER_DEBUG=1` 会显示 `(calibrated override)`。

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `dp4a_gemv` | `--no-dp4a-gemv` | `ENGINE_NO_DP4A_GEMV` | bool | `true` | Q5_K/Q6_K decode 使用 int8 dp4a GEMV |
| `dp4a_q6k` | `--no-dp4a-q6k` | `ENGINE_NO_DP4A_Q6K` | bool | `true` | Q6_K dp4a；关闭回退 fp32 x4 路径 |
| `tiny_sk` | `--no-tiny-sk` | `ENGINE_NO_TINY_SK` | bool | `true` | 小输出维（Q8_0, n_out<=128）split-K |
| `rms4` | `--no-rms4` | `ENGINE_NO_RMS4` | bool | `true` | float4 版 rmsnorm（1 个 block 的 norm 是延迟受限的：load/store 指令数 4× 减少，reduction 从 log2(blockDim) 次 `__syncthreads` 降到 1 次 warp-shuffle + 1 barrier）。要求宽度 %4==0 且 16 B 对齐，否则内核内自动回退标量路径 |
| `head_topk` | `--no-head-topk` | `ENGINE_NO_HEAD_TOPK` | bool | `true` | decode 头在**设备端**把 logits 行归约成 top-5（`topk_partial` + `topk_final`），只回传 5 组 (index,value) 而不是整行（0.4–1 MB，WDDM 上 pinned D2H ~0.27 ms/token）；`forward()` 仍返回整行 logits，只有 `forward_topk()` 走快路径 |
| `gd_delta_128` | `--no-gd-delta-128` | `ENGINE_NO_GD_DELTA_128` | bool | `true` | qwen35 gated-delta-net 的寄存器分块 kernel（head dim 128，state 布局 `[n_vh][KD][VD]`，每 token 只读写一次 2 MB state）；关闭回退通用 kernel |
| `ssm_fuse` | `--no-ssm-fuse` | `ENGINE_NO_SSM_FUSE` | bool | `true` | gated-delta-net decode 融合：`gd_ssm_prep`（conv1d+silu+写 q/k/v+conv state 移位+beta/g 一次发射）替换 4 次发射；`gd_delta_rule_128f` 把 q/k L2 norm 与 gated RMSNorm 折进 delta-rule kernel。仅 T==1 |
| `quant_reuse` | `--no-quant-reuse` | `ENGINE_NO_QUANT_REUSE` | bool | `true` | 连续 dp4a GEMV 读同一激活时复用 int8 两级量化 staging（gate/up、qkv/z、q/k/v）；用 `cuda::g_launch_calls` 保证只有"期间无任何内核发射"才复用 |
| `splitk` | `--splitk` | `ENGINE_SPLITK` | bool | `false` | x4 GEMV 的 split-K（实测持平，默认关） |
| `dp4a` | `--dp4a` | `ENGINE_DP4A` | bool | `false` | Q8_0/Q5_0 的 dp4a 实验路径 |
| `row_threshold` | `--row-threshold <v>` | `ENGINE_ROW_THRESHOLD` | int | `2048` | 长行判定：`n_in>=2048 && n_out>=row_threshold` 用 4 行/ warp kernel |
| `q6k_x4_min` | `--q6k-x4-min <v>` | `ENGINE_Q6K_X4_MIN` | int | `32768` | Q6_K 行数低于该值时 1-row kernel 更快（实测） |
| `q6k_split` | `--q6k-split <v>` | `ENGINE_Q6K_SPLIT` | int [0,8] | `0` | 固定 Q6_K dp4a split-K 因子，0 = 自动 |
| `half` | `--half` | `ENGINE_HALF` | bool | `false` | fp16 NT prefill 镜像（实验） |
| `half_t` | `--half-t` | `ENGINE_HALF_T` | bool | `false` | fp16 TN prefill 镜像（实验） |
| `bm_types` | `--bm-types <v>` | `ENGINE_BM_TYPES` | string | `""` | 需要 block-major 重排的量化类型列表（默认 `Q8_0`） |
| `no_bm` | `--no-bm` | `ENGINE_NO_BM` | bool | `false` | 禁用 block-major 重排 |
| `no_gemv` | `--no-gemv` | `ENGINE_NO_GEMV` | bool | `false` | 禁用专用 GEMV（诊断 A/B） |
| `no_avx2` | `--no-avx2` | `ENGINE_NO_AVX2` | bool | `false` | CPU 量化 GEMV 禁用 AVX2 |
| `cpu_moe_batch` | `--no-cpu-moe-batch` | `ENGINE_NO_CPU_MOE_BATCH` | bool | `true` | 批量 CPU MoE：每层只开 2 个并行区（gate+up、down），不再每个 matvec 唤醒一次线程池（env 反转：设了 env 就关闭） |
| `cpu_int8` | `--cpu-int8 <n>` | `ENGINE_CPU_INT8` | int [0,2] | `1` | CPU 侧 int8 行内核（两级 int8 激活量化 + AVX2 `maddubs`，按量化类型/宽度分派，不按模型）：`0`=关闭（fp32 行内核）、`1`=默认，只在实测更快的量化类型上用（Q6_K 行；隔离 35B CPU 层配对中位 **−10.1%**，0.81→0.73 ms/层，覆盖 Coder 的 Q5_K/Q5_1 专家不受影响）、`2`=Q4_K 也用（实测更慢 +4.6%，仅供 A/B）。数值 rel_l2 ≤3.1e-5，见 `docs/CPU-INT8.md` |
| `tiny_sk_max_nout` | `--tiny-sk-max-nout <v>` | `ENGINE_TINY_SK_MAX_NOUT` | int [0,1e6] | `128` | Q8_0 的 tiny-n_out split-K 门槛（n_out ≤ 该值才用）。默认 128 = 本机实测：再大时非 split 内核也能填满 SM |
| `tiny_sk_min_nin` | `--tiny-sk-min-nin <v>` | `ENGINE_TINY_SK_MIN_NIN` | int [0,1e6] | `512` | 同上：n_in ≥ 该值才值得 split（512 = 至少 16 个 K-block，否则归约比省下的波次还贵） |
| `cpu_pool_spin` | `--cpu-pool-spin <v>` | `ENGINE_POOL_SPIN` | int [0,1e6] | `512` | CPU worker 空闲自旋长度（PAUSE 次数，512 ≈ 20 µs），覆盖上一个并行区的尾部后再退回 condvar |
| `attn_warp_ctx_mid` | `--attn-warp-mid <v>` | `ENGINE_ATTN_WARP_CTX_MID` | int [1,1e6] | `96` | cached-attention 的 warp 分桶下界：ctx 超过它用 16 warp 而不是 8。**CUDA-graph 分桶与 launch 用同一个函数**，capture/replay 一定一致；默认 96 = 本机实测 |
| `attn_warp_ctx_hi` | `--attn-warp-hi <v>` | `ENGINE_ATTN_WARP_CTX_HI` | int [1,1e6] | `384` | 上界：ctx 超过它用 32 warp（随后被 shared-memory 预算收敛，见 `LONGCTX-REPORT.md` — 这就是 >384 token 崩溃的修复） |
| `q6k_dp4a_split_max_nout` | `--q6k-dp4a-split-max-nout <v>` | `ENGINE_Q6K_DP4A_SPLIT_MAX_NOUT` | int [0,1e6] | `4096` | Q6_K dp4a 自动 split-K 的 n_out 交叉点（实测；具体形状仍可被 `--kernel-calibrate` 缓存覆盖） |
| `small_nout_split_max` | `--small-nout-split-max <v>` | `ENGINE_SMALL_NOUT_SPLIT_MAX` | int [0,1e6] | `256` | Q4_K/Q5_0 用 2-warp split-K 的 n_out 上限（256 = 8 行/block × 32 block） |
| `kernel_cache` | `--kernel-cache <v>` | — | string | `engine-kernel-cache.txt` | 校准缓存路径 |
| `calib_iters` | `--calib-iters <v>` | — | int [1,10000] | `40` | 每个候选的计时迭代数 |

#### 3.5.1 Windows 进程/线程调度优先级（真实桌面机）

本机是**真实物理机**（i5-10400F 6C/12T + RTX 3060 12 GB + Windows 11），桌面后台程序
（LANDrop / Steam / QQ / 浏览器 / 远程工具）会真实抢 CPU。宿主 GEMV（CPU 层专家 matvec，
9 线程、算力受限）一旦被打断，24 个 CPU 层每 token 的损失是毫秒级。这三个键把调度面
收进配置系统，默认值就有收益、也能一键回退做 A/B：

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `win_priority` | `--win-priority <n>` | `ENGINE_WIN_PRIORITY` | int [0,2] | `1` | 进程优先级类：`1`=`HIGH_PRIORITY_CLASS`（默认），`2`=`ABOVE_NORMAL_PRIORITY_CLASS`，`0`=`NORMAL`（关闭 / 回退到历史行为）。**不用 REALTIME**：会让鼠标/IO/驱动提交线程饿死，且需要额外权限。非 Windows 上解析但不生效 |
| `win_thread_priority` | `--win-thread-priority <n>` | `ENGINE_WIN_THREAD_PRIORITY` | int [-1,2] | `-1` | CPU worker 线程的**相对**优先级：`-1`=auto（`win_priority>0` 时 `THREAD_PRIORITY_HIGHEST`，否则继承），`0`=继承，`1`=ABOVE_NORMAL，`2`=HIGHEST（base+2）。用于区分"主线程还是 worker 是瓶颈"的 A/B |
| `win_affinity` | `--win-affinity <n>` | `ENGINE_WIN_AFFINITY` | int [0,2] | **`2`** | `2`=**SMT 成对绑核（默认）**：逻辑线程 2k/2k+1 绑到物理核 k 的两个硬件兄弟，兄弟对**从 `GetLogicalProcessorInformationEx(RelationProcessorCore)` 拓扑动态读取**（不假设 i,i+n；主线程=逻辑线程 0；`--threads` 为奇数时 = 若干整对 + 1 个单绑）；`0`=关闭（OS 调度，2026-11 前的旧默认）；`1`=物理核优先顺序。默认线程数同时改为"能读到 SMT=2 兄弟对时 = 2×物理核（本机 12），否则回退 1.5×物理核"（本机实测成对 12/11/10 比 9 不绑快 4–12%，完整 8×2 扫描见 `reports/THREAD-AFFINITY-REPORT.md`） |

生效位置：

- 进程类在**程序启动时**（`run` / `serve` / `test_gen` 等入口，加载模型之前）由
  `config::apply_win_priority()` 设置一次；`serve` 与 `run` 同一个入口，因此长期服务同样受益。
- worker 线程相对优先级 / 亲和性在 CPU 线程池创建时（`src/cpu_matvec.cpp` 的 `cpu::Pool`）
  按配置设置，正好在第一次 CPU 层 matvec 之前。
- 启动日志会打印一行 `[win] process priority class = HIGH (win_priority=1)`，脚本可据此
  确认测量环境；`ENGINE_DBG_SCHED=1` 额外打印 worker 优先级/亲和性。
- 数值：优先级只改变调度顺序，**不改变任何计算**，逐 token 输出 bit-exact（`chk`/`gen`
  序列在 normal 与 HIGH 下完全相同）。

### 3.6 后端路径与诊断

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `nvrtc_path` | `--nvrtc-path <v>` | `DSH_NVRTC_PATH` | string | `""`（回退 pip 路径） | `nvrtc64_120_0.dll` 路径 |
| `cublas_dir` | `--cublas-dir <v>` | `DSH_CUBLAS_DIR` | string | `""`（回退 pip 路径） | `cublas64_12.dll` 所在目录 |
| `debug` | `--debug` | `RUNNER_DEBUG` | bool | `false` | 打印放置/内核选择/显存决策依据 |
| `print_config` | `--print-config` | — | bool | `false` | 打印生效配置后退出 |
| `kernel_calibrate` | `--kernel-calibrate` | — | bool | `false` | 内进程微基准校准内核选择并写缓存后退出 |
| `config` | `--config <v>` | — | string | `""` | 配置文件路径 |

### 3.7 HTTP 服务（`engine serve`）

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `serve_host` | `--host <v>` | `ENGINE_SERVE_HOST` | string | `127.0.0.1` | 绑定地址。`127.0.0.1` = 仅本机；`0.0.0.0` = 局域网可访问（**无鉴权**，仅建议可信网络）。`/shutdown` 端点只在回环地址时开放 |
| `serve_port` | `--port <v>` | `ENGINE_SERVE_PORT` | int [1,65535] | `7030` | HTTP 端口 |
| `serve_ctx` | `--serve-ctx <v>` | `ENGINE_SERVE_CTX` | int [0,1M] | `0` | 服务用上下文长度；`0` = 若给了 `--ctx` 用它，否则 `4096`。KV cache 在启动时按此分配，也是"prompt+生成长度"的硬上限 |
| `default_max_tokens` | `--default-max-tokens <v>` | `ENGINE_DEFAULT_MAX_TOKENS` | int [1,1M] | `1048576` | 请求没带 `max_tokens`/`n_predict` 时用（= 只受 ctx 限制 = 不截断思考） |
| `default_temperature` | `--default-temperature <v>` | `ENGINE_DEFAULT_TEMPERATURE` | float [0,∞) | `0.8` | 默认采样温度；`0` = greedy |
| `default_top_p` | `--default-top-p <v>` | `ENGINE_DEFAULT_TOP_P` | float [0,1] | `0.95` | 默认 nucleus |
| `default_top_k` | `--default-top-k <v>` | `ENGINE_DEFAULT_TOP_K` | int [0,1M] | `40` | 默认 top-k（`0` = 关闭） |
| `default_repeat_penalty` | `--default-repeat-penalty <v>` | `ENGINE_DEFAULT_REPEAT_PENALTY` | float | `1.0` | 默认重复惩罚（`1.0` = 关闭） |
| `serve_prefix_cache` | `--prefix-cache[=0]` | `ENGINE_SERVE_PREFIX_CACHE` | bool | `true` | 多轮前缀 KV 缓存：命中上一轮对话历史就只 prefill 新增 token（混合模型同时恢复 SSM conv/rec 状态快照）。`=0` 关闭（A/B 对比用）。见 `SERVE.md` §5.2 |
| `serve_no_limit_tokens` | `--no-limit-tokens <v>` | `ENGINE_SERVE_NO_LIMIT_TOKENS` | int [1,1M] | `1024` | 客户端请求"不限制"（`max_tokens<=0`，llama.cpp webui 发 `-1`）时的实际回复上限，避免思考模型单条跑几分钟 |
| `serve_think_budget` | `--think-budget <v>` | `ENGINE_SERVE_THINK_BUDGET` | int [0,1M] | `512` | 思考段超过 N token 仍未闭合 ⇒ 强制 `</think>`，并用"关闭思考"模板重跑回答（`0` = 关闭）。Qwen3.x 思考模型在 temp≥0.8 时常见"自述循环不闭合"，见 `SERVE.md` §5.3 |
| `serve_repeat_guard` | `--repeat-guard[=0]` | `ENGINE_SERVE_REPEAT_GUARD` | bool | `true` | 复读兜底：连续 3 句完全相同（≥8 字符）就提前 stop，避免"同一句话复读 10 遍"刷到 token 上限 |

`--model <path>` / 位置参数照旧指定模型；`--ctx`、`--threads`、`--gpu-layers`、`--vram-margin` 等
全局参数在 `serve` 下同样生效（`--ctx` 只决定 KV cache 大小，除非显式给 `--serve-ctx`）。

**serve 的采样默认值**：请求没有显式给采样参数时，serve 默认用 Qwen3 官方推荐的
`temperature=0.6, top_k=20, top_p=0.95`（`/props.default_generation_settings` 可见；
显式 `--default-temperature` / `--default-top-k` / `--default-top-p` 会覆盖它）。
服务端托管的 llama.cpp webui 前端默认值同步改成这组，旧 localStorage 配置会做一次性迁移。
模型在更高温度（如 0.8）下容易在 thinking 里自述循环，见 `SERVE.md` §5.3。

**serve 的 prefill chunk 默认值**：若没有显式设置 `prefill_rows`，`engine serve` 会把
`prefill_rows`（和 `prefill_rows_max`）钉在 `128`。前缀缓存的快照/切分点按该 chunk 边界对齐，
128 让"每轮最多多 prefill <128 token"；同时 128 行的 activation/logits 缓冲更小，能多放 GPU 专家。
显式 `--prefill-rows <v>` / `ENGINE_PREFILL_ROWS` 会覆盖它（此时缓存粒度变粗，但仍逐 token 一致）。

---

### 3.8 MTP / 投机解码（`--mtp`）

模型若带 **MTP（multi-token prediction）头**（GGUF 里 `blk.N.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`
四个张量 + 一个完整的 attention+FFN 层，`block_count = 主层数 + nextn_predict_layers`），
引擎会把它当作"第 N+1 个 block"单独装载（`ir.mtp_block`），并用它做**贪心投机解码**：

1. **draft**：MTP 层从目标模型最后一个位置的残差流 `h_p` + 待发 token 的 embedding 预测
   `t_{p+2}`，链式重复 k 次（`--mtp-k`）；
2. **verify**：把 `[t_next, d_1..d_k]` 一次批前向送给目标模型，逐行取 arg-max；
3. **accept**：只有与目标模型自身 arg-max 完全一致的最长 draft 前缀被保留，否则丢弃。

⇒ 输出 token 序列与纯贪心**逐 token 完全一致**（投机解码是无损的，draft 只影响速度不影响结果）。
能力判定只看 `ir.mtp_block >= 0`（"有没有 nextn 层"），不做模型名特判。

| key | CLI | env | 类型 | 默认 | 说明 |
|---|---|---|---|---|---|
| `mtp` | `--mtp` | `ENGINE_MTP` | bool | `false` | 开启 MTP 投机解码；模型没有 nextn 层时自动无效（`[mtp]` 日志说明原因） |
| `mtp_k` | `--mtp-k <v>` | `ENGINE_MTP_K` | int [1,3] | `2` | 每轮链式 draft 的 token 数 |
| `mtp_gpu` | `--no-mtp-gpu` | `ENGINE_NO_MTP_GPU` | bool | `true` | MTP 层的 256 个路由专家放显存（放不下自动退回 CPU，退回时每个 draft 步多一次 host 往返） |
| `mtp_vram_margin_mb` | `--mtp-vram-margin <v>` | `ENGINE_MTP_VRAM_MARGIN_MB` | int | `256` | 把 MTP 专家放 GPU 前要求的空闲显存余量 |
| `mtp_ctx_prefill` | `--no-mtp-ctx-prefill` | `ENGINE_NO_MTP_CTX_PREFILL` | bool | `true` | 用一次批前向把 prompt 各位置的 MTP KV 历史填满（否则 draft 层对着一窗零值做 attention，接受率明显下降） |

**状态与回滚**：attention KV 是位置索引的（回滚免费，直接重写槽位）；gated-delta-net 的
`conv_state`/`rec_state` 是原地递推，verify 时线性层按 token 逐个推进并保存每个位置的快照
（`[mtp] ssm rollback slots` 日志给出显存账），被拒绝的 draft 全部回滚。

**收益判据**：`S = (1+α+…+α^k) / (c + k·d)`，`c` = 一次 (k+1) token 验证前向 / 一次 decode，
`d` = 一次 draft 步 / 一次 decode。本机实测（2026-11-15）：
- **Qwythos-9B-MTP（稠密、全 GPU）** α≈0.7–0.8，`c`≈1.2 ⇒ 净加速；
- **Qwen3.6-35B-A3B（24/40 层 CPU 专家）** α≈0.83 但 `c`≈2.0（批大小翻倍 = CPU 专家对翻倍），
  ⇒ 投机解码**不划算**，35B 上默认不要开（详见 `MTP-IMPL-REPORT.md`）。

示例：
```bash
./build/test_gen.exe --model Qwythos-9B-...-MTP-Q4_K_M.gguf --ids 760 --n 32 --mtp --mtp-k 2
./build/mtp_run.exe --model <m.gguf> --ids 760 --mode acc --mtp --n 16 --verbose   # draft 命中率
./build/mtp_run.exe --model <m.gguf> --ids 760 --mode spec --mtp --mtp-k 2 --n 32   # 对拍 + 测速
```

## 4. 兼容环境变量（保留但不推荐）

所有仍被代码读取的环境变量：

**已收编进配置系统**（可以用 config/CLI 覆盖，见上表 env 列）：
`ENGINE_CTX`, `ENGINE_THREADS`, `ENGINE_GPU_LAYERS`, `ENGINE_GPU_EXPERTS`, `ENGINE_LAYERS`,
`ENGINE_VRAM_MARGIN_MB`, `ENGINE_VRAM_BUDGET_MB`, `ENGINE_WORKSPACE_MB`, `ENGINE_DEVICE`,
`ENGINE_NO_GRAPH`, `ENGINE_LAYER_GRAPH`, `ENGINE_MLA_NO_GRAPH`, `ENGINE_PREFILL_F32`,
`ENGINE_GEMM`, `ENGINE_NO_MOE_FUSED`, `ENGINE_KEEP_GPU_BUSY`, `ENGINE_NORM_PLUS1`,
`ENGINE_NO_YARN`, `ENGINE_ROUTER`, `ENGINE_NORM_TOPK`, `ENGINE_EXPERTS_TOPK`,
`ENGINE_NO_SHARED`, `ENGINE_MLA_Q_LAYOUT`, `ENGINE_MLA_KV_LAYOUT`,
`ENGINE_NO_DP4A_GEMV`, `ENGINE_NO_DP4A_Q6K`, `ENGINE_NO_TINY_SK`, `ENGINE_SPLITK`,
`ENGINE_DP4A`, `ENGINE_ROW_THRESHOLD`, `ENGINE_Q6K_X4_MIN`, `ENGINE_Q6K_SPLIT`,
`ENGINE_HALF`, `ENGINE_HALF_T`, `ENGINE_BM_TYPES`, `ENGINE_NO_BM`, `ENGINE_NO_GEMV`,
`ENGINE_NO_AVX2`, `ENGINE_SERVE_HOST`, `ENGINE_SERVE_PORT`, `ENGINE_SERVE_CTX`,
`ENGINE_DEFAULT_MAX_TOKENS`, `ENGINE_DEFAULT_TEMPERATURE`, `ENGINE_DEFAULT_TOP_P`,
`ENGINE_DEFAULT_TOP_K`, `ENGINE_DEFAULT_REPEAT_PENALTY`, `DSH_NVRTC_PATH`,
`DSH_CUBLAS_DIR`, `RUNNER_DEBUG`.

**本轮补齐/新增的兼容变量**（同样走注册表，优先级 CLI > config > env > default）：
`ENGINE_MODEL`, `ENGINE_NO_VRAM_CHECK`, `ENGINE_HOST_MARGIN_MB`, `ENGINE_PLACEMENT`,
`ENGINE_SSD_STREAM`, `ENGINE_SSD_CACHE`, `ENGINE_ALLOW_OVERSUBSCRIBE`, `ENGINE_INTERACTIVE`,
`ENGINE_ASSUME_YES`, `ENGINE_MODEL_DIR`, `ENGINE_PRINT_CONFIG`, `ENGINE_KERNEL_CALIBRATE`,
`ENGINE_KERNEL_CACHE`, `ENGINE_CALIB_ITERS`。

**仅诊断/转储，未进配置**（打印或 dump 用，不改变数值行为）：
`ENGINE_PROFILE`, `ENGINE_KPROF`, `ENGINE_PIPE_PROF`, `ENGINE_DBG_VRAM`, `ENGINE_DBG_LOAD`,
`ENGINE_DBG_LOAD2`, `ENGINE_DBG_LG`, `ENGINE_DBG_LG2`, `ENGINE_DBG_LAYERMS`,
`ENGINE_DBG_MATVEC`, `ENGINE_DBG_GEMVTIME`, `ENGINE_DBG_BIGGEMV`, `ENGINE_DBG_SCRATCH`,
`ENGINE_DBG_CPULAYER`, `ENGINE_DBG_ARENA`, `ENGINE_DBG_DEVROUTE`, `ENGINE_DBG_DEVWHY`,
`ENGINE_DBG_MOEHASH`, `ENGINE_DBG_MOEHASH_LAYER`, `ENGINE_DBG_Q35LOAD`, `ENGINE_DUMP`,
`ENGINE_DUMP_EVERY`, `ENGINE_DUMP_LAYER`, `ENGINE_DUMP_MLA`, `ENGINE_DUMP_Q35`,
`ENGINE_DUMP_ROPE`, `ENGINE_DUMP_MOE_SPLIT`, `ENGINE_LOGITS`, `ENGINE_TOPK`,
`ENGINE_TARGET`, `ENGINE_SKIP_WARMUP`, `ENGINE_SPLIT_PREFILL`, `ENGINE_SPLIT_ONLY`,
`ENGINE_SKIP_LMHEAD`, `ENGINE_LAYER_GRAPH`（实验开关，已同时暴露为 config）, `ENGINE_LG_*`,
`ENGINE_DEV_ROUTE`, `ENGINE_NO_DEV_ROUTE`, `ENGINE_DEV_NO_Q51`, `ENGINE_ROUTER_KERNEL`,
`ENGINE_MOE_FIXED_ROUTING`, `ENGINE_EXPERT_ARENA`, `ENGINE_SH_SCALE`, `ENGINE_NO_SHARED`（已进配置）,
`ENGINE_MLA_ROPE_NEOX`, `ENGINE_SKIP_MLA_ATTN`, `ENGINE_MOE_SWAP_GU`, `ENGINE_Q35_RAW_A`,
`ENGINE_SH_SWAP_GU`, `ENGINE_NO_SHARED`, `PROBE_DEV_KERNEL`, `ENGINE_CPU_LAYER_GRAPH`（CPU 驻留层也录图，默认关，实测净变慢）、`ENGINE_NO_CPU_LAYER_GRAPH` / `ENGINE_NO_GPU_LAYER_GRAPH`（分别关闭两类层图）、`ENGINE_GPU_FULL_LAYER_GRAPH`（GPU 层整层录图，含设备路由 MoE）、`ENGINE_GRAPH_PRECAPTURE`（在构造函数里预捕获，本机驱动上重放会 illegal access，仅诊断）、`ENGINE_GRAPH_ONLY_LAYER=<n>` / `ENGINE_GRAPH_SELFTEST` / `ENGINE_GRAPH_KERNEL_LIMIT=<n>` / `ENGINE_GRAPH_FORCE`（图捕获的定位工具）、`ENGINE_DBG_GRAPH` / `ENGINE_DBG_GRAPH_SYNC`、`ENGINE_DEC_SKIP=<n>`（test_gen：跳过前 n 个 decode token 再计时，用于排除一次性捕获开销）。

> 完整清单与分类（共 64 个仅 env 变量：纯诊断 / 模型语义取证 / 实验性执行路径）见
> `FEATURE-MATRIX.md` §6.1；本轮新增测试钩子 `ENGINE_FORCE_INTERACTIVE` 也在其中。
> 产品功能开关都已进注册表，这里保留的都是取证/实验用途。

---

## 5. 内核校准（`--kernel-calibrate`）

目标：把 `q6k_x4_min`、dp4a 分派、tiny-SK 等硬编码阈值替换为**本机实测**。

```bash
# 1) 用真实模型跑一次校淮（默认写 engine-kernel-cache.txt）
./build/test_gen.exe --model ./models/Qwen3.5-9B-Q5_K_M.gguf --ids 760 --n 1 --kernel-calibrate
# 2) 后续运行自动读取缓存；--print-config 会列出命中的 shape
./build/engine.exe --print-config
```

- 校准器枚举模型里所有不同的 `(quant type, n_out, n_in)` GEMV 形状（含 lm_head、注意力、
  FFN、MoE 专家、qwen3.5 SSM 投影），对每个形状跑候选 kernel（plain/x4/k2/p4/BM/dp4a/
  dp4a+shared/split-K），同步计时，比较 GB/s；
- 只有当校准结果比启发式**快 4% 以上**才覆盖启发式（避免小形状测量噪声导致抖动）；
- 缓存键是精确形状，换模型/换卡后旧条目不会误用（会用兼容性检查拒绝）；
- 缓存格式见 §2，删除该文件即可回到纯启发式。

---

## 6. 自适应显存放置

1. **规划阶段**（装载前）：用 GGUF 字节数 + KV/workspace 预留 + `vram_margin_mb`
   估算最多能放多少层 MoE 专家到显存（层粒度，与 llama.cpp `--n-cpu-moe` 同构）；
2. **装载阶段**：每装一层 MoE 专家前调用 `cuMemGetInfo` 实测空闲显存，
   若 `free - 该层专家字节 - KV字节 < vram_margin_mb`，把该层及之后所有 MoE 层放 CPU，
   并打印一行决策依据（层号 / 需要多少 / 实际空闲 / margin / 前后层数）；
3. **装载后**：再校验一次 `free >= margin`，打印 `[vram] check OK` 或告警并给出
   `--gpu-layers` / `--vram-margin` 调整建议。

`RUNNER_DEBUG=1` 会额外打印每层的判定数值；`--print-config` 打印生效的
`vram_margin_mb / vram_budget_mb / workspace_mb / gpu_layers` 及其来源。

### 6.1 KV cache 与上下文长度自适应推导

装载前（IR 解析完、权重加载前）按下面的公式给 `--ctx` 封顶（`ctx_auto=true` 时）：

```
kv_bytes_per_token = Σ_attention_layers (n_kv · head_dim · 2 · bytes_per_value)   # f32=4, q8_0=34/32, q4_0=18/32
kv_total(ctx)      = kv_bytes_per_token · ctx
act(rows)          = rows · (6·hidden + 2·n_head·head_dim + 2·n_kv·head_dim + 3·ffn + vocab) · 4
reserve            = vram_margin_mb + workspace_mb
max_ctx            = floor((free_at_load − resident_weights − act(prefill_rows) − reserve) / kv_bytes_per_token)
final_ctx          = min(--ctx, max_ctx)
```

- `free_at_load` 是装载前的 `cuMemGetInfo`（此时权重尚未占显存），`resident_weights` 取 IR 的
  非专家权重字节（量化 GGUF 的驻留字节 ≈ 文件字节）。
- `act(rows)` 是**分块 prefill** 下激活缓冲的实际大小（`rows = min(final_ctx, prefill_rows)`），
  其中 `vocab·4` 是 logits 行；这也是 128k/256k 能加载的前提。
- 打印四类行：`[ctx] requested=…->… kv_type=… kv/token=…`、`[ctx] budget: free − weights − act − reserve
  -> max_ctx`、被夹紧时的 `[ctx] CLAMPED: … would need … MiB, budget … MiB` + **三种 kv_type 各自
  的最大 ctx**、以及 `[ctx] prefill chunk reduced … rows`。被夹紧时若 `kv_offload=on` 会额外说明。
- 权重本身超过显存（MoE 需要 CPU 专家）时公式仍按"非专家权重"计算，KV 的余量由模型放置的
  自适应逻辑（§6）统一使用同一份 `kv_layer_bytes` 数字。

### 6.2 1M 上下文可行性（9B 的显存账，实测）

9B（qwen35 hybrid）32 层里 **8 层**是全注意力，`n_kv=4`、`head_dim=256` → 每 token KV = 17 KiB(q8_0)。
所以"把 KV 拉到模型上限 1M"需要：KV 17 GiB(q8_0) / 9.66 GiB(q4_0) / 64 GiB(f32)，而权重本身
6.13 GiB、本机 12 GiB 显存 ⇒ **1M 在单卡上不可行**（q4_0 也差 ~5 GiB）。GPU 可支持的上限
（`--vram-margin 1200`、`--prefill-rows 512`、实测 `free≈11.2 GiB`）：f32 ≈ 28–38k、q8_0 ≈ 105–143k、
q4_0 ≈ 199–270k（区间来自模型/参数差异）。要 1M 只能 KV offload 到主机（`kv_offload`，当前版本
未实现）：即使实现，decode 每步要重读 ~9.66 GiB KV，PCIe 4.4–12 GB/s ⇒ 0.8–2.2 s/token，
prefill 更是不可用，因此本机把 1M 当作"不可行"报告，推荐 9B 使用 `--ctx 131072`（q8_0）或
`--ctx 262144`（q4_0）。

### 6.3 放置决策器（gpu / vram+ram split / ssd stream，本轮新增）

装载前 `engine serve/run/plan` 会打印一份 `[placement]` 账目，并给出三档决策：

| 判定 | 条件（全部为引擎真实驻留字节） | 后续路径 |
|---|---|---|
| `gpu` | `fixed + experts + KV(ctx) + workspace <= 可用VRAM` | 全部权重常驻显存（最快） |
| `vram+ram split` | 上面不成立，但 `fixed + KV + workspace <= 可用VRAM` 且剩余专家字节 <= 可用RAM（只有 MoE 能走） | 现有 CPU 专家卸载路径；预计 N/M 层常驻显存，实际由 runner 装载时实测逐层决定 |
| `ssd stream` | 连 VRAM+RAM 都装不下 | **候选**，默认关闭；`ssd_stream`/`--placement ssd` 显式开启 → 本版本报“未实现”（接口占位）；`--allow-oversubscribe` 可强行让驱动分页 |

其中 `可用VRAM = 实测 free VRAM - vram_margin_mb`，`可用RAM = 实测 free RAM - host_margin_mb`，
`KV(ctx)` 与 `DenseRunner::kv_layer_bytes()` 同口径（f32=4 B/元素、q8_0=34 B/32 值、
q4_0=18 B/32 值；MLA 按引擎真实分配的 per-head K/V 展开；hybrid 只算持 KV 的层）。

`--placement gpu/split/ssd` 是显式策略：`gpu` 装不下会打印 WARN 但继续（用户强制）；
`split` 遇到 dense 模型会直接报错（无 host dense 路径）；`ssd` 报未实现。
`engine plan --vram-gib <F>` 可以离线模拟设备预算，打印同一份账目（JSON 输出字段不变）。

本机实测（GPU 空闲基线，ctx 4096，kv_type f32）：0.5B / 9B / Qwythos-9B → `gpu`；
Coder-V2-Lite（11.03 GiB，专家 10.14 GiB）→ `split ≈13/26 层`；
35B-A3B（20.33 GiB，专家 18.16 GiB）→ `split ≈14/40 层`；**没有模型需要 SSD**。
完整表格与门控规则见 `FEATURE-MATRIX.md` §1。

### 6.4 SSD 流式加载（接口占位，未实现）

- 触发条件只有一条：模型总字节 > 可用 VRAM + 可用 RAM。**默认不启用**，且目前**没有实现**。
- 显式 `--ssd-stream` / `--placement ssd` 会调用 `placement::SsdExpertStreamer::open()`，
  打印确定性错误（含 cache 目录、prefetch 深度、I/O 粒度、`FILE_FLAG_OVERLAPPED`、设计文档位置），
  进程以退出码 3 结束——不会静默回退到“边跑边分页”。
- 方案（下一轮实现）：专家粒度流式（attention/norm/embedding 常驻）、pinned host staging
  双缓冲、`pread`/`ReadFile(FILE_FLAG_OVERLAPPED)`、`(layer,expert)` LRU、每槽 event 等待
  （不做全局同步/UVM 零拷贝）。详见 `FEATURE-MATRIX.md` §2。

## 7. HTTP 服务（`engine serve`）

> 面向使用者的完整说明（启动命令、curl/浏览器/OpenAI SDK 示例、已知限制）见
> [SERVE.md](SERVE.md)；本节聚焦配置项与端点清单。

零第三方依赖（自写 WinSock2 HTTP/1.1，`ws2_32.dll` 运行时动态加载，不改 build.sh 链接选项）。
**单进程只加载一次模型、请求串行处理**（单用户；第二个请求在 listen backlog 里排队，
不会并发跑两份模型）。

### 7.1 端点

| method | path | 说明 |
|---|---|---|
| GET | `/` | 内置浏览器聊天页面（流式输出、多轮上下文、思考开关） |
| GET | `/health` | `{"status":"ok","model":...,"n_ctx":...,"slots_idle":...}` |
| GET | `/v1/models` | OpenAI 模型列表 |
| GET | `/props` | 模型路径/上下文/分词器/chat template/端口（llama.cpp `/props` 风格） |
| POST | `/v1/chat/completions` | OpenAI chat：`messages[]`、`stream`（SSE）、`temperature`/`top_p`/`top_k`/`min_p`/`max_tokens`/`stop`/`seed`、`enable_thinking` / `chat_template_kwargs.enable_thinking` |
| POST | `/v1/completions` | OpenAI legacy：`prompt` 字符串 |
| POST | `/completion` | llama.cpp 风格：`prompt`、`n_predict`、`stream`；返回 `content`/`stop`/`tokens_predicted`/`generation_settings` |
| POST | `/tokenize` | `{"content": "..."}` -> `{"tokens":[ids],"pieces":[...]}` |
| POST | `/detokenize` | `{"tokens":[ids]}` -> `{"content": "..."}` |
| POST | `/shutdown` | 优雅停机（**仅回环地址生效**），退出前释放显存 |
| OPTIONS | any | CORS 预检 |

生成响应额外带 `timings`（prefill/decode 毫秒、tok/s）和 `generated_token_ids`（便于和
`test_gen` 做逐 token 对拍）。chat 响应把 Qwen 的思考段拆到 `reasoning_content`，
`content` 只放最终回答；SSE 流里思考段用 `delta.reasoning_content`。

### 7.2 例子

```bash
# 健康检查 / 模型信息
curl -s http://127.0.0.1:7030/health
curl -s http://127.0.0.1:7030/props

# 非流式对话
curl -s http://127.0.0.1:7030/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages":[{"role":"user","content":"用一句话介绍你自己"}],
  "temperature":0.8,"top_p":0.95,"top_k":40,"max_tokens":256,"stream":false}'

# SSE 流式
curl -N http://127.0.0.1:7030/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages":[{"role":"user","content":"讲个笑话"}],"stream":true}'

# llama.cpp 风格
curl -s http://127.0.0.1:7030/completion -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0}'
```

浏览器直接打开 `http://127.0.0.1:7030/` 即可聊天。

### 7.3 行为与限制

- **串行**：同一时刻只跑一个请求；生成中客户端断开 → SSE 写失败 → 立即中止生成（不浪费 GPU）。
- **上下文**：`--ctx` / `--serve-ctx` 是硬上限，prompt 超限返回 400。
- **多轮前缀缓存（默认开，`--prefix-cache=0` 可关）**：请求的对话历史与上一轮缓存前缀比较，
  命中就复用 KV cache（qwen35 混合层还会恢复 SSM conv/rec 状态快照），只 prefill 新增 token。
  快照点对齐 `prefill_rows`（serve 默认 128）边界，缓存路径与整段 prefill 的 chunk 序列完全一致，
  贪心输出**逐 token 相同**（`tools/serve_prefix_cache_check.py` 硬验收）。`cache_prompt:false`
  可对单请求关闭；`/props.prefix_cache` 与日志 `[new N, cached M]` 可观测。
- **思考预算/复读兜底**：默认 `--think-budget 512`（强制闭合后用关闭思考的模板重答）+
  `--repeat-guard`（连续 3 句相同提前停）；`timings.think_budget_hit` / `repeat_guard_hit` 可观测。
- **首字延迟**：整段 prefill 仍是 35B 的瓶颈（pp 48–55 tok/s），但多轮对话只 prefill 新增 token，
  webui 实测首字（思考段第一个 delta）0.8–1.9 s；单轮/首轮仍按 `timings.prefill_ms` 如实给出。
  响应里的 `timings.prefill_ms`/`cache_hit_tokens` 可确认缓存是否生效。
- **webui 思考段**：llama.cpp webui 只渲染 `delta.content`，所以思考段同时以
  `<think>思考</think>` 写进 `content`（可折叠），同一文本仍发 `reasoning_content`。
  `max_tokens<=0`（webui 默认 `-1`）映射到 `serve_no_limit_tokens`（默认 1024）。
- **chat template**：实现的是 Qwen ChatML 家族子集（system/user/assistant/tool 文本消息、
  `<think>` 生成提示、`enable_thinking`、assistant 历史里的 `reasoning_content`）。
  不支持 tools / vision / 完整 jinja；模板不匹配时 `/v1/chat/completions` 返回 400 并提示改用
  `/v1/completions`（`/props` 的 `chat_template_format` 会是 `none`）。
- **分词器**：`gpt2`（BPE）与 `llama`（SPM）都支持；pre-tokenizer 的 `qwen35`/`qwen2` 是
  llama.cpp 精确端口，其他 pre 名回退 GPT-2 规则（见 `tools/verify_tokenizer.sh`）。
- **优雅退出**：Ctrl+C / Ctrl+Break / 关窗口 / `taskkill` 均走同一路径停止 accept 循环、
  释放 CUDA 显存；`POST /shutdown` 只在回环地址开放，用于脚本化验证。


## 8. 交互式启动（serve / run）

启动模型时如果 **stdin 是终端**，serve/run 会在加载权重前询问三项（全部有默认值，Enter 直过）：

1. **KV 量化类型** `f32 / q8_0 / q4_0`：先打印 `KV/token` 三档大小与各自可支持的最大 ctx
   （`ctx fit: f32 <= N | q8_0 <= M | q4_0 <= K`，由实测空闲显存 - 权重 - reserve 推导）。
   MLA 模型跳过并说明“MLA 保持 f32”。默认 `f32`（与历史默认一致；仅当 f32 连 4096 ctx
   都放不下而 q8_0 放得下时默认升级为 q8_0）。
2. **是否 CPU 专家卸载**（仅 MoE 模型）：默认为放置决策器的推荐（split → yes，gpu → no）。
   选 no 会强制全部专家进显存；若决策器认为放不下会打印“预期 OOM”的警告。
3. **ctx 长度**：`[Enter] = auto`（等价 `--ctx 0`：由装载期自适应推导并打印 `[ctx]` 账目）。
   输入超过推荐值时提示会被夹到上限；超过 `ctx_train` 时提示质量风险；非法输入回退 auto。

`run` 子命令额外询问：模型（未给路径时列出 `model_dir`/`<MODEL_DIR>` 下的 `.gguf`）与
生成 token 数 n。所有询问都在**模型加载之前**完成（只读 metadata），因为 ctx/kv_type 决定 KV 分配。

跳过与脚本化：

- 非 TTY（管道、服务管理器、CI）**零提示**，行为与旧版一致（`serve` 默认 ctx=4096）；
- `--no-interactive` / `--yes` / 配置文件 `interactive=false` / `assume_yes=true` / 环境变量
  `ENGINE_INTERACTIVE=0` / `ENGINE_ASSUME_YES=1` 都可跳过；
- 显式给出的 `--ctx`/`--serve-ctx`/`--kv-type`/`--gpu-layers`/`--n` 对应的项不再询问；
- `ENGINE_FORCE_INTERACTIVE=1` 是测试钩子：允许从管道驱动交互流程（正常产品路径不要设置）。

交互选择会通过 `Config::set(..., Source::Cli)` 写回统一配置，和命令行参数完全同源；
失败的解析一律回退默认值，不会崩溃。实测转录见 `FEATURE-MATRIX.md` §3.3。
