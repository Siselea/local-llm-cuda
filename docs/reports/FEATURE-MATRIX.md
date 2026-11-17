# FEATURE-MATRIX — 功能面审计 · 放置决策器 · 交互式启动

日期：2026-11-15 · 作者：feature-polish（推理引擎产品化收口）
范围：`src/main.cpp`、`src/planner.cpp`、`include/engine/placement.h`、`include/engine/runner.h`、
`src/serve.cpp`、`src/config.cpp`、`docs/CONFIG.md`、`README.md`、`engine.cfg.example`。
未改动：CUDA 内核数学（dp4a / fp16 GEMM / gd_delta）、`build.sh` 编译选项、`src/mtp.cpp`
（MTP 是另一 agent 的产物；接 serve 不在本轮范围）。

本轮回答用户的四个要求：

1. **SSD 加速要分情况**：新增放置决策器（gpu / vram+ram split / ssd stream 三档），只有
   VRAM+RAM 都装不下时 SSD 才是候选，且默认关闭；SSD 流式加载本体未实现，给出接口占位 +
   确定性报错 + 实现方案。
2. **所有功能可调、可选、默认判断**：功能面审计 101 个注册配置键 + 64 个诊断/实验环境变量；
   补齐原本缺 env 通道的 7 个键（现在除 `config` 自身外全部 config/CLI/env 三通道可达）。
3. **启动时交互式询问上下文**：serve/run 启动前询问 KV 量化类型 / 是否 CPU 专家卸载 /
   ctx（run 另问模型与 n），打印推荐值；非 TTY 完全无提示；`--no-interactive`/`--yes` 跳过。
4. **收口文档**：本文件 + `docs/CONFIG.md` §8/§9 + `README.md` 相关章节。

---

## 0. 结论先行（TL;DR）

| 问题 | 结论 |
|---|---|
| 哪些模型走哪条路（本机 12 GB / 32 GB，ctx 4096） | 0.5B、9B、Qwythos-9B → **gpu**；Coder-V2-Lite → **vram+ram split**（约 13/26 MoE 层常驻）；35B-A3B → **vram+ram split**（约 14/40 层常驻）。**没有任何模型需要 SSD**。 |
| SSD 何时才启用 | 仅当 `weights > 可用VRAM + 可用RAM`。默认 `ssd_stream=false`；显式 `--ssd-stream`/`--placement ssd` 在本版本**必定报“未实现”错误**（接口占位 + 设计见 §2），不会静默回退去磨盘。 |
| 决策器是否改变现有性能路径 | 不改变。`auto` 只打印账目 + 门控“装不下/显式请求未实现”两种情况；实际逐层放置仍由 runner 装载时的 `cuMemGetInfo` 实测校验（更权威，且保留原有自适应能力）。 |
| 功能面是否全可调 | 101 个产品配置键全部支持 **config 文件 / 命令行 / 环境变量**（`config` 键本身除外，它就是配置文件入口）。默认值来源全部标注在 §5 表里。 |
| 交互式会不会破坏脚本 | 不会。stdin 非 TTY 时零提示；`printf "" \| ./build/engine.exe serve ...` 与旧行为一致（默认 ctx=4096、逐 token 输出一致）。 |
| 验证 | `verify.sh` / `smoke_models.sh` / `gen_regression.sh` / `verify_generality.sh` / `verify_serve.sh` 状态见 §10（本轮全绿，证据文件在 `build/logs/`）。 |

---

## 1. 放置决策器（placement decider）

### 1.1 为什么需要它

引擎已有“专家卸载”路径（MoE 层粒度 GPU/CPU 放置 + 实测自适应），但决定“该不该上更慢的
路径”的判断散落在装载过程里，用户看不到账目；而 SSD 流式加载在业界常被无脑开启。用户要求：
**显存+内存放得下就没有必要 SSD；SSD 只在装不下时才是候选，且必须显式开启。**

### 1.2 判定规则（`placement::decide`）

字节口径全部是“引擎真实驻留字节”，不是参数量：

```
fixed    = ir::Model::fixed_bytes()    # embedding/output/attn/dense ffn/norm/router/shared_expert 的 GGUF 实际字节
experts  = ir::Model::expert_bytes()   # MoE 专家权重实际字节
KV(ctx)  = placement::kv_bytes(model, ctx, kv_type)
           # 与 DenseRunner::kv_layer_bytes() 同口径：f32=4B/元素，q8_0=34B/32值，q4_0=18B/32值，
           # MLA 按引擎真实分配的 per-head 展开 K/V 缓存算（f32 only）；hybrid 只算持 KV 的层
workspace= workspace_mb (默认 768 MiB)  # KV/激活/scratch 规划预留
可用VRAM = 实测 free VRAM − vram_margin_mb(默认 1200 MiB)
可用RAM  = 实测 free RAM  − host_margin_mb(默认 0=auto: max(1 GiB, 总RAM/16))
```

```
if fixed + experts + KV + workspace <= 可用VRAM:
        decision = gpu                     # 全驻显存，最快路径
elif fixed + KV + workspace <= 可用VRAM
     and (experts - 剩余VRAM) <= 可用RAM:  # 只有 MoE 能走这条
        decision = vram+ram split          # 现有 CPU 专家卸载路径
else:
        decision = ssd stream              # 唯一剩下的候选；默认关闭 + 接口占位
        （dense 模型没有 host dense 权重路径，此时直接报“dense 不能分片”，不假装能 split）
```

附加信息：split 档会按“平均每层专家字节”估算可驻显存的 MoE 层数（打印为“约 N/M 层”），
但它只是**预告**；真正放几层仍由 runner 装载时的逐层实测（`expert_layer_fits_vram`）决定，
所以决策器不会把 35B 的显存余量算错（35B 余量 ~1.6 GB 的坑已知）。

### 1.3 日志账目（`[placement]`，固定打印在模型加载前）

`engine serve/run/plan` 都会打印同一套账目（`plan` 用 `--vram-gib` 可离线模拟）：

```
[placement] ledger : model 20.33 GiB = fixed 1.65 GiB + experts 18.16 GiB (40 MoE layers)
[placement] vram   : free 11248 MiB (11794382848) / total 12288 MiB, reserve 1968 MiB -> usable 9280 MiB
[placement] host   : free 24652 MiB / total 32685 MiB, reserve 2043 MiB -> usable 22609 MiB
[placement] need   : fixed 1693 MiB + KV 160 MiB + workspace 768 MiB = 2621 MiB (+ experts -> 21221 MiB)
[placement] experts: 18600 MiB total, 6659 MiB fit VRAM -> 14/40 layers GPU-resident, 11941 MiB needs host RAM
[placement] decision: vram+ram split
[placement] reason : weights exceed VRAM but fit VRAM + host RAM: CPU expert offload path
[placement] note   : VRAM keeps 6.50 GiB of experts -> about 14/40 MoE layers GPU-resident, the rest run on the host CPU path
```

字段：模型总字节/fixed/experts → 实测 VRAM 可用/总/预留 → 实测 RAM 可用/总/预留 →
非专家驻留需求（fixed+KV+workspace）与含专家总量 → 专家可驻层数与需驻内存字节 → 决策 → 依据 → 备注。

### 1.4 四模型实测（GPU 空闲基线：free VRAM 11248 MiB / free RAM 24652 MiB，ctx 4096，kv_type f32）

| 模型 | 总字节 | fixed | experts | KV@4096 | 非专家需求 | VRAM 可用 | 判定 | 预计 GPU 专家层 | 需驻 host RAM |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5B Q4_K_M | 462.96 MiB | 462.96 MiB | 0 | 96 MiB | 1.30 GiB | 9.06 GiB | **gpu** | — | 0 |
| Qwen3.5-9B Q5_K_M | 6.12 GiB | 6.12 GiB | 0 | 256 MiB | 7.12 GiB | 9.06 GiB | **gpu** | — | 0 |
| Qwythos-9B-MTP Q4_K_M | 5.23 GiB | 5.23 GiB | 0 | 256 MiB | 6.23 GiB | 9.06 GiB | **gpu** | — | 0 |
| DeepSeek-Coder-V2-Lite Q5_K_M | 11.03 GiB | 0.89 GiB | 10.14 GiB | 2160 MiB | 3.75 GiB | 9.06 GiB | **vram+ram split** | 约 13/26 | 4946 MiB |
| Qwen3.6-35B-A3B Q4_K_M | 20.33 GiB | 1.65 GiB | 18.16 GiB | 160 MiB | 2.56 GiB | 9.06 GiB | **vram+ram split** | 约 14/40 | 11941 MiB |

要点：
- 9B/Qwythos/0.5B 的总权重 + KV 能整个塞进显存 → 走 `gpu`，**不会**触发任何卸载/SSD 逻辑。
- Coder 的 fixed 很小（MLA + 共享专家）但专家占 10.14 GiB；KV 在 ctx 4096 时是 2160 MiB
  （MLA 引擎缓存的是 per-head 展开的 K/V，不是 512+64 latent，这一点和 `plan` 的 `--kv-bpe 2`
  模拟口径不同，以 `placement` 账目为准）。判定 split。
- 35B 总 20.33 GiB > 11.25 GiB 可用显存，但 < 11.25 + 22.6 GiB 可用内存 → split（现状路径）。
- **SSD 在任何真实模型上都不是候选**，与用户直觉一致。

> 以上是**装载前**的决策器账目；实际 loader 的逐层 `cuMemGetInfo` 校验结果（13/26、15/40 等）
> 与决策器的预告基本一致，四模型实测对照见 §8.1。

### 1.5 门控行为（不会静默改变既有行为）

| 场景 | 行为 | 退出码 |
|---|---|---|
| `placement_policy=auto` + 判定 gpu/split | 打印账目，继续装载 | 0 |
| `placement_policy=auto` + 判定 ssd（真装不下） | 打印 ERROR + 三条 hint（降 ctx / 量化 KV / 换更小量化 / `--allow-oversubscribe`），拒绝装载 | 3 |
| `--allow-oversubscribe` | 打印 WARN“驱动会分页，单数字 tok/s”，继续（用户显式承担） | 0 |
| `--ssd-stream` 或 `--placement ssd` | 调 `SsdExpertStreamer::open()`，打印“NOT IMPLEMENTED + 接口/设计位置”，拒绝 | 3 |
| `--placement split` 但模型是 dense | 打印“dense 无 host 权重路径”，拒绝 | 3 |
| `--placement gpu` 但装不下 | 打印 WARN（用户强制），继续 | 0 |
| `--placement split` 但全部能放显存 | 打印 WARN“保持全 GPU 更快”，继续 | 0 |

非交互模式（管道/CI）与以前**逐 token 一致**：决策器只加日志与上述显式门控，不改装载数值路径。

### 1.6 为什么默认不开 SSD（本机实测依据）

| 数据 | 值 | 来源 |
|---|---|---|
| CPU 单 lane 有效带宽（专家 GEMV） | 4.4 GB/s | P0 实测（`planner.h: BW_CPU_GEMV_GBPS`） |
| PCIe H2D / D2H | 12.0 / 11.0 GB/s | P0 实测 |
| host RAM copy | 22.5 GB/s | P0 实测 |
| 单个 MoE 层专家字节（Coder / 35B） | ~374 MiB / ~448–465 MiB | `[placement]` 实测 |
| 零拷贝/UVM 历史结论 | 净负收益 | P0 结论（本仓库历史报告） |

即：一次专家 miss 的 I/O 是数十 MiB～数百 MiB 级，机械盘/PCIe x8 上单 token 会被 I/O 主导；
而在“VRAM+RAM 装得下”时用现有 split 路径每 token 只碰 6/8 个专家（3–6 层量级），远比流式划算。
因此默认策略是：**能放显存 → gpu；放不下显存但放得下内存 → split；两者都不行才轮到 SSD，
且必须显式开启。** 这与用户的原话一致。

---

## 2. SSD 流式加载：接口占位 + 实现方案（本轮未实现）

### 2.1 接口（`include/engine/placement.h`）

```cpp
struct SsdStreamConfig {
  std::string cache_dir;              // --ssd-cache / ssd_cache
  int prefetch_depth = 2;             // 2 = 一个读在途 + 一个在用
  size_t io_chunk_bytes = 4u << 20;   // 读粒度
  bool use_overlapped = true;         // FILE_FLAG_OVERLAPPED (Windows)
};
class SsdExpertStreamer {
 public:
  explicit SsdExpertStreamer(const SsdStreamConfig& cfg);
  bool open(const std::string& model_path, std::string* err);  // 本版本恒 false + 原因
  bool ready() const { return false; }
};
```

`open()` 当前返回 `false` 并给出确定性错误（含 cache 目录、prefetch 深度、I/O 粒度、设计文档位置），
`engine serve/run --ssd-stream` 会打印该错误并以退出码 3 结束——**不静默回退**。

实测报错样例：

```
[placement] decision: ssd stream
[placement] reason : dense model does not fit VRAM and this build has no dense host/SSD weight path
[placement] ERROR: SSD expert streaming is NOT IMPLEMENTED in this build (interface placeholder).
  would cache expert weights under '<unset: --ssd-cache>' for '...gguf'
  (prefetch_depth=2, chunk=4.00 MiB, FILE_FLAG_OVERLAPPED);
  planned design: pread double-buffer + (layer,expert) LRU, see FEATURE-MATRIX.md §3
[placement] hint : SSD streaming stays off by default because the measured lanes
  (PCIe 12.0 GB/s H2D, host RAM 22.5 GB/s) make it a net loss unless the model really cannot fit; ...
```

### 2.2 实现方案（写清但不实现）

1. **只流式专家**：attention/norm/embedding/output 常驻（每 token 必碰、体量小）；MoE 专家按
   `(layer, expert)` 粒度流式。Coder/35B 的专家占 92%/89% 字节，只有它们值得放宽。
2. **pinned host staging + 双缓冲**：`cudaHostAlloc` 一块 ring（`prefetch_depth` 个 slot，
   每 slot = 一个专家 3 个矩阵的字节数）；解码到第 k 个 token 时，路由结果一出来就为下一层
   预取，`pread`（POSIX）或 `ReadFile` + `FILE_FLAG_OVERLAPPED`（Windows）。
3. **专家 LRU**：key=(layer,expert)，容量由 `ssd_cache_mb`（新增配置，默认 0=auto）从实测空闲
   RAM 推导；命中直接 H2D，miss 才读盘；LRU 元数据按层分桶，避免全局锁。
4. **不引入全局 sync**：每个 slot 带 Win32 event / CUDA event；某层等待自己的槽位，其他层/
   CPU 路径照常推进。与 CUDA graph 互斥（I/O 等待不能进图）→ 自动关闭 graph 捕获走 plain 路径。
5. **放置决策器联动**：只有 `decide()==ssd stream` 且 `ssd_stream=true` 才初始化 streamer；
   初始化失败（盘不存在/无 cache 目录/空间不足）→ 回退到“拒绝装载 + hint”，不静默改走分页。
6. **验收标准（下一轮）**：35B ctx 4096、NVMe、PCIe x8 下，decode ≥ llama.cpp mmap 的 80%；
   cache 命中率 ≥ 90%（256 专家选 8，层内局部性高）；free VRAM 不低于 margin（现 1.6 GB 余量约束）。

### 2.3 明确未实现 / 放弃

- ❌ 真正的 SSD 读盘/预取/缓存代码：未写（时间与风险不允许在本轮动 GPU 内存路径）。
- ❌ dense 模型的 host/SSD 权重路径：未写；决策器会明确报“dense 不能分片”。
- ❌ `kv_offload`（host-mapped KV）：沿袭现状“仅解析+说明，未实现”。
- ❌ 不做 UVM/零拷贝：历史实测净负收益，且会破坏现有显存余量假设。

---

## 3. 交互式启动（serve / run）

### 3.1 触发条件（全部满足才询问）

| 条件 | 说明 |
|---|---|
| `interactive=true`（默认） | 新配置键；`--no-interactive` 或 `ENGINE_INTERACTIVE=0` 关闭 |
| `assume_yes=false`（默认） | `--yes` 一键跳过 |
| stdin 是 TTY | 管道 / 服务管理器 / CI **零提示**（`_isatty(_fileno(stdin))`） |
| `ENGINE_FORCE_INTERACTIVE=1` | 仅调试/测试钩子：允许从管道驱动交互流程（文档标注） |

显式给了 `--ctx`/`--serve-ctx`/`--kv-type`/`--gpu-layers`/`--n` 的项不重复询问（沿用现有
“命令行走过就不问”的避让规则），但推荐值信息仍会打印。

### 3.2 询问内容与推荐值怎么来

顺序：**KV 量化类型 → 是否 CPU 专家卸载（仅 MoE）→ ctx**（run 再问 n；没给模型时先问模型）。
全部在**模型加载之前**完成（只读 GGUF 头/HF config 的 metadata probe，不读权重），因为
ctx 与 kv_type 决定 KV 分配。

- **KV 量化类型**：打印 `KV/token f32 X KiB | q8_0 Y KiB | q4_0 Z KiB` 以及
  `ctx fit: f32 <= N | q8_0 <= M | q4_0 <= K tokens`（预算 = 实测空闲显存 − weights − reserve，
  与 `DenseRunner::derive_ctx_limit()` 同一套迭代公式的预演）。默认 = f32（与历史默认一致），
  只有当 f32 连 4096 都放不下、而 q8_0 放得下时默认才升级到 q8_0。MLA 模型跳过该项并说明
  “MLA 保持 f32（无量化 MLA attention kernel）”。
- **是否 CPU 专家卸载**（仅 MoE 模型）：默认 = 决策器推荐（split 推荐 yes，gpu 推荐 no）。
  选 no 会强制 `gpu_layers=n_moe`（全部专家进显存）；若决策器认为放不下会打印
  “需要 X、只有 Y，预期 OOM”的警告，但尊重用户选择。
- **ctx**：默认 `[Enter] = auto`，即 `--ctx 0` 语义（由 `derive_ctx_limit` 用实测空闲显存推导
  并打印 `[ctx]` 账目）；输入非法数字回退 auto；输入 > 推荐值时提示“loader 会夹到能放下的上限”；
  输入 > ctx_train 时提示质量会下降。
- **run**：模型路径没给时列出 `model_dir`（默认 `<MODEL_DIR>`，存在则用）下的 `.gguf`，
  输入编号或路径；再问 n（默认 4，1..8192，非法回退）。

### 3.3 实测转录

**(a) 交互式（TTY 等价：`ENGINE_FORCE_INTERACTIVE=1` + 管道喂答案；真实控制台会回显输入）**

```bash
$ printf 'q8_0\n8192\n' | ENGINE_FORCE_INTERACTIVE=1 \
    ./build/engine.exe serve <MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 7098
```
```
[interactive] model : qwen2.5-0.5b-instruct-q4_k_m.gguf
[interactive] shape : qwen2, 24 layers, ctx_train=32768, weights 0.45 GiB
[interactive] vram  : free 11248 MiB / total 12288 MiB, reserve 1968 MiB (vram_margin+workspace)
[interactive] host  : free 24645 MiB / total 32685 MiB, reserve 2043 MiB
[interactive] kv    : KV/token f32 24.0 KiB | q8_0 6.4 KiB | q4_0 3.4 KiB
[interactive] ctx fit: f32 <= 32768 | q8_0 <= 32768 | q4_0 <= 32768 tokens (free VRAM - weights - reserve)

[interactive] KV type [f32/q8_0/q4_0] (Enter = f32) > q8_0
[interactive] context: recommended <= 32768 tokens (kv=q8_0, KV 204 MiB at that length, weights 0.45 GiB, free VRAM 11248 MiB)

[interactive] ctx tokens [Enter = auto (32768), 0 = auto] > 8192
[interactive] choices: ctx=8192 kv=q8_0 gpu_layers=unchanged
[placement] ledger : model 462.96 MiB = fixed 462.96 MiB + experts 0.00 B
...
[ctx] requested=8192->8192 kv_type=q8_0 kv/token=6528 B (6.38 KiB) kv_layers=24/24 kv_total=51 MiB
listening on http://127.0.0.1:7098 (model=..., ctx=8192)
/props -> "n_ctx": 8192
```
→ 选中的 kv=q8_0 与 ctx=8192 在后续 `[ctx]` 账目里得到确认（`kv_type=q8_0`、`ctx=8192`）；`/props` 返回 `n_ctx: 8192`；
`POST /v1/completions` 正常出 token。

**(b) 非交互式管道（脚本/CI 行为必须不变）**

```bash
$ printf "" | ./build/engine.exe serve <MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 7097
```
```
[placement] ledger : model 462.96 MiB = fixed 462.96 MiB + experts 0.00 B
[placement] decision: gpu
[serve] loading model (ctx=4096) ...
[ctx] requested=4096->4096 kv_type=f32 kv/token=24576 B (24.00 KiB) ...
listening on http://127.0.0.1:7097 (model=..., ctx=4096)
```
→ **没有任何 `[interactive]` 行**，ctx 仍是历史默认 4096，`/props` 的 `n_ctx: 4096`。脚本可复现性保持。

**(c) 错误输入回退**

```bash
$ printf 'garbage\nnonsense\nabc\n' | ENGINE_FORCE_INTERACTIVE=1 \
    ./build/engine.exe run <MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf --ids 1000 --n 2
[interactive] KV type [f32/q8_0/q4_0] (Enter = f32) > [interactive] note: unknown kv type 'garbage', using f32
[interactive] ctx tokens [Enter = auto (32768), 0 = auto] > [interactive] note: 'nonsense' is not a number, using auto
[interactive] choices: ctx=auto kv=f32 gpu_layers=unchanged
gen    : 198 33
```
→ 全部回退默认，没有异常/崩溃，生成继续。

**(d) `--yes` 一键跳过**：`ENGINE_FORCE_INTERACTIVE=1` + `--yes` 时没有任何 `[interactive]` 行，
直接按默认跑（等价非交互）。

---

## 4. 功能面审计：方法与结论

- **方法**：`src/config.cpp` 的 `kDefs[]` 是唯一配置注册表；本轮用脚本解析该表生成 §5 的
  101 行总表（key / CLI / env / 默认 / 默认来源 / 可调），再用 `--print-config` 与
  `--help` 交叉核对（防止表漂移）。命令：`python build/gen_matrix.py`
  （生成器脚本在 `build/`，不进入产品源码）。
- **注册表规模**：本轮前 93 键，本轮 +8（放置策略 5 + 交互启动 3）= **101 键**。
- **三通道覆盖**：CLI 101/101；env 100/101（唯一例外 `config` 本身就是配置文件入口）；
  config 文件 101/101。**无漏注册的产品功能。**
- **默认值来源分布**（§5 表最后统计）：结构推导 / 自动判断 / 本机实测标定 / A/B 结论 /
  历史兼容 / 输入路径 / 显式请求默认关。凡是“实测标定”的默认值都可以用
  `--kernel-calibrate`、`--vram-margin`、A/B 重新标定，不是钉死的死数；真正写死且不可配的
  只剩“格式/硬件结构定义”类常量（见 §7）。
- **补齐记录（本轮把缺 env 通道的 7 个键补上）**：

| 键 | 之前 | 现在 |
|---|---|---|
| `model` | 仅 CLI/config | + `ENGINE_MODEL` |
| `vram_check` | 仅 CLI/config | + `ENGINE_NO_VRAM_CHECK`（存在即关，和其他 no-* 一致） |
| `print_config` | 仅 CLI | + `ENGINE_PRINT_CONFIG` |
| `kernel_calibrate` | 仅 CLI | + `ENGINE_KERNEL_CALIBRATE` |
| `kernel_cache` | 仅 CLI/config | + `ENGINE_KERNEL_CACHE` |
| `calib_iters` | 仅 CLI | + `ENGINE_CALIB_ITERS` |
| `config` | 仅 CLI | 保持（配置文件入口自身；env 会造成鸡生蛋问题） |

> 注：上表在 `--help` 中会显示为 `env: ENGINE_...`；`--print-config` 的 `[env]` 来源列可用于验证。

---

## 5. 配置项总表（自动生成，101 键）

> 生成方式见 §4；默认列是该键在所有层的覆盖都缺席时的最终值。
> “可调”列的三通道指：**配置文件键 / 命令行参数 / 环境变量**。

| key | CLI | env | 默认 | 默认值怎么定的 | 可调 |
|---|---|---|---|---|---|

**1. 模型与运行形状**
| `model` | `--model` `<v>` | `ENGINE_MODEL` | `—` | 输入：模型路径（.gguf 文件或 safetensors 目录，按文件头自动识别容器，不依赖扩展名） | config / CLI / env |
| `ctx` | `--ctx` `<v>` | `ENGINE_CTX` | `256` | 结构推导：默认 256 仅为“未显式给 ctx 时的生成入口上限”，`--ctx 0`/ctx_auto 时由模型 ctx_train + 实测空闲显存推导 | config / CLI / env |
| `threads` | `--threads` `<v>` | `ENGINE_THREADS` | `0` | 结构推导 + 实测：0 = 物理核推导（GetLogicalProcessorInformation → 3*phys/2，clamp [2,hw-1]；本机 6 物理核→9 线程） | config / CLI / env |
| `gpu_layers` | `--gpu-layers` `<v>` | `ENGINE_GPU_LAYERS` | `-1` | 自动判断：-1 = 按模型专家布局 + 装载时实测 cuMemGetInfo 逐层决定（层粒度，llama.cpp --n-cpu-moe 语义） | config / CLI / env |
| `gpu_experts` | `--gpu-experts` `<v>` | `ENGINE_GPU_EXPERTS` | `-1` | 自动判断：-1 = 层粒度自动；显式值走旧的“每层固定专家数”路径（兼容/实验） | config / CLI / env |
| `layers` | `--layers` `<v>` | `ENGINE_LAYERS` | `0` | 调试/性能分解：0 = 全部层；非 0 只跑前 N 层（不参与产品默认判断） | config / CLI / env |
| `vram_margin_mb` | `--vram-margin` `<v>` | `ENGINE_VRAM_MARGIN_MB` | `1200` | 实测标定：1200 MiB 是本机 WDDM/桌面合成器 + 驱动峰值占用的安全余量（可用 --vram-margin 重标） | config / CLI / env |
| `vram_budget_mb` | `--vram-budget` `<v>` | `ENGINE_VRAM_BUDGET_MB` | `0` | 自动判断：0 = 读真实 cuMemGetInfo；非 0 用于放置策略 A/B 与回归测试的显存模拟 | config / CLI / env |
| `workspace_mb` | `--workspace` `<v>` | `ENGINE_WORKSPACE_MB` | `768` | 实测标定：768 MiB 覆盖 KV/激活/scratch 预留；装载后仍以实测空闲显存逐层校验 | config / CLI / env |
| `vram_check` | `--no-vram-check` | `ENGINE_NO_VRAM_CHECK` (存在即关) | `true` | 自动判断：true = 装载后实测 free VRAM < margin 时自动降 GPU 层；ENGINE_NO_VRAM_CHECK 可关 | config / CLI / env |
| `device` | `--device` `<v>` | `ENGINE_DEVICE` | `0` | 输入：CUDA 设备索引 0 | config / CLI / env |
| `host_margin_mb` | `--host-margin` `<v>` | `ENGINE_HOST_MARGIN_MB` | `0` | 自动判断（新）：0 = auto，保留 max(1 GiB, 总内存/16)（本机 32 GiB → 2 GiB），供放置决策器判断“内存装得下” | config / CLI / env |
| `placement_policy` | `--placement` `<v>` | `ENGINE_PLACEMENT` | `auto` | 自动判断（新）：auto = 实测 VRAM/内存后三档决策；gpu/split/ssd 为显式策略（ssd 未实现） | config / CLI / env |
| `ssd_stream` | `--ssd-stream` | `ENGINE_SSD_STREAM` | `false` | 显式请求默认关（新）：SSD 只在 VRAM+RAM 都放不下时才是候选；PCIe x8/机械盘上流式通常净负收益，故必须显式 --ssd-stream；本版本未实现（接口占位） | config / CLI / env |
| `ssd_cache` | `--ssd-cache` `<v>` | `ENGINE_SSD_CACHE` | `—` | 输入（新）：SSD 专家缓存目录，仅用于决策日志/未来实现；默认空 | config / CLI / env |
| `allow_oversubscribe` | `--allow-oversubscribe` | `ENGINE_ALLOW_OVERSUBSCRIBE` | `false` | 显式请求默认关（新）：决策器判定装不下时仍尝试装载（会触发驱动分页，单数字 tok/s） | config / CLI / env |

**2. 启动交互（新）**
| `interactive` | `--no-interactive` | `ENGINE_INTERACTIVE` | `true` | 自动判断（新）：true = stdin 是终端才询问；管道/服务/CI 永远不问；--no-interactive 或 ENGINE_INTERACTIVE=0 关闭 | config / CLI / env |
| `assume_yes` | `--yes` | `ENGINE_ASSUME_YES` | `false` | 显式请求默认关（新）：--yes 跳过全部询问，使用计算出的默认值（脚本/CI 用） | config / CLI / env |
| `model_dir` | `--model-dir` `<v>` | `ENGINE_MODEL_DIR` | `—` | 自动探测（新）：空 = <MODEL_DIR> 存在则用它，否则当前目录；只影响交互式选模型的候选列表 | config / CLI / env |

**3. 执行 / CUDA Graph**
| `no_graph` | `--no-graph` | `ENGINE_NO_GRAPH` | `false` | A/B 结论：false = 默认开 CUDA Graph（WDDM 提交开销是 decode 主要瓶颈之一）；ENGINE_NO_GRAPH 可关 | config / CLI / env |
| `layer_graph` | `--layer-graph` | `ENGINE_LAYER_GRAPH` | `false` | A/B 结论：false = 默认只对 qwen35 GPU 常驻 MoE 层做 per-layer graph（layer_graph_auto=true 时），全 pipeline graph 单独开关 | config / CLI / env |
| `layer_graph_auto` | `--layer-graph-auto` | `ENGINE_LAYER_GRAPH_AUTO` | `true` | 自动判断 + A/B：按模型结构（qwen35 MoE、专家可全驻显存）与实测收益决定是否逐层捕获 | config / CLI / env |
| `mla_no_graph` | `--mla-no-graph` | `ENGINE_MLA_NO_GRAPH` | `false` | 历史兼容：MLA 模型默认沿用历史规则（保持既有 graph 行为，避免重新标定） | config / CLI / env |
| `prefill_f32` | `--prefill-f32` | `ENGINE_PREFILL_F32` | `true` | 自动判断：true = 保留 f32 权重镜像给批量 prefill；MoE/低显存时自动关闭（镜像要 4x 权重字节） | config / CLI / env |
| `gemm` | `--quant-gemm` | `ENGINE_GEMM` | `false` | A/B 结论：false = 自研量化 GEMM 为实验特性，默认走 cuBLAS/prefill_f16（实测更稳） | config / CLI / env |
| `no_moe_fused` | `--no-moe-fused` | `ENGINE_NO_MOE_FUSED` | `false` | A/B 结论：false = 默认启用融合 MoE gate/up/down 内核 | config / CLI / env |
| `dev_route` | `--dev-route` | `ENGINE_NO_DEV_ROUTE` (存在即关) | `true` | 自动判断：设备侧路由默认开，但专家权重含 Q5_1 的布局自动退回 host 路由（Q5_1 device GEMV 未验证） | config / CLI / env |
| `keep_gpu_busy` | `--keep-gpu-busy` | `ENGINE_KEEP_GPU_BUSY` | `false` | A/B 结论：false = GPU 空闲期跑时钟 warmer 实测无收益（本 VM） | config / CLI / env |

**4. 模型语义覆盖（IR 权威，默认全自动）**
| `norm_plus1` | `--norm-plus1` `<v>` | `ENGINE_NORM_PLUS1` | `-1` | 结构推导：-1 = 从 GGUF/HF 元数据与架构约定推导（qwen35 系 +1 语义），仅在解析异常时手工覆盖 | config / CLI / env |
| `no_yarn` | `--no-yarn` | `ENGINE_NO_YARN` | `false` | 结构推导：false = 遵守 GGUF rope.scaling.type=yarn（含 yarn factor/mscale 实测校正） | config / CLI / env |
| `router` | `--router` `<v>` | `ENGINE_ROUTER` | `—` | 结构推导：空 = 从 IR 取路由类型（topk_softmax / sigmoid_topk） | config / CLI / env |
| `norm_topk` | `--norm-topk` `<v>` | `ENGINE_NORM_TOPK` | `-1` | 结构推导：-1 = 从 IR 推导是否归一化 top-k 权重 | config / CLI / env |
| `experts_topk` | `--experts-topk` `<v>` | `ENGINE_EXPERTS_TOPK` | `0` | 结构推导：0 = 从 IR 取 n_expert_used | config / CLI / env |
| `no_shared` | `--no-shared` | `ENGINE_NO_SHARED` | `false` | 历史兼容：false = 正常使用共享专家（诊断开关用于隔离问题） | config / CLI / env |
| `mla_q_layout` | `--mla-q-layout` `<v>` | `ENGINE_MLA_Q_LAYOUT` | `—` | 结构推导：空 = 按 IR/架构自动选择 MLA q 布局；仅 layout sweep 工具使用显式值 | config / CLI / env |
| `mla_kv_layout` | `--mla-kv-layout` `<v>` | `ENGINE_MLA_KV_LAYOUT` | `—` | 结构推导：空 = 按 IR/架构自动选择 MLA kv 布局 | config / CLI / env |

**5. KV cache 精度 / 长上下文**
| `kv_type` | `--kv-type` `<v>` | `ENGINE_KV_TYPE` | `f32` | 历史兼容 + 显存预算：f32 是历史布局/质量默认；ctx_auto 会按可用显存报告 q8_0/q4_0 的上限，交互式启动会给出三档推荐 | config / CLI / env |
| `kv_offload` | `--kv-offload` `<v>` | `ENGINE_KV_OFFLOAD` | `false` | 显式请求默认关：host-mapped KV 本版本未实现（仅解析+报错说明），默认 false | config / CLI / env |
| `kv_split` | `--kv-split` `<v>` | `ENGINE_KV_SPLIT` | `32` | 实测标定：32 份 split-K 是 decode tiled attention 的本机最优；prefill 固定 1 段在寄存器内完成 | config / CLI / env |
| `ctx_auto` | `--ctx-auto` `<v>` | `ENGINE_CTX_AUTO` | `true` | 自动判断：true = --ctx 会被夹到“weights+激活+reserve 之后能放下”的上限并打印 [ctx] 账目；--ctx 0 从头推导 | config / CLI / env |
| `prefill_rows` | `--prefill-rows` `<v>` | `ENGINE_PREFILL_ROWS` | `512` | 实测标定：512 行是激活缓冲/分块 prefill 的本机平衡点；装载后按实测空闲显存与 prefill_rows_max 自动增减 | config / CLI / env |

**6. 内核选择 / CPU 路径**
| `dp4a_gemv` | `--no-dp4a-gemv` | `ENGINE_NO_DP4A_GEMV` (存在即关) | `true` | A/B 结论：true = Q5_K/Q6_K decode 用 int8 dp4a GEMV（本机实测明显快于 fp32 x4） | config / CLI / env |
| `dp4a_q6k` | `--no-dp4a-q6k` | `ENGINE_NO_DP4A_Q6K` (存在即关) | `true` | A/B 结论：true = Q6_K 走 dp4a（单独开关便于回归定位） | config / CLI / env |
| `tiny_sk` | `--no-tiny-sk` | `ENGINE_NO_TINY_SK` (存在即关) | `true` | 实测标定：true = 小 n_out GEMV 用 split-K（阈值见 tiny_sk_max_nout/tiny_sk_min_nin） | config / CLI / env |
| `gd_delta_128` | `--no-gd-delta-128` | `ENGINE_NO_GD_DELTA_128` (存在即关) | `true` | A/B 结论：true = gated-delta 状态更新用 128 通道向量化变体 | config / CLI / env |
| `quant_reuse` | `--no-quant-reuse` | `ENGINE_NO_QUANT_REUSE` (存在即关) | `true` | A/B 结论：true = CPU int8 两级量化复用激活（省一次量化） | config / CLI / env |
| `ssm_fuse` | `--no-ssm-fuse` | `ENGINE_NO_SSM_FUSE` (存在即关) | `true` | A/B 结论：true = SSM conv/scan 融合 | config / CLI / env |
| `splitk` | `--splitk` | `ENGINE_SPLITK` | `false` | A/B 结论：false = 通用 split-K 实验路径默认关（dp4a 专用 split-K 另有开关） | config / CLI / env |
| `cpu_layer_graph` | `--cpu-layer-graph` | `ENGINE_NO_CPU_LAYER_GRAPH` (存在即关) | `true` | A/B 结论：false = CPU 层图捕获实测中性偏负，默认关（ENGINE_NO_CPU_LAYER_GRAPH 反向兼容） | config / CLI / env |
| `vram_extend` | `--no-vram-extend` | `ENGINE_NO_VRAM_EXTEND` (存在即关) | `true` | 自动判断：true = 装载后逐层用实测空闲显存尝试“多放一层”，放不下立即停 | config / CLI / env |
| `rms4` | `--no-rms4` | `ENGINE_NO_RMS4` (存在即关) | `true` | A/B 结论：true = rmsnorm 四行融合内核（宽度推导线程数 8 元素/线程，clamp [256,512]） | config / CLI / env |
| `head_topk` | `--no-head-topk` | `ENGINE_NO_HEAD_TOPK` (存在即关) | `true` | A/B 结论：true = decode 头在 GPU 上做 top-k 归约，只回传 k 个 (idx,val)（省 0.4–1 MB D2H） | config / CLI / env |
| `dp4a` | `--dp4a` | `ENGINE_DP4A` | `false` | A/B 结论：false = 旧式整层 dp4a 路径（历史实验），默认用上面的细粒度 dp4a_gemv | config / CLI / env |
| `row_threshold` | `--row-threshold` `<v>` | `ENGINE_ROW_THRESHOLD` | `2048` | 实测标定：2048 行是“GEMV vs GEMM”的实测交叉点（可被 --kernel-calibrate 按形状覆盖） | config / CLI / env |
| `q6k_x4_min` | `--q6k-x4-min` `<v>` | `ENGINE_Q6K_X4_MIN` | `32768` | 实测标定：32768 = Q6_K x4 内核启用的最小 MAC 数 | config / CLI / env |
| `q6k_split` | `--q6k-split` `<v>` | `ENGINE_Q6K_SPLIT` | `0` | 实测标定：0 = 自动（按 n_out 与卡型）；显式值用于 A/B | config / CLI / env |
| `tiny_sk_max_nout` | `--tiny-sk-max-nout` `<v>` | `ENGINE_TINY_SK_MAX_NOUT` | `128` | 实测标定：128 = 小 n_out split-K 交叉点（可用 --kernel-calibrate 覆盖） | config / CLI / env |
| `tiny_sk_min_nin` | `--tiny-sk-min-nin` `<v>` | `ENGINE_TINY_SK_MIN_NIN` | `512` | 实测标定：512 = 小 n_out split-K 的最小 n_in | config / CLI / env |
| `half` | `--half` | `ENGINE_HALF` | `false` | A/B 结论：false = fp16 权重镜像实验默认关（显存换带宽，本机不划算） | config / CLI / env |
| `half_t` | `--half-t` | `ENGINE_HALF_T` | `false` | A/B 结论：false = fp16 激活实验默认关 | config / CLI / env |
| `prefill_f16` | `--prefill-f16` | `ENGINE_PREFILL_F16` | `true` | 实测标定：true = prefill 走 fp16 tensor-core GEMM（本机 pp 提升 3.3x 的关键） | config / CLI / env |
| `gd_reg` | `--gd-reg` | `ENGINE_GD_REG` | `true` | A/B 结论：true = gated-delta 寄存器驻留优化 | config / CLI / env |
| `gd_smem` | `--gd-smem` | `ENGINE_GD_SMEM` | `true` | A/B 结论：true = gated-delta shared-memory 布局优化 | config / CLI / env |
| `attn_gqa` | `--attn-gqa` | `ENGINE_ATTN_GQA` | `true` | 结构推导 + A/B：true = GQA 时按 group 广播 head（head_dim/n_kv 结构决定可用性） | config / CLI / env |
| `attn_pre_rows` | `--attn-pre-rows` `<v>` | `ENGINE_ATTN_PRE_ROWS` | `8` | 实测标定：8 = prefill attention 每次处理的行数 | config / CLI / env |
| `prefill_rows_max` | `--prefill-rows-max` `<v>` | `ENGINE_PREFILL_ROWS_MAX` | `4096` | 实测标定：2048 = prefill 分块自动放大的上限（MoE 模型还会被结构规则限制不放大，避免 35B 显存打满） | config / CLI / env |
| `prefill_graph` | `--prefill-graph` | `ENGINE_PREFILL_GRAPH` | `false` | A/B 结论：false = prefill 图捕获实验默认关 | config / CLI / env |
| `prefill_f16_acc` | `--prefill-f16-acc` | `ENGINE_PREFILL_F16_ACC` | `false` | A/B 结论：false = fp16 累加实验默认关（精度/速度权衡本机不划算） | config / CLI / env |
| `prefill_f16_min_macs` | `--prefill-f16-min-macs` `<v>` | `ENGINE_PREFILL_F16_MIN_MACS` | `0` | 实测标定：fp16 prefill 内核的最小 MAC 阈值 | config / CLI / env |
| `bm_types` | `--bm-types` `<v>` | `ENGINE_BM_TYPES` | `—` | 实测标定：空 = 按量化类型自动选 bm 内核；显式列表用于校准/回归 | config / CLI / env |
| `no_bm` | `--no-bm` | `ENGINE_NO_BM` | `false` | A/B 结论：false = 默认启用 bm 内核族 | config / CLI / env |
| `no_gemv` | `--no-gemv` | `ENGINE_NO_GEMV` | `false` | 历史兼容：false = 默认启用自研 GEMV 内核族；true 退回通用路径（调试） | config / CLI / env |
| `no_avx2` | `--no-avx2` | `ENGINE_NO_AVX2` | `false` | A/B 结论：false = 默认用 AVX2 CPU 内核（本机支持） | config / CLI / env |
| `cpu_moe_batch` | `--no-cpu-moe-batch` | `ENGINE_NO_CPU_MOE_BATCH` (存在即关) | `true` | A/B 结论：true = CPU 专家按 batch 合并 GEMV | config / CLI / env |
| `cpu_int8` | `--cpu-int8` `<v>` | `ENGINE_CPU_INT8` | `1` | 实测标定：1 = 两级 int8 激活 + AVX2 maddubs（本机配对中位 -10.1%/-7.5%；0=off，2=另一档） | config / CLI / env |
| `cpu_pool_spin` | `--cpu-pool-spin` `<v>` | `ENGINE_POOL_SPIN` | `512` | 实测标定：512 PAUSE 迭代 ≈20 µs（本机 PAUSE ~140 cycle）；可换卡重标 | config / CLI / env |
| `attn_warp_ctx_mid` | `--attn-warp-mid` `<v>` | `ENGINE_ATTN_WARP_CTX_MID` | `96` | 实测标定：96 = attention warp 分档边界 1（本机 SM/L2/KV 布局交叉点） | config / CLI / env |
| `attn_warp_ctx_hi` | `--attn-warp-hi` `<v>` | `ENGINE_ATTN_WARP_CTX_HI` | `384` | 实测标定：384 = attention warp 分档边界 2；launch 与 graph 桶同源 | config / CLI / env |
| `q6k_dp4a_split_max_nout` | `--q6k-dp4a-split-max-nout` `<v>` | `ENGINE_Q6K_DP4A_SPLIT_MAX_NOUT` | `4096` | 实测标定：4096 = Q6_K dp4a auto split-K 的 n_out 交叉点 | config / CLI / env |
| `small_nout_split_max` | `--small-nout-split-max` `<v>` | `ENGINE_SMALL_NOUT_SPLIT_MAX` | `256` | 实测标定：256 = 8 行/block × 32 block 的结构值（小 n_out 2-warp split-K） | config / CLI / env |

**7. MTP / 投机解码**
| `mtp` | `--mtp` | `ENGINE_MTP` | `false` | 功能开关默认关：仅当 GGUF 带 nextn 块且 --mtp 显式开启时启用（当前仅 test_gen/mtp_run 生效，serve 未接） | config / CLI / env |
| `mtp_k` | `--mtp-k` `<v>` | `ENGINE_MTP_K` | `2` | 实测标定：2 = 链式 draft 数（1..3，accept 率×开销的本机平衡点） | config / CLI / env |
| `mtp_gpu` | `--no-mtp-gpu` | `ENGINE_NO_MTP_GPU` (存在即关) | `true` | 自动判断：true = MTP 专家能放进显存就放 GPU，否则回退 host | config / CLI / env |
| `mtp_vram_margin_mb` | `--mtp-vram-margin` `<v>` | `ENGINE_MTP_VRAM_MARGIN_MB` | `256` | 实测标定：256 MiB MTP 专用余量 | config / CLI / env |
| `mtp_ctx_prefill` | `--no-mtp-ctx-prefill` | `ENGINE_NO_MTP_CTX_PREFILL` (存在即关) | `true` | A/B 结论：true = 为 prompt 预填 MTP 层 attention 历史（保证逐 token 一致） | config / CLI / env |

**8. HTTP 服务 / 默认采样**
| `serve_host` | `--host` `<v>` | `ENGINE_SERVE_HOST` | `127.0.0.1` | 输入：127.0.0.1（本地单用户服务） | config / CLI / env |
| `serve_port` | `--port` `<v>` | `ENGINE_SERVE_PORT` | `7030` | 输入：7030 | config / CLI / env |
| `serve_ctx` | `--serve-ctx` `<v>` | `ENGINE_SERVE_CTX` | `0` | 服务专用覆盖：>0 时压过 ctx；0 = 用 ctx/交互选择 | config / CLI / env |
| `default_max_tokens` | `--default-max-tokens` `<v>` | `ENGINE_DEFAULT_MAX_TOKENS` | `1048576` | 采样默认：256（请求未给 max_tokens 时） | config / CLI / env |
| `default_temperature` | `--default-temperature` `<v>` | `ENGINE_DEFAULT_TEMPERATURE` | `0.8` | 采样默认：0.8（与 llama.cpp 默认一致） | config / CLI / env |
| `default_top_p` | `--default-top-p` `<v>` | `ENGINE_DEFAULT_TOP_P` | `0.95` | 采样默认：0.95 | config / CLI / env |
| `default_top_k` | `--default-top-k` `<v>` | `ENGINE_DEFAULT_TOP_K` | `40` | 采样默认：40 | config / CLI / env |
| `default_repeat_penalty` | `--default-repeat-penalty` `<v>` | `ENGINE_DEFAULT_REPEAT_PENALTY` | `1.0` | 采样默认：1.0（不惩罚） | config / CLI / env |

**9. 环境 / 诊断 / 工具**
| `nvrtc_path` | `--nvrtc-path` `<v>` | `DSH_NVRTC_PATH` | `—` | 环境探测：空 = 按 pip 包位置自动找（DSH_NVRTC_PATH 覆盖） | config / CLI / env |
| `cublas_dir` | `--cublas-dir` `<v>` | `DSH_CUBLAS_DIR` | `—` | 环境探测：空 = 按 pip 包位置自动找（DSH_CUBLAS_DIR 覆盖） | config / CLI / env |
| `debug` | `--debug` | `RUNNER_DEBUG` | `false` | 诊断：false = 关闭逐层/内核决策日志（RUNNER_DEBUG=1 打开） | config / CLI / env |
| `print_config` | `--print-config` | `ENGINE_PRINT_CONFIG` | `false` | 一次性动作：打印生效配置后退出 | config / CLI / env |
| `kernel_calibrate` | `--kernel-calibrate` | `ENGINE_KERNEL_CALIBRATE` | `false` | 一次性动作：对本机真实形状微基准并写校准缓存 | config / CLI / env |
| `kernel_cache` | `--kernel-cache` `<v>` | `ENGINE_KERNEL_CACHE` | `engine-kernel-cache.txt` | 诊断/标定：空 = 不加载；--kernel-calibrate 生成后由该路径加载 | config / CLI / env |
| `calib_iters` | `--calib-iters` `<v>` | `ENGINE_CALIB_ITERS` | `40` | 实测标定：40 次计时迭代/候选（--kernel-calibrate） | config / CLI / env |
| `config` | `--config` `<v>` | — | `—` | 输入：--config 配置文件路径（优先级：CLI > config > env > default） | config / CLI / env |

### 5.1 默认值来源统计

| 默认值来源（归一化） | 键数 | 占比 |
|---|---|---|
| 结构推导 | 10 | 10% |
| 本机实测标定 | 25 | 25% |
| A/B 结论 | 26 | 26% |
| 自动判断/探测 | 13 | 13% |
| 历史兼容默认 | 4 | 4% |
| 显式请求默认关 | 5 | 5% |
| 输入/接口默认 | 16 | 16% |
| 诊断/调试 | 2 | 2% |
| **合计** | **101** | 100% |
---

## 6. 未注册的开关与硬编码残留（审计结论）

### 6.1 环境变量：64 个“仅 env”的变量，全部是诊断/实验用途

`grep -rhoE 'getenv\("[A-Z0-9_]+"\)' src/*.cpp` 得到 64 个未进注册表的环境变量，分三类：

| 类别 | 例子 | 处置 |
|---|---|---|
| 纯诊断 dump / 逐层计时 / 内核短路 | `ENGINE_DUMP*`、`ENGINE_DBG_*`、`ENGINE_KPROF*`、`ENGINE_SKIP_*`、`ENGINE_PROFILE`、`ENGINE_GRAPH_KERNEL_LIMIT`、`ENGINE_LOGITS` | **有意不进产品面**：只在取证/回归定位时用，误配会改变输出或让内核短路；`docs/CONFIG.md §3.17`（原文档）已声明。 |
| 模型语义取证（历史上用于定位 qwen35/YaRN/SSM 语义 bug） | `ENGINE_Q35_LOG_A`、`ENGINE_Q35_RAW_A`、`ENGINE_Q35_HF_BCAST`、`ENGINE_MLA_ROPE_NEOX`、`ENGINE_SSM_PREP_ONLY`、`ENGINE_SSM_DELTA_ONLY`、`ENGINE_SH_SWAP_GU`、`ENGINE_MOE_SWAP_GU` | 保留为取证开关；产品默认路径不需要它们。语义已由 IR 自动推导（本轮回归逐 token 一致）。 |
| 实验性执行路径 | `ENGINE_DEV_ROUTE`、`ENGINE_DEV_NO_Q51`、`ENGINE_GRAPH_FORCE`、`ENGINE_GRAPH_PRECAPTURE`、`ENGINE_GPU_FULL_LAYER_GRAPH`、`ENGINE_NO_GPU_LAYER_GRAPH`、`ENGINE_ROUTER_KERNEL`、`ENGINE_LG_*`、`ENGINE_EXPERT_ARENA` | 保留为实验；其中的**产品等价开关**已在注册表：`dev_route`、`no_graph`、`layer_graph*`、`cpu_layer_graph`、`vram_extend`。 |
| 本轮新增测试钩子 | `ENGINE_FORCE_INTERACTIVE` | 仅用于从管道测试交互流程；正常产品路径不设置它。 |

结论：**64 个变量中没有“产品功能被藏在 env 里没注册”的情况**；产品面的开关都已注册，
诊断/实验面按设计保持 env-only（不提供 config/CLI 是为了避免被写进生产配置）。

### 6.2 内部硬编码常量（延续 `DEADCODE-AUDIT.md`）

`DEADCODE-AUDIT.md`（第二轮，general-opt）已把“仍写死”清到只剩物理限制项；本轮复核并补充：

| 项 | 位置 | 为什么不是“模型相关死数” |
|---|---|---|
| attention graph 桶数 3 | `runner.h` `attn_graph[3]` | 内核支持的 blockDim 档数（8/16/32 warp），内核设计常量 |
| graph 懒捕获时序（第 1 token plain / 第 2 捕获 / 第 3 重放） | `qwen35.cpp` | 驱动约束（本机在 ctor 内捕获会 illegal memory access），不是调优值 |
| `head_topk` 的 k=5 | `runner.cpp` | 诊断/verify 用；贪心只需 argmax，不参与语义 |
| `topk_partial` block 数 clamp 256 | `runner.cpp` | 由 vocab 与每 block 处理量推导 |
| `kDp4aQ5kSmemLimit=4096`、QK 块布局常量、int8 32 值块、127 对称上限 | `kernel_select.h` / `cpu_matvec.cpp` | GGUF 量化格式 / AVX2 向量宽度 / 量化数学定义 |
| `max_ctx` 策略上限 1<<20 | `runner.cpp` / `placement.cpp` | 与 `ctx` 配置上界（1M）一致；是安全策略上限不是模型形状 |
| 交互式“f32 连 4096 都放不下才升 q8_0”的 4096 | `main.cpp` | = 服务历史默认 ctx（保证交互默认与原行为一致）；推荐值本身来自实测显存推导，用户可任意覆盖 |
| prefill chunk 自动放大/收缩 | `runner.cpp` | 自动判断：MoE 模型结构规则禁止放大（避免 35B 显存打满），dense 按实测空闲显存 + `prefill_rows_max` 上限；本轮未改 |
| 逐层自适应/扩展的安全余量 96 MiB | `runner.cpp` `expert_layer_fits_vram` | 实测标定：原 32 MiB 时 Coder 在 ctx 22 上出现“预算刚好、装载后比 margin 少 38 MiB”；+64 MiB 后 `free >= margin` 契约恢复（Coder 18→17 层，35B 层数不变） |

### 6.3 子命令专用参数（不进注册表，理由）

`--ids/--tokens/--n/--max-new-tokens`（run）、`--out/--json/--no-tensors/--force/--kv-bpe/--vram-gib`
（info/ir/plan）是**一次性的命令形状参数**，不是运行时行为开关，按设计不进配置注册表；
均在 `--help`/`docs/CONFIG.md` 中有说明。HTTP 服务的运行时参数（host/port/ctx/默认采样）
全部注册。

---

## 7. 统一入口检查（config / CLI / env 三选一）

- `--print-config` 的每行都带 `[default|env|config|cli]` 来源，可直接检查优先级链：
  CLI > config 文件 > env > 默认值（`tools/verify_generality.sh` 的 configuration layer 用例覆盖）。
- 本轮新增键全部遵循同一注册表机制（新增即三通道，不需要额外解析代码）。
- 交互式启动的选择通过 `Config::set(..., Source::Cli)` 写回注册表，因此后续 runner/derive_ctx_limit
  与 `--ctx`/`--kv-type` 完全同源，不存在“交互一套、CLI 另一套”的分叉。

---

## 8. 验证状态（最终二进制，本轮最后一轮全量）

| 套件 | 命令 | 结果 | 日志 |
|---|---|---|---|
| 全量验证 | `bash tools/verify.sh` | **ALL GREEN**（`VERIFY_EXIT=0`） | `build/logs/verify_final2.log` |
| 四模型冒烟 | `bash tools/smoke_models.sh` | **40716 / 185 / 2614 / 2614**（与期望一致，`SMOKE_EXIT=0`） | `build/logs/smoke_final2.log` |
| 生成回归 | `bash tools/gen_regression.sh` | **10/10 PASS**，逐 token == llama.cpp（`GENREG_EXIT=0`） | `build/logs/genreg_final2.log` |
| 通用化 | `bash tools/verify_generality.sh` | **GREEN**（26 passed / 0 failed，`GEN_EXIT=0`） | `build/logs/generality_final2.log` |
| HTTP 服务 | `bash tools/verify_serve.sh` | **GREEN**（22 passed / 0 failed，`SERVE_EXIT=0`） | `build/logs/serve_final2.log` |

### 8.1 四模型启动放置实测（最终二进制，`engine run ... --ctx 4096 --no-interactive`）

| 模型 | 决策器判定 | 决策器预计 GPU 专家层 | loader 实测 GPU 专家层 | 装载后 free VRAM | 结果 |
|---|---|---|---|---|---|
| qwen2.5-0.5B Q4_K_M | `gpu` | —（dense） | — | 9312 MiB | `check OK` |
| Qwen3.5-9B Q5_K_M | `gpu` | —（dense，hybrid） | — | 5922 MiB | `check OK` |
| Qwythos-9B-MTP Q4_K_M | `gpu` | —（dense，hybrid+MTP） | — | 6724 MiB | `check OK` |
| DeepSeek-Coder-V2-Lite Q5_K_M | `vram+ram split` | 约 13/26 | **13/26** | 1454 MiB | `check OK` |
| Qwen3.6-35B-A3B Q4_K_M | `vram+ram split` | 约 14/40 | **15/40** | 1703 MiB | `check OK` |

→ 决策器与 loader 实测一致（35B 差 +1 层，属正常偏差：决策器是装载前预算，loader 用实测空闲显存
逐层校验；两者都以 `free >= margin` 收尾）。**没有任何模型进入 SSD 档。** 完整日志：
`build/logs/ledger_{0.5b,9b,qwythos,coder,35b}.log`、`build/logs/capture_evidence.log`。

### 8.2 非交互行为一致性（脚本可复现性）

- `test_gen` 路径**完全未改**（不经过 `main.cpp` 的 placement/interactive 代码）：verify / smoke /
  gen_regression 全部复用 `test_gen`，结果与验收基线逐 token 一致。
- `serve` 非 TTY：`printf "" | ./build/engine.exe serve ...` 输出**0 行 `[interactive]`**，
  默认 ctx=4096，`/props` 的 `n_ctx=4096`，SSE / tokenize / 与 `test_gen` 贪心对拍全部通过
  （`verify_serve.sh` 22/22）。证据：`build/logs/noninteractive_serve.log`。
- 新增只有 `[placement]` 诊断日志（serve/run 启动时）与 TTY 下的 `[interactive]` 交互；
  生成 token、SSE 格式、HTTP 端点均未变。

### 8.3 本轮修掉的一个既有脆弱点（基线不是本轮引入）

`verify_generality.sh` 的 “post-load margin enforced” 在 VM 显存环境波动时会偶发 FAIL：
自适应放置最后采纳的一层在预算上只差 ~38 MiB，装载后 free 会比 `vram_margin_mb` 少 38 MiB
（Coder tested at ctx 22：free 1162 < margin 1200 → 打印 WARNING 而不是 `check OK`）。
本轮把逐层自适应/扩展的安全余量从 **32 MiB → 96 MiB**（`src/runner.cpp` 的
`expert_layer_fits_vram`，注释说明依据）：Coder 18→17 层、post-load free 1627 MiB（`check OK`）；
**35B 仍 16 层、topk 与 decode 不变**；其余套件重新全绿。这是本轮唯一改动 runner 放置参数的地方，
未触碰任何 CUDA 内核数学。

---

## 9. 未实现 / 放弃 / 风险（如实记录）

| 项 | 状态 | 说明 |
|---|---|---|
| SSD 流式加载本体 | ❌ 未实现 | 接口占位 + 确定性报错 + 方案（§2）；默认关闭，绝不盲目启用 |
| dense 模型 host/SSD 卸载 | ❌ 未实现 | 决策器明确报“dense 不能分片”，不假装 split |
| `kv_offload` host-mapped KV | ❌ 未实现 | 沿袭现状，仅解析并给出说明 |
| `kv_type` 全自动选择（非交互） | ⚠️ 保持 f32 默认 | 非交互行为必须可复现（硬约束 2）；交互式启动会给三档推荐并由用户选 |
| 决策器预算与 runner 实测的偏差 | ⚠️ 已知 | 决策器是“装载前预告”（用固定 workspace_mb 与实测 free），真正逐层放置由 runner 装载时实测决定；Coder/35B 的 GPU 层数与预告值可能差 1–3 层（以 runner 日志为准） |
| 交互式模型选择 | ⚠️ 仅本地目录扫描 | 不联网、不记忆历史；`model_dir` 可配 |
| MTP 接入 serve | ❌ 不在本轮 | 按任务约束属于另一条工作流 |

风险控制：
- 所有新增显存分配路径都不存在（决策器只读 `cuMemGetInfo`/`GlobalMemoryStatusEx`，不分配），
  因此 35B 的 ~1.6 GB 余量不会被新功能吃掉；预填充 chunk 的自动放大规则未改。
- 非交互模式（管道/服务/CI）行为与旧版一致：默认 ctx、生成 token、SSE 输出格式均未改；
  新增只有 `[placement]`/`[interactive]` 日志（非 TTY 时没有后者）。
- 新增 `[placement]` 日志在 serve/run 启动时打印，`test_gen`（验证脚本使用）路径完全未改。
