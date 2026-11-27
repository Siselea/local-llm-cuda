# 热路径"隐藏 API 成本"审计（HOTPATH-API-AUDIT）

日期：2026-11-21 · 触发：上一轮发现 `std::getenv` = **2.13 µs/次**、热路径每层 ~30 次 ⇒ **2.8 ms/token**
（35B 相当于 8%），是同类 bug 里最值钱的一个。本轮按同一方法论把这一类扫干净。
**性质：纯审计报告，未改任何产品代码。** 审计对象：`HEAD = v0.1.1 (b068b5b)` + 工作树里
edge-kernels 未提交的 MLA graph / `tail_zero_copy` / `cpu_layer_ffn_graph` 改动。

工具：`tools/bench_hostapi.cpp`（纯 STL 微基准，**不需要 GPU 窗口**，可与 A/B 并行跑）、
`ENGINE_PROF_ACC=1`（上一轮的低开销逐段记账，给出每 token 的 launch/sync/拷贝计数）。

---

## 0. TL;DR

> **1) 还有 2 处 `std::getenv` 漏在热路径上**（上一轮 2.8 ms 修复的遗漏）：
> `ENGINE_GPU_FULL_LAYER_GRAPH`（`run_layer` 每层 1 次）+ `ENGINE_DUMP_MOE_SPLIT`（每 CPU 层 1 次）
> ⇒ 35B **64 次/token × 2.1 µs = 0.13 ms/token**；9B/Coder 按层数同理。
> **2) serve 的"非 greedy"采样路径是每 token 0.34 ms 的隐藏成本**（默认 `temp=0.6` 的真实用户请求都走这条）：
> 整行 248320 logits D2H + 1 MB 拷贝 + 全词表 `fill` + `partial_sort`。
> 若客户端发 `top_k=0`（只 top_p，llama.cpp 系 UI 常见）⇒ **~1.9 ms/token**（9B 的 9%）。
> **3) `capturing_now()` 每次都是一次驱动查询 `cuStreamIsCapturing`**，每 token ~97 次
> （25 次 sync + 72 次拷贝），估计 0.05–0.15 ms/token（单价待 GPU 窗口实测）。
>
> **实施状态（2026-11-21 17:35 复核，工作树已被其他 agent 推进）**：
> **#1 与 #3 已经落地**（见 §9.1/§9.2：工作树里 2 处 getenv 已改成 `ENGINE_ENV_ON`；
> `capturing_now()` 已改成 `g_capturing` 标志，cuda.cpp 注释里直接引用了本节测到的 "~97 次"），
> **#2 已由 server-mode 落成 bounded-candidate 采样**（`sample_bounded`，cap=1024，未提交），
> 本报告 §9.4 给出它的**残余成本实测**（默认 top_k=20 路径仍有 ~0.33 ms/token 可再省）。
> 原本的 1.9 ms `top_k=0` 风险已被 cap 消除；**top_k=0 不是 WebUI/默认路径**（§9.3 证据）。
>
> **除这 3 条外，热路径的"宿主 API"是干净的**（见 §6：`std::filesystem` 单次 22–44 µs 但**不在**热路径、
> config map 查找 0 次/token、TLS 0 次、shared_ptr 0 次、每 token 的 printf/snprintf 0 次、cudaMalloc 0 次、
> mutex/cv 合计 <2 µs/token）。
> **宿主侧剩下的真正大头是驱动调用**：`cuLaunchKernel` **408 次 × ~12.5 µs = 5.1 ms/token**、
> `cuMemcpy*Async` ~72 次 × ~13 µs ≈ **0.9 ms/token** —— 这是 WDDM 固有成本，只能靠"少 launch /
> 图 / 零拷贝"（edge-kernels 的 tail 融合与 MLA 图正是这条线）。

---

## 1. 方法

1. **单位成本**：`build/bench_hostapi.exe`（源码 `tools/bench_hostapi.cpp`）逐 API 紧循环测量，
   每项 2e6–2e7 次、取整轮均值；用 `volatile` sink + volatile 循环变量阻止编译器消除/提升
   （踩过的坑：`snprintf`/`malloc` 的整段会被优化掉，见 §7.1）。测量时 CPU 负载 15%。
2. **每 token 次数**：
   - launch / sync / 拷贝次数来自 `ENGINE_PROF_ACC=1` 的实测（35B：408 launch、25 sync、24 次
     每层 D2H+H2D 往返），
   - 其余按调用点逐个读码计数（`runner.cpp` 热区 >1450 行、`qwen35.cpp` 层路径、
     `cuda.cpp` 的 Buffer/Event/launch/stream_sync、`cpu_matvec.cpp` 的 Pool/moe_forward_mt、
     `sampler.cpp`、`serve.cpp` 的生成循环）。
3. **归属**：`getenv` 类成本只与调用次数×单价有关；驱动类（cuLaunchKernel/cuMemcpy/…）用实测的
   12.5–13 µs 单价（上一轮 `_syncbench` 在空闲 GPU 上测得的复数值，本轮沿用）。

---

## 2. 单位成本表（本机实测，2026-11-21）

| 操作 | ns/op | 备注 |
|---|---|---|
| `std::getenv`（未设，16 字符名） | **2099** | 罪魁已修，剩 2 处（§4.1） |
| `std::filesystem::exists`（命中 / 未命中） | **43 574 / 22 556** | **单次 22–44 µs**，热路径绝不能出现 |
| `snprintf`（1 次格式化） | 246 | |
| `fprintf(FILE*, 一行)` | 350 | |
| 3× `snprintf`（serve 的 SSE 帧，实测） | 99 | 每 token 1 次 ⇒ 0.1 µs，可忽略 |
| `std::string` 由字面量构造（SSO ≤15 字符） | 1.3 | 无堆分配 |
| `std::string` 由字面量构造（40 字符） | 45 | 堆分配 |
| `std::string` 拷贝（40 字符堆）/ `operator+` | 55 / 46 | |
| `std::to_string(int)` | 10 | SSO |
| `std::string::substr(2,4)`（SSO） | 1.3 | |
| `std::string == 字面量` | 1.1 | |
| `std::vector<float>` 构造 2048 + 析构 | 140 | 1 次堆分配 |
| `std::vector<int>` 构造 257 / `assign` 8 / 同尺寸 `resize` | 55 / 46 / 1.1 | 同尺寸 resize 不分配 |
| `std::map<std::string,X>` 查找（200 项，命中/未命中） | 27 / 59 | 键比较开销 |
| `std::unordered_map<std::string,X>` 未命中 | 36 | |
| `std::map<void*,X>` 查找（64 项） | 1.5 | 指针键 |
| `std::function` 构造(2 引用捕获)+调用 / 调用 / 构造(6 引用) | 1.3 / 2.0 / **47** | **>SBO 会堆分配** |
| `std::mutex` lock+unlock（无竞争） | 17 | |
| `std::condition_variable::notify_all`（无等待者） | 3.0 | |
| `thread_local int` 读写 / `thread_local std::string` 访问 | 1.3 / 1.3 | 本机很便宜 |
| `shared_ptr` 拷贝+析构（原子引用计数） | 10.9 | |
| `new/delete` 小对象 / `make_unique<int>` / `vector<int>` 新建(1 分配) | 47 / 49 / 44 | **堆分配 ≈ 45–50 ns** |
| `steady_clock::now()` | 36.9 | |
| `std::partial_sort(256, top 8)`（宿主路由） | **1928** | |
| 宿主路由整趟（256 sigmoid + partial_sort） | **3907** | 每 CPU 层 1 次 |
| `std::partial_sort(248320, top 20)`（serve 采样） | **152 394** | |
| `sampler` 整步（fill 248320 + partial_sort top20） | **199 736** | 非 greedy 每 token 1 次 |
| `std::sort(248320 pairs)`（top_k=0 时 top_p 排序） | **448 910** | |
| `std::exp` × 248320（top_k=0 时整词表 softmax） | **1 334 093** | |
| `memcpy` 1 MB（float[248320] assign） | 37 908 | |
| `argmax` 248320（greedy 全行路径，serve 不走） | 258 129 | |

---

## 3. 每 token 账（35B `--gpu-layers 15`：24 CPU MoE 层 + 16 非 CPU 层，34.3 ms/token）

| 类别 | 次数/token | 单价 | 成本 | 状态 |
|---|---|---|---|---|
| 宿主 GEMV（算力） | 24 层 | 0.617 ms/层 | **14.8 ms** | 已到隔离基准，不动 |
| `cuLaunchKernel` | 408 | ~12.5 µs | **5.1 ms** | 驱动成本 → 靠少 launch/图 |
| `stream_sync`（等待 GPU + flush） | 25 | 2.1 µs（+GPU 等待） | 0.05 ms 自身 | 主要部分是 GPU 工作 |
| `cuMemcpy*Async`（每层 rl/hn D2H + host_part H2D） | ~72 | ~13 µs | **~0.9 ms** | 驱动成本；tail 零拷贝可减 1 次/层 |
| **未缓存的 `std::getenv`** | **64** | **2.1 µs** | **0.13 ms** | **确认 bug（§4.1）** |
| 宿主路由（256 sigmoid + partial_sort） | 24 | 3.9 µs | 0.094 ms | exp 不可改（bit-exact）；排序可试 |
| `std::chrono::now()`（记账） | ~900–1000 | 36.9 ns | 0.034 ms | 可忽略/可编译期关掉 |
| `std::vector` 新建（run_moe 每层 ~11 个） | ~264 | 44 ns | 0.012 ms | 可用成员缓冲 |
| `std::function`（>SBO 堆分配，3/CPU 层） | 72 | 47 ns | 0.003 ms | 可改模板 |
| `mutex`+`condvar`（3–4 区/CPU 层） | ~84 | 17+3 ns | 0.002 ms | 可忽略 |
| `capturing_now()`（驱动查询 `cuStreamIsCapturing`） | ~97 | **待测**（估 0.5–1.5 µs） | **0.05–0.15 ms?** | 可疑（§5.1） |
| `cuEventCreate`+`Destroy`（run_pipeline 的 2 个局部 Event） | 4 | **待测**（估 1–5 µs） | **4–20 µs?** | 可疑（§5.2） |
| printf/fprintf / snprintf | **0** | — | 0 | ✅ 全部 env 门控 |
| `std::filesystem` | **0** | 22–44 µs | 0 | ✅ 只在装载期（§6） |
| config map 查找 / TLS / shared_ptr 拷贝 / cudaMalloc | **0** | — | 0 | ✅（§6） |

**纯宿主 API（不含 GEMV/驱动）合计 ≈ 0.25 ms/token（0.7%）**；
其中 **0.13 ms 是本轮新确认的遗留 getenv**。
9B 的情况不同：它的整条 pipeline 是**一张 CUDA Graph**（每 token 1 次 `cuGraphLaunch` + 1 次 sync），
所以 launch/拷贝类成本几乎为 0，9B 的宿主 API 成本只有**非 greedy 采样路径**（§4.2）：
**~0.34 ms/token（1.6%）——1.9 ms（9%）如果 top_k=0**。

### 3.1 对真实用户（serve 非 greedy，默认 `temp=0.6 top_k=20 top_p=0.95`）

| 段 | 每次 token | 证据 |
|---|---|---|
| 整行 logits：`forward(..., last_row_only=true)` → 248320 floats D2H（1 MB pinned） | ~0.10–0.13 ms | 1 MB PCIe ~83 µs + memcpy 37.9 µs + 驱动 |
| `last.assign(row.end()-vocab, row.end())`（1 MB 拷贝） | 37.9 µs | 微基准 |
| `Sampler::sample`：`cand_.clear()+resize(248320)` + fill | ~52 µs | 微基准 |
| `std::partial_sort(248320, top 20)` | ~152 µs | 微基准 |
| top_p/min_p/temp（只在 top-20 上） | ~1 µs | 代码 + 微基准 |
| **合计** | **≈ 0.34 ms/token**（9B 21.2 ms 的 **1.6%**、35B 34.3 ms 的 **1.0%**） | |
| **`top_k=0`（top_p-only）时**：整词表 `exp`(1.33 ms) + `sort 248320 pairs`(0.45 ms) + 2 MB 临时数组 | **≈ 1.9 ms/token** | 9B **9.0%**、35B **5.5%** |

---

## 4. 确认的 bug（本轮新增）

### 4.1 两处未缓存的 `std::getenv`（同类 bug 的遗漏）——35B 0.13 ms/token
- `src/runner.cpp:1607`：`if (gpu_layer && std::getenv("ENGINE_GPU_FULL_LAYER_GRAPH"))`
  —— 在 `run_layer` 的 `layer_graph_auto` 分支里，**每层执行 1 次**（35B 40 层 = 40 次/token）。
- `src/runner.cpp:3266`：`if (any_cpu && std::getenv("ENGINE_DUMP_MOE_SPLIT"))`
  —— **每 CPU 层 1 次**（35B 24 次/token）。
- 合计 **64 × 2.1 µs = 0.13 ms/token**（35B 0.4%；Coder 同类）。
- **修复**：改用已有宏 `ENGINE_ENV_ON("…")`（`include/engine/config.h`），零风险、行为 bit-exact。
- 注：`qwen35.cpp` 里剩余的 `getenv` 全部被 `layer == 0 &&` 短路或只在装载期，**安全**；
  `cuda.cpp` 里的是 kprof/图捕获/一次性的，**安全**。

### 4.2 serve 非 greedy 采样每次 token 走全词表（0.34 ms；`top_k=0` 时 1.9 ms）
> **状态（17:35）**：server-mode 已在工作树落地 `Sampler::sample_bounded()`（候选上限
> `kDefaultCandidateCap=1024`，`std::nth_element` 选候选，去掉了第二次 1 MB 拷贝）⇒
> **`top_k=0` 的 1.9 ms 风险已消除**；默认 top_k=20 路径的残余成本见 §9.4（实测 ~0.33 ms/token）。
- 依据：`serve.cpp:376` `greedy = !(gp.temperature > 0.0)`；默认 `temp=0.60` ⇒ 走
  `model.forward(...)` 全行 logits + `sampler.sample()`；
  `sampler.cpp:108-125`：`top_k>0` 时 `cand_` 填满 248320 + `partial_sort`；
  `top_k==0` 时 `probs_of(cand_)` 对 248320 个候选逐个 `exp`，再用 `std::sort` 排序（2 MB 临时）。
- 影响：**所有带温度的请求**（用户默认就这样）。
- **修复建议（按侵入性排序）**：
  1. **设备端 top-k 头扩到 k=64/128**：host 只对 64–128 个候选做 top_p/min_p/temp。
     省掉 1 MB D2H + 1 MB拷贝 + 全词表 fill/sort（−0.3 ms 常规 / −1.9 ms `top_k=0`）。
     代价：采样分布从"全词表 top_p"变成"top-128 内的 top_p"——数学上与 llama.cpp 不同
     （但 llama.cpp 的 top_p 也是在全词表上算的；要把差距控制住需要 k 足够大）。
     **归属：server-mode（serve.cpp / sampler.cpp / 设备头 kernel）**。
  2. **去掉 `row` → `last` 的第二次 1 MB 拷贝**：`forward()` 已经把整行写进 `out`（1 MB），
     `serve` 又 assign 进 `last`。可以让 forward 直接写进一个复用的成员缓冲（−37.9 µs/token）。
  3. **`top_k=0` 时先取 top-N 再做 top_p**（N=128/256，`std::partial_sort` 由 152 µs→ 不变但省掉
     1.33 ms 的整词表 exp 和 0.45 ms 的整表 sort）。属于 sampler 语义修改，需 server-mode 决策。
- **验证方法**：用一个 `temperature>0` 的请求 A/B `top_k=20` vs `top_k=0` 的 tok/s（Task B 里做），
  预期差 ~1.5–2 ms/token（9B 约 7–9%）。

---

## 5. 可疑、需要 GPU 窗口实测

### 5.1 `capturing_now()` 每次都是一次驱动查询
> **状态（17:35）**：**已落地**（工作树 `src/cuda.cpp:511-512` 改为 `g_capturing` 标志，
> capture_begin/end/abort 维护），下面的分析保留作案情记录；**仍需加固的边界见 §9.2**
> （`capture_end` 异常路径会漏置位、Graph 析构漏置位）。
`cuda.cpp:507`：
```cpp
static bool capturing_now() {
  if (!g_driver.StreamIsCapturing || !g_stream) return false;
  int mode = 0;
  if (g_driver.StreamIsCapturing(g_stream, &mode) != CUDA_SUCCESS) return false;
  return mode != 0;
}
```
调用点：`Buffer::upload_async` / `download_async` / `upload` / `download` / `copy_d2d_async` /
`stream_sync` / `sync` ⇒ **每 token ~97 次**（25 sync + ~72 拷贝）。
`cuStreamIsCapturing` 是驱动调用；单价估计 0.5–1.5 µs（从上一轮 `_syncbench` 的
sync=2.1 µs 总时长反推上限）⇒ **0.05–0.15 ms/token**。
- **修复**：在 `Graph::capture_begin/end/abort` 里维护一个全局 capture 标志（或 depth 计数），
  `capturing_now()` 只读该标志；行为等价（引擎自己控制所有 capture 入口）。
- **归属**：`src/cuda.cpp`（当前在 edge-kernels 的未提交改动里，需协调后落）。

### 5.2 `run_pipeline` 每 token 构造 2 个 `cuda::Event`
`src/runner.cpp` 中 `run_pipeline` 顶部 `cuda::Event pev0, pev1;` 是**无条件**的
（只在 `pipe_prof` 为真时 record），但构造/析构会做 2×`cuEventCreate` + 2×`cuEventDestroy`。
估计 4–20 µs/token（单价待测）。
- **修复**：改成 `DenseRunner` 的成员（或 `static` 懒构造）；或在 `pipe_prof` 为假时用空对象。
- **归属**：`src/runner.cpp`。

### 5.3 每 token 的图启动次数（`g_graph_launches`）与 `cuGraphLaunch` 单价
35B CPU 层 attention 图（24 次）+ GPU 层 attention 图（16 次）≈ 40 次 `cuGraphLaunch`/token，
加上工作树里 edge-kernels 新加的 `cpu_layer_ffn_graph`。历史上本机 `cuGraphLaunch` 约
0.05–0.1 ms/次（GPU 层实测 0.275→0.199 的净收益反推），如果是真的，40 次 = 2–4 ms/token，
值得单列实测（挂 `g_graph_launches` 计数器 + 事件计时）。
> 这也是"为什么 CPU 层 attention 图化收益不稳定"的最可能解释：图启动本身不便宜。

### 5.4 拷贝次数的精确值
当前工作树有 `tail_zero_copy`（把每 CPU 层最后 1 次 H2D 变成零拷贝 PCIe 读）：预期每 token
拷贝从 ~72 降到 ~48。需要 GPU 窗口用 `ENGINE_PROF_ACC`/kprof 确认。

---

## 6. 已验证"干净"的项（覆盖面证明）

| 候选 | 结论 | 证据 |
|---|---|---|
| `std::filesystem`（每 token 调用） | **0 次** | 全仓 `filesystem::` 只在装载期（`runner.cpp:45` detect_container）与 serve 启动期（`serve.cpp:1794 module_dir`，`webui_html()` 是 static 缓存） |
| config `std::map` 查找（每 token） | **0 次** | 热区里的 `engine::config::get_*` 只出现在 `static const` 初始化（`cpu_moe_batch`、`gd_smem` 等）与装载期 `load_config_snapshot()` |
| `thread_local`（含析构的 TLS） | **0 处** | 全仓唯一相关是 `cpu_matvec.cpp:1421` 的注释（明确用普通 static） |
| `shared_ptr` 拷贝 / `std::to_string` / `std::string` 堆构造（每 token） | **0 次** | 热区里只有 `ENGINE_DUMP` 路径（env 门控）与错误路径；`make_row_dot` 拷的量化类型名是 SSO（1.3 ns） |
| 每 token 的 `printf/fprintf/snprintf` | **0 次**（test_gen）；serve 每 token 1 次 SSE 帧 = 0.1 µs | 所有 `fprintf` 都在 `if (kDebug)`/`ENGINE_*` 门控后；serve 的 `[req]` 日志是每请求 |
| 每 token 的 `cudaMalloc/cudaFree/`pinned 分配 | **0 次**（`PinnedBuffer::resize` 是 grow-only，`cuda::Buffer` 全部持久） | 上一轮已确认；本轮复核 `moe_host_part_`/`hn_pin_`/`xout_pin_` |
| `std::mutex`/`condition_variable` | 84 次/token = **0.002 ms** | `Pool::run` 每区 1 次；无竞争 |
| `std::function` | 72 次/token（>SBO 堆分配）= **0.003 ms** | `moe_forward_mt` 的 `run_chunked` |
| `std::vector` 分配 | ~264 次/token = **0.012 ms** | `run_moe` 每层的临时容器 |
| 驱动查询 `cuMemGetInfo` | **0 次**（只在装载/放置阶段） | `mem_info()` 全部调用点在构造/放置/校准路径 |
| `cudaGetDevice`/`cuCtxGetDevice` | **0 次** | 全仓无此调用 |
| Coder 的 MLA 路径（`run_mla_attention`） | **干净** | 其中的 `stream_sync`/`download`/`std::vector`/`ofstream` 全部在 `ENGINE_DUMP_MLA` / `mla_dump` 门控之内；热路径只有 `matmul()`（无字符串/无分配） |

### 6.1 证据文件
- 微基准源码/二进制：`tools/bench_hostapi.cpp` → `build/bench_hostapi.exe`；本轮原始输出 `build/hostapi.log`。
- 每 token 计数：上一轮 `ENGINE_PROF_ACC` 记录（408 launch / 25 sync / 每 CPU 层 9 launch），
  `docs/reports/HOTPATH-ACC-REPORT.md` §2。

---

## 7. 方法论与坑

### 7.1 微基准必须先排除"编译器消除"
第一版 `bench_hostapi` 里 `snprintf`、`malloc/free`、`3×snprintf` 得到 1.3 ns（假的）：
编译器把与结果无关的整段删掉/提升到循环外。修法：结果写 `volatile` sink、
循环变量用 volatile 传入（让被测量依赖迭代号）、`fprintf` 输出到临时文件再删。
**教训：任何"某 API 很便宜"的结论，先确认它没被优化掉**（用 `malloc(8)` 应约 45 ns 自检）。

### 7.2 测量环境
本轮微基准在 CPU 负载 15% 下跑（另一 subagent 在做 GPU A/B），量级可信；绝对值偏保守。
受影响最大的是内存带宽型条目（`partial_sort 248320`、`memcpy 1MB`、`argmax`）；
纯算术/系统调用类（getenv、snprintf、mutex、chrono、堆分配）稳定。

---

## 8. 修复清单（按收益/风险排序，供 lead 派活）

| # | 修复 | 预期收益 | 风险 | 归属 |
|---|---|---|---|---|
| 1 | 2 处遗留 getenv → `ENGINE_ENV_ON` | 35B **0.13 ms/token** | 零（bit-exact） | runner.cpp |
| 2 | serve 非 greedy：设备端 top-k 扩到 64/128，host 只对候选采样 | **0.34 ms/token**（`top_k=0` 时 **1.9 ms**） | 中（采样分布变化，需产品决策） | server-mode |
| 3 | serve 去掉 `row`→`last` 的 1 MB 二次拷贝 | 0.038 ms/token | 低 | server-mode |
| 4 | `capturing_now()` 改缓存标志（去掉 ~97 次驱动查询/token） | 0.05–0.15 ms/token | 低（需 GPU 实测单价确认） | cuda.cpp（与 edge-kernels 协调） |
| 5 | `run_pipeline` 的 2 个 `cuda::Event` 改成员/懒构造 | 4–20 µs/token | 低 | runner.cpp |
| 6 | 宿主路由 `partial_sort(256,8)` 换手写 top-8 | ≤0.05 ms/token | 中（tie 顺序需 bit-exact 验证） | runner.cpp |
| 7 | `g_launch_ms`/`g_sync_ms` 记账编译期开关 | 0.03 ms/token | 零 | cuda.cpp |
| 8 | run_moe 临时 vector 改成员缓冲 | 0.012 ms/token | 低（重入性需确认） | runner.cpp |

**结论**：这一类的"金矿"（getenv 2.8 ms）已经在上一轮挖掉；本轮扫出的是
**0.13 ms 遗留 + 0.34–1.9 ms 的 serve 采样路径 + 0.05–0.15 ms 的 capture 查询**。
宿主侧真正的大钱只剩驱动调用（5.1 ms launch + 0.9 ms 拷贝），那是"少 launch / 图 / 零拷贝"的活。


---

## 9. 补丁片段与实施状态（2026-11-21 17:35 复核）

> 复核时工作树已被其他 agent 推进（HEAD `949f21a` + 未提交的 edge-kernels MLA/tail 改动 +
> 未提交的 server-mode bounded-sampling 改动）。**§4/§5 的三条发现中，#1、#3 已落地，#2 已换实现**；
> 下面给出「已落地部分的确认 + 仍需落的加固/下一步补丁」，可直接照抄。

### 9.1 修复 #1（2 处热路径 getenv）—— **已在工作树落地** ✅

`src/runner.cpp` 的精确替换（当前行号，anchor 是唯一上下文）：

```diff
@@ run_layer() —— layer_graph_auto 分支内（src/runner.cpp:1607）
-      if (gpu_layer && std::getenv("ENGINE_GPU_FULL_LAYER_GRAPH")) {
+      if (gpu_layer && ENGINE_ENV_ON("ENGINE_GPU_FULL_LAYER_GRAPH")) {
         run_gpu_layer_graph(layer, start_pos);   // whole layer (needs device routing)
       } else {
         run_q35_attn_graph(layer, start_pos);    // mixer/attention part only
       }
```

```diff
@@ run_moe() —— DUMP_MOE_SPLIT 分支（src/runner.cpp:3266）
-  if (any_cpu && std::getenv("ENGINE_DUMP_MOE_SPLIT")) {
+  if (any_cpu && ENGINE_ENV_ON("ENGINE_DUMP_MOE_SPLIT")) {
     std::vector<float> gpu_out(static_cast<size_t>(n_tokens) * hidden_);
```

验证命令（应该只剩 `ENGINE_PROF_ACC` 的 `static` 与 `layer == 0 &&` 短路的 2 条，均安全）：
```bash
grep -n 'std::getenv' src/runner.cpp | awk -F: '$1>1450'
```
收益：35B **64 次/token × 2.1 µs = 0.13 ms/token**；零风险（值相同，bit-exact）。

### 9.2 修复 #3（capturing_now 去掉驱动查询）—— **已落地，但有两个漏置位边界需加固** ⚠

已落地代码（`src/cuda.cpp:511-512` + 597-620，注释已引用 "~97 calls per decode token"）：
```cpp
static bool g_capturing = false;
static bool capturing_now() { return g_capturing; }
// capture_begin():  ... StreamBeginCapture(...);  g_capturing = true;
// capture_end():    check(StreamEndCapture(...)); g_capturing = false; ...
// capture_abort():  g_capturing = false; if (StreamEndCapture(...) == SUCCESS) GraphDestroy(g);
```

**行为等价性论证**：`cuStreamBeginCapture` 全仓只在 `Graph::capture_begin` 里调用
（`grep -rn StreamBeginCapture src/` 只有 cuda.cpp:606），所以
`g_capturing == true` 的区间正好等于原来 `cuStreamIsCapturing(g_stream)` 返回真的区间；
8 个 `capturing_now()` 调用点（upload_async/download_async/copy_d2d_async/upload/download/
stream_sync/sync/launch）语义完全不变：**捕获期跳过、非捕获期不跳过**。

**仍需加固的两个边界**（都可导致 `g_capturing` 永久为 true ⇒ 之后所有 sync/拷贝被静默跳过 ⇒
卡死/结果错，且只在异常路径触发，平时测不出来）：

1. `capture_end()` 里 `check(StreamEndCapture(...))` **先抛异常、后清标志**：
   `StreamEndCapture` 失败时 `g_capturing` 永远停在 true。
2. `Graph` 在 capture 中间被析构（录制体内抛异常但调用方忘了 `capture_abort()`，或未来新代码）
   ⇒ 标志永远 true。

建议补丁（cuda.cpp；同时把"归谁所有"显式化，防止 `capture_abort()` 误清别人的捕获）：

```diff
-static bool g_capturing = false;
+// Owner pointer makes capture_abort()/the destructor idempotent and prevents an abort of
+// an unrelated Graph from clearing a live capture's guard.  CUDA refuses nested captures
+// on one stream, so at most one owner exists at a time.
+static Graph* g_capture_owner = nullptr;
+static bool g_capturing = false;   // == (g_capture_owner != nullptr)
 static bool capturing_now() { return g_capturing; }
@@ void Graph::capture_begin()
   check(g_driver.StreamBeginCapture(g_stream, 0 /* global */), "cuStreamBeginCapture");
+  g_capture_owner = this;
   g_capturing = true;
@@ void Graph::capture_end()
-  check(g_driver.StreamEndCapture(g_stream, &graph_), "cuStreamEndCapture");
+  // Clear the guard FIRST: if StreamEndCapture fails the stream must not keep looking
+  // like it is capturing (that would silently skip every later sync/copy = hang).
+  const CUresult rc = g_driver.StreamEndCapture(g_stream, &graph_);
   g_capturing = false;
+  g_capture_owner = nullptr;
+  check(rc, "cuStreamEndCapture");
   check(g_driver.GraphInstantiate(&exec_, graph_, 0), "cuGraphInstantiate");
@@ void Graph::capture_abort()
+  if (g_capture_owner != this) return;   // nothing of ours is open (also: don't touch a foreign capture)
   g_capturing = false;
+  g_capture_owner = nullptr;
   if (!g_driver.StreamEndCapture || !g_stream) return;
   void* g = nullptr;
   if (g_driver.StreamEndCapture(g_stream, &g) == CUDA_SUCCESS && g && g_driver.GraphDestroy)
     g_driver.GraphDestroy(g);
@@ Graph::~Graph()
+  // Safety net: a Graph destroyed while its own capture is still open must not leave the
+  // guard stuck (see the two failure modes above).
+  if (g_capture_owner == this) {
+    std::fprintf(stderr, "[cuda] WARN Graph destroyed during capture; aborting\n");
+    capture_abort();
+  }
   if (exec_) g_driver.GraphExecDestroy(exec_);
   if (graph_) g_driver.GraphDestroy(graph_);
```

**对称性检查（建议加上，捕获期偏差会立刻暴露）**：
```cpp
// cuda.cpp（诊断用，默认零开销：只在析构时看一眼）
bool capturing_active() { return g_capturing; }
// DenseRunner::~DenseRunner()（已有 kprof_dump/prof_acc::dump 的地方）
if (cuda::capturing_active())
  std::fprintf(stderr, "[cuda] WARN capture still open at shutdown\n");
```
线程安全：捕获只发生在主线程（Pool 工作线程不调 CUDA），普通 `bool` 足够；要更保险可用
`std::atomic<bool>`（relaxed）——原驱动查询本身是线程安全的，但引擎里不存在并发捕获。

### 9.3 `top_k=0` 证据 —— **不是 WebUI / 默认路径**（优先级下调）

`build/webui/index.html`（llama.cpp 单文件 webui，4.41 MB）里 `top_k` 共 5 处，证据如下：

| # | 证据（原文） | 结论 |
|---|---|---|
| 1 | `samplers:"edkypmxt",temperature:.6,dynatemp_range:0,dynatemp_exponent:1,`**`top_k:20`**`,top_p:.95,min_p:0,...` | WebUI **默认 top_k = 20** |
| 2 | `const me={messages:H,stream:!0,cache_prompt:!0,samplers:...,temperature:...,`**`top_k:Xt.top_k`**`,top_p:...,min_p:...}` | **每次请求都发送** top_k（不是省略） |
| 3 | `if(c.temperature===0.8&&c.top_k===40&&c.top_p===0.95&&c.min_p===0.05){c.temperature=0.6;c.top_k=20;c.min_p=0}` | 迁移规则把 llama.cpp 原生默认（0.8/40/0.95/0.05）改写成 **0.6/20/0.95/0** |
| 4 | `jQ=["temperature","top_k","top_p","min_p","max_tokens"]` | top_k 是 UI 滑块之一（**用户理论上可拖到 0**，但默认值与每次发送值都是 20） |

我们的 serve 侧：`serve.cpp:1984` `st.opts.default_top_k = 20`（未显式配置时）、
`serve.cpp:782` `gp.top_k = int_or(body, "top_k", st.opts.default_top_k)` ⇒ 请求缺省也是 **20**。

⇒ **结论：默认用户路径是 top_k=20**（0.2–0.35 ms/token 级别），**`top_k=0` 的 1.9 ms 属于边缘场景**
（只有显式传 0 的 API 客户端或把滑块拖到 0 的用户）。因此 #2 的优先级从"默认用户路径"下调为
"边缘保护"；而且 server-mode 的 bounded-candidate 实现（cap=1024）已经把这个边缘场景也封顶到
~0.24 ms。**真正该继续做的是默认路径的残余成本（§9.4）。**

### 9.4 serve 采样：bounded 实现的残余成本（默认 top_k=20，实测）

server-mode 的 `sample_bounded()` 每次非 greedy token 仍要做：
1. **整行 1 MB logits D2H**（`forward()` 全行）≈ 96 µs（PCIe 83 + 驱动 13，上一轮实测口径）；
2. `select_candidates()`：`rank_idx_.resize(248320)` + **填充 248320 个下标** + `std::nth_element(248320, n)`。

本机实测（`tools/bench_hostapi.cpp`，两次一致）：

| 实现 | 实测 | 说明 |
|---|---|---|
| `fill 248320 + nth_element(n=64)`（当前默认路径） | **235 µs** | n=1024 时同值（233 µs）——代价在 fill+partition |
| `fill 248320 + partial_sort(248320, k=20)`（旧全词表路径） | 185 µs | 旧路径的排序部分更快，但额外有 1 MB `last.assign`(38 µs) |
| **单遍 top-64 扫描（不建下标数组）** | **128 µs** | 实测快 ~107 µs，且不需要 1 MB rank_idx_ |
| 设备端 top-k 头 + `sampler.set_candidates()`（新 API 已存在） | — | 直接消灭 (1)+(2)，预计 **−0.33 ms/token** |

**建议的下一步（server-mode）**：
- 短期：`select_candidates()` 换成单遍 top-N（`n = top_k>0 ? max(top_k,64) : cap`），
  去掉 `rank_idx_` 的 248K 下标填充与 nth_element ⇒ **−0.1 ms/token**，语义不变（候选集大小相同）。
- 中期：非 greedy 也走**设备端 top-k 头**（把 `k` 从 5 扩到 64/128，GPU 侧成本可忽略），
  host 用已有的 `Sampler::set_candidates(ids, logits, n)` 直接喂候选 ⇒ 去掉整行 1 MB D2H
  + 全部 248K 扫描 ⇒ **−0.33 ms/token**（9B 约 1.5%、35B 约 1%）。
  代价：top_p/min_p 在 64/128 候选上归一化（已有 cap=1024 的近似前提，方向一致）。
