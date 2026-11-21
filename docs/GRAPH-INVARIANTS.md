# CUDA Graph 的安全不变量（graph replay 与宿主状态的交互）

本引擎在 decode 路径上用了 CUDA Graph：整模型图（稠密小模型）、qwen35 per-layer
mixer 图、prefill chunk 图，以及 MLA attention 段图。graph **只改变提交方式，不改变
kernel 序列**；但 `cuGraphLaunch` 不经过 `cuda::launch()`、也不会让"宿主看到的发射
计数"前进一样多。凡是依赖宿主发射顺序/计数的状态，都必须遵守下面的不变量，否则会
出现"graph 开/关结果不一致"甚至逐 token 发散的静默错误。

本文写给后续任何 graph 化 / 新宿主缓存改动。

---

## 不变量 1：任何一次 replay 都必须让"按发射计数判活"的宿主缓存失效

**背景**：`quant_reuse_` 是 dp4a GEMV 的 int8 激活暂存优化。如果连续两个 GEMV 读
*同一块*激活（gate/up、qkv、q/k/v），第二次不再重算 `qx_i8_`，复用第一次的结果。
判活条件是

```cpp
qc_ptr_ == in.data() && qc_nin_ == w.n_in && qc_stamp_ == cuda::g_launch_calls;
```

即"暂存写入之后没有任何 kernel 发射"（这也证明缓冲没有被覆盖）。

**问题**：一次 graph replay 在设备上执行 N 个 kernel，但 `cuda::g_launch_calls`
一次都不增加。于是"计数器没变"不再等价于"暂存仍有效"：replay 之后如果宿主紧接着
做 GEMV（中间没有真实发射），会误复用**上一个 token** 的 int8 暂存。

**实测触发路径**（2026-11-21，DeepSeek-V2-Lite / Coder + MLA per-layer 图）：
decode 时 head 的 2 次 top-k 发射在 `ENGINE_NO_HEAD_TOPK=1` 下消失，计数器恰好停在
旧的 `qc_stamp_` 上；第 3 个 token（第一次 replay）起输出变成重复 token。
`ENGINE_NO_HEAD_TOPK=0`（默认）时那 2 次发射掩盖了 bug —— 所以这类 bug 只在特定开关
组合下暴露，必须靠不变量而不是靠"跑一遍看起来对"。

**规定**（已实现）：`cuda::Graph::launch()` 递增独立的代数计数器
`cuda::g_graph_launches`；所有"按发射计数判活"的缓存把该代数一并纳入判活条件：

```cpp
qc_graph_ == cuda::g_graph_launches
```

于是 **任何一次 replay 都使缓存失效**（runner.h: `qc_graph_`，runner.cpp:
`run_gemv_plan` 的 dp4a 分支）。`g_launch_calls` 保持原语义（kernel 发射次数，用于
profiling），不要拿它替代代数计数器。

新增此类缓存时自检：
1. 判活条件里是否包含 `cuda::g_graph_launches`？
2. 是否假设"两次 kernel 发射之间没有别的 kernel"？若是，必须同时比代数。
3. 该缓存是否可能跨越一次 `Graph::launch()` 存活？若有，必须加代数判断。

回归测试：`tools/graph_cache_regression.sh`（graph 开/关、top-k head 开/关，逐 token
比对 + logits 逐字节比对）。

---

## 不变量 2：capture 期间不允许"被静默跳过"的宿主操作

`cuda.cpp` 在录制中会**静默跳过**下列调用（避免在 capture 里做非法操作）：

| 调用 | capture 期间行为 |
|---|---|
| `Buffer::upload` / `download` | 跳过（前 8 次打 `[capture] skipped` 警告） |
| `Buffer::upload_async` / `download_async` | 跳过（无警告） |
| `copy_d2d_async` | 跳过 |
| `stream_sync` / 默认流同步 | 跳过（打一次警告） |

所以被录制的序列里**不能有依赖这些调用的语义**：同步不会真的发生、拷贝不会真的执行，
而 `capture_end()` 不会报错。需要同步/下载的诊断路径（`ENGINE_DUMP*`、`ENGINE_PROFILE`、
`ENGINE_PIPE_PROF`、`ENGINE_KPROF`）必须在 graph 之前用 env 关掉 graph，而不是靠
"capture 里同步也能跑"。

同理，**capture 里不能 cuMemAlloc / 首次 cuBLAS 调用**（cuBLAS 内部会分配 workspace
并可能同步）。decode 段能图的先决条件就是"全 kernel 序列"：例如 MLA 段之所以能整段
录制，是因为 `mla_quant` 时四个投影都走 GEMV kernel，且 fp32 的 k/v 扩张在
`n_tokens<=4` 时走 `gemv_f32` kernel 而不是 cuBLAS。

---

## 不变量 3：replay 的语义必须与"录制期间看到的宿主状态"无关

- **capture 只记录、不执行**。所以设备状态（KV cache、recurrent state）由录制前
  那一次**正常执行**产生，capture 不得被当作"已经跑过"。
- 因此**懒捕获**的写法是：第一次 decode 正常执行（本 token 的结果由它产生），
  然后在同一个调用里录制；之后的重放才生效（见 `qwen35.cpp` 与 MLA 段的实现）。
- 录制时"宿主算出来的"内容会被冻结进 graph（kernel 参数值、grid/block、动态 shared
  size）。任何**逐 token 变化**的量都不能这样冻结：
  - 位置：必须走设备内存（`pos_dev_`），kernel 运行时读；
  - 序列长度相关的 launch 配置：要么与 ctx 无关，要么按桶分别录制
    （示例：qwen35 attention 的 warp 数是"桶"化的，故 `attn_graph[3]`）；
  - 指针：所有 buffer 地址在 load 后就固定，不能有"首次使用时分配"的缓冲出现在
    被录制序列里（`prepare_graph_scratch` 的存在原因）。

---

## 不变量 4：测量与真实账本

- 一次性捕捉成本发生在**第一个 decode token**（或 prefill chunk），做 perf A/B 时用
  `ENGINE_DEC_SKIP=2` 跳过它，或单独报告"含捕获成本"的口径；不要把它算进稳态。
- `ENGINE_DBG_LAYERMS` 每层一行 stderr，在本机会给每 token 加 ~11 ms，不能用来算账；
  用 `ENGINE_PROF_ACC=1`（内存累加，退出打印一次）。
- graph replay 不会让 `ENGINE_KPROF` 看到逐 kernel 事件（replay 不经过
  `cuda::launch` 的 kprof 钩子）；要证明"发射次数下降"，用 `ENGINE_PIPE_PROF`
  （打印累计发射计数，取相邻 token 差值）。
