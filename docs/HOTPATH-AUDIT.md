# 热路径静态体检（HOTPATH-AUDIT）

**日期**：2026-11-21 · **性质**：只读审计（未改引擎代码）· **作者**：edge-kernels
**口径**：GPU 被另一个 agent 占用，本轮**不跑 GPU 测量**。结论 = 代码路径 + 每 token 调用计数 +
**同工具链（ziglang c++，同 libc）主机微基准**（`tools/bench_host.cpp`，不碰 CUDA）。

> **⚠️ 并发编辑**：审计期间 `overhead-17ms` 正在改 `runner.cpp`/`qwen35.cpp`（11:48:48 把热路径
> `std::getenv` 迁移到 `ENGINE_ENV_ON`）。本报告行号以 **11:48:48 快照**为准：
> `src/runner.cpp` 210,203 B / **4392 行** / mtime 11:48:48 · `src/qwen35.cpp` 901 行 (11:48:48) ·
> `src/cuda.cpp` 771 行 (11-20 19:08) · `src/cpu_matvec.cpp` 1540 行 · `include/engine/config.h` 163 行
> （新宏 `ENGINE_ENV_ON`）。源文件快照在 `build/audit-snap/`。引用请以**函数名 + 代码片段**为准；
> §4 会标注"已被并发修复"，避免重复派工。

---

## 0. 结论（先看这张）

| # | 发现 | 状态 | 每 token 预估 |
|---|---|---|---|
| 1 | 热路径 `std::getenv`：本机 **2.1 µs/次**，35B 曾达 ≈937 次/token | **已被并发修复 95%**（剩 2 处热路径，约 84 µs） | 修复收益本该是 **−1.9 ms**（35B） |
| 2 | **Coder（MLA）完全没有 CUDA Graph 覆盖**：≈500–700 次裸发射/token | **未动** | **−2.0~2.5 ms 潜空间** |
| 3 | `run_moe` 逐专家循环在 `cpu_batched` 已算完后仍跑满 ne=256 轮（含 3 次时钟读） | **未动** | **−0.5 ms**（35B） |
| 4 | 35B GPU 层 MoE 段（16 层 × ≈11 次发射）不在任何图里；上次"整层图"变慢的真实原因是 `layer_graph_mode_` 关掉了融合专家内核 | **未动** | **−0.8~1.0 ms 潜空间** |
| 5 | Coder GPU 层 host 路由：每层 1 次 D2H sync + **2 次同步 H2D**（Q5_1 ⇒ dev-route 默认关） | 未动 | −0.7~1.0 ms 潜空间 |
| 6 | 9B/0.5B：每 token 两次 `stream_sync` 可合并为一次（图 replay → 固定 D2H） | 未动 | −20~40 µs |
| 7 | CPU 层 mixer 图尾部还有 `rmsnorm(ln2)` + router GEMM 在图外（每层 2 次发射） | 未动 | −0.2~0.3 ms（35B） |
| 8 | `x_`/`pos_` 用 **pageable** 内存做 `upload_async`（每 token 2 次） | 未动 | −20~50 µs（全模型） |
| 9 | `cuda::launch` 每次无条件 2 次 `steady_clock::now()`；`run_pipeline` 每 token 造 2 个 Event | 未动 | −35~55 µs（35B/Coder） |
| 10 | router GEMM 走 cuBLAS（40 次/token）而非现成 `router_gemm` kernel | 未动 | −0.1~0.4 ms？未测 |

**净结论**：getenv 这条已经被并发修掉了（但我的微基准给出了它的**单价 2.1 µs**，可用来核对收益，
并指路剩下 2 处）。**未被触碰的最大两块是：Coder 没有图（−2 ms 级）和 35B GPU 层 MoE 不进图
（−1 ms 级）；最容易落地的是 35B 逐专家空转循环（−0.5 ms，几行改动）。**

---

## 1. 方法与证据：主机微基准（本轮新增，零 GPU）

`tools/bench_host.cpp`（zig 同工具链、`-O2`、无 CUDA）：

| 项 | 实测 | 备注 |
|---|---|---|
| **`std::getenv`**（命中 / 不命中 / 长名） | **2037 / 2091 / 2126 ns** | 72 个 env；`sys`≈0 ⇒ 用户态逐条扫描，不是系统调用；命中位置靠前更快（PATH 1380 ns） |
| `std::chrono::steady_clock::now()` | **37.1 ns** | Win = QPC |
| 逐专家循环骨架（256 轮：2×`assign` + 3×时钟） | **22,250 ns/层** | ×24 层 = **0.53 ms/token**（35B） |
| route 构建（ne=256 softmax + partial_sort top8） | 1,360 ns/层 | ×24 = 0.03 ms ⇒ **不是问题** |
| `gemv_shape_key()`（仅校准缓存命中时） | 46 ns | 不在默认路径 |
| `choose_gemv` 类型名比较 | ~1 ns（-O2 折叠） | 不是问题 |

> 复现：`D:/p0/py312/python.exe -m ziglang c++ -std=c++17 -O2 tools/bench_host.cpp -o build/bench_host.exe && ./build/bench_host.exe`
> （该文件是**新增的独立小工具**，`build.sh` 未引用；不属于引擎代码。）

---

## 2. 确认的浪费（按剩余收益排序）

### 2.1 【#2，Coder −2.0~2.5 ms 潜空间】Coder（MLA）**零 CUDA Graph 覆盖**

三条证据，全部代码可查：

1. `if (ir_.moe.n_expert > 0) use_graph_ = false;`（runner.cpp:100）⇒ Coder 关整图（MoE 需要
   host 路由同步，capture 禁止 sync，设计上如此）。
2. 逐层自动图分支要求 `q35_layer_graph_ok(layer)`（runner.cpp:1585-1586），而它
   `if (!L.q35.attn_kind && !L.is_linear) return false;`（qwen35.cpp:718-720）⇒ **MLA 层直接判否**。
3. `layer_graph_`（config，默认 false）同样只覆盖 qwen35。

⇒ Coder 每 token ≈ 27 层 × 18–27 次裸发射 ≈ **500–700 次 `cuLaunchKernel`**。
本机每次 kick 的实测成本 ≈ **6.9 µs**（OVERHEAD-17MS §6：图替换 ~12 次发射仅省 0.076 ms/层）
⇒ **3.5–4.8 ms/token** 纯提交开销，全部落在 Coder 的 43 ms 里。

**为什么现在可修**：Coder 的 MLA 投影张量在 Q5_K_M 里全为 Q5_K（`blk.N.attn_q / attn_kv_a_mqa /
attn_kv_b / attn_output`，`mla_quant = qa&&qb&&qc&&qd` 为真），所以 `run_mla_attention` 全程只用
`matmul()`（GEMV kernel）+ `rms_raw` + 小 kernel，**没有 cuBLAS**（只有默认关闭的 `mla_q_global_`
分支用 cuBLAS）⇒ 可 capture。

**最小改法**：照抄 `capture_attn_graph`（qwen35.cpp:760-790）写
`capture_mla_attn_graph()`：条件 `L.mla_quant && !mla_q_global_ && 所有 attn 张量非 Q5_1`，
图内容 = `run_mla_attention(layer,1,pos) + rms_raw(x_, L.ln2, hn_, 1, hidden_)`，
每层 1 次 replay。预计 Coder **43 → ~40.5–41 ms（24.5 tok/s）**。
**风险**：中。capture 内出现 cublas/sync 会直接失败；先 `ENGINE_GRAPH_ONLY_LAYER=<n>` 单层验证
replay + gen 序列 md5，并保留 `ENGINE_NO_MLA_GRAPH` 回退。

### 2.2 【#3，35B −0.53 ms】逐专家循环在 `cpu_batched` 之后仍空转 ne=256 轮

runner.cpp **2955–2978**（`run_moe`）：

```cpp
for (int e = 0; e < ne && !fused_done; ++e) {           // ne = 256 (35B)
  ti.assign(...); tw.assign(...);                        // 2 次 vector::assign（批量路径已算过）
  const auto t_iter = std::chrono::steady_clock::now();  // 37 ns
  struct ActiveTimer { ... ~ActiveTimer(){ g_prof_active_ms += now()-t0; } } active_timer{t_iter};  // +37 ns
  if (m == 0) { g_prof_skip_ms += ...now()-t_iter...; continue; }   // +37 ns
  if (e >= gpu_n) { if (cpu_batched) continue; ... }      // ← 主机专家已在上面批量算完
}
```
- 层粒度的 CPU 层 `gpu_n == 0`（`w.cpu_begin = 0`），`cpu_batched == true`（`h_gate_bytes` 非空）
  ⇒ **整个循环没有任何有效工作**，只剩记账 + 时钟；
- 微基准同构造 **22.3 µs/层** × 24 个 CPU 层 = **0.53 ms/token**（放置变化时按 CPU 层数线性缩放；
  本机 35B auto 放置为 16 GPU + 24 CPU，见 OVERHEAD-17MS §3.2）；
- 其中约 70% 是给 `g_prof_active_ms` / `g_prof_skip_ms` 这两个**常开**诊断累加器读时钟。
- 注：GPU 层（dev-route 分支在 runner.cpp:2949 提前 return）不会进这个循环，只影响 CPU 层。

**最小改法**（注意别破坏混合放置的 GPU 专家路径）：
```cpp
// 只有"批量主机路径已覆盖全部专家"时才整段跳过；gpu_n>0 的混合层仍需跑 e<gpu_n 的 GPU 分支
const bool skip_expert_loop = cpu_batched && gpu_n == 0;
if (!skip_expert_loop) { for (int e = 0; ...) { ... } }
```
并把 `ActiveTimer`/`g_prof_skip_ms` 的时钟读收进 `ENGINE_PIPE_PROF` 门控（同 §4 的思路）。
**风险**：低。`[layerms]` 的 host 列会变成 0，需要改注释/口径（真正 host GEMV 计时在
`[cpuexec]`）。verify.sh 里 `ENGINE_GPU_EXPERTS=2` 的混合放置 fixture 会走到 `gpu_n>0`，必须保留。

### 2.3 【#4，35B −0.8~1.0 ms 潜空间】GPU 层 MoE 段不在图里；上次整层图变慢的原因已定位

35B 的 16 个 GPU 层默认只把 **mixer** 录进图（`run_q35_attn_graph`），图 replay 之后 MoE 段仍是
裸发射：`router GEMM(1) + zerofill(1) + 共享专家(6) + route kernel(1) + moe_gateup(1) +
moe_down(1) + add(1)` ≈ **11 次/层 × 16 = 176 次/token ≈ 1.2 ms 提交**。

`ENGINE_GPU_FULL_LAYER_GRAPH=1`（整层图）实测更慢（43.15 vs 41.72 ms，OVERHEAD-17MS §6.1）。
**代码里能看出原因**：`run_gpu_layer_graph()` 进图前置 `layer_graph_mode_ = true`，而

```cpp
const bool can_fuse = fused_env && !layer_graph_mode_ && n_tokens == 1 && gpu_n == ne && ...
```
（runner.cpp:**2764**）——整层图把这个 flag 打开后，**默认的融合专家内核
`moe_gateup_silu`/`moe_down_scatter` 被关掉**，换回逐专家 GEMV（节点更多、更慢），把"省下的提交"
吃掉了。

**v2 方案**：录整层图时**不要**置 `layer_graph_mode_`（35B 的 dev-route 本来就由
`dev_route_default_` 打开，不需要这个 flag），router 强制走 `k_routergemm_`（避免 cuBLAS 进 capture）。
图内 ≈ mixer(15)+router(1)+zerofill(1)+共享(6)+route(1)+融合(2)+add(1)，1 次 replay 替 ≈27 次发射。
**风险**：中（capture 内必须无 sync/cublas；`can_fuse` 依赖的 `all_idx` 在 dev-route 下仍会被算，
需确认）。先 `ENGINE_GRAPH_ONLY_LAYER` 单层验证。

### 2.4 【#5，Coder −0.7~1.0 ms 潜空间】Coder GPU 层 host 路由：1 次 D2H sync + 2 次**同步** H2D

- `dev_route_default_ = !q51_expert`（runner.cpp:109-119）；Coder 26 个 MoE 层里 **14 层**含 Q5_1
  专家（`build/coder_meta.json` 实测）⇒ Coder 整体 `dev_route_default_ = false`。
- 于是 GPU 层也走 host 路由：`router_logits_ D2H → stream_sync（runner.cpp:2586）→ 主机
  softmax/top-k → d_idx_/d_w_ 两次 Buffer::upload()（runner.cpp:2692-2693，**同步**
  cuMemcpyHtoD，WDDM 上等于刷一次队列）→ 融合专家 kernel`。每层 3 次往返，≈17 个 GPU 层。

**两档修复**：
1. **小改**：把两次同步 `upload()` 换成 **pinned 暂存 + `upload_async`**（`all_idx`/`all_w` 是栈上
   vector，异步拷贝前必须先落到常驻 pinned 缓冲，否则悬垂）⇒ −0.3~0.6 ms。
2. **大改**：补验 **Q5_1 设备端 GEMV 分支**（代码注释明说"unverified，所以含 Q5_1 的模型保留
   host 往返"）⇒ Coder 可开 dev-route，删掉 3 次往返/层并解锁 GPU 层整层图，约 −1 ms 以上。

### 2.5 【#6，9B/0.5B −20~40 µs】每 token 两次 `stream_sync` 可合并

runner.cpp **2213-2251**：

```cpp
graph_->launch();
cuda::stream_sync();                       // ① 等整图（21 ms）
...
topk_ov_.download_async(pin, kk*4);        // ② 两次 D2H（流序在 ① 之后）
topk_oi_.download_async(pin+8, kk*4);
cuda::stream_sync();                       // ③ 又等一次
```
② 在流序上必然排在 ① 之后 ⇒ **① 可以删除**：先 enqueue ②（或非 topk 的
`logits_.download_async`）再单次 ③ sync，语义等价。省一次 `cuStreamSynchronize` + 一次 WDDM
往返，且 GPU 结束时拷贝已在队列里（不必等主机被唤醒后再提交）。
**风险**：低（`kProfFwd/kProfK` 分支的 sync 不动，只合并"图 replay → 固定 D2H"这一对）。

### 2.6 【#7，35B −0.2~0.3 ms】CPU 层 mixer 图尾部仍有 2 次图外发射

`run_q35_attn_graph`（qwen35.cpp:729-757）replay 后调用 `run_ffn()`，其中
`rms_raw(x_, ln2, …)` 与 `run_moe` 里的 router GEMM 都在图外，各 1 次发射/层（×24）。
这两步都只依赖图内产物、无 sync ⇒ 可把 `rms(ln2)`（以及 router GEMM kernel）录进同一张 mixer 图。
**风险**：中低（需要 `run_ffn` 知道 rms 已在图内做掉，加个 per-layer flag）。

### 2.7 【#8，全模型 −20~50 µs】`x_`/`pos_` 用 pageable 内存做 `upload_async`

- `x_.upload_async(x.data(), …)`（runner.cpp:2127）：`x` 是 `std::vector<float>`（pageable）；
- `pos_dev_.upload_async(&pos_host_, 4)`（runner.cpp:1862）：成员 int（pageable）。

CUDA 对 pageable 的 async H2D 通常是"驱动内部暂存 + 同步语义"，拿不到真正异步。
热路径其余三处拷贝（`hn_`/`router_logits_`/`moe_host_part_`）都已是 pinned ✔。
修：加 `PinnedBuffer x_pin_, pos_pin_`，memcpy 后 async。9B 约 0.1–0.25%，**0.5B 约 0.4–1%**
（4.89 ms 基数小，占比更高）。

### 2.8 【#9，35B/Coder −35~55 µs】常开计时

- `cuda::launch`（cuda.cpp:704、726-728）：每次发射无条件两次 `steady_clock::now()`（74 ns）+
  `g_launch_ms` 累加。35B/Coder 每 token 约 400–500 次裸发射 ⇒ **30–40 µs**。
  修：`static const bool timing = ENGINE_ENV_ON("ENGINE_PIPE_PROF") || ENGINE_ENV_ON("ENGINE_DBG_LAYERMS")`。
- `run_pipeline`（runner.cpp:**1920**）：每 token `cuda::Event pev0, pev1;`（cuEventCreate×2 +
  Destroy×2），但只在 `ENGINE_PIPE_PROF` 时使用 ⇒ 35B 每 token 5–15 µs。
  修：`std::unique_ptr<cuda::Event>`，仅 pipe_prof 时创建。

### 2.9 【#10，35B −0.1~0.4 ms？未测】router GEMM 走 cuBLAS 而非现成 kernel

decode 时 `router_kernel_mode = ENGINE_ENV_ON("ENGINE_ROUTER_KERNEL") || layer_graph_ ||
layer_graph_mode_`（runner.cpp ≈2396）三者皆 false ⇒ 每层 `cublas::gemm_f32_nt(1, ne, hidden, …)`
（runner.cpp:2407），40 次 cuBLAS 调用/token。`k_routergemm_` 已存在且在 capture 路径用过。
未测 cuBLAS M=1 的 host 开销（估计 3–10 µs/次）。**低风险**切换，但需 gen 序列 A/B 确认数值。

---

## 3. 前 5 条最小改动清单（可直接派工）

| 顺序 | 改动 | 位置 | 预期 | 验证 |
|---|---|---|---|---|
| **1** | `capture_mla_attn_graph()`：`mla_quant && !mla_q_global_` 时把 `run_mla_attention + rms(ln2)` 录成一张图/层 | 新函数（照 qwen35.cpp:760 模式）+ runner.cpp:1585 分支前置检查 | Coder **−2.0~2.5 ms** | `ENGINE_GRAPH_ONLY_LAYER` 单层 replay 自检；n=32 `gen:` md5；`ENGINE_NO_MLA_GRAPH` 回退 |
| **2** | `if (!(cpu_batched && gpu_n == 0))` 包住逐专家循环；诊断时钟收进 `ENGINE_PIPE_PROF` | runner.cpp:2955–2978 | 35B **−0.5 ms** | 混合放置 fixture（`ENGINE_GPU_EXPERTS=2`）必须仍过 verify.sh；`[acc]` UNACCOUNTED 下降 |
| **3** | GPU 层整层图 v2：capture 时**不置** `layer_graph_mode_`、router 用 `k_routergemm_` | runner.cpp:2764（`can_fuse`）+ qwen35.cpp:879-899 | 35B **−0.8~1.0 ms** | 单层先验；`ENGINE_NO_GPU_LAYER_GRAPH` 回退；gen md5 |
| **4** | 9B/0.5B 合并图 replay 与 topk/logits D2H 的两次 sync | runner.cpp:2213-2251 | 9B/0.5B **−20~40 µs** | `ENGINE_DBG_FWD` 的 topk-dl/logits-dl 不升 |
| **5** | Coder 路由表改 pinned `upload_async`（常驻缓冲） | runner.cpp:2692-2693 | Coder **−0.3~0.6 ms** | `[layerms]` sync 计数；gen md5 |
| 附 | `x_/pos_` pinned 暂存；`cuda::launch` 计时门控；Event 懒创建；剩余 2 处裸 `getenv`（runner.cpp:1607 / 3186） | — | 各 −10~50 µs | 全模型 tok/s 不降 |

---

## 4. 并发修复确认（避免重复派工）：getenv 已被 `overhead-17ms` 改掉

审计开始（11:38）时热路径裸 `std::getenv` 的静态计数：**35B ≈937 次/token ↔ 1.97 ms**。
审计期间（**11:48:48**）`runner.cpp`/`qwen35.cpp` 被并发修改：新增 `config.h:159 ENGINE_ENV_ON`
（`static const bool v = getenv(name)` 首调用缓存）并把 56+15 处热站点点迁移完成
（`run_pipeline`/`run_layer`/`run_moe`/`run_q35_attn_graph` 全部覆盖）。**我的微基准正好量化了
这次修复的单价收益：2.1 µs × 937 ≈ 1.9 ms/token（35B），≈1.05 ms/token（Coder）。**

**仍剩下的 2 处热路径裸调用**（每次 2.1 µs，合计 ≈84 µs/token，顺手可清）：
- `runner.cpp:1607` `if (gpu_layer && std::getenv("ENGINE_GPU_FULL_LAYER_GRAPH"))` — 35B GPU 层 ×16；
- `runner.cpp:3186` `if (any_cpu && std::getenv("ENGINE_DUMP_MOE_SPLIT"))` — 35B CPU 层 ×24。
（其余 13 处裸调用都在装载/构造/`layer==0` dump 路径，冷路径，无需处理。）

**兼容性备注**：`ENGINE_ENV_ON` 是"首调用缓存"，单进程内再改 env 不生效；本仓库
`tests/`、`tools/` 里没有任何 `setenv/_putenv`（已 grep 确认），三套验证都是"一个用例一个新进程"，
所以安全。

---

## 5. 35B「7.7 ms 未归因」的代码侧候选（静态账）

现有账本只覆盖：`[cpuexec]` 的 host GEMV、`g_launch_ms`（只含 `cuLaunchKernel` 时间）、
`g_sync_ms`（只含 `cuStreamSynchronize`）。**没被任何账本覆盖**的主机侧工作：

| 候选 | 每 token | 依据 |
|---|---|---|
| 逐专家空转循环 + 诊断时钟（§2.2） | **0.53 ms** | 微基准同构造 22.3 µs/层 × 24 |
| `run_moe` 每层 vector 重建（route/idx/pr/all_idx/all_w/exp_off/ag-au-ad/atok/aw/ti/tw） | 0.10~0.20 ms | ~12 次堆分配 × 40 层；exp_off 2048 次循环 |
| `xout_host` 清零 + route/softmax（§1：1.36 µs/层） | 0.03~0.05 ms | 微基准 |
| 40 次 cuGraphLaunch + ≈420 次裸发射的 kick 成本 | 部分在 "launch 2.5 ms" 内 | 若图 launch 单价更高则另计（未测） |
| 24×(2 D2H + 1 H2D) 拷贝提交（pinned，但仍是 3 次/层） | 0.3~0.8 ms？ | 未测（WDDM 每次提交 5–15 µs 量级） |
| 2×Event 创建/销毁 + launch 计时（§2.8） | 0.04~0.06 ms | 微基准 |
| **可静态确认合计** | **≈0.66~0.78 ms** | |

> 说明：getenv 的 ~1.9 ms 曾是这份清单里最大的一项，已被并发修掉；**剩余静态可解释部分约
> 0.7 ms**。若 `[acc] UNACCOUNTED` 仍显著大于 0.7 ms，剩余大头只能在
> cuGraphLaunch 单价 / WDDM 拷贝提交 / GPU 侧间隙里，需要用"图 vs 裸发射"的 microbench 钉死
> （§6 第 1 行）。

---

## 6. 可疑但未测（要 GPU 窗口）

| 项 | 为什么可疑 | 怎么测 |
|---|---|---|
| **cuGraphLaunch 单价** | 35B 每 token 40 次图 replay；若 30–50 µs 则 1.2–2.0 ms，若 8 µs 则 0.3 ms。OVERHEAD-17MS §6.2 的"0.05–0.1 ms"与"省 0.076 ms/层"不自洽 | 同一串 12 kernel：图 launch vs 裸 launch，各 200 次取中位 |
| **`capturing_now()` 每次拷贝/sync 一次 `cuStreamIsCapturing`** | 35B 每 token ~100 次（24×3 拷贝 + 25 sync）；若是 1 µs 级 ⇒ 0.1 ms | 改成 capture_begin/end 置位的全局 bool，A/B |
| **pageable `upload_async` 的真实代价**（§2.7） | 驱动可能内部同步 | 换 pinned 前后 `ENGINE_DBG_FWD` 的 pre 段 |
| **router cuBLAS 开销**（§2.9） | 40 次/token | 切 `k_routergemm_` A/B（gen md5 必须一致） |
| **CPU 层能否软件流水**（host GEMV 与下一层 GPU attention 重叠） | 当前每层 sync 后 GPU 队列为空，GPU 空转 ≈70%（MOE-GAP-REPORT §5） | 需要双缓冲 x/hn；先 2 层原型 |
| **CPU 层 GPU 段能否录成图**（zerofill+共享专家+add3，无 sync） | 每层 9–10 次裸发射；但 `moe_host_part_` 的 async H2D 在 capture 里会被**静默跳过**（cuda.cpp:467 `if (capturing_now()) return;`）⇒ 需要显式 memcpy node 或把 upload 留在图外 | 先测"2 节点图（upload 之外的 add3）"是否比裸发射快 |

---

## 7. 每模型"每 token 主机侧操作"计数（静态，11:52 快照）

| 模型 | 图覆盖 | 裸发射/token | sync/token | 裸 getenv/token | 主机↔设备拷贝/token |
|---|---|---|---|---|---|
| qwen2.5-0.5b | 整图 ×1 | 0 | **2**（可并为 1） | 4（冷） | 2 pageable H2D + 2 topk D2H |
| Qwen3.5-9B | 整图 ×1 | 0 | **2**（可并为 1） | 4（冷） | 2 pageable H2D + 2 topk D2H |
| Coder-V2-Lite | **无** | **≈500–700** | ≈24 | ≈0（已迁移） | 每 GPU 层 2 同步 H2D + 1 D2H |
| Qwen3.6-35B | 逐层 mixer 图 ×40 | **≈420** | 25 | **2 处热路径 ≈40 次** | 24×(2 D2H + 1 H2D) + 2 topk D2H |

（9B/0.5B 是整图 replay ⇒ 逐层主机代码只在 capture/prefill 执行一次，decode 稳态不受影响；
这正是"9B 没有 7.7 ms 未归因、35B 有"的结构性原因。）

---

## 8. 测量纪律（本轮顺带核对）

1. `ENGINE_DBG_LAYERMS=1` 每层一行 stderr 在本机约 200–300 µs ⇒ **40 层给每 token 加 ≈11 ms**
   （OVERHEAD-17MS 已记录）。不要用该开关的逐层和去反推端到端。
2. 新增的 `prof_acc` 埋点已用 `acc_*` 门控（好），请保持"**门控判断在前、时钟读在后**"。
3. `std::getenv` 在本机是 µs 级 ⇒ 任何"只在 debug 用"的 env 判断放在逐层路径上都是真实成本；
   统一走 `config.h: ENGINE_ENV_ON / ENGINE_ENV_STR`。

---

## 9. 本轮产物（未改动引擎代码）

- `docs/HOTPATH-AUDIT.md`（本文）
- `tools/bench_host.cpp`（新增，独立主机微基准，`build.sh` 未引用）
- `build/audit-snap/`（runner.cpp / qwen35.cpp / cuda.cpp / cpu_matvec.cpp / runner.h 只读快照，
  固定本文行号）
- 引擎源码：**零改动**（审计期间 `runner.cpp`/`qwen35.cpp` 的变化来自 `overhead-17ms` 的
  `ENGINE_ENV_ON` 迁移，非本次审计所为）

## 10. 复现

```bash
# 1) 主机微基准（不占 GPU）
D:/p0/py312/python.exe -m ziglang c++ -std=c++17 -O2 tools/bench_host.cpp -o build/bench_host.exe
./build/bench_host.exe

# 2) 逐层主机开销（GPU 窗口空闲时；注意 §8 的打印成本）
ENGINE_PROF_ACC=1 ENGINE_DEC_SKIP=4 ./build/test_gen.exe "<35B>" 760 24
#   关注 [acc] 的 "CPU sub: … rest"（＝本文 §2.2/§5 的候选）与 "UNACCOUNTED"

# 3) 交错 A/B（gen 序列 md5 作正确性门槛）
ENGINE_TOPK=1 ENGINE_DEC_SKIP=2 ./build/test_gen.exe "<35B>" 760 40   # 改动前后各 3 轮
```
