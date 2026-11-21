# 长上下文的"首 token 慢 / 每轮固定开销"：定位与修复（2026-11-21）

> 结论摘要
> 1. **22–43 s 的 9.1k cache-hit 慢点没有在独占 GPU 上复现**（实测 0.65 s），此前
>    overhead-17ms 的 22/43/23.9 s 应判定为**跨进程 VRAM 超订 / WDDM 换页污染，已撤回**。
> 2. **多轮每轮固定 0.2–0.5 s 的真实来源是"短 tail prefill 的固定成本"**：9B 上
>    52-token tail = 233 ms（4.5 ms/tok），而 128-token chunk ≈190 ms（1.5 ms/tok）——
>    固定成本来自 prefill 路径每次 forward 都要**把整个模型权重 dequant 成 fp16**
>    （9B 注释实测 ~95 ms/chunk）+ 小 n GEMM 占用率低。
> 3. 因此落地：**post-load VRAM guard**（拒绝超订启动，防 3 tok/s 灾难）、**post-load
>    prewarm**（+0.44 s 启动，首请求约 −0.5 s，收益小但保留）、**prefill 四段计时**
>    （`ENGINE_SERVE_PREFILL_TRACE=1`）。短 tail 的根治需要 runner 侧"小 n 跳过 fp16
>    dequant 的量化 GEMM 路径"（A/B 计划见 §5，归 edge-kernels 的 runner.cpp 范围，待协调）。

## 1. 独占 GPU 复测（9B Q5_K_M，ctx32768）

条件：GPU 仅本进程（~9 GB），`ENGINE_NO_PREWARM=1`，rows=128，`ENGINE_SERVE_PREFILL_TRACE=1`：

| 请求 | wall | prefill | restore | save | chunk forward | tail forward | cached/new |
|---|---|---|---|---|---|---|---|
| 冷启动首次 9012-token prompt | **19.0 s** | 18.6 s | 0 ms | 2.5 ms | 18.37 s（8960 tok，488 tok/s） | 233 ms（52 tok） | 0 / 9012 |
| 相同 prompt 再请求 | **0.65 s** | 234 ms | **0.6 ms** | 0.6 ms | 0 | 233 ms | 8960 / 52 |
| 再请求 | 0.65 s | 233 ms | 0.6 ms | 0.6 ms | 0 | 233 ms | 8960 / 52 |

对照（污染条件，两 agent 各占 ~7 GB / 共 11.9 GB）：同一 prompt 冷启动 112.8 s、
tail 首次 5.2 s 第二次 233 ms ⇒ **绝对秒数完全不可信**；"首次 5.2 s" 也是换页不是捕获。

## 2. 22–43 s 数据撤回

- `prefill_graph` 配置默认 **false**（`src/config.cpp:299`），`prefill_graphs_[n]` 的懒捕获
  路径默认不触发；且独占复测里新 tail n=52 首次就是 233 ms（不是 5.2 s）。
- 结论：overhead-17ms 记录的 `[new 26, cached 9088]` 22/43/23.9 s 应标注为
  **"跨进程 VRAM 超订/WDDM 换页污染，已撤回"**；需要他们在独占 GPU 下重测确认
  （或直接看新版本 VRAM guard 是否会拒绝那种环境）。

## 3. 多轮固定开销：短 tail prefill

独占实测（9B，每轮 48 token 解码）：

| 轮 | new(cached) tokens | prefill ms | 备注 |
|---|---|---|---|
| 1 | 13 (0) | 236 | 首次 |
| 2 | 34 (0) | 179 | |
| 3 | 95 (0) | 197 | |
| 4 | 157 (0) | 362 | 跨过 128 边界 |
| 5 | 92 (128) | 193 | cache 命中 |
| 6 | 153 (128) | 364 | |

rows=128 total prefill = 1531 ms；rows=512 total = 1568 ms（短对话 <512 时 cache 全不命中）。
⇒ 短多轮对 chunk 512 没有收益；每轮 ~0.2–0.36 s prefill 的主体是 tail 固定成本。

机制（runner.cpp 现状，只读分析）：prefill 走 fp16 tensor-core GEMM 路径
（`src/runner.cpp:3636-3704`），每次 forward 调用都按 slab 把权重 **dequant 成 fp16**
写 `dequant_scratch_`——注释实测 9B ~95 ms/chunk；n 越小，这部分固定成本占比越高。
52 tok：~95 ms dequant + 小 n GEMM 低占用 ≈ 233 ms。

候选修法（待 runner 窗口 A/B，5 分钟）：
1. `ENGINE_GEMM=1`（自研量化 GEMM，直接吃量化权重、无 dequant）在 n=8..128 的耗时
   （`build/tail_prefill_probe.py --env ENGINE_GEMM=1`）；
2. `ENGINE_SKIP_GEMM=1` 归因（跳过 dequant+GEMM 后剩多少）；
3. 若 1 对 n≤64 明显更快（目标 233→≤100 ms），建议 runner 加"小 n 走量化 GEMM"分支
   （n 阈值配置化），serve 侧无需再改。

## 4. 已落地的 serve 侧改动（v0.1.3）

1. **post-load VRAM guard**：`free < vram_margin_mb` 且未 `--allow-oversubscribe` ⇒
   `ERROR` + 明确提示并退出，防止驱动换页导致 3 tok/s。
2. **post-load prewarm**：128-token 预热 prefill + 8-token tail + 1-token decode，
   用公开 `prefix_state_save(0)/restore(0)` 复位 SSM 状态；`ENGINE_NO_PREWARM=1` 关闭。
   实测：启动 +0.44 s（噪声内），冷启动首请求 19.36→19.83 s 差别 ~0.5 s（**收益有限**，
   冷启动大头是首次触碰权重，不是 capture）。
3. **`ENGINE_SERVE_PREFILL_TRACE=1`**：`restore/save/chunk/tail` 四段计时（见 §1 表），
   诊断长上下文问题可直接用。

## 5. chunk 默认值与 cache 边界（结论）

- 长文档：chunk 512–1024 是 1.85× 纯 prefill 提速（prefill-fast 实测）。
- 短多轮：实测无收益，且 512 让短对话完全用不上 prefix cache。
- **建议：cache 对齐粒度（`reuse_pos` 取整）与前向 chunk 宽度解耦**：
  cache 对齐保持 128/256（多轮少 re-prefill），前向 chunk 用 512–1024（长文档提速）。
  需要确认 decoupled 后 cached 路径与全量 prefill 的逐 token 一致性
  （用 `tools/serve_prefix_cache_check.py` 验证），并把对齐粒度做成配置项。
  该方案待 prefill-fast 的 rows 曲线数据一起定，未在本轮实现。

## 6. 复现

```bash
# 3 段计时
ENGINE_SERVE_PREFILL_TRACE=1 ./build/engine.exe serve --model <9B> --ctx 32768 ...
# 冷启动 → cache-hit → 再命中
${PY:-python} -u build/pfdiag_repro.py
# 短 tail 扫描（需 prefix-cache 0 + rows=128）
${PY:-python} -u build/tail_prefill_probe.py
${PY:-python} -u build/tail_prefill_probe.py --tag gemm --env ENGINE_GEMM=1
```


## 7. 短 tail 固定成本：静态分析 + 实验矩阵（2026-11-21 18:50 追加）

### 7.1 已知测量点

| 场景 | 总 prefill | 每 token | 说明 |
|---|---|---|---|
| 128-token chunk（稳态） | ~190 ms | ~1.5 ms/tok | `[fwd]` 稳定段；含每 chunk 的全模型 dequant |
| 52-token tail（cache hit） | 233 ms | **4.5 ms/tok** | 同一请求里更小的 n 反而总时间更长 |
| 9k prompt 冷启动 | 18.6 s | 2.1 ms/tok | 70 个 128 chunk + tail |

关键异常：**52 token（233 ms）比 128 token（190 ms）总时间还长**。两种解释：
- **H1 固定成本**：每次 forward 都要把整个模型 dequant 成 fp16（runner 注释 9B ~95 ms，
  prefill-fast 估 ~137 ms）+ f32→f16 激活转换 + 每 matmul 的 launch 开销；n 越小摊得越薄。
- **H2 小 n GEMM 利用率/分块**：fp16 GEMM 在 m=52 时按 64/128 的 tile padding 或选到更差的
  cuBLAS 算法；也可能 attention 的 split 阈值在小 n 走另一条 kernel。

`build/tail_prefill_probe.py`（`--prefix-cache 0`、rows=128、`max_tokens=1`、n=8/16/32/52/64/96/128）
+ `ENGINE_GEMM=1`（自研量化 GEMM，**无 fp16 镜像**）对照可区分：

* 若 plain 曲线在 n=8..64 基本水平（≈固定成本）且 `ENGINE_GEMM=1` 明显更低
  ⇒ H1 主导，量化 GEMM 可救；
* 若 plain 曲线在 64/96 处出现台阶 ⇒ H2（tile）主导，需要改 tile/阈值而不是换路径。

### 7.2 候选方案评估（按可行性）

| 方案 | 收益 | 风险/成本 | 结论 |
|---|---|---|---|
| A. n≤64 走自研量化 GEMM（无 dequant） | 若 H1 成立，tail 233→~80-120 ms | **prefill 数值路径改变**（非 fp16 GEMM 累加顺序）⇒ KV/后续 token 可能微变；greedy bit-exact 回归需重跑；影响 v0.1.2 的 tail 零拷贝/bit-exact 范围 | **需 lead 决策**：仅非 greedy 或全量? 建议先 A/B 拿数据再定 |
| B. 层主序 prefill（每层权重 dequant 一次、所有 token 复用） | 根治：长文档 + tail 都省掉重复 dequant（固定成本 ~137ms×请求次数 → ×1） | 需重构 forward 循环顺序（跨 chunk 持有全部 token 激活/attention），runner.cpp 大改 | **下一轮正道**（prefill-fast 已提） |
| C. 把 tail 并入更大 chunk（重算历史填满 512/1024） | 摊薄固定成本 | 额外重算历史 token（512 填满约 +0.4 s compute），净亏 | 放弃 |
| D. 缓存部分权重 fp16 镜像（lm_head/attention） | 减少每次 dequant 量 | 9B ctx32768 free 仅 ~4.1 GB，最多缓存 ~1B 参数（<1/9 模型），收益小 | 低优先 |
| E. cache 边界解耦（cache 128/256、前向 chunk 512/1024） | 长多轮少 re-prefill，间接减少 tail 次数/大小 | 需 bit-exact 验证 + 配置化 | 与 prefill-fast 联调，下一轮 |

### 7.3 v0.1.3 交付范围

- 不实现 A/B（等数据 + lead 决策）与 B（下一轮）；
- 交付：`ENGINE_SERVE_PREFILL_TRACE` 计时 + post-load prewarm + **VRAM guard**；
- prewarm 的收益已实测有限（启动 +0.44 s，首请求 −0.5 s）：保留但文档写清"不是冷启动解药"。


## 8. v0.1.3 窗口实测（2026-11-21 19:00，9B ctx32768/4096）

### 8.1 VRAM guard（`build/validate_vram_guard.sh`，0.5B）

- case A（`--vram-margin 100000`）：**PASS**，serve 拒绝启动并打印 "below the ... margin"；
- case B（默认 margin）：**PASS**，正常启动；
- case C（大 margin + `--allow-oversubscribe`）：**首轮 FAIL** —— 超订 override 让 loader 把
  rows/ctx 压到 1，**prewarm 用 chunk=1 调 forward 抛 `forward: bad token count` 直接退出**。
  已修：prewarm 改为 **best-effort**（chunk<16 / ctx<chunk+16 时跳过；forward 全部包 try/catch，
  失败只 WARN 继续启动）。修复后复测见 §8.5。

### 8.2 prewarm + 新 chunk 默认（9B ctx32768，rows 自动长到 1024）

| 请求 | wall | prefill | new/cached |
|---|---|---|---|
| 冷启动 9012-token（prewarm 开） | **10.15 s** | 9.71 s | 9012 / 0 |
| 同 prompt 再请求 | 1.64 s | **1.20 s** | **820 / 8192** |

对比 chunk=128 时代（§1）：冷启动 19.0 s（慢 2×），但 cache-hit 只要 0.65 s（tail 52 tok）。
⇒ **chunk 512/1024 把长 prompt 首次 prefill 提速约 1.9×，代价是 cache 边界粒度变粗**：
9.1k 对话每轮 tail 从 52 → 820 token，prefill 从 233 ms → **1.2 s/轮**。这是
"长文档 vs 长多轮"的明确取舍，支持 lead 的判断——**cache 对齐粒度必须与前向 chunk 解耦**
（cache 128/256 + 前向 512–1024）；需要 `tools/serve_prefix_cache_check.py` 验证解耦后
cached/fresh 输出一致，v0.1.4 做。

### 8.3 短 tail 固定成本 probe（`build/tail_prefill_probe.py`，prefix-cache 0、rows=128、n=8..128）

| 路径 | 拟合 | 结论 |
|---|---|---|
| 默认 fp16 | **fixed 179.8 ms + 0.124 ms/token** | n=52 ≈ 186 ms，固定成本占 96% |
| `ENGINE_GEMM=1`（自研量化 GEMM，无 dequant 镜像） | fixed **381.3 ms** + 0.581 ms/token | **更差**，小 n 不可行 |
| `ENGINE_PREFILL_F16=0` | fixed **179.6 ms** + 0.124 ms/token | 与默认相同 ⇒ 固定成本不在"f16 dequant"这一个开关上，还有每 chunk 的其它固定开销（launch/转换/attention 准备等） |

⇒ 已现成开关（gemm/f16）**无法把 tail 233 ms 降到 100 ms**；根治仍是 prefill-fast 提的
**层主序 prefill**（每层权重只 dequant 一次）或改变小 n 的 kernel 组织，属于下一轮 runner 改造。
v0.1.3 不实现。

### 8.4 浏览器 e2e（9B serve 7042，3 轮）

`build/logs/e2e_v013/`：3 轮真打字+真鼠标全部完成（截图 01–03_done.png），
webui 默认 top_k=20 路径正常（采样候选优化生效）。

### 8.5 修复后 VRAM guard 复测

`bash build/validate_vram_guard.sh` → **GREEN**：

- case A（`--vram-margin 100000`）：PASS —— placement 先拒绝（`[placement] ERROR: model does not
  fit ...`，exit 3）；即使 placement 放过，post-load guard 也会二次拦截。v0.1.3 的 guard 是
  **第二道防线**（placement 之后显存还可能因 graph/其它进程增长）。
- case B（默认 margin）：PASS，正常启动。
- case C（大 margin + `--allow-oversubscribe`）：PASS，override 可启动（prewarm 改为 best-effort 后
  不再因 rows 被压成 1 而崩溃）。

### 8.6 五套验证（v0.1.3 工作树，19:02–19:17）

`HEAD=d426b1c`（工作树 dirty=13，含本 WIP/他人文件；commit 后状态见下）：
build.sh **OK**、verify **ALL GREEN**、smoke_models **PASS**、gen_regression **10/10 GREEN**、
verify_generality **26/26 GREEN**、verify_serve 0.5B **22/22** / 9B **21/21 GREEN**、
test_sampler **PASS**。


## 9. v0.1.4：cache 对齐粒度与前向 chunk 解耦（2026-11-21 19:30）

### 9.1 实现
`serve_cache_align`（CLI `--cache-align`，默认 **128**；`0` = 旧行为=对齐到 prefill chunk）：
`reuse_pos = (boundary / cache_align) * cache_align`，其余 snapshot/restore/start_from 逻辑不变。
缓存路径的历史前向可能出现 **partial chunk**（长度不是 rows_ 的倍数），这是本轮 bit-exact
验证的重点路径。只改 `serve.cpp` + `config.cpp`（一行配置），不碰 runner.cpp。

### 9.2 bit-exact 硬门槛（先验证再上默认）
9B ctx32768、greedy（temp=0/top_k=1）、5 轮增长对话（每轮 +~2k token，历史到 8.3k），
`cache_prompt=false`（整段 prefill） vs `cache_prompt=true`（缓存路径）逐 token 对比：

| 轮 | prompt | 整段 prefill | 缓存 prefill | new/reused | tokens |
|---|---|---|---|---|---|
| 1 | 24 | 234 ms | 163 ms | 24 / 0 | SAME |
| 2 | 2092 | 1670 ms | 1672 ms | 2092 / 0 | SAME |
| 3 | 4162 | 3520 ms | 2030 ms | 2114 / 2048 | SAME |
| 4 | 6229 | 5731 ms | 2406 ms | 2133 / 4096 | SAME |
| 5 | 8291 | 8281 ms | 2771 ms | 2147 / 6144 | SAME |

⇒ **PREFIX CACHE GREEN**（含 partial chunk 路径），与对齐 128 的显式 `--cache-align 128`
结果相同；默认值置 128 后再跑一遍仍 GREEN。

### 9.3 A/B：turn-2 省多少（9B ctx32768，~5.8k prompt，3 次交错）

| align | turn1 prefill | turn2 prompt | turn2 new | turn2 prefill | turn2 wall |
|---|---|---|---|---|---|
| 0（旧） | 5191 ms | 5814 | **694** | **832 ms** | 1.21 s |
| 128（新默认） | 5331 ms | 5814 | **54** | **210 ms** | **0.59 s** |

⇒ **每轮省 622 ms prefill（−75%）、wall −0.62 s（−51%）**；长文档首次 prefill 不变
（大 chunk 仍在）。短/中多轮（历史 ≥128）也开始能命中缓存（旧对齐 1024 时 <1024 历史完全
用不上缓存）。

### 9.4 验证
- `tools/serve_prefix_cache_check.py --filler-repeat 100 --turns 5`：GREEN（默认 128）；
- 五套验证（`build.sh` 重建后的 engine.exe，19:32–19:45）：
  `verify` **ALL GREEN**、`smoke_models` **PASS**、`gen_regression` **10/10 GREEN**、
  `verify_generality` **26/26 GREEN**、`verify_serve` 0.5B **22/22** / 9B **21/21 GREEN**、
  `test_sampler` **PASS**；`build.sh` **OK**。
- 浏览器 e2e：9B serve（默认 cache_align=128）3 轮真打字+真鼠标全部完成，
  截图 `build/logs/e2e_v014/01–03_done.png`。
- 工作树口径：验证开始时 `HEAD=2aa850e dirty=14`，结束时 `HEAD=b6efdc1 dirty=15`
  （**其他 agent 在此期间提交/改文件**；build.sh 在 19:32 产生 engine.exe 后未再重建，
  所有 suite 跑的都是同一份二进制；本项改动只有 serve.cpp/config.cpp 的 cache-align hunk）。


## 10. v0.1.4 第二项：非 greedy 采样走 device top-k 候选（2026-11-21 20:xx）

### 10.0 为什么 serve 用 `forward_topk()` 而不是 `forward_topk_candidates()`
`forward_topk_candidates()` 在返回 false（设备头不可用）之前已经执行了一次
`forward()`，调用方如果再用整行 `forward()` 回退，会把同一个 token 前向两次、破坏
KV/SSM 状态。serve 因此直接用公开的 `forward_topk(tokens, pos, k, &tk)`：
设备头可用时它返回空 row + 候选；不可用时返回**完整 row**（此时直接用 row 采样），
天然无二次前向。`forward_topk_candidates()` 仍由独立验证程序使用（只读调用一次）。

### 10.1 背景与语义
`forward_topk_candidates(k)` 在设备端把 248320 维 logits 归约成 k 个 `(id, logit)` 候选，
host 只收 k*8 字节（k=64 时候选 512 B vs 整行 ~1 MB）。serve 只在 **top_k>0 且
max(top_k,64)<=head_topk_max** 时使用；`top_k=0`（host top-1024 近似）和
`top_k>head_topk_max` 回退整行，保证采样语义不因候选不足而截断。
v0.1.4 只提交接线：**保持 stock 默认 `head_topk_max=8`**，此时典型 non-greedy 请求
（top_k=20 ⇒ needed=max(20,64)=64 > 8）自动回退整行路径，**行为与之前完全一致**；
k=64 的正确性已验证（10.2），但性能和采样质量 A/B 后才会在 v0.1.5 把默认提到 64。

### 10.2 正确性验证（独立于 serve）
`build/validate_topk_candidates.cpp`（提交版在 `tools/`）：prefill →
`prefix_state_save(0)` → device `forward_topk_candidates(k)` → `restore(0)` → 整行
`forward()` → 对比 k=1/5/8/64/128；另做"两次 prefill+decode from state0"的确定性检查。

判据：**候选 id/顺序与 host 参考完全一致 + 值差 ≤2e-3**。原因：9B hybrid 的 decode
路径本身不是逐位确定的——两次相同输入的 logits 差 **~1.0–1.8e-3**（host-vs-host restore
后 ~6e-4），0.5B/Coder 为 **0**。这是 split-k/atoms 求和顺序的量级，对 argmax/token 级
bit-exact（缓存 A/B、gen_regression）没有影响，但"logits 逐字节一致"这类断言需要改成
"逐 token 一致（argmax 稳定）/数值差 ≤ 噪声量级"。

实测（`ENGINE_HEAD_TOPK_MAX=64/128`）：

| 模型 | k=1 | k=5 | k=8 | k=64 | k=128 | 结果 |
|---|---|---|---|---|---|---|
| 9B Qwen3.5 (hybrid) | ✓ Δ4.8e-5 | ✓ Δ3.2e-4 | ✓ Δ3.3e-4 | ✓ Δ3.1e-4 | ✓(128) Δ3.5e-4 | **GREEN** |
| 0.5B Qwen2.5 | ✓ Δ0 | ✓ Δ0 | ✓ Δ0 | ✓ Δ0 | SKIP(>64) | **GREEN** |
| Coder-V2-Lite (MLA) | ✓ Δ0 | ✓ Δ0 | ✓ Δ0 | ✓ Δ0 | SKIP(>64) | **GREEN** |

⇒ 设备 top-k 候选在 dense/MLA/MoE 三类头上都正确；serve 默认 `head_topk_max=64` 覆盖
webui 默认 top_k=20/top_p=0.95。


### 10.3 A/B（9B 采样）
（填：before/after temp0.6/top_k20 decode tok/s、wall；top_k=0 组应不变）

### 10.4 验证（v0.1.4 交付范围：只接线，默认 k=8）
- validator 三模型 GREEN（见 10.2，当前源码重编后 9B 复跑仍 GREEN）；
- `build.sh` + verify/smoke/gen_regression/generality/verify_serve + test_sampler：见 10.5；
- **k=64 的性能/采样质量 A/B 与把默认提到 64 留到 v0.1.5**（v0.1.4 保持 stock 默认 8，
  典型 top_k=20 请求自动回退整行路径 ⇒ 行为与之前完全一致，零风险）。

### 10.5 五套验证（v0.1.4 工作树，当前源码，21:30–21:44）

`build.sh` **OK**（engine.exe 2,395,648 B，21:30）；`verify` **ALL GREEN**；`smoke_models` **PASS**
（40716/185/2614/2614）；`gen_regression` **10/10 GREEN**；`verify_generality` **26/26 GREEN**；
`verify_serve` 0.5B **22/22**、9B **21/21 GREEN**；`test_sampler` **PASS**。
验证时 `HEAD=89e8b3a dirty=15`；提交后工作树只剩他人改动。


## 11. v0.1.5 计划中的 A/B：device top-k k=64 是**净亏**（2026-11-21 21:57，负结果）

`98f4a9d`（v0.1.4）只接线、保持 stock `head_topk_max=8`。计划 v0.1.5 把它默认提到 64，
因此先做了**同二进制、只改 env** 的交易对照（避免把 prefill x8/chunk 收益混进来）：

| 场景 | k=8（off） | k=64（on） | 差值 |
|---|---|---|---|
| greedy decode（temp0，128 tok） | **45.04 tok/s / 22.20 ms/tok** | **35.69 tok/s / 28.02 ms/tok** | **+5.8 ms/token（−26%）** |
| 采样 decode（temp0.6/top_k20） | 42.42 tok/s / 23.57 ms | 38.56 tok/s / 25.93 ms | +2.36 ms/token（−10%） |
| 采样 decode（top_k=0，应走整行回退） | 41.45 tok/s / 24.13 ms | 36.93 tok/s / 27.08 ms | +2.95 ms/token（−12%） |

（greedy 两次的 generated_token_ids **完全相同**，排除采样差异。）

**机制**：`head_topk_max` 同时决定 decode CUDA graph 里 `topk_partial/topk_final` 的 kk；
topk 内核按 k 轮选择（每轮扫一遍本 block 的 248k/nblk 个 logits），k=64 的代价（~2.4–5.8 ms/token）
**远大于**省掉的 1 MB D2H + host 扫描（~0.3 ms），而且对 greedy 和回退到整行的请求同样生效
⇒ 提高 default 是负优化。

**结论/下一步**：v0.1.5 **不启用** `head_topk_max=64`；保持 stock 8（top_k>8 自动回退整行）。
若要启用设备候选，需要先优化 device top-k 内核（例如 k≤8 保持、k>8 改用一次扫描的
阈值/基数选择，或让 graph 按请求配置选 kk），这属于 runner/kernels 侧（edge-kernels）；
serve 侧接线与 `sample_external` 已就绪，默认值一行即可切换。
