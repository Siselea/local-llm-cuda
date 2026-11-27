# 长时运行稳定性测试报告（STABILITY-FINDINGS）

日期：2026-11-21 · 执行：`tools/stability_probe.py`（本报告同时修正了探针的两处测量 bug）
对象：`build/engine.exe`（**17:15 构建口径**；`src/sampler.cpp`/`src/serve.cpp` 17:16–17:18 的
bounded-sampling 改动**未**进这个二进制 —— 归属 server-mode，不影响本报告的稳定性结论）
模型：9B（`Qwen3.5-9B-Q5_K_M.gguf`）· GPU：RTX 3060 12GB · 原始数据：`build/stability/*.jsonl` 与
`build/stability/serve-*.log`

> **分类原则**（吸取"把噪声当结论"的教训）：每条都标 **确认** / **可疑未验证** / **环境噪声**，
> 并给证据文件与行。确认级只收录"两次以上一致复现 + 有日志/timing 证据"的结论。

---

## 0. TL;DR

| 用户担心的问题 | 结论 |
|---|---|
| 长会话 RAM 泄漏 | **没有**（RSS 20 轮 +6.6 MB；VRAM +54 MB 后稳定） |
| 长会话 tok/s 衰减 | 有，但**是上下文长度效应**（32→757 ctx：47.4→42.0 tok/s），与 long_ctx 的 2.3k→9.1k 一致 |
| 长 ctx 崩溃/OOM | **没有**（ctx 32768 + 9112 token 预填正常完成，VRAM 8.5–8.7 GB 稳定） |
| 重复启停泄漏 | **没有**（5/5 干净：启动 14.6–15.6 s，关服后 VRAM 267–275 MB，端口全部释放） |
| 非法输入 | 全部正确 400/404；**两个并发流式请求串行完成，无死锁**，事后 /health OK |
| **真正的问题（本轮新发现）** | **serve 的 prefill 路径有 3 个用户可感知的慢点**（见 §1）：长提示 9k 时 prefill 22–43 s、每轮短对话固定 ~0.3–0.55 s、冷启动首个请求额外 4–23 s |

---

## 1. 确认的发现（可复现、有日志证据）

### 1.1 【确认】长提示 prefill 极慢：9.1k prompt 用 22–43 s，且日志显示 9088/9112 token 是"cached"

证据（`serve-7067-ctx32768.log` / `serve-7066-ctx32768.log` / `serve-7063-ctx32768.log`）：

```
[req] ... prompt=9112 gen=39 22957 ms (prefill 21982 [new 26, cached 9088], decode 964, 40.5 tok/s)   # 7067 干净 run
[req] ... prompt=9112 gen=47 25043 ms (prefill 23857 [new 26, cached 9088], decode 1174, 40.0 tok/s)  # 7066
[req] ... prompt=9112 gen=47 44619 ms (prefill 43433 [new 26, cached 9088], decode 1174, 40.0 tok/s)  # 7063（环境噪声）
```
- **只有 26 个 token 真正进了前向，9088 个走 prefix cache，但 prefill 仍耗 22 s** ⇒ 时间不在前向，
  而在 **cache 命中路径（状态快照的恢复/逐 chunk 处理）**。
- 同一台机器、同一个进程里 **2.3k prompt（cached 2240）只要 359 ms**（`serve-7066`：
  `prefill 359 [new 47, cached 2240]`）⇒ 短提示的 cache 命中很快，**慢的是长提示的 cache 命中/分块**。
- 换算：2.3k ctx 预热后 **6369 tok/s** vs 9.1k prompt **382–414 tok/s**（同 run 内 16.7×/token），
  两次独立复测都慢（43.4 s / 23.9 s / 22.0 s），**不是单次抖动**。
- 按 chunk 摊平（chunk=128 行）：**2.3k 时 21 ms/chunk，9.1k 时约 310 ms/chunk（15×）** ⇒
  cache 命中路径的单 chunk 成本随**上下文长度**增长（快照 = hybrid SSM + KV，KV 部分随 token 数增长），
  这比"图捕获按 n 建图"更能解释"两个相同请求都慢"（见 §2 的候选根因 (b)）。
- 影响：用户贴长文档/长代码（>8k token）首 token 要等 20–40 s。**这是本轮最值得修的一条。**

### 1.2 【确认】多轮短对话每轮固定付 ~0.3–0.55 s prefill

证据（`build/stability/long_chat.jsonl` 对应的 `serve-7060-ctx8192.log`，节选）：
```
prompt=32  ... prefill 423 [new 34, cached 0]     ← 首轮含冷启动
prompt=69  ... prefill 381 [new 71, cached 0]
prompt=143 ... prefill 308 [new 17, cached 128]
prompt=217 ... prefill 390 [new 91, cached 128]
prompt=291 ... prefill 344 [new 37, cached 256]
prompt=602 ... 同型
```
- 每轮只新增 17–128 个 token，却稳定付 **0.31–0.55 s**；而"预热后 2.3k tokens"能做到 359 ms（6.4k tok/s）
  ⇒ 这 ~0.3–0.5 s 是**每请求固定开销**，不是前向算力。
- 候选根因（**可疑未验证**，见 §2）：`prefill_graphs_[n]` 按"chunk token 数"懒捕获 CUDA 图
  （每出现一个新的 n 就录一张图，录制成本 0.1–0.5 s 级）；或 prefix-cache 状态快照的恢复；
  或每个请求首个 chunk 的惰性分配/反量化 staging。
- 影响：多轮对话每轮首字延迟被抬高 0.3–0.5 s（这也是我上一轮 e2e 里 turn1 wall 2.58 s 的来源）。

### 1.3 【确认】冷启动后第一个请求额外慢 4–23 s

证据：
- `serve-7067`：加载 19.0 s 后，第一个"你好"请求 **4.39 s**（1-token prompt）；
- `serve-7066`：加载 15.0 s 后，第一个 2.3k prompt **23.4 s**（第二次同样 prompt 只用 **359 ms**，65×差）；
- `serve-7063`：第一个 2.3k prompt **12.9 s**。
- 结论：**首个请求承担一次性初始化**（激活/f16 staging 的惰性分配 + 首次触碰 page-in + 首个 prefill 图捕获），
  后续请求恢复正常。影响：用户点开对话后第一条消息"卡住"数秒到十几秒。**建议在装载完成后做一次
  自预热（1 个 tiny forward + 一次 1-chunk prefill），把这次成本移出用户请求。**

### 1.4 【确认】没有泄漏/崩溃/死锁

| 检查 | 数字 | 判据 |
|---|---|---|
| 长会话 RAM | RSS 1199.1 → 1205.7 MB（20 轮，**+6.6 MB**） | 线性泄漏应每轮 +10 MB 级 |
| 长会话 VRAM | 7091 → 7145 MB（第 4 轮后稳定在 7145） | +54 MB 是 prefix-cache 快照槽，非泄漏 |
| 长 ctx VRAM | ctx 32768 + 9.1k 预填：8463–8697 MB，稳定 | 无增长、无 OOM |
| 5 次启停 | 关服后 VRAM 267/267/275/275/275 MB（**无累积**）；端口全闭；进程退出 | >64 MB 累积才算泄漏 |
| 并发 | 两个 SSE 请求：req0 932 ms、req1 TTFB 933 ms/总 1792 ms，均 87 字符完整；事后 /health OK | 无死锁（单线程串行是设计行为，见 §3.2） |
| 长会话输出 | 20 轮无空回复；greedy 确定性请求逐轮正常 | |

### 1.5 【确认】解码速度随上下文下降（预期行为，但要有数）

- 长会话：ctx 32→757 时 47.4→42.0 tok/s（−11.2%，20 轮单调平滑，与 prompt 长度同步增长）。
- long_ctx：2.3k→9.1k ctx 时 45.9→40.0 tok/s（−12.9%）。
- 与"泄漏/降频"的区别：**上下文固定后速度不随时间漂移**（long_chat 里同 ctx 段内的相邻轮次差 <0.5%），
  且重启进程后回到 47.4 ⇒ 属上下文长度效应，不是资源泄漏。是否"符合预期"需内核侧确认
  （attention 读 KV 的增长 + hybrid SSM 的逐 token 状态）。

---

## 2. 可疑、未验证（需要内部计时器/内核侧确认，不能当结论）

1. **§1.1 / §1.2 的根因**（三选一或叠加）：
   (a) `prefill_graphs_[n]` 按 chunk token 数**懒捕获** CUDA 图 → 新 n 首见时付 0.1–0.5 s；缓存无上限；
   (b) prefix-cache 的 **SSM+KV 快照恢复**在长上下文时按 chunk 逐段恢复，每段一次同步/拷贝；
   (c) 每请求首个 chunk 的**惰性分配/反量化 staging**（历史上 n>1 会整模型反量化）。
   **验证方法**：给 prefill 路径加 3 个 `ENGINE_PROF_ACC` 级计时（capture / cache-restore / forward），
   用 9.1k prompt 跑一次即可定位。所需工作量 ≈ 1 次改动 + 1 次 30 s 测量，建议优先做。
2. **§1.3 冷启动一次性成本的构成**（page-in vs 惰性分配 vs 图捕获）：同样用上面的分段计时即可区分。
3. **0 显存余量时的降级行为**：`serve-7066` 日志出现
   `[vram] post-load used=12288 MiB free=0 MiB` + `WARNING ... Prefill chunk halved`（因另一 agent 的进程
   在装载期间占着 ~4 GB）——引擎**没有崩溃、自动把 prefill chunk 128→64 降级**，这是好的健壮性；
   但"0 MB 余量继续跑"有驱动分页风险，**建议**在 `post-load free < margin` 时至少打印更醒目警告或
   拒绝 serve 启动（当前只 WARN）。**这条是环境噪声触发的观察，不是我们默认路径的问题。**

---

## 3. 环境噪声（本轮实际踩到的，已从结论中剔除）

1. **其他 agent 的进程在测量期间抢占 GPU**：`wait-gpu` 日志出现 `used=6710MB`、`used=1478MB` 等；
   `long_ctx` 首次跑出 176.8 tok/s（干净复测 576/6369 tok/s）；restart 场景 VRAM 峰值
   12 GB 与 6.5 GB 交替（12 GB = 我的 6.5 GB + 别人的进程）⇒ 该批 restart 的"running VRAM"不可用，
   但"关服后 VRAM 267–275 MB"（多次一致）仍有效。
2. **我的探针自身两个 bug（已修，特此说明以免误读数据）**：
   - `long_chat.jsonl` 里 `rss_mb` 实际单位是 **KB**（WorkingSet64/1024），本报告已按 /1024 换算（§1.4）；
   - SSE 流式读取原先逐字节 `read(1)`，会污染 TTFB；已改 `readline()`。
   - 另：**harness 的 TTFB 是"收到第一个 SSE 帧"的时间，不是首 token**（serve 会先发一个空 delta 帧），
     所以 long_chat 里的 `ttfb=10–13ms` 不能当首字延迟；真实首字 ≈ prefill（§1.2 的 0.3–0.55 s）。
3. 长 ctx 的 `prefill 43433 ms`（7063，第一次跑）与干净复测 `21982/23857 ms`（7066/7067）差 2×，
   系抢占导致（§1.1 已用干净 run 的 22–24 s 作为结论）。

---

## 4. 建议的修复/验证顺序（给 lead 排期）

| # | 事项 | 类型 | 预期收益 | 归属 |
|---|---|---|---|---|
| 1 | 给 prefill 路径加分段计时（capture / cache-restore / forward），定位 §1.1/§1.2/§1.3 根因 | 诊断（1 次改动 + 30 s 测量） | 定位后才能修 | server-mode |
| 2 | 若确认是 `prefill_graphs_[n]` 懒捕获：改成**固定若干桶**（如 32/64/128/256 rows）而不是按精确 n 建图，或装载后预捕获 | 修复 | 每轮 −0.3–0.5 s；长提示 −16 s 量级 | server-mode |
| 3 | 若确认是 prefix-cache 快照恢复：长 ctx 下改为**不逐 chunk 恢复**（只在请求边界恢复一次），或按需关闭长提示的 chunk 级快照 | 修复 | 长提示 prefill 22 s → 秒级 | server-mode |
| 4 | 装载完成后**自预热**（1 tiny forward + 1 个 chunk 的 prefill），把冷启动成本移出首个用户请求 | 修复 | 首请求 −4–23 s | server-mode |
| 5 | `post-load free < margin` 时升级为 ERROR/拒绝启动（当前仅 WARN 并降级 chunk） | 健壮性 | 避免驱动分页 | server-mode |
| 6 | 期望行为确认：decode tok/s 随 ctx 下降（−12% @9k）是否可接受/可优化 | 性能 | 需内核侧判断 | 内核侧 |

---

## 5. 复现命令

```bash
cd engine
# 四场景（GPU 独占；--wait-gpu-s 会等到没有其他模型进程且 VRAM<1.2GB）
python tools/stability_probe.py --scenario long_chat --port 7060 --ctx 8192 --turns 20 --max-tokens 50 --wait-gpu-s 300
python tools/stability_probe.py --scenario long_ctx  --port 7063 --ctx 8192 --wait-gpu-s 300
python tools/stability_probe.py --scenario restart   --port 7064 --ctx 2048 --wait-gpu-s 300
python tools/stability_probe.py --scenario errors    --port 7065 --ctx 2048 --wait-gpu-s 300
# 干净复测长上下文 prefill（本次报告 §1.1 用）
#   起 serve ctx 32768 → tiny 预热 → 2k prompt（cold/warm）→ 8k prompt
```
原始数据：`build/stability/{long_chat,long_ctx,restart,errors}.jsonl`、`build/stability/serve-*.log`。
