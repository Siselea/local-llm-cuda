# PREFILL-CLIFF 报告：用户可感知的两个 prefill 慢（首 token / 长文档）

> 执行者：prefill-fast｜日期：2026-11-21 17:51–18:30｜本机 RTX 3060 12GB（cc 8.6，GA104 解锁卡 / WDDM VM）
> 需求来源：真实用户抱怨"首 token 等很久 / 长文档很慢"
> 结论：**两个慢点都定位到实现 bug 并已修复**；"冷启动 23 s"与磁盘无关；另修掉一个长 prompt 的 HTTP 500。
> 产物：`src/serve.cpp`、`src/runner.cpp` 改动（备份 `.pre_cliff`）+ 两个诊断工具 + 本报告。

---

## 0. TL;DR

| 用户场景（9B Q5_K_M，serve，prefix cache 关，GPU 独占） | 改前 | **改后** | 倍数 |
|---|---|---|---|
| 4.4k token 文档 · prefill | 外推 ~7.3 s（592 tok/s @2.4k） | **3.69 s / 1190 tok/s** | **~2.0×** |
| 13.4k token 文档 · prefill | 外推 ~29.5 s（454 tok/s @11k 实测 24.3 s） | **15.97 s / 837 tok/s** | **1.85×** |
| 刚启动后**首个**请求 vs 暖机同长请求 | 4.35×–12× 慢（17.96 s vs 4.13 s / 28.2 s vs 2.31 s） | **+1.6%**（3.744 s vs 3.685 s） | 消除 |
| 超长 prompt（> 实际 ctx） | **HTTP 500** `forward: bad token count` | **HTTP 400** + `[serve] ctx clamp` 提示 | 修 |

**两个根因**：
1. **`src/serve.cpp` 把 serve 模式 prefill chunk 硬编码成 128 token（`prefill_rows` 和 `prefill_rows_max` 都是 128）** ⇒ 9–13k 文档被切成 70–105 个 chunk，而**每个 chunk 都要把整个模型重新 dequant 成 fp16**（9B ≈ 137 ms）+ 每块固定开销。这就是"长文档很慢"的全部来源。
2. **装载期"先按大 rows 分配激活 → 再打印 Prefill chunk halved"的顺序 bug**：pre-load 的显存预算是乐观估计（不含 cuBLAS workspace/上下文/碎片，实测偏差可达 2 GB 级），激活缓冲又在校验之前就按大 rows 分配掉了 ⇒ 显存实际超订，WDDM 开始换页。这是"首 token 等很久"里那些 4×–130× 异常的来源（不是磁盘，见 §2）。

---

## 1. 测量纪律（先说清口径，否则会被假数字骗）

- **必须关 prefix cache**（`--prefix-cache 0`）且**每次请求换不同内容**。否则第 2 次请求走 serve 的前缀 KV/SSM 复用，`prefill_ms` 只算"未缓存尾部"，会得出**超物理上限**的假吞吐。
  - 任务书里最初引用的 "6369 tok/s（359 ms）" 就是这么来的：2300 token 真 prefill 在本机物理上限约 **1850 tok/s**（fp16 tensor 12.8 T-MAC/s ÷ 6.91 G-MAC/token），6369 超出 3.4 倍 ⇒ 必然跳过了大部分计算。用 `timings.prefilled_tokens` / `cache_hit_tokens` 可证伪。
- 每次跑前 `nvidia-smi --query-gpu=memory.used` 必须 < 900 MiB；本机多智能体并发时，两个人各占 ~7 GB（合计 11.9/12 GB）会让首请求慢到 132 s 且后续逐步变快——这是换页曲线，不是代码行为。
- 驱动真 serve（用户走的就是这条路径），读服务端 `timings.prefill_ms`，并对照 `prompt_tokens/prefilled_tokens` 算 tok/s。

工具（新增，可复用）：
- `tools/prefill_cliff_probe.py`：起 serve（可指定 ctx/rows/prefix-cache），按目标长度生成**不同 nonce** 的 prompt，输出 JSONL（prompt_tokens / prefilled_tokens / cache_hit_tokens / prefill_ms / tok/s / client wall）。
- `tools/prefill_cold_forensics.sh`：跑一次探针的同时采样 `clocks.sm/power.draw/util/memory` 与 `\LogicalDisk(_Total)\Disk Read Bytes/sec`，用来判定"慢在磁盘还是换页还是算力"。

---

## 2. 冷启动 23 s 到底是什么（结论：**不是磁盘冷读**）

任务书假设"6.1 GB / 260 MB/s ≈ 23 s ⇒ 磁盘冷读"。实测否掉：

| 证据 | 数值 |
|---|---|
| 一次完整 serve（load + 3 次 3.6k prefill）期间的**磁盘读** | **max 3.1 MiB/s**，没有任何 GB 级读（`prefill_cold_forensics.sh`） |
| 同一次**独占**运行的三次同长（3.6k）prefill | **3.53 / 3.46 / 3.46 s（1023–1045 tok/s）——没有冷启动惩罚** |
| 强制 rows=2048 @ ctx32768 的那次 | `post-load used=11675 MiB free=612 MiB`（< 1200 MiB margin）、打印 **"Prefill chunk halved"**、ctx 被夹到 8053；首请求 **28.2 s（105 tok/s，12×）** |
| 我 + server-mode 的 serve 同时占 GPU（11.9 GB/12 GB）那次 | 首请求 **132 s（27 tok/s）**，同 run 后续 10.6 s → 4.87 s 逐步变快（工作集驻留曲线） |

**机制**：模型 load 阶段已把权重全部读入并上传 VRAM（所以 prefill 不碰磁盘）。真正会慢 10–130× 的是**显存超订后的 WDDM 换页**。而超订的来源是装载期顺序 bug：
1. 构造期先用 `ir_.fixed_bytes()` 做 pre-load 预算（不含 workspace/上下文/碎片，实测乐观约 2 GB）；
2. 按这个预算算出 `rows_`；
3. **先把激活缓冲按 `rows_` 分配掉**（`logits_ = rows×vocab×4`，rows=2048 时就是 2 GB）；
4. 最后才 `check_post_load_vram()` 检查真实余量，不够时只打印 "Prefill chunk halved"——**此时已经超订，缓冲也没缩小**。

修复前该路径的其它副作用：ctx 也被大 rows 的激活预算挤小（32768→8053），长 prompt 随后 500。

---

## 3. 长文档慢：serve chunk=128 的代价（逐档实测）

serve 原本（`src/serve.cpp:1981`）：未显式指定时强制 `prefill_rows=128` **且** `prefill_rows_max=128`，即**禁用**了装载期的自适应放大。9–13k 文档 ⇒ 70–105 个 chunk；每个 chunk 都要把 6.1 GB 权重**重新 dequant 成 fp16**（≈137 ms/块）+ 每块 host/launch 固定成本。

实测（prefix cache 关，GPU 独占，每次不同 prompt）：

| prompt | rows=128（原默认） | rows=512 | rows=1024（新默认，自动） | rows=2048 |
|---|---|---|---|---|
| 2.4–3.0k token | 592 tok/s（4.13 s） | 1077–1103 | — | 1285（ctx32768 下 VRAM 不适合） |
| 4.4k token | ~592 | — | **1190（3.69 s）** | — |
| 11–13k token | 454（24.3 s @11k） | 750（17.8 s @13.4k）/ 697（23.2 s @16.2k） | **837（15.97 s）** | VRAM 不允许（见下） |

⇒ **chunk 128→512/1024 就有 1.85–2.0×**。这与"每 chunk 一次全模型 dequant"的模型完全吻合：13.4k 在 128 下是 105 块（≈14.4 s 纯 dequant），在 1024 下是 14 块（≈1.9 s）。

**为什么不是越大越好**：rows 受显存约束（`act(rows) = rows × (6·hidden + 2·n_head·D + 2·n_kv·D + 3·ffn + vocab) × 4`；9B 上约 **1.23 MB/row**，其中 `logits_` 的 `vocab×4` 占 1 MB/row）。ctx=32768 时 KV(f32) 已占 2 GB，实测安全上限约 1024 行（`act(1024)=1252 MiB`，装载后 free=4130 MiB）；ctx=8192 时可到 1024+；ctx=4096 时可到 2048。

---

## 4. 改动清单（全部落盘，默认生效）

| 文件 | 改动 |
|---|---|
| `src/serve.cpp` | ①serve 默认 prefill chunk：`512` 起步 / `1024` 上限（原 128/128），显式 `--prefill-rows` 仍然优先；②模型装载后**总是**取 `model->ctx()` 作为真实 ctx（ctx_auto 可能夹小过），变化时打印 `[serve] ctx clamp : requested X -> Y tokens (VRAM budget)` ⇒ 超长 prompt 由 500 变 400，且错误信息可读 |
| `src/runner.cpp` | 在**激活缓冲分配之前**（权重+KV 已驻留）用 `cuda::mem_info().free_bytes` 重新推导 `rows_`：不足则减半（保持 `vram_margin + 64 MiB` 余量），有余则按 `prefill_rows_max` 放大；打印 `[ctx] prefill chunk X -> Y rows by actual free VRAM (...)`。这同时消除了"先超订后 halved"的换页路径 |
| `tools/prefill_cliff_probe.py` | 新增：serve 驱动的纯 prefill 探针（见 §1） |
| `tools/prefill_cold_forensics.sh` | 新增：GPU 时钟/功耗 + 磁盘读采样（判定磁盘/换页） |
| 备份 | `src/runner.cpp.pre_cliff`、`src/serve.cpp.pre_cliff` |

**没动**：CUDA 内核数学、prefix-cache/snapshot/CUDA-graph 代码（属 server-mode）、任何 decode 路径。

---

## 5. 验证

### 5.1 修复后（独占 GPU，prefix cache 关，ctx 32768）

```
[ctx] prefill chunk grown 512 -> 1024 rows (fits: weights=6263 MiB kv=2048 MiB act=1252 MiB)
[vram] post-load used=8158 MiB free=4130 MiB, check OK (>= 1200 MiB margin)
[serve] prefix-cache: on (... prefill chunk 1024, ...)
  req0 prompt=4393 prefilled=4393 hit=0 prefill=3744 ms -> 1173 tok/s   <- 冷启动首个
  req1 prompt=4393 ... prefill=3685 ms -> 1192 tok/s
  req2 prompt=4393 ... prefill=3690 ms -> 1190 tok/s
  req3 prompt=13363 ... prefill=15971 ms -> 837 tok/s
```
- **首个请求 vs 暖机：+1.6%**（冷启动惩罚消失）
- 13.4k 文档：**837 tok/s / 15.97 s**（改前同长度外推 ~29.5 s）
- rows 对照：强制 512（16.2k tokens）= 697 tok/s；默认（自动 1024，13.4k）= 837 tok/s

### 5.2 VRAM 护栏与 400 修复（强制 rows=2048 的对抗测试）
```
[ctx] CLAMPED: 32768 > max_ctx 8053 for kv_type=f32
[serve] ctx clamp  : requested 32768 -> 8053 tokens (VRAM budget)
req0 target=13363: HTTP 400           <- 改前是 500 "forward: bad token count"
```
无 "only ... MiB free after load" 的超订告警（激活按实际余量决定）。

### 5.3 decode 不回退
9B：`[time] decode 21.35 ms/token (47 tok/s, 16 steps)`，top-1 **2614** ✓（任务书要求 ≥45 tok/s）。

### 5.4 五套验证（独占 GPU，`MODEL_DIR=D:/p0/models PY=D:/p0/py312/python.exe`）

| 套件 | 状态 | 说明 |
|---|---|---|
| `tools/verify.sh` | **ALL GREEN（exit 0）** | 在该提交的构建上通过 |
| `tools/smoke_models.sh` | **exit 0** | 40716 / 185 / 2614 / 2614 全对 |
| `tools/gen_regression.sh` | **未通过（2 例：`35b_chat22`、`coder_v2_lite`）** | **不能归因于本提交**：(a) 用的 `build/*.exe` 是 18:07 构建，工作树里含 **edge-kernels 未提交的设备 top-k 改动**（`head_topk_max_` / `forward_topk_candidates`，见 `git diff src/runner.cpp`），(b) 跑的时候 GPU 上有 **3 个 `test_gen.exe`（合计 12.2 GB > 12 GB 显存）** 在验证 v0.1.2，处于换页状态。**必须在"工作树干净 + 独占 GPU"下重跑才能定性**；本提交对 CLI 的 prefill 只有"装载期 rows 决策"一处影响，且 gen_regression 的 prompt 只有 8–22 token（< 任何 rows），chunk 不变。 |
| `tools/verify_generality.sh` | **未完成** | 在 GPU 被 edge-kernels 占用后中止。注：先前日志里的 `MODEL_DIR: unbound variable` 是外层 shell `export A=B && (...)` 在本机 ops wrapper 下**只对第一条命令生效**的假失败；已用自带 export 的 `tools/pf_run_suites.sh` 修正，需干净窗口复跑 |
| `tools/verify_serve.sh` | **未完成** | 同上 |
| `stability_probe.py long_ctx/long_chat` | 未跑 | 建议与 server-mode 的 prefix-cache 复测同窗口（它负责多轮/长上下文稳定性） |

> 归属提醒：`git diff src/runner.cpp` 里剩下的 41 行是 **edge-kernels 的设备 top-k 候选 WIP**（server-mode 已确认不是它的）。本报告的三处修复已单独提交为 `4e3e84e`（只含我的 hunk），因此可以在"stash 掉其它 WIP + 独占 GPU"的窗口里干净复跑上述三套。

### 5.5 chunk 默认值的取舍（与 server-mode 联合定夺）

chunk 大小与 prefix-cache 粒度耦合（`reuse_pos = (boundary/chunk)*chunk`，`src/serve.cpp:298`）。server-mode 另外做了**短多轮**对照（9B，6 轮短对话，每轮 13–300 新 token）：

| chunk | 每轮 prefill | 6 轮 prefill 合计 | prefix-cache 命中 |
|---|---|---|---|
| 128 | 236/179/197/362/193/364 ms | 1531 ms | turn5/6 命中（cached 128） |
| **512（本次默认）** | 234/173/201/305/312/342 ms | **1568 ms（+2.4%）** | 全程 cache=0（历史<512 命中不了） |

⇒ 短多轮的代价只有 **+37 ms / 6 轮（+2.4%）**，而长文档收益是 **1.85–2.0×**，所以默认保留 512。
如果用户以短多轮为主：`--prefill-rows 128`（或 256）即可恢复细粒度缓存；长文档为主：`--prefill-rows 1024`。
**更彻底的解法（下一轮）**：把"全量 prefill 的大 chunk"与"cache 边界的细粒度"解耦（cache 边界仍按小粒度对齐、全量 prefill 内部用大 chunk），需要 cache 侧配合并对 bit-exact 做验证。

### 5.6 冷启动结论的独立交叉验证（server-mode）
server-mode 在同一台机器上独立复测并撤回了它最初的 22–112 s 数字：
- 独占 GPU、rows=128、无 prewarm：9012-token prompt wall **19.0 s / prefill 18.6 s（484 tok/s）**；
- 同 prompt 再请求（cached 8960 / new 52）：wall **0.65 s**，两次一致 ⇒ **cache-hit 路径本地不存在 22 s**；
- 之前的 112.8 s 是**两个进程各占 ~7 GB（合计 11.9/12 GB）时的 WDDM 换页**结果。
⇒ 与本文 §2 的结论一致：**"冷启动 23 s"不是磁盘冷读，是显存超订/并发占卡的换页**。

---

## 6. 给后续的边界与建议

1. **serve chunk 默认值与 prefix-cache 粒度是耦合的**：`reuse_pos = (boundary / chunk) * chunk`（`src/serve.cpp:298`）。chunk 128→512/1024 后，多轮对话每轮可能多 re-prefill 最多 511/1023 历史 token（1k tok/s 下 ~0.5/1 s）。**这需要 server-mode 在它那一侧用多轮 wall 复测后一起定默认值**；若多轮体验变差，推荐"新鲜全量 prefill 用大 chunk、cache 命中后的 tail 用小 chunk"（边界对齐由 cache 侧配合）。
2. **"按 n 懒捕获 prefill CUDA graph"路径在当前配置是关闭的**（`prefill_graph` 默认 false；`load_config_snapshot()` 在构造期就设置，`forward_impl` 里的 `static const bool pf_graph` 首次 forward 时读到 false）。如果 server-mode 观察到"同 n 首次 5.2 s vs 第二次 233 ms"，建议先用 `ENGINE_DBG_PFGRAPH=1` 确认是否真的在捕获；若是，应改成"只捕获 `n == prefill_chunk`、不捕获小 tail"。
3. **`derive_ctx_limit()` 仍用请求的 `prefill_rows` 预留激活**：用户若手动 `--prefill-rows 2048`，ctx 会先被按 2 GB 激活预算夹小（32768→8053）。更合理的做法是 ctx 限制按一个固定的参考 rows（如 512）推导，装载后再按实际余量放大 rows。本轮未改（避免动长上下文策略），已在 §5.2 的对护栏测试里覆盖其后果（400 而非 500）。
4. 若用户既想要"很长的文档"又想要"多轮前缀复用"，建议调 `--prefill-rows 512`（多轮）或 `--prefill-rows 1024`（长文档）显式切换；报告里的两个收益是同一根杠杆的两端。
