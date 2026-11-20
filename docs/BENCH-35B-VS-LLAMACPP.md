# 35B 同条件对比：自研引擎 serve vs llama.cpp b12128（2026-11-17）

用户真实两轮对话（同一会话、逐字）：

1. `你好`
2. `帮我写1000字小说`

采样参数（用户 llama.cpp 实际使用）：`temp 0.7 / top_p 0.8 / top_k 20 / min_p 0`。

> 结论先行：**我们的 decode 是 25.5 tok/s，llama.cpp 同请求 21.2 s 完成（含 MTP），我们 82.2 s
> （无 MTP，且 2048 token 上限截断）；差距来自三块：① serve 没接 MTP 投机解码（`--mtp`
> 在 serve 里甚至加载失败，见 §4）② 我们的 CPU 专家 GEMV 吞吐明显低于 llama.cpp（同为
> CPU 专家 offload，llama.cpp 只 6 层上 GPU 仍比我们 15 层快 ~2.3×）③ 我们本轮第 2 轮出现
> 字符级退化复读（待修 bug，见 §6），llama.cpp 同参数下输出了完整故事。**

## 1. 测试条件

| 项 | llama.cpp 侧（用户实测） | 我们（本次实测） |
|---|---|---|
| 程序 | llama-server b12128（MTP draft, n-max 2） | `engine serve`，2026-11-17 21:46 构建 |
| 模型 | Huihui-Qwen3.6-35B-A3B-…-Q4_K_M.gguf | 同一个文件（21.8 GB） |
| ctx / KV | 262144 / q8_0 | 见 §2（主对比用 4096；262144 也实测） |
| 采样 | temp 0.7 / top_p 0.8 / top_k 20 / min_p 0 | 相同 |
| 监控 | `/slots` 1s 采样 | `nvidia-smi` 1s 采样 + 引擎 `timings` |
| prompt tokens | 第 1 轮 375，第 2 轮 895（含 prefix cache 复用） | 第 1 轮 11，第 2 轮 41（模型 chat 模板；无 system prompt） |

⚠️ **上下文条件不同**：llama.cpp 第 1 轮 375 token 说明它带了约 360 token 的 system prompt/历史，
我们按同样的用户消息渲染只有 11 token；两边的"同两轮对话"文本相同，但**上下文长度差一个量级**，
这会影响模型行为（我们更易在短上下文下退化复读）。本表对比的是"同模型、同用户输入、同采样参数"。

## 2. 主对比：ctx 4096 / KV q8_0 / MTP off（干净 CPU，两次重复）

启动账（引擎日志）：`ctx=4096, kv_type=q8_0, kv/token=10880 B, kv_layers=10/40, kv_total=42 MiB`，
fixed 2526 MiB，**gpu_moe_layers=15/40**（其余 25 层 CPU 专家），load 59.6–77.2 s，
post-load free 1710 MiB；`prefix-cache on, prefill chunk 128`。

| 轮次 | wall (s) | prompt tok | prefill ms | decode ms | gen tok | 首 token ms | decode tok/s | finish |
|---|---|---|---|---|---|---|---|---|
| 1 你好（run A） | 7.01 | 11 | 565 | 6434 | 166 | 577 | 25.8 | stop |
| 2 写小说（run A） | 82.58 | 42 | 1851 | 80712 | 2048 | 922 | 25.37 | length（撞 2048 cap） |
| 1 你好（run B） | 7.87 | 11 | 631 | 7224 | 185 | 644 | 25.61 | stop |
| 2 写小说（run B） | 82.23 | 41 | 1771 | 80448 | 2048 | 886 | 25.46 | length（撞 2048 cap） |

GPU：avg util **42.4%**，峰值 100%，显存峰值 11.5 GiB，功率峰值 88 W。

### 与 llama.cpp 并列

| 轮次 | llama.cpp wall / prompt / GPU 峰值 | 我们 wall / prompt / prefill / decode / tok/s | 差异 |
|---|---|---|---|
| 1「你好」 | 2.1 s / 375 tok / 41% | 7.9 s / 11 tok / 631 ms / 7.22 s / 25.6 tok/s | 慢 3.8×；但首 token 我们 644 ms；其 2.1 s 含 375 tok prefill |
| 2「写1000字小说」 | 21.2 s / 895 tok（含 prefix cache）/ 41% | 82.2 s / 41 tok / 1771 ms / 80.4 s / 25.5 tok/s / gen 2048（截断） | 慢 3.9×；其输出完整《月光邮局》约 1000 字，我们撞 cap 且出现复读 |

**差在哪（按贡献排序）**：
1. **MTP 投机解码缺失**：llama.cpp 有 `draft-mtp n-max 2`（历史实测 +29.5%）；`engine serve` 的
   生成循环**只调用 `forward()`，从未调用 `runner::spec_generate()`**（后者只在
   `tests/test_gen.cpp` / `tools/mtp_run.cpp` 用）⇒ 这一块我们直接少了 ~30%。
2. **CPU 专家 GEMV 吞吐差距（最大头）**：同是 CPU offload，llama.cpp 只 6/40 层上 GPU，
   却 ~21 s 完成（按 ~1000 token 估算 ≈ 50–60 tok/s）；我们 15/40 层上 GPU 只有 25.5 tok/s。
   说明我们的 CPU 专家 matvec 比 llama.cpp 的量化 kernel 慢 ~2–4×（我们 9 线程 int8；
   llama.cpp 用 AVX2/AVX512 量化点积 + 多线程）。这是 decode 的主要瓶颈，不是 GPU 层数。
3. **上下文/模板差异**：其 prompt 375/895 vs 我们 11/41，模型行为不同（见 §1 ⚠️）。
4. **质量 bug（待修）**：见 §6，撞 cap 前已退化为字符级复读，进一步拉长 wall。

## 3. 附录：ctx 262144 / KV q8_0（**受 CPU 争用污染，不可用于性能结论**）

该组测量时 `prio-quiet` 的 9–12 个 HIGHEST 线程与我们的 9 个同优先级线程抢 12 个逻辑核，
11-token 上下文 decode 只有 4.4 tok/s（不可能的正常值），**整组作废**，仅保留"能不能装下"的信息：

| 项 | 实测 |
|---|---|
| 启动 | **成功**：ctx 262144，q8_0 `kv/token=10880 B`，`kv_total=2720 MiB`，fixed 5204 MiB，**gpu_moe_layers=9/40**，load 56.8 s，post-load free 2125 MiB |
| 轮 1（污染） | wall 53.6 s，prompt 11，prefill 1956 ms，decode 51.5 s，gen 228，4.43 tok/s |
| 轮 2（污染） | wall 800.3 s（撞 4096 cap），prompt 41，prefill 4468 ms，decode 795.8 s，gen 4096，5.15 tok/s |

**显存账（q8_0）**：fixed 1693 MiB（权重非专家部分） + KV 2720 MiB + act(rows128) 146 MiB
+ workspace 768 MiB + margin 1200 MiB ≈ 6.5 GiB，剩余 ~4.7 GiB 只能放 **9 层**专家（ctx 4096 时是 15 层）。
⇒ **256K ctx 我们装得下，但只剩 9 层 GPU 专家，decode 会显著变慢；12GB 卡上建议聊天用 ctx 4096。**

## 4. MTP 对比（实测：serve `--mtp` **加载失败**）

命令：`engine serve … --ctx 4096 --kv-type q8_0 --mtp --mtp-k 2`，日志：

```
[placement] gpu_moe_layers=15/40 ...
[mtp] MTP experts: CPU (need 498 MiB, free 548 MiB, margin 256 MiB)
[serve] ERROR: model load failed: Buffer::upload exceeds allocation
```

即 **serve 打开 `--mtp` 时 MTP 专家走 CPU offload 分支会触发加载 bug**（Buffer 大小与实际
upload 不一致），服务起不来；因此本轮无法给出 MTP 的 serve 端速度对比。
叠加代码事实（serve 不调用 `spec_generate`），**当前 `engine serve --mtp` 既没有投机解码收益，
还存在加载 bug** —— MTP 的 +29.5% 要先在 serve 侧接线（用 `spec_generate` 替换 decode 循环的
greedy 路径）才能兑现；本项标为待修。

## 5. 输出质量

- 轮 1「你好」：`你好！很高兴认识你。有什么我可以帮你的吗？`（18 汉字，正常，无复读）
- 轮 2「写1000字小说」：**质量不达标**。两次运行分别：
  - run A：先写了 1701 汉字的《时间缝补者》（开头通顺），随后退化成整段句子复读
    `老城的秘密，是老人。` ×28、`老人的秘密，是老人。` ×13，直到 2048 cap；
  - run B：故事开头正常（《雨夜的最后一通电话》），随后退化成**字符级复读**
    `—你——你—…—134—134—…—22—22—` 直到 cap（同句重复统计=1，因为已不是完整句子）。
  - 完整文本：`build/compare/our-turn2-35b_4k_q8_mtpoff.txt`（run B）、
    `build/compare/our-turn2-35b_4k_q8_mtpoff_clean.txt`；JSON：
    `build/logs/bench_35b_4k_q8_mtpoff*.json`。
- 对照：llama.cpp 同参数输出了完整《月光邮局》约 1000 汉字。

## 6. 本轮实测暴露、需要修的问题（本轮只测不改）

1. **repeat guard 漏检变体/字符级复读**：现有实现按"连续 3 句完全相同（≥8 字符）"判断；
   run A 的整句复读带轻微变体、run B 是字符片段复读，都没触发 ⇒ 需要基于
   n-gram/字符重复率的兜底（建议：最近窗口内重复 n-gram 占比超阈值即停）。
2. **think-budget 强制闭合后的重答质量差**：强制 `</think>` 后模型常在"重答"里继续规划/退化；
   本轮第 2 轮即如此。需要换策略（如：预算到点直接停 + 以"关闭思考"的干净 prompt 重跑一次，
   而不是同一状态续写）。
3. **serve `--mtp` 加载失败**：MTP 专家 CPU offload 分支 `Buffer::upload exceeds allocation`。
4. **serve 未接 MTP 投机解码**（`spec_generate` 未在 serve 调用）——修完 3 后才能发挥。

## 7. 复现

```bash
# 干净 GPU、无其它服务
bash tools/bench_35b_engine.sh 35b_4k_q8_mtpoff_clean 0 512 4096 q8_0 2048
bash tools/bench_35b_engine.sh 35b_4k_q8_mtpon_clean 1 512 4096 q8_0 2048   # 当前会加载失败（§4）

# 262144 对照（先确认无 CPU 争用，否则 decode 会掉到 4-5 tok/s）
bash tools/bench_35b_engine.sh 35b_262k_q8_mtpoff 0 512 262144 q8_0 4096
```

产物：

| 文件 | 内容 |
|---|---|
| `build/logs/bench_35b_4k_q8_mtpoff_clean.json` | 主对比 run A（每轮全部指标 + 输出文本） |
| `build/logs/bench_35b_4k_q8_mtpoff_clean.gpu.log` | 1s 采样的 GPU util/显存/功率 |
| `build/logs/bench_35b_262k_q8_mtpoff.json` | 262144 附录（污染，见 §3） |
| `build/compare/our-turn2*.txt` | 第 2 轮完整输出文本 |
