# 35B 每 token 速度差距：根因分析（MOE-GAP，进行中）

**日期**：2026-11-17 · **触发**：用户觉得「llama.cpp 只放 6 层 GPU 却比我们 15 层还快 ~1.5×」，
怀疑 CPU MoE 慢 2–4×。
**状态**：前提已被数据推翻（见 §1）；真正杠杆排序见 §3；`tools/moe_gap_probe.sh` 待 GPU 窗口
跑同条件 `llama-bench -ngl {0,6,15}` vs 我们 `test_gen --gpu-layers {6,15}` 做最终钉死（§4）。

---

## 1. 数据还原：墙钟差主要是"生成了 2.4–2.9× 的 token"，不是每 token 慢

来自我们对比跑的 serve JSON（`build/logs/bench_35b_4k_q8_mtpoff_clean.json`，35B Q4_K_M、ctx 4096、
KV q8_0、MTP off、`gpu_moe_layers=15/40`）：

| 轮次 | wall | prompt | prefill | gen tokens | decode | decode tok/s | 备注 |
|---|---|---|---|---|---|---|---|
| turn1「你好」 | 7.87 s | 11 | 630.8 ms | **185** | 7223.7 ms | **25.61** | finish=stop |
| turn2「写1000字小说」 | 82.23 s | 41 | 1770.8 ms | **2048** | 80447.8 ms | **25.46** | **finish=length（撞 2048 cap）、think_budget_hit=true** |

llama.cpp 侧（用户表）：turn2 wall 21.2 s、decode ≈33–40 tok/s 记为"含 MTP +35%"。
⇒ 反推它的生成量 = 21.2 s × (33…40 tok/s) ≈ **700–850 tokens**（约 2.4–2.9× 少于我们的 2048）。

**分解 82.2 / 21.2 = 3.88×**：

| 因子 | 倍数 | 证据 |
|---|---|---|
| token 数（2048 vs ~700–850） | **2.4–2.9×** | 我们 finish_reason=length + think_budget_hit |
| 每 token 速度（含 MTP 差异） | **1.3–1.5×** | 25.5 tok/s vs 33–40（含 MTP）；**去掉 MTP 后 llama 基础 ≈24–30 tok/s ≈ 我们 25.5** |

⇒ **"llama.cpp CPU MoE 快 2–4×"不成立**：拿掉 MTP 后每 token 速度基本持平。
（turn1 的 7.87 s vs 2.1 s 同理：我们生成 185 token、它大概率只生成 ~60–80 token。）

## 2. 我们的 39.3 ms/token 花在哪（账本，待 `moe_gap_probe` 用 `[layerms]` 精确确认）

35B serve 配置（`--gpu-layers 15` → 15 GPU 专家层 / **25 CPU 专家层**），decode 39.28 ms/token：

| 项 | ms/token | 说明 |
|---|---|---|
| 宿主 GEMV（CPU 专家 FFN） | **18.9** | 25 层 × 0.755 ms（`ENGINE_DBG_CPUEXEC` 实测 last-25 窗口） |
| CPU 层 launch+sync | ~7.2 | 25 × ~0.29 ms（历史分项：launch 0.118 + sync 0.17） |
| 15 个 GPU 层 | ~3.0 | 15 × 0.2 ms |
| head（lm_head Q8_0 540 MB ≥1.6）+ 采样/top-k | ~2.5 | 硬底 |
| 其它固定/未归因 | **~7.7** | 待 `[layerms]` + `ENGINE_PIPE_PROF` 在窗口内拆 |

llama.cpp（34 CPU 层）的推算：若其 ~28 ms/token（≈35 tok/s 基础），则每 CPU 层 ≈
(28 − 6×0.2 − ~2) / 34 ≈ **0.7 ms/层**（含 CPU 侧 attention，短上下文可忽略）。我们的
**CPU 层总成本 1.35–1.55 ms/层**（0.755 GEMV + ~0.6 往返/同步）——即差距在**每层 GPU 往返开销**，
不在 CPU GEMV 内核本身；而且我们只跑 25 个 CPU 层（它 34 个），CPU FFN 总时间我们并不吃亏
（18.9 vs ~22–24 ms）。

## 3. 真正杠杆（按预期收益排序）

1. **生成长度/复读**：已修（repeat guard v2 + think-budget 干净重答）——墙钟 2.5× 的主因。
2. **serve 接 MTP**（+35%）：正在做；接上后 25.5 → ~34 tok/s 级别。
3. **每层 GPU 往返开销 ~7 ms/token**：CPU 层的 attention 在 GPU 上、每层一次 stream_sync
   （~0.17 ms × 25）+ H2D/launch。方向：软件流水（CPU 层 i 的 GEMV 与 GPU 上第 i+1 层的
   attention 重叠，或每 2–4 个 CPU 层合并一次 sync），属结构性改动、需单独评估。
4. **CPU GEMV 内核（整数点积）**：**优先级最低**。现有证据：Q6_K 的 int8 内核已带来 −10%
   （0.811→0.734 ms/层），Q4_K 的 int8（两级量化）实测反而 +4.6%；即使把 Q4_K 优化到与 fp32
   打平，也只影响 GEMV 的 ~45%×token ≈ 4–10 ms 中的一部分，端到端 <5%。**若 `moe_gap_probe`
   证明每 token 持平，这条线直接放弃。**

## 4. 待窗口执行的钉死实验（`tools/moe_gap_probe.sh`）

```bash
# A) llama.cpp：decode-only（-p 0 -n 32），-ngl 0（纯 CPU）/ 6 / 15
${LLAMA_BENCH} -m <35B> -ngl 6  -t 12 -p 0 -n 32 -r 2 -o md
# B) 我们：同模型同放置，decode-only n=32；同时抓 [layerms]（sync/launch/host 分项）
ENGINE_DBG_CPUEXEC=1 ENGINE_DBG_LAYERMS=1 build/test_gen.exe <35B> 760 32 --gpu-layers 6
```
判据：
- llama `-ngl 0` 的 tg t/s ⇒ llama 每 CPU 层成本（40 层全 CPU，短上下文 attention 可忽略）。
- `-ngl 6` vs 我们 `--gpu-layers 6`：**同放置**的 decode tok/s 直接可比；若差 ≤20%，
  则"CPU MoE 慢 2–4×"彻底排除，整数点积内核放弃。
- 我们 `--gpu-layers 15` 的 `[layerms]` 会给出每层总 ms、sync/launch 次数与 host GEMV ms，
  据此把 §2 的 "~0.6 ms/层往返" 钉死，作为"合成/流水"改动的输入。

## 5. 证据文件

- `build/logs/bench_35b_4k_q8_mtpoff_clean.json` / `.serve.log`（我们的 turn1/turn2 完整数字）
- `build/logs/bench_35b_4k_q8_mtpon_clean.serve.log`（MTP on 仅启动，未产生请求）
- `tools/moe_gap_probe.sh`（待跑）
