# MTP 投机解码的数值一致性（为什么 --mtp 不是逐位无损）

日期：2026-11-21 · 状态：**限制已确认并写入 WARN；真正的修复 = P0-A 多行量化 GEMV**

---

## 0. TL;DR

- MTP 投机解码的**接受/回滚逻辑是正确的**：emit 的每个 token 恒等于 target 模型在
  *batch-verify 前向*里的 arg-max，被拒 draft 的 conv/rec 状态回滚后状态与"已接受前缀"一致。
  Probe 三流对拍（`tools/mtp_numprobe.cpp`）证明：**spec 流 == "batch 数值下的 greedy" 流**
  （逐 token 相同），而 **≠ plain（n=1 decode 数值）greedy 流**，分叉点 D 与实测一致。
- 分叉的根因是**数值路径不同**，不是状态腐坏：同一个 target 模型、同一个前缀状态，
  "verify 批前向"和"单 token decode 前向"走的是不同 kernel / 不同精度路径（下表），
  hidden 有 ~1e-3 级差异；长序列里遇到 top1-top2 差距小于该差异的位置，arg-max 就会翻。
- 实测（默认配置，Qwen3.6-35B-A3B，prompt `The capital of France is`，greedy）：
  **plain 300 token vs `--mtp --mtp-k 2` 在前 147 个 token 完全一致，第 148 个分叉**
  （D=147，0-based；plain=211 vs spec=223），与 server-mode 的独立观测一致。
- 因此：**默认保持 `--mtp` 关闭**（本来就是），并在 `--mtp` 装载成功时打印明确 WARN
  （`src/mtp.cpp` load_mtp）；需要逐位一致的用户请勿使用 `--mtp`。
- 真正的修复是 P0-A（2–4 行量化 GEMV）：同一 kernel、同一累加顺序 ⇒ batch 与 decode
  **逐位一致**，同时把 verify 的权重流量摊薄 ⇒ MTP 才有可能从净亏转为净赚。见 §5。

---

## 1. 两条数值路径的差异表（qwen35 hybrid，n = verify 批大小 = k+1）

| 算子 | 单 token decode（plain greedy） | (k+1) 行 verify（spec） | 数值关系 |
|---|---|---|---|
| 权重 matmul | **dp4a GEMV**（int8 激活量化 + int8 dp4a） | **fp16 权重 cuBLAS tensor-core GEMM**（`matmul()` 的 `n_tokens>1` 分支，f32 累加） | 不同精度/不同累加顺序，hidden rel ~1e-3 |
| lm_head | 量化 GEMV + device top-k kernel | **量化 head 的多行 GEMM**（`runner.cpp` `verify_multi_head_` 分支）+ host arg-max | 不同 reduce 顺序 |
| attention | `attn_tiled`（n=1，split-K） | `attn_tiled`/`attn_tiled_pre`（n>1，K/V 复用布局不同） | softmax 累加顺序不同 |
| SSM（gated delta net） | fused：`gd_ssm_prep` + `gd_delta_rule_128f`（T=1） | spec_snapshot 模式：conv/delta 逐 token 分离内核（`gd_conv1d_silu`+`gd_conv_state`+`gd_delta_rule_128`+`gd_norm`） | 同一递推、不同内核，2e-7 级 |
| MoE（35B） | n=1：fused GPU expert kernel / CPU 逐 token | n>1：gather/scatter + 多行 expert 路径 / CPU 批量 | 累加顺序不同 |

> 结论：verify 的 logits 与 decode 的 logits 不是同一函数的同一实现，差 1e-3 量级；
> 而 greedy 的 argmax 决策只要求 top1-top2 差 < 该误差就可能翻转。这类翻转是
> **随机稀疏**的（35B 实测 ~1/150 token），所以短测试（12 token golden、58 token A/B）
> 通不过也发现不了。

---

## 2. 证据 A：probe 三流对拍（把"数值"与"回滚"分开）

`tools/mtp_numprobe.cpp` 在一个进程里跑三条 greedy 流：

- `plain`：n=1 decode 逐步（参照）；
- `batchgreedy`：每个 token 用 **verify 形状的 3 行批**前向，再 `ssm_restore(0)` 回滚到第 0 行
  （= "batch 数值下的 greedy"；状态推进方式与 spec 完全相同）；
- `spec`：引擎真实 `spec_generate`。

判据：
- `spec == batchgreedy` ⇒ 接受/回滚逻辑 faithful；
- `batchgreedy ≠ plain` ⇒ 分叉来自数值路径。

**实测结果（2026-11-21，35B，180 token）**：待此轮 GPU 窗口实测填入（见 §6 运行记录）。

---

## 3. 证据 B：分叉点对数值开关的敏感性

`tools/mtp_divergence.sh` 对 plain-vs-spec 的分叉点做扰动扫描。若分叉点随数值路径开关
（`ENGINE_PREFILL_F16=0`、`ENGINE_NO_GEMV=1`、`ENGINE_NO_DP4A_GEMV=1`、
`ENGINE_GPU_EXPERTS=0`）移动 ⇒ 数值原因；若完全不动 ⇒ 状态/回滚 bug。

**实测结果**：待填。

---

## 4. 证据 C：serve prefix-cache 的 cold/warm 交叉一致性

`tools/cache_cold_warm_check.py`：同一 prompt 连发 3 次（greedy），对比
cache miss（chunk+tail 两次 batch prefill）/ cache hit（restore 快照 + 同一 tail）
/ `--prefix-cache=0`（全 cold）的 generated_token_ids。

**实测结果**：待填。
（代码审查预期一致：`src/serve.cpp:350-380` 里 cold-with-cache 与 warm 进入 tail 的
状态和 tail 宽度相同；真正有风险的只有 restore 不保真或 tail 宽度不同的跨轮次场景。）

---

## 5. P0-A 为什么是正解（价值论证）

**P0-A = "2–4 行量化 GEMV"**：decode 用的 GEMV kernel 改为一次读权重块、并行算 2–4 行激活
（读一遍权重算多行）。

1. **逐位一致**：同一 kernel、同样的点积顺序、同样的 int8 激活量化规则（每行独立量化）⇒
   verify 批的第 r 行结果与"单独算第 r 行"逐位相同 ⇒ batch-verify 数值 == 单 token decode 数值。
   MTP 的分叉从根上消失，同时**修掉所有"批 vs 单"数值分歧**（prefix cache cold/warm、
   任何 prefill+decode 混合路径的隐性不一致）。
2. **成本摊薄**：decode 的 GEMV 是**权重带宽受限**（35B 非专家权重 + CPU 专家读取）；
   读一遍权重算 2–4 行，verify 的 per-token 成本降到接近 1× decode。
   这正是 MTP-IMPL 报告里"MTP 在 35B 净亏"的根因：`n>1` 验证成本 c≈2.0–2.6× 单步，
   而收益 `E≤1+α`；有了 P0-A，c 有望压到 ~1.1–1.2×，再配合 α≈0.8 就有 (1+α+α²)/(c+2d) > 1。
3. **顺带收益**：prefill（长文档）也读一遍权重算多行是已有方向；P0-A 的 kernel 可以同时
   服务 prefill 的小 batch 尾部与 MTP verify。
4. **代价/风险**：需要为 dp4a/x4 GEMV 的每种量化（Q4_K/Q5_K/Q6_K/Q8_0）实现多行变体，
   并保证"逐位同序"（乘加顺序不能变，激活 int8 量化必须逐行独立）；CPU 专家路径已经是
   逐行独立（`cpu::moe_forward_mt` 同函数同序）⇒ 35B 的 MoE 部分天然满足，工作量集中在
   GPU 侧 matmul/head。

⇒ **建议排进下一轮优先级：P0-A 同时是 MTP 可用性的前置条件与一类正确性隐患的根治。**

---

## 6. 复现命令

```bash
cd engine
# 三流对拍（35B，180 token；ENGINE_MTP=1 打开 spec 流）
ENGINE_MTP=1 ENGINE_MTP_TRACE=1 ./build/mtp_numprobe.exe \
  "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf" \
  --ids 760,6511,314,9338,369 --n-gen 180
# 分叉点 + 数值开关扫描（300 token）
bash tools/mtp_divergence.sh "${MODEL_DIR}/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf" 300
# serve prefix-cache cold/warm
python tools/cache_cold_warm_check.py --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf"
```

运行记录见 §7。

## 7. 运行记录（2026-11-21 实测）
