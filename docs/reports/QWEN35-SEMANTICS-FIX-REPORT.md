# Qwen3.5-9B / Qwen3.6-35B-A3B 真实文本生成修复报告（GGUF 张量语义）

日期：2026-11-15
作者：qwen35-prefill-bug（agent）
状态：**已修复并在本机 9B / 35B 上与 llama.cpp 逐 token 一致**；新增生成回归测试；
`verify.sh` 唯一失败项（`test_gguf`）经隔离确认与本修复无关（见 §9）。

---

## 0. TL;DR

"qwen35 系列在真实文本 prompt 上生成乱码 / 复读"的根因**不是**多 token prefill 的批处理
kernel，而是**两处 GGUF 张量语义被按 HF（transformers）约定错误解释**：

| # | 位置 | 错误逻辑（修复前） | 正确逻辑（GGUF = llama.cpp 约定） |
|---|---|---|---|
| **A** | `qwen35.ssm_a`，消费点 `src/qwen35.cpp` 的 `run_q35_linear_attn()`（修复前 `:352`、`:423`，现在集中在 `:333-353`） | `g = -exp(ssm_a) * softplus(a + dt_bias)`（把 `ssm_a` 当 `log A`） | `g = ssm_a * softplus(a + dt_bias)`（GGUF 存的是 `-A`） |
| **B** | 5 个 gated-delta 内核的 q/k→v head 广播，`include/engine/kernels.h`（修复前 `:1205/:1264/:1390/:3699/:3773`，修复后 `:1211/:1275/:1406/:3721/:3801`） | `kh = h / rep`（HF `repeat_interleave`：v head h 读 key head h//2） | `kh = h % n_kh`（llama.cpp **块重复**：v head h 读 key head h % 16） |

两者叠加的后果：每个 SSM 层每步的衰减被放大成 `exp(-softplus)≈0.1`（应≈0.9），且 24/30 个
SSM 层里 32 个 value head 有 3/4 配错了 key head ⇒ 模型"记忆"只剩 1–2 个 token，于是
**复读输入 / 数字符号汤**；而 **position 0 几乎退化**（attention 只有 1 个 key、SSM state
从 0 开始、衰减不起作用），所以"单 token 冒烟 test_gen"一直是绿的。

修复后本机实测（greedy，temp=0，与 `U:\llama\llama-server` 同 GGUF 逐 id 对比）：

| prompt | llama.cpp 生成 ids | 修复后引擎 ids | 结果 |
|---|---|---|---|
| 9B `The capital of France is` | `11751 13 198 760 6511 314 9338 369 11751 13 198 760` | 同 | ✓ 12/12 = `" Paris.\nThe capital of France is Paris.\nThe"` |
| 9B `The` | `2614 513 5081 5576 4602 318 4058 10926 8 883` | 同 | ✓ 10/10 = `" following are multiple choice questions (with answers) about"` |
| 9B `France is` | `264 3046 303 10550 4357 13 1049 369` | 同 | ✓ 8/8 = `" a country in Western Europe. It is"` |
| 9B `你好` | `3709 95815 111354 103801 3709 96846 96442 97235` | 同 | ✓ 8/8 |
| 9B chat22（system+user 你好，thinking） | `90700 8340 25 271 16 13 220 2972` | 同 | ✓ 8/8 = `"Thinking Process:\n\n1.  **"` |
| 35B `The capital of France is` | `11751 11 264 3177 34756 364 1141 25438 57902 1680 430 279` | 同 | ✓ 12/12 |
| 35B chat22（同上） | `40 2688 28378 310 279 1156 579 40719` | 同 | ✓ 8/8 = `"I'm responding to the user's greeting"`（不再是符号汤） |
| 35B `你好` | `3709 95815 111354 98292 97432 3709 99443 97706` | 同 | ✓ 8/8 |

---

## 1. 为什么"和 HF 参考逐层 1e-6 一致"反而掩盖了这个 bug

1. 这台机器上的 qwen35 权重是 **GGUF**，而 GGUF 转换器 / 加载器是 **llama.cpp** 生态：
   文件里 `qwen35.ssm_a` 存 `-A`、q/k→v 用块重复广播（llama.cpp
   `src/models/qwen35.cpp` 里 `gate = alpha_softplus * ssm_a`，注释 `// -A_log.exp() * softplus`；
   `ggml/src/ggml-cuda/gated_delta_net.cu` 里 `iq1 = fastmodulo(h_idx, neqk1)`，
   `ggml.h` 里还有 `[TAG_GGML_GDN_BCAST]` 的 "tiled vs interleaved" TODO）。
2. 官方 HF `qwen3_5` 模块实现的是 `A_log` + `repeat_interleave`（`transformers/models/qwen3_5/
   modeling_qwen3_5.py`：`g = -A_log.exp()*softplus(...)`、`query.repeat_interleave(...)`）。
3. 旧引擎按 HF 公式实现（缺省 `g = -exp(ssm_a)*softplus`、`kh = h/rep`），而
   `tools/q35_torch_chain.py` 也是把 GGUF 张量直接塞进 HF 模块 —— **两边用同一套错误语义，
   所以对拍到 1e-6 只证明了"实现与参考一致"，没有证明"语义与文件一致"**。
   旧 README 甚至把这写成"已知差异：llama.cpp 与官方 HF 参考不同（我方与 HF 一致）"，
   方向判断反了：**GGUF 文件应该按 llama.cpp 语义读**。

---

## 2. 决定性证据：可切换语义的 numpy 裁判 + llama.cpp 对照

新增 `tools/q35_variants.py`（约 300 行，numpy 重写 dense qwen35 栈，可独立切换两处语义）。
它先自校验：**在"旧引擎语义"下逐层复现旧引擎 `ENGINE_DUMP_EVERY` 输出到 ≤7.3e-6**
（layer27–31 最大 7.25e-6，首层 4.2e-7），因此它不是在拿引擎证明自己。

同一 5-token prompt（`760,6511,314,9338,369`，llama.cpp 首 token = **11751 " Paris"**）：

| 变体 | ssm_a | q/k→v 广播 | last-position top-5 | 结论 |
|---|---|---|---|---|
| V0 旧引擎 | log(A) | h//rep（HF） | `279(-1.518) 264(-1.801) 524(-2.929)`，与旧引擎逐位相同 | 错（复读 " the"） |
| V1 只修 A | **-A** | h//rep | `279(-2.006) 264(-2.391) …` | 仍错 |
| V2 A+B 都修 | **-A** | **h%16** | **`11751(-0.509)`** 264(-3.066) 3750(-3.554) | **= llama.cpp " Paris"** |
| V3 只修 B | log(A) | h%16 | `11751(-0.649)` 264(-3.011) 7172(-3.336) | 首 token 对，但衰减错 |

结论：**B 是"首 token 就对"的决定性修复；A 必须一起修**（否则长上下文记忆衰减错约 10×，
短 prompt 首 token 可能看不出来）。两者都已在引擎中落地。

---

### 2.1 35B ground truth 的来源复核（防止"拿错模型自证"）

35B 的 golden 与 9B 在 `The` 上**前 8 个 token 完全相同**（`2614 513 5081 5576 4602 318 4058 10926`），
初看可疑，已用三条证据确认**确实是 35B 模型**、不是端口串到 9B：
1. llama-server 启动日志：`srv load_model: loading model '<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf'`；
2. `/props`：`model_path = ...Huihui-Qwen3.6-35B...gguf`、`model_ftype = Q4_K - Medium`；
3. **同一 35B server 的其它 prompt 与 9B 完全不同**：`The capital of France is` → 35B golden
   `11751 11 264 3177 34756 364 1141 25438 57902 1680 430 279`，而 9B 是
   `11751 13 198 760 6511 314 9338 369 …`；`你好` → 35B `…111354 98292 97432 3709 99443 97706`，
   9B `…111354 103801 3709 96846 96442 97235`。修复后引擎在 12/8 token 上逐 id 对上 35B golden。
   结论：`The` 的前缀相同是两个模型共享同一"MMLU 式"强先验的合理巧合，不是取错模型。



## 3. 修复实现

### 3.1 内核：`include/engine/kernels.h`

5 个 gated-delta 内核都加了一个 `int kh_mode` 尾参（0 = HF `h/rep`，1 = GGUF `h % n_kh`）：
`gd_delta_rule`(`:1199`)、`gd_delta_rule_128`(`:1258`)、`gd_delta_rule_128f`(`:1388`)、
`gd_delta_rule_128_smem`(`:3702`)、`gd_delta_rule_128_reg`(`:3784`)；映射统一改为
`const int kh = kh_mode ? (h % n_kh) : (h / rep);`（带注释说明两种约定与 llama.cpp 出处）。

### 3.2 层实现：`src/qwen35.cpp`

`run_q35_linear_attn()` 顶部新增语义选择块（`:333-353`）：

```cpp
const bool gguf_sem = container_ == Container::Gguf;
// env 回退：ENGINE_Q35_RAW_A / ENGINE_Q35_LOG_A / ENGINE_Q35_HF_BCAST
const bool raw_a  = force_raw_a || (gguf_sem && !force_log_a);
const int  kh_mode = (!force_hf_bcast && gguf_sem) ? 1 : 0;
```

- Fused decode 路径的 `gd_ssm_prep`（`:374`）与 batch 路径的 `k_q35beta_raw_`（`:444`）
  统一用 `raw_a`；
- delta rule 的 args/fargs 增加 `&khm`（`:463` 起，覆盖 fused / smem / reg / 128 / generic 5 条分支）。

**设计取舍**（按要求）：GGUF 走 llama.cpp 语义，Safetensors 走 HF 语义，由 `container_`
决定，而不是全局翻转。当前 qwen35 的 safetensors 加载器尚未实现（IR 只支持 GGUF 的
`ssm.*` 元数据），但分支已就位，未来接 safetensors 时自动是对的。
一键回退：`ENGINE_Q35_LOG_A=1`（HF A 约定）、`ENGINE_Q35_HF_BCAST=1`（HF 广播）、
`ENGINE_Q35_RAW_A=1`（强制 -A，兼容旧开关）。

### 3.3 未触碰

- `DenseRunner::matmul()`（prefill-fast 正在改）、`build.sh` 编译选项、所有量化/dp4a 数学。
- 备份（仓库无 git）：`src/qwen35.cpp.pre-q35sem`、`include/engine/kernels.h.pre-q35sem`。
- 注意：这两个文件在我动手前（10:24/10:32）已被其他 agent 改过（新增
  `gd_delta_rule_128_smem/_reg` prefill 内核）；我把它们一起纳入 `kh_mode` 修改，
  prefill-fast 已用 `tools/probe_gd_parity.cpp` 对拍确认两 kh_mode 下 smem/reg 与参考内核
  BIT-EXACT。

---

## 4. 新增/更新的防线

### 4.1 `tools/gen_regression.sh`（新增，核心防线）

多 token prefill + greedy 生成 8–12 token，**逐 id** 与 llama.cpp ground truth 对比；golden
全部来自 `llama-server /completion`（`temperature=0`、`cache_prompt=false`，id 取自
`completion_probabilities[].id`）。共 8 组：

```
9b_capital_of_france  12 tokens   PASS
9b_the                10 tokens   PASS
9b_france_is           8 tokens   PASS
9b_zh_nihao            8 tokens   PASS
9b_chat22              8 tokens   PASS
35b_capital_of_france 12 tokens   PASS
35b_chat22             8 tokens   PASS
35b_zh_nihao           8 tokens   PASS
coder/qwen2 守卫      (185 / 40716) PASS
```

### 4.2 `tools/smoke_models.sh`（更新期望值）

**35B 的期望值 `69014 → 2614`**（`69014` 是旧 HF 语义下的错值）。证据：llama-server 加载
`Huihui-Qwen3.6-35B...Q4_K_M.gguf`（`/props.model_path` 与启动日志均确认，`model_ftype=Q4_K`）
对 prompt id `760` 的 greedy 首 token = `2614`，且其 4-token 输出 `[2614,513,5081,5576]`
与 9B 相同——这是合理的（两个模型共享同一 MMLU 式先验），用 `The capital of France is`
即可区分：35B 给 `11751 11 264 3177…`，9B 给 `11751 13 198 760…`。9B 的 `2614` 不变。
脚本末尾新增一行提醒：**单 token smoke 不足以发现 qwen35 语义错误**。

### 4.3 `tools/verify_reference.sh`（判据修正）

qwen35 的逐层对拍从 `q35_torch_chain.py`（HF 语义，判据本身错）改为
`tools/q35_variants.py --a-fix 1 --bcast tile --tolerance 2e-4`（GGUF/llama.cpp 语义，
per-layer rel_l2 超限即 fail）。`q35_torch_chain.py` 保留给 safetensors 场景（文件头已注明）。

### 4.4 `tools/q35_variants.py`（新增，裁判工具）

- 默认运行 4 变体对照表（根因证据，一次跑完，dequant 只做一遍）。
- `--a-fix 1 --bcast tile --dump <prefix> --layer N --tolerance T`：作为 GGUF 语义的
  per-layer 参考（超限返回非 0）。
- 文件头写明了 GGUF vs HF 两处语义差异、llama.cpp 源码出处、以及"不要拿 HF 参考判 GGUF"。

---

## 5. 验证状态（最终验收，2026-11-15 11:2x）

| 套件 | 状态 | 日志 |
|---|---|---|
| `tools/verify.sh` | **ALL GREEN，exit 0**（51 项；M2 plan parity 8/8 模型 PASS，M3 CUDA/gemm/layer/quant/moe/gemv/kv/model/mla/decode 全 PASS） | `build/verify-final-q35.log` |
| `tools/gen_regression.sh` | **10/10 PASS，exit 0**（9B 5 + 35B 3 + Coder/0.5B 守卫；共 74 个 greedy token 逐 id 与 llama.cpp 一致） | `build/genreg-final-q35.log` |
| `tools/smoke_models.sh` | **4/4 正确，exit 0**：40716 / 185 / 2614 / 2614（35B 期望已按 llama.cpp 更新） | `build/smoke-final-q35.log` |
| `tools/verify_generality.sh` | **26 passed / 0 failed，GENERALITY GREEN，exit 0** | `build/generality-final-q35c.log` |
| `tools/verify_reference.sh`（q35 部分） | **逐层 2.3e-7 ~ 4.5e-7 PASS**（GGUF/llama.cpp 语义参考，`--a-fix 1 --bcast tile --tolerance 2e-4`，dump 用 `ENGINE_PREFILL_F16=0` 走精确 f32 路径） | `build/ref9b-check.log` |

> 验收期间的两次环境噪声（均与修复无关）：① `verify_generality.sh` 在 11:14 被并发编辑
> 保存成 **CRLF** 行尾导致 `for ... do\r` 语法错（已修复为 LF，属机械修复）；② 全绿前有
> 一次 `post-load margin enforced` FAIL，GPU 空闲后重跑即 PASS（VRAM 残留导致的 flaky）。



### 5.1 无回退证据（非 qwen35 栈）

- Coder-V2-Lite（MLA，官方 deepseek_v2 参考）：`549,6077,280,7239 → 185` ✓（与 git 前一致）
- qwen2.5-0.5B：`1000,1001 → 40716` ✓；decode 5.00 ms/token 与之前同量级
- 这两个模型完全不经过 gated-delta 内核；`gd_delta` 系列内核签名变化不影响它们。

### 5.2 性能

修复只改语义常量（`kh` 的取模/除法、`g` 的公式），不改数据布局、不改访存模式：
9B decode 22–24 ms/token、prefill 5 token ~190–300 ms 与修复前同量级；
35B decode ~79–97 ms/token、chat22 prefill ~0.9 s 与修复前同量级。

---

## 6. 测试盲区分析（为什么之前 51 项 + 四模型冒烟全绿）

1. **只测 position 0**：`smoke_models.sh` 的 9B/35B case 是 1 个 token 的 prompt。position 0
   处 attention 只有自身一个 key（q/k、rope 失效）、SSM 状态为 0（衰减 `g` 失效）、conv 只剩
   最后一个 tap ⇒ 两个 bug 在该位置几乎不可见，top-1 甚至与 llama.cpp 巧合一致。
2. **没有生成级断言**：`verify.sh` 全部是数值/夹具/IR/plan 测试，不含"真实文本 greedy 续写"。
3. **参考链选错**：`verify_reference.sh` 拿 HF 语义当 GGUF 的基准（见 §1），把真实的语义错位
   当成"与官方参考一致"。
4. **"4 token 冒烟正确"是误传**：旧 smoke 的 9B case 实际只喂 `760`；而"The capital of
   France" 的下一 token 恰好仍是 `is`，即使语义错也不翻 argmax。
5. **decode 也错了**：从 1-token prompt 连续 decode 时第 3 个 token 就偏离 llama.cpp
   （` following are multiple choice...` vs ` following are the first 10000`），
   所以不是"prefill batch 特有"的问题，靠 `ENGINE_SPLIT_PREFILL` 对比也定位不到。

**新防线**：`gen_regression.sh`（多 token prefill + 8–12 greedy token vs llama.cpp，逐 id）
+ smoke 脚本内的显式警告 + `verify_reference.sh` 的语义分路 + `q35_variants.py` 裁判工具。

---

## 7. 复现命令（供独立复核）

```bash
cd engine
bash build.sh
bash tools/verify.sh                    # 期望 ALL GREEN, exit 0（日志 build/verify-final-q35.log）
bash tools/gen_regression.sh            # 期望 10 passed, 0 failed（8 组 qwen35 + 2 守卫）
bash tools/smoke_models.sh              # 期望 40716 / 185 / 2614 / 2614
bash tools/verify_generality.sh         # 期望 "26 passed, 0 failed / GENERALITY GREEN"

# --- 单 case 手测（9B）---
./build/test_gen.exe "<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf" \
  --ids 760,6511,314,9338,369 --n 12
# 期望 gen: 11751 13 198 760 6511 314 9338 369 11751 13 198 760
#   -> " Paris.\nThe capital of France is Paris.\nThe"（= llama.cpp /completion 原文）

# --- 单 case 手测（35B，chat 模板 22 ids，thinking 开）---
ENGINE_GPU_EXPERTS=16 ./build/test_gen.exe \
  "<MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf" \
  --ids 248045,8678,198,2523,513,264,10631,17313,13,248046,198,248045,846,198,109266,248046,198,248045,74455,198,248068,198 \
  --n 8
# 期望 gen: 40 2688 28378 310 279 1156 579 40719 -> "I'm responding to the user's greeting"

# --- 根因证据：4 变体对照表（CPU，约 3-5 分钟）---
"python" tools/q35_variants.py \
  --model "<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf" --ids "760,6511,314,9338,369"
# 期望只有 V2_llama(-A,tile) 的 top1 = 11751（llama.cpp 的 " Paris"），
# V0（旧语义）top1 = 279 且 top5 与修复前引擎逐位一致。

# --- GGUF 语义逐层回归（约 2 分钟，期望 2e-7 级）---
ENGINE_SKIP_WARMUP=1 ENGINE_PREFILL_F16=0 ENGINE_DUMP_EVERY=build/ref9b \
  ./build/test_gen.exe "<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf" "760,6511,314,9338" 1 >/dev/null 2>&1
"python" tools/q35_variants.py --model "<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf" \
  --dump build/ref9b --ids "760,6511,314,9338" --a-fix 1 --bcast tile --layer 7 --tolerance 2e-4
# 期望：DUMP CHECK PASSED: max per-layer rel_l2 = 2.3e-7 ~ 4.5e-7 <= 2.0e-4

# --- 旧行为回退（A/B，证明根因单一）---
ENGINE_Q35_LOG_A=1 ENGINE_Q35_HF_BCAST=1 ./build/test_gen.exe \
  "<MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf" --ids 760,6511,314,9338,369 --n 10
# 期望回到旧错误输出：gen: 279 6511 314 9338 11 279 6511 314 279 6511
#   -> " the capital of France, the capital of the capital"

# --- llama.cpp ground truth 自取（<LLAMA_DIR>，可选）---
#   llama-server -m <gguf> -ngl 0 -c 1024 --port 8123
#   POST /completion {"prompt":"The capital of France is","n_predict":12,
#                     "temperature":0,"cache_prompt":false,"n_probs":1}
#   生成 id 取 completion_probabilities[].id
```


---

## 8. 文件清单

新增：
- `tools/gen_regression.sh` — 生成回归（llama.cpp golden，8 组）
- `tools/q35_variants.py` — 可切换语义的 numpy 裁判 / 逐层回归
- `QWEN35-SEMANTICS-FIX-REPORT.md` — 本报告

修改：
- `include/engine/kernels.h` — 5 个 delta 内核加 `kh_mode`
- `src/qwen35.cpp` — container 语义分路、A 约定默认值、delta 内核传参
- `tools/smoke_models.sh` — 35B 期望 69014→2614 + 盲区提醒
- `tools/verify_reference.sh` — qwen35 判据改为 GGUF/llama.cpp 语义
- `README.md` — 模型验证基准、已知差异、开关、防线说明
- 备份：`src/qwen35.cpp.pre-q35sem`、`include/engine/kernels.h.pre-q35sem`

---

## 9. 遗留问题 / 归属

1. **`test_gguf` 非法访存（曾使 verify.sh 50/51）已由 prefill-fast 修复**：该失败模型是
   qwen2.5-0.5B（24 层，无 SSM，不会启动任何 gated-delta 内核），隔离证据是
   `ENGINE_ATTN_PRE_ROWS=1`（关掉并发新增的 prefill attention 内核 `attn_tiled_pre`，
   `src/runner.cpp:887-903`）时 PASS、`ENGINE_NO_GRAPH=1` 时也 PASS；把证据转给
   prefill-fast 后其修复，最终验收 `verify.sh` 已 **ALL GREEN**。
2. 验收期间 `tools/verify_generality.sh` 被并发编辑保存成 CRLF 行尾（`for ...; do\r`
   导致 `bash -n` 语法错），已机械转回 LF；`post-load margin enforced` 曾因 GPU 残留
   显存 flaky 一次，空闲后重跑 26/26 PASS。
3. 历史报告（`FINAL-SPRINT-REPORT.md`、`SERVE-REPORT.md`、`SPEC-COMPLIANCE.md` 等）里
   出现的 `69014` 是**旧语义下的快照数据**，不再作为可引用结论；活动期望值只在
   `tools/smoke_models.sh` / `tools/gen_regression.sh` 中维护。
4. `tools/util_results*/**/measurement.engine.log` 里还有旧语义的 `69014` 记录，属历史
   测量日志，未改（不影响任何测试）。

