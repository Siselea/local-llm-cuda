# HTTP 服务 + 分词器 + 采样器 交付报告

> 任务：让自研 C++17+CUDA 引擎"真的能用起来"——端口 7030 跑 Qwen3.6-35B-A3B，
> 浏览器/客户端发文本提示词对话。交付：GGUF 分词器、采样器、OpenAI 兼容 + llama.cpp
> 风格 HTTP 服务、配置项、验证脚本与文档。
>
> 报告状态：**进行中**（端到端 35B 验证等待 GPU 窗口；标记 `[待跑]` 的条目会在服务
> 端到端测试后补齐真实输出）。

---

## 0. 结论先行

| 交付物 | 状态 |
|---|---|
| GGUF 分词器（BPE + SPM）：编码/解码/特殊 token/ChatML 模板 | **完成**，与 llama.cpp `llama-tokenize` 逐 token 一致（415 token 合成语料 + 15,500+ token 真实文档 + chat 模板提示词） |
| 采样器（temperature/top-k/top-p/min-p/repetition penalty/seed） | **完成**，13 项单元测试全绿 |
| `engine serve`（OpenAI `/v1/chat/completions` 流式+非流式、`/v1/completions`、llama.cpp `/completion`、`/health` `/v1/models` `/props` `/tokenize` `/detokenize`、浏览器 UI） | **完成**，HTTP 层 12 项纯逻辑单测全绿 |
| 0.5B 轻量端到端 `verify_serve.sh`（7031，22 项） | **完成 GREEN**（22/22；含与 test_gen greedy 逐 token 对拍、SSE、错误码、优雅退出） |
| 35B 端口 7030 端到端（中文/多轮/流式/中等 prompt/与 test_gen 对拍/优雅退出） | **最终验收 GREEN：17/17 PASS**（bug 修复后；首轮 11 PASS / 5 FAIL 的失败全部定位为测试脚本 bug + 引擎 bug，均已修，见 §6/§9） |
| 最终验收（0.5B / 9B / 35B / 4k 长上下文） | **FINAL ACCEPTANCE GREEN（4/4 档全 PASS）**，日志 `build/final-acceptance.log` |
| 三套硬性验证（`verify.sh` / `smoke_models.sh` / `verify_generality.sh`） | lead 统一验收已确认：**ALL GREEN / 四模型预测 40716·185·2614·69014 全对 / GENERALITY GREEN** |
| 配置项 + 文档 | **完成**（`serve_host`/`serve_port`/`serve_ctx`/`default_*` 8 个新键；`docs/SERVE.md`、`docs/CONFIG.md §3.7 §7`、`tools/README.md`） |

---

## 1. 分词器

### 1.1 实现

- 新增 `include/engine/tokenizer.h` + `src/tokenizer.cpp`（约 1000 行，零第三方依赖）。
- **BPE（`tokenizer.ggml.model=gpt2`）**：GPT-2 byte-level unicode 映射 + merges rank 贪心合并；
  `pre-tokenizer` 是 llama.cpp b12128 的**精确端口**：
  - `qwen35`：`(?:'[sS]|…)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    → `unicode_regex_split_custom_qwen35()` 的手写等价实现（含 M 类合并记号）；
  - `qwen2`：同上但字母类不含 `\p{M}`；
  - `gpt2`/未知 pre：经典 GPT-2 splitter（文档标注为 best-effort）。
- **SPM（`model=llama`）**：▁ metaspace 转义 + `add_space_prefix` + 按 score 的 bigram
  贪心合并 + 反查 resegment + `<0xNN>` byte fallback（移植 `llm_tokenizer_spm_session`）。
- **解码**：byte-level 反向映射后**按字节拼接再统一 UTF-8 解码**（token 可能截断多字节字符）；
  控制/未知 token 跳过、user-defined token 原样输出；SPM 的 ▁→空格、`<0xNN>`→字节。
- **特殊 token**：CONTROL/USER_DEFINED/UNKNOWN 按文本最长优先匹配，编码时保持原子
  （`<|im_start|>` 等不会被普通切分打散）；`parse_special=false` 不识别 control（同 llama.cpp）。
- **Unicode 表**：`tools/gen_unicode_tables.py` 从 Python `unicodedata` 生成
  `include/engine/unicode_tables.h`（2270 个区间，llama.cpp `unicode_cpt_flags` 兼容的
  general-category bits + 精确 whitespace 集合），`--check` 可验证新鲜度。
- **Chat template（Qwen ChatML 子集）**：system 合并（前两条 system/developer 用 `\n` 合并）、
  user/assistant/tool 文本消息、consecutive tool 共用一个 user turn、assistant 历史
  `reasoning_content`/`</think>` 拆分、`add_generation_prompt`、
  `enable_thinking`（true → `<think>\n`；false → 空 think 前缀）。
  生成提示与 Qwen3 格式一致：`<|im_start|>assistant\n<think>\n`。

### 1.2 正确性证据

**（a）与 llama.cpp 官方分词器逐 token 对拍**（`tools/check_tokenizer.py cpp`，
llama-tokenize = `<LLAMA_DIR>/llama-tokenize.exe` b12128，与 GGUF 同一 tokenizer）：

```
pre-tokenizer  : 'qwen35' (implemented)
cpp encode : 415 tokens
oracle     : 415 tokens
encode vs oracle : MATCH                    # 合成语料（中英日韩/emoji/组合记号/ZWJ/NBSP/代码/数字/空白）
real file README.md: 4243 tokens -> MATCH
real file docs/CONFIG.md: 6227 tokens -> MATCH
real file FINAL-SPRINT-REPORT.md: 5057 tokens -> MATCH
decode roundtrip : OK
special encode vs oracle : MATCH  ids=[248045, 846, 198, 5834, 248046, 198]
special tokens : OK
chat template render : OK (118 bytes)
chat prompt token ids vs oracle : MATCH (22 tokens)
chat prompt ids : [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198, 248045] ...
enable_thinking=false : OK (empty think prefill)
assistant reasoning render : OK
ALL TOKENIZER CHECKS PASSED
```

同一验证在 **Qwen3.6-35B-A3B（248320 词表）** 与 **Qwen3.5-9B / Qwythos-9B（同词表）** 上
全部 ALL PASSED；Qwen2.5-0.5B（`pre=qwen2`）同样 ALL PASSED（394/394 token 一致）。
DeepSeek-Coder-V2（`pre=deepseek-llm`）不需要本服务，其 pre-tokenizer 未实现：
encode 与 llama.cpp 有差异（466 vs 465 token，已在报告与脚本中标注为已知限制），
decode 往返正确。

**（b）ChatML 模板渲染样例**（35B，system+user，thinking 开）：
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n你好<|im_end|>\n
<|im_start|>assistant\n<think>\n
```
token id：`248045 8678 198 2523 513 264 10631 17313 13 248046 198 248045 846 198 109266 248046 198 248045 74455 198 248068 198`
（248045=`<|im_start|>`，248046=`<|im_end|>`，248068=`<think>`；与 llama-tokenize 完全一致）

**（c）往返测试**：`decode(encode(s)) == s` 对合成语料 + 3 份真实文档（15,500+ token）
逐字节相等；`tests/test_tokenizer.cpp --self-test` 是自包含验收（不依赖外部 oracle）。

**（d）一键验收**：`bash tools/verify_tokenizer.sh` → **TOKENIZER GREEN（12/12）**：
5 个本地 GGUF（35B / 9B / Qwythos / 0.5B / Coder）的 self-test + 与 llama-tokenize 的
逐 token 对拍（Coder 的 `deepseek-llm` pre 未实现，对拍按已知限制跳过并标注）+ Unicode 表
新鲜度 + ids-file/CLI 往返。完整日志 `build/logs/verify_tokenizer.log`。

### 1.3 工具/脚本

`tools/dump_tok.py`（dump GGUF tokenizer.* 键）、`tools/tok_ref.py`（Python 参考实现 +
候选 pre-tokenizer 实验）、`tools/check_tokenizer.py`（oracle 对拍）、
`tools/gen_unicode_tables.py`、`tests/test_tokenizer.cpp`、`tools/verify_tokenizer.sh`。

---

## 2. 采样器

`include/engine/sampler.h` + `src/sampler.cpp`：自带 xorshift128+ PRNG（splitmix64 扩种），
temperature / top_k / top_p(nucleus) / min_p / repetition_penalty(含 repeat_last_n 窗口)，
`temperature<=0` → greedy；同 seed + 同参数可复现。
`tests/test_sampler.exe` 13 项全 PASS（seed 复现、greedy、top-k 只从 top3 采样、
top-p 锁定主导 token、flat 分布覆盖、min-p 过滤、重复惩罚>95% 转移、窗口过期）。

---

## 3. HTTP 服务

- `include/engine/http_server.h` + `src/http_server.cpp`：自写 WinSock2 HTTP/1.1，
  `ws2_32.dll` **运行时动态加载**（因此 `build.sh` 的编译/链接选项一个都没改）。
  支持 keep-alive、Content-Length、chunked 流式（SSE）、CORS、pipelined 请求、
  HTTP/1.0 close 语义、100-continue、优雅 stop。
- `include/engine/serve.h` + `src/serve.cpp`：模型/分词器加载、路由、生成循环、SSE 分片、
  stop 字符串跨 token holdback、UTF-8 边界安全流式、思考段拆分（reasoning_content/content）、
  请求日志、`/shutdown`（仅回环）、Ctrl+C/Ctrl+Break/关窗口优雅退出并释放显存。
- 单进程只加载一次模型；请求串行（单用户）。

**HTTP 层纯逻辑单测**（`bash tools/verify_http.sh`，不加载模型、不用 GPU）：
12/12 PASS —— GET、keep-alive 复用、128 KiB POST 往返、chunked 5 分片完整收到、
418/404 JSON error、CORS 预检 204、畸形请求行→400、HTTP/1.0 body 后关闭、
pipelined 两个响应、`/shutdown` 200 + 端口释放。

**0.5B 轻量端到端**（`bash tools/verify_serve.sh`，7031 端口）：**SERVE VERIFY GREEN 22/22**。
真实输出片段：

```
startup line: listening on http://127.0.0.1:7031 (model=<MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf, ctx=1024) PASS
GET /health PASS | GET /v1/models PASS | GET /props PASS | GET / browser chat UI PASS
POST /tokenize -> [9707,1879,11,220,108386,99489,0] PASS | POST /detokenize round trip PASS
POST /v1/chat/completions -> content='OK' finish=stop PASS
SSE stream: 10 chunks + [DONE] PASS
POST /v1/completions PASS | POST /completion (llama.cpp style) PASS | POST /completion SSE stream PASS
GET /nope -> 404 PASS | empty messages -> 400 PASS | invalid JSON -> 400 PASS
prompt > ctx -> 400 with clear error PASS | server still healthy after error responses PASS
server greedy ids == test_gen ids [12095,13,1084,374,279,7772] PASS
POST /shutdown -> 200 PASS | process exits cleanly (rc=0) PASS
VRAM released message logged PASS | port released after shutdown PASS
results: 22 passed, 0 failed -> SERVE VERIFY GREEN
```

首轮跑出 21/22：**无效 JSON 被当成 500**（真实 bug，已修：`parse_json_body()` 把 JSON
语法错误转成 400 UserError），修复后重跑 22/22。

**配置项**（`src/config.cpp`，全部进统一配置系统 + `docs/CONFIG.md §3.7`）：
`serve_host`（默认 127.0.0.1，CLI `--host`）、`serve_port`（7030，`--port`）、
`serve_ctx`（0=用 --ctx 否则 4096）、`default_max_tokens`（256）、`default_temperature`（0.8）、
`default_top_p`（0.95）、`default_top_k`（40）、`default_repeat_penalty`（1.0）。

---

## 4. 启动命令

```bash
# 35B，端口 7030（日常使用）
./build/engine.exe serve \
  <MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf \
  --host 127.0.0.1 --port 7030 --ctx 4096
# -> listening on http://127.0.0.1:7030 (model=..., ctx=4096)
# 浏览器打开 http://127.0.0.1:7030/ 即可聊天
```

详见 [docs/SERVE.md](../SERVE.md)（curl/浏览器/OpenAI SDK 示例、参数、已知限制）。

---

## 5. 三套硬性验证（lead 统一验收，2026-11-15 09:20 全量重建后）

| 验证 | 结果 |
|---|---|
| `bash build.sh` | 19 个二进制全部重建成功（含新增的 `engine serve` / `test_tokenizer` / `test_sampler`） |
| `bash tools/verify.sh` | **ALL GREEN**（kv-longctx 扩到 64 PASS） |
| `bash tools/smoke_models.sh` | 四模型预测 **40716 / 185 / 2614 / 69014 全对** |
| `bash tools/verify_generality.sh` | **GENERALITY GREEN** |

> ⚠️ 注意 smoke_models 只校验**第一个生成 token**，这正是 qwen35 多 token prefill bug
> 长期没被发现的原因（见 §7）。服务层的 `tools/verify_serve.sh` / `serve_35b_e2e.sh`
> 会额外做多 token 推理与对话内容检查。
>
> 本轮新增的自测（不属于上述三套，由本任务交付）：
> `tools/verify_http.sh` 12/12 GREEN、`tools/verify_tokenizer.sh` TOKENIZER GREEN 12/12、
> `tools/verify_serve.sh` SERVE VERIFY GREEN 22/22（0.5B）。
> **输出质量闸门**：`tools/check_output_quality.py`（空输出 / 符号汤 / 复读 / 复述 prompt →
> FAIL）已内嵌进 e2e 脚本——第一轮 35B e2e 的"HTTP 200 但内容是符号汤"就是因为没有这道闸门
> 才看起来像大部分通过。固定 fixtures 验证：35B 真实符号汤 artifacts → QUALITY FAIL；
> 9B 模板复读 → FAIL；正常中文/英文回答 → OK。

---

## 6. 35B 端到端（最终验收，2026-11-15 11:45–11:52，RTX 3060 12GB）

命令：`bash tools/final_acceptance.sh --with-longctx`（内部 35B 档 =
`engine serve <35B> --host 127.0.0.1 --port 7030 --ctx 4096 --gpu-layers 16`）。
完整日志：[build/final-acceptance.log](build/final-acceptance.log)。结果：**17/17 PASS，35B E2E GREEN**。

### 6.1 真实中文对话原文（lead 要求）

**中文单轮（non-stream，temperature 0.8 / top_p 0.95 / top_k 40 / max_tokens 192，thinking 开）**

```
输入 messages:
  system: 你是一个简洁的中文助手，回答不超过 60 字。
  user  : 你好，请用一句话介绍你自己。
输出 content  : "\n\n你好！我是简洁的中文助手，擅长用精炼的语言为你解答各种问题。"
输出 reasoning: "用户要求用一句话介绍自己，回答不超过 60 字。我是中文助手，简洁明了地回应。\n"
finish_reason : stop
usage/timings : prompt 36 tok, completion 43 tok; prefill 1180 ms, decode 2278 ms (18.9 tok/s)
```

**多轮 messages（system / user / assistant / user，temperature 0，enable_thinking=false）**

```
输入:
  system   : 你是一个助手。
  user     : 法国的首都是哪里？
  assistant: 巴黎。
  user     : 那德国呢？
输出 content : "德国的首都是柏林。"
finish_reason: stop
usage/timings: prompt 42 tok, completion 5 tok; prefill 1073 ms, decode 260 ms (19.3 tok/s)
```

**stream=true SSE（中文，enable_thinking=false，max_tokens 64）**

```
输入: user: 数到五，用中文，只输出数字。
SSE : 3 个 data 分片（role delta / content delta "五" / finish_reason=stop）+ data: [DONE]
每个 data: 是单行压缩 JSON（本次验收修掉了早期 pretty JSON 多行的兼容问题）
```

**4k 长上下文（ctx 8192，默认 f32 KV，未用 --kv-type）**

```
输入: ~4023 token 中文 prompt（重复的"长上下文预填充测试"文本 + "请只回复：收到。"）
输出 content : "收到。"   finish: stop
timings     : prefill 105479 ms（38 tok/s over 4004 tokens）, decode 362 ms（2 tok）
结论        : 不崩溃、答案正确；但 4k prompt 首字延迟 ~106 s（引擎 prefill 仍是瓶颈，如实记录）
```

### 6.2 性能（最终构建）

| 项目 | 数值 |
|---|---|
| 模型加载 | 53.7 s（ctx 4096）/ 52.6 s（ctx 8192） |
| 冷启动首个请求 | 3.14 s（含懒 CUDA 初始化/图捕获） |
| prefill（36 tok，预热后） | 1.18 s（30 tok/s） |
| prefill（348 tok） | 9.16 s（38 tok/s） |
| prefill（573 tok） | 12.0 s（48 tok/s） |
| prefill（4k，4023 tok） | 105.5 s（38 tok/s） |
| decode | 18.9–22.1 tok/s（本构建；历史构建 24–27 tok/s） |
| 显存（运行中） | ~11.8 GB（12 GB 卡，ctx 4096，KV f32） |
| 退出后显存 / 端口 / 进程 | 457 MiB / 已释放 / 无残留（脚本自检） |

### 6.3 通过项（17）

启动行(model/ctx)、`/health`、`/props`（arch/quant_mix/attn/MoE/tokenizer/chat_template）、
浏览器首页、冷请求、中文单轮、多轮、SSE 流式、`/tokenize`+`/detokenize` 中文往返、
348/573 token 中等 prompt（跨越历史 >384 崩溃边界）、**greedy 与 test_gen 逐 token 对拍
（`[40,2688,28378,310,279,1156,579,40719]` 完全一致）**、输出质量闸门（符号汤/复读/复述判 FAIL）、
`/shutdown` 优雅退出、显存释放、端口释放。

> 第一轮 35B e2e 是 11 PASS / 5 FAIL：其中 4 个是我测试脚本的 GBK/引号 bug（已修）、
> 1 个是 thinking 吃光 token（已改成关思考/给足预算）、**其余是引擎 qwen35 prefill bug**
> （见 §7，修复后本档变 17/17 GREEN）。另外验收中发现并修复了一个**真实服务 bug**：
> SSE 事件原本是多行 pretty JSON，已改为单行压缩 JSON（见 §9）。

## 7. 发现的引擎级 bug（本轮最有价值产出）

**现象**：qwen35 混合架构模型（Qwen3.5-9B / Qwen3.6-35B-A3B）在 prompt **>1 token** 时
生成乱码：9B raw `The capital of France is` → `" the capital of France, the capital of the capital"`；
35B 中文对话 → 数字/符号汤（96 token 无 CJK、thinking 不闭合）。

**证据（最小复现已交 `qwen35-prefill-bug`：`tools/repro_qwen35_prefill.sh`）**：
- llama.cpp（llama-server，同 GGUF，temp=0）：`" Paris.\nThe capital of France is Paris."`
- 本引擎：首 token `279(" the")`，llama.cpp 是 `" Paris"` ⇒ **第一个生成 token 就错**
- `ENGINE_SPLIT_PREFILL=1`（逐 token 喂）给出**同样错误**的 top-1 ⇒ 不是新 batch prefill kernel
- 单 token prompt（ids 760）预测 2614，与 llama.cpp 一致 ⇒ 旧 smoke 测试的盲区
- `build/test_gen.pre.exe` / `test_gen_pre.exe` / `test_gen_pre_opt.exe` 三个旧二进制
  输出**完全相同** ⇒ 既有 bug，不是 2026-11-15 当天改动引入
- 对照：qwen2.5-0.5B（dense attention）多步对话正常（'OK' 正确）

**定位方向**：pos=0 logits 正确、>1 position 后错误，且 batch/逐 token 一致
⇒ 位置/状态累积（KV 位置或 qwen35 hybrid conv/rec state）在 >1 position 时算错。
**状态：已修复（2026-11-15）。** 根因是两处 GGUF 张量语义错位：
1. `ssm_a` 张量存的是 `-A` 而不是 `log A`（qwen35 gated-delta-net 的衰减项）；
2. q/k→v 的 head 广播是块重复 `h % n_kh`，按 GGUF 语义应为 `h // rep`。
`qwen35-prefill-bug` 修复后验证：9B raw 用例与 llama.cpp **逐字节一致**
（`" Paris.\nThe capital of France is Paris."`），单 token 冒烟 2614 仍正确；
35B 中文对话、多轮、SSE 全部正常，最终验收 4 档全绿（§9）。

## 8. 已知限制

1. **首字延迟（引擎 prefill）**：服务每次请求整段预填充，35B 实测 pp 38–48 tok/s
   （36 tok → 1.18 s；348 tok → 9.16 s；573 tok → 12.0 s；**4023 tok → 105.5 s**），
   冷启动首请求还要 +3 s 懒初始化。响应 `timings.prefill_ms` 如实给出。
   9B 的稠密 prefill 已被 fp16 GEMM 优化到 ~3.3×（`engine-prefill-fp16-gemm`），
   35B MoE（CPU 专家）仍是瓶颈——这是后续工作，不是 HTTP 层能修的。
2. **单用户串行**、**无跨请求 KV 复用**（每轮重新预填 history）。
3. **chat 模板是子集**：无 tools/vision/完整 jinja；非 ChatML 模板的模型返回 400。
4. **分词器**：`qwen35`/`qwen2` 精确；其他 pre 名回退 GPT-2 规则。
5. **无鉴权**：默认仅回环；绑 0.0.0.0 时任何人可用。
6. `--ctx` 是硬上限（prompt 超限 400，不截断）。

---

## 9. 最终验收（2026-11-15 11:45–11:52）

一键命令与完整日志：

```bash
bash tools/final_acceptance.sh --with-longctx
# 完整日志: build/final-acceptance.log
```

| 档 | 命令 | 结果 | 耗时 | 备注 |
|---|---|---|---|---|
| 0.5B HTTP e2e | `verify_serve.sh`（7031, ctx 1024） | **GREEN 21/21** | ~50 s | 含 greedy 与 test_gen 对拍 `[12095,13,1084,374,279,7772]` |
| 9B HTTP e2e | `SKIP_PARITY=1 verify_serve.sh <9B> 7032` | **GREEN 21/21** | ~40 s | 中文/多轮/SSE/错误码/优雅退出；单卡放不下双份 9B 故跳过 test_gen 对拍 |
| 35B e2e | `serve_35b_e2e.sh`（7030, ctx 4096, fp32 KV） | **GREEN 17/17** | ~6.5 min | 中文单轮/多轮/SSE/348·573tok/greedy 对拍/质量闸门/优雅退出 |
| 35B 4k 长上下文 | `serve_longctx_check.sh <35B> 7034 8192 4000` | **GREEN 5/5** | ~3.5 min | ctx 8192 **f32 KV 直接装下**（未用 `--kv-type q8_0`）；prefill 105.5 s，答案"收到。" |
| **总计** | | **FINAL ACCEPTANCE GREEN（4/4 档）** | ~11.5 min | |

**收尾自检（脚本自动打印）**：`leftover model processes: (空)`、`nvidia-smi memory.used = 457 MiB`、
端口 7030/7031/7032/7034 全部释放；每个 server 都走 `/shutdown` → rc=0 → 打印
`stopped; VRAM released`。

**本轮验收中发现并修复的问题**（全部有回归证据）：
1. **真实服务 bug（已修）**：SSE 事件原本是多行 pretty JSON（`json::Writer` 带缩进换行），
   只按行解析 `data:` 的客户端会读不到中间分片；`sse_data()` 现在压缩成单行 JSON
   （保留 `data: [DONE]` 终止符）——修复后 35B SSE 测试从 FAIL 变 PASS。
2. 测试脚本 bug ×3（已修）：GBK 控制台下 `jget` 打印中文破坏字符串比较（改为
   `ensure_ascii=True` / python 内部比较）；`/v1/models` 硬编码 0.5B 模型名；
   以及 `jget` 返回值带引号导致 token-id 字符串比较失败。
3. 测试设计问题（已修）：thinking 打开的 short max_tokens 会被思考吃光 → 相关测试改用
   `enable_thinking=false` 或给足预算；新增**输出质量闸门**（符号汤/复读/复述 prompt → FAIL），
   第一轮的"HTTP 200 但内容是符号汤"现在不会再被误判为 PASS。

---

# 第 2 轮（2026-11-16）：webui 思考可见 + 多轮前缀缓存

用户实际用 **launcher + llama.cpp b6000 webui** 跑 35B 聊天，反馈两个问题：
（1）一直转圈、很久才出第一个字；（2）没有前缀缓存，每轮都把整个对话重新 prefill，越聊越慢。
本轮针对这两个问题（外加两个小项）修改并实测，**四套验证全绿**。

## 根因

1. **webui 只渲染 `content`**：llama.cpp 的 webui 只取 `choices[0].delta.content`，
   然后在客户端按 `<think>`/`</think>` 标签折叠思考段，**完全不读 `reasoning_content`**。
   本服务原来把思考只放 `reasoning_content`，于是思考阶段 `content` 为空 → 页面一直转圈，
   直到正式回答开始才有字。另外 webui 默认发 `max_tokens:-1`，原来映射成 ctx 上限（4096），
   思考模型一条回复可能跑几分钟。
2. **无跨请求 KV 复用**：每个请求都从 `pos=0` 整段 prefill；35B 的 pp 只有 48–55 tok/s，
   第 5 轮 1250 token 的历史 ≈ 25 s 白等。混合 qwen35 还有 SSM conv/rec 状态，
   单纯复用 KV 不够。

## 修改

| 位置 | 改动 |
|---|---|
| `src/serve.cpp`（流式/非流式 chat、completions） | 思考段**同时**写进 `content`：`<think>思考…</think>` + 正式回答（webui 约定，可折叠）；`reasoning_content` 原样保留给 OpenAI 客户端。思考被长度截断时补发 `</think>`。 |
| `src/serve.cpp`（`params_from_body`） | `max_tokens<=0`（webui 的 `-1`）→ `serve_no_limit_tokens`（默认 **1024**），不再按 ctx 4096 无限生成。 |
| `src/config.cpp` | 新增 `serve_prefix_cache`（`--prefix-cache[=0]`，默认 true）与 `serve_no_limit_tokens`（`--no-limit-tokens`，默认 1024）。 |
| `include/engine/runner.h` + `src/runner.cpp` | 新增前缀缓存状态 API：`prefix_state_supported()/bytes()/save()/restore()`（复用 `src/mtp.cpp` 里为投机解码准备的 SSM rollback 槽位；纯 attention 模型是 no-op）+ `prefill_chunk()`。 |
| `src/serve.cpp`（`generate`） | 计算"对话历史/生成提示"边界 → 与上一轮快照求 LCP → 命中则恢复状态、从 LCP 继续 prefill；快照/切分点**对齐到 prefill chunk 边界**，使缓存路径与整段 prefill 的 chunk 序列完全相同（bit-exact 关键）。未命中则整段 prefill 后重新拍快照。 |
| `src/serve.cpp`（启动） | 未显式设置 `prefill_rows` 时，serve 默认钉在 **128**（缓存粒度 <128 token，且 activation/logits 缓冲更小、能多放 GPU 专家）；`/props` 暴露 `prefix_cache` 统计、`no_limit_tokens`；日志 `[req] ... prefill X ms [new N, cached M]`。 |
| `tools/` | 新增 `serve_webui_probe.py`（webui 行为/首字探针）与 `serve_prefix_cache_check.py`（A/B 逐 token 硬验收）。 |

## 实测证据（35B，ctx 4096，RTX 3060 12GB，自适应 15 层 GPU 专家，launcher 同款启动参数）

**webui 首字（思考段第一个 delta，`tools/serve_webui_probe.py`）**

| 轮次 | 首字 delta | prefill | 说明 |
|---|---|---|---|
| 1（15 token prompt） | **0.99 s** | 963 ms | 冷启动首请求 |
| 2（76 token） | **1.48 s** | 1467 ms | 思考段内容随首 delta 一起出现 |
| 3（107 token） | **1.94 s** | 1938 ms | 思考段在 `content` 里、`</think>` 后是回答 |

修改前：思考阶段 `content` 始终为空，页面无任何字（要等思考结束/几十秒），即用户说的"一直转圈"。

**多轮前缀缓存 A/B（`tools/serve_prefix_cache_check.py --turns 8`）** —— 每轮 `cache_prompt=false`
（整段 prefill）与 `true`（缓存）**贪心 token 序列逐 token 对比全部 SAME**：

| 轮 | prompt | 整段 prefill | 缓存 prefill | 复用 |
|---|---|---|---|---|
| 1 | 24 | 549 ms | 544 ms | 0（冷） |
| 4 | 225 | 4182 ms | 1810 ms | 128 |
| 6 | 353 | 6601 ms | 1978 ms | 256 |
| **8** | **478** | **9440 ms** | **1773 ms（5.3×）** | **384** |

同一 1331-token prompt 再问一次：**25.9 s → 1.07 s（复用 1280 token）**。
`PREFIX CACHE GREEN`（4 项检查全 PASS）。

## 已知限制（如实）

1. 前缀缓存是**单槽 + 128 token 对齐**的：只保留上一轮历史快照，每轮多 prefill 最多 ~128 token
   的对齐余量；编辑历史/重新生成会退化为整段 prefill。多槽/主机内存快照是后续优化。
2. 单轮首字仍受 35B prefill 速度限制（pp 48–55 tok/s）：第一轮 15 token ≈ 1.0 s，
   **<0.5 s 目标未达到**——需要继续提 35B 的 pp（9B 稠密线的 prefill-fast 已 3.3×，35B 未吃到）。
3. 思考模型在 1024 token 上限内仍可能把预算耗在思考上（finish=length，回答为空）；
   服务端会补 `</think>` 让 webui 结束"思考中"状态，如需更长回答可调 `--no-limit-tokens`。

## 验证（全绿）

| 验证 | 结果 |
|---|---|
| `tools/verify.sh` | **ALL GREEN**（build + fixtures + Python 单测 + G0/M2/M3 IR 与 CUDA 对拍） |
| `tools/smoke_models.sh` | **PASS**（0.5B 40716 / Coder 185 / 9B 2614 / 35B 2614，与既有参考点一致） |
| `tools/gen_regression.sh` | **GEN REGRESSION GREEN 10/10**（qwen35 真文本逐 id 回归） |
| `tools/verify_generality.sh` | **GENERALITY GREEN 26/26** |
| `tools/verify_serve.sh`（0.5B@7031） | **SERVE VERIFY GREEN 22/22**（含 greedy 与 test_gen 对拍） |
| `SKIP_PARITY=1 tools/verify_serve.sh <9B> 7032` | **SERVE VERIFY GREEN 21/21**（混合模型走新缓存路径） |
| `tools/serve_35b_e2e.sh`（35B@7030） | **35B E2E GREEN 17/17**（中文单轮/多轮/SSE/中等 prompt/greedy 对拍/优雅退出+显存 458 MiB） |
| `tools/serve_webui_probe.py` | **WEBUI PROBE OK**（首字可见、思考在 `content` 里、答案干净） |
| `tools/serve_prefix_cache_check.py --turns 8` | **PREFIX CACHE GREEN**（逐 token SAME + 5.3×） |

本轮改动只涉及 serve 层（`src/serve.cpp`、`src/config.cpp`）与 runner 的新增快照 API
（`include/engine/runner.h`、`src/runner.cpp`），**不改 kernels.h / build.sh 编译选项 /
src/mtp.cpp / src/launcher.cpp**。旧测试脚本里两处 `GET /` 断言仍期待内嵌页面的 "local chat"，
与 llama.cpp webui 静态页不符，已改为接受 `local chat|llama.cpp`（webui 轮遗留的过期期望）。

---

# 第 3 轮（2026-11-16）：思考阶段"漂移/复读"根因判定与兜底

用户用 launcher + llama.cpp webui 跑 9B/35B 中文对话，反馈**思考段一开始就漂移/复读**
（9B "I should be helpful." 重复 10+ 次；35B "The user's message seems to be asking me to
continue with the novel." 重复多段）；另有一条线索：webui 侧栏会话标题显示成 `n你好`。

## 根因判定（按"先排除、再对照"的顺序，每步都有数据）

1. **不是 prefix cache**（第 2 轮加的缓存）：同一段 2 轮对话用 `cache_prompt=false`
   （强制每轮整段 prefill）在 35B 上仍然复读（回答里 "classical gardens, classical gardens…"
   连续重复）。缓存开关前后一致 ⇒ 排除。
2. **不是 sampler / 解码**：同一 chat prompt（带 `<think>` 生成模板）+ **greedy（temperature=0）
   生成 80 token**，引擎输出与 llama.cpp b12128 **文本完全一致**（271 字节，逐字相同；
   prompt 18 token，双方生成 80 token）。greedy 不漂、且与 llama.cpp 同文 ⇒ 解码/采样链一致。
3. **是模型/参数行为**：**同一 GGUF、同一 prompt、同一采样参数**（temp 0.8 / top_k 40 /
   top_p 0.95 / min_p 0.05）跑 llama.cpp b12128 的 webui 接口 —— 它同样把 512 token 预算
   全耗在 "Thinking Process: 1. Analyze… 2. Draft… 3. Refine…" 上，**始终不输出 `</think>`、
   回答为空**。模型在 webui 默认温度下会陷入自述循环，llama.cpp 也一样。
4. **模板/分词排除**：`tools/check_tokenizer.py cpp` 对 9B/35B 全 PASS —— 415 token 合成语料、
   README/CONFIG 全文与 `llama-tokenize` **逐 token MATCH**；thinking 生成模板
   `...<|im_start|>assistant\n<think>\n` 渲染 + token ids 也 MATCH（22 token）；
   detokenize roundtrip、特殊 token 原子性 OK。
5. **`n你好` 不在服务端**：真实浏览器键入 `你好，请用三句话介绍杭州。` 发送后，侧栏标题就是
   这整句（无 `n` 前缀）；9B/35B 两轮均一致。`/detokenize` 的 roundtrip 也通过，
   保存的会话名 = 用户输入原文（webui 本地 IndexedDB 生成），该现象更像是输入侧/历史数据的
   偶发现象，服务端无对应路径。

## 修改

| 位置 | 改动 |
|---|---|
| `src/sampler.cpp` | **采样链顺序对齐 llama.cpp**：`penalties → top_k → top_p → min_p → temp → dist`（原实现先 temp 再 top_p/min_p，且 min_p 在 top_p 前）。top_p/min_p 现在在**未温度缩放**的 softmax 概率上过滤，最后才按 `logit/T` 采样；greedy 路径不变。 |
| `src/serve.cpp` | **思考预算兜底**（`serve_think_budget`，默认 512）：思考超过 N token 仍未闭合 ⇒ 发 `</think>` 结束思考，然后**用关闭思考的模板**（`<|im_start|>assistant\n<think>\n\n</think>\n\n`）重新渲染同一对话、只生成回答，追加到同一响应流（前缀缓存命中，二次 prefill 很便宜）。避免"硬插 `</think>` 继续生成"把规划文本当回答输出。 |
| `src/serve.cpp` | **复读兜底**（`serve_repeat_guard`，默认开）：连续 3 句完全相同（≥8 字符）即 `finish_reason=stop`，不让"同一句复读 10 遍"刷满 token 上限；`timings.repeat_guard_hit` / 日志 WARN 可观测。 |
| `src/serve.cpp` | **推荐采样默认值**：请求未显式给参数时 temp 0.6 / top_k 20 / top_p 0.95（Qwen3 官方推荐），`/props.default_generation_settings` 体现；`timings.think_budget_hit/repeat_guard_hit`、`/props.think_budget/repeat_guard` 可观测。 |
| `src/config.cpp` | 新增 `serve_think_budget`（`--think-budget`）、`serve_repeat_guard`（`--repeat-guard=0` 可关）。 |
| `build/webui/index.html` | 托管的 llama.cpp webui 默认采样同步改 `temp 0.6 / top_k 20 / top_p 0.95 / min_p 0`，并对旧 localStorage 配置（0.8/40/0.95/0.05）做一次性迁移（老用户刷新即生效）。 |
| `tools/` | 新增 `webui_e2e.py`（headless Edge + CDP 真实打字/点发送/截图/读 DOM）与 `serve_drift_probe.py`（思考复读统计探针）。 |

## 真实验收（用户规矩：像真实用户一样在浏览器里打字对话）

`python tools/webui_e2e.py`：headless Edge/CDP 打开 `http://127.0.0.1:PORT/`，在输入框
`Input.insertText` 键入中文 → `Input.dispatchMouseEvent` 点击发送按钮 → 轮询流式渲染 →
每轮截图 → 读取侧栏标题与消息 DOM。

| 模型 | 键入文字 | 首字 | 总耗时 | 思考段 | 回答 | 标题 |
|---|---|---|---|---|---|---|
| 9B（ctx 4096，3 轮） | 你好，请用三句话介绍杭州。／再详细一点，重点说说它的历史和文化。／用同样的风格介绍一下苏州。 | 3.89 / 0.42 / 1.03 s | 20.5 / 19.0 / 19.9 s | 规划段落，单轮同一句最多 1 次，无 5+ 重复 | 3 句/3 段中文正常 | `你好，请用三句话介绍杭州。`（正确） |
| 35B（ctx 4096，2 轮） | 你好，请用三句话介绍杭州。／用同样的风格介绍一下苏州。 | 3.44 / 1.79 s | 27.2 / 29.7 s | 规划段落，无 5+ 重复 | 3 句中文正常（无复读） | `你好，请用三句话介绍杭州。`（正确） |

对照（改前同一 35B 对话第 2 轮）：回答里同一句重复 **7 次**、耗时 **45.4 s**；改后干净、29.7 s。
截图：`build/logs/e2e_9b_final/`（3 轮 typed/done）、`build/logs/e2e_35b_final/`（2 轮），
DOM/延迟明细在各自 `e2e_report.json`。服务端日志对应
`WARN think-budget: forced </think> after 513 tokens`（35B 两轮均触发后自动重答）。

**llama.cpp 对照原始数据**（9B，`llama-server b12128`，同一 GGUF/同一 prompt/同参数
temp0.8/top_k40/top_p0.95/min_p0.05，max_tokens 512）：两轮 `predicted_n=512`、
`content` 为空、无 `</think>` —— 与改前引擎一致 ⇒ 模型行为。
推荐参数 temp0.6/top_k20/top_p0.95 下本模型仍可能在思考里超过 512 token，所以
**think budget 兜底是必要项**。

## 已知限制 / 建议

1. 兜底不改变模型能力：思考预算 512 对"需要长思考"的问题是硬截断，可通过
   `--think-budget` 调大（同时调 `--no-limit-tokens`）；复读兜底只在"连续相同句子"时触发。
2. 用户曾用 `--ctx 131072` 启动 35B（f32 KV 占 5 GiB，只剩 3/40 层专家上 GPU），
   decode 仅 **4 tok/s**，表现为"很久才出字"。12GB 卡 + 32GB 内存的 35B 聊天建议
   `--ctx 4096`（15 层专家上 GPU，decode ~25 tok/s）。这是放置配置，不是采样问题。
3. `n你好` 未能在服务端复现；浏览器实测标题就是用户输入原文。

## 验证

（见下方"四套验证 + serve e2e"结果表；本轮改动只涉及 sampler/serve/config + 托管 webui
静态资源的默认值，未改 `kernels.h` / `build.sh` 编译项 / `src/mtp.cpp` / `src/launcher.cpp`。）

### 第 3 轮验证结果（自动汇总自 build/logs/）

| 验证 | 结果 |
|---|---|
| `tools/verify.sh` | **ALL GREEN** |
| `tools/smoke_models.sh` | **PASS**（qwen2.5-0.5b: topk: 40716(-2.406) | decode 5.08 ms/token, Coder-V2-Lite: topk: 185(-0.003) | no decode, Qwen3.5-9B: topk: 2614(-1.756) | no decode, Qwen3.6-35B-A3B: topk: 2614(-2.582) | no decode） |
| `tools/gen_regression.sh` | **GEN REGRESSION GREEN 10/10** |
| `tools/verify_generality.sh` | **GENERALITY GREEN 26/26** |
| `tools/verify_serve.sh (0.5B)` | **SERVE VERIFY GREEN 22/22** |
| `tools/verify_serve.sh (9B)` | **SERVE VERIFY GREEN 21/21** |
| `tools/webui_e2e.py (9B, real browser)` | **3 轮 PASS**（轮1: 首字 3.89s / 20.5s；轮2: 首字 0.42s / 19.0s；轮3: 首字 1.03s / 19.9s） |
| `tools/webui_e2e.py (35B, real browser)` | **2 轮 PASS**（轮1: 首字 3.44s / 27.2s；轮2: 首字 1.79s / 29.7s） |

