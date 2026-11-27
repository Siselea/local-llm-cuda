# HTTP 服务使用说明（`engine serve`）

自研引擎的 OpenAI 兼容 + llama.cpp 风格 HTTP API。零第三方依赖（自写 WinSock2
HTTP/1.1，`ws2_32.dll` 运行时加载，不需要改 `build.sh` 链接选项），单进程只加载一次模型，
请求串行处理（单用户；并发请求在 listen backlog 里排队，不会同时跑两份模型）。

配置项（`serve_host` / `serve_port` / `serve_ctx` / `default_*`）见
[CONFIG.md §3.7](CONFIG.md)，端点的完整列表见 [CONFIG.md §7](CONFIG.md)。

---

## 1. 启动

```bash
# 35B（12GB 显存，启动约 10–20 s，占约 10.5 GB 显存）
./build/engine.exe serve <MODEL_DIR>/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-Q4_K_M.gguf \
    --host 127.0.0.1 --port 7030 --ctx 4096

# 9B（更快，日常调试用）
./build/engine.exe serve <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf --port 7031 --ctx 4096

# 0.5B（秒级加载，验证 HTTP/分词/采样用）
./build/engine.exe serve <MODEL_DIR>/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 7032 --ctx 1024
```

启动成功后打印：

```
[serve] model      : <path>
[serve] tokenizer  : gpt2/BPE pre=qwen35 vocab=248320 bos=248044 eos=248046 (...)
[serve] chat tmpl  : chatml (thinking-aware)
[serve] loading model (ctx=4096) ...
[serve] model ready: vocab=248320 ctx=4096 load=... s
listening on http://127.0.0.1:7030 (model=<path>, ctx=4096)
```

之后每处理一个请求会在 stderr 打印一行 `[req] ip:port POST /v1/chat/completions -> 200
prompt=N gen=M ... ms (prefill ..., decode ..., tok/s)`。

**退出**：Ctrl+C / Ctrl+Break / 关闭控制台窗口 / `taskkill` 都会走同一条路径停止 accept
循环、释放显存，然后打印 `[serve] stopped; VRAM released`。浏览器访问的页面右上角没有退出按钮；
需要脚本化停机时 `POST /shutdown`（仅当绑定地址是回环时开放）。

浏览器直接打开 `http://127.0.0.1:7030/` 即可聊天。页面优先使用 **llama.cpp 官方单文件
webui**（MIT，放在 `build/webui/index.html`，`engine serve` 从 exe 同目录静态提供；与官方
`llama.cpp - chat` 界面一致：多轮对话 / 流式 / 停止 / 深色浅色 / 移动端 / 会话存 IndexedDB）。
`build/webui/index.html` 不存在时回退到编进 exe 的极简备用页（无 CDN、离线可用）。

双击启动有两个入口：

* `build/launcher.exe`（推荐）——Win32 深色 GUI（无控制台窗口），窗口可缩放、随 DPI 缩放：
  * 分组：**模型 / 显存与上下文 / 性能开关 / 服务 / 高级（可折叠）**，每个控件都有中文标签
    + 单位/含义，悬停显示引擎原始帮助（tooltip 直接复用配置表里的 help）；
  * 两个滑块：**GPU 专家层数**（0..N，含「自动」）与**上下文长度**（自动/4K..128K），拖动时
    实时显示 `显存 ≈ x GB / 可用 y GB`、`CPU n 层 ≈ z GB 主机内存`、`KV 显存 ≈ …`，超出可用
    显存标红；字节口径与 `src/planner.cpp` 一致（`engine info --json` 提供 per-layer 字节）；
  * **SSD 加速** 开关默认关：模型能装进「显存+内存」时按放置判据自动禁用置灰并写明原因
    （gpu / vram+ram split），只有装不下才可勾选，勾选后提示「引擎报告：ssd 流式未实现」；
  * **高级** 面板由 `engine.exe serve --print-config --json` 动态生成：引擎配置表的 107 个
    选项每项一个控件（bool→开关、枚举→下拉、数值→输入框），带搜索框；改动写入
    `build/launcher-engine.cfg` 并以 `--config` 传给引擎，所以引擎新增选项时 GUI 自动跟随；
  * 启动命令、生效参数摘要、引擎日志路径与引擎实测放置决策（`[placement] …`）都显示在底部
    状态区；日志在 `build/launcher-engine.log`（首行是 launcher 实际执行的完整命令行）；
  * 关闭窗口时若引擎在跑会询问；验收脚本/截图见 `build/test_launcher.py`;

* `启动.bat` —— 无 GUI 环境的控制台兜底：扫描模型 → 选编号 → 输入 ctx → 启动并自动开浏览器。

模型扫描逻辑在引擎里（`engine models` 子命令，供 GUI / bat / CLI 共用）：

```bash
./build/engine.exe models            # 编号/大小/类型/目录，按大小降序
./build/engine.exe models --tsv     # launcher.exe 用的制表符格式
./build/engine.exe models --json    # 脚本/测试
./build/engine.exe serve --pick 3   # 用列表里的第 3 个模型启动
./build/engine.exe serve --model auto  # 选最大的模型
```

---

## 2. curl 示例

```bash
BASE=http://127.0.0.1:7030

# --- 健康检查 / 元信息 ---
curl -s $BASE/health
# {"status": "ok", "model": "...", "n_ctx": 4096, "slots_idle": 1, "slots_processing": 0}
curl -s $BASE/v1/models
curl -s $BASE/props            # 模型路径/arch/量化混合/ctx/tokenizer/chat template/端口

# --- 非流式对话（OpenAI 兼容） ---
curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "你是一个简洁的中文助手。"},
    {"role": "user", "content": "用一句话介绍你自己"}
  ],
  "temperature": 0.8, "top_p": 0.95, "top_k": 40, "max_tokens": 256, "stream": false
}'
# 返回: choices[0].message.content（<think>思考段</think> + 最终回答；llama.cpp webui 约定）
#       message.reasoning_content（思考段，OpenAI 客户端用）
#       finish_reason / usage / timings{prefill_ms,decode_ms,tokens_per_second,cache_hit_tokens}
#       generated_token_ids（便于与 build/test_gen.exe 逐 token 对拍）

# --- SSE 流式对话 ---
curl -N $BASE/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "讲个笑话"}],
  "temperature": 0.8, "max_tokens": 256, "stream": true
}'
# data: {"object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""}}]}
# data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"<think>","reasoning_content":"..."}}]}
# data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"...</think>回答..."}}]}
# data: {"object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
# data: [DONE]

# --- 多轮对话（客户端自己维护 messages 历史） ---
curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "user", "content": "法国的首都是哪里？"},
    {"role": "assistant", "content": "巴黎。"},
    {"role": "user", "content": "那它的人口大概多少？"}
  ],
  "temperature": 0, "max_tokens": 64
}'

# --- OpenAI legacy completions（原始 prompt，不做 chat 模板） ---
curl -s $BASE/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "temperature": 0, "max_tokens": 16}'

# --- llama.cpp 风格 /completion ---
curl -s $BASE/completion -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "n_predict": 16, "temperature": 0}'
# {"content": " Paris", "stop": true, "tokens_predicted": 1, "tokens_evaluated": 5,
#  "generation_settings": {...}, "timings": {...}, "generated_token_ids": [...]}
curl -N $BASE/completion -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "n_predict": 8, "stream": true}'

# --- 分词器（调试/对拍用） ---
curl -s $BASE/tokenize   -H "Content-Type: application/json" -d '{"content": "你好，世界"}'
# {"tokens": [108386, 3837, ...], "pieces": ["你", "好", ...]}
curl -s $BASE/detokenize -H "Content-Type: application/json" -d '{"tokens": [108386, 3837]}'

# --- 脚本化停机 ---
curl -s -X POST $BASE/shutdown
```

### 请求参数（OpenAI 端）

| 字段 | 说明 |
|---|---|
| `messages` | `[{role, content, reasoning_content?}]`，`content` 可以是字符串或 `[{type:"text",text}]` 数组（图片/音频 part 被忽略，本服务只做文本） |
| `temperature` | `0` = greedy；默认取 `default_temperature`（0.8） |
| `top_p` / `top_k` / `min_p` | 采样过滤；默认 0.95 / 40 / 0 |
| `max_tokens` / `max_completion_tokens` / `n_predict` | 生成长度上限；`-1`/`0`（llama.cpp webui 默认）= "不额外限制"，实际映射到 `serve_no_limit_tokens`（默认 1024），避免思考模型单条回复跑几分钟；缺省取 `default_max_tokens` 后再受 `ctx - prompt` 限制 |
| `cache_prompt` | `false` 对本请求关闭多轮前缀缓存（默认 `true`，见 §5.2） |
| `stop` | 字符串或字符串数组；跨 token 匹配（内部有 holdback 缓冲，不会漏判） |
| `seed` | 固定后同参数可复现；缺省随机 |
| `stream` | `true` 走 SSE |
| `enable_thinking` / `chat_template_kwargs.enable_thinking` | Qwen3.5+：`true` 打开 `<think>` 生成，`false` 生成空 think 前缀（不思考） |

### Python 客户端（OpenAI SDK）

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:7030/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "用一句话介绍你自己"}],
    temperature=0.8, top_p=0.95, max_tokens=256)
print(resp.choices[0].message.content)
```

`top_k` / `min_p` / `enable_thinking` 是扩展字段：普通 SDK 会忽略；需要时用
`extra_body={"top_k": 40, "enable_thinking": False}`。

---

## 3. 正确性/可复现性

- **同一 seed + `temperature=0` 与 `build/test_gen.exe` 的贪心序列逐 token 一致**：
  `POST /completion {"prompt": ..., "temperature": 0, "n_predict": N}` 返回的
  `generated_token_ids` 可直接与 `test_gen --ids <prompt token ids> --n N` 的 `gen:` 行对比
  （`tools/verify_serve.sh` 自动做这件事）。
- **分词器与 llama.cpp 的 `llama-tokenize` 逐 token 一致**（qwen35/qwen2 pre-tokenizer 是
  llama.cpp 精确端口）：`tools/verify_tokenizer.sh`。
- chat 模板渲染出的提示词 token 序列与已知 Qwen ChatML 格式一致：
  `<|im_start|>role\ncontent<|im_end|>\n`，生成提示为 `<|im_start|>assistant\n<think>\n`
  （`enable_thinking=true` 时）。

---

## 4. 自测

| 命令 | 说明 |
|---|---|
| `bash tools/check_output_quality.py --json <resp>` | **输出质量闸门**：空输出/符号汤/复读/复述 prompt 判 FAIL（服务测试里已经内嵌，避免"HTTP 200 就当 PASS"的盲区） |
| `bash tools/verify_http.sh` | HTTP 层纯逻辑单测（不加载模型/不用 GPU：keep-alive、128 KiB POST、chunked 流、CORS、404/400、HTTP/1.0 close、pipelined、/shutdown），12 项 |
| `bash tools/verify_serve.sh` | 轻量端到端（默认 qwen2.5-0.5b@7031）：启动行、`/health`、`/v1/models`、`/props`、浏览器页、`/tokenize`+`/detokenize`、chat 非流式+SSE、`/v1/completions`、llama.cpp `/completion`（含流式）、404/400/prompt 超长 400、与 `test_gen` greedy 逐 token 对拍、`/shutdown` 退出+端口释放 |
| `bash tools/final_acceptance.sh [--with-longctx]` | **最终验收**：0.5B→9B→35B 三档 e2e（+可选 4k 长上下文），一键汇总 |
| `bash tools/serve_longctx_check.sh [model] [port] [ctx] [tokens]` | 长上下文请求（默认 4000 token 中文 prompt）prefill 延迟/首答/优雅退出验收 |
| `bash tools/serve_35b_e2e.sh` | 35B@7030 完整往返：中文单轮、多轮 messages、stream=true SSE、300/500 token 中等 prompt、中文 tokenize 往返、greedy 与 `test_gen` 对拍（两阶段，避免同时占两份显存）、优雅退出+显存释放 |
| `python tools/serve_webui_probe.py --url ...` | **llama.cpp webui 行为探针**：按 webui 的请求体（`stream:true, cache_prompt:true, max_tokens:-1, samplers=...`）发多轮中文请求，测首字 delta 延迟、检查 `content` 里的 `<think>`/`</think>` 与最终回答、打印 `timings` 和日志里的 `[new,cached]` |
| `python tools/webui_e2e.py --url ... --turns 3` | **真实浏览器验收**：headless Edge + CDP，打开 webui、在输入框真打字（`Input.insertText`）、点发送按钮（`Input.dispatchMouseEvent`）、流式渲染轮询、每轮截图、读会话标题与 DOM 文本（首字延迟/思考/回答/复读） |
| `python tools/serve_drift_probe.py --url ... --turns 3` | 思考漂移探针：按 webui 参数跑多轮对话，统计每轮思考段里"同一句话重复次数"（`DRIFT DETECTED` / `NO DRIFT`），支持 `--greedy`、`--temp/--top-k/--top-p/--min-p` |
| `python tools/serve_prefix_cache_check.py --url ... --turns 8` | **多轮前缀缓存 A/B**：同一对话逐轮发 `cache_prompt=false`（整段 prefill 基线）与 `true`（缓存），逐 token 对比贪心输出 + 对比 prefill 时间，硬验收输出 `PREFIX CACHE GREEN` |
| `bash tools/verify_tokenizer.sh` | 分词器与 llama.cpp `llama-tokenize` 的逐 token 对拍（不需要 GPU） |
| `bash tools/repro_qwen35_prefill.sh` | qwen35 多 token prefill bug 的最小复现材料（给引擎侧定位用） |

## 5. llama.cpp webui 兼容 + 多轮前缀缓存

### 5.1 webui 只渲染 `content`：思考段同时写进 `content`
llama.cpp b6000 的 webui 只读取 `choices[0].delta.content`（它自己在客户端按 `<think>` /
`</think>` 标签把思考段折叠显示），**完全忽略 `reasoning_content`**。服务端现在同时满足两类客户端：

- `content`：思考段包在 `<think>` ... `</think>` 之间，后面接正式回答 —— webui 从第一个思考
  token 起就有东西显示（可折叠），不再一直转圈；
- `reasoning_content`：同一份思考文本仍单独发（OpenAI SDK / 自研客户端用）。

非流式响应同样规则（`message.content` 带标签，`message.reasoning_content` 保留）。如果生成在
思考中被长度上限截断，服务端会补发一个 `</think>`，webui 不会一直停在"思考中"。

### 5.2 多轮前缀缓存（`engine serve` 默认开启）
每个请求会把渲染后的对话历史与上一轮的缓存前缀比较，命中就复用上一轮的 KV cache（混合 qwen35
架构还会把 SSM 的 conv/rec 状态快照一并恢复），**只 prefill 新增 token**（上一轮回答 + 新提问 +
生成提示），不再整段重算历史。要点：

- **逐 token 一致（硬验收）**：缓存路径与 `cache_prompt=false` 的整段 prefill 必须产出
  **逐 token 相同**的结果；`tools/serve_prefix_cache_check.py` 实测（9B ctx32768、5 轮、历史
  长到 8.3k token、greedy）所有轮次 SAME。
- **缓存对齐粒度与前向 chunk 解耦**（`serve_cache_align` / `--cache-align`，默认 **128**）：
  快照点落在 128 token 网格上，而前向 chunk 仍用 `--prefill-rows`（serve 默认 512、按显存
  生长到 1024）。这样长 prompt 首次 prefill 走大 chunk（9B 9k prompt 约 9.7 s），而多轮
  命中时一轮最多只多 prefill <128 token 的历史（对齐余量）。若对齐到 chunk 本身（旧行为，
  `--cache-align 0`），9.1k 对话每轮会多 prefill 最多 ~1023 token（实测 1.2 s/轮）；
  解耦后同一场景第二轮 prefill 832 ms → **210 ms**（−75%），整轮 wall 1.21 s → 0.59 s。
  解耦后缓存路径的历史前向可能出现 partial chunk，这是逐 token 一致性验证覆盖的路径。
- `--cache-align 0` = 旧行为（对齐到当前 prefill chunk）；`/props.prefix_cache.align` 可观测。
- `cache_prompt:false` 可对单个请求关闭（llama.cpp webui 默认发 `true`）；`--prefix-cache=0`
  全局关闭（A/B 对比用）。
- 状态快照 62.8 MiB（35B，31 个线性层 rec+conv），单槽，直接复用 mtp 投机解码的 rollback
  槽位；显存不足分配失败时自动退回"整段 prefill"并在日志说明。
- 观测：`/props` → `prefix_cache`{enabled,state_bytes,have_snapshot,cached_tokens,hits,misses,
  reused_tokens,prefilled_tokens,last_hit_tokens,alloc_failed}；日志
  `[req] ... prefill X ms [new N, cached M] ...`；响应 `timings.cache_hit_tokens`。

实测（35B，ctx 4096，本机 RTX 3060 12GB，自适应放置 15 层 GPU 专家）：

| 场景 | 整段 prefill | 前缀缓存 |
|---|---|---|
| 同一 1331 token prompt 再问一次 | 25.9 s | **1.07 s**（复用 1280 token） |
| 5 轮对话（每轮 +~50 token），第 5 轮 | 3.93 s / 290 token | **1.4 s / 72 token** |
| 8 轮对话（每轮 +~60 token），第 8 轮 | 9.44 s / 478 token | **1.77 s / 94 token（5.3×）** |

### 5.3 思考预算 / 复读兜底 / 推荐采样（2026-11-16 新增）

**根因判定（实测）**：Qwen3.x 思考模型在 webui 默认采样（temp 0.8 / top_k 40 /
top_p 0.95 / min_p 0.05）下经常陷入"自述循环"——几百 token 都在写
"Thinking Process: 1. Analyze… 2. Draft… 3. Refine…"，**始终不输出 `</think>`**，
把整个 max_tokens 预算耗在思考里（用户看到的就是"思考阶段一直漂/复读、回答不来"）。
用**同一 GGUF、同一 prompt、同一参数**跑 llama.cpp b12128 对照：它同样把 512 token
预算耗尽在 thinking、不闭合、无回答 ⇒ **这是模型/参数行为，不是引擎 bug**
（分词/模板/贪心解码另有逐 token 对照，见下）。35B 在强制闭合后还会把"规划文本"
当回答继续输出并复读（改前的 7 次重复就是这么来的）。

因此本轮加了三个兜底（都不改变模型本身能力，只是让体验可控）：

1. **思考预算 `serve_think_budget`（默认 512）**：思考段超过 512 token 仍未闭合时，
   引擎强制发 `</think>` 结束思考，然后**用"关闭思考"的模板**
   （`<|im_start|>assistant
<think>

</think>

`）重新渲染同一段对话、只生成回答，
   追加到同一个流式响应里（前缀缓存命中使第二次 prefill 很便宜）。
   这样不会像"硬插 `</think>` 继续生成"那样把模型的规划循环当回答输出。
   `timings.think_budget_hit` / `/props.think_budget` / 日志
   `[serve] WARN think-budget: forced </think> after N tokens` 可观测；`0` 关闭。
2. **复读兜底 `serve_repeat_guard`（默认开）**：连续 3 句完全相同（且每句 ≥8 字符）时
   提前 `finish_reason=stop`（`timings.repeat_guard_hit` / 日志 WARN），
   避免把"同一句话复读 10 遍"刷屏到 token 上限。
3. **推荐采样默认值**：请求未显式给采样参数时用 **temp 0.6 / top_k 20 / top_p 0.95**
   （Qwen3 官方推荐），`/props.default_generation_settings` 直接体现；服务端托管的
   llama.cpp webui 前端默认值也改成这组，并对浏览器里**旧 localStorage 配置**
   （0.8/40/40/0.95/0.05）做一次性迁移，老用户刷新页面即生效。

实测（真实浏览器，`tools/webui_e2e.py`：真打字 + 点发送按钮 + 流式渲染截图）：

| 模型 | 轮次 | 首字 | 总耗时 | 思考 | 回答 |
|---|---|---|---|---|---|
| 9B | 1/2/3 | 3.89 / 0.42 / 1.03 s | 20.5 / 19.0 / 19.9 s | 规划段落，无 5 次重复 | 3 段中文正常 |
| 35B | 1/2 | 3.44 / 1.79 s | 27.2 / 29.7 s | 规划段落，无 5 次重复 | 3 句中文正常 |

对照：改前 35B 同一对话第 2 轮回答里同一句重复 **7 次**、耗时 45.4 s；现在干净、29.7 s。
截图见 `build/logs/e2e_9b_final/`、`build/logs/e2e_35b_final/`。

### 5.4 "不限制长度"回复的默认上限
llama.cpp webui 默认发 `max_tokens:-1`。以前按 ctx 上限（4096）生成，思考模型可能一条回复跑几分钟；
现在 `max_tokens<=0` 映射为 `serve_no_limit_tokens`（默认 1024，`--no-limit-tokens` 可改），
`/props` 暴露 `no_limit_tokens`。

### 5.5 采样性能：候选集采样（`perf(serve)`，2026-11-21）

**问题**：serve 对**每个非 greedy token** 走 `forward()` 取整行 248320 维 logits
（1 MB D2H）+ `last.assign` 再复制 1 MB + `Sampler` 整词表 `fill` + `partial_sort(248320,20)`
≈ 0.2 ms/token；如果客户端发 `top_k=0`（只 top_p，llama.cpp 系 UI 可配），
整词表 `exp`（1.33 ms）+ `sort 248320 pairs`（0.45 ms）≈ **1.9 ms/token**（9B 的 ~9%）。

**修复**：

* serve 不再复制 logits 行（move 进 sampler）；greedy 路径逐 token 不变（仍是设备 top-k + argmax）。
* 新增 `Sampler::sample_bounded()`：先用 host **单遍 top-N 扫描**（sorted insertion window，AVX2 8 路跳过）选候选集，再在候选集上跑
  **同一套** sampler 链（repetition penalty → top_k → top_p → min_p → temp → dist）：
  * `top_k > 0`：候选 = 前 `max(top_k,64)`，top-k 集合精确；
  * `top_k = 0`：候选 = 前 1024（`kDefaultCandidateCap`），即 **top-1024 近似 top_p**；
  * penalty 只作用候选 id（不再整行复制）。
* 预留设备候选接口：`Sampler::set_candidates(ids, logits, n)`，等 runner 侧设备 top-k
  扩到 64/128 后直接接入（serve 不需要再改）。
* `/props.sampling` 暴露实现（`implementation=bounded-candidates`、`candidate_cap`、
  `top_k_zero=approx-top-1024`）——客户端/评测可据此判断是否近似。
* 已知差异：极端长尾分布下排名 >1024 的 token 概率被截断；`top_k>0` 时若被 penalty 压出
  前 64 的 token 才会引入顺序差异（概率可忽略）。单测 `tests/test_sampler.cpp` 覆盖
  greedy 一致、top_k 精确、top_k=0 截断（对比整词表实现）、penalty、外部候选、可复现性。

**device top-k 候选（2026-11-21 新增，v0.1.4 仅接线）**：解码头可在设备端直接产出 top-k
候选（`head_topk_max`，**stock 默认 8**；实测 k=64 的 device top-k 内核本身要 2.4–5.8 ms/token，
大于省掉的 1 MB D2H ⇒ **默认保持 8**，k>8 自动回退整行，见报告 §11），非 greedy 请求在
`top_k>0 && max(top_k,64) <= head_topk_max` 时只把候选 `(id, logit)` 拷回 host
（省掉整行 248320×4 B ≈1 MB D2H + host 扫描），其余请求（`top_k=0`、`top_k>64`、
设备头不可用）**自动回退整行路径**。`/props.sampling.device_candidates` 暴露
`{enabled,k,scope}`。正确性判据：候选 id/顺序与 host 参考完全一致、值差 ≤ 解码路径本身的
调用间数值噪声（实测 host-vs-host max|diff| ~7.6e-4，split-k/atomics 求和顺序）；
独立验证 `build/validate_topk_candidates.cpp`（先 prefill、save/restore SSM、再 device vs
full-row 对比，不依赖 serve）。

**webui 实际参数（证据）**：bundle 里请求体默认
`temperature:0.6, top_k:20, top_p:.95, min_p:0`（`getConfig()` 还会把 llama.cpp 旧默认
`0.8/40/0.95/0.05` 一次性迁移到 `0.6/20/0.95/0`）——所以 webui 主路径是 `top_k=20`
（省 ~0.2 ms/token），`top_k=0` 是显式关闭 top-k 的客户端才会走的边缘路径（省 ~1.7 ms/token）。
实测数据见 `docs/reports/SERVE-SAMPLING-PERF.md`。

---

### 5.6 启动自检与 prefill 诊断（2026-11-21 新增）

**加载后 VRAM 守卫（默认开，v0.1.10 起为“警告带 + 严格模式”两段式）**：
模型加载完成后实测剩余显存：

| 实测 free | 行为 |
|---|---|
| `>= vram_margin_mb`（默认 1200 MiB） | 正常启动 |
| `[700 MiB, margin)` | **WARN 后继续启动**：日志 `[serve] WARN: only N MiB free after load (< margin M, >= hard floor 700); starting anyway...`，并提示 `lower --ctx / --gpu-layers`、关闭其它 GPU 程序、或用 `--strict-vram` 拒绝 |
| `< 700 MiB` | `ERROR` 拒绝启动（可重试的测量夹紧 / 明确报错） |

**为什么改成两段式**：实测桌面 WDDM 会在 loader 的 `[vram] check OK` 与 serve 守卫之间
异步吃掉 **1.5–3 GiB**（同一 35B 命令 gap 从 1604 波动到 3230 MiB），固定“free<margin 就拒绝”
会让默认 launcher 启动随环境抖动而间歇失败（v0.1.10 实测 35B auto ctx8192 就因此失败过）。
为补偿这段，serve 模式对 **MoE 部分卸载模型**额外预留 `ENGINE_SERVE_RESERVE_MB`（默认 2048 MiB），
让 placement 少放 1–2 层而不是装载后失败。

- **严格模式**：`--strict-vram` / `ENGINE_STRICT_VRAM=1` 恢复 v0.1.6 行为（`free < margin` 就拒绝，
  不进入警告带）；适合追求稳定 tok/s、明确不想承担 WDDM 换页风险的场景。
- `--allow-oversubscribe` 仍然是“决策器判定装不下也尝试”的总开关（预期单数字 tok/s）。
日志：
```
[serve] WARN: only 1093 MiB free after load (< margin 1200 MiB, >= hard floor 700 MiB); starting anyway...
[serve] ERROR: only N MiB VRAM free after load, below the M MiB margin: ...
```

**装载后自预热（默认开）**：模型加载完成后先跑一次 128-token 哑 prefill + 8-token tail +
1-token decode（捕获 canonical CUDA graph/首次触碰），再用 `prefix_state_save/restore`
复位 SSM 状态，保证用户请求 #1 从干净缓存开始。`ENGINE_NO_PREWARM=1` 关闭。
实测代价：启动约 +0.44 s；收益：冷启动首请求约 −0.5 s（**不是**冷启动的根治——
首请求大头仍是首次触碰权重的 page-in）。

**prefill 四段计时**：`ENGINE_SERVE_PREFILL_TRACE=1` 时每个请求在日志打一行：

```
[serve] prefill-trace: total=... ms = restore=... + save=... + chunk=...(N tok) + tail=...(M tok) [cached=X prefilled=Y]
```

用于区分"前缀缓存恢复/快照"与"实际 prefill 计算"的成本（实测 cache-hit 的
restore=0.6 ms、save=0.6 ms；长上下文慢点定位见 `docs/reports/SERVE-PREFILL-CACHE.md`）。

> 更正记录：2026-11-21 早前把 9.1k prompt 的 22–43 s 归因为"按 token 数懒捕获 CUDA graph"
> 是**错误**的（`prefill_graph` 配置默认 false）；独占 GPU 复测该请求仅 **0.65 s**，
> 旧数据被判定为跨进程显存超订/WDDM 换页污染，已撤回。

## 6. 已知限制（如实）

1. **首字延迟仍受 prefill 速度限制（已大幅改善）**：35B 的 pp 仍是 48–55 tok/s 量级
   （MoE/CPU 专家瓶颈），但多轮前缀缓存把"每轮都要重算全部历史"降成"只算新增 token"，
   webui 实测首字（思考段第一个 delta）0.8–1.9 s：第一轮 15 token 约 1.0 s，之后每轮只
   prefill 上一轮回答 + 新提问（几十 token）。第一轮/整段 prefill 基线（2026-11-15 11:52
   构建）：36 token → 1.18 s（首次请求含懒初始化 3.1 s）；348 token → 9.16 s；573 token →
   12.0 s；4023 token（4k）→ 105.5 s；decode 18.9–22.1 tok/s；加载 52–54 s。**要让单轮
   首字稳定 <0.5 s 需要继续提 pp（9B 稠密线的 prefill-fast 已经 3.3×，35B 还没吃到）。**
2. **单用户串行**：一次只处理一个请求；长生成时第二个请求会在 backlog 排队（没有多槽并发，
   也不做 continuous batching）。
3. **前缀缓存是"单槽 + 128 token 对齐"的**：只保留上一轮的对话历史快照，且按 prefill chunk
   边界对齐（serve 默认 `prefill_rows=128`），所以每轮会多 prefill 最多 ~128 token 对齐余量；
   编辑历史/重新生成（分支）会退化为整段 prefill。多槽/主机内存快照是后续优化方向。
   关闭：`--prefix-cache=0`。
4. **chat 模板是子集**：支持 Qwen ChatML 家族的 system/user/assistant/tool 文本消息、
   `<think>` 生成提示、`enable_thinking`、assistant 历史里的 `reasoning_content`；
   **不支持 tools 参数（工具调用）、vision/图片、以及完整 jinja 语义**。模板不是 ChatML 的模型
   （如 DeepSeek-Coder-V2 的模板）会在 `/v1/chat/completions` 返回 400，请改用
   `/v1/completions` / `/completion` 传原始 prompt。
5. **分词器 pre-tokenizer**：`qwen35` / `qwen2` 与 llama.cpp 完全一致；其他 pre 名
   （如 `deepseek-llm`）回退 GPT-2 规则，token 序列可能与 llama.cpp 不同（解码仍正确）。
6. **无鉴权**：默认只绑 `127.0.0.1`。绑 `0.0.0.0` 时任何能访问该端口的人都能用，
   且 `/shutdown` 仍只对回环开放。不要在不可信网络上暴露。
7. **`--ctx` 是硬上限**：prompt 超限返回 400，不会自动截断。
8. **SSE 格式**：每个 `data:` 事件是**单行压缩 JSON**（已修掉早期多行 pretty JSON 的兼容问题），
   `data: [DONE]` 结束。`Content-Type: text/event-stream`，chunked 传输。
9. **停止**：Ctrl+C / 关窗口 / `taskkill` 都会释放显存；但 `taskkill /F` 属于强杀，
   CUDA 上下文由 OS 回收，不会留后台进程。
10. **流式里的思考段**：llama.cpp webui 只读 `delta.content`，所以思考段**同时**以
   `<think>思考</think>` 形式写进 `content`（webui 折叠显示），同一份文本仍放在
   `delta.reasoning_content` 给 OpenAI 客户端；正式回答在 `</think>` 之后。只用 `content`
   的客户端会看到思考标签（webui 语义），只用 `reasoning_content` 的客户端不受影响。
11. **生成质量（引擎层，2026-11-15 已修复）**：qwen35 混合架构模型（Qwen3.5-9B / Qwen3.6-35B-A3B）
    曾因 GGUF 张量语义错位（`ssm_a` 存的是 -A 不是 log A、q/k→v head 广播用 `h % n_kh` 而不是
    `h // rep`）导致 **prompt >1 token 后 logits 偏离 llama.cpp**（复读输入/数字符号汤）。
    该 bug 由 `qwen35-prefill-bug` 修复，最小复现见 `tools/repro_qwen35_prefill.sh`；
    修复后 9B raw `The capital of France is` 与 llama.cpp 逐字节一致（`" Paris.\nThe capital
    of France is Paris."`），35B 中文对话正常。**这不是 HTTP/分词/采样层的 bug**（服务端 greedy
    输出与 `build/test_gen.exe` 逐 token 一致）。历史版本若出现符号汤，升级引擎即可。
12. **思考模型的固有自述循环（模型行为，已兜底）**：Qwen3.x 思考型 GGUF 在
    temp ≥0.8 时容易在 thinking 里自述循环且不闭合；llama.cpp 同参数同样如此。
    本轮默认采样降为 Qwen 推荐值（0.6/20/0.95）+ `serve_think_budget` 512 +
    `serve_repeat_guard` 兜底。若想要更长的思考，调大 `--think-budget`
    （或 `=0` 关闭），并相应调大 `--no-limit-tokens`。
13. **35B 的 KV 配置会影响解码速度**：实测用户曾用 `--ctx 131072` 启动 35B
    （f32 KV 占 5 GiB，只有 3/40 层专家能放 GPU），decode 只有 **4 tok/s**，
    表现为"很久才出字"。32GB 内存 + 12GB 卡上 35B 聊天建议 `--ctx 4096`
    （15 层专家上 GPU，decode ~25 tok/s）。这是放置/显存配置问题，与采样无关。
14. **KV 精度**：默认 f32；显存紧张时可用 `--kv-type q8_0`（9B 上 q8_0 与 f32 的 40 步贪心序列
    逐 token 一致；长上下文 4k+ 建议保持 f32，除非显存不足）。


### 5.7 思考块与 stop 序列（2026-11-22）

- **`</think>` 不会被 stop holdback 切断**：thinking 打开时先在整个待发缓冲上判定
  `</think>` 再落盘（并额外 hold 整 marker），旧实现会因 stop 保留字节把 marker 切半，
  导致答案全部留在 reasoning、content 为空。
- **stop 只匹配可见 content**：思考块内部的文本不参与 stop；`</think>` 之后的模板分隔空行
  （可能跨 token 到达）会被吞掉，`stop:["\n\n"]` 不会一上来就命中分隔符。
- 与 think-budget 的关系：预算判定在 decode 循环、stop 在 flush；两者互不改写顺序。
  思考超过预算仍按原策略强制闭合 + 关闭 thinking 重试。

### 5.8 任意 ctx 的可见性（serve 侧，与 feature-polish 的 CLI 配合）

- `/props` 同时返回 `n_ctx`（实际值，llama.cpp 兼容）、`n_ctx_requested` / `requested_ctx`
  （用户请求值，0=auto）、`actual_ctx`、`ctx_clamped`、`ctx_train_exceeded`。
- 启动日志打印 `[serve] ctx : requested N -> actual M (CLAMPED: ...)`，并保留 runner 的
  `[ctx] requested=... / [ctx] budget ... / [ctx] CLAMPED ...` 细账。
- 超 ctx prompt 的 400 消息带齐：`prompt is too long: N tokens >= actual context M
  (requested R, clamped to M), over by X tokens; shorten the prompt, raise --ctx, or use a
  smaller kv_type`。

### 5.10 响应编码保证：非流式 JSON 一定是合法 UTF-8（2026-11-22 修复）

模型在 `max_tokens` 处被截断时可能把**多字节字符切半**（尤其 emoji、CJK 扩展 B 这类 byte-fallback
token）。旧实现把不完整的 UTF-8 尾字节直接写进 `content`，`json::Writer` 又原样透传，导致
`/completion`、`/v1/chat/completions` 的**非流式响应体不是合法 UTF-8**——严格客户端
（`json.loads(raw_bytes)`、浏览器/JS SDK）会直接抛
`'utf-8' codec can't decode bytes ... invalid continuation byte`；SSE 路径碰巧被浏览器容错掩盖。

修复（`perf` 之外的协议正确性，v0.1.6 hotfix）：
1. `generate()` 最终 force-flush 前丢弃不完整的 UTF-8 尾字节（被 token 预算截断的半个字符本就不存在）；
2. `json::Writer::quote` 校验 UTF-8：合法序列原样透传，非法序列（孤立 continuation / 截断 /
   overlong / surrogate）替换为 **U+FFFD**，保证任何响应都能被严格解析。
回归：`tests/test_json_utf8.cpp`（7 类非法/合法序列）+ `tools/serve_edge_cases.py` 的 K1
（非流式 max_tokens=1/2/3 严格 UTF-8 解码 + JSON 解析）；旧二进制 30 请求 10 个非法 → 修复后 0 个
（`build/logs/utf8_repro.json`）。

### 5.9 MTP 的限制（`--mtp`）

`--mtp` 的 verify 是 (k+1) 行 batch 前向，与单 token greedy decode 的数值路径不同：
Qwen3.6-35B-A3B 实测前 147 token 一致、第 148 个分叉（`docs/MTP-NUMERICS.md`、
`docs/reports/MTP-IMPL-REPORT.md`）。**需要逐位一致的用户不要用 `--mtp`**；服务启动时
hybrid 模型会打印明确 WARN。另：MTP 回滚快照用 SSM slot 0..k-1、prefix cache 用 slot 0，
两者同时启用时 serve 会 WARN 并禁用投机（独立 slot 留 v0.2）。

### 5.11 输入预算与 DoS 防御（2026-11-27，v0.1.8）

默认 `--host 127.0.0.1` 只本机可达；**绑定 `--host 0.0.0.0` 会让局域网任意设备使用
API（无鉴权）**，只应在可信网络使用，不要暴露公网。所有用户输入都有有界工作量：

| 输入 | 预算 | 超限 |
|---|---|---|
| `/tokenize` content | **200k 字符**（全局，不占 KV，因此不受 `--ctx` 限制） | 400 `prompt too long to tokenize: N characters (limit M)` |
| chat/completions、/completion prompt | `min(200k, ctx*8)` 字符（+ 渲染后 prompt 的 ctx 校验） | 400 |
| 单次 encode | 1500 ms 墙钟兜底 | 400 `tokenization failed: ...` |
| `stop` 数组 | 总长 4096 字节 | 400 |
| `/detokenize` ids | 200k 个 | 400 |
| HTTP body / headers | 32 MiB / 64 KiB（严格 Content-Length；重复 CL→400；chunked 请求体→501） | 413/431/400/501 |
| 数值参数 | finite + 区间（temperature [0,100]、top_p/min_p [0,1]、top_k [-1,1M]、max_tokens [-1,1M]、seed [-1,INT_MAX]、repetition_penalty [0,100]） | 400 |
| 文本编码 | 必须合法 UTF-8（拒绝 overlong/代理项/>U+10FFFF） | 400 `... is not valid UTF-8` |

tokenizer 的 BPE merge 已从"每轮全量重扫"改为**符号链表 + (rank,left) 最小堆**
（O(n log n)，输出 id 不变）：20k CJK 字符编码从 92 s 降到 ~23 ms，1 MB 无空格文本在
编码前即被字符预算拒绝（~20 ms）。回归测试：`tests/test_tokenizer_budget.cpp`、
`tests/test_text_util.cpp`，HTTP 用例见 `tools/serve_edge_cases.py`（L1–L5）。
