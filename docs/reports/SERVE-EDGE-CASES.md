# serve 边缘场景审计（真实用户路径）

日期：2026-11-21 22:xx　审计者：server-mode（lead 指派）
对象：`engine serve`（HEAD `4c921be` + 本轮 WIP 无；二进制由当前源码重编）
方法：①静态代码审计（`src/serve.cpp`、`src/http_server.cpp`、`src/tokenizer.cpp`、`src/mtp.cpp`/`qwen35.cpp`）。
②离线 tokenizer 边界（`build/check_tok_edges.py`）③真实 HTTP 边缘用例（`build/serve_edge_cases.py`：断连/中断/编辑历史/stop/并发/slowloris/keep-alive/completions 一致性）。
每条：场景 → 期望 → 实测 → 判定；确认 bug 给最小复现 + 修复建议。

## 结论速览

| # | 场景 | 判定 | 严重度 | 状态 |
|---|---|---|---|---|
| 1 | thinking 开启 + 任意长度≥2 的 stop 序列 | **bug**：`</think>` 被 stop holdback 切成两半后检测不到，答案全部留在 reasoning，content 为空 | 高（API 用户常带 stop） | 静态已证（离线仿真复现），待 GPU 实测 |
| 2 | thinking 开启 + stop 匹配到思考文本（如 `\n\n`） | **bug/语义问题**：思考块里出现空行就停止生成，用户只拿到空的 content | 高 | 静态 + 待实测 |
| 3 | 客户端发坏请求（JSON 语法错、messages 非数组/为空、prompt 数组、prompt 超 ctx、/tokenize 缺 content） | **bug**：全部返回 **500**（应为 400）——`parse_json_body`/`parse_messages`/`validate_prompt` 在 try 块之外抛 `UserError` | 中（SDK 重试逻辑/可观测性） | 静态已证，待 curl 实测 |
| 4 | 开启 `--mtp` 后交替使用前缀缓存 | **bug**：MTP 投机验证把 per-token 回滚快照写进 SSM slot **0..k-1**（`qwen35.cpp` `ssm_save_conv/rec(layer, t)`），而前缀缓存快照也用 **slot 0**（`serve.cpp` `prefix_state_save(0)`）⇒ 一次非 cacheable 的 MTP 请求会覆盖缓存快照，下一次 cache hit 恢复出**错误 recurrent state**（静默错误输出） | 中（需显式开 MTP） | 静态已证 |
| 5 | 第二个客户端 / `/health` / favicon 在流式生成期间 | **bug/限制**：HTTP 层严格单线程 `serve_connection`，一次只服务一个连接 ⇒ 其它连接排队到生成结束；`SO_RCVTIMEO` 只在 listening socket 上设置（占用 30 s 的 slowloris 会阻塞后续请求）| 中（webui/IDE 插件并发轮询会卡） | 静态 + 待实测 |
| 6 | 请求体使用 `Transfer-Encoding: chunked`（无 Content-Length） | **限制**：服务器只按 Content-Length 读 body，chunked 请求体被丢弃 ⇒ JSON 解析失败 | 低（浏览器 fetch/常用 SDK 都带 CL） | 静态 |
| 7 | `max_tokens: 0` | **语义**：按 llama.cpp 约定映射为 no-limit（默认 cap 1024），不是"生成 0 token" | 低（文档化即可） | 待实测确认 |
| 8 | `/props` 暴露本机 `model_path` | 信息暴露（llama.cpp 也这样） | 低 | 静态 |
| 9 | tokenizer：emoji ZWJ/家庭、CJK 扩展 B、RTL、组合字符、3000 字符无空格串、特殊 token | **正常**：9B 自身 encode→decode 往返全 PASS、特殊 token 原子；0.5B 与 Python 参考实现**逐 id 相同**（16 类边界） | — | 离线 GREEN |
| 10 | 客户端中途断开（RST）/停止生成 | 静态看：send 失败→`failed_`→`on_delta` 返回 false→`finish_reason=aborted`，`busy` 复位；前缀快照仍有效（下轮 restore+重算 tail） | 待实测 | 待实测 |
| 11 | 编辑历史 + 重新生成 | 静态看：LCP 不匹配时 `restore` 不发生、`forward(chunk,0)` 会 `reset_kv_and_states()`（start_pos==0）⇒ 正确；待与 `cache_prompt:false` 逐 token 对比 | 待实测 | 待实测 |
| 12 | `/v1/completions` vs `/completion` 同 prompt 同参数 | 静态看两者都走 `generate(st, prompt, "", gp, ...)`，应同 token；待实测 | 待实测 | 待实测 |
| 13 | 并发一个流式 + 一个非流式 | 串行执行（无数据竞争），第二个请求要等第一个结束；不会污染 prefix cache（无交错） | 限制，非数据污染 | 待实测 |

## 最小复现

### Bug 1：thinking + stop（长度≥2）→ content 为空
```bash
curl -s http://127.0.0.1:PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"用一句话介绍杭州。"}],
  "max_tokens":96,"temperature":0,
  "stop":["zzz"],
  "chat_template_kwargs":{"enable_thinking":true}
}' | python -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print('content=',repr(m.get('content')),'reasoning_len=',len(m.get('reasoning_content') or ''))"
```
期望：content 非空（stop 永远不出现，应正常答完）。
实际（静态推演）：content 为空、答案全在 reasoning。原因：`flush()` 的 stop holdback
`keep=max_stop-1=2` 字节，`</think>` 到达时 `cut=len(pending)-2` 正好落在 marker 内；
`feed_emitted` 收到 `"</thin"` 找不到标记，剩下的 `"k>"` 也再拼不回标记 ⇒ `in_think` 永远为真。
修复建议：`keep = max(max_stop-1, 8-1)`（thinking 激活时额外 hold 住 `</think>` 的 7 字节），
或把思考标记的切分挪到 token-piece 层（在 stop holdback 之前判定）。

### Bug 2：thinking + stop 匹配思考文本
```bash
# 同上，stop 换成 ["\n\n"]
```
期望：正常回答；实际：思考块第一个空行就把请求结束，content 为空。
修复建议：stop 只在 content（`</think>` 之后）里匹配；或在思考里命中 stop 时自动关闭
think 块并按服务端既定策略重试（类似 think budget）。

### Bug 3：坏请求返回 500
```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:PORT/v1/chat/completions \
  -H 'Content-Type: application/json' --data '{not json'      # 期望 400，实测 500
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:PORT/v1/chat/completions \
  -H 'Content-Type: application/json' --data '{"messages":[]}' # 期望 400，实测 500
```
原因：三个 POST handler 的 `parse_json_body`/`parse_messages`/`validate_prompt` 都在
`try` 之前；异常冒到 `http_server.cpp` 的通用 `catch (std::exception)` → 500。
修复：把 body 解析/校验移进 handler 的 try，或在 `route()`/server 捕获层对 `UserError`
返回 400（最小改动：`try { ... } catch (const UserError&) { conn.send_error(400,...) }` 包住整个 handler）。

### Bug 4：MTP 与 prefix-cache 共用 SSM slot 0
```bash
# 需要 35B-A3B（带 nextn）：
engine.exe serve --model 35B.gguf --mtp --ctx 4096 ...
# 1) cacheable 多轮请求 A（默认 cache_prompt=true）→ prefix_state_save(0)
# 2) 任意非 cacheable greedy 请求 B（如 /completion 且 prompt 长度是 128 的倍数，或 cache_prompt=false）
#    → spec_round() 的 verify 把 slot 0..k-1 覆盖
# 3) 再来一次 A 的前缀 → restore slot0 = B 的 draft 状态 → 错误 logits
```
修复建议：前缀缓存改用独立 slot（如 slot 3：`ensure_ssm_slots(4)` + `prefix_state_save(3)`），
或在 `st.mtp_enabled && st.pc.have_state` 时禁用投机（`greedy && !cacheable && !have_state`）。

### Bug 5：单线程 HTTP + 30 s slowloris
```bash
# 终端1：流式生成中
curl -N http://127.0.0.1:PORT/v1/chat/completions ... "stream":true ...
# 终端2：
time curl -s -o /dev/null -w '%{http_code} %{time_total}\n' http://127.0.0.1:PORT/health
```
期望：/health 立即返回；实际：等第一个流结束（数十秒）。
slowloris：
```python
s=socket.create_connection(("127.0.0.1",PORT)); s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost:x\r\nContent-Length: 100\r\n\r\n")  # 不发 body
# 之后 curl /health 会阻塞到 recv 超时（30 s）
```
修复建议：设置 accepted socket 的 `SO_RCVTIMEO`（例如 5 s）并在线程池/每次 accept 后
非阻塞服务（或至少单独处理 /health、限制并发连接数）。

## 修复（2026-11-21 22:xx，serve.cpp/http_server.cpp）

| # | 修复 | 位置 |
|---|---|---|
| 1 | thinking 块在**整段 pending 上先做 `</think>` 判定再落盘**（不是靠 holdback 字节数），`feed_emitted` 不会看到半截 marker；thinking 打开时 holdback 额外保留 8 字节（整 marker）作双保险 | `generate()` 的 `flush`/`feed_emitted` |
| 2 | stop 只在**可见 content** 里匹配：`in_think` 期间不匹配；`</think>` 之后的模板分隔换行（可能跨 token 到达）统一丢弃，`stop:["\n\n"]` 不会命中分隔符 | 同上 + `drop_leading_newlines` |
| 4 | MTP 与 prefix cache 共用 slot 0：启动时若两者都启用，打印 `WARN` 并**禁用投机**（更简单安全；独立 slot 方案留 v0.2） | `run()` 中 `st.mtp_enabled` 初始化后 |
| 5 | accepted socket 设 **5 s `SO_RCVTIMEO`**（slowloris 最多阻塞 5 s，不再 30 s）；线程池留 v0.2 | `http_server.cpp` accept 循环 |
| prewarm | warm 文本从固定 48× 改成循环加长到 `min(pw_chunk+8, ctx-8)`；预热切片/条件用 `pw_chunk=min(chunk,1024)`（预热 2048 要 +1.6 s 不划算）；修复 chunk>184 时永远跳过预热 | `run()` prewarm 段 |

说明：bug #3（坏请求 400）静态复核发现 `set_handler` 层已有 `catch (UserError) -> 400`，**大概率不是 bug**；
待 curl 实测 6 类确认后把报告条目改为"正常"或按需补修。

## 实测（`build/serve_edge_cases.py`，0.5B + 9B，2026-11-22 15:53–15:55）

两模型各 **25/25 PASS**（`build/logs/serve_edge_cases.json` 两侧均被覆盖，最新为 9B）：

| 用例 | 0.5B | 9B | 实测 |
|---|---|---|---|
| A1–A5 六类坏请求 | PASS | PASS | 全部 **400** + JSON error body（bug #3 确认**不存在**，`set_handler` 已映射） |
| A6–A9 max_tokens 边界 | PASS | PASS | 1/0/-1/1e6 都 200，不崩；0/-1 映射 no-limit（≤1024，文档化） |
| B1–B3 stop 排除 | PASS | PASS | `。`/`\n`/`###` 均不进入 content、finish=stop |
| **B4 thinking + stop=["zzz"]** | PASS | **PASS** | 9B 自然闭合（`--think-budget 0`）：reason_len=614、`</think>` 后 content="2"（非空）——bug #1 修复生效 |
| **B5 thinking + stop=["\n\n"]** | PASS | **PASS** | 同上：分隔换行被吞掉、答案 "2" 在 content（bug #2 修复生效） |
| C1 completions 一致性 | PASS | PASS | 同 prompt/参数输出逐字相同 |
| D1/D2 客户端断连 | PASS | PASS | 断连后下一请求 200、`/health` slots_idle=1（无 slot 泄漏） |
| E1 编辑历史 + 重新生成 | PASS | PASS | 与 `cache_prompt:false` 整段 prefill 的 token ids 完全相同 |
| F1–F3 并发 | PASS | PASS | 串行执行、两请求都成功；/health 在流式期间也能返回 |
| G slowloris | PASS | PASS | 半开请求只阻塞 **4.7 s**（5 s recv timeout 生效，修前 30 s） |
| H keep-alive | PASS | PASS | 不卡死 |
| I tokenize 往返 | PASS | PASS | emoji ZWJ/CJK-B/RTL 全部字节级往返 |
| J 采样复现 | PASS | PASS | 同 seed 下 cache on/off token ids 相同 |

补充（自然闭合 + stop 的定向验收，9B ctx4096，`--think-budget 0`，`--allow-oversubscribe` 因浏览器占 4.3 GB）：
`stop=["zzz"]`、`["\n\n"]`、`["。"]` 三种情况下 `</think>` 后 content 都**非空**（`"2"`），流式 SSE 同样
`</think>` 后 1 个字符（旧实现会把 marker 切半导致 content 为空）。
启动 prewarm：ctx4096/chunk2048 日志 `prewarm: 1024-token prefill ~0.9–1.0 s, 8-token tail ~0.13 s,
1-token decode ~0.03 s`（pw_chunk=1024；GPU 被浏览器占用时启动 wall 噪声大，干净窗口待补）。

## 结论
- bug #1/#2/#4/#5 + prewarm 跳过问题已修（本文档"修复"节），动态用例与定向验收 PASS。
- bug #3 撤销：坏请求本来就走 400，已有测试覆盖（A1–A5），无需改代码。
- 遗留（非本轮）：单线程 HTTP 的排队问题（线程池 v0.2）；MTP 需独立 SSM slot 才能与 prefix cache 共存。


## 发布闸门验证（2026-11-22 17:33–17:46，干净 worktree @ `53396ef`）

| 项 | 结果 |
|---|---|
| `build.sh`（干净 worktree） | OK |
| `verify.sh` | **ALL GREEN** |
| `smoke_models.sh` | **PASS**（40716/185/2614/2614） |
| `gen_regression.sh` | **10/10 GREEN**（含 coder_v2_lite=185；`dd96fa8` 修复后复验） |
| `verify_generality.sh` | **26/26 GREEN** |
| `verify_serve.sh` | 0.5B 22/22 + 9B 21/21 **GREEN** |
| `test_sampler.exe` | **PASS** |
| `tools/check_clean_build.sh --quick`（HEAD `8d1f1b1`） | **build OK / CLEAN BUILD GATE GREEN** |

过程发现并修复的两个"commit 不自洽"问题（已单独 commit）：
`3ca98ad` 的 `serve::run(Options&)` 定义与 `serve.h` 声明不一致（clean checkout 链接失败）→ `47555d2` 修复；
`gen_regression` 9/10 的一次假失败 = 主树 `generalize` 未提交 Wemma4 WIP 的加载污染（clean HEAD 10/10），
不是已提交 commit 的回归（真回归是 `d9bb37c` 的 MLA `upload_norm` → `dd96fa8` 已修并复验）。


## 追加发现（2026-11-22 18:0x，feature-polish 探针命中）：非流式 JSON 含非法 UTF-8

**场景**：9B 贪心生成在 `max_tokens` 处把多字节字符（emoji/CJK 扩展 B 的 byte-fallback token）切半，
`generate()` 的 force-flush 把不完整 UTF-8 字节直接写进 `out.content`，`json::Writer` 又原样透传
⇒ `/completion`、`/v1/chat/completions` 非流式响应体不是合法 UTF-8，严格客户端（`json.loads(raw_bytes)`、
浏览器/JS SDK）报 `'utf-8' codec can't decode bytes ... invalid continuation byte`；SSE 路径浏览器容错。

**复现**（9B，旧二进制 vs 修复后；`build/logs/utf8_repro.json`）：prompt 含 `𠀀`/emoji/罕见字，
`max_tokens=1..6`，30 个请求中旧二进制 **10 个响应严格 UTF-8 解码失败**（例：`byte 0xf0 ... invalid`、
`generated_token_ids` 尾部仍有完整 id，说明是文字截断不是写坏），修复后 **0/30**。

**修复**（两层，均已提交）：
1. `serve.cpp`：force-flush 前用 `text::incomplete_utf8_tail` 丢弃不完整尾字节（被 max_tokens 截断的半个字符
   本来就不存在）；
2. `include/engine/json.h`：`Writer::quote` 做 UTF-8 校验，非法序列替换为 **U+FFFD**（overlong/surrogate/
   截断都覆盖；合法序列原样透传），保证任何响应在严格客户端都可解析。
回归测试：`tests/test_json_utf8.cpp`（7 类非法/合法序列，build.sh 构建，`tools/check_clean_build.sh` 执行）；
`tools/serve_edge_cases.py` 新增 **K1**（0.5B/9B 26/26 PASS，非流式 JSON 严格解码+解析）。


## v0.1.6 hotfix 验收（2026-11-22 19:00，HEAD `d623cda`）

### 完整 clean-build gate（新 HEAD，含 UTF-8 修复 + 各 agent 文档）
`tools/check_clean_build.sh`（MODEL_DIR=/d/p0/models）输出
**`CLEAN BUILD GATE GREEN: d623cda is ready to tag`**（gate_rc=0）：
build.sh OK、verify **ALL GREEN**、smoke **OK**、**gen_regression GREEN**、generality **GREEN**、
verify_serve **GREEN**、JSON UTF-8 unit test **GREEN**。日志 `build/logs/full_gate_v016.log`。

### 真实浏览器 stop+thinking e2e（headless Edge + webui 页面内 fetch）
`build/run_stop_e2e.py` + `tools/serve_stop_e2e.py --skip-ui`（页面 fetch 非流式，规避 CDP 对
页面内 SSE 读取的挂起）：
- 200，耗时 4.6 s，文本 630 字，**`</think>` 存在**、其后 content = `"2"`（非空）、stop `zzz` 未出现 ✓
- 证据：`build/logs/e2e_stop/result.json` + `01_webui.png` / `02_stop_thinking_done.png`
  （上一版工具在 UI 流式 settle 上会挂住 → 已改为 UI 轮可选 + 非流式 fetch）。

### prewarm ON/OFF（9B ctx4096，2 次交错，独立 GPU）
| | startup（中位） | req0 prefill | req1 prefill |
|---|---|---|---|
| prewarm ON | **15.92 s** | **429 ms** | 428 ms |
| prewarm OFF（`ENGINE_NO_PREWARM=1`） | **15.16 s** | **495 ms** | 428 ms |

⇒ 预热（pw_chunk=1024）成本 **+0.76 s 启动**，仅省 **~66 ms** 冷首请求 prefill；本机页面缓存热时收益有限
（此前 chunk 1024 老代码测到 +0.44 s / −0.5 s，属不同工况）。建议：**默认保持开启但 pw_chunk 可再降到 256/512**，
或在 launcher 里做成可选；数据存 `build/logs/prewarm_ab.json`。
