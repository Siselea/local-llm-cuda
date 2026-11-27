# SERVE 安全与健壮性审计（HTTP / serve / tokenizer）

日期：2026-11-27　范围：`src/serve.cpp`、`src/http_server.cpp`、`src/json_parse.cpp`、`src/sampler.cpp`、`src/tokenizer.cpp`、`include/engine/text_util.h`、`src/main.cpp`
基线：v0.1.7 源码归档（`D:/v017_src/lingshi`，与 tag 一致）+ 当日修复后的 `dd1c61d`
修复 commit：**`68270dc`**（tokenizer/HTTP/数值/UTF-8）　相关既有修复：`e0f79d3`（JSON 嵌套深度，generalize）、`c1df298`（非流式 JSON U+FFFD 清洗）
方法：静态审计 + 原始 socket 构造 + 纯 HTTP/JSON fuzz + 0.5B serve（默认 loopback）+ 独立 worktree 构建；GPU 紧张时用 CPU 路径/离线二进制。

---

## 1. 结论速览

| # | 发现 | 严重度 | 判定 | 状态 |
|---|---|---|---|---|
| D1 | **tokenizer BPE 超线性 ⇒ 单请求 DoS**（长无分隔文本，如 1 MB `"中"`），1 MB `/tokenize` 让单线程服务 CPU 589 s+ 不返回 | **高** | REAL | **已修 + 复现/回归证据** |
| D2 | 用户可控输入缺少**工作量预算**（字符/时间/数量）：`stop` 可 32 MiB、`/detokenize` ids 可数百万、prompt 编码无上限 | 中 | REAL | 已修（`encode_bounded` + stop/detokenize/encode 预算） |
| D3 | 头部上限可被**边界绕过**（>64 KiB 头若终止符落在同一次 recv 中会被接受，最多 +~8 KiB） | 低 | REAL | 已修（终止符后复检 → 431） |
| D4 | HTTP framing 不严格：`atoll` 宽松解析 CL、重复 CL 被接受、chunked 请求体被静默错帧 | 中 | REAL | 已修（严格 CL→400、重复→400、TE→501） |
| D5 | 数值字段 `static_cast<int>(double)` 对 NaN/Inf/超大值 = **UB**；temperature/top_p/top_k/seed 等无范围校验 | 中 | REAL | 已修（finite+range → 400） |
| D6 | `text_util` UTF-8 解码过宽：接受 overlong（`C0 AF`）、代理项（`ED A0 80`）、>U+10FFFF（`F5 80 80 80`） | 中 | REAL | 已修（strict 校验 + 单测） |
| D7 | 单线程 HTTP：半开 body/慢头连接占住服务 ≤5 s/连接；无请求级总时限；任何超线性 handler 即 DoS 面 | 中（架构） | 已评估 | 缓解：输入预算化；根治 v0.2 线程池/总时限 |
| A1 | JSON 嵌套深度无限 ⇒ 栈溢出打崩进程 | 高 | REAL | 已由 generalize 在 `e0f79d3` 修复并验证（本报告不复测） |
| — | 静态文件路径遍历 | — | **FALSE POSITIVE** | `route()` 固定路径表；webui 是启动时按 exe 目录读的单个 index.html，无 URL→FS 映射 |
| — | 坏请求返回 500 | — | **FALSE POSITIVE** | `set_handler` 已有 `catch(UserError)→400`；6 类坏请求实测 400（v0.1.6 轮） |
| — | 客户端中途断开泄漏 slot/KV | — | **未发现** | RST 后下一请求正常、`/health` slots_idle=1；200 次压测待 GPU 窗口（见 §6） |

---

## 2. 已确认缺陷（修复前，代码位置 + 触发 + 后果）

### D1（HIGH）tokenizer BPE 超线性 ⇒ 单请求 DoS
- **位置**：`src/tokenizer.cpp: Tokenizer::bpe_word()`（修复前 ~549-601 行）。预分词器把连续 `\p{L}+` 当一个 word；中文长串/长无空格 ASCII 整段是一个 word。旧 merge 循环每轮**全量重扫所有相邻 pair**，每对还重新拼 `"a b"` 字符串 key：最坏 n 轮 × n 对 × O(串长) ⇒ 超线性（实测近似 O(n²)）。
- **触发**：任意 POST 编码入口：`/tokenize`、`/v1/chat/completions`、`/completion` 的 prompt/messages。
- **后果**：单线程 accept 循环被一个请求占满，所有其它请求（含 `/health`）排队无响应；1 MB 合法 body（远低于 32 MiB 上限）即可让服务"假死"数十分钟~数小时。默认 `127.0.0.1` 只影响本机；**`--host 0.0.0.0` 时局域网任意人可打**。
- **实测（修复前）**：`"中"*N`（含 ~0.25 s 进程启动）1k=0.33 s、2k=0.99 s、5k=5.17 s、**20k=92.0 s**；HTTP 发 1 MB `/tokenize` 后 engine CPU **589 s+** 仍在跑（RSS 434→617 MB），期间 `/health` 无响应。
- **修复（`68270dc`）**：
  - 算法：符号链表 + `(rank, left)` 最小堆 + lazy 失效（相邻性/rank 复核），同 rank 仍取最左位置 ⇒ O(n log n)，输出 id 不变；
  - 输入预算：`Tokenizer::encode_bounded(text, ..., max_chars, max_ms)`；serve 统一 `encode_user_text()`（生成端点 `min(200k, ctx*8)` 字符；`/tokenize` 用**全局 200k**，不受 ctx 限制）+ **1500 ms** 墙钟兜底；超限 **400** `prompt too long to tokenize: N characters (limit M)`；`stop` 总长 ≤4096、`/detokenize` ids ≤200k。
- **修复后实测**：20k CJK 编码 **~23 ms**（测试程序；进程整体 0.31 s）、50k 0.07 s、200k 0.47 s；**1 MB `/tokenize` → 400，20 ms**（不再进入编码逻辑）。42/42 golden 用例逐 id 与修复前相同。

### D2（中）缺少请求级输入预算
- 旧实现只有 body 32 MiB 硬上限；body 内的文本规模/`stop` 长度/ids 数量没有语义预算，配合超线性 handler 可放大 CPU/RAM。
- 修复：见 D1 的预算；`stop` 总字节 >4096 → 400；`/detokenize` ids >200k → 400；非数字 id / 非有限 / 超 int 范围 → 400。

### D3（低）header 上限边界绕过
- **位置**：`src/http_server.cpp: Conn::read_request()`。旧循环只在 `recv()` 前检查 `pending_.size() > kMaxHeaderBytes`；终止符在同一 recv 中出现时直接 break，`pending_` 可超限最多一个 recv 块（~8 KiB）。
- **实测（修复前）**：70 KB 头 + 立即终止的请求被正常处理（200），而不是 431。
- 修复：解析出 `header_end` 后复检 `header_end > kMaxHeaderBytes` → 431（`A5` 实测 70 KB→431）。

### D4（中）HTTP framing 不严格
- **位置**：`src/http_server.cpp` 请求头解析。
- 旧行为：`atoll(value)`（`"12abc"`→12、`"+5"`→5、`"-1"`→SIZE_MAX 靠 body 上限兜底、超长数字溢出 UB）；重复 `Content-Length` 后者覆盖前者（请求走私面）；`Transfer-Encoding: chunked` 被忽略并按 CL=0 读，残留 chunk 字节被当作下一请求（错帧）。
- 修复：`parse_content_length()` 严格 ASCII 数字/无符号/溢出界限：非法 → 400、超 body 上限 → 413；第二次 CL → 400 `duplicate Content-Length header`；任何 `Transfer-Encoding` → **501**（明确契约："不支持 chunked，请用 Content-Length"）。`tools/serve_edge_cases.py` A5/A7/A9 覆盖。

### D5（中）数值字段 UB / 无范围校验
- **位置**：`src/serve.cpp: params_from_body()` / `handle_detokenize()`。
- 旧代码 `int_or()` → `static_cast<int>(v->num)`：`1e999`→inf、`-1e999`→-inf、超大值转 int 是 **C++ UB**（gcc 常见产出 INT_MIN），后续 `max_tokens/ top_k / seed` 逻辑不可预测；temperature/top_p/min_p 无范围校验（inf 会改变采样分布）。
- 修复：`int_field()/num_field()`（`isfinite` + 文档区间）；`temperature∈[0,100]`、`top_p/min_p∈[0,1]`、`top_k∈[-1,1M]`、`max_tokens∈[-1,1M]`（-1=llama.cpp no-limit）、`seed∈[-1,INT_MAX]`、`repetition_penalty∈[0,100]`；越界 **400**。detokenize 的 id 同样 finite/range 检查。

### D6（中）UTF-8 解码过宽
- **位置**：`include/engine/text_util.h: utf8_next() / decode_utf8_lossy()`。
- 旧 `utf8_next` 只检查 continuation 字节，接受 overlong（`C0 AF`）、UTF-16 代理项（`ED A0 80`）、>U+10FFFF（`F5 80 80 80`），返回的"码点"非法；该 header 被 tokenizer（预分词）与 HTTP 共用。
- 修复：`utf8_decode_strict()`（最短编码长度、代理项、U+10FFFF 上限）；`utf8_next` 对非法输入返回 U+FFFD 并前进 1 字节；新增 `is_valid_utf8()/count_utf8_codepoints()`；`decode_utf8_lossy` 对 malformed 序列按 maximal-subpart 替换为**一个** U+FFFD（与 `json::Writer::quote` 的 `c1df298` 清洗一致）。
- **兼容性**：合法 UTF-8 的 tokenizer 输出逐 id 不变（42/42 golden）；serve 在 `/tokenize` content 与渲染后 prompt 上校验 UTF-8，非法 → 400（L3）。

### D7（中，架构）单线程 HTTP + 无请求总时限
- `Server::run()` 串行 accept，`serve_connection()` 同步处理请求；生成期间 `/health`、第二个客户端、favicon 全部排队。accepted socket 只有 5 s `SO_RCVTIMEO`；**半开 body / 每 5 s 重开的慢头连接**可持续占用；没有"每连接总时长"上限。
- 影响：任何超线性/长耗时 handler 都是 DoS 面（D1 是实例）；慢客户端也可造成队头阻塞。
- 本轮缓解：所有输入有界 + 编码墙钟预算 + 生成 max_tokens 受 ctx 约束；**根治**：v0.2 线程池/每连接总 deadline（记录在 §7）。

---

## 3. 已复现失败（修复前原始数据）

| 用例 | 修复前 | 修复后 |
|---|---|---|
| 20k CJK tokenizer | 92.0 s | ~23 ms（离线程序）；含启动 0.31 s |
| 200k CJK tokenizer | 不可用（外推 ~2.5 h） | 0.47 s（含启动） |
| 1 MB CJK `/tokenize` | 单线程 CPU 589 s+ 不返回；期间 /health 无响应 | **400 in 20 ms**（`prompt too long to tokenize: 349525 characters (limit 200000)`） |
| 50k CJK `/tokenize`（ctx=4096） | 会进入慢编码、且旧行为受 ctx 限制的说法不成立（旧无限制） | 200 in 0.07 s，tokens=50000（全局 cap，不受 ctx 限制） |
| 70 KB header 请求 | 200（上限被绕过） | 431 |
| duplicate Content-Length | 后值生效（可走私） | 400 |
| chunked 请求体 | 按 CL=0 处理、残留错帧 | 501 |
| `temperature:1e999` / `top_p:-1` / `top_k:1e999` / `seed:1e999` / `max_tokens:-5` | int 转换 UB / 无范围校验（多返回 200 或不可预测） | 全部 400 |
| `/tokenize` content 含 `C0 AF`/`ED A0 80` | 接受为非法码点 | 400 `'content' is not valid UTF-8` |
| 10 KB stop 序列 | 接受（holdback 巨量、流式被延迟） | 400 |

---

## 4. 修复验证（`68270dc`，独立 worktree `engine_tokenfix` @ dd1c61d）

- **构建**：主树 `bash build.sh` **exit 0**；worktree 构建 OK；新增 `tests/test_text_util.cpp`、`tests/test_tokenizer_budget.cpp` 编入 build.sh。
- **单测**：`test_text_util` GREEN（overlong/surrogate/>U+10FFFF/truncated/lone continuation + lossy U+FFFD）；`test_tokenizer_budget` GREEN（20k CJK < 500 ms 实测 23 ms；1 MB 拒绝 0.38 ms；消息清晰）。
- **数值一致性**：用 **HEAD 源码重编的 pre-fix 二进制**生成 golden，修复后 **42/42 用例逐 id 完全相同**（0.5B qwen2 + 9B qwen35 tokenizer；emoji ZWJ、CJK 扩展 B、RTL、组合字符、3000 无空格、2000 CJK、ChatML、docs 节选等）。注意：旧共享 `build/tt_ref.exe` 早于 HEAD 的 tokenizer 改动，用它做 golden 会误报 14/42——最终以 HEAD 重编的 golden 为准。
- **HTTP 用例**：`tools/serve_edge_cases.py`（0.5B）**31/31 PASS**：A1–A12 坏请求/CL/头/chunked、B stop/数值/消息/JSON 边界、C completions 一致性、D 断连/健康恢复、E 编辑历史、F 并发、G slowloris 4.7 s、H keep-alive、I tokenize 往返、J 采样复现、K 非流式 JSON UTF-8、**L1–L5 输入预算（1MB/50k tokenize、invalid UTF-8、1MB chat、10KB stop）**。
- **发布门禁（干净 worktree @ `70c8f0c`，完整 6/6）**：`build.sh` OK、`verify.sh` ALL GREEN、
  `smoke_models.sh` OK、**`gen_regression` GREEN（F16C 默认路径端到端）**、`verify_generality` GREEN、
  `verify_serve` GREEN（22/22），最终打印 **`CLEAN BUILD GATE GREEN: 70c8f0c is ready to tag`**。
  关键 loader 测试：`test_fp16` **131090 checks PASS**、`test_loader_bounds` **28 checks PASS**、
  `test_gguf_moe_offload` **F16/F32 双模型 PASS**（GPU / all-CPU / mixed；rel_l2 2.71e-7 / 2.68e-7）。
  注：首次 RED 仅为 `webui/index.html`（未入库抓取资产）缺失导致 `/` 404，种入资产后 22/22（代码零改动）。
- **修复前后对照**：见 §3 表。

---

## 5. 已核对为正常（避免误报）

| 检查 | 结论 |
|---|---|
| 静态文件路径遍历（`..`、`%2e%2e`、反斜杠、NTFS ADS、绝对路径） | **无风险**：`route()` 只匹配固定路径表；`/` 从 exe 相邻目录/`ENGINE_WEBUI_FILE` 读单个 index.html，**不存在 URL→文件系统映射**；8 种 traversal 变体均 404/无泄露（A2） |
| 方法白名单 | PUT/DELETE/TRACE/PATCH → 404，无副作用（A3）；OPTIONS 显式 204 + CORS |
| 404/错误响应泄露内部路径 | 404 只回显客户端 method+path；未出现 `C:\...`、`webui/index.html`、`candidates` 等内部串（A2/A4）；500 catch 的 `e.what()` 目前是业务错误文本（如 forward 错误），无栈/路径；建议 v0.2 对 500 也做 generic 化 |
| body/header 上限 | 32 MiB / 64 KiB 存在；本次修复补上边界绕过与严格 CL |
| JSON 嵌套深度 | 已由 `e0f79d3` 加 128 层上限（127/129/200k 层实测 400、不崩、服务存活；B1） |
| `/shutdown` 暴露面 | 仅当 `serve_host` 是 loopback（默认 `127.0.0.1`）时可调用；绑定 `0.0.0.0` 时返回 404（路由 loopback 判断） |
| 客户端 RST 后服务状态 | 单次 RST：send 失败→`on_delta` false→生成中止（finish_reason=aborted）、busy 复位；下一请求 200、`/health` slots_idle=1（D1/D2）；slot/prefix 快照复用同一 slot，无泄漏迹象 |
| 两并发流式请求 | 单线程串行：两个流式请求都能完整收完（最终 [DONE]+chunked 终止），无死锁/交错（D4）；第二个请求等待属设计取舍 |
| 生成中 `/shutdown` | 单线程 ⇒ `/shutdown` 要等当前生成结束才被处理；shutdown 后 `engine.exe` 正常退出、端口释放、VRAM 释放（D7/D9 实测；详见 §6 的并发变体） |
| `/tokenize` / `/detokenize` 边界 | 空串/id 数组、超长内容（现受全局 200k 上限）、非法 UTF-8、特殊 token 文本均 200/400 有界，无崩溃 |
| OpenAI 契约 | `max_tokens` 0/-1 映射 no-limit（llama.cpp 约定，文档化）；`messages` role 非法走模板/系统提示；`content` 数组（多模态）只取 text part；`stop` 空串被过滤 |

---

## 6. 仍需运行时验证（已由 Lead 决策推迟到 v0.1.9；不阻塞 v0.1.8）

1. **200 次"流式发起后立即 RST"压测**：连续 200 次观察 engine RSS、VRAM、`/health` slot 计数是否稳定（本轮只做了 1–3 次单发断连 + 断连后正确性）。脚本已就绪：`build/rst_shutdown_test.py`（200 cycles 采样 RSS/VRAM + shutdown 并发），v0.1.9 窗口执行。计划命令：
   `python build/keepalive_rst_stress.py --url http://127.0.0.1:7020 --cycles 200 --max-tokens 16`（启动 0.5B serve；每 25 次采样 RSS/VRAM）。
2. **慢读客户端 + 大输出**：客户端只读 header 不读 body，服务端 `send()` 是否会阻塞（需要 >socket 缓冲的输出；本机单线程下会体现在服务不可用窗口）。SO_SNDTIMEO 只在 listening socket 上设置，accepted socket 是否继承需实测。
3. **`/shutdown` 并发变体**：生成中 + 两个 `/shutdown` 同时到达（第二个连接在 backlog 中被处理还是被丢弃？）；shutdown 后进程退出码/端口/VRAM 清理。
4. **`--host 0.0.0.0` 局域网可达性**：默认 loopback 安全；绑定 0.0.0.0 后无鉴权，/props 暴露 model_path、/tokenize 等可被任意 LAN 客户使用；本次修复让单请求有界，但仍应只在可信网络使用。
5. **静态文件/协议细节**：HEAD 请求是否带 body（语义小问题）、`/props` 的 model_path 隐私（llama.cpp 也暴露，低危）。

---

## 7. 输入驱动 DoS 面清单 + 通用防御（本轮沉淀）

**攻击面清单**（所有"用户文本/数量进入 CPU 或内存"的路径）：

| 入口 | 用户可控工作量 | 修复前风险 | 现状 |
|---|---|---|---|
| `/tokenize` content | tokenizer 编码 | 超线性 BPE（D1） | 全局 200k 字符 + O(n log n) |
| chat/completions、`/completion` prompt/messages | 渲染模板 + 编码 + prefill | 同上 + 模板拼接 | `min(200k, ctx*8)` + 1500 ms 预算 + ctx 校验 |
| `stop` 数组 | holdback 长度、find 扫描 | 32 MiB stop 使流式 hold 巨量、延迟 abort 检测 | 4096 B 总长上限 |
| `/detokenize` ids | vector+解码 | 数百万 id（body 32MB 内）→ 内存/时间 | 200k ids + 数值范围 |
| JSON body | 解析递归/大数 | 嵌套深度栈溢出（已修）；超长数字串 O(n) | 深度 128；数字语法严格 |
| headers | 解析/内存 | 64 KiB 上限 + 边界绕过（已修） | 严格上限 431 |
| 生成 max_tokens/ctx | decode 时间（用户本就该请求长文） | 有 ctx 约束；no-limit 默认 1024 | 不变（产品语义） |

**通用防御原则（v0.2 建议固化为基础设施）**：
1. **每个 handler 声明工作量预算**：`(input_bytes, chars, items, cpu_ms)` 四元组，进入昂贵路径前检查，超限 400/413。
2. **请求级总 deadline**：不只在阶段级（5 s recv），而是"单请求从 accept 到结束 ≤ N s"；超时中止生成并释放 slot（配合取消标志）。
3. **算法复杂度契约**：所有"用户输入 → 循环"的代码路径（tokenizer、模板、holdback、JSON、detokenize）必须 O(n log n) 以内，并有 20k/200k 输入的回归基准（见 `tests/test_tokenizer_budget.cpp`）。
4. **单线程不是并发缺陷，但放大 DoS**：v0.2 用"小线程池 + 生成队列"或至少把 `/health` 放独立线程，避免一个 handler 让服务整体不可达。
5. **默认最小暴露**：默认 `127.0.0.1`；`--host 0.0.0.0`/公网必须显式，文档/启动日志加警告。
6. **错误信息脱敏**：4xx 带客户端信息（method+path）可接受；500 不输出内部路径/栈（当前是业务 `e.what()`，建议后续 generic）。

---

## 8. `--host 0.0.0.0` 的暴露面（重要）

默认 `serve_host = 127.0.0.1`（`src/config.cpp`），只本机可达；**API 无鉴权**（/v1/*、/tokenize、/props 全部开放，`/shutdown` 也仅 loopback 可调）。绑定 `--host 0.0.0.0` 后同局域网任何设备可：
- 发起生成/占满 GPU、读 `/props`（含 `model_path`）、用 `/tokenize` 消耗 CPU（本轮已把单请求压到有界，但高频请求仍可耗尽服务）、提交任意 prompt（数据留在本机日志/缓存）；
- DoS 面：本轮 D1 级别的问题在无预算前可被 LAN 任意人触发；修复后单请求有界，但**无速率限制/无鉴权**。

**建议**：保持默认 loopback；只有可信局域网才 `--host 0.0.0.0`，且不要在公网暴露；若必须远程使用，放在反向代理/SSH 隧道 + 鉴权之后。文档已在 `docs/SERVE.md` §5.11 记录。

---

## 9. 最小复现命令（修复前 → 修复后）

```bash
# 启动（0.5B，默认 loopback）
engine.exe serve D:/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 7020 --ctx 4096

# D1：1 MB 无空格 CJK（修复前会卡死服务；修复后 400 in ~20 ms）
python - <<'PY'
import json, urllib.request
body = json.dumps({"content": "中" * 349525}).encode()
try:
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:7020/tokenize", data=body,
        headers={"Content-Type": "application/json"}), timeout=10)
except urllib.error.HTTPError as e:
    print(e.code, e.read()[:160])
PY

# D3：70 KB 头（修复前 200；修复后 431）
python - <<'PY'
import socket
s = socket.create_connection(("127.0.0.1", 7020))
s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nX-Big: " + b"a"*70000 + b"\r\n\r\n")
print(s.recv(200).split(b"\r\n")[0])
PY

# D4：重复 Content-Length（修复前后者生效；修复后 400）
printf 'POST /tokenize HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\nContent-Length: 1\r\n\r\n{}' | \
  nc 127.0.0.1 7020

# D5：非有限数值（修复前 UB / 200；修复后 400）
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"temperature":1e999}'

# D6：invalid UTF-8 content（修复前接受；修复后 400）
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7020/tokenize \
  -H 'Content-Type: application/json' --data-binary $'{"content":"\xC0\xAF"}'
```

---

## 10. 状态与后续

- 修复 commit：**`68270dc`**（push 到 main），新增测试 `tests/test_text_util.cpp`、`tests/test_tokenizer_budget.cpp`，HTTP 用例扩展在 `tools/serve_edge_cases.py`（L1–L5）。
- 已完成的验证：主树/worktree 构建、两个单测 GREEN、42/42 tokenizer golden、HTTP 用例 31/31。
- **发布门禁已完成（2026-11-27，`70c8f0c`）**：干净 worktree 完整 6/6 → `CLEAN BUILD GATE GREEN`（详见 §4）。
  唯一环境项：`webui/index.html` 为未入库抓取资产，需在 gate worktree 中种入，否则 `/` 会 404（已知假失败）。
- **推迟到 v0.1.9**：§6 的 RST 200 次泄漏采样、shutdown 并发、慢读客户端大输出实测（Lead 决策，不阻塞 v0.1.8；
  脚本 `build/rst_shutdown_test.py` 已就绪）。
