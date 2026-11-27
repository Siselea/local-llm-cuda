# REVIEW-HTTP-UNICODE — 外部 34 条清单：HTTP / serve 数值 / Unicode 段落的逐条判定

日期：2026-11-27　负责人：server-mode　修复 commit：`68270dc`
源码基线：`dd1c61d`（v0.1.7 + 当日修复），修复后主树 `build.sh exit 0`，独立 worktree
`engine_tokenfix` 完成单测与 HTTP 用例。

判定口径：
- **REAL/FIXED** = 外部审查成立，已修 + 测试
- **REAL/OWNED-ELSEWHERE** = 成立但归其他 agent 的文件（不在本段）
- **FALSE POSITIVE** = 代码已经正确拒绝
- **NEEDS RUNTIME** = 静态看不构成缺陷，需专门运行时窗口确认

---

## A. HTTP 层（4 条）

| # | 指控 | 判定 | 依据 / 修法 | 验证 |
|---|---|---|---|---|
| 21 | `Content-Length` 用 `atoll()` 不严格验证 | **REAL/FIXED** | 旧代码 `static_cast<size_t>(std::atoll(value.c_str()))`：`"12abc"`→12、`"+5"`→5、`"-1"`→SIZE_MAX（靠 body 上限兜底）、超长数字是 `atoll` 溢出 UB。改为 `parse_content_length()`：只接受 ASCII 数字、不允许符号/空串、逐位溢出保护；`Invalid` → **400**，`TooLarge` → **413** | 边界用例：`-1`/`99999999999999999999`/33 MiB 声明 → 413；`12abc`/`+5`/空 → 400 |
| 22 | 重复 `Content-Length` 未拒绝 | **REAL/FIXED** | 旧代码后出现的值覆盖前面（请求走私面）。现在第二次出现立即 **400 `duplicate Content-Length header`** 并关闭连接 | 原始 socket 发两条 CL → 400（`tools/serve_edge_cases.py` A7） |
| 23 | 传入 chunked 请求体未解码 | **REAL/FIXED（明确契约）** | 服务器**不支持** chunked 请求体；旧代码忽略 `Transfer-Encoding`、按 CL=0 读空 body，残留 chunk 字节会被当作下一个请求（错帧）。现在任何 `Transfer-Encoding` → **501 `request Transfer-Encoding is not supported; send a Content-Length body`**，客户端必须用 `Content-Length` | 原始 socket chunked → 501（A9） |
| 24 | 收齐后未复检头部上限 | **REAL/FIXED** | 旧循环只在每次 `recv()` **前**检查 `pending > 64 KiB`；当最后一个 recv 把 `\r\n\r\n` 带进来时直接 `break`，最多可超限 ~8 KiB 被接受（实测 70 KB 头 → 200）。现在 header 终止符解析后复检 `header_end > kMaxHeaderBytes` → **431** | 70 KB header → 431（A5）；正常大 header 仍 200（A6） |

补充（已知设计取舍，非缺陷）：单线程 accept ⇒ 半开 body/慢头连接可占住服务 **≤5 s/连接**（`SO_RCVTIMEO`，A10/A11 实测 5.0 s/4.7 s）。属"单线程 + 无总时长预算"的架构面，建议 v0.2 线程池或每连接总时限；本轮输入预算已把最坏单请求 CPU 时间压到有界。

## B. serve 数值 API（2 条：原编号 12 / 25）

| 指控 | 判定 | 依据 / 修法 | 验证 |
|---|---|---|---|
| 浮点→整数转换未做范围检查（`max_tokens`、`n_predict`、`top_k`、`seed`、`n_probs`… NaN/Inf/负数/巨大值 ⇒ UB） | **REAL/FIXED** | 旧代码 `static_cast<int>(v->num)` 对 `inf`/`1e999`/超 int 范围是 C++ **UB**（常见结果 INT_MIN，后续逻辑不可预测）。新增 `int_field()` / `num_field()`：`isfinite` + 区间检查，越界 **400**；`max_tokens` 允许 -1（llama.cpp no-limit 约定），其余字段文档化区间（temperature [0,100]、top_p/min_p [0,1]、top_k [-1,1M]、seed [-1,INT_MAX]…）。同时 `stop` 总长 4096、`/detokenize` ids ≤200k | `1e999`/`-1e999`/`top_p=-1`/`top_k=-1e999` 等 11 组 → 全部 400 且服务存活（B2/L5）；`max_tokens=0/-1` 仍按文档映射 no-limit |
| 另注：JSON 的 `NaN`/`Infinity` 字面量 | **FALSE POSITIVE** | 解析器 number 只吃 `[-+0-9.eE]`，`NaN`/`Infinity` 会走 parse 错误 → 400；真正的 Inf 只能由 `1e999` 溢出产生，已被 B 段拦下 | 与 json_parse 的 strict commit 一致 |

## C. Unicode 工具（3 条：28 / 29 / 30，`include/engine/text_util.h`）

| # | 指控 | 判定 | 依据 / 修法 | 验证 |
|---|---|---|---|---|
| 28 | 接受 overlong UTF-8（如 `C0 AF`） | **REAL/FIXED** | 旧 `utf8_next` 只检查 continuation 字节，`C0 AF` 会被当作 U+2F；`utf8_seq_len(C0)` 也得 2。新增 `utf8_decode_strict()`：按最短编码长度校验（2字节<0x80、3字节<0x800、4字节<0x10000 一律拒绝），`utf8_next` 对失败返回 U+FFFD 并前进 1 字节 | `test_text_util.exe`：C0 AF / E0 80 AF / F0 80 80 AF 全拒绝 |
| 29 | 接受 UTF-16 代理项编码（`ED A0 80`） | **REAL/FIXED** | strict 检查 `0xD800..0xDFFF` → 拒绝；`decode_utf8_lossy` 对代理项输出 U+FFFD（采用 maximal-subpart，一个坏字符只替换一次，和 `json::Writer::quote` 一致） | surrogate ED A0 80 拒绝 + lossy→U+FFFD |
| 30 | 接受 >U+10FFFF（`F5 80 80 80`+） | **REAL/FIXED** | strict 检查 `cp > 0x10FFFF` → 拒绝；`decode_utf8_lossy` 同样 | F5 80 80 80 拒绝 |

**兼容性**：这些是 inline helper，tokenizer 与 HTTP 共用。修复后：
- **合法 UTF-8 的 tokenizer 行为逐 id 不变**：42/42 golden 用例（0.5B+9B）与 HEAD pre-fix 二进制完全一致。
- 输出端 U+FFFD 清洗（`c1df298`：serve 截断尾字节丢弃 + `json::Writer::quote` 非法序列→U+FFFD）与本次 strict 解码方向一致，没有冲突。
- serve 入口额外校验：`/tokenize` 的 `content`、chat/completions 渲染后的 prompt 若含非法 UTF-8 → **400 `not valid UTF-8`**（L3 实测）。

## D. 额外（本轮审计发现、不在 34 条里）

| 发现 | 判定 | 说明 |
|---|---|---|
| **tokenizer O(n²) DoS**（长无空格文本 ⇒ BPE 每轮全量重扫） | **REAL/FIXED（HIGH）** | 20k CJK 92 s → ~23 ms；1 MB `/tokenize` 从"单线程阻塞 589 s+"变 **20 ms 400**。详见 `SERVE-SECURITY-AUDIT.md` |
| 输入驱动 DoS 通用面（stop 超长、detokenize 超大 ids、chat prompt 超长） | **REAL/FIXED（MEDIUM）** | 统一字符/时间/数量预算，全部 400（L1–L5） |

## 结论
- 外部 HTTP/数值/Unicode 9 条：**8 REAL/FIXED + 1 FALSE POSITIVE**；另修 1 条 HIGH（tokenizer DoS）。
- 修复已提交 `68270dc`，单测 + HTTP 用例 31/31 + golden 42/42；五套与干净 worktree 发布 gate 待 GPU 窗口。
