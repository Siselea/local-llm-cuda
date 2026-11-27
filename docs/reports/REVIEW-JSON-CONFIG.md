# 外部审查 v0.1.7 — JSON 解析器 + 配置系统 缺陷核对（7 条）

范围：外部审查 34 条缺陷清单中分给 `generalize` 的 7 条（`src/json_parse.cpp`、
`src/config.cpp`）。规则：先逐条实测/静态核实 → 真缺陷修复 + 单测；假缺陷给出证据；
已修项标注 commit。

审计对象：v0.1.7 源码归档（`D:/v017_src/lingshi/`，与 `v0.1.7` tag 逐字节一致）
及当前 main（`dd1c61d`，JSON 解析器/配置系统与 v0.1.7 相同）。

结论速览：

| # | 缺陷 | 判定 | 说明 |
|---|---|---|---|
| 1 | JSON 接受无效数字 | **已确认，已修** | 实测接受 `01`/`-01`/`1.`/`+1`/`.`/`-`/`1e`/`1.e5`/`e5`，且 `1e999` → `inf` |
| 2 | JSON 接受未转义控制字符 | **已确认，已修** | 实测接受字符串内直接出现的 0x00–0x1F（NUL/TAB/LF/CR 等） |
| 3 | JSON 接受无效 Unicode 代理对 | **已确认，已修** | 实测接受孤立 `\ud800`/`\udc00`、`\ud800\ud800`，并输出非法 UTF-8 |
| 4 | 不完整转义可能读越界 | **误报（无 OOB）** | 静态 + 全截断前缀实测均安全；已加边界检查作为纵深防御 |
| 5 | 无嵌套深度限制 | **今日已修（e0f79d3）** | `kMaxDepth=128` + `tests/test_json_depth.cpp`（14 项） |
| 6 | 配置浮点选项接受 NaN ⇒ 整数转换 UB | **已确认，已修** | 实测 `nan` 通过范围检查；`get_int()` 得到 `INT64_MIN`（`(int64_t)NaN` 为 UB） |
| 7 | 配置 include 循环 ⇒ 无限递归崩溃 | **部分确认，已修** | `config = <file>` 递归确实存在且无环检测；本机（Windows/zig 运行时）在 ~512 层文件句柄耗尽后以误导性 `cannot open config file` 结束，非栈溢出崩溃；平台相关，仍属真实健壮性缺陷 |

---

## #1 JSON 接受无效数字 — 已确认（已修）

**修复前行为**（实测，`tests/json_probe.cpp` 探针）：

```
num_leading_zero       ACCEPT num=1
num_leading_zero_neg   ACCEPT num=-1
num_trailing_dot       ACCEPT num=1
num_leading_plus       ACCEPT num=1
num_dot_only           ACCEPT num=0
num_minus_only         ACCEPT num=0
num_exp_no_digits      ACCEPT num=1      (1e)
num_dot_exp            ACCEPT num=100000 (1.e5)
num_bare_e             ACCEPT num=0      (e5)
num_huge_exp           ACCEPT num=inf    (1e999)
```

根因：`parse_number()`（v0.1.7 `src/json_parse.cpp:221-236`）先扫描
`[-+.eE0-9]` 字符集，再把整段交给 `strtod()`，从不校验 RFC 8259 数法，也不检查
有限性。`01`/`1.`/`+1` 被 `strtod` 宽松接受；`"."`/`"-"`/`"e5"` 这类无数字串被
映射成 0；`1e999` 得到 `inf`。

**修复**：按 RFC 8259 逐字节校验
`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`，然后只把**通过校验的完整
token** 交给 `strtod`，并用 `std::isfinite()` 拒绝溢出到 `±inf` 的结果（下游
`serve` 会把 JSON 数字转成 `int`/`float`，非有限值会引发 UB）。

## #2 JSON 接受未转义控制字符 — 已确认（已修）

**修复前行为**：

```
str_raw_nul     ACCEPT bytes=61 00 62
str_raw_0x01    ACCEPT bytes=61 01 62
str_raw_0x1f    ACCEPT bytes=61 1f 62
str_tab_raw     ACCEPT bytes=61 09 62
str_newline_raw ACCEPT bytes=61 0a 62
```

根因：`parse_string()` 只把 `"` 和 `\` 当特殊字符，其余字节原样进结果
（v0.1.7 `src/json_parse.cpp:187-191`）。

**修复**：字符串内出现 `c < 0x20` 的非转义字节立即
`fail("unescaped control character in string")`；`\u0000`–`\u001f` 等合法
转义仍然接受，DEL(0x7f) 按 RFC 允许原样通过。

## #3 JSON 接受无效 Unicode 代理对 — 已确认（已修）

**修复前行为**：

```
str_lone_high      ACCEPT bytes=ED A0 80        (\ud800 孤立，CESU-8)
str_lone_low       ACCEPT bytes=ED B0 80        (\udc00 孤立)
str_high_high      ACCEPT bytes=EF B0 80        (\ud800\ud800 → 非法 3 字节序列)
str_high_then_text ACCEPT bytes=ED A0 80 78     (\ud800 后直接普通文本)
```

根因：v0.1.7 `parse_string()` 只在**下一个两字节恰好是 `\u`** 时尝试拼低代理，
否则把 `0xD800-0xDBFF` 直接交给 `append_utf8()`；低代理完全不校验
（`src/json_parse.cpp:203-214`）。

**修复**：高代理后必须紧跟 `\u` + 合法低代理 `DC00-DFFF`，否则
`unpaired high surrogate` / `invalid low surrogate`；`DC00-DFFF` 出现在首位报
`unpaired low surrogate`。合法代理对仍正确编码为 4 字节 UTF-8（测试断言
`\ud83d\ude00` → `F0 9F 98 80`）。

## #4 不完整转义可能读越界 — 误报（无 OOB）

外部审查者担心 `parse_string()` 的 `take()` 在 `"abc\`（EOF 前反斜杠）与
`\u12`（截断的 `\u`）路径上读到字符串末尾之外。

**结论：不构成越界读/UB。** 依据：

1. `take()` 的唯一实现是 `s_[pos_++]`。C++11 起
   `std::string::operator[](size())` 是**有定义**的（返回 `charT()` 的引用，
   即 `'\0'`）；`operator[]` 仅在 `pos > size()` 时 UB。
2. 逐个调用点核对：`parse_string` 循环入口已有 `pos_ >= s_.size()` 检查（报
   `unterminated string`）；`parse_hex4()` 在 `pos_ == s_.size()` 时 `take()`
   返回 `'\0'`，立即 `fail("bad \\u escape")` 抛出；`parse_object/parse_array`
   的 `take()` 最多在 `pos_ == size()` 后自增到 `size()+1` 就立即 `fail()` 抛出，
   之后不再访问字符串。没有可达路径在 `pos_ > size()` 时访问 `s_[pos_]`。
3. 实测：对 8 组含转义/`\u` 的种子串的**全部前缀**逐一 parse（含 `"abc\`、
   `\u`、`\u1..3`、截断代理对）— 全部正常抛异常，无崩溃；另做 5000 条随机
   畸形文档 fuzz，同样只有异常、无崩溃。

**加固**（非修 bug）：`take()` 现在也做边界检查，任何未来调用点都不会意外
越界；`parse_hex4()` 增加显式 `pos_ + 4 > size()` 截断检查，错误信息更准确
（`truncated \u escape`）。对合法输入行为不变。

## #5 无嵌套深度限制 — 今日已修（e0f79d3）

已在 commit `e0f79d3 fix(serve): cap JSON nesting depth` 修复：`kMaxDepth=128`，
`tests/test_json_depth.cpp` 14 项（含 200000/1000000 层不崩溃、127/128 接受、
129 拒绝）。本次核对确认归档/当前 main 均包含该修复，且 `build.sh` /
`tools/check_clean_build.sh` 已接入该测试。

## #6 配置浮点选项接受 NaN ⇒ 整数转换 UB — 已确认（已修）

**修复前行为**（实测探针 `config_probe float`）：

```
valid decimal  ACCEPT double=0.59999999999999998 finite=1  int=0
nan lowercase  ACCEPT double=nan               finite=0  int=-9223372036854775808
NaN mixed      ACCEPT double=nan               finite=0  int=-9223372036854775808
-nan           ACCEPT double=-nan              finite=0  int=-9223372036854775808
inf            REJECT (range check)
1e999          REJECT (strtod errno=ERANGE)
```

根因：`Config::apply_text()` Float 分支（v0.1.7 `src/config.cpp:592-608`）用
`strtod` 解析，随后只做 `v < minv || v > maxv` 范围检查。**任何与 NaN 的比较都是
false**，所以 NaN 通过范围检查；接着 `it.i = static_cast<int64_t>(v)` 把 NaN 转
整数 —— 按 C++ 标准是未定义行为（实测 x86-64 得到 `INT64_MIN`）。

**修复**：范围检查前加 `std::isfinite(v)` 检查，非有限值报
`option '<key>' expects a finite number, got '<text>'`。合法值行为不变；
`1e999`/`inf` 仍被拒。

## #7 配置 include 循环 ⇒ 无限递归崩溃 — 部分确认（已修）

配置文件的 `config = <file>` 键确实会递归加载（`src/config.cpp:757-762`，
`load_file()` 自身递归），且 v0.1.7 **没有任何已访问集合/深度上限**。构造
`a.cfg: config = b.cfg` + `b.cfg: config = a.cfg` 实测：

```
[dbg-load] open 'build/cfg_a.cfg' ... good=1
[dbg-load] open 'build/cfg_b.cfg' ... good=1
... (重复 512 层) ...
[dbg-load] open 'build/cfg_b.cfg' len=15 good=0
load_file returned 0 err=cannot open config file 'build/cfg_b.cfg'
```

即在本机（Windows + zig 运行时）递归先撞到**同时打开文件句柄上限（~512）**，以
误导性的 “cannot open config file” 结束，**没有**栈溢出崩溃；每层栈帧很小，
栈先耗尽的场景需要更大句柄上限（其他 CRT/平台）或更深的链。无论是否崩溃，这
都是真实缺陷：无环检测、无深度上限、错误信息误导。

**修复**：`load_file()` 维护**规范化路径的嵌套加载链**（`std::filesystem::
weakly_canonical`，能识别 `build/./a.cfg` 与 `build/a.cfg` 同一文件）：
- 重复路径 → `config include cycle detected: a -> b -> a`（含完整链）；
- 深度 ≥ 8 → `config include depth exceeds 8`；
- 合法单层/多层无环 include 行为不变（仍叠加加载，后加载覆盖前者）。

---

## 修复与验证

- 代码：`src/json_parse.cpp`（严格数字/控制字符/代理对 + `take()`/`parse_hex4`
  边界加固）、`src/config.cpp`（浮点有限性检查 + include 环检测/深度上限）。
- 单测：
  - `tests/test_json_strict.cpp` — 69 项：合法请求体/Unicode/转义必须通过，
    #1/#2/#3/#4 全部畸形用例必须拒绝，全部截断前缀 + 5000 条随机畸形文档不崩溃。
  - `tests/test_config_strict.cpp` — 65 项：三个浮点选项 NaN/Inf/溢出/垃圾/越界
    拒绝且不破坏旧值；整数/布尔严格性；合法 include 链生效；自环/双文件环/
    不同拼写环/超深链全部报明确错误；失败后普通配置仍可加载。
- 已接入 `build.sh`（两个新目标）与 `tools/check_clean_build.sh`（clean build
  gate 两项，失败即 RED）。
- 文档：`docs/CONFIG.md` 补充“嵌套 include 有环检测/深度上限”“浮点必须是有限
  数字”。
- 构建/运行均在不含他人 WIP 的独立 worktree（branch `fix/json-config-strict`）
  完成；`bash build.sh` 全量通过，`test_json_strict` / `test_config_strict` /
  既有 `test_json_depth` / `test_json_utf8` 全绿。

## 残留观察（未在本次 7 条内，建议后续排期）

1. **输入侧仍不校验原始 UTF-8 字节**：`parse_string()` 现在拒绝控制字符和非法
   代理对，但未校验未转义的多字节序列（如孤立 `0x80`、超长编码）。RFC 8259
   要求 JSON 文本是合法 UTF-8；当前该类输入仍会被接受。输出侧 `json::Writer`
   已有 U+FFFD 清洗（`tests/test_json_utf8.cpp`）。建议下一步在读取侧做严格
   UTF-8 校验（可能影响接受任意字节的旧客户端，需单独评估）。
2. **配置浮点仍接受十六进制浮点**（`strtod` 的 `0x1p3` → 8）。不影响安全，
   但 `docs/CONFIG.md` 只承诺十进制；如需严格可另加数法校验。
3. 配置 include 的相对路径仍按**进程 CWD** 解析（不是相对父配置文件），与
   v0.1.7 行为一致；若要改成语义更自然的“相对包含者”需要单独变更说明。
