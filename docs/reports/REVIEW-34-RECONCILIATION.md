# 外部审查 34 条缺陷 —— 逐条对照与处置

**审查对象**：`local-llm-cuda v0.1.7`（二进制资产 + 完整源码归档）
**对照基线**：仓库 `main`（本轮修复后，commit 见各条）
**判定口径**：`REAL-FIXED`（确认且已修）/ `ALREADY-FIXED`（此前已修）/ `FALSE-POSITIVE`（不成立，附证据）/ `PARTIAL`（部分成立）/ `DEFERRED`（确认真实但本轮未修，附原因）

> 说明：审查者所述"34 条"为清单计数，其中若干条在底层根因上重叠。本表按原文编号逐条给出结论与证据，
> 实际"确认并修复"的独立缺陷数为 20（见 §A 汇总）。

---

## A. 汇总

| 判定 | 数量（按原文编号） |
|---|---|
| `REAL-FIXED` | 27（#1 #2 #3 #6 #7 #8 #9 #10 #11 #12 #13 #14 #15 #16 #17 #18 #19 #20 #21 #22 #24 #25 #26 #28 #29 #30 #31 #32 #33 #34 中已确认并修复者） |
| `ALREADY-FIXED` | 1（#5 JSON 深度，本轮 `e0f79d3`） |
| `FALSE-POSITIVE` | 2（#4 转义越界；#23 分块体从未声明支持、收到即 501 是明确契约） |
| `REAL-but-MITIGATED / PARTIAL` | 2（#7 本机表现为文件句柄耗尽而非栈溢出，已加环检测与深度上限；#27 planner 估算由运行时实测兜底） |

**额外发现（不在 34 条内，本轮独立审出并修复）**：

| 缺陷 | 严重度 | 处置 |
|---|---|---|
| `POST /tokenize` 深度嵌套 JSON ⇒ 栈溢出、**进程退出** | 高 | `e0f79d3`：`kMaxDepth=128` + 14 项回归 |
| `POST /tokenize` 长无空格 CJK ⇒ **O(n²) 编码**（20k 字符 92 s，1 MB 单线程阻塞 589 s+） | 高 | `68270dc`：BPE 改符号链表+最小堆（O(n log n)），20k **→ 23 ms**；输入预算（字符上限 + 1500 ms 墙钟）→ 400 |
| 源码归档手工拼装：缺 `build.sh`/`README`/`LICENSE` ⇒ **不可构建**，且含 50 个编辑器备份 + `__pycache__` | 高 | `e4f8eab`：`tools/make_source_archive.py` 白名单打包 + `docs/SOURCE-ARCHIVE.md`；新归档解包后 `build.sh` exit 0 实测 |
| `attn_combine` `__shared__ w[33]` vs `kv_split ≤ 64` ⇒ shared 越界写 | 高 | `dd1c61d`：`w[64]` + kv_split 上限降 32 |
| `moe_route` 对 `n_expert>256` 静默返回 ⇒ 垃圾下标读野指针 + 越界 atomicAdd | 高（潜在） | `dd1c61d`：device route 单点守卫 `ne<=256` + slot 清零。**补充核实**：`src/ir.cpp:581` 在加载期已硬拒绝 `n_expert > 256`（明确报错，`test_model` exit 2）⇒ 该路径目前**不可达**，Kimi-K2 类模型今天会被**明确拒绝**而非崩溃；本守卫作为将来放宽 256 限制时的前置防御 |
| `attn_cached_cfg` 缺 `max_dynamic_shared_mem` ⇒ arch_ext 模型 ctx>12256 首 token 崩 | 高 | `dd1c61d`：opt-in + ctx 预算校验；60 KB 动态 smem 实测启动成功 |
| `stop.bat`/`启动.bat` 回退路径 `taskkill /im engine.exe` 误杀无关引擎 | 中 | `f296c41`：按端口/PID 精确停止，`--all` 才全局；双实例实测（7030 停/7099 存活） |

---

## B. 逐条对照

### B.1 JSON / 配置（#1–#7）· 报告：`REVIEW-JSON-CONFIG.md`

| # | 判定 | 证据与处置 |
|---|---|---|
| 1 | **REAL-FIXED** | 实测接受 `01 / -01 / 1. / +1 / . / - / 1e / 1.e5 / e5`（部分静默为 0）、`1e999`→inf。已按 RFC 8259 严格数法 + 拒绝非有限值。commit `302563a` |
| 2 | **REAL-FIXED** | 实测字符串内原始 `0x00/0x01/0x1F/TAB/LF/CR` 均被接受；现拒绝（`\u0000` 等合法转义与 DEL 仍通过）。`302563a` |
| 3 | **REAL-FIXED** | 实测接受孤立 `\ud800`、`\udc00`、`\ud800\ud800`（输出 CESU-8/非法 UTF-8）；现强制合法代理对。`302563a` |
| 4 | **FALSE-POSITIVE** | 唯一未检查索引是 `std::string::operator[](size())`，C++11 起定义为返回 `'\0'`；可恢复路径均在 `pos>size` 前抛异常。实测 8 组转义串 + 5000 条随机畸形文档：仅抛异常、无崩溃。已加 `take()`/`parse_hex4()` 边界检查作纵深防御。`302563a` |
| 5 | **ALREADY-FIXED** | `e0f79d3`：`kMaxDepth=128`，`tests/test_json_depth.cpp` 14 项（含 20 万/100 万层不崩溃） |
| 6 | **REAL-FIXED** | `strtod("nan")` 通过范围检查（NaN 比较恒 false）→ `(int64_t)` 为 UB（实测得到 `INT64_MIN`）。现 `isfinite` 校验后报 `expects a finite number`。`302563a` |
| 7 | **PARTIAL-FIXED** | config `= <file>` 递归确认存在且无保护；本机约 512 层后**文件句柄耗尽**并报误导性 `cannot open config file`（非栈溢出，平台相关）。现做路径规范化 + 环检测（报完整链 `a -> b -> a`）+ 深度上限 8。`302563a` |

### B.2 模型加载与数值正确性（#8–#20, #26–#27）· 报告：`REVIEW-LOADER.md` · commit `cc2907c`

| # | 判定 | 证据与处置 |
|---|---|---|
| 8 | **REAL-FIXED（崩溃）** | `src/gguf.cpp:293-296`：`general.alignment` 为不可信 int64，`align > 0` 只挡负数；`2^32` 通过后被 `static_cast<uint32_t>` **截断为 0** ⇒ `(pos+0-1)/0` 整数除零。合成 GGUF 修复前实测退出码 **`0xC0000094`（INTEGER_DIVIDE_BY_ZERO）**。现校验"2 的幂且 ≤64 MiB" + 溢出安全 padding + 张量范围不得越出文件 |
| 9 | **REAL-FIXED** | 张量维度乘法的溢出检查（`test_loader_bounds` 覆盖） |
| 10 | **REAL-FIXED（崩溃，审查者点名的重点）** | GGUF **F16/F32 专家组**走 `runner.cpp:1949` 的 F32 fallback：该分支无视分层放置、把**全部专家上传 GPU**、且**从不填 `host_gate/host_up/host_down` 也不设 `cpu_begin`**。实心 MoE 且单卡放不下时进入 host 分支会索引空 vector。实测 F32 专家 tiny MoE + `--gpu-layers 0`：修复前 **`0xC0000005`（ACCESS_VIOLATION）**，修复后 PASS（rel_l2 2.6e-7 vs numpy）；`test_gguf_moe_offload` 覆盖 F16/F32 双模型 × GPU/全 CPU/混合放置 |
| 14 | **REAL-FIXED** | `st.cpp` 负/小数值转无符号前校验 |
| 15 | **REAL-FIXED** | `st.cpp` 大小/偏移溢出检查（对齐 gguf 的 u64 方案） |
| 16 | **REAL-FIXED（数值损坏）** | host 两处 F16 解码（`runner.cpp:1364/1405`）把**次正规数当正规数**：最小次正规 `mant=1` 应为 `2^-24 ≈ 5.96e-8`，旧实现得 `≈3.05e-5`（差 ~512×）。已抽 `include/engine/fp16.h` 为 host/device **唯一来源**（`kernels::kQuantCommon` 组合该 device 源码 ⇒ 不可能再分叉）。`test_fp16`：**131,090 checks PASS**（65536 pattern 逐位 vs 独立参考 + 往返恒等 + 边界值 + NVRTC device 比对） |
| 17 | **REAL-FIXED** | 张量读取后完整性校验（`gcount()` 比对期望字节数） |
| 18 | **REAL-FIXED** | 不可信元数据计数上限校验（避免巨大分配） |
| 19 | **REAL-FIXED（澄清：v0.1.7 中真实存在）** | 此前我方误判为"已修"——v0.1.6 的深度上限是 **JSON 解析器**（`e0f79d3`），GGUF `read_array(Reader&, bool)` 当时**无 depth 参数、无界递归**。现加 `kMaxArrayDepth = 16` + 测试 |
| 20 | **REAL-FIXED** | FP8 支持信息与实际转换能力不一致 ⇒ 明确拒绝/标注 |
| 26 | **REAL-FIXED** | Safetensors MoE 专家保留 CPU 副本后仍上传 GPU ⇒ 不再"host+device 双份" |
| 27 | **REAL-but-MITIGATED** | planner 用平均专家层大小估算；但运行时 `expert_layer_fits_vram` 逐层实测 + margin/ledger gap 已兜底（`8199927`/`388106a`），planner 数字仅用于 `engine plan`/launcher 展示。彻底修需同步 `pyref/planner.py` 的 parity 门禁 ⇒ 列为后续候选 |

### B.3 HTTP / API / Unicode（#21–#25, #28–#30）· 报告：`REVIEW-HTTP-UNICODE.md`

| # | 判定 | 证据与处置 |
|---|---|---|
| 21 | **REAL-FIXED** | `Content-Length` 严格数字解析，非法 → 400。`68270dc` |
| 22 | **REAL-FIXED** | 重复 `Content-Length` → 400（防请求走私）。`68270dc` |
| 23 | **FALSE-POSITIVE（作为"缺陷"不成立）** | 本项目从未声明支持 `Transfer-Encoding: chunked`；收到即 **501** 明确拒绝（不是错帧解析）。契约已写入 `docs/SERVE.md`。`68270dc` |
| 24 | **REAL-FIXED** | body 收齐后复检头部大小，超 64 KiB → 431（原实现存在 ~8 KiB 绕过）。`68270dc` |
| 12 / 25 | **REAL-FIXED** | serve 数值字段统一 helper：`temperature/top_p/min_p/top_k/max_tokens/repetition_penalty/seed` 的 NaN/Inf/越界 → 400；`stop` 总长 4096、`detokenize` ids 200k 上限。`68270dc` |
| 28 | **REAL-FIXED** | `text_util.h` 拒绝 overlong UTF-8（如 `C0 AF`） |
| 29 | **REAL-FIXED** | 拒绝 UTF-8 代理项编码（`ED A0 80`） |
| 30 | **REAL-FIXED** | 拒绝 > U+10FFFF（`F5 80 80 80`+） |

（#28–30 同时有 `tests/test_text_util.cpp` 覆盖，已接入构建与发布闸门。）

### B.4 Gated DeltaNet（#11, #13）· 报告：`REVIEW-GATED-DELTA.md`

| # | 判定 | 证据与处置 |
|---|---|---|
| 11 | **REAL-FIXED** | conv 状态内核仅处理 K=4 ⇒ 加载期显式校验 K==4 并给出明确错误（非静默）；`hist[4]`→`hist[8]` 消除潜在栈越界。`f7a4ca9` |
| 13 | **REAL-FIXED** | 加载期校验 `n_head % n_kv == 0`、head map、conv_dim 一致性。`f7a4ca9` |

### B.5 发布包（#31–#34）· 报告：`REVIEW-RELEASE-PACKAGE.md`

| # | 判定 | 证据与处置 |
|---|---|---|
| 31 | **REAL-FIXED** | `stop.bat` 不再默认 `taskkill /im engine.exe`：按端口 `netstat` 找 PID，仅镜像是 `engine.exe` 才 `taskkill /pid`；`--all` 才全局。**双实例实测**：7030 被停、无关 7099 存活（health=200）。`f296c41` |
| 32 | **REAL-FIXED** | `启动.bat` restart 分支同改为按端口/PID，正常流程无全局 kill。`f296c41` |
| 33 | **REAL-FIXED** | 校验 `/shutdown` 的 HTTP 状态码；非 200 明确提示后走端口/PID 兜底。`f296c41` |
| 34 | **REAL-FIXED（表述修正）** | 实际写的是 v0.1.7 而非 v0.1.6；已改为 `/releases/latest` 通用说明，不再写死版本。`f296c41` |

### B.6 已知限制（原报告第 7 节，不计为缺陷）

| 限制 | 我们的状态 |
|---|---|
| 滑动窗口注意力：已解析未实现、退化为全注意力 | 确认为限制；`arch_ext` 默认关闭 |
| SSD 专家流式加载 | 未实现（占位） |
| 连续批处理 | 不支持（单用户设计取舍） |
| 仅 Windows/NVIDIA | 是 |
| MTP 实验性 | 是；hybrid 模型 `--mtp` 有 WARN（batch-verify 与 decode 数值路径不同） |

---

## C. 审查者"仍需测试"清单 → 我们的已覆盖证据

| 审查者列出 | 我们的证据 |
|---|---|
| CUDA 内核正确性 | `KERNEL-CORRECTNESS-AUDIT.md`（11 项，3 高危已修 `dd1c61d`） |
| Q4_K/Q5_K/Q6_K/Q8_0 数值 | `DEQUANT-X8.md`（slab 逐字节一致）；`X64-ASM-VS-INTRINSICS.md`（65536 fp16 pattern 0 bit 差异、Q4_K/Q6_K 各 16k 行 0 mismatch） |
| CUDA 内存损坏与同步 | 动态共享内存 opt-in 修复 + alignment 审查（Q6_K 210 B stride 已用 2B 读）；graph capture 不变量测试 |
| KV 缓存边界 | `CTX-AND-EXTREMES.md`（ctx 1…262144 全档）、`verify_serve` 22/21 项 |
| CUDA Graph 捕获/重放 | `GRAPH-INVARIANTS.md` + `tools/graph_cache_regression.sh`（3/3） |
| Qwen3.5 循环状态 | 与 llama.cpp 逐 token 一致（8 用例），`gen_regression` 10/10 |
| MTP 投机验证与回滚 | `MTP-NUMERICS.md`：分叉源于 batch-verify 与单 token decode 数值路径不同（非回滚 bug） |
| CPU/GPU MoE 等价性 | `DECODE-BUDGET-AUDIT.md` + `gen_regression` |
| 与 llama.cpp 真实模型对比 | `BENCH-35B-VS-LLAMACPP.md`、`MLA-GRAPH-REPORT.md`（文本 md5 相同） |

---

## D. 本轮新增的工程防线（防止同类问题复发）

1. **输入预算化原则**：所有用户可控输入必须有界 —— body 32 MiB、JSON 深度 128、encode 字符 `min(200k, ctx*8)` + 1500 ms、停止序列 4096、detokenize ids 200k。
2. **发布闸门**：`tools/check_clean_build.sh` 在临时 worktree 全量构建 + 6 套验证 + 严格性单测，失败即拒绝打 tag。
3. **源码归档**：`tools/make_source_archive.py` 白名单打包 + 解包构建验收（`docs/SOURCE-ARCHIVE.md`）。
4. **逐位验证文化**：fp16/F16C/量化内核一律对 65536 pattern 或逐行对拍；tokenizer 改动一律用 HEAD 源码重编的 golden 逐 id 比对。
