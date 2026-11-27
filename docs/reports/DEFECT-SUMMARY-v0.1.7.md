# v0.1.7 缺陷总结（外部审查 + 我方复核）

> 对应审查输入：`local-llm-cuda-v0.1.7-win64.zip`（官方二进制资产）与
> `local-llm-cuda-v0.1.7-win64(1).zip`（"完整源码归档"）。
> 本文按 **已确认缺陷 / 已复现失败 / 仍需运行时验证** 三档归档，并记录已核对为正常的面。
> 源码口径：`v0.1.7 = 93245b3`（归档内源码抽样与 tag 逐字节一致）。

---

## A. 已确认缺陷（有代码位置 + 复现）

### A1. JSON 读取器无嵌套深度上限 ⇒ 单个请求即可打崩服务（高危，已修复）

| 项 | 内容 |
|---|---|
| 位置 | `src/json_parse.cpp`：`parse_value → parse_object/parse_array → parse_value` 互递归，无深度参数 |
| 触发 | 任意 POST 端点 + 深度嵌套 body，例如 200k 个 `[` |
| 后果 | **递归耗尽原生栈**，进程直接消失（连接 reset，无 4xx），属未授权的**远程 DoS**（服务监听非 loopback 时可被任意人对端触发） |
| 复现 | `engine.exe serve <0.5B> --port 7099` 后 `curl --data "$(python -c 'print("["*200000+"]"*200000)')" -H 'Content-Type: application/json' http://127.0.0.1:7099/tokenize` ⇒ 修复前：连接重置 + 进程退出；修复后：`HTTP 400 {"message":"invalid JSON body: JSON parse error: nesting too deep at offset 128"}` |
| 修复 | `src/json_parse.cpp` 增加 `kMaxDepth = 128` 与 `depth_` 计数，超限抛解析错误（正常映射到 400） |
| 回归 | 新增 `tests/test_json_depth.cpp`（14 检查：真实请求形状可解析；127/128 层可解析；129/300/20k/200k/1M 层一律拒绝且**不崩溃**；拒绝后解析器仍可用），已接入 `build.sh` 与 `tools/check_clean_build.sh` 发布闸门 |

**同一类问题的历史对照**：`src/gguf.cpp` 曾在 v0.1.6 前有完全相同的"metadata 数组无限递归 ⇒ 栈溢出"缺陷（`0xC00000FD`），当时修复为深度上限 16。HTTP 输入路径此前未被同一标准覆盖，本次补齐。

### A2. "完整源码归档"不可构建（打包缺陷）

| 项 | 内容 |
|---|---|
| 现象 | 归档根只有 `docs/ include/ pyref/ src/ tests/ tools/`，**缺少全部根文件**：`build.sh`、`README.md`、`LICENSE`、`CONTRIBUTING.md`、`THIRD-PARTY.md`、`engine.cfg.example` |
| 后果 | 拿到"完整源码"的用户**无法构建**，也无法确认许可证与使用方式；CI 配置（`.github/`）同样缺失 |
| 修复 | 新增 `tools/make_source_archive.py`：**白名单打包**（必需根文件 + 六个源码/文档树 + `.github/`），自动拒绝备份文件、`__pycache__`、临时日志；`--list` 可预览；打包后自检 debris 并在发现时返回非 0 |
| 产物 | `dist/local-llm-cuda-v0.1.7-src.zip`（580 文件，6,376,702 B，含全部 6 个根文件、0 debris），并已在空目录解包实跑 `bash build.sh` 验证可构建 |

### A3. 归档混入 50 个开发残留与字节码缓存（打包缺陷）

| 项 | 内容 |
|---|---|
| 现象 | `src/*.cpp.pre`、`.pre-17ms`、`.pre_ui2`、`.pre-kvzero`、`.v1`、`include/engine/*.h.pre` 等 **50 个编辑器备份**，以及 `pyref/__pycache__`、`tests/__pycache__` |
| 后果 | 读者无法判断哪份是正式源码；字节码缓存与解释器版本绑定，属无意义体积与噪声 |
| 修复 | 同上（`tools/make_source_archive.py` 的排除规则）；另建议：这些文件本就应删除或移入 `build/`（已被 gitignore），本次仅保证**发布归档不含它们** |

---

## B. 已复现失败

### B1. `POST /tokenize` 深嵌套 body ⇒ 进程退出（A1 的复现记录）

修复前实测输出（0.5B，ctx 512）：

```
$ curl -s -m 5 http://127.0.0.1:7099/health        # 修复前
{"status":"ok", ...}
$ (POST 200k 层嵌套数组 → /tokenize)
ConnectionResetError [WinError 10054]
$ curl -s -m 5 http://127.0.0.1:7099/health
SERVER DEAD                                        # 进程已退出
```

修复后同一请求：`400 invalid JSON body: JSON parse error: nesting too deep at offset 128`，`/health` 正常。

---

## C. 已核对为正常的面（避免误报）

| 检查项 | 结论 |
|---|---|
| 静态文件路径遍历 | **无风险**：`route()` 只匹配固定路径表（`/`、`/index.html`、`/health`、`/props`、`/v1/*` 等），webui 是启动时按 exe 相邻目录加载的单个 `index.html`，**不存在 URL → 文件系统映射** |
| 请求体大小 | `kMaxBodyBytes = 32 MiB`，超限直接拒绝 |
| 请求头大小 | `kMaxHeaderBytes = 64 KiB`，超限拒绝 |
| `Content-Length` 溢出/负数 | `atoll` 结果转 `size_t` 后被 `kMaxBodyBytes` 拦截；畸形值退化为 0 并最终 400 |
| 官方二进制资产完整性 | `D:/local-llm-cuda-v0.1.7-win64.zip` SHA256 = `f38df1c5ac74b19ebce7b5c25c953f83c4b9e6bf067c405e1683c76429e246b3`，与 release 记录一致 |
| 归档源码与 tag 一致性 | 抽样 `src/serve.cpp`、`src/runner.cpp`、`include/engine/kernels.h`、`src/gguf.cpp` 与 `v0.1.7` tag **sha1 逐一相同**（源码未遭篡改，问题只在打包与代码缺陷） |

---

## D. 仍需运行时验证（本次未覆盖）

1. **CUDA 内核边界与对齐**：`include/engine/kernels.h` 约 4 千行手写内核，需系统核对 `n_in/n_out` 非 32 倍数时的尾部处理、动态 shared memory 下标、`float4` 强制转换的对齐前提（已知 Q6_K 210 B stride 会 misaligned，历史上触发过 `CUresult=716`）。建议以"合成张量 + 随机 shape 对拍 + compute-sanitizer 类工具"方式验收（本机无管理员权限，需评估可用工具）。
2. **客户端中途断开的资源释放**：流式生成期间 RST 是否彻底中止并释放 slot / prefix 快照（已有 `--help` 级测试，但未做大规模 RST 压测）。
3. **并发请求**：HTTP 层为单线程串行服务，队列饥饿与公平性未量化（属设计取舍，非缺陷）。
4. **`--host 0.0.0.0` 暴露面**：默认 loopback，但用户可显式绑定到局域网；在 A1 修复前该场景等同"可被他人一请求打崩"。修复后建议在文档中明确"无鉴权、勿直接暴露公网"。

---

## E. 附：本次修复清单（代码/工具）

| 文件 | 变更 |
|---|---|
| `src/json_parse.cpp` | 深度上限 128 + `depth_` 计数 |
| `tests/test_json_depth.cpp` | 新增（14 检查） |
| `build.sh` / `tools/check_clean_build.sh` | 接入新测试到构建与发布闸门 |
| `tools/make_source_archive.py` | 新增（白名单源码打包器） |
| `docs/SOURCE-ARCHIVE.md` | 新增（源码归档规范） |
