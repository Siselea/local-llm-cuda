# 外部审查清单：Windows 发布包（4 条）对照

审查来源：外部审查者 34 条清单中的发布包条目（#31–#34）。本文件记录逐条判定、修复与实测证据。

| # | 文件 | 判定 | 修复 | 实测 |
|---|---|---|---|---|
| 31 | `stop.bat` | **REAL** | fallback 不再 `taskkill /im engine.exe`；改为按端口找监听 PID，且只 kill 镜像为 `engine.exe` 的 PID；`--all` 才全局 kill | 见下 |
| 32 | `启动.bat` | **REAL** | restart 分支同样按端口/PID，删掉全局 `taskkill /im`；非 engine 进程拒绝 kill | 见下/静态核对 |
| 33 | `stop.bat` | **REAL** | `/shutdown` 用 `curl -w "%%{http_code}"` 检查状态码；非 200 明确提示并继续走端口/PID 兜底 | 见下 |
| 34 | `README.md` | **REAL** | 下载引用改为 `/releases/latest` 的通用说明，不再写死 v0.1.6/7 zip 名 | README 本地链接检查通过；push 后核验远端 |

## 实测（stop.bat）

环境：Windows 11 x64，本机 `build/engine.exe` + qwen2.5-0.5B，两个独立实例：

- 实例 A：`127.0.0.1:7030`
- 实例 B（无关）：`127.0.0.1:7099`

| 场景 | 期望 | 结果 |
|---|---|---|
| 起 A+B 后执行 `stop.bat` | 只停 7030；7099 仍 `/health` OK | PASS |
| 再执行 `stop.bat --port 7098` | 明确输出 `Nothing is listening on port 7098` | PASS |
| 7098 上放非 engine 的 HTTP 服务并执行 `stop.bat --port 7098` | 显示 HTTP 非 200；识别非 engine 进程、拒绝 kill，服务仍存活 | PASS |
| 全程 | 不出现 `taskkill /im engine.exe`（除非显式 `--all`） | PASS |
| 7030 停止 | `/shutdown` HTTP **200**，port 释放 | PASS |
| 7099 无关实例 | 停止 7030 后 `/health` 仍 **200** | PASS |
| 7098 非 engine HTTP 服务 | HTTP **501** 被报告；识别 `python.exe` 并拒绝 kill，服务仍存活 | PASS |
| 7030/7098/7099 实测日志 | `build/test-stop-bat.log` | PASS |

`启动.bat` 的 restart 分支没有实际重启（会改变登录/模型状态），按相同代码路径做静态核对：
- 无 `taskkill /im`；
- HTTP 状态码有检查；
- 端口仍占用时只对监听 PID 且镜像为 `engine.exe` 执行 `taskkill /pid`；
- 非 engine 进程拒绝 kill 并退出。

## 编码

`stop.bat` 与 `启动.bat` 均保持 **ASCII + CRLF**（原文件就是纯 ASCII；`启动.bat` 仅是文件名含中文）。
修改后用 `cmd.exe` 实跑 `stop.bat`，输出与退出码正常；`启动.bat` 的 restart 分支 grep
确认无 `taskkill /im`（全局 kill 仅保留在 `stop.bat --all` 的显式分支）。
