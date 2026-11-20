# tools — 验证与对拍脚本

所有脚本都在仓库根目录下执行（`python tools/xxx.py`）。Python 解释器默认
`python`（`verify.sh` 里可用 `PY=` 覆盖）。

## 日常验证

| 脚本 | 作用 |
|---|---|
| `verify.sh` | **主验证入口**，51 项检查，应输出 ALL GREEN；所有夹具自助生成 |
| `gen_regression.sh` | **真文本生成回归（qwen35 GGUF 语义防线）**：多 token prefill 后 greedy 生成 8–12 token，与 llama.cpp ground truth **逐 id** 对比；9B 5 组 + 35B 3 组 + Coder/0.5B 守卫（约 6 分钟） |
| `verify_reference.sh` | 可选扩展：与官方 transformers 参考实现逐层对拍（约 15 分钟）。**qwen35 的 GGUF 部分已改为 llama.cpp 语义判据**（HF 链只对 safetensors 有效，见 `q35_variants.py`） |
| `smoke_models.sh` | **四个本机模型冒烟测试**：各自跑一次并打印预测与速度（约 5 分钟）。⚠ 单 token 冒烟**不能**发现 qwen35 GGUF 语义错误，请同时跑 `gen_regression.sh` |
| `diff_ir.py <model>` | G0 门槛：C++ IR 与 Python 参考 IR 逐字段对比 |
| `diff_plan.py <model>` | M2 门槛：C++ 与 Python 的放置计划对比（18 个字段） |
| `oracle_compare.py` | 与 llama.cpp（llama-server）比对贪心生成结果 |
| `q51_check.py` | 校验 Q5_1 的设备端点积公式与 llama.cpp dequant 是否一致 |
| `bench_cpu_matvec.cpp` | CPU 量化 GEMV 微基准（`ziglang c++ -O2 -mavx2 -mfma -Iinclude tools/bench_cpu_matvec.cpp src/cpu_matvec.cpp -o build/bench_cpu_matvec.exe`）。注意：单副本矩阵（2.4 MB）落在 L3 里，量到的是 cache 带宽，不是 DRAM |
| `bench_cpu_layer.cpp` | **CPU 专家层基准**（SPEC §29）：24 个独立副本 × 16.3 MB 轮转，一个副本远大于 12 MB L3，因此量的是 DRAM。`layer` 测整个 `cpu::moe_forward_mt`，`matvec` 测单矩阵，`read` 测同字节纯读上限（线程数可调，用于判断是否触及 DRAM 天花板）。构建：`ziglang c++ -std=c++17 -O2 -Iinclude tools/bench_cpu_layer.cpp src/cpu_matvec.cpp src/config.cpp -o build/bench_cpu_layer.exe` |

## 夹具生成（由 verify.sh 调用）

| 脚本 | 作用 |
|---|---|
| `make_quant_fixture.py <gguf> <tensor> <out> <n>` | 导出真实 GGUF 的量化块 + f32 参考 |
| `make_gemv_fixture.py <gguf> <tensor> <out> <n>` | 导出 GEMV 夹具（权重 + x + y） |
| `make_gguf_ref.py` | 生成小尺寸参考 GGUF |

## 官方参考实现对拍（本地 `transformers` 包）

| 脚本 | 目标模型 | 说明 |
|---|---|---|
| `q35_torch_chain.py` | Qwen3.5-9B (`qwen3_5`) | 逐层跑官方模块，与引擎 dump 对比。**只对 safetensors/HF 语义有效**；GGUF 文件请用 `q35_variants.py --a-fix 1 --bcast tile`（GGUF 转换器写的是 llama.cpp 约定：`ssm_a=-A`、q/k→v 块重复广播） |
| `q35_variants.py` | Qwen3.5-9B (GGUF) | **GGUF vs HF 语义裁判工具**：numpy 重写 dense qwen35 栈，可切 `ssm_a` 约定（log A / -A）与 q/k→v 广播（repeat_interleave / 块重复）。默认跑 4 变体对照表（根因证据）；`--a-fix 1 --bcast tile --dump P --layer N --tolerance T` 作为 GGUF 逐层回归参考（超限返回 1）。用于定位"单 token 绿、真文本崩"这类 GGUF 语义错位 |
| `q35moe_torch_chain.py` | Qwen3.6-35B (`qwen3_5_moe`) | 同上（MoE 变体，逐层加载 256 专家）。**同样需要注意 GGUF/HF 语义区分** |
| `dsv2_torch_chain.py` | Coder-V2-Lite (`deepseek_v2`) | 同上（MLA + MoE） |
| `q35_torch_ref.py` / `q35_torch_attn.py` | 单层/单模块对比 | 定位到具体 kernel 时使用 |

## 诊断探针（排查过程中使用，保留作为证据）

| 脚本 | 用途 |
|---|---|
| `q35_full_probe.py` | qwen35 全模型 numpy 前向，逐层对比引擎 dump |
| `q35_probe.py` / `q35_attn_probe.py` | 单层线性层/注意力层的 numpy 对拍 |
| `q35_state_probe.py` | 分块 prefill vs 逐 token 的状态一致性（曾抓到 conv 状态 bug） |
| `q35_moe_probe.py` / `ds2_moe_probe.py` / `ds2_full_probe.py` | MoE 层对拍 |
| `mla_v_probe.py` / `mla_q_probe.py` / `mla_q_yarn_probe.py` | MLA 的 V / q / YaRN rope 三步定位 |
| `q35moe_attn_probe.py` | 35B 注意力层对拍 |
| `layout_sweep.py` | MLA 布局组合扫描（历史实验） |

## 通用化 / 配置面验证（v1 新增）

| 脚本 | 作用 |
|---|---|
| `verify_generality.sh` | 配置优先级（CLI>config>env>默认）、非法值报错、校准缓存生成/回读、显存 margin 扫描、优雅失败、四模型 info 校验（约 2 分钟） |
| `make_bad_models.py` | 从 `build/fixtures/tiny_gguf_moe` 变异出三类坏模型：未知架构 / 不支持量化类型 / 未知张量命名，用于验证错误信息质量 |
| `bench_perf.sh [tag]` | 四模型 decode 基准（`BIN=./build/test_gen.exe` 可切换二进制，输出 `build/perf-<tag>.txt`） |
| `bench_perf_ab.sh` | 改造前/后两个二进制顺序对比（先 baseline 后 after，减少时钟漂移） |

## 分词器 / HTTP 服务（v1.1 新增）

| 脚本 / 工具 | 作用 |
|---|---|
| `verify_tokenizer.sh` | **分词器验收**：Unicode 表新鲜度、C++ self-test（roundtrip + 特殊 token + ChatML 模板）、与 llama.cpp `llama-tokenize` 的 token-id 对拍（合成语料 + 真实文档），约 1–2 分钟；`--quick` 跳过对拍 |
| `check_output_quality.py --json/--sse <file> [--expect-cjk/--expect-ascii]` | **输出质量闸门**（纯 CPU）：空输出、符号汤、复读、复述 prompt 判 FAIL；已内嵌进 `verify_serve.sh` / `serve_35b_e2e.sh`，防止"HTTP 200 即 PASS"的盲区 |
| `final_acceptance.sh [--with-longctx]` | **最终验收一键跑**：0.5B→9B→35B 三档 e2e + 可选 4k 长上下文；每档打印 PASS/FAIL，结束打印汇总 + 残留进程检查 |
| `serve_longctx_check.sh [model] [port] [ctx] [tokens]` | 长上下文请求验收（默认 9B/7034/8192/4000 token）：用服务自己的 /tokenize 构造 ~4k 中文 prompt，量 prefill 延迟与首答是否正常，退出关服务 |
| `repro_qwen35_prefill.sh` | **qwen35 多 token prefill bug 最小复现**（9B llama.cpp ground truth 对照、batched vs 逐 token、旧二进制对照、35B 符号汤案例），输出 `build/logs/repro_qwen35.log` |
| `serve_35b_e2e.sh [model] [port]` | **35B 端到端验收**（7030）：test_gen greedy 参考、中文单轮、多轮 messages、stream=true SSE、300/500 token 中等 prompt、`/tokenize` 中文往返、与 test_gen 逐 token 对拍、`/shutdown` 优雅退出+显存释放；要求 GPU 干净（>2 GB 占用会拒绝启动） |
| `verify_http.sh [port]` | **HTTP 层纯逻辑单测**（不加载模型/不用 GPU）：keep-alive、128 KiB POST、chunked 流、CORS、404/400、HTTP/1.0 close、pipelined、`/shutdown`（`tests/test_http.cpp` + `tools/verify_http.py`） |
| `verify_serve.sh [model] [port]` | **HTTP 服务端到端验收**（默认 qwen2.5-0.5b@7031）：启动/启动行、`/health` `/v1/models` `/props` `/`、`/tokenize`+`/detokenize`、chat 非流式/SSE、`/v1/completions`、llama.cpp `/completion`、404/400、与 `test_gen` 的 greedy 逐 token 对拍、`/shutdown` 优雅退出+端口释放 |
| `serve_stability_check.py --port <p> [--tag 35b] [--json out.json]` | **稳定性实机检查**（2026-11-17）：两轮 `你好 / 帮我写1000字小说`（期望 `finish_reason=repetition`、不再跑到 cap）+ 三条对照（你好/介绍杭州/写 Python，期望不误杀）+ 一个需要思考的问题（think-budget 到点后干净重答，检查无规划文本残留） |
| `serve_mtp_check.py --port <p> --tag off --save base.json` / `--tag on --baseline base.json` | **MTP serve 验证**：同一 greedy prompt 在 `--mtp` 开/关两个 server 下逐 token 对比（投机解码必须 bit-exact）+ tok/s；加载失败会直接报错 |
| `tests/test_serve_guard.cpp`（构建：`ziglang c++ -std=c++17 -O2 -Iinclude tests/test_serve_guard.cpp -o build/test_serve_guard.exe`） | **repeat guard 单测 24 项**：真实复读模式（句×28/字符级/交错 5 句循环/变体句）+ 不误杀（正常故事/英文/对话/思考表头/markdown/表格/Python 代码） |
| `serve_webui_probe.py --url <base> [--turns 3] [--log <serve.log>]` | **llama.cpp webui 行为探针**（2026-11-16 新增）：按 webui 真实请求体（`stream:true, cache_prompt:true, max_tokens:-1, samplers=..., timings_per_token:true`）发多轮中文请求，量**首字 delta 延迟**、验证 `delta.content` 里的 `<think>`/`</think>` 与最终回答、解析 `timings` 与日志 `[new N, cached M]` |
| `webui_e2e.py --url <base> --turns 3 [--messages "a|b|c"]` | **真实浏览器验收**（2026-11-16）：headless Edge + CDP 驱动 llama.cpp webui —— 真打字、点发送、流式轮询、每轮截图、读会话标题/DOM；`e2e_report.json` 记录首字延迟、总耗时、渲染文本。用于"像真实用户一样"的端到端验收 |
| `serve_drift_probe.py --url <base> [--turns 3] [--greedy] [--temp/--top-k/--top-p/--min-p]` | **思考漂移探针**：按 webui 参数跑多轮对话并统计思考段"同一句话重复次数"，输出 `DRIFT DETECTED` / `NO DRIFT`；`--greedy` 用 temp=0 对照 |
| `serve_prefix_cache_check.py --url <base> [--turns 8] [--filler-repeat N]` | **多轮前缀缓存 A/B 硬验收**（2026-11-16 新增）：同一对话逐轮交替发 `cache_prompt=false`（整段 prefill 基线）与 `true`（前缀缓存），**逐 token 对比贪心输出**并对比 prefill 时间；全绿输出 `PREFIX CACHE GREEN` |
| `dump_tok.py <gguf>` | dump GGUF 里所有 tokenizer.* 键（类型/长度/特殊 token 表/token_type 直方图/chat template），确认 pre-tokenizer 家族 |
| `tok_ref.py <gguf> [--candidates]` | Python 参考分词器（BPE+SPM，多个候选 pre-tokenizer 正则），用于给 C++ 实现定标 |
| `check_tokenizer.py candidates --model <gguf>` | 用 llama-tokenize 反推 GGUF 的 pre 名对应哪条 llama.cpp 正则（本仓库已确认 qwen35 精确端口） |
| `check_tokenizer.py cpp --model <gguf> --exe build/test_tokenizer.exe` | C++ 分词器 vs llama-tokenize：合成语料 + README/CONFIG 等真实文档 + chat 模板 token id 逐 token 对拍 |
| `gen_unicode_tables.py [--check]` | 生成 `include/engine/unicode_tables.h`（Unicode 一般类别 + llama.cpp whitespace 集合，供 pre-tokenizer 使用） |
| `build/test_tokenizer.exe <gguf> --self-test` | 分词器自检；`--info` 打印元数据；`--ids-file/--decode-ids` 给脚本调用 |
| `build/test_sampler.exe` | 采样器单元测试（seed 可复现、greedy、top-k/top-p/min-p、重复惩罚窗口） |

## 说明

- 调试探针保留是为了让报告里的每个结论都能复现；它们不属于日常验证路径。
- `q35_*` 里的数字是**增量偏移**（dump 文件后缀），对应引擎的 `ENGINE_DUMP=EVERY` 机制。

## 利用率 / 气泡度量（v1 新增，交付见 `../UTILIZATION-REPORT.md`）

这一组工具回答"引擎跑起来 CPU/GPU 占用都上不去"到底空在哪里。它们**不修改引擎**，只通过
环境变量打开引擎已有的剖析口，再从外部按 10 ms 采样 NVML + 进程 CPU 时间。

| 文件 | 作用 |
|---|---|
| `measure_utilization.cpp` | 外部采样器（C++，动态加载 `nvml.dll`）：spawn 子进程、按 `--interval-ms` 采 GPU 利用率/显存/功耗/SM 时钟/温度/pstate + 子进程 CPU 时间/整机 CPU 时间，并给子进程每一行 stderr 打时间戳。**刻意不用 `nvidia-smi --loop-ms`**：每次 spawn nvidia-smi 要 50–100 ms，10 ms 档位做不出来；单进程 NVML 一次采样约几微秒。 |
| `utilization_report.py` | 解析器/报告器：把 `*.timeline.csv` + 带时间戳的 `*.engine.log` 对齐，解析 `[time]/[fwd]/[layerms]/[kprof]/[cpuexec]/[pipe]/[moe-prof]`，输出每层"GPU 时间 / 主机时间 / 空闲"表、按 layer class 的 NVML 利用率、kprof 逐内核表、文本时间序列和 `summary.md/summary.json`。 |
| `run_utilization_suite.sh` | 四模型取证套件（每个 run 都是 load→采样→退出，不常驻）。模式：`graph`（默认执行方式，真墙钟+真 NVML）、`probe`（`--no-graph` + `ENGINE_KPROF + ENGINE_DBG_LAYERMS + ENGINE_DBG_CPUEXEC + ENGINE_PIPE_PROF`，逐层）、`prefix1`/`prefixN`（`--layers N` 做前缀差分）、`prof`（`ENGINE_PROFILE`，MoE 分段）。 |
| `run_utilization_extras.sh` | 主套件不覆盖的补充 run：35B 的 `layerms`（无 kprof 开销的真逐层墙钟）、35B/Coder 显式钉住放置的 `probe/prefix`（绕开会挂死的 adaptive 放置路径）、稠密模型的 graph `--layers` 差分。报告的 40 行逐层表来自这里。 |
| `run_class_gpu_probe.sh` | 稠密模型（0.5B/9B）在 **CUDA Graph 模式**下用 `--layers 1` 与 `--layers N` 的墙钟差分出"每层真实 GPU 时间"（graph 模式无提交气泡，W(N)=N·p+H）。`--no-graph` 下的 kprof 事件会把主机提交气泡算进 kernel elapsed，这个工具绕开该假象。 |

### 构建与用法

```bash
# 构建采样器（与引擎同一套 ziglang 工具链，无 MSVC/CUDA toolkit 依赖）
python -m ziglang c++ -std=c++17 -O2 tools/measure_utilization.cpp \
    -o tools/util_bin/measure_utilization.exe

# 单次采样（会先等 GPU/CPU 静默，再 spawn；超时未静默会记录到 meta.json）
./tools/util_bin/measure_utilization.exe --out-prefix tools/util_results/x/graph/measurement \
    --interval-ms 10 --wait-gpu-idle-sec 900 --idle-mem-max-mb 1200 --cpu-idle-max 45 \
    --during-check-ms 5000 --timeout-ms 300000 \
    -- ./tools/util_bin/test_gen_snap.exe <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf 760 257

# 解析单次 / 整套
python tools/utilization_report.py --run tools/util_results/qwen3.5-9B/graph
python tools/utilization_report.py --root tools/util_results \
    --md tools/util_results/summary.md --json tools/util_results/summary.json

# 四模型全量（约 12–15 分钟 GPU 时间，13 个 run）
bash tools/run_utilization_suite.sh
# 稠密模型的 graph 前缀差分（4 个短 run，约 1.5 分钟）
bash tools/run_class_gpu_probe.sh
```

输出目录：`tools/util_results/<model>/<mode>/measurement.{timeline.csv,engine.log,meta.json}`；
`summary.md` 是自动生成的表（报告里的数字都可由它复核）。引擎二进制用
`tools/util_bin/test_gen_snap.exe`（`sha256` 记录在 `tools/util_results/MANIFEST.md`），
避免并发重建 `build/test_gen.exe` 让测量失效。

### 度量口径与已知假象（读报告前必看）

1. **NVML 的时间分辨率是 ~1 s 级**：`utilization.gpu` / `power.draw` 由驱动按滚动窗口更新，
   10 ms 只是"读得勤"，不是"1 s 内有 100 个独立样本"。因此利用率均值只对 **≥5 s 的解码窗口**
   有意义（`graph` 模式每个 run 6–13 s）；`probe`/`prefix` 这类 <2 s 的窗口只用于逐层时间，
   不看 NVML 均值。
2. **桌面本身有 ~25 % 的"利用率底噪"**：本机常驻 NVIDIA Overlay/Edge WebView/Explorer 等
   CEF 进程，空闲时 NVML 也报 10–27 %、~17 W。采样器在每个 run 前采 2 s
   `ambient_*` 基线写进 `meta.json`，报告里并列给出；不要把原始 util 当成"引擎占用"。
3. **`ENGINE_KPROF` 的 kernel elapsed 在 `--no-graph` 下会被主机提交气泡污染**：CUDA event
   在 host 提交慢时把 GPU 空等时间也算进 kernel 里（0.5B 实测 kprof 10.98 ms vs graph 模式
   整 token 6.9 ms）。所以 `kprof GPU busy%` 只在"GPU 是瓶颈"时可信（9B：kprof 95.8 %，
   NVML 96 %，两者一致）；主机受限的场景（0.5B no-graph、35B CPU 层）它给出的是
   "kernel 时间+提交气泡"的上界。稠密模型每层真实 GPU 时间请用 `run_class_gpu_probe.sh`。
4. **并发智能体会污染采样**：本机是两个 agent 共享的。采样器在启动前等待"无
   `test_gen*/engine*/llama*` 进程 + 显存 < `--idle-mem-max-mb` + 整机 CPU < 阈值"，运行中每 5 s
   复查一次并写进 `meta.json`（`foreign_before_start` / `foreign_during_run_hits` /
   `foreign_after_run`），报告里 `contaminated` 一栏为 `YES` 的 run 全部作废重跑。
   另外引擎在长 decode 时本身有一个偶发挂死（`cpu::Pool` 丢唤醒的特征，详见
   `../UTILIZATION-REPORT.md` §10），所以采样器提供 `--timeout-ms`；被超时 kill 的 run
   在 `meta.json` 里标 `timed_out:true`，挂死前已完成的数据仍可用。
5. **`[placement] adaptive:` 自适应减层路径会挂死**：显存紧张时引擎会打印
   `[placement] adaptive: ... -> gpu_moe_layers N -> M` 然后卡住（本次 3 次复现）。
   采样时用 `--gpu-layers N --no-vram-check` 显式钉住干净机器的放置即可绕开；
   跑基线/对拍时同理（看到 adaptive 那行就 kill 重跑）。
6. **warmup / CUDA Graph capture token 要排除**：稠密模型第一个 decode token 要现场捕获
   整图（0.5B 实测 ~15 ms vs 稳态 6.9 ms）。`[fwd]` 行把每个 token 切开，解析器自动识别
   token 0 的 capture 首帧并剔除；`engine avg decode`（引擎自报）仍包含它，报告用
   "steady decode (median)" 作为对比口径。

### general-opt 新增的诊断/度量工具（2026-11-15，长上下文 + CPU int8 轮）

| 工具 | 用途 |
|---|---|
| `longctx.sh` | 9B 长上下文探针：短 prompt vs 384/512/1024/2048 token 预填（`--ctx 4096/8192`），打印 prefill ms/tok/s、decode ms/token、top-5 与生成 token 序列。用于 `LONGCTX-REPORT.md` 的全部数字 |
| `ab35b.sh` | 35B 双二进制交错 A/B（`test_gen_pre_opt.exe` vs `test_gen.exe`）+ `ENGINE_DBG_CPUEXEC`/`ENGINE_DBG_MOE_PHASES` 相位数据；每轮打印 topk 以确认预测不变 |
| `ab_cpu.sh` | CPU host GEMV 的 A/B：两个 `bench_cpu_layer` 构建交错跑同一形状，报 **min 与 median**（本机噪声下 min 更稳） |
| `dbg_i8.cpp` | int8 行内核最小复现/数值对拍：`#include "../src/cpu_matvec.cpp"` 直接调匿名命名空间里的 kernel，逐行对 fp64 反量化参照，打印每个 XQ8 块的 d1/d2/sx32 |
| `dbg_i8_alu.cpp` | 行内核的**纯 ALU** 吞吐（同一行反复 dot，L1 热）：分离"指令受限 vs 带宽受限"，是"Q4_K int8 反而慢、Q6_K int8 快 1.31×"的判据 |
| `dbg_i8b.cpp` | Q4_K int8 的 256 值单块手工复算（打印每个 sub-block 的 int32 部分和 vs fp64 参照），定位 sub-block 索引 bug 用 |
| `ab_int8.sh` | 35B CPU int8 的**同二进制**端到端 A/B（只切 `ENGINE_CPU_INT8=0/1`，所以隔离的正是 int8 路径）：交错 N 次，同时抓 `[cpuexec]` 单层 host matvec、`[moeph]` 相位 A/B、topk 与 decode 中位。`BIN=` 可指定二进制 |
| `ab_int8_cpu.sh` | 纯 CPU 的 int8 A/B：`bench_cpu_layer` 的 layer 模式（24 副本轮转）交错跑 `ENGINE_CPU_INT8=0/1`，报 min/中位/配对差。**不需要 GPU**，是默认开关的判据来源 |
| `dbg_topo.cpp` | 打印 `physical_cores()` 与 `hardware_concurrency()`（验证 CPU 线程池的"按物理核推导"）：本机 6 核 / 12 逻辑 → 9 线程 |
| `longctx.sh` | （已在上一版加入）长上下文探针；现在支持 `BIN=` 指定构建，因为 `build/test_gen.exe` 可能被其它智能体的进程占用 |

---

## 诊断 / 一次性开发脚本（不参与构建，不参与四套验证）

以下脚本是开发过程中用来定位具体问题的一次性/诊断工具，**保留作为可复现证据**，
但**不会**被 `build.sh`、`verify.sh`、CI、launcher 使用，后续 PR 也不需要跑它们。
它们依赖本机模型路径/环境变量（`MODEL_DIR`、`ENGINE_*`），按需使用即可。

| 类别 | 文件 | 用途 |
|---|---|---|
| CPU 内核数值/ALU 诊断 | `dbg_i8.cpp`, `dbg_i8_alu.cpp`, `dbg_i8b.cpp`, `dbg_topo.cpp` | int8 行内核逐块复算、纯 ALU vs 带宽、Q4_K 索引 bug 定位、物理核数打印 |
| MoE/MLA/qwen35 逐层探针 | `q35_*_probe.py`, `q35_full_probe.py`, `q35_state_probe.py`, `q35_variants.py`, `q35moe_attn_probe.py`, `ds2_*_probe.py`, `ds2_full_probe.py`, `mla_*_probe.py` | 与 numpy/torch 官方实现逐层对拍、定位 SSM/MLA/MoE 语义差 |
| 利用/带宽探针 | `cpu_probe.py`, `cpu_watch.py`, `cpu_watch_trigger.py`, `measure_utilization.cpp`, `utilization_report.py`, `bench_cpu_layer.cpp`, `bench_cpu_matvec.cpp`, `bench_gd.cpp`, `bench_gemv_hbm.cpp`, `bench_q6k_dp4a.cpp`, `bench_rp.cpp` | GPU/CPU 利用率、CPU 专家 DRAM 带宽、内核微基准（历史报告用） |
| MTP/布局实验 | `mtp_probe.cpp`, `probe_blas.cpp`, `layout_sweep.py`, `q51_check.py`, `q35_torch_*.py` | MTP 头布局、BLAS 行为、MLA 布局扫描、Q5_1 反量化对拍 |
| 一次性 A/B/复现脚本 | `ab35*.sh`, `b35.sh`, `ab_*.sh`, `bench_*.sh`, `prof9b*.sh`, `prof35b*.sh`, `run_utilization_*.sh`, `mtp_base_35b.sh`, `numcheck.sh`, `margin35.sh`, `pf_final_measure.sh`, `repro_qwen35_prefill.sh`, `final_table.sh`, `serve_35b_e2e.sh`, `serve_longctx_check.sh` | 具体报告里的 A/B、复现、端到端数字；多数需要本机模型/GPU，不适合 CI |
| 采样/夹具辅助 | `make_bad_models.py`, `make_quant_fixture.py`, `make_gemv_fixture.py`, `make_gguf_ref.py`, `check_output_quality.py`, `diff_ir.py`, `diff_plan.py`, `oracle_compare.py`, `serve_webui_probe.py`, `webui_e2e.py`, `serve_drift_probe.py`, `serve_prefix_cache_check.py` | 验证/诊断辅助（部分被 `verify_serve.sh` 调用，保留在原位） |

> 说明：`util_bin/`、`util_results*/` 是本地测量产物，已在 `.gitignore` 中排除；
> `tools/utilization_report.py` 只依赖 Python 标准库，可对采样结果重新出表。


### 2026-11-17/20 轮新增（线程绑核 / serve 稳定性 / 发布集成）

| 工具 | 用途 |
|---|---|
| `thread_scan.py [--reps 6] [--iters 150]` | CPU host-GEMV 的线程数 × `win_affinity` 扫描（默认 4..12 × {0,2}），轮转消除漂移，输出 min/p25/median/max 与 GB/s；`THREAD-AFFINITY-REPORT.md` 的数据来源 |
| `thread_e2e.sh` | 35B 端到端 decode A/B：旧 9 线程/OS vs 新 12 线程/SMT 成对 vs 10 线程，记录 ms/token、`[layerms]`、placement 与 token md5；需 35B 模型和空闲 GPU |
| `bench_35b_engine.sh <tag> [0|1] [budget] [ctx] [kv] [max_tokens]` | 启动 `engine serve` 的两轮真实对话 benchmark（含 GPU 采样与 JSON 输出）；`MODEL_DIR` 必填 |
| `moe_gap_probe.sh` | 同模型同 GPU 层数对比 llama-bench vs 自研引擎 decode，并给出逐层归因；`MODEL_DIR` + `LLAMA_BENCH` |
| `genreg_case_probe.sh` | 重跑 gen_regression 的 35B 单例并保留完整 stderr，区分 OOM/loader 错误与真实 token 差异 |
| `window_round.sh [1..5|all]` | 安静窗口的一键收尾流程：build → verify → gen_regression → 线程 e2e → MoE gap |
| `serve_twoturn_bench.py` | 两轮真实对话（你好 → 写 1000 字小说）的采样/延迟/token 统计，供 serve 稳定性与 llama.cpp 对比 |
| `serve_stability_check.py` / `serve_mtp_check.py` | serve 复读/think-budget 实机检查 / MTP 开关逐 token 对拍（见 `docs/SERVE-STABILITY-FIXES.md`） |
| `monitor_session.py` | 被动监控 llama.cpp server 的 GPU/功耗/槽位与 prefill/decode 速率，输出 CSV |
| `tests/test_launcher.py` | launcher 80 项 GUI/参数/启动停止验收（`MODEL_DIR=... python tests/test_launcher.py`，需 GPU） |

这些脚本都是诊断/验收用，**不参与 `build.sh`**；`verify.sh` 只跑其中的 serve guard 单测
（`tests/test_serve_guard.cpp`）。
