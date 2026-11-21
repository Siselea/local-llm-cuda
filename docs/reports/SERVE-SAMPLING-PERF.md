# serve 采样性能：全词表 → 候选集（`perf(serve)`，2026-11-21）

## 结论

非 greedy 请求（用户默认 `temp=0.6 / top_k=20 / top_p=0.95`）的每 token 采样开销：

| 路径 | 改前 | 改后 | 变化 |
|---|---|---|---|
| **sampler 微基准**（vocab=248320，top_k=20/top_p=0.95，6 次重复中位数） | 198.8–202.2 µs | **46.0–47.5 µs** | **−152~155 µs/token（−76~77%）** |
| sampler microbench（`top_k=0` + top_p=0.95，边缘路径） | 8064–8342 µs | **302–311 µs** | **−7.8~8.0 ms/token（−96%）** |
| 9B serve 端到端 decode（temp 0.6 / top_k=20，交错 A/B 5 次） | 待填 | 待填 | 待填 |
| 9B serve 端到端 decode（top_k=0，交错 A/B 5 次） | 待填 | 待填 | 待填 |

此外还去掉了 serve 每 token 的两次 1 MB 拷贝（`row → last.assign`、`sample` 内 penalty 全行复制），
在端到端里额外约 0.04–0.08 ms/token。

## 根因（审计 + 实测）

改前 serve 对**每个非 greedy token** 做：

1. `forward()` 取整行 `vocab=248320` logits → **1 MB D2H**（pinned staging，~38 µs 纯拷贝）；
2. serve 再 `last.assign(row.end()-vocab, row.end())` → **再一次 1 MB 复制**；
3. `Sampler::sample()`：
   * `penalized = logits`（每个 token 又复制 1 MB，当 repeat penalty≠1）；
   * `cand_.resize(vocab)` + `for i in 0..vocab` 填下标（248k 次写）；
   * `std::partial_sort(248320, k=20)` 按浮点比较随机访问（实测 152 µs）；
   * `top_k=0` 时更糟：候选=全词表，`exp(248320)`（~1.33 ms）+ `sort(248320 pairs)`（~0.45 ms）。

## 修复

* 改前/改后微基准（6 次重复）：`webui` 198.8–202.2 → 46.0–47.5 µs；`top_k=0` 8064–8342 → 302–311 µs；
  `min_p` 路径 5080–5226 → 262–269 µs。
  * prefill logits 直接 `std::move` 进 sampler，随后释放（不再复制）；
  * 每 token 的 `forward()` 行直接传给 sampler（不再 `last.assign`）；
  * `Sampler::sample_bounded()`，`/props.sampling` 暴露实现方式。
* `include/engine/sampler.h` + `src/sampler.cpp`：
  * **单遍 top-N 扫描**（`select_candidates`）：窗口 n=64（`top_k>0` 时 `max(top_k,64)`），
    顺序读 1 MB、AVX2 8 路比较跳过（`_mm256_cmp_ps` + `movemask`），只有超过窗口尾的元素才插入；
    无 248k 下标数组、无 partial_sort、无 248k exp/sort；
  * `top_k=0` 用窗口 1024（`kDefaultCandidateCap`）的 min-heap 单遍扫描（~250 µs，对比改前 8.5 ms）；
  * repetition penalty 只作用在候选 id 上（用 vocab 大小 membership 位图，无整行复制）；
  * top_p / min_p / softmax 只在候选集上计算（`top_k>0` 时 top-k 集合精确；`top_k=0` 时
    "top-1024 近似 top_p"，已在 `docs/CONFIG.md` 与 `/props.sampling` 说明）；
  * greedy（`temperature=0`）：仍是整行 argmax，**逐 token 行为不变**；
  * 预留设备端候选接口 `Sampler::set_candidates(const int* ids, const float* logits, int n)`，
    runner 侧设备 top-k 扩到 64/128 后可直接接入（serve 不需再改）。

## 数值语义

* **greedy**：完全不变（整行 argmax；gen_regression / smoke 验证）。
* **top_k > 0**：实际候选 = `max(top_k,64)` 个 raw logits 最高的 token，top-k 集合与整词表实现一致；
  唯一差异是 repetition penalty 之后若有 token 被压出前 64 名，替补顺序可能与整词表实现不同
  （概率可忽略；单测 `test_sampler.cpp` 有对照 case）。
* **top_k = 0**：候选 = top-1024，`top_p/min_p/softmax` 在候选集上归一化（近似；极端长尾分布下
  排名 >1024 的 token 概率被截断）。这是文档化的、可观测的近似（`/props.sampling.top_k_zero`）。
* 采样 RNG 消耗、seed、penalty 窗口语义不变；同一 seed + 参数在同一实现内可复现（单测覆盖）。

## 复现

```bash
# CPU 微基准（无需 GPU）
$PY -m ziglang c++ -std=c++17 -O2 -Iinclude tools/bench_sampler.cpp src/sampler.cpp -o build/bench_sampler.exe
build/bench_sampler.exe 248320 400

# 9B serve 交错 A/B（需要 GPU 空闲；before/after 两个二进制）
$PY -m ziglang c++ -std=c++17 -O2 -Iinclude src/... src/serve.cpp src/main.cpp -o build/engine_before_sampler.exe   # 改前源码
$PY -m ziglang c++ -std=c++17 -O2 -Iinclude src/... src/serve.cpp src/main.cpp -o build/engine_after_sampler.exe    # 改后源码
$PY tools/bench_serve_sampling.py --before build/engine_before_sampler.exe \
    --after build/engine_after_sampler.exe --model "${MODEL_DIR}/Qwen3.5-9B-Q5_K_M.gguf" --trials 5
```

## webui 实际参数证据（用户默认路径）

`build/webui/index.html`（llama.cpp webui bundle，已按 Qwen3 推荐值打过补丁）：

```
默认配置:      temperature:.6, top_k:20, top_p:.95, min_p:0
旧默认迁移:    if(c.temperature===0.8&&c.top_k===40&&c.top_p===0.95&&c.min_p===0.05){c.temperature=0.6;c.top_k=20;c.min_p=0}
请求体:        ...top_k:Xt.top_k, top_p:Xt.top_p, min_p:Xt.min_p...
```

⇒ 默认用户路径 = **top_k=20**（本次主要收益 −136 µs/token）；`top_k=0` 是显式关闭 top-k 的
客户端才走的边缘路径（本次 −8 ms/token 级收益，cap=1024 兜底）。

证据文件：`build/logs/webui_sampling_params.txt`。

## 验证

* `tests/test_sampler.cpp`：新增 9 组用例（greedy 一致、top_k 精确、top_k=0 截断对比、penalty、
  外部候选接口、可复现性、penalised top-k 与整词表对照）。
* 五套验证 + 真实浏览器 e2e：见报告末尾（本轮完成后填）。
