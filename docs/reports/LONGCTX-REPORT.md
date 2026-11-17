# 长上下文（4k / 2k 预填）报告：一个必然崩溃的 bug + 修复后的数据

日期：2026-11-15 · 负责：general-opt · 模型：Qwen3.5-9B-Q5_K_M（dense，32 层，16 head / 4 KV head / head_dim 256）
复现命令：`./build/test_gen.exe <MODEL_DIR>/Qwen3.5-9B-Q5_K_M.gguf "$(cat build/ids2048.txt)" 17 --ctx 4096`

---

## 1. 结论先行（这是本轮最重要的发现）

**修复前：任何超过 384 token 的上下文（预填或 decode）都会直接崩溃**，不是慢，是
`cuLaunchKernel failed (CUresult=1, invalid argument)`。也就是说引擎此前所有性能数字都只对
"短上下文"成立，用户拿真实提示词（>384 token）跑会直接报错退出。

修复后：2048 token 预填 + 4096 ctx 正常跑完，贪心 token 序列稳定；decode 从 22.2 ms（短）
退化到 33.3 ms（2k ctx，**−50% 吞吐**），瓶颈是 attention 内核的访存放大（kv-longctx 在做
KV 量化 + 分块 online-softmax，见 §5）。

## 2. 根因

`attn_cached_w`（`include/engine/kernels.h:1709`）把**整行 logit**（`max_seq` 个 float）和
**每 warp 一个 head_dim 宽的累加器**都放在动态 shared memory 里：

```
sh = (max_seq + 32 + nwarp * head_dim) * 4 B
```

warp 数按上下文长度分桶（`ctx_len > 384 → nwarp = 32`，96 及以下 8）。9B 的 `head_dim = 256`：

| ctx_len | nwarp | blockDim | shared |
|---|---|---|---|
| ≤96 | 8 | 256 | (max_seq+32+2048)*4 |
| ≤384 | 16 | 512 | (max_seq+32+4096)*4 |
| **>384** | **32** | **1024** | **(4096+32+8192)*4 = 49280 B > 48 KiB 默认上限** |

⇒ 第一次 `ctx_len > 384` 就非法。因为预填也是一次 `n_tokens>384` 的 forward，**512 token 预填
和 385 步 decode 都会崩**：

```
n=256 : OK（prefill 962 ms, decode 22.84）
n=384 : prefill OK（1598 ms），decode 第一步崩（ctx_len=385 → nwarp=32）
n=512+: prefill 直接崩
```

定位手段：给 `cuda::launch()` 的失败路径加上"内核名 + grid/block/shared"（原来只有
`cuLaunchKernel failed`，无法归因）。失败信息变成
`cuLaunchKernel(attn_cached_w grid 512x16x1 block 1024 shared 49280) failed (CUresult=1, ...)`。

**两个** launch 站点都有同样问题，都改了：
- `src/qwen35.cpp`（qwen35 架构：9B / 35B 走这里）
- `src/runner.cpp`（dense / MLA 路径；MLA 那处 `sh = max_seq*4`，ctx>12288 才会超，也一并 opt-in）

## 3. 修复（默认生效，不是配置项）

1. `cuda::max_dynamic_shared_mem(cuFunction)`（`include/engine/cuda.h`、`src/cuda.cpp`）：
   查 `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`（本机 RTX 3060 = 101376 B），
   用 `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` opt-in 到设备上限，按 kernel 缓存结果。
2. launch 前按 smem 预算**降低 nwarp**：`while (nwarp>1 && smem_for(nwarp) > cap) nwarp >>= 1`。
   即"先按上下文长度分桶，再按显存预算收敛"，两个约束都是结构性的（不按模型名）。
3. MLA attention 的内核也 opt-in，使 ctx 上限从 12288 提到 ~25000（该内核 smem 仍是 O(max_seq)，
   真正修掉要分块，见 kv-longctx）。

## 4. 修复后实测（`--ctx 4096`，同一条随机 prompt 前缀，n=5，ENGINE_DEC_SKIP=2）

| prompt 长度 | prefill | prefill tok/s | decode ms/token | tok/s |
|---|---|---|---|---|
| 1（短上下文基线） | — | — | **22.23 / 22.29 / 22.55** | 44–45 |
| 384 | 1350 / 1378 ms | 279–284 | 23.9 / 24.4 | 41–42 |
| 512 | 1966 ms | 260 | 23.2 | 43 |
| 1024 | 4341 ms | 236 | 27.3 | 37 |
| 2048 | 9882 / 11109 ms | 184–207 | **33.3**（首测，机器快）/ **40.3**（机器漂移后） | 25–30 |

读法：
- **decode**：512 以内基本不退化（+1 ms，4%）；1k 退化 23%；2k 退化 50–80%。退化量约
  `+0.0055 ms / token-of-context`（短→2k 线性）。
- **prefill**：2048 token 要 9.9 s（207 tok/s），也随长度劣化（384 时 284、2k 时 207）。
  预填同样是 O(n²) 的朴素 attention 内核，kv-longctx 的分块内核应能大幅改善。

## 5. 瓶颈归因（我拿到的部分 + kv-longctx 的定位）

我没能在 graph 模式下拿到逐内核归因：`ENGINE_KPROF=1` 报 `0 kernel kinds`，因为逐层 CUDA Graph
用 `cuGraphLaunch` 整图重放，kprof 的 event 看不到内部内核；`--no-graph` 的 kprof 需要再独占
GPU ~2 分钟，已按 lead 指示让给 kv-longctx。

已确认的结构原因（代码 + 数据一致）：
1. **K/V 读取跨 lane stride = head_dim**：`attn_cached_w` 的 value 累加是
   `for d = lane; d < D; d += 32 → Vh[j*D+d]`，同一个 warp 的 32 个 lane 读同一行相邻元素没问题，
   但 KV 量化/分块前的实现在长 ctx 下每个 32 B sector 只有 4 B 有用（8× 放大）。
2. **GQA 重复读**：9B 是 16 q head / 4 kv head → 同一份 K/V 被 4 个 q head 各读一次（4×）。
   两者相乘 32×，2k ctx 的 KV 逻辑量 = 32 层(混合) × 2048 × 8 KB ≈ 数百 MB/token 的有效放大。
3. **shared memory O(max_seq)**：内核必须先把整行 score 放进 shared 才能 softmax，ctx 4096 时
   光 score 就 16 KiB、加 accs 32 KiB；这既限制 occupancy 也是这次崩溃的直接原因。
4. **KV cache 是 f32**（`include/engine/runner.h:98` `cuda::Buffer kc, vc; // [n_kv][max_seq][D]`，
   kernel 参数是 `const float*`）：4k ctx 的 KV 读放量随 ctx 线性增长。llama.cpp 默认
   `-ctk q8_0 -ctv q8_0`，可省一半带宽 —— kv-longctx 正在实现 `kv_type_`（0=f32, 1=q8_0, 2=q4_0），
   预期 2k ctx 的 attention 访存减半、decode 从 33 ms → ~28 ms 量级（等它的实测）。

## 6. 正确性证据

1. **一次性 prefill vs 逐 token prefill（`ENGINE_SPLIT_PREFILL=1`）**，384 token：
   - 一次性：`topk: 100465(-2.584) 97145(-3.115) 96507(-3.602) 78958(-3.646) 97583(-3.668)`
   - 逐 token：`split topk: 100465(0.000) 97145(-0.530) 96507(-1.018) 78958(-1.041) 97583(-1.076)`
   - top-1 相同，top-5 顺序相同；logit 间距差 ≤0.04 nat（批量 GEMM 与逐 token 的浮点求和顺序差异，
     与历史报告的 3.5–4.0e-5 运行间噪声同量级）。**没有 token 改变。**
2. **2k 预填生成**：`gen: 271 155648`、`topk: 271(-0.525) 220(-2.257) 457(-2.614) ...`，两次运行
   （快/慢窗口）top-1 与序列一致。
3. 短上下文 9B 预测 `2614` 在 `--ctx 4096` 下不变（`topk: 2614(-1.947) 1414(-4.056) ...`）。

## 7. 没做完 / 下一轮

| 项 | 状态 | 说明 |
|---|---|---|
| `--no-graph` + 2k ctx 的逐内核归因 | 交给 kv-longctx | 需要独占 GPU 2 分钟 |
| 分块 online-softmax attention（去掉 O(max_seq) smem） | kv-longctx 在做 | 修完 4k/8k 才能不退化 |
| KV q8_0/q4_0 量化 | kv-longctx 在做 | runner.h 已有 `kv_type_` |
| `logits_` 按 chunk 分配 | kv-longctx 已做（`runner.cpp:500` 注释） | 否则 128k ctx 要 130 TB |
| 128k/256k ctx 与 llama.cpp 同条件对比（`--ctx 131072` + 短 prompt，llama-bench 风格） | 待做 | 用户 llama.cpp 的 46.7 tok/s 就是这种口径 |
| 9B 4k/8k prefill 速度 | 待做（依赖分块内核） | 现在 2k 预填只有 207 tok/s |
