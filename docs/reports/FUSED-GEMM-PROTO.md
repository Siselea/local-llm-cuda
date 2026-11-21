# FUSED-GEMM-PROTO：fused dequant+GEMM 最小原型（2026-11-21）

> 作者：prefill-fast｜性质：**原型实验报告（负结果 + 修正后的判断）**
> 背景：`LAYER-MAJOR-PREFILL-PLAN.md` §3 曾判断"tail 固定成本 ≈ 130 ms 主要是显存流量（dequant 读 6.1 GB + 写 fp16 镜像 12.2 GB + cuBLAS 再读 12.2 GB ≈ 30 GB ≈ 91 ms），fused（只读 6.1 GB）可把 tail 从 190 ms 压到 ~85 ms"。本文件是那个判断的**实测检验**，结论需要修正。

---

## 1. 原型做了什么

`tests/fused_gemm_proto.h`（test-only，`test_gen --bench-fused-gemm`）：

1. **baseline**：产品路径 = `dequant_q5k_f16`（Q5_K → fp16 slab）+ `cublas::gemm_f16_nt_ld`；
2. **fused**：单内核 `proto_gemm_q5k_f16_wmma`——block 128 线程/4 warps，tile 16(M)×64(N)；K 按 256 宽的 Q5_K super-block 推进，x tile 与**反量化后的权重 tile** 落 shared memory，然后用 `mma.sync.m16n8k16`（fp16 输入 / f32 累加）做 GEMM。只从 DRAM 读量化权重。
3. 合成数据（合法 Q5_K 结构，d=0.0625/dmin=0.03125），4 组形状，min-of-5 event 计时，并与 baseline 对比 rel_l2。

## 2. 实测（RTX 3060 12GB，sm_86）

| 形状 | baseline = dequant + cuBLAS | fused（naive） | rel_l2 |
|---|---|---|---|
| M512 N8192 K2048 | **0.805 ms**（dq 0.306 + gemm 0.499） | 41.8 ms（**0.02×**） | 1.4e-06 |
| M512 N2048 K8192 | **0.754 ms**（dq 0.283 + gemm 0.471） | 50.1 ms（0.02×） | 7.2e-06 |
| M512 N2048 K2048 | **0.251 ms**（dq 0.078 + gemm 0.173） | 10.3 ms（0.02×） | 1.8e-06 |
| M16 N8192 K2048（tail 形） | **0.401 ms**（dq 0.287 + gemm 0.115） | 1.91 ms（0.21×） | 1.3e-06 |

**数值上完全正确**（rel_l2 ≤ 7.2e-6，max|diff| ~1e-2，属 fp16 输入 + 不同累加顺序的正常量级）——说明内核布局（A/B fragment、mma PTX、shared staging）是对的。

**性能上慢 50 倍**，原因不是"流量假设错了"而是实现层的结构性代价：
- **反量化被重复 M/tile_M 次**：M=512、tile_M=16 ⇒ 每个权重元素被反量化 32 次（baseline 只 1 次）；
- 4 warps/block、无 double buffering、单缓冲 shared，mma 与反量化串行；
- 反量化本身是标量公式（分支 + 每元素索引），不是向量友好。

## 3. 更关键的修正：dequant 是**计算**瓶颈，不是流量瓶颈

| 量 | 实测/推算 |
|---|---|
| 产品 dequant 内核吞吐（合成数据） | 11.5 MB slab / 0.30 ms ≈ **115 GB/s**（HBM 331 GB/s 的 ~35%） |
| 向量化变体（8 halves/线程、16 B store） | 0.93×——**没有变快**，说明不是 store 带宽问题 |
| 全模型 6.1 GB 权重的 dequant 计算 | ≈ 95 ms（与早前实测一致） |
| cuBLAS 读 fp16 镜像（12.2 GB） | ≈ 37 ms |

⇒ 每个 forward 的 ~130 ms 固定成本里，**约 95 ms 是反量化这个计算本身**，fused 内核同样要付它（一个元素都省不掉）。它能省的只是"写 12.2 GB 镜像 + cuBLAS 再读 12.2 GB"≈ 74 ms 的一部分，而且：

- **大 M（全量 prefill）**：baseline 0.805 ms 已经接近算力下限（8.6 G-MAC / 12.8 T-MAC/s = 0.67 ms）⇒ fused 的理论最好情况 ≈ 0.7 ms，**最多 1.1–1.2×**，不值得重写；
- **小 M（tail，52 token）**：baseline 0.401 ms 里 gemm 只占 0.115 ms ⇒ fused 去掉镜像写读后可到 ~0.29 ms，**约 1.3–1.4×**——即 tail 固定成本 130 ms → ~100 ms，不是之前估的 85 ms。

**结论：`LAYER-MAJOR-PREFILL-PLAN.md` §3 的"tail 190→~85 ms"高估了**（把 dequant 计算误当成可被 fused 消掉）。修正后的预期是 **tail ~190→~140–150 ms（约 1.3×）**，且需要把 fused 内核做到"大 M tile + double buffer + 反量化向量化"才有；naive 版就是 50× 慢。

## 4. 修正后的优先级建议

| 方案 | 预期收益 | 成本 | 建议 |
|---|---|---|---|
| **层主序 prefill**（已设计，§2） | 长文档 **−8~−10%**（13.4k：−1.2 s）；tail 0 | 2–3 h，chunk 边界不变可保持 token 级一致 | **先做**（收益确定、实现清晰） |
| **反量化内核提速**（无分支/预计算 scale-bias/多元素/线程） | 若 95 ms → 50–60 ms，则 tail 固定成本 −30~40 ms（~1.2×），长文档也受益 | 2–4 h，局限在 `dequant_q*k_f16` | 值得做（与层主序互补） |
| **fused dequant+GEMM（调优版）** | tail ~1.3×；大 M 无收益 | 数天（大 tile/持久权重/double buffer），且要重新验证数值与 cache 交互 | **降级**：等前两项做完、且反量化提速后固定成本结构变化再评估 |
| cache 对齐解耦（server-mode 已完成） | 长多轮每轮 −75% | 已完成（`adb3909`） | 完成 |

## 5. 复现

```bash
bash build.sh
./build/test_gen --bench-fused-gemm     # 需要 GPU（约 100 MB 显存，几秒）
```
原型代码在 `tests/fused_gemm_proto.h`，只在带 `--bench-fused-gemm` 时运行，不影响任何产品路径。
