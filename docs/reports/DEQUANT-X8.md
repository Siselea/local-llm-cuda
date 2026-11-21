# DEQUANT-X8：反量化内核提速（2026-11-21）

> 作者：prefill-fast｜改动：`include/engine/kernels.h`（新增 `dequant_q*k_f16_x8`）、`include/engine/runner.h`、`src/runner.cpp`（prefill f16 路径默认改用 x8，`ENGINE_DQ_OLD=1` 可回退）
> 目标（lead 指派）：把单次整模型反量化从 ~95 ms 压到 50–60 ms（115 GB/s → 180–230 GB/s）。
> 结果：**达到目标**——微基准 181–207 GB/s → **~305 GB/s（1.48–1.70×）**，整模型外推 **~115 ms → ~60 ms**；**fp16 slab 与旧内核逐字节一致（memcmp）**，因此模型输出不可能改变。

---

## 1. 根因：每 2 个 store 一个 CUDA block

旧的 `dequant_q5k_f16` 等内核：**一个 super-block（256 值 / 176 B）一个 128 线程的 CUDA block**，每个线程只做 2 个元素 + 1 个 4 B store。9B 的 6.1 GB 量化权重 = **34.7M 个 super-block = 34.7M 个 block**，每个 block 的有效工作只有 512 B 输出 ⇒ **block 调度/发射开销主导**，实测只有 181–207 GB/s（HBM 331 GB/s 的 55–63%）。

## 2. 改动：一个 warp 一个 super-block，向量化 16 B store

`dequant_q{4k,5k,6k}_f16_x8`：
- **8 个 super-block / CUDA block**（256 线程 = 8 warps，每 warp 一个 super-block）⇒ block 数 ÷8；
- 每 warp 的 32 条 lane 各处理 **8 个连续元素**：这 8 个元素必在同一个 32 值子块内 ⇒ **每 lane 只需解一次 (scale, min)**（旧的每元素调一次带分支的 `q4k_scale_min`）；
- 量化字节是 8+8 个连续字节（`qp[i]`、`qh[l0+i]`），寄存器里做 nibble/high-bit 组合；
- 每 lane 一次 **16 B `uint4` store**（旧的 4 B）。
- **算术顺序与旧内核完全相同**（`scale * (float)q - bias`，再 `cvt.rn.f16.f32`）⇒ slab 逐字节一致。

## 3. 微基准（`test_gen --bench-dequant`，合成合法数据，min-of-5 event 计时）

| 类型（12288×4096 单矩阵） | 旧（1 block/super-block） | **x8（1 warp/super-block）** | 加速 | slab 一致性 |
|---|---|---|---|---|
| Q4_K（28.3 MB 量化 + 100.7 MB fp16） | 0.678 ms（181 GB/s） | **0.399 ms（308 GB/s）** | **1.70×** | **逐字节一致 ✓** |
| Q5_K（28.3 MB + 100.7 MB） | 0.698 ms（185 GB/s） | **0.426 ms（303 GB/s）** | **1.64×** | **逐字节一致 ✓** |
| Q6_K（33.0 MB + 100.7 MB） | 0.653 ms（207 GB/s） | **0.442 ms（306 GB/s）** | **1.48×** | **逐字节一致 ✓** |

外推整模型 6.1 GB 量化权重（读 6.1 GB + 写 12.2 GB fp16 = 18.3 GB）：旧 ≈ **115 ms** → x8 ≈ **60 ms**。

## 4. 端到端验证（HEAD `62cb37f` + 本改动；独占 GPU）

### 4.1 真实 prefill A/B（9B，ctx 32768，prefix cache 关，rows 自动 512→1024，**交错 old/new × 2 轮**）

| prompt | 旧内核（r1/r2） | **x8（r1/r2）** | 耗时变化 |
|---|---|---|---|
| 4394 tok | 4113 / 4041 ms（1068 / 1087 tok/s） | **3877 / 3918 ms（1133 / 1121 tok/s）** | **−179 ms（−4.4%）** |
| 13364 tok | 17184 / 17196 ms（778 / 777 tok/s） | **16732 / 16766 ms（799 / 797 tok/s）** | **−441 ms（−2.6%）** |

（同一窗口内被早前并发测试污染的两行数据已丢弃；A/B 为交错执行，两侧同受环境漂移影响。
绝对 tok/s 在不同窗口间会随机器温度/负载漂移 ~10%，以交错差值看收益。）

### 4.2 五套验证（同一二进制）

| 套件 | 结果 |
|---|---|
| `verify.sh` | **ALL GREEN** |
| `smoke_models.sh` | **exit 0**：40716 / 185 / 2614 / 2614 |
| `gen_regression.sh` | **GREEN 10/10** |
| `verify_generality.sh` | **GREEN 26/26** |
| `verify_serve.sh` | **GREEN 22/22** |

### 4.3 输出不变性

- 微基准：三种类型的 fp16 slab 与旧内核 **memcmp 逐字节一致**；
- 端到端：9B `gen: 2614 513 5081` 在 x8 与 `ENGINE_DQ_OLD=1` 下 **完全相同** ⇒ 该路径 **token 级严格不变**（不是"在噪声内"，是真的同一位）。

## 5. 预期收益（按整模型反量化 ~115→~60–72 ms 推算；4.1 已实测到 −2.6~−4.4%）

| 场景 | 每 forward 固定成本变化 | 端到端 |
|---|---|---|
| tail（52 tok，1 chunk） | −35~45 ms | ~190 ms → **~150 ms** |
| 4.4k 文档（5 chunk @1024） | 5 × −40 ms | 3.69 s → **~3.50 s**（−5%） |
| 13.4k 文档（14 chunk @1024） | 14 × −40 ms | 15.97 s → **~15.4 s**（−4%） |
| 长文档首 token | 同 13.4k | 同上 |

（lead 备注：反量化变快后，层主序省下的"每 chunk 一次 dequant"也等比缩小，所以先做 x8 再做层主序的顺序是对的。）

## 6. 复现

```bash
bash build.sh
./build/test_gen --bench-dequant                 # 微基准（旧 vs x8，含逐字节一致性）
./build/test_gen <model> 760 3 --ctx 512         # 默认走 x8
ENGINE_DQ_OLD=1 ./build/test_gen <model> 760 3 --ctx 512   # 回退旧内核（A/B）
```
