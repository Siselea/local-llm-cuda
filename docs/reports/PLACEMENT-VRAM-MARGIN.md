# 自动分层显存余量修复（PLACEMENT-VRAM-MARGIN）

> 作者：prefill-fast｜日期：2026-11-21｜改动：`src/runner.cpp`、`src/runner.h`、`src/config.cpp`（+ 三个标定/验收脚本）
> 起因：`verify_generality` 的 `post-load margin enforced` 断言间歇性失败。查下去不是测试 flaky，而是**用户侧会真实遇到的"启动后 3 tok/s"**：自动分层（auto placement）没有为"引擎账本看不见的显存"预留余量，装载后可能只剩 0 MiB，驱动开始换页。

---

## 1. 现象与取证

同一条命令、同一二进制：

| 场景 | `gpu_moe_layers` | 装载后 free | 结果 |
|---|---|---|---|
| 刚跑完两次 `--vram-margin` 600/2400 后立即执行 | 15 | 2784 MiB | check OK（PASS） |
| 空闲 3 s 后执行 | 17 | **0 MiB** | WARNING（FAIL） |
| 再执行一次 | 14 | 3296 MiB | check OK（PASS） |

即层数在 14/15/17 之间摇摆，装载后余量在 0–3.3 GiB 之间摇摆。**根因有两层**：

1. **自动分层用的是"启动瞬间的 free VRAM"**（`budget = cuda::mem_info().free_bytes`），多进程/驱动释放时序会让它每次不同；
2. **`planner`/分层的显存估算没有计入"引擎账本看不见的分配"**：`alloc_sum`（引擎自己记录的所有 `cuda::Buffer` 之和）与设备实际占用之间始终有 **1.1–2.1 GiB** 的缺口——CUDA 上下文、cuBLAS workspace、分配器 padding，都不在账本里。估算按"free − 固定部分 − margin"选层，缺口就把 margin 吃掉了。

---

## 2. 修复

### 2.1 新增可标定的预留项 `vram_extra_reserve_mb`

- 默认 **1024 MiB**（env `ENGINE_VRAM_EXTRA_RESERVE_MB`，flag `--vram-extra-reserve`，0 = 恢复旧行为）；
- 在**自动分层的初始预算**里扣除；
- 在**逐层自适应检查**（`vram_layer_fits`，装载每一层前用**实测** free VRAM 判定）的 `need` 里加上——这是真正的"先算后分配"执行点：它在上传该层权重**之前**判定，所以不需要"装载后再重载"；
- `[vram] check OK` 现在会打印 **ledger gap**（`设备 used − 引擎账本`）与当前 reserve，后续标定可直接从日志读；
- 装载后余量仍不足时，输出从 `WARNING` 升级为 **`[vram] ERROR`**，并给出明确修法（`--gpu-layers` / `--vram-margin` / `--vram-extra-reserve`）。同时删掉了原来"halve rows"的无效代码（激活缓冲此时早已按真实余量分配，改 `rows_` 不会释放任何显存）。

### 2.2 标定数据（独占 GPU，每项两次结果完全一致）

| 模型 | ledger gap | reserve=1024 时装载后 free | 层数 |
|---|---|---|---|
| Coder-V2-Lite（MoE，原 flaky 案例） | **2097 MiB** | **2135 MiB ≥ 1200 margin ✓** | 18–19 |
| Qwen3.6-35B-A3B（MoE） | **1130 MiB** | **2429 MiB ✓** | 13 |
| Qwen3.5-9B（dense） | 1254 MiB | 6176 MiB ✓ | —（不涉及自动分层） |

### 2.3 reserve 取值曲线（为什么是 1024 而不是 2048）

| reserve | Coder 层数/free/解码 | 35B 层数/free/解码 |
|---|---|---|
| 0 | 20 / 1623 / 37.47 ms | 15 / 1405 / 32.01 ms |
| 512 | 19 / 2135 / 37.56 ms | 14 / 1917 / 32.72 ms |
| **1024** | **18 / 2135 / 37.68 ms（+0.6%）** | **13 / 2429 / 32.77 ms（+2.4%）** |
| 1536 | 17 / 2776 / 39.19 ms | 12 / 3070 / 33.13 ms |
| 2048 | 15 / 3800 / 40.88 ms（+9.1%） | 11 / 3582 / 34.09 ms（+6.5%） |

每多 1 MiB reserve 都可能挤掉一整个 MoE 层（~400–500 MiB，解码 −5~10%），所以取 **1024**：装载后 free ≈ 2.1–2.4 GiB（margin 1200 + ~1 GiB 余量），解码代价只有 **0.6–2.4%**；9B 这类 dense 模型完全不受影响（21.3–21.4 ms/token 不变）。

---

## 3. 验收（HEAD 上的最终二进制，独占 GPU）

| 项 | 结果 |
|---|---|
| `verify_generality.sh` **连续 3 次** | **3/3 GREEN，每次 26 passed / 0 failed，且 0 次触发 retry**（修复生效，不再依赖 `has_retry`） |
| `gen_regression.sh` | **GREEN 10/10** |
| `smoke_models.sh` | **exit 0**：40716 / 185 / 2614 / 2614（四模型 top-1 全对） |
| 9B decode 不回退 | 21.31 → 21.36 ms/token（噪声内） |
| Coder / 35B decode | +0.6% / +2.4%（见 2.3，已是最小代价点） |

标定与验收脚本：`tools/vram_ledger_probe.sh`（测量 ledger gap）、`tools/vram_reserve_curve.sh`（reserve 曲线）、`tools/placement_fix_accept.sh`（3×generality + gen_regression + smoke + 解码 A/B）。

---

## 4. 边界与后续

1. **并发占用无法完全防住**：预算用的是**启动瞬间**的设备级 free VRAM；如果另一个进程在本进程装载期间抢走显存，仍可能低于 margin。此时现在会打印 `[vram] ERROR` + 明确修法（而不是静默换页）。彻底解法是"装载后真减层 + 重载"，改动较大，建议排进后续（需要动 loader 的层上传循环，或在上层做 retry 包装）。
2. **`planner.cpp` 的 `engine plan` 建议值**尚未加同样的预留项（该路径只用于展示建议，不影响真实分层）。若要口径一致，可在 `planner::compute` 里加同一常数。
3. 相关：`has_retry`（`verify_generality.sh`）保留为诊断兜底，但验收证明它已不是必需。

---

## 2.4 2026-11-27 更新：ledger gap 在 WDDM 下不是常数（serve 侧预算缺口）

v0.1.9/v0.1.10 复测 35B（auto 放置 + `engine serve`）时发现：**同一条命令、同一个二进制**，
装载后 loader 的 `[vram] check OK` 与 serve 守卫之间还会再少 **1.5–3 GiB**，且 `ledger gap`
本身在 1604 → 3230 MiB 之间波动：

```
[vram] check OK: 2035 MiB free >= 1200 margin (gpu_moe_layers=11) [ledger gap 3230 MiB]
[serve] ERROR: only 487 MiB VRAM free after load, below 1200 MiB margin   # Δ=1548 MiB
```

此前本节把 gap 当作可标定常数（1024/1152）。**新结论：在桌面 WDDM 共享显存活跃的机器上，
gap 包含“桌面异步增长 + CUDA 上下文/cuBLAS/图池”两部分，是可变量**，因此：

1. **serve 模式为 MoE 部分卸载模型额外预留** `ENGINE_SERVE_RESERVE_MB`（默认 **2048 MiB**，
   在 placement / 逐层自适应 / post-load check 中扣除）：让 placement 少放 1–2 层，而不是装载后
   serve 守卫失败。dense 模型（9B/0.5B）不受影响。
2. **serve 守卫改为两段式**：free ∈ [700 MiB hard floor, margin) ⇒ WARN 后启动（提示环境原因与
   修法）；free < 700 ⇒ 拒绝。`--strict-vram` 恢复旧行为（free < margin 直接拒绝）。
3. **标定建议**：`vram_extra_reserve_mb=1024` 仍是引擎自身账本外的驱动/allocator 预留；
   但“桌面 WDDM 波动”不应塞进这个全局标定（会把 9B 的 ctx 上限也压低），而应由
   `ENGINE_SERVE_RESERVE_MB` + 警告带处理。
4. 实测验收（2026-11-27，clean window）：35B serve auto ctx4096/8192 在 reserve=1280 时已能
   listening（serve free 2364/1919 MiB ≥ margin 1200）；reserve=2048 + 警告带后，显式
   `--gpu-layers 13` 在抖动条件下也不再失败。数据见 `docs/reports/CTX-AND-EXTREMES.md` §2.16/§2.17。
