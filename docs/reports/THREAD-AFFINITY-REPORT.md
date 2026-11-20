# 线程数 × SMT 绑核扫描报告（THREAD-AFFINITY）

**日期**：2026-11-17 · **机器**：i5-10400F（6 物理核 / 12 逻辑核，SMT 2/核）· 32 GB · Windows 11
**触发问题**（用户观察）：「1 个核心对应 2 个线程，应该用 2/4/6/8/12 双数线程；现在用 9（单数）
在优化层面是性能浪费。」
**状态（2026-11-17 23:10）**：默认值已实现（`win_affinity=2` SMT 成对绑核 + 默认线程数 2×物理核=12）
且全部源码编译通过（`engine_new.exe` + 全测试二进制；`build/engine.exe` 因 `server-mode` 长上下文
基准仍在运行被文件锁占用，待其结束后重跑 `build.sh` 即生效）。四套验证与端到端 A/B 待安静窗口。

**结论先说**：**用户方向对了一半，而且比"偶数"更关键的是"成对绑核"**——把线程数提到
**10/11/12 并用拓扑动态读到的 SMT 兄弟对做硬绑核**，比默认的 9 线程不绑核快 **~8–12%**
（min 0.709→0.623 ms/层；p25 0.733→0.675）；**只绑核不加线程（6/8 成对）反而更慢**，
12 线程**不**绑核也几乎没收益——所以不是"偶数"本身，是"成对 + 吃满 6 个物理核"。

---

## 1. 方法

- 工具：`tools/bench_cpu_layer.cpp layer 24 <iters>`（24 个独立 15.56 MiB DRAM 副本轮转，
  真 DRAM、非 L3，一个副本=35B 的 CPU 专家层形状）。
- 矩阵：线程数 **{4,6,7,8,9,10,11,12}** × 绑核 **{no-affinity, SMT 成对}**，共 16 配置；
  每个配置 **10 reps × 400 iters**，**每 rep 把所有配置的启动顺序旋转一格**（抵消分钟级漂移/争用）。
- SMT 成对：逻辑线程 2k / 2k+1 绑到物理核 k 的两个硬件兄弟，兄弟对**从拓扑动态读取**
  （`GetLogicalProcessorInformation(Ex)(RelationProcessorCore)` 的 ProcessorMask），不写死。
- 主进程/线程优先级沿用生产设置（HIGH + worker HIGHEST）。
- 结果：`build/thread-scan2.{json,csv}`（本轮主数据），`build/thread-scan.{json,csv}`（首轮 6 reps 复核）。
  工具：`tools/thread_scan.py`。

**重要校正（历史结论）**：本机 SMT 兄弟对实测是 **`(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)`**，
**不是常见的 `(i, i+6)`**。之前"绑核是噪声/无效"的结论，大概率是在错误的兄弟对假设下测的
（按 i,i+6 绑会把两个互不为兄弟的逻辑核配到一起，等于随机绑）。**本轮成对绑核 + 高线程数明显更快。**

---

## 2. 完整扫描表（单位：ms/层，越小越好；10 reps × 400 iters）

| 线程数 | no-affinity<br>min / p25 / median | **SMT 成对**<br>min / p25 / median | 成对 vs no-aff（min） |
|---|---|---|---|
| 4 | 1.005 / 1.142 / 1.156 | 1.366 / 1.391 / 1.423 | +36%（成对=只用 2 核，更慢） |
| 6 | 0.907 / 0.957 / 0.963 | 0.938 / 0.993 / 1.030 | +3%（成对=3 核） |
| 7 | 0.838 / 0.846 / 0.864 | 0.869 / 0.917 / 0.931 | +4%（成对=3.5 核） |
| 8 | 0.753 / 0.771 / 0.787 | 0.759 / 0.783 / 0.867 | +1%（成对=4 核） |
| **9（现默认）** | **0.709 / 0.733 / 0.762** | 0.701 / 0.726 / 0.766 | −1%（成对=4.5 核） |
| 10 | 0.701 / 0.723 / 0.740 | **0.677 / 0.683 / 0.758** | **−3%**（成对=5 核） |
| 11 | 0.662 / 0.687 / 0.759 | **0.644 / 0.675 / 0.719** | **−3%**（成对=5.5 核） |
| 12 | 0.709 / 0.798 / 0.816 | **0.623 / 0.681 / 0.747** | **−12%**（成对=6 核全双线程） |

按线程数看**最好的一列**（min / p25）：

| 配置 | min ms/层 | p25 ms/层 | GB/s（min） | vs 9 no-aff |
|---|---|---|---|---|
| **12 + 成对** | **0.623** | 0.681 | **26.2** | **−12.1%** |
| **11 + 成对** | 0.644 | **0.675** | 25.3 | −9.2% / **−7.9%（p25）** |
| 11 no-aff | 0.662 | 0.687 | 24.6 | −6.6% |
| 10 + 成对 | 0.677 | 0.683 | 24.1 | −4.5% |
| 10 no-aff | 0.701 | 0.723 | 23.3 | −1.1% |
| **9 no-aff（现默认）** | **0.709** | **0.733** | **23.0** | — |
| 12 no-aff | 0.709 | 0.798 | 23.0 | ≈0 |

**读法**：
- **11/12 + 成对、10 + 成对**是第一梯队，min 比现默认快 4.5–12%，p25 快 3.8–7.9%。
- **"只成对不加线程"是负优化**（4/6/8 成对都更慢）：成对在 N 线程时只用 N/2 个物理核，
  线程数必须提到 2×物理核（12）才把 6 个核全部吃满。
- **"只是偶数"也不够**：12 不绑核 0.709/0.798 还不如 9；成对才把 12 的潜力（26.2 GB/s）放出来。
- 与物理预期一致：6 核每核 2 线程（SMT 额外 ~30% 吞吐）理论上限 ≈ 9 线程（6 核 + 3 兄弟）
  的 1.1–1.3 倍，实测 9→12 成对 22.9→26.2 GB/s（+14%）与此吻合。

---

## 3. 已落地的默认值（源码改完、编译通过；`engine.exe` 待服务器停止后重链接）

1. **默认 = SMT 成对绑核 + 偶数线程 = 2×物理核（本机 12）**；保守可选 10（5 核成对，
   收益 4–8%、SMT 争用更小，给 OS/WDDM 留 2 个核）。11/pair 与 12/pair 统计并列，取 12 最符合
   "偶数成对 + 吃满物理核"。**端到端验证前先以 12 为默认**；若端到端显示 12 反而慢（哪怕 2%），
   按预案改成 10/pair。
2. **实现（已写入源码）**：
   - `cpu::Pool` 创建时用 **`GetLogicalProcessorInformationEx(RelationProcessorCore)`** 动态读兄弟对
     （fallback 旧 `GetLogicalProcessorInformation`），逻辑线程 2k/2k+1 → 物理核 k 的两个兄弟，
     主线程=逻辑线程 0；`--threads N` 为奇数时 = k 对 + 1 个单绑；
   - 配置键沿用 **`win_affinity`（0..2）**，**默认改为 `2`**（SMT 成对）；`0` = 关闭（真正的旧行为，
     线程数也回到 1.5×物理核=9），`1` = 物理核优先顺序；
   - 默认线程数：`win_affinity=2` 且 SMT=2 且能读到兄弟对时 = **2×物理核**；否则回退 1.5×物理核
     （`--threads N` 显式指定始终优先）。
3. **验证状态**（23:09 微检查，非基准）：默认 → `[cpu] sched: … affinity=2 pairs=6 nthreads=12`、
   `pool threads: 12`；`ENGINE_WIN_AFFINITY=0` → `affinity=0 nthreads=9`（精确旧行为）；
   `ENGINE_THREADS=9` → 9 线程 + 6 对可用（奇数=4 对 + 1 单绑）。
4. **实现过程中发现并修掉一个真 bug（值得记下）**：`GetLogicalProcessorInformationEx` 的条目是
   **变长的**（本机 RelationProcessorCore 条目 48 B，而 `sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)`
   = 80 B）。第一版循环用 `off + sizeof(EX) <= len` 作条件，会在最后一个条目前停止 → **只读到 5 个核**
   （12 线程成对绑核会静默留 2 个线程不绑）。已改为按 `e->Size` 前进、只要求 8 B 固定头能放下，
   复验 `pairs=6`。**本轮扫描数据不受此 bug 影响**：扫描用的 bench 是修改前的版本（用经典
   `GetLogicalProcessorInformation` 路径，正确返回 6 对，且当时已验证 thread t→cpu t、绑定 100% 命中）。
5. **launcher 侧已由 ux-launcher 同步完成（2026-11-17 23:11，已验证源码）**：
   - `default_thread_count()` 现在是 `logical > physical ? physical*2 : 1.5×physical` → 本机 **12**；
   - UI 状态 `win_affinity = true`（`opt_int("win_affinity", 2) == 2`，即引擎默认的 SMT 成对绑核）；
   - launcher 启动命令仍显式传 `--threads 12`，与引擎自动默认一致。
   ⇒ 普通用户从 launcher 启动也会拿到"12 线程 + SMT 成对"。无需再改 launcher。
3. **落地前必须补的验证**（等 GPU/CPU 安静窗口）：
   - 四套验证全绿（verify / smoke / gen_regression / generality）；
   - 35B/Coder 端到端 `test_gen --threads {9,10,12} × {no-aff 后 default}`，确认
     **12 线程吃满 12 LP 时不会拖慢 GPU/WDDM 提交**（CPU 阶段没有空闲核留给驱动，端到端可能
     抵消一部分纯 CPU 收益）——这是唯一可能翻盘的风险点；
   - 干净窗口复测一次 10 vs 12 成对的排序（下面第 4 条）。
4. **数据可信度**：扫描期间机器有 ~40% 背景争用（另一个 agent 的 35B 长上下文 server-mode 基准，
   9 个 HIGHEST 线程），`min/p25` 作为"能力上限"使用（争用只会拖慢）。注意争用下**线程多的配置
   天然抢到更多 CPU 份额**，min 可能偏向高线程数——所以"12 vs 10"需要在干净窗口复测确认，
   但"成对 + ≥10 线程 明显优于 9 no-aff"这个方向是稳的。
5. **历史结论纠正**（必须写进后续文档）：本机 SMT 兄弟对是 `(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)`，
   不是 `(i,i+6)`；之前的"绑核是噪声"结论基于错误的兄弟对假设，**不成立**。

---

## 4. 当前状态 / 待办

- **默认值已改、全量源码编译通过**（2026-11-17 23:09）：`win_affinity` 默认 `2`、默认线程数
  `2×物理核`（本机 12）。`bash build.sh` 的引擎目标因 `server-mode` 长上下文基准（PID 8420）
  仍占着 `build/engine.exe` 的文件锁而暂时只能编到 `build/engine_new.exe`；其余 19 个二进制
  （含 launcher/test_gen）已用新默认重编。**服务器结束后重跑一次 `bash build.sh` 即全部生效。**
  运行中的旧 `engine.exe` 不受影响（Image 已加载，默认仍 `win_affinity=0`）。
- **已改文件**：`src/cpu_matvec.cpp`（Ex 动态兄弟对、成对绑核、15/两倍物理核默认线程数、
  `ENGINE_DBG_SCHED`/`ENGINE_DBG_AFFINITY` 诊断）、`src/config.cpp`（win_affinity 默认 2 + 范围 0..2）、
  `docs/CONFIG.md`、`engine.cfg.example`、`THREAD-AFFINITY-REPORT.md`；工具
  `tools/thread_scan.py`、`tools/thread_e2e.sh`。
- **验证状态（2026-11-17 23:25，新默认 12 线程 + SMT 成对）**：
  - `smoke_models.sh`：**PASS（RC=0）**，40716 / 185 / 2614 / 2614 全命中（绑核/线程数不改变模型数值）；
  - `verify_generality.sh`：**26/26 PASS · GREEN（RC=0）**；
  - `gen_regression.sh`：**9/10**，唯一失败 `35b_capital_of_france` 的 **got 为空**（test_gen 没有
    输出 gen 行 = 该次运行加载/执行失败，不是 token 序列错误；同批 35b_chat22 / 35b_zh_nihao 均 PASS）
    —— 判断为当时 GPU 被其它进程占用导致的装载失败，**必须窗口内重跑确认**，不作为通过；
  - `verify.sh`：本轮在**第一步 build** 就失败（`lld-link: failed to write build/engine.exe:
    Permission denied`，23:13 有一个瞬时 engine.exe 在跑），测试未执行，需窗口内重跑。
  - 结论：**没有任何证据表明绑核/线程数改动破坏数值**（smoke + generality + 9/10 genreg 一致），
    但四套验证尚未全绿，落地前必须补齐。
- **待办（等安静窗口，顺序）**：
  1. `bash build.sh` 全量（launcher.exe 若被运行中的 launcher 锁定就先重命名，已处理过一次）；
  2. `verify.sh` 重跑（仅被 build 文件锁挡住）→ `gen_regression.sh` 重跑（那个空输出的 35B 用例
     必须 PASS 或给出确定性原因）；
  3. 端到端 A/B：`old-9-os` vs `new-default`(12/pair) vs `10/pair`（`N=20 REPS=2 bash tools/thread_e2e.sh`，
     带 GPU 空闲自动等待与逐 run 守卫），重点看 **12 线程吃满 12 LP 后 GPU/WDDM 提交是否被拖慢**；
  4. 若 12 端到端反而慢（哪怕 2%）→ 默认改 `10/pair`（5 核成对，给驱动留 2 个逻辑核）。
- **launcher 需要同步**（见 §3.5）：launcher 固定传 `--threads 9`，需 ux-launcher 把
  `default_thread_count()` 改成 `2×物理核` 或增加"自动"选项，否则从 launcher 启动拿不到新默认。
- **数据可信度**：扫描期有 ~40% 背景争用（另一个 agent 的 server-mode 基准，9 个 HIGHEST 线程），
  `min/p25` 作为能力上限；争用下线程多的配置天然抢更多 CPU 份额，故"12 vs 10"排序需安静窗口复测，
  但"成对 + ≥10 线程 显著优于 9 no-aff"的结论不受影响（首轮 6 reps 与深扫 10 reps 一致）。
- 证据：`build/thread-scan2.json/csv`（主）、`build/thread-scan.json/csv`（复核）。
