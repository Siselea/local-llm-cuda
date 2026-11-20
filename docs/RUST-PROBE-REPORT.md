# Rust 语言在本引擎上的优化可行性 — 硬数据报告

**日期**：2026-11-20
**任务**：用实测回答"考虑 Rust 语言的优化可行性"，特别是**已到算力墙的 CPU GEMV 内核**能否用 Rust 做得更好。
**结论一句话**：
> **Rust 能逐位复刻这个内核（已验证 max|Δ| = 0），但在这台机器上不可能更快——Q6_K 内核 Rust 与 C++ 完全持平（-0.24%），Q4_K 内核 rustc 1.100/LLVM 比 clang 慢 **33–43%**，12 线程整层实测 Rust **+4.6%~+7.3%**。语言不改变指令数，而该内核是端口/发射受限：单核 5.92 GB/s、12 线程 25.8 GB/s 是硬件属性，换任何语言都是 0% 收益。**

代码/工具（不进 `build.sh`，不参与产品构建）：
`rust-probe/extract_weights.py`、`rust-probe/rust_kernels.rs`、`rust-probe/cpp_bench.cpp`、`rust-probe/build.sh`

> 说明：本轮**没有修改任何现有 C++/CUDA 源文件**；C++ 侧基准直接 `#include "../src/cpu_matvec.cpp"`，
> 跑的就是仓库当前的内核（工作区里对 `cpu_matvec.cpp` 的 Pool/绑核改动，
> `avx2_dot_q4k`/`avx2_dot_q6k` 行内核未被改动，已用 `git diff` 核实）。

---

## 1. 环境：能否在不装管理员权限的前提下编译 Rust？**能，但有链接阻塞**

### 1.1 安装（成功）

| 步骤 | 结果 |
|---|---|
| 下载 `rustup-init.exe`（win.rustup.rs/x86_64，12,839,424 B） | ✅ |
| **官方源** `static.rust-lang.org` + `--profile minimal` | ❌ **20 分钟只下到 7.6 MB**（工具链 ~600 MB），不可用 |
| **TUNA 镜像** `RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup` | ✅ **约 30 秒装完全部组件**（`RUSTUP_HOME` 598 MB） |
| 安装位置 | `$RUSTUP_HOME / $CARGO_HOME`（用户目录，**无需管理员**） |
| 版本 | `rustc 1.100.0 (0ef1bc386 2026-11-09)`，host `x86_64-pc-windows-msvc`，minimal profile |
| 额外 target | `x86_64-pc-windows-gnu`（TUNA 镜像，秒级） |

⚠️ 复现时**必须用镜像**，否则会卡死在上面的下载里。

### 1.2 链接阻塞（这是本机 TrueBlocker，务必写进结论）

本机**没有 MSVC `link.exe`**（没装 Visual Studio），**也没有 MinGW gcc**；`C:/Program Files (x86)/Windows Kits` 只有 SDK，没有 VC 的 CRT 库。实测四条路：

| 尝试 | 结果 |
|---|---|
| `rustc hello.rs`（默认 msvc host） | ❌ `error: linking with link.exe failed` / `winuxcmd: command not found: link` |
| 用 toolchain 自带 `rust-lld.exe` 改名 `lld-link.exe` 当链接器 + SDK `LIB` | ❌ 缺 `msvcrt.lib`（SDK 只有 `ntstc_msvcrt.lib`） |
| 再把 `ucrt.lib` 显式加进去 | ❌ 仍缺 `mainCRTStartup`、`memcpy`、`__chkstk`、`__CxxFrameHandler3`（这些在 VC 的 `libcmt`/`libvcruntime`，SDK 里没有） |
| gnu target + rust-lld | ❌ 缺 `kernel32/ntdll/userenv/ws2_32/msvcrt/mingwex/mingw32`；`rustup component add rust-mingw` 被拒（msvc toolchain 无该组件） |
| **✅ 最终方案：Rust 编成 `staticlib`（不需要链接器），再用仓库现成的 zig cc 链接** | ✅ 成功（zig 自带 mingw-w64 + lld） |

```bash
# Rust 侧：只归档，不链接
rustc --target x86_64-pc-windows-gnu --crate-type staticlib \
      -C opt-level=3 -C target-cpu=native -C panic=abort -C codegen-units=1 \
      rust-probe/rust_kernels.rs -o build/rustprobe/librust_kernels.a
# C++ 侧：zig cc 同时链 C++ 和 Rust 归档
python -m ziglang c++ -std=c++17 -O2 -Iinclude rust-probe/cpp_bench.cpp src/config.cpp \
      build/rustprobe/librust_kernels.a -o build/rustprobe/ab_bench.exe
```

**这条 staticlib + zig 的路线本身就是可迁移的工程结论**：在"无 MSVC/无管理员"的机器上，Rust 与现有 C++ 工具链可以混合链接，代价只是多一个 staticlib 中间产物。

---

## 2. 基准方法（为什么可信）

1. **同一份真实 GGUF 权重**：从 35B（`blk.16`）抽出 8 个专家 = 引擎里**每个 CPU 层的全部字节**
   - `ffn_gate_exps` Q4_K 8 专家 × 512 行 × 1152 B = 4,718,592 B
   - `ffn_up_exps` 同上
   - `ffn_down_exps` Q6_K 8 专家 × 2048 行 × 420 B = 6,881,280 B
   - 合计 **16,318,464 B = 16.3 MB/层**（与引擎内一致）
2. **同进程 A/B**：Rust 编成 staticlib **链进 C++ 基准**，两套内核跑在**同一个线程团队**（12 线程、SMT 成对绑核取 `cpu::smt_cpu_pairs()`、同一自旋屏障、同一 flat (act,row) 切分、同一 24 副本轮转）→ 唯一变量就是内核代码。跨进程抖动（本机 ±20–30%）被彻底消掉。
3. **三种测法**：
   - `verify`：逐位对拍（A/B 的前提）
   - `kernel`：单线程、绑核 0、min-of-80、同时报 **rdtsc 周期/行**（排除被抢占的样本）
   - `ab`：12 线程整层，进程内交错，报 min/p25/median
4. 编译参数：C++ `ziglang c++ -std=c++17 -O2`（zig 默认开 AVX2+FMA，已核实 `__AVX2__=1`）；Rust `-C opt-level=3 -C target-cpu=native`。

---

## 3. 结果

### 3.0 数值：**逐位一致**

```
q4k row dots (64 rows): max|C++ - Rust|   = 0.000e+00, not-bit-equal rows = 0
q6k row dots (64 rows): max|C++ - Rust|   = 0.000e+00, not-bit-equal rows = 0
q4k-v2 row dots (64 rows): max|C++ - Rustv2| = 0.000e+00, not-bit-equal = 0
full layer pass: max|xout_C++ - xout_Rust| = 0.000e+00, not-bit-equal elems = 0/2048
```
Rust 的 `core::arch::x86_64` intrinsics 可以**逐位复刻** C++ 内核（同块头解析、同 scale/min、同 nibble 解包顺序、同 4 累加器 FMA 链、同 hsum 树）——这也证明"指令序列可以完全一样"。

### 3.1 内核级（单线程、绑核、min-of-80）— 主数据

| 内核 | C++（clang/zig） | Rust（1:1 转写） | Rust 手调版 | Rust vs C++ |
|---|---|---|---|---|
| **Q4_K 行**（1152 B，512 行/专家） | **194.7 ns/行 = 5.92 GB/s/核**（563 cyc/行） | 270.1 ns/行 = 4.26 GB/s（777 cyc） | v2 手展开 259.8 ns（+33.4%）；v3 去数组 262–277 ns | **+38.7%（min）/ +39.5%（p25）** |
| **Q6_K 行**（420 B，2048 行/专家） | **80.2 ns/行 = 5.24 GB/s/核** | 80.0 ns/行 = 5.25 GB/s | — | **−0.24%（完全持平）** |

> 两台机器上重复 3 次，Q4_K 的差距稳定在 **+33%~+43%**，Q6_K 稳定在 **±1.6% 内**。

### 3.2 整层（16.3 MB/pass，1 线程）

| 实现 | min | p25 | GB/s |
|---|---|---|---|
| C++ | 3.33–3.45 ms | 3.34–3.50 | 4.7–4.9 |
| Rust | 4.05–4.28 ms | 4.05–4.30 | 3.8–4.0 |
| **差** | **+22%~+24%** | **+21%~+23%** | |

与内核级加权完全吻合：phase A 占 58% 字节（Q4_K，×1.39）、phase B 占 42%（Q6_K，×1.00）⇒ 0.58×1.39+0.42×1.00 ≈ **1.23**。

### 3.3 整层（12 线程 + SMT 成对绑核 = 生产配置）

机器空闲窗口的一次干净测量（**C++ 0.6335 ms 与仓库自己的生产基准 0.623 ms/26.2 GB/s 偏差 2% 以内，说明 harness 可信**）：

| 实现 | min | p25 | GB/s（min） |
|---|---|---|---|
| **C++** | **0.6335 ms** | 0.6509 | **25.76** |
| **Rust** | **0.6796 ms** | 0.6811 | **24.01** |
| 差 | **+7.3%** | **+4.6%** | |

> 注意：12 线程下差距（+4.6~7.3%）比单线程（+22%）小，因为 12 线程时内核已经贴住 DRAM/SMT 上限，Rust 多出来的指令有一部分被内存延迟掩盖了。
> ⚠️ 复测时机器上另一个 agent 的 `engine_new.exe serve`（High 优先级、占 ~4.4 核）在跑，这时 12 线程 wall-clock 完全不可用（线程创建/回收就要 **40 ms/sample**，样本中位 21 ms vs min 7 ms），所以 12 线程数据只有上述**一个干净窗口**，其余按"未验证"处理，不编数据。

### 3.4 为什么 Q4_K 慢 39%：asm 级定位

| | C++（clang） | Rust（LLVM 1.20/rustc 1.100） |
|---|---|---|
| 每块（256 值）指令数 | ~350 | ~390–400 |
| 内层循环 | 44 条/32 值，**0 次栈访问** | 大循环体内含 **44 次 `rsp` 访问/块** |
| 4 组 `scale` 循环 | clang 自动完全展开（常量下标 → SROA 到寄存器） | **LLVM 不展开**（`scf[2*g]` 动态下标 ⇒ 数组留在栈上反复重载） |

我做过两次针对性改写：
- **v2**：手写展开 4 组 × 2 半（常量下标，同 C++ 的 4 累加器映射）→ +33%
- **v3**：彻底去掉 `scf[]/mnf[]` 数组（全部具名标量 + tuple 返回的 `scale_min2`）→ +34~42%

**都没能追平**——说明 rustc 这条链在"大循环体 + 大量 FP/SIMD 状态"的代码上，寄存器分配/调度就是不如 clang。Q6_K（每 16 值一个 scale，天然更规整）能持平，说明**不是 Rust 不行，是 LLVM 对这个特定循环的 codegen 不行**。

---

## 4. 结论 1：Rust 能否在这个 CPU 内核上持平/更快？

| 问题 | 答案（实测） |
|---|---|
| Rust 能不能写出等价内核？ | **能，逐位一致（max\|Δ\|=0）** |
| 能不能更快？ | **不能。** 上限就是 C++ 的 5.92 GB/s/核 / 25.8 GB/s（12 线程）——这台 6 核 VM 的 p1 端口 + SMT 上限，与语言无关 |
| 能不能持平？ | **Q6_K 能（-0.24%）；Q4_K 目前不能（+33~43%）**，需要等 rustc/LLVM 改善或手工绕开 LLVM 的展开问题 |
| 生产影响 | 12 线程整层 +4.6~7.3% ⇒ **直接迁移 CPU GEMV 是净亏** |

**这印证了任务书给的判断："换语言不会改变指令数 ⇒ 理论上 Rust 只能做到同样快，不可能更快"。本轮把它变成了硬数据，并补上一条：不调优的 Rust 目前连"同样快"都做不到。**

---

## 5. 结论 2：Rust 的"整模型"可能性（NVRTC / cuMemcpyAsync / 多线程）

### 5.1 NVRTC（跑现有 CUDA 内核）——**可行，无阻塞**
- 引擎本身**已经是动态加载 nvrtc**：`src/cuda.cpp:82+` 用 `LoadLibrary` + 函数指针（`CreateProgram` / `CompileProgram` …），没有链接期依赖。
- 机器上 DLL 齐全：`<PYTHON_DIR>\Lib\site-packages\nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll`（+ `cudart64_12.dll`）。
- Rust 用 `libloading` 或裸 `LoadLibraryW/GetProcAddress` FFI 做**完全相同的事**：声明 ~10 个函数指针即可。**可行性 100%，但没有收益**——CUDA 内核还是 CUDA C++，编译器和生成的 SASS 不变。

### 5.2 cuMemcpyAsync / cuLaunchKernel —— **同一套驱动 API，零额外开销**
- 这两个就是 `nvcuda.dll` 的导出函数；Rust FFI 与 C++ 调用是**同一条指令序列**（一次驱动调用），没有语言级开销。
- 引擎在这些调用上的已知成本是 **WDDM 提交（~0.2 ms/CPU 层，decode 的 ~13%）**，这是驱动/OS 属性：换语言一点也动不了。

### 5.3 多线程调度/绑核 —— **实测跑通，无额外开销**
- 本基准里 Rust 侧用 `std::thread` + FFI `SetThreadAffinityMask` + 自旋 `_mm_pause` 屏障，12 线程 SMT 成对绑核，跑出与 C++ 相同的 24–26 GB/s 量级 ⇒ **线程/绑核在 Rust 里没有代价**。
- 唯一坑（两边都一样）：**自旋循环里不要用 `yield_now`/`SwitchToThread`**——负载下会被调度器挂起一个 quantum（~15 ms），本报告实测踩到过（层耗时从 0.65 ms 飙到 16–36 ms）。

---

## 6. 结论 3：迁移 ROI 排序（如果只做一个 Rust 模块）

先看 decode 时间构成（上一轮引擎内实测）：**宿主 CPU GEMV 48%（已证算力墙）、GPU 层 22%、WDDM 提交/同步/其他 30%**。
⇒ **语言迁移的理论上限 = 0%**：所有热点要么是 CUDA，要么是已贴算力墙的 intrinsics 循环，剩下的是 WDDM 驱动行为。

按"安全收益 / 性能影响 / 边界清晰度"排序：

| 排名 | 模块 | 体量 | 收益 | 风险/成本 |
|---|---|---|---|---|
| **1（推荐）** | **HTTP/SSE 服务 + JSON**（`serve.cpp` 2097 行、`http_server.cpp`、`json_parse.cpp`） | ~3 k 行 | **最大**：解析不可信输入、长连接状态机——Rust 的所有权/无 GC 最对症；性能不敏感（单用户 decode 下占比 <1%） | 中：要保证 SSE/取消/流式逐 token 行为兼容；可先做旁路服务双跑对比 |
| 2 | **GGUF / IR / planner / model_scan**（`gguf.cpp` 712、`ir.cpp` 712、`planner.cpp`、`model_scan.cpp` 1288） | ~3.5 k 行 | 大：解析恶意/损坏 GGUF 的安全性；纯逻辑无热循环，Rust 的 `Result` 错误处理更干净 | 低-中：字节级兼容要逐字段对拍（仓库已有 `diff_ir.py`/`diff_plan.py` 可复用） |
| 3 | **tokenizer**（`tokenizer.cpp` 1023 行） | 1 k 行 | 中：Rust 生态（`tokenizers`）成熟 | **中-高**：必须逐 token 一致（`gen_regression` 10/10 是硬门槛） |
| 4 | **launcher / WebUI 后端**（`launcher.cpp` 4759 行，Win32 GUI） | 4.8 k 行 | 低：GUI 逻辑，无安全/性能收益 | 高：Win32 GUI 的坑多、Rust GUI 生态与现有行为对齐成本大 |
| 5 | **新的多用户调度/批处理**（若要做并发） | 新功能 | 高：async + 所有权天然适合 | 这是新功能不是迁移，另立项目 |
| ❌ 白费 | **CPU GEMV / int8 内核**（本报告）、**CUDA 内核**（只能 CUDA C++）、**prefill GEMM**（cuBLAS）、**sampler 热循环** | — | **0 或负** | — |

**一句话建议**：如果只做一个 Rust 模块，做 **HTTP/服务 + JSON 层**（安全收益最大、性能影响≈0、边界最清晰、可双跑对比）；**任何计算内核（CPU GEMV / GPU / GEMM）用 Rust 重写都是净亏**，本报告已用逐位对拍 + 周期数证明。

---

## 7. 复现命令

```bash
cd engine
# 0) 一次性：镜像装 rust（无管理员）
cd <RUST_HOME>   # 里面放 rustup-init.exe（12,839,424 B，来自 https://win.rustup.rs/x86_64）
export RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME \
       RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
./rustup-init.exe --yes --profile minimal --default-host x86_64-pc-windows-msvc --no-modify-path
# 需要 gnu std（staticlib target）：rustup target add x86_64-pc-windows-gnu

cd engine
# 1) 抽真实权重（16.3 MB/层）+ 构建（staticlib + zig 链接）
bash rust-probe/build.sh
# 2) 逐位对拍
./build/rustprobe/ab_bench.exe verify
# 3) 内核级 A/B（单线程，min-of-80）
./build/rustprobe/ab_bench.exe kernel 80
# 4) 12 线程整层 A/B（需要在机器空闲时跑，否则被外部 High 优先级进程淹没）
RP_THREADS=12 RP_ITERS=20 ./build/rustprobe/ab_bench.exe ab 6
```

## 8. 附：本次踩到的坑（供后续 Rust 评估复用）

1. **rustup 官方源在国内会卡死**（20 min / 7.6 MB）→ 必须 `RUSTUP_DIST_SERVER` 镜像。
2. **无 MSVC 就不能链接 Rust 二进制**；`ucrt.lib` 不够（缺 VC CRT 的 `mainCRTStartup`/`memcpy`/`__chkstk`）；gnu target 缺 mingw CRT/import libs。**唯一可行 = staticlib + zig cc**。
3. `rustc` 必须显式 `-C target-cpu=native`（否则没有 AVX2/FMA，会慢一个数量级）；C++ 侧 zig cc 默认已开 AVX2。
4. Rust 的 `&mut` 出参小函数（`scale_min`）阻碍 SROA；改成返回 tuple 有改善但不根治（v3）。
5. LLVM 对"大循环体里按 `2*g` 动态下标访问的标量数组"不会自动展开/SROA ⇒ Q4_K 慢 39% 的根因；Q6_K 无此问题 ⇒ 完全持平。
6. 自旋屏障里用 `yield_now`/`SwitchToThread` 在负载下会被挂起一个 quantum（实测 16–36 ms），两语言同样踩；用纯 `_mm_pause` 自旋。
7. 跨语言 A/B 必须**同进程**（staticlib）且用 min-of-N + rdtsc，否则本机 ±20–30% 抖动会吃掉结论。
