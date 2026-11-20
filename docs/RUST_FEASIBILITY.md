# Rust 优化可行性评估（35B 推理引擎）

日期：2026-11-20 · 机器：RTX 3060 12GB + 12 线程（SMT 成对绑核）· 模型：35B Q4_K_M / Qwythos-9B-MTP Q4_K_M
结论先行：**热路径不值得用 Rust 重写**——GPU 侧 Rust 无法编译 CUDA（零收益），CPU 专家 GEMV
用 Rust 逐指令重写后**输出 bit-exact、但实测慢 1.35–1.55×**；12 线程 38–39 GB/s 已接近 DRAM 带宽墙。
有价值的是**新增的外围服务/工具**用 Rust（安全、并发、可维护性），而不是现有推理内核。

本文所有数据都可复现：`engine/rust-probe/`（原型，独立于 `build.sh`）。

---

## 1. 原型实测（回答"Rust 会不会更快"的硬证据）

### 1.1 实验设置

* **同一个内核**：`src/cpu_matvec.cpp` 的 `avx2_dot_q4k()`（fp32 Q4_K 行点积，4 个独立累加器 +
  L1/L2 预取；这是 CPU 专家路径的 Q4_K hot loop）；
* **Rust 端**：`rust-probe/src/lib.rs` 是**逐指令移植**（同样的 `_mm256_fmadd_ps`/`fmsub_ps` 顺序、
  同样的 8 个 sub-block scale/bias 解码、同样的 4 累加器、同样的 `_mm_prefetch` 模式），
  编译 `-C target-cpu=native`（`opt-level=3, lto, codegen-units=1`）；
* **调接口**：Rust 编成 `staticlib`（`rust_probe.lib`），通过 `extern "C"` 在**同一个进程**里
  和 C++ 内核 A/B（`rust-probe/bench_q4k_rust.cpp`，C++ 端用 `rust-probe/cpp_ref.cpp` 暴露
  引擎自己的 `avx2_dot_q4k`，保证跑的就是生产代码）；
* **权重**：同一份确定性合成 Q4_K 数据（每 block 有真实量级的 d/dmin/scales/qs），两个内核读同一块内存；
* **负载形状**：每个"token 层"= 24 个 slab（8 专家 × gate/up/down）× 1152 行 × 2048 输入，
  每 slab 1.27 MB，共 30.4 MB（远超 L3 12 MB），**8 轮轮转**保证每次从 DRAM 读；
* **线程**：12 线程，SMT 兄弟对绑核 `(0,1),(2,3)…`（与 `tools/bench_cpu_layer.cpp` 相同）；
* 机器空闲（四套验证全部结束后）连跑 3 次。

### 1.2 结果（3 次重复，机器空闲）

| 指标 | C++（生产内核） | Rust 原型 | Rust / C++ |
|---|---|---|---|
| 正确性（1152 行逐行对比） | — | **max\|diff\| = 0.000e+00，bit-exact** | — |
| 单线程 | **181–184 ns/row（6.27–6.36 GB/s）** | 242–256 ns/row（4.50–4.76 GB/s） | 慢 **1.34–1.37×** |
| 12 线程 × 24 slab / 层 | **0.813–0.848 ms（37.6–39.2 GB/s）** | 1.10–1.32 ms（24.2–28.9 GB/s） | 慢 **1.35–1.55×** |

原始输出（3 次）：

```
== rep 1 == parity max|diff|=0.000e+00  rows differing=0/1152
1-thread: C++ 183.9 ns/row (6.27 GB/s) | Rust 256.2 ns/row (4.50 GB/s)
12-thread: C++ 0.848 ms/layer (37.57 GB/s) | Rust 1.319 ms/layer (24.15 GB/s)
== rep 2 == 1-thread: C++ 181.5 (6.35 GB/s) | Rust 243.6 (4.73 GB/s)
            12-thread: C++ 0.822 ms (38.75 GB/s) | Rust 1.103 ms (28.88 GB/s)
== rep 3 == 1-thread: C++ 181.2 (6.36 GB/s) | Rust 242.1 (4.76 GB/s)
            12-thread: C++ 0.813 ms (39.17 GB/s) | Rust 1.134 ms (28.09 GB/s)
```

### 1.3 解读

* **语言不改变指令数**：Rust 与 C++ 在这条路径上都能生成 AVX2/FMA，且**输出完全一致**；
  1.35× 的差距来自 codegen/调度/跨 staticlib 内联边界，而不是"语言快慢"。
  更重要的推论：即使 Rust 与 C++ **完全打平**，收益也是 **0**——瓶颈是 DRAM 带宽，不是语言。
* **上限已到**：12 线程 38–39 GB/s 聚合 ≈ 本机双通道 DDR4 实测带宽。35B 每 token 要走
  25 个 CPU 层 × 24 slab × 1.27 MB ≈ **762 MB**，按 39 GB/s 算下限 **≈ 20 ms/token（≈50 tok/s）**；
  实测 24.8 tok/s（其余开销：int8 down 内核、路由、同步、GPU 侧 15 层）。想再快只能改**数据布局/
  算法**（int8 dp4a、更少 CPU 专家/更多 GPU 专家、更小量化）或换硬件，**换语言不行**。

---

## 2. 热路径盘点（哪些地方 Rust 有意义 / 没意义）

| 组件 | 现状/数据 | Rust 可行性 | 收益判断 |
|---|---|---|---|
| **CUDA 内核（95 个）** | `include/engine/kernels.h` 3959 行、**95 个 `__global__`**，启动时用 **NVRTC 运行时编译**（`src/cuda.cpp` 加载 nvrtc） | Rust 没有 CUDA 编译器：`rust-cuda`/`nvptx` target 仍是实验状态，无法编译现有 CUDA C++ 源码，也不能替代 NVRTC 运行时编译 | **零收益/不可行**（且 GPU 不是 35B 的瓶颈：实测 GPU 峰值占用 ~41%） |
| **CPU 专家 GEMV** | Q4_K/Q6_K fp32 AVX2 内核 + int8 内核（`moe_forward_mt`）；见 §1 实测 | 可移植，已验证 bit-exact | **零收益**（实测慢 1.35×，且已到 DRAM 墙） |
| **GGUF 解析** | 模型加载 60–77 s（21.76 GB，含读盘+上传）；tokenizer 元数据 129–156 ms | Rust `gguf` crate 可读，但要逐字段复刻现有校验/量化走线 | 低收益、中风险（加载只发生一次；重写要重新验证 4 模型 + fixtures） |
| **分词器** | 启动 129–156 ms；单请求 prompt 11–42 token，耗时 µs 级 | `tokenizers`/`tiktoken-rs` 有生态 | **零收益**（不在热路径） |
| **HTTP 服务 / JSON** | `src/http_server.cpp` + `json_parse.cpp`；每轮对话 3–20 个请求；解码 40–50 ms/token | `axum`/`hyper` 成熟，TLS/并发更好 | 无可测性能收益（解码主导）；若将来要 TLS/auth/限流/multi-tenant，值得用 Rust **新增 sidecar** 而不是重写 |
| **launcher（Win32 GUI）** | 707 KB 原生 exe，只在启动模型时用 | Rust GUI（egui/winit）体积/启动更重，且要重做 Win32 细节 | 无性能收益；**新增功能**可考虑 Rust |
| **工具/基准** | Python 脚本 + C++ microbench | Rust 单二进制、无 Python 依赖 | 有价值（可维护性），但不是性能问题 |

---

## 3. 工具链可行性（本机实测：无管理员权限）

| 项 | 结果 |
|---|---|
| rustup 安装 | **可行，但用户目录受限**：`rustup-init.exe` 默认装 `<HOME>\.cargo\bin` 报 `拒绝访问 (os error 5)`；改为 `CARGO_HOME`/`RUSTUP_HOME` 指到**工作区内**（`engine/rust-probe/.cargo`、`.rustup`）后安装成功，全程无需管理员 |
| 下载 | `static.rust-lang.org` 本机 ~25 KB/s（12.8 MB 要十几分钟）；**USTC 镜像** `https://mirrors.ustc.edu.cn/rust-static/...` 3.7 MB/s，`RUSTUP_DIST_SERVER` 指过去 1 分钟内装完 |
| 编译器 | `rustc 1.100.0`（stable，x86_64-pc-windows-**msvc** host），`cargo build --release` 正常 |
| 链接器（关键） | 本机**没有 Visual Studio / MSVC 工具链**（Windows SDK 只有 rc.exe/mt.exe/导入库，没有 link.exe），因此 rustc 不能链接 exe；但 **`staticlib` 不需要外部链接器**（rustc 自带归档），产出 `rust_probe.lib` 后用现有 **`ziglang c++` 链接成功**（需要补一个 `extern "C" int _fltused = 0;` 符号，MSVC CRT 的浮点标记） |
| 结论 | 无管理员权限下 **Rust 原型完全可行**：cargo 编 staticlib + zig c++ 链最终 exe。生产化建议统一 ABI（全 msvc 或全 gnu），不要长期依赖 msvc-staticlib + mingw 链接的混搭 |

产物：`rust-probe/target/release/rust_probe.lib`（13 MB，含 std）、`build/bench_q4k_rust.exe`。
工具链目录占用约 **649 MB**（`.rustup` 598 MB），实验完可删：
`rm -rf engine/rust-probe/{.cargo,.rustup,target}`。

复现命令：

```bash
cd engine/rust-probe
CARGO_HOME="$PWD/.cargo" RUSTUP_HOME="$PWD/.rustup" \
RUSTFLAGS="-C target-cpu=native" ./.cargo/bin/cargo build --release
cd ..
python -m ziglang c++ -std=c++17 -O2 -Iinclude -Irust-probe \
  rust-probe/bench_q4k_rust.cpp rust-probe/cpp_ref.cpp src/config.cpp \
  rust-probe/target/release/rust_probe.lib -o build/bench_q4k_rust.exe
./build/bench_q4k_rust.exe 80 300     # 正确性 + 单线程 + 12 线程 SMT 对绑核
```

---

## 4. 互操作方案（如果确实要引入 Rust）

1. **C ABI staticlib（本次已验证）**：`#[no_mangle] pub extern "C" fn`，Rust 编译成
   `staticlib`，C++ 侧 `extern "C"` 声明 + `ziglang c++` 链接。适合**叶子函数**（无回调、无
   STL 类型穿越边界）。成本最低，本次已跑通；注意 C++/Rust 的内存所有权（谁分配谁释放）与
   `panic=abort`。
2. **`cxx` crate（双向）**：定义共享 `#[cxx::bridge]`，自动生成两边的 glue；支持 `String`/
   `Vec`/`UniquePtr` 等类型。需要 `cc` crate 能驱动 C++ 编译器（可用 `CC/CXX` 指向 zig c++）。
   适合新增模块、中等规模接口。
3. **`bindgen`（Rust 调现有 C++）**：从 `include/engine/*.h` 生成 FFI 声明，适合工具侧复用
   引擎类型；对模板/异常/`std::string` 需要额外包装层。
4. **反向（C++ 调 Rust）用 sidecar 进程**：Rust 写独立服务（如管理/metrics/网关），通过
   `http://127.0.0.1:7030` 和引擎通信。**零 ABI 风险、可独立重启**，是性价比最高的切入方式。

不建议：用 Rust 重写 CUDA 内核（不可行）；用 Rust 重写 CPU GEMV（实测无收益且要重新验证
bit-exact 和 int8 内核选择逻辑）。

---

## 5. 迁移 ROI 排序与建议

| 优先级 | 方向 | 工作量 | 风险 | 收益 |
|---|---|---|---|---|
| **1（推荐）** | **Rust sidecar / 新工具**：管理 API（模型扫描、下载、metrics、日志）、bench 工具、CI 校验器 | 中 | 低（不改热路径） | 安全性/可维护性、单二进制交付；无性能承诺 |
| 2 | 若将来要 TLS / 鉴权 / 多租户：用 Rust 写**反向代理/网关** sidecar | 中 | 低 | 生态成熟（axum/rustls），不动引擎 |
| 3 | 分词器/采样器/JSON 等**纯 CPU 叶子模块**（若因安全需要） | 高 | 中（bit-exact/采样链对齐风险） | 仅维护性 |
| 4 | HTTP 服务核心重写 | 高 | 中 | 无可测性能收益 |
| 5 | GGUF 解析重写 | 高 | 中高 | 低（一次性） |
| **6（不要做）** | CPU GEMV / CUDA 内核 / cuBLAS 胶水 | 很高 | 高 | **实测：0 或负收益** |

**总体建议**：性能瓶颈在"CPU 专家 GEMV 的 DRAM 带宽 + GPU 专家层数 + MTP 是否有效"，
不在语言。Rust 适合作为**外围服务/工具**的第二语言，而不是替换推理内核；
每 token 速度的下一步杠杆是：更多 GPU 专家层（腾 VRAM）、CPU int8 内核（dp4a 路线）、
MTP 修复后的有效投机（当前 35B hybrid SSM 不保真，见 `docs/SERVE-STABILITY-FIXES.md` §5.3）、
以及 reasoning/think-budget 的默认值调优。
