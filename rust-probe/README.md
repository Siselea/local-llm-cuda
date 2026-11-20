# rust-probe：Rust 可行性原型（不参与 build.sh）

回答"要不要用 Rust 优化引擎"的实测原型，结论与完整数据见
[`../docs/RUST_FEASIBILITY.md`](../docs/RUST_FEASIBILITY.md)。

内容：

| 文件 | 作用 |
|---|---|
| `src/lib.rs` | Q4_K AVX2 行点积的 Rust **逐指令移植**（与 `src/cpu_matvec.cpp` 的 `avx2_dot_q4k` 同序：4 累加器、sub-block scale/bias、L1/L2 预取）；C ABI `rust_q4k_dot_row` / `rust_q4k_matvec` |
| `cpp_ref.cpp` | `#include "../src/cpu_matvec.cpp"` 并暴露引擎自己的 `avx2_dot_q4k`，保证 A/B 跑的是生产代码 |
| `bench_q4k_rust.cpp` | 同进程 A/B：正确性逐行对比 + 单线程 + 12 线程 SMT 成对绑核、24 slab（8 专家×gate/up/down）轮转、30.4 MB 工作集 |
| `Cargo.toml` | `crate-type = ["staticlib","rlib"]`，`panic=abort`，LTO |

复现（本机无管理员权限、无 MSVC，zig 作为链接器）：

```bash
cd engine/rust-probe
# 一次性：Rust 工具链装进工作区（用户目录 .cargo 被 ACL 拒绝）
curl -sSL -o rustup-init.exe https://mirrors.ustc.edu.cn/rust-static/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe
CARGO_HOME="$PWD/.cargo" RUSTUP_HOME="$PWD/.rustup" \
  RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static \
  ./rustup-init.exe -y --no-modify-path --profile minimal --default-toolchain stable
CARGO_HOME="$PWD/.cargo" RUSTUP_HOME="$PWD/.rustup" \
  RUSTFLAGS="-C target-cpu=native" ./.cargo/bin/cargo build --release
cd ..
python -m ziglang c++ -std=c++17 -O2 -Iinclude -Irust-probe \
  rust-probe/bench_q4k_rust.cpp rust-probe/cpp_ref.cpp src/config.cpp \
  rust-probe/target/release/rust_probe.lib -o build/bench_q4k_rust.exe
./build/bench_q4k_rust.exe 80 300
```

实测结论（2026-11-20，三次重复）：

* 输出 **bit-exact**（max|diff| = 0）；
* 单线程：C++ 181–184 ns/row（6.3 GB/s）vs Rust 242–256 ns/row（4.5–4.8 GB/s）；
* 12 线程 24 slab：C++ 0.813–0.848 ms（37.6–39.2 GB/s）vs Rust 1.10–1.32 ms（24.2–28.9 GB/s）；
* **Rust 慢 1.34–1.55×**，且 12 线程聚合已接近 DDR4 带宽墙 → 换语言对 CPU GEMV 没有收益。

清理（工具链约 649 MB）：`rm -rf engine/rust-probe/{.cargo,.rustup,target}`
