// In-process A/B of the CPU Q4_K dot kernel: our C++ AVX2 kernel vs a Rust port
// (rust-probe/src/lib.rs, same instruction order) on identical weights.
//
// The measurement mirrors the 35B decode hot path: each token evaluates 24 slabs
// (8 experts x gate/up/down) of 1152 rows x 2048 inputs = 1.33 MB each (32 MB
// total, far beyond L3), rotated so every pass re-reads from DRAM, with 12 std
// threads bound as SMT sibling pairs (0,1), (2,3) ... - the same configuration as
// tools/bench_cpu_layer.cpp.
//
// Build (zig c++):
//   ziglang c++ -std=c++17 -O2 -Iinclude -Irust-probe \
//       rust-probe/bench_q4k_rust.cpp src/cpu_matvec.cpp src/config.cpp \
//       rust-probe/target/x86_64-pc-windows-gnu/release/librust_probe.a \
//       -o build/bench_q4k_rust.exe
//
// Usage: bench_q4k_rust [iters] [single-thread-iters]
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "engine/config.h"
#include "engine/cpu_matvec.h"

#ifdef _WIN32
#include <windows.h>
#endif

// MSVC-built Rust objects reference the CRT's _fltused marker; this Zig/LLVM
// mingw-flavoured link does not pull it in, so define it here.
extern "C" {
int _fltused = 0;
}

extern "C" {
void rust_q4k_matvec(const unsigned char* w, int n_out, int n_in, const float* x, float* y);
void rust_q4k_dot_row(const unsigned char* row, const float* x, int n_in, float* out);
// the engine's own kernel (exposed by rust-probe/cpp_ref.cpp)
void cpp_q4k_dot_row(const unsigned char* row, const float* x, int n_in, float* out);
void cpp_q4k_matvec(const unsigned char* w, int n_out, int n_in, const float* x, float* y);
}

namespace {

constexpr int kHidden = 2048;      // n_in (Q4_K: 8 blocks x 144 B = 1152 B/row)
constexpr int kInter = 1152;       // n_out (expert intermediate size)
constexpr int kSlabs = 24;         // 8 experts x (gate, up, down) per token
constexpr int kThreads = 12;

constexpr size_t row_bytes() { return static_cast<size_t>(kHidden / 256) * 144; }
constexpr size_t slab_bytes() { return row_bytes() * kInter; }

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

unsigned rng_state = 12345u;
unsigned rng() {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
}

void put_half(unsigned char* p, float v) {
  unsigned int bits;
  std::memcpy(&bits, &v, 4);
  const unsigned int sign = (bits >> 16) & 0x8000u;
  int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  unsigned int mant = (bits >> 13) & 0x3FFu;
  if (exp < -14) {
    exp = 0;
    mant = 0;
    const unsigned int h = sign;
    p[0] = static_cast<unsigned char>(h & 0xFF);
    p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
    return;
  }
  if (exp > 15) exp = 15;
  const unsigned int h = sign | (static_cast<unsigned int>(exp + 15) << 10) | mant;
  p[0] = static_cast<unsigned char>(h & 0xFF);
  p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
}

// Deterministic plausible Q4_K weights (same bytes are fed to both kernels).
void fill_slab(unsigned char* base) {
  for (int r = 0; r < kInter; ++r) {
    unsigned char* row = base + static_cast<size_t>(r) * row_bytes();
    for (int b = 0; b < kHidden / 256; ++b) {
      unsigned char* p = row + static_cast<size_t>(b) * 144;
      put_half(p, 0.02f + 0.001f * static_cast<float>((r + b) % 7));
      put_half(p + 2, 0.005f);
      for (int i = 0; i < 12; ++i) p[4 + i] = static_cast<unsigned char>(rng() & 0xFF);
      for (int i = 0; i < 128; ++i) p[16 + i] = static_cast<unsigned char>(rng() & 0xFF);
    }
  }
}

// SMT sibling pairs on this machine are (0,1), (2,3), ... (not (i, i+6)).
void bind_pair(int t) {
#ifdef _WIN32
  const DWORD_PTR mask = static_cast<DWORD_PTR>(3) << (2 * t);
  SetThreadAffinityMask(GetCurrentThread(), mask);
#else
  (void)t;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 40;
  const int st_iters = argc > 2 ? std::atoi(argv[2]) : 20000;
  std::printf("config: %d slabs x %.2f MB = %.1f MB working set, %d threads (SMT pairs)\n",
              kSlabs, slab_bytes() / 1048576.0, kSlabs * slab_bytes() / 1048576.0, kThreads);

  std::vector<std::vector<unsigned char>> slabs(kSlabs);
  for (auto& s : slabs) {
    s.resize(slab_bytes());
    fill_slab(s.data());
  }
  std::vector<float> x(kHidden);
  for (int i = 0; i < kHidden; ++i) x[i] = 0.01f * static_cast<float>(static_cast<int>(rng() % 200) - 100);
  std::vector<float> y_cpp(static_cast<size_t>(kInter) * kThreads);
  std::vector<float> y_rust(static_cast<size_t>(kInter) * kThreads);

  // ---- 1) parity: every row of the first slab, C++ matvec vs Rust matvec ----
  {
    cpp_q4k_matvec(slabs[0].data(), kInter, kHidden, x.data(), y_cpp.data());
    rust_q4k_matvec(slabs[0].data(), kInter, kHidden, x.data(), y_rust.data());
    double max_diff = 0.0;
    int n_diff = 0;
    for (int i = 0; i < kInter; ++i) {
      const double d = std::fabs(static_cast<double>(y_cpp[i]) - static_cast<double>(y_rust[i]));
      if (d > 0.0) ++n_diff;
      max_diff = std::max(max_diff, d);
    }
    std::printf("parity : C++ vs Rust max|diff|=%.3e, rows differing=%d/%d\n", max_diff,
                n_diff, kInter);
    if (n_diff != 0) std::printf("WARNING: kernels are not bit-identical\n");
  }

  // ---- 2) single-thread kernel bandwidth (rotating slabs) ----
  {
    std::vector<float> y(static_cast<size_t>(kInter));
    double t_cpp = 0.0, t_rust = 0.0;
    volatile double sink = 0.0;
    int it = 0;
    for (int warm = 0; warm < 2; ++warm) {
      for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = now_ms();
        for (int i = 0; i < st_iters; ++i) {
          const auto& slab = slabs[static_cast<size_t>((it++) % kSlabs)];
          cpp_q4k_matvec(slab.data(), kInter, kHidden, x.data(), y.data());
          if (pass) t_cpp += 0.0;
        }
        const auto t1 = now_ms();
        const auto t2 = now_ms();
        for (int i = 0; i < st_iters; ++i) {
          const auto& slab = slabs[static_cast<size_t>((it++) % kSlabs)];
          rust_q4k_matvec(slab.data(), kInter, kHidden, x.data(), y.data());
        }
        const auto t3 = now_ms();
        sink += y[0];
        if (pass) {
          t_cpp = t1 - t0;
          t_rust = t3 - t2;
        }
      }
    }
    (void)sink;
    const double rows = static_cast<double>(st_iters) * kInter;
    const double bytes = rows * row_bytes();
    std::printf("1-thread: C++ %.1f ns/row (%.2f GB/s) | Rust %.1f ns/row (%.2f GB/s) | "
                "ratio C/R = %.3f\n",
                t_cpp * 1e6 / rows, bytes / t_cpp / 1e6, t_rust * 1e6 / rows,
                bytes / t_rust / 1e6, t_cpp / t_rust);
  }

  // ---- 3) 12-thread layer pass over the 24 rotated slabs ----
  {
    std::vector<std::thread> ths;
    std::vector<double> ms_cpp(kThreads, 0.0), ms_rust(kThreads, 0.0);
    for (int t = 0; t < kThreads; ++t) {
      ths.emplace_back([&, t]() {
        bind_pair(t);
        // each thread owns slabs t, t+12 (2 of 24); y slice offset trick
        std::vector<float> y(static_cast<size_t>(kInter));
        for (int warm = 0; warm < 2; ++warm) {
          for (int pass = 0; pass < 2; ++pass) {
            const auto t0 = now_ms();
            for (int it = 0; it < iters; ++it) {
              for (int s = t; s < kSlabs; s += kThreads) {
                const auto& slab = slabs[static_cast<size_t>(s)];
                cpp_q4k_matvec(slab.data(), kInter, kHidden, x.data(), y.data());
              }
            }
            const auto t1 = now_ms();
            const auto t2 = now_ms();
            for (int it = 0; it < iters; ++it) {
              for (int s = t; s < kSlabs; s += kThreads) {
                const auto& slab = slabs[static_cast<size_t>(s)];
                rust_q4k_matvec(slab.data(), kInter, kHidden, x.data(), y.data());
              }
            }
            const auto t3 = now_ms();
            if (pass) {
              ms_cpp[t] = t1 - t0;
              ms_rust[t] = t3 - t2;
            }
          }
        }
      });
    }
    for (auto& th : ths) th.join();
    double cpp = 0.0, rust = 0.0;
    for (int t = 0; t < kThreads; ++t) {
      cpp = std::max(cpp, ms_cpp[t]);
      rust = std::max(rust, ms_rust[t]);
    }
    const double total = static_cast<double>(iters) * kThreads * 2 * slab_bytes();
    std::printf("12-thread: C++ %.3f ms/layer (%.2f GB/s) | Rust %.3f ms/layer (%.2f GB/s) | "
                "ratio C/R = %.3f\n",
                cpp / iters, total / cpp / 1e6, rust / iters, total / rust / 1e6, cpp / rust);
  }
  return 0;
}
