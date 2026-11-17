// Standalone correctness + bandwidth bench for the Q6_K dp4a GEMV variants.
//
//   build: python -m ziglang c++ -std=c++17 -O2 -Iinclude <engine sources> \
//          tools/bench_q6k_dp4a.cpp -Itools -o build/bench_q6k_dp4a.exe
//   run:   ./build/bench_q6k_dp4a.exe [n_out] [n_in]
//
// Reference is the validated fp32 kernel gemv_q6k; a variant must reach
// rel_l2 <= 5e-5 against it (the x side is int8-quantised, the weight side is exact).
//
// The timing loop rotates over several copies of the matrix so that back-to-back
// launches cannot serve part of the next pass out of L2 (3 MB on the 3060) -- without
// that, small shapes look 10-20% faster than they are inside the engine, where every
// tensor is read exactly once per token.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif
// Driver API attribute query (debug only): cuFuncGetAttribute(4 = NUM_REGS).
static int kernel_regs(void* fn, int* max_threads = nullptr) {
#ifdef _WIN32
  HMODULE lib = GetModuleHandleA("nvcuda.dll");
  if (!lib) return -1;
  using Fn = int(__stdcall*)(int*, int, void*);
  auto get = reinterpret_cast<Fn>(GetProcAddress(lib, "cuFuncGetAttribute"));
  if (!get) return -1;
  int regs = -1;
  if (get(&regs, 4 /*CU_FUNC_ATTRIBUTE_NUM_REGS*/, fn) != 0) return -1;
  if (max_threads) get(max_threads, 0 /*MAX_THREADS_PER_BLOCK*/, fn);
  return regs;
#else
  (void)fn; (void)max_threads; return -1;
#endif
}

#include "engine/cuda.h"
#include "engine/kernels.h"
#include "q6k_dp4a_kernels.h"

namespace {

std::unique_ptr<cuda::Module> g_mod;
std::unique_ptr<cuda::Module> g_ref;

// Fill a host Q6_K matrix with well-formed random blocks.
void fill_q6k(std::vector<unsigned char>& host, int n_out, int n_in, unsigned seed) {
  const int sb = n_in / 256;
  const size_t row_bytes = static_cast<size_t>(sb) * 210;
  host.resize(row_bytes * static_cast<size_t>(n_out));
  unsigned s = seed;
  auto rnd = [&s]() {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
  };
  for (int r = 0; r < n_out; ++r) {
    unsigned char* p = host.data() + static_cast<size_t>(r) * row_bytes;
    for (int b = 0; b < sb; ++b) {
      unsigned char* q = p + static_cast<size_t>(b) * 210;
      for (int i = 0; i < 192; ++i) q[i] = static_cast<unsigned char>(rnd() & 0xFF);
      for (int i = 192; i < 208; ++i) q[i] = static_cast<unsigned char>((int)(rnd() % 81) - 40);
      const unsigned short dh = 0x1400;   // 2^-10 in fp16
      std::memcpy(q + 208, &dh, 2);
    }
  }
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / den);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  cuda::init();
  const int n_out = argc > 1 ? std::atoi(argv[1]) : 12288;
  const int n_in = argc > 2 ? std::atoi(argv[2]) : 4096;
  if (n_in % 256) {
    std::printf("n_in must be a multiple of 256\n");
    return 1;
  }
  {
    const std::string src = std::string(kernels::kQuantCommon) + kernels::kGemvQ6K +
                            kernels::kGemvQ6K_x4 + kernels::kGemvDp4a;
    g_ref = std::make_unique<cuda::Module>(cuda::compile_ptx(src, "q6k_ref.cu"));
  }
  g_mod = std::make_unique<cuda::Module>(cuda::compile_ptx(
      std::string(kernels::kQuantCommon) + q6kdp4a::kQ6Kdp4a, "q6k_dp4a.cu"));

  const int sb = n_in / 256;
  const size_t row_bytes = static_cast<size_t>(sb) * 210;
  const size_t mat_bytes = row_bytes * static_cast<size_t>(n_out);
  // enough distinct copies that consecutive launches do not hit L2
  const size_t target = 96u << 20;
  const int copies = static_cast<int>(std::min<size_t>(8, std::max<size_t>(1, target / mat_bytes)));
  std::vector<cuda::Buffer> ws;
  std::vector<unsigned char> host;
  fill_q6k(host, n_out, n_in, 12345u);
  for (int c = 0; c < copies; ++c) {
    ws.emplace_back(mat_bytes);
    std::vector<unsigned char> h = host;
    h[0] ^= static_cast<unsigned char>(c);
    ws.back().upload(h.data(), h.size());
  }
  std::printf("matrix %dx%d  %.1f MB x%d copies  (row_bytes %zu)\n", n_out, n_in,
              mat_bytes / 1e6, copies, row_bytes);
  auto wof = [&](int i) { return ws[i % copies].data(); };

  cuda::Buffer x(static_cast<size_t>(n_in) * 4);
  std::vector<float> hx(static_cast<size_t>(n_in));
  for (int i = 0; i < n_in; ++i) hx[i] = std::sin(i * 0.01f) * 0.7f + 0.02f * (i % 13);
  x.upload(hx.data(), hx.size() * 4);

  // reference
  std::vector<float> hy_ref(static_cast<size_t>(n_out));
  {
    cuda::Buffer yref(static_cast<size_t>(n_out) * 4);
    void* fn = g_ref->kernel("gemv_q6k");
    void* pw = wof(0);
    void* px = x.data();
    void* py = yref.data();
    int ni = n_in, no = n_out;
    void* args[] = {&pw, &px, &py, &ni, &no};
    cuda::launch(fn, (n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
    cuda::sync();
    yref.download(hy_ref.data(), hy_ref.size() * 4);
  }

  // int8 activation quantisation (same as the engine's dp4a path)
  cuda::Buffer qx(2 * static_cast<size_t>(n_in)), dx(static_cast<size_t>(n_in) / 16 * 4),
      xs8(static_cast<size_t>(n_in) / 32 * 4);
  {
    void* pxf = x.data();
    void* pq = qx.data();
    void* pd = dx.data();
    void* pxs = xs8.data();
    int nn = n_in;
    void* qargs[] = {&pxf, &pq, &pd, &pxs, &nn};
    cuda::launch(g_ref->kernel("quant_x_i8"), (n_in + 31) / 32, 1, 1, 32, 1, 1, 0, qargs);
    cuda::sync();
  }
  cuda::Buffer ydp(static_cast<size_t>(n_out) * 4);

  const int iters = mat_bytes > (200u << 20) ? 5 : (mat_bytes > (40u << 20) ? 10 : 30);

  // A kernel variant plus the launch geometry it needs.
  struct Variant {
    const char* kern;
    const char* label;
    int split;     // warps per row
    int threads;
  };
  const Variant variants[] = {
      {"gemv_q6k_dp4a", "gemv_q6k_dp4a", 1, 256},
      {"gemv_q6k_dp4a", "gemv_q6k_dp4a_128t", 1, 128},
      {"gemv_q6k_dp4a_lb8", "gemv_q6k_dp4a_lb8", 1, 256},
      {"gemv_q6k_dp4a_lb6", "gemv_q6k_dp4a_lb6", 1, 256},
      {"gemv_q6k_dp4a_sk", "gemv_q6k_dp4a_sk2", 2, 256},
      {"gemv_q6k_dp4a_sk", "gemv_q6k_dp4a_sk4", 4, 256},
      {"gemv_q6k_dp4a_sk", "gemv_q6k_dp4a_sk8", 8, 256},
      {"gemv_q6k_dp4a_sk", "gemv_q6k_dp4a_sk2_128t", 2, 128},
      {"gemv_q6k_dp4a_sk", "gemv_q6k_dp4a_sk4_128t", 4, 128},
  };
  auto run = [&](const Variant& v) {
    if (v.split > 1 && sb % v.split) return;
    void* fn = g_mod->kernel(v.kern);
    const int regs = kernel_regs(fn);
    const int warps = v.threads / 32;
    const int nr = warps / v.split;
    const int grid = (n_out + nr - 1) / nr;
    int sp = v.split;
    // clear the output so a launch that misses rows cannot pass the check
    std::vector<float> zeros(static_cast<size_t>(n_out), 0.f);
    ydp.upload(zeros.data(), zeros.size() * 4);
    {
      void* pw = wof(0);
      void* pq = qx.data();
      void* pdx = dx.data();
      void* py = ydp.data();
      int ni = n_in, no = n_out;
      void* args[] = {&pw, &pq, &pdx, &py, &ni, &no, &sp};
      cuda::launch(fn, grid, 1, 1, v.threads, 1, 1, 0, args);
      cuda::sync();
    }
    std::vector<float> hy(static_cast<size_t>(n_out));
    ydp.download(hy.data(), hy.size() * 4);
    const double err = rel_l2(hy, hy_ref);
    std::vector<double> reps;
    for (int rep = 0; rep < 5; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        void* pw = wof(i);
        void* pq = qx.data();
        void* pdx = dx.data();
        void* py = ydp.data();
        int ni = n_in, no = n_out;
        void* args[] = {&pw, &pq, &pdx, &py, &ni, &no, &sp};
        cuda::launch(fn, grid, 1, 1, v.threads, 1, 1, 0, args);
      }
      cuda::sync();
      reps.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count() /
                     iters);
    }
    const double ms = median(reps);
    std::printf("  %-24s %8.3f ms  %6.1f GB/s  rel_l2 %.2e %s  regs=%d\n", v.label, ms,
                static_cast<double>(mat_bytes) / (ms * 1e6), err,
                err <= 5e-5 ? "OK" : "FAIL", regs);
  };

  // fp32 reference timing
  {
    void* fn = g_ref->kernel("gemv_q6k");
    std::vector<double> reps;
    for (int rep = 0; rep < 5; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        void* pw = wof(i);
        void* px = x.data();
        void* py = ydp.data();
        int ni = n_in, no = n_out;
        void* args[] = {&pw, &px, &py, &ni, &no};
        cuda::launch(fn, (n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
      }
      cuda::sync();
      reps.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count() /
                     iters);
    }
    const double ms = median(reps);
    std::printf("  %-24s %8.3f ms  %6.1f GB/s  (fp32 baseline)\n", "gemv_q6k", ms,
                static_cast<double>(mat_bytes) / (ms * 1e6));
  }

  // the fp32 x4 kernel is what the engine picked for n_out >= 32768 before this change
  if (n_out >= 32768) {
    void* fn = g_ref->kernel("gemv_q6k_x4");
    std::vector<double> reps;
    for (int rep = 0; rep < 5; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        void* pw = wof(i);
        void* px = x.data();
        void* py = ydp.data();
        int ni = n_in, no = n_out;
        void* args[] = {&pw, &px, &py, &ni, &no};
        cuda::launch(fn, (n_out + 31) / 32, 1, 1, 256, 1, 1, 0, args);
      }
      cuda::sync();
      reps.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count() /
                     iters);
    }
    const double ms = median(reps);
    std::printf("  %-24s %8.3f ms  %6.1f GB/s  (fp32 x4, old engine choice)\n", "gemv_q6k_x4", ms,
                static_cast<double>(mat_bytes) / (ms * 1e6));
  }

  for (const auto& v : variants) run(v);

  // HBM read ceiling over the same number of distinct bytes
  {
    void* fn = g_mod->kernel("read_probe");
    const long long n4 = static_cast<long long>(mat_bytes / 16);
    size_t total = 0;
    for (int i = 0; i < iters; ++i) total += static_cast<size_t>(n4);
    std::vector<double> reps;
    for (int rep = 0; rep < 5; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        int* out = reinterpret_cast<int*>(ydp.data());
        void* pp = wof(i);
        void* po = out;
        long long nn = n4;
        void* args[] = {&pp, &po, &nn};
        cuda::launch(fn, 2048, 1, 1, 256, 1, 1, 0, args);
      }
      cuda::sync();
      reps.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count() /
                     iters);
    }
    const double ms = median(reps);
    std::printf("  %-24s %8.3f ms  %6.1f GB/s  (pure read ceiling)\n", "read_probe", ms,
                static_cast<double>(mat_bytes) / (ms * 1e6));
  }
  return 0;
}
