// Bandwidth benchmark for the quantized GEMV kernels.
//
// Runs each kernel against a matrix large enough that the working set does not fit
// in L2 (RTX 3060: 3 MB), so the number reported is HBM bandwidth, not cache.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

#include "engine/cuda.h"
#include "engine/kernels.h"

namespace {

struct KSpec {
  const char* name;
  const char* type;
  int block_bytes;   // bytes per row block
  int block_elems;   // elements per row block
  const char* kern;  // kernel name (x4 variant when available)
};

double bench(const std::string& kern_name, const std::string& type, int n_out, int n_in,
             int iters, int threads_in = 256) {
  const int block_bytes = type == "Q4_K" ? 144 : type == "Q5_K" ? 176 : type == "Q6_K" ? 210 : 34;
  const int block_elems = (type == "Q8_0") ? 32 : 256;
  const size_t row_bytes = static_cast<size_t>(n_in / block_elems) * block_bytes;
  cuda::Buffer w(row_bytes * static_cast<size_t>(n_out));
  cuda::Buffer x(static_cast<size_t>(n_in) * 4);
  cuda::Buffer y(static_cast<size_t>(n_out) * 4);
  std::vector<unsigned char> host(w.size());
  for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<unsigned char>((i * 37u) & 0xFF);
  // keep scales finite: write a small fp16 d into each block header
  for (int r = 0; r < n_out; ++r) {
    unsigned char* p = host.data() + static_cast<size_t>(r) * row_bytes;
    for (int b = 0; b * block_elems < n_in; ++b) {
      const unsigned char d[2] = {0x00, 0x2C};   // 0.0625 in fp16
      std::memcpy(p + static_cast<size_t>(b) * block_bytes, d, 2);
    }
  }
  w.upload(host.data(), host.size());
  std::vector<float> hx(static_cast<size_t>(n_in), 0.01f);
  x.upload(hx.data(), hx.size() * 4);

  static std::unique_ptr<cuda::Module> module;
  if (!module) {
    const std::string src = std::string(kernels::kQuantCommon) + kernels::kGemvQ4K +
                            kernels::kGemvQ5K + kernels::kGemvQ8_0 +
                            kernels::kGemvQ6K + kernels::kGemvQ5_0 +
                            kernels::kGemvQ4K_x4 + kernels::kGemvQ5K_x4 +
                            kernels::kGemvQ6K_x4 + kernels::kGemvQ50_x4;
    module = std::make_unique<cuda::Module>(cuda::compile_ptx(src, "bench.cu"));
  }
  void* fn = module->kernel(kern_name);
  void* pw = w.data();
  void* px = x.data();
  void* py = y.data();
  int nin = n_in, nout = n_out;
  void* args[] = {&pw, &px, &py, &nin, &nout};

  const bool x4 = kern_name.find("_x4") != std::string::npos;   // warp sweep below
  const bool splitk = kern_name.find("_k2") != std::string::npos;
  const bool p4 = kern_name.find("_p4") != std::string::npos;
  int threads = threads_in;
  auto launch = [&]() {
    const int rows_per_block = (threads / 32) * 4;
    if (x4 || splitk) cuda::launch(fn, (n_out + rows_per_block - 1) / rows_per_block, 1, 1, threads, 1, 1, 0, args);
    else if (p4) cuda::launch(fn, (n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
    else cuda::launch(fn, (n_out + 7) / 8, 1, 1, 256, 1, 1, 0, args);
  };
  launch();
  cuda::sync();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) launch();
  cuda::sync();
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
      iters;
  return static_cast<double>(w.size()) / (ms * 1e6);   // GB/s
}

}  // namespace

int main() {
  cuda::init();
  const int n_in = 2048;
  const int n_out = 102400; (void)n_out;   // lm_head shape of the Coder
  struct Case {
    const char* type;
    const char* kern;
  } cases[] = {
      {"Q6_K", "gemv_q6k_x4"}, {"Q6_K", "gemv_q6k"},   {"Q5_K", "gemv_q5k_x4"},
      {"Q5_K", "gemv_q5k"},    {"Q4_K", "gemv_q4k_x4"}, {"Q4_K", "gemv_q4k"},
      {"Q8_0", "gemv_q8_0"},
  };
  std::printf("GEMV bandwidth, n_in=%d (HBM-bound) -- warp sweep\n", n_in);
  std::printf(" q5k_x4 12288x4096 (9B FFN up/gate shape):\n");
  for (int th : {128, 256, 512, 1024}) {
    const double gbs = bench("gemv_q5k_x4", "Q5_K", 12288, 4096, 20, th);
    std::printf("   threads=%-5d %6.1f GB/s\n", th, gbs);
  }
  // dp4a path: quantise x to int8 once (two levels: 32 values per scale), then the
  // whole matvec is an int8 dot product.  The "_s" variant stages the activations
  // in shared memory and is the one the engine uses for these row lengths.
  {
    const int ni = 4096, no = 12288;
    const size_t row_bytes = static_cast<size_t>(ni / 256) * 176;
    cuda::Buffer w(row_bytes * static_cast<size_t>(no));
    std::vector<unsigned char> host(w.size());
    for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<unsigned char>((i * 37u) & 0xFF);
    for (int r = 0; r < no; ++r) {
      unsigned char* pp = host.data() + static_cast<size_t>(r) * row_bytes;
      for (int b = 0; b * 256 < ni; ++b) {
        const unsigned char d[2] = {0x00, 0x2C};
        std::memcpy(pp + static_cast<size_t>(b) * 176, d, 2);
      }
    }
    w.upload(host.data(), host.size());
    cuda::Buffer x(static_cast<size_t>(ni) * 4), qxb(2 * static_cast<size_t>(ni)),
        dxb(static_cast<size_t>(ni / 16) * 4), xs8b(static_cast<size_t>(ni / 32) * 4),
        y(static_cast<size_t>(no) * 4);
    std::vector<float> hx(static_cast<size_t>(ni), 0.01f);
    x.upload(hx.data(), hx.size() * 4);
    void* pxf = x.data();
    void* pq = qxb.data();
    void* pdx = dxb.data();
    void* pxs8 = xs8b.data();
    static std::unique_ptr<cuda::Module> mod2;
    if (!mod2) {
      const std::string src2 = std::string(kernels::kQuantCommon) + kernels::kGemvDp4a;
      mod2 = std::make_unique<cuda::Module>(cuda::compile_ptx(src2, "dp4a.cu"));
    }
    int nn = ni;
    void* qargs[] = {&pxf, &pq, &pdx, &pxs8, &nn};
    cuda::launch(mod2->kernel("quant_x_i8"), (ni + 31) / 32, 1, 1, 32, 1, 1, 0, qargs);
    void* pw = w.data();
    void* py = y.data();
    int nout = no;
    void* sargs[] = {&pw, &pq, &pdx, &pxs8, &py, &nn, &nout};
    auto bench_launch = [&](void* fn, size_t shmem) {
      auto launch = [&]() { cuda::launch(fn, (no + 31) / 32, 1, 1, 256, 1, 1, shmem, sargs); };
      launch();
      cuda::sync();
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) launch();
        cuda::sync();
        best = std::min(best, std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count() /
                                  20);
      }
      return best;
    };
    const double ms_g = bench_launch(mod2->kernel("gemv_q5k_x4_dp4a"), 0);
    const double ms_s = bench_launch(mod2->kernel("gemv_q5k_x4_dp4a_s"), 3 * static_cast<size_t>(ni));
    const double bytes = static_cast<double>(w.size());
    std::printf("   dp4a q5k_x4      %6.1f GB/s\n", bytes / (ms_g * 1e6));
    std::printf("   dp4a q5k_x4_shmem%6.1f GB/s  <- engine default\n", bytes / (ms_s * 1e6));
  }
  std::printf(" q6k 8192x4096:\n");
  for (int th : {128, 256, 512}) {
    const double gbs = bench("gemv_q6k", "Q6_K", 8192, 4096, 20, th);
    std::printf("   threads=%-5d %6.1f GB/s\n", th, gbs);
  }
  for (int ny : {4096, 8192, 32768, 102400}) {
    std::printf(" n_out=%d (~%.1f MB Q5_K)\n", ny, ny * (2048.0 / 256 * 176) / 1e6);
    for (const auto& c : cases) {
      const double gbs = bench(c.kern, c.type, ny, n_in, 20);
      std::printf("   %-16s %-6s  %6.1f GB/s\n", c.kern, c.type, gbs);
    }
  }
  return 0;
}
