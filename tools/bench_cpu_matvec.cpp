// Micro-benchmark for the CPU quantized matvec used by the CPU-resident MoE experts.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/cpu_matvec.h"

int main(int argc, char** argv) {
  const std::string type = argc > 1 ? argv[1] : "Q4_K";
  const int n_out = argc > 2 ? std::atoi(argv[2]) : 2048;
  const int n_in = argc > 3 ? std::atoi(argv[3]) : 2048;
  const int iters = argc > 4 ? std::atoi(argv[4]) : 50;

  size_t row_bytes = 0;
  if (type == "Q4_K") row_bytes = static_cast<size_t>(n_in / 256) * 144;
  else if (type == "Q5_K") row_bytes = static_cast<size_t>(n_in / 256) * 176;
  else if (type == "Q6_K") row_bytes = static_cast<size_t>(n_in / 256) * 210;
  else if (type == "Q8_0") row_bytes = static_cast<size_t>(n_in / 32) * 34;
  else if (type == "Q5_1") row_bytes = static_cast<size_t>(n_in / 32) * 24;
  else { std::fprintf(stderr, "unsupported type %s\n", type.c_str()); return 1; }

  std::vector<unsigned char> w(row_bytes * static_cast<size_t>(n_out));
  // fill with a plausible pattern (small scales, non-zero quants)
  for (size_t i = 0; i < w.size(); ++i) w[i] = static_cast<unsigned char>((i * 37u) & 0xFF);
  for (int r = 0; r < n_out; ++r) {
    unsigned char* p = w.data() + static_cast<size_t>(r) * row_bytes;
    for (int b = 0; b * 256 < n_in; ++b) p[b * 144] = 0x00;   // d = 0 for Q4_K blocks
  }
  std::vector<float> x(static_cast<size_t>(n_in), 0.01f);
  std::vector<float> y(static_cast<size_t>(n_out), 0.f);

  cpu::matvec_mt(type, w.data(), n_out, n_in, x.data(), y.data());   // warm up
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it)
    cpu::matvec_mt(type, w.data(), n_out, n_in, x.data(), y.data());
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  const double gmac = static_cast<double>(n_out) * n_in / (ms * 1e6);
  const double gbs = static_cast<double>(w.size()) / (ms * 1e6);

  std::printf("%s %dx%d: %.3f ms/row-block, %.2f GMAC/s, %.2f GB/s (1 thread)\n",
              type.c_str(), n_out, n_in, ms, gmac, gbs);
  return 0;
}
