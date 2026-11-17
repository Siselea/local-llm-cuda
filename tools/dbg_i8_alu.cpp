// Isolate the ALU cost of the row kernels: call one row dot repeatedly with the row
// (and activation) hot in L1/L2, so the loop measures instruction throughput, not DRAM.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#include "../src/cpu_matvec.cpp"

namespace {
void put_half(unsigned char* p, float v) {
  unsigned int bits;
  std::memcpy(&bits, &v, 4);
  const unsigned int sign = (bits >> 16) & 0x8000u;
  const int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  const unsigned int mant = (bits >> 13) & 0x3FFu;
  const unsigned int h = sign | (static_cast<unsigned int>(exp + 15) << 10) | mant;
  p[0] = static_cast<unsigned char>(h & 0xFF);
  p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
}
void fill_q4k(unsigned char* p, int n_blocks, unsigned seed) {
  for (int b = 0; b < n_blocks; ++b) {
    unsigned char* blk = p + static_cast<std::size_t>(b) * 144;
    put_half(blk + 0, 0.015625f);
    put_half(blk + 2, 0.015625f);
    for (int i = 0; i < 12; ++i) blk[4 + i] = static_cast<unsigned char>((seed * 13 + i * 7) & 0x3F);
    for (int i = 0; i < 128; ++i)
      blk[16 + i] = static_cast<unsigned char>((seed * 37 + i * 13 + b) & 0xFF);
  }
}
void fill_q6k(unsigned char* p, int n_blocks, unsigned seed) {
  for (int b = 0; b < n_blocks; ++b) {
    unsigned char* blk = p + static_cast<std::size_t>(b) * 210;
    for (int i = 0; i < 128; ++i) blk[i] = static_cast<unsigned char>((seed * 29 + i * 11 + b) & 0xFF);
    for (int i = 0; i < 64; ++i) blk[128 + i] = static_cast<unsigned char>((seed * 17 + i * 19) & 0xFF);
    for (int i = 0; i < 16; ++i) blk[192 + i] = static_cast<unsigned char>(static_cast<signed char>(1 + ((seed + i) % 9)));
    put_half(blk + 208, 0.02f);
  }
}
double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main() {
  const int n_in = 2048;
  std::vector<unsigned char> wq4(static_cast<std::size_t>(2) * (n_in / 256) * 144);
  std::vector<unsigned char> wq6(static_cast<std::size_t>(2) * (n_in / 256) * 210);
  fill_q4k(wq4.data(), static_cast<int>(wq4.size() / 144), 3);
  fill_q6k(wq6.data(), static_cast<int>(wq6.size() / 210), 5);
  std::vector<float> x(static_cast<std::size_t>(n_in));
  for (int i = 0; i < n_in; ++i) x[static_cast<std::size_t>(i)] = 0.4f * std::sin(0.013f * (i + 1));
  cpu::XQ8 xq;
  cpu::xq_build(x.data(), n_in, &xq);

  const int iters = 200000;
  volatile float sink = 0.f;
  double t0, t1;

  // fp32 kernels
  t0 = now_ms();
  for (int it = 0; it < iters; ++it) {
    float y;
    cpu::avx2_dot_q4k(wq4.data() + (it & 1) * (n_in / 256) * 144, x.data(), n_in, &y);
    sink = y;
  }
  t1 = now_ms();
  std::printf("fp32 Q4_K row: %.1f ns/row  => %.2f elem/ns => %.2f GB/s\n", (t1 - t0) * 1e6 / iters,
              n_in / ((t1 - t0) * 1e6 / iters), (n_in / 256.0) * 144 / ((t1 - t0) * 1e6 / iters));
  t0 = now_ms();
  for (int it = 0; it < iters; ++it) {
    float y;
    cpu::avx2_dot_q6k(wq6.data() + (it & 1) * (n_in / 256) * 210, x.data(), n_in, &y);
    sink = y;
  }
  t1 = now_ms();
  std::printf("fp32 Q6_K row: %.1f ns/row  => %.2f elem/ns\n", (t1 - t0) * 1e6 / iters,
              n_in / ((t1 - t0) * 1e6 / iters));
  // int8 kernels
  t0 = now_ms();
  for (int it = 0; it < iters; ++it) {
    float y;
    cpu::avx2_dot_q4k_i8(wq4.data() + (it & 1) * (n_in / 256) * 144, xq, n_in, &y);
    sink = y;
  }
  t1 = now_ms();
  std::printf("int8 Q4_K row: %.1f ns/row  => %.2f elem/ns\n", (t1 - t0) * 1e6 / iters,
              n_in / ((t1 - t0) * 1e6 / iters));
  t0 = now_ms();
  for (int it = 0; it < iters; ++it) {
    float y;
    cpu::avx2_dot_q6k_i8(wq6.data() + (it & 1) * (n_in / 256) * 210, xq, n_in, &y);
    sink = y;
  }
  t1 = now_ms();
  std::printf("int8 Q6_K row: %.1f ns/row  => %.2f elem/ns\n", (t1 - t0) * 1e6 / iters,
              n_in / ((t1 - t0) * 1e6 / iters));
  std::printf("(sink %.6f)\n", sink);
  // xq_build cost
  const int qiters = 20000;
  t0 = now_ms();
  for (int it = 0; it < qiters; ++it) {
    x[static_cast<std::size_t>(it & 2047)] += 1e-9f;
    cpu::xq_build(x.data(), n_in, &xq);
  }
  t1 = now_ms();
  std::printf("xq_build(2048): %.1f ns\n", (t1 - t0) * 1e6 / qiters);
  return 0;
}
