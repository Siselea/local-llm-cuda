// Debug harness for the int8 host row kernels: includes src/cpu_matvec.cpp so the
// anonymous-namespace kernels are reachable, then checks one row at a time against
// a double-precision reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
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
    // valid 6-bit (scale, min) pairs
    for (int sub = 0; sub < 8; ++sub) {
      const unsigned sc = 1 + ((seed + sub * 7) % 32);
      const unsigned mn = 1 + ((seed + sub * 5) % 16);
      // pack into the 12-byte table exactly as scale_min() decodes it
      if (sub < 4) {
        blk[4 + sub] = static_cast<unsigned char>(sc & 63);
        blk[8 + sub] = static_cast<unsigned char>(mn & 63);
      } else {
        blk[4 + sub - 4] |= static_cast<unsigned char>((sc >> 4) & 3) << 6;
        blk[8 + sub - 4] |= static_cast<unsigned char>((mn >> 4) & 3) << 6;
        blk[12 + sub - 4] = static_cast<unsigned char>((sc & 0xF) | ((mn & 0xF) << 4));
      }
    }
    for (int i = 0; i < 128; ++i)
      blk[16 + i] = static_cast<unsigned char>((seed * 37 + i * 13 + b) & 0xFF);
  }
}

void fill_q6k(unsigned char* p, int n_blocks, unsigned seed) {
  for (int b = 0; b < n_blocks; ++b) {
    unsigned char* blk = p + static_cast<std::size_t>(b) * 210;
    for (int i = 0; i < 128; ++i) blk[i] = static_cast<unsigned char>((seed * 29 + i * 11 + b) & 0xFF);
    for (int i = 0; i < 64; ++i) blk[128 + i] = static_cast<unsigned char>((seed * 17 + i * 19) & 0xFF);
    for (int i = 0; i < 16; ++i)
      blk[192 + i] = static_cast<unsigned char>(static_cast<signed char>(1 + ((seed + i) % 9)));
    put_half(blk + 208, 0.02f);
  }
}

// double-precision reference dot for one Q4_K row
double ref_q4k(const unsigned char* row, int n_in, const std::vector<float>& x) {
  double acc = 0.0;
  const int nb = n_in / 256;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
    const float d = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    const unsigned char* scales = p + 4;
    const unsigned char* qs = p + 16;
    for (int e = 0; e < 256; ++e) {
      const int g = e >> 6, half = (e & 63) >> 5, l = e & 31;
      const int sub = g * 2 + half;
      const unsigned char qb = qs[g * 32 + l];
      const int nib = half ? (qb >> 4) : (qb & 0xF);
      int sc = 0, mn = 0;
      cpu::scale_min(sub, scales, &sc, &mn);
      const double w = static_cast<double>(d) * sc * nib - static_cast<double>(dmin) * mn;
      acc += w * static_cast<double>(x[static_cast<std::size_t>(b) * 256 + e]);
    }
  }
  return acc;
}

double ref_q6k(const unsigned char* row, int n_in, const std::vector<float>& x) {
  double acc = 0.0;
  const int nb = n_in / 256;
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 210;
    const float d = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 208));
    const signed char* sc = reinterpret_cast<const signed char*>(p + 192);
    for (int e = 0; e < 256; ++e) {
      const int n2 = e >> 7, r = e & 127, c = r >> 5, l = r & 31;
      const unsigned char* ql = p + n2 * 64;
      const unsigned char* qh = p + 128 + n2 * 32;
      int q;
      if (c == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
      else if (c == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
      else if (c == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
      else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
      const int sci = n2 * 8 + 2 * c + (l >> 4);
      const double w = static_cast<double>(d) * sc[sci] * (q - 32);
      acc += w * static_cast<double>(x[static_cast<std::size_t>(b) * 256 + e]);
    }
  }
  return acc;
}

}  // namespace

int main() {
  const int n_in = 2048;
  std::vector<unsigned char> wq4(static_cast<std::size_t>(512) * (n_in / 256) * 144);
  std::vector<unsigned char> wq6(static_cast<std::size_t>(2048) * (n_in / 256) * 210);
  fill_q4k(wq4.data(), static_cast<int>(wq4.size() / 144), 3);
  fill_q6k(wq6.data(), static_cast<int>(wq6.size() / 210), 5);
  std::vector<float> x(static_cast<std::size_t>(n_in));
  for (int i = 0; i < n_in; ++i) x[static_cast<std::size_t>(i)] = 0.4f * std::sin(0.013f * (i + 1));

  cpu::XQ8 xq;
  cpu::xq_build(x.data(), n_in, &xq);
  for (int b = 0; b < 4; ++b) std::printf("blk %d: d1=%.8g d2=%.8g sx32=%.8g q1[0]=%d q2[0]=%d\n", b, xq.d1[b], xq.d2[b], xq.sx32[b], (int)xq.q1[b*32], (int)xq.q2[b*32]);

  const int nrow = 16;
  double worst_rel4 = 0, worst_rel6 = 0;
  double abs4 = 0, abs6 = 0;
  for (int r = 0; r < nrow; ++r) {
    float y4 = 0.f, y6 = 0.f;
    cpu::avx2_dot_q4k_i8(wq4.data() + static_cast<std::size_t>(r) * (n_in / 256) * 144, xq, n_in, &y4);
    cpu::avx2_dot_q6k_i8(wq6.data() + static_cast<std::size_t>(r) * (n_in / 256) * 210, xq, n_in, &y6);
    const double d4 = ref_q4k(wq4.data() + static_cast<std::size_t>(r) * (n_in / 256) * 144, n_in, x);
    const double d6 = ref_q6k(wq6.data() + static_cast<std::size_t>(r) * (n_in / 256) * 210, n_in, x);
    abs4 = std::max(abs4, std::fabs(d4));
    abs6 = std::max(abs6, std::fabs(d6));
    worst_rel4 = std::max(worst_rel4, std::fabs(static_cast<double>(y4) - d4) / (std::fabs(d4) + 1e-12));
    worst_rel6 = std::max(worst_rel6, std::fabs(static_cast<double>(y6) - d6) / (std::fabs(d6) + 1e-12));
    std::printf("row %2d  q4k i8=%.6f ref=%.6f   q6k i8=%.6f ref=%.6f\n", r, y4, d4, y6, d6);
  }
  std::printf("worst rel err: q4k %.3e  q6k %.3e   (max|ref| q4k %.3f q6k %.3f)\n", worst_rel4,
              worst_rel6, abs4, abs6);

  // also the MT path vs 1-thread reference
  std::vector<float> y1(512), y9(512);
  cpu::matvec_mt("Q4_K", wq4.data(), 512, n_in, x.data(), y1.data());
  cpu::matvec_mt("Q4_K", wq4.data(), 512, n_in, x.data(), y9.data());
  double md = 0;
  for (int r = 0; r < 512; ++r) {
    const double d = ref_q4k(wq4.data() + static_cast<std::size_t>(r) * (n_in / 256) * 144, n_in, x);
    md = std::max(md, std::fabs(y1[static_cast<std::size_t>(r)] - d));
  }
  std::printf("matvec_mt Q4_K max|y-ref| = %.3e  (repeat diff %.3e)\n", md,
              [&] { double d = 0; for (int r = 0; r < 512; ++r) d = std::max(d, std::fabs((double)y1[r] - y9[r])); return d; }());
  return 0;
}
