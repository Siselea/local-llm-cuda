// Minimal reproducer: copy of the int8 Q4_K inner loop with prints.
#include <immintrin.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

static float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1Fu;
  std::uint32_t mant = h & 0x3FFu;
  if (exp == 0) {
    if (mant == 0) {
      const std::uint32_t bits = sign;
      float f;
      std::memcpy(&f, &bits, 4);
      return f;
    }
    float v = static_cast<float>(mant) * 5.9604644775390625e-08f;
    return sign ? -v : v;
  }
  if (exp == 31) {
    const std::uint32_t bits = sign | 0x7F800000u | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  exp += 112;
  const std::uint32_t bits = sign | (exp << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

static void put_half(unsigned char* p, float v) {
  unsigned int bits;
  std::memcpy(&bits, &v, 4);
  const unsigned int sign = (bits >> 16) & 0x8000u;
  const int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  const unsigned int mant = (bits >> 13) & 0x3FFu;
  const unsigned int h = sign | (static_cast<unsigned int>(exp + 15) << 10) | mant;
  p[0] = static_cast<unsigned char>(h & 0xFF);
  p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
}

static void scale_min(int sub, const unsigned char* s, int* sc, int* mn) {
  if (sub < 4) {
    *sc = s[sub] & 63;
    *mn = s[sub + 4] & 63;
  } else {
    *sc = (s[sub + 4] & 0xF) | ((s[sub - 4] >> 2) & 0x30);
    *mn = (s[sub + 4] >> 4) | ((s[sub] >> 2) & 0x30);
  }
}

static float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

int main() {
  const int n_in = 256;
  std::vector<unsigned char> row(144);
  put_half(row.data() + 0, 0.015625f);
  put_half(row.data() + 2, 0.015625f);
  for (int sub = 0; sub < 8; ++sub) {
    const unsigned sc = 1 + sub * 7, mn = 1 + sub * 5;
    if (sub < 4) {
      row[4 + sub] = static_cast<unsigned char>(sc & 63);
      row[8 + sub] = static_cast<unsigned char>(mn & 63);
    } else {
      row[4 + sub - 4] |= static_cast<unsigned char>((sc >> 4) & 3) << 6;
      row[8 + sub - 4] |= static_cast<unsigned char>((mn >> 4) & 3) << 6;
      row[12 + sub - 4] = static_cast<unsigned char>((sc & 0xF) | ((mn & 0xF) << 4));
    }
  }
  for (int i = 0; i < 128; ++i) row[16 + i] = static_cast<unsigned char>((i * 37) & 0xFF);

  std::vector<float> x(n_in);
  for (int i = 0; i < n_in; ++i) x[i] = 0.4f * std::sin(0.013f * (i + 1));

  // ---- two-level quantisation, 32-value blocks
  std::vector<signed char> q1(n_in), q2(n_in);
  std::vector<float> d1(8), d2(8), sx32(8);
  for (int b = 0; b < 8; ++b) {
    const float* xb = x.data() + b * 32;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
    const float d = amax / 127.f;
    const float inv = d > 0.f ? 1.f / d : 0.f;
    float r[32];
    int s1 = 0;
    for (int i = 0; i < 32; ++i) {
      int q = static_cast<int>(std::lrint(xb[i] * inv));
      q = std::min(127, std::max(-127, q));
      q1[b * 32 + i] = static_cast<signed char>(q);
      s1 += q;
      r[i] = xb[i] - d * q;
    }
    float amax2 = 0.f;
    for (int i = 0; i < 32; ++i) amax2 = std::max(amax2, std::fabs(r[i]));
    const float d2b = amax2 / 127.f;
    const float inv2 = d2b > 0.f ? 1.f / d2b : 0.f;
    int s2 = 0;
    for (int i = 0; i < 32; ++i) {
      int q = static_cast<int>(std::lrint(r[i] * inv2));
      q = std::min(127, std::max(-127, q));
      q2[b * 32 + i] = static_cast<signed char>(q);
      s2 += q;
    }
    d1[b] = d;
    d2[b] = d2b;
    sx32[b] = d * s1 + d2b * s2;
  }

  // ---- kernel copy
  const unsigned char* p = row.data();
  const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
  const float dmin = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
  const unsigned char* scales = p + 4;
  const unsigned char* qs = p + 16;
  float scf[8], mnf[8];
  for (int sub = 0; sub < 8; ++sub) {
    int sc = 0, mn = 0;
    scale_min(sub, scales, &sc, &mn);
    scf[sub] = d * static_cast<float>(sc);
    mnf[sub] = dmin * static_cast<float>(mn);
  }
  const __m256i m0f = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc = _mm256_setzero_ps();
  float bias = 0.f;
  for (int g = 0; g < 4; ++g) {
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + g * 32));
    const __m256i wlo = _mm256_and_si256(raw, m0f);
    const __m256i whi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f);
    for (int half = 0; half < 2; ++half) {
      const int sub = g * 2 + half;
      const __m256i w = half ? whi : wlo;
      const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q1.data() + sub * 32));
      const __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q2.data() + sub * 32));
      const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x1), ones);
      const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x2), ones);
      int32_t P1[8], P2[8];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(P1), p1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(P2), p2);
      int sP1 = 0, sP2 = 0;
      for (int i = 0; i < 8; ++i) { sP1 += P1[i]; sP2 += P2[i]; }
      // reference for this 32-value group
      double r1 = 0.0, r2 = 0.0;
      for (int e = 0; e < 32; ++e) {
        const int elem = sub * 32 + e;
        const unsigned char qb = qs[g * 32 + (elem & 31)];
        const int nib = half ? (qb >> 4) : (qb & 0xF);
        r1 += static_cast<double>(nib) * q1[elem];
      }
      const float c1 = scf[sub] * d1[sub];
      const float c2 = scf[sub] * d2[sub];
      printf("sub %d: sP1=%d ref1=%.0f  d1=%.6f scf=%.6f c1=%.6f  nan_in_c1=%d\n", sub, sP1, r1,
             d1[sub], scf[sub], c1, std::isnan(c1) ? 1 : 0);
      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(c1), acc);
      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p2), _mm256_set1_ps(c2), acc);
      bias -= mnf[sub] * sx32[sub];
      printf("   acc[0..3]=%.6f %.6f %.6f %.6f  bias=%.6f\n", acc[0], acc[1], acc[2], acc[3], bias);
      (void)sP2; (void)r2;
    }
  }
  const float y = hsum256(acc) + bias;
  printf("y = %.6f\n", y);
  return 0;
}
