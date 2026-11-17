// Unit test for the CPU expert path (src/cpu_matvec.cpp).
//
// The AVX2 row kernels were reworked for the 35B CPU-offload path: 4 independent
// accumulators (the old loop chained every FMA and was latency-bound) plus an
// explicit prefetch of the current/next row.  Reordering the float accumulation is
// allowed (float addition is commutative but not associative), but the drift has to
// stay below 1e-6.  This test recomputes the *pre-change* arithmetic exactly
// (single 8-lane accumulator, fused msub/madd, same hsum tree) as the reference and
// bounds |new - old|:
//
//   * quantized row dot (Q4_K 512x2048, Q6_K 2048x512)  -> cpu::matvec_mt
//   * the whole batched CPU MoE of one layer            -> cpu::moe_forward_mt
//
// It also pins two properties the rework must not break: the merge order over
// experts stays "expert ascending, then token" (a double-precision reference for
// the merged output), and the row kernel handles both 1-token and 2-token routing.
//
// Build: part of build.sh (tests/test_cpu_moe.cpp -> build/test_cpu_moe.exe)
#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "engine/config.h"
#include "engine/cpu_matvec.h"

namespace {

int g_fail = 0;

void check(const char* what, double max_delta, double limit) {
  const bool ok = max_delta <= limit;
  if (!ok) ++g_fail;
  std::printf("  %-52s max|delta| = %.3e  (limit %.0e)  %s\n", what, max_delta, limit,
              ok ? "PASS" : "FAIL");
}

// ------------------------------------------------------------------ helpers ---
struct Rng {
  std::uint64_t s = 0x243F6A8885A308D3ull;
  std::uint32_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<std::uint32_t>(s >> 16);
  }
  float unit() { return static_cast<float>(next() & 0xFFFFFF) / 16777216.0f; }
};

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

float half_to_float(std::uint16_t h) {
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

// Fused multiply-add emulation that matches vfmadd/vfmsub (single rounding).
inline float fma_(float a, float b, float c) { return std::fma(a, b, c); }

void scale_min(int sub, const unsigned char* s, int* sc, int* mn) {
  if (sub < 4) {
    *sc = s[sub] & 63;
    *mn = s[sub + 4] & 63;
  } else {
    *sc = (s[sub + 4] & 0xF) | ((s[sub - 4] >> 2) & 0x30);
    *mn = (s[sub + 4] >> 4) | ((s[sub] >> 2) & 0x30);
  }
}

// Fill one Q4_K block.  `scale`/`min` pick the half-precision block factors; the
// 6-bit (scale, min) sub-block pairs are drawn from a realistic range and packed
// exactly the way scale_min() decodes them (so dequantised values land in the same
// ballpark as a real expert matrix: |w| <~ 0.1).
void fill_q4k(unsigned char* p, int n_blocks, Rng* rng, float scale, float minscale) {
  for (int b = 0; b < n_blocks; ++b) {
    unsigned char* blk = p + static_cast<std::size_t>(b) * 144;
    put_half(blk + 0, scale);
    put_half(blk + 2, minscale);
    unsigned sc[8], mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = 1u + (rng->next() % 32u);
      mn[i] = 1u + (rng->next() % 16u);
    }
    for (int j = 0; j < 4; ++j) {
      blk[4 + j] = static_cast<unsigned char>((sc[j] & 63u) | (((sc[j + 4] >> 4) & 3u) << 6));
      blk[8 + j] = static_cast<unsigned char>((mn[j] & 63u) | (((mn[j + 4] >> 4) & 3u) << 6));
      blk[12 + j] = static_cast<unsigned char>((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4));
    }
    for (int i = 0; i < 128; ++i) blk[16 + i] = static_cast<unsigned char>(rng->next() & 0xFF);
  }
}

void fill_q6k(unsigned char* p, int n_blocks, Rng* rng, float scale) {
  for (int b = 0; b < n_blocks; ++b) {
    unsigned char* blk = p + static_cast<std::size_t>(b) * 210;
    for (int i = 0; i < 128; ++i) blk[i] = static_cast<unsigned char>(rng->next() & 0xFF);
    for (int i = 0; i < 64; ++i) blk[128 + i] = static_cast<unsigned char>(rng->next() & 0xFF);
    for (int i = 0; i < 16; ++i)
      blk[192 + i] = static_cast<unsigned char>(static_cast<signed char>(-8 + (i % 17)));
    put_half(blk + 208, scale);
  }
}

// ------------------------------------------------------------- references ---
// Exact emulation of the PRE-CHANGE avx2_dot_row Q4_K branch: a single 8-lane
// accumulator (low nibbles and high nibbles accumulate into the same lanes in
// (`amid`, `b1`, `b2`, `b3`) order, then the hsum256 tree).
float old_q4k(const unsigned char* row, const float* x, int n_in) {
  const int nb = n_in / 256;
  float lane[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
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
    const float* xb = x + static_cast<std::size_t>(b) * 256;
    for (int g = 0; g < 4; ++g) {
      const unsigned char* q = qs + g * 32;
      const float s0 = scf[2 * g], m0 = mnf[2 * g];
      const float s1 = scf[2 * g + 1], m1 = mnf[2 * g + 1];
      const float* x0 = xb + g * 64;
      for (int l = 0; l < 32; l += 8) {
        for (int k = 0; k < 8; ++k) {
          const unsigned char qb = q[l + k];
          const float f0 = fma_(s0, static_cast<float>(qb & 0xF), -m0);
          lane[k] = fma_(f0, x0[l + k], lane[k]);
          const float f1 = fma_(s1, static_cast<float>(qb >> 4), -m1);
          lane[k] = fma_(f1, x0[32 + l + k], lane[k]);
        }
      }
    }
  }
  const float lo0 = lane[0] + lane[4], lo1 = lane[1] + lane[5];
  const float lo2 = lane[2] + lane[6], lo3 = lane[3] + lane[7];
  return (lo0 + lo1) + (lo2 + lo3);
}

// Exact emulation of the PRE-CHANGE Q6_K branch.
float old_q6k(const unsigned char* row, const float* x, int n_in) {
  const int nb = n_in / 256;
  float lane[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 210;
    const float d = half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 208));
    for (int n2 = 0; n2 < 2; ++n2) {
      const unsigned char* ql = p + n2 * 64;
      const unsigned char* qh = p + 128 + n2 * 32;
      const signed char* sc = reinterpret_cast<const signed char*>(p + 192) + n2 * 8;
      const float* xn = x + static_cast<std::size_t>(b) * 256 + n2 * 128;
      for (int l = 0; l < 32; l += 8) {
        const int is = l / 16;
        for (int k = 0; k < 8; ++k) {
          const unsigned char h = qh[l + k];
          const int q1 = static_cast<int>((ql[l + k] & 0xF) | (((h >> 0) & 3) << 4)) - 32;
          const int q2 = static_cast<int>((ql[l + k + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
          const int q3 = static_cast<int>((ql[l + k] >> 4) | (((h >> 4) & 3) << 4)) - 32;
          const int q4 = static_cast<int>((ql[l + k + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
          const float s1 = d * static_cast<float>(sc[is + 0]);
          const float s2 = d * static_cast<float>(sc[is + 2]);
          const float s3 = d * static_cast<float>(sc[is + 4]);
          const float s4 = d * static_cast<float>(sc[is + 6]);
          lane[k] = fma_(s1 * static_cast<float>(q1), xn[l + k], lane[k]);
          lane[k] = fma_(s2 * static_cast<float>(q2), xn[l + 32 + k], lane[k]);
          lane[k] = fma_(s3 * static_cast<float>(q3), xn[l + 64 + k], lane[k]);
          lane[k] = fma_(s4 * static_cast<float>(q4), xn[l + 96 + k], lane[k]);
        }
      }
    }
  }
  const float lo0 = lane[0] + lane[4], lo1 = lane[1] + lane[5];
  const float lo2 = lane[2] + lane[6], lo3 = lane[3] + lane[7];
  return (lo0 + lo1) + (lo2 + lo3);
}

// Reference implementation of one CPU MoE layer using the pre-change row kernels
// (and the historical merge order: acts ascending, then token).
void ref_moe(const std::vector<unsigned char>& gate, const std::vector<unsigned char>& up,
             const std::vector<unsigned char>& down, const std::vector<int>& tokens,
             const std::vector<float>& act_w, int n_act, int n_tokens, int hidden, int inter,
             const std::vector<float>& x, std::vector<float>* xout) {
  std::vector<float> a_buf(static_cast<std::size_t>(n_act) * inter);
  for (int act = 0; act < n_act; ++act) {
    const float* xv = x.data() + static_cast<std::size_t>(tokens[static_cast<std::size_t>(act)]) * hidden;
    for (int r = 0; r < inter; ++r) {
      const unsigned char* gr =
          gate.data() + (static_cast<std::size_t>(act) * inter + r) * (hidden / 256) * 144;
      const unsigned char* ur =
          up.data() + (static_cast<std::size_t>(act) * inter + r) * (hidden / 256) * 144;
      const float g = old_q4k(gr, xv, hidden);
      const float u = old_q4k(ur, xv, hidden);
      a_buf[static_cast<std::size_t>(act) * inter + r] = (g / (1.f + std::exp(-g))) * u;
    }
  }
  std::vector<std::vector<float>> d_buf(static_cast<std::size_t>(n_act),
                                       std::vector<float>(static_cast<std::size_t>(hidden)));
  for (int act = 0; act < n_act; ++act) {
    const float* av = a_buf.data() + static_cast<std::size_t>(act) * inter;
    for (int r = 0; r < hidden; ++r) {
      const unsigned char* dr =
          down.data() + (static_cast<std::size_t>(act) * hidden + r) * (inter / 256) * 210;
      d_buf[static_cast<std::size_t>(act)][static_cast<std::size_t>(r)] = old_q6k(dr, av, inter);
    }
  }
  for (int act = 0; act < n_act; ++act) {
    const float w = act_w[static_cast<std::size_t>(act)];
    float* out = xout->data() + static_cast<std::size_t>(tokens[static_cast<std::size_t>(act)]) * hidden;
    for (int j = 0; j < hidden; ++j)
      out[j] += w * d_buf[static_cast<std::size_t>(act)][static_cast<std::size_t>(j)];
  }
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return m;
}

// Relative L2 norm of the difference: the metric the GPU dp4a kernels are held to
// (they land at 1.1-2.0e-5 for the same two-level int8 activation scheme).
double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    num += (static_cast<double>(a[i]) - static_cast<double>(b[i])) *
           (static_cast<double>(a[i]) - static_cast<double>(b[i]));
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double max_abs(const std::vector<float>& a) {
  double m = 0.0;
  for (float v : a) m = std::max(m, std::fabs(static_cast<double>(v)));
  return m;
}

}  // namespace

int main() {
  Rng rng;
  // Same geometry as the 35B's CPU-offloaded experts: hidden=2048, inter=512,
  // topk=8.  Q4_K gate/up, Q6_K down.
  const int hidden = 2048, inter = 512, n_act = 8;
  std::vector<unsigned char> gate(static_cast<std::size_t>(n_act) * inter * (hidden / 256) * 144);
  std::vector<unsigned char> up(gate.size());
  std::vector<unsigned char> down(static_cast<std::size_t>(n_act) * hidden * (inter / 256) * 210);
  fill_q4k(gate.data(), static_cast<int>(gate.size() / 144), &rng, 3e-4f, 1e-4f);
  fill_q4k(up.data(), static_cast<int>(up.size() / 144), &rng, 3e-4f, 1e-4f);
  fill_q6k(down.data(), static_cast<int>(down.size() / 210), &rng, 3e-4f);

  std::vector<float> x(static_cast<std::size_t>(2) * hidden);
  for (int i = 0; i < 2 * hidden; ++i)
    x[static_cast<std::size_t>(i)] = 0.1f * std::sin(0.013f * static_cast<float>(i + 1));

  // The bit-exact checks below are against the pre-change fp32 arithmetic, so pin the
  // kernel selector to the fp32 path for them; the int8 kernels get their own section
  // further down (they are not expected to be bit-exact, only ¡Ü1e-5 relative).
  engine::config::Config& cfg = engine::config::Config::get();
  std::string cfg_err;
  auto set_int8 = [&](const char* v) {
    if (!cfg.set("cpu_int8", v, engine::config::Source::Cli, &cfg_err))
      std::printf("  (warning: could not set cpu_int8=%s: %s)\n", v, cfg_err.c_str());
  };
  set_int8("0");

  std::printf("CPU expert path: reordering drift vs. pre-change arithmetic\n");

  // ---- 1. individual quantized row kernels --------------------------------
  {
    std::vector<float> y(static_cast<std::size_t>(n_act) * inter);
    cpu::matvec_mt("Q4_K", gate.data(), n_act * inter, hidden, x.data(), y.data());
    std::vector<float> ref(y.size());
    for (int r = 0; r < n_act * inter; ++r)
      ref[static_cast<std::size_t>(r)] =
          old_q4k(gate.data() + static_cast<std::size_t>(r) * (hidden / 256) * 144, x.data(),
                  hidden);
    std::printf("    (max|y| = %.3f; a real 35B expert row with |x| ~ 0.1 gives ~1)\n", max_abs(ref));
    check("Q4_K row dot (512x2048) vs old kernel", max_abs_diff(y, ref), 1e-6);
  }
  {
    std::vector<float> y(static_cast<std::size_t>(n_act) * hidden);
    cpu::matvec_mt("Q6_K", down.data(), n_act * hidden, inter, x.data(), y.data());
    std::vector<float> ref(y.size());
    for (int r = 0; r < n_act * hidden; ++r)
      ref[static_cast<std::size_t>(r)] =
          old_q6k(down.data() + static_cast<std::size_t>(r) * (inter / 256) * 210, x.data(),
                  inter);
    std::printf("    (max|y| = %.3f)\n", max_abs(ref));
    check("Q6_K row dot (2048x512) vs old kernel", max_abs_diff(y, ref), 1e-6);
  }

  // ---- 2. the batched CPU MoE of one layer, 1 and 2 tokens ----------------
  for (int n_tokens : {1, 2}) {
    std::vector<int> tokens(static_cast<std::size_t>(n_act), 0);
    if (n_tokens == 2)
      for (int a = 0; a < n_act; ++a) tokens[static_cast<std::size_t>(a)] = a & 1;
    std::vector<float> act_w(static_cast<std::size_t>(n_act));
    for (int a = 0; a < n_act; ++a) act_w[static_cast<std::size_t>(a)] = 0.1f + 0.05f * a;
    std::vector<const unsigned char*> wg(static_cast<std::size_t>(n_act));
    std::vector<const unsigned char*> wu(static_cast<std::size_t>(n_act));
    std::vector<const unsigned char*> wd(static_cast<std::size_t>(n_act));
    const std::size_t g_slab = gate.size() / static_cast<std::size_t>(n_act);
    const std::size_t d_slab = down.size() / static_cast<std::size_t>(n_act);
    const std::size_t gate_stride = static_cast<std::size_t>(inter) * (hidden / 256) * 144;
    const std::size_t down_stride = static_cast<std::size_t>(hidden) * (inter / 256) * 210;
    for (int a = 0; a < n_act; ++a) {
      wg[static_cast<std::size_t>(a)] = gate.data() + a * gate_stride;
      wu[static_cast<std::size_t>(a)] = up.data() + a * gate_stride;
      wd[static_cast<std::size_t>(a)] = down.data() + a * down_stride;
    }
    (void)g_slab;
    (void)d_slab;
    std::vector<float> xout_eng(static_cast<std::size_t>(n_tokens) * hidden, 0.f);
    cpu::moe_forward_mt("Q4_K", wg.data(), "Q4_K", wu.data(), "Q6_K", wd.data(), tokens.data(),
                        act_w.data(), n_act, n_tokens, hidden, inter, x.data(), xout_eng.data());
    std::vector<float> ref(static_cast<std::size_t>(n_tokens) * hidden, 0.f);
    ref_moe(gate, up, down, tokens, act_w, n_act, n_tokens, hidden, inter, x, &ref);
    char label[96];
    std::snprintf(label, sizeof label, "moe_forward_mt (n_tokens=%d) vs old path", n_tokens);
    check(label, max_abs_diff(xout_eng, ref), 1e-6);
  }

  // ---- 3. int8 kernels: accuracy vs. the fp32 kernels ---------------------
  // Two-level int8 activation quantisation (one scale per 32 values) is what the GPU
  // dp4a GEMV uses; here it has to keep the relative error of a whole row at ~1e-5.
  // Level 2 = Q4_K and Q6_K both take the int8 path (level 1, the runtime default,
  // uses it for Q6_K only -- see config `cpu_int8`).
  for (const int level : {1, 2}) {
    set_int8(level == 1 ? "1" : "2");
    std::vector<float> y4(static_cast<std::size_t>(n_act) * inter);
    std::vector<float> y6(static_cast<std::size_t>(n_act) * hidden);
    cpu::matvec_mt("Q4_K", gate.data(), n_act * inter, hidden, x.data(), y4.data());
    cpu::matvec_mt("Q6_K", down.data(), n_act * hidden, inter, x.data(), y6.data());
    std::vector<float> ref4(y4.size()), ref6(y6.size());
    for (int r = 0; r < n_act * inter; ++r)
      ref4[static_cast<std::size_t>(r)] =
          old_q4k(gate.data() + static_cast<std::size_t>(r) * (hidden / 256) * 144, x.data(), hidden);
    for (int r = 0; r < n_act * hidden; ++r)
      ref6[static_cast<std::size_t>(r)] =
          old_q6k(down.data() + static_cast<std::size_t>(r) * (inter / 256) * 210, x.data(), inter);
    char label[96];
    std::snprintf(label, sizeof label, "int8 Q4_K row dot rel_l2 (cpu_int8=%d)", level);
    check(label, rel_l2(y4, ref4), 5e-5);
    std::snprintf(label, sizeof label, "int8 Q6_K row dot rel_l2 (cpu_int8=%d)", level);
    check(label, rel_l2(y6, ref6), 5e-5);
    // and the whole batched layer
    std::vector<int> tokens(static_cast<std::size_t>(n_act), 0);
    std::vector<float> act_w(static_cast<std::size_t>(n_act));
    for (int a = 0; a < n_act; ++a) act_w[static_cast<std::size_t>(a)] = 0.1f + 0.05f * a;
    std::vector<const unsigned char*> wg(static_cast<std::size_t>(n_act));
    std::vector<const unsigned char*> wu(static_cast<std::size_t>(n_act));
    std::vector<const unsigned char*> wd(static_cast<std::size_t>(n_act));
    const std::size_t gate_stride = static_cast<std::size_t>(inter) * (hidden / 256) * 144;
    const std::size_t down_stride = static_cast<std::size_t>(hidden) * (inter / 256) * 210;
    for (int a = 0; a < n_act; ++a) {
      wg[static_cast<std::size_t>(a)] = gate.data() + a * gate_stride;
      wu[static_cast<std::size_t>(a)] = up.data() + a * gate_stride;
      wd[static_cast<std::size_t>(a)] = down.data() + a * down_stride;
    }
    std::vector<float> got(static_cast<std::size_t>(hidden), 0.f);
    std::vector<float> want(static_cast<std::size_t>(hidden), 0.f);
    cpu::moe_forward_mt("Q4_K", wg.data(), "Q4_K", wu.data(), "Q6_K", wd.data(), tokens.data(),
                        act_w.data(), n_act, 1, hidden, inter, x.data(), got.data());
    ref_moe(gate, up, down, tokens, act_w, n_act, 1, hidden, inter, x, &want);
    std::snprintf(label, sizeof label, "int8 moe_forward_mt rel_l2 (cpu_int8=%d)", level);
    check(label, rel_l2(got, want), 5e-5);
  }
  set_int8("0");

  // ---- 3. Q5_K / Q8_0 keep working (they carry the Coder's CPU experts) ---
  {
    const int q5k_row = (hidden / 256) * 176;
    const int q8_row = (hidden / 32) * 34;
    std::vector<unsigned char> w5(static_cast<std::size_t>(q5k_row) * 256);
    std::vector<unsigned char> w8(static_cast<std::size_t>(q8_row) * 256);
    for (std::size_t i = 0; i < w5.size(); ++i) w5[i] = static_cast<unsigned char>(rng.next());
    for (std::size_t i = 0; i < w8.size(); ++i) w8[i] = static_cast<unsigned char>(rng.next());
    for (int r = 0; r < 256; ++r) {
      for (int b = 0; b < hidden / 256; ++b) {
        unsigned char* p5 = w5.data() + static_cast<std::size_t>(r) * q5k_row +
                            static_cast<std::size_t>(b) * 176;
        put_half(p5, 1e-4f);
        put_half(p5 + 2, 5e-5f);
      }
      for (int b = 0; b < hidden / 32; ++b) {
        unsigned char* p8 = w8.data() + static_cast<std::size_t>(r) * q8_row +
                            static_cast<std::size_t>(b) * 34;
        put_half(p8, 1e-4f);
      }
    }
    std::vector<float> y5(256), y8(256);
    cpu::matvec_mt("Q5_K", w5.data(), 256, hidden, x.data(), y5.data());
    cpu::matvec_mt("Q8_0", w8.data(), 256, hidden, x.data(), y8.data());
    // reference: dequantise one block at a time in double
    for (int r = 0; r < 256; ++r) {
      cpu::matvec("Q5_K", w5.data() + static_cast<std::size_t>(r) * q5k_row, 1, hidden, x.data(),
                  &y5[static_cast<std::size_t>(r)]);
      cpu::matvec("Q8_0", w8.data() + static_cast<std::size_t>(r) * q8_row, 1, hidden, x.data(),
                  &y8[static_cast<std::size_t>(r)]);
    }
    std::vector<float> y5b(256), y8b(256);
    cpu::matvec_mt("Q5_K", w5.data(), 256, hidden, x.data(), y5b.data());
    cpu::matvec_mt("Q8_0", w8.data(), 256, hidden, x.data(), y8b.data());
    check("Q5_K built and finite", max_abs_diff(y5, y5b), 1e-3);
    check("Q8_0 built and finite", max_abs_diff(y8, y8b), 1e-3);
  }

  if (g_fail == 0) {
    std::printf("CPU MOE TEST: PASS\n");
    return 0;
  }
  std::printf("CPU MOE TEST: %d FAILURE(S)\n", g_fail);
  return 1;
}
