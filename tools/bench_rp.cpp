// Probe benchmark for the CPU-resident MoE expert path (weight-repack sprint).
//
// Why a second bench: tools/bench_cpu_layer.cpp rotates 24 copies of one layer
// (373 MB, > 10x L3) and times cpu::moe_forward_mt end to end, but this machine
// drifts +/-20% between runs, so an A/B between two binaries measured minutes
// apart cannot resolve a 10% change.  This probe keeps ALL kernel variants inside
// ONE process and interleaves them round by round, so every variant sees the same
// machine state; it reports min / p25 / median.
//
// The production translation unit is #included so the probe benchmarks the exact
// kernels that ship (or the ones being trialled in src/cpu_matvec.cpp), never a copy.
//
// It measures one phase at a time (phase A = gate+up over 8 experts = 9.4 MB;
// phase B = down over 8 experts = 6.9 MB) with the production flat (act,row)
// partitioning, so numbers are comparable to ENGINE_DBG_MOE_PHASES.
//
// usage: bench_rp a|b <copies> <iters> [threads] [variant...]
//   phase A variants: read prod i8
//   phase B variants: read f32 i8
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../src/cpu_matvec.cpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// 35B CPU layer geometry (matches tools/bench_cpu_layer.cpp).
constexpr int kHidden = 2048;
constexpr int kInter = 512;
constexpr int kActs = 8;

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void put_half(unsigned char* p, float v) {
  unsigned int bits;
  std::memcpy(&bits, &v, 4);
  const unsigned int sign = (bits >> 16) & 0x8000u;
  int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  unsigned int mant = (bits >> 13) & 0x3FFu;
  if (exp < -14) {
    const unsigned int h = sign;
    p[0] = static_cast<unsigned char>(h & 0xFF);
    p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
    return;
  }
  const unsigned int h = sign | (static_cast<unsigned int>(exp + 15) << 10) | mant;
  p[0] = static_cast<unsigned char>(h & 0xFF);
  p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
}
static void fill_q4k_block(unsigned char* p) {
  put_half(p + 0, 0.015625f);
  put_half(p + 2, 0.015625f);
  for (int i = 0; i < 12; ++i) p[4 + i] = static_cast<unsigned char>(0x21 + i * 7);
  for (int i = 0; i < 128; ++i) p[16 + i] = static_cast<unsigned char>((i * 37) & 0xFF);
}
static void fill_q6k_block(unsigned char* p) {
  for (int i = 0; i < 128; ++i) p[i] = static_cast<unsigned char>((i * 37) & 0xFF);
  for (int i = 0; i < 64; ++i) p[128 + i] = static_cast<unsigned char>((i * 91) & 0xFF);
  for (int i = 0; i < 16; ++i)
    p[192 + i] = static_cast<unsigned char>(static_cast<signed char>(1 + (i % 7)));
  put_half(p + 208, 0.02f);
}
static void fill_tensor(int type, unsigned char* p, std::size_t bytes) {
  const std::size_t bb = type == 0 ? 144 : 210;
  for (std::size_t o = 0; o + bb <= bytes; o += bb) {
    if (type == 0) fill_q4k_block(p + o);
    else fill_q6k_block(p + o);
  }
}

// ---------------------------------------------------------------------------
// thread team: threads created once, spin barrier between rounds
// ---------------------------------------------------------------------------
struct Team {
  int n = 0;
  std::atomic<int> gen{0};
  std::atomic<int> arrived{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> th;
  void (*body)(int tid, int nthreads) = nullptr;

  int aff = 0;  // 0 = none, 1 = cpu i, 2 = physical-core packing
  void start(int nthreads, void (*body_fn)(int, int)) {
    n = nthreads;
    body = body_fn;
    const char* ae = std::getenv("ENGINE_RP_AFF");
    aff = ae ? std::atoi(ae) : 0;
    for (int i = 0; i < n; ++i)
      th.emplace_back([this, i] {
#ifdef _WIN32
        if (aff == 1) {
          SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << i);
        } else if (aff == 2) {
          // 6 physical cores: first six logical CPUs are the physical cores
          const int cpu = (i < 6) ? i : (6 + (i - 6) % 6);
          SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
        } else if (aff == 3) {
          // pack pairs: thread 2k and 2k+1 on core k (cpu k and cpu k+6)
          const int cpu = (i % 2 == 0) ? (i / 2) : (6 + i / 2);
          SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
        }
#endif
        int seen = 0;
        for (;;) {
          if (stop.load(std::memory_order_relaxed)) return;
          const int g = gen.load(std::memory_order_acquire);
          if (g == seen) {
            for (int s = 0; s < 64; ++s) _mm_pause();
            std::this_thread::yield();
            continue;
          }
          seen = g;
          body(i, n);
          arrived.fetch_add(1, std::memory_order_acq_rel);
        }
      });
  }
  void run() {
    arrived.store(0, std::memory_order_relaxed);
    gen.fetch_add(1, std::memory_order_release);
    while (arrived.load(std::memory_order_acquire) < n) _mm_pause();
  }
  void stop_team() {
    stop.store(true);
    gen.fetch_add(1, std::memory_order_release);
    for (auto& t : th) t.join();
    th.clear();
  }
};

// ---------------------------------------------------------------------------
// data + current variant
// ---------------------------------------------------------------------------
static const std::size_t kRowBytes = (kHidden / 256) * 144;   // gate/up row: 1152 B
static const std::size_t kDRowBytes = (kInter / 256) * 210;    // down row:   420 B
static std::vector<std::vector<unsigned char>> g_gate, g_up, g_down;  // [copy][8 experts]
static std::vector<float> g_x;
static std::vector<float> g_act;    // [kActs][kInter]  phase A output
static std::vector<float> g_dbuf;   // [kActs][kHidden] phase B output
static cpu::XQ8 g_xq_tok;           // activation quantisation for phase A
static std::vector<cpu::XQ8> g_xq_act;
static int g_copy = 0;
static int g_variant = 0;
static std::atomic<unsigned long long> g_sink{0};

enum {
  V_READ = 0,
  V_PROD,     // production fp32 row kernel
  V_I8,       // production int8 row kernel
  V_NOSCALE,  // diagnostic: no scale decode
  V_NOX,      // diagnostic: no activation loads
  V_UNPACK,   // diagnostic: loads + nibble unpack only
  V_SIMD,     // SIMD scale decode, otherwise identical to prod
  V_SIMD2,    // SIMD decode hoisted per row
  V_SIMD3,    // SIMD decode + vpermps broadcasts
  V_PFROLL,   // + in-loop rolling prefetch 288 B ahead
  V_PFROLL576,
  V_PFROLL1152,
  V_NUM
};
static const char* variant_name(int v) {
  switch (v) {
    case V_READ: return "read32";
    case V_PROD: return "prod";
    case V_I8: return "i8";
    case V_NOSCALE: return "noscale";
    case V_NOX: return "nox";
    case V_UNPACK: return "unpack";
    case V_SIMD: return "simd";
    case V_SIMD2: return "simd2";
    case V_SIMD3: return "simd3";
    case V_PFROLL: return "pf288";
    case V_PFROLL576: return "pf576";
    case V_PFROLL1152: return "pf1152";
  }
  return "?";
}


// ---- diagnostic kernel variants (same memory traffic as avx2_dot_q4k) --------
// V_NOSCALE: scale decode replaced by constants -> isolates the scale_min cost.
static inline void k_q4k_noscale(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  for (int i = 0; i < nb * 144; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  const __m256 s0c = _mm256_set1_ps(0.5f), m0c = _mm256_set1_ps(0.25f);
  const __m256 s1c = _mm256_set1_ps(0.125f), m1c = _mm256_set1_ps(0.0625f);
  for (int b = 0; b < nb; ++b) {
    const unsigned char* qs = row + static_cast<std::size_t>(b) * 144 + 16;
    const float* xb = x + static_cast<std::size_t>(b) * 256;
    for (int g = 0; g < 4; ++g) {
      const unsigned char* q = qs + g * 32;
      const float* x0 = xb + g * 64;
      for (int l = 0; l < 32; l += 16) {
        const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
        const __m256i lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
        const __m256i hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
        const __m256i lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
        const __m256i hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
        const __m256 f00 = _mm256_fmsub_ps(s0c, _mm256_cvtepi32_ps(lo0), m0c);
        const __m256 f01 = _mm256_fmsub_ps(s1c, _mm256_cvtepi32_ps(hi0), m1c);
        const __m256 f10 = _mm256_fmsub_ps(s0c, _mm256_cvtepi32_ps(lo1), m0c);
        const __m256 f11 = _mm256_fmsub_ps(s1c, _mm256_cvtepi32_ps(hi1), m1c);
        a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(x0 + l), a0);
        a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(x0 + 32 + l), a1);
        a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(x0 + l + 8), a2);
        a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(x0 + 32 + l + 8), a3);
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = cpu::hsum256(_mm256_add_ps(a0, a2));
}

// V_NOX: x loads replaced by a fixed vector -> isolates the activation-load cost.
static inline void k_q4k_nox(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  for (int i = 0; i < nb * 144; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  const __m256 xk = _mm256_set1_ps(0.125f);
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
    const float d = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    const unsigned char* scales = p + 4;
    const unsigned char* qs = p + 16;
    float scf[8], mnf[8];
    for (int sub = 0; sub < 8; ++sub) {
      int sc = 0, mn = 0;
      cpu::scale_min(sub, scales, &sc, &mn);
      scf[sub] = d * static_cast<float>(sc);
      mnf[sub] = dmin * static_cast<float>(mn);
    }
    (void)x;
    const float* x0 = x + 0;
    for (int g = 0; g < 4; ++g) {
      const unsigned char* q = qs + g * 32;
      const __m256 s0 = _mm256_set1_ps(scf[2 * g]), m0 = _mm256_set1_ps(mnf[2 * g]);
      const __m256 s1 = _mm256_set1_ps(scf[2 * g + 1]), m1 = _mm256_set1_ps(mnf[2 * g + 1]);
      for (int l = 0; l < 32; l += 16) {
        const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
        const __m256i lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
        const __m256i hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
        const __m256i lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
        const __m256i hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
        a0 = _mm256_fmadd_ps(_mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo0), m0), xk, a0);
        a1 = _mm256_fmadd_ps(_mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi0), m1), xk, a1);
        a2 = _mm256_fmadd_ps(_mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo1), m0), xk, a2);
        a3 = _mm256_fmadd_ps(_mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi1), m1), xk, a3);
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = cpu::hsum256(_mm256_add_ps(a0, a2));
}

// V_UNPACK: same weight loads and nibble unpack as avx2_dot_q4k, but no float
// conversion / FMA / activation: isolates "loads + bit ops" from the FP ALU.
static inline void k_q4k_unpack(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  for (int i = 0; i < nb * 144; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  for (int b = 0; b < nb; ++b) {
    const unsigned char* qs = row + static_cast<std::size_t>(b) * 144 + 16;
    (void)x;
    for (int g = 0; g < 4; ++g) {
      const unsigned char* q = qs + g * 32;
      for (int l = 0; l < 32; l += 16) {
        const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
        a0 = _mm256_add_epi32(
            a0, _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15));
        a1 = _mm256_add_epi32(
            a1, _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4));
      }
    }
  }
  *out = static_cast<float>(cpu::hsum_epi32(_mm256_add_epi32(a0, a1)) & 0xFFFF);
}


// ---- SIMD Q4_K scale decode ------------------------------------------------
// The 12 packed scale bytes hold 8 six-bit scales and 8 six-bit minimums:
//   sc[0..3] = s[0..3] & 63                     mn[0..3] = s[4..7] & 63
//   sc[4..7] = (s[8..11] & 0xF) | ((s[0..3] >> 2) & 0x30)
//   mn[4..7] = (s[8..11] >> 4)  | ((s[4..7] >> 2) & 0x30)
// All byte-wise, so one 16-byte load + shifts/masks/pshufb replaces the eight
// scalar scale_min() calls (~96 uops/block in the shipped kernel).
static inline void q4k_scale8_simd(const unsigned char* scales, float d, float dmin,
                                   float* scf, float* mnf) {
  const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scales));
  const __m128i c0f = _mm_set1_epi8(0x0F);
  const __m128i c30 = _mm_set1_epi8(0x30);
  const __m128i c3f = _mm_set1_epi8(0x3F);
  const __m128i idx4567 = _mm_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i idx8_11 = _mm_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i a3f = _mm_and_si128(v, c3f);                    // s & 63
  const __m128i a0f = _mm_and_si128(v, c0f);                    // s & 15
  const __m128i sr2 = _mm_and_si128(_mm_srli_epi16(v, 2), c30); // (s>>2)&0x30
  const __m128i sr4 = _mm_and_si128(_mm_srli_epi16(v, 4), c0f); // (s>>4)&0x0F
  const __m128i sc_hi = _mm_or_si128(_mm_shuffle_epi8(a0f, idx8_11), sr2);
  const __m128i mn_lo = _mm_shuffle_epi8(a3f, idx4567);
  const __m128i mn_hi =
      _mm_or_si128(_mm_shuffle_epi8(sr4, idx8_11), _mm_shuffle_epi8(sr2, idx4567));
  const __m128i sc8 = _mm_unpacklo_epi32(a3f, sc_hi);  // sc[0..7] in bytes 0..7
  const __m128i mn8 = _mm_unpacklo_epi32(mn_lo, mn_hi);
  const __m256 dv = _mm256_set1_ps(d), dminv = _mm256_set1_ps(dmin);
  _mm256_storeu_ps(scf, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sc8)), dv));
  _mm256_storeu_ps(mnf, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(mn8)), dminv));
}

// avx2_dot_q4k with the SIMD scale decode (same arithmetic, same order).
static inline void k_q4k_simd(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 144;
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  alignas(32) float sca[8], mna[8];
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
    const float d = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    q4k_scale8_simd(p + 4, d, dmin, sca, mna);
    const unsigned char* qs = p + 16;
    const float* xb = x + static_cast<std::size_t>(b) * 256;
    for (int g = 0; g < 4; ++g) {
      const unsigned char* q = qs + g * 32;
      const __m256 s0 = _mm256_broadcast_ss(sca + 2 * g);
      const __m256 m0 = _mm256_broadcast_ss(mna + 2 * g);
      const __m256 s1 = _mm256_broadcast_ss(sca + 2 * g + 1);
      const __m256 m1 = _mm256_broadcast_ss(mna + 2 * g + 1);
      const float* x0 = xb + g * 64;
      for (int l = 0; l < 32; l += 16) {
        const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
        const __m256i lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
        const __m256i hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
        const __m256i lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
        const __m256i hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
        const __m256 f00 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo0), m0);
        const __m256 f01 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi0), m1);
        const __m256 f10 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo1), m0);
        const __m256 f11 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi1), m1);
        a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(x0 + l), a0);
        a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(x0 + 32 + l), a1);
        a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(x0 + l + 8), a2);
        a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(x0 + 32 + l + 8), a3);
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = cpu::hsum256(_mm256_add_ps(a0, a2));
}


// k_q4k_simd + whole-row scale predecode (breaks the per-block store->broadcast
// dependency chain, one bulk prologue per row instead).
static inline void k_q4k_simd2(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 144;
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
  alignas(32) float sca[8 * 8], mna[8 * 8];
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
    const float d = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    q4k_scale8_simd(p + 4, d, dmin, sca + b * 8, mna + b * 8);
  }
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  for (int b = 0; b < nb; ++b) {
    const unsigned char* qs = row + static_cast<std::size_t>(b) * 144 + 16;
    const float* xb = x + static_cast<std::size_t>(b) * 256;
    const float* scb = sca + b * 8;
    const float* mnb = mna + b * 8;
    for (int g = 0; g < 4; ++g) {
      const unsigned char* q = qs + g * 32;
      const __m256 s0 = _mm256_broadcast_ss(scb + 2 * g);
      const __m256 m0 = _mm256_broadcast_ss(mnb + 2 * g);
      const __m256 s1 = _mm256_broadcast_ss(scb + 2 * g + 1);
      const __m256 m1 = _mm256_broadcast_ss(mnb + 2 * g + 1);
      const float* x0 = xb + g * 64;
      for (int l = 0; l < 32; l += 16) {
        const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
        const __m256i lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
        const __m256i hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
        const __m256i lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
        const __m256i hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
        const __m256 f00 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo0), m0);
        const __m256 f01 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi0), m1);
        const __m256 f10 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo1), m0);
        const __m256 f11 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi1), m1);
        a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(x0 + l), a0);
        a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(x0 + 32 + l), a1);
        a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(x0 + l + 8), a2);
        a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(x0 + 32 + l + 8), a3);
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = cpu::hsum256(_mm256_add_ps(a0, a2));
}

// k_q4k_simd with lane broadcasts via vpermps (no store->load forwarding).
static inline void k_q4k_simd3(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 144;
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + i), _MM_HINT_T0);
  for (int i = 0; i < row_bytes; i += 64)
    _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + i), _MM_HINT_T1);
  alignas(32) float sca[8], mna[8];
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  for (int b = 0; b < nb; ++b) {
    const unsigned char* p = row + static_cast<std::size_t>(b) * 144;
    const float d = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p));
    const float dmin = cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2));
    q4k_scale8_simd(p + 4, d, dmin, sca, mna);
    const __m256 scv = _mm256_load_ps(sca), mnv = _mm256_load_ps(mna);
    const unsigned char* qs = p + 16;
    const float* xb = x + static_cast<std::size_t>(b) * 256;
    for (int g = 0; g < 4; ++g) {
      const __m256 s0 = _mm256_permutevar8x32_ps(scv, _mm256_set1_epi32(2 * g));
      const __m256 m0 = _mm256_permutevar8x32_ps(mnv, _mm256_set1_epi32(2 * g));
      const __m256 s1 = _mm256_permutevar8x32_ps(scv, _mm256_set1_epi32(2 * g + 1));
      const __m256 m1 = _mm256_permutevar8x32_ps(mnv, _mm256_set1_epi32(2 * g + 1));
      const unsigned char* q = qs + g * 32;
      const float* x0 = xb + g * 64;
      for (int l = 0; l < 32; l += 16) {
        const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
        const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
        const __m256i lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
        const __m256i hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
        const __m256i lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
        const __m256i hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
        const __m256 f00 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo0), m0);
        const __m256 f01 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi0), m1);
        const __m256 f10 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo1), m0);
        const __m256 f11 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi1), m1);
        a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(x0 + l), a0);
        a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(x0 + 32 + l), a1);
        a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(x0 + l + 8), a2);
        a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(x0 + 32 + l + 8), a3);
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = cpu::hsum256(_mm256_add_ps(a0, a2));
}


// prod + a continuous in-loop prefetch stream: the shipped kernel prefetches the
// whole row in one burst at row entry (which overruns the ~10-12 line fill buffers
// and then issues nothing for the rest of the row).  Here every block issues the
// prefetches for a block `D` bytes ahead, so the miss stream is steady.
template <int D, int HINT>
static inline void k_q4k_pfroll(const unsigned char* row, const float* x, int n_in, float* out) {
  const int nb = n_in / 256;
  const int row_bytes = nb * 144;
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  const __m256i c15 = _mm256_set1_epi32(0xF);
  alignas(32) float sca[8 * 8], mna[8 * 8];
  for (int b0 = 0; b0 < nb; b0 += 8) {
    const int bn = (nb - b0 < 8) ? (nb - b0) : 8;
    for (int b = 0; b < bn; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b0 + b) * 144;
      q4k_scale8_simd(p + 4, cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p)),
                      cpu::half_to_float(*reinterpret_cast<const std::uint16_t*>(p + 2)),
                      sca + b * 8, mna + b * 8);
    }
    for (int b = 0; b < bn; ++b) {
      const unsigned char* p = row + static_cast<std::size_t>(b0 + b) * 144;
      const unsigned char* ahead = p + D;
      _mm_prefetch(reinterpret_cast<const char*>(ahead), static_cast<int>(HINT));
      _mm_prefetch(reinterpret_cast<const char*>(ahead + 64), static_cast<int>(HINT));
      const float* sc = sca + b * 8;
      const float* mn = mna + b * 8;
      const unsigned char* qs = p + 16;
      const float* xb = x + static_cast<std::size_t>(b0 + b) * 256;
      for (int g = 0; g < 4; ++g) {
        const unsigned char* q = qs + g * 32;
        const __m256 s0 = _mm256_broadcast_ss(sc + 2 * g);
        const __m256 m0 = _mm256_broadcast_ss(mn + 2 * g);
        const __m256 s1 = _mm256_broadcast_ss(sc + 2 * g + 1);
        const __m256 m1 = _mm256_broadcast_ss(mn + 2 * g + 1);
        const float* x0 = xb + g * 64;
        for (int l = 0; l < 32; l += 16) {
          const __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
          const __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l + 8));
          const __m256i lo0 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw0), c15);
          const __m256i hi0 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw0), 4);
          const __m256i lo1 = _mm256_and_si256(_mm256_cvtepu8_epi32(raw1), c15);
          const __m256i hi1 = _mm256_srli_epi32(_mm256_cvtepu8_epi32(raw1), 4);
          const __m256 f00 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo0), m0);
          const __m256 f01 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi0), m1);
          const __m256 f10 = _mm256_fmsub_ps(s0, _mm256_cvtepi32_ps(lo1), m0);
          const __m256 f11 = _mm256_fmsub_ps(s1, _mm256_cvtepi32_ps(hi1), m1);
          a0 = _mm256_fmadd_ps(f00, _mm256_loadu_ps(x0 + l), a0);
          a1 = _mm256_fmadd_ps(f01, _mm256_loadu_ps(x0 + 32 + l), a1);
          a2 = _mm256_fmadd_ps(f10, _mm256_loadu_ps(x0 + l + 8), a2);
          a3 = _mm256_fmadd_ps(f11, _mm256_loadu_ps(x0 + 32 + l + 8), a3);
        }
      }
    }
  }
  a0 = _mm256_add_ps(a0, a1);
  a2 = _mm256_add_ps(a2, a3);
  *out = cpu::hsum256(_mm256_add_ps(a0, a2));
}

// pure 32B read of a row in the production visit order
static inline unsigned long long read_row32(const unsigned char* row, int row_bytes) {
  __m256i a = _mm256_setzero_si256();
  __m256i b = _mm256_setzero_si256();
  int i = 0;
  for (; i + 64 <= row_bytes; i += 64) {
    a = _mm256_add_epi64(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i)));
    b = _mm256_add_epi64(b, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i + 32)));
  }
  for (; i + 32 <= row_bytes; i += 32)
    a = _mm256_add_epi64(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i)));
  a = _mm256_add_epi64(a, b);
  __m128i lo = _mm256_castsi256_si128(a);
  __m128i hi = _mm256_extracti128_si256(a, 1);
  lo = _mm_add_epi64(lo, hi);
  unsigned long long o[2];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(o), lo);
  return o[0] + o[1];
}

// ---------------------------------------------------------------------------
// phase A: gate+up over 8 experts
// ---------------------------------------------------------------------------
static void phaseA_body(int tid, int nthr) {
  const long long total = static_cast<long long>(kActs) * kInter;
  const long long chunk = (total + nthr - 1) / nthr;
  const long long b0 = static_cast<long long>(tid) * chunk;
  const long long e0 = std::min(total, b0 + chunk);
  const std::vector<unsigned char>& G = g_gate[static_cast<std::size_t>(g_copy)];
  const std::vector<unsigned char>& U = g_up[static_cast<std::size_t>(g_copy)];
  const std::size_t g_slab = static_cast<std::size_t>(kInter) * kRowBytes;
  unsigned long long sink = 0;
  for (long long idx = b0; idx < e0; ++idx) {
    const int act = static_cast<int>(idx / kInter);
    const int r = static_cast<int>(idx - static_cast<long long>(act) * kInter);
    const unsigned char* gr = G.data() + static_cast<std::size_t>(act) * g_slab +
                              static_cast<std::size_t>(r) * kRowBytes;
    const unsigned char* ur = U.data() + static_cast<std::size_t>(act) * g_slab +
                              static_cast<std::size_t>(r) * kRowBytes;
    const cpu::XQ8& xq = g_xq_tok;
    float g = 0.f, u = 0.f;
    switch (g_variant) {
      case V_READ:
        sink += read_row32(gr, static_cast<int>(kRowBytes));
        sink += read_row32(ur, static_cast<int>(kRowBytes));
        break;
      case V_PROD:
        cpu::avx2_dot_q4k(gr, g_x.data(), kHidden, &g);
        cpu::avx2_dot_q4k(ur, g_x.data(), kHidden, &u);
        break;
      case V_I8:
        cpu::avx2_dot_q4k_i8(gr, xq, kHidden, &g);
        cpu::avx2_dot_q4k_i8(ur, xq, kHidden, &u);
        break;
      case V_NOSCALE:
        k_q4k_noscale(gr, g_x.data(), kHidden, &g);
        k_q4k_noscale(ur, g_x.data(), kHidden, &u);
        break;
      case V_NOX:
        k_q4k_nox(gr, g_x.data(), kHidden, &g);
        k_q4k_nox(ur, g_x.data(), kHidden, &u);
        break;
      case V_UNPACK:
        k_q4k_unpack(gr, g_x.data(), kHidden, &g);
        k_q4k_unpack(ur, g_x.data(), kHidden, &u);
        break;
      case V_SIMD:
        k_q4k_simd(gr, g_x.data(), kHidden, &g);
        k_q4k_simd(ur, g_x.data(), kHidden, &u);
        break;
      case V_SIMD2:
        k_q4k_simd2(gr, g_x.data(), kHidden, &g);
        k_q4k_simd2(ur, g_x.data(), kHidden, &u);
        break;
      case V_SIMD3:
        k_q4k_simd3(gr, g_x.data(), kHidden, &g);
        k_q4k_simd3(ur, g_x.data(), kHidden, &u);
        break;
      case V_PFROLL:
        k_q4k_pfroll<288, _MM_HINT_T0>(gr, g_x.data(), kHidden, &g);
        k_q4k_pfroll<288, _MM_HINT_T0>(ur, g_x.data(), kHidden, &u);
        break;
      case V_PFROLL576:
        k_q4k_pfroll<576, _MM_HINT_T0>(gr, g_x.data(), kHidden, &g);
        k_q4k_pfroll<576, _MM_HINT_T0>(ur, g_x.data(), kHidden, &u);
        break;
      case V_PFROLL1152:
        k_q4k_pfroll<1152, _MM_HINT_T1>(gr, g_x.data(), kHidden, &g);
        k_q4k_pfroll<1152, _MM_HINT_T1>(ur, g_x.data(), kHidden, &u);
        break;
      default: break;
    }
    if (g_variant != V_READ) {
      g_act[static_cast<std::size_t>(act) * kInter + r] = (g / (1.f + std::exp(-g))) * u;
      sink += static_cast<unsigned long long>(g + u);
    }
  }
  g_sink.fetch_add(sink, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// phase B: down over 8 experts
// ---------------------------------------------------------------------------
static void phaseB_body(int tid, int nthr) {
  const long long total = static_cast<long long>(kActs) * kHidden;
  const long long chunk = (total + nthr - 1) / nthr;
  const long long b0 = static_cast<long long>(tid) * chunk;
  const long long e0 = std::min(total, b0 + chunk);
  const std::vector<unsigned char>& D = g_down[static_cast<std::size_t>(g_copy)];
  const std::size_t d_slab = static_cast<std::size_t>(kHidden) * kDRowBytes;
  unsigned long long sink = 0;
  for (long long idx = b0; idx < e0; ++idx) {
    const int act = static_cast<int>(idx / kHidden);
    const int r = static_cast<int>(idx - static_cast<long long>(act) * kHidden);
    const unsigned char* dr = D.data() + static_cast<std::size_t>(act) * d_slab +
                              static_cast<std::size_t>(r) * kDRowBytes;
    float d = 0.f;
    switch (g_variant) {
      case V_READ:
        sink += read_row32(dr, static_cast<int>(kDRowBytes));
        break;
      case V_PROD:
        cpu::avx2_dot_q6k(dr, g_act.data() + static_cast<std::size_t>(act) * kInter, kInter,
                          &d);
        break;
      case V_I8:
        cpu::avx2_dot_q6k_i8(dr, g_xq_act[static_cast<std::size_t>(act)], kInter, &d);
        break;
      default: break;
    }
    if (g_variant != V_READ) {
      g_dbuf[static_cast<std::size_t>(act) * kHidden + r] = d;
      sink += static_cast<unsigned long long>(d);
    }
  }
  g_sink.fetch_add(sink, std::memory_order_relaxed);
}

static double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const std::size_t i = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
  return v[std::min(i, v.size() - 1)];
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s a|b <copies> <iters> [threads] [variant...]\n", argv[0]);
    std::printf("  phase A variants: read prod i8\n  phase B variants: read f32 i8\n");
    return 2;
  }
  const std::string mode = argv[1];
  const int copies = std::atoi(argv[2]);
  const int iters = std::atoi(argv[3]);
  int nthr = 9;
  int argi = 4;
  if (argi < argc && std::atoi(argv[argi]) > 0) nthr = std::atoi(argv[argi++]);
  std::vector<int> vars;
  for (; argi < argc; ++argi) {
    const std::string s = argv[argi];
    if (s == "read") vars.push_back(V_READ);
    else if (s == "prod" || s == "f32") vars.push_back(V_PROD);
    else if (s == "i8") vars.push_back(V_I8);
    else if (s == "noscale") vars.push_back(V_NOSCALE);
    else if (s == "nox") vars.push_back(V_NOX);
    else if (s == "unpack") vars.push_back(V_UNPACK);
    else if (s == "simd") vars.push_back(V_SIMD);
    else if (s == "simd2") vars.push_back(V_SIMD2);
    else if (s == "simd3") vars.push_back(V_SIMD3);
    else if (s == "pf288") vars.push_back(V_PFROLL);
    else if (s == "pf576") vars.push_back(V_PFROLL576);
    else if (s == "pf1152") vars.push_back(V_PFROLL1152);
  }
  if (vars.empty()) vars = {V_READ, V_PROD, V_I8};

  const std::size_t gate_bytes = static_cast<std::size_t>(kActs) * kInter * kRowBytes;
  const std::size_t down_bytes = static_cast<std::size_t>(kActs) * kHidden * kDRowBytes;
  const std::size_t bytes_per_phase =
      (mode == "b") ? down_bytes : 2 * gate_bytes;
  std::printf("bench_rp %s: %d copies x %.2f MB, %d iters, %d threads\n", mode.c_str(), copies,
              static_cast<double>(bytes_per_phase) / 1048576.0, iters, nthr);

  g_gate.resize(static_cast<std::size_t>(copies));
  g_up.resize(static_cast<std::size_t>(copies));
  g_down.resize(static_cast<std::size_t>(copies));
  for (int c = 0; c < copies; ++c) {
    g_gate[static_cast<std::size_t>(c)].resize(gate_bytes);
    g_up[static_cast<std::size_t>(c)].resize(gate_bytes);
    g_down[static_cast<std::size_t>(c)].resize(down_bytes);
    fill_tensor(0, g_gate[static_cast<std::size_t>(c)].data(), gate_bytes);
    fill_tensor(0, g_up[static_cast<std::size_t>(c)].data(), gate_bytes);
    fill_tensor(1, g_down[static_cast<std::size_t>(c)].data(), down_bytes);
  }
  g_x.resize(static_cast<std::size_t>(kHidden));
  for (int i = 0; i < kHidden; ++i) g_x[static_cast<std::size_t>(i)] = std::sin(0.01f * (i + 1));
  g_act.resize(static_cast<std::size_t>(kActs) * kHidden);
  for (std::size_t i = 0; i < g_act.size(); ++i)
    g_act[i] = 0.1f * std::sin(0.003f * static_cast<float>(i + 1));
  g_dbuf.resize(static_cast<std::size_t>(kActs) * kHidden);
  cpu::xq_build(g_x.data(), kHidden, &g_xq_tok);
  g_xq_act.resize(static_cast<std::size_t>(kActs));
  for (int a = 0; a < kActs; ++a)
    cpu::xq_build(g_act.data() + static_cast<std::size_t>(a) * kInter, kInter,
                  &g_xq_act[static_cast<std::size_t>(a)]);

  Team team;
  team.start(nthr, mode == "b" ? phaseB_body : phaseA_body);

  std::vector<std::vector<double>> times(static_cast<std::size_t>(vars.size()));
  for (int rep = 0; rep < iters; ++rep) {
    for (std::size_t vi = 0; vi < vars.size(); ++vi) {
      g_variant = vars[vi];
      // Every variant must read a *fresh* copy: 9 MB would otherwise stay resident
      // in the 12 MB L3 and later variants in the same round would run from cache.
      g_copy = (rep * static_cast<int>(vars.size()) + static_cast<int>(vi)) % copies;
      const double t0 = now_ms();
      team.run();
      const double t1 = now_ms();
      times[vi].push_back(t1 - t0);
    }
  }
  team.stop_team();

  const double b = static_cast<double>(bytes_per_phase);
  for (std::size_t vi = 0; vi < vars.size(); ++vi) {
    const std::vector<double>& t = times[vi];
    const double mn = *std::min_element(t.begin(), t.end());
    const double p25 = quantile(t, 0.25), md = quantile(t, 0.5);
    std::printf("  %-6s min %7.3f ms (%6.2f GB/s)  p25 %7.3f (%6.2f)  med %7.3f (%6.2f)\n",
                variant_name(vars[vi]), mn, b / (mn * 1e6), p25, b / (p25 * 1e6), md,
                b / (md * 1e6));
  }
  std::printf("  sink %llu\n", g_sink.load());
  return 0;
}
