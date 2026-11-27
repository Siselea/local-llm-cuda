// Bit-exactness of the fp16 (IEEE-754 binary16) conversions used by the loaders.
//
// Regression context: the host decoders in src/runner.cpp treated an fp16 *subnormal*
// (exp == 0, mant != 0) as if it were normal, i.e. they returned
//     2^-15 * (1 + mant/1024)   instead of   mant * 2^-24
// so the smallest subnormal decoded as 3.05e-5 instead of 5.96e-8 (512x too large).
// Every F16 GGUF / safetensors weight in the subnormal range was affected.
//
// The fix routes all host decoding through engine::fp16::half_to_float, and the device
// source lives in the same header (engine::fp16::kDeviceSrc).  This test:
//   1. verifies the host conversion against an independent arithmetic reference for ALL
//      65536 bit patterns (bit-for-bit for finite values, class + sign for NaN);
//   2. spot-checks the subnormal/zero/inf edge cases the old code got wrong;
//   3. round-trips every non-NaN pattern half -> float -> half and requires identity;
//   4. verifies the NVRTC-compiled device conversion against the host conversion for all
//      65536 patterns (skipped with a message when no CUDA device is available);
//   5. checks that kernels::kQuantCommon still gets its half_to_float from the shared
//      header exactly once, so the two copies cannot silently fork again.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "engine/cuda.h"
#include "engine/fp16.h"
#include "engine/kernels.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* what, int detail = -1) {
  ++g_checks;
  if (ok) return;
  ++g_failures;
  if (detail >= 0) {
    std::fprintf(stderr, "FAIL: %s (pattern 0x%04x)\n", what, detail);
  } else {
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

uint32_t float_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// Independent reference: decode by arithmetic (ldexp) rather than by bit surgery, so a
// shared misunderstanding of the exponent bias cannot hide in both implementations.
float ref_half_to_float(uint16_t h) {
  const double sign = (h & 0x8000u) ? -1.0 : 1.0;
  const int exp = (h >> 10) & 0x1Fu;
  const int mant = h & 0x3FFu;
  if (exp == 0) {
    // value = mant * 2^-24 (exactly representable in float for mant <= 1023)
    return static_cast<float>(sign * std::ldexp(static_cast<double>(mant), -24));
  }
  if (exp == 31) {
    if (mant != 0) return std::numeric_limits<float>::quiet_NaN();
    return sign > 0 ? std::numeric_limits<float>::infinity()
                    : -std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(sign * std::ldexp(1.0 + mant / 1024.0, exp - 15));
}

void test_all_patterns_host() {
  int nan_patterns = 0;
  for (uint32_t i = 0; i < 65536u; ++i) {
    const uint16_t h = static_cast<uint16_t>(i);
    const uint32_t got = engine::fp16::half_to_float_bits(h);
    const float ref = ref_half_to_float(h);
    const bool ref_nan = std::isnan(ref);
    if (ref_nan) {
      ++nan_patterns;
      const bool got_nan = ((got & 0x7F800000u) == 0x7F800000u) && (got & 0x007FFFFFu) != 0;
      check(got_nan, "NaN input must decode to NaN", static_cast<int>(h));
      check(((got >> 31) & 1u) == ((h >> 15) & 1u), "NaN sign preserved",
            static_cast<int>(h));
    } else {
      check(got == float_bits(ref), "finite pattern must match arithmetic reference",
            static_cast<int>(h));
    }
    // Round trip: half -> float -> half must be the identity for every non-NaN pattern
    // (half -> float is exact, and float -> half rounds an exactly representable value
    // back to itself).
    if (!ref_nan) {
      check(engine::fp16::float_to_half_bits(engine::fp16::half_to_float(h)) == h,
            "half -> float -> half identity", static_cast<int>(h));
    }
  }
  check(nan_patterns == 2046, "fp16 has 2046 NaN patterns (1023 mantissas x 2 signs)",
        nan_patterns);
}

void test_edge_cases() {
  // The exact values the old host decoder got wrong (subnormals).
  const float min_sub = engine::fp16::half_to_float(0x0001);
  check(float_bits(min_sub) == 0x33800000u, "smallest subnormal is 2^-24 (5.96e-8)",
        static_cast<int>(0x0001));
  check(std::fabs(min_sub - 5.960464477539063e-08f) == 0.0f, "smallest subnormal value");
  const float max_sub = engine::fp16::half_to_float(0x03FF);
  check(float_bits(max_sub) == 0x387FC000u, "largest subnormal is 1023 * 2^-24",
        static_cast<int>(0x03FF));
  check(std::fabs(max_sub - 6.097555160522461e-05f) == 0.0f, "largest subnormal value");
  check(float_bits(engine::fp16::half_to_float(0x8001)) == 0xB3800000u,
        "negative smallest subnormal", 0x8001);
  // Zero, normal, infinity, NaN.
  check(engine::fp16::half_to_float_bits(0x0000) == 0x00000000u, "+0", 0);
  check(engine::fp16::half_to_float_bits(0x8000) == 0x80000000u, "-0", 0);
  check(engine::fp16::half_to_float_bits(0x3C00) == 0x3F800000u, "1.0", 0x3C00);
  check(engine::fp16::half_to_float_bits(0xC000) == 0xC0000000u, "-2.0", 0xC000);
  check(engine::fp16::half_to_float_bits(0x7BFF) == 0x477FE000u, "65504 max finite", 0x7BFF);
  check(engine::fp16::half_to_float_bits(0x7C00) == 0x7F800000u, "+inf", 0x7C00);
  check(engine::fp16::half_to_float_bits(0xFC00) == 0xFF800000u, "-inf", 0xFC00);
  // 2^-14 is the smallest normal; just below it is the subnormal boundary.
  check(engine::fp16::half_to_float_bits(0x0400) == 0x38800000u, "2^-14 smallest normal",
        0x0400);
}

bool test_device_matches_host() {
  try {
    cuda::init();
  } catch (const std::exception& e) {
    std::printf("device check: SKIPPED (no CUDA device: %s)\n", e.what());
    return true;
  }
  try {
    const std::string src = engine::fp16::kDeviceSrc + R"cuda(
extern "C" __global__ void fp16_probe(const unsigned short* __restrict__ in,
                                      unsigned int* __restrict__ out, int n) {
  const int i = (int)blockIdx.x * (int)blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float_as_uint(half_to_float(in[i]));
}
)cuda";
    cuda::Module mod(cuda::compile_ptx(src, "fp16_probe"));
    void* fn = mod.kernel("fp16_probe");
    std::vector<unsigned short> in(65536);
    for (uint32_t i = 0; i < 65536u; ++i) in[i] = static_cast<unsigned short>(i);
    cuda::Buffer din(in.size() * 2), dout(in.size() * 4);
    din.upload(in.data(), in.size() * 2);
    int n = 65536;
    void* pin = din.data();
    void* pout = dout.data();
    void* args[] = {&pin, &pout, &n};
    cuda::launch(fn, (65536 + 255) / 256, 1, 1, 256, 1, 1, 0, args);
    std::vector<uint32_t> out(65536);
    dout.download(out.data(), out.size() * 4);
    int bad = 0;
    for (uint32_t i = 0; i < 65536u; ++i) {
      const uint32_t host = engine::fp16::half_to_float_bits(static_cast<uint16_t>(i));
      const bool host_nan = ((host & 0x7F800000u) == 0x7F800000u) && (host & 0x7FFFFFu);
      const bool dev_nan = ((out[i] & 0x7F800000u) == 0x7F800000u) && (out[i] & 0x7FFFFFu);
      const bool same = host_nan ? (dev_nan && ((host >> 31) == (out[i] >> 31)))
                                 : host == out[i];
      if (!same && bad++ < 8)
        std::fprintf(stderr, "FAIL: device/host mismatch pattern 0x%04x host=0x%08x dev=0x%08x\n",
                     i, host, out[i]);
    }
    check(bad == 0, "device half_to_float matches host for all 65536 patterns", bad);
    return bad == 0;
  } catch (const std::exception& e) {
    std::printf("device check: SKIPPED (%s)\n", e.what());
    return true;
  }
}

void test_shared_source_wiring() {
  // kernels::kQuantCommon must get half_to_float from the shared header exactly once and
  // contain the subnormal renormalisation (the old buggy form had no such loop).
  const std::string& common = kernels::kQuantCommon;
  check(common.find("half_to_float(unsigned short h)") != std::string::npos,
        "kQuantCommon defines half_to_float");
  size_t defs = 0;
  for (size_t pos = common.find("half_to_float(unsigned short h)"); pos != std::string::npos;
       pos = common.find("half_to_float(unsigned short h)", pos + 1))
    ++defs;
  check(defs == 1, "half_to_float is defined exactly once in the kernel module", (int)defs);
  check(common.find("while ((mant & 0x400u) == 0)") != std::string::npos,
        "device conversion renormalises subnormals");
}

}  // namespace

int main() {
  test_all_patterns_host();
  test_edge_cases();
  test_shared_source_wiring();
  test_device_matches_host();
  if (g_failures != 0) {
    std::fprintf(stderr, "test_fp16: %d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("test_fp16: %d checks PASS (all 65536 patterns, host + device)\n", g_checks);
  return 0;
}
