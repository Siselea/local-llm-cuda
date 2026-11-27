// Half-precision (IEEE-754 binary16) conversion helpers -- the single source of truth
// for BOTH the host loaders and the NVRTC-compiled device code.
//
// Why this file exists: the host loaders (F16/BF16 GGUF and safetensors weights) decode
// fp16 by hand, and the GPU dequant/attention kernels decode it too.  Those two copies
// had diverged: the host copy treated a *subnormal* fp16 (exp == 0, mant != 0) as if it
// were normal, i.e. it returned 2^-15 * (1 + mant/1024) instead of mant * 2^-24.  For the
// smallest subnormal (mant == 1) that is 3.05e-5 vs the correct 5.96e-8 -- a 512x error
// on every F16 weight that falls into the subnormal range.  Keeping this file as the
// only definition (host function below, device source in kDeviceSrc) stops it happening
// again.
//
// NVRTC cannot #include this header (the device sources are raw strings compiled at
// runtime), so kDeviceSrc holds the device copy right next to the host copy, and
// tests/test_fp16.cpp verifies all 65536 patterns of the host implementation against an
// independent reference AND, when a GPU is present, against the NVRTC-compiled device
// implementation -- any divergence between the two copies fails that test.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace engine {
namespace fp16 {

// IEEE-754 binary16 -> binary32 bit pattern.  Bit-exact: zero, subnormal (renormalised),
// normal, infinity and NaN all map as the hardware conversion does; NaN payloads are
// preserved in the high mantissa bits.
inline uint32_t half_to_float_bits(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +-0
    } else {
      // Subnormal: value = mant * 2^-24.  Renormalise so the implicit leading 1 sits
      // where binary32 expects it (this is the part the old host copy got wrong).
      int e = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --e;
      }
      mant &= 0x3FFu;
      bits = sign | (static_cast<uint32_t>(e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / NaN (payload kept)
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  return bits;
}

inline float half_to_float(uint16_t h) {
  const uint32_t bits = half_to_float_bits(h);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// IEEE-754 binary32 -> binary16 bit pattern, round-to-nearest-even with subnormal
// support (what cvt.rn.f16.f32 does).  Used by tests (and available to host callers);
// the device prefill path has its own historical helper for compatibility.
inline uint16_t float_to_half_bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const int e32 = static_cast<int>((x >> 23) & 0xFFu);
  const uint32_t mant32 = x & 0x7FFFFFu;
  if (e32 == 0xFF) {  // inf / NaN
    return static_cast<uint16_t>(sign | 0x7C00u | (mant32 ? 0x0200u : 0u));
  }
  int e = e32 - 127 + 15;
  if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign;  // rounds to zero
    // Subnormal result: shift the (1.)mantissa right by the missing exponent with
    // round-to-nearest-even.
    uint32_t m = mant32 | 0x800000u;
    const int shift = 14 - e;  // e <= 0  =>  shift >= 14
    const uint32_t half = 1u << (shift - 1);
    m += half - 1 + ((m >> shift) & 1u);  // round to nearest, ties to even
    return static_cast<uint16_t>(sign | (m >> shift));
  }
  uint32_t m = mant32 + 0x1000u;  // round to nearest even at bit 12
  if (m & 0x800000u) {
    m = 0;
    ++e;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint16_t>(e) << 10) | (m >> 13));
}

// Canonical *device* source for the same conversion.  Prepended to every NVRTC module
// that includes kernels::kQuantCommon, so all kernels get it from this single text.
// Keep it identical to half_to_float_bits above (tests/test_fp16.cpp compares the host
// function against this compiled-by-NVRTC version for all 65536 patterns on a GPU).
inline const std::string kDeviceSrc = R"cuda(
__device__ __forceinline__ float half_to_float(unsigned short h) {
  unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal: renormalise (value = mant * 2^-24) before storing the binary32 bits.
      int e = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
      mant &= 0x3FFu;
      bits = sign | ((unsigned int)e << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  return __uint_as_float(bits);
}
)cuda";

}  // namespace fp16
}  // namespace engine
