// Unit test for the proportional-rope frequency factors (GGUF gemma4
// `rope_freqs.weight`).  Pure host code, no CUDA, no model files.
//
// The reference semantics come from llama.cpp:
//   src/models/gemma4.cpp:100-104          rope_freqs bound to !is_swa layers only
//   ggml/src/ggml-cpu/ops.cpp:6335-6349    theta = pos*base^(-2i/n_rot) / ff[i]
//   ggml/src/ggml-cuda/rope.cu:110-115     same formula on CUDA
// The real 12b table is 64 x 1.0 + 192 x 1e30: theta/1e30 -> 0 so only the first
// 128 of 512 head dims rotate (HF partial_rotary_factor = 0.25) on full layers.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "engine/rope_freqs.h"

namespace {

int fails = 0;
int checks = 0;

void check(const char* name, bool ok) {
  ++checks;
  if (!ok) {
    std::printf("  FAIL %s\n", name);
    ++fails;
  }
}

void expect_near(const char* name, float got, float want, float tol) {
  check(name, std::fabs(got - want) <= tol);
  if (std::fabs(got - want) > tol)
    std::printf("    got %.9g want %.9g (tol %.3g)\n", got, want, tol);
}

// gemma4-12b shape: n_rot = 512, table = 64 rotating pairs then sentinels.
std::vector<float> gemma_table() {
  std::vector<float> ff(256, 1e30f);
  for (int i = 0; i < 64; ++i) ff[i] = 1.0f;
  return ff;
}

// Rotate one head vector the way the rope kernel does for one position:
// pairs (i, i+half) with theta_i = pos * inv_freq[i] / ff[i] (NEOX layout).
std::vector<float> rope_head(const std::vector<float>& x,
                             const std::vector<float>& inv_freq,
                             const std::vector<float>* ff, float pos) {
  const size_t half = x.size() / 2;
  std::vector<float> out = x;
  for (size_t i = 0; i < half; ++i) {
    float theta = pos * inv_freq[i];
    if (ff) theta /= (*ff)[i];
    const float c = std::cos(theta), s = std::sin(theta);
    out[i] = x[i] * c - x[i + half] * s;
    out[i + half] = x[i] * s + x[i + half] * c;
  }
  return out;
}

bool bit_equal(float a, float b) {
  return std::memcmp(&a, &b, sizeof(float)) == 0;
}

}  // namespace

int main() {
  std::printf("proportional-rope frequency factors\n");
  using namespace engine::rope_ff;

  // ---- table shape / detection -------------------------------------------
  const std::vector<float> gemma = gemma_table();
  check("gemma table has 256 pairs", gemma.size() == 256);
  check("gemma rotates 64 pairs", rotating_pairs(gemma) == 64);
  check("gemma is proportional", is_proportional(gemma));
  check("all-ones table is not proportional",
        !is_proportional(std::vector<float>(256, 1.0f)));
  check("empty table is not proportional", !is_proportional({}));
  check("expected_len(512) == 256", expected_len(512) == 256);
  check("expected_len(256) == 128", expected_len(256) == 128);
  check("expected_len(0) == 0", expected_len(0) == 0);
  check("applies to full layer with matching table", applies(true, 256, 512));
  check("does not apply to sliding layer", !applies(false, 256, 512));
  check("does not apply on length mismatch", !applies(true, 128, 512));
  check("does not apply with empty table", !applies(true, 0, 512));
  check("sentinel 1e30 is not a usable rotation",
        !usable_factor(1e30f));
  check("1.0 is usable", usable_factor(1.0f));
  check("1e19 is usable (below sentinel)", usable_factor(1e19f));

  // ---- sanitize -----------------------------------------------------------
  {
    float nan = std::nanf("");
    float inf = std::numeric_limits<float>::infinity();
    std::vector<float> bad = {1.0f, nan, inf, 0.0f, -2.0f, 1e30f};
    const bool clean = sanitize(bad);
    check("sanitize reports repairs", !clean);
    check("sanitize maps nan to sentinel", bad[1] == kNoRotation);
    check("sanitize maps inf to sentinel", bad[2] == kNoRotation);
    check("sanitize maps 0 to sentinel", bad[3] == kNoRotation);
    check("sanitize maps negative to sentinel", bad[4] == kNoRotation);
    check("sanitize keeps valid factors", bad[0] == 1.0f && bad[5] == 1e30f);
    std::vector<float> good = gemma_table();
    check("sanitize leaves a clean gemma table untouched", sanitize(good));
  }

  // ---- inverse frequencies / angles --------------------------------------
  {
    const std::vector<float> inv = inverse_frequencies(1000000.0, 512);
    check("inv_freq length", inv.size() == 256);
    expect_near("inv_freq[0] == 1", inv[0], 1.0f, 1e-6f);
    expect_near("inv_freq[1] == 1e6^(-2/512)",
                inv[1], static_cast<float>(std::pow(1000000.0, -2.0 / 512.0)), 1e-7f);
    expect_near("inv_freq[63]", inv[63],
                static_cast<float>(std::pow(1000000.0, -2.0 * 63 / 512.0)), 1e-7f);
    // pos=7, pair 0: theta = 7 (factor 1)
    expect_near("angle factor 1", pair_angle(7.0f, inv[0], 1.0f), 7.0f, 1e-6f);
    // factor 2 halves the angle
    expect_near("angle factor 2", pair_angle(7.0f, inv[0], 2.0f), 3.5f, 1e-6f);
    // sentinel makes the angle tiny -> cos=1, sin~0 (no rotation)
    expect_near("angle sentinel", pair_angle(7.0f, inv[0], 1e30f), 0.0f, 1e-20f);
  }

  // ---- the actual behaviour on a head vector ------------------------------
  {
    // 8 pairs = 16 dims; first 2 pairs rotate (25% proportional), 6 are identity.
    std::vector<float> x(16);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.5f + 0.25f * static_cast<float>(i);
    const std::vector<float> inv = inverse_frequencies(1000000.0, 16);
    const std::vector<float> ff = {1.0f, 1.0f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f};
    const std::vector<float> rot = rope_head(x, inv, &ff, 3.0f);
    const std::vector<float> plain = rope_head(x, inv, nullptr, 3.0f);
    // With ff=1 on the first pairs both variants rotate identically there; the
    // difference is on the sentinel pairs, which plain rope rotates and the
    // proportional table leaves untouched.
    check("first pair rotated identically to plain rope",
          bit_equal(rot[0], plain[0]) && bit_equal(rot[8], plain[8]));
    check("plain rope rotates a sentinel pair, proportional rope does not",
          !bit_equal(plain[2], x[2]) && bit_equal(rot[2], x[2]));
    bool tail_unchanged = true;
    for (size_t i = 2; i < 8; ++i) {
      tail_unchanged &= bit_equal(rot[i], x[i]);
      tail_unchanged &= bit_equal(rot[i + 8], x[i + 8]);
    }
    check("sentinel pairs are exactly unrotated", tail_unchanged);
    // And the first pair equals the plain-rope result for that pair.
    expect_near("rotated pair matches plain rope d0", rot[0], plain[0], 1e-6f);
    expect_near("rotated pair matches plain rope d8", rot[8], plain[8], 1e-6f);
  }

  // ---- real 12b shape: 64 of 256 pairs rotate = 128 of 512 dims -----------
  {
    const std::vector<float> inv = inverse_frequencies(1000000.0, 512);
    std::vector<float> x(512);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25f;
    const std::vector<float> rot = rope_head(x, inv, &gemma, 5.0f);
    bool tail_ok = true;
    for (size_t i = 64; i < 256; ++i) {
      tail_ok &= bit_equal(rot[i], x[i]);
      tail_ok &= bit_equal(rot[i + 256], x[i + 256]);
    }
    check("gemma4-12b: dims 128..511 are unrotated", tail_ok);
    check("gemma4-12b: 64 pairs rotating == 128 dims", rotating_pairs(gemma) * 2 == 128);
  }

  std::printf("ROPE FREQ FACTORS: %d checks, %d failed\n", checks, fails);
  return fails == 0 ? 0 : 1;
}
