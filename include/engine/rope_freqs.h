// Proportional-rope frequency factors (GGUF gemma4 `rope_freqs.weight`).
//
// llama.cpp gemma4 binds one global F32 tensor `rope_freqs.weight` (shape
// n_embd_head/2) to every *full-attention* layer (src/models/gemma4.cpp:100-104,
// TENSOR_DUPLICATED).  The rope cache divides the angle of pair i by ff[i]
// (ggml/src/ggml-cpu/ops.cpp:6335-6349, identical in ggml-cuda/rope.cu):
//
//     theta_i = pos * freq_base^(-2i/n_rot) / ff[i]     i = 0 .. n_rot/2-1
//
// The real gemma4-12b table is 64 x 1.0 followed by 192 x 1e30: theta/1e30
// underflows to 0, so cos=1/sin=0 and the pair is left unrotated.  That is the
// "proportional rope" of HF (partial_rotary_factor = 0.25): only the first
// 128 of 512 head dims rotate on full layers; sliding layers use full rotary.
//
// The sentinel is a huge *finite* factor rather than 0 because the code divides
// by it; a 0 factor would produce inf.  This header keeps the parsing/validation
// logic independent of CUDA so it can be unit-tested without a GPU.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace engine {
namespace rope_ff {

// Frequency factors at or above this value are "do not rotate" sentinels.
// llama.cpp's kvalues are 1.0 and 1e30 for gemma4; 1e20 leaves headroom while
// still keeping theta/ff below float epsilon for any position we can reach.
constexpr float kNoRotation = 1e20f;

inline bool usable_factor(float f) {
  return std::isfinite(f) && f > 0.0f && f < kNoRotation;
}

// Number of pairs that are actually rotated (factors below kNoRotation).
inline size_t rotating_pairs(const std::vector<float>& ff) {
  size_t n = 0;
  for (const float f : ff)
    if (usable_factor(f)) ++n;
  return n;
}

inline bool is_proportional(const std::vector<float>& ff) {
  return !ff.empty() && rotating_pairs(ff) < ff.size();
}

// Repair a factor table loaded from a GGUF: non-finite / non-positive entries are
// replaced with the no-rotation sentinel so the rope kernel can never divide by
// zero or NaN.  Returns false when anything had to be repaired (caller warns).
inline bool sanitize(std::vector<float>& ff) {
  bool clean = true;
  for (float& f : ff) {
    if (!std::isfinite(f) || f <= 0.0f) {
      f = kNoRotation;
      clean = false;
    } else if (f == 0.0f) {
      f = kNoRotation;
      clean = false;
    }
  }
  return clean;
}

// Expected table length for a full-attention layer with `rope_dim` rotary dims.
inline size_t expected_len(int rope_dim) {
  return rope_dim > 0 ? static_cast<size_t>(rope_dim) / 2 : 0;
}

// The table applies only to full-attention layers (llama.cpp creates the tensor
// only for !is_swa layers) and only when its length matches that layer's n_rot/2.
inline bool applies(bool full_attention_layer, size_t table_len, int rope_dim) {
  return full_attention_layer && table_len > 0 &&
         table_len == expected_len(rope_dim);
}

// Angle of rotary pair i, for tests and for documentation: theta = pos*inv/ff.
inline float pair_angle(float pos, float inv_freq, float factor) {
  return pos * inv_freq / factor;
}

// Build the per-pair inverse frequencies used by the rope kernels:
// inv_freq[i] = freq_base^(-2i/rope_dim), matching ggml's theta_scale^i.
inline std::vector<float> inverse_frequencies(double freq_base, int rope_dim) {
  std::vector<float> inv;
  if (rope_dim <= 0) return inv;
  const size_t half = static_cast<size_t>(rope_dim) / 2;
  inv.resize(half);
  for (size_t i = 0; i < half; ++i)
    inv[i] = static_cast<float>(std::pow(freq_base, -2.0 * static_cast<double>(i) /
                                                       static_cast<double>(rope_dim)));
  return inv;
}

}  // namespace rope_ff
}  // namespace engine
