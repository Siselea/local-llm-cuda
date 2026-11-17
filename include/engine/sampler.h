// Token sampling for the generation loop.
//
// Deterministic per (seed, params, logits): the PRNG is the engine's own
// xorshift128+ (no <random> dependency, identical sequence on every platform).
// Supports temperature / top-k / top-p (nucleus) / min-p / repetition penalty.
#pragma once

#include <cstdint>
#include <vector>

namespace engine {
namespace sampler {

// Same knobs as the OpenAI request body / llama.cpp.
struct Params {
  double temperature = 1.0;      // <= 0 -> greedy
  double top_p = 1.0;            // 1.0 disables nucleus filtering
  int top_k = 0;                 // 0 disables top-k filtering
  double min_p = 0.0;            // 0 disables min-p filtering
  double repetition_penalty = 1.0;  // 1.0 disables
  int repeat_last_n = 256;       // repetition window (-1 = whole context)
  int seed = -1;                 // -1 = random seed from the clock
};

// xorshift128+ -- fast, deterministic, good enough for sampling.
class Rng {
 public:
  explicit Rng(uint64_t seed);
  uint64_t next_u64();
  double uniform();              // [0, 1)
  void seed(uint64_t s);

 private:
  uint64_t s0_ = 0;
  uint64_t s1_ = 0;
};

class Sampler {
 public:
  Sampler() : rng_(0) {}
  explicit Sampler(uint64_t seed);

  void set_params(const Params& p) { params_ = p; }
  const Params& params() const { return params_; }
  void reseed(uint64_t seed) { rng_.seed(seed); }

  // Pick one token id from a full logits row.  `recent_ids` is the tail of the
  // context used for the repetition penalty (may be empty).
  // The logits vector is not modified.
  int sample(const std::vector<float>& logits, const std::vector<int>& recent_ids);

  // Greedy argmax (used when temperature <= 0 and by the tests).
  static int argmax(const std::vector<float>& logits);

 private:
  void apply_repetition_penalty(std::vector<float>* logits,
                                const std::vector<int>& recent_ids) const;

  Params params_;
  Rng rng_;
  // scratch buffers reused across steps
  mutable std::vector<float> scores_;
  mutable std::vector<int> cand_;
};

// Seed helper: -1 means "derive from the clock".
uint64_t resolve_seed(int seed);

}  // namespace sampler
}  // namespace engine
