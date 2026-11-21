// Token sampling for the generation loop.
//
// Deterministic per (seed, params, logits): the PRNG is the engine's own
// xorshift128+ (no <random> dependency, identical sequence on every platform).
// Supports temperature / top-k / top-p (nucleus) / min-p / repetition penalty.
//
// Two entry points:
//   sample()         - the reference implementation: applies the sampler chain to
//                      the full vocabulary.  Kept for tests and for callers that
//                      need the exact full-vocab distribution.
//   sample_bounded() - the serve default for non-greedy requests: the same sampler
//                      chain, but evaluated on at most `candidate_cap()` candidates
//                      (host-side top-N by logit, or a candidate set supplied by the
//                      device via set_candidates()).  Avoids a full-vocab fill /
//                      exp / sort per token.
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
  // How many top logits sample_bounded() looks at when top_k == 0 (top_k > 0
  // requests use exactly max(top_k, 64) candidates).  Serve exposes this in /props.
  static constexpr int kDefaultCandidateCap = 1024;

  Sampler() : rng_(0) {}
  explicit Sampler(uint64_t seed);

  void set_params(const Params& p) { params_ = p; }
  const Params& params() const { return params_; }
  void reseed(uint64_t seed) { rng_.seed(seed); }

  // Pick one token id from a full logits row.  `recent_ids` is the tail of the
  // context used for the repetition penalty (may be empty).
  // The logits vector is not modified.
  int sample(const std::vector<float>& logits, const std::vector<int>& recent_ids);

  // Bounded-candidate variant used by serve for every non-greedy token:
  //  * candidates = the top-N raw logits (N = candidate_cap() when top_k == 0,
  //    else max(top_k, 64)), selected with std::nth_element on the host; or the
  //    set installed by set_candidates() (device top-k hook);
  //  * repetition penalty, top_k, top_p, min_p and temperature are evaluated on
  //    those candidates only; probabilities are renormalised inside the set.
  //  * greedy (temperature <= 0) is still an exact full-row argmax, identical to
  //    sample(), so greedy output never changes.
  // The logits vector is not modified.  `logits.size()` must equal the vocabulary.
  int sample_bounded(const std::vector<float>& logits, const std::vector<int>& recent_ids);

  // Candidate-set cap for the top_k == 0 path (clamped to [64, 1'000'000]).
  void set_candidate_cap(int cap);
  int candidate_cap() const { return candidate_cap_; }

  // Device top-k hook: install a candidate set (ids + their raw logits) that
  // sample_bounded() should use instead of selecting the candidates itself.
  // The data is copied; n <= 0 (or null pointers) clears the override.  Intended
  // for a future runner that reduces the logits on the device (k = 64/128).
  void set_candidates(const int* ids, const float* logits, int n);
  void clear_candidates();
  bool has_external_candidates() const { return have_ext_candidates_; }

  // Greedy argmax (used when temperature <= 0 and by the tests).
  static int argmax(const std::vector<float>& logits);

 private:
  void apply_repetition_penalty(std::vector<float>* logits,
                                const std::vector<int>& recent_ids) const;

  // Build the candidate id list for sample_bounded() (host selection).
  void select_candidates(const std::vector<float>& logits);
  // Apply repetition penalty to cand_val_ (candidate logits) in place.
  void penalise_candidates(const std::vector<int>& recent_ids, int vocab);

  Params params_;
  Rng rng_;
  // scratch buffers reused across steps
  mutable std::vector<float> scores_;
  mutable std::vector<int> cand_;

  // sample_bounded() scratch (reused across tokens)
  int candidate_cap_ = kDefaultCandidateCap;
  bool have_ext_candidates_ = false;
  mutable std::vector<int> ext_ids_;
  mutable std::vector<float> ext_vals_;
  mutable std::vector<float> cand_val_;
  struct HeapItem {
    float v;
    int id;
  };
  mutable std::vector<HeapItem> heap_;   // min-heap for top-N selection
  mutable std::vector<int> order_;       // top_p ordering workspace
  mutable std::vector<unsigned char> pen_flag_;  // vocab-sized penalty membership
  mutable std::vector<int> pen_ids_;     // ids set in pen_flag_ (for clearing)
};

// Seed helper: -1 means "derive from the clock".
uint64_t resolve_seed(int seed);

}  // namespace sampler
}  // namespace engine
