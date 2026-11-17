#include "engine/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace engine {
namespace sampler {

// ---------------------------------------------------------------------------
// xorshift128+ (period 2^128-1)
// ---------------------------------------------------------------------------
Rng::Rng(uint64_t seed) { this->seed(seed); }

void Rng::seed(uint64_t s) {
  // splitmix64 expansion so that small seeds still give a full state
  auto splitmix = [](uint64_t& x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  uint64_t x = s;
  s0_ = splitmix(x);
  s1_ = splitmix(x);
  if (s0_ == 0 && s1_ == 0) s0_ = 1;
}

uint64_t Rng::next_u64() {
  uint64_t x = s0_;
  const uint64_t y = s1_;
  s0_ = y;
  x ^= x << 23;
  s1_ = x ^ y ^ (x >> 17) ^ (y >> 26);
  return s1_ + y;
}

double Rng::uniform() {
  // 53-bit mantissa -> [0, 1)
  return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

uint64_t resolve_seed(int seed) {
  if (seed >= 0) return static_cast<uint64_t>(seed);
  const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
  uint64_t t = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  // mix with a process-local counter so two requests in the same clock tick
  // still get different streams
  static uint64_t counter = 0;
  t ^= (++counter * 0x9E3779B97F4A7C15ull);
  return t;
}

Sampler::Sampler(uint64_t seed) : rng_(seed) {}

// ---------------------------------------------------------------------------
int Sampler::argmax(const std::vector<float>& logits) {
  if (logits.empty()) return -1;
  int best = 0;
  float bv = logits[0];
  for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
    if (logits[i] > bv) {
      bv = logits[i];
      best = i;
    }
  }
  return best;
}

void Sampler::apply_repetition_penalty(std::vector<float>* logits,
                                       const std::vector<int>& recent_ids) const {
  const double penalty = params_.repetition_penalty;
  if (penalty == 1.0 || recent_ids.empty()) return;
  const int n = static_cast<int>(recent_ids.size());
  int begin = 0;
  if (params_.repeat_last_n >= 0 && params_.repeat_last_n < n)
    begin = n - params_.repeat_last_n;
  std::vector<float>& l = *logits;
  const int vocab = static_cast<int>(l.size());
  for (int i = begin; i < n; ++i) {
    const int id = recent_ids[i];
    if (id < 0 || id >= vocab) continue;
    if (l[id] > 0.0f) l[id] = static_cast<float>(l[id] / penalty);
    else l[id] = static_cast<float>(l[id] * penalty);
  }
}

int Sampler::sample(const std::vector<float>& logits, const std::vector<int>& recent_ids) {
  const int vocab = static_cast<int>(logits.size());
  if (vocab <= 0) return -1;
  // greedy: no sampler chain at all
  if (!(params_.temperature > 0.0)) return argmax(logits);

  // ---- 1) penalties (llama.cpp applies repeat/presence/freq penalties to the logits
  //        before the sampler chain) ----
  const std::vector<float>* src = &logits;
  std::vector<float> penalized;
  if (params_.repetition_penalty != 1.0 && !recent_ids.empty()) {
    penalized = logits;
    apply_repetition_penalty(&penalized, recent_ids);
    src = &penalized;
  }
  const std::vector<float>& l = *src;

  // ---- 2) top_k: rank on raw logits (temperature is monotonic, order is the same) ----
  cand_.clear();
  if (params_.top_k > 0 && params_.top_k < vocab) {
    cand_.resize(vocab);
    for (int i = 0; i < vocab; ++i) cand_[i] = i;
    const int k = params_.top_k;
    std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(),
                      [&](int a, int b) { return l[a] > l[b]; });
    cand_.resize(k);
  } else {
    cand_.resize(vocab);
    for (int i = 0; i < vocab; ++i) cand_[i] = i;
  }

  // Helper: softmax probabilities over the current candidate set, using the
  // untempered logits (this is what llama.cpp's top_p / min_p filters see).
  auto probs_of = [&](const std::vector<int>& ids) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int id : ids) mx = std::max(mx, l[id]);
    double sum = 0.0;
    std::vector<float> probs(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      const double v = std::exp(static_cast<double>(l[ids[i]] - mx));
      probs[i] = static_cast<float>(v);
      sum += v;
    }
    if (sum > 0.0)
      for (auto& v : probs) v = static_cast<float>(v / sum);
    return probs;
  };

  // ---- 3) top_p: keep the most probable prefix whose cumulative mass >= top_p ----
  if (params_.top_p < 1.0 && cand_.size() > 1) {
    const std::vector<float> probs = probs_of(cand_);
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(cand_.size());
    for (size_t i = 0; i < cand_.size(); ++i) ranked.emplace_back(probs[i], cand_[i]);
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    double cum = 0.0;
    size_t keep = 0;
    while (keep < ranked.size()) {
      cum += ranked[keep].first;
      ++keep;
      if (cum >= params_.top_p) break;
    }
    keep = std::max<size_t>(keep, 1);
    cand_.clear();
    cand_.reserve(keep);
    for (size_t i = 0; i < keep; ++i) cand_.push_back(ranked[i].second);
  }

  // ---- 4) min_p: keep tokens with p >= min_p * p_max (renormalised over the
  //        remaining candidates, exactly like llama.cpp's min_p filter) ----
  if (params_.min_p > 0.0 && cand_.size() > 1) {
    const std::vector<float> probs = probs_of(cand_);
    float pmax = 0.0f;
    for (float v : probs) pmax = std::max(pmax, v);
    const float floor_p = pmax * static_cast<float>(params_.min_p);
    size_t keep = 0;
    for (size_t i = 0; i < cand_.size(); ++i)
      if (probs[i] >= floor_p) cand_[keep++] = cand_[i];
    cand_.resize(std::max<size_t>(keep, 1));
  }

  // ---- 5) temp + dist: sample proportional to exp(logit / T) over the kept set ----
  float mx = -std::numeric_limits<float>::infinity();
  for (int id : cand_) mx = std::max(mx, l[id]);
  const float inv_temp = static_cast<float>(1.0 / params_.temperature);
  double sum = 0.0;
  scores_.resize(cand_.size());
  for (size_t i = 0; i < cand_.size(); ++i) {
    const double v = std::exp(static_cast<double>((l[cand_[i]] - mx) * inv_temp));
    scores_[i] = static_cast<float>(v);
    sum += v;
  }
  if (!(sum > 0.0)) return argmax(logits);
  double u = rng_.uniform() * sum;
  for (size_t i = 0; i < cand_.size(); ++i) {
    u -= scores_[i];
    if (u <= 0.0) return cand_[i];
  }
  return cand_.back();
}
}  // namespace sampler
}  // namespace engine
