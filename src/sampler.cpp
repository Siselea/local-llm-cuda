#include "engine/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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
// ---------------------------------------------------------------------------
// Bounded-candidate sampling: the serve default for every non-greedy token.
//
// The reference sample() above walks the full vocabulary (fill 248k ids, partial
// sort / exp / sort per token).  For 35B/9B vocab=248320 that is ~0.2 ms (top_k>0)
// to ~1.8 ms (top_k=0) per token, all of it on the calling thread.  This variant
// reduces the row to <= candidate_cap() candidates first (std::nth_element, O(n))
// and runs the identical sampler chain on that set:
//   * repetition penalty only has to touch the candidate ids (no 1 MB row copy);
//   * no complete-vocabulary exp / sort ever happens;
//   * greedy (temperature <= 0) is still an exact full-row argmax, so greedy output
//     is bit-identical to before.
// The approximation is that top_p / min_p / softmax probabilities are normalised
// over the candidates instead of the full vocabulary.  For top_k > 0 the top_k set
// itself is exact (candidates = max(top_k, 64) raw-top logits); for top_k == 0 the
// nuclei are approximated by the top candidate_cap() logits (documented in
// docs/CONFIG.md and exposed through /props.sampling).
// ---------------------------------------------------------------------------
void Sampler::set_candidate_cap(int cap) {
  candidate_cap_ = std::max(64, std::min(cap, 1000000));
}

void Sampler::set_candidates(const int* ids, const float* logits, int n) {
  if (!ids || !logits || n <= 0) {
    clear_candidates();
    return;
  }
  ext_ids_.assign(ids, ids + n);
  ext_vals_.assign(logits, logits + n);
  have_ext_candidates_ = true;
}

void Sampler::clear_candidates() {
  have_ext_candidates_ = false;
  ext_ids_.clear();
  ext_vals_.clear();
}

void Sampler::select_candidates(const std::vector<float>& logits) {
  const int vocab = static_cast<int>(logits.size());
  const int k = params_.top_k;
  const int cap = k > 0 ? std::max(k, 64) : candidate_cap_;
  const int n = std::max(1, std::min(cap, vocab));
  if (n <= 128) {
    // Single-pass top-N scan with a sorted insertion window (the serve default:
    // top_k=20 -> n=64).  No 248k-id index array and no heap bookkeeping: the row
    // is read sequentially and almost every element is rejected by one compare
    // against the window tail.  The rejection loop is vectorised (8 floats per
    // compare against the tail) because the 1 MB row scan dominates the cost.
    cand_.resize(static_cast<size_t>(n));
    cand_val_.assign(static_cast<size_t>(n), -std::numeric_limits<float>::infinity());
    size_t cnt = 0;
    auto insert_one = [&](int id, float v) {
      if (cnt == static_cast<size_t>(n) && v <= cand_val_[static_cast<size_t>(n - 1)]) return;
      size_t pos = cnt < static_cast<size_t>(n) ? cnt : static_cast<size_t>(n - 1);
      while (pos > 0 && cand_val_[pos - 1] < v) {
        cand_val_[pos] = cand_val_[pos - 1];
        cand_[pos] = cand_[pos - 1];
        --pos;
      }
      cand_val_[pos] = v;
      cand_[pos] = id;
      if (cnt < static_cast<size_t>(n)) ++cnt;
    };
    int i = 0;
    // 1) fill the window (n <= 128 elements)
    for (; i < vocab && cnt < static_cast<size_t>(n); ++i)
      insert_one(i, logits[static_cast<size_t>(i)]);
    // 2) vectorised skip over the rest of the row
#if defined(__AVX2__)
    for (; i + 8 <= vocab; i += 8) {
      const __m256 v = _mm256_loadu_ps(logits.data() + i);
      const __m256 tail = _mm256_set1_ps(cand_val_[static_cast<size_t>(n - 1)]);
      const unsigned mask =
          static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(v, tail, _CMP_GT_OQ)));
      if (mask == 0) continue;
      for (int lane = 0; lane < 8; ++lane)
        if (mask & (1u << lane))
          insert_one(i + lane, logits[static_cast<size_t>(i + lane)]);
    }
#endif
    // 3) scalar tail + non-AVX2 fallback
    for (; i < vocab; ++i) insert_one(i, logits[static_cast<size_t>(i)]);
    cand_.resize(cnt);
    cand_val_.resize(cnt);
    return;
  }
  // Large candidate sets (top_k=0 -> 1024): fixed-size min-heap, one sequential pass.
  // Most elements fail the heap-root comparison (one float compare, no random access).
  auto cmp = [](const HeapItem& a, const HeapItem& b) { return a.v > b.v; };
  heap_.clear();
  heap_.reserve(static_cast<size_t>(n) + 1);
  for (int i = 0; i < vocab; ++i) {
    const float v = logits[static_cast<size_t>(i)];
    if (static_cast<int>(heap_.size()) < n) {
      heap_.push_back(HeapItem{v, i});
      std::push_heap(heap_.begin(), heap_.end(), cmp);
    } else if (v > heap_.front().v) {
      std::pop_heap(heap_.begin(), heap_.end(), cmp);
      heap_.back() = HeapItem{v, i};
      std::push_heap(heap_.begin(), heap_.end(), cmp);
    }
  }
  cand_.resize(heap_.size());
  cand_val_.resize(heap_.size());
  for (size_t i = 0; i < heap_.size(); ++i) {
    cand_[i] = heap_[i].id;
    cand_val_[i] = heap_[i].v;
  }
}

void Sampler::penalise_candidates(const std::vector<int>& recent_ids, int vocab) {
  const double penalty = params_.repetition_penalty;
  if (penalty == 1.0 || recent_ids.empty() || cand_.empty()) return;
  const int n = static_cast<int>(recent_ids.size());
  int begin = 0;
  if (params_.repeat_last_n >= 0 && params_.repeat_last_n < n)
    begin = n - params_.repeat_last_n;
  if (pen_flag_.size() != static_cast<size_t>(vocab))
    pen_flag_.assign(static_cast<size_t>(vocab), 0);
  pen_ids_.clear();
  for (int i = begin; i < n; ++i) {
    const int id = recent_ids[static_cast<size_t>(i)];
    if (id < 0 || id >= vocab) continue;
    if (!pen_flag_[static_cast<size_t>(id)]) {
      pen_flag_[static_cast<size_t>(id)] = 1;
      pen_ids_.push_back(id);
    }
  }
  for (size_t i = 0; i < cand_.size(); ++i) {
    if (!pen_flag_[static_cast<size_t>(cand_[i])]) continue;
    float& v = cand_val_[i];
    if (v > 0.0f)
      v = static_cast<float>(v / penalty);
    else
      v = static_cast<float>(v * penalty);
  }
  for (int id : pen_ids_) pen_flag_[static_cast<size_t>(id)] = 0;
  pen_ids_.clear();
}

int Sampler::sample_bounded(const std::vector<float>& logits,
                            const std::vector<int>& recent_ids) {
  const int vocab = static_cast<int>(logits.size());
  if (vocab <= 0) return -1;
  // greedy: identical to sample() (exact full-row argmax)
  if (!(params_.temperature > 0.0)) return argmax(logits);

  if (have_ext_candidates_ && !ext_ids_.empty()) {
    cand_ = ext_ids_;
    cand_val_ = ext_vals_;
  } else {
    select_candidates(logits);
  }
  if (cand_.empty()) return argmax(logits);
  penalise_candidates(recent_ids, vocab);

  // ---- 1) top_k: keep at most k candidates (candidate list is already bounded) ----
  const int k = params_.top_k;
  if (k > 0 && k < static_cast<int>(cand_.size())) {
    order_.resize(cand_.size());
    for (size_t i = 0; i < cand_.size(); ++i) order_[i] = static_cast<int>(i);
    std::partial_sort(order_.begin(), order_.begin() + k, order_.end(),
                      [&](int a, int b) { return cand_val_[static_cast<size_t>(a)] >
                                                 cand_val_[static_cast<size_t>(b)]; });
    std::vector<int> ids(static_cast<size_t>(k));
    std::vector<float> vals(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
      ids[static_cast<size_t>(i)] = cand_[static_cast<size_t>(order_[static_cast<size_t>(i)])];
      vals[static_cast<size_t>(i)] =
          cand_val_[static_cast<size_t>(order_[static_cast<size_t>(i)])];
    }
    cand_.swap(ids);
    cand_val_.swap(vals);
  }

  // ---- 2) top_p / min_p on the candidate set (untempered probabilities) ----
  if ((params_.top_p < 1.0 || params_.min_p > 0.0) && cand_.size() > 1) {
    auto probs = [&]() {
      float mx = -std::numeric_limits<float>::infinity();
      for (float v : cand_val_) mx = std::max(mx, v);
      double sum = 0.0;
      scores_.resize(cand_val_.size());
      for (size_t i = 0; i < cand_val_.size(); ++i) {
        const double v = std::exp(static_cast<double>(cand_val_[i] - mx));
        scores_[i] = static_cast<float>(v);
        sum += v;
      }
      if (sum > 0.0)
        for (auto& v : scores_) v = static_cast<float>(v / sum);
    };
    probs();
    if (params_.top_p < 1.0) {
      order_.resize(cand_.size());
      for (size_t i = 0; i < cand_.size(); ++i) order_[i] = static_cast<int>(i);
      std::sort(order_.begin(), order_.end(), [&](int a, int b) {
        const float pa = scores_[static_cast<size_t>(a)];
        const float pb = scores_[static_cast<size_t>(b)];
        if (pa != pb) return pa > pb;
        return cand_[static_cast<size_t>(a)] < cand_[static_cast<size_t>(b)];
      });
      double cum = 0.0;
      size_t keep = 0;
      while (keep < order_.size()) {
        cum += scores_[static_cast<size_t>(order_[keep])];
        ++keep;
        if (cum >= params_.top_p) break;
      }
      keep = std::max<size_t>(keep, 1);
      std::vector<int> ids; ids.reserve(keep);
      std::vector<float> vals; vals.reserve(keep);
      for (size_t i = 0; i < keep; ++i) {
        const size_t ci = static_cast<size_t>(order_[i]);
        ids.push_back(cand_[ci]);
        vals.push_back(cand_val_[ci]);
      }
      cand_.swap(ids);
      cand_val_.swap(vals);
    }
    if (params_.min_p > 0.0 && cand_.size() > 1) {
      probs();
      float pmax = 0.0f;
      for (float v : scores_) pmax = std::max(pmax, v);
      const float floor_p = pmax * static_cast<float>(params_.min_p);
      size_t keep = 0;
      for (size_t i = 0; i < cand_.size(); ++i)
        if (scores_[i] >= floor_p) {
          cand_[keep] = cand_[i];
          cand_val_[keep] = cand_val_[i];
          ++keep;
        }
      cand_.resize(std::max<size_t>(keep, 1));
      cand_val_.resize(cand_.size());
    }
  }

  // ---- 3) temperature + distribution over the kept candidates ----
  float mx = -std::numeric_limits<float>::infinity();
  for (float v : cand_val_) mx = std::max(mx, v);
  const float inv_temp = static_cast<float>(1.0 / params_.temperature);
  double sum = 0.0;
  scores_.resize(cand_val_.size());
  for (size_t i = 0; i < cand_val_.size(); ++i) {
    const double v = std::exp(static_cast<double>((cand_val_[i] - mx) * inv_temp));
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
