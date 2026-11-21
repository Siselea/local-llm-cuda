// Sampler unit test: seeding/reproducibility, greedy, temperature, top-k,
// top-p, min-p and repetition penalty behaviour on synthetic logits.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "engine/sampler.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  std::printf("  %-62s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++failures;
}

// Deterministic pseudo logits: id 0 has the highest value, then a long decay.
std::vector<float> make_logits(int vocab, float top = 4.0f) {
  std::vector<float> l(static_cast<size_t>(vocab));
  for (int i = 0; i < vocab; ++i) {
    l[i] = top - static_cast<float>(i) * 0.01f - (((i * 37) % 11) * 0.001f);
  }
  return l;
}

std::vector<float> flat_logits(int vocab, float v = 0.0f) {
  return std::vector<float>(static_cast<size_t>(vocab), v);
}

void test_rng() {
  engine::sampler::Rng a(1234), b(1234), c(4321);
  bool same = true, differ = false, in_range = true;
  for (int i = 0; i < 1000; ++i) {
    const uint64_t x = a.next_u64(), y = b.next_u64();
    if (x != y) same = false;
    if (x != c.next_u64()) differ = true;
    const double u = a.uniform();
    const double v = b.uniform();  // keep both streams in lock step
    if (u != v) same = false;
    if (!(u >= 0.0 && u < 1.0)) in_range = false;
  }
  check(in_range, "uniform() stays in [0,1)");
  check(same, "xorshift128+ is deterministic for a seed and stays in [0,1)");
  check(differ, "different seeds produce different streams");
}

void test_greedy() {
  auto logits = make_logits(1000);
  engine::sampler::Params p;
  p.temperature = 0.0;
  engine::sampler::Sampler s(7);
  s.set_params(p);
  bool all_zero = true;
  for (int i = 0; i < 200; ++i) all_zero = all_zero && s.sample(logits, {}) == 0;
  check(all_zero, "temperature=0 always picks the argmax");
  logits[500] = 100.0f;
  check(engine::sampler::Sampler::argmax(logits) == 500, "argmax finds the maximum");
}

void test_reproducible() {
  auto logits = make_logits(2000);
  engine::sampler::Params p;
  p.temperature = 0.9;
  p.top_p = 0.9;
  p.top_k = 40;
  p.seed = 42;
  engine::sampler::Sampler s1(42), s2(42);
  s1.set_params(p);
  s2.set_params(p);
  bool equal = true;
  for (int i = 0; i < 256; ++i) {
    // feed the same growing context to both
    std::vector<int> ctx(static_cast<size_t>(i), i % 7);
    if (s1.sample(logits, ctx) != s2.sample(logits, ctx)) equal = false;
  }
  check(equal, "same seed + params -> identical sampled sequence");

  // fresh sampler with same seed reproduces from scratch
  engine::sampler::Sampler s3(42);
  s3.set_params(p);
  engine::sampler::Sampler s4(42);
  s4.set_params(p);
  bool restart_equal = true;
  for (int i = 0; i < 128; ++i)
    if (s3.sample(logits, {}) != s4.sample(logits, {})) restart_equal = false;
  check(restart_equal, "reseeded sampler reproduces the same sequence");
}

void test_top_k() {
  auto logits = make_logits(5000);
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_k = 3;
  p.seed = 1;
  engine::sampler::Sampler s(1);
  s.set_params(p);
  bool in_range = true;
  for (int i = 0; i < 500; ++i) {
    const int id = s.sample(logits, {});
    if (id < 0 || id > 2) in_range = false;  // only the 3 best logits are possible
  }
  check(in_range, "top_k=3 only samples the 3 best ids");
}

void test_top_p() {
  // one dominant token -> top_p 0.5 must lock onto it
  std::vector<float> logits = flat_logits(1000, -10.0f);
  logits[77] = 0.0f;  // ~e^10 times more likely than the rest
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_p = 0.5;
  p.top_k = 0;
  p.seed = 3;
  engine::sampler::Sampler s(3);
  s.set_params(p);
  bool always_77 = true;
  for (int i = 0; i < 200; ++i)
    if (s.sample(logits, {}) != 77) always_77 = false;
  check(always_77, "top_p keeps the dominant token only");

  // flat distribution with top_p=1 must reach many different ids
  auto flat = flat_logits(100, 0.0f);
  engine::sampler::Params q;
  q.temperature = 1.0;
  q.top_p = 1.0;
  q.top_k = 0;
  q.seed = 11;
  engine::sampler::Sampler s2(11);
  s2.set_params(q);
  std::map<int, int> hist;
  for (int i = 0; i < 2000; ++i) hist[s2.sample(flat, {})]++;
  check(hist.size() > 50, "top_p=1 samples the whole flat distribution");
}

void test_min_p() {
  std::vector<float> logits = flat_logits(100, -30.0f);
  logits[5] = 10.0f;
  logits[6] = 9.9f;
  logits[7] = 9.8f;
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_p = 1.0;
  p.top_k = 0;
  p.min_p = 0.99;  // only tokens with p >= 0.99 * p_max survive
  p.seed = 5;
  engine::sampler::Sampler s(5);
  s.set_params(p);
  bool only_best = true;
  for (int i = 0; i < 500; ++i) {
    const int id = s.sample(logits, {});
    if (id != 5) only_best = false;
  }
  check(only_best, "min_p=0.99 keeps only the most likely token here");
}

void test_repetition_penalty() {
  std::vector<float> logits = flat_logits(100, 0.0f);
  logits[0] = 1.0f;
  logits[1] = 0.9f;
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_k = 2;
  p.repetition_penalty = 10.0;
  p.seed = 9;
  engine::sampler::Sampler s(9);
  s.set_params(p);
  // Penalise id 0 hard: id 1 must dominate despite the smaller logit.
  logits[0] = 5.0f;
  logits[1] = 4.0f;
  std::vector<int> recent = {0};
  int n1 = 0;
  for (int i = 0; i < 2000; ++i)
    if (s.sample(logits, recent) == 1) ++n1;
  check(n1 > 1900, "repetition_penalty suppresses a repeated token (>95% shifted)");
  int n0 = 0;
  for (int i = 0; i < 2000; ++i)
    if (s.sample(logits, {}) == 0) ++n0;
  check(n0 > 1000, "without the penalty the higher logit still wins");

  // penalty only applies within repeat_last_n
  p.repeat_last_n = 1;
  p.repetition_penalty = 10.0;
  s.set_params(p);
  std::vector<int> older = {0, 3, 3, 3, 3};
  bool can_pick_0 = false;
  for (int i = 0; i < 200; ++i)
    if (s.sample(logits, older) == 0) can_pick_0 = true;
  check(can_pick_0, "repeat_last_n window expires older tokens");
}

// ---------------------------------------------------------------------------
// Bounded-candidate sampling (perf(serve) 2026-11-21): the serve non-greedy path.
// Must keep greedy bit-identical, keep top_k exact, and only approximate top_p
// when top_k == 0 (top-1024 candidate cap).
// ---------------------------------------------------------------------------
void test_bounded_greedy_parity() {
  engine::sampler::Params p;
  p.temperature = 0.0;
  bool same = true;
  for (int seed = 0; seed < 5; ++seed) {
    auto logits = make_logits(5000, 3.0f);
    logits[static_cast<size_t>(17 + seed)] = 50.0f;
    engine::sampler::Sampler a(static_cast<uint64_t>(seed));
    engine::sampler::Sampler b(static_cast<uint64_t>(seed));
    a.set_params(p);
    b.set_params(p);
    for (int i = 0; i < 50; ++i) {
      const int ra = a.sample(logits, {});
      const int rb = b.sample_bounded(logits, {});
      if (!(ra == rb && ra == engine::sampler::Sampler::argmax(logits))) same = false;
    }
  }
  check(same, "bounded: temperature=0 is an exact full-row argmax (same as sample)");
}

void test_bounded_top_k_exact() {
  auto logits = make_logits(5000, 4.0f);
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_k = 3;
  p.seed = 1;
  engine::sampler::Sampler s(1);
  s.set_params(p);
  s.set_candidate_cap(64);
  bool in_range = true;
  for (int i = 0; i < 500; ++i) {
    const int id = s.sample_bounded(logits, {});
    if (id < 0 || id > 2) in_range = false;
  }
  check(in_range, "bounded: top_k=3 only samples the 3 best ids (candidate cap 64)");
}

void test_bounded_top_p() {
  std::vector<float> logits = flat_logits(1000, -10.0f);
  logits[77] = 0.0f;
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_p = 0.5;
  p.top_k = 0;
  p.seed = 3;
  engine::sampler::Sampler s(3);
  s.set_params(p);
  bool always_77 = true;
  for (int i = 0; i < 200; ++i)
    if (s.sample_bounded(logits, {}) != 77) always_77 = false;
  check(always_77, "bounded: top_p keeps the dominant token only");

  auto flat = flat_logits(100, 0.0f);
  engine::sampler::Params q;
  q.temperature = 1.0;
  q.top_p = 1.0;
  q.top_k = 0;
  q.seed = 11;
  engine::sampler::Sampler s2(11);
  s2.set_params(q);
  std::map<int, int> hist;
  for (int i = 0; i < 2000; ++i) hist[s2.sample_bounded(flat, {})]++;
  check(hist.size() > 50, "bounded: candidate set == vocab samples the whole distribution");
}

void test_bounded_reproducible() {
  auto logits = make_logits(2000);
  engine::sampler::Params p;
  p.temperature = 0.9;
  p.top_p = 0.9;
  p.top_k = 40;
  p.seed = 42;
  engine::sampler::Sampler s1(42), s2(42);
  s1.set_params(p);
  s2.set_params(p);
  bool equal = true;
  for (int i = 0; i < 256; ++i) {
    std::vector<int> ctx(static_cast<size_t>(i), i % 7);
    if (s1.sample_bounded(logits, ctx) != s2.sample_bounded(logits, ctx)) equal = false;
  }
  check(equal, "bounded: same seed + params -> identical sampled sequence");
}

void test_bounded_top_k0_truncation() {
  // flat logits over a 5000-token vocab: top_k=0 must stay inside the top
  // candidate_cap() logits (documented approximation), while the full-vocab
  // reference can reach the whole distribution.
  auto flat = flat_logits(5000, 0.0f);
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_p = 1.0;
  p.top_k = 0;
  p.seed = 21;
  engine::sampler::Sampler a(21), b(21);
  a.set_params(p);
  b.set_params(p);
  std::map<int, int> bounded_hist;
  int tail_full = 0;
  for (int i = 0; i < 3000; ++i) {
    const int idb = b.sample_bounded(flat, {});
    bounded_hist[idb]++;
    if (a.sample(flat, {}) >= engine::sampler::Sampler::kDefaultCandidateCap) ++tail_full;
  }
  const bool bounded_small = bounded_hist.size() <=
                             static_cast<size_t>(engine::sampler::Sampler::kDefaultCandidateCap);
  check(bounded_small, "bounded: top_k=0 candidate set capped at 1024 tokens");
  check(tail_full > 0, "reference: full-vocab top_k=0 can sample below top-1024 (documented diff)");
}

void test_bounded_penalty() {
  std::vector<float> logits = flat_logits(100, 0.0f);
  logits[0] = 5.0f;
  logits[1] = 4.0f;
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_k = 2;
  p.repetition_penalty = 10.0;
  p.seed = 9;
  engine::sampler::Sampler s(9);
  s.set_params(p);
  s.set_candidate_cap(64);
  std::vector<int> recent = {0};
  int n1 = 0;
  for (int i = 0; i < 2000; ++i)
    if (s.sample_bounded(logits, recent) == 1) ++n1;
  check(n1 > 1900, "bounded: repetition penalty suppresses a repeated candidate (>95%)");
}

void test_external_candidates() {
  // Simulates the future device-top-k hook: the runner hands back 5 candidates.
  std::vector<float> logits = flat_logits(1000, 0.0f);
  const int ids[5] = {3, 7, 11, 42, 99};
  const float vals[5] = {-2.0f, -1.5f, -1.0f, -0.5f, 0.0f};
  for (int i = 0; i < 5; ++i) logits[static_cast<size_t>(ids[i])] = vals[i];
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_k = 0;
  p.top_p = 1.0;
  p.seed = 5;
  engine::sampler::Sampler s(5);
  s.set_params(p);
  s.set_candidates(ids, vals, 5);
  bool only_candidates = true;
  std::map<int, int> hist;
  for (int i = 0; i < 1000; ++i) {
    const int id = s.sample_bounded(logits, {});
    if (!(id == 3 || id == 7 || id == 11 || id == 42 || id == 99)) only_candidates = false;
    hist[id]++;
  }
  check(only_candidates, "external candidates: only the supplied ids are sampled");
  check(hist[99] > hist[3], "external candidates: their own logits drive the distribution");
  s.clear_candidates();
  check(!s.has_external_candidates(), "external candidates: clear_candidates() resets the hook");

  // deterministic given the same seed
  engine::sampler::Sampler a(99), b(99);
  a.set_params(p);
  b.set_params(p);
  a.set_candidates(ids, vals, 5);
  b.set_candidates(ids, vals, 5);
  bool same = true;
  for (int i = 0; i < 256; ++i)
    if (a.sample_bounded(logits, {}) != b.sample_bounded(logits, {})) same = false;
  check(same, "external candidates: deterministic for a seed");
}

void test_bounded_penalised_topk_matches_full() {
  // With top_k>0 the bounded path must select the same top-k set as the
  // full-vocab reference: a penalised leader is replaced by the next logit.
  auto logits = make_logits(200, 5.0f);
  logits[0] = 6.0f;  // leader, but repeated
  logits[1] = 5.999f;
  logits[2] = 5.998f;
  logits[3] = 5.997f;
  engine::sampler::Params p;
  p.temperature = 1.0;
  p.top_k = 3;
  p.repetition_penalty = 100.0;
  p.seed = 13;
  engine::sampler::Sampler a(13), b(13);
  a.set_params(p);
  b.set_params(p);
  std::vector<int> recent = {0};
  std::map<int, int> ha, hb;
  for (int i = 0; i < 1500; ++i) {
    ha[a.sample(logits, recent)]++;
    hb[b.sample_bounded(logits, recent)]++;
  }
  auto keys = [](const std::map<int, int>& h) {
    std::vector<int> k;
    for (const auto& kv : h) k.push_back(kv.first);
    return k;
  };
  check(keys(ha) == keys(hb), "bounded: penalised top_k set matches the full-vocab reference");
}

}  // namespace

int main() {
  std::printf("sampler unit tests\n");
  test_rng();
  test_greedy();
  test_reproducible();
  test_top_k();
  test_top_p();
  test_min_p();
  test_repetition_penalty();
  test_bounded_greedy_parity();
  test_bounded_top_k_exact();
  test_bounded_top_p();
  test_bounded_reproducible();
  test_bounded_top_k0_truncation();
  test_bounded_penalty();
  test_external_candidates();
  test_bounded_penalised_topk_matches_full();
  std::printf("\n%s\n", failures == 0 ? "SAMPLER TESTS PASSED" : "SAMPLER TESTS FAILED");
  return failures == 0 ? 0 : 1;
}
