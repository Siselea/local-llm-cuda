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
  std::printf("\n%s\n", failures == 0 ? "SAMPLER TESTS PASSED" : "SAMPLER TESTS FAILED");
  return failures == 0 ? 0 : 1;
}
