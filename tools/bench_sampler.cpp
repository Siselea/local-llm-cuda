// Microbenchmark: full-vocabulary sampler vs bounded-candidate sampler.
//
//   before = Sampler::sample()         (fill 248k ids + partial_sort/exp/sort)
//   after  = Sampler::sample_bounded() (nth_element top-N + sampler chain on <=1024)
//
// Build: $PY -m ziglang c++ -std=c++17 -O2 -Iinclude tools/bench_sampler.cpp src/sampler.cpp \
//            -o build/bench_sampler.exe
// Run:   build/bench_sampler.exe [vocab] [iters]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine/sampler.h"

namespace {

using engine::sampler::Params;
using engine::sampler::Sampler;

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Qwen3-family logits: a smooth decay with some structure (not flat: a flat row
// would make the top-p path unrealistically expensive in the reference sampler).
std::vector<float> make_logits(int vocab) {
  std::vector<float> l(static_cast<size_t>(vocab));
  for (int i = 0; i < vocab; ++i) {
    l[static_cast<size_t>(i)] =
        -0.001f * static_cast<float>(i) - static_cast<float>((i * 37) % 13) * 0.01f;
  }
  l[42] += 3.0f;
  l[1000] += 2.0f;
  l[17] += 1.5f;
  return l;
}

double time_sample(Sampler& s, const std::vector<float>& logits,
                   const std::vector<int>& ctx, int iters, bool bounded) {
  // warm-up
  for (int i = 0; i < 5; ++i) (void)(bounded ? s.sample_bounded(logits, ctx) : s.sample(logits, ctx));
  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(iters));
  for (int it = 0; it < iters; ++it) {
    const auto t0 = std::chrono::steady_clock::now();
    volatile int id = bounded ? s.sample_bounded(logits, ctx) : s.sample(logits, ctx);
    const auto t1 = std::chrono::steady_clock::now();
    (void)id;
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return median(ms);
}

void run_group(const char* name, int top_k, double top_p, double min_p,
               const std::vector<float>& logits, const std::vector<int>& ctx, int iters) {
  Params p;
  p.temperature = 0.6;
  p.top_p = top_p;
  p.top_k = top_k;
  p.min_p = min_p;
  p.seed = 1;
  Sampler ref(7), bounded(7);
  ref.set_params(p);
  bounded.set_params(p);
  const double before = time_sample(ref, logits, ctx, iters, false);
  const double after = time_sample(bounded, logits, ctx, iters, true);
  std::printf("%-10s top_k=%-4d top_p=%.2f min_p=%.2f | before %8.1f us/token | "
              "after %7.1f us/token | saved %8.1f us (%+.0f%%)\n",
              name, top_k, top_p, min_p, before * 1000.0, after * 1000.0,
              (before - after) * 1000.0,
              before > 0 ? 100.0 * (before - after) / before : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  const int vocab = argc > 1 ? std::atoi(argv[1]) : 248320;  // Qwen3.x vocab
  const int iters = argc > 2 ? std::atoi(argv[2]) : 300;
  std::printf("sampler microbench: vocab=%d, %d iters each (median)\n", vocab, iters);
  const std::vector<float> logits = make_logits(vocab);
  {
    // baseline: a single sequential pass over the row (argmax)
    long long sum = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) sum += engine::sampler::Sampler::argmax(logits);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("scan baseline: argmax %.1f us/pass (sink=%lld)\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count() * 1000.0 / iters,
                static_cast<long long>(sum % 7));
  }
  std::vector<int> ctx;
  ctx.reserve(256);
  for (int i = 0; i < 200; ++i) ctx.push_back(i * 977 % vocab);
  run_group("webui", 20, 0.95, 0.0, logits, ctx, iters);
  run_group("topk-nofilt", 20, 1.0, 0.0, logits, ctx, iters);
  run_group("nofilt-k0", 0, 1.0, 0.0, logits, ctx, iters);
  run_group("top_p-only", 0, 0.95, 0.0, logits, ctx, iters);
  run_group("min_p-only", 0, 1.0, 0.05, logits, ctx, iters);
  return 0;
}
