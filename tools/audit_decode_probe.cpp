// Decode budget probe: drives DenseRunner directly (no CLI token-list limit) so long
// contexts can be measured, and prints per-step decode wall time.  Kernel-level numbers
// come from ENGINE_KPROF (dumped when the runner is destroyed) and segment accounting
// from ENGINE_PROF_ACC.
//
//   PROBE_CTX=9400 ./build/decode_probe.exe <model.gguf> --prompt-len 9100 --steps 3
//
// Output: one "step N: X.XXX ms" line per decode step plus the median, and a gen token
// sequence for sanity (greedy).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/runner.h"

namespace {

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int arg_int(int argc, char** argv, const char* name, int def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], name)) return std::atoi(argv[i + 1]);
  return def;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> [--prompt-len N] [--steps K] [--ctx M] [--topk-k K]\n",
                 argv[0]);
    return 2;
  }
  const std::string model = argv[1];
  const int prompt_len = arg_int(argc, argv, "--prompt-len", 32);
  const int steps = arg_int(argc, argv, "--steps", 3);
  const int ctx = arg_int(argc, argv, "--ctx", prompt_len + steps + 64);
  const int topk_k = arg_int(argc, argv, "--topk-k", 5);
  const int token = arg_int(argc, argv, "--token", 760);

  std::printf("probe model=%s prompt_len=%d steps=%d ctx=%d topk_k=%d\n", model.c_str(),
              prompt_len, steps, ctx, topk_k);
  runner::DenseRunner r(model, ctx, runner::Container::Gguf);
  const int vocab = r.vocab();
  std::vector<int> ids(static_cast<size_t>(prompt_len), token);
  // vary the ids a little so the routed-expert pattern is not degenerate
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>((i * 131 + 7) % vocab);

  const auto t_pre0 = std::chrono::steady_clock::now();
  r.forward(ids, 0, /*last_row_only=*/true);
  const auto t_pre1 = std::chrono::steady_clock::now();
  std::printf("prefill %d tokens: %.1f ms (%.2f tok/s)\n", prompt_len,
              std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count(),
              prompt_len * 1000.0 /
                  std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

  std::vector<double> ms;
  std::vector<int> gen;
  int pos = prompt_len;
  int last = ids.back();
  for (int s = 0; s < steps; ++s) {
    std::vector<int> one = {last};
    const auto t0 = std::chrono::steady_clock::now();
    runner::DenseRunner::TopKResult tk;
    const auto full = r.forward_topk(one, pos, topk_k, &tk);
    const auto t1 = std::chrono::steady_clock::now();
    const double one_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms.push_back(one_ms);
    int next = -1;
    if (!tk.idx.empty()) next = tk.idx[0];
    else if (!full.empty()) {
      next = 0;
      for (int i = 1; i < vocab; ++i)
        if (full[i] > full[next]) next = i;
    }
    if (s == 0) std::printf("head path: %s\n", tk.idx.empty() ? "full-row" : "device-topk");
    gen.push_back(next);
    std::printf("step %d: %.3f ms\n", s, one_ms);
    last = next;
    ++pos;
  }
  std::printf("decode median %.3f ms/token (%.2f tok/s)\n", median(ms),
              1000.0 / median(ms));
  std::printf("gen:");
  for (int g : gen) std::printf(" %d", g);
  std::printf("\n");
  return 0;
}
