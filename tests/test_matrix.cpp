// Loads a model once and evaluates many (prompt, mla-layout) combinations.
// Input on stdin: one line per prompt: "<comma separated ids>"
// Output per line: "<layout>|<prompt#>: <top5 ids>"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "engine/runner.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf> [n_topk]\n", argv[0]);
    return 2;
  }
  try {
    int n_top = argc > 2 ? std::atoi(argv[2]) : 5;
    std::vector<std::vector<int>> prompts;
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      std::vector<int> ids;
      size_t p = 0;
      while (p < line.size()) {
        size_t q = line.find(',', p);
        if (q == std::string::npos) q = line.size();
        ids.push_back(std::atoi(line.substr(p, q - p).c_str()));
        p = q + 1;
      }
      if (!ids.empty()) prompts.push_back(ids);
    }
    if (prompts.empty()) throw std::runtime_error("no prompts on stdin");
    int max_len = 0;
    for (auto& p : prompts) max_len = std::max<int>(max_len, static_cast<int>(p.size()));
    auto container = std::string(argv[1]).find(".gguf") != std::string::npos
                         ? runner::Container::Gguf
                         : runner::Container::Safetensors;
    runner::DenseRunner r(argv[1], max_len + 16, container);
    const int vocab = static_cast<int>(r.ir().vocab);

    const char* qnames[3] = {"perhead", "global", "global_pe"};
    const char* knames[4] = {"global", "perhead", "global_vk", "perhead_vk"};
    const char* rnames[3] = {"yarn", "plain", "noscale"};
    const char* qsel = std::getenv("MATRIX_Q");
    const int qlo = qsel ? std::atoi(qsel) : 0;
    const int qhi = qsel ? qlo + 1 : 3;
    const char* ksel = std::getenv("MATRIX_K");
    const int klo = ksel ? std::atoi(ksel) : 0;
    const int khi = ksel ? klo + 1 : 4;
    const bool sweep_rope = std::getenv("MATRIX_ROPE") != nullptr;
    for (int qm = qlo; qm < qhi && !r.ir().attn.variant.empty(); ++qm) {
      if (r.ir().attn.variant != "mla") break;
      for (int km = klo; km < khi; ++km) {
        r.set_mla_layout(qm, km);
        const int rope_lo = sweep_rope ? 0 : 0, rope_hi = sweep_rope ? 3 : 1;
        for (int rm = rope_lo; rm < rope_hi; ++rm) {
          r.set_rope_mode(rm);
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
          auto logits = r.forward(prompts[pi], 0);
          std::vector<float> last(logits.end() - vocab, logits.end());
          std::vector<int> order(vocab);
          for (int i = 0; i < vocab; ++i) order[i] = i;
          std::partial_sort(order.begin(), order.begin() + n_top, order.end(),
                            [&](int a, int b) { return last[a] > last[b]; });
          std::printf("%s/%s/%s|%zu:", qnames[qm], knames[km], rnames[rm], pi);
          for (int k = 0; k < n_top; ++k) std::printf(" %d", order[k]);
          std::printf("\n");
          std::fflush(stdout);
        }
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
