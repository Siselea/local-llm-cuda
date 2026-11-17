// M5: incremental decode must match full prefill (KV cache continuity),
// and a greedy loop must be self-consistent.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "engine/runner.h"

namespace {

std::vector<int> read_i32(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<int> v(n / 4);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

double rel_l2(const float* a, const float* b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double e = static_cast<double>(a[i]) - b[i];
    num += e * e;
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string path = argc > 1 ? argv[1] : "build/fixtures/tiny_model";
    std::string tokens_file = argc > 2 ? argv[2] : path + "/tokens.i32";
    auto container = runner::Container::Safetensors;
    if (path.size() > 5 && path.substr(path.size() - 5) == ".gguf")
      container = runner::Container::Gguf;
    else if (std::ifstream(path + "/model.gguf").good()) {
      path += "/model.gguf";
      container = runner::Container::Gguf;
    }
    runner::DenseRunner r(path, 64, container);
    const int vocab = static_cast<int>(r.ir().vocab);
    auto tokens = read_i32(tokens_file);
    const int n = static_cast<int>(tokens.size());
    if (n < 2) throw std::runtime_error("need >=2 tokens");

    // 1) full prefill (twice, to catch state/ordering bugs)
    const auto full = r.forward(tokens, 0);
    const auto full_again = r.forward(tokens, 0);
    std::printf("determinism    : rel_l2=%.3e\n",
                rel_l2(full_again.data(), full.data(), full.size()));
    {
      std::vector<int> pre(tokens.begin(), tokens.end() - 1);
      const auto p7 = r.forward(pre, 0);
      std::printf("prefix row %d  : rel_l2=%.3e (vs full)\n", n - 2,
                  rel_l2(p7.data() + static_cast<size_t>(n - 2) * vocab,
                         full.data() + static_cast<size_t>(n - 2) * vocab, vocab));
    }
    const float* full_last = full.data() + static_cast<size_t>(n - 1) * vocab;

    // 1b) single-token "decode" at position 0 must equal prefill row 0
    {
      const auto one = r.forward({tokens[0]}, 0);
      std::printf("1-token @pos0  : rel_l2=%.3e (vs full row 0)\n",
                  rel_l2(one.data(), full.data(), vocab));
      r.forward({tokens[0]}, 0);
      const auto step1 = r.forward({tokens[1]}, 1);
      std::printf("1-token @pos1  : rel_l2=%.3e (vs full row 1)\n",
                  rel_l2(step1.data(), full.data() + static_cast<size_t>(1) * vocab, vocab));
      const auto two = r.forward(std::vector<int>(tokens.begin(), tokens.begin() + 2), 0);
      std::printf("2-token @pos0  : rel_l2=%.3e (vs full rows 0-1)\n",
                  rel_l2(two.data(), full.data(), static_cast<size_t>(2) * vocab));
    }

    // 2) prefix prefill + 1-token decode
    std::vector<int> prefix(tokens.begin(), tokens.end() - 1);
    r.forward(prefix, 0);
    const auto step = r.forward({tokens.back()}, n - 1);
    const double rel = rel_l2(step.data(), full_last, vocab);

    // 3) greedy loop from the prefix, then recompute the whole sequence
    r.forward(prefix, 0);
    std::vector<int> gen;
    std::vector<float> last;
    int pos = n - 1;
    int next = tokens.back();
    // warm-up step so the one-off CUDA-graph capture is not counted
    {
      auto warm = r.forward({tokens[0]}, 0);
      (void)warm;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 8; ++k) {
      last = r.forward({next}, pos);
      int best = 0;
      for (int i = 1; i < vocab; ++i)
        if (last[i] > last[best]) best = i;
      gen.push_back(next);
      next = best;
      ++pos;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<int> all(prefix);
    all.insert(all.end(), gen.begin(), gen.end());
    const auto full2 = r.forward(all, 0);
    const double rel2 = rel_l2(last.data(), full2.data() + static_cast<size_t>(all.size() - 1) * vocab,
                               vocab);

    std::printf("model      : %s\n", r.ir().arch.c_str());
    std::printf("prefill/decode : rel_l2=%.3e  %s\n", rel, rel < 1e-4 ? "PASS" : "FAIL");
    std::printf("greedy loop    : %zu tokens, recompute rel_l2=%.3e  %s\n", gen.size(), rel2,
                rel2 < 1e-4 ? "PASS" : "FAIL");
    const double dt = std::chrono::duration<double>(t1 - t0).count();
    std::printf("decode speed   : %.2f tok/s (%.1f ms/token)\n",
                gen.size() / dt, dt / gen.size() * 1e3);
    std::printf("generated ids  :");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    return (rel < 1e-4 && rel2 < 1e-4) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
