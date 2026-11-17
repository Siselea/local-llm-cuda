// M3 gate: full 2-layer model forward (embedding -> layers -> norm -> lm_head)
// against the numpy reference logits.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "engine/runner.h"

namespace {

std::vector<float> read_f32(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<float> v(n / 4);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

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

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "build/fixtures/tiny_model";
  try {
    std::string model_path = dir;
    auto container = runner::Container::Safetensors;
    if (dir.size() > 5 && dir.substr(dir.size() - 5) == ".gguf") {
      container = runner::Container::Gguf;
    } else if (std::ifstream(dir + "/model.gguf").good()) {
      model_path = dir + "/model.gguf";
      container = runner::Container::Gguf;
    }
    runner::DenseRunner r(model_path, 64, container);
    const auto tokens = read_i32(dir + "/tokens.i32");
    const auto ref = read_f32(dir + "/ref_logits.f32");
    const auto got = r.forward(tokens);
    if (got.size() != ref.size()) {
      std::fprintf(stderr, "size mismatch: %zu vs %zu\n", got.size(), ref.size());
      return 2;
    }
    double num = 0, den = 0, max_abs = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double e = static_cast<double>(got[i]) - ref[i];
      num += e * e;
      den += static_cast<double>(ref[i]) * ref[i];
      max_abs = std::max(max_abs, std::fabs(e));
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));

    const int vocab = r.ir().vocab;
    const int n = static_cast<int>(tokens.size());
    auto argmax_last = [&](const std::vector<float>& v) {
      int best = 0;
      for (int i = 1; i < vocab; ++i)
        if (v[static_cast<size_t>(n - 1) * vocab + i] > v[static_cast<size_t>(n - 1) * vocab + best])
          best = i;
      return best;
    };
    const int a_got = argmax_last(got), a_ref = argmax_last(ref);
    std::printf("model      : %d layers, %d tokens, vocab=%d\n", r.ir().n_layer, n, vocab);
    std::printf("logits     : max_abs=%.3e  rel_l2=%.3e  argmax %d vs %d  %s\n", max_abs, rel,
                a_got, a_ref,
                (rel < 1e-4 && a_got == a_ref) ? "PASS" : "FAIL");
    return (rel < 1e-4 && a_got == a_ref) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
