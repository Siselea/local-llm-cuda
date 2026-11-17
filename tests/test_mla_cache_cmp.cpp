#include <cstdio>
#include <cmath>
#include <algorithm>
#include "engine/runner.h"
int main() {
  runner::DenseRunner r("build/fixtures/tiny_gguf_mla/model.gguf", 64, runner::Container::Gguf);
  std::vector<int> t = {3, 17};
  r.forward({t[0]}, 0);
  r.forward({t[1]}, 1);
  auto a = r.debug_mla_cache(0);
  r.forward(t, 0);
  auto b = r.debug_mla_cache(0);
  // compare only the first two positions per head (rest is uninitialised)
  const int n_head = r.ir().attn.n_head, qk = r.ir().attn.key_length, max_seq = 64;
  for (int pos = 0; pos < 2; ++pos) {
    double num = 0, den = 0, mx = 0;
    for (int h = 0; h < n_head; ++h)
      for (int d = 0; d < qk; ++d) {
        const size_t i = (static_cast<size_t>(h) * max_seq + pos) * qk + d;
        double e = (double)a[i] - b[i];
        num += e * e; den += (double)b[i] * b[i]; mx = std::max(mx, std::fabs(e));
      }
    std::printf("  pos %d: rel=%.3e max_abs=%.3e\n", pos,
                std::sqrt(num / std::max(den, 1e-30)), mx);
  }
  return 0;
}
