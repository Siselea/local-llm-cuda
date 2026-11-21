// Independent validation: device top-k candidates vs the full-row host reference.
//
// For one decode step after a short prefill:
//   1. prefill prompt          -> state at position len
//   2. prefix_state_save(0)    -> clean recurrent state snapshot
//   3. forward_topk_candidates({t}, len, k)     (device head, no full row to host)
//   4. prefix_state_restore(0)                  (same state before the host call)
//   5. forward({t}, len, true)                  (full logits row to host)
//   6. compare the k (id,logit) pairs with the host selection rule (value desc,
//      ties -> smaller token id).
// The two calls start from the same SSM state; the KV row written by one call at
// position `len` is overwritten by the other before attention reads it, so this is
// a valid A/B for every model type (dense/MLA/MoE) without two model loads.
//
// Build/run:
//   PY=... ziglang c++ -std=c++17 -O2 -Iinclude <sources> build/validate_topk_candidates.cpp \
//       -o build/validate_topk_candidates.exe
//   ENGINE_HEAD_TOPK_MAX=128 ./build/validate_topk_candidates.exe <model.gguf> [prompt ids...]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/config.h"
#include "engine/runner.h"

namespace {

struct Pair {
  float v;
  int i;
};

std::vector<Pair> host_topk(const std::vector<float>& row, int k) {
  std::vector<int> idx(row.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
  std::vector<Pair> best;
  best.reserve(static_cast<size_t>(k));
  std::vector<char> taken(row.size(), 0);
  for (int q = 0; q < k; ++q) {
    float bv = 0.f;
    int bi = -1;
    for (size_t i = 0; i < row.size(); ++i) {
      if (taken[i]) continue;
      if (bi < 0 || row[i] > bv || (row[i] == bv && static_cast<int>(i) < bi)) {
        bv = row[i];
        bi = static_cast<int>(i);
      }
    }
    if (bi < 0) break;
    taken[static_cast<size_t>(bi)] = 1;
    best.push_back({bv, bi});
  }
  return best;
}

int failures = 0;

void check(runner::DenseRunner& r, const char* model, const std::vector<int>& prompt,
           int next_token) {
  std::vector<float> row;
  {
    // prefill the prompt, then snapshot the state at position prompt.size()
    r.forward(prompt, 0, /*last_row_only=*/true);
    if (!r.prefix_state_save(0)) {
      std::printf("FAIL %s: prefix_state_save failed\n", model);
      ++failures;
      return;
    }
  }
  // Pure determinism check that does NOT rely on a restored mid-sequence state:
  // save the clean state at position 0, run prefill+decode twice from it, compare.
  {
    std::vector<float> r1, r2;
    const std::vector<int> one = {next_token};
    if (r.prefix_state_save(0)) {
      r.forward(prompt, 0, /*last_row_only=*/true);
      r1 = r.forward(one, static_cast<int>(prompt.size()), /*last_row_only=*/true);
      r.prefix_state_restore(0);
      r.forward(prompt, 0, /*last_row_only=*/true);
      r2 = r.forward(one, static_cast<int>(prompt.size()), /*last_row_only=*/true);
      double mx = 0.0;
      for (size_t i = 0; i < r1.size() && i < r2.size(); ++i)
        mx = std::max(mx, std::fabs(static_cast<double>(r1[i]) - r2[i]));
      std::printf("  [determinism] two prefill+decode runs from state0: max|diff| = %.3e\n", mx);
    }
    r.prefix_state_restore(0);
    r.forward(prompt, 0, /*last_row_only=*/true);
  }

  const int pos = static_cast<int>(prompt.size());
  const std::vector<int> one = {next_token};
  for (int k : {1, 5, 8, 64, 128}) {
    // Both calls must start from the same post-prefill state: restore before each.
    r.prefix_state_restore(0);
    std::vector<int> cid;
    std::vector<float> cval;
    const bool ok = r.forward_topk_candidates(one, pos, k, &cid, &cval);
    if (!ok) {
      if (k > static_cast<int>(engine::config::get_int("head_topk_max"))) {
        std::printf("SKIP %-24s k=%-3d (k > head_topk_max)\n", model, k);
        continue;
      }
      std::printf("FAIL %s k=%d: device head unavailable\n", model, k);
      ++failures;
      continue;
    }
    r.prefix_state_restore(0);
    row = r.forward(one, pos, /*last_row_only=*/true);
    const std::vector<Pair> ref = host_topk(row, k);
    if (k == 5) {
      // determinism baseline: the same host call twice from the same state
      r.prefix_state_restore(0);
      const std::vector<float> row2 = r.forward(one, pos, /*last_row_only=*/true);
      double mx = 0.0;
      for (size_t i = 0; i < row.size() && i < row2.size(); ++i)
        mx = std::max(mx, std::fabs(static_cast<double>(row[i]) - row2[i]));
      std::printf("  [determinism] host-vs-host max|diff| = %.3e\n", mx);
    }
    if (static_cast<int>(cid.size()) != k || static_cast<int>(cval.size()) != k ||
        static_cast<int>(ref.size()) != k) {
      std::printf("FAIL %s k=%d: size %zu/%zu ref %zu\n", model, k, cid.size(), cval.size(),
                  ref.size());
      ++failures;
      continue;
    }
    // The decode path itself is not bit-deterministic across calls (split-k /
    // atomics sum in a different order: measured host-vs-host max|diff| ~7.6e-4),
    // so the acceptance rule is: identical candidate ids/order and values within
    // that numerical noise (2e-3), which is far below any sampling-relevant gap.
    constexpr double kTol = 2e-3;
    bool same = true;
    double max_diff = 0.0;
    for (int q = 0; q < k; ++q) {
      max_diff = std::max(max_diff, std::fabs(static_cast<double>(cval[static_cast<size_t>(q)]) -
                                              static_cast<double>(ref[static_cast<size_t>(q)].v)));
      if (cid[static_cast<size_t>(q)] != ref[static_cast<size_t>(q)].i) {
        std::printf("FAIL %s k=%d q=%d: dev_id=%d ref_id=%d\n", model, k, q,
                    cid[static_cast<size_t>(q)], ref[static_cast<size_t>(q)].i);
        same = false;
        break;
      }
    }
    if (same && max_diff > kTol) {
      std::printf("FAIL %s k=%d: ids match but max|dv|=%.3e > %.0e\n", model, k, max_diff, kTol);
      same = false;
    }
    if (same)
      std::printf("PASS %-24s k=%-3d max|dv|=%.2e (candidate bytes %d vs full row %d)\n", model,
                  k, max_diff, k * 8, r.vocab() * 4);
    else
      ++failures;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = argc > 1 ? argv[1] : std::getenv("MODEL");
  if (!model) {
    std::fprintf(stderr, "usage: %s <model.gguf> [prompt ids...]\n", argv[0]);
    return 2;
  }
  std::vector<int> prompt;
  for (int i = 2; i < argc; ++i) prompt.push_back(std::atoi(argv[i]));
  if (prompt.empty()) prompt = {760, 2614, 1234};
  try {
    runner::DenseRunner r(model, 256, runner::Container::Gguf);
    std::printf("model %s vocab=%d head_topk_max=%lld\n", model, r.vocab(),
                static_cast<long long>(engine::config::get_int("head_topk_max")));
    check(r, model, prompt, /*next_token=*/271);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
  std::printf("\n%s (%d failures)\n", failures ? "TOPK-CANDIDATES FAILURES" : "TOPK-CANDIDATES GREEN",
              failures);
  return failures ? 1 : 0;
}
