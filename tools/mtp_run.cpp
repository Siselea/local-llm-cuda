// MTP tooling: draft accuracy (P0) + speculative-decoding A/B (P1).
//
//   mtp_run --model M.gguf --ids "760" --mode acc  [-n steps] [--mtp-k 2] [--verbose]
//   mtp_run --model M.gguf --ids "760" --mode spec [-n tokens] [--mtp-k 2]
//   mtp_run --model M.gguf --ids "760" --mode plain [-n tokens]   (reference output)
//
// `acc`   : plain greedy loop, compares every MTP draft with the target's arg-max.
// `spec`  : speculative greedy generation (prints the generated ids, timing and the
//           acceptance rate).  `plain` prints the ids the plain greedy loop produces;
//           the two `gen:` lines must be identical (lossless speculative decoding).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/config.h"
#include "engine/cuda.h"
#include "engine/runner.h"

namespace {

std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> ids;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    ids.push_back(std::atoi(s.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return ids;
}

runner::Container detect_container(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) return runner::Container::Safetensors;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f) {
    char magic[4] = {0, 0, 0, 0};
    const size_t got = std::fread(magic, 1, 4, f);
    std::fclose(f);
    if (got == 4 && std::memcmp(magic, "GGUF", 4) == 0) return runner::Container::Gguf;
    return runner::Container::Safetensors;
  }
  return path.find(".gguf") != std::string::npos ? runner::Container::Gguf
                                                 : runner::Container::Safetensors;
}

}  // namespace

int main(int argc, char** argv) {
  engine::config::Config& cfg = engine::config::Config::get();
  std::string ids_opt, mode = "acc";
  int n_opt = 16;
  bool verbose = false;
  std::vector<std::string> filtered;
  filtered.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
      return argv[++i];
    };
    if (a == "--ids") ids_opt = need("--ids");
    else if (a == "--mode") mode = need("--mode");
    else if (a == "--n" || a == "--max-new-tokens") n_opt = std::atoi(need("--n").c_str());
    else if (a == "--verbose") verbose = true;
    else filtered.push_back(a);
  }
  std::vector<char*> cargv;
  for (auto& s : filtered) cargv.push_back(const_cast<char*>(s.c_str()));
  std::vector<std::string> positional, unparsed;
  std::string cfg_err;
  if (!cfg.parse_command_line(static_cast<int>(cargv.size()), cargv.data(), 1, &positional,
                              &unparsed, &cfg_err)) {
    std::fprintf(stderr, "config error: %s\n", cfg_err.c_str());
    return 2;
  }
  std::string cache_err;
  cfg.load_kernel_cache_if_exists(&cache_err);
  try {
    std::string model = cfg.get_string("model");
    size_t pi = 0;
    if (model.empty() && pi < positional.size()) model = positional[pi++];
    if (model.empty()) {
      std::fprintf(stderr, "missing model\n");
      return 2;
    }
    if (ids_opt.empty() && pi < positional.size()) ids_opt = positional[pi++];
    if (ids_opt.empty()) ids_opt = "760";
    std::vector<int> ids = parse_ids(ids_opt);
    int ctx = static_cast<int>(ids.size()) + n_opt + 64;
    if (cfg.was_set("ctx")) ctx = static_cast<int>(cfg.get_int("ctx"));
    if (mode != "plain" && !cfg.get_bool("mtp"))
      std::fprintf(stderr, "note: --mode %s needs --mtp (enabling it)\n", mode.c_str());
    runner::DenseRunner r(model, ctx, detect_container(model));
    const int vocab = r.vocab();
    auto argmax = [&](const std::vector<float>& row) {
      int best = 0;
      for (int i = 1; i < vocab; ++i)
        if (row[i] > row[best]) best = i;
      return best;
    };

    if (mode == "acc") {
      if (!r.mtp_available()) {
        std::fprintf(stderr, "MTP not available\n");
        return 1;
      }
      r.mtp_accuracy(ids, n_opt, verbose ? 1 : 0);
      return 0;
    }
    if (mode == "spec") {
      if (!r.mtp_available()) {
        std::fprintf(stderr, "MTP not available\n");
        return 1;
      }
      runner::DenseRunner::SpecStats st;
      std::vector<int> gen = r.spec_generate(ids, n_opt, &st);
      std::printf("gen:");
      for (int t : gen) std::printf(" %d", t);
      std::printf("\n");
      const double alpha = st.drafted ? static_cast<double>(st.accepted) / st.drafted : 0.0;
      std::fprintf(stderr,
                   "[spec] k=%d rounds=%d drafted=%d accepted=%d alpha=%.3f "
                   "prefill %.1f ms total %.1f ms\n",
                   st.k, st.rounds, st.drafted, st.accepted, alpha, st.prefill_ms,
                   st.total_ms);
      std::fprintf(stderr,
                   "[spec] draft %.2f ms/round, verify %.2f ms/round, "
                   "verify_batch %.2f ms\n",
                   st.rounds ? st.draft_ms / st.rounds : 0.0,
                   st.rounds ? st.verify_ms / st.rounds : 0.0,
                   st.verify_fwd ? st.verify_ms / st.verify_fwd : 0.0);
      if (st.decode_tokens > 0)
        std::fprintf(stderr, "[spec] steady decode %.2f ms/token (%.1f tok/s, %d tokens)\n",
                     st.decode_ms / st.decode_tokens,
                     st.decode_tokens * 1000.0 / st.decode_ms, st.decode_tokens);
      return 0;
    }
    // plain greedy reference
    auto logits = r.forward(ids, 0, true);
    std::vector<float> last(logits.end() - vocab, logits.end());
    int pos = static_cast<int>(ids.size());
    int next = argmax(last);
    std::vector<int> gen;
    const auto t0 = std::chrono::steady_clock::now();
    double ms = 0;
    for (int i = 0; i < n_opt; ++i) {
      gen.push_back(next);
      if (i + 1 == n_opt) break;
      const auto s0 = std::chrono::steady_clock::now();
      runner::DenseRunner::TopKResult tk;
      auto out = r.forward_topk({next}, pos++, 5, &tk);
      ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0)
                .count();
      if (!out.empty()) {
        last.assign(out.end() - vocab, out.end());
        next = argmax(last);
      } else {
        next = tk.idx[0];
      }
    }
    const double total = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "[plain] decode %.2f ms/token (%.1f tok/s, %d steps)\n",
                 ms / std::max(1, n_opt - 1), (n_opt - 1) * 1000.0 / ms, n_opt - 1);
    (void)total;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
