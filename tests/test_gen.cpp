// Greedy generation CLI (also used by the llama.cpp oracle harness).
//
// Usage:
//   test_gen <model.gguf> "<comma ids>" <n_generate> [options]
//   test_gen --model <model> --ids "<comma ids>" --n <n> [options]
//
// Any --option from docs/CONFIG.md is accepted (priority: cli > config > env > default),
// including --config <file>, --ctx, --threads, --gpu-layers, --vram-margin, --no-graph,
// --print-config and --kernel-calibrate.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>

#include "engine/config.h"
#include "engine/runner.h"

#include "dequant_proto.h"
#include "fused_gemm_proto.h"

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
  // A file is GGUF if it starts with the GGUF magic; fall back to the extension for
  // unusual filesystems where the magic cannot be read.
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
  // Prototype benches (test-only, inert unless the flag is passed).
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bench-fused-gemm") == 0) {
      fused_proto::run_bench();
      return 0;
    }
    if (std::strcmp(argv[i], "--bench-dequant") == 0) {
      dequant_proto::run_bench();
      return 0;
    }
  }
  engine::config::Config& cfg = engine::config::Config::get();
  // test_gen specific options: --ids "<a,b,c>" and --n <count>; everything else
  // goes through the shared config parser.
  std::string ids_opt;
  int n_opt = -1;
  std::vector<std::string> filtered;
  filtered.push_back(argv[0]);
  try {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto need = [&](const char* what) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
        return argv[++i];
      };
      if (a == "--ids" || a == "--tokens") ids_opt = need("--ids");
      else if (a == "--n" || a == "--max-new-tokens") n_opt = std::atoi(need("--n").c_str());
      else filtered.push_back(a);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
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
  for (const auto& u : unparsed) {
    if (u == "--help" || u == "-h") {
      cfg.print_options(stdout);
      return 0;
    }
    std::fprintf(stderr, "unknown option %s (try --help)\n", u.c_str());
    return 2;
  }
  std::string cache_err;
  cfg.load_kernel_cache_if_exists(&cache_err);
  if (cfg.get_bool("print_config")) {
    cfg.print_effective(stdout);
    return 0;
  }
  // Same real-desktop scheduling hook as the engine (win_priority); the A/B scripts
  // toggle it through ENGINE_WIN_PRIORITY / --win-priority.
  engine::config::apply_win_priority();

  try {
    std::string model = cfg.get_string("model");
    std::string ids_arg = ids_opt;
    int n_gen = n_opt;
    size_t pi = 0;
    if (model.empty()) {
      if (pi < positional.size()) model = positional[pi++];
      else {
        std::fprintf(stderr, "missing model (positional or --model)\n");
        return 2;
      }
    }
    if (pi < positional.size() && ids_arg.empty()) ids_arg = positional[pi++];
    if (pi < positional.size() && n_gen < 0) n_gen = std::atoi(positional[pi++].c_str());
    if (ids_arg.empty()) {
      std::fprintf(stderr, "missing token ids (positional or --ids)\n");
      return 2;
    }
    if (n_gen < 0) n_gen = 4;
    std::vector<int> ids = parse_ids(ids_arg);
    if (ids.empty()) throw std::runtime_error("no prompt ids");
    // --ctx wins; otherwise keep the historical auto size so scripts and
    // before/after benchmarks see identical KV footprints.
    int max_seq = static_cast<int>(ids.size()) + n_gen + 16;
    if (cfg.was_set("ctx")) max_seq = static_cast<int>(cfg.get_int("ctx"));
    if (max_seq < static_cast<int>(ids.size())) max_seq = static_cast<int>(ids.size());
    const auto container = detect_container(model);
    runner::DenseRunner r(model, max_seq, container);

    if (cfg.get_bool("kernel_calibrate")) {
      const int n = r.calibrate_kernels(/*verbose=*/true);
      std::fprintf(stderr, "[calib] %d shapes measured\n", n);
      return n > 0 ? 0 : 1;
    }

    // ---- MTP speculative decoding -------------------------------------------
    // Same greedy contract as the loop below (a draft is only kept where the target
    // model's own arg-max agrees), so `gen:` must be identical to the plain path.
    if (cfg.get_bool("mtp") && r.mtp_available()) {
      runner::DenseRunner::SpecStats st;
      std::vector<int> sgen = r.spec_generate(ids, n_gen, &st);
      std::printf("gen:");
      for (int t : sgen) std::printf(" %d", t);
      std::printf("\n");
      std::fprintf(stderr, "[time] prefill %d tokens: %.1f ms (spec path)\n",
                   static_cast<int>(ids.size()), st.prefill_ms);
      const double alpha =
          st.drafted > 0 ? static_cast<double>(st.accepted) / st.drafted : 0.0;
      std::fprintf(stderr,
                   "[mtp] k=%d rounds=%d drafted=%d accepted=%d alpha=%.3f "
                   "draft %.2f ms/round verify %.2f ms/round\n",
                   st.k, st.rounds, st.drafted, st.accepted, alpha,
                   st.rounds ? st.draft_ms / st.rounds : 0.0,
                   st.rounds ? st.verify_ms / st.rounds : 0.0);
      if (st.decode_tokens > 0)
        std::fprintf(stderr, "[time] decode %.2f ms/token (%.0f tok/s, %d tokens, spec)\n",
                     st.decode_ms / st.decode_tokens,
                     st.decode_tokens * 1000.0 / st.decode_ms, st.decode_tokens);
      else
        std::fprintf(stderr, "[time] decode n/a (too few rounds)\n");
      return 0;
    }

    // warm up lazy initialisation (first forward pays for it). Use two tokens so the
    // batch GEMM (cuBLAS) path is initialised too.
    // optional: feed the prompt one token at a time (exercises the recurrent states)
    if (std::getenv("ENGINE_SPLIT_PREFILL")) {
      for (size_t i = 0; i < ids.size(); ++i) {
        std::vector<int> one{ids[i]};
        auto logits = r.forward(one, static_cast<int>(i), true);
        if (i + 1 == ids.size()) {
          std::vector<int> idx(logits.size());
          for (size_t j = 0; j < idx.size(); ++j) idx[j] = static_cast<int>(j);
          std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                            [&](int a, int b) { return logits[a] > logits[b]; });
          std::printf("split topk: ");
          const float mx = logits[idx[0]];
          for (int k = 0; k < 5; ++k)
            std::printf("%d(%.3f) ", idx[k], static_cast<double>(logits[idx[k]] - mx));
          std::printf("\n");
        }
      }
      if (std::getenv("ENGINE_SPLIT_ONLY")) return 0;
    }
    if (!std::getenv("ENGINE_SKIP_WARMUP"))
      r.forward({ids[0], ids.size() > 1 ? ids[1] : ids[0]}, 0, /*last_row_only=*/true);
    const auto t_pre0 = std::chrono::steady_clock::now();
    auto logits = r.forward(ids, 0, /*last_row_only=*/true);
    const double prefill_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_pre0).count();
    std::fprintf(stderr, "[time] prefill %d tokens: %.1f ms (%.0f tok/s)\n",
                 static_cast<int>(ids.size()), prefill_ms,
                 ids.size() * 1000.0 / prefill_ms);
    const int vocab = r.vocab();   // last_row_only returns a single row
    auto argmax = [&](const std::vector<float>& row) {
      int best = 0;
      float bv = row[0];
      for (int i = 1; i < vocab; ++i)
        if (row[i] > bv) {
          bv = row[i];
          best = i;
        }
      return best;
    };
    std::vector<float> last(logits.end() - vocab, logits.end());
    if (const char* tgt = std::getenv("ENGINE_TARGET")) {
      const int target = std::atoi(tgt);
      std::vector<int> order(vocab);
      for (int i = 0; i < vocab; ++i) order[i] = i;
      std::partial_sort(order.begin(), order.begin() + 200, order.end(),
                        [&](int a, int b) { return last[a] > last[b]; });
      int rank = -1;
      for (int k = 0; k < 200; ++k)
        if (order[k] == target) {
          rank = k;
          break;
        }
      double sum = 0, mx = last[order[0]];
      for (int i = 0; i < vocab; ++i) sum += std::exp(double(last[i]) - mx);
      const double lp = std::log(std::exp(double(last[target]) - mx) / sum);
      std::printf("target %d: rank=%d logprob=%.3f (top1=%d)\n", target, rank, lp, order[0]);
    }
    if (std::getenv("ENGINE_TOPK")) {
      std::vector<int> order(vocab);
      for (int i = 0; i < vocab; ++i) order[i] = i;
      std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                        [&](int a, int b) { return last[a] > last[b]; });
      double mx = last[order[0]];
      double sum = 0;
      for (int i = 0; i < vocab; ++i) sum += std::exp(double(last[i]) - mx);
      std::printf("topk:");
      for (int k = 0; k < 5; ++k) {
        const double lp = std::log(std::exp(double(last[order[k]]) - mx) / sum);
        std::printf(" %d(%.3f)", order[k], lp);
      }
      std::printf("\n");
    }
    std::vector<int> gen;
    int pos = static_cast<int>(ids.size());
    // Diagnostic: skip the first N decode steps from the timing average (the first
    // steps after prefill pay for lazy per-layer CUDA-graph capture).
    const int dec_skip = std::getenv("ENGINE_DEC_SKIP") ? std::atoi(std::getenv("ENGINE_DEC_SKIP")) : 0;
    double dec_ms = 0.0;
    int dec_cnt = 0;
    const auto t_dec0 = std::chrono::steady_clock::now();
    int next = argmax(last);
    for (int k = 0; k < n_gen; ++k) {
      gen.push_back(next);
      if (k + 1 == n_gen) break;
      const auto step0 = std::chrono::steady_clock::now();
      // Decode head: ask for the top-k only.  The engine reduces the logits row on the
      // device and copies back k pairs; when the fast path is unavailable it returns
      // the full row and we fall back to the same host argmax as before.
      runner::DenseRunner::TopKResult tk;
      auto out = r.forward_topk({next}, pos, 5, &tk);
      ++pos;
      if (!out.empty()) {
        last.assign(out.end() - vocab, out.end());
        next = argmax(last);
      } else {
        next = tk.idx[0];
      }
      if (k >= dec_skip) {
        dec_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - step0)
                      .count();
        ++dec_cnt;
      }
    }
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    const auto t_dec1 = std::chrono::steady_clock::now();
    (void)t_dec1;
    if (dec_cnt <= 0) {
      dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
      dec_cnt = 1;
    }
    if (n_gen > 1)
      std::fprintf(stderr, "[time] decode %.2f ms/token (%.0f tok/s, %d steps%s)\n",
                   dec_ms / dec_cnt, dec_cnt * 1000.0 / dec_ms, dec_cnt,
                   dec_skip ? " timed" : "");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
