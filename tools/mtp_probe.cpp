// MTP feasibility probe: measures T(n) for the *verify* forward, i.e. how much a
// k+1-token batch forward costs relative to one decode step.  This is the `c` term of
// S = (1+a+a^2)/(c + k*d); the whole MTP bet depends on c being small.
//
// The verify forward is measured with last_row_only=true (1-row head) so the f32
// lm_head materialisation (2 GB on the 35B) never enters the number: that is what the
// real verify path will do once it has a quantized multi-row head.
//
// Usage:
//   mtp_probe --model M.gguf --ids "760" [--ctx 26] [--rows 2,3,4] [--reps 5] [--full]
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

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
      .count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  engine::config::Config& cfg = engine::config::Config::get();
  std::string ids_opt;
  std::string rows_opt = "2,3,4";
  int reps = 5;
  bool want_full = false;
  std::vector<std::string> filtered;
  filtered.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
      return argv[++i];
    };
    if (a == "--ids" || a == "--tokens") ids_opt = need("--ids");
    else if (a == "--rows") rows_opt = need("--rows");
    else if (a == "--reps") reps = std::atoi(need("--reps").c_str());
    else if (a == "--full") want_full = true;
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
    int ctx = 0;
    if (cfg.was_set("ctx")) ctx = static_cast<int>(cfg.get_int("ctx"));
    const int max_seq = ctx > 0 ? ctx : static_cast<int>(ids.size()) + 64;
    runner::DenseRunner r(model, max_seq, detect_container(model));
    const int vocab = r.vocab();

    auto t0 = std::chrono::steady_clock::now();
    auto logits = r.forward(ids, 0, true);
    std::printf("[probe] ctx=%d prefill %zu ids: %.2f ms\n", max_seq, ids.size(), ms_since(t0));
    std::vector<float> last(logits.end() - vocab, logits.end());
    auto argmax = [&](const std::vector<float>& row) {
      int best = 0;
      for (int i = 1; i < vocab; ++i)
        if (row[i] > row[best]) best = i;
      return best;
    };
    int pos = static_cast<int>(ids.size());
    int next = argmax(last);

    auto one = [&](const std::vector<int>& toks) -> double {
      auto tt = std::chrono::steady_clock::now();
      auto o = r.forward(toks, pos, true);
      const double ms = ms_since(tt);
      pos += static_cast<int>(toks.size());
      if (!o.empty()) next = argmax(std::vector<float>(o.end() - vocab, o.end()));
      return ms;
    };
    // warm up (graph capture, kernel caches, CPU thread pool)
    for (int i = 0; i < 5; ++i) one({next});
    std::vector<double> v1;
    for (int i = 0; i < reps; ++i) v1.push_back(one({next}));
    const double t1 = median(v1);
    std::printf("[probe] n=1 decode %.2f ms (%.1f tok/s)\n", t1, 1000.0 / t1);

    std::vector<int> rows;
    {
      size_t p = 0;
      while (p < rows_opt.size()) {
        size_t q = rows_opt.find(',', p);
        if (q == std::string::npos) q = rows_opt.size();
        rows.push_back(std::atoi(rows_opt.substr(p, q - p).c_str()));
        p = q + 1;
      }
    }
    std::printf("[probe] %-4s %-10s %-10s %-12s\n", "n", "ms", "c=T(n)/T(1)", "per-token");
    for (int n : rows) {
      std::vector<int> toks(static_cast<size_t>(n), next);
      one(toks);   // warm the shape (allocations, kernel picks)
      std::vector<double> v;
      for (int i = 0; i < reps; ++i) v.push_back(one(toks));
      const double m = median(v);
      std::printf("[probe] %-4d %-10.2f %-10.3f %.2f ms/token  (E=1+a+..)\n", n, m, m / t1,
                  m / n);
    }
    if (want_full) {
      std::vector<int> toks(2, next);
      auto tt = std::chrono::steady_clock::now();
      auto o = r.forward(toks, pos, false);
      std::printf("[probe] n=2 FULL-logits (f32 head path) %.1f ms rows=%zu\n", ms_since(tt),
                  o.size() / static_cast<size_t>(vocab));
      pos += 2;
    }
    std::printf("[probe] free vram %.0f MiB\n",
                static_cast<double>(cuda::mem_info().free_bytes) / 1048576.0);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
