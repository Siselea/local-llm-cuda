// Regression: bounded tokenizer work for network-facing callers.
//
// Before v0.1.8 a long run without separators (e.g. 1 MB of CJK) made the BPE merge
// loop superlinear: 20k chars took 92 s and a single /tokenize request could block
// the single-threaded server for hours.  The merge is now O(n log n) and serve passes
// a character + wall-clock budget; this test pins both properties.
//
// Usage: test_tokenizer_budget <model.gguf>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "engine/tokenizer.h"

namespace {
double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
      .count();
}
int failures = 0;
void check(bool ok, const char* what, const std::string& detail = "") {
  std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", what, detail.empty() ? "" : ": ",
              detail.c_str());
  if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
    return 2;
  }
  const engine::tokenizer::Tokenizer tk = engine::tokenizer::Tokenizer::load(argv[1]);
  std::string cjk;
  for (int i = 0; i < 20000; ++i) cjk += "\xE4\xB8\xAD";  // U+4E2D x20000

  {
    const auto t0 = std::chrono::steady_clock::now();
    const auto ids = tk.encode_bounded(cjk, false, true, 200000, 5000);
    const double ms = ms_since(t0);
    check(ids.size() == 20000, "20k CJK encodes to 20k ids", std::to_string(ids.size()));
    check(ms < 500.0, "20k CJK encode < 500 ms", std::to_string(ms) + " ms");
  }
  {
    std::string one_mb;
    one_mb.reserve(1024 * 1024);
    for (int i = 0; i < 1024 * 1024 / 3; ++i) one_mb += "\xE4\xB8\xAD";
    const auto t0 = std::chrono::steady_clock::now();
    bool rejected = false;
    std::string msg;
    try {
      tk.encode_bounded(one_mb, false, true, 200000, 1500);
    } catch (const std::exception& e) {
      rejected = true;
      msg = e.what();
    }
    const double ms = ms_since(t0);
    check(rejected, "1 MB CJK rejected by the character budget");
    check(ms < 100.0, "1 MB rejection < 100 ms", std::to_string(ms) + " ms");
    check(msg.find("input too large") != std::string::npos, "clear budget message", msg);
  }
  {
    // unlimited encode() keeps working for CLI/tests
    const auto ids = tk.encode("hello", false, true);
    check(!ids.empty(), "unlimited encode still works");
  }
  std::printf(failures ? "TOKENIZER BUDGET FAILURES\n" : "TOKENIZER BUDGET GREEN\n");
  return failures ? 1 : 0;
}
