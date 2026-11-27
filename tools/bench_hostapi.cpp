// Host API cost microbenchmark for the engine decode hot path.
//
// WHY THIS EXISTS
// ---------------
// The 35B decode path is dominated by the host GEMV, then by the per-layer GPU
// round trip.  Everything else that runs per token is "small" -- but the decode path
// executes it 24-40 times per token, so a 1-2 us operation is a whole millisecond.
// The std::getenv audit (2.13 us/call x ~1300 calls = ~2.8 ms/token) found ~2.1 ms of
// real, removable cost this way, so the other candidates are measured the same way:
// one microbenchmark per operation, then multiply by the per-token call count.
//
// This tool is PURE STL (no CUDA, no engine state) so it can run while the GPU is busy.
//
//   build/bench_hostapi.exe
//
// Output: one line per operation, ns/op on this host.  tools/hotpath_api_audit.py
// combines these numbers with the call-site counts.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Volatile sink: keeps the measured work observable so the optimiser cannot delete it.
volatile long long g_sink = 0;
volatile long long g_iter = 0;  // blocks loop-invariant hoisting
volatile void* g_sink_ptr = nullptr;  // forces allocations/side effects to happen

using clk = std::chrono::steady_clock;

template <class F>
double bench(const char* label, long long iters, F&& f) {
  for (int w = 0; w < 3; ++w) f();
  const auto t0 = clk::now();
  for (long long i = 0; i < iters; ++i) {
    g_iter = i;   // volatile: the lambda body cannot be hoisted out of the loop
    f();
  }
  const auto t1 = clk::now();
  const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                    static_cast<double>(iters);
  std::printf("%-52s %9.2f ns/op  (%lld iters)\n", label, ns, iters);
  return ns;
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
  const long long S = quick ? 20000 : 200000;     // ~2e5 "expensive" ops
  const long long M = quick ? 200000 : 2000000;   // ~2e6 medium ops
  const long long L = quick ? 2000000 : 20000000; // ~2e7 cheap ops

  std::printf("== bench_hostapi (pure STL, no CUDA) ==\n");

  // ---- reference points -----------------------------------------------------
  bench("steady_clock::now()", L, [] {
    g_sink += static_cast<long long>(clk::now().time_since_epoch().count() & 1);
  });
  bench("std::getenv(unset, 16 chars)", S, [] { g_sink += std::getenv("ENGINE_NOT_SET_X") != nullptr; });
  bench("std::exp(double) x256", S, [] {
    double acc = 0;
    for (int i = 0; i < 256; ++i) acc += std::exp(static_cast<double>(i) * 0.01);
    g_sink += static_cast<long long>(acc);
  });
  bench("sigmoid(x) x256  (routing loop)", S, [] {
    double acc = 0;
    for (int i = 0; i < 256; ++i) acc += 1.0 / (1.0 + std::exp(-static_cast<double>(i) * 0.01));
    g_sink += static_cast<long long>(acc);
  });

  // ---- std::string ----------------------------------------------------------
  const std::string n16 = "ENGINE_DUMP_X16";   // SSO on MSVC/libstdc++ (<=15/22 chars)
  const std::string n40 = "ENGINE_SOME_VERY_LONG_DIAGNOSTIC_FLAG_NAME_40CH";
  bench("std::string from literal (15 chars, SSO)", M,
        [] { std::string s("ENGINE_DUMP_15"); g_sink += static_cast<long long>(s.size()); });
  bench("std::string from literal (40 chars, heap)", M,
        [] { std::string s("ENGINE_SOME_VERY_LONG_DIAGNOSTIC_FLAG_NAME_40CH");
             g_sink += static_cast<long long>(s.size()); });
  bench("std::string copy (SSO)", M, [&] {
    std::string c(n16);
    g_sink += static_cast<long long>(c.size());
  });
  bench("std::string operator+ (small)", M, [&] {
    std::string c = n16 + n16;
    g_sink += static_cast<long long>(c.size());
  });
  bench("std::to_string(int)", M, [] {
    std::string s = std::to_string(12345);
    g_sink += static_cast<long long>(s.size());
  });
  bench("std::string::c_str()", L, [&] { g_sink += n16.c_str()[0]; });
  bench("std::string == literal (SSO)", L, [&] { g_sink += n16 == "ENGINE_DUMP_15"; });
  bench("std::string::substr(2,4)", M, [&] {
    std::string s = n16.substr(2, 4);
    g_sink += static_cast<long long>(s.size());
  });
  bench("std::string(40) copy (heap alloc)", M, [&] {
    std::string c(n40);
    g_sink += static_cast<long long>(c.size());
  });

  // ---- std::vector ----------------------------------------------------------
  std::vector<float> vf(2048, 0.f);
  std::vector<int> vi(257, 0);
  bench("vector<float>::resize(same size 2048)", M, [&] {
    vf.resize(2048);
    g_sink += static_cast<long long>(vf.size());
  });
  bench("vector<int>::resize(same size 257)", M, [&] {
    vi.resize(257);
    g_sink += static_cast<long long>(vi.size());
  });
  bench("vector<int>::assign(8 elems)", M, [&] {
    std::vector<int> t;
    t.assign(vi.begin(), vi.begin() + 8);
    g_sink += static_cast<long long>(t.size());
  });
  bench("vector<float> construct(2048) + destroy", M, [] {
    std::vector<float> t(2048, 0.f);
    g_sink += static_cast<long long>(t.size());
  });
  bench("vector<array-ish> fresh reserve(8)+8 push_back", M, [] {
    std::vector<std::pair<int, float>> t;
    t.reserve(8);
    for (int i = 0; i < 8; ++i) t.push_back({i, 1.f});
    g_sink += static_cast<long long>(t.size());
  });
  bench("vector<int> construct(257) + destroy", M, [] {
    std::vector<int> t(257, 0);
    g_sink += static_cast<long long>(t.size());
  });

  // ---- maps -----------------------------------------------------------------
  std::map<std::string, double> smap;
  for (int i = 0; i < 200; ++i) smap["key_" + std::to_string(i)] = i;
  std::unordered_map<std::string, double> umap(smap.begin(), smap.end());
  std::map<void*, int> pmap;
  int dummy[64];
  for (int i = 0; i < 64; ++i) pmap[&dummy[i]] = i;
  bench("std::map<string,double>::find (200 entries, hit)", L, [&] {
    auto it = smap.find("key_100");
    g_sink += it != smap.end();
  });
  bench("std::map<string,double>::find (miss)", L, [&] {
    auto it = smap.find("key_zzz");
    g_sink += it != smap.end();
  });
  bench("std::unordered_map<string,double>::find (miss)", L, [&] {
    auto it = umap.find("key_zzz");
    g_sink += it != umap.end();
  });
  bench("std::map<void*,int>::find (64 entries)", L, [&] {
    g_sink += pmap.find(&dummy[7]) != pmap.end();
  });

  // ---- std::function --------------------------------------------------------
  int a = 1, b = 2;
  bench("std::function construct from lambda(2 refs) + call", M, [&] {
    std::function<int(int)> f = [&](int x) { return x + a + b; };
    g_sink += f(1);
  });
  std::function<int(int)> cached = [&](int x) { return x + a + b; };
  bench("std::function call only", L, [&] { g_sink += cached(1); });
  bench("std::function construct from lambda(6 refs)", M, [&] {
    int c = 3, d = 4, e = 5, f2 = 6;
    std::function<int(int)> f = [&](int x) { return x + a + b + c + d + e + f2; };
    g_sink += f(1);
  });

  // ---- printf family --------------------------------------------------------
  char buf[128];
  bench("snprintf(buf,128,\"%d %.3f\", ...) ", M, [&] {
    const int n = std::snprintf(buf, sizeof(buf), "%d %.3f", a, 1.5);
    g_sink += buf[n - 1];                 // consume the result: not elidable
  });
  FILE* devnull = std::fopen("build/bench_hostapi_printf.tmp", "w");
  if (devnull) {
    bench("fprintf(FILE*, small formatted line)", M,
          [&] { g_sink += std::fprintf(devnull, "x=%d y=%.2f\n", a, 1.5); });
    std::fclose(devnull);
    std::remove("build/bench_hostapi_printf.tmp");
  }

  // ---- synchronisation ------------------------------------------------------
  std::mutex mu;
  bench("std::mutex lock+unlock (uncontended)", L, [&] {
    mu.lock();
    g_sink += 1;
    mu.unlock();
  });
  std::condition_variable cv;
  bench("condition_variable::notify_all (no waiters)", M, [&] { cv.notify_all(); });

  // ---- TLS ------------------------------------------------------------------
  bench("thread_local int read+write", L, [&] {
    thread_local int tls = 0;
    ++tls;
    g_sink += tls;
  });
  bench("thread_local std::string (non-trivial dtor) access", L, [&] {
    thread_local std::string tls;
    g_sink += static_cast<long long>(tls.size());
  });

  // ---- filesystem -----------------------------------------------------------
  bench("std::filesystem::exists(existing file)", S, [] {
    g_sink += static_cast<long long>(std::filesystem::exists("build/test_gen.exe"));
  });
  bench("std::filesystem::exists(missing path)", S, [] {
    g_sink += static_cast<long long>(std::filesystem::exists("build/no_such_file_xyz"));
  });

  // ---- memory / refcount ----------------------------------------------------
  std::shared_ptr<int> sp = std::make_shared<int>(7);
  bench("shared_ptr copy + destroy (atomic refcount)", L, [&] {
    std::shared_ptr<int> c = sp;
    g_sink += *c;
  });
  bench("new/delete small object", M, [] {
    int* p = new int(5);
    g_sink += *p;
    delete p;
  });
  bench("malloc/free (size from volatile)", M, [] {
    void* p = std::malloc(static_cast<size_t>(g_iter & 63) + 8);
    g_sink_ptr = p;                       // volatile store: cannot be elided
    std::free(p);
  });
  bench("vector<int> fresh (1 alloc) + destroy", M, [] {
    std::vector<int> t(8, 1);
    g_sink_ptr = t.data();
  });
  bench("make_unique<int>", M, [] {
    auto p = std::make_unique<int>(5);
    g_sink += *p;
  });

  // ---- algorithms used by the host router -----------------------------------
  std::vector<int> idx(256);
  std::vector<double> pr(256);
  for (int i = 0; i < 256; ++i) {
    idx[i] = i;
    pr[i] = static_cast<double>(i) * 0.003;
  }
  bench("std::partial_sort(256, top 8) (host router)", S, [&] {
    for (int i = 0; i < 256; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(),
                      [&](int x, int y) { return pr[x] > pr[y]; });
    g_sink += idx[0];
  });
  bench("host router pass: 256 sigmoid + partial_sort", S, [&] {
    for (int i = 0; i < 256; ++i) pr[i] = 1.0 / (1.0 + std::exp(-pr[i]));
    for (int i = 0; i < 256; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(),
                      [&](int x, int y) { return pr[x] > pr[y]; });
    g_sink += idx[0];
  });

  // ---- json / string formatting used by the server per token? --------------
  // (serve formats one SSE frame per token; measured for completeness)
  const char* piece = "hello";
  bench("sse frame: 3x snprintf (serve, per token)", M, [&] {
    char fr[256];
    int n = std::snprintf(fr, sizeof(fr), "{\"delta\":{\"content\":\"");
    n += std::snprintf(fr + n, sizeof(fr) - n, "%s%d", piece, static_cast<int>(g_iter & 7));
    n += std::snprintf(fr + n, sizeof(fr) - n, "\"}}");
    g_sink += fr[n - 1];
  });

  // ---- serve non-greedy sampling path (per generated token) ------------------
  // serve requests with temperature > 0 bypass the device top-k head, take the FULL
  // 248320-wide logits row and run Sampler::sample -> the work below, EVERY token.
  const int V = 248320;
  std::vector<float> logits(V, 0.1f);
  std::vector<float> row(V, 0.2f);
  std::vector<int> cand;
  bench("memcpy 1MB (float[248320] assign)", M / 20, [&] {
    row.assign(logits.begin(), logits.end());
    g_sink += static_cast<long long>(row.size());
  });
  bench("sampler: fill cand_ 0..V-1 (248320 ints)", S / 20, [&] {
    cand.clear();
    cand.resize(V);
    for (int i = 0; i < V; ++i) cand[i] = i;
    g_sink += cand[17];
  });
  bench("sampler: partial_sort V=248320 top_k=20", S / 50, [&] {
    std::partial_sort(cand.begin(), cand.begin() + 20, cand.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    g_sink += cand[0];
  });
  bench("sampler: nth_element V=248320 top_k=20 (alternative)", S / 50, [&] {
    std::nth_element(cand.begin(), cand.begin() + 20, cand.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    g_sink += cand[0];
  });
  bench("argmax over V=248320 (greedy full-row path)", S / 20, [&] {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < V; ++i)
      if (logits[i] > bv) { bv = logits[i]; best = i; }
    g_sink += best;
  });
  bench("FULL sampler step (fill + partial_sort V=248320 k=20)", S / 50, [&] {
    cand.clear();
    cand.resize(V);
    for (int i = 0; i < V; ++i) cand[i] = i;
    std::partial_sort(cand.begin(), cand.begin() + 20, cand.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    g_sink += cand[0];
  });

  // ---- serve path with top_k=0 (top_p-only sampling, a common client setting) -------
  // Sampler::sample then keeps ALL 248320 candidates: probs_of() runs one exp() per
  // candidate and top_p ranks them with a full std::sort -> this is the worst case.
  {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(V);
    bench("sort 248320 (float,int) pairs (top_p over full vocab)", S / 200, [&] {
      ranked.clear();
      for (int i = 0; i < V; ++i) ranked.emplace_back(logits[i], i);
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      g_sink += ranked[0].second;
    });
    bench("exp over full vocab (248320)", S / 200, [&] {
      double acc = 0;
      for (int i = 0; i < V; ++i) acc += std::exp(static_cast<double>(logits[i]));
      g_sink += static_cast<long long>(acc);
    });
  }

  // ---- bounded-candidate sampling (server-mode's new sample_bounded path) --------
  // select_candidates(): resize/fill a 248320-entry index array, then nth_element to
  // keep n candidates.  Measured for the two shipped caps (top_k>0 -> max(k,64);
  // top_k==0 -> 1024) plus a single-pass top-N alternative.
  std::vector<int> ridx(V);
  bench("bounded sel: fill 248320 + nth_element(n=64)   [top_k>0]", S / 100, [&] {
    for (int i = 0; i < V; ++i) ridx[static_cast<size_t>(i)] = i;
    std::nth_element(ridx.begin(), ridx.begin() + 64, ridx.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    g_sink += ridx[0];
  });
  bench("bounded sel: fill 248320 + nth_element(n=1024) [top_k=0]", S / 100, [&] {
    for (int i = 0; i < V; ++i) ridx[static_cast<size_t>(i)] = i;
    std::nth_element(ridx.begin(), ridx.begin() + 1024, ridx.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    g_sink += ridx[0];
  });
  bench("single-pass top-64 scan over 248320 (alternative)", S / 20, [&] {
    std::vector<std::pair<float, int>> best;
    best.reserve(64);
    float floor_v = -1e30f;
    for (int i = 0; i < V; ++i) {
      const float v = logits[i];
      if (v <= floor_v) continue;
      size_t pos = best.size();
      while (pos > 0 && best[pos - 1].first < v) --pos;
      best.insert(best.begin() + static_cast<long>(pos), {v, i});
      if (best.size() > 64) { best.pop_back(); floor_v = best.back().first; }
    }
    g_sink += best[0].second;
  });

  std::printf("== done (sink=%lld) ==\n", static_cast<long long>(g_sink));
  return 0;
}
