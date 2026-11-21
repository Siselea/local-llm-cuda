// Host-only micro-costs for the hot-path audit (no GPU, no engine state).
//   - std::getenv  : how expensive is an env lookup in this process's env block
//   - steady_clock::now(): the two clock reads cuda::launch() does per launch
//   - std::string  : the type-name comparisons choose_gemv() does per GEMV
// Build: python -m ziglang c++ -std=c++17 -O2 tools/bench_host.cpp -o build/bench_host.exe
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

template <class F>
double ns_per_call(F&& f, int iters) {
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) f();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

volatile int g_sink = 0;

}  // namespace

int main() {
  std::printf("env vars visible: ");
  int nenv = 0;
  for (char** e = environ; *e; ++e) ++nenv;
  std::printf("%d\n", nenv);

  const int N = 2000000;
  const auto wall0 = std::chrono::steady_clock::now();
  const int N2 = 200000;
  const double ns_getenv2 =
      ns_per_call([] { volatile const char* p = "ENGINE_SOME_DEBUG_FLAG_NAME"; g_sink += std::getenv((const char*)p) != nullptr; }, N2);
  std::printf("getenv(2.1M calls) total wall includes setup; N=%d\n", N);
  const double ns_getenv =
      ns_per_call([] { g_sink += std::getenv("ENGINE_DUMP") != nullptr; }, N);
  const double ns_getenv_miss =
      ns_per_call([] { g_sink += std::getenv("ENGINE_NOT_SET_AT_ALL") != nullptr; }, N);
  const double ns_getenv_last =
      ns_per_call([] { g_sink += std::getenv("PATH") != nullptr; }, N);
  const double ns_clock = ns_per_call([] { g_sink += std::chrono::steady_clock::now().time_since_epoch().count() & 1; }, N);

  const std::string t_q5k = "Q5_K";
  const std::string t_q6k = "Q6_K";
  const double ns_strcmp = ns_per_call(
      [&] {
        // the comparison chain choose_gemv() runs: Q5_1, then known-types, then branch
        bool b = t_q5k == "Q5_1";
        b = b || t_q5k == "Q4_K" || t_q5k == "Q5_K" || t_q5k == "Q6_K" || t_q5k == "Q5_0" ||
            t_q5k == "Q8_0";
        g_sink += b;
      },
      N);
  const double ns_strkey = ns_per_call(
      [&] {
        // gemv_shape_key(): type + " " + to_string(n_out) + "x" + to_string(n_in)
        const std::string k = t_q6k + " " + std::to_string(8192) + "x" + std::to_string(4096);
        g_sink += (int)k.size();
      },
      N);

  std::printf("getenv(long name) %6.1f ns/call\n", ns_getenv2);
  std::printf("getenv(hit)      %7.1f ns/call\n", ns_getenv);
  std::printf("getenv(miss)     %7.1f ns/call\n", ns_getenv_miss);
  std::printf("getenv(PATH hit) %7.1f ns/call\n", ns_getenv_last);
  std::printf("steady_clock::now %6.1f ns/call\n", ns_clock);
  std::printf("choose_gemv strcmp %5.1f ns/call\n", ns_strcmp);
  std::printf("gemv_shape_key() %7.1f ns/call (heap alloc)\n", ns_strkey);
  // per-expert bookkeeping loop shape (2 assigns of an empty range + 3 clock reads)
  const double ns_loop = ns_per_call([] {
    int sink = 0;
    std::vector<int> ti;
    std::vector<float> tw;
    const std::vector<int> all_idx(8);
    const std::vector<float> all_w(8);
    int exp_off[257] = {0};
    for (int e = 0; e < 256; ++e) {
      const int off = exp_off[e], off1 = exp_off[e + 1];
      ti.assign(all_idx.begin() + off, all_idx.begin() + off1);
      tw.assign(all_w.begin() + off, all_w.begin() + off1);
      const auto t0 = std::chrono::steady_clock::now();
      if (ti.empty()) {
        sink += (int)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count() & 1;
        continue;
      }
      sink += (int)std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count() & 1;
    }
    g_sink += sink;
  }, 2000);
  std::printf("per-expert loop (256 it, 3 clocks) %8.1f ns/layer-shape  => 24 layers %.3f ms\n",
              ns_loop, ns_loop * 24 / 1e6);
  // 256-slot softmax + partial_sort (route building)
  const double ns_route = ns_per_call([] {
    float pr[256];
    int idx[256];
    for (int i = 0; i < 256; ++i) idx[i] = i;
    for (int i = 0; i < 256; ++i) pr[i] = 0.5f;
    float mx = 0.f;
    for (int i = 0; i < 256; ++i) mx = std::max(mx, pr[i]);
    double sum = 0;
    for (int i = 0; i < 256; ++i) { pr[i] = std::exp(pr[i] - mx); sum += pr[i]; }
    for (int i = 0; i < 256; ++i) pr[i] = float(pr[i] / sum);
    std::partial_sort(idx, idx + 8, idx + 256, [&](int a, int b) { return pr[a] > pr[b]; });
    g_sink += idx[0];
  }, 20000);
  std::printf("route build (softmax+top8, ne=256)   %8.1f ns/layer      => 24 layers %.3f ms\n",
              ns_route, ns_route * 24 / 1e6);
  std::printf("sink=%d, wall %.2f s (sum of call loops)\n", g_sink,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count());
  return 0;
}
