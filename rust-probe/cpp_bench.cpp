// C++ side of the Rust-vs-C++ CPU kernel comparison (rust-probe/).
//
// Key design point: the Rust kernels are linked into THIS binary as a staticlib
// (see rust_kernels.rs), so both implementations run in the same process, on the
// same pinned thread team, over the same rotating 24-copy working set.  Cross-process
// noise (this VM drifts 20-30% when other agents are busy) cannot skew the comparison;
// the only variable left is the kernel code.
//
// The C++ kernels are the production ones (this TU includes src/cpu_matvec.cpp).
// Threading = 12 threads, SMT-pair affinity read from the production topology helper
// (cpu::smt_cpu_pairs()) unless RP_PAIRS is given, flat (act,row) partitioning, spin
// barrier -- identical to the Rust side's requirements.
//
// modes:
//   verify            run one layer pass with each implementation, compare outputs
//   ab <rounds>       interleaved A/B of the full CPU layer (16.3 MB/pass)
//   abphasea <rounds> interleaved A/B of phase A only (9.44 MB/pass)
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../src/cpu_matvec.cpp"

#include <windows.h>
#include <intrin.h>

extern "C" {
float rust_dot_q4k(const unsigned char* row, const float* x, size_t n_in);
float rust_dot_q4k_v2(const unsigned char* row, const float* x, size_t n_in);
float rust_dot_q4k_v3(const unsigned char* row, const float* x, size_t n_in);
float rust_dot_q6k(const unsigned char* row, const float* x, size_t n_in);
}

namespace {

constexpr int kHidden = 2048, kInter = 512, kActs = 8;
constexpr std::size_t kGateBytes = static_cast<std::size_t>(kActs) * kInter * (kHidden / 256) * 144;
constexpr std::size_t kDownBytes = static_cast<std::size_t>(kActs) * kHidden * (kInter / 256) * 210;
constexpr std::size_t kCopyBytes = 2 * kGateBytes + kDownBytes;  // 16,318,464
constexpr int kCopies = 24;

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<unsigned char> load_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> b(static_cast<std::size_t>(n));
  if (std::fread(b.data(), 1, b.size(), f) != b.size()) std::exit(1);
  std::fclose(f);
  return b;
}

// ---- identical spin barrier in both harnesses -------------------------------
struct SpinBarrier {
  std::atomic<int> count{0};
  std::atomic<unsigned> gen{0};
  int n = 1;
  void wait() {
    const unsigned g = gen.load(std::memory_order_acquire);
    if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
      count.store(0, std::memory_order_relaxed);
      gen.fetch_add(1, std::memory_order_release);
    } else {
      while (gen.load(std::memory_order_acquire) == g)
        for (int i = 0; i < 8; ++i) _mm_pause();
    }
  }
};

struct Shared {
  const unsigned char* base = nullptr;
  const float* x = nullptr;
  const float* aw = nullptr;
  float* abuf = nullptr;
  float* dbuf = nullptr;
  float* xout = nullptr;
  int nthreads = 12;
  int use_rust = 0;   // 0 = C++ kernels, 1 = Rust kernels
  int phase_a_only = 0;
  SpinBarrier bar;
  std::atomic<long long> sink{0};
};

inline void chunk(long long total, int tid, int nthr, long long* b, long long* e) {
  const long long c = (total + nthr - 1) / nthr;
  *b = static_cast<long long>(tid) * c;
  *e = std::min<long long>(total, *b + c);
}

inline float dot_q4k(int use_rust, const unsigned char* r, const float* x, int n) {
  if (use_rust) return rust_dot_q4k(r, x, static_cast<size_t>(n));
  float y = 0.f;
  cpu::avx2_dot_q4k(r, x, n, &y);
  return y;
}
inline float dot_q6k(int use_rust, const unsigned char* r, const float* x, int n) {
  if (use_rust) return rust_dot_q6k(r, x, static_cast<size_t>(n));
  float y = 0.f;
  cpu::avx2_dot_q6k(r, x, n, &y);
  return y;
}

void run_pass(Shared* sh, int tid, int copy_idx) {
  const unsigned char* copy = sh->base + static_cast<std::size_t>(copy_idx % kCopies) * kCopyBytes;
  const unsigned char* gate = copy;
  const unsigned char* up = copy + kGateBytes;
  const unsigned char* down = copy + 2 * kGateBytes;
  {
    long long b, e;
    chunk(static_cast<long long>(kActs) * kInter, tid, sh->nthreads, &b, &e);
    long long sink = 0;
    for (long long idx = b; idx < e; ++idx) {
      const int act = static_cast<int>(idx / kInter);
      const int r = static_cast<int>(idx - static_cast<long long>(act) * kInter);
      const unsigned char* gr = gate + static_cast<std::size_t>(act) * kInter * (kHidden / 256) * 144 +
                                static_cast<std::size_t>(r) * (kHidden / 256) * 144;
      const unsigned char* ur = up + static_cast<std::size_t>(act) * kInter * (kHidden / 256) * 144 +
                                static_cast<std::size_t>(r) * (kHidden / 256) * 144;
      const float g = dot_q4k(sh->use_rust, gr, sh->x, kHidden);
      const float u = dot_q4k(sh->use_rust, ur, sh->x, kHidden);
      sh->abuf[static_cast<std::size_t>(act) * kInter + r] = (g / (1.f + std::exp(-g))) * u;
      sink += static_cast<long long>(g + u);
    }
    sh->sink.fetch_add(sink, std::memory_order_relaxed);
  }
  sh->bar.wait();
  if (!sh->phase_a_only) {
    {
      long long b, e;
      chunk(static_cast<long long>(kActs) * kHidden, tid, sh->nthreads, &b, &e);
      long long sink = 0;
      for (long long idx = b; idx < e; ++idx) {
        const int act = static_cast<int>(idx / kHidden);
        const int r = static_cast<int>(idx - static_cast<long long>(act) * kHidden);
        const unsigned char* dr =
            down + static_cast<std::size_t>(act) * kHidden * (kInter / 256) * 210 +
            static_cast<std::size_t>(r) * (kInter / 256) * 210;
        const float d =
            dot_q6k(sh->use_rust, dr, sh->abuf + static_cast<std::size_t>(act) * kInter, kInter);
        sh->dbuf[static_cast<std::size_t>(act) * kHidden + r] = d;
        sink += static_cast<long long>(d);
      }
      sh->sink.fetch_add(sink, std::memory_order_relaxed);
    }
    sh->bar.wait();
    {
      long long b, e;
      chunk(kHidden, tid, sh->nthreads, &b, &e);
      for (long long j = b; j < e; ++j)
        for (int act = 0; act < kActs; ++act)
          sh->xout[static_cast<std::size_t>(j)] +=
              sh->aw[act] * sh->dbuf[static_cast<std::size_t>(act) * kHidden + j];
    }
    sh->bar.wait();
  } else {
    sh->bar.wait();
  }
}

std::vector<std::pair<int, int>> g_pairs;
void pin_self(int cpu) {
  if (cpu >= 0) SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
}
int thread_cpu(int tid) {
  if (g_pairs.empty()) return -1;
  const std::size_t k = static_cast<std::size_t>(tid / 2);
  if (k >= g_pairs.size()) return -1;
  return (tid % 2 == 0) ? g_pairs[k].first : g_pairs[k].second;
}

void run_once(Shared* sh, int copy_idx) {
  std::vector<std::thread> th;
  for (int t = 1; t < sh->nthreads; ++t)
    th.emplace_back([&, t] {
      pin_self(thread_cpu(t));
      run_pass(sh, t, copy_idx);
    });
  pin_self(thread_cpu(0));
  run_pass(sh, 0, copy_idx);
  for (auto& t : th) t.join();
}

// run `iters` passes on the persistent-once-per-call thread team and return ms/pass
double time_run(Shared* sh, int iters, int use_rust, int* copy_start) {
  sh->use_rust = use_rust;
  sh->sink.store(0);
  std::vector<std::thread> th;
  const double t0 = now_ms();
  int next_copy = *copy_start;
  for (int t = 1; t < sh->nthreads; ++t)
    th.emplace_back([&, t] {
      pin_self(thread_cpu(t));
      for (int i = 0; i < iters; ++i) run_pass(sh, t, (next_copy + i) % kCopies);
    });
  pin_self(thread_cpu(0));
  for (int i = 0; i < iters; ++i) run_pass(sh, 0, (next_copy + i) % kCopies);
  for (auto& t : th) t.join();
  const double t1 = now_ms();
  *copy_start = (next_copy + iters) % kCopies;
  return (iters > 0) ? (t1 - t0) / iters : (t1 - t0);
}


// ---- single-thread kernel comparison (contention-robust) ---------------------
// One thread pinned to one core; C++ and Rust kernels alternate on the same real
// rows.  Under a busy machine a single sample can be preempted, so we report
// min / p25 / median of many reps; the min is the clean-core number.
static void stats3(const std::vector<double>& v, double* mn, double* p25, double* md) {
  if (v.empty()) { *mn = *p25 = *md = 0; return; }
  std::vector<double> s = v;
  std::sort(s.begin(), s.end());
  *mn = s.front();
  *p25 = s[s.size() / 4];
  *md = s[s.size() / 2];
}

int mode_kernel(const std::vector<unsigned char>& gate, const std::vector<float>& x,
                const std::vector<unsigned char>& down, const std::vector<float>& abuf, int reps) {
  pin_self(0);
  const int q4rows = kInter;    // 512 rows x 1152 B
  const int q6rows = kHidden;   // 2048 rows x 420 B
  std::vector<double> cpp4, rust4, cpp6, rust6, cpp4c, rust4c, v2_4, v2_4c, v3_4, v3_4c;
  volatile float sink = 0;
  auto time_q4v3 = [&]() -> double {
    float acc = 0;
    const unsigned long long c0 = __rdtsc();
    const double t0 = now_ms();
    for (int i = 0; i < q4rows; ++i)
      acc += rust_dot_q4k_v3(gate.data() + static_cast<std::size_t>(i) * 1152, x.data(), kHidden);
    const double t1 = now_ms();
    const unsigned long long c1 = __rdtsc();
    sink += acc;
    v3_4c.push_back(double(c1 - c0) / q4rows);
    return (t1 - t0) * 1e6 / q4rows;
  };
  auto time_q4v2 = [&]() -> double {
    float acc = 0;
    const unsigned long long c0 = __rdtsc();
    const double t0 = now_ms();
    for (int i = 0; i < q4rows; ++i)
      acc += rust_dot_q4k_v2(gate.data() + static_cast<std::size_t>(i) * 1152, x.data(), kHidden);
    const double t1 = now_ms();
    const unsigned long long c1 = __rdtsc();
    sink += acc;
    v2_4c.push_back(double(c1 - c0) / q4rows);
    return (t1 - t0) * 1e6 / q4rows;
  };
  auto time_q4 = [&](int use_rust) -> double {
    float acc = 0;
    const unsigned long long c0 = __rdtsc();
    const double t0 = now_ms();
    for (int i = 0; i < q4rows; ++i)
      acc += dot_q4k(use_rust, gate.data() + static_cast<std::size_t>(i) * 1152, x.data(), kHidden);
    const double t1 = now_ms();
    const unsigned long long c1 = __rdtsc();
    sink += acc;
    if (use_rust) rust4c.push_back(double(c1 - c0) / q4rows);
    else cpp4c.push_back(double(c1 - c0) / q4rows);
    return (t1 - t0) * 1e6 / q4rows;  // ns/row
  };
  auto time_q6 = [&](int use_rust) -> double {
    float acc = 0;
    const double t0 = now_ms();
    for (int i = 0; i < q6rows; ++i)
      acc += dot_q6k(use_rust, down.data() + static_cast<std::size_t>(i) * 420, abuf.data(),
                     kInter);
    const double t1 = now_ms();
    sink += acc;
    return (t1 - t0) * 1e6 / q6rows;
  };
  for (int r = 0; r < 6; ++r) {  // warm-up
    time_q4(0); time_q4(1); time_q6(0); time_q6(1);
  }
  for (int r = 0; r < reps; ++r) {
    const bool cpp_first = (r % 2 == 0);
    const int v0 = cpp_first ? 0 : 1, v1 = 1 - v0;
    const double a = time_q4(v0), b = time_q4(v1);
    (v0 == 0 ? cpp4 : rust4).push_back(a);
    (v1 == 0 ? cpp4 : rust4).push_back(b);
    v2_4.push_back(time_q4v2());
    v3_4.push_back(time_q4v3());
    const double c = time_q6(v0), d = time_q6(v1);
    (v0 == 0 ? cpp6 : rust6).push_back(c);
    (v1 == 0 ? cpp6 : rust6).push_back(d);
  }
  double a, b, c, d, e, f;
  stats3(cpp4, &a, &b, &c);
  stats3(rust4, &d, &e, &f);
  std::printf("[kern] Q4_K row (1152 B, 512 rows/expert) ns/row:\n");
  std::printf("[kern]   C++  min %8.1f  p25 %8.1f  med %8.1f  -> %.2f GB/s/core (min)\n", a, b, c,
              1152.0 / a);
  std::printf("[kern]   Rust min %8.1f  p25 %8.1f  med %8.1f  -> %.2f GB/s/core (min)\n", d, e, f,
              1152.0 / d);
  std::printf("[kern]   Rust vs C++: min %+.2f%%  p25 %+.2f%%\n", 100.0 * (d - a) / a,
              100.0 * (e - b) / b);
  stats3(cpp4c, &a, &b, &c);
  stats3(rust4c, &d, &e, &f);
  std::printf("[kern] Q4_K rdtsc cycles/row: C++ min %.0f / Rust min %.0f  (%+.2f%%)\n", a, d,
              100.0 * (d - a) / a);
  double v2a, v2b, v2c2;
  stats3(v2_4, &v2a, &v2b, &v2c2);
  stats3(v2_4c, &a, &b, &c);
  std::printf("[kern] Rust-v2 (hand-unrolled) min %8.1f  p25 %8.1f  med %8.1f  -> %.2f GB/s/core, "
              "%.0f cyc/row\n", v2a, v2b, v2c2, 1152.0 / v2a, a);
  double v3a, v3b, v3c;
  stats3(v3_4, &v3a, &v3b, &v3c);
  stats3(v3_4c, &a, &b, &c);
  std::printf("[kern] Rust-v3 (no arrays)       min %8.1f  p25 %8.1f  med %8.1f  -> %.2f GB/s/core, "
              "%.0f cyc/row\n", v3a, v3b, v3c, 1152.0 / v3a, a);
  stats3(cpp6, &a, &b, &c);
  stats3(rust6, &d, &e, &f);
  std::printf("[kern] Q6_K row (420 B, 2048 rows/expert) ns/row:\n");
  std::printf("[kern]   C++  min %8.1f  p25 %8.1f  med %8.1f  -> %.2f GB/s/core (min)\n", a, b, c,
              420.0 / a);
  std::printf("[kern]   Rust min %8.1f  p25 %8.1f  med %8.1f  -> %.2f GB/s/core (min)\n", d, e, f,
              420.0 / d);
  std::printf("[kern]   Rust vs C++: min %+.2f%%  p25 %+.2f%%\n", 100.0 * (d - a) / a,
              100.0 * (e - b) / b);
  std::printf("[kern] sink=%g\n", static_cast<float>(sink));
  return 0;
}

double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const std::size_t i = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
  return v[std::min(i, v.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s verify | ab <rounds> | abphasea <rounds> [datadir]\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  int argi = 2;
  int rounds = 8;
  if (mode != "verify") rounds = (argc > 2) ? std::atoi(argv[2]) : 8, argi = 3;
  const std::string dir = (argc > argi) ? argv[argi] : "build/rustprobe";
  int nthreads = 12;
  if (const char* e = std::getenv("RP_THREADS")) nthreads = std::atoi(e);
  const bool phase_a_only = (mode == "abphasea");

  // affinity pairs: default to the production topology helper (SMT siblings)
  if (const char* e = std::getenv("RP_PAIRS")) {
    std::string str(e);
    std::size_t pos = 0;
    while (pos < str.size()) {
      std::size_t semi = str.find(';', pos);
      if (semi == std::string::npos) semi = str.size();
      const std::string item = str.substr(pos, semi - pos);
      const std::size_t comma = item.find(',');
      if (comma != std::string::npos)
        g_pairs.emplace_back(std::atoi(item.substr(0, comma).c_str()),
                             std::atoi(item.substr(comma + 1).c_str()));
      pos = semi + 1;
    }
  } else {
    g_pairs = cpu::smt_cpu_pairs();
  }

  const std::vector<unsigned char> gate = load_file(dir + "/gate8.bin");
  const std::vector<unsigned char> up = load_file(dir + "/up8.bin");
  const std::vector<unsigned char> down = load_file(dir + "/down8.bin");
  if (gate.size() != kGateBytes || up.size() != kGateBytes || down.size() != kDownBytes) {
    std::fprintf(stderr, "bad slab sizes\n");
    return 1;
  }
  std::vector<unsigned char> copies(static_cast<std::size_t>(kCopies) * kCopyBytes);
  for (int c = 0; c < kCopies; ++c) {
    unsigned char* p = copies.data() + static_cast<std::size_t>(c) * kCopyBytes;
    std::memcpy(p, gate.data(), kGateBytes);
    std::memcpy(p + kGateBytes, up.data(), kGateBytes);
    std::memcpy(p + 2 * kGateBytes, down.data(), kDownBytes);
  }
  std::vector<float> x(kHidden), aw(kActs);
  for (int i = 0; i < kHidden; ++i)
    x[static_cast<std::size_t>(i)] = static_cast<float>(std::sin(0.01 * (i + 1)));
  for (int a = 0; a < kActs; ++a) aw[static_cast<std::size_t>(a)] = 0.1f + 0.01f * a;
  // dump x so the standalone Rust binary can use the identical activation
  {
    FILE* f = std::fopen((dir + "/x.bin").c_str(), "wb");
    if (f) {
      std::fwrite(x.data(), 4, x.size(), f);
      std::fwrite(aw.data(), 4, aw.size(), f);
      std::fclose(f);
    }
  }
  std::vector<float> abuf(static_cast<std::size_t>(kActs) * kInter);
  std::vector<float> dbuf(static_cast<std::size_t>(kActs) * kHidden);
  std::vector<float> xout(kHidden, 0.f);

  Shared sh;
  sh.base = copies.data();
  sh.x = x.data();
  sh.aw = aw.data();
  sh.abuf = abuf.data();
  sh.dbuf = dbuf.data();
  sh.xout = xout.data();
  sh.nthreads = nthreads;
  sh.phase_a_only = phase_a_only ? 1 : 0;
  sh.bar.n = nthreads;

  SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
  std::printf("[ab] mode=%s threads=%d  pairs:", mode.c_str(), nthreads);
  for (auto& p : g_pairs) std::printf(" (%d,%d)", p.first, p.second);
  const double bytes = phase_a_only ? static_cast<double>(2 * kGateBytes) : kCopyBytes;
  std::printf("\n[ab] bytes/pass=%.2f MB  C++ kernels=src/cpu_matvec.cpp  Rust=rustc "
              "-C opt-level=3 -C target-cpu=native\n",
              bytes / 1048576.0);

  // single-thread kernel sanity: bit-identical dot products?
  {
    double max_abs = 0;
    int nz = 0;
    for (int r = 0; r < 64; ++r) {
      float a = 0, b = 0;
      cpu::avx2_dot_q4k(gate.data() + static_cast<std::size_t>(r) * (kHidden / 256) * 144, x.data(),
                        kHidden, &a);
      b = rust_dot_q4k(gate.data() + static_cast<std::size_t>(r) * (kHidden / 256) * 144, x.data(),
                       kHidden);
      const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
      if (d > max_abs) max_abs = d;
      if (a != b) ++nz;
    }
    std::printf("[ab] q4k row dots (64 rows): max|C++ - Rust| = %.3e, not-bit-equal rows = %d\n",
                max_abs, nz);
    max_abs = 0;
    nz = 0;
    for (int r = 0; r < 64; ++r) {
      float a = 0;
      cpu::avx2_dot_q6k(down.data() + static_cast<std::size_t>(r) * (kInter / 256) * 210,
                        abuf.data(), kInter, &a);
      const float b = rust_dot_q6k(down.data() + static_cast<std::size_t>(r) * (kInter / 256) * 210,
                                   abuf.data(), kInter);
      const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
      if (d > max_abs) max_abs = d;
      if (a != b) ++nz;
    }
    std::printf("[ab] q6k row dots (64 rows): max|C++ - Rust| = %.3e, not-bit-equal rows = %d\n",
                max_abs, nz);
    max_abs = 0;
    nz = 0;
    for (int r = 0; r < 64; ++r) {
      float a = 0;
      cpu::avx2_dot_q4k(gate.data() + static_cast<std::size_t>(r) * (kHidden / 256) * 144, x.data(),
                        kHidden, &a);
      const float b = rust_dot_q4k_v2(gate.data() + static_cast<std::size_t>(r) * (kHidden / 256) * 144,
                                      x.data(), kHidden);
      const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
      if (d > max_abs) max_abs = d;
      if (a != b) ++nz;
    }
    std::printf("[ab] q4k-v2 row dots (64 rows): max|C++ - Rustv2| = %.3e, not-bit-equal = %d\n",
                max_abs, nz);
  }

  if (mode == "kernel") {
    return mode_kernel(gate, x, down, abuf, rounds);
  }
  if (mode == "verify") {
    std::vector<float> out_cpp(kHidden), out_rust(kHidden);
    std::fill(xout.begin(), xout.end(), 0.f);
    sh.use_rust = 0;
    run_once(&sh, 0);
    out_cpp = xout;
    std::fill(xout.begin(), xout.end(), 0.f);
    std::fill(abuf.begin(), abuf.end(), 0.f);
    sh.use_rust = 1;
    run_once(&sh, 0);
    out_rust = xout;
    double max_abs = 0;
    int not_equal = 0;
    for (int j = 0; j < kHidden; ++j) {
      const double d = std::fabs(static_cast<double>(out_cpp[static_cast<std::size_t>(j)]) -
                                 static_cast<double>(out_rust[static_cast<std::size_t>(j)]));
      if (d > max_abs) max_abs = d;
      if (out_cpp[static_cast<std::size_t>(j)] != out_rust[static_cast<std::size_t>(j)])
        ++not_equal;
    }
    std::printf("[ab] full layer pass: max|xout_C++ - xout_Rust| = %.3e, not-bit-equal elems = "
                "%d/%d\n",
                max_abs, not_equal, kHidden);
    return 0;
  }

  // warm-up
  {
    std::fill(xout.begin(), xout.end(), 0.f);
    int warm_copy = 0;
    time_run(&sh, 2, 0, &warm_copy);
    time_run(&sh, 2, 1, &warm_copy);
  }

  int sample_iters = 4;
  if (const char* e = std::getenv("RP_ITERS")) sample_iters = std::atoi(e);
  // measure the per-sample fixed cost (thread create/join + first-touch) so the
  // report can show it: a 0-iteration time_run still creates and joins the team.
  double create_overhead_ms = 0;
  {
    int c0 = 0;
    create_overhead_ms = time_run(&sh, 0, 0, &c0);
  }
  std::printf("[ab] sample_iters=%d  thread create/join fixed cost=%.3f ms/sample\n",
              sample_iters, create_overhead_ms);
  std::vector<double> cpp_t, rust_t;
  int copy_cpp = 0, copy_rust = 12;
  // interleaved: each implementation gets a fresh copy index every round
  for (int r = 0; r < rounds; ++r) {
    // alternate which one runs first to cancel any within-round order effect
    if (r % 2 == 0) {
      cpp_t.push_back(time_run(&sh, sample_iters, 0, &copy_cpp));
      rust_t.push_back(time_run(&sh, sample_iters, 1, &copy_rust));
    } else {
      rust_t.push_back(time_run(&sh, sample_iters, 1, &copy_rust));
      cpp_t.push_back(time_run(&sh, sample_iters, 0, &copy_cpp));
    }
  }
  auto report = [&](const char* name, const std::vector<double>& v) {
    const double mn = *std::min_element(v.begin(), v.end());
    const double p25 = quantile(v, 0.25), md = quantile(v, 0.5);
    std::printf("[ab] %-5s min %7.4f ms (%6.2f GB/s)  p25 %7.4f (%6.2f)  med %7.4f (%6.2f)\n",
                name, mn, bytes / (mn * 1e6), p25, bytes / (p25 * 1e6), md, bytes / (md * 1e6));
  };
  report("C++", cpp_t);
  report("Rust", rust_t);
  const double cpp_min = *std::min_element(cpp_t.begin(), cpp_t.end());
  const double rust_min = *std::min_element(rust_t.begin(), rust_t.end());
  const double cpp_p25 = quantile(cpp_t, 0.25), rust_p25 = quantile(rust_t, 0.25);
  std::printf("[ab] net of setup: C++ min %.4f ms  Rust min %.4f ms (setup %.3f ms removed)\n",
              cpp_min - create_overhead_ms / static_cast<double>(sample_iters),
              rust_min - create_overhead_ms / static_cast<double>(sample_iters),
              create_overhead_ms / static_cast<double>(sample_iters));
  std::printf("[ab] Rust vs C++: min %+.2f%%  p25 %+.2f%%  (positive = Rust slower)\n",
              100.0 * (rust_min - cpp_min) / cpp_min, 100.0 * (rust_p25 - cpp_p25) / cpp_p25);
  std::printf("[ab] paired samples C++: ");
  for (double v : cpp_t) std::printf("%.4f ", v);
  std::printf("\n[ab] paired samples Rust:");
  for (double v : rust_t) std::printf(" %.4f", v);
  std::printf("\n");
  return 0;
}
