// Micro-benchmark for the CPU-resident MoE expert path of the 35B model.
//
// Why this tool exists: tools/bench_cpu_matvec.cpp uses one matrix that fits in
// L3, so it measures cache bandwidth, not DRAM.  This tool rotates over several
// independent copies of a full CPU layer's expert weights (16.3 MB each) so the
// working set is many times L3 (12 MB) and every iteration streams from DRAM --
// the same situation as decode, where each token walks 25 CPU layers x 16 MB.
//
// Modes:
//   layer  <copies> <iters>            full cpu::moe_forward_mt (8 acts, 35B shape)
//   matvec <type> <n_out> <n_in> <copies> <iters>   cpu::matvec_mt on rotated copies
//   read   <copies> <iters> <nthr> [aff]  pure AVX2 32B-load read of the same
//                                       number of bytes (thread-count / affinity probe)
//   readrows <copies> <iters> <nthr> [aff] same but row-shaped (1152/1152/420 B)
//
// aff: "compact" = first n physical cores, "smt" = leave SMT siblings idle,
//      "none" (default).
#include <immintrin.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "engine/config.h"
#include "engine/cpu_matvec.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace {

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void put_half(unsigned char* p, float v) {
  // enough for the values we synthesise (2^-6 .. 2^6)
  unsigned int bits;
  std::memcpy(&bits, &v, 4);
  const unsigned int sign = (bits >> 16) & 0x8000u;
  int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  unsigned int mant = (bits >> 13) & 0x3FFu;
  if (exp < -14) {  // flush to zero, we never need subnormals
    exp = 0;
    mant = 0;
    unsigned int h = sign;
    p[0] = static_cast<unsigned char>(h & 0xFF);
    p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
    return;
  }
  const unsigned int h = sign | (static_cast<unsigned int>(exp + 15) << 10) | mant;
  p[0] = static_cast<unsigned char>(h & 0xFF);
  p[1] = static_cast<unsigned char>((h >> 8) & 0xFF);
}

// Fill one quantized block with finite, plausible data: block_elems values that
// dequantise to roughly [-2, 2].  Without this the half scales would be NaNs.
void fill_block(const std::string& type, unsigned char* p) {
  if (type == "Q4_K") {
    put_half(p + 0, 0.015625f);
    put_half(p + 2, 0.015625f);
    for (int i = 0; i < 12; ++i) p[4 + i] = static_cast<unsigned char>(0x21 + i * 7);
    for (int i = 0; i < 128; ++i) p[16 + i] = static_cast<unsigned char>((i * 37) & 0xFF);
  } else if (type == "Q5_K") {
    put_half(p + 0, 0.015625f);
    put_half(p + 2, 0.015625f);
    for (int i = 0; i < 12; ++i) p[4 + i] = static_cast<unsigned char>(0x21 + i * 7);
    for (int i = 0; i < 160; ++i) p[16 + i] = static_cast<unsigned char>((i * 37) & 0xFF);
  } else if (type == "Q6_K") {
    for (int i = 0; i < 128; ++i) p[i] = static_cast<unsigned char>((i * 37) & 0xFF);
    for (int i = 0; i < 64; ++i) p[128 + i] = static_cast<unsigned char>((i * 91) & 0xFF);
    for (int i = 0; i < 16; ++i)
      p[192 + i] = static_cast<unsigned char>(static_cast<signed char>(1 + (i % 7)));
    put_half(p + 208, 0.02f);
  } else if (type == "Q8_0") {
    put_half(p + 0, 0.02f);
    for (int i = 0; i < 32; ++i) p[2 + i] = static_cast<unsigned char>((i * 29) & 0xFF);
  } else {
    std::fprintf(stderr, "unsupported type %s\n", type.c_str());
    std::exit(1);
  }
}

void fill_tensor(const std::string& type, unsigned char* p, std::size_t bytes) {
  int bb = 0;
  if (type == "Q4_K") bb = 144;
  else if (type == "Q5_K") bb = 176;
  else if (type == "Q6_K") bb = 210;
  else if (type == "Q8_0") bb = 34;
  else {
    std::fprintf(stderr, "unsupported type %s\n", type.c_str());
    std::exit(1);
  }
  for (std::size_t o = 0; o + static_cast<std::size_t>(bb) <= bytes; o += static_cast<std::size_t>(bb))
    fill_block(type, p + o);
}

int row_bytes_of(const std::string& type, int n_in) {
  if (type == "Q4_K") return (n_in / 256) * 144;
  if (type == "Q5_K") return (n_in / 256) * 176;
  if (type == "Q6_K") return (n_in / 256) * 210;
  if (type == "Q8_0") return (n_in / 32) * 34;
  return 0;
}

// blocks/nibbles are byte-wise; avoid the compiler folding everything away.
std::atomic<unsigned long long> g_sink{0};

inline void stream_read(const unsigned char* p, std::size_t bytes) {
  __m256i a = _mm256_setzero_si256();
  __m256i b = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 64 <= bytes; i += 64) {
    a = _mm256_add_epi64(a, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)));
    b = _mm256_add_epi64(b, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i + 32)));
  }
  for (; i + 16 <= bytes; i += 16) {
    __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i));
    a = _mm256_add_epi64(a, _mm256_castsi128_si256(x));
  }
  a = _mm256_add_epi64(a, b);
  __m128i lo = _mm256_castsi256_si128(a);
  __m128i hi = _mm256_extracti128_si256(a, 1);
  lo = _mm_add_epi64(lo, hi);
  std::uint64_t out[2];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out), lo);
  g_sink.fetch_add(out[0] + out[1], std::memory_order_relaxed);
}

#ifdef _WIN32
unsigned long long g_mask_override = 0;
void apply_affinity(int id) {
  if (g_mask_override) {
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(g_mask_override));
  }
}
#else
void apply_affinity(int) {}
#endif

// 35B CPU layer geometry.
constexpr int kHidden = 2048;
constexpr int kInter = 512;
constexpr int kActs = 8;  // topk

double bytes_per_layer() {
  const std::size_t gate = static_cast<std::size_t>(kActs) * kInter * (kHidden / 256) * 144;
  const std::size_t up = gate;
  const std::size_t down = static_cast<std::size_t>(kActs) * kHidden * (kInter / 256) * 210;
  return static_cast<double>(gate + up + down);
}

int mode_layer(int copies, int iters) {
  const std::size_t gate_bytes = static_cast<std::size_t>(kActs) * kInter * (kHidden / 256) * 144;
  const std::size_t down_bytes = static_cast<std::size_t>(kActs) * kHidden * (kInter / 256) * 210;
  std::vector<std::vector<unsigned char>> g(static_cast<std::size_t>(copies));
  std::vector<std::vector<unsigned char>> u(static_cast<std::size_t>(copies));
  std::vector<std::vector<unsigned char>> d(static_cast<std::size_t>(copies));
  for (int c = 0; c < copies; ++c) {
    g[static_cast<std::size_t>(c)].resize(gate_bytes);
    u[static_cast<std::size_t>(c)].resize(gate_bytes);
    d[static_cast<std::size_t>(c)].resize(down_bytes);
    fill_tensor("Q4_K", g[static_cast<std::size_t>(c)].data(), gate_bytes);
    fill_tensor("Q4_K", u[static_cast<std::size_t>(c)].data(), gate_bytes);
    fill_tensor("Q6_K", d[static_cast<std::size_t>(c)].data(), down_bytes);
  }
  std::vector<const unsigned char*> wg(static_cast<std::size_t>(kActs));
  std::vector<const unsigned char*> wu(static_cast<std::size_t>(kActs));
  std::vector<const unsigned char*> wd(static_cast<std::size_t>(kActs));
  std::vector<int> tok(static_cast<std::size_t>(kActs), 0);
  std::vector<float> aw(static_cast<std::size_t>(kActs));
  for (int a = 0; a < kActs; ++a) aw[static_cast<std::size_t>(a)] = 0.1f + 0.01f * a;
  std::vector<float> x(static_cast<std::size_t>(1) * kHidden);
  for (int i = 0; i < kHidden; ++i) x[static_cast<std::size_t>(i)] = std::sin(0.01f * (i + 1));
  std::vector<float> xout(static_cast<std::size_t>(kHidden), 0.f);

  const std::size_t g_slab = gate_bytes / kActs;
  const std::size_t d_slab = down_bytes / kActs;
  auto bind = [&](int c) {
    for (int a = 0; a < kActs; ++a) {
      wg[static_cast<std::size_t>(a)] = g[static_cast<std::size_t>(c)].data() + a * g_slab;
      wu[static_cast<std::size_t>(a)] = u[static_cast<std::size_t>(c)].data() + a * g_slab;
      wd[static_cast<std::size_t>(a)] = d[static_cast<std::size_t>(c)].data() + a * d_slab;
    }
  };
  bind(0);
  cpu::moe_forward_mt("Q4_K", wg.data(), "Q4_K", wu.data(), "Q6_K", wd.data(), tok.data(),
                      aw.data(), kActs, 1, kHidden, kInter, x.data(), xout.data());

  const double t0 = now_ms();
  for (int it = 0; it < iters; ++it) {
    bind(it % copies);
    cpu::moe_forward_mt("Q4_K", wg.data(), "Q4_K", wu.data(), "Q6_K", wd.data(), tok.data(),
                        aw.data(), kActs, 1, kHidden, kInter, x.data(), xout.data());
  }
  const double t1 = now_ms();
  double sum = 0;
  for (float v : xout) sum += v;
  const double ms = (t1 - t0) / iters;
  const double b = bytes_per_layer();
  std::printf("layer   : %d copies x %.2f MB, %d iters | %.3f ms/layer | %.2f GB/s | chk %.6f\n",
              copies, b / 1048576.0, iters, ms, b / (ms * 1e6), sum);
  return 0;
}

int mode_matvec(const std::string& type, int n_out, int n_in, int copies, int iters) {
  const int rb = row_bytes_of(type, n_in);
  const std::size_t mat_bytes = static_cast<std::size_t>(rb) * n_out;
  std::vector<std::vector<unsigned char>> w(static_cast<std::size_t>(copies));
  for (int c = 0; c < copies; ++c) {
    w[static_cast<std::size_t>(c)].resize(mat_bytes);
    fill_tensor(type, w[static_cast<std::size_t>(c)].data(), mat_bytes);
  }
  std::vector<float> x(static_cast<std::size_t>(n_in));
  for (int i = 0; i < n_in; ++i) x[static_cast<std::size_t>(i)] = std::sin(0.01f * (i + 1));
  std::vector<float> y(static_cast<std::size_t>(n_out));
  cpu::matvec_mt(type, w[0].data(), n_out, n_in, x.data(), y.data());
  const double t0 = now_ms();
  for (int it = 0; it < iters; ++it)
    cpu::matvec_mt(type, w[static_cast<std::size_t>(it % copies)].data(), n_out, n_in, x.data(),
                   y.data());
  const double t1 = now_ms();
  double sum = 0;
  for (float v : y) sum += v;
  const double ms = (t1 - t0) / iters;
  std::printf("matvec  : %s %dx%d, %d copies x %.2f MB, %d iters | %.3f ms | %.2f GB/s | chk %.6f\n",
              type.c_str(), n_out, n_in, copies, mat_bytes / 1048576.0, iters, ms,
              mat_bytes / (ms * 1e6), sum);
  return 0;
}

// Pure read bandwidth over the same layer-sized working set.
int mode_read(int copies, int iters, int nthr, const char* aff) {
  const std::size_t layer_bytes =
      static_cast<std::size_t>(kActs) * kInter * (kHidden / 256) * 144 * 2 +
      static_cast<std::size_t>(kActs) * kHidden * (kInter / 256) * 210;
  std::vector<unsigned char> buf(layer_bytes * static_cast<std::size_t>(copies));
  fill_tensor("Q4_K", buf.data(), buf.size());
#ifdef _WIN32
  if (aff && std::strcmp(aff, "smt") == 0) g_mask_override = 0x555;      // one thread per core
  else if (aff && std::strcmp(aff, "compact") == 0) g_mask_override = 0xFFF;
#endif
  std::vector<std::thread> th;
  for (int t = 0; t < nthr; ++t) {
    th.emplace_back([&, t] {
      apply_affinity(t);
      const std::size_t chunk = layer_bytes / static_cast<std::size_t>(nthr);
      for (int it = 0; it < iters; ++it) {
        const unsigned char* base = buf.data() + static_cast<std::size_t>(it % copies) * layer_bytes;
        // each thread reads the row-shaped pattern of its contiguous slice
        std::size_t off = static_cast<std::size_t>(t) * chunk;
        std::size_t end = (t == nthr - 1) ? layer_bytes : off + chunk;
        std::size_t p = off;
        while (p + 1024 <= end) {
          stream_read(base + p, 1024);
          p += 1024;
        }
        if (p < end) stream_read(base + p, end - p);
      }
    });
  }
  const double t0 = now_ms();
  for (auto& x : th) x.join();
  // threads start before t0 only marginally; measure from spawn instead
  (void)t0;
  return 0;
}

// Read the same working set measured with a clean start (threads spawned, then timed).
int mode_read2(int copies, int iters, int nthr, const char* aff) {
  const std::size_t layer_bytes =
      static_cast<std::size_t>(kActs) * kInter * (kHidden / 256) * 144 * 2 +
      static_cast<std::size_t>(kActs) * kHidden * (kInter / 256) * 210;
  std::vector<unsigned char> buf(layer_bytes * static_cast<std::size_t>(copies));
  fill_tensor("Q4_K", buf.data(), buf.size());
  // warm-up pass so pages are resident
  for (std::size_t off = 0; off < buf.size(); off += 64) buf[off] = 0;
#ifdef _WIN32
  if (aff && std::strcmp(aff, "smt") == 0) g_mask_override = 0x555;
  else if (aff && std::strcmp(aff, "compact") == 0) g_mask_override = 0xFFF;
#endif
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> th;
  for (int t = 0; t < nthr; ++t) {
    th.emplace_back([&, t] {
      apply_affinity(t);
      const std::size_t chunk = layer_bytes / static_cast<std::size_t>(nthr);
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) _mm_pause();
      for (int it = 0; it < iters; ++it) {
        const unsigned char* base = buf.data() + static_cast<std::size_t>(it % copies) * layer_bytes;
        std::size_t off = static_cast<std::size_t>(t) * chunk;
        std::size_t end = (t == nthr - 1) ? layer_bytes : off + chunk;
        std::size_t p = off;
        while (p + 1024 <= end) {
          stream_read(base + p, 1024);
          p += 1024;
        }
        if (p < end) stream_read(base + p, end - p);
      }
    });
  }
  while (ready.load() != nthr) std::this_thread::yield();
  const double t0 = now_ms();
  go.store(true, std::memory_order_release);
  for (auto& x : th) x.join();
  const double t1 = now_ms();
  const double ms = (t1 - t0) / iters;
  std::printf("read    : %d thr, %d copies x %.2f MB | %.3f ms/pass | %.2f GB/s | sink %llu\n", nthr,
              copies, layer_bytes / 1048576.0, ms, layer_bytes / (ms * 1e6),
              g_sink.load());
  return 0;
}

// ---------------------------------------------------------------------------
// Priority experiment: normal vs HIGH vs ABOVE_NORMAL, plus the "main thread vs
// pool workers" split.  Single process, interleaved arms, order flipped every
// round -> paired comparison that cancels the minutes-scale drift of this box
// (the same methodology tools/bench_rp.cpp uses).
// ---------------------------------------------------------------------------
#ifdef _WIN32
void bench_set_process_class(int lvl) {
  DWORD cls = NORMAL_PRIORITY_CLASS;
  if (lvl == 1) cls = HIGH_PRIORITY_CLASS;
  else if (lvl == 2) cls = ABOVE_NORMAL_PRIORITY_CLASS;
  SetPriorityClass(GetCurrentProcess(), cls);
}

void bench_set_thread_priority(DWORD tid, int rel) {
  HANDLE h = OpenThread(THREAD_SET_INFORMATION, FALSE, tid);
  if (h) {
    SetThreadPriority(h, rel);
    CloseHandle(h);
  }
}

// Every thread of this process except the calling one (i.e. the cpu::Pool workers).
void bench_set_worker_priorities(int rel) {
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snap == INVALID_HANDLE_VALUE) return;
  THREADENTRY32 te;
  std::memset(&te, 0, sizeof(te));
  te.dwSize = sizeof(te);
  const DWORD pid = GetCurrentProcessId();
  const DWORD self = GetCurrentThreadId();
  if (Thread32First(snap, &te)) {
    do {
      if (te.th32OwnerProcessID == pid && te.th32ThreadID != self)
        bench_set_thread_priority(te.th32ThreadID, rel);
    } while (Thread32Next(snap, &te));
  }
  CloseHandle(snap);
}

struct PrioArm {
  const char* name;
  int proc_class;  // 0 = normal, 1 = high, 2 = above normal (SetPriorityClass)
  int main_rel;    // THREAD_PRIORITY_* for the calling thread
  int worker_rel;  // THREAD_PRIORITY_* for the pool workers
};

int mode_prio(int copies, int iters, int rounds) {
#else
int mode_prio(int copies, int iters, int rounds) {
  (void)copies; (void)iters; (void)rounds;
  std::printf("prio    : Windows-only experiment\n");
  return 2;
#endif
#if defined(_WIN32)
  const int kActs = 8;
  const std::size_t gate_bytes = static_cast<std::size_t>(8) * 512 * (2048 / 256) * 144;
  const std::size_t down_bytes = static_cast<std::size_t>(8) * 2048 * (512 / 256) * 210;
  const int kHidden = 2048, kInter = 512;
  std::vector<std::vector<unsigned char>> g(static_cast<std::size_t>(copies));
  std::vector<std::vector<unsigned char>> u(static_cast<std::size_t>(copies));
  std::vector<std::vector<unsigned char>> d(static_cast<std::size_t>(copies));
  for (int c = 0; c < copies; ++c) {
    g[static_cast<std::size_t>(c)].resize(gate_bytes);
    u[static_cast<std::size_t>(c)].resize(gate_bytes);
    d[static_cast<std::size_t>(c)].resize(down_bytes);
    fill_tensor("Q4_K", g[static_cast<std::size_t>(c)].data(), gate_bytes);
    fill_tensor("Q4_K", u[static_cast<std::size_t>(c)].data(), gate_bytes);
    fill_tensor("Q6_K", d[static_cast<std::size_t>(c)].data(), down_bytes);
  }
  std::vector<const unsigned char*> wg(static_cast<std::size_t>(kActs));
  std::vector<const unsigned char*> wu(static_cast<std::size_t>(kActs));
  std::vector<const unsigned char*> wd(static_cast<std::size_t>(kActs));
  std::vector<int> tok(static_cast<std::size_t>(kActs), 0);
  std::vector<float> aw(static_cast<std::size_t>(kActs));
  for (int a = 0; a < kActs; ++a) aw[static_cast<std::size_t>(a)] = 0.1f + 0.01f * a;
  std::vector<float> x(static_cast<std::size_t>(kHidden));
  for (int i = 0; i < kHidden; ++i) x[static_cast<std::size_t>(i)] = std::sin(0.01f * (i + 1));
  std::vector<float> xout(static_cast<std::size_t>(kHidden), 0.f);
  const std::size_t g_slab = gate_bytes / kActs;
  const std::size_t d_slab = down_bytes / kActs;
  auto bind = [&](int c) {
    for (int a = 0; a < kActs; ++a) {
      wg[static_cast<std::size_t>(a)] = g[static_cast<std::size_t>(c)].data() + a * g_slab;
      wu[static_cast<std::size_t>(a)] = u[static_cast<std::size_t>(c)].data() + a * g_slab;
      wd[static_cast<std::size_t>(a)] = d[static_cast<std::size_t>(c)].data() + a * d_slab;
    }
  };
  auto run_cell = [&](int iters_cell) {
    const double t0 = now_ms();
    for (int it = 0; it < iters_cell; ++it) {
      bind(it % copies);
      cpu::moe_forward_mt("Q4_K", wg.data(), "Q4_K", wu.data(), "Q6_K", wd.data(), tok.data(),
                          aw.data(), kActs, 1, kHidden, kInter, x.data(), xout.data());
    }
    const double t1 = now_ms();
    return (t1 - t0) / iters_cell;
  };
  bind(0);
  cpu::moe_forward_mt("Q4_K", wg.data(), "Q4_K", wu.data(), "Q6_K", wd.data(), tok.data(),
                      aw.data(), kActs, 1, kHidden, kInter, x.data(), xout.data());  // creates the pool
  std::vector<PrioArm> arms = {
      {"normal", 0, THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_NORMAL},
      {"proc-high", 1, THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_NORMAL},
      {"high+workers-highest", 1, THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_HIGHEST},
      {"above-normal+workers-highest", 2, THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_HIGHEST},
      {"workers-highest-only", 0, THREAD_PRIORITY_NORMAL, THREAD_PRIORITY_HIGHEST},
      {"main-highest-only", 0, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_NORMAL},
  };
  std::vector<std::vector<double>> cells(arms.size());
  for (int r = 0; r < rounds; ++r) {
    const bool flip = (r % 2) != 0;
    for (int k = 0; k < static_cast<int>(arms.size()); ++k) {
      const int ai = flip ? static_cast<int>(arms.size()) - 1 - k : k;
      const PrioArm& a = arms[static_cast<std::size_t>(ai)];
      bench_set_process_class(a.proc_class);
      SetThreadPriority(GetCurrentThread(), a.main_rel);
      bench_set_worker_priorities(a.worker_rel);
      const double ms = run_cell(iters);
      cells[static_cast<std::size_t>(ai)].push_back(ms);
      std::fprintf(stderr, "[prio] round %d arm %-30s %.4f ms\n", r + 1, a.name, ms);
    }
  }
  // restore for the rest of the process
  bench_set_process_class(0);
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
  bench_set_worker_priorities(THREAD_PRIORITY_NORMAL);
  double sum = 0;
  for (float v : xout) sum += v;
  const double b = bytes_per_layer();
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0;
  };
  const double base_med = med(cells[0]);
  std::printf("prio    : %d copies x %.2f MB, %d iters/cell, %d rounds, %d arms, interleaved\n",
              copies, b / 1048576.0, iters, rounds, static_cast<int>(arms.size()));
  std::printf("%-30s %8s %8s %8s %8s %8s\n", "arm", "min", "p25", "median", "mean", "vs-normal");
  for (size_t i = 0; i < arms.size(); ++i) {
    auto v = cells[i];
    std::sort(v.begin(), v.end());
    const double mn = v.front();
    const double p25 = v[v.size() / 4];
    const double md = med(v);
    double mean = 0;
    for (double x2 : v) mean += x2;
    mean /= static_cast<double>(v.size());
    std::printf("%-30s %8.4f %8.4f %8.4f %8.4f %+7.1f%%\n", arms[i].name, mn, p25, md, mean,
                100.0 * (md - base_med) / base_med);
  }
  std::printf("chk     : %.6f (identical across arms => bit-exact)  %.2f GB/s @ normal median\n",
              sum, b / (base_med * 1e6));
  return 0;
#endif
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // Same startup hook as the engine: makes ENGINE_WIN_PRIORITY / --win-priority apply to
  // the `layer`/`matvec` arms too (the `prio` mode drives the classes itself, interleaved).
  engine::config::apply_win_priority();
  if (argc < 2) {
    std::printf(
        "usage:\n"
        "  %s layer <copies> <iters>\n"
        "  %s matvec <type> <n_out> <n_in> <copies> <iters>\n"
        "  %s read <copies> <iters> <nthr> [none|compact|smt]\n"
        "  %s prio <copies> <iters> [rounds]   # normal/high/above-normal, interleaved\n",
        argv[0], argv[0], argv[0], argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  if (mode == "layer") {
    return mode_layer(std::atoi(argv[2]), std::atoi(argv[3]));
  }
  if (mode == "matvec") {
    return mode_matvec(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                       std::atoi(argv[6]));
  }
  if (mode == "read") {
    return mode_read2(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
                      argc > 5 ? argv[5] : "none");
  }
  if (mode == "prio") {
    const int rounds = argc > 4 ? std::atoi(argv[4]) : 5;
    return mode_prio(std::atoi(argv[2]), std::atoi(argv[3]), rounds);
  }
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
