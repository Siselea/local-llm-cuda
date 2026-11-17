// Microbenchmark for the qwen35 gated-delta-net decode kernels.
//
// The 9B decode profile shows `gd_delta_rule` is the single largest non-GEMV
// kernel (~0.12 ms/call x 24 calls).  This tool measures the stock kernel and
// candidate replacements in isolation on the model's real shapes (T = 1 decode).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "engine/cuda.h"
#include "engine/kernels.h"
#include "bench_gd_kernels.h"

namespace {

constexpr int kNH = 32;   // n_value_head
constexpr int kNKH = 16;  // n_key_head
constexpr int kKD = 128;  // key_head_dim
constexpr int kVD = 128;  // value_head_dim

double bench_launch(void* fn, int gx, int bx, size_t shared, void** args, int iters,
                    const char* tag) {
  cuda::launch(fn, gx, 1, 1, bx, 1, 1, shared, args);
  cuda::sync();
  std::vector<double> med;
  for (int rep = 0; rep < 5; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) cuda::launch(fn, gx, 1, 1, bx, 1, 1, shared, args);
    cuda::sync();
    med.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                      .count() /
                  iters);
  }
  std::sort(med.begin(), med.end());
  std::printf("  %-30s %9.5f ms/launch\n", tag, med[2]);
  return med[2];
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double na = 0, nb = 0, nd = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
    nd += double(a[i] - b[i]) * (a[i] - b[i]);
  }
  (void)nb;
  return std::sqrt(nd / (na > 1e-30 ? na : 1e-30));
}

// CPU reference (double): one delta-rule step, layout [nh][VD][KD] (logical).
void cpu_step(const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& g,
              const std::vector<float>& beta, std::vector<float>& S,
              std::vector<float>& o, bool state_kd_major) {
  const int rep = kNH / kNKH;
  for (int h = 0; h < kNH; ++h) {
    const int kh = h / rep;
    const float* qh = q.data() + (size_t)kh * kKD;
    const float* khv = k.data() + (size_t)kh * kKD;
    const float decay = std::exp(g[h]);
    const float bh = beta[h];
    for (int j = 0; j < kVD; ++j) {
      double kv = 0.0;
      for (int i = 0; i < kKD; ++i) {
        const size_t off = state_kd_major ? (size_t)h * kKD * kVD + (size_t)i * kVD + j
                                          : (size_t)h * kVD * kKD + (size_t)j * kKD + i;
        kv += double(S[off]) * khv[i];
      }
      const double delta = (double(v[(size_t)h * kVD + j]) - kv) * bh;
      double acc = 0.0;
      for (int i = 0; i < kKD; ++i) {
        const size_t off = state_kd_major ? (size_t)h * kKD * kVD + (size_t)i * kVD + j
                                          : (size_t)h * kVD * kKD + (size_t)j * kKD + i;
        const double s = double(S[off]) * decay + double(khv[i]) * delta;
        S[off] = float(s);
        acc += s * qh[i];
      }
      o[(size_t)h * kVD + j] = float(acc);
    }
  }
}

std::vector<float> shuffle_to_layout(const std::vector<float>& S0, bool kd_major) {
  // S0 is [nh][VD][KD]; produce [nh][KD][VD] when kd_major, else copy.
  if (!kd_major) return S0;
  std::vector<float> out(S0.size());
  for (int h = 0; h < kNH; ++h)
    for (int j = 0; j < kVD; ++j)
      for (int i = 0; i < kKD; ++i)
        out[(size_t)h * kKD * kVD + (size_t)i * kVD + j] =
            S0[(size_t)h * kVD * kKD + (size_t)j * kKD + i];
  return out;
}

}  // namespace

int main() {
  cuda::init();
  const auto& di = cuda::device_info();
  std::printf("device: %s sm=%d\n", di.name.c_str(), di.sm_count);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  std::vector<float> hq(kNKH * kKD), hk(kNKH * kKD), hv(kNH * kVD);
  for (auto& x : hq) x = dist(rng);
  for (auto& x : hk) x = dist(rng);
  for (auto& x : hv) x = dist(rng);
  std::vector<float> hg(kNH), hbeta(kNH);
  for (auto& x : hg) x = -0.05f;
  for (auto& x : hbeta) x = 0.5f;
  std::vector<float> hS0((size_t)kNH * kVD * kKD);
  for (auto& x : hS0) x = dist(rng);

  // Reference states.
  std::vector<float> Sref_vd = hS0, Sref_kd = shuffle_to_layout(hS0, true);
  std::vector<float> oref_vd((size_t)kNH * kVD, 0.f), oref_kd(oref_vd.size(), 0.f);
  cpu_step(hq, hk, hv, hg, hbeta, Sref_vd, oref_vd, false);
  cpu_step(hq, hk, hv, hg, hbeta, Sref_kd, oref_kd, true);

  cuda::Buffer dq(hq.size() * 4), dk(hk.size() * 4), dv(hv.size() * 4);
  cuda::Buffer dg(hg.size() * 4), dbeta(hbeta.size() * 4);
  cuda::Buffer dS((size_t)kNH * kVD * kKD * 4);
  cuda::Buffer dO((size_t)kNH * kVD * 4);
  dq.upload(hq.data(), hq.size() * 4);
  dk.upload(hk.data(), hk.size() * 4);
  dv.upload(hv.data(), hv.size() * 4);
  dg.upload(hg.data(), hg.size() * 4);
  dbeta.upload(hbeta.data(), hbeta.size() * 4);

  auto& mod = *new cuda::Module(cuda::compile_ptx(
      std::string(kernels::kGdDelta) + gdk::kGdDeltaV2 + gdk::kGdDeltaV3_128, "gd.cu"));

  int T = 1, nh = kNH, nkh = kNKH, kd = kKD, vd = kVD;
  void* pq = dq.data();
  void* pk = dk.data();
  void* pv = dv.data();
  void* pg = dg.data();
  void* pb = dbeta.data();
  void* pS = dS.data();
  void* po = dO.data();
  void* args[] = {&pq, &pk, &pv, &pg, &pb, &pS, &po, &T, &nh, &nkh, &kd, &vd};

  const std::vector<float> hS0_kd = shuffle_to_layout(hS0, true);
  std::vector<float> stock_S, stock_O;
  auto run_kernel = [&](void* fn, int bx, size_t shared, bool kd_major) {
    dS.upload(kd_major ? hS0_kd.data() : hS0.data(), hS0.size() * 4);
    std::vector<float> o((size_t)kNH * kVD, 0.f);
    dO.upload(o.data(), o.size() * 4);
    cuda::launch(fn, kNH, 1, 1, bx, 1, 1, shared, args);
    cuda::sync();
    std::vector<float> gotS(hS0.size()), gotO(o.size());
    dS.download(gotS.data(), gotS.size() * 4);
    dO.download(gotO.data(), gotO.size() * 4);
    return std::make_tuple(gotS, gotO, kd_major);
  };
  auto to_vd = [](std::vector<float> S, bool kd_major) {
    if (!kd_major) return S;
    std::vector<float> out(S.size());
    for (int h = 0; h < kNH; ++h)
      for (int j = 0; j < kVD; ++j)
        for (int i = 0; i < kKD; ++i)
          out[(size_t)h * kVD * kKD + (size_t)j * kKD + i] =
              S[(size_t)h * kKD * kVD + (size_t)i * kVD + j];
    return out;
  };
  std::printf("correctness (T=1, vs stock float kernel):\n");
  {
    auto r = run_kernel(mod.kernel("gd_delta_rule"), kVD, (size_t)(2 * kKD) * 4, false);
    stock_S = std::get<0>(r);
    stock_O = std::get<1>(r);
    std::printf("  %-24s state rel_l2 %.3e  o rel_l2 %.3e\n", "stock",
                rel_l2(stock_S, stock_S), rel_l2(stock_O, stock_O));
  }
  auto check = [&](void* fn, int bx, size_t shared, bool kd_major, const char* tag) {
    auto r = run_kernel(fn, bx, shared, kd_major);
    const std::vector<float> gotS = to_vd(std::get<0>(r), kd_major);
    const std::vector<float> gotO = std::get<1>(r);
    std::printf("  %-24s state rel_l2 %.3e  o rel_l2 %.3e\n", tag, rel_l2(stock_S, gotS),
                rel_l2(stock_O, gotO));
  };
  check(mod.kernel("gd_delta_v2"), kVD, (size_t)(2 * kKD) * 4, true, "v2 (KD-major)");
  check(mod.kernel("gd_delta_v3_128"), 4 * 128, 0, true, "v3 (register tiled)");

  std::printf("\ntiming (grid=%d, %d iters):\n", kNH, 300);
  bench_launch(mod.kernel("gd_delta_rule"), kNH, kVD, (size_t)(2 * kKD) * 4, args, 300,
               "gd_delta_rule (stock)");
  bench_launch(mod.kernel("gd_delta_v2"), kNH, kVD, (size_t)(2 * kKD) * 4, args, 300,
               "gd_delta_v2 (KD-major 2-pass)");
  bench_launch(mod.kernel("gd_delta_v3_128"), kNH, 4 * 128, 0, args, 300,
               "gd_delta_v3 (register tiled)");
  return 0;
}
