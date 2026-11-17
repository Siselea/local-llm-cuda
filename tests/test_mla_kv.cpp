// Isolates the MLA cache kernels: write 2 tokens in two separate calls, then
// attend for query row 1 and compare against a host reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "engine/cuda.h"
#include "engine/kernels.h"

int main() {
  try {
    cuda::init();
    const int n_head = 4, qk = 12, vdim = 8, max_seq = 16;
    const int T = 2;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> q(static_cast<size_t>(T) * n_head * qk), k(q.size()),
        v(static_cast<size_t>(T) * n_head * vdim);
    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng);
    for (auto& x : v) x = dist(rng);

    std::string src = std::string(kernels::kQuantCommon) + kernels::kMlaKvWrite +
                      kernels::kMlaAttnCached;
    cuda::Module mod(cuda::compile_ptx(src, "mlakv.cu"));
    void* fn_w = mod.kernel("mla_kv_write");
    void* fn_a = mod.kernel("mla_attn_cached");

    cuda::Buffer dq(q.size() * 4), dk(k.size() * 4), dv(v.size() * 4),
        dKc(static_cast<size_t>(n_head) * max_seq * qk * 4),
        dVc(static_cast<size_t>(n_head) * max_seq * vdim * 4),
        dout(static_cast<size_t>(T) * n_head * vdim * 4), dpos(4);
    std::vector<float> q1(q.begin() + static_cast<size_t>(1) * n_head * qk,
                          q.begin() + static_cast<size_t>(2) * n_head * qk);
    dq.upload(q1.data(), q1.size() * 4);
    dk.upload(k.data(), k.size() * 4);
    dv.upload(v.data(), v.size() * 4);

    // write token 0 at pos 0, then token 1 at pos 1 (two separate launches)
    for (int t = 0; t < T; ++t) {
      cuda::Buffer dk1(static_cast<size_t>(n_head) * qk * 4),
          dv1(static_cast<size_t>(n_head) * vdim * 4);
      dk1.upload(k.data() + static_cast<size_t>(t) * n_head * qk,
                 static_cast<size_t>(n_head) * qk * 4);
      dv1.upload(v.data() + static_cast<size_t>(t) * n_head * vdim,
                 static_cast<size_t>(n_head) * vdim * 4);
      dpos.upload(&t, 4);
      int nh = n_head, qk_ = qk, vd = vdim, ms = max_seq;
      void* pk = dk1.data();
      void* pv = dv1.data();
      void* pK = dKc.data();
      void* pV = dVc.data();
      void* pp = dpos.data();
      void* args[] = {&pk, &pv, &pK, &pV, &pp, &nh, &qk_, &vd, &ms};
      cuda::launch(fn_w, n_head, 1, 1, 128, 1, 1, 0, args);
    }
    cuda::sync();

    // attention for query row 1 only
    const int start = 1;
    dpos.upload(&start, 4);
    {
      int nh = n_head, qk_ = qk, vd = vdim, ms = max_seq;
      void* pq = dq.data();
      void* pK = dKc.data();
      void* pV = dVc.data();
      void* po = dout.data();
      void* pp = dpos.data();
      void* args[] = {&pq, &pK, &pV, &po, &pp, &nh, &qk_, &vd, &ms};
      const size_t sh = static_cast<size_t>(max_seq) * sizeof(float);
      cuda::launch(fn_a, 1, n_head, 1, 32, 1, 1, sh, args);
    }
    cuda::sync();

    std::vector<float> got(static_cast<size_t>(n_head) * vdim);
    dout.download(got.data(), got.size() * 4);

    // host reference for query row 1 over keys 0..1
    std::vector<float> ref(got.size(), 0.f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(qk));
    for (int h = 0; h < n_head; ++h) {
      float scores[2];
      float mx = -1e30f;
      for (int j = 0; j < 2; ++j) {
        const float* qh = q.data() + (static_cast<size_t>(1) * n_head + h) * qk;
        const float* kh = k.data() + (static_cast<size_t>(j) * n_head + h) * qk;
        float dot = 0;
        for (int d = 0; d < qk; ++d) dot += qh[d] * kh[d];
        scores[j] = dot * scale;
        mx = std::max(mx, scores[j]);
      }
      float sum = 0;
      for (int j = 0; j < 2; ++j) {
        scores[j] = std::exp(scores[j] - mx);
        sum += scores[j];
      }
      for (int d = 0; d < vdim; ++d) {
        float a = 0;
        for (int j = 0; j < 2; ++j)
          a += scores[j] / sum * v[(static_cast<size_t>(j) * n_head + h) * vdim + d];
        ref[static_cast<size_t>(h) * vdim + d] = a;
      }
    }
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double e = got[i] - ref[i];
      num += e * e;
      den += static_cast<double>(ref[i]) * ref[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    std::printf("mla cache  : rel_l2=%.3e  %s\n", rel, rel < 1e-5 ? "PASS" : "FAIL");
    return rel < 1e-5 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
