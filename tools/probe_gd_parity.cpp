// Numerical parity probe for the three gated-delta-rule kernels.
//   gd_delta_rule_128        reference (state in global memory)
//   gd_delta_rule_128_smem   state staged in shared memory
//   gd_delta_rule_128_reg    state kept in registers
// All three must be bit-identical for both kh_mode values (GGUF block repeat vs HF
// repeat_interleave) and for chunked calls that carry the recurrent state.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "engine/cuda.h"
#include "engine/kernels.h"

namespace {

constexpr int D = 128;

struct Ctx {
  int T = 0, n_vh = 0, n_kh = 0;
  std::vector<float> q, k, v, g, beta;
};

Ctx make_ctx(int T, int n_vh, int n_kh, unsigned seed) {
  Ctx c;
  c.T = T;
  c.n_vh = n_vh;
  c.n_kh = n_kh;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::uniform_real_distribution<float> gd(-0.6f, 0.6f);
  const size_t nqk = static_cast<size_t>(T) * n_kh * D;
  const size_t nv = static_cast<size_t>(T) * n_vh * D;
  c.q.resize(nqk);
  c.k.resize(nqk);
  c.v.resize(nv);
  for (auto& x : c.q) x = u(rng);
  for (auto& x : c.k) x = u(rng);
  for (auto& x : c.v) x = u(rng);
  c.g.resize(static_cast<size_t>(T) * n_vh);
  for (auto& x : c.g) x = -std::fabs(gd(rng));   // decay factor exp(g) in (0,1]
  c.beta.resize(static_cast<size_t>(T) * n_vh);
  for (auto& x : c.beta) x = 0.1f + 0.8f * (u(rng) * 0.5f + 0.5f);
  return c;
}

struct Run {
  std::vector<float> S, o;
};

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cuda::init();
  const std::string src = std::string(kernels::kGdDelta) + kernels::kGdDelta128 +
                          kernels::kGdDelta128Smem + kernels::kGdDelta128Reg;
  cuda::Module mod(cuda::compile_ptx(src, "gd_parity.cu"));
  void* k_ref = mod.kernel("gd_delta_rule_128");
  void* k_smem = mod.kernel("gd_delta_rule_128_smem");
  void* k_reg = mod.kernel("gd_delta_rule_128_reg");
  const size_t shbytes = cuda::max_dynamic_shared_mem(k_smem);
  std::printf("smem opt-in bytes = %zu (need %d)\n", shbytes, (D * D + 4 * D + 4 * D) * 4);

  int failures = 0;
  for (int kh_mode = 0; kh_mode < 2; ++kh_mode) {
    for (int n_vh : {8, 32}) {
      const int n_kh = n_vh / 2;
      for (int T : {1, 7, 33}) {
        const Ctx c = make_ctx(T, n_vh, n_kh, 1234u + T + kh_mode);
        cuda::Buffer dq(c.q.size() * 4), dk(c.k.size() * 4), dv(c.v.size() * 4),
            dg(c.g.size() * 4), db(c.beta.size() * 4);
        dq.upload(c.q.data(), c.q.size() * 4);
        dk.upload(c.k.data(), c.k.size() * 4);
        dv.upload(c.v.data(), c.v.size() * 4);
        dg.upload(c.g.data(), c.g.size() * 4);
        db.upload(c.beta.data(), c.beta.size() * 4);
        const size_t sn = static_cast<size_t>(n_vh) * D * D, on = static_cast<size_t>(T) * n_vh * D;
        std::vector<Run> runs(3);
        void* fns[3] = {k_ref, k_smem, k_reg};
        for (int r = 0; r < 3; ++r) {
          cuda::Buffer S(sn * 4), O(on * 4);
          std::vector<float> zeros(sn, 0.f);
          S.upload(zeros.data(), sn * 4);   // the state must start zeroed
          void* pq = dq.data();
          void* pk = dk.data();
          void* pv = dv.data();
          void* pg = dg.data();
          void* pb = db.data();
          void* pS = S.data();
          void* po = O.data();
          int t = T, nvh = n_vh, nkh = n_kh, kd = D, vd = D, khm = kh_mode;
          void* args[] = {&pq, &pk, &pv, &pg, &pb, &pS, &po, &t, &nvh, &nkh, &kd, &vd, &khm};
          cuda::launch(fns[r], n_vh, 1, 1, 512, 1, 1, r == 1 ? 69120 : 0, args);
          cuda::sync();
          runs[r].S.resize(sn);
          runs[r].o.resize(on);
          S.download(runs[r].S.data(), sn * 4);
          O.download(runs[r].o.data(), on * 4);
        }
        for (int r = 1; r < 3; ++r) {
          const char* nm = r == 1 ? "smem" : "reg";
          size_t ds = 0, do_ = 0;
          for (size_t i = 0; i < runs[0].S.size(); ++i)
            if (std::memcmp(&runs[0].S[i], &runs[r].S[i], 4) != 0) ++ds;
          for (size_t i = 0; i < runs[0].o.size(); ++i)
            if (std::memcmp(&runs[0].o[i], &runs[r].o[i], 4) != 0) ++do_;
          std::printf("kh_mode=%d n_vh=%2d T=%2d  %-4s vs ref: %zu/%zu state diffs, %zu/%zu "
                      "out diffs  %s\n",
                      kh_mode, n_vh, T, nm, ds, runs[0].S.size(), do_, runs[0].o.size(),
                      (ds == 0 && do_ == 0) ? "BIT-EXACT" : "MISMATCH");
          if (ds || do_)
            std::printf("    dbg ref S0=%.6g o=%.6g | %s S0=%.6g o=%.6g\n", runs[0].S[0],
                        runs[0].o[0], nm, runs[r].S[0], runs[r].o[0]);
          if (ds || do_) ++failures;
        }
      }
    }
  }
  // chunked call: state carried between two calls must match one long call
  {
    const int n_vh = 32, n_kh = 16, T = 11, split = 5;
    const Ctx c = make_ctx(T, n_vh, n_kh, 777);
    const size_t sn = static_cast<size_t>(n_vh) * D * D, on = static_cast<size_t>(T) * n_vh * D;
    cuda::Buffer dq(c.q.size() * 4), dk(c.k.size() * 4), dv(c.v.size() * 4),
        dg(c.g.size() * 4), db(c.beta.size() * 4);
    dq.upload(c.q.data(), c.q.size() * 4);
    dk.upload(c.k.data(), c.k.size() * 4);
    dv.upload(c.v.data(), c.v.size() * 4);
    dg.upload(c.g.data(), c.g.size() * 4);
    db.upload(c.beta.data(), c.beta.size() * 4);
    for (int kh_mode = 0; kh_mode < 2; ++kh_mode) {
      std::vector<float> ref_o(on), ch_o(on);
      std::vector<float> ref_S(sn), ch_S(sn);
      {   // one call
        cuda::Buffer S(sn * 4), O(on * 4);
        S.upload(std::vector<float>(sn, 0.f).data(), sn * 4);
        void* pq = dq.data(); void* pk = dk.data(); void* pv = dv.data();
        void* pg = dg.data(); void* pb = db.data();
        void* pS = S.data(); void* po = O.data();
        int t = T, nvh = n_vh, nkh = n_kh, kd = D, vd = D, khm = kh_mode;
        void* args[] = {&pq, &pk, &pv, &pg, &pb, &pS, &po, &t, &nvh, &nkh, &kd, &vd, &khm};
        cuda::launch(k_reg, n_vh, 1, 1, 512, 1, 1, 0, args);
        cuda::sync();
        S.download(ref_S.data(), sn * 4);
        O.download(ref_o.data(), on * 4);
      }
      {   // two chunks: first `split` tokens then the rest, same state buffer
        cuda::Buffer S(sn * 4), O(on * 4);
        S.upload(std::vector<float>(sn, 0.f).data(), sn * 4);
        for (int part = 0; part < 2; ++part) {
          const int t0 = part == 0 ? 0 : split;
          const int len = part == 0 ? split : T - split;
          // offset the pointers of q/k/v/g/beta by t0 rows
          void* pq = static_cast<float*>(dq.data()) + static_cast<size_t>(t0) * n_kh * D;
          void* pk = static_cast<float*>(dk.data()) + static_cast<size_t>(t0) * n_kh * D;
          void* pv = static_cast<float*>(dv.data()) + static_cast<size_t>(t0) * n_vh * D;
          void* pg = static_cast<float*>(dg.data()) + static_cast<size_t>(t0) * n_vh;
          void* pb = static_cast<float*>(db.data()) + static_cast<size_t>(t0) * n_vh;
          void* pS = S.data();
          void* po = static_cast<float*>(O.data()) + static_cast<size_t>(t0) * n_vh * D;
          int t = len, nvh = n_vh, nkh = n_kh, kd = D, vd = D, khm = kh_mode;
          void* args[] = {&pq, &pk, &pv, &pg, &pb, &pS, &po, &t, &nvh, &nkh, &kd, &vd, &khm};
          cuda::launch(k_reg, n_vh, 1, 1, 512, 1, 1, 0, args);
          cuda::sync();
        }
        S.download(ch_S.data(), sn * 4);
        O.download(ch_o.data(), on * 4);
      }
      size_t d1 = 0, d2 = 0;
      for (size_t i = 0; i < ch_S.size(); ++i)
        if (std::memcmp(&ref_S[i], &ch_S[i], 4) != 0) ++d1;
      for (size_t i = 0; i < ch_o.size(); ++i)
        if (std::memcmp(&ref_o[i], &ch_o[i], 4) != 0) ++d2;
      std::printf("chunked(5+6) kh_mode=%d: %zu state diffs, %zu out diffs  %s\n", kh_mode, d1,
                  d2, (d1 == 0 && d2 == 0) ? "BIT-EXACT" : "MISMATCH");
      if (d1 || d2) ++failures;
    }
  }
  std::printf("%s\n", failures ? "GD PARITY: FAIL" : "GD PARITY: PASS");
  return failures ? 1 : 0;
}
