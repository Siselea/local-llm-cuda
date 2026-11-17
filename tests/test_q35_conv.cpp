// Unit test: the qwen35 depthwise conv must give identical results whether the
// sequence is processed as one chunk or token by token (state carried).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "engine/cuda.h"
#include "engine/kernels.h"

namespace {

std::vector<float> run_conv(const std::vector<float>& x, int T, int C, int K,
                            const std::vector<float>& w, bool reset_state,
                            cuda::Buffer& state, void* kconv, void* kstate) {
  cuda::Buffer dx(static_cast<size_t>(T) * C * 4), dy(static_cast<size_t>(T) * C * 4),
      dw(static_cast<size_t>(C) * K * 4);
  dx.upload(x.data(), x.size() * 4);
  dw.upload(w.data(), w.size() * 4);
  if (reset_state) {
    std::vector<float> z((K - 1) * C, 0.f);
    state.upload(z.data(), z.size() * 4);
  }
  int T_ = T, C_ = C, K_ = K;
  void* px = dx.data();
  void* pw = dw.data();
  void* ps = state.data();
  void* py = dy.data();
  void* args[] = {&px, &pw, &ps, &py, &T_, &C_, &K_};
  cuda::launch(kconv, T, 1, 1, 256, 1, 1, 0, args);
  void* args2[] = {&px, &ps, &T_, &C_, &K_};
  cuda::launch(kstate, 64, 1, 1, 256, 1, 1, 0, args2);
  std::vector<float> out(static_cast<size_t>(T) * C);
  dy.download(out.data(), out.size() * 4);
  return out;
}

}  // namespace

int main() {
  cuda::init();
  const std::string src(std::string(kernels::kGdConv1d) + kernels::kGdConvState);
  cuda::Module mod(cuda::compile_ptx(src, "q35conv.cu"));
  void* kconv = mod.kernel("gd_conv1d_silu");
  void* kstate = mod.kernel("gd_conv_state");

  const int C = 64, K = 4, T = 3, total = 7;
  std::mt19937 rng(1234);
  std::normal_distribution<float> gauss(0.f, 1.f);
  std::vector<float> w(C * K);
  for (auto& v : w) v = gauss(rng);
  std::vector<float> x(total * C);
  for (auto& v : x) v = gauss(rng);

  cuda::Buffer st1((K - 1) * C * 4), st2((K - 1) * C * 4);
  // reference: one chunk over every prefix, re-run from scratch
  std::vector<std::vector<float>> chunks;
  for (int t = 1; t <= total; ++t) {
    std::vector<float> prefix(x.begin(), x.begin() + static_cast<size_t>(t) * C);
    chunks.push_back(run_conv(prefix, t, C, K, w, true, st1, kconv, kstate));
  }
  // incremental: one token at a time with carried state
  std::vector<std::vector<float>> steps;
  for (int t = 0; t < total; ++t) {
    std::vector<float> xi(x.begin() + t * C, x.begin() + (t + 1) * C);
    steps.push_back(run_conv(xi, 1, C, K, w, t == 0, st2, kconv, kstate));
  }

  double worst = 0.0;
  int worst_t = -1;
  for (int t = 0; t < total; ++t) {
    double num = 0.0, den = 0.0;
    for (int c = 0; c < C; ++c) {
      const double a = chunks[t][(t)*C + c];
      const double b = steps[t][c];
      num += (a - b) * (a - b);
      den += a * a;
    }
    const double rel = std::sqrt(num / den);
    if (rel > worst) { worst = rel; worst_t = t; }
  }
  for (int c = 0; c < 3; ++c) {
    std::printf("c=%d step(t=3)=%.6f chunk(t=3)[3]=%.6f\n", c, steps[3][c], chunks[3][3 * C + c]);
  }
  // manual expectation for the last token of the 7-length sequence processed stepwise
  {
    int t = total - 1;
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const int j = t - (K - 1) + k;
      acc += w[k] * (j >= 0 ? x[j * C + 0] : 0.0);
    }
    std::printf("manual(t=%d,c=0)=%.6f step=%.6f chunk=%.6f\n", t,
                acc / (1.0 + std::exp(-acc)), steps[t][0], chunks[t][t * C + 0]);
  }
  // inspect the carried state after the incremental walk
  {
    std::vector<float> st((K - 1) * C);
    st2.download(st.data(), st.size() * 4);
    std::printf("state[age][c=0]: ");
    for (int a = 0; a < K - 1; ++a) std::printf("%.4f ", st[a * C + 0]);
    std::printf(" expected x[t-1..t-3]=%.4f %.4f %.4f\n", x[(total - 1) * C + 0],
                x[(total - 2) * C + 0], x[(total - 3) * C + 0]);
  }
  // micro-test of the state shift: state [1,2,3] + new input 9 -> [9,1,2]
  {
    const int Cm = 8;
    cuda::Buffer st((K - 1) * Cm * 4), dxm(Cm * 4);
    std::vector<float> init{1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 1.f, 2.f,
                            3.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 1.f,
                            2.f, 3.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f};
    std::vector<float> xin(Cm, 9.f);
    st.upload(init.data(), init.size() * 4);
    dxm.upload(xin.data(), xin.size() * 4);
    int T_ = 1, C_ = Cm, K_ = K;
    void* pxm = dxm.data();
    void* psm = st.data();
    void* argsm[] = {&pxm, &psm, &T_, &C_, &K_};
    cuda::launch(kstate, 64, 1, 1, 256, 1, 1, 0, argsm);
    std::vector<float> got((K - 1) * Cm);
    st.download(got.data(), got.size() * 4);
    std::printf("state shift: got [");
    for (int a = 0; a < K - 1; ++a) std::printf("%.1f ", got[a * Cm]);
    std::printf("] expected [9.0 1.0 2.0]\n");
  }
  const bool ok = worst < 1e-6;
  std::printf("q35 conv chunked vs incremental: worst rel_l2=%.3e at t=%d  %s\n", worst,
              worst_t, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
