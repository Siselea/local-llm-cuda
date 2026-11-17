// M3: KV cache path — prefill writes L positions, decode reads all of them.
// Compares both attention outputs against the numpy reference from make_tiny_layer.py.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "engine/cuda.h"
#include "engine/kernels.h"

namespace {

std::vector<float> read_f32(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<float> v(n / 4);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = static_cast<double>(got[i]) - ref[i];
    num += e * e;
    den += static_cast<double>(ref[i]) * ref[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "build/fixtures/tiny_layer";
  try {
    cuda::init();
    const int hidden = 64, n_head = 4, n_kv = 2, D = 16, L = 6, max_seq = 16;

    const auto q = read_f32(dir + "/q.f32");
    const auto k = read_f32(dir + "/k.f32");
    const auto v = read_f32(dir + "/v.f32");
    const auto ref_prefill = read_f32(dir + "/attn_ref.f32");
    const auto q7 = read_f32(dir + "/q7.f32");
    const auto k7 = read_f32(dir + "/k7.f32");
    const auto v7 = read_f32(dir + "/v7.f32");
    const auto ref_decode = read_f32(dir + "/attn7_ref.f32");

    auto src = std::string(kernels::kKvWrite) + "\n" + kernels::kAttnCached;
    cuda::Module mod(cuda::compile_ptx(src, "kv.cu"));
    void* fn_write = mod.kernel("kv_write");
    void* fn_attn = mod.kernel("attn_cached");

    cuda::Buffer dpos(4);
    const size_t cache_elems = static_cast<size_t>(n_kv) * max_seq * D;
    cuda::Buffer Kc(cache_elems * 4), Vc(cache_elems * 4);
    cuda::Buffer dk(k.size() * 4), dv(v.size() * 4), dq(q.size() * 4), dout(q.size() * 4);
    dk.upload(k.data(), k.size() * 4);
    dv.upload(v.data(), v.size() * 4);
    dq.upload(q.data(), q.size() * 4);

    // prefill
    {
      const int start = 0;
      dpos.upload(&start, 4);
      int nkv = n_kv, d = D, ms = max_seq;
      void* pk = dk.data();
      void* pv = dv.data();
      void* pK = Kc.data();
      void* pV = Vc.data();
      void* pp = dpos.data();
      void* args[] = {&pk, &pv, &pK, &pV, &pp, &nkv, &d, &ms};
      cuda::launch(fn_write, L * n_kv, 1, 1, 32, 1, 1, 0, args);
    }
    {
      int nh = n_head, nkv = n_kv, d = D, ms = max_seq;
      void* pq = dq.data();
      void* pK = Kc.data();
      void* pV = Vc.data();
      void* po = dout.data();
      void* pp = dpos.data();
      void* args[] = {&pq, &pK, &pV, &po, &pp, &nh, &nkv, &d, &ms};
      const size_t sh = (static_cast<size_t>(max_seq) + D) * sizeof(float);
      cuda::launch(fn_attn, L, n_head, 1, 32, 1, 1, sh, args);
    }
    cuda::sync();
    std::vector<float> got(q.size());
    dout.download(got.data(), got.size() * 4);
    const double r1 = rel_l2(got, ref_prefill);
    std::printf("prefill    : L=%d  rel_l2=%.3e  %s\n", L, r1, r1 < 1e-5 ? "PASS" : "FAIL");

    // decode one more token (position 6) reading the whole cache
    cuda::Buffer dk7(k7.size() * 4), dv7(v7.size() * 4), dq7(q7.size() * 4),
        dout7(q7.size() * 4);
    dk7.upload(k7.data(), k7.size() * 4);
    dv7.upload(v7.data(), v7.size() * 4);
    dq7.upload(q7.data(), q7.size() * 4);
    {
      const int start = L;
      dpos.upload(&start, 4);
      int nkv = n_kv, d = D, ms = max_seq;
      void* pk = dk7.data();
      void* pv = dv7.data();
      void* pK = Kc.data();
      void* pV = Vc.data();
      void* pp = dpos.data();
      void* args[] = {&pk, &pv, &pK, &pV, &pp, &nkv, &d, &ms};
      cuda::launch(fn_write, n_kv, 1, 1, 32, 1, 1, 0, args);
    }
    {
      int nh = n_head, nkv = n_kv, d = D, ms = max_seq;
      void* pq = dq7.data();
      void* pK = Kc.data();
      void* pV = Vc.data();
      void* po = dout7.data();
      void* pp = dpos.data();
      void* args[] = {&pq, &pK, &pV, &po, &pp, &nh, &nkv, &d, &ms};
      const size_t sh = (static_cast<size_t>(max_seq) + D) * sizeof(float);
      cuda::launch(fn_attn, 1, n_head, 1, 32, 1, 1, sh, args);
    }
    cuda::sync();
    std::vector<float> got7(q7.size());
    dout7.download(got7.data(), got7.size() * 4);
    const double r2 = rel_l2(got7, ref_decode);
    std::printf("decode     : pos=%d reads %d kv  rel_l2=%.3e  %s\n", L, L + 1, r2,
                r2 < 1e-5 ? "PASS" : "FAIL");

    (void)hidden;
    return (r1 < 1e-5 && r2 < 1e-5) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
