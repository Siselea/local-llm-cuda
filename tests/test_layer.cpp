// M3 gate: one full transformer layer (RMSNorm -> QKV -> RoPE -> causal GQA ->
// O proj -> residual -> RMSNorm -> SwiGLU MLP -> residual) through the CUDA path,
// compared against tests/make_tiny_layer.py's numpy reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "engine/cublas.h"
#include "engine/cuda.h"
#include "engine/ir.h"
#include "engine/kernels.h"
#include "engine/st.h"

namespace {

std::string g_kernel_src;

void add_kernel(const char* src) {
  g_kernel_src += src;
  g_kernel_src += "\n";
}

std::vector<float> load_bf16(const st::Tensor& t) {
  std::ifstream f(t.file, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + t.file);
  f.seekg(static_cast<std::streamoff>(t.data_start + t.begin));
  const size_t n = static_cast<size_t>(t.nbytes() / 2);
  std::vector<uint16_t> raw(n);
  f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(n * 2));
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
    std::memcpy(&out[i], &bits, 4);
  }
  return out;
}

std::vector<float> read_f32_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<float> v(n / 4);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

struct Layer {
  int hidden, n_head, n_kv, head_dim, inter, L;

  cuda::Buffer x, hn, q, k, v, attn, o, gate, up, act, down;
  cuda::Buffer w_ln1, w_ln2, w_q, w_k, w_v, w_o, w_g, w_u, w_d;

  void* k_rms = nullptr;
  void* k_rope = nullptr;
  void* k_attn = nullptr;
  void* k_silu = nullptr;
  void* k_add = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "build/fixtures/tiny_layer";
  try {
    cuda::init();
    cublas::init();

    // ---- model metadata (validated IR) ----
    ir::Model ir_model = ir::build_ir_from_hf(dir);
    auto errs = ir::validate(ir_model);
    if (!errs.empty()) {
      std::fprintf(stderr, "IR validation failed:\n");
      for (auto& e : errs) std::fprintf(stderr, "  - %s\n", e.c_str());
      return 2;
    }
    const int hidden = ir_model.hidden;
    const int n_head = ir_model.attn.n_head;
    const int n_kv = ir_model.attn.n_head_kv;
    const int head_dim = ir_model.attn.key_length;
    const int inter = ir_model.ffn;
    const float rope_base = static_cast<float>(ir_model.attn.rope_freq_base);

    st::Shards shards = st::load_dir(dir);
    auto get = [&](const std::string& name) {
      auto it = shards.tensors.find(name);
      if (it == shards.tensors.end()) throw std::runtime_error("missing tensor " + name);
      return load_bf16(it->second);
    };

    const auto input = read_f32_file(dir + "/input.f32");
    const auto ref = read_f32_file(dir + "/ref_out.f32");
    const int L = static_cast<int>(input.size() / hidden);
    std::printf("layer      : hidden=%d heads=%d/%d D=%d inter=%d L=%d\n", hidden, n_head,
                n_kv, head_dim, inter, L);

    // ---- kernels ----
    add_kernel(kernels::kRmsNorm);
    add_kernel(kernels::kRope);
    add_kernel(kernels::kAttn);
    add_kernel(kernels::kSiluMul);
    add_kernel(kernels::kAdd);
    cuda::Module mod(cuda::compile_ptx(g_kernel_src, "layer_kernels.cu"));
    void* k_rms = mod.kernel("rmsnorm");
    void* k_rope = mod.kernel("rope");
    void* k_attn = mod.kernel("attn");
    void* k_silu = mod.kernel("silu_mul");
    void* k_add = mod.kernel("add_inplace");

    // ---- buffers ----
    const size_t hd = static_cast<size_t>(n_head) * head_dim;
    const size_t kd = static_cast<size_t>(n_kv) * head_dim;
    const size_t rows = static_cast<size_t>(L);

    auto upload = [&](const std::vector<float>& w) {
      cuda::Buffer b(w.size() * 4);
      b.upload(w.data(), w.size() * 4);
      return b;
    };

    cuda::Buffer x(rows * hidden * 4), hn(rows * hidden * 4), q(rows * hd * 4),
        k(rows * kd * 4), v(rows * kd * 4), attn(rows * hd * 4), o(rows * hidden * 4),
        gate(rows * inter * 4), up(rows * inter * 4), act(rows * inter * 4),
        down(rows * hidden * 4);
    x.upload(input.data(), input.size() * 4);

    cuda::Buffer w_ln1 = upload(get("model.layers.0.input_layernorm.weight"));
    cuda::Buffer w_ln2 = upload(get("model.layers.0.post_attention_layernorm.weight"));
    cuda::Buffer w_q = upload(get("model.layers.0.self_attn.q_proj.weight"));
    cuda::Buffer w_k = upload(get("model.layers.0.self_attn.k_proj.weight"));
    cuda::Buffer w_v = upload(get("model.layers.0.self_attn.v_proj.weight"));
    cuda::Buffer w_o = upload(get("model.layers.0.self_attn.o_proj.weight"));
    cuda::Buffer w_g = upload(get("model.layers.0.mlp.gate_proj.weight"));
    cuda::Buffer w_u = upload(get("model.layers.0.mlp.up_proj.weight"));
    cuda::Buffer w_d = upload(get("model.layers.0.mlp.down_proj.weight"));

    const float eps = 1e-5f;

    auto rms = [&](cuda::Buffer& in, cuda::Buffer& w, cuda::Buffer& out) {
      int n = hidden;
      float e = eps;
      void* px = in.data();
      void* pw = w.data();
      void* py = out.data();
      void* args[] = {&px, &pw, &py, &n, &e};
      cuda::launch(k_rms, L, 1, 1, 256, 1, 1, 256 * sizeof(float), args);
    };
    cuda::Buffer dpos(4);
    const int pos0 = 0;
    dpos.upload(&pos0, 4);
    std::vector<float> freqs(static_cast<size_t>(head_dim / 2));
    for (size_t i = 0; i < freqs.size(); ++i)
      freqs[i] = static_cast<float>(std::pow(static_cast<double>(rope_base),
                                             -2.0 * static_cast<double>(i) / head_dim));
    cuda::Buffer dfreq(static_cast<size_t>(head_dim / 2) * 4);
    dfreq.upload(freqs.data(), freqs.size() * 4);
    auto rope = [&](cuda::Buffer& buf, int nheads) {
      int nh = nheads, d = head_dim;
      float ms = 1.0f;
      void* pb = buf.data();
      void* pp = dpos.data();
      void* pf = dfreq.data();
      void* args[] = {&pb, &pp, &nh, &d, &pf, &ms};
      cuda::launch(k_rope, L, nheads, 1, static_cast<int>(head_dim / 2), 1, 1, 0, args);
    };
    auto add = [&](cuda::Buffer& a, cuda::Buffer& b, long long total) {
      void* pa = a.data();
      void* pb = b.data();
      void* args[] = {&pa, &pb, &total};
      const int threads = 256;
      const int blocks = static_cast<int>((total + threads - 1) / threads);
      cuda::launch(k_add, blocks, 1, 1, threads, 1, 1, 0, args);
    };

    // h = rmsnorm(x, ln1)
    rms(x, w_ln1, hn);
    auto stats = [&](const char* tag, cuda::Buffer& b, size_t n) {
      std::vector<float> t(n);
      b.download(t.data(), n * 4);
      double s = 0;
      for (float v : t) s += static_cast<double>(v) * v;
      std::printf("  %-8s norm=%.5f  first=%.5f\n", tag, std::sqrt(s), t.empty() ? 0.f : t[0]);
    };
    if (std::getenv("LAYER_DEBUG")) {
      stats("x", x, rows * hidden);
      stats("hn", hn, rows * hidden);
    }
    // q,k,v projections: y = h @ W^T
    cublas::gemm_f32_nt(L, static_cast<int>(hd), hidden, static_cast<const float*>(hn.data()),
                        static_cast<const float*>(w_q.data()), static_cast<float*>(q.data()));
    cublas::gemm_f32_nt(L, static_cast<int>(kd), hidden, static_cast<const float*>(hn.data()),
                        static_cast<const float*>(w_k.data()), static_cast<float*>(k.data()));
    cublas::gemm_f32_nt(L, static_cast<int>(kd), hidden, static_cast<const float*>(hn.data()),
                        static_cast<const float*>(w_v.data()), static_cast<float*>(v.data()));
    if (std::getenv("LAYER_DEBUG")) {
      stats("q_raw", q, rows * hd);
      stats("k_raw", k, rows * kd);
      stats("v_raw", v, rows * kd);
    }
    // rope
    rope(q, n_head);
    rope(k, n_kv);
    // causal attention
    {
      int l = L, nh = n_head, nkv = n_kv, d = head_dim;
      void* pq = q.data();
      void* pk = k.data();
      void* pv = v.data();
      void* po = attn.data();
      void* args[] = {&pq, &pk, &pv, &po, &l, &nh, &nkv, &d};
      const size_t shared = (static_cast<size_t>(L) + head_dim) * sizeof(float);
      cuda::launch(k_attn, L, n_head, 1, 32, 1, 1, shared, args);
    }
    if (std::getenv("LAYER_DEBUG")) stats("attn", attn, rows * hd);
    // o = attn @ Wo^T ; x += o
    cublas::gemm_f32_nt(L, hidden, static_cast<int>(hd), static_cast<const float*>(attn.data()),
                        static_cast<const float*>(w_o.data()), static_cast<float*>(o.data()));
    add(x, o, static_cast<long long>(rows) * hidden);

    // h2 = rmsnorm(x, ln2)
    rms(x, w_ln2, hn);
    cublas::gemm_f32_nt(L, inter, hidden, static_cast<const float*>(hn.data()),
                        static_cast<const float*>(w_g.data()), static_cast<float*>(gate.data()));
    cublas::gemm_f32_nt(L, inter, hidden, static_cast<const float*>(hn.data()),
                        static_cast<const float*>(w_u.data()), static_cast<float*>(up.data()));
    {
      long long total = static_cast<long long>(rows) * inter;
      void* pg = gate.data();
      void* pu = up.data();
      void* pa = act.data();
      void* args[] = {&pg, &pu, &pa, &total};
      const int threads = 256;
      cuda::launch(k_silu, static_cast<int>((total + threads - 1) / threads), 1, 1, threads, 1,
                   1, 0, args);
    }
    cublas::gemm_f32_nt(L, hidden, inter, static_cast<const float*>(act.data()),
                        static_cast<const float*>(w_d.data()),
                        static_cast<float*>(down.data()));
    add(x, down, static_cast<long long>(rows) * hidden);

    if (std::getenv("LAYER_DEBUG")) {
      stats("o", o, rows * hidden);
      stats("act", act, rows * inter);
      stats("down", down, rows * hidden);
      stats("y", x, rows * hidden);
    }
    cuda::sync();
    std::vector<float> y(input.size());
    x.download(y.data(), y.size() * 4);

    // ---- compare ----
    double num = 0, den = 0, max_abs = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      const double e = static_cast<double>(y[i]) - ref[i];
      num += e * e;
      den += static_cast<double>(ref[i]) * ref[i];
      max_abs = std::max(max_abs, std::fabs(e));
    }
    const double rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    std::printf("layer out  : max_abs=%.3e  rel_l2=%.3e  %s\n", max_abs, rel_l2,
                rel_l2 < 1e-3 ? "PASS" : "FAIL");
    return rel_l2 < 1e-3 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
