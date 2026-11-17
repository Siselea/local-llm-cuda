// M3: fused dequant+matvec (decode path) vs dequantized reference, on real GGUF rows.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "engine/cuda.h"
#include "engine/json_parse.h"
#include "engine/kernels.h"

namespace {

std::vector<unsigned char> read_bytes(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<unsigned char> v(n);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

std::vector<float> read_f32(const std::string& p) {
  const auto raw = read_bytes(p);
  std::vector<float> v(raw.size() / 4);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

struct Case {
  std::string dir, kernel, common, src;
  int threads;
};

int run(const Case& c) {
  const auto meta = jsonp::parse(std::string(
      reinterpret_cast<const char*>(read_bytes(c.dir + "/meta.json").data()),
      read_bytes(c.dir + "/meta.json").size()));
  const int n_out = static_cast<int>(meta.get_num("n_out", 0));
  const int n_in = static_cast<int>(meta.get_num("n_in", 0));
  const int block_bytes = static_cast<int>(meta.get_num("block_bytes", 0));
  const int block_elems = static_cast<int>(meta.get_num("block_elems", 0));
  const std::string qtype = meta.get_str("qtype", "?");

  const auto blocks = read_bytes(c.dir + "/blocks.bin");
  const auto x = read_f32(c.dir + "/x.f32");
  const auto ref = read_f32(c.dir + "/y.f32");
  if (static_cast<int>(x.size()) != n_in || static_cast<int>(ref.size()) != n_out)
    throw std::runtime_error(c.dir + ": fixture shape mismatch");
  if (static_cast<long long>(blocks.size()) !=
      static_cast<long long>(n_out) * (n_in / block_elems) * block_bytes)
    throw std::runtime_error(c.dir + ": blocks size mismatch");

  cuda::Module mod(cuda::compile_ptx(c.common + "\n" + c.src, c.kernel + ".cu"));
  void* fn = mod.kernel(c.kernel);

  cuda::Buffer dW(blocks.size()), dx(x.size() * 4), dy(ref.size() * 4);
  dW.upload(blocks.data(), blocks.size());
  dx.upload(x.data(), x.size() * 4);

  int nin = n_in, nout = n_out;
  void* pw = dW.data();
  void* px = dx.data();
  void* py = dy.data();
  void* args[] = {&pw, &px, &py, &nin, &nout};
  const int grid = (n_out + 7) / 8;
  const size_t shmem = 0;
  cuda::launch(fn, grid, 1, 1, c.threads, 1, 1, shmem, args);
  cuda::sync();

  std::vector<float> got(ref.size());
  dy.download(got.data(), got.size() * 4);
  double num = 0, den = 0, max_abs = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = static_cast<double>(got[i]) - ref[i];
    num += e * e;
    den += static_cast<double>(ref[i]) * ref[i];
    max_abs = std::max(max_abs, std::fabs(e));
  }
  const double rel_l2 = std::sqrt(num / std::max(den, 1e-30));

  const int iters = 50;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) cuda::launch(fn, grid, 1, 1, c.threads, 1, 1, shmem, args);
  cuda::sync();
  auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
  const double gbs = static_cast<double>(blocks.size()) / sec / 1e9;

  std::printf("%-14s %s  %dx%d  max_abs=%.3e rel_l2=%.3e  %.2f ms  %.1f GB/s  %s\n",
              c.kernel.c_str(), qtype.c_str(), n_out, n_in, max_abs, rel_l2, sec * 1e3, gbs,
              rel_l2 < 1e-5 ? "PASS" : "FAIL");
  return rel_l2 < 1e-5 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : "build/fixtures";
  try {
    cuda::init();
    int rc = 0;
    rc |= run({root + "/gemv_q4k", "gemv_q4k_x4", kernels::kQuantCommon, kernels::kGemvQ4K_x4, 64});
    rc |= run({root + "/gemv_q5k", "gemv_q5k_x4", kernels::kQuantCommon, kernels::kGemvQ5K_x4, 64});
    rc |= run({root + "/gemv_q8_0", "gemv_q8_0", kernels::kQuantCommon, kernels::kGemvQ8_0, 256});
    std::ifstream probe(root + "/gemv_q4k_big/meta.json");
    if (probe.good())
      rc |= run({root + "/gemv_q4k_big", "gemv_q4k_x4", kernels::kQuantCommon, kernels::kGemvQ4K_x4, 64});
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
