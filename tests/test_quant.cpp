// M3: Q4_K / Q5_K dequantization kernel vs gguf.quants reference (real GGUF blocks).
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

std::vector<unsigned char> read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const auto n = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<unsigned char> v(n);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
  return v;
}

std::vector<float> read_f32(const std::string& path) {
  const auto raw = read_bytes(path);
  std::vector<float> v(raw.size() / 4);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

struct Result {
  double rel_l2 = 0, max_abs = 0;
};

Result compare(const std::vector<float>& got, const std::vector<float>& ref) {
  Result r;
  double num = 0, den = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = static_cast<double>(got[i]) - ref[i];
    num += e * e;
    den += static_cast<double>(ref[i]) * ref[i];
    r.max_abs = std::max(r.max_abs, std::fabs(e));
  }
  r.rel_l2 = std::sqrt(num / std::max(den, 1e-30));
  return r;
}

int run_one(const std::string& dir, const char* kernel_name, int block_bytes,
            long long block_elems, int threads, const std::string& common,
            const std::string& kernel_src) {
  const auto blocks = read_bytes(dir + "/blocks.bin");
  const auto ref = read_f32(dir + "/ref.f32");
  const long long n_blocks = static_cast<long long>(blocks.size() / block_bytes);
  if (static_cast<size_t>(n_blocks) * static_cast<size_t>(block_elems) != ref.size())
    throw std::runtime_error(dir + ": ref size mismatch");

  cuda::Module mod(cuda::compile_ptx(common + "\n" + kernel_src, kernel_name));
  void* fn = mod.kernel(kernel_name);

  cuda::Buffer db(blocks.size());
  db.upload(blocks.data(), blocks.size());
  cuda::Buffer out(ref.size() * 4);

  long long nb = n_blocks;
  void* pb = db.data();
  void* po = out.data();
  void* args[] = {&pb, &po, &nb};
  cuda::launch(fn, static_cast<int>(n_blocks), 1, 1, threads, 1, 1, 0, args);
  cuda::sync();

  std::vector<float> got(ref.size());
  out.download(got.data(), got.size() * 4);
  const Result r = compare(got, ref);
  std::printf("%-10s blocks=%lld elems=%zu  max_abs=%.3e rel_l2=%.3e %s\n", kernel_name,
              n_blocks, ref.size(), r.max_abs, r.rel_l2, r.rel_l2 < 1e-6 ? "PASS" : "FAIL");
  return r.rel_l2 < 1e-6 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : "build/fixtures";
  try {
    cuda::init();
    int rc = 0;
    rc |= run_one(root + "/q4k", "dequant_q4k", 144, 256, 256, kernels::kQuantCommon,
                  kernels::kDequantQ4K);
    rc |= run_one(root + "/q5k", "dequant_q5k", 176, 256, 256, kernels::kQuantCommon,
                  kernels::kDequantQ5K);
    rc |= run_one(root + "/q6k", "dequant_q6k", 210, 256, 32, kernels::kQuantCommon,
                  kernels::kDequantQ6K);
    rc |= run_one(root + "/q8k", "dequant_q8_0", 34, 32, 32, kernels::kQuantCommon,
                  kernels::kDequantQ8_0);
    rc |= run_one(root + "/q5_0", "dequant_q5_0", 22, 32, 32, kernels::kQuantCommon,
                  kernels::kDequantQ5_0);
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 2;
  }
}
