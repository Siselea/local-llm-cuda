// Dumps cpu::dequant_block output for the first few blocks of a GGUF tensor so it
// can be compared against gguf.quants in python.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "engine/cpu_matvec.h"
#include "engine/gguf.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.gguf> <tensor> [n_blocks]\n", argv[0]);
    return 2;
  }
  try {
    auto m = gguf::read(argv[1]);
    const gguf::TensorInfo* ti = nullptr;
    for (const auto& t : m.tensors)
      if (t.name == argv[2]) ti = &t;
    if (!ti) {
      std::fprintf(stderr, "tensor not found\n");
      return 1;
    }
    const int nb = argc > 3 ? std::atoi(argv[3]) : 2;
    std::ifstream f(m.path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(m.data_offset + ti->offset));
    std::printf("type=%s nbytes=%lld\n", ti->type_name.c_str(),
                static_cast<long long>(ti->nbytes));
    std::printf("raw:");
    for (int i = 0; i < 96; ++i) {
      unsigned char c = 0;
      f.read(reinterpret_cast<char*>(&c), 1);
      std::printf("%02x", c);
    }
    std::printf("\n");
    f.seekg(static_cast<std::streamoff>(m.data_offset + ti->offset));
    std::vector<unsigned char> raw(static_cast<size_t>(nb) * 512 + 64);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    int bb = 0, be = 0;
    if (ti->type_name == "Q5_1") { bb = 24; be = 32; }
    else if (ti->type_name == "Q5_0") { bb = 22; be = 32; }
    else if (ti->type_name == "Q8_0") { bb = 34; be = 32; }
    else if (ti->type_name == "Q6_K") { bb = 210; be = 256; }
    else if (ti->type_name == "Q5_K") { bb = 176; be = 256; }
    else if (ti->type_name == "Q4_K") { bb = 144; be = 256; }
    else { std::fprintf(stderr, "unsupported type\n"); return 1; }
    if (std::getenv("MATVEC")) {
      // full-tensor matvec: y = W x, x deterministic
      const int n_out = argc > 5 ? std::atoi(argv[4]) : static_cast<int>(ti->shape[0]);
      const int n_in = argc > 5 ? std::atoi(argv[5]) : static_cast<int>(ti->shape[1]);
      const long long off = argc > 6 ? std::atoll(argv[6]) : 0;
      int bb2 = 0, be2 = 0;
      if (ti->type_name == "Q5_1") { bb2 = 24; be2 = 32; }
      else if (ti->type_name == "Q5_0") { bb2 = 22; be2 = 32; }
      else if (ti->type_name == "Q8_0") { bb2 = 34; be2 = 32; }
      else if (ti->type_name == "Q6_K") { bb2 = 210; be2 = 256; }
      else if (ti->type_name == "Q5_K") { bb2 = 176; be2 = 256; }
      else if (ti->type_name == "Q4_K") { bb2 = 144; be2 = 256; }
      else { std::fprintf(stderr, "unsupported\n"); return 1; }
      const size_t need = static_cast<size_t>(n_out) * n_in / be2 * bb2;
      std::vector<unsigned char> all(need);
      f.seekg(static_cast<std::streamoff>(m.data_offset + ti->offset + off));
      f.read(reinterpret_cast<char*>(all.data()), static_cast<std::streamsize>(all.size()));
      std::vector<float> x(static_cast<size_t>(n_in));
      for (int i = 0; i < n_in; ++i) x[i] = std::sin(0.01f * static_cast<float>(i + 1));
      std::vector<float> y(static_cast<size_t>(n_out));
      cpu::matvec(ti->type_name, all.data(), n_out, n_in, x.data(), y.data());
      std::printf("matvec %d x %d:", n_out, n_in);
      for (int i = 0; i < 4; ++i) std::printf(" %.7g", y[i]);
      std::printf("\n");
      std::ofstream o("build/cpu_matvec_out.f32", std::ios::binary);
      o.write(reinterpret_cast<const char*>(y.data()),
              static_cast<std::streamsize>(y.size() * 4));
      return 0;
    }
    std::vector<float> out(static_cast<size_t>(be));
    std::vector<float> all_out;
    for (int b = 0; b < nb; ++b) {
      cpu::dequant_block(ti->type_name, raw.data() + static_cast<size_t>(b) * bb, out.data());
      all_out.insert(all_out.end(), out.begin(), out.end());
    }
    {
      std::ofstream o("build/cpu_block.f32", std::ios::binary);
      o.write(reinterpret_cast<const char*>(all_out.data()),
              static_cast<std::streamsize>(all_out.size() * 4));
      std::printf("wrote %zu floats\n", all_out.size());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
