// Malformed-input bounds tests for the model loaders (GGUF header/metadata/tensor table
// and safetensors header).  A downloaded model file is untrusted input: every count,
// dimension, offset and alignment read from it must be validated before it is used for
// an allocation, a division or a read.
//
// These tests WRITE synthetic malformed files (no real model needed) and require
// gguf::read() / st::read_file() to throw std::runtime_error -- never crash, never
// allocate gigabytes, never accept a tensor that cannot exist in the file.
//
// Covered review items: #8 (alignment -> division by zero), #9 (dimension/size
// multiplication overflow), #14 (negative/fractional safetensors dimensions and
// offsets), #15 (safetensors size/offset overflow), #18 (untrusted metadata counts),
// #19 (nested metadata arrays recursion depth).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "engine/gguf.h"
#include "engine/ir.h"
#include "engine/st.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* what) {
  ++g_checks;
  if (ok) return;
  ++g_failures;
  std::fprintf(stderr, "FAIL: %s\n", what);
}

// ---------------------------------------------------------------- GGUF writer
struct GgufWriter {
  std::string buf;
  uint64_t n_tensors = 0;
  uint64_t n_kv = 0;

  void bytes(const void* p, size_t n) { buf.append(static_cast<const char*>(p), n); }
  void u8(uint8_t v) { bytes(&v, 1); }
  void u32(uint32_t v) { bytes(&v, 4); }
  void u64(uint64_t v) { bytes(&v, 8); }
  void str(const std::string& s) {
    u64(s.size());
    bytes(s.data(), s.size());
  }
  void kv_key(const std::string& k) { str(k); }
  void kv_u32(const std::string& k, uint32_t v) {
    kv_key(k);
    u32(4);  // gguf ValueType::UINT32
    u32(v);
    ++n_kv;
  }
  void kv_u64(const std::string& k, uint64_t v) {
    kv_key(k);
    u32(10);  // UINT64
    u64(v);
    ++n_kv;
  }
  void kv_str(const std::string& k, const std::string& v) {
    kv_key(k);
    u32(8);  // STRING
    str(v);
    ++n_kv;
  }
  // Nested ARRAY-of-ARRAY metadata: depth levels of [ [ [ ... 1 ] ] ].
  // depth nested arrays: the kv value type is ARRAY; each array element is an ARRAY of
  // the next level; the innermost is an ARRAY of one UINT32.
  void kv_nested_arrays(const std::string& k, int depth) {
    kv_key(k);
    u32(9);  // kv value type ARRAY
    for (int d = 1; d < depth; ++d) {
      u32(9);  // array of ARRAY
      u64(1);  // one element
    }
    u32(4);  // innermost array of UINT32
    u64(1);
    u32(1);
    ++n_kv;
  }
  void kv_array_u32(const std::string& k, const std::vector<uint32_t>& vals) {
    kv_key(k);
    u32(9);  // ARRAY
    u32(7);  // UINT32 elements
    u64(vals.size());
    for (uint32_t v : vals) u32(v);
    ++n_kv;
  }
  void tensor(const std::string& name, const std::vector<uint64_t>& dims, uint32_t type_id,
              uint64_t offset) {
    str(name);
    u32(static_cast<uint32_t>(dims.size()));
    for (uint64_t d : dims) u64(d);
    u32(type_id);
    u64(offset);
    ++n_tensors;
  }
  // Finish: header + payload of `data` bytes, aligned to `align`.
  std::string finish(uint32_t align, size_t data_bytes = 0) {
    std::string out;
    out.append("GGUF", 4);
    auto put = [&](const void* p, size_t n) { out.append(static_cast<const char*>(p), n); };
    const uint32_t version = 3;
    put(&version, 4);
    put(&n_tensors, 8);
    put(&n_kv, 8);
    out += buf;
    // pad to alignment like a genuine file
    const uint64_t a = align ? align : 1;
    const uint64_t pos = out.size();
    const uint64_t padded = (pos + a - 1) / a * a;
    out.append(static_cast<size_t>(padded - pos), '\0');
    out.append(data_bytes, '\0');
    return out;
  }
};

std::string write_file(const std::string& dir, const std::string& name,
                       const std::string& bytes) {
  const std::string path = dir + "/" + name;
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  f.close();
  return path;
}

bool rejects(const std::string& path, const char* what) {
  try {
    gguf::read(path);
  } catch (const std::exception&) {
    return true;
  }
  std::fprintf(stderr, "FAIL: accepted malformed GGUF: %s\n", what);
  return false;
}

void test_gguf_alignment(const std::string& dir) {
  // #8: an alignment whose value does not fit uint32 truncates to 0 and the data
  // offset computation divides by zero (SIGFPE / 0xC0000094) instead of erroring.
  {
    GgufWriter w;
    w.kv_u64("general.alignment", 1ull << 32);
    check(rejects(write_file(dir, "align_2_32.gguf", w.finish(32)), "alignment 2^32"), "align 2^32 rejected");
  }
  {
    GgufWriter w;
    w.kv_u64("general.alignment", 1ull << 40);
    check(rejects(write_file(dir, "align_2_40.gguf", w.finish(32)), "alignment 2^40"),
          "align 2^40 rejected");
  }
  {
    GgufWriter w;
    w.kv_u64("general.alignment", 0);
    check(rejects(write_file(dir, "align_0.gguf", w.finish(32)), "alignment 0"),
          "align 0 rejected");
  }
  {
    GgufWriter w;
    w.kv_u32("general.alignment", 3);  // not a power of two
    check(rejects(write_file(dir, "align_3.gguf", w.finish(32)), "alignment 3"),
          "non-power-of-two alignment rejected");
  }
  // A legal alignment must still be accepted (no over-rejection).
  {
    GgufWriter w;
    w.kv_u32("general.alignment", 64);
    w.tensor("t", {32, 1}, 0 /* F32 */, 0);
    try {
      const auto m = gguf::read(write_file(dir, "align_ok.gguf", w.finish(64, 32 * 4)));
      check(m.alignment == 64, "valid alignment 64 accepted");
      check(m.tensors.size() == 1, "valid tensor table accepted");
    } catch (const std::exception& e) {
      std::fprintf(stderr, "FAIL: valid GGUF rejected: %s\n", e.what());
      ++g_failures;
    }
  }
}

void test_gguf_dimension_overflow(const std::string& dir) {
  // #9: ne[0]*ne[1]*...*type_size must not wrap around.
  {
    GgufWriter w;
    w.tensor("huge", {1ull << 32, 1ull << 32}, 0 /* F32 */, 0);
    check(rejects(write_file(dir, "dims_overflow.gguf", w.finish(32)), "dims overflow"),
          "dimension product overflow rejected");
  }
  {
    GgufWriter w;
    w.tensor("bytes_overflow", {1ull << 40, 1ull << 24}, 0 /* F32 */, 0);
    check(rejects(write_file(dir, "nbytes_overflow.gguf", w.finish(32)), "nbytes overflow"),
          "nbytes overflow rejected");
  }
  {
    // A tensor whose element count does not divide by the block size.
    GgufWriter w;
    w.tensor("bad_block", {100}, 12 /* Q4_K, block 256 */, 0);
    check(rejects(write_file(dir, "bad_block.gguf", w.finish(32)), "block divisibility"),
          "non-divisible block count rejected");
  }
  {
    // A tensor that points past the end of the file (offset/nbytes vs file size).
    GgufWriter w;
    w.tensor("oob", {1024}, 0 /* F32, 4096 bytes */, 1ull << 20);
    check(rejects(write_file(dir, "offset_oob.gguf", w.finish(32, 16)), "offset out of file"),
          "tensor past end of file rejected");
  }
  {
    // Offset+nbytes arithmetic overflow.
    GgufWriter w;
    w.tensor("wrap", {1024}, 0, ~0ull - 128);
    check(rejects(write_file(dir, "offset_wrap.gguf", w.finish(32, 16)), "offset wrap"),
          "offset+nbytes overflow rejected");
  }
}

void test_gguf_counts(const std::string& dir) {
  // #18: counts from the header must be bounded by what the file can possibly hold
  // before they are used for reserve()/loops.
  for (uint64_t n : {1ull << 20, 1ull << 40, ~0ull}) {
    std::string out = "GGUF";
    const uint32_t version = 3;
    out.append(reinterpret_cast<const char*>(&version), 4);
    out.append(reinterpret_cast<const char*>(&n), 8);  // n_tensors
    const uint64_t zero = 0;
    out.append(reinterpret_cast<const char*>(&zero), 8);  // n_kv
    char name[64];
    std::snprintf(name, sizeof(name), "n_tensors_%llu.gguf",
                  static_cast<unsigned long long>(n));
    check(rejects(write_file(dir, name, out), "absurd n_tensors"),
          "absurd n_tensors rejected");
  }
}

void test_gguf_nested_arrays(const std::string& dir) {
  // #19: nested metadata arrays must not recurse without a depth limit (stack
  // overflow).  Depth 16 is fine; 4096 must be rejected.
  {
    GgufWriter w;
    w.kv_nested_arrays("deep", 8);
    try {
      gguf::read(write_file(dir, "arrays_depth8.gguf", w.finish(32)));
      check(true, "moderately nested array accepted");
    } catch (const std::exception& e) {
      std::fprintf(stderr, "FAIL: depth-8 array rejected: %s\n", e.what());
      ++g_failures;
    }
  }
  {
    GgufWriter w;
    w.kv_nested_arrays("deep", 4096);
    check(rejects(write_file(dir, "arrays_depth4096.gguf", w.finish(32)),
                  "4096-deep arrays"),
          "4096-deep nested array rejected");
  }
}

// ------------------------------------------------------------ safetensors
std::string st_file(const std::string& dir, const std::string& name,
                    const std::string& header, size_t data_bytes) {
  std::string out;
  const uint64_t hlen = header.size();
  out.append(reinterpret_cast<const char*>(&hlen), 8);
  out += header;
  out.append(data_bytes, '\0');
  return write_file(dir, name, out);
}

bool st_rejects(const std::string& path) {
  try {
    st::read_file(path);
  } catch (const std::exception&) {
    return true;
  }
  std::fprintf(stderr, "FAIL: accepted malformed safetensors: %s\n", path.c_str());
  return false;
}

void test_safetensors(const std::string& dir) {
  // #14: negative and fractional dimensions / offsets must be rejected, not cast to
  // uint64 (a negative -1 becomes 0xFFFF... and a fractional 1.5 silently truncates).
  check(st_rejects(st_file(dir, "st_neg_dim.st",
                            R"({"a":{"dtype":"F32","shape":[2,-1],"data_offsets":[0,8]}})", 8)),
        "negative shape dimension rejected");
  check(st_rejects(st_file(dir, "st_frac_dim.st",
                            R"({"a":{"dtype":"F32","shape":[2.5],"data_offsets":[0,8]}})", 8)),
        "fractional shape dimension rejected");
  check(st_rejects(st_file(dir, "st_neg_off.st",
                            R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[-8,0]}})", 8)),
        "negative data offset rejected");
  check(st_rejects(st_file(dir, "st_frac_off.st",
                            R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4.5]}})", 8)),
        "fractional data offset rejected");
  // #15: size arithmetic overflow and end < begin.
  check(st_rejects(st_file(dir, "st_overflow.st",
                            R"({"a":{"dtype":"F32","shape":[4294967296,4294967296],
                            "data_offsets":[0,8]}})", 8)),
        "shape product overflow rejected");
  check(st_rejects(st_file(dir, "st_end_lt_begin.st",
                            R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})", 8)),
        "end < begin rejected");
  check(st_rejects(st_file(dir, "st_oob.st",
                            R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,18446744073709551000]}})", 8)),
        "offset past end of file rejected");
  // A valid safetensors header must still be accepted.
  {
    const std::string ok = R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    const auto path = st_file(dir, "st_ok.st", ok, 8);
    try {
      const auto ts = st::read_file(path);
      check(ts.count("a") == 1 && ts.at("a").n_elements() == 2, "valid safetensors accepted");
    } catch (const std::exception& e) {
      std::fprintf(stderr, "FAIL: valid safetensors rejected: %s\n", e.what());
      ++g_failures;
    }
  }
}

// -------------------------------------------------- #13 IR geometry validation
gguf::Value kv_str(const std::string& v) {
  gguf::Value x;
  x.kind = gguf::Value::Kind::STRING;
  x.s = v;
  return x;
}
gguf::Value kv_int(int64_t v) {
  gguf::Value x;
  x.kind = gguf::Value::Kind::INT;
  x.i = v;
  return x;
}

// A minimal llama-style metadata set: n_layer=1, hidden=64, one attention layer.
gguf::Model base_llama_model() {
  gguf::Model m;
  m.path = "synthetic";
  m.kv["general.architecture"] = kv_str("llama");
  m.kv["llama.block_count"] = kv_int(1);
  m.kv["llama.embedding_length"] = kv_int(64);
  m.kv["llama.feed_forward_length"] = kv_int(128);
  m.kv["llama.context_length"] = kv_int(128);
  m.kv["llama.attention.head_count"] = kv_int(4);
  m.kv["llama.attention.head_count_kv"] = kv_int(2);
  m.kv["llama.attention.key_length"] = kv_int(16);
  m.kv["llama.attention.value_length"] = kv_int(16);
  return m;
}

bool ir_throws(const gguf::Model& m, const char* want) {
  try {
    ir::build_from_gguf(m);
  } catch (const std::exception& e) {
    if (std::strstr(e.what(), want) != nullptr) return true;
    std::fprintf(stderr, "FAIL: IR rejected with unexpected message (%s): %s\n", want,
                 e.what());
    return false;
  }
  std::fprintf(stderr, "FAIL: IR accepted bad geometry (%s)\n", want);
  return false;
}

void test_ir_geometry() {
  // #13: GQA head geometry must be integral (n_head % n_head_kv == 0).
  {
    auto m = base_llama_model();
    m.kv["llama.attention.head_count_kv"] = kv_int(3);  // 4 % 3 != 0
    check(ir_throws(m, "not divisible by attention.head_count_kv"), "GQA head divisibility");
  }
  {
    auto m = base_llama_model();
    m.kv["llama.attention.head_count_kv"] = kv_int(8);  // more kv heads than q heads
    check(ir_throws(m, "exceeds attention.head_count"), "GQA kv heads > q heads");
  }
  // #13: gated-delta-net geometry (inner_size divisible by time_step_rank, tsr > 0).
  {
    gguf::Model m;
    m.path = "synthetic";
    m.kv["general.architecture"] = kv_str("qwen35");
    m.kv["qwen35.block_count"] = kv_int(1);
    m.kv["qwen35.embedding_length"] = kv_int(64);
    m.kv["qwen35.feed_forward_length"] = kv_int(128);
    m.kv["qwen35.context_length"] = kv_int(128);
    m.kv["qwen35.attention.head_count"] = kv_int(4);
    m.kv["qwen35.attention.head_count_kv"] = kv_int(2);
    m.kv["qwen35.ssm.conv_kernel"] = kv_int(4);
    m.kv["qwen35.ssm.inner_size"] = kv_int(10);
    m.kv["qwen35.ssm.state_size"] = kv_int(5);
    m.kv["qwen35.ssm.group_count"] = kv_int(2);
    m.kv["qwen35.ssm.time_step_rank"] = kv_int(3);  // 10 % 3 != 0
    check(ir_throws(m, "not divisible by ssm.time_step_rank"), "GDN inner % tsr divisibility");
    m.kv["qwen35.ssm.time_step_rank"] = kv_int(0);
    check(ir_throws(m, "ssm.time_step_rank must be positive"), "GDN tsr > 0");
  }
}

}  // namespace

int main() {
  // Write the malformed files under build/ (the temp dir can be restricted in the
  // sandbox): a fixed relative location keeps the test self-contained.
  std::error_code ec;
  const auto dir = std::filesystem::path("build") / "loader_bounds_tmp";
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    std::fprintf(stderr, "cannot create %s: %s\n", dir.string().c_str(),
                 ec.message().c_str());
    return 2;
  }
  test_gguf_alignment(dir.string());
  test_gguf_dimension_overflow(dir.string());
  test_gguf_counts(dir.string());
  test_gguf_nested_arrays(dir.string());
  test_safetensors(dir.string());
  test_ir_geometry();
  if (g_failures != 0) {
    std::fprintf(stderr, "test_loader_bounds: %d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("test_loader_bounds: %d checks PASS (malformed GGUF/safetensors rejected)\n",
              g_checks);
  return 0;
}
