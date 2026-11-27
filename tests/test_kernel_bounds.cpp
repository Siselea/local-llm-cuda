// Regression tests for the v0.1.7 CUDA kernel correctness audit (v0.1.8 fixes):
//
//   #1 kv_split is capped at 32 (attn_combine's shared weight array had w[33] while
//      the config allowed up to 64) and attn_combine merges 8/16/32 splits correctly.
//   #2 device routing falls back to the host router for n_expert > 256.
//   #3 attn_cached_cfg's dynamic shared memory above 48 KiB works after the opt-in.
//
// The config cases run in child processes (the Config singleton reads the environment
// once), the kernel cases run on synthetic tensors; no model weights are needed.
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/config.h"
#include "engine/cuda.h"
#include "engine/kernels.h"
#include "engine/runner.h"

// Geometry helpers defined in src/runner.cpp (extern linkage on purpose).
namespace runner {
int topk_partial_blocks(int vocab);
int moe_slots_per_launch(int topk);
bool dp4a_nin_ok(int n_in);
}  // namespace runner

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
  std::printf("  %-64s %s%s%s\n", what.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : ": ",
              detail.c_str());
  if (!ok) ++g_fail;
}

std::string exe_path() {
  char buf[4096];
  const DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
  if (n == 0 || n >= sizeof(buf)) return "test_kernel_bounds.exe";
  return std::string(buf, n);
}

// Spawn this executable with one environment variable set (env_name may be null).
int spawn_self(const char* env_name, const char* env_val, const std::string& args) {
  if (env_name) _putenv_s(env_name, env_val ? env_val : "");
  const std::string cmd = "\"" + exe_path() + "\" " + args + " >NUL 2>&1";
  const int rc = std::system(cmd.c_str());
  if (env_name) _putenv_s(env_name, "");
  return rc;
}

int spawn_tool(const char* env1, const char* val1, const char* env2, const char* val2,
               const std::string& cmd) {
  if (env1) _putenv_s(env1, val1 ? val1 : "");
  if (env2) _putenv_s(env2, val2 ? val2 : "");
  const int rc = std::system(cmd.c_str());
  if (env1) _putenv_s(env1, "");
  if (env2) _putenv_s(env2, "");
  return rc;
}

std::string read_text(const std::string& path) {
  std::string out;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return out;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

bool file_exists(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fclose(f);
  return true;
}

// ---------------------------------------------------------------- #1 config ----
int child_kvsplit_reject(int value) {
  try {
    const int v = engine::config::Config::get().get_int("kv_split");
    std::printf("child: kv_split=%d (expected an out-of-range rejection)\n", v);
    return 2;
  } catch (const std::exception& e) {
    std::printf("child: rejected: %s\n", e.what());
    return 0;
  }
}

int child_kvsplit_eq(int want) {
  const int v = engine::config::Config::get().get_int("kv_split");
  std::printf("child: kv_split=%d (want %d)\n", v, want);
  return v == want ? 0 : 3;
}

// ------------------------------------------------------ attn_combine (#1) --------
bool test_attn_combine() {
  std::printf("== attn_combine merge correctness (nsplit 8/16/32) ==\n");
  bool all_ok = true;
  cuda::Module mod(cuda::compile_ptx(std::string(kernels::kAttnCombine), "combine_test"));
  void* fn = mod.kernel("attn_combine");
  const int D = 64;
  const int stride = D + 2;
  for (int nsplit : {8, 16, 32}) {
    std::vector<float> hscr(static_cast<size_t>(nsplit) * stride, 0.f);
    for (int sp = 0; sp < nsplit; ++sp) {
      float* base = hscr.data() + static_cast<size_t>(sp) * stride;
      base[D] = -0.5f + 0.25f * static_cast<float>(sp);      // m
      base[D + 1] = 0.5f + 0.1f * static_cast<float>(sp);    // s
      for (int d = 0; d < D; ++d)
        base[d] = std::sin(0.37f * static_cast<float>((d + 1) * (sp + 1)));
    }
    // one empty split: the kernel must ignore it (s == 0)
    {
      float* base = hscr.data() + static_cast<size_t>(nsplit - 1) * stride;
      base[D] = -1e30f;
      base[D + 1] = 0.f;
    }
    cuda::Buffer scr(static_cast<size_t>(nsplit) * stride * 4), out(static_cast<size_t>(D) * 4);
    scr.upload(hscr.data(), hscr.size() * 4);
    int n_head = 1, dd = D, ns = nsplit, st = stride;
    void* args[] = {&scr, &out, &n_head, &dd, &ns, &st};
    cuda::launch(fn, 1, 1, 1, D, 1, 1, 0, args);
    cuda::sync();
    std::vector<float> got(static_cast<size_t>(D));
    out.download(got.data(), got.size() * 4);
    // host reference: softmax-weighted merge of the split partials
    double mx = -1e30;
    for (int sp = 0; sp < nsplit; ++sp) {
      const float* b = hscr.data() + static_cast<size_t>(sp) * stride;
      if (b[D + 1] > 0.f && b[D] > mx) mx = b[D];
    }
    double tot = 0.0;
    std::vector<double> num(static_cast<size_t>(D), 0.0);
    for (int sp = 0; sp < nsplit; ++sp) {
      const float* b = hscr.data() + static_cast<size_t>(sp) * stride;
      const double w = (b[D + 1] > 0.f) ? static_cast<double>(b[D + 1]) * std::exp(b[D] - mx) : 0.0;
      tot += w;
      for (int d = 0; d < D; ++d) num[static_cast<size_t>(d)] += w * b[d];
    }
    double max_err = 0.0;
    for (int d = 0; d < D; ++d) {
      const double ref = tot > 0 ? num[static_cast<size_t>(d)] / tot : 0.0;
      const double err = std::fabs(ref - static_cast<double>(got[static_cast<size_t>(d)]));
      max_err = std::max(max_err, err);
    }
    const bool ok = max_err < 1e-5;
    all_ok = all_ok && ok;
    char detail[96];
    std::snprintf(detail, sizeof(detail), "max|err|=%.2e", max_err);
    check(ok, "nsplit=" + std::to_string(nsplit) + " matches host merge", detail);
  }
  return all_ok;
}

// ------------------------------------------------ attn_cached_cfg smem (#3) ----
bool test_attn_cfg_smem() {
  std::printf("== attn_cached_cfg large dynamic shared memory ==\n");
  cuda::Module mod(cuda::compile_ptx(std::string(kernels::kAttnCfg), "cfg_smem_test"));
  void* fn = mod.kernel("attn_cached_cfg");
  const int D = 64;
  const int max_seq = 1;
  cuda::Buffer q(static_cast<size_t>(D) * 4), kc(static_cast<size_t>(D) * 4),
      vc(static_cast<size_t>(D) * 4), out(static_cast<size_t>(D) * 4), posbuf(4);
  int pos = 0;
  posbuf.upload(&pos, 4);
  int n_head = 1, n_kv = 1, dd = D, ms = max_seq;
  float scale = 0.125f;
  int window = 0;
  void* args[] = {&q, &kc, &vc, &out, &posbuf, &n_head, &n_kv, &dd, &ms, &scale, &window};
  const size_t cap = cuda::max_dynamic_shared_mem(fn);
  char detail[96];
  std::snprintf(detail, sizeof(detail), "device opt-in cap=%zu bytes", cap);
  check(cap >= 48u * 1024u, "max_dynamic_shared_mem reports the opt-in window", detail);
  bool ok48 = true;
  try {
    cuda::launch(fn, 1, 1, 1, 32, 1, 1, 48u * 1024u, args);
    cuda::sync();
  } catch (const std::exception& e) {
    ok48 = false;
    std::printf("    default-window launch failed: %s\n", e.what());
  }
  check(ok48, "48 KiB dynamic shared memory launch succeeds");
  if (cap >= 60000u) {
    bool ok60 = true;
    try {
      cuda::launch(fn, 1, 1, 1, 32, 1, 1, 60000u, args);
      cuda::sync();
    } catch (const std::exception& e) {
      ok60 = false;
      std::printf("    large-window launch failed after the opt-in: %s\n", e.what());
    }
    check(ok60, "60 KB dynamic shared memory launch succeeds after opt-in");
  } else {
    std::printf("  (device opt-in window < 60 KB: large-window case skipped)\n");
  }
  return ok48;
}

// -------------------------------------------------- device router (#2) ----------
void test_device_route_guard() {
  std::printf("== device-route expert-count guard ==\n");
  check(runner::device_route_supported(256), "n_expert=256 uses the device router");
  check(runner::device_route_supported(64), "n_expert=64 uses the device router");
  check(!runner::device_route_supported(257), "n_expert=257 falls back to the host router");
  check(!runner::device_route_supported(300), "n_expert=300 falls back to the host router");
  check(!runner::device_route_supported(384), "n_expert=384 falls back to the host router");
  check(!runner::device_route_supported(0), "n_expert=0 is not routed by the device");
}

// -------------------------------------- 300-expert synthetic model smoke (#2) ----
void test_expert_overflow_fixture() {
  std::printf("== 300-expert synthetic model (host-router fallback) ==\n");
  const std::string fixture = "build/fixtures/tiny_gguf_moe300/model.gguf";
  if (!file_exists(fixture) || !file_exists("build/test_model.exe")) {
    std::printf("  (fixture/build/test_model.exe missing: skipped; tools/verify.sh builds it)\n");
    return;
  }
  // v0.1.7/v0.1.8 contract: expert_count > 256 is rejected at IR validation time
  // (src/ir.cpp), so the moe_route no-op can never be reached with a real model today.
  // This test pins that clear rejection; the device_route_supported() checks above pin
  // the host-router fallback that would take over if the limit is ever relaxed.
  const std::string log = "build\\fixtures\\tiny_gguf_moe300\\load.log";
  for (const char* env : {"0", "300"}) {
    // cmd.exe only accepts backslashes for the executable and the redirect target.
    const int rc = spawn_tool("ENGINE_GPU_EXPERTS", env, nullptr, nullptr,
                              "build\\test_model.exe build/fixtures/tiny_gguf_moe300 > \"" + log +
                                  "\" 2>&1");
    const std::string text = read_text(log);
    const bool mentions = text.find("expert_count 300 > 256") != std::string::npos &&
                          text.find("not supported") != std::string::npos;
    check(rc != 0, std::string("300-expert model rejected at load (GPU_EXPERTS=") + env + ")");
    check(mentions, "rejection message names the 256-expert device-router limit");
  }
  check(!runner::device_route_supported(300),
        "if the IR limit is relaxed, n_expert=300 takes the host router");
}

// D4: the device top-k partial kernel keeps a 512-word (16384-index) selection bitmap
// per block; the host block count must keep every slice inside it.
void test_topk_bitmap_bound() {
  std::printf("== #D4 top-k partial bitmap bound ==\n");
  const int vocabs[] = {128, 1024, 248320, 1024 * 1024, 1 << 26};
  for (int v : vocabs) {
    const int nblk = runner::topk_partial_blocks(v);
    const int chunk = nblk > 0 ? (v + nblk - 1) / nblk : -1;
    check(nblk > 0 && chunk > 0 && chunk <= 16384,
          "vocab=" + std::to_string(v) + " chunks fit the 16384-index bitmap",
          "nblk=" + std::to_string(nblk) + " chunk=" + std::to_string(chunk));
  }
  check(runner::topk_partial_blocks(1 << 27) < 0,
        "vocab=2^27 is refused instead of overflowing the bitmap");
}

// D8: dp4a kernels require a 32-multiple activation length.
void test_dp4a_guard() {
  std::printf("== #D8 dp4a n_in guard ==\n");
  check(runner::dp4a_nin_ok(4096), "n_in=4096 accepted");
  check(runner::dp4a_nin_ok(32), "n_in=32 accepted");
  check(!runner::dp4a_nin_ok(33), "n_in=33 refused");
  check(!runner::dp4a_nin_ok(0), "n_in=0 refused");
}

// D13: MoE expert kernels use grid.y as the (token,expert) slot axis (limit 65535).
void test_moe_slot_chunking() {
  std::printf("== #D13 MoE slot chunking ==\n");
  const int topks[] = {1, 2, 4, 6, 8, 16};
  for (int topk : topks) {
    const int slots = runner::moe_slots_per_launch(topk);
    check(slots >= 1 && static_cast<long long>(slots) * topk <= 65535,
          "topk=" + std::to_string(topk) + " chunk fits grid.y",
          "slots=" + std::to_string(slots));
  }
  // 35B topk=8, a 16384-token prefill batch: 131072 slots need 17 launches, not one.
  const int slots = runner::moe_slots_per_launch(8);
  const int launches = (16384 * 8 + slots - 1) / slots;
  check(launches > 1 && launches <= 32, "16384-token prefill is split into bounded launches",
        "launches=" + std::to_string(launches));
}

// D9: an all -inf score row (numerical overflow) must produce zeros, not NaN.
bool test_softmax_all_inf() {
  std::printf("== #D9 all -inf softmax row ==\n");
  cuda::Module mod(cuda::compile_ptx(std::string(kernels::kMlaAttnCached), "softmax_inf_test"));
  void* fn = mod.kernel("mla_attn_cached");
  const int D = 4, max_seq = 2;
  std::vector<float> hq(static_cast<size_t>(D), 3e38f);
  std::vector<float> hk(static_cast<size_t>(max_seq) * D, -3e38f);
  std::vector<float> hv(static_cast<size_t>(max_seq) * D, 0.f);
  cuda::Buffer q(static_cast<size_t>(D) * 4), kc(static_cast<size_t>(max_seq) * D * 4),
      vc(static_cast<size_t>(max_seq) * D * 4), out(static_cast<size_t>(D) * 4), posbuf(4);
  q.upload(hq.data(), hq.size() * 4);
  kc.upload(hk.data(), hk.size() * 4);
  vc.upload(hv.data(), hv.size() * 4);
  int pos = 0;
  posbuf.upload(&pos, 4);
  int n_head = 1, qk = D, vdim = D, ms = max_seq;
  void* args[] = {&q, &kc, &vc, &out, &posbuf, &n_head, &qk, &vdim, &ms};
  bool ok = true;
  try {
    cuda::launch(fn, 1, 1, 1, 32, 1, 1, static_cast<size_t>(max_seq) * 4, args);
    cuda::sync();
  } catch (const std::exception& e) {
    ok = false;
    std::printf("    launch failed: %s\n", e.what());
  }
  std::vector<float> got(static_cast<size_t>(D), -1.f);
  out.download(got.data(), got.size() * 4);
  for (float v : got) ok = ok && std::isfinite(v) && v == 0.f;
  check(ok, "all -inf row yields finite zeros (no NaN/inf)");
  return ok;
}

// D11: uploading a buffer while a CUDA graph is being recorded is a capture-unsafe
// operation; with ENGINE_CAPTURE_STRICT it must throw (default keeps the old skip+warn).
int child_capture_probe(bool expect_throw) {
  cuda::init();
  void* stream = cuda::create_stream();
  cuda::set_stream(stream);
  cuda::Buffer b(64);
  float host[4] = {1, 2, 3, 4};
  cuda::Graph g;
  g.capture_begin();
  bool threw = false;
  try {
    b.upload(host, sizeof(host));
  } catch (const std::exception&) {
    threw = true;
  }
  g.capture_abort();
  if (threw != expect_throw) {
    std::printf("child: upload threw=%d expected=%d\n", static_cast<int>(threw),
                static_cast<int>(expect_throw));
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3) {
    const std::string mode = argv[1];
    const int value = std::atoi(argv[2]);
    if (mode == "--child-kvsplit-reject") return child_kvsplit_reject(value);
    if (mode == "--child-kvsplit-eq") return child_kvsplit_eq(value);
    if (mode == "--child-capture-strict") return child_capture_probe(true);
    if (mode == "--child-capture-lenient") return child_capture_probe(false);
  }

  std::printf("== kv_split config bounds (#1) ==\n");
  // In-process default (ENGINE_KV_SPLIT unset) is 32.
  {
    const char* env = std::getenv("ENGINE_KV_SPLIT");
    if (!env || !*env) {
      const int v = engine::config::Config::get().get_int("kv_split");
      check(v == 32, "default kv_split is 32", "got " + std::to_string(v));
    } else {
      std::printf("  (ENGINE_KV_SPLIT=%s in the environment: default case skipped)\n", env);
    }
  }
  // The config layer rejects values above the new 32 maximum (used to be accepted up
  // to 64, where attn_combine's shared w[33] overflowed).
  check(spawn_self("ENGINE_KV_SPLIT", "34", "--child-kvsplit-reject 34") == 0,
        "kv_split=34 is rejected by the config layer");
  check(spawn_self("ENGINE_KV_SPLIT", "64", "--child-kvsplit-reject 64") == 0,
        "kv_split=64 is rejected by the config layer");
  check(spawn_self("ENGINE_KV_SPLIT", "8", "--child-kvsplit-eq 8") == 0,
        "kv_split=8 is accepted");
  check(spawn_self("ENGINE_KV_SPLIT", "16", "--child-kvsplit-eq 16") == 0,
        "kv_split=16 is accepted");
  check(spawn_self("ENGINE_KV_SPLIT", "32", "--child-kvsplit-eq 32") == 0,
        "kv_split=32 is accepted");

  cuda::init();
  void* stream = cuda::create_stream();
  cuda::set_stream(stream);
  test_device_route_guard();
  test_topk_bitmap_bound();
  test_dp4a_guard();
  test_moe_slot_chunking();
  if (!test_softmax_all_inf()) ++g_fail;
  check(spawn_self("ENGINE_CAPTURE_STRICT", "1", "--child-capture-strict") == 0,
        "capture-strict: upload during graph capture throws");
  check(spawn_self(nullptr, nullptr, "--child-capture-lenient") == 0,
        "default: upload during graph capture is skipped (unchanged contract)");
  if (!test_attn_combine()) ++g_fail;
  if (!test_attn_cfg_smem()) ++g_fail;
  test_expert_overflow_fixture();

  std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "KERNEL BOUNDS: GREEN" : "KERNEL BOUNDS: FAILED",
              g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
