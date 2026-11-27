// Strictness regression for the config system (external review v0.1.7 items #6-#7).
//
//  #6 float options accepted NaN:  strtod("nan") succeeds, every comparison with
//     NaN is false so the range check passed, and the following (int64_t) cast of
//     a non-finite double is undefined behaviour (observed payload: INT64_MIN).
//     Fixed with std::isfinite() before the range check.
//
//  #7 recursive config includes ("config = other.cfg" inside a config file) had no
//     cycle/depth guard.  a.cfg -> b.cfg -> a.cfg recursed until the process ran
//     out of file handles ("cannot open config file", misleading) and on platforms
//     with a larger handle limit it would exhaust the native stack.  Fixed with a
//     normalised include chain (cycle error with the chain) and a depth cap of 8.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "engine/config.h"

namespace {

int fails = 0;
int checks = 0;

void check(const char* name, bool ok) {
  ++checks;
  if (!ok) {
    std::printf("  FAIL %s\n", name);
    ++fails;
  }
}

using engine::config::Config;
using engine::config::Source;

bool set_ok(const char* key, const char* value) {
  std::string err;
  return Config::get().set(key, value, Source::File, &err);
}

bool contains(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

bool write_file(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << text;
  return static_cast<bool>(f);
}

void cleanup(const std::string& path) { std::remove(path.c_str()); }

// All paths live in a unique temp directory so the test does not depend on the
// working directory (build.sh / tools/check_clean_build.sh run it from the repo
// root, but a developer may run build/test_config_strict.exe from anywhere).
std::string g_tmp_dir;

std::string build_path(const std::string& name) {
  return (std::filesystem::path(g_tmp_dir) / name).string();
}

void check_float_options() {
  std::printf("-- float options must be finite\n");
  const char* keys[] = {"default_temperature", "default_top_p", "default_repeat_penalty"};
  for (const char* key : keys) {
    const std::string label = std::string(key) + "=";
    check((label + "0.6 accepted").c_str(), set_ok(key, "0.6"));
    check((label + "0.6 value").c_str(), Config::get().get_double(key) == 0.6);
    check((label + "nan rejected").c_str(), !set_ok(key, "nan"));
    check((label + "NaN rejected").c_str(), !set_ok(key, "NaN"));
    check((label + "-nan rejected").c_str(), !set_ok(key, "-nan"));
    check((label + "inf rejected").c_str(), !set_ok(key, "inf"));
    check((label + "-inf rejected").c_str(), !set_ok(key, "-inf"));
    check((label + "Infinity rejected").c_str(), !set_ok(key, "Infinity"));
    check((label + "1e999 rejected").c_str(), !set_ok(key, "1e999"));
    check((label + "garbage rejected").c_str(), !set_ok(key, "abc"));
    check((label + "trailing junk rejected").c_str(), !set_ok(key, "0.5x"));
    check((label + "range out rejected").c_str(), !set_ok(key, "1e6"));
  }
  // A rejected value must not corrupt the previous good value.
  check("rejected NaN keeps previous value",
        set_ok("default_temperature", "0.7") && !set_ok("default_temperature", "nan") &&
            Config::get().get_double("default_temperature") == 0.7);
  // get_int() on a float option must stay a plain truncation of the finite value.
  check("float get_int truncates", set_ok("default_temperature", "0.75") &&
                                       Config::get().get_int("default_temperature") == 0);
}

void check_int_and_bool_validation() {
  std::printf("-- int/bool options reject non-numbers\n");
  check("ctx=128 accepted", set_ok("ctx", "128") && Config::get().get_int("ctx") == 128);
  check("ctx=nan rejected", !set_ok("ctx", "nan"));
  check("ctx=1e3 rejected", !set_ok("ctx", "1e3"));
  check("ctx=12x rejected", !set_ok("ctx", "12x"));
  check("ctx huuuge rejected", !set_ok("ctx", "999999999999999999999"));
  check("bool yes accepted", set_ok("no_graph", "yes") && Config::get().get_bool("no_graph"));
  check("bool nan rejected", !set_ok("no_graph", "nan"));
}

void check_include_chain() {
  std::printf("-- recursive config includes: valid chains work, cycles/depth fail\n");
  Config& cfg = Config::get();

  // A valid single include: build/cfg_strict_inc_a.cfg names b.cfg which sets ctx.
  const std::string a = build_path("cfg_strict_inc_a.cfg");
  const std::string b = build_path("cfg_strict_inc_b.cfg");
  check("write valid include a", write_file(a, "config = " + b + "\n"));
  check("write valid include b", write_file(b, "ctx = 321\n"));
  {
    std::string err;
    const bool ok = cfg.load_file(a, &err);
    check("valid include chain loads", ok);
    if (!ok) std::printf("    err=%s\n", err.c_str());
    check("valid include applied", cfg.get_int("ctx") == 321);
  }
  cleanup(a);
  cleanup(b);

  // Self include: config = itself.
  const std::string self = build_path("cfg_strict_self.cfg");
  check("write self include", write_file(self, "config = " + self + "\n"));
  {
    std::string err;
    check("self include rejected", !cfg.load_file(self, &err));
    check("self include error mentions cycle", contains(err, "cycle"));
  }
  cleanup(self);

  // Two-file cycle.
  const std::string ca = build_path("cfg_strict_cyc_a.cfg");
  const std::string cb = build_path("cfg_strict_cyc_b.cfg");
  check("write cycle a", write_file(ca, "config = " + cb + "\n"));
  check("write cycle b", write_file(cb, "config = " + ca + "\n"));
  {
    std::string err;
    check("two-file cycle rejected", !cfg.load_file(ca, &err));
    check("cycle error mentions cycle", contains(err, "cycle"));
    check("cycle error shows chain", contains(err, "cfg_strict_cyc_a.cfg") &&
                                          contains(err, "cfg_strict_cyc_b.cfg"));
  }
  cleanup(ca);
  cleanup(cb);

  // Same file through a differently-spelled relative path must still be a cycle.
  const std::string ra = build_path("cfg_strict_rel_a.cfg");
  const std::string rb = build_path("cfg_strict_rel_b.cfg");
  // b names a.cfg through a differently-spelled (x/./y) path: same file.
  check("write relative cycle a", write_file(ra, "config = " + rb + "\n"));
  const std::string ra_spelled = std::filesystem::path(ra).parent_path().string() +
                                 "/./" + std::filesystem::path(ra).filename().string();
  check("write relative cycle b", write_file(rb, "config = " + ra_spelled + "\n"));
  {
    std::string err;
    check("spelling-normalised cycle rejected", !cfg.load_file(ra, &err));
    check("normalised cycle mentions cycle", contains(err, "cycle"));
  }
  cleanup(ra);
  cleanup(rb);

  // Deep (but acyclic) chain must be rejected by the depth cap, not recurse forever.
  const int depth = 12;
  for (int i = 0; i < depth; ++i) {
    const std::string cur = build_path(("cfg_strict_deep_" + std::to_string(i) + ".cfg").c_str());
    const std::string nxt =
        build_path(("cfg_strict_deep_" + std::to_string(i + 1) + ".cfg").c_str());
    if (i + 1 < depth) write_file(cur, "config = " + nxt + "\n");
    else write_file(cur, "ctx = 111\n");
  }
  {
    std::string err;
    check("depth cap rejects deep chain",
          !cfg.load_file(build_path("cfg_strict_deep_0.cfg"), &err));
    check("depth error mentions depth", contains(err, "depth"));
  }
  for (int i = 0; i < depth; ++i)
    cleanup(build_path(("cfg_strict_deep_" + std::to_string(i) + ".cfg").c_str()));

  // A normal top-level load after all the failures must still work.
  const std::string ok_file = build_path("cfg_strict_ok.cfg");
  check("write plain config", write_file(ok_file, "ctx = 64\n"));
  {
    std::string err;
    check("plain config loads after failures", cfg.load_file(ok_file, &err) &&
                                                   cfg.get_int("ctx") == 64);
  }
  cleanup(ok_file);
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  std::printf("config validation regression\n");
  {
    // Locate the repo root from the executable path (build/test_config_strict.exe),
    // so the test works regardless of the current working directory.
    std::error_code ec;
    std::filesystem::path exe = argv[0];
    std::filesystem::path root = exe.parent_path().parent_path();
    if (root.empty()) root = ".";
    g_tmp_dir = (root / "build" / "cfg_strict_tmp").string();
    std::filesystem::create_directories(g_tmp_dir, ec);
    if (ec) {
      // Fall back to a CWD-relative directory (old behaviour) if the executable
      // path is not resolvable.
      g_tmp_dir = "build/cfg_strict_tmp";
      std::filesystem::create_directories(g_tmp_dir, ec);
    }
  }
  check_float_options();
  check_int_and_bool_validation();
  check_include_chain();
  std::error_code ec;
  std::filesystem::remove_all(g_tmp_dir, ec);
  std::printf("CONFIG STRICT: %d checks, %d failed\n", checks, fails);
  return fails == 0 ? 0 : 1;
}
