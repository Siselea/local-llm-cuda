// Unified run-time configuration for the engine.
//
// Priority (lowest to highest): built-in default -> environment variable
// (compatibility layer, the historical ENGINE_*/RUNNER_*/DSH_* variables) ->
// config file (--config <file>, key = value) -> command line (--ctx 512).
//
// The engine reads every tuning knob from here; the environment variables keep
// working so existing scripts (tools/verify.sh, tools/smoke_models.sh) are not
// affected.  Debug/dump variables that only emit diagnostics are intentionally
// left as plain getenv() and are documented in docs/CONFIG.md.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace engine {
namespace config {

enum class Type { Int, Float, Bool, String };

enum class Source { Default = 0, Env = 1, File = 2, Cli = 3 };

const char* source_name(Source s);

// One configurable option.  The table lives in src/config.cpp.
struct OptionDef {
  const char* key;          // canonical config key, e.g. "vram_margin_mb"
  const char* env;          // legacy env variable (nullable)
  const char* cli;          // long command line option without "--" (nullable)
  bool cli_takes_value;     // false -> boolean switch such as --no-graph
  bool env_invert;          // env variable presence/truth means "false"
  Type type;
  const char* def;          // default value as text
  double minv, maxv;        // numeric range (inclusive)
  const char* help;
};

struct Item {
  std::string key;
  Type type = Type::String;
  std::string raw;
  int64_t i = 0;
  double f = 0.0;
  bool b = false;
  Source src = Source::Default;
};

class Config {
 public:
  static Config& get();

  // Load defaults + environment.  Idempotent; called lazily by accessors.
  void ensure_initialized() const;

  // config file: "# comment" lines, "key = value", "[kernel]" section for
  // calibration overrides.  Returns false and fills *err on the first problem.
  bool load_file(const std::string& path, std::string* err);
  const std::string& loaded_file() const { return loaded_file_; }

  // Parse command line tokens argv[first..argc-1].  A --config option is
  // honoured before everything else (config file outranks the environment),
  // then the remaining options are applied (outranking the file).
  // Positional arguments and unrecognised options are returned to the caller.
  bool parse_command_line(int argc, char** argv, int first,
                          std::vector<std::string>* positional,
                          std::vector<std::string>* unparsed, std::string* err);

  // Programmatic set (used by the parser, tests and the calibrator).
  bool set(const std::string& key, const std::string& value, Source src, std::string* err);

  // ---- typed getters ----
  int64_t get_int(const std::string& key) const;
  double get_double(const std::string& key) const;
  bool get_bool(const std::string& key) const;
  std::string get_string(const std::string& key) const;
  Source source_of(const std::string& key) const;
  bool was_set(const std::string& key) const;  // source != Default
  const OptionDef* def_of(const std::string& key) const;
  const std::vector<OptionDef>& definitions() const { return defs_; }

  // ---- kernel calibration cache ----
  // Shape key format: "<quant type> <n_out>x<n_in>", e.g. "Q6_K 248320x4096".
  void set_kernel_override(const std::string& shape, const std::string& plan);
  const std::map<std::string, std::string>& kernel_overrides() const { return kernel_over_; }
  const std::string* find_kernel_override(const std::string& shape) const;
  bool load_kernel_cache(const std::string& path, std::string* err);
  bool save_kernel_cache(const std::string& path, std::string* err) const;
  // Load the cache named by the kernel_cache option when the file exists.
  // Returns 0 = not found (fine), 1 = loaded, -1 = error (fills *err).
  int load_kernel_cache_if_exists(std::string* err);

  // ---- diagnostics ----
  void print_effective(FILE* f) const;
  void print_options(FILE* f) const;
  // Machine-readable configuration surface (key / cli / type / default / range /
  // help / effective value / value source).  Output only: the GUI launcher builds
  // its "advanced" panel from this so a new option shows up without touching the
  // GUI.  Reachable as `engine.exe <cmd> --print-config --json`.
  void print_schema_json(FILE* f) const;

 private:
  Config();
  const OptionDef* find_def(const std::string& key) const;
  bool apply_text(const OptionDef& def, const std::string& text, Source src,
                  std::string* err, const char* where);
  bool parse_bool_text(const std::string& text, bool* out) const;
  bool env_loaded_ = false;
  std::string loaded_file_;
  std::map<std::string, Item> items_;
  std::map<std::string, std::string> kernel_over_;
  std::vector<OptionDef> defs_;
};

// ---------------------------------------------------------------------------
// Convenience wrappers used by the hot paths (no exceptions, cached table).
// ---------------------------------------------------------------------------

// Generic access; initialises from the environment on first use.
inline int64_t get_int(const std::string& key) { return Config::get().get_int(key); }
inline double get_double(const std::string& key) { return Config::get().get_double(key); }
inline bool get_bool(const std::string& key) { return Config::get().get_bool(key); }
inline std::string get_string(const std::string& key) { return Config::get().get_string(key); }
inline bool was_set(const std::string& key) { return Config::get().was_set(key); }

// Kernel calibration override; returns nullptr when the shape was not calibrated.
inline const std::string* kernel_override(const std::string& shape) {
  return Config::get().find_kernel_override(shape);
}

// ---- Windows scheduling (real-desktop priority control) -------------------
// Apply the configured process priority class (`win_priority`: 0 = normal,
// 1 = HIGH_PRIORITY_CLASS, 2 = ABOVE_NORMAL_PRIORITY_CLASS) to the current
// process.  Call once at program start, before the model loads: on a real
// Windows desktop the CPU expert GEMV threads otherwise lose cycles to
// background apps (LANDrop / Steam / browsers), which is the difference
// between the in-engine host GEMV time and the isolated benchmark.
//
// Idempotent.  No-op outside Windows.  REALTIME is deliberately not offered:
// it starves the input/IO/WDDM-submission threads (mouse freeze) and needs
// extra privileges.  The CPU pool workers get their *thread* priority from
// `win_thread_priority` (see cpu_matvec.cpp), not from here.
void apply_win_priority();

}  // namespace config
}  // namespace engine

// ---------------------------------------------------------------------------
// Cached environment lookups for decode hot paths.
// std::getenv() measures 2.13 us/call on this host (the CRT walks the whole environment
// block) and the decode path evaluated ~30 of them per layer => >2 ms/token.  The
// environment cannot change after start-up, so each call site caches its own answer on
// first use.  Values are identical to the uncached ones (bit-exact behaviour); plain
// getenv() is still fine for load-time / one-shot diagnostics.
#ifndef ENGINE_ENV_ON
#define ENGINE_ENV_ON(name) \
  ([]() -> bool { static const bool v = std::getenv(name) != nullptr; return v; }())
#define ENGINE_ENV_STR(name) \
  ([]() -> const char* { static const char* const v = std::getenv(name); return v; }())
#endif
