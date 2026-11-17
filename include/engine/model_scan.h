// Model discovery for the interactive picker, `engine models` and the
// double-click launchers (`启动.bat`).
//
// Design notes:
//   * The scanner is built into engine.exe (not `dir` inside a .bat) so the
//     logic is reusable, testable and shared by `serve` / `models`.  A .bat
//     only has to call `engine.exe`.
//   * No model file name is hard-coded: the "type" column is a *guess* from the
//     path (MoE~ / dense~, param count, quantisation); the loader is the source
//     of truth and prints the real architecture once the model is opened.
//   * Only *.gguf is listed, directories are scanned one level deep, split
//     GGUFs (-00001-of-000NN) are merged into one entry, `mmproj-*` vision
//     projector files are skipped.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine {
namespace model_scan {

struct Entry {
  std::string path;        // file to pass to the loader (first shard when split)
  std::string name;        // display name (file name; shard suffix stripped)
  std::string dir;         // directory the file was found in
  uint64_t size_bytes = 0; // total bytes (sum over shards)
  int shards = 1;          // >1 when a split GGUF was merged
  std::string quant;       // guessed quantisation, e.g. "Q4_K_M" ("" = unknown)
  std::string params;      // guessed parameter count, e.g. "35B" ("" = unknown)
  std::string kind;        // guessed "MoE~" / "dense~" (name based, only a hint)
  bool mtp = false;        // name mentions MTP (multi-token prediction)
  std::string drive;       // "D:" (full-disk scan) / derived from path otherwise
};

// Per-drive statistics for a full-disk scan (also serialized into the cache so
// `engine models --json` can report where the models came from).
struct DriveStat {
  std::string root;        // "D:\\"
  int dirs = 0;
  int found = 0;
  double seconds = 0;
  uint64_t root_mtime = 0; // FILETIME of the drive root when scanned (cache validity)
  bool ok = false;
  std::string error;
};

struct ScanInfo {
  std::string source;      // "cache" | "quick" | "full" | "no-scan"
  std::string cache_path;
  bool cache_hit = false;
  bool cache_stale = false;
  std::string generated_at; // ISO-8601 local time of the cached full scan
  double scan_seconds = 0;
  int dirs_scanned = 0;
  std::vector<DriveStat> drives;
};

// Full-disk scan options (used by `engine models` / the launcher background scan).
struct ScanOptions {
  std::vector<std::string> extra_dirs;  // --model-dir, searched first / higher priority
  bool include_builtin = true;          // keep the old built-in dirs as fallback
  bool full_disk = false;               // recurse over all fixed/removable drives
  bool refresh = false;                 // force a full scan and rewrite the cache
  bool no_scan = false;                 // only `extra_dirs`, ignore cache/full disk
  bool progress = false;                // per-drive progress lines on stderr
  int max_depth = 7;                    // directory depth limit for the full scan
  uint64_t min_size_bytes = 100ull << 20; // ignore tiny/fragment .gguf files
  std::string cache_path;               // empty = next to engine.exe (env override)
};

struct Report {
  std::vector<std::string> dirs;         // search order (deduplicated)
  std::vector<std::string> missing_dirs; // searched but absent (used by the hint)
  std::vector<std::string> warnings;     // skipped/merged files worth telling about
  std::vector<Entry> entries;            // sorted by size, largest first
  // Extra provenance for the launcher / cache diagnostics.
  std::string source;                    // "cache" | "quick" | "full" | "no-scan"
  std::string cache_path;
  std::string generated_at;
  bool cache_hit = false;
  bool cache_stale = false;
  double scan_seconds = 0;
  int dirs_scanned = 0;
  std::vector<DriveStat> drive_stats;
};

// Scan the built-in directories (exe-dir/models, exe-dir/../models, %USERPROFILE%/models,
// %USERPROFILE%/Downloads, %USERPROFILE%/.cache).  extra_dirs are searched first,
// so --model-dir can override the discovery order or add more directories.
Report scan(const std::vector<std::string>& extra_dirs = {});

// Cache-aware + full-disk scan used by `engine models` and the launcher.
//  * refresh=true            -> enumerate all fixed/removable drives recursively,
//                               write the cache and return that result.
//  * refresh=false, cache ok -> return the cached result immediately.
//  * no cache / no_scan      -> scan only the explicitly requested dirs (fast).
Report scan_with_options(const ScanOptions& opts, ScanInfo* info = nullptr);

// Fast list for `serve --pick` / the interactive picker: cached full-scan result
// when available, otherwise the legacy one-level built-in scan.
Report scan_cached_or_quick(const std::vector<std::string>& extra_dirs = {},
                            ScanInfo* info = nullptr);

// Cache file path (next to engine.exe; ENGINE_MODELS_CACHE overrides it).
std::string cache_file_path();

// Human-readable listing (used by `engine models` and the interactive picker).
std::string format_table(const Report& r);
// Machine-readable "index|path" lines, one per model (batch launchers / scripts).
std::string format_plain(const Report& r);
// Tab-separated: index, path, name, size_bytes, type (launcher.exe parses this).
std::string format_tsv(const Report& r);
// JSON listing (tests / tooling).
std::string format_json(const Report& r);
// Clear "no model found, put one here / pass --model" message.
std::string no_models_hint(const Report& r);

// "21.84 GB" (bytes / 1e9, matching the sizes the project docs quote).
std::string size_gb(uint64_t bytes);

}  // namespace model_scan
}  // namespace engine
