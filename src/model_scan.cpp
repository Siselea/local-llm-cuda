// Model discovery for `engine models`, the interactive picker and the
// double-click launcher.  See include/engine/model_scan.h for the rules.
#include "engine/model_scan.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "engine/json_parse.h"
#include "engine/json.h"
#include <thread>
#include <mutex>
#include <map>
#include <cstdarg>
#include <chrono>
#include <atomic>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace engine {
namespace model_scan {

namespace {

namespace fs = std::filesystem;

std::string drive_of_path(const std::string& path) {
  if (path.size() >= 2 && path[1] == ':') {
    char c = static_cast<char>(std::toupper(static_cast<unsigned char>(path[0])));
    return std::string(1, c) + ":";
  }
  return std::string();
}

std::string lower_ascii(const std::string& s) {
  std::string r = s;
  for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return r;
}

bool ends_with_ci(const std::string& lower_s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return lower_s.size() >= n && lower_s.compare(lower_s.size() - n, n, suffix) == 0;
}

bool all_digits(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s)
    if (c < '0' || c > '9') return false;
  return true;
}

// Directory containing engine.exe.  The double-click launcher changes the
// working directory, so "models next to the exe" must not be resolved relative
// to the current directory.
std::string exe_dir() {
#ifdef _WIN32
  wchar_t buf[4096];
  const DWORD n = GetModuleFileNameW(nullptr, buf, 4096);
  if (n == 0 || n >= 4096) return std::string();
  return fs::path(buf).parent_path().string();
#else
  std::error_code ec;
  const fs::path p = fs::read_symlink("/proc/self/exe", ec);
  if (ec) return std::string();
  return p.parent_path().string();
#endif
}

std::vector<std::string> builtin_dirs() {
  // Portability/test hook: ENGINE_MODEL_SCAN_DIRS (semicolon separated) replaces
  // the built-in list.  It exists so the "no models found" path can be tested
  // (and so another machine can point the scan somewhere else) without moving
  // real model files around.
  if (const char* override_dirs = std::getenv("ENGINE_MODEL_SCAN_DIRS");
      override_dirs != nullptr && *override_dirs != '\0') {
    std::vector<std::string> out;
    const std::string all = override_dirs;
    size_t begin = 0;
    while (begin <= all.size()) {
      const size_t end = all.find(';', begin);
      const std::string part =
          all.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
      if (!part.empty()) out.push_back(part);
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    return out;
  }
  std::vector<std::string> dirs;
  const std::string ed = exe_dir();
  if (!ed.empty()) {
    dirs.push_back((fs::path(ed) / "models").string());
    dirs.push_back((fs::path(ed) / ".." / "models").string());
  }
  if (const char* up = std::getenv("USERPROFILE"); up != nullptr && *up != '\0') {
    // One of several candidates, never a hard-coded model name/path.
    dirs.push_back((fs::path(up) / "models").string());
    dirs.push_back((fs::path(up) / "Downloads").string());
    dirs.push_back((fs::path(up) / ".cache").string());
  }
  return dirs;
}

// "model-00001-of-00002.gguf" -> idx=1, total=2, base="model.gguf"
bool parse_shard_name(const std::string& file_name, int* idx, int* total, std::string* base) {
  const std::string lower = lower_ascii(file_name);
  if (!ends_with_ci(lower, ".gguf")) return false;
  const std::string stem = file_name.substr(0, file_name.size() - 5);
  const std::string marker = "-of-";
  const size_t m = stem.rfind(marker);
  if (m == std::string::npos) return false;
  const size_t left = stem.rfind('-', m == 0 ? 0 : m - 1);
  if (left == std::string::npos || left == m) return false;
  const std::string a = stem.substr(left + 1, m - left - 1);
  const std::string b = stem.substr(m + marker.size());
  if (a.size() < 3 || b.size() < 3 || !all_digits(a) || !all_digits(b)) return false;
  const long ia = std::strtol(a.c_str(), nullptr, 10);
  const long it = std::strtol(b.c_str(), nullptr, 10);
  if (ia < 1 || it < 1 || ia > it || it > 99999) return false;
  *idx = static_cast<int>(ia);
  *total = static_cast<int>(it);
  *base = stem.substr(0, left) + ".gguf";
  return true;
}

bool is_gguf_name(const std::string& file_name) {
  return ends_with_ci(lower_ascii(file_name), ".gguf");
}

// --- file-name heuristics (reference only) ---------------------------------

bool looks_like_quant_token(const std::string& tok) {
  if (tok.size() < 2 || tok.size() > 10) return false;
  size_t i = 0;
  if (tok[0] == 'i' && tok.size() > 2 && tok[1] == 'q') i = 1;
  else if (tok[0] == 'q') i = 0;
  else return false;
  if (tok[i] != 'q' || i + 1 >= tok.size() || !std::isdigit(static_cast<unsigned char>(tok[i + 1])))
    return false;
  for (; i < tok.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(tok[i]);
    if (!std::isalnum(c) && c != '_') return false;
  }
  return true;
}

// Last quant-looking token, e.g. "qwen2.5-0.5b-instruct-q4_k_m.gguf" -> Q4_K_M.
std::string guess_quant(const std::string& lower) {
  std::string best;
  for (size_t i = 0; i < lower.size(); ++i) {
    const char c = lower[i];
    const bool start_q = (c == 'q' && i + 1 < lower.size() &&
                          std::isdigit(static_cast<unsigned char>(lower[i + 1])));
    const bool start_iq = (c == 'i' && i + 2 < lower.size() && lower[i + 1] == 'q' &&
                           std::isdigit(static_cast<unsigned char>(lower[i + 2])));
    if (!start_q && !start_iq) continue;
    size_t j = i + 1;
    while (j < lower.size() &&
           (std::isalnum(static_cast<unsigned char>(lower[j])) || lower[j] == '_'))
      ++j;
    const std::string tok = lower.substr(i, j - i);
    if (looks_like_quant_token(tok)) best = tok;
    i = j - 1;
  }
  if (best.empty()) {
    if (lower.find("bf16") != std::string::npos) return "BF16";
    if (lower.find("f16") != std::string::npos) return "F16";
    if (lower.find("f32") != std::string::npos) return "F32";
    return std::string();
  }
  for (char& c : best) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return best;
}

// Largest "<n>b"-style parameter count at a token boundary, so
// "Qwen3.6-35B-A3B" -> 35B (not the 3B active-parameter marker of A3B).
std::string guess_params(const std::string& lower) {
  std::string best;
  double best_value = -1.0;
  for (size_t i = 0; i + 1 < lower.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(lower[i]))) continue;
    if (i > 0 && (std::isalnum(static_cast<unsigned char>(lower[i - 1])) || lower[i - 1] == '.'))
      continue;
    size_t j = i;
    while (j < lower.size()) {
      if (std::isdigit(static_cast<unsigned char>(lower[j]))) {
        ++j;
      } else if (lower[j] == '.' && j + 1 < lower.size() &&
                 std::isdigit(static_cast<unsigned char>(lower[j + 1]))) {
        ++j;
      } else {
        break;
      }
    }
    if (j >= lower.size() || lower[j] != 'b') continue;
    if (j + 1 < lower.size() && std::isalnum(static_cast<unsigned char>(lower[j + 1]))) continue;
    const std::string num = lower.substr(i, j - i);
    if (num.size() > 5) continue;
    const double v = std::strtod(num.c_str(), nullptr);
    if (v > best_value) {
      best_value = v;
      best = num;
    }
    i = j;
  }
  if (best.empty()) return std::string();
  return best + "B";
}

// MoE hint: explicit "moe"/"expert" keyword, or the Qwen-style active-parameter
// suffix ("...-A3B", "...-A22B") that only MoE checkpoints use.
std::string guess_kind(const std::string& lower) {
  if (lower.find("moe") != std::string::npos || lower.find("expert") != std::string::npos)
    return "MoE~";
  for (size_t i = 0; i + 2 < lower.size(); ++i) {
    if (lower[i] != 'a') continue;
    if (i > 0 && std::isalnum(static_cast<unsigned char>(lower[i - 1]))) continue;
    size_t j = i + 1;
    if (!std::isdigit(static_cast<unsigned char>(lower[j]))) continue;
    while (j < lower.size() && std::isdigit(static_cast<unsigned char>(lower[j]))) ++j;
    if (j < lower.size() && lower[j] == 'b' &&
        (j + 1 == lower.size() || !std::isalnum(static_cast<unsigned char>(lower[j + 1]))))
      return "MoE~";
  }
  return "dense~";
}

void fill_guess(Entry* e) {
  const std::string lower = lower_ascii(e->name);
  e->quant = guess_quant(lower);
  e->params = guess_params(lower);
  e->kind = guess_kind(lower);
  e->mtp = lower.find("mtp") != std::string::npos;
}

std::string normalized_dir(const std::string& dir) {
  return fs::path(dir).lexically_normal().string();
}

}  // namespace

std::string size_gb(uint64_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / 1e9);
  return buf;
}

Report scan(const std::vector<std::string>& extra_dirs) {
  Report report;
  std::vector<std::string> candidates = extra_dirs;
  for (const std::string& d : builtin_dirs()) candidates.push_back(d);

  std::vector<std::string> seen_dirs;
  for (const std::string& d : candidates) {
    const std::string norm = normalized_dir(d);
    const std::string key = lower_ascii(norm);
    bool dup = false;
    for (const std::string& s : seen_dirs)
      if (s == key) dup = true;
    if (dup) continue;
    seen_dirs.push_back(key);
    report.dirs.push_back(norm);
  }

  struct ShardGroup {
    std::string base;
    std::string key;
    int total = 0;
    std::vector<int> index;
    std::vector<std::string> path;
    std::vector<uint64_t> size;
  };

  std::vector<Entry> found;
  std::vector<std::string> found_keys;

  for (const std::string& dir : report.dirs) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
      report.missing_dirs.push_back(dir);
      continue;
    }

    std::vector<fs::directory_entry> files;
    std::error_code iter_ec;
    for (fs::directory_iterator it(dir, iter_ec), end; !iter_ec && it != end; it.increment(iter_ec)) {
      std::error_code fec;
      if (!it->is_regular_file(fec)) continue;
      const std::string name = it->path().filename().string();
      if (!is_gguf_name(name)) continue;
      files.push_back(*it);
    }

    std::vector<ShardGroup> groups;
    int skipped_projectors = 0;
    for (const fs::directory_entry& f : files) {
      const std::string name = f.path().filename().string();
      const std::string lower = lower_ascii(name);
      // Vision projectors are companion files, not chat models.
      if (lower.find("mmproj") != std::string::npos) {
        ++skipped_projectors;
        continue;
      }
      int idx = 0, total = 0;
      std::string base;
      if (parse_shard_name(name, &idx, &total, &base)) {
        const std::string key = lower_ascii(base) + "|" + std::to_string(total);
        std::error_code sec;
        const uint64_t bytes = f.file_size(sec);
        ShardGroup* g = nullptr;
        for (ShardGroup& cand : groups)
          if (cand.key == key) g = &cand;
        if (g == nullptr) {
          groups.push_back(ShardGroup{});
          g = &groups.back();
          g->base = base;
          g->key = key;
          g->total = total;
          g->index.assign(static_cast<size_t>(total), 0);
          g->path.assign(static_cast<size_t>(total), std::string());
          g->size.assign(static_cast<size_t>(total), 0);
        }
        // Missing/short files are handled below; never index out of range.
        if (idx >= 1 && idx <= total && g->index[static_cast<size_t>(idx - 1)] == 0) {
          g->index[static_cast<size_t>(idx - 1)] = 1;
          g->path[static_cast<size_t>(idx - 1)] = f.path().string();
          g->size[static_cast<size_t>(idx - 1)] = sec ? 0 : bytes;
        }
        continue;
      }
      Entry e;
      e.path = f.path().string();
      e.name = name;
      e.dir = dir;
      std::error_code sec;
      e.size_bytes = f.file_size(sec);
      if (sec) e.size_bytes = 0;
      fill_guess(&e);
      found.push_back(e);
    }

    for (const ShardGroup& g : groups) {
      int have = 0;
      uint64_t total_bytes = 0;
      for (size_t i = 0; i < g.index.size(); ++i) {
        if (g.index[i] != 0) {
          ++have;
          total_bytes += g.size[i];
        }
      }
      if (g.index.empty() || g.index[0] == 0) {
        report.warnings.push_back("skipped split GGUF without its first shard: " + g.base +
                                  " (" + dir + ")");
        continue;
      }
      if (have != g.total) {
        report.warnings.push_back("split GGUF is incomplete (" + std::to_string(have) + "/" +
                                  std::to_string(g.total) + " shards found): " + g.base + " (" +
                                  dir + ")");
      }
      Entry e;
      e.path = g.path[0];
      e.name = g.base;
      e.dir = dir;
      e.size_bytes = total_bytes;
      e.shards = g.total;
      fill_guess(&e);
      found.push_back(e);
    }
    if (skipped_projectors > 0) {
      report.warnings.push_back("skipped " + std::to_string(skipped_projectors) +
                                " mmproj-*.gguf vision projector file(s) in " + dir);
    }
  }

  for (const Entry& e : found) {
    const std::string key = lower_ascii(fs::path(e.path).lexically_normal().string());
    bool dup = false;
    for (const std::string& s : found_keys)
      if (s == key) dup = true;
    if (dup) continue;
    found_keys.push_back(key);
    report.entries.push_back(e);
  }
  std::sort(report.entries.begin(), report.entries.end(), [](const Entry& a, const Entry& b) {
    if (a.size_bytes != b.size_bytes) return a.size_bytes > b.size_bytes;
    return lower_ascii(a.name) < lower_ascii(b.name);
  });
  return report;
}

std::string format_table(const Report& r) {
  std::string out;
  out += "search paths (" + std::to_string(r.dirs.size()) + ", order = priority):\n";
  for (const std::string& dir : r.dirs) {
    std::error_code ec;
    out += fs::is_directory(dir, ec) ? "  [ok] " : "  [--] ";
    out += dir + "\n";
  }
  out += "\nGGUF models (" + std::to_string(r.entries.size()) + ", largest first):\n";
  if (r.entries.empty()) {
    out += "  (none)\n";
  } else {
    out += "  #   size        type        model\n";
    for (size_t i = 0; i < r.entries.size(); ++i) {
      const Entry& e = r.entries[i];
      std::string type = e.kind;
      if (!e.params.empty()) type += " " + e.params;
      if (!e.quant.empty()) type += " " + e.quant;
      char head[128];
      std::snprintf(head, sizeof(head), "  %-3zu %-11s %-22s ", i + 1,
                    size_gb(e.size_bytes).c_str(), type.c_str());
      out += head;
      out += e.name;
      if (e.shards > 1) out += " (split: " + std::to_string(e.shards) + " shards)";
      if (e.mtp) out += " [MTP]";
      out += "\n      [" + (e.drive.empty() ? drive_of_path(e.path) : e.drive) + "] " +
             e.dir + "\n";
    }
  }
  for (const std::string& w : r.warnings) out += "note: " + w + "\n";
  return out;
}

std::string format_plain(const Report& r) {
  std::string out;
  for (size_t i = 0; i < r.entries.size(); ++i) {
    out += std::to_string(i + 1);
    out += "|";
    out += r.entries[i].path;
    out += "\n";
  }
  return out;
}

std::string format_tsv(const Report& r) {
  std::string out;
  for (size_t i = 0; i < r.entries.size(); ++i) {
    const Entry& e = r.entries[i];
    std::string type = e.kind;
    if (!e.params.empty()) type += " " + e.params;
    if (!e.quant.empty()) type += " " + e.quant;
    out += std::to_string(i + 1);
    out += "\t" + e.path + "\t" + e.name + "\t" + std::to_string(e.size_bytes) + "\t" + type;
    out += "\n";
  }
  return out;
}

std::string format_json(const Report& r) {
  json::Writer w;
  w.begin_obj();
  w.key("source");
  w.str(r.source.empty() ? "quick" : r.source);
  w.key("cache_path");
  w.str(r.cache_path);
  w.key("cache_hit");
  w.boolean(r.cache_hit);
  w.key("cache_stale");
  w.boolean(r.cache_stale);
  w.key("generated_at");
  w.str(r.generated_at);
  w.key("scan_seconds");
  w.number(r.scan_seconds);
  w.key("dirs_scanned");
  w.integer(r.dirs_scanned);
  w.key("drive_stats");
  w.begin_arr();
  for (const DriveStat& d : r.drive_stats) {
    w.begin_obj();
    w.key("root");
    w.str(d.root);
    w.key("dirs");
    w.integer(d.dirs);
    w.key("found");
    w.integer(d.found);
    w.key("seconds");
    w.number(d.seconds);
    w.key("root_mtime");
    w.str(std::to_string(d.root_mtime));
    w.key("ok");
    w.boolean(d.ok);
    w.end_obj();
  }
  w.end_arr();
  w.key("search_dirs");
  w.begin_arr();
  for (const std::string& d : r.dirs) w.str(d);
  w.end_arr();
  w.key("missing_dirs");
  w.begin_arr();
  for (const std::string& d : r.missing_dirs) w.str(d);
  w.end_arr();
  w.key("warnings");
  w.begin_arr();
  for (const std::string& s : r.warnings) w.str(s);
  w.end_arr();
  w.key("count");
  w.integer(static_cast<int64_t>(r.entries.size()));
  w.key("models");
  w.begin_arr();
  for (size_t i = 0; i < r.entries.size(); ++i) {
    const Entry& e = r.entries[i];
    w.begin_obj();
    w.key("index");
    w.integer(static_cast<int64_t>(i + 1));
    w.key("path");
    w.str(e.path);
    w.key("name");
    w.str(e.name);
    w.key("dir");
    w.str(e.dir);
    w.key("drive");
    w.str(e.drive.empty() ? drive_of_path(e.path) : e.drive);
    w.key("size_bytes");
    w.uinteger(e.size_bytes);
    w.key("size_gb");
    w.number(static_cast<double>(e.size_bytes) / 1e9);
    w.key("shards");
    w.integer(e.shards);
    w.key("quant");
    w.str(e.quant);
    w.key("params");
    w.str(e.params);
    w.key("type");
    w.str(e.kind);
    w.key("mtp");
    w.boolean(e.mtp);
    w.end_obj();
  }
  w.end_arr();
  w.end_obj();
  return w.out;
}

std::string no_models_hint(const Report& r) {
  std::string out;
  out += "\n没有找到任何 .gguf 模型。\n";
  out += "全盘扫描没有找到模型（或扫描缓存为空）。可以：\n";
  out += "  - engine.exe models --refresh --progress     # 重新全盘扫描（固定盘/移动盘）\n";
  out += "  - 把 .gguf 放到下面任一目录（默认扫描目录顶层）：\n";
  for (const std::string& d : r.dirs) out += "  - " + d + "\n";
  out += "或者用命令行显式指定：\n";
  out += "  engine.exe serve \"D:/path/to/model.gguf\" --ctx 4096\n";
  out += "  engine.exe serve --model-dir \"D:/my/models\" --pick 1\n";
  out += "  engine.exe models                 # 查看扫描结果\n";
  return out;
}

// ===========================================================================
// Full-disk scanning (`engine models --refresh`) + cache (`engine-models-cache.json`)
// ===========================================================================
namespace {

#ifdef _WIN32
std::string utf8_from_wide(const std::wstring& w) {
  if (w.empty()) return std::string();
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0,
                                    nullptr, nullptr);
  if (n <= 0) return std::string();
  std::string out(static_cast<size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), out.data(), n, nullptr,
                      nullptr);
  return out;
}

std::wstring wide_from_utf8(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
  if (n <= 0) return std::wstring();
  std::wstring out(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n);
  return out;
}

std::wstring lower_wide(std::wstring s) {
  for (wchar_t& c : s) c = static_cast<wchar_t>(std::towlower(c));
  return s;
}

uint64_t filetime_u64(const FILETIME& ft) {
  return (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}

uint64_t drive_root_mtime(const std::wstring& root) {
  WIN32_FILE_ATTRIBUTE_DATA fad;
  if (!GetFileAttributesExW(root.c_str(), GetFileExInfoStandard, &fad)) return 0;
  return filetime_u64(fad.ftLastWriteTime);
}

std::string now_iso() {
  SYSTEMTIME st;
  GetLocalTime(&st);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04u-%02u-%02u %02u:%02u:%02u", st.wYear, st.wMonth, st.wDay,
                st.wHour, st.wMinute, st.wSecond);
  return buf;
}

// Drives to scan: fixed + removable only.  Network/CD/RAM drives are skipped on
// purpose (a dead network share can hang a scan for minutes).
std::vector<std::wstring> full_scan_roots() {
  std::vector<std::wstring> roots;
  if (const char* env = std::getenv("ENGINE_SCAN_DRIVES"); env != nullptr && *env != '\0') {
    std::string all = env;
    size_t begin = 0;
    while (begin <= all.size()) {
      size_t end = all.find(';', begin);
      std::string part = all.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
      const size_t b = part.find_first_not_of(" \t");
      const size_t e = part.find_last_not_of(" \t");
      if (b != std::string::npos) {
        std::wstring w = wide_from_utf8(part.substr(b, e - b + 1));
        while (w.size() > 3 && (w.back() == L'\\' || w.back() == L'/')) w.pop_back();
        if (!w.empty()) roots.push_back(w);
      }
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    return roots;
  }
  wchar_t buf[512];
  const DWORD n = GetLogicalDriveStringsW(512, buf);
  if (n == 0) return roots;
  for (DWORD i = 0; i < n;) {
    const wchar_t* root = buf + i;
    const size_t len = std::wcslen(root);
    if (len == 0) break;
    const UINT type = GetDriveTypeW(root);
    if (type == DRIVE_FIXED || type == DRIVE_REMOVABLE) roots.push_back(root);
    i += static_cast<DWORD>(len) + 1;
  }
  return roots;
}
#endif  // _WIN32

struct FullCandidate {
  std::string path;
  std::string dir;
  std::string name;
  std::string drive;
  uint64_t size = 0;
};

struct ScanCounters {
  bool progress = false;
  int drives_total = 0;
  std::atomic<int> drives_done{0};
  std::atomic<uint64_t> dirs{0};
  std::atomic<int> found{0};
  std::mutex mu;  // serialize stderr lines
  std::chrono::steady_clock::time_point t0;
  std::chrono::steady_clock::time_point last_progress;
};

void scan_log(ScanCounters& c, const char* fmt, ...) {
  if (!c.progress) return;
  char line[4096];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(line, sizeof(line), fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lock(c.mu);
  std::fprintf(stderr, "[scan] %s\n", line);
  std::fflush(stderr);
}

uint64_t elapsed_ms(ScanCounters& c) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - c.t0)
          .count());
}

void scan_progress(ScanCounters& c) {
  scan_log(c, "progress drives_done=%d drives_total=%d dirs=%llu found=%d elapsed_ms=%llu",
           c.drives_done.load(), c.drives_total, static_cast<unsigned long long>(c.dirs.load()),
           c.found.load(), static_cast<unsigned long long>(elapsed_ms(c)));
}

bool skip_scan_dir(const std::wstring& name_lower, int depth, const std::wstring& full_lower) {
  // Never descend these (any depth): VCS caches, package caches, recycle/volume.
  static const wchar_t* const any_depth[] = {L"$recycle.bin", L"system volume information",
                                             L"node_modules", L".git", L"recovery",
                                             L"__pycache__", L".cache"};
  for (const wchar_t* s : any_depth)
    if (name_lower == s) return true;
  if (full_lower.find(L"\\appdata\\local\\temp") != std::wstring::npos) return true;
  if (full_lower.find(L"\\windows\\installer") != std::wstring::npos) return true;
  // System/OS trees only at the drive root.
  if (depth == 1) {
    if (name_lower == L"windows" || name_lower == L"programdata" || name_lower == L"recovery" ||
        name_lower == L"system volume information" || name_lower == L"$recycle.bin")
      return true;
    if (name_lower.size() >= 13 && name_lower.compare(0, 13, L"program files") == 0) return true;
  }
  return false;
}

void walk_drive_dir(const std::wstring& dir, int depth, ScanCounters& counters, int max_depth,
                    uint64_t min_size, std::vector<FullCandidate>& out, uint64_t* local_dirs,
                    int* local_found) {
  std::wstring pattern = dir;
  if (!pattern.empty() && pattern.back() != L'\\' && pattern.back() != L'/') pattern += L'\\';
  pattern += L'*';
  WIN32_FIND_DATAW fd;
  HANDLE h = FindFirstFileExW(pattern.c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr,
                              FIND_FIRST_EX_LARGE_FETCH);
  if (h == INVALID_HANDLE_VALUE) {
    // Old/odd volumes may not support LARGE_FETCH / BASIC info: retry normally.
    h = FindFirstFileW(pattern.c_str(), &fd);
  }
  if (h == INVALID_HANDLE_VALUE) return;
  const uint64_t dirs_now = counters.dirs.fetch_add(1) + 1;
  if (local_dirs != nullptr) ++*local_dirs;
  const auto now = std::chrono::steady_clock::now();
  if (dirs_now % 10000 == 0 ||
      std::chrono::duration_cast<std::chrono::milliseconds>(now - counters.last_progress).count() >
          1500) {
    counters.last_progress = now;
    scan_progress(counters);
  }
  do {
    const wchar_t* wn = fd.cFileName;
    if (wn[0] == L'.' && (wn[1] == L'\0' || (wn[1] == L'.' && wn[2] == L'\0'))) continue;
    const bool is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    const bool reparse = (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
    if (reparse) continue;  // symlink/junction: never follow (loop prevention)
    std::wstring child = dir;
    if (!child.empty() && child.back() != L'\\' && child.back() != L'/') child += L'\\';
    child += wn;
    if (is_dir) {
      const std::wstring name_lower = lower_wide(wn);
      const std::wstring full_lower = lower_wide(child);
      if (depth < max_depth && !skip_scan_dir(name_lower, depth, full_lower))
        walk_drive_dir(child, depth + 1, counters, max_depth, min_size, out, local_dirs,
                       local_found);
      continue;
    }
    const std::wstring name_lower = lower_wide(wn);
    if (name_lower.size() < 6 ||
        name_lower.compare(name_lower.size() - 5, 5, L".gguf") != 0)
      continue;
    const uint64_t size =
        (static_cast<uint64_t>(fd.nFileSizeHigh) << 32) | static_cast<uint64_t>(fd.nFileSizeLow);
    if (size < min_size) continue;
    FullCandidate cand;
    cand.path = utf8_from_wide(child);
    cand.dir = utf8_from_wide(dir);
    cand.name = utf8_from_wide(wn);
    cand.drive = drive_of_path(cand.path);
    cand.size = size;
    const std::string log_drive = cand.drive;
    const std::string log_path = cand.path;
    out.push_back(std::move(cand));
    const int found = counters.found.fetch_add(1) + 1;
    if (local_found != nullptr) ++*local_found;
    scan_log(counters, "found [%s] %.2f MB %s", log_drive.c_str(),
             static_cast<double>(size) / 1e6, log_path.c_str());
    if (found % 500 == 0) scan_progress(counters);
  } while (FindNextFileW(h, &fd));
  FindClose(h);
}

void build_entries_from_candidates(const std::vector<FullCandidate>& candidates,
                                   std::vector<std::string>* warnings, std::vector<Entry>* entries) {
  struct ShardGroup {
    std::string base;
    std::string key;
    std::string dir;
    std::string drive;
    int total = 0;
    std::vector<int> have;
    std::vector<std::string> path;
    std::vector<uint64_t> size;
  };
  std::map<std::string, std::vector<const FullCandidate*>> by_dir;
  for (const auto& c : candidates) by_dir[c.dir].push_back(&c);
  int skipped_projectors = 0;
  for (const auto& [dir, files] : by_dir) {
    std::vector<ShardGroup> groups;
    for (const FullCandidate* f : files) {
      const std::string lower = lower_ascii(f->name);
      if (lower.find("mmproj") != std::string::npos) {
        ++skipped_projectors;
        continue;
      }
      int idx = 0, total = 0;
      std::string base;
      if (parse_shard_name(f->name, &idx, &total, &base)) {
        const std::string key = lower_ascii(base) + "|" + std::to_string(total);
        ShardGroup* g = nullptr;
        for (ShardGroup& cand : groups)
          if (cand.key == key) g = &cand;
        if (g == nullptr) {
          groups.push_back(ShardGroup{});
          g = &groups.back();
          g->base = base;
          g->key = key;
          g->dir = dir;
          g->drive = f->drive;
          g->total = total;
          g->have.assign(static_cast<size_t>(total), 0);
          g->path.assign(static_cast<size_t>(total), std::string());
          g->size.assign(static_cast<size_t>(total), 0);
        }
        if (idx >= 1 && idx <= total && g->have[static_cast<size_t>(idx - 1)] == 0) {
          g->have[static_cast<size_t>(idx - 1)] = 1;
          g->path[static_cast<size_t>(idx - 1)] = f->path;
          g->size[static_cast<size_t>(idx - 1)] = f->size;
        }
        continue;
      }
      Entry e;
      e.path = f->path;
      e.name = f->name;
      e.dir = dir;
      e.drive = f->drive;
      e.size_bytes = f->size;
      fill_guess(&e);
      entries->push_back(std::move(e));
    }
    for (const ShardGroup& g : groups) {
      int have = 0;
      uint64_t total_bytes = 0;
      for (size_t i = 0; i < g.have.size(); ++i) {
        if (g.have[i] != 0) {
          ++have;
          total_bytes += g.size[i];
        }
      }
      if (g.have.empty() || g.have[0] == 0) {
        if (warnings != nullptr)
          warnings->push_back("skipped split GGUF without its first shard: " + g.base + " (" +
                              g.dir + ")");
        continue;
      }
      if (have != g.total) {
        if (warnings != nullptr)
          warnings->push_back("split GGUF is incomplete (" + std::to_string(have) + "/" +
                              std::to_string(g.total) + " shards found): " + g.base + " (" + g.dir +
                              ")");
      }
      Entry e;
      e.path = g.path[0];
      e.name = g.base;
      e.dir = g.dir;
      e.drive = g.drive;
      e.size_bytes = total_bytes;
      e.shards = g.total;
      fill_guess(&e);
      entries->push_back(std::move(e));
    }
  }
  if (skipped_projectors > 0 && warnings != nullptr)
    warnings->push_back("skipped " + std::to_string(skipped_projectors) +
                        " mmproj-*.gguf vision projector file(s)");
}

void finalize_entries(std::vector<Entry>* entries) {
  std::vector<Entry> dedup;
  std::vector<std::string> keys;
  for (const Entry& e : *entries) {
    Entry copy = e;
    if (copy.drive.empty()) copy.drive = drive_of_path(copy.path);
    const std::string key = lower_ascii(fs::path(copy.path).lexically_normal().string());
    bool dup = false;
    for (const std::string& s : keys)
      if (s == key) dup = true;
    if (dup) continue;
    keys.push_back(key);
    dedup.push_back(std::move(copy));
  }
  std::sort(dedup.begin(), dedup.end(), [](const Entry& a, const Entry& b) {
    if (a.size_bytes != b.size_bytes) return a.size_bytes > b.size_bytes;
    return lower_ascii(a.name) < lower_ascii(b.name);
  });
  *entries = std::move(dedup);
}

// Top-level scan of explicitly requested dirs (--model-dir / --no-scan).
void scan_explicit_dirs(const std::vector<std::string>& dirs, uint64_t min_size,
                        std::vector<std::string>* warnings, std::vector<std::string>* scanned,
                        std::vector<Entry>* entries) {
  std::vector<FullCandidate> cands;
  for (const std::string& raw : dirs) {
    const std::string dir = normalized_dir(raw);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
      if (warnings != nullptr) warnings->push_back("missing model dir: " + dir);
      continue;
    }
    if (scanned != nullptr) scanned->push_back(dir);
    std::error_code iec;
    for (fs::directory_iterator it(dir, iec), end; !iec && it != end; it.increment(iec)) {
      std::error_code fec;
      if (!it->is_regular_file(fec)) continue;
      const std::string name = it->path().filename().string();
      if (!is_gguf_name(name)) continue;
      const uint64_t size = static_cast<uint64_t>(it->file_size(fec));
      if (fec || size < min_size) continue;
      FullCandidate c;
      c.path = it->path().string();
      c.dir = dir;
      c.name = name;
      c.drive = drive_of_path(c.path);
      c.size = size;
      cands.push_back(std::move(c));
    }
  }
  build_entries_from_candidates(cands, warnings, entries);
}

std::string cache_default_path() {
  if (const char* env = std::getenv("ENGINE_MODELS_CACHE"); env != nullptr && *env != '\0')
    return env;
  const std::string ed = exe_dir();
  if (!ed.empty()) return (fs::path(ed) / "engine-models-cache.json").string();
  if (const char* la = std::getenv("LOCALAPPDATA"); la != nullptr && *la != '\0')
    return (fs::path(la) / "engine-models-cache.json").string();
  return "engine-models-cache.json";
}

bool read_cache(const std::string& path, int depth, uint64_t min_size, Report* out, bool* stale,
                ScanInfo* info) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::string text;
  char buf[65536];
  size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, got);
  std::fclose(f);
  if (text.empty()) return false;
  jsonp::Value root;
  try {
    root = jsonp::parse(text);
  } catch (...) {
    return false;
  }
  if (!root.is_object()) return false;
  if (static_cast<int>(root.get_num("version", 0)) != 1) return false;
  if (static_cast<int>(root.get_num("depth", -1)) != depth) return false;
  const long long cached_min = static_cast<long long>(root.get_num("min_size_mb", -1));
  if (cached_min != static_cast<long long>(min_size >> 20)) return false;
  out->entries.clear();
  out->warnings.clear();
  out->dirs.clear();
  out->missing_dirs.clear();
  if (const jsonp::Value* arr = root.find("models"); arr != nullptr && arr->is_array()) {
    for (const jsonp::Value& m : arr->arr) {
      Entry e;
      e.path = m.get_str("path", "");
      e.name = m.get_str("name", "");
      e.dir = m.get_str("dir", "");
      e.drive = m.get_str("drive", drive_of_path(e.path));
      e.size_bytes = static_cast<uint64_t>(m.get_num("size_bytes", 0));
      e.shards = static_cast<int>(m.get_num("shards", 1));
      e.quant = m.get_str("quant", "");
      e.params = m.get_str("params", "");
      e.kind = m.get_str("type", "");
      e.mtp = m.get_bool("mtp", false);
      if (!e.path.empty()) out->entries.push_back(std::move(e));
    }
  }
  out->dirs.push_back(path);
  *stale = false;
  if (const jsonp::Value* drives = root.find("drives"); drives != nullptr && drives->is_array()) {
    for (const jsonp::Value& d : drives->arr) {
      DriveStat ds;
      ds.root = d.get_str("root", "");
      ds.dirs = static_cast<int>(d.get_num("dirs", 0));
      ds.found = static_cast<int>(d.get_num("found", 0));
      ds.seconds = d.get_num("seconds", 0);
      ds.root_mtime = std::strtoull(d.get_str("root_mtime", "0").c_str(), nullptr, 10);
      ds.ok = d.get_bool("ok", false);
      out->drive_stats.push_back(ds);
#ifdef _WIN32
      const uint64_t now = drive_root_mtime(wide_from_utf8(ds.root));
      if (now != 0 && ds.root_mtime != 0 && now != ds.root_mtime) *stale = true;
#endif
      if (info != nullptr) info->drives.push_back(ds);
    }
  }
  if (info != nullptr) {
    info->source = "cache";
    info->cache_hit = true;
    info->cache_stale = *stale;
    info->cache_path = path;
    info->generated_at = root.get_str("generated_at", "");
    info->scan_seconds = root.get_num("scan_seconds", 0);
    info->dirs_scanned = static_cast<int>(root.get_num("dirs_scanned", 0));
  }
  return true;
}

bool write_cache(const std::string& path, const Report& report, const ScanInfo& info, int depth,
                 uint64_t min_size) {
  json::Writer w;
  w.begin_obj();
  w.key("version");
  w.integer(1);
  w.key("generated_at");
  w.str(info.generated_at);
  w.key("scan_seconds");
  w.number(info.scan_seconds);
  w.key("depth");
  w.integer(depth);
  w.key("min_size_mb");
  w.integer(static_cast<int64_t>(min_size >> 20));
  w.key("dirs_scanned");
  w.integer(info.dirs_scanned);
  w.key("count");
  w.integer(static_cast<int64_t>(report.entries.size()));
  w.key("drives");
  w.begin_arr();
  for (const DriveStat& d : info.drives) {
    w.begin_obj();
    w.key("root");
    w.str(d.root);
    w.key("dirs");
    w.integer(d.dirs);
    w.key("found");
    w.integer(d.found);
    w.key("seconds");
    w.number(d.seconds);
    w.key("root_mtime");
    w.uinteger(d.root_mtime);
    w.key("ok");
    w.boolean(d.ok);
    w.end_obj();
  }
  w.end_arr();
  w.key("models");
  w.begin_arr();
  for (const Entry& e : report.entries) {
    w.begin_obj();
    w.key("path");
    w.str(e.path);
    w.key("name");
    w.str(e.name);
    w.key("dir");
    w.str(e.dir);
    w.key("drive");
    w.str(e.drive.empty() ? drive_of_path(e.path) : e.drive);
    w.key("size_bytes");
    w.uinteger(e.size_bytes);
    w.key("shards");
    w.integer(e.shards);
    w.key("quant");
    w.str(e.quant);
    w.key("params");
    w.str(e.params);
    w.key("type");
    w.str(e.kind);
    w.key("mtp");
    w.boolean(e.mtp);
    w.end_obj();
  }
  w.end_arr();
  w.end_obj();
  const std::string tmp = path + ".tmp";
  std::FILE* f = std::fopen(tmp.c_str(), "wb");
  if (f == nullptr) return false;
  const bool ok = std::fwrite(w.out.data(), 1, w.out.size(), f) == w.out.size();
  std::fclose(f);
  if (!ok) {
    std::remove(tmp.c_str());
    return false;
  }
#ifdef _WIN32
  if (!MoveFileExW(wide_from_utf8(tmp).c_str(), wide_from_utf8(path).c_str(),
                   MOVEFILE_REPLACE_EXISTING)) {
    std::remove(tmp.c_str());
    return false;
  }
#else
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    return false;
  }
#endif
  return true;
}

}  // namespace

std::string cache_file_path() { return cache_default_path(); }

Report scan_with_options(const ScanOptions& opts, ScanInfo* info) {
  ScanInfo local;
  if (info == nullptr) info = &local;
  *info = ScanInfo();
  Report report;
  const std::string cache_path = opts.cache_path.empty() ? cache_default_path() : opts.cache_path;
  info->cache_path = cache_path;

  if (opts.no_scan) {
    // Only the explicitly requested directories; no full disk, no cache.
    scan_explicit_dirs(opts.extra_dirs, 0, &report.warnings, &report.dirs, &report.entries);
    finalize_entries(&report.entries);
    info->source = "no-scan";
    report.source = info->source;
    report.cache_path = cache_path;
    return report;
  }

  if (!opts.refresh) {
    bool stale = false;
    if (read_cache(cache_path, opts.max_depth, opts.min_size_bytes, &report, &stale, info)) {
      report.source = info->source;
      report.cache_path = cache_path;
      report.cache_hit = true;
      report.cache_stale = stale;
      report.generated_at = info->generated_at;
      report.scan_seconds = info->scan_seconds;
      report.dirs_scanned = info->dirs_scanned;
      report.drive_stats = info->drives;
      for (Entry& e : report.entries)
        if (e.drive.empty()) e.drive = drive_of_path(e.path);
      return report;
    }
    // No usable cache: fast legacy scan (built-in dirs + --model-dir).
    report = scan(opts.extra_dirs);
    info->source = "quick";
    info->cache_path = cache_path;
    report.source = "quick";
    report.cache_path = cache_path;
    return report;
  }

  // ---- full scan ----
#ifdef _WIN32
  SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
  ScanCounters counters;
  counters.progress = opts.progress;
  counters.t0 = std::chrono::steady_clock::now();
  counters.last_progress = counters.t0;
  std::vector<std::wstring> roots = full_scan_roots();
  counters.drives_total = static_cast<int>(roots.size());
  std::string root_list;
  for (const auto& r : roots) {
    if (!root_list.empty()) root_list += ",";
    root_list += utf8_from_wide(r);
  }
  scan_log(counters, "start drives=%s depth=%d min_size_mb=%d", root_list.c_str(),
           opts.max_depth, static_cast<int>(opts.min_size_bytes >> 20));
  std::vector<FullCandidate> cands;
  std::mutex cand_mu;
  // Explicit --model-dir entries are scanned first and keep priority on dedupe.
  {
    std::vector<FullCandidate> explicit_cands;
    std::vector<std::string> scanned;
    scan_explicit_dirs(opts.extra_dirs, opts.min_size_bytes, nullptr, &scanned, nullptr);
    // We need candidates, not merged entries, so redo the top-level walk here.
    for (const std::string& raw : opts.extra_dirs) {
      const std::string dir = normalized_dir(raw);
      std::error_code ec;
      if (!fs::is_directory(dir, ec)) continue;
      std::error_code iec;
      for (fs::directory_iterator it(dir, iec), end; !iec && it != end; it.increment(iec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        const std::string name = it->path().filename().string();
        if (!is_gguf_name(name)) continue;
        const uint64_t size = static_cast<uint64_t>(it->file_size(fec));
        if (fec || size < opts.min_size_bytes) continue;
        FullCandidate c;
        c.path = it->path().string();
        c.dir = dir;
        c.name = name;
        c.drive = drive_of_path(c.path);
        c.size = size;
        explicit_cands.push_back(std::move(c));
      }
    }
    if (!explicit_cands.empty()) {
      std::lock_guard<std::mutex> lock(cand_mu);
      for (auto& c : explicit_cands) cands.push_back(std::move(c));
    }
  }
  std::vector<DriveStat> drive_stats(roots.size());
  const int n_roots = static_cast<int>(roots.size());
  const int workers = std::max(1, std::min(4, n_roots));
  std::vector<std::thread> threads;
  std::atomic<int> next_root{0};
  for (int w = 0; w < workers; ++w) {
    threads.emplace_back([&]() {
      SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
      for (;;) {
        const int i = next_root.fetch_add(1);
        if (i >= n_roots) break;
        const std::wstring& root = roots[static_cast<size_t>(i)];
        DriveStat st;
        st.root = utf8_from_wide(root);
        st.root_mtime = drive_root_mtime(root);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<FullCandidate> local;
        std::wstring scan_root = root;
        if (scan_root.size() == 3 && scan_root[1] == L':' && scan_root[2] == L'\\')
          scan_root = scan_root.substr(0, 2);
        uint64_t local_dirs = 0;
        int local_found = 0;
        walk_drive_dir(scan_root, 0, counters, opts.max_depth, opts.min_size_bytes, local,
                       &local_dirs, &local_found);
        st.dirs = static_cast<int>(local_dirs);
        st.found = local_found;
        st.ok = true;
        st.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        drive_stats[static_cast<size_t>(i)] = st;
        if (!local.empty()) {
          std::lock_guard<std::mutex> lock(cand_mu);
          for (auto& c : local) cands.push_back(std::move(c));
        }
        const int done = counters.drives_done.fetch_add(1) + 1;
        scan_log(counters, "drive %s done dirs=%d found=%d seconds=%.2f", st.root.c_str(), st.dirs,
                 st.found, st.seconds);
        scan_log(counters, "progress drives_done=%d drives_total=%d dirs=%llu found=%d elapsed_ms=%llu",
                 done, counters.drives_total,
                 static_cast<unsigned long long>(counters.dirs.load()), counters.found.load(),
                 static_cast<unsigned long long>(elapsed_ms(counters)));
      }
    });
  }
  for (auto& t : threads) t.join();
  build_entries_from_candidates(cands, &report.warnings, &report.entries);
  finalize_entries(&report.entries);
  const double total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - counters.t0).count();
  report.dirs.clear();
  for (const auto& r : roots) report.dirs.push_back(utf8_from_wide(r));
  report.drive_stats = drive_stats;
  report.dirs_scanned = static_cast<int>(counters.dirs.load());
  report.scan_seconds = total_seconds;
  report.generated_at = now_iso();
  info->source = "full";
  info->drives = drive_stats;
  info->dirs_scanned = report.dirs_scanned;
  info->generated_at = report.generated_at;
  info->scan_seconds = total_seconds;
  report.source = "full";
  report.cache_path = cache_path;
  scan_log(counters, "done drives=%d dirs=%llu found=%d elapsed_ms=%llu", counters.drives_total,
           static_cast<unsigned long long>(counters.dirs.load()), counters.found.load(),
           static_cast<unsigned long long>(elapsed_ms(counters)));
  write_cache(cache_path, report, *info, opts.max_depth, opts.min_size_bytes);
#else
  report = scan(opts.extra_dirs);
  info->source = "quick";
  report.source = "quick";
#endif
  return report;
}

Report scan_cached_or_quick(const std::vector<std::string>& extra_dirs, ScanInfo* info) {
  ScanOptions opts;
  opts.extra_dirs = extra_dirs;
  opts.max_depth = 7;
  opts.min_size_bytes = 100ull << 20;
  Report cached;
  bool stale = false;
  if (read_cache(cache_default_path(), opts.max_depth, opts.min_size_bytes, &cached, &stale, info)) {
    cached.source = "cache";
    cached.cache_hit = true;
    cached.cache_stale = stale;
    return cached;
  }
  Report r = scan(extra_dirs);
  if (info != nullptr) info->source = "quick";
  r.source = "quick";
  return r;
}

}  // namespace model_scan
}  // namespace engine

