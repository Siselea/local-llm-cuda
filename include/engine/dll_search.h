#pragma once
// Portable discovery of the NVIDIA DLLs that the engine loads at runtime.
//
// The engine deliberately does not link CUDA/cuBLAS at build time: it resolves
// nvrtc64_120_0.dll / cublas64_12.dll at runtime from (in order):
//   explicit config / env  ->  executable directory  ->  CUDA_PATH/CUDA_HOME
//   ->  pip nvidia-* wheels under LOCALAPPDATA / USERPROFILE  ->  system PATH
//
// This header only contains path helpers; the actual LoadLibrary calls stay in
// src/cuda.cpp / src/cublas.cpp.  No machine-specific paths belong here.

#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace engine::dll_search {

namespace fs = std::filesystem;

inline std::string env_value(const char* name) {
  const DWORD n = GetEnvironmentVariableA(name, nullptr, 0);
  if (n == 0) return {};
  std::string out(n, '\0');
  const DWORD got = GetEnvironmentVariableA(name, &out[0], n);
  if (got == 0) return {};
  out.resize(got);
  return out;
}

inline std::string exe_dir() {
  char buf[32768] = {0};
  const DWORD n = GetModuleFileNameA(nullptr, buf, static_cast<DWORD>(sizeof(buf)));
  if (n == 0 || n >= sizeof(buf)) return {};
  std::string path(buf, n);
  const size_t slash = path.find_last_of("\\/");
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

inline std::string join(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (a.back() == '\\' || a.back() == '/') return a + b;
  return a + "\\" + b;
}

inline std::string parent_dir(const std::string& path) {
  const size_t slash = path.find_last_of("\\/");
  return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

inline bool is_file(const std::string& path) {
  std::error_code ec;
  return fs::is_regular_file(fs::path(path), ec);
}

inline bool is_dir(const std::string& path) {
  std::error_code ec;
  return fs::is_directory(fs::path(path), ec);
}

inline void add_dir(std::vector<std::string>& out, const std::string& dir) {
  if (dir.empty() || !is_dir(dir)) return;
  if (std::find(out.begin(), out.end(), dir) == out.end()) out.push_back(dir);
}

inline std::vector<std::string> cuda_bin_dirs() {
  std::vector<std::string> out;
  for (const char* var : {"CUDA_PATH", "CUDA_HOME"}) {
    const std::string root = env_value(var);
    if (root.empty()) continue;
    add_dir(out, join(root, "bin"));
    add_dir(out, root);
  }
  return out;
}

inline std::vector<std::string> path_dirs() {
  std::vector<std::string> out;
  const std::string path = env_value("PATH");
  size_t start = 0;
  while (start <= path.size()) {
    const size_t end = path.find(';', start);
    std::string dir = path.substr(start, end == std::string::npos ? std::string::npos
                                                                  : end - start);
    if (dir.size() >= 2 && dir.front() == '"' && dir.back() == '"')
      dir = dir.substr(1, dir.size() - 2);
    add_dir(out, dir);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return out;
}

inline std::vector<std::string> python_roots() {
  std::vector<std::string> out;
  const std::string local = env_value("LOCALAPPDATA");
  const std::string profile = env_value("USERPROFILE");
  const std::string roaming = env_value("APPDATA");
  auto add = [&](const std::string& root) { add_dir(out, root); };

  add(join(local, "Programs\\Python"));
  add(join(profile, "AppData\\Local\\Programs\\Python"));
  add(join(roaming, "Python"));
  add(join(profile, "AppData\\Roaming\\Python"));
  add(join(profile, "miniconda3"));
  add(join(profile, "anaconda3"));
  add(join(local, "miniconda3"));
  add(join(local, "anaconda3"));
  add(join(profile, "AppData\\Local\\miniconda3"));
  add(join(profile, "AppData\\Local\\anaconda3"));
  // Python on PATH (covers custom/portable install roots without hard-coding
  // them; the developer machine's pip wheels are found this way).
  for (const std::string& dir : path_dirs()) {
    if (is_file(join(dir, "python.exe"))) add(dir);
  }
  // PYTHONHOME points at an installation prefix.
  add(env_value("PYTHONHOME"));
  return out;
}

// site-packages directories below one Python/conda root (root itself, Python3*,
// and conda envs/*).  Kept shallow on purpose: a bounded fixed layout is much
// faster than scanning the whole home directory.
inline std::vector<std::string> site_packages_dirs(const std::string& root) {
  std::vector<std::string> out;
  auto add = [&](const std::string& site) { add_dir(out, site); };
  add(join(root, "Lib\\site-packages"));
  add(join(root, "site-packages"));
  std::error_code ec;
  fs::directory_iterator it(fs::path(root), ec);
  if (ec) return out;
  int entries = 0;
  std::vector<std::string> envs;
  for (const auto& entry : it) {
    if (++entries > 256) break;
    std::error_code e;
    if (!entry.is_directory(e)) continue;
    const std::string name = entry.path().filename().string();
    if (name.rfind("Python3", 0) == 0) {
      add(join(entry.path().string(), "Lib\\site-packages"));
      add(join(entry.path().string(), "site-packages"));
    } else if (name == "envs") {
      envs.push_back(entry.path().string());
    }
  }
  for (const std::string& envs_dir : envs) {
    std::error_code e;
    fs::directory_iterator env_it(fs::path(envs_dir), e);
    if (e) continue;
    int env_count = 0;
    for (const auto& entry : env_it) {
      if (++env_count > 64) break;
      std::error_code e2;
      if (!entry.is_directory(e2)) continue;
      add(join(entry.path().string(), "Lib\\site-packages"));
    }
  }
  return out;
}

// Pip wheel layout: <site-packages>\nvidia\<package>\<bin>.
inline std::vector<std::string> pip_nvidia_bin_dirs(const std::string& package,
                                                    const std::string& bin_subdir) {
  std::vector<std::string> out;
  for (const std::string& root : python_roots()) {
    for (const std::string& site : site_packages_dirs(root)) {
      add_dir(out, join(site, join("nvidia", join(package, bin_subdir))));
    }
  }
  return out;
}

inline std::string dll_path(const std::string& value, const std::string& filename) {
  if (value.empty()) return {};
  if (is_dir(value)) return join(value, filename);
  if (value.size() >= 4 && value.compare(value.size() - 4, 4, ".dll") == 0) return value;
  return join(value, filename);
}

inline std::string find_in_dirs(const std::vector<std::string>& dirs,
                                const std::string& filename) {
  for (const std::string& dir : dirs) {
    const std::string candidate = join(dir, filename);
    if (is_file(candidate)) return candidate;
  }
  return {};
}

inline std::string join_lines(const std::vector<std::string>& values) {
  std::string out;
  for (const std::string& v : values) {
    out += "\n  - ";
    out += v;
  }
  return out;
}

}  // namespace engine::dll_search
