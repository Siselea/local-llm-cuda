// measure_utilization.exe -- external GPU/CPU utilisation sampler for one child process.
//
// Why a C++ NVML sampler instead of `nvidia-smi --loop-ms`:
//   * one process, one NVML init => per-sample cost is a few microseconds, so a 10 ms
//     cadence is actually achievable (subprocess nvidia-smi costs ~50-100 ms each);
//   * NVML gives GPU utilisation + power + SM/mem clocks + temperature from the same
//     instant, so a "bubble" can be cross-checked with power/clock, not only util;
//   * the child's CPU time (GetProcessTimes) and the machine-wide CPU time
//     (GetSystemTimes) are sampled in the same loop, giving the process CPU% that the
//     engine's own profiler cannot report;
//   * every line the child prints gets a timestamp, so the engine's own [time]/[fwd]/
//     [layerms] markers can be aligned with the GPU timeline by the Python analyser.
//
// It never touches the engine binary: it spawns it with inherited stdout/stderr pipes,
// samples, and writes three files:
//   <prefix>.timeline.csv  t_ms,gpu_util,gpu_mem_used_mib,gpu_power_w,sm_mhz,mem_mhz,
//                          gpu_temp_c,pstate,sys_cpu_busy_pct,proc_cpu_pct,proc_cpu_ms,dt_ms
//   <prefix>.engine.log     "<t_ms>\t<stream>\t<line>"
//   <prefix>.meta.json     command, exit code, wall, actual sampling interval, GPU info,
//                          foreign-process checks at start/end (+ during the run), and
//                          whether the GPU/machine was idle when the child was started.
//
// Build (from the repo root, same toolchain as the engine):
//   python -m ziglang c++ -std=c++17 -O2 tools/measure_utilization.cpp \
//       -o tools/util_bin/measure_utilization.exe
//
// Usage:
//   tools/util_bin/measure_utilization.exe --out-prefix tools/util_results/qwen9b/graph \
//       --wait-gpu-idle-sec 600 -- ./build/test_gen.exe <model> "760" 33
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ----------------------------------------------------------------------------------
// NVML (loaded dynamically from nvml.dll -- shipped with the display driver, not the
// CUDA toolkit, so it is present on this machine).
// ----------------------------------------------------------------------------------
namespace nvml {

typedef void* Device;

struct Utilization {
  unsigned int gpu;
  unsigned int memory;
};
struct Memory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct Api {
  HMODULE lib = nullptr;
  int (*Init)(void) = nullptr;
  int (*Shutdown)(void) = nullptr;
  int (*DeviceGetHandleByIndex)(unsigned int, Device*) = nullptr;
  int (*DeviceGetName)(Device, char*, unsigned int) = nullptr;
  int (*DeviceGetUtilizationRates)(Device, Utilization*) = nullptr;
  int (*DeviceGetMemoryInfo)(Device, Memory*) = nullptr;
  int (*DeviceGetPowerUsage)(Device, unsigned int*) = nullptr;
  int (*DeviceGetClockInfo)(Device, int, unsigned int*) = nullptr;
  int (*DeviceGetTemperature)(Device, int, unsigned int*) = nullptr;
  int (*DeviceGetPerformanceState)(Device, int*) = nullptr;
  int (*SystemGetDriverVersion)(char*, unsigned int) = nullptr;
  int (*SystemGetNVMLVersion)(char*, unsigned int) = nullptr;
};

enum { CLOCK_SM = 1, CLOCK_MEM = 2, TEMP_GPU = 0 };

bool load(Api& a) {
  a.lib = LoadLibraryA("nvml.dll");
  if (!a.lib) a.lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
  if (!a.lib) return false;
  auto sym = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(a.lib, n)); };
  a.Init = reinterpret_cast<decltype(a.Init)>(sym("nvmlInit_v2"));
  if (!a.Init) a.Init = reinterpret_cast<decltype(a.Init)>(sym("nvmlInit"));
  a.Shutdown = reinterpret_cast<decltype(a.Shutdown)>(sym("nvmlShutdown"));
  a.DeviceGetHandleByIndex = reinterpret_cast<decltype(a.DeviceGetHandleByIndex)>(
      sym("nvmlDeviceGetHandleByIndex_v2"));
  if (!a.DeviceGetHandleByIndex)
    a.DeviceGetHandleByIndex =
        reinterpret_cast<decltype(a.DeviceGetHandleByIndex)>(sym("nvmlDeviceGetHandleByIndex"));
  a.DeviceGetName = reinterpret_cast<decltype(a.DeviceGetName)>(sym("nvmlDeviceGetName"));
  a.DeviceGetUtilizationRates = reinterpret_cast<decltype(a.DeviceGetUtilizationRates)>(
      sym("nvmlDeviceGetUtilizationRates"));
  a.DeviceGetMemoryInfo =
      reinterpret_cast<decltype(a.DeviceGetMemoryInfo)>(sym("nvmlDeviceGetMemoryInfo"));
  a.DeviceGetPowerUsage =
      reinterpret_cast<decltype(a.DeviceGetPowerUsage)>(sym("nvmlDeviceGetPowerUsage"));
  a.DeviceGetClockInfo =
      reinterpret_cast<decltype(a.DeviceGetClockInfo)>(sym("nvmlDeviceGetClockInfo"));
  a.DeviceGetTemperature =
      reinterpret_cast<decltype(a.DeviceGetTemperature)>(sym("nvmlDeviceGetTemperature"));
  a.DeviceGetPerformanceState = reinterpret_cast<decltype(a.DeviceGetPerformanceState)>(
      sym("nvmlDeviceGetPerformanceState"));
  a.SystemGetDriverVersion = reinterpret_cast<decltype(a.SystemGetDriverVersion)>(
      sym("nvmlSystemGetDriverVersion"));
  a.SystemGetNVMLVersion = reinterpret_cast<decltype(a.SystemGetNVMLVersion)>(
      sym("nvmlSystemGetNVMLVersion"));
  return a.Init && a.Shutdown && a.DeviceGetHandleByIndex && a.DeviceGetUtilizationRates &&
         a.DeviceGetMemoryInfo;
}

}  // namespace nvml

namespace {

double qpc_freq = 0.0;
double now_ms(const LARGE_INTEGER& t0) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return 1000.0 * static_cast<double>(t.QuadPart - t0.QuadPart) / qpc_freq;
}

// ---- winmm timer resolution (loaded dynamically: zig/mingw does not auto-link winmm) --
typedef UINT(WINAPI* TimeBeginPeriodFn)(UINT);
typedef UINT(WINAPI* TimeEndPeriodFn)(UINT);
TimeBeginPeriodFn g_timeBegin = nullptr;
TimeEndPeriodFn g_timeEnd = nullptr;
void raise_timer_resolution() {
  HMODULE m = LoadLibraryA("winmm.dll");
  if (!m) return;
  g_timeBegin = reinterpret_cast<TimeBeginPeriodFn>(GetProcAddress(m, "timeBeginPeriod"));
  g_timeEnd = reinterpret_cast<TimeEndPeriodFn>(GetProcAddress(m, "timeEndPeriod"));
  if (g_timeBegin) g_timeBegin(1);
}
void restore_timer_resolution() {
  if (g_timeEnd) g_timeEnd(1);
}

struct SysCpu {
  unsigned long long idle = 0, kernel = 0, user = 0;
};
SysCpu read_sys_cpu() {
  SysCpu c;
  FILETIME i, k, u;
  GetSystemTimes(&i, &k, &u);
  auto to64 = [](const FILETIME& f) {
    return (static_cast<unsigned long long>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
  };
  c.idle = to64(i);
  c.kernel = to64(k);
  c.user = to64(u);
  return c;
}
double sys_busy_pct(const SysCpu& a, const SysCpu& b) {
  const double idle = static_cast<double>(b.idle - a.idle);
  const double total = static_cast<double>((b.kernel - a.kernel) + (b.user - a.user));
  if (total <= 0) return 0.0;
  return 100.0 * (1.0 - idle / total);
}
unsigned long long filetime_100ns(const FILETIME& f) {
  return (static_cast<unsigned long long>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
}
// Returns {"kernel","user"} 100ns ticks of a process.
bool proc_cpu_ticks(HANDLE p, unsigned long long* kernel, unsigned long long* user) {
  FILETIME c, e, k, u;
  if (!GetProcessTimes(p, &c, &e, &k, &u)) return false;
  *kernel = filetime_100ns(k);
  *user = filetime_100ns(u);
  return true;
}

struct ProcInfo {
  unsigned pid = 0;
  std::string name;
};

// Query compute apps through nvidia-smi -- used only around the run, never per sample.
bool query_gpu_apps(std::vector<ProcInfo>* out, std::string* raw, int* exit_code) {
  out->clear();
  raw->clear();
  FILE* f = _popen(
      "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>&1", "r");
  if (!f) {
    if (exit_code) *exit_code = -1;
    return false;
  }
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), f)) {
    std::string line(buf);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) continue;
    raw->append(line).append("\n");
    const size_t comma = line.find(',');
    if (comma != std::string::npos) {
      ProcInfo p;
      p.pid = static_cast<unsigned>(std::atoi(line.substr(0, comma).c_str()));
      p.name = line.substr(comma + 1);
      while (!p.name.empty() && (p.name.front() == ' ' || p.name.front() == '\t'))
        p.name.erase(p.name.begin());
      out->push_back(p);
    }
  }
  const int rc = _pclose(f);
  if (exit_code) *exit_code = rc;
  return true;
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
  return s;
}

struct ForeignCheck {
  int count = 0;
  std::string pids;
  std::string raw;
  int rc = 0;
};

// A "foreign compute process" is any compute app whose executable name matches one of the
// patterns (default: any test_gen*/engine*/llama* binary) and is not our child.
ForeignCheck check_foreign(const std::vector<std::string>& patterns, unsigned child_pid) {
  ForeignCheck fc;
  std::vector<ProcInfo> apps;
  std::string raw;
  int rc = 0;
  if (!query_gpu_apps(&apps, &raw, &rc)) {
    fc.rc = rc;
    fc.raw = "(nvidia-smi failed)";
    return fc;
  }
  fc.rc = rc;
  fc.raw = raw;
  for (const auto& a : apps) {
    if (a.pid == child_pid) continue;
    const std::string nm = lower(a.name);
    bool match = false;
    for (const auto& p : patterns) {
      if (!p.empty() && nm.find(lower(p)) != std::string::npos) match = true;
    }
    if (match) {
      ++fc.count;
      fc.pids.append(std::to_string(a.pid)).append(" ");
    }
  }
  return fc;
}

struct Options {
  std::string out_prefix = "measure";
  int interval_ms = 10;
  int wait_idle_sec = 0;       // 0 = do not wait
  double idle_mem_max_mb = 1500.0;
  double idle_util_max = 60.0;
  double cpu_idle_max_pct = 25.0;
  int during_check_ms = 5000;  // 0 = never check during the run
  int timeout_ms = 0;          // 0 = no timeout; otherwise TerminateProcess the child
  std::string child_cwd;
  std::vector<std::string> foreign_patterns{"test_gen", "engine.exe", "llama"};
  std::vector<std::string> cmd;
};

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default: o += c;
    }
  }
  return o;
}

std::wstring widen(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(static_cast<size_t>(n > 0 ? n - 1 : 0), L'\0');
  if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
  return w;
}

std::string quote_arg(const std::string& a) {
  if (a.find_first_of(" \t\"") == std::string::npos) return a;
  std::string o = "\"";
  for (char c : a) {
    if (c == '"') o += "\\\"";
    else o += c;
  }
  o += "\"";
  return o;
}

// Rolling reader: one thread drains the child's stdout+stderr pipe and timestamps lines.
struct Logger {
  HANDLE read_pipe = nullptr;
  FILE* log = nullptr;
  const LARGE_INTEGER* t0 = nullptr;
  std::thread th;
  long long lines = 0;
  void run() {
    char buf[8192];
    std::string carry;
    DWORD got = 0;
    while (ReadFile(read_pipe, buf, sizeof(buf) - 1, &got, nullptr) && got > 0) {
      buf[got] = 0;
      carry.append(buf, got);
      size_t pos = 0;
      while (true) {
        const size_t nl = carry.find('\n', pos);
        if (nl == std::string::npos) break;
        const std::string line = carry.substr(pos, nl - pos);
        pos = nl + 1;
        const double t = now_ms(*t0);
        std::fprintf(log, "%.3f\terr\t%s\n", t, line.c_str());
        ++lines;
      }
      carry.erase(0, pos);
      std::fflush(log);
    }
    if (!carry.empty()) {
      std::fprintf(log, "%.3f\terr\t%s\n", now_ms(*t0), carry.c_str());
      ++lines;
      std::fflush(log);
    }
  }
};

struct Sample {
  double t = 0;
  int util = -1;
  double mem_mb = -1;
  double power_w = -1;
  int sm_mhz = -1;
  int mem_mhz = -1;
  int temp_c = -1;
  int pstate = -1;
  double sys_busy = -1;
  double proc_pct = -1;
  double proc_cpu_ms = -1;
  double dt = 0;
};

void write_usage() {
  std::printf(
      "usage: measure_utilization.exe --out-prefix <path> [options] -- <command...>\n"
      "  --interval-ms N        sampling cadence (default 10)\n"
      "  --wait-gpu-idle-sec N  wait up to N s for a quiet GPU/CPU before starting (default 0)\n"
      "  --idle-mem-max-mb N    quiet GPU = used VRAM below this (default 1500)\n"
      "  --idle-util-max N      quiet GPU = nvml util below this %% (default 60; the\n"
      "                         Windows desktop keeps an ambient ~25%% busy)\n"
      "  --cpu-idle-max N       quiet machine = system CPU below this %% (default 25)\n"
      "  --during-check-ms N    foreign-process check cadence during run, 0=off (default 5000)\n"
      "  --timeout-ms N         kill the child after N ms (0 = never; a hung engine keeps the\n"
      "                         GPU/VRAM allocated, partial samples are still written)\n"
      "  --foreign-name PAT     executable substring counted as a foreign engine (repeatable)\n"
      "  --child-cwd DIR        working directory of the child\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  bool saw_ddash = false;
  std::vector<std::string> foreign_user;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: missing value for %s\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (saw_ddash) {
      opt.cmd.push_back(a);
    } else if (a == "--") {
      saw_ddash = true;
    } else if (a == "--out-prefix") opt.out_prefix = need("--out-prefix");
    else if (a == "--interval-ms") opt.interval_ms = std::atoi(need("--interval-ms").c_str());
    else if (a == "--wait-gpu-idle-sec") opt.wait_idle_sec = std::atoi(need("--wait-gpu-idle-sec").c_str());
    else if (a == "--idle-mem-max-mb") opt.idle_mem_max_mb = std::atof(need("--idle-mem-max-mb").c_str());
    else if (a == "--idle-util-max") opt.idle_util_max = std::atof(need("--idle-util-max").c_str());
    else if (a == "--cpu-idle-max") opt.cpu_idle_max_pct = std::atof(need("--cpu-idle-max").c_str());
    else if (a == "--during-check-ms") opt.during_check_ms = std::atoi(need("--during-check-ms").c_str());
    else if (a == "--timeout-ms") opt.timeout_ms = std::atoi(need("--timeout-ms").c_str());
    else if (a == "--foreign-name") foreign_user.push_back(need("--foreign-name"));
    else if (a == "--child-cwd") opt.child_cwd = need("--child-cwd");
    else if (a == "-h" || a == "--help") {
      write_usage();
      return 0;
    } else {
      std::fprintf(stderr, "error: unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (opt.cmd.empty()) {
    write_usage();
    return 2;
  }
  for (const auto& f : foreign_user) opt.foreign_patterns.push_back(f);

  if (opt.interval_ms < 1) opt.interval_ms = 1;

  // np: outputs live next to the prefix; create parent dirs is left to the caller.
  const std::string csv_path = opt.out_prefix + ".timeline.csv";
  const std::string log_path = opt.out_prefix + ".engine.log";
  const std::string meta_path = opt.out_prefix + ".meta.json";
  const std::string foreign_start_path = opt.out_prefix + ".foreign-start.txt";
  const std::string foreign_end_path = opt.out_prefix + ".foreign-end.txt";
  const std::string foreign_during_path = opt.out_prefix + ".foreign-during.txt";

  // ---- GPU info ----
  nvml::Api nv;
  bool have_nvml = nvml::load(nv);
  nvml::Device dev = nullptr;
  std::string gpu_name = "(none)";
  std::string driver = "";
  if (have_nvml) {
    if (nv.Init() != 0) have_nvml = false;
  }
  if (have_nvml) {
    if (nv.DeviceGetHandleByIndex(0, &dev) != 0) have_nvml = false;
  }
  if (have_nvml) {
    char nm[128] = {0};
    if (nv.DeviceGetName(dev, nm, sizeof(nm)) == 0) gpu_name = nm;
    char dv[128] = {0};
    if (nv.SystemGetDriverVersion && nv.SystemGetDriverVersion(dv, sizeof(dv)) == 0) driver = dv;
  }

  raise_timer_resolution();
  LARGE_INTEGER t0;
  QueryPerformanceCounter(&t0);
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  qpc_freq = static_cast<double>(freq.QuadPart);
  if (qpc_freq <= 0) qpc_freq = 1e7;

  auto gpu_read = [&](Sample& s) {
    if (!have_nvml) return;
    nvml::Utilization u{0, 0};
    if (nv.DeviceGetUtilizationRates(dev, &u) == 0) {
      s.util = static_cast<int>(u.gpu);
    }
    nvml::Memory m{0, 0, 0};
    if (nv.DeviceGetMemoryInfo(dev, &m) == 0) {
      s.mem_mb = static_cast<double>(m.used) / 1048576.0;
    }
    unsigned v = 0;
    if (nv.DeviceGetPowerUsage && nv.DeviceGetPowerUsage(dev, &v) == 0)
      s.power_w = static_cast<double>(v) / 1000.0;
    if (nv.DeviceGetClockInfo && nv.DeviceGetClockInfo(dev, nvml::CLOCK_SM, &v) == 0)
      s.sm_mhz = static_cast<int>(v);
    if (nv.DeviceGetClockInfo && nv.DeviceGetClockInfo(dev, nvml::CLOCK_MEM, &v) == 0)
      s.mem_mhz = static_cast<int>(v);
    if (nv.DeviceGetTemperature && nv.DeviceGetTemperature(dev, nvml::TEMP_GPU, &v) == 0)
      s.temp_c = static_cast<int>(v);
    int ps = -1;
    if (nv.DeviceGetPerformanceState && nv.DeviceGetPerformanceState(dev, &ps) == 0)
      s.pstate = ps;
  };

  // ---- wait for a quiet machine (another agent shares this GPU/CPU) ----
  auto idle_wait_start = std::chrono::steady_clock::now();
  int idle_consecutive = 0;
  bool started_quiet = true;
  std::string idle_wait_note = "no wait requested";
  ForeignCheck pre_fc;
  if (opt.wait_idle_sec > 0) {
    started_quiet = false;
    int waited = 0;
    int polls = 0;
    SysCpu c0 = read_sys_cpu();
    while (waited < opt.wait_idle_sec * 1000) {
      Sample s;
      gpu_read(s);
      SysCpu c1 = read_sys_cpu();
      Sleep(250);
      SysCpu c2 = read_sys_cpu();
      const double busy = sys_busy_pct(c1, c2);
      ForeignCheck fc = check_foreign(opt.foreign_patterns, 0);
      const bool gpu_ok = s.util >= 0 && s.util < opt.idle_util_max && s.mem_mb < opt.idle_mem_max_mb;
      const bool cpu_ok = busy < opt.cpu_idle_max_pct;
      // a heavy foreign engine may be in a CPU-only phase: the pattern check still catches it
      const bool foreign_ok = fc.count == 0 && fc.rc == 0;
      if (gpu_ok && cpu_ok && foreign_ok) {
        ++idle_consecutive;
        if (idle_consecutive >= 3) {
          started_quiet = true;
          idle_wait_note = "quiet for 3 consecutive checks";
          break;
        }
      } else {
        idle_consecutive = 0;
      }
      waited += 250 + 250;
      ++polls;
      if (polls % 8 == 0) {
        std::printf(
            "[measure] waiting (waited %.0fs): gpu_util=%d gpu_mem=%.0fMiB sys_cpu=%.1f%% "
            "foreign=%d\n",
            waited / 1000.0, s.util, s.mem_mb, busy, fc.count);
        std::fflush(stdout);
      }
    }
    if (!started_quiet) idle_wait_note = "timed out waiting for a quiet machine";
    pre_fc = check_foreign(opt.foreign_patterns, 0);
  } else {
    pre_fc = check_foreign(opt.foreign_patterns, 0);
  }
  {
    FILE* f = std::fopen(foreign_start_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "# quiet_before_start=%d (%s)\n", started_quiet ? 1 : 0,
                   idle_wait_note.c_str());
      std::fprintf(f, "# waited_ms=%.0f\n",
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - idle_wait_start)
                       .count());
      std::fprintf(f, "%s", pre_fc.raw.c_str());
      std::fclose(f);
    }
  }
  std::printf("[measure] gpu=%s driver=%s nvml=%d quiet_before_start=%d (%s)\n", gpu_name.c_str(),
              driver.c_str(), have_nvml ? 1 : 0, started_quiet ? 1 : 0, idle_wait_note.c_str());

  // ---- ambient baseline (desktop compositor / CEF apps keep the GPU lightly busy;
  //      recording it lets the analyser report engine-attributable numbers) ----
  double base_util = 0.0, base_power = 0.0, base_sys = 0.0;
  int base_n = 0;
  {
    SysCpu sb0 = read_sys_cpu();
    double sum_util = 0.0, sum_power = 0.0;
    int n = 0;
    for (int i = 0; i < 20; ++i) {
      Sample s;
      gpu_read(s);
      if (s.util >= 0) { sum_util += s.util; ++n; }
      if (s.power_w >= 0) sum_power += s.power_w;
      Sleep(100);
    }
    SysCpu sb1 = read_sys_cpu();
    if (n > 0) { base_util = sum_util / n; base_power = sum_power / n; base_n = n; }
    base_sys = sys_busy_pct(sb0, sb1);
    std::printf("[measure] ambient baseline: gpu_util %.1f%%, power %.1f W, sys_cpu %.1f%% over %.1fs\n",
                base_util, base_power, base_sys, n * 0.1);
    std::fflush(stdout);
  }

  // ---- spawn the child ----
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = nullptr;
  sa.bInheritHandle = TRUE;
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 1 << 20)) {
    std::fprintf(stderr, "error: CreatePipe failed %lu\n", GetLastError());
    return 3;
  }
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

  std::string cmdline;
  for (const auto& a : opt.cmd) cmdline += (cmdline.empty() ? "" : " ") + quote_arg(a);

  STARTUPINFOW si;
  std::memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdOutput = wr;
  si.hStdError = wr;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION pi;
  std::memset(&pi, 0, sizeof(pi));
  std::wstring wcmd = widen(cmdline);
  std::vector<wchar_t> mutable_cmd(wcmd.begin(), wcmd.end());
  mutable_cmd.push_back(L'\0');
  std::wstring wcwd = widen(opt.child_cwd);
  LARGE_INTEGER child_t0;
  QueryPerformanceCounter(&child_t0);
  const BOOL ok = CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE,
                                 CREATE_NO_WINDOW, nullptr,
                                 opt.child_cwd.empty() ? nullptr : wcwd.c_str(), &si, &pi);
  CloseHandle(wr);
  if (!ok) {
    std::fprintf(stderr, "error: CreateProcess failed %lu for [%s]\n", GetLastError(),
                 cmdline.c_str());
    CloseHandle(rd);
    return 3;
  }

  // ---- CSV + log ----
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "error: cannot write %s\n", csv_path.c_str());
    TerminateProcess(pi.hProcess, 1);
    return 3;
  }
  std::fprintf(csv,
               "t_ms,gpu_util,gpu_mem_used_mib,gpu_power_w,sm_mhz,mem_mhz,gpu_temp_c,pstate,"
               "sys_cpu_busy_pct,proc_cpu_pct,proc_cpu_ms,dt_ms\n");
  FILE* log = std::fopen(log_path.c_str(), "w");
  if (!log) {
    std::fprintf(stderr, "error: cannot write %s\n", log_path.c_str());
    TerminateProcess(pi.hProcess, 1);
    return 3;
  }
  Logger logger;
  logger.read_pipe = rd;
  logger.log = log;
  logger.t0 = &child_t0;
  logger.th = std::thread([&logger]() { logger.run(); });

  // ---- sampling loop ----
  SysCpu prev_sys = read_sys_cpu();
  unsigned long long prev_k = 0, prev_u = 0;
  proc_cpu_ticks(pi.hProcess, &prev_k, &prev_u);
  double prev_t = 0.0;
  long long samples = 0, missed_foreign = 0;
  long long foreign_seen = 0;
  FILE* fduring = std::fopen(foreign_during_path.c_str(), "w");
  if (fduring) std::fprintf(fduring, "# t_ms\tforeign_count\tpids\n");
  std::string during_foreign_pids;
  double min_dt = 1e9, max_dt = 0.0, sum_dt = 0.0;
  int next_foreign_check = 0;
  auto next_sample_due = std::chrono::steady_clock::now();
  bool running = true;
  bool timed_out = false;
  while (running) {
    const DWORD wr2 = WaitForSingleObject(pi.hProcess, 0);
    running = (wr2 == WAIT_TIMEOUT);
    Sample s;
    s.t = now_ms(child_t0);
    s.dt = s.t - prev_t;
    gpu_read(s);
    SysCpu sc = read_sys_cpu();
    s.sys_busy = sys_busy_pct(prev_sys, sc);
    prev_sys = sc;
    unsigned long long k = 0, u = 0;
    if (proc_cpu_ticks(pi.hProcess, &k, &u)) {
      s.proc_cpu_ms = static_cast<double>((k - prev_k) + (u - prev_u)) / 10000.0;
      if (s.dt > 0.1) s.proc_pct = 100.0 * s.proc_cpu_ms / s.dt;
      prev_k = k;
      prev_u = u;
    }
    std::fprintf(csv,
                 "%.3f,%d,%.1f,%.2f,%d,%d,%d,%d,%.1f,%.1f,%.3f,%.3f\n", s.t, s.util, s.mem_mb,
                 s.power_w, s.sm_mhz, s.mem_mhz, s.temp_c, s.pstate, s.sys_busy, s.proc_pct,
                 s.proc_cpu_ms, s.dt);
    ++samples;
    if (s.t > 100) {  // ignore the first samples (process startup noise)
      min_dt = std::min(min_dt, s.dt);
      max_dt = std::max(max_dt, s.dt);
      sum_dt += s.dt;
    }
    prev_t = s.t;

    if (opt.during_check_ms > 0 && static_cast<int>(s.t) >= next_foreign_check) {
      next_foreign_check = static_cast<int>(s.t) + opt.during_check_ms;
      ForeignCheck fc = check_foreign(opt.foreign_patterns, pi.dwProcessId);
      if (fduring) std::fprintf(fduring, "%.0f\t%d\t%s\n", s.t, fc.count, fc.pids.c_str());
      if (fc.count > 0) {
        during_foreign_pids = fc.pids;
        ++foreign_seen;
      }
      ++missed_foreign;
    }
    if (!running) break;
    if (opt.timeout_ms > 0 && s.t > static_cast<double>(opt.timeout_ms)) {
      std::printf("[measure] timeout after %.0f ms -> TerminateProcess\n", s.t);
      TerminateProcess(pi.hProcess, 124);
      timed_out = true;
      running = false;
      break;
    }
    next_sample_due += std::chrono::milliseconds(opt.interval_ms);
    auto now = std::chrono::steady_clock::now();
    const auto sleep_for = std::chrono::duration_cast<std::chrono::milliseconds>(next_sample_due - now);
    if (sleep_for.count() > 0)
      Sleep(static_cast<DWORD>(sleep_for.count()));
    else
      next_sample_due = now;  // fell behind: do not try to catch up
  }
  std::fprintf(csv, "# end child_exit_wait\n");

  DWORD exit_code = 0;
  WaitForSingleObject(pi.hProcess, 5000);
  GetExitCodeProcess(pi.hProcess, &exit_code);
  const double wall_ms = now_ms(child_t0);

  // let the reader drain the pipe (child is gone -> EOF)
  logger.th.join();
  CloseHandle(rd);
  std::fclose(log);
  std::fclose(csv);
  if (fduring) std::fclose(fduring);

  // final CPU time of the child (the process object is still alive until we close it)
  unsigned long long fk = 0, fu = 0;
  double child_cpu_ms = -1;
  if (proc_cpu_ticks(pi.hProcess, &fk, &fu))
    child_cpu_ms = static_cast<double>(fk + fu) / 10000.0;
  Sample last;
  gpu_read(last);

  ForeignCheck post_fc = check_foreign(opt.foreign_patterns, 0);
  {
    FILE* f = std::fopen(foreign_end_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "# during_run_foreign_checks=%lld hits=%lld pids=%s\n", missed_foreign,
                   foreign_seen, during_foreign_pids.c_str());
      std::fprintf(f, "%s", post_fc.raw.c_str());
      std::fclose(f);
    }
  }

  {
    FILE* f = std::fopen(meta_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "{\n");
      std::fprintf(f, "  \"command\": \"%s\",\n", json_escape(cmdline).c_str());
      std::fprintf(f, "  \"child_exit_code\": %lu,\n", static_cast<unsigned long>(exit_code));
      std::fprintf(f, "  \"wall_ms\": %.3f,\n", wall_ms);
      std::fprintf(f, "  \"child_cpu_ms\": %.3f,\n", child_cpu_ms);
      std::fprintf(f, "  \"samples\": %lld,\n", samples);
      std::fprintf(f, "  \"interval_requested_ms\": %d,\n", opt.interval_ms);
      std::fprintf(f, "  \"interval_mean_ms\": %.3f,\n",
                   samples > 1 ? wall_ms / static_cast<double>(samples) : 0.0);
      std::fprintf(f, "  \"interval_min_ms\": %.3f,\n", min_dt < 1e8 ? min_dt : 0.0);
      std::fprintf(f, "  \"interval_max_ms\": %.3f,\n", max_dt);
      std::fprintf(f, "  \"engine_log_lines\": %lld,\n", logger.lines);
      std::fprintf(f, "  \"gpu_name\": \"%s\",\n", json_escape(gpu_name).c_str());
      std::fprintf(f, "  \"driver\": \"%s\",\n", json_escape(driver).c_str());
      std::fprintf(f, "  \"quiet_before_start\": %s,\n", started_quiet ? "true" : "false");
      std::fprintf(f, "  \"quiet_wait_note\": \"%s\",\n", json_escape(idle_wait_note).c_str());
      std::fprintf(f, "  \"foreign_before_start\": %d,\n", pre_fc.count);
      std::fprintf(f, "  \"foreign_after_run\": %d,\n", post_fc.count);
      std::fprintf(f, "  \"foreign_during_run_checks\": %lld,\n", missed_foreign);
      std::fprintf(f, "  \"foreign_during_run_hits\": %lld,\n", foreign_seen);
      std::fprintf(f, "  \"foreign_during_run_pids\": \"%s\",\n",
                   json_escape(during_foreign_pids).c_str());
      std::fprintf(f, "  \"ambient_gpu_util\": %.1f,\n", base_util);
      std::fprintf(f, "  \"ambient_gpu_power_w\": %.2f,\n", base_power);
      std::fprintf(f, "  \"ambient_sys_cpu\": %.1f,\n", base_sys);
      std::fprintf(f, "  \"ambient_samples\": %d,\n", base_n);
      std::fprintf(f, "  \"timed_out\": %s,\n", timed_out ? "true" : "false");
      std::fprintf(f, "  \"gpu_end_util\": %d,\n", last.util);
      std::fprintf(f, "  \"gpu_end_mem_mb\": %.1f\n", last.mem_mb);
      std::fprintf(f, "}\n");
      std::fclose(f);
    }
  }

  std::printf(
      "[measure] done: wall %.1f ms, %lld samples (dt mean %.1f ms, max %.1f ms), child CPU "
      "%.1f ms, exit=%lu, foreign(before=%d during=%lld after=%d)\n",
      wall_ms, samples, samples > 1 ? wall_ms / static_cast<double>(samples) : 0.0, max_dt,
      child_cpu_ms, static_cast<unsigned long>(exit_code), pre_fc.count, foreign_seen,
      post_fc.count);
  std::printf("[measure] wrote %s / %s / %s\n", csv_path.c_str(), log_path.c_str(),
              meta_path.c_str());

  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  restore_timer_resolution();
  if (have_nvml) nv.Shutdown();
  return exit_code == 0 ? 0 : static_cast<int>(exit_code);
}
