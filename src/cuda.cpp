#include "engine/cuda.h"
#include "engine/config.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cuda {

namespace {

// ---------------------------------------------------------------- driver ----
using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUdeviceptr = unsigned long long;

constexpr int CUDA_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

struct Driver {
  HMODULE lib = nullptr;
  CUresult(__stdcall* Init)(unsigned) = nullptr;
  CUresult(__stdcall* DriverGetVersion)(int*) = nullptr;
  CUresult(__stdcall* DeviceGet)(CUdevice*, int) = nullptr;
  CUresult(__stdcall* DeviceGetName)(char*, int, CUdevice) = nullptr;
  CUresult(__stdcall* DeviceGetAttribute)(int*, int, CUdevice) = nullptr;
  CUresult(__stdcall* DeviceTotalMem)(size_t*, CUdevice) = nullptr;
  CUresult(__stdcall* CtxCreate)(CUcontext*, unsigned, CUdevice) = nullptr;
  CUresult(__stdcall* CtxDestroy)(CUcontext) = nullptr;
  CUresult(__stdcall* CtxSynchronize)() = nullptr;
  CUresult(__stdcall* ModuleLoadDataEx)(CUmodule*, const void*, unsigned, int*, void**) = nullptr;
  CUresult(__stdcall* ModuleUnload)(CUmodule) = nullptr;
  CUresult(__stdcall* ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult(__stdcall* MemAlloc)(CUdeviceptr*, size_t) = nullptr;
  CUresult(__stdcall* MemAllocHost)(void**, size_t) = nullptr;
  CUresult(__stdcall* MemFreeHost)(void*) = nullptr;
  CUresult(__stdcall* MemFree)(CUdeviceptr) = nullptr;
  CUresult(__stdcall* MemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
  CUresult(__stdcall* MemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
  CUresult(__stdcall* MemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, void*) = nullptr;
  CUresult(__stdcall* MemcpyDtoHAsync)(void*, CUdeviceptr, size_t, void*) = nullptr;
  CUresult(__stdcall* MemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t,
                                        void*) = nullptr;
  CUresult(__stdcall* LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, unsigned, void*, void**, void**) = nullptr;
  CUresult(__stdcall* MemGetInfo)(size_t*, size_t*) = nullptr;
  CUresult(__stdcall* EventCreate)(void**, unsigned) = nullptr;
  CUresult(__stdcall* EventDestroy)(void*) = nullptr;
  CUresult(__stdcall* EventRecord)(void*, void*) = nullptr;
  CUresult(__stdcall* EventSynchronize)(void*) = nullptr;
  CUresult(__stdcall* EventElapsedTime)(float*, void*, void*) = nullptr;
  CUresult(__stdcall* StreamCreate)(void**, unsigned) = nullptr;
  CUresult(__stdcall* StreamDestroy)(void*) = nullptr;
  CUresult(__stdcall* StreamSynchronize)(void*) = nullptr;
  CUresult(__stdcall* StreamIsCapturing)(void*, int*) = nullptr;
  CUresult(__stdcall* StreamBeginCapture)(void*, int) = nullptr;
  CUresult(__stdcall* StreamEndCapture)(void*, void**) = nullptr;
  CUresult(__stdcall* GraphInstantiate)(void**, void*, unsigned long long) = nullptr;
  CUresult(__stdcall* GraphLaunch)(void*, void*) = nullptr;
  CUresult(__stdcall* GraphDestroy)(void*) = nullptr;
  CUresult(__stdcall* GraphExecDestroy)(void*) = nullptr;
  CUresult(__stdcall* FuncSetAttribute)(CUfunction, int, int) = nullptr;

  template <typename T>
  void bind(T& fn, const char* name) {
    fn = reinterpret_cast<T>(GetProcAddress(lib, name));
    if (!fn) throw std::runtime_error(std::string("nvcuda: missing symbol ") + name);
  }
};

// ----------------------------------------------------------------- nvrtc ----
using nvrtcResult = int;
using nvrtcProgram = void*;

struct Nvrtc {
  HMODULE lib = nullptr;
  nvrtcResult(__stdcall* CreateProgram)(nvrtcProgram*, const char*, const char*, int,
                                        const char**, const char**) = nullptr;
  nvrtcResult(__stdcall* CompileProgram)(nvrtcProgram, int, const char**) = nullptr;
  nvrtcResult(__stdcall* GetPTXSize)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult(__stdcall* GetPTX)(nvrtcProgram, char*) = nullptr;
  nvrtcResult(__stdcall* GetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult(__stdcall* GetProgramLog)(nvrtcProgram, char*) = nullptr;
  nvrtcResult(__stdcall* DestroyProgram)(nvrtcProgram*) = nullptr;
};

Driver g_driver;
Nvrtc g_nvrtc;
CUcontext g_ctx = nullptr;
CUdevice g_device = -1;
DeviceInfo g_info;
bool g_ready = false;

std::string default_nvrtc_path() {
  // config key `nvrtc_path` (env DSH_NVRTC_PATH is the compatibility layer).
  const std::string cfg_path = engine::config::get_string("nvrtc_path");
  if (!cfg_path.empty()) return cfg_path;
  return "D:\\p0\\py312\\Lib\\site-packages\\nvidia\\cuda_nvrtc\\bin\\nvrtc64_120_0.dll";
}

const char* g_err = nullptr;

void check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* name = "(see cuGetErrorString)";
  void* fn = nullptr;
  if (g_driver.lib) {
    // cuGetErrorString is exported by the driver
    using Fn = CUresult(__stdcall*)(CUresult, const char**);
    fn = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(g_driver.lib),
                                                "cuGetErrorString"));
  }
  if (fn) {
    const char* s2 = nullptr;
    reinterpret_cast<CUresult(__stdcall*)(CUresult, const char**)>(fn)(r, &s2);
    if (s2) name = s2;
  }
  throw std::runtime_error(std::string(what) + " failed (CUresult=" + std::to_string(r) +
                           ", " + name + ")");
}

void load_driver() {
  g_driver.lib = LoadLibraryA("nvcuda.dll");
  if (!g_driver.lib)
    throw std::runtime_error("nvcuda.dll not found: is the NVIDIA driver installed?");
  g_driver.bind(g_driver.Init, "cuInit");
  g_driver.bind(g_driver.DriverGetVersion, "cuDriverGetVersion");
  g_driver.bind(g_driver.DeviceGet, "cuDeviceGet");
  g_driver.bind(g_driver.DeviceGetName, "cuDeviceGetName");
  g_driver.bind(g_driver.DeviceGetAttribute, "cuDeviceGetAttribute");
  // versioned entry points
  g_driver.DeviceTotalMem = reinterpret_cast<decltype(g_driver.DeviceTotalMem)>(
      GetProcAddress(g_driver.lib, "cuDeviceTotalMem_v2"));
  g_driver.CtxCreate = reinterpret_cast<decltype(g_driver.CtxCreate)>(
      GetProcAddress(g_driver.lib, "cuCtxCreate_v2"));
  g_driver.CtxDestroy = reinterpret_cast<decltype(g_driver.CtxDestroy)>(
      GetProcAddress(g_driver.lib, "cuCtxDestroy_v2"));
  g_driver.MemAlloc = reinterpret_cast<decltype(g_driver.MemAlloc)>(
      GetProcAddress(g_driver.lib, "cuMemAlloc_v2"));
  g_driver.MemAllocHost = reinterpret_cast<decltype(g_driver.MemAllocHost)>(
      GetProcAddress(g_driver.lib, "cuMemAllocHost_v2"));
  g_driver.MemFreeHost = reinterpret_cast<decltype(g_driver.MemFreeHost)>(
      GetProcAddress(g_driver.lib, "cuMemFreeHost"));
  g_driver.MemFree = reinterpret_cast<decltype(g_driver.MemFree)>(
      GetProcAddress(g_driver.lib, "cuMemFree_v2"));
  g_driver.MemcpyHtoD = reinterpret_cast<decltype(g_driver.MemcpyHtoD)>(
      GetProcAddress(g_driver.lib, "cuMemcpyHtoD_v2"));
  g_driver.MemcpyHtoDAsync = reinterpret_cast<decltype(g_driver.MemcpyHtoDAsync)>(
      GetProcAddress(g_driver.lib, "cuMemcpyHtoDAsync_v2"));
  g_driver.MemcpyDtoHAsync = reinterpret_cast<decltype(g_driver.MemcpyDtoHAsync)>(
      GetProcAddress(g_driver.lib, "cuMemcpyDtoHAsync_v2"));
  g_driver.MemcpyDtoH = reinterpret_cast<decltype(g_driver.MemcpyDtoH)>(
      GetProcAddress(g_driver.lib, "cuMemcpyDtoH_v2"));
  g_driver.MemcpyDtoDAsync = reinterpret_cast<decltype(g_driver.MemcpyDtoDAsync)>(
      GetProcAddress(g_driver.lib, "cuMemcpyDtoDAsync_v2"));
  g_driver.bind(g_driver.CtxSynchronize, "cuCtxSynchronize");
  g_driver.bind(g_driver.ModuleLoadDataEx, "cuModuleLoadDataEx");
  g_driver.bind(g_driver.ModuleUnload, "cuModuleUnload");
  g_driver.bind(g_driver.ModuleGetFunction, "cuModuleGetFunction");
  g_driver.bind(g_driver.LaunchKernel, "cuLaunchKernel");
  g_driver.bind(g_driver.MemGetInfo, "cuMemGetInfo_v2");
  g_driver.bind(g_driver.EventCreate, "cuEventCreate");
  g_driver.bind(g_driver.EventDestroy, "cuEventDestroy");
  g_driver.bind(g_driver.EventRecord, "cuEventRecord");
  g_driver.bind(g_driver.EventSynchronize, "cuEventSynchronize");
  g_driver.bind(g_driver.EventElapsedTime, "cuEventElapsedTime");
  g_driver.bind(g_driver.StreamCreate, "cuStreamCreate");
  g_driver.bind(g_driver.StreamDestroy, "cuStreamDestroy");
  g_driver.bind(g_driver.StreamSynchronize, "cuStreamSynchronize");
  g_driver.bind(g_driver.StreamIsCapturing, "cuStreamIsCapturing");
  g_driver.bind(g_driver.StreamBeginCapture, "cuStreamBeginCapture");
  g_driver.bind(g_driver.StreamEndCapture, "cuStreamEndCapture");
  // NB: plain "cuGraphInstantiate" is the legacy 5-argument entry point; the
  // 3-argument (CUDA >= 11.4) one is exported as cuGraphInstantiateWithFlags.
  g_driver.bind(g_driver.GraphInstantiate, "cuGraphInstantiateWithFlags");
  g_driver.bind(g_driver.GraphLaunch, "cuGraphLaunch");
  g_driver.bind(g_driver.GraphDestroy, "cuGraphDestroy");
  g_driver.bind(g_driver.GraphExecDestroy, "cuGraphExecDestroy");
  g_driver.bind(g_driver.FuncSetAttribute, "cuFuncSetAttribute");
  g_driver.bind(g_driver.DeviceGetAttribute, "cuDeviceGetAttribute");
  auto require = [](bool ok, const char* name) {
    if (!ok) throw std::runtime_error(std::string("nvcuda: missing versioned symbol ") + name);
  };
  require(g_driver.DeviceTotalMem != nullptr, "cuDeviceTotalMem_v2");
  require(g_driver.CtxCreate != nullptr, "cuCtxCreate_v2");
  require(g_driver.CtxDestroy != nullptr, "cuCtxDestroy_v2");
  require(g_driver.MemAlloc != nullptr, "cuMemAlloc_v2");
  require(g_driver.MemFree != nullptr, "cuMemFree_v2");
  require(g_driver.MemcpyHtoD != nullptr, "cuMemcpyHtoD_v2");
  require(g_driver.MemcpyDtoH != nullptr, "cuMemcpyDtoH_v2");
}

void load_nvrtc() {
  std::string path = default_nvrtc_path();
  // nvrtc loads nvrtc-builtins64_*.dll at compile time via its own loader, which
  // uses the process search path — so register its directory before loading.
  const size_t slash = path.find_last_of("\\/");
  std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);
  SetDllDirectoryA(dir.c_str());
  {
    std::wstring wdir(dir.begin(), dir.end());
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    AddDllDirectory(wdir.c_str());
  }
  g_nvrtc.lib = LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!g_nvrtc.lib) {
    g_nvrtc.lib = LoadLibraryA("nvrtc64_120_0.dll");
    if (!g_nvrtc.lib)
      throw std::runtime_error("nvrtc64_120_0.dll not found (set DSH_NVRTC_PATH). tried: " +
                               path);
  }
  auto bind = [&](auto& fn, const char* name) {
    fn = reinterpret_cast<std::decay_t<decltype(fn)>>(GetProcAddress(g_nvrtc.lib, name));
    if (!fn) throw std::runtime_error(std::string("nvrtc: missing symbol ") + name);
  };
  bind(g_nvrtc.CreateProgram, "nvrtcCreateProgram");
  bind(g_nvrtc.CompileProgram, "nvrtcCompileProgram");
  bind(g_nvrtc.GetPTXSize, "nvrtcGetPTXSize");
  bind(g_nvrtc.GetPTX, "nvrtcGetPTX");
  bind(g_nvrtc.GetProgramLogSize, "nvrtcGetProgramLogSize");
  bind(g_nvrtc.GetProgramLog, "nvrtcGetProgramLog");
  bind(g_nvrtc.DestroyProgram, "nvrtcDestroyProgram");
}

std::string nvrtc_log(nvrtcProgram prog) {
  size_t n = 0;
  if (g_nvrtc.GetProgramLogSize(prog, &n) != 0 || n == 0) return {};
  std::string log(n, '\0');
  if (g_nvrtc.GetProgramLog(prog, &log[0]) != 0) return {};
  if (!log.empty() && log.back() == '\0') log.pop_back();
  return log;
}

}  // namespace

void init() {
  if (g_ready) return;
  load_driver();
  check(g_driver.Init(0), "cuInit");
  const int64_t dev_index = engine::config::get_int("device");
  CUdevice dev = 0;
  if (dev_index != 0) {
    if (g_driver.DeviceGet(&dev, static_cast<int>(dev_index)) != 0)
      throw std::runtime_error("CUDA device " + std::to_string(dev_index) +
                               " not found (config key 'device', range 0..)" );
  } else {
    check(g_driver.DeviceGet(&dev, 0), "cuDeviceGet");
  }
  g_device = dev;
  check(g_driver.CtxCreate(&g_ctx, 0, dev), "cuCtxCreate");

  char name[256] = {0};
  check(g_driver.DeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
  g_info.name = name;
  check(g_driver.DeviceGetAttribute(&g_info.cc_major,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev),
        "cc_major");
  check(g_driver.DeviceGetAttribute(&g_info.cc_minor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev),
        "cc_minor");
  check(g_driver.DeviceGetAttribute(&g_info.sm_count,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev),
        "sm_count");
  size_t total = 0;
  check(g_driver.DeviceTotalMem(&total, dev), "cuDeviceTotalMem");
  g_info.mem_total = total;
  check(g_driver.DriverGetVersion(&g_info.driver_version), "cuDriverGetVersion");
  g_info.runtime_version = 12090;
  load_nvrtc();
  g_ready = true;
}

void shutdown() {
  if (g_ctx) {
    g_driver.CtxDestroy(g_ctx);
    g_ctx = nullptr;
  }
  g_ready = false;
}

const DeviceInfo& device_info() {
  if (!g_ready) throw std::runtime_error("cuda::init() not called");
  return g_info;
}

std::string compile_ptx(const std::string& src, const std::string& program_name,
                        const std::vector<std::string>& extra_opts) {
  if (!g_ready) throw std::runtime_error("cuda::init() not called");
  std::string arch = "--gpu-architecture=compute_" + std::to_string(g_info.cc_major) +
                     std::to_string(g_info.cc_minor);
  std::vector<std::string> opts = {arch, "--std=c++17"};
  for (const auto& o : extra_opts) opts.push_back(o);

  std::vector<const char*> copts;
  for (const auto& o : opts) copts.push_back(o.c_str());

  nvrtcProgram prog = nullptr;
  nvrtcResult rc = g_nvrtc.CreateProgram(&prog, src.c_str(), program_name.c_str(), 0,
                                         nullptr, nullptr);
  if (rc != 0) throw std::runtime_error("nvrtcCreateProgram failed: " + std::to_string(rc));

  rc = g_nvrtc.CompileProgram(prog, static_cast<int>(copts.size()), copts.data());
  if (rc != 0) {
    std::string log = nvrtc_log(prog);
    g_nvrtc.DestroyProgram(&prog);
    throw std::runtime_error("NVRTC compile failed (" + std::to_string(rc) + "):\n" + log);
  }

  size_t ptx_size = 0;
  if (g_nvrtc.GetPTXSize(prog, &ptx_size) != 0) {
    g_nvrtc.DestroyProgram(&prog);
    throw std::runtime_error("nvrtcGetPTXSize failed");
  }
  std::string ptx(ptx_size, '\0');
  if (g_nvrtc.GetPTX(prog, &ptx[0]) != 0) {
    g_nvrtc.DestroyProgram(&prog);
    throw std::runtime_error("nvrtcGetPTX failed");
  }
  g_nvrtc.DestroyProgram(&prog);
  if (!ptx.empty() && ptx.back() == '\0') ptx.pop_back();
  return ptx;
}

Module::Module(const std::string& ptx) {
  if (!g_ready) throw std::runtime_error("cuda::init() not called");
  CUmodule mod = nullptr;
  check(g_driver.ModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr), "cuModuleLoadDataEx");
  mod_ = mod;
}

Module::~Module() {
  if (mod_) g_driver.ModuleUnload(static_cast<CUmodule>(mod_));
}

void* Module::kernel(const std::string& name) const {
  CUfunction fn = nullptr;
  check(g_driver.ModuleGetFunction(&fn, static_cast<CUmodule>(mod_), name.c_str()),
        ("cuModuleGetFunction(" + name + ")").c_str());
  kprof_register(fn, name.c_str());
  return fn;
}

PinnedBuffer::PinnedBuffer(size_t bytes) {
  if (!g_ready) throw std::runtime_error("cuda::init() not called");
  if (!bytes) return;
  check(g_driver.MemAllocHost(&ptr_, bytes), "cuMemAllocHost");
  bytes_ = bytes;
}
PinnedBuffer::~PinnedBuffer() {
  if (ptr_) g_driver.MemFreeHost(ptr_);
}
PinnedBuffer::PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
  o.ptr_ = nullptr;
  o.bytes_ = 0;
}
PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& o) noexcept {
  if (this != &o) {
    if (ptr_) g_driver.MemFreeHost(ptr_);
    ptr_ = o.ptr_;
    bytes_ = o.bytes_;
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }
  return *this;
}
void PinnedBuffer::resize(size_t bytes) {
  if (bytes <= bytes_) return;
  *this = PinnedBuffer(bytes);
}

Buffer::Buffer(size_t bytes) {
  if (!g_ready) throw std::runtime_error("cuda::init() not called");
  if (!bytes) return;
  CUdeviceptr p = 0;
  check(g_driver.MemAlloc(&p, bytes), "cuMemAlloc");
  ptr_ = reinterpret_cast<void*>(static_cast<uintptr_t>(p));
  bytes_ = bytes;
  g_buf_bytes += static_cast<long long>(bytes);
  ++g_buf_count;
  g_buf_log.emplace_back(bytes, 0);
}

Buffer::~Buffer() {
  if (ptr_) g_driver.MemFree(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr_)));
}

Buffer::Buffer(Buffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
  o.ptr_ = nullptr;
  o.bytes_ = 0;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
  if (this != &o) {
    if (ptr_) g_driver.MemFree(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr_)));
    ptr_ = o.ptr_;
    bytes_ = o.bytes_;
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }
  return *this;
}

void Buffer::upload_at(const void* host, size_t bytes, void* dev_dst) {
  check(g_driver.MemcpyHtoD(reinterpret_cast<CUdeviceptr>(dev_dst), const_cast<void*>(host),
                            bytes),
        "cuMemcpyHtoD(offset)");
}

void* g_stream = nullptr;   // current stream (nullptr = legacy default)
// Debug aid for CUDA-graph capture: -1 = record everything, >= 0 = stop recording after
// that many kernels (ENGINE_GRAPH_KERNEL_LIMIT).  Reset at capture_begin().
int g_capture_kernel_limit = -1;
int g_capture_kernel_count = 0;

static bool capturing_now() {
  if (!g_driver.StreamIsCapturing || !g_stream) return false;
  int mode = 0;
  if (g_driver.StreamIsCapturing(g_stream, &mode) != CUDA_SUCCESS) return false;
  return mode != 0;   // 0 == CU_STREAM_CAPTURE_STATUS_NONE
}



void Buffer::upload_async(const void* host, size_t bytes) const {
  if (capturing_now()) return;
  if (bytes > bytes_) throw std::runtime_error("Buffer::upload_async exceeds allocation");
  check(g_driver.MemcpyHtoDAsync(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr_)),
                                 host, bytes, g_stream), "cuMemcpyHtoDAsync");
}
void Buffer::download_async(void* host, size_t bytes) const {
  if (capturing_now()) return;
  if (bytes > bytes_) throw std::runtime_error("Buffer::download_async exceeds allocation");
  check(g_driver.MemcpyDtoHAsync(host, static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr_)),
                                 bytes, g_stream), "cuMemcpyDtoHAsync");
}

// Stream-ordered device-to-device copy (rollback snapshots, hidden-state saves).
void copy_d2d_async(void* dst, const void* src, size_t bytes) {
  if (capturing_now() || bytes == 0) return;
  check(g_driver.MemcpyDtoDAsync(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dst)),
                                 static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(src)),
                                 bytes, g_stream),
        "cuMemcpyDtoDAsync");
}

void Buffer::upload(const void* host, size_t bytes) {
  if (capturing_now()) {
    static int w = 0;
    if (w++ < 8) std::fprintf(stderr, "[capture] skipped Buffer::upload(%zu)\n", bytes);
    return;
  }
  if (bytes > bytes_) throw std::runtime_error("Buffer::upload exceeds allocation");
  check(g_driver.MemcpyHtoD(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr_)), host,
                            bytes),
        "cuMemcpyHtoD");
}

void Buffer::download(void* host, size_t bytes) const {
  if (capturing_now()) {
    static int w = 0;
    if (w++ < 8) std::fprintf(stderr, "[capture] skipped Buffer::download(%zu)\n", bytes);
    return;
  }
  if (bytes > bytes_) throw std::runtime_error("Buffer::download exceeds allocation");
  check(g_driver.MemcpyDtoH(host,
                            static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr_)), bytes),
        "cuMemcpyDtoH");
}

void* create_stream() {
  void* s = nullptr;
  check(g_driver.StreamCreate(&s, 0), "cuStreamCreate");
  return s;
}
void destroy_stream(void* s) {
  if (s) g_driver.StreamDestroy(s);
}
void set_stream(void* s) { g_stream = s; }
void* current_stream() { return g_stream; }
void stream_sync(void* s) {
  if (capturing_now()) {
    static int warned = 0;
    if (warned++ < 8) std::fprintf(stderr, "[capture] skipped cuda::stream_sync()\n");
    return;
  }
  void* use = s ? s : g_stream;
  if (use) {
    // Count the stream flush like cuda::sync() so the per-layer debugging output and
    // the pipe summary cover the WDDM round trips too.
    const auto t0 = std::chrono::steady_clock::now();
    check(g_driver.StreamSynchronize(use), "cuStreamSynchronize");
    g_sync_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++g_sync_calls;
  } else {
    sync();
  }
}

Graph::Graph() = default;
Graph::~Graph() {
  if (exec_) g_driver.GraphExecDestroy(exec_);
  if (graph_) g_driver.GraphDestroy(graph_);
}
void Graph::capture_begin() {
  {
    static const int lim = []() {
      const char* e = std::getenv("ENGINE_GRAPH_KERNEL_LIMIT");
      return e ? std::atoi(e) : -1;
    }();
    g_capture_kernel_limit = lim;
    g_capture_kernel_count = 0;
  }
  check(g_driver.StreamBeginCapture(g_stream, 0 /* global */), "cuStreamBeginCapture");
}
void Graph::capture_end() {
  check(g_driver.StreamEndCapture(g_stream, &graph_), "cuStreamEndCapture");
  check(g_driver.GraphInstantiate(&exec_, graph_, 0), "cuGraphInstantiate");
}
void Graph::launch() { check(g_driver.GraphLaunch(exec_, g_stream), "cuGraphLaunch"); }

long long g_launch_calls = 0;
double g_launch_ms = 0.0;
namespace {
struct KProfAgg { double ms = 0.0; long long calls = 0; };
std::map<std::string, KProfAgg> g_kprof_agg;
std::map<void*, std::string> g_kprof_names;
struct KProfPair { void* fn; void* e0; void* e1; };
std::vector<KProfPair> g_kprof_pairs;
std::vector<void*> g_kprof_events;
bool g_kprof_on = false;
bool g_kprof_init = false;
double g_kprof_wall_ms = 0.0;
}  // namespace

void kprof_enable() {
  if (g_kprof_init) return;
  g_kprof_init = true;
  g_kprof_on = std::getenv("ENGINE_KPROF") != nullptr;
  if (!g_kprof_on) return;
  g_kprof_events.resize(2 * 32768);
  for (auto& e : g_kprof_events)
    g_driver.EventCreate(&e, 0u);
}
void kprof_register(void* fn, const char* name) {
  if (fn) g_kprof_names[fn] = name;
}
void kprof_add_wall(double ms) { g_kprof_wall_ms += ms; }
void kprof_reset() {
  g_kprof_agg.clear();
  g_kprof_pairs.clear();
  g_kprof_wall_ms = 0.0;
}
void kprof_dump() {
  if (std::getenv("ENGINE_KPROF_DEBUG"))
    std::fprintf(stderr, "[kprof] dump on=%d pairs=%zu names=%zu\n", (int)g_kprof_on,
                 g_kprof_pairs.size(), g_kprof_names.size());
  if (!g_kprof_on || g_kprof_pairs.empty()) return;
  check(g_driver.CtxSynchronize(), "cuCtxSynchronize");
  for (auto& pr : g_kprof_pairs) {
    float ms = 0.f;
    if (g_driver.EventElapsedTime(&ms, pr.e0, pr.e1) != CUDA_SUCCESS) continue;
    const auto it = g_kprof_names.find(pr.fn);
    const std::string nm = it == g_kprof_names.end() ? "?" : it->second;
    g_kprof_agg[nm].ms += ms;
    g_kprof_agg[nm].calls += 1;
  }
  std::vector<std::pair<std::string, KProfAgg>> v(g_kprof_agg.begin(), g_kprof_agg.end());
  std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
  double total = 0.0;
  for (const auto& kv : v) total += kv.second.ms;
  std::fprintf(stderr, "\n[kprof] total GPU time %.2f ms over %zu kernel kinds\n", total, v.size());
  std::fprintf(stderr, "[kprof] wall (decoded tokens) %.2f ms -> GPU busy %.1f%%\n", g_kprof_wall_ms,
               g_kprof_wall_ms > 0 ? 100.0 * total / g_kprof_wall_ms : 0.0);
  const int top = static_cast<int>(v.size() < 28 ? v.size() : 28);
  for (int i = 0; i < top; ++i)
    std::fprintf(stderr, "[kprof] %6.2f ms %7lld calls  %-24s %.1f%%\n", v[i].second.ms,
                 v[i].second.calls, v[i].first.c_str(),
                 total > 0 ? 100.0 * v[i].second.ms / total : 0.0);
}

namespace {
// Diagnostic A/B aid: ENGINE_SKIP_KERNELS="name1,name2" drops every launch of the
// named kernels (wall-clock attribution only; numerics are junk).
bool skip_kernel(void* fn) {
  static const char* spec = std::getenv("ENGINE_SKIP_KERNELS");
  if (!spec || !*spec) return false;
  static std::map<void*, bool> memo;
  const auto it = memo.find(fn);
  if (it != memo.end()) return it->second;
  bool skip = false;
  const auto nm = g_kprof_names.find(fn);
  if (nm != g_kprof_names.end()) {
    const std::string& name = nm->second;
    const std::string list = spec;
    size_t pos = 0;
    while (pos <= list.size()) {
      const size_t comma = list.find(',', pos);
      const std::string item =
          list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!item.empty() && name == item) { skip = true; break; }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  memo[fn] = skip;
  return skip;
}
}  // namespace

// Largest dynamic shared-memory size a kernel may request on this device, and opt the
// kernel in to it (the default is 48 KiB; Ampere allows up to 99/100 KiB with
// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES set to the device limit).  Returns
// the usable maximum in bytes, or 0 when the attribute call fails (the caller should
// fall back to the 48 KiB default).
size_t max_dynamic_shared_mem(void* fn) {
  static size_t device_max = 0;  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
  static std::map<void*, size_t> per_fn;
  if (!g_ready || g_device < 0) return 0;
  const auto it = per_fn.find(fn);
  if (it != per_fn.end()) return it->second;
  if (device_max == 0) {
    int v = 0;
    if (g_driver.DeviceGetAttribute(&v, 97 /* MAX_SHARED_MEMORY_PER_BLOCK_OPTIN */,
                                    g_device) != 0 ||
        v <= 0) {
      per_fn[fn] = 0;
      return 0;
    }
    device_max = static_cast<size_t>(v);
  }
  const CUresult r = g_driver.FuncSetAttribute(static_cast<CUfunction>(fn), 8 /* MAX_DYNAMIC_SHARED_SIZE_BYTES */,
                                               static_cast<int>(device_max));
  const size_t usable = (r == 0) ? device_max : 0;
  per_fn[fn] = usable;
  return usable;
}

void launch(void* fn, int gx, int gy, int gz, int bx, int by, int bz, size_t shared,
            void** args, void* stream) {
  kprof_enable();
  if (skip_kernel(fn)) return;
  // Debug aid: while capturing, stop recording after N kernels so a failing replay can
  // be bisected down to the exact kernel (ENGINE_GRAPH_KERNEL_LIMIT=<n>).
  if (g_capture_kernel_limit >= 0 && capturing_now()) {
    if (g_capture_kernel_count >= g_capture_kernel_limit) {
      ++g_capture_kernel_count;
      return;
    }
    ++g_capture_kernel_count;
  }
  const bool prof = g_kprof_on && g_kprof_pairs.size() * 2 + 2 <= g_kprof_events.size();
  void* e0 = nullptr;
  void* e1 = nullptr;
  if (prof) {
    e0 = g_kprof_events[g_kprof_pairs.size() * 2];
    e1 = g_kprof_events[g_kprof_pairs.size() * 2 + 1];
    g_driver.EventRecord(e0, stream ? stream : g_stream);
  }
  const auto lt0 = std::chrono::steady_clock::now();
  {
    const CUresult rc =
        g_driver.LaunchKernel(static_cast<CUfunction>(fn), static_cast<unsigned>(gx),
                              static_cast<unsigned>(gy), static_cast<unsigned>(gz),
                              static_cast<unsigned>(bx), static_cast<unsigned>(by),
                              static_cast<unsigned>(bz), static_cast<unsigned>(shared),
                              stream ? stream : g_stream, args, nullptr);
    if (rc != CUDA_SUCCESS) {
      // Name the kernel and the exact launch tuple: without them a configuration error
      // is almost impossible to attribute (this is how the >384-token shared-memory bug
      // was found).  The lookup only happens on the failure path.
      const auto nm = g_kprof_names.find(fn);
      const std::string what =
          nm == g_kprof_names.end()
              ? std::string("cuLaunchKernel")
              : ("cuLaunchKernel(" + nm->second + " grid " + std::to_string(gx) + "x" +
                 std::to_string(gy) + "x" + std::to_string(gz) + " block " + std::to_string(bx) +
                 " shared " + std::to_string(shared) + ")");
      check(rc, what.c_str());
    }
  }
  g_launch_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lt0).count();
  ++g_launch_calls;
  if (prof) {
    g_driver.EventRecord(e1, stream ? stream : g_stream);
    g_kprof_pairs.push_back({fn, e0, e1});
  }
}

std::vector<std::pair<size_t, int>> g_buf_log;
long long g_buf_bytes = 0;
long long g_buf_count = 0;
long long g_sync_calls = 0;
double g_sync_ms = 0.0;
void sync() {
  if (capturing_now()) {
    static int warned = 0;
    if (warned++ < 8) std::fprintf(stderr, "[capture] skipped cuda::sync()\n");
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  check(g_driver.CtxSynchronize(), "cuCtxSynchronize");
  g_sync_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                   .count();
  ++g_sync_calls;
}

MemInfo mem_info() {
  MemInfo m;
  check(g_driver.MemGetInfo(&m.free_bytes, &m.total_bytes), "cuMemGetInfo");
  return m;
}

Event::Event() { check(g_driver.EventCreate(&ev_, 0), "cuEventCreate"); }
Event::~Event() {
  if (ev_) g_driver.EventDestroy(ev_);
}
void Event::record() { check(g_driver.EventRecord(ev_, nullptr), "cuEventRecord"); }
void Event::synchronize() { check(g_driver.EventSynchronize(ev_), "cuEventSynchronize"); }
double Event::elapsed_ms(const Event& end) const {
  float ms = 0;
  check(g_driver.EventElapsedTime(&ms, ev_, end.ev_), "cuEventElapsedTime");
  return static_cast<double>(ms);
}

}  // namespace cuda
