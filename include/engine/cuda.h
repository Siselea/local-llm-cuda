// Minimal CUDA runtime for the engine: driver API + NVRTC (no CUDA Toolkit needed).
//
// Both cudart/cublas and nvrtc come from pip wheels:
//   nvidia/cuda_runtime/bin/cudart64_12.dll   (we link cuda.lib -> nvcuda.dll)
//   nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll   (loaded dynamically)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cuda {

struct DeviceInfo {
  std::string name;
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  uint64_t mem_total = 0;
  int driver_version = 0;
  int runtime_version = 0;
};

void init();
void shutdown();
const DeviceInfo& device_info();

// Compile CUDA C source to PTX for this GPU via NVRTC. Throws on error with the
// compiler log attached.
std::string compile_ptx(const std::string& src, const std::string& program_name,
                        const std::vector<std::string>& extra_opts = {});

// Module (loaded PTX) + kernel lookup.
class Module {
 public:
  explicit Module(const std::string& ptx);
  ~Module();
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  void* kernel(const std::string& name) const;  // CUfunction

 private:
  void* mod_ = nullptr;
};

// Device buffer with RAII.
// Page-locked host memory. WDDM charges a large fixed cost for pageable copies;
// staging the per-layer MoE activations through pinned memory avoids it.
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t bytes);
  ~PinnedBuffer();
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&&) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&&) noexcept;
  void resize(size_t bytes);
  void* data() { return ptr_; }
  const void* data() const { return ptr_; }
  size_t size() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(size_t bytes);
  ~Buffer();
  Buffer(Buffer&& o) noexcept;
  Buffer& operator=(Buffer&& o) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void* data() const { return ptr_; }
  size_t size() const { return bytes_; }
  void upload(const void* host, size_t bytes);
  void upload_async(const void* host, size_t bytes) const;
  void download_async(void* host, size_t bytes) const;
  // Copy into an arbitrary device offset (expert streaming arena).
  void upload_at(const void* host, size_t bytes, void* dev_dst);
  void download(void* host, size_t bytes) const;

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

void kprof_enable();
void kprof_register(void* fn, const char* name);
void kprof_add_wall(double ms);
void kprof_reset();
void kprof_dump();

void launch(void* fn, int gx, int gy, int gz, int bx, int by, int bz, size_t shared,
            void** args, void* stream = nullptr);
// Stream-ordered device-to-device copy (rollback snapshots, hidden-state saves).
void copy_d2d_async(void* dst, const void* src, size_t bytes);
// Opt `fn` in to the largest dynamic shared-memory size the device supports and return
// that size in bytes (0 = unsupported; the 48 KiB default still applies then).
size_t max_dynamic_shared_mem(void* fn);
void sync();
extern long long g_sync_calls;
extern double g_sync_ms;
extern long long g_launch_calls;
extern double g_launch_ms;
// Monotonic count of Graph::launch() calls.  A graph replay executes its recorded
// kernels without going through cuda::launch(), so g_launch_calls does not advance for
// it; host caches that use "nothing was launched since" as a validity proof must also
// compare this generation (see docs/GRAPH-INVARIANTS.md).
extern long long g_graph_launches;
extern long long g_buf_bytes;
extern long long g_buf_count;
extern std::vector<std::pair<size_t, int>> g_buf_log;
// graph-capture bisection aid (ENGINE_GRAPH_KERNEL_LIMIT)
extern int g_capture_kernel_limit;
extern int g_capture_kernel_count;

struct MemInfo {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
};
MemInfo mem_info();

// Streams: when a launch passes stream == nullptr the *current* stream is used.
void* create_stream();
void destroy_stream(void* s);
void set_stream(void* s);
void* current_stream();
void stream_sync(void* s = nullptr);

// CUDA graph capture (records the launches made on the current stream).
class Graph {
 public:
  Graph();
  ~Graph();
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  void capture_begin();
  void capture_end();
  // Abandon an in-progress recording so the stream is usable again after a failed
  // capture (an exception thrown inside the recorded sequence would otherwise leave
  // the stream in capture mode).  A no-op when the stream is not capturing.
  void capture_abort();
  void launch();
  bool ready() const { return exec_ != nullptr; }

 private:
  void* graph_ = nullptr;
  void* exec_ = nullptr;
};

// Timing events (GPU-side elapsed time).
class Event {
 public:
  Event();
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  void record();
  // Cross-stream ordering without a host synchronisation: record on one stream, then
  // make another stream wait for it (cuStreamWaitEvent).
  void record_on(void* stream);
  void stream_wait(void* stream);
  double elapsed_ms(const Event& end) const;   // valid after end.sync()
  void synchronize();

 private:
  void* ev_ = nullptr;
};

}  // namespace cuda
