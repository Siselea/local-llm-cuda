// Minimal HTTP/1.1 server on WinSock2 -- no third-party dependencies.
//
// Single-threaded serial accept loop: one request at a time (this is a
// single-user local engine, and it also guarantees only one generation runs at
// once on the GPU).  Supports keep-alive, Content-Length bodies, chunked
// streaming responses (for SSE), CORS preflight and graceful shutdown.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace http {

struct Request {
  std::string method;
  std::string path;     // decoded path, no query string
  std::string query;    // raw query string (?a=b&c=d), without '?'
  std::string version;
  std::map<std::string, std::string> headers;  // keys lower-cased
  std::string body;
  std::string client;   // "ip:port"

  std::string header(const std::string& name) const {
    const auto it = headers.find(name);
    return it == headers.end() ? std::string() : it->second;
  }
  std::string param(const std::string& name) const;  // query parameter
};

// One client connection.  A handler either sends a complete response with
// send()/send_json(), or streams chunks with begin_stream()/write_chunk()/
// end_stream().  All writes return false once the client is gone; the caller
// should then abort whatever it was doing (generation stops, no GPU waste).
class Conn {
 public:
  bool send(int status, const std::string& content_type, const std::string& body,
            const std::vector<std::pair<std::string, std::string>>& extra_headers = {});
  bool send_json(int status, const std::string& body);
  bool begin_stream(const std::string& content_type);
  bool write_chunk(const std::string& data);  // raw bytes, chunk framing added
  bool end_stream();

  // JSON error response: {"error":{"message":...,"type":"invalid_request_error"}}
  bool send_error(int status, const std::string& msg);

  bool failed() const { return failed_; }
  bool stream_open() const { return streaming_; }
  void set_close_after(bool v) { close_after_ = v; }
  bool close_after() const { return close_after_; }
  const Request& request() const { return req_; }

 private:
  friend class Server;
  void reset(intptr_t fd, const std::string& client);
  bool read_request(bool* keep_alive);
  bool write_raw(const char* data, size_t n);

  intptr_t fd_ = -1;  // SOCKET; intptr_t keeps winsock2.h out of this header
  std::string client_;
  bool failed_ = false;
  bool streaming_ = false;
  bool close_after_ = false;
  std::string pending_;  // bytes read past the current request (keep-alive/pipelining)
  Request req_;
};

class Server {
 public:
  using Handler = std::function<void(const Request&, Conn&)>;

  Server(std::string host, int port);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  void set_handler(Handler h) { handler_ = std::move(h); }

  // Bind + listen.  run() calls this lazily, so callers can report the actual
  // bound address before entering the accept loop.
  bool start(std::string* err);

  // Serves until stop().  Returns false and fills *err on socket errors.
  bool run(std::string* err);

  // Async-safe: may be called from a signal handler / control handler.
  void stop() { stopping_.store(true, std::memory_order_relaxed); }
  bool stopping() const { return stopping_.load(std::memory_order_relaxed); }

  int port() const { return port_; }
  const std::string& host() const { return host_; }

 private:
  bool listen_socket(std::string* err);
  void serve_connection(intptr_t fd, const std::string& client);

  std::string host_;
  int port_ = 0;
  intptr_t listen_fd_ = -1;
  Handler handler_;
  std::atomic<bool> stopping_{false};
};

const char* status_text(int status);

}  // namespace http
}  // namespace engine
