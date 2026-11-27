// Minimal HTTP/1.1 server on WinSock2.
//
// The MSVC-style headers declare the winsock entry points as __declspec
// (dllimport), which would require adding -lws2_32 to build.sh.  To keep the
// build command line unchanged (see the task constraints) ws2_32.dll is loaded
// with LoadLibrary/GetProcAddress and every call goes through a function
// pointer.  The same trick keeps the server testable on other targets.
#include "engine/http_server.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#else
#error "engine/http_server.cpp currently targets WinSock2 (Windows)"
#endif

namespace engine {
namespace http {

namespace {

// ---------------------------------------------------------------------------
// ws2_32.dll loader
// ---------------------------------------------------------------------------
struct Winsock {
  using WSAStartupFn = int(WINAPI*)(WORD, LPWSADATA);
  using WSACleanupFn = int(WINAPI*)();
  using socketFn = SOCKET(WINAPI*)(int, int, int);
  using bindFn = int(WINAPI*)(SOCKET, const struct sockaddr*, int);
  using listenFn = int(WINAPI*)(SOCKET, int);
  using acceptFn = SOCKET(WINAPI*)(SOCKET, struct sockaddr*, int*);
  using selectFn = int(WINAPI*)(int, fd_set*, fd_set*, fd_set*, const struct timeval*);
  using setsockoptFn = int(WINAPI*)(SOCKET, int, int, const char*, int);
  using recvFn = int(WINAPI*)(SOCKET, char*, int, int);
  using sendFn = int(WINAPI*)(SOCKET, const char*, int, int);
  using closesocketFn = int(WINAPI*)(SOCKET);
  using shutdownFn = int(WINAPI*)(SOCKET, int);

  HMODULE dll = nullptr;
  WSAStartupFn WSAStartup = nullptr;
  WSACleanupFn WSACleanup = nullptr;
  socketFn socket = nullptr;
  bindFn bind = nullptr;
  listenFn listen = nullptr;
  acceptFn accept = nullptr;
  selectFn select = nullptr;
  setsockoptFn setsockopt = nullptr;
  recvFn recv = nullptr;
  sendFn send = nullptr;
  closesocketFn closesocket = nullptr;
  shutdownFn shutdown = nullptr;

  bool load(std::string* err) {
    dll = ::LoadLibraryA("ws2_32.dll");
    if (!dll) {
      *err = "cannot load ws2_32.dll";
      return false;
    }
    auto get = [&](const char* name) -> FARPROC { return ::GetProcAddress(dll, name); };
    WSAStartup = reinterpret_cast<WSAStartupFn>(get("WSAStartup"));
    WSACleanup = reinterpret_cast<WSACleanupFn>(get("WSACleanup"));
    socket = reinterpret_cast<socketFn>(get("socket"));
    bind = reinterpret_cast<bindFn>(get("bind"));
    listen = reinterpret_cast<listenFn>(get("listen"));
    accept = reinterpret_cast<acceptFn>(get("accept"));
    select = reinterpret_cast<selectFn>(get("select"));
    setsockopt = reinterpret_cast<setsockoptFn>(get("setsockopt"));
    recv = reinterpret_cast<recvFn>(get("recv"));
    send = reinterpret_cast<sendFn>(get("send"));
    closesocket = reinterpret_cast<closesocketFn>(get("closesocket"));
    shutdown = reinterpret_cast<shutdownFn>(get("shutdown"));
    if (!WSAStartup || !WSACleanup || !socket || !bind || !listen || !accept || !select ||
        !setsockopt || !recv || !send || !closesocket || !shutdown) {
      *err = "ws2_32.dll is missing expected entry points";
      return false;
    }
    return true;
  }
};

Winsock& winsock() {
  static Winsock ws;
  static bool loaded = false;
  static std::string load_err;
  if (!loaded) {
    loaded = true;
    if (!ws.load(&load_err)) throw std::runtime_error(load_err);
    WSADATA data{};
    if (ws.WSAStartup(MAKEWORD(2, 2), &data) != 0)
      throw std::runtime_error("WSAStartup failed");
  }
  return ws;
}

// Little-endian host byte order (Windows on x86/x64/arm64).
uint16_t htons_u(uint16_t v) { return static_cast<uint16_t>((v >> 8) | (v << 8)); }
uint16_t ntohs_u(uint16_t v) { return htons_u(v); }
uint32_t htonl_u(uint32_t v) {
  return ((v & 0xFFu) << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | ((v >> 24) & 0xFFu);
}

bool parse_ipv4(const std::string& host, uint32_t* out, std::string* err) {
  if (host.empty() || host == "0.0.0.0" || host == "*") {
    *out = 0;  // INADDR_ANY
    return true;
  }
  unsigned a = 0, b = 0, c = 0, d = 0;
  char tail = 0;
  if (std::sscanf(host.c_str(), "%u.%u.%u.%u%c", &a, &b, &c, &d, &tail) != 4 || a > 255 ||
      b > 255 || c > 255 || d > 255) {
    *err = "invalid IPv4 bind address '" + host + "'";
    return false;
  }
  *out = htonl_u((a << 24) | (b << 16) | (c << 8) | d);
  return true;
}

constexpr size_t kMaxHeaderBytes = 64 * 1024;
constexpr size_t kMaxBodyBytes = 32 * 1024 * 1024;
constexpr int kRecvTimeoutMs = 30000;

// Strict Content-Length: ASCII digits only, no sign, no empty value, bounded so the
// conversion cannot overflow.  `TooLarge` means syntactically valid but above the
// body cap (-> 413); `Invalid` means malformed (-> 400).
enum class ContentLengthParse { Ok, Invalid, TooLarge };
ContentLengthParse parse_content_length(const std::string& v, size_t* out) {
  if (v.empty()) return ContentLengthParse::Invalid;
  for (char c : v)
    if (c < '0' || c > '9') return ContentLengthParse::Invalid;
  size_t n = 0;
  for (char c : v) {
    const size_t digit = static_cast<size_t>(c - '0');
    if (n > (kMaxBodyBytes - digit) / 10) return ContentLengthParse::TooLarge;
    n = n * 10 + digit;
  }
  if (n > kMaxBodyBytes) return ContentLengthParse::TooLarge;
  *out = n;
  return ContentLengthParse::Ok;
}

std::string lower_ascii(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string trim_ws(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
  return s.substr(b, e - b);
}

std::string percent_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      const int hi = hex(s[i + 1]), lo = hex(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>(hi * 16 + lo));
        i += 2;
        continue;
      }
    }
    out.push_back(s[i]);
  }
  return out;
}

}  // namespace

const char* status_text(int status) {
  switch (status) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Unknown";
  }
}

// ---------------------------------------------------------------------------
// Request helpers
// ---------------------------------------------------------------------------
std::string Request::param(const std::string& name) const {
  if (query.empty()) return "";
  size_t pos = 0;
  while (pos <= query.size()) {
    size_t amp = query.find('&', pos);
    if (amp == std::string::npos) amp = query.size();
    const std::string kv = query.substr(pos, amp - pos);
    const size_t eq = kv.find('=');
    const std::string key = percent_decode(eq == std::string::npos ? kv : kv.substr(0, eq));
    if (key == name)
      return percent_decode(eq == std::string::npos ? std::string() : kv.substr(eq + 1));
    if (amp == query.size()) break;
    pos = amp + 1;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Conn
// ---------------------------------------------------------------------------
void Conn::reset(intptr_t fd, const std::string& client) {
  fd_ = fd;
  client_ = client;
  failed_ = false;
  streaming_ = false;
  close_after_ = false;
  pending_.clear();
  req_ = Request();
  req_.client = client;
}

bool Conn::write_raw(const char* data, size_t n) {
  if (failed_) return false;
  Winsock& ws = winsock();
  size_t off = 0;
  while (off < n) {
    const int chunk = static_cast<int>(std::min<size_t>(n - off, 1 << 20));
    const int r = ws.send(static_cast<SOCKET>(fd_), data + off, chunk, 0);
    if (r <= 0) {
      failed_ = true;
      return false;
    }
    off += static_cast<size_t>(r);
  }
  return true;
}

bool Conn::send(int status, const std::string& content_type, const std::string& body,
                const std::vector<std::pair<std::string, std::string>>& extra_headers) {
  std::string head = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
  head += "Server: engine/1.0\r\n";
  head += "Content-Type: " + content_type + "\r\n";
  head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  head += "Access-Control-Allow-Origin: *\r\n";
  head += "Connection: " +
          std::string(close_after_ ? "close" : "keep-alive") + "\r\n";
  for (const auto& h : extra_headers) head += h.first + ": " + h.second + "\r\n";
  head += "\r\n";
  return write_raw(head.data(), head.size()) && write_raw(body.data(), body.size());
}

bool Conn::send_json(int status, const std::string& body) {
  return send(status, "application/json; charset=utf-8", body);
}

bool Conn::begin_stream(const std::string& content_type) {
  std::string head = "HTTP/1.1 200 OK\r\n";
  head += "Server: engine/1.0\r\n";
  head += "Content-Type: " + content_type + "\r\n";
  head += "Cache-Control: no-cache\r\n";
  head += "Access-Control-Allow-Origin: *\r\n";
  head += "X-Accel-Buffering: no\r\n";
  head += "Connection: keep-alive\r\n";
  head += "Transfer-Encoding: chunked\r\n";
  head += "\r\n";
  if (!write_raw(head.data(), head.size())) return false;
  streaming_ = true;
  return true;
}

bool Conn::write_chunk(const std::string& data) {
  if (failed_) return false;
  if (data.empty()) return true;
  char sizebuf[24];
  std::snprintf(sizebuf, sizeof(sizebuf), "%zx\r\n", data.size());
  if (!write_raw(sizebuf, std::strlen(sizebuf))) return false;
  if (!write_raw(data.data(), data.size())) return false;
  return write_raw("\r\n", 2);
}

bool Conn::end_stream() {
  if (!streaming_) return true;
  streaming_ = false;
  return write_raw("0\r\n\r\n", 5);
}

bool Conn::send_error(int status, const std::string& msg) {
  std::string esc;
  for (unsigned char c : msg) {
    switch (c) {
      case '"': esc += "\\\""; break;
      case '\\': esc += "\\\\"; break;
      case '\n': esc += "\\n"; break;
      case '\r': esc += "\\r"; break;
      case '\t': esc += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", c);
          esc += b;
        } else {
          esc.push_back(static_cast<char>(c));
        }
    }
  }
  const std::string body = "{\"error\":{\"message\":\"" + esc + "\",\"type\":\"invalid_request_error\",\"code\":" +
                           std::to_string(status) + "}}";
  return send_json(status, body);
}

bool Conn::read_request(bool* keep_alive) {
  Winsock& ws = winsock();
  *keep_alive = false;
  streaming_ = false;
  close_after_ = false;
  req_ = Request();
  req_.client = client_;

  // read until the end of the headers
  size_t header_end = pending_.find("\r\n\r\n");
  char tmp[8192];
  while (header_end == std::string::npos) {
    if (pending_.size() > kMaxHeaderBytes) {
      send_error(431, "request headers too large");
      failed_ = true;
      return false;
    }
    const int n = ws.recv(static_cast<SOCKET>(fd_), tmp, sizeof(tmp), 0);
    if (n <= 0) {
      failed_ = true;
      return false;
    }
    pending_.append(tmp, static_cast<size_t>(n));
    header_end = pending_.find("\r\n\r\n");
  }

  // request line + headers
  const std::string head = pending_.substr(0, header_end);
  size_t line_start = 0;
  bool first = true;
  size_t content_length = 0;
  int content_length_count = 0;
  bool bad_content_length = false;
  bool has_transfer_encoding = false;
  std::string conn_header;
  bool expect_continue = false;
  while (line_start < head.size()) {
    size_t line_end = head.find("\r\n", line_start);
    if (line_end == std::string::npos) line_end = head.size();
    const std::string line = head.substr(line_start, line_end - line_start);
    line_start = line_end + 2;
    if (first) {
      first = false;
      size_t s1 = line.find(' ');
      size_t s2 = s1 == std::string::npos ? std::string::npos : line.find(' ', s1 + 1);
      if (s1 == std::string::npos || s2 == std::string::npos) {
        send_error(400, "malformed request line");
        failed_ = true;
        return false;
      }
      req_.method = line.substr(0, s1);
      std::string target = line.substr(s1 + 1, s2 - s1 - 1);
      req_.version = trim_ws(line.substr(s2 + 1));
      const size_t q = target.find('?');
      if (q == std::string::npos) {
        req_.path = percent_decode(target);
      } else {
        req_.path = percent_decode(target.substr(0, q));
        req_.query = target.substr(q + 1);
      }
      continue;
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = lower_ascii(trim_ws(line.substr(0, colon)));
    const std::string value = trim_ws(line.substr(colon + 1));
    req_.headers[key] = value;
    if (key == "content-length") {
      ++content_length_count;
      size_t parsed = 0;
      const ContentLengthParse r = parse_content_length(value, &parsed);
      if (content_length_count > 1) {
        // Request-smuggling defence: never accept two framing headers.
        send_error(400, "duplicate Content-Length header");
        failed_ = true;
        return false;
      }
      if (r == ContentLengthParse::Invalid) {
        bad_content_length = true;
      } else if (r == ContentLengthParse::TooLarge) {
        send_error(413, "request body too large");
        failed_ = true;
        return false;
      } else {
        content_length = parsed;
      }
    } else if (key == "connection") {
      conn_header = lower_ascii(value);
    } else if (key == "transfer-encoding") {
      // Chunked request bodies are not implemented; treating them as a normal body
      // would desynchronise the stream, so reject explicitly (contract: send
      // Content-Length).
      has_transfer_encoding = true;
    } else if (key == "expect" && lower_ascii(value).find("100-continue") != std::string::npos) {
      expect_continue = true;
    }
  }
  if (bad_content_length) {
    send_error(400, "invalid Content-Length header (ASCII digits only)");
    failed_ = true;
    return false;
  }
  if (has_transfer_encoding) {
    send_error(501, "request Transfer-Encoding is not supported; send a Content-Length body");
    failed_ = true;
    return false;
  }
  // Re-check the header size after the CRLFCRLF terminator was found: one recv() can
  // overshoot the limit while the read loop only looks for the terminator.
  if (header_end > kMaxHeaderBytes) {
    send_error(431, "request headers too large");
    failed_ = true;
    return false;
  }

  if (expect_continue && content_length > 0) {
    if (!write_raw("HTTP/1.1 100 Continue\r\n\r\n", 25)) return false;
  }

  const size_t body_start = header_end + 4;
  while (pending_.size() < body_start + content_length) {
    const int n = ws.recv(static_cast<SOCKET>(fd_), tmp, sizeof(tmp), 0);
    if (n <= 0) {
      failed_ = true;
      return false;
    }
    pending_.append(tmp, static_cast<size_t>(n));
  }
  req_.body = pending_.substr(body_start, content_length);
  pending_.erase(0, body_start + content_length);

  const bool http11 = req_.version == "HTTP/1.1";
  if (conn_header == "close") *keep_alive = false;
  else if (conn_header == "keep-alive") *keep_alive = true;
  else *keep_alive = http11;
  close_after_ = !*keep_alive;
  return true;
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------
Server::Server(std::string host, int port) : host_(std::move(host)), port_(port) {}

Server::~Server() {
  if (listen_fd_ >= 0) {
    try {
      winsock().closesocket(static_cast<SOCKET>(listen_fd_));
    } catch (...) {
    }
    listen_fd_ = -1;
  }
}

bool Server::listen_socket(std::string* err) {
  Winsock& ws = winsock();
  uint32_t addr = 0;
  if (!parse_ipv4(host_, &addr, err)) return false;

  const SOCKET s = ws.socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET) {
    *err = "socket() failed";
    return false;
  }
  int one = 1;
  ws.setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
  int timeout = kRecvTimeoutMs;
  ws.setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
  ws.setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));

  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port = htons_u(static_cast<uint16_t>(port_));
  sa.sin_addr.s_addr = addr;
  if (ws.bind(s, reinterpret_cast<const sockaddr*>(&sa), sizeof(sa)) != 0) {
    *err = "bind(" + host_ + ":" + std::to_string(port_) + ") failed (port in use?)";
    ws.closesocket(s);
    return false;
  }
  if (ws.listen(s, SOMAXCONN) != 0) {
    *err = "listen() failed";
    ws.closesocket(s);
    return false;
  }
  listen_fd_ = static_cast<intptr_t>(s);
  return true;
}

void Server::serve_connection(intptr_t fd, const std::string& client) {
  Conn conn;
  conn.reset(fd, client);
  while (!stopping()) {
    bool keep_alive = false;
    if (!conn.read_request(&keep_alive)) {
      // read_request wrote a 4xx when the request was malformed; a plain
      // disconnect just ends the loop.
      break;
    }
    if (!handler_) {
      conn.send_error(404, "no handler");
    } else {
      try {
        handler_(conn.request(), conn);
        if (conn.stream_open()) conn.end_stream();
      } catch (const std::exception& e) {
        if (conn.stream_open()) {
          conn.end_stream();
        } else {
          conn.send_error(500, std::string("handler error: ") + e.what());
        }
      }
    }
    if (!keep_alive || conn.failed() || conn.close_after() || stopping()) break;
  }
  try {
    winsock().closesocket(static_cast<SOCKET>(fd));
  } catch (...) {
  }
}

bool Server::start(std::string* err) {
  if (listen_fd_ >= 0) return true;
  return listen_socket(err);
}

namespace {
// 503 body answered to connections that were already queued when /shutdown stopped
// the accept loop (best effort; a connection that arrives after the listen socket is
// closed still gets a normal TCP reset).
void refuse_one(SOCKET c) {
  static const char kBody[] =
      "{\"error\":{\"message\":\"shutting down\",\"type\":"
      "\"server_shutting_down\",\"code\":503}}";
  std::string resp = "HTTP/1.1 503 Service Unavailable\r\n";
  resp += "Server: engine/1.0\r\n";
  resp += "Content-Type: application/json\r\n";
  resp += "Connection: close\r\n";
  resp += "Content-Length: " + std::to_string(sizeof(kBody) - 1) + "\r\n\r\n";
  resp += kBody;
  Winsock& ws = winsock();
  ws.send(c, resp.data(), static_cast<int>(resp.size()), 0);
}
}  // namespace

void Server::refuse_pending_connections() {
  if (listen_fd_ < 0) return;
  Winsock& ws = winsock();
  for (int served = 0; served < 16; ++served) {
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(static_cast<SOCKET>(listen_fd_), &readfds);
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 20 * 1000;  // 20 ms per poll: shutdown latency stays tiny
    if (ws.select(listen_fd_ + 1, &readfds, nullptr, nullptr, &tv) <= 0) break;
    sockaddr_in peer{};
    int peer_len = sizeof(peer);
    const SOCKET c = ws.accept(static_cast<SOCKET>(listen_fd_),
                               reinterpret_cast<sockaddr*>(&peer), &peer_len);
    if (c == INVALID_SOCKET) break;
    refuse_one(c);
    ws.closesocket(c);
  }
}

bool Server::run(std::string* err) {
  if (!start(err)) return false;
  Winsock& ws = winsock();
  while (!stopping()) {
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(static_cast<SOCKET>(listen_fd_), &readfds);
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 200 * 1000;  // poll the stop flag every 200 ms
    const int n = ws.select(listen_fd_ + 1, &readfds, nullptr, nullptr, &tv);
    if (stopping()) break;
    if (n <= 0) continue;
    sockaddr_in peer{};
    int peer_len = sizeof(peer);
    const SOCKET c = ws.accept(static_cast<SOCKET>(listen_fd_),
                               reinterpret_cast<sockaddr*>(&peer), &peer_len);
    if (c == INVALID_SOCKET) continue;
    {
      // Per-connection receive timeout: a client that opens the socket but never
      // finishes its headers/body must not block the single-threaded accept loop.
      int rcv_timeout = 5000;  // ms
      ws.setsockopt(c, SOL_SOCKET, SO_RCVTIMEO,
                    reinterpret_cast<const char*>(&rcv_timeout), sizeof(rcv_timeout));
    }
    char ip[64] = {0};
    std::snprintf(ip, sizeof(ip), "%u.%u.%u.%u",
                  static_cast<unsigned>((peer.sin_addr.s_addr) & 0xFF),
                  static_cast<unsigned>((peer.sin_addr.s_addr >> 8) & 0xFF),
                  static_cast<unsigned>((peer.sin_addr.s_addr >> 16) & 0xFF),
                  static_cast<unsigned>((peer.sin_addr.s_addr >> 24) & 0xFF));
    const std::string client = std::string(ip) + ":" + std::to_string(ntohs_u(peer.sin_port));
    serve_connection(static_cast<intptr_t>(c), client);
  }
  if (listen_fd_ >= 0) {
    // The handler that called stop() (POST /shutdown) has already answered 200;
    // reply 503 to any second client already waiting in the accept backlog.
    if (stopping_) refuse_pending_connections();
    ws.closesocket(static_cast<SOCKET>(listen_fd_));
    listen_fd_ = -1;
  }
  return true;
}

}  // namespace http
}  // namespace engine
