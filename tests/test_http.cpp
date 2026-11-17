// HTTP layer self-test server (no model, no GPU): exercises the WinSock2
// server independently of the engine so the HTTP code can be validated before
// model bring-up.
//
//   build/test_http.exe [--host 127.0.0.1] [--port 7039]
//
// Endpoints used by tools/verify_http.py:
//   GET  /ping               -> {"pong":true}
//   POST /echo               -> {"length":N,"body":<body>}
//   POST /stream             -> chunked text/plain, 5 chunks then end
//   GET  /latin1             -> a non-JSON body
//   any  /status/<code>      -> response with that status code
//   POST /shutdown           -> stops the server
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "engine/http_server.h"

namespace {

std::string json_escape(const std::string& s) {
  std::string r;
  for (unsigned char c : s) {
    switch (c) {
      case '"': r += "\\\""; break;
      case '\\': r += "\\\\"; break;
      case '\n': r += "\\n"; break;
      case '\r': r += "\\r"; break;
      case '\t': r += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", c);
          r += b;
        } else {
          r.push_back(static_cast<char>(c));
        }
    }
  }
  return r;
}

engine::http::Server* g_server = nullptr;

void handler(const engine::http::Request& req, engine::http::Conn& conn) {
  std::fprintf(stderr, "[test_http] %s %s %s body=%zu\n", req.client.c_str(),
               req.method.c_str(), req.path.c_str(), req.body.size());
  std::fflush(stderr);

  if (req.method == "OPTIONS") {
    conn.send(204, "text/plain", "",
              {{"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
               {"Access-Control-Allow-Headers", "Content-Type"}});
    return;
  }
  if (req.method == "GET" && req.path == "/ping") {
    conn.send_json(200, "{\"pong\": true}");
    return;
  }
  if (req.method == "POST" && req.path == "/echo") {
    conn.send_json(200, "{\"length\": " + std::to_string(req.body.size()) + ", \"body\": \"" +
                            json_escape(req.body) + "\"}");
    return;
  }
  if (req.method == "POST" && req.path == "/stream") {
    if (!conn.begin_stream("text/plain; charset=utf-8")) return;
    for (int i = 0; i < 5; ++i) {
      if (!conn.write_chunk("chunk" + std::to_string(i) + "\n")) break;
    }
    conn.end_stream();
    return;
  }
  if (req.method == "GET" && req.path == "/latin1") {
    conn.send(200, "text/plain", std::string("\xE4\xBD\xA0\xE5\xA5\xBD\n", 7));
    return;
  }
  if (req.path.rfind("/status/", 0) == 0) {
    const int code = std::atoi(req.path.c_str() + 8);
    conn.send_json(code, "{\"status\": " + std::to_string(code) + "}");
    return;
  }
  if (req.method == "POST" && req.path == "/shutdown") {
    conn.send_json(200, "{\"status\": \"shutting down\"}");
    if (g_server) g_server->stop();
    return;
  }
  conn.send_error(404, "no such test endpoint");
}

}  // namespace

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  int port = 7039;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) host = argv[++i];
    else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: test_http [--host H] [--port P]\n");
      return 2;
    }
  }
  try {
    engine::http::Server server(host, port);
    server.set_handler(handler);
    g_server = &server;
    std::string err;
    if (!server.start(&err)) {
      std::fprintf(stderr, "cannot start: %s\n", err.c_str());
      return 1;
    }
    std::printf("test_http listening on http://%s:%d\n", host.c_str(), server.port());
    std::fflush(stdout);
    if (!server.run(&err)) {
      std::fprintf(stderr, "serve error: %s\n", err.c_str());
      return 1;
    }
    g_server = nullptr;
    std::printf("test_http stopped\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
