// `engine serve`: OpenAI-compatible + llama.cpp-style HTTP API for the local
// engine.  One process loads the model once and serves requests serially
// (single-user): a second client simply waits in the accept backlog.
#pragma once

#include <string>

namespace engine {
namespace serve {

struct Options {
  std::string model;
  std::string host = "127.0.0.1";
  int port = 7030;
  int ctx = 4096;
  // Open the built-in chat UI in the default browser once the listener is up
  // (`serve --open-browser`, used by the double-click launcher). No effect when
  // the platform cannot launch a browser.
  bool open_browser = false;
  int default_max_tokens = 256;
  double default_temperature = 0.8;
  double default_top_p = 0.95;
  int default_top_k = 40;
  double default_repetition_penalty = 1.0;
};

// Blocking: loads the model, serves until Ctrl+C / Ctrl+Break / window close.
// Returns 0 on a clean shutdown, 1 on a fatal startup/serve error.
int run(const Options& opts);

}  // namespace serve
}  // namespace engine
