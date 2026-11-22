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
  // Context length: any positive integer is a valid request (0 = auto).  The loader
  // clamps it only when the KV cache physically does not fit; see ctx_requested /
  // ctx_suggested below and the run() contract.
  int ctx = 4096;
  // Original user request (0 = auto), reported by /props as n_ctx_requested even when
  // an internal retry runs with a smaller `ctx`.  run() defaults it to `ctx`.
  int ctx_requested = 0;
  // OUT: when run() returns 4 (measured VRAM guard) this holds a feasible ctx the
  // caller may retry with; 0 otherwise.
  int ctx_suggested = 0;
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
// Returns:
//   0 = clean shutdown
//   1 = fatal startup/serve error
//   4 = requested ctx physically does not fit (measured free VRAM < margin);
//       opts.ctx_suggested holds a feasible value and the caller may retry
//   5 = model load failed while a positive ctx was requested (possible OOM from the
//       requested ctx; the caller may retry with a smaller ctx)
// The options are taken by reference (in/out) because run() writes ctx_suggested back.
int run(Options& opts);

}  // namespace serve
}  // namespace engine
