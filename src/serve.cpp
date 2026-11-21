// `engine serve`: OpenAI-compatible and llama.cpp-style HTTP endpoints on top of
// the existing DenseRunner + the GGUF tokenizer.
//
//   POST /v1/chat/completions   OpenAI chat (messages[], stream, sampling params,
//                               enable_thinking / chat_template_kwargs)
//   POST /v1/completions        OpenAI legacy completions (prompt)
//   POST /completion            llama.cpp style (prompt, n_predict, stream)
//   POST /tokenize /detokenize  llama.cpp style helpers (used by the test suite)
//   GET  /health /v1/models /props /  (GET / is a small browser chat UI)
//
// Requests are handled serially by the single-threaded HTTP server, so only one
// generation ever runs on the GPU; Ctrl+C / Ctrl+Break / window close stops the
// accept loop and frees the VRAM before exiting.
#include "engine/config.h"
#include "engine/serve.h"
#include "engine/serve_guard.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#include <shellapi.h>
#endif

#include "engine/config.h"
#include "engine/gguf.h"
#include "engine/http_server.h"
#include "engine/json.h"
#include "engine/json_parse.h"
#include "engine/runner.h"
#include "engine/sampler.h"
#include "engine/text_util.h"
#include "engine/tokenizer.h"

namespace engine {
namespace serve {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int64_t now_unix() {
  return static_cast<int64_t>(std::time(nullptr));
}

std::string random_hex(size_t bytes) {
  static std::atomic<uint64_t> counter{0};
  const uint64_t seed = sampler::resolve_seed(-1) ^ (counter.fetch_add(1) * 0x9E3779B97F4A7C15ull);
  sampler::Rng rng(seed);
  static const char* hex = "0123456789abcdef";
  std::string out;
  out.reserve(bytes * 2);
  for (size_t i = 0; i < bytes; ++i) {
    const uint64_t v = rng.next_u64();
    out.push_back(hex[(v >> 4) & 0xF]);
    out.push_back(hex[v & 0xF]);
  }
  return out;
}

// Runtime error that maps to HTTP 400 (bad request), not 500.
class UserError : public std::runtime_error {
 public:
  explicit UserError(const std::string& msg) : std::runtime_error(msg) {}
};

// ---------------------------------------------------------------------------
// Request parsing helpers
// ---------------------------------------------------------------------------
std::string content_to_text(const jsonp::Value* v) {
  if (!v) return "";
  if (v->is_string()) return v->str;
  if (v->is_array()) {
    std::string out;
    for (const auto& item : v->arr) {
      if (item.is_string()) {
        out += item.str;
      } else if (item.is_object()) {
        const std::string type = item.get_str("type", "text");
        if (type == "text") {
          const jsonp::Value* t = item.find("text");
          if (t && t->is_string()) out += t->str;
        }
        // image_url / input_audio parts are ignored (text-only server)
      }
    }
    return out;
  }
  return "";
}

std::vector<tokenizer::ChatMessage> parse_messages(const jsonp::Value& body) {
  const jsonp::Value* msgs = body.find("messages");
  if (!msgs || !msgs->is_array()) throw UserError("'messages' must be an array");
  std::vector<tokenizer::ChatMessage> out;
  out.reserve(msgs->arr.size());
  for (const auto& m : msgs->arr) {
    if (!m.is_object()) throw UserError("every messages[] entry must be an object");
    tokenizer::ChatMessage msg;
    msg.role = m.get_str("role", "user");
    msg.content = content_to_text(m.find("content"));
    msg.reasoning_content = m.get_str("reasoning_content", "");
    out.push_back(std::move(msg));
  }
  if (out.empty()) throw UserError("'messages' must not be empty");
  return out;
}

std::vector<std::string> parse_stop(const jsonp::Value& body) {
  std::vector<std::string> stops;
  const jsonp::Value* stop = body.find("stop");
  if (!stop) return stops;
  if (stop->is_string()) {
    if (!stop->str.empty()) stops.push_back(stop->str);
  } else if (stop->is_array()) {
    for (const auto& s : stop->arr)
      if (s.is_string() && !s.str.empty()) stops.push_back(s.str);
  }
  return stops;
}

double num_or(const jsonp::Value& body, const char* key, double fallback) {
  const jsonp::Value* v = body.find(key);
  if (!v) return fallback;
  if (v->is_number()) return v->num;
  return fallback;
}

int int_or(const jsonp::Value& body, const char* key, int fallback) {
  const jsonp::Value* v = body.find(key);
  if (!v) return fallback;
  if (v->is_number()) return static_cast<int>(v->num);
  return fallback;
}

bool bool_or(const jsonp::Value& body, const char* key, bool fallback) {
  const jsonp::Value* v = body.find(key);
  if (!v) return fallback;
  if (v->type == jsonp::Value::Type::Bool) return v->b;
  return fallback;
}

// ---------------------------------------------------------------------------
// Generation parameters / output
// ---------------------------------------------------------------------------
struct GenParams {
  int max_tokens = 256;
  double temperature = 0.8;
  double top_p = 0.95;
  double min_p = 0.0;
  int top_k = 40;
  double repetition_penalty = 1.0;
  int seed = -1;
  std::vector<std::string> stop;
  bool thinking_active = false;  // prompt ends inside an open <think> block
  bool prefix_cache = true;      // request body `cache_prompt` (default on)
};

struct GenOutcome {
  std::string reasoning;   // text before </think> (empty when not thinking)
  std::string content;     // answer text
  std::string raw_text;    // everything, including the thinking part
  std::vector<int> token_ids;
  int prompt_tokens = 0;
  int gen_tokens = 0;
  const char* finish_reason = "stop";
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  int cache_hit_tokens = 0;   // prefix-cache tokens reused by this request
  int prefilled_tokens = 0;   // tokens actually fed through forward()
  bool think_budget_hit = false;  // forced </think> because thinking ran too long
  bool repeat_guard_hit = false;  // generation stopped because it looped
};

// Single-slot multi-turn prefix cache.  The conversation-history part of a chat
// prompt is prefilled once; its recurrent state (hybrid models) is snapshotted and
// the attention KV cache stays in place, so the next turn only prefills the new
// tokens (assistant reply + new user turn + generation prompt) instead of the
// whole conversation.
struct PrefixCache {
  bool enabled = true;
  bool have_state = false;          // slot 0 holds the state for state_ids
  std::vector<int> state_ids;       // token ids covered by slot 0
  uint64_t hits = 0, misses = 0;    // requests
  uint64_t reused_tokens = 0;       // cumulative tokens saved from prefill
  uint64_t prefilled_tokens = 0;    // cumulative tokens actually prefilled
  int last_hit_tokens = 0;
  bool alloc_failed = false;        // hybrid state slot could not be allocated
};

struct ServerState {
  tokenizer::Tokenizer tk;
  std::unique_ptr<runner::DenseRunner> model;
  Options opts;
  int ctx = 4096;
  std::string model_name;
  bool thinking_aware = false;  // chat template opens a <think> block
  PrefixCache pc;
  int cache_align = 0;          // prefix-cache snapshot alignment (0 = prefill chunk)
  int no_limit_tokens = 1024;   // reply cap for max_tokens <= 0 (webui sends -1)
  int think_budget = 512;       // force </think> after N thinking tokens (0 = off)
  int repeat_guard = 2;         // loop guard level: 0 off, 1 sentences, 2 +chars/phrases
  bool mtp_enabled = false;     // --mtp and the loaded model has a usable MTP block
  int mtp_k = 2;                // drafts per speculative round
  std::atomic<bool> busy{false};
};

ServerState* g_state = nullptr;
http::Server* g_server = nullptr;

#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD type) {
  if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT ||
      type == CTRL_SHUTDOWN_EVENT) {
    if (g_server) g_server->stop();
  }
  return TRUE;
}
#endif

// ---------------------------------------------------------------------------
// The generation loop
// ---------------------------------------------------------------------------
// on_delta(text, is_reasoning) is called with UTF-8 text as it becomes
// available; returning false aborts generation (used when the client has gone).
// JSON body parse: syntax errors are client errors (400), not server errors.
jsonp::Value parse_json_body(const http::Request& req) {
  try {
    return jsonp::parse(req.body);
  } catch (const std::exception& e) {
    throw UserError(std::string("invalid JSON body: ") + e.what());
  }
}

// Cheap validation used before an HTTP response (and especially before an SSE
// stream) is started: generate() must not throw after the 200 header went out.
void validate_prompt(const ServerState& st, const std::string& prompt, const GenParams& gp) {
  const std::vector<int> ids = st.tk.encode(prompt, false, true);
  if (ids.empty()) throw UserError("prompt encodes to zero tokens");
  if (static_cast<int>(ids.size()) >= st.ctx)
    throw UserError("prompt is too long: " + std::to_string(ids.size()) + " tokens >= context " +
                    std::to_string(st.ctx));
  if (std::min(gp.max_tokens, st.ctx - static_cast<int>(ids.size())) <= 0)
    throw UserError("no room left to generate (context full)");
}

GenOutcome generate(ServerState& st, const std::string& prompt, const std::string& history_prompt,
                    const GenParams& gp,
                    const std::function<bool(const std::string&, bool)>& on_delta) {
  GenOutcome out;
  tokenizer::Tokenizer& tk = st.tk;
  runner::DenseRunner& model = *st.model;

  std::vector<int> ids = tk.encode(prompt, /*add_special=*/false, /*parse_special=*/true);
  if (ids.empty()) throw UserError("prompt encodes to zero tokens");
  out.prompt_tokens = static_cast<int>(ids.size());
  if (static_cast<int>(ids.size()) >= st.ctx)
    throw UserError("prompt is too long: " + std::to_string(ids.size()) + " tokens >= context " +
                    std::to_string(st.ctx));
  int max_new = std::min(gp.max_tokens, st.ctx - static_cast<int>(ids.size()));
  if (max_new <= 0) throw UserError("no room left to generate (context full)");

  // ---- prefix-cache boundary ------------------------------------------------
  // tokens[0:boundary) is the rendered conversation history; everything after it
  // is the generation prompt (and is always re-prefilled).  Raw /completion calls
  // pass no history and therefore skip the cache.
  int boundary = static_cast<int>(ids.size());
  if (!history_prompt.empty()) {
    const std::vector<int> hid = tk.encode(history_prompt, false, true);
    if (hid.size() <= ids.size() && std::equal(hid.begin(), hid.end(), ids.begin()))
      boundary = static_cast<int>(hid.size());
  }
  // Cache point alignment.  forward() chunks long prompts in rows_ tokens; the
  // snapshot point used to be aligned to rows_ too, which with the large serve
  // chunk (512-1024) makes a multi-turn cache hit re-prefill up to rows_-1 extra
  // history tokens every turn (measured 820 tokens / 1.2 s on a 9.1k chat).
  // `serve_cache_align` (default 128; 0 = follow the prefill chunk) decouples them: the snapshot
  // stays on a fine grid (e.g. 128) while forward() still uses the large chunk for
  // long prompts.  The cached path may then run a partial history chunk; that is
  // verified token-for-token against the full-prefill path
  // (tools/serve_prefix_cache_check.py --filler-repeat ...).
  const int chunk_rows = std::max(1, model.prefill_chunk());
  const int cache_align = st.cache_align > 0 ? st.cache_align : chunk_rows;
  const int reuse_pos = (boundary / cache_align) * cache_align;
  const bool cacheable = st.pc.enabled && gp.prefix_cache && reuse_pos > 0 &&
                         reuse_pos < static_cast<int>(ids.size());
  int start_from = 0;
  bool cache_hit = false;
  double restore_ms = 0.0, save_ms = 0.0, chunk_ms = 0.0, tail_ms = 0.0;
  int chunk_tokens = 0;
  if (cacheable && st.pc.have_state &&
      st.pc.state_ids.size() <= static_cast<size_t>(reuse_pos)) {
    const int n = static_cast<int>(st.pc.state_ids.size());
    int lcp = 0;
    while (lcp < n && ids[static_cast<size_t>(lcp)] == st.pc.state_ids[static_cast<size_t>(lcp)])
      ++lcp;
    if (lcp == n && n > 0) {
      const auto t_r0 = Clock::now();
      const bool restored = st.model->prefix_state_restore(0);
      restore_ms = ms_since(t_r0);
      if (restored) {
        start_from = n;
        cache_hit = true;
      } else {
        st.pc.have_state = false;
      }
    }
  }

  const auto t_pre0 = Clock::now();
  std::vector<float> logits;
  if (!cacheable) {
    logits = model.forward(ids, 0, /*last_row_only=*/true);
    out.prefilled_tokens = static_cast<int>(ids.size());
  } else {
    // 1) history chunk up to the chunk-aligned cache point (only its last row is
    //    requested; the logits come from the tail below).  Its length is a
    //    multiple of chunk_rows, so forward() chunks it exactly like the full
    //    prefill would for the same absolute positions.
    if (reuse_pos > start_from) {
      const std::vector<int> chunk(ids.begin() + start_from, ids.begin() + reuse_pos);
      const auto t_c0 = Clock::now();
      model.forward(chunk, start_from, /*last_row_only=*/true);
      chunk_ms = ms_since(t_c0);
      chunk_tokens = static_cast<int>(chunk.size());
    }
    // 2) snapshot the recurrent state at the aligned cache point for the next turn.
    if (st.pc.enabled && gp.prefix_cache) {
      const auto t_s0 = Clock::now();
      const bool saved = model.prefix_state_save(0);
      save_ms = ms_since(t_s0);
      if (saved) {
        st.pc.have_state = true;
        st.pc.state_ids.assign(ids.begin(), ids.begin() + reuse_pos);
      } else {
        st.pc.have_state = false;
        st.pc.alloc_failed = true;
      }
    }
    // 3) remainder (history tail + generation prompt): always prefilled; its last
    //    row is the logits.  Its chunks are the canonical ones as well.
    const std::vector<int> tail(ids.begin() + reuse_pos, ids.end());
    const auto t_t0 = Clock::now();
    logits = model.forward(tail, reuse_pos, /*last_row_only=*/true);
    tail_ms = ms_since(t_t0);
    out.cache_hit_tokens = start_from;
    out.prefilled_tokens = static_cast<int>(ids.size()) - start_from;
    st.pc.last_hit_tokens = start_from;
    if (cache_hit) {
      ++st.pc.hits;
      st.pc.reused_tokens += static_cast<uint64_t>(start_from);
    } else {
      ++st.pc.misses;
    }
    st.pc.prefilled_tokens += static_cast<uint64_t>(out.prefilled_tokens);
  }
  out.prefill_ms = ms_since(t_pre0);
  static const bool pf_trace = std::getenv("ENGINE_SERVE_PREFILL_TRACE") != nullptr;
  if (pf_trace)
    std::fprintf(stderr,
                 "[serve] prefill-trace: total=%.1f ms = restore=%.1f + save=%.1f + "
                 "chunk=%.1f(%d tok) + tail=%.1f(%d tok) [cached=%d prefilled=%d]\n",
                 out.prefill_ms, restore_ms, save_ms, chunk_ms, chunk_tokens, tail_ms,
                 static_cast<int>(ids.size()) - reuse_pos, out.cache_hit_tokens,
                 out.prefilled_tokens);
  const int vocab = model.vocab();
  if (static_cast<int>(logits.size()) < vocab)
    throw std::runtime_error("model returned fewer logits than the vocabulary");
  // forward(last_row_only=true) returns exactly one vocabulary row.  Move it instead
  // of copying ~1 MB (this used to be a hidden per-request cost).
  std::vector<float> last;
  if (static_cast<int>(logits.size()) == vocab) {
    last = std::move(logits);
  } else {
    last.assign(logits.end() - vocab, logits.end());
  }

  sampler::Sampler sampler(0);
  sampler::Params sp;
  sp.temperature = gp.temperature;
  sp.top_p = gp.top_p;
  sp.min_p = gp.min_p;
  sp.top_k = gp.top_k;
  sp.repetition_penalty = gp.repetition_penalty;
  sp.seed = gp.seed;
  sampler.set_params(sp);
  sampler.reseed(sampler::resolve_seed(gp.seed));

  const bool greedy = !(gp.temperature > 0.0);

  // ---- stop-string holdback + thinking/content split ----------------------
  size_t max_stop = 0;
  for (const auto& s : gp.stop) max_stop = std::max(max_stop, s.size());
  std::string pending;                       // decoded bytes not yet emitted
  bool stop_hit = false;
  bool in_think = gp.thinking_active;
  bool aborted = false;

  // ---- degenerate-loop guard (sentence / char levels, see engine/serve_guard.h) ----
  engine::serve::RepeatGuard guard;
  guard.configure(st.repeat_guard);
  bool repeat_stop = false;

  auto feed_emitted = [&](const std::string& text) {
    if (text.empty()) return;
    out.raw_text += text;
    if (guard.update(text, gp.thinking_active && in_think)) {
      repeat_stop = true;
      out.repeat_guard_hit = true;
    }
    if (!gp.thinking_active) {
      out.content += text;
      if (!on_delta(text, false)) aborted = true;
      return;
    }
    if (!in_think) {
      out.content += text;
      if (!on_delta(text, false)) aborted = true;
      return;
    }
    const size_t p = text.find("</think>");
    if (p == std::string::npos) {
      out.reasoning += text;
      if (!on_delta(text, true)) aborted = true;
      return;
    }
    if (p > 0) {
      out.reasoning += text.substr(0, p);
      if (!on_delta(text.substr(0, p), true)) aborted = true;
    }
    in_think = false;
    std::string rest = text.substr(p + std::strlen("</think>"));
    // drop the newlines the model emits right after closing the think block
    size_t b = 0;
    while (b < rest.size() && rest[b] == '\n') ++b;
    rest = rest.substr(b);
    if (!rest.empty()) {
      out.content += rest;
      if (!on_delta(rest, false)) aborted = true;
    }
  };

  auto flush = [&](bool force) -> bool {  // false = stop string hit
    while (true) {
      size_t stop_pos = std::string::npos;
      for (const auto& s : gp.stop) {
        const size_t p = pending.find(s);
        if (p != std::string::npos && (stop_pos == std::string::npos || p < stop_pos))
          stop_pos = p;
      }
      if (stop_pos != std::string::npos) {
        feed_emitted(pending.substr(0, stop_pos));
        pending.clear();
        stop_hit = true;
        return false;
      }
      if (force) {
        feed_emitted(pending);
        pending.clear();
        return true;
      }
      const size_t keep = max_stop > 1 ? max_stop - 1 : 0;
      if (pending.size() <= keep) return true;
      size_t cut = pending.size() - keep;
      // never emit half of a UTF-8 sequence
      while (cut > 0 && (static_cast<unsigned char>(pending[cut]) & 0xC0) == 0x80) --cut;
      const size_t incomplete = text::incomplete_utf8_tail(pending.substr(0, cut));
      if (incomplete >= cut) return true;
      cut -= incomplete;
      if (cut == 0) return true;
      feed_emitted(pending.substr(0, cut));
      pending.erase(0, cut);
      if (aborted) return true;
    }
  };

  // ---- think budget: force </think> when the thinking block never closes ----
  // Qwen3.x thinking models sometimes self-narrate for hundreds of tokens and
  // never emit </think>; without a guard the whole max_tokens budget is spent
  // inside the thought block and the user receives no answer at all.
  std::vector<int> think_close_ids;
  if (gp.thinking_active && st.think_budget > 0) {
    think_close_ids = tk.encode("</think>", /*add_special=*/false, /*parse_special=*/true);
    if (think_close_ids.empty())
      think_close_ids = tk.encode("</think>", /*add_special=*/false,
                                  /*parse_special=*/false);
  }
  size_t think_close_at = 0;
  bool think_forcing = false;  // currently feeding the forced </think> token(s)

  // ---- prefill context bookkeeping ---------------------------------------
  std::vector<int> context = ids;  // prompt + generated (for repetition penalty)
  int pos = static_cast<int>(ids.size());
  int next = greedy ? sampler::Sampler::argmax(last)
                    : sampler.sample_bounded(last, context);
  std::vector<float>().swap(last);  // release the 1 MB prefill row after the first pick
  int produced = 0;

  // ---- optional MTP speculative decoding (greedy + single-call prefill only) ----
  // spec_begin() needs the prompt residual rows in x_, so it is limited to the
  // non-prefix-cache prefill path (first turn / short prompts).  Sampling requests
  // and prefix-cache-hit turns keep the plain token-by-token loop.
  bool spec_active = false;
  std::vector<int> spec_emit;
  if (greedy && st.mtp_enabled && !cacheable) {
    spec_active = model.spec_begin(ids, next, static_cast<int>(ids.size()) - 1);
    if (spec_active)
      std::fprintf(stderr, "[serve] mtp: speculative decoding active (k=%d)\n", st.mtp_k);
  }

  const auto t_dec0 = Clock::now();
  while (true) {
    int cur;
    if (spec_active) {
      if (spec_emit.empty()) {
        if (!model.spec_round(st.mtp_k, &spec_emit)) {
          // context nearly exhausted: stop cleanly at the current position
          spec_active = false;
          out.finish_reason = "length";
          break;
        }
      }
      cur = spec_emit.front();
      spec_emit.erase(spec_emit.begin());
    } else {
      cur = next;
    }
    // still thinking after budget tokens -> switch to the forced </think> token.
    // Cap the effective budget to max_new-1: if max_tokens <= think_budget the old
    // check could never run (the length break happened first) and the reply ended
    // with an empty answer.
    const int budget_eff =
        (st.think_budget > 0)
            ? std::min(st.think_budget, std::max(1, max_new - 1))
            : 0;
    if (!think_forcing && in_think && !think_close_ids.empty() && budget_eff > 0 &&
        produced >= budget_eff) {
      if (spec_active) {
        // a verified batch cannot carry an injected token: stop here and let the
        // chat handler re-run the answer with thinking disabled
        if (!flush(true)) {
          out.finish_reason = "stop";
          break;
        }
        feed_emitted("</think>");
        out.think_budget_hit = true;
        std::fprintf(stderr,
                     "[serve] WARN think-budget: forced </think> after %d tokens "
                     "(model did not close its thinking block)\n",
                     produced);
        out.finish_reason = "stop";
        break;
      }
      think_forcing = true;
      think_close_at = 0;
    }
    if (think_forcing) cur = think_close_ids[think_close_at];
    if (cur < 0) {
      out.finish_reason = "error";
      break;
    }
    // end of generation token: stop without emitting it
    if (tk.is_eog(cur)) {
      out.finish_reason = "stop";
      break;
    }
    out.token_ids.push_back(cur);
    ++produced;
    context.push_back(cur);
    if (think_forcing) {
      // the forced </think> is a special token (its piece is empty): feed it to
      // the model as context, and once the last forced id is consumed emit the
      // literal closing tag so the streaming client folds the thought block.
      ++think_close_at;
      if (think_close_at >= think_close_ids.size()) {
        think_forcing = false;
        out.think_budget_hit = true;
        flush(true);  // any buffered thinking text must precede the closing tag
        feed_emitted("</think>");
        std::fprintf(stderr,
                     "[serve] WARN think-budget: forced </think> after %d tokens "
                     "(model did not close its thinking block)\n",
                     produced);
        // Do NOT keep generating in this state: the model would continue its
        // planning loop as visible content.  The chat handler re-runs the answer
        // with thinking disabled (cheap thanks to the prefix cache).
        out.finish_reason = "stop";
        break;
      }
    } else {
      pending += tk.token_piece(cur, /*skip_special=*/true);
    }
    if (!flush(false)) {
      out.finish_reason = "stop";
      break;
    }
    if (aborted) {
      out.finish_reason = "aborted";
      break;
    }
    if (repeat_stop) {
      out.finish_reason = "repetition";
      std::fprintf(stderr,
                   "[serve] WARN repeat-guard(level %d): stopped a degenerate loop "
                   "after %d generated tokens (rule=%s, tail=%s)\n",
                   st.repeat_guard, produced, guard.rule().c_str(),
                   guard.snippet().c_str());
      break;
    }
    if (produced >= max_new) {
      out.finish_reason = "length";
      break;
    }
    if (spec_active) continue;  // the next tokens come from the next spec round
    // plain single-token forward to obtain the logits for the next token
    if (greedy) {
      runner::DenseRunner::TopKResult tkr;
      auto row = model.forward_topk({cur}, pos, 5, &tkr);
      ++pos;
      if (!row.empty()) {
        // device top-k header disabled -> the full row came back: argmax in place
        next = sampler::Sampler::argmax(row);
      } else if (!tkr.idx.empty()) {
        next = tkr.idx[0];
      } else {
        out.finish_reason = "error";
        break;
      }
    } else {
      auto row = model.forward({cur}, pos, /*last_row_only=*/true);
      ++pos;
      if (static_cast<int>(row.size()) < vocab) {
        out.finish_reason = "error";
        break;
      }
      // sample directly from the moved-in row: no second 1 MB copy, and the
      // candidate set is selected inside the sampler (top_k>0: exact top-k;
      // top_k=0: top-1024 approximation, see /props.sampling and docs/CONFIG.md).
      next = sampler.sample_bounded(row, context);
    }
  }
  if (spec_active) model.spec_end();

  if (!stop_hit) flush(true);
  out.decode_ms = ms_since(t_dec0);
  out.gen_tokens = produced;
  return out;
}

// ---------------------------------------------------------------------------
// JSON writers
// ---------------------------------------------------------------------------
void write_usage(json::Writer& w, const GenOutcome& r) {
  w.key("usage");
  w.begin_obj();
  w.key("prompt_tokens");
  w.integer(r.prompt_tokens);
  w.key("completion_tokens");
  w.integer(r.gen_tokens);
  w.key("total_tokens");
  w.integer(r.prompt_tokens + r.gen_tokens);
  w.end_obj();
}

void write_timings(json::Writer& w, const GenOutcome& r) {
  w.key("timings");
  w.begin_obj();
  w.key("prefill_ms");
  w.number(r.prefill_ms);
  w.key("decode_ms");
  w.number(r.decode_ms);
  const double tps = r.decode_ms > 0 ? r.gen_tokens * 1000.0 / r.decode_ms : 0.0;
  w.key("tokens_per_second");
  w.number(tps);
  w.key("cache_hit_tokens");
  w.integer(r.cache_hit_tokens);
  w.key("prefilled_tokens");
  w.integer(r.prefilled_tokens);
  w.key("think_budget_hit");
  w.boolean(r.think_budget_hit);
  w.key("repeat_guard_hit");
  w.boolean(r.repeat_guard_hit);
  w.end_obj();
  w.key("generated_token_ids");
  w.begin_arr();
  for (int id : r.token_ids) w.integer(id);
  w.end_arr();
}

// Streaming responses end with a choices[]-finish chunk instead of a full body.
// Clients that show a tok/s figure need the token counts there too, so the last
// SSE chunk carries the same `usage` object (plus decode timings) as the
// non-streaming response.  generated_token_ids is deliberately omitted: a long
// generation would otherwise dump a large array into the final chunk.
void write_stream_usage(json::Writer& w, const GenOutcome& r) {
  write_usage(w, r);
  w.key("timings");
  w.begin_obj();
  // llama.cpp names (the official webui reads these to show t/s) ...
  w.key("prompt_n");
  w.integer(r.prompt_tokens);
  w.key("prompt_ms");
  w.number(r.prefill_ms);
  w.key("predicted_n");
  w.integer(r.gen_tokens);
  w.key("predicted_ms");
  w.number(r.decode_ms);
  // ... plus the engine's own names for the built-in fallback page.
  w.key("prefill_ms");
  w.number(r.prefill_ms);
  w.key("decode_ms");
  w.number(r.decode_ms);
  w.key("tokens_per_second");
  w.number(r.decode_ms > 0 ? r.gen_tokens * 1000.0 / r.decode_ms : 0.0);
  w.key("cache_hit_tokens");
  w.integer(r.cache_hit_tokens);
  w.key("prefilled_tokens");
  w.integer(r.prefilled_tokens);
  w.key("think_budget_hit");
  w.boolean(r.think_budget_hit);
  w.key("repeat_guard_hit");
  w.boolean(r.repeat_guard_hit);
  w.end_obj();
}

void write_generation_settings(json::Writer& w, const GenParams& gp) {
  w.key("generation_settings");
  w.begin_obj();
  w.key("temperature");
  w.number(gp.temperature);
  w.key("top_p");
  w.number(gp.top_p);
  w.key("top_k");
  w.integer(gp.top_k);
  w.key("min_p");
  w.number(gp.min_p);
  w.key("repeat_penalty");
  w.number(gp.repetition_penalty);
  w.key("seed");
  w.integer(gp.seed);
  w.key("n_predict");
  w.integer(gp.max_tokens);
  w.end_obj();
}

// SSE helpers -------------------------------------------------------------
// json::Writer pretty-prints (newline + indentation).  SSE events must be one
// physical line per "data:" field for clients that parse line-by-line, so the
// formatting newlines are stripped here (JSON allows whitespace between tokens).
std::string sse_single_line(const std::string& payload) {
  std::string out;
  out.reserve(payload.size());
  for (char c : payload) {
    if (c != '\n' && c != '\r') out.push_back(c);
  }
  return out;
}

void sse_data(http::Conn& conn, const std::string& payload) {
  conn.write_chunk("data: " + sse_single_line(payload) + "\n\n");
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------
GenParams params_from_body(const ServerState& st, const jsonp::Value& body) {
  GenParams gp;
  // Default: no token cap beyond the context window.  Qwen3.x-style models emit a long
  //  thinking block first, so a small default (256) truncated every reply before the
  // answer even started (content empty, finish=length).  "思考长度不限" = let it run to
  // the context limit unless the client asks for less.
  gp.max_tokens = int_or(body, "max_tokens",
                         int_or(body, "max_completion_tokens",
                                int_or(body, "n_predict", st.opts.default_max_tokens)));
  // llama.cpp clients (including the official webui) send max_tokens:-1 for
  // "no limit beyond the context".  Fully unlimited means a thinking model can
  // spend minutes in one reply, so this maps to serve_no_limit_tokens (default
  // 1024, configurable) and never exceeds the context window.
  if (gp.max_tokens <= 0) {
    const int cap = st.no_limit_tokens > 0 ? st.no_limit_tokens : st.ctx;
    gp.max_tokens = std::min(st.ctx, cap);
  }
  gp.max_tokens = std::max(1, std::min(gp.max_tokens, st.ctx));
  gp.temperature = num_or(body, "temperature", st.opts.default_temperature);
  if (gp.temperature < 0.0) gp.temperature = 0.0;
  gp.top_p = num_or(body, "top_p", st.opts.default_top_p);
  gp.min_p = num_or(body, "min_p", 0.0);
  gp.top_k = int_or(body, "top_k", st.opts.default_top_k);
  gp.repetition_penalty =
      num_or(body, "repetition_penalty",
             num_or(body, "repeat_penalty", st.opts.default_repetition_penalty));
  gp.seed = int_or(body, "seed", -1);
  gp.stop = parse_stop(body);
  return gp;
}

void handle_health(const ServerState& st, http::Conn& conn) {
  json::Writer w;
  w.begin_obj();
  w.key("status");
  w.str("ok");
  w.key("model");
  w.str(st.model_name);
  w.key("n_ctx");
  w.integer(st.ctx);
  w.key("slots_idle");
  w.integer(st.busy.load() ? 0 : 1);
  w.key("slots_processing");
  w.integer(st.busy.load() ? 1 : 0);
  w.end_obj();
  conn.send_json(200, w.out);
}

void handle_models(const ServerState& st, http::Conn& conn) {
  json::Writer w;
  w.begin_obj();
  w.key("object");
  w.str("list");
  w.key("data");
  w.begin_arr();
  w.begin_obj();
  w.key("id");
  w.str(st.model_name);
  w.key("object");
  w.str("model");
  w.key("owned_by");
  w.str("engine");
  w.key("created");
  w.integer(now_unix());
  w.end_obj();
  w.end_arr();
  w.end_obj();
  conn.send_json(200, w.out);
}

void handle_props(const ServerState& st, http::Conn& conn) {
  const ir::Model& m = st.model->ir();
  json::Writer w;
  w.begin_obj();
  w.key("model_path");
  w.str(st.opts.model);
  w.key("model_name");
  w.str(st.model_name);
  // llama.cpp /props aliases used by its webui.
  w.key("model_alias");
  w.str(st.model_name);
  w.key("modalities");
  w.begin_arr();
  w.end_arr();
  w.key("n_ctx");
  w.integer(st.ctx);
  w.key("ctx_train");
  w.integer(m.ctx_train);
  w.key("n_vocab");
  w.integer(st.tk.vocab_size());
  w.key("arch");
  w.str(m.arch);
  w.key("container");
  w.str(m.container);
  w.key("file_size");
  w.uinteger(m.file_size);
  w.key("n_layer");
  w.integer(m.n_layer);
  w.key("hidden");
  w.integer(m.hidden);
  w.key("ffn");
  w.integer(m.ffn);
  w.key("mtp_layers");
  w.integer(m.mtp_layers);
  w.key("quant_mix");
  w.begin_obj();
  for (const auto& [q, bytes] : m.quant_mix) {
    w.key(q);
    w.uinteger(bytes);
  }
  w.end_obj();
  w.key("attn");
  w.begin_obj();
  w.key("variant");
  w.str(m.attn.variant);
  w.key("n_head");
  w.integer(m.attn.n_head);
  w.key("n_head_kv");
  w.integer(m.attn.n_head_kv);
  w.key("key_length");
  w.integer(m.attn.key_length);
  w.key("value_length");
  w.integer(m.attn.value_length);
  w.key("rope_freq_base");
  w.number(m.attn.rope_freq_base);
  w.key("full_attention_interval");
  w.integer(m.attn.full_attention_interval);
  w.end_obj();
  w.key("moe");
  w.begin_obj();
  w.key("n_expert");
  w.integer(m.moe.n_expert);
  w.key("n_expert_used");
  w.integer(m.moe.n_expert_used);
  w.key("n_shared");
  w.integer(m.moe.n_shared);
  w.key("expert_ffn_length");
  w.integer(m.moe.expert_ffn_length);
  w.end_obj();
  w.key("ssm");
  w.begin_obj();
  w.key("variant");
  w.str(m.ssm.variant);
  w.key("conv_kernel");
  w.integer(m.ssm.conv_kernel);
  w.end_obj();
  w.key("tokenizer");
  w.begin_obj();
  w.key("model");
  w.str(st.tk.kind() == tokenizer::Kind::Bpe ? "gpt2" : "llama");
  w.key("pre");
  w.str(st.tk.pre());
  w.key("n_vocab");
  w.integer(st.tk.vocab_size());
  w.key("bos_token_id");
  w.integer(st.tk.bos_id());
  w.key("eos_token_id");
  w.integer(st.tk.eos_id());
  w.key("chat_template_format");
  w.str(st.tk.template_name());
  w.key("thinking_aware");
  w.boolean(st.thinking_aware);
  w.end_obj();
  w.key("chat_template");
  w.str(st.tk.chat_template());
  w.key("prefix_cache");
  w.begin_obj();
  w.key("enabled");
  w.boolean(st.pc.enabled);
  w.key("align");
  w.integer(st.cache_align);
  w.key("state_supported");  // hybrid model: recurrent state snapshot needed
  w.boolean(st.model->prefix_state_supported());
  w.key("state_bytes");
  w.uinteger(static_cast<uint64_t>(st.model->prefix_state_bytes()));
  w.key("have_snapshot");
  w.boolean(st.pc.have_state);
  w.key("cached_tokens");
  w.integer(static_cast<int64_t>(st.pc.state_ids.size()));
  w.key("hits");
  w.uinteger(st.pc.hits);
  w.key("misses");
  w.uinteger(st.pc.misses);
  w.key("reused_tokens");
  w.uinteger(st.pc.reused_tokens);
  w.key("prefilled_tokens");
  w.uinteger(st.pc.prefilled_tokens);
  w.key("last_hit_tokens");
  w.integer(st.pc.last_hit_tokens);
  w.key("alloc_failed");
  w.boolean(st.pc.alloc_failed);
  w.end_obj();
  w.key("no_limit_tokens");
  w.integer(st.no_limit_tokens);
  w.key("think_budget");
  w.integer(st.think_budget);
  w.key("repeat_guard");
  w.integer(st.repeat_guard);
  w.key("mtp");
  w.begin_obj();
  w.key("available");
  w.boolean(st.model->mtp_available());
  w.key("enabled");
  w.boolean(st.mtp_enabled);
  w.key("k");
  w.integer(st.mtp_k);
  w.end_obj();
  w.key("default_generation_settings");
  w.begin_obj();
  w.key("n_predict");
  w.integer(st.opts.default_max_tokens);
  w.key("temperature");
  w.number(st.opts.default_temperature);
  w.key("top_p");
  w.number(st.opts.default_top_p);
  w.key("top_k");
  w.integer(st.opts.default_top_k);
  w.key("repeat_penalty");
  w.number(st.opts.default_repetition_penalty);
  w.end_obj();
  // How non-greedy sampling is implemented (perf(serve) change 2026-11-21):
  // the sampler chain runs on a bounded candidate set instead of the full
  // vocabulary.  Exposed so benchmark tools / UIs can see the approximation.
  w.key("sampling");
  w.begin_obj();
  w.key("implementation");
  w.str("bounded-candidates");
  w.key("candidate_cap");
  w.integer(sampler::Sampler::kDefaultCandidateCap);
  w.key("top_k_candidates");
  w.str("max(top_k,64)");
  w.key("top_k_zero");
  w.str("approx-top-1024");
  w.key("greedy");
  w.str("exact argmax over the full vocabulary (unchanged)");
  w.key("notes");
  w.str("non-greedy: repetition penalty/top_k/top_p/min_p/temperature are evaluated on "
        "the candidate set; top_p/min_p probabilities are renormalised over it; with "
        "top_k=0 the nuclei are approximated by the top-1024 logits (long-tail mass "
        "below is truncated). top_k>0 keeps the exact top-k set.");
  w.end_obj();
  w.key("host");
  w.str(st.opts.host);
  w.key("port");
  w.integer(st.opts.port);
  w.key("build_info");
  w.str("engine serve (c++17, WinSock2, zero third-party deps)");
  w.end_obj();
  conn.send_json(200, w.out);
}

void log_request(const ServerState& st, const http::Request& req, int status,
                 const GenOutcome* r, double wall_ms) {
  if (r) {
    std::fprintf(stderr,
                 "[req] %s %s %s -> %d  prompt=%d gen=%d  %.0f ms (prefill %.0f [new %d, "
                 "cached %d], decode %.0f, %.1f tok/s)\n",
                 req.client.c_str(), req.method.c_str(), req.path.c_str(), status,
                 r->prompt_tokens, r->gen_tokens, wall_ms, r->prefill_ms,
                 r->prefilled_tokens, r->cache_hit_tokens, r->decode_ms,
                 r->decode_ms > 0 ? r->gen_tokens * 1000.0 / r->decode_ms : 0.0);
  } else {
    std::fprintf(stderr, "[req] %s %s %s -> %d  %.1f ms\n", req.client.c_str(),
                 req.method.c_str(), req.path.c_str(), status, wall_ms);
  }
  std::fflush(stderr);
}

void handle_chat_completions(ServerState& st, const http::Request& req, http::Conn& conn) {
  const auto t0 = Clock::now();
  const jsonp::Value body = parse_json_body(req);
  GenParams gp = params_from_body(st, body);
  const std::vector<tokenizer::ChatMessage> messages = parse_messages(body);

  bool thinking = true;
  const jsonp::Value* kwargs = body.find("chat_template_kwargs");
  if (kwargs && kwargs->is_object()) thinking = kwargs->get_bool("enable_thinking", true);
  const jsonp::Value* et = body.find("enable_thinking");
  if (et && et->type == jsonp::Value::Type::Bool) thinking = et->b;

  const std::string prompt = st.tk.apply_chat_template(messages, /*add_generation_prompt=*/true,
                                                       thinking);
  if (prompt.empty())
    throw UserError("this model's chat template is not supported by the engine; use "
                    "/v1/completions with a raw prompt instead");
  // Same messages rendered without the generation prompt: its tokens are the
  // prefix the next turn of this conversation will reuse.
  const std::string history_prompt =
      st.tk.apply_chat_template(messages, /*add_generation_prompt=*/false, thinking);
  gp.thinking_active = st.thinking_aware && thinking;
  gp.prefix_cache = bool_or(body, "cache_prompt", true);

  const bool stream = bool_or(body, "stream", false);
  const std::string model_name = body.get_str("model", st.model_name);
  const std::string id = "chatcmpl-" + random_hex(12);
  const int64_t created = now_unix();
  validate_prompt(st, prompt, gp);  // 400 before any response bytes
  st.busy.store(true);
  GenOutcome result;
  int status = 200;

  try {
    if (stream) {
      if (!conn.begin_stream("text/event-stream; charset=utf-8"))
        throw UserError("client disconnected");
      {  // OpenAI clients expect an initial role delta
        json::Writer w;
        w.begin_obj();
        w.key("id");
        w.str(id);
        w.key("object");
        w.str("chat.completion.chunk");
        w.key("created");
        w.integer(created);
        w.key("model");
        w.str(model_name);
        w.key("choices");
        w.begin_arr();
        w.begin_obj();
        w.key("index");
        w.integer(0);
        w.key("delta");
        w.begin_obj();
        w.key("role");
        w.str("assistant");
        w.key("content");
        w.str("");
        w.end_obj();
        w.key("finish_reason");
        w.null();
        w.end_obj();
        w.end_arr();
        w.end_obj();
        sse_data(conn, w.out);
      }
      bool client_gone = false;
      // The llama.cpp webui only renders delta.content and parses  thinking...
      // <｜end▁of▁thinking｜> tags client-side, so thinking text is ALSO streamed through
      // content (wrapped in the tags); OpenAI-style clients keep using the
      // separate reasoning_content field.
      bool think_tag_open = false;
      auto send_delta = [&](const std::string& text, bool is_reasoning) -> bool {
        if (text.empty()) return true;
        std::string content_delta;
        std::string reasoning_delta;
        if (is_reasoning) {
          if (!think_tag_open) {
            content_delta += "<think>";
            think_tag_open = true;
          }
          content_delta += text;
          reasoning_delta = text;
        } else {
          if (think_tag_open) {
            content_delta += "</think>";
            think_tag_open = false;
          }
          content_delta += text;
        }
        json::Writer w;
        w.begin_obj();
        w.key("id");
        w.str(id);
        w.key("object");
        w.str("chat.completion.chunk");
        w.key("created");
        w.integer(created);
        w.key("model");
        w.str(model_name);
        w.key("choices");
        w.begin_arr();
        w.begin_obj();
        w.key("index");
        w.integer(0);
        w.key("delta");
        w.begin_obj();
        w.key("content");
        w.str(content_delta);
        if (!reasoning_delta.empty()) {
          w.key("reasoning_content");
          w.str(reasoning_delta);
        }
        w.end_obj();
        w.key("finish_reason");
        w.null();
        w.end_obj();
        w.end_arr();
        w.end_obj();
        sse_data(conn, w.out);
        if (conn.failed()) client_gone = true;
        return !conn.failed();
      };
      result = generate(st, prompt, history_prompt, gp, send_delta);
      if (result.think_budget_hit && !conn.failed()) {
        // thinking hit its budget while still planning: ask for the answer with
        // thinking disabled so the user gets a clean answer (not planning text)
        const std::string prompt2 = st.tk.apply_chat_template(messages, true, false);
        GenParams gp2 = gp;
        gp2.thinking_active = false;
        gp2.max_tokens = std::max(64, gp.max_tokens - result.gen_tokens);
        GenOutcome r2 = generate(st, prompt2, history_prompt, gp2, send_delta);
        const int first_gen = result.gen_tokens;
        result.content = r2.content;
        result.gen_tokens = first_gen + r2.gen_tokens;
        result.token_ids.insert(result.token_ids.end(), r2.token_ids.begin(),
                                r2.token_ids.end());
        result.finish_reason = r2.finish_reason;
        result.prefill_ms += r2.prefill_ms;
        result.decode_ms += r2.decode_ms;
        result.prefilled_tokens = r2.prefilled_tokens;
        result.cache_hit_tokens = r2.cache_hit_tokens;
        result.repeat_guard_hit = result.repeat_guard_hit || r2.repeat_guard_hit;
        result.think_budget_hit = result.think_budget_hit || r2.think_budget_hit;
      }
      if (think_tag_open && !conn.failed()) {
        // thinking ran into the token limit without a final answer: close the
        // think block so the webui does not render an endless "thinking" state
        json::Writer w;
        w.begin_obj();
        w.key("id");
        w.str(id);
        w.key("object");
        w.str("chat.completion.chunk");
        w.key("created");
        w.integer(created);
        w.key("model");
        w.str(model_name);
        w.key("choices");
        w.begin_arr();
        w.begin_obj();
        w.key("index");
        w.integer(0);
        w.key("delta");
        w.begin_obj();
        w.key("content");
        w.str("</think>");
        w.end_obj();
        w.key("finish_reason");
        w.null();
        w.end_obj();
        w.end_arr();
        w.end_obj();
        sse_data(conn, w.out);
      }
      if (!conn.failed()) {
        // role delta first? clients accept a content-only stream; send the
        // final chunk with the finish reason and usage.
        json::Writer w;
        w.begin_obj();
        w.key("id");
        w.str(id);
        w.key("object");
        w.str("chat.completion.chunk");
        w.key("created");
        w.integer(created);
        w.key("model");
        w.str(model_name);
        w.key("choices");
        w.begin_arr();
        w.begin_obj();
        w.key("index");
        w.integer(0);
        w.key("delta");
        w.begin_obj();
        w.end_obj();
        w.key("finish_reason");
        w.str(result.finish_reason);
        w.end_obj();
        w.end_arr();
        write_stream_usage(w, result);
        w.end_obj();
        sse_data(conn, w.out);
        sse_data(conn, "[DONE]");
      }
      conn.end_stream();
      if (client_gone) status = 0;  // aborted; do not log as success
    } else {
      result = generate(st, prompt, history_prompt, gp,
                        [](const std::string&, bool) { return true; });
      if (result.think_budget_hit) {
        const std::string prompt2 = st.tk.apply_chat_template(messages, true, false);
        GenParams gp2 = gp;
        gp2.thinking_active = false;
        gp2.max_tokens = std::max(64, gp.max_tokens - result.gen_tokens);
        GenOutcome r2 = generate(st, prompt2, history_prompt, gp2,
                                 [](const std::string&, bool) { return true; });
        const int first_gen = result.gen_tokens;
        result.content = r2.content;
        result.gen_tokens = first_gen + r2.gen_tokens;
        result.token_ids.insert(result.token_ids.end(), r2.token_ids.begin(),
                                r2.token_ids.end());
        result.finish_reason = r2.finish_reason;
        result.prefill_ms += r2.prefill_ms;
        result.decode_ms += r2.decode_ms;
        result.prefilled_tokens = r2.prefilled_tokens;
        result.cache_hit_tokens = r2.cache_hit_tokens;
        result.repeat_guard_hit = result.repeat_guard_hit || r2.repeat_guard_hit;
        result.think_budget_hit = result.think_budget_hit || r2.think_budget_hit;
      }
      json::Writer w;
      w.begin_obj();
      w.key("id");
      w.str(id);
      w.key("object");
      w.str("chat.completion");
      w.key("created");
      w.integer(created);
      w.key("model");
      w.str(model_name);
      w.key("choices");
      w.begin_arr();
      w.begin_obj();
      w.key("index");
      w.integer(0);
      w.key("message");
      w.begin_obj();
      w.key("role");
      w.str("assistant");
      w.key("content");
      // webui convention: thinking is part of content, wrapped in  thinking tags;
      // reasoning_content is kept for OpenAI-style clients.
      if (result.reasoning.empty()) {
        w.str(result.content);
      } else {
        w.str("<think>" + result.reasoning + "</think>" + result.content);
      }
      if (!result.reasoning.empty()) {
        w.key("reasoning_content");
        w.str(result.reasoning);
      }
      w.end_obj();
      w.key("finish_reason");
      w.str(result.finish_reason);
      w.end_obj();
      w.end_arr();
      write_usage(w, result);
      write_timings(w, result);
      w.end_obj();
      conn.send_json(200, w.out);
    }
  } catch (const UserError& e) {
    status = 400;
    if (conn.stream_open()) conn.end_stream();
    else if (!conn.failed()) conn.send_error(400, e.what());
  } catch (const std::exception& e) {
    status = 500;
    if (conn.stream_open()) conn.end_stream();
    else if (!conn.failed()) conn.send_error(500, e.what());
  }
  st.busy.store(false);
  log_request(st, req, status, status == 200 ? &result : nullptr, ms_since(t0));
}

void handle_completions(ServerState& st, const http::Request& req, http::Conn& conn) {
  const auto t0 = Clock::now();
  const jsonp::Value body = parse_json_body(req);
  const jsonp::Value* prompt_v = body.find("prompt");
  if (!prompt_v) throw UserError("'prompt' is required");
  if (!prompt_v->is_string())
    throw UserError("'prompt' must be a string (prompt arrays are not supported)");
  GenParams gp = params_from_body(st, body);
  const bool stream = bool_or(body, "stream", false);
  const std::string model_name = body.get_str("model", st.model_name);
  const std::string id = "cmpl-" + random_hex(12);
  const int64_t created = now_unix();
  validate_prompt(st, prompt_v->str, gp);
  st.busy.store(true);
  GenOutcome result;
  int status = 200;
  try {
    if (stream) {
      conn.begin_stream("text/event-stream; charset=utf-8");
      auto send_delta = [&](const std::string& text, bool) -> bool {
        if (!text.empty()) {
          json::Writer w;
          w.begin_obj();
          w.key("id");
          w.str(id);
          w.key("object");
          w.str("text_completion");
          w.key("created");
          w.integer(created);
          w.key("model");
          w.str(model_name);
          w.key("choices");
          w.begin_arr();
          w.begin_obj();
          w.key("index");
          w.integer(0);
          w.key("text");
          w.str(text);
          w.key("finish_reason");
          w.null();
          w.end_obj();
          w.end_arr();
          w.end_obj();
          sse_data(conn, w.out);
        }
        return !conn.failed();
      };
      result = generate(st, prompt_v->str, std::string(), gp, send_delta);
      if (!conn.failed()) {
        json::Writer w;
        w.begin_obj();
        w.key("id");
        w.str(id);
        w.key("object");
        w.str("text_completion");
        w.key("created");
        w.integer(created);
        w.key("model");
        w.str(model_name);
        w.key("choices");
        w.begin_arr();
        w.begin_obj();
        w.key("index");
        w.integer(0);
        w.key("text");
        w.str("");
        w.key("finish_reason");
        w.str(result.finish_reason);
        w.end_obj();
        w.end_arr();
        write_stream_usage(w, result);
        w.end_obj();
        sse_data(conn, w.out);
        sse_data(conn, "[DONE]");
      }
      conn.end_stream();
    } else {
      result = generate(st, prompt_v->str, std::string(), gp,
                        [](const std::string&, bool) { return true; });
      json::Writer w;
      w.begin_obj();
      w.key("id");
      w.str(id);
      w.key("object");
      w.str("text_completion");
      w.key("created");
      w.integer(created);
      w.key("model");
      w.str(model_name);
      w.key("choices");
      w.begin_arr();
      w.begin_obj();
      w.key("index");
      w.integer(0);
      w.key("text");
      w.str(result.reasoning.empty()
                ? result.raw_text
                : "<think>" + result.reasoning + "</think>" + result.content);
      w.key("finish_reason");
      w.str(result.finish_reason);
      w.end_obj();
      w.end_arr();
      write_usage(w, result);
      write_timings(w, result);
      w.end_obj();
      conn.send_json(200, w.out);
    }
  } catch (const UserError& e) {
    status = 400;
    if (conn.stream_open()) conn.end_stream();
    else if (!conn.failed()) conn.send_error(400, e.what());
  } catch (const std::exception& e) {
    status = 500;
    if (conn.stream_open()) conn.end_stream();
    else if (!conn.failed()) conn.send_error(500, e.what());
  }
  st.busy.store(false);
  log_request(st, req, status, status == 200 ? &result : nullptr, ms_since(t0));
}

// llama.cpp-style /completion
void handle_llamacpp_completion(ServerState& st, const http::Request& req, http::Conn& conn) {
  const auto t0 = Clock::now();
  const jsonp::Value body = parse_json_body(req);
  const jsonp::Value* prompt_v = body.find("prompt");
  if (!prompt_v || !prompt_v->is_string()) throw UserError("'prompt' must be a string");
  GenParams gp = params_from_body(st, body);
  const bool stream = bool_or(body, "stream", false);
  validate_prompt(st, prompt_v->str, gp);
  st.busy.store(true);
  GenOutcome result;
  int status = 200;
  try {
    if (stream) {
      conn.begin_stream("text/event-stream; charset=utf-8");
      auto send_delta = [&](const std::string& text, bool) -> bool {
        json::Writer w;
        w.begin_obj();
        w.key("content");
        w.str(text);
        w.key("stop");
        w.boolean(false);
        w.end_obj();
        sse_data(conn, w.out);
        return !conn.failed();
      };
      result = generate(st, prompt_v->str, std::string(), gp, send_delta);
      if (!conn.failed()) {
        json::Writer w;
        w.begin_obj();
        w.key("content");
        w.str("");
        w.key("stop");
        w.boolean(true);
        w.key("stop_reason");
        w.str(result.finish_reason);
        w.key("tokens_predicted");
        w.integer(result.gen_tokens);
        w.key("tokens_evaluated");
        w.integer(result.prompt_tokens);
        w.end_obj();
        sse_data(conn, w.out);
      }
      conn.end_stream();
    } else {
      result = generate(st, prompt_v->str, std::string(), gp,
                        [](const std::string&, bool) { return true; });
      json::Writer w;
      w.begin_obj();
      w.key("content");
      w.str(result.reasoning.empty() ? result.raw_text
                                     : "<think>" + result.reasoning + "</think>" + result.content);
      w.key("stop");
      w.boolean(true);
      w.key("stop_reason");
      w.str(result.finish_reason);
      w.key("model");
      w.str(st.model_name);
      w.key("tokens_predicted");
      w.integer(result.gen_tokens);
      w.key("tokens_evaluated");
      w.integer(result.prompt_tokens);
      write_generation_settings(w, gp);
      write_timings(w, result);
      w.end_obj();
      conn.send_json(200, w.out);
    }
  } catch (const UserError& e) {
    status = 400;
    if (conn.stream_open()) conn.end_stream();
    else if (!conn.failed()) conn.send_error(400, e.what());
  } catch (const std::exception& e) {
    status = 500;
    if (conn.stream_open()) conn.end_stream();
    else if (!conn.failed()) conn.send_error(500, e.what());
  }
  st.busy.store(false);
  log_request(st, req, status, status == 200 ? &result : nullptr, ms_since(t0));
}

void handle_tokenize(const ServerState& st, const http::Request& req, http::Conn& conn) {
  const jsonp::Value body = parse_json_body(req);
  std::string content;
  const jsonp::Value* c = body.find("content");
  if (c && c->is_string()) content = c->str;
  else if (body.is_string()) content = body.str;
  else throw UserError("'content' (string) is required");
  const bool add_special = body.get_bool("add_special", false);
  const std::vector<int> ids = st.tk.encode(content, add_special, true);
  json::Writer w;
  w.begin_obj();
  w.key("tokens");
  w.begin_arr();
  for (int id : ids) w.integer(id);
  w.end_arr();
  w.key("pieces");
  w.begin_arr();
  for (int id : ids) w.str(st.tk.token_text(id));
  w.end_arr();
  w.end_obj();
  conn.send_json(200, w.out);
}

void handle_detokenize(const ServerState& st, const http::Request& req, http::Conn& conn) {
  const jsonp::Value body = parse_json_body(req);
  const jsonp::Value* t = body.find("tokens");
  if (!t || !t->is_array()) throw UserError("'tokens' (array of ids) is required");
  std::vector<int> ids;
  ids.reserve(t->arr.size());
  for (const auto& v : t->arr) {
    if (!v.is_number()) throw UserError("'tokens' must contain numbers");
    ids.push_back(static_cast<int>(v.num));
  }
  json::Writer w;
  w.begin_obj();
  w.key("content");
  w.str(st.tk.decode(ids, /*skip_special=*/true));
  w.end_obj();
  conn.send_json(200, w.out);
}

const char* kIndexHtml = R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>engine · 本地对话</title>
<style>
*{box-sizing:border-box}
:root{color-scheme:dark;--bg:#0a0d12;--bg2:#0e1218;--panel:#121822;--panel2:#171f2b;--line:#232c3b;--fg:#e7ebf3;--dim:#8b95a8;--dim2:#6b7688;--acc:#5b8cff;--acc2:#8f7bff;--ok:#3ddc97;--err:#ff6b6b;--radius:14px}
html,body{height:100%}
body{margin:0;background:radial-gradient(1100px 560px at 72% -12%,#16213a 0,transparent 62%),var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;display:flex;flex-direction:column;height:100vh;height:100dvh;overflow:hidden}
header{display:flex;align-items:center;gap:8px;padding:10px 16px;border-bottom:1px solid var(--line);background:rgba(14,18,24,.86);backdrop-filter:blur(8px);flex-wrap:wrap;z-index:2}
.logo{font-weight:700;letter-spacing:.3px;display:flex;align-items:center;gap:7px;font-size:15px;white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 10px var(--ok)}
.chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center;min-width:0;flex:1}
.chip{font-size:12px;color:var(--dim);border:1px solid var(--line);background:var(--bg2);padding:2px 8px;border-radius:999px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:46vw}
.chip.model{max-width:38vw;color:#c5d2ea}
.chip.tps{color:var(--ok);border-color:#1d4a38}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--panel2);color:var(--fg);border-radius:10px;padding:8px 14px;transition:filter .15s,background .15s}
button:hover{filter:brightness(1.16)}
button:disabled{opacity:.45;cursor:default;filter:none}
#newBtn{padding:6px 12px;font-size:13px;background:transparent}
main{flex:1;overflow-y:auto;overflow-x:hidden;padding:22px 16px 8px}
.wrap{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
.empty{margin:13vh auto 0;text-align:center;color:var(--dim);max-width:520px;padding:0 10px}
.empty h1{font-size:22px;color:var(--fg);margin:0 0 8px;font-weight:650}
.empty .logo{justify-content:center;font-size:20px;margin-bottom:14px}
.empty p{margin:6px 0;font-size:14px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:0 5px;font:12px ui-monospace,Consolas,monospace;color:var(--fg);background:var(--panel2)}
.msg{display:flex;gap:10px;max-width:100%}
.msg.user{justify-content:flex-end}
.avatar{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:flex;align-items:center;justify-content:center;font-size:12px;color:#fff;flex:none;margin-top:2px}
.bubble{max-width:82%;padding:10px 14px;border-radius:var(--radius);background:var(--panel);border:1px solid var(--line);overflow-wrap:anywhere;white-space:pre-wrap}
.msg.user .bubble{background:linear-gradient(135deg,#1e3a66,#1a2c4f);border-color:#2b4775;border-bottom-right-radius:4px}
.msg.assistant .bubble{border-bottom-left-radius:4px;min-width:80px}
.bubble.err{border-color:#5a2b2b;background:#221518;color:#ffb3b3}
.think{margin:0 0 8px;border-left:2px solid #3d3560;padding-left:9px}
.think[hidden]{display:none}
.think-hd{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--dim);background:none;border:0;padding:2px 0;width:100%;text-align:left;border-radius:0}
.think-hd:hover{color:var(--fg);filter:none}
.arrow{display:inline-block;font-size:10px;transition:transform .15s}
.think.open .arrow{transform:rotate(90deg)}
.think-bd{margin-top:6px;padding:2px 0 6px;color:var(--dim);font-size:13.5px;line-height:1.6;max-height:44vh;overflow:auto;white-space:pre-wrap}
.think-bd[hidden]{display:none}
.stats{font-size:11px;color:var(--dim2);margin-top:7px;white-space:normal}
.stats span+span:before{content:"·";margin:0 6px;opacity:.6}
footer{border-top:1px solid var(--line);background:rgba(12,15,20,.92);backdrop-filter:blur(10px)}
.fwrap{max-width:860px;margin:0 auto;padding:8px 16px 12px}
.cfg{font-size:12px;color:var(--dim)}
.cfg summary{cursor:pointer;list-style:none;display:inline-flex;align-items:center;gap:6px;padding:3px 0;user-select:none}
.cfg summary::-webkit-details-marker{display:none}
.cfg summary:before{content:"⚙";font-size:12px}
.cfg[open] summary{color:var(--fg)}
.cfgrow{display:flex;gap:14px;flex-wrap:wrap;align-items:center;padding:8px 0 10px}
.cfgrow label{display:flex;align-items:center;gap:6px;white-space:nowrap}
.cfgrow input[type=number],.cfgrow input[type=text]{width:92px;background:var(--bg2);border:1px solid var(--line);color:var(--fg);border-radius:8px;padding:5px 8px;font:inherit;font-size:13px;outline:none}
.cfgrow input:focus{border-color:#35558f}
.cfgrow input[type=checkbox]{accent-color:var(--acc);width:15px;height:15px}
.note{color:var(--dim2);font-size:11px}
.composer{display:flex;gap:9px;align-items:flex-end}
#input{flex:1;resize:none;min-height:45px;max-height:190px;background:var(--panel);border:1px solid var(--line);border-radius:12px;color:var(--fg);padding:11px 14px;font:inherit;line-height:1.45;outline:none;transition:border-color .15s}
#input:focus{border-color:#35558f}
#send{min-height:45px;background:linear-gradient(135deg,var(--acc),#4f6fff);border-color:transparent;font-weight:600;padding:0 20px}
#stop{min-height:45px;background:#3a1f24;border-color:#5a2b32;color:#ff9a9a;display:none}
#stop.show{display:block}
.hint{display:flex;justify-content:space-between;gap:10px;font-size:11px;color:var(--dim2);padding-top:7px;min-height:16px}
#status.err{color:var(--err)}
@media(max-width:640px){
header{padding:8px 12px;gap:6px}
.hide-sm{display:none}
.chip.model{max-width:50vw}
main{padding:14px 10px 4px}
.wrap{gap:12px}
.bubble{max-width:90%}
.fwrap{padding:6px 10px 10px}
#send{min-height:43px;padding:0 14px}
.empty{margin-top:9vh}
}
</style>
</head>
<body>
<header>
<div class="logo"><span class="dot"></span>engine</div>
<div class="chips">
<span class="chip model" id="cModel" title="">模型加载中…</span>
<span class="chip hide-sm" id="cCtx"></span>
<span class="chip hide-sm" id="cPort"></span>
<span class="chip tps" id="cTps" hidden title="上一个回复的生成速度">tok/s —</span>
</div>
<button id="newBtn" title="清空当前对话">＋ 新对话</button>
</header>
<main id="main">
<div class="wrap" id="wrap">
<div class="empty" id="empty">
<div class="logo"><span class="dot"></span>engine</div>
<h1>本地对话</h1>
<p>输入消息开始 · 推理全程在本机完成</p>
<p><kbd>Enter</kbd> 发送 &nbsp; <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行</p>
</div>
</div>
</main>
<footer>
<div class="fwrap">
<details class="cfg" id="cfg">
<summary>生成参数</summary>
<div class="cfgrow">
<label>temperature <input type="number" id="temp" min="0" max="2" step="0.05"></label>
<label>max_tokens <input type="text" id="maxTok" inputmode="numeric" placeholder="不限"></label>
<label><input type="checkbox" id="noThink"> 关闭思考</label>
<span class="note">max_tokens 留空 = 服务端不限（受 ctx 限制）</span>
</div>
</details>
<div class="composer">
<textarea id="input" rows="1" placeholder="输入消息…" autocomplete="off" spellcheck="false"></textarea>
<button id="send">发送</button>
<button id="stop">停止</button>
</div>
<div class="hint"><span id="status">就绪</span><span class="hide-sm">Enter 发送 · Shift+Enter 换行</span></div>
</div>
</footer>
<script>
(function(){
"use strict";
var $=function(s){return document.querySelector(s)};
var main=$('#main'),wrap=$('#wrap'),input=$('#input'),sendBtn=$('#send'),stopBtn=$('#stop'),
statusEl=$('#status'),cModel=$('#cModel'),cCtx=$('#cCtx'),cPort=$('#cPort'),cTps=$('#cTps'),
tempEl=$('#temp'),maxEl=$('#maxTok'),noThinkEl=$('#noThink'),emptyEl=$('#empty');
var messages=[],busy=false,ctrl=null,gen=0,props=null;
var CFG='engine-ui';
function saveCfg(){try{localStorage.setItem(CFG,JSON.stringify({t:tempEl.value,m:maxEl.value,n:noThinkEl.checked}))}catch(e){}}
function loadCfg(){try{var c=JSON.parse(localStorage.getItem(CFG)||'null');if(!c)return false;if(c.t!==undefined)tempEl.value=c.t;if(c.m!==undefined)maxEl.value=c.m;if(typeof c.n==='boolean')noThinkEl.checked=c.n;return true}catch(e){return false}}
function st(t,err){statusEl.textContent=t;statusEl.className=err?'err':''}
function nearBottom(){return main.scrollHeight-main.scrollTop-main.clientHeight<90}
function scroll(force){if(force||nearBottom())main.scrollTop=main.scrollHeight}
function el(tag,cls,text){var d=document.createElement(tag);if(cls)d.className=cls;if(text!==undefined)d.textContent=text;return d}
function newMsg(role,text){
emptyEl.hidden=true;
var row=el('div','msg '+role);
if(role==='assistant')row.appendChild(el('div','avatar','e'));
var bubble=el('div','bubble');
row.appendChild(bubble);
if(role==='user'){bubble.textContent=text}
else{
var think=el('div','think');think.hidden=true;
var hd=el('button','think-hd');hd.type='button';
hd.appendChild(el('span','arrow','▶'));
var label=el('span','think-label','思考');hd.appendChild(label);
var body=el('div','think-bd');body.hidden=true;
think.appendChild(hd);think.appendChild(body);
hd.addEventListener('click',function(){var open=!body.hidden;body.hidden=open;think.classList.toggle('open',!open)});
bubble.appendChild(think);
row.appendChild(bubble);
row._think=think;row._label=label;row._body=body;
row._content=el('div','content');bubble.appendChild(row._content);
row._stats=el('div','stats');bubble.appendChild(row._stats);
}
wrap.appendChild(row);scroll(true);
return row;
}
function setThink(r,text){r._think.hidden=!text;if(text){r._body.textContent=text;r._label.textContent='思考中 · '+text.length+' 字'}}
function setStats(r,parts){r._stats.innerHTML='';for(var i=0;i<parts.length;i++)r._stats.appendChild(el('span','',parts[i]))}
fetch('/props').then(function(r){return r.json()}).then(function(p){
props=p;
var name=p.model_name||'?';cModel.textContent=name;cModel.title=p.model_path||name;
cCtx.textContent='ctx '+p.n_ctx;cPort.textContent='端口 '+(location.port||p.port||'');
var g=p.default_generation_settings||{};
var aware=!!(p.tokenizer&&p.tokenizer.thinking_aware);
CFG='engine-ui:'+(name==='?'?'default':name);
if(!loadCfg()){if(typeof g.temperature==='number')tempEl.value=g.temperature}
noThinkEl.disabled=!aware;
if(!aware)noThinkEl.checked=true;
if(typeof g.temperature==='number')tempEl.placeholder=String(g.temperature);
if(g.n_predict&&g.n_predict<1000000)maxEl.placeholder=String(g.n_predict);
document.title=name+' · engine';
}).catch(function(){cModel.textContent='服务信息不可用'});
function bodyFor(){
var body={model:'engine',messages:messages,stream:true};
var t=parseFloat(tempEl.value);if(isFinite(t)&&t>=0)body.temperature=t;
var m=parseInt(maxEl.value,10);if(isFinite(m)&&m>0)body.max_tokens=m;
body.enable_thinking=!noThinkEl.checked;
return body;
}
async function send(){
var text=input.value.trim();
if(!text||busy)return;
saveCfg();
var my=++gen;
input.value='';autoGrow();
newMsg('user',text);messages.push({role:'user',content:text});
var row=newMsg('assistant');var content='',reason='',usage=null,timings=null;
var t0=performance.now();
busy=true;sendBtn.disabled=true;stopBtn.classList.add('show');ctrl=new AbortController();st('生成中…');
try{
var resp=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(bodyFor()),signal:ctrl.signal});
if(!resp.ok){var t=await resp.text();throw new Error('HTTP '+resp.status+' '+t.slice(0,300))}
var reader=resp.body.getReader(),dec=new TextDecoder(),buf='';
while(true){
var rd=await reader.read();if(rd.done)break;
if(my!==gen)break;
buf+=dec.decode(rd.value,{stream:true});
var idx;
while((idx=buf.indexOf('\n\n'))>=0){
var evt=buf.slice(0,idx);buf=buf.slice(idx+2);
var line=evt.split('\n').filter(function(x){return x.indexOf('data:')===0})[0];
if(!line)continue;
var data=line.slice(5).trim();
if(data==='[DONE]')continue;
var j;try{j=JSON.parse(data)}catch(e){continue}
if(j.usage)usage=j.usage;
if(j.timings)timings=j.timings;
var d=j.choices&&j.choices[0]&&j.choices[0].delta;
if(d&&d.reasoning_content){reason+=d.reasoning_content;setThink(row,reason);scroll()}
if(d&&d.content){content+=d.content;row._content.textContent=content;scroll()}
}
}
if(my!==gen)return;
var elapsed=(performance.now()-t0)/1000;
var ct=usage&&usage.completion_tokens?usage.completion_tokens:0;
var dec_s=timings&&timings.decode_ms?timings.decode_ms/1000:elapsed;
var tps=ct&&dec_s>0?ct/dec_s:0;
if(!content&&!reason)row._content.textContent='（模型没有输出内容）';
else if(!content&&reason)row._content.textContent='（只输出了思考内容，被截断）';
var parts=[];
if(ct)parts.push(ct+' tokens');
if(tps)parts.push(tps.toFixed(1)+' tok/s');
if(elapsed)parts.push(elapsed.toFixed(1)+' s');
if(reason)row._label.textContent='思考 · '+reason.length+' 字';
setStats(row,parts);
if(tps){cTps.hidden=false;cTps.textContent='tok/s '+tps.toFixed(1)}
messages.push({role:'assistant',content:content,reasoning_content:reason});
st('就绪');
}catch(err){
if(my!==gen)return;
if(err&&err.name==='AbortError'){st('已停止');if(!content&&!reason)row._content.textContent='（已停止）';row._label.textContent='思考 · 已停止'}
else{row._bubble.classList.add('err');row._content.textContent='请求失败：'+((err&&err.message)||err);setStats(row,[]);st('错误',true)}
}finally{
if(my===gen){busy=false;sendBtn.disabled=false;stopBtn.classList.remove('show');ctrl=null;scroll();}
}
}
function autoGrow(){input.style.height='auto';input.style.height=Math.min(input.scrollHeight,190)+'px'}
input.addEventListener('input',autoGrow);
input.addEventListener('keydown',function(e){
if(e.key==='Enter'&&!e.shiftKey&&!(e.isComposing||e.keyCode===229)){e.preventDefault();send()}
});
sendBtn.addEventListener('click',send);
stopBtn.addEventListener('click',function(){if(ctrl){ctrl.abort();st('正在停止…')}});
$('#newBtn').addEventListener('click',function(){
gen++;if(ctrl){ctrl.abort();ctrl=null}
busy=false;sendBtn.disabled=false;stopBtn.classList.remove('show');
messages=[];wrap.innerHTML='';wrap.appendChild(emptyEl);emptyEl.hidden=false;
st('新对话');if(!matchMedia('(pointer:coarse)').matches)input.focus();
});
[tempEl,maxEl,noThinkEl].forEach(function(e){e.addEventListener('change',saveCfg)});
autoGrow();
if(!matchMedia('(pointer:coarse)').matches)input.focus();
})();
</script>
</body></html>
)HTML";

// Graceful shutdown over HTTP: only exposed when the server is bound to a
// loopback address, so a remote client can never stop the engine.
void handle_shutdown(http::Conn& conn) {
  json::Writer w;
  w.begin_obj();
  w.key("status");
  w.str("shutting down");
  w.end_obj();
  conn.send_json(200, w.out);
  if (g_server) g_server->stop();
}

// Directory containing engine.exe (the cwd is not reliable: a launcher may be
// started from anywhere).
std::string module_dir() {
#ifdef _WIN32
  wchar_t buf[4096];
  const DWORD n = GetModuleFileNameW(nullptr, buf, 4096);
  if (n == 0 || n >= 4096) return std::string();
  return std::filesystem::path(buf).parent_path().string();
#else
  std::error_code ec;
  const std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) return std::string();
  return p.parent_path().string();
#endif
}

std::string read_file_bytes(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return std::string();
  std::string out;
  char buf[1 << 16];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  std::fclose(f);
  return out;
}

// llama.cpp's official single-file webui (MIT) is shipped next to the exe as
// webui/index.html: a static file instead of a multi-megabyte string in the
// binary.  If it is not there, the small page compiled into engine.exe is used,
// so `engine serve` never serves a blank tab.  ENGINE_WEBUI_FILE overrides the
// path (tests / custom builds).
const std::string& webui_html() {
  static std::string html;
  static bool loaded = false;
  if (loaded) return html;
  loaded = true;
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("ENGINE_WEBUI_FILE"); env != nullptr && *env != '\0')
    candidates.push_back(env);
  const std::string dir = module_dir();
  if (!dir.empty()) {
    candidates.push_back(dir + "/webui/index.html");
    candidates.push_back(dir + "/../webui/index.html");
  }
  candidates.push_back("webui/index.html");
  for (const std::string& path : candidates) {
    html = read_file_bytes(path);
    if (!html.empty()) {
      std::printf("[serve] webui      : %s (%zu bytes, llama.cpp webui)\n", path.c_str(),
                  html.size());
      std::fflush(stdout);
      return html;
    }
  }
  std::printf("[serve] webui      : built-in minimal page (webui/index.html not found)\n");
  std::fflush(stdout);
  return html;
}

void handle_index(http::Conn& conn) {
  const std::string& webui = webui_html();
  // no-store: a rebuilt exe/webui must not be masked by a heuristic HTTP cache.
  conn.send(200, "text/html; charset=utf-8", webui.empty() ? std::string(kIndexHtml) : webui,
            {{"Cache-Control", "no-store, max-age=0"}});
}

// ---------------------------------------------------------------------------
void route(ServerState& st, const http::Request& req, http::Conn& conn) {
  const bool post = req.method == "POST";
  const bool get = req.method == "GET" || req.method == "HEAD";
  const std::string& path = req.path;

  if (req.method == "OPTIONS") {
    conn.send(204, "text/plain", "",
              {{"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
               {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
               {"Access-Control-Max-Age", "86400"}});
    return;
  }
  if (get && (path == "/health" || path == "/healthz")) return handle_health(st, conn);
  if (get && path == "/v1/models") return handle_models(st, conn);
  if (get && path == "/props") return handle_props(st, conn);
  if (get && path == "/slots") {  // llama.cpp compatibility stub (no slot UI here)
    conn.send_json(200, "[]");
    return;
  }
  if (get && (path == "/" || path == "/index.html")) return handle_index(conn);
  if (post && (path == "/v1/chat/completions" || path == "/chat/completions"))
    return handle_chat_completions(st, req, conn);
  if (post && (path == "/v1/completions" || path == "/completions"))
    return handle_completions(st, req, conn);
  if (post && (path == "/completion" || path == "/v1/completion"))
    return handle_llamacpp_completion(st, req, conn);
  if (post && path == "/tokenize") return handle_tokenize(st, req, conn);
  if (post && path == "/detokenize") return handle_detokenize(st, req, conn);
  if (post && (path == "/v1/tokenize")) return handle_tokenize(st, req, conn);
  const bool loopback = st.opts.host.empty() || st.opts.host == "127.0.0.1" ||
                        st.opts.host == "localhost" || st.opts.host == "::1";
  if (post && loopback && (path == "/shutdown" || path == "/v1/shutdown"))
    return handle_shutdown(conn);
  conn.send_error(404, "no such endpoint: " + req.method + " " + path);
}

std::string base_name(const std::string& path) {
  try {
    std::filesystem::path p(path);
    std::string name = p.filename().string();
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos && dot > 0) name = name.substr(0, dot);
    return name.empty() ? path : name;
  } catch (...) {
    return path;
  }
}

runner::Container detect_container(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) return runner::Container::Safetensors;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f) {
    char magic[4] = {0, 0, 0, 0};
    const size_t got = std::fread(magic, 1, 4, f);
    std::fclose(f);
    if (got == 4 && std::memcmp(magic, "GGUF", 4) == 0) return runner::Container::Gguf;
    return runner::Container::Safetensors;
  }
  return path.find(".gguf") != std::string::npos ? runner::Container::Gguf
                                                 : runner::Container::Safetensors;
}

}  // namespace

int run(const Options& opts) {
  ServerState st;
  st.opts = opts;
  st.ctx = std::max(32, opts.ctx);
  st.model_name = base_name(opts.model);

  std::printf("[serve] model      : %s\n", opts.model.c_str());
  std::printf("[serve] tokenizer  : loading GGUF metadata ...\n");
  std::fflush(stdout);
  const auto t_tok0 = Clock::now();
  try {
      st.tk = tokenizer::Tokenizer::load(opts.model);
    } catch (const std::exception& e) {
    std::fprintf(stderr, "[serve] ERROR: cannot load tokenizer: %s\n", e.what());
    return 1;
  }
  const bool has_think_marker = st.tk.chat_template().find("enable_thinking") != std::string::npos &&
                                st.tk.chat_template().find("<think>") != std::string::npos;
  st.thinking_aware = has_think_marker;
  // Serve default: finer prefill chunks (128 tokens).  The prefix cache aligns its
  // snapshot/split points to the prefill chunk size; 128 keeps the cache granularity
  // fine (a turn only re-prefills the new tokens + <128 history tokens) while both
  // the cached and the full-prefill path still use identical absolute chunk
  // boundaries (bit-exact).  Explicit --prefill-rows / ENGINE_PREFILL_ROWS wins.
  // Serve default prefill chunk: 512 tokens (was 128).  128 keeps the prefix-cache
  // snapshot granularity fine, but it costs every prefill a full-model dequant pass and
  // a fixed per-chunk overhead: measured on the 9B (exclusive GPU, prefix cache off),
  // a 2.4-3.6k prompt runs 592 tok/s at 128 vs 1023-1103 tok/s at 512, and an 11-13k
  // prompt 454 vs 750 tok/s.  A 512-token boundary means a multi-turn request may
  // re-prefill up to 511 extra history tokens (~0.5 s at 1k tok/s); the exact value can
  // still be overridden with --prefill-rows.  1024 is the growth cap so long documents
  // can use the largest chunk the VRAM budget allows (the loader re-derives rows from
  // the actual free VRAM before allocating activations).
  if (!engine::config::was_set("prefill_rows")) {
    std::string perr;
    engine::config::Config::get().set("prefill_rows", "512", engine::config::Source::Cli, &perr);
    if (!engine::config::was_set("prefill_rows_max"))
      engine::config::Config::get().set("prefill_rows_max", "1024",
                                        engine::config::Source::Cli, &perr);
  }
  st.cache_align = static_cast<int>(engine::config::get_int("serve_cache_align"));
  if (st.cache_align < 0) st.cache_align = 0;
  st.pc.enabled = engine::config::get_bool("serve_prefix_cache");
  st.no_limit_tokens = static_cast<int>(engine::config::get_int("serve_no_limit_tokens"));
  if (st.no_limit_tokens <= 0) st.no_limit_tokens = 1024;
  st.think_budget = static_cast<int>(engine::config::get_int("serve_think_budget"));
  if (st.think_budget < 0) st.think_budget = 0;
  st.repeat_guard = static_cast<int>(engine::config::get_int("serve_repeat_guard"));
  if (st.repeat_guard < 0) st.repeat_guard = 0;
  if (st.repeat_guard > 2) st.repeat_guard = 2;
  st.mtp_k = static_cast<int>(engine::config::get_int("mtp_k"));
  if (st.mtp_k < 1) st.mtp_k = 1;
  if (st.mtp_k > 3) st.mtp_k = 3;
  // NOTE: st.mtp_enabled must be initialised AFTER the model is constructed (it
  // queries st.model->mtp_available()); doing it here used to null-deref whenever
  // --mtp was enabled.  See the block after the "model ready" banner.
  // Qwen3-family recommended sampling defaults, used when the client does not send
  // its own values (the llama.cpp webui bundle ships the same defaults).
  if (!engine::config::was_set("default_temperature")) st.opts.default_temperature = 0.6;
  if (!engine::config::was_set("default_top_k")) st.opts.default_top_k = 20;
  if (!engine::config::was_set("default_top_p")) st.opts.default_top_p = 0.95;
  std::printf("[serve] tokenizer  : %s pre=%s vocab=%d bos=%d eos=%d (%.0f ms)\n",
              st.tk.kind() == tokenizer::Kind::Bpe ? "gpt2/BPE" : "llama/SPM",
              st.tk.pre().empty() ? "-" : st.tk.pre().c_str(), st.tk.vocab_size(),
              st.tk.bos_id(), st.tk.eos_id(), ms_since(t_tok0));
  std::printf("[serve] chat tmpl  : %s%s\n", st.tk.template_name(),
              has_think_marker ? " (thinking-aware)" : "");
  std::printf("[serve] loading model (ctx=%s) ...\n",
              opts.ctx > 0 ? std::to_string(opts.ctx).c_str() : "auto");
  std::fflush(stdout);

  const auto t_load0 = Clock::now();
  try {
    st.model = std::make_unique<runner::DenseRunner>(opts.model, st.ctx,
                                                     detect_container(opts.model));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[serve] ERROR: model load failed: %s\n", e.what());
    std::fprintf(stderr, "[serve] hint: try a smaller --ctx (e.g. --ctx 2048)\n");
    return 1;
  }
  {
    // The loader may clamp the requested ctx (ctx_auto, kv_type budget) or reduce it
    // while fitting the prefill chunk.  Always take the REAL ctx from the model: the
    // prompt-length checks must reject an over-long prompt with 400 instead of letting
    // forward() throw "bad token count" (observed as HTTP 500 for an 11k prompt when
    // ctx 32768 had been clamped to 8053).
    const int real_ctx = st.model->ctx();
    if (real_ctx != st.ctx)
      std::printf("[serve] ctx clamp  : requested %d -> %d tokens (VRAM budget)\n", st.ctx,
                  real_ctx);
    st.ctx = real_ctx;
    if (opts.ctx <= 0)
      std::printf("[serve] auto ctx   : %d tokens (derived from free VRAM)\n", st.ctx);
  }
  const double load_ms = ms_since(t_load0);
  std::printf("[serve] model ready: vocab=%d ctx=%d load=%.1f s\n", st.model->vocab(), st.ctx,
              load_ms / 1000.0);
  st.mtp_enabled = engine::config::get_bool("mtp") && st.model->mtp_available();
  if (engine::config::get_bool("mtp"))
    std::printf("[serve] mtp        : %s (k=%d)\n",
                st.mtp_enabled ? "speculative decoding ON for greedy requests"
                               : "requested but not available on this model",
                st.mtp_k);
  std::printf("[serve] prefix-cache: %s (state snapshot: %s, %.1f MiB/slot, prefill chunk %d, "
              "cache align %s, no-limit cap %d tok)\n",
              st.pc.enabled ? "on" : "off",
              st.model->prefix_state_supported() ? "hybrid SSM+KV" : "KV only",
              static_cast<double>(st.model->prefix_state_bytes()) / 1048576.0,
              st.model->prefill_chunk(),
              st.cache_align > 0 ? std::to_string(st.cache_align).c_str() : "chunk",
              st.no_limit_tokens);
  std::printf("[serve] sampling: temp=%.2f top_k=%d top_p=%.2f | sampler: bounded candidates "
              "(cap %d; top_k=0 -> top-%d approx) | think budget: %s%d tokens "
              "| repeat guard: level %d\n",
              st.opts.default_temperature, st.opts.default_top_k, st.opts.default_top_p,
              sampler::Sampler::kDefaultCandidateCap, sampler::Sampler::kDefaultCandidateCap,
              st.think_budget > 0 ? "force </think> after " : "off",
              st.think_budget, st.repeat_guard);
  if (st.tk.vocab_size() != st.model->vocab())
    std::fprintf(stderr,
                 "[serve] WARN: tokenizer vocab (%d) != model vocab (%d); sampling uses the "
                 "model vocabulary\n",
                 st.tk.vocab_size(), st.model->vocab());
  std::fflush(stdout);

  // ---- post-load VRAM guard ------------------------------------------------
  // The loader degrades (halves the prefill chunk) when free VRAM drops below the
  // margin.  In the worst case the driver starts paging and decode collapses from
  // ~25 tok/s to 3-4 tok/s, which is much worse than failing fast: refuse to start
  // unless the operator explicitly allows oversubscription.
  if (!engine::config::get_bool("allow_oversubscribe")) {
    const size_t margin_bytes =
        static_cast<size_t>(engine::config::get_int("vram_margin_mb")) << 20;
    const cuda::MemInfo mem = cuda::mem_info();
    if (mem.free_bytes < margin_bytes) {
      std::fprintf(stderr,
                   "[serve] ERROR: only %.0f MiB VRAM free after load, below the %zu MiB "
                   "margin: the CUDA driver will page and decode can collapse to a few "
                   "tok/s. Lower --ctx / --gpu-layers, free VRAM, or pass "
                   "--allow-oversubscribe to start anyway.\n",
                   static_cast<double>(mem.free_bytes) / 1048576.0, margin_bytes >> 20);
      return 1;
    }
  }

  // ---- post-load self-warmup -----------------------------------------------
  // The first user request otherwise pays the lazy CUDA-graph captures (canonical
  // prefill chunk + decode graph) plus the first-touch driver/page-in cost: measured
  // as +4..23 s on request #1 (and every *new* tail token-count n used to pay its
  // own prefill-graph capture - that part needs the runner-side bucket fix).
  // Warm the canonical chunk and the decode graph once here, then reset the KV/SSM
  // state so request #1 starts from a clean cache.  ENGINE_NO_PREWARM=1 disables.
  if (std::getenv("ENGINE_NO_PREWARM") == nullptr) {
    const int chunk = std::max(1, st.model->prefill_chunk());
    std::string warm_text;
    for (int i = 0; i < 48; ++i) warm_text += "你好，世界。";
    const std::vector<int> warm_ids = st.tk.encode(warm_text, false, true);
    // Best-effort only: a degraded placement (e.g. --vram-margin huge with
    // --allow-oversubscribe) can shrink rows_/ctx to the point where the warmup
    // does not fit; never let the warmup prevent the server from starting.
    if (chunk < 16 || st.ctx < chunk + 16 ||
        static_cast<int>(warm_ids.size()) < chunk + 8) {
      std::printf("[serve] prewarm   : skipped (chunk=%d ctx=%d warm_tokens=%zu)\n", chunk,
                  st.ctx, warm_ids.size());
    } else {
      // Save the clean (position 0) recurrent state first: reset_kv_and_states() is
      // private to the runner, but the public prefix-state API covers exactly the
      // part warmup would corrupt (SSM conv/rec).  KV rows written by the warmup are
      // simply overwritten because request #1 always prefills from position 0.
      const bool ssm_state = st.model->prefix_state_supported();
      bool state_saved = false;
      if (ssm_state)
        state_saved = st.model->prefix_state_save(0);
      if (ssm_state && !state_saved)
        std::fprintf(stderr, "[serve] WARN: prewarm could not snapshot the clean state\n");
      try {
      const auto t_w0 = Clock::now();
      const std::vector<int> head(warm_ids.begin(), warm_ids.begin() + chunk);
      st.model->forward(head, 0, /*last_row_only=*/true);  // canonical prefill chunk
      const double chunk_ms = ms_since(t_w0);
      const auto t_t1 = Clock::now();
      const std::vector<int> tail(warm_ids.begin() + chunk, warm_ids.begin() + chunk + 8);
      st.model->forward(tail, chunk, /*last_row_only=*/true);  // small-tail bucket
      const double tail_ms = ms_since(t_t1);
      const auto t_d0 = Clock::now();
      st.model->forward({warm_ids[0]}, chunk + 8, /*last_row_only=*/true);  // decode graph
      const double dec_ms = ms_since(t_d0);
      if (ssm_state) st.model->prefix_state_restore(0);
      std::printf("[serve] prewarm   : %d-token prefill %.0f ms, 8-token tail %.0f ms, "
                  "1-token decode %.0f ms (graphs captured, KV/SSM reset)\n",
                  chunk, chunk_ms, tail_ms, dec_ms);
      std::fflush(stdout);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[serve] WARN: prewarm failed (%s); continuing without prewarm\n",
                     e.what());
      }
    }
  }

  http::Server server(opts.host, opts.port);
  server.set_handler([&st](const http::Request& req, http::Conn& conn) {
    try {
      route(st, req, conn);
    } catch (const UserError& e) {
      if (conn.stream_open()) conn.end_stream();
      else if (!conn.failed()) conn.send_error(400, e.what());
    } catch (const std::exception& e) {
      if (conn.stream_open()) conn.end_stream();
      else if (!conn.failed()) conn.send_error(500, e.what());
    }
  });
  g_state = &st;
  g_server = &server;
#ifdef _WIN32
  SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#endif

  std::string err;
  if (!server.start(&err)) {
    std::fprintf(stderr, "[serve] ERROR: %s\n", err.c_str());
    g_server = nullptr;
    g_state = nullptr;
    return 1;
  }
  std::printf("listening on http://%s:%d (model=%s, ctx=%d)\n",
              opts.host == "0.0.0.0" ? "127.0.0.1" : opts.host.c_str(), server.port(),
              opts.model.c_str(), st.ctx);
  std::fflush(stdout);

  // Double-click launcher: the model is loaded and the listener is up, so the UI
  // is guaranteed to answer.  Hooks through the shell so the default browser and
  // its tab reuse behaviour are used.
  if (opts.open_browser) {
#if defined(_WIN32)
    const std::string ui_host =
        (opts.host.empty() || opts.host == "0.0.0.0" || opts.host == "::")
            ? std::string("127.0.0.1")
            : opts.host;
    const std::string url =
        "http://" + ui_host + ":" + std::to_string(server.port()) + "/";
    std::printf("[serve] browser    : %s\n", url.c_str());
    std::fflush(stdout);
    const auto rc = reinterpret_cast<intptr_t>(
        ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (rc <= 32)
      std::fprintf(stderr,
                   "[serve] WARN: could not launch a browser (ShellExecute=%lld); open %s manually\n",
                   static_cast<long long>(rc), url.c_str());
#else
    std::fprintf(stderr, "[serve] note: --open-browser is only implemented on Windows\n");
#endif
  }

  const bool ok = server.run(&err);
  g_server = nullptr;
  std::printf("[serve] shutting down ...\n");
  std::fflush(stdout);
  st.model.reset();  // free VRAM before returning
  g_state = nullptr;
  if (!ok) {
    std::fprintf(stderr, "[serve] ERROR: %s\n", err.c_str());
    return 1;
  }
  std::printf("[serve] stopped; VRAM released\n");
  std::fflush(stdout);
  return 0;
}

}  // namespace serve
}  // namespace engine
