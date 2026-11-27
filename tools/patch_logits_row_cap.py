#!/usr/bin/env python3
"""Reserve ONE logits row for prefill instead of rows*vocab, so the prefill chunk can grow.

Why: activation_bytes counted vocab*4 per row (993 KB of the 1.28 MB per-row estimate
for a 248k vocab, 78%), and logits_ was really allocated rows*vocab*4 -- but a
last_row_only prefill head computes exactly one logits row.  At ctx 32768 the 9B is
capped at ~1024 prefill rows by that buffer, so a 13.4k prompt runs 14 chunks and
re-dequantises the whole model 14 times (~65 ms each after the x8 dequant work).

Change:
  * logits_ reserved for `prefill_logits_rows` (default 1) rows;
  * a full-logits call (last_row_only=false / MTP verify) grows it on demand;
  * activation_bytes() counts the same constant, so the VRAM-derived chunk growth
    (and the placement decision that uses the same function) sees the real footprint.
"""
CFG = 'src/config.cpp'
HDR = 'include/engine/runner.h'
RUN = 'src/runner.cpp'
SRV = 'src/serve.cpp'

# ---------------------------------------------------------------- config.cpp
s = open(CFG, encoding='utf-8').read()
anchor = '    {"prefill_small_graph", "ENGINE_PREFILL_SMALL_GRAPH", "prefill-small-graph", false,'
assert s.count(anchor) == 1
new_entry = '''    {"prefill_logits_rows", "ENGINE_PREFILL_LOGITS_ROWS", "prefill-logits-rows", true,
     false, Type::Int, "1", 0, 1 << 20,
     "how many logits rows the prefill activation budget reserves (0 = one per chunk row, "
     "the old behaviour).  A last_row_only prefill head computes exactly ONE row, so "
     "reserving 1 instead of rows*vocab*4 frees ~1 MB per row at a 248k vocab and lets the "
     "prefill chunk grow ~4x at the same VRAM (fewer whole-model dequant passes); a "
     "full-logits call grows the buffer on demand"},
''' + anchor
s = s.replace(anchor, new_entry)
open(CFG, 'w', encoding='utf-8', newline='\n').write(s)
print('config.cpp: prefill_logits_rows added')

# ---------------------------------------------------------------- runner.h
s = open(HDR, encoding='utf-8').read()
anchor = '  bool prefill_small_graph_ = true;          // default graph for small n_tokens'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''
  // Logits rows reserved for prefill chunk sizing (default 1: a last_row_only prefill
  // head computes one row; 0 restores rows*vocab*4).  ensure_logits_rows() grows the
  // buffer when a caller asks for full logits.
  int prefill_logits_rows_ = 1;''')
anchor2 = '  size_t activation_bytes(int rows) const;'
if anchor2 in s:
    s = s.replace(anchor2, anchor2 + '\n  // Grow logits_ if a full-logits call needs more rows than were reserved.\n  void ensure_logits_rows(int rows);')
else:
    # find the private section around activation_bytes declaration
    raise SystemExit('activation_bytes declaration not found')
open(HDR, 'w', encoding='utf-8', newline='\n').write(s)
print('runner.h: member + helper declared')

# ---------------------------------------------------------------- runner.cpp
s = open(RUN, encoding='utf-8').read()

# 1) read the config (load_config_snapshot runs first in the constructor)
anchor = '  prefill_small_graph_ = cfg.get_bool("prefill_small_graph");'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '\n  prefill_logits_rows_ = static_cast<int>(cfg.get_int("prefill_logits_rows"));')

# 2) allocate logits_ for the reserved rows only
old = '  logits_ = cuda::Buffer(rows * static_cast<size_t>(vocab_) * 4);'
assert s.count(old) == 1
new = '''  // Reserve only the configured number of logits rows: a last_row_only prefill head
  // computes exactly one row, so rows*vocab*4 (1 MB/row at a 248k vocab, 78% of the
  // per-row activation estimate) needlessly capped the prefill chunk at ~1024 rows on
  // a 9B at ctx 32768, and every extra chunk re-dequantises the whole model.  A caller
  // that wants full logits grows the buffer on demand (ensure_logits_rows).
  {
    int logits_rows = static_cast<int>(rows);
    if (prefill_logits_rows_ > 0 && prefill_logits_rows_ < logits_rows)
      logits_rows = prefill_logits_rows_;
    logits_ = cuda::Buffer(static_cast<size_t>(std::max(1, logits_rows)) *
                           static_cast<size_t>(vocab_) * 4);
  }'''
s = s.replace(old, new)

# 3) activation_bytes: the vocab term is a constant, not per row
old = '''  per_row += static_cast<size_t>(inter_) * 4 * 3;
  per_row += static_cast<size_t>(vocab_) * 4;              // one logits row per token
  return r * per_row + (2ull << 20);
}'''
assert s.count(old) == 1
new = '''  per_row += static_cast<size_t>(inter_) * 4 * 3;
  // Logits: a last_row_only prefill head computes ONE row, so only the reserved rows
  // (default 1) are part of the footprint.  Keeping this in sync with the allocation
  // above is what lets the VRAM-derived chunk growth pick a much larger chunk.
  const int logits_rows =
      prefill_logits_rows_ > 0
          ? std::min<int>(prefill_logits_rows_, static_cast<int>(r))
          : static_cast<int>(r);
  return r * per_row +
         static_cast<size_t>(std::max(1, logits_rows)) * static_cast<size_t>(vocab_) * 4 +
         (2ull << 20);
}

void DenseRunner::ensure_logits_rows(int rows) {
  const size_t need = static_cast<size_t>(std::max(1, rows)) *
                      static_cast<size_t>(vocab_) * 4;
  if (logits_.size() >= need) return;
  logits_ = cuda::Buffer(need);
  std::fprintf(stderr,
               "[logits] grew the logits buffer to %d rows (%.0f MiB) for a full-logits "
               "call (prefill_logits_rows=%d)\\n",
               std::max(1, rows), static_cast<double>(need) / 1048576.0, prefill_logits_rows_);
}'''
s = s.replace(old, new)

# 4) grow before the head uses it (full-logits / verify calls)
old = '''  const int rows = last_row_only ? 1 : n;
  {
    void* px = static_cast<float*>(x_.data()) +'''
assert s.count(old) == 1
new = '''  const int rows = last_row_only ? 1 : n;
  ensure_logits_rows(rows);
  {
    void* px = static_cast<float*>(x_.data()) +'''
s = s.replace(old, new)
open(RUN, 'w', encoding='utf-8', newline='\n').write(s)
print('runner.cpp: 1-row logits reservation + lazy grow + activation_bytes synced')

# ---------------------------------------------------------------- serve.cpp: let VRAM decide
s = open(SRV, encoding='utf-8').read()
old = '''    if (!engine::config::was_set("prefill_rows_max"))
      engine::config::Config::get().set("prefill_rows_max", "1024",
                                        engine::config::Source::Cli, &perr);'''
assert s.count(old) == 1
new = '''    if (!engine::config::was_set("prefill_rows_max"))
      engine::config::Config::get().set("prefill_rows_max", "4096",
                                        engine::config::Source::Cli, &perr);'''
s = s.replace(old, new)
old2 = '''  // Serve default prefill chunk: 512 tokens (was 128).'''
assert s.count(old2) == 1
s = s.replace(old2, '''  // Serve default prefill chunk: 512 tokens (was 128), up to 4096 when VRAM allows --
  // logits are reserved for one row now, so the row budget is ~4x smaller per token.'''
  )
open(SRV, 'w', encoding='utf-8', newline='\n').write(s)
print('serve.cpp: prefill_rows_max 1024 -> 4096 (VRAM decides)')
