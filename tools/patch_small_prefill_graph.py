#!/usr/bin/env python3
"""Apply the small-n prefill CUDA-graph change to the isolated worktree.

Run from the worktree root:  python tools/patch_small_prefill_graph.py
Every anchor is asserted to appear exactly once, so a drifted tree fails loudly.
"""
import sys


def patch(path, subs):
    s = open(path, encoding='utf-8').read()
    for old, new in subs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f'{path}: anchor count={n} for {old[:80]!r}')
        s = s.replace(old, new)
    open(path, 'w', encoding='utf-8', newline='\n').write(s)
    print('patched', path)


# ------------------------------------------------------------------ config.cpp
CFG_OLD = '''    {"prefill_graph", "ENGINE_PREFILL_GRAPH", "prefill-graph", false, false, Type::Bool, "false",
     0, 0,
     "capture/replay a whole-pipeline CUDA graph per prefill chunk size (one launch "
     "instead of ~2300: the WDDM launch overhead is ~390 ms per 1.6k-token chunk)"},
'''
CFG_NEW = '''    {"prefill_small_graph", "ENGINE_PREFILL_SMALL_GRAPH", "prefill-small-graph", false,
     false, Type::Bool, "true", 0, 0,
     "CUDA-graph the whole prefill pipeline for SHORT batches (the tail of a cache-hit "
     "request): a 50-token tail costs ~180-200 ms, most of it ~700 WDDM kernel launches, "
     "so replay is the cheapest lever.  Captured per exact n_tokens, LRU bounded; "
     "--no-graph / ENGINE_NO_GRAPH=1 still disables it"},
    {"prefill_graph_small_min", "ENGINE_PREFILL_GRAPH_SMALL_MIN", "prefill-graph-small-min",
     true, false, Type::Int, "8", 2, 65536,
     "smallest n_tokens that gets a small-n prefill graph (below it the capture cost is "
     "not worth it)"},
    {"prefill_graph_small_max", "ENGINE_PREFILL_GRAPH_SMALL_MAX", "prefill-graph-small-max",
     true, false, Type::Int, "128", 2, 65536,
     "largest n_tokens that gets a small-n prefill graph: exact-n buckets, tails outside "
     "this range and large prefill chunks stay un-graphed unless --prefill-graph is set"},
    {"prefill_graph_slots", "ENGINE_PREFILL_GRAPH_SLOTS", "prefill-graph-slots", true, false,
     Type::Int, "16", 1, 256,
     "how many small-n prefill graphs are kept before the least recently used one is "
     "dropped (tail lengths vary, so a recapture may happen)"},
''' + CFG_OLD

# ------------------------------------------------------------------ runner.h
RH_OLD = '''  std::map<int, std::unique_ptr<cuda::Graph>> prefill_graphs_;
  bool prefill_graph_ = true;
  int x_half_max_nin_ = 0;   // widest matmul input seen (sizes the fp16 staging buffer)
'''
RH_NEW = '''  // Prefill CUDA graphs, keyed by n_tokens * 2 + (last_row_only ? 1 : 0): every frozen
  // launch configuration depends on those two, while the per-call position lives in
  // device memory (pos_dev_), so one graph is valid for any chunk boundary.
  //   * small n (cache-hit tail): default ON, LRU bounded -- tails are launch bound;
  //   * large chunk (whole-prompt prefill): opt-in (prefill_graph_), measured no gain.
  std::map<long long, std::unique_ptr<cuda::Graph>> prefill_graphs_;
  std::list<long long> prefill_graph_lru_;   // most recently used first
  bool prefill_graph_ = false;               // opt-in graph for ANY n_tokens
  bool prefill_small_graph_ = true;          // default graph for small n_tokens
  int prefill_graph_small_min_ = 8;
  int prefill_graph_small_max_ = 128;
  int prefill_graph_slots_ = 16;
  int x_half_max_nin_ = 0;   // widest matmul input seen (sizes the fp16 staging buffer)
'''
RH_INC_OLD = '#include <map>'
RH_INC_NEW = '#include <list>\n#include <map>'

# ------------------------------------------------------------------ runner.cpp
RC_CFG_OLD = '''  prefill_graph_ = cfg.get_bool("prefill_graph");'''
RC_CFG_NEW = '''  prefill_graph_ = cfg.get_bool("prefill_graph");
  prefill_small_graph_ = cfg.get_bool("prefill_small_graph");
  prefill_graph_small_min_ = static_cast<int>(cfg.get_int("prefill_graph_small_min"));
  prefill_graph_small_max_ = static_cast<int>(cfg.get_int("prefill_graph_small_max"));
  prefill_graph_slots_ = static_cast<int>(cfg.get_int("prefill_graph_slots"));'''

RC_GRAPH_OLD = '''  // Prefill chunks: replay a captured graph keyed by the chunk size.  The pipeline is
  // position-agnostic (the position lives in pos_dev_, uploaded before the launch), so
  // one graph per token count is valid for every chunk boundary.
  static const bool pf_graph = prefill_graph_;
  if (n > 1 && use_graph_ && pf_graph) {
    auto& slot = prefill_graphs_[n];
    if (!slot) {
      // The fp16 staging buffer must be big enough before capture starts: allocating
      // inside a capture is illegal.
      const size_t cap = static_cast<size_t>(rows_) * static_cast<size_t>(x_half_max_nin_) * 2;
      if (cap > x_half_.size()) x_half_ = cuda::Buffer(cap);
      try {
        cuda::stream_sync();
        slot = std::make_unique<cuda::Graph>();
        slot->capture_begin();
        run_pipeline(n, start_pos, last_row_only);
        slot->capture_end();
        if (ENGINE_ENV_ON("ENGINE_DBG_PFGRAPH"))
          std::fprintf(stderr, "[pfgraph] captured n=%d\\n", n);
      } catch (const std::exception& e) {
        slot.reset();
        prefill_graph_ = false;
        cuda::stream_sync();
        std::fprintf(stderr, "[pfgraph] disabled: %s\\n", e.what());
        run_pipeline(n, start_pos, last_row_only);
      }
    }
    // capture and replay both run the pipeline through the graph: a capture does not
    // execute anything, so the graph is launched immediately after it is recorded.
    slot->launch();
    cuda::stream_sync();
    if (ENGINE_ENV_ON("ENGINE_DBG_PFGRAPH"))
      std::fprintf(stderr, "[pfgraph] ran n=%d (ready=%d)\\n", n, (int)slot->ready());
    if (kProfFwd) {
      cuda::stream_sync();
      prof_ev0_->record();
      prof_ev1_->record();
    }
  } else if (n == 1 && use_graph_) {'''

RC_GRAPH_NEW = '''  // Prefill graphs, two regimes:
  //  * small n (the tail of a cache-hit request): measured ~180-200 ms fixed cost for a
  //    50-token prefill, most of it ~700 WDDM kernel launches -> default ON, captured by
  //    exact n and replayed; LRU-bounded because tail lengths vary;
  //  * large chunk (whole-prompt prefill): GPU bound (measured 1506 vs 1478 ms) -> opt-in.
  // The pipeline is position-agnostic (the position lives in pos_dev_, uploaded before
  // the launch), so a graph is valid for any chunk boundary with the same (n, head mode).
  static const bool pf_graph_big = prefill_graph_;
  const bool pf_graph_small = prefill_small_graph_ && n >= prefill_graph_small_min_ &&
                              n <= prefill_graph_small_max_;
  if (n > 1 && use_graph_ && (pf_graph_big || pf_graph_small)) {
    const long long gkey = static_cast<long long>(n) * 2 + (last_row_only ? 1 : 0);
    auto it = prefill_graphs_.find(gkey);
    bool graph_ok = false;
    if (it == prefill_graphs_.end()) {
      // Capture must not cuMemAlloc (illegal while capturing), so size the fp16 staging
      // buffer first; the widest matmul input is known after any real matmul and is also
      // pre-sized in the constructor.
      const size_t nin = static_cast<size_t>(x_half_max_nin_ > 0 ? x_half_max_nin_ : 32768);
      const size_t cap = static_cast<size_t>(rows_) * nin * 2;
      if (cap > x_half_.size()) x_half_ = cuda::Buffer(cap);
      // LRU: drop the least recently used pipeline once the slot budget is reached.
      while (static_cast<int>(prefill_graphs_.size()) >= prefill_graph_slots_ &&
             !prefill_graph_lru_.empty()) {
        const long long victim = prefill_graph_lru_.back();
        prefill_graph_lru_.pop_back();
        auto vit = prefill_graphs_.find(victim);
        if (vit != prefill_graphs_.end()) prefill_graphs_.erase(vit);
      }
      auto slot = std::make_unique<cuda::Graph>();
      try {
        cuda::stream_sync();
        slot->capture_begin();
        run_pipeline(n, start_pos, last_row_only);
        slot->capture_end();
        it = prefill_graphs_.emplace(gkey, std::move(slot)).first;
        prefill_graph_lru_.push_front(gkey);
        graph_ok = true;
        if (ENGINE_ENV_ON("ENGINE_DBG_PFGRAPH"))
          std::fprintf(stderr, "[pfgraph] captured n=%d rowsonly=%d\\n", n, (int)last_row_only);
      } catch (const std::exception& e) {
        // Never fall through to a null graph: run this call the plain way and stop
        // capturing (a capture failure is usually a one-time allocation/host-sync).
        prefill_graph_ = false;
        prefill_small_graph_ = false;
        cuda::stream_sync();
        std::fprintf(stderr, "[pfgraph] disabled: %s\\n", e.what());
      }
    } else {
      prefill_graph_lru_.remove(gkey);
      prefill_graph_lru_.push_front(gkey);
      graph_ok = true;
    }
    if (graph_ok) {
      // A capture does not execute anything, so the first call is served by launching the
      // freshly recorded graph; the capture cost is paid once per n.
      it->second->launch();
      cuda::stream_sync();
      if (ENGINE_ENV_ON("ENGINE_DBG_PFGRAPH"))
        std::fprintf(stderr, "[pfgraph] ran n=%d rowsonly=%d ready=%d\\n", n,
                     (int)last_row_only, (int)it->second->ready());
    } else {
      run_pipeline(n, start_pos, last_row_only);
    }
    if (kProfFwd) {
      cuda::stream_sync();
      prof_ev0_->record();
      prof_ev1_->record();
    }
  } else if (n == 1 && use_graph_) {'''

RC_XH_OLD = '  pos_dev_ = cuda::Buffer(4);'
RC_XH_NEW = '''  // Pre-size the fp16 activation staging buffer: a CUDA-graph capture must not allocate
  // (cuMemAlloc is illegal while capturing) and matmul()'s lazy grow would do exactly
  // that on the first prefill of a new shape.  The widest matmul input is one of these
  // products for every supported architecture (hidden, ffn, head_dim * heads).
  {
    const size_t max_nin = static_cast<size_t>(
        std::max(std::max(hidden_, inter_), std::max(n_head_ * head_dim_, n_kv_ * head_dim_)));
    const size_t cap = static_cast<size_t>(rows_) * max_nin * 2;
    if (cap > x_half_.size()) x_half_ = cuda::Buffer(cap);
  }
  pos_dev_ = cuda::Buffer(4);'''

patch('src/config.cpp', [(CFG_OLD, CFG_NEW)])
patch('include/engine/runner.h', [(RH_INC_OLD, RH_INC_NEW), (RH_OLD, RH_NEW)])
patch('src/runner.cpp', [(RC_CFG_OLD, RC_CFG_NEW), (RC_GRAPH_OLD, RC_GRAPH_NEW),
                         (RC_XH_OLD, RC_XH_NEW)])
print('done; next: bash build.sh')
