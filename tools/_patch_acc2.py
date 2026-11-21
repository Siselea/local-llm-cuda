p = 'src/runner.cpp'
s = open(p, encoding='utf-8').read()

# ------------------------------------------------------------------ 1) accumulators
old = """double layers_ms = 0, cpu_ms = 0, gpu_ms = 0, gemv_ms = 0;"""
new = """double layers_ms = 0, cpu_ms = 0, gpu_ms = 0, gemv_ms = 0;
double moe_ms = 0, moe_pre_ms = 0, moe_route_ms = 0, moe_tail_ms = 0;
int warm_left = 4;   // decode tokens skipped before counting (graph capture / page-in)"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """static const bool on = std::getenv("ENGINE_PROF_ACC") != nullptr;"""
new = """static const bool on = std::getenv("ENGINE_PROF_ACC") != nullptr;
inline bool counting(int n_tokens) { return on && n_tokens == 1 && warm_left <= 0; }"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# init warm_left from env (lazy, first use)
old = """inline bool counting(int n_tokens) { return on && n_tokens == 1 && warm_left <= 0; }"""
new = """bool warm_init = false;
inline bool counting(int n_tokens) {
  if (!warm_init) {
    warm_init = true;
    const char* e = std::getenv("ENGINE_PROF_ACC_SKIP");
    if (e) warm_left = std::atoi(e);
  }
  return on && n_tokens == 1 && warm_left <= 0;
}
inline void tick_warm(int n_tokens) {
  if (on && n_tokens == 1 && warm_left > 0) --warm_left;
}"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# dump: add the new sub-segments
old = """  std::fprintf(stderr, "[acc]   CPU layer parts: host GEMV %.3f ms/layer (%6.2f%%)  sync %.3f ms/layer (%5.1f%%)  launch %.3f ms/layer (%4.1f%%)  other %.3f ms/layer\\n","""
new = """  std::fprintf(stderr,
               "[acc]   CPU sub: pre(router+D2H+shared enqueue) %.3f  route(host) %.3f  "
               "tail(upload+add3+host part) %.3f  gemv %.3f  sync %.3f  launch %.3f  rest %.3f ms/layer\\n",
               ncpu > 0 ? moe_pre_ms / per / ncpu : 0.0,
               ncpu > 0 ? moe_route_ms / per / ncpu : 0.0,
               ncpu > 0 ? moe_tail_ms / per / ncpu : 0.0, ncpu > 0 ? gemv / ncpu : 0.0,
               ncpu > 0 ? csync / ncpu : 0.0, ncpu > 0 ? claunch / ncpu : 0.0,
               ncpu > 0 ? (cpu - gemv - csync - claunch) / ncpu : 0.0);
  std::fprintf(stderr, "[acc]   run_moe total %.3f ms/token (%.3f ms/layer over %.2f layers)\\n",
               moe_ms / per, ncpu + ngpu > 0 ? moe_ms / per / (ncpu + ngpu) : 0.0, ncpu + ngpu);
  std::fprintf(stderr, "[acc]   CPU layer parts: host GEMV %.3f ms/layer (%6.2f%%)  sync %.3f ms/layer (%5.1f%%)  launch %.3f ms/layer (%4.1f%%)  other %.3f ms/layer\\n","""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# ------------------------------------------------------- 2) forward_impl: warm-up skip
old = """  const auto acc_t_entry = std::chrono::steady_clock::now();"""
new = """  const auto acc_t_entry = std::chrono::steady_clock::now();
  prof_acc::tick_warm(n);"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """  const bool acc_tok = prof_acc::on && n == 1;
  if (acc_tok) prof_acc::pre_ms += prof_acc::since(acc_t_entry);"""
new = """  const bool acc_tok = prof_acc::counting(n);
  if (acc_tok) prof_acc::pre_ms += prof_acc::since(acc_t_entry);"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# ------------------------------------------------------- 3) run_pipeline: counting()
old = """  const bool acc_on = prof_acc::on && n == 1;"""
new = """  const bool acc_on = prof_acc::counting(n);"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# ------------------------------------------------------- 4) run_moe: segment marks
old = """void DenseRunner::run_moe(int layer, int n_tokens) {
  auto& L = layers_[layer];"""
new = """void DenseRunner::run_moe(int layer, int n_tokens) {
  auto& L = layers_[layer];
  const bool acc_moe = prof_acc::counting(n_tokens);
  const auto acc_moe0 = std::chrono::steady_clock::now();"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """  // Host-resident experts: issue the shared-expert GPU block now so the flush below is
  // the only one and the queue drains before the host GEMV starts.
  if (any_cpu_pre) run_shared_block();"""
new = """  if (acc_moe) prof_acc::moe_pre_ms += prof_acc::since(acc_moe0);
  // Host-resident experts: issue the shared-expert GPU block now so the flush below is
  // the only one and the queue drains before the host GEMV starts.
  if (any_cpu_pre) run_shared_block();"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """  if (dl_routing) {
    cuda::stream_sync();
    rl = static_cast<const float*>(rl_pin_.data());
  }

  // host-side routing: softmax -> top-k -> renormalize (matches the reference)"""
new = """  if (dl_routing) {
    cuda::stream_sync();
    rl = static_cast<const float*>(rl_pin_.data());
  }
  const auto acc_t_route0 = std::chrono::steady_clock::now();

  // host-side routing: softmax -> top-k -> renormalize (matches the reference)"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """  g_prof_router_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();"""
new = """  g_prof_router_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
  if (acc_moe && !use_dev) prof_acc::moe_route_ms += prof_acc::since(acc_t_route0);"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """  // ---- fused path: one kernel for gate+up+silu and one for down+scatter ----"""
new = """  const auto acc_t_tail0 = std::chrono::steady_clock::now();
  // ---- fused path: one kernel for gate+up+silu and one for down+scatter ----"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """  if (std::getenv("ENGINE_PIPE_PROF")) cuda::sync();
  if (kCpuDbg)"""
new = """  if (std::getenv("ENGINE_PIPE_PROF")) cuda::sync();
  if (acc_moe) {
    prof_acc::moe_ms += prof_acc::since(acc_moe0);
    prof_acc::moe_tail_ms += prof_acc::since(acc_t_tail0);
  }
  if (kCpuDbg)"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# gemv accumulation honours the warm-up skip
old = """      if (prof_acc::on && n_tokens == 1) prof_acc::gemv_ms += prof_acc::since(bt0);"""
new = """      if (prof_acc::counting(n_tokens)) prof_acc::gemv_ms += prof_acc::since(bt0);"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

open(p, 'w', encoding='utf-8', newline='').write(s)
print("patched acc v2 OK")
