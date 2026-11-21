import re

p = 'src/runner.cpp'
s = open(p, encoding='utf-8').read()

# ---------------------------------------------------------------- 1) prof_acc block
anchor = """double g_prof_mla_ms = 0.0, g_prof_moe_ms = 0.0;
double g_prof_router_ms = 0.0, g_prof_experts_ms = 0.0, g_prof_shared_ms = 0.0;
double g_prof_skip_ms = 0.0, g_prof_active_ms = 0.0;
"""
assert s.count(anchor) == 1
acc_block = anchor + """
// ---- per-token accounting (ENGINE_PROF_ACC=1) --------------------------------
// WHY: the historical 35B ledger was built from ENGINE_DBG_LAYERMS, which writes 40
// lines/token to an unbuffered stderr.  On this host that costs ~10 ms/token, so the
// sum of the per-layer lines can never be reconciled with the decoded-token wall clock
// (measured: 32.1 ms of layer lines inside a 43.1 ms token).  This accumulator keeps
// the very same segments in memory and dumps them ONCE at exit: a handful of
// steady_clock reads per layer, no I/O in the hot path.
namespace prof_acc {
static const bool on = std::getenv("ENGINE_PROF_ACC") != nullptr;
double token_ms = 0, pre_ms = 0, pipe_ms = 0, head_gpu_ms = 0, head_copy_ms = 0;
double layers_ms = 0, cpu_ms = 0, gpu_ms = 0, gemv_ms = 0;
double cpu_sync_ms = 0, cpu_launch_ms = 0, gpu_sync_ms = 0, gpu_launch_ms = 0;
long long tokens = 0, layers = 0, cpu_layers = 0, gpu_layers = 0;
long long cpu_syncs = 0, cpu_launches = 0, gpu_syncs = 0, gpu_launches = 0;

inline double since(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a)
      .count();
}
void dump() {
  if (!on || tokens <= 0) return;
  const double per = static_cast<double>(tokens);
  const double wall = token_ms / per, pre = pre_ms / per, pipe = pipe_ms / per;
  const double gpuh = head_gpu_ms / per, hcopy = head_copy_ms / per;
  const double lsum = layers_ms / per, cpu = cpu_ms / per, gpu = gpu_ms / per;
  const double gemv = gemv_ms / per;
  const double csync = cpu_sync_ms / per, claunch = cpu_launch_ms / per;
  const double gsync = gpu_sync_ms / per, glaunch = gpu_launch_ms / per;
  const double ncpu = static_cast<double>(cpu_layers) / per;
  const double ngpu = static_cast<double>(gpu_layers) / per;
  std::fprintf(stderr, "\\n[acc] ===== per decode token (n=%lld, %lld layers) =====\\n", tokens,
               layers / tokens);
  std::fprintf(stderr, "[acc] wall                %7.3f ms\\n", wall);
  std::fprintf(stderr, "[acc]   pre (embed/upload/setpos) %7.3f ms\\n", pre);
  std::fprintf(stderr, "[acc]   pipeline          %7.3f ms\\n", pipe);
  std::fprintf(stderr, "[acc]     sum(run_layer) %7.3f   -> loop/other %7.3f\\n", lsum,
               pipe - lsum - gpuh);
  std::fprintf(stderr, "[acc]     head in pipeline (final norm + lm_head + topk) %7.3f\\n",
               gpuh);
  std::fprintf(stderr, "[acc]   head copy (topk/logits D2H + sync) %7.3f ms\\n", hcopy);
  std::fprintf(stderr, "[acc]   accounted %7.3f ms   UNACCOUNTED %7.3f ms\\n",
               pre + pipe + hcopy, wall - pre - pipe - hcopy);
  std::fprintf(stderr, "[acc] layers: %lld total | CPU %.2f layers x %.3f ms = %.3f | GPU %.2f x %.3f = %.3f\\n",
               layers / tokens, ncpu, ncpu > 0 ? cpu / ncpu : 0.0, cpu, ngpu,
               ngpu > 0 ? gpu / ngpu : 0.0, gpu);
  std::fprintf(stderr, "[acc]   CPU layer parts: host GEMV %.3f ms/layer (%6.2f%%)  sync %.3f ms/layer (%5.1f%%)  launch %.3f ms/layer (%4.1f%%)  other %.3f ms/layer\\n",
               ncpu > 0 ? gemv / ncpu : 0.0, cpu > 0 ? 100.0 * gemv / cpu : 0.0,
               ncpu > 0 ? csync / ncpu : 0.0, cpu > 0 ? 100.0 * csync / cpu : 0.0,
               ncpu > 0 ? claunch / ncpu : 0.0, cpu > 0 ? 100.0 * claunch / cpu : 0.0,
               ncpu > 0 ? (cpu - gemv - csync - claunch) / ncpu : 0.0);
  std::fprintf(stderr, "[acc]   GPU layer parts: sync %.3f ms/layer  launch %.3f ms/layer  other %.3f ms/layer\\n",
               ngpu > 0 ? gsync / ngpu : 0.0, ngpu > 0 ? glaunch / ngpu : 0.0,
               ngpu > 0 ? (gpu - gsync - glaunch) / ngpu : 0.0);
  std::fprintf(stderr, "[acc] per token: launches %lld (CPU %lld / GPU %lld)  syncs %lld (CPU %lld / GPU %lld)\\n",
               (cpu_launches + gpu_launches) / tokens, cpu_launches / tokens,
               gpu_launches / tokens, (cpu_syncs + gpu_syncs) / tokens, cpu_syncs / tokens,
               gpu_syncs / tokens);
  std::fprintf(stderr,
               "[acc] CPU-layer overhead total = %7.3f ms/token (sync %.3f + launch %.3f + other %.3f)\\n",
               cpu - gemv, csync, claunch, cpu - gemv - csync - claunch);
}
}  // namespace prof_acc
"""
s = s.replace(anchor, acc_block, 1)

# ---------------------------------------------------------------- 2) destructor dump
old = """DenseRunner::~DenseRunner() {
  cuda::kprof_dump();
}"""
new = """DenseRunner::~DenseRunner() {
  cuda::kprof_dump();
  prof_acc::dump();
}"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# ------------------------------------------------- 3) run_pipeline: layer accounting
old = """  const auto pipe_t0 = std::chrono::steady_clock::now();
  for (int l = 0; l < n_layer; ++l) {
    if (pipe_prof) pev0.record();
    const auto lt0 = std::chrono::steady_clock::now();"""
new = """  const auto pipe_t0 = std::chrono::steady_clock::now();
  const bool acc_on = prof_acc::on && n == 1;
  double acc_layers = 0.0;
  for (int l = 0; l < n_layer; ++l) {
    if (pipe_prof) pev0.record();
    const auto lt0 = std::chrono::steady_clock::now();"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """    cpu_launch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lt0).count();
    if (pipe_prof) pev1.record();"""
new = """    const double layer_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lt0)
            .count();
    cpu_launch_ms += layer_ms;
    if (acc_on) {
      const auto& LW = layers_[l];
      const bool is_cpu = ir_.moe.n_expert > 0 && LW.is_moe && LW.cpu_begin == 0;
      acc_layers += layer_ms;
      if (is_cpu) {
        prof_acc::cpu_ms += layer_ms;
        prof_acc::cpu_sync_ms += cuda::g_sync_ms - layer_sm0;
        prof_acc::cpu_launch_ms += cuda::g_launch_ms - layer_lm0;
        prof_acc::cpu_syncs += cuda::g_sync_calls - layer_sc0;
        prof_acc::cpu_launches += cuda::g_launch_calls - layer_lc0;
        ++prof_acc::cpu_layers;
      } else {
        prof_acc::gpu_ms += layer_ms;
        prof_acc::gpu_sync_ms += cuda::g_sync_ms - layer_sm0;
        prof_acc::gpu_launch_ms += cuda::g_launch_ms - layer_lm0;
        prof_acc::gpu_syncs += cuda::g_sync_calls - layer_sc0;
        prof_acc::gpu_launches += cuda::g_launch_calls - layer_lc0;
        ++prof_acc::gpu_layers;
      }
      ++prof_acc::layers;
    }
    if (pipe_prof) pev1.record();"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# head inside pipeline: measure from after the loop to end of run_pipeline
old = """  if (pipe_prof) {
    cuda::sync();
    const double wall ="""
new = """  const auto acc_head0 = std::chrono::steady_clock::now();
  if (pipe_prof) {
    cuda::sync();
    const double wall ="""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """    void* a2[] = {&ppv, &ppi, &nc, &kk, &poi, &pov};
    cuda::launch(k_topkfinal_, 1, 1, 1, 256, 1, 1, 0, a2);
  }
}"""
new = """    void* a2[] = {&ppv, &ppi, &nc, &kk, &poi, &pov};
    cuda::launch(k_topkfinal_, 1, 1, 1, 256, 1, 1, 0, a2);
  }
  if (acc_on) {
    prof_acc::layers_ms += acc_layers;
    prof_acc::head_gpu_ms += prof_acc::since(acc_head0);
  }
}"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# ----------------------------------------------------- 4) forward_impl: token split
old = """  const auto fwd_t_entry = std::chrono::steady_clock::now();
  std::vector<float> x(static_cast<size_t>(n) * hidden_);"""
new = """  const auto fwd_t_entry = std::chrono::steady_clock::now();
  const auto acc_t_entry = std::chrono::steady_clock::now();
  std::vector<float> x(static_cast<size_t>(n) * hidden_);"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = """  const auto fwd_t_pre = std::chrono::steady_clock::now();

  static const bool kProfFwd = std::getenv("ENGINE_PROFILE") != nullptr;"""
new = """  const auto fwd_t_pre = std::chrono::steady_clock::now();
  const bool acc_tok = prof_acc::on && n == 1;
  if (acc_tok) prof_acc::pre_ms += prof_acc::since(acc_t_entry);

  static const bool kProfFwd = std::getenv("ENGINE_PROFILE") != nullptr;"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# end of the pipeline branch (just before the top-k head): accumulate pipe time
old = """  // Device top-k head: only `k` (index,value) pairs cross PCIe instead of the logits row.
  const bool want_topk"""
new = """  const auto acc_pipe_end = std::chrono::steady_clock::now();
  if (acc_tok) prof_acc::pipe_ms += prof_acc::since(fwd_t_pre);
  // Device top-k head: only `k` (index,value) pairs cross PCIe instead of the logits row.
  const bool want_topk"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# topk path return: head copy + token end
old = """                   tk->idx[0]);
    }
    return {};
  }"""
new = """                   tk->idx[0]);
    }
    if (acc_tok) {
      prof_acc::head_copy_ms += prof_acc::since(acc_pipe_end);
      prof_acc::token_ms += prof_acc::since(acc_t_entry);
      ++prof_acc::tokens;
    }
    return {};
  }"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# normal path (full logits download) token end
old = """  if (const char* lp = std::getenv("ENGINE_LOGITS")) {
    std::ofstream f(lp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()),
            static_cast<std::streamsize>(out.size() * 4));
  }
  return out;
}"""
new = """  if (const char* lp = std::getenv("ENGINE_LOGITS")) {
    std::ofstream f(lp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()),
            static_cast<std::streamsize>(out.size() * 4));
  }
  if (acc_tok) {
    prof_acc::head_copy_ms += prof_acc::since(acc_pipe_end);
    prof_acc::token_ms += prof_acc::since(acc_t_entry);
    ++prof_acc::tokens;
  }
  return out;
}"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

# ------------------------------------------------------- 5) run_moe: host GEMV sum
old = """      const auto bt0 = std::chrono::steady_clock::now();
      cpu::moe_forward_mt(L.h_type_g, ag.data(), L.h_type_u, au.data(), L.h_type_d, ad.data(),
                          atok.data(), aw.data(), static_cast<int>(atok.size()), n_tokens,
                          hidden_, inter, hn_host, xout_host);
      cpu_batched = true;"""
new = """      const auto bt0 = std::chrono::steady_clock::now();
      cpu::moe_forward_mt(L.h_type_g, ag.data(), L.h_type_u, au.data(), L.h_type_d, ad.data(),
                          atok.data(), aw.data(), static_cast<int>(atok.size()), n_tokens,
                          hidden_, inter, hn_host, xout_host);
      if (prof_acc::on && n_tokens == 1) prof_acc::gemv_ms += prof_acc::since(bt0);
      cpu_batched = true;"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

open(p, 'w', encoding='utf-8', newline='').write(s)
print("patched runner.cpp OK")
