// Multi-token prediction (MTP) / greedy speculative decoding for qwen35-family GGUFs.
//
// The MTP block is a full attention+FFN layer that lives *past* the transformer stack
// (`blk.{n_layer}`) and carries four extra tensors:
//     nextn.eh_proj            [2*hidden -> hidden]  fuse(enorm(emb(t)), hnorm(h))
//     nextn.enorm / hnorm      RMSNorm weights for the two halves
//     nextn.shared_head_norm   RMSNorm applied before the *shared* output head
// It reuses the target model's `token_embd` and `output.weight` (no own embeddings).
//
// One draft step for position p+1 (predicting the token at p+2):
//     u = eh_proj( concat(enorm(emb(t_{p+1})), hnorm(h_p)) )     # [2H] -> [H]
//     y = mtp_layer(u, pos=p+1)                                  # own KV cache
//     d = argmax( output.weight @ shared_head_norm(y) )
// Chain drafting repeats the last two steps with h := y, t := d at the next position.
//
// Verification runs the *target* model on [t_next, d_1..d_k]; the draft prefix that
// agrees with the target's own arg-max is kept, so the emitted sequence is exactly the
// greedy sequence (speculative decoding is lossless).  Attention KV needs no rollback
// (position-indexed); the gated-delta-net `conv_state`/`rec_state` are snapshotted per
// token by the linear layers themselves (see spec_snapshot_ in qwen35.cpp).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/config.h"
#include "engine/cpu_matvec.h"
#include "engine/cuda.h"
#include "engine/ir.h"
#include "engine/runner.h"
#include "engine/runner_internal.h"

namespace runner {

namespace {
double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

// ------------------------------------------------------------------ loading ----
void DenseRunner::load_mtp() {
  mtp_ok_ = false;
  mtp_block_ = -1;
  if (ir_.mtp_block < 0 || ir_.mtp_layers <= 0) return;
  auto& cfg = engine::config::Config::get();
  if (!cfg.get_bool("mtp")) return;
  if (ir_.ssm.variant == "none") {
    std::fprintf(stderr,
                 "[mtp] MTP block detected (blk.%d) but this architecture has no qwen35 "
                 "layer implementation; MTP disabled\n",
                 ir_.mtp_block);
    return;
  }
  if (!lm_head_qt_.valid) {
    std::fprintf(stderr, "[mtp] output.weight is not a supported quantized tensor; MTP disabled\n");
    return;
  }
  const int b = ir_.mtp_block;
  const int H = hidden_;
  mtp_block_ = b;
  if (static_cast<int>(layers_.size()) <= b) layers_.resize(static_cast<size_t>(b) + 1);
  // The MTP block is always a full-attention layer, independent of the main stack's
  // full_attention_interval (its tensor list has attn_q/k/v/output + q/k norms).
  {
    const bool want_gpu = cfg.get_int("mtp_gpu") != 0;
    const int ne = ir_.moe.n_expert;
    int gpu_experts = 0;
    if (want_gpu && ne > 0) {
      // per expert: gate+up Q4_K (144 B / 256 values) + down Q6_K (210 B / 256)
      const size_t per_expert = static_cast<size_t>(expert_inter_) *
                                static_cast<size_t>(hidden_) * (144 * 2 + 210) / 256;
      const size_t need = static_cast<size_t>(ne) * per_expert;
      const size_t margin = static_cast<size_t>(cfg.get_int("mtp_vram_margin_mb")) << 20;
      const size_t free_now = cuda::mem_info().free_bytes;
      if (free_now > need + margin) gpu_experts = ne;
      std::fprintf(stderr,
                   "[mtp] MTP experts: %s (need %.0f MiB, free %.0f MiB, margin %.0f MiB)\n",
                   gpu_experts ? "GPU" : "CPU", static_cast<double>(need) / 1048576.0,
                   static_cast<double>(free_now) / 1048576.0,
                   static_cast<double>(margin) / 1048576.0);
    }
    mtp_gpu_experts_ = gpu_experts;
  }
  load_q35_attn(b, layers_[b]);
  const std::string p = "blk." + std::to_string(b) + ".nextn.";
  if (!load_qtensor(p + "eh_proj.weight", mtp_eh_))
    throw std::runtime_error("mtp: nextn.eh_proj must be a supported quantized tensor");
  mtp_enorm_ = upload_norm(tensor_f32(p + "enorm.weight"));
  mtp_hnorm_ = upload_norm(tensor_f32(p + "hnorm.weight"));
  mtp_shn_ = upload_norm(tensor_f32(p + "shared_head_norm.weight"));
  mtp_cat_ = cuda::Buffer(static_cast<size_t>(2) * H * 4);
  mtp_emb_ = cuda::Buffer(static_cast<size_t>(H) * 4);
  mtp_h_ = cuda::Buffer(static_cast<size_t>(H) * 4);
  mtp_hrow_ = cuda::Buffer(static_cast<size_t>(H) * 4);
  mtp_pin_.resize(static_cast<size_t>(std::max(vocab_, 1)) * 4 * 4);
  // The draft KV cache of the MTP layer starts as garbage; it is filled by
  // mtp_prefill_ctx() for prompt positions, and positions past the prompt are written
  // before they are ever read.
  //
  // NOTE: upload the *allocated* buffer size, not a f32-sized zero vector - with
  // --kv-type q8_0/q4_0 the KV buffer is quantized (smaller), and a float-sized
  // upload used to throw "Buffer::upload exceeds allocation" (MTP could not be
  // enabled together with a quantized KV cache).
  {
        const size_t kb = layers_[b].kc.size();  // already kv-type dependent
    std::vector<unsigned char> z(kb, 0);
    if (kb > 0 && layers_[b].vc.size() == kb) {
      layers_[b].kc.upload(z.data(), kb);
      layers_[b].vc.upload(z.data(), kb);
    } else {
      std::vector<float> zf(static_cast<size_t>(n_kv_) * max_seq_ * head_dim_, 0.f);
      layers_[b].kc.upload(zf.data(), zf.size() * 4);
      layers_[b].vc.upload(zf.data(), zf.size() * 4);
    }
  }
  mtp_ok_ = true;
  std::fprintf(stderr,
               "[mtp] block blk.%d loaded: eh_proj=%s %dx%d, %d experts on %s, kv %d tokens\n",
               b, mtp_eh_.type.c_str(), mtp_eh_.n_out, mtp_eh_.n_in,
               layers_[b].is_moe ? ir_.moe.n_expert : 0,
               mtp_gpu_experts_ > 0 ? "GPU" : (layers_[b].is_moe ? "CPU" : "n/a"),
               max_seq_);
}

// ---------------------------------------------------- recurrent-state slots ----
void DenseRunner::ensure_ssm_slots(int nslots) {
  if (nslots <= ssm_slots_) return;
  if (ir_.ssm.variant == "none") {
    ssm_slots_ = nslots;
    return;
  }
  std::vector<size_t> rec_off(static_cast<size_t>(ir_.n_layer), 0);
  std::vector<size_t> conv_off(static_cast<size_t>(ir_.n_layer), 0);
  size_t rec_total = 0, conv_total = 0;
  const size_t rec_layer = static_cast<size_t>(ir_.ssm.n_value_head) *
                           ir_.ssm.value_head_dim * ir_.ssm.key_head_dim * 4;
  const size_t conv_layer = static_cast<size_t>(ir_.ssm.conv_kernel - 1) * ir_.ssm.conv_dim * 4;
  for (int i = 0; i < ir_.n_layer; ++i) {
    if (!layers_[i].is_linear) continue;
    rec_off[static_cast<size_t>(i)] = rec_total;
    conv_off[static_cast<size_t>(i)] = conv_total;
    rec_total += rec_layer;
    conv_total += conv_layer;
  }
  if (rec_total == 0) {
    ssm_slots_ = nslots;
    return;
  }
  ssm_rec_off_ = std::move(rec_off);
  ssm_conv_off_ = std::move(conv_off);
  ssm_rec_stride_ = rec_total;
  ssm_conv_stride_ = conv_total;
  ssm_rec_slots_ = cuda::Buffer(rec_total * static_cast<size_t>(nslots));
  ssm_conv_slots_ = cuda::Buffer(conv_total * static_cast<size_t>(nslots));
  ssm_slots_ = nslots;
  std::fprintf(stderr, "[mtp] ssm rollback slots: %d x (rec %.1f MiB + conv %.1f MiB)\n",
               nslots, static_cast<double>(rec_total) / 1048576.0,
               static_cast<double>(conv_total) / 1048576.0);
}

void DenseRunner::ssm_save_conv(int layer, int slot) {
  if (slot < 0 || slot >= ssm_slots_ || ssm_conv_stride_ == 0) return;
  auto& L = layers_[layer];
  unsigned char* dst = static_cast<unsigned char*>(ssm_conv_slots_.data()) +
                       static_cast<size_t>(slot) * ssm_conv_stride_ +
                       ssm_conv_off_[static_cast<size_t>(layer)];
  cuda::copy_d2d_async(dst, L.q35.conv_state.data(), L.q35.conv_state.size());
}

void DenseRunner::ssm_save_rec(int layer, int slot) {
  if (slot < 0 || slot >= ssm_slots_ || ssm_rec_stride_ == 0) return;
  auto& L = layers_[layer];
  unsigned char* dst = static_cast<unsigned char*>(ssm_rec_slots_.data()) +
                       static_cast<size_t>(slot) * ssm_rec_stride_ +
                       ssm_rec_off_[static_cast<size_t>(layer)];
  cuda::copy_d2d_async(dst, L.q35.rec_state.data(), L.q35.rec_state.size());
}

// Put the recurrent state back to "after the first (slot+1) accepted tokens".
void DenseRunner::ssm_restore(int slot) {
  if (ssm_rec_stride_ == 0 || slot < 0 || slot >= ssm_slots_) return;
  for (int i = 0; i < ir_.n_layer; ++i) {
    if (!layers_[i].is_linear) continue;
    auto& L = layers_[i];
    const unsigned char* src_r = static_cast<const unsigned char*>(ssm_rec_slots_.data()) +
                                 static_cast<size_t>(slot) * ssm_rec_stride_ +
                                 ssm_rec_off_[static_cast<size_t>(i)];
    cuda::copy_d2d_async(L.q35.rec_state.data(), src_r, L.q35.rec_state.size());
    const unsigned char* src_c = static_cast<const unsigned char*>(ssm_conv_slots_.data()) +
                                 static_cast<size_t>(slot) * ssm_conv_stride_ +
                                 ssm_conv_off_[static_cast<size_t>(i)];
    cuda::copy_d2d_async(L.q35.conv_state.data(), src_c, L.q35.conv_state.size());
  }
}

// --------------------------------------------------------------- draft step ----
// Fill mtp_cat_ = concat(enorm(emb(tok)), hnorm(h_p)) for ONE token and project it
// through eh_proj.  The embedding is dequantised on the host (same lazy path the
// token embedding uses in forward_impl).
void DenseRunner::mtp_build_input(int tok) {
  if (mtp_emb_host_.size() < static_cast<size_t>(hidden_)) mtp_emb_host_.resize(hidden_);
  float* dst = mtp_emb_host_.data();
  if (!embed_raw_.empty()) {
    const int row_blocks = hidden_ / embed_be_;
    const unsigned char* row =
        embed_raw_.data() + static_cast<size_t>(tok) * row_blocks * embed_bb_;
    for (int b = 0; b < row_blocks; ++b)
      cpu::dequant_block(embed_type_, row + static_cast<size_t>(b) * embed_bb_,
                         dst + static_cast<size_t>(b) * embed_be_);
  } else {
    std::memcpy(dst, embed_host_.data() + static_cast<size_t>(tok) * hidden_,
                static_cast<size_t>(hidden_) * 4);
  }
  mtp_emb_.upload(dst, static_cast<size_t>(hidden_) * 4);
  const int H = hidden_;
  rms_raw(mtp_emb_.data(), mtp_enorm_.data(), mtp_cat_.data(), 1, H);
  rms_raw(mtp_h_.data(), mtp_hnorm_.data(),
          static_cast<float*>(mtp_cat_.data()) + H, 1, H);
}

// arg-max of the shared output head applied to mtp_h_ (already normalised by the
// caller into `hn_`).
int DenseRunner::head_argmax_mtp() {
  rms_raw(mtp_h_.data(), mtp_shn_.data(), hn_.data(), 1, hidden_);
  matmul(lm_head_qt_, 1, hn_, logits_);
  if (head_topk_ && k_topkpart_ && k_topkfinal_ && topk_nblk_ > 0) {
    int kk = 1;
    void* px = logits_.data();
    void* ppv = topk_pv_.data();
    void* ppi = topk_pi_.data();
    int nv = vocab_, nb = topk_nblk_;
    void* a1[] = {&px, &ppv, &ppi, &nv, &nb, &kk};
    cuda::launch(k_topkpart_, nb, 1, 1, 256, 1, 1, 0, a1);
    void* pov = topk_ov_.data();
    void* poi = topk_oi_.data();
    int nc = nb * kk;
    void* a2[] = {&ppv, &ppi, &nc, &kk, &poi, &pov};
    cuda::launch(k_topkfinal_, 1, 1, 1, 256, 1, 1, 0, a2);
    int idx = -1;
    topk_oi_.download(&idx, 4);   // blocking: also drains the head GEMV
    return idx;
  }
  logits_.download(mtp_pin_.data(), static_cast<size_t>(vocab_) * 4);
  const float* p = static_cast<const float*>(mtp_pin_.data());
  int best = 0;
  for (int i = 1; i < vocab_; ++i)
    if (p[i] > p[best]) best = i;
  return best;
}

int DenseRunner::mtp_draft(int tok, int pos, int row) {
  if (!mtp_ok_) return -1;
  if (row >= 0) {
    // h_p = residual stream of the target model at that row (x_ still holds it)
    cuda::copy_d2d_async(mtp_h_.data(),
                         static_cast<float*>(x_.data()) + static_cast<size_t>(row) * hidden_,
                         static_cast<size_t>(hidden_) * 4);
  }
  mtp_build_input(tok);
  // u = eh_proj(concat(...)) -> x_ [1, H]
  matmul(mtp_eh_, 1, mtp_cat_, x_);
  set_position(pos);
  // one MTP transformer layer at `pos` (own KV cache, own attention window)
  run_layer_body(mtp_block_, 1, pos);
  // y is now in x_; keep it as the chained hidden state
  cuda::copy_d2d_async(mtp_h_.data(), x_.data(), static_cast<size_t>(hidden_) * 4);
  return head_argmax_mtp();
}

// ------------------------------------------------------------- verification ----
void DenseRunner::verify_rows_forward(const std::vector<int>& tokens, int pos) {
  verify_multi_head_ = true;
  spec_snapshot_ = true;
  try {
    forward(tokens, pos, /*last_row_only=*/false);
  } catch (...) {
    verify_multi_head_ = false;
    spec_snapshot_ = false;
    throw;
  }
  verify_multi_head_ = false;
  spec_snapshot_ = false;
}

// Fill the MTP layer's KV cache for the prompt positions in one batched pass, so the
// draft attention window is not full of zeros.  `toks` holds the n prompt ids plus the
// pending next token (toks[n]); x_ must still hold the n residual rows of the prompt.
void DenseRunner::mtp_prefill_ctx(const std::vector<int>& toks, int n) {
  if (!mtp_ok_ || n <= 0) return;
  const int H = hidden_;
  // 1) residuals -> host
  std::vector<float> h_all(static_cast<size_t>(n) * H);
  if (static_cast<int>(mtp_hrow_.size()) < n * H * 4) mtp_hrow_ = cuda::Buffer(static_cast<size_t>(n) * H * 4);
  cuda::copy_d2d_async(mtp_hrow_.data(), x_.data(), static_cast<size_t>(n) * H * 4);
  mtp_hrow_.download(h_all.data(), h_all.size() * 4);
  // 2) host RMSNorm of both halves (eps must match rms_raw)
  std::vector<float> cat(static_cast<size_t>(n) * 2 * H);
  std::vector<float> emb(static_cast<size_t>(H));
  for (int i = 0; i < n; ++i) {
    const int tok = toks[static_cast<size_t>(i) + 1];
    float* dst = emb.data();
    if (!embed_raw_.empty()) {
      const int row_blocks = H / embed_be_;
      const unsigned char* row =
          embed_raw_.data() + static_cast<size_t>(tok) * row_blocks * embed_bb_;
      for (int b = 0; b < row_blocks; ++b)
        cpu::dequant_block(embed_type_, row + static_cast<size_t>(b) * embed_bb_,
                           dst + static_cast<size_t>(b) * embed_be_);
    } else {
      std::memcpy(dst, embed_host_.data() + static_cast<size_t>(tok) * H,
                  static_cast<size_t>(H) * 4);
    }
    float* c = cat.data() + static_cast<size_t>(i) * 2 * H;
    auto rms = [&](const float* src, const cuda::Buffer& wbuf, float* out) {
      std::vector<float> w(static_cast<size_t>(H));
      wbuf.download(w.data(), w.size() * 4);
      double ss = 0;
      for (int j = 0; j < H; ++j) ss += static_cast<double>(src[j]) * src[j];
      const float scale = static_cast<float>(1.0 / std::sqrt(ss / H + eps_));
      for (int j = 0; j < H; ++j) out[j] = src[j] * scale * w[static_cast<size_t>(j)];
    };
    rms(emb.data(), mtp_enorm_, c);
    rms(h_all.data() + static_cast<size_t>(i) * H, mtp_hnorm_, c + H);
  }
  // 3) batched eh_proj then one MTP layer pass over the prompt
  cuda::Buffer catb(static_cast<size_t>(n) * 2 * H * 4);
  catb.upload(cat.data(), cat.size() * 4);
  if (x_.size() < static_cast<size_t>(n) * H * 4) x_ = cuda::Buffer(static_cast<size_t>(n) * H * 4);
  matmul(mtp_eh_, n, catb, x_);
  set_position(0);
  run_layer_body(mtp_block_, n, 0);
}

// ------------------------------------------------------------- spec decode ----
std::vector<int> DenseRunner::spec_generate(const std::vector<int>& prompt, int n_gen,
                                            SpecStats* st) {
  SpecStats dummy;
  if (!st) st = &dummy;
  *st = SpecStats{};
  if (!mtp_ok_ || n_gen <= 0) return {};
  auto& cfg = engine::config::Config::get();
  int k = static_cast<int>(cfg.get_int("mtp_k"));
  if (k < 1) k = 1;
  if (k > 3) k = 3;
  st->k = k;
  const double t_start = now_ms();
  // ---- prefill ----
  TopKResult tk;
  std::vector<float> logits = forward_topk(prompt, 0, 5, &tk);
  std::vector<float> last;
  int pending;
  if (!logits.empty()) {
    last.assign(logits.end() - vocab_, logits.end());
    pending = static_cast<int>(std::max_element(last.begin(), last.end()) - last.begin());
  } else {
    pending = tk.idx.empty() ? -1 : tk.idx[0];
  }
  if (pending < 0) return {};
  const int prompt_len = static_cast<int>(prompt.size());
  int pos = prompt_len;
  int gen_at_first_round = -1;
  double t_first_round_end = 0;
  std::vector<int> gen;
  // h of the row that produced `pending` is the last prompt row in x_
  cuda::copy_d2d_async(mtp_h_.data(),
                       static_cast<float*>(x_.data()) +
                           static_cast<size_t>(prompt_len - 1) * hidden_,
                       static_cast<size_t>(hidden_) * 4);
  // populate the MTP attention history for the prompt (one batched MTP pass)
  if (cfg.get_bool("mtp_ctx_prefill")) {
    std::vector<int> toks(prompt);
    toks.push_back(pending);
    mtp_prefill_ctx(toks, prompt_len);
  }
  ensure_ssm_slots(k);
  st->prefill_ms = now_ms() - t_start;
  std::vector<int> drafts(static_cast<size_t>(k), -1);
  std::vector<int> vt;
  const int draft_alloc = k;
  while (static_cast<int>(gen.size()) < n_gen) {
    // ---- draft k tokens ----
    const double d0 = now_ms();
    int tok = pending;
    int dpos = pos;
    for (int i = 0; i < k; ++i) {
      drafts[static_cast<size_t>(i)] = mtp_draft(tok, dpos, -1);
      tok = drafts[static_cast<size_t>(i)];
      ++dpos;
      if (tok < 0) break;
    }
    st->draft_ms += now_ms() - d0;
    // ---- verify [pending, d_1..d_k] in one batch ----
    vt.clear();
    vt.push_back(pending);
    for (int i = 0; i < k; ++i) vt.push_back(drafts[static_cast<size_t>(i)]);
    const double v0 = now_ms();
    verify_rows_forward(vt, pos);
    st->verify_ms += now_ms() - v0;
    ++st->verify_fwd;
    ++st->rounds;
    st->drafted += k;
    // ---- longest accepted prefix ----
    int m = 0;
    while (m < k && verify_argmax_[static_cast<size_t>(m)] == drafts[static_cast<size_t>(m)])
      ++m;
    st->accepted += m;
    gen.push_back(pending);
    for (int i = 0; i < m; ++i) gen.push_back(drafts[static_cast<size_t>(i)]);
    const int bonus = verify_argmax_[static_cast<size_t>(m)];
    if (m < k) ssm_restore(m);
    // residual row `m` of the verify batch is the hidden state that produced `bonus`
    cuda::copy_d2d_async(mtp_h_.data(),
                         static_cast<float*>(x_.data()) + static_cast<size_t>(m) * hidden_,
                         static_cast<size_t>(hidden_) * 4);
    pending = bonus;
    pos += m + 1;
    if (st->rounds == 1) {
      gen_at_first_round = static_cast<int>(gen.size());
      t_first_round_end = now_ms();
    }
    if (static_cast<int>(gen.size()) >= n_gen) break;
    if (pos + draft_alloc + 2 >= max_seq_) break;   // out of context
  }
  if (static_cast<int>(gen.size()) > n_gen) gen.resize(static_cast<size_t>(n_gen));
  st->total_ms = now_ms() - t_start;
  if (gen_at_first_round >= 0) {
    st->decode_tokens = static_cast<int>(gen.size()) - gen_at_first_round;
    st->decode_ms = now_ms() - t_first_round_end;
  }
  return gen;
}

// ---------------------------------------- incremental spec decode (serve) ----
bool DenseRunner::spec_begin(const std::vector<int>& prompt_ids, int first_token,
                             int last_row) {
  spec_active_ = false;
  spec_pending_ = -1;
  spec_pos_ = 0;
  if (!mtp_ok_ || prompt_ids.empty() || first_token < 0 || last_row < 0) return false;
  const int prompt_len = static_cast<int>(prompt_ids.size());
  if (last_row >= prompt_len) return false;
  if (prompt_len + 8 >= max_seq_) return false;
  // h of the row that produced `first_token`
  cuda::copy_d2d_async(mtp_h_.data(),
                       static_cast<float*>(x_.data()) +
                           static_cast<size_t>(last_row) * hidden_,
                       static_cast<size_t>(hidden_) * 4);
  spec_pending_ = first_token;
  spec_pos_ = prompt_len;
  auto& cfg = engine::config::Config::get();
  if (cfg.get_bool("mtp_ctx_prefill")) {
    std::vector<int> toks(prompt_ids);
    toks.push_back(first_token);
    mtp_prefill_ctx(toks, prompt_len);
  }
  ensure_ssm_slots(3);
  spec_active_ = true;
  return true;
}

bool DenseRunner::spec_round(int k, std::vector<int>* emit) {
  if (!spec_active_ || !mtp_ok_ || !emit) return false;
  if (k < 1) k = 1;
  if (k > 3) k = 3;
  if (spec_pos_ + k + 2 >= max_seq_) {
    spec_active_ = false;
    return false;
  }
  std::vector<int> drafts(static_cast<size_t>(k), -1);
  int tok = spec_pending_;
  int dpos = spec_pos_;
  for (int i = 0; i < k; ++i) {
    drafts[static_cast<size_t>(i)] = mtp_draft(tok, dpos, -1);
    tok = drafts[static_cast<size_t>(i)];
    ++dpos;
    if (tok < 0) break;
  }
  std::vector<int> vt;
  vt.reserve(static_cast<size_t>(k) + 1);
  vt.push_back(spec_pending_);
  for (int i = 0; i < k; ++i) vt.push_back(drafts[static_cast<size_t>(i)]);
  verify_rows_forward(vt, spec_pos_);
  int m = 0;
  while (m < k && verify_argmax_[static_cast<size_t>(m)] == drafts[static_cast<size_t>(m)])
    ++m;
  emit->clear();
  emit->push_back(spec_pending_);
  for (int i = 0; i < m; ++i) emit->push_back(drafts[static_cast<size_t>(i)]);
  const int bonus = verify_argmax_[static_cast<size_t>(m)];
  if (m < k) ssm_restore(m);
  // residual row `m` is the hidden state that produced `bonus`
  cuda::copy_d2d_async(mtp_h_.data(),
                       static_cast<float*>(x_.data()) + static_cast<size_t>(m) * hidden_,
                       static_cast<size_t>(hidden_) * 4);
  spec_pending_ = bonus;
  spec_pos_ += m + 1;
  if (bonus < 0) {
    spec_active_ = false;
    return false;
  }
  return true;
}

// ------------------------------------------------------- draft accuracy (P0) ----
void DenseRunner::mtp_accuracy(const std::vector<int>& prompt, int n_gen, int verbose) {
  if (!mtp_ok_) {
    std::fprintf(stderr, "[mtp-acc] MTP is not available for this model\n");
    return;
  }
  auto& cfg = engine::config::Config::get();
  TopKResult tk;
  std::vector<float> logits = forward_topk(prompt, 0, 5, &tk);
  std::vector<float> last;
  int pending;
  if (!logits.empty()) {
    last.assign(logits.end() - vocab_, logits.end());
    pending = static_cast<int>(std::max_element(last.begin(), last.end()) - last.begin());
  } else {
    pending = tk.idx[0];
  }
  int pos = static_cast<int>(prompt.size());
  // h of the row that produced `pending` lives in x_ (last prefill row): save it
  // before the MTP context pass rewrites x_.
  cuda::copy_d2d_async(mtp_h_.data(),
                       static_cast<float*>(x_.data()) +
                           static_cast<size_t>(pos - 1) * hidden_,
                       static_cast<size_t>(hidden_) * 4);
  if (cfg.get_bool("mtp_ctx_prefill")) {
    std::vector<int> toks(prompt);
    toks.push_back(pending);
    mtp_prefill_ctx(toks, static_cast<int>(prompt.size()));
  }
  int hits = 0, total = 0, hits2 = 0;
  std::vector<int> gen;
  gen.push_back(pending);
  bool h_ready = true;   // mtp_h_ already holds h for `pending`
  for (int step = 0; step < n_gen; ++step) {
    const int d = mtp_draft(pending, pos, h_ready ? -1 : 0);
    h_ready = false;
    TopKResult tk2;
    std::vector<float> out = forward_topk({pending}, pos, 5, &tk2);
    int actual;
    if (!out.empty()) {
      std::vector<float> row(out.end() - vocab_, out.end());
      actual = static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
    } else {
      actual = tk2.idx[0];
    }
    ++total;
    if (d == actual) ++hits;
    if (verbose)
      std::printf("[mtp-acc] step %2d pos %3d pending %6d draft %6d target %6d %s\n", step,
                  pos, pending, d, actual, d == actual ? "HIT" : "miss");
    gen.push_back(actual);
    pending = actual;
    ++pos;
  }
  std::printf("mtp_draft_acc: %d/%d = %.3f  (prompt %zu, %d steps)\n", hits, total,
              total ? static_cast<double>(hits) / total : 0.0, prompt.size(), total);
  (void)hits2;
  std::printf("mtp_gen:");
  for (int t : gen) std::printf(" %d", t);
  std::printf("\n");
}

}  // namespace runner
