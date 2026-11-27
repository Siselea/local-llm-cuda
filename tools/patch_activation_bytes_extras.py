#!/usr/bin/env python3
"""Make activation_bytes() account for the SSM/MLA/MoE per-row working buffers.

It used to count only the generic x/hn/q/k/v/attn/gate/up/act + one logits row, so on a
Qwen3.5-9B (hybrid SSM) it missed ~221 KB/row of q35_* buffers and on a Coder (MLA) the
mla_* ones.  With the logits row cap that estimate decides how far the prefill chunk can
grow, so an undercount lets the chunk grow past the real VRAM (observed: rows 4096 left
the serve post-load guard with 930 MiB free) and also makes the placement decision
optimistic.  This patch adds every per-row buffer that the constructor allocates.
"""
p = 'src/runner.cpp'
s = open(p, encoding='utf-8').read()

old = '''  per_row += static_cast<size_t>(inter_) * 4 * 3;
  // Logits: a last_row_only prefill head computes ONE row, so only the reserved rows
  // (default 1) are part of the footprint.  Keeping this in sync with the allocation
  // above is what lets the VRAM-derived chunk growth pick a much larger chunk.
  const int logits_rows ='''
new = '''  per_row += static_cast<size_t>(inter_) * 4 * 3;
  // Hybrid SSM (qwen3.5): the mixer keeps its own per-token working set -- two conv
  // buffers, z/o/v, q/k, four per-value-head vectors, the qfull/gate/attn staging.
  // Missing these made the chunk-growth estimate ~221 KB/row optimistic on the 9B.
  if (ir_.ssm.variant != "none") {
    const auto& S = ir_.ssm;
    per_row += static_cast<size_t>(S.conv_dim) * 4 * 2;              // mixed, conv
    per_row += static_cast<size_t>(S.value_dim) * 4 * 3;             // z, o, v
    per_row += static_cast<size_t>(S.key_dim) * 4 * 2;               // q, k
    per_row += static_cast<size_t>(S.n_value_head) * 4 * 4;          // beta, beta_b, a, g
    per_row += static_cast<size_t>(n_head_ * head_dim_ * 2) * 4;     // qfull
    per_row += static_cast<size_t>(n_head_ * head_dim_) * 4 * 2;     // gate, attn_q
    per_row += static_cast<size_t>(n_kv_ * head_dim_) * 4 * 2;       // attn_k, attn_v
  }
  // MLA projections (DeepSeek-V2 style): the absorbed q/kv staging buffers.
  if (ir_.attn.variant == "mla") {
    const size_t qk = static_cast<size_t>(ir_.attn.key_length);
    const size_t v_dim = static_cast<size_t>(ir_.attn.value_length);
    const size_t kv_lora = static_cast<size_t>(ir_.attn.kv_lora_rank);
    const size_t qk_rope = static_cast<size_t>(ir_.attn.rope_dim);
    const size_t nh = static_cast<size_t>(n_head_);
    per_row += nh * qk * 4 * 3;                                      // mq, mk, mla_qraw
    per_row += (kv_lora + qk_rope) * 4;                              // mkva
    per_row += kv_lora * 4 * 2;                                      // mckv, mckvn
    per_row += qk_rope * 4;                                          // mkpe
    per_row += nh * (qk - qk_rope) * 4;                              // mkn
    per_row += nh * v_dim * 4 * 2;                                   // mv, mout
    per_row += nh * (qk - qk_rope + v_dim) * 4;                      // mla_kvraw
  }
  // MoE router logits and the gather staging buffers.
  if (ir_.moe.n_expert > 0)
    per_row += static_cast<size_t>(ir_.moe.n_expert) * 4 + static_cast<size_t>(hidden_) * 4 * 2;
  // Logits: a last_row_only prefill head computes ONE row, so only the reserved rows
  // (default 1) are part of the footprint.  Keeping this in sync with the allocation
  // above is what lets the VRAM-derived chunk growth pick a much larger chunk.
  const int logits_rows ='''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('activation_bytes: SSM/MLA/MoE per-row buffers added')
