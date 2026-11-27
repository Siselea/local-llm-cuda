#!/usr/bin/env python3
"""Clean up the fused-gemm prototype harness (remove dead scaffolding, fix the trailer)."""
p = 'tests/fused_gemm_proto.h'
s = open(p, encoding='utf-8').read()

old = '''  // ---- baseline: dequant into the fp16 slab, then cuBLAS ----
  void* dq_args[] = {(void*)&w_dev, (void*)&slab_dev, (void*)&nblk_total_ptr_dummy_never};
  (void)dq_args;
  // build args explicitly to keep the pointer values stable
  void* dq_args_real[3];
  void* w_ptr = w_dev.data();
  void* slab_ptr = slab_dev.data();
  long long nb = (long long)nblk_total;
  dq_args_real[0] = &w_ptr;
  dq_args_real[1] = &slab_ptr;
  dq_args_real[2] = &nb;
  const double dq_ms =
      time_ms(dq_fn, (int)nblk_total, 1, 1, 128, 1, 1, 0, dq_args_real, 5);

  void* x_ptr = x_dev.data();
  void* out_ref_ptr = out_ref_dev.data();
  int m = s.M, n = s.N, k = s.K;'''
new = '''  // ---- baseline: dequant into the fp16 slab, then cuBLAS ----
  void* dq_args_real[3];
  void* w_ptr = w_dev.data();
  void* slab_ptr = slab_dev.data();
  long long nb = (long long)nblk_total;
  dq_args_real[0] = &w_ptr;
  dq_args_real[1] = &slab_ptr;
  dq_args_real[2] = &nb;
  const double dq_ms = time_ms(dq_fn, (int)nblk_total, 1, 1, 128, 1, 1, 0, dq_args_real, 5);

  int m = s.M, n = s.N, k = s.K;'''
assert s.count(old) == 1
s = s.replace(old, new)

old2 = '''  (void)x_ptr;
  (void)out_ref_ptr;

'''
assert s.count(old2) == 1
s = s.replace(old2, '')

old3 = '''  for (const auto& s : shapes) bench_one(s, dq_fn, fused_fn);
  std::printf(
      "[fused-gemm] traffic: baseline reads quant %zu MB + writes/reads fp16 slab %zu MB; "
      "fused reads quant only\\n",
      (size_t)0, (size_t)0);
}'''
new3 = '''  for (const auto& s : shapes) bench_one(s, dq_fn, fused_fn);
  std::printf(
      "[fused-gemm] traffic per matrix: baseline reads quantised weights once, writes and "
      "re-reads the fp16 slab (3x the quantised bytes); fused reads the quantised weights "
      "only\\n");
}'''
assert s.count(old3) == 1
s = s.replace(old3, new3)

# unused x_dev_buf copy
old4 = '''  std::vector<float> out_ref((size_t)s.M * s.N);
  std::vector<unsigned short> x_dev_buf((size_t)s.M * s.K);
'''
assert s.count(old4) == 1
s = s.replace(old4, '  std::vector<float> out_ref((size_t)s.M * s.N);\n')

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('prototype harness cleaned')
