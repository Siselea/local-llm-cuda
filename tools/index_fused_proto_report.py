#!/usr/bin/env python3
"""Index the fused-GEMM prototype report and update the plan doc's §3 expectation."""
import re

p = 'docs/reports/INDEX.md'
s = open(p, encoding='utf-8').read()
row = ('| [FUSED-GEMM-PROTO.md](FUSED-GEMM-PROTO.md) | fused dequant+GEMM 最小原型（负结果 + 修正判断） | '
       '单内核 fused（shared dequant + mma.m16n8k16）数值正确（rel_l2≤7e-6）但 naive 版慢 50×；'
       '实测 dequant 是计算瓶颈（115 GB/s）而非流量，修正"tail 190→85 ms"为"~190→140-150 ms"，'
       '建议先做层主序 + 反量化提速。 |\n')
if 'FUSED-GEMM-PROTO.md' not in s:
    anchor = '| [LAYER-MAJOR-PREFILL-PLAN.md](LAYER-MAJOR-PREFILL-PLAN.md) |'
    assert s.count(anchor) == 1
    s = s.replace(anchor, row + anchor)
    open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print('index updated')

# amend the plan doc: the §3.3 expectation was measured and needs the corrected number
p2 = 'docs/reports/LAYER-MAJOR-PREFILL-PLAN.md'
s2 = open(p2, encoding='utf-8').read()
old = ('- **tail**：52-token prefill 从 ~190 ms → 预期 **~80–90 ms**（剩下的是 conv/attention/head 等其它固定项）；')
assert s2.count(old) == 1
new = ('- **tail**：52-token prefill 从 ~190 ms → 预期 **~140–150 ms**。⚠ 2026-11-21 的原型实测修正了'
       '原来的"~85 ms"：`FUSED-GEMM-PROTO.md` 证明 dequant **计算**本身（~95 ms）不是流量、fused 也省不掉，'
       'fused 只去掉镜像写读（~74 ms 的一部分），naive 版实测慢 50×；要拿到这 ~1.3× 需要调优版内核；')
assert s2.count(old) == 1
s2 = s2.replace(old, new)
open(p2, 'w', encoding='utf-8', newline='\n').write(s2)
print('plan doc corrected')
