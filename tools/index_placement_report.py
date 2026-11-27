#!/usr/bin/env python3
"""Add the placement report to the reports index (LF-safe, idempotent)."""
p = 'docs/reports/INDEX.md'
s = open(p, encoding='utf-8').read()
row = ('| [PLACEMENT-VRAM-MARGIN.md](PLACEMENT-VRAM-MARGIN.md) | 自动分层显存余量修复'
       '（PLACEMENT-VRAM-MARGIN） | 自动分层漏算"账本看不见"的 1.1-2.1 GiB 开销'
       '（驱动/cuBLAS/分配器）→ 装载后可能 0 MiB free、驱动换页（3 tok/s）：新增可标定 '
       'vram_extra_reserve_mb=1024、逐层自适应检查纳入该预留、装载后不足改为明确 ERROR；'
       '3×generality + genreg + smoke 全绿。 |\n')
if 'PLACEMENT-VRAM-MARGIN.md' in s:
    print('already indexed')
else:
    anchor = '| [PREFILL-CLIFF-REPORT.md](PREFILL-CLIFF-REPORT.md) |'
    assert s.count(anchor) == 1, s.count(anchor)
    s = s.replace(anchor, row + anchor)
    open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print('index updated')
