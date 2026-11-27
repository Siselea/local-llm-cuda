#!/usr/bin/env python3
"""Set vram_extra_reserve_mb default to the measured tuning-curve optimum (1024 MiB)."""
p = 'src/config.cpp'
s = open(p, encoding='utf-8').read()
old = ('    {"vram_extra_reserve_mb", "ENGINE_VRAM_EXTRA_RESERVE_MB", "vram-extra-reserve", true,\n'
       '     false, Type::Int, "2048", 0, 1 << 20,\n')
new = ('    {"vram_extra_reserve_mb", "ENGINE_VRAM_EXTRA_RESERVE_MB", "vram-extra-reserve", true,\n'
       '     false, Type::Int, "1024", 0, 1 << 20,\n')
assert s.count(old) == 1
s = s.replace(old, new)

old2 = ('"(measured on this machine as [vram] post-load used - alloc_sum: 1.25 GiB on the 9B, "\n'
        '     "2.1-2.8 GiB on the Coder MoE; without it an auto-placed MoE model can load to 0 MiB "\n'
        '     "free and the driver starts paging, 3 tok/s instead of 25).  0 restores the old "\n'
        '     "optimistic behaviour"},')
new2 = ('"(measured on this machine as [vram] post-load used - alloc_sum: 1.13-1.25 GiB on the "\n'
        '     "9B/35B, 2.1 GiB on the Coder MoE).  Tuning curve on an exclusive GPU: reserve 1024 "\n'
        '     "keeps 2.1-2.4 GiB free after load (margin 1200) at a decode cost of only 0.6-2.4%, "\n'
        '     "while 2048 costs 6.5-9% (every MiB of reserve can drop a whole MoE layer).  Without "\n'
        '     "it an auto-placed MoE model can load to 0 MiB free and the driver pages: 3 tok/s "\n'
        '     "instead of 25.  0 restores the old optimistic behaviour"},')
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('default -> 1024 with calibration documented')
