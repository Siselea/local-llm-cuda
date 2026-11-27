"""CLI: inspect a GGUF/safetensors model, build IR, validate it, and plan placement.

Usage:
  python -m pyref.cli info  <model.gguf> [--ctx 8192] [--vram-gib 12]
  python -m pyref.cli ir    <model.gguf> [--out ir.json]
  python -m pyref.cli plan  <model.gguf> [--ctx 8192] [--vram-gib 12]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .gguf_parse import read_gguf, GGUFError
from .ir import build_ir_from_gguf, validate_ir, ModelIR
from .planner import plan


def _load(path: str) -> ModelIR:
    t0 = time.perf_counter()
    if os.path.isdir(path):
        from .st_parse import build_ir_from_hf
        ir = build_ir_from_hf(path)
    else:
        g = read_gguf(path)
        ir = build_ir_from_gguf(g)
    dt = (time.perf_counter() - t0) * 1e3
    ir.warnings.append(f"parsed in {dt:.1f} ms")
    return ir


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def cmd_info(args) -> int:
    ir = _load(args.model)
    errs = validate_ir(ir)
    print(f"model      : {ir.path}")
    print(f"arch       : {ir.arch}   container={ir.container}  file={_human(ir.file_size)}")
    print(f"layers     : {ir.n_layer}   hidden={ir.hidden}  ffn={ir.ffn}  vocab={ir.vocab}")
    print(f"attn       : variant={ir.attn.variant} heads={ir.attn.n_head}/{ir.attn.n_head_kv} "
          f"klen={ir.attn.key_length} vlen={ir.attn.value_length} rope_base={ir.attn.rope_freq_base:.0f}")
    if ir.moe.n_expert:
        print(f"moe        : experts={ir.moe.n_expert} used={ir.moe.n_expert_used} "
              f"shared={ir.moe.n_shared} inter={ir.moe.expert_ffn_length}")
    print(f"tokenizer  : {ir.tokenizer.model} pre={ir.tokenizer.pre!r} "
          f"vocab={ir.tokenizer.n_vocab} chat_template={ir.tokenizer.chat_template}")
    print(f"bytes      : total={_human(ir.total_bytes())} fixed={_human(ir.fixed_bytes())} "
          f"experts={_human(ir.expert_bytes())}")
    print(f"quant mix  : " + ", ".join(
        f"{k}={_human(v)}" for k, v in sorted(ir.quant_mix.items(), key=lambda kv: -kv[1])[:6]))
    roles: dict[str, int] = {}
    for t in ir.tensors:
        roles[t.role.value] = roles.get(t.role.value, 0) + 1
    print("roles      : " + ", ".join(f"{k}:{v}" for k, v in sorted(roles.items())))
    if errs:
        print(f"VALIDATION : FAIL ({len(errs)} errors)")
        for e in errs[:10]:
            print(f"  - {e}")
    else:
        print("VALIDATION : PASS")
    for w in ir.warnings[:6]:
        print(f"warn       : {w}")
    return 0 if not errs else 2


def cmd_ir(args) -> int:
    ir = _load(args.model)
    errs = validate_ir(ir)
    if errs and not args.force:
        print("refusing to dump IR: validation failed", file=sys.stderr)
        for e in errs[:10]:
            print(f"  - {e}", file=sys.stderr)
        return 2
    doc = ir.to_json(include_tensors=not args.no_tensors)
    out = json.dumps(doc, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.out} ({len(out)} bytes)")
    else:
        print(out)
    return 0


def _host_memory() -> "object":
    """Best-effort host RAM probe (no third-party dependency).  The parity gate always
    passes --host-gib, so this path only matters for ad-hoc use."""
    from .placement import MemoryInfo
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return MemoryInfo(True, int(st.ullTotalPhys), int(st.ullAvailPhys))
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            return MemoryInfo(True, int(total), int(avail))
    except Exception:
        pass
    return MemoryInfo(False, 0, 0)


def _kv_type(value: str) -> int:
    v = str(value).strip().lower()
    if v in ("f32", "0", "auto", ""):
        return 0
    if v in ("q8_0", "q8", "1"):
        return 1
    if v in ("q4_0", "q4", "2"):
        return 2
    raise SystemExit(f"--kv-type expects f32|q8_0|q4_0 (or 0|1|2), got {value}")


def cmd_plan(args) -> int:
    from . import placement as pl

    ir = _load(args.model)
    kv_type = _kv_type(args.kv_type)
    vram = pl.MemoryInfo(True, int(args.vram_gib * (1 << 30)), int(args.vram_gib * (1 << 30)))
    host = (_host_memory() if args.host_gib is None
            else pl.MemoryInfo(True, int(args.host_gib * (1 << 30)),
                               int(args.host_gib * (1 << 30))))
    mib = 1 << 20
    vram_reserve = int((args.vram_margin_mb + args.workspace_mb) * mib)
    workspace = int(args.workspace_mb * mib)
    host_reserve = (int(args.host_margin_mb * mib) if args.host_margin_mb
                    else pl.default_host_reserve(host.total_bytes))
    p = plan(ir, ctx=args.ctx, kv_type=kv_type,
             vram_total=int(args.vram_gib * (1 << 30)))
    have = vram.valid
    d = None
    fit = None
    if have:
        d = pl.decide(ir, p.ctx, kv_type, vram, host, vram_reserve, host_reserve, workspace)
        fit = pl.estimate_ctx(ir, kv_type, vram.free_bytes, vram_reserve, args.row_cap)

    print(f"model        : {ir.path}")
    print(f"ctx={p.ctx} kv_type={p.kv_name} vram={args.vram_gib} GiB")
    print(f"fixed        : {_human(p.fixed_bytes)}")
    print(f"kv cache     : {_human(p.kv_bytes)}")
    print(f"workspace    : {_human(p.workspace_bytes)}")
    print(f"expert budget: {_human(p.expert_budget)} / total experts {_human(p.expert_total)}")
    if ir.moe.n_expert:
        print(f"slots/layer  : {p.slots_per_layer} (uniform advisory)")
        print(f"layers       : {p.layers_on_gpu} on GPU / {p.layers_on_cpu} on CPU"
              f"  (fully_resident={p.fully_resident})")
        for layer, where in p.layer_placement.items():
            print(f"layer {layer:<6} : {where}")
    else:
        need = p.fixed_bytes + p.kv_bytes + p.workspace_bytes
        print(f"dense model  : no experts; resident need = {_human(need)} "
              f"(vram usable = {_human(p.vram_total - p.vram_reserve)})")
        print(f"fits         : {need <= p.vram_total - p.vram_reserve}")
    print(f"projection   : cpu-lane {p.proj_decode_cpu_ms:.1f} ms/token, "
          f"pcie {p.proj_decode_pcie_ms:.1f} ms/token")
    for n in p.notes:
        print(f"note         : {n}")
    if have:
        print(f"[placement] decision: {pl.kind_name(d.kind)}")
        print(f"[placement] reason  : {d.reason}")
        for n in d.notes:
            print(f"[placement] note    : {n}")
        print(f"[suggested]  ctx {fit.max_ctx} ({fit.kv_name}, {fit.kv_per_token} B/token, "
              f"KV {_human(fit.kv_at_max)} at max)")

    if args.json:
        doc = {"model": ir.path, **p.to_json()}
        if have:
            doc["placement"] = {
                "kind": pl.kind_name(d.kind),
                "n_moe_layers": d.n_moe_layers,
                "gpu_moe_layers": d.gpu_moe_layers,
                "fits_vram": d.fits_vram,
                "fits_all": d.fits_all,
                "dense_offload_unsupported": d.dense_offload_unsupported,
                "model_bytes": d.model_bytes,
                "fixed_bytes": d.fixed_bytes,
                "expert_bytes": d.expert_bytes,
                "kv_bytes": d.kv_bytes,
                "workspace_bytes": d.workspace_bytes,
                "vram_total": d.vram_total,
                "vram_free": d.vram_free,
                "vram_reserve": d.vram_reserve,
                "vram_usable": d.vram_usable,
                "vram_need": d.vram_need,
                "vram_left": d.vram_left,
                "host_total": d.host_total,
                "host_free": d.host_free,
                "host_reserve": d.host_reserve,
                "host_usable": d.host_usable,
                "host_need": d.host_need,
                "reason": d.reason,
                "notes": list(d.notes),
            }
            doc["suggested_ctx"] = {
                "kv_type": fit.kv_type,
                "kv_name": fit.kv_name,
                "kv_per_token": fit.kv_per_token,
                "max_ctx": fit.max_ctx,
                "kv_at_max": fit.kv_at_max,
                "row_cap": args.row_cap,
            }
        else:
            doc["placement"] = None
            doc["suggested_ctx"] = None
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pyref")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("info"); a.add_argument("model"); a.set_defaults(fn=cmd_info)
    b = sub.add_parser("ir")
    b.add_argument("model"); b.add_argument("--out"); b.add_argument("--no-tensors", action="store_true")
    b.add_argument("--force", action="store_true"); b.set_defaults(fn=cmd_ir)
    c = sub.add_parser("plan")
    c.add_argument("model"); c.add_argument("--ctx", type=int, default=8192)
    c.add_argument("--kv-type", default="f32")
    c.add_argument("--vram-gib", type=float, default=12)
    c.add_argument("--host-gib", type=float, default=None)
    c.add_argument("--row-cap", type=int, default=512)
    c.add_argument("--vram-margin-mb", type=int, default=1200)
    c.add_argument("--workspace-mb", type=int, default=768)
    c.add_argument("--host-margin-mb", type=int, default=0)
    c.add_argument("--json"); c.set_defaults(fn=cmd_plan)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
