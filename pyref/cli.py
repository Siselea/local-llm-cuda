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


def cmd_plan(args) -> int:
    ir = _load(args.model)
    p = plan(ir, ctx=args.ctx, kv_bpe=args.kv_bpe,
             vram_total=int(args.vram_gib * (1 << 30)))
    print(f"model        : {ir.path}")
    print(f"ctx={p.ctx} kv_bpe={p.kv_bpe} vram={args.vram_gib} GiB")
    print(f"fixed        : {_human(p.fixed_bytes)}")
    print(f"kv cache     : {_human(p.kv_bytes)}")
    print(f"workspace    : {_human(p.workspace_bytes)}")
    print(f"expert budget: {_human(p.expert_budget)} / total experts {_human(p.expert_total)}")
    if ir.moe.n_expert:
        print(f"slots/layer  : {p.slots_per_layer}")
        print(f"layers       : {p.layers_on_gpu} on GPU / {p.layers_on_cpu} on CPU"
              f"  (fully_resident={p.fully_resident})")
    else:
        need = p.fixed_bytes + p.kv_bytes + p.workspace_bytes
        print(f"dense model  : no experts; resident need = {_human(need)} "
              f"(vram usable = {_human(p.vram_total - p.vram_reserve)})")
        print(f"fits         : {need <= p.vram_total - p.vram_reserve}")
    print(f"projection   : cpu-lane {p.proj_decode_cpu_ms:.1f} ms/token, "
          f"pcie {p.proj_decode_pcie_ms:.1f} ms/token")
    for n in p.notes:
        print(f"note         : {n}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": ir.path, **p.to_json()}, f, ensure_ascii=False, indent=2)
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
    c.add_argument("--kv-bpe", type=int, default=2); c.add_argument("--vram-gib", type=float, default=12)
    c.add_argument("--json"); c.set_defaults(fn=cmd_plan)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
