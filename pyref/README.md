# pyref — Python reference implementation (for cross-checks only)

`pyref/` is a **small independent Python implementation** of the same IR / GGUF /
safetensors parsing and placement planning that the C++ engine implements. It is
used by the verification tooling (`tools/diff_ir.py`, `tools/diff_plan.py`) as a
second opinion: the C++ engine's parsed IR and placement plan are compared against
this implementation field by field.

- **Not used at runtime** and **not built** by `build.sh`; it is not a dependency of
  `engine.exe` / `launcher.exe`.
- It is intentionally kept minimal/slow; if it and the C++ side disagree, the C++
  side is authoritative and the report is investigated.
- It is useful when porting the engine to another language or when debugging a
  GGUF/safetensors layout issue.
- Keep it in sync when changing the IR/plan **format** (not when changing kernels).

Files:

| file | purpose |
|---|---|
| `gguf_parse.py` | minimal GGUF metadata/tensor parser |
| `st_parse.py` | safetensors header parser |
| `ir.py` | model IR (layers/tensors/attention/MoE/SSM) |
| `planner.py` | VRAM/host placement plan (mirrors `src/planner.cpp`) |
| `cli.py` | `python -m pyref.cli info|ir|plan ...` entry point |
| `__init__.py` | package marker |

Run it from the repository root, e.g.:

```bash
python -m pyref.cli info <MODEL_DIR>/model.gguf
python tools/diff_ir.py <MODEL_DIR>/model.gguf
```
